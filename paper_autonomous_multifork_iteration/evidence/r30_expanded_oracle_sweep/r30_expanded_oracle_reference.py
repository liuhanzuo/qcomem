from __future__ import annotations

"""Candidate-import-free CPU/NumPy FP32 replay for the R30 sweep."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_SCHEMA = "forkaudit-r30-expanded-oracle-reference-v1"
CAPTURE_SCHEMA = "forkaudit-r30-expanded-oracle-capture-v1"
PREREG_SCHEMA = "forkaudit-r30-expanded-oracle-preregistration-v1"
ATTENTION_FAULTS = ("unit_scale", "roll_kv_heads", "drop_self_key", "reverse_kv_tokens")
GDN_FAULTS = ("omit_decay", "complement_beta", "pre_decay_memory", "roll_value_heads")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_array(root: Path, receipt: dict[str, Any]) -> np.ndarray:
    path = root / receipt["relative_path"]
    if sha256_file(path) != receipt["sha256"]:
        raise ValueError(f"sidecar SHA drift: {path}")
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise ValueError(f"{path} is not one float32 ndarray")
    if list(value.shape) != receipt["shape"] or not np.isfinite(value).all():
        raise ValueError(f"{path} shape/finite gate failed")
    return np.ascontiguousarray(value)


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} != {reference.shape}")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64)
    return {
        "finite": bool(np.isfinite(candidate).all() and np.isfinite(reference).all()),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / max(float(np.linalg.norm(reference64.ravel())), 1e-30)),
        "max_abs": float(np.max(np.abs(difference))),
        "normalized_max_abs": float(np.max(np.abs(difference)) / max(float(np.max(np.abs(reference64))), 1e-30)),
    }


def dense_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    query_positions: list[int],
    key_positions: list[int],
    scale: float,
    *,
    fault: str | None = None,
) -> np.ndarray:
    if fault is not None and fault not in ATTENTION_FAULTS:
        raise ValueError(f"unknown attention fault {fault}")
    if query.ndim != 4 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("attention tensor rank/shape drift")
    batch, query_heads, query_tokens, head_dim = query.shape
    if key.shape[0] != batch or key.shape[-1] != head_dim:
        raise ValueError("attention K/V geometry drift")
    kv_heads, key_tokens = int(key.shape[1]), int(key.shape[2])
    if query_heads % kv_heads:
        raise ValueError("query heads are not divisible by KV heads")
    if len(query_positions) != query_tokens or len(key_positions) != key_tokens:
        raise ValueError("attention position geometry drift")
    k = key
    v = value
    effective_scale = np.float32(scale)
    if fault == "unit_scale":
        effective_scale = np.float32(1.0)
    elif fault == "roll_kv_heads":
        k = np.roll(k, shift=1, axis=1).copy()
        v = np.roll(v, shift=1, axis=1).copy()
    elif fault == "reverse_kv_tokens":
        k = k[:, :, ::-1, :].copy()
        v = v[:, :, ::-1, :].copy()
    groups = query_heads // kv_heads
    k = np.repeat(k, groups, axis=1)
    v = np.repeat(v, groups, axis=1)
    scores = np.einsum("bhqd,bhkd->bhqk", query, k, optimize=False).astype(np.float32, copy=False)
    scores = np.ascontiguousarray(scores * effective_scale)
    qpos = np.asarray(query_positions, dtype=np.int64)
    kpos = np.asarray(key_positions, dtype=np.int64)
    visible = kpos[None, :] <= qpos[:, None]
    if fault == "drop_self_key":
        visible = kpos[None, :] < qpos[:, None]
    scores = np.where(visible[None, None, :, :], scores, np.float32(-np.inf))
    maximum = np.max(scores, axis=-1, keepdims=True)
    weights = np.exp(scores - maximum, dtype=np.float32)
    weights = np.ascontiguousarray(weights / np.sum(weights, axis=-1, keepdims=True, dtype=np.float32))
    output = np.einsum("bhqk,bhkd->bhqd", weights, v, optimize=False).astype(np.float32, copy=False)
    return np.ascontiguousarray(output.transpose(0, 2, 1, 3))


def recurrent_transition(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
    scale: float,
    *,
    fault: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if fault is not None and fault not in GDN_FAULTS:
        raise ValueError(f"unknown GDN fault {fault}")
    q = np.ascontiguousarray(query * np.float32(scale))
    k = np.ascontiguousarray(key)
    v = np.ascontiguousarray(value)
    if fault == "roll_value_heads":
        v = np.roll(v, shift=1, axis=2).copy()
    recurrent = np.ascontiguousarray(initial_state.copy())
    outputs: list[np.ndarray] = []
    for position in range(int(query.shape[1])):
        q_t, k_t, v_t = q[:, position], k[:, position], v[:, position]
        beta_t = beta[:, position]
        if fault == "complement_beta":
            beta_t = np.float32(1.0) - beta_t
        previous = recurrent.copy() if fault == "pre_decay_memory" else None
        decay = np.exp(g[:, position], dtype=np.float32)[..., None, None]
        if fault == "omit_decay":
            decay = np.ones_like(decay)
        recurrent = np.ascontiguousarray(recurrent * decay)
        memory_source = previous if previous is not None else recurrent
        memory = np.einsum("bhkv,bhk->bhv", memory_source, k_t, optimize=False)
        delta = np.ascontiguousarray((v_t - memory) * beta_t[..., None])
        recurrent = np.ascontiguousarray(recurrent + k_t[..., None] * delta[..., None, :])
        outputs.append(np.einsum("bhkv,bhk->bhv", recurrent, q_t, optimize=False))
    return np.ascontiguousarray(np.stack(outputs, axis=1).astype(np.float32, copy=False)), recurrent


def _passes(metric: dict[str, float | bool], tolerance: dict[str, float]) -> bool:
    return bool(
        metric["finite"]
        and metric["relative_l2"] <= tolerance["relative_l2"]
        and (
            "normalized_max_abs" not in tolerance
            or metric["normalized_max_abs"] <= tolerance["normalized_max_abs"]
        )
    )


def evaluate(capture_path: Path, prereg_path: Path, output_path: Path) -> dict[str, Any]:
    capture_raw = capture_path.read_bytes()
    prereg_raw = prereg_path.read_bytes()
    capture = json.loads(capture_raw)
    prereg = json.loads(prereg_raw)
    if capture.get("schema_version") != CAPTURE_SCHEMA or prereg.get("schema_version") != PREREG_SCHEMA:
        raise ValueError("capture/preregistration schema drift")
    if capture.get("preregistration_raw_sha256") != hashlib.sha256(prereg_raw).hexdigest():
        raise ValueError("capture/preregistration binding drift")
    if prereg["code_bindings"]["reference_raw_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("preregistration/reference source binding drift")
    root = capture_path.parent
    attn_tolerance = prereg["tolerances"]["attention"]
    gdn_output_tolerance = prereg["tolerances"]["gdn_output"]
    gdn_state_tolerance = prereg["tolerances"]["gdn_state"]
    attention_results: list[dict[str, Any]] = []
    gdn_results: list[dict[str, Any]] = []
    attention_fault_results: list[dict[str, Any]] = []
    gdn_fault_results: list[dict[str, Any]] = []
    attention_fault_plan = prereg["fault_assignment"]["attention"]
    gdn_fault_plan = prereg["fault_assignment"]["gdn"]
    if [row["row_id"] for row in capture["attention_rows"]] != [row["row_id"] for row in attention_fault_plan]:
        raise ValueError("captured attention row order differs from frozen fault plan")
    if [row["row_id"] for row in capture["gdn_rows"]] != [row["row_id"] for row in gdn_fault_plan]:
        raise ValueError("captured GDN row order differs from frozen fault plan")
    for row, fault_plan in zip(capture["attention_rows"], attention_fault_plan):
        arrays = {name: load_array(root, receipt) for name, receipt in row["arrays"].items()}
        reference = dense_attention(
            arrays["query"], arrays["key"], arrays["value"],
            row["query_positions"], row["key_positions"], float(row["softmax_scale"]),
        )
        clean = metrics(arrays["candidate_output"], reference)
        fault = str(fault_plan["fault"])
        if fault not in ATTENTION_FAULTS:
            raise ValueError("frozen attention fault is unsupported")
        wrong = dense_attention(
            arrays["query"], arrays["key"], arrays["value"],
            row["query_positions"], row["key_positions"], float(row["softmax_scale"]), fault=fault,
        )
        wrong_metric = metrics(arrays["candidate_output"], wrong)
        attention_results.append({"row_id": row["row_id"], "clean_pass": _passes(clean, attn_tolerance), "metrics": clean})
        attention_fault_results.append({"row_id": row["row_id"], "fault": fault, "rejected": not _passes(wrong_metric, attn_tolerance), "metrics": wrong_metric})
    for row, fault_plan in zip(capture["gdn_rows"], gdn_fault_plan):
        arrays = {name: load_array(root, receipt) for name, receipt in row["arrays"].items()}
        scale = float(row["kernel_semantics"]["query_scale"])
        reference_output, reference_state = recurrent_transition(
            arrays["query"], arrays["key"], arrays["value"], arrays["g"], arrays["beta"], arrays["initial_state"], scale,
        )
        output_metric = metrics(arrays["candidate_output"], reference_output)
        state_metric = metrics(arrays["candidate_state"], reference_state)
        clean_pass = _passes(output_metric, gdn_output_tolerance) and _passes(state_metric, gdn_state_tolerance)
        fault = str(fault_plan["fault"])
        if fault not in GDN_FAULTS:
            raise ValueError("frozen GDN fault is unsupported")
        wrong_output, wrong_state = recurrent_transition(
            arrays["query"], arrays["key"], arrays["value"], arrays["g"], arrays["beta"], arrays["initial_state"], scale, fault=fault,
        )
        wrong_output_metric = metrics(arrays["candidate_output"], wrong_output)
        wrong_state_metric = metrics(arrays["candidate_state"], wrong_state)
        wrong_rejected = not (
            _passes(wrong_output_metric, gdn_output_tolerance)
            and _passes(wrong_state_metric, gdn_state_tolerance)
        )
        gdn_results.append({"row_id": row["row_id"], "clean_pass": clean_pass, "output_metrics": output_metric, "state_metrics": state_metric})
        gdn_fault_results.append({"row_id": row["row_id"], "fault": fault, "rejected": wrong_rejected, "output_metrics": wrong_output_metric, "state_metrics": wrong_state_metric})
    result = {
        "schema_version": REFERENCE_SCHEMA,
        "reference_implementation": "candidate-import-free-cpu-numpy-fp32",
        "candidate_code_imported": False,
        "capture_manifest_raw_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "preregistration_raw_sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "reference_source_raw_sha256": sha256_file(Path(__file__)),
        "coverage": capture["coverage"],
        "attention_rows": attention_results,
        "gdn_rows": gdn_results,
        "attention_faults": attention_fault_results,
        "gdn_faults": gdn_fault_results,
        "all_attention_clean_rows_pass": all(row["clean_pass"] for row in attention_results),
        "all_gdn_clean_rows_pass": all(row["clean_pass"] for row in gdn_results),
        "all_attention_faults_rejected": all(row["rejected"] for row in attention_fault_results),
        "all_gdn_faults_rejected": all(row["rejected"] for row in gdn_fault_results),
        "claim_boundary": capture["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if not all(
        result[key]
        for key in (
            "all_attention_clean_rows_pass", "all_gdn_clean_rows_pass",
            "all_attention_faults_rejected", "all_gdn_faults_rejected",
        )
    ):
        raise RuntimeError("one or more pre-registered clean/fault gates failed")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.capture_manifest, args.preregistration, args.output), sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
