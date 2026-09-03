from __future__ import annotations

"""Candidate-code-free NumPy reference for selected Qwen3.5 GDN transitions.

This module deliberately imports neither torch nor any qcomem candidate module.
It consumes frozen FP32 arrays captured immediately at the recurrent-rule call
boundary and implements the mathematical recurrence directly with NumPy.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


REFERENCE_SCHEMA = "qcomem-gdn-transition-reference-v1"
FAULTS = (
    "omit_decay",
    "complement_beta",
    "pre_decay_memory",
    "roll_value_heads",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_array(path: Path) -> np.ndarray:
    value = np.load(path, allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype != np.float32:
        raise ValueError(f"{path} must contain one float32 ndarray")
    if not np.isfinite(value).all():
        raise ValueError(f"{path} contains a non-finite value")
    return np.ascontiguousarray(value)


def recurrent_transition(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    g: np.ndarray,
    beta: np.ndarray,
    initial_state: np.ndarray,
    *,
    use_qk_l2norm_in_kernel: bool = False,
    query_scale: float = 1.0,
    fault: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the explicit gated-delta recurrence in IEEE float32.

    Shapes are [B,T,H,K] for query/key, [B,T,H,V] for value,
    [B,T,H] for g/beta, and [B,H,K,V] for the recurrent state.
    """

    if fault is not None and fault not in FAULTS:
        raise ValueError(f"unknown fault {fault!r}")
    arrays = (query, key, value, g, beta, initial_state)
    if any(item.dtype != np.float32 for item in arrays):
        raise ValueError("all reference inputs must be float32")
    if query.ndim != 4 or key.shape != query.shape or value.ndim != 4:
        raise ValueError("invalid query/key/value ranks or shapes")
    batch, tokens, heads, key_width = query.shape
    if value.shape[:3] != (batch, tokens, heads):
        raise ValueError("value prefix shape differs from query")
    value_width = value.shape[-1]
    if g.shape != (batch, tokens, heads) or beta.shape != g.shape:
        raise ValueError("g/beta shapes differ from query prefix")
    if initial_state.shape != (batch, heads, key_width, value_width):
        raise ValueError("initial recurrent-state shape differs from inputs")

    q = np.ascontiguousarray(query)
    k = np.ascontiguousarray(key)
    if use_qk_l2norm_in_kernel:
        q = np.ascontiguousarray(
            q * np.reciprocal(
                np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + np.float32(1e-6))
            )
        )
        k = np.ascontiguousarray(
            k * np.reciprocal(
                np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + np.float32(1e-6))
            )
        )
    if not np.isfinite(query_scale):
        raise ValueError("query_scale must be finite")
    q = np.ascontiguousarray(q * np.float32(query_scale))
    v = np.ascontiguousarray(value)
    if fault == "roll_value_heads":
        v = np.roll(v, shift=1, axis=2).copy()
    recurrent = np.ascontiguousarray(initial_state.copy())
    outputs: list[np.ndarray] = []
    for position in range(tokens):
        q_t = q[:, position]
        k_t = k[:, position]
        v_t = v[:, position]
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
        recurrent = np.ascontiguousarray(
            recurrent + k_t[..., None] * delta[..., None, :]
        )
        outputs.append(
            np.einsum("bhkv,bhk->bhv", recurrent, q_t, optimize=False)
        )
    output = np.stack(outputs, axis=1).astype(np.float32, copy=False)
    return np.ascontiguousarray(output), recurrent


def metrics(candidate: np.ndarray, reference: np.ndarray) -> dict[str, float | bool]:
    if candidate.shape != reference.shape:
        raise ValueError(f"shape mismatch: {candidate.shape} != {reference.shape}")
    difference = candidate.astype(np.float64) - reference.astype(np.float64)
    reference64 = reference.astype(np.float64)
    l2_denominator = max(float(np.linalg.norm(reference64.ravel())), 1e-30)
    max_denominator = max(float(np.max(np.abs(reference64))), 1e-30)
    return {
        "finite": bool(np.isfinite(candidate).all() and np.isfinite(reference).all()),
        "relative_l2": float(np.linalg.norm(difference.ravel()) / l2_denominator),
        "max_abs": float(np.max(np.abs(difference))),
        "normalized_max_abs": float(np.max(np.abs(difference)) / max_denominator),
    }


def coordinate_max_abs(
    candidate: np.ndarray,
    reference: np.ndarray,
    coordinates: list[list[int]],
) -> float:
    values = []
    for raw in coordinates:
        coordinate = tuple(int(item) for item in raw)
        values.append(abs(float(candidate[coordinate]) - float(reference[coordinate])))
    return max(values, default=0.0)


def evaluate_capture(
    capture_manifest_path: Path,
    preregistration_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    capture_raw = capture_manifest_path.read_bytes()
    prereg_raw = preregistration_path.read_bytes()
    capture = json.loads(capture_raw)
    prereg = json.loads(prereg_raw)
    if capture.get("schema_version") != "qcomem-gdn-transition-capture-v1":
        raise ValueError("capture schema drift")
    if prereg.get("schema_version") != "qcomem-gdn-transition-preregistration-v1":
        raise ValueError("preregistration schema drift")
    if capture.get("preregistration_raw_sha256") != hashlib.sha256(prereg_raw).hexdigest():
        raise ValueError("capture/preregistration binding drift")
    root = capture_manifest_path.parent
    rows = []
    faults = []
    tolerances = prereg["tolerances"]
    selected = prereg["selected_rows"]
    if [row["row_id"] for row in capture["rows"]] != [row["row_id"] for row in selected]:
        raise ValueError("capture row order differs from preregistration")
    for capture_row, selected_row in zip(capture["rows"], selected):
        semantics = capture_row["kernel_semantics"]
        if semantics.get("native_use_qk_l2norm_in_kernel") is not True:
            raise ValueError("native q/k normalization receipt drift")
        if semantics.get("reference_inputs_post_native_qk_l2norm") is not True:
            raise ValueError("reference q/k boundary receipt drift")
        if semantics.get("query_key_capture_boundary") != "post_native_qk_l2norm_pre_fp32_recurrence":
            raise ValueError("reference q/k capture boundary drift")
        if semantics.get("use_qk_l2norm_in_kernel") is not False:
            raise ValueError("reference would normalize captured q/k twice")
        if semantics.get("query_scale_source") != "native_default_inverse_sqrt_key_width":
            raise ValueError("native query scale source drift")
        if int(semantics.get("query_key_width", 0)) != 128:
            raise ValueError("native query key width drift")
        arrays = {}
        for name, receipt in capture_row["arrays"].items():
            path = root / receipt["relative_path"]
            if sha256_file(path) != receipt["sha256"]:
                raise ValueError(f"{name} sidecar SHA drift")
            arrays[name] = load_array(path)
            if list(arrays[name].shape) != receipt["shape"]:
                raise ValueError(f"{name} sidecar shape drift")
        reference_output, reference_state = recurrent_transition(
            arrays["query"],
            arrays["key"],
            arrays["value"],
            arrays["g"],
            arrays["beta"],
            arrays["initial_state"],
            use_qk_l2norm_in_kernel=bool(
                capture_row["kernel_semantics"]["use_qk_l2norm_in_kernel"]
            ),
            query_scale=float(capture_row["kernel_semantics"]["query_scale"]),
        )
        output_metrics = metrics(arrays["candidate_output"], reference_output)
        state_metrics = metrics(arrays["candidate_state"], reference_state)
        output_coordinate_error = coordinate_max_abs(
            arrays["candidate_output"], reference_output, selected_row["output_coordinates"]
        )
        state_coordinate_error = coordinate_max_abs(
            arrays["candidate_state"], reference_state, selected_row["state_coordinates"]
        )
        clean_pass = bool(
            output_metrics["finite"]
            and state_metrics["finite"]
            and output_metrics["relative_l2"] <= tolerances["output_relative_l2"]
            and output_metrics["normalized_max_abs"] <= tolerances["output_normalized_max_abs"]
            and state_metrics["relative_l2"] <= tolerances["state_relative_l2"]
            and state_metrics["normalized_max_abs"] <= tolerances["state_normalized_max_abs"]
            and output_coordinate_error <= tolerances["output_coordinate_max_abs"]
            and state_coordinate_error <= tolerances["state_coordinate_max_abs"]
        )
        fault = selected_row["fault"]
        wrong_output, wrong_state = recurrent_transition(
            arrays["query"],
            arrays["key"],
            arrays["value"],
            arrays["g"],
            arrays["beta"],
            arrays["initial_state"],
            use_qk_l2norm_in_kernel=bool(
                capture_row["kernel_semantics"]["use_qk_l2norm_in_kernel"]
            ),
            query_scale=float(capture_row["kernel_semantics"]["query_scale"]),
            fault=fault,
        )
        wrong_output_metrics = metrics(arrays["candidate_output"], wrong_output)
        wrong_state_metrics = metrics(arrays["candidate_state"], wrong_state)
        wrong_transition_rejected = bool(
            wrong_output_metrics["relative_l2"] > tolerances["output_relative_l2"]
            or wrong_output_metrics["normalized_max_abs"] > tolerances["output_normalized_max_abs"]
            or wrong_state_metrics["relative_l2"] > tolerances["state_relative_l2"]
            or wrong_state_metrics["normalized_max_abs"] > tolerances["state_normalized_max_abs"]
        )
        rows.append(
            {
                "row_id": selected_row["row_id"],
                "layer_index": selected_row["layer_index"],
                "clean_pass": clean_pass,
                "output_metrics": output_metrics,
                "state_metrics": state_metrics,
                "output_coordinate_max_abs": output_coordinate_error,
                "state_coordinate_max_abs": state_coordinate_error,
            }
        )
        faults.append(
            {
                "row_id": selected_row["row_id"],
                "fault": fault,
                "rejected": wrong_transition_rejected,
                "output_metrics": wrong_output_metrics,
                "state_metrics": wrong_state_metrics,
            }
        )
    result = {
        "schema_version": REFERENCE_SCHEMA,
        "capture_manifest_raw_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "preregistration_raw_sha256": hashlib.sha256(prereg_raw).hexdigest(),
        "reference_implementation": "numpy-float32-explicit-token-loop",
        "candidate_code_imported": False,
        "rows": rows,
        "faults": faults,
        "all_clean_rows_pass": all(row["clean_pass"] for row in rows),
        "all_seeded_wrong_transitions_rejected": all(row["rejected"] for row in faults),
        "claim_boundary": (
            "selected recurrent-core transitions from frozen post-native-qk-normalization "
            "inputs only; excludes q/k normalization, projections, causal convolution, "
            "gated RMS normalization, output projection, and end-to-end logits"
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result
