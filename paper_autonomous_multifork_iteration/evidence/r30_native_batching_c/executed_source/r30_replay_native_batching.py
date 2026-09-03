from __future__ import annotations

"""Independent replay for the corrected R30C native-vLLM receipts."""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


TRACE_SCHEMA = "forkaudit-r30-native-vllm-scheduler-trace-v1"
INPUT_SCHEMA = "forkaudit-r30c-native-vllm-input-manifest-v2"
OUTPUT_SCHEMA = "forkaudit-r30c-native-vllm-output-comparison-v2"
REPLAY_SCHEMA = "forkaudit-r30c-native-vllm-replay-v2"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows = []
    phase = None
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        require(row.get("schema_version") == TRACE_SCHEMA, f"trace schema line {line_number}")
        if row.get("kind") == "phase_marker":
            phase = row.get("phase")
        row["replayed_phase"] = phase
        rows.append(row)
    require(rows, "empty scheduler trace")
    return rows


def _role_map(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, str]:
    sha_to_role = {
        cell["prompt_token_ids_sha256"]: cell["role"] for cell in manifest["requests"]
    }
    mapping: dict[str, str] = {}
    for row in rows:
        request = row.get("request") or {}
        digest = request.get("prompt_token_ids_sha256")
        request_id = request.get("request_id")
        if digest in sha_to_role and request_id is not None:
            mapping[str(request_id)] = sha_to_role[digest]
        for new in row.get("scheduled_new_requests", []):
            digest = new.get("prompt_token_ids_sha256")
            request_id = new.get("request_id")
            if digest in sha_to_role and request_id is not None:
                mapping[str(request_id)] = sha_to_role[digest]
    return mapping


def _group_overlap(states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overlaps = []
    live = [row for row in states if any(row.get("block_ids_by_group", []))]
    for left, right in combinations(live, 2):
        groups = max(
            len(left.get("block_ids_by_group", [])),
            len(right.get("block_ids_by_group", [])),
        )
        for group in range(groups):
            left_ids = set(
                left.get("block_ids_by_group", [])[group]
                if group < len(left.get("block_ids_by_group", []))
                else []
            )
            right_ids = set(
                right.get("block_ids_by_group", [])[group]
                if group < len(right.get("block_ids_by_group", []))
                else []
            )
            shared = sorted(left_ids & right_ids)
            if shared:
                overlaps.append(
                    {
                        "left": left["request_id"],
                        "right": right["request_id"],
                        "group": group,
                        "shared_block_ids": shared,
                    }
                )
    return overlaps


def _load_output_sidecar(
    root: Path,
    receipt: dict[str, Any],
    *,
    role: str,
    model_vocab_size: int,
) -> np.ndarray:
    require(int(receipt["model_vocab_size"]) == model_vocab_size, f"{role} model vocab receipt drift")
    require(bool(receipt["mapping_keys_exact_model_vocab_range"]), f"{role} key-range receipt absent")
    generated = [int(value) for value in receipt["generated_token_ids"]]
    require(
        receipt["mapping_key_count_per_step"] == [model_vocab_size] * len(generated),
        f"{role} mapping-key counts drift",
    )
    expected_shape = [len(generated), model_vocab_size]
    require(receipt["full_vocab_logprobs_shape"] == expected_shape, f"{role} declared shape drift")
    require(receipt["full_vocab_logprobs_dtype"] == "float32", f"{role} declared dtype drift")
    path = (root / receipt["full_vocab_logprobs_path"]).resolve()
    require(path.is_relative_to(root.resolve()), f"{role} sidecar escapes run root")
    require(path.is_file(), f"{role} sidecar missing")
    require(sha256_file(path) == receipt["full_vocab_logprobs_sha256"], f"{role} sidecar SHA drift")
    values = np.load(path, allow_pickle=False)
    require(values.dtype == np.float32, f"{role} sidecar dtype drift")
    require(list(values.shape) == expected_shape, f"{role} sidecar shape drift")
    return values


def _recompute_output_comparisons(
    root: Path,
    manifest: dict[str, Any],
    outputs: dict[str, Any],
) -> dict[str, Any]:
    model_vocab_size = int(manifest["model_vocab_size"])
    tolerance = manifest["full_vocab_logprob_tolerance"]
    comparisons = {}
    for role in ("A", "B", "C"):
        batch_receipt = outputs["rows"]["native_batch"][role]
        control_receipt = outputs["rows"]["sequential"][role]
        batch = _load_output_sidecar(
            root, batch_receipt, role=f"native_batch/{role}", model_vocab_size=model_vocab_size
        )
        control = _load_output_sidecar(
            root, control_receipt, role=f"sequential/{role}", model_vocab_size=model_vocab_size
        )
        require(batch.shape == control.shape, f"{role} native/control shape drift")
        finite = np.isfinite(batch) & np.isfinite(control)
        same_nonfinite = bool(np.array_equal(np.isfinite(batch), np.isfinite(control)))
        absolute = np.abs(batch[finite].astype(np.float64) - control[finite].astype(np.float64))
        comparisons[role] = {
            "token_ids_exact": (
                batch_receipt["generated_token_ids"] == control_receipt["generated_token_ids"]
            ),
            "batch_token_ids": batch_receipt["generated_token_ids"],
            "sequential_token_ids": control_receipt["generated_token_ids"],
            "same_nonfinite_pattern": same_nonfinite,
            "full_vocab_logprobs_max_abs_error": float(absolute.max()) if absolute.size else 0.0,
            "full_vocab_logprobs_mean_abs_error": float(absolute.mean()) if absolute.size else 0.0,
            "full_vocab_logprobs_atol": float(tolerance["atol"]),
            "full_vocab_logprobs_rtol": float(tolerance["rtol"]),
            "full_vocab_logprobs_within_preregistered_tolerance": bool(
                same_nonfinite
                and np.allclose(
                    batch,
                    control,
                    atol=float(tolerance["atol"]),
                    rtol=float(tolerance["rtol"]),
                    equal_nan=True,
                )
            ),
        }
    return comparisons


def analyze(root: Path) -> dict[str, Any]:
    manifest_path = root / "static" / "input_manifest.json"
    trace_path = root / "raw" / "scheduler_trace.jsonl"
    outputs_path = root / "raw" / "outputs.json"
    manifest = load_json(manifest_path)
    outputs = load_json(outputs_path)
    require(manifest.get("schema_version") == INPUT_SCHEMA, "input manifest schema drift")
    require(outputs.get("schema_version") == OUTPUT_SCHEMA, "outputs schema drift")
    require(int(manifest["tokenizer_vocab_size"]) == 248077, "fixed tokenizer vocabulary drift")
    require(int(manifest["model_vocab_size"]) == 248320, "fixed model vocabulary drift")
    require(int(manifest["padded_model_vocab_slots"]) == 243, "fixed padded-slot count drift")
    require(manifest["expected_model_token_id_range"] == [0, 248319], "model token range drift")
    require(int(outputs["tokenizer_vocab_size"]) == 248077, "outputs tokenizer vocabulary drift")
    require(int(outputs["model_vocab_size"]) == 248320, "outputs model vocabulary drift")
    require(outputs["expected_model_token_id_range"] == [0, 248319], "outputs token range drift")
    rows = load_trace(trace_path)
    request_roles = _role_map(manifest, rows)

    batch_rows = [row for row in rows if row.get("replayed_phase") == "native_batch"]
    schedule_rows = [row for row in batch_rows if row.get("kind") == "schedule"]
    require(schedule_rows, "native batch has no scheduler rows")

    batch_request_ids = {
        request.get("request_id")
        for row in batch_rows
        for request in (
            ([row.get("request", {})] if row.get("request") else [])
            + row.get("scheduled_new_requests", [])
        )
        if request.get("request_id") is not None
    }
    role_to_id = {
        role: req_id
        for req_id, role in request_roles.items()
        if req_id in batch_request_ids
    }
    require(set(role_to_id) >= {"A", "B", "C"}, "native request role mapping incomplete")
    a_id, b_id, c_id = role_to_id["A"], role_to_id["B"], role_to_id["C"]

    initial_candidates = [
        row
        for row in schedule_rows
        if {item["request_id"] for item in row["scheduled_new_requests"]} == {a_id, b_id}
    ]
    require(initial_candidates, "no native two-request ragged admission step")
    initial = initial_candidates[0]
    expected_lengths = {
        cell["role"]: int(cell["prompt_tokens"]) for cell in manifest["requests"]
    }
    initial_lengths = {
        request_roles[item["request_id"]]: int(item["prompt_tokens"])
        for item in initial["scheduled_new_requests"]
    }
    require(initial_lengths == {"A": expected_lengths["A"], "B": expected_lengths["B"]}, "ragged prompt lengths drift")
    require(len(set(initial_lengths.values())) == 2, "initial admission is not ragged")
    require(set(initial["num_scheduled_tokens"]) == {a_id, b_id}, "initial scheduler batch membership drift")

    decode_overlap = [
        row
        for row in schedule_rows
        if {a_id, b_id}.issubset(set(row["scheduled_cached_request_ids"]))
    ]
    require(decode_overlap, "A/B cached decode overlap absent")

    turnover = [
        row
        for row in schedule_rows
        if c_id in {item["request_id"] for item in row["scheduled_new_requests"]}
        and b_id in set(row["scheduled_cached_request_ids"])
    ]
    require(turnover, "continuous-batching B/C turnover absent")

    free_begin = [row for row in batch_rows if row.get("kind") == "free_request_begin"]
    free_order = [request_roles.get(row["request"]["request_id"], "unknown") for row in free_begin]
    require(set(free_order) >= {"A", "B", "C"}, "terminal free receipts incomplete")
    require(free_order.index("A") < free_order.index("B"), "short A did not finish before B")

    overlaps = []
    for row in schedule_rows:
        overlaps.extend(
            {"event_index": row["event_index"], **item}
            for item in _group_overlap(row.get("active_requests", []))
        )
    require(not overlaps, "simultaneously live request block ownership overlaps")

    zeroing_rows = []
    zeroing_complete = True
    for row in schedule_rows:
        allocated = {
            block_id
            for request in row.get("scheduled_new_requests", [])
            for group in request.get("block_ids_by_group", [])
            for block_id in group
        }
        if allocated:
            zeroed = set(row.get("new_block_ids_to_zero", []))
            covered = allocated.issubset(zeroed)
            zeroing_complete &= covered
            zeroing_rows.append(
                {
                    "event_index": row["event_index"],
                    "allocated_block_count": len(allocated),
                    "zeroed_block_count": len(zeroed),
                    "allocation_covered_by_zeroing": covered,
                }
            )
    require(zeroing_rows and zeroing_complete, "fresh native allocations were not all zeroed")

    control_max_members = {}
    for role in ("A", "B", "C"):
        phase = f"sequential_{role.lower()}"
        sizes = [
            len(row.get("num_scheduled_tokens", {}))
            for row in rows
            if row.get("replayed_phase") == phase and row.get("kind") == "schedule"
        ]
        require(sizes, f"missing {phase} scheduler rows")
        control_max_members[role] = max(sizes)
        require(max(sizes) == 1, f"{phase} was not a one-request control")

    comparisons = _recompute_output_comparisons(root, manifest, outputs)
    require(comparisons == outputs["comparisons"], "serialized output comparisons drift")
    for role in ("A", "B", "C"):
        row = comparisons[role]
        require(row["token_ids_exact"], f"{role} output tokens differ")
        require(row["full_vocab_logprobs_within_preregistered_tolerance"], f"{role} log-probs exceed tolerance")

    a_freed_blocks = {
        block_id
        for row in free_begin
        if request_roles.get(row["request"]["request_id"]) == "A"
        for group in row["request"].get("block_ids_by_group", [])
        for block_id in group
    }
    c_new_blocks = {
        block_id
        for row in turnover
        for request in row["scheduled_new_requests"]
        if request["request_id"] == c_id
        for group in request.get("block_ids_by_group", [])
        for block_id in group
    }

    return {
        "schema_version": REPLAY_SCHEMA,
        "status": "passed",
        "trace_sha256": sha256_file(trace_path),
        "input_manifest_sha256": sha256_file(manifest_path),
        "outputs_sha256": sha256_file(outputs_path),
        "vocabulary_contract": {
            "tokenizer_vocab_size": int(manifest["tokenizer_vocab_size"]),
            "model_vocab_size": int(manifest["model_vocab_size"]),
            "padded_model_vocab_slots": int(manifest["padded_model_vocab_slots"]),
            "exact_model_token_id_range": manifest["expected_model_token_id_range"],
            "dense_sidecars_independently_replayed": True,
        },
        "native_engine_evidence": {
            "initial_ragged_batch_roles": ["A", "B"],
            "initial_prompt_lengths": initial_lengths,
            "initial_scheduler_event_index": initial["event_index"],
            "cached_decode_overlap_event_indices": [row["event_index"] for row in decode_overlap],
            "continuous_turnover_event_indices": [row["event_index"] for row in turnover],
            "free_order": free_order,
            "simultaneously_live_block_overlap_count": len(overlaps),
            "all_new_allocations_zeroed_before_execution": zeroing_complete,
            "zeroing_rows": zeroing_rows,
            "short_A_block_ids_reused_by_C_count": len(a_freed_blocks & c_new_blocks),
            "sequential_control_max_scheduler_members": control_max_members,
        },
        "output_comparisons": comparisons,
        "claim_boundary": {
            "established": [
                "vLLM 0.26 V1 Scheduler admitted different-length A/B requests in one native SchedulerOutput",
                "A/B shared cached decode steps and B remained live when waiting C was admitted",
                "scheduler-observed live KV block ownership was pairwise disjoint by cache group",
                "native-batch greedy tokens and all 248,320 model-vocabulary output log-probabilities matched sequential controls within the preregistered tolerance",
            ],
            "not_established": [
                "the in-process CoMem/ForkAudit cache facade is integrated into vLLM EngineCore",
                "ForkAudit GDN ownership receipts under native batching",
                "independent observer capture of GPU memory contents",
                "native-batching throughput, capacity, cancellation, or production safety",
                "cross-runtime or cross-model generality",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = analyze(args.root)
    output = args.output or args.root / "replay_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
