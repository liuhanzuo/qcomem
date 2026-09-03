from __future__ import annotations

"""Independent replay for the Round-29 concurrent-lifecycle result."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from qcomem_forkaudit_lifecycle_transfer import replay_slot_events
from run_downstream import atomic_json


RESULT_SCHEMA = "qcomem-forkaudit-true-concurrent-lifecycle-result-v1"
REPLAY_SCHEMA = "qcomem-forkaudit-true-concurrent-lifecycle-replay-v1"
DESIGN_SCHEMA = "qcomem-forkaudit-true-concurrent-lifecycle-design-v1"
FULL_LAYERS = tuple(range(3, 40, 4))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_bound_json(path: Path, expected_sha256: str, label: str) -> Any:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"{label} raw SHA mismatch")
    return json.loads(raw)


def interval_overlap_ms(intervals: Sequence[Mapping[str, float]]) -> float:
    require(len(intervals) == 2, "replay expects exactly two CUDA intervals")
    require(
        all(float(row["end_ms"]) > float(row["start_ms"]) for row in intervals),
        "non-positive CUDA interval duration",
    )
    return min(float(row["end_ms"]) for row in intervals) - max(
        float(row["start_ms"]) for row in intervals
    )


def read_sidecar(
    artifact_dir: Path,
    manifest: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    require(
        manifest.get("schema_version") == "qcomem-forkaudit-r29-logit-sidecar-v1",
        "sidecar schema drift",
    )
    relative = manifest.get("path")
    require(
        isinstance(relative, str)
        and relative
        and Path(relative).name == relative,
        "sidecar path is not one basename",
    )
    raw = (artifact_dir / relative).read_bytes()
    require(len(raw) == manifest.get("bytes"), "sidecar byte count drift")
    require(sha256_bytes(raw) == manifest.get("sha256"), "sidecar SHA drift")
    records = manifest.get("records")
    require(
        isinstance(records, list)
        and len(records) == manifest.get("record_count") == 4,
        "sidecar record count drift",
    )
    result: dict[str, np.ndarray] = {}
    cursor = 0
    for row in records:
        sample_id = row.get("sample_id")
        shape = row.get("shape")
        offset = row.get("offset_bytes")
        nbytes = row.get("nbytes")
        require(
            isinstance(sample_id, str)
            and sample_id
            and sample_id not in result,
            "sidecar sample ID drift",
        )
        require(
            row.get("dtype") == "float32-le"
            and isinstance(shape, list)
            and len(shape) == 2
            and shape[0] == 1
            and all(type(item) is int and item > 0 for item in shape),
            "sidecar dtype/shape drift",
        )
        expected_nbytes = int(np.prod(shape)) * 4
        require(
            type(offset) is int
            and type(nbytes) is int
            and offset == cursor
            and nbytes == expected_nbytes,
            "sidecar record byte geometry drift",
        )
        payload = raw[offset : offset + nbytes]
        require(len(payload) == nbytes, "sidecar record truncated")
        require(
            sha256_bytes(payload) == row.get("content_sha256"),
            "sidecar record SHA drift",
        )
        array = np.frombuffer(payload, dtype="<f4").reshape(shape).copy()
        require(bool(np.isfinite(array).all()), "sidecar contains non-finite logits")
        require(int(array.argmax(axis=-1)[0]) == row.get("token_id"), "token replay drift")
        result[sample_id] = array
        cursor += nbytes
    require(
        cursor == len(raw)
        and manifest.get("terminal_exact_byte_coverage") is True,
        "sidecar terminal byte coverage drift",
    )
    return result


def compare_logits(
    serialized: Mapping[str, np.ndarray],
    concurrent: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    require(set(serialized) == set(concurrent), "cross-arm sidecar sample IDs differ")
    rows = []
    for sample_id in sorted(serialized):
        left = serialized[sample_id]
        right = concurrent[sample_id]
        require(left.shape == right.shape, "cross-arm logit shape drift")
        difference = np.abs(left - right)
        rows.append(
            {
                "sample_id": sample_id,
                "torch_equal": bool(np.array_equal(left, right)),
                "max_abs_error": float(difference.max()),
                "mean_abs_error": float(difference.mean()),
                "serialized_token_id": int(left.argmax(axis=-1)[0]),
                "concurrent_token_id": int(right.argmax(axis=-1)[0]),
                "token_equal": bool(
                    left.argmax(axis=-1)[0] == right.argmax(axis=-1)[0]
                ),
            }
        )
    return {
        "sample_count": len(rows),
        "rows": rows,
        "all_full_vocab_logits_torch_equal": all(row["torch_equal"] for row in rows),
        "all_generated_tokens_equal": all(row["token_equal"] for row in rows),
        "maximum_abs_error": max(row["max_abs_error"] for row in rows),
    }


def replay_ledger(label: str, receipt: Mapping[str, Any], expected_rounds: int) -> None:
    require(receipt.get("verified") is True, f"{label} ledger is not verified")
    require(
        receipt.get("same_unified_attention_kernel") is True
        and receipt.get("dense_fallback_calls") == 0
        and receipt.get("full_kv_concatenations") == 0,
        f"{label} operator receipt drift",
    )
    calls = receipt.get("calls")
    require(
        isinstance(calls, list)
        and len(calls) == len(FULL_LAYERS) * expected_rounds,
        f"{label} call cardinality drift",
    )
    expected_order = list(FULL_LAYERS) * expected_rounds
    require(
        [row.get("layer_idx") for row in calls] == expected_order,
        f"{label} layer/round order drift",
    )
    require(
        receipt.get("total_calls") == len(calls),
        f"{label} total call count drift",
    )
    require(
        all(
            row.get("kernel_mode")
            == "vllm_0_26_triton_unified_attention_q16_block_pool"
            and row.get("full_kv_concatenations") == 0
            for row in calls
        ),
        f"{label} per-call operator path drift",
    )


def replay_arm(label: str, arm: Mapping[str, Any]) -> dict[str, Any]:
    require(arm.get("arm") == label, f"{label} arm label drift")
    require(
        arm.get("source_document_immutable") is True
        and arm.get("source_document_sha256_before")
        == arm.get("source_document_sha256_after"),
        f"{label} source document immutability drift",
    )
    ownership = arm.get("ownership")
    require(
        isinstance(ownership, Mapping)
        and set(ownership) == {"pre", "after_initial", "after_rebind", "final"}
        and all(row.get("passed") is True for row in ownership.values()),
        f"{label} ownership receipt drift",
    )
    lifecycle = arm.get("lifecycle")
    replay = replay_slot_events(lifecycle)
    require(replay == arm.get("lifecycle_replay"), f"{label} lifecycle replay drift")
    require(
        replay["final_epochs"] == [0, 1]
        and replay["event_count"] == 4,
        f"{label} lifecycle terminal drift",
    )
    require(
        arm.get("stale_cancelled_lease_gate") == "STALE_SLOT_LEASE"
        and arm.get("scrub", {}).get("pre_scrub_positive_control_gate")
        == "RECLAIM_NOT_ZERO"
        and arm.get("scrub", {}).get("zero_scrubbed") is True
        and arm.get("scrub", {}).get("exact_physical_ids_verified") is True
        and arm.get("exact_private_reservation_reuse") is True,
        f"{label} cancel/scrub/reclaim receipt drift",
    )
    ledgers = arm.get("ledger_receipts")
    require(isinstance(ledgers, Mapping), f"{label} ledger map missing")
    replay_ledger(f"{label}/survivor", ledgers["survivor"], 2)
    replay_ledger(f"{label}/cancelled", ledgers["cancelled"], 1)
    replay_ledger(f"{label}/replacement", ledgers["replacement"], 1)

    phase_receipts = arm.get("phase_receipts")
    require(
        isinstance(phase_receipts, list) and len(phase_receipts) == 2,
        f"{label} phase receipt cardinality drift",
    )
    overlaps = []
    if label == "concurrent":
        for phase in phase_receipts:
            require(
                phase.get("execution")
                == "two-host-workers-two-distinct-cuda-streams-one-barrier"
                and phase.get("distinct_host_thread_count") == 2
                and phase.get("distinct_cuda_stream_count") == 2,
                "concurrent execution identity drift",
            )
            intervals = phase.get("intervals")
            require(
                len({row.get("host_thread_id") for row in intervals}) == 2
                and len({row.get("cuda_stream_handle") for row in intervals}) == 2,
                "concurrent worker/stream uniqueness drift",
            )
            overlap = interval_overlap_ms(intervals)
            require(overlap == phase.get("overlap_ms"), "stream overlap arithmetic drift")
            require(
                phase.get("overlap_gate_passed") is (overlap > 0.0),
                "stream overlap gate drift",
            )
            require(
                phase.get("simultaneous_kernel_execution_claimed") is False
                and phase.get("continuous_batching_claimed") is False,
                "concurrent phase overclaims its treatment",
            )
            overlaps.append(overlap)
    else:
        require(
            all(
                phase.get("concurrent_execution_claimed") is False
                for phase in phase_receipts
            ),
            "serialized arm claims concurrency",
        )
    return {"lifecycle_event_count": replay["event_count"], "overlap_ms": overlaps}


def replay_result(
    design: Mapping[str, Any],
    result: Mapping[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    require(design.get("schema_version") == DESIGN_SCHEMA, "design schema drift")
    require(result.get("schema_version") == RESULT_SCHEMA, "result schema drift")
    require(result.get("status") == "completed", "formal execution did not complete")
    serialized_receipt = replay_arm("serialized", result["serialized"])
    concurrent_receipt = replay_arm("concurrent", result["concurrent"])
    serialized_logits = read_sidecar(artifact_dir, result["sidecars"]["serialized"])
    concurrent_logits = read_sidecar(artifact_dir, result["sidecars"]["concurrent"])
    oracle = compare_logits(serialized_logits, concurrent_logits)
    producer_oracle = result.get("output_oracle")
    require(
        oracle["sample_count"] == producer_oracle.get("sample_count")
        and oracle["all_full_vocab_logits_torch_equal"]
        is producer_oracle.get("all_full_vocab_logits_torch_equal")
        and oracle["all_generated_tokens_equal"]
        is producer_oracle.get("all_generated_tokens_equal")
        and oracle["maximum_abs_error"] == producer_oracle.get("maximum_abs_error"),
        "independent logit oracle differs from producer summary",
    )
    for observed, producer in zip(oracle["rows"], producer_oracle["rows"]):
        require(
            observed["sample_id"] == producer.get("sample_id")
            and observed["torch_equal"] is producer.get("torch_equal")
            and observed["token_equal"] is producer.get("token_equal")
            and math.isclose(
                observed["max_abs_error"],
                producer.get("max_abs_error"),
                rel_tol=1e-6,
                abs_tol=1e-12,
            )
            and math.isclose(
                observed["mean_abs_error"],
                producer.get("mean_abs_error"),
                rel_tol=1e-6,
                abs_tol=1e-12,
            ),
            "per-sample logit replay drift",
        )
    final_kv_equal = (
        result["serialized"]["final_logical_kv"]
        == result["concurrent"]["final_logical_kv"]
    )
    final_gdn_equal = (
        result["serialized"]["final_gdn_state"]
        == result["concurrent"]["final_gdn_state"]
    )
    treatment_valid = all(value > 0.0 for value in concurrent_receipt["overlap_ms"])
    primary_success = (
        treatment_valid
        and oracle["all_full_vocab_logits_torch_equal"]
        and oracle["all_generated_tokens_equal"]
        and final_kv_equal
        and final_gdn_equal
    )
    require(
        result.get("scientific_run_valid") is treatment_valid
        and result.get("concurrency_treatment_valid") is treatment_valid
        and result.get("formal_evidence_eligible") is treatment_valid
        and result.get("primary_success") is primary_success,
        "producer terminal disposition drift",
    )
    require(
        result.get("claim_boundary") == design.get("claim_boundary"),
        "claim boundary drift",
    )
    return {
        "schema_version": REPLAY_SCHEMA,
        "status": "completed",
        "independent_replay_passed": True,
        "scientific_run_valid": treatment_valid,
        "primary_success": primary_success,
        "serialized": serialized_receipt,
        "concurrent": concurrent_receipt,
        "output_oracle": oracle,
        "final_logical_kv_equal": final_kv_equal,
        "final_gdn_state_equal": final_gdn_equal,
        "claim_boundary": design["claim_boundary"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--design-preregistration", type=Path, required=True)
    result.add_argument("--expected-design-sha256", required=True)
    result.add_argument("--formal-result", type=Path, required=True)
    result.add_argument("--expected-formal-result-sha256", required=True)
    result.add_argument("--artifact-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    design = load_bound_json(
        args.design_preregistration,
        args.expected_design_sha256,
        "design preregistration",
    )
    formal = load_bound_json(
        args.formal_result,
        args.expected_formal_result_sha256,
        "formal result",
    )
    replay = replay_result(design, formal, args.artifact_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, replay)
    print(json.dumps(replay, sort_keys=True))


if __name__ == "__main__":
    main()
