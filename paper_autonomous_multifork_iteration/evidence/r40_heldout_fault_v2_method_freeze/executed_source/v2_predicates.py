"""General method-v2 semantic, allocator, and atomic-coherence predicates."""

from __future__ import annotations

from array import array
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from v2_common import (
    ContractError,
    require,
    require_sha256,
    safe_existing_file,
    safe_relative_path,
    sha256_bytes,
    sha256_json,
    verify_seal,
)


SEMANTIC_SCHEMA = "forkaudit-method-v2-semantic-arm-v1"
ALLOCATOR_SCHEMA = "forkaudit-method-v2-allocator-arm-v1"
ATOMIC_RECEIPT_SCHEMA = "forkaudit-method-v2-atomic-call-receipt-v1"
LIVE_SOURCE_KIND = "live-state-independent-reread-v1"
ALLOCATOR_PHASES = ("H0", "H1", "H4", "H6", "H7")


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    require(set(value.keys()) == set(expected), label + " fields")


def _call_key(row: Mapping[str, Any]) -> tuple[int, int, str]:
    key = row.get("call_key")
    require(isinstance(key, Mapping), "call key")
    _exact_keys(key, ("call_index", "round_index", "request_id"), "call key")
    call_index = key.get("call_index")
    round_index = key.get("round_index")
    request_id = key.get("request_id")
    require(type(call_index) is int and call_index >= 0, "call index")
    require(type(round_index) is int and round_index >= 0, "round index")
    require(isinstance(request_id, str) and request_id != "", "request id")
    return call_index, round_index, request_id


def _load_logits(root: Path, row: Mapping[str, Any], vocab_size: int, label: str) -> tuple[bytes, list[float]]:
    require(isinstance(row, Mapping), label + " descriptor")
    _exact_keys(row, ("path", "sha256", "nbytes", "shape", "dtype"), label)
    relative = safe_relative_path(row.get("path"), label + " path")
    path = safe_existing_file(root, relative, label)
    raw = path.read_bytes()
    require(row.get("dtype") == "float32-little-endian", label + " dtype")
    require(row.get("shape") == [1, vocab_size], label + " shape")
    require(row.get("nbytes") == vocab_size * 4 == len(raw), label + " bytes")
    require(require_sha256(row.get("sha256"), label + " SHA") == sha256_bytes(raw), label + " hash")
    values = array("f")
    values.frombytes(raw)
    if sys.byteorder != "little":
        values.byteswap()
    result = list(values)
    require(len(result) == vocab_size and all(math.isfinite(item) for item in result), label + " finite")
    return raw, result


def _semantic_arm(value: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    require(isinstance(value, Mapping), label + " semantic arm")
    _exact_keys(value, ("schema_version", "arm", "calls"), label + " semantic arm")
    require(value.get("schema_version") == SEMANTIC_SCHEMA, label + " schema")
    require(value.get("arm") == label, label + " arm")
    calls = value.get("calls")
    require(isinstance(calls, list) and calls, label + " calls")
    for row in calls:
        require(isinstance(row, Mapping), label + " semantic call")
        _exact_keys(row, ("call_key", "token_id", "logits"), label + " semantic call")
    keys = [_call_key(row) for row in calls]
    require(keys == sorted(keys), label + " call order")
    require(len(keys) == len(set(keys)), label + " duplicate call")
    return calls


def evaluate_semantic_pair(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    reference_root: Path,
    candidate_root: Path,
    policy: Mapping[str, Any],
    vocab_size: int,
) -> dict[str, Any]:
    """Compare complete ordered calls and independently read FP32 sidecars."""

    require(type(vocab_size) is int and vocab_size > 0, "vocab size")
    mode = policy.get("mode")
    require(mode in ("exact", "declared_tolerance"), "semantic mode")
    max_abs_limit = policy.get("max_abs_threshold")
    rel_l2_limit = policy.get("relative_l2_threshold")
    require(type(max_abs_limit) in (int, float) and max_abs_limit >= 0, "max abs threshold")
    require(type(rel_l2_limit) in (int, float) and rel_l2_limit >= 0, "relative L2 threshold")
    require(math.isfinite(float(max_abs_limit)), "finite max abs threshold")
    require(math.isfinite(float(rel_l2_limit)), "finite relative L2 threshold")
    if mode == "exact":
        require(float(max_abs_limit) == 0.0 and float(rel_l2_limit) == 0.0, "exact thresholds")

    ref_calls = _semantic_arm(reference, "reference")
    cand_calls = _semantic_arm(candidate, "candidate")
    ref_keys = [_call_key(row) for row in ref_calls]
    cand_keys = [_call_key(row) for row in cand_calls]
    cardinality_and_order_exact = ref_keys == cand_keys
    comparisons = []
    if cardinality_and_order_exact:
        for index, (ref_row, cand_row) in enumerate(zip(ref_calls, cand_calls)):
            require(type(ref_row.get("token_id")) is int, "reference token")
            require(type(cand_row.get("token_id")) is int, "candidate token")
            ref_raw, ref_values = _load_logits(reference_root, ref_row.get("logits"), vocab_size, "reference logits")
            cand_raw, cand_values = _load_logits(candidate_root, cand_row.get("logits"), vocab_size, "candidate logits")
            token_exact = ref_row["token_id"] == cand_row["token_id"]
            exact = ref_raw == cand_raw
            differences = [abs(left - right) for left, right in zip(ref_values, cand_values)]
            max_abs = max(differences) if differences else 0.0
            numerator = math.sqrt(sum((left - right) ** 2 for left, right in zip(ref_values, cand_values)))
            denominator = max(math.sqrt(sum(left * left for left in ref_values)), 1e-30)
            relative_l2 = numerator / denominator
            logit_pass = exact if mode == "exact" else max_abs <= float(max_abs_limit) and relative_l2 <= float(rel_l2_limit)
            comparisons.append({
                "call_index": index,
                "call_key": ref_row["call_key"],
                "token_exact": token_exact,
                "logit_bytes_exact": exact,
                "max_abs": max_abs,
                "relative_l2": relative_l2,
                "logit_pass": logit_pass,
            })
    passed = cardinality_and_order_exact and all(row["token_exact"] and row["logit_pass"] for row in comparisons)
    return {
        "schema_version": "forkaudit-method-v2-semantic-pair-verdict-v1",
        "mode": mode,
        "vocab_size": vocab_size,
        "reference_call_count": len(ref_calls),
        "candidate_call_count": len(cand_calls),
        "call_cardinality_and_order_exact": cardinality_and_order_exact,
        "comparisons": comparisons,
        "passed": passed,
        "attribution": "paired_semantic_baseline",
    }


def _allocator_arm(value: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    require(isinstance(value, Mapping), label + " allocator arm")
    _exact_keys(value, ("schema_version", "arm", "peak_reset_before_h0", "endpoints"),
                label + " allocator arm")
    require(value.get("schema_version") == ALLOCATOR_SCHEMA, label + " allocator schema")
    require(value.get("arm") == label, label + " allocator arm")
    require(value.get("peak_reset_before_h0") is True, label + " peak reset")
    rows = value.get("endpoints")
    require(isinstance(rows, list) and len(rows) == len(ALLOCATOR_PHASES), label + " allocator rows")
    require(all(isinstance(row, Mapping) for row in rows), label + " allocator row")
    require([row.get("phase") for row in rows] == list(ALLOCATOR_PHASES), label + " allocator phase order")
    for row in rows:
        _exact_keys(row, (
            "phase", "synchronized", "sync_event_id", "current_allocated_bytes",
            "peak_allocated_bytes",
        ), label + " allocator row")
        require(row.get("synchronized") is True, label + " allocator synchronization")
        require(isinstance(row.get("sync_event_id"), str) and row["sync_event_id"] != "", label + " sync event")
        current = row.get("current_allocated_bytes")
        peak = row.get("peak_allocated_bytes")
        require(type(current) is int and current >= 0, label + " current bytes")
        require(type(peak) is int and peak >= current, label + " peak bytes")
    return rows


def evaluate_allocator_pair(reference: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    ref_rows = _allocator_arm(reference, "reference")
    cand_rows = _allocator_arm(candidate, "candidate")
    comparisons = []
    for ref_row, cand_row in zip(ref_rows, cand_rows):
        current_exact = ref_row["current_allocated_bytes"] == cand_row["current_allocated_bytes"]
        peak_exact = ref_row["peak_allocated_bytes"] == cand_row["peak_allocated_bytes"]
        comparisons.append({
            "phase": ref_row["phase"],
            "current_exact": current_exact,
            "peak_exact": peak_exact,
            "current_delta_bytes": cand_row["current_allocated_bytes"] - ref_row["current_allocated_bytes"],
            "peak_delta_bytes": cand_row["peak_allocated_bytes"] - ref_row["peak_allocated_bytes"],
        })
    reference_restored = ref_rows[-1]["current_allocated_bytes"] == ref_rows[0]["current_allocated_bytes"]
    candidate_restored = cand_rows[-1]["current_allocated_bytes"] == cand_rows[0]["current_allocated_bytes"]
    passed = reference_restored and candidate_restored and all(
        row["current_exact"] and row["peak_exact"] for row in comparisons
    )
    return {
        "schema_version": "forkaudit-method-v2-allocator-pair-verdict-v1",
        "comparisons": comparisons,
        "reference_restored": reference_restored,
        "candidate_restored": candidate_restored,
        "passed": passed,
        "attribution": "paired_allocator_baseline",
    }


LIVE_FIELDS = (
    "request_id",
    "kv_logical_length",
    "kv_content_sha256",
    "gdn_content_sha256",
    "kv_version",
    "gdn_version",
    "kv_commit_epoch",
    "gdn_commit_epoch",
    "observation_id",
    "source_kind",
    "synchronized",
)

ATOMIC_RECEIPT_FIELDS = (
    "schema_version",
    "policy_sha256",
    "call_key",
    "input_token_count",
    "surfaced_token_id",
    "logits",
    "live_pre",
    "live_post",
    "model_reported_state_used_by_gate",
    "payload_sha256",
)


def validate_live_snapshot(value: Mapping[str, Any], request_id: str, label: str) -> None:
    _exact_keys(value, LIVE_FIELDS, label)
    require(value.get("request_id") == request_id, label + " request")
    require(type(value.get("kv_logical_length")) is int and value["kv_logical_length"] >= 0, label + " KV length")
    for field in ("kv_version", "gdn_version", "kv_commit_epoch", "gdn_commit_epoch"):
        require(type(value.get(field)) is int and value[field] >= 0, label + " " + field)
    require_sha256(value.get("kv_content_sha256"), label + " KV digest")
    require_sha256(value.get("gdn_content_sha256"), label + " GDN digest")
    require(isinstance(value.get("observation_id"), str) and value["observation_id"] != "", label + " observation")
    require(value.get("source_kind") == LIVE_SOURCE_KIND, label + " live source")
    require(value.get("synchronized") is True, label + " synchronization")


def _atomic_key(receipt: Mapping[str, Any]) -> tuple[int, int, str]:
    return _call_key({"call_key": receipt.get("call_key")})


def evaluate_atomic_sequence(
    receipts: Sequence[Mapping[str, Any]],
    expected_schedule: Sequence[Mapping[str, Any]],
    policy_sha256: str,
) -> dict[str, Any]:
    require_sha256(policy_sha256, "atomic policy SHA")
    require(isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)), "atomic receipts")
    require(isinstance(expected_schedule, Sequence) and not isinstance(expected_schedule, (str, bytes)), "atomic schedule")
    require(len(expected_schedule) > 0, "nonempty atomic schedule")
    expected_keys = [_call_key({"call_key": row}) for row in expected_schedule]
    require(expected_keys == sorted(expected_keys), "atomic schedule order")
    require(len(expected_keys) == len(set(expected_keys)), "atomic schedule duplicate")
    require(all(isinstance(row, Mapping) for row in receipts), "atomic receipt row")
    observed_keys = [_atomic_key(row) for row in receipts]
    cardinality_and_order_exact = observed_keys == expected_keys
    rows = []
    previous_post: dict[str, Mapping[str, Any]] = {}
    if cardinality_and_order_exact:
        for receipt in receipts:
            _exact_keys(receipt, ATOMIC_RECEIPT_FIELDS, "atomic receipt")
            require(receipt.get("schema_version") == ATOMIC_RECEIPT_SCHEMA, "atomic receipt schema")
            verify_seal(receipt, "atomic receipt")
            require(receipt.get("policy_sha256") == policy_sha256, "atomic policy binding")
            require(receipt.get("model_reported_state_used_by_gate") is False, "atomic state provenance")
            request_id = receipt["call_key"]["request_id"]
            pre = receipt.get("live_pre")
            post = receipt.get("live_post")
            require(isinstance(pre, Mapping) and isinstance(post, Mapping), "atomic live snapshots")
            validate_live_snapshot(pre, request_id, "atomic pre")
            validate_live_snapshot(post, request_id, "atomic post")
            require(pre["observation_id"] != post["observation_id"], "atomic distinct live reads")
            input_token_count = receipt.get("input_token_count")
            require(type(input_token_count) is int and input_token_count > 0, "atomic input token count")
            require(type(receipt.get("surfaced_token_id")) is int, "atomic surfaced token")
            logits = receipt.get("logits")
            require(isinstance(logits, Mapping), "atomic logit binding")
            _exact_keys(logits, ("path", "sha256", "nbytes", "shape", "dtype"), "atomic logits")
            require_sha256(logits.get("sha256"), "atomic logits SHA")
            checks = {
                "kv_length_delta": post["kv_logical_length"] - pre["kv_logical_length"] == input_token_count,
                "kv_version_delta": post["kv_version"] - pre["kv_version"] == 1,
                "gdn_version_delta": post["gdn_version"] - pre["gdn_version"] == 1,
                "kv_epoch_delta": post["kv_commit_epoch"] - pre["kv_commit_epoch"] == 1,
                "gdn_epoch_delta": post["gdn_commit_epoch"] - pre["gdn_commit_epoch"] == 1,
                "post_epoch_coherent": post["kv_commit_epoch"] == post["gdn_commit_epoch"],
                "pre_epoch_coherent": pre["kv_commit_epoch"] == pre["gdn_commit_epoch"],
            }
            if request_id in previous_post:
                previous = previous_post[request_id]
                continuity_fields = (
                    "kv_logical_length", "kv_content_sha256", "gdn_content_sha256",
                    "kv_version", "gdn_version", "kv_commit_epoch", "gdn_commit_epoch",
                )
                checks["cross_call_continuity"] = all(pre[field] == previous[field] for field in continuity_fields)
            else:
                checks["cross_call_continuity"] = True
            previous_post[request_id] = post
            rows.append({
                "call_key": receipt["call_key"],
                "checks": checks,
                "passed": all(checks.values()),
                "receipt_sha256": sha256_json(receipt),
            })
    return {
        "schema_version": "forkaudit-method-v2-atomic-sequence-verdict-v1",
        "expected_call_count": len(expected_keys),
        "observed_call_count": len(observed_keys),
        "call_cardinality_and_order_exact": cardinality_and_order_exact,
        "rows": rows,
        "passed": cardinality_and_order_exact and all(row["passed"] for row in rows),
        "attribution": "hybrid_atomic_version_coherence",
    }
