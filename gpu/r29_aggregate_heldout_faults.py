from __future__ import annotations

"""Strictly aggregate three Round-29 held-out-fault rank artifacts.

The aggregator independently reopens every FP32 sidecar, recomputes the
semantic comparisons, validates tri-state detector semantics and lifecycle
receipts, and emits per-fault rows only.  It contains no expected gate or
expected outcome mapping and never computes a detection rate.
"""

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import r29_heldout_fault_suite as fault_suite


INPUT_SCHEMA = "forkaudit-r29-heldout-execution-input-v1"
RANK_SCHEMA = "forkaudit-r29-heldout-fault-rank-v1"
SUMMARY_SCHEMA = "forkaudit-r29-heldout-fault-summary-v1"
LANE_ORDER = ("clean", "fault_conventional", "fault_forkaudit")
RECEIPT_ORDER = (
    "frozen_input_and_request_provenance",
    "live_kv_ownership_and_construction_binding",
    "gdn_phase_storage_snapshot_and_pointer_free_replay",
    "advertised_scheduler_action_sequence_replay",
    "persistent_kv_and_gdn_immutability",
    "mutation_restoration_or_fresh_case_disposal",
)
SIDE_CAR_SHAPE = (1, 248320)
SIDE_CAR_NBYTES = 993280
SHA256_RE = re.compile(r"[0-9a-f]{64}")
PRODUCTION_ASSERTION_ALLOWLIST = {
    "PA-Q16-CANONICAL-MASK-v1": {
        "message": "vLLM fused backend cannot replace this non-canonical attention mask",
        "function": "validate_canonical_tail_causal_mask",
    },
    "PA-Q16-PAIRED-VIEWS-v1": {
        "message": "fused backend requires paired Q16 paged views",
        "function": "_paired_sequence",
    },
}
MUTATION_DESCRIPTOR_KEYS = {
    "schema_version",
    "request_index",
    "layer_index",
    "field",
    "shape",
    "stride",
    "dtype",
    "device",
    "values",
    "values_sha256",
    "contains_absolute_pointer",
}
MUTATION_DESCRIPTOR_GEOMETRY_KEYS = (
    "schema_version",
    "request_index",
    "layer_index",
    "field",
    "shape",
    "stride",
    "dtype",
    "device",
    "contains_absolute_pointer",
)
DELTA_RESTORATION_KEYS = {
    "schema_version",
    "fault_id",
    "target_kind",
    "applied_pre_sha256",
    "applied_mutated_sha256",
    "mutation_coordinate_indices",
    "pre_restore_descriptor",
    "restored_descriptor",
    "pre_restore_sha256",
    "restored_sha256",
    "target_pre_values_sha256",
    "target_mutated_values_sha256",
    "target_pre_restore_values_sha256",
    "target_restored_values_sha256",
    "target_remained_mutated_through_horizon",
    "target_restored_exact",
    "non_target_preserved_across_undo",
    "restoration_observed",
    "contains_absolute_pointer",
}


class HeldOutAggregationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldOutAggregationError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def read_bound_file(path: Path, expected_sha256: str, label: str) -> bytes:
    require(SHA256_RE.fullmatch(expected_sha256 or "") is not None, f"{label} SHA format")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected_sha256, f"{label} raw SHA drift")
    return payload


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    require(not pending.exists(), "stale aggregate pending file")
    payload = canonical_bytes(value) + b"\n"
    with pending.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_execution_input(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict) and value.get("schema_version") == INPUT_SCHEMA, "execution input schema")
    require(value.get("status") == "frozen_before_candidate_outputs", "execution input status")
    require(value.get("fixed_protocol", {}).get("rank_assignment") == {"0": "H01", "1": "H02", "2": "H03"}, "execution rank assignment")
    require(value.get("fixed_protocol", {}).get("lane_order") == list(LANE_ORDER), "execution lane order")
    require(value.get("fixed_protocol", {}).get("sidecar_shape") == list(SIDE_CAR_SHAPE), "execution sidecar shape")
    require(value.get("fixed_protocol", {}).get("sidecar_dtype") == "float32", "execution sidecar dtype")
    require(value.get("fixed_protocol", {}).get("sidecar_nbytes") == SIDE_CAR_NBYTES, "execution sidecar bytes")
    claim = value.get("claim_boundary")
    require(
        claim == {
            "historical_pattern_inspired_only": True,
            "naturally_occurring_claimed": False,
            "upstream_implementation_evaluated": False,
            "detection_rate_reported": False,
        },
        "execution claim boundary",
    )
    return dict(value)


def _validate_detector_cell(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == {"status", "caught", "reason", "evidence"}, f"{label} detector schema")
    status = value["status"]
    require(status in ("evaluated", "not_evaluated"), f"{label} detector status")
    if status == "evaluated":
        require(type(value["caught"]) is bool, f"{label} evaluated detector caught")
    else:
        require(value["caught"] is None, f"{label} missing output must not become not-caught")
    require(isinstance(value["reason"], str) and value["reason"], f"{label} detector reason")
    return dict(value)


def _validate_production_assertion(cell: Mapping[str, Any], label: str) -> None:
    if cell["status"] != "evaluated" or cell["caught"] is False:
        return
    evidence = cell.get("evidence")
    require(isinstance(evidence, dict) and evidence.get("classification") == "exact_production_assertion", f"{label} production classification")
    allowlist_id = evidence.get("production_assertion_allowlist_id")
    require(allowlist_id in PRODUCTION_ASSERTION_ALLOWLIST, f"{label} production allowlist id")
    exception = evidence.get("exception")
    expected = PRODUCTION_ASSERTION_ALLOWLIST[allowlist_id]
    require(isinstance(exception, dict) and exception.get("module") == "qcomem_vllm_paged_kernel" and exception.get("type") == "QComemPagedKernelError", f"{label} production exception authority")
    require(exception.get("message") == expected["message"], f"{label} production message")
    stack = exception.get("stack")
    require(
        isinstance(stack, list)
        and any(row.get("filename") == "qcomem_vllm_paged_kernel.py" and row.get("function") == expected["function"] for row in stack if isinstance(row, dict)),
        f"{label} production traceback provenance",
    )


def _load_sidecar(reference: Any, raw_root: Path, label: str) -> tuple[np.ndarray, dict[str, Any]]:
    require(isinstance(reference, dict), f"{label} sidecar reference")
    require(set(reference) == {"path", "sha256", "dtype", "shape", "nbytes", "finite", "contains_absolute_pointer"}, f"{label} sidecar schema")
    require(reference["dtype"] == "float32" and reference["shape"] == list(SIDE_CAR_SHAPE) and reference["nbytes"] == SIDE_CAR_NBYTES, f"{label} sidecar geometry")
    require(reference["finite"] is True and reference["contains_absolute_pointer"] is False, f"{label} sidecar flags")
    relative = Path(reference["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, f"{label} unsafe sidecar path")
    path = raw_root / relative
    payload = path.read_bytes()
    require(len(payload) == SIDE_CAR_NBYTES, f"{label} sidecar byte length")
    require(sha256_bytes(payload) == reference["sha256"], f"{label} sidecar SHA")
    values = np.frombuffer(payload, dtype=np.float32).copy().reshape(SIDE_CAR_SHAPE)
    require(bool(np.isfinite(values).all()), f"{label} sidecar non-finite")
    return values, dict(reference)


def _recompute_comparisons(clean: Mapping[str, Any], faulty: Mapping[str, Any], raw_root: Path, label: str) -> dict[str, Any]:
    recorded = faulty.get("semantic_comparisons")
    require(isinstance(recorded, dict) and set(recorded) == {"greedy_token", "full_fp32_logits"}, f"{label} comparison schema")
    token_cell = _validate_detector_cell(recorded["greedy_token"], f"{label} token")
    logit_cell = _validate_detector_cell(recorded["full_fp32_logits"], f"{label} logits")
    token_ready = clean.get("semantic_horizon_reached") is True and faulty.get("semantic_horizon_reached") is True and type(clean.get("greedy_token_id")) is int and type(faulty.get("greedy_token_id")) is int
    if token_ready:
        expected_caught = clean["greedy_token_id"] != faulty["greedy_token_id"]
        require(token_cell["status"] == "evaluated" and token_cell["caught"] is expected_caught, f"{label} token verdict drift")
    else:
        require(token_cell["status"] == "not_evaluated" and token_cell["caught"] is None, f"{label} token missing-output semantics")

    logit_ready = clean.get("semantic_horizon_reached") is True and faulty.get("semantic_horizon_reached") is True and clean.get("full_logits") is not None and faulty.get("full_logits") is not None
    recomputed: dict[str, Any] | None = None
    if logit_ready:
        clean_values, clean_ref = _load_sidecar(clean["full_logits"], raw_root, f"{label} clean")
        fault_values, fault_ref = _load_sidecar(faulty["full_logits"], raw_root, f"{label} fault")
        exact = clean_ref["sha256"] == fault_ref["sha256"]
        delta = fault_values.astype(np.float64) - clean_values.astype(np.float64)
        max_abs = float(np.max(np.abs(delta)))
        denominator = max(float(np.linalg.norm(clean_values.astype(np.float64).ravel())), 1e-30)
        relative_l2 = float(np.linalg.norm(delta.ravel())) / denominator
        argmax_equal = int(np.argmax(clean_values, axis=-1)[0]) == int(np.argmax(fault_values, axis=-1)[0])
        require(logit_cell["status"] == "evaluated" and logit_cell["caught"] is (not exact), f"{label} full-logit verdict drift")
        evidence = logit_cell["evidence"]
        require(isinstance(evidence, dict), f"{label} full-logit evidence")
        require(evidence.get("exact") is exact and evidence.get("clean_sha256") == clean_ref["sha256"] and evidence.get("fault_sha256") == fault_ref["sha256"], f"{label} full-logit SHA evidence")
        require(evidence.get("argmax_equal") is argmax_equal, f"{label} argmax evidence")
        require(math.isfinite(float(evidence.get("max_absolute_difference"))) and math.isclose(float(evidence["max_absolute_difference"]), max_abs, rel_tol=1e-6, abs_tol=1e-7), f"{label} max-absolute metric")
        require(math.isfinite(float(evidence.get("relative_l2"))) and math.isclose(float(evidence["relative_l2"]), relative_l2, rel_tol=2e-5, abs_tol=1e-12), f"{label} relative-L2 metric")
        recomputed = {
            "exact": exact,
            "argmax_equal": argmax_equal,
            "max_absolute_difference": max_abs,
            "relative_l2": relative_l2,
            "clean_sha256": clean_ref["sha256"],
            "fault_sha256": fault_ref["sha256"],
        }
    else:
        require(logit_cell["status"] == "not_evaluated" and logit_cell["caught"] is None, f"{label} logit missing-output semantics")
    return {"greedy_token": token_cell, "full_fp32_logits": logit_cell, "aggregator_recomputed_full_logits": recomputed}


def _validate_receipt_prefix(case: Mapping[str, Any], label: str) -> None:
    completed = case.get("completed_receipts")
    require(isinstance(completed, list), f"{label} completed receipts")
    receipt_ids = []
    for row in completed:
        require(isinstance(row, dict) and set(row) == {"receipt_id", "status", "payload"} and row["status"] == "passed", f"{label} receipt schema")
        receipt_ids.append(row["receipt_id"])
    require(tuple(receipt_ids) == RECEIPT_ORDER[: len(receipt_ids)], f"{label} receipt order/prefix")
    rejection = case.get("first_authenticated_rejection")
    if rejection is not None:
        require(isinstance(rejection, dict) and rejection.get("authenticated") is True, f"{label} authenticated rejection")
        receipt_id = rejection.get("receipt_id")
        if receipt_id == "model_execution_registered_runtime_receipt":
            require(not completed, f"{label} pre-horizon rejection cannot have completed battery receipts")
        else:
            require(len(completed) < len(RECEIPT_ORDER) and receipt_id == RECEIPT_ORDER[len(completed)], f"{label} first-rejection position")
        require(isinstance(rejection.get("predicate_id"), str) and rejection["predicate_id"], f"{label} rejection predicate")
        exception = rejection.get("exception")
        require(isinstance(exception, dict), f"{label} rejection exception receipt")
        require(exception.get("gate_id") == rejection["predicate_id"], f"{label} rejection predicate/exception binding")
        authority = (exception.get("module"), exception.get("type"))
        imported_authorities = {
            ("qcomem_vllm_paged_multifork_resident", "RuntimeInvariantError"),
            ("qcomem_forkaudit_storage_witness", "GDNStorageWitnessError"),
        }
        executor_authority = exception.get("type") == "ReceiptPredicateRejection" and exception.get("module") in {"__main__", "r29_execute_heldout_faults"}
        require(authority in imported_authorities or executor_authority, f"{label} unregistered rejection authority")
        stack = exception.get("stack")
        require(isinstance(stack, list) and bool(stack), f"{label} rejection traceback missing")
        if executor_authority:
            require(any(isinstance(frame, dict) and frame.get("filename") == "r29_execute_heldout_faults.py" for frame in stack), f"{label} executor rejection source provenance")


def _validate_schedule_outcome(case: Mapping[str, Any], label: str) -> None:
    """Recompute the advertised-action predicate without consulting fault id."""

    action = case.get("action_sequence")
    require(isinstance(action, dict), f"{label} action sequence")
    events = action.get("events")
    advertised = action.get("advertised_logical_advance_tokens")
    actual = action.get("actual_model_invocations")
    require(isinstance(events, list) and type(advertised) is int and type(actual) is int and len(events) == actual, f"{label} action sequence geometry")
    observed = [
        {
            "event_index": index,
            "phase": "advertised-model-boundary" if event.get("externally_advertised") is True else "hidden-model-boundary",
            "slot_id": 0,
            "round_index": index,
            "request_id": "request-0",
        }
        for index, event in enumerate(events)
    ]
    expected = [
        {
            "event_index": index,
            "phase": "advertised-model-boundary",
            "slot_id": 0,
            "round_index": index,
            "request_id": "request-0",
        }
        for index in range(advertised)
    ]
    exact = observed == expected
    completed_ids = [row["receipt_id"] for row in case.get("completed_receipts", [])]
    rejection = case.get("first_authenticated_rejection")
    if RECEIPT_ORDER[3] in completed_ids:
        require(exact, f"{label} passed scheduler receipt is false")
    if isinstance(rejection, dict) and rejection.get("receipt_id") == RECEIPT_ORDER[3]:
        require(not exact, f"{label} rejected an exact scheduler receipt")


def _validate_mutation_descriptor(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == MUTATION_DESCRIPTOR_KEYS, f"{label} descriptor schema")
    values = value["values"]
    require(isinstance(values, list) and bool(values) and all(type(item) is int for item in values), f"{label} descriptor values")
    require(value["values_sha256"] == sha256_json(values), f"{label} descriptor value digest")
    require(value["contains_absolute_pointer"] is False, f"{label} descriptor pointer flag")
    return dict(value)


def _validate_delta_scoped_restoration(
    applied: Mapping[str, Any],
    restoration: Any,
    *,
    label: str,
) -> None:
    """Independently replay the generic mutation-delta restoration receipt."""

    require(isinstance(restoration, dict) and set(restoration) == DELTA_RESTORATION_KEYS, f"{label} restoration schema")
    require(restoration["schema_version"] == "forkaudit-r29-heldout-restoration-v2", f"{label} restoration version")
    require(restoration["fault_id"] == applied["fault_id"] and restoration["target_kind"] == applied["target_kind"], f"{label} restoration binding")
    pre = _validate_mutation_descriptor(applied["pre_descriptor"], f"{label} applied pre")
    mutated = _validate_mutation_descriptor(applied["mutated_descriptor"], f"{label} applied mutated")
    pre_restore = _validate_mutation_descriptor(restoration["pre_restore_descriptor"], f"{label} pre-restore")
    restored = _validate_mutation_descriptor(restoration["restored_descriptor"], f"{label} restored")
    require(
        all(
            pre[key] == mutated[key] == pre_restore[key] == restored[key]
            for key in MUTATION_DESCRIPTOR_GEOMETRY_KEYS
        ),
        f"{label} restoration geometry drift",
    )
    require(len(pre["values"]) == len(mutated["values"]) == len(pre_restore["values"]) == len(restored["values"]), f"{label} restoration value length")
    changed_indices = [
        index
        for index, (left, right) in enumerate(zip(pre["values"], mutated["values"]))
        if left != right
    ]
    require(bool(changed_indices) and restoration["mutation_coordinate_indices"] == changed_indices, f"{label} mutation delta coordinates")
    changed_set = set(changed_indices)
    require(all(pre_restore["values"][index] == mutated["values"][index] for index in changed_indices), f"{label} target did not remain mutated through horizon")
    require(all(restored["values"][index] == pre["values"][index] for index in changed_indices), f"{label} target not restored")
    require(
        all(
            restored["values"][index] == pre_restore["values"][index]
            for index in range(len(restored["values"]))
            if index not in changed_set
        ),
        f"{label} non-target changed during undo",
    )
    target_pre = [pre["values"][index] for index in changed_indices]
    target_mutated = [mutated["values"][index] for index in changed_indices]
    target_pre_restore = [pre_restore["values"][index] for index in changed_indices]
    target_restored = [restored["values"][index] for index in changed_indices]
    require(restoration["applied_pre_sha256"] == applied["pre_sha256"] == sha256_json(pre), f"{label} applied pre digest")
    require(restoration["applied_mutated_sha256"] == applied["mutated_sha256"] == sha256_json(mutated), f"{label} applied mutated digest")
    require(restoration["pre_restore_sha256"] == sha256_json(pre_restore) and restoration["restored_sha256"] == sha256_json(restored), f"{label} transition descriptor digests")
    require(restoration["target_pre_values_sha256"] == sha256_json(target_pre), f"{label} target pre digest")
    require(restoration["target_mutated_values_sha256"] == sha256_json(target_mutated), f"{label} target mutated digest")
    require(restoration["target_pre_restore_values_sha256"] == sha256_json(target_pre_restore), f"{label} target pre-restore digest")
    require(restoration["target_restored_values_sha256"] == sha256_json(target_restored), f"{label} target restored digest")
    for field in (
        "target_remained_mutated_through_horizon",
        "target_restored_exact",
        "non_target_preserved_across_undo",
        "restoration_observed",
    ):
        require(restoration[field] is True, f"{label} restoration flag {field}")
    require(restoration["contains_absolute_pointer"] is False, f"{label} restoration pointer flag")


def _validate_intervention(case: Mapping[str, Any], fault_id: str, lane: str, label: str) -> None:
    intervention = case.get("intervention")
    require(isinstance(intervention, dict), f"{label} intervention")
    if lane == "clean":
        require(intervention == {"kind": "none", "fault_active": False, "mutation_observed": False}, f"{label} clean intervention")
        return
    if fault_id in fault_suite.STATE_MUTATION_FAULT_IDS:
        require(intervention.get("kind") == "reversible_state_mutation" and intervention.get("fault_active") is True, f"{label} state intervention kind")
        applied = intervention.get("applied_receipt")
        require(isinstance(applied, dict) and applied.get("fault_id") == fault_id and applied.get("mutation_observed") is True, f"{label} state intervention receipt")
        require(applied.get("pre_sha256") != applied.get("mutated_sha256"), f"{label} state mutation no-op")
        require(applied.get("contains_absolute_pointer") is False, f"{label} state receipt pointer")
        pre = applied.get("pre_descriptor")
        mutated = applied.get("mutated_descriptor")
        require(isinstance(pre, dict) and isinstance(mutated, dict) and set(pre) == MUTATION_DESCRIPTOR_KEYS and set(mutated) == MUTATION_DESCRIPTOR_KEYS, f"{label} state descriptor schema")
        require(applied.get("pre_sha256") == sha256_json(pre) and applied.get("mutated_sha256") == sha256_json(mutated), f"{label} state descriptor digest")
        geometry = ("schema_version", "request_index", "layer_index", "field", "shape", "stride", "dtype", "device", "contains_absolute_pointer")
        require(all(pre[field] == mutated[field] for field in geometry), f"{label} state target geometry changed")
        require(pre["contains_absolute_pointer"] is False and pre["values"] != mutated["values"], f"{label} state target value mutation")
        require(pre["values_sha256"] == sha256_json(pre["values"]) and mutated["values_sha256"] == sha256_json(mutated["values"]), f"{label} state target value digest")
        restoration = case.get("restoration_receipt")
        _validate_delta_scoped_restoration(applied, restoration, label=label)
    else:
        require(fault_id in fault_suite.ACTION_SEQUENCE_FAULT_IDS, f"{label} unknown intervention type")
        expected = fault_suite.h02_action_sequence(request_index=0)
        require(intervention.get("kind") == "immutable_action_sequence" and intervention.get("fault_active") is True, f"{label} action intervention kind")
        require(intervention.get("action_sequence") == expected, f"{label} action sequence drift")
        require(intervention.get("action_sequence_sha256") == sha256_json(expected), f"{label} action sequence wrapper SHA")
        require(intervention.get("fresh_case_disposal_required") is True, f"{label} action disposal")


def _validate_lane(case: Any, *, fault_id: str, lane: str, baseline: Mapping[str, Any], global_nonces: set[str], label: str) -> list[str]:
    require(isinstance(case, dict) and case.get("lane") == lane, f"{label} lane binding")
    nonce = case.get("case_nonce")
    require(isinstance(nonce, str) and re.fullmatch(r"[0-9a-f]{32}", nonce) is not None and nonce not in global_nonces, f"{label} fresh case nonce")
    global_nonces.add(nonce)
    require(case.get("fresh_case") is True and case.get("state_reused_from_prior_lane") is False, f"{label} fresh state")
    require(case.get("allocator_before") == baseline and case.get("allocator_baseline") == baseline, f"{label} allocator precondition")
    cleanup = case.get("cleanup")
    require(isinstance(cleanup, dict), f"{label} cleanup receipt")
    require(cleanup.get("fresh_case_disposed") is True and cleanup.get("gc_collect_completed") is True and cleanup.get("cuda_empty_cache_completed") is True and cleanup.get("cuda_synchronize_completed") is True, f"{label} cleanup operations")
    require(cleanup.get("allocator_after") == baseline and cleanup.get("allocator_baseline_exact") is True, f"{label} allocator cleanup")
    _validate_intervention(case, fault_id, lane, label)
    production = _validate_detector_cell(case.get("production_assertion"), f"{label} production assertion")
    _validate_production_assertion(production, label)
    forkaudit = _validate_detector_cell(case.get("forkaudit"), f"{label} ForkAudit")
    _validate_receipt_prefix(case, label)
    if lane != "fault_conventional":
        _validate_schedule_outcome(case, label)
    if lane == "clean":
        require(case.get("completion_status") == "completed" and case.get("semantic_horizon_reached") is True, f"{label} clean completion")
        require(production["status"] == "evaluated" and production["caught"] is False, f"{label} clean production detector")
        require(forkaudit["status"] == "evaluated" and forkaudit["caught"] is False, f"{label} clean receipt battery")
        require(len(case["completed_receipts"]) == len(RECEIPT_ORDER) and case.get("first_authenticated_rejection") is None, f"{label} clean receipt coverage")
    elif lane == "fault_conventional":
        require(forkaudit["status"] == "not_evaluated" and forkaudit["caught"] is None, f"{label} conventional ForkAudit withholding")
        require(case.get("first_authenticated_rejection") is None and case.get("completed_receipts") == [], f"{label} conventional cannot retain receipt verdict")
    else:
        require(forkaudit["status"] == "evaluated", f"{label} ForkAudit detector must be evaluated")
        require((case.get("first_authenticated_rejection") is not None) is forkaudit["caught"], f"{label} ForkAudit first rejection mapping")
    invalid: list[str] = []
    if case.get("operational_invalid") is not None:
        invalid.append("recorded_operational_invalid")
    if cleanup.get("cleanup_passed") is not True or cleanup.get("registered_backend_restored") is not True or cleanup.get("cleanup_error") is not None:
        invalid.append("cleanup_invalid")
    if fault_id in fault_suite.STATE_MUTATION_FAULT_IDS and lane != "clean" and case.get("restoration_receipt") is None:
        invalid.append("state_restoration_missing")
    return invalid


def _forbidden_rate_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            require(key not in {"detection_rate", "sensitivity_rate", "caught_count", "detected_count"}, f"pooled detector field forbidden at {path}.{key}")
            _forbidden_rate_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _forbidden_rate_fields(child, f"{path}[{index}]")


def aggregate_documents(
    *,
    suite: Mapping[str, Any],
    execution_input: Mapping[str, Any],
    execution_input_raw_sha256: str,
    suite_raw_sha256: str,
    suite_canonical_sha256: str,
    ranks: Sequence[Mapping[str, Any]],
    raw_root: Path,
) -> dict[str, Any]:
    require(len(ranks) == 3, "exactly three rank artifacts required")
    rank_rows = sorted(ranks, key=lambda row: row.get("rank", -1))
    require([row.get("rank") for row in rank_rows] == [0, 1, 2], "rank coverage/order")
    require([row.get("fault_id") for row in rank_rows] == ["H01", "H02", "H03"], "fault coverage/order")
    expected_bindings = {
        "executor_sha256": execution_input["code"]["executor_sha256"],
        "aggregator_sha256": execution_input["code"]["aggregator_sha256"],
        "fault_module_sha256": execution_input["suite_binding"]["fault_module_sha256"],
        "launcher_sha256": execution_input["suite_binding"]["launcher_sha256"],
        "imported_rr2_code_ledger_raw_sha256": execution_input["code"]["imported_rr2_code_ledger_raw_sha256"],
    }
    uuids: list[str] = []
    global_nonces: set[str] = set()
    per_fault_rows: list[dict[str, Any]] = []
    operational_invalid_count = 0
    invalid_reasons: list[dict[str, Any]] = []
    for rank, row in enumerate(rank_rows):
        fault_id = ("H01", "H02", "H03")[rank]
        _forbidden_rate_fields(row, f"$rank[{rank}]")
        require(row.get("schema_version") == RANK_SCHEMA and row.get("status") == "completed_rank_artifact", f"rank {rank} artifact schema/status")
        require(row.get("run_id") == execution_input["run_id"], f"rank {rank} run id")
        require(row.get("suite_raw_sha256") == suite_raw_sha256 and row.get("suite_canonical_sha256") == suite_canonical_sha256, f"rank {rank} suite binding")
        require(row.get("execution_input_raw_sha256") == execution_input_raw_sha256, f"rank {rank} execution input binding")
        require(row.get("source_bindings") == expected_bindings, f"rank {rank} source bindings")
        require(row.get("detection_rate_reported") is False and row.get("naturally_occurring_claimed") is False and row.get("upstream_implementation_evaluated") is False, f"rank {rank} claim flags")
        hardware = row.get("hardware")
        require(isinstance(hardware, dict) and hardware.get("name") == "NVIDIA H20-3e" and hardware.get("compute_capability") == [9, 0], f"rank {rank} H20 receipt")
        uuid = hardware.get("uuid")
        require(isinstance(uuid, str) and uuid.startswith("GPU-"), f"rank {rank} GPU UUID")
        uuids.append(uuid)
        input_receipt = row.get("input_receipt")
        require(isinstance(input_receipt, dict) and input_receipt.get("rank") == rank and input_receipt.get("model_revision") == execution_input["model"]["revision"], f"rank {rank} input receipt")
        require(input_receipt.get("imported_rr2_code", {}).get("raw_sha256") == execution_input["code"]["imported_rr2_code_ledger_raw_sha256"], f"rank {rank} code-ledger receipt")
        warmup = row.get("discarded_warmup")
        require(isinstance(warmup, dict) and warmup.get("performed") is True and warmup.get("discarded") is True, f"rank {rank} discarded warmup")
        baseline = warmup.get("post_warmup_allocator_baseline")
        require(isinstance(baseline, dict) and set(baseline) == {"allocated_bytes", "reserved_bytes"}, f"rank {rank} allocator baseline")
        lanes = row.get("lanes")
        require(isinstance(lanes, list) and [case.get("lane") for case in lanes] == list(LANE_ORDER), f"rank {rank} lanes")
        lane_map = {case["lane"]: case for case in lanes}
        rank_invalid: list[str] = []
        for lane in LANE_ORDER:
            reasons = _validate_lane(lane_map[lane], fault_id=fault_id, lane=lane, baseline=baseline, global_nonces=global_nonces, label=f"rank {rank}/{lane}")
            operational_invalid_count += len(reasons) > 0
            rank_invalid.extend(f"{lane}:{reason}" for reason in reasons)
        clean = lane_map["clean"]
        clean_values, _clean_ref = _load_sidecar(clean.get("full_logits"), raw_root, f"rank {rank} clean")
        require(int(np.argmax(clean_values, axis=-1)[0]) == clean.get("greedy_token_id"), f"rank {rank} clean token/logit binding")
        conventional_comparison = _recompute_comparisons(clean, lane_map["fault_conventional"], raw_root, f"rank {rank} conventional")
        forkaudit_comparison = _recompute_comparisons(clean, lane_map["fault_forkaudit"], raw_root, f"rank {rank} ForkAudit")
        require(row.get("operational_invalid_count") == sum(case.get("operational_invalid") is not None for case in lanes), f"rank {rank} producer invalid count")
        if rank_invalid:
            invalid_reasons.append({"rank": rank, "fault_id": fault_id, "reasons": rank_invalid})
        per_fault_rows.append(
            {
                "rank": rank,
                "fault_id": fault_id,
                "hardware_uuid": uuid,
                "historical_pattern_inspired_only": True,
                "intervention": {
                    "conventional": lane_map["fault_conventional"]["intervention"],
                    "forkaudit": lane_map["fault_forkaudit"]["intervention"],
                },
                "clean": {
                    "completion_status": clean["completion_status"],
                    "semantic_horizon_reached": clean["semantic_horizon_reached"],
                    "receipt_battery": clean["forkaudit"],
                },
                "conventional_lane": {
                    "completion_status": lane_map["fault_conventional"]["completion_status"],
                    "production_assertion": lane_map["fault_conventional"]["production_assertion"],
                    "greedy_token": conventional_comparison["greedy_token"],
                    "full_fp32_logits": conventional_comparison["full_fp32_logits"],
                    "aggregator_recomputed_full_logits": conventional_comparison["aggregator_recomputed_full_logits"],
                },
                "forkaudit_lane": {
                    "completion_status": lane_map["fault_forkaudit"]["completion_status"],
                    "first_authenticated_rejection": lane_map["fault_forkaudit"]["first_authenticated_rejection"],
                    "completed_receipts": lane_map["fault_forkaudit"]["completed_receipts"],
                    "receipt_detector": lane_map["fault_forkaudit"]["forkaudit"],
                    "production_assertion": lane_map["fault_forkaudit"]["production_assertion"],
                    "greedy_token": forkaudit_comparison["greedy_token"],
                    "full_fp32_logits": forkaudit_comparison["full_fp32_logits"],
                    "aggregator_recomputed_full_logits": forkaudit_comparison["aggregator_recomputed_full_logits"],
                },
                "operationally_valid": not rank_invalid,
                "operational_invalid_reasons": rank_invalid,
            }
        )
    require(len(set(uuids)) == 3, "rank GPUs must be distinct")
    referenced_sidecars = {
        Path(case["full_logits"]["path"])
        for row in rank_rows
        for case in row["lanes"]
        if isinstance(case.get("full_logits"), dict)
    }
    present_sidecars = {
        path.relative_to(raw_root)
        for path in raw_root.glob("sidecars/rank-*/*.bin")
        if path.is_file()
    }
    require(present_sidecars == referenced_sidecars, "unreferenced or missing full-logit sidecar")
    scientific_valid = operational_invalid_count == 0 and not invalid_reasons
    result = {
        "schema_version": SUMMARY_SCHEMA,
        "status": "completed_strict_aggregation" if scientific_valid else "completed_operationally_invalid_aggregation",
        "run_id": execution_input["run_id"],
        "scientific_valid": scientific_valid,
        "operational_invalid_count": operational_invalid_count,
        "operational_invalid_reasons": invalid_reasons,
        "fault_ids": ["H01", "H02", "H03"],
        "per_fault_rows": per_fault_rows,
        "distinct_h20_uuid_count": len(set(uuids)),
        "suite_raw_sha256": suite_raw_sha256,
        "suite_canonical_sha256": suite_canonical_sha256,
        "execution_input_raw_sha256": execution_input_raw_sha256,
        "detection_rate_reported": False,
        "naturally_occurring_claimed": False,
        "upstream_implementation_evaluated": False,
        "negative_or_escaped_faults_retained": True,
    }
    _forbidden_rate_fields(result)
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    suite_raw = read_bound_file(args.suite, args.expected_suite_raw_sha256, "held-out suite")
    suite = json.loads(suite_raw)
    suite_receipt = fault_suite.validate_frozen_suite(suite)
    require(suite_receipt["suite_sha256"] == args.expected_suite_canonical_sha256, "suite canonical SHA drift")
    input_raw = read_bound_file(args.execution_input, args.expected_execution_input_sha256, "execution input")
    execution_input = _validate_execution_input(json.loads(input_raw))
    require(execution_input["suite_binding"]["raw_sha256"] == args.expected_suite_raw_sha256 and execution_input["suite_binding"]["canonical_sha256"] == args.expected_suite_canonical_sha256, "execution-input suite binding")
    require(sha256_file(Path(__file__).resolve()) == execution_input["code"]["aggregator_sha256"], "aggregator source SHA drift")
    require(Path(__file__).resolve() == Path(execution_input["code"]["aggregator_path"]).resolve(), "aggregator path binding drift")
    require(sha256_file(Path(execution_input["code"]["executor_path"])) == execution_input["code"]["executor_sha256"], "executor source SHA drift")
    rank_root = args.rank_root.resolve()
    require(rank_root == Path(execution_input["output"]["raw_root"]).resolve(), "rank root/output binding")
    expected_paths = [rank_root / f"heldout-fault-rank-{rank}.json" for rank in range(3)]
    require(all(path.is_file() for path in expected_paths), "missing rank artifact")
    present_rank_json = sorted(rank_root.glob("heldout-fault-rank-*.json"))
    require(present_rank_json == expected_paths, "unexpected/missing rank JSON artifact")
    ranks = [json.loads(path.read_text(encoding="utf-8")) for path in expected_paths]
    result = aggregate_documents(
        suite=suite,
        execution_input=execution_input,
        execution_input_raw_sha256=args.expected_execution_input_sha256,
        suite_raw_sha256=args.expected_suite_raw_sha256,
        suite_canonical_sha256=args.expected_suite_canonical_sha256,
        ranks=ranks,
        raw_root=rank_root,
    )
    _write_json_atomic(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--suite", type=Path, required=True)
    value.add_argument("--expected-suite-raw-sha256", required=True)
    value.add_argument("--expected-suite-canonical-sha256", required=True)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--expected-execution-input-sha256", required=True)
    value.add_argument("--rank-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    result = aggregate(parser().parse_args(argv))
    return 0 if result["scientific_valid"] else 3


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "HeldOutAggregationError",
    "LANE_ORDER",
    "RANK_SCHEMA",
    "RECEIPT_ORDER",
    "SUMMARY_SCHEMA",
    "aggregate",
    "aggregate_documents",
    "sha256_file",
]
