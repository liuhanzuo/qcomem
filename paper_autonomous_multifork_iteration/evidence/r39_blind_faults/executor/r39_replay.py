#!/usr/bin/env python3
"""Detached, model-import-free replay of R39 lane and four-observer artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import r39_contract as contract


CASE_SCHEMA = "forkaudit-r39-blind-fault-lane-v1"
FORBIDDEN_MODULE_PREFIXES = (
    "torch", "transformers", "vllm", "triton", "qcomem_",
    "r29_", "r33_", "r39_live_common", "r39_lane",
)


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_sidecar(root: Path, row: Any, label: str) -> dict[str, Any]:
    require(isinstance(row, Mapping), f"{label} sidecar")
    relative = Path(str(row.get("path", "")))
    require(not relative.is_absolute() and ".." not in relative.parts, f"{label} unsafe path")
    path = root / relative
    require(path.is_file(), f"{label} absent")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(len(raw) == row.get("nbytes") and digest == row.get("sha256"), f"{label} byte binding")
    require(row.get("shape") == [1, 248320] and row.get("dtype") == "float32-little-endian", f"{label} ABI")
    return {"path": relative.as_posix(), "sha256": digest, "nbytes": len(raw)}


def _verify_injection(case: Mapping[str, Any], lane: str) -> dict[str, Any]:
    value = case.get("byte_bound_injection_receipt")
    require(isinstance(value, Mapping), f"{lane} injection receipt")
    base = dict(value)
    digest = base.pop("receipt_sha256", None)
    require(digest == contract.sha256_json(base), f"{lane} injection receipt SHA")
    require(value.get("lane") == lane and value.get("fault_id") == case.get("fault_id"), f"{lane} injection identity")
    require(value.get("exactly_one_named_locus") is True, f"{lane} locus cardinality")
    require(value.get("semantic_outputs_used_for_target_selection") is False, f"{lane} outcome-guided selector")
    if lane == "mutant":
        require(value.get("payload_applied") is True and value.get("eligible_noop") is False, "mutant payload missing")
    else:
        require(value.get("payload_applied") is False and value.get("eligible_noop") is True, f"{lane} no-op missing")
    return dict(value)


def _verify_trace(root: Path, case: Mapping[str, Any], lane: str) -> dict[str, Any]:
    ref = case.get("forkaudit")
    require(isinstance(ref, Mapping), f"{lane} ForkAudit reference")
    relative = Path(str(ref.get("trace_path", "")))
    require(not relative.is_absolute() and ".." not in relative.parts, f"{lane} ForkAudit path")
    path = root / relative
    require(path.is_file() and contract.sha256_file(path) == ref.get("trace_sha256"), f"{lane} ForkAudit byte binding")
    trace = _read(path)
    require(trace.get("verdict") == ref.get("verdict"), f"{lane} ForkAudit verdict drift")
    require(trace.get("new_r39_predicates_added") is False, f"{lane} post-freeze predicate")
    require(trace.get("compiled_binary_identity_coverage") == "partial", f"{lane} compiled coverage inflation")
    require(trace.get("autotuning_choice_coverage") == "partial", f"{lane} autotune coverage inflation")
    rows = trace.get("predicate_rows")
    require(isinstance(rows, list) and len(rows) >= 4, f"{lane} ForkAudit predicate rows")
    return trace


def verify_lane(
    *, case: Any, lane: str, root: Path, fault_id: str,
    expected_feasibility_sha256: str,
) -> dict[str, Any]:
    require(isinstance(case, Mapping), f"{lane} case")
    require(case.get("schema_version") == CASE_SCHEMA, f"{lane} schema")
    require(case.get("run_id") == contract.RUN_ID and case.get("fault_id") == fault_id, f"{lane} identity")
    require(case.get("lane") == lane and case.get("status") == "expected_horizon_completed", f"{lane} status")
    require(case.get("operational_invalid") is None, f"{lane} invalid")
    require(case.get("expected_horizon") == contract.EXPECTED_HORIZON[fault_id], f"{lane} expected horizon")
    require(case.get("reached_horizon") == contract.EXPECTED_HORIZON[fault_id], f"{lane} horizon not reached")
    require(case.get("all_production_assertions_enabled") is True and case.get("selective_gate_suppression") is False, f"{lane} assertions")
    require(case.get("feasibility", {}).get("sha256") == expected_feasibility_sha256, f"{lane} feasibility binding")
    schedule = case.get("ordered_schedule")
    expected_schedule = [
        {"call_index": round_index * 2 + request_index, "round_index": round_index, "request_index": request_index}
        for round_index in range(8) for request_index in range(2)
    ]
    require(schedule == expected_schedule, f"{lane} schedule")
    sidecars = case.get("logit_sidecars")
    require(isinstance(sidecars, list) and len(sidecars) == 16, f"{lane} full-logit cardinality")
    verified_sidecars = [
        _safe_sidecar(root, row, f"{lane} logit {index}")
        for index, row in enumerate(sidecars)
    ]
    semantic = case.get("semantic_results")
    require(isinstance(semantic, Mapping), f"{lane} semantic results")
    tokens = semantic.get("generated_token_ids")
    require(isinstance(tokens, list) and len(tokens) == 2 and all(isinstance(row, list) and len(row) == 8 for row in tokens), f"{lane} tokens")
    persistent = case.get("persistent_base_snapshots")
    require(isinstance(persistent, Mapping) and set(persistent) == {"H1", "H4", "H6", "H7_pre_release"}, f"{lane} persistent phases")
    alloc = case.get("allocator_endpoints")
    require(isinstance(alloc, Mapping) and set(alloc) == {"H0", "H1", "H4", "H6", "H7"}, f"{lane} allocator phases")
    for phase in ("H0", "H1", "H4", "H6", "H7"):
        row = alloc[phase]
        require(isinstance(row, Mapping), f"{lane} allocator {phase}")
        for field in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes", "peak_reserved_bytes"):
            require(type(row.get(field)) is int and row[field] >= 0, f"{lane} allocator {phase}/{field}")
    injection = _verify_injection(case, lane)
    trace = _verify_trace(root, case, lane)
    return {
        "case": case,
        "tokens": tokens,
        "sidecars": verified_sidecars,
        "persistent": persistent,
        "allocator": alloc,
        "injection": injection,
        "trace": trace,
    }


def _persistent_invariant(value: Mapping[str, Any], *, require_h7: bool) -> dict[str, Any]:
    setup = value["H1"]
    phases = ["H4", "H6"] + (["H7_pre_release"] if require_h7 else [])
    missing = [phase for phase in phases if value.get(phase) is None]
    if missing:
        return {"status": "open", "passed": False, "missing_phases": missing}
    mismatches = [phase for phase in phases if value[phase] != setup]
    return {
        "status": "pass" if not mismatches else "fail",
        "passed": not mismatches,
        "compared_phases": phases,
        "mismatch_phases": mismatches,
    }


def _allocation_comparison(clean: Mapping[str, Any], mutant: Mapping[str, Any]) -> dict[str, Any]:
    rows = []
    for phase in ("H0", "H1", "H4", "H6", "H7"):
        fields = ("allocated_bytes", "peak_allocated_bytes")
        equal = all(clean[phase][field] == mutant[phase][field] for field in fields)
        rows.append({
            "phase": phase,
            "exact": equal,
            "clean": {field: clean[phase][field] for field in fields},
            "mutant": {field: mutant[phase][field] for field in fields},
            "delta": {field: mutant[phase][field] - clean[phase][field] for field in fields},
        })
    restoration = mutant["H7"]["allocated_bytes"] == mutant["H0"]["allocated_bytes"]
    return {
        "status": "pass" if all(row["exact"] for row in rows) and restoration else "fail",
        "passed": all(row["exact"] for row in rows) and restoration,
        "rows": rows,
        "mutant_h7_restores_h0_exact": restoration,
    }


def _output_comparison(clean: Mapping[str, Any], mutant: Mapping[str, Any]) -> dict[str, Any]:
    comparable = len(clean["sidecars"]) == len(mutant["sidecars"])
    return {
        "status": "evaluated" if comparable else "not_evaluable",
        "call_cardinality_equal": comparable,
        "tokens_exact": comparable and clean["tokens"] == mutant["tokens"],
        "complete_fp32_logits_byte_exact": comparable and [row["sha256"] for row in clean["sidecars"]] == [row["sha256"] for row in mutant["sidecars"]],
    }


def _fault_reached(fault_id: str, details: Mapping[str, Any]) -> bool:
    if fault_id == "R39-BF01":
        return bool(details.get("destination_matches_selected_source") and not details.get("destination_matches_true_source") and details.get("destination_is_private"))
    if fault_id == "R39-BF03":
        return bool(details.get("page_bytes_restored_exact") and details.get("logical_lengths_restored_exact"))
    if fault_id == "R39-BF04":
        return bool(details.get("post_matches_stale") and not details.get("post_matches_correct") and details.get("private_storage_unchanged"))
    if fault_id == "R39-BF05":
        return bool(details.get("bindings_transposed") and not details.get("bytes_modified_by_swap"))
    if fault_id == "R39-BF06":
        return bool(details.get("cross_family_storage_overlap") and details.get("overlap_outside_terminal_valid_kv") and details.get("logical_gdn_content_preserved_at_injection"))
    if fault_id == "R39-BF07":
        return bool(details.get("digest_changed") and details.get("private_scrub_completed_first"))
    if fault_id == "R39-BF08":
        return bool(details.get("retained_backing_storage_count") == 1 and details.get("selected_bytes_zero") and not details.get("reused_by_live_request"))
    if fault_id == "R39-BF10":
        component = details.get("component_binding", {})
        return bool(component.get("request_0_kv_document_sha256") != component.get("request_0_gdn_document_sha256") and details.get("changed_gdn_coordinate_count") == 60)
    raise ReplayError(f"fault {fault_id} cannot have an executed lane")


def replay_clean_gate(
    *, fault_id: str, reference: Mapping[str, Any], clean: Mapping[str, Any],
    reference_root: Path, clean_root: Path, feasibility_sha256: str,
) -> dict[str, Any]:
    ref = verify_lane(case=reference, lane="reference", root=reference_root, fault_id=fault_id, expected_feasibility_sha256=feasibility_sha256)
    candidate = verify_lane(case=clean, lane="clean", root=clean_root, fault_id=fault_id, expected_feasibility_sha256=feasibility_sha256)
    output_exact = ref["tokens"] == candidate["tokens"] and [row["sha256"] for row in ref["sidecars"]] == [row["sha256"] for row in candidate["sidecars"]]
    persistent_ref = _persistent_invariant(ref["persistent"], require_h7=contract.EXPECTED_HORIZON[fault_id] == "H7")
    persistent_clean = _persistent_invariant(candidate["persistent"], require_h7=contract.EXPECTED_HORIZON[fault_id] == "H7")
    setup_exact = ref["persistent"]["H1"] == candidate["persistent"]["H1"]
    allocator = _allocation_comparison(ref["allocator"], candidate["allocator"])
    passed = (
        output_exact and setup_exact and persistent_ref["passed"] and persistent_clean["passed"]
        and allocator["passed"] and ref["trace"]["verdict"] == "pass"
        and candidate["trace"]["verdict"] == "pass"
    )
    return {
        "schema_version": "forkaudit-r39-detached-clean-gate-v1",
        "run_id": contract.RUN_ID,
        "fault_id": fault_id,
        "status": "clean_gate_passed" if passed else "clean_gate_failed",
        "mutant_authorized": passed,
        "independent_reference_clean_output_exact": output_exact,
        "independent_reference_clean_persistent_setup_exact": setup_exact,
        "reference_persistent_base_invariant": persistent_ref,
        "clean_persistent_base_invariant": persistent_clean,
        "reference_clean_allocation_exact": allocator,
        "reference_forkaudit_pass": ref["trace"]["verdict"] == "pass",
        "clean_forkaudit_pass": candidate["trace"]["verdict"] == "pass",
        "candidate_model_modules_imported": False,
    }


def replay_pair(
    *, fault_id: str, reference: Mapping[str, Any], clean: Mapping[str, Any],
    mutant: Mapping[str, Any], reference_root: Path, clean_root: Path,
    mutant_root: Path, feasibility_sha256: str,
) -> dict[str, Any]:
    clean_gate = replay_clean_gate(
        fault_id=fault_id, reference=reference, clean=clean,
        reference_root=reference_root, clean_root=clean_root,
        feasibility_sha256=feasibility_sha256,
    )
    require(clean_gate["mutant_authorized"] is True, "mutant lacks passing clean gate")
    clean_row = verify_lane(case=clean, lane="clean", root=clean_root, fault_id=fault_id, expected_feasibility_sha256=feasibility_sha256)
    mutant_row = verify_lane(case=mutant, lane="mutant", root=mutant_root, fault_id=fault_id, expected_feasibility_sha256=feasibility_sha256)
    reached = _fault_reached(fault_id, mutant_row["injection"]["details"])
    output = _output_comparison(clean_row, mutant_row)
    persistent_clean = _persistent_invariant(clean_row["persistent"], require_h7=contract.EXPECTED_HORIZON[fault_id] == "H7")
    persistent_mutant = _persistent_invariant(mutant_row["persistent"], require_h7=contract.EXPECTED_HORIZON[fault_id] == "H7")
    matched_setup_exact = clean_row["persistent"]["H1"] == mutant_row["persistent"]["H1"]
    allocation = _allocation_comparison(clean_row["allocator"], mutant_row["allocator"])
    forkaudit_failed = mutant_row["trace"]["verdict"] == "fail"
    valid = reached and matched_setup_exact and persistent_clean["passed"] and clean_gate["mutant_authorized"]
    return {
        "schema_version": "forkaudit-r39-detached-four-observer-pair-v1",
        "run_id": contract.RUN_ID,
        "fault_id": fault_id,
        "status": "valid_reached" if valid else "invalid",
        "valid_pair": valid,
        "fault_reached": reached,
        "matched_clean_mutant_persistent_setup_exact": matched_setup_exact,
        "clean_gate": clean_gate,
        "observers": {
            "output_equality": output,
            "persistent_base_invariant": {
                "clean": persistent_clean,
                "mutant": persistent_mutant,
                "detected": persistent_clean["passed"] and not persistent_mutant["passed"],
            },
            "allocation_assertions": {
                **allocation,
                "detected": not allocation["passed"],
            },
            "forkaudit": {
                "clean_verdict": clean_row["trace"]["verdict"],
                "mutant_verdict": mutant_row["trace"]["verdict"],
                "detected": forkaudit_failed,
                "first_failed_predicate": mutant_row["trace"].get("first_failed_predicate"),
                "first_failed_gate_id": mutant_row["trace"].get("first_failed_gate_id"),
                "compiled_binary_identity_coverage": "partial",
                "autotuning_choice_coverage": "partial",
            },
        },
        "negative_or_escape_retained": True,
        "population_detection_rate_computed": False,
        "candidate_model_modules_imported": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("clean-gate", "pair"), required=True)
    parser.add_argument("--fault-id", choices=contract.FAULT_IDS, required=True)
    parser.add_argument("--reference-case", type=Path, required=True)
    parser.add_argument("--clean-case", type=Path, required=True)
    parser.add_argument("--mutant-case", type=Path)
    parser.add_argument("--feasibility-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = _read(args.reference_case)
    clean = _read(args.clean_case)
    if args.mode == "clean-gate":
        value = replay_clean_gate(
            fault_id=args.fault_id, reference=reference, clean=clean,
            reference_root=args.reference_case.parent,
            clean_root=args.clean_case.parent,
            feasibility_sha256=args.feasibility_sha256,
        )
    else:
        require(args.mutant_case is not None, "pair replay needs mutant")
        mutant = _read(args.mutant_case)
        value = replay_pair(
            fault_id=args.fault_id, reference=reference, clean=clean,
            mutant=mutant, reference_root=args.reference_case.parent,
            clean_root=args.clean_case.parent,
            mutant_root=args.mutant_case.parent,
            feasibility_sha256=args.feasibility_sha256,
        )
    forbidden = sorted(
        name for name in sys.modules
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN_MODULE_PREFIXES)
    )
    require(not forbidden, f"detached replay imported candidate modules: {forbidden}")
    contract.atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
