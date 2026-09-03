from __future__ import annotations

"""Candidate-import-free replay of the five frozen R33 primary predicates."""

import hashlib
import json
import argparse
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


FAULT_IDS = (
    "HF01_DELAYED_TAIL_DETACH",
    "HF02_INACTIVE_DOCUMENT_LANE_SCRIBBLE",
    "HF03_DUPLICATE_COMMITTED_DISPATCH",
    "HF04_EFFECTIVE_SCALE_DRIFT",
    "HF05_STALE_GDN_BINDING_TOKEN_AFTER_REBIND",
)

EXPECTED_PRIMARY_GATES = {
    FAULT_IDS[0]: "TAIL_COPY_BEFORE_FIRST_APPEND_WRITE",
    FAULT_IDS[1]: "PHYSICAL_DOCUMENT_PREFIX_IMMUTABLE",
    FAULT_IDS[2]: "ORDERED_CALL_CARDINALITY",
    FAULT_IDS[3]: "ATTENTION_EFFECTIVE_SCALE",
    FAULT_IDS[4]: "GDN_COMPLETED_BINDING_TOKEN_ADVANCE",
}

FAULT_POLICIES = {
    FAULT_IDS[0]: {
        "rank": 0,
        "kv_policy": "vllm-q16-shared-document-reuse",
        "gdn_policy": "borrow-immutable-base-functional-rebind",
        "target_request": 0,
    },
    FAULT_IDS[1]: {
        "rank": 1,
        "kv_policy": "vllm-q16-shared-document-reuse",
        "gdn_policy": "materialize-request-base-functional-rebind",
        "target_request": None,
    },
    FAULT_IDS[2]: {
        "rank": 2,
        "kv_policy": "vllm-q16-shared-document-reuse",
        "gdn_policy": "borrow-immutable-base-functional-rebind",
        "target_request": 0,
    },
    FAULT_IDS[3]: {
        "rank": 3,
        "kv_policy": "vllm-q16-shared-document-reuse",
        "gdn_policy": "materialize-request-base-functional-rebind",
        "target_request": 0,
    },
    FAULT_IDS[4]: {
        "rank": 4,
        "kv_policy": "vllm-q16-shared-document-reuse",
        "gdn_policy": "borrow-immutable-base-functional-rebind",
        "target_request": 1,
    },
}

PREDICATE_PREFIX = (
    "FROZEN_INPUT_AND_SOURCE_BINDING",
    "MANDATORY_RECORD_COVERAGE",
    "BYTE_BINDING",
    "REQUEST_AND_STORAGE_IDENTITY",
    "PYTHON_CALL_SCOPE_DISPATCH",
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class R33ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R33ReplayError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} SHA")
    return value


def validate_sidecar(root: Path, value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} sidecar")
    relative = Path(str(value.get("path", "")))
    require(not relative.is_absolute() and ".." not in relative.parts, f"{label} unsafe sidecar path")
    path = root / relative
    require(path.is_file(), f"{label} sidecar missing")
    payload = path.read_bytes()
    require(len(payload) == value.get("nbytes"), f"{label} sidecar size")
    observed = hashlib.sha256(payload).hexdigest()
    require(observed == _sha(value.get("sha256"), f"{label} sidecar"), f"{label} sidecar digest")
    return {"path": relative.as_posix(), "sha256": observed, "nbytes": len(payload)}


def expected_clean_schedule() -> list[dict[str, Any]]:
    return [
        {
            "event_index": round_index * 2 + request_index,
            "round_index": round_index,
            "request_index": request_index,
            "duplicate_discarded_output": False,
        }
        for round_index in range(8)
        for request_index in range(2)
    ]


def expected_hf03_schedule() -> list[dict[str, Any]]:
    rows = expected_clean_schedule()
    insertion = {
        "event_index": 2,
        "round_index": 1,
        "request_index": 0,
        "duplicate_discarded_output": True,
    }
    # The first execution is committed but its output is discarded; the
    # ordinary scheduled row immediately following it is the surfaced retry.
    rows.insert(2, insertion)
    for index, row in enumerate(rows):
        row["event_index"] = index
    return rows


def _validate_common(case: Any, *, lane: str, fault_id: str, artifact_root: Path) -> dict[str, Any]:
    require(isinstance(case, Mapping), f"{lane} case")
    require(case.get("schema_version") == "forkaudit-r33-executed-case-v1", f"{lane} case schema")
    require(case.get("fault_id") == fault_id and case.get("lane") == lane, f"{lane} case binding")
    require(case.get("status") == "full_horizon_completed", f"{lane} case status")
    require(case.get("operational_invalid") is None, f"{lane} operational invalid")
    require(case.get("all_existing_gates_enabled") is True, f"{lane} gates disabled")
    require(case.get("mandatory_coverage_complete") is True, f"{lane} mandatory coverage")
    require(case.get("byte_binding_passed") is True, f"{lane} byte binding flag")
    require(
        case.get("dispatch_scope")
        == {
            "python_call_scope": "full",
            "compiled_binary_identity": "partial",
            "autotuning_choice": "partial",
        },
        f"{lane} dispatch scope",
    )
    policy = FAULT_POLICIES[fault_id]
    require(case.get("rank") == policy["rank"], f"{lane} rank")
    require(case.get("kv_policy") == policy["kv_policy"], f"{lane} KV policy")
    require(case.get("gdn_policy") == policy["gdn_policy"], f"{lane} GDN policy")
    source = case.get("source_physical_digests")
    require(isinstance(source, Mapping) and set(source) == {"setup", "transition", "final"}, f"{lane} source phases")
    for phase in source.values():
        require(isinstance(phase, Mapping) and len(phase) == 10, f"{lane} source layer coverage")
        for digest in phase.values():
            _sha(digest, f"{lane} source digest")
    binding = case.get("gdn_binding_witness")
    require(isinstance(binding, Mapping), f"{lane} GDN binding witness")
    rows = binding.get("rows")
    require(isinstance(rows, list) and len(rows) == 120, f"{lane} GDN binding rows")
    require(binding.get("rows_sha256") == sha256_json(rows), f"{lane} GDN binding row digest")
    for row in rows:
        require(row.get("expected_relation") == "rebound", f"{lane} completed relation")
        for field in (
            "baseline_binding_token",
            "observed_binding_token",
            "baseline_storage_token",
            "observed_storage_token",
        ):
            _sha(row.get(field), f"{lane} {field}")
        require(row["baseline_storage_token"] != row["observed_storage_token"], f"{lane} storage token did not advance")
    sidecars = case.get("logit_sidecars")
    require(isinstance(sidecars, list) and len(sidecars) in (16, 17), f"{lane} logit sidecar count")
    sidecar_rows = [validate_sidecar(artifact_root, row, f"{lane} logit {index}") for index, row in enumerate(sidecars)]
    require(len({row["path"] for row in sidecar_rows}) == len(sidecar_rows), f"{lane} duplicate sidecar path")
    cleanup = case.get("cleanup")
    require(isinstance(cleanup, Mapping), f"{lane} cleanup")
    require(cleanup.get("completed") is True, f"{lane} cleanup completion")
    require(cleanup.get("registered_backend_restored") is True, f"{lane} backend cleanup")
    require(cleanup.get("strong_references_released") is True, f"{lane} references")
    require(cleanup.get("gc_collect_completed") is True, f"{lane} GC")
    require(cleanup.get("accelerator_cache_cleanup_completed") is True, f"{lane} accelerator cleanup")
    require(cleanup.get("accelerator_synchronize_completed") is True, f"{lane} accelerator sync")
    require(cleanup.get("allocator_baseline_exact") is True and cleanup.get("cleanup_error") is None, f"{lane} allocator cleanup")
    return {"source": source, "binding_rows": rows, "sidecars": sidecar_rows}


def _clean_primary_pass(fault_id: str, clean: Mapping[str, Any]) -> bool:
    evidence = clean.get("fault_specific_evidence")
    require(isinstance(evidence, Mapping), "clean fault-specific evidence")
    if fault_id == FAULT_IDS[0]:
        events = evidence.get("ordered_tail_events")
        require(isinstance(events, list) and bool(events), "HF01 clean ordered events")
        copies = [row["ordinal"] for row in events if row.get("kind") == "tail_copy"]
        writes = [row["ordinal"] for row in events if row.get("kind") == "append_write"]
        require(len(copies) == 1 and bool(writes), "HF01 clean event coverage")
        return copies[0] < min(writes)
    if fault_id == FAULT_IDS[1]:
        source = clean["source_physical_digests"]
        return source["setup"] == source["transition"] == source["final"]
    if fault_id == FAULT_IDS[2]:
        return clean.get("ordered_model_schedule") == expected_clean_schedule()
    if fault_id == FAULT_IDS[3]:
        target = evidence.get("target_call")
        require(isinstance(target, Mapping), "HF04 clean target call")
        return target.get("observed_scale_hex") == target.get("frozen_scale_hex")
    if fault_id == FAULT_IDS[4]:
        return all(row["baseline_binding_token"] != row["observed_binding_token"] for row in clean["gdn_binding_witness"]["rows"])
    raise R33ReplayError("unknown fault")


def _mutant_primary_fail(fault_id: str, mutant: Mapping[str, Any]) -> bool:
    evidence = mutant.get("fault_specific_evidence")
    require(isinstance(evidence, Mapping), "mutant fault-specific evidence")
    if fault_id == FAULT_IDS[0]:
        events = evidence.get("ordered_tail_events")
        require(isinstance(events, list) and bool(events), "HF01 mutant ordered events")
        copies = [row["ordinal"] for row in events if row.get("kind") == "tail_copy"]
        writes = [row["ordinal"] for row in events if row.get("kind") == "append_write"]
        require(len(copies) == 1 and bool(writes), "HF01 mutant event coverage")
        return min(writes) <= copies[0]
    if fault_id == FAULT_IDS[1]:
        selected_layer = str(evidence.get("layer_index"))
        source = mutant["source_physical_digests"]
        require(selected_layer in source["setup"], "HF02 selected layer")
        return (
            source["setup"][selected_layer] != source["transition"][selected_layer]
            and source["transition"][selected_layer] == source["final"][selected_layer]
        )
    if fault_id == FAULT_IDS[2]:
        return mutant.get("ordered_model_schedule") == expected_hf03_schedule()
    if fault_id == FAULT_IDS[3]:
        target = evidence.get("target_call")
        require(isinstance(target, Mapping), "HF04 mutant target call")
        frozen = float.fromhex(str(target.get("frozen_scale_hex")))
        observed = float.fromhex(str(target.get("observed_scale_hex")))
        return observed == 2.0 * frozen
    if fault_id == FAULT_IDS[4]:
        target_rows = [
            row
            for row in mutant["gdn_binding_witness"]["rows"]
            if row.get("request_index") == 1
        ]
        require(len(target_rows) == 60, "HF05 target row count")
        return all(
            row["baseline_binding_token"] == row["observed_binding_token"]
            and row["baseline_storage_token"] != row["observed_storage_token"]
            for row in target_rows
        )
    raise R33ReplayError("unknown fault")


def replay_pair(
    *,
    fault_id: str,
    clean_case: Mapping[str, Any],
    mutant_case: Mapping[str, Any],
    artifact_root: Path,
    expected_fault_definition_sha256: str,
) -> dict[str, Any]:
    require(fault_id in FAULT_IDS, "unregistered R33 fault")
    clean = _validate_common(clean_case, lane="clean", fault_id=fault_id, artifact_root=artifact_root)
    mutant = _validate_common(mutant_case, lane="mutant", fault_id=fault_id, artifact_root=artifact_root)
    require(clean_case.get("injection_witness") is None, "clean injection present")
    injection = mutant_case.get("injection_witness")
    require(isinstance(injection, Mapping), "mutant injection witness")
    require(injection.get("fault_id") == fault_id, "mutant injection fault binding")
    require(injection.get("fault_definition_sha256") == expected_fault_definition_sha256, "mutant fault-definition binding")
    require(injection.get("mutation_observed") is True, "mutant injection no-op")
    require(injection.get("exactly_one_named_injection") is True, "mutant multiple/ambiguous injection")
    clean_pass = _clean_primary_pass(fault_id, clean_case)
    require(clean_pass, f"{fault_id} matched clean primary predicate failed")
    target_failed = _mutant_primary_fail(fault_id, mutant_case)
    earlier = mutant_case.get("earlier_predicates")
    require(isinstance(earlier, list), "mutant earlier predicate list")
    require([row.get("predicate_id") for row in earlier] == list(PREDICATE_PREFIX), "mutant predicate precedence")
    require(all(row.get("passed") is True for row in earlier), "mutant earlier unrelated predicate failed")
    expected_gate = EXPECTED_PRIMARY_GATES[fault_id]
    classification = (
        "caught_by_expected_primary_gate" if target_failed else "escaped_expected_primary_gate"
    )
    return {
        "schema_version": "forkaudit-r33-detached-pair-replay-v1",
        "fault_id": fault_id,
        "expected_primary_gate": expected_gate,
        "first_failed_predicate": expected_gate if target_failed else None,
        "clean_gate_passed": True,
        "injection_witness_passed": True,
        "earlier_unrelated_predicates_passed": True,
        "target_predicate_failed": target_failed,
        "classification": classification,
        "clean_logit_sidecars_verified": len(clean["sidecars"]),
        "mutant_logit_sidecars_verified": len(mutant["sidecars"]),
        "candidate_modules_imported": False,
        "negative_or_escape_retained": True,
    }


def validate_clean_case(
    *,
    fault_id: str,
    clean_case: Mapping[str, Any],
    artifact_root: Path,
) -> dict[str, Any]:
    """Fail-closed clean gate usable before the mutant is allowed to start."""

    require(fault_id in FAULT_IDS, "unregistered R33 fault")
    common = _validate_common(
        clean_case,
        lane="clean",
        fault_id=fault_id,
        artifact_root=artifact_root,
    )
    require(clean_case.get("injection_witness") is None, "clean injection present")
    require(_clean_primary_pass(fault_id, clean_case), f"{fault_id} clean primary predicate")
    return {
        "schema_version": "forkaudit-r33-detached-clean-gate-v1",
        "fault_id": fault_id,
        "expected_primary_gate": EXPECTED_PRIMARY_GATES[fault_id],
        "status": "clean_gate_passed",
        "clean_logit_sidecars_verified": len(common["sidecars"]),
        "candidate_modules_imported": False,
        "fault_module_loaded": False,
        "mutant_authorized": True,
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("clean", "pair"), required=True)
    parser.add_argument("--fault-id", choices=FAULT_IDS, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--clean-case", type=Path, required=True)
    parser.add_argument("--mutant-case", type=Path)
    parser.add_argument("--fault-definition-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        clean_case = _read_json(args.clean_case)
        if args.mode == "clean":
            require(args.mutant_case is None, "clean mode received mutant case")
            result = validate_clean_case(
                fault_id=args.fault_id,
                clean_case=clean_case,
                artifact_root=args.artifact_root,
            )
        else:
            require(args.mutant_case is not None, "pair mode missing mutant case")
            require(
                isinstance(args.fault_definition_sha256, str),
                "pair mode missing fault-definition SHA",
            )
            result = replay_pair(
                fault_id=args.fault_id,
                clean_case=clean_case,
                mutant_case=_read_json(args.mutant_case),
                artifact_root=args.artifact_root,
                expected_fault_definition_sha256=args.fault_definition_sha256,
            )
        _write_json(args.output, result)
        return 0
    except BaseException as exc:
        error = {
            "schema_version": "forkaudit-r33-detached-replay-error-v1",
            "status": "operational_invalid",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "candidate_modules_imported": False,
        }
        _write_json(args.output, error)
        print(json.dumps(error, sort_keys=True), file=sys.stderr)
        return 2


__all__ = [
    "EXPECTED_PRIMARY_GATES",
    "FAULT_IDS",
    "FAULT_POLICIES",
    "PREDICATE_PREFIX",
    "R33ReplayError",
    "expected_clean_schedule",
    "expected_hf03_schedule",
    "replay_pair",
    "sha256_json",
    "validate_clean_case",
    "validate_sidecar",
]


if __name__ == "__main__":
    raise SystemExit(main())
