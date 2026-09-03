from __future__ import annotations

"""Fail-closed verifier for the preregistered R39 dual-producer repeat.

The two R33 results are audited independently against the same census before
this module compares them.  Receiver-local storage/view tokens are deliberately
not compared across processes; the relation labels independently reconstructed
from those tokens are compared exactly.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from audit_independent_slot_census import (
    audit_result,
    canonical_bytes,
    derive_expected_census,
    relation_vector,
    sha256_file,
    sha256_json,
)


PREREGISTRATION_SCHEMA = "forkaudit-r39-dual-producer-preregistration-v1"
SUMMARY_SCHEMA = "forkaudit-r39-dual-producer-repeat-summary-v1"
SEMANTIC_FIELDS = (
    "owner_kind",
    "request_index",
    "layer_index",
    "state_family",
    "state_index",
)
STABLE_DESCRIPTOR_FIELDS = (
    "shape",
    "stride",
    "storage_offset",
    "dtype",
    "device",
    "storage_nbytes",
    "tensor_nbytes",
    "byte_start",
    "byte_end_exclusive",
)


class DualRepeatFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise DualRepeatFailure(code, message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _row_map(capture: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = capture.get("rows")
    require(isinstance(rows, list), "rows_missing", "capture rows missing")
    by_id = {str(row.get("slot_id")): row for row in rows}
    require(len(by_id) == len(rows), "duplicate_slot_id", "capture duplicates a slot id")
    return by_id


def _cell_map(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cells = result.get("cells")
    require(isinstance(cells, list), "cells_missing", "producer cells missing")
    by_policy = {str(cell.get("policy")): cell for cell in cells}
    require(len(by_policy) == len(cells), "duplicate_policy", "duplicate policy cell")
    return by_policy


def _capture_map(cell: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    captures = cell.get("captures")
    require(isinstance(captures, list), "captures_missing", "producer captures missing")
    by_id = {str(capture.get("capture_id")): capture for capture in captures}
    require(len(by_id) == len(captures), "duplicate_capture", "duplicate capture id")
    return by_id


def _process_receipt(result: Mapping[str, Any], label: str) -> dict[str, Any]:
    producer_pids: set[int] = set()
    observer_pids: set[int] = set()
    commitments: set[str] = set()
    per_policy: list[dict[str, Any]] = []
    for policy, cell in sorted(_cell_map(result).items()):
        captures = list(_capture_map(cell).values())
        cell_producers = {int(row["producer_pid"]) for row in captures}
        cell_observers = {int(row["observer_pid"]) for row in captures}
        cell_commitments = {
            str(row["observer_session_commitment_sha256"]) for row in captures
        }
        require(
            len(cell_producers) == 1,
            "producer_pid_drift",
            f"{label}/{policy} does not have one producer PID",
        )
        require(
            len(cell_observers) == 1,
            "observer_session_drift",
            f"{label}/{policy} does not have one observer PID",
        )
        require(
            len(cell_commitments) == 1,
            "observer_session_drift",
            f"{label}/{policy} does not have one observer commitment",
        )
        producer_pid = next(iter(cell_producers))
        observer_pid = next(iter(cell_observers))
        require(
            producer_pid != observer_pid,
            "process_separation",
            f"{label}/{policy} observer equals producer",
        )
        require(
            all(row.get("process_separated") is True for row in captures),
            "process_separation",
            f"{label}/{policy} lacks process separation",
        )
        require(
            all(row.get("transport") == "torch-cuda-ipc-reduction" for row in captures),
            "transport_drift",
            f"{label}/{policy} is not CUDA-IPC capture",
        )
        producer_pids.update(cell_producers)
        observer_pids.update(cell_observers)
        commitments.update(cell_commitments)
        per_policy.append(
            {
                "policy": policy,
                "producer_pid": producer_pid,
                "observer_pid": observer_pid,
                "observer_session_commitment_sha256": next(iter(cell_commitments)),
            }
        )
    require(
        len(producer_pids) == 1,
        "producer_pid_drift",
        f"{label} policy cells did not share one top-level producer PID",
    )
    require(
        len(observer_pids) == 2 and len(commitments) == 2,
        "observer_process_count",
        f"{label} did not use two independent policy-cell receiver sessions",
    )
    return {
        "label": label,
        "producer_pid": next(iter(producer_pids)),
        "observer_pids": sorted(observer_pids),
        "observer_session_commitments_sha256": sorted(commitments),
        "per_policy": per_policy,
    }


def _validate_census(
    census: Mapping[str, Any],
    census_receipt: Mapping[str, Any],
    protocol: Mapping[str, Any],
    *,
    protocol_raw_sha256: str,
    census_file_sha256: str,
    source_ledger_raw_sha256: str,
) -> tuple[dict[str, Mapping[str, Any]], str]:
    derived = derive_expected_census(protocol)
    expected_rows = [derived[slot_id] for slot_id in sorted(derived)]
    observed_rows = census.get("slots")
    require(observed_rows == expected_rows, "census_semantic_drift", "frozen census differs from independent derivation")
    semantic_sha = sha256_json(expected_rows)
    require(census.get("census_semantic_sha256") == semantic_sha, "census_digest", "census semantic digest drift")
    require(census.get("protocol_raw_sha256") == protocol_raw_sha256, "census_protocol_binding", "census/protocol binding drift")
    require(census.get("producer_manifest_used") is False, "census_independence", "census used producer manifest")
    require(census.get("producer_rows_used") is False, "census_independence", "census used producer rows")
    require(census_receipt.get("status") == "frozen_before_fresh_h20_producer_start", "census_timing", "census receipt timing drift")
    require(census_receipt.get("producer_started") is False, "census_timing", "census was not frozen before producers")
    require(census_receipt.get("producer_manifest_available_to_derivation") is False, "census_independence", "producer manifest was available to census")
    require(census_receipt.get("producer_rows_available_to_derivation") is False, "census_independence", "producer rows were available to census")
    require(census_receipt.get("protocol_raw_sha256") == protocol_raw_sha256, "census_protocol_binding", "receipt/protocol binding drift")
    require(census_receipt.get("source_ledger_raw_sha256") == source_ledger_raw_sha256, "census_source_binding", "receipt/source-ledger binding drift")
    require(census_receipt.get("census_file_sha256") == census_file_sha256, "census_file_binding", "receipt/census-file binding drift")
    require(census_receipt.get("census_semantic_sha256") == semantic_sha, "census_digest", "receipt/census semantic digest drift")
    return derived, semantic_sha


def _validate_upstream_bindings(
    result: Mapping[str, Any], preregistration: Mapping[str, Any], label: str
) -> None:
    frozen = preregistration["upstream_bindings"]
    exact_top_level = {
        "preregistration_sha256": frozen["r33_preregistration_raw_sha256"],
        "source_ledger_raw_sha256": frozen["r33_source_ledger_raw_sha256"],
        "candidate_runtime_code_ledger_raw_sha256": frozen[
            "candidate_runtime_code_ledger_raw_sha256"
        ],
    }
    for field, expected in exact_top_level.items():
        require(
            result.get(field) == expected,
            "upstream_binding_drift",
            f"{label} {field} differs from preregistration",
        )
    require(
        result.get("input_receipt") == frozen["input_receipt"],
        "input_binding_drift",
        f"{label} input receipt differs from preregistration",
    )
    require(
        result.get("source_sha256") == frozen["r33_source_sha256"],
        "upstream_binding_drift",
        f"{label} R33 source binding differs from preregistration",
    )


def _validate_individual_artifacts(
    result: Mapping[str, Any],
    replay: Mapping[str, Any],
    audit: Mapping[str, Any],
    protocol: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    *,
    label: str,
    raw_file_sha256: str,
    census_semantic_sha256: str,
    protocol_raw_sha256: str,
) -> None:
    _validate_upstream_bindings(result, preregistration, label)
    independent = audit_result(result, protocol)
    require(independent.get("passed") is True, "individual_census_audit", f"{label} independent census audit failed")
    require(audit.get("passed") is True, "individual_census_audit", f"{label} archived census audit did not pass")
    require(audit.get("input_raw_sha256") == raw_file_sha256, "individual_audit_binding", f"{label} audit/raw hash drift")
    require(audit.get("protocol_raw_sha256") == protocol_raw_sha256, "individual_audit_binding", f"{label} audit/protocol hash drift")
    require(audit.get("expected_census_sha256") == census_semantic_sha256, "individual_audit_binding", f"{label} audit/census hash drift")
    require(audit.get("preexecution_census_bound") is True, "individual_audit_binding", f"{label} audit lacks preexecution census binding")
    require(audit.get("producer_manifest_used_as_expectation") is False, "individual_audit_independence", f"{label} audit used producer manifest as expectation")
    require(audit.get("producer_rows_used_as_expected_census") is False, "individual_audit_independence", f"{label} audit used producer rows as expectation")
    require(audit.get("audited_row_observations") == 1080, "individual_count", f"{label} row count drift")
    require(audit.get("audited_relation_observations") == 96660, "individual_count", f"{label} relation count drift")
    require(replay.get("passed") is True, "individual_replay", f"{label} R33 lifecycle replay failed")
    require(replay.get("input_result_sha256") == sha256_json(result), "individual_replay_binding", f"{label} replay/raw semantic hash drift")
    require(replay.get("cell_count") == 2, "individual_replay_count", f"{label} replay policy count drift")
    require(replay.get("row_observations") == 1080, "individual_replay_count", f"{label} replay row count drift")
    require(replay.get("relation_observations") == 96660, "individual_replay_count", f"{label} replay relation count drift")


def verify_dual_repeat(
    *,
    preregistration: Mapping[str, Any],
    protocol: Mapping[str, Any],
    census: Mapping[str, Any],
    census_receipt: Mapping[str, Any],
    producer_a: Mapping[str, Any],
    producer_b: Mapping[str, Any],
    producer_a_replay: Mapping[str, Any],
    producer_b_replay: Mapping[str, Any],
    producer_a_audit: Mapping[str, Any],
    producer_b_audit: Mapping[str, Any],
    file_hashes: Mapping[str, str],
) -> dict[str, Any]:
    require(preregistration.get("schema_version") == PREREGISTRATION_SCHEMA, "preregistration_schema", "dual-repeat preregistration schema drift")
    design = preregistration.get("frozen_design")
    require(isinstance(design, dict), "preregistration_design", "frozen design missing")
    equality = design.get("cross_producer_equality")
    require(isinstance(equality, dict), "preregistration_design", "equality contract missing")
    require(equality.get("content_digest") == "sha256-byte-exact", "content_policy", "content equality is not byte exact")
    require(equality.get("tolerance") == 0 and equality.get("fallback") == "none", "content_policy", "post-hoc tolerance/fallback is not forbidden")
    require(int(design.get("producer_processes", 0)) == 2, "preregistration_design", "producer count drift")
    require(design.get("execution_order") == "serial-on-one-selected-gpu", "preregistration_design", "execution order drift")

    expected_protocol_sha = preregistration["upstream_bindings"]["r39_slot_protocol_raw_sha256"]
    require(file_hashes["protocol"] == expected_protocol_sha, "protocol_binding", "slot protocol differs from preregistration")
    expected, census_semantic_sha = _validate_census(
        census,
        census_receipt,
        protocol,
        protocol_raw_sha256=file_hashes["protocol"],
        census_file_sha256=file_hashes["census"],
        source_ledger_raw_sha256=file_hashes["source_ledger"],
    )
    _validate_individual_artifacts(
        producer_a,
        producer_a_replay,
        producer_a_audit,
        protocol,
        preregistration,
        label="producer-a",
        raw_file_sha256=file_hashes["producer_a"],
        census_semantic_sha256=census_semantic_sha,
        protocol_raw_sha256=file_hashes["protocol"],
    )
    _validate_individual_artifacts(
        producer_b,
        producer_b_replay,
        producer_b_audit,
        protocol,
        preregistration,
        label="producer-b",
        raw_file_sha256=file_hashes["producer_b"],
        census_semantic_sha256=census_semantic_sha,
        protocol_raw_sha256=file_hashes["protocol"],
    )

    require(producer_a.get("input_receipt") == producer_b.get("input_receipt"), "cross_producer_input_binding", "producer input receipts differ")
    require(producer_a.get("source_sha256") == producer_b.get("source_sha256"), "cross_producer_source_binding", "producer source receipts differ")
    require(producer_a.get("hardware") == producer_b.get("hardware"), "cross_producer_stack_binding", "producer hardware/runtime receipts differ")
    require(file_hashes["producer_a"] != file_hashes["producer_b"], "fresh_process_evidence", "raw producer artifacts are byte-identical")

    receipt_a = _process_receipt(producer_a, "producer-a")
    receipt_b = _process_receipt(producer_b, "producer-b")
    require(receipt_a["producer_pid"] != receipt_b["producer_pid"], "fresh_producer_process", "producer PIDs are not distinct")
    require(set(receipt_a["observer_pids"]).isdisjoint(receipt_b["observer_pids"]), "fresh_observer_process", "observer PID sets overlap across repeats")
    require(set(receipt_a["observer_session_commitments_sha256"]).isdisjoint(receipt_b["observer_session_commitments_sha256"]), "fresh_observer_session", "observer session commitments overlap across repeats")

    policies = list(protocol["schedule"]["policy_cells"])
    capture_ids = [row["capture_id"] for row in protocol["schedule"]["captures"]]
    cells_a = _cell_map(producer_a)
    cells_b = _cell_map(producer_b)
    require(set(cells_a) == set(cells_b) == set(policies), "cross_producer_policy_set", "producer policy sets differ")

    semantic_matches = 0
    content_matches = 0
    descriptor_matches = 0
    relation_matches = 0
    relation_vector_matches = 0
    capture_reports: list[dict[str, Any]] = []
    for policy in policies:
        captures_a = _capture_map(cells_a[policy])
        captures_b = _capture_map(cells_b[policy])
        require(set(captures_a) == set(captures_b) == set(capture_ids), "cross_producer_capture_set", f"{policy} capture sets differ")
        for capture_id in capture_ids:
            capture_a = captures_a[capture_id]
            capture_b = captures_b[capture_id]
            rows_a = _row_map(capture_a)
            rows_b = _row_map(capture_b)
            require(set(rows_a) == set(rows_b) == set(expected), "cross_producer_slot_set", f"{policy}/{capture_id} slot sets differ")
            for slot_id in sorted(expected):
                left = rows_a[slot_id]
                right = rows_b[slot_id]
                expected_semantic = tuple(expected[slot_id][field] for field in SEMANTIC_FIELDS)
                left_semantic = tuple(left[field] for field in SEMANTIC_FIELDS)
                right_semantic = tuple(right[field] for field in SEMANTIC_FIELDS)
                require(left_semantic == right_semantic == expected_semantic, "cross_producer_semantic_mismatch", f"{policy}/{capture_id}/{slot_id} semantic coordinate differs")
                semantic_matches += 1
                require(left["content_sha256"] == right["content_sha256"], "cross_producer_content_mismatch", f"{policy}/{capture_id}/{slot_id} content digest differs")
                content_matches += 1
                left_descriptor = tuple(left[field] for field in STABLE_DESCRIPTOR_FIELDS)
                right_descriptor = tuple(right[field] for field in STABLE_DESCRIPTOR_FIELDS)
                require(left_descriptor == right_descriptor, "cross_producer_descriptor_mismatch", f"{policy}/{capture_id}/{slot_id} stable descriptor differs")
                descriptor_matches += 1
            vector_a = relation_vector(list(rows_a.values()))
            vector_b = relation_vector(list(rows_b.values()))
            require(len(vector_a) == len(vector_b), "cross_producer_relation_count", f"{policy}/{capture_id} relation counts differ")
            for index, (left, right) in enumerate(zip(vector_a, vector_b)):
                require(left == right, "cross_producer_relation_mismatch", f"{policy}/{capture_id} relation {index} differs")
                relation_matches += 1
            relation_vector_matches += 1
            capture_reports.append(
                {
                    "policy": policy,
                    "capture_id": capture_id,
                    "slot_count": len(rows_a),
                    "relation_count": len(vector_a),
                    "producer_a_relation_vector_sha256": sha256_json(vector_a),
                    "producer_b_relation_vector_sha256": sha256_json(vector_b),
                    "relation_vectors_exact": True,
                }
            )

    counts = design["expected_counts"]
    require(semantic_matches == int(counts["semantic_coordinates_per_producer"]), "total_semantic_count", "semantic match count drift")
    require(content_matches == int(counts["content_digests_per_producer"]), "total_content_count", "content match count drift")
    require(descriptor_matches == int(counts["descriptor_rows_per_producer"]), "total_descriptor_count", "descriptor match count drift")
    require(relation_matches == int(counts["relations_per_producer"]), "total_relation_count", "relation match count drift")
    require(relation_vector_matches == int(counts["captures_per_producer"]), "total_capture_count", "relation-vector count drift")

    return {
        "schema_version": SUMMARY_SCHEMA,
        "passed": True,
        "experiment_id": preregistration["experiment_id"],
        "preregistration_raw_sha256": file_hashes["preregistration"],
        "source_ledger_raw_sha256": file_hashes["source_ledger"],
        "slot_protocol_raw_sha256": file_hashes["protocol"],
        "preexecution_census_file_sha256": file_hashes["census"],
        "preexecution_census_semantic_sha256": census_semantic_sha,
        "producer_a_raw_sha256": file_hashes["producer_a"],
        "producer_b_raw_sha256": file_hashes["producer_b"],
        "producer_a_audit_sha256": file_hashes["producer_a_audit"],
        "producer_b_audit_sha256": file_hashes["producer_b_audit"],
        "producer_a_replay_sha256": file_hashes["producer_a_replay"],
        "producer_b_replay_sha256": file_hashes["producer_b_replay"],
        "preexecution_census_frozen_before_both_producers": True,
        "producer_processes_distinct": True,
        "receiver_process_sets_distinct": True,
        "receiver_session_commitments_distinct": True,
        "producer_receipts": [receipt_a, receipt_b],
        "policy_cells_per_producer": 2,
        "captures_per_producer": relation_vector_matches,
        "matched_semantic_coordinates": semantic_matches,
        "matched_content_digests": content_matches,
        "matched_stable_descriptors": descriptor_matches,
        "matched_relation_labels": relation_matches,
        "exact_content_policy": "sha256-byte-exact",
        "numeric_tolerance": 0,
        "canonical_semantic_fallback": False,
        "receiver_local_tokens_compared_across_processes": False,
        "capture_reports": capture_reports,
        "supported_claim": preregistration["claim_boundary"]["supported_if_passed"],
        "unsupported_claims": preregistration["claim_boundary"]["not_supported"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--slot-protocol", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--census-receipt", type=Path, required=True)
    parser.add_argument("--producer-a", type=Path, required=True)
    parser.add_argument("--producer-b", type=Path, required=True)
    parser.add_argument("--producer-a-replay", type=Path, required=True)
    parser.add_argument("--producer-b-replay", type=Path, required=True)
    parser.add_argument("--producer-a-audit", type=Path, required=True)
    parser.add_argument("--producer-b-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = {
        "preregistration": args.preregistration,
        "protocol": args.slot_protocol,
        "source_ledger": args.source_ledger,
        "census": args.census,
        "producer_a": args.producer_a,
        "producer_b": args.producer_b,
        "producer_a_audit": args.producer_a_audit,
        "producer_b_audit": args.producer_b_audit,
        "producer_a_replay": args.producer_a_replay,
        "producer_b_replay": args.producer_b_replay,
    }
    summary = verify_dual_repeat(
        preregistration=load_json(args.preregistration),
        protocol=load_json(args.slot_protocol),
        census=load_json(args.census),
        census_receipt=load_json(args.census_receipt),
        producer_a=load_json(args.producer_a),
        producer_b=load_json(args.producer_b),
        producer_a_replay=load_json(args.producer_a_replay),
        producer_b_replay=load_json(args.producer_b_replay),
        producer_a_audit=load_json(args.producer_a_audit),
        producer_b_audit=load_json(args.producer_b_audit),
        file_hashes={name: sha256_file(path) for name, path in paths.items()},
    )
    write_json(args.output, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
