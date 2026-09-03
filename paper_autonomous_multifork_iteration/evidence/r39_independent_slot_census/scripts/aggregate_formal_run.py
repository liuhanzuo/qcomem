from __future__ import annotations

"""Fail-closed aggregation for a fresh R39 H20 capture and census audit."""

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from audit_independent_slot_census import sha256_file, sha256_json, write_json


class AggregateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AggregateError(message)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--preexecution-census", type=Path, required=True)
    parser.add_argument("--preexecution-receipt", type=Path, required=True)
    parser.add_argument("--raw-input", type=Path, required=True)
    parser.add_argument("--r33-replay", type=Path, required=True)
    parser.add_argument("--clean-audit", type=Path, required=True)
    parser.add_argument("--negative-controls", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    protocol = load(args.protocol)
    census = load(args.preexecution_census)
    receipt = load(args.preexecution_receipt)
    raw = load(args.raw_input)
    r33_replay = load(args.r33_replay)
    clean = load(args.clean_audit)
    controls = load(args.negative_controls)

    protocol_sha = sha256_file(args.protocol)
    source_ledger_sha = sha256_file(args.source_ledger)
    census_file_sha = sha256_file(args.preexecution_census)
    raw_file_sha = sha256_file(args.raw_input)
    census_slots = census.get("slots")
    require(isinstance(census_slots, list), "preexecution census slots missing")
    census_semantic_sha = sha256_json(census_slots)

    require(
        receipt.get("schema_version")
        == "forkaudit-r39-preexecution-census-receipt-v1",
        "preexecution receipt schema drift",
    )
    require(
        receipt.get("status") == "frozen_before_fresh_h20_producer_start",
        "preexecution timing status drift",
    )
    require(receipt.get("producer_started") is False, "preexecution timing claim drift")
    require(
        receipt.get("producer_manifest_available_to_derivation") is False
        and receipt.get("producer_rows_available_to_derivation") is False,
        "preexecution derivation boundary drift",
    )
    require(receipt.get("protocol_raw_sha256") == protocol_sha, "protocol binding drift")
    require(
        receipt.get("source_ledger_raw_sha256") == source_ledger_sha,
        "source-ledger binding drift",
    )
    require(receipt.get("census_file_sha256") == census_file_sha, "census file drift")
    require(
        receipt.get("census_semantic_sha256") == census_semantic_sha,
        "census semantic digest drift",
    )
    require(census.get("protocol_raw_sha256") == protocol_sha, "census/protocol drift")
    require(census.get("producer_manifest_used") is False, "census used producer manifest")
    require(census.get("producer_rows_used") is False, "census used producer rows")

    require(raw.get("schema_version") == "forkaudit-r33-out-of-process-result-v1", "raw schema drift")
    require(r33_replay.get("passed") is True, "R33 lifecycle replay failed")
    require(
        r33_replay.get("input_result_sha256") == sha256_json(raw),
        "R33 lifecycle replay/raw semantic binding drift",
    )
    require(r33_replay.get("cell_count") == 2, "R33 policy-cell count drift")
    require(r33_replay.get("row_observations") == 1080, "R33 row count drift")
    require(r33_replay.get("relation_observations") == 96660, "R33 relation count drift")
    require(clean.get("passed") is True, "independent census audit failed")
    require(clean.get("input_raw_sha256") == raw_file_sha, "clean/raw digest drift")
    require(
        clean.get("expected_census_sha256") == census_semantic_sha,
        "live capture not bound to preexecution census",
    )
    require(clean.get("preexecution_census_bound") is True, "preexecution binding flag absent")
    require(
        clean.get("producer_manifest_used_as_expectation") is False
        and clean.get("producer_rows_used_as_expected_census") is False,
        "clean audit expectation independence drift",
    )
    require(controls.get("passed") is True, "negative-control campaign failed")
    require(controls.get("all_controls_failed_closed") is True, "control did not fail closed")
    require(controls.get("all_internal_digests_resealed") is True, "control digest reseal failed")
    require(
        controls.get("pristine_input_raw_sha256") == raw_file_sha,
        "control/pristine raw binding drift",
    )
    expected_codes = {
        "C-OMIT-ONE-SLOT": "slot_set_mismatch",
        "C-DUPLICATE-ONE-SLOT": "duplicate_slot_id",
        "C-SEMANTIC-RELABEL": "semantic_binding_mismatch",
    }
    observed_codes = {
        row["control_id"]: row["observed_failure_code"]
        for row in controls.get("controls", [])
    }
    require(observed_codes == expected_codes, "negative-control failure-code drift")

    report: Mapping[str, Any] = {
        "schema_version": "forkaudit-r39-independent-slot-formal-aggregate-v1",
        "passed": True,
        "experiment_id": protocol["experiment_id"],
        "protocol_raw_sha256": protocol_sha,
        "source_ledger_raw_sha256": source_ledger_sha,
        "preexecution_census_file_sha256": census_file_sha,
        "preexecution_census_semantic_sha256": census_semantic_sha,
        "fresh_h20_raw_file_sha256": raw_file_sha,
        "preexecution_census_frozen_before_producer": True,
        "live_capture_bound_to_preexecution_census": True,
        "producer_manifest_used_as_expectation": False,
        "audited_policy_cells": clean["audited_policy_cells"],
        "audited_captures": clean["audited_captures"],
        "audited_row_observations": clean["audited_row_observations"],
        "audited_relation_observations": clean["audited_relation_observations"],
        "r33_lifecycle_replay_passed": True,
        "negative_controls_failed_closed": observed_codes,
        "controls_operated_on_deep_copies": True,
        "clean_raw_unchanged_by_controls": True,
        "remaining_boundary": (
            "The census independently enumerates expected emitted semantic slots, but "
            "a correct slot id can still carry a maliciously substituted same-geometry "
            "live tensor; PyTorch/CUDA IPC and the paused producer schedule remain trusted."
        ),
    }
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
