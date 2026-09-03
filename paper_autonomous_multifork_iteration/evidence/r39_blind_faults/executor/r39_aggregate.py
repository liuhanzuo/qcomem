#!/usr/bin/env python3
"""Aggregate exactly eleven R39 fault outcomes without pooling detection rates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import r39_contract as contract


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    freeze = contract.verify_freeze(args.protocol, args.plan)
    rows = []
    for fault_id in contract.FAULT_IDS:
        root = args.run_root / fault_id
        outcome_path = root / "outcome.json"
        invalid_path = root / "operational-invalid.json"
        if outcome_path.is_file():
            value = json.loads(outcome_path.read_text(encoding="utf-8"))
            contract.require(value.get("fault_id") == fault_id, f"{fault_id} outcome binding")
            status = value.get("status")
            contract.require(status in {"ineligible_preexecution", "valid_reached", "invalid"}, f"{fault_id} status")
            row = {
                "fault_id": fault_id,
                "fault_row_sha256": freeze["fault_row_sha256"][fault_id],
                "status": status,
                "outcome_path": outcome_path.relative_to(args.run_root).as_posix(),
                "outcome_sha256": contract.sha256_file(outcome_path),
                "eligible": status != "ineligible_preexecution",
                "valid_pair": value.get("valid_pair"),
                "fault_reached": value.get("fault_reached"),
                "ineligible_reason": value.get("ineligible_reason"),
                "observer_outcomes": value.get("observer_outcomes"),
            }
        elif invalid_path.is_file():
            value = json.loads(invalid_path.read_text(encoding="utf-8"))
            contract.require(value.get("fault_id") == fault_id, f"{fault_id} invalid binding")
            row = {
                "fault_id": fault_id,
                "fault_row_sha256": freeze["fault_row_sha256"][fault_id],
                "status": "operational_invalid",
                "outcome_path": invalid_path.relative_to(args.run_root).as_posix(),
                "outcome_sha256": contract.sha256_file(invalid_path),
                "eligible": None,
                "valid_pair": False,
                "fault_reached": False,
                "ineligible_reason": None,
                "observer_outcomes": None,
            }
        else:
            raise contract.ContractError(f"missing terminal outcome for {fault_id}")
        rows.append(row)
    contract.require([row["fault_id"] for row in rows] == list(contract.FAULT_IDS), "aggregate order")
    return {
        "schema_version": "forkaudit-r39-eleven-fault-summary-v1",
        "run_id": contract.RUN_ID,
        "status": "completed_with_all_individual_outcomes",
        "fault_count": 11,
        "rows": rows,
        "campaign_success_rule": {
            "every_fault_reported_individually": True,
            "ineligible_and_invalid_rows_retained": True,
            "all_four_observers_unsuppressed_for_valid_pairs": True,
        },
        "claim_boundary": {
            "fixed_fault_sensitivity_only": True,
            "population_detection_rate_computed": False,
            "recall_or_accuracy_claim_allowed": False,
            "single_model_single_stack": True,
            "escapes_are_contract_boundaries_not_undetectability_proofs": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = aggregate(args)
    contract.atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
