#!/usr/bin/env python3
"""Fail-closed local audit of existing held-out-fault evidence.

Inputs are read-only.  Outputs are created only in this directory and are
never overwritten: a later invocation verifies existing bytes instead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def preserve(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"existing output drift: {path}")
        return
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    expected = load_json(HERE / "frozen_inputs.json")
    input_rows = []
    for relative, expected_sha in expected["files"].items():
        path = PAPER / relative
        require(path.is_file(), f"missing frozen input: {relative}")
        actual = sha256_file(path)
        require(actual == expected_sha, f"frozen input drift: {relative}")
        input_rows.append({"path": relative, "sha256": actual, "nbytes": path.stat().st_size})

    r39_root = PAPER / (
        "evidence/r39_blind_faults/formal_h20/"
        "r39-blind-faults-20260826g-metadata/r39-blind-faults-20260826g"
    )
    ledger_path = r39_root / "terminal-files.sha256"
    ledger: dict[str, str] = {}
    marker = "/r39-blind-faults-20260826g/"
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        digest, absolute = line.split("  ", 1)
        require(marker in absolute, "unexpected R39 terminal-ledger path")
        relative = absolute.split(marker, 1)[1]
        require(relative not in ledger, f"duplicate R39 ledger path: {relative}")
        ledger[relative] = digest
    require(len(ledger) == 605, "R39 terminal-ledger cardinality drift")

    local_files = sorted(
        path for path in r39_root.rglob("*")
        if path.is_file() and path != ledger_path
    )
    for path in local_files:
        relative = path.relative_to(r39_root).as_posix()
        require(relative in ledger, f"locally retained R39 file absent from terminal ledger: {relative}")
        require(sha256_file(path) == ledger[relative], f"R39 local terminal byte drift: {relative}")
    local_ledger_count = sum((r39_root / relative).is_file() for relative in ledger)
    require(len(local_files) == 173 and local_ledger_count == 173, "R39 retained-file count drift")

    executor = PAPER / "evidence/r39_blind_faults/executor"
    sys.path.insert(0, str(executor))
    aggregate_module = importlib.import_module("r39_aggregate")
    args = argparse.Namespace(
        run_root=r39_root,
        protocol=PAPER / "evidence/r39_blind_faults/designer_freeze/PROTOCOL.md",
        plan=PAPER / "evidence/r39_blind_faults/designer_freeze/plan.json",
        output=HERE / "unused.json",
    )
    reaggregated = aggregate_module.aggregate(args)
    archived_summary = load_json(r39_root / "summary.json")
    require(reaggregated == archived_summary, "R39 aggregate object mismatch")
    reaggregated_payload = canonical_bytes(reaggregated)
    require(reaggregated_payload == (r39_root / "summary.json").read_bytes(), "R39 aggregate bytes mismatch")
    preserve(HERE / "r39_reaggregated_summary.json", reaggregated_payload)

    rows = reaggregated["rows"]
    valid = [row for row in rows if row["status"] == "valid_reached" and row["valid_pair"]]
    ineligible = [row for row in rows if row["status"] == "ineligible_preexecution"]
    invalid = [row for row in rows if row["status"] == "operational_invalid"]
    forkaudit = [row["fault_id"] for row in valid if row["observer_outcomes"]["forkaudit"]["detected"]]
    allocator = [row["fault_id"] for row in valid if row["observer_outcomes"]["allocation_assertions"]["detected"]]
    persistent = [row["fault_id"] for row in valid if row["observer_outcomes"]["persistent_base_invariant"]["detected"]]
    output = [
        row["fault_id"] for row in valid
        if not row["observer_outcomes"]["output_equality"]["tokens_exact"]
        or not row["observer_outcomes"]["output_equality"]["complete_fp32_logits_byte_exact"]
    ]
    exposed = sorted(set(forkaudit + allocator + persistent + output))
    escaped = [row["fault_id"] for row in valid if row["fault_id"] not in exposed]
    require(len(valid) == 7 and len(ineligible) == 3 and len(invalid) == 1, "R39 disposition drift")
    require(forkaudit == [], "R39 ForkAudit outcome drift")
    require(output == ["R39-BF01", "R39-BF04", "R39-BF05", "R39-BF10"], "R39 output outcome drift")
    require(allocator == ["R39-BF06", "R39-BF08"], "R39 allocator outcome drift")
    require(persistent == [] and escaped == ["R39-BF03"], "R39 escape outcome drift")

    r35 = load_json(PAPER / "evidence/r35_historical_alias_regression/formal_h20/RESULT_VERIFICATION.json")
    r33 = load_json(PAPER / "evidence/r33_fresh_faults/formal_h20/RESULT_VERIFICATION.json")
    r30 = load_json(PAPER / "evidence/r30_fresh_faults/closure/closure_ledger.json")
    r29 = load_json(PAPER / "evidence/r29_heldout_faults/cross_execution/attempt-c-internal-operational-invalid.json")
    registry = load_json(PAPER / "evidence/experiment_registry.json")
    registry_ids = {row.get("evidence_id") for row in registry["experiments"]}
    manuscript = (PAPER / "main_r39_revised.tex").read_text(encoding="utf-8")

    result = {
        "schema_version": "forkaudit-r40-heldout-evidence-audit-v1",
        "status": "PASS_METADATA_AUDIT__HOLD_POSITIVE_HELDOUT_CLAIM",
        "audit_boundary": {
            "inputs_read_only": True,
            "paper_modified": False,
            "registry_modified": False,
            "gpu_or_remote_accessed": False,
            "full_r39_pair_replay_possible_locally": False,
            "reason_full_pair_replay_unavailable": (
                "Only 173 of 605 terminal-ledger files are retained locally; 432 entries, "
                "including all complete-vocabulary FP32 sidecars and compiled artifacts, are absent."
            ),
            "r39_metadata_aggregate_replay_byte_exact": True,
        },
        "r39_blind_campaign": {
            "faults_frozen": len(rows),
            "valid_reached_pairs": len(valid),
            "preexecution_ineligible": len(ineligible),
            "operational_invalid": len(invalid),
            "forkaudit_replay_rejected_ids": forkaudit,
            "output_or_logit_comparison_exposed_ids": output,
            "allocator_comparison_exposed_ids": allocator,
            "persistent_base_invariant_exposed_ids": persistent,
            "exposed_by_any_of_four_observers": exposed,
            "escaped_all_four_observers": escaped,
            "retained_terminal_files_verified": len(local_files),
            "terminal_ledger_entries": len(ledger),
            "terminal_ledger_entries_not_local": len(ledger) - local_ledger_count,
            "interpretation": (
                "Outcome-held-out fixed-fault boundary evidence for the frozen R39 method. "
                "It is development/audit evidence and does not support a positive ForkAudit held-out-sensitivity claim."
            ),
        },
        "other_fault_evidence": {
            "r28_m1_m9": {
                "classification": "design_set",
                "reason": "The nine mechanisms and their target/gate mappings are designed cases reused by the detector matrix."
            },
            "r23_scheduler_faults": {
                "classification": "design_set_preregistered_expected_gate_controls",
                "reason": "The three scheduler-path faults are frozen expected-gate trials, not unforeseen implementation defects."
            },
            "seeded_attention_and_gdn_wrong_operators": {
                "classification": "oracle_design_controls",
                "reason": "These rows deliberately perturb the numerical operator used to validate oracle sensitivity; they are not held-out system faults."
            },
            "r33_hf01_hf05": {
                "classification": "preregistered_designer_executor_separated_but_contract_targeted",
                "pair_count": r33["summary"]["pair_count"],
                "expected_primary_gate_catches": r33["summary"]["caught_by_expected_primary_gate_count"],
                "reason": "The PDF-only designer froze an expected primary predicate for every fault from the public contract."
            },
            "r35_historical_alias": {
                "classification": "one_organic_postdiscovery_defect_mechanism_retrospectively_reproduced",
                "coordinate_cells": r35["coordinate_cell_counts"]["all_coordinates"]["cell_count"],
                "independent_defect_count": 1,
                "positive_headline_rule_satisfied": r35["headline_outcomes"]["registered_positive_headline_rule_satisfied"],
                "reason": "Eight coordinates exercise one defect mechanism; they are not eight natural bugs."
            },
            "r29_h01_h03": {
                "classification": "operationally_invalid_and_outcomes_contaminated_for_future_holdout_use",
                "scientific_outcome": r29["disposition"]["scientific_outcome"],
                "reason": "The clean gate failed and fault-lane diagnostics were observed; these rows cannot be reused as unseen v2 outcomes."
            },
            "r30_f1_f3": {
                "classification": "outcome_unrun_but_definitions_now_part_of_design_knowledge",
                "faults_run": r30["experimental_outcomes"]["faults_run"],
                "clean_controls_run": 0 if r30["experimental_outcomes"]["clean_control"] == "not_run" else None,
                "reason": "No result exists, but the definitions and executor review are already visible and must not be relabeled as a fresh v2 holdout."
            },
            "r40_binding_substitutions": {
                "classification": "fixed_mechanism_design_tests",
                "reason": "The CPU/live-binding substitutions were preregistered to exercise known binding failure classes and do not form an unseen defect cohort."
            }
        },
        "current_submission_visibility": {
            "r39_registered": "E-R39-BLIND-FAULTS-A" in registry_ids,
            "r39_named_in_manuscript": (
                "R39-BF" in manuscript
                or "r39_blind_faults" in manuscript
                or "R39 blind" in manuscript
            ),
            "current_manuscript_explicitly_disclaims_unseen_fault_recall": "unseen-fault recall" in manuscript,
        },
        "decision": {
            "positive_heldout_evidence_gate": "HOLD",
            "paper_change_authorized_by_this_audit": False,
            "narrow_current_claim": (
                "ForkAudit reproduced and localized one organically encountered borrowed-state alias defect; "
                "the evidence does not estimate unseen-fault recall."
            ),
            "v2_path": (
                "Treat every R39 row and all earlier fault definitions as development data; freeze a revised general contract "
                "and executable hash before a fresh isolated designer sees the revised public contract; retain all new outcomes."
            )
        },
        "frozen_inputs": input_rows,
    }
    preserve(HERE / "audit_result.json", canonical_bytes(result))

    manifest_files = [
        "README.md",
        "V2_PROTOCOL.md",
        "v2_campaign_plan.json",
        "frozen_inputs.json",
        "audit_existing_evidence.py",
        "run_local_audit.sh",
        "REGISTRY_SUGGESTION.json",
        "r39_reaggregated_summary.json",
        "audit_result.json",
    ]
    manifest_lines = []
    for name in manifest_files:
        path = HERE / name
        require(path.is_file(), f"missing local package file: {name}")
        manifest_lines.append(f"{sha256_file(path)}  {name}\n")
    preserve(HERE / "RESULTS.sha256", "".join(manifest_lines).encode("utf-8"))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
