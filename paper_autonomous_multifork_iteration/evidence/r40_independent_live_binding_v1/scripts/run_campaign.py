from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

from r40lib.protocol import CAMPAIGN_SCHEMA, seal_payload, validate_preregistration
from r40lib.provenance import (
    load_json,
    sha256_file,
    verify_source_ledger,
    write_json_new,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--source-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    preregistration_path = args.preregistration.resolve()
    source_ledger_path = args.source_ledger.resolve()
    preregistration = load_json(preregistration_path)
    validate_preregistration(preregistration)
    verify_source_ledger(root, source_ledger_path)
    preregistration_sha = sha256_file(preregistration_path)
    source_ledger_sha = sha256_file(source_ledger_path)
    lane_script = root / "scripts/run_lane.py"
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(root)
    lanes: list[dict] = []
    commands: list[list[str]] = []
    for fault in preregistration["faults"]:
        for lane_type in ("clean", "mutant"):
            command = [
                sys.executable,
                str(lane_script),
                "--preregistration",
                str(preregistration_path),
                "--expected-preregistration-sha256",
                preregistration_sha,
                "--source-ledger",
                str(source_ledger_path),
                "--expected-source-ledger-sha256",
                source_ledger_sha,
                "--fault-id",
                fault["fault_id"],
                "--lane-type",
                lane_type,
            ]
            commands.append(command)
            completed = subprocess.run(
                command,
                cwd=root,
                env=env,
                check=False,
                text=True,
                capture_output=True,
                timeout=180,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"lane failed {fault['fault_id']}/{lane_type}:\n"
                    f"stdout={completed.stdout}\nstderr={completed.stderr}"
                )
            lines = [line for line in completed.stdout.splitlines() if line.strip()]
            if len(lines) != 1:
                raise RuntimeError(
                    f"lane stdout contract drift {fault['fault_id']}/{lane_type}: {lines}"
                )
            lanes.append(json.loads(lines[0]))
    controls = [row for row in lanes if row["lane_type"] == "clean"]
    mutants = [row for row in lanes if row["lane_type"] == "mutant"]
    pair_oracles_exact = all(
        next(
            row["stable_oracle_projection_sha256"]
            for row in controls
            if row["fault_id"] == fault["fault_id"]
        )
        == next(
            row["stable_oracle_projection_sha256"]
            for row in mutants
            if row["fault_id"] == fault["fault_id"]
        )
        for fault in preregistration["faults"]
    )
    producer_pids = [int(row["producer_pid"]) for row in lanes]
    campaign_passed = all(
        (
            len(controls) == int(preregistration["acceptance"]["matched_clean_controls_required"]),
            len(mutants) == int(preregistration["acceptance"]["mutants_required"]),
            all(row["acceptance_passed"] and row["detector"]["passed"] for row in controls),
            all(row["acceptance_passed"] and not row["detector"]["passed"] for row in mutants),
            pair_oracles_exact,
            len(set(producer_pids)) == len(producer_pids),
        )
    )
    result = {
        "schema_version": CAMPAIGN_SCHEMA,
        "experiment_id": preregistration["experiment_id"],
        "status": "completed" if campaign_passed else "failed",
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "preregistration_sha256": preregistration_sha,
        "source_ledger_sha256": source_ledger_sha,
        "lane_order": [f"{row['fault_id']}/{row['lane_type']}" for row in lanes],
        "fresh_lane_command_count": len(commands),
        "lanes": lanes,
        "aggregate": {
            "matched_clean_controls": len(controls),
            "clean_controls_accepted": sum(row["detector"]["passed"] for row in controls),
            "mutants": len(mutants),
            "mutants_failed_closed": sum(not row["detector"]["passed"] for row in mutants),
            "pair_oracle_projections_exact": pair_oracles_exact,
            "producer_pids_all_distinct": len(set(producer_pids)) == len(producer_pids),
            "schema_or_label_mutation_faults": sum(
                bool(row["injection_receipt"]["schema_or_label_row_mutation_used"])
                for row in mutants
            ),
            "actual_live_handle_faults": sum(
                bool(row["injection_receipt"]["actual_live_tensor_references_changed"])
                for row in mutants
            ),
            "campaign_passed": campaign_passed,
        },
        "claim_boundary": preregistration["claim_boundary"],
        "campaign_payload_sha256": None,
    }
    sealed = seal_payload(result, "campaign_payload_sha256")
    write_json_new(args.output.resolve(), sealed)
    print(
        f"{'PASS' if campaign_passed else 'FAIL'} controls={len(controls)} "
        f"mutants={len(mutants)} output={args.output.resolve()}"
    )
    return 0 if campaign_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

