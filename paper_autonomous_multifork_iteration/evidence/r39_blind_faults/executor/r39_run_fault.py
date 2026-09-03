#!/usr/bin/env python3
"""Orchestrate one frozen R39 fault with fresh subprocesses and clean gating."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import r39_contract as contract


def run_command(command: list[str], log: Path, *, timeout: int = 10800) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    log.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    contract.require(completed.returncode == 0, f"subprocess failed ({completed.returncode}): {' '.join(command[:3])}")


def lane_command(args: argparse.Namespace, lane: str, feasibility_sha: str) -> list[str]:
    return [
        sys.executable,
        str(args.lane_source.resolve()),
        "--fault-id", args.fault_id,
        "--lane", lane,
        "--gpu-index", str(args.gpu_index),
        "--expected-gpu-uuid", args.expected_gpu_uuid,
        "--lane-dir", str((args.run_dir / lane).resolve()),
        "--protocol", str(args.protocol.resolve()),
        "--plan", str(args.plan.resolve()),
        "--execution-input", str(args.execution_input.resolve()),
        "--feasibility", str(args.feasibility.resolve()),
        "--expected-feasibility-sha256", feasibility_sha,
        "--source-root", str(args.source_root.resolve()),
        "--source-manifest", str(args.source_manifest.resolve()),
    ]


def replay_command(
    args: argparse.Namespace, mode: str, feasibility_sha: str, output: Path
) -> list[str]:
    command = [
        sys.executable, "-I", str(args.replay_source.resolve()),
        "--mode", mode,
        "--fault-id", args.fault_id,
        "--reference-case", str((args.run_dir / "reference" / "case.json").resolve()),
        "--clean-case", str((args.run_dir / "clean" / "case.json").resolve()),
        "--feasibility-sha256", feasibility_sha,
        "--output", str(output.resolve()),
    ]
    if mode == "pair":
        command.extend([
            "--mutant-case", str((args.run_dir / "mutant" / "case.json").resolve())
        ])
    return command


def run(args: argparse.Namespace) -> dict[str, Any]:
    contract.require(args.fault_id in contract.FAULT_IDS, "unknown fault")
    contract.require(contract.FAULT_TO_GPU[args.fault_id] == args.gpu_index, "fault/GPU assignment drift")
    contract.require(not args.run_dir.exists(), "fault run directory already exists")
    freeze = contract.verify_freeze(args.protocol, args.plan)
    contract.verify_source_manifest(args.source_manifest, args.source_root)
    feasibility = json.loads(args.feasibility.read_text(encoding="utf-8"))
    contract.validate_feasibility(feasibility, fault_id=args.fault_id, freeze=freeze)
    feasibility_sha = contract.sha256_file(args.feasibility)
    args.run_dir.mkdir(parents=True)
    binding = {
        "schema_version": "forkaudit-r39-fault-run-binding-v1",
        "run_id": contract.RUN_ID,
        "fault_id": args.fault_id,
        "fault_row_sha256": freeze["fault_row_sha256"][args.fault_id],
        "gpu_index": args.gpu_index,
        "expected_gpu_uuid": args.expected_gpu_uuid,
        "trial_id": args.trial_id,
        "feasibility_sha256": feasibility_sha,
        "source_manifest_sha256": contract.sha256_file(args.source_manifest),
    }
    contract.atomic_json(args.run_dir / "binding.json", binding)
    if not feasibility["eligible"]:
        outcome = {
            "schema_version": "forkaudit-r39-ineligible-fault-outcome-v1",
            "run_id": contract.RUN_ID,
            "fault_id": args.fault_id,
            "status": "ineligible_preexecution",
            "feasibility_sha256": feasibility_sha,
            "ineligible_reason": feasibility["ineligible_reason"],
            "candidate_lanes_started": False,
            "fault_substituted": False,
            "observer_outcomes": None,
            "negative_or_escape_retained": True,
        }
        contract.atomic_json(args.run_dir / "outcome.json", outcome)
        return outcome
    try:
        for lane in ("reference", "clean"):
            run_command(
                lane_command(args, lane, feasibility_sha),
                args.run_dir / "logs" / f"{lane}.log",
            )
        clean_gate_path = args.run_dir / "replay" / "clean-gate.json"
        run_command(
            replay_command(args, "clean-gate", feasibility_sha, clean_gate_path),
            args.run_dir / "logs" / "clean-gate.log",
            timeout=900,
        )
        clean_gate = json.loads(clean_gate_path.read_text(encoding="utf-8"))
        contract.require(clean_gate.get("mutant_authorized") is True, "mutant blocked by clean gate")
        run_command(
            lane_command(args, "mutant", feasibility_sha),
            args.run_dir / "logs" / "mutant.log",
        )
        pair_path = args.run_dir / "replay" / "pair.json"
        run_command(
            replay_command(args, "pair", feasibility_sha, pair_path),
            args.run_dir / "logs" / "pair-replay.log",
            timeout=900,
        )
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        outcome = {
            "schema_version": "forkaudit-r39-fault-outcome-v1",
            "run_id": contract.RUN_ID,
            "fault_id": args.fault_id,
            "status": pair["status"],
            "valid_pair": pair["valid_pair"],
            "fault_reached": pair["fault_reached"],
            "feasibility_sha256": feasibility_sha,
            "reference_case_sha256": contract.sha256_file(args.run_dir / "reference" / "case.json"),
            "clean_case_sha256": contract.sha256_file(args.run_dir / "clean" / "case.json"),
            "mutant_case_sha256": contract.sha256_file(args.run_dir / "mutant" / "case.json"),
            "clean_gate_sha256": contract.sha256_file(clean_gate_path),
            "pair_replay_sha256": contract.sha256_file(pair_path),
            "observer_outcomes": pair["observers"],
            "negative_or_escape_retained": True,
            "population_detection_rate_computed": False,
        }
        contract.atomic_json(args.run_dir / "outcome.json", outcome)
        return outcome
    except BaseException as exc:
        invalid = {
            "schema_version": "forkaudit-r39-fault-run-operational-invalid-v1",
            "run_id": contract.RUN_ID,
            "fault_id": args.fault_id,
            "status": "operational_invalid",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "selective_rerun_permitted": False,
            "negative_outcome_preserved": True,
        }
        if not (args.run_dir / "operational-invalid.json").exists():
            contract.atomic_json(args.run_dir / "operational-invalid.json", invalid)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fault-id", choices=contract.FAULT_IDS, required=True)
    value.add_argument("--gpu-index", type=int, choices=range(8), required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    value.add_argument("--trial-id", type=int, required=True)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--plan", type=Path, required=True)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--feasibility", type=Path, required=True)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--source-manifest", type=Path, required=True)
    value.add_argument("--lane-source", type=Path, required=True)
    value.add_argument("--replay-source", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args()), sort_keys=True), flush=True)
