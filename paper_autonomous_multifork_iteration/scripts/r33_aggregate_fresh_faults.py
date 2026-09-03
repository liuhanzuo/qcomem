from __future__ import annotations

"""Detached fail-closed aggregation of exactly five frozen R33 pair artifacts."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import r33_executor_core as core
import r33_fault_replay as replay


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_bound(root: Path, reference: Any, label: str) -> tuple[Path, Any]:
    require(isinstance(reference, Mapping), f"{label} reference")
    relative = Path(str(reference.get("path", "")))
    require(not relative.is_absolute() and ".." not in relative.parts, f"{label} unsafe path")
    path = root / relative
    require(path.is_file(), f"{label} missing")
    require(sha256_file(path) == reference.get("sha256"), f"{label} SHA drift")
    return path, json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    require(not path.exists(), "aggregate output already exists")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    require(sha256_file(args.protocol) == args.expected_protocol_sha256, "protocol raw SHA drift")
    protocol = core.validate_protocol(json.loads(args.protocol.read_text(encoding="utf-8")))
    require(protocol["mode"] == "formal_fresh_faults", "non-formal R33 protocol")
    require(tuple(protocol["fault_ids"]) == replay.FAULT_IDS, "protocol exact fault order")
    rows: list[dict[str, Any]] = []
    for rank, fault_id in enumerate(replay.FAULT_IDS):
        rank_root = args.rank_run_root / f"rank-run-{rank}"
        pair_path = rank_root / f"rank-{rank}" / "pair.json"
        require(pair_path.is_file(), f"missing pair artifact rank {rank}")
        pair = json.loads(pair_path.read_text(encoding="utf-8"))
        binding = protocol["fault_bindings"][fault_id]
        require(
            pair.get("schema_version") == "forkaudit-r33-rank-pair-v1"
            and pair.get("run_id") == protocol["run_id"]
            and pair.get("rank") == rank
            and pair.get("fault_id") == fault_id,
            f"rank {rank} pair identity",
        )
        require(pair.get("fault_definition_sha256") == binding["fault_definition_sha256"], f"rank {rank} definition binding")
        require(pair.get("expected_primary_gate") == binding["expected_primary_gate"], f"rank {rank} gate binding")
        clean_path, clean = read_bound(rank_root, pair.get("clean_case"), f"rank {rank} clean")
        mutant_path, mutant = read_bound(rank_root, pair.get("mutant_case"), f"rank {rank} mutant")
        _, clean_replay = read_bound(rank_root, pair.get("clean_gate_replay"), f"rank {rank} clean replay")
        _, stored_pair_replay = read_bound(rank_root, pair.get("pair_replay"), f"rank {rank} pair replay")
        require(clean_replay.get("status") == "clean_gate_passed", f"rank {rank} clean gate")
        detached = replay.replay_pair(
            fault_id=fault_id,
            clean_case=clean,
            mutant_case=mutant,
            artifact_root=rank_root,
            expected_fault_definition_sha256=binding["fault_definition_sha256"],
        )
        require(detached == stored_pair_replay, f"rank {rank} replay drift")
        require(detached["expected_primary_gate"] == binding["expected_primary_gate"], f"rank {rank} expected gate")
        if detached["classification"] == "caught_by_expected_primary_gate":
            require(detached["first_failed_predicate"] == binding["expected_primary_gate"], f"rank {rank} first gate")
        else:
            require(
                detached["classification"] == "escaped_expected_primary_gate"
                and detached["first_failed_predicate"] is None,
                f"rank {rank} invalid classification",
            )
        rows.append(
            {
                "rank": rank,
                "fault_id": fault_id,
                "fault_definition_sha256": binding["fault_definition_sha256"],
                "expected_primary_gate": binding["expected_primary_gate"],
                "classification": detached["classification"],
                "first_failed_predicate": detached["first_failed_predicate"],
                "clean_case_sha256": sha256_file(clean_path),
                "mutant_case_sha256": sha256_file(mutant_path),
                "clean_gate_passed": True,
                "byte_binding_passed": True,
                "operational_invalid": False,
            }
        )
    require(len(rows) == 5, "R33 exact pair count")
    return {
        "schema_version": "forkaudit-r33-five-pair-summary-v1",
        "run_id": protocol["run_id"],
        "status": "completed_strict_scientific_aggregation",
        "scientific_valid": True,
        "pair_count": 5,
        "clean_gate_pass_count": 5,
        "operational_invalid_count": 0,
        "missing_pair_count": 0,
        "caught_by_expected_primary_gate_count": sum(
            row["classification"] == "caught_by_expected_primary_gate" for row in rows
        ),
        "escaped_expected_primary_gate_count": sum(
            row["classification"] == "escaped_expected_primary_gate" for row in rows
        ),
        "negative_or_escape_retained": True,
        "population_detection_rate_computed": False,
        "rows": rows,
        "claim_boundary": {
            "per_frozen_fault_outcomes_only": True,
            "heldout_population_claim_allowed": False,
            "single_model_single_stack_fixed_case_only": True,
            "candidate_import_free_replay_is_not_independent_live_recapture": True,
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--expected-protocol-sha256", required=True)
    value.add_argument("--rank-run-root", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


if __name__ == "__main__":
    args = parser().parse_args()
    summary = aggregate(args)
    write_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True), flush=True)
