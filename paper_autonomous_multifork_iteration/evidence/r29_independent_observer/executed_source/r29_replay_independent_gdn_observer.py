from __future__ import annotations

"""CPU replay of the R29 independent-observer result.

This replay imports only the independent observer.  It treats the candidate's
serialized rows as untrusted input and recomputes descriptor and pair-relation
agreement from both raw snapshots.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from r29_independent_gdn_observer import (
    compare_candidate_snapshot,
    evaluate_lifecycle,
    evaluate_phase,
    sha256_json,
    validate_snapshot,
)


RESULT_SCHEMA = "forkaudit-r29-independent-gdn-observer-result-v1"
REPLAY_SCHEMA = "forkaudit-r29-independent-gdn-observer-replay-v1"


class R29ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R29ReplayError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return sha256_json(value)


def replay_result(value: Mapping[str, Any]) -> dict[str, Any]:
    require(value.get("schema_version") == RESULT_SCHEMA, "result schema drift")
    require(value.get("status") == "completed_valid_scientific_execution", "run did not complete scientifically")
    cells = value.get("cells")
    require(isinstance(cells, list) and len(cells) == 2, "policy-cell coverage drift")
    cell_reports = []
    total_rows = total_relations = 0
    for cell in cells:
        phases = cell.get("phases")
        require(isinstance(phases, list) and len(phases) == 3, "phase coverage drift")
        independent_snapshots = []
        phase_reports = []
        for phase in phases:
            before = phase["independent_before_candidate_capture"]
            after = phase["independent_after_candidate_capture"]
            candidate = phase["candidate_capture"]
            before_validation = validate_snapshot(before)
            after_validation = validate_snapshot(after)
            require(before == after, "candidate capture mutated live observed state")
            independent = evaluate_phase(before)
            comparison = compare_candidate_snapshot(before, candidate)
            require(independent["passed"] is True, "independent phase verdict failed")
            require(comparison["passed"] is True, "candidate/observer mismatch")
            require(independent == phase["independent_verdict"], "stored independent verdict drift")
            require(
                comparison == phase["independent_candidate_comparison"],
                "stored candidate comparison drift",
            )
            require(before_validation == after_validation, "before/after validation drift")
            total_rows += int(comparison["row_count"])
            total_relations += int(comparison["relation_count"])
            independent_snapshots.append(before)
            phase_reports.append(
                {
                    "phase": phase["phase"],
                    "row_count": comparison["row_count"],
                    "relation_count": comparison["relation_count"],
                    "row_descriptor_sha256": independent["row_descriptor_sha256"],
                    "relation_vector_sha256": independent["relation_vector_sha256"],
                    "candidate_comparison_exact": True,
                    "candidate_capture_nonmutating": True,
                }
            )
        lifecycle = evaluate_lifecycle(independent_snapshots)
        require(
            lifecycle == cell["independent_lifecycle_verdict"],
            "stored lifecycle verdict drift",
        )
        cell_reports.append(
            {
                "cell_id": cell["cell_id"],
                "gdn_policy": cell["gdn_policy"],
                "phase_reports": phase_reports,
                "lifecycle": lifecycle,
            }
        )
    return {
        "schema_version": REPLAY_SCHEMA,
        "passed": True,
        "valid_cell_count": len(cell_reports),
        "phase_count": sum(len(cell["phase_reports"]) for cell in cell_reports),
        "independently_recomputed_row_observations": total_rows,
        "independently_recomputed_pair_relations": total_relations,
        "candidate_capture_replay_imported": False,
        "candidate_passed_booleans_authoritative": False,
        "cell_reports": cell_reports,
        "result_semantic_sha256": canonical_sha(value),
    }


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(payload)
    pending.replace(path)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--input", type=Path, required=True)
    value.add_argument("--expected-input-raw-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    require(sha256_file(args.input) == args.expected_input_raw_sha256, "input raw SHA drift")
    value = json.loads(args.input.read_text(encoding="utf-8"))
    report = replay_result(value)
    write_json(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
