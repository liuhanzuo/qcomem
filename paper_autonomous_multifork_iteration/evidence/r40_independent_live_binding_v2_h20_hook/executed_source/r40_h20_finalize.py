from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from r40_h20_binding_protocol import require, seal, verify_seal


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json_new(path: Path, value: Mapping[str, Any]) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def finalize(
    capture_root: Path, preregistration: Mapping[str, Any], *, expected_ranks: int
) -> dict[str, Any]:
    rows = []
    for rank in range(expected_ranks):
        path = capture_root / f"rank-{rank}" / "raw" / "independent-live-binding.json"
        require(path.is_file(), f"rank {rank} live-binding result missing")
        value = json.loads(path.read_text(encoding="utf-8"))
        verify_seal(value, "payload_sha256")
        require(value["rank"] == rank, "rank binding drift")
        require(value["process_workers"] is True, "rank did not use process workers")
        require(value["process_separated"] is True, "rank process separation failed")
        require(value["producer_manifest_sent"] is False, "producer manifest sent")
        require(value["producer_slot_ids_sent_to_registrar"] is False, "producer slot ids sent")
        require(value["clean_captures_passed"] == 4, "rank clean count drift")
        require(value["mutants_failed_closed"] == 4, "rank mutant count drift")
        faults = {row["fault_id"]: row for row in value["fault_results"]}
        require(set(faults) == {row["fault_id"] for row in preregistration["faults"]}, "fault set drift")
        for fault in preregistration["faults"]:
            row = faults[fault["fault_id"]]
            require(row["clean_detector"]["passed"] is True, "formal clean failed")
            require(row["mutant_detector"]["passed"] is False, "formal mutant escaped")
            require(
                all(code in row["mutant_detector"]["failure_codes"] for code in fault["required_detection_codes"]),
                "formal fault code drift",
            )
            require(row["registration_acknowledged_before_capture"] is True, "registration order drift")
            require(row["semantic_labels_mutated"] is False, "semantic labels mutated")
        rows.append(
            {
                "rank": rank,
                "path": path.relative_to(capture_root).as_posix(),
                "sha256": _sha256_file(path),
                "payload_sha256": value["payload_sha256"],
                "registration_event_count": value["registration_event_count"],
            }
        )
    return seal(
        {
            "schema_version": "forkaudit-r40-h20-live-binding-formal-aggregate-v1",
            "experiment_id": preregistration["experiment_id"],
            "status": "pass",
            "formal_evidence_eligible": True,
            "rank_count": len(rows),
            "clean_captures_passed": len(rows) * 4,
            "mutants_failed_closed": len(rows) * 4,
            "registrar_manifest_fields_received": 0,
            "registrar_slot_id_fields_received": 0,
            "rank_results": rows,
            "claim_boundary": preregistration["claim_boundary"],
            "payload_sha256": None,
        },
        "payload_sha256",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--expected-ranks", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(
        _sha256_file(args.preregistration) == args.expected_preregistration_sha256,
        "preregistration hash drift",
    )
    prereg = json.loads(args.preregistration.read_text(encoding="utf-8"))
    result = finalize(args.capture_root, prereg, expected_ranks=args.expected_ranks)
    _write_json_new(args.output, result)
    print("PASS ranks=8 clean=32 mutants=32")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

