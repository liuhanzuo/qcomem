#!/usr/bin/env python3
"""Inventory replay entrypoints, evidence bytes, and log mtimes without mutation."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
EVIDENCE = PAPER / "evidence"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())


def size_row(root: Path) -> dict[str, Any]:
    files = regular_files(root)
    return {
        "path": root.relative_to(PAPER).as_posix(),
        "exists": root.is_dir(),
        "file_count": len(files),
        "logical_bytes": sum(path.stat().st_size for path in files),
        "allocated_bytes_from_st_blocks": sum(path.stat().st_blocks * 512 for path in files),
    }


def selected_size(paths: Iterable[Path]) -> dict[str, int]:
    unique = sorted({path.resolve() for path in paths if path.is_file() and not path.is_symlink()})
    return {"file_count": len(unique), "logical_bytes": sum(path.stat().st_size for path in unique)}


def iso_utc(seconds: float) -> str:
    return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def log_mtimes(root: Path) -> dict[str, Any]:
    logs = [
        path for path in regular_files(root)
        if path.suffix == ".log" or "logs" in path.relative_to(root).parts
    ]
    mtimes = [path.stat().st_mtime for path in logs]
    return {
        "log_file_count": len(logs),
        "nonempty_log_file_count": sum(path.stat().st_size > 0 for path in logs),
        "minimum_mtime_utc": iso_utc(min(mtimes)) if mtimes else None,
        "maximum_mtime_utc": iso_utc(max(mtimes)) if mtimes else None,
        "epoch_normalized_log_count": sum(path.stat().st_mtime == 0 for path in logs),
        "interpretation": "filesystem provenance only; not a capture-duration measurement",
    }


def tar_row(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": path.relative_to(PAPER).as_posix(), "exists": False}
    with tarfile.open(path, "r:gz") as handle:
        members = handle.getmembers()
    files = [member for member in members if member.isfile()]
    logs = [member for member in files if member.name.endswith(".log") or "/logs/" in member.name]
    mtimes = [member.mtime for member in logs]
    return {
        "path": path.relative_to(PAPER).as_posix(),
        "exists": True,
        "archive_bytes": path.stat().st_size,
        "archive_sha256": sha256_file(path),
        "member_file_count": len(files),
        "member_logical_bytes": sum(member.size for member in files),
        "log_member_count": len(logs),
        "minimum_log_mtime_utc": iso_utc(min(mtimes)) if mtimes else None,
        "maximum_log_mtime_utc": iso_utc(max(mtimes)) if mtimes else None,
        "epoch_normalized_log_count": sum(value == 0 for value in mtimes),
        "interpretation": "archive metadata only; not a capture-duration measurement",
    }


def entrypoint(path: Path, *, status: str, reason: str, expected_sha256: str | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "path": path.relative_to(PAPER).as_posix(),
        "exists": path.is_file(),
        "status": status,
        "reason": reason,
    }
    if path.is_file():
        row.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
    if expected_sha256 is not None:
        row["expected_sha256"] = expected_sha256
        row["matches_expected"] = row.get("sha256") == expected_sha256
    return row


def build_inventory() -> dict[str, Any]:
    primary = EVIDENCE / "round_04_rr2_package"
    census = EVIDENCE / "r39_independent_slot_census"
    dual = EVIDENCE / "r39_dual_producer_repeat"
    falcon = EVIDENCE / "r39_falcon_h1_transfer_v2"
    falcon_result = falcon / "formal_h20/20260827a/r39-falcon-h1-transfer-20260827b"
    blind = EVIDENCE / "r39_blind_faults"
    r33 = EVIDENCE / "r33_fresh_faults"
    r35 = EVIDENCE / "r35_historical_alias_regression"
    r30 = EVIDENCE / "r30_expanded_oracle_sweep"

    packages = {
        "primary_rr2": {
            "footprint": size_row(primary),
            "trace": selected_size(regular_files(primary / "upstream/raw")),
            "logs": log_mtimes(primary),
            "entrypoints": [entrypoint(primary / "replay/run_replay.sh", status="locally_runnable", reason="complete one-command CPU replay")],
        },
        "r39_preproducer_census": {
            "footprint": size_row(census),
            "trace": selected_size(regular_files(census / "r39-independent-slot-census-trial1907355-20260826a/r33-live/raw")),
            "logs": log_mtimes(census),
            "archive": tar_row(census / "formal_run_trial1907355_20260826a.tar.gz"),
            "entrypoints": [
                entrypoint(census / "scripts/audit_independent_slot_census.py", status="locally_runnable", reason="archived detached census audit"),
                entrypoint(census / "scripts/run_negative_controls.py", status="locally_runnable", reason="copy-only controls"),
            ],
        },
        "r39_dual_producer_repeat": {
            "footprint": size_row(dual),
            "trace": tar_row(dual / "formal_h20/r39-dual-producer-repeat-20260826a-formal-complete.tar.gz"),
            "logs": log_mtimes(dual),
            "entrypoints": [entrypoint(dual / "scripts/verify_dual_producer_repeat.py", status="locally_runnable_after_safe_unpack", reason="zero-tolerance detached verifier")],
        },
        "r39_falcon_h1_v2": {
            "footprint": size_row(falcon),
            "trace": selected_size(regular_files(falcon_result / "raw")),
            "logs": log_mtimes(falcon_result),
            "archive": tar_row(falcon / "formal_h20/r39-falcon-h1-transfer-v2-formal-h20-20260827a.tar.gz"),
            "entrypoints": [
                entrypoint(
                    EVIDENCE / "r39_falcon_h1_transfer_v2/executed_source/replay_r39_falcon_transfer.py",
                    status="blocked_unmeasured",
                    reason="formal v2 verifier source is absent locally",
                    expected_sha256="aa62966c00ffeeb1f7c9cf9f619cd082987235a66c3353a08485ed7afe79860f",
                ),
                entrypoint(
                    EVIDENCE / "r39_falcon_h1_transfer/executed_source/replay_r39_falcon_transfer.py",
                    status="not_substitutable",
                    reason="older v1 source has a different hash/schema",
                    expected_sha256="aa62966c00ffeeb1f7c9cf9f619cd082987235a66c3353a08485ed7afe79860f",
                ),
            ],
        },
        "r39_pdf_only_blind_faults": {
            "footprint": size_row(blind),
            "trace": selected_size(regular_files(blind / "formal_h20/r39-blind-faults-20260826g-metadata")),
            "logs": log_mtimes(blind),
            "archive": tar_row(blind / "formal_h20/r39-blind-faults-20260826g-metadata.tar.gz"),
            "entrypoints": [
                entrypoint(blind / "executor/r39_replay.py", status="blocked_unmeasured_full_replay", reason="352 bound FP32 sidecars are absent locally"),
                entrypoint(blind / "executor/r39_aggregate.py", status="metadata_only_not_full_replay", reason="can aggregate outcomes but cannot replace pair replay"),
            ],
        },
        "r33_designer_executor_faults": {
            "footprint": size_row(r33),
            "trace": selected_size(regular_files(r33 / "formal_h20/r33-fresh-faults-20260825b")),
            "logs": log_mtimes(r33),
            "archive": tar_row(r33 / "formal_h20/r33-fresh-faults-20260825b-result.tar.gz"),
            "entrypoints": [entrypoint(PAPER / "scripts/r33_aggregate_fresh_faults.py", status="locally_runnable", reason="replays all five pairs and aggregates")],
        },
        "r35_historical_alias": {
            "footprint": size_row(r35),
            "trace": selected_size(regular_files(r35 / "formal_h20/r35-historical-alias-20260826a")),
            "logs": log_mtimes(r35),
            "archive": tar_row(r35 / "formal_h20/r35-historical-alias-20260826a-output.tar.gz"),
            "entrypoints": [entrypoint(PAPER / "scripts/validate_r35_historical_alias_evidence.py", status="locally_runnable", reason="full hash graph plus fresh detached re-execution")],
        },
        "r30_expanded_oracle": {
            "footprint": size_row(r30),
            "trace": selected_size(regular_files(r30 / "raw/sidecars")),
            "logs": log_mtimes(r30),
            "entrypoints": [entrypoint(r30 / "r30_expanded_oracle_reference.py", status="locally_runnable", reason="candidate-import-free NumPy replay")],
        },
    }
    return {
        "schema_version": "forkaudit-r40-ci-cost-inventory-v1",
        "scope": "local evidence footprint, replay entrypoints, and filesystem/archive log metadata",
        "packages": packages,
        "unmeasured": [
            "current_package_h20_capture_wall_time",
            "gpu_perturbation_or_slowdown",
            "matched_uninstrumented_current_package_baseline",
            "engineering_or_adoption_effort",
            "cold_download_and_extraction_cost",
        ],
        "timestamp_boundary": "mtime and tar mtime are provenance metadata only; no duration is derived",
    }


def main() -> int:
    output = HERE / "inventory.json"
    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    output.write_text(json.dumps(build_inventory(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

