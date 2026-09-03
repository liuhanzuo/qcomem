from __future__ import annotations

"""Build a self-contained R39/R33/R29 source bundle for an existing H20 node."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Sequence


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def add_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def deterministic_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.mtime = 0
    return info


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paper-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    paper = args.paper_root.resolve()
    r39 = paper / "evidence/r39_independent_slot_census"
    r33_scripts = paper / "scripts"
    r33_evidence = paper / "evidence/r33_independent_capture"
    r29 = paper / "evidence/r29_independent_observer/executed_source"
    if args.output.exists() or args.build_receipt.exists():
        raise FileExistsError("output and build receipt must both be absent")

    r39_files = [
        "protocol.json",
        "README.md",
        "DESIGN.md",
        "scripts/audit_independent_slot_census.py",
        "scripts/run_negative_controls.py",
        "scripts/generate_preexecution_census.py",
        "scripts/aggregate_formal_run.py",
        "tests/test_independent_slot_census.py",
        "tests/test_formal_pipeline.py",
        "formal/launch_r39_h20.sh",
        "formal/launch_trial_1907358.sh",
        "formal/build_execution_bundle.py",
    ]
    r33_files = [
        "r33_ipc_capture_protocol.py",
        "r33_independent_capture_worker.py",
        "r33_out_of_process_capture.py",
        "r33_replay_independent_capture.py",
        "r33_run_h20_independent_capture.py",
        "r33_run_local_capture_gate.py",
        "r33_test_independent_capture.py",
        "r33_launch_h20_independent_capture_1gpu.sh",
    ]
    r29_files = [
        "r29_independent_gdn_observer.py",
        "r29_run_independent_gdn_observer.py",
    ]
    with tempfile.TemporaryDirectory(prefix="forkaudit-r39-bundle-") as temp:
        stage = Path(temp) / "r39_independent_slot_census"
        for relative in r39_files:
            add_file(r39 / relative, stage / relative)
        for name in r33_files:
            add_file(r33_scripts / name, stage / "vendor/r33" / name)
        add_file(
            r33_evidence / "formal_h20/result/preregistration/source-code.sha256",
            stage / "vendor/r33/source-code.sha256",
        )
        add_file(
            r33_evidence / "formal_h20/result/preregistration/preregistration.json",
            stage / "vendor/r33/preregistration.json",
        )
        for name in r29_files:
            add_file(r29 / name, stage / "vendor/r29" / name)
        included = sorted(
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_file()
        )
        file_hashes = {
            relative: sha256_file(stage / relative) for relative in included
        }
        manifest = {
            "schema_version": "forkaudit-r39-independent-slot-execution-bundle-v1",
            "status": "source_frozen_before_fresh_h20_execution",
            "entrypoint": "formal/launch_trial_1907358.sh",
            "manages_qs_resources": False,
            "files": file_hashes,
        }
        write_json(stage / "BUNDLE_MANIFEST.json", manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(args.output, "w:gz", compresslevel=9) as archive:
            archive.add(stage, arcname=stage.name, filter=deterministic_filter)
    receipt = {
        "schema_version": "forkaudit-r39-execution-bundle-build-v1",
        "status": "ready_for_transfer_to_existing_h20_node",
        "package_path": args.output.name,
        "package_sha256": sha256_file(args.output),
        "entrypoint_after_extraction": (
            "r39_independent_slot_census/formal/launch_trial_1907358.sh"
        ),
        "required_external_state": (
            "existing mounted model, RR2 source, data, and frozen Python environment"
        ),
        "creates_or_stops_qs_resources": False,
    }
    args.build_receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.build_receipt, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
