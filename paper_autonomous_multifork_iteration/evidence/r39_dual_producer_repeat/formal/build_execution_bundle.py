from __future__ import annotations

"""Build a deterministic self-contained dual-producer execution archive."""

import argparse
import gzip
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


def verify_source_ledger(root: Path) -> None:
    ledger = root / "source-code.sha256"
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(None, 1)
        relative = relative.strip()
        path = root / relative
        if sha256_file(path) != expected:
            raise RuntimeError(f"source ledger mismatch: {relative}")


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
    dual = paper / "evidence/r39_dual_producer_repeat"
    r39 = paper / "evidence/r39_independent_slot_census/scripts"
    r33_scripts = paper / "scripts"
    r33_evidence = paper / "evidence/r33_independent_capture"
    r29 = paper / "evidence/r29_independent_observer/executed_source"
    if args.output.exists() or args.build_receipt.exists():
        raise FileExistsError("output and build receipt must both be absent")

    dual_files = [
        "README.md",
        "DESIGN.md",
        "preregistration.json",
        "slot_protocol.json",
        "source-code.sha256",
        "artifacts/prior-repeatability-audit.json",
        "scripts/verify_dual_producer_repeat.py",
        "tests/test_dual_producer_repeat.py",
        "formal/launch_dual_producer_h20.sh",
        "formal/launch_existing_h20_node.sh",
        "formal/build_execution_bundle.py",
    ]
    r39_files = [
        "audit_independent_slot_census.py",
        "generate_preexecution_census.py",
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

    with tempfile.TemporaryDirectory(prefix="forkaudit-r39-dual-bundle-") as temp:
        stage = Path(temp) / "r39_dual_producer_repeat"
        for relative in dual_files:
            add_file(dual / relative, stage / relative)
        for name in r39_files:
            add_file(r39 / name, stage / "vendor/r39" / name)
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

        verify_source_ledger(stage)
        included = sorted(
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_file()
        )
        manifest = {
            "schema_version": "forkaudit-r39-dual-producer-execution-bundle-v1",
            "status": "source_frozen_before_dual_producer_execution",
            "entrypoint": "formal/launch_existing_h20_node.sh",
            "execution": "two fresh producer processes, serial on one selected GPU",
            "manages_qs_resources": False,
            "source_ledger_raw_sha256": sha256_file(stage / "source-code.sha256"),
            "preregistration_raw_sha256": sha256_file(stage / "preregistration.json"),
            "files": {
                relative: sha256_file(stage / relative) for relative in included
            },
        }
        write_json(stage / "BUNDLE_MANIFEST.json", manifest)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as raw_handle:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_handle,
                mtime=0,
            ) as gzip_handle:
                with tarfile.open(fileobj=gzip_handle, mode="w") as archive:
                    archive.add(
                        stage,
                        arcname=stage.name,
                        recursive=False,
                        filter=deterministic_filter,
                    )
                    for path in sorted(stage.rglob("*")):
                        archive.add(
                            path,
                            arcname=(Path(stage.name) / path.relative_to(stage)).as_posix(),
                            recursive=False,
                            filter=deterministic_filter,
                        )

    receipt = {
        "schema_version": "forkaudit-r39-dual-producer-bundle-build-v1",
        "status": "ready_for_transfer_to_an_existing_h20_node_not_launched",
        "package_path": args.output.name,
        "package_sha256": sha256_file(args.output),
        "entrypoint_after_extraction": (
            "r39_dual_producer_repeat/formal/launch_existing_h20_node.sh"
        ),
        "producer_execution": "serial-one-gpu",
        "required_external_state": (
            "existing mounted model, RR2 source, data, and frozen Python environment"
        ),
        "creates_stops_or_deletes_qs_resources": False,
    }
    args.build_receipt.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.build_receipt, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
