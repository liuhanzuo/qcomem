#!/usr/bin/env python3
"""Run R40 Attempt B from a verified manifest-only RR2 package view."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Sequence

import measure_replays as base


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_attempt_b"
SOURCE_PRIMARY = base.EVIDENCE / "round_04_rr2_package"
CLEAN_PRIMARY = base.PREPARED / "primary_manifest_view"
PRIMARY_RECEIPT = base.PREPARED / "PRIMARY_MANIFEST_VIEW.json"
AMENDMENT = HERE / "AMENDMENT_ATTEMPT_B.md"


def validate_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    base.require(
        bool(value)
        and value != "."
        and not path.is_absolute()
        and ".." not in path.parts
        and path.as_posix() == value,
        f"unsafe manifest path: {value!r}",
    )
    return path


def observed_member_paths(root: Path) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.relative_to(root).as_posix() not in {"MANIFEST.json", "MANIFEST.sha256"}
    )


def copy_exclusive(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("xb") as destination_handle:
        shutil.copyfileobj(source_handle, destination_handle, length=1024 * 1024)
        destination_handle.flush()
        os.fsync(destination_handle.fileno())
    os.chmod(destination, stat.S_IMODE(source.stat().st_mode))


def build_primary_manifest_view() -> dict[str, Any]:
    if PRIMARY_RECEIPT.exists():
        receipt = base.load_json(PRIMARY_RECEIPT)
        base.require(CLEAN_PRIMARY.is_dir(), "primary manifest view missing")
        base.require(
            base.sha256_file(CLEAN_PRIMARY / "MANIFEST.json") == receipt["manifest_sha256"],
            "primary manifest view binding drift",
        )
        return receipt

    base.require(not CLEAN_PRIMARY.exists(), "primary manifest view exists without receipt")
    manifest_path = SOURCE_PRIMARY / "MANIFEST.json"
    sidecar_path = SOURCE_PRIMARY / "MANIFEST.sha256"
    manifest = base.load_json(manifest_path)
    rows = manifest["files"]
    expected_paths = [row["relative_path"] for row in rows]
    base.require(expected_paths == sorted(expected_paths), "manifest paths are not sorted")
    base.require(len(expected_paths) == len(set(expected_paths)), "manifest paths are not unique")
    base.require(len(rows) == manifest["file_count"], "manifest file count drift")
    base.require(sum(row["bytes"] for row in rows) == manifest["total_bytes"], "manifest byte total drift")
    manifest_sha256 = base.sha256_file(manifest_path)
    expected_sidecar = f"{manifest_sha256}  MANIFEST.json"
    base.require(sidecar_path.read_text(encoding="ascii").strip() == expected_sidecar, "manifest sidecar drift")

    actual_paths = observed_member_paths(SOURCE_PRIMARY)
    extras = sorted(set(actual_paths) - set(expected_paths))
    missing = sorted(set(expected_paths) - set(actual_paths))
    base.require(not missing, f"manifest members absent from source package: {missing}")

    start = time.monotonic_ns()
    CLEAN_PRIMARY.mkdir(parents=True)
    copied_bytes = 0
    for row in rows:
        relative = validate_relative_path(str(row["relative_path"]))
        source = SOURCE_PRIMARY.joinpath(*relative.parts)
        destination = CLEAN_PRIMARY.joinpath(*relative.parts)
        base.require(source.is_file() and not source.is_symlink(), f"not a regular source member: {relative}")
        base.require(source.stat().st_size == row["bytes"], f"source size drift: {relative}")
        base.require(base.sha256_file(source) == row["sha256"], f"source SHA drift: {relative}")
        copy_exclusive(source, destination)
        base.require(destination.stat().st_size == row["bytes"], f"copied size drift: {relative}")
        base.require(base.sha256_file(destination) == row["sha256"], f"copied SHA drift: {relative}")
        copied_bytes += destination.stat().st_size
    copy_exclusive(manifest_path, CLEAN_PRIMARY / "MANIFEST.json")
    copy_exclusive(sidecar_path, CLEAN_PRIMARY / "MANIFEST.sha256")

    verification = subprocess.run(
        [
            sys.executable,
            "-B",
            str(CLEAN_PRIMARY / "replay/verify_package_manifest.py"),
            "--package-root",
            str(CLEAN_PRIMARY),
        ],
        cwd=base.PAPER,
        env=base.replay_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    end = time.monotonic_ns()
    base.require(verification.returncode == 0, "copied primary package did not pass its own verifier")
    base.require(observed_member_paths(CLEAN_PRIMARY) == expected_paths, "copied primary file set drift")
    receipt = {
        "schema_version": "forkaudit-r40-primary-manifest-view-v1",
        "source_package": SOURCE_PRIMARY.relative_to(base.PAPER).as_posix(),
        "destination": CLEAN_PRIMARY.relative_to(HERE).as_posix(),
        "manifest_sha256": manifest_sha256,
        "manifest_sidecar_sha256": base.sha256_file(sidecar_path),
        "declared_file_count": manifest["file_count"],
        "declared_logical_bytes": manifest["total_bytes"],
        "copied_logical_bytes": copied_bytes,
        "all_listed_members_size_and_sha256_verified_before_and_after_copy": True,
        "copied_package_own_verifier_passed": True,
        "verification_stdout": verification.stdout.decode("utf-8", errors="replace").strip(),
        "verification_stderr": verification.stderr.decode("utf-8", errors="replace").strip(),
        "source_unmanifested_file_count": len(extras),
        "source_unmanifested_paths": extras,
        "source_manifest_missing_file_count": 0,
        "preparation_wall_seconds": (end - start) / 1_000_000_000,
        "included_in_replay_timing": False,
        "source_evidence_modified": False,
    }
    base.write_new_json(PRIMARY_RECEIPT, receipt)
    return receipt


def component_primary_attempt_b(output_root: Path) -> dict[str, Any]:
    command = [str(CLEAN_PRIMARY / "replay/run_replay.sh")]
    row = base.run_subcommand("primary-replay", command, output_root)
    value = base.last_json_line(output_root / "primary-replay.stdout.txt")
    base.require(value.get("passed") is True and value.get("raw_artifacts") == 536, "primary replay verdict drift")
    return {
        "passed": True,
        "subcommands": [row],
        "verdict": value,
        "primary_input_view": "manifest_only_copy",
        "source_evidence_modified": False,
    }


def attempt_b_environment(primary_receipt: dict[str, Any], dual_receipt: dict[str, Any]) -> dict[str, Any]:
    value = base.host_environment()
    value.update({
        "schema_version": "forkaudit-r40-measurement-environment-attempt-b-v1",
        "attempt": "B",
        "measurement_script_sha256": base.sha256_file(Path(__file__).resolve()),
        "base_measurement_script_sha256": base.sha256_file(HERE / "measure_replays.py"),
        "protocol_amendment_sha256": base.sha256_file(AMENDMENT),
        "primary_manifest_view_preparation": primary_receipt,
        "dual_archive_preparation": dual_receipt,
    })
    return value


def run_profiles(repetitions: int) -> int:
    base.require(repetitions == 3, "frozen Attempt B requires exactly three repetitions")
    base.require(not RAW.exists(), "Attempt B raw measurement directory already exists")
    dual_receipt = base.prepare_inputs()
    primary_receipt = build_primary_manifest_view()
    RAW.mkdir(parents=True)
    base.write_new_json(RAW / "measurement_environment.json", attempt_b_environment(primary_receipt, dual_receipt))
    component_rows_path = RAW / "component_timing_rows.jsonl"
    profile_rows_path = RAW / "profile_timing_rows.jsonl"
    any_failure = False
    for iteration in range(1, repetitions + 1):
        profile_started_at = base.utc_now()
        profile_start = time.monotonic_ns()
        iteration_rows: list[dict[str, Any]] = []
        for component in base.COMPONENTS:
            output_root = RAW / f"iteration-{iteration:02d}" / component
            base.require(not output_root.exists(), f"component output exists: {output_root}")
            output_root.mkdir(parents=True)
            time_path = output_root / "darwin-time.txt"
            stdout_path = output_root / "worker.stdout.txt"
            stderr_path = output_root / "worker.stderr.txt"
            result_path = output_root / "component-result.json"
            command = [
                "/usr/bin/time", "-l", "-o", str(time_path),
                sys.executable, "-B", str(Path(__file__).resolve()),
                "--component", component,
                "--component-output-root", str(output_root),
                "--component-result", str(result_path),
            ]
            started_at = base.utc_now()
            start = time.monotonic_ns()
            with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
                completed = subprocess.run(
                    command,
                    cwd=base.PAPER,
                    env=base.replay_env(),
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    check=False,
                )
            end = time.monotonic_ns()
            timing = base.parse_time_report(time_path)
            result = base.load_json(result_path) if result_path.is_file() else None
            passed = completed.returncode == 0 and isinstance(result, dict) and result.get("passed") is True
            row = {
                "schema_version": "forkaudit-r40-component-timing-row-attempt-b-v1",
                "attempt": "B",
                "profile": "extended_supporting",
                "iteration": iteration,
                "component": component,
                "started_at_utc": started_at,
                "ended_at_utc": base.utc_now(),
                "wall_seconds_monotonic": (end - start) / 1_000_000_000,
                "exit_code": completed.returncode,
                "passed": passed,
                "worker_result_sha256": base.sha256_file(result_path) if result_path.is_file() else None,
                "worker_stdout_sha256": base.sha256_file(stdout_path),
                "worker_stderr_sha256": base.sha256_file(stderr_path),
                **timing,
            }
            base.append_jsonl(component_rows_path, row)
            iteration_rows.append(row)
            any_failure = any_failure or not passed
        profile_end = time.monotonic_ns()
        primary = next(row for row in iteration_rows if row["component"] == "primary_rr2")
        profile_row = {
            "schema_version": "forkaudit-r40-profile-timing-row-attempt-b-v1",
            "attempt": "B",
            "iteration": iteration,
            "started_at_utc": profile_started_at,
            "ended_at_utc": base.utc_now(),
            "minimal_core_wall_seconds": primary["wall_seconds_monotonic"],
            "minimal_core_peak_rss_bytes": primary["maximum_resident_set_size_raw"],
            "extended_supporting_wall_seconds": (profile_end - profile_start) / 1_000_000_000,
            "extended_supporting_max_observed_component_peak_rss_bytes": max(
                row["maximum_resident_set_size_raw"] for row in iteration_rows
            ),
            "component_count": len(iteration_rows),
            "components_serial_no_overlap": True,
            "passed": all(row["passed"] for row in iteration_rows),
        }
        base.append_jsonl(profile_rows_path, profile_row)
    completion = {
        "schema_version": "forkaudit-r40-run-completion-attempt-b-v1",
        "attempt": "B",
        "completed_at_utc": base.utc_now(),
        "repetitions": repetitions,
        "component_rows": repetitions * len(base.COMPONENTS),
        "all_rows_passed": not any_failure,
        "component_rows_sha256": base.sha256_file(component_rows_path),
        "profile_rows_sha256": base.sha256_file(profile_rows_path),
        "attempt_a_retained": (HERE / "raw/RUN_COMPLETE.json").is_file(),
        "no_gpu_or_qs_access": True,
    }
    base.write_new_json(RAW / "RUN_COMPLETE.json", completion)
    return 0 if not any_failure else 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--component", choices=base.COMPONENTS)
    value.add_argument("--repetitions", type=int, default=3)
    value.add_argument("--component-output-root", type=Path)
    value.add_argument("--component-result", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.run:
        return run_profiles(args.repetitions)
    base.require(args.component_output_root is not None and args.component_result is not None, "component paths required")
    base.COMPONENT_FUNCTIONS["primary_rr2"] = component_primary_attempt_b
    return base.run_component(
        args.component,
        args.component_output_root.resolve(),
        args.component_result.resolve(),
    )


if __name__ == "__main__":
    raise SystemExit(main())

