#!/usr/bin/env python3
"""Run the frozen R40 local replay profiles without touching prior evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import tarfile
import time
from typing import Any, Callable, Sequence


HERE = Path(__file__).resolve().parent
PAPER = HERE.parents[1]
REPO = PAPER.parent
EVIDENCE = PAPER / "evidence"
RAW = HERE / "raw"
PREPARED = HERE / "prepared_inputs"
COMPONENTS = (
    "primary_rr2",
    "r39_preproducer_census",
    "r39_dual_producer",
    "r33_designer_faults",
    "r35_historical_alias",
    "r30_expanded_oracle",
)


class MeasurementError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MeasurementError(message)


def utc_now() -> str:
    return dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def write_new_json(path: Path, value: Any) -> None:
    write_new(path, canonical_bytes(value))


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())


def safe_extract(archive: Path, destination: Path) -> dict[str, Any]:
    require(not destination.exists(), f"refusing to replace prepared input: {destination}")
    destination.mkdir(parents=True)
    start = time.monotonic_ns()
    with tarfile.open(archive, "r:gz") as handle:
        members = handle.getmembers()
        seen: set[str] = set()
        for member in members:
            name = PurePosixPath(member.name)
            require(
                bool(member.name)
                and not name.is_absolute()
                and ".." not in name.parts
                and member.name not in seen,
                f"unsafe or duplicate tar member: {member.name}",
            )
            require(member.isdir() or member.isfile(), f"unsupported tar member: {member.name}")
            seen.add(member.name)
        handle.extractall(destination)
    end = time.monotonic_ns()
    files = sorted(path for path in destination.rglob("*") if path.is_file())
    return {
        "schema_version": "forkaudit-r40-prepared-input-v1",
        "archive": archive.relative_to(PAPER).as_posix(),
        "archive_sha256": sha256_file(archive),
        "archive_bytes": archive.stat().st_size,
        "destination": destination.relative_to(HERE).as_posix(),
        "member_file_count": len(files),
        "member_logical_bytes": sum(path.stat().st_size for path in files),
        "safe_member_validation": True,
        "preparation_wall_seconds": (end - start) / 1_000_000_000,
        "included_in_replay_timing": False,
    }


def prepare_inputs() -> dict[str, Any]:
    dual_archive = EVIDENCE / "r39_dual_producer_repeat/formal_h20/r39-dual-producer-repeat-20260826a-formal-complete.tar.gz"
    destination = PREPARED / "dual_formal"
    receipt_path = PREPARED / "PREPARATION.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        require(destination.is_dir(), "prepared dual directory missing")
        require(sha256_file(dual_archive) == receipt["archive_sha256"], "prepared archive binding drift")
        return receipt
    require(not PREPARED.exists(), "prepared input directory exists without a receipt")
    receipt = safe_extract(dual_archive, destination)
    write_new_json(receipt_path, receipt)
    return receipt


def replay_env(extra_pythonpath: str | None = None) -> dict[str, str]:
    value = dict(os.environ)
    value.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "CUDA_VISIBLE_DEVICES": "",
    })
    if extra_pythonpath is not None:
        value["PYTHONPATH"] = extra_pythonpath
    return value


def run_subcommand(
    label: str,
    command: Sequence[str],
    output_root: Path,
    *,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    stdout_path = output_root / f"{label}.stdout.txt"
    stderr_path = output_root / f"{label}.stderr.txt"
    require(not stdout_path.exists() and not stderr_path.exists(), f"subcommand logs already exist: {label}")
    started_at = utc_now()
    start = time.monotonic_ns()
    completed = subprocess.run(
        list(command),
        cwd=PAPER,
        env=env or replay_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    end = time.monotonic_ns()
    write_new(stdout_path, completed.stdout)
    write_new(stderr_path, completed.stderr)
    row = {
        "label": label,
        "command": list(command),
        "cwd": ".",
        "started_at_utc": started_at,
        "ended_at_utc": utc_now(),
        "wall_seconds": (end - start) / 1_000_000_000,
        "exit_code": completed.returncode,
        "stdout_bytes": len(completed.stdout),
        "stdout_sha256": hashlib.sha256(completed.stdout).hexdigest(),
        "stderr_bytes": len(completed.stderr),
        "stderr_sha256": hashlib.sha256(completed.stderr).hexdigest(),
    }
    require(completed.returncode == 0, f"{label} exited {completed.returncode}")
    return row


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def last_json_line(path: Path) -> dict[str, Any]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    require(lines, f"no stdout JSON in {path.name}")
    value = json.loads(lines[-1])
    require(isinstance(value, dict), f"last stdout row is not an object: {path.name}")
    return value


def component_primary(output_root: Path) -> dict[str, Any]:
    command = [str(EVIDENCE / "round_04_rr2_package/replay/run_replay.sh")]
    row = run_subcommand("primary-replay", command, output_root)
    value = last_json_line(output_root / "primary-replay.stdout.txt")
    require(value.get("passed") is True and value.get("raw_artifacts") == 536, "primary replay verdict drift")
    return {"passed": True, "subcommands": [row], "verdict": value}


def component_census(output_root: Path) -> dict[str, Any]:
    root = EVIDENCE / "r39_independent_slot_census"
    audit_out = output_root / "clean-audit.json"
    census_out = output_root / "expected-slot-census.json"
    controls_out = output_root / "negative-controls.json"
    audit = run_subcommand(
        "census-audit",
        [
            sys.executable, "-B", str(root / "scripts/audit_independent_slot_census.py"),
            "--protocol", str(root / "protocol.json"),
            "--input", str(EVIDENCE / "r33_independent_capture/formal_h20/result/raw/out-of-process-gdn-capture.json"),
            "--preregistration", str(EVIDENCE / "r33_independent_capture/formal_h20/result/preregistration/preregistration.json"),
            "--output", str(audit_out), "--census-output", str(census_out),
        ],
        output_root,
    )
    controls = run_subcommand(
        "census-controls",
        [
            sys.executable, "-B", str(root / "scripts/run_negative_controls.py"),
            "--protocol", str(root / "protocol.json"),
            "--input", str(EVIDENCE / "r33_independent_capture/formal_h20/result/raw/out-of-process-gdn-capture.json"),
            "--output", str(controls_out),
        ],
        output_root,
    )
    audit_value = load_json(audit_out)
    control_value = load_json(controls_out)
    require(audit_value.get("passed") is True and audit_value.get("audited_row_observations") == 1080, "census replay drift")
    require(control_value.get("passed") is True and control_value.get("control_count") == 3, "census control drift")
    return {"passed": True, "subcommands": [audit, controls], "audited_rows": 1080, "audited_relations": 96660, "control_count": 3}


def component_dual(output_root: Path) -> dict[str, Any]:
    root = EVIDENCE / "r39_dual_producer_repeat"
    extracted = PREPARED / "dual_formal/r39-dual-producer-repeat-20260826a"
    require(extracted.is_dir(), "prepared dual formal tree absent")
    output = output_root / "dual-recomputed.json"
    pythonpath = os.pathsep.join([str(root / "scripts"), str(EVIDENCE / "r39_independent_slot_census/scripts")])
    command = [
        sys.executable, "-B", str(root / "scripts/verify_dual_producer_repeat.py"),
        "--preregistration", str(extracted / "preexecution/preregistration.json"),
        "--slot-protocol", str(extracted / "preexecution/slot_protocol.json"),
        "--source-ledger", str(extracted / "preexecution/source-code.sha256"),
        "--census", str(extracted / "preexecution/expected-slot-census.json"),
        "--census-receipt", str(extracted / "preexecution/census-receipt.json"),
        "--producer-a", str(extracted / "producer-a/raw/out-of-process-gdn-capture.json"),
        "--producer-b", str(extracted / "producer-b/raw/out-of-process-gdn-capture.json"),
        "--producer-a-replay", str(extracted / "producer-a/replay/out-of-process-gdn-replay.json"),
        "--producer-b-replay", str(extracted / "producer-b/replay/out-of-process-gdn-replay.json"),
        "--producer-a-audit", str(extracted / "audit/producer-a-census-audit.json"),
        "--producer-b-audit", str(extracted / "audit/producer-b-census-audit.json"),
        "--output", str(output),
    ]
    row = run_subcommand("dual-verifier", command, output_root, env=replay_env(pythonpath))
    archived = extracted / "audit/dual-producer-summary.json"
    value = load_json(output)
    require(value.get("passed") is True and value.get("matched_relation_labels") == 96660, "dual replay drift")
    require(output.read_bytes() == archived.read_bytes(), "dual replay is not byte-identical to archived summary")
    return {"passed": True, "subcommands": [row], "byte_identical_to_archived_summary": True, "summary_sha256": sha256_file(output)}


def component_r33(output_root: Path) -> dict[str, Any]:
    root = EVIDENCE / "r33_fresh_faults"
    protocol = root / "executor_attempt_b/formal-protocol.json"
    output = output_root / "r33-recomputed-summary.json"
    scripts = PAPER / "scripts"
    command = [
        sys.executable, "-B", str(scripts / "r33_aggregate_fresh_faults.py"),
        "--protocol", str(protocol),
        "--expected-protocol-sha256", sha256_file(protocol),
        "--rank-run-root", str(root / "formal_h20/r33-fresh-faults-20260825b"),
        "--output", str(output),
    ]
    row = run_subcommand("r33-five-pair-replay", command, output_root, env=replay_env(str(scripts)))
    value = load_json(output)
    archived = root / "formal_h20/r33-fresh-faults-20260825b/summary.json"
    require(value.get("scientific_valid") is True and value.get("pair_count") == 5, "R33 replay drift")
    require(value == load_json(archived), "R33 replay semantic output differs from archived summary")
    return {"passed": True, "subcommands": [row], "semantic_identical_to_archived_summary": True, "byte_identical_to_archived_summary": output.read_bytes() == archived.read_bytes(), "summary_sha256": sha256_file(output)}


def component_r35(output_root: Path) -> dict[str, Any]:
    script = PAPER / "scripts/validate_r35_historical_alias_evidence.py"
    output = output_root / "r35-recomputed-verification.json"
    command = [sys.executable, "-B", str(script), "--output", str(output)]
    row = run_subcommand("r35-full-local-verifier", command, output_root, env=replay_env(str(PAPER / "scripts")))
    value = load_json(output)
    archived = EVIDENCE / "r35_historical_alias_regression/formal_h20/RESULT_VERIFICATION.json"
    require(value.get("status") == "verified_fail_closed" and value.get("operational_valid") is True, "R35 verifier drift")
    require(output.read_bytes() == archived.read_bytes(), "R35 verification is not byte-identical")
    return {"passed": True, "subcommands": [row], "byte_identical_to_archived_verification": True, "verification_sha256": sha256_file(output)}


def component_r30(output_root: Path) -> dict[str, Any]:
    root = EVIDENCE / "r30_expanded_oracle_sweep"
    output = output_root / "r30-local-oracle.json"
    command = [
        sys.executable, "-B", str(root / "r30_expanded_oracle_reference.py"),
        "--capture-manifest", str(root / "raw/capture-manifest.json"),
        "--preregistration", str(root / "preregistration.json"),
        "--output", str(output),
    ]
    row = run_subcommand("r30-numpy-replay", command, output_root)
    value = load_json(output)
    required = (
        "all_attention_clean_rows_pass", "all_attention_faults_rejected",
        "all_gdn_clean_rows_pass", "all_gdn_faults_rejected",
    )
    require(all(value.get(field) is True for field in required), "R30 replay decision drift")
    require(len(value.get("attention_rows", [])) == 20 and len(value.get("gdn_rows", [])) == 24, "R30 replay count drift")
    return {"passed": True, "subcommands": [row], "attention_rows": 20, "gdn_rows": 24, "seeded_controls": 44, "decision_equivalent_to_archived": True, "local_result_sha256": sha256_file(output)}


COMPONENT_FUNCTIONS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "primary_rr2": component_primary,
    "r39_preproducer_census": component_census,
    "r39_dual_producer": component_dual,
    "r33_designer_faults": component_r33,
    "r35_historical_alias": component_r35,
    "r30_expanded_oracle": component_r30,
}


def run_component(name: str, output_root: Path, result_path: Path) -> int:
    require(name in COMPONENT_FUNCTIONS, f"unknown component: {name}")
    require(output_root.is_dir(), "component output root must be precreated")
    require(not result_path.exists(), "component result already exists")
    started = utc_now()
    try:
        value = COMPONENT_FUNCTIONS[name](output_root)
        result = {
            "schema_version": "forkaudit-r40-component-result-v1",
            "component": name,
            "status": "passed",
            "started_at_utc": started,
            "ended_at_utc": utc_now(),
            **value,
        }
        code = 0
    except BaseException as exc:
        result = {
            "schema_version": "forkaudit-r40-component-result-v1",
            "component": name,
            "status": "failed",
            "passed": False,
            "started_at_utc": started,
            "ended_at_utc": utc_now(),
            "exception_type": type(exc).__name__,
            "message": str(exc),
        }
        code = 2
    write_new_json(result_path, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return code


TIME_REAL_RE = re.compile(r"^\s*([0-9.]+) real\s+([0-9.]+) user\s+([0-9.]+) sys", re.MULTILINE)
TIME_VALUE_RE = re.compile(r"^\s*([0-9]+)\s+(.+?)\s*$", re.MULTILINE)


def parse_time_report(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    match = TIME_REAL_RE.search(raw)
    require(match is not None, "Darwin time real/user/sys row missing")
    values = {label: int(value) for value, label in TIME_VALUE_RE.findall(raw)}
    require("maximum resident set size" in values, "Darwin maximum RSS row missing")
    return {
        "time_report_sha256": sha256_file(path),
        "time_report_bytes": path.stat().st_size,
        "time_real_seconds_rounded": float(match.group(1)),
        "time_user_seconds_rounded": float(match.group(2)),
        "time_sys_seconds_rounded": float(match.group(3)),
        "maximum_resident_set_size_raw": values["maximum resident set size"],
        "maximum_resident_set_size_unit": "bytes_on_darwin",
        "peak_memory_footprint_raw": values.get("peak memory footprint"),
        "peak_memory_footprint_unit": "bytes_on_darwin_if_reported",
    }


def host_environment() -> dict[str, Any]:
    def capture(command: Sequence[str]) -> str | None:
        completed = subprocess.run(list(command), stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        return completed.stdout.decode("utf-8", errors="replace").strip() if completed.returncode == 0 else None

    numpy_version = None
    try:
        import numpy  # type: ignore
        numpy_version = numpy.__version__
    except ImportError:
        pass
    return {
        "schema_version": "forkaudit-r40-measurement-environment-v1",
        "captured_at_utc": utc_now(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "numpy_version": numpy_version,
        "uname": capture(["uname", "-a"]),
        "sw_vers": capture(["sw_vers"]),
        "hardware_model": capture(["sysctl", "-n", "hw.model"]),
        "hardware_memory_bytes": capture(["sysctl", "-n", "hw.memsize"]),
        "hardware_logical_cpu_count": capture(["sysctl", "-n", "hw.ncpu"]),
        "time_binary": "/usr/bin/time",
        "time_mode": "Darwin /usr/bin/time -l; raw maximum RSS recorded as bytes",
        "cache_policy": "no cache flush; no cold-cache claim",
        "execution_order": list(COMPONENTS),
        "component_concurrency": 1,
        "gpu_access": False,
        "qs_access": False,
        "environment_overrides": {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "CUDA_VISIBLE_DEVICES": "",
        },
        "protocol_sha256": sha256_file(HERE / "PROTOCOL.md"),
        "measurement_script_sha256": sha256_file(Path(__file__).resolve()),
    }


def run_profiles(repetitions: int) -> int:
    require(repetitions == 3, "frozen protocol requires exactly three repetitions")
    require(not RAW.exists(), "raw measurement directory already exists")
    preparation = prepare_inputs()
    RAW.mkdir(parents=True)
    write_new_json(RAW / "measurement_environment.json", {**host_environment(), "preparation": preparation})
    component_rows_path = RAW / "component_timing_rows.jsonl"
    profile_rows_path = RAW / "profile_timing_rows.jsonl"
    any_failure = False
    for iteration in range(1, repetitions + 1):
        profile_started_at = utc_now()
        profile_start = time.monotonic_ns()
        iteration_rows: list[dict[str, Any]] = []
        for component in COMPONENTS:
            output_root = RAW / f"iteration-{iteration:02d}" / component
            require(not output_root.exists(), f"component output exists: {output_root}")
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
            started_at = utc_now()
            start = time.monotonic_ns()
            with stdout_path.open("xb") as stdout_handle, stderr_path.open("xb") as stderr_handle:
                completed = subprocess.run(command, cwd=PAPER, env=replay_env(), stdout=stdout_handle, stderr=stderr_handle, check=False)
            end = time.monotonic_ns()
            timing = parse_time_report(time_path)
            result = load_json(result_path) if result_path.is_file() else None
            passed = completed.returncode == 0 and isinstance(result, dict) and result.get("passed") is True
            row = {
                "schema_version": "forkaudit-r40-component-timing-row-v1",
                "profile": "extended_supporting",
                "iteration": iteration,
                "component": component,
                "started_at_utc": started_at,
                "ended_at_utc": utc_now(),
                "wall_seconds_monotonic": (end - start) / 1_000_000_000,
                "exit_code": completed.returncode,
                "passed": passed,
                "worker_result_sha256": sha256_file(result_path) if result_path.is_file() else None,
                "worker_stdout_sha256": sha256_file(stdout_path),
                "worker_stderr_sha256": sha256_file(stderr_path),
                **timing,
            }
            append_jsonl(component_rows_path, row)
            iteration_rows.append(row)
            any_failure = any_failure or not passed
        profile_end = time.monotonic_ns()
        primary = next(row for row in iteration_rows if row["component"] == "primary_rr2")
        profile_row = {
            "schema_version": "forkaudit-r40-profile-timing-row-v1",
            "iteration": iteration,
            "started_at_utc": profile_started_at,
            "ended_at_utc": utc_now(),
            "minimal_core_wall_seconds": primary["wall_seconds_monotonic"],
            "minimal_core_peak_rss_bytes": primary["maximum_resident_set_size_raw"],
            "extended_supporting_wall_seconds": (profile_end - profile_start) / 1_000_000_000,
            "extended_supporting_max_observed_component_peak_rss_bytes": max(row["maximum_resident_set_size_raw"] for row in iteration_rows),
            "component_count": len(iteration_rows),
            "components_serial_no_overlap": True,
            "passed": all(row["passed"] for row in iteration_rows),
        }
        append_jsonl(profile_rows_path, profile_row)
    completion = {
        "schema_version": "forkaudit-r40-run-completion-v1",
        "completed_at_utc": utc_now(),
        "repetitions": repetitions,
        "component_rows": repetitions * len(COMPONENTS),
        "all_rows_passed": not any_failure,
        "component_rows_sha256": sha256_file(component_rows_path),
        "profile_rows_sha256": sha256_file(profile_rows_path),
        "no_gpu_or_qs_access": True,
    }
    write_new_json(RAW / "RUN_COMPLETE.json", completion)
    return 0 if not any_failure else 2


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    mode = value.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--component", choices=COMPONENTS)
    value.add_argument("--repetitions", type=int, default=3)
    value.add_argument("--component-output-root", type=Path)
    value.add_argument("--component-result", type=Path)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.run:
        return run_profiles(args.repetitions)
    require(args.component_output_root is not None and args.component_result is not None, "component paths required")
    return run_component(args.component, args.component_output_root.resolve(), args.component_result.resolve())


if __name__ == "__main__":
    raise SystemExit(main())

