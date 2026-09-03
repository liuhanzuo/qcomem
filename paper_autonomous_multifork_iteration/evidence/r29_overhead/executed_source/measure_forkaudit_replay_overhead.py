#!/usr/bin/env python3
"""Measure frozen ForkAudit reviewer-package footprint and replay time."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_footprint(root: Path) -> dict[str, int]:
    files = 0
    logical_bytes = 0
    for path in sorted(root.rglob("*")):
        if path.is_file() and not path.is_symlink():
            stat = path.stat()
            files += 1
            logical_bytes += int(stat.st_size)
    return {"regular_files": files, "logical_bytes": logical_bytes}


def cpu_brand() -> str:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return result.stdout.strip()
    return platform.processor() or "unknown"


def physical_memory_bytes() -> int | None:
    if sys.platform == "darwin":
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return int(result.stdout.strip())
    return None


def run_once(entrypoint: Path, package_root: Path, *, measured: bool) -> dict[str, Any]:
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = dt.datetime.now(dt.timezone.utc)
    start_ns = __import__("time").perf_counter_ns()
    process = subprocess.run(
        [str(entrypoint)],
        cwd=package_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    end_ns = __import__("time").perf_counter_ns()
    ended = dt.datetime.now(dt.timezone.utc)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    row = {
        "measured": measured,
        "started_at_utc": started.isoformat(),
        "ended_at_utc": ended.isoformat(),
        "exit_code": int(process.returncode),
        "wall_seconds": (end_ns - start_ns) / 1_000_000_000,
        "child_user_seconds": float(after.ru_utime - before.ru_utime),
        "child_system_seconds": float(after.ru_stime - before.ru_stime),
        "stdout_sha256": sha256_bytes(process.stdout),
        "stderr_sha256": sha256_bytes(process.stderr),
        "stdout_bytes": len(process.stdout),
        "stderr_bytes": len(process.stderr),
    }
    if process.returncode != 0:
        row["stderr_tail"] = process.stderr.decode("utf-8", errors="replace")[-2000:]
    return row


def median(rows: list[dict[str, Any]], field: str) -> float:
    return float(statistics.median(float(row[field]) for row in rows))


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--package-root", type=Path, required=True)
    value.add_argument("--entrypoint", type=Path, required=True)
    value.add_argument("--expected-entrypoint-sha256", required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--warmups", type=int, default=1)
    value.add_argument("--repetitions", type=int, default=5)
    return value


def main() -> int:
    args = parser().parse_args()
    package_root = args.package_root.resolve()
    entrypoint = args.entrypoint.resolve()
    output = args.output.resolve()
    if args.warmups != 1 or args.repetitions != 5:
        raise SystemExit("the frozen protocol requires one warmup and five repetitions")
    if output.exists():
        raise SystemExit(f"refusing to overwrite result: {output}")
    if not package_root.is_dir() or not entrypoint.is_file():
        raise SystemExit("package root or replay entrypoint is missing")
    observed_entrypoint_sha = sha256_file(entrypoint)
    if observed_entrypoint_sha != args.expected_entrypoint_sha256:
        raise SystemExit("replay entrypoint SHA-256 differs from preregistration")

    footprint_before = package_footprint(package_root)
    warmups = [run_once(entrypoint, package_root, measured=False)]
    rows = [run_once(entrypoint, package_root, measured=True) for _ in range(5)]
    footprint_after = package_footprint(package_root)
    valid = (
        all(row["exit_code"] == 0 for row in warmups + rows)
        and footprint_before == footprint_after
    )
    wall_values = [float(row["wall_seconds"]) for row in rows]
    result = {
        "schema_version": "forkaudit-r29-local-replay-overhead-result-v1",
        "evidence_id": "E-R29-LOCAL-REPLAY-OVERHEAD",
        "scientific_run_valid": valid,
        "package_root": str(args.package_root),
        "entrypoint": str(args.entrypoint),
        "entrypoint_sha256": observed_entrypoint_sha,
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_brand": cpu_brand(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": physical_memory_bytes(),
            "python": platform.python_version(),
        },
        "cache_protocol": "one unreported warmup followed by five measured warm-cache runs",
        "footprint_before": {
            **footprint_before,
            "logical_mib": footprint_before["logical_bytes"] / (1024 * 1024),
        },
        "footprint_after": footprint_after,
        "warmup_rows": warmups,
        "measured_rows": rows,
        "summary": {
            "repetitions": 5,
            "median_wall_seconds": median(rows, "wall_seconds"),
            "min_wall_seconds": min(wall_values),
            "max_wall_seconds": max(wall_values),
            "median_child_user_seconds": median(rows, "child_user_seconds"),
            "median_child_system_seconds": median(rows, "child_system_seconds"),
        },
        "claim_boundary": "Warm-cache CPU-only offline replay and logical package footprint only; no live-capture, production-latency, download, extraction, or engineering-effort claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    print(json.dumps({"valid": valid, **result["summary"], **result["footprint_before"]}))
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
