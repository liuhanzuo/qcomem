#!/usr/bin/env python3
"""Aggregate only completed R40 local measurements and keep blockers explicit."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values: Iterable[float | int]) -> dict[str, float]:
    rows = [float(value) for value in values]
    if not rows:
        raise ValueError("empty aggregate")
    return {
        "count": float(len(rows)),
        "minimum": min(rows),
        "median": statistics.median(rows),
        "maximum": max(rows),
    }


def write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def build_aggregate() -> dict[str, Any]:
    completion_path = RAW / "RUN_COMPLETE.json"
    component_path = RAW / "component_timing_rows.jsonl"
    profile_path = RAW / "profile_timing_rows.jsonl"
    environment_path = RAW / "measurement_environment.json"
    inventory_path = HERE / "inventory.json"
    for path in (completion_path, component_path, profile_path, environment_path, inventory_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    if completion.get("all_rows_passed") is not True:
        raise ValueError("R40 run is incomplete or contains a failed row")
    component_rows = read_jsonl(component_path)
    profile_rows = read_jsonl(profile_path)
    if len(component_rows) != 18 or len(profile_rows) != 3:
        raise ValueError("frozen 18-component/3-profile cardinality mismatch")
    if not all(row.get("passed") is True for row in component_rows + profile_rows):
        raise ValueError("failed timing row")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    by_component: dict[str, list[dict[str, Any]]] = {}
    for row in component_rows:
        by_component.setdefault(str(row["component"]), []).append(row)
    components = {
        name: {
            "wall_seconds": stats(row["wall_seconds_monotonic"] for row in rows),
            "peak_rss_bytes": stats(row["maximum_resident_set_size_raw"] for row in rows),
            "all_exit_zero_and_passed": all(row["exit_code"] == 0 and row["passed"] for row in rows),
            "raw_iterations": [row["iteration"] for row in rows],
        }
        for name, rows in sorted(by_component.items())
    }
    packages = inventory["packages"]
    measured_package_keys = (
        "primary_rr2",
        "r39_preproducer_census",
        "r39_dual_producer_repeat",
        "r33_designer_executor_faults",
        "r35_historical_alias",
        "r30_expanded_oracle",
    )
    extended_local_source_bytes = sum(packages[key]["footprint"]["logical_bytes"] for key in measured_package_keys)
    extended_local_source_files = sum(packages[key]["footprint"]["file_count"] for key in measured_package_keys)
    return {
        "schema_version": "forkaudit-r40-ci-cost-aggregate-v1",
        "status": "verified_local_cpu_replay_measurement",
        "profile_repetitions": 3,
        "host_scope": "one Apple host; no cache flush; consecutive local CPU replay",
        "profiles": {
            "minimal_core": {
                "definition": "primary RR2 one-command replay only",
                "wall_seconds": stats(row["minimal_core_wall_seconds"] for row in profile_rows),
                "peak_rss_bytes": stats(row["minimal_core_peak_rss_bytes"] for row in profile_rows),
                "local_package_file_count": packages["primary_rr2"]["footprint"]["file_count"],
                "local_package_logical_bytes": packages["primary_rr2"]["footprint"]["logical_bytes"],
                "raw_trace_file_count": packages["primary_rr2"]["trace"]["file_count"],
                "raw_trace_logical_bytes": packages["primary_rr2"]["trace"]["logical_bytes"],
            },
            "extended_supporting": {
                "definition": "minimal core plus five locally complete supporting replays, serial",
                "wall_seconds": stats(row["extended_supporting_wall_seconds"] for row in profile_rows),
                "max_observed_component_peak_rss_bytes": stats(row["extended_supporting_max_observed_component_peak_rss_bytes"] for row in profile_rows),
                "profile_peak_rss_boundary": "maximum separately measured component RSS; not a sampled whole-profile process-tree peak",
                "local_source_root_file_count_sum_no_cross_root_overlap": extended_local_source_files,
                "local_source_root_logical_bytes_sum_no_cross_root_overlap": extended_local_source_bytes,
                "components": list(by_component),
            },
        },
        "components": components,
        "blocked_or_unmeasured_replays": {
            "r39_falcon_h1_v2": "formal v2 verifier source absent locally; older v1 source hash differs",
            "r39_pdf_only_blind_faults_full_replay": "352 full-vocabulary FP32 sidecars absent from local metadata package",
        },
        "explicitly_unmeasured_costs": inventory["unmeasured"],
        "prohibitions": {
            "local_cpu_replay_is_h20_capture_overhead": False,
            "mtime_is_capture_duration": False,
            "metadata_only_blind_fault_aggregation_is_full_pair_replay": False,
            "falcon_v1_verifier_substituted_for_v2": False,
        },
        "raw_bindings": {
            "component_rows_sha256": sha256_file(component_path),
            "profile_rows_sha256": sha256_file(profile_path),
            "environment_sha256": sha256_file(environment_path),
            "inventory_sha256": sha256_file(inventory_path),
            "run_completion_sha256": sha256_file(completion_path),
        },
    }


def render_summary(value: dict[str, Any]) -> str:
    minimal = value["profiles"]["minimal_core"]
    extended = value["profiles"]["extended_supporting"]
    lines = [
        "# R40 measured local replay cost",
        "",
        "All figures below are local CPU replay measurements on one Apple host. They are not H20 capture overhead.",
        "",
        "| Profile | Repeats | Wall median (range), s | Peak-RSS median (range), MiB | Local logical bytes |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, row, rss_key, bytes_key in (
        ("minimal core", minimal, "peak_rss_bytes", "local_package_logical_bytes"),
        ("extended supporting", extended, "max_observed_component_peak_rss_bytes", "local_source_root_logical_bytes_sum_no_cross_root_overlap"),
    ):
        wall = row["wall_seconds"]
        rss = row[rss_key]
        lines.append(
            f"| {name} | 3 | {wall['median']:.3f} ({wall['minimum']:.3f}--{wall['maximum']:.3f}) | "
            f"{rss['median'] / 1048576:.2f} ({rss['minimum'] / 1048576:.2f}--{rss['maximum'] / 1048576:.2f}) | "
            f"{row[bytes_key]} |"
        )
    lines.extend([
        "",
        "The extended RSS row is the maximum separately observed component peak in each serial repetition, not a whole-profile process-tree sample.",
        "",
        "Falcon-H1 v2 replay time remains unmeasured because its hash-bound v2 verifier source is absent locally. The R39 PDF-only blind-fault full replay remains unmeasured because the local metadata package omits 352 bound FP32 sidecars.",
        "",
        "H20 capture time, GPU perturbation, a matched current uninstrumented baseline, cold download/extraction cost, and engineering effort remain unmeasured.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    aggregate_path = HERE / "aggregate.json"
    summary_path = HERE / "SUMMARY.md"
    if aggregate_path.exists() or summary_path.exists():
        raise FileExistsError("refusing to replace aggregate outputs")
    value = build_aggregate()
    write_new(aggregate_path, json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    write_new(summary_path, render_summary(value).encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

