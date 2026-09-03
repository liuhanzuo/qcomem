#!/usr/bin/env python3
"""Aggregate the successful R40 Attempt B and retain Attempt A as a blocker row."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import statistics
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
RAW_A = HERE / "raw"
RAW_B = HERE / "raw_attempt_b"
PREPARED = HERE / "prepared_inputs"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stats(values: Iterable[float | int]) -> dict[str, float | int]:
    rows = [float(value) for value in values]
    require(bool(rows), "empty aggregate")
    return {
        "count": len(rows),
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


def validate_attempt_a() -> dict[str, Any]:
    completion_path = RAW_A / "RUN_COMPLETE.json"
    component_path = RAW_A / "component_timing_rows.jsonl"
    profile_path = RAW_A / "profile_timing_rows.jsonl"
    completion = load_json(completion_path)
    components = read_jsonl(component_path)
    profiles = read_jsonl(profile_path)
    require(completion.get("all_rows_passed") is False, "Attempt A failure record drift")
    require(len(components) == 18 and len(profiles) == 3, "Attempt A cardinality drift")
    failed = [row for row in components if not row.get("passed")]
    passed = [row for row in components if row.get("passed")]
    require(len(failed) == 3 and all(row["component"] == "primary_rr2" for row in failed), "Attempt A failure pattern drift")
    require(len(passed) == 15 and all(row["component"] != "primary_rr2" for row in passed), "Attempt A supporting pattern drift")
    messages = []
    for iteration in range(1, 4):
        result = load_json(RAW_A / f"iteration-{iteration:02d}/primary_rr2/component-result.json")
        messages.append({
            "iteration": iteration,
            "exception_type": result.get("exception_type"),
            "message": result.get("message"),
        })
    return {
        "status": "invalidated_before_primary_scientific_replay",
        "component_rows": 18,
        "passed_supporting_rows": 15,
        "failed_primary_rows": 3,
        "primary_failures": messages,
        "retention": "retained verbatim; not pooled into Attempt B timing",
        "bindings": {
            "component_rows_sha256": sha256_file(component_path),
            "profile_rows_sha256": sha256_file(profile_path),
            "completion_sha256": sha256_file(completion_path),
        },
    }


def primary_artifact(inventory: dict[str, Any], receipt: dict[str, Any]) -> dict[str, Any]:
    manifest_path = PREPARED / "primary_manifest_view/MANIFEST.json"
    sidecar_path = PREPARED / "primary_manifest_view/MANIFEST.sha256"
    manifest = load_json(manifest_path)
    require(manifest["file_count"] == receipt["declared_file_count"] == 628, "primary manifest file count drift")
    require(manifest["total_bytes"] == receipt["declared_logical_bytes"] == 892144066, "primary manifest byte count drift")
    require(sha256_file(manifest_path) == receipt["manifest_sha256"], "prepared manifest binding drift")
    distribution_bytes = manifest["total_bytes"] + manifest_path.stat().st_size + sidecar_path.stat().st_size
    distribution_files = manifest["file_count"] + 2
    observed = inventory["packages"]["primary_rr2"]["footprint"]
    trace = inventory["packages"]["primary_rr2"]["trace"]
    require(
        observed["file_count"] == distribution_files + receipt["source_unmanifested_file_count"],
        "observed primary source file count does not explain contamination",
    )
    require(trace["file_count"] == 536 and trace["logical_bytes"] == 888785811, "primary trace inventory drift")
    return {
        "manifest_payload_file_count": manifest["file_count"],
        "manifest_payload_logical_bytes": manifest["total_bytes"],
        "distribution_control_file_count": 2,
        "distribution_control_logical_bytes": manifest_path.stat().st_size + sidecar_path.stat().st_size,
        "clean_distribution_file_count": distribution_files,
        "clean_distribution_logical_bytes": distribution_bytes,
        "raw_trace_file_count": trace["file_count"],
        "raw_trace_logical_bytes": trace["logical_bytes"],
        "raw_trace_share_of_manifest_payload": trace["logical_bytes"] / manifest["total_bytes"],
        "observed_source_tree_file_count_with_unmanifested_pyc": observed["file_count"],
        "observed_source_tree_logical_bytes_with_unmanifested_pyc": observed["logical_bytes"],
        "source_unmanifested_file_count": receipt["source_unmanifested_file_count"],
        "source_unmanifested_paths": receipt["source_unmanifested_paths"],
    }


def build_aggregate() -> dict[str, Any]:
    required = (
        RAW_B / "RUN_COMPLETE.json",
        RAW_B / "component_timing_rows.jsonl",
        RAW_B / "profile_timing_rows.jsonl",
        RAW_B / "measurement_environment.json",
        HERE / "inventory.json",
        PREPARED / "PRIMARY_MANIFEST_VIEW.json",
        HERE / "AMENDMENT_ATTEMPT_B.md",
    )
    for path in required:
        require(path.is_file(), f"required input absent: {path}")
    completion = load_json(RAW_B / "RUN_COMPLETE.json")
    components = read_jsonl(RAW_B / "component_timing_rows.jsonl")
    profiles = read_jsonl(RAW_B / "profile_timing_rows.jsonl")
    environment = load_json(RAW_B / "measurement_environment.json")
    inventory = load_json(HERE / "inventory.json")
    receipt = load_json(PREPARED / "PRIMARY_MANIFEST_VIEW.json")
    require(completion.get("all_rows_passed") is True, "Attempt B did not complete successfully")
    require(completion.get("no_gpu_or_qs_access") is True, "GPU/QS boundary drift")
    require(len(components) == 18 and len(profiles) == 3, "Attempt B 18-component/3-profile cardinality mismatch")
    require(all(row.get("passed") is True and row.get("exit_code") == 0 for row in components), "failed Attempt B component row")
    require(all(row.get("passed") is True for row in profiles), "failed Attempt B profile row")
    require(environment["measurement_script_sha256"] == sha256_file(HERE / "measure_replays_attempt_b.py"), "Attempt B script changed after timing")
    require(environment["protocol_amendment_sha256"] == sha256_file(HERE / "AMENDMENT_ATTEMPT_B.md"), "Attempt B amendment changed after timing")

    component_order = [
        "primary_rr2",
        "r39_preproducer_census",
        "r39_dual_producer",
        "r33_designer_faults",
        "r35_historical_alias",
        "r30_expanded_oracle",
    ]
    by_component: dict[str, list[dict[str, Any]]] = {}
    for row in components:
        by_component.setdefault(str(row["component"]), []).append(row)
    require(list(by_component) == component_order, "Attempt B component order drift")
    require(all(len(rows) == 3 for rows in by_component.values()), "Attempt B component repetition drift")
    component_aggregates = {
        name: {
            "wall_seconds": stats(row["wall_seconds_monotonic"] for row in rows),
            "peak_rss_bytes": stats(row["maximum_resident_set_size_raw"] for row in rows),
            "iterations": [row["iteration"] for row in rows],
            "all_exit_zero_and_passed": True,
        }
        for name, rows in by_component.items()
    }

    artifact = primary_artifact(inventory, receipt)
    supporting_keys = (
        "r39_preproducer_census",
        "r39_dual_producer_repeat",
        "r33_designer_executor_faults",
        "r35_historical_alias",
        "r30_expanded_oracle",
    )
    supporting_root_bytes = sum(inventory["packages"][key]["footprint"]["logical_bytes"] for key in supporting_keys)
    supporting_root_files = sum(inventory["packages"][key]["footprint"]["file_count"] for key in supporting_keys)
    minimal_wall = [row["minimal_core_wall_seconds"] for row in profiles]
    extended_wall = [row["extended_supporting_wall_seconds"] for row in profiles]
    return {
        "schema_version": "forkaudit-r40-ci-cost-aggregate-attempt-b-v1",
        "status": "verified_local_cpu_replay_measurement",
        "measurement_scope": "three consecutive serial local CPU replays on one Apple host; no cache flush; no cold-cache claim",
        "attempt_a": validate_attempt_a(),
        "attempt_b": {
            "status": "18_of_18_component_rows_and_3_of_3_profiles_passed",
            "primary_input": "manifest-only copy built and verified outside timing; source evidence unchanged",
            "primary_preparation_wall_seconds_excluded_from_replay_timing": receipt["preparation_wall_seconds"],
            "profile_repetitions": 3,
        },
        "profiles": {
            "minimal_core": {
                "definition": "one invocation of primary RR2 replay/run_replay.sh from the verified manifest-only view",
                "wall_seconds": stats(minimal_wall),
                "peak_rss_bytes": stats(row["minimal_core_peak_rss_bytes"] for row in profiles),
                "artifact": artifact,
            },
            "extended_supporting": {
                "definition": "minimal core plus five locally complete supporting replays, serial",
                "wall_seconds": stats(extended_wall),
                "incremental_after_minimal_core_wall_seconds": stats(
                    extended - minimal for extended, minimal in zip(extended_wall, minimal_wall)
                ),
                "max_observed_component_peak_rss_bytes": stats(
                    row["extended_supporting_max_observed_component_peak_rss_bytes"] for row in profiles
                ),
                "profile_peak_rss_boundary": "maximum separately measured component RSS in each repetition; not a sampled whole-profile process-tree peak",
                "clean_primary_distribution_plus_supporting_audited_roots_file_count": artifact["clean_distribution_file_count"] + supporting_root_files,
                "clean_primary_distribution_plus_supporting_audited_roots_logical_bytes": artifact["clean_distribution_logical_bytes"] + supporting_root_bytes,
                "root_sum_boundary": "sum of disjoint local evidence roots; excludes R40 prepared copies and does not imply a compressed upload size",
                "components": component_order,
            },
        },
        "components": component_aggregates,
        "inventory": {
            "inventory_sha256": sha256_file(HERE / "inventory.json"),
            "packages_audited": list(inventory["packages"]),
            "log_timestamp_interpretation": inventory["timestamp_boundary"],
        },
        "blocked_or_unmeasured_replays": {
            "r39_falcon_h1_v2": "formal v2 verifier source absent locally; older v1 source hash differs and was not substituted",
            "r39_pdf_only_blind_faults_full_replay": "352 bound full-vocabulary FP32 sidecars absent locally; metadata aggregation was not timed as full replay",
        },
        "explicitly_unmeasured_costs": inventory["unmeasured"],
        "prohibitions": {
            "local_cpu_replay_is_h20_capture_overhead": False,
            "local_cpu_replay_is_gpu_perturbation": False,
            "mtime_is_capture_duration": False,
            "engineering_effort_was_inferred": False,
            "metadata_only_blind_fault_aggregation_is_full_pair_replay": False,
            "falcon_v1_verifier_substituted_for_v2": False,
        },
        "raw_bindings": {
            "attempt_b_component_rows_sha256": sha256_file(RAW_B / "component_timing_rows.jsonl"),
            "attempt_b_profile_rows_sha256": sha256_file(RAW_B / "profile_timing_rows.jsonl"),
            "attempt_b_environment_sha256": sha256_file(RAW_B / "measurement_environment.json"),
            "attempt_b_completion_sha256": sha256_file(RAW_B / "RUN_COMPLETE.json"),
            "primary_manifest_view_receipt_sha256": sha256_file(PREPARED / "PRIMARY_MANIFEST_VIEW.json"),
            "inventory_sha256": sha256_file(HERE / "inventory.json"),
        },
    }


def range_string(row: dict[str, Any], digits: int = 3) -> str:
    return f"{row['median']:.{digits}f} ({row['minimum']:.{digits}f}--{row['maximum']:.{digits}f})"


def render_summary(value: dict[str, Any]) -> str:
    minimal = value["profiles"]["minimal_core"]
    extended = value["profiles"]["extended_supporting"]
    artifact = minimal["artifact"]
    lines = [
        "# R40 measured local replay and CI-storage cost",
        "",
        "Attempt B passed all 18 component rows and all three serial profiles. All timings are local CPU replay measurements on one Apple host; they are not H20 capture overhead, GPU perturbation, or an uninstrumented-baseline delta.",
        "",
        "| Profile | Repeats | Wall median (range), s | Peak RSS median (range), MiB | Audited local logical bytes |",
        "|---|---:|---:|---:|---:|",
        (
            f"| minimal core | 3 | {range_string(minimal['wall_seconds'])} | "
            f"{range_string({key: value_ / 1048576 for key, value_ in minimal['peak_rss_bytes'].items() if key != 'count'})} | "
            f"{artifact['clean_distribution_logical_bytes']} |"
        ),
        (
            f"| extended supporting | 3 | {range_string(extended['wall_seconds'])} | "
            f"{range_string({key: value_ / 1048576 for key, value_ in extended['max_observed_component_peak_rss_bytes'].items() if key != 'count'})} | "
            f"{extended['clean_primary_distribution_plus_supporting_audited_roots_logical_bytes']} |"
        ),
        "",
        (
            f"The supporting profile added a measured median {extended['incremental_after_minimal_core_wall_seconds']['median']:.3f} s "
            f"after the minimal replay. Its RSS value is the maximum separately observed component peak, not a sampled whole-profile process-tree peak."
        ),
        "",
        "## Artifact accounting",
        "",
        (
            f"The clean primary distribution contains {artifact['manifest_payload_file_count']} manifest payload files plus two manifest controls: "
            f"{artifact['clean_distribution_file_count']} files and {artifact['clean_distribution_logical_bytes']} logical bytes "
            f"({artifact['clean_distribution_logical_bytes'] / 1048576:.2f} MiB). "
            f"Its 536 raw trace artifacts occupy {artifact['raw_trace_logical_bytes']} bytes "
            f"({artifact['raw_trace_logical_bytes'] / 1048576:.2f} MiB; {100 * artifact['raw_trace_share_of_manifest_payload']:.2f}% of manifest payload bytes)."
        ),
        "",
        (
            f"Attempt A remains retained and invalid: all 15 supporting rows passed, but all three primary rows stopped before replay because the source tree had "
            f"{artifact['source_unmanifested_file_count']} unmanifested `.pyc` files. Attempt B copied and verified only manifest-listed bytes without modifying the source evidence; its one-time {value['attempt_b']['primary_preparation_wall_seconds_excluded_from_replay_timing']:.3f} s preparation was excluded from replay timing."
        ),
        "",
        "## Still unmeasured",
        "",
        "Current-package H20 capture wall time, GPU slowdown/perturbation, a matched current uninstrumented baseline, cold download/extraction cost, and engineering/adoption effort remain unmeasured. Falcon-H1 v2 replay is blocked by its absent hash-bound v2 verifier source; the PDF-only blind-fault full replay is blocked by 352 absent FP32 sidecars. Filesystem and archive timestamps are retained only as provenance and never converted to duration.",
        "",
    ]
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

