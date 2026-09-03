"""Aggregate the A4/A5 generation-length-sweep shards.

Deliberately torch-free: an archived shard can be re-validated on a laptop with
no CUDA and no Transformers install, which is what a reproducibility reviewer
will actually have.  Everything this script checks is checked again from the
raw per-row fields, not taken from the runner's own summaries.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from qcomem_eq3_accounting import (
    identity_violations,
    parse_arm_name,
    summarize_arm,
    validate_row,
)


SHARD_GLOB = "length-sweep-shard-*.json"
REFERENCE_ARM_CONFIG = "full-prefix-q16"


def load_shards(run_dir: Path, *, expected_shards: int | None = None) -> list[dict]:
    paths = sorted(run_dir.glob(SHARD_GLOB))
    if not paths:
        raise ValueError(f"no shards matching {SHARD_GLOB} under {run_dir}")
    if expected_shards is not None and len(paths) != expected_shards:
        raise ValueError(f"expected {expected_shards} shards, found {len(paths)}")
    shards = []
    for path in paths:
        shard = json.loads(path.read_text())
        shard["_path"] = str(path)
        shards.append(shard)
    return shards


def check_shard_consistency(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    statuses = {shard.get("status") for shard in shards}
    if statuses != {"completed"}:
        raise ValueError(f"not every shard completed: {sorted(statuses)}")
    declared_arms = [arm["arm"] for arm in shards[0].get("arms", ())]
    if not declared_arms:
        raise ValueError("the first shard declares no arms")
    for shard in shards:
        arms = [arm["arm"] for arm in shard.get("arms", ())]
        if arms != declared_arms:
            raise ValueError(f"{shard['_path']} declares a different arm list")
    protocols = {
        json.dumps(shard.get("protocol", {}), sort_keys=True) for shard in shards
    }
    if len(protocols) != 1:
        raise ValueError("shards disagree about the protocol block")
    ranks = sorted(int(shard["rank"]) for shard in shards)
    if ranks != list(range(len(shards))):
        raise ValueError(f"shard ranks are not a contiguous range: {ranks}")
    return {
        "ranks": ranks,
        "arms": declared_arms,
        "protocol": shards[0].get("protocol", {}),
    }


def check_gates(shards: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    failures = []
    summary: dict[str, Any] = {}
    for shard in shards:
        gates = shard.get("gates") or {}
        if not gates:
            failures.append(f"{shard['_path']} records no gates")
            continue
        for name, gate in gates.items():
            if not isinstance(gate, dict) or not gate.get("passed"):
                failures.append(f"{shard['_path']} gate {name} did not pass")
            summary.setdefault(name, 0)
            summary[name] += 1
    if failures:
        raise ValueError("; ".join(failures))
    return {"gates_seen": summary, "all_passed": True}


def check_rows(shards: Sequence[Mapping[str, Any]]) -> tuple[list[dict], dict]:
    all_rows: list[dict] = []
    problems: list[str] = []
    identity_failures: list[dict[str, Any]] = []
    for shard in shards:
        for row in shard.get("rows", ()):
            row = dict(row)
            row["_rank"] = shard["rank"]
            all_rows.append(row)
            row_problems = validate_row(row)
            if row_problems:
                problems.append(
                    f"rank {shard['rank']} {row.get('arm')} "
                    f"{row.get('workload_id')} repeat {row.get('repeat')}: "
                    f"{row_problems}"
                )
            components = (row.get("store_breakdown") or {}).get("components", ())
            for violation in identity_violations(components):
                identity_failures.append(
                    {
                        "arm": row.get("arm"),
                        "workload_id": row.get("workload_id"),
                        "repeat": row.get("repeat"),
                        **violation,
                    }
                )
    if not all_rows:
        raise ValueError("the shards contain no rows")
    if problems:
        raise ValueError("invalid rows: " + "; ".join(problems[:20]))
    if identity_failures:
        raise ValueError(
            "Eq. 3 byte identity violated: "
            + json.dumps(identity_failures[:10], indent=2)
        )
    return all_rows, {
        "rows": len(all_rows),
        "eq3_identity_violations": 0,
        "eq3_components_checked": sum(
            sum(
                1
                for component in (row.get("store_breakdown") or {}).get(
                    "components", ()
                )
                if component.get("eq3_identity_checked")
            )
            for row in all_rows
        ),
    }


def check_coverage(
    rows: Sequence[Mapping[str, Any]], declared_arms: Sequence[str]
) -> dict[str, Any]:
    cells: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in rows:
        cells[(row["workload_id"], int(row["repeat"]))].add(row["arm"])
    expected = set(declared_arms)
    missing = {
        f"{workload_id}#repeat{repeat}": sorted(expected - arms)
        for (workload_id, repeat), arms in cells.items()
        if arms != expected
    }
    if missing:
        raise ValueError(f"incomplete arm coverage: {json.dumps(missing, indent=2)}")
    return {
        "workloads": len({row["workload_id"] for row in rows}),
        "repeats": len({int(row["repeat"]) for row in rows}),
        "cells": len(cells),
        "arms_per_cell": len(expected),
    }


def _median(values: Sequence[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return float(statistics.median(values)) if values else None


def _mean(values: Sequence[float]) -> float | None:
    values = [float(value) for value in values if value is not None]
    return float(statistics.fmean(values)) if values else None


def paired_against_reference(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference_config: str = REFERENCE_ARM_CONFIG,
) -> list[dict[str, Any]]:
    """Pair every arm against the exact full-prefix arm at the same length.

    Pairing is by (workload, repeat, generation length), which is the unit the
    randomized interleave was defined over, so the comparison is within-item
    and within-repeat rather than across cohorts.
    """

    by_key: dict[tuple[str, int, int, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (
            row["workload_id"],
            int(row["repeat"]),
            int(row["max_new_tokens_requested"]),
            row["config"],
        )
        by_key[key] = row
    comparisons = []
    configs = sorted({row["config"] for row in rows} - {reference_config})
    lengths = sorted({int(row["max_new_tokens_requested"]) for row in rows})
    for config in configs:
        for length in lengths:
            pairs = []
            for row in rows:
                if row["config"] != config:
                    continue
                if int(row["max_new_tokens_requested"]) != length:
                    continue
                reference = by_key.get(
                    (
                        row["workload_id"],
                        int(row["repeat"]),
                        length,
                        reference_config,
                    )
                )
                if reference is None:
                    continue
                pairs.append((reference, row))
            if not pairs:
                continue
            comparisons.append(
                {
                    "config": config,
                    "reference_config": reference_config,
                    "max_new_tokens": length,
                    "pairs": len(pairs),
                    "store_reduction_ratio_median": _median(
                        [
                            reference["persistent_document_nbytes"]
                            / candidate["persistent_document_nbytes"]
                            for reference, candidate in pairs
                            if candidate["persistent_document_nbytes"]
                        ]
                    ),
                    "store_reduction_ratio_bf16_reference_median": _median(
                        [
                            reference["store_breakdown"]["bf16_reference_nbytes"]
                            / candidate["persistent_document_nbytes"]
                            for reference, candidate in pairs
                            if candidate["persistent_document_nbytes"]
                        ]
                    ),
                    "store_reduction_ratio_native_reference_median": _median(
                        [
                            reference["store_breakdown"][
                                "native_dtype_reference_nbytes"
                            ]
                            / candidate["persistent_document_nbytes"]
                            for reference, candidate in pairs
                            if candidate["persistent_document_nbytes"]
                        ]
                    ),
                    "ttft_ratio_median": _median(
                        [
                            candidate["ttft_seconds"] / reference["ttft_seconds"]
                            for reference, candidate in pairs
                            if reference["ttft_seconds"]
                        ]
                    ),
                    "decode_median_ratio_median": _median(
                        [
                            candidate["decode_latency"]["decode_seconds_median"]
                            / reference["decode_latency"]["decode_seconds_median"]
                            for reference, candidate in pairs
                            if reference["decode_latency"]["decode_seconds_median"]
                        ]
                    ),
                    "wall_tokens_per_second_ratio_median": _median(
                        [
                            candidate["throughput"]["wall_tokens_per_second"]
                            / reference["throughput"]["wall_tokens_per_second"]
                            for reference, candidate in pairs
                            if reference["throughput"]["wall_tokens_per_second"]
                        ]
                    ),
                    "online_tokens_per_second_ratio_median": _median(
                        [
                            candidate["throughput"]["online_tokens_per_second"]
                            / reference["throughput"]["online_tokens_per_second"]
                            for reference, candidate in pairs
                            if reference["throughput"]["online_tokens_per_second"]
                        ]
                    ),
                    "f1_delta_mean": _mean(
                        [
                            candidate["f1"] - reference["f1"]
                            for reference, candidate in pairs
                            if candidate.get("f1") is not None
                            and reference.get("f1") is not None
                        ]
                    ),
                }
            )
    return comparisons


def throughput_model_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """How wrong ``n / (TTFT + n * TPOT)`` is against the measured wall clock."""

    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["config"], int(row["max_new_tokens_requested"]))].append(row)
    audit = []
    for (config, length), group in sorted(grouped.items()):
        audit.append(
            {
                "config": config,
                "max_new_tokens": length,
                "rows": len(group),
                "measured_online_tokens_per_second_median": _median(
                    [row["throughput"]["online_tokens_per_second"] for row in group]
                ),
                "measured_wall_tokens_per_second_median": _median(
                    [row["throughput"]["wall_tokens_per_second"] for row in group]
                ),
                "reconstructed_tokens_per_second_median": _median(
                    [
                        row["throughput"]["reconstructed_tokens_per_second"]
                        for row in group
                    ]
                ),
                "reconstructed_over_measured_median": _median(
                    [row["throughput"]["reconstructed_over_measured"] for row in group]
                ),
                "instrumentation_overhead_seconds_median": _median(
                    [
                        row["throughput"]["instrumentation_overhead_seconds"]
                        for row in group
                    ]
                ),
                "decode_first_quarter_mean": _mean(
                    [
                        row["decode_latency"]["decode_seconds_first_quarter_mean"]
                        for row in group
                    ]
                ),
                "decode_last_quarter_mean": _mean(
                    [
                        row["decode_latency"]["decode_seconds_last_quarter_mean"]
                        for row in group
                    ]
                ),
            }
        )
    return audit


def store_reference_audit(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Per-arm native-dtype versus all-BF16 reference counts and their ratio."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    audit = []
    for config, group in sorted(grouped.items()):
        breakdowns = [row["store_breakdown"] for row in group]
        audit.append(
            {
                "config": config,
                "mode": group[0]["mode"],
                "rows": len(group),
                "packed_store_nbytes_median": _median(
                    [breakdown["packed_store_nbytes"] for breakdown in breakdowns]
                ),
                "native_dtype_reference_nbytes_median": _median(
                    [
                        breakdown["native_dtype_reference_nbytes"]
                        for breakdown in breakdowns
                    ]
                ),
                "bf16_reference_nbytes_median": _median(
                    [breakdown["bf16_reference_nbytes"] for breakdown in breakdowns]
                ),
                "native_dtype_ratio_median": _median(
                    [breakdown["native_dtype_ratio"] for breakdown in breakdowns]
                ),
                "bf16_ratio_median": _median(
                    [breakdown["bf16_ratio"] for breakdown in breakdowns]
                ),
                "dtype_inconsistent_components_max": max(
                    int(breakdown["dtype_inconsistent_components"])
                    for breakdown in breakdowns
                ),
                "state_types": sorted(
                    {
                        state_type
                        for breakdown in breakdowns
                        for state_type in breakdown["by_state_type"]
                    }
                ),
                "by_state_type_nbytes_median": {
                    state_type: _median(
                        [
                            breakdown["by_state_type"][state_type]["total_nbytes"]
                            for breakdown in breakdowns
                            if state_type in breakdown["by_state_type"]
                        ]
                    )
                    for state_type in sorted(
                        {
                            state_type
                            for breakdown in breakdowns
                            for state_type in breakdown["by_state_type"]
                        }
                    )
                },
                "reconciles_with_frozen_accountant": all(
                    breakdown.get("reconciliation", {}).get("matches", True)
                    for breakdown in breakdowns
                ),
            }
        )
    return audit


def aggregate(
    run_dir: Path,
    *,
    expected_shards: int | None = None,
    expected_arms: Sequence[str] | None = None,
) -> dict[str, Any]:
    shards = load_shards(run_dir, expected_shards=expected_shards)
    consistency = check_shard_consistency(shards)
    if expected_arms is not None and list(expected_arms) != consistency["arms"]:
        raise ValueError(
            f"arms {consistency['arms']} do not match frozen {list(expected_arms)}"
        )
    gates = check_gates(shards)
    rows, row_audit = check_rows(shards)
    coverage = check_coverage(rows, consistency["arms"])

    by_arm: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_dataset_arm: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_arm[row["arm"]].append(row)
        by_dataset_arm[(row.get("dataset"), row["arm"])].append(row)
    return {
        "run_dir": str(run_dir),
        "shards": [shard["_path"] for shard in shards],
        "consistency": consistency,
        "gates": gates,
        "row_audit": row_audit,
        "coverage": coverage,
        "arm_summary": [summarize_arm(by_arm[arm]) for arm in sorted(by_arm)],
        "arm_summary_by_dataset": [
            {"dataset": dataset, **summarize_arm(group)}
            for (dataset, _arm), group in sorted(
                by_dataset_arm.items(), key=lambda item: (str(item[0][0]), item[0][1])
            )
        ],
        "paired_against_reference": paired_against_reference(rows),
        "throughput_model_audit": throughput_model_audit(rows),
        "store_reference_audit": store_reference_audit(rows),
        "lengths": sorted({int(parse_arm_name(arm)[1]) for arm in consistency["arms"]}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate and re-validate A4/A5 length-sweep shards"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--expected-arms", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(
        args.run_dir,
        expected_shards=args.expected_shards,
        expected_arms=args.expected_arms,
    )
    destination = args.output or (args.run_dir / "length-sweep-aggregate.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(destination)
    print(json.dumps(result["coverage"], indent=2))
    print(f"SAVED {destination}")


if __name__ == "__main__":
    main()
