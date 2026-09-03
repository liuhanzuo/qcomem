from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_downstream import atomic_json


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def describe(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return {
        "count": len(values),
        "median": statistics.median(values) if values else None,
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p05": percentile(values, 0.05),
        "p95": percentile(values, 0.95),
    }


def direct_cow_gate_passed(gate: dict[str, Any]) -> bool:
    """Fail closed when a COW run omits or bypasses its same-source gate."""

    if gate.get("fork_strategy") != "paged-cow-staging":
        return True
    direct = gate.get("cow_vs_deep_clone_q16")
    if not isinstance(direct, dict):
        return False
    comparison = direct.get("comparison")
    immutable = direct.get("cow_immutable_audit")
    return bool(
        direct.get("passed")
        and direct.get("same_persistent_source")
        and direct.get("cow_was_exercised")
        and direct.get("strategy_effective") == "paged-cow-staging"
        and direct.get("source_after_eager", {}).get("verified")
        and direct.get("source_after_cow", {}).get("verified")
        and isinstance(immutable, dict)
        and immutable.get("verified")
        and isinstance(comparison, dict)
        and comparison.get("passed")
        and comparison.get("token_sequence_exact")
        and comparison.get("logits_bitwise_exact")
    )


def aggregate(run_dir: Path, expected_shards: int | None = None) -> dict[str, Any]:
    paths = sorted(run_dir.glob("deployment-shard-*.json"))
    shards = [json.loads(path.read_text()) for path in paths]
    if not shards:
        raise ValueError("no deployment shards found")
    if expected_shards is None:
        expected_shards = int(shards[0]["world_size"])
    if len(shards) != expected_shards:
        raise ValueError(f"expected {expected_shards} shards, found {len(shards)}")
    if any(shard.get("status") != "completed" for shard in shards):
        statuses = {shard.get("rank"): shard.get("status") for shard in shards}
        raise ValueError(f"not all deployment shards completed: {statuses}")
    if not all(shard["exactness_gate"]["passed"] for shard in shards):
        raise ValueError("at least one exactness gate failed")
    if not all(direct_cow_gate_passed(shard["exactness_gate"]) for shard in shards):
        raise ValueError(
            "at least one COW shard omitted or failed the same-source direct gate"
        )

    rows = [row for shard in shards for row in shard["rows"]]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    grouped_dataset: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    grouped_overall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (row["dataset"], row["config"], int(row["document_tokens"]))
        ].append(row)
        grouped_dataset[(row["dataset"], row["config"])].append(row)
        grouped_overall[row["config"]].append(row)

    fields = (
        "write_build_seconds",
        "persistent_document_nbytes",
        "persistent_materialized_staging_nbytes",
        "persistent_total_resident_nbytes",
        "capacity_document_denominator_nbytes",
        "selected_fork_active_state_peak_nbytes",
        "selected_fork_active_state_steady_nbytes",
        "cow_initial_shared_nbytes",
        "cow_initial_private_nbytes",
        "cow_after_query_shared_nbytes",
        "cow_after_query_private_nbytes",
        "cow_final_shared_nbytes",
        "cow_final_private_nbytes",
        "decode_kv_peak_nbytes",
        "decode_kv_steady_nbytes",
        "cuda_peak_allocated_bytes",
        "cuda_peak_reserved_bytes",
        "steady_state_cuda_allocated_bytes",
        "steady_state_cuda_reserved_bytes",
        "nvml_sampled_peak_process_bytes",
        "steady_state_nvml_process_bytes",
        "ttft_seconds",
        "median_tpot_seconds",
        "throughput_tokens_per_second",
        "max_resident_documents_store_only",
        "max_resident_documents_with_one_active_request",
        "f1",
    )
    summary_by_length = []
    for (dataset, config, tokens), matching in sorted(grouped.items()):
        summary_by_length.append(
            {
                "dataset": dataset,
                "config": config,
                "document_tokens": tokens,
                "measurements": len(matching),
                **{field: describe(matching, field) for field in fields},
            }
        )
    summary_by_dataset_config = [
        {
            "dataset": dataset,
            "config": config,
            "workloads": len({row["workload_id"] for row in matching}),
            "measurements": len(matching),
            **{field: describe(matching, field) for field in fields},
        }
        for (dataset, config), matching in sorted(grouped_dataset.items())
    ]
    summary_overall_config = [
        {
            "dataset": "overall",
            "config": config,
            "workloads": len({row["workload_id"] for row in matching}),
            "measurements": len(matching),
            **{field: describe(matching, field) for field in fields},
        }
        for config, matching in sorted(grouped_overall.items())
    ]

    by_key = {
        (row["workload_id"], row["repeat"], row["config"]): row for row in rows
    }
    qcomem_configs = sorted(
        {row["config"] for row in rows if row["config"].startswith("qcomem-")}
    )
    paired = []
    for config in qcomem_configs:
        pairs = []
        for row in rows:
            if row["config"] != config:
                continue
            baseline = by_key.get(
                (row["workload_id"], row["repeat"], "full-prefix-q16")
            ) or by_key.get((row["workload_id"], row["repeat"], "full-prefix"))
            if baseline is None:
                continue
            pairs.append(
                {
                    "persistent_compression_vs_full_prefix": (
                        baseline["persistent_document_nbytes"]
                        / row["persistent_document_nbytes"]
                    ),
                    "peak_cuda_saved_bytes_vs_full_prefix": (
                        baseline["cuda_peak_allocated_bytes"]
                        - row["cuda_peak_allocated_bytes"]
                    ),
                    "ttft_ratio_vs_full_prefix": (
                        row["ttft_seconds"] / baseline["ttft_seconds"]
                        if baseline["ttft_seconds"]
                        else None
                    ),
                    "tpot_ratio_vs_full_prefix": (
                        row["median_tpot_seconds"]
                        / baseline["median_tpot_seconds"]
                        if row["median_tpot_seconds"] is not None
                        and baseline["median_tpot_seconds"]
                        else None
                    ),
                    "f1_delta_vs_full_prefix": (
                        row["f1"] - baseline["f1"]
                        if row["f1"] is not None and baseline["f1"] is not None
                        else None
                    ),
                }
            )
        paired.append(
            {
                "config": config,
                "pairs": len(pairs),
                **{
                    field: describe(pairs, field)
                    for field in (
                        "persistent_compression_vs_full_prefix",
                        "peak_cuda_saved_bytes_vs_full_prefix",
                        "ttft_ratio_vs_full_prefix",
                        "tpot_ratio_vs_full_prefix",
                        "f1_delta_vs_full_prefix",
                    )
                },
            }
        )

    return {
        "status": "completed",
        "shards": len(shards),
        "rows": len(rows),
        "workload": shards[0]["workload"],
        "workload_metadata": shards[0]["workload_metadata"],
        "all_exactness_gates_passed": True,
        "all_direct_cow_gates_passed": (
            True
            if any(
                shard["exactness_gate"].get("fork_strategy")
                == "paged-cow-staging"
                for shard in shards
            )
            else None
        ),
        "configs": shards[0]["configs"],
        # Main paper-facing tables.  The length table remains available for
        # capacity curves, but natural LongBench samples are not fragmented
        # into nearly one-row groups in the dataset/overall summaries.
        "summary": summary_by_dataset_config,
        "summary_by_dataset_config": summary_by_dataset_config,
        "summary_overall_config": summary_overall_config,
        "summary_by_dataset_config_and_length": summary_by_length,
        "paired_vs_full_prefix": paired,
        "source_shards": [path.name for path in paths],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-shards", type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(args.run_dir, args.expected_shards)
    destination = args.output or args.run_dir / "deployment-summary.json"
    atomic_json(destination, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
