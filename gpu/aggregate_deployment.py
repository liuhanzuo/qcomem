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
    """Fail closed unless the incremental three-way COW gate is complete."""

    if gate.get("fork_strategy") != "paged-cow-staging":
        return True
    direct = gate.get("cow_vs_deep_clone_q16")
    if not isinstance(direct, dict):
        return False
    comparisons = direct.get("comparisons")
    full_vs_eager = (
        comparisons.get("full_prefix_vs_eager_q16")
        if isinstance(comparisons, dict)
        else None
    )
    full_vs_cow = (
        comparisons.get("full_prefix_vs_cow_q16")
        if isinstance(comparisons, dict)
        else None
    )
    eager_vs_cow = (
        comparisons.get("eager_q16_vs_cow_q16")
        if isinstance(comparisons, dict)
        else None
    )
    immutable = direct.get("cow_immutable_audit")
    return bool(
        gate.get("hard_gate_reference") == "incremental-full-prefix-q16"
        and gate.get("dense_single_chunk_diagnostic_only") is True
        and gate.get("incremental_hard_gate", {}).get("passed")
        and direct.get("passed")
        and direct.get("semantic_version") == "incremental-three-way-v1"
        and direct.get("caller_boundary_match")
        and direct.get("incremental_three_way_token_exact")
        and direct.get("same_persistent_source")
        and direct.get("cow_was_exercised")
        and direct.get("strategy_effective") == "paged-cow-staging"
        and direct.get("source_after_eager", {}).get("verified")
        and direct.get("source_after_cow", {}).get("verified")
        and isinstance(immutable, dict)
        and immutable.get("verified")
        and isinstance(full_vs_eager, dict)
        and full_vs_eager.get("passed")
        and full_vs_eager.get("token_sequence_exact")
        and isinstance(full_vs_cow, dict)
        and full_vs_cow.get("passed")
        and full_vs_cow.get("token_sequence_exact")
        and isinstance(eager_vs_cow, dict)
        and eager_vs_cow.get("passed")
        and eager_vs_cow.get("token_sequence_exact")
        and eager_vs_cow.get("logits_bitwise_exact")
    )


def validate_complete_protocol(
    shards: list[dict[str, Any]],
    *,
    expected_shards: int,
    expected_configs: list[str] | None = None,
    expected_workloads: int | None = None,
    expected_source_indices: list[int] | None = None,
    expected_data_sha256: str | None = None,
    expected_source_revision: str | None = None,
    expected_fork_strategy: str | None = None,
    expected_warmups: int | None = None,
    expected_repeats: int | None = None,
    expected_max_new_tokens: int | None = None,
    require_complete_measurements: bool = False,
    require_no_test_v2: bool = False,
) -> dict[str, Any]:
    """Validate rank, matrix, provenance, and required measurement coverage."""

    ranks = [int(shard.get("rank", -1)) for shard in shards]
    if sorted(ranks) != list(range(expected_shards)):
        raise ValueError(f"rank coverage is not 0..{expected_shards - 1}: {ranks}")
    if any(int(shard.get("world_size", -1)) != expected_shards for shard in shards):
        raise ValueError("world_size is inconsistent with expected shard count")
    metadata = shards[0].get("workload_metadata")
    if not isinstance(metadata, dict) or any(
        shard.get("workload_metadata") != metadata for shard in shards[1:]
    ):
        raise ValueError("workload metadata is missing or differs across shards")
    if require_no_test_v2 and metadata.get("test_v2_consumed") is not False:
        raise ValueError("test-v2 consumption must be explicitly false")
    if expected_data_sha256 is not None and (
        metadata.get("data_sha256") != expected_data_sha256
    ):
        raise ValueError("data SHA256 does not match the frozen protocol")
    if expected_source_revision is not None and metadata.get("source_revisions") != [
        expected_source_revision
    ]:
        raise ValueError("source revision does not match the frozen protocol")

    declared_configs = [
        config.get("name") for config in shards[0].get("configs", ())
    ]
    if not declared_configs or len(set(declared_configs)) != len(declared_configs):
        raise ValueError("declared configs are missing or duplicated")
    if any(
        [config.get("name") for config in shard.get("configs", ())]
        != declared_configs
        for shard in shards[1:]
    ):
        raise ValueError("declared config order differs across shards")
    if expected_configs is not None and declared_configs != expected_configs:
        raise ValueError(
            f"configs {declared_configs} do not match frozen {expected_configs}"
        )
    if "qcomem-d7-frozen-static" in declared_configs:
        frozen = next(
            config
            for config in shards[0]["configs"]
            if config.get("name") == "qcomem-d7-frozen-static"
        )
        if not (
            frozen.get("depth") == 7
            and frozen.get("residual_bits") == 4
            and frozen.get("attention_bits") == 4
            and frozen.get("linear_bits") == 8
            and frozen.get("cache_layer_bits") == [8, 8, 8, 4, 8, 8, 8]
        ):
            raise ValueError("frozen-static effective bit policy is incorrect")

    fork_strategies = {shard.get("fork_strategy") for shard in shards}
    if len(fork_strategies) != 1:
        raise ValueError("fork strategy differs across shards")
    if expected_fork_strategy is not None and fork_strategies != {
        expected_fork_strategy
    }:
        raise ValueError("fork strategy does not match the frozen protocol")
    if expected_warmups is not None and any(
        int(shard.get("warmups", -1)) != expected_warmups for shard in shards
    ):
        raise ValueError("warmup count does not match the frozen protocol")
    repeats = expected_repeats
    if repeats is None:
        repeat_values = {int(shard.get("repeats", -1)) for shard in shards}
        if len(repeat_values) != 1:
            raise ValueError("repeat count differs across shards")
        repeats = repeat_values.pop()
    elif any(int(shard.get("repeats", -1)) != repeats for shard in shards):
        raise ValueError("repeat count does not match the frozen protocol")
    if repeats < 1:
        raise ValueError("repeat count must be positive")

    all_rows: list[dict[str, Any]] = []
    workload_keys: set[tuple[str, str, int]] = set()
    measurement_keys: set[tuple[str, int, str]] = set()
    for shard in shards:
        rows = shard.get("rows")
        orders = shard.get("randomized_orders")
        if not isinstance(rows, list) or not rows or not isinstance(orders, dict):
            raise ValueError(f"rank {shard['rank']} has no complete rows/orders")
        workload_ids = {str(row.get("workload_id")) for row in rows}
        if set(orders) != workload_ids:
            raise ValueError(f"rank {shard['rank']} workload/order coverage differs")
        for workload_id in sorted(workload_ids):
            workload_rows = [row for row in rows if row.get("workload_id") == workload_id]
            workload_orders = orders[workload_id]
            if len(workload_orders) != repeats:
                raise ValueError(f"{workload_id} has incomplete randomized orders")
            for repeat, order in enumerate(workload_orders):
                if sorted(order) != sorted(declared_configs):
                    raise ValueError(f"{workload_id} repeat {repeat} is not a config permutation")
                matching = [
                    row for row in workload_rows if int(row.get("repeat", -1)) == repeat
                ]
                if len(matching) != len(declared_configs):
                    raise ValueError(f"{workload_id} repeat {repeat} has incomplete rows")
                positions = {
                    int(row.get("randomized_order_position", -1)): row.get("config")
                    for row in matching
                }
                if [positions.get(index) for index in range(len(order))] != order:
                    raise ValueError(f"{workload_id} repeat {repeat} order/rows differ")
        for row in rows:
            key = (
                str(row.get("workload_id")),
                int(row.get("repeat", -1)),
                str(row.get("config")),
            )
            if key in measurement_keys:
                raise ValueError(f"duplicate measurement row: {key}")
            measurement_keys.add(key)
            if row.get("config") not in declared_configs:
                raise ValueError(f"undeclared config in row: {row.get('config')}")
            source_index = row.get("source_index")
            if shards[0].get("workload") == "longbench" and source_index is None:
                raise ValueError("LongBench rows require source_index")
            workload_keys.add(
                (
                    str(row.get("workload_id")),
                    str(row.get("dataset")),
                    int(source_index) if source_index is not None else None,
                )
            )
            if expected_max_new_tokens is not None and int(
                row.get("max_new_tokens", -1)
            ) != expected_max_new_tokens:
                raise ValueError("max_new_tokens does not match the frozen protocol")
            all_rows.append(row)

    if expected_workloads is not None and len(workload_keys) != expected_workloads:
        raise ValueError(
            f"expected {expected_workloads} workloads, found {len(workload_keys)}"
        )
    actual_source_indices = sorted(
        {source for _, _, source in workload_keys if source is not None}
    )
    if expected_source_indices is not None and actual_source_indices != sorted(
        set(expected_source_indices)
    ):
        raise ValueError(
            f"source indices {actual_source_indices} do not match frozen "
            f"{sorted(set(expected_source_indices))}"
        )

    if require_complete_measurements:
        required = (
            "persistent_total_resident_nbytes",
            "persistent_materialized_staging_nbytes",
            "capacity_document_denominator_nbytes",
            "cuda_peak_allocated_bytes",
            "cuda_peak_reserved_bytes",
            "nvml_sampled_peak_process_bytes",
            "ttft_seconds",
        )
        cow_fields = (
            "cow_initial_shared_nbytes",
            "cow_initial_private_nbytes",
            "cow_after_query_shared_nbytes",
            "cow_after_query_private_nbytes",
            "cow_final_shared_nbytes",
            "cow_final_private_nbytes",
        )
        for row in all_rows:
            missing = [field for field in required if row.get(field) is None]
            if missing:
                raise ValueError(f"{row['config']} row omits measurements: {missing}")
            if row["capacity_document_denominator_nbytes"] != row[
                "persistent_total_resident_nbytes"
            ]:
                raise ValueError("capacity denominator is not total resident bytes")
            if row["config"].startswith("qcomem-"):
                missing_cow = [field for field in cow_fields if row.get(field) is None]
                if missing_cow:
                    raise ValueError(f"COW row omits shared/private fields: {missing_cow}")
                fork_memory = row.get("fork_memory")
                if not isinstance(fork_memory, dict) or (
                    fork_memory.get("strategy_effective") != "paged-cow-staging"
                    or fork_memory.get("fallback_reason") is not None
                ):
                    raise ValueError("Q-CoMem row bypassed paged-cow-staging")

    return {
        "ranks": ranks,
        "workloads": len(workload_keys),
        "source_indices": actual_source_indices,
        "configs": declared_configs,
        "measurements": len(all_rows),
        "complete_measurements_required": require_complete_measurements,
    }


def aggregate(
    run_dir: Path,
    expected_shards: int | None = None,
    *,
    expected_configs: list[str] | None = None,
    expected_workloads: int | None = None,
    expected_source_indices: list[int] | None = None,
    expected_data_sha256: str | None = None,
    expected_source_revision: str | None = None,
    expected_fork_strategy: str | None = None,
    expected_warmups: int | None = None,
    expected_repeats: int | None = None,
    expected_max_new_tokens: int | None = None,
    require_complete_measurements: bool = False,
    require_no_test_v2: bool = False,
) -> dict[str, Any]:
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
    protocol_validation = validate_complete_protocol(
        shards,
        expected_shards=expected_shards,
        expected_configs=expected_configs,
        expected_workloads=expected_workloads,
        expected_source_indices=expected_source_indices,
        expected_data_sha256=expected_data_sha256,
        expected_source_revision=expected_source_revision,
        expected_fork_strategy=expected_fork_strategy,
        expected_warmups=expected_warmups,
        expected_repeats=expected_repeats,
        expected_max_new_tokens=expected_max_new_tokens,
        require_complete_measurements=require_complete_measurements,
        require_no_test_v2=require_no_test_v2,
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
        "protocol_validation": protocol_validation,
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
    parser.add_argument("--expected-configs", nargs="+")
    parser.add_argument("--expected-workloads", type=int)
    parser.add_argument("--expected-source-indices", type=int, nargs="+")
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-fork-strategy")
    parser.add_argument("--expected-warmups", type=int)
    parser.add_argument("--expected-repeats", type=int)
    parser.add_argument("--expected-max-new-tokens", type=int)
    parser.add_argument("--require-complete-measurements", action="store_true")
    parser.add_argument("--require-no-test-v2", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = aggregate(
        args.run_dir,
        args.expected_shards,
        expected_configs=args.expected_configs,
        expected_workloads=args.expected_workloads,
        expected_source_indices=args.expected_source_indices,
        expected_data_sha256=args.expected_data_sha256,
        expected_source_revision=args.expected_source_revision,
        expected_fork_strategy=args.expected_fork_strategy,
        expected_warmups=args.expected_warmups,
        expected_repeats=args.expected_repeats,
        expected_max_new_tokens=args.expected_max_new_tokens,
        require_complete_measurements=args.require_complete_measurements,
        require_no_test_v2=args.require_no_test_v2,
    )
    destination = args.output or args.run_dir / "deployment-summary.json"
    atomic_json(destination, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
