from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

from analyze_validation import bootstrap_mean_ci
from run_replay_diagnostic import CONFIG_SUITES


def row_key(row: dict[str, Any]) -> tuple[str, str | None, int | None]:
    return row["dataset"], row.get("id"), row.get("source_index")


def token_position_agreement(reference: list[int], candidate: list[int]) -> float:
    """Compare aligned tokens and count a length mismatch as disagreement."""
    denominator = max(len(reference), len(candidate))
    if denominator == 0:
        return 1.0
    matches = sum(
        left == right for left, right in zip(reference, candidate)
    )
    return matches / denominator


def paired_lora_summary(
    reference_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    *,
    catastrophic_delta: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    """Paired quality/trajectory/TTFT audit for one fixed store policy."""
    reference = {row_key(row): row for row in reference_rows}
    candidate = {row_key(row): row for row in candidate_rows}
    if reference.keys() != candidate.keys():
        raise ValueError("LoRA and untrained-static sample keys differ")
    pairs = [(candidate[key], reference[key]) for key in sorted(reference)]
    f1_deltas = [left["f1"] - right["f1"] for left, right in pairs]
    datasets = sorted({left["dataset"] for left, _ in pairs})
    per_dataset = {}
    for offset, dataset in enumerate(datasets, start=1):
        selected = [
            (left, right)
            for left, right in pairs
            if left["dataset"] == dataset
        ]
        deltas = [left["f1"] - right["f1"] for left, right in selected]
        per_dataset[dataset] = {
            "samples": len(selected),
            "untrained_mean_f1": statistics.fmean(
                right["f1"] for _, right in selected
            ),
            "lora_mean_f1": statistics.fmean(left["f1"] for left, _ in selected),
            "mean_f1_delta_lora_minus_untrained": statistics.fmean(deltas),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(
                deltas, seed=bootstrap_seed + offset
            ),
        }

    ttft_deltas = [
        float(left["ttft_seconds"]) - float(right["ttft_seconds"])
        for left, right in pairs
    ]
    reference_ttft = [float(right["ttft_seconds"]) for _, right in pairs]
    candidate_ttft = [float(left["ttft_seconds"]) for left, _ in pairs]
    return {
        "samples": len(pairs),
        "untrained_mean_f1": statistics.fmean(right["f1"] for _, right in pairs),
        "lora_mean_f1": statistics.fmean(left["f1"] for left, _ in pairs),
        "mean_f1_delta_lora_minus_untrained": statistics.fmean(f1_deltas),
        "paired_bootstrap_95_ci": bootstrap_mean_ci(
            f1_deltas, seed=bootstrap_seed
        ),
        "per_dataset": per_dataset,
        "prediction_exact_agreement_rate": statistics.fmean(
            left["prediction"] == right["prediction"] for left, right in pairs
        ),
        "token_sequence_exact_agreement_rate": statistics.fmean(
            left["generated_token_ids"] == right["generated_token_ids"]
            for left, right in pairs
        ),
        "mean_token_position_agreement": statistics.fmean(
            token_position_agreement(
                right["generated_token_ids"], left["generated_token_ids"]
            )
            for left, right in pairs
        ),
        "catastrophic_regression_threshold": catastrophic_delta,
        "catastrophic_regression_rate": statistics.fmean(
            delta <= catastrophic_delta for delta in f1_deltas
        ),
        "ttft": {
            "semantic": (
                "request time from persistent-state fork/dequantization through "
                "the first-token logits; tokenizer and offline document Write excluded"
            ),
            "untrained_median_seconds": statistics.median(reference_ttft),
            "lora_median_seconds": statistics.median(candidate_ttft),
            "median_delta_seconds_lora_minus_untrained": statistics.median(
                ttft_deltas
            ),
            "mean_delta_seconds_lora_minus_untrained": statistics.fmean(
                ttft_deltas
            ),
            "ratio_of_medians_lora_over_untrained": (
                statistics.median(candidate_ttft)
                / statistics.median(reference_ttft)
                if statistics.median(reference_ttft) > 0
                else None
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--suite", choices=sorted(CONFIG_SUITES), default="exact")
    parser.add_argument("--overall-margin", type=float, default=-0.02)
    parser.add_argument("--dataset-margin", type=float, default=-0.03)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--catastrophic-delta", type=float, default=-0.5)
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    args = parser.parse_args()
    configs = CONFIG_SUITES[args.suite]
    shards = [
        json.loads(path.read_text())
        for path in sorted(args.run_dir.glob("shard-*-*.json"))
    ]
    expected_shards = args.expected_world_size * len(configs)
    if len(shards) != expected_shards:
        raise SystemExit(f"expected {expected_shards} shards, found {len(shards)}")
    if args.suite == "quant-lora-validation" and (
        not args.expected_data_sha256 or not args.expected_checkpoint_sha256
    ):
        raise SystemExit(
            "quant-lora-validation requires expected data and checkpoint SHA256"
        )
    data_hashes = {shard.get("data_sha256") for shard in shards}
    checkpoint_hashes = {
        shard.get("lora", {}).get("checkpoint_sha256") for shard in shards
    }
    if args.expected_data_sha256 and data_hashes != {args.expected_data_sha256}:
        raise SystemExit(f"validation data SHA mismatch: {data_hashes}")
    if (
        args.expected_checkpoint_sha256
        and checkpoint_hashes != {args.expected_checkpoint_sha256}
    ):
        raise SystemExit(
            f"LoRA checkpoint SHA mismatch: {checkpoint_hashes}"
        )

    merged: dict[str, dict[str, Any]] = {}
    for config_name in configs:
        matching = [shard for shard in shards if shard["config"] == config_name]
        if len(matching) != args.expected_world_size:
            raise SystemExit(
                f"{config_name}: expected {args.expected_world_size} shards, "
                f"found {len(matching)}"
            )
        rows = [row for shard in matching for row in shard["rows"]]
        keys = {(row["dataset"], row["id"], row["source_index"]) for row in rows}
        if len(rows) != len(keys):
            raise SystemExit(f"{config_name}: duplicate rows detected")
        merged[config_name] = {
            "config": config_name,
            "mode": matching[0]["mode"],
            "depth": matching[0]["depth"],
            "residual_bits": matching[0].get(
                "residual_bits", matching[0].get("bits")
            ),
            "attention_bits": matching[0].get("attention_bits"),
            "linear_bits": matching[0].get("linear_bits"),
            "cache_layer_bits": matching[0].get("cache_layer_bits"),
            "policy": matching[0].get("policy"),
            "lora": matching[0].get("lora"),
            "shards": matching,
            "rows": rows,
        }

    dense_by_key = {
        row_key(row): row
        for row in merged["dense"]["rows"]
    }
    q16_config_name = (
        "replay-d7-layer-q16"
        if "replay-d7-layer-q16" in merged
        else None
    )
    q16_by_key = (
        {row_key(row): row for row in merged[q16_config_name]["rows"]}
        if q16_config_name is not None
        else {}
    )
    expected_keys = set(dense_by_key)
    for config_name, config in merged.items():
        actual_keys = {row_key(row) for row in config["rows"]}
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            extra = sorted(actual_keys - expected_keys)
            raise SystemExit(
                f"{config_name}: sample keys differ from dense; "
                f"missing={missing[:3]} extra={extra[:3]}"
            )
    if args.suite == "quant-lora-validation":
        expected_indices = set(range(6, 36))
        for config_name, config in merged.items():
            enabled = {
                bool(shard.get("lora", {}).get("enabled_for_this_config"))
                for shard in config["shards"]
            }
            expected_enabled = config_name == "replay-d7-frozen-static-lora"
            if enabled != {expected_enabled}:
                raise SystemExit(
                    f"{config_name}: invalid LoRA activation states {enabled}; "
                    f"expected {expected_enabled}"
                )
            for shard in config["shards"]:
                if shard.get("lora", {}).get(
                    "semantic_mismatch_explicitly_allowed"
                ):
                    raise SystemExit("quant LoRA semantic mismatch override is forbidden")
            if len(config["rows"]) != 60:
                raise SystemExit(
                    f"{config_name}: expected 60 validation rows, "
                    f"found {len(config['rows'])}"
                )
            for dataset in ("qasper", "2wikimqa"):
                indices = {
                    int(row["source_index"])
                    for row in config["rows"]
                    if row["dataset"] == dataset
                }
                if indices != expected_indices:
                    raise SystemExit(
                        f"{config_name}/{dataset}: source indices are not 6--35"
                    )
    summaries = []
    for config_name in configs:
        config = merged[config_name]
        config_index = configs.index(config_name)
        matches = []
        token_matches = []
        token_position_agreements = []
        f1_deltas = []
        q16_prediction_matches = []
        q16_token_matches = []
        q16_token_position_agreements = []
        q16_f1_deltas = []
        dataset_f1_deltas: dict[str, list[float]] = {}
        for row in config["rows"]:
            key = row_key(row)
            dense = dense_by_key[key]
            matches.append(row["prediction"] == dense["prediction"])
            token_matches.append(
                row["generated_token_ids"] == dense["generated_token_ids"]
            )
            token_position_agreements.append(
                token_position_agreement(
                    dense["generated_token_ids"], row["generated_token_ids"]
                )
            )
            delta = row["f1"] - dense["f1"]
            f1_deltas.append(delta)
            dataset_f1_deltas.setdefault(row["dataset"], []).append(delta)
            if q16_by_key:
                q16 = q16_by_key[key]
                q16_prediction_matches.append(row["prediction"] == q16["prediction"])
                q16_token_matches.append(
                    row["generated_token_ids"] == q16["generated_token_ids"]
                )
                q16_token_position_agreements.append(
                    token_position_agreement(
                        q16["generated_token_ids"], row["generated_token_ids"]
                    )
                )
                q16_f1_deltas.append(row["f1"] - q16["f1"])
        state_rows = [
            row for row in config["rows"] if row["stored_persistent_nbytes"] is not None
        ]
        residual_rows = [
            row for row in state_rows if row["stored_residual_nbytes"] is not None
        ]
        error_rows = [
            row for row in residual_rows if row.get("residual_error_sums") is not None
        ]
        cache_error_rows = [
            row for row in state_rows if row.get("cache_error_sums") is not None
        ]
        ttft_rows = [
            row for row in config["rows"] if row.get("ttft_seconds") is not None
        ]

        def cache_relative_rmse(category: str) -> float | None:
            if not cache_error_rows:
                return None
            squared_error = sum(
                row["cache_error_sums"][category]["squared_error_sum"]
                for row in cache_error_rows
            )
            reference_squared = sum(
                row["cache_error_sums"][category]["reference_squared_sum"]
                for row in cache_error_rows
            )
            if reference_squared == 0:
                return 0.0
            return math.sqrt(squared_error / reference_squared)

        dataset_mean_f1 = {
            dataset: statistics.fmean(
                row["f1"] for row in config["rows"] if row["dataset"] == dataset
            )
            for dataset in sorted({row["dataset"] for row in config["rows"]})
        }
        dense_dataset_mean_f1 = {
            dataset: statistics.fmean(
                row["f1"]
                for row in dense_by_key.values()
                if row["dataset"] == dataset
            )
            for dataset in dataset_mean_f1
        }
        mean_f1 = statistics.fmean(row["f1"] for row in config["rows"])
        dense_mean_f1 = statistics.fmean(row["f1"] for row in dense_by_key.values())
        dataset_deltas = {
            dataset: value - dense_dataset_mean_f1[dataset]
            for dataset, value in dataset_mean_f1.items()
        }
        summaries.append(
            {
                "config": config_name,
                "depth": config["depth"],
                "residual_bits": config["residual_bits"],
                # Legacy alias retained for the existing result tables.
                "bits": config["residual_bits"],
                "attention_bits": config["attention_bits"],
                "linear_bits": config["linear_bits"],
                "cache_layer_bits": config["cache_layer_bits"],
                "policy": config["policy"],
                "lora": config["lora"],
                "samples": len(config["rows"]),
                "mean_f1": mean_f1,
                "mean_f1_delta_vs_dense": mean_f1 - dense_mean_f1,
                "paired_bootstrap_95_ci_vs_dense": bootstrap_mean_ci(
                    f1_deltas, seed=args.bootstrap_seed + config_index * 100
                ),
                "dataset_mean_f1": dataset_mean_f1,
                "dataset_mean_f1_delta_vs_dense": dataset_deltas,
                "dataset_paired_bootstrap_95_ci_vs_dense": {
                    dataset: bootstrap_mean_ci(
                        deltas,
                        seed=args.bootstrap_seed
                        + config_index * 100
                        + dataset_index,
                    )
                    for dataset_index, (dataset, deltas) in enumerate(
                        sorted(dataset_f1_deltas.items()), start=1
                    )
                },
                "catastrophic_regression_rate_delta_le_minus_0_5": (
                    statistics.fmean(
                        delta <= args.catastrophic_delta for delta in f1_deltas
                    )
                ),
                "mean_f1_delta_vs_q16_replay": (
                    statistics.fmean(q16_f1_deltas) if q16_f1_deltas else None
                ),
                "paired_bootstrap_95_ci_vs_q16_replay": (
                    bootstrap_mean_ci(
                        q16_f1_deltas,
                        seed=args.bootstrap_seed + config_index * 100 + 50,
                    )
                    if q16_f1_deltas
                    else None
                ),
                "prediction_exact_agreement_rate_vs_q16_replay": (
                    statistics.fmean(q16_prediction_matches)
                    if q16_prediction_matches
                    else None
                ),
                "token_sequence_exact_agreement_rate_vs_q16_replay": (
                    statistics.fmean(q16_token_matches)
                    if q16_token_matches
                    else None
                ),
                "mean_token_position_agreement_vs_q16_replay": (
                    statistics.fmean(q16_token_position_agreements)
                    if q16_token_position_agreements
                    else None
                ),
                "passes_mean_noninferiority_vs_dense": (
                    mean_f1 - dense_mean_f1 >= args.overall_margin
                    and all(
                        delta >= args.dataset_margin
                        for delta in dataset_deltas.values()
                    )
                ),
                "prediction_matches_dense": sum(matches),
                "prediction_exact_agreement_rate_vs_dense": statistics.fmean(
                    matches
                ),
                "token_matches_dense": sum(token_matches),
                "token_sequence_exact_agreement_rate_vs_dense": statistics.fmean(
                    token_matches
                ),
                "mean_token_position_agreement_vs_dense": statistics.fmean(
                    token_position_agreements
                ),
                "all_tokens_match_dense": all(token_matches),
                "mean_write_seconds": (
                    statistics.fmean(row["write_seconds"] for row in state_rows)
                    if state_rows
                    else None
                ),
                "mean_generation_seconds": statistics.fmean(
                    row["generation_seconds"] for row in config["rows"]
                ),
                "mean_ttft_seconds": (
                    statistics.fmean(row["ttft_seconds"] for row in ttft_rows)
                    if ttft_rows
                    else None
                ),
                "median_ttft_seconds": (
                    statistics.median(row["ttft_seconds"] for row in ttft_rows)
                    if ttft_rows
                    else None
                ),
                "mean_residual_mib": (
                    statistics.fmean(
                        row["stored_residual_nbytes"] for row in residual_rows
                    )
                    / 2**20
                    if residual_rows
                    else None
                ),
                "mean_lower_cache_mib": (
                    statistics.fmean(
                        row["stored_lower_cache_nbytes"] for row in state_rows
                    )
                    / 2**20
                    if state_rows
                    else None
                ),
                "mean_persistent_mib": (
                    statistics.fmean(
                        row["stored_persistent_nbytes"] for row in state_rows
                    )
                    / 2**20
                    if state_rows
                    else None
                ),
                "persistent_bytes": (
                    {
                        "mean": statistics.fmean(
                            row["stored_persistent_nbytes"] for row in state_rows
                        ),
                        "min": min(
                            row["stored_persistent_nbytes"] for row in state_rows
                        ),
                        "max": max(
                            row["stored_persistent_nbytes"] for row in state_rows
                        ),
                        "sum": sum(
                            row["stored_persistent_nbytes"] for row in state_rows
                        ),
                    }
                    if state_rows
                    else None
                ),
                "residual_relative_rmse": (
                    math.sqrt(
                        sum(
                            row["residual_error_sums"]["squared_error_sum"]
                            for row in error_rows
                        )
                        / sum(
                            row["residual_error_sums"]["reference_squared_sum"]
                            for row in error_rows
                        )
                    )
                    if error_rows
                    else None
                ),
                "attention_cache_relative_rmse": cache_relative_rmse(
                    "attention"
                ),
                "linear_cache_relative_rmse": cache_relative_rmse("linear"),
                "max_peak_allocated_gib": max(
                    row["peak_allocated_bytes"] for row in config["rows"]
                )
                / 2**30,
            }
        )

    prefix_summary = next(
        (summary for summary in summaries if summary["config"] == "prefix"), None
    )
    if prefix_summary is not None:
        prefix_mib = prefix_summary["mean_persistent_mib"]
        for summary in summaries:
            persistent_mib = summary["mean_persistent_mib"]
            summary["persistent_compression_vs_prefix"] = (
                prefix_mib / persistent_mib
                if prefix_mib is not None and persistent_mib is not None
                else None
            )
    q16_summary = next(
        (summary for summary in summaries if summary["config"] == q16_config_name),
        None,
    )
    if q16_summary is not None:
        q16_mib = q16_summary["mean_persistent_mib"]
        for summary in summaries:
            persistent_mib = summary["mean_persistent_mib"]
            summary["persistent_compression_vs_q16_replay"] = (
                q16_mib / persistent_mib
                if q16_mib is not None and persistent_mib is not None
                else None
            )

    quant_lora_paired = None
    adapter_runtime = None
    if args.suite == "quant-lora-validation":
        baseline_name = "replay-d7-frozen-static"
        candidate_name = "replay-d7-frozen-static-lora"
        baseline = merged[baseline_name]
        candidate = merged[candidate_name]
        store_fields = (
            "depth",
            "residual_bits",
            "attention_bits",
            "linear_bits",
            "cache_layer_bits",
            "policy",
        )
        mismatched = {
            field: (baseline[field], candidate[field])
            for field in store_fields
            if baseline[field] != candidate[field]
        }
        if mismatched:
            raise SystemExit(
                "trained and untrained frozen-static store policies differ: "
                f"{mismatched}"
            )
        quant_lora_paired = {
            "candidate_config": candidate_name,
            "reference_config": baseline_name,
            **paired_lora_summary(
                baseline["rows"],
                candidate["rows"],
                catastrophic_delta=args.catastrophic_delta,
                bootstrap_seed=args.bootstrap_seed + 50_000,
            ),
        }
        target_shards = candidate["shards"]

        def unique_lora_field(field: str) -> Any:
            values = {shard["lora"].get(field) for shard in target_shards}
            if len(values) != 1:
                raise SystemExit(f"inconsistent {field} across ranks: {values}")
            return next(iter(values))

        adapter_bytes = unique_lora_field("adapter_parameter_nbytes")
        checkpoint_file_bytes = unique_lora_field("checkpoint_file_nbytes")
        installed_modules = unique_lora_field("installed_lora_modules")
        if not adapter_bytes or not checkpoint_file_bytes or not installed_modules:
            raise SystemExit("adapter byte/module accounting is missing or zero")
        adapter_load_times = [
            float(shard["lora"]["adapter_load_seconds"])
            for shard in target_shards
        ]
        adapter_runtime = {
            "adapter_parameter_nbytes": adapter_bytes,
            "checkpoint_file_nbytes": checkpoint_file_bytes,
            "installed_lora_modules": installed_modules,
            "load_seconds_by_rank": adapter_load_times,
            "median_adapter_load_seconds": statistics.median(adapter_load_times),
            "min_adapter_load_seconds": min(adapter_load_times),
            "max_adapter_load_seconds": max(adapter_load_times),
            "residency_semantic": (
                "LoRA FP32 A/B parameter storage only; checkpoint file bytes are "
                "reported separately and disabled baselines keep the adapter resident"
            ),
        }

    analysis = {
        "status": "completed",
        "samples": len(dense_by_key),
        "suite": args.suite,
        "mean_noninferiority_thresholds": {
            "overall_mean_f1_delta": args.overall_margin,
            "per_dataset_mean_f1_delta": args.dataset_margin,
        },
        "catastrophic_regression_threshold": args.catastrophic_delta,
        "q16_replay_reference_config": q16_config_name,
        "lora_checkpoints": sorted(
            {
                config["lora"]["checkpoint_sha256"]
                for config in merged.values()
                if config.get("lora")
                and config["lora"].get("checkpoint_sha256")
            }
        ),
        "source_index_protocol": {
            "start_inclusive": shards[0].get("source_index_start"),
            "end_inclusive": shards[0].get("source_index_end"),
            "excluded": shards[0].get("exclude_source_indices", []),
        },
        "all_replay_tokens_match_dense": all(
            summary["all_tokens_match_dense"]
            for summary in summaries
            if summary["config"].startswith("replay")
        ),
        "quant_lora_paired_validation": quant_lora_paired,
        "adapter_runtime": adapter_runtime,
        "summary": summaries,
    }
    destination = args.run_dir / "replay_analysis.json"
    destination.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
