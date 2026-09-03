from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Optional

from analyze_validation import comparison
from run_sft_full_qcomem_downstream import CONFIGS, EXPECTED_POLICIES


STAGES = ("base", "sft")
EXPECTED_RANKS = set(range(8))
EXPECTED_INDICES = set(range(6, 36))


def read_jsons(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["dataset"], str(row.get("id")), int(row["source_index"])


def validate_config_contract(shard: dict[str, Any], config: str) -> None:
    expected = EXPECTED_POLICIES[config]
    actual = {
        "mode": shard.get("mode"),
        "depth": shard.get("depth"),
        "residual_bits": shard.get("residual_bits", shard.get("bits")),
        "attention_bits": shard.get("attention_bits"),
        "linear_bits": shard.get("linear_bits"),
        "cache_layer_bits": (
            tuple(shard["cache_layer_bits"])
            if shard.get("cache_layer_bits") is not None
            else None
        ),
    }
    if actual != expected:
        raise ValueError(f"{config}: policy drifted: {actual} != {expected}")


def merge_config(
    shards: list[dict[str, Any]], stage: str, config: str
) -> dict[str, Any]:
    selected = [
        shard
        for shard in shards
        if shard.get("model_stage") == stage and shard.get("config") == config
    ]
    if len(selected) != 8:
        raise ValueError(f"{stage}/{config}: expected 8 shards, got {len(selected)}")
    if {int(shard["rank"]) for shard in selected} != EXPECTED_RANKS:
        raise ValueError(f"{stage}/{config}: rank coverage is not 0--7")
    for shard in selected:
        validate_config_contract(shard, config)
        if shard.get("full_lower_state_qcomem") is not True:
            raise ValueError(f"{stage}/{config}: full-state semantic marker missing")
    rows = [row for shard in selected for row in shard["rows"]]
    keys = {row_key(row) for row in rows}
    if len(rows) != 60 or len(keys) != 60:
        raise ValueError(
            f"{stage}/{config}: expected 60 unique rows, got {len(rows)}/{len(keys)}"
        )
    for dataset in ("qasper", "2wikimqa"):
        indices = {
            int(row["source_index"]) for row in rows if row["dataset"] == dataset
        }
        if indices != EXPECTED_INDICES:
            raise ValueError(
                f"{stage}/{config}/{dataset}: source indices are not 6--35"
            )
    return {
        "config": f"{stage}-{config}",
        "stage": stage,
        "source_config": config,
        "rows": rows,
        "shards": selected,
        "mean_f1": statistics.fmean(float(row["f1"]) for row in rows),
        "dataset_mean_f1": {
            dataset: statistics.fmean(
                float(row["f1"]) for row in rows if row["dataset"] == dataset
            )
            for dataset in ("qasper", "2wikimqa")
        },
    }


def token_sequence_agreement(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> float:
    left = {row_key(row): row for row in candidate["rows"]}
    right = {row_key(row): row for row in reference["rows"]}
    if left.keys() != right.keys():
        raise ValueError("paired token comparison sample keys differ")
    return statistics.fmean(
        left[key]["generated_token_ids"] == right[key]["generated_token_ids"]
        for key in left
    )


def storage_summary(result: dict[str, Any]) -> Optional[dict[str, Any]]:
    if result["source_config"] == "dense":
        return None
    rows = result["rows"]
    values = [int(row["stored_persistent_nbytes"]) for row in rows]
    residual = [int(row["stored_residual_nbytes"]) for row in rows]
    lower = [int(row["stored_lower_cache_nbytes"]) for row in rows]
    return {
        "mean_persistent_bytes": statistics.fmean(values),
        "min_persistent_bytes": min(values),
        "max_persistent_bytes": max(values),
        "aggregate_persistent_bytes": sum(values),
        "mean_residual_bytes": statistics.fmean(residual),
        "mean_lower_cache_bytes": statistics.fmean(lower),
        "scope": "document residual plus complete lower-layer KV/recurrent/conv state",
    }


def finite_memory_summary(shards: list[dict[str, Any]]) -> dict[str, Any]:
    by_rank: dict[int, tuple[int, int, float]] = {}
    for shard in shards:
        rank = int(shard["rank"])
        values = (
            int(shard["model_allocated_bytes"]),
            int(shard["peak_after_dcp_load_bytes"]),
            float(shard["dcp_load_seconds"]),
        )
        if rank in by_rank and by_rank[rank] != values:
            raise ValueError(f"rank {rank} repeated inconsistent memory/load metadata")
        by_rank[rank] = values
    if set(by_rank) != EXPECTED_RANKS:
        raise ValueError("memory/load metadata rank coverage is not 0--7")
    ordered = [by_rank[rank] for rank in sorted(by_rank)]
    allocated = [values[0] for values in ordered]
    peaks = [values[1] for values in ordered]
    dcp_seconds = [values[2] for values in ordered]
    if any(not math.isfinite(value) or value <= 0 for value in dcp_seconds):
        raise ValueError("invalid DCP load timing")
    return {
        "model_allocated_bytes_per_rank": allocated,
        "max_model_allocated_bytes": max(allocated),
        "peak_after_dcp_load_bytes_per_rank": peaks,
        "max_peak_after_dcp_load_bytes": max(peaks) if peaks else None,
        "dcp_load_seconds_per_rank": dcp_seconds,
        "max_dcp_load_seconds": max(dcp_seconds) if dcp_seconds else None,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate paired base/SFT complete lower-state Q-CoMem evaluation"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shards = read_jsons(args.run_dir.glob("*-shard-*-*.json"))
    expected_count = len(STAGES) * len(CONFIGS) * 8
    if len(shards) != expected_count:
        raise SystemExit(f"expected {expected_count} shards, got {len(shards)}")
    if {shard.get("data_sha256") for shard in shards} != {
        args.expected_data_sha256
    }:
        raise SystemExit("validation data SHA256 drifted")
    if not all(
        shard.get("raw_test_v2_read") is False
        and shard.get("prompt_protocol") == "longbench-v1-official"
        and int(shard.get("max_input_tokens", -1)) == 4096
        and int(shard.get("group_size", -1)) == 64
        and shard.get("source_index_start") == 6
        and shard.get("source_index_end") == 35
        for shard in shards
    ):
        raise SystemExit("governance or inference protocol gate failed")
    for shard in shards:
        caps = {"qasper": 128, "2wikimqa": 32}
        if any(
            int(row.get("max_new_tokens", -1)) != caps[row["dataset"]]
            for row in shard["rows"]
        ):
            raise SystemExit("per-dataset generation cap drifted")

    base_shards = [shard for shard in shards if shard.get("model_stage") == "base"]
    sft_shards = [shard for shard in shards if shard.get("model_stage") == "sft"]
    if {shard.get("checkpoint_manifest_sha256") for shard in base_shards} != {None}:
        raise SystemExit("base shards were not produced before DCP load")
    if {shard.get("checkpoint_manifest_sha256") for shard in sft_shards} != {
        args.expected_checkpoint_manifest_sha256
    }:
        raise SystemExit("SFT shards do not bind the selected checkpoint")
    if not all(
        shard.get("checkpoint_payload_integrity_verified") is True
        and int(shard.get("checkpoint_step", -1)) == 128
        for shard in sft_shards
    ):
        raise SystemExit("SFT DCP integrity/step gate failed")
    if not all(
        all(int(audit["changed_elements"]) == 0 for audit in shard["replica_sample_audits"])
        for shard in base_shards
    ):
        raise SystemExit("base model parameter mutation gate failed")
    if not all(
        all(int(audit["changed_elements"]) > 0 for audit in shard["replica_sample_audits"])
        for shard in sft_shards
    ):
        raise SystemExit("SFT checkpoint sampled-change gate failed")

    merged = {
        stage: {
            config: merge_config(shards, stage, config) for config in CONFIGS
        }
        for stage in STAGES
    }
    expected_keys = {
        row_key(row) for row in merged["base"]["dense"]["rows"]
    }
    for stage in STAGES:
        for config in CONFIGS:
            if {row_key(row) for row in merged[stage][config]["rows"]} != expected_keys:
                raise SystemExit(f"{stage}/{config}: paired sample set drifted")

    comparisons: dict[str, Any] = {}
    seed = 20260830
    for index, config in enumerate(CONFIGS):
        comparisons[f"sft_vs_base__{config}"] = comparison(
            merged["sft"][config], merged["base"][config], seed=seed + index
        )
        comparisons[f"sft_vs_base__{config}"]["token_sequence_exact_agreement"] = (
            token_sequence_agreement(merged["sft"][config], merged["base"][config])
        )
    for stage_index, stage in enumerate(STAGES):
        dense = merged[stage]["dense"]
        q16 = merged[stage]["replay-d7-layer-q16"]
        for config_index, config in enumerate(CONFIGS[1:], start=1):
            comparisons[f"{stage}__{config}_vs_dense"] = comparison(
                merged[stage][config], dense, seed=seed + 100 + stage_index * 10 + config_index
            )
        for config_index, config in enumerate(CONFIGS[2:], start=1):
            comparisons[f"{stage}__{config}_vs_q16"] = comparison(
                merged[stage][config], q16, seed=seed + 200 + stage_index * 10 + config_index
            )

    stores = {
        stage: {
            config: storage_summary(merged[stage][config]) for config in CONFIGS
        }
        for stage in STAGES
    }
    for stage in STAGES:
        q16_bytes = stores[stage]["replay-d7-layer-q16"]["aggregate_persistent_bytes"]
        for config in CONFIGS[1:]:
            stores[stage][config]["compression_vs_same_model_q16"] = (
                q16_bytes / stores[stage][config]["aggregate_persistent_bytes"]
            )

    result = {
        "schema_version": "qcomem-sft-full-state-downstream-v1",
        "status": "completed",
        "samples": 60,
        "validation_data_sha256": args.expected_data_sha256,
        "source_indices": [6, 35],
        "checkpoint_manifest_sha256": args.expected_checkpoint_manifest_sha256,
        "checkpoint_step": 128,
        "raw_test_v2_read": False,
        "artifact_ledgers": {
            "code_ledger_sha256": sha256_file(args.run_dir / "code.sha256"),
            "model_artifact_ledger_sha256": sha256_file(
                args.run_dir / "model-artifacts.sha256"
            ),
            "model_weight_ledger_sha256": sha256_file(
                args.run_dir / "model-weights.sha256"
            ),
            "checkpoint_manifest_ledger_sha256": sha256_file(
                args.run_dir / "checkpoint-manifest.sha256"
            ),
            "validation_data_ledger_sha256": sha256_file(
                args.run_dir / "validation-data.sha256"
            ),
        },
        "protocol": {
            "same_job_before_after": True,
            "prompt": "longbench-v1-official",
            "max_input_tokens": 4096,
            "dataset_max_new_tokens": {"qasper": 128, "2wikimqa": 32},
            "depth": 7,
            "group_size": 64,
            "paired_examples": True,
            "bootstrap_repetitions": 10000,
        },
        "mean_f1": {
            stage: {
                config: merged[stage][config]["mean_f1"] for config in CONFIGS
            }
            for stage in STAGES
        },
        "dataset_mean_f1": {
            stage: {
                config: merged[stage][config]["dataset_mean_f1"]
                for config in CONFIGS
            }
            for stage in STAGES
        },
        "paired_comparisons": comparisons,
        "persistent_store": stores,
        "memory_and_load": finite_memory_summary(sft_shards),
        "claim_boundaries": {
            "q16": "residual plus all lower-layer KV/recurrent/conv state at 16-bit",
            "q8": "residual plus all lower-layer KV/recurrent/conv state at 8-bit",
            "frozen_static": (
                "residual Q4, attention state Q4, linear state Q8, and lower "
                "layer bits [8,8,8,4,8,8,8]"
            ),
            "model_weights": "BF16 inference weights; Q values apply to persistent document state only",
            "sft_training": "dense full-model QA SFT; not quantization-aware training",
            "heldout_ce": "train-split checkpoint-selection diagnostic only",
            "downstream": "frozen LongBench validation source 6--35; not test-v2",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
