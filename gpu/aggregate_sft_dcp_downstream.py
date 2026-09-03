from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Iterable

from analyze_validation import comparison
from run_sft_dcp_downstream import CONFIGS as SFT_CONFIGS


BASELINE_CONFIGS = ("dense", "chunk-d7", "chunk-lora-d7")
EXPECTED_INDICES = set(range(6, 36))


def _read(paths: Iterable[Path]) -> list[dict[str, Any]]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]


def merge_config(
    shards: list[dict[str, Any]], config: str, *, expected_prefix: str
) -> dict[str, Any]:
    selected = [row for row in shards if row.get("config") == config]
    if len(selected) != 8:
        raise ValueError(f"{expected_prefix}/{config}: expected 8 shards, got {len(selected)}")
    if {int(row["rank"]) for row in selected} != set(range(8)):
        raise ValueError(f"{expected_prefix}/{config}: rank coverage is not 0--7")
    rows = [item for shard in selected for item in shard["rows"]]
    keys = {(row["dataset"], row["id"], int(row["source_index"])) for row in rows}
    if len(rows) != 60 or len(keys) != 60:
        raise ValueError(
            f"{expected_prefix}/{config}: expected 60 unique rows, got "
            f"{len(rows)}/{len(keys)}"
        )
    for dataset in ("qasper", "2wikimqa"):
        indices = {
            int(row["source_index"]) for row in rows if row["dataset"] == dataset
        }
        if indices != EXPECTED_INDICES:
            raise ValueError(f"{expected_prefix}/{config}/{dataset}: indices are not 6--35")
    return {
        "config": f"{expected_prefix}-{config}",
        "rows": rows,
        "mean_f1": statistics.fmean(float(row["f1"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate SFT DCP inference and the frozen Interface-LoRA baseline"
    )
    parser.add_argument("--sft-run-dir", type=Path, required=True)
    parser.add_argument("--interface-lora-run-dir", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--expected-lora-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sft_shards = _read(args.sft_run_dir.glob("sft-shard-*-*.json"))
    if len(sft_shards) != 8 * len(SFT_CONFIGS):
        raise SystemExit(
            f"expected {8 * len(SFT_CONFIGS)} SFT shards, got {len(sft_shards)}"
        )
    if {row.get("data_sha256") for row in sft_shards} != {
        args.expected_data_sha256
    }:
        raise SystemExit("SFT shards do not bind the expected validation SHA256")
    if {row.get("checkpoint_manifest_sha256") for row in sft_shards} != {
        args.expected_checkpoint_manifest_sha256
    }:
        raise SystemExit("SFT shards do not bind the selected checkpoint manifest")
    if not all(
        row.get("raw_test_v2_read") is False
        and row.get("checkpoint_payload_integrity_verified") is True
        for row in sft_shards
    ):
        raise SystemExit("SFT shard governance/integrity gate failed")

    baseline_shards = _read(args.interface_lora_run_dir.glob("shard-*-*.json"))
    if len(baseline_shards) != 8 * len(BASELINE_CONFIGS):
        raise SystemExit(
            f"expected {8 * len(BASELINE_CONFIGS)} Interface-LoRA shards, "
            f"got {len(baseline_shards)}"
        )
    if {row.get("data_sha256") for row in baseline_shards} != {
        args.expected_data_sha256
    }:
        raise SystemExit("Interface-LoRA shards use a different validation artifact")
    if {
        row.get("lora", {}).get("checkpoint_sha256") for row in baseline_shards
    } != {args.expected_lora_checkpoint_sha256}:
        raise SystemExit("Interface-LoRA shards use a different adapter checkpoint")
    frozen = {
        (
            int(row.get("max_input_tokens", -1)),
            int(row.get("chunk_size", -1)),
            int(row.get("overlap", -1)),
        )
        for row in sft_shards + baseline_shards
    }
    if frozen != {(4096, 512, 0)}:
        raise SystemExit(f"unified generation protocol drifted: {frozen}")
    generation_caps = {"qasper": 128, "2wikimqa": 32}
    for shard in sft_shards + baseline_shards:
        for row in shard["rows"]:
            if int(row.get("max_new_tokens", -1)) != generation_caps[row["dataset"]]:
                raise SystemExit("per-dataset generation cap drifted")

    sft = {
        config: merge_config(sft_shards, config, expected_prefix="sft")
        for config in SFT_CONFIGS
    }
    baseline = {
        config: merge_config(baseline_shards, config, expected_prefix="base")
        for config in BASELINE_CONFIGS
    }
    result = {
        "schema_version": "qcomem-unified-sft-downstream-validation-v1",
        "status": "completed",
        "samples": 60,
        "validation_data_sha256": args.expected_data_sha256,
        "source_indices": [6, 35],
        "checkpoint_manifest_sha256": args.expected_checkpoint_manifest_sha256,
        "interface_lora_checkpoint_sha256": args.expected_lora_checkpoint_sha256,
        "raw_test_v2_read": False,
        "protocol": {
            "prompt": "longbench-v1-official",
            "max_input_tokens": 4096,
            "dataset_max_new_tokens": generation_caps,
            "chunk_size": 512,
            "overlap": 0,
            "group_size": 64,
            "paired_examples": True,
        },
        "mean_f1": {
            "base_dense": baseline["dense"]["mean_f1"],
            "base_chunk_d7_q16": baseline["chunk-d7"]["mean_f1"],
            "interface_lora_chunk_d7_q16": baseline["chunk-lora-d7"]["mean_f1"],
            "sft_dense": sft["dense"]["mean_f1"],
        },
        "paired_comparisons": {
            "sft_dense_vs_base_dense": comparison(
                sft["dense"], baseline["dense"], seed=20260821
            ),
            "interface_lora_vs_base_chunk_q16": comparison(
                baseline["chunk-lora-d7"], baseline["chunk-d7"], seed=20260825
            ),
        },
        "claim_boundaries": {
            "sft_run_scope": "dense only; no CoMem or quantized state is run",
            "interface_lora_precision": "Q16 residual store; not Q4/Q8",
            "interface_lora_role": "frozen prior artifact, descriptive comparison",
            "heldout_ce_role": "checkpoint selection diagnostic only",
            "longbench_role": "frozen downstream validation source 6--35",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
