from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from run_downstream import atomic_json


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["config"]].append(row)
    summaries = {}
    for config, config_rows in sorted(grouped.items()):
        hypic_ttft = [float(row["median_ttft_seconds"]) for row in config_rows]
        qcomem_ttft = [
            float(row["current_qcomem"]["median_ttft_seconds"])
            for row in config_rows
        ]
        full_ttft = [
            float(row["full_prefix_median_ttft_seconds"])
            for row in config_rows
        ]
        persistent = [
            int(
                row["persistent_bytes"]["profiles"]["full_suffix_local_cache"]
                ["persistent_nbytes"]
            )
            for row in config_rows
        ]
        summaries[config] = {
            "samples": len(config_rows),
            "median_ttft_seconds": statistics.median(hypic_ttft),
            "median_current_qcomem_ttft_seconds": statistics.median(qcomem_ttft),
            "median_full_prefix_ttft_seconds": statistics.median(full_ttft),
            "ratio_of_medians_vs_qcomem": (
                statistics.median(qcomem_ttft) / statistics.median(hypic_ttft)
            ),
            "ratio_of_medians_vs_full_prefix": (
                statistics.median(full_ttft) / statistics.median(hypic_ttft)
            ),
            "median_full_suffix_local_persistent_nbytes": statistics.median(
                persistent
            ),
            "same_packed_qcomem_top1_agreement": statistics.fmean(
                float(row["same_packed_qcomem_logits"]["top1_match"])
                for row in config_rows
            ),
            "exact_full_prefix_top1_agreement": statistics.fmean(
                float(row["exact_full_prefix_logits"]["top1_match"])
                for row in config_rows
            ),
            "median_relative_logit_l2_vs_qcomem": statistics.median(
                float(row["same_packed_qcomem_logits"]["relative_logit_l2_error"])
                for row in config_rows
            ),
            "median_saved_suffix_document_compute_fraction": statistics.median(
                float(row["request_work"]["saved_fraction"])
                for row in config_rows
            ),
        }
    return {
        "prototype_status": (
            "HYPIC-inspired reference prototype; independent full-attention KV "
            "splicing is approximate and this is not a complete HYPIC reproduction"
        ),
        "rows": len(rows),
        "configs": summaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate HYPIC-lite GPU shards")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-shards", type=int, default=8)
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob("hypic-lite-shard-*.json"))
    if len(paths) != args.expected_shards:
        raise SystemExit(
            f"expected {args.expected_shards} HYPIC-lite shards, found {len(paths)}"
        )
    shards = [json.loads(path.read_text()) for path in paths]
    failed = [
        str(path)
        for path, shard in zip(paths, shards)
        if shard.get("status") != "completed"
        or not shard.get("prototype_gate", {}).get("passed")
    ]
    if failed:
        raise SystemExit(f"incomplete or failed shards: {failed}")
    protocols = [json.dumps(shard["protocol"], sort_keys=True) for shard in shards]
    if len(set(protocols)) != 1:
        raise SystemExit("shards used inconsistent protocols")
    rows = [row for shard in shards for row in shard["rows"]]
    result = {
        "status": "completed",
        "shards": len(shards),
        "protocol": shards[0]["protocol"],
        "depth7_4k_storage_ledger": shards[0]["depth7_4k_storage_ledger"],
        "summary": summarize_rows(rows),
    }
    destination = args.run_dir / "hypic-lite-analysis.json"
    atomic_json(destination, result)
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
