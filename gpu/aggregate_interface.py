from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from analyze_validation import comparison
from run_interface_diagnostic import CONFIGS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    shards = [
        json.loads(path.read_text())
        for path in sorted(args.run_dir.glob("shard-*-*.json"))
    ]
    expected_shards = 8 * len(CONFIGS)
    if len(shards) != expected_shards:
        raise SystemExit(f"expected {expected_shards} shard files, found {len(shards)}")

    merged: dict[str, dict[str, Any]] = {}
    for config_name in CONFIGS:
        matching = [shard for shard in shards if shard["config"] == config_name]
        if len(matching) != 8:
            raise SystemExit(f"{config_name}: expected 8 shards, found {len(matching)}")
        rows = [row for shard in matching for row in shard["rows"]]
        keys = {(row["dataset"], row["id"], row["source_index"]) for row in rows}
        if len(rows) != 64 or len(keys) != 64:
            raise SystemExit(
                f"{config_name}: expected 64 unique rows, found {len(rows)}/{len(keys)}"
            )
        merged[config_name] = {
            "config": config_name,
            "mode": matching[0]["mode"],
            "depth": matching[0]["depth"],
            "rows": rows,
            "mean_f1": statistics.fmean(row["f1"] for row in rows),
        }

    summaries = []
    for config_name in CONFIGS:
        config = merged[config_name]
        by_dataset = {
            dataset: statistics.fmean(
                row["f1"] for row in config["rows"] if row["dataset"] == dataset
            )
            for dataset in sorted({row["dataset"] for row in config["rows"]})
        }
        summaries.append(
            {
                "config": config_name,
                "mode": config["mode"],
                "depth": config["depth"],
                "mean_f1": config["mean_f1"],
                "dataset_f1": by_dataset,
                "mean_generation_seconds": statistics.fmean(
                    row["generation_seconds"] for row in config["rows"]
                ),
                "mean_write_seconds": (
                    statistics.fmean(
                        row["write_seconds"]
                        for row in config["rows"]
                        if row["write_seconds"] is not None
                    )
                    if config["mode"] in {"document", "chunk"}
                    else None
                ),
                "max_peak_allocated_bytes": max(
                    row["peak_allocated_bytes"] for row in config["rows"]
                ),
            }
        )

    dense = merged["dense"]
    analysis = {
        "status": "completed",
        "samples": 64,
        "summary": summaries,
        "oracle_d10_vs_dense": comparison(
            merged["oracle-d10"], dense, seed=20260811
        ),
        "depth": {},
    }
    for offset, depth in enumerate((7, 10, 13), start=1):
        document = merged[f"document-d{depth}"]
        chunk = merged[f"chunk-d{depth}"]
        analysis["depth"][str(depth)] = {
            "document_vs_dense": comparison(
                document, dense, seed=20260811 + offset * 100
            ),
            "chunk_vs_dense": comparison(
                chunk, dense, seed=20260811 + offset * 100 + 10
            ),
            "chunk_vs_document": comparison(
                chunk, document, seed=20260811 + offset * 100 + 20
            ),
        }
    destination = args.run_dir / "interface_analysis.json"
    destination.write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(analysis, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
