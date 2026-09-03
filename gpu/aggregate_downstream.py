from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    configs = [json.loads(path.read_text()) for path in sorted(args.run_dir.glob("config-*.json"))]
    expected = 10
    if len(configs) != expected:
        raise SystemExit(f"expected {expected} config files, found {len(configs)}")

    rows = []
    for config in configs:
        config_rows = config["rows"]
        by_dataset = {}
        for row in config_rows:
            by_dataset.setdefault(row["dataset"], []).append(row["f1"])
        total_generated_tokens = sum(row["generated_tokens"] for row in config_rows)
        total_generation_seconds = sum(row["generation_seconds"] for row in config_rows)
        rows.append(
            {
                "config": config["config"],
                "depth": config["depth"],
                "bits": config["bits"],
                "mean_f1": config["mean_f1"],
                "dataset_f1": {
                    dataset: statistics.fmean(scores)
                    for dataset, scores in sorted(by_dataset.items())
                },
                "mean_generation_seconds": config["mean_generation_seconds"],
                "total_generated_tokens": total_generated_tokens,
                "total_generation_seconds": total_generation_seconds,
                "seconds_per_generated_token": (
                    total_generation_seconds / max(total_generated_tokens, 1)
                ),
                "model_allocated_bytes": config["model_allocated_bytes"],
                "max_peak_allocated_bytes": max(
                    row["peak_allocated_bytes"] for row in config_rows
                ),
                "max_incremental_peak_allocated_bytes": max(
                    row["incremental_peak_allocated_bytes"] for row in config_rows
                ),
                "mean_write_seconds": (
                    statistics.fmean(row["write_seconds"] for row in config_rows)
                    if config["bits"] is not None
                    else None
                ),
                "mean_quantize_dequantize_seconds": (
                    statistics.fmean(
                        row["quantize_dequantize_seconds"] for row in config_rows
                    )
                    if config["bits"] is not None
                    else None
                ),
                "mean_stored_residual_nbytes": (
                    statistics.fmean(
                        row["stored_residual_nbytes"] for row in config_rows
                    )
                    if config["bits"] is not None
                    else None
                ),
                "mean_compression_ratio": (
                    statistics.fmean(
                        row["compression_ratio"] for row in config_rows
                    )
                    if config["bits"] is not None
                    else None
                ),
                "mean_relative_rmse": (
                    statistics.fmean(
                        row["residual_relative_rmse"] for row in config_rows
                    )
                    if config["bits"] is not None
                    else None
                ),
            }
        )
    rows.sort(key=lambda row: (row["depth"] is not None, row["depth"] or -1, -(row["bits"] or 0)))
    summary = {
        "status": "completed",
        "configs": len(rows),
        "results": rows,
    }
    (args.run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
