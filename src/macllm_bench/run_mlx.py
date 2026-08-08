from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


METRIC_PATTERNS = {
    "prompt_tokens": re.compile(r"Prompt:\s*(\d+)\s*tokens", re.I),
    "prompt_tokens_per_second": re.compile(
        r"Prompt:.*?([0-9.]+)\s*tokens-per-sec", re.I
    ),
    "generation_tokens": re.compile(r"Generation:\s*(\d+)\s*tokens", re.I),
    "generation_tokens_per_second": re.compile(
        r"Generation:.*?([0-9.]+)\s*tokens-per-sec", re.I
    ),
    "peak_memory_gb": re.compile(r"Peak memory:\s*([0-9.]+)\s*GB", re.I),
}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    required = {"name", "model", "prompt", "max_tokens", "measured_runs"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    return config


def parse_metrics(output: str) -> dict[str, float | int | None]:
    parsed: dict[str, float | int | None] = {}
    for name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(output)
        if not match:
            parsed[name] = None
        elif name.endswith("tokens"):
            parsed[name] = int(match.group(1))
        else:
            parsed[name] = float(match.group(1))
    return parsed


def run_once(config: dict[str, Any], run_index: int, warmup: bool) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "mlx_lm.generate",
        "--model",
        config["model"],
        "--prompt",
        config["prompt"],
        "--max-tokens",
        str(config["max_tokens"]),
        "--temp",
        str(config.get("temperature", 0.0)),
        "--seed",
        str(config.get("seed", 42)),
    ]
    started = time.perf_counter()
    process = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - started
    combined = process.stdout + "\n" + process.stderr
    raw_path = Path("results/raw") / f"{config['name']}-{run_index}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(combined)
    if process.returncode:
        raise RuntimeError(f"MLX-LM failed; inspect {raw_path}")
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "experiment": config["name"],
        "runtime": "mlx-lm",
        "model": config["model"],
        "run_index": run_index,
        "warmup": warmup,
        "wall_time_seconds": elapsed,
        "max_tokens": config["max_tokens"],
        "temperature": config.get("temperature", 0.0),
        "seed": config.get("seed", 42),
        **parse_metrics(combined),
        "raw_output": str(raw_path),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row for row in rows if not row["warmup"]]
    metrics = [
        "wall_time_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "peak_memory_gb",
    ]
    summary: dict[str, Any] = {"measured_runs": len(measured)}
    for metric in metrics:
        values = [float(row[metric]) for row in measured if row.get(metric) is not None]
        summary[metric] = (
            {
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "min": min(values),
                "max": max(values),
            }
            if values
            else None
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/smoke.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    if platform_machine := os.uname().machine:
        if platform_machine != "arm64":
            raise SystemExit(f"MLX requires Apple Silicon arm64; found {platform_machine}")

    warmups = int(config.get("warmup_runs", 1))
    measured = int(config["measured_runs"])
    rows: list[dict[str, Any]] = []
    for index in range(warmups + measured):
        warmup = index < warmups
        print(f"Run {index + 1}/{warmups + measured} ({'warmup' if warmup else 'measured'})")
        row = run_once(config, index, warmup)
        rows.append(row)
        print(json.dumps(row, indent=2))

    result_path = Path("results/runs.jsonl")
    with result_path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = summarize(rows)
    summary_path = Path("results/summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\nSaved: {result_path}")
    print(f"Saved: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
