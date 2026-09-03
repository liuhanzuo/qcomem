#!/usr/bin/env python3
"""Blind replay for the same-protocol vLLM prefix-cache result bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


REMOTE_PREFIX = (
    "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/"
    "indep-bench_assets/runs/qcomem/"
    "related-vllm-prefix-bootstrap-f-20260820a/"
)
EXPECTED_PAIRS = [
    ("qasper", 6), ("qasper", 7), ("qasper", 8), ("qasper", 9),
    ("2wikimqa", 6), ("2wikimqa", 7), ("2wikimqa", 8), ("2wikimqa", 9),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-12)


def replay(root: Path) -> dict:
    summary = load_json(root / "summary.json")
    protocol = load_json(root / "protocol.json")
    environment = load_json(root / "environment.json")
    assignment = load_json(root / "gpu-assignment.json")

    ledger_rows = []
    for line in (root / "artifact-ledger.sha256").read_text(encoding="utf-8").splitlines():
        expected, remote_path = line.split(None, 1)
        require(remote_path.startswith(REMOTE_PREFIX), "artifact path escaped frozen run root")
        local_path = root / remote_path.removeprefix(REMOTE_PREFIX)
        require(local_path.is_file(), f"missing artifact: {local_path}")
        require(sha256(local_path) == expected, f"artifact hash mismatch: {local_path}")
        ledger_rows.append(remote_path.removeprefix(REMOTE_PREFIX))
    require(len(ledger_rows) == 66, "artifact ledger cardinality drift")
    require(len(ledger_rows) == len(set(ledger_rows)), "duplicate artifact-ledger path")

    require(summary["schema"] == "forkaudit-related-serving-summary-v1", "summary schema drift")
    require(summary["scientific_run_valid"] is True, "scientific run invalid")
    require(summary["scientific_outcome"] == "valid_positive", "unexpected outcome")
    require(summary["pairs"] == [list(row) for row in EXPECTED_PAIRS], "pair order drift")
    require(protocol["world_size"] == 8, "world size drift")
    require(protocol["datasets"] == ["qasper", "2wikimqa"], "dataset drift")
    require(protocol["source_indices"] == [6, 7, 8, 9], "source-index drift")
    require(protocol["input_token_cap"] == 4096, "input cap drift")
    require(protocol["max_new_tokens"] == 32, "generation cap drift")
    require(protocol["decoding"]["temperature"] == 0.0, "decoding drift")
    require(environment == {
        "python": "3.11.13",
        "torch": "2.11.0+cu129",
        "transformers": "5.14.1",
        "vllm": "0.26.0",
    }, "environment drift")
    require(assignment["schema"] == "related-baseline-gpu-assignment-v1", "GPU schema drift")
    require(len(assignment["uuids"]) == 8, "GPU count drift")
    require(len(set(assignment["uuids"])) == 8, "GPU UUID reuse")

    by_phase = {}
    raw_names = []
    for phase in ("cache_off", "cache_on"):
        rows = []
        for rank, pair in enumerate(EXPECTED_PAIRS):
            path = root / "raw" / f"{phase}-rank-{rank}.json"
            row = load_json(path)
            raw_names.append(path.name)
            require(row["schema"] == "forkaudit-related-serving-shard-v1", "shard schema drift")
            require(row["phase"] == phase and row["rank"] == rank, "phase/rank drift")
            require(row["status"] == "completed" and row["world_size"] == 8, "shard incomplete")
            require((row["workload"]["dataset"], row["workload"]["source_index"]) == pair,
                    "workload assignment drift")
            require(row["protocol"]["max_input_tokens"] == 4096, "shard input cap drift")
            require(row["protocol"]["max_new_tokens"] == 32, "shard generation cap drift")
            measured = row["measured"]
            require(measured["ttft_seconds"] > 0.0, "nonpositive TTFT")
            require(measured["median_tpot_seconds"] > 0.0, "nonpositive TPOT")
            require(measured["generated_tokens_per_second"] > 0.0, "nonpositive throughput")
            require(measured["usage"]["prompt_tokens"] <= 4096, "prompt cap exceeded")
            counters = row["prefix_counters"]["measured_delta"]
            if phase == "cache_off":
                require(counters["hits"] == 0.0 and counters["queries"] == 0.0,
                        "cache-off recorded prefix traffic")
            else:
                require(counters["hits"] > 0.0 and counters["queries"] > 0.0,
                        "cache-on failed to record a hit")
                require(close(counters["hit_rate"], counters["hits"] / counters["queries"]),
                        "prefix hit-rate arithmetic drift")
            rows.append(row)
        by_phase[phase] = rows

    require(len(raw_names) == 16 and len(set(raw_names)) == 16, "raw shard-set drift")
    predictions_equal = []
    for off, on in zip(by_phase["cache_off"], by_phase["cache_on"]):
        predictions_equal.append(off["measured"]["prediction"] == on["measured"]["prediction"])
        require(close(off["measured"]["f1"], on["measured"]["f1"]), "F1 changed across phases")
    require(predictions_equal == [True] * 8, "cache-on prediction divergence")

    derived = {}
    for phase, rows in by_phase.items():
        measured = [row["measured"] for row in rows]
        derived[phase] = {
            "mean_f1": statistics.fmean(row["f1"] for row in measured),
            "median_generated_tokens_per_second": statistics.median(
                row["generated_tokens_per_second"] for row in measured
            ),
            "median_tpot_seconds": statistics.median(row["median_tpot_seconds"] for row in measured),
            "median_ttft_seconds": statistics.median(row["ttft_seconds"] for row in measured),
        }
        for key, value in derived[phase].items():
            require(close(value, summary["phases"][phase][key]), f"aggregate drift: {phase}.{key}")

    warning_needles = {
        "cache_on": [
            "support for Mamba layers is experimental",
            "enable_prefix_caching=True",
            "Setting attention block size to 1056 tokens",
            "Padding mamba page size by 0.76%",
            "Using default MoE config. Performance might be sub-optimal",
            "Enforce eager set, disabling torch.compile and CUDAGraphs",
        ],
        "cache_off": [
            "enable_prefix_caching=False",
            "Setting attention block size to 1056 tokens",
            "Padding mamba page size by 0.76%",
            "Using default MoE config. Performance might be sub-optimal",
            "Enforce eager set, disabling torch.compile and CUDAGraphs",
        ],
    }
    for phase, needles in warning_needles.items():
        for rank in range(8):
            log = (root / "server-logs" / f"{phase}-rank-{rank}.log").read_text(
                encoding="utf-8", errors="replace"
            )
            for needle in needles:
                require(needle in log, f"missing runtime disclosure in {phase} rank {rank}: {needle}")

    stages = sorted(path.name for path in (root / "stages").iterdir() if path.is_file())
    require(stages == [
        "00_start", "10_preflight_complete", "20_cache_off_complete",
        "20_cache_on_complete", "30_aggregate_complete", "COMPLETE",
    ], "stage closure drift")

    return {
        "schema": "related-vllm-prefix-independent-validation-v1",
        "passed": True,
        "artifact_ledger_rows": len(ledger_rows),
        "raw_shards": 16,
        "unique_gpu_uuids": 8,
        "workloads": [list(row) for row in EXPECTED_PAIRS],
        "derived": derived,
        "cache_on_measured_hit_rates": [
            row["prefix_counters"]["measured_delta"]["hit_rate"]
            for row in by_phase["cache_on"]
        ],
        "cache_on_off_predictions_exact": predictions_equal,
        "runtime_disclosures_replayed": [
            "Mamba align prefix caching is experimental in vLLM 0.26",
            "eager mode disabled torch.compile and CUDA graphs",
            "attention block size resolved to 1056 tokens with 0.76% Mamba page padding",
            "the H20-3e MoE config was absent and vLLM used default tactics",
        ],
        "comparison_boundary": summary["comparison_boundary"],
        "not_comparable_to": summary["not_comparable_to"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = replay(args.root.resolve())
    encoded = json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
