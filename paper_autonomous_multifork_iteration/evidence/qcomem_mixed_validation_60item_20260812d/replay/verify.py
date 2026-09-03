#!/usr/bin/env python3
"""Independent raw-first verifier for the archival 60-item Q-CoMem run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import string
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "artifacts" / "raw"
EVIDENCE_ID = "E-QCOMEM-60-VALIDATION-20260812D-A"
DATA_SHA256 = "1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe"
SUMMARY_SHA256 = "09112a8b524de9ae47f0cdb38507659e9fb634cacf5e8ae4d080e7c59ad67266"
REMOTE_LEDGER_SHA256 = "2ae62ffa4839bf16a52f5c0f6b1e21b0fd55916e43f436af461dd072dce4c83d"
POLICY_SHA256 = "f34d4b89e9936c8d58d27df69268250f7985c9f4db1d9cef4d3041a06df36e87"
BOOTSTRAP_SEED = 20260811
BOOTSTRAP_REPETITIONS = 10_000
OVERALL_MARGIN = -0.02
DATASET_MARGIN = -0.03
CATASTROPHIC_DELTA = -0.5

CONFIGS = (
    "dense",
    "prefix",
    "replay-d7-layer-q16",
    "replay-d7-frozen-static",
    "replay-d7-same-memory-mixed",
    "replay-d7-minus25-mixed",
)

CONFIG_METADATA = {
    "dense": {
        "mode": "dense",
        "depth": None,
        "residual_bits": None,
        "attention_bits": None,
        "linear_bits": None,
        "cache_layer_bits": None,
        "policy": None,
    },
    "prefix": {
        "mode": "prefix",
        "depth": None,
        "residual_bits": None,
        "attention_bits": None,
        "linear_bits": None,
        "cache_layer_bits": None,
        "policy": None,
    },
    "replay-d7-layer-q16": {
        "mode": "replay",
        "depth": 7,
        "residual_bits": 16,
        "attention_bits": 16,
        "linear_bits": 16,
        "cache_layer_bits": [16, 16, 16, 16, 16, 16, 16],
        "policy": "layer-q16",
    },
    "replay-d7-frozen-static": {
        "mode": "replay",
        "depth": 7,
        "residual_bits": 4,
        "attention_bits": 4,
        "linear_bits": 8,
        "cache_layer_bits": [8, 8, 8, 4, 8, 8, 8],
        "policy": "frozen-static",
    },
    "replay-d7-same-memory-mixed": {
        "mode": "replay",
        "depth": 7,
        "residual_bits": 4,
        "attention_bits": 4,
        "linear_bits": None,
        "cache_layer_bits": [8, 8, 4, 4, 8, 8, 8],
        "policy": "same-memory-mixed",
    },
    "replay-d7-minus25-mixed": {
        "mode": "replay",
        "depth": 7,
        "residual_bits": 4,
        "attention_bits": 2,
        "linear_bits": None,
        "cache_layer_bits": [8, 8, 2, 2, 2, 8, 2],
        "policy": "minus25-mixed",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def require_equal(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, float):
        require(
            isinstance(actual, (int, float))
            and math.isclose(float(actual), expected, rel_tol=1e-15, abs_tol=1e-15),
            f"{label}: expected {expected!r}, observed {actual!r}",
        )
        return
    require(actual == expected, f"{label}: expected {expected!r}, observed {actual!r}")


def verify_package_manifest() -> None:
    manifest_path = ROOT / "MANIFEST.json"
    sidecar_path = ROOT / "MANIFEST.sha256"
    require(manifest_path.is_file(), "MANIFEST.json is missing")
    require(sidecar_path.is_file(), "MANIFEST.sha256 is missing")
    manifest = load_json(manifest_path)
    excludes = set(manifest["excludes"])
    require(excludes == {"MANIFEST.json", "MANIFEST.sha256"}, "manifest excludes drift")
    entries = manifest["files"]
    require(manifest["file_count"] == len(entries), "manifest file_count drift")
    expected_paths = {entry["relative_path"] for entry in entries}
    actual_paths = {
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.relative_to(ROOT).as_posix() not in excludes
    }
    require(expected_paths == actual_paths, "package manifest path closure failed")
    for entry in entries:
        path = ROOT / entry["relative_path"]
        require(path.stat().st_size == entry["bytes"], f"byte count drift: {path}")
        require(sha256(path) == entry["sha256"], f"SHA drift: {path}")
    expected_sidecar = f"{sha256(manifest_path)}  MANIFEST.json\n"
    require(sidecar_path.read_text() == expected_sidecar, "manifest sidecar drift")


def verify_remote_raw_ledger() -> int:
    ledger = RAW / "REMOTE_SHA256SUMS"
    require(sha256(ledger) == REMOTE_LEDGER_SHA256, "remote SHA ledger drift")
    entries: dict[str, str] = {}
    for line in ledger.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        require(relative.startswith("./"), f"non-relative remote path: {relative}")
        relative = relative[2:]
        require(relative not in entries, f"duplicate remote path: {relative}")
        entries[relative] = digest
    require(len(entries) == 66, f"expected 66 remote files, found {len(entries)}")
    actual = {
        path.relative_to(RAW).as_posix()
        for path in RAW.rglob("*")
        if path.is_file() and path.name != "REMOTE_SHA256SUMS"
    }
    require(actual == set(entries), "remote raw file-set closure failed")
    for relative, expected in entries.items():
        require(sha256(RAW / relative) == expected, f"remote mirror SHA drift: {relative}")
    return len(entries)


def row_key(row: dict[str, Any]) -> tuple[str, str, int]:
    return row["dataset"], row["id"], int(row["source_index"])


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    common = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def token_position_agreement(reference: list[int], candidate: list[int]) -> float:
    denominator = max(len(reference), len(candidate))
    if denominator == 0:
        return 1.0
    matches = sum(left == right for left, right in zip(reference, candidate))
    return matches / denominator


def bootstrap_mean_ci(values: list[float], *, seed: int) -> list[float]:
    require(bool(values), "bootstrap values are empty")
    generator = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(BOOTSTRAP_REPETITIONS)
    )
    return [
        estimates[int(0.025 * (BOOTSTRAP_REPETITIONS - 1))],
        estimates[int(0.975 * (BOOTSTRAP_REPETITIONS - 1))],
    ]


def error_rmse(rows: list[dict[str, Any]], field: str) -> float | None:
    selected = [row for row in rows if row.get(field) is not None]
    if not selected:
        return None
    squared = sum(row[field]["squared_error_sum"] for row in selected)
    reference = sum(row[field]["reference_squared_sum"] for row in selected)
    return 0.0 if reference == 0 else math.sqrt(squared / reference)


def cache_rmse(rows: list[dict[str, Any]], category: str) -> float | None:
    selected = [row for row in rows if row.get("cache_error_sums") is not None]
    if not selected:
        return None
    squared = sum(
        row["cache_error_sums"][category]["squared_error_sum"] for row in selected
    )
    reference = sum(
        row["cache_error_sums"][category]["reference_squared_sum"]
        for row in selected
    )
    return 0.0 if reference == 0 else math.sqrt(squared / reference)


def merge_and_verify_shards() -> tuple[dict[str, list[dict[str, Any]]], int]:
    paths = sorted(RAW.glob("shard-*-*.json"))
    require(len(paths) == 48, f"expected 48 shards, found {len(paths)}")
    shards = [load_json(path) for path in paths]
    merged: dict[str, list[dict[str, Any]]] = {}
    f1_checks = 0
    expected_model = (
        "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/"
        "indep-bench_assets/models/Qwen3.5-35B-A3B-59d61f3"
    )
    expected_data = (
        "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/"
        "indep-bench_assets/data/qcomem-longbench-validation/longbench_validation.jsonl"
    )

    for config_name in CONFIGS:
        matching = [shard for shard in shards if shard["config"] == config_name]
        require(len(matching) == 8, f"{config_name}: expected 8 shards")
        require({shard["rank"] for shard in matching} == set(range(8)), f"{config_name}: rank drift")
        expected_metadata = CONFIG_METADATA[config_name]
        rows: list[dict[str, Any]] = []
        for shard in matching:
            require(shard["world_size"] == 8, f"{config_name}: world_size drift")
            require(shard["suite"] == "layer-validation", f"{config_name}: suite drift")
            require(shard["data_sha256"] == DATA_SHA256, f"{config_name}: data SHA drift")
            require(shard["data"] == expected_data, f"{config_name}: data path drift")
            require(shard["model"] == expected_model, f"{config_name}: model path drift")
            require(shard["prompt_protocol"] == "longbench-v1-official", f"{config_name}: prompt drift")
            require(shard["max_input_tokens"] == 4096, f"{config_name}: input cap drift")
            require(shard["source_index_start"] == 6, f"{config_name}: start drift")
            require(shard["source_index_end"] == 35, f"{config_name}: end drift")
            require(shard["exclude_source_indices"] == [4, 5], f"{config_name}: exclusion drift")
            for field, expected in expected_metadata.items():
                require_equal(shard.get(field), expected, f"{config_name}/{field}")
            require(shard["samples"] == len(shard["rows"]), f"{config_name}: shard row count drift")
            rows.extend(shard["rows"])

        keys = [row_key(row) for row in rows]
        require(len(rows) == 60, f"{config_name}: expected 60 rows")
        require(len(set(keys)) == 60, f"{config_name}: duplicate rows")
        for dataset in ("qasper", "2wikimqa"):
            observed = {
                int(row["source_index"]) for row in rows if row["dataset"] == dataset
            }
            require(observed == set(range(6, 36)), f"{config_name}/{dataset}: cohort drift")

        for row in rows:
            source_index = int(row["source_index"])
            require(source_index not in {4, 5}, f"{config_name}: calibration leakage")
            require(not 68 <= source_index <= 99, f"{config_name}: test-v2 leakage")
            expected_max_new = 128 if row["dataset"] == "qasper" else 32
            require(row["max_new_tokens"] == expected_max_new, f"{config_name}: max-new drift")
            require(row["input_tokens"] <= 4096, f"{config_name}: input cap exceeded")
            require("ttft_seconds" not in row, f"{config_name}: unexpected TTFT field")
            recomputed_f1 = max(
                answer_f1(row["prediction"], reference)
                for reference in row["references"]
            )
            require_equal(row["f1"], recomputed_f1, f"{config_name}: F1 drift")
            f1_checks += 1
            persistent = row["stored_persistent_nbytes"]
            if persistent is not None:
                components = (row["stored_residual_nbytes"] or 0) + (
                    row["stored_lower_cache_nbytes"] or 0
                )
                require(persistent == components, f"{config_name}: Store denominator drift")
        merged[config_name] = rows

    expected_keys = {row_key(row) for row in merged["dense"]}
    for config_name, rows in merged.items():
        require(
            {row_key(row) for row in rows} == expected_keys,
            f"{config_name}: paired cohort mismatch",
        )
    return merged, f1_checks


def verify_summary(merged: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], int]:
    summary_path = RAW / "replay_analysis.json"
    require(sha256(summary_path) == SUMMARY_SHA256, "archival summary SHA drift")
    require((RAW / "aggregate.log").read_bytes() == summary_path.read_bytes(), "aggregate log drift")
    archival = load_json(summary_path)
    require(archival["status"] == "completed", "aggregate status drift")
    require(archival["samples"] == 60, "aggregate sample count drift")
    require(archival["suite"] == "layer-validation", "aggregate suite drift")
    require(
        archival["mean_noninferiority_thresholds"]
        == {
            "overall_mean_f1_delta": OVERALL_MARGIN,
            "per_dataset_mean_f1_delta": DATASET_MARGIN,
        },
        "aggregate margins drift",
    )
    require(
        archival["catastrophic_regression_threshold"] == CATASTROPHIC_DELTA,
        "catastrophic threshold drift",
    )
    require(
        archival["q16_replay_reference_config"] == "replay-d7-layer-q16",
        "Q16 reference drift",
    )
    require(
        archival["source_index_protocol"]
        == {"start_inclusive": 6, "end_inclusive": 35, "excluded": [4, 5]},
        "source-index protocol drift",
    )

    dense_by_key = {row_key(row): row for row in merged["dense"]}
    q16_by_key = {row_key(row): row for row in merged["replay-d7-layer-q16"]}
    dense_mean = statistics.fmean(row["f1"] for row in merged["dense"])
    prefix_mean_store = statistics.fmean(
        row["stored_persistent_nbytes"] for row in merged["prefix"]
    )
    q16_mean_store = statistics.fmean(
        row["stored_persistent_nbytes"] for row in merged["replay-d7-layer-q16"]
    )
    verified_metrics: list[dict[str, Any]] = []
    bootstrap_intervals = 0

    require(len(archival["summary"]) == len(CONFIGS), "summary config count drift")
    for config_index, config_name in enumerate(CONFIGS):
        rows = merged[config_name]
        observed = archival["summary"][config_index]
        require(observed["config"] == config_name, f"{config_name}: summary order drift")
        metadata = CONFIG_METADATA[config_name]
        for field in ("depth", "residual_bits", "attention_bits", "linear_bits", "cache_layer_bits", "policy"):
            require_equal(observed[field], metadata[field], f"{config_name}: summary {field}")
        require_equal(observed["bits"], metadata["residual_bits"], f"{config_name}: bits alias")
        require(observed["samples"] == 60, f"{config_name}: summary samples drift")

        f1_deltas = [row["f1"] - dense_by_key[row_key(row)]["f1"] for row in rows]
        q16_deltas = [row["f1"] - q16_by_key[row_key(row)]["f1"] for row in rows]
        mean_f1 = statistics.fmean(row["f1"] for row in rows)
        require_equal(observed["mean_f1"], mean_f1, f"{config_name}: mean F1")
        require_equal(observed["mean_f1_delta_vs_dense"], mean_f1 - dense_mean, f"{config_name}: dense delta")
        expected_ci_dense = bootstrap_mean_ci(
            f1_deltas, seed=BOOTSTRAP_SEED + config_index * 100
        )
        require_equal(observed["paired_bootstrap_95_ci_vs_dense"], expected_ci_dense, f"{config_name}: dense CI")
        bootstrap_intervals += 1

        datasets = sorted({row["dataset"] for row in rows})
        dataset_means = {
            dataset: statistics.fmean(
                row["f1"] for row in rows if row["dataset"] == dataset
            )
            for dataset in datasets
        }
        dense_dataset_means = {
            dataset: statistics.fmean(
                row["f1"]
                for row in dense_by_key.values()
                if row["dataset"] == dataset
            )
            for dataset in datasets
        }
        dataset_deltas = {
            dataset: dataset_means[dataset] - dense_dataset_means[dataset]
            for dataset in datasets
        }
        require_equal(observed["dataset_mean_f1"], dataset_means, f"{config_name}: dataset means")
        require_equal(observed["dataset_mean_f1_delta_vs_dense"], dataset_deltas, f"{config_name}: dataset deltas")
        for dataset_index, dataset in enumerate(datasets, start=1):
            values = [
                row["f1"] - dense_by_key[row_key(row)]["f1"]
                for row in rows
                if row["dataset"] == dataset
            ]
            expected_ci = bootstrap_mean_ci(
                values,
                seed=BOOTSTRAP_SEED + config_index * 100 + dataset_index,
            )
            require_equal(
                observed["dataset_paired_bootstrap_95_ci_vs_dense"][dataset],
                expected_ci,
                f"{config_name}/{dataset}: dense CI",
            )
            bootstrap_intervals += 1

        catastrophic_rate = statistics.fmean(
            delta <= CATASTROPHIC_DELTA for delta in f1_deltas
        )
        require_equal(
            observed["catastrophic_regression_rate_delta_le_minus_0_5"],
            catastrophic_rate,
            f"{config_name}: catastrophic rate",
        )
        require_equal(
            observed["mean_f1_delta_vs_q16_replay"],
            statistics.fmean(q16_deltas),
            f"{config_name}: Q16 delta",
        )
        expected_ci_q16 = bootstrap_mean_ci(
            q16_deltas, seed=BOOTSTRAP_SEED + config_index * 100 + 50
        )
        require_equal(
            observed["paired_bootstrap_95_ci_vs_q16_replay"],
            expected_ci_q16,
            f"{config_name}: Q16 CI",
        )
        bootstrap_intervals += 1

        prediction_q16 = [
            row["prediction"] == q16_by_key[row_key(row)]["prediction"] for row in rows
        ]
        token_q16 = [
            row["generated_token_ids"]
            == q16_by_key[row_key(row)]["generated_token_ids"]
            for row in rows
        ]
        token_position_q16 = [
            token_position_agreement(
                q16_by_key[row_key(row)]["generated_token_ids"],
                row["generated_token_ids"],
            )
            for row in rows
        ]
        require_equal(
            observed["prediction_exact_agreement_rate_vs_q16_replay"],
            statistics.fmean(prediction_q16),
            f"{config_name}: Q16 prediction agreement",
        )
        require_equal(
            observed["token_sequence_exact_agreement_rate_vs_q16_replay"],
            statistics.fmean(token_q16),
            f"{config_name}: Q16 token agreement",
        )
        require_equal(
            observed["mean_token_position_agreement_vs_q16_replay"],
            statistics.fmean(token_position_q16),
            f"{config_name}: Q16 token-position agreement",
        )

        expected_pass = (mean_f1 - dense_mean >= OVERALL_MARGIN) and all(
            delta >= DATASET_MARGIN for delta in dataset_deltas.values()
        )
        require(
            observed["passes_mean_noninferiority_vs_dense"] == expected_pass,
            f"{config_name}: mean-margin gate drift",
        )
        prediction_dense = [
            row["prediction"] == dense_by_key[row_key(row)]["prediction"] for row in rows
        ]
        token_dense = [
            row["generated_token_ids"]
            == dense_by_key[row_key(row)]["generated_token_ids"]
            for row in rows
        ]
        token_position_dense = [
            token_position_agreement(
                dense_by_key[row_key(row)]["generated_token_ids"],
                row["generated_token_ids"],
            )
            for row in rows
        ]
        require(observed["prediction_matches_dense"] == sum(prediction_dense), f"{config_name}: dense prediction count")
        require_equal(observed["prediction_exact_agreement_rate_vs_dense"], statistics.fmean(prediction_dense), f"{config_name}: dense prediction agreement")
        require(observed["token_matches_dense"] == sum(token_dense), f"{config_name}: dense token count")
        require_equal(observed["token_sequence_exact_agreement_rate_vs_dense"], statistics.fmean(token_dense), f"{config_name}: dense token agreement")
        require_equal(observed["mean_token_position_agreement_vs_dense"], statistics.fmean(token_position_dense), f"{config_name}: dense token-position agreement")
        require(observed["all_tokens_match_dense"] == all(token_dense), f"{config_name}: dense all-token gate")

        state_rows = [row for row in rows if row["stored_persistent_nbytes"] is not None]
        residual_rows = [row for row in rows if row["stored_residual_nbytes"] is not None]
        expected_write = (
            statistics.fmean(row["write_seconds"] for row in state_rows)
            if state_rows
            else None
        )
        require_equal(observed["mean_write_seconds"], expected_write, f"{config_name}: write mean")
        require_equal(
            observed["mean_generation_seconds"],
            statistics.fmean(row["generation_seconds"] for row in rows),
            f"{config_name}: generation mean",
        )

        expected_residual_mib = (
            statistics.fmean(row["stored_residual_nbytes"] for row in residual_rows)
            / 2**20
            if residual_rows
            else None
        )
        expected_lower_mib = (
            statistics.fmean(row["stored_lower_cache_nbytes"] for row in state_rows)
            / 2**20
            if state_rows
            else None
        )
        expected_persistent_mib = (
            statistics.fmean(row["stored_persistent_nbytes"] for row in state_rows)
            / 2**20
            if state_rows
            else None
        )
        require_equal(observed["mean_residual_mib"], expected_residual_mib, f"{config_name}: residual MiB")
        require_equal(observed["mean_lower_cache_mib"], expected_lower_mib, f"{config_name}: cache MiB")
        require_equal(observed["mean_persistent_mib"], expected_persistent_mib, f"{config_name}: persistent MiB")
        expected_persistent_bytes = (
            {
                "mean": statistics.fmean(
                    row["stored_persistent_nbytes"] for row in state_rows
                ),
                "min": min(row["stored_persistent_nbytes"] for row in state_rows),
                "max": max(row["stored_persistent_nbytes"] for row in state_rows),
                "sum": sum(row["stored_persistent_nbytes"] for row in state_rows),
            }
            if state_rows
            else None
        )
        require_equal(observed["persistent_bytes"], expected_persistent_bytes, f"{config_name}: persistent bytes")
        require_equal(observed["residual_relative_rmse"], error_rmse(rows, "residual_error_sums"), f"{config_name}: residual RMSE")
        require_equal(observed["attention_cache_relative_rmse"], cache_rmse(rows, "attention"), f"{config_name}: attention RMSE")
        require_equal(observed["linear_cache_relative_rmse"], cache_rmse(rows, "linear"), f"{config_name}: linear RMSE")
        require_equal(
            observed["max_peak_allocated_gib"],
            max(row["peak_allocated_bytes"] for row in rows) / 2**30,
            f"{config_name}: peak allocated",
        )
        expected_prefix_compression = (
            prefix_mean_store
            / statistics.fmean(row["stored_persistent_nbytes"] for row in state_rows)
            if state_rows
            else None
        )
        expected_q16_compression = (
            q16_mean_store
            / statistics.fmean(row["stored_persistent_nbytes"] for row in state_rows)
            if state_rows
            else None
        )
        require_equal(observed["persistent_compression_vs_prefix"], expected_prefix_compression, f"{config_name}: prefix compression")
        require_equal(observed["persistent_compression_vs_q16_replay"], expected_q16_compression, f"{config_name}: Q16 compression")

        verified_metrics.append(
            {
                "config": config_name,
                "mean_f1": observed["mean_f1"],
                "mean_f1_delta_vs_dense": observed["mean_f1_delta_vs_dense"],
                "mean_f1_delta_vs_q16_replay": observed[
                    "mean_f1_delta_vs_q16_replay"
                ],
                "paired_bootstrap_95_ci_vs_q16_replay": observed[
                    "paired_bootstrap_95_ci_vs_q16_replay"
                ],
                "mean_persistent_mib": observed["mean_persistent_mib"],
                "persistent_compression_vs_prefix": observed[
                    "persistent_compression_vs_prefix"
                ],
                "catastrophic_regressions": round(catastrophic_rate * 60),
                "token_sequence_exact_agreement_rate_vs_q16_replay": observed[
                    "token_sequence_exact_agreement_rate_vs_q16_replay"
                ],
            }
        )

    replay_configs = [
        config for config in archival["summary"] if config["config"].startswith("replay-")
    ]
    require(
        archival["all_replay_tokens_match_dense"]
        == all(config["all_tokens_match_dense"] for config in replay_configs),
        "all-replay-token gate drift",
    )
    return verified_metrics, bootstrap_intervals


def verify_auxiliary_receipts() -> dict[str, Any]:
    platform = load_json(ROOT / "platform_receipt.json")
    require(platform["job_id"] == 234340, "platform Job drift")
    require(platform["trial_id"] == 1830116, "platform Trial drift")
    require(platform["status"] == "Complete", "platform terminal status drift")
    command = platform["command"]
    for fragment in (
        "LIMIT_PER_DATASET=30",
        "SOURCE_INDEX_START=6",
        "SOURCE_INDEX_END=35",
        "EXCLUDE_SOURCE_INDICES=4,5",
        "MAX_INPUT_TOKENS=4096",
        "MAX_NEW_TOKENS=128",
        "OVERALL_MARGIN=-0.02",
        "DATASET_MARGIN=-0.03",
        "mixed-validation-rlmain-20260812d",
    ):
        require(fragment in command, f"platform command missing {fragment}")
    require("test-v2" not in command.lower(), "test-v2 path appears in submitted command")

    submitted = ROOT / "submission" / "qcomem-mixed-validation-reasoning.yaml"
    require(
        sha256(submitted)
        == "46a30fef2376ba5703b1bf7aa392a0f67e523d2235d94833abfb3759d5e3cb3c",
        "submitted YAML drift",
    )
    require(command in submitted.read_text(), "submitted YAML command mismatch")

    input_receipt = load_json(ROOT / "input_receipt.json")
    require(input_receipt["data"]["sha256"] == DATA_SHA256, "input receipt data SHA drift")
    require(
        input_receipt["data"]["source_revision"]
        == "5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
        "LongBench revision drift",
    )
    require(input_receipt["model"]["full_weight_ledger_bound_at_execution"] is False, "weight-boundary drift")
    require(input_receipt["source_ledger_bound_at_execution"] is False, "source-boundary drift")

    policy_path = ROOT / "calibration" / "layer_policy.json"
    require(sha256(policy_path) == POLICY_SHA256, "layer policy SHA drift")
    policy = load_json(policy_path)
    require(policy["status"] == "completed", "layer policy status drift")
    require(
        policy["frozen_static_policy"]
        == {
            "residual_bits": 4,
            "attention_bits": 4,
            "linear_bits": 8,
            "predicted_bytes": 6956256,
        },
        "frozen-static policy drift",
    )
    derived_frozen_layers = [
        policy["frozen_static_policy"][
            "linear_bits" if component["is_linear"] else "attention_bits"
        ]
        for component in policy["components"]
        if component["component"].startswith("cache.")
    ]
    require(derived_frozen_layers == [8, 8, 8, 4, 8, 8, 8], "frozen layer vector drift")
    require(
        policy["policies"]["same_memory_as_frozen"]["cache_layer_bits"]
        == [8, 8, 4, 4, 8, 8, 8],
        "same-memory policy drift",
    )
    require(
        policy["policies"]["minus_25_percent"]["cache_layer_bits"]
        == [8, 8, 2, 2, 2, 8, 2],
        "minus-25 policy drift",
    )
    calibration = load_json(ROOT / "calibration" / "receipt.json")
    require(calibration["trial_id"] == 1827870, "calibration Trial drift")
    require(calibration["status"] == "Complete", "calibration terminal drift")
    require(calibration["calibration_indices_per_dataset"] == [4, 5], "calibration split drift")
    require(calibration["policy_before_validation"] is True, "calibration chronology drift")

    stages = {
        path.name: path.read_text().strip() for path in sorted((RAW / "stages").iterdir())
    }
    require(
        set(stages) == {"00_start", "01_protocol_ok", "02_exactness_ok", "03_shards_done", "99_done"},
        "terminal stage set drift",
    )
    parsed_stages = [datetime.fromisoformat(stages[name].replace("Z", "+00:00")) for name in sorted(stages)]
    require(parsed_stages == sorted(parsed_stages), "stage chronology drift")

    smoke = load_json(RAW / "cached_smoke.json")
    require(smoke["status"] == "passed", "cached smoke status drift")
    require(all(smoke["matches_oracle"].values()), "cached smoke exactness drift")
    require(smoke["data_sha256"] == DATA_SHA256, "cached smoke data SHA drift")
    require(smoke["gpu"] == "NVIDIA H20-3e", "cached smoke GPU drift")
    require(smoke["torch"] == "2.11.0+cu129", "cached smoke torch drift")
    require(smoke["transformers"] == "5.14.1", "cached smoke transformers drift")

    gpu_lines = (RAW / "gpus.csv").read_text().splitlines()
    require(len(gpu_lines) == 9, "GPU inventory row count drift")
    require(all("NVIDIA H20-3e" in line and "143771 MiB" in line for line in gpu_lines[1:]), "GPU inventory drift")
    for rank in range(8):
        log = (RAW / "logs" / f"rank-{rank}.log").read_text()
        header = json.loads(log.splitlines()[0])
        require(header["rank"] == rank, f"rank {rank} log header drift")
        require(header["configs"] == list(CONFIGS), f"rank {rank} config list drift")
        require(header["suite"] == "layer-validation", f"rank {rank} suite drift")
        require(header["source_index_range"] == [6, 35], f"rank {rank} source range drift")
        require(header["excluded_source_indices"] == [4, 5], f"rank {rank} exclusion drift")
        require(header["torch"] == "2.11.0+cu129", f"rank {rank} torch drift")
        require(header["transformers"] == "5.14.1", f"rank {rank} transformers drift")
        require(header["gpu"] == "NVIDIA H20-3e", f"rank {rank} GPU drift")
        require(log.count("SAVED ") == 6, f"rank {rank} saved-shard count drift")

    claim_boundary = load_json(ROOT / "claim_boundary.json")
    require(claim_boundary["evidence_id"] == EVIDENCE_ID, "claim-boundary ID drift")
    require(claim_boundary["source_frozen_at_execution"] is False, "source-freeze boundary drift")
    require(claim_boundary["test_v2_used_by_this_package"] is False, "test-v2 boundary drift")
    return stages


def build_report() -> dict[str, Any]:
    remote_files = verify_remote_raw_ledger()
    merged, f1_checks = merge_and_verify_shards()
    metrics, bootstrap_intervals = verify_summary(merged)
    stages = verify_auxiliary_receipts()
    metric_by_name = {entry["config"]: entry for entry in metrics}
    frozen = metric_by_name["replay-d7-frozen-static"]
    prefix = metric_by_name["prefix"]
    same = metric_by_name["replay-d7-same-memory-mixed"]
    minus = metric_by_name["replay-d7-minus25-mixed"]
    return {
        "schema_version": "qcomem-60item-independent-replay-v1",
        "audit_status": "pass",
        "evidence_id": EVIDENCE_ID,
        "classification": "verified_bounded_archival_raw_first_validation",
        "platform": {
            "job_id": 234340,
            "trial_id": 1830116,
            "terminal_status": "Complete",
            "terminal_stage_markers": stages,
        },
        "raw_artifacts": {
            "authoritative_remote_files": remote_files,
            "configuration_rank_shards": 48,
            "rank_logs": 8,
            "remote_sha256_ledger": REMOTE_LEDGER_SHA256,
            "aggregate_sha256": SUMMARY_SHA256,
            "all_remote_mirror_hashes_match": True,
        },
        "cohort": {
            "datasets": {"2wikimqa": 30, "qasper": 30},
            "paired_items": 60,
            "configurations": 6,
            "raw_prediction_rows": 360,
            "source_indices_per_dataset": [6, 35],
            "calibration_indices_absent": [4, 5],
            "test_v2_indices_absent": [68, 99],
            "data_sha256": DATA_SHA256,
            "same_keys_in_all_configurations": True,
        },
        "recomputation": {
            "f1_rows_recomputed": f1_checks,
            "paired_bootstrap_intervals_recomputed": bootstrap_intervals,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "all_archival_summary_fields_match": True,
            "store_component_identity_holds_for_all_state_rows": True,
        },
        "metrics": metrics,
        "headline_checks": {
            "frozen_static_store_reduction_fraction_vs_prefix": (
                1 - frozen["mean_persistent_mib"] / prefix["mean_persistent_mib"]
            ),
            "frozen_static_store_compression_vs_prefix": frozen[
                "persistent_compression_vs_prefix"
            ],
            "frozen_static_q16_relative_f1_ci_crosses_zero": (
                frozen["paired_bootstrap_95_ci_vs_q16_replay"][0] <= 0
                <= frozen["paired_bootstrap_95_ci_vs_q16_replay"][1]
            ),
            "frozen_static_catastrophic_regressions": frozen[
                "catastrophic_regressions"
            ],
            "same_memory_store_reduction_fraction_vs_frozen_static": (
                1 - same["mean_persistent_mib"] / frozen["mean_persistent_mib"]
            ),
            "same_memory_catastrophic_regressions": same[
                "catastrophic_regressions"
            ],
            "minus25_q16_relative_f1_ci_upper_below_zero": (
                minus["paired_bootstrap_95_ci_vs_q16_replay"][1] < 0
            ),
            "minus25_catastrophic_regressions": minus[
                "catastrophic_regressions"
            ],
        },
        "measurement_availability": {
            "f1": True,
            "retained_tensor_payload_store": True,
            "token_and_prediction_agreement": True,
            "ttft": False,
            "tpot": False,
            "throughput": False,
            "recall": False,
            "process_or_nvml_memory": False,
        },
        "provenance_boundary": {
            "data_sha_bound_in_every_shard": True,
            "policy_precedes_validation": True,
            "source_ledger_bound_at_execution": False,
            "full_model_weight_ledger_bound_at_execution": False,
            "fresh_gpu_regeneration_supported": False,
            "offline_raw_outcome_replay_supported": True,
            "later_test_v2_used_or_cited": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-package-manifest", action="store_true")
    args = parser.parse_args()
    if not args.skip_package_manifest:
        verify_package_manifest()
    report = build_report()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
