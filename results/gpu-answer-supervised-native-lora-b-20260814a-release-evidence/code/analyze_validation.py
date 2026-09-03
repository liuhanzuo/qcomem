from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any


OVERALL_NONINFERIORITY_MARGIN = -0.02
DATASET_NONINFERIORITY_MARGIN = -0.03


def row_key(row: dict[str, Any]) -> tuple[str, str, int | None]:
    return row["dataset"], row["id"], row.get("source_index")


def paired_rows(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    candidate_by_key = {row_key(row): row for row in candidate["rows"]}
    reference_by_key = {row_key(row): row for row in reference["rows"]}
    if candidate_by_key.keys() != reference_by_key.keys():
        missing_candidate = sorted(reference_by_key.keys() - candidate_by_key.keys())
        missing_reference = sorted(candidate_by_key.keys() - reference_by_key.keys())
        raise ValueError(
            f"sample mismatch; missing candidate={missing_candidate[:3]}, "
            f"missing reference={missing_reference[:3]}"
        )
    return [
        (candidate_by_key[key], reference_by_key[key])
        for key in sorted(candidate_by_key)
    ]


def bootstrap_mean_ci(
    values: list[float], *, seed: int, repetitions: int = 10_000
) -> list[float]:
    if not values:
        raise ValueError("bootstrap requires at least one value")
    generator = random.Random(seed)
    count = len(values)
    estimates = sorted(
        statistics.fmean(values[generator.randrange(count)] for _ in range(count))
        for _ in range(repetitions)
    )
    lower = estimates[int(0.025 * (repetitions - 1))]
    upper = estimates[int(0.975 * (repetitions - 1))]
    return [lower, upper]


def comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    pairs = paired_rows(candidate, reference)
    deltas = [candidate_row["f1"] - reference_row["f1"] for candidate_row, reference_row in pairs]
    datasets = sorted({candidate_row["dataset"] for candidate_row, _ in pairs})
    by_dataset = {}
    for offset, dataset in enumerate(datasets, start=1):
        dataset_deltas = [
            candidate_row["f1"] - reference_row["f1"]
            for candidate_row, reference_row in pairs
            if candidate_row["dataset"] == dataset
        ]
        by_dataset[dataset] = {
            "samples": len(dataset_deltas),
            "mean_f1_delta": statistics.fmean(dataset_deltas),
            "paired_bootstrap_95_ci": bootstrap_mean_ci(
                dataset_deltas, seed=seed + offset
            ),
        }
    overall_mean = statistics.fmean(deltas)
    qualifies = (
        overall_mean >= OVERALL_NONINFERIORITY_MARGIN
        and all(
            metrics["mean_f1_delta"] >= DATASET_NONINFERIORITY_MARGIN
            for metrics in by_dataset.values()
        )
    )
    return {
        "candidate": candidate["config"],
        "reference": reference["config"],
        "samples": len(pairs),
        "mean_f1_delta": overall_mean,
        "mean_absolute_sample_f1_delta": statistics.fmean(abs(delta) for delta in deltas),
        "paired_bootstrap_95_ci": bootstrap_mean_ci(deltas, seed=seed),
        "dataset": by_dataset,
        "prediction_exact_agreement": statistics.fmean(
            candidate_row["prediction"] == reference_row["prediction"]
            for candidate_row, reference_row in pairs
        ),
        "output_length_agreement": statistics.fmean(
            candidate_row["generated_tokens"] == reference_row["generated_tokens"]
            for candidate_row, reference_row in pairs
        ),
        "catastrophic_regression_rate_delta_le_minus_0_5": statistics.fmean(
            delta <= -0.5 for delta in deltas
        ),
        "passes_preregistered_mean_margins": qualifies,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    args = parser.parse_args()
    configs = {
        payload["config"]: payload
        for path in sorted(args.run_dir.glob("config-*.json"))
        for payload in [json.loads(path.read_text())]
    }
    expected = {"dense"} | {
        f"d{depth}-q{bits}" for depth in (7, 10, 13) for bits in (4, 8, 16)
    }
    if configs.keys() != expected:
        raise SystemExit(
            f"expected {sorted(expected)}, found {sorted(configs)}"
        )

    interface = {}
    quantization = {}
    policy = {}
    for depth_index, depth in enumerate((7, 10, 13), start=1):
        q16 = configs[f"d{depth}-q16"]
        interface[str(depth)] = comparison(
            q16, configs["dense"], seed=args.bootstrap_seed + depth_index * 100
        )
        candidates = {}
        for bits_index, bits in enumerate((8, 4), start=1):
            candidates[str(bits)] = comparison(
                configs[f"d{depth}-q{bits}"],
                q16,
                seed=args.bootstrap_seed + depth_index * 100 + bits_index * 10,
            )
        quantization[str(depth)] = candidates
        policy[str(depth)] = next(
            bits
            for bits in (4, 8, 16)
            if bits == 16
            or candidates[str(bits)]["passes_preregistered_mean_margins"]
        )

    example = configs["dense"]
    report = {
        "status": "completed",
        "data": example["data"],
        "data_sha256": example["data_sha256"],
        "prompt_protocol": example["prompt_protocol"],
        "samples": len(example["rows"]),
        "preregistered_thresholds": {
            "overall_mean_f1_delta": OVERALL_NONINFERIORITY_MARGIN,
            "per_dataset_mean_f1_delta": DATASET_NONINFERIORITY_MARGIN,
            "bootstrap_seed": args.bootstrap_seed,
            "bootstrap_repetitions": 10_000,
            "confidence_interval_is_reported_not_used_as_gate": True,
        },
        "interface_gap_q16_vs_dense": interface,
        "quantization_gap_vs_same_depth_q16": quantization,
        "selected_depth_bit_policy": policy,
    }
    destination = args.output or args.run_dir / "validation_analysis.json"
    destination.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
