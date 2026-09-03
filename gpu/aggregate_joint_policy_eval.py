from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path
from typing import Any

from qcomem_joint_policy import merge_metric_sums, q16_exactness_passes
from run_downstream import atomic_json


def row_key(row: dict[str, Any]) -> tuple[str, int]:
    return str(row["source_object"]), int(row["start_token"])


def window_objective(row: dict[str, Any]) -> float:
    return float(merge_metric_sums([row["metrics"]])["joint_objective"])


def paired_bootstrap_ci(
    deltas: list[float], *, seed: int, samples: int = 20_000
) -> list[float]:
    if not deltas:
        raise ValueError("paired bootstrap requires non-empty deltas")
    generator = random.Random(seed)
    count = len(deltas)
    means = []
    for _ in range(samples):
        means.append(
            sum(deltas[generator.randrange(count)] for _ in range(count)) / count
        )
    means.sort()
    return [means[round(0.025 * (samples - 1))], means[round(0.975 * (samples - 1))]]


def _policy_bits(policy: dict[str, Any]) -> tuple[int, ...]:
    return (
        int(policy["residual_bits"]),
        *(int(value) for value in policy["cache_layer_bits"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze expanded PG-19 joint mixed-bit calibration candidates"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-shards", type=int, default=8)
    parser.add_argument("--bootstrap-seed", type=int, default=20260812)
    args = parser.parse_args()
    candidate_path = args.run_dir / "joint-policy-candidates.json"
    candidate_bytes = candidate_path.read_bytes()
    candidates = json.loads(candidate_bytes)
    expected_policies = {
        item["name"]: item for item in candidates["evaluation_policies"]
    }
    paths = sorted(args.run_dir.glob("joint-eval-*.json"))
    if len(paths) != args.expected_shards:
        raise SystemExit(f"expected {args.expected_shards} eval shards, found {len(paths)}")
    shards = [json.loads(path.read_text()) for path in paths]
    if any(shard.get("status") != "completed" for shard in shards):
        raise SystemExit("one or more joint eval shards are incomplete")
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    if {shard.get("candidate_file_sha256") for shard in shards} != {candidate_sha256}:
        raise SystemExit("eval shards did not use the frozen candidate file")
    if len({json.dumps(shard["protocol"], sort_keys=True) for shard in shards}) != 1:
        raise SystemExit("joint eval shards used different PG-19 windows/protocols")
    protocol = shards[0]["protocol"]
    if (
        protocol.get("longbench_labels_used") is not False
        or protocol.get("formal_validation_source_6_35_used") is not False
        or protocol.get("frozen_test_v2_source_68_99_used") is not False
    ):
        raise SystemExit("LongBench data leaked into joint policy selection")

    summaries: dict[str, dict[str, Any]] = {}
    rows_by_policy: dict[str, list[dict[str, Any]]] = {}
    for shard in shards:
        for summary in shard["summaries"]:
            name = str(summary["name"])
            if name in summaries:
                raise SystemExit(f"policy {name} was evaluated on multiple ranks")
            summaries[name] = summary
        for row in shard["rows"]:
            rows_by_policy.setdefault(str(row["policy"]), []).append(row)
    if summaries.keys() != expected_policies.keys():
        raise SystemExit(
            "evaluated policy set differs from frozen candidates: "
            f"actual={sorted(summaries)} expected={sorted(expected_policies)}"
        )
    expected_windows = int(protocol["calibration_books"])
    expected_keys: set[tuple[str, int]] | None = None
    for name, rows in rows_by_policy.items():
        keys = {row_key(row) for row in rows}
        if len(rows) != expected_windows or len(keys) != expected_windows:
            raise SystemExit(f"{name} does not have one row per PG-19 calibration book")
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise SystemExit(f"{name} used different PG-19 windows")

    q16 = summaries["q16-control"]
    frozen = summaries["frozen-static-control"]
    if not q16_exactness_passes(q16["metrics"]):
        raise SystemExit("joint Q16 control failed exactness")
    frozen_bytes = int(frozen["mean_persistent_nbytes"])
    frozen_objective = float(frozen["metrics"]["joint_objective"])
    minus25_measured_budget = round(frozen_bytes * 0.75)

    eligible: dict[str, list[dict[str, Any]]] = {
        "same_memory": [],
        "minus_25_percent": [],
    }
    for name, expected in expected_policies.items():
        summary = summaries[name]
        if _policy_bits(summary) != _policy_bits(expected):
            raise SystemExit(f"{name} policy bits drifted during joint evaluation")
        for group in expected.get("eligible_budgets", []):
            byte_count = int(summary["mean_persistent_nbytes"])
            if group == "same_memory" and byte_count > frozen_bytes:
                raise SystemExit(f"{name} exceeds measured frozen-static bytes")
            if group == "minus_25_percent" and byte_count > minus25_measured_budget:
                raise SystemExit(f"{name} exceeds measured minus-25% bytes")
            eligible[group].append(summary)
    if any(not values for values in eligible.values()):
        raise SystemExit("a joint policy budget has no evaluated automatic candidates")
    selected = {
        group: min(
            values,
            key=lambda item: (
                float(item["metrics"]["joint_objective"]),
                int(item["mean_persistent_nbytes"]),
                str(item["name"]),
            ),
        )
        for group, values in eligible.items()
    }

    frozen_rows = {row_key(row): row for row in rows_by_policy["frozen-static-control"]}
    comparisons = {}
    for offset, (group, summary) in enumerate(selected.items(), start=1):
        candidate_rows = {row_key(row): row for row in rows_by_policy[summary["name"]]}
        deltas = [
            window_objective(candidate_rows[key]) - window_objective(frozen_rows[key])
            for key in sorted(frozen_rows)
        ]
        comparisons[group] = {
            "candidate": summary["name"],
            "mean_joint_objective_delta_vs_frozen": statistics.fmean(deltas),
            "paired_bootstrap_95_ci": paired_bootstrap_ci(
                deltas, seed=args.bootstrap_seed + offset
            ),
            "candidate_beats_frozen_on_pg19_calibration": (
                float(summary["metrics"]["joint_objective"]) < frozen_objective
            ),
            "claim_limit": (
                "PG-19 train calibration comparison only; it is not a LongBench "
                "downstream result and cannot inherit a prior validation label"
            ),
        }

    policies = {
        "q16_control": {
            "residual_bits": q16["residual_bits"],
            "cache_layer_bits": q16["cache_layer_bits"],
            "mean_persistent_nbytes": q16["mean_persistent_nbytes"],
            "metrics": q16["metrics"],
        },
        "frozen_static_control": {
            "residual_bits": frozen["residual_bits"],
            "cache_layer_bits": frozen["cache_layer_bits"],
            "mean_persistent_nbytes": frozen["mean_persistent_nbytes"],
            "metrics": frozen["metrics"],
        },
    }
    for group, summary in selected.items():
        policies[f"pg19_joint_{group}_candidate"] = {
            "source_candidate_name": summary["name"],
            "residual_bits": summary["residual_bits"],
            "cache_layer_bits": summary["cache_layer_bits"],
            "mean_persistent_nbytes": summary["mean_persistent_nbytes"],
            "metrics": summary["metrics"],
            "promotion_status": (
                "calibration_candidate_only; requires a new independent downstream "
                "protocol after all policy choices freeze"
            ),
        }

    result = {
        "status": "pg19_joint_candidates_selected_not_downstream_validated",
        "stage": "expanded_pg19_joint_policy_selection",
        "protocol": protocol,
        "candidate_file_sha256": candidate_sha256,
        "budgets": {
            **candidates["budgets"],
            "measured_frozen_static_bytes": frozen_bytes,
            "measured_minus_25_percent_bytes": minus25_measured_budget,
        },
        "all_joint_summaries": [summaries[name] for name in sorted(summaries)],
        "selected_comparisons": comparisons,
        "policies": policies,
        "claim_boundary": {
            "policy_selection_data": "PG-19 train development calibration only",
            "natural_next_token_proxy_is_downstream_QA": False,
            "longbench_source_6_35_labels_used": False,
            "longbench_source_68_99_read": False,
            "reuse_prior_formal_validation_label": False,
            "reuse_legacy_policy_name": False,
            "frozen_static_remains_published_validation_control": True,
            "automatic_policy_is_new_downstream_result": False,
        },
    }
    destination = args.run_dir / "joint_policy.json"
    atomic_json(destination, result)
    print(
        json.dumps(
            {
                "saved": str(destination),
                "selected": {
                    group: summary["name"] for group, summary in selected.items()
                },
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
