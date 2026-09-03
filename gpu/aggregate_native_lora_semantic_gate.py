from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from run_downstream import atomic_json


def aggregate_native_semantic(
    shards: list[dict[str, Any]],
    *,
    expected_world_size: int,
    expected_data_sha256: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    if len(shards) != expected_world_size or {
        shard.get("rank") for shard in shards
    } != set(range(expected_world_size)):
        raise ValueError("native semantic shards do not cover every rank")
    if any(shard.get("status") != "completed_shard" for shard in shards):
        raise ValueError("native semantic shard is incomplete")
    if {shard.get("data_sha256") for shard in shards} != {expected_data_sha256}:
        raise ValueError("native semantic data SHA256 drifted")
    if {shard.get("checkpoint_sha256") for shard in shards} != {
        expected_checkpoint_sha256
    }:
        raise ValueError("native semantic checkpoint SHA256 drifted")
    if any(shard.get("test_v2_used") is not False for shard in shards):
        raise ValueError("native semantic gate read test-v2")
    requested = {shard.get("global_samples_requested") for shard in shards}
    if len(requested) != 1:
        raise ValueError("native semantic requested sample count drifted")
    rows = [row for shard in shards for row in shard["rows"]]
    if len(rows) != next(iter(requested)) or len(
        {row["example_id"] for row in rows}
    ) != len(rows):
        raise ValueError("native semantic examples are incomplete or duplicated")
    if any(row["cache_audit"].get("hard_gate_passed") is not True for row in rows):
        raise ValueError("native functional cache audit failed")
    positions = [position for row in rows for position in row["positions"]]
    top1 = statistics.fmean(position["top1_match"] for position in positions)
    mean_kl = statistics.fmean(
        position["kl_functional_to_mutable"] for position in positions
    )
    passed = top1 >= 1.0 and mean_kl <= 1e-6
    return {
        "status": "passed" if passed else "failed",
        "decision_semantic": "global_position_weighted_all_query_gate",
        "functional_execution": "native_functional_same_document_query_boundary",
        "mutable_execution": "standard_mutable_same_document_query_boundary",
        "world_size": expected_world_size,
        "samples": len(rows),
        "query_positions": len(positions),
        "position_top1_match_rate": top1,
        "mean_position_kl_functional_to_mutable": mean_kl,
        "max_position_kl_functional_to_mutable": max(
            position["kl_functional_to_mutable"] for position in positions
        ),
        "max_abs_logit_error": max(
            position["max_abs_logit_error"] for position in positions
        ),
        "thresholds": {"min_top1_match": 1.0, "max_mean_kl": 1e-6},
        "data_sha256": expected_data_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "test_v2_used": False,
        "single_token_autograd_claimed": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    args = parser.parse_args()
    shards = [
        json.loads(path.read_text())
        for path in sorted(args.run_dir.glob("native-semantic-rank-*.json"))
    ]
    try:
        result = aggregate_native_semantic(
            shards,
            expected_world_size=args.expected_world_size,
            expected_data_sha256=args.expected_data_sha256,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    atomic_json(args.run_dir / "native-semantic-gate.json", result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
