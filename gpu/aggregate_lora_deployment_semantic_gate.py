from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from run_downstream import atomic_json


def aggregate_semantic_shards(
    shards: list[dict[str, Any]],
    *,
    expected_world_size: int,
    expected_data_sha256: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    if len(shards) != expected_world_size:
        raise ValueError(
            f"expected {expected_world_size} semantic shards, found {len(shards)}"
        )
    ranks = {int(shard.get("rank", -1)) for shard in shards}
    if ranks != set(range(expected_world_size)):
        raise ValueError(f"semantic shard ranks are incomplete: {sorted(ranks)}")
    if any(shard.get("status") != "completed_shard" for shard in shards):
        raise ValueError("semantic shards are not all completed diagnostics")
    if {shard.get("world_size") for shard in shards} != {expected_world_size}:
        raise ValueError("semantic shards disagree on world size")
    if {shard.get("data_sha256") for shard in shards} != {
        expected_data_sha256
    }:
        raise ValueError("semantic-gate data SHA256 mismatch")
    if {shard.get("checkpoint_sha256") for shard in shards} != {
        expected_checkpoint_sha256
    }:
        raise ValueError("semantic-gate checkpoint SHA256 mismatch")
    if any(shard.get("test_v2_used") is not False for shard in shards):
        raise ValueError("semantic gate must not consume test-v2")
    if {shard.get("comparison_scope") for shard in shards} != {
        "all_query_positions"
    }:
        raise ValueError("semantic gate did not cover all query positions")
    if {shard.get("training_suffix_execution") for shard in shards} != {
        "uncached_full_document_plus_query_sequence"
    }:
        raise ValueError("training suffix semantic changed across shards")
    if {shard.get("deployment_suffix_execution") for shard in shards} != {
        "cached_document_prefill_then_full_query_continuation"
    }:
        raise ValueError("deployment suffix semantic changed across shards")

    thresholds = {
        (
            float(shard["thresholds"]["min_top1_match"]),
            float(shard["thresholds"]["max_mean_kl"]),
        )
        for shard in shards
    }
    if len(thresholds) != 1:
        raise ValueError(f"semantic shards used inconsistent thresholds: {thresholds}")
    min_top1, max_mean_kl = next(iter(thresholds))
    requested = {int(shard["global_samples_requested"]) for shard in shards}
    if len(requested) != 1:
        raise ValueError("semantic shards disagree on requested sample count")
    requested_samples = next(iter(requested))
    rows = [row for shard in shards for row in shard["rows"]]
    if len(rows) != requested_samples:
        raise ValueError(
            f"expected {requested_samples} semantic samples, found {len(rows)}"
        )
    source_ids = [row["source_id"] for row in rows]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("semantic sample source IDs are duplicated")
    positions = [position for row in rows for position in row["positions"]]
    if sum(int(row["query_positions"]) for row in rows) != len(positions):
        raise ValueError("query-position ledger is inconsistent")
    top1 = statistics.fmean(bool(position["top1_match"]) for position in positions)
    mean_kl = statistics.fmean(
        float(position["kl_training_to_deployment"]) for position in positions
    )
    passed = top1 >= min_top1 and mean_kl <= max_mean_kl
    return {
        "status": "passed" if passed else "failed",
        "claim": "training and deployment suffix execution are not assumed equivalent",
        "decision_semantic": (
            "global position-weighted gate; local shard failures are retained and "
            "thresholds are not relaxed"
        ),
        "training_suffix_execution": "uncached_full_document_plus_query_sequence",
        "deployment_suffix_execution": (
            "cached_document_prefill_then_full_query_continuation"
        ),
        "comparison_scope": "all_query_positions",
        "world_size": expected_world_size,
        "samples": len(rows),
        "query_positions": len(positions),
        "position_top1_match_rate": top1,
        "mean_position_kl_training_to_deployment": mean_kl,
        "max_position_kl_training_to_deployment": max(
            float(position["kl_training_to_deployment"])
            for position in positions
        ),
        "max_abs_logit_error": max(
            float(position["max_abs_logit_error"]) for position in positions
        ),
        "thresholds": {
            "min_top1_match": min_top1,
            "max_mean_kl": max_mean_kl,
        },
        "data_sha256": expected_data_sha256,
        "checkpoint_sha256": expected_checkpoint_sha256,
        "test_v2_used": False,
        "local_threshold_passed_by_rank": {
            str(shard["rank"]): bool(shard["local_threshold_passed"])
            for shard in sorted(shards, key=lambda item: int(item["rank"]))
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate distributed quant-LoRA deployment-semantic shards"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob("deployment-semantic-shard-*.json"))
    shards = [json.loads(path.read_text()) for path in paths]
    try:
        result = aggregate_semantic_shards(
            shards,
            expected_world_size=args.expected_world_size,
            expected_data_sha256=args.expected_data_sha256,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    destination = args.run_dir / "deployment-semantic-gate.json"
    atomic_json(destination, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
