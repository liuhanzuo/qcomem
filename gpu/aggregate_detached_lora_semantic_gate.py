from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from run_downstream import atomic_json


def aggregate_detached_shards(
    shards: list[dict[str, Any]],
    *,
    expected_world_size: int,
    expected_data_sha256: str,
    expected_checkpoint_sha256: str,
    expected_query_positions: int,
) -> dict[str, Any]:
    if len(shards) != expected_world_size:
        raise ValueError(
            f"expected {expected_world_size} detached shards, found {len(shards)}"
        )
    ranks = {int(shard.get("rank", -1)) for shard in shards}
    if ranks != set(range(expected_world_size)):
        raise ValueError(f"detached shard ranks are incomplete: {sorted(ranks)}")
    if any(shard.get("status") != "completed_shard" for shard in shards):
        raise ValueError("detached shards are not all completed diagnostics")
    if {shard.get("world_size") for shard in shards} != {expected_world_size}:
        raise ValueError("detached shards disagree on world size")
    if {shard.get("data_sha256") for shard in shards} != {expected_data_sha256}:
        raise ValueError("detached semantic data SHA256 mismatch")
    if {shard.get("checkpoint_sha256") for shard in shards} != {
        expected_checkpoint_sha256
    }:
        raise ValueError("detached semantic checkpoint SHA256 mismatch")
    if any(shard.get("test_v2_used") is not False for shard in shards):
        raise ValueError("detached semantic gate must not consume test-v2")
    if {shard.get("comparison_scope") for shard in shards} != {
        "all_query_positions"
    }:
        raise ValueError("detached semantic gate did not cover all query positions")
    if {shard.get("training_suffix_execution") for shard in shards} != {
        "cached_document_prefill_detached_then_full_query_continuation"
    }:
        raise ValueError("detached training execution changed across shards")
    if {shard.get("deployment_suffix_execution") for shard in shards} != {
        "cached_document_prefill_then_full_query_continuation"
    }:
        raise ValueError("deployment execution changed across shards")

    thresholds = {
        (
            float(shard["thresholds"]["min_top1_match"]),
            float(shard["thresholds"]["max_mean_kl"]),
            float(shard["thresholds"]["max_logit_error"]),
        )
        for shard in shards
    }
    if len(thresholds) != 1:
        raise ValueError("detached shards used inconsistent thresholds")
    min_top1, max_mean_kl, max_logit_error = next(iter(thresholds))
    requested = {int(shard["global_samples_requested"]) for shard in shards}
    if len(requested) != 1:
        raise ValueError("detached shards disagree on requested sample count")
    rows = [row for shard in shards for row in shard["rows"]]
    if len(rows) != next(iter(requested)):
        raise ValueError("detached semantic sample count is incomplete")
    source_ids = [row["source_id"] for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("detached semantic source IDs are duplicated")
    if any(row["query_positions"] != expected_query_positions for row in rows):
        raise ValueError("detached gate did not retain every configured query position")
    if any(
        row["cache_immutability"].get("hard_gate_passed") is not True
        for row in rows
    ):
        raise ValueError("detached cache immutability failed on at least one sample")
    positions = [position for row in rows for position in row["positions"]]
    top1 = statistics.fmean(bool(row["top1_match"]) for row in positions)
    mean_kl = statistics.fmean(
        float(row["kl_detached_to_deployment"]) for row in positions
    )
    max_kl = max(float(row["kl_detached_to_deployment"]) for row in positions)
    max_error = max(float(row["max_abs_logit_error"]) for row in positions)
    cache_gate = all(
        bool(row["cache_immutability"].get("hard_gate_passed")) for row in rows
    )
    passed = (
        top1 >= min_top1
        and mean_kl <= max_mean_kl
        and max_error <= max_logit_error
        and cache_gate
    )
    return {
        "status": "passed" if passed else "failed",
        "claim": "query-continuation-only detached capability; not full two-stage gradient training",
        "decision_semantic": "global position-weighted hard gate",
        "training_suffix_execution": (
            "cached_document_prefill_detached_then_full_query_continuation"
        ),
        "deployment_suffix_execution": (
            "cached_document_prefill_then_full_query_continuation"
        ),
        "comparison_scope": "all_query_positions",
        "world_size": expected_world_size,
        "samples": len(rows),
        "query_positions": len(positions),
        "position_top1_match_rate": top1,
        "mean_position_kl_detached_to_deployment": mean_kl,
        "max_position_kl_detached_to_deployment": max_kl,
        "max_abs_logit_error": max_error,
        "cache_immutability_gate_passed": cache_gate,
        "thresholds": {
            "min_top1_match": min_top1,
            "max_mean_kl": max_mean_kl,
            "max_logit_error": max_logit_error,
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
        description="Aggregate detached-document-cache LoRA semantic shards"
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-query-positions", type=int, default=128)
    args = parser.parse_args()
    paths = sorted(args.run_dir.glob("detached-semantic-shard-*.json"))
    shards = [json.loads(path.read_text()) for path in paths]
    try:
        result = aggregate_detached_shards(
            shards,
            expected_world_size=args.expected_world_size,
            expected_data_sha256=args.expected_data_sha256,
            expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            expected_query_positions=args.expected_query_positions,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    atomic_json(args.run_dir / "detached-semantic-gate.json", result)
    print(json.dumps(result, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
