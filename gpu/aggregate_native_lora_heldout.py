from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from run_downstream import atomic_json


def aggregate_heldout(
    shards: list[dict[str, Any]],
    *,
    expected_world_size: int,
    expected_examples: int,
    expected_data_sha256: str,
) -> dict[str, Any]:
    if len(shards) != expected_world_size or {
        shard.get("rank") for shard in shards
    } != set(range(expected_world_size)):
        raise ValueError("heldout shards do not cover every rank")
    if any(shard.get("status") != "completed_shard" for shard in shards):
        raise ValueError("at least one heldout shard is incomplete")
    if {shard.get("data_sha256") for shard in shards} != {expected_data_sha256}:
        raise ValueError("heldout view SHA256 drifted across shards")
    if any(shard.get("test_v2_used") is not False for shard in shards):
        raise ValueError("heldout selection must not read test-v2")
    summaries = []
    for step in (0, 64, 128):
        checkpoint_groups = [
            checkpoint
            for shard in shards
            for checkpoint in shard["checkpoints"]
            if checkpoint["training_step"] == step
        ]
        if len(checkpoint_groups) != expected_world_size:
            raise ValueError(f"checkpoint-{step} heldout shards are incomplete")
        hashes = {group["checkpoint_sha256"] for group in checkpoint_groups}
        paths = {group["checkpoint"] for group in checkpoint_groups}
        if len(hashes) != 1 or len(paths) != 1:
            raise ValueError(f"checkpoint-{step} identity drifted across ranks")
        rows = [row for group in checkpoint_groups for row in group["rows"]]
        ids = [row["example_id"] for row in rows]
        if len(rows) != expected_examples or len(set(ids)) != expected_examples:
            raise ValueError(f"checkpoint-{step} did not evaluate unique heldout examples")
        if any(row["cache_audit"].get("hard_gate_passed") is not True for row in rows):
            raise ValueError(f"checkpoint-{step} native cache audit failed")
        summaries.append(
            {
                "training_step": step,
                "checkpoint": next(iter(paths)),
                "checkpoint_sha256": next(iter(hashes)),
                "examples": len(rows),
                "mean_example_loss": statistics.fmean(row["loss"] for row in rows),
                "mean_example_forward_kl": statistics.fmean(
                    row["forward_kl"] for row in rows
                ),
                "mean_example_reverse_kl": statistics.fmean(
                    row["reverse_kl"] for row in rows
                ),
                "rows": rows,
            }
        )
    best = min(summaries, key=lambda row: (row["mean_example_loss"], row["training_step"]))
    baseline = summaries[0]
    return {
        "status": "passed",
        "selection_metric": "example_equal_mean_topk_bidirectional_kl",
        "tie_break": "earliest_training_step",
        "world_size": expected_world_size,
        "heldout_examples": expected_examples,
        "data_sha256": expected_data_sha256,
        "summaries": summaries,
        "selected": {
            **{key: best[key] for key in (
                "training_step",
                "checkpoint",
                "checkpoint_sha256",
                "mean_example_loss",
            )},
            "mean_loss_delta_vs_step0": best["mean_example_loss"]
            - baseline["mean_example_loss"],
        },
        "test_v2_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--expected-examples", type=int, default=26)
    parser.add_argument("--expected-data-sha256", required=True)
    args = parser.parse_args()
    shards = [
        json.loads(path.read_text())
        for path in sorted(args.run_dir.glob("heldout-rank-*.json"))
    ]
    try:
        result = aggregate_heldout(
            shards,
            expected_world_size=args.expected_world_size,
            expected_examples=args.expected_examples,
            expected_data_sha256=args.expected_data_sha256,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    atomic_json(args.run_dir / "heldout-selection.json", result)
    (args.run_dir / "best-checkpoint.path").write_text(
        result["selected"]["checkpoint"] + "\n"
    )
    (args.run_dir / "best-checkpoint.sha256").write_text(
        result["selected"]["checkpoint_sha256"] + "\n"
    )
    print(json.dumps(result["selected"], ensure_ascii=False))


if __name__ == "__main__":
    main()
