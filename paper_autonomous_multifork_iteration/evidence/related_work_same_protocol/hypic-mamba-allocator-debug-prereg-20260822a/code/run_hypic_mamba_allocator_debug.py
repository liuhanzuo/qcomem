#!/usr/bin/env python3
"""Single-cell read-only diagnostic for the HYPIC Mamba free-list representation."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from run_hypic_retained_state_bytes import (
    DATA_SHA256,
    HYPIC_COMMIT,
    _post_flush,
    _require,
    _target,
    _wait_for_json,
    atomic_json,
    cached_tokens_from_completion,
    load_segmented_workload,
    request_prompts,
    sha256_file,
    stream_completion,
)


SCHEMA = "hypic-rwd5-mamba-allocator-debug-v1"
MODE = "transition_rope_recompute"


def validate(row: dict, *, workload_id: str, target_sha256: str) -> dict:
    _require(row.get("schema") == SCHEMA, "allocator debug schema")
    _require(
        row.get("status") == "read_only_post_prime_allocator_captured",
        "allocator debug status",
    )
    _require(row.get("official_commit") == HYPIC_COMMIT, "allocator debug commit")
    _require(
        row.get("paper_evidence") is False
        and row.get("formal_receipt_emitted") is False
        and row.get("mutation_performed") is False,
        "allocator debug nonformal read-only boundary",
    )
    target = row["target"]
    _require(
        target["mode"] == MODE
        and int(target["rank"]) == 0
        and target["workload_id"] == workload_id
        and target["target_file_sha256"] == target_sha256,
        "allocator debug target binding",
    )
    cache = row["cache"]
    _require(
        cache["class"] == "PICache"
        and cache["target_slots_distinct"] is True
        and len(cache["target_mamba_state_slots"]) == 2,
        "allocator debug target cache topology",
    )
    target_entries = [entry for entry in cache["entries"] if entry["is_target_segment"]]
    _require(len(target_entries) == 2, "allocator debug exact target entries")
    _require(
        sorted(entry["mamba_state_slot"] for entry in target_entries)
        == sorted(cache["target_mamba_state_slots"]),
        "allocator debug target entry slots",
    )
    allocator = row["allocator"]
    _require(
        allocator["class"] == "MambaSlotAllocator"
        and allocator["tree_cache_alias"] is True
        and allocator["req_pool_alias"] is True,
        "allocator debug live alias identity",
    )
    _require(
        allocator["dict_keys"] == ["_alloc_iter", "device", "free_slots", "size"],
        "allocator debug exact fields",
    )
    _require(
        allocator["release_representations"]
        == {
            "release_pages": {
                "attribute_present": False,
                "raw_values": None,
                "value_type": None,
            },
            "release_slots": {
                "attribute_present": False,
                "raw_values": None,
                "value_type": None,
            },
        },
        "allocator debug exact absence of release representation",
    )
    raw = [int(value) for value in allocator["raw_free_slots"]]
    size = int(allocator["size"])
    _require(size > 0, "allocator debug capacity")
    _require(
        allocator["raw_count"] == len(raw)
        and allocator["available_size"] == len(raw),
        "allocator debug raw available count",
    )
    counts = Counter(raw)
    positions: dict[int, list[int]] = {}
    for index, slot in enumerate(raw):
        positions.setdefault(slot, []).append(index)
    duplicates = [
        {"slot": slot, "count": counts[slot], "positions": positions[slot]}
        for slot in sorted(counts)
        if counts[slot] > 1
    ]
    expected = set(range(1, size + 1))
    _require(allocator["duplicates"] == duplicates, "allocator debug duplicate derivation")
    _require(
        allocator["unique_count"] == len(counts)
        and allocator["duplicate_excess_count"]
        == sum(item["count"] - 1 for item in duplicates),
        "allocator debug duplicate counts",
    )
    _require(
        allocator["out_of_domain"] == sorted(set(raw) - expected)
        and allocator["missing_from_raw_unique_domain"]
        == sorted(expected - set(raw)),
        "allocator debug domain derivation",
    )
    tensor = allocator["free_slots_tensor"]
    _require(
        tensor["tensor_name"] == "mamba_allocator.free_slots"
        and tensor["dtype"] == "torch.int64"
        and tensor["shape"] == [len(raw)]
        and tensor["stride"] == [1]
        and tensor["element_size"] == 8
        and tensor["storage_offset_elements"] == 0
        and tensor["range_bytes"] == len(raw) * 8,
        "allocator debug raw tensor metadata",
    )
    _require(allocator["duplicate_excess_count"] > 0, "allocator debug reproduces P duplicate")
    return {
        "schema": "hypic-rwd5-mamba-allocator-debug-validation-v1",
        "status": "passed_exact_duplicate_representation_capture",
        "paper_evidence": False,
        "official_commit": HYPIC_COMMIT,
        "workload_id": workload_id,
        "allocator_size": size,
        "raw_count": len(raw),
        "unique_count": len(counts),
        "duplicate_excess_count": allocator["duplicate_excess_count"],
        "duplicates": duplicates,
        "target_mamba_state_slots": cache["target_mamba_state_slots"],
        "target_slots_present_in_raw_free_slots": cache[
            "target_slots_present_in_raw_free_slots"
        ],
    }


def run(args: argparse.Namespace) -> None:
    _require(args.mode == MODE and args.rank == 0, "one affected HYPIC debug cell")
    _require(sha256_file(args.data) == DATA_SHA256, "debug data digest drift")
    for path in (args.target_file, args.allocator_debug_receipt, args.output):
        _require(not path.exists(), f"fresh debug output: {path}")
    workload = load_segmented_workload(args.model, args.data, 0)
    prompts = request_prompts(workload, MODE)
    common = {
        "model": args.served_model_name,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    # Reproduce P exactly through the failure boundary: disjoint warm prime,
    # warm cache hit, then target publication and formal prime.
    warm_prime = stream_completion(
        args.base_url.rstrip("/") + "/v1/completions",
        {**common, "prompt": prompts["warm_prime"], "max_tokens": 1},
        timeout=args.timeout,
        require_text=False,
    )
    warmup = stream_completion(
        args.base_url.rstrip("/") + "/v1/completions",
        {**common, "prompt": prompts["warm_measured"], "max_tokens": 1},
        timeout=args.timeout,
        require_text=False,
    )
    warm_cached = cached_tokens_from_completion(warmup)
    _require(isinstance(warm_cached, int) and warm_cached > 0, "debug warm cache path")
    target = _target(workload, MODE, 0, {})
    target["snapshot_id"] = "allocator-debug-transition_rope_recompute-rank-0"
    atomic_json(args.target_file, target)
    target_sha = sha256_file(args.target_file)
    prime = stream_completion(
        args.base_url.rstrip("/") + "/v1/completions",
        {
            **common,
            "prompt": prompts["formal_prime"],
            "max_tokens": 1,
        },
        timeout=args.timeout,
        require_text=False,
    )
    row = _wait_for_json(args.allocator_debug_receipt, timeout=90.0)
    validation = validate(
        row, workload_id=workload["workload_id"], target_sha256=target_sha
    )
    flush = _post_flush(args.base_url, args.timeout)
    atomic_json(
        args.output,
        {
            "schema": "hypic-rwd5-mamba-allocator-debug-run-v1",
            "status": "completed_debug_only_not_formal_evidence",
            "official_commit": HYPIC_COMMIT,
            "mode": MODE,
            "rank": 0,
            "workload_id": workload["workload_id"],
            "target_sha256": target_sha,
            "allocator_debug_receipt_sha256": sha256_file(
                args.allocator_debug_receipt
            ),
            "validation": validation,
            "warm_prime": warm_prime,
            "warmup": warmup,
            "prime": prime,
            "flush_response": flush,
            "formal_receipts_emitted": 0,
            "paper_evidence": False,
        },
    )


def validate_file(args: argparse.Namespace) -> None:
    row = json.loads(args.allocator_debug_receipt.read_text())
    validation = validate(
        row, workload_id=args.workload_id, target_sha256=args.target_sha256
    )
    validation["allocator_debug_receipt_sha256"] = sha256_file(
        args.allocator_debug_receipt
    )
    atomic_json(args.output, validation)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("run", "validate"), required=True)
    parser.add_argument("--mode", default=MODE)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:33600")
    parser.add_argument("--served-model-name", default="qwen35-hypic")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--target-file", type=Path)
    parser.add_argument("--allocator-debug-receipt", type=Path, required=True)
    parser.add_argument("--workload-id")
    parser.add_argument("--target-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "run":
        for name in ("model", "data", "target_file"):
            _require(getattr(args, name) is not None, f"debug run {name}")
        run(args)
    else:
        _require(args.workload_id and args.target_sha256, "debug validation binding")
        validate_file(args)


if __name__ == "__main__":
    main()
