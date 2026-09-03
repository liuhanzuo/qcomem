#!/usr/bin/env python3
"""Single-cell read-only diagnostic for the HYPIC Mamba free-list representation."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
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
WORKLOAD_ID = "qasper-6"
SNAPSHOT_ID = "allocator-debug-transition_rope_recompute-rank-0"


def _token_bytes(token_ids: list[int]) -> bytes:
    payload = bytearray()
    for token_id in token_ids:
        token_id = int(token_id)
        _require(0 <= token_id < 2**31, "debug token id range")
        payload.extend(struct.pack("<i", token_id))
    return bytes(payload)


def token_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(_token_bytes(token_ids)).hexdigest()


def segment_hash_hex(token_ids: list[int]) -> str:
    return hashlib.sha256(_token_bytes(token_ids)).digest()[:16].hex()


def _exact_keys(row: dict, expected: set[str], label: str) -> None:
    _require(isinstance(row, dict) and set(row) == expected, f"{label} exact keys")


def _validate_target(target: dict, target_sha256: str) -> dict:
    _exact_keys(
        target,
        {
            "authority",
            "document_token_ids",
            "document_token_sha256",
            "mode",
            "official_commit",
            "rank",
            "schema",
            "seam_tokens",
            "segment_token_ids",
            "snapshot_id",
            "workload_binding",
            "workload_id",
        },
        "target",
    )
    _require(
        target["schema"] == "forkaudit-hypic-retained-state-target-v2"
        and target["mode"] == MODE
        and int(target["rank"]) == 0
        and target["snapshot_id"] == SNAPSHOT_ID
        and target["workload_id"] == WORKLOAD_ID
        and target["official_commit"] == HYPIC_COMMIT
        and int(target["seam_tokens"]) == 8,
        "target exact literal cell",
    )
    document = [int(value) for value in target["document_token_ids"]]
    segments = [[int(value) for value in segment] for segment in target["segment_token_ids"]]
    _require(len(segments) == 2 and segments[0] + segments[1] == document, "target segment/document binding")
    document_sha = token_sha256(document)
    _require(target["document_token_sha256"] == document_sha, "target document token hash")
    binding = target["workload_binding"]
    _exact_keys(
        binding,
        {
            "dataset",
            "document_token_sha256",
            "document_tokens",
            "prompt_token_sha256",
            "query_tokens",
            "segment_offsets",
            "source_index",
            "token_identity_verified",
            "workload_id",
        },
        "target workload binding",
    )
    query_tokens = int(binding["query_tokens"])
    expected_offsets = [
        [0, len(segments[0])],
        [len(segments[0]), len(document)],
        [len(document), len(document) + query_tokens],
    ]
    prompt_sha = binding["prompt_token_sha256"]
    _require(
        binding["dataset"] == "qasper"
        and int(binding["source_index"]) == 6
        and binding["workload_id"] == WORKLOAD_ID
        and binding["token_identity_verified"] is True
        and int(binding["document_tokens"]) == len(document)
        and binding["document_token_sha256"] == document_sha
        and query_tokens == 99
        and binding["segment_offsets"] == expected_offsets
        and isinstance(prompt_sha, str)
        and len(prompt_sha) == 64
        and all(char in "0123456789abcdef" for char in prompt_sha),
        "target exact qasper-6 workload binding",
    )
    return {
        "target_sha256": target_sha256,
        "document": document,
        "document_sha256": document_sha,
        "segments": segments,
        "segment_token_sha256": [token_sha256(segment) for segment in segments],
        "segment_hash_hex": [segment_hash_hex(segment) for segment in segments],
    }


def validate(
    row: dict,
    *,
    target: dict,
    target_sha256: str,
    expected_allocator_device: str = "cuda",
    expected_tensor_device: str = "cuda:0",
    expected_allocator_size: int = 183,
) -> dict:
    expected_target = _validate_target(target, target_sha256)
    _exact_keys(
        row,
        {
            "schema",
            "status",
            "official_commit",
            "paper_evidence",
            "formal_receipt_emitted",
            "mutation_performed",
            "target",
            "cache",
            "allocator",
        },
        "allocator debug receipt",
    )
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
    target_row = row["target"]
    _exact_keys(
        target_row,
        {
            "mode",
            "rank",
            "snapshot_id",
            "workload_id",
            "target_file_sha256",
            "document_token_sha256",
            "document_tokens",
            "segment_token_counts",
            "segment_token_sha256",
            "segment_hash_hex",
        },
        "allocator debug target receipt",
    )
    _require(
        target_row["mode"] == MODE
        and int(target_row["rank"]) == 0
        and target_row["snapshot_id"] == SNAPSHOT_ID
        and target_row["workload_id"] == WORKLOAD_ID
        and target_row["target_file_sha256"] == target_sha256
        and target_row["document_token_sha256"] == expected_target["document_sha256"]
        and int(target_row["document_tokens"]) == len(expected_target["document"])
        and target_row["segment_token_counts"]
        == [len(segment) for segment in expected_target["segments"]]
        and target_row["segment_token_sha256"]
        == expected_target["segment_token_sha256"]
        and target_row["segment_hash_hex"] == expected_target["segment_hash_hex"],
        "allocator debug target binding",
    )
    allocator = row["allocator"]
    _exact_keys(
        allocator,
        {
            "class",
            "module",
            "tree_cache_alias",
            "req_pool_alias",
            "size",
            "device",
            "available_size",
            "alloc_iter_is_none",
            "dict_keys",
            "fields",
            "release_representations",
            "free_slots_tensor",
            "raw_free_slots",
            "raw_count",
            "unique_count",
            "duplicates",
            "duplicate_excess_count",
            "out_of_domain",
            "missing_from_raw_unique_domain",
        },
        "allocator receipt",
    )
    _require(
        allocator["class"] == "MambaSlotAllocator"
        and allocator["module"] == "sglang.srt.mem_cache.allocator.mamba"
        and allocator["tree_cache_alias"] is True
        and allocator["req_pool_alias"] is True,
        "allocator debug live alias identity",
    )
    _require(
        isinstance(allocator["raw_free_slots"], list)
        and all(type(value) is int for value in allocator["raw_free_slots"]),
        "allocator raw slot types",
    )
    raw = [int(value) for value in allocator["raw_free_slots"]]
    size = int(allocator["size"])
    _require(size == expected_allocator_size, "allocator debug exact capacity")
    _require(
        allocator["device"] == expected_allocator_device
        and allocator["available_size"] == len(raw)
        and allocator["alloc_iter_is_none"] is True,
        "allocator scalar fields",
    )
    _require(
        allocator["dict_keys"] == ["_alloc_iter", "device", "free_slots", "size"],
        "allocator debug exact fields",
    )
    expected_fields = {
        "_alloc_iter": {"kind": "NoneType", "value": None},
        "device": {"kind": "str", "value": expected_allocator_device},
        "free_slots": {
            "kind": "tensor",
            "dtype": "torch.int64",
            "shape": [len(raw)],
            "device": expected_tensor_device,
        },
        "size": {"kind": "int", "value": size},
    }
    _require(allocator["fields"] == expected_fields, "allocator dict field replay")
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
    cache = row["cache"]
    _exact_keys(
        cache,
        {
            "class",
            "module",
            "entry_count",
            "entries",
            "target_mamba_state_slots",
            "target_slots_distinct",
            "target_slots_present_in_raw_free_slots",
        },
        "cache receipt",
    )
    _require(
        cache["class"] == "PICache"
        and cache["module"] == "sglang.srt.pic.picache"
        and cache["target_slots_distinct"] is True
        and len(cache["target_mamba_state_slots"]) == 2,
        "allocator debug target cache topology",
    )
    _require(
        all(type(value) is int for value in cache["target_mamba_state_slots"])
        and len(set(cache["target_mamba_state_slots"])) == 2,
        "allocator debug target slot distinctness replay",
    )
    _require(
        isinstance(cache["entries"], list)
        and cache["entry_count"] == len(cache["entries"])
        and cache["entry_count"] > 0,
        "cache exact entry count",
    )
    expected_hashes = expected_target["segment_hash_hex"]
    entries_by_hash: dict[str, dict] = {}
    for entry in cache["entries"]:
        _exact_keys(
            entry,
            {
                "segment_hash_hex",
                "is_target_segment",
                "token_ids",
                "token_count",
                "token_sha256",
                "mamba_state_slot",
                "lock_ref",
            },
            "cache entry",
        )
        token_ids = [int(value) for value in entry["token_ids"]]
        segment_hash = segment_hash_hex(token_ids)
        _require(
            entry["segment_hash_hex"] == segment_hash
            and entry["token_sha256"] == token_sha256(token_ids)
            and int(entry["token_count"]) == len(token_ids)
            and entry["is_target_segment"] is (segment_hash in expected_hashes)
            and isinstance(entry["mamba_state_slot"], int)
            and 1 <= int(entry["mamba_state_slot"]) <= size
            and isinstance(entry["lock_ref"], int)
            and int(entry["lock_ref"]) >= 0,
            "cache entry content/hash/count/lock replay",
        )
        _require(segment_hash not in entries_by_hash, "cache unique segment hash")
        entries_by_hash[segment_hash] = entry
    _require(
        [entry["segment_hash_hex"] for entry in cache["entries"]]
        == sorted(entries_by_hash),
        "cache canonical entry order",
    )
    _require(all(value in entries_by_hash for value in expected_hashes), "cache exact target hashes")
    target_entries = [entries_by_hash[value] for value in expected_hashes]
    _require(len(target_entries) == 2, "allocator debug exact target entries")
    _require(
        [entry["token_ids"] for entry in target_entries] == expected_target["segments"]
        and all(entry["lock_ref"] == 0 for entry in target_entries)
        and [entry["mamba_state_slot"] for entry in target_entries]
        == cache["target_mamba_state_slots"],
        "allocator debug target entry slots",
    )
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
    _exact_keys(
        tensor,
        {
            "device",
            "storage_data_ptr",
            "storage_nbytes",
            "storage_id",
            "dtype",
            "shape",
            "stride",
            "element_size",
            "storage_offset_elements",
            "tensor_data_ptr",
            "tensor_name",
            "component",
            "byte_start",
            "byte_end",
            "absolute_byte_start",
            "absolute_byte_end",
            "range_bytes",
            "selection",
        },
        "free slots tensor",
    )
    storage_ptr = int(tensor["storage_data_ptr"])
    storage_nbytes = int(tensor["storage_nbytes"])
    storage_offset = int(tensor["storage_offset_elements"])
    byte_start = storage_offset * 8
    byte_end = byte_start + len(raw) * 8
    _require(
        tensor["tensor_name"] == "mamba_allocator.free_slots"
        and tensor["component"] == "cache_index_metadata"
        and tensor["selection"] == {"kind": "whole_tensor"}
        and tensor["device"] == expected_tensor_device
        and tensor["dtype"] == "torch.int64"
        and tensor["shape"] == [len(raw)]
        and tensor["stride"] == [1]
        and tensor["element_size"] == 8
        and storage_ptr > 0
        and storage_nbytes >= byte_end > byte_start >= 0
        and tensor["storage_id"]
        == hashlib.sha256(
            f"{expected_tensor_device}:{storage_ptr}:{storage_nbytes}".encode()
        ).hexdigest()
        and int(tensor["tensor_data_ptr"]) == storage_ptr + byte_start
        and int(tensor["byte_start"]) == byte_start
        and int(tensor["byte_end"]) == byte_end
        and int(tensor["absolute_byte_start"]) == storage_ptr + byte_start
        and int(tensor["absolute_byte_end"]) == storage_ptr + byte_end
        and tensor["range_bytes"] == len(raw) * 8,
        "allocator debug raw tensor metadata",
    )
    target_slots = [int(value) for value in cache["target_mamba_state_slots"]]
    expected_membership = {
        str(slot): [index for index, raw_slot in enumerate(raw) if raw_slot == slot]
        for slot in target_slots
    }
    _require(
        cache["target_slots_present_in_raw_free_slots"] == expected_membership,
        "target slot/free-list membership replay",
    )
    _require(allocator["duplicate_excess_count"] > 0, "allocator debug reproduces P duplicate")
    return {
        "schema": "hypic-rwd5-mamba-allocator-debug-validation-v1",
        "status": "passed_exact_duplicate_representation_capture",
        "paper_evidence": False,
        "official_commit": HYPIC_COMMIT,
        "workload_id": WORKLOAD_ID,
        "allocator_size": size,
        "raw_count": len(raw),
        "unique_count": len(counts),
        "duplicate_excess_count": allocator["duplicate_excess_count"],
        "duplicates": duplicates,
        "target_mamba_state_slots": cache["target_mamba_state_slots"],
        "target_slots_present_in_raw_free_slots": expected_membership,
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
        row, target=target, target_sha256=target_sha
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
    target = json.loads(args.target_file.read_text())
    _require(sha256_file(args.target_file) == args.target_sha256, "debug target file hash")
    validation = validate(
        row, target=target, target_sha256=args.target_sha256
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
    parser.add_argument("--target-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "run":
        for name in ("model", "data", "target_file"):
            _require(getattr(args, name) is not None, f"debug run {name}")
        run(args)
    else:
        _require(args.target_file and args.target_sha256, "debug validation binding")
        validate_file(args)


if __name__ == "__main__":
    main()
