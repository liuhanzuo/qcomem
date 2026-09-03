#!/usr/bin/env python3
"""Read-only, object-owned byte receipts for RW-D5.

This module is copied into an instrumented *temporary* HYPIC worktree.  It
never reads NVML or process/pool allocation deltas.  Instead it resolves the
exact cache entries that own a preregistered document, maps their KV and
recurrent-state slot indices to byte ranges in the live backing tensors, and
deduplicates overlapping ranges by storage identity.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from array import array
from pathlib import Path
from typing import Any, Iterable

import torch


SCHEMA = "forkaudit-hypic-retained-state-receipt-v1"
TARGET_SCHEMA = "forkaudit-hypic-retained-state-target-v1"
OFFICIAL_COMMIT = "98147c01909004e66d98bcb18b886927d41b0ee5"
TARGET_ENV = "FORKAUDIT_RWD5_TARGET_PATH"
OUTPUT_ENV = "FORKAUDIT_RWD5_RECEIPT_DIR"


class ReceiptError(RuntimeError):
    pass


class TargetNotReady(ReceiptError):
    """The preregistered target is not in the cache yet."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_canonical_bytes(value))
    os.replace(temporary, path)


def _token_bytes(token_ids: Iterable[int]) -> bytes:
    payload = bytearray()
    for token_id in token_ids:
        token_id = int(token_id)
        _require(0 <= token_id < 2**31, "token id outside signed int32")
        payload.extend(struct.pack("<i", token_id))
    return bytes(payload)


def token_sha256(token_ids: Iterable[int]) -> str:
    return hashlib.sha256(_token_bytes(token_ids)).hexdigest()


def segment_hash_hex(token_ids: Iterable[int]) -> str:
    return hashlib.sha256(_token_bytes(token_ids)).digest()[:16].hex()


def _load_target() -> tuple[dict[str, Any], Path] | None:
    target_path = os.environ.get(TARGET_ENV)
    output_dir = os.environ.get(OUTPUT_ENV)
    if not target_path or not output_dir:
        return None
    path = Path(target_path)
    if not path.is_file():
        return None
    target = json.loads(path.read_text())
    _require(target.get("schema") == TARGET_SCHEMA, "target schema")
    snapshot_id = target.get("snapshot_id")
    _require(
        isinstance(snapshot_id, str)
        and snapshot_id
        and all(ch.isalnum() or ch in "-_" for ch in snapshot_id),
        "snapshot id",
    )
    _require(target.get("official_commit") == OFFICIAL_COMMIT, "target commit")
    _require(target.get("mode") in {"prefix_cache", "transition_rope_recompute"}, "mode")
    document = target.get("document_token_ids")
    _require(isinstance(document, list) and document, "document token ids")
    _require(token_sha256(document) == target.get("document_token_sha256"), "document hash")
    return target, Path(output_dir)


def _tensor_storage(tensor: torch.Tensor) -> dict[str, Any]:
    _require(isinstance(tensor, torch.Tensor), "tensor expected")
    storage = tensor.untyped_storage()
    base = int(storage.data_ptr())
    nbytes = int(storage.nbytes())
    device = str(tensor.device)
    return {
        "device": device,
        "storage_data_ptr": base,
        "storage_nbytes": nbytes,
        "storage_id": hashlib.sha256(f"{device}:{base}:{nbytes}".encode()).hexdigest(),
        "dtype": str(tensor.dtype),
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
        "element_size": int(tensor.element_size()),
        "storage_offset_elements": int(tensor.storage_offset()),
        "tensor_data_ptr": int(tensor.data_ptr()),
    }


def _record_range(
    tensor: torch.Tensor,
    *,
    tensor_name: str,
    component: str,
    start: int,
    end: int,
    selection: dict[str, Any],
) -> dict[str, Any]:
    info = _tensor_storage(tensor)
    _require(0 <= start < end <= info["storage_nbytes"], f"range outside {tensor_name}")
    return {
        **info,
        "tensor_name": tensor_name,
        "component": component,
        "byte_start": int(start),
        "byte_end": int(end),
        "absolute_byte_start": int(info["storage_data_ptr"] + start),
        "absolute_byte_end": int(info["storage_data_ptr"] + end),
        "range_bytes": int(end - start),
        "selection": selection,
    }


def _contiguous_runs(indices: list[int]) -> list[tuple[int, int]]:
    unique = sorted(set(int(value) for value in indices))
    _require(len(unique) == len(indices), "duplicate owned slot index")
    if not unique:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = unique[0]
    for value in unique[1:]:
        if value != previous + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def _kv_payload_ranges(tree_cache, slots: list[int]) -> list[dict[str, Any]]:
    allocator = tree_cache.token_to_kv_pool_allocator
    pool = allocator.get_kvcache()
    full = getattr(pool, "full_kv_pool", pool)
    _require(getattr(full, "kv_cache_layout", "nhd") == "nhd", "unsupported KV layout")
    buffers = [("key", value) for value in full.k_buffer] + [
        ("value", value) for value in full.v_buffer
    ]
    records: list[dict[str, Any]] = []
    for kind, tensors in (("key", full.k_buffer), ("value", full.v_buffer)):
        for layer, tensor in enumerate(tensors):
            _require(tensor.ndim == 3 and tensor.is_contiguous(), "NHD KV tensor layout")
            element = int(tensor.element_size())
            row_elements = int(tensor.shape[1] * tensor.shape[2])
            _require(int(tensor.stride(0)) == row_elements, "KV row stride")
            base = int(tensor.storage_offset()) * element
            for first, after in _contiguous_runs(slots):
                _require(first > 0 and after <= int(tensor.shape[0]), "KV slot bounds")
                start = base + first * row_elements * element
                end = base + after * row_elements * element
                records.append(
                    _record_range(
                        tensor,
                        tensor_name=f"full_kv.{kind}[{layer}]",
                        component=f"full_attention_{kind}",
                        start=start,
                        end=end,
                        selection={"slot_start": first, "slot_end_exclusive": after},
                    )
                )
    _require(bool(buffers) and records, "empty KV payload ledger")
    return records


def _mamba_payload_ranges(tree_cache, slots: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache = tree_cache.req_to_token_pool.mamba_pool.mamba_cache
    fields: list[tuple[str, torch.Tensor | None]] = []
    for index, tensor in enumerate(cache.conv):
        fields.append((f"conv[{index}]", tensor))
    fields.append(("temporal", cache.temporal))
    fields.append(("transition", cache.transition))
    if cache.conv_tails is None:
        fields.append(("conv_tails", None))
    else:
        for index, tensor in enumerate(cache.conv_tails):
            fields.append((f"conv_tails[{index}]", tensor))

    records: list[dict[str, Any]] = []
    presence: dict[str, Any] = {}
    for name, tensor in fields:
        if tensor is None:
            presence[name] = {"present": False}
            continue
        presence[name] = {"present": True, "shape": [int(v) for v in tensor.shape]}
        _require(tensor.ndim >= 3 and tensor.is_contiguous(), f"{name} layout")
        element = int(tensor.element_size())
        row_elements = 1
        for value in tensor.shape[2:]:
            row_elements *= int(value)
        _require(int(tensor.stride(1)) == row_elements, f"{name} slot stride")
        base = int(tensor.storage_offset()) * element
        for layer in range(int(tensor.shape[0])):
            layer_base = base + layer * int(tensor.stride(0)) * element
            for first, after in _contiguous_runs(slots):
                _require(first > 0 and after <= int(tensor.shape[1]), f"{name} slot bounds")
                start = layer_base + first * row_elements * element
                end = layer_base + after * row_elements * element
                records.append(
                    _record_range(
                        tensor,
                        tensor_name=f"mamba.{name}",
                        component=name.split("[")[0],
                        start=start,
                        end=end,
                        selection={
                            "mamba_layer_index": layer,
                            "slot_start": first,
                            "slot_end_exclusive": after,
                        },
                    )
                )
    _require(records, "empty recurrent-state payload ledger")
    return records, presence


def _full_tensor_metadata(tensor: torch.Tensor, name: str) -> dict[str, Any]:
    _require(tensor.is_contiguous(), f"metadata tensor not contiguous: {name}")
    info = _tensor_storage(tensor)
    start = int(tensor.storage_offset()) * int(tensor.element_size())
    end = start + int(tensor.numel()) * int(tensor.element_size())
    return _record_range(
        tensor,
        tensor_name=name,
        component="cache_index_metadata",
        start=start,
        end=end,
        selection={"whole_tensor_view": True},
    )


def _union_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_storage: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    for record in records:
        key = (
            str(record["device"]),
            int(record["storage_data_ptr"]),
            int(record["storage_nbytes"]),
        )
        start, end = int(record["byte_start"]), int(record["byte_end"])
        _require(0 <= start < end <= key[2], "range outside backing storage")
        by_storage.setdefault(key, []).append((start, end))
    total = 0
    storage_rows = []
    for key, intervals in sorted(by_storage.items()):
        intervals.sort()
        merged: list[list[int]] = []
        for start, end in intervals:
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        unique = sum(end - start for start, end in merged)
        total += unique
        storage_rows.append(
            {
                "device": key[0],
                "storage_data_ptr": key[1],
                "storage_nbytes": key[2],
                "merged_byte_ranges": merged,
                "unique_bytes": unique,
            }
        )
    return {
        "record_count": len(records),
        "storage_count": len(storage_rows),
        "naive_range_bytes": sum(int(row["range_bytes"]) for row in records),
        "unique_overlap_aware_bytes": total,
        "storages": storage_rows,
    }


def _prefix_selection(tree_cache, target: dict[str, Any]) -> dict[str, Any]:
    document = [int(value) for value in target["document_token_ids"]]
    candidates: list[tuple[int, Any, list[int]]] = []

    def visit(node, prefix: list[int]) -> None:
        for child in node.children.values():
            key = [int(value) for value in child.key.raw_token_ids()]
            path = prefix + key
            if len(path) <= len(document) and path == document[: len(path)]:
                if child.mamba_value is not None:
                    candidates.append((len(path), child, path))
                visit(child, path)

    visit(tree_cache.root_node, [])
    if not candidates:
        raise TargetNotReady("target prefix not present yet")
    _, terminal, cached_tokens = max(candidates, key=lambda row: row[0])
    path_nodes = []
    node = terminal
    while node != tree_cache.root_node:
        path_nodes.append(node)
        node = node.parent
    path_nodes.reverse()
    reconstructed: list[int] = []
    entries = []
    full_slots: list[int] = []
    mamba_slots: list[int] = []
    metadata_records: list[dict[str, Any]] = []
    metadata_exact_bytes = 0
    for node in path_nodes:
        key = [int(value) for value in node.key.raw_token_ids()]
        reconstructed.extend(key)
        slots = [int(value) for value in node.value.detach().cpu().tolist()]
        full_slots.extend(slots)
        metadata_records.append(_full_tensor_metadata(node.value, f"prefix.node[{node.id}].full_kv_slots"))
        key_array = array("q", key)
        key_bytes = int(key_array.buffer_info()[1] * key_array.itemsize)
        metadata_exact_bytes += key_bytes
        node_mamba = []
        if node.mamba_value is not None:
            node_mamba = [int(value) for value in node.mamba_value.detach().cpu().tolist()]
            mamba_slots.extend(node_mamba)
            metadata_records.append(_full_tensor_metadata(node.mamba_value, f"prefix.node[{node.id}].mamba_slots"))
        entries.append(
            {
                "node_id": int(node.id),
                "token_ids": key,
                "token_sha256": token_sha256(key),
                "token_count": len(key),
                "full_kv_slots": slots,
                "mamba_state_slots": node_mamba,
                "lock_refs": {
                    "full": int(node.full_lock_ref),
                    "mamba": int(node.mamba_lock_ref),
                },
                "exact_key_array_bytes": key_bytes,
                "python_node_shallow_bytes": int(sys.getsizeof(node)),
            }
        )
    _require(reconstructed == cached_tokens, "prefix path reconstruction")
    _require(len(full_slots) == len(cached_tokens), "prefix slot coverage")
    _require(len(set(full_slots)) == len(full_slots), "prefix KV slot alias")
    _require(mamba_slots and len(set(mamba_slots)) == len(mamba_slots), "prefix Mamba slots")
    return {
        "cache_kind": "MambaRadixCache",
        "entries": entries,
        "owned_document_token_ids": cached_tokens,
        "owned_document_token_sha256": token_sha256(cached_tokens),
        "owned_document_tokens": len(cached_tokens),
        "expected_measured_cached_tokens": len(cached_tokens),
        "full_kv_slots": full_slots,
        "mamba_state_slots": mamba_slots,
        "metadata_tensor_records": metadata_records,
        "metadata_exact_non_tensor_bytes": metadata_exact_bytes,
    }


def _hypic_selection(tree_cache, target: dict[str, Any]) -> dict[str, Any]:
    segments = target.get("segment_token_ids")
    _require(isinstance(segments, list) and len(segments) == 2, "HYPIC segment target")
    entries = []
    full_slots: list[int] = []
    mamba_slots: list[int] = []
    metadata_records: list[dict[str, Any]] = []
    metadata_exact_bytes = 0
    reconstructed: list[int] = []
    for index, raw in enumerate(segments):
        token_ids = [int(value) for value in raw]
        seg_hash = bytes.fromhex(segment_hash_hex(token_ids))
        entry = tree_cache._entries.get(seg_hash)
        if entry is None:
            raise TargetNotReady(f"target HYPIC segment absent yet: {index}")
        observed = [int(value) for value in entry.token_ids.detach().cpu().tolist()]
        _require(observed == token_ids, f"target HYPIC segment token drift: {index}")
        slots = [int(value) for value in entry.full_kv_slots.detach().cpu().tolist()]
        _require(len(slots) == len(token_ids), "HYPIC KV slot coverage")
        mamba_slot = int(entry.mamba_state_slot)
        _require(mamba_slot > 0, "reserved HYPIC mamba slot")
        reconstructed.extend(token_ids)
        full_slots.extend(slots)
        mamba_slots.append(mamba_slot)
        metadata_records.extend(
            [
                _full_tensor_metadata(entry.full_kv_slots, f"hypic.segment[{index}].full_kv_slots"),
                _full_tensor_metadata(entry.token_ids, f"hypic.segment[{index}].token_ids"),
            ]
        )
        metadata_exact_bytes += len(entry.seg_hash)
        entries.append(
            {
                "segment_index": index,
                "segment_hash_hex": entry.seg_hash.hex(),
                "token_ids": token_ids,
                "token_sha256": token_sha256(token_ids),
                "token_count": len(token_ids),
                "full_kv_slots": slots,
                "mamba_state_slot": mamba_slot,
                "lock_ref": int(entry.lock_ref),
                "exact_segment_hash_bytes": len(entry.seg_hash),
                "python_entry_shallow_bytes": int(sys.getsizeof(entry)),
            }
        )
    _require(reconstructed == target["document_token_ids"], "HYPIC document reconstruction")
    _require(len(set(full_slots)) == len(full_slots), "HYPIC KV slot alias")
    _require(len(set(mamba_slots)) == len(mamba_slots), "HYPIC Mamba slot alias")
    seam = int(target.get("seam_tokens", 8))
    return {
        "cache_kind": "PICache",
        "entries": entries,
        "owned_document_token_ids": reconstructed,
        "owned_document_token_sha256": token_sha256(reconstructed),
        "owned_document_tokens": len(reconstructed),
        "expected_measured_cached_tokens": len(reconstructed) - seam,
        "full_kv_slots": full_slots,
        "mamba_state_slots": mamba_slots,
        "metadata_tensor_records": metadata_records,
        "metadata_exact_non_tensor_bytes": metadata_exact_bytes,
    }


def _selection(tree_cache, target: dict[str, Any]) -> dict[str, Any]:
    if target["mode"] == "prefix_cache":
        _require(type(tree_cache).__name__ == "MambaRadixCache", "prefix cache type")
        return _prefix_selection(tree_cache, target)
    _require(type(tree_cache).__name__ == "PICache", "HYPIC cache type")
    return _hypic_selection(tree_cache, target)


def maybe_emit_owned_state_snapshot(tree_cache) -> None:
    loaded = _load_target()
    if loaded is None:
        return
    target, output_dir = loaded
    output = output_dir / f"{target['snapshot_id']}.json"
    if output.exists():
        return
    try:
        selection = _selection(tree_cache, target)
    except TargetNotReady:
        # The final SSE event can become visible to the client immediately
        # before the scheduler releases the prior warm request.  A target file
        # may therefore exist for one harmless hook invocation before its
        # formal-prime entry exists.  Only absence is retryable; malformed or
        # partially bound entries remain fatal.
        return
    payload_records = _kv_payload_ranges(tree_cache, selection["full_kv_slots"])
    mamba_records, presence = _mamba_payload_ranges(tree_cache, selection["mamba_state_slots"])
    payload_records.extend(mamba_records)
    metadata_records = selection.pop("metadata_tensor_records")
    payload = {
        "schema": SCHEMA,
        "status": "owned_state_snapshot_complete",
        "official_commit": OFFICIAL_COMMIT,
        "target": {
            "snapshot_id": target["snapshot_id"],
            "mode": target["mode"],
            "rank": int(target["rank"]),
            "workload_id": target["workload_id"],
            "document_token_sha256": target["document_token_sha256"],
            "document_tokens": len(target["document_token_ids"]),
            "seam_tokens": int(target.get("seam_tokens", 0)),
        },
        "selection": selection,
        "tensor_payload": {
            "records": payload_records,
            "union": _union_summary(payload_records),
            "denominator": "unique backing-storage byte ranges owned by exact cache entries",
        },
        "metadata": {
            "tensor_records": metadata_records,
            "tensor_union": _union_summary(metadata_records),
            "exact_non_tensor_bytes": int(selection["metadata_exact_non_tensor_bytes"]),
            "excluded_from_store_mib": True,
            "python_allocator_overhead": "not attributed; shallow object sizes retained per entry",
        },
        "component_presence": presence,
        "allocator_observation": {
            "kv_available_tokens": int(tree_cache.token_to_kv_pool_allocator.available_size()),
            "kv_capacity_tokens": int(tree_cache.token_to_kv_pool_allocator.size),
            "kv_page_size": int(tree_cache.token_to_kv_pool_allocator.page_size),
            "mamba_available_slots": int(tree_cache.req_to_token_pool.mamba_allocator.available_size()),
            "mamba_capacity_slots": int(tree_cache.req_to_token_pool.mamba_allocator.size),
        },
        "forbidden_denominators": ["NVML", "process_allocation", "pool_capacity_delta"],
    }
    _atomic_json(output, payload)


def _free_kv_slots(allocator) -> set[int]:
    pages = []
    for name in ("free_pages", "release_pages"):
        value = getattr(allocator, name, None)
        if isinstance(value, torch.Tensor):
            pages.extend(int(v) for v in value.detach().cpu().tolist())
    page_size = int(allocator.page_size)
    if page_size == 1:
        return set(pages)
    return {page * page_size + offset for page in pages for offset in range(page_size)}


def maybe_emit_terminal_ownership_snapshot(scheduler) -> None:
    loaded = _load_target()
    if loaded is None:
        return
    target, output_dir = loaded
    main_path = output_dir / f"{target['snapshot_id']}.json"
    terminal_path = output_dir / f"{target['snapshot_id']}.terminal.json"
    if terminal_path.exists() or not main_path.is_file():
        return
    prior = json.loads(main_path.read_text())
    old_kv = [int(value) for value in prior["selection"]["full_kv_slots"]]
    old_mamba = [int(value) for value in prior["selection"]["mamba_state_slots"]]
    kv_free = _free_kv_slots(scheduler.token_to_kv_pool_allocator)
    mamba_free = set(
        int(value)
        for value in scheduler.req_to_token_pool.mamba_allocator.free_slots.detach().cpu().tolist()
    )
    if target["mode"] == "transition_rope_recompute":
        target_entries_after = sum(
            1
            for segment in target["segment_token_ids"]
            if bytes.fromhex(segment_hash_hex(segment)) in scheduler.tree_cache._entries
        )
    else:
        target_entries_after = 0
        document = target["document_token_ids"]

        def visit(node, prefix: list[int]) -> None:
            nonlocal target_entries_after
            for child in node.children.values():
                key = [int(value) for value in child.key.raw_token_ids()]
                path = prefix + key
                if len(path) <= len(document) and path == document[: len(path)]:
                    target_entries_after += 1
                    visit(child, path)

        visit(scheduler.tree_cache.root_node, [])
    checks = {
        "target_entries_after": target_entries_after,
        "old_kv_slots_all_free": all(slot in kv_free for slot in old_kv),
        "old_mamba_slots_all_free": all(slot in mamba_free for slot in old_mamba),
        "kv_available_tokens": int(scheduler.token_to_kv_pool_allocator.available_size()),
        "kv_capacity_tokens": int(scheduler.token_to_kv_pool_allocator.size),
        "mamba_available_slots": int(scheduler.req_to_token_pool.mamba_allocator.available_size()),
        "mamba_capacity_slots": int(scheduler.req_to_token_pool.mamba_allocator.size),
    }
    passed = (
        target_entries_after == 0
        and checks["old_kv_slots_all_free"]
        and checks["old_mamba_slots_all_free"]
        and checks["kv_available_tokens"] == checks["kv_capacity_tokens"]
        and checks["mamba_available_slots"] == checks["mamba_capacity_slots"]
    )
    _require(passed, f"terminal ownership removal failed: {checks}")
    _atomic_json(
        terminal_path,
        {
            "schema": "forkaudit-hypic-retained-state-terminal-v1",
            "status": "terminal_ownership_removal_complete",
            "official_commit": OFFICIAL_COMMIT,
            "snapshot_id": target["snapshot_id"],
            "passed": True,
            "checks": checks,
            "prior_receipt_sha256": hashlib.sha256(main_path.read_bytes()).hexdigest(),
        },
    )
