#!/usr/bin/env python3
"""Blind replay for RW-D5 raw object-owned byte receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import struct
from pathlib import Path
from typing import Any


MODES = ("prefix_cache", "transition_rope_recompute")
EXPECTED_PAIRS = (
    ("qasper", 6),
    ("qasper", 7),
    ("qasper", 8),
    ("qasper", 9),
    ("2wikimqa", 6),
    ("2wikimqa", 7),
    ("2wikimqa", 8),
    ("2wikimqa", 9),
)
OFFICIAL_COMMIT = "98147c01909004e66d98bcb18b886927d41b0ee5"


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    data = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def token_sha256(values: list[int]) -> str:
    payload = b"".join(struct.pack("<i", int(value)) for value in values)
    return hashlib.sha256(payload).hexdigest()


def replay_union(records: list[dict[str, Any]]) -> dict[str, int]:
    grouped: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    naive = 0
    for row in records:
        key = (
            str(row["device"]),
            int(row["storage_data_ptr"]),
            int(row["storage_nbytes"]),
        )
        start, end = int(row["byte_start"]), int(row["byte_end"])
        require(0 <= start < end <= key[2], "range outside backing storage")
        require(end - start == int(row["range_bytes"]), "range byte drift")
        require(
            int(row["absolute_byte_start"]) == key[1] + start
            and int(row["absolute_byte_end"]) == key[1] + end,
            "absolute range drift",
        )
        naive += end - start
        grouped.setdefault(key, []).append((start, end))
    unique = 0
    for intervals in grouped.values():
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        unique += sum(end - start for start, end in merged)
    return {"record_count": len(records), "naive_range_bytes": naive, "unique_bytes": unique}


def replay_one(receipt_path: Path, terminal_path: Path, raw_path: Path) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text())
    terminal = json.loads(terminal_path.read_text())
    raw = json.loads(raw_path.read_text())
    require(receipt.get("schema") == "forkaudit-hypic-retained-state-receipt-v1", "receipt schema")
    require(receipt.get("official_commit") == OFFICIAL_COMMIT, "receipt commit")
    require(terminal.get("schema") == "forkaudit-hypic-retained-state-terminal-v1", "terminal schema")
    require(terminal.get("passed") is True, "terminal gate")
    require(raw.get("schema") == "forkaudit-hypic-retained-state-shard-v1", "raw schema")
    require(raw.get("status") == "completed", "raw status")
    target = receipt["target"]
    require(target["mode"] == raw["mode"] and target["rank"] == raw["rank"], "cell binding")
    require(target["workload_id"] == raw["workload"]["workload_id"], "workload binding")
    require(target["document_token_sha256"] == raw["workload"]["document_token_sha256"], "document binding")
    selection = receipt["selection"]
    owned = selection["owned_document_token_ids"]
    require(token_sha256(owned) == selection["owned_document_token_sha256"], "owned token hash")
    require(len(owned) == selection["owned_document_tokens"], "owned token count")
    full_slots = [int(value) for value in selection["full_kv_slots"]]
    mamba_slots = [int(value) for value in selection["mamba_state_slots"]]
    require(len(full_slots) == len(set(full_slots)) == selection["owned_document_tokens"], "KV ownership coverage")
    require(mamba_slots and len(mamba_slots) == len(set(mamba_slots)), "Mamba ownership coverage")
    for entry in selection["entries"]:
        require(token_sha256(entry["token_ids"]) == entry["token_sha256"], "entry token hash")
        if raw["mode"] == "transition_rope_recompute":
            seg_hash = hashlib.sha256(
                b"".join(struct.pack("<i", int(value)) for value in entry["token_ids"])
            ).digest()[:16].hex()
            require(seg_hash == entry["segment_hash_hex"], "segment hash")
    payload_replay = replay_union(receipt["tensor_payload"]["records"])
    reported = receipt["tensor_payload"]["union"]
    require(payload_replay["record_count"] == reported["record_count"], "payload record count")
    require(payload_replay["naive_range_bytes"] == reported["naive_range_bytes"], "payload naive sum")
    require(payload_replay["unique_bytes"] == reported["unique_overlap_aware_bytes"], "payload unique sum")
    metadata_replay = replay_union(receipt["metadata"]["tensor_records"])
    require(
        metadata_replay["unique_bytes"]
        == receipt["metadata"]["tensor_union"]["unique_overlap_aware_bytes"],
        "metadata unique sum",
    )
    require(receipt["metadata"]["excluded_from_store_mib"] is True, "metadata denominator")
    require(
        int(raw["cache_observation"]["cached_tokens"])
        == int(selection["expected_measured_cached_tokens"]),
        "cache-hit coverage",
    )
    require(
        terminal["prior_receipt_sha256"] == sha256_file(receipt_path),
        "terminal receipt binding",
    )
    checks = terminal["checks"]
    require(
        checks["target_entries_after"] == 0
        and checks["old_kv_slots_all_free"] is True
        and checks["old_mamba_slots_all_free"] is True
        and checks["kv_available_tokens"] == checks["kv_capacity_tokens"]
        and checks["mamba_available_slots"] == checks["mamba_capacity_slots"],
        "terminal ownership restoration",
    )
    return {
        "mode": raw["mode"],
        "rank": raw["rank"],
        "workload_id": target["workload_id"],
        "owned_document_tokens": selection["owned_document_tokens"],
        "measured_cached_tokens": raw["cache_observation"]["cached_tokens"],
        "payload_bytes": payload_replay["unique_bytes"],
        "payload_mib": payload_replay["unique_bytes"] / (1024 * 1024),
        "metadata_tensor_bytes": metadata_replay["unique_bytes"],
        "metadata_non_tensor_bytes": receipt["metadata"]["exact_non_tensor_bytes"],
        "receipt_sha256": sha256_file(receipt_path),
        "terminal_sha256": sha256_file(terminal_path),
        "raw_sha256": sha256_file(raw_path),
    }


def replay_all(root: Path, output: Path) -> None:
    rows = []
    for mode in MODES:
        for rank, pair in enumerate(EXPECTED_PAIRS):
            snapshot_id = f"{mode}-rank-{rank}"
            row = replay_one(
                root / "store-receipts" / f"{snapshot_id}.json",
                root / "store-receipts" / f"{snapshot_id}.terminal.json",
                root / "raw" / f"{snapshot_id}.json",
            )
            require(row["workload_id"] == f"{pair[0]}-{pair[1]}", "frozen workload order")
            rows.append(row)
    summaries = {}
    for mode in MODES:
        cells = [row for row in rows if row["mode"] == mode]
        summaries[mode] = {
            "median_payload_bytes": int(statistics.median(row["payload_bytes"] for row in cells)),
            "median_payload_mib": statistics.median(row["payload_mib"] for row in cells),
            "payload_bytes": [row["payload_bytes"] for row in cells],
            "owned_document_tokens": [row["owned_document_tokens"] for row in cells],
            "measured_cached_tokens": [row["measured_cached_tokens"] for row in cells],
        }
    atomic_json(
        output,
        {
            "schema": "forkaudit-hypic-retained-state-blind-replay-v1",
            "passed": True,
            "official_commit": OFFICIAL_COMMIT,
            "denominator": "unique object-owned tensor payload byte ranges; metadata excluded",
            "rows": rows,
            "modes": summaries,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    replay_all(args.root, args.output)


if __name__ == "__main__":
    main()
