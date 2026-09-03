#!/usr/bin/env python3
"""Blind, producer-independent replay for RW-D5 retained-state receipts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import struct
import subprocess
from pathlib import Path
from typing import Any


MODES = ("prefix_cache", "transition_rope_recompute")
EXPECTED_PAIRS = (
    ("qasper", 6), ("qasper", 7), ("qasper", 8), ("qasper", 9),
    ("2wikimqa", 6), ("2wikimqa", 7), ("2wikimqa", 8), ("2wikimqa", 9),
)
OFFICIAL_COMMIT = "98147c01909004e66d98bcb18b886927d41b0ee5"
DTYPE_BYTES = {
    "torch.float32": 4, "torch.float16": 2, "torch.bfloat16": 2,
    "torch.int64": 8, "torch.int32": 4, "torch.int8": 1, "torch.uint8": 1,
}


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


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_rwd5_canonical_sha256(value: Any, digest: str, label: str) -> None:
    require(canonical_sha256(value) == digest, label)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def token_sha256(values: list[int]) -> str:
    payload = b"".join(struct.pack("<i", int(value)) for value in values)
    return hashlib.sha256(payload).hexdigest()


def _load(path: Path, schema: str, label: str) -> dict[str, Any]:
    require(path.is_file(), f"{label} file")
    value = json.loads(path.read_text())
    require(value.get("schema") == schema, f"{label} schema")
    return value


def _is_c_contiguous(shape: list[int], stride: list[int]) -> bool:
    expected = 1
    for size, observed in zip(reversed(shape), reversed(stride)):
        if size != 1 and observed != expected:
            return False
        expected *= size
    return True


def _numel(shape: list[int]) -> int:
    total = 1
    for value in shape:
        require(isinstance(value, int) and value > 0, "positive tensor shape")
        total *= value
    return total


def derive_range(row: dict[str, Any]) -> tuple[tuple[str, int, int], int, int]:
    """Derive the selected physical range without using producer offsets."""
    dtype = row.get("dtype")
    require(dtype in DTYPE_BYTES, "known dtype")
    element = DTYPE_BYTES[str(dtype)]
    require(int(row.get("element_size", -1)) == element, "dtype/element size")
    shape = row.get("shape")
    stride = row.get("stride")
    require(isinstance(shape, list) and isinstance(stride, list), "shape/stride metadata")
    shape = [int(value) for value in shape]
    stride = [int(value) for value in stride]
    require(len(shape) == len(stride) and shape, "shape/stride rank")
    require(_is_c_contiguous(shape, stride), "C-contiguous stride")
    storage_offset = int(row.get("storage_offset_elements", -1))
    storage_base = int(row.get("storage_data_ptr", -1))
    storage_nbytes = int(row.get("storage_nbytes", -1))
    require(storage_offset >= 0 and storage_base > 0 and storage_nbytes > 0, "storage metadata")
    require(
        int(row.get("tensor_data_ptr", -1)) == storage_base + storage_offset * element,
        "pointer-relative tensor identity",
    )
    device = str(row.get("device"))
    expected_storage_id = hashlib.sha256(f"{device}:{storage_base}:{storage_nbytes}".encode()).hexdigest()
    require(row.get("storage_id") == expected_storage_id, "storage id")
    full_end = (storage_offset + _numel(shape)) * element
    require(full_end <= storage_nbytes, "full tensor inside storage")
    selection = row.get("selection")
    require(isinstance(selection, dict), "selection metadata")
    kind = selection.get("kind")
    if kind == "axis0_slots":
        require(len(shape) == 3, "KV rank")
        first = int(selection.get("slot_start", -1))
        after = int(selection.get("slot_end_exclusive", -1))
        require(0 < first < after <= shape[0], "KV selected slots")
        start = (storage_offset + first * stride[0]) * element
        end = (storage_offset + after * stride[0]) * element
    elif kind == "axis1_slots_at_layer":
        require(len(shape) >= 3, "recurrent rank")
        layer = int(selection.get("mamba_layer_index", -1))
        first = int(selection.get("slot_start", -1))
        after = int(selection.get("slot_end_exclusive", -1))
        require(0 <= layer < shape[0] and 0 < first < after <= shape[1], "recurrent selection")
        start = (storage_offset + layer * stride[0] + first * stride[1]) * element
        end = (storage_offset + layer * stride[0] + after * stride[1]) * element
    elif kind == "whole_tensor":
        start = storage_offset * element
        end = full_end
    else:
        raise ReplayError("unknown range selection")
    require(0 <= start < end <= storage_nbytes, "derived range inside storage")
    # Producer fields are assertions to check, never inputs to the derivation.
    require(int(row.get("byte_start", -1)) == start, "producer byte_start drift")
    require(int(row.get("byte_end", -1)) == end, "producer byte_end drift")
    require(int(row.get("range_bytes", -1)) == end - start, "producer range_bytes drift")
    require(int(row.get("absolute_byte_start", -1)) == storage_base + start, "absolute start drift")
    require(int(row.get("absolute_byte_end", -1)) == storage_base + end, "absolute end drift")
    return (device, storage_base, storage_nbytes), start, end


def replay_union(records: list[dict[str, Any]]) -> dict[str, int]:
    require(isinstance(records, list) and records, "nonempty range records")
    grouped: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    naive = 0
    tensor_metadata: dict[str, tuple[Any, ...]] = {}
    for row in records:
        key, start, end = derive_range(row)
        tensor_name = str(row.get("tensor_name"))
        signature = (
            row["device"], row["storage_data_ptr"], row["storage_nbytes"],
            row["storage_id"], row["dtype"], tuple(row["shape"]),
            tuple(row["stride"]), row["element_size"],
            row["storage_offset_elements"], row["tensor_data_ptr"],
        )
        if tensor_name in tensor_metadata:
            require(tensor_metadata[tensor_name] == signature, "tensor metadata stability")
        tensor_metadata[tensor_name] = signature
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


def _slots_from_rows(rows: list[dict[str, Any]], *, layer: int | None = None) -> list[int]:
    values = []
    for row in rows:
        selection = row["selection"]
        if layer is not None and int(selection["mamba_layer_index"]) != layer:
            continue
        values.extend(range(int(selection["slot_start"]), int(selection["slot_end_exclusive"])))
    require(len(values) == len(set(values)), "duplicate tensor slot coverage")
    return sorted(values)


def _validate_selection(selection: dict[str, Any], mode: str, target: dict[str, Any],
                        prereg: dict[str, Any]) -> tuple[list[int], list[int], int]:
    owned = [int(value) for value in selection["owned_document_token_ids"]]
    require(token_sha256(owned) == selection["owned_document_token_sha256"], "owned token hash")
    require(len(owned) == int(selection["owned_document_tokens"]), "owned token count")
    full_slots = [int(value) for value in selection["full_kv_slots"]]
    mamba_slots = [int(value) for value in selection["mamba_state_slots"]]
    require(len(full_slots) == len(set(full_slots)) == len(owned), "KV ownership coverage")
    require(mamba_slots and len(mamba_slots) == len(set(mamba_slots)), "Mamba ownership coverage")
    entry_tokens, entry_kv, entry_mamba = [], [], []
    entries = selection["entries"]
    require(isinstance(entries, list) and entries, "nonempty selection entries")
    if mode == "transition_rope_recompute":
        require(len(entries) == 2, "exact HYPIC entry count")
        require([int(entry["segment_index"]) for entry in entries] == [0, 1], "ordered HYPIC segment indices")
        target_segments = target.get("segment_token_ids")
        require(isinstance(target_segments, list) and len(target_segments) == 2, "exact target HYPIC segments")
    for position, entry in enumerate(entries):
        tokens = [int(value) for value in entry["token_ids"]]
        require(token_sha256(tokens) == entry["token_sha256"], "entry token hash")
        require(len(tokens) == int(entry["token_count"]), "entry token count")
        entry_tokens.extend(tokens)
        entry_kv.extend(int(value) for value in entry["full_kv_slots"])
        if mode == "transition_rope_recompute":
            require(tokens == [int(value) for value in target_segments[position]], "HYPIC entry/target segment binding")
            require(int(entry["lock_ref"]) == 0, "HYPIC stable lock ref")
            seg_hash = hashlib.sha256(b"".join(struct.pack("<i", value) for value in tokens)).digest()[:16].hex()
            require(seg_hash == entry["segment_hash_hex"], "segment hash")
            entry_mamba.append(int(entry["mamba_state_slot"]))
        else:
            require(entry["lock_refs"] == {"full": 0, "mamba": 0}, "Prefix stable lock refs")
            entry_mamba.extend(int(value) for value in entry["mamba_state_slots"])
    require(entry_tokens == owned, "entry token reconstruction")
    require(entry_kv == full_slots and entry_mamba == mamba_slots, "entry slot reconstruction")
    require(selection["cache_kind"] == ("PICache" if mode == "transition_rope_recompute" else "MambaRadixCache"), "cache kind")
    document = [int(value) for value in target["document_token_ids"]]
    if mode == "prefix_cache":
        require(int(target["seam_tokens"]) == 0, "Prefix seam contract")
        require(owned == document[:len(owned)], "Prefix owned/target exact-prefix binding")
        independently_expected_cached_tokens = len(owned)
    else:
        seam = int(prereg["design"]["hypic_seam_tokens"])
        require(0 <= seam < len(document) and int(target["seam_tokens"]) == seam, "HYPIC preregistered seam")
        require(entry_tokens == document and owned == document, "HYPIC owned/target document binding")
        independently_expected_cached_tokens = len(document) - seam
        require(independently_expected_cached_tokens >= 0, "HYPIC cached-token denominator")
    require(
        int(selection["expected_measured_cached_tokens"]) == independently_expected_cached_tokens,
        "producer expected cache-hit field",
    )
    return full_slots, mamba_slots, independently_expected_cached_tokens


def _validate_structure(receipt: dict[str, Any], prereg: dict[str, Any],
                        full_slots: list[int], mamba_slots: list[int]) -> None:
    contract = prereg["model"]["storage_contract"]
    require(receipt["storage_contract"] == contract, "receipt/static storage contract")
    require(canonical_sha256(contract) == prereg["model"]["storage_contract_sha256"], "contract hash")
    require(contract.get("schema") == "hypic-rwd5-model-storage-contract-v3", "storage contract schema")
    require("dtype" not in contract, "legacy unified dtype forbidden")
    component_dtypes = contract.get("mamba_component_dtypes")
    require(
        isinstance(component_dtypes, dict)
        and set(component_dtypes) == {"conv", "temporal", "transition", "conv_tails"},
        "exact component dtype contract",
    )
    require(contract.get("kv_dtype") in DTYPE_BYTES, "known KV dtype contract")
    require(all(value in DTYPE_BYTES for value in component_dtypes.values()), "known component dtype contract")
    require(component_dtypes["conv_tails"] == component_dtypes["conv"], "conv tail dtype authority")
    require(component_dtypes["transition"] == component_dtypes["temporal"], "transition dtype authority")
    require(contract["enable_int8_mamba_checkpoint"] is False, "int8 checkpoint contract")
    mode = receipt["target"]["mode"]
    records = receipt["tensor_payload"]["records"]
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_name.setdefault(str(row["tensor_name"]), []).append(row)
    full_count = int(contract["full_attention_layer_count"])
    expected_k = [f"full_kv.key[{i}]" for i in range(full_count)]
    expected_v = [f"full_kv.value[{i}]" for i in range(full_count)]
    mode_contract = contract["mode_components"][mode]
    expected_mamba = [f"mamba.conv[{i}]" for i in range(int(contract["conv_tensor_count"]))] + ["mamba.temporal"]
    if mode == "transition_rope_recompute":
        expected_mamba += ["mamba.transition"] + [
            f"mamba.conv_tails[{i}]" for i in range(int(mode_contract["conv_tails_tensor_count"]))
        ]
    expected_names = set(expected_k + expected_v + expected_mamba)
    require(set(by_name) == expected_names, "exact payload tensor keys")
    kv_capacity = int(receipt["allocator_observation"]["kv_capacity_tokens"])
    mamba_capacity = int(receipt["allocator_observation"]["mamba_capacity_slots"])
    pre = receipt["allocator_observation"]["pre_free_ownership"]
    kv_pre = pre["kv"]
    require(kv_pre["page_size"] == 1 and int(kv_pre["size"]) == kv_capacity, "pre KV domain type")
    kv_free = [int(value) for value in kv_pre["free_pages"]]
    kv_release = [int(value) for value in kv_pre["release_pages"]]
    kv_combined = kv_free + kv_release
    kv_expected_domain = set(range(1, kv_capacity + 1))
    require(kv_free == sorted(kv_free) and kv_release == sorted(kv_release), "canonical pre KV lists")
    require(len(kv_combined) == len(set(kv_combined)), "duplicate pre KV free domain")
    require(set(kv_combined).issubset(kv_expected_domain), "pre KV free subset domain")
    require(kv_pre["canonical_free_domain"] == sorted(kv_combined), "canonical pre KV free domain")
    require(
        kv_pre["canonical_allocated_domain"] == sorted(kv_expected_domain - set(kv_combined)),
        "canonical pre KV allocated domain",
    )
    require(receipt["allocator_observation"]["kv_available_tokens"] == len(kv_combined), "pre KV available count")
    require(set(full_slots).issubset(kv_expected_domain - set(kv_combined)), "selected KV preallocated")
    mamba_pre = pre["mamba"]
    require(int(mamba_pre["size"]) == mamba_capacity, "pre Mamba domain type")
    mamba_free = [int(value) for value in mamba_pre["free_slots"]]
    mamba_expected_domain = set(range(1, mamba_capacity + 1))
    require(
        mamba_free == sorted(mamba_free) and len(mamba_free) == len(set(mamba_free)),
        "canonical pre Mamba free domain",
    )
    require(set(mamba_free).issubset(mamba_expected_domain), "pre Mamba free subset domain")
    require(mamba_pre["canonical_free_domain"] == mamba_free, "recorded pre Mamba free domain")
    require(
        mamba_pre["canonical_allocated_domain"] == sorted(mamba_expected_domain - set(mamba_free)),
        "canonical pre Mamba allocated domain",
    )
    require(receipt["allocator_observation"]["mamba_available_slots"] == len(mamba_free), "pre Mamba available count")
    require(set(mamba_slots).issubset(mamba_expected_domain - set(mamba_free)), "selected Mamba preallocated")
    kv_shapes: dict[int, list[int]] = {}
    for name in expected_k + expected_v:
        rows = by_name[name]
        require(all(row["selection"]["kind"] == "axis0_slots" for row in rows), "KV selection kind")
        require(_slots_from_rows(rows) == sorted(full_slots), "KV layer x slot coverage")
        shape = [int(value) for value in rows[0]["shape"]]
        require(len(shape) == 3 and shape[0] == kv_capacity + 1, "KV capacity axis")
        require(rows[0]["dtype"] == contract["kv_dtype"], "KV contract dtype")
        layer = int(name.rsplit("[", 1)[1][:-1])
        if name.startswith("full_kv.key"):
            kv_shapes[layer] = shape
        else:
            require(shape == kv_shapes[layer], "K/V shape equality")
    recurrent_layers = int(contract["recurrent_layer_count"])
    presence = receipt["component_presence"]
    require(set(presence) == {name.removeprefix("mamba.") for name in expected_mamba}, "exact presence keys")
    for name in expected_mamba:
        rows = by_name[name]
        shape = [int(value) for value in rows[0]["shape"]]
        require(shape[0] == recurrent_layers and shape[1] == mamba_capacity + 1, "recurrent layer/slot axes")
        component = "conv_tails" if name.startswith("mamba.conv_tails[") else (
            "conv" if name.startswith("mamba.conv[") else name.removeprefix("mamba.")
        )
        require(rows[0]["dtype"] == component_dtypes[component], "recurrent component dtype")
        for layer in range(recurrent_layers):
            require(_slots_from_rows(rows, layer=layer) == sorted(mamba_slots), "recurrent layer x slot coverage")
        key = name.removeprefix("mamba.")
        require(
            presence[key]
            == {
                "present": True,
                "shape": shape,
                "dtype": component_dtypes[component],
                "element_size": DTYPE_BYTES[component_dtypes[component]],
            },
            "component presence/shape/dtype",
        )
    if mode == "prefix_cache":
        require("mamba.transition" not in by_name and not any(name.startswith("mamba.conv_tails") for name in by_name), "Prefix PIC components absent")
    else:
        require("mamba.transition" in by_name and any(name.startswith("mamba.conv_tails") for name in by_name), "HYPIC PIC components required")


def _validate_metadata(receipt: dict[str, Any], mode: str) -> None:
    selection = receipt["selection"]
    expected: dict[str, int] = {}
    if mode == "prefix_cache":
        for entry in selection["entries"]:
            node = int(entry["node_id"])
            expected[f"prefix.node[{node}].full_kv_slots"] = len(entry["full_kv_slots"])
            if entry["mamba_state_slots"]:
                expected[f"prefix.node[{node}].mamba_slots"] = len(entry["mamba_state_slots"])
    else:
        for entry in selection["entries"]:
            index = int(entry["segment_index"])
            expected[f"hypic.segment[{index}].full_kv_slots"] = len(entry["full_kv_slots"])
            expected[f"hypic.segment[{index}].token_ids"] = len(entry["token_ids"])
    rows = receipt["metadata"]["tensor_records"]
    require(len(rows) == len(expected), "exact metadata record count")
    by_name = {str(row["tensor_name"]): row for row in rows}
    require(len(by_name) == len(rows) and set(by_name) == set(expected), "exact metadata keys")
    for name, count in expected.items():
        row = by_name[name]
        require(row["component"] == "cache_index_metadata", "metadata component")
        require(row["selection"] == {"kind": "whole_tensor"}, "whole metadata tensor")
        require(row["dtype"] == "torch.int64" and row["shape"] == [count], "metadata dtype/shape")


def _manifest_rows(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        require(bool(line) and not line.startswith("\\"), "frozen manifest row")
        fields = line.split(None, 1)
        require(len(fields) == 2, "frozen manifest row")
        digest, relative = fields
        if relative.startswith("*"):
            relative = relative[1:]
        require(
            len(digest) == 64 and all(character in "0123456789abcdefABCDEF" for character in digest),
            "frozen manifest digest",
        )
        require(relative and "\x00" not in relative and "\\" not in relative, "frozen manifest path")
        require(not relative.startswith("/"), "absolute frozen manifest path")
        parts = relative.split("/")
        require(".." not in parts, "frozen manifest traversal")
        canonical = "/".join(part for part in parts if part not in ("", "."))
        require(canonical and canonical not in rows, "unique canonical frozen manifest path")
        rows[canonical] = digest.lower()
    return rows


def _validate_server_info_readiness(server: dict[str, Any], raw: dict[str, Any]) -> None:
    mode = str(raw["mode"])
    rank = int(raw["rank"])
    require(server["mode"] == mode and int(server["rank"]) == rank, "server/raw readiness cell")
    expected_base_url = f"http://127.0.0.1:{33400 + rank}"
    expected_endpoint = expected_base_url + "/server_info"
    require(server["base_url"] == expected_base_url, "frozen server base URL")
    require(server["server_info_endpoint"] == expected_endpoint, "frozen server_info endpoint")
    readiness_row = server["server_info_readiness"]
    readiness = readiness_row["identity"]
    require(canonical_sha256(readiness) == readiness_row["sha256"], "server_info readiness file hash")
    attempts = readiness["attempts"]
    require(
        readiness["schema"] == "hypic-rwd5-server-info-readiness-v1"
        and readiness["status"] == "ready"
        and readiness["mode"] == mode
        and int(readiness["rank"]) == rank
        and int(readiness["server_pid"]) == int(server["frontend_process"]["pid"])
        and readiness["endpoint"] == expected_endpoint
        and readiness["total_timeout_seconds"] == 300.0
        and readiness["single_timeout_seconds"] == 3.0
        and readiness["poll_interval_seconds"] == 1.0
        and int(readiness["attempt_count"]) == len(attempts)
        and attempts
        and [row["attempt"] for row in attempts] == list(range(1, len(attempts) + 1))
        and attempts[-1]["outcome"] == "ready"
        and all(row["outcome"] == "not_ready" for row in attempts[:-1]),
        "server_info readiness exact cell/endpoint/poll closure",
    )
    require(
        attempts[-1]["response_sha256"]
        == readiness["server_info_sha256"]
        == server["server_info_sha256"],
        "server_info readiness/server binding",
    )


def _validate_authority(receipt: dict[str, Any], terminal: dict[str, Any], raw: dict[str, Any],
                        target: dict[str, Any], server: dict[str, Any], worker: dict[str, Any],
                        prereg: dict[str, Any], *, receipt_path: Path, target_path: Path,
                        server_path: Path, worker_path: Path, prereg_path: Path,
                        manifest_path: Path, expected_manifest_sha256: str,
                        static_dir: Path) -> None:
    require(sha256_file(manifest_path) == expected_manifest_sha256, "external manifest SHA")
    subprocess.check_call(["sha256sum", "-c", str(manifest_path)], cwd=manifest_path.parent)
    manifest = _manifest_rows(manifest_path)
    code_hashes = {key: row["sha256"] for key, row in sorted(prereg["code"].items())}
    for row in prereg["code"].values():
        require(manifest.get(f"code/{row['path']}") == row["sha256"], "code/external manifest binding")
    require(sha256_file(static_dir / "official-source-ledger.json") == prereg["official_source_ledger_sha256"], "source ledger")
    require(sha256_file(static_dir / "environment-ledger.json") == prereg["environment_ledger_sha256"], "environment ledger")
    require(sha256_file(static_dir / "model-storage-contract.json") == prereg["model"]["storage_contract_sha256"], "storage contract file")
    overlay = json.loads((static_dir / "instrumentation-overlay.json").read_text())
    require(overlay == prereg["instrumentation"]["overlay"], "overlay ledger")
    diff_sha = sha256_file(static_dir / "instrumentation-overlay.diff")
    require(diff_sha == overlay["canonical_diff_sha256"], "canonical overlay diff")
    require(overlay["no_other_tracked_or_untracked"] is True, "exact overlay closure")
    require(server["instrumented_overlay"] == {"porcelain_v1": overlay["porcelain_v1"], "canonical_diff_sha256": diff_sha}, "server overlay binding")
    require_rwd5_canonical_sha256(
        server["server_configuration"], server["server_configuration_sha256"], "server config hash"
    )
    _validate_server_info_readiness(server, raw)
    require(server["server_configuration"]["rwd5_expected"] == {"enable_int8_mamba_checkpoint": False, "page_size": 1}, "server int8/page contract")
    frontend = server["frontend_process"]
    process = worker["process"]
    for row, label in ((frontend, "frontend"), (process, "worker")):
        require(all(key in row for key in ("pid", "ppid", "cmdline", "cmdline_sha256", "environment", "environment_sha256")), f"{label} identity fields")
        require_rwd5_canonical_sha256(row["cmdline"], row["cmdline_sha256"], f"{label} cmdline hash")
        require_rwd5_canonical_sha256(
            row["environment"], row["environment_sha256"], f"{label} environment hash"
        )
    require(worker["schema"] == "forkaudit-hypic-scheduler-worker-v2", "worker schema")
    require(worker["frontend_pid"] == frontend["pid"] and frontend["pid"] in process["ancestry_pids"], "scheduler child lineage")
    expected_environment = {
        "FORKAUDIT_RWD5_TARGET_PATH": str(target_path),
        "FORKAUDIT_RWD5_WORKER_RECEIPT_PATH": str(worker_path),
        "FORKAUDIT_RWD5_SERVER_RECEIPT_PATH": str(server_path),
        "FORKAUDIT_RWD5_PREREGISTRATION_PATH": str(prereg_path),
        "FORKAUDIT_RWD5_FREEZE_MANIFEST_PATH": str(manifest_path),
        "FORKAUDIT_RWD5_FREEZE_MANIFEST_SHA256": expected_manifest_sha256,
        "FORKAUDIT_RWD5_MODE": raw["mode"],
        "FORKAUDIT_RWD5_RANK": str(raw["rank"]),
        "FORKAUDIT_RWD5_FRONTEND_PID": str(frontend["pid"]),
        "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
        "SGLANG_MAMBA_SSM_DTYPE": "float32",
    }
    for key, expected in expected_environment.items():
        require(frontend["environment"].get(key) == expected, f"frontend environment {key}")
        require(process["environment"].get(key) == expected, f"worker environment {key}")
    require(
        "FORKAUDIT_RWD5_DTYPE_DEBUG_PATH" not in frontend["environment"]
        and "FORKAUDIT_RWD5_DTYPE_DEBUG_PATH" not in process["environment"],
        "formal replay excludes dtype debug mode",
    )
    require(server["scheduler_worker"]["identity"] == worker, "server worker identity")
    require(server["scheduler_worker"]["receipt_sha256"] == sha256_file(worker_path), "server worker hash")
    require(worker["int8_mamba_checkpoint_enabled"] is False, "worker int8 checkpoint")
    if raw["mode"] == "transition_rope_recompute":
        require(worker["picache_mamba_pool_identity"] is True, "PIC MambaPool identity")
    bindings = {
        "official_commit": OFFICIAL_COMMIT,
        "target_sha256": sha256_file(target_path),
        "preregistration_sha256": sha256_file(prereg_path),
        "server_launch_receipt_sha256": sha256_file(server_path),
        "scheduler_worker_receipt_sha256": sha256_file(worker_path),
        "freeze_manifest_sha256": expected_manifest_sha256,
        "official_source_ledger_sha256": prereg["official_source_ledger_sha256"],
        "environment_ledger_sha256": prereg["environment_ledger_sha256"],
        "data_sha256": prereg["data"]["sha256"],
        "model_weight_ledger_sha256": prereg["model"]["weight_ledger_raw_sha256"],
        "model_artifact_ledger_sha256": prereg["model"]["artifact_ledger_raw_sha256"],
        "model_config_sha256": prereg["model"]["config_sha256"],
        "storage_contract_sha256": prereg["model"]["storage_contract_sha256"],
        "overlay_diff_sha256": diff_sha,
        "code_sha256": code_hashes,
        "server_configuration_sha256": server["server_configuration_sha256"],
    }
    target_authority = dict(bindings); target_authority.pop("target_sha256")
    require(target["authority"] == target_authority, "target authority")
    require(server["authority"] == {key: bindings[key] for key in server["authority"]}, "server authority")
    require(receipt["authority"]["bindings"] == bindings, "receipt authority")
    require(terminal["authority"] == receipt["authority"], "terminal authority")
    require(raw["authority"] == receipt["authority"], "raw authority")
    scheduler_process = receipt["authority"]["scheduler_process"]
    for key in ("pid", "ppid", "cmdline_sha256", "environment"):
        require(scheduler_process[key] == process[key], f"receipt scheduler {key}")
    require(raw["target"] == target and raw["target_sha256"] == bindings["target_sha256"], "raw target binding")
    require(raw["server_launch_receipt_sha256"] == bindings["server_launch_receipt_sha256"], "raw server binding")
    require(raw["preregistration_sha256"] == bindings["preregistration_sha256"], "raw prereg binding")
    require(raw["freeze_manifest_sha256"] == bindings["freeze_manifest_sha256"], "raw manifest binding")
    require(raw["store_receipt"]["sha256"] == sha256_file(receipt_path), "raw store binding")


def _validate_terminal(terminal: dict[str, Any], selection: dict[str, Any],
                       receipt: dict[str, Any], receipt_path: Path) -> None:
    require(terminal["passed"] is True and terminal["prior_receipt_sha256"] == sha256_file(receipt_path), "terminal receipt binding")
    checks = terminal["checks"]
    kv = checks["kv_free_list"]; mamba = checks["mamba_free_list"]
    kv_free = [int(value) for value in kv["free_pages"]]
    kv_release = [int(value) for value in kv["release_pages"]]
    kv_expected = list(range(1, int(kv["size"]) + 1))
    require(kv["page_size"] == 1 and not kv_release, "terminal KV domain type")
    require(len(kv_free) == len(set(kv_free)) and sorted(kv_free) == kv_expected and kv["exact_domain"] == kv_expected, "terminal exact KV free domain")
    mamba_free = [int(value) for value in mamba["free_slots"]]
    mamba_expected = list(range(1, int(mamba["size"]) + 1))
    require(len(mamba_free) == len(set(mamba_free)) and sorted(mamba_free) == mamba_expected and mamba["exact_domain"] == mamba_expected, "terminal exact Mamba free domain")
    require(all(int(slot) in kv_free for slot in selection["full_kv_slots"]), "old KV slots returned")
    require(all(int(slot) in mamba_free for slot in selection["mamba_state_slots"]), "old Mamba slots returned")
    pre = receipt["allocator_observation"]["pre_free_ownership"]
    require(
        set(int(slot) for slot in selection["full_kv_slots"]).isdisjoint(pre["kv"]["canonical_free_domain"]),
        "old KV slots preallocated",
    )
    require(
        set(int(slot) for slot in selection["mamba_state_slots"]).isdisjoint(pre["mamba"]["canonical_free_domain"]),
        "old Mamba slots preallocated",
    )
    require(checks["target_entries_after"] == 0 and checks["all_cache_entries_after"] == 0, "terminal cache index empty")
    require(
        checks["old_kv_slots_preallocated"] is True and checks["old_mamba_slots_preallocated"] is True,
        "preallocated ownership flags",
    )
    require(checks["old_kv_slots_all_free"] is True and checks["old_mamba_slots_all_free"] is True, "terminal ownership flags")
    require(checks["kv_available_tokens"] == checks["kv_capacity_tokens"] == len(kv_free), "KV ownership count closure")
    require(checks["mamba_available_slots"] == checks["mamba_capacity_slots"] == len(mamba_free), "Mamba ownership count closure")


def replay_one(receipt_path: Path, terminal_path: Path, raw_path: Path, *,
               target_path: Path, server_path: Path, worker_path: Path,
               prereg_path: Path, manifest_path: Path,
               expected_manifest_sha256: str, static_dir: Path,
               expected_mode: str, expected_rank: int,
               expected_snapshot_id: str, expected_workload_id: str) -> dict[str, Any]:
    require(expected_mode in MODES, "external expected mode")
    require(0 <= expected_rank < len(EXPECTED_PAIRS), "external expected rank")
    require(
        expected_snapshot_id == f"{expected_mode}-rank-{expected_rank}",
        "external expected snapshot",
    )
    expected_pair = EXPECTED_PAIRS[expected_rank]
    require(
        expected_workload_id == f"{expected_pair[0]}-{expected_pair[1]}",
        "external expected workload",
    )
    receipt = _load(receipt_path, "forkaudit-hypic-retained-state-receipt-v2", "receipt")
    terminal = _load(terminal_path, "forkaudit-hypic-retained-state-terminal-v2", "terminal")
    raw = _load(raw_path, "forkaudit-hypic-retained-state-shard-v2", "raw")
    target = _load(target_path, "forkaudit-hypic-retained-state-target-v2", "target")
    server = _load(server_path, "hypic-rwd5-server-launch-receipt-v2", "server")
    worker = _load(worker_path, "forkaudit-hypic-scheduler-worker-v2", "worker")
    prereg = _load(prereg_path, "hypic-rwd5-retained-state-preregistration-v2", "prereg")
    require(receipt["official_commit"] == terminal["official_commit"] == raw["official_commit"] == OFFICIAL_COMMIT, "official commit closure")
    require(raw["status"] == "completed", "raw status")
    require(
        raw["mode"] == expected_mode
        and int(raw["rank"]) == expected_rank
        and target["mode"] == expected_mode
        and int(target["rank"]) == expected_rank
        and target["snapshot_id"] == expected_snapshot_id
        and target["workload_id"] == expected_workload_id
        and target["workload_binding"]["workload_id"] == expected_workload_id
        and target["workload_binding"]["dataset"] == expected_pair[0]
        and int(target["workload_binding"]["source_index"]) == expected_pair[1],
        "external frozen cell/location closure",
    )
    _validate_authority(receipt, terminal, raw, target, server, worker, prereg,
                        receipt_path=receipt_path, target_path=target_path,
                        server_path=server_path, worker_path=worker_path,
                        prereg_path=prereg_path, manifest_path=manifest_path,
                        expected_manifest_sha256=expected_manifest_sha256,
                        static_dir=static_dir)
    observed = receipt["target"]
    require(observed["mode"] == raw["mode"] == target["mode"] and observed["rank"] == raw["rank"] == target["rank"], "cell binding")
    require(observed["workload_id"] == raw["workload"]["workload_id"] == target["workload_id"], "workload binding")
    require(observed["document_token_sha256"] == raw["workload"]["document_token_sha256"] == target["document_token_sha256"], "document binding")
    require(token_sha256(target["document_token_ids"]) == target["document_token_sha256"], "target document hash")
    require(raw["workload"] == target["workload_binding"], "exact raw/target workload binding")
    selection = receipt["selection"]
    full_slots, mamba_slots, independently_expected_cached_tokens = _validate_selection(
        selection, raw["mode"], target, prereg
    )
    _validate_structure(receipt, prereg, full_slots, mamba_slots)
    payload_replay = replay_union(receipt["tensor_payload"]["records"])
    reported = receipt["tensor_payload"]["union"]
    require(payload_replay["record_count"] == reported["record_count"], "payload record count")
    require(payload_replay["naive_range_bytes"] == reported["naive_range_bytes"], "payload naive sum")
    require(payload_replay["unique_bytes"] == reported["unique_overlap_aware_bytes"], "payload unique sum")
    metadata_replay = replay_union(receipt["metadata"]["tensor_records"])
    _validate_metadata(receipt, raw["mode"])
    require(metadata_replay["unique_bytes"] == receipt["metadata"]["tensor_union"]["unique_overlap_aware_bytes"], "metadata unique sum")
    require(receipt["metadata"]["excluded_from_store_mib"] is True, "metadata denominator")
    require(
        int(raw["cache_observation"]["cached_tokens"]) == independently_expected_cached_tokens,
        "independently derived cache-hit coverage",
    )
    _validate_terminal(terminal, selection, receipt, receipt_path)
    require(raw["terminal_receipt"]["sha256"] == sha256_file(terminal_path), "raw terminal binding")
    return {
        "mode": raw["mode"], "rank": raw["rank"],
        "snapshot_id": target["snapshot_id"],
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


def replay_all(root: Path, output: Path, manifest_path: Path,
               expected_manifest_sha256: str) -> None:
    rows = []
    prereg = root / "static/preregistration.json"
    for mode in MODES:
        for rank, pair in enumerate(EXPECTED_PAIRS):
            snapshot_id = f"{mode}-rank-{rank}"
            row = replay_one(
                root / "store-receipts" / f"{snapshot_id}.json",
                root / "store-receipts" / f"{snapshot_id}.terminal.json",
                root / "raw" / f"{snapshot_id}.json",
                target_path=root / "targets" / f"{snapshot_id}.json",
                server_path=root / "server-receipts" / f"{snapshot_id}.json",
                worker_path=root / "scheduler-workers" / f"{snapshot_id}.json",
                prereg_path=prereg, manifest_path=manifest_path,
                expected_manifest_sha256=expected_manifest_sha256,
                static_dir=root / "static",
                expected_mode=mode,
                expected_rank=rank,
                expected_snapshot_id=snapshot_id,
                expected_workload_id=f"{pair[0]}-{pair[1]}",
            )
            require(
                row["mode"] == mode
                and int(row["rank"]) == rank
                and row["snapshot_id"] == snapshot_id
                and row["workload_id"] == f"{pair[0]}-{pair[1]}",
                "frozen row/file-position closure before append",
            )
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
    atomic_json(output, {
        "schema": "forkaudit-hypic-retained-state-blind-replay-v2",
        "passed": True, "official_commit": OFFICIAL_COMMIT,
        "freeze_manifest_sha256": expected_manifest_sha256,
        "denominator": "independently derived unique object-owned tensor ranges; metadata excluded",
        "rows": rows, "modes": summaries,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--expected-freeze-manifest-sha256", required=True)
    args = parser.parse_args()
    replay_all(args.root, args.output, args.freeze_manifest,
               args.expected_freeze_manifest_sha256)


if __name__ == "__main__":
    main()
