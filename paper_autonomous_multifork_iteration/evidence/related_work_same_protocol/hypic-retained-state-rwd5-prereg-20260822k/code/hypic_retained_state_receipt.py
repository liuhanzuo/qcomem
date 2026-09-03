#!/usr/bin/env python3
"""Fail-closed, object-owned physical-byte receipts for RW-D5.

The producer records live tensor metadata and cache ownership. The blind
replay derives every selected byte range again from immutable metadata and the
slot selection; producer byte offsets and totals are never authoritative.
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


SCHEMA = "forkaudit-hypic-retained-state-receipt-v2"
TARGET_SCHEMA = "forkaudit-hypic-retained-state-target-v2"
TERMINAL_SCHEMA = "forkaudit-hypic-retained-state-terminal-v2"
WORKER_SCHEMA = "forkaudit-hypic-scheduler-worker-v2"
SERVER_SCHEMA = "hypic-rwd5-server-launch-receipt-v2"
PREREG_SCHEMA = "hypic-rwd5-retained-state-preregistration-v2"
OFFICIAL_COMMIT = "98147c01909004e66d98bcb18b886927d41b0ee5"
TARGET_ENV = "FORKAUDIT_RWD5_TARGET_PATH"
OUTPUT_ENV = "FORKAUDIT_RWD5_RECEIPT_DIR"
WORKER_ENV = "FORKAUDIT_RWD5_WORKER_RECEIPT_PATH"
SERVER_ENV = "FORKAUDIT_RWD5_SERVER_RECEIPT_PATH"
PREREG_ENV = "FORKAUDIT_RWD5_PREREGISTRATION_PATH"
MANIFEST_ENV = "FORKAUDIT_RWD5_FREEZE_MANIFEST_PATH"
MANIFEST_SHA_ENV = "FORKAUDIT_RWD5_FREEZE_MANIFEST_SHA256"
FRONTEND_PID_ENV = "FORKAUDIT_RWD5_FRONTEND_PID"
MODE_ENV = "FORKAUDIT_RWD5_MODE"
RANK_ENV = "FORKAUDIT_RWD5_RANK"
DTYPE_DEBUG_ENV = "FORKAUDIT_RWD5_DTYPE_DEBUG_PATH"

PROCESS_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES", "PIC_SEAM_SINK", "PYTHONPATH",
    "PYTHONDONTWRITEBYTECODE", "SGLANG_NUMA_BIND_V2",
    "SGLANG_IS_FLASHINFER_AVAILABLE", "SGLANG_MAMBA_CONV_DTYPE",
    "SGLANG_MAMBA_SSM_DTYPE", TARGET_ENV, OUTPUT_ENV, WORKER_ENV,
    SERVER_ENV, PREREG_ENV, MANIFEST_ENV, MANIFEST_SHA_ENV, FRONTEND_PID_ENV,
    MODE_ENV, RANK_ENV, DTYPE_DEBUG_ENV,
)


class ReceiptError(RuntimeError):
    pass


class TargetNotReady(ReceiptError):
    """The exact preregistered target has not entered the cache yet."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptError(message)


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _process_canonical_sha256(value: Any) -> str:
    """Match RW-D5 runner/replay canonical JSON for process rows."""
    return _canonical_sha256(value)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _proc_row(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    _require(pid > 1 and root.is_dir(), f"live process {pid}")
    cmdline_raw = (root / "cmdline").read_bytes()
    cmdline = [part.decode("utf-8") for part in cmdline_raw.split(b"\0") if part]
    environment: dict[str, str] = {}
    for item in (root / "environ").read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        name = key.decode("utf-8")
        if name in PROCESS_ENV_KEYS:
            environment[name] = value.decode("utf-8")
    stat = (root / "stat").read_text()
    close = stat.rfind(")")
    _require(close > 0, "process stat")
    fields = stat[close + 2 :].split()
    return {
        "pid": pid,
        "ppid": int(fields[1]),
        "cmdline": cmdline,
        "cmdline_sha256": _process_canonical_sha256(cmdline),
        "environment": environment,
        "environment_sha256": _process_canonical_sha256(environment),
    }


def _process_identity(pid: int | None = None) -> dict[str, Any]:
    pid = os.getpid() if pid is None else int(pid)
    process = _proc_row(pid)
    ancestry = []
    seen = {pid}
    current = int(process["ppid"])
    while current > 1 and current not in seen:
        seen.add(current)
        row = _proc_row(current)
        ancestry.append({
            "pid": row["pid"], "ppid": row["ppid"],
            "cmdline_sha256": row["cmdline_sha256"],
        })
        current = int(row["ppid"])
    process["ancestry"] = ancestry
    process["ancestry_pids"] = [int(row["pid"]) for row in ancestry]
    return process


def _read_required_json(path: Path, schema: str, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} file")
    value = json.loads(path.read_text())
    _require(value.get("schema") == schema, f"{label} schema")
    return value


def _required_env_path(name: str) -> Path:
    value = os.environ.get(name)
    _require(bool(value), f"environment {name}")
    return Path(str(value))


def _storage_contract(prereg: dict[str, Any], mode: str) -> dict[str, Any]:
    contract = prereg["model"]["storage_contract"]
    _require(contract.get("schema") == "hypic-rwd5-model-storage-contract-v3", "storage contract")
    _require("dtype" not in contract, "legacy unified dtype forbidden")
    component_dtypes = contract.get("mamba_component_dtypes")
    _require(
        isinstance(component_dtypes, dict)
        and set(component_dtypes) == {"conv", "temporal", "transition", "conv_tails"},
        "exact component dtype contract",
    )
    allowed = {"torch.float32", "torch.float16", "torch.bfloat16"}
    _require(contract.get("kv_dtype") in allowed, "KV dtype contract")
    _require(all(value in allowed for value in component_dtypes.values()), "known component dtypes")
    _require(component_dtypes["conv_tails"] == component_dtypes["conv"], "conv tail dtype authority")
    _require(component_dtypes["transition"] == component_dtypes["temporal"], "transition dtype authority")
    live = contract.get("dtype_authority", {}).get("live_debug_validation")
    _require(
        isinstance(live, dict)
        and live.get("schema") == "hypic-rwd5-live-component-debug-binding-v1"
        and live.get("status") == "passed_debug_only_not_paper_evidence"
        and live.get("paper_evidence") is False
        and live.get("official_commit") == OFFICIAL_COMMIT
        and live.get("mirror_manifest_sha256")
        == "59530c0c8bc10cedbf4b0bde51d04e5490adeaf369e8738d9df363fc83941026"
        and live.get("formal_receipts_emitted") == 0
        and set(live.get("modes", {})) == {"prefix_cache", "transition_rope_recompute"},
        "live component dtype authority",
    )
    _require(
        set(live["modes"]["prefix_cache"]["components"]) == {"conv[0]", "temporal"}
        and set(live["modes"]["transition_rope_recompute"]["components"])
        == {"conv[0]", "temporal", "transition", "conv_tails[0]"},
        "live component topology authority",
    )
    _require(contract["enable_int8_mamba_checkpoint"] is False, "int8 checkpoint disabled")
    _require(contract["page_size"] == 1 and contract["kv_layout"] == "nhd", "layout contract")
    _require(mode in contract["mode_components"], "mode storage contract")
    _require(prereg["model"]["storage_contract_sha256"] == _canonical_sha256(contract), "storage contract hash")
    return contract


def emit_scheduler_worker_identity(scheduler) -> None:
    """Emit scheduler-child identity after pools/cache are constructed."""
    worker_path_value = os.environ.get(WORKER_ENV)
    if not worker_path_value:
        return
    worker_path = Path(worker_path_value)
    mode = os.environ.get(MODE_ENV)
    rank = int(os.environ.get(RANK_ENV, "-1"))
    _require(mode in {"prefix_cache", "transition_rope_recompute"}, "worker mode")
    _require(0 <= rank < 8, "worker rank")
    prereg_path = _required_env_path(PREREG_ENV)
    manifest_path = _required_env_path(MANIFEST_ENV)
    manifest_sha = os.environ.get(MANIFEST_SHA_ENV, "")
    prereg = _read_required_json(prereg_path, PREREG_SCHEMA, "preregistration")
    _require(_sha256_file(manifest_path) == manifest_sha, "worker external manifest")
    _storage_contract(prereg, str(mode))
    frontend_pid = int(os.environ.get(FRONTEND_PID_ENV, "-1"))
    process = _process_identity()
    _require(frontend_pid in process["ancestry_pids"], "scheduler/frontend lineage")
    tree_cache = scheduler.tree_cache
    req_pool = scheduler.req_to_token_pool
    _require(getattr(req_pool, "mamba_ckpt_pool", None) is None, "int8 checkpoint pool absent")
    if mode == "prefix_cache":
        _require(type(tree_cache).__name__ == "MambaRadixCache", "prefix worker cache")
        _require(getattr(tree_cache, "int8_ckpt_pool", None) is None, "prefix active mamba slots")
    else:
        _require(type(tree_cache).__name__ == "PICache", "HYPIC worker cache")
        _require(tree_cache.mamba_pool is req_pool.mamba_pool, "PICache MambaPool identity")
    payload = {
        "schema": WORKER_SCHEMA,
        "official_commit": OFFICIAL_COMMIT,
        "mode": mode,
        "rank": rank,
        "frontend_pid": frontend_pid,
        "process": process,
        "tree_cache_class": type(tree_cache).__name__,
        "picache_mamba_pool_identity": (
            tree_cache.mamba_pool is req_pool.mamba_pool
            if mode == "transition_rope_recompute" else None
        ),
        "int8_mamba_checkpoint_enabled": False,
        "preregistration_sha256": _sha256_file(prereg_path),
        "freeze_manifest_sha256": manifest_sha,
        "storage_contract_sha256": prereg["model"]["storage_contract_sha256"],
    }
    if worker_path.exists():
        _require(json.loads(worker_path.read_text()) == payload, "worker identity stability")
    else:
        _atomic_json(worker_path, payload)


def _load_target() -> tuple[dict[str, Any], Path, Path] | None:
    target_value = os.environ.get(TARGET_ENV)
    output_value = os.environ.get(OUTPUT_ENV)
    if not target_value or not output_value:
        return None
    path = Path(target_value)
    if not path.is_file():
        return None
    target = json.loads(path.read_text())
    _require(target.get("schema") == TARGET_SCHEMA, "target schema")
    snapshot_id = target.get("snapshot_id")
    _require(
        isinstance(snapshot_id, str) and snapshot_id
        and all(ch.isalnum() or ch in "-_" for ch in snapshot_id),
        "snapshot id",
    )
    _require(target.get("official_commit") == OFFICIAL_COMMIT, "target commit")
    _require(target.get("mode") in {"prefix_cache", "transition_rope_recompute"}, "mode")
    document = target.get("document_token_ids")
    _require(isinstance(document, list) and document, "document token ids")
    _require(token_sha256(document) == target.get("document_token_sha256"), "document hash")
    _require(isinstance(target.get("authority"), dict), "target authority")
    binding = target.get("workload_binding")
    _require(
        isinstance(binding, dict)
        and binding.get("workload_id") == target.get("workload_id")
        and binding.get("document_token_sha256") == target.get("document_token_sha256")
        and binding.get("document_tokens") == len(document)
        and binding.get("token_identity_verified") is True,
        "target workload binding",
    )
    return target, Path(output_value), path


def _bound_authority(target: dict[str, Any], target_path: Path) -> dict[str, Any]:
    prereg_path = _required_env_path(PREREG_ENV)
    server_path = _required_env_path(SERVER_ENV)
    worker_path = _required_env_path(WORKER_ENV)
    manifest_path = _required_env_path(MANIFEST_ENV)
    prereg = _read_required_json(prereg_path, PREREG_SCHEMA, "preregistration")
    server = _read_required_json(server_path, SERVER_SCHEMA, "server receipt")
    worker = _read_required_json(worker_path, WORKER_SCHEMA, "scheduler worker")
    expected = {
        "official_commit": OFFICIAL_COMMIT,
        "target_sha256": _sha256_file(target_path),
        "preregistration_sha256": _sha256_file(prereg_path),
        "server_launch_receipt_sha256": _sha256_file(server_path),
        "scheduler_worker_receipt_sha256": _sha256_file(worker_path),
        "freeze_manifest_sha256": _sha256_file(manifest_path),
        "official_source_ledger_sha256": prereg["official_source_ledger_sha256"],
        "environment_ledger_sha256": prereg["environment_ledger_sha256"],
        "data_sha256": prereg["data"]["sha256"],
        "model_weight_ledger_sha256": prereg["model"]["weight_ledger_raw_sha256"],
        "model_artifact_ledger_sha256": prereg["model"]["artifact_ledger_raw_sha256"],
        "model_config_sha256": prereg["model"]["config_sha256"],
        "storage_contract_sha256": prereg["model"]["storage_contract_sha256"],
        "overlay_diff_sha256": prereg["instrumentation"]["overlay"]["canonical_diff_sha256"],
        "code_sha256": {key: row["sha256"] for key, row in sorted(prereg["code"].items())},
        "server_configuration_sha256": server["server_configuration_sha256"],
    }
    # target_sha256 cannot be embedded in the target without a hash cycle.
    target_authority = dict(expected)
    target_authority.pop("target_sha256")
    _require(target["authority"] == target_authority, "target complete authority binding")
    _require(os.environ.get(MANIFEST_SHA_ENV) == expected["freeze_manifest_sha256"], "external manifest identity")
    _require(server["scheduler_worker"]["receipt_sha256"] == expected["scheduler_worker_receipt_sha256"], "server/worker binding")
    _require(server["authority"] == {key: expected[key] for key in server["authority"]}, "server authority closure")
    process = _process_identity()
    _require(process["pid"] == worker["process"]["pid"], "scheduler worker PID")
    _require(process["ppid"] == worker["process"]["ppid"], "scheduler worker PPID")
    _require(process["cmdline_sha256"] == worker["process"]["cmdline_sha256"], "scheduler cmdline")
    _require(process["environment"] == worker["process"]["environment"], "scheduler environment")
    _require(int(server["frontend_process"]["pid"]) in process["ancestry_pids"], "server scheduler lineage")
    return {
        "bindings": expected,
        "scheduler_process": process,
        "storage_contract": _storage_contract(prereg, target["mode"]),
    }


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


def _record_range(tensor: torch.Tensor, *, tensor_name: str, component: str,
                  start: int, end: int, selection: dict[str, Any]) -> dict[str, Any]:
    info = _tensor_storage(tensor)
    _require(0 <= start < end <= info["storage_nbytes"], f"range outside {tensor_name}")
    return {
        **info, "tensor_name": tensor_name, "component": component,
        "byte_start": int(start), "byte_end": int(end),
        "absolute_byte_start": int(info["storage_data_ptr"] + start),
        "absolute_byte_end": int(info["storage_data_ptr"] + end),
        "range_bytes": int(end - start), "selection": selection,
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


def _is_c_contiguous(shape: list[int], stride: list[int]) -> bool:
    expected = 1
    for size, observed in zip(reversed(shape), reversed(stride)):
        if size != 1 and observed != expected:
            return False
        expected *= size
    return True


def _kv_payload_ranges(tree_cache, slots: list[int], contract: dict[str, Any]) -> list[dict[str, Any]]:
    allocator = tree_cache.token_to_kv_pool_allocator
    _require(int(allocator.page_size) == 1, "KV page size one")
    full = getattr(allocator.get_kvcache(), "full_kv_pool", allocator.get_kvcache())
    _require(getattr(full, "kv_cache_layout", "nhd") == contract["kv_layout"], "KV layout")
    keys, values = list(full.k_buffer), list(full.v_buffer)
    expected_layers = int(contract["full_attention_layer_count"])
    _require(len(keys) == len(values) == expected_layers and expected_layers > 0, "exact K/V layer count")
    records: list[dict[str, Any]] = []
    expected_capacity = int(allocator.size) + 1
    for kind, tensors in (("key", keys), ("value", values)):
        for layer, tensor in enumerate(tensors):
            shape = [int(value) for value in tensor.shape]
            stride = [int(value) for value in tensor.stride()]
            _require(tensor.ndim == 3 and _is_c_contiguous(shape, stride), "NHD KV tensor layout")
            _require(shape[0] == expected_capacity, "KV slot-axis capacity")
            _require(str(tensor.dtype) == contract["kv_dtype"], "KV dtype")
            if kind == "value":
                _require(shape == [int(v) for v in keys[layer].shape], "K/V shape equality")
            element = int(tensor.element_size())
            base = int(tensor.storage_offset()) * element
            for first, after in _contiguous_runs(slots):
                _require(first > 0 and after <= shape[0], "KV slot bounds")
                records.append(_record_range(
                    tensor, tensor_name=f"full_kv.{kind}[{layer}]",
                    component=f"full_attention_{kind}",
                    start=base + first * stride[0] * element,
                    end=base + after * stride[0] * element,
                    selection={"kind": "axis0_slots", "slot_start": first,
                               "slot_end_exclusive": after},
                ))
    _require(records, "empty KV payload ledger")
    return records


def _mamba_payload_ranges(tree_cache, slots: list[int], contract: dict[str, Any],
                           mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    req_pool = tree_cache.req_to_token_pool
    _require(getattr(req_pool, "mamba_ckpt_pool", None) is None, "int8 checkpoint pool absent")
    if mode == "prefix_cache":
        _require(getattr(tree_cache, "int8_ckpt_pool", None) is None, "prefix MambaPool slots")
    else:
        _require(tree_cache.mamba_pool is req_pool.mamba_pool, "PICache MambaPool identity")
    cache = req_pool.mamba_pool.mamba_cache
    expected = contract["mode_components"][mode]
    conv = list(cache.conv)
    tails = None if cache.conv_tails is None else list(cache.conv_tails)
    _require(len(conv) == int(contract["conv_tensor_count"]), "exact conv tensor count")
    _require(isinstance(cache.temporal, torch.Tensor), "temporal tensor")
    if mode == "prefix_cache":
        _require(cache.transition is None and cache.conv_tails is None, "Prefix PIC components absent")
    else:
        _require(isinstance(cache.transition, torch.Tensor), "HYPIC transition required")
        _require(tails is not None and tails and len(tails) == int(expected["conv_tails_tensor_count"]), "HYPIC conv_tails required")
    fields: list[tuple[str, torch.Tensor]] = [(f"conv[{i}]", value) for i, value in enumerate(conv)]
    fields.append(("temporal", cache.temporal))
    if cache.transition is not None:
        fields.append(("transition", cache.transition))
    if tails is not None:
        fields.extend((f"conv_tails[{i}]", value) for i, value in enumerate(tails))
    expected_names = [f"conv[{i}]" for i in range(int(contract["conv_tensor_count"]))] + ["temporal"]
    if mode == "transition_rope_recompute":
        expected_names += ["transition"] + [
            f"conv_tails[{i}]" for i in range(int(expected["conv_tails_tensor_count"]))
        ]
    _require([name for name, _ in fields] == expected_names, "exact recurrent component keys")
    records: list[dict[str, Any]] = []
    presence: dict[str, Any] = {}
    expected_layers = int(contract["recurrent_layer_count"])
    expected_capacity = int(req_pool.mamba_allocator.size) + 1
    component_dtypes = contract["mamba_component_dtypes"]
    for name, tensor in fields:
        shape = [int(value) for value in tensor.shape]
        stride = [int(value) for value in tensor.stride()]
        _require(tensor.ndim >= 3 and _is_c_contiguous(shape, stride), f"{name} layout")
        _require(shape[0] == expected_layers and shape[1] == expected_capacity, f"{name} axes")
        component = "conv_tails" if name.startswith("conv_tails[") else (
            "conv" if name.startswith("conv[") else name
        )
        _require(str(tensor.dtype) == component_dtypes[component], f"{name} dtype")
        element = int(tensor.element_size())
        presence[name] = {
            "present": True,
            "shape": shape,
            "dtype": str(tensor.dtype),
            "element_size": element,
        }
        base = int(tensor.storage_offset()) * element
        for layer in range(expected_layers):
            for first, after in _contiguous_runs(slots):
                _require(first > 0 and after <= shape[1], f"{name} slot bounds")
                records.append(_record_range(
                    tensor, tensor_name=f"mamba.{name}",
                    component=name.split("[")[0],
                    start=base + layer * stride[0] * element + first * stride[1] * element,
                    end=base + layer * stride[0] * element + after * stride[1] * element,
                    selection={"kind": "axis1_slots_at_layer",
                               "mamba_layer_index": layer,
                               "slot_start": first, "slot_end_exclusive": after},
                ))
    _require(records, "empty recurrent-state payload ledger")
    return records, presence


def _component_dtype_debug_inventory(tree_cache, mode: str) -> dict[str, Any]:
    """Read-only live pool topology for debug; never a formal byte receipt."""
    _require(mode in {"prefix_cache", "transition_rope_recompute"}, "debug mode")
    cache = tree_cache.req_to_token_pool.mamba_pool.mamba_cache
    conv = list(cache.conv)
    tails = None if cache.conv_tails is None else list(cache.conv_tails)
    if mode == "prefix_cache":
        _require(cache.transition is None and tails is None, "debug Prefix topology")
    else:
        _require(isinstance(cache.transition, torch.Tensor), "debug HYPIC transition")
        _require(tails is not None and tails, "debug HYPIC conv tails")
    fields: list[tuple[str, torch.Tensor]] = [
        (f"conv[{index}]", tensor) for index, tensor in enumerate(conv)
    ]
    fields.append(("temporal", cache.temporal))
    if cache.transition is not None:
        fields.append(("transition", cache.transition))
    if tails is not None:
        fields.extend((f"conv_tails[{index}]", tensor) for index, tensor in enumerate(tails))
    components = {}
    for name, tensor in fields:
        _require(isinstance(tensor, torch.Tensor), f"debug tensor {name}")
        components[name] = {
            "dtype": str(tensor.dtype),
            "element_size": int(tensor.element_size()),
            "shape": [int(value) for value in tensor.shape],
            "stride": [int(value) for value in tensor.stride()],
            "device": str(tensor.device),
            "c_contiguous": bool(tensor.is_contiguous()),
        }
    return {
        "schema": "hypic-rwd5-component-dtype-debug-v1",
        "status": "debug_only_not_formal_evidence",
        "official_commit": OFFICIAL_COMMIT,
        "mode": mode,
        "tree_cache_class": type(tree_cache).__name__,
        "mamba_pool_class": type(tree_cache.req_to_token_pool.mamba_pool).__name__,
        "mamba_allocator_size": int(tree_cache.req_to_token_pool.mamba_allocator.size),
        "mamba_capacity_axis": int(tree_cache.req_to_token_pool.mamba_allocator.size) + 1,
        "runtime_environment": {
            "SGLANG_MAMBA_CONV_DTYPE": os.environ.get("SGLANG_MAMBA_CONV_DTYPE"),
            "SGLANG_MAMBA_SSM_DTYPE": os.environ.get("SGLANG_MAMBA_SSM_DTYPE"),
        },
        "components": components,
        "formal_receipt_emitted": False,
    }


def _full_tensor_metadata(tensor: torch.Tensor, name: str) -> dict[str, Any]:
    shape = [int(value) for value in tensor.shape]
    stride = [int(value) for value in tensor.stride()]
    _require(_is_c_contiguous(shape, stride), f"metadata tensor not contiguous: {name}")
    start = int(tensor.storage_offset()) * int(tensor.element_size())
    end = start + int(tensor.numel()) * int(tensor.element_size())
    return _record_range(tensor, tensor_name=name, component="cache_index_metadata",
                         start=start, end=end, selection={"kind": "whole_tensor"})


def _union_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_storage: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    for record in records:
        key = (str(record["device"]), int(record["storage_data_ptr"]),
               int(record["storage_nbytes"]))
        start, end = int(record["byte_start"]), int(record["byte_end"])
        _require(0 <= start < end <= key[2], "range outside backing storage")
        by_storage.setdefault(key, []).append((start, end))
    total = 0
    storage_rows = []
    for key, intervals in sorted(by_storage.items()):
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        unique = sum(end - start for start, end in merged)
        total += unique
        storage_rows.append({"device": key[0], "storage_data_ptr": key[1],
                             "storage_nbytes": key[2], "merged_byte_ranges": merged,
                             "unique_bytes": unique})
    return {"record_count": len(records), "storage_count": len(storage_rows),
            "naive_range_bytes": sum(int(row["range_bytes"]) for row in records),
            "unique_overlap_aware_bytes": total, "storages": storage_rows}


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
    entries, full_slots, mamba_slots, metadata_records = [], [], [], []
    metadata_exact_bytes = 0
    for node in path_nodes:
        _require(int(node.full_lock_ref) == 0 and int(node.mamba_lock_ref) == 0, "stable Prefix lock refs")
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
        entries.append({
            "node_id": int(node.id), "token_ids": key,
            "token_sha256": token_sha256(key), "token_count": len(key),
            "full_kv_slots": slots, "mamba_state_slots": node_mamba,
            "lock_refs": {"full": int(node.full_lock_ref), "mamba": int(node.mamba_lock_ref)},
            "exact_key_array_bytes": key_bytes,
            "python_node_shallow_bytes": int(sys.getsizeof(node)),
        })
    _require(reconstructed == cached_tokens, "prefix path reconstruction")
    _require(len(full_slots) == len(cached_tokens), "prefix slot coverage")
    _require(len(set(full_slots)) == len(full_slots), "prefix KV slot alias")
    _require(mamba_slots and len(set(mamba_slots)) == len(mamba_slots), "prefix Mamba slots")
    return {
        "cache_kind": "MambaRadixCache", "entries": entries,
        "owned_document_token_ids": cached_tokens,
        "owned_document_token_sha256": token_sha256(cached_tokens),
        "owned_document_tokens": len(cached_tokens),
        "expected_measured_cached_tokens": len(cached_tokens),
        "full_kv_slots": full_slots, "mamba_state_slots": mamba_slots,
        "metadata_tensor_records": metadata_records,
        "metadata_exact_non_tensor_bytes": metadata_exact_bytes,
    }


def _hypic_selection(tree_cache, target: dict[str, Any]) -> dict[str, Any]:
    segments = target.get("segment_token_ids")
    _require(isinstance(segments, list) and len(segments) == 2, "HYPIC segment target")
    entries, full_slots, mamba_slots, metadata_records, reconstructed = [], [], [], [], []
    metadata_exact_bytes = 0
    for index, raw in enumerate(segments):
        token_ids = [int(value) for value in raw]
        seg_hash = bytes.fromhex(segment_hash_hex(token_ids))
        entry = tree_cache._entries.get(seg_hash)
        if entry is None:
            raise TargetNotReady(f"target HYPIC segment absent yet: {index}")
        _require(int(entry.lock_ref) == 0, "stable HYPIC lock ref")
        observed = [int(value) for value in entry.token_ids.detach().cpu().tolist()]
        _require(observed == token_ids, f"target HYPIC segment token drift: {index}")
        slots = [int(value) for value in entry.full_kv_slots.detach().cpu().tolist()]
        _require(len(slots) == len(token_ids), "HYPIC KV slot coverage")
        mamba_slot = int(entry.mamba_state_slot)
        _require(mamba_slot > 0, "reserved HYPIC mamba slot")
        reconstructed.extend(token_ids); full_slots.extend(slots); mamba_slots.append(mamba_slot)
        metadata_records.extend([
            _full_tensor_metadata(entry.full_kv_slots, f"hypic.segment[{index}].full_kv_slots"),
            _full_tensor_metadata(entry.token_ids, f"hypic.segment[{index}].token_ids"),
        ])
        metadata_exact_bytes += len(entry.seg_hash)
        entries.append({
            "segment_index": index, "segment_hash_hex": entry.seg_hash.hex(),
            "token_ids": token_ids, "token_sha256": token_sha256(token_ids),
            "token_count": len(token_ids), "full_kv_slots": slots,
            "mamba_state_slot": mamba_slot, "lock_ref": int(entry.lock_ref),
            "exact_segment_hash_bytes": len(entry.seg_hash),
            "python_entry_shallow_bytes": int(sys.getsizeof(entry)),
        })
    _require(reconstructed == target["document_token_ids"], "HYPIC document reconstruction")
    _require(len(set(full_slots)) == len(full_slots), "HYPIC KV slot alias")
    _require(len(set(mamba_slots)) == len(mamba_slots), "HYPIC Mamba slot alias")
    seam = int(target.get("seam_tokens", 8))
    return {
        "cache_kind": "PICache", "entries": entries,
        "owned_document_token_ids": reconstructed,
        "owned_document_token_sha256": token_sha256(reconstructed),
        "owned_document_tokens": len(reconstructed),
        "expected_measured_cached_tokens": len(reconstructed) - seam,
        "full_kv_slots": full_slots, "mamba_state_slots": mamba_slots,
        "metadata_tensor_records": metadata_records,
        "metadata_exact_non_tensor_bytes": metadata_exact_bytes,
    }


def _selection(tree_cache, target: dict[str, Any]) -> dict[str, Any]:
    if target["mode"] == "prefix_cache":
        _require(type(tree_cache).__name__ == "MambaRadixCache", "prefix cache type")
        return _prefix_selection(tree_cache, target)
    _require(type(tree_cache).__name__ == "PICache", "HYPIC cache type")
    return _hypic_selection(tree_cache, target)


def _kv_pre_free_snapshot(allocator) -> dict[str, Any]:
    _require(int(allocator.page_size) == 1, "pre-snapshot page size one")
    size = int(allocator.size)
    expected = set(range(1, size + 1))
    free = sorted(int(value) for value in allocator.free_pages.detach().cpu().tolist())
    release = sorted(int(value) for value in allocator.release_pages.detach().cpu().tolist())
    combined = free + release
    _require(len(combined) == len(set(combined)), "duplicate pre-snapshot KV free entry")
    _require(set(combined).issubset(expected), "pre-snapshot KV free subset domain")
    canonical_free = sorted(combined)
    _require(int(allocator.available_size()) == len(canonical_free), "pre-snapshot KV available count")
    return {
        "page_size": 1,
        "size": size,
        "free_pages": free,
        "release_pages": release,
        "canonical_free_domain": canonical_free,
        "canonical_allocated_domain": sorted(expected - set(canonical_free)),
    }


def _mamba_pre_free_snapshot(allocator) -> dict[str, Any]:
    size = int(allocator.size)
    expected = set(range(1, size + 1))
    free = sorted(int(value) for value in allocator.free_slots.detach().cpu().tolist())
    _require(len(free) == len(set(free)), "duplicate pre-snapshot Mamba free entry")
    _require(set(free).issubset(expected), "pre-snapshot Mamba free subset domain")
    _require(int(allocator.available_size()) == len(free), "pre-snapshot Mamba available count")
    return {
        "size": size,
        "free_slots": free,
        "canonical_free_domain": free,
        "canonical_allocated_domain": sorted(expected - set(free)),
    }


def maybe_emit_owned_state_snapshot(tree_cache) -> None:
    loaded = _load_target()
    if loaded is None:
        return
    target, output_dir, target_path = loaded
    output = output_dir / f"{target['snapshot_id']}.json"
    if output.exists():
        return
    try:
        selection = _selection(tree_cache, target)
    except TargetNotReady:
        return
    debug_path = os.environ.get(DTYPE_DEBUG_ENV)
    if debug_path:
        output = Path(debug_path)
        if not output.exists():
            _atomic_json(output, _component_dtype_debug_inventory(tree_cache, target["mode"]))
        return
    authority = _bound_authority(target, target_path)
    contract = authority.pop("storage_contract")
    kv_pre_free = _kv_pre_free_snapshot(tree_cache.token_to_kv_pool_allocator)
    mamba_pre_free = _mamba_pre_free_snapshot(tree_cache.req_to_token_pool.mamba_allocator)
    _require(
        set(selection["full_kv_slots"]).issubset(kv_pre_free["canonical_allocated_domain"]),
        "selected KV slots allocated at snapshot",
    )
    _require(
        set(selection["mamba_state_slots"]).issubset(mamba_pre_free["canonical_allocated_domain"]),
        "selected Mamba slots allocated at snapshot",
    )
    payload_records = _kv_payload_ranges(tree_cache, selection["full_kv_slots"], contract)
    mamba_records, presence = _mamba_payload_ranges(
        tree_cache, selection["mamba_state_slots"], contract, target["mode"]
    )
    payload_records.extend(mamba_records)
    metadata_records = selection.pop("metadata_tensor_records")
    _atomic_json(output, {
        "schema": SCHEMA, "status": "owned_state_snapshot_complete",
        "official_commit": OFFICIAL_COMMIT, "authority": authority,
        "target": {"snapshot_id": target["snapshot_id"], "mode": target["mode"],
                   "rank": int(target["rank"]), "workload_id": target["workload_id"],
                   "document_token_sha256": target["document_token_sha256"],
                   "document_tokens": len(target["document_token_ids"]),
                   "seam_tokens": int(target.get("seam_tokens", 0))},
        "storage_contract": contract, "selection": selection,
        "tensor_payload": {"records": payload_records,
                           "union": _union_summary(payload_records),
                           "denominator": "blindly rederived unique backing-storage byte ranges"},
        "metadata": {"tensor_records": metadata_records,
                     "tensor_union": _union_summary(metadata_records),
                     "exact_non_tensor_bytes": int(selection["metadata_exact_non_tensor_bytes"]),
                     "excluded_from_store_mib": True,
                     "python_allocator_overhead": "not attributed; shallow object sizes retained per entry"},
        "component_presence": presence,
        "allocator_observation": {
            "kv_available_tokens": int(tree_cache.token_to_kv_pool_allocator.available_size()),
            "kv_capacity_tokens": int(tree_cache.token_to_kv_pool_allocator.size),
            "kv_page_size": int(tree_cache.token_to_kv_pool_allocator.page_size),
            "mamba_available_slots": int(tree_cache.req_to_token_pool.mamba_allocator.available_size()),
            "mamba_capacity_slots": int(tree_cache.req_to_token_pool.mamba_allocator.size),
            "pre_free_ownership": {"kv": kv_pre_free, "mamba": mamba_pre_free}},
        "forbidden_denominators": ["NVML", "process_allocation", "pool_capacity_delta"],
    })


def _kv_free_snapshot(allocator) -> dict[str, Any]:
    _require(int(allocator.page_size) == 1, "terminal page size one")
    free = [int(v) for v in allocator.free_pages.detach().cpu().tolist()]
    release = [int(v) for v in allocator.release_pages.detach().cpu().tolist()]
    combined = free + release
    expected = list(range(1, int(allocator.size) + 1))
    _require(len(combined) == len(set(combined)), "duplicate KV free-list entry")
    _require(sorted(combined) == expected, "exact KV free-list domain")
    _require(not release, "terminal KV release list empty")
    return {"page_size": 1, "size": int(allocator.size), "free_pages": free,
            "release_pages": release, "exact_domain": expected}


def _mamba_free_snapshot(allocator) -> dict[str, Any]:
    free = [int(v) for v in allocator.free_slots.detach().cpu().tolist()]
    expected = list(range(1, int(allocator.size) + 1))
    _require(len(free) == len(set(free)), "duplicate Mamba free-list entry")
    _require(sorted(free) == expected, "exact Mamba free-list domain")
    return {"size": int(allocator.size), "free_slots": free, "exact_domain": expected}


def maybe_emit_terminal_ownership_snapshot(scheduler) -> None:
    loaded = _load_target()
    if loaded is None:
        return
    target, output_dir, target_path = loaded
    main_path = output_dir / f"{target['snapshot_id']}.json"
    terminal_path = output_dir / f"{target['snapshot_id']}.terminal.json"
    if terminal_path.exists() or not main_path.is_file():
        return
    prior = json.loads(main_path.read_text())
    _require(prior.get("schema") == SCHEMA, "prior receipt schema")
    old_kv = [int(value) for value in prior["selection"]["full_kv_slots"]]
    old_mamba = [int(value) for value in prior["selection"]["mamba_state_slots"]]
    pre = prior["allocator_observation"]["pre_free_ownership"]
    old_kv_preallocated = set(old_kv).isdisjoint(pre["kv"]["canonical_free_domain"])
    old_mamba_preallocated = set(old_mamba).isdisjoint(pre["mamba"]["canonical_free_domain"])
    kv = _kv_free_snapshot(scheduler.token_to_kv_pool_allocator)
    mamba = _mamba_free_snapshot(scheduler.req_to_token_pool.mamba_allocator)
    if target["mode"] == "transition_rope_recompute":
        target_entries_after = sum(
            1 for segment in target["segment_token_ids"]
            if bytes.fromhex(segment_hash_hex(segment)) in scheduler.tree_cache._entries
        )
        all_cache_entries_after = len(scheduler.tree_cache._entries)
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
        all_cache_entries_after = len(scheduler.tree_cache.root_node.children)
    checks = {
        "target_entries_after": target_entries_after,
        "all_cache_entries_after": all_cache_entries_after,
        "old_kv_slots_all_free": all(slot in kv["free_pages"] for slot in old_kv),
        "old_mamba_slots_all_free": all(slot in mamba["free_slots"] for slot in old_mamba),
        "old_kv_slots_preallocated": old_kv_preallocated,
        "old_mamba_slots_preallocated": old_mamba_preallocated,
        "kv_available_tokens": int(scheduler.token_to_kv_pool_allocator.available_size()),
        "kv_capacity_tokens": int(scheduler.token_to_kv_pool_allocator.size),
        "mamba_available_slots": int(scheduler.req_to_token_pool.mamba_allocator.available_size()),
        "mamba_capacity_slots": int(scheduler.req_to_token_pool.mamba_allocator.size),
        "kv_free_list": kv, "mamba_free_list": mamba,
    }
    passed = (
        target_entries_after == 0 and all_cache_entries_after == 0
        and checks["old_kv_slots_preallocated"] and checks["old_mamba_slots_preallocated"]
        and checks["old_kv_slots_all_free"] and checks["old_mamba_slots_all_free"]
        and checks["kv_available_tokens"] == checks["kv_capacity_tokens"]
        and checks["mamba_available_slots"] == checks["mamba_capacity_slots"]
    )
    _require(passed, f"terminal ownership removal failed: {checks}")
    authority = _bound_authority(target, target_path)
    authority.pop("storage_contract")
    _require(authority == prior["authority"], "terminal/prior authority")
    _atomic_json(terminal_path, {
        "schema": TERMINAL_SCHEMA, "status": "terminal_ownership_removal_complete",
        "official_commit": OFFICIAL_COMMIT, "snapshot_id": target["snapshot_id"],
        "passed": True, "checks": checks, "authority": authority,
        "prior_receipt_sha256": _sha256_file(main_path),
    })
