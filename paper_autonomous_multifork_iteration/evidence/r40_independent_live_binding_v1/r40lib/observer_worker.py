from __future__ import annotations

import hashlib
import os
from itertools import combinations
from math import prod
from typing import Any, Mapping

import torch

from .protocol import (
    OBSERVATION_SCHEMA,
    challenge_nonce,
    require,
    seal_payload,
    sha256_json,
    validate_live_items,
)


def _observer_payload_bytes(tensor: torch.Tensor) -> bytes:
    contiguous = tensor.detach().contiguous().view(torch.uint8).cpu()
    return contiguous.numpy().tobytes()


def _observer_interval(tensor: torch.Tensor) -> tuple[int, int]:
    dimensions = [int(value) for value in tensor.shape]
    steps = [int(value) for value in tensor.stride()]
    low = high = int(tensor.storage_offset())
    for dimension, step in zip(dimensions, steps):
        endpoint = (dimension - 1) * step
        low += endpoint if endpoint < 0 else 0
        high += endpoint if endpoint > 0 else 0
    width = int(tensor.element_size())
    return low * width, (high + 1) * width


def _observer_relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    same_storage = (
        left["_device"] == right["_device"]
        and left["_storage_pointer"] == right["_storage_pointer"]
        and left["storage_nbytes"] == right["storage_nbytes"]
    )
    if not same_storage:
        return "disjoint"
    low = max(int(left["byte_start"]), int(right["byte_start"]))
    high = min(int(left["byte_end_exclusive"]), int(right["byte_end_exclusive"]))
    if low >= high:
        return "disjoint"
    exact = all(
        left[field] == right[field]
        for field in ("byte_start", "byte_end_exclusive", "shape", "stride")
    )
    return "exact_alias" if exact else "partial_overlap"


def observe_candidate(request: Mapping[str, Any]) -> dict[str, Any]:
    manifest = request["manifest"]
    items = request["items"]
    seed = str(request["challenge_seed_sha256"])
    validate_live_items(items, manifest)
    internal: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: value["slot_id"]):
        tensor = item["tensor"]
        require(type(tensor) is torch.Tensor, "observer did not receive a base Tensor")
        require(tensor.is_floating_point() and tensor.numel() > 0, "observer tensor drift")
        require(tensor.device.type == "cpu", "observer device drift")
        slot = str(item["slot_id"])
        payload = _observer_payload_bytes(tensor)
        nonce = challenge_nonce(seed, slot)
        start, end = _observer_interval(tensor)
        storage = tensor.untyped_storage()
        internal.append(
            {
                "slot_id": slot,
                "shape": list(map(int, tensor.size())),
                "stride": list(map(int, tensor.stride())),
                "storage_offset": int(tensor.storage_offset()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "tensor_nbytes": int(prod(list(tensor.size()))) * int(tensor.element_size()),
                "storage_nbytes": int(storage.nbytes()),
                "byte_start": start,
                "byte_end_exclusive": end,
                "challenge_response_sha256": hashlib.sha256(
                    b"r40-live-binding-challenge-v1\0" + nonce + b"\0" + payload
                ).hexdigest(),
                "_device": str(tensor.device),
                "_storage_pointer": int(storage.data_ptr()),
            }
        )
    relations = [
        [left["slot_id"], right["slot_id"], _observer_relation(left, right)]
        for left, right in combinations(internal, 2)
    ]
    rows = [
        {
            key: value
            for key, value in row.items()
            if key not in {"_device", "_storage_pointer"}
        }
        for row in internal
    ]
    result = {
        "schema_version": OBSERVATION_SCHEMA,
        "role": "post_binding_observer",
        "worker_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "source_implementation": "r40lib.observer_worker",
        "manifest_sha256": manifest["manifest_sha256"],
        "live_item_fields_received": ["slot_id", "tensor"],
        "row_count": len(rows),
        "relation_count": len(relations),
        "rows": rows,
        "rows_sha256": sha256_json(rows),
        "relations": relations,
        "relations_sha256": sha256_json(relations),
        "raw_addresses_serialized": False,
        "payload_sha256": None,
    }
    return seal_payload(result, "payload_sha256")


def worker_main(request_queue: Any, response_queue: Any) -> None:
    try:
        response_queue.put({"ok": True, "result": observe_candidate(request_queue.get())})
    except BaseException as exc:
        response_queue.put(
            {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        )


__all__ = ["observe_candidate", "worker_main"]

