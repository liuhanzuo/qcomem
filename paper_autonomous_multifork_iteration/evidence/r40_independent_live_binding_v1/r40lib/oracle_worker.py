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


def _payload_bytes(tensor: torch.Tensor) -> bytes:
    array = tensor.detach().cpu().contiguous().numpy()
    return array.tobytes(order="C")


def _interval(tensor: torch.Tensor) -> tuple[int, int]:
    shape = tuple(int(value) for value in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    minimum = maximum = int(tensor.storage_offset())
    for size, step in zip(shape, stride):
        displacement = (size - 1) * step
        minimum += min(0, displacement)
        maximum += max(0, displacement)
    element_size = int(tensor.element_size())
    return minimum * element_size, (maximum + 1) * element_size


def _relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if left["_storage_key"] != right["_storage_key"]:
        return "disjoint"
    overlap = (
        int(left["byte_start"]) < int(right["byte_end_exclusive"])
        and int(right["byte_start"]) < int(left["byte_end_exclusive"])
    )
    if not overlap:
        return "disjoint"
    if (
        left["byte_start"] == right["byte_start"]
        and left["byte_end_exclusive"] == right["byte_end_exclusive"]
        and left["shape"] == right["shape"]
        and left["stride"] == right["stride"]
    ):
        return "exact_alias"
    return "partial_overlap"


def observe_oracle(request: Mapping[str, Any]) -> dict[str, Any]:
    manifest = request["manifest"]
    items = request["items"]
    seed = str(request["challenge_seed_sha256"])
    validate_live_items(items, manifest)
    rows_internal: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda value: value["slot_id"]):
        tensor = item["tensor"]
        require(
            isinstance(tensor, torch.Tensor)
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and tensor.numel() > 0,
            "oracle tensor contract drift",
        )
        payload = _payload_bytes(tensor)
        slot = str(item["slot_id"])
        nonce = challenge_nonce(seed, slot)
        start, end = _interval(tensor)
        storage = tensor.untyped_storage()
        rows_internal.append(
            {
                "slot_id": slot,
                "shape": [int(value) for value in tensor.shape],
                "stride": [int(value) for value in tensor.stride()],
                "storage_offset": int(tensor.storage_offset()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "tensor_nbytes": int(prod(tensor.shape)) * int(tensor.element_size()),
                "storage_nbytes": int(storage.nbytes()),
                "byte_start": start,
                "byte_end_exclusive": end,
                "challenge_response_sha256": hashlib.sha256(
                    b"r40-live-binding-challenge-v1\0" + nonce + b"\0" + payload
                ).hexdigest(),
                "_storage_key": (
                    str(tensor.device),
                    int(storage.data_ptr()),
                    int(storage.nbytes()),
                ),
            }
        )
    relations = [
        [left["slot_id"], right["slot_id"], _relation(left, right)]
        for left, right in combinations(rows_internal, 2)
    ]
    rows = [
        {key: value for key, value in row.items() if key != "_storage_key"}
        for row in rows_internal
    ]
    result = {
        "schema_version": OBSERVATION_SCHEMA,
        "role": "pre_injection_oracle",
        "worker_pid": os.getpid(),
        "parent_pid": os.getppid(),
        "source_implementation": "r40lib.oracle_worker",
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
        response_queue.put({"ok": True, "result": observe_oracle(request_queue.get())})
    except BaseException as exc:
        response_queue.put(
            {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        )


__all__ = ["observe_oracle", "worker_main"]

