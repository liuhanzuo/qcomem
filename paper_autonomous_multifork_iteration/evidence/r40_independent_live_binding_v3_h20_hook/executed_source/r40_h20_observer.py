from __future__ import annotations

"""Source-distinct post-binding observer and fail-closed comparison."""

import hashlib
import json
import os
import queue
import sys
from itertools import combinations
from math import prod
from typing import Any, Mapping

import torch
import torch.multiprocessing as mp

from r40_h20_binding_protocol import (
    OBSERVATION_SCHEMA,
    challenge_nonce,
    require,
    seal,
    selected_slot_map,
    sha256_json,
    verify_seal,
)


def _observer_slot_id(coordinate: Mapping[str, Any]) -> str:
    # Deliberately independent implementation from the registrar helper.
    fields = {
        key: coordinate[key]
        for key in ("owner_kind", "request_index", "layer_index", "state_family", "state_index")
    }
    payload = json_bytes({"domain": "r40-h20-preserialization-slot-v1", **fields})
    return "s-" + hashlib.sha256(payload).hexdigest()[:20]


def json_bytes(value: Any) -> bytes:
    import json

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _payload(tensor: torch.Tensor) -> bytes:
    require(sys.byteorder == "little", "canonical tensor bytes require little endian")
    contiguous = tensor.detach().contiguous().cpu()
    raw = contiguous.view(torch.uint8).numpy().tobytes(order="C")
    header = json.dumps(
        {"dtype": str(contiguous.dtype), "shape": [int(v) for v in contiguous.shape]},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return len(header).to_bytes(4, "big") + header + raw


def observe(items: list[dict[str, Any]], preregistration: Mapping[str, Any]) -> dict[str, Any]:
    expected = selected_slot_map(preregistration)
    require(isinstance(items, list) and len(items) == len(expected), "observer item count drift")
    observed_slots = [str(item.get("slot_id")) for item in items]
    require(all(set(item) == {"slot_id", "tensor"} for item in items), "observer wire drift")
    require(set(observed_slots) == set(expected), "observer slot-set drift")
    require(
        set(expected) == {_observer_slot_id(row) for row in preregistration["selected_coordinates"]},
        "observer independent slot derivation drift",
    )
    seed = preregistration["geometry"]["challenge_seed_sha256"]
    internal: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: row["slot_id"]):
        slot = str(item["slot_id"])
        tensor = item["tensor"]
        require(isinstance(tensor, torch.Tensor) and tensor.numel() > 0, "observer tensor drift")
        storage = tensor.untyped_storage()
        nonce = challenge_nonce(seed, slot)
        internal.append(
            {
                "slot_id": slot,
                "shape": list(map(int, tensor.shape)),
                "stride": list(map(int, tensor.stride())),
                "storage_offset": int(tensor.storage_offset()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "tensor_nbytes": int(prod(tensor.shape)) * int(tensor.element_size()),
                "storage_nbytes": int(storage.nbytes()),
                "challenge_response_sha256": hashlib.sha256(
                    b"r40-h20-live-binding-challenge-v1\0" + nonce + b"\0" + _payload(tensor)
                ).hexdigest(),
                "_storage_key": (str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())),
            }
        )
    relations = []
    for left, right in combinations(internal, 2):
        if left["_storage_key"] != right["_storage_key"]:
            relation = "disjoint"
        else:
            exact = all(
                left[field] == right[field]
                for field in ("shape", "stride", "storage_offset", "tensor_nbytes")
            )
            relation = "exact_alias" if exact else "partial_overlap"
        relations.append([left["slot_id"], right["slot_id"], relation])
    rows = [
        {key: value for key, value in row.items() if key != "_storage_key"}
        for row in internal
    ]
    return seal(
        {
            "schema_version": OBSERVATION_SCHEMA,
            "role": "independent_postbinding_observer",
            "worker_pid": os.getpid(),
            "row_count": len(rows),
            "relation_count": len(relations),
            "rows": rows,
            "rows_sha256": sha256_json(rows),
            "relations": relations,
            "relations_sha256": sha256_json(relations),
            "raw_addresses_serialized": False,
            "payload_sha256": None,
        },
        "payload_sha256",
    )


def detect(oracle: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    verify_seal(oracle, "payload_sha256")
    verify_seal(candidate, "payload_sha256")
    oracle_rows = {row["slot_id"]: row for row in oracle["rows"]}
    candidate_rows = {row["slot_id"]: row for row in candidate["rows"]}
    require(set(oracle_rows) == set(candidate_rows), "detector slot-set drift")
    challenge = sorted(
        slot
        for slot in oracle_rows
        if oracle_rows[slot]["challenge_response_sha256"]
        != candidate_rows[slot]["challenge_response_sha256"]
    )
    oracle_relations = {(a, b): r for a, b, r in oracle["relations"]}
    candidate_relations = {(a, b): r for a, b, r in candidate["relations"]}
    relation = [
        [a, b, oracle_relations[(a, b)], candidate_relations[(a, b)]]
        for a, b in sorted(oracle_relations)
        if oracle_relations[(a, b)] != candidate_relations[(a, b)]
    ]
    codes = []
    if challenge:
        codes.append("challenge_response_mismatch")
    if relation:
        codes.append("storage_relation_mismatch")
    return {
        "passed": not codes,
        "failure_codes": codes,
        "challenge_mismatch_slot_ids": challenge,
        "relation_mismatch_pairs": relation,
        "numeric_tolerance": 0,
    }


def worker_main(request_queue: Any, response_queue: Any, preregistration: Mapping[str, Any]) -> None:
    try:
        response_queue.put({"ok": True, "kind": "ready", "worker_pid": os.getpid()})
        capture_count = 0
        while True:
            message = request_queue.get()
            if message.get("operation") == "capture":
                require(set(message) == {"operation", "items"}, "observer request drift")
                response_queue.put(
                    {"ok": True, "kind": "capture", "capture": observe(message["items"], preregistration)}
                )
                capture_count += 1
            elif message.get("operation") == "stop":
                require(set(message) == {"operation"}, "observer stop drift")
                response_queue.put(
                    {"ok": True, "kind": "stopped", "worker_pid": os.getpid(), "capture_count": capture_count}
                )
                return
            else:
                raise RuntimeError("unknown observer operation")
    except BaseException as exc:
        response_queue.put({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})


class ObserverClient:
    def __init__(self, preregistration: Mapping[str, Any], *, timeout: float = 180.0) -> None:
        context = mp.get_context("spawn")
        self.timeout = float(timeout)
        self.request_queue = context.Queue()
        self.response_queue = context.Queue()
        self.process = context.Process(
            target=worker_main,
            args=(self.request_queue, self.response_queue, dict(preregistration)),
        )
        self.process.start()
        self.worker_pid = int(self._receive("ready")["worker_pid"])

    def _receive(self, kind: str) -> dict[str, Any]:
        try:
            response = self.response_queue.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise RuntimeError(f"observer timeout: {kind}") from exc
        require(response.get("ok") is True, f"observer failed: {response}")
        require(response.get("kind") == kind, "observer response kind drift")
        return response

    def capture(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        self.request_queue.put({"operation": "capture", "items": items})
        return self._receive("capture")["capture"]

    def close(self) -> int:
        count = 0
        if self.process.is_alive():
            self.request_queue.put({"operation": "stop"})
            count = int(self._receive("stopped")["capture_count"])
            self.process.join(timeout=30.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10.0)
        self.request_queue.close()
        self.response_queue.close()
        return count


__all__ = ["ObserverClient", "detect", "observe"]
