from __future__ import annotations

"""Independent raw-container registrar; never accepts a producer slot manifest."""

import hashlib
import os
import queue
from itertools import combinations
from math import prod
from typing import Any, Mapping

import torch
import torch.multiprocessing as mp

from r40_h20_binding_protocol import (
    OBSERVATION_SCHEMA,
    REGISTRATION_SCHEMA,
    challenge_nonce,
    coordinate_key,
    protocol_slot_id,
    require,
    seal,
    selected_by_event,
    selected_slot_map,
    sha256_json,
)


FORBIDDEN_REGISTRATION_FIELDS = frozenset(
    {
        "manifest",
        "slot_manifest",
        "slot_id",
        "slot_tensors",
        "candidate_rows",
        "expected_verdict",
        "phase",
        "passed",
        "verdict",
    }
)
REGISTER_FIELDS = frozenset(
    {
        "schema_version",
        "operation",
        "owner_kind",
        "request_index",
        "layer_index",
        "conv_states",
        "recurrent_states",
    }
)


def _payload(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().numpy().tobytes(order="C")


def _row(seed: str, slot: str, tensor: torch.Tensor, epoch: int) -> dict[str, Any]:
    require(isinstance(tensor, torch.Tensor) and tensor.numel() > 0, "registrar tensor drift")
    storage = tensor.untyped_storage()
    nonce = challenge_nonce(seed, slot)
    return {
        "slot_id": slot,
        "epoch": int(epoch),
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
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


def _relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    if left["_storage_key"] != right["_storage_key"]:
        return "disjoint"
    exact = all(
        left[field] == right[field]
        for field in ("shape", "stride", "storage_offset", "tensor_nbytes")
    )
    return "exact_alias" if exact else "partial_overlap"


class RegistrationOracle:
    def __init__(self, preregistration: Mapping[str, Any]) -> None:
        self.preregistration = dict(preregistration)
        self.expected = selected_slot_map(self.preregistration)
        self.seed = self.preregistration["geometry"]["challenge_seed_sha256"]
        self.tensors: dict[str, torch.Tensor] = {}
        self.epochs: dict[str, int] = {}
        self.event_count = 0
        self.forbidden_fields_received = 0

    def register_containers(self, event: Mapping[str, Any]) -> dict[str, Any]:
        require(isinstance(event, dict), "registration event type drift")
        self.forbidden_fields_received += len(set(event) & FORBIDDEN_REGISTRATION_FIELDS)
        require(not (set(event) & FORBIDDEN_REGISTRATION_FIELDS), "forbidden registration field")
        require(set(event) == REGISTER_FIELDS, "registration field-set drift")
        require(event.get("schema_version") == REGISTRATION_SCHEMA, "registration schema drift")
        require(event.get("operation") in {"initial", "refresh"}, "registration operation drift")
        owner_kind = str(event["owner_kind"])
        request_index = event["request_index"]
        layer_index = int(event["layer_index"])
        selected = selected_by_event(
            self.preregistration,
            owner_kind=owner_kind,
            request_index=request_index,
            layer_index=layer_index,
        )
        containers = {
            "conv": event["conv_states"],
            "recurrent": event["recurrent_states"],
        }
        registered: list[str] = []
        for coordinate in selected:
            states = containers[coordinate["state_family"]]
            require(isinstance(states, dict), "raw state container is not a dict")
            state_index = int(coordinate["state_index"])
            require(state_index in states, "selected state missing from raw container")
            tensor = states[state_index]
            require(isinstance(tensor, torch.Tensor), "raw state value is not a tensor")
            slot = protocol_slot_id(coordinate)
            prior_epoch = self.epochs.get(slot, -1)
            if event["operation"] == "initial":
                require(slot not in self.tensors, "initial slot registered twice")
                epoch = 0
            else:
                require(slot in self.tensors, "refresh before initial registration")
                epoch = prior_epoch + 1
            self.tensors[slot] = tensor
            self.epochs[slot] = epoch
            registered.append(slot)
        self.event_count += 1
        return {
            "operation": event["operation"],
            "registered_slot_ids": sorted(registered),
            "event_index": self.event_count - 1,
            "forbidden_fields_received": self.forbidden_fields_received,
        }

    def snapshot(self) -> dict[str, Any]:
        require(set(self.tensors) == set(self.expected), "registrar selected-slot coverage drift")
        internal = [
            _row(self.seed, slot, self.tensors[slot], self.epochs[slot])
            for slot in sorted(self.tensors)
        ]
        relations = [
            [left["slot_id"], right["slot_id"], _relation(left, right)]
            for left, right in combinations(internal, 2)
        ]
        rows = [
            {key: value for key, value in row.items() if key != "_storage_key"}
            for row in internal
        ]
        result = {
            "schema_version": OBSERVATION_SCHEMA,
            "role": "independent_prebinder_registrar",
            "worker_pid": os.getpid(),
            "expected_slot_count": len(self.expected),
            "row_count": len(rows),
            "relation_count": len(relations),
            "rows": rows,
            "rows_sha256": sha256_json(rows),
            "relations": relations,
            "relations_sha256": sha256_json(relations),
            "producer_manifest_received": False,
            "producer_slot_ids_received": False,
            "forbidden_fields_received": self.forbidden_fields_received,
            "registrar_derived_slot_ids": True,
            "registrar_derived_expected_set": True,
            "raw_addresses_serialized": False,
            "payload_sha256": None,
        }
        return seal(result, "payload_sha256")


def worker_main(request_queue: Any, response_queue: Any, preregistration: Mapping[str, Any]) -> None:
    try:
        oracle = RegistrationOracle(preregistration)
        response_queue.put({"ok": True, "kind": "ready", "worker_pid": os.getpid()})
        while True:
            message = request_queue.get()
            operation = message.get("operation") if isinstance(message, dict) else None
            if operation in {"initial", "refresh"}:
                response_queue.put(
                    {"ok": True, "kind": "ack", "receipt": oracle.register_containers(message)}
                )
            elif operation == "snapshot":
                require(set(message) == {"operation"}, "snapshot field drift")
                response_queue.put({"ok": True, "kind": "snapshot", "snapshot": oracle.snapshot()})
            elif operation == "stop":
                require(set(message) == {"operation"}, "stop field drift")
                response_queue.put({"ok": True, "kind": "stopped", "worker_pid": os.getpid()})
                return
            else:
                raise RuntimeError("unknown registrar operation")
    except BaseException as exc:
        response_queue.put({"ok": False, "error_type": type(exc).__name__, "error": str(exc)})


class RegistrarClient:
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
        ready = self._receive("ready")
        self.worker_pid = int(ready["worker_pid"])

    def _receive(self, kind: str) -> dict[str, Any]:
        try:
            response = self.response_queue.get(timeout=self.timeout)
        except queue.Empty as exc:
            raise RuntimeError(f"registrar timeout: {kind}") from exc
        require(response.get("ok") is True, f"registrar failed: {response}")
        require(response.get("kind") == kind, "registrar response kind drift")
        return response

    def register(self, event: Mapping[str, Any]) -> dict[str, Any]:
        self.request_queue.put(dict(event))
        return self._receive("ack")["receipt"]

    def snapshot(self) -> dict[str, Any]:
        self.request_queue.put({"operation": "snapshot"})
        return self._receive("snapshot")["snapshot"]

    def close(self) -> None:
        if self.process.is_alive():
            self.request_queue.put({"operation": "stop"})
            self._receive("stopped")
            self.process.join(timeout=30.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=10.0)
        self.request_queue.close()
        self.response_queue.close()


__all__ = [
    "FORBIDDEN_REGISTRATION_FIELDS",
    "REGISTER_FIELDS",
    "RegistrarClient",
    "RegistrationOracle",
]
