from __future__ import annotations

"""Separate-process storage observer for the R33 trusted-capture audit.

This module imports no ForkAudit, QComem, model, runtime-adapter, or candidate
capture implementation.  It receives tensor views through torch's
multiprocessing reduction, rebuilds descriptors inside its own process, and
emits only raw observations.  It never receives or computes a pass/fail label.
"""

import hashlib
import hmac
import os
import secrets
import traceback
from itertools import combinations
from math import prod
from typing import Any, Mapping, Sequence

import torch

from r33_ipc_capture_protocol import (
    CAPTURE_SCHEMA,
    FORBIDDEN_JUDGMENT_FIELDS,
    RESPONSE_SCHEMA,
    canonical_bytes,
    require,
    sha256_json,
    validate_live_request,
    validate_manifest,
)


WORKER_SCHEMA = "forkaudit-r33-out-of-process-worker-v1"
CONTROL_SCHEMA = "forkaudit-r33-ipc-control-v1"


def _tensor_payload(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .contiguous()
        .cpu()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    shape = tuple(int(value) for value in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    require(shape and len(shape) == len(stride), "tensor shape/stride rank drift")
    require(all(value > 0 for value in shape), "tensor contains an empty axis")
    minimum = int(tensor.storage_offset())
    maximum = minimum
    for size, step in zip(shape, stride):
        displacement = (size - 1) * step
        minimum += min(displacement, 0)
        maximum += max(displacement, 0)
    require(minimum >= 0 and maximum >= minimum, "tensor view lies outside storage")
    element_size = int(tensor.element_size())
    return minimum * element_size, (maximum + 1) * element_size


def _row_key(row: Mapping[str, Any]) -> tuple[str, int, int, str, int]:
    request_index = -1 if row["request_index"] is None else int(row["request_index"])
    return (
        str(row["owner_kind"]),
        request_index,
        int(row["layer_index"]),
        str(row["state_family"]),
        int(row["state_index"]),
    )


def _relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    same_storage = left["storage_token"] == right["storage_token"]
    overlap = (
        same_storage
        and int(left["byte_start"]) < int(right["byte_end_exclusive"])
        and int(right["byte_start"]) < int(left["byte_end_exclusive"])
    )
    if not overlap:
        return "disjoint"
    exact_fields = (
        "byte_start",
        "byte_end_exclusive",
        "shape",
        "stride",
        "storage_offset",
        "dtype",
        "device",
        "storage_nbytes",
        "tensor_nbytes",
        "content_sha256",
    )
    if all(left[field] == right[field] for field in exact_fields):
        return "exact_alias"
    return "partial_overlap"


def _relation_vector(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    ordered = sorted(rows, key=_row_key)
    return [
        [list(_row_key(left)), list(_row_key(right)), _relation(left, right)]
        for left, right in combinations(ordered, 2)
    ]


class RawObserver:
    def __init__(self, manifest: Mapping[str, Any]) -> None:
        self.manifest = dict(manifest)
        self.slots = validate_manifest(self.manifest)
        self.secret = secrets.token_bytes(32)
        self.captured_ids: set[str] = set()
        # Keep every imported view alive until shutdown.  This prevents an old
        # mapping from disappearing and its receiver address being reused (ABA).
        self.pinned_views: dict[str, list[torch.Tensor]] = {}

    @property
    def commitment(self) -> str:
        return hashlib.sha256(self.secret).hexdigest()

    def _token(self, domain: str, payload: str) -> str:
        return hmac.new(
            self.secret,
            f"{domain}\0{payload}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def capture(self, message: Mapping[str, Any]) -> dict[str, Any]:
        validate_live_request(message, self.manifest)
        capture_id = str(message["capture_id"])
        require(capture_id not in self.captured_ids, "capture id replayed")
        rows: list[dict[str, Any]] = []
        refs: list[torch.Tensor] = []
        any_cuda = False
        all_cpu_shared = True
        items = sorted(message["slot_tensors"], key=lambda value: value["slot_id"])
        for item in items:
            slot_id = str(item["slot_id"])
            tensor = item["tensor"]
            require(
                isinstance(tensor, torch.Tensor)
                and tensor.is_floating_point()
                and tensor.numel() > 0,
                "live value is not a non-empty floating tensor",
            )
            refs.append(tensor)
            any_cuda = any_cuda or bool(tensor.is_cuda)
            all_cpu_shared = all_cpu_shared and (
                bool(tensor.is_cuda) or bool(tensor.is_shared())
            )
            storage = tensor.untyped_storage()
            storage_nbytes = int(storage.nbytes())
            byte_start, byte_end = _byte_interval(tensor)
            require(0 <= byte_start < byte_end <= storage_nbytes, "view exceeds storage")
            # The address is used only inside this independent process to join
            # views of the same imported mapping.  It is HMACed and never emitted.
            storage_fact = f"{tensor.device}\0{int(storage.data_ptr())}\0{storage_nbytes}"
            storage_token = self._token("receiver-storage", storage_fact)
            coordinate = self.slots[slot_id]
            view_fields = {
                "shape": [int(value) for value in tensor.shape],
                "stride": [int(value) for value in tensor.stride()],
                "storage_offset": int(tensor.storage_offset()),
                "dtype": str(tensor.dtype),
                "device": str(tensor.device),
                "storage_nbytes": storage_nbytes,
                "tensor_nbytes": int(prod(tensor.shape)) * int(tensor.element_size()),
                "byte_start": byte_start,
                "byte_end_exclusive": byte_end,
                "content_sha256": hashlib.sha256(_tensor_payload(tensor)).hexdigest(),
            }
            rows.append(
                {
                    "slot_id": slot_id,
                    "owner_kind": coordinate["owner_kind"],
                    "request_index": coordinate["request_index"],
                    "layer_index": int(coordinate["layer_index"]),
                    "state_family": coordinate["state_family"],
                    "state_index": int(coordinate["state_index"]),
                    **view_fields,
                    "storage_token": storage_token,
                    "view_token": self._token(
                        "receiver-view",
                        storage_token + "\0" + canonical_bytes(view_fields).decode("ascii"),
                    ),
                }
            )
        rows.sort(key=_row_key)
        relations = _relation_vector(rows)
        self.pinned_views[capture_id] = refs
        self.captured_ids.add(capture_id)
        serialized = canonical_bytes(rows).decode("ascii").lower()
        require(
            not any(term in serialized for term in ("data_ptr", "storage_ptr", "pointer")),
            "raw address leaked",
        )
        return {
            "schema_version": CAPTURE_SCHEMA,
            "capture_id": capture_id,
            "observer_pid": os.getpid(),
            "producer_pid": os.getppid(),
            "process_separated": os.getpid() != os.getppid(),
            "observer_session_commitment_sha256": self.commitment,
            "slot_manifest_sha256": self.manifest["manifest_sha256"],
            "live_request_fields_received": sorted(message),
            "live_slot_fields_received": sorted(message["slot_tensors"][0]),
            "judgment_fields_received": sorted(set(message) & FORBIDDEN_JUDGMENT_FIELDS),
            "candidate_verdict_fields_received": False,
            "raw_addresses_serialized": False,
            "receiver_derived_descriptors": True,
            "receiver_derived_relations": True,
            "transport": (
                "torch-cuda-ipc-reduction"
                if any_cuda
                else "torch-cpu-shared-memory-reduction"
            ),
            "all_cpu_tensors_shared_in_receiver": all_cpu_shared,
            "imported_views_pinned_against_receiver_aba": True,
            "row_count": len(rows),
            "relation_count": len(relations),
            "rows": rows,
            "rows_sha256": sha256_json(rows),
            "relation_vector_sha256": sha256_json(relations),
        }


def worker_main(request_queue: Any, response_queue: Any, manifest: Mapping[str, Any]) -> None:
    """Entrypoint used by a spawn-created process."""

    try:
        observer = RawObserver(manifest)
        response_queue.put(
            {
                "schema_version": RESPONSE_SCHEMA,
                "kind": "ready",
                "worker_schema_version": WORKER_SCHEMA,
                "observer_pid": os.getpid(),
                "producer_pid": os.getppid(),
                "process_separated": os.getpid() != os.getppid(),
                "observer_session_commitment_sha256": observer.commitment,
                "slot_manifest_sha256": manifest["manifest_sha256"],
                "candidate_modules_imported": False,
                "observer_generates_verdicts": False,
            }
        )
        while True:
            message = request_queue.get()
            if isinstance(message, dict) and message.get("schema_version") == CONTROL_SCHEMA:
                require(set(message) == {"schema_version", "operation"}, "control field drift")
                require(message.get("operation") == "stop", "unknown control operation")
                response_queue.put(
                    {
                        "schema_version": RESPONSE_SCHEMA,
                        "kind": "stopped",
                        "observer_pid": os.getpid(),
                        "capture_count": len(observer.captured_ids),
                        "pinned_capture_count": len(observer.pinned_views),
                    }
                )
                return
            capture = observer.capture(message)
            response_queue.put(
                {
                    "schema_version": RESPONSE_SCHEMA,
                    "kind": "capture",
                    "capture": capture,
                }
            )
    except BaseException as exc:
        response_queue.put(
            {
                "schema_version": RESPONSE_SCHEMA,
                "kind": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


__all__ = ["CONTROL_SCHEMA", "RawObserver", "WORKER_SCHEMA", "worker_main"]
