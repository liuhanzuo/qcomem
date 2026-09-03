from __future__ import annotations

"""Producer-side coordinator for the R33 out-of-process raw observer.

This file is the only R33 component that traverses the live owner objects.  It
binds each tensor to an opaque slot from a frozen manifest and sends no policy,
phase, completion, expectation, candidate row, or verdict field to the worker.
"""

import os
import queue
from typing import Any, Mapping

import torch
import torch.multiprocessing as mp

from r33_independent_capture_worker import CONTROL_SCHEMA, worker_main
from r33_ipc_capture_protocol import (
    REQUEST_SCHEMA,
    RESPONSE_SCHEMA,
    STATE_ATTRIBUTES,
    require,
    validate_manifest,
)


class OutOfProcessCaptureError(RuntimeError):
    pass


def bind_live_tensors(
    manifest: Mapping[str, Any], persistent: Any, requests: list[Any] | tuple[Any, ...]
) -> list[dict[str, Any]]:
    """Traverse owners according to the frozen manifest and return opaque slots."""

    slots = validate_manifest(manifest)
    require(len(requests) == int(manifest["resident_count"]), "resident count drift")
    items: list[dict[str, Any]] = []
    for slot_id, coordinate in sorted(slots.items()):
        if coordinate["owner_kind"] == "persistent":
            owner = persistent
        else:
            request_index = int(coordinate["request_index"])
            require(0 <= request_index < len(requests), "request coordinate drift")
            owner = requests[request_index]
        layers = getattr(owner, "layers", None)
        require(isinstance(layers, (list, tuple)), "owner layers are not a sequence")
        layer_index = int(coordinate["layer_index"])
        require(0 <= layer_index < len(layers), "owner layer missing")
        family = str(coordinate["state_family"])
        states = getattr(layers[layer_index], STATE_ATTRIBUTES[family], None)
        state_index = int(coordinate["state_index"])
        require(isinstance(states, dict) and state_index in states, "state mapping missing")
        tensor = states[state_index]
        require(isinstance(tensor, torch.Tensor), "state value is not a tensor")
        items.append({"slot_id": slot_id, "tensor": tensor})
    return items


class OutOfProcessCaptureSession:
    def __init__(self, manifest: Mapping[str, Any], *, timeout_seconds: float = 120.0) -> None:
        validate_manifest(manifest)
        self.manifest = dict(manifest)
        self.timeout_seconds = float(timeout_seconds)
        self.producer_pid = os.getpid()
        self._closed = False
        self._captured_ids: set[str] = set()
        context = mp.get_context("spawn")
        self._request_queue = context.Queue()
        self._response_queue = context.Queue()
        self._process = context.Process(
            target=worker_main,
            args=(self._request_queue, self._response_queue, self.manifest),
            name="forkaudit-r33-independent-capture",
        )
        self._process.start()
        self.ready = self._receive("ready")
        if not self.ready.get("process_separated"):
            self.close(force=True)
            raise OutOfProcessCaptureError("observer did not enter a separate process")
        if self.ready.get("observer_pid") == self.producer_pid:
            self.close(force=True)
            raise OutOfProcessCaptureError("observer PID equals producer PID")
        if self.ready.get("producer_pid") != self.producer_pid:
            self.close(force=True)
            raise OutOfProcessCaptureError("observer parent PID receipt drift")
        if self.ready.get("slot_manifest_sha256") != self.manifest["manifest_sha256"]:
            self.close(force=True)
            raise OutOfProcessCaptureError("observer manifest binding drift")

    def _receive(self, expected_kind: str) -> dict[str, Any]:
        try:
            response = self._response_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            alive = bool(getattr(self, "_process", None) and self._process.is_alive())
            raise OutOfProcessCaptureError(
                f"observer response timeout (expected={expected_kind}, alive={alive})"
            ) from exc
        if not isinstance(response, dict) or response.get("schema_version") != RESPONSE_SCHEMA:
            raise OutOfProcessCaptureError("observer response schema drift")
        if response.get("kind") == "error":
            raise OutOfProcessCaptureError(
                f"observer error {response.get('error_type')}: {response.get('error')}\n"
                f"{response.get('traceback', '')}"
            )
        if response.get("kind") != expected_kind:
            raise OutOfProcessCaptureError(
                f"observer response kind drift: {response.get('kind')!r}"
            )
        return response

    def capture(
        self,
        capture_id: str,
        persistent: Any,
        requests: list[Any] | tuple[Any, ...],
    ) -> dict[str, Any]:
        if self._closed:
            raise OutOfProcessCaptureError("observer session is closed")
        if capture_id in self._captured_ids:
            raise OutOfProcessCaptureError("capture id replayed")
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        message = {
            "schema_version": REQUEST_SCHEMA,
            "capture_id": str(capture_id),
            "slot_tensors": bind_live_tensors(self.manifest, persistent, requests),
        }
        self._request_queue.put(message)
        response = self._receive("capture")
        capture = response["capture"]
        if capture.get("capture_id") != capture_id:
            raise OutOfProcessCaptureError("capture id receipt drift")
        if capture.get("producer_pid") != self.producer_pid:
            raise OutOfProcessCaptureError("capture producer PID drift")
        self._captured_ids.add(capture_id)
        return capture

    def close(self, *, force: bool = False) -> dict[str, Any] | None:
        if self._closed:
            return None
        self._closed = True
        receipt: dict[str, Any] | None = None
        try:
            if self._process.is_alive() and not force:
                self._request_queue.put(
                    {"schema_version": CONTROL_SCHEMA, "operation": "stop"}
                )
                receipt = self._receive("stopped")
                self._process.join(timeout=min(self.timeout_seconds, 30.0))
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=10.0)
        finally:
            self._request_queue.close()
            self._response_queue.close()
        return receipt

    def __enter__(self) -> "OutOfProcessCaptureSession":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(force=exc is not None)


__all__ = [
    "OutOfProcessCaptureError",
    "OutOfProcessCaptureSession",
    "bind_live_tensors",
]
