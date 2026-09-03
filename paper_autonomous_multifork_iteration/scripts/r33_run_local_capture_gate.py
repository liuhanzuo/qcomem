from __future__ import annotations

"""Run the R33 separate-process capture protocol on a local CPU mock cache.

This is an engineering gate only.  It exercises real multiprocessing shared
storage, alias preservation, descriptor reconstruction, lifecycle replay, and
the no-live-verdict wire contract; it is not model/runtime evidence.
"""

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from r33_ipc_capture_protocol import build_slot_manifest, sha256_json
from r33_out_of_process_capture import OutOfProcessCaptureSession
from r33_replay_independent_capture import (
    PHASE_GENERATION,
    PHASE_SETUP,
    PHASE_TRANSITION,
    POLICY_MATERIALIZED,
    POLICY_SHARED,
    RESULT_SCHEMA,
    replay_result,
)


LAYERS = tuple(range(3))
CAPTURE_IDS = (
    "c-2d8d91660bc7",
    "c-7b91ee24a5d3",
    "c-f109e345a0c8",
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_persistent() -> SimpleNamespace:
    layers = []
    for layer_index in LAYERS:
        layers.append(
            SimpleNamespace(
                conv_states={
                    0: torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
                    + layer_index
                },
                recurrent_states={
                    0: torch.arange(20, dtype=torch.float32).reshape(1, 2, 2, 5)
                    + 100
                    + layer_index
                },
            )
        )
    return SimpleNamespace(layers=layers)


def make_request(persistent: SimpleNamespace, *, borrowed: bool) -> SimpleNamespace:
    layers = []
    for source in persistent.layers:
        layers.append(
            SimpleNamespace(
                conv_states={
                    0: source.conv_states[0] if borrowed else source.conv_states[0].clone()
                },
                recurrent_states={
                    0: (
                        source.recurrent_states[0]
                        if borrowed
                        else source.recurrent_states[0].clone()
                    )
                },
            )
        )
    return SimpleNamespace(layers=layers)


def rebind(request: SimpleNamespace, delta: float) -> None:
    for layer in request.layers:
        layer.conv_states[0] = layer.conv_states[0].clone() + delta
        layer.recurrent_states[0] = layer.recurrent_states[0].clone() + delta


def run_cell(policy: str) -> dict[str, Any]:
    persistent = make_persistent()
    requests = [
        make_request(persistent, borrowed=policy == POLICY_SHARED),
        make_request(persistent, borrowed=policy == POLICY_SHARED),
    ]
    manifest = build_slot_manifest(
        LAYERS, resident_count=2, capture_ids=CAPTURE_IDS, state_index=0
    )
    session = OutOfProcessCaptureSession(manifest, timeout_seconds=60.0)
    try:
        captures = [session.capture(CAPTURE_IDS[0], persistent, requests)]
        rebind(requests[0], 1.0)
        captures.append(session.capture(CAPTURE_IDS[1], persistent, requests))
        rebind(requests[1], 2.0)
        captures.append(session.capture(CAPTURE_IDS[2], persistent, requests))
        stop_receipt = session.close()
    except BaseException:
        session.close(force=True)
        raise
    return {
        "cell_id": f"local-cpu-N2-{policy}",
        "policy": policy,
        "slot_manifest": manifest,
        "capture_plan": [
            {
                "capture_id": CAPTURE_IDS[0],
                "phase": PHASE_SETUP,
                "completed_request_indices": [],
            },
            {
                "capture_id": CAPTURE_IDS[1],
                "phase": PHASE_TRANSITION,
                "completed_request_indices": [0],
            },
            {
                "capture_id": CAPTURE_IDS[2],
                "phase": PHASE_GENERATION,
                "completed_request_indices": [0, 1],
            },
        ],
        "observer_ready_receipt": session.ready,
        "observer_stop_receipt": stop_receipt,
        "captures": captures,
    }


def run() -> dict[str, Any]:
    script_root = Path(__file__).resolve().parent
    source_names = (
        "r33_ipc_capture_protocol.py",
        "r33_independent_capture_worker.py",
        "r33_out_of_process_capture.py",
        "r33_replay_independent_capture.py",
        "r33_run_local_capture_gate.py",
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_valid_engineering_gate_not_scientific_model_execution",
        "claim_authorized": False,
        "scientific_result_available": False,
        "purpose": "transport, process-separation, raw-descriptor, alias, and lifecycle engineering gate",
        "producer_pid": os.getpid(),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": str(torch.__version__),
            "cuda_available": bool(torch.cuda.is_available()),
            "device": "cpu",
        },
        "source_sha256": {
            name: _sha256_file(script_root / name) for name in source_names
        },
        "cells": [run_cell(POLICY_SHARED), run_cell(POLICY_MATERIALIZED)],
    }
    result["result_sha256_without_self"] = sha256_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    replay = replay_result(result)
    _write_json(args.output, result)
    _write_json(args.replay_output, replay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
