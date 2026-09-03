from __future__ import annotations

"""Run the R33 out-of-process GDN capture on the frozen Qwen3.5/H20 stack.

The model/runtime setup is intentionally reused from the validated R29 runner.
Unlike R29, this runner never invokes the candidate storage-witness producer or
the same-process observer.  The producer process sends only opaque slot/tensor
pairs; raw descriptors and relations are reconstructed in a spawn-created
process and evaluated later by ``r33_replay_independent_capture.py``.
"""

import argparse
import gc
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import r29_run_independent_gdn_observer as r29
from qcomem_vllm_paged_fair_control import SHARED_REUSE
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    GDN_MATERIALIZE_REQUEST_BASE,
    build_resident_request_group,
)
from run_qcomem_qwen35_vllm_paged_multifork_resident import _set_production_no_mask

from r33_ipc_capture_protocol import build_slot_manifest
from r33_out_of_process_capture import OutOfProcessCaptureSession
from r33_replay_independent_capture import (
    PHASE_GENERATION,
    PHASE_SETUP,
    PHASE_TRANSITION,
    POLICY_MATERIALIZED,
    POLICY_SHARED,
    RESULT_SCHEMA,
)


PREREG_SCHEMA = "forkaudit-r33-out-of-process-preregistration-v1"
CAPTURE_IDS = (
    "c-2d8d91660bc7",
    "c-7b91ee24a5d3",
    "c-f109e345a0c8",
)
POLICIES = (
    (POLICY_SHARED, GDN_BORROW_IMMUTABLE_BASE),
    (POLICY_MATERIALIZED, GDN_MATERIALIZE_REQUEST_BASE),
)


class R33RunError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R33RunError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _run_policy_cell(
    runtime: r29.Runtime, policy_name: str, runtime_policy: str
) -> dict[str, Any]:
    persistent = group = session = None
    backends: list[str] = []
    try:
        persistent, conversion = r29.rr2._convert_persistent(
            runtime.backbone, runtime.plan, runtime.document, resident_count=2
        )
        group = build_resident_request_group(
            persistent,
            runtime.plan,
            resident_count=2,
            policy=SHARED_REUSE,
            gdn_base_policy=runtime_policy,
        )
        _set_production_no_mask(group, runtime.plan.full_attention_layer_indices)
        manifest = build_slot_manifest(
            runtime.plan.linear_layer_indices,
            resident_count=2,
            capture_ids=CAPTURE_IDS,
            state_index=0,
        )
        # One observer process per policy cell bounds imported-mapping lifetime
        # and makes every three-phase lifecycle one process/session.
        session = OutOfProcessCaptureSession(manifest, timeout_seconds=300.0)
        ready_receipt = dict(session.ready)
        captures = [session.capture(CAPTURE_IDS[0], persistent, group.requests)]
        ledgers, backends = r29._make_backends(runtime, group)
        steps = [r29._model_step(runtime, group, backends[0], 0)]
        captures.append(session.capture(CAPTURE_IDS[1], persistent, group.requests))
        steps.append(r29._model_step(runtime, group, backends[1], 1))
        captures.append(session.capture(CAPTURE_IDS[2], persistent, group.requests))
        stop_receipt = session.close()
        session = None
        ledger_receipts = [
            r29.rr2._pointer_free_kernel_ledger(ledger.verify_complete())
            for ledger in ledgers
        ]
        return {
            "cell_id": f"H20-N2-kv-shared-gdn-{policy_name}",
            "policy": policy_name,
            "runtime_policy": runtime_policy,
            "kv_policy": SHARED_REUSE,
            "fresh_persistent_cache": True,
            "fresh_request_group": True,
            "conversion_receipt": asdict(conversion),
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
            "observer_ready_receipt": ready_receipt,
            "observer_stop_receipt": stop_receipt,
            "captures": captures,
            "model_steps": steps,
            "kernel_ledger_receipts": ledger_receipts,
        }
    finally:
        if session is not None:
            session.close(force=True)
        if backends:
            r29.rr2._unregister_backends(backends)
        persistent = group = session = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def run(args: argparse.Namespace) -> dict[str, Any]:
    prereg_raw = r29.check_file(
        args.preregistration,
        args.expected_preregistration_sha256,
        "R33 preregistration",
    )
    prereg = json.loads(prereg_raw)
    require(prereg.get("schema_version") == PREREG_SCHEMA, "preregistration schema drift")
    r29.check_file(args.source_ledger, args.expected_source_ledger_sha256, "R33 source ledger")
    r29.check_file(
        args.candidate_code_ledger,
        args.expected_candidate_code_ledger_sha256,
        "candidate runtime code ledger",
    )
    source_binding = prereg.get("source_binding", {})
    local_sources = {
        "runner_sha256": Path(__file__).resolve(),
        "protocol_sha256": Path(__file__).with_name("r33_ipc_capture_protocol.py"),
        "worker_sha256": Path(__file__).with_name("r33_independent_capture_worker.py"),
        "coordinator_sha256": Path(__file__).with_name("r33_out_of_process_capture.py"),
        "replay_sha256": Path(__file__).with_name("r33_replay_independent_capture.py"),
        "r29_runtime_adapter_sha256": Path(r29.__file__).resolve(),
    }
    for field, path in local_sources.items():
        require(sha256_file(path) == source_binding.get(field), f"{field} drift")
    require(
        args.expected_source_ledger_sha256 == source_binding.get("source_ledger_raw_sha256"),
        "source ledger binding drift",
    )
    require(
        args.expected_candidate_code_ledger_sha256
        == prereg["input_binding"]["candidate_code_ledger_raw_sha256"],
        "candidate runtime ledger binding drift",
    )
    runtime = r29._load_runtime(args, prereg)
    with torch.inference_mode():
        cells = [
            _run_policy_cell(runtime, policy_name, runtime_policy)
            for policy_name, runtime_policy in POLICIES
        ]
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_pending_independent_replay",
        "claim_authorized": False,
        "preregistration_sha256": args.expected_preregistration_sha256,
        "source_ledger_raw_sha256": args.expected_source_ledger_sha256,
        "candidate_runtime_code_ledger_raw_sha256": args.expected_candidate_code_ledger_sha256,
        "source_sha256": {field: sha256_file(path) for field, path in local_sources.items()},
        "independence_boundary": prereg["independence_boundary"],
        "hardware": runtime.hardware,
        "input_receipt": runtime.input_receipt,
        "cells": cells,
    }
    require(len(cells) == 2, "policy coverage drift")
    require(
        all(
            capture["process_separated"]
            and not capture["judgment_fields_received"]
            and capture["candidate_verdict_fields_received"] is False
            for cell in cells
            for capture in cell["captures"]
        ),
        "process/wire independence gate failed",
    )
    write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--preregistration", type=Path, required=True)
    value.add_argument("--expected-preregistration-sha256", required=True)
    value.add_argument("--source-ledger", type=Path, required=True)
    value.add_argument("--expected-source-ledger-sha256", required=True)
    value.add_argument("--candidate-code-ledger", type=Path, required=True)
    value.add_argument("--expected-candidate-code-ledger-sha256", required=True)
    value.add_argument("--model-dir", type=Path, required=True)
    value.add_argument("--model-weight-ledger", type=Path, required=True)
    value.add_argument("--model-artifact-ledger", type=Path, required=True)
    value.add_argument("--expected-weight-ledger-sha256", required=True)
    value.add_argument("--expected-artifact-ledger-sha256", required=True)
    value.add_argument("--pg19-data", type=Path, required=True)
    value.add_argument("--pg19-manifest", type=Path, required=True)
    value.add_argument("--expected-pg19-sha256", required=True)
    value.add_argument("--expected-pg19-manifest-sha256", required=True)
    value.add_argument("--expected-windows-sha256", required=True)
    value.add_argument("--frozen-query-banks", type=Path, required=True)
    value.add_argument("--expected-query-banks-sha256", required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
