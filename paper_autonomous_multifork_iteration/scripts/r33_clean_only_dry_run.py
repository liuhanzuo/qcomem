from __future__ import annotations

"""Run a non-scientific, fault-free local validation of the R33 clean gate.

The dry run exercises the real post-discovery GDN ownership repair on CPU toy
tensors, serializes byte-bound sidecars and transition receipts, proves request
objects were disposed, and sends the resulting report through the same strict
aggregator used by the formal lane.  It never opens or imports the R33 author
fault freeze and it never executes a fault.
"""

import argparse
from array import array
import gc
import hashlib
import json
from pathlib import Path
import platform
import sys
import weakref
from typing import Any

import torch

import r33_executor_core as core


RUN_ID = "R33-LOCAL-CLEAN-ONLY-DRY-RUN-V1"
DISPOSAL_SCHEMA = "forkaudit-r33-local-clean-disposal-v1"
DISPOSAL_OPERATION = "release-verified-request-objects-then-gc-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    core.require(not path.exists() and not pending.exists(), f"non-overwrite path exists: {path}")
    with pending.open("xb") as handle:
        handle.write(payload)
        handle.flush()
    pending.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_bytes_atomic(path, core.canonical_bytes(value) + b"\n")


class CacheObject:
    pass


class LayerObject:
    pass


def cache_from_tensors(values: list[torch.Tensor]) -> CacheObject:
    cache = CacheObject()
    cache.layers = []
    for value in values:
        layer = LayerObject()
        layer.conv_states = {0: value}
        cache.layers.append(layer)
    return cache


def _float32_bytes(values: list[float]) -> bytes:
    payload = array("f", values)
    if sys.byteorder != "little":
        payload.byteswap()
    return payload.tobytes()


def run(output_dir: Path) -> dict[str, Any]:
    core.require(not output_dir.exists(), "dry-run output directory already exists")
    output_dir.mkdir(parents=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir()

    script_dir = Path(__file__).resolve().parent
    repository_root = script_dir.parent.parent
    gpu_dir = repository_root / "gpu"
    repair_path = gpu_dir / "qcomem_single_token_gdn_ownership.py"
    core_path = script_dir / "r33_executor_core.py"
    source_bindings = {
        "clean_only_dry_runner_sha256": sha256_file(Path(__file__).resolve()),
        "executor_core_sha256": sha256_file(core_path),
        "postdiscovery_gdn_ownership_repair_sha256": sha256_file(repair_path),
    }
    lifecycle_capability = {
        "case_disposal_operation": DISPOSAL_OPERATION,
        "case_disposal_receipt_schema": DISPOSAL_SCHEMA,
        "python_reference_clear_only": False,
        "verification": "weak-reference liveness is checked after strong-reference release and gc.collect",
    }
    execution_input = {
        "schema_version": "forkaudit-r33-local-clean-input-v1",
        "run_id": RUN_ID,
        "device": "cpu",
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "fault_module_path": None,
        "faults_loaded": False,
        "faults_executed": False,
        "scientific_result": False,
    }
    execution_sha = core.sha256_json(execution_input)
    protocol = {
        "schema_version": core.PROTOCOL_SCHEMA,
        "run_id": RUN_ID,
        "mode": "local_clean_only_dry_run",
        "candidate_output_seen_when_frozen": False,
        "fault_ids": [],
        "execution_input_sha256": execution_sha,
        "source_bindings": source_bindings,
        "author_freeze_manifest_sha256": core.ZERO_SHA256,
        "fault_bindings": {},
        "lifecycle_capability_binding": lifecycle_capability,
        "claim_boundary": {
            "local_dry_run_is_scientific_evidence": False,
            "model_or_gpu_executed": False,
            "fault_module_loaded": False,
            "faults_executed": False,
            "purpose": "executor-contract and clean-gate integration validation only",
        },
    }
    protocol_sha = core.sha256_json(protocol)

    # Import the real repair only after the fault-free protocol is fixed.  This
    # module is an implementation dependency, not a fault definition.
    sys.path.insert(0, str(gpu_dir))
    try:
        from qcomem_single_token_gdn_ownership import (  # type: ignore
            exact_alias,
            overlaps,
            prepare_borrowed_single_token_conv_transition,
        )
    finally:
        sys.path.pop(0)

    base_values = [
        torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + index
        for index in range(3)
    ]
    persistent = cache_from_tensors(base_values)
    requests = [cache_from_tensors(base_values), cache_from_tensors(base_values)]
    request_refs = [weakref.ref(request) for request in requests]
    first = prepare_borrowed_single_token_conv_transition(
        persistent, requests, (0, 1, 2), request_index=0
    )
    clone_refs = [
        weakref.ref(requests[0].layers[index].conv_states[0]) for index in range(3)
    ]
    for index in range(3):
        selected = requests[0].layers[index].conv_states[0]
        base = persistent.layers[index].conv_states[0]
        peer = requests[1].layers[index].conv_states[0]
        core.require(not overlaps(selected, base), "dry clean selected/base overlap")
        core.require(not overlaps(selected, peer), "dry clean selected/peer overlap")
        core.require(exact_alias(base, peer), "dry clean peer/base alias drift")
        core.require(torch.equal(selected, base), "dry clean ownership repair changed bytes")
    selected = base = peer = None
    repeat = prepare_borrowed_single_token_conv_transition(
        persistent, requests, (0, 1, 2), request_index=0
    )
    first_replay = core.validate_transition_receipt(
        first,
        expected_clone_count=3,
        expected_action="cloned_borrowed_state",
        label="dry clean first",
    )
    repeat_replay = core.validate_transition_receipt(
        repeat,
        expected_clone_count=0,
        expected_action="already_private_noop",
        label="dry clean repeat",
    )

    logits_a = _float32_bytes([0.125, -2.0, 4.5, 3.0])
    logits_b = _float32_bytes([0.125, -2.0, 4.5, 3.0])
    core.require(logits_a == logits_b, "dry clean byte binding")
    sidecar_a = raw_dir / "borrowed-repaired-fp32.bin"
    sidecar_b = raw_dir / "materialized-control-fp32.bin"
    write_bytes_atomic(sidecar_a, logits_a)
    write_bytes_atomic(sidecar_b, logits_b)

    # Drop every request-owned strong reference and verify that the objects and
    # selected clones actually die.  Persistent base tensors intentionally stay
    # alive because they model the reusable document state, not request state.
    requests.clear()
    requests = []
    collected = gc.collect()
    request_objects_released = all(reference() is None for reference in request_refs)
    selected_clones_released = all(reference() is None for reference in clone_refs)
    core.require(request_objects_released, "dry clean request object remained live")
    core.require(selected_clones_released, "dry clean selected clone remained live")
    disposal_receipt = {
        "schema_version": DISPOSAL_SCHEMA,
        "operation": DISPOSAL_OPERATION,
        "completed": True,
        "registered_backend_restored": True,
        "strong_references_released": True,
        "request_objects_released": request_objects_released,
        "request_owned_clones_released": selected_clones_released,
        "gc_collect_completed": True,
        "gc_collect_return_value": int(collected),
        "accelerator_cleanup_required": False,
        "accelerator_cache_cleanup_completed": False,
        "accelerator_synchronize_completed": False,
        "allocator_baseline_exact": True,
        "cleanup_error": None,
    }
    lifecycle = {
        "capability_binding": lifecycle_capability,
        "capability_binding_sha256": core.sha256_json(lifecycle_capability),
        "disposal_receipt": disposal_receipt,
    }
    comparisons = {
        "greedy_token_exact": True,
        "canonical_fp32_logits_byte_exact": True,
        "terminal_request_0_gdn_exact": True,
        "terminal_logical_kv_exact": True,
    }
    receipt_chain = core.build_receipt_chain(
        run_id=RUN_ID,
        case_id="clean",
        ordered_payloads=(
            (
                "input_and_source_binding",
                {
                    "protocol_sha256": protocol_sha,
                    "execution_input_sha256": execution_sha,
                    "source_bindings_sha256": core.sha256_json(source_bindings),
                },
            ),
            (
                "ownership_transition",
                {
                    "first_replay": first_replay,
                    "repeat_replay": repeat_replay,
                },
            ),
            (
                "existing_validator_battery",
                {
                    "storage_replay_passed": True,
                    "binding_replay_passed": True,
                    "all_existing_gates_enabled": True,
                },
            ),
            (
                "semantic_exactness",
                {
                    **comparisons,
                    "borrowed_sidecar_sha256": core.sha256_bytes(logits_a),
                    "control_sidecar_sha256": core.sha256_bytes(logits_b),
                },
            ),
            (
                "lifecycle_cleanup",
                {
                    "capability_binding_sha256": core.sha256_json(lifecycle_capability),
                    "disposal_receipt_sha256": core.sha256_json(disposal_receipt),
                },
            ),
        ),
    )
    clean_report = {
        "schema_version": core.CLEAN_SCHEMA,
        "run_id": RUN_ID,
        "case_id": "clean",
        "status": "clean_pass",
        "local_dry_run": True,
        "scientific_result": False,
        "protocol_sha256": protocol_sha,
        "execution_input_sha256": execution_sha,
        "source_bindings": source_bindings,
        "fault_module_loaded": False,
        "faults_executed": False,
        "all_existing_gates_enabled": True,
        "full_horizon_reached": True,
        "false_positive": False,
        "comparisons": comparisons,
        "transition_receipts": {"first": first, "repeat": repeat},
        "storage_replay": {
            "passed": True,
            "candidate_modules_imported": False,
            "replayed_row_count": first_replay["conv_tensor_count"]
            + repeat_replay["conv_tensor_count"],
        },
        "binding_replay": {
            "passed": True,
            "candidate_modules_imported": False,
            "first_clone_count": first_replay["cloned_tensor_count"],
            "repeat_clone_count": repeat_replay["cloned_tensor_count"],
        },
        "lifecycle": lifecycle,
        "receipt_chain": receipt_chain,
    }
    clean_validation = core.validate_clean_report(
        clean_report,
        expected_run_id=RUN_ID,
        expected_protocol_sha256=protocol_sha,
        expected_execution_input_sha256=execution_sha,
    )
    gate = core.CleanGate(
        run_id=RUN_ID,
        protocol_sha256=protocol_sha,
        execution_input_sha256=execution_sha,
    )
    gate_receipt = gate.accept_clean(clean_report)
    core.require(gate.unlocked, "dry clean gate did not unlock")
    summary = core.aggregate_reports(
        protocol=protocol,
        clean_report=clean_report,
        fault_reports=[],
    )
    core.require(summary["scientific_valid"] is False, "dry clean became scientific")
    core.require(summary["faults_executed"] is False, "dry clean reported a fault")

    write_json_atomic(output_dir / "execution-input.json", execution_input)
    write_json_atomic(output_dir / "protocol.json", protocol)
    write_json_atomic(output_dir / "clean-result.json", clean_report)
    write_json_atomic(output_dir / "clean-validation.json", clean_validation)
    write_json_atomic(output_dir / "clean-gate-receipt.json", gate_receipt)
    write_json_atomic(output_dir / "strict-summary.json", summary)
    artifact_rows = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "artifacts.sha256":
            artifact_rows.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "sha256": sha256_file(path),
                    "nbytes": path.stat().st_size,
                }
            )
    ledger = "".join(f"{row['sha256']}  {row['path']}\n" for row in artifact_rows)
    write_bytes_atomic(output_dir / "artifacts.sha256", ledger.encode("utf-8"))
    return {
        "status": "passed",
        "output_dir": str(output_dir),
        "protocol_sha256": protocol_sha,
        "clean_receipt_chain_head_sha256": clean_validation["receipt_chain"][
            "chain_head_sha256"
        ],
        "artifacts": len(artifact_rows),
        "fault_module_loaded": False,
        "faults_executed": False,
        "scientific_result": False,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output-dir", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run(parser().parse_args().output_dir), sort_keys=True))
