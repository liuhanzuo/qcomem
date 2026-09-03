from __future__ import annotations

"""Clean-first post-discovery regression for the one-token GDN repair.

No faults are loaded or executed.  The repaired borrowed-base path is compared
with a separately rebuilt materialized-state control, while the original
storage and binding predicates remain unchanged.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import types
from types import SimpleNamespace
from typing import Any

import torch

import qcomem_forkaudit_storage_witness as storage_witness
import qcomem_single_token_gdn_ownership as repair

# The frozen R29 executor is reused only as a clean-stack adapter.  Prevent its
# known H01--H03 module from being loaded at all: none of the clean helpers
# below resolve an attribute on this deliberately empty module.
if "r29_heldout_fault_suite" in sys.modules:
    raise RuntimeError("known R29 fault module was loaded before the D clean runner")
_disabled_fault_module = types.ModuleType("r29_heldout_fault_suite")
_disabled_fault_module.__dict__["POSTDISCOVERY_D_CLEAN_DISABLED"] = True
sys.modules["r29_heldout_fault_suite"] = _disabled_fault_module
import r29_execute_heldout_faults as executor
import r30_clean_single_token_gdn_diagnostic as diagnostic
from run_qcomem_qwen35_vllm_paged_multifork_resident import (
    _linear_state_digest,
    _request_logical_kv_digests,
)


SCHEMA_VERSION = "forkaudit-r30-postdiscovery-d-clean-v1"
RUN_ID = "R30-POSTDISCOVERY-D-CLEAN-20260825C"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def numeric_gpu_receipt(expected_uuid: str, execution_input: dict[str, Any]) -> dict[str, Any]:
    """Bind the parent-mandated numeric CUDA selector to the frozen GPU UUID."""

    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "2", "D clean requires CUDA_VISIBLE_DEVICES=2")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "D clean requires one visible GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    capability = list(torch.cuda.get_device_capability(0))
    expected = execution_input["environment"]
    require(capability == expected["compute_capability"] and "H20" in properties.name, "assigned GPU environment drift")
    output = subprocess.run(
        [
            "nvidia-smi",
            "--id=2",
            "--query-gpu=index,uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    columns = [item.strip() for item in output.split(",")]
    require(
        len(columns) == 4
        and columns[0] == "2"
        and columns[1] == expected_uuid
        and columns[2] == expected["gpu_name"],
        "numeric CUDA selector/UUID binding drift",
    )
    return {
        "physical_index": 2,
        "cuda_visible_devices": "2",
        "uuid": columns[1],
        "name": columns[2],
        "memory_mib": int(columns[3]),
        "compute_capability": capability,
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def source_bindings(args: argparse.Namespace, execution_input: dict[str, Any]) -> dict[str, str]:
    bindings = {
        "runner_sha256": args.expected_runner_sha256,
        "repair_sha256": args.expected_repair_sha256,
        "diagnostic_sha256": args.expected_diagnostic_sha256,
        "repair_test_sha256": args.expected_repair_test_sha256,
        "detached_replay_sha256": args.expected_replay_sha256,
        "r29_executor_sha256": args.expected_executor_sha256,
        "r29_execution_input_raw_sha256": args.expected_execution_input_sha256,
        "protocol_raw_sha256": args.expected_protocol_sha256,
        "imported_rr2_code_ledger_raw_sha256": execution_input["code"][
            "imported_rr2_code_ledger_raw_sha256"
        ],
    }
    local = {
        "runner_sha256": Path(__file__).resolve(),
        "repair_sha256": Path(repair.__file__).resolve(),
        "diagnostic_sha256": Path(diagnostic.__file__).resolve(),
        "repair_test_sha256": args.repair_test.resolve(),
        "detached_replay_sha256": args.detached_replay.resolve(),
        "r29_executor_sha256": Path(execution_input["code"]["executor_path"]),
        "protocol_raw_sha256": args.protocol.resolve(),
    }
    for field, path in local.items():
        require(executor.sha256_file(path) == bindings[field], f"{field} drift")
    return bindings


def build_materialized_case(runtime: Any) -> tuple[Any, Any, dict[str, str]]:
    persistent, _ = executor.rr2._convert_persistent(
        runtime.backbone, runtime.plan, runtime.document, resident_count=2
    )
    source_guard = executor.resident.source_document_physical_digests(
        persistent, runtime.plan.full_attention_layer_indices
    )
    group = executor.build_resident_request_group(
        persistent,
        runtime.plan,
        resident_count=2,
        policy=executor.SHARED_REUSE,
        gdn_base_policy=executor.resident.GDN_MATERIALIZE_REQUEST_BASE,
    )
    executor._set_production_no_mask(group, runtime.plan.full_attention_layer_indices)
    return persistent, group, source_guard


def case_digests(runtime: Any, group: Any) -> dict[str, Any]:
    linear = dict(_linear_state_digest(group.requests[0], runtime.plan.linear_layer_indices))
    storage_keys = linear.pop("storage_keys", None)
    require(isinstance(storage_keys, list) and len(storage_keys) == 60, "linear digest storage-key count")
    require(linear.get("tensor_count") == 60, "linear digest tensor count")
    return {
        "request_0_gdn": linear,
        "logical_kv": _request_logical_kv_digests(
            group, runtime.plan.full_attention_layer_indices
        ),
        "absolute_storage_keys_persisted": False,
    }


def run_borrowed_repaired(runtime: Any, execution_input: dict[str, Any]) -> tuple[dict[str, Any], torch.Tensor]:
    persistent = group = persistent_guard = request_guard = kv_guard = None
    source_guard = None
    backend = ""
    logits: torch.Tensor | None = None
    try:
        (
            persistent,
            group,
            persistent_guard,
            request_guard,
            kv_guard,
            source_guard,
        ) = executor._build_fresh_case(runtime)
        registry = diagnostic.IdentityRegistry()
        setup = diagnostic.capture_snapshot(
            persistent, group, runtime.plan.linear_layer_indices, registry
        )
        setup_relations = diagnostic.ownership_relations(setup)
        transition_receipt = repair.prepare_borrowed_single_token_conv_transition(
            persistent,
            group.requests,
            runtime.plan.linear_layer_indices,
            request_index=0,
        )
        require(transition_receipt["cloned_tensor_count"] == 30, "first transition clone count")
        pre_kernel = diagnostic.capture_snapshot(
            persistent, group, runtime.plan.linear_layer_indices, registry
        )
        pre_kernel_relations = diagnostic.ownership_relations(pre_kernel)
        ledger, backend = executor._make_backend(runtime, group, 1)
        logits, model_step = executor._model_step(runtime, group, backend)
        kernel_ledger = executor.rr2._pointer_free_kernel_ledger(ledger.verify_complete())
        post = diagnostic.capture_snapshot(
            persistent, group, runtime.plan.linear_layer_indices, registry
        )
        post_relations = diagnostic.ownership_relations(post)
        gdn_phase = storage_witness.capture_gdn_phase_witness(
            persistent,
            group.requests,
            runtime.plan.linear_layer_indices,
            run_id=RUN_ID,
            cell_id="borrowed-repaired-single-token-N2",
            kv_policy=executor.SHARED_REUSE,
            phase=storage_witness.PHASE_POST_TRANSITION,
            policy=executor.GDN_BORROW_IMMUTABLE_BASE,
            persistent_guard=persistent_guard,
            request_guard=request_guard,
            completed_request_indices=[0],
        )
        storage_replay = storage_witness.replay_gdn_storage_witness(
            json.loads(json.dumps(gdn_phase["storage_witness"]))
        )
        binding_replay = storage_witness.replay_request_gdn_binding_witness(
            json.loads(json.dumps(gdn_phase["binding_witness"]))
        )
        persistent_receipt = storage_witness.verify_persistent_gdn_guard(
            persistent_guard, persistent
        )
        source_after = executor.resident.source_document_physical_digests(
            persistent, runtime.plan.full_attention_layer_indices
        )
        require(source_after == source_guard, "borrowed persistent KV changed")
        no_op = repair.prepare_borrowed_single_token_conv_transition(
            persistent,
            group.requests,
            runtime.plan.linear_layer_indices,
            request_index=0,
        )
        require(no_op["cloned_tensor_count"] == 0, "second transition helper was not no-op")
        result = {
            "policy": "borrow-immutable-base-with-single-token-conv-privatization",
            "model_step": model_step,
            "kernel_ledger": kernel_ledger,
            "setup_snapshot": setup,
            "setup_relations": setup_relations,
            "pre_kernel_snapshot": pre_kernel,
            "pre_kernel_relations": pre_kernel_relations,
            "post_snapshot": post,
            "post_relations": post_relations,
            "transition_receipt": transition_receipt,
            "repeat_helper_receipt": no_op,
            "gdn_phase_witness": gdn_phase,
            "detached_in_process_replay": {
                "storage": storage_replay,
                "binding": binding_replay,
            },
            "persistent_guard": persistent_receipt,
            "digests": case_digests(runtime, group),
        }
        return result, logits.detach().cpu().float().contiguous()
    finally:
        if backend:
            executor.rr2._unregister_backends([backend])
        logits = None
        persistent = group = persistent_guard = request_guard = kv_guard = None
        source_guard = None


def run_materialized_control(runtime: Any) -> tuple[dict[str, Any], torch.Tensor]:
    persistent = group = None
    source_guard = None
    backend = ""
    logits: torch.Tensor | None = None
    try:
        persistent, group, source_guard = build_materialized_case(runtime)
        ledger, backend = executor._make_backend(runtime, group, 1)
        logits, model_step = executor._model_step(runtime, group, backend)
        kernel_ledger = executor.rr2._pointer_free_kernel_ledger(ledger.verify_complete())
        source_after = executor.resident.source_document_physical_digests(
            persistent, runtime.plan.full_attention_layer_indices
        )
        require(source_after == source_guard, "materialized persistent KV changed")
        result = {
            "policy": "materialize-request-base-functional-rebind-control",
            "model_step": model_step,
            "kernel_ledger": kernel_ledger,
            "digests": case_digests(runtime, group),
        }
        return result, logits.detach().cpu().float().contiguous()
    finally:
        if backend:
            executor.rr2._unregister_backends([backend])
        logits = None
        persistent = group = None
        source_guard = None


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(not args.run_dir.exists(), "D clean run directory already exists")
    execution_raw = executor.read_bound_file(
        args.execution_input,
        args.expected_execution_input_sha256,
        "R29 frozen execution input",
    )
    execution_input = executor.validate_execution_input(json.loads(execution_raw))
    bindings = source_bindings(args, execution_input)
    protocol = json.loads(args.protocol.read_text())
    require(protocol["schema_version"] == "forkaudit-r30-postdiscovery-d-clean-prereg-v1", "protocol schema")
    require(protocol["candidate_output_seen_when_frozen"] is False, "protocol not outcome blind")
    require(protocol["run_id"] == RUN_ID, "protocol run id")
    require(protocol["run_dir"] == str(args.run_dir), "protocol run directory")
    require(protocol["gpu"]["cuda_visible_devices"] == "2", "protocol visible GPU")
    require(protocol["gpu"]["expected_gpu_uuid"] == args.expected_gpu_uuid, "protocol GPU UUID")
    require(
        protocol["source_bindings"]
        == {key: value for key, value in bindings.items() if key != "protocol_raw_sha256"},
        "protocol source binding drift",
    )
    require(protocol["faults"]["loaded"] is False, "protocol permits a loaded fault")
    require(protocol["faults"]["executed"] is False, "protocol permits a fault execution")
    require(protocol["clean_cell"]["request_index"] == 0, "protocol request index")
    require(protocol["clean_cell"]["resident_count"] == 2, "protocol resident count")
    require(protocol["clean_cell"]["token_count"] == 1, "protocol token count")
    require(protocol["clean_cell"]["expected_first_conv_clones"] == 30, "protocol clone count")
    require(protocol["clean_cell"]["expected_repeat_conv_clones"] == 0, "protocol repeat clone count")
    # This is a pure launch-integration adapter: the parent requires the
    # numeric selector ``2``, whereas the frozen R29 executor expected the UUID
    # string in CUDA_VISIBLE_DEVICES.  Bind index 2 back to that exact UUID and
    # leave every model, ownership, semantic, and acceptance rule unchanged.
    executor._gpu_receipt = numeric_gpu_receipt
    runtime = executor._load_runtime(
        SimpleNamespace(rank=0, expected_gpu_uuid=args.expected_gpu_uuid), execution_input
    )
    args.run_dir.mkdir(parents=True)
    (args.run_dir / "raw").mkdir()
    (args.run_dir / "receipts").mkdir()
    (args.run_dir / "logs").mkdir()
    executor.write_bytes_atomic(args.run_dir / "preregistration.json", args.protocol.read_bytes())
    with torch.inference_mode():
        warmup = executor._discarded_warmup(runtime)
        require(runtime.allocator_baseline is not None, "allocator baseline missing")
        before = executor._snapshot_allocator()
        require(before == runtime.allocator_baseline, "pre-clean allocator drift")
        borrowed, borrowed_logits = run_borrowed_repaired(runtime, execution_input)
        after_borrowed = executor._cleanup_allocator()
        require(after_borrowed == runtime.allocator_baseline, "borrowed cleanup drift")
        control, control_logits = run_materialized_control(runtime)
        after_control = executor._cleanup_allocator()
        require(after_control == runtime.allocator_baseline, "control cleanup drift")
    require(tuple(borrowed_logits.shape) == executor.SIDE_CAR_SHAPE, "borrowed logits shape")
    require(tuple(control_logits.shape) == executor.SIDE_CAR_SHAPE, "control logits shape")
    borrowed_bytes = executor.tensor_bytes(borrowed_logits)
    control_bytes = executor.tensor_bytes(control_logits)
    executor.write_bytes_atomic(args.run_dir / "raw" / "borrowed-repaired-fp32-logits.bin", borrowed_bytes)
    executor.write_bytes_atomic(args.run_dir / "raw" / "materialized-control-fp32-logits.bin", control_bytes)
    logits_exact = borrowed_bytes == control_bytes
    token_exact = borrowed["model_step"]["greedy_token_id"] == control["model_step"]["greedy_token_id"]
    gdn_exact = borrowed["digests"]["request_0_gdn"] == control["digests"]["request_0_gdn"]
    kv_exact = borrowed["digests"]["logical_kv"] == control["digests"]["logical_kv"]
    require(logits_exact, "borrowed repaired logits differ from materialized control")
    require(token_exact, "borrowed repaired token differs from materialized control")
    require(gdn_exact, "borrowed repaired GDN differs from materialized control")
    require(kv_exact, "borrowed repaired logical KV differs from materialized control")
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "valid_clean_positive",
        "post_discovery": True,
        "faults_executed": False,
        "known_r29_fault_module_loaded": False,
        "heldout_fault_claim_allowed": False,
        "paper_import_allowed": False,
        "source_bindings": bindings,
        "hardware": runtime.hardware,
        "input_receipt": runtime.input_receipt,
        "discarded_warmup": warmup,
        "borrowed_repaired": borrowed,
        "materialized_control": control,
        "comparisons": {
            "greedy_token_exact": token_exact,
            "canonical_fp32_logits_byte_exact": logits_exact,
            "terminal_request_0_gdn_exact": gdn_exact,
            "terminal_logical_kv_exact": kv_exact,
            "borrowed_logits_sha256": executor.sha256_bytes(borrowed_bytes),
            "control_logits_sha256": executor.sha256_bytes(control_bytes),
        },
        "cleanup": {
            "allocator_baseline": runtime.allocator_baseline,
            "before": before,
            "after_borrowed": after_borrowed,
            "after_control": after_control,
            "exact": True,
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    executor.write_json_atomic(args.run_dir / "raw" / "clean-result.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--expected-execution-input-sha256", required=True)
    value.add_argument("--expected-executor-sha256", required=True)
    value.add_argument("--expected-runner-sha256", required=True)
    value.add_argument("--expected-repair-sha256", required=True)
    value.add_argument("--expected-diagnostic-sha256", required=True)
    value.add_argument("--repair-test", type=Path, required=True)
    value.add_argument("--expected-repair-test-sha256", required=True)
    value.add_argument("--detached-replay", type=Path, required=True)
    value.add_argument("--expected-replay-sha256", required=True)
    value.add_argument("--expected-protocol-sha256", required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    return value


if __name__ == "__main__":
    run(parser().parse_args())
