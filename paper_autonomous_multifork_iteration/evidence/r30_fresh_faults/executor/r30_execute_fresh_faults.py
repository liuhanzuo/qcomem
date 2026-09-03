from __future__ import annotations

"""Execute the sealed R30 fresh cache-state perturbations on one fixed stack.

This source is intentionally separate from the author freeze.  It imports the
clean post-discovery runner only after the fault semantics were sealed.  The
known earlier fault module remains replaced by the clean runner's empty module.
"""

import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable

import torch

import r30_run_postdiscovery_d_clean as clean


RUN_ID = "R30-FRESH-FAULTS-20260825A"
SCHEMA_VERSION = "forkaudit-r30-fresh-fault-executor-v1"
FAULT_IDS = (
    "R30F1_BORROWED_GDN_DETACH_EQUAL",
    "R30F2_IMMUTABLE_KV_TRANSIENT_FLIP_RESTORE",
    "R30F3_RELEASE_BEFORE_FINAL_OBSERVATION",
)

executor = clean.executor
repair = clean.repair
diagnostic = clean.diagnostic
storage_witness = clean.storage_witness


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(raw)


def write_json(path: Path, value: Any) -> None:
    executor.write_json_atomic(path, value)


def exception_record(exc: BaseException, stage: str) -> dict[str, Any]:
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "module": type(exc).__module__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def gpu5_receipt(expected_uuid: str, execution_input: dict[str, Any]) -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "5", "requires CUDA_VISIBLE_DEVICES=5")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "requires one visible GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    capability = list(torch.cuda.get_device_capability(0))
    expected = execution_input["environment"]
    require(capability == expected["compute_capability"], "GPU compute capability drift")
    output = subprocess.run(
        [
            "nvidia-smi",
            "--id=5",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    columns = [item.strip() for item in output.split(",")]
    require(
        len(columns) == 6
        and columns[0] == "5"
        and columns[1] == expected_uuid
        and columns[2] == expected["gpu_name"]
        and int(columns[4]) == 0
        and int(columns[5]) == 0,
        "GPU5 identity or idle-state drift",
    )
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--id=5",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    require(not compute, "GPU5 acquired a compute process before model load")
    return {
        "physical_index": 5,
        "cuda_visible_devices": "5",
        "uuid": columns[1],
        "name": columns[2],
        "memory_mib": int(columns[3]),
        "memory_used_mib_before_load": int(columns[4]),
        "utilization_percent_before_load": int(columns[5]),
        "compute_processes_before_load": [],
        "compute_capability": capability,
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def compare_float32(left: bytes, right: bytes) -> dict[str, Any]:
    require(len(left) == len(right) and len(left) % 4 == 0, "float32 sidecar size drift")
    left_values = array("f")
    right_values = array("f")
    left_values.frombytes(left)
    right_values.frombytes(right)
    if sys.byteorder != "little":
        left_values.byteswap()
        right_values.byteswap()
    max_abs = 0.0
    diff_sq = 0.0
    ref_sq = 0.0
    finite = True
    for observed, reference in zip(left_values, right_values):
        finite = finite and math.isfinite(observed) and math.isfinite(reference)
        delta = float(observed) - float(reference)
        max_abs = max(max_abs, abs(delta))
        diff_sq += delta * delta
        ref_sq += float(reference) * float(reference)
    if ref_sq == 0.0:
        relative_l2: float | None = 0.0 if diff_sq == 0.0 else None
        denominator_zero = True
    else:
        relative_l2 = math.sqrt(diff_sq / ref_sq)
        denominator_zero = False
    return {
        "byte_exact": left == right,
        "left_sha256": sha256_bytes(left),
        "right_sha256": sha256_bytes(right),
        "nbytes": len(left),
        "scalar_count": len(left_values),
        "all_finite": finite,
        "max_abs_error": max_abs,
        "relative_l2": relative_l2,
        "reference_l2_denominator_zero": denominator_zero,
    }


def verify_source_bindings(args: argparse.Namespace, prereg: dict[str, Any]) -> dict[str, str]:
    paths = {
        "executor_source_sha256": Path(__file__).resolve(),
        "detached_replay_source_sha256": args.replay_source.resolve(),
        "clean_runner_source_sha256": Path(clean.__file__).resolve(),
        "repaired_source_sha256": Path(repair.__file__).resolve(),
        "diagnostic_source_sha256": Path(diagnostic.__file__).resolve(),
        "reference_clean_result_sha256": args.reference_clean_result.resolve(),
        "reference_clean_replay_sha256": args.reference_clean_replay.resolve(),
        "author_freeze_manifest_sha256": args.author_freeze_manifest.resolve(),
    }
    expected = prereg["source_bindings"]
    require(set(paths) == set(expected), "preregistered source-binding key drift")
    observed = {key: sha256_file(path) for key, path in paths.items()}
    require(observed == expected, "source or reference binding drift")
    return observed


def run_detached_replay(
    args: argparse.Namespace,
    *,
    case_id: str,
    result_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-I",
        str(args.replay_source.resolve()),
        "--run-dir",
        str(args.run_dir.resolve()),
        "--case-id",
        case_id,
        "--result",
        str(result_path.resolve()),
        "--output",
        str(output_path.resolve()),
        "--expected-prereg-sha256",
        args.expected_prereg_sha256,
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300)
    log_path = args.run_dir / "logs" / f"{case_id}-detached-replay.log"
    executor.write_bytes_atomic(
        log_path,
        (completed.stdout + completed.stderr).encode("utf-8", errors="replace"),
    )
    require(completed.returncode == 0, f"detached replay failed for {case_id}")
    replay = json.loads(output_path.read_text())
    require(replay["status"] == "passed", f"detached replay non-pass for {case_id}")
    require(replay["candidate_modules_imported"] is False, "detached replay imported candidate modules")
    return replay


def clean_control(runtime: Any, execution_input: dict[str, Any], case_dir: Path) -> tuple[dict[str, Any], bytes]:
    before = executor._snapshot_allocator()
    require(before == runtime.allocator_baseline, "pre-clean allocator drift")
    borrowed, borrowed_logits = clean.run_borrowed_repaired(runtime, execution_input)
    after_borrowed = executor._cleanup_allocator()
    require(after_borrowed == runtime.allocator_baseline, "borrowed clean cleanup drift")
    control, control_logits = clean.run_materialized_control(runtime)
    after_control = executor._cleanup_allocator()
    require(after_control == runtime.allocator_baseline, "materialized clean cleanup drift")
    borrowed_bytes = executor.tensor_bytes(borrowed_logits)
    control_bytes = executor.tensor_bytes(control_logits)
    sidecar_borrowed = case_dir / "raw" / "borrowed-repaired-fp32-logits.bin"
    sidecar_control = case_dir / "raw" / "materialized-control-fp32-logits.bin"
    executor.write_bytes_atomic(sidecar_borrowed, borrowed_bytes)
    executor.write_bytes_atomic(sidecar_control, control_bytes)
    logits = compare_float32(borrowed_bytes, control_bytes)
    comparisons = {
        "greedy_token_exact": borrowed["model_step"]["greedy_token_id"]
        == control["model_step"]["greedy_token_id"],
        "canonical_fp32_logits": logits,
        "terminal_request_0_gdn_exact": borrowed["digests"]["request_0_gdn"]
        == control["digests"]["request_0_gdn"],
        "terminal_logical_kv_exact": borrowed["digests"]["logical_kv"]
        == control["digests"]["logical_kv"],
    }
    require(comparisons["greedy_token_exact"], "clean token mismatch")
    require(logits["byte_exact"], "clean full-logit mismatch")
    require(comparisons["terminal_request_0_gdn_exact"], "clean GDN mismatch")
    require(comparisons["terminal_logical_kv_exact"], "clean logical-KV mismatch")
    reference = json.loads(args_global.reference_clean_result.read_text())
    reference_replay = json.loads(args_global.reference_clean_replay.read_text())
    reference_equivalence = {
        "reference_status": reference["status"],
        "reference_detached_replay_status": reference_replay["status"],
        "borrowed_token_exact": borrowed["model_step"]["greedy_token_id"]
        == reference["borrowed_repaired"]["model_step"]["greedy_token_id"],
        "control_token_exact": control["model_step"]["greedy_token_id"]
        == reference["materialized_control"]["model_step"]["greedy_token_id"],
        "borrowed_full_logit_sha256_exact": sha256_bytes(borrowed_bytes)
        == reference["comparisons"]["borrowed_logits_sha256"],
        "control_full_logit_sha256_exact": sha256_bytes(control_bytes)
        == reference["comparisons"]["control_logits_sha256"],
        "borrowed_terminal_gdn_exact": borrowed["digests"]["request_0_gdn"]
        == reference["borrowed_repaired"]["digests"]["request_0_gdn"],
        "borrowed_terminal_logical_kv_exact": borrowed["digests"]["logical_kv"]
        == reference["borrowed_repaired"]["digests"]["logical_kv"],
    }
    require(reference_equivalence["reference_status"] == "valid_clean_positive", "reference clean invalid")
    require(
        reference_equivalence["reference_detached_replay_status"] == "detached_clean_replay_passed",
        "reference detached replay invalid",
    )
    require(all(value is True for key, value in reference_equivalence.items() if key.endswith("_exact")), "fresh/reference clean drift")
    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "case_id": "clean",
        "status": "fresh_clean_passed",
        "classification": "clean_pass",
        "all_gates_enabled": True,
        "false_positive": False,
        "borrowed_repaired": borrowed,
        "materialized_control": control,
        "comparisons": comparisons,
        "reference_equivalence": reference_equivalence,
        "sidecars": {
            "borrowed": {
                "path": str(sidecar_borrowed.relative_to(args_global.run_dir)),
                "sha256": sha256_bytes(borrowed_bytes),
                "nbytes": len(borrowed_bytes),
                "shape": list(borrowed_logits.shape),
                "dtype": "float32-little-endian",
            },
            "control": {
                "path": str(sidecar_control.relative_to(args_global.run_dir)),
                "sha256": sha256_bytes(control_bytes),
                "nbytes": len(control_bytes),
                "shape": list(control_logits.shape),
                "dtype": "float32-little-endian",
            },
        },
        "allocator": {
            "baseline": runtime.allocator_baseline,
            "before": before,
            "after_borrowed": after_borrowed,
            "after_control": after_control,
            "exact": True,
        },
    }
    return result, borrowed_bytes


def request_state_references(request: Any, layer_indices: Any) -> list[torch.Tensor]:
    output: list[torch.Tensor] = []
    for layer_index in sorted(int(value) for value in layer_indices):
        layer = request.layers[layer_index]
        output.append(layer.conv_states[0])
        output.append(layer.recurrent_states[0])
    return output


def run_fault_f1(runtime: Any, case_dir: Path) -> dict[str, Any]:
    persistent = group = None
    request = base = current = clone = None
    before_refs: list[torch.Tensor] = []
    after_refs: list[torch.Tensor] = []
    stage = "build_fresh_case"
    result: dict[str, Any]
    try:
        persistent, group, *_ = executor._build_fresh_case(runtime)
        layer_index = min(int(value) for value in runtime.plan.linear_layer_indices)
        request = group.requests[0]
        base = persistent.layers[layer_index].conv_states[0]
        current = request.layers[layer_index].conv_states[0]
        require(repair.exact_alias(current, base), "F1 precondition: selected slot is not a base alias")
        registry = diagnostic.IdentityRegistry()
        before_refs = request_state_references(request, runtime.plan.linear_layer_indices)
        before_ids = [id(value) for value in before_refs]
        before_sha = repair.tensor_sha256(current)
        before_interval = list(repair.byte_interval(current))
        before_storage = registry.storage_label(current)
        stage = "inject_detached_equal_clone"
        clone = current.detach().clone(memory_format=torch.preserve_format)
        request.layers[layer_index].conv_states[0] = clone
        after_refs = request_state_references(request, runtime.plan.linear_layer_indices)
        changed_reference_count = sum(
            left != id(right) for left, right in zip(before_ids, after_refs)
        )
        after_sha = repair.tensor_sha256(clone)
        witness = {
            "nonce": "r30f1-borrowed-detach-9f7c2a48b16e03d5",
            "seed_hex": "0x9f7c2a48b16e03d5",
            "request_index": 0,
            "layer_index": layer_index,
            "state_family": "conv_states",
            "state_index": 0,
            "before_content_sha256": before_sha,
            "after_content_sha256": after_sha,
            "before_storage_token": before_storage,
            "after_storage_token": registry.storage_label(clone),
            "before_byte_interval": before_interval,
            "after_byte_interval": list(repair.byte_interval(clone)),
            "pre_exact_alias": True,
            "post_base_disjoint": not repair.overlaps(clone, base),
            "shape_exact": tuple(clone.shape) == tuple(current.shape),
            "stride_exact": tuple(clone.stride()) == tuple(current.stride()),
            "dtype_exact": clone.dtype == current.dtype,
            "device_exact": clone.device == current.device,
            "changed_reference_count": changed_reference_count,
        }
        require(before_sha == after_sha, "F1 injection changed tensor bytes")
        require(witness["post_base_disjoint"], "F1 clone still overlaps base")
        require(changed_reference_count == 1, "F1 changed more than one request reference")
        stage = "existing_setup_capture"
        snapshot = diagnostic.capture_snapshot(
            persistent, group, runtime.plan.linear_layer_indices, registry
        )
        relations = diagnostic.ownership_relations(snapshot)
        stage = "ordinary_repaired_transition"
        repair.prepare_borrowed_single_token_conv_transition(
            persistent,
            group.requests,
            runtime.plan.linear_layer_indices,
            request_index=0,
        )
        stage = "harness_unexpected_no_stop"
        raise RuntimeError(
            "F1 passed the repaired transition; a full-horizon continuation is required"
        )
    except BaseException as exc:
        if stage == "harness_unexpected_no_stop":
            raise
        record = exception_record(exc, stage)
        if stage.startswith("existing_"):
            classification = "existing_validator_rejection"
            catch_gate = stage
        else:
            classification = "ordinary_assertion_or_exception"
            catch_gate = None
        result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "case_id": FAULT_IDS[0],
            "status": "classified_stop",
            "classification": classification,
            "catch_gate": catch_gate,
            "ordinary_exception": record if classification == "ordinary_assertion_or_exception" else None,
            "validator_exception": record if classification == "existing_validator_rejection" else None,
            "full_horizon_reached": False,
            "tokens_available": False,
            "full_logits_available": False,
            "injector_witness": locals().get("witness"),
            "setup_snapshot": locals().get("snapshot"),
            "setup_relations": locals().get("relations"),
            "semantic_comparisons": None,
        }
    finally:
        persistent = group = None
        request = base = current = clone = None
        before_refs = []
        after_refs = []
    cleanup = executor._cleanup_allocator()
    result["allocator_cleanup"] = {
        "observed": cleanup,
        "baseline": runtime.allocator_baseline,
        "exact": cleanup == runtime.allocator_baseline,
    }
    require(result["allocator_cleanup"]["exact"], "F1 allocator cleanup drift")
    return result


def kv_document_k_sha(arena: Any) -> str:
    document_blocks = arena.batch_size * arena.document_blocks_per_sequence
    return repair.tensor_sha256(arena.key_cache[:document_blocks])


def run_fault_f2(runtime: Any, clean_logits: bytes, clean_result: dict[str, Any], case_dir: Path) -> dict[str, Any]:
    persistent = group = persistent_guard = request_guard = kv_guard = None
    source_guard = None
    backend = ""
    logits: torch.Tensor | None = None
    arena = byte_view = ledger = None
    stage = "build_fresh_case"
    capture_count = 0
    model_call_count = 0
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
        stage = "existing_setup_capture"
        setup = diagnostic.capture_snapshot(
            persistent, group, runtime.plan.linear_layer_indices, registry
        )
        capture_count += 1
        setup_relations = diagnostic.ownership_relations(setup)
        stage = "inject_transient_kv_flip_restore"
        layer_index = min(int(value) for value in runtime.plan.full_attention_layer_indices)
        arena = persistent.layers[layer_index].arena
        physical_page = int(arena.document_block_table[0, -1].item())
        valid_on_last_page = int(arena.document_length % arena.page_size) or int(arena.page_size)
        token_index = valid_on_last_page - 1
        byte_view = arena.key_cache[physical_page, token_index, 0, 0:1].view(torch.uint8)
        require(int(byte_view.numel()) >= 1, "F2 selected scalar has no byte")
        torch.cuda.synchronize()
        pre_sha = kv_document_k_sha(arena)
        saved_byte = int(byte_view[0].item())
        mutated_byte = saved_byte ^ 0x01
        captures_before = capture_count
        calls_before = model_call_count
        mutation_start_ns = time.time_ns()
        mutated_sha: str | None = None
        mutation_observed_ns: int | None = None
        restored_sha: str | None = None
        restoration_complete_ns: int | None = None
        try:
            byte_view[0] = mutated_byte
            torch.cuda.synchronize()
            mutated_sha = kv_document_k_sha(arena)
            mutation_observed_ns = time.time_ns()
        finally:
            byte_view[0] = saved_byte
            torch.cuda.synchronize()
            restored_sha = kv_document_k_sha(arena)
            restoration_complete_ns = time.time_ns()
        require(mutated_sha is not None and mutation_observed_ns is not None, "F2 mutation observation missing")
        require(restored_sha is not None and restoration_complete_ns is not None, "F2 restoration observation missing")
        witness = {
            "nonce": "r30f2-kv-transient-5b81d43e7a20c6f9",
            "seed_hex": "0x5b81d43e7a20c6f9",
            "layer_index": layer_index,
            "component": "K",
            "physical_page": physical_page,
            "logical_page": int(arena.document_blocks_per_sequence) - 1,
            "logical_token_on_page": token_index,
            "head_index": 0,
            "head_dim_index": 0,
            "byte_within_scalar": 0,
            "saved_byte": saved_byte,
            "mutated_byte": mutated_byte,
            "xor_mask": 1,
            "pre_sha256": pre_sha,
            "mutated_sha256": mutated_sha,
            "restored_sha256": restored_sha,
            "mutation_start_ns": mutation_start_ns,
            "mutation_observed_ns": mutation_observed_ns,
            "restoration_complete_ns": restoration_complete_ns,
            "candidate_capture_count_before": captures_before,
            "candidate_capture_count_after": capture_count,
            "model_call_count_before": calls_before,
            "model_call_count_after": model_call_count,
            "device_synchronization_bracketed": True,
        }
        require(pre_sha != mutated_sha, "F2 mutation did not change the K digest")
        require(pre_sha == restored_sha, "F2 restoration did not recover exact K bytes")
        require(capture_count == captures_before and model_call_count == calls_before, "F2 activity occurred inside mutation interval")
        stage = "ordinary_repaired_transition"
        transition_receipt = repair.prepare_borrowed_single_token_conv_transition(
            persistent,
            group.requests,
            runtime.plan.linear_layer_indices,
            request_index=0,
        )
        require(transition_receipt["cloned_tensor_count"] == 30, "F2 transition clone count")
        stage = "existing_pre_kernel_capture"
        pre_kernel = diagnostic.capture_snapshot(
            persistent, group, runtime.plan.linear_layer_indices, registry
        )
        capture_count += 1
        pre_kernel_relations = diagnostic.ownership_relations(pre_kernel)
        stage = "ordinary_model_step"
        ledger, backend = executor._make_backend(runtime, group, 1)
        logits, model_step = executor._model_step(runtime, group, backend)
        model_call_count += 1
        kernel_ledger = executor.rr2._pointer_free_kernel_ledger(ledger.verify_complete())
        stage = "existing_post_transition_capture"
        post = diagnostic.capture_snapshot(
            persistent, group, runtime.plan.linear_layer_indices, registry
        )
        capture_count += 1
        post_relations = diagnostic.ownership_relations(post)
        stage = "existing_gdn_storage_witness"
        gdn_phase = storage_witness.capture_gdn_phase_witness(
            persistent,
            group.requests,
            runtime.plan.linear_layer_indices,
            run_id=RUN_ID,
            cell_id="r30f2-transient-kv-N2",
            kv_policy=executor.SHARED_REUSE,
            phase=storage_witness.PHASE_POST_TRANSITION,
            policy=executor.GDN_BORROW_IMMUTABLE_BASE,
            persistent_guard=persistent_guard,
            request_guard=request_guard,
            completed_request_indices=[0],
        )
        stage = "existing_storage_replay"
        storage_replay = storage_witness.replay_gdn_storage_witness(
            json.loads(json.dumps(gdn_phase["storage_witness"]))
        )
        stage = "existing_binding_replay"
        binding_replay = storage_witness.replay_request_gdn_binding_witness(
            json.loads(json.dumps(gdn_phase["binding_witness"]))
        )
        stage = "existing_persistent_guard"
        persistent_receipt = storage_witness.verify_persistent_gdn_guard(
            persistent_guard, persistent
        )
        stage = "existing_persistent_kv_guard"
        source_after = executor.resident.source_document_physical_digests(
            persistent, runtime.plan.full_attention_layer_indices
        )
        require(source_after == source_guard, "F2 persistent KV changed at registered horizon")
        stage = "ordinary_repeat_transition"
        no_op = repair.prepare_borrowed_single_token_conv_transition(
            persistent,
            group.requests,
            runtime.plan.linear_layer_indices,
            request_index=0,
        )
        require(no_op["cloned_tensor_count"] == 0, "F2 repeat helper not no-op")
        digests = clean.case_digests(runtime, group)
        fault_bytes = executor.tensor_bytes(logits.detach().cpu().float().contiguous())
        sidecar = case_dir / "raw" / "fault-fp32-logits.bin"
        executor.write_bytes_atomic(sidecar, fault_bytes)
        logit_comparison = compare_float32(fault_bytes, clean_logits)
        semantic = {
            "greedy_token_exact": model_step["greedy_token_id"]
            == clean_result["borrowed_repaired"]["model_step"]["greedy_token_id"],
            "canonical_fp32_logits": logit_comparison,
            "terminal_request_0_gdn_exact": digests["request_0_gdn"]
            == clean_result["borrowed_repaired"]["digests"]["request_0_gdn"],
            "terminal_logical_kv_exact": digests["logical_kv"]
            == clean_result["borrowed_repaired"]["digests"]["logical_kv"],
        }
        result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "case_id": FAULT_IDS[1],
            "status": "full_horizon_completed",
            "classification": "escape_full_horizon",
            "catch_gate": None,
            "ordinary_exception": None,
            "full_horizon_reached": True,
            "tokens_available": True,
            "full_logits_available": True,
            "injector_witness": witness,
            "setup_snapshot": setup,
            "setup_relations": setup_relations,
            "pre_kernel_snapshot": pre_kernel,
            "pre_kernel_relations": pre_kernel_relations,
            "post_snapshot": post,
            "post_relations": post_relations,
            "transition_receipt": transition_receipt,
            "repeat_helper_receipt": no_op,
            "kernel_ledger": kernel_ledger,
            "model_step": model_step,
            "gdn_phase_witness": gdn_phase,
            "in_process_replay": {
                "storage": storage_replay,
                "binding": binding_replay,
            },
            "persistent_guard": persistent_receipt,
            "digests": digests,
            "semantic_comparisons": semantic,
            "sidecar": {
                "path": str(sidecar.relative_to(args_global.run_dir)),
                "sha256": sha256_bytes(fault_bytes),
                "nbytes": len(fault_bytes),
                "shape": list(logits.shape),
                "dtype": "float32-little-endian",
            },
            "event_counts": {
                "candidate_captures": capture_count,
                "model_calls": model_call_count,
            },
        }
    except BaseException as exc:
        record = exception_record(exc, stage)
        classification = (
            "existing_validator_rejection" if stage.startswith("existing_") else "ordinary_assertion_or_exception"
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "case_id": FAULT_IDS[1],
            "status": "classified_stop",
            "classification": classification,
            "catch_gate": stage if classification == "existing_validator_rejection" else None,
            "ordinary_exception": record if classification == "ordinary_assertion_or_exception" else None,
            "validator_exception": record if classification == "existing_validator_rejection" else None,
            "full_horizon_reached": False,
            "tokens_available": logits is not None,
            "full_logits_available": logits is not None,
            "injector_witness": locals().get("witness"),
            "semantic_comparisons": None,
        }
    finally:
        if backend:
            executor.rr2._unregister_backends([backend])
        logits = None
        arena = byte_view = ledger = None
        persistent = group = persistent_guard = request_guard = kv_guard = None
        source_guard = None
    cleanup = executor._cleanup_allocator()
    result["allocator_cleanup"] = {
        "observed": cleanup,
        "baseline": runtime.allocator_baseline,
        "exact": cleanup == runtime.allocator_baseline,
    }
    require(result["allocator_cleanup"]["exact"], "F2 allocator cleanup drift")
    return result


def run_fault_f3(clean_runner_path: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    capability = prereg["lifecycle_capability_binding"]
    require(
        capability["clean_runner_sha256"] == sha256_file(clean_runner_path),
        "F3 lifecycle-capability source binding drift",
    )
    require(
        capability["authoritative_request_release_operation"] is None
        and capability["authoritative_request_release_receipt_schema"] is None,
        "F3 prereg unexpectedly binds an authoritative release capability",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "case_id": FAULT_IDS[2],
        "status": "operationally_invalid_precondition",
        "classification": "operationally_invalid_precondition",
        "catch_gate": None,
        "ordinary_exception": None,
        "full_horizon_reached": False,
        "tokens_available": False,
        "full_logits_available": False,
        "injector_witness": None,
        "semantic_comparisons": None,
        "precondition_evidence": {
            "clean_runner_sha256": sha256_file(clean_runner_path),
            "clean_authoritative_release_transition_present": False,
            "lifecycle_capability_binding": capability,
            "lifecycle_capability_binding_sha256": canonical_sha256(capability),
            "clean_finalization_observed": "Python references are cleared in finally without an authoritative release operation or receipt.",
            "decision": "Do not synthesize a lifecycle API; preserve the frozen fault and record no mutation run.",
        },
    }


def terminal_ledger(run_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name == "terminal-files.sha256":
            continue
        rows.append(
            {
                "path": str(path.relative_to(run_dir)),
                "sha256": sha256_file(path),
                "nbytes": path.stat().st_size,
            }
        )
    return rows


def run(args: argparse.Namespace) -> dict[str, Any]:
    global args_global
    args_global = args
    require(not args.run_dir.exists(), "non-overwrite run directory already exists")
    require(sha256_file(args.prereg) == args.expected_prereg_sha256, "preregistration SHA drift")
    prereg = json.loads(args.prereg.read_text())
    require(prereg["schema_version"] == "forkaudit-r30-fresh-fault-executor-prereg-v1", "prereg schema")
    require(prereg["run_id"] == RUN_ID, "preregistered run ID drift")
    require(prereg["fault_order"] == list(FAULT_IDS), "frozen fault order drift")
    require(prereg["expected_gate_names_assigned"] is False, "prereg assigned expected detector gates")
    require(
        prereg["execution_input"]["path"] == str(args.execution_input)
        and prereg["execution_input"]["sha256"] == args.expected_execution_input_sha256,
        "execution-input CLI binding differs from preregistration",
    )
    require(
        prereg["gpu"]["physical_index"] == 5
        and prereg["gpu"]["cuda_visible_devices"] == "5"
        and prereg["gpu"]["expected_uuid"] == args.expected_gpu_uuid,
        "GPU CLI binding differs from preregistration",
    )
    bindings = verify_source_bindings(args, prereg)
    author_freeze = args.author_freeze_manifest.parent
    freeze_seal = author_freeze / "FREEZE.sha256"
    require(freeze_seal.read_text().strip().split()[0] == bindings["author_freeze_manifest_sha256"], "author freeze seal drift")
    execution_raw = executor.read_bound_file(
        args.execution_input,
        args.expected_execution_input_sha256,
        "bound clean-stack execution input",
    )
    execution_input = executor.validate_execution_input(json.loads(execution_raw))
    executor._gpu_receipt = gpu5_receipt
    clean.RUN_ID = RUN_ID
    runtime = executor._load_runtime(
        SimpleNamespace(rank=0, expected_gpu_uuid=args.expected_gpu_uuid), execution_input
    )
    args.run_dir.mkdir(parents=True)
    (args.run_dir / "logs").mkdir()
    (args.run_dir / "receipts").mkdir()
    shutil.copy2(args.prereg, args.run_dir / "preregistration.json")
    shutil.copytree(author_freeze, args.run_dir / "author_freeze")
    source_dir = args.run_dir / "executor_source"
    source_dir.mkdir()
    shutil.copy2(Path(__file__).resolve(), source_dir / Path(__file__).name)
    shutil.copy2(args.replay_source.resolve(), source_dir / args.replay_source.name)
    shutil.copy2(Path(repair.__file__).resolve(), source_dir / Path(repair.__file__).name)
    write_json(
        args.run_dir / "receipts" / "binding-receipt.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": RUN_ID,
            "source_bindings": bindings,
            "execution_input_sha256": args.expected_execution_input_sha256,
            "gpu": runtime.hardware,
            "trial_id": 1899487,
            "pod": "qs-249885-1899487-ai-1443683-master-0",
            "author_freeze_manifest_sha256": bindings["author_freeze_manifest_sha256"],
        },
    )
    with torch.inference_mode():
        warmup = executor._discarded_warmup(runtime)
        require(runtime.allocator_baseline is not None, "allocator baseline missing")
        clean_dir = args.run_dir / "clean"
        (clean_dir / "raw").mkdir(parents=True)
        clean_result, clean_logits = clean_control(runtime, execution_input, clean_dir)
        clean_result["hardware"] = runtime.hardware
        clean_result["discarded_warmup"] = warmup
        clean_result["source_bindings"] = bindings
        clean_path = clean_dir / "raw" / "result.json"
        write_json(clean_path, clean_result)
        clean_replay_path = clean_dir / "detached-replay.json"
        clean_replay = run_detached_replay(
            args, case_id="clean", result_path=clean_path, output_path=clean_replay_path
        )
        require(clean_replay["status"] == "passed", "fresh clean replay failed; faults blocked")
        print(json.dumps({"case_id": "clean", "status": "fresh_clean_passed"}), flush=True)
        fault_results: list[dict[str, Any]] = []
        f1_dir = args.run_dir / FAULT_IDS[0]
        (f1_dir / "raw").mkdir(parents=True)
        f1 = run_fault_f1(runtime, f1_dir)
        f1["source_bindings"] = bindings
        f1_path = f1_dir / "raw" / "result.json"
        write_json(f1_path, f1)
        f1["detached_replay"] = run_detached_replay(
            args,
            case_id=FAULT_IDS[0],
            result_path=f1_path,
            output_path=f1_dir / "detached-replay.json",
        )
        fault_results.append(f1)
        print(json.dumps({"case_id": FAULT_IDS[0], "classification": f1["classification"]}), flush=True)
        f2_dir = args.run_dir / FAULT_IDS[1]
        (f2_dir / "raw").mkdir(parents=True)
        f2 = run_fault_f2(runtime, clean_logits, clean_result, f2_dir)
        f2["source_bindings"] = bindings
        f2_path = f2_dir / "raw" / "result.json"
        write_json(f2_path, f2)
        f2["detached_replay"] = run_detached_replay(
            args,
            case_id=FAULT_IDS[1],
            result_path=f2_path,
            output_path=f2_dir / "detached-replay.json",
        )
        fault_results.append(f2)
        print(json.dumps({"case_id": FAULT_IDS[1], "classification": f2["classification"]}), flush=True)
        f3_dir = args.run_dir / FAULT_IDS[2]
        (f3_dir / "raw").mkdir(parents=True)
        f3 = run_fault_f3(Path(clean.__file__).resolve(), prereg)
        f3["source_bindings"] = bindings
        f3_path = f3_dir / "raw" / "result.json"
        write_json(f3_path, f3)
        f3["detached_replay"] = run_detached_replay(
            args,
            case_id=FAULT_IDS[2],
            result_path=f3_path,
            output_path=f3_dir / "detached-replay.json",
        )
        fault_results.append(f3)
        print(json.dumps({"case_id": FAULT_IDS[2], "classification": f3["classification"]}), flush=True)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "complete",
        "clean": {
            "status": clean_result["status"],
            "false_positive_count": 0,
            "detached_replay_status": clean_replay["status"],
            "comparisons": clean_result["comparisons"],
            "reference_equivalence": clean_result["reference_equivalence"],
        },
        "fault_outcomes": [
            {
                "fault_id": item["case_id"],
                "classification": item["classification"],
                "catch_gate": item.get("catch_gate"),
                "full_horizon_reached": item["full_horizon_reached"],
                "semantic_comparisons": item.get("semantic_comparisons"),
                "detached_replay_status": item["detached_replay"]["status"],
            }
            for item in fault_results
        ],
        "fault_count": len(fault_results),
        "population_detection_rate_computed": False,
        "claim_boundary": {
            "per_fault_outcomes_only": True,
            "population_detection_rate_allowed": False,
            "heldout_population_claim_allowed": False,
            "F3_has_no_runtime_outcome_because_frozen_validity_precondition_failed": True,
            "candidate_import_free_replay_is_not_independent_live_recapture": True,
            "single_model_single_gpu_single_token_fixed_stack_only": True,
        },
    }
    write_json(args.run_dir / "summary.json", summary)
    ledger = terminal_ledger(args.run_dir)
    write_json(args.run_dir / "receipts" / "terminal-files.json", ledger)
    lines = "".join(f"{row['sha256']}  {row['path']}\n" for row in ledger)
    executor.write_bytes_atomic(args.run_dir / "receipts" / "terminal-files.sha256", lines.encode())
    print(json.dumps({"run_id": RUN_ID, "status": "complete"}), flush=True)
    return summary


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--prereg", type=Path, required=True)
    value.add_argument("--expected-prereg-sha256", required=True)
    value.add_argument("--author-freeze-manifest", type=Path, required=True)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--expected-execution-input-sha256", required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    value.add_argument("--reference-clean-result", type=Path, required=True)
    value.add_argument("--reference-clean-replay", type=Path, required=True)
    value.add_argument("--replay-source", type=Path, required=True)
    return value


args_global: argparse.Namespace


if __name__ == "__main__":
    run(parser().parse_args())
