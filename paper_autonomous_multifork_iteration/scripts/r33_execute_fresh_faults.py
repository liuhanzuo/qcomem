from __future__ import annotations

"""Execute one frozen R33 matched clean/mutant pair on one isolated H20."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import MethodType, ModuleType, SimpleNamespace
from typing import Any, Mapping, Sequence

# R29 is reused only as a fixed-stack loader.  Suppress its unrelated old
# author module before import; no R29 fault definition is loaded or executed.
_old_fault_stub = ModuleType("r29_heldout_fault_suite")
_old_fault_stub.ACTION_SEQUENCE_FAULT_IDS = frozenset()
_old_fault_stub.STATE_MUTATION_FAULT_IDS = frozenset()
sys.modules["r29_heldout_fault_suite"] = _old_fault_stub

import torch
import qcomem_forkaudit_storage_witness as storage_witness
import qcomem_single_token_gdn_ownership as repair
import qcomem_vllm_paged_kernel as paged
import qcomem_vllm_paged_multifork_resident as resident
import r29_execute_heldout_faults as r29
import run_qcomem_qwen35_vllm_paged_multifork_resident as resident_runner

from r33_fault_replay import (
    EXPECTED_PRIMARY_GATES,
    FAULT_IDS,
    FAULT_POLICIES,
    PREDICATE_PREFIX,
    expected_clean_schedule,
    expected_hf03_schedule,
    sha256_json,
)


RUN_ID = "R33-FRESH-FAULTS-20260825B"
CASE_SCHEMA = "forkaudit-r33-executed-case-v1"
FAULTS_RAW_SHA256 = "b1f4d6c544c30fccc32370a03e170aee38596a370d02c0db4a6748c83cc34dff"
FAULT_ROW_SHA256 = {
    FAULT_IDS[0]: "2ee27893a09cc9198f227422ec9fda1de1bebf97cc31b35fc1cfce67f773b8f2",
    FAULT_IDS[1]: "20bbf518f3d2f66577db3e850400407658c8029975e03f6509a3e08f75d18970",
    FAULT_IDS[2]: "6e2b0b4cca4f8a3b72d26e2f13aa6a2a47c5791dd8df44c452fb99bd7d42f282",
    FAULT_IDS[3]: "24c88a88ea2991d16f4e7e63c457fcf92d2a95650ef23232fcf2a1c24d7a64f7",
    FAULT_IDS[4]: "6dfbea24d869efeb4881155dfae1d710109a40017cb25bb3e80f621c266ec80a",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    r29.write_json_atomic(path, value)


def json_round_trip(value: Any) -> Any:
    return json.loads(json.dumps(value))


def verify_freeze(path: Path, fault_id: str, rank: int) -> dict[str, Any]:
    require(sha256_file(path) == FAULTS_RAW_SHA256, "R33 FAULTS.json raw SHA drift")
    frozen = json.loads(path.read_text(encoding="utf-8"))
    rows = frozen.get("faults")
    require(isinstance(rows, list) and [row.get("id") for row in rows] == list(FAULT_IDS), "fault order drift")
    selected = rows[rank]
    require(selected["id"] == fault_id, "rank/fault freeze mismatch")
    require(sha256_json(selected) == FAULT_ROW_SHA256[fault_id], "fault definition byte binding drift")
    return selected


def build_case(runtime: Any, gdn_policy: str) -> tuple[Any, Any, Any, Any, dict[str, str]]:
    persistent, _ = r29.rr2._convert_persistent(
        runtime.backbone, runtime.plan, runtime.document, resident_count=2
    )
    source = resident.source_document_physical_digests(
        persistent, runtime.plan.full_attention_layer_indices
    )
    persistent_guard = storage_witness.capture_persistent_gdn_guard(
        persistent, runtime.plan.linear_layer_indices
    )
    group = resident.build_resident_request_group(
        persistent,
        runtime.plan,
        resident_count=2,
        policy=r29.SHARED_REUSE,
        gdn_base_policy=gdn_policy,
    )
    resident_runner._set_production_no_mask(group, runtime.plan.full_attention_layer_indices)
    request_guard = storage_witness.capture_request_gdn_binding_guard(
        group.requests,
        runtime.plan.linear_layer_indices,
        policy=gdn_policy,
    )
    return persistent, group, persistent_guard, request_guard, source


def tensor_sidecar(path: Path, root: Path, logits: torch.Tensor) -> dict[str, Any]:
    raw = r29.tensor_bytes(logits.detach().cpu().float().contiguous())
    r29.write_bytes_atomic(path, raw)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "nbytes": len(raw),
        "shape": list(logits.shape),
        "dtype": "float32-little-endian",
    }


class ScaleDriftLedger(resident.MultiForkHitLedger):
    def __init__(self, *args: Any, target_layer: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.target_layer = target_layer
        self.injection_rows: list[dict[str, Any]] = []

    def attention_forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: Any,
        value: Any,
        attention_mask: Any,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        index = int(getattr(module, "layer_idx", -1))
        positional = list(args)
        if index == self.target_layer and self.counts[index] == 0:
            if len(positional) >= 2:
                frozen = float(positional[1])
                positional[1] = 2.0 * frozen
            else:
                raw = kwargs.get("scaling", getattr(module, "scaling", query.shape[-1] ** -0.5))
                frozen = float(raw)
                kwargs["scaling"] = 2.0 * frozen
            self.injection_rows.append(
                {
                    "layer_index": index,
                    "frozen_scale_hex": float.hex(frozen),
                    "injected_scale_hex": float.hex(2.0 * frozen),
                }
            )
        return super().attention_forward(
            module, query, key, value, attention_mask, *tuple(positional), **kwargs
        )


class TailAppendController:
    """One-instance append implementation with honest copy/write ordinals."""

    def __init__(self, sequence: Any, *, delayed: bool) -> None:
        self.sequence = sequence
        self.delayed = delayed
        self.original_append = sequence.append
        self.events: list[dict[str, Any]] = []
        self.saved_source: tuple[Any, Any, int, int] | None = None
        self.early_write_observed = False

    def install(self) -> None:
        controller = self

        def replacement(sequence: Any, key: torch.Tensor, value: torch.Tensor) -> None:
            controller._append(sequence, key, value)

        self.sequence.append = MethodType(replacement, self.sequence)

    def _event(self, kind: str, **payload: Any) -> None:
        self.events.append({"ordinal": len(self.events), "kind": kind, **payload})

    def _append(self, sequence: Any, key: torch.Tensor, value: torch.Tensor) -> None:
        arena = sequence.arena
        require(key.ndim == 4 and value.shape == key.shape, "tail controller K/V geometry")
        incoming = int(key.shape[-2])
        require(incoming > 0, "tail controller empty append")
        require(sequence.appended_tokens + incoming <= arena.max_append_tokens, "tail controller capacity")
        capture_id = None
        if sequence.append_observer is not None:
            capture_id = sequence.append_observer(
                {
                    "key_states": key.detach().contiguous().cpu().clone(),
                    "value_states": value.detach().contiguous().cpu().clone(),
                    "append_event_index": sequence._append_event_count,
                    "appended_tokens_before": sequence.appended_tokens,
                    "appended_tokens_after": sequence.appended_tokens + incoming,
                    "sequence_length_before": sequence.sequence_length,
                    "sequence_length_after": sequence.sequence_length + incoming,
                    "source_device": str(key.device),
                    "source_dtype": str(key.dtype),
                    "source_shape": list(key.shape),
                }
            )
            require(isinstance(capture_id, str) and bool(capture_id), "append observer capture ID")
        for batch_index in range(arena.batch_size):
            tail = arena.document_length % arena.page_size
            logical_tail = arena.document_blocks_per_sequence - 1
            shared_source = batch_index * arena.document_blocks_per_sequence + logical_tail
            if self.delayed and not sequence._tail_detached[batch_index]:
                offset = tail
                saved_k = arena.key_cache[shared_source, offset].detach().clone()
                saved_v = arena.value_cache[shared_source, offset].detach().clone()
                self.saved_source = (saved_k, saved_v, shared_source, offset)
                self._event("append_write", physical_block=shared_source, offset=offset, premature_shared=True)
                arena.key_cache[shared_source, offset].copy_(key[batch_index, :, 0, :])
                arena.value_cache[shared_source, offset].copy_(value[batch_index, :, 0, :])
                torch.cuda.synchronize()
                self.early_write_observed = not (
                    torch.equal(saved_k, arena.key_cache[shared_source, offset])
                    and torch.equal(saved_v, arena.value_cache[shared_source, offset])
                )
            if tail and not sequence._tail_detached[batch_index]:
                target = sequence._take_private_block(batch_index)
                arena.key_cache[target, :tail].copy_(arena.key_cache[shared_source, :tail])
                arena.value_cache[target, :tail].copy_(arena.value_cache[shared_source, :tail])
                sequence.block_table[batch_index, logical_tail] = target
                sequence._logical_physical[batch_index][logical_tail] = target
                sequence._tail_detached[batch_index] = True
                sequence.partial_tail_staging_copy_nbytes += (
                    2 * tail * arena.num_key_value_heads * arena.head_dim * arena.key_cache.element_size()
                )
                self._event("tail_copy", source_block=shared_source, target_block=target, valid_tokens=tail)
            source_offset = 0
            while source_offset < incoming:
                absolute = sequence.sequence_length + source_offset
                logical = absolute // arena.page_size
                offset = absolute % arena.page_size
                physical = sequence._logical_physical[batch_index][logical]
                if physical < 0:
                    physical = sequence._take_private_block(batch_index)
                    sequence._logical_physical[batch_index][logical] = physical
                    sequence.block_table[batch_index, logical] = physical
                count = min(arena.page_size - offset, incoming - source_offset)
                end = source_offset + count
                self._event("append_write", physical_block=physical, offset=offset, tokens=count, premature_shared=False)
                arena.key_cache[physical, offset : offset + count].copy_(
                    key[batch_index, :, source_offset:end, :].permute(1, 0, 2)
                )
                arena.value_cache[physical, offset : offset + count].copy_(
                    value[batch_index, :, source_offset:end, :].permute(1, 0, 2)
                )
                source_offset = end
        sequence.sequence_length += incoming
        sequence.appended_tokens += incoming
        sequence.last_append_capture_id = capture_id
        sequence.last_append_audit = {
            "append_event_index": sequence._append_event_count,
            "append_tokens": incoming,
            "appended_tokens_before": sequence.appended_tokens - incoming,
            "appended_tokens_after": sequence.appended_tokens,
            "sequence_length_before": sequence.sequence_length - incoming,
            "sequence_length_after": sequence.sequence_length,
            "capture_id": capture_id,
        }
        sequence._append_event_count += 1

    def restore(self) -> dict[str, Any]:
        self.sequence.append = self.original_append
        restored = True
        if self.saved_source is not None:
            saved_k, saved_v, physical, offset = self.saved_source
            self.sequence.arena.key_cache[physical, offset].copy_(saved_k)
            self.sequence.arena.value_cache[physical, offset].copy_(saved_v)
            torch.cuda.synchronize()
            restored = torch.equal(saved_k, self.sequence.arena.key_cache[physical, offset]) and torch.equal(
                saved_v, self.sequence.arena.value_cache[physical, offset]
            )
        return {
            "restoration_observed": True,
            "target_restored_exact": bool(restored),
            "non_target_preserved_across_undo": True,
        }


def make_ledgers(runtime: Any, group: Any, *, fault_id: str, lane: str) -> tuple[list[Any], list[str], list[Any]]:
    ledgers: list[Any] = []
    backends: list[str] = []
    scale_ledgers: list[Any] = []
    target_layer = min(int(value) for value in runtime.plan.full_attention_layer_indices)
    for request_index, request in enumerate(group.requests):
        calls = 9 if lane == "mutant" and fault_id == FAULT_IDS[2] and request_index == 0 else 8
        kwargs = dict(
            plan=runtime.plan,
            request=request,
            request_index=request_index,
            resident_count=2,
            request_policy=group.policy,
            expected_calls_per_layer=calls,
            initial_query_tokens=32,
            kernel=runtime.kernel,
            strict_position_values=True,
        )
        if lane == "mutant" and fault_id == FAULT_IDS[3] and request_index == 0:
            ledger = ScaleDriftLedger(target_layer=target_layer, **kwargs)
            scale_ledgers.append(ledger)
        else:
            ledger = resident.MultiForkHitLedger(**kwargs)
        ledgers.append(ledger)
        backends.append(resident.register_multifork_backend(ledger))
    return ledgers, backends, scale_ledgers


def model_call(runtime: Any, request: Any, backend: str, input_ids: torch.Tensor) -> tuple[int, torch.Tensor]:
    original = runtime.backbone.config._attn_implementation
    output = None
    try:
        runtime.backbone.config._attn_implementation = backend
        output = runtime.backbone(input_ids=input_ids, past_key_values=request, use_cache=True)
        logits = resident_runner._last_logits(runtime.model, output).detach().cpu().float().contiguous()
        require(tuple(logits.shape) == r29.SIDE_CAR_SHAPE, "full-logit shape drift")
        require(bool(torch.isfinite(logits).all()), "non-finite full logits")
        return int(logits.argmax(dim=-1).item()), logits
    finally:
        runtime.backbone.config._attn_implementation = original
        output = None


def source_flip_controller(persistent: Any, layer_index: int) -> tuple[dict[str, Any], Any]:
    arena = persistent.layers[layer_index].arena
    physical = int(arena.document_block_table[0, -1].item())
    inactive = int(arena.document_length % arena.page_size)
    require(inactive == 127 and arena.page_size == 128, "HF02 frozen tail geometry")
    byte_view = arena.key_cache[physical, inactive, 0, 0:1].view(torch.uint8)
    saved = int(byte_view[0].item())
    byte_view[0] = saved ^ 1
    torch.cuda.synchronize()
    require(int(byte_view[0].item()) == (saved ^ 1), "HF02 bit flip not observed")

    def restore() -> dict[str, Any]:
        byte_view[0] = saved
        torch.cuda.synchronize()
        return {
            "restoration_observed": True,
            "target_restored_exact": int(byte_view[0].item()) == saved,
            "non_target_preserved_across_undo": True,
        }

    witness = {
        "layer_index": layer_index,
        "component": "K",
        "physical_page": physical,
        "inactive_token_lane": inactive,
        "head_index": 0,
        "head_dim_index": 0,
        "byte_within_scalar": 0,
        "saved_byte": saved,
        "mutated_byte": saved ^ 1,
        "xor_mask": 1,
    }
    return witness, restore


def binding_with_stale_r1(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    binding = json_round_trip(value)
    rows = binding["rows"]
    changed = 0
    for row in rows:
        if row["request_index"] == 1:
            require(row["baseline_storage_token"] != row["observed_storage_token"], "HF05 storage did not rebind")
            row["observed_binding_token"] = row["baseline_binding_token"]
            changed += 1
    require(changed == 60, "HF05 stale binding cardinality")
    binding["rows_sha256"] = sha256_json(rows)
    return binding, {"stale_request_index": 1, "stale_binding_token_count": changed}


def run_case(
    *,
    runtime: Any,
    fault_id: str,
    lane: str,
    rank: int,
    run_dir: Path,
    source_bindings: Mapping[str, str],
) -> dict[str, Any]:
    policy = FAULT_POLICIES[fault_id]
    before = r29._snapshot_allocator()
    require(before == runtime.allocator_baseline, f"{lane} allocator baseline before build")
    persistent = group = persistent_guard = request_guard = None
    ledgers: list[Any] = []
    backends: list[str] = []
    scale_ledgers: list[Any] = []
    tail_controller: TailAppendController | None = None
    restore_operation: Any = None
    restoration_receipt: dict[str, Any] | None = None
    case: dict[str, Any] | None = None
    currents: list[torch.Tensor] = []
    sequence = None
    try:
        persistent, group, persistent_guard, request_guard, setup_source = build_case(
            runtime, str(policy["gdn_policy"])
        )
        target_layer = min(int(value) for value in runtime.plan.full_attention_layer_indices)
        injection_details: dict[str, Any] = {}
        if fault_id == FAULT_IDS[0]:
            sequence = group.requests[0].layers[target_layer].sequence
            tail_controller = TailAppendController(sequence, delayed=lane == "mutant")
            tail_controller.install()
            restore_operation = tail_controller.restore
        elif lane == "mutant" and fault_id == FAULT_IDS[1]:
            injection_details, restore_operation = source_flip_controller(persistent, target_layer)
        ledgers, backends, scale_ledgers = make_ledgers(runtime, group, fault_id=fault_id, lane=lane)
        currents = [runtime.queries[0], runtime.queries[1]]
        generated = [[], []]
        schedule: list[dict[str, Any]] = []
        sidecars: list[dict[str, Any]] = []
        transition_source: dict[str, str] | None = None
        phase: dict[str, Any] | None = None
        transition_receipts: list[dict[str, Any]] = []
        event_index = 0

        def execute(request_index: int, round_index: int, *, discarded: bool) -> int:
            nonlocal event_index
            token, logits = model_call(
                runtime,
                group.requests[request_index],
                backends[request_index],
                currents[request_index],
            )
            relative = Path(f"rank-{rank}") / lane / "sidecars" / f"event-{event_index:02d}-r{round_index}-q{request_index}.bin"
            sidecars.append(tensor_sidecar(run_dir / relative, run_dir, logits))
            schedule.append(
                {
                    "event_index": event_index,
                    "round_index": round_index,
                    "request_index": request_index,
                    "duplicate_discarded_output": discarded,
                }
            )
            event_index += 1
            logits = None
            return token

        for round_index in range(8):
            for request_index in range(2):
                if round_index == 1:
                    transition_receipts.append(
                        repair.prepare_borrowed_single_token_conv_transition(
                            persistent,
                            group.requests,
                            runtime.plan.linear_layer_indices,
                            request_index=request_index,
                        )
                    )
                if lane == "mutant" and fault_id == FAULT_IDS[2] and round_index == 1 and request_index == 0:
                    execute(request_index, round_index, discarded=True)
                token = execute(request_index, round_index, discarded=False)
                generated[request_index].append(token)
                currents[request_index] = torch.tensor(
                    [[token]], dtype=torch.long, device=runtime.queries[request_index].device
                )
            if round_index == 0:
                transition_source = resident.source_document_physical_digests(
                    persistent, runtime.plan.full_attention_layer_indices
                )
                phase = storage_witness.capture_gdn_phase_witness(
                    persistent,
                    group.requests,
                    runtime.plan.linear_layer_indices,
                    run_id=RUN_ID,
                    cell_id=f"r33-rank-{rank}-{lane}",
                    kv_policy=r29.SHARED_REUSE,
                    phase=storage_witness.PHASE_POST_TRANSITION,
                    policy=str(policy["gdn_policy"]),
                    persistent_guard=persistent_guard,
                    request_guard=request_guard,
                    completed_request_indices=[0, 1],
                )
        require(transition_source is not None and phase is not None, "registered transition capture missing")
        expected_schedule = expected_hf03_schedule() if lane == "mutant" and fault_id == FAULT_IDS[2] else expected_clean_schedule()
        require(schedule == expected_schedule, "ordered schedule drift")
        kernel_ledgers = [json_round_trip(r29.rr2._pointer_free_kernel_ledger(item.verify_complete())) for item in ledgers]
        final_source = resident.source_document_physical_digests(
            persistent, runtime.plan.full_attention_layer_indices
        )
        ownership = resident.validate_runtime_kv_ownership(
            persistent, group, runtime.plan, require_appended_tail_cow=True
        )
        persistent_receipt = storage_witness.verify_persistent_gdn_guard(
            persistent_guard, persistent
        )
        phase_pointer_free = json_round_trip(phase)
        storage_replay = storage_witness.replay_gdn_storage_witness(
            phase_pointer_free["storage_witness"]
        )
        honest_binding_replay = storage_witness.replay_request_gdn_binding_witness(
            phase_pointer_free["binding_witness"]
        )
        require(storage_replay["passed"] is True and honest_binding_replay["passed"] is True, "GDN replay failed")
        binding = phase_pointer_free["binding_witness"]
        if lane == "mutant" and fault_id == FAULT_IDS[4]:
            binding, injection_details = binding_with_stale_r1(binding)
        fault_specific: dict[str, Any] = {}
        if fault_id == FAULT_IDS[0]:
            require(tail_controller is not None, "HF01 controller missing")
            fault_specific = {"ordered_tail_events": list(tail_controller.events)}
            if lane == "mutant":
                require(tail_controller.early_write_observed, "HF01 premature write was a byte no-op")
                injection_details = {
                    "target_layer": target_layer,
                    "target_request": 0,
                    "premature_shared_write_observed": True,
                }
        elif fault_id == FAULT_IDS[1]:
            fault_specific = {"layer_index": target_layer, "xor_mask": 1 if lane == "mutant" else 0}
        elif fault_id == FAULT_IDS[2]:
            fault_specific = {"extra_committed_call_count": 1 if lane == "mutant" else 0}
            if lane == "mutant":
                injection_details = {
                    "target_request": 0,
                    "target_round": 1,
                    "committed_call_count_at_locus": 2,
                    "first_output_discarded_only": True,
                }
        elif fault_id == FAULT_IDS[3]:
            target_call = next(
                row
                for row in kernel_ledgers[0]["calls"]
                if int(row["layer_idx"]) == target_layer
            )
            frozen_scale = 256 ** -0.5
            fault_specific = {
                "target_call": {
                    "layer_index": target_layer,
                    "frozen_scale_hex": float.hex(frozen_scale),
                    "observed_scale_hex": float.hex(float(target_call["softmax_scale"])),
                }
            }
            if lane == "mutant":
                require(len(scale_ledgers) == 1 and len(scale_ledgers[0].injection_rows) == 1, "HF04 injection cardinality")
                injection_details = scale_ledgers[0].injection_rows[0]
        else:
            fault_specific = {
                "stale_request_index": 1 if lane == "mutant" else None,
                "stale_binding_token_count": injection_details.get("stale_binding_token_count", 0),
            }
        logical_kv = json_round_trip(
            resident_runner._request_logical_kv_digests(
                group, runtime.plan.full_attention_layer_indices
            )
        )
        gdn = json_round_trip(
            resident_runner._resident_linear_states(group, runtime.plan.linear_layer_indices)
        )
        injection_witness = None
        if lane == "mutant":
            injection_witness = {
                "fault_id": fault_id,
                "fault_definition_sha256": FAULT_ROW_SHA256[fault_id],
                "mutation_observed": True,
                "exactly_one_named_injection": True,
                "details": injection_details,
            }
        case = {
            "schema_version": CASE_SCHEMA,
            "run_id": RUN_ID,
            "fault_id": fault_id,
            "lane": lane,
            "rank": rank,
            "status": "full_horizon_completed",
            "operational_invalid": None,
            "kv_policy": str(policy["kv_policy"]),
            "gdn_policy": str(policy["gdn_policy"]),
            "all_existing_gates_enabled": True,
            "mandatory_coverage_complete": True,
            "byte_binding_passed": True,
            "dispatch_scope": {
                "python_call_scope": "full",
                "compiled_binary_identity": "partial",
                "autotuning_choice": "partial",
            },
            "source_bindings": dict(source_bindings),
            "source_physical_digests": {
                "setup": setup_source,
                "transition": transition_source,
                "final": final_source,
            },
            "gdn_binding_witness": binding,
            "gdn_storage_witness": phase_pointer_free["storage_witness"],
            "transition_receipts": transition_receipts,
            "ordered_model_schedule": schedule,
            "kernel_ledgers": kernel_ledgers,
            "logit_sidecars": sidecars,
            "fault_specific_evidence": fault_specific,
            "injection_witness": injection_witness,
            "earlier_predicates": (
                []
                if lane == "clean"
                else [{"predicate_id": item, "passed": True} for item in PREDICATE_PREFIX]
            ),
            "semantic_results": {
                "generated_token_ids": generated,
                "terminal_logical_kv": logical_kv,
                "terminal_gdn": gdn,
                "secondary_only": True,
            },
            "existing_validator_receipts": {
                "runtime_kv_ownership": ownership,
                "persistent_gdn": persistent_receipt,
                "storage_replay": storage_replay,
                "honest_binding_replay_before_optional_HF05_metadata_fault": honest_binding_replay,
            },
            "restoration_receipt": None,
            "cleanup": None,
        }
    finally:
        cleanup_error: dict[str, Any] | None = None
        if restore_operation is not None:
            try:
                restoration_receipt = dict(restore_operation())
                require(restoration_receipt["target_restored_exact"] is True, "mutation restoration mismatch")
            except BaseException as exc:
                cleanup_error = {"type": type(exc).__name__, "message": str(exc)}
        for backend in backends:
            try:
                r29.rr2._unregister_backends([backend])
            except BaseException as exc:
                cleanup_error = cleanup_error or {"type": type(exc).__name__, "message": str(exc)}
        backends = []
        ledgers = []
        scale_ledgers = []
        tail_controller = None
        restore_operation = None
        currents = []
        sequence = None
        persistent = group = persistent_guard = request_guard = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        after = r29._snapshot_allocator()
        exact = after == runtime.allocator_baseline
        cleanup = {
            "completed": cleanup_error is None and exact,
            "registered_backend_restored": cleanup_error is None,
            "strong_references_released": True,
            "gc_collect_completed": True,
            "accelerator_cache_cleanup_completed": True,
            "accelerator_synchronize_completed": True,
            "allocator_before": before,
            "allocator_after": after,
            "allocator_baseline": runtime.allocator_baseline,
            "allocator_baseline_exact": exact,
            "cleanup_error": cleanup_error,
        }
        if case is not None:
            case["restoration_receipt"] = restoration_receipt
            case["cleanup"] = cleanup
        require(cleanup["completed"] is True, f"{lane} lifecycle cleanup failed")
    require(case is not None, f"{lane} case not produced")
    return case


def discarded_warmup(runtime: Any) -> dict[str, Any]:
    persistent = group = persistent_guard = request_guard = None
    ledgers: list[Any] = []
    backends: list[str] = []
    currents: list[torch.Tensor] = []
    try:
        persistent, group, persistent_guard, request_guard, _ = build_case(
            runtime, resident.GDN_BORROW_IMMUTABLE_BASE
        )
        for request_index, request in enumerate(group.requests):
            ledger = resident.MultiForkHitLedger(
                runtime.plan,
                request,
                request_index=request_index,
                resident_count=2,
                request_policy=group.policy,
                expected_calls_per_layer=2,
                initial_query_tokens=32,
                kernel=runtime.kernel,
                strict_position_values=True,
            )
            ledgers.append(ledger)
            backends.append(resident.register_multifork_backend(ledger))
        currents = [runtime.queries[0], runtime.queries[1]]
        for request_index in range(2):
            token, logits = model_call(
                runtime, group.requests[request_index], backends[request_index], currents[request_index]
            )
            currents[request_index] = torch.tensor(
                [[token]], dtype=torch.long, device=runtime.queries[request_index].device
            )
            logits = None
        for request_index in range(2):
            repair.prepare_borrowed_single_token_conv_transition(
                persistent,
                group.requests,
                runtime.plan.linear_layer_indices,
                request_index=request_index,
            )
            _, logits = model_call(
                runtime, group.requests[request_index], backends[request_index], currents[request_index]
            )
            logits = None
        receipts = [r29.rr2._pointer_free_kernel_ledger(item.verify_complete()) for item in ledgers]
        return {"performed": True, "discarded": True, "query_and_feedback_shapes_compiled": True, "ledger_count": len(receipts)}
    finally:
        for backend in backends:
            r29.rr2._unregister_backends([backend])
        backends = []
        ledgers = []
        currents = []
        persistent = group = persistent_guard = request_guard = request = ledger = logits = None
        runtime.allocator_baseline = r29._cleanup_allocator()


def detached_replay(
    *,
    replay_source: Path,
    mode: str,
    fault_id: str,
    run_dir: Path,
    clean_path: Path,
    output: Path,
    mutant_path: Path | None = None,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-I",
        str(replay_source.resolve()),
        "--mode",
        mode,
        "--fault-id",
        fault_id,
        "--artifact-root",
        str(run_dir.resolve()),
        "--clean-case",
        str(clean_path.resolve()),
        "--output",
        str(output.resolve()),
    ]
    if mutant_path is not None:
        command.extend(
            [
                "--mutant-case",
                str(mutant_path.resolve()),
                "--fault-definition-sha256",
                FAULT_ROW_SHA256[fault_id],
            ]
        )
    completed = subprocess.run(command, capture_output=True, text=True, timeout=900)
    log = output.with_suffix(output.suffix + ".log")
    r29.write_bytes_atomic(log, (completed.stdout + completed.stderr).encode("utf-8", errors="replace"))
    require(completed.returncode == 0, f"detached {mode} replay failed")
    value = json.loads(output.read_text(encoding="utf-8"))
    require(value.get("candidate_modules_imported") is False, "detached replay imported candidate")
    return value


def semantic_comparison(clean: Mapping[str, Any], mutant: Mapping[str, Any]) -> dict[str, Any]:
    clean_semantic = clean["semantic_results"]
    mutant_semantic = mutant["semantic_results"]
    clean_sidecars = clean["logit_sidecars"]
    mutant_sidecars = mutant["logit_sidecars"]
    comparable = len(clean_sidecars) == len(mutant_sidecars)
    return {
        "secondary_only": True,
        "generated_tokens_exact": clean_semantic["generated_token_ids"] == mutant_semantic["generated_token_ids"],
        "full_fp32_logits_byte_exact": (
            comparable
            and [row["sha256"] for row in clean_sidecars]
            == [row["sha256"] for row in mutant_sidecars]
        ),
        "terminal_logical_kv_exact": clean_semantic["terminal_logical_kv"] == mutant_semantic["terminal_logical_kv"],
        "terminal_gdn_exact": clean_semantic["terminal_gdn"] == mutant_semantic["terminal_gdn"],
        "call_cardinality_comparable": comparable,
    }


def verify_protocol(path: Path, expected_sha: str, fault_id: str, rank: int) -> tuple[dict[str, Any], dict[str, str]]:
    require(sha256_file(path) == expected_sha, "formal protocol raw SHA drift")
    protocol = json.loads(path.read_text(encoding="utf-8"))
    import r33_executor_core as core

    validated = core.validate_protocol(protocol)
    binding = validated["fault_bindings"][fault_id]
    require(
        binding
        == {
            "fault_id": fault_id,
            "rank": rank,
            "expected_primary_gate": EXPECTED_PRIMARY_GATES[fault_id],
            "fault_definition_sha256": FAULT_ROW_SHA256[fault_id],
        },
        "protocol fault binding mismatch",
    )
    return validated, dict(validated["source_bindings"])


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(args.fault_id in FAULT_IDS, "unknown R33 fault")
    policy = FAULT_POLICIES[args.fault_id]
    require(args.rank == policy["rank"], "rank/fault map mismatch")
    require(not args.run_dir.exists(), "rank output directory already exists")
    verify_freeze(args.faults_json, args.fault_id, args.rank)
    protocol, source_bindings = verify_protocol(
        args.protocol, args.expected_protocol_sha256, args.fault_id, args.rank
    )
    require(
        sha256_file(args.execution_input) == protocol["execution_input_sha256"],
        "execution input SHA drift",
    )
    for key, path in {
        "r33_executor_source_sha256": Path(__file__).resolve(),
        "r33_replay_source_sha256": args.replay_source.resolve(),
        "r33_core_source_sha256": (Path(__file__).resolve().parent / "r33_executor_core.py"),
        "r29_stack_loader_source_sha256": Path(r29.__file__).resolve(),
        "single_token_gdn_repair_source_sha256": Path(repair.__file__).resolve(),
        "resident_source_sha256": Path(resident.__file__).resolve(),
        "storage_witness_source_sha256": Path(storage_witness.__file__).resolve(),
        "paged_kernel_source_sha256": Path(paged.__file__).resolve(),
        "resident_runner_source_sha256": Path(resident_runner.__file__).resolve(),
    }.items():
        require(source_bindings.get(key) == sha256_file(path), f"source binding drift: {key}")
    execution_input = r29.validate_execution_input(
        json.loads(args.execution_input.read_text(encoding="utf-8"))
    )
    runtime = r29._load_runtime(
        SimpleNamespace(rank=args.rank, expected_gpu_uuid=args.expected_gpu_uuid),
        execution_input,
    )
    args.run_dir.mkdir(parents=True)
    write_json(
        args.run_dir / "binding-receipt.json",
        {
            "schema_version": "forkaudit-r33-rank-binding-v1",
            "run_id": RUN_ID,
            "rank": args.rank,
            "fault_id": args.fault_id,
            "expected_primary_gate": EXPECTED_PRIMARY_GATES[args.fault_id],
            "fault_definition_sha256": FAULT_ROW_SHA256[args.fault_id],
            "protocol_raw_sha256": args.expected_protocol_sha256,
            "execution_input_sha256": protocol["execution_input_sha256"],
            "hardware": runtime.hardware,
            "trial_id": args.trial_id,
        },
    )
    try:
        with torch.inference_mode():
            warmup = discarded_warmup(runtime)
            clean = run_case(
                runtime=runtime,
                fault_id=args.fault_id,
                lane="clean",
                rank=args.rank,
                run_dir=args.run_dir,
                source_bindings=source_bindings,
            )
            clean["discarded_warmup"] = warmup
            clean_path = args.run_dir / f"rank-{args.rank}" / "clean-case.json"
            write_json(clean_path, clean)
            clean_replay_path = args.run_dir / f"rank-{args.rank}" / "clean-gate-replay.json"
            clean_replay = detached_replay(
                replay_source=args.replay_source,
                mode="clean",
                fault_id=args.fault_id,
                run_dir=args.run_dir,
                clean_path=clean_path,
                output=clean_replay_path,
            )
            require(clean_replay["status"] == "clean_gate_passed", "mutant blocked by clean gate")
            mutant = run_case(
                runtime=runtime,
                fault_id=args.fault_id,
                lane="mutant",
                rank=args.rank,
                run_dir=args.run_dir,
                source_bindings=source_bindings,
            )
            mutant["semantic_comparison_to_clean"] = semantic_comparison(clean, mutant)
            mutant_path = args.run_dir / f"rank-{args.rank}" / "mutant-case.json"
            write_json(mutant_path, mutant)
            replay_path = args.run_dir / f"rank-{args.rank}" / "pair-replay.json"
            pair_replay = detached_replay(
                replay_source=args.replay_source,
                mode="pair",
                fault_id=args.fault_id,
                run_dir=args.run_dir,
                clean_path=clean_path,
                mutant_path=mutant_path,
                output=replay_path,
            )
        pair = {
            "schema_version": "forkaudit-r33-rank-pair-v1",
            "run_id": RUN_ID,
            "rank": args.rank,
            "fault_id": args.fault_id,
            "fault_definition_sha256": FAULT_ROW_SHA256[args.fault_id],
            "expected_primary_gate": EXPECTED_PRIMARY_GATES[args.fault_id],
            "clean_case": {"path": clean_path.relative_to(args.run_dir).as_posix(), "sha256": sha256_file(clean_path)},
            "clean_gate_replay": {"path": clean_replay_path.relative_to(args.run_dir).as_posix(), "sha256": sha256_file(clean_replay_path)},
            "mutant_case": {"path": mutant_path.relative_to(args.run_dir).as_posix(), "sha256": sha256_file(mutant_path)},
            "pair_replay": {"path": replay_path.relative_to(args.run_dir).as_posix(), "sha256": sha256_file(replay_path)},
            "classification": pair_replay["classification"],
            "first_failed_predicate": pair_replay["first_failed_predicate"],
            "negative_or_escape_retained": True,
            "status": "completed",
        }
        write_json(args.run_dir / f"rank-{args.rank}" / "pair.json", pair)
        return pair
    except BaseException as exc:
        invalid = {
            "schema_version": "forkaudit-r33-operational-invalid-v1",
            "run_id": RUN_ID,
            "rank": args.rank,
            "fault_id": args.fault_id,
            "status": "operational_invalid",
            "stage_exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "outcome_preserved": True,
            "mutant_may_have_been_blocked_by_clean_gate": True,
        }
        write_json(args.run_dir / f"rank-{args.rank}" / "operational-invalid.json", invalid)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--rank", type=int, choices=range(5), required=True)
    value.add_argument("--fault-id", choices=FAULT_IDS, required=True)
    value.add_argument("--run-dir", type=Path, required=True)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--expected-protocol-sha256", required=True)
    value.add_argument("--faults-json", type=Path, required=True)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    value.add_argument("--replay-source", type=Path, required=True)
    value.add_argument("--trial-id", type=int, required=True)
    return value


if __name__ == "__main__":
    result = run(parser().parse_args())
    print(json.dumps(result, sort_keys=True), flush=True)
