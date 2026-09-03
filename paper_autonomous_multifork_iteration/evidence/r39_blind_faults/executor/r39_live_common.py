#!/usr/bin/env python3
"""Live H20 primitives for the R39 blind-fault executor.

The frozen detector implementation is imported unchanged.  R39 mutations are
installed only on request-local objects and are described by authenticated
receipts; no ForkAudit predicate is added or weakened here.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

# R29 is used solely as the identity-bound stack loader.  Its old held-out fault
# module is prevented from loading, exactly as in the audited R33 executor.
_old_fault_stub = ModuleType("r29_heldout_fault_suite")
_old_fault_stub.ACTION_SEQUENCE_FAULT_IDS = frozenset()
_old_fault_stub.STATE_MUTATION_FAULT_IDS = frozenset()
sys.modules["r29_heldout_fault_suite"] = _old_fault_stub

import torch

import qcomem_forkaudit_storage_witness as storage_witness
import qcomem_joint_policy as joint_policy
import qcomem_single_token_gdn_ownership as repair
import qcomem_vllm_paged_multifork_resident as resident
import r29_execute_heldout_faults as r29
import run_qcomem_qwen35_forkaudit_review_revision as rr2
import run_qcomem_qwen35_vllm_paged_multifork_resident as resident_runner


BORROW = resident.GDN_BORROW_IMMUTABLE_BASE
SHARED = r29.SHARED_REUSE
SIDE_CAR_SHAPE = (1, 248320)


class LiveError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LiveError(message)


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha(value: torch.Tensor) -> str:
    return hashlib.sha256(tensor_bytes(value)).hexdigest()


def storage_key(value: torch.Tensor) -> tuple[str, int, int]:
    storage = value.untyped_storage()
    return str(value.device), int(storage.data_ptr()), int(storage.nbytes())


def byte_interval(value: torch.Tensor) -> tuple[int, int]:
    return repair.byte_interval(value)


def tensor_descriptor(value: torch.Tensor) -> dict[str, Any]:
    start, end = byte_interval(value)
    return {
        "shape": [int(item) for item in value.shape],
        "stride": [int(item) for item in value.stride()],
        "storage_offset": int(value.storage_offset()),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "element_size": int(value.element_size()),
        "tensor_nbytes": int(value.numel()) * int(value.element_size()),
        "storage_nbytes": int(value.untyped_storage().nbytes()),
        "byte_start": start,
        "byte_end_exclusive": end,
        "content_sha256": tensor_sha(value),
    }


def allocator_snapshot(*, reset_peak: bool = False) -> dict[str, int]:
    torch.cuda.synchronize()
    if reset_peak:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def gdn_coordinates(owner: Any, layer_indices: Iterable[int]) -> list[tuple[int, str, int, torch.Tensor]]:
    rows: list[tuple[int, str, int, torch.Tensor]] = []
    for layer_index in sorted(int(item) for item in layer_indices):
        layer = owner.layers[layer_index]
        for family in ("conv_states", "recurrent_states"):
            values = getattr(layer, family)
            require(isinstance(values, dict), f"GDN family absent: {layer_index}/{family}")
            for state_index in sorted(values):
                tensor = values[state_index]
                require(isinstance(tensor, torch.Tensor), "GDN coordinate is not tensor")
                rows.append((layer_index, family, int(state_index), tensor))
    return rows


def gdn_digest_map(owner: Any, layer_indices: Iterable[int]) -> dict[str, str]:
    return {
        f"{layer_index}:{family}:{state_index}": tensor_sha(tensor)
        for layer_index, family, state_index, tensor in gdn_coordinates(owner, layer_indices)
    }


def persistent_digest_bundle(persistent: Any, plan: Any) -> dict[str, Any]:
    kv = resident.source_document_physical_digests(
        persistent, plan.full_attention_layer_indices
    )
    gdn = gdn_digest_map(persistent, plan.linear_layer_indices)
    return {
        "document_kv": kv,
        "persistent_gdn": gdn,
        "document_kv_manifest_sha256": hashlib.sha256(
            json.dumps(kv, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "persistent_gdn_manifest_sha256": hashlib.sha256(
            json.dumps(gdn, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def load_runtime(*, input_rank: int, expected_gpu_uuid: str, execution_input: Mapping[str, Any]) -> Any:
    require(0 <= input_rank < 8, "frozen input rank outside eight-book set")
    runtime = r29._load_runtime(
        SimpleNamespace(rank=input_rank, expected_gpu_uuid=expected_gpu_uuid),
        execution_input,
    )
    return runtime


def discarded_warmup(runtime: Any) -> dict[str, Any]:
    """Compile query and one-token paths without retaining semantic outputs."""

    persistent = group = None
    ledgers: list[Any] = []
    backends: list[str] = []
    currents: list[torch.Tensor] = []
    output = logits = None
    try:
        persistent, _ = rr2._convert_persistent(
            runtime.backbone, runtime.plan, runtime.document, resident_count=2
        )
        group = resident.build_resident_request_group(
            persistent, runtime.plan, resident_count=2, policy=SHARED,
            gdn_base_policy=BORROW,
        )
        resident_runner._set_production_no_mask(
            group, runtime.plan.full_attention_layer_indices
        )
        for request_index, request in enumerate(group.requests):
            ledger = resident.MultiForkHitLedger(
                runtime.plan, request, request_index=request_index,
                resident_count=2, request_policy=SHARED,
                expected_calls_per_layer=2, initial_query_tokens=32,
                kernel=runtime.kernel, strict_position_values=True,
            )
            ledgers.append(ledger)
            backends.append(resident.register_multifork_backend(ledger))
        currents = [runtime.queries[0], runtime.queries[1]]
        for request_index in range(2):
            _, logits = model_call(runtime, group.requests[request_index], backends[request_index], currents[request_index])
            token = int(logits.argmax(dim=-1).item())
            currents[request_index] = torch.tensor([[token]], dtype=torch.long, device="cuda:0")
        for request_index in range(2):
            repair.prepare_borrowed_single_token_conv_transition(
                persistent, group.requests, runtime.plan.linear_layer_indices,
                request_index=request_index,
            )
            _, logits = model_call(runtime, group.requests[request_index], backends[request_index], currents[request_index])
        for ledger in ledgers:
            ledger.verify_complete()
        return {
            "performed": True,
            "semantic_outputs_persisted": False,
            "query_and_feedback_paths_primed": True,
        }
    finally:
        if backends:
            rr2._unregister_backends(backends)
        persistent = group = output = logits = None
        ledgers = []
        backends = []
        currents = []
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        runtime.allocator_baseline = r29._snapshot_allocator()


def build_default_case(runtime: Any) -> tuple[Any, Any, Any, Any]:
    persistent, _ = rr2._convert_persistent(
        runtime.backbone, runtime.plan, runtime.document, resident_count=2
    )
    persistent_guard = storage_witness.capture_persistent_gdn_guard(
        persistent, runtime.plan.linear_layer_indices
    )
    group = resident.build_resident_request_group(
        persistent, runtime.plan, resident_count=2, policy=SHARED,
        gdn_base_policy=BORROW,
    )
    resident_runner._set_production_no_mask(
        group, runtime.plan.full_attention_layer_indices
    )
    request_guard = storage_witness.capture_request_gdn_binding_guard(
        group.requests, runtime.plan.linear_layer_indices, policy=BORROW
    )
    return persistent, group, persistent_guard, request_guard


def make_ledgers(runtime: Any, group: Any) -> tuple[list[Any], list[str]]:
    ledgers: list[Any] = []
    backends: list[str] = []
    for request_index, request in enumerate(group.requests):
        ledger = resident.MultiForkHitLedger(
            runtime.plan, request, request_index=request_index,
            resident_count=2, request_policy=SHARED,
            expected_calls_per_layer=8, initial_query_tokens=32,
            kernel=runtime.kernel, strict_position_values=True,
        )
        ledgers.append(ledger)
        backends.append(resident.register_multifork_backend(ledger))
    return ledgers, backends


def model_call(runtime: Any, request: Any, backend: str, input_ids: torch.Tensor) -> tuple[Any, torch.Tensor]:
    original = runtime.backbone.config._attn_implementation
    output = None
    try:
        runtime.backbone.config._attn_implementation = backend
        output = runtime.backbone(
            input_ids=input_ids, past_key_values=request, use_cache=True
        )
        logits = resident_runner._last_logits(runtime.model, output).detach().cpu().float().contiguous()
        require(tuple(logits.shape) == SIDE_CAR_SHAPE, "complete-vocabulary logit shape drift")
        require(bool(torch.isfinite(logits).all()), "non-finite logit sidecar")
        return output, logits
    finally:
        runtime.backbone.config._attn_implementation = original


def rebuild_primary_documents(execution_input: Mapping[str, Any], runtime: Any) -> tuple[list[Any], list[int]]:
    """Rebuild the eight frozen document token tensors without model execution."""

    import build_qcomem_forkaudit_rr2_input_manifest as rr2_builder

    data = execution_input["data"]
    raw = Path(data["pg19_data_path"]).read_bytes()
    manifest = Path(data["pg19_manifest_path"]).read_bytes()
    records, _ = rr2_builder._audit_pg19_train64_bytes(
        raw, manifest, expectations=rr2_builder.FORMAL_EXPECTATIONS
    )
    tokenizer = rr2_builder.load_local_tokenizer(Path(execution_input["model"]["model_dir"]))
    windows, observed = joint_policy.build_pg19_calibration_windows(
        records, tokenizer, books=rr2.FORMAL_BOOKS,
        document_tokens=rr2.FORMAL_DOCUMENT_TOKENS,
        query_tokens=rr2.FORMAL_QUERY_TOKENS,
        stride=rr2.FORMAL_WINDOW_STRIDE,
        candidate_windows_per_book=8, seed=20260817,
    )
    require(observed == data["pg19_windows_canonical_sha256"], "primary-window digest drift")
    documents = [window.document_ids.detach().contiguous().unsqueeze(0) for window in windows]
    hashes = [tensor_sha(value) for value in documents]
    order = sorted(range(len(documents)), key=lambda index: hashes[index])
    require(len(order) == 8 and len(set(hashes)) == 8, "primary documents are not eight distinct identities")
    return documents, order


def build_two_document_case(
    runtime: Any, execution_input: Mapping[str, Any], *, document_a_rank: int,
    document_b_rank: int, mutant: bool,
) -> tuple[Any, Any, Any, Any, Any, Any, dict[str, Any]]:
    documents, order = rebuild_primary_documents(execution_input, runtime)
    require(order[:2] == [document_a_rank, document_b_rank], "BF10 selector/order drift")
    require(tensor_sha(runtime.document) == tensor_sha(documents[document_a_rank]), "loaded BF10 document A drift")
    document_b = documents[document_b_rank].to(device="cuda:0", non_blocking=False)
    persistent_a, _ = rr2._convert_persistent(
        runtime.backbone, runtime.plan, runtime.document, resident_count=2
    )
    persistent_b, _ = rr2._convert_persistent(
        runtime.backbone, runtime.plan, document_b, resident_count=2
    )
    guard_a = storage_witness.capture_persistent_gdn_guard(
        persistent_a, runtime.plan.linear_layer_indices
    )
    guard_b = storage_witness.capture_persistent_gdn_guard(
        persistent_b, runtime.plan.linear_layer_indices
    )
    group = resident.build_resident_request_group(
        persistent_a, runtime.plan, resident_count=2, policy=SHARED,
        gdn_base_policy=BORROW,
    )
    resident_runner._set_production_no_mask(group, runtime.plan.full_attention_layer_indices)
    changed = []
    if mutant:
        request = group.requests[0]
        for layer_index, family, state_index, source in gdn_coordinates(
            persistent_b, runtime.plan.linear_layer_indices
        ):
            target_map = getattr(request.layers[layer_index], family)
            before = target_map[state_index]
            require(repair.exact_alias(before, getattr(persistent_a.layers[layer_index], family)[state_index]), "BF10 clean A binding absent")
            target_map[state_index] = source
            changed.append(f"{layer_index}:{family}:{state_index}")
        require(len(changed) == 60, "BF10 GDN component cardinality")
    request_guard = storage_witness.capture_request_gdn_binding_guard(
        group.requests, runtime.plan.linear_layer_indices, policy=BORROW
    )
    identity = {
        "document_a_rank": document_a_rank,
        "document_b_rank": document_b_rank,
        "document_a_sha256": tensor_sha(runtime.document),
        "document_b_sha256": tensor_sha(document_b),
        "component_binding": {
            "request_0_kv_document_sha256": tensor_sha(runtime.document),
            "request_0_gdn_document_sha256": tensor_sha(document_b) if mutant else tensor_sha(runtime.document),
        },
        "changed_gdn_coordinate_count": len(changed),
    }
    return persistent_a, persistent_b, group, guard_a, guard_b, request_guard, identity


def privatize_all_requests(persistent: Any, group: Any, plan: Any) -> list[dict[str, Any]]:
    rows = []
    for request_index in range(2):
        rows.append(
            repair.prepare_borrowed_single_token_conv_transition(
                persistent, group.requests, plan.linear_layer_indices,
                request_index=request_index,
            )
        )
    return rows


def first_gdn_coordinate(owner: Any, plan: Any) -> tuple[int, str, int, torch.Tensor]:
    rows = gdn_coordinates(owner, plan.linear_layer_indices)
    require(len(rows) == 60, "formal GDN coordinate count")
    return rows[0]


def select_gdn_permutation(owner: Any, plan: Any) -> tuple[tuple[int, str, int, torch.Tensor], tuple[int, str, int, torch.Tensor]]:
    rows = gdn_coordinates(owner, plan.linear_layer_indices)
    for left_index, left in enumerate(rows):
        left_tensor = left[3]
        left_geometry = (
            tuple(left_tensor.shape), tuple(left_tensor.stride()), str(left_tensor.dtype),
            str(left_tensor.device), int(left_tensor.numel()) * int(left_tensor.element_size()),
        )
        left_digest = tensor_sha(left_tensor)
        for right in rows[left_index + 1 :]:
            right_tensor = right[3]
            right_geometry = (
                tuple(right_tensor.shape), tuple(right_tensor.stride()), str(right_tensor.dtype),
                str(right_tensor.device), int(right_tensor.numel()) * int(right_tensor.element_size()),
            )
            if left_geometry == right_geometry and left_digest != tensor_sha(right_tensor):
                return left, right
    raise LiveError("no frozen equal-geometry unequal-content GDN pair")


def kv_unused_suffix(sequence: Any, *, terminal_append_tokens: int = 39) -> dict[str, Any]:
    arena = sequence.arena
    require(arena.document_length == 4095 and arena.page_size == 128, "BF06 frozen KV geometry")
    logical_block = arena.document_blocks_per_sequence
    physical = int(sequence._logical_physical[0][logical_block])
    require(physical >= 0, "BF06 new private page is unpublished at H4")
    valid_at_h4 = sequence.sequence_length - arena.document_length - 1
    # One query token fills the detached document tail; remaining query tokens
    # occupy this new page.  Through H6 the page contains 38 appended tokens.
    terminal_valid = terminal_append_tokens - 1
    require(valid_at_h4 == 31 and terminal_valid == 38, "BF06 suffix schedule drift")
    token_bytes = int(arena.num_key_value_heads * arena.head_dim * arena.key_cache.element_size())
    page_base = int(arena.key_cache[physical].storage_offset()) * int(arena.key_cache.element_size())
    start = page_base + terminal_valid * token_bytes
    end = page_base + arena.page_size * token_bytes
    return {
        "logical_block": logical_block,
        "physical_block": physical,
        "terminal_valid_tokens": terminal_valid,
        "byte_start": start,
        "byte_end_exclusive": end,
        "available_nbytes": end - start,
        "token_bytes": token_bytes,
    }


def select_cross_family_coordinate(request: Any, plan: Any, suffix: Mapping[str, Any]) -> tuple[int, str, int, torch.Tensor]:
    for row in gdn_coordinates(request, plan.linear_layer_indices):
        tensor = row[3]
        nbytes = int(tensor.numel()) * int(tensor.element_size())
        if tensor.is_contiguous() and suffix["byte_start"] % int(tensor.element_size()) == 0 and nbytes <= suffix["available_nbytes"]:
            return row
    raise LiveError("no aligned private GDN tensor fits the frozen KV suffix")


def make_kv_backed_view(tensor: torch.Tensor, key_cache: torch.Tensor, byte_start: int) -> torch.Tensor:
    require(tensor.is_contiguous(), "BF06 selected GDN tensor must be contiguous")
    require(byte_start % tensor.element_size() == 0, "BF06 byte offset alignment")
    view = torch.empty(0, device=tensor.device, dtype=tensor.dtype)
    view.set_(
        key_cache.untyped_storage(),
        int(byte_start // tensor.element_size()),
        tuple(int(item) for item in tensor.shape),
        tuple(int(item) for item in tensor.stride()),
    )
    require(tuple(view.shape) == tuple(tensor.shape) and view.dtype == tensor.dtype, "BF06 view ABI drift")
    view.copy_(tensor)
    return view


def select_nonzero_base_slice(persistent: Any, plan: Any) -> dict[str, Any]:
    alignment = 16
    for layer_index, family, state_index, tensor in gdn_coordinates(
        persistent, plan.linear_layer_indices
    ):
        raw = tensor.detach().contiguous().view(torch.uint8)
        length = min(4096, int(raw.numel()))
        for offset in range(0, int(raw.numel()) - length + 1, alignment):
            if bool(torch.count_nonzero(raw[offset : offset + length]).item()):
                return {
                    "layer_index": layer_index,
                    "family": family,
                    "state_index": state_index,
                    "byte_offset": offset,
                    "byte_length": length,
                    "content_sha256": tensor_sha(tensor),
                }
    raise LiveError("no nonzero aligned persistent GDN base slice")


def scrub_private_state(persistent: Any, group: Any, plan: Any) -> dict[str, Any]:
    base_keys = {storage_key(row[3]) for row in gdn_coordinates(persistent, plan.linear_layer_indices)}
    gdn_storages: dict[tuple[str, int, int], torch.Tensor] = {}
    for request in group.requests:
        for _, _, _, tensor in gdn_coordinates(request, plan.linear_layer_indices):
            key = storage_key(tensor)
            if key not in base_keys:
                gdn_storages.setdefault(key, tensor)
    for tensor in gdn_storages.values():
        tensor.zero_()
    kv_blocks = 0
    for layer_index in plan.full_attention_layer_indices:
        arena = persistent.layers[layer_index].arena
        for request_index in range(2):
            ids = arena.private_block_reservations[request_index].reshape(-1).tolist()
            for physical in ids:
                arena.key_cache[int(physical)].zero_()
                arena.value_cache[int(physical)].zero_()
                kv_blocks += 1
    torch.cuda.synchronize()
    require(all(torch.count_nonzero(tensor).item() == 0 for tensor in gdn_storages.values()), "private GDN scrub incomplete")
    return {
        "private_gdn_backing_storage_count": len(gdn_storages),
        "private_kv_block_role_count": kv_blocks,
        "all_selected_private_gdn_bytes_zero": True,
        "gdn_storage_rows": [
            {"storage_key_hash": hashlib.sha256(repr(key).encode()).hexdigest(), "nbytes": key[2]}
            for key in sorted(gdn_storages)
        ],
        "live_private_tensors": list(gdn_storages.values()),
    }


def pointer_free_scrub_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "live_private_tensors"}


def live_forkaudit_receipts(
    *, persistent: Any, group: Any, plan: Any, ledgers: Sequence[Any],
    persistent_guard: Any, request_guard: Any, phase: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Run only pre-existing validators, retaining failures individually."""

    def observe(name: str, operation: Any) -> dict[str, Any]:
        try:
            receipt = operation()
        except BaseException as exc:
            return {
                "predicate": name,
                "passed": False,
                "exception_type": type(exc).__name__,
                "gate_id": getattr(exc, "gate_id", None),
                "message": str(exc),
                "receipt": None,
            }
        return {
            "predicate": name,
            "passed": True,
            "exception_type": None,
            "gate_id": None,
            "message": None,
            "receipt": json.loads(json.dumps(receipt)),
        }

    rows = [
        observe("MULTIFORK_KERNEL_LEDGER_COMPLETE", lambda: [
            rr2._pointer_free_kernel_ledger(ledger.verify_complete()) for ledger in ledgers
        ]),
        observe("KV_RUNTIME_OWNERSHIP", lambda: resident.validate_runtime_kv_ownership(
            persistent, group, plan, require_appended_tail_cow=True
        )),
        observe("PERSISTENT_GDN_IMMUTABLE", lambda: storage_witness.verify_persistent_gdn_guard(
            persistent_guard, persistent
        )),
    ]
    if phase is None:
        rows.append({
            "predicate": "GDN_PHASE_STORAGE_AND_BINDING_REPLAY",
            "passed": False,
            "exception_type": "MissingMandatoryRecord",
            "gate_id": "MANDATORY_RECORD_COVERAGE",
            "message": "registered H4 phase witness absent",
            "receipt": None,
        })
    else:
        rows.append(observe("GDN_PHASE_STORAGE_AND_BINDING_REPLAY", lambda: {
            "storage": storage_witness.replay_gdn_storage_witness(phase["storage_witness"]),
            "binding": storage_witness.replay_request_gdn_binding_witness(phase["binding_witness"]),
        }))
    first = next((row for row in rows if row["passed"] is False), None)
    return {
        "implementation": "unmodified frozen RR2 ForkAudit validators",
        "predicate_rows": rows,
        "verdict": "pass" if first is None else "fail",
        "first_failed_predicate": None if first is None else first["predicate"],
        "first_failed_gate_id": None if first is None else first["gate_id"],
        "compiled_binary_identity_coverage": "partial",
        "autotuning_choice_coverage": "partial",
        "new_r39_predicates_added": False,
    }


def release_case(*objects: Any, backends: Sequence[str] = ()) -> dict[str, Any]:
    if backends:
        rr2._unregister_backends(list(backends))
    # Callers must clear their own strong references before using this helper;
    # this receipt supplies the synchronized allocator endpoint.
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return allocator_snapshot()


__all__ = [name for name in globals() if not name.startswith("_")]
