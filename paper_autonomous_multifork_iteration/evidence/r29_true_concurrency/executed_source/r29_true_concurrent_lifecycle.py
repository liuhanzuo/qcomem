from __future__ import annotations

"""Round-29 true CUDA-stream concurrency and lifecycle experiment.

This affected-only runner reuses the frozen Qwen3.5-35B-A3B + ForkAudit
vLLM-Q16 implementation.  Unlike the earlier round-major scheduler case, two
host worker threads enqueue complete model steps onto two distinct CUDA
streams behind one barrier.  The experiment proves only overlapping CUDA
stream intervals for this bounded stack; it does not claim simultaneous
kernel execution, vLLM-engine continuous batching, throughput, or capacity.
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from qcomem_forkaudit_lifecycle_transfer import (
    SlotEpochRegistry,
    replay_slot_events,
)
from qcomem_joint_policy import (
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
    sha256_file,
)
from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
from qcomem_qwen35_vllm_paged_integration import (
    register_qwen35_vllm_q16_backend,
)
from qcomem_vllm_paged_fair_control import SHARED_REUSE
from qcomem_vllm_paged_kernel import (
    AUDITED_PACKAGES,
    KERNEL_MODE,
    Q16KernelPagedTensorView,
    _resolve_vllm_unified_attention,
    audit_frozen_kernel_environment,
)
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    MultiForkHitLedger,
    ResidentRequestGroup,
    _capture_kv_binding_guard,
    _request_with_gdn_policy,
    build_pg19_train_query_bank,
    build_resident_request_group,
    validate_runtime_kv_ownership,
)
from run_downstream import atomic_json
from run_qcomem_qwen35_vllm_paged_multifork_resident import (
    _audit_model_config_geometry,
    _build_document_cache,
    _last_logits,
    _linear_state_digest,
    _request_logical_kv_digests,
    _resolve_backbone,
    _source_document_digests,
    _unregister_backend,
)


SCHEMA = "qcomem-forkaudit-true-concurrent-lifecycle-result-v1"
DESIGN_SCHEMA = "qcomem-forkaudit-true-concurrent-lifecycle-design-v1"
PROTOCOL = "qcomem-forkaudit-two-stream-cancel-replace-v1"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
EXPECTED_DATA_SHA256 = (
    "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
)
EXPECTED_DATA_MANIFEST_SHA256 = (
    "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"
)
EXPECTED_UPSTREAM_LEDGER_SHA256 = (
    "7620f05821fc5435a9aaa260ae82577988a5a20eff0f42901b66e6c6871fd2b9"
)
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256 = (
    "c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb"
)
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256 = (
    "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014"
)
FULL_LAYERS = tuple(range(3, 40, 4))
LINEAR_LAYERS = tuple(index for index in range(40) if index not in FULL_LAYERS)
RESIDENT_COUNT = 2
CANCEL_SLOT = 1
DOCUMENT_TOKENS = 4033
PAGE_SIZE = 128
QUERY_TOKENS = 16
QUERY_STRIDE = 64
WINDOW_STRIDE = 263
CANDIDATE_WINDOWS = 8
WINDOW_BOOKS = 8
SEED = 20260821


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return sha256_bytes(raw)


def json_safe_device_uuid(value: Any) -> str | None:
    """Normalize PyTorch's private ``_CUuuid`` wrapper for JSON receipts."""

    if value is None:
        return None
    rendered = str(value)
    require(bool(rendered), "CUDA device UUID rendered as an empty string")
    return rendered


def require_json_serializable(value: Any, *, label: str) -> Any:
    """Fail closed before the atomic writer sees a partially built receipt."""

    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not strict-JSON serializable: {error}") from error
    return value


def device_hardware_receipt(
    properties: Any,
    *,
    cuda_visible_devices: str | None,
    visible_device_count: int,
    name: str,
) -> dict[str, Any]:
    """Build the hardware row with only strict-JSON scalar values."""

    return {
        "cuda_visible_devices": cuda_visible_devices,
        "visible_device_count": int(visible_device_count),
        "name": str(name),
        "uuid": json_safe_device_uuid(getattr(properties, "uuid", None)),
        "total_memory_bytes": int(properties.total_memory),
        "compute_capability": [int(properties.major), int(properties.minor)],
    }


def build_formal_result_payload(
    *,
    expected_design_sha256: str,
    design: Mapping[str, Any],
    input_receipt: Mapping[str, Any],
    hardware: Mapping[str, Any],
    environment: Mapping[str, Any],
    serialized: Mapping[str, Any],
    concurrent: Mapping[str, Any],
    oracle: Mapping[str, Any],
    sidecars: Mapping[str, Any],
    cross_arm: Mapping[str, Any],
    treatment_valid: bool,
    primary_success: bool,
) -> dict[str, Any]:
    """Assemble and preflight the complete formal-result JSON envelope."""

    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "scientific_execution_completed": True,
        "concurrency_treatment_valid": bool(treatment_valid),
        "scientific_run_valid": bool(treatment_valid),
        "formal_evidence_eligible": bool(treatment_valid),
        "primary_success": bool(primary_success),
        "protocol": PROTOCOL,
        "design_preregistration_raw_sha256": expected_design_sha256,
        "input": dict(input_receipt),
        "hardware": dict(hardware),
        "environment": dict(environment),
        "serialized": dict(serialized),
        "concurrent": dict(concurrent),
        "output_oracle": dict(oracle),
        "sidecars": dict(sidecars),
        "cross_arm": dict(cross_arm),
        "claim_boundary": design["claim_boundary"],
    }
    return require_json_serializable(result, label="formal result payload")


def load_bound_json(path: Path, expected_sha256: str, label: str) -> Any:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"{label} raw SHA mismatch")
    return json.loads(raw)


def interval_overlap_ms(intervals: Sequence[Mapping[str, float]]) -> float:
    require(len(intervals) >= 2, "overlap requires at least two intervals")
    for row in intervals:
        require(
            float(row["end_ms"]) > float(row["start_ms"]),
            "CUDA interval duration must be positive",
        )
    return min(float(row["end_ms"]) for row in intervals) - max(
        float(row["start_ms"]) for row in intervals
    )


def validate_design(design: Mapping[str, Any]) -> None:
    require(design.get("schema_version") == DESIGN_SCHEMA, "design schema drift")
    expected = {
        "resident_count": RESIDENT_COUNT,
        "cancel_slot": CANCEL_SLOT,
        "document_tokens": DOCUMENT_TOKENS,
        "page_size": PAGE_SIZE,
        "query_tokens": QUERY_TOKENS,
        "query_stride": QUERY_STRIDE,
        "window_stride": WINDOW_STRIDE,
        "candidate_windows_per_book": CANDIDATE_WINDOWS,
        "window_books": WINDOW_BOOKS,
        "seed": SEED,
    }
    require(design.get("geometry") == expected, "design geometry drift")
    require(
        design.get("model", {}).get("revision") == MODEL_REVISION,
        "model revision drift",
    )
    require(
        design.get("success_rule", {}).get("minimum_overlap_ms_exclusive") == 0.0,
        "concurrency overlap threshold drift",
    )
    require(
        design.get("success_rule", {}).get("full_vocab_logits_torch_equal") is True,
        "logit equality rule drift",
    )


def verify_bound_files(args: argparse.Namespace) -> None:
    require(
        sha256_file(args.pg19_data) == EXPECTED_DATA_SHA256,
        "PG19 data SHA drift",
    )
    require(
        sha256_file(args.pg19_manifest) == EXPECTED_DATA_MANIFEST_SHA256,
        "PG19 manifest SHA drift",
    )
    require(
        sha256_file(args.upstream_code_ledger) == EXPECTED_UPSTREAM_LEDGER_SHA256,
        "upstream code-ledger SHA drift",
    )
    require(
        sha256_file(args.model_artifact_ledger)
        == EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256,
        "model artifact-ledger SHA drift",
    )
    require(
        sha256_file(args.model_weight_ledger)
        == EXPECTED_MODEL_WEIGHT_LEDGER_SHA256,
        "model weight-ledger SHA drift",
    )


def frozen_input_material(
    args: argparse.Namespace,
    design: Mapping[str, Any],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[str, Any]]:
    records, data_audit = audit_pg19_train_calibration(
        args.pg19_data,
        args.pg19_manifest,
        expected_data_sha256=EXPECTED_DATA_SHA256,
        expected_manifest_sha256=EXPECTED_DATA_MANIFEST_SHA256,
        minimum_books=WINDOW_BOOKS,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    windows, windows_sha256 = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=WINDOW_BOOKS,
        document_tokens=DOCUMENT_TOKENS,
        query_tokens=QUERY_TOKENS,
        stride=WINDOW_STRIDE,
        candidate_windows_per_book=CANDIDATE_WINDOWS,
        seed=SEED,
    )
    input_binding = design.get("input_binding", {})
    require(
        windows_sha256 == input_binding.get("windows_sha256"),
        "calibration windows SHA drift",
    )
    window = windows[0]
    require(
        window.source_object == input_binding.get("source_object")
        and str(window.source_id) == input_binding.get("source_id"),
        "frozen rank-0 source drift",
    )
    queries, query_audit = build_pg19_train_query_bank(
        records,
        tokenizer,
        window,
        document_tokens=DOCUMENT_TOKENS,
        query_tokens=QUERY_TOKENS,
        count=RESIDENT_COUNT,
        query_stride=QUERY_STRIDE,
    )
    document = window.document_ids[:DOCUMENT_TOKENS].unsqueeze(0)
    observed_document_sha = sha256_tensor(document)
    observed_query_shas = [sha256_tensor(query) for query in queries]
    require(
        observed_document_sha == input_binding.get("document_token_ids_sha256"),
        "document token SHA drift",
    )
    require(
        observed_query_shas == input_binding.get("query_token_ids_sha256"),
        "query token SHA drift",
    )
    require(
        query_audit.get("source_role")
        == "same-pg19-train-book-raw-nonoverlapping-query-chunks"
        and query_audit.get("synthetic_markers_used") is False
        and query_audit.get("pairwise_nonoverlapping") is True
        and query_audit.get("pairwise_distinct") is True,
        "query audit failed",
    )
    observed_offsets = [row["source_token_offset"] for row in query_audit["rows"]]
    require(
        observed_offsets == input_binding.get("query_source_token_offsets"),
        "query source offsets drift",
    )
    return document, tuple(queries), {
        "pg19_train_only": True,
        "data_audit": data_audit,
        "windows_sha256": windows_sha256,
        "source_object": window.source_object,
        "source_id": str(window.source_id),
        "document_token_ids_sha256": observed_document_sha,
        "query_token_ids_sha256": observed_query_shas,
        "query_audit": query_audit,
    }


class SchedulerGateError(RuntimeError):
    def __init__(self, gate_id: str, message: str) -> None:
        self.gate_id = gate_id
        super().__init__(f"{gate_id}: {message}")


@dataclass(frozen=True)
class DispatchBinding:
    request_id: str
    slot_id: int
    lease: Any
    physical_block_ids: tuple[int, ...]


def require_dispatch(
    registry: SlotEpochRegistry,
    binding: DispatchBinding,
    *,
    request_id: str,
    slot_id: int,
) -> None:
    try:
        registry.validate(binding.lease)
    except RuntimeError as error:
        gate_id = getattr(error, "gate_id", "STALE_SLOT_LEASE")
        raise SchedulerGateError(str(gate_id), str(error)) from error
    if binding.request_id != request_id or binding.lease.request_id != request_id:
        raise SchedulerGateError(
            "LEASE_REQUEST_MISMATCH",
            "lease and dispatched request identifiers differ",
        )
    if binding.slot_id != slot_id or binding.lease.slot_id != slot_id:
        raise SchedulerGateError(
            "LEASE_SLOT_MISMATCH",
            "lease and dispatched slot identifiers differ",
        )


def require_live_reservations_disjoint(
    bindings: Sequence[DispatchBinding],
) -> None:
    seen: dict[int, str] = {}
    for binding in bindings:
        for block_id in binding.physical_block_ids:
            prior = seen.get(block_id)
            if prior is not None:
                raise SchedulerGateError(
                    "LIVE_RESERVATION_OVERLAP",
                    f"physical block {block_id} is live for both {prior} and {binding.request_id}",
                )
            seen[block_id] = binding.request_id


def require_zero_scrubbed(
    blocks: Sequence[Any],
    *,
    expected_physical_block_ids: Sequence[int],
    observed_physical_block_ids: Sequence[int],
) -> None:
    if tuple(observed_physical_block_ids) != tuple(expected_physical_block_ids):
        raise SchedulerGateError(
            "RECLAIM_RESERVATION_MISMATCH",
            "reassigned physical reservation differs from the cancelled slot",
        )
    for block in blocks:
        value = block.count_nonzero()
        nonzero = int(value.item() if hasattr(value, "item") else value)
        if nonzero:
            raise SchedulerGateError(
                "RECLAIM_NOT_ZERO",
                "cancelled private storage was not zero-scrubbed",
            )


def observe_gate(callable_: Any) -> str | None:
    try:
        callable_()
    except SchedulerGateError as error:
        return error.gate_id
    return None


@dataclass
class RequestState:
    semantic_id: str
    request_id: str
    slot_id: int
    request: Any
    lease: Any
    ledger: MultiForkHitLedger
    current: torch.Tensor
    generated: list[int] = field(default_factory=list)
    logits_cpu: dict[str, torch.Tensor] = field(default_factory=dict)


def reservation_ids(request: Any, plan: Any) -> tuple[int, ...]:
    result = []
    for layer_index in plan.full_attention_layer_indices:
        raw = request.layers[layer_index].sequence.reservations.reshape(-1).tolist()
        result.extend(
            int(layer_index) * 1_000_000 + int(block_id) for block_id in raw
        )
    return tuple(result)


def reservation_rows(requests: Sequence[Any], plan: Any) -> list[dict[str, Any]]:
    rows = []
    for slot_id, request in enumerate(requests):
        layers = {
            str(layer_index): [
                int(value)
                for value in request.layers[layer_index]
                .sequence.reservations.reshape(-1)
                .tolist()
            ]
            for layer_index in plan.full_attention_layer_indices
        }
        rows.append(
            {
                "slot_id": slot_id,
                "layers": layers,
                "sha256": sha256_bytes(canonical_bytes(layers)),
            }
        )
    return rows


def private_blocks(
    persistent: Any,
    plan: Any,
    slot_id: int,
) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    tensors: list[torch.Tensor] = []
    ids = []
    for layer_index in plan.full_attention_layer_indices:
        arena = persistent.layers[layer_index].arena
        physical_ids = (
            arena.private_block_reservations[slot_id].reshape(-1).tolist()
        )
        for physical_id in physical_ids:
            physical_id = int(physical_id)
            ids.append(int(layer_index) * 1_000_000 + physical_id)
            tensors.extend(
                (arena.key_cache[physical_id], arena.value_cache[physical_id])
            )
    return tensors, tuple(ids)


def group_digest(
    states: Sequence[RequestState],
    plan: Any,
) -> tuple[Any, Any]:
    ordered = sorted(states, key=lambda state: state.slot_id)
    group = SimpleNamespace(requests=tuple(state.request for state in ordered))
    kv = _request_logical_kv_digests(group, plan.full_attention_layer_indices)
    gdn = []
    for state in ordered:
        row = _linear_state_digest(state.request, plan.linear_layer_indices)
        row.pop("storage_keys")
        gdn.append({"slot_id": state.slot_id, **row})
    return kv, gdn


class ConcurrentLedgerDispatcher:
    """Route one immutable registered backend by live request sequence ID."""

    mask_contract = "prevalidated-no-padding-tail-causal"

    def __init__(self, ledgers: Sequence[MultiForkHitLedger]) -> None:
        self._routes: dict[tuple[int, int], MultiForkHitLedger] = {}
        for ledger in ledgers:
            for layer_index, sequence_id in ledger.sequence_ids.items():
                key = (int(layer_index), int(sequence_id))
                require(key not in self._routes, "dispatcher route collision")
                self._routes[key] = ledger

    def add(self, ledger: MultiForkHitLedger) -> None:
        for layer_index, sequence_id in ledger.sequence_ids.items():
            key = (int(layer_index), int(sequence_id))
            require(key not in self._routes, "replacement dispatcher route collision")
            self._routes[key] = ledger

    def attention_forward(
        self,
        module: Any,
        query: torch.Tensor,
        key: Any,
        value: Any,
        attention_mask: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        layer_index = getattr(module, "layer_idx", None)
        require(
            isinstance(key, Q16KernelPagedTensorView)
            and isinstance(value, Q16KernelPagedTensorView),
            "concurrent dispatcher received dense K/V",
        )
        require(key.sequence is value.sequence, "concurrent dispatcher K/V mismatch")
        ledger = self._routes.get((int(layer_index), id(key.sequence)))
        require(ledger is not None, "concurrent dispatcher request route missing")
        return ledger.attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            **kwargs,
        )


def state_binding(state: RequestState, plan: Any) -> DispatchBinding:
    return DispatchBinding(
        request_id=state.request_id,
        slot_id=state.slot_id,
        lease=state.lease,
        physical_block_ids=reservation_ids(state.request, plan),
    )


def make_state(
    plan: Any,
    request: Any,
    registry: SlotEpochRegistry,
    *,
    semantic_id: str,
    request_id: str,
    slot_id: int,
    query: torch.Tensor,
    expected_calls: int,
    kernel: Any,
) -> RequestState:
    for layer_index in plan.full_attention_layer_indices:
        request.layers[layer_index].sequence.strict_mask_check = False
    lease = registry.acquire(slot_id, request_id)
    ledger = MultiForkHitLedger(
        plan,
        request,
        request_index=slot_id,
        resident_count=RESIDENT_COUNT,
        request_policy=SHARED_REUSE,
        expected_calls_per_layer=expected_calls,
        initial_query_tokens=QUERY_TOKENS,
        kernel=kernel,
        strict_position_values=False,
    )
    return RequestState(
        semantic_id=semantic_id,
        request_id=request_id,
        slot_id=slot_id,
        request=request,
        lease=lease,
        ledger=ledger,
        current=query,
    )


def active_group(requests: Sequence[Any], plan: Any) -> ResidentRequestGroup:
    packed = tuple(requests)
    return ResidentRequestGroup(
        policy=SHARED_REUSE,
        resident_count=RESIDENT_COUNT,
        requests=packed,
        audit={"active_rebound_after_reclamation": True},
        kv_binding_guard=_capture_kv_binding_guard(
            packed,
            plan,
            resident_count=RESIDENT_COUNT,
            policy=SHARED_REUSE,
        ),
    )


def finalize_logits(state: RequestState, logits: torch.Tensor) -> dict[str, Any]:
    require(logits.ndim == 2 and int(logits.shape[0]) == 1, "logits shape drift")
    require(bool(torch.isfinite(logits).all().item()), "non-finite logits")
    cpu = logits.detach().contiguous().cpu().float().clone()
    round_index = len(state.generated)
    sample_id = f"{state.semantic_id}-round-{round_index}"
    require(sample_id not in state.logits_cpu, "duplicate semantic logit sample")
    token = int(cpu.argmax(dim=-1).item())
    state.generated.append(token)
    state.logits_cpu[sample_id] = cpu
    state.current = torch.tensor(
        [[token]], dtype=torch.long, device=logits.device
    )
    return {
        "sample_id": sample_id,
        "token_id": token,
        "full_vocab_logit_sha256": sha256_tensor(cpu),
        "shape": list(cpu.shape),
        "finite": True,
    }


def serialized_batch(
    model: Any,
    backbone: Any,
    registry: SlotEpochRegistry,
    states: Sequence[RequestState],
    plan: Any,
) -> dict[str, Any]:
    require_live_reservations_disjoint([state_binding(state, plan) for state in states])
    rows = []
    for state in states:
        require_dispatch(
            registry,
            state_binding(state, plan),
            request_id=state.request_id,
            slot_id=state.slot_id,
        )
        with torch.inference_mode():
            output = backbone(
                input_ids=state.current,
                past_key_values=state.request,
                use_cache=True,
            )
            logits = _last_logits(model, output)
        torch.cuda.synchronize()
        rows.append(finalize_logits(state, logits))
        del output, logits
    return {
        "execution": "single-host-thread-single-default-stream-serialized",
        "request_count": len(states),
        "rows": rows,
        "concurrent_execution_claimed": False,
    }


def concurrent_batch(
    model: Any,
    backbone: Any,
    registry: SlotEpochRegistry,
    states: Sequence[RequestState],
    plan: Any,
    *,
    phase: str,
) -> dict[str, Any]:
    packed = tuple(states)
    require(len(packed) == RESIDENT_COUNT, "concurrent batch cardinality drift")
    require_live_reservations_disjoint([state_binding(state, plan) for state in packed])
    for state in packed:
        require_dispatch(
            registry,
            state_binding(state, plan),
            request_id=state.request_id,
            slot_id=state.slot_id,
        )

    streams = [torch.cuda.Stream(device=0) for _ in packed]
    require(
        len({int(stream.cuda_stream) for stream in streams}) == len(streams),
        "CUDA stream handles are not distinct",
    )
    origin = torch.cuda.Event(enable_timing=True)
    starts = [torch.cuda.Event(enable_timing=True) for _ in packed]
    ends = [torch.cuda.Event(enable_timing=True) for _ in packed]
    torch.cuda.synchronize()
    origin.record(torch.cuda.default_stream(0))
    origin.synchronize()
    barrier = threading.Barrier(len(packed) + 1)

    def worker(index: int) -> tuple[int, int, Any, torch.Tensor]:
        torch.cuda.set_device(0)
        stream = streams[index]
        barrier.wait(timeout=30.0)
        with torch.inference_mode(), torch.cuda.stream(stream):
            stream.wait_event(origin)
            starts[index].record(stream)
            state = packed[index]
            output = backbone(
                input_ids=state.current,
                past_key_values=state.request,
                use_cache=True,
            )
            logits = _last_logits(model, output)
            ends[index].record(stream)
        return index, threading.get_ident(), output, logits

    returned: list[tuple[int, int, Any, torch.Tensor]] = []
    with ThreadPoolExecutor(
        max_workers=len(packed),
        thread_name_prefix=f"r29-{phase}",
    ) as executor:
        futures = [executor.submit(worker, index) for index in range(len(packed))]
        barrier.wait(timeout=30.0)
        returned = [future.result() for future in futures]
    torch.cuda.synchronize()
    returned.sort(key=lambda row: row[0])
    require(
        len({thread_id for _, thread_id, _, _ in returned}) == len(returned),
        "concurrent workers did not use distinct host threads",
    )
    intervals = []
    output_rows = []
    for index, thread_id, output, logits in returned:
        state = packed[index]
        intervals.append(
            {
                "slot_id": state.slot_id,
                "request_id": state.request_id,
                "host_thread_id": int(thread_id),
                "cuda_stream_handle": int(streams[index].cuda_stream),
                "start_ms": float(origin.elapsed_time(starts[index])),
                "end_ms": float(origin.elapsed_time(ends[index])),
            }
        )
        output_rows.append(finalize_logits(state, logits))
        del output, logits
    overlap = interval_overlap_ms(intervals)
    return {
        "execution": "two-host-workers-two-distinct-cuda-streams-one-barrier",
        "phase": phase,
        "request_count": len(packed),
        "distinct_host_thread_count": len(
            {row["host_thread_id"] for row in intervals}
        ),
        "distinct_cuda_stream_count": len(
            {row["cuda_stream_handle"] for row in intervals}
        ),
        "intervals": intervals,
        "overlap_ms": overlap,
        "overlap_gate_passed": overlap > 0.0,
        "rows": output_rows,
        "simultaneous_kernel_execution_claimed": False,
        "continuous_batching_claimed": False,
    }


def scrub_and_reclaim(
    persistent: Any,
    plan: Any,
    registry: SlotEpochRegistry,
    cancelled: RequestState,
) -> dict[str, Any]:
    blocks, physical_ids = private_blocks(persistent, plan, CANCEL_SLOT)
    pre_scrub_gate = observe_gate(
        lambda: require_zero_scrubbed(
            blocks,
            expected_physical_block_ids=physical_ids,
            observed_physical_block_ids=physical_ids,
        )
    )
    require(pre_scrub_gate == "RECLAIM_NOT_ZERO", "pre-scrub positive control drift")
    registry.cancel(cancelled.lease)
    rows = []
    for layer_index in plan.full_attention_layer_indices:
        arena = persistent.layers[layer_index].arena
        require(arena._fork_cursor == RESIDENT_COUNT, "arena cursor drift")
        ids = [
            int(value)
            for value in arena.private_block_reservations[CANCEL_SLOT]
            .reshape(-1)
            .tolist()
        ]
        for physical_id in ids:
            arena.key_cache[physical_id].zero_()
            arena.value_cache[physical_id].zero_()
        arena._fork_cursor = CANCEL_SLOT
        rows.append(
            {
                "layer_index": int(layer_index),
                "scrubbed_physical_block_ids": ids,
            }
        )
    torch.cuda.synchronize()
    clean_blocks, clean_ids = private_blocks(persistent, plan, CANCEL_SLOT)
    require_zero_scrubbed(
        clean_blocks,
        expected_physical_block_ids=physical_ids,
        observed_physical_block_ids=clean_ids,
    )
    return {
        "cancelled_request_id": cancelled.request_id,
        "slot_id": CANCEL_SLOT,
        "pre_scrub_positive_control_gate": pre_scrub_gate,
        "zero_scrubbed": True,
        "exact_physical_ids_verified": clean_ids == physical_ids,
        "physical_block_ids": list(physical_ids),
        "layers": rows,
    }


def flatten_logits(states: Sequence[RequestState]) -> dict[str, torch.Tensor]:
    rows: dict[str, torch.Tensor] = {}
    for state in states:
        for sample_id, tensor in state.logits_cpu.items():
            require(sample_id not in rows, "duplicate cross-state logit sample")
            rows[sample_id] = tensor
    return rows


def run_arm(
    model: Any,
    backbone: Any,
    plan: Any,
    document: torch.Tensor,
    queries: Sequence[torch.Tensor],
    kernel: Any,
    *,
    arm: str,
) -> dict[str, Any]:
    require(arm in ("serialized", "concurrent"), "unknown arm")
    persistent = _build_document_cache(backbone, document)
    from qcomem_qwen35_vllm_paged_integration import (
        convert_all_qwen35_full_layers_to_vllm_q16,
    )

    conversion = convert_all_qwen35_full_layers_to_vllm_q16(
        persistent,
        plan,
        page_size=PAGE_SIZE,
        max_append_tokens=QUERY_TOKENS + 1,
        max_request_forks=RESIDENT_COUNT,
    )
    source_before = _source_document_digests(
        persistent, plan.full_attention_layer_indices
    )
    initial_group = build_resident_request_group(
        persistent,
        plan,
        resident_count=RESIDENT_COUNT,
        policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    registry = SlotEpochRegistry(RESIDENT_COUNT)
    initial = [
        make_state(
            plan,
            request,
            registry,
            semantic_id=f"initial-slot-{slot_id}",
            request_id=f"{arm}-initial-slot-{slot_id}",
            slot_id=slot_id,
            query=queries[slot_id],
            expected_calls=2 if slot_id == 0 else 1,
            kernel=kernel,
        )
        for slot_id, request in enumerate(initial_group.requests)
    ]
    initial_reservations = reservation_rows(initial_group.requests, plan)
    ownership_pre = validate_runtime_kv_ownership(
        persistent,
        initial_group,
        plan,
        require_appended_tail_cow=False,
    )
    dispatcher = ConcurrentLedgerDispatcher([state.ledger for state in initial])
    registered = register_qwen35_vllm_q16_backend(dispatcher)
    original_backend = backbone.config._attn_implementation
    phase_receipts = []
    cancelled_receipt: dict[str, Any] | None = None
    survivor_receipt: dict[str, Any] | None = None
    replacement_receipt: dict[str, Any] | None = None
    replacement: RequestState | None = None
    try:
        backbone.config._attn_implementation = registered.name
        if arm == "concurrent":
            phase_receipts.append(
                concurrent_batch(
                    model,
                    backbone,
                    registry,
                    initial,
                    plan,
                    phase="pre-cancel",
                )
            )
        else:
            phase_receipts.append(
                serialized_batch(model, backbone, registry, initial, plan)
            )
        ownership_after_initial = validate_runtime_kv_ownership(
            persistent,
            initial_group,
            plan,
            require_appended_tail_cow=True,
        )
        cancelled = initial[CANCEL_SLOT]
        cancelled_receipt = cancelled.ledger.verify_complete()
        scrub = scrub_and_reclaim(persistent, plan, registry, cancelled)
        stale_gate = observe_gate(
            lambda: require_dispatch(
                registry,
                state_binding(cancelled, plan),
                request_id=cancelled.request_id,
                slot_id=cancelled.slot_id,
            )
        )
        require(stale_gate == "STALE_SLOT_LEASE", "stale lease gate drift")
        request, _request_audit = _request_with_gdn_policy(
            persistent,
            plan,
            request_policy=SHARED_REUSE,
            gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
        )
        replacement = make_state(
            plan,
            request,
            registry,
            semantic_id="replacement-slot-1",
            request_id=f"{arm}-replacement-slot-1",
            slot_id=CANCEL_SLOT,
            query=queries[CANCEL_SLOT],
            expected_calls=1,
            kernel=kernel,
        )
        dispatcher.add(replacement.ledger)
        require(
            reservation_ids(replacement.request, plan)
            == reservation_ids(cancelled.request, plan),
            "replacement did not reuse the cancelled reservation",
        )
        active = [initial[0], replacement]
        rebound_group = active_group([state.request for state in active], plan)
        ownership_after_rebind = validate_runtime_kv_ownership(
            persistent,
            rebound_group,
            plan,
            require_appended_tail_cow=False,
        )
        if arm == "concurrent":
            phase_receipts.append(
                concurrent_batch(
                    model,
                    backbone,
                    registry,
                    active,
                    plan,
                    phase="post-reclaim",
                )
            )
        else:
            phase_receipts.append(
                serialized_batch(model, backbone, registry, active, plan)
            )
        ownership_final = validate_runtime_kv_ownership(
            persistent,
            rebound_group,
            plan,
            require_appended_tail_cow=True,
        )
        survivor_receipt = initial[0].ledger.verify_complete()
        replacement_receipt = replacement.ledger.verify_complete()
        source_after = _source_document_digests(
            persistent, plan.full_attention_layer_indices
        )
        require(source_before == source_after, "source document pages mutated")
        lifecycle = registry.receipt()
        lifecycle_replay = replay_slot_events(lifecycle)
        require(
            lifecycle_replay["final_epochs"] == [0, 1]
            and lifecycle_replay["final_owners"]
            == [initial[0].request_id, replacement.request_id],
            "lifecycle terminal replay drift",
        )
        final_kv, final_gdn = group_digest(active, plan)
        logits = flatten_logits([initial[0], cancelled, replacement])
        require(len(logits) == 4, "arm did not produce four semantic logit samples")
        return {
            "arm": arm,
            "execution": (
                "two-host-workers-two-distinct-cuda-streams"
                if arm == "concurrent"
                else "single-host-thread-single-default-stream"
            ),
            "phase_receipts": phase_receipts,
            "source_document_sha256_before": source_before,
            "source_document_sha256_after": source_after,
            "source_document_immutable": True,
            "ownership": {
                "pre": ownership_pre,
                "after_initial": ownership_after_initial,
                "after_rebind": ownership_after_rebind,
                "final": ownership_final,
            },
            "initial_reservations": initial_reservations,
            "replacement_reservation": {
                **reservation_rows((replacement.request,), plan)[0],
                "slot_id": CANCEL_SLOT,
            },
            "exact_private_reservation_reuse": True,
            "scrub": scrub,
            "stale_cancelled_lease_gate": stale_gate,
            "lifecycle": lifecycle,
            "lifecycle_replay": lifecycle_replay,
            "ledger_receipts": {
                "survivor": survivor_receipt,
                "cancelled": cancelled_receipt,
                "replacement": replacement_receipt,
            },
            "trajectories": {
                initial[0].semantic_id: list(initial[0].generated),
                cancelled.semantic_id: list(cancelled.generated),
                replacement.semantic_id: list(replacement.generated),
            },
            "final_logical_kv": final_kv,
            "final_gdn_state": final_gdn,
            "conversion": {
                "document_length": conversion.document_length,
                "page_size": conversion.page_size,
                "max_append_tokens": conversion.max_append_tokens,
                "max_request_forks": conversion.max_request_forks,
                "full_attention_layer_count": len(conversion.layer_indices),
            },
            "logits": logits,
        }
    finally:
        backbone.config._attn_implementation = original_backend
        _unregister_backend(registered.name)
        torch.cuda.synchronize()


def write_logit_sidecar(
    path: Path,
    logits: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    offset = 0
    with path.open("wb") as handle:
        for sample_id in sorted(logits):
            tensor = logits[sample_id].detach().contiguous().cpu().float()
            raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
            handle.write(raw)
            records.append(
                {
                    "sample_id": sample_id,
                    "dtype": "float32-le",
                    "shape": list(tensor.shape),
                    "offset_bytes": offset,
                    "nbytes": len(raw),
                    "content_sha256": sha256_bytes(raw),
                    "token_id": int(tensor.argmax(dim=-1).item()),
                }
            )
            offset += len(raw)
    raw_bundle = path.read_bytes()
    return {
        "schema_version": "qcomem-forkaudit-r29-logit-sidecar-v1",
        "path": path.name,
        "bytes": len(raw_bundle),
        "sha256": sha256_bytes(raw_bundle),
        "record_count": len(records),
        "records": records,
        "terminal_exact_byte_coverage": (
            bool(records)
            and records[0]["offset_bytes"] == 0
            and records[-1]["offset_bytes"] + records[-1]["nbytes"]
            == len(raw_bundle)
        ),
    }


def compare_arms(
    serialized: Mapping[str, torch.Tensor],
    concurrent: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    require(set(serialized) == set(concurrent), "oracle sample IDs differ")
    rows = []
    for sample_id in sorted(serialized):
        left = serialized[sample_id]
        right = concurrent[sample_id]
        require(left.shape == right.shape, "oracle logit shape differs")
        difference = (left - right).abs()
        row = {
            "sample_id": sample_id,
            "serialized_sha256": sha256_tensor(left),
            "concurrent_sha256": sha256_tensor(right),
            "torch_equal": bool(torch.equal(left, right)),
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "serialized_token_id": int(left.argmax(dim=-1).item()),
            "concurrent_token_id": int(right.argmax(dim=-1).item()),
        }
        row["token_equal"] = (
            row["serialized_token_id"] == row["concurrent_token_id"]
        )
        rows.append(row)
    return {
        "sample_count": len(rows),
        "rows": rows,
        "all_full_vocab_logits_torch_equal": all(row["torch_equal"] for row in rows),
        "all_generated_tokens_equal": all(row["token_equal"] for row in rows),
        "maximum_abs_error": max(row["max_abs_error"] for row in rows),
    }


def strip_logits(arm: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    result = dict(arm)
    logits = dict(result.pop("logits"))
    return result, logits


def run_gpu(args: argparse.Namespace, design: Mapping[str, Any]) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.device_count() == 1, "formal process requires exactly one visible GPU")
    torch.cuda.set_device(0)
    document_cpu, queries_cpu, input_receipt = frozen_input_material(args, design)

    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    require(tuple(plan.full_attention_layer_indices) == FULL_LAYERS, "full layer plan drift")
    require(tuple(plan.linear_layer_indices) == LINEAR_LAYERS, "linear layer plan drift")
    kernel_environment = audit_frozen_kernel_environment()
    require(
        kernel_environment.get("matches_frozen_environment") is True,
        "frozen vLLM kernel environment drift",
    )
    kernel = _resolve_vllm_unified_attention()
    torch.cuda.synchronize()
    document = document_cpu.to(device="cuda:0", dtype=torch.long)
    queries = tuple(query.to(device="cuda:0", dtype=torch.long) for query in queries_cpu)

    with torch.inference_mode():
        serialized_raw = run_arm(
            model,
            backbone,
            plan,
            document,
            queries,
            kernel,
            arm="serialized",
        )
    serialized, serialized_logits = strip_logits(serialized_raw)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    with torch.inference_mode():
        concurrent_raw = run_arm(
            model,
            backbone,
            plan,
            document,
            queries,
            kernel,
            arm="concurrent",
        )
    concurrent, concurrent_logits = strip_logits(concurrent_raw)
    oracle = compare_arms(serialized_logits, concurrent_logits)
    final_logical_kv_equal = (
        serialized["final_logical_kv"] == concurrent["final_logical_kv"]
    )
    final_gdn_state_equal = (
        serialized["final_gdn_state"] == concurrent["final_gdn_state"]
    )
    treatment_valid = all(
        phase.get("overlap_gate_passed") is True
        and phase.get("overlap_ms", 0.0) > 0.0
        for phase in concurrent["phase_receipts"]
    )
    primary_success = (
        treatment_valid
        and oracle["all_full_vocab_logits_torch_equal"] is True
        and oracle["all_generated_tokens_equal"] is True
        and final_logical_kv_equal
        and final_gdn_state_equal
    )

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    serialized_sidecar = write_logit_sidecar(
        args.artifact_dir / "serialized-logits.fp32.bin", serialized_logits
    )
    concurrent_sidecar = write_logit_sidecar(
        args.artifact_dir / "concurrent-logits.fp32.bin", concurrent_logits
    )
    properties = torch.cuda.get_device_properties(0)
    kernel_identity = {
        "module": str(getattr(kernel, "__module__", type(kernel).__module__)),
        "qualname": str(getattr(kernel, "__qualname__", type(kernel).__qualname__)),
        "signature": str(inspect.signature(kernel)),
    }
    hardware = device_hardware_receipt(
        properties,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        visible_device_count=torch.cuda.device_count(),
        name=torch.cuda.get_device_name(0),
    )
    environment = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
        "audited_packages": dict(AUDITED_PACKAGES),
        "kernel_mode": KERNEL_MODE,
        "kernel_environment": kernel_environment,
        "kernel_identity": kernel_identity,
        "model_geometry": _audit_model_config_geometry(args.model),
    }
    sidecars = {
        "serialized": serialized_sidecar,
        "concurrent": concurrent_sidecar,
    }
    cross_arm = {
        "full_vocab_logits_torch_equal": oracle[
            "all_full_vocab_logits_torch_equal"
        ],
        "generated_tokens_equal": oracle["all_generated_tokens_equal"],
        "final_logical_kv_equal": final_logical_kv_equal,
        "final_gdn_state_equal": final_gdn_state_equal,
        "source_document_immutable_both_arms": True,
        "ownership_receipts_passed_both_arms": True,
        "lifecycle_replay_passed_both_arms": True,
        "concurrent_phases_with_positive_stream_overlap": sum(
            phase.get("overlap_gate_passed") is True
            for phase in concurrent["phase_receipts"]
        ),
    }
    return build_formal_result_payload(
        expected_design_sha256=args.expected_design_sha256,
        design=design,
        input_receipt=input_receipt,
        hardware=hardware,
        environment=environment,
        serialized=serialized,
        concurrent=concurrent,
        oracle=oracle,
        sidecars=sidecars,
        cross_arm=cross_arm,
        treatment_valid=treatment_valid,
        primary_success=primary_success,
    )


def mock_result(design: Mapping[str, Any]) -> dict[str, Any]:
    validate_design(design)
    positive = interval_overlap_ms(
        (
            {"start_ms": 1.0, "end_ms": 5.0},
            {"start_ms": 2.0, "end_ms": 6.0},
        )
    )
    registry = SlotEpochRegistry(2)
    old = registry.acquire(1, "old")
    registry.acquire(0, "survivor")
    registry.cancel(old)
    replacement = registry.acquire(1, "replacement")
    replay = replay_slot_events(registry.receipt())
    require(positive == 3.0, "mock overlap arithmetic drift")
    require(replay["final_owners"] == ["survivor", "replacement"], "mock replay drift")
    require(replacement.epoch == 1, "mock replacement epoch drift")
    return {
        "schema_version": "qcomem-forkaudit-true-concurrent-lifecycle-mock-v1",
        "passed": True,
        "gpu_executed": False,
        "positive_overlap_ms": positive,
        "lifecycle_replay": replay,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("mock", "formal"), required=True)
    result.add_argument("--design-preregistration", type=Path, required=True)
    result.add_argument("--expected-design-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--artifact-dir", type=Path)
    result.add_argument("--model", type=Path)
    result.add_argument("--model-artifact-ledger", type=Path)
    result.add_argument("--model-weight-ledger", type=Path)
    result.add_argument("--pg19-data", type=Path)
    result.add_argument("--pg19-manifest", type=Path)
    result.add_argument("--upstream-code-ledger", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    design = load_bound_json(
        args.design_preregistration,
        args.expected_design_sha256,
        "design preregistration",
    )
    validate_design(design)
    if args.stage == "mock":
        value = mock_result(design)
    else:
        required = (
            args.artifact_dir,
            args.model,
            args.model_artifact_ledger,
            args.model_weight_ledger,
            args.pg19_data,
            args.pg19_manifest,
            args.upstream_code_ledger,
        )
        require(all(item is not None for item in required), "formal paths are required")
        verify_bound_files(args)
        value = run_gpu(args, design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
