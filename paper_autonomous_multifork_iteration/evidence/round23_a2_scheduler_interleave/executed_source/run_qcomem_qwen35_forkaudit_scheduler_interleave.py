from __future__ import annotations

"""Affected-only Round-22 A2 scheduler/interleaving experiment.

This runner deliberately reuses the frozen Qwen3.5 + vLLM-Q16 ForkAudit
implementation.  It adds two prefix/page geometries, a deterministic
scheduler-managed request interleaving, cancellation/reclamation, and three
preregistered scheduler-path faults.  Debug outputs are never formal evidence
and never report a detection rate.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from qcomem_forkaudit_lifecycle_transfer import (
    SlotEpochRegistry,
    replay_slot_events,
)
from qcomem_forkaudit_scheduler_contract import (
    DispatchBinding,
    SCHEDULER_PROTOCOL,
    observe_gate,
    replay_schedule,
    require_dispatch,
    require_live_reservations_disjoint,
    require_zero_scrubbed,
)
from qcomem_joint_policy import (
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
    sha256_file,
)
from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
from qcomem_qwen35_vllm_paged_integration import convert_all_qwen35_full_layers_to_vllm_q16
from qcomem_vllm_paged_fair_control import SHARED_REUSE
from qcomem_vllm_paged_kernel import (
    AUDITED_PACKAGES,
    KERNEL_MODE,
    _resolve_vllm_unified_attention,
    audit_frozen_kernel_environment,
)
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    MultiForkHitLedger,
    _request_with_gdn_policy,
    build_pg19_train_query_bank,
    build_resident_request_group,
    register_multifork_backend,
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


MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
EXPECTED_DATA_SHA256 = "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
EXPECTED_DATA_MANIFEST_SHA256 = "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"
EXPECTED_UPSTREAM_LEDGER_SHA256 = "7620f05821fc5435a9aaa260ae82577988a5a20eff0f42901b66e6c6871fd2b9"
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256 = "c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb"
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256 = "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014"
STATIC_SCHEMA = "qcomem-forkaudit-scheduler-interleave-static-v1"
DEBUG_SCHEMA = "qcomem-forkaudit-scheduler-interleave-debug-v1"
FORMAL_SHARD_SCHEMA = "qcomem-forkaudit-scheduler-interleave-formal-shard-v1"
FORMAL_AGGREGATE_SCHEMA = "qcomem-forkaudit-scheduler-interleave-formal-aggregate-v1"
FORMAL_WORLD_SIZE = 8
RESIDENT_COUNT = 4
TOTAL_ROUNDS = 4
CANCEL_SLOT = 3
CANCEL_AFTER_ROUNDS = 2
QUERY_TOKENS = 16
QUERY_STRIDE = 64
WINDOW_STRIDE = 263
CANDIDATE_WINDOWS = 8
SEED = 20260821
FULL_LAYERS = tuple(range(3, 40, 4))
LINEAR_LAYERS = tuple(index for index in range(40) if index not in FULL_LAYERS)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_tensor(value: torch.Tensor) -> str:
    return sha256_bytes(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def load_bound_json(path: Path, expected_sha256: str, label: str) -> Any:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"{label} raw SHA mismatch")
    return json.loads(raw)


def geometry_cells(design: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = design.get("geometry_cells")
    require(isinstance(cells, list) and len(cells) == 2, "design must bind two geometry cells")
    return [dict(cell) for cell in cells]


def schedule_spec(cell_id: str, *, lifecycle: bool) -> list[dict[str, Any]]:
    if lifecycle:
        compact = [
            ("initial", 1, 0, "initial"),
            ("initial", 0, 0, "initial"),
            ("initial", 3, 0, "initial"),
            ("initial", 2, 0, "initial"),
            ("initial", 3, 1, "initial"),
            ("initial", 1, 1, "initial"),
            ("initial", 0, 1, "initial"),
            ("initial", 2, 1, "initial"),
            ("post-reclaim", 0, 2, "initial"),
            ("post-reclaim", 3, 0, "replacement"),
            ("post-reclaim", 1, 2, "initial"),
            ("post-reclaim", 2, 2, "initial"),
            ("post-reclaim", 3, 1, "replacement"),
            ("post-reclaim", 2, 3, "initial"),
            ("post-reclaim", 1, 3, "initial"),
            ("post-reclaim", 3, 2, "replacement"),
            ("post-reclaim", 0, 3, "initial"),
            ("post-reclaim", 3, 3, "replacement"),
        ]
    else:
        compact = [
            ("control", 1, 0, "control"),
            ("control", 0, 0, "control"),
            ("control", 3, 0, "control"),
            ("control", 2, 0, "control"),
            ("control", 3, 1, "control"),
            ("control", 1, 1, "control"),
            ("control", 0, 1, "control"),
            ("control", 2, 1, "control"),
            ("control", 0, 2, "control"),
            ("control", 3, 2, "control"),
            ("control", 2, 2, "control"),
            ("control", 1, 2, "control"),
            ("control", 2, 3, "control"),
            ("control", 1, 3, "control"),
            ("control", 0, 3, "control"),
            ("control", 3, 3, "control"),
        ]
    return [
        {
            "event_index": index,
            "phase": phase,
            "slot_id": slot,
            "round_index": round_index,
            "request_id": f"{cell_id}-{kind}-slot-{slot}",
        }
        for index, (phase, slot, round_index, kind) in enumerate(compact)
    ]


def verify_design_sources(args: argparse.Namespace, design: Mapping[str, Any]) -> None:
    source = design.get("source_bundle")
    require(isinstance(source, Mapping), "design source bundle missing")
    require(sha256_file(Path(__file__)) == source.get("runner_sha256"), "runner SHA drift")
    helper = Path(__file__).with_name("qcomem_forkaudit_scheduler_contract.py")
    require(sha256_file(helper) == source.get("scheduler_contract_sha256"), "scheduler contract SHA drift")
    require(
        sha256_file(args.upstream_code_ledger) == EXPECTED_UPSTREAM_LEDGER_SHA256,
        "upstream code ledger SHA drift",
    )
    require(
        source.get("upstream_code_ledger_sha256") == EXPECTED_UPSTREAM_LEDGER_SHA256,
        "design upstream ledger binding drift",
    )


def input_material(args: argparse.Namespace, design: Mapping[str, Any]) -> tuple[Any, Any, list[Any], dict[str, Any]]:
    require(sha256_file(args.pg19_data) == EXPECTED_DATA_SHA256, "PG19 data SHA drift")
    require(sha256_file(args.pg19_manifest) == EXPECTED_DATA_MANIFEST_SHA256, "PG19 manifest SHA drift")
    records, data_audit = audit_pg19_train_calibration(
        args.pg19_data,
        args.pg19_manifest,
        expected_data_sha256=EXPECTED_DATA_SHA256,
        expected_manifest_sha256=EXPECTED_DATA_MANIFEST_SHA256,
        minimum_books=FORMAL_WORLD_SIZE,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    max_document_tokens = max(int(cell["document_tokens"]) for cell in geometry_cells(design))
    windows, windows_sha256 = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=FORMAL_WORLD_SIZE,
        document_tokens=max_document_tokens,
        query_tokens=QUERY_TOKENS,
        stride=WINDOW_STRIDE,
        candidate_windows_per_book=CANDIDATE_WINDOWS,
        seed=SEED,
    )
    require(len(windows) == FORMAL_WORLD_SIZE, "input window count drift")
    return records, tokenizer, windows, {"data_audit": data_audit, "windows_sha256": windows_sha256}


def build_static(args: argparse.Namespace, design: Mapping[str, Any]) -> dict[str, Any]:
    verify_design_sources(args, design)
    require(sha256_file(args.model_artifact_ledger) == EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256, "model artifact ledger drift")
    require(sha256_file(args.model_weight_ledger) == EXPECTED_MODEL_WEIGHT_LEDGER_SHA256, "model weight ledger drift")
    records, tokenizer, windows, audit = input_material(args, design)
    ranks = []
    for rank, window in enumerate(windows):
        cells = []
        for cell in geometry_cells(design):
            document_tokens = int(cell["document_tokens"])
            queries, query_audit = build_pg19_train_query_bank(
                records,
                tokenizer,
                window,
                document_tokens=document_tokens,
                query_tokens=QUERY_TOKENS,
                count=RESIDENT_COUNT,
                query_stride=QUERY_STRIDE,
            )
            document = window.document_ids[:document_tokens]
            cells.append(
                {
                    "cell_id": cell["cell_id"],
                    "document_token_ids_sha256": sha256_tensor(document),
                    "query_token_ids_sha256": [sha256_tensor(query) for query in queries],
                    "query_audit": query_audit,
                    "control_schedule_sha256": sha256_json(schedule_spec(cell["cell_id"], lifecycle=False)),
                    "lifecycle_schedule_sha256": sha256_json(schedule_spec(cell["cell_id"], lifecycle=True)),
                }
            )
        ranks.append(
            {
                "rank": rank,
                "source_object": window.source_object,
                "source_id": str(window.source_id),
                "cells": cells,
            }
        )
    return {
        "schema_version": STATIC_SCHEMA,
        "created_before_gpu_execution": True,
        "design_preregistration_raw_sha256": args.expected_design_sha256,
        "windows_sha256": audit["windows_sha256"],
        "pg19_data_sha256": EXPECTED_DATA_SHA256,
        "pg19_manifest_sha256": EXPECTED_DATA_MANIFEST_SHA256,
        "upstream_code_ledger_sha256": EXPECTED_UPSTREAM_LEDGER_SHA256,
        "model_artifact_ledger_sha256": EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256,
        "model_weight_ledger_sha256": EXPECTED_MODEL_WEIGHT_LEDGER_SHA256,
        "ranks": ranks,
        "claim_boundary": design["claim_boundary"],
    }


class MockBlock:
    def __init__(self, nonzero: int) -> None:
        self.nonzero = nonzero

    def count_nonzero(self) -> int:
        return self.nonzero


def run_mock(design: Mapping[str, Any]) -> dict[str, Any]:
    registry = SlotEpochRegistry(2)
    old_lease = registry.acquire(0, "old")
    old = DispatchBinding("old", 0, old_lease, (10, 11))
    other = DispatchBinding("other", 1, registry.acquire(1, "other"), (12, 13))
    require_dispatch(registry, old, request_id="old", slot_id=0)
    require_live_reservations_disjoint((old, other))
    cross_gate = observe_gate(lambda: require_dispatch(registry, old, request_id="other", slot_id=0))
    unscrubbed_gate = observe_gate(
        lambda: require_zero_scrubbed(
            (MockBlock(0), MockBlock(1)),
            expected_physical_block_ids=(10, 11),
            observed_physical_block_ids=(10, 11),
        )
    )
    registry.cancel(old_lease)
    registry.acquire(0, "replacement")
    stale_gate = observe_gate(lambda: require_dispatch(registry, old, request_id="old", slot_id=0))
    observed = {
        "FH1_cancelled_lease_dispatch": stale_gate,
        "FH2_cross_request_dispatch": cross_gate,
        "FH3_reclaim_without_zero_scrub": unscrubbed_gate,
    }
    expected = {row["fault_id"]: row["expected_gate"] for row in design["heldout_faults"]}
    require(observed == expected, "CPU/mock fault-to-gate mapping failed")
    return {
        "schema_version": "qcomem-forkaudit-scheduler-interleave-mock-v1",
        "passed": True,
        "gpu_executed": False,
        "fault_outcomes": [
            {"fault_id": key, "expected_gate": expected[key], "observed_gate": value}
            for key, value in observed.items()
        ],
        "detection_rate_reported": False,
    }


@dataclass
class RequestState:
    request_id: str
    slot_id: int
    request: Any
    lease: Any
    backend: str
    ledger: Any
    current: torch.Tensor
    generated: list[int]
    logit_sha256: list[str]
    logits_cpu: list[torch.Tensor]


def reservation_ids(request: Any, plan: Any) -> tuple[int, ...]:
    result = []
    for layer_index in plan.full_attention_layer_indices:
        raw = request.layers[layer_index].sequence.reservations.reshape(-1).tolist()
        result.extend(int(layer_index) * 1_000_000 + int(block_id) for block_id in raw)
    return tuple(result)


def reservation_rows(requests: Sequence[Any], plan: Any) -> list[dict[str, Any]]:
    rows = []
    for slot_id, request in enumerate(requests):
        layers = {
            str(layer_index): [
                int(value)
                for value in request.layers[layer_index].sequence.reservations.reshape(-1).tolist()
            ]
            for layer_index in plan.full_attention_layer_indices
        }
        rows.append({"slot_id": slot_id, "layers": layers, "sha256": sha256_json(layers)})
    return rows


def private_blocks(persistent: Any, plan: Any, slot_id: int) -> tuple[list[torch.Tensor], tuple[int, ...]]:
    tensors: list[torch.Tensor] = []
    ids = []
    for layer_index in plan.full_attention_layer_indices:
        arena = persistent.layers[layer_index].arena
        for physical_id in arena.private_block_reservations[slot_id].reshape(-1).tolist():
            physical_id = int(physical_id)
            ids.append(int(layer_index) * 1_000_000 + physical_id)
            tensors.extend((arena.key_cache[physical_id], arena.value_cache[physical_id]))
    return tensors, tuple(ids)


def binding(state: RequestState, plan: Any) -> DispatchBinding:
    return DispatchBinding(
        request_id=state.request_id,
        slot_id=state.slot_id,
        lease=state.lease,
        physical_block_ids=reservation_ids(state.request, plan),
    )


def register_state(
    plan: Any,
    request: Any,
    registry: SlotEpochRegistry,
    *,
    request_id: str,
    slot_id: int,
    query: torch.Tensor,
    expected_rounds: int,
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
        expected_calls_per_layer=expected_rounds,
        initial_query_tokens=QUERY_TOKENS,
        kernel=kernel,
    )
    return RequestState(
        request_id=request_id,
        slot_id=slot_id,
        request=request,
        lease=lease,
        backend=register_multifork_backend(ledger),
        ledger=ledger,
        current=query,
        generated=[],
        logit_sha256=[],
        logits_cpu=[],
    )


def dispatch(
    model: Any,
    backbone: Any,
    registry: SlotEpochRegistry,
    state: RequestState,
    plan: Any,
    event: Mapping[str, Any],
) -> dict[str, Any]:
    require_dispatch(registry, binding(state, plan), request_id=state.request_id, slot_id=state.slot_id)
    original = backbone.config._attn_implementation
    try:
        backbone.config._attn_implementation = state.backend
        output = backbone(input_ids=state.current, past_key_values=state.request, use_cache=True)
        logits = _last_logits(model, output)
        require(bool(torch.isfinite(logits).all()), "non-finite scheduler logits")
        token = int(logits.argmax(-1).item())
        cpu = logits.detach().contiguous().cpu().float()
        state.generated.append(token)
        state.logit_sha256.append(sha256_tensor(cpu))
        state.logits_cpu.append(cpu)
        state.current = torch.tensor([[token]], dtype=torch.long, device=logits.device)
        del output, logits
    finally:
        backbone.config._attn_implementation = original
    return dict(event)


def close_state(state: RequestState) -> dict[str, Any]:
    receipt = state.ledger.verify_complete()
    _unregister_backend(state.backend)
    return receipt


def make_persistent(backbone: Any, plan: Any, document: torch.Tensor, cell: Mapping[str, Any]) -> tuple[Any, Any]:
    persistent = _build_document_cache(backbone, document)
    conversion = convert_all_qwen35_full_layers_to_vllm_q16(
        persistent,
        plan,
        page_size=int(cell["page_size"]),
        max_append_tokens=QUERY_TOKENS + TOTAL_ROUNDS,
        max_request_forks=RESIDENT_COUNT,
    )
    require(conversion.max_request_forks == RESIDENT_COUNT, "conversion slot count drift")
    return persistent, conversion


def group_digest(states: Sequence[RequestState], plan: Any) -> tuple[Any, Any]:
    ordered = sorted(states, key=lambda state: state.slot_id)
    group = SimpleNamespace(requests=tuple(state.request for state in ordered))
    kv = _request_logical_kv_digests(group, plan.full_attention_layer_indices)
    gdn = []
    for state in ordered:
        row = _linear_state_digest(state.request, plan.linear_layer_indices)
        row.pop("storage_keys")
        gdn.append({"slot_id": state.slot_id, **row})
    return kv, gdn


def trajectory(states: Sequence[RequestState]) -> list[dict[str, Any]]:
    return [
        {
            "slot_id": state.slot_id,
            "generated_token_ids": list(state.generated),
            "full_vocab_logit_sha256": list(state.logit_sha256),
        }
        for state in sorted(states, key=lambda state: state.slot_id)
    ]


def assert_exact(control: Sequence[RequestState], lifecycle: Sequence[RequestState]) -> None:
    left = sorted(control, key=lambda state: state.slot_id)
    right = sorted(lifecycle, key=lambda state: state.slot_id)
    require(len(left) == len(right) == RESIDENT_COUNT, "trajectory cardinality drift")
    for lhs, rhs in zip(left, right):
        require(lhs.slot_id == rhs.slot_id, "slot order drift")
        require(lhs.generated == rhs.generated, "generated token mismatch")
        require(lhs.logit_sha256 == rhs.logit_sha256, "full-vocabulary logit hash mismatch")
        require(
            all(torch.equal(a, b) for a, b in zip(lhs.logits_cpu, rhs.logits_cpu)),
            "full-vocabulary logits are not torch.equal",
        )


def run_control(model: Any, backbone: Any, plan: Any, document: torch.Tensor, queries: Sequence[torch.Tensor], kernel: Any, cell: Mapping[str, Any]) -> tuple[list[RequestState], dict[str, Any]]:
    persistent, conversion = make_persistent(backbone, plan, document, cell)
    document_before = _source_document_digests(persistent, plan.full_attention_layer_indices)
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=RESIDENT_COUNT,
        policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    registry = SlotEpochRegistry(RESIDENT_COUNT)
    states = [
        register_state(
            plan,
            request,
            registry,
            request_id=f"{cell['cell_id']}-control-slot-{slot_id}",
            slot_id=slot_id,
            query=queries[slot_id],
            expected_rounds=TOTAL_ROUNDS,
            kernel=kernel,
        )
        for slot_id, request in enumerate(group.requests)
    ]
    require_live_reservations_disjoint(tuple(binding(state, plan) for state in states))
    by_slot = {state.slot_id: state for state in states}
    expected = schedule_spec(str(cell["cell_id"]), lifecycle=False)
    observed = [dispatch(model, backbone, registry, by_slot[int(event["slot_id"])], plan, event) for event in expected]
    intercepts = [close_state(state) for state in states]
    torch.cuda.synchronize()
    document_after = _source_document_digests(persistent, plan.full_attention_layer_indices)
    require(document_before == document_after, "control document mutated")
    kv, gdn = group_digest(states, plan)
    return states, {
        "schedule": observed,
        "schedule_replay": replay_schedule(observed, expected=expected),
        "document_immutable": True,
        "document_sha256_before": document_before,
        "document_sha256_after": document_after,
        "reservation_rows": reservation_rows(group.requests, plan),
        "conversion": {
            "page_size": int(cell["page_size"]),
            "document_tokens": int(cell["document_tokens"]),
            "document_tail_tokens": int(cell["document_tokens"]) % int(cell["page_size"]),
            "document_payload_nbytes": int(conversion.document_payload_nbytes),
        },
        "trajectory": trajectory(states),
        "logical_kv": kv,
        "gdn_state": gdn,
        "intercepts": intercepts,
    }


def scrub_cancelled_slot(persistent: Any, plan: Any) -> dict[str, Any]:
    rows = []
    for layer_index in plan.full_attention_layer_indices:
        arena = persistent.layers[layer_index].arena
        require(arena._fork_cursor == RESIDENT_COUNT, "arena cursor drift before reclaim")
        ids = [int(value) for value in arena.private_block_reservations[CANCEL_SLOT].reshape(-1).tolist()]
        for physical_id in ids:
            arena.key_cache[physical_id].zero_()
            arena.value_cache[physical_id].zero_()
        arena._fork_cursor = CANCEL_SLOT
        rows.append({"layer_index": int(layer_index), "scrubbed_physical_block_ids": ids})
    torch.cuda.synchronize()
    return {"zero_scrubbed": True, "rewound_slot": CANCEL_SLOT, "layers": rows}


def run_lifecycle(model: Any, backbone: Any, plan: Any, document: torch.Tensor, queries: Sequence[torch.Tensor], kernel: Any, cell: Mapping[str, Any], expected_faults: Mapping[str, str]) -> tuple[list[RequestState], dict[str, Any]]:
    persistent, conversion = make_persistent(backbone, plan, document, cell)
    document_before = _source_document_digests(persistent, plan.full_attention_layer_indices)
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=RESIDENT_COUNT,
        policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    initial_reservations = reservation_rows(group.requests, plan)
    registry = SlotEpochRegistry(RESIDENT_COUNT)
    initial = [
        register_state(
            plan,
            request,
            registry,
            request_id=f"{cell['cell_id']}-initial-slot-{slot_id}",
            slot_id=slot_id,
            query=queries[slot_id],
            expected_rounds=CANCEL_AFTER_ROUNDS if slot_id == CANCEL_SLOT else TOTAL_ROUNDS,
            kernel=kernel,
        )
        for slot_id, request in enumerate(group.requests)
    ]
    require_live_reservations_disjoint(tuple(binding(state, plan) for state in initial))
    by_request_id = {state.request_id: state for state in initial}
    expected_schedule = schedule_spec(str(cell["cell_id"]), lifecycle=True)
    observed_schedule = []
    for event in expected_schedule[: RESIDENT_COUNT * CANCEL_AFTER_ROUNDS]:
        observed_schedule.append(dispatch(model, backbone, registry, by_request_id[str(event["request_id"])], plan, event))

    # FH2 and FH3 exercise the actual scheduled request/reservation path before
    # cleanup.  The expected mapping was frozen in the design preregistration.
    cross_gate = observe_gate(
        lambda: require_dispatch(
            registry,
            binding(initial[0], plan),
            request_id=initial[1].request_id,
            slot_id=initial[0].slot_id,
        )
    )
    cancelled_blocks, cancelled_ids = private_blocks(persistent, plan, CANCEL_SLOT)
    unscrubbed_gate = observe_gate(
        lambda: require_zero_scrubbed(
            cancelled_blocks,
            expected_physical_block_ids=cancelled_ids,
            observed_physical_block_ids=cancelled_ids,
        )
    )
    cancelled = initial[CANCEL_SLOT]
    cancelled_intercept = close_state(cancelled)
    registry.cancel(cancelled.lease)
    scrub = scrub_cancelled_slot(persistent, plan)
    clean_blocks, clean_ids = private_blocks(persistent, plan, CANCEL_SLOT)
    require_zero_scrubbed(
        clean_blocks,
        expected_physical_block_ids=cancelled_ids,
        observed_physical_block_ids=clean_ids,
    )
    request, _audit = _request_with_gdn_policy(
        persistent,
        plan,
        request_policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    replacement = register_state(
        plan,
        request,
        registry,
        request_id=f"{cell['cell_id']}-replacement-slot-{CANCEL_SLOT}",
        slot_id=CANCEL_SLOT,
        query=queries[CANCEL_SLOT],
        expected_rounds=TOTAL_ROUNDS,
        kernel=kernel,
    )
    require(reservation_ids(replacement.request, plan) == cancelled_ids, "reclaimed reservation differs")
    stale_gate = observe_gate(
        lambda: require_dispatch(
            registry,
            binding(cancelled, plan),
            request_id=cancelled.request_id,
            slot_id=cancelled.slot_id,
        )
    )
    faults = {
        "FH1_cancelled_lease_dispatch": stale_gate,
        "FH2_cross_request_dispatch": cross_gate,
        "FH3_reclaim_without_zero_scrub": unscrubbed_gate,
    }
    fault_outcomes = [
        {
            "fault_id": fault_id,
            "expected_gate": expected_faults[fault_id],
            "observed_gate": observed_gate,
            "matched_expected_gate": observed_gate == expected_faults[fault_id],
        }
        for fault_id, observed_gate in faults.items()
    ]

    active = initial[:CANCEL_SLOT] + [replacement]
    require_live_reservations_disjoint(tuple(binding(state, plan) for state in active))
    by_request_id.update({replacement.request_id: replacement})
    for event in expected_schedule[RESIDENT_COUNT * CANCEL_AFTER_ROUNDS :]:
        observed_schedule.append(dispatch(model, backbone, registry, by_request_id[str(event["request_id"])], plan, event))
    active_intercepts = [close_state(state) for state in active]
    torch.cuda.synchronize()
    document_after = _source_document_digests(persistent, plan.full_attention_layer_indices)
    require(document_before == document_after, "lifecycle document mutated")
    replay = replay_slot_events(registry.receipt())
    require(replay["final_epochs"] == [0, 0, 0, 1], "slot epoch replay drift")
    replayed_schedule = replay_schedule(observed_schedule, expected=expected_schedule)
    kv, gdn = group_digest(active, plan)
    return active, {
        "schedule": observed_schedule,
        "schedule_replay": replayed_schedule,
        "lease_events": registry.receipt(),
        "lease_replay": replay,
        "document_immutable": True,
        "document_sha256_before": document_before,
        "document_sha256_after": document_after,
        "initial_reservation_rows": initial_reservations,
        "replacement_reservation": reservation_rows((replacement.request,), plan)[0],
        "exact_private_reservation_reuse": True,
        "scrub_receipt": scrub,
        "conversion": {
            "page_size": int(cell["page_size"]),
            "document_tokens": int(cell["document_tokens"]),
            "document_tail_tokens": int(cell["document_tokens"]) % int(cell["page_size"]),
            "document_payload_nbytes": int(conversion.document_payload_nbytes),
        },
        "trajectory": trajectory(active),
        "logical_kv": kv,
        "gdn_state": gdn,
        "intercepts": {"cancelled": cancelled_intercept, "active": active_intercepts},
        "heldout_fault_outcomes": fault_outcomes,
        "detection_rate_reported": False,
    }


def run_gpu_shard(
    args: argparse.Namespace,
    design: Mapping[str, Any],
    static: Mapping[str, Any],
    *,
    debug_only: bool,
) -> dict[str, Any]:
    require(0 <= args.rank < FORMAL_WORLD_SIZE, "debug rank outside formal rank range")
    verify_design_sources(args, design)
    require(static.get("schema_version") == STATIC_SCHEMA, "static schema drift")
    require(static.get("design_preregistration_raw_sha256") == args.expected_design_sha256, "static/design drift")
    records, tokenizer, windows, audit = input_material(args, design)
    require(audit["windows_sha256"] == static.get("windows_sha256"), "runtime windows differ from static preregistration")
    window = windows[args.rank]
    rank_static = static["ranks"][args.rank]
    require(rank_static["rank"] == args.rank, "static rank order drift")
    require(rank_static["source_object"] == window.source_object, "static source object drift")

    torch.cuda.set_device(0)
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
    require(kernel_environment["matches_frozen_environment"] is True, "kernel environment drift")
    kernel = _resolve_vllm_unified_attention()
    expected_faults = {row["fault_id"]: row["expected_gate"] for row in design["heldout_faults"]}
    cells = []
    with torch.inference_mode():
        for cell_index, cell in enumerate(geometry_cells(design)):
            queries_cpu, query_audit = build_pg19_train_query_bank(
                records,
                tokenizer,
                window,
                document_tokens=int(cell["document_tokens"]),
                query_tokens=QUERY_TOKENS,
                count=RESIDENT_COUNT,
                query_stride=QUERY_STRIDE,
            )
            document_cpu = window.document_ids[: int(cell["document_tokens"])]
            frozen_cell = rank_static["cells"][cell_index]
            require(frozen_cell["cell_id"] == cell["cell_id"], "static cell order drift")
            require(sha256_tensor(document_cpu) == frozen_cell["document_token_ids_sha256"], "document token binding drift")
            require(
                [sha256_tensor(query) for query in queries_cpu] == frozen_cell["query_token_ids_sha256"],
                "query token binding drift",
            )
            document = document_cpu.unsqueeze(0).to(device="cuda:0", dtype=torch.int64)
            queries = [query.to(device="cuda:0", dtype=torch.int64) for query in queries_cpu]
            control_states, control = run_control(model, backbone, plan, document, queries, kernel, cell)
            lifecycle_states, lifecycle = run_lifecycle(
                model, backbone, plan, document, queries, kernel, cell, expected_faults
            )
            assert_exact(control_states, lifecycle_states)
            require(control["logical_kv"] == lifecycle["logical_kv"], "final logical KV mismatch")
            require(control["gdn_state"] == lifecycle["gdn_state"], "final GDN state mismatch")
            cells.append(
                {
                    "cell_id": cell["cell_id"],
                    "page_size": int(cell["page_size"]),
                    "document_tokens": int(cell["document_tokens"]),
                    "document_tail_tokens": int(cell["document_tokens"]) % int(cell["page_size"]),
                    "query_audit": query_audit,
                    "control": control,
                    "lifecycle": lifecycle,
                    "cross_cell": {
                        "full_vocab_logits_torch_equal": True,
                        "generated_tokens_equal": True,
                        "final_logical_kv_equal": True,
                        "final_gdn_state_equal": True,
                        "document_immutable": True,
                    },
                }
            )

    all_faults_match = all(
        row["matched_expected_gate"]
        for cell in cells
        for row in cell["lifecycle"]["heldout_fault_outcomes"]
    )
    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": DEBUG_SCHEMA if debug_only else FORMAL_SHARD_SCHEMA,
        "status": "completed",
        "debug_only": debug_only,
        "formal_candidate_shard": not debug_only,
        "formal_evidence_eligible": False,
        "rank": args.rank,
        "formal_world_size": FORMAL_WORLD_SIZE,
        "protocol": SCHEDULER_PROTOCOL,
        "design_preregistration_raw_sha256": args.expected_design_sha256,
        "static_manifest_raw_sha256": args.expected_static_sha256,
        "source": {
            "pg19_train_only": True,
            "source_object": window.source_object,
            "source_id": str(window.source_id),
            "windows_sha256": audit["windows_sha256"],
        },
        "hardware": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "audited_packages": dict(AUDITED_PACKAGES),
            "kernel_mode": KERNEL_MODE,
            "kernel_environment_matches": True,
            "model_geometry": _audit_model_config_geometry(args.model),
        },
        "cells": cells,
        "debug_gates": {
            "geometry_count": len(cells),
            "scheduler_managed_forced_interleaving": True,
            "cancellation_reclamation_executed": True,
            "all_clean_semantic_and_storage_gates_passed": True,
            "all_faults_hit_preregistered_gate": all_faults_match,
            "detection_rate_reported": False,
        },
        "ready_for_fresh_formal_launch": all_faults_match and len(cells) == 2,
        "formal_launch_requirements": design["formal_launch_requirements"],
        "claim_boundary": design["claim_boundary"],
    }


def aggregate_formal(
    args: argparse.Namespace,
    design: Mapping[str, Any],
    static: Mapping[str, Any],
) -> dict[str, Any]:
    require(args.shard_dir is not None, "formal aggregate requires shard directory")
    paths = sorted(args.shard_dir.glob("scheduler-interleave-formal-shard-*.json"))
    require(len(paths) == FORMAL_WORLD_SIZE, "formal aggregate requires exactly eight shards")
    expected_faults = {row["fault_id"]: row["expected_gate"] for row in design["heldout_faults"]}
    receipts = []
    source_objects = []
    fault_rows = []
    for expected_rank, path in enumerate(paths):
        raw = path.read_bytes()
        shard = json.loads(raw)
        require(shard.get("schema_version") == FORMAL_SHARD_SCHEMA, "formal shard schema drift")
        require(shard.get("rank") == expected_rank, "formal rank order drift")
        require(shard.get("debug_only") is False, "debug shard entered formal aggregate")
        require(shard.get("design_preregistration_raw_sha256") == args.expected_design_sha256, "shard/design drift")
        require(shard.get("static_manifest_raw_sha256") == args.expected_static_sha256, "shard/static drift")
        require(len(shard.get("cells", [])) == len(geometry_cells(design)), "formal geometry count drift")
        for cell, design_cell in zip(shard["cells"], geometry_cells(design)):
            require(cell["cell_id"] == design_cell["cell_id"], "formal geometry order drift")
            replay_schedule(cell["control"]["schedule"], expected=schedule_spec(cell["cell_id"], lifecycle=False))
            replay_schedule(cell["lifecycle"]["schedule"], expected=schedule_spec(cell["cell_id"], lifecycle=True))
            require(
                replay_slot_events(cell["lifecycle"]["lease_events"])
                == cell["lifecycle"]["lease_replay"],
                "formal lease replay drift",
            )
            require(all(cell["cross_cell"].values()), "formal clean cross-cell gate failed")
            for row in cell["lifecycle"]["heldout_fault_outcomes"]:
                require(row["expected_gate"] == expected_faults[row["fault_id"]], "fault prereg mapping drift")
                require(row["observed_gate"] == row["expected_gate"], "held-out fault missed expected gate")
                fault_rows.append({"rank": expected_rank, "cell_id": cell["cell_id"], **row})
        source_objects.append(shard["source"]["source_object"])
        receipts.append({"rank": expected_rank, "path": path.name, "sha256": sha256_bytes(raw), "bytes": len(raw)})
    require(len(set(source_objects)) == FORMAL_WORLD_SIZE, "formal ranks do not use distinct PG19 books")
    return {
        "schema_version": FORMAL_AGGREGATE_SCHEMA,
        "status": "completed",
        "debug_only": False,
        "formal_evidence_eligible": True,
        "scientific_run_valid": True,
        "protocol": SCHEDULER_PROTOCOL,
        "rank_count": FORMAL_WORLD_SIZE,
        "geometry_count_per_rank": len(geometry_cells(design)),
        "distinct_pg19_train_books": len(set(source_objects)),
        "all_clean_semantic_and_storage_gates_passed": True,
        "heldout_fault_outcomes": fault_rows,
        "heldout_fault_trial_count": len(fault_rows),
        "heldout_fault_expected_gate_misses": 0,
        "design_preregistration_raw_sha256": args.expected_design_sha256,
        "static_manifest_raw_sha256": args.expected_static_sha256,
        "shards": receipts,
        "claim_boundary": design["claim_boundary"],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--stage",
        choices=("static", "mock", "debug", "formal-shard", "formal-aggregate"),
        required=True,
    )
    result.add_argument("--design-preregistration", type=Path, required=True)
    result.add_argument("--expected-design-sha256", required=True)
    result.add_argument("--static-manifest", type=Path)
    result.add_argument("--expected-static-sha256", default="")
    result.add_argument("--model", type=Path)
    result.add_argument("--model-artifact-ledger", type=Path)
    result.add_argument("--model-weight-ledger", type=Path)
    result.add_argument("--pg19-data", type=Path)
    result.add_argument("--pg19-manifest", type=Path)
    result.add_argument("--upstream-code-ledger", type=Path, required=True)
    result.add_argument("--rank", type=int, default=0)
    result.add_argument("--shard-dir", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    design = load_bound_json(args.design_preregistration, args.expected_design_sha256, "design preregistration")
    require(design.get("schema_version") == "qcomem-forkaudit-scheduler-interleave-design-v1", "design schema drift")
    if args.stage == "mock":
        value = run_mock(design)
    elif args.stage == "formal-aggregate":
        require(args.static_manifest is not None, "formal aggregate requires static manifest")
        static = load_bound_json(args.static_manifest, args.expected_static_sha256, "static manifest")
        value = aggregate_formal(args, design, static)
    else:
        required = (
            args.model,
            args.model_artifact_ledger,
            args.model_weight_ledger,
            args.pg19_data,
            args.pg19_manifest,
        )
        require(all(path is not None for path in required), "model/data/ledger paths are required")
        if args.stage == "static":
            value = build_static(args, design)
        else:
            require(args.static_manifest is not None, "GPU shard stage requires static manifest")
            static = load_bound_json(args.static_manifest, args.expected_static_sha256, "static manifest")
            value = run_gpu_shard(args, design, static, debug_only=args.stage == "debug")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
