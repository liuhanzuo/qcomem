from __future__ import annotations

"""Formal aligned-prefix cancellation/reclamation transfer for ForkAudit.

This is deliberately a lifecycle transfer on the already audited Qwen3.5 +
vLLM Q16 adapter, not evidence for a second model or runtime implementation.
Each rank uses one independently selected PG-19 train book.  A clean control
keeps four requests alive for four rounds.  The lifecycle arm runs two rounds,
cancels the suffix slots, scrubs and reclaims their exact private reservation
pages, rejects one stale-handle schedule, and completes both surviving and
replacement requests.  Full-vocabulary logits, final KV/GDN state, immutable
document bytes, aligned-page behavior, and pointer-free lease receipts are
checked without trusting producer ``passed`` fields during aggregation.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from qcomem_forkaudit_lifecycle_transfer import (
    LIFECYCLE_PROTOCOL,
    LifecycleContractError,
    SlotEpochRegistry,
    SlotLease,
    pairwise_disjoint,
    replay_slot_events,
)
from qcomem_joint_policy import (
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
    sha256_file,
)
from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
from qcomem_qwen35_vllm_paged_integration import (
    convert_all_qwen35_full_layers_to_vllm_q16,
)
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
    _model_manifest_sha,
    _request_logical_kv_digests,
    _resolve_backbone,
    _source_document_digests,
    _unregister_backend,
)


SHARD_SCHEMA = "qcomem-forkaudit-lifecycle-transfer-shard-v1"
STATIC_SCHEMA = "qcomem-forkaudit-lifecycle-transfer-static-v1"
AGGREGATE_SCHEMA = "qcomem-forkaudit-lifecycle-transfer-aggregate-v1"
FORMAL_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
FORMAL_MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
FORMAL_WORLD_SIZE = 8
FORMAL_BOOKS = 8
FORMAL_DOCUMENT_TOKENS = 4096
FORMAL_QUERY_TOKENS = 32
FORMAL_PAGE_SIZE = 128
FORMAL_RESIDENT_COUNT = 4
FORMAL_TOTAL_ROUNDS = 4
FORMAL_CANCEL_AFTER_ROUNDS = 2
FORMAL_CANCEL_SLOTS = (2, 3)
FORMAL_WINDOW_STRIDE = 263
FORMAL_QUERY_STRIDE = 64
FORMAL_CANDIDATES = 8
FORMAL_SEED = 20260819
FORMAL_FULL_LAYERS = tuple(range(3, 40, 4))
FORMAL_LINEAR_LAYERS = tuple(index for index in range(40) if index not in FORMAL_FULL_LAYERS)
EXPECTED_PG19_DATA_SHA256 = "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
EXPECTED_PG19_MANIFEST_SHA256 = "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _sha256_tensor(value: torch.Tensor) -> str:
    return _sha256_bytes(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_bound_json(path: Path, expected_sha256: str, label: str) -> Any:
    raw = path.read_bytes()
    _require(_sha256_bytes(raw) == expected_sha256, f"{label} raw SHA mismatch")
    return json.loads(raw)


def _input_material(args: argparse.Namespace) -> tuple[Any, Any, Any, list[Any], dict[str, Any]]:
    records, data_audit = audit_pg19_train_calibration(
        args.pg19_data,
        args.pg19_manifest,
        expected_data_sha256=args.expected_pg19_sha256,
        expected_manifest_sha256=args.expected_pg19_manifest_sha256,
        minimum_books=FORMAL_BOOKS,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    windows, windows_sha = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=FORMAL_BOOKS,
        document_tokens=FORMAL_DOCUMENT_TOKENS,
        query_tokens=FORMAL_QUERY_TOKENS,
        stride=FORMAL_WINDOW_STRIDE,
        candidate_windows_per_book=FORMAL_CANDIDATES,
        seed=FORMAL_SEED,
    )
    _require(len(windows) == FORMAL_WORLD_SIZE, "formal lifecycle transfer requires eight windows")
    if args.expected_windows_sha256:
        _require(windows_sha == args.expected_windows_sha256, "aligned transfer windows SHA mismatch")
    query_banks = []
    for rank, window in enumerate(windows):
        queries, audit = build_pg19_train_query_bank(
            records,
            tokenizer,
            window,
            document_tokens=FORMAL_DOCUMENT_TOKENS,
            query_tokens=FORMAL_QUERY_TOKENS,
            count=FORMAL_RESIDENT_COUNT,
            query_stride=FORMAL_QUERY_STRIDE,
        )
        query_banks.append(
            {
                "rank": rank,
                "window_index": rank,
                "source_object": window.source_object,
                "source_id": str(window.source_id),
                "document_token_ids_sha256": _sha256_tensor(window.document_ids),
                "query_bank": audit,
                "query_token_ids_sha256": [_sha256_tensor(query) for query in queries],
            }
        )
    return records, tokenizer, windows, query_banks, {
        "data_audit": data_audit,
        "windows_sha256": windows_sha,
    }


def _formal_config() -> dict[str, Any]:
    return {
        "protocol": LIFECYCLE_PROTOCOL,
        "model_id": FORMAL_MODEL_ID,
        "model_revision": FORMAL_MODEL_REVISION,
        "world_size": FORMAL_WORLD_SIZE,
        "pg19_train_books": FORMAL_BOOKS,
        "document_tokens": FORMAL_DOCUMENT_TOKENS,
        "query_tokens": FORMAL_QUERY_TOKENS,
        "page_size": FORMAL_PAGE_SIZE,
        "document_tail_tokens": 0,
        "resident_count": FORMAL_RESIDENT_COUNT,
        "total_rounds": FORMAL_TOTAL_ROUNDS,
        "cancel_after_rounds": FORMAL_CANCEL_AFTER_ROUNDS,
        "cancel_slots": list(FORMAL_CANCEL_SLOTS),
        "window_stride": FORMAL_WINDOW_STRIDE,
        "query_stride": FORMAL_QUERY_STRIDE,
        "candidate_windows": FORMAL_CANDIDATES,
        "seed": FORMAL_SEED,
        "kv_policy": SHARED_REUSE,
        "gdn_policy": GDN_BORROW_IMMUTABLE_BASE,
        "fault": "schedule-a-cancelled-handle-after-slot-reclamation",
        "expected_fault_gate": "STALE_SLOT_LEASE",
        "scheduler": "single-cuda-stream-phase-interleaved-sequential",
    }


def build_static(args: argparse.Namespace) -> dict[str, Any]:
    _require(args.expected_pg19_sha256 == EXPECTED_PG19_DATA_SHA256, "PG19 data binding drift")
    _require(
        args.expected_pg19_manifest_sha256 == EXPECTED_PG19_MANIFEST_SHA256,
        "PG19 manifest binding drift",
    )
    _records, _tokenizer, _windows, query_banks, input_audit = _input_material(args)
    model_manifest_sha, model_manifest_rows = _model_manifest_sha(args.model)
    value = {
        "schema_version": STATIC_SCHEMA,
        "created_before_gpu_execution": True,
        "formal_config": _formal_config(),
        "formal_config_sha256": _sha256_json(_formal_config()),
        "pg19_data_sha256": sha256_file(args.pg19_data),
        "pg19_manifest_sha256": sha256_file(args.pg19_manifest),
        "windows_sha256": input_audit["windows_sha256"],
        "model_manifest_sha256": model_manifest_sha,
        "model_manifest_rows": model_manifest_rows,
        "query_banks": query_banks,
        "design_boundary": {
            "second_model_or_runtime_implementation": False,
            "different_recurrent_backend": False,
            "same_qwen35_vllm_adapter": True,
            "new_aligned_page_geometry": True,
            "new_cancellation_reclamation_lifecycle": True,
            "new_slot_epoch_fault": True,
            "concurrent_kernel_execution": False,
            "transfer_claim_limited_to_lifecycle_and_geometry": True,
        },
    }
    return value


@dataclass
class RequestState:
    slot_id: int
    request: Any
    lease: SlotLease
    backend: str
    ledger: Any
    current: torch.Tensor
    query_sha256: str
    generated: list[int]
    logit_sha256: list[str]
    logits_cpu: list[torch.Tensor]


def _register_state(
    plan: Any,
    request: Any,
    *,
    slot_id: int,
    lease: SlotLease,
    query: torch.Tensor,
    expected_rounds: int,
    kernel: Any,
) -> RequestState:
    for layer_index in plan.full_attention_layer_indices:
        request.layers[layer_index].sequence.strict_mask_check = False
    ledger = MultiForkHitLedger(
        plan,
        request,
        request_index=slot_id,
        resident_count=FORMAL_RESIDENT_COUNT,
        request_policy=SHARED_REUSE,
        expected_calls_per_layer=expected_rounds,
        initial_query_tokens=FORMAL_QUERY_TOKENS,
        kernel=kernel,
    )
    backend = register_multifork_backend(ledger)
    return RequestState(
        slot_id=slot_id,
        request=request,
        lease=lease,
        backend=backend,
        ledger=ledger,
        current=query,
        query_sha256=_sha256_tensor(query),
        generated=[],
        logit_sha256=[],
        logits_cpu=[],
    )


def _run_rounds(
    model: Any,
    backbone: Any,
    registry: SlotEpochRegistry,
    states: Sequence[RequestState],
    *,
    start_round: int,
    rounds: int,
) -> list[dict[str, int]]:
    schedule = []
    original = backbone.config._attn_implementation
    try:
        for round_index in range(start_round, start_round + rounds):
            for state in states:
                registry.validate(state.lease)
                backbone.config._attn_implementation = state.backend
                output = backbone(
                    input_ids=state.current,
                    past_key_values=state.request,
                    use_cache=True,
                )
                logits = _last_logits(model, output)
                _require(bool(torch.isfinite(logits).all()), "non-finite lifecycle logits")
                token = int(logits.argmax(-1).item())
                cpu = logits.detach().contiguous().cpu().float()
                state.generated.append(token)
                state.logit_sha256.append(_sha256_tensor(cpu))
                state.logits_cpu.append(cpu)
                state.current = torch.tensor([[token]], dtype=torch.long, device=logits.device)
                schedule.append({"round_index": round_index, "slot_id": state.slot_id})
                del output, logits
    finally:
        backbone.config._attn_implementation = original
    torch.cuda.synchronize()
    return schedule


def _reservation_rows(requests: Sequence[Any], plan: Any) -> list[dict[str, Any]]:
    result = []
    for slot_id, request in enumerate(requests):
        layers = {}
        for layer_index in plan.full_attention_layer_indices:
            sequence = request.layers[layer_index].sequence
            layers[str(layer_index)] = [int(value) for value in sequence.reservations.reshape(-1).tolist()]
        result.append({"slot_id": slot_id, "layers": layers, "sha256": _sha256_json(layers)})
    return result


def _scrub_and_rewind_suffix(persistent: Any, plan: Any) -> dict[str, Any]:
    rows = []
    for layer_index in plan.full_attention_layer_indices:
        arena = persistent.layers[layer_index].arena
        _require(arena._fork_cursor == FORMAL_RESIDENT_COUNT, "arena fork cursor drift before reclaim")
        scrubbed = []
        for slot_id in FORMAL_CANCEL_SLOTS:
            ids = [int(value) for value in arena.private_block_reservations[slot_id].reshape(-1).tolist()]
            for physical_id in ids:
                arena.key_cache[physical_id].zero_()
                arena.value_cache[physical_id].zero_()
            scrubbed.extend(ids)
        arena._fork_cursor = min(FORMAL_CANCEL_SLOTS)
        rows.append(
            {
                "layer_index": int(layer_index),
                "reclaimed_slots": list(FORMAL_CANCEL_SLOTS),
                "scrubbed_physical_block_ids": scrubbed,
                "rewound_fork_cursor": int(arena._fork_cursor),
            }
        )
    torch.cuda.synchronize()
    return {"zero_scrub_before_reassignment": True, "layers": rows}


def _make_persistent(backbone: Any, plan: Any, document: torch.Tensor) -> Any:
    persistent = _build_document_cache(backbone, document)
    conversion = convert_all_qwen35_full_layers_to_vllm_q16(
        persistent,
        plan,
        page_size=FORMAL_PAGE_SIZE,
        max_append_tokens=FORMAL_QUERY_TOKENS + FORMAL_TOTAL_ROUNDS,
        max_request_forks=FORMAL_RESIDENT_COUNT,
    )
    _require(conversion.max_request_forks == FORMAL_RESIDENT_COUNT, "conversion slot count drift")
    _require(conversion.document_payload_nbytes > 0, "empty aligned document payload")
    return persistent


def _group_digest(requests: Sequence[Any], plan: Any) -> tuple[list[Any], list[Any]]:
    group = SimpleNamespace(requests=tuple(requests))
    kv = _request_logical_kv_digests(group, plan.full_attention_layer_indices)
    gdn = []
    for slot_id, request in enumerate(requests):
        row = _linear_state_digest(request, plan.linear_layer_indices)
        row.pop("storage_keys")
        gdn.append({"slot_id": slot_id, **row})
    return kv, gdn


def _trajectory_receipt(states: Sequence[RequestState]) -> list[dict[str, Any]]:
    return [
        {
            "slot_id": state.slot_id,
            "query_token_ids_sha256": state.query_sha256,
            "generated_token_ids": list(state.generated),
            "full_vocab_step_logit_sha256": list(state.logit_sha256),
        }
        for state in states
    ]


def _assert_exact(control: Sequence[RequestState], observed: Sequence[RequestState]) -> None:
    _require(len(control) == len(observed), "trajectory cardinality drift")
    for left, right in zip(control, observed):
        _require(left.slot_id == right.slot_id, "slot order drift")
        _require(left.query_sha256 == right.query_sha256, "query identity drift")
        _require(left.generated == right.generated, "generated tokens differ after lifecycle transfer")
        _require(left.logit_sha256 == right.logit_sha256, "logit hashes differ after lifecycle transfer")
        _require(len(left.logits_cpu) == len(right.logits_cpu) == FORMAL_TOTAL_ROUNDS, "logit step count drift")
        _require(
            all(torch.equal(a, b) for a, b in zip(left.logits_cpu, right.logits_cpu)),
            "full-vocabulary logits are not torch.equal",
        )


def _close_states(states: Sequence[RequestState]) -> list[dict[str, Any]]:
    rows = []
    for state in states:
        rows.append(state.ledger.verify_complete())
        _unregister_backend(state.backend)
    return rows


def _run_control(model: Any, backbone: Any, plan: Any, document: torch.Tensor, queries: Sequence[torch.Tensor], kernel: Any) -> tuple[list[RequestState], dict[str, Any]]:
    persistent = _make_persistent(backbone, plan, document)
    before = _source_document_digests(persistent, plan.full_attention_layer_indices)
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=FORMAL_RESIDENT_COUNT,
        policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    registry = SlotEpochRegistry(FORMAL_RESIDENT_COUNT)
    states = [
        _register_state(
            plan,
            request,
            slot_id=slot_id,
            lease=registry.acquire(slot_id, f"control-slot-{slot_id}"),
            query=queries[slot_id],
            expected_rounds=FORMAL_TOTAL_ROUNDS,
            kernel=kernel,
        )
        for slot_id, request in enumerate(group.requests)
    ]
    schedule = _run_rounds(model, backbone, registry, states, start_round=0, rounds=FORMAL_TOTAL_ROUNDS)
    intercepts = _close_states(states)
    after = _source_document_digests(persistent, plan.full_attention_layer_indices)
    _require(before == after, "control document mutated")
    kv, gdn = _group_digest([state.request for state in states], plan)
    receipt = {
        "trajectory": _trajectory_receipt(states),
        "schedule": schedule,
        "lease_events": registry.receipt(),
        "document_sha256_before": before,
        "document_sha256_after": after,
        "document_immutable": True,
        "reservation_rows": _reservation_rows([state.request for state in states], plan),
        "logical_kv": kv,
        "gdn_state": gdn,
        "intercepts": intercepts,
    }
    return states, receipt


def _run_lifecycle(model: Any, backbone: Any, plan: Any, document: torch.Tensor, queries: Sequence[torch.Tensor], kernel: Any) -> tuple[list[RequestState], dict[str, Any]]:
    persistent = _make_persistent(backbone, plan, document)
    document_before = _source_document_digests(persistent, plan.full_attention_layer_indices)
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=FORMAL_RESIDENT_COUNT,
        policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    initial_reservations = _reservation_rows(group.requests, plan)
    for layer_index in plan.full_attention_layer_indices:
        sets = [row["layers"][str(layer_index)] for row in initial_reservations]
        _require(pairwise_disjoint(sets), "initial private reservations overlap")
    registry = SlotEpochRegistry(FORMAL_RESIDENT_COUNT)
    initial_states = []
    for slot_id, request in enumerate(group.requests):
        expected = FORMAL_CANCEL_AFTER_ROUNDS if slot_id in FORMAL_CANCEL_SLOTS else FORMAL_TOTAL_ROUNDS
        initial_states.append(
            _register_state(
                plan,
                request,
                slot_id=slot_id,
                lease=registry.acquire(slot_id, f"initial-slot-{slot_id}"),
                query=queries[slot_id],
                expected_rounds=expected,
                kernel=kernel,
            )
        )
    schedule = _run_rounds(
        model,
        backbone,
        registry,
        initial_states,
        start_round=0,
        rounds=FORMAL_CANCEL_AFTER_ROUNDS,
    )
    cancelled = [initial_states[index] for index in FORMAL_CANCEL_SLOTS]
    survivors = [initial_states[index] for index in range(min(FORMAL_CANCEL_SLOTS))]
    cancelled_intercepts = _close_states(cancelled)
    for state in cancelled:
        registry.cancel(state.lease)
    scrub = _scrub_and_rewind_suffix(persistent, plan)
    replacements = []
    for slot_id in FORMAL_CANCEL_SLOTS:
        request, _audit = _request_with_gdn_policy(
            persistent,
            plan,
            request_policy=SHARED_REUSE,
            gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
        )
        lease = registry.acquire(slot_id, f"replacement-slot-{slot_id}")
        replacements.append(
            _register_state(
                plan,
                request,
                slot_id=slot_id,
                lease=lease,
                query=queries[slot_id],
                expected_rounds=FORMAL_TOTAL_ROUNDS,
                kernel=kernel,
            )
        )
    reclaimed_reservations = _reservation_rows([state.request for state in replacements], plan)
    for offset, slot_id in enumerate(FORMAL_CANCEL_SLOTS):
        _require(
            reclaimed_reservations[offset]["layers"] == initial_reservations[slot_id]["layers"],
            "reclaimed request did not reuse the cancelled slot reservation",
        )

    stale_gate = None
    try:
        registry.validate(cancelled[0].lease)
    except LifecycleContractError as error:
        stale_gate = error.gate_id
    _require(stale_gate == "STALE_SLOT_LEASE", "stale-handle mutant escaped or hit wrong gate")
    registry.validate(replacements[0].lease)

    schedule.extend(
        _run_rounds(
            model,
            backbone,
            registry,
            survivors,
            start_round=FORMAL_CANCEL_AFTER_ROUNDS,
            rounds=FORMAL_TOTAL_ROUNDS - FORMAL_CANCEL_AFTER_ROUNDS,
        )
    )
    schedule.extend(
        _run_rounds(
            model,
            backbone,
            registry,
            replacements,
            start_round=0,
            rounds=FORMAL_TOTAL_ROUNDS,
        )
    )
    survivor_intercepts = _close_states(survivors)
    replacement_intercepts = _close_states(replacements)
    document_after = _source_document_digests(persistent, plan.full_attention_layer_indices)
    _require(document_before == document_after, "lifecycle document mutated")
    all_final = survivors + replacements
    all_final.sort(key=lambda state: state.slot_id)
    for state in all_final:
        for layer_index in plan.full_attention_layer_indices:
            sequence = state.request.layers[layer_index].sequence
            _require(sequence.partial_tail_staging_copy_nbytes == 0, "aligned prefix performed tail COW")
    replay = replay_slot_events(registry.receipt())
    _require(replay["final_epochs"] == [0, 0, 1, 1], "lifecycle epoch replay drift")
    kv, gdn = _group_digest([state.request for state in all_final], plan)
    return all_final, {
        "trajectory": _trajectory_receipt(all_final),
        "schedule": schedule,
        "lease_events": registry.receipt(),
        "lease_replay": replay,
        "document_sha256_before": document_before,
        "document_sha256_after": document_after,
        "document_immutable": True,
        "initial_reservation_rows": initial_reservations,
        "reclaimed_reservation_rows": reclaimed_reservations,
        "cancelled_suffix_slots": list(FORMAL_CANCEL_SLOTS),
        "exact_private_reservation_slot_reuse": True,
        "scrub_receipt": scrub,
        "aligned_prefix_no_partial_tail_copy": True,
        "partial_tail_staging_copy_nbytes": 0,
        "stale_handle_mutant": {
            "fault": "schedule-cancelled-handle-after-reclaim",
            "expected_gate": "STALE_SLOT_LEASE",
            "observed_gate": stale_gate,
            "detected": True,
            "wrong_gate": False,
            "matched_clean_replacement_accepted": True,
        },
        "logical_kv": kv,
        "gdn_state": gdn,
        "intercepts": {
            "cancelled": cancelled_intercepts,
            "survivors": survivor_intercepts,
            "replacements": replacement_intercepts,
        },
    }


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    _require(0 <= args.rank < FORMAL_WORLD_SIZE, "invalid rank")
    static = _load_bound_json(args.static_manifest, args.expected_static_manifest_sha256, "static manifest")
    _require(static.get("schema_version") == STATIC_SCHEMA, "static schema drift")
    _require(static.get("formal_config") == _formal_config(), "static formal config drift")
    _require(static.get("windows_sha256") == args.expected_windows_sha256, "static windows binding drift")
    _records, _tokenizer, windows, query_banks, _audit = _input_material(args)
    window = windows[args.rank]
    static_rank = static["query_banks"][args.rank]
    _require(query_banks[args.rank] == static_rank, "runtime inputs differ from preregistration")
    queries_cpu, query_audit = build_pg19_train_query_bank(
        _records,
        _tokenizer,
        window,
        document_tokens=FORMAL_DOCUMENT_TOKENS,
        query_tokens=FORMAL_QUERY_TOKENS,
        count=FORMAL_RESIDENT_COUNT,
        query_stride=FORMAL_QUERY_STRIDE,
    )
    torch.cuda.set_device(0)
    expected_uuid = args.expected_gpu_uuid
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid,
        "rank CUDA isolation differs from launcher assignment",
    )
    identity = subprocess.run(
        [
            "nvidia-smi",
            f"--id={expected_uuid}",
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    columns = [item.strip() for item in identity.stdout.strip().split(",")]
    _require(len(columns) == 3 and columns[0] == expected_uuid, "rank GPU UUID differs from launcher assignment")
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=FORMAL_MODEL_REVISION,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    _require(tuple(plan.full_attention_layer_indices) == FORMAL_FULL_LAYERS, "full-layer plan drift")
    _require(tuple(plan.linear_layer_indices) == FORMAL_LINEAR_LAYERS, "linear-layer plan drift")
    geometry = _audit_model_config_geometry(args.model)
    kernel_environment = audit_frozen_kernel_environment()
    _require(kernel_environment["matches_frozen_environment"] is True, "kernel environment drift")
    kernel = _resolve_vllm_unified_attention()
    document = window.document_ids.unsqueeze(0).to(device="cuda:0", dtype=torch.int64)
    _require(tuple(document.shape) == (1, FORMAL_DOCUMENT_TOKENS), "aligned document shape drift")
    _require(int(document.shape[1]) % FORMAL_PAGE_SIZE == 0, "prefix is not page aligned")
    queries = [query.to(device="cuda:0", dtype=torch.int64) for query in queries_cpu]

    with torch.inference_mode():
        control_states, control = _run_control(model, backbone, plan, document, queries, kernel)
        lifecycle_states, lifecycle = _run_lifecycle(model, backbone, plan, document, queries, kernel)
        _assert_exact(control_states, lifecycle_states)
        _require(control["logical_kv"] == lifecycle["logical_kv"], "final logical KV differs")
        _require(control["gdn_state"] == lifecycle["gdn_state"], "final GDN state differs")

    properties = torch.cuda.get_device_properties(0)
    return {
        "schema_version": SHARD_SCHEMA,
        "status": "completed",
        "passed": True,
        "rank": args.rank,
        "world_size": FORMAL_WORLD_SIZE,
        "protocol": LIFECYCLE_PROTOCOL,
        "formal_config": _formal_config(),
        "static_manifest_raw_sha256": args.expected_static_manifest_sha256,
        "code_ledger_raw_sha256": args.expected_code_ledger_sha256,
        "model_weight_ledger_raw_sha256": args.expected_model_weight_ledger_sha256,
        "source": {
            "pg19_train_only": True,
            "source_object": window.source_object,
            "source_id": str(window.source_id),
            "document_token_ids_sha256": _sha256_tensor(document),
            "query_bank": query_audit,
            "windows_sha256": args.expected_windows_sha256,
        },
        "hardware": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "uuid": columns[0],
            "name": columns[1],
            "total_memory_mib": int(columns[2]),
            "compute_capability": [int(properties.major), int(properties.minor)],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "audited_packages": dict(AUDITED_PACKAGES),
            "kernel_mode": KERNEL_MODE,
            "kernel_environment_matches": True,
            "model_geometry": geometry,
        },
        "control": control,
        "lifecycle": lifecycle,
        "cross_cell": {
            "all_full_vocab_logits_torch_equal": True,
            "all_full_vocab_logit_sha256_equal": True,
            "all_generated_tokens_equal": True,
            "all_final_logical_kv_equal": True,
            "all_final_gdn_state_equal": True,
            "document_bytes_immutable": True,
            "eight_core_obligations": {
                "input_binding": True,
                "immutable_document": True,
                "private_reservation_disjointness": True,
                "aligned_append_without_tail_copy": True,
                "cancel_invalidation": True,
                "zero_scrub_before_reassignment": True,
                "epoch_bound_reclamation": True,
                "semantic_equivalence_after_reclamation": True,
            },
        },
        "claim_boundaries": static["design_boundary"],
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    static = _load_bound_json(args.static_manifest, args.expected_static_manifest_sha256, "static manifest")
    paths = sorted(args.shard_dir.glob("forkaudit-lifecycle-shard-*.json"))
    _require(len(paths) == FORMAL_WORLD_SIZE, "aggregate requires exactly eight lifecycle shards")
    shards = []
    receipts = []
    for expected_rank, path in enumerate(paths):
        raw = path.read_bytes()
        shard = json.loads(raw)
        _require(shard.get("schema_version") == SHARD_SCHEMA, "shard schema drift")
        _require(shard.get("rank") == expected_rank, "shard rank order drift")
        _require(shard.get("static_manifest_raw_sha256") == args.expected_static_manifest_sha256, "shard/static binding drift")
        _require(
            shard.get("model_weight_ledger_raw_sha256")
            == args.expected_model_weight_ledger_sha256,
            "shard/model-weight binding drift",
        )
        _require(shard.get("formal_config") == static["formal_config"], "shard config drift")
        cross = shard.get("cross_cell")
        _require(isinstance(cross, dict) and all(value is True for key, value in cross.items() if key != "eight_core_obligations"), "cross-cell gate failed")
        _require(all(cross["eight_core_obligations"].values()), "a core lifecycle obligation failed")
        lifecycle = shard["lifecycle"]
        _require(replay_slot_events(lifecycle["lease_events"]) == lifecycle["lease_replay"], "lease replay differs")
        _require(lifecycle["stale_handle_mutant"]["observed_gate"] == "STALE_SLOT_LEASE", "mutant gate drift")
        _require(lifecycle["aligned_prefix_no_partial_tail_copy"] is True, "aligned no-tail-copy gate failed")
        shards.append(shard)
        receipts.append({"rank": expected_rank, "path": path.name, "sha256": _sha256_bytes(raw), "bytes": len(raw)})
    source_objects = [shard["source"]["source_object"] for shard in shards]
    _require(len(set(source_objects)) == FORMAL_BOOKS, "rank source books are not distinct")
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "passed": True,
        "scientific_run_valid": True,
        "scientific_outcome": "valid_positive_lifecycle_transfer",
        "protocol": LIFECYCLE_PROTOCOL,
        "formal_config": static["formal_config"],
        "rank_count": len(shards),
        "distinct_pg19_train_books": len(set(source_objects)),
        "all_ranks_full_vocab_exact": True,
        "all_ranks_final_kv_exact": True,
        "all_ranks_final_gdn_exact": True,
        "all_ranks_document_immutable": True,
        "all_ranks_aligned_no_tail_copy": True,
        "all_ranks_cancel_reclaim_passed": True,
        "all_ranks_stale_handle_detected": True,
        "stale_handle_gate": "STALE_SLOT_LEASE",
        "mutant_count": FORMAL_WORLD_SIZE,
        "mutant_escapes": 0,
        "wrong_gate_outcomes": 0,
        "shards": receipts,
        "static_manifest_raw_sha256": args.expected_static_manifest_sha256,
        "model_weight_ledger_raw_sha256": args.expected_model_weight_ledger_sha256,
        "claim_boundaries": static["design_boundary"],
        "contribution_ceiling_note": (
            "This closes aligned-geometry and cancellation/reclamation transfer only; "
            "it is not a second independently implemented model or runtime."
        ),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("static", "shard", "aggregate"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--pg19-data", type=Path)
    parser.add_argument("--pg19-manifest", type=Path)
    parser.add_argument("--expected-pg19-sha256", default=EXPECTED_PG19_DATA_SHA256)
    parser.add_argument("--expected-pg19-manifest-sha256", default=EXPECTED_PG19_MANIFEST_SHA256)
    parser.add_argument("--expected-windows-sha256", default="")
    parser.add_argument("--static-manifest", type=Path)
    parser.add_argument("--expected-static-manifest-sha256", default="")
    parser.add_argument("--expected-code-ledger-sha256", default="")
    parser.add_argument("--expected-model-weight-ledger-sha256", default="")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--expected-gpu-uuid", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shard-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.stage in {"static", "shard"}:
        _require(args.model is not None and args.pg19_data is not None and args.pg19_manifest is not None, "model and PG19 inputs are required")
    if args.stage == "static":
        value = build_static(args)
    elif args.stage == "shard":
        _require(re.fullmatch(r"[0-9a-f]{64}", args.expected_static_manifest_sha256) is not None, "static SHA is required")
        _require(re.fullmatch(r"[0-9a-f]{64}", args.expected_code_ledger_sha256) is not None, "code ledger SHA is required")
        _require(re.fullmatch(r"[0-9a-f]{64}", args.expected_model_weight_ledger_sha256) is not None, "model weight ledger SHA is required")
        value = run_shard(args)
    else:
        _require(args.static_manifest is not None and args.shard_dir is not None, "aggregate inputs are required")
        _require(re.fullmatch(r"[0-9a-f]{64}", args.expected_model_weight_ledger_sha256) is not None, "model weight ledger SHA is required")
        value = aggregate(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
