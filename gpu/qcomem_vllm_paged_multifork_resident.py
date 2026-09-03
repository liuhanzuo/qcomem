from __future__ import annotations

"""Multi-request resident controls for the Q16 vLLM paged-cache path.

This module is deliberately separate from the archived single-request fair-v2
runner.  One immutable document arena reserves private append pages for ``N``
simultaneously live requests.  The control arm instead materializes ``N``
independent request-owned document pools.  Both arms dispatch through the same
``unified_attention`` callable and retain every request object until the whole
round-robin generation has finished.
"""

import copy
import hashlib
import inspect
import json
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

import torch

from qcomem_qwen35_native_cache import install_native_functional_linear_cache
from qcomem_qwen35_vllm_paged_integration import (
    POST_ROPE_POSITION_IDS_CONTRACT,
    register_qwen35_vllm_q16_backend,
    validate_qwen35_post_rope_position_ids,
)
from qcomem_vllm_paged_fair_control import FRESH_CONTROL, SHARED_REUSE
from qcomem_vllm_paged_kernel import (
    KERNEL_MODE,
    Q16KernelPagedDocumentLayer,
    Q16KernelPagedLayer,
    Q16KernelPagedTensorView,
    Q16PagedArena,
    _resolve_vllm_unified_attention,
    vllm_triton_q16_paged_attention_forward,
)


MULTIFORK_COUNTS = (1, 2, 4, 8, 16, 32)
MULTIFORK_PROTOCOL = "same-vllm-unified-attention-q16-multifork-resident-v1"
MULTIFORK_POLICIES = (FRESH_CONTROL, SHARED_REUSE)
PRODUCTION_MASK_CONTRACT = "prevalidated-no-padding-tail-causal"
GDN_BORROW_IMMUTABLE_BASE = "borrow-immutable-base-functional-rebind"
GDN_MATERIALIZE_REQUEST_BASE = "materialize-request-base-functional-rebind"
MULTIFORK_GDN_BASE_POLICIES = (
    GDN_BORROW_IMMUTABLE_BASE,
    GDN_MATERIALIZE_REQUEST_BASE,
)


class QComemMultiForkError(RuntimeError):
    """Raised when a resident multi-request invariant is violated."""


class RuntimeInvariantError(QComemMultiForkError):
    """Raised with one stable gate identifier for a live runtime violation."""

    def __init__(self, gate_id: str, message: str) -> None:
        self.gate_id = gate_id
        super().__init__(f"{gate_id}: {message}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise QComemMultiForkError(message)


def _runtime_require(condition: bool, gate_id: str, message: str) -> None:
    if not condition:
        raise RuntimeInvariantError(gate_id, message)


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _storage_inventory(*roots: Any) -> dict[tuple[str, int, int], int]:
    seen: set[int] = set()
    result: dict[tuple[str, int, int], int] = {}

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, torch.Tensor):
            key = _storage_key(value)
            result[key] = key[2]
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        elif hasattr(value, "__dict__"):
            for item in vars(value).values():
                visit(item)

    for root in roots:
        visit(root)
    return result


def _inventory_nbytes(
    inventory: dict[tuple[str, int, int], int], *, accelerator_only: bool
) -> int:
    return sum(
        size
        for (device, _pointer, _size), size in inventory.items()
        if not accelerator_only or device.startswith("cuda")
    )


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def build_deterministic_distinct_queries(
    base_query_ids: torch.Tensor,
    tokenizer: Any,
    *,
    count: int,
    query_tokens: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Create deterministic, pairwise-distinct token content for resident slots.

    A human-readable slot marker is appended before taking the final fixed-size
    tail.  The exact token digest is recorded and pairwise uniqueness is a hard
    gate; no random sampler or Python hash seed is involved.
    """

    _validate_count(count)
    _require(type(query_tokens) is int and query_tokens >= 8, "query_tokens must be >= 8")
    if base_query_ids.ndim == 2:
        _require(int(base_query_ids.shape[0]) == 1, "base query batch must be one")
        base = base_query_ids[0]
    else:
        _require(base_query_ids.ndim == 1, "base query must be rank one or [1,tokens]")
        base = base_query_ids
    _require(int(base.numel()) >= 1, "base query is empty")
    variants: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    for request_index in range(count):
        marker_text = f"\nResident request slot {request_index:02d}; continue independently.\n"
        marker_ids = tokenizer.encode(marker_text, add_special_tokens=False)
        _require(isinstance(marker_ids, (list, tuple)) and marker_ids, "tokenizer returned empty marker")
        marker = torch.tensor(marker_ids, dtype=base.dtype, device=base.device)
        combined = torch.cat((base, marker))
        if int(combined.numel()) >= query_tokens:
            variant = combined[-query_tokens:]
        else:
            repeats = math.ceil((query_tokens - int(marker.numel())) / int(base.numel()))
            prefix = base.repeat(repeats + 1)
            variant = torch.cat((prefix, marker))[-query_tokens:]
        variant = variant.contiguous().unsqueeze(0)
        digest = _tensor_digest(variant)
        variants.append(variant)
        rows.append(
            {
                "request_index": request_index,
                "query_token_ids_sha256": digest,
                "query_tokens": query_tokens,
                "marker_text": marker_text,
            }
        )
    digests = [row["query_token_ids_sha256"] for row in rows]
    _require(len(set(digests)) == count, "resident query token content is not pairwise distinct")
    return variants, {
        "deterministic": True,
        "pairwise_distinct": True,
        "count": count,
        "query_tokens": query_tokens,
        "rows": rows,
    }


def build_pg19_train_query_bank(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    window: Any,
    *,
    document_tokens: int,
    query_tokens: int,
    count: int,
    query_stride: int,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Freeze non-overlapping raw-token queries from the document's train book.

    Query chunks begin after the selected document and the calibration
    builder's adjacent query.  They are therefore outside the 4095-token
    document while remaining in the exact same PG-19 train object.  No
    synthetic marker or evaluation text is introduced.
    """

    _validate_count(count)
    _require(type(document_tokens) is int and document_tokens > 0, "document_tokens must be positive")
    _require(type(query_tokens) is int and query_tokens > 0, "query_tokens must be positive")
    _require(type(query_stride) is int and query_stride >= query_tokens, "query stride must prevent overlap")
    source_object = str(getattr(window, "source_object", ""))
    start_token = getattr(window, "start_token", None)
    _require(source_object and type(start_token) is int and start_token >= 0, "window provenance is incomplete")
    matches = [record for record in records if str(record.get("_source_object")) == source_object]
    _require(len(matches) == 1, "query bank source object is not unique in PG19 records")
    record = matches[0]
    text = record.get("text")
    _require(isinstance(text, str) and text, "query bank source has no raw text")
    document_start = start_token
    document_end = document_start + document_tokens
    bank_start = document_end + query_tokens
    offsets = [bank_start + index * query_stride for index in range(count)]
    required_tokens = offsets[-1] + query_tokens
    guarded_tokens = required_tokens + 64
    character_limit = min(len(text), max(guarded_tokens * 6, 8192))
    while True:
        ids = tokenizer.encode(text[:character_limit], add_special_tokens=False)
        if len(ids) >= guarded_tokens or character_limit == len(text):
            break
        character_limit = min(len(text), character_limit * 2)
    _require(len(ids) >= required_tokens, "PG19 train book is too short for the frozen query bank")
    queries = [
        torch.tensor(ids[offset : offset + query_tokens], dtype=torch.long).unsqueeze(0)
        for offset in offsets
    ]
    rows = [
        {
            "request_index": request_index,
            "source_token_offset": offsets[request_index],
            "query_tokens": query_tokens,
            "query_token_ids_sha256": _tensor_digest(query),
        }
        for request_index, query in enumerate(queries)
    ]
    digests = [row["query_token_ids_sha256"] for row in rows]
    _require(len(set(digests)) == count, "PG19 raw query chunks are not pairwise distinct")
    for left, offset in enumerate(offsets):
        _require(offset >= document_end, "query bank overlaps the document window")
        for right in range(left + 1, len(offsets)):
            _require(
                offset + query_tokens <= offsets[right]
                or offsets[right] + query_tokens <= offset,
                "query bank chunks overlap",
            )
    bank_digest = hashlib.sha256()
    bank_digest.update(source_object.encode())
    bank_digest.update(f"{document_start}:{document_end}:{query_tokens}:{query_stride}\n".encode())
    for row, query in zip(rows, queries):
        bank_digest.update(f"{row['request_index']}:{row['source_token_offset']}\0".encode())
        bank_digest.update(query.numpy().tobytes())
    return queries, {
        "source_role": "same-pg19-train-book-raw-nonoverlapping-query-chunks",
        "synthetic_markers_used": False,
        "source_object": source_object,
        "source_id": str(getattr(window, "source_id", record.get("id", ""))),
        "document_start_token": document_start,
        "document_end_token_exclusive": document_end,
        "query_bank_start_token": bank_start,
        "query_stride_tokens": query_stride,
        "query_tokens": query_tokens,
        "count": count,
        "pairwise_nonoverlapping": True,
        "pairwise_distinct": True,
        "rows": rows,
        "query_bank_sha256": bank_digest.hexdigest(),
    }


def _validate_count(count: int) -> None:
    _require(type(count) is int and count in MULTIFORK_COUNTS, "unsupported resident request count")


def _seed_tensor_memo(value: Any, memo: dict[int, Any], seen: set[int]) -> None:
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        memo[identity] = value
    elif isinstance(value, dict):
        for item in value.values():
            _seed_tensor_memo(item, memo, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _seed_tensor_memo(item, memo, seen)
    elif hasattr(value, "__dict__"):
        for item in vars(value).values():
            _seed_tensor_memo(item, memo, seen)


@dataclass(frozen=True)
class KVRequestBindingRow:
    request_index: int
    layer_index: int
    sequence_id: int
    arena_id: int
    reservation_ids: tuple[int, ...]
    block_table_shape: tuple[int, ...]
    construction_document_prefix_ids: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class KVBindingGuard:
    """Live-only construction binding; never serialized as pointer evidence."""

    policy: str
    resident_count: int
    layer_indices: tuple[int, ...]
    request_object_ids: tuple[int, ...]
    rows: tuple[KVRequestBindingRow, ...]


@dataclass(frozen=True)
class ResidentRequestGroup:
    policy: str
    resident_count: int
    requests: tuple[Any, ...]
    audit: dict[str, Any]
    kv_binding_guard: KVBindingGuard


def _capture_kv_binding_guard(
    requests: Sequence[Any],
    plan: Any,
    *,
    resident_count: int,
    policy: str,
) -> KVBindingGuard:
    indices = tuple(int(index) for index in plan.full_attention_layer_indices)
    rows = []
    for request_index, request in enumerate(requests):
        for layer_index in indices:
            sequence = request.layers[layer_index].sequence
            rows.append(
                KVRequestBindingRow(
                    request_index=request_index,
                    layer_index=layer_index,
                    sequence_id=id(sequence),
                    arena_id=id(sequence.arena),
                    reservation_ids=tuple(
                        int(value)
                        for value in sequence.reservations.reshape(-1).tolist()
                    ),
                    block_table_shape=tuple(int(value) for value in sequence.block_table.shape),
                    construction_document_prefix_ids=tuple(
                        tuple(int(value) for value in batch_row)
                        for batch_row in sequence.active_block_table[
                            :, : sequence.arena.document_blocks_per_sequence
                        ].tolist()
                    ),
                )
            )
    return KVBindingGuard(
        policy=policy,
        resident_count=resident_count,
        layer_indices=indices,
        request_object_ids=tuple(id(request) for request in requests),
        rows=tuple(rows),
    )


def _materialize_fresh_layer(
    document: Q16KernelPagedDocumentLayer,
) -> tuple[Q16KernelPagedLayer, dict[str, int | bool]]:
    source = document.arena
    document_physical = source.batch_size * source.document_blocks_per_sequence
    private_physical = source.batch_size * source.private_blocks_per_sequence
    physical_blocks = document_physical + private_physical
    shape = (
        physical_blocks,
        source.page_size,
        source.num_key_value_heads,
        source.head_dim,
    )
    key_cache = torch.empty(shape, dtype=source.key_cache.dtype, device=source.key_cache.device)
    value_cache = torch.empty_like(key_cache)
    key_cache[:document_physical].copy_(source.key_cache[:document_physical])
    value_cache[:document_physical].copy_(source.value_cache[:document_physical])
    document_table = source.document_block_table.clone()
    reservations = torch.arange(
        document_physical,
        physical_blocks,
        dtype=torch.int64,
        device="cpu",
    ).reshape(1, source.batch_size, source.private_blocks_per_sequence)
    arena = Q16PagedArena(
        key_cache,
        value_cache,
        document_table,
        document_length=source.document_length,
        max_append_tokens=source.max_append_tokens,
        max_forks=1,
        private_block_reservations=reservations,
    )
    _require(not (arena.storage_keys & source.storage_keys), "fresh pool shares source storage")
    document_copy = (
        2
        * document_physical
        * source.page_size
        * source.num_key_value_heads
        * source.head_dim
        * source.key_cache.element_size()
    )
    return Q16KernelPagedLayer(
        arena.fork(strict_mask_check=document.strict_mask_check)
    ), {
        "document_block_copy_nbytes_including_padding": document_copy,
        "document_payload_nbytes": source.audit.document_payload_nbytes,
        "copied_padding_nbytes": document_copy - source.audit.document_payload_nbytes,
        "allocated_request_pool_nbytes": arena.allocated_pool_nbytes,
        "source_storage_shared": False,
    }


def _fresh_request(cache: Any, plan: Any) -> tuple[Any, dict[str, Any]]:
    memo: dict[int, Any] = {}
    for index in plan.full_attention_layer_indices:
        layer = cache.layers[index]
        _require(isinstance(layer, Q16KernelPagedDocumentLayer), f"source layer {index} is not paged")
        memo[id(layer)] = layer
    _seed_tensor_memo(cache, memo, set())
    request = copy.deepcopy(cache, memo)
    rows = []
    copied = allocated = payload = 0
    for index in plan.full_attention_layer_indices:
        target, row = _materialize_fresh_layer(cache.layers[index])
        request.layers[index] = target
        copied += int(row["document_block_copy_nbytes_including_padding"])
        allocated += int(row["allocated_request_pool_nbytes"])
        payload += int(row["document_payload_nbytes"])
        rows.append({"layer_idx": int(index), **row})
    gdn = _prepare_request_gdn_base(
        cache,
        request,
        plan,
        policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    install = install_native_functional_linear_cache(request, plan.gdn)
    _require(
        tuple(install.linear_layer_indices) == tuple(plan.linear_layer_indices),
        "fresh request missed functional GDN layers",
    )
    return request, {
        "document_block_copy_nbytes_including_padding": copied,
        "allocated_request_pool_nbytes": allocated,
        "document_payload_nbytes": payload,
        "source_document_storage_shared": False,
        "gdn_base": gdn,
        "layers": rows,
    }


def _reuse_request(cache: Any, plan: Any) -> tuple[Any, dict[str, Any]]:
    memo: dict[int, Any] = {}
    for index in plan.full_attention_layer_indices:
        layer = cache.layers[index]
        _require(isinstance(layer, Q16KernelPagedDocumentLayer), f"source layer {index} is not paged")
        memo[id(layer)] = layer
    _seed_tensor_memo(cache, memo, set())
    request = copy.deepcopy(cache, memo)
    for index in plan.full_attention_layer_indices:
        source = cache.layers[index]
        target = source.fork()
        _require(target.sequence.arena is source.arena, "reuse request did not share source arena")
        request.layers[index] = target
    gdn = _prepare_request_gdn_base(
        cache,
        request,
        plan,
        policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    install = install_native_functional_linear_cache(request, plan.gdn)
    _require(
        tuple(install.linear_layer_indices) == tuple(plan.linear_layer_indices),
        "reuse request missed functional GDN layers",
    )
    return request, {
        "document_block_copy_nbytes_including_padding": 0,
        "allocated_request_pool_nbytes": 0,
        "source_document_storage_shared": True,
        "gdn_base": gdn,
    }


def _prepare_request_gdn_base(
    persistent: Any,
    request: Any,
    plan: Any,
    *,
    policy: str,
) -> dict[str, Any]:
    _require(policy in MULTIFORK_GDN_BASE_POLICIES, "unknown GDN base policy")
    tensor_count = 0
    materialized_nbytes = 0
    exact_base_aliases = 0
    for index in plan.linear_layer_indices:
        source_layer = persistent.layers[index]
        request_layer = request.layers[index]
        for family in ("conv_states", "recurrent_states"):
            source_values = getattr(source_layer, family)
            request_values = getattr(request_layer, family)
            _require(
                isinstance(source_values, dict)
                and isinstance(request_values, dict)
                and sorted(source_values) == sorted(request_values),
                "GDN state family schema drift",
            )
            for state_index in sorted(source_values):
                source_tensor = source_values[state_index]
                request_tensor = request_values[state_index]
                _require(
                    isinstance(source_tensor, torch.Tensor)
                    and isinstance(request_tensor, torch.Tensor),
                    "GDN state is not tensor",
                )
                if policy == GDN_BORROW_IMMUTABLE_BASE:
                    _require(
                        _storage_key(request_tensor) == _storage_key(source_tensor),
                        "borrowed GDN base did not preserve exact storage alias",
                    )
                    exact_base_aliases += 1
                else:
                    cloned = request_tensor.clone()
                    request_values[state_index] = cloned
                    _require(
                        _storage_key(cloned) != _storage_key(source_tensor),
                        "materialized GDN base still aliases persistent storage",
                    )
                    materialized_nbytes += cloned.untyped_storage().nbytes()
                tensor_count += 1
    _require(tensor_count == 60, "formal Qwen3.5 request must expose 60 GDN tensors")
    return {
        "policy": policy,
        "tensor_count": tensor_count,
        "borrowed_immutable_base_alias_count": exact_base_aliases,
        "materialized_request_base_nbytes": materialized_nbytes,
        "functional_rebind_after_transition": True,
    }


def _request_with_gdn_policy(
    cache: Any,
    plan: Any,
    *,
    request_policy: str,
    gdn_base_policy: str,
) -> tuple[Any, dict[str, Any]]:
    request, audit = (
        _fresh_request(cache, plan)
        if request_policy == FRESH_CONTROL
        else _reuse_request(cache, plan)
    )
    if gdn_base_policy == GDN_BORROW_IMMUTABLE_BASE:
        return request, audit
    _require(
        gdn_base_policy == GDN_MATERIALIZE_REQUEST_BASE,
        "unknown GDN base policy",
    )
    # The default constructor installed the functional seam over an immutable
    # borrowed base. Clone those 60 tensors now; subsequent updates still
    # rebind functionally, so the factorial changes ownership only.
    gdn = _prepare_request_gdn_base(
        cache,
        request,
        plan,
        policy=gdn_base_policy,
    )
    audit = {**audit, "gdn_base": gdn}
    return request, audit


def build_resident_request_group(
    cache: Any,
    plan: Any,
    *,
    resident_count: int,
    policy: Literal[
        "vllm-q16-fresh-full-copy-control",
        "vllm-q16-shared-document-reuse",
    ],
    gdn_base_policy: str = GDN_BORROW_IMMUTABLE_BASE,
) -> ResidentRequestGroup:
    """Create all ``N`` requests before returning; none is released in-loop."""

    _validate_count(resident_count)
    _require(policy in MULTIFORK_POLICIES, "unknown multi-fork policy")
    _require(
        gdn_base_policy in MULTIFORK_GDN_BASE_POLICIES,
        "unknown GDN base policy",
    )
    for index in plan.full_attention_layer_indices:
        source = cache.layers[index]
        _require(isinstance(source, Q16KernelPagedDocumentLayer), "source is not Q16 paged")
        _require(source.arena.max_forks == resident_count, "source max_forks differs from N")
        _require(source.arena._fork_cursor == 0, "source arena was already forked")
    requests: list[Any] = []
    rows = []
    for request_index in range(resident_count):
        request, row = _request_with_gdn_policy(
            cache,
            plan,
            request_policy=policy,
            gdn_base_policy=gdn_base_policy,
        )
        requests.append(request)
        rows.append({"request_index": request_index, **row})
    _require(len({id(request) for request in requests}) == resident_count, "request objects alias")
    ownership = validate_resident_group_ownership(
        cache, requests, plan, resident_count=resident_count, policy=policy
    )
    kv_binding_guard = _capture_kv_binding_guard(
        requests,
        plan,
        resident_count=resident_count,
        policy=policy,
    )
    return ResidentRequestGroup(
        policy=policy,
        resident_count=resident_count,
        requests=tuple(requests),
        audit={
            "protocol": MULTIFORK_PROTOCOL,
            "policy": policy,
            "gdn_base_policy": gdn_base_policy,
            "resident_count": resident_count,
            "all_requests_materialized_before_measurement": True,
            "strong_reference_count": len(requests),
            "rows": rows,
            "ownership": ownership,
            "physical_document_block_copy_nbytes_including_padding": sum(
                int(row["document_block_copy_nbytes_including_padding"])
                for row in rows
            ),
            "allocated_fresh_request_pool_nbytes": sum(
                int(row["allocated_request_pool_nbytes"]) for row in rows
            ),
        },
        kv_binding_guard=kv_binding_guard,
    )


def validate_resident_group_ownership(
    cache: Any,
    requests: Sequence[Any],
    plan: Any,
    *,
    resident_count: int,
    policy: str,
) -> dict[str, Any]:
    _require(len(requests) == resident_count, "resident group lost a strong request reference")
    sequence_ids: set[int] = set()
    arena_storage_sets: list[frozenset[tuple[str, int, int]]] = []
    reservation_id_sets_by_layer: dict[int, list[set[int]]] = {
        int(index): [] for index in plan.full_attention_layer_indices
    }
    for request in requests:
        for index in plan.full_attention_layer_indices:
            source = cache.layers[index].arena
            layer = request.layers[index]
            _require(isinstance(layer, Q16KernelPagedLayer), "resident layer is not writable paged")
            sequence = layer.sequence
            sequence_ids.add(id(sequence))
            if policy == SHARED_REUSE:
                _require(sequence.arena is source, "reuse request uses a different source arena")
            else:
                _require(sequence.arena is not source, "fresh request shares the source arena")
            reservation_id_sets_by_layer[int(index)].append(
                {int(value) for value in sequence.reservations.reshape(-1).tolist()}
            )
        arena_storage_sets.append(
            frozenset().union(
                *(request.layers[index].sequence.arena.storage_keys for index in plan.full_attention_layer_indices)
            )
        )
    expected_sequences = resident_count * len(tuple(plan.full_attention_layer_indices))
    _require(len(sequence_ids) == expected_sequences, "request sequences are not pairwise distinct")
    if policy == SHARED_REUSE:
        # Physical ids are comparable only inside the one shared arena.  Each
        # fresh arena has its own local id namespace and is instead proven
        # disjoint by its K/V storage identity below.
        for index, id_sets in reservation_id_sets_by_layer.items():
            for left in range(len(id_sets)):
                for right in range(left + 1, len(id_sets)):
                    _require(not (id_sets[left] & id_sets[right]), f"layer {index} private reservations overlap")
    if policy == FRESH_CONTROL:
        for left in range(len(arena_storage_sets)):
            for right in range(left + 1, len(arena_storage_sets)):
                _require(
                    not (arena_storage_sets[left] & arena_storage_sets[right]),
                    "fresh request pools share storage",
                )
    else:
        _require(len(set(arena_storage_sets)) == 1, "reuse requests do not share one source pool")
    return {
        "passed": True,
        "resident_count": resident_count,
        "request_object_ids_pairwise_distinct": True,
        "request_sequence_ids_pairwise_distinct": True,
        "private_physical_reservation_ids_pairwise_disjoint": policy == SHARED_REUSE,
        "fresh_private_id_namespace_is_per_arena": policy == FRESH_CONTROL,
        "reuse_requests_share_source_arena": policy == SHARED_REUSE,
        "fresh_request_arena_storages_pairwise_disjoint": policy == FRESH_CONTROL,
        "all_requests_strongly_referenced": True,
    }


def source_document_physical_digests(
    cache: Any,
    layer_indices: Iterable[int],
) -> dict[str, str]:
    """Hash complete source document blocks, including partial-tail padding."""

    rows: dict[str, str] = {}
    for index in layer_indices:
        layer = cache.layers[index]
        arena = getattr(layer, "arena", None)
        _require(isinstance(arena, Q16PagedArena), "source layer has no Q16 arena")
        scalar_fields = {
            "document_length": arena.document_length,
            "max_append_tokens": arena.max_append_tokens,
            "max_forks": arena.max_forks,
            "page_size": arena.page_size,
            "num_key_value_heads": arena.num_key_value_heads,
            "head_dim": arena.head_dim,
            "batch_size": arena.batch_size,
            "document_blocks_per_sequence": arena.document_blocks_per_sequence,
            "private_blocks_per_sequence": arena.private_blocks_per_sequence,
        }
        _require(
            all(type(value) is int and value > 0 for value in scalar_fields.values()),
            "source physical arena scalar schema drift",
        )
        expected_document_blocks = math.ceil(
            arena.document_length / arena.page_size
        )
        tail_tokens = arena.document_length % arena.page_size
        expected_private_blocks = math.ceil(
            ((tail_tokens if tail_tokens else 0) + arena.max_append_tokens)
            / arena.page_size
        )
        _require(
            arena.document_blocks_per_sequence == expected_document_blocks
            and arena.private_blocks_per_sequence == expected_private_blocks,
            "source physical arena block geometry drift",
        )
        _require(
            arena.key_cache.ndim == 4
            and arena.value_cache.shape == arena.key_cache.shape
            and arena.key_cache.dtype == arena.value_cache.dtype
            and arena.key_cache.device == arena.value_cache.device
            and arena.key_cache.is_floating_point(),
            "source physical K/V pool schema drift",
        )
        _require(
            arena.key_cache.is_contiguous()
            and arena.value_cache.is_contiguous()
            and arena.key_cache.storage_offset() == 0
            and arena.value_cache.storage_offset() == 0
            and arena.key_cache.untyped_storage().nbytes()
            == arena.key_cache.numel() * arena.key_cache.element_size()
            and arena.value_cache.untyped_storage().nbytes()
            == arena.value_cache.numel() * arena.value_cache.element_size(),
            "source physical cache layout must be contiguous full storage",
        )
        _require(
            tuple(arena.key_cache.shape[1:])
            == (arena.page_size, arena.num_key_value_heads, arena.head_dim),
            "source physical arena geometry differs from K/V pool shape",
        )
        _require(
            arena.document_block_table.dtype == torch.int32
            and arena.document_block_table.device == arena.key_cache.device
            and arena.document_block_table.is_contiguous()
            and arena.document_block_table.storage_offset() == 0
            and arena.document_block_table.untyped_storage().nbytes()
            == arena.document_block_table.numel()
            * arena.document_block_table.element_size()
            and tuple(arena.document_block_table.shape)
            == (arena.batch_size, arena.document_blocks_per_sequence),
            "source document block-table physical schema drift",
        )
        _require(
            arena.private_block_reservations.dtype == torch.int64
            and arena.private_block_reservations.device.type == "cpu"
            and arena.private_block_reservations.is_contiguous()
            and arena.private_block_reservations.storage_offset() == 0
            and arena.private_block_reservations.untyped_storage().nbytes()
            == arena.private_block_reservations.numel()
            * arena.private_block_reservations.element_size()
            and tuple(arena.private_block_reservations.shape)
            == (
                arena.max_forks,
                arena.batch_size,
                arena.private_blocks_per_sequence,
            ),
            "source private reservation physical schema drift",
        )
        expected_physical_blocks = (
            arena.batch_size * arena.document_blocks_per_sequence
            + arena.max_forks
            * arena.batch_size
            * arena.private_blocks_per_sequence
        )
        _require(
            int(arena.key_cache.shape[0]) == expected_physical_blocks,
            "source physical pool omits document or private reservations",
        )
        document_blocks = arena.batch_size * arena.document_blocks_per_sequence
        expected_document_table = torch.arange(
            document_blocks,
            dtype=torch.int32,
            device=arena.document_block_table.device,
        ).reshape(arena.batch_size, arena.document_blocks_per_sequence)
        expected_private_reservations = torch.arange(
            document_blocks,
            expected_physical_blocks,
            dtype=torch.int64,
            device="cpu",
        ).reshape(
            arena.max_forks,
            arena.batch_size,
            arena.private_blocks_per_sequence,
        )
        _require(
            bool(torch.equal(arena.document_block_table, expected_document_table)),
            "source document block-table IDs drift",
        )
        _require(
            bool(
                torch.equal(
                    arena.private_block_reservations,
                    expected_private_reservations,
                )
            ),
            "source private reservation IDs drift",
        )
        digest = hashlib.sha256()
        metadata = {
            "schema": "qcomem-source-document-physical-v3",
            "layer_index": int(index),
            "document_length": arena.document_length,
            "max_append_tokens": arena.max_append_tokens,
            "page_size": arena.page_size,
            "batch_size": arena.batch_size,
            "document_blocks_per_sequence": arena.document_blocks_per_sequence,
            "key_dtype": str(arena.key_cache.dtype),
            "value_dtype": str(arena.value_cache.dtype),
            "key_cache_shape": list(arena.key_cache.shape),
            "value_cache_shape": list(arena.value_cache.shape),
            "key_document_shape": list(arena.key_cache[:document_blocks].shape),
            "value_document_shape": list(arena.value_cache[:document_blocks].shape),
            "key_stride": list(arena.key_cache.stride()),
            "value_stride": list(arena.value_cache.stride()),
            "key_storage_offset": int(arena.key_cache.storage_offset()),
            "value_storage_offset": int(arena.value_cache.storage_offset()),
            "key_storage_nbytes": int(arena.key_cache.untyped_storage().nbytes()),
            "value_storage_nbytes": int(arena.value_cache.untyped_storage().nbytes()),
            "key_device": str(arena.key_cache.device),
            "value_device": str(arena.value_cache.device),
            "num_key_value_heads": arena.num_key_value_heads,
            "head_dim": arena.head_dim,
            "max_forks": arena.max_forks,
            "private_blocks_per_sequence": arena.private_blocks_per_sequence,
            "document_block_table_dtype": str(arena.document_block_table.dtype),
            "document_block_table_shape": list(arena.document_block_table.shape),
            "document_block_table_stride": list(arena.document_block_table.stride()),
            "document_block_table_storage_offset": int(
                arena.document_block_table.storage_offset()
            ),
            "document_block_table_storage_nbytes": int(
                arena.document_block_table.untyped_storage().nbytes()
            ),
            "document_block_table_device": str(arena.document_block_table.device),
            "document_block_table": arena.document_block_table.detach()
            .cpu()
            .tolist(),
            "private_block_reservations_dtype": str(
                arena.private_block_reservations.dtype
            ),
            "private_block_reservations_shape": list(
                arena.private_block_reservations.shape
            ),
            "private_block_reservations_stride": list(
                arena.private_block_reservations.stride()
            ),
            "private_block_reservations_storage_offset": int(
                arena.private_block_reservations.storage_offset()
            ),
            "private_block_reservations_storage_nbytes": int(
                arena.private_block_reservations.untyped_storage().nbytes()
            ),
            "private_block_reservations_device": str(
                arena.private_block_reservations.device
            ),
            "private_block_reservations": arena.private_block_reservations.tolist(),
        }
        digest.update(
            json.dumps(
                metadata,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(
            arena.key_cache[:document_blocks]
            .detach()
            .contiguous()
            .cpu()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        digest.update(
            arena.value_cache[:document_blocks]
            .detach()
            .contiguous()
            .cpu()
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        rows[str(index)] = digest.hexdigest()
    return rows


def validate_runtime_kv_ownership(
    persistent: Any,
    group: ResidentRequestGroup,
    plan: Any,
    *,
    require_appended_tail_cow: bool,
) -> dict[str, Any]:
    """Validate live sequence binding, reservations, and active private pages."""

    _runtime_require(
        len(group.requests) == group.resident_count,
        "KV_SEQUENCE_ID",
        "resident request cardinality changed",
    )
    guard = group.kv_binding_guard
    _runtime_require(
        guard.policy == group.policy
        and guard.resident_count == group.resident_count
        and guard.layer_indices
        == tuple(int(index) for index in plan.full_attention_layer_indices)
        and guard.request_object_ids
        == tuple(id(request) for request in group.requests),
        "KV_SEQUENCE_ID",
        "resident request construction binding drift",
    )
    frozen_rows = {
        (row.request_index, row.layer_index): row for row in guard.rows
    }
    per_layer_active: dict[int, list[tuple[int, set[int], set[int], int]]] = {
        int(index): [] for index in plan.full_attention_layer_indices
    }
    for request_index, request in enumerate(group.requests):
        for index in plan.full_attention_layer_indices:
            source = persistent.layers[index].arena
            sequence = request.layers[index].sequence
            frozen = frozen_rows.get((request_index, int(index)))
            _runtime_require(
                frozen is not None
                and id(sequence) == frozen.sequence_id
                and id(sequence.arena) == frozen.arena_id,
                "KV_SEQUENCE_ID",
                "request sequence or arena differs from construction binding",
            )
            current_reservation_ids = tuple(
                int(value)
                for value in sequence.reservations.reshape(-1).tolist()
            )
            _runtime_require(
                current_reservation_ids == frozen.reservation_ids,
                "KV_RESERVATION_DISJOINT",
                "request reservation differs from construction binding",
            )
            _runtime_require(
                tuple(int(value) for value in sequence.block_table.shape)
                == frozen.block_table_shape,
                "KV_SEQUENCE_ID",
                "request block-table geometry differs from construction binding",
            )
            document_blocks_per_sequence = (
                sequence.arena.document_blocks_per_sequence
            )
            tail_tokens = sequence.arena.document_length % sequence.arena.page_size
            immutable_document_blocks = document_blocks_per_sequence
            if sequence.appended_tokens > 0 and tail_tokens:
                immutable_document_blocks -= 1
            current_document_prefix = tuple(
                tuple(int(value) for value in batch_row)
                for batch_row in sequence.active_block_table[
                    :, :immutable_document_blocks
                ].tolist()
            )
            frozen_document_prefix = tuple(
                row[:immutable_document_blocks]
                for row in frozen.construction_document_prefix_ids
            )
            _runtime_require(
                current_document_prefix == frozen_document_prefix,
                "KV_SEQUENCE_ID",
                "immutable document block-table prefix drifted after construction",
            )
            reservations = {
                int(value) for value in current_reservation_ids
            }
            _runtime_require(
                len(reservations)
                == sequence.arena.batch_size
                * sequence.arena.private_blocks_per_sequence,
                "KV_RESERVATION_DISJOINT",
                "request reservation contains duplicate physical blocks",
            )
            active_private: set[int] = set()
            for batch_index in range(sequence.arena.batch_size):
                if sequence.appended_tokens > 0:
                    tail = source.document_length % source.page_size
                    first_private_logical = source.document_blocks_per_sequence
                    if tail:
                        first_private_logical -= 1
                        _runtime_require(
                            bool(sequence._tail_detached[batch_index]),
                            "KV_TAIL_COW",
                            "partial document tail was not detached",
                        )
                    for logical in range(
                        first_private_logical,
                        sequence.logical_block_count,
                    ):
                        physical = int(sequence.active_block_table[batch_index, logical])
                        _runtime_require(
                            physical in reservations,
                            "KV_ACTIVE_BLOCK_OWNERSHIP",
                            "active private block is outside this request reservation",
                        )
                        active_private.add(physical)
            if require_appended_tail_cow:
                _runtime_require(
                    sequence.appended_tokens > 0,
                    "KV_TAIL_COW",
                    "tail-COW validation requested before append",
                )
            per_layer_active[int(index)].append(
                (request_index, reservations, active_private, id(sequence.arena))
            )
    for index, rows in per_layer_active.items():
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                left_request, left_reservations, left_active, left_arena = rows[left]
                right_request, right_reservations, right_active, right_arena = rows[right]
                if left_arena == right_arena:
                    _runtime_require(
                        not (left_reservations & right_reservations),
                        "KV_RESERVATION_DISJOINT",
                        f"layer {index} requests {left_request}/{right_request} share reservations",
                    )
                    _runtime_require(
                        not (left_active & right_active),
                        "KV_ACTIVE_BLOCK_OWNERSHIP",
                        f"layer {index} requests {left_request}/{right_request} share active private blocks",
                    )
    return {
        "passed": True,
        "gate_ids": [
            "KV_SEQUENCE_ID",
            "KV_RESERVATION_DISJOINT",
            "KV_TAIL_COW",
            "KV_ACTIVE_BLOCK_OWNERSHIP",
        ],
        "resident_count": group.resident_count,
        "require_appended_tail_cow": require_appended_tail_cow,
        "construction_binding_verified": True,
    }


def resident_storage_breakdown(
    persistent: Any,
    group: ResidentRequestGroup,
    plan: Any,
) -> dict[str, Any]:
    """Split common source capacity, N fresh pools and active private pages."""

    n = group.resident_count
    policy = group.policy
    _validate_count(n)
    _require(len(group.requests) == n, "storage audit requires all resident requests alive")
    rows: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    for index in plan.full_attention_layer_indices:
        source = persistent.layers[index].arena
        _require(source.max_forks == n, "source private reservation multiplier differs from N")
        block_bytes = (
            2
            * source.page_size
            * source.num_key_value_heads
            * source.head_dim
            * source.key_cache.element_size()
        )
        document_allocated = source.batch_size * source.document_blocks_per_sequence * block_bytes
        source_private = n * source.batch_size * source.private_blocks_per_sequence * block_bytes
        _require(
            source.allocated_pool_nbytes == document_allocated + source_private,
            "source arena storage formula drift",
        )
        fresh_document = n * document_allocated if policy == FRESH_CONTROL else 0
        fresh_padding = (
            n * (document_allocated - source.audit.document_payload_nbytes)
            if policy == FRESH_CONTROL
            else 0
        )
        one_private = source.batch_size * source.private_blocks_per_sequence * block_bytes
        fresh_private = n * one_private if policy == FRESH_CONTROL else 0
        active_payload = active_pages = active_blocks = 0
        request_table = 0
        fresh_document_table = 0
        detached_tail_total = appended_total = 0
        partial_tail_copy = 0
        for request in group.requests:
            sequence = request.layers[index].sequence
            appended = int(sequence.appended_tokens)
            tail = source.document_length % source.page_size if appended > 0 else 0
            private_tokens = tail + appended
            blocks = math.ceil(private_tokens / source.page_size) if private_tokens else 0
            active_payload += (
                2
                * source.batch_size
                * private_tokens
                * source.num_key_value_heads
                * source.head_dim
                * source.key_cache.element_size()
            )
            active_pages += blocks * source.batch_size * block_bytes
            active_blocks += blocks * source.batch_size
            detached_tail_total += tail
            appended_total += appended
            partial_tail_copy += int(sequence.partial_tail_staging_copy_nbytes)
            request_table += sequence.block_table.untyped_storage().nbytes()
            if policy == FRESH_CONTROL:
                fresh_document_table += (
                    sequence.arena.document_block_table.untyped_storage().nbytes()
                )
        reservation_for_requests = n * one_private
        _require(active_payload <= active_pages <= reservation_for_requests, "active pages exceed reservation")
        row = {
            "layer_idx": int(index),
            "resident_count": n,
            "block_bytes": block_bytes,
            "valid_document_payload_nbytes": source.audit.document_payload_nbytes,
            "source_document_allocated_nbytes": document_allocated,
            "source_document_padding_nbytes": document_allocated - source.audit.document_payload_nbytes,
            "source_private_reservation_nbytes": source_private,
            "source_total_arena_allocated_nbytes": document_allocated + source_private,
            "fresh_duplicate_document_allocated_nbytes": fresh_document,
            "fresh_duplicate_document_padding_nbytes": fresh_padding,
            "fresh_duplicate_private_reservation_nbytes": fresh_private,
            "active_request_private_payload_nbytes": active_payload,
            "active_request_private_allocated_page_nbytes": active_pages,
            "active_request_private_blocks": active_blocks,
            "request_private_reserved_unused_nbytes": reservation_for_requests - active_pages,
            "active_request_appended_tokens_sum": appended_total,
            "active_request_detached_tail_tokens_sum": detached_tail_total,
            "partial_tail_staging_copy_nbytes": partial_tail_copy,
            "request_block_table_accelerator_nbytes": request_table,
            "source_document_table_accelerator_nbytes": source.document_block_table.untyped_storage().nbytes(),
            "fresh_document_table_accelerator_nbytes": fresh_document_table,
            "source_cpu_reservation_metadata_nbytes": source.private_block_reservations.untyped_storage().nbytes(),
            "fresh_cpu_reservation_metadata_nbytes": (
                sum(
                    request.layers[index].sequence.arena.private_block_reservations.untyped_storage().nbytes()
                    for request in group.requests
                )
                if policy == FRESH_CONTROL
                else 0
            ),
            "physical_document_block_copy_nbytes_including_padding": fresh_document,
        }
        for key, value in row.items():
            if key.endswith("_nbytes") or key in (
                "active_request_private_blocks",
                "active_request_appended_tokens_sum",
                "active_request_detached_tail_tokens_sum",
                "physical_document_block_copy_nbytes_including_padding",
            ):
                totals[key] += int(value)
        rows.append(row)
    persistent_inventory = _storage_inventory(persistent)
    request_inventory = _storage_inventory(*group.requests)
    combined = dict(persistent_inventory)
    combined.update(request_inventory)
    return {
        "protocol": MULTIFORK_PROTOCOL,
        "policy": policy,
        "resident_count": n,
        "simultaneous_lifetime": True,
        "full_attention_layer_count": len(rows),
        "source_private_reservation_is_common_pack_capacity": True,
        "active_private_payload_is_subset_not_additive": True,
        "fresh_duplicate_pool_is_separate_from_source": policy == FRESH_CONTROL,
        "totals": dict(totals),
        "layers": rows,
        "unique_storage": {
            "persistent_total_nbytes": _inventory_nbytes(persistent_inventory, accelerator_only=False),
            "persistent_accelerator_nbytes": _inventory_nbytes(persistent_inventory, accelerator_only=True),
            "requests_total_nbytes": _inventory_nbytes(request_inventory, accelerator_only=False),
            "requests_accelerator_nbytes": _inventory_nbytes(request_inventory, accelerator_only=True),
            "combined_unique_total_nbytes": _inventory_nbytes(combined, accelerator_only=False),
            "combined_unique_accelerator_nbytes": _inventory_nbytes(combined, accelerator_only=True),
        },
    }


def strict_group_logical_parity(
    fresh: ResidentRequestGroup,
    reuse: ResidentRequestGroup,
    layer_indices: Iterable[int],
) -> dict[str, Any]:
    from qcomem_vllm_paged_fair_control import strict_logical_layout_parity

    _require(fresh.resident_count == reuse.resident_count, "resident counts differ")
    rows = []
    for request_index in range(fresh.resident_count):
        for layer_index in layer_indices:
            parity = strict_logical_layout_parity(
                fresh.requests[request_index].layers[layer_index].sequence,
                reuse.requests[request_index].layers[layer_index].sequence,
            )
            rows.append(
                {"request_index": request_index, "layer_idx": int(layer_index), **parity}
            )
    return {
        "passed": True,
        "resident_count": fresh.resident_count,
        "row_count": len(rows),
        "rows": rows,
    }


class MultiForkHitLedger:
    """Per-request same-kernel ledger carrying multi-resident identity."""

    def __init__(
        self,
        plan: Any,
        request: Any,
        *,
        request_index: int,
        resident_count: int,
        request_policy: str,
        expected_calls_per_layer: int,
        initial_query_tokens: int,
        kernel: Any | None = None,
        strict_position_values: bool = False,
        call_observer: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        _validate_count(resident_count)
        _require(0 <= request_index < resident_count, "request index is outside resident group")
        _require(request_policy in MULTIFORK_POLICIES, "unknown ledger policy")
        _require(type(expected_calls_per_layer) is int and expected_calls_per_layer > 0, "call budget must be positive")
        _require(type(initial_query_tokens) is int and initial_query_tokens > 0, "initial query tokens must be positive")
        self.indices = tuple(plan.full_attention_layer_indices)
        _require(len(self.indices) == 10, "formal Qwen3.5 run requires ten full layers")
        self.request_index = request_index
        self.resident_count = resident_count
        self.request_policy = request_policy
        self.expected_calls_per_layer = expected_calls_per_layer
        self.initial_query_tokens = initial_query_tokens
        self.strict_position_values = bool(strict_position_values)
        _require(
            call_observer is None or callable(call_observer),
            "call observer must be callable or None",
        )
        self.call_observer = call_observer
        self.kernel = _resolve_vllm_unified_attention() if kernel is None else kernel
        _require(callable(self.kernel), "unified_attention kernel is not callable")
        try:
            signature = str(inspect.signature(self.kernel))
        except (TypeError, ValueError):
            signature = "<signature-unavailable>"
        self.kernel_identity = {
            "callable_id": id(self.kernel),
            "module": str(getattr(self.kernel, "__module__", type(self.kernel).__module__)),
            "qualname": str(getattr(self.kernel, "__qualname__", type(self.kernel).__qualname__)),
            "signature": signature,
        }
        self._frozen_kernel = self.kernel
        self.mask_contract = PRODUCTION_MASK_CONTRACT
        self.arena_ids = {}
        self.sequence_ids = {}
        self.last_lengths = {}
        self.last_append_event_counts = {}
        self.observed_append_capture_ids: set[str] = set()
        for index in self.indices:
            layer = request.layers[index]
            _require(isinstance(layer, Q16KernelPagedLayer), f"request layer {index} is not paged")
            self.arena_ids[index] = id(layer.sequence.arena)
            self.sequence_ids[index] = id(layer.sequence)
            self.last_lengths[index] = int(layer.sequence.sequence_length)
            self.last_append_event_counts[index] = int(
                layer.sequence._append_event_count
            )
        self.counts: Counter[int] = Counter()
        self.calls: list[dict[str, Any]] = []

    def attention_forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: Q16KernelPagedTensorView,
        value: Q16KernelPagedTensorView,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        index = getattr(module, "layer_idx", None)
        _runtime_require(index in self.arena_ids, "KV_PAGED_VIEW", f"unexpected full-attention layer {index}")
        _runtime_require(
            self.kernel is self._frozen_kernel,
            "KERNEL_CALLABLE_ID",
            "unified_attention callable changed after ledger construction",
        )
        _runtime_require(
            isinstance(key, Q16KernelPagedTensorView)
            and isinstance(value, Q16KernelPagedTensorView),
            "KV_PAGED_VIEW",
            "dense fallback",
        )
        sequence = key.sequence
        _runtime_require(value.sequence is sequence, "KV_SEQUENCE_ID", "K/V sequence differs")
        _runtime_require(
            id(sequence) == self.sequence_ids[index],
            "KV_SEQUENCE_ID",
            "request sequence changed or another request was misbound",
        )
        _runtime_require(id(sequence.arena) == self.arena_ids[index], "KV_SEQUENCE_ID", "request arena changed")
        _require(self.counts[index] < self.expected_calls_per_layer, "call budget exceeded")
        query_length = int(query.shape[-2])
        delta = int(sequence.sequence_length) - self.last_lengths[index]
        _require(delta == query_length, "current append delta differs from query tokens")
        append_event_count = int(sequence._append_event_count)
        _runtime_require(
            append_event_count == self.last_append_event_counts[index] + 1,
            "KV_APPEND_EVENT",
            "attention must consume exactly one cache append event",
        )
        append_audit = sequence.last_append_audit
        _runtime_require(
            isinstance(append_audit, dict)
            and append_audit.get("append_event_index")
            == self.last_append_event_counts[index]
            and append_audit.get("append_tokens") == delta
            and append_audit.get("sequence_length_after")
            == int(sequence.sequence_length),
            "KV_APPEND_EVENT",
            "cache append receipt differs from attention delta",
        )
        _runtime_require(
            attention_mask is None and not sequence.strict_mask_check,
            "MASK_CONTRACT",
            "production path materialized a mask",
        )
        position_ids = kwargs.pop("position_ids", None)
        try:
            position = validate_qwen35_post_rope_position_ids(
                position_ids,
                query=query,
                total_length=int(sequence.sequence_length),
                strict_tail_values=self.strict_position_values,
            )
        except Exception as exc:
            raise RuntimeInvariantError(
                "POSITION_CANONICAL_VALUES",
                f"post-RoPE position validation failed: {exc}",
            ) from exc
        audit: dict[str, Any] = {}
        result = vllm_triton_q16_paged_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            audit=audit,
            _kernel=self.kernel,
            **kwargs,
        )
        if self.call_observer is not None:
            capture_id = append_audit.get("capture_id")
            _runtime_require(
                isinstance(capture_id, str)
                and bool(capture_id)
                and capture_id not in self.observed_append_capture_ids,
                "KV_APPEND_EVENT",
                "oracle observer requires one unique append capture per call",
            )
            candidate_output = result[0]
            # The observer receives only detached CPU clones, but a hostile or
            # buggy callback could still close over the live request.  Freeze
            # the cheap identity/version/state surface around the callback so
            # audit instrumentation cannot append again, swap a cache/table,
            # or mutate one of those tensors and have the ledger consume the
            # altered state as if it belonged to this call.
            arena = sequence.arena

            def tensor_guard(tensor: torch.Tensor, *, include_values: bool) -> tuple[Any, ...]:
                # Inference tensors intentionally have no PyTorch version
                # counter.  Bind identity, storage/layout metadata and the
                # small routing-table values without consulting ``_version``.
                base: tuple[Any, ...] = (
                    id(tensor),
                    int(tensor.data_ptr()),
                    int(tensor.storage_offset()),
                    tuple(int(item) for item in tensor.shape),
                    tuple(int(item) for item in tensor.stride()),
                    str(tensor.dtype),
                    str(tensor.device),
                )
                if not include_values:
                    return base
                return base + (
                    tuple(int(item) for item in tensor.detach().reshape(-1).cpu().tolist()),
                )

            def observer_guard() -> tuple[Any, ...]:
                return (
                    id(sequence),
                    id(arena),
                    int(sequence.sequence_length),
                    int(sequence.appended_tokens),
                    int(sequence._append_event_count),
                    sequence.last_append_capture_id,
                    dict(sequence.last_append_audit or {}),
                    tuple(sequence._next_private),
                    tuple(sequence._tail_detached),
                    tuple(tuple(row) for row in sequence._logical_physical),
                    int(sequence.partial_tail_staging_copy_nbytes),
                    tensor_guard(sequence.block_table, include_values=True),
                    tensor_guard(sequence.reservations, include_values=True),
                    tensor_guard(arena.key_cache, include_values=False),
                    tensor_guard(arena.value_cache, include_values=False),
                )

            frozen_observer_guard = observer_guard()
            self.call_observer(
                {
                    "observer_schema": "qcomem-forkaudit-call-observer-v2",
                    "query_cpu": query.detach().contiguous().cpu().clone(),
                    "candidate_output_cpu": candidate_output.detach()
                    .contiguous()
                    .cpu()
                    .clone(),
                    "attention_mask_is_none": attention_mask is None,
                    "position_ids_cpu": (
                        None
                        if position_ids is None
                        else position_ids.detach().contiguous().cpu().clone()
                    ),
                    "position_audit": dict(position),
                    "kernel_audit": dict(audit),
                    "effective_scaling": float(audit["softmax_scale"]),
                    "append_capture_id": capture_id,
                    "append_audit": dict(append_audit),
                    "layer_idx": int(index),
                    "request_index": self.request_index,
                    "resident_count": self.resident_count,
                    "request_policy": self.request_policy,
                }
            )
            _runtime_require(
                observer_guard() == frozen_observer_guard,
                "KV_APPEND_EVENT",
                "call observer mutated the live request/cache state",
            )
            self.observed_append_capture_ids.add(capture_id)
        # Advance consumption only after the kernel and optional observer both
        # completed.  A failed audit cannot silently consume an append event.
        self.last_lengths[index] = int(append_audit["sequence_length_after"])
        self.last_append_event_counts[index] = append_event_count
        self.counts[index] += 1
        self.calls.append(
            {
                "request_index": self.request_index,
                "resident_count": self.resident_count,
                "layer_idx": int(index),
                "request_policy": self.request_policy,
                "protocol": MULTIFORK_PROTOCOL,
                "kernel_identity": dict(self.kernel_identity),
                "current_append_delta_tokens": delta,
                "mask_contract": self.mask_contract,
                "materialized_attention_mask_nbytes": 0,
                "mask_validation_host_syncs": 0,
                "append_capture_id": append_audit.get("capture_id"),
                "append_audit": dict(append_audit),
                **audit,
                **position,
            }
        )
        return result

    def verify_complete(self) -> dict[str, Any]:
        expected = {index: self.expected_calls_per_layer for index in self.indices}
        actual = {index: self.counts[index] for index in self.indices}
        _require(actual == expected, "multi-fork same-kernel intercept incomplete")
        expected_layer_order = list(self.indices) * self.expected_calls_per_layer
        _require(
            [int(row["layer_idx"]) for row in self.calls] == expected_layer_order,
            "multi-fork layer/round call order drift",
        )
        expected_deltas = [self.initial_query_tokens] + [1] * (
            self.expected_calls_per_layer - 1
        )
        actual_deltas = [
            [
                int(row["current_append_delta_tokens"])
                for row in self.calls[
                    round_index * len(self.indices) : (round_index + 1) * len(self.indices)
                ]
            ]
            for round_index in range(self.expected_calls_per_layer)
        ]
        _require(
            actual_deltas
            == [[delta] * len(self.indices) for delta in expected_deltas],
            "multi-fork round append-delta schedule drift",
        )
        _require(
            all(row.get("kernel_identity") == self.kernel_identity for row in self.calls),
            "multi-fork per-call kernel identity drift",
        )
        return {
            "verified": True,
            "protocol": MULTIFORK_PROTOCOL,
            "request_index": self.request_index,
            "resident_count": self.resident_count,
            "request_policy": self.request_policy,
            "kernel_mode": KERNEL_MODE,
            "kernel_identity": dict(self.kernel_identity),
            "same_unified_attention_kernel": True,
            "counts": actual,
            "total_calls": sum(actual.values()),
            "round_major_request_local_layer_order_verified": True,
            "initial_query_tokens": self.initial_query_tokens,
            "dense_fallback_calls": 0,
            "full_kv_concatenations": 0,
            "mask_contract": self.mask_contract,
            "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
            "strict_position_values": self.strict_position_values,
            "call_observer_enabled": self.call_observer is not None,
            "calls": tuple(self.calls),
        }


def register_multifork_backend(ledger: MultiForkHitLedger) -> str:
    return register_qwen35_vllm_q16_backend(ledger).name


def linear_capacity_fit(rows: Sequence[dict[str, Any]], field: str) -> dict[str, Any]:
    """Fit one auditable bytes-vs-N line without external dependencies."""

    _require(len(rows) == len(MULTIFORK_COUNTS), "capacity curve must contain every N")
    raw_counts = [row.get("resident_count") for row in rows]
    _require(
        all(type(value) is int for value in raw_counts),
        "capacity curve resident_count must be a present non-bool integer",
    )
    xs = list(raw_counts)
    _require(tuple(xs) == MULTIFORK_COUNTS, "capacity curve N order drift")
    raw_values = [row.get(field) for row in rows]
    _require(
        all(type(value) is int and value >= 0 for value in raw_values),
        "capacity fit bytes must be present non-bool integers >= 0",
    )
    ys = [float(value) for value in raw_values]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    _require(denominator > 0, "capacity fit has zero denominator")
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    intercept = y_mean - slope * x_mean
    predictions = [intercept + slope * x for x in xs]
    residual = sum((y - prediction) ** 2 for y, prediction in zip(ys, predictions))
    total = sum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 if total == 0 and residual == 0 else 1.0 - residual / total
    _require(
        all(math.isfinite(value) for value in (slope, intercept, r_squared)),
        "capacity fit produced a non-finite result",
    )
    return {
        "field": field,
        "sample_count": len(rows),
        "resident_counts": xs,
        "values_nbytes": ys,
        "slope_nbytes_per_request": slope,
        "intercept_nbytes": intercept,
        "r_squared": r_squared,
    }


__all__ = [
    "GDN_BORROW_IMMUTABLE_BASE",
    "GDN_MATERIALIZE_REQUEST_BASE",
    "KVBindingGuard",
    "KVRequestBindingRow",
    "MULTIFORK_COUNTS",
    "MULTIFORK_GDN_BASE_POLICIES",
    "MULTIFORK_POLICIES",
    "MULTIFORK_PROTOCOL",
    "MultiForkHitLedger",
    "QComemMultiForkError",
    "RuntimeInvariantError",
    "ResidentRequestGroup",
    "build_deterministic_distinct_queries",
    "build_pg19_train_query_bank",
    "build_resident_request_group",
    "linear_capacity_fit",
    "register_multifork_backend",
    "resident_storage_breakdown",
    "strict_group_logical_parity",
    "source_document_physical_digests",
    "validate_runtime_kv_ownership",
    "validate_resident_group_ownership",
]
