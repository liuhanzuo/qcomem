from __future__ import annotations

"""Fair, same-kernel controls for the Q16 vLLM paged-cache path.

The original fused experiment compared Transformers eager attention with
vLLM ``unified_attention``.  That comparison mixes two questions: cache
ownership/layout and backend arithmetic.  This module keeps the backend fixed
and exposes two request policies over the same NHD/block-table contract:

``fresh-full-copy``
    Allocate a request-owned block pool and materialize every physical
    document block into it before the query.

``shared-document-reuse``
    Reuse the immutable document blocks and allocate only request-private
    append blocks (plus a small logical block table).

Both policies ultimately call the same vLLM 0.26 ``unified_attention`` entry
point through :mod:`qcomem_vllm_paged_kernel`.  Physical block ids and padding
are deliberately *not* correctness invariants: parity is checked after the
active block tables are canonicalized and only over valid logical tokens.
"""

import copy
import inspect
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Literal

import torch

from qcomem_qwen35_native_cache import install_native_functional_linear_cache
from qcomem_qwen35_vllm_paged_integration import (
    POST_ROPE_POSITION_IDS_CONTRACT,
    Qwen35VllmPagedIntegrationError,
    register_qwen35_vllm_q16_backend,
    validate_qwen35_post_rope_position_ids,
)
from qcomem_vllm_paged_kernel import (
    KERNEL_MODE,
    Q16KernelPagedDocumentLayer,
    Q16KernelPagedLayer,
    Q16KernelPagedTensorView,
    Q16PagedArena,
    Q16PagedSequence,
    QComemPagedKernelError,
    _resolve_vllm_unified_attention,
    vllm_triton_q16_paged_attention_forward,
)


FRESH_CONTROL = "vllm-q16-fresh-full-copy-control"
SHARED_REUSE = "vllm-q16-shared-document-reuse"
FAIR_CONFIGS = (FRESH_CONTROL, SHARED_REUSE)
FAIR_PROTOCOL = "same-vllm-unified-attention-q16-single-request-v2"
PRODUCTION_MASK_CONTRACT = "prevalidated-no-padding-tail-causal"


class QComemFairControlError(RuntimeError):
    """Raised when the same-kernel fairness contract is violated."""


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def storage_inventory(*roots: Any) -> dict[tuple[str, int, int], int]:
    """Return unique tensor storages reachable from one or more objects."""

    seen_objects: set[int] = set()
    inventory: dict[tuple[str, int, int], int] = {}

    def visit(value: Any) -> None:
        identity = id(value)
        if identity in seen_objects:
            return
        seen_objects.add(identity)
        if isinstance(value, torch.Tensor):
            key = _storage_key(value)
            inventory[key] = key[2]
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
    return inventory


def _inventory_nbytes(
    inventory: dict[tuple[str, int, int], int], *, accelerator_only: bool
) -> int:
    return sum(
        size
        for (device, _pointer, _size), size in inventory.items()
        if not accelerator_only or device.startswith("cuda")
    )


def storage_residency(persistent: Any, request: Any) -> dict[str, int]:
    """Account shared and request-unique storage without double counting."""

    persistent_inventory = storage_inventory(persistent)
    request_inventory = storage_inventory(request)
    shared = set(persistent_inventory) & set(request_inventory)
    request_unique = set(request_inventory) - set(persistent_inventory)
    union = dict(persistent_inventory)
    union.update(request_inventory)

    def total(keys: Iterable[tuple[str, int, int]], accelerator_only: bool) -> int:
        return sum(
            key[2]
            for key in keys
            if not accelerator_only or key[0].startswith("cuda")
        )

    return {
        "persistent_total_nbytes": _inventory_nbytes(
            persistent_inventory, accelerator_only=False
        ),
        "persistent_accelerator_nbytes": _inventory_nbytes(
            persistent_inventory, accelerator_only=True
        ),
        "request_total_nbytes": _inventory_nbytes(
            request_inventory, accelerator_only=False
        ),
        "request_accelerator_nbytes": _inventory_nbytes(
            request_inventory, accelerator_only=True
        ),
        "shared_total_nbytes": total(shared, False),
        "shared_accelerator_nbytes": total(shared, True),
        "request_unique_total_nbytes": total(request_unique, False),
        "request_unique_accelerator_nbytes": total(request_unique, True),
        "combined_unique_total_nbytes": _inventory_nbytes(
            union, accelerator_only=False
        ),
        "combined_unique_accelerator_nbytes": _inventory_nbytes(
            union, accelerator_only=True
        ),
    }


def full_attention_storage_breakdown(
    persistent: Any,
    request: Any,
    layer_indices: Iterable[int],
    *,
    request_policy: str,
) -> dict[str, Any]:
    """Split Q16 arena capacity, valid payload and metadata by ownership.

    ``Q16PagedArena`` preallocates one request reservation in the persistent
    source arena.  Consequently its full allocation must never be described
    as pure document storage.  This ledger makes that reserved capacity and a
    fresh control's duplicate pool explicit.
    """

    if request_policy not in FAIR_CONFIGS:
        raise QComemFairControlError("unknown storage-ledger request policy")
    rows = []
    totals: Counter[str] = Counter()
    for index in layer_indices:
        source_layer = persistent.layers[index]
        request_layer = request.layers[index]
        if not isinstance(source_layer, Q16KernelPagedDocumentLayer):
            raise QComemFairControlError(f"persistent layer {index} is not a document facade")
        if not isinstance(request_layer, Q16KernelPagedLayer):
            raise QComemFairControlError(f"request layer {index} is not paged")
        source = source_layer.arena
        target = request_layer.sequence.arena
        if source.max_forks != 1 or target.max_forks != 1:
            raise QComemFairControlError("storage formula requires max_forks=1")
        source_geometry = (
            source.batch_size,
            source.page_size,
            source.num_key_value_heads,
            source.head_dim,
            source.document_length,
            source.max_append_tokens,
            source.document_blocks_per_sequence,
            source.private_blocks_per_sequence,
        )
        target_geometry = (
            target.batch_size,
            target.page_size,
            target.num_key_value_heads,
            target.head_dim,
            target.document_length,
            target.max_append_tokens,
            target.document_blocks_per_sequence,
            target.private_blocks_per_sequence,
        )
        if source_geometry != target_geometry:
            raise QComemFairControlError(f"layer {index} source/target geometry differs")
        element_size = source.key_cache.element_size()
        block_bytes = (
            2
            * source.page_size
            * source.num_key_value_heads
            * source.head_dim
            * element_size
        )
        source_document_allocated = (
            source.batch_size * source.document_blocks_per_sequence * block_bytes
        )
        valid_payload = source.audit.document_payload_nbytes
        source_padding = source_document_allocated - valid_payload
        source_private_reservation = (
            source.batch_size * source.private_blocks_per_sequence * block_bytes
        )
        is_fresh = request_policy == FRESH_CONTROL
        if is_fresh and target.storage_keys & source.storage_keys:
            raise QComemFairControlError(f"fresh layer {index} shares source K/V storage")
        if not is_fresh and target is not source:
            raise QComemFairControlError(f"reuse layer {index} did not share source arena")
        appended = int(request_layer.sequence.appended_tokens)
        tail = source.document_length % source.page_size
        detached_tail = tail if appended > 0 and tail else 0
        active_private_payload = (
            2
            * source.batch_size
            * (detached_tail + appended)
            * source.num_key_value_heads
            * source.head_dim
            * element_size
        )
        active_private_blocks = (
            math.ceil((detached_tail + appended) / source.page_size)
            if appended > 0
            else 0
        )
        active_private_allocated = active_private_blocks * source.batch_size * block_bytes
        reserved_private_allocated = source.batch_size * source.private_blocks_per_sequence * block_bytes
        fresh_document_allocated = source_document_allocated if is_fresh else 0
        fresh_private_reservation = source_private_reservation if is_fresh else 0
        row = {
            "layer_idx": int(index),
            "block_bytes_formula": (
                "2*page_size*kv_heads*head_dim*element_size"
            ),
            "block_bytes": block_bytes,
            "valid_document_payload_nbytes": valid_payload,
            "source_document_allocated_nbytes": source_document_allocated,
            "source_document_padding_nbytes": source_padding,
            "source_private_reservation_nbytes": source_private_reservation,
            "source_total_arena_allocated_nbytes": (
                source_document_allocated + source_private_reservation
            ),
            "fresh_duplicate_document_allocated_nbytes": fresh_document_allocated,
            "fresh_duplicate_document_padding_nbytes": source_padding if is_fresh else 0,
            "fresh_private_reservation_nbytes": fresh_private_reservation,
            "active_request_private_payload_nbytes": active_private_payload,
            "active_request_private_blocks": active_private_blocks * source.batch_size,
            "active_request_private_allocated_page_nbytes": active_private_allocated,
            "request_private_reserved_unused_nbytes": (
                reserved_private_allocated - active_private_allocated
            ),
            "active_request_appended_tokens": appended,
            "active_request_detached_tail_tokens": detached_tail,
            "request_block_table_accelerator_nbytes": (
                request_layer.sequence.block_table.untyped_storage().nbytes()
            ),
            "source_document_table_accelerator_nbytes": (
                source.document_block_table.untyped_storage().nbytes()
            ),
            "fresh_document_table_accelerator_nbytes": (
                target.document_block_table.untyped_storage().nbytes()
                if is_fresh
                else 0
            ),
            "source_cpu_reservation_metadata_nbytes": (
                source.private_block_reservations.untyped_storage().nbytes()
            ),
            "fresh_cpu_reservation_metadata_nbytes": (
                target.private_block_reservations.untyped_storage().nbytes()
                if is_fresh
                else 0
            ),
            "source_document_storage_shared_by_request": not is_fresh,
        }
        for key, value in row.items():
            if key.endswith("_nbytes") or key in ("block_bytes", "active_request_private_blocks"):
                totals[key] += int(value)
        rows.append(row)
    return {
        "request_policy": request_policy,
        "full_attention_layer_count": len(rows),
        "scope": "ten-full-attention-layers-only",
        "linear_gdn_included": False,
        "source_arena_includes_preallocated_private_reservation": True,
        "invalid_final_block_padding_is_payload": False,
        "totals": dict(totals),
        "layers": rows,
    }


def linear_gdn_shared_base_contract(
    persistent: Any,
    request: Any,
    linear_layer_indices: Iterable[int],
) -> dict[str, Any]:
    """Prove both arms start GDN from shared tensors and functional rebind."""

    rows = []
    for index in linear_layer_indices:
        source = storage_inventory(persistent.layers[index])
        target = storage_inventory(request.layers[index])
        same = set(source) == set(target)
        mode = getattr(request.layers[index], "_qcomem_update_mode", None)
        if not same or mode != "functional-state-rebind":
            raise QComemFairControlError(
                f"linear layer {index} does not share the functional persistent base"
            )
        rows.append(
            {
                "layer_idx": int(index),
                "persistent_tensor_storage_keys_equal_at_request_start": same,
                "update_mode": mode,
                "persistent_base_nbytes": sum(source.values()),
            }
        )
    return {
        "passed": True,
        "linear_layer_count": len(rows),
        "persistent_tensor_base_shared_at_request_start": True,
        "request_updates_are_functional_rebind": True,
        "scope": "common-to-both-full-attention-ownership-arms",
        "layers": rows,
    }


def _named_tensor_leaves(value: Any, prefix: str = "") -> dict[str, torch.Tensor]:
    leaves: dict[str, torch.Tensor] = {}
    seen: set[int] = set()

    def visit(item: Any, path: str) -> None:
        identity = id(item)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(item, torch.Tensor):
            leaves[path] = item
        elif isinstance(item, dict):
            for key in sorted(item, key=lambda value: str(value)):
                visit(item[key], f"{path}[{key!r}]")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif hasattr(item, "__dict__"):
            for name in sorted(vars(item)):
                # Bound methods create cycles and hold no cache tensor payload.
                if name.startswith("_qcomem_original_"):
                    continue
                visit(vars(item)[name], f"{path}.{name}")

    visit(value, prefix or "root")
    return leaves


def snapshot_linear_gdn_state(
    cache: Any,
    linear_layer_indices: Iterable[int],
) -> dict[str, torch.Tensor]:
    """Clone a train-gate-only snapshot of persistent GDN tensor leaves."""

    snapshot: dict[str, torch.Tensor] = {}
    for index in linear_layer_indices:
        for path, tensor in _named_tensor_leaves(
            cache.layers[index], f"layers[{int(index)}]"
        ).items():
            snapshot[path] = tensor.clone()
    return snapshot


def verify_linear_gdn_state_parity(
    left: Any,
    right: Any,
    persistent: Any,
    persistent_snapshot: dict[str, torch.Tensor],
    linear_layer_indices: Iterable[int],
) -> dict[str, Any]:
    """Require bitwise-equal functional GDN results and immutable base state."""

    left_leaves: dict[str, torch.Tensor] = {}
    right_leaves: dict[str, torch.Tensor] = {}
    persistent_leaves: dict[str, torch.Tensor] = {}
    for index in linear_layer_indices:
        prefix = f"layers[{int(index)}]"
        left_leaves.update(_named_tensor_leaves(left.layers[index], prefix))
        right_leaves.update(_named_tensor_leaves(right.layers[index], prefix))
        persistent_leaves.update(_named_tensor_leaves(persistent.layers[index], prefix))
    if set(left_leaves) != set(right_leaves):
        raise QComemFairControlError("fresh/reuse GDN tensor schemas differ")
    if set(persistent_leaves) != set(persistent_snapshot):
        raise QComemFairControlError("persistent GDN tensor schema changed")
    for path in sorted(left_leaves):
        if not torch.equal(left_leaves[path], right_leaves[path]):
            raise QComemFairControlError(f"fresh/reuse GDN state differs at {path}")
    for path in sorted(persistent_leaves):
        if not torch.equal(persistent_leaves[path], persistent_snapshot[path]):
            raise QComemFairControlError(f"persistent GDN base mutated at {path}")
    return {
        "passed": True,
        "linear_layer_count": len(tuple(linear_layer_indices)),
        "fresh_reuse_tensor_schema_equal": True,
        "fresh_reuse_functional_state_bitwise_exact": True,
        "persistent_tensor_base_unchanged": True,
        "tensor_leaf_count": len(left_leaves),
    }


@dataclass(frozen=True)
class FreshQ16Materialization:
    document_payload_nbytes: int
    document_block_copy_nbytes: int
    copied_padding_nbytes: int
    allocated_request_pool_nbytes: int
    document_blocks: int
    private_blocks: int
    source_storage_shared: bool
    request_policy: str = FRESH_CONTROL


def materialize_fresh_q16_request_layer(
    document: Q16KernelPagedDocumentLayer,
    *,
    strict_mask_check: bool | None = None,
) -> tuple[Q16KernelPagedLayer, FreshQ16Materialization]:
    """Copy a document facade into one independent request-owned block pool.

    The physical document blocks (including ignored padding in the final
    block) are copied exactly once.  The fresh pool has capacity for one
    request only.  A later partial-tail detach is performed by the same
    :class:`Q16PagedSequence` implementation used by the shared path.
    """

    if not isinstance(document, Q16KernelPagedDocumentLayer):
        raise QComemFairControlError("fresh control requires a Q16 document layer")
    source = document.arena
    if source.max_forks != 1:
        raise QComemFairControlError(
            "formal fair control is single-request and requires max_forks=1"
        )
    if strict_mask_check is None:
        strict_mask_check = document.strict_mask_check
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
    # This is the full-copy control's defining operation.  Padding is copied
    # but never compared or consumed because seqused_k supplies the valid end.
    key_cache[:document_physical].copy_(source.key_cache[:document_physical])
    value_cache[:document_physical].copy_(source.value_cache[:document_physical])
    document_block_table = source.document_block_table.clone()
    reservations = torch.arange(
        document_physical,
        physical_blocks,
        device="cpu",
        dtype=torch.int64,
    ).reshape(1, source.batch_size, source.private_blocks_per_sequence)
    fresh = Q16PagedArena(
        key_cache,
        value_cache,
        document_block_table,
        document_length=source.document_length,
        max_append_tokens=source.max_append_tokens,
        max_forks=1,
        private_block_reservations=reservations,
    )
    if fresh.storage_keys & source.storage_keys:
        raise QComemFairControlError("fresh control accidentally shares source K/V storage")
    copied = (
        2
        * document_physical
        * source.page_size
        * source.num_key_value_heads
        * source.head_dim
        * source.key_cache.element_size()
    )
    payload = source.audit.document_payload_nbytes
    layer = Q16KernelPagedLayer(
        fresh.fork(strict_mask_check=bool(strict_mask_check))
    )
    return layer, FreshQ16Materialization(
        document_payload_nbytes=payload,
        document_block_copy_nbytes=copied,
        copied_padding_nbytes=copied - payload,
        allocated_request_pool_nbytes=fresh.allocated_pool_nbytes,
        document_blocks=source.document_blocks_per_sequence,
        private_blocks=source.private_blocks_per_sequence,
        source_storage_shared=False,
    )


def _valid_block_slices(sequence: Q16PagedSequence):
    arena = sequence.arena
    length = sequence.sequence_length
    blocks = math.ceil(length / arena.page_size)
    for batch_index in range(arena.batch_size):
        for logical_block in range(blocks):
            start = logical_block * arena.page_size
            valid = min(arena.page_size, length - start)
            physical = int(sequence.active_block_table[batch_index, logical_block])
            if physical < 0 or physical >= int(arena.key_cache.shape[0]):
                raise QComemFairControlError("active block table points outside the pool")
            yield batch_index, logical_block, valid, physical


def strict_logical_layout_parity(
    reference: Q16PagedSequence,
    candidate: Q16PagedSequence,
) -> dict[str, Any]:
    """Compare canonical logical layout and valid K/V payload exactly.

    This is an untimed gate and may synchronize to inspect block ids.  Raw
    physical ids, total pool capacity and invalid final-block padding are not
    invariants.
    """

    left, right = reference.arena, candidate.arena
    geometry = (
        left.batch_size,
        left.page_size,
        left.num_key_value_heads,
        left.head_dim,
        reference.sequence_length,
    )
    other_geometry = (
        right.batch_size,
        right.page_size,
        right.num_key_value_heads,
        right.head_dim,
        candidate.sequence_length,
    )
    if geometry != other_geometry:
        raise QComemFairControlError(
            f"same-kernel logical geometry differs: {geometry} != {other_geometry}"
        )
    if left.key_cache.dtype != right.key_cache.dtype or left.key_cache.device != right.key_cache.device:
        raise QComemFairControlError("same-kernel cache dtype/device differs")
    left_rows = list(_valid_block_slices(reference))
    right_rows = list(_valid_block_slices(candidate))
    left_canonical = [(b, logical, valid) for b, logical, valid, _ in left_rows]
    right_canonical = [(b, logical, valid) for b, logical, valid, _ in right_rows]
    if left_canonical != right_canonical:
        raise QComemFairControlError("canonical active logical block layouts differ")
    key_exact = True
    value_exact = True
    valid_tokens = 0
    if len(left_rows) != len(right_rows):
        raise QComemFairControlError("canonical block row counts differ")
    for left_row, right_row in zip(left_rows, right_rows):
        batch, _logical, valid, left_physical = left_row
        right_batch, _logical2, right_valid, right_physical = right_row
        if (batch, valid) != (right_batch, right_valid):
            raise QComemFairControlError("canonical block rows differ")
        key_exact = key_exact and bool(
            torch.equal(
                left.key_cache[left_physical, :valid],
                right.key_cache[right_physical, :valid],
            )
        )
        value_exact = value_exact and bool(
            torch.equal(
                left.value_cache[left_physical, :valid],
                right.value_cache[right_physical, :valid],
            )
        )
        valid_tokens += valid
    if not key_exact or not value_exact:
        raise QComemFairControlError("valid logical K/V payload is not bitwise exact")
    return {
        "passed": True,
        "canonical_layout_equal": True,
        "valid_key_payload_bitwise_exact": key_exact,
        "valid_value_payload_bitwise_exact": value_exact,
        "invalid_final_block_padding_compared": False,
        "raw_physical_block_ids_required_equal": False,
        "physical_pool_capacity_required_equal": False,
        "valid_block_rows": len(left_rows),
        "valid_tokens_across_batch": valid_tokens,
        "sequence_length": reference.sequence_length,
        "page_size": left.page_size,
        "dtype": str(left.key_cache.dtype),
        "device": str(left.key_cache.device),
        "untimed_host_synchronizing_gate": True,
    }


def _snapshot_valid_document_blocks(arena: Q16PagedArena) -> list[tuple[torch.Tensor, torch.Tensor]]:
    rows: list[tuple[torch.Tensor, torch.Tensor]] = []
    for batch_index in range(arena.batch_size):
        for logical in range(arena.document_blocks_per_sequence):
            start = logical * arena.page_size
            valid = min(arena.page_size, arena.document_length - start)
            physical = batch_index * arena.document_blocks_per_sequence + logical
            rows.append(
                (
                    arena.key_cache[physical, :valid].clone(),
                    arena.value_cache[physical, :valid].clone(),
                )
            )
    return rows


def _document_snapshot_unchanged(
    arena: Q16PagedArena,
    snapshot: list[tuple[torch.Tensor, torch.Tensor]],
) -> bool:
    cursor = 0
    for batch_index in range(arena.batch_size):
        for logical in range(arena.document_blocks_per_sequence):
            start = logical * arena.page_size
            valid = min(arena.page_size, arena.document_length - start)
            physical = batch_index * arena.document_blocks_per_sequence + logical
            before_key, before_value = snapshot[cursor]
            cursor += 1
            if not torch.equal(arena.key_cache[physical, :valid], before_key):
                return False
            if not torch.equal(arena.value_cache[physical, :valid], before_value):
                return False
    return True


def build_same_kernel_q16_sequence_pair(
    document_key: torch.Tensor,
    document_value: torch.Tensor,
    append_key: torch.Tensor,
    append_value: torch.Tensor,
    *,
    page_size: int,
    max_append_tokens: int,
) -> tuple[Q16KernelPagedLayer, Q16KernelPagedLayer, dict[str, Any]]:
    """Build and strictly compare fresh-control and shared-reuse sequences."""

    source = Q16PagedArena.from_dense_document(
        document_key,
        document_value,
        page_size=page_size,
        max_append_tokens=max_append_tokens,
        max_forks=1,
    )
    source_document = Q16KernelPagedDocumentLayer(source, strict_mask_check=True)
    snapshot = _snapshot_valid_document_blocks(source)
    fresh, materialization = materialize_fresh_q16_request_layer(source_document)
    reuse = source_document.fork()
    before_fresh = fresh.sequence.sequence_length
    before_reuse = reuse.sequence.sequence_length
    fresh.update(append_key, append_value)
    reuse.update(append_key, append_value)
    incoming = int(append_key.shape[-2])
    if (
        fresh.sequence.sequence_length - before_fresh != incoming
        or reuse.sequence.sequence_length - before_reuse != incoming
    ):
        raise QComemFairControlError("current append delta differs from query length")
    layout = strict_logical_layout_parity(fresh.sequence, reuse.sequence)
    source_immutable = _document_snapshot_unchanged(source, snapshot)
    if not source_immutable:
        raise QComemFairControlError("shared immutable document blocks were modified")
    tail = source.document_length % source.page_size
    expected_tail_copy = (
        2
        * source.batch_size
        * tail
        * source.num_key_value_heads
        * source.head_dim
        * source.key_cache.element_size()
        if tail
        else 0
    )
    if (
        fresh.sequence.partial_tail_staging_copy_nbytes != expected_tail_copy
        or reuse.sequence.partial_tail_staging_copy_nbytes != expected_tail_copy
    ):
        raise QComemFairControlError("partial-tail copy accounting differs")
    audit = {
        "passed": True,
        "protocol": FAIR_PROTOCOL,
        "quantization": "Q16",
        "batch_semantics": "batch-1-equal-length-only",
        "single_request_only": True,
        "layout": layout,
        "document_source_immutable": source_immutable,
        "fresh_source_storage_shared": materialization.source_storage_shared,
        "fresh_document_block_copy_nbytes": materialization.document_block_copy_nbytes,
        "fresh_document_payload_nbytes": materialization.document_payload_nbytes,
        "fresh_copied_padding_nbytes": materialization.copied_padding_nbytes,
        "fresh_allocated_request_pool_nbytes": materialization.allocated_request_pool_nbytes,
        "shared_full_document_copy_nbytes": 0,
        "fresh_partial_tail_copy_nbytes": fresh.sequence.partial_tail_staging_copy_nbytes,
        "shared_partial_tail_copy_nbytes": reuse.sequence.partial_tail_staging_copy_nbytes,
        "document_tail_tokens": tail,
        "append_crossed_block_boundary": (
            math.ceil(source.document_length / source.page_size)
            != math.ceil((source.document_length + incoming) / source.page_size)
        ),
        "current_append_delta_tokens": incoming,
    }
    return fresh, reuse, audit


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


def materialize_qwen35_fresh_full_copy_request(
    cache: Any,
    plan: Any,
) -> tuple[Any, dict[str, Any]]:
    """Create a request-owned full-copy pool for every Qwen3.5 full layer."""

    memo: dict[int, Any] = {}
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, (list, tuple)):
        raise QComemFairControlError("cache.layers must be a sequence")
    for index in plan.full_attention_layer_indices:
        layer = layers[index]
        if not isinstance(layer, Q16KernelPagedDocumentLayer):
            raise QComemFairControlError(f"layer {index} is not a Q16 document facade")
        memo[id(layer)] = layer
    _seed_tensor_memo(cache, memo, set())
    request = copy.deepcopy(cache, memo)
    layer_rows = []
    copied = allocated = payload = 0
    arena_ids: dict[int, int] = {}
    for index in plan.full_attention_layer_indices:
        source = layers[index]
        target, audit = materialize_fresh_q16_request_layer(source)
        request.layers[index] = target
        copied += audit.document_block_copy_nbytes
        allocated += audit.allocated_request_pool_nbytes
        payload += audit.document_payload_nbytes
        arena_ids[index] = id(target.sequence.arena)
        layer_rows.append({"layer_idx": index, **audit.__dict__})
    install = install_native_functional_linear_cache(request, plan.gdn)
    if tuple(install.linear_layer_indices) != tuple(plan.linear_layer_indices):
        raise QComemFairControlError("fresh request missed a GDN functional seam")
    if tuple(install.full_attention_layer_indices) != tuple(
        plan.full_attention_layer_indices
    ):
        raise QComemFairControlError("fresh request plan differs from Q16 layers")
    return request, {
        "request_policy": FRESH_CONTROL,
        "single_request_only": True,
        "same_unified_attention_kernel": True,
        "full_document_staging_copy_nbytes": copied,
        "document_payload_nbytes": payload,
        "allocated_request_pool_nbytes": allocated,
        "source_document_storage_shared": False,
        "linear_functional_rebind": True,
        "full_attention_layer_count": len(arena_ids),
        "request_arena_ids": arena_ids,
        "layers": layer_rows,
    }


class Qwen35FairHitLedger:
    """Fail-closed per-policy ledger for one same-kernel request."""

    def __init__(
        self,
        plan: Any,
        request: Any,
        *,
        request_policy: Literal[
            "vllm-q16-fresh-full-copy-control",
            "vllm-q16-shared-document-reuse",
        ],
        expected_calls_per_layer: int,
        strict_tail_values: bool,
        kernel: Any | None = None,
    ) -> None:
        if request_policy not in FAIR_CONFIGS:
            raise QComemFairControlError("unknown fair request policy")
        if expected_calls_per_layer < 1:
            raise QComemFairControlError("expected calls must be positive")
        self.indices = tuple(plan.full_attention_layer_indices)
        if len(self.indices) != 10:
            raise QComemFairControlError("formal Qwen3.5 run requires ten full layers")
        self.request_policy = request_policy
        self.expected_calls_per_layer = expected_calls_per_layer
        self.strict_tail_values = bool(strict_tail_values)
        self.kernel = (
            _resolve_vllm_unified_attention() if kernel is None else kernel
        )
        if not callable(self.kernel):
            raise QComemFairControlError("unified_attention kernel must be callable")
        try:
            kernel_signature = str(inspect.signature(self.kernel))
        except (TypeError, ValueError):
            kernel_signature = "<signature-unavailable>"
        self.kernel_identity = {
            "callable_id": id(self.kernel),
            "module": str(getattr(self.kernel, "__module__", type(self.kernel).__module__)),
            "qualname": str(
                getattr(self.kernel, "__qualname__", type(self.kernel).__qualname__)
            ),
            "signature": kernel_signature,
        }
        self.mask_contract = (
            "strict-canonical-audit"
            if strict_tail_values
            else PRODUCTION_MASK_CONTRACT
        )
        self.arena_ids: dict[int, int] = {}
        self.last_sequence_lengths: dict[int, int] = {}
        for index in self.indices:
            layer = request.layers[index]
            if not isinstance(layer, Q16KernelPagedLayer):
                raise QComemFairControlError(f"request layer {index} is not paged")
            self.arena_ids[index] = id(layer.sequence.arena)
            self.last_sequence_lengths[index] = layer.sequence.sequence_length
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
        if index not in self.arena_ids:
            raise QComemFairControlError(f"unexpected full-attention layer {index}")
        if not isinstance(key, Q16KernelPagedTensorView) or not isinstance(
            value, Q16KernelPagedTensorView
        ):
            raise QComemFairControlError(f"layer {index} reached dense fallback")
        sequence = key.sequence
        if value.sequence is not sequence or id(sequence.arena) != self.arena_ids[index]:
            raise QComemFairControlError(f"layer {index} used the wrong request arena")
        if self.counts[index] >= self.expected_calls_per_layer:
            raise QComemFairControlError(f"layer {index} exceeded its call budget")
        query_length = int(query.shape[-2])
        delta = sequence.sequence_length - self.last_sequence_lengths[index]
        if delta != query_length:
            raise QComemFairControlError(
                f"layer {index} current append delta {delta} != query length {query_length}"
            )
        self.last_sequence_lengths[index] = sequence.sequence_length
        if not self.strict_tail_values:
            if attention_mask is not None or sequence.strict_mask_check:
                raise QComemFairControlError(
                    "timed same-kernel path requires prevalidated unpadded/no-mask input"
                )
        position_ids = kwargs.pop("position_ids", None)
        position = validate_qwen35_post_rope_position_ids(
            position_ids,
            query=query,
            total_length=sequence.sequence_length,
            strict_tail_values=self.strict_tail_values,
        )
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
        self.counts[index] += 1
        row = {
            "layer_idx": index,
            "request_policy": self.request_policy,
            "fair_protocol": FAIR_PROTOCOL,
            "same_unified_attention_kernel": True,
            "kernel_identity": dict(self.kernel_identity),
            "current_append_delta_tokens": delta,
            **audit,
            **position,
            "mask_contract": self.mask_contract,
        }
        if self.strict_tail_values:
            mask = attention_mask
            if isinstance(mask, dict):
                mask = mask.get("full_attention")
            row["materialized_attention_mask_nbytes"] = (
                0 if mask is None else mask.numel() * mask.element_size()
            )
            row["mask_validation_host_syncs"] = 1 if mask is not None else 0
        else:
            row["materialized_attention_mask_nbytes"] = 0
            row["mask_validation_host_syncs"] = 0
        self.calls.append(row)
        return result

    def verify_complete(self) -> dict[str, Any]:
        expected = {
            index: self.expected_calls_per_layer for index in self.indices
        }
        actual = {index: self.counts[index] for index in self.indices}
        if actual != expected or set(self.counts) != set(self.indices):
            raise QComemFairControlError(
                f"same-kernel intercept incomplete: expected={expected}, actual={actual}"
            )
        return {
            "verified": True,
            "fair_protocol": FAIR_PROTOCOL,
            "request_policy": self.request_policy,
            "kernel_mode": KERNEL_MODE,
            "same_unified_attention_kernel": True,
            "kernel_identity": dict(self.kernel_identity),
            "expected_layer_indices": self.indices,
            "counts": actual,
            "total_calls": sum(actual.values()),
            "dense_fallback_calls": 0,
            "full_kv_concatenations": 0,
            "mask_contract": self.mask_contract,
            "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
            "materialized_attention_mask_nbytes": sum(
                int(row["materialized_attention_mask_nbytes"])
                for row in self.calls
            ),
            "mask_validation_host_syncs": sum(
                int(row["mask_validation_host_syncs"]) for row in self.calls
            ),
            "position_ids_validation_host_syncs": sum(
                int(row["position_ids_validation_host_syncs"])
                for row in self.calls
            ),
            "calls": tuple(self.calls),
        }


def register_qwen35_fair_backend(ledger: Qwen35FairHitLedger) -> str:
    """Register one policy while retaining the audited TF5.14 mask seam."""

    return register_qwen35_vllm_q16_backend(ledger).name


__all__ = [
    "FAIR_CONFIGS",
    "FAIR_PROTOCOL",
    "FRESH_CONTROL",
    "PRODUCTION_MASK_CONTRACT",
    "SHARED_REUSE",
    "FreshQ16Materialization",
    "QComemFairControlError",
    "Qwen35FairHitLedger",
    "build_same_kernel_q16_sequence_pair",
    "full_attention_storage_breakdown",
    "linear_gdn_shared_base_contract",
    "snapshot_linear_gdn_state",
    "materialize_fresh_q16_request_layer",
    "materialize_qwen35_fresh_full_copy_request",
    "register_qwen35_fair_backend",
    "storage_inventory",
    "storage_residency",
    "strict_logical_layout_parity",
    "verify_linear_gdn_state_parity",
]
