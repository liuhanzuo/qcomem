from __future__ import annotations

"""Q16 block-pool adapter for vLLM's fused Triton paged attention.

This module is intentionally independent from the existing Python two-pass
reference.  The reference stores a tuple of arbitrary-sized pages and is useful
as an oracle, while the vLLM kernel requires one fixed-size physical block pool
plus a logical block table.  ``Q16PagedArena`` owns that production-shaped
layout and ``Q16PagedSequence`` gives each request private append blocks while
sharing immutable document blocks.

The only full-document copy happens once at document preparation.  Forking a
request copies a small block table.  If the document ends in a partial block,
the first append copies only that one partial block into a request-private
physical block before writing query K/V.  No request path concatenates the
complete document K/V tensor.

The fused call is resolved lazily so CPU unit tests and Apple-side development
do not require vLLM.  The audited deployment environment is recorded below;
GPU execution still has to pass an H20 capability/parity gate before any
latency claim is allowed.
"""

import importlib.metadata
import importlib.util
import math
import threading
from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch


try:  # Keep CPU tests runnable without Transformers installed.
    from transformers.cache_utils import CacheLayerMixin as _CacheLayerMixin
except ModuleNotFoundError as error:  # pragma: no cover - local Mac environment.
    if error.name != "transformers":
        raise

    class _CacheLayerMixin:  # type: ignore[no-redef]
        supports_early_init = False
        is_compileable = False

        def __init__(self, **kwargs: Any) -> None:
            del kwargs


KERNEL_MODE = "vllm_0_26_triton_unified_attention_q16_block_pool"
AUDITED_PACKAGES = {
    "torch": "2.11.0+cu129",
    "transformers": "5.14.1",
    "vllm": "0.26.0+cu129",
    "flashinfer-python": "0.6.14",
    "triton": "3.6.0",
}
QWEN35_AUDITED_GEOMETRY = {
    "num_query_heads": 16,
    "num_key_value_heads": 2,
    "num_key_value_groups": 8,
    "head_dim": 256,
    "full_attention_layers": 10,
}


class QComemPagedKernelError(RuntimeError):
    """Raised when a fused-kernel assumption is not satisfied."""


def audit_frozen_kernel_environment() -> dict[str, Any]:
    """Read package metadata without importing CUDA/platform initializers."""

    observed: dict[str, str | None] = {}
    locations: dict[str, str | None] = {}
    package_to_module = {
        "torch": "torch",
        "transformers": "transformers",
        "vllm": "vllm",
        "flashinfer-python": "flashinfer",
        "triton": "triton",
    }
    for package, module in package_to_module.items():
        try:
            observed[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            observed[package] = None
        spec = importlib.util.find_spec(module)
        locations[package] = None if spec is None else spec.origin
    mismatches = {
        package: {"expected": expected, "actual": observed[package]}
        for package, expected in AUDITED_PACKAGES.items()
        if observed[package] != expected
    }
    return {
        "expected_versions": dict(AUDITED_PACKAGES),
        "observed_versions": observed,
        "module_locations": locations,
        "matches_frozen_environment": not mismatches,
        "mismatches": mismatches,
        "kernel_entrypoint": (
            "vllm.v1.attention.ops.triton_unified_attention.unified_attention"
        ),
        "kernel_mode": KERNEL_MODE,
    }


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


@dataclass(frozen=True)
class Q16PagedArenaAudit:
    batch_size: int
    document_length: int
    page_size: int
    document_blocks_per_sequence: int
    private_blocks_per_sequence: int
    max_forks: int
    physical_blocks: int
    document_payload_nbytes: int
    allocated_pool_nbytes: int


class Q16PagedArena:
    """Fixed Q16 K/V block pool shared by document and request sequences.

    Physical layout is the NHD layout consumed directly by vLLM Triton:
    ``[physical_blocks, page_size, kv_heads, head_dim]`` for both K and V.
    """

    def __init__(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        document_block_table: torch.Tensor,
        *,
        document_length: int,
        max_append_tokens: int,
        max_forks: int,
        private_block_reservations: torch.Tensor,
    ) -> None:
        if key_cache.ndim != 4 or value_cache.ndim != 4:
            raise QComemPagedKernelError("K/V block pools must be rank four")
        if key_cache.shape != value_cache.shape:
            raise QComemPagedKernelError("Q16 fused path requires matching K/V shapes")
        if key_cache.dtype != value_cache.dtype or not key_cache.is_floating_point():
            raise QComemPagedKernelError("Q16 fused path requires floating K/V dtype")
        if key_cache.device != value_cache.device:
            raise QComemPagedKernelError("K/V block pools must share a device")
        if document_block_table.ndim != 2 or document_block_table.dtype != torch.int32:
            raise QComemPagedKernelError("document block table must be int32 [batch, blocks]")
        if document_block_table.device != key_cache.device:
            raise QComemPagedKernelError("block table and K/V pool must share a device")
        if private_block_reservations.ndim != 3:
            raise QComemPagedKernelError(
                "private reservations must be [fork, batch, private_blocks]"
            )
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.document_block_table = document_block_table
        self.document_length = int(document_length)
        self.max_append_tokens = int(max_append_tokens)
        self.max_forks = int(max_forks)
        self.private_block_reservations = private_block_reservations
        self.page_size = int(key_cache.shape[1])
        self.num_key_value_heads = int(key_cache.shape[2])
        self.head_dim = int(key_cache.shape[3])
        self.batch_size = int(document_block_table.shape[0])
        self.document_blocks_per_sequence = int(document_block_table.shape[1])
        self.private_blocks_per_sequence = int(private_block_reservations.shape[2])
        self._fork_cursor = 0
        self._fork_lock = threading.Lock()

    @classmethod
    def from_dense_document(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_size: int,
        max_append_tokens: int,
        max_forks: int,
    ) -> "Q16PagedArena":
        if key.ndim != 4 or value.ndim != 4:
            raise QComemPagedKernelError(
                "dense document K/V must be [batch, kv_heads, tokens, head_dim]"
            )
        if key.shape != value.shape:
            raise QComemPagedKernelError("dense document K/V shapes must match")
        if key.dtype != value.dtype or key.device != value.device:
            raise QComemPagedKernelError("dense document K/V dtype/device must match")
        if not key.is_floating_point():
            raise QComemPagedKernelError("Q16 document K/V must be floating point")
        if isinstance(page_size, bool) or page_size < 16 or page_size % 16:
            raise QComemPagedKernelError(
                "vLLM Triton block size must be a positive multiple of 16"
            )
        if isinstance(max_append_tokens, bool) or max_append_tokens < 1:
            raise QComemPagedKernelError("max_append_tokens must be positive")
        if isinstance(max_forks, bool) or max_forks < 1:
            raise QComemPagedKernelError("max_forks must be positive")
        batch, kv_heads, document_length, head_dim = map(int, key.shape)
        if document_length < 1:
            raise QComemPagedKernelError("document cache must contain at least one token")
        document_blocks = math.ceil(document_length / page_size)
        tail = document_length % page_size
        # A partial immutable document block must be cloned for each request;
        # the clone also accommodates appended tokens until it fills.
        private_blocks = math.ceil(
            ((tail if tail else 0) + max_append_tokens) / page_size
        )
        document_physical_blocks = batch * document_blocks
        private_physical_blocks = max_forks * batch * private_blocks
        physical_blocks = document_physical_blocks + private_physical_blocks
        key_cache = torch.empty(
            (physical_blocks, page_size, kv_heads, head_dim),
            device=key.device,
            dtype=key.dtype,
        )
        value_cache = torch.empty_like(key_cache)
        # The immutable document allocation includes the unused suffix of a
        # partial final page.  Initialize those bytes deterministically before
        # packing valid tokens so a full-physical-block immutability witness is
        # reproducible across independent arena builds.  Private request pages
        # remain outside this common document-pack operation.
        key_cache[:document_physical_blocks].zero_()
        value_cache[:document_physical_blocks].zero_()
        document_block_table = torch.arange(
            document_physical_blocks,
            device=key.device,
            dtype=torch.int32,
        ).reshape(batch, document_blocks)
        for batch_index in range(batch):
            for logical_block in range(document_blocks):
                start = logical_block * page_size
                end = min(start + page_size, document_length)
                # Physical document ids are deterministic; avoid synchronizing
                # on a CUDA scalar during the one-time pack loop.
                physical_block = batch_index * document_blocks + logical_block
                key_cache[physical_block, : end - start].copy_(
                    key[batch_index, :, start:end, :].permute(1, 0, 2)
                )
                value_cache[physical_block, : end - start].copy_(
                    value[batch_index, :, start:end, :].permute(1, 0, 2)
                )
        # Allocation decisions stay on CPU.  Reading a CUDA block id with
        # ``int(cuda_tensor)`` would synchronize every decode layer/step.
        # Only the finished block table lives on the accelerator.
        private_block_reservations = torch.arange(
            document_physical_blocks,
            physical_blocks,
            device="cpu",
            dtype=torch.int64,
        ).reshape(max_forks, batch, private_blocks)
        return cls(
            key_cache,
            value_cache,
            document_block_table,
            document_length=document_length,
            max_append_tokens=max_append_tokens,
            max_forks=max_forks,
            private_block_reservations=private_block_reservations,
        )

    @property
    def allocated_pool_nbytes(self) -> int:
        return (
            self.key_cache.untyped_storage().nbytes()
            + self.value_cache.untyped_storage().nbytes()
        )

    @property
    def storage_keys(self) -> frozenset[tuple[str, int, int]]:
        return frozenset((_storage_key(self.key_cache), _storage_key(self.value_cache)))

    @property
    def audit(self) -> Q16PagedArenaAudit:
        document_elements = (
            2
            * self.batch_size
            * self.document_length
            * self.num_key_value_heads
            * self.head_dim
        )
        return Q16PagedArenaAudit(
            batch_size=self.batch_size,
            document_length=self.document_length,
            page_size=self.page_size,
            document_blocks_per_sequence=self.document_blocks_per_sequence,
            private_blocks_per_sequence=self.private_blocks_per_sequence,
            max_forks=self.max_forks,
            physical_blocks=int(self.key_cache.shape[0]),
            document_payload_nbytes=document_elements * self.key_cache.element_size(),
            allocated_pool_nbytes=self.allocated_pool_nbytes,
        )

    def fork(self, *, strict_mask_check: bool = True) -> "Q16PagedSequence":
        with self._fork_lock:
            fork_index = self._fork_cursor
            if fork_index >= self.max_forks:
                raise QComemPagedKernelError(
                    "paged arena exhausted its preallocated request reservations"
                )
            self._fork_cursor += 1
        reservations = self.private_block_reservations[fork_index]
        max_logical_blocks = self.document_blocks_per_sequence + self.private_blocks_per_sequence
        block_table = torch.full(
            (self.batch_size, max_logical_blocks),
            -1,
            device=self.key_cache.device,
            dtype=torch.int32,
        )
        block_table[:, : self.document_blocks_per_sequence].copy_(
            self.document_block_table
        )
        return Q16PagedSequence(
            self,
            block_table,
            reservations,
            strict_mask_check=strict_mask_check,
        )


class Q16PagedSequence:
    """One request's logical block table and append cursor."""

    def __init__(
        self,
        arena: Q16PagedArena,
        block_table: torch.Tensor,
        reservations: torch.Tensor,
        *,
        strict_mask_check: bool,
    ) -> None:
        self.arena = arena
        self.block_table = block_table
        self.reservations = reservations
        self.sequence_length = arena.document_length
        self.appended_tokens = 0
        self._next_private = [0] * arena.batch_size
        self._tail_detached = [False] * arena.batch_size
        self.strict_mask_check = bool(strict_mask_check)
        self.partial_tail_staging_copy_nbytes = 0
        # Optional audit-only hook.  It is disabled in production memory
        # cells; an oracle cell may clone the dense append K/V before any
        # paged write, then bind that capture ID to the following kernel call.
        self.append_observer: Callable[[dict[str, Any]], str] | None = None
        self.last_append_capture_id: str | None = None
        self.last_append_audit: dict[str, Any] | None = None
        self._append_event_count = 0
        self._logical_physical = [
            [
                batch_index * arena.document_blocks_per_sequence + logical_block
                for logical_block in range(arena.document_blocks_per_sequence)
            ]
            + [-1] * arena.private_blocks_per_sequence
            for batch_index in range(arena.batch_size)
        ]

    @property
    def logical_block_count(self) -> int:
        return math.ceil(self.sequence_length / self.arena.page_size)

    @property
    def active_block_table(self) -> torch.Tensor:
        return self.block_table[:, : self.logical_block_count]

    @property
    def storage_keys(self) -> frozenset[tuple[str, int, int]]:
        return self.arena.storage_keys

    def _take_private_block(self, batch_index: int) -> int:
        cursor = self._next_private[batch_index]
        if cursor >= self.arena.private_blocks_per_sequence:
            raise QComemPagedKernelError(
                "request exceeded its preallocated private block reservation"
            )
        self._next_private[batch_index] += 1
        return int(self.reservations[batch_index, cursor])

    def _detach_partial_document_tail(self, batch_index: int) -> None:
        tail = self.arena.document_length % self.arena.page_size
        if tail == 0 or self._tail_detached[batch_index]:
            return
        logical_block = self.arena.document_blocks_per_sequence - 1
        source = (
            batch_index * self.arena.document_blocks_per_sequence + logical_block
        )
        target = self._take_private_block(batch_index)
        self.arena.key_cache[target, :tail].copy_(self.arena.key_cache[source, :tail])
        self.arena.value_cache[target, :tail].copy_(
            self.arena.value_cache[source, :tail]
        )
        self.block_table[batch_index, logical_block] = target
        self._logical_physical[batch_index][logical_block] = target
        self._tail_detached[batch_index] = True
        self.partial_tail_staging_copy_nbytes += (
            2
            * tail
            * self.arena.num_key_value_heads
            * self.arena.head_dim
            * self.arena.key_cache.element_size()
        )

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        arena = self.arena
        expected_prefix = (arena.batch_size, arena.num_key_value_heads)
        if key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
            raise QComemPagedKernelError("appended K/V must have one matching rank-four shape")
        if tuple(key.shape[:2]) != expected_prefix or key.shape[-1] != arena.head_dim:
            raise QComemPagedKernelError("appended K/V geometry differs from the arena")
        if key.dtype != arena.key_cache.dtype or value.dtype != arena.value_cache.dtype:
            raise QComemPagedKernelError("appended K/V dtype differs from the arena")
        if key.device != arena.key_cache.device or value.device != arena.value_cache.device:
            raise QComemPagedKernelError("appended K/V device differs from the arena")
        incoming = int(key.shape[-2])
        if incoming < 1:
            return
        if self.appended_tokens + incoming > arena.max_append_tokens:
            raise QComemPagedKernelError(
                "request exceeded max_append_tokens reserved at document preparation"
            )
        append_capture_id: str | None = None
        if self.append_observer is not None:
            # Oracle-only instrumentation receives isolated CPU clones, not
            # the live sequence or the tensors that will be written below.
            append_capture_id = self.append_observer(
                {
                    "key_states": key.detach().contiguous().cpu().clone(),
                    "value_states": value.detach().contiguous().cpu().clone(),
                    "append_event_index": self._append_event_count,
                    "appended_tokens_before": self.appended_tokens,
                    "appended_tokens_after": self.appended_tokens + incoming,
                    "sequence_length_before": self.sequence_length,
                    "sequence_length_after": self.sequence_length + incoming,
                    "source_device": str(key.device),
                    "source_dtype": str(key.dtype),
                    "source_shape": list(key.shape),
                }
            )
            if not isinstance(append_capture_id, str) or not append_capture_id:
                raise QComemPagedKernelError(
                    "append observer must return one non-empty capture ID"
                )
        for batch_index in range(arena.batch_size):
            self._detach_partial_document_tail(batch_index)
            source_offset = 0
            while source_offset < incoming:
                absolute_position = self.sequence_length + source_offset
                logical_block = absolute_position // arena.page_size
                offset_in_block = absolute_position % arena.page_size
                physical_block = self._logical_physical[batch_index][logical_block]
                if physical_block < 0:
                    physical_block = self._take_private_block(batch_index)
                    self._logical_physical[batch_index][logical_block] = physical_block
                    self.block_table[batch_index, logical_block] = physical_block
                count = min(arena.page_size - offset_in_block, incoming - source_offset)
                end = source_offset + count
                arena.key_cache[
                    physical_block, offset_in_block : offset_in_block + count
                ].copy_(key[batch_index, :, source_offset:end, :].permute(1, 0, 2))
                arena.value_cache[
                    physical_block, offset_in_block : offset_in_block + count
                ].copy_(value[batch_index, :, source_offset:end, :].permute(1, 0, 2))
                source_offset = end
        self.sequence_length += incoming
        self.appended_tokens += incoming
        self.last_append_capture_id = append_capture_id
        self.last_append_audit = {
            "append_event_index": self._append_event_count,
            "append_tokens": incoming,
            "appended_tokens_before": self.appended_tokens - incoming,
            "appended_tokens_after": self.appended_tokens,
            "sequence_length_before": self.sequence_length - incoming,
            "sequence_length_after": self.sequence_length,
            "capture_id": append_capture_id,
        }
        self._append_event_count += 1


@dataclass(frozen=True)
class Q16KernelPagedTensorView:
    sequence: Q16PagedSequence
    kind: Literal["key", "value"]

    @property
    def shape(self) -> torch.Size:
        arena = self.sequence.arena
        return torch.Size(
            (
                arena.batch_size,
                arena.num_key_value_heads,
                self.sequence.sequence_length,
                arena.head_dim,
            )
        )


class Q16KernelPagedLayer(_CacheLayerMixin):
    """Transformers cache-layer facade over one request sequence."""

    is_sliding = False
    is_compileable = False
    supports_early_init = False

    def __init__(self, sequence: Q16PagedSequence) -> None:
        super().__init__()
        self.sequence = sequence
        self.keys = Q16KernelPagedTensorView(sequence, "key")
        self.values = Q16KernelPagedTensorView(sequence, "value")
        self.is_initialized = True
        self.dtype = sequence.arena.key_cache.dtype
        self.device = sequence.arena.key_cache.device

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        del key_states, value_states
        raise QComemPagedKernelError(
            "Q16 request layers are constructed from an initialized document arena"
        )

    @classmethod
    def from_dense_document(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_size: int,
        max_append_tokens: int,
        max_forks: int = 1,
        strict_mask_check: bool = True,
    ) -> "Q16KernelPagedLayer":
        arena = Q16PagedArena.from_dense_document(
            key,
            value,
            page_size=page_size,
            max_append_tokens=max_append_tokens,
            max_forks=max_forks,
        )
        return cls(arena.fork(strict_mask_check=strict_mask_check))

    @property
    def stored_nbytes(self) -> int:
        return self.sequence.arena.allocated_pool_nbytes

    @property
    def storage_keys(self) -> frozenset[tuple[str, int, int]]:
        return self.sequence.storage_keys

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Q16KernelPagedTensorView, Q16KernelPagedTensorView]:
        del args, kwargs
        self.sequence.append(key_states, value_states)
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self.sequence.sequence_length

    def get_max_length(self) -> int:
        return self.sequence.arena.document_length + self.sequence.arena.max_append_tokens

    def fork(self, *, strict_mask_check: bool | None = None) -> "Q16KernelPagedLayer":
        if strict_mask_check is None:
            strict_mask_check = self.sequence.strict_mask_check
        return Q16KernelPagedLayer(
            self.sequence.arena.fork(strict_mask_check=strict_mask_check)
        )


class Q16KernelPagedDocumentLayer(_CacheLayerMixin):
    """Immutable document facade; only ``fork`` may create writable requests."""

    is_sliding = False
    is_compileable = False
    supports_early_init = False

    def __init__(self, arena: Q16PagedArena, *, strict_mask_check: bool = True) -> None:
        super().__init__()
        self.arena = arena
        self.strict_mask_check = bool(strict_mask_check)
        self.is_initialized = True
        self.dtype = arena.key_cache.dtype
        self.device = arena.key_cache.device

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        del key_states, value_states
        raise QComemPagedKernelError(
            "Q16 document layers are constructed from initialized dense K/V"
        )

    @classmethod
    def from_dense_document(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_size: int,
        max_append_tokens: int,
        max_request_forks: int,
        strict_mask_check: bool = True,
    ) -> "Q16KernelPagedDocumentLayer":
        return cls(
            Q16PagedArena.from_dense_document(
                key,
                value,
                page_size=page_size,
                max_append_tokens=max_append_tokens,
                max_forks=max_request_forks,
            ),
            strict_mask_check=strict_mask_check,
        )

    @property
    def stored_nbytes(self) -> int:
        return self.arena.allocated_pool_nbytes

    @property
    def storage_keys(self) -> frozenset[tuple[str, int, int]]:
        return self.arena.storage_keys

    def get_seq_length(self) -> int:
        return self.arena.document_length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self.get_seq_length() + query_length, 0

    def get_max_length(self) -> int:
        return self.arena.document_length + self.arena.max_append_tokens

    def update(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise QComemPagedKernelError(
            "immutable document layer must be forked before query append"
        )

    def fork(self) -> Q16KernelPagedLayer:
        return Q16KernelPagedLayer(
            self.arena.fork(strict_mask_check=self.strict_mask_check)
        )


def _paired_sequence(
    key: Q16KernelPagedTensorView,
    value: Q16KernelPagedTensorView,
) -> Q16PagedSequence:
    if not isinstance(key, Q16KernelPagedTensorView) or not isinstance(
        value, Q16KernelPagedTensorView
    ):
        raise QComemPagedKernelError("fused backend requires paired Q16 paged views")
    if key.kind != "key" or value.kind != "value" or key.sequence is not value.sequence:
        raise QComemPagedKernelError("K/V views do not refer to the same request")
    return key.sequence


def _extract_attention_mask(
    attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
) -> torch.Tensor | None:
    if isinstance(attention_mask, dict):
        if "full_attention" not in attention_mask:
            raise QComemPagedKernelError(
                "attention-mask mapping has no full_attention entry"
            )
        attention_mask = attention_mask["full_attention"]
    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
        raise QComemPagedKernelError("attention mask must be a tensor, mapping, or None")
    return attention_mask


def validate_canonical_tail_causal_mask(
    attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
    *,
    batch_size: int,
    query_length: int,
    total_length: int,
    device: torch.device,
) -> None:
    """Fail closed if a supplied mask contains padding or custom bias.

    Qwen3.5 applies RoPE before this backend, so the fused kernel uses
    ``pos_encoding_mode=NONE``.  It may therefore replace only the ordinary
    no-padding tail-causal mask, not an arbitrary additive attention bias.
    """

    mask = _extract_attention_mask(attention_mask)
    if mask is None:
        return
    if mask.ndim != 4 or mask.shape[0] not in (1, batch_size):
        raise QComemPagedKernelError(
            "fused Q16 path expects [batch|1, heads|1, query, key] mask"
        )
    if mask.shape[-2:] != (query_length, total_length):
        raise QComemPagedKernelError("attention mask length differs from Q/KV length")
    past_length = total_length - query_length
    query_positions = torch.arange(query_length, device=device) + past_length
    key_positions = torch.arange(total_length, device=device)
    allowed = key_positions.view(1, -1) <= query_positions.view(-1, 1)
    expanded_allowed = allowed.view(1, 1, query_length, total_length)
    if mask.dtype == torch.bool:
        valid = mask == expanded_allowed
    else:
        # Transformers eager masks use zero for visible entries and -inf or
        # dtype.min for hidden entries.  Any other bias/padding fails closed.
        visible_ok = torch.where(expanded_allowed, mask == 0, torch.ones_like(mask, dtype=torch.bool))
        hidden_ok = torch.where(
            expanded_allowed,
            torch.ones_like(mask, dtype=torch.bool),
            torch.isneginf(mask) | (mask <= -1.0e4),
        )
        valid = visible_ok & hidden_ok
    if not bool(valid.all().item()):
        raise QComemPagedKernelError(
            "vLLM fused backend cannot replace this non-canonical attention mask"
        )


def _resolve_vllm_unified_attention() -> Callable[..., Any]:
    try:
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )
    except ImportError as error:  # pragma: no cover - requires frozen GPU env.
        raise QComemPagedKernelError(
            "vLLM Triton unified_attention is unavailable in this environment"
        ) from error
    return unified_attention


def _run_vllm_unified_attention(
    *,
    kernel: Callable[..., Any],
    query: torch.Tensor,
    sequence: Q16PagedSequence,
    scale: float,
) -> torch.Tensor:
    arena = sequence.arena
    batch, query_heads, query_length, head_dim = map(int, query.shape)
    q = query.transpose(1, 2).contiguous().reshape(
        batch * query_length, query_heads, head_dim
    )
    output = torch.empty_like(q)
    cu_seqlens_q = torch.arange(
        0,
        (batch + 1) * query_length,
        query_length,
        dtype=torch.int32,
        device=query.device,
    )
    seq_lens = torch.full(
        (batch,),
        sequence.sequence_length,
        dtype=torch.int32,
        device=query.device,
    )
    kernel(
        q=q,
        k=arena.key_cache,
        v=arena.value_cache,
        out=output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=query_length,
        seqused_k=seq_lens,
        max_seqlen_k=sequence.sequence_length,
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1),
        block_table=sequence.active_block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    return output.reshape(batch, query_length, query_heads, head_dim)


def vllm_triton_q16_paged_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: Q16KernelPagedTensorView,
    value: Q16KernelPagedTensorView,
    attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Transformers attention-interface entrypoint backed by one GPU kernel."""

    if dropout != 0.0:
        raise QComemPagedKernelError("fused paged path is inference-only (dropout=0)")
    if query.ndim != 4:
        raise QComemPagedKernelError("query must be [batch, query_heads, tokens, dim]")
    if not bool(getattr(module, "is_causal", False)):
        raise QComemPagedKernelError("fused Q16 adapter currently requires causal attention")
    sequence = _paired_sequence(key, value)
    arena = sequence.arena
    batch, query_heads, query_length, head_dim = map(int, query.shape)
    if (batch, head_dim) != (arena.batch_size, arena.head_dim):
        raise QComemPagedKernelError("query batch/head_dim differs from paged arena")
    if query.dtype != arena.key_cache.dtype or query.device != arena.key_cache.device:
        raise QComemPagedKernelError("query dtype/device differs from paged arena")
    if query_heads % arena.num_key_value_heads:
        raise QComemPagedKernelError("query heads must be divisible by KV heads")
    groups = query_heads // arena.num_key_value_heads
    if int(getattr(module, "num_key_value_groups", groups)) != groups:
        raise QComemPagedKernelError("module GQA grouping differs from Q/KV geometry")
    if query_length > sequence.appended_tokens:
        raise QComemPagedKernelError(
            "current query chunk must already have been appended to paged K/V"
        )
    if sequence.strict_mask_check:
        validate_canonical_tail_causal_mask(
            attention_mask,
            batch_size=batch,
            query_length=query_length,
            total_length=sequence.sequence_length,
            device=query.device,
        )
    # Transformers 5.14 forwards this model-level control kwarg through the
    # Qwen3.5 decoder and attention module. Cache update has already happened;
    # it does not alter attention math, but False would contradict this path.
    use_cache = kwargs.pop("use_cache", None)
    if use_cache is not None and use_cache is not True:
        raise QComemPagedKernelError("fused paged attention requires use_cache=True")
    audit = kwargs.pop("audit", None)
    kernel = kwargs.pop("_kernel", None)
    if kwargs:
        # Do not silently drop future attention features such as sinks/softcap.
        unsupported = sorted(kwargs)
        raise QComemPagedKernelError(
            f"unsupported fused-attention keyword arguments: {unsupported}"
        )
    if kernel is None:
        kernel = _resolve_vllm_unified_attention()
    scale = float(scaling) if scaling is not None else float(
        getattr(module, "scaling", head_dim**-0.5)
    )
    output = _run_vllm_unified_attention(
        kernel=kernel,
        query=query,
        sequence=sequence,
        scale=scale,
    )
    if audit is not None:
        if not isinstance(audit, dict):
            raise QComemPagedKernelError("audit must be a mutable dictionary")
        audit.update(
            {
                "kernel_mode": KERNEL_MODE,
                "fused_gpu_kernel_calls": 1,
                "full_kv_concatenations": 0,
                "full_document_staging_copy_nbytes": 0,
                "partial_tail_staging_copy_nbytes": (
                    sequence.partial_tail_staging_copy_nbytes
                ),
                "physical_block_pool_shape": tuple(arena.key_cache.shape),
                "active_block_table_shape": tuple(sequence.active_block_table.shape),
                "query_tokens": query_length,
                "kv_tokens": sequence.sequence_length,
                "softmax_scale": scale,
                "gqa_groups": groups,
                "quantization": "Q16",
            }
        )
    return output, None


__all__ = [
    "AUDITED_PACKAGES",
    "KERNEL_MODE",
    "QWEN35_AUDITED_GEOMETRY",
    "Q16KernelPagedLayer",
    "Q16KernelPagedDocumentLayer",
    "Q16KernelPagedTensorView",
    "Q16PagedArena",
    "Q16PagedArenaAudit",
    "Q16PagedSequence",
    "QComemPagedKernelError",
    "audit_frozen_kernel_environment",
    "validate_canonical_tail_causal_mask",
    "vllm_triton_q16_paged_attention_forward",
]
