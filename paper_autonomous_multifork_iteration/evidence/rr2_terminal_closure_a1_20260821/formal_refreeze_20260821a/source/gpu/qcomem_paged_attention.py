from __future__ import annotations

"""Page-wise full attention reference for QCoMem replay caches.

The production ``DynamicLayer.update`` contract returns one contiguous K/V
tensor.  That forces a ``torch.cat`` of the immutable document prefix and the
request-local suffix before every full-attention call.  This module keeps the
same outer cache update contract, but returns lightweight page views and uses
an online softmax to consume them without materialising the complete dense KV
sequence.

This is deliberately a correctness/reference implementation.  It supports
inference (dropout == 0), non-sliding self attention, additive/bool masks, and
GQA.  It does not implement beam reordering, cache cropping, offload, or a
fused accelerator kernel.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, Literal, Union

import torch

from qcomem_torch import PackedTensor, quantize_tensor


try:  # Keep the tiny reference tests runnable without Transformers installed.
    from transformers.cache_utils import CacheLayerMixin as _CacheLayerMixin
except ModuleNotFoundError as error:  # pragma: no cover - local Mac environment.
    if error.name != "transformers":
        raise

    class _CacheLayerMixin:  # type: ignore[no-redef]
        supports_early_init = False
        is_compileable = False

        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.keys = None
            self.values = None
            self.is_initialized = False


Payload = Union[torch.Tensor, PackedTensor]


def _payload_shape(payload: Payload) -> tuple[int, ...]:
    if isinstance(payload, torch.Tensor):
        return tuple(payload.shape)
    return payload.original_shape


def _payload_dtype(payload: Payload) -> torch.dtype:
    if isinstance(payload, torch.Tensor):
        return payload.dtype
    return payload.original_dtype


def _payload_tensors(payload: Payload) -> Iterator[torch.Tensor]:
    if isinstance(payload, torch.Tensor):
        yield payload
        return
    yield payload.data
    if payload.scales is not None:
        yield payload.scales
    if payload.biases is not None:
        yield payload.biases


def _unique_storage_nbytes(tensors: Iterable[torch.Tensor]) -> int:
    seen: set[tuple[str, int, int]] = set()
    total = 0
    for tensor in tensors:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            total += storage.nbytes()
    return total


def _storage_keys(tensors: Iterable[torch.Tensor]) -> frozenset[tuple[str, int, int]]:
    keys = []
    for tensor in tensors:
        storage = tensor.untyped_storage()
        keys.append((str(tensor.device), storage.data_ptr(), storage.nbytes()))
    return frozenset(keys)


def _materialize_payload(
    payload: Payload,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    dense = payload if isinstance(payload, torch.Tensor) else payload.dequantize()
    return dense.to(device=device, dtype=dtype)


@dataclass(frozen=True)
class KVPage:
    """One K/V page in native KV-head layout ``[B, Hkv, S, D]``."""

    key: Payload
    value: Payload

    def __post_init__(self) -> None:
        key_shape = _payload_shape(self.key)
        value_shape = _payload_shape(self.value)
        if len(key_shape) != 4 or len(value_shape) != 4:
            raise ValueError("page key/value must have shape [batch, kv_heads, tokens, dim]")
        if key_shape[:3] != value_shape[:3]:
            raise ValueError("page key/value batch, head, and token dimensions must match")
        if key_shape[-2] < 1:
            raise ValueError("a KV page must contain at least one token")

    @property
    def batch_size(self) -> int:
        return _payload_shape(self.key)[0]

    @property
    def num_key_value_heads(self) -> int:
        return _payload_shape(self.key)[1]

    @property
    def length(self) -> int:
        return _payload_shape(self.key)[-2]

    @property
    def key_head_dim(self) -> int:
        return _payload_shape(self.key)[-1]

    @property
    def value_head_dim(self) -> int:
        return _payload_shape(self.value)[-1]

    @property
    def is_packed(self) -> bool:
        return not isinstance(self.key, torch.Tensor) or not isinstance(
            self.value, torch.Tensor
        )

    @property
    def dense_nbytes(self) -> int:
        key_elements = 1
        for size in _payload_shape(self.key):
            key_elements *= size
        value_elements = 1
        for size in _payload_shape(self.value):
            value_elements *= size
        return (
            key_elements * torch.empty((), dtype=_payload_dtype(self.key)).element_size()
            + value_elements
            * torch.empty((), dtype=_payload_dtype(self.value)).element_size()
        )

    @property
    def stored_nbytes(self) -> int:
        return _unique_storage_nbytes(
            [*_payload_tensors(self.key), *_payload_tensors(self.value)]
        )

    @property
    def storage_keys(self) -> frozenset[tuple[str, int, int]]:
        return _storage_keys([*_payload_tensors(self.key), *_payload_tensors(self.value)])

    def materialize(
        self, *, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            _materialize_payload(self.key, device=device, dtype=dtype),
            _materialize_payload(self.value, device=device, dtype=dtype),
        )


def _make_pages(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    page_size: int,
    bits: int | None,
    group_size: int,
) -> tuple[KVPage, ...]:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("key/value must have shape [batch, kv_heads, tokens, dim]")
    if key.shape[:3] != value.shape[:3]:
        raise ValueError("key/value batch, head, and token dimensions must match")
    if key.shape[-2] < 1:
        return ()
    if page_size < 1:
        raise ValueError("page_size must be positive")
    if bits is not None and bits not in (2, 4, 8, 16):
        raise ValueError("bits must be one of (2, 4, 8, 16) or None")

    pages = []
    for start in range(0, key.shape[-2], page_size):
        end = min(start + page_size, key.shape[-2])
        # A page owns its storage.  This prevents a short slice from retaining
        # the complete input allocation and makes the storage audit meaningful.
        page_key = key[..., start:end, :].detach().clone()
        page_value = value[..., start:end, :].detach().clone()
        if bits is not None:
            key_payload: Payload = quantize_tensor(
                page_key, bits=bits, group_size=group_size
            )
            value_payload: Payload = quantize_tensor(
                page_value, bits=bits, group_size=group_size
            )
        else:
            key_payload = page_key
            value_payload = page_value
        pages.append(KVPage(key_payload, value_payload))
    return tuple(pages)


class PagedKVStore:
    """Shared immutable prefix pages plus request-private appended pages."""

    def __init__(
        self,
        document_pages: Iterable[KVPage] = (),
        *,
        append_page_size: int = 1,
    ) -> None:
        if append_page_size < 1:
            raise ValueError("append_page_size must be positive")
        self.document_pages = tuple(document_pages)
        self.request_pages: list[KVPage] = []
        self.append_page_size = append_page_size
        self._validate_pages(self.document_pages)

    @classmethod
    def from_dense_document(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_size: int,
        bits: int | None = None,
        group_size: int = 64,
        append_page_size: int = 1,
    ) -> "PagedKVStore":
        return cls(
            _make_pages(
                key,
                value,
                page_size=page_size,
                bits=bits,
                group_size=group_size,
            ),
            append_page_size=append_page_size,
        )

    @staticmethod
    def _validate_pages(pages: Iterable[KVPage]) -> None:
        signature = None
        for page in pages:
            current = (
                page.batch_size,
                page.num_key_value_heads,
                page.key_head_dim,
                page.value_head_dim,
                _payload_dtype(page.key),
                _payload_dtype(page.value),
            )
            if signature is None:
                signature = current
            elif current != signature:
                raise ValueError("all KV pages must share batch/head/dim/dtype metadata")

    @property
    def pages(self) -> tuple[KVPage, ...]:
        return (*self.document_pages, *self.request_pages)

    @property
    def is_initialized(self) -> bool:
        return bool(self.document_pages or self.request_pages)

    @property
    def sequence_length(self) -> int:
        return sum(page.length for page in self.pages)

    @property
    def document_length(self) -> int:
        return sum(page.length for page in self.document_pages)

    @property
    def batch_size(self) -> int:
        return self.pages[0].batch_size if self.pages else -1

    @property
    def dtype(self) -> torch.dtype | None:
        return _payload_dtype(self.pages[0].key) if self.pages else None

    @property
    def device(self) -> torch.device | None:
        if not self.pages:
            return None
        first = next(_payload_tensors(self.pages[0].key))
        return first.device

    @property
    def stored_nbytes(self) -> int:
        tensors = []
        for page in self.pages:
            tensors.extend(_payload_tensors(page.key))
            tensors.extend(_payload_tensors(page.value))
        return _unique_storage_nbytes(tensors)

    @property
    def dense_nbytes(self) -> int:
        return sum(page.dense_nbytes for page in self.pages)

    @property
    def storage_keys(self) -> frozenset[tuple[str, int, int]]:
        tensors = []
        for page in self.pages:
            tensors.extend(_payload_tensors(page.key))
            tensors.extend(_payload_tensors(page.value))
        return _storage_keys(tensors)

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        new_pages = _make_pages(
            key,
            value,
            page_size=self.append_page_size,
            bits=None,
            group_size=1,
        )
        self._validate_pages((*self.pages, *new_pages))
        self.request_pages.extend(new_pages)

    def fork(self) -> "PagedKVStore":
        # All pages visible at the fork boundary become immutable shared pages;
        # the child gets its own empty request-page list.
        return PagedKVStore(self.pages, append_page_size=self.append_page_size)


@dataclass(frozen=True)
class PagedTensorView:
    """A non-Tensor marker passed through the Qwen attention forward."""

    store: PagedKVStore
    kind: Literal["key", "value"]

    @property
    def shape(self) -> torch.Size:
        if not self.store.pages:
            return torch.Size((0,))
        first = self.store.pages[0]
        dim = first.key_head_dim if self.kind == "key" else first.value_head_dim
        return torch.Size(
            (
                first.batch_size,
                first.num_key_value_heads,
                self.store.sequence_length,
                dim,
            )
        )


class PagedKVLayer(_CacheLayerMixin):
    """Transformers-compatible non-sliding dynamic cache layer facade."""

    is_sliding = False
    is_compileable = False
    supports_early_init = False

    def __init__(self, store: PagedKVStore) -> None:
        super().__init__()
        self.store = store
        self.keys = PagedTensorView(store, "key")
        self.values = PagedTensorView(store, "value")
        self.is_initialized = store.is_initialized
        self.dtype = store.dtype
        self.device = store.device

    @classmethod
    def from_dense_document(
        cls,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        page_size: int,
        bits: int | None = None,
        group_size: int = 64,
        append_page_size: int = 1,
    ) -> "PagedKVLayer":
        return cls(
            PagedKVStore.from_dense_document(
                key,
                value,
                page_size=page_size,
                bits=bits,
                group_size=group_size,
                append_page_size=append_page_size,
            )
        )

    @property
    def stored_nbytes(self) -> int:
        return self.store.stored_nbytes

    @property
    def dense_nbytes(self) -> int:
        return self.store.dense_nbytes

    @property
    def batch_size(self) -> int:
        return self.store.batch_size

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        # Metadata is established by the first owned appended page.
        if key_states.shape[-2] == 0:
            self.dtype = key_states.dtype
            self.device = key_states.device
            return
        self.update(key_states, value_states)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[PagedTensorView, PagedTensorView]:
        del args, kwargs
        self.store.append(key_states, value_states)
        self.is_initialized = True
        self.dtype = self.store.dtype
        self.device = self.store.device
        return self.keys, self.values

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        # Called before ``update`` by Transformers mask construction.
        return self.get_seq_length() + query_length, 0

    def get_seq_length(self) -> int:
        return self.store.sequence_length

    def get_max_length(self) -> int:
        return -1

    def fork(self) -> "PagedKVLayer":
        return PagedKVLayer(self.store.fork())

    def offload(self) -> None:
        raise NotImplementedError("the reference paged layer does not implement offload")

    def prefetch(self) -> None:
        raise NotImplementedError("the reference paged layer does not implement prefetch")

    def reset(self) -> None:
        raise NotImplementedError("the reference paged layer does not implement reset")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        del beam_idx
        raise NotImplementedError("the reference paged layer does not implement beam search")

    def crop(self, tokens_to_remove: int) -> None:
        del tokens_to_remove
        raise NotImplementedError("the reference paged layer does not implement crop")

    def batch_repeat_interleave(self, repeats: int) -> None:
        del repeats
        raise NotImplementedError("the reference paged layer does not implement batch repeat")

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        del indices
        raise NotImplementedError("the reference paged layer does not implement batch select")


def replace_dynamic_cache_layer(
    cache: Any,
    layer_idx: int,
    *,
    page_size: int,
    bits: int | None = None,
    group_size: int = 64,
    append_page_size: int = 1,
) -> PagedKVLayer:
    """Replace one initialized dense, non-sliding cache layer in-place.

    The conversion is intended for the one-time document-store preparation
    boundary.  It accepts dense ``keys``/``values`` only; an existing monolithic
    ``PackedTensor`` cannot be split without first materialising it densely.
    """

    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError("cache must expose a layers sequence")
    if layer_idx < 0 or layer_idx >= len(layers):
        raise IndexError("cache layer index is out of range")
    source = layers[layer_idx]
    if bool(getattr(source, "is_sliding", False)):
        raise ValueError("sliding attention is not supported by this reference")
    key = getattr(source, "keys", None)
    value = getattr(source, "values", None)
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise TypeError(
            "source keys/values must be dense tensors; convert before monolithic packing"
        )
    layer = PagedKVLayer.from_dense_document(
        key,
        value,
        page_size=page_size,
        bits=bits,
        group_size=group_size,
        append_page_size=append_page_size,
    )
    layers[layer_idx] = layer
    return layer


def _resolve_pages(
    key: torch.Tensor | PagedTensorView,
    value: torch.Tensor | PagedTensorView,
) -> tuple[tuple[KVPage, ...], PagedKVStore | None]:
    if isinstance(key, torch.Tensor) and isinstance(value, torch.Tensor):
        return (KVPage(key, value),), None
    if not isinstance(key, PagedTensorView) or not isinstance(
        value, PagedTensorView
    ):
        raise TypeError("key and value must both be tensors or paired paged views")
    if key.kind != "key" or value.kind != "value" or key.store is not value.store:
        raise ValueError("key/value paged views must refer to the same store")
    return key.store.pages, key.store


def _mask_page_scores(
    scores: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None,
    start: int,
    end: int,
    total_length: int,
    query_length: int,
    is_causal: bool,
) -> torch.Tensor:
    supplied_attention_mask = attention_mask is not None
    if attention_mask is not None:
        if attention_mask.shape[-1] < total_length:
            raise ValueError(
                "attention_mask key dimension is shorter than the paged KV sequence"
            )
        page_mask = attention_mask[..., start:end]
        if page_mask.dtype == torch.bool:
            scores = scores.masked_fill(~page_mask, -torch.inf)
        else:
            # Match the eager expression ``scores + attention_mask`` exactly,
            # including PyTorch's dtype promotion.  Qwen's eager mask normally
            # has the hidden-state dtype, but a caller-provided FP32 mask must
            # promote rather than be silently narrowed.
            scores = scores + page_mask.to(device=scores.device)

    if is_causal and not supplied_attention_mask:
        # The current query chunk was appended at the tail immediately before
        # attention, so query i has absolute key position past_length + i.
        past_length = total_length - query_length
        query_positions = torch.arange(
            query_length, device=scores.device
        ) + past_length
        key_positions = torch.arange(start, end, device=scores.device)
        allowed = key_positions.view(1, 1, 1, -1) <= query_positions.view(
            1, 1, -1, 1
        )
        scores = scores.masked_fill(~allowed, -torch.inf)
    return scores


def paged_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor | PagedTensorView,
    value: torch.Tensor | PagedTensorView,
    attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Online-softmax full attention over dense or independently packed pages.

    The returned layout matches the Transformers attention interface:
    ``[batch, query_tokens, query_heads, value_dim]``.  ``audit`` may be passed
    in ``kwargs`` as a mutable dictionary to record the largest page-level
    dense working set used by this call.
    """

    if dropout != 0.0:
        raise ValueError("paged reference attention supports inference dropout=0 only")
    if query.ndim != 4:
        raise ValueError("query must have shape [batch, query_heads, tokens, dim]")
    if isinstance(attention_mask, dict):
        if "full_attention" not in attention_mask:
            raise KeyError("attention mask mapping has no 'full_attention' entry")
        attention_mask = attention_mask["full_attention"]
    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a tensor, full-attention dict, or None")

    pages, store = _resolve_pages(key, value)
    if not pages:
        raise ValueError("paged attention requires at least one KV page")
    batch, query_heads, query_length, query_dim = query.shape
    first = pages[0]
    kv_heads = first.num_key_value_heads
    if batch != first.batch_size or query_dim != first.key_head_dim:
        raise ValueError("query batch/head_dim do not match the KV pages")
    if query_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads for GQA")
    groups = query_heads // kv_heads
    configured_groups = getattr(module, "num_key_value_groups", groups)
    if configured_groups != groups:
        raise ValueError("module.num_key_value_groups does not match query/KV heads")

    total_length = sum(page.length for page in pages)
    if total_length < query_length and bool(getattr(module, "is_causal", False)):
        raise ValueError("causal self attention requires KV length >= query length")
    scale = scaling
    if scale is None:
        scale = float(getattr(module, "scaling", query_dim**-0.5))

    # Keep native KV heads.  The extra G dimension maps contiguous query-head
    # groups to their one KV head without repeat_interleave storage.
    grouped_query = query.reshape(batch, kv_heads, groups, query_length, query_dim)
    value_dim = first.value_head_dim
    running_max = torch.full(
        (batch, query_heads, query_length),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    running_sum = torch.zeros_like(running_max)
    audit = kwargs.pop("audit", None)
    if audit is not None and not isinstance(audit, dict):
        raise TypeError("audit must be a mutable dictionary")
    materialized_lengths = []
    max_page_nbytes = 0
    # Pass 1 computes the global FP32 softmax normalizer without retaining a
    # full-KV score tensor.  Qwen's eager reference does *not* multiply FP32
    # softmax weights by FP32 values: it casts the normalized probabilities
    # back to query.dtype before the value matmul.  Keeping unnormalized FP32
    # page weights in the value path materially changes a BF16 model after a
    # few decoder layers, even when the top token happens to stay unchanged.
    offset = 0
    for page in pages:
        page_key, page_value = page.materialize(device=query.device, dtype=query.dtype)
        if page_key.shape[:2] != (batch, kv_heads):
            raise ValueError("all page batch/head metadata must match the query")
        page_nbytes = (
            page_key.numel() * page_key.element_size()
            + page_value.numel() * page_value.element_size()
        )
        max_page_nbytes = max(max_page_nbytes, page_nbytes)
        materialized_lengths.append(page.length)

        page_scores = torch.einsum(
            "bhgqd,bhkd->bhgqk", grouped_query, page_key
        ).reshape(batch, query_heads, query_length, page.length)
        page_scores = page_scores * scale
        page_scores = _mask_page_scores(
            page_scores,
            attention_mask=attention_mask,
            start=offset,
            end=offset + page.length,
            total_length=total_length,
            query_length=query_length,
            is_causal=bool(getattr(module, "is_causal", False)),
        ).float()

        page_max = page_scores.amax(dim=-1)
        merged_max = torch.maximum(running_max, page_max)
        finite_merged = torch.isfinite(merged_max)
        old_scale = torch.where(
            torch.isfinite(running_max) & finite_merged,
            torch.exp(running_max - merged_max),
            torch.zeros_like(merged_max),
        )
        safe_max = torch.where(finite_merged, merged_max, torch.zeros_like(merged_max))
        page_weights = torch.exp(page_scores - safe_max.unsqueeze(-1))
        page_weights = torch.where(
            finite_merged.unsqueeze(-1), page_weights, torch.zeros_like(page_weights)
        )
        page_sum = page_weights.sum(dim=-1)
        running_sum = running_sum * old_scale + page_sum
        running_max = merged_max
        offset += page.length
        # Do not retain one page's unpacked K/V while materialising the next.
        del page_key, page_value, page_scores, page_weights

    # Pass 2 reconstructs normalized probabilities one page at a time, casts
    # them to the eager path's dtype, and accumulates weight@value products in
    # FP32 before the single final cast.  FP32 partial accumulation avoids a
    # BF16 rounding at every page boundary while preserving bounded page-wise
    # materialisation and no dense KV concatenation.
    running_value = torch.zeros(
        (batch, query_heads, query_length, value_dim),
        dtype=torch.float32,
        device=query.device,
    )
    safe_sum = running_sum.clamp_min(torch.finfo(torch.float32).tiny)
    finite_rows = torch.isfinite(running_max) & (running_sum > 0)
    offset = 0
    for page in pages:
        page_key, page_value = page.materialize(device=query.device, dtype=query.dtype)
        page_scores = torch.einsum(
            "bhgqd,bhkd->bhgqk", grouped_query, page_key
        ).reshape(batch, query_heads, query_length, page.length)
        page_scores = page_scores * scale
        page_scores = _mask_page_scores(
            page_scores,
            attention_mask=attention_mask,
            start=offset,
            end=offset + page.length,
            total_length=total_length,
            query_length=query_length,
            is_causal=bool(getattr(module, "is_causal", False)),
        ).float()
        normalized = torch.exp(page_scores - running_max.unsqueeze(-1))
        normalized = torch.where(
            finite_rows.unsqueeze(-1),
            normalized / safe_sum.unsqueeze(-1),
            torch.zeros_like(normalized),
        ).to(query.dtype)
        # Cast both operands to FP32 only after quantising the probabilities to
        # query.dtype.  This models BF16 GEMM's FP32 accumulator while allowing
        # independent pages to contribute without a full value concatenation.
        page_output = torch.einsum(
            "bhgqk,bhkd->bhgqd",
            normalized.reshape(batch, kv_heads, groups, query_length, page.length).float(),
            page_value.float(),
        ).reshape(batch, query_heads, query_length, value_dim)
        running_value += page_output
        offset += page.length
        del page_key, page_value, page_scores, normalized, page_output

    output = torch.where(
        finite_rows.unsqueeze(-1), running_value, torch.zeros_like(running_value)
    ).to(query.dtype)

    if audit is not None:
        audit.update(
            {
                "page_count": len(pages),
                "materialized_page_token_lengths": tuple(materialized_lengths),
                "max_materialized_kv_tokens": max(materialized_lengths),
                "max_materialized_kv_nbytes": max_page_nbytes,
                "max_single_unpack_page_nbytes": max_page_nbytes,
                "total_kv_tokens": total_length,
                "dense_full_kv_nbytes": sum(
                    page.batch_size
                    * page.num_key_value_heads
                    * page.length
                    * (page.key_head_dim + page.value_head_dim)
                    * query.element_size()
                    for page in pages
                ),
                "persistent_paged_nbytes": (
                    store.stored_nbytes
                    if store is not None
                    else sum(page.stored_nbytes for page in pages)
                ),
                "full_kv_concatenations": 0,
                "gqa_kv_repeat_materializations": 0,
                "softmax_passes": 2,
                "normalized_weight_dtype": str(query.dtype),
                "value_accumulator_dtype": str(torch.float32),
            }
        )

    return output.transpose(1, 2).contiguous(), None


def register_transformers_paged_attention(
    name: str = "qcomem_paged_reference",
) -> str:
    """Register the reference attention and eager additive-mask builders.

    Call this before setting ``config._attn_implementation = name``.  Qwen3.5
    applies RoPE before ``cache.update`` and passes the returned objects directly
    to the selected attention interface, so no model-source patch is required.
    """

    try:
        from transformers.masking_utils import AttentionMaskInterface, eager_mask
        from transformers.modeling_utils import AttentionInterface
    except ImportError as error:  # pragma: no cover - depends on deployment env.
        raise RuntimeError("Transformers is required for interface registration") from error
    AttentionInterface.register(name, paged_attention_forward)
    AttentionMaskInterface.register(name, eager_mask)
    return name


__all__ = [
    "KVPage",
    "PagedKVLayer",
    "PagedKVStore",
    "PagedTensorView",
    "paged_attention_forward",
    "register_transformers_paged_attention",
    "replace_dynamic_cache_layer",
]
