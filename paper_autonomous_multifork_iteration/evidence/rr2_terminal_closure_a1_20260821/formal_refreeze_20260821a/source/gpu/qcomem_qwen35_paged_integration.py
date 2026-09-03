from __future__ import annotations

"""Fail-closed Qwen3.5 integration for the QCoMem paged-attention reference.

The API-audited Transformers 5.14/5.15 Qwen3.5 path applies RoPE to Q/K, calls
``past_key_values.update(k, v, layer_idx)``, then forwards the two returned
objects directly to the configured attention interface.  This module binds
that seam without patching model source:

* discover the requested full-attention layers from ``config.layer_types``;
* replace every requested dense cache layer with ``PagedKVLayer``;
* register a per-run attention backend whose ledger rejects dense fallback;
* require every requested layer to hit that backend the expected number of
  times; and
* run the exact same caller against standard mutable eager and paged caches.

Layer counts and indices are always derived from the loaded text config; no
model-layout count is hard-coded here.
"""

import inspect
import math
import re
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import torch

from qcomem_paged_attention import (
    KVPage,
    PagedKVLayer,
    PagedKVStore,
    PagedTensorView,
    paged_attention_forward,
)
from qcomem_torch import clone_cache


SUPPORTED_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5_moe_text"})
KERNEL_MODE = "reference_python_two_pass_paged_softmax"
REQUIRED_ATTENTION_FORWARD_PARAMETERS = frozenset(
    {"hidden_states", "position_embeddings", "attention_mask", "past_key_values"}
)


class Qwen35PagedIntegrationError(RuntimeError):
    """Raised when any assumption needed by the paged path is not satisfied."""


def _resolve_text_backbone(model: Any) -> Any:
    candidates = [model]
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(inner)
        language_model = getattr(inner, "language_model", None)
        if language_model is not None:
            candidates.insert(0, language_model)
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        candidates.insert(0, language_model)
    for candidate in candidates:
        if hasattr(candidate, "layers") and hasattr(candidate, "config"):
            return candidate
    raise Qwen35PagedIntegrationError(
        "cannot resolve a text backbone exposing both layers and config"
    )


def _require_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise Qwen35PagedIntegrationError(f"{label} must be an integer")
    return value


@dataclass(frozen=True)
class Qwen35FullAttentionPlan:
    model_type: str
    num_hidden_layers: int
    layer_start: int
    layer_end: int
    full_attention_layer_indices: tuple[int, ...]
    layer_types: tuple[str, ...]
    transformers_api: dict[str, Any]

    @property
    def full_attention_layer_count(self) -> int:
        return len(self.full_attention_layer_indices)


def audit_qwen35_full_attention_plan(
    model: Any,
    *,
    layer_start: int = 0,
    layer_end: int | None = None,
    expected_full_attention_layers: int | None = None,
) -> Qwen35FullAttentionPlan:
    """Resolve and validate every full-attention module in a layer interval."""

    backbone = _resolve_text_backbone(model)
    config = backbone.config
    layers = backbone.layers
    model_type = getattr(config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise Qwen35PagedIntegrationError(
            f"unsupported text model_type {model_type!r}; expected Qwen3.5 text"
        )
    num_hidden_layers = _require_int(
        getattr(config, "num_hidden_layers", None), "config.num_hidden_layers"
    )
    if len(layers) != num_hidden_layers:
        raise Qwen35PagedIntegrationError(
            "decoder layer count differs from config.num_hidden_layers"
        )
    raw_layer_types = getattr(config, "layer_types", None)
    if not isinstance(raw_layer_types, (list, tuple)) or len(
        raw_layer_types
    ) != num_hidden_layers:
        raise Qwen35PagedIntegrationError(
            "config.layer_types must have one entry per decoder layer"
        )
    layer_types = tuple(raw_layer_types)
    unknown_types = sorted(set(layer_types) - {"full_attention", "linear_attention"})
    if unknown_types:
        raise Qwen35PagedIntegrationError(
            f"unsupported Qwen3.5 layer types: {unknown_types}"
        )
    layer_start = _require_int(layer_start, "layer_start")
    if layer_end is None:
        layer_end = num_hidden_layers
    layer_end = _require_int(layer_end, "layer_end")
    if layer_start < 0 or layer_end > num_hidden_layers or layer_start >= layer_end:
        raise Qwen35PagedIntegrationError(
            "require 0 <= layer_start < layer_end <= num_hidden_layers"
        )
    target = tuple(
        index
        for index in range(layer_start, layer_end)
        if layer_types[index] == "full_attention"
    )
    if not target:
        raise Qwen35PagedIntegrationError(
            "selected layer interval contains no full-attention layers"
        )
    if expected_full_attention_layers is not None:
        expected = _require_int(
            expected_full_attention_layers, "expected_full_attention_layers"
        )
        if len(target) != expected:
            raise Qwen35PagedIntegrationError(
                "dynamic full-attention count mismatch: "
                f"expected={expected}, actual={len(target)}, indices={target}"
            )

    forward_signatures = {}
    for index in target:
        decoder = layers[index]
        if getattr(decoder, "block_type", None) != "full_attention":
            raise Qwen35PagedIntegrationError(
                f"decoder layer {index} block_type disagrees with config.layer_types"
            )
        attention = getattr(decoder, "self_attn", None)
        if attention is None:
            raise Qwen35PagedIntegrationError(
                f"full-attention decoder layer {index} has no self_attn module"
            )
        if getattr(attention, "layer_idx", None) != index:
            raise Qwen35PagedIntegrationError(
                f"attention layer_idx mismatch at decoder layer {index}"
            )
        if getattr(attention, "config", None) is not config:
            raise Qwen35PagedIntegrationError(
                f"attention layer {index} does not share the text config object"
            )
        groups = getattr(attention, "num_key_value_groups", None)
        if isinstance(groups, bool) or not isinstance(groups, int) or groups < 1:
            raise Qwen35PagedIntegrationError(
                f"attention layer {index} has invalid num_key_value_groups"
            )
        parameters = inspect.signature(attention.forward).parameters
        missing = REQUIRED_ATTENTION_FORWARD_PARAMETERS - set(parameters)
        if missing:
            raise Qwen35PagedIntegrationError(
                f"attention layer {index} forward API is missing {sorted(missing)}"
            )
        forward_signatures[str(index)] = tuple(parameters)

    if not hasattr(config, "_attn_implementation"):
        raise Qwen35PagedIntegrationError(
            "text config has no _attn_implementation dispatch property"
        )
    return Qwen35FullAttentionPlan(
        model_type=model_type,
        num_hidden_layers=num_hidden_layers,
        layer_start=layer_start,
        layer_end=layer_end,
        full_attention_layer_indices=target,
        layer_types=layer_types,
        transformers_api={
            "kernel_mode": KERNEL_MODE,
            "production_ttft_optimization_claim_allowed": False,
            "cache_update_position": "after_qk_norm_and_rope",
            "cache_update_return_consumed_by_attention_interface": True,
            "attention_dispatch": "config._attn_implementation",
            "required_forward_parameters": sorted(
                REQUIRED_ATTENTION_FORWARD_PARAMETERS
            ),
            "observed_forward_signatures": forward_signatures,
            "position_semantics": (
                "Qwen3.5 computes four position-id streams; text mask consumes "
                "stream 0 and RoPE consumes streams 1..3 before cache.update"
            ),
        },
    )


def _dense_layer_tensors(layer: Any, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    if bool(getattr(layer, "is_sliding", False)):
        raise Qwen35PagedIntegrationError(
            f"cache layer {index} is sliding; only non-sliding full attention is supported"
        )
    key = getattr(layer, "keys", None)
    value = getattr(layer, "values", None)
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise Qwen35PagedIntegrationError(
            f"cache layer {index} must contain dense Tensor keys/values before conversion"
        )
    if key.ndim != 4 or value.ndim != 4 or key.shape[:3] != value.shape[:3]:
        raise Qwen35PagedIntegrationError(
            f"cache layer {index} has incompatible K/V shapes"
        )
    if key.shape[-2] < 1:
        raise Qwen35PagedIntegrationError(
            f"cache layer {index} has no document KV to page"
        )
    return key, value


@dataclass(frozen=True)
class PagedCacheConversion:
    layer_indices: tuple[int, ...]
    document_length: int
    layer_store_ids: dict[int, int]
    dense_document_nbytes: int
    paged_document_nbytes: int
    page_size: int
    bits: int | None
    group_size: int
    append_page_size: int
    preserve_autograd: bool = False
    preserve_document_graph: bool = False


def _autograd_dense_pages(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    page_size: int,
    preserve_graph: bool,
) -> tuple[KVPage, ...]:
    if key.ndim != 4 or value.ndim != 4:
        raise Qwen35PagedIntegrationError(
            "autograd page key/value must have shape [batch, kv_heads, tokens, dim]"
        )
    if key.shape[:3] != value.shape[:3]:
        raise Qwen35PagedIntegrationError(
            "autograd page key/value batch, head, and token dimensions must match"
        )
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
        raise Qwen35PagedIntegrationError("page_size must be a positive integer")
    pages = []
    for start in range(0, key.shape[-2], page_size):
        end = min(start + page_size, key.shape[-2])
        key_slice = key[..., start:end, :]
        value_slice = value[..., start:end, :]
        if preserve_graph:
            page_key = key_slice.clone()
            page_value = value_slice.clone()
        else:
            page_key = key_slice.detach().clone()
            page_value = value_slice.detach().clone()
        pages.append(KVPage(page_key, page_value))
    return tuple(pages)


class AutogradPagedKVStore(PagedKVStore):
    """Experimental dense-page functional/autograd reference.

    Document pages optionally preserve their producer graph.  Appended query
    pages always preserve it.  No quantizer is used: ``bits=16`` is accepted as
    a dense-clone storage label, while sub-16-bit modes fail before page build.
    """

    def __init__(
        self,
        document_pages: tuple[KVPage, ...] = (),
        *,
        append_page_size: int = 1,
        bits: int | None = None,
        preserve_document_graph: bool = False,
    ) -> None:
        if bits not in (None, 16):
            raise Qwen35PagedIntegrationError(
                "autograd paged KV supports only dense bits=None or bits=16"
            )
        super().__init__(document_pages, append_page_size=append_page_size)
        self.bits = bits
        self.preserve_document_graph = bool(preserve_document_graph)

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
        preserve_document_graph: bool,
    ) -> "AutogradPagedKVStore":
        del group_size
        if bits not in (None, 16):
            raise Qwen35PagedIntegrationError(
                "autograd paged KV supports only dense bits=None or bits=16"
            )
        pages = _autograd_dense_pages(
            key,
            value,
            page_size=page_size,
            preserve_graph=preserve_document_graph,
        )
        return cls(
            pages,
            append_page_size=append_page_size,
            bits=bits,
            preserve_document_graph=preserve_document_graph,
        )

    def append(self, key: torch.Tensor, value: torch.Tensor) -> None:
        pages = _autograd_dense_pages(
            key,
            value,
            page_size=self.append_page_size,
            preserve_graph=True,
        )
        self._validate_pages((*self.pages, *pages))
        self.request_pages.extend(pages)

    def fork(self) -> "AutogradPagedKVStore":
        # Current request pages become immutable shared document pages at the
        # fork boundary; their graph connectivity is intentionally retained.
        return AutogradPagedKVStore(
            self.pages,
            append_page_size=self.append_page_size,
            bits=self.bits,
            preserve_document_graph=self.preserve_document_graph,
        )


class AutogradPagedKVLayer(PagedKVLayer):
    """Experimental layer whose appended request pages remain differentiable.

    Formal full-attention training continues to use Transformers DynamicLayer;
    this class validates graph semantics and is not wired into the inference
    preparation runner below.
    """

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
        preserve_document_graph: bool,
    ) -> "AutogradPagedKVLayer":
        return cls(
            AutogradPagedKVStore.from_dense_document(
                key,
                value,
                page_size=page_size,
                bits=bits,
                group_size=group_size,
                append_page_size=append_page_size,
                preserve_document_graph=preserve_document_graph,
            )
        )

    def fork(self) -> "AutogradPagedKVLayer":
        return AutogradPagedKVLayer(self.store.fork())


def convert_all_planned_cache_layers(
    cache: Any,
    plan: Qwen35FullAttentionPlan,
    *,
    page_size: int,
    bits: int | None = None,
    group_size: int = 64,
    append_page_size: int = 1,
    preserve_autograd: bool = False,
    preserve_document_graph: bool = False,
) -> PagedCacheConversion:
    """Atomically prepare and install pages for every layer in ``plan``.

    Validation and page construction complete for all layers before the cache
    is mutated.  Thus an unsupported/missing layer cannot leave a partially
    converted cache that might silently mix dense and paged attention.
    """

    if preserve_document_graph and not preserve_autograd:
        raise Qwen35PagedIntegrationError(
            "preserve_document_graph requires preserve_autograd=True"
        )
    if preserve_autograd and bits not in (None, 16):
        raise Qwen35PagedIntegrationError(
            "autograd paged KV supports only dense bits=None or bits=16"
        )
    layers = getattr(cache, "layers", None)
    if layers is None or len(layers) < plan.num_hidden_layers:
        raise Qwen35PagedIntegrationError(
            "cache must expose all configured Qwen3.5 layers"
        )
    dense_inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    lengths = set()
    dense_nbytes = 0
    for index in plan.full_attention_layer_indices:
        key, value = _dense_layer_tensors(layers[index], index)
        dense_inputs[index] = (key, value)
        lengths.add(int(key.shape[-2]))
        dense_nbytes += (
            key.numel() * key.element_size()
            + value.numel() * value.element_size()
        )
    if len(lengths) != 1:
        raise Qwen35PagedIntegrationError(
            "all target full-attention layers must have the same document length"
        )
    document_length = lengths.pop()

    replacements: dict[int, PagedKVLayer] = {}
    try:
        for index, (key, value) in dense_inputs.items():
            if preserve_autograd:
                replacement = AutogradPagedKVLayer.from_dense_document(
                    key,
                    value,
                    page_size=page_size,
                    bits=bits,
                    group_size=group_size,
                    append_page_size=append_page_size,
                    preserve_document_graph=preserve_document_graph,
                )
            else:
                replacement = PagedKVLayer.from_dense_document(
                    key,
                    value,
                    page_size=page_size,
                    bits=bits,
                    group_size=group_size,
                    append_page_size=append_page_size,
                )
            if replacement.get_seq_length() != document_length:
                raise Qwen35PagedIntegrationError(
                    f"paged cache layer {index} changed document length"
                )
            replacements[index] = replacement
    except Exception as error:
        if isinstance(error, Qwen35PagedIntegrationError):
            raise
        raise Qwen35PagedIntegrationError(
            f"paged layer construction failed before installation: {error}"
        ) from error

    if set(replacements) != set(plan.full_attention_layer_indices):
        raise Qwen35PagedIntegrationError(
            "internal conversion error: replacement layer set is incomplete"
        )
    for index, replacement in replacements.items():
        layers[index] = replacement
    installed = tuple(
        index
        for index in plan.full_attention_layer_indices
        if isinstance(layers[index], PagedKVLayer)
        and layers[index] is replacements[index]
    )
    if installed != plan.full_attention_layer_indices:
        raise Qwen35PagedIntegrationError(
            "not every planned cache layer retained its paged replacement"
        )
    paged_nbytes = sum(layer.stored_nbytes for layer in replacements.values())
    return PagedCacheConversion(
        layer_indices=plan.full_attention_layer_indices,
        document_length=document_length,
        layer_store_ids={
            index: id(layer.store) for index, layer in replacements.items()
        },
        dense_document_nbytes=dense_nbytes,
        paged_document_nbytes=paged_nbytes,
        page_size=page_size,
        bits=bits,
        group_size=group_size,
        append_page_size=append_page_size,
        preserve_autograd=preserve_autograd,
        preserve_document_graph=preserve_document_graph,
    )


def convert_all_planned_cache_layers_for_training(
    cache: Any,
    plan: Qwen35FullAttentionPlan,
    *,
    page_size: int,
    bits: int | None = None,
    group_size: int = 64,
    append_page_size: int = 1,
    preserve_document_graph: bool,
) -> PagedCacheConversion:
    """Experimental functional conversion of a live cache graph.

    This is a focused capability helper, not a production trainer integration.
    """

    return convert_all_planned_cache_layers(
        cache,
        plan,
        page_size=page_size,
        bits=bits,
        group_size=group_size,
        append_page_size=append_page_size,
        preserve_autograd=True,
        preserve_document_graph=preserve_document_graph,
    )


@dataclass(frozen=True)
class SameCallerCachePair:
    dense_cache: Any
    paged_cache: Any
    conversion: PagedCacheConversion


def clone_dense_and_prepare_paged_cache_pair(
    source_cache: Any,
    plan: Qwen35FullAttentionPlan,
    *,
    page_size: int,
    bits: int | None = None,
    group_size: int = 64,
    append_page_size: int = 1,
) -> SameCallerCachePair:
    """Clone one document cache twice, preserving an eager parity baseline."""

    dense_cache = clone_cache(source_cache)
    paged_cache = clone_cache(source_cache)
    conversion = convert_all_planned_cache_layers(
        paged_cache,
        plan,
        page_size=page_size,
        bits=bits,
        group_size=group_size,
        append_page_size=append_page_size,
    )
    for index in plan.full_attention_layer_indices:
        dense_key, dense_value = _dense_layer_tensors(dense_cache.layers[index], index)
        source_key, source_value = _dense_layer_tensors(source_cache.layers[index], index)
        if not torch.equal(dense_key, source_key) or not torch.equal(
            dense_value, source_value
        ):
            raise Qwen35PagedIntegrationError(
                f"dense parity clone changed document KV at layer {index}"
            )
        if dense_cache.layers[index].get_seq_length() != conversion.document_length:
            raise Qwen35PagedIntegrationError(
                f"dense parity cache length mismatch at layer {index}"
            )
    return SameCallerCachePair(
        dense_cache=dense_cache,
        paged_cache=paged_cache,
        conversion=conversion,
    )


class PagedAttentionHitLedger:
    """Per-run attention interface that rejects missed or dense invocations."""

    def __init__(
        self,
        plan: Qwen35FullAttentionPlan,
        conversion: PagedCacheConversion,
        *,
        expected_calls_per_layer: int = 1,
    ) -> None:
        expected_calls_per_layer = _require_int(
            expected_calls_per_layer, "expected_calls_per_layer"
        )
        if expected_calls_per_layer < 1:
            raise Qwen35PagedIntegrationError(
                "expected_calls_per_layer must be positive"
            )
        if conversion.layer_indices != plan.full_attention_layer_indices:
            raise Qwen35PagedIntegrationError(
                "hit ledger plan and conversion layer sets differ"
            )
        self.expected_layer_indices = plan.full_attention_layer_indices
        self.expected_store_ids = dict(conversion.layer_store_ids)
        self.expected_calls_per_layer = expected_calls_per_layer
        self.counts: Counter[int] = Counter()
        self.calls: list[dict[str, Any]] = []
        self.verified = False

    def attention_forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor | PagedTensorView,
        value: torch.Tensor | PagedTensorView,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx not in self.expected_store_ids:
            raise Qwen35PagedIntegrationError(
                f"paged backend intercepted unexpected attention layer {layer_idx!r}"
            )
        if not isinstance(key, PagedTensorView) or not isinstance(
            value, PagedTensorView
        ):
            raise Qwen35PagedIntegrationError(
                f"attention layer {layer_idx} reached paged backend with dense K/V"
            )
        if key.kind != "key" or value.kind != "value" or key.store is not value.store:
            raise Qwen35PagedIntegrationError(
                f"attention layer {layer_idx} returned unpaired paged K/V views"
            )
        if id(key.store) != self.expected_store_ids[layer_idx]:
            raise Qwen35PagedIntegrationError(
                f"attention layer {layer_idx} used the wrong paged cache store"
            )
        if self.counts[layer_idx] >= self.expected_calls_per_layer:
            raise Qwen35PagedIntegrationError(
                f"attention layer {layer_idx} exceeded its expected intercept count"
            )
        if "audit" in kwargs:
            raise Qwen35PagedIntegrationError(
                "caller-provided audit would bypass the integration ledger"
            )
        call_audit: dict[str, Any] = {}
        output = paged_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            audit=call_audit,
            **kwargs,
        )
        self.counts[layer_idx] += 1
        self.calls.append(
            {
                "layer_idx": layer_idx,
                "query_tokens": int(query.shape[-2]),
                "kv_tokens": key.store.sequence_length,
                "page_count": len(key.store.pages),
                "store_id": id(key.store),
                "materialization": call_audit,
            }
        )
        return output

    def verify_complete(self) -> dict[str, Any]:
        expected = {
            index: self.expected_calls_per_layer
            for index in self.expected_layer_indices
        }
        actual = {index: self.counts[index] for index in self.expected_layer_indices}
        unexpected = sorted(set(self.counts) - set(self.expected_layer_indices))
        if actual != expected or unexpected:
            raise Qwen35PagedIntegrationError(
                "paged attention intercept coverage mismatch: "
                f"expected={expected}, actual={actual}, unexpected={unexpected}"
            )
        self.verified = True
        return {
            "verified": True,
            "kernel_mode": KERNEL_MODE,
            "production_ttft_optimization_claim_allowed": False,
            "expected_layer_indices": self.expected_layer_indices,
            "expected_calls_per_layer": self.expected_calls_per_layer,
            "counts": actual,
            "total_calls": sum(actual.values()),
            "dense_fallback_calls": 0,
            "calls": tuple(self.calls),
        }


@dataclass(frozen=True)
class RegisteredPagedBackend:
    name: str
    transformers_version: str
    ledger: PagedAttentionHitLedger
    kernel_mode: str = KERNEL_MODE


def register_qwen35_paged_backend(
    ledger: PagedAttentionHitLedger,
    *,
    name: str | None = None,
    supported_transformers_major_minor: tuple[tuple[int, int], ...] = (
        (5, 14),
        (5, 15),
    ),
) -> RegisteredPagedBackend:
    """Register one unique ledger-bound backend after a registry API audit."""

    try:
        import transformers
        from transformers.masking_utils import (
            ALL_MASK_ATTENTION_FUNCTIONS,
            AttentionMaskInterface,
            eager_mask,
        )
        from transformers.modeling_utils import (
            ALL_ATTENTION_FUNCTIONS,
            AttentionInterface,
        )
    except ImportError as error:
        raise Qwen35PagedIntegrationError(
            "Transformers with attention and mask registries is required"
        ) from error
    version = getattr(transformers, "__version__", "")
    match = re.match(r"^(\d+)\.(\d+)(?:\.|$)", version)
    observed_major_minor = (
        tuple(map(int, match.groups())) if match is not None else None
    )
    if observed_major_minor not in supported_transformers_major_minor:
        raise Qwen35PagedIntegrationError(
            "unsupported Transformers API version: "
            f"supported={supported_transformers_major_minor}, actual={version!r}"
        )
    registry_api = {
        "AttentionInterface.register": callable(
            getattr(AttentionInterface, "register", None)
        ),
        "AttentionMaskInterface.register": callable(
            getattr(AttentionMaskInterface, "register", None)
        ),
        "ALL_ATTENTION_FUNCTIONS.valid_keys": callable(
            getattr(ALL_ATTENTION_FUNCTIONS, "valid_keys", None)
        ),
        "ALL_MASK_ATTENTION_FUNCTIONS.valid_keys": callable(
            getattr(ALL_MASK_ATTENTION_FUNCTIONS, "valid_keys", None)
        ),
    }
    if not all(registry_api.values()):
        raise Qwen35PagedIntegrationError(
            f"Transformers registry API audit failed: {registry_api}"
        )
    if name is None:
        name = f"qcomem_paged_qwen35_{uuid.uuid4().hex}"
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise Qwen35PagedIntegrationError("attention backend name is invalid")
    if name in ALL_ATTENTION_FUNCTIONS.valid_keys() or name in (
        ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
    ):
        raise Qwen35PagedIntegrationError(
            f"attention backend name is already registered: {name}"
        )
    AttentionInterface.register(name, ledger.attention_forward)
    AttentionMaskInterface.register(name, eager_mask)
    if name not in ALL_ATTENTION_FUNCTIONS.valid_keys() or name not in (
        ALL_MASK_ATTENTION_FUNCTIONS.valid_keys()
    ):
        raise Qwen35PagedIntegrationError(
            "Transformers registries did not retain the paged backend"
        )
    return RegisteredPagedBackend(
        name=name,
        transformers_version=version,
        ledger=ledger,
    )


@contextmanager
def temporary_attention_implementation(
    config: Any, implementation: str
) -> Iterator[None]:
    if not hasattr(config, "_attn_implementation"):
        raise Qwen35PagedIntegrationError(
            "config has no _attn_implementation dispatch property"
        )
    original = config._attn_implementation
    config._attn_implementation = implementation
    if config._attn_implementation != implementation:
        config._attn_implementation = original
        raise Qwen35PagedIntegrationError(
            "config rejected the requested attention implementation"
        )
    try:
        yield
    finally:
        config._attn_implementation = original


def _extract_final_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        logits = output
    elif isinstance(getattr(output, "logits", None), torch.Tensor):
        logits = output.logits
    elif isinstance(output, dict) and isinstance(output.get("logits"), torch.Tensor):
        logits = output["logits"]
    else:
        raise Qwen35PagedIntegrationError(
            "same-caller output must be logits tensor or expose .logits"
        )
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    if logits.ndim != 2:
        raise Qwen35PagedIntegrationError(
            "final logits must have shape [batch, vocabulary]"
        )
    if not torch.isfinite(logits).all():
        raise Qwen35PagedIntegrationError("same-caller logits contain NaN or Inf")
    return logits.detach()


def _cache_lengths(cache: Any, indices: tuple[int, ...]) -> dict[int, int]:
    values = {}
    for index in indices:
        layer = cache.layers[index]
        get_length = getattr(layer, "get_seq_length", None)
        if not callable(get_length):
            raise Qwen35PagedIntegrationError(
                f"cache layer {index} has no get_seq_length method"
            )
        values[index] = int(get_length())
    return values


def run_same_caller_eager_paged_gate(
    caller: Callable[[Any], Any],
    *,
    text_config: Any,
    caches: SameCallerCachePair,
    backend: RegisteredPagedBackend,
    rtol: float = 1e-3,
    atol: float = 1e-3,
) -> dict[str, Any]:
    """Invoke one caller twice and require final-logit/token/cache parity.

    ``caller`` must close over the same input tokens, positions, and mask for
    both calls and accept only the request-local cache.  The dense invocation
    uses Transformers' standard mutable eager path; the paged invocation uses
    the unique ledger-bound backend.
    """

    if not callable(caller):
        raise Qwen35PagedIntegrationError("caller must be callable")
    if rtol < 0 or atol < 0 or not math.isfinite(rtol) or not math.isfinite(atol):
        raise Qwen35PagedIntegrationError("rtol and atol must be finite and nonnegative")
    ledger = backend.ledger
    if ledger.calls or ledger.counts or ledger.verified:
        raise Qwen35PagedIntegrationError(
            "same-caller gate requires a fresh, empty hit ledger"
        )
    if ledger.expected_store_ids != caches.conversion.layer_store_ids:
        raise Qwen35PagedIntegrationError(
            "registered ledger is not bound to the supplied paged cache"
        )

    with torch.inference_mode(), temporary_attention_implementation(
        text_config, "eager"
    ):
        dense_logits = _extract_final_logits(caller(caches.dense_cache))
    with torch.inference_mode(), temporary_attention_implementation(
        text_config, backend.name
    ):
        paged_logits = _extract_final_logits(caller(caches.paged_cache))
    intercept = ledger.verify_complete()

    if dense_logits.shape != paged_logits.shape:
        raise Qwen35PagedIntegrationError(
            "dense and paged final-logit shapes differ"
        )
    dense_tokens = torch.argmax(dense_logits, dim=-1)
    paged_tokens = torch.argmax(paged_logits, dim=-1)
    token_exact = torch.equal(dense_tokens, paged_tokens)
    close = torch.allclose(dense_logits, paged_logits, rtol=rtol, atol=atol)
    error = (dense_logits.float() - paged_logits.float()).abs()
    denominator = torch.linalg.vector_norm(dense_logits.float()).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    relative_l2 = float(
        (torch.linalg.vector_norm(error.reshape(-1)) / denominator).item()
    )
    dense_lengths = _cache_lengths(
        caches.dense_cache, caches.conversion.layer_indices
    )
    paged_lengths = _cache_lengths(
        caches.paged_cache, caches.conversion.layer_indices
    )
    cache_length_exact = dense_lengths == paged_lengths
    result = {
        "passed": bool(close and token_exact and cache_length_exact),
        "kernel_mode": KERNEL_MODE,
        "production_ttft_optimization_claim_allowed": False,
        "same_caller_object": True,
        "caller_identity": id(caller),
        "final_logits_close": bool(close),
        "final_tokens_exact": token_exact,
        "cache_lengths_exact": cache_length_exact,
        "dense_cache_lengths": dense_lengths,
        "paged_cache_lengths": paged_lengths,
        "max_abs_logit_error": float(error.max().item()),
        "relative_l2_logit_error": relative_l2,
        "rtol": rtol,
        "atol": atol,
        "dense_tokens": dense_tokens.cpu().tolist(),
        "paged_tokens": paged_tokens.cpu().tolist(),
        "intercept": intercept,
    }
    if not result["passed"]:
        raise Qwen35PagedIntegrationError(
            "same-caller eager/paged parity gate failed: "
            f"logits_close={close}, token_exact={token_exact}, "
            f"cache_length_exact={cache_length_exact}, "
            f"max_abs_error={result['max_abs_logit_error']}"
        )
    return result


def require_passed_reference_gate_before_benchmark(
    gate_result: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed unless the same runner already passed correctness gates.

    This authorizes measurement of the Python reference only.  It explicitly
    does not authorize describing the result as a production TTFT optimization.
    """

    if not isinstance(gate_result, dict) or gate_result.get("passed") is not True:
        raise Qwen35PagedIntegrationError(
            "benchmarking requires a passed same-caller eager/paged gate"
        )
    intercept = gate_result.get("intercept")
    if not isinstance(intercept, dict) or intercept.get("verified") is not True:
        raise Qwen35PagedIntegrationError(
            "benchmarking requires verified all-layer intercept coverage"
        )
    if gate_result.get("kernel_mode") != KERNEL_MODE:
        raise Qwen35PagedIntegrationError("benchmark kernel mode is not the reference")
    return {
        "benchmark_gate_passed": True,
        "kernel_mode": KERNEL_MODE,
        "benchmark_scope": "correctness/reference measurement only",
        "production_ttft_optimization_claim_allowed": False,
        "intercept_total_calls": intercept.get("total_calls"),
    }


@dataclass(frozen=True)
class PreparedQwen35PagedIntegration:
    plan: Qwen35FullAttentionPlan
    caches: SameCallerCachePair
    ledger: PagedAttentionHitLedger
    backend: RegisteredPagedBackend


def prepare_qwen35_paged_integration(
    model: Any,
    source_cache: Any,
    *,
    layer_start: int,
    layer_end: int | None = None,
    expected_full_attention_layers: int | None = None,
    expected_calls_per_layer: int = 1,
    page_size: int,
    bits: int | None = None,
    group_size: int = 64,
    append_page_size: int = 1,
    backend_name: str | None = None,
) -> PreparedQwen35PagedIntegration:
    """One-call preparation used by a real Qwen3.5 same-caller runner."""

    plan = audit_qwen35_full_attention_plan(
        model,
        layer_start=layer_start,
        layer_end=layer_end,
        expected_full_attention_layers=expected_full_attention_layers,
    )
    caches = clone_dense_and_prepare_paged_cache_pair(
        source_cache,
        plan,
        page_size=page_size,
        bits=bits,
        group_size=group_size,
        append_page_size=append_page_size,
    )
    ledger = PagedAttentionHitLedger(
        plan,
        caches.conversion,
        expected_calls_per_layer=expected_calls_per_layer,
    )
    backend = register_qwen35_paged_backend(
        ledger,
        name=backend_name,
    )
    return PreparedQwen35PagedIntegration(
        plan=plan,
        caches=caches,
        ledger=ledger,
        backend=backend,
    )


__all__ = [
    "AutogradPagedKVLayer",
    "AutogradPagedKVStore",
    "KERNEL_MODE",
    "PagedAttentionHitLedger",
    "PagedCacheConversion",
    "PreparedQwen35PagedIntegration",
    "Qwen35FullAttentionPlan",
    "Qwen35PagedIntegrationError",
    "RegisteredPagedBackend",
    "SameCallerCachePair",
    "audit_qwen35_full_attention_plan",
    "clone_dense_and_prepare_paged_cache_pair",
    "convert_all_planned_cache_layers",
    "convert_all_planned_cache_layers_for_training",
    "prepare_qwen35_paged_integration",
    "require_passed_reference_gate_before_benchmark",
    "register_qwen35_paged_backend",
    "run_same_caller_eager_paged_gate",
    "temporary_attention_implementation",
]
