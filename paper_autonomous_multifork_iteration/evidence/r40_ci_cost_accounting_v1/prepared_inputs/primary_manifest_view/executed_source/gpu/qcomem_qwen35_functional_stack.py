from __future__ import annotations

"""Reusable functional Qwen3.5 document-cache stack.

The stack combines two independently audited state implementations:

* full-attention K/V is held as immutable document pages plus request-local
  append pages and consumed by the reference online-softmax attention; and
* Qwen3.5 GatedDeltaNet convolution/recurrent state is held as an immutable
  document base plus out-of-place request-local replacement state.

No decoder layer is allowed to fall back to ``DecoderLayer.forward``.  Layer
types and coverage are derived from the loaded text config on every build.
The Python attention and recurrent kernels are correctness references, not a
production latency claim.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import torch

from qcomem_paged_attention import PagedKVLayer, PagedTensorView, paged_attention_forward
from qcomem_qwen35_gdn_functional import (
    AUTOGRAD_PRESERVING,
    INFERENCE_DETACHED,
    GDNContractError,
    GradientSemantics,
    ImmutableGDNBase,
    QueryLocalGDNState,
    Qwen35GDNDispatchPlan,
    audit_qwen35_gdn_dispatch_plan,
    dispatch_qwen35_decoder_layer,
    immutable_base_from_transformers_cache,
    zero_gdn_base,
)
from qcomem_qwen35_paged_integration import (
    AutogradPagedKVLayer,
    KERNEL_MODE,
    Qwen35FullAttentionPlan,
    Qwen35PagedIntegrationError,
    audit_qwen35_full_attention_plan,
)
from qcomem_qwen35_native_cache import (
    NativeFunctionalCacheInstall,
    NativeFunctionalCacheError,
    functional_linear_cache_telemetry,
    install_native_functional_linear_cache,
)


class Qwen35FunctionalStackError(RuntimeError):
    """Raised when the all-layer functional contract cannot be proven."""


def resolve_qwen35_text_backbone(model: Any) -> Any:
    """Resolve the text decoder without depending on one outer auto class."""

    candidates = []
    inner = getattr(model, "model", None)
    if inner is not None:
        language_model = getattr(inner, "language_model", None)
        if language_model is not None:
            candidates.append(language_model)
        candidates.append(inner)
    language_model = getattr(model, "language_model", None)
    if language_model is not None:
        candidates.append(language_model)
    candidates.append(model)
    for candidate in candidates:
        if all(
            hasattr(candidate, name)
            for name in ("layers", "config", "embed_tokens", "norm", "rotary_emb")
        ):
            return candidate
    raise Qwen35FunctionalStackError(
        "cannot resolve Qwen3.5 text backbone with layers/config/embed/norm/rope"
    )


@dataclass(frozen=True)
class Qwen35FunctionalStackPlan:
    """One config-derived routing plan for all decoder layers."""

    gdn: Qwen35GDNDispatchPlan
    full: Qwen35FullAttentionPlan

    @property
    def total_layers(self) -> int:
        return self.gdn.total_layers

    @property
    def linear_layer_indices(self) -> tuple[int, ...]:
        return self.gdn.linear_layer_indices

    @property
    def full_attention_layer_indices(self) -> tuple[int, ...]:
        return self.gdn.full_attention_layer_indices

    def metadata(self) -> dict[str, Any]:
        return {
            "total_layers": self.total_layers,
            "linear_layer_indices": self.linear_layer_indices,
            "linear_layer_count": len(self.linear_layer_indices),
            "full_attention_layer_indices": self.full_attention_layer_indices,
            "full_attention_layer_count": len(self.full_attention_layer_indices),
            "layer_types": self.gdn.layer_types,
            "model_type": self.full.model_type,
            "kernel_mode": KERNEL_MODE,
            "production_ttft_optimization_claim_allowed": False,
        }


def audit_qwen35_functional_stack_plan(model: Any) -> Qwen35FunctionalStackPlan:
    backbone = resolve_qwen35_text_backbone(model)
    try:
        gdn = audit_qwen35_gdn_dispatch_plan(backbone.layers, backbone.config)
        full = audit_qwen35_full_attention_plan(backbone)
    except (GDNContractError, Qwen35PagedIntegrationError) as error:
        raise Qwen35FunctionalStackError(str(error)) from error
    if gdn.full_attention_layer_indices != full.full_attention_layer_indices:
        raise Qwen35FunctionalStackError(
            "GDN and paged-attention audits derived different full layer sets"
        )
    if gdn.total_layers != full.num_hidden_layers:
        raise Qwen35FunctionalStackError(
            "GDN and paged-attention audits derived different layer counts"
        )
    if not gdn.linear_layer_indices or not gdn.full_attention_layer_indices:
        raise Qwen35FunctionalStackError(
            "functional stack requires both linear and full-attention layers"
        )
    return Qwen35FunctionalStackPlan(gdn=gdn, full=full)


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _sample_fingerprint(tensor: torch.Tensor) -> tuple[float, ...]:
    if tensor.numel() == 0:
        return ()
    flat = tensor.detach().reshape(-1)
    count = min(8, flat.numel())
    indices = torch.linspace(0, flat.numel() - 1, count, device=flat.device)
    indices = indices.round().long()
    return tuple(float(item) for item in flat[indices].float().cpu().tolist())


@dataclass(frozen=True)
class _TensorGuard:
    tensor: torch.Tensor
    label: str
    storage_key: tuple[str, int, int]
    version: int | None
    sample: tuple[float, ...]

    @classmethod
    def capture(cls, tensor: torch.Tensor, label: str) -> "_TensorGuard":
        try:
            version = tensor._version
        except RuntimeError:
            version = None
        return cls(
            tensor=tensor,
            label=label,
            storage_key=_storage_key(tensor),
            version=version,
            sample=_sample_fingerprint(tensor),
        )

    def verify(self) -> None:
        if _storage_key(self.tensor) != self.storage_key:
            raise Qwen35FunctionalStackError(f"{self.label} storage changed")
        if self.version is not None:
            try:
                current_version = self.tensor._version
            except RuntimeError as error:
                raise Qwen35FunctionalStackError(
                    f"{self.label} version counter became unavailable"
                ) from error
            if current_version != self.version:
                raise Qwen35FunctionalStackError(
                    f"{self.label} version changed {self.version}->{current_version}"
                )
        if _sample_fingerprint(self.tensor) != self.sample:
            raise Qwen35FunctionalStackError(f"{self.label} sampled values changed")


def _payload_tensors(payload: Any) -> Iterable[torch.Tensor]:
    if isinstance(payload, torch.Tensor):
        yield payload
        return
    for name in ("data", "scales", "biases"):
        value = getattr(payload, name, None)
        if isinstance(value, torch.Tensor):
            yield value


def _paged_layer_tensors(layer: PagedKVLayer) -> Iterable[torch.Tensor]:
    for page in layer.store.pages:
        yield from _payload_tensors(page.key)
        yield from _payload_tensors(page.value)


def _document_state_tensors(
    gdn_bases: Mapping[int, ImmutableGDNBase],
    full_layers: Mapping[int, PagedKVLayer],
) -> Iterable[torch.Tensor]:
    for base in gdn_bases.values():
        yield base.conv_state
        yield base.recurrent_state
    for layer in full_layers.values():
        yield from _paged_layer_tensors(layer)


def _unique_nbytes(tensors: Iterable[torch.Tensor]) -> int:
    storages: dict[tuple[str, int, int], int] = {}
    for tensor in tensors:
        key = _storage_key(tensor)
        storages[key] = key[-1]
    return sum(storages.values())


def _unique_storage_map(
    tensors: Iterable[torch.Tensor],
) -> dict[tuple[str, int, int], int]:
    result = {}
    for tensor in tensors:
        key = _storage_key(tensor)
        result[key] = key[-1]
    return result


@dataclass(frozen=True)
class Qwen35FunctionalDocumentState:
    """Immutable reusable state after one document prefill."""

    plan: Qwen35FunctionalStackPlan
    document_length: int
    gdn_bases: Mapping[int, ImmutableGDNBase]
    full_layers: Mapping[int, PagedKVLayer]
    gradient_semantics: GradientSemantics
    training_mode: bool
    attention_bits: int | None
    page_size: int
    group_size: int
    append_page_size: int
    source: str
    _guards: tuple[_TensorGuard, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.document_length < 1:
            raise Qwen35FunctionalStackError("document_length must be positive")
        if set(self.gdn_bases) != set(self.plan.linear_layer_indices):
            raise Qwen35FunctionalStackError(
                "document GDN base layers do not match the config-derived plan"
            )
        if set(self.full_layers) != set(self.plan.full_attention_layer_indices):
            raise Qwen35FunctionalStackError(
                "document paged layers do not match the config-derived plan"
            )
        if self.gradient_semantics not in (
            AUTOGRAD_PRESERVING,
            INFERENCE_DETACHED,
        ):
            raise Qwen35FunctionalStackError("invalid document gradient semantics")
        if self.training_mode and self.attention_bits not in (None, 16):
            raise Qwen35FunctionalStackError(
                "training paged K/V supports only dense/16-bit storage"
            )
        for index, layer in self.full_layers.items():
            if layer.get_seq_length() != self.document_length:
                raise Qwen35FunctionalStackError(
                    f"full layer {index} document length mismatch"
                )
            if self.training_mode and not isinstance(layer, AutogradPagedKVLayer):
                raise Qwen35FunctionalStackError(
                    f"training full layer {index} is not autograd-aware"
                )
        guards = tuple(
            _TensorGuard.capture(tensor, f"document-state[{position}]")
            for position, tensor in enumerate(
                _document_state_tensors(self.gdn_bases, self.full_layers)
            )
        )
        object.__setattr__(self, "_guards", guards)

    def assert_unchanged(self) -> None:
        for base in self.gdn_bases.values():
            base.assert_unchanged()
        for guard in self._guards:
            guard.verify()

    @property
    def gdn_base_nbytes(self) -> int:
        return _unique_nbytes(
            tensor
            for base in self.gdn_bases.values()
            for tensor in (base.conv_state, base.recurrent_state)
        )

    @property
    def paged_kv_nbytes(self) -> int:
        return _unique_nbytes(
            tensor
            for layer in self.full_layers.values()
            for tensor in _paged_layer_tensors(layer)
        )

    @property
    def persistent_nbytes(self) -> int:
        return _unique_nbytes(
            _document_state_tensors(self.gdn_bases, self.full_layers)
        )

    def memory_report(self) -> dict[str, Any]:
        return {
            "persistent_gdn_base_nbytes": self.gdn_base_nbytes,
            "persistent_paged_kv_nbytes": self.paged_kv_nbytes,
            "persistent_total_nbytes": self.persistent_nbytes,
            "attention_bits": self.attention_bits,
            "page_size": self.page_size,
            "gradient_semantics": self.gradient_semantics,
            "training_mode": self.training_mode,
            "source": self.source,
        }

    def fork(self) -> "Qwen35FunctionalRequestState":
        self.assert_unchanged()
        return Qwen35FunctionalRequestState(
            document=self,
            current_length=self.document_length,
            gdn_states={
                index: QueryLocalGDNState.from_base(base)
                for index, base in self.gdn_bases.items()
            },
            full_layers={
                index: layer.fork() for index, layer in self.full_layers.items()
            },
        )


@dataclass(frozen=True)
class Qwen35FunctionalRequestState:
    """One request's out-of-place deltas over an immutable document state."""

    document: Qwen35FunctionalDocumentState
    current_length: int
    gdn_states: Mapping[int, QueryLocalGDNState]
    full_layers: Mapping[int, PagedKVLayer]

    def __post_init__(self) -> None:
        if self.current_length < self.document.document_length:
            raise Qwen35FunctionalStackError(
                "request length cannot precede the document boundary"
            )
        if set(self.gdn_states) != set(self.document.plan.linear_layer_indices):
            raise Qwen35FunctionalStackError("request GDN layer set is incomplete")
        if set(self.full_layers) != set(
            self.document.plan.full_attention_layer_indices
        ):
            raise Qwen35FunctionalStackError("request paged layer set is incomplete")
        for index, state in self.gdn_states.items():
            if state.base is not self.document.gdn_bases[index]:
                raise Qwen35FunctionalStackError(
                    f"request GDN layer {index} does not reference document base"
                )
        for index, layer in self.full_layers.items():
            if layer.get_seq_length() != self.current_length:
                raise Qwen35FunctionalStackError(
                    f"request full layer {index} length mismatch"
                )

    def memory_report(self) -> dict[str, Any]:
        document_storages = _unique_storage_map(
            _document_state_tensors(
                self.document.gdn_bases, self.document.full_layers
            )
        )
        request_tensors = []
        for state in self.gdn_states.values():
            request_tensors.extend(
                (state.base.conv_state, state.base.recurrent_state)
            )
            if state.conv_delta is not None:
                assert state.recurrent_delta is not None
                request_tensors.extend((state.conv_delta, state.recurrent_delta))
        for layer in self.full_layers.values():
            request_tensors.extend(_paged_layer_tensors(layer))
        request_storages = _unique_storage_map(request_tensors)
        shared = set(document_storages) & set(request_storages)
        private = set(request_storages) - set(document_storages)
        return {
            "shared_document_nbytes": sum(request_storages[key] for key in shared),
            "query_private_nbytes": sum(request_storages[key] for key in private),
            "request_total_referenced_nbytes": sum(request_storages.values()),
            "document_persistent_nbytes": self.document.persistent_nbytes,
            "current_length": self.current_length,
        }


@dataclass(frozen=True)
class Qwen35FunctionalPass:
    last_hidden_state: torch.Tensor
    state: Qwen35FunctionalDocumentState | Qwen35FunctionalRequestState
    telemetry: dict[str, Any]


@dataclass(frozen=True)
class Qwen35NativeFunctionalState:
    """Real Transformers cache with functional linear-state rebinds installed.

    This is the preferred training path: Qwen's real DecoderLayer, FLA and
    causal-conv kernels remain untouched.  Only the two stock linear-cache
    ``copy_`` writes are replaced by tensor rebinding.  Full-attention retains
    Transformers' already out-of-place ``DynamicLayer.update``.
    """

    plan: Qwen35FunctionalStackPlan
    cache: Any
    install: NativeFunctionalCacheInstall
    current_length: int = 0
    passes: int = 0

    def __post_init__(self) -> None:
        if self.current_length < 0 or self.passes < 0:
            raise Qwen35FunctionalStackError(
                "native functional cache counters must be non-negative"
            )
        if self.install.linear_layer_indices != self.plan.linear_layer_indices:
            raise Qwen35FunctionalStackError(
                "native cache install linear layers differ from stack plan"
            )
        if self.install.full_attention_layer_indices != (
            self.plan.full_attention_layer_indices
        ):
            raise Qwen35FunctionalStackError(
                "native cache install full layers differ from stack plan"
            )


@dataclass(frozen=True)
class Qwen35NativeFunctionalPass:
    last_hidden_state: torch.Tensor
    state: Qwen35NativeFunctionalState
    telemetry: dict[str, Any]


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    first = value[..., : value.shape[-1] // 2]
    second = value[..., value.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rotary(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotary_dim = cos.shape[-1]
    query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query = query_rotary * cos + _rotate_half(query_rotary) * sin
    key = key_rotary * cos + _rotate_half(key_rotary) * sin
    return torch.cat((query, query_pass), dim=-1), torch.cat(
        (key, key_pass), dim=-1
    )


def _position_context(
    backbone: Any,
    hidden_states: torch.Tensor,
    *,
    offset: int,
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    batch, length, _ = hidden_states.shape
    positions = torch.arange(
        offset, offset + length, device=hidden_states.device, dtype=torch.long
    )
    positions = positions.view(1, 1, length).expand(4, batch, -1)
    text_positions = positions[0]
    rope_positions = positions[1:]
    embeddings = backbone.rotary_emb(hidden_states, rope_positions)
    if not isinstance(embeddings, tuple) or len(embeddings) != 2:
        raise Qwen35FunctionalStackError("rotary_emb must return (cos, sin)")
    return text_positions, embeddings


def _causal_mask(
    hidden_states: torch.Tensor,
    *,
    past_length: int,
) -> torch.Tensor:
    batch, query_length, _ = hidden_states.shape
    total_length = past_length + query_length
    query_positions = torch.arange(
        past_length,
        total_length,
        device=hidden_states.device,
    )
    key_positions = torch.arange(total_length, device=hidden_states.device)
    allowed = key_positions.view(1, -1) <= query_positions.view(-1, 1)
    mask = torch.zeros(
        (batch, 1, query_length, total_length),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )
    return mask.masked_fill(
        ~allowed.view(1, 1, query_length, total_length),
        torch.finfo(hidden_states.dtype).min,
    )


def _project_full_attention(
    attention: Any,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    required = (
        "q_proj",
        "k_proj",
        "v_proj",
        "q_norm",
        "k_norm",
        "o_proj",
        "head_dim",
        "num_key_value_groups",
        "scaling",
    )
    missing = [name for name in required if not hasattr(attention, name)]
    if missing:
        raise Qwen35FunctionalStackError(
            f"full attention module is missing attributes {missing}"
        )
    input_shape = hidden_states.shape[:-1]
    head_dim = int(attention.head_dim)
    query_and_gate = attention.q_proj(hidden_states)
    if query_and_gate.shape[-1] % (2 * head_dim):
        raise Qwen35FunctionalStackError("q_proj output cannot be split into heads+gate")
    query_and_gate = query_and_gate.view(*input_shape, -1, head_dim * 2)
    query, gate = torch.chunk(query_and_gate, 2, dim=-1)
    gate = gate.reshape(*input_shape, -1)
    hidden_shape = (*input_shape, -1, head_dim)
    query = attention.q_norm(query.view(hidden_shape)).transpose(1, 2)
    key = attention.k_norm(attention.k_proj(hidden_states).view(hidden_shape))
    key = key.transpose(1, 2)
    value = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    query, key = _apply_rotary(query, key, *position_embeddings)
    return query, key, value, gate


def _repeat_kv(value: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return value
    batch, heads, length, dim = value.shape
    return (
        value[:, :, None, :, :]
        .expand(batch, heads, groups, length, dim)
        .reshape(batch, heads * groups, length, dim)
    )


def _dense_eager_attention(
    attention: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    groups = int(attention.num_key_value_groups)
    key = _repeat_kv(key, groups)
    value = _repeat_kv(value, groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * float(attention.scaling)
    scores = scores + attention_mask
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


def _finish_full_decoder_layer(
    decoder_layer: Any,
    residual: torch.Tensor,
    attention_output: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    input_shape = residual.shape[:-1]
    attention_output = attention_output.reshape(*input_shape, -1).contiguous()
    attention_output = attention_output * torch.sigmoid(gate)
    hidden_states = residual + decoder_layer.self_attn.o_proj(attention_output)
    residual = hidden_states
    normalized = decoder_layer.post_attention_layernorm(hidden_states)
    mlp_output = decoder_layer.mlp(normalized)
    if isinstance(mlp_output, tuple):
        if not mlp_output or not isinstance(mlp_output[0], torch.Tensor):
            raise Qwen35FunctionalStackError(
                "MoE MLP tuple must begin with a tensor"
            )
        mlp_output = mlp_output[0]
    if not isinstance(mlp_output, torch.Tensor) or mlp_output.shape != residual.shape:
        raise Qwen35FunctionalStackError("decoder MLP output shape mismatch")
    return residual + mlp_output


def _new_document_page_layer(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    training_mode: bool,
    preserve_document_graph: bool,
    attention_bits: int | None,
    page_size: int,
    group_size: int,
    append_page_size: int,
) -> PagedKVLayer:
    if training_mode:
        return AutogradPagedKVLayer.from_dense_document(
            key,
            value,
            page_size=page_size,
            bits=attention_bits,
            group_size=group_size,
            append_page_size=append_page_size,
            preserve_document_graph=preserve_document_graph,
        )
    return PagedKVLayer.from_dense_document(
        key,
        value,
        page_size=page_size,
        bits=attention_bits,
        group_size=group_size,
        append_page_size=append_page_size,
    )


def _prefill_full_attention_dispatch(
    decoder_layer: Any,
    hidden_states: torch.Tensor,
    *,
    layer_idx: int,
    attention_mask: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    training_mode: bool,
    preserve_document_graph: bool,
    attention_bits: int | None,
    page_size: int,
    group_size: int,
    append_page_size: int,
) -> tuple[torch.Tensor, PagedKVLayer, dict[str, Any]]:
    if getattr(decoder_layer, "block_type", None) != "full_attention":
        raise Qwen35FunctionalStackError("full prefill reached a non-full layer")
    residual = hidden_states
    normalized = decoder_layer.input_layernorm(hidden_states)
    query, key, value, gate = _project_full_attention(
        decoder_layer.self_attn, normalized, position_embeddings
    )
    # Document attention uses the live dense projections. Packing happens only
    # after its output is defined, matching normal cache quantization semantics.
    attention_output = _dense_eager_attention(
        decoder_layer.self_attn, query, key, value, attention_mask
    )
    page_layer = _new_document_page_layer(
        key,
        value,
        training_mode=training_mode,
        preserve_document_graph=preserve_document_graph,
        attention_bits=attention_bits,
        page_size=page_size,
        group_size=group_size,
        append_page_size=append_page_size,
    )
    hidden_states = _finish_full_decoder_layer(
        decoder_layer, residual, attention_output, gate
    )
    return hidden_states, page_layer, {
        "kind": "full_attention",
        "layer_idx": layer_idx,
        "route": "functional-qwen35-full-document-prefill",
        "kernel": "dense_eager_document_then_page_pack",
        "kernel_mode": KERNEL_MODE,
        "sequence_length": int(hidden_states.shape[1]),
        "page_count": len(page_layer.store.pages),
        "stored_nbytes": page_layer.stored_nbytes,
        "mutable_cache_used": False,
        "fallback_used": False,
        "full_kv_concatenations": 0,
        "production_ttft_optimization_claim_allowed": False,
    }


def _continue_full_attention_dispatch(
    decoder_layer: Any,
    hidden_states: torch.Tensor,
    layer_state: PagedKVLayer,
    *,
    layer_idx: int,
    attention_mask: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, PagedKVLayer, dict[str, Any]]:
    if not isinstance(layer_state, PagedKVLayer):
        raise Qwen35FunctionalStackError(
            f"full layer {layer_idx} is missing a paged state"
        )
    residual = hidden_states
    normalized = decoder_layer.input_layernorm(hidden_states)
    query, key, value, gate = _project_full_attention(
        decoder_layer.self_attn, normalized, position_embeddings
    )
    # Fork before append so the caller's request state remains immutable.
    next_layer = layer_state.fork()
    key_view, value_view = next_layer.update(key, value)
    if not isinstance(key_view, PagedTensorView) or not isinstance(
        value_view, PagedTensorView
    ):
        raise Qwen35FunctionalStackError(
            f"full layer {layer_idx} did not return paged views"
        )
    if key_view.store is not next_layer.store or value_view.store is not next_layer.store:
        raise Qwen35FunctionalStackError(
            f"full layer {layer_idx} returned views from a different store"
        )
    audit: dict[str, Any] = {}
    attention_output, weights = paged_attention_forward(
        decoder_layer.self_attn,
        query,
        key_view,
        value_view,
        attention_mask,
        dropout=0.0,
        scaling=float(decoder_layer.self_attn.scaling),
        audit=audit,
    )
    if weights is not None:
        raise Qwen35FunctionalStackError("paged reference unexpectedly returned weights")
    if audit.get("full_kv_concatenations") != 0:
        raise Qwen35FunctionalStackError("paged path materialized a full K/V concat")
    if audit.get("gqa_kv_repeat_materializations") != 0:
        raise Qwen35FunctionalStackError("paged path materialized repeated GQA K/V")
    hidden_states = _finish_full_decoder_layer(
        decoder_layer, residual, attention_output, gate
    )
    return hidden_states, next_layer, {
        "kind": "full_attention",
        "layer_idx": layer_idx,
        "route": "functional-qwen35-paged-continuation",
        "kernel": KERNEL_MODE,
        "kernel_mode": KERNEL_MODE,
        "sequence_length": int(hidden_states.shape[1]),
        "mutable_cache_used": False,
        "fallback_used": False,
        "production_ttft_optimization_claim_allowed": False,
        **audit,
    }


def _verify_pass_coverage(
    plan: Qwen35FunctionalStackPlan,
    layer_telemetry: list[dict[str, Any]],
    *,
    phase: str,
) -> dict[str, Any]:
    if len(layer_telemetry) != plan.total_layers:
        raise Qwen35FunctionalStackError(
            f"{phase} intercepted {len(layer_telemetry)}/{plan.total_layers} layers"
        )
    linear = tuple(
        int(row["layer_idx"])
        for row in layer_telemetry
        if row.get("kind") == "linear_attention"
    )
    full = tuple(
        int(row["layer_idx"])
        for row in layer_telemetry
        if row.get("kind") == "full_attention"
    )
    if linear != plan.linear_layer_indices:
        raise Qwen35FunctionalStackError(
            f"{phase} linear intercept mismatch: expected={plan.linear_layer_indices}, actual={linear}"
        )
    if full != plan.full_attention_layer_indices:
        raise Qwen35FunctionalStackError(
            f"{phase} full intercept mismatch: expected={plan.full_attention_layer_indices}, actual={full}"
        )
    unsafe = [
        int(row["layer_idx"])
        for row in layer_telemetry
        if row.get("fallback_used") is not False
        or row.get("mutable_cache_used") is not False
    ]
    if unsafe:
        raise Qwen35FunctionalStackError(
            f"{phase} used fallback or mutable cache at layers {unsafe}"
        )
    max_unpack = max(
        (
            int(row.get("max_single_unpack_page_nbytes", 0))
            for row in layer_telemetry
        ),
        default=0,
    )
    return {
        "verified": True,
        "phase": phase,
        "kernel_mode": KERNEL_MODE,
        "production_ttft_optimization_claim_allowed": False,
        "expected_linear_layer_count": len(plan.linear_layer_indices),
        "observed_linear_layer_count": len(linear),
        "expected_full_attention_layer_count": len(
            plan.full_attention_layer_indices
        ),
        "observed_full_attention_layer_count": len(full),
        "linear_layer_indices": linear,
        "full_attention_layer_indices": full,
        "fallback_layers": (),
        "mutable_cache_layers": (),
        "max_single_unpack_page_nbytes": max_unpack,
        "layers": tuple(layer_telemetry),
    }


def _validate_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim == 1:
        tokens = tokens.unsqueeze(0)
    if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
        raise Qwen35FunctionalStackError(
            "input_ids must have shape [batch,tokens] with non-empty dimensions"
        )
    if tokens.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise Qwen35FunctionalStackError("input_ids must be integer token IDs")
    return tokens


def new_qwen35_native_functional_state(model: Any) -> Qwen35NativeFunctionalState:
    """Create the real Qwen cache and install all functional rebind seams."""

    plan = audit_qwen35_functional_stack_plan(model)
    backbone = resolve_qwen35_text_backbone(model)
    try:
        from transformers.cache_utils import DynamicCache
    except ImportError as error:  # pragma: no cover - deployment environment.
        raise Qwen35FunctionalStackError(
            "Transformers DynamicCache is required for the native functional path"
        ) from error
    cache = DynamicCache(config=backbone.config)
    try:
        install = install_native_functional_linear_cache(cache, backbone.config)
    except NativeFunctionalCacheError as error:
        raise Qwen35FunctionalStackError(str(error)) from error
    return Qwen35NativeFunctionalState(
        plan=plan,
        cache=cache,
        install=install,
    )


def _native_linear_snapshot(
    state: Qwen35NativeFunctionalState,
) -> dict[int, tuple[torch.Tensor | None, torch.Tensor | None]]:
    snapshot = {}
    for index in state.plan.linear_layer_indices:
        layer = state.cache.layers[index]
        conv = getattr(layer, "conv_states", {}).get(0)
        recurrent = getattr(layer, "recurrent_states", {}).get(0)
        snapshot[index] = (
            conv if isinstance(conv, torch.Tensor) else None,
            recurrent if isinstance(recurrent, torch.Tensor) else None,
        )
    return snapshot


def _native_full_lengths(
    state: Qwen35NativeFunctionalState,
) -> dict[int, int]:
    lengths = {}
    for index in state.plan.full_attention_layer_indices:
        layer = state.cache.layers[index]
        get_length = getattr(layer, "get_seq_length", None)
        if not callable(get_length):
            raise Qwen35FunctionalStackError(
                f"native full cache layer {index} has no get_seq_length"
            )
        lengths[index] = int(get_length())
    return lengths


def native_qwen35_functional_forward(
    model: Any,
    input_ids: torch.Tensor,
    state: Qwen35NativeFunctionalState | None = None,
) -> Qwen35NativeFunctionalPass:
    """Run the unchanged real Qwen decoder through the functional cache seam.

    Every configured linear layer must replace both state tensor objects, and
    every configured full-attention layer must extend by the same token count.
    These are per-call hard gates rather than inferred coverage claims.
    """

    input_ids = _validate_tokens(input_ids)
    if state is None:
        state = new_qwen35_native_functional_state(model)
    plan = audit_qwen35_functional_stack_plan(model)
    if state.plan.gdn.layer_types != plan.gdn.layer_types:
        raise Qwen35FunctionalStackError(
            "native functional state plan differs from loaded model"
        )
    try:
        install_telemetry = functional_linear_cache_telemetry(
            state.cache, state.install
        )
    except NativeFunctionalCacheError as error:
        raise Qwen35FunctionalStackError(str(error)) from error
    if not install_telemetry.get("all_linear_layers_intercepted"):
        raise Qwen35FunctionalStackError(
            "not every linear cache layer retains the native functional seam"
        )

    before_linear = _native_linear_snapshot(state)
    before_full = _native_full_lengths(state)
    backbone = resolve_qwen35_text_backbone(model)
    output = backbone(
        input_ids=input_ids,
        past_key_values=state.cache,
        use_cache=True,
    )
    hidden = getattr(output, "last_hidden_state", None)
    returned_cache = getattr(output, "past_key_values", None)
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise Qwen35FunctionalStackError(
            "native Qwen forward did not return rank-3 last_hidden_state"
        )
    if returned_cache is not state.cache:
        raise Qwen35FunctionalStackError(
            "native Qwen forward returned a different cache object"
        )

    linear_rows = []
    for index in plan.linear_layer_indices:
        layer = state.cache.layers[index]
        after_conv = getattr(layer, "conv_states", {}).get(0)
        after_recurrent = getattr(layer, "recurrent_states", {}).get(0)
        if not isinstance(after_conv, torch.Tensor) or not isinstance(
            after_recurrent, torch.Tensor
        ):
            raise Qwen35FunctionalStackError(
                f"native linear layer {index} did not initialize both states"
            )
        before_conv, before_recurrent = before_linear[index]
        if before_conv is not None and after_conv is before_conv:
            raise Qwen35FunctionalStackError(
                f"native linear layer {index} reused its conv tensor object"
            )
        if before_recurrent is not None and after_recurrent is before_recurrent:
            raise Qwen35FunctionalStackError(
                f"native linear layer {index} reused its recurrent tensor object"
            )
        if getattr(layer, "_qcomem_update_mode", None) != "functional-state-rebind":
            raise Qwen35FunctionalStackError(
                f"native linear layer {index} lost functional update mode"
            )
        linear_rows.append(
            {
                "layer_idx": index,
                "conv_rebound": before_conv is None or after_conv is not before_conv,
                "recurrent_rebound": (
                    before_recurrent is None or after_recurrent is not before_recurrent
                ),
                "conv_requires_grad": after_conv.requires_grad,
                "recurrent_requires_grad": after_recurrent.requires_grad,
            }
        )

    expected_length = state.current_length + int(input_ids.shape[1])
    after_full = _native_full_lengths(state)
    full_rows = []
    for index in plan.full_attention_layer_indices:
        if after_full[index] != expected_length:
            raise Qwen35FunctionalStackError(
                f"native full layer {index} length {after_full[index]} != {expected_length}"
            )
        if after_full[index] - before_full[index] != int(input_ids.shape[1]):
            raise Qwen35FunctionalStackError(
                f"native full layer {index} did not extend by the caller chunk"
            )
        layer = state.cache.layers[index]
        if not isinstance(getattr(layer, "keys", None), torch.Tensor) or not isinstance(
            getattr(layer, "values", None), torch.Tensor
        ):
            raise Qwen35FunctionalStackError(
                f"native full layer {index} lost dense DynamicLayer K/V"
            )
        full_rows.append(
            {
                "layer_idx": index,
                "length_before": before_full[index],
                "length_after": after_full[index],
                "tokens_appended": int(input_ids.shape[1]),
                "update_mode": state.install.full_attention_update_mode,
            }
        )

    next_state = Qwen35NativeFunctionalState(
        plan=state.plan,
        cache=state.cache,
        install=state.install,
        current_length=expected_length,
        passes=state.passes + 1,
    )
    telemetry = {
        "verified": True,
        "kernel_mode": "native-qwen-kernels-functional-cache-rebind",
        "production_ttft_optimization_claim_allowed": False,
        "tokens": int(input_ids.shape[1]),
        "expected_linear_layer_count": len(plan.linear_layer_indices),
        "observed_linear_layer_count": len(linear_rows),
        "expected_full_attention_layer_count": len(
            plan.full_attention_layer_indices
        ),
        "observed_full_attention_layer_count": len(full_rows),
        "linear_layers": tuple(linear_rows),
        "full_attention_layers": tuple(full_rows),
        "install": install_telemetry,
        "mutable_linear_copy_updates_used": False,
        "fallback_layers": (),
    }
    return Qwen35NativeFunctionalPass(
        last_hidden_state=hidden,
        state=next_state,
        telemetry=telemetry,
    )


def prefill_qwen35_functional_document(
    model: Any,
    input_ids: torch.Tensor,
    *,
    gradient_semantics: GradientSemantics,
    training_mode: bool,
    attention_bits: int | None = None,
    page_size: int = 128,
    group_size: int = 64,
    append_page_size: int = 16,
) -> Qwen35FunctionalPass:
    """Prefill a document through all layers without a mutable cache.

    ``autograd-preserving`` retains the document graph in every GDN base and
    K/V page.  ``inference-detached`` creates an immutable detached document;
    when ``training_mode=True`` its later query pages are still differentiable.
    """

    input_ids = _validate_tokens(input_ids)
    if gradient_semantics == AUTOGRAD_PRESERVING and not training_mode:
        raise Qwen35FunctionalStackError(
            "autograd-preserving document state requires training_mode=True"
        )
    if training_mode and attention_bits not in (None, 16):
        raise Qwen35FunctionalStackError(
            "training functional prefill does not detach through sub-16-bit quantization"
        )
    plan = audit_qwen35_functional_stack_plan(model)
    backbone = resolve_qwen35_text_backbone(model)
    hidden_states = backbone.embed_tokens(input_ids)
    _, position_embeddings = _position_context(backbone, hidden_states, offset=0)
    full_mask = _causal_mask(hidden_states, past_length=0)
    gdn_bases: dict[int, ImmutableGDNBase] = {}
    full_layers: dict[int, PagedKVLayer] = {}
    telemetry_rows: list[dict[str, Any]] = []

    for index, decoder_layer in enumerate(backbone.layers):
        block_type = plan.gdn.layer_types[index]
        if block_type == "linear_attention":
            zero = zero_gdn_base(
                decoder_layer.linear_attn,
                hidden_states,
                gradient_semantics=gradient_semantics,
            )
            hidden_states, next_state, telemetry = dispatch_qwen35_decoder_layer(
                decoder_layer,
                hidden_states,
                zero,
                layer_idx=index,
                attention_mask=None,
            )
            if not isinstance(next_state, QueryLocalGDNState):
                raise Qwen35FunctionalStackError(
                    f"linear layer {index} did not return query-local state"
                )
            gdn_bases[index] = next_state.promote_to_base(
                gradient_semantics=gradient_semantics,
                source="functional-stack-document-prefill",
            )
            row = telemetry.as_dict()
            row["kind"] = "linear_attention"
            telemetry_rows.append(row)
            continue

        def full_dispatch(
            layer: Any,
            values: torch.Tensor,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, PagedKVLayer, dict[str, Any]]:
            return _prefill_full_attention_dispatch(
                layer,
                values,
                layer_idx=int(kwargs["layer_idx"]),
                attention_mask=kwargs["attention_mask"],
                position_embeddings=kwargs["position_embeddings"],
                training_mode=training_mode,
                preserve_document_graph=(
                    gradient_semantics == AUTOGRAD_PRESERVING
                ),
                attention_bits=attention_bits,
                page_size=page_size,
                group_size=group_size,
                append_page_size=append_page_size,
            )

        hidden_states, next_layer, telemetry = dispatch_qwen35_decoder_layer(
            decoder_layer,
            hidden_states,
            object(),
            layer_idx=index,
            attention_mask=full_mask,
            position_embeddings=position_embeddings,
            full_attention_dispatch=full_dispatch,
        )
        if not isinstance(next_layer, PagedKVLayer):
            raise Qwen35FunctionalStackError(
                f"full layer {index} did not return paged state"
            )
        full_layers[index] = next_layer
        telemetry_rows.append(telemetry)

    hidden_states = backbone.norm(hidden_states)
    coverage = _verify_pass_coverage(plan, telemetry_rows, phase="document_prefill")
    document = Qwen35FunctionalDocumentState(
        plan=plan,
        document_length=int(input_ids.shape[1]),
        gdn_bases=gdn_bases,
        full_layers=full_layers,
        gradient_semantics=gradient_semantics,
        training_mode=training_mode,
        attention_bits=attention_bits,
        page_size=page_size,
        group_size=group_size,
        append_page_size=append_page_size,
        source="all-layer-functional-prefill",
    )
    return Qwen35FunctionalPass(
        last_hidden_state=hidden_states,
        state=document,
        telemetry={
            "coverage": coverage,
            "memory": document.memory_report(),
        },
    )


def capture_qwen35_document_from_mutable_cache(
    model: Any,
    source_cache: Any,
    *,
    document_length: int,
    training_mode: bool,
    attention_bits: int | None = 16,
    page_size: int = 128,
    group_size: int = 64,
    append_page_size: int = 16,
) -> Qwen35FunctionalDocumentState:
    """Capture a one-time standard prefill for detached inference/training.

    This path is deliberately detached: mutable Transformers caches cannot be
    represented as an autograd-preserving training graph.  It is the practical
    4K deployment path and the detached-document LoRA path.  Query continuation
    remains fully functional, and training mode keeps query K/V append graphs.
    """

    if document_length < 1:
        raise Qwen35FunctionalStackError("document_length must be positive")
    if training_mode and attention_bits not in (None, 16):
        raise Qwen35FunctionalStackError(
            "detached-document training supports only dense/16-bit K/V"
        )
    plan = audit_qwen35_functional_stack_plan(model)
    backbone = resolve_qwen35_text_backbone(model)
    cache_layers = getattr(source_cache, "layers", None)
    if cache_layers is None or len(cache_layers) != plan.total_layers:
        raise Qwen35FunctionalStackError(
            "source mutable cache does not expose every configured layer"
        )
    gdn_bases = {}
    for index in plan.linear_layer_indices:
        try:
            gdn_bases[index] = immutable_base_from_transformers_cache(
                backbone.layers[index].linear_attn,
                source_cache,
                gradient_semantics=INFERENCE_DETACHED,
                layer_idx=index,
            )
        except GDNContractError as error:
            raise Qwen35FunctionalStackError(str(error)) from error

    full_layers: dict[int, PagedKVLayer] = {}
    for index in plan.full_attention_layer_indices:
        source = cache_layers[index]
        key = getattr(source, "keys", None)
        value = getattr(source, "values", None)
        if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise Qwen35FunctionalStackError(
                f"source full cache layer {index} does not contain dense K/V"
            )
        if key.shape[-2] != document_length or value.shape[-2] != document_length:
            raise Qwen35FunctionalStackError(
                f"source full cache layer {index} length differs from document"
            )
        full_layers[index] = _new_document_page_layer(
            key,
            value,
            training_mode=training_mode,
            preserve_document_graph=False,
            attention_bits=attention_bits,
            page_size=page_size,
            group_size=group_size,
            append_page_size=append_page_size,
        )
    document = Qwen35FunctionalDocumentState(
        plan=plan,
        document_length=document_length,
        gdn_bases=gdn_bases,
        full_layers=full_layers,
        gradient_semantics=INFERENCE_DETACHED,
        training_mode=training_mode,
        attention_bits=attention_bits,
        page_size=page_size,
        group_size=group_size,
        append_page_size=append_page_size,
        source="captured-standard-prefill-detached",
    )
    document.assert_unchanged()
    return document


def continue_qwen35_functional(
    model: Any,
    input_ids: torch.Tensor,
    state: Qwen35FunctionalDocumentState | Qwen35FunctionalRequestState,
) -> Qwen35FunctionalPass:
    """Continue every decoder layer with paged/GDN out-of-place state."""

    input_ids = _validate_tokens(input_ids)
    plan = audit_qwen35_functional_stack_plan(model)
    document = state if isinstance(state, Qwen35FunctionalDocumentState) else state.document
    if plan.gdn.layer_types != document.plan.gdn.layer_types:
        raise Qwen35FunctionalStackError("state plan differs from loaded model")
    request = document.fork() if isinstance(state, Qwen35FunctionalDocumentState) else state
    document.assert_unchanged()
    backbone = resolve_qwen35_text_backbone(model)
    hidden_states = backbone.embed_tokens(input_ids)
    text_positions, position_embeddings = _position_context(
        backbone, hidden_states, offset=request.current_length
    )
    full_mask = _causal_mask(hidden_states, past_length=request.current_length)
    next_gdn: dict[int, QueryLocalGDNState] = {}
    next_full: dict[int, PagedKVLayer] = {}
    telemetry_rows: list[dict[str, Any]] = []

    for index, decoder_layer in enumerate(backbone.layers):
        if plan.gdn.layer_types[index] == "linear_attention":
            hidden_states, next_state, telemetry = dispatch_qwen35_decoder_layer(
                decoder_layer,
                hidden_states,
                request.gdn_states[index],
                layer_idx=index,
                attention_mask=None,
            )
            if not isinstance(next_state, QueryLocalGDNState):
                raise Qwen35FunctionalStackError(
                    f"linear continuation layer {index} returned invalid state"
                )
            next_gdn[index] = next_state
            row = telemetry.as_dict()
            row["kind"] = "linear_attention"
            telemetry_rows.append(row)
            continue

        def full_dispatch(
            layer: Any,
            values: torch.Tensor,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, PagedKVLayer, dict[str, Any]]:
            return _continue_full_attention_dispatch(
                layer,
                values,
                kwargs["layer_state"],
                layer_idx=int(kwargs["layer_idx"]),
                attention_mask=kwargs["attention_mask"],
                position_embeddings=kwargs["position_embeddings"],
            )

        hidden_states, next_layer, telemetry = dispatch_qwen35_decoder_layer(
            decoder_layer,
            hidden_states,
            request.full_layers[index],
            layer_idx=index,
            attention_mask=full_mask,
            position_ids=text_positions,
            position_embeddings=position_embeddings,
            full_attention_dispatch=full_dispatch,
        )
        if not isinstance(next_layer, PagedKVLayer):
            raise Qwen35FunctionalStackError(
                f"full continuation layer {index} returned invalid state"
            )
        next_full[index] = next_layer
        telemetry_rows.append(telemetry)

    hidden_states = backbone.norm(hidden_states)
    coverage = _verify_pass_coverage(plan, telemetry_rows, phase="continuation")
    next_request = Qwen35FunctionalRequestState(
        document=document,
        current_length=request.current_length + int(input_ids.shape[1]),
        gdn_states=next_gdn,
        full_layers=next_full,
    )
    document.assert_unchanged()
    return Qwen35FunctionalPass(
        last_hidden_state=hidden_states,
        state=next_request,
        telemetry={
            "coverage": coverage,
            "memory": next_request.memory_report(),
        },
    )


def qwen35_functional_logits(
    model: Any,
    value: Qwen35FunctionalPass | Qwen35NativeFunctionalPass | torch.Tensor,
    *,
    last_token_only: bool = False,
) -> torch.Tensor:
    hidden = (
        value.last_hidden_state
        if isinstance(value, (Qwen35FunctionalPass, Qwen35NativeFunctionalPass))
        else value
    )
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise Qwen35FunctionalStackError("hidden states must have shape [batch,tokens,width]")
    lm_head = getattr(model, "lm_head", None)
    if not callable(lm_head):
        raise Qwen35FunctionalStackError("model has no callable lm_head")
    if last_token_only:
        hidden = hidden[:, -1:, :]
    return lm_head(hidden)


__all__ = [
    "Qwen35FunctionalDocumentState",
    "Qwen35FunctionalPass",
    "Qwen35FunctionalRequestState",
    "Qwen35FunctionalStackError",
    "Qwen35FunctionalStackPlan",
    "Qwen35NativeFunctionalPass",
    "Qwen35NativeFunctionalState",
    "audit_qwen35_functional_stack_plan",
    "capture_qwen35_document_from_mutable_cache",
    "continue_qwen35_functional",
    "native_qwen35_functional_forward",
    "new_qwen35_native_functional_state",
    "prefill_qwen35_functional_document",
    "qwen35_functional_logits",
    "resolve_qwen35_text_backbone",
]
