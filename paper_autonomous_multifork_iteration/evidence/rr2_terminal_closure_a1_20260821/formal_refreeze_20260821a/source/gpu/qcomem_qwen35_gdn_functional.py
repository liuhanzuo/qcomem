from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch.nn import functional as F


AUTOGRAD_PRESERVING = "autograd-preserving"
INFERENCE_DETACHED = "inference-detached"
GradientSemantics = Literal["autograd-preserving", "inference-detached"]


class GDNContractError(ValueError):
    """Raised when a Qwen3.5 module or state cannot be adapted safely."""


def _tensor_storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _unique_storage_nbytes(tensors: tuple[torch.Tensor, ...]) -> int:
    storages = {_tensor_storage_key(tensor) for tensor in tensors}
    return sum(storage_nbytes for _, _, storage_nbytes in storages)


def _sample_fingerprint(tensor: torch.Tensor) -> tuple[float, ...]:
    if tensor.numel() == 0:
        return ()
    flat = tensor.detach().reshape(-1)
    count = min(flat.numel(), 16)
    indices = torch.linspace(
        0, flat.numel() - 1, count, device=flat.device
    ).round().long()
    return tuple(float(value) for value in flat[indices].float().cpu().tolist())


@dataclass(frozen=True)
class _TensorGuard:
    label: str
    tensor: torch.Tensor
    storage_key: tuple[str, int, int]
    version: int | None
    sample: tuple[float, ...]

    @classmethod
    def capture(cls, label: str, tensor: torch.Tensor) -> "_TensorGuard":
        try:
            version = tensor._version
        except RuntimeError:
            # Inference tensors intentionally do not expose version counters.
            version = None
        return cls(
            label=label,
            tensor=tensor,
            storage_key=_tensor_storage_key(tensor),
            version=version,
            sample=_sample_fingerprint(tensor),
        )

    def verify(self) -> str | None:
        if _tensor_storage_key(self.tensor) != self.storage_key:
            return f"{self.label} storage binding changed"
        if self.version is not None:
            try:
                version = self.tensor._version
            except RuntimeError:
                return f"{self.label} version counter became unavailable"
            if version != self.version:
                return f"{self.label} version changed ({self.version} -> {version})"
        if _sample_fingerprint(self.tensor) != self.sample:
            return f"{self.label} sampled values changed"
        return None


def _validate_gradient_semantics(value: str) -> GradientSemantics:
    if value not in (AUTOGRAD_PRESERVING, INFERENCE_DETACHED):
        raise GDNContractError(
            "gradient_semantics must be 'autograd-preserving' or "
            "'inference-detached'"
        )
    return value  # type: ignore[return-value]


@dataclass(frozen=True)
class ImmutableGDNBase:
    """Document GatedDeltaNet state that query transitions may only read.

    Construction always clones both tensors.  ``autograd-preserving`` uses a
    normal clone and therefore keeps the document graph.  ``inference-detached``
    deliberately severs it before cloning.  The distinction is metadata, not
    an inference made from ``requires_grad`` (a valid graph can contain tensors
    that do not themselves require gradients).
    """

    conv_state: torch.Tensor
    recurrent_state: torch.Tensor
    gradient_semantics: GradientSemantics
    source: str = "functional-prefill"
    _guards: tuple[_TensorGuard, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        semantics = _validate_gradient_semantics(self.gradient_semantics)
        if not isinstance(self.conv_state, torch.Tensor) or not isinstance(
            self.recurrent_state, torch.Tensor
        ):
            raise GDNContractError("conv_state and recurrent_state must be tensors")
        if not self.conv_state.is_floating_point() or not self.recurrent_state.is_floating_point():
            raise GDNContractError("GDN base tensors must be floating point")
        if not isinstance(self.source, str) or not self.source:
            raise GDNContractError("base source metadata must be a non-empty string")

        if semantics == AUTOGRAD_PRESERVING:
            conv_state = self.conv_state.clone()
            recurrent_state = self.recurrent_state.clone()
        else:
            conv_state = self.conv_state.detach().clone()
            recurrent_state = self.recurrent_state.detach().clone()
        object.__setattr__(self, "conv_state", conv_state)
        object.__setattr__(self, "recurrent_state", recurrent_state)
        object.__setattr__(
            self,
            "_guards",
            (
                _TensorGuard.capture("base.conv_state", conv_state),
                _TensorGuard.capture("base.recurrent_state", recurrent_state),
            ),
        )

    @property
    def preserves_autograd(self) -> bool:
        return self.gradient_semantics == AUTOGRAD_PRESERVING

    @property
    def nbytes(self) -> int:
        return _unique_storage_nbytes((self.conv_state, self.recurrent_state))

    def assert_unchanged(self) -> None:
        failures = [failure for guard in self._guards if (failure := guard.verify())]
        if failures:
            raise GDNContractError(
                "immutable GDN base was modified: " + "; ".join(failures)
            )


@dataclass(frozen=True)
class QueryLocalGDNState:
    """A base reference plus optional query-owned replacement state tensors.

    ``conv_delta`` and ``recurrent_delta`` are functional deltas in the state
    transition sense: after the first query transition they replace, rather
    than mutate or numerically add to, the corresponding base tensors.
    """

    base: ImmutableGDNBase
    conv_delta: torch.Tensor | None = None
    recurrent_delta: torch.Tensor | None = None
    tokens_processed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.base, ImmutableGDNBase):
            raise GDNContractError("query-local state requires an ImmutableGDNBase")
        if (self.conv_delta is None) != (self.recurrent_delta is None):
            raise GDNContractError(
                "conv_delta and recurrent_delta must either both be absent or both be present"
            )
        if not isinstance(self.tokens_processed, int) or self.tokens_processed < 0:
            raise GDNContractError("tokens_processed must be a non-negative integer")
        if self.conv_delta is None:
            if self.tokens_processed != 0:
                raise GDNContractError("a state without deltas cannot have processed tokens")
            return
        assert self.recurrent_delta is not None
        if not self.conv_delta.is_floating_point() or not self.recurrent_delta.is_floating_point():
            raise GDNContractError("query-local GDN tensors must be floating point")
        base_keys = {
            _tensor_storage_key(self.base.conv_state),
            _tensor_storage_key(self.base.recurrent_state),
        }
        delta_keys = {
            _tensor_storage_key(self.conv_delta),
            _tensor_storage_key(self.recurrent_delta),
        }
        if base_keys & delta_keys:
            raise GDNContractError("query-local delta aliases immutable base storage")

    @classmethod
    def from_base(cls, base: ImmutableGDNBase) -> "QueryLocalGDNState":
        return cls(base=base)

    @property
    def conv_state(self) -> torch.Tensor:
        return self.base.conv_state if self.conv_delta is None else self.conv_delta

    @property
    def recurrent_state(self) -> torch.Tensor:
        return (
            self.base.recurrent_state
            if self.recurrent_delta is None
            else self.recurrent_delta
        )

    @property
    def shared_base_nbytes(self) -> int:
        return self.base.nbytes

    @property
    def query_private_nbytes(self) -> int:
        if self.conv_delta is None:
            return 0
        assert self.recurrent_delta is not None
        return _unique_storage_nbytes((self.conv_delta, self.recurrent_delta))

    def memory_report(self) -> dict[str, int | str]:
        return {
            "gradient_semantics": self.base.gradient_semantics,
            "shared_base_nbytes": self.shared_base_nbytes,
            "query_private_nbytes": self.query_private_nbytes,
            "total_referenced_nbytes": (
                self.shared_base_nbytes + self.query_private_nbytes
            ),
        }

    def promote_to_base(
        self,
        *,
        gradient_semantics: GradientSemantics,
        source: str = "functional-prefill",
    ) -> ImmutableGDNBase:
        """Freeze the current state as a separately owned document base."""

        return ImmutableGDNBase(
            conv_state=self.conv_state,
            recurrent_state=self.recurrent_state,
            gradient_semantics=gradient_semantics,
            source=source,
        )


@dataclass(frozen=True)
class Qwen35GDNSpec:
    hidden_size: int
    num_v_heads: int
    num_k_heads: int
    head_k_dim: int
    head_v_dim: int
    key_dim: int
    value_dim: int
    conv_dim: int
    conv_kernel_size: int
    layer_idx: int
    activation: str

    @property
    def recurrent_shape_tail(self) -> tuple[int, int, int]:
        return self.num_v_heads, self.head_k_dim, self.head_v_dim


@dataclass(frozen=True)
class Qwen35GDNDispatchPlan:
    """Audited decoder routing derived from ``config.layer_types``."""

    total_layers: int
    layer_types: tuple[str, ...]
    linear_layer_indices: tuple[int, ...]
    full_attention_layer_indices: tuple[int, ...]

    @property
    def linear_layer_count(self) -> int:
        return len(self.linear_layer_indices)

    def validate_linear_states(self, states: Mapping[int, object]) -> None:
        missing = [index for index in self.linear_layer_indices if index not in states]
        if missing:
            raise GDNContractError(
                f"functional GDN state is missing for linear layers {missing}"
            )
        invalid = [
            index
            for index in self.linear_layer_indices
            if not isinstance(states[index], (ImmutableGDNBase, QueryLocalGDNState))
        ]
        if invalid:
            raise GDNContractError(
                f"functional GDN state has an invalid type for linear layers {invalid}"
            )


@dataclass(frozen=True)
class GDNDispatchTelemetry:
    layer_idx: int
    route: str
    kernel: str
    module_class: str
    sequence_length: int
    state_tokens_before: int
    state_tokens_after: int
    gradient_semantics: str
    base_source: str
    shared_base_nbytes: int
    query_private_nbytes: int
    mutable_cache_used: bool = False
    fallback_used: bool = False

    def as_dict(self) -> dict[str, int | str | bool]:
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }


def _require_positive_int(module: Any, name: str, *, allow_zero: bool = False) -> int:
    if not hasattr(module, name):
        raise GDNContractError(f"Qwen3.5 GDN module is missing {name!r}")
    value = getattr(module, name)
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise GDNContractError(f"module.{name} must be a {qualifier} integer")
    return value


def _require_projection(
    module: Any,
    name: str,
    *,
    out_features: int,
    in_features: int,
) -> None:
    if not hasattr(module, name):
        raise GDNContractError(f"Qwen3.5 GDN module is missing {name!r}")
    projection = getattr(module, name)
    if not callable(projection):
        raise GDNContractError(f"module.{name} must be callable")
    weight = getattr(projection, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise GDNContractError(f"module.{name}.weight must be a tensor")
    expected = (out_features, in_features)
    if tuple(weight.shape) != expected:
        raise GDNContractError(
            f"module.{name}.weight shape {tuple(weight.shape)} != {expected}"
        )
    bias = getattr(projection, "bias", None)
    if bias is not None and (
        not isinstance(bias, torch.Tensor) or tuple(bias.shape) != (out_features,)
    ):
        raise GDNContractError(
            f"module.{name}.bias must be absent or have shape {(out_features,)}"
        )


def audit_qwen35_gdn_module(module: Any) -> Qwen35GDNSpec:
    """Validate the public attributes used by Qwen3.5 GatedDeltaNet.

    This intentionally uses structural checks rather than importing one exact
    Transformers class, so dense and MoE Qwen3.5 modules and LoRA-wrapped
    projections can share the adapter.  Missing or inconsistent fields are not
    guessed.
    """

    hidden_size = _require_positive_int(module, "hidden_size")
    num_v_heads = _require_positive_int(module, "num_v_heads")
    num_k_heads = _require_positive_int(module, "num_k_heads")
    head_k_dim = _require_positive_int(module, "head_k_dim")
    head_v_dim = _require_positive_int(module, "head_v_dim")
    key_dim = _require_positive_int(module, "key_dim")
    value_dim = _require_positive_int(module, "value_dim")
    conv_dim = _require_positive_int(module, "conv_dim")
    conv_kernel_size = _require_positive_int(module, "conv_kernel_size")
    layer_idx = _require_positive_int(module, "layer_idx", allow_zero=True)

    if key_dim != num_k_heads * head_k_dim:
        raise GDNContractError("module.key_dim is inconsistent with key heads")
    if value_dim != num_v_heads * head_v_dim:
        raise GDNContractError("module.value_dim is inconsistent with value heads")
    if conv_dim != 2 * key_dim + value_dim:
        raise GDNContractError("module.conv_dim must equal 2 * key_dim + value_dim")
    if num_v_heads % num_k_heads:
        raise GDNContractError("value heads must be divisible by key heads")
    if getattr(module, "layer_type", None) != "linear_attention":
        raise GDNContractError("module.layer_type must be 'linear_attention'")
    activation = getattr(module, "activation", None)
    if activation not in ("silu", "swish"):
        raise GDNContractError(
            "functional Qwen3.5 GDN currently supports only silu/swish activation"
        )

    _require_projection(
        module,
        "in_proj_qkv",
        out_features=conv_dim,
        in_features=hidden_size,
    )
    _require_projection(
        module,
        "in_proj_z",
        out_features=value_dim,
        in_features=hidden_size,
    )
    _require_projection(
        module,
        "in_proj_b",
        out_features=num_v_heads,
        in_features=hidden_size,
    )
    _require_projection(
        module,
        "in_proj_a",
        out_features=num_v_heads,
        in_features=hidden_size,
    )
    _require_projection(
        module,
        "out_proj",
        out_features=hidden_size,
        in_features=value_dim,
    )

    conv1d = getattr(module, "conv1d", None)
    if conv1d is None or not callable(conv1d):
        raise GDNContractError("module.conv1d must be a callable depthwise convolution")
    conv_weight = getattr(conv1d, "weight", None)
    expected_conv_weight = (conv_dim, 1, conv_kernel_size)
    if not isinstance(conv_weight, torch.Tensor) or tuple(conv_weight.shape) != expected_conv_weight:
        actual = None if not isinstance(conv_weight, torch.Tensor) else tuple(conv_weight.shape)
        raise GDNContractError(
            f"module.conv1d.weight shape {actual} != {expected_conv_weight}"
        )
    conv_bias = getattr(conv1d, "bias", None)
    if conv_bias is not None and (
        not isinstance(conv_bias, torch.Tensor) or tuple(conv_bias.shape) != (conv_dim,)
    ):
        raise GDNContractError(
            f"module.conv1d.bias must be absent or have shape {(conv_dim,)}"
        )

    for name in ("dt_bias", "A_log"):
        value = getattr(module, name, None)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (num_v_heads,):
            actual = None if not isinstance(value, torch.Tensor) else tuple(value.shape)
            raise GDNContractError(
                f"module.{name} shape {actual} != {(num_v_heads,)}"
            )
    if not callable(getattr(module, "norm", None)):
        raise GDNContractError("module.norm must be callable")

    return Qwen35GDNSpec(
        hidden_size=hidden_size,
        num_v_heads=num_v_heads,
        num_k_heads=num_k_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
        key_dim=key_dim,
        value_dim=value_dim,
        conv_dim=conv_dim,
        conv_kernel_size=conv_kernel_size,
        layer_idx=layer_idx,
        activation=activation,
    )


def _validate_hidden(
    hidden_states: torch.Tensor,
    spec: Qwen35GDNSpec,
    attention_mask: torch.Tensor | None,
) -> None:
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise GDNContractError("hidden_states must have shape [batch,tokens,hidden]")
    if hidden_states.shape[-1] != spec.hidden_size:
        raise GDNContractError(
            f"hidden width {hidden_states.shape[-1]} != {spec.hidden_size}"
        )
    if hidden_states.shape[0] < 1 or hidden_states.shape[1] < 1:
        raise GDNContractError("functional GDN requires non-empty batch and tokens")
    if not hidden_states.is_floating_point():
        raise GDNContractError("hidden_states must be floating point")
    if attention_mask is not None:
        expected = tuple(hidden_states.shape[:2])
        if not isinstance(attention_mask, torch.Tensor) or tuple(attention_mask.shape) != expected:
            raise GDNContractError(f"attention_mask shape must be {expected}")
        if attention_mask.device != hidden_states.device:
            raise GDNContractError("attention_mask and hidden_states must be on one device")


def _validate_state(
    state: QueryLocalGDNState,
    spec: Qwen35GDNSpec,
    hidden_states: torch.Tensor,
) -> None:
    expected_conv = (
        hidden_states.shape[0],
        spec.conv_dim,
        spec.conv_kernel_size,
    )
    expected_recurrent = (hidden_states.shape[0], *spec.recurrent_shape_tail)
    if tuple(state.conv_state.shape) != expected_conv:
        raise GDNContractError(
            f"conv state shape {tuple(state.conv_state.shape)} != {expected_conv}"
        )
    if tuple(state.recurrent_state.shape) != expected_recurrent:
        raise GDNContractError(
            "recurrent state shape "
            f"{tuple(state.recurrent_state.shape)} != {expected_recurrent}"
        )
    for label, tensor in (
        ("conv", state.conv_state),
        ("recurrent", state.recurrent_state),
    ):
        if tensor.device != hidden_states.device:
            raise GDNContractError(
                f"{label} state device {tensor.device} != hidden device {hidden_states.device}"
            )
        if not tensor.is_floating_point():
            raise GDNContractError(f"{label} state must be floating point")


def zero_gdn_base(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    gradient_semantics: GradientSemantics = AUTOGRAD_PRESERVING,
) -> ImmutableGDNBase:
    """Create the explicit zero state used before a first document token."""

    spec = audit_qwen35_gdn_module(module)
    _validate_hidden(hidden_states, spec, None)
    conv_state = hidden_states.new_zeros(
        hidden_states.shape[0], spec.conv_dim, spec.conv_kernel_size
    )
    # Transformers' torch gated-delta fallbacks accumulate recurrent state in
    # float32 even when hidden/conv inputs are bf16.
    recurrent_state = torch.zeros(
        hidden_states.shape[0],
        *spec.recurrent_shape_tail,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    return ImmutableGDNBase(
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        gradient_semantics=gradient_semantics,
        source="zero-state",
    )


def _functional_causal_conv(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    module: Any,
    spec: Qwen35GDNSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.cat((conv_state, mixed_qkv), dim=-1)
    weight = module.conv1d.weight.squeeze(1)
    convolved = F.conv1d(
        combined.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=module.conv1d.bias,
        padding=0,
        groups=spec.conv_dim,
    )
    convolved = convolved[:, :, -mixed_qkv.shape[-1] :]
    if spec.activation in ("silu", "swish"):
        convolved = F.silu(convolved)
    # Clone compacts the tail. A slice alone would retain the whole temporary
    # concatenation storage and make query-private memory scale with tokens.
    next_conv_state = combined[:, :, -spec.conv_kernel_size :].clone()
    return convolved.to(mixed_qkv.dtype), next_conv_state


def _l2norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return value * torch.rsqrt((value * value).sum(dim=-1, keepdim=True) + eps)


def _functional_recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3.5 torch recurrent rule without output-buffer or state mutation."""

    initial_dtype = query.dtype
    query, key, value, beta, g = (
        tensor.transpose(1, 2).contiguous().to(torch.float32)
        for tensor in (query, key, value, beta, g)
    )
    query = query * (query.shape[-1] ** -0.5)
    recurrent = initial_state.to(value)
    outputs = []
    for position in range(query.shape[2]):
        query_t = query[:, :, position]
        key_t = key[:, :, position]
        value_t = value[:, :, position]
        decay_t = g[:, :, position].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, position].unsqueeze(-1)

        recurrent = recurrent * decay_t
        memory = (recurrent * key_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value_t - memory) * beta_t
        recurrent = recurrent + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(
            (recurrent * query_t.unsqueeze(-1)).sum(dim=-2)
        )
    output = torch.stack(outputs, dim=2).transpose(1, 2).contiguous()
    return output.to(initial_dtype), recurrent


def functional_qwen35_gdn_forward(
    module: Any,
    hidden_states: torch.Tensor,
    state: QueryLocalGDNState | ImmutableGDNBase | None = None,
    *,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, QueryLocalGDNState]:
    """Run one Qwen3.5 GDN segment and return a query-owned next state.

    The adapter consumes the attributes of ``Qwen3_5GatedDeltaNet`` or
    ``Qwen3_5MoeGatedDeltaNet`` but never calls ``DynamicCache``,
    ``causal_conv1d_update``, or either cache ``update_*`` method.
    """

    spec = audit_qwen35_gdn_module(module)
    _validate_hidden(hidden_states, spec, attention_mask)
    if state is None:
        base = zero_gdn_base(module, hidden_states)
        current = QueryLocalGDNState.from_base(base)
    elif isinstance(state, ImmutableGDNBase):
        current = QueryLocalGDNState.from_base(state)
    elif isinstance(state, QueryLocalGDNState):
        current = state
    else:
        raise GDNContractError(
            "state must be None, ImmutableGDNBase, or QueryLocalGDNState"
        )
    current.base.assert_unchanged()
    _validate_state(current, spec, hidden_states)

    if attention_mask is not None:
        hidden_states = (
            hidden_states * attention_mask[:, :, None]
        ).to(hidden_states.dtype)

    batch_size, sequence_length, _ = hidden_states.shape
    mixed_qkv = module.in_proj_qkv(hidden_states).transpose(1, 2)
    expected_mixed = (batch_size, spec.conv_dim, sequence_length)
    if tuple(mixed_qkv.shape) != expected_mixed:
        raise GDNContractError(
            f"in_proj_qkv output shape {tuple(mixed_qkv.shape)} != {expected_mixed}"
        )
    z = module.in_proj_z(hidden_states)
    b = module.in_proj_b(hidden_states)
    a = module.in_proj_a(hidden_states)
    expected_z = (batch_size, sequence_length, spec.value_dim)
    expected_ab = (batch_size, sequence_length, spec.num_v_heads)
    if tuple(z.shape) != expected_z:
        raise GDNContractError(f"in_proj_z output shape {tuple(z.shape)} != {expected_z}")
    if tuple(a.shape) != expected_ab or tuple(b.shape) != expected_ab:
        raise GDNContractError(
            "in_proj_a and in_proj_b outputs must both have shape "
            f"{expected_ab}"
        )

    mixed_qkv, next_conv_state = _functional_causal_conv(
        mixed_qkv, current.conv_state, module, spec
    )
    mixed_qkv = mixed_qkv.transpose(1, 2)
    query, key, value = torch.split(
        mixed_qkv,
        (spec.key_dim, spec.key_dim, spec.value_dim),
        dim=-1,
    )
    query = query.reshape(
        batch_size, sequence_length, spec.num_k_heads, spec.head_k_dim
    )
    key = key.reshape(
        batch_size, sequence_length, spec.num_k_heads, spec.head_k_dim
    )
    value = value.reshape(
        batch_size, sequence_length, spec.num_v_heads, spec.head_v_dim
    )
    query = _l2norm(query)
    key = _l2norm(key)
    repeat = spec.num_v_heads // spec.num_k_heads
    if repeat > 1:
        query = query.repeat_interleave(repeat, dim=2)
        key = key.repeat_interleave(repeat, dim=2)

    beta = b.sigmoid()
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)
    core_output, next_recurrent_state = _functional_recurrent_gated_delta_rule(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=current.recurrent_state,
    )
    flat_output = core_output.reshape(-1, spec.head_v_dim)
    flat_gate = z.reshape(-1, spec.head_v_dim)
    normalized = module.norm(flat_output, flat_gate)
    if tuple(normalized.shape) != tuple(flat_output.shape):
        raise GDNContractError(
            f"module.norm output shape {tuple(normalized.shape)} != {tuple(flat_output.shape)}"
        )
    output = module.out_proj(
        normalized.reshape(batch_size, sequence_length, spec.value_dim)
    )
    expected_output = (batch_size, sequence_length, spec.hidden_size)
    if tuple(output.shape) != expected_output:
        raise GDNContractError(
            f"module.out_proj output shape {tuple(output.shape)} != {expected_output}"
        )

    next_state = QueryLocalGDNState(
        base=current.base,
        conv_delta=next_conv_state,
        recurrent_delta=next_recurrent_state,
        tokens_processed=current.tokens_processed + sequence_length,
    )
    # This audits the most important contract at the call boundary. It detects
    # both ordinary PyTorch in-place writes and sampled inference-tensor writes.
    current.base.assert_unchanged()
    old_keys = {
        _tensor_storage_key(current.conv_state),
        _tensor_storage_key(current.recurrent_state),
    }
    new_keys = {
        _tensor_storage_key(next_state.conv_state),
        _tensor_storage_key(next_state.recurrent_state),
    }
    if old_keys & new_keys:
        raise GDNContractError("functional transition reused input state storage")
    return output, next_state


def functional_qwen35_gdn_prefill(
    module: Any,
    hidden_states: torch.Tensor,
    *,
    gradient_semantics: GradientSemantics,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ImmutableGDNBase]:
    """Run document prefill and explicitly choose the resulting base semantics."""

    output, state = functional_qwen35_gdn_forward(
        module, hidden_states, attention_mask=attention_mask
    )
    return output, state.promote_to_base(
        gradient_semantics=gradient_semantics,
        source="functional-prefill",
    )


def immutable_base_from_transformers_cache(
    module: Any,
    cache: Any,
    *,
    gradient_semantics: GradientSemantics,
    layer_idx: int | None = None,
    state_idx: int = 0,
) -> ImmutableGDNBase:
    """Clone one initialized Transformers linear-attention cache layer.

    The audited Transformers interface is ``cache.layers[layer_idx]`` with
    dict-valued ``conv_states``, ``recurrent_states``, initialization flags,
    ``has_previous_state``, and ``conv_kernel_size``.  Any deviation fails
    closed instead of falling back to a mutable cache.
    """

    spec = audit_qwen35_gdn_module(module)
    if layer_idx is None:
        layer_idx = spec.layer_idx
    if not isinstance(layer_idx, int) or isinstance(layer_idx, bool) or layer_idx < 0:
        raise GDNContractError("layer_idx must be a non-negative integer")
    if layer_idx != spec.layer_idx:
        raise GDNContractError(
            f"requested cache layer {layer_idx} != module.layer_idx {spec.layer_idx}"
        )
    if not isinstance(state_idx, int) or isinstance(state_idx, bool) or state_idx < 0:
        raise GDNContractError("state_idx must be a non-negative integer")
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, (list, tuple)) or layer_idx >= len(layers):
        raise GDNContractError("cache.layers does not contain the requested GDN layer")
    layer = layers[layer_idx]

    required_mappings = (
        "conv_states",
        "recurrent_states",
        "is_conv_states_initialized",
        "is_recurrent_states_initialized",
        "has_previous_state",
        "conv_kernel_size",
    )
    mappings = {}
    for name in required_mappings:
        value = getattr(layer, name, None)
        if not isinstance(value, dict) or state_idx not in value:
            raise GDNContractError(
                f"cache layer {layer_idx}.{name} must be a dict containing state {state_idx}"
            )
        mappings[name] = value
    if not mappings["is_conv_states_initialized"][state_idx]:
        raise GDNContractError("Transformers conv state is not initialized")
    if not mappings["is_recurrent_states_initialized"][state_idx]:
        raise GDNContractError("Transformers recurrent state is not initialized")
    if not mappings["has_previous_state"][state_idx]:
        raise GDNContractError("Transformers cache has no completed previous GDN state")
    if mappings["conv_kernel_size"][state_idx] != spec.conv_kernel_size:
        raise GDNContractError(
            "Transformers cache conv_kernel_size does not match the Qwen module"
        )
    conv_state = mappings["conv_states"][state_idx]
    recurrent_state = mappings["recurrent_states"][state_idx]
    if not isinstance(conv_state, torch.Tensor) or not isinstance(
        recurrent_state, torch.Tensor
    ):
        raise GDNContractError("initialized Transformers GDN states must be tensors")

    expected_conv = (conv_state.shape[0], spec.conv_dim, spec.conv_kernel_size)
    expected_recurrent = (conv_state.shape[0], *spec.recurrent_shape_tail)
    if tuple(conv_state.shape) != expected_conv:
        raise GDNContractError(
            f"Transformers conv state shape {tuple(conv_state.shape)} != {expected_conv}"
        )
    if tuple(recurrent_state.shape) != expected_recurrent:
        raise GDNContractError(
            "Transformers recurrent state shape "
            f"{tuple(recurrent_state.shape)} != {expected_recurrent}"
        )
    if conv_state.device != recurrent_state.device:
        raise GDNContractError("Transformers GDN states must share one device")
    return ImmutableGDNBase(
        conv_state=conv_state,
        recurrent_state=recurrent_state,
        gradient_semantics=gradient_semantics,
        source=f"transformers-cache-layer:{layer_idx}:state:{state_idx}",
    )


def _require_decoder_callable(decoder_layer: Any, name: str, index: int) -> None:
    if not callable(getattr(decoder_layer, name, None)):
        raise GDNContractError(
            f"decoder layer {index} is missing callable {name!r}"
        )


def audit_qwen35_gdn_dispatch_plan(
    layers: Sequence[Any],
    config: Any,
) -> Qwen35GDNDispatchPlan:
    """Audit every Qwen3.5 DecoderLayer before functional dispatch.

    ``config.layer_types`` is the source of truth.  The function checks the
    actual DecoderLayer ``block_type`` and every nested GDN ``layer_idx`` so a
    renamed/missing layer cannot silently run the ordinary mutable forward.
    """

    try:
        resolved_layers = tuple(layers)
    except TypeError as error:
        raise GDNContractError("layers must be an iterable of decoder layers") from error
    layer_types_value = getattr(config, "layer_types", None)
    if not isinstance(layer_types_value, (list, tuple)):
        raise GDNContractError("config.layer_types must be a list or tuple")
    layer_types = tuple(layer_types_value)
    if len(layer_types) != len(resolved_layers):
        raise GDNContractError(
            f"config has {len(layer_types)} layer types but model has {len(resolved_layers)} layers"
        )
    configured_count = getattr(config, "num_hidden_layers", len(layer_types))
    if configured_count != len(layer_types):
        raise GDNContractError(
            "config.num_hidden_layers does not match config.layer_types"
        )
    configured_width = getattr(config, "hidden_size", None)
    if not isinstance(configured_width, int) or configured_width < 1:
        raise GDNContractError("config.hidden_size must be a positive integer")

    linear_indices = []
    full_indices = []
    for index, (decoder_layer, expected_type) in enumerate(
        zip(resolved_layers, layer_types)
    ):
        if expected_type not in ("linear_attention", "full_attention"):
            raise GDNContractError(
                f"config.layer_types[{index}] has unsupported value {expected_type!r}"
            )
        actual_type = getattr(decoder_layer, "block_type", None)
        if actual_type != expected_type:
            raise GDNContractError(
                f"decoder layer {index}.block_type {actual_type!r} != {expected_type!r}"
            )
        if getattr(decoder_layer, "hidden_size", None) != configured_width:
            raise GDNContractError(
                f"decoder layer {index}.hidden_size does not match config.hidden_size"
            )
        _require_decoder_callable(decoder_layer, "input_layernorm", index)
        _require_decoder_callable(decoder_layer, "post_attention_layernorm", index)
        _require_decoder_callable(decoder_layer, "mlp", index)
        if expected_type == "linear_attention":
            linear_attn = getattr(decoder_layer, "linear_attn", None)
            if linear_attn is None:
                raise GDNContractError(
                    f"linear decoder layer {index} has no linear_attn module"
                )
            spec = audit_qwen35_gdn_module(linear_attn)
            if spec.layer_idx != index:
                raise GDNContractError(
                    f"decoder layer {index} contains GDN for layer {spec.layer_idx}"
                )
            if spec.hidden_size != configured_width:
                raise GDNContractError(
                    f"decoder layer {index} GDN width does not match config.hidden_size"
                )
            if hasattr(decoder_layer, "self_attn"):
                raise GDNContractError(
                    f"linear decoder layer {index} unexpectedly exposes self_attn"
                )
            linear_indices.append(index)
        else:
            _require_decoder_callable(decoder_layer, "self_attn", index)
            attention_idx = getattr(decoder_layer.self_attn, "layer_idx", None)
            if attention_idx != index:
                raise GDNContractError(
                    f"decoder layer {index} contains attention for layer {attention_idx}"
                )
            if hasattr(decoder_layer, "linear_attn"):
                raise GDNContractError(
                    f"full-attention decoder layer {index} unexpectedly exposes linear_attn"
                )
            full_indices.append(index)
    return Qwen35GDNDispatchPlan(
        total_layers=len(resolved_layers),
        layer_types=layer_types,
        linear_layer_indices=tuple(linear_indices),
        full_attention_layer_indices=tuple(full_indices),
    )


def _audit_linear_decoder_layer(decoder_layer: Any) -> tuple[Any, Qwen35GDNSpec]:
    if getattr(decoder_layer, "block_type", None) != "linear_attention":
        raise GDNContractError(
            "functional GDN decoder helper requires block_type='linear_attention'"
        )
    linear_attn = getattr(decoder_layer, "linear_attn", None)
    if linear_attn is None:
        raise GDNContractError("linear decoder layer has no linear_attn module")
    spec = audit_qwen35_gdn_module(linear_attn)
    if getattr(decoder_layer, "hidden_size", None) != spec.hidden_size:
        raise GDNContractError("decoder layer and nested GDN hidden widths differ")
    for name in ("input_layernorm", "post_attention_layernorm", "mlp"):
        _require_decoder_callable(decoder_layer, name, spec.layer_idx)
    if hasattr(decoder_layer, "self_attn"):
        raise GDNContractError("linear decoder layer unexpectedly exposes self_attn")
    return linear_attn, spec


def functional_qwen35_linear_decoder_layer_forward(
    decoder_layer: Any,
    hidden_states: torch.Tensor,
    state: QueryLocalGDNState | ImmutableGDNBase,
    *,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, QueryLocalGDNState, GDNDispatchTelemetry]:
    """Functional replacement for a real dense or MoE linear DecoderLayer.

    This mirrors the Transformers layer order, including MoE's tuple return,
    but the token mixer is always :func:`functional_qwen35_gdn_forward`.  The
    helper has no cache argument by design, so it cannot reach ``update_*``.
    """

    if not isinstance(state, (QueryLocalGDNState, ImmutableGDNBase)):
        raise GDNContractError(
            "linear decoder dispatch requires an explicit immutable/query-local state"
        )
    linear_attn, spec = _audit_linear_decoder_layer(decoder_layer)
    if not isinstance(hidden_states, torch.Tensor) or hidden_states.ndim != 3:
        raise GDNContractError("decoder hidden_states must have rank 3")
    if hidden_states.shape[-1] != spec.hidden_size:
        raise GDNContractError(
            f"decoder hidden width {hidden_states.shape[-1]} != {spec.hidden_size}"
        )
    before_tokens = state.tokens_processed if isinstance(
        state, QueryLocalGDNState
    ) else 0
    base = state.base if isinstance(state, QueryLocalGDNState) else state

    residual = hidden_states
    normalized = decoder_layer.input_layernorm(hidden_states)
    if not isinstance(normalized, torch.Tensor) or tuple(normalized.shape) != tuple(
        hidden_states.shape
    ):
        raise GDNContractError(
            "decoder input_layernorm must return a tensor matching hidden_states"
        )
    mixed, next_state = functional_qwen35_gdn_forward(
        linear_attn,
        normalized,
        state,
        attention_mask=attention_mask,
    )
    hidden_states = residual + mixed

    residual = hidden_states
    normalized = decoder_layer.post_attention_layernorm(hidden_states)
    if not isinstance(normalized, torch.Tensor) or tuple(normalized.shape) != tuple(
        hidden_states.shape
    ):
        raise GDNContractError(
            "decoder post_attention_layernorm must return a tensor matching hidden_states"
        )
    mlp_output = decoder_layer.mlp(normalized)
    if isinstance(mlp_output, tuple):
        if not mlp_output or not isinstance(mlp_output[0], torch.Tensor):
            raise GDNContractError("MoE MLP tuple must start with a tensor")
        mlp_output = mlp_output[0]
    if not isinstance(mlp_output, torch.Tensor) or tuple(mlp_output.shape) != tuple(
        hidden_states.shape
    ):
        raise GDNContractError(
            "decoder MLP must return hidden_states shape (directly or tuple[0])"
        )
    hidden_states = residual + mlp_output
    telemetry = GDNDispatchTelemetry(
        layer_idx=spec.layer_idx,
        route="functional-qwen35-gdn",
        kernel="torch-depthwise-conv1d+torch-recurrent-gated-delta-scan",
        module_class=type(linear_attn).__name__,
        sequence_length=int(hidden_states.shape[1]),
        state_tokens_before=before_tokens,
        state_tokens_after=next_state.tokens_processed,
        gradient_semantics=base.gradient_semantics,
        base_source=base.source,
        shared_base_nbytes=next_state.shared_base_nbytes,
        query_private_nbytes=next_state.query_private_nbytes,
    )
    return hidden_states, next_state, telemetry


FullAttentionDispatch = Callable[..., tuple[torch.Tensor, object, object]]


def dispatch_qwen35_decoder_layer(
    decoder_layer: Any,
    hidden_states: torch.Tensor,
    layer_state: object,
    *,
    layer_idx: int,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    full_attention_dispatch: FullAttentionDispatch | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, object, object]:
    """Fail-closed DecoderLayer router for a functional ``_run_layers``.

    Linear layers are intercepted locally.  Full-attention layers must provide
    a separate explicit dispatcher; this function never calls the ordinary
    ``decoder_layer.forward`` as a fallback.
    """

    if "past_key_values" in kwargs or "cache_params" in kwargs:
        raise GDNContractError(
            "mutable cache arguments are forbidden in functional decoder dispatch"
        )
    if not isinstance(layer_idx, int) or isinstance(layer_idx, bool) or layer_idx < 0:
        raise GDNContractError("layer_idx must be a non-negative integer")
    block_type = getattr(decoder_layer, "block_type", None)
    if block_type == "linear_attention":
        linear_attn = getattr(decoder_layer, "linear_attn", None)
        actual_idx = getattr(linear_attn, "layer_idx", None)
        if actual_idx != layer_idx:
            raise GDNContractError(
                f"linear dispatch index {layer_idx} != module.layer_idx {actual_idx}"
            )
        if not isinstance(layer_state, (ImmutableGDNBase, QueryLocalGDNState)):
            raise GDNContractError(
                f"linear layer {layer_idx} is missing explicit functional GDN state"
            )
        return functional_qwen35_linear_decoder_layer_forward(
            decoder_layer,
            hidden_states,
            layer_state,
            attention_mask=attention_mask,
        )
    if block_type == "full_attention":
        attention_idx = getattr(getattr(decoder_layer, "self_attn", None), "layer_idx", None)
        if attention_idx != layer_idx:
            raise GDNContractError(
                f"full-attention dispatch index {layer_idx} != module.layer_idx {attention_idx}"
            )
        if layer_state is None:
            raise GDNContractError(
                f"full-attention layer {layer_idx} is missing explicit functional state"
            )
        if full_attention_dispatch is None:
            raise GDNContractError(
                f"full-attention layer {layer_idx} requires an out-of-place dispatcher"
            )
        result = full_attention_dispatch(
            decoder_layer,
            hidden_states,
            layer_state=layer_state,
            layer_idx=layer_idx,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if not isinstance(result, tuple) or len(result) != 3:
            raise GDNContractError(
                "full_attention_dispatch must return (hidden, next_state, telemetry)"
            )
        output, next_state, telemetry = result
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != tuple(
            hidden_states.shape
        ):
            raise GDNContractError(
                "full_attention_dispatch output must match hidden_states shape"
            )
        if next_state is None or telemetry is None:
            raise GDNContractError(
                "full_attention_dispatch must return explicit state and telemetry"
            )
        return output, next_state, telemetry
    raise GDNContractError(
        f"decoder layer {layer_idx} has unsupported block_type {block_type!r}"
    )
