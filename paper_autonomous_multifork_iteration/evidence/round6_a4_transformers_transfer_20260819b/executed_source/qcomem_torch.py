from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import torch


SUPPORTED_BITS = (2, 4, 8, 16)


def tensor_nbytes(tensor: torch.Tensor | None) -> int:
    if tensor is None:
        return 0
    return tensor.numel() * tensor.element_size()


def _pack_unsigned(values: torch.Tensor, bits: int) -> torch.Tensor:
    if bits == 8:
        return values.to(torch.uint8)
    values_per_byte = 8 // bits
    if values.shape[-1] % values_per_byte:
        raise ValueError("the packed dimension must fill complete bytes")
    grouped = values.to(torch.int16).reshape(
        *values.shape[:-1], values.shape[-1] // values_per_byte, values_per_byte
    )
    shifts = (
        torch.arange(values_per_byte, device=values.device, dtype=torch.int16)
        * bits
    )
    return torch.sum(grouped << shifts, dim=-1).to(torch.uint8)


def _unpack_unsigned(
    packed: torch.Tensor, bits: int, output_length: int
) -> torch.Tensor:
    if bits == 8:
        return packed[..., :output_length]
    values_per_byte = 8 // bits
    shifts = (
        torch.arange(values_per_byte, device=packed.device, dtype=torch.int16)
        * bits
    )
    mask = (1 << bits) - 1
    unpacked = (
        (packed.to(torch.int16).unsqueeze(-1) >> shifts) & mask
    ).reshape(*packed.shape[:-1], -1)
    return unpacked[..., :output_length].to(torch.uint8)


@dataclass
class PackedResidual:
    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    data: torch.Tensor
    scales: torch.Tensor | None = None
    biases: torch.Tensor | None = None

    @property
    def nbytes(self) -> int:
        return sum(
            tensor_nbytes(tensor)
            for tensor in (self.data, self.scales, self.biases)
        )

    @property
    def dense_nbytes(self) -> int:
        return math.prod(self.original_shape) * 2

    @property
    def compression_ratio(self) -> float:
        return self.dense_nbytes / self.nbytes

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        if self.bits == 16:
            # Intentionally retain zero-copy storage when dtype already
            # matches: a document boundary residual is immutable by the
            # LowerReplayState contract.  Generation only reads it to seed the
            # suffix cache and then drops the request-local reference.  This
            # rule does not apply to PackedTensor cache leaves, which kernels
            # may update in-place and therefore always clone on Q16 fork.
            return self.data.to(dtype)
        assert self.scales is not None and self.biases is not None
        hidden_size = self.original_shape[-1]
        values = _unpack_unsigned(self.data, self.bits, hidden_size)
        grouped = values.reshape(
            *self.original_shape[:-1], hidden_size // self.group_size, self.group_size
        ).float()
        restored = grouped * self.scales.float().unsqueeze(-1)
        restored = restored + self.biases.float().unsqueeze(-1)
        return restored.reshape(self.original_shape).to(dtype)


def quantize_residual(
    residual: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
) -> PackedResidual:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    if residual.ndim != 3:
        raise ValueError("residual must have shape [batch, tokens, hidden]")
    hidden_size = residual.shape[-1]
    if hidden_size % group_size:
        raise ValueError("hidden size must be divisible by group size")
    if bits == 16:
        return PackedResidual(
            bits=bits,
            group_size=group_size,
            original_shape=tuple(residual.shape),
            data=residual.to(torch.bfloat16),
        )

    grouped = residual.float().reshape(
        *residual.shape[:-1], hidden_size // group_size, group_size
    )
    biases = grouped.amin(dim=-1)
    maxima = grouped.amax(dim=-1)
    levels = (1 << bits) - 1
    scales = (maxima - biases) / levels
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.round((grouped - biases.unsqueeze(-1)) / scales.unsqueeze(-1))
    quantized = quantized.clamp_(0, levels).to(torch.uint8)
    flat = quantized.reshape(*residual.shape[:-1], hidden_size)
    packed = _pack_unsigned(flat, bits)
    return PackedResidual(
        bits=bits,
        group_size=group_size,
        original_shape=tuple(residual.shape),
        data=packed,
        scales=scales.to(torch.bfloat16),
        biases=biases.to(torch.bfloat16),
    )


@dataclass
class PackedTensor:
    """Affine group quantization for an arbitrary floating-point cache tensor."""

    bits: int
    group_size: int
    original_shape: tuple[int, ...]
    original_dtype: torch.dtype
    data: torch.Tensor
    scales: torch.Tensor | None = None
    biases: torch.Tensor | None = None
    squared_error_sum: float = 0.0
    reference_squared_sum: float = 0.0
    max_abs_error: float = 0.0

    @property
    def nbytes(self) -> int:
        return sum(
            tensor_nbytes(tensor)
            for tensor in (self.data, self.scales, self.biases)
        )

    def dequantize(self) -> torch.Tensor:
        elements = math.prod(self.original_shape)
        if self.bits == 16:
            # ``reshape().to(same_dtype)`` may return a view of the persistent
            # packed store.  Cache leaves are mutable (notably Qwen3.5
            # recurrent/conv state), so every fork must own independent
            # storage even though Q16 does not require numeric conversion.
            return (
                self.data.reshape(self.original_shape)
                .to(self.original_dtype)
                .clone()
            )
        assert self.scales is not None and self.biases is not None
        padded_elements = self.scales.numel() * self.group_size
        values = _unpack_unsigned(self.data, self.bits, padded_elements).float()
        grouped = values.reshape(-1, self.group_size)
        restored = grouped * self.scales.float().unsqueeze(-1)
        restored = restored + self.biases.float().unsqueeze(-1)
        return restored.reshape(-1)[:elements].reshape(self.original_shape).to(
            self.original_dtype
        )


def quantize_tensor(
    tensor: torch.Tensor,
    *,
    bits: int,
    group_size: int = 64,
) -> PackedTensor:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}")
    if group_size < 1 or group_size % (8 // min(bits, 8)):
        raise ValueError("group size must fill complete packed bytes")
    if not tensor.is_floating_point():
        raise ValueError("only floating-point tensors can be quantized")
    original_shape = tuple(tensor.shape)
    flat = tensor.detach().reshape(-1)
    reference_squared_sum = float(torch.square(flat.float()).sum().item())
    if bits == 16:
        data = flat.clone()
        return PackedTensor(
            bits=bits,
            group_size=group_size,
            original_shape=original_shape,
            original_dtype=tensor.dtype,
            data=data,
            reference_squared_sum=reference_squared_sum,
        )
    if flat.numel() == 0:
        empty = torch.empty(0, device=tensor.device, dtype=torch.uint8)
        metadata = torch.empty(0, device=tensor.device, dtype=torch.bfloat16)
        return PackedTensor(
            bits=bits,
            group_size=group_size,
            original_shape=original_shape,
            original_dtype=tensor.dtype,
            data=empty,
            scales=metadata,
            biases=metadata,
        )

    groups = (flat.numel() + group_size - 1) // group_size
    padded_elements = groups * group_size
    if padded_elements != flat.numel():
        padding = flat[-1:].expand(padded_elements - flat.numel())
        flat = torch.cat([flat, padding])
    grouped = flat.float().reshape(groups, group_size)
    biases = grouped.amin(dim=-1)
    maxima = grouped.amax(dim=-1)
    levels = (1 << bits) - 1
    scales = (maxima - biases) / levels
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quantized = torch.round((grouped - biases.unsqueeze(-1)) / scales.unsqueeze(-1))
    quantized = quantized.clamp_(0, levels).to(torch.uint8)
    restored = quantized.float() * scales.unsqueeze(-1) + biases.unsqueeze(-1)
    error = grouped.reshape(-1)[: tensor.numel()] - restored.reshape(-1)[: tensor.numel()]
    packed = _pack_unsigned(quantized.reshape(-1), bits)
    return PackedTensor(
        bits=bits,
        group_size=group_size,
        original_shape=original_shape,
        original_dtype=tensor.dtype,
        data=packed,
        scales=scales.to(torch.bfloat16),
        biases=biases.to(torch.bfloat16),
        squared_error_sum=float(torch.square(error).sum().item()),
        reference_squared_sum=reference_squared_sum,
        max_abs_error=float(error.abs().max().item()),
    )


def residual_error_sums(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | int]:
    error = reference.float() - candidate.float()
    return {
        "squared_error_sum": float(torch.square(error).sum().item()),
        "reference_squared_sum": float(torch.square(reference.float()).sum().item()),
        "max_abs_error": float(error.abs().max().item()),
        "elements": error.numel(),
    }


def cache_nbytes(cache: Any) -> int:
    """Count unique tensor storage held by a Transformers cache."""
    seen_objects: set[int] = set()
    seen_tensors: set[tuple[str, int, int]] = set()

    def visit(value: Any) -> int:
        object_id = id(value)
        if object_id in seen_objects:
            return 0
        seen_objects.add(object_id)
        if isinstance(value, torch.Tensor):
            key = (
                str(value.device),
                value.untyped_storage().data_ptr(),
                value.untyped_storage().nbytes(),
            )
            if key in seen_tensors:
                return 0
            seen_tensors.add(key)
            return value.untyped_storage().nbytes()
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(visit(item) for item in value)
        if hasattr(value, "__dict__"):
            return sum(visit(item) for item in vars(value).values())
        return 0

    return visit(cache)


def active_cache_layer_indices(cache: Any) -> tuple[int, ...]:
    """Return cache layers that actually hold tensor storage.

    Recent Transformers ``DynamicCache(config=...)`` builds allocate one layer
    object for every model layer up front.  A depth-7 split therefore exposes
    40 layer objects on Qwen3.5 even though only layers 0--6 contain persistent
    state.  Compact mixed-bit policies intentionally describe only those
    stored layers.
    """

    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError("expected a Transformers cache with a layers attribute")
    return tuple(
        index for index, layer in enumerate(layers) if cache_nbytes(layer) > 0
    )


def clone_cache(cache: Any) -> Any:
    """Deep-copy a cache, explicitly cloning non-leaf/inference tensors."""
    memo: dict[int, Any] = {}
    visited: set[int] = set()

    def seed_tensor_copies(value: Any) -> None:
        object_id = id(value)
        if object_id in visited:
            return
        visited.add(object_id)
        if isinstance(value, torch.Tensor):
            memo[object_id] = value.detach().clone()
        elif isinstance(value, dict):
            for item in value.values():
                seed_tensor_copies(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                seed_tensor_copies(item)
        elif hasattr(value, "__dict__"):
            for item in vars(value).values():
                seed_tensor_copies(item)

    seed_tensor_copies(cache)
    return copy.deepcopy(cache, memo)


def _merge_error_sums(
    target: dict[str, float | int], packed: PackedTensor
) -> None:
    target["squared_error_sum"] += packed.squared_error_sum
    target["reference_squared_sum"] += packed.reference_squared_sum
    target["max_abs_error"] = max(
        float(target["max_abs_error"]), packed.max_abs_error
    )
    target["elements"] += math.prod(packed.original_shape)


@dataclass
class PackedCache:
    """Transformers cache whose tensor leaves are independently bit-packed."""

    cache: Any
    attention_bits: int
    linear_bits: int
    layer_bits: tuple[int, ...]
    group_size: int
    error_sums: dict[str, dict[str, float | int]]
    layer_error_sums: tuple[dict[str, float | int], ...]

    @property
    def nbytes(self) -> int:
        return cache_nbytes(self.cache)

    def dequantize(self) -> Any:
        memo: dict[int, Any] = {}
        visited: set[int] = set()

        def seed(value: Any) -> None:
            object_id = id(value)
            if object_id in visited:
                return
            visited.add(object_id)
            if isinstance(value, PackedTensor):
                memo[object_id] = value.dequantize()
            elif isinstance(value, torch.Tensor):
                memo[object_id] = value.detach().clone()
            elif isinstance(value, dict):
                for item in value.values():
                    seed(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    seed(item)
            elif hasattr(value, "__dict__"):
                for item in vars(value).values():
                    seed(item)

        seed(self.cache)
        return copy.deepcopy(self.cache, memo)


@torch.inference_mode()
def quantize_transformers_cache(
    cache: Any,
    *,
    attention_bits: int,
    linear_bits: int,
    cache_layer_bits: Sequence[int] | None = None,
    group_size: int = 64,
) -> PackedCache:
    """Pack full-attention and linear-attention cache layers separately."""
    for bits in (attention_bits, linear_bits):
        if bits not in SUPPORTED_BITS:
            raise ValueError(f"cache bits must be one of {SUPPORTED_BITS}")
    memo: dict[int, Any] = {}
    visited: set[int] = set()
    errors = {
        category: {
            "squared_error_sum": 0.0,
            "reference_squared_sum": 0.0,
            "max_abs_error": 0.0,
            "elements": 0,
        }
        for category in ("attention", "linear")
    }

    def seed(
        value: Any,
        *,
        bits: int,
        target: dict[str, float | int],
    ) -> None:
        object_id = id(value)
        if object_id in visited:
            return
        visited.add(object_id)
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                packed = quantize_tensor(value, bits=bits, group_size=group_size)
                memo[object_id] = packed
                _merge_error_sums(target, packed)
            else:
                memo[object_id] = value.detach().clone()
        elif isinstance(value, dict):
            for item in value.values():
                seed(item, bits=bits, target=target)
        elif isinstance(value, (list, tuple)):
            for item in value:
                seed(item, bits=bits, target=target)
        elif hasattr(value, "__dict__"):
            for item in vars(value).values():
                seed(item, bits=bits, target=target)

    layers = getattr(cache, "layers", None)
    if layers is None:
        raise ValueError("expected a Transformers cache with a layers attribute")
    if cache_layer_bits is None:
        resolved_layer_bits = tuple(
            linear_bits
            if hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")
            else attention_bits
            for layer in layers
        )
    else:
        requested_layer_bits = tuple(cache_layer_bits)
        if len(requested_layer_bits) == len(layers):
            resolved_layer_bits = requested_layer_bits
        else:
            active_indices = active_cache_layer_indices(cache)
            if len(requested_layer_bits) != len(active_indices):
                raise ValueError(
                    "cache_layer_bits must have one entry per active stored "
                    f"cache layer ({len(active_indices)}) or per allocated "
                    f"cache layer ({len(layers)})"
                )
            expanded = [16] * len(layers)
            for index, bits in zip(active_indices, requested_layer_bits):
                expanded[index] = bits
            resolved_layer_bits = tuple(expanded)
    for bits in resolved_layer_bits:
        if bits not in SUPPORTED_BITS:
            raise ValueError(f"cache bits must be one of {SUPPORTED_BITS}")

    layer_errors = []
    for layer, bits in zip(layers, resolved_layer_bits):
        is_linear = hasattr(layer, "conv_states") or hasattr(
            layer, "recurrent_states"
        )
        category = "linear" if is_linear else "attention"
        layer_error: dict[str, float | int] = {
            "squared_error_sum": 0.0,
            "reference_squared_sum": 0.0,
            "max_abs_error": 0.0,
            "elements": 0,
        }
        seed(layer, bits=bits, target=layer_error)
        layer_errors.append(layer_error)
        errors[category]["squared_error_sum"] += layer_error["squared_error_sum"]
        errors[category]["reference_squared_sum"] += layer_error[
            "reference_squared_sum"
        ]
        errors[category]["max_abs_error"] = max(
            float(errors[category]["max_abs_error"]),
            float(layer_error["max_abs_error"]),
        )
        errors[category]["elements"] += layer_error["elements"]
    packed_cache = copy.deepcopy(cache, memo)
    return PackedCache(
        cache=packed_cache,
        attention_bits=attention_bits,
        linear_bits=linear_bits,
        layer_bits=resolved_layer_bits,
        group_size=group_size,
        error_sums=errors,
        layer_error_sums=tuple(layer_errors),
    )


@dataclass
class LowerReplayState:
    """Reusable document boundary plus lower-layer attention state.

    ``cache`` contains ordinary-attention K/V tensors and, for hybrid models
    such as Qwen3.5, linear-attention convolution and recurrent states.
    ``fork`` shares the immutable boundary residual but clones the mutable
    cache so that independent queries can reuse the same document write.
    """

    depth: int
    document_length: int
    current_length: int
    document_residual: torch.Tensor
    cache: Any

    @property
    def stored_nbytes(self) -> int:
        return tensor_nbytes(self.document_residual) + cache_nbytes(self.cache)

    def fork(self) -> "LowerReplayState":
        return LowerReplayState(
            depth=self.depth,
            document_length=self.document_length,
            current_length=self.current_length,
            document_residual=self.document_residual,
            cache=clone_cache(self.cache),
        )

    def quantize(
        self,
        *,
        bits: int,
        attention_bits: int | None = None,
        linear_bits: int | None = None,
        cache_layer_bits: Sequence[int] | None = None,
        group_size: int = 64,
    ) -> "PackedLowerReplayState":
        packed_cache = (
            quantize_transformers_cache(
                self.cache,
                attention_bits=attention_bits or 16,
                linear_bits=linear_bits or 16,
                cache_layer_bits=cache_layer_bits,
                group_size=group_size,
            )
            if (
                attention_bits is not None
                or linear_bits is not None
                or cache_layer_bits is not None
            )
            else self.cache
        )
        return PackedLowerReplayState(
            depth=self.depth,
            document_length=self.document_length,
            current_length=self.current_length,
            document_residual=quantize_residual(
                self.document_residual,
                bits=bits,
                group_size=group_size,
            ),
            cache=packed_cache,
        )


@dataclass
class PackedLowerReplayState:
    """Persistent replay state with a genuinely packed split residual."""

    depth: int
    document_length: int
    current_length: int
    document_residual: PackedResidual
    cache: Any

    @property
    def stored_nbytes(self) -> int:
        cache_bytes = (
            self.cache.nbytes
            if isinstance(self.cache, PackedCache)
            else cache_nbytes(self.cache)
        )
        return self.document_residual.nbytes + cache_bytes

    @property
    def cache_error_sums(self) -> dict[str, dict[str, float | int]] | None:
        return (
            self.cache.error_sums
            if isinstance(self.cache, PackedCache)
            else None
        )

    def fork(self) -> LowerReplayState:
        return LowerReplayState(
            depth=self.depth,
            document_length=self.document_length,
            current_length=self.current_length,
            document_residual=self.document_residual.dequantize(),
            cache=(
                self.cache.dequantize()
                if isinstance(self.cache, PackedCache)
                else clone_cache(self.cache)
            ),
        )


@dataclass
class FullPrefixState:
    """Reusable exact full-model prefix cache baseline."""

    document_length: int
    current_length: int
    cache: Any

    @property
    def stored_nbytes(self) -> int:
        return cache_nbytes(self.cache)

    def fork(self) -> "FullPrefixState":
        return FullPrefixState(
            document_length=self.document_length,
            current_length=self.current_length,
            cache=clone_cache(self.cache),
        )


class TorchSplitCausalLM:
    """Manual lower/suffix adapter for Qwen2 and Qwen3.5-MoE text backbones."""

    def __init__(self, model: torch.nn.Module) -> None:
        if hasattr(model.model, "language_model"):
            self.language_model = model.model.language_model
        else:
            self.language_model = model.model
        self.model = model
        self.layers = self.language_model.layers
        self.config = self.language_model.config
        self.num_layers = len(self.layers)
        self.lm_head = model.lm_head

    def _validate_depth(self, depth: int) -> None:
        if depth < 0 or depth > self.num_layers:
            raise ValueError(f"depth must be in [0, {self.num_layers}]")

    def make_cache(self) -> Any:
        try:
            from transformers.cache_utils import DynamicCache
        except ImportError as error:
            raise RuntimeError(
                "cached replay requires a Transformers build with DynamicCache"
            ) from error
        return DynamicCache(config=self.config)

    @staticmethod
    def _batch_tokens(tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim == 1:
            return tokens.unsqueeze(0)
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [tokens] or [batch, tokens]")
        return tokens

    def _layer_context(
        self,
        hidden: torch.Tensor,
        *,
        past_key_values: Any = None,
        position_offset: int = 0,
        layer_start: int = 0,
    ) -> SimpleNamespace:
        length = hidden.shape[1]
        device = hidden.device
        batch = hidden.shape[0]
        model_type = getattr(self.config, "model_type", "")
        layer_types = getattr(
            self.config,
            "layer_types",
            ["full_attention"] * self.num_layers,
        )
        try:
            full_attention_layer = next(
                index
                for index in range(layer_start, self.num_layers)
                if layer_types[index] == "full_attention"
            )
        except StopIteration:
            full_attention_layer = 0

        if model_type == "qwen3_5_moe_text":
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                create_causal_mask,
                create_recurrent_attention_mask,
            )

            position_ids = torch.arange(
                position_offset, position_offset + length, device=device
            )
            position_ids = position_ids.view(1, 1, -1).expand(4, batch, -1)
            text_position_ids = position_ids[0]
            rotary_position_ids = position_ids[1:]
            kwargs = {
                "config": self.config,
                "inputs_embeds": hidden,
                "attention_mask": None,
                "past_key_values": past_key_values,
                "position_ids": text_position_ids,
            }
            try:
                full_attention_mask = create_causal_mask(
                    **kwargs, layer_idx=full_attention_layer
                )
            except TypeError:
                # Compatibility with Transformers versions before per-layer
                # mask sizing.  The H20 environment uses the newer interface.
                full_attention_mask = create_causal_mask(**kwargs)
            masks = {
                "full_attention": full_attention_mask,
                "linear_attention": create_recurrent_attention_mask(**kwargs),
            }
        else:
            from transformers.masking_utils import create_causal_mask

            text_position_ids = torch.arange(
                position_offset, position_offset + length, device=device
            ).unsqueeze(0)
            rotary_position_ids = text_position_ids
            masks = {
                "full_attention": create_causal_mask(
                    config=self.config,
                    inputs_embeds=hidden,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    position_ids=text_position_ids,
                    layer_idx=full_attention_layer,
                )
            }
        position_embeddings = self.language_model.rotary_emb(
            hidden, rotary_position_ids
        )
        return SimpleNamespace(
            masks=masks,
            position_ids=text_position_ids,
            position_embeddings=position_embeddings,
        )

    def _run_layers(
        self,
        hidden: torch.Tensor,
        start: int,
        end: int,
        *,
        past_key_values: Any = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        context = self._layer_context(
            hidden,
            past_key_values=past_key_values,
            position_offset=position_offset,
            layer_start=start,
        )
        layer_types = getattr(
            self.config,
            "layer_types",
            ["full_attention"] * self.num_layers,
        )
        for index in range(start, end):
            output = self.layers[index](
                hidden,
                attention_mask=context.masks[layer_types[index]],
                position_ids=context.position_ids,
                position_embeddings=context.position_embeddings,
                past_key_values=past_key_values,
                use_cache=past_key_values is not None,
            )
            hidden = output[0] if isinstance(output, tuple) else output
        return hidden

    def run_to_depth(self, tokens: torch.Tensor, depth: int) -> torch.Tensor:
        self._validate_depth(depth)
        tokens = self._batch_tokens(tokens)
        hidden = self.language_model.embed_tokens(tokens)
        return self._run_layers(hidden, 0, depth)

    @torch.inference_mode()
    def write_lower_replay(
        self, tokens: torch.Tensor, depth: int
    ) -> LowerReplayState:
        """Write a reusable document residual and its lower-layer cache."""
        self._validate_depth(depth)
        tokens = self._batch_tokens(tokens)
        if tokens.shape[1] == 0:
            raise ValueError("the replay document must contain at least one token")
        cache = self.make_cache()
        hidden = self.language_model.embed_tokens(tokens)
        residual = self._run_layers(
            hidden,
            0,
            depth,
            past_key_values=cache,
            position_offset=0,
        )
        length = int(tokens.shape[1])
        return LowerReplayState(
            depth=depth,
            document_length=length,
            current_length=length,
            document_residual=residual,
            cache=cache,
        )

    @torch.inference_mode()
    def write_lower_replay_documents(
        self,
        documents: Iterable[torch.Tensor],
        depth: int,
    ) -> LowerReplayState:
        """Write fixed-order documents into one exact continuous lower cache."""

        token_batches = [self._batch_tokens(tokens) for tokens in documents]
        if not token_batches:
            raise ValueError("at least one document is required")
        if any(tokens.shape[1] == 0 for tokens in token_batches):
            raise ValueError("replay documents must contain at least one token")
        if len({tokens.shape[0] for tokens in token_batches}) != 1:
            raise ValueError("all replay documents must use the same batch size")

        state = self.write_lower_replay(token_batches[0], depth)
        residuals = [state.document_residual]
        for tokens in token_batches[1:]:
            residuals.append(self.continue_lower_replay(state, tokens))
        state.document_residual = torch.cat(residuals, dim=1)
        state.document_length = sum(tokens.shape[1] for tokens in token_batches)
        return state

    @torch.inference_mode()
    def continue_lower_replay(
        self, state: LowerReplayState, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Run new tokens through lower layers while reading/updating replay."""
        self._validate_depth(state.depth)
        tokens = self._batch_tokens(tokens)
        if tokens.shape[1] == 0:
            raise ValueError("replay continuation must contain at least one token")
        hidden = self.language_model.embed_tokens(tokens)
        residual = self._run_layers(
            hidden,
            0,
            state.depth,
            past_key_values=state.cache,
            position_offset=state.current_length,
        )
        state.current_length += int(tokens.shape[1])
        return residual

    def chunk_local_write_parts(
        self,
        tokens: torch.Tensor,
        depth: int,
        *,
        chunk_size: int,
        overlap: int,
    ) -> list[torch.Tensor]:
        self._validate_depth(depth)
        tokens = self._batch_tokens(tokens)
        if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("require chunk_size > 0 and 0 <= overlap < chunk_size")
        parts = []
        for start in range(0, tokens.shape[1], chunk_size):
            end = min(start + chunk_size, tokens.shape[1])
            context_start = max(0, start - overlap)
            local = self.run_to_depth(tokens[:, context_start:end], depth)
            parts.append(local[:, start - context_start :, :])
        return parts

    def run_suffix_last_logits(
        self, residuals: Iterable[torch.Tensor], depth: int
    ) -> torch.Tensor:
        self._validate_depth(depth)
        residuals = list(residuals)
        if not residuals:
            raise ValueError("at least one residual is required")
        hidden = torch.cat(residuals, dim=1)
        hidden = self._run_layers(hidden, depth, self.num_layers)
        hidden = self.language_model.norm(hidden[:, -1:, :])
        return self.lm_head(hidden)[:, -1, :]

    def run_suffix_cached_last_logits(
        self,
        residuals: Iterable[torch.Tensor],
        depth: int,
        cache: Any,
        *,
        position_offset: int,
    ) -> torch.Tensor:
        """Run only new suffix positions while updating a suffix-layer cache."""

        self._validate_depth(depth)
        residuals = list(residuals)
        if not residuals:
            raise ValueError("at least one residual is required")
        hidden = torch.cat(residuals, dim=1)
        hidden = self._run_layers(
            hidden,
            depth,
            self.num_layers,
            past_key_values=cache,
            position_offset=position_offset,
        )
        hidden = self.language_model.norm(hidden[:, -1:, :])
        return self.lm_head(hidden)[:, -1, :]

    def full_last_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._batch_tokens(tokens)
        output = self.model(input_ids=tokens, use_cache=False, logits_to_keep=1)
        return output.logits[:, -1, :]

    @torch.inference_mode()
    def write_full_prefix(self, tokens: torch.Tensor) -> FullPrefixState:
        """Build the standard exact cache used as the memory/TTFT baseline."""
        tokens = self._batch_tokens(tokens)
        if tokens.shape[1] == 0:
            raise ValueError("the prefix document must contain at least one token")
        output = self.language_model(input_ids=tokens, use_cache=True)
        length = int(tokens.shape[1])
        return FullPrefixState(
            document_length=length,
            current_length=length,
            cache=output.past_key_values,
        )

    @torch.inference_mode()
    def prefill_full_prefix(
        self, tokens: torch.Tensor
    ) -> tuple[torch.Tensor, FullPrefixState]:
        """Prefill a dense prefix once and return final logits plus its cache."""

        tokens = self._batch_tokens(tokens)
        if tokens.shape[1] == 0:
            raise ValueError("the dense prefix must contain at least one token")
        output = self.language_model(input_ids=tokens, use_cache=True)
        length = int(tokens.shape[1])
        state = FullPrefixState(
            document_length=length,
            current_length=length,
            cache=output.past_key_values,
        )
        return self.lm_head(output.last_hidden_state[:, -1, :]), state

    @torch.inference_mode()
    def continue_full_prefix(
        self, state: FullPrefixState, tokens: torch.Tensor
    ) -> torch.Tensor:
        """Extend a full-model prefix cache and return final-token logits."""
        tokens = self._batch_tokens(tokens)
        if tokens.shape[1] == 0:
            raise ValueError("prefix continuation must contain at least one token")
        output = self.language_model(
            input_ids=tokens,
            past_key_values=state.cache,
            use_cache=True,
        )
        state.current_length += int(tokens.shape[1])
        return self.lm_head(output.last_hidden_state[:, -1, :])


@torch.inference_mode()
def greedy_generate_dense(
    adapter: TorchSplitCausalLM,
    tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> list[int]:
    tokens = TorchSplitCausalLM._batch_tokens(tokens)
    logits, state = adapter.prefill_full_prefix(tokens)
    generated: list[int] = []
    for step in range(max_new_tokens):
        token = int(torch.argmax(logits, dim=-1).item())
        if token in eos_token_ids:
            break
        generated.append(token)
        if step + 1 < max_new_tokens:
            token_tensor = torch.tensor([[token]], device=tokens.device)
            logits = adapter.continue_full_prefix(state, token_tensor)
    return generated


@torch.inference_mode()
def greedy_generate_comem(
    adapter: TorchSplitCausalLM,
    document_residuals: list[torch.Tensor],
    query_tokens: torch.Tensor,
    *,
    depth: int,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> list[int]:
    query_tokens = TorchSplitCausalLM._batch_tokens(query_tokens)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        query_residual = adapter.run_to_depth(query_tokens, depth)
        logits = adapter.run_suffix_last_logits(
            [*document_residuals, query_residual], depth
        )
        token = int(torch.argmax(logits, dim=-1).item())
        if token in eos_token_ids:
            break
        generated.append(token)
        query_tokens = torch.cat(
            [query_tokens, torch.tensor([[token]], device=query_tokens.device)],
            dim=1,
        )
    return generated


@torch.inference_mode()
def greedy_generate_oracle(
    adapter: TorchSplitCausalLM,
    document_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    *,
    depth: int,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> list[int]:
    """Exact split oracle that recomputes the full lower prefix each token.

    This deliberately gives up document reuse.  Its purpose is to verify that
    the manual layer boundary itself is exact before measuring the error from
    query-local or chunk-local reusable document writes.
    """
    document_tokens = TorchSplitCausalLM._batch_tokens(document_tokens)
    query_tokens = TorchSplitCausalLM._batch_tokens(query_tokens)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        full_tokens = torch.cat([document_tokens, query_tokens], dim=1)
        residual = adapter.run_to_depth(full_tokens, depth)
        logits = adapter.run_suffix_last_logits([residual], depth)
        token = int(torch.argmax(logits, dim=-1).item())
        if token in eos_token_ids:
            break
        generated.append(token)
        query_tokens = torch.cat(
            [query_tokens, torch.tensor([[token]], device=query_tokens.device)],
            dim=1,
        )
    return generated


@torch.inference_mode()
def greedy_generate_replay(
    adapter: TorchSplitCausalLM,
    document_state: LowerReplayState | PackedLowerReplayState,
    query_tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> list[int]:
    """Generate with exact lower-layer document K/V and recurrent replay.

    Both lower and suffix caches are updated in-place.  The complete boundary
    sequence is consumed once; later steps run only the newly generated token.
    This preserves the exact split interface without quadratic recomputation.
    """
    query_tokens = TorchSplitCausalLM._batch_tokens(query_tokens)
    state = document_state.fork()
    query_residual = adapter.continue_lower_replay(state, query_tokens)
    suffix_cache = adapter.make_cache()
    # Match the chunk boundaries used by the standard prefix-cache path.
    # Qwen3.5 GatedDeltaNet/conv states are numerically sensitive to whether
    # document and query positions arrive in one chunk or two.  Seeding the
    # suffix with the stored document boundary first avoids an otherwise
    # growing decode discrepancy that is unrelated to state quantization.
    adapter.run_suffix_cached_last_logits(
        [state.document_residual],
        document_state.depth,
        suffix_cache,
        position_offset=0,
    )
    logits = adapter.run_suffix_cached_last_logits(
        [query_residual],
        document_state.depth,
        suffix_cache,
        position_offset=document_state.document_length,
    )
    suffix_length = state.current_length
    generated: list[int] = []
    for step in range(max_new_tokens):
        token = int(torch.argmax(logits, dim=-1).item())
        if token in eos_token_ids:
            break
        generated.append(token)
        if step + 1 < max_new_tokens:
            token_tensor = torch.tensor([[token]], device=query_tokens.device)
            token_residual = adapter.continue_lower_replay(state, token_tensor)
            logits = adapter.run_suffix_cached_last_logits(
                [token_residual],
                document_state.depth,
                suffix_cache,
                position_offset=suffix_length,
            )
            suffix_length += 1
    return generated


@torch.inference_mode()
def greedy_generate_full_prefix(
    adapter: TorchSplitCausalLM,
    document_state: FullPrefixState,
    query_tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> list[int]:
    """Generate from an exact full-model document prefix cache."""
    query_tokens = TorchSplitCausalLM._batch_tokens(query_tokens)
    state = document_state.fork()
    logits = adapter.continue_full_prefix(state, query_tokens)
    generated: list[int] = []
    for step in range(max_new_tokens):
        token = int(torch.argmax(logits, dim=-1).item())
        if token in eos_token_ids:
            break
        generated.append(token)
        if step + 1 < max_new_tokens:
            token_tensor = torch.tensor([[token]], device=query_tokens.device)
            logits = adapter.continue_full_prefix(state, token_tensor)
    return generated
