from __future__ import annotations

"""Autograd-safe Qwen3.5 linear-attention cache updates.

Transformers' Qwen3.5 GatedDeltaNet reads the recurrent/conv tensors from a
``LinearAttentionLayer`` and writes the new tensors back through
``Cache.update_*``.  The stock cache keeps static addresses with ``copy_``;
that is useful for inference/CUDA graphs, but it invalidates tensors saved for
backward across a document-prefill -> query-continuation boundary.

This adapter keeps the model kernels and decoder implementation unchanged and
only replaces those two writes with functional tensor rebinding.  Full
attention remains on Transformers' ordinary ``DynamicLayer.update`` whose
``torch.cat`` assignment is already out of place and differentiable.
"""

from dataclasses import dataclass
from types import MethodType
from typing import Any

import torch


class NativeFunctionalCacheError(RuntimeError):
    """Raised when a cache cannot be made functional without guessing."""


@dataclass(frozen=True)
class NativeFunctionalCacheInstall:
    linear_layer_indices: tuple[int, ...]
    full_attention_layer_indices: tuple[int, ...]
    installed_linear_layers: int
    update_mode: str = "native-kernels-functional-state-rebind"
    full_attention_update_mode: str = "transformers-dynamic-layer-out-of-place-cat"


def _require_state_index(layer: Any, state_idx: int) -> None:
    if isinstance(state_idx, bool) or not isinstance(state_idx, int) or state_idx < 0:
        raise NativeFunctionalCacheError("state_idx must be a non-negative integer")
    required = (
        "conv_states",
        "recurrent_states",
        "is_conv_states_initialized",
        "is_recurrent_states_initialized",
        "has_previous_state",
        "conv_kernel_size",
    )
    for name in required:
        value = getattr(layer, name, None)
        if not isinstance(value, dict):
            raise NativeFunctionalCacheError(
                f"linear cache layer.{name} must be a dict"
            )


def _functional_update_conv_state(
    layer: Any,
    conv_states: torch.Tensor,
    state_idx: int = 0,
    conv_kernel_size: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    del kwargs
    _require_state_index(layer, state_idx)
    if not isinstance(conv_states, torch.Tensor) or conv_states.ndim < 1:
        raise NativeFunctionalCacheError("conv_states must be a non-empty-rank tensor")
    if not layer.is_conv_states_initialized.get(state_idx, False):
        layer.lazy_initialization(
            conv_states=conv_states,
            state_idx=state_idx,
            conv_kernel_size=conv_kernel_size,
        )
    kernel_size = layer.conv_kernel_size.get(state_idx)
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int) or kernel_size < 1:
        raise NativeFunctionalCacheError("conv kernel size was not initialized")

    if not layer.has_previous_state.get(state_idx, False):
        full_conv_states = conv_states
        layer.has_previous_state[state_idx] = True
        if not layer.record_past and full_conv_states.shape[-1] < kernel_size:
            full_conv_states = torch.nn.functional.pad(
                full_conv_states,
                (kernel_size - full_conv_states.shape[-1], 0),
                value=0,
            )
    else:
        previous = layer.conv_states.get(state_idx)
        if not isinstance(previous, torch.Tensor):
            raise NativeFunctionalCacheError("initialized conv state is not a tensor")
        full_conv_states = torch.cat([previous, conv_states], dim=-1)

    # The stock implementation uses ``copy_`` here. Rebinding preserves the
    # old tensor/version for autograd and makes the new state query-owned.
    next_state = (
        # Clone compacts the tail. A view would keep the entire document/query
        # projection allocation alive and defeat the fixed-size cache contract.
        full_conv_states[..., -kernel_size:].clone()
        if not layer.record_past
        else full_conv_states
    )
    layer.conv_states[state_idx] = next_state
    return full_conv_states


def _functional_update_recurrent_state(
    layer: Any,
    recurrent_states: torch.Tensor,
    state_idx: int = 0,
    **kwargs: Any,
) -> torch.Tensor:
    del kwargs
    _require_state_index(layer, state_idx)
    if not isinstance(recurrent_states, torch.Tensor) or recurrent_states.ndim < 1:
        raise NativeFunctionalCacheError(
            "recurrent_states must be a non-empty-rank tensor"
        )
    if not layer.is_recurrent_states_initialized.get(state_idx, False):
        layer.lazy_initialization(
            recurrent_states=recurrent_states,
            state_idx=state_idx,
        )
    layer.recurrent_states[state_idx] = recurrent_states
    return recurrent_states


def install_native_functional_linear_cache(
    cache: Any,
    config: Any,
) -> NativeFunctionalCacheInstall:
    """Patch every configured linear cache layer, or fail without partial use."""

    layers = getattr(cache, "layers", None)
    layer_types = getattr(config, "layer_types", None)
    if not isinstance(layers, (list, tuple)):
        raise NativeFunctionalCacheError("cache.layers must be a sequence")
    if not isinstance(layer_types, (list, tuple)) or len(layer_types) != len(layers):
        raise NativeFunctionalCacheError(
            "config.layer_types must match the initialized cache layers"
        )
    unknown = sorted(set(layer_types) - {"linear_attention", "full_attention"})
    if unknown:
        raise NativeFunctionalCacheError(f"unsupported layer types: {unknown}")
    linear = tuple(i for i, kind in enumerate(layer_types) if kind == "linear_attention")
    full = tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention")
    if not linear or not full:
        raise NativeFunctionalCacheError(
            "Qwen3.5 hybrid cache must contain both linear and full attention"
        )

    # Validate the entire target set before mutating any method binding.
    for index in linear:
        layer = layers[index]
        _require_state_index(layer, 0)
        if not callable(getattr(layer, "lazy_initialization", None)):
            raise NativeFunctionalCacheError(
                f"linear cache layer {index} has no lazy_initialization"
            )
        if not isinstance(getattr(layer, "record_past", None), bool):
            raise NativeFunctionalCacheError(
                f"linear cache layer {index}.record_past must be bool"
            )

    for index in linear:
        layer = layers[index]
        layer.update_conv_state = MethodType(_functional_update_conv_state, layer)
        layer.update_recurrent_state = MethodType(
            _functional_update_recurrent_state, layer
        )
        layer._qcomem_update_mode = "functional-state-rebind"

    installed = tuple(
        index
        for index in linear
        if getattr(layers[index], "_qcomem_update_mode", None)
        == "functional-state-rebind"
    )
    if installed != linear:
        raise NativeFunctionalCacheError(
            "not every configured linear cache layer was patched"
        )
    return NativeFunctionalCacheInstall(
        linear_layer_indices=linear,
        full_attention_layer_indices=full,
        installed_linear_layers=len(installed),
    )


def functional_linear_cache_telemetry(cache: Any, install: NativeFunctionalCacheInstall) -> dict[str, Any]:
    layers = cache.layers
    rebound = tuple(
        index
        for index in install.linear_layer_indices
        if getattr(layers[index], "_qcomem_update_mode", None)
        == "functional-state-rebind"
    )
    return {
        "update_mode": install.update_mode,
        "linear_layer_indices": install.linear_layer_indices,
        "full_attention_layer_indices": install.full_attention_layer_indices,
        "installed_linear_layers": install.installed_linear_layers,
        "currently_rebound_linear_layers": rebound,
        "all_linear_layers_intercepted": rebound == install.linear_layer_indices,
        "mutable_copy_updates_used": False,
        "full_attention_update_mode": install.full_attention_update_mode,
    }


__all__ = [
    "NativeFunctionalCacheError",
    "NativeFunctionalCacheInstall",
    "functional_linear_cache_telemetry",
    "install_native_functional_linear_cache",
]
