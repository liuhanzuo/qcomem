from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class FunctionalKVState:
    """Immutable full-attention K/V history, shaped ``[B,H,T,Dh]``."""

    keys: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class FunctionalGDNState:
    """Immutable causal-convolution tail and recurrent delta state."""

    conv_buffer: torch.Tensor
    recurrent: torch.Tensor


@dataclass(frozen=True)
class FunctionalHybridState:
    attention: FunctionalKVState | None = None
    gdn: FunctionalGDNState | None = None


def state_tensors(state: object) -> tuple[torch.Tensor, ...]:
    if isinstance(state, torch.Tensor):
        return (state,)
    if state is None:
        return ()
    if hasattr(state, "__dataclass_fields__"):
        result: list[torch.Tensor] = []
        for name in state.__dataclass_fields__:
            result.extend(state_tensors(getattr(state, name)))
        return tuple(result)
    raise TypeError(f"unsupported functional state type: {type(state)!r}")


def assert_out_of_place_transition(old: object, new: object) -> None:
    """Fail if a returned tensor aliases storage from its input state."""

    old_pointers = {tensor.untyped_storage().data_ptr() for tensor in state_tensors(old)}
    for tensor in state_tensors(new):
        if tensor.untyped_storage().data_ptr() in old_pointers:
            raise RuntimeError("functional cache transition reused mutable input storage")


class FunctionalFullAttention(nn.Module):
    """Small causal attention reference whose cache update is always ``cat``."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads:
            raise ValueError("width must be divisible by heads")
        self.width = width
        self.heads = heads
        self.head_dim = width // heads
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden: torch.Tensor,
        state: FunctionalKVState | None = None,
    ) -> tuple[torch.Tensor, FunctionalKVState]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError("hidden must have shape [batch,tokens,width]")
        query = self._heads(self.q_proj(hidden))
        new_keys = self._heads(self.k_proj(hidden))
        new_values = self._heads(self.v_proj(hidden))
        if state is None:
            keys = new_keys
            values = new_values
            past_length = 0
        else:
            keys = torch.cat((state.keys, new_keys), dim=2)
            values = torch.cat((state.values, new_values), dim=2)
            past_length = state.keys.shape[2]
        scores = torch.matmul(query.float(), keys.float().transpose(-1, -2))
        scores = scores / self.head_dim**0.5
        query_positions = past_length + torch.arange(
            hidden.shape[1], device=hidden.device
        )
        key_positions = torch.arange(keys.shape[2], device=hidden.device)
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal.view(1, 1, *causal.shape), float("-inf"))
        probabilities = F.softmax(scores, dim=-1).to(values.dtype)
        output = torch.matmul(probabilities, values)
        output = output.transpose(1, 2).reshape(hidden.shape)
        next_state = FunctionalKVState(keys=keys, values=values)
        if state is not None:
            assert_out_of_place_transition(state, next_state)
        return self.o_proj(output), next_state


class FunctionalGatedDeltaNet(nn.Module):
    """Minimal functional GDN reference with conv and recurrent state.

    This is not a numerical reproduction of the fused Qwen3.5 kernel. It
    captures the two state categories and the required recurrence contract:
    every token produces a new recurrent matrix and every call returns a new
    convolution tail, with no write into the input state.
    """

    def __init__(self, width: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size must be at least 2")
        self.width = width
        self.kernel_size = kernel_size
        self.conv_weight = nn.Parameter(torch.empty(kernel_size, width))
        self.conv_bias = nn.Parameter(torch.zeros(width))
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.beta_proj = nn.Linear(width, width, bias=True)
        self.decay_proj = nn.Linear(width, width, bias=True)
        self.o_proj = nn.Linear(width, width, bias=False)
        nn.init.normal_(self.conv_weight, std=0.2)

    def _initial_state(self, hidden: torch.Tensor) -> FunctionalGDNState:
        return FunctionalGDNState(
            conv_buffer=hidden.new_zeros(
                hidden.shape[0], self.kernel_size - 1, self.width
            ),
            recurrent=hidden.new_zeros(hidden.shape[0], self.width, self.width),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        state: FunctionalGDNState | None = None,
    ) -> tuple[torch.Tensor, FunctionalGDNState]:
        if hidden.ndim != 3 or hidden.shape[-1] != self.width:
            raise ValueError("hidden must have shape [batch,tokens,width]")
        previous = self._initial_state(hidden) if state is None else state
        combined = torch.cat((previous.conv_buffer, hidden), dim=1)
        windows = torch.stack(
            [
                combined[:, offset : offset + hidden.shape[1], :]
                for offset in range(self.kernel_size)
            ],
            dim=2,
        )
        convolved = torch.sum(
            windows * self.conv_weight.view(1, 1, self.kernel_size, self.width),
            dim=2,
        ) + self.conv_bias
        next_conv_buffer = combined[:, -(self.kernel_size - 1) :, :].clone()

        queries = torch.tanh(self.q_proj(convolved))
        keys = torch.tanh(self.k_proj(convolved))
        values = self.v_proj(convolved)
        betas = torch.sigmoid(self.beta_proj(convolved))
        decays = torch.sigmoid(self.decay_proj(convolved))
        recurrent = previous.recurrent
        outputs = []
        for position in range(hidden.shape[1]):
            key = keys[:, position, :]
            value = values[:, position, :]
            beta = betas[:, position, :]
            decay = decays[:, position, :]
            recurrent = (
                recurrent * decay.unsqueeze(-1)
                + torch.einsum("bi,bj->bij", key, beta * value)
            )
            outputs.append(
                torch.einsum("bi,bij->bj", queries[:, position, :], recurrent)
            )
        output = torch.stack(outputs, dim=1)
        next_state = FunctionalGDNState(
            conv_buffer=next_conv_buffer,
            recurrent=recurrent,
        )
        if state is not None:
            assert_out_of_place_transition(state, next_state)
        return self.o_proj(output), next_state


class TinyFunctionalHybridLayer(nn.Module):
    """One causal full-attention + GDN layer for segmentation tests."""

    def __init__(self, width: int = 8, heads: int = 2, kernel_size: int = 3) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = FunctionalFullAttention(width, heads)
        self.gdn_norm = nn.LayerNorm(width)
        self.gdn = FunctionalGatedDeltaNet(width, kernel_size)

    def forward(
        self,
        hidden: torch.Tensor,
        state: FunctionalHybridState | None = None,
    ) -> tuple[torch.Tensor, FunctionalHybridState]:
        previous = FunctionalHybridState() if state is None else state
        attention_output, attention_state = self.attention(
            self.attention_norm(hidden), previous.attention
        )
        hidden = hidden + attention_output
        gdn_output, gdn_state = self.gdn(self.gdn_norm(hidden), previous.gdn)
        hidden = hidden + gdn_output
        next_state = FunctionalHybridState(
            attention=attention_state,
            gdn=gdn_state,
        )
        if state is not None:
            assert_out_of_place_transition(state, next_state)
        return hidden, next_state


def run_segments(
    layer: TinyFunctionalHybridLayer,
    segments: Iterable[torch.Tensor],
) -> tuple[torch.Tensor, FunctionalHybridState]:
    state = FunctionalHybridState()
    outputs = []
    for segment in segments:
        output, state = layer(segment, state)
        outputs.append(output)
    if not outputs:
        raise ValueError("at least one segment is required")
    return torch.cat(outputs, dim=1), state
