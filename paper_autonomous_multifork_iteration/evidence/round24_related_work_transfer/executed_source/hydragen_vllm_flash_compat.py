"""Compatibility frontend from Hydragen's FlashAttention-2 API to vLLM's fork.

The pinned Hydragen release imports two private functions from the standalone
``flash_attn`` package.  The frozen Qwen3.5 environment instead ships the
FlashAttention-2 fork embedded in vLLM.  This module exposes those two private
call signatures and replaces the pinned Triton-2 unique-KV frontend with the
equivalent variable-length call supported by the current stack.  Hydragen's
shared/unique decomposition and LSE-combination implementation are unchanged.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import torch


def _padded_lse(
    raw_lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
) -> torch.Tensor:
    batch = int(cu_seqlens_q.numel() - 1)
    heads = int(raw_lse.shape[0])
    output = torch.full(
        (batch, heads, int(max_seqlen_q)),
        -float("inf"),
        dtype=raw_lse.dtype,
        device=raw_lse.device,
    )
    for row in range(batch):
        start = int(cu_seqlens_q[row].item())
        end = int(cu_seqlens_q[row + 1].item())
        output[row, :, : end - start].copy_(raw_lse[:, start:end])
    return output


def _run_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    causal: bool,
    softmax_scale: float | None,
    window_size: tuple[int, int] | list[int],
    return_softmax: bool,
    uniform_query_length: bool = False,
) -> tuple[Any, ...]:
    if dropout_p != 0.0:
        raise ValueError("Hydragen transfer requires dropout_p=0")
    if return_softmax:
        raise ValueError("Hydragen transfer does not request attention probabilities")
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    out, raw_lse = flash_attn_varlen_func(
        q,
        k,
        v,
        max_seqlen_q=int(max_seqlen_q),
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_k=int(max_seqlen_k),
        cu_seqlens_k=cu_seqlens_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        window_size=list(window_size),
        return_attn_probs=False,
        return_softmax_lse=True,
        fa_version=2,
    )
    if uniform_query_length:
        batch = int(cu_seqlens_q.numel() - 1)
        lse = raw_lse.reshape(
            raw_lse.shape[0], batch, int(max_seqlen_q)
        ).permute(1, 0, 2).contiguous()
    else:
        lse = _padded_lse(raw_lse, cu_seqlens_q, int(max_seqlen_q))
    return out, q, k, v, None, lse, None, None


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    causal: bool,
    softmax_scale: float | None,
    window_size: tuple[int, int] | list[int],
    return_softmax: bool,
    **_: Any,
) -> tuple[Any, ...]:
    return _run_varlen(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size=window_size,
        return_softmax=return_softmax,
    )


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    dropout_p: float,
    causal: bool,
    softmax_scale: float | None,
    window_size: tuple[int, int] | list[int],
    return_softmax: bool,
    **_: Any,
) -> tuple[Any, ...]:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("fixed FlashAttention inputs must be rank four")
    batch, qlength = q.shape[:2]
    klength = k.shape[1]
    q_flat = q.reshape(batch * qlength, *q.shape[2:])
    k_flat = k.reshape(batch * klength, *k.shape[2:])
    v_flat = v.reshape(batch * klength, *v.shape[2:])
    cu_q = torch.arange(
        0,
        (batch + 1) * qlength,
        qlength,
        dtype=torch.int32,
        device=q.device,
    )
    cu_k = torch.arange(
        0,
        (batch + 1) * klength,
        klength,
        dtype=torch.int32,
        device=q.device,
    )
    result = _run_varlen(
        q_flat,
        k_flat,
        v_flat,
        cu_seqlens_q=cu_q,
        cu_seqlens_k=cu_k,
        max_seqlen_q=qlength,
        max_seqlen_k=klength,
        dropout_p=dropout_p,
        causal=causal,
        softmax_scale=softmax_scale,
        window_size=window_size,
        return_softmax=return_softmax,
        uniform_query_length=True,
    )
    out = result[0].reshape_as(q)
    return out, q, k, v, None, result[5], None, None


def install_flash_attn_compat() -> None:
    """Install the two-function compatibility module before Hydragen import."""
    package = types.ModuleType("flash_attn")
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface._flash_attn_forward = _flash_attn_forward
    interface._flash_attn_varlen_forward = _flash_attn_varlen_forward
    package.flash_attn_interface = interface
    sys.modules["flash_attn"] = package
    sys.modules["flash_attn.flash_attn_interface"] = interface


def flash_attention_seqlen_compat(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_len: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Current-stack replacement for Hydragen's Triton-2 unique-KV kernel.

    The padded K/V storage is passed directly to vLLM FlashAttention-2 and the
    preregistered per-row lengths are supplied through ``seqused_k``.  No K/V
    concatenation or compaction is performed inside the measured call.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Hydragen seqlen inputs must be rank four")
    if k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError("Hydragen seqlen batch/KV shape mismatch")
    batch, qlength = q.shape[:2]
    klength = k.shape[1]
    if tuple(seq_len.shape) != (batch,):
        raise ValueError("seq_len geometry mismatch")
    if klength % 16 != 0:
        raise ValueError("paged unique-KV block size must be divisible by 16")
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    q_flat = q.reshape(batch * qlength, *q.shape[2:])
    cu_q = torch.arange(
        0,
        (batch + 1) * qlength,
        qlength,
        dtype=torch.int32,
        device=q.device,
    )
    # vLLM's ``seqused_k`` path is the paged-KV interface.  Treat every
    # request's already-padded unique K/V row as one page; this preserves the
    # input storage and avoids concatenation or length-dependent compaction.
    block_table = torch.arange(
        batch,
        dtype=torch.int32,
        device=q.device,
    ).reshape(batch, 1)
    out, raw_lse = flash_attn_varlen_func(
        q_flat,
        k,
        v,
        max_seqlen_q=qlength,
        cu_seqlens_q=cu_q,
        max_seqlen_k=klength,
        cu_seqlens_k=None,
        seqused_k=seq_len.to(torch.int32),
        block_table=block_table,
        dropout_p=0.0,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        window_size=[-1, -1],
        return_attn_probs=False,
        return_softmax_lse=True,
        fa_version=2,
    )
    out = out.reshape_as(q)
    lse = raw_lse.transpose(0, 1).reshape(batch, qlength, q.shape[2]).contiguous()
    return out, lse


def install_hydragen_runtime_compat(attention_module: Any, flash_module: Any) -> None:
    """Patch only the obsolete unique-KV kernel frontend in the pinned release."""
    flash_module.flash_attention_seqlen = flash_attention_seqlen_compat
    attention_module.flash_attention_seqlen = flash_attention_seqlen_compat
