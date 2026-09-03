"""Bounded Palu head-wise SVD helpers for Qwen3.5 K/V projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


class PaluTransferError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PaluTransferError(message)


@dataclass(frozen=True)
class HeadFactors:
    left: torch.Tensor
    right: torch.Tensor


def headwise_svd_factors(weight: torch.Tensor, *, heads: int) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Return full per-head U/S/Vh factors using Palu's decomposition axis."""
    require(weight.ndim == 2, "projection weight must be rank two")
    require(weight.shape[0] % heads == 0, "output width must divide into heads")
    rows = weight.reshape(heads, weight.shape[0] // heads, weight.shape[1])
    factors = []
    for row in rows:
        factors.append(torch.linalg.svd(row.float(), full_matrices=False))
    return factors


def activation_whitener(hidden: torch.Tensor) -> tuple[torch.Tensor, dict[str, float | int]]:
    """Reproduce Palu's Cholesky whitening matrix on frozen activations."""
    require(hidden.ndim == 3, "calibration hidden states must be rank three")
    flat = hidden.reshape(-1, hidden.shape[-1]).float()
    covariance = flat.transpose(0, 1) @ flat
    covariance = (covariance + covariance.transpose(0, 1)) * 0.5
    minimum_eigenvalue = float(torch.linalg.eigvalsh(covariance).amin().item())
    jitter = max(0.0, -minimum_eigenvalue + 1e-3)
    if jitter:
        covariance = covariance + jitter * torch.eye(
            covariance.shape[0], dtype=covariance.dtype, device=covariance.device
        )
    scale = torch.linalg.cholesky(covariance)
    return scale, {
        "calibration_tokens": int(flat.shape[0]),
        "hidden_size": int(flat.shape[1]),
        "minimum_eigenvalue_before_jitter": minimum_eigenvalue,
        "diagonal_jitter": jitter,
    }


def whitened_headwise_svd_factors(
    weight: torch.Tensor,
    *,
    heads: int,
    scale: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Apply Palu's W@S SVD and postmultiply Vh by inv(S)."""
    require(scale.shape == (weight.shape[1], weight.shape[1]), "whitener geometry drift")
    inverse = torch.linalg.inv(scale.float())
    rows = weight.reshape(heads, weight.shape[0] // heads, weight.shape[1])
    factors = []
    for row in rows:
        u, singular, vh = torch.linalg.svd(row.float() @ scale.float(), full_matrices=False)
        factors.append((u, singular, vh @ inverse))
    return factors


def truncate_factors(
    factors: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    rank: int,
    dtype: torch.dtype,
) -> list[HeadFactors]:
    result = []
    for u, singular, vh in factors:
        require(0 < rank <= singular.numel(), "invalid Palu rank")
        root = singular[:rank].sqrt()
        left = (u[:, :rank] * root.unsqueeze(0)).to(dtype).contiguous()
        right = (root.unsqueeze(1) * vh[:rank]).to(dtype).contiguous()
        result.append(HeadFactors(left=left, right=right))
    return result


def apply_headwise(
    hidden: torch.Tensor,
    factors: list[HeadFactors],
    *,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    outputs = []
    offset = 0
    for factor in factors:
        head_bias = None
        if bias is not None:
            head_bias = bias[offset : offset + factor.left.shape[0]]
        latent = F.linear(hidden, factor.right)
        outputs.append(F.linear(latent, factor.left, head_bias))
        offset += factor.left.shape[0]
    return torch.cat(outputs, dim=-1)


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    delta = candidate.float() - reference.float()
    denominator = torch.linalg.vector_norm(reference.float()).item()
    return float(torch.linalg.vector_norm(delta).item() / max(denominator, 1e-30))


def logical_kv_storage(*, rank: int, heads: int = 2, head_dim: int = 256) -> dict[str, float | int]:
    dense = 2 * heads * head_dim * 2
    low_rank = 2 * heads * rank * 2
    return {
        "dense_kv_bytes_per_token": dense,
        "palu_latent_kv_bytes_per_token": low_rank,
        "dense_over_palu_ratio": dense / low_rank,
        "bytes_avoided_per_token": dense - low_rank,
    }
