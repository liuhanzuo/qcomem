"""Independent FP32 attention oracle for the reviewer-triggered fork audit.

The module deliberately has no model, vLLM, or CUDA initialization at import
time.  It accepts already-computed post-RoPE Q/K/V tensors, constructs a
canonical cached-suffix causal mask from explicit positions, and evaluates a
candidate output against a dense FP32 reference.

Tensor layouts are explicit:

* query: ``[batch, query_heads, query_tokens, head_dim]``
* key/value: ``[batch, kv_heads, key_tokens, head_dim]``
* returned reference and candidate: ``[batch, query_tokens, query_heads,
  head_dim]``

The position contract is intentionally narrower than a general attention API:
key positions must be contiguous and query positions must equal the suffix of
the key positions.  This is the contract exercised by cached decoding and
makes position off-by-one mutants observable rather than silently accepted.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Mapping

import torch


ORACLE_SCHEMA_VERSION = "qcomem.forkaudit.oracle.v2"
DEFAULT_MAX_RELATIVE_L2 = 0.005


class OracleGateError(ValueError):
    """A named precondition or numerical gate failed."""

    def __init__(self, gate_id: str, message: str) -> None:
        super().__init__(f"{gate_id}: {message}")
        self.gate_id = gate_id
        self.detail = message


class ThresholdLockedError(RuntimeError):
    """Raised when an observed preregistration is changed."""


def _require(condition: bool, gate_id: str, message: str) -> None:
    if not condition:
        raise OracleGateError(gate_id, message)


def _is_supported_float(tensor: torch.Tensor) -> bool:
    return bool(tensor.is_floating_point() and not tensor.is_complex())


def _canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _float32_backend_state() -> dict[str, Any]:
    """Read only the process-global controls used by the dense reference."""

    state: dict[str, Any] = {
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    cuda_matmul = torch.backends.cuda.matmul
    if hasattr(cuda_matmul, "fp32_precision"):
        state["cuda_matmul_control"] = "fp32_precision"
        state["cuda_matmul_fp32_precision"] = str(cuda_matmul.fp32_precision)
    else:
        state["cuda_matmul_control"] = "allow_tf32"
        state["cuda_matmul_allow_tf32"] = bool(cuda_matmul.allow_tf32)
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        state["cudnn_allow_tf32"] = bool(torch.backends.cudnn.allow_tf32)
    return state


def _set_ieee_float32_backend() -> None:
    torch.set_float32_matmul_precision("highest")
    cuda_matmul = torch.backends.cuda.matmul
    if hasattr(cuda_matmul, "fp32_precision"):
        cuda_matmul.fp32_precision = "ieee"
    else:
        cuda_matmul.allow_tf32 = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False


def _restore_float32_backend(state: Mapping[str, Any]) -> None:
    cuda_matmul = torch.backends.cuda.matmul
    if state["cuda_matmul_control"] == "fp32_precision":
        cuda_matmul.fp32_precision = str(state["cuda_matmul_fp32_precision"])
    else:
        cuda_matmul.allow_tf32 = bool(state["cuda_matmul_allow_tf32"])
    if "cudnn_allow_tf32" in state:
        torch.backends.cudnn.allow_tf32 = bool(state["cudnn_allow_tf32"])
    # Set this last: on legacy PyTorch, assigning ``allow_tf32`` changes what
    # ``get_float32_matmul_precision`` reports (for example medium -> high).
    torch.set_float32_matmul_precision(str(state["float32_matmul_precision"]))


def _restore_cpu_float32_backend(state: Mapping[str, Any]) -> None:
    """Restore the CPU reference control without assigning CUDA sentinels."""

    expected = str(state["float32_matmul_precision"])
    if torch.get_float32_matmul_precision() != expected:
        torch.set_float32_matmul_precision(expected)


def _ieee_backend_is_effective(state: Mapping[str, Any]) -> bool:
    if state.get("float32_matmul_precision") != "highest":
        return False
    if state.get("cuda_matmul_control") == "fp32_precision":
        return state.get("cuda_matmul_fp32_precision") == "ieee"
    return state.get("cuda_matmul_allow_tf32") is False


@contextmanager
def ieee_fp32_reference_context(*, device_type: str):
    """Temporarily force IEEE FP32 matmul for the reference, then restore it.

    This context surrounds only the independent dense reference.  The
    candidate fused kernel must be executed before entering it.
    """

    reference_device_type = str(device_type)
    before = _float32_backend_state()
    try:
        # A CPU-only diagnostic must not mutate CUDA controls.  On PyTorch
        # builds where CUDA has not been initialized, ``fp32_precision`` can
        # report the read-only sentinel ``none``; assigning that string back
        # is ignored and would make an otherwise CPU-only check appear to
        # leak global state.  The formal CUDA oracle still exercises the full
        # CUDA/cuDNN IEEE policy below.
        if reference_device_type == "cuda":
            _set_ieee_float32_backend()
            effective = _float32_backend_state()
        else:
            # Keep the reference's explicit highest-precision contract, but
            # do not assign CUDA/cuDNN controls from a CPU-only process.  In
            # particular, PyTorch 2.11 can expose the pre-initialization
            # ``fp32_precision=none`` sentinel, which is not a writable
            # restoration value.
            if before.get("float32_matmul_precision") != "highest":
                torch.set_float32_matmul_precision("highest")
            effective = _float32_backend_state()
        _require(
            (
                _ieee_backend_is_effective(effective)
                if reference_device_type == "cuda"
                else effective.get("float32_matmul_precision") == "highest"
            ),
            "ORACLE_REFERENCE_PRECISION",
            "IEEE FP32 matmul controls did not become effective",
        )
    except Exception as exc:
        try:
            if reference_device_type == "cuda":
                _restore_float32_backend(before)
            else:
                _restore_cpu_float32_backend(before)
        except Exception as restore_exc:
            raise OracleGateError(
                "ORACLE_REFERENCE_PRECISION_RESTORE",
                f"failed to restore backend after configuration error: {restore_exc}",
            ) from restore_exc
        if isinstance(exc, OracleGateError):
            raise
        raise OracleGateError(
            "ORACLE_REFERENCE_PRECISION",
            f"failed to configure IEEE FP32 reference: {exc}",
        ) from exc

    audit: dict[str, Any] = {
        "policy": "ieee-fp32-reference-only",
        "device_type": reference_device_type,
        "applies_to_cuda_reference": reference_device_type == "cuda",
        "candidate_executed_outside_context": True,
        "before": before,
        "effective": effective,
        "effective_ieee_fp32": True,
    }
    try:
        yield audit
    finally:
        try:
            if reference_device_type == "cuda":
                _restore_float32_backend(before)
            else:
                _restore_cpu_float32_backend(before)
            after = _float32_backend_state()
        except Exception as exc:
            raise OracleGateError(
                "ORACLE_REFERENCE_PRECISION_RESTORE",
                f"failed to restore float32 backend controls: {exc}",
            ) from exc
        audit["after"] = after
        audit["restored"] = after == before
        if not audit["restored"]:
            raise OracleGateError(
                "ORACLE_REFERENCE_PRECISION_RESTORE",
                "float32 backend controls differ after reference evaluation",
            )


@dataclass(frozen=True)
class OracleThresholds:
    """Immutable numerical gates chosen before observing an oracle result."""

    max_relative_l2: float = DEFAULT_MAX_RELATIVE_L2
    max_absolute_error: float | None = None
    require_logits_top1_match: bool = False
    max_mean_forward_kl: float | None = None

    def __post_init__(self) -> None:
        _require(
            math.isfinite(self.max_relative_l2)
            and self.max_relative_l2 >= 0.0,
            "ORACLE_THRESHOLD_CONFIG",
            "max_relative_l2 must be finite and non-negative",
        )
        if self.max_absolute_error is not None:
            _require(
                math.isfinite(self.max_absolute_error)
                and self.max_absolute_error >= 0.0,
                "ORACLE_THRESHOLD_CONFIG",
                "max_absolute_error must be finite and non-negative",
            )
        if self.max_mean_forward_kl is not None:
            _require(
                math.isfinite(self.max_mean_forward_kl)
                and self.max_mean_forward_kl >= 0.0,
                "ORACLE_THRESHOLD_CONFIG",
                "max_mean_forward_kl must be finite and non-negative",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_relative_l2": float(self.max_relative_l2),
            "max_absolute_error": (
                None
                if self.max_absolute_error is None
                else float(self.max_absolute_error)
            ),
            "require_logits_top1_match": bool(self.require_logits_top1_match),
            "max_mean_forward_kl": (
                None
                if self.max_mean_forward_kl is None
                else float(self.max_mean_forward_kl)
            ),
        }


@dataclass(frozen=True)
class TensorErrorMetrics:
    bitwise_exact: bool
    finite: bool
    max_abs: float
    mean_abs: float
    relative_l2: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "bitwise_exact": self.bitwise_exact,
            "finite": self.finite,
            "max_abs": self.max_abs,
            "mean_abs": self.mean_abs,
            "relative_l2": self.relative_l2,
        }


@dataclass(frozen=True)
class LogitMetrics:
    top1_match: bool
    top1_agreement: float
    mean_forward_kl: float
    max_forward_kl: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "top1_match": self.top1_match,
            "top1_agreement": self.top1_agreement,
            "mean_forward_kl": self.mean_forward_kl,
            "max_forward_kl": self.max_forward_kl,
        }


@dataclass(frozen=True)
class DenseAttentionResult:
    output: torch.Tensor
    position_contract: Mapping[str, Any]
    scaling: float
    grouped_query_factor: int
    precision_audit: Mapping[str, Any]


@dataclass(frozen=True)
class GateResult:
    gate_id: str
    passed: bool
    is_gate: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "passed": self.passed,
            "is_gate": self.is_gate,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class OracleOutcome:
    """JSON-serializable result apart from its convenience dataclass wrapper."""

    status: str
    passed: bool
    preregistration_sha256: str
    evaluation_index: int
    thresholds: Mapping[str, Any]
    gates: tuple[GateResult, ...]
    attention_metrics: TensorErrorMetrics | None
    logits_metrics: LogitMetrics | None
    reference: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ORACLE_SCHEMA_VERSION,
            "status": self.status,
            "passed": self.passed,
            "preregistration_sha256": self.preregistration_sha256,
            "evaluation_index": self.evaluation_index,
            "thresholds": dict(self.thresholds),
            "gates": [gate.to_dict() for gate in self.gates],
            "attention_metrics": (
                None
                if self.attention_metrics is None
                else self.attention_metrics.to_dict()
            ),
            "logits_metrics": (
                None if self.logits_metrics is None else self.logits_metrics.to_dict()
            ),
            "reference": None if self.reference is None else dict(self.reference),
            "failure": None if self.failure is None else dict(self.failure),
        }


def _normalize_positions(
    positions: torch.Tensor | None,
    *,
    batch: int,
    length: int,
    device: torch.device,
    default_start: int,
    label: str,
) -> torch.Tensor:
    gate_id = "ORACLE_POSITION_CONTRACT"
    if positions is None:
        normalized = torch.arange(
            default_start, default_start + length, device=device, dtype=torch.int64
        ).unsqueeze(0).expand(batch, -1)
    else:
        _require(
            isinstance(positions, torch.Tensor),
            gate_id,
            f"{label} positions must be a tensor or None",
        )
        _require(
            positions.dtype
            in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            },
            gate_id,
            f"{label} positions must have an integer dtype",
        )
        _require(
            positions.ndim in {1, 2},
            gate_id,
            f"{label} positions must have shape [tokens] or [batch, tokens]",
        )
        if positions.ndim == 1:
            _require(
                tuple(positions.shape) == (length,),
                gate_id,
                f"{label} positions length does not match its tensor",
            )
            normalized = positions.to(device=device, dtype=torch.int64).unsqueeze(0)
            normalized = normalized.expand(batch, -1)
        else:
            _require(
                tuple(positions.shape) == (batch, length),
                gate_id,
                f"{label} positions must have shape [{batch}, {length}]",
            )
            normalized = positions.to(device=device, dtype=torch.int64)
    _require(
        bool((normalized >= 0).all().item()),
        gate_id,
        f"{label} positions must be non-negative",
    )
    if length > 1:
        deltas = normalized[:, 1:] - normalized[:, :-1]
        _require(
            bool((deltas == 1).all().item()),
            gate_id,
            f"{label} positions must be contiguous with unit stride",
        )
    return normalized


def _validate_qkv(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> tuple[int, int, int, int, int, int]:
    _require(
        all(isinstance(tensor, torch.Tensor) for tensor in (query, key, value)),
        "ORACLE_INPUT_TYPE",
        "query, key, and value must be torch tensors",
    )
    _require(
        query.ndim == key.ndim == value.ndim == 4,
        "ORACLE_INPUT_SHAPE",
        "query, key, and value must be rank-4 BHSD tensors",
    )
    _require(
        _is_supported_float(query)
        and _is_supported_float(key)
        and _is_supported_float(value),
        "ORACLE_INPUT_DTYPE",
        "query, key, and value must have real floating dtypes",
    )
    _require(
        query.dtype == key.dtype == value.dtype,
        "ORACLE_INPUT_DTYPE",
        "query, key, and value must have the same dtype",
    )
    _require(
        query.device == key.device == value.device,
        "ORACLE_INPUT_DEVICE",
        "query, key, and value must be on the same device",
    )
    batch, query_heads, query_tokens, head_dim = map(int, query.shape)
    key_batch, kv_heads, key_tokens, key_dim = map(int, key.shape)
    value_batch, value_heads, value_tokens, value_dim = map(int, value.shape)
    _require(
        batch > 0
        and query_heads > 0
        and query_tokens > 0
        and head_dim > 0
        and kv_heads > 0
        and key_tokens > 0,
        "ORACLE_INPUT_SHAPE",
        "Q/K/V dimensions must be non-empty",
    )
    _require(
        batch == key_batch == value_batch,
        "ORACLE_INPUT_SHAPE",
        "Q/K/V batch dimensions differ",
    )
    _require(
        kv_heads == value_heads,
        "ORACLE_INPUT_SHAPE",
        "key and value head counts differ",
    )
    _require(
        key_tokens == value_tokens,
        "ORACLE_INPUT_SHAPE",
        "key and value token counts differ",
    )
    _require(
        head_dim == key_dim == value_dim,
        "ORACLE_INPUT_SHAPE",
        "Q/K/V head dimensions differ",
    )
    _require(
        query_heads % kv_heads == 0,
        "ORACLE_INPUT_SHAPE",
        "query head count must be divisible by KV head count",
    )
    _require(
        query_tokens <= key_tokens,
        "ORACLE_POSITION_CONTRACT",
        "cached-suffix attention requires query_tokens <= key_tokens",
    )
    _require(
        bool(torch.isfinite(query).all().item())
        and bool(torch.isfinite(key).all().item())
        and bool(torch.isfinite(value).all().item()),
        "ORACLE_INPUT_FINITE",
        "query, key, and value must be finite",
    )
    return batch, query_heads, query_tokens, head_dim, kv_heads, key_tokens


def fp32_dense_attention_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    visibility_mask: torch.Tensor | None = None,
    scaling: float | None = None,
) -> DenseAttentionResult:
    """Compute the canonical cached-suffix causal reference in FP32.

    ``visibility_mask`` is an optional boolean restriction applied in addition
    to causality; ``True`` means visible.  It must be broadcastable to the
    attention score shape ``[batch, query_heads, query_tokens, key_tokens]``.
    Additive masks are rejected so that mask polarity is never ambiguous.
    """

    batch, query_heads, query_tokens, head_dim, kv_heads, key_tokens = _validate_qkv(
        query, key, value
    )
    key_pos = _normalize_positions(
        key_positions,
        batch=batch,
        length=key_tokens,
        device=query.device,
        default_start=0,
        label="key",
    )
    query_pos = _normalize_positions(
        query_positions,
        batch=batch,
        length=query_tokens,
        device=query.device,
        default_start=key_tokens - query_tokens,
        label="query",
    )
    _require(
        bool(torch.equal(query_pos, key_pos[:, -query_tokens:])),
        "ORACLE_POSITION_CONTRACT",
        "query positions must equal the suffix of key positions",
    )

    scale = 1.0 / math.sqrt(head_dim) if scaling is None else float(scaling)
    _require(
        math.isfinite(scale) and scale > 0.0,
        "ORACLE_SCALING",
        "attention scaling must be finite and positive",
    )

    groups = query_heads // kv_heads
    with ieee_fp32_reference_context(device_type=query.device.type) as precision_audit:
        key_f = (
            key.float()
            .unsqueeze(2)
            .expand(batch, kv_heads, groups, key_tokens, head_dim)
            .reshape(batch, query_heads, key_tokens, head_dim)
        )
        value_f = (
            value.float()
            .unsqueeze(2)
            .expand(batch, kv_heads, groups, key_tokens, head_dim)
            .reshape(batch, query_heads, key_tokens, head_dim)
        )
        scores = torch.matmul(query.float(), key_f.transpose(-2, -1)) * scale
        keep = key_pos[:, None, None, :] <= query_pos[:, None, :, None]
        if visibility_mask is not None:
            _require(
                isinstance(visibility_mask, torch.Tensor),
                "ORACLE_MASK_TYPE",
                "visibility_mask must be a tensor or None",
            )
            _require(
                visibility_mask.dtype == torch.bool,
                "ORACLE_MASK_DTYPE",
                "visibility_mask must be boolean with True meaning visible",
            )
            try:
                visible = torch.broadcast_to(
                    visibility_mask.to(query.device), scores.shape
                )
            except RuntimeError as exc:
                raise OracleGateError(
                    "ORACLE_MASK_SHAPE",
                    "visibility_mask is not broadcastable to attention scores",
                ) from exc
            keep = keep & visible
        _require(
            bool(keep.any(dim=-1).all().item()),
            "ORACLE_MASK_EMPTY_ROW",
            "every query head must retain at least one visible key",
        )
        scores = scores.masked_fill(~keep, -torch.inf)
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        _require(
            bool(torch.isfinite(weights).all().item()),
            "ORACLE_REFERENCE_FINITE",
            "dense FP32 attention probabilities are non-finite",
        )
        output = torch.matmul(weights, value_f).transpose(1, 2).contiguous()
        _require(
            output.dtype == torch.float32
            and bool(torch.isfinite(output).all().item()),
            "ORACLE_REFERENCE_FINITE",
            "dense FP32 attention output is not finite FP32",
        )
    return DenseAttentionResult(
        output=output,
        position_contract={
            "mode": "cached_suffix_causal",
            "query_is_key_suffix": True,
            "positions_contiguous_unit_stride": True,
            "query_tokens": query_tokens,
            "key_tokens": key_tokens,
            "query_position_min": int(query_pos.min().item()),
            "query_position_max": int(query_pos.max().item()),
            "key_position_min": int(key_pos.min().item()),
            "key_position_max": int(key_pos.max().item()),
            "additional_visibility_mask": visibility_mask is not None,
        },
        scaling=scale,
        grouped_query_factor=groups,
        precision_audit=dict(precision_audit),
    )


def tensor_error_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> TensorErrorMetrics:
    """Return FP32 error metrics after strict candidate validation."""

    _require(
        isinstance(reference, torch.Tensor) and isinstance(candidate, torch.Tensor),
        "ORACLE_CANDIDATE_TYPE",
        "reference and candidate must be tensors",
    )
    _require(
        tuple(reference.shape) == tuple(candidate.shape) and reference.numel() > 0,
        "ORACLE_CANDIDATE_SHAPE",
        "reference and candidate shapes must match and be non-empty",
    )
    _require(
        _is_supported_float(reference) and _is_supported_float(candidate),
        "ORACLE_CANDIDATE_DTYPE",
        "reference and candidate must have real floating dtypes",
    )
    reference_f = reference.float()
    candidate_f = candidate.float()
    finite = bool(
        torch.isfinite(reference_f).all().item()
        and torch.isfinite(candidate_f).all().item()
    )
    _require(
        finite,
        "ORACLE_CANDIDATE_FINITE",
        "reference and candidate must be finite",
    )
    error = candidate_f - reference_f
    _require(
        bool(torch.isfinite(error).all().item()),
        "ORACLE_METRICS_FINITE",
        "FP32 candidate error overflowed",
    )
    numerator = float(torch.linalg.vector_norm(error).item())
    denominator = float(torch.linalg.vector_norm(reference_f).item())
    _require(
        math.isfinite(numerator) and math.isfinite(denominator),
        "ORACLE_METRICS_FINITE",
        "FP32 norm reduction overflowed",
    )
    # Match the earlier fair-v2 diagnostic while doing the final division in
    # Python's wider scalar range so the structured result remains valid JSON.
    relative_l2 = numerator / max(denominator, 1e-30)
    return TensorErrorMetrics(
        bitwise_exact=bool(torch.equal(reference, candidate)),
        finite=finite,
        max_abs=float(error.abs().max().item()),
        mean_abs=float(error.abs().mean().item()),
        relative_l2=relative_l2,
    )


def logit_metrics(
    reference_logits: torch.Tensor, candidate_logits: torch.Tensor
) -> LogitMetrics:
    """Return top-1 agreement and forward KL(reference || candidate)."""

    _require(
        isinstance(reference_logits, torch.Tensor)
        and isinstance(candidate_logits, torch.Tensor),
        "ORACLE_LOGITS_TYPE",
        "reference and candidate logits must be tensors",
    )
    _require(
        tuple(reference_logits.shape) == tuple(candidate_logits.shape)
        and reference_logits.ndim >= 1
        and reference_logits.shape[-1] > 0
        and reference_logits.numel() > 0,
        "ORACLE_LOGITS_SHAPE",
        "logit shapes must match, be non-empty, and include a vocabulary axis",
    )
    _require(
        _is_supported_float(reference_logits)
        and _is_supported_float(candidate_logits),
        "ORACLE_LOGITS_DTYPE",
        "logits must have real floating dtypes",
    )
    reference_f = reference_logits.float()
    candidate_f = candidate_logits.float()
    _require(
        bool(torch.isfinite(reference_f).all().item())
        and bool(torch.isfinite(candidate_f).all().item()),
        "ORACLE_LOGITS_FINITE",
        "logits must be finite",
    )
    reference_top1 = reference_f.argmax(dim=-1)
    candidate_top1 = candidate_f.argmax(dim=-1)
    matches = reference_top1 == candidate_top1
    reference_log_probs = torch.log_softmax(reference_f, dim=-1)
    candidate_log_probs = torch.log_softmax(candidate_f, dim=-1)
    forward_kl = (
        reference_log_probs.exp() * (reference_log_probs - candidate_log_probs)
    ).sum(dim=-1)
    _require(
        bool(torch.isfinite(forward_kl).all().item()),
        "ORACLE_LOGITS_METRICS_FINITE",
        "forward KL is non-finite",
    )
    # Roundoff can produce a tiny negative value for theoretically non-negative KL.
    forward_kl = forward_kl.clamp_min(0.0)
    return LogitMetrics(
        top1_match=bool(matches.all().item()),
        top1_agreement=float(matches.float().mean().item()),
        mean_forward_kl=float(forward_kl.mean().item()),
        max_forward_kl=float(forward_kl.max().item()),
    )


class OraclePreregistration:
    """Own thresholds and permanently lock them at the first evaluation attempt."""

    def __init__(self, thresholds: OracleThresholds | None = None) -> None:
        self._thresholds = thresholds or OracleThresholds()
        self._evaluation_count = 0
        self._locked = False

    @property
    def thresholds(self) -> OracleThresholds:
        return self._thresholds

    @property
    def locked(self) -> bool:
        return self._locked

    @property
    def evaluation_count(self) -> int:
        return self._evaluation_count

    @property
    def sha256(self) -> str:
        return _canonical_json_sha256(
            {
                "schema_version": ORACLE_SCHEMA_VERSION,
                "thresholds": self._thresholds.to_dict(),
            }
        )

    def configure(self, **changes: Any) -> None:
        """Change thresholds only before any result, valid or invalid, is observed."""

        if self._locked:
            raise ThresholdLockedError(
                "oracle thresholds were locked by an evaluation attempt"
            )
        self._thresholds = replace(self._thresholds, **changes)

    def _begin_evaluation(self) -> tuple[int, str, OracleThresholds]:
        self._locked = True
        self._evaluation_count += 1
        return self._evaluation_count, self.sha256, self._thresholds

    def evaluate_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        candidate_output: torch.Tensor,
        *,
        query_positions: torch.Tensor | None = None,
        key_positions: torch.Tensor | None = None,
        visibility_mask: torch.Tensor | None = None,
        scaling: float | None = None,
        reference_logits: torch.Tensor | None = None,
        candidate_logits: torch.Tensor | None = None,
    ) -> OracleOutcome:
        """Compute, compare, and return a structured, threshold-bound outcome."""

        evaluation_index, preregistration_sha256, thresholds = self._begin_evaluation()
        gates: list[GateResult] = []
        try:
            _require(
                (reference_logits is None) == (candidate_logits is None),
                "ORACLE_LOGITS_PAIR",
                "reference_logits and candidate_logits must be supplied together",
            )
            reference = fp32_dense_attention_reference(
                query,
                key,
                value,
                query_positions=query_positions,
                key_positions=key_positions,
                visibility_mask=visibility_mask,
                scaling=scaling,
            )
            gates.extend(
                (
                    GateResult("ORACLE_INPUT_SHAPE", True, True, "Q/K/V shapes valid"),
                    GateResult("ORACLE_INPUT_DTYPE", True, True, "Q/K/V dtypes valid"),
                    GateResult("ORACLE_INPUT_FINITE", True, True, "Q/K/V finite"),
                    GateResult(
                        "ORACLE_POSITION_CONTRACT",
                        True,
                        True,
                        "query positions equal the contiguous key suffix",
                    ),
                )
            )
            attention = tensor_error_metrics(reference.output, candidate_output)
            gates.extend(
                (
                    GateResult(
                        "ORACLE_CANDIDATE_SHAPE",
                        True,
                        True,
                        "candidate shape matches dense reference",
                    ),
                    GateResult(
                        "ORACLE_CANDIDATE_DTYPE",
                        True,
                        True,
                        "candidate dtype is real floating",
                    ),
                    GateResult(
                        "ORACLE_CANDIDATE_FINITE",
                        attention.finite,
                        True,
                        "candidate and reference finite",
                    ),
                    GateResult(
                        "ORACLE_RELATIVE_L2",
                        attention.relative_l2 <= thresholds.max_relative_l2,
                        True,
                        (
                            f"relative_l2={attention.relative_l2:.9g} <= "
                            f"{thresholds.max_relative_l2:.9g}"
                        ),
                    ),
                )
            )
            if thresholds.max_absolute_error is not None:
                gates.append(
                    GateResult(
                        "ORACLE_MAX_ABS",
                        attention.max_abs <= thresholds.max_absolute_error,
                        True,
                        (
                            f"max_abs={attention.max_abs:.9g} <= "
                            f"{thresholds.max_absolute_error:.9g}"
                        ),
                    )
                )
            else:
                gates.append(
                    GateResult(
                        "ORACLE_MAX_ABS",
                        True,
                        False,
                        f"diagnostic max_abs={attention.max_abs:.9g}",
                    )
                )

            logits = None
            if reference_logits is not None and candidate_logits is not None:
                logits = logit_metrics(reference_logits, candidate_logits)
                gates.append(
                    GateResult(
                        "ORACLE_LOGITS_TOP1",
                        logits.top1_match,
                        thresholds.require_logits_top1_match,
                        (
                            f"top1_match={logits.top1_match}; "
                            f"agreement={logits.top1_agreement:.9g}"
                        ),
                    )
                )
                if thresholds.max_mean_forward_kl is not None:
                    gates.append(
                        GateResult(
                            "ORACLE_LOGITS_FORWARD_KL",
                            logits.mean_forward_kl
                            <= thresholds.max_mean_forward_kl,
                            True,
                            (
                                f"mean_forward_kl={logits.mean_forward_kl:.9g} <= "
                                f"{thresholds.max_mean_forward_kl:.9g}"
                            ),
                        )
                    )
                else:
                    gates.append(
                        GateResult(
                            "ORACLE_LOGITS_FORWARD_KL",
                            True,
                            False,
                            (
                                "diagnostic mean_forward_kl="
                                f"{logits.mean_forward_kl:.9g}"
                            ),
                        )
                    )

            passed = all(gate.passed for gate in gates if gate.is_gate)
            return OracleOutcome(
                status="completed",
                passed=passed,
                preregistration_sha256=preregistration_sha256,
                evaluation_index=evaluation_index,
                thresholds=thresholds.to_dict(),
                gates=tuple(gates),
                attention_metrics=attention,
                logits_metrics=logits,
                reference={
                    "output_layout": "BQHD",
                    "output_dtype": str(reference.output.dtype),
                    "output_shape": list(reference.output.shape),
                    "scaling": reference.scaling,
                    "grouped_query_factor": reference.grouped_query_factor,
                    "position_contract": dict(reference.position_contract),
                    "precision_audit": dict(reference.precision_audit),
                },
                failure=None,
            )
        except OracleGateError as exc:
            gates.append(GateResult(exc.gate_id, False, True, exc.detail))
            return OracleOutcome(
                status="invalid",
                passed=False,
                preregistration_sha256=preregistration_sha256,
                evaluation_index=evaluation_index,
                thresholds=thresholds.to_dict(),
                gates=tuple(gates),
                attention_metrics=None,
                logits_metrics=None,
                reference=None,
                failure={
                    "gate_id": exc.gate_id,
                    "error_type": type(exc).__name__,
                    "detail": exc.detail,
                },
            )


__all__ = [
    "DEFAULT_MAX_RELATIVE_L2",
    "DenseAttentionResult",
    "GateResult",
    "LogitMetrics",
    "ORACLE_SCHEMA_VERSION",
    "OracleGateError",
    "OracleOutcome",
    "OraclePreregistration",
    "OracleThresholds",
    "TensorErrorMetrics",
    "ThresholdLockedError",
    "fp32_dense_attention_reference",
    "ieee_fp32_reference_context",
    "logit_metrics",
    "tensor_error_metrics",
]
