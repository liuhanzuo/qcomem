from __future__ import annotations

"""Fail-closed Qwen3.5 integration for the vLLM Triton Q16 paged kernel."""

import copy
import re
import uuid
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch

from qcomem_qwen35_native_cache import install_native_functional_linear_cache
from qcomem_vllm_paged_kernel import (
    KERNEL_MODE,
    Q16KernelPagedDocumentLayer,
    Q16KernelPagedLayer,
    Q16KernelPagedTensorView,
    QComemPagedKernelError,
    vllm_triton_q16_paged_attention_forward,
)


class Qwen35VllmPagedIntegrationError(RuntimeError):
    pass


POST_ROPE_POSITION_IDS_CONTRACT = "qwen3.5-text-tail-post-rope-v1"


def validate_qwen35_post_rope_position_ids(
    position_ids: Any,
    *,
    query: torch.Tensor,
    total_length: int,
    strict_tail_values: bool,
) -> dict[str, Any]:
    """Consume Qwen3.5's text positions after mask/RoPE have used them.

    Transformers 5.14's Qwen3.5 text model first uses ``text_position_ids``
    to construct the causal mask and uses the remaining position planes to
    construct RoPE embeddings.  The attention module applies those embeddings
    to Q/K and then forwards ``text_position_ids`` to the registered attention
    interface.  Stock eager attention accepts that keyword through ``**kwargs``
    but performs no further position-dependent math.

    The strict calibration path checks the actual contiguous tail values.  The
    timed production path checks only tensor metadata because the values have
    already been consumed upstream; this deliberately avoids a GPU-to-host
    synchronization in every full-attention layer.
    """

    if not isinstance(position_ids, torch.Tensor):
        raise Qwen35VllmPagedIntegrationError(
            "Qwen3.5 fused attention requires tensor position_ids"
        )
    if query.ndim != 4:
        raise Qwen35VllmPagedIntegrationError("query must be rank four")
    batch = int(query.shape[0])
    query_length = int(query.shape[-2])
    if position_ids.dtype != torch.long:
        raise Qwen35VllmPagedIntegrationError(
            f"position_ids must be torch.long, found {position_ids.dtype}"
        )
    if tuple(position_ids.shape) != (batch, query_length):
        raise Qwen35VllmPagedIntegrationError(
            "position_ids must have shape [query_batch, query_tokens]"
        )
    if position_ids.device != query.device:
        raise Qwen35VllmPagedIntegrationError(
            "position_ids must be on the query device"
        )
    if total_length < query_length:
        raise Qwen35VllmPagedIntegrationError(
            "position_ids total length is shorter than the query"
        )
    expected_start = int(total_length) - query_length
    if strict_tail_values:
        expected = torch.arange(
            expected_start,
            int(total_length),
            dtype=torch.long,
            device=query.device,
        ).view(1, query_length).expand(batch, query_length)
        if not torch.equal(position_ids, expected):
            raise Qwen35VllmPagedIntegrationError(
                "position_ids are not the canonical contiguous causal tail"
            )
    return {
        "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
        "position_ids_validated": True,
        "position_ids_semantically_consumed_upstream": True,
        "position_ids_shape": (batch, query_length),
        "position_ids_dtype": str(position_ids.dtype),
        "position_ids_expected_tail_start": expected_start,
        "position_ids_expected_tail_end_exclusive": int(total_length),
        "position_ids_strict_tail_values_checked": bool(strict_tail_values),
        "position_ids_validation_host_syncs": 1 if strict_tail_values else 0,
    }


@dataclass(frozen=True)
class Qwen35VllmPagedConversion:
    layer_indices: tuple[int, ...]
    document_length: int
    page_size: int
    max_append_tokens: int
    max_request_forks: int
    layer_arena_ids: dict[int, int]
    dense_document_nbytes: int
    allocated_block_pool_nbytes: int
    document_payload_nbytes: int
    quantization: str = "Q16"


def _dense_layer(layer: Any, index: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = getattr(layer, "keys", None)
    value = getattr(layer, "values", None)
    if bool(getattr(layer, "is_sliding", False)):
        raise Qwen35VllmPagedIntegrationError(f"layer {index} is sliding")
    if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
        raise Qwen35VllmPagedIntegrationError(
            f"layer {index} does not contain dense K/V tensors"
        )
    if key.ndim != 4 or key.shape != value.shape or key.shape[-2] < 1:
        raise Qwen35VllmPagedIntegrationError(f"layer {index} K/V shape is invalid")
    return key, value


def convert_all_qwen35_full_layers_to_vllm_q16(
    cache: Any,
    plan: Any,
    *,
    page_size: int,
    max_append_tokens: int,
    max_request_forks: int,
    bits: int = 16,
) -> Qwen35VllmPagedConversion:
    """Build all arenas before installing any replacement (atomic fail-close)."""

    if bits != 16:
        raise Qwen35VllmPagedIntegrationError(
            "vLLM kernel integration currently supports Q16 only; Q8/Q4 formats differ"
        )
    indices = tuple(plan.full_attention_layer_indices)
    if len(indices) != 10:
        raise Qwen35VllmPagedIntegrationError(
            f"formal Qwen3.5 kernel run requires 10 full layers, found {len(indices)}"
        )
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, (list, tuple)):
        raise Qwen35VllmPagedIntegrationError("cache.layers must be a sequence")
    inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    lengths = set()
    dense_bytes = 0
    for index in indices:
        key, value = _dense_layer(layers[index], index)
        inputs[index] = (key, value)
        lengths.add(int(key.shape[-2]))
        dense_bytes += key.numel() * key.element_size() + value.numel() * value.element_size()
    if len(lengths) != 1:
        raise Qwen35VllmPagedIntegrationError("full layers disagree on document length")
    replacements: dict[int, Q16KernelPagedDocumentLayer] = {}
    try:
        for index, (key, value) in inputs.items():
            replacements[index] = Q16KernelPagedDocumentLayer.from_dense_document(
                key,
                value,
                page_size=page_size,
                max_append_tokens=max_append_tokens,
                max_request_forks=max_request_forks,
            )
    except (QComemPagedKernelError, RuntimeError) as error:
        raise Qwen35VllmPagedIntegrationError(
            f"Q16 block-pool construction failed before install: {error}"
        ) from error
    for index, replacement in replacements.items():
        layers[index] = replacement
    if any(layers[index] is not replacements[index] for index in indices):
        raise Qwen35VllmPagedIntegrationError("full-layer replacement is incomplete")
    allocated = sum(layer.stored_nbytes for layer in replacements.values())
    payload = sum(layer.arena.audit.document_payload_nbytes for layer in replacements.values())
    return Qwen35VllmPagedConversion(
        layer_indices=indices,
        document_length=lengths.pop(),
        page_size=page_size,
        max_append_tokens=max_append_tokens,
        max_request_forks=max_request_forks,
        layer_arena_ids={index: id(layer.arena) for index, layer in replacements.items()},
        dense_document_nbytes=dense_bytes,
        allocated_block_pool_nbytes=allocated,
        document_payload_nbytes=payload,
    )


def _seed_tensor_memo(value: Any, memo: dict[int, Any], seen: set[int]) -> None:
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, torch.Tensor):
        memo[identity] = value
    elif isinstance(value, dict):
        for item in value.values():
            _seed_tensor_memo(item, memo, seen)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _seed_tensor_memo(item, memo, seen)
    elif hasattr(value, "__dict__"):
        for item in vars(value).values():
            _seed_tensor_memo(item, memo, seen)


def fork_qwen35_vllm_q16_request(cache: Any, plan: Any) -> tuple[Any, dict[str, Any]]:
    """Fork metadata/GDN state while sharing arenas and document tensors."""

    memo: dict[int, Any] = {}
    for index in plan.full_attention_layer_indices:
        layer = cache.layers[index]
        if not isinstance(layer, Q16KernelPagedDocumentLayer):
            raise Qwen35VllmPagedIntegrationError(
                f"persistent layer {index} is not a Q16 document facade"
            )
        memo[id(layer)] = layer  # avoid deepcopying the arena's thread lock
    _seed_tensor_memo(cache, memo, set())
    request = copy.deepcopy(cache, memo)
    if request is cache:
        raise Qwen35VllmPagedIntegrationError("cache fork returned persistent object")
    shared_block_pool = 0
    for index in plan.full_attention_layer_indices:
        source = cache.layers[index]
        target = source.fork()
        request.layers[index] = target
        if target.sequence.arena is not source.arena:
            raise Qwen35VllmPagedIntegrationError(f"layer {index} did not share arena")
        shared_block_pool += source.stored_nbytes
    install = install_native_functional_linear_cache(request, plan.gdn)
    if tuple(install.linear_layer_indices) != tuple(plan.linear_layer_indices):
        raise Qwen35VllmPagedIntegrationError("request missed a GDN functional seam")
    if tuple(install.full_attention_layer_indices) != tuple(
        plan.full_attention_layer_indices
    ):
        raise Qwen35VllmPagedIntegrationError(
            "GDN cache contract and Q16 full-attention plan differ"
        )
    return request, {
        "request_policy": "vllm-q16-shared-document-reuse",
        "single_request_only": True,
        "same_unified_attention_kernel": True,
        "shared_q16_block_pool_nbytes": shared_block_pool,
        "query_private_block_table_only_at_fork": True,
        "full_document_staging_copy_nbytes": 0,
        "allocated_request_pool_nbytes": 0,
        "source_document_storage_shared": True,
        "linear_functional_rebind": True,
        "full_attention_layer_count": len(plan.full_attention_layer_indices),
    }


class Qwen35VllmPagedHitLedger:
    def __init__(
        self,
        plan: Any,
        conversion: Qwen35VllmPagedConversion,
        *,
        expected_calls_per_layer: int = 1,
        mask_contract: str = "strict-canonical-audit",
    ) -> None:
        if tuple(plan.full_attention_layer_indices) != conversion.layer_indices:
            raise Qwen35VllmPagedIntegrationError("plan/conversion layers differ")
        if expected_calls_per_layer < 1:
            raise Qwen35VllmPagedIntegrationError("expected calls must be positive")
        self.indices = conversion.layer_indices
        self.arena_ids = dict(conversion.layer_arena_ids)
        self.expected_calls_per_layer = expected_calls_per_layer
        if mask_contract not in (
            "strict-canonical-audit",
            "prevalidated-no-padding-tail-causal",
        ):
            raise Qwen35VllmPagedIntegrationError("unknown production mask contract")
        self.mask_contract = mask_contract
        self.counts: Counter[int] = Counter()
        self.calls: list[dict[str, Any]] = []

    def attention_forward(
        self,
        module: torch.nn.Module,
        query: torch.Tensor,
        key: Q16KernelPagedTensorView,
        value: Q16KernelPagedTensorView,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, None]:
        index = getattr(module, "layer_idx", None)
        if index not in self.arena_ids:
            raise Qwen35VllmPagedIntegrationError(f"unexpected attention layer {index}")
        if not isinstance(key, Q16KernelPagedTensorView) or not isinstance(
            value, Q16KernelPagedTensorView
        ):
            raise Qwen35VllmPagedIntegrationError(f"layer {index} reached dense fallback")
        if id(key.sequence.arena) != self.arena_ids[index]:
            raise Qwen35VllmPagedIntegrationError(f"layer {index} used wrong arena")
        if self.counts[index] >= self.expected_calls_per_layer:
            raise Qwen35VllmPagedIntegrationError(f"layer {index} exceeded call budget")
        if self.mask_contract == "prevalidated-no-padding-tail-causal":
            if attention_mask is not None:
                raise Qwen35VllmPagedIntegrationError(
                    "production fused backend expected no materialized attention mask"
                )
            if key.sequence.strict_mask_check:
                raise Qwen35VllmPagedIntegrationError(
                    "production request retained per-layer strict mask validation"
                )
        position_ids = kwargs.pop("position_ids", None)
        position_audit = validate_qwen35_post_rope_position_ids(
            position_ids,
            query=query,
            total_length=int(key.shape[-2]),
            strict_tail_values=self.mask_contract == "strict-canonical-audit",
        )
        audit: dict[str, Any] = {}
        result = vllm_triton_q16_paged_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            audit=audit,
            **kwargs,
        )
        self.counts[index] += 1
        self.calls.append({"layer_idx": index, **audit, **position_audit})
        self.calls[-1]["mask_contract"] = self.mask_contract
        if self.mask_contract == "prevalidated-no-padding-tail-causal":
            self.calls[-1]["materialized_attention_mask_nbytes"] = 0
            self.calls[-1]["mask_validation_host_syncs"] = 0
        else:
            mask = attention_mask
            if isinstance(mask, dict):
                mask = mask.get("full_attention")
            self.calls[-1]["materialized_attention_mask_nbytes"] = (
                0
                if mask is None
                else mask.numel() * mask.element_size()
            )
            self.calls[-1]["mask_validation_host_syncs"] = 1 if mask is not None else 0
        return result

    def verify_complete(self) -> dict[str, Any]:
        expected = {index: self.expected_calls_per_layer for index in self.indices}
        actual = {index: self.counts[index] for index in self.indices}
        if actual != expected or set(self.counts) != set(self.indices):
            raise Qwen35VllmPagedIntegrationError(
                f"fused intercept incomplete: expected={expected}, actual={actual}"
            )
        return {
            "verified": True,
            "kernel_mode": KERNEL_MODE,
            "expected_layer_indices": self.indices,
            "counts": actual,
            "total_calls": sum(actual.values()),
            "dense_fallback_calls": 0,
            "full_kv_concatenations": 0,
            "mask_contract": self.mask_contract,
            "materialized_attention_mask_nbytes": sum(
                int(call["materialized_attention_mask_nbytes"])
                for call in self.calls
            ),
            "mask_validation_host_syncs": sum(
                int(call["mask_validation_host_syncs"]) for call in self.calls
            ),
            "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
            "position_ids_validation_host_syncs": sum(
                int(call["position_ids_validation_host_syncs"])
                for call in self.calls
            ),
            "calls": tuple(self.calls),
        }


@dataclass(frozen=True)
class RegisteredQwen35VllmPagedBackend:
    name: str
    transformers_version: str
    ledger: Qwen35VllmPagedHitLedger


def register_qwen35_vllm_q16_backend(
    ledger: Qwen35VllmPagedHitLedger,
    *,
    name: str | None = None,
) -> RegisteredQwen35VllmPagedBackend:
    try:
        import transformers
        from transformers.masking_utils import AttentionMaskInterface, eager_mask
        from transformers.modeling_utils import AttentionInterface
    except ImportError as error:
        raise Qwen35VllmPagedIntegrationError("Transformers registry API unavailable") from error
    version = str(getattr(transformers, "__version__", ""))
    if re.match(r"^5\.14(?:\.|$)", version) is None:
        raise Qwen35VllmPagedIntegrationError(
            f"fused integration is pinned to Transformers 5.14.x, found {version}"
        )
    if name is None:
        name = f"qcomem_vllm_q16_{uuid.uuid4().hex}"
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None:
        raise Qwen35VllmPagedIntegrationError("invalid backend name")
    AttentionInterface.register(name, ledger.attention_forward)
    if ledger.mask_contract == "prevalidated-no-padding-tail-causal":
        def no_materialized_mask(*args: Any, attention_mask=None, **kwargs: Any):
            del args, kwargs
            if attention_mask is not None:
                raise Qwen35VllmPagedIntegrationError(
                    "production Q16 benchmark forbids padding/custom attention masks"
                )
            return None

        AttentionMaskInterface.register(name, no_materialized_mask)
    else:
        AttentionMaskInterface.register(name, eager_mask)
    return RegisteredQwen35VllmPagedBackend(name, version, ledger)


def full_vocab_forward_kl(reference: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise Qwen35VllmPagedIntegrationError("logits must be matching [batch,vocab]")
    ref_logp = torch.log_softmax(reference.float(), dim=-1)
    cand_logp = torch.log_softmax(candidate.float(), dim=-1)
    return torch.sum(ref_logp.exp() * (ref_logp - cand_logp), dim=-1)


__all__ = [
    "KERNEL_MODE",
    "POST_ROPE_POSITION_IDS_CONTRACT",
    "Qwen35VllmPagedConversion",
    "Qwen35VllmPagedHitLedger",
    "Qwen35VllmPagedIntegrationError",
    "RegisteredQwen35VllmPagedBackend",
    "convert_all_qwen35_full_layers_to_vllm_q16",
    "fork_qwen35_vllm_q16_request",
    "full_vocab_forward_kl",
    "register_qwen35_vllm_q16_backend",
    "validate_qwen35_post_rope_position_ids",
]
