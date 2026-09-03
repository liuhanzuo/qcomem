from __future__ import annotations

"""Formal v2 protocol for a fair Q16 paged-cache infrastructure comparison.

The primary comparison is deliberately *not* HF eager versus vLLM.  It is a
same-kernel comparison between a fresh, fully materialized request block pool
and a shared-document/private-tail request, both dispatched through the exact
same vLLM 0.26 ``unified_attention`` callable.  HF eager and an FP32 dense
oracle are retained as a separate backend-compatibility diagnostic.

The first GPU phase consumes only frozen PG-19 train windows.  A SHA-addressed
authorization artifact is required before this file will hash or open the
LongBench validation artifact.  Validation is restricted to source 6--9 from
QASPER and 2WikiMQA; test-v2 and source 68--99 are outside this protocol.
"""

import argparse
import copy
import gc
import hashlib
import inspect
import json
import math
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from qcomem_joint_policy import (
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
    sha256_file,
)
from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
from qcomem_qwen35_native_cache import install_native_functional_linear_cache
from qcomem_qwen35_vllm_paged_integration import (
    POST_ROPE_POSITION_IDS_CONTRACT,
    convert_all_qwen35_full_layers_to_vllm_q16,
    fork_qwen35_vllm_q16_request,
    full_vocab_forward_kl,
    validate_qwen35_post_rope_position_ids,
)
from qcomem_torch import clone_cache
from qcomem_vllm_paged_fair_control import (
    FAIR_CONFIGS,
    FAIR_PROTOCOL,
    FRESH_CONTROL,
    PRODUCTION_MASK_CONTRACT,
    SHARED_REUSE,
    QComemFairControlError,
    Qwen35FairHitLedger,
    build_same_kernel_q16_sequence_pair,
    full_attention_storage_breakdown,
    linear_gdn_shared_base_contract,
    materialize_qwen35_fresh_full_copy_request,
    register_qwen35_fair_backend,
    snapshot_linear_gdn_state,
    storage_residency,
    verify_linear_gdn_state_parity,
)
from qcomem_vllm_paged_kernel import (
    KERNEL_MODE,
    QWEN35_AUDITED_GEOMETRY,
    _resolve_vllm_unified_attention,
    audit_frozen_kernel_environment,
    vllm_triton_q16_paged_attention_forward,
)
from run_deployment_bench import longbench_workloads
from run_downstream import atomic_json


TEST_V2_SHA256 = "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f"
FULL_LAYERS = tuple(range(3, 40, 4))
EXPECTED_VALIDATION_PAIRS = {
    (dataset, source)
    for dataset in ("qasper", "2wikimqa")
    for source in range(6, 10)
}
FROZEN_IDENTITY_FIELDS = (
    "code_ledger_sha256",
    "model_manifest_sha256",
    "model_artifact_ledger_sha256",
    "model_weight_ledger_sha256",
    "pg19_data_sha256",
    "pg19_manifest_sha256",
    "pg19_windows_sha256",
    "validation_expected_sha256_recorded_but_not_hashed",
    "protocol_manifest_sha256",
    "protocol_config_sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _required_nonnegative_int(value: Any, label: str) -> int:
    """Return a required non-bool, non-negative integer or fail closed."""

    _require(
        type(value) is int and value >= 0,
        f"{label} must be a present non-bool integer >= 0",
    )
    return value


def _required_positive_int(value: Any, label: str) -> int:
    result = _required_nonnegative_int(value, label)
    _require(result > 0, f"{label} must be > 0")
    return result


def _validate_kernel_identity_schema(identity: Any, label: str) -> dict[str, Any]:
    """Validate the machine-auditable identity before comparing identities."""

    _require(isinstance(identity, dict), f"{label} kernel identity must be a dict")
    _required_positive_int(identity.get("callable_id"), f"{label} callable_id")
    for field in ("module", "qualname", "signature"):
        value = identity.get(field)
        _require(
            isinstance(value, str) and bool(value.strip()),
            f"{label} kernel identity {field} must be a nonempty string",
        )
    return identity


def _validate_sha256_hex(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be a lowercase 64-hex SHA256",
    )
    return value


def _load_frozen_json(path: Path, expected_sha256: str, label: str) -> Any:
    """Hash and decode the exact same bytes, closing hash/open TOCTOU gaps."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"{label} cannot be read") from error
    observed = hashlib.sha256(payload).hexdigest()
    _require(observed == expected_sha256, f"{label} SHA256 mismatch")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON") from error


def _safe_ratio(numerator: float | int, denominator: float | int) -> float | None:
    """Return no ratio when a zero baseline would make it misleading."""

    if float(denominator) <= 0.0:
        return None
    return float(numerator) / float(denominator)


def _sync() -> None:
    torch.cuda.synchronize()


def _allocator_snapshot(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "current_allocated_bytes": torch.cuda.memory_allocated(),
        "current_reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _fresh_allocator_cleanup(label: str) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    _sync()
    return _allocator_snapshot(label)


def _unregister_backend(name: str) -> None:
    """Remove bound-method registries so fresh trials cannot retain caches."""

    from transformers.masking_utils import AttentionMaskInterface
    from transformers.modeling_utils import AttentionInterface

    attention = AttentionInterface._global_mapping.pop(name, None)
    mask = AttentionMaskInterface._global_mapping.pop(name, None)
    if attention is None or mask is None:
        raise RuntimeError(f"backend registry cleanup failed for {name}")


def _resolve_backbone(model: Any) -> Any:
    if hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model.model, "layers"):
        return model.model
    raise RuntimeError("cannot resolve Qwen3.5 text backbone")


def _tokens(value: torch.Tensor, limit: int) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or int(value.shape[0]) != 1:
        raise RuntimeError("formal v2 supports unpadded batch-1 token tensors only")
    return value[:, :limit]


def _model_manifest_sha(model_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    names = ("config.json", "generation_config.json", "model.safetensors.index.json")
    manifest = hashlib.sha256()
    rows = []
    for name in names:
        path = model_dir / name
        if not path.is_file():
            raise RuntimeError(f"frozen model manifest file is missing: {path}")
        digest = sha256_file(path)
        size = path.stat().st_size
        rows.append({"name": name, "sha256": digest, "bytes": size})
        manifest.update(f"{name}\0{digest}\0{size}\n".encode())
    return manifest.hexdigest(), rows


def _static_frozen_identity(static: dict[str, Any]) -> dict[str, str]:
    identity = {field: str(static.get(field, "")) for field in FROZEN_IDENTITY_FIELDS}
    for field, digest in identity.items():
        _require(
            len(digest) == 64 and all(ch in "0123456789abcdef" for ch in digest),
            f"static frozen identity is missing {field}",
        )
    return identity


def _protocol_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "bits": int(args.bits),
        "page_size": int(args.page_size),
        "world_size": int(args.world_size),
        "pg19_books": int(args.pg19_books),
        "pg19_document_tokens": int(args.pg19_document_tokens),
        "pg19_query_tokens": int(args.pg19_query_tokens),
        "pg19_window_stride": int(args.pg19_window_stride),
        "pg19_candidate_windows": int(args.pg19_candidate_windows),
        "pg19_seed": int(args.pg19_seed),
        "max_input_tokens": int(args.max_input_tokens),
        "max_query_tokens": int(args.max_query_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "source_index_start": int(args.source_index_start),
        "source_index_end": int(args.source_index_end),
        "limit_per_dataset": int(args.limit_per_dataset),
        "min_input_tokens": int(args.min_input_tokens),
        "expected_source_revision": str(args.expected_source_revision),
        "quantization": "Q16",
        "single_request_only": True,
        "batch_semantics": "batch-1-equal-length-only",
    }


def _protocol_config_sha256(config: dict[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _kernel_identity(kernel: Any) -> dict[str, Any]:
    try:
        signature = str(inspect.signature(kernel))
    except (TypeError, ValueError):
        signature = "<signature-unavailable>"
    return {
        "callable_id": id(kernel),
        "module": str(getattr(kernel, "__module__", type(kernel).__module__)),
        "qualname": str(getattr(kernel, "__qualname__", type(kernel).__qualname__)),
        "signature": signature,
    }


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    _require(reference.shape == candidate.shape, "diagnostic tensor shapes differ")
    reference_f = reference.float()
    candidate_f = candidate.float()
    error = candidate_f - reference_f
    denominator = reference_f.square().sum().sqrt().clamp_min(1e-30)
    return {
        "bitwise_exact": bool(torch.equal(reference, candidate)),
        "finite": bool(torch.isfinite(candidate_f).all()),
        "max_abs": float(error.abs().max().item()),
        "mean_abs": float(error.abs().mean().item()),
        "relative_l2": float(error.square().sum().sqrt().div(denominator).item()),
    }


def _extract_mask(attention_mask: Any) -> torch.Tensor | None:
    if isinstance(attention_mask, dict):
        attention_mask = attention_mask.get("full_attention")
    if attention_mask is not None and not isinstance(attention_mask, torch.Tensor):
        raise RuntimeError("attention mask is neither tensor nor None")
    return attention_mask


def _hf_eager_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    *,
    scaling: float,
    dropout: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if dropout != 0.0:
        raise RuntimeError("formal inference requires dropout=0")
    batch, kv_heads, length, dim = key.shape
    groups = int(module.num_key_value_groups)
    key = key[:, :, None].expand(batch, kv_heads, groups, length, dim).reshape(
        batch, kv_heads * groups, length, dim
    )
    value = value[:, :, None].expand(batch, kv_heads, groups, length, dim).reshape(
        batch, kv_heads * groups, length, dim
    )
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    mask = _extract_mask(attention_mask)
    if mask is not None:
        scores = scores + mask
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(weights, value).transpose(1, 2).contiguous()
    return output, weights


def _fp32_dense_attention(
    module: Any,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    *,
    scaling: float,
) -> torch.Tensor:
    """Dense FP32 diagnostic oracle; never used as the CoMem gate target."""

    batch, kv_heads, length, dim = key.shape
    groups = int(module.num_key_value_groups)
    key_f = key.float()[:, :, None].expand(
        batch, kv_heads, groups, length, dim
    ).reshape(batch, kv_heads * groups, length, dim)
    value_f = value.float()[:, :, None].expand(
        batch, kv_heads, groups, length, dim
    ).reshape(batch, kv_heads * groups, length, dim)
    scores = torch.matmul(query.float(), key_f.transpose(2, 3)) * float(scaling)
    mask = _extract_mask(attention_mask)
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, -torch.inf)
        else:
            scores = scores + mask.float()
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
    return torch.matmul(weights, value_f).transpose(1, 2).contiguous()


def _build_document_cache(backbone: Any, document: torch.Tensor, *, functional: bool):
    from transformers.cache_utils import DynamicCache

    original = backbone.config._attn_implementation
    cache = DynamicCache(config=backbone.config)
    if functional:
        install_native_functional_linear_cache(cache, backbone.config)
    try:
        backbone.config._attn_implementation = "eager"
        output = backbone(input_ids=document, past_key_values=cache, use_cache=True)
    finally:
        backbone.config._attn_implementation = original
    if output.past_key_values is not cache:
        raise RuntimeError("document prefill returned a different cache")
    return cache


def _fork_dense_functional(cache: Any, plan: Any):
    memo: dict[int, Any] = {}
    seen: set[int] = set()

    def seed(value: Any) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, torch.Tensor):
            memo[identity] = value
        elif isinstance(value, dict):
            for item in value.values():
                seed(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                seed(item)
        elif hasattr(value, "__dict__"):
            for item in vars(value).values():
                seed(item)

    seed(cache)
    request = copy.deepcopy(cache, memo)
    install_native_functional_linear_cache(request, plan.gdn)
    return request


def _last_logits(model: Any, output: Any) -> torch.Tensor:
    logits = model.lm_head(output.last_hidden_state[:, -1, :])
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise RuntimeError("invalid final logits")
    return logits


class SameKernelIsolatedGate:
    """Compare fresh and reuse using identical post-RoPE inputs and kernel."""

    def __init__(self, indices: tuple[int, ...], page_size: int, kernel: Any) -> None:
        self.indices = indices
        self.page_size = page_size
        self.kernel = kernel
        self.kernel_identity = _kernel_identity(kernel)
        self.counts = {index: 0 for index in indices}
        self.rows: list[dict[str, Any]] = []

    def attention_forward(
        self,
        module: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any,
        *args: Any,
        **kwargs: Any,
    ):
        del args
        index = int(module.layer_idx)
        if index not in self.counts or self.counts[index]:
            raise RuntimeError(f"isolated same-kernel gate unexpected/repeated layer {index}")
        scaling = float(kwargs.pop("scaling", module.scaling))
        dropout = float(kwargs.pop("dropout", 0.0))
        use_cache = kwargs.pop("use_cache", None)
        if use_cache is not None and use_cache is not True:
            raise RuntimeError("isolated cached attention requires use_cache=True")
        position_ids = kwargs.pop("position_ids", None)
        position_audit = validate_qwen35_post_rope_position_ids(
            position_ids,
            query=query,
            total_length=int(key.shape[-2]),
            strict_tail_values=True,
        )
        if kwargs:
            raise RuntimeError(f"isolated gate unsupported kwargs: {sorted(kwargs)}")
        hf_output, hf_weights = _hf_eager_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
            dropout=dropout,
        )
        fp32_output = _fp32_dense_attention(
            module,
            query,
            key,
            value,
            attention_mask,
            scaling=scaling,
        )
        query_length = int(query.shape[-2])
        prefix_length = int(key.shape[-2]) - query_length
        if prefix_length < 1:
            raise RuntimeError("isolated same-kernel gate has no document prefix")
        fresh, reuse, layout = build_same_kernel_q16_sequence_pair(
            key[..., :prefix_length, :],
            value[..., :prefix_length, :],
            key[..., prefix_length:, :],
            value[..., prefix_length:, :],
            page_size=self.page_size,
            max_append_tokens=query_length,
        )
        fresh_audit: dict[str, Any] = {}
        reuse_audit: dict[str, Any] = {}
        fresh_output, _ = vllm_triton_q16_paged_attention_forward(
            module,
            query,
            fresh.keys,
            fresh.values,
            attention_mask,
            scaling=scaling,
            audit=fresh_audit,
            _kernel=self.kernel,
        )
        reuse_output, _ = vllm_triton_q16_paged_attention_forward(
            module,
            query,
            reuse.keys,
            reuse.values,
            attention_mask,
            scaling=scaling,
            audit=reuse_audit,
            _kernel=self.kernel,
        )
        exact = bool(torch.equal(fresh_output, reuse_output))
        row = {
            "layer_idx": index,
            "passed": exact,
            "same_kernel_output_bitwise_exact": exact,
            "same_post_rope_query_object": True,
            "query_sha256": _tensor_sha256(query),
            "position_ids_sha256": _tensor_sha256(position_ids),
            "position_ids": position_audit,
            "same_scale": True,
            "scaling": scaling,
            "same_gqa_groups": True,
            "num_key_value_groups": int(module.num_key_value_groups),
            "same_causal_contract": True,
            "same_mask_object": True,
            "mask_sha256": (
                None
                if _extract_mask(attention_mask) is None
                else _tensor_sha256(_extract_mask(attention_mask))
            ),
            "same_kernel_callable_identity": True,
            "kernel_identity": dict(self.kernel_identity),
            "layout": layout,
            "fresh_kernel_audit": fresh_audit,
            "reuse_kernel_audit": reuse_audit,
            "fresh_vs_reuse": _error_metrics(fresh_output, reuse_output),
            "backend_compatibility_nonblocking": {
                "hf_eager_vs_fp32_dense": _error_metrics(fp32_output, hf_output),
                "vllm_fresh_vs_fp32_dense": _error_metrics(fp32_output, fresh_output),
                "vllm_reuse_vs_fp32_dense": _error_metrics(fp32_output, reuse_output),
                "hf_eager_vs_vllm_fresh": _error_metrics(hf_output, fresh_output),
            },
        }
        self.rows.append(row)
        self.counts[index] += 1
        if not exact:
            raise RuntimeError(f"same unified_attention output diverged at layer {index}")
        return hf_output, hf_weights

    def verify(self) -> dict[str, Any]:
        expected = {index: 1 for index in self.indices}
        if self.counts != expected:
            raise RuntimeError(f"isolated same-kernel coverage failed: {self.counts}")
        _require(all(row["passed"] for row in self.rows), "isolated same-kernel row failed")
        return {
            "passed": True,
            "fair_protocol": FAIR_PROTOCOL,
            "layer_indices": self.indices,
            "layer_count": len(self.indices),
            "same_kernel_callable_identity": True,
            "kernel_identity": dict(self.kernel_identity),
            "full_document_fresh_copy_control": True,
            "shared_document_private_tail_reuse": True,
            "backend_compatibility_is_gate": False,
            "rows": self.rows,
        }


def _register_isolated_backend(gate: SameKernelIsolatedGate) -> str:
    from transformers.masking_utils import AttentionMaskInterface, eager_mask
    from transformers.modeling_utils import AttentionInterface

    name = f"qcomem_fair_v2_isolated_{uuid.uuid4().hex}"
    AttentionInterface.register(name, gate.attention_forward)
    AttentionMaskInterface.register(name, eager_mask)
    return name


def _set_production_no_mask(request: Any, indices: tuple[int, ...]) -> None:
    for index in indices:
        request.layers[index].sequence.strict_mask_check = False


@torch.inference_mode()
def run_pg19_gate(args: argparse.Namespace) -> dict[str, Any]:
    """Run train-only same-kernel correctness and backend diagnostics."""

    records, data_audit = audit_pg19_train_calibration(
        args.pg19_data,
        args.pg19_manifest,
        expected_data_sha256=args.expected_pg19_sha256,
        expected_manifest_sha256=args.expected_pg19_manifest_sha256,
        minimum_books=args.pg19_books,
    )
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    windows, windows_sha = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=args.pg19_books,
        document_tokens=args.pg19_document_tokens,
        query_tokens=args.pg19_query_tokens,
        stride=args.pg19_window_stride,
        candidate_windows_per_book=args.pg19_candidate_windows,
        seed=args.pg19_seed,
    )
    _require(windows_sha == args.expected_pg19_windows_sha256, "PG19 windows SHA mismatch")
    assigned = list(enumerate(windows))[args.rank :: args.world_size]
    _require(len(assigned) == 1, "formal 8-rank PG19 gate requires one window per rank")
    torch.cuda.set_device(0)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    indices = tuple(plan.full_attention_layer_indices)
    _require(indices == FULL_LAYERS, "formal Qwen3.5 full-layer geometry drift")
    kernel = _resolve_vllm_unified_attention()
    kernel_identity = _kernel_identity(kernel)
    rows = []
    for window_index, window in assigned:
        document = window.document_ids.unsqueeze(0).cuda()
        query = window.query_ids.unsqueeze(0).cuda()
        _require(int(document.shape[0]) == int(query.shape[0]) == 1, "batch1 gate required")
        _require(
            int(document.shape[1]) % args.page_size != 0,
            "PG19 gate must exercise a non-page-aligned document tail",
        )
        stock = _build_document_cache(backbone, document, functional=False)
        hf_baseline = clone_cache(stock)
        isolated = SameKernelIsolatedGate(indices, args.page_size, kernel)
        isolated_name = _register_isolated_backend(isolated)
        original = backbone.config._attn_implementation
        try:
            backbone.config._attn_implementation = isolated_name
            backbone(input_ids=query, past_key_values=stock, use_cache=True)
        finally:
            backbone.config._attn_implementation = original
            _unregister_backend(isolated_name)
        try:
            backbone.config._attn_implementation = "eager"
            hf_output = backbone(
                input_ids=query, past_key_values=hf_baseline, use_cache=True
            )
        finally:
            backbone.config._attn_implementation = original
        hf_logits = _last_logits(model, hf_output)

        persistent = _build_document_cache(backbone, document, functional=True)
        conversion = convert_all_qwen35_full_layers_to_vllm_q16(
            persistent,
            plan,
            page_size=args.page_size,
            max_append_tokens=int(query.shape[1]),
            max_request_forks=1,
        )
        persistent_gdn_snapshot = snapshot_linear_gdn_state(
            persistent, plan.linear_layer_indices
        )
        fresh, fresh_fork = materialize_qwen35_fresh_full_copy_request(persistent, plan)
        reuse, reuse_fork = fork_qwen35_vllm_q16_request(persistent, plan)
        fresh_gdn_base = linear_gdn_shared_base_contract(
            persistent, fresh, plan.linear_layer_indices
        )
        reuse_gdn_base = linear_gdn_shared_base_contract(
            persistent, reuse, plan.linear_layer_indices
        )
        fresh_storage_before = full_attention_storage_breakdown(
            persistent,
            fresh,
            plan.full_attention_layer_indices,
            request_policy=FRESH_CONTROL,
        )
        reuse_storage_before = full_attention_storage_breakdown(
            persistent,
            reuse,
            plan.full_attention_layer_indices,
            request_policy=SHARED_REUSE,
        )
        _validate_storage_accounting(
            fresh_storage_before,
            allocated_source_pool_nbytes=conversion.allocated_block_pool_nbytes,
            request_audit=fresh_fork,
            request_policy=FRESH_CONTROL,
            document_length=conversion.document_length,
            page_size=conversion.page_size,
            expected_appended_tokens=0,
        )
        _validate_storage_accounting(
            reuse_storage_before,
            allocated_source_pool_nbytes=conversion.allocated_block_pool_nbytes,
            request_audit=reuse_fork,
            request_policy=SHARED_REUSE,
            document_length=conversion.document_length,
            page_size=conversion.page_size,
            expected_appended_tokens=0,
        )
        fresh_ledger = Qwen35FairHitLedger(
            plan,
            fresh,
            request_policy=FRESH_CONTROL,
            expected_calls_per_layer=1,
            strict_tail_values=True,
            kernel=kernel,
        )
        reuse_ledger = Qwen35FairHitLedger(
            plan,
            reuse,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            strict_tail_values=True,
            kernel=kernel,
        )
        _require(
            fresh_ledger.kernel is reuse_ledger.kernel is kernel,
            "PG19 arms do not share one unified_attention callable",
        )
        fresh_backend = register_qwen35_fair_backend(fresh_ledger)
        reuse_backend = register_qwen35_fair_backend(reuse_ledger)
        try:
            backbone.config._attn_implementation = fresh_backend
            fresh_output = backbone(input_ids=query, past_key_values=fresh, use_cache=True)
            backbone.config._attn_implementation = reuse_backend
            reuse_output = backbone(input_ids=query, past_key_values=reuse, use_cache=True)
        finally:
            backbone.config._attn_implementation = original
            _unregister_backend(fresh_backend)
            _unregister_backend(reuse_backend)
        fresh_logits = _last_logits(model, fresh_output)
        reuse_logits = _last_logits(model, reuse_output)
        logit_exact = bool(torch.equal(fresh_logits, reuse_logits))
        fresh_sha = _tensor_sha256(fresh_logits)
        reuse_sha = _tensor_sha256(reuse_logits)
        _require(logit_exact and fresh_sha == reuse_sha, "PG19 same-kernel logits diverged")
        gdn_after = verify_linear_gdn_state_parity(
            fresh,
            reuse,
            persistent,
            persistent_gdn_snapshot,
            plan.linear_layer_indices,
        )
        fresh_storage_after = full_attention_storage_breakdown(
            persistent,
            fresh,
            plan.full_attention_layer_indices,
            request_policy=FRESH_CONTROL,
        )
        reuse_storage_after = full_attention_storage_breakdown(
            persistent,
            reuse,
            plan.full_attention_layer_indices,
            request_policy=SHARED_REUSE,
        )
        _validate_storage_accounting(
            fresh_storage_after,
            allocated_source_pool_nbytes=conversion.allocated_block_pool_nbytes,
            request_audit=fresh_fork,
            request_policy=FRESH_CONTROL,
            document_length=conversion.document_length,
            page_size=conversion.page_size,
            expected_appended_tokens=int(query.shape[1]),
        )
        _validate_storage_accounting(
            reuse_storage_after,
            allocated_source_pool_nbytes=conversion.allocated_block_pool_nbytes,
            request_audit=reuse_fork,
            request_policy=SHARED_REUSE,
            document_length=conversion.document_length,
            page_size=conversion.page_size,
            expected_appended_tokens=int(query.shape[1]),
        )
        isolated_result = isolated.verify()
        fresh_intercept = fresh_ledger.verify_complete()
        reuse_intercept = reuse_ledger.verify_complete()
        _require(
            fresh_intercept["kernel_identity"] == reuse_intercept["kernel_identity"],
            "PG19 ledger kernel identities differ",
        )
        rows.append(
            {
                "window_index": window_index,
                "source_object": window.source_object,
                "document_tokens": int(document.shape[1]),
                "query_tokens": int(query.shape[1]),
                "document_tail_tokens": int(document.shape[1]) % args.page_size,
                "kernel_identity": kernel_identity,
                "isolated_same_kernel": isolated_result,
                "semantic_same_kernel": {
                    "passed": True,
                    "full_vocab_logits_bitwise_exact": logit_exact,
                    "fresh_logits_sha256": fresh_sha,
                    "reuse_logits_sha256": reuse_sha,
                    "top1_exact": bool(
                        torch.equal(fresh_logits.argmax(-1), reuse_logits.argmax(-1))
                    ),
                    "full_vocab_forward_kl": float(
                        full_vocab_forward_kl(fresh_logits, reuse_logits).mean().item()
                    ),
                    "fresh_intercept": fresh_intercept,
                    "reuse_intercept": reuse_intercept,
                    "fresh_request": fresh_fork,
                    "reuse_request": reuse_fork,
                    "fresh_residency": storage_residency(persistent, fresh),
                    "reuse_residency": storage_residency(persistent, reuse),
                    "fresh_full_attention_storage_before": fresh_storage_before,
                    "reuse_full_attention_storage_before": reuse_storage_before,
                    "fresh_full_attention_storage_after": fresh_storage_after,
                    "reuse_full_attention_storage_after": reuse_storage_after,
                    "fresh_linear_gdn_shared_base": fresh_gdn_base,
                    "reuse_linear_gdn_shared_base": reuse_gdn_base,
                    "linear_gdn_after_query": gdn_after,
                    "only_full_attention_ownership_differs": True,
                },
                "backend_compatibility_nonblocking": {
                    "is_gate": False,
                    "reason": "HF eager and vLLM use different BF16 reduction/rounding orders",
                    "hf_vs_vllm_full_vocab_kl": float(
                        full_vocab_forward_kl(hf_logits, fresh_logits).mean().item()
                    ),
                    "hf_vs_vllm_top1_exact": bool(
                        torch.equal(hf_logits.argmax(-1), fresh_logits.argmax(-1))
                    ),
                    "hf_vs_vllm_logit_error": _error_metrics(hf_logits, fresh_logits),
                },
            }
        )
        del stock, hf_baseline, persistent, fresh, reuse, document, query
        _fresh_allocator_cleanup("pg19-shard-cleanup")
    result = {
        "status": "completed_pg19_fair_v2_gate_shard",
        "passed": True,
        "rank": args.rank,
        "world_size": args.world_size,
        "fair_protocol": FAIR_PROTOCOL,
        "kernel_mode": KERNEL_MODE,
        "kernel_identity": kernel_identity,
        "quantization": "Q16",
        "single_request_only": True,
        "batch_semantics": "batch-1-equal-length-only",
        "data_audit": data_audit,
        "windows_sha256": windows_sha,
        "rows": rows,
        "backend_compatibility_is_authorization_gate": False,
        "validation_consumed": False,
        "validation_hashed": False,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
    }
    return result


def aggregate_pg19_gate_shards(
    paths: list[Path],
    *,
    expected_windows_sha256: str,
    expected_frozen_identity: dict[str, str],
) -> dict[str, Any]:
    _require(len(paths) == 8, "PG19 authorization requires eight shard artifacts")
    shards = [json.loads(path.read_text()) for path in paths]
    _require(sorted(int(row["rank"]) for row in shards) == list(range(8)), "PG19 ranks drift")
    source_objects = []
    kernel_descriptors = set()
    backend_rows = []
    frozen_identities = []
    protocol_configs = []
    for shard in shards:
        _require(
            shard.get("status") == "completed_pg19_fair_v2_gate_shard"
            and shard.get("passed") is True,
            "PG19 v2 shard failed",
        )
        _require(shard.get("world_size") == 8, "PG19 world size drift")
        _require(shard.get("fair_protocol") == FAIR_PROTOCOL, "PG19 protocol drift")
        _require(shard.get("kernel_mode") == KERNEL_MODE, "PG19 kernel mode drift")
        _require(shard.get("quantization") == "Q16", "PG19 non-Q16 result")
        _require(shard.get("single_request_only") is True, "PG19 serving scope drift")
        static = shard.get("static")
        _require(
            isinstance(static, dict)
            and static.get("status") == "fair_v2_static_dry_run_passed"
            and static.get("gpu_initialized") is False
            and static.get("validation_consumed") is False
            and static.get("validation_hashed") is False
            and static.get("environment", {}).get("matches_frozen_environment") is True,
            "PG19 shard static identity/environment failed",
        )
        shard_identity = _static_frozen_identity(static)
        _require(
            shard_identity == expected_frozen_identity,
            "PG19 shard belongs to a different current frozen snapshot",
        )
        frozen_identities.append(shard_identity)
        protocol = static.get("protocol_config")
        _require(
            isinstance(protocol, dict)
            and _protocol_config_sha256(protocol)
            == static.get("protocol_config_sha256"),
            "PG19 protocol config SHA drift",
        )
        protocol_configs.append(protocol)
        _require(shard.get("windows_sha256") == expected_windows_sha256, "PG19 windows drift")
        _require(
            shard.get("validation_consumed") is False
            and shard.get("validation_hashed") is False
            and shard.get("source_68_99_consumed") is False
            and shard.get("test_v2_consumed") is False,
            "PG19 phase consumed forbidden evaluation data",
        )
        rows = shard.get("rows")
        _require(isinstance(rows, list) and len(rows) == 1, "PG19 shard row count drift")
        row = rows[0]
        _require(
            int(row.get("window_index", -1)) == int(shard["rank"]),
            "PG19 rank/window index assignment drift",
        )
        source_objects.append(str(row["source_object"]))
        _require(int(row["document_tail_tokens"]) > 0, "PG19 missed partial-tail case")
        isolated = row["isolated_same_kernel"]
        _require(isolated.get("passed") is True, "isolated same-kernel gate failed")
        _require(isolated.get("layer_count") == 10, "isolated layer count drift")
        _require(len(isolated.get("rows", ())) == 10, "isolated row count drift")
        for layer in isolated["rows"]:
            _require(layer.get("passed") is True, "isolated layer failed")
            _require(
                layer.get("same_kernel_output_bitwise_exact") is True
                and layer.get("same_post_rope_query_object") is True
                and layer.get("same_scale") is True
                and layer.get("same_gqa_groups") is True
                and layer.get("same_causal_contract") is True
                and layer.get("same_mask_object") is True
                and layer.get("same_kernel_callable_identity") is True,
                "isolated same-input/kernel contract drift",
            )
            layout = layer.get("layout", {}).get("layout", {})
            _require(
                layout.get("canonical_layout_equal") is True
                and layout.get("valid_key_payload_bitwise_exact") is True
                and layout.get("valid_value_payload_bitwise_exact") is True,
                "isolated logical K/V layout drift",
            )
            _require(
                layer["layout"].get("document_source_immutable") is True
                and layer["layout"].get("fresh_source_storage_shared") is False,
                "isolated storage ownership drift",
            )
            for arm in ("fresh_kernel_audit", "reuse_kernel_audit"):
                audit = layer[arm]
                _require(audit.get("fused_gpu_kernel_calls") == 1, "isolated kernel call drift")
                _require(audit.get("full_kv_concatenations") == 0, "isolated full KV cat")
        semantic = row["semantic_same_kernel"]
        fresh_logits_sha = _validate_sha256_hex(
            semantic.get("fresh_logits_sha256"), "PG19 fresh logits"
        )
        reuse_logits_sha = _validate_sha256_hex(
            semantic.get("reuse_logits_sha256"), "PG19 reuse logits"
        )
        _require(
            semantic.get("passed") is True
            and semantic.get("full_vocab_logits_bitwise_exact") is True
            and semantic.get("top1_exact") is True
            and semantic.get("full_vocab_forward_kl") == 0.0
            and fresh_logits_sha == reuse_logits_sha,
            "PG19 same-kernel semantic parity failed",
        )
        fresh = semantic["fresh_intercept"]
        reuse = semantic["reuse_intercept"]
        _validate_intercept(fresh, FRESH_CONTROL, 1, strict=True)
        _validate_intercept(reuse, SHARED_REUSE, 1, strict=True)
        _require(fresh["kernel_identity"] == reuse["kernel_identity"], "semantic kernels differ")
        _require(semantic.get("only_full_attention_ownership_differs") is True, "arm scope drift")
        for key in ("fresh_linear_gdn_shared_base", "reuse_linear_gdn_shared_base"):
            contract = semantic[key]
            _require(
                contract.get("passed") is True
                and contract.get("linear_layer_count") == 30
                and contract.get("persistent_tensor_base_shared_at_request_start") is True
                and contract.get("request_updates_are_functional_rebind") is True,
                "linear GDN initial-base contract failed",
            )
        gdn_after = semantic["linear_gdn_after_query"]
        _require(
            gdn_after.get("passed") is True
            and gdn_after.get("linear_layer_count") == 30
            and gdn_after.get("fresh_reuse_functional_state_bitwise_exact") is True
            and gdn_after.get("persistent_tensor_base_unchanged") is True,
            "linear GDN functional parity failed",
        )
        for policy, key in (
            (FRESH_CONTROL, "fresh_full_attention_storage_before"),
            (SHARED_REUSE, "reuse_full_attention_storage_before"),
            (FRESH_CONTROL, "fresh_full_attention_storage_after"),
            (SHARED_REUSE, "reuse_full_attention_storage_after"),
        ):
            storage = semantic[key]
            _require(storage.get("request_policy") == policy, "storage policy drift")
            _require(storage.get("full_attention_layer_count") == 10, "storage layer drift")
            _require(
                storage.get("source_arena_includes_preallocated_private_reservation") is True,
                "source private reservation is hidden",
            )
        descriptor = fresh["kernel_identity"]
        kernel_descriptors.add(
            (descriptor["module"], descriptor["qualname"], descriptor["signature"])
        )
        backend_rows.append(row["backend_compatibility_nonblocking"])
    _require(len(set(source_objects)) == 8, "PG19 train source objects are not unique")
    _require(
        sorted(int(row["rows"][0]["window_index"]) for row in shards)
        == list(range(8)),
        "PG19 authorization does not cover exact window indices 0--7",
    )
    _require(len(kernel_descriptors) == 1, "PG19 kernel signature differs across ranks")
    _require(
        all(identity == frozen_identities[0] for identity in frozen_identities),
        "PG19 frozen code/model/data/protocol identity differs across ranks",
    )
    _require(
        frozen_identities[0] == expected_frozen_identity,
        "PG19 shards belong to a different current frozen snapshot",
    )
    _require(
        all(config == protocol_configs[0] for config in protocol_configs),
        "PG19 protocol config differs across ranks",
    )
    return {
        "status": "pg19_fair_v2_authorized",
        "passed": True,
        "fair_protocol": FAIR_PROTOCOL,
        "kernel_mode": KERNEL_MODE,
        "kernel_descriptor": list(kernel_descriptors)[0],
        "parallel_gate_world_size": 8,
        "source_objects": source_objects,
        "windows_sha256": expected_windows_sha256,
        "frozen_identity": frozen_identities[0],
        "protocol_config": protocol_configs[0],
        "same_kernel_layout_gate_passed": True,
        "same_kernel_full_vocab_logit_gate_passed": True,
        "backend_compatibility_is_authorization_gate": False,
        "backend_compatibility_nonblocking_rows": backend_rows,
        "quantization": "Q16",
        "single_request_only": True,
        "batch_semantics": "batch-1-equal-length-only",
        "validation_consumed": False,
        "validation_hashed": False,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
    }


def _validate_authorization(value: Any) -> None:
    _require(isinstance(value, dict), "PG19 authorization must be an object")
    _require(
        value.get("status") == "pg19_fair_v2_authorized"
        and value.get("passed") is True,
        "PG19 fair-v2 authorization is absent or failed",
    )
    _require(value.get("fair_protocol") == FAIR_PROTOCOL, "authorization protocol drift")
    _require(
        value.get("same_kernel_layout_gate_passed") is True
        and value.get("same_kernel_full_vocab_logit_gate_passed") is True,
        "same-kernel authorization gates did not pass",
    )
    _require(
        value.get("validation_consumed") is False
        and value.get("validation_hashed") is False
        and value.get("source_68_99_consumed") is False
        and value.get("test_v2_consumed") is False,
        "authorization data-governance fields drifted",
    )
    frozen = value.get("frozen_identity")
    _require(isinstance(frozen, dict), "authorization lacks frozen identity")
    identity = _static_frozen_identity(frozen)
    protocol = value.get("protocol_config")
    _require(isinstance(protocol, dict), "authorization lacks protocol config")
    _require(
        _protocol_config_sha256(protocol) == identity["protocol_config_sha256"],
        "authorization protocol config SHA drift",
    )


def _validate_intercept(
    intercept: Any,
    request_policy: str,
    expected_calls_per_layer: int,
    *,
    strict: bool,
) -> None:
    _require(isinstance(intercept, dict), "missing same-kernel intercept")
    _require(intercept.get("verified") is True, "same-kernel intercept unverified")
    _require(intercept.get("fair_protocol") == FAIR_PROTOCOL, "intercept protocol drift")
    _require(intercept.get("request_policy") == request_policy, "request policy drift")
    _require(intercept.get("kernel_mode") == KERNEL_MODE, "intercept kernel drift")
    _require(intercept.get("same_unified_attention_kernel") is True, "kernel claim absent")
    intercept_kernel_identity = _validate_kernel_identity_schema(
        intercept.get("kernel_identity"), "intercept"
    )
    _require(tuple(intercept.get("expected_layer_indices", ())) == FULL_LAYERS, "layer drift")
    expected = {index: expected_calls_per_layer for index in FULL_LAYERS}
    counts = {int(key): int(value) for key, value in intercept.get("counts", {}).items()}
    _require(counts == expected, f"kernel hit counts drifted: {counts}")
    _require(intercept.get("total_calls") == 10 * expected_calls_per_layer, "call total drift")
    _require(intercept.get("dense_fallback_calls") == 0, "dense fallback observed")
    _require(intercept.get("full_kv_concatenations") == 0, "full KV cat observed")
    _require(
        intercept.get("position_ids_contract") == POST_ROPE_POSITION_IDS_CONTRACT,
        "position_ids contract drift",
    )
    expected_mask = "strict-canonical-audit" if strict else PRODUCTION_MASK_CONTRACT
    _require(intercept.get("mask_contract") == expected_mask, "mask contract drift")
    if not strict:
        _require(intercept.get("materialized_attention_mask_nbytes") == 0, "timed mask materialized")
        _require(intercept.get("mask_validation_host_syncs") == 0, "timed mask synchronized")
        _require(
            intercept.get("position_ids_validation_host_syncs") == 0,
            "timed position_ids synchronized",
        )
    calls = intercept.get("calls")
    _require(isinstance(calls, (list, tuple)) and len(calls) == 10 * expected_calls_per_layer, "call ledger drift")
    replayed_counts = Counter(int(call.get("layer_idx", -1)) for call in calls)
    _require(
        dict(replayed_counts) == expected,
        "call-layer replay counts differ from intercept counts",
    )
    _require(
        [int(call.get("layer_idx", -1)) for call in calls]
        == list(FULL_LAYERS) * expected_calls_per_layer,
        "call-layer execution order drift",
    )
    for call in calls:
        call_kernel_identity = _validate_kernel_identity_schema(
            call.get("kernel_identity"), "intercept call"
        )
        _require(call.get("request_policy") == request_policy, "call policy drift")
        _require(call.get("fair_protocol") == FAIR_PROTOCOL, "call protocol drift")
        _require(call.get("same_unified_attention_kernel") is True, "call kernel claim absent")
        _require(call.get("kernel_mode") == KERNEL_MODE, "call kernel mode drift")
        _require(
            call_kernel_identity == intercept_kernel_identity,
            "call kernel identity differs from intercept identity",
        )
        _require(call.get("fused_gpu_kernel_calls") == 1, "call is not one fused dispatch")
        _require(call.get("full_kv_concatenations") == 0, "call concatenated KV")
        append_delta = _required_positive_int(
            call.get("current_append_delta_tokens"),
            "call current_append_delta_tokens",
        )
        query_tokens = _required_positive_int(
            call.get("query_tokens"), "call query_tokens"
        )
        _require(append_delta == query_tokens, "append delta drift")
        _require(call.get("quantization") == "Q16", "call is not Q16")
        if not strict:
            _require(
                call.get("materialized_attention_mask_nbytes") == 0
                and call.get("mask_validation_host_syncs") == 0
                and call.get("position_ids_validation_host_syncs") == 0,
                "timed call introduced host/mask overhead",
            )


def _validate_storage_accounting(
    storage: dict[str, Any],
    *,
    allocated_source_pool_nbytes: int,
    request_audit: dict[str, Any],
    request_policy: str,
    document_length: int,
    page_size: int,
    expected_appended_tokens: int,
) -> None:
    _require(isinstance(storage, dict), "storage ledger must be a dict")
    _require(request_policy in FAIR_CONFIGS, "storage request policy is unknown")
    _require(storage.get("request_policy") == request_policy, "storage policy drift")
    _require(
        storage.get("full_attention_layer_count") == 10
        and storage.get("scope") == "ten-full-attention-layers-only"
        and storage.get("linear_gdn_included") is False
        and storage.get("source_arena_includes_preallocated_private_reservation")
        is True
        and storage.get("invalid_final_block_padding_is_payload") is False,
        "storage ledger scope/governance drift",
    )
    layers = storage.get("layers")
    _require(
        isinstance(layers, list)
        and len(layers) == 10
        and [row.get("layer_idx") for row in layers] == list(FULL_LAYERS),
        "storage ledger does not contain the exact ten full-attention layers",
    )
    totals = storage.get("totals")
    _require(isinstance(totals, dict), "storage totals must be a dict")
    document_length = _required_positive_int(
        document_length, "storage document_length"
    )
    page_size = _required_positive_int(page_size, "storage page_size")
    expected_appended_tokens = _required_nonnegative_int(
        expected_appended_tokens, "storage expected appended tokens"
    )
    expected_detached_tail = (
        document_length % page_size if expected_appended_tokens > 0 else 0
    )
    expected_active_tokens = expected_detached_tail + expected_appended_tokens
    expected_active_blocks = (
        math.ceil(expected_active_tokens / page_size)
        if expected_appended_tokens > 0
        else 0
    )
    expected_total_keys: set[str] | None = None
    recomputed_totals: Counter[str] = Counter()
    is_fresh = request_policy == FRESH_CONTROL
    for row in layers:
        _require(isinstance(row, dict), "storage layer row must be a dict")
        numeric_keys = {
            key
            for key in row
            if key.endswith("_nbytes")
            or key in ("block_bytes", "active_request_private_blocks")
        }
        if expected_total_keys is None:
            expected_total_keys = numeric_keys
        _require(
            numeric_keys == expected_total_keys,
            "storage layer numeric schema differs across layers",
        )
        for key in numeric_keys:
            recomputed_totals[key] += _required_nonnegative_int(
                row.get(key), f"storage layer {row.get('layer_idx')} {key}"
            )
        block_bytes = _required_positive_int(
            row.get("block_bytes"),
            f"storage layer {row.get('layer_idx')} block_bytes",
        )
        _require(
            block_bytes % page_size == 0,
            "storage block bytes are not divisible by the formal page size",
        )
        valid_payload = row["valid_document_payload_nbytes"]
        source_document = row["source_document_allocated_nbytes"]
        source_padding = row["source_document_padding_nbytes"]
        source_private = row["source_private_reservation_nbytes"]
        source_arena_total = row["source_total_arena_allocated_nbytes"]
        _require(
            source_document - valid_payload == source_padding
            and source_document + source_private == source_arena_total,
            "storage source payload/padding/arena formula drift",
        )
        fresh_document = row["fresh_duplicate_document_allocated_nbytes"]
        fresh_padding = row["fresh_duplicate_document_padding_nbytes"]
        fresh_private = row["fresh_private_reservation_nbytes"]
        if is_fresh:
            _require(
                fresh_document == source_document
                and fresh_padding == source_padding
                and fresh_private == source_private
                and row.get("source_document_storage_shared_by_request") is False,
                "fresh storage duplicate document/private formula drift",
            )
            reservation = fresh_private
        else:
            _require(
                fresh_document == fresh_padding == fresh_private == 0
                and row.get("source_document_storage_shared_by_request") is True,
                "reuse storage reports fresh allocation or unshared source",
            )
            reservation = source_private
        active_payload = row["active_request_private_payload_nbytes"]
        active_blocks = row["active_request_private_blocks"]
        active_allocated = row["active_request_private_allocated_page_nbytes"]
        reserved_unused = row["request_private_reserved_unused_nbytes"]
        appended_tokens = _required_nonnegative_int(
            row.get("active_request_appended_tokens"),
            f"storage layer {row.get('layer_idx')} appended tokens",
        )
        detached_tail_tokens = _required_nonnegative_int(
            row.get("active_request_detached_tail_tokens"),
            f"storage layer {row.get('layer_idx')} detached tail tokens",
        )
        _require(
            appended_tokens == expected_appended_tokens
            and detached_tail_tokens == expected_detached_tail
            and active_payload
            == expected_active_tokens * block_bytes // page_size
            and active_blocks == expected_active_blocks
            and active_allocated == active_blocks * block_bytes
            and 0 <= active_payload <= active_allocated <= reservation
            and reserved_unused == reservation - active_allocated,
            "storage active-private token/payload/page/reservation formula drift",
        )
    _require(expected_total_keys is not None, "storage numeric schema is absent")
    _require(
        set(totals) == expected_total_keys,
        "storage totals numeric schema differs from layer rows",
    )
    for key in sorted(expected_total_keys):
        _require(
            _required_nonnegative_int(totals.get(key), f"storage totals {key}")
            == recomputed_totals[key],
            f"storage totals {key} differs from ten-layer replay",
        )
    allocated_source_pool = _required_nonnegative_int(
        allocated_source_pool_nbytes, "allocated source pool bytes"
    )
    source_total = totals["source_document_allocated_nbytes"] + totals[
        "source_private_reservation_nbytes"
    ]
    _require(
        source_total == totals["source_total_arena_allocated_nbytes"]
        == allocated_source_pool,
        "source document/private reservation accounting does not equal arena allocation",
    )
    _require(
        totals["source_document_padding_nbytes"]
        == totals["source_document_allocated_nbytes"]
        - totals["valid_document_payload_nbytes"],
        "storage total document padding formula drift",
    )
    active_allocated = totals["active_request_private_allocated_page_nbytes"]
    active_payload = totals["active_request_private_payload_nbytes"]
    source_reserved = totals["source_private_reservation_nbytes"]
    request_pool = _required_nonnegative_int(
        request_audit.get("allocated_request_pool_nbytes"),
        "request audit allocated_request_pool_nbytes",
    )
    physical_copy = _required_nonnegative_int(
        request_audit.get("full_document_staging_copy_nbytes"),
        "request audit full_document_staging_copy_nbytes",
    )
    if is_fresh:
        fresh_total = totals["fresh_duplicate_document_allocated_nbytes"] + totals[
            "fresh_private_reservation_nbytes"
        ]
        _require(
            fresh_total == request_pool,
            "fresh duplicate document/private accounting does not equal request pool",
        )
        _require(
            physical_copy == totals["fresh_duplicate_document_allocated_nbytes"]
            and physical_copy > 0,
            "fresh physical document copy differs from duplicate document allocation",
        )
        _require(
            totals["fresh_duplicate_document_padding_nbytes"]
            == totals["source_document_padding_nbytes"],
            "fresh duplicate padding total drift",
        )
        reservation = totals["fresh_private_reservation_nbytes"]
    else:
        _require(
            totals["fresh_duplicate_document_allocated_nbytes"] == 0
            and totals["fresh_duplicate_document_padding_nbytes"] == 0
            and totals["fresh_private_reservation_nbytes"] == 0
            and request_pool == 0
            and physical_copy == 0,
            "reuse path reports a fresh duplicate pool",
        )
        reservation = source_reserved
    _require(
        0 <= active_payload <= active_allocated <= reservation,
        "active private pages are not a subset of reserved private capacity",
    )
    _require(
        totals["request_private_reserved_unused_nbytes"]
        == reservation - active_allocated,
        "reserved-unused private capacity accounting drift",
    )


def _build_q16_source(args: argparse.Namespace, backbone: Any, plan: Any, document: torch.Tensor):
    _sync()
    torch.cuda.reset_peak_memory_stats()
    prefill_before = _allocator_snapshot("before-common-dense-document-prefill")
    prefill_started = time.perf_counter()
    persistent = _build_document_cache(backbone, document, functional=True)
    _sync()
    prefill_after = _allocator_snapshot("after-common-dense-document-prefill")
    prefill = {
        "dense_document_prefill_seconds": time.perf_counter() - prefill_started,
        "dense_document_prefill_cuda_current_allocated_delta_bytes": (
            prefill_after["current_allocated_bytes"]
            - prefill_before["current_allocated_bytes"]
        ),
        "dense_document_prefill_cuda_current_reserved_delta_bytes": (
            prefill_after["current_reserved_bytes"]
            - prefill_before["current_reserved_bytes"]
        ),
        "dense_document_prefill_cuda_peak_delta_bytes": (
            prefill_after["peak_allocated_bytes"]
            - prefill_before["current_allocated_bytes"]
        ),
        "dense_document_prefill_cuda_peak_reserved_delta_bytes": (
            prefill_after["peak_reserved_bytes"]
            - prefill_before["current_reserved_bytes"]
        ),
        "dense_document_prefill_allocator_before": prefill_before,
        "dense_document_prefill_allocator_after": prefill_after,
    }
    _sync()
    torch.cuda.reset_peak_memory_stats()
    pack_before = _allocator_snapshot("before-common-dense-to-nhd-pack")
    pack_started = time.perf_counter()
    conversion = convert_all_qwen35_full_layers_to_vllm_q16(
        persistent,
        plan,
        page_size=args.page_size,
        max_append_tokens=int(args.max_query_tokens) + args.max_new_tokens,
        max_request_forks=1,
    )
    _sync()
    pack_after = _allocator_snapshot("after-common-dense-to-nhd-pack")
    pack = {
        "q16_pool_build_seconds": time.perf_counter() - pack_started,
        "q16_pool_build_cuda_current_allocated_delta_bytes": (
            pack_after["current_allocated_bytes"]
            - pack_before["current_allocated_bytes"]
        ),
        "q16_pool_build_cuda_current_reserved_delta_bytes": (
            pack_after["current_reserved_bytes"]
            - pack_before["current_reserved_bytes"]
        ),
        "q16_pool_build_cuda_peak_delta_bytes": (
            pack_after["peak_allocated_bytes"] - pack_before["current_allocated_bytes"]
        ),
        "q16_pool_build_cuda_peak_reserved_delta_bytes": (
            pack_after["peak_reserved_bytes"] - pack_before["current_reserved_bytes"]
        ),
        "dense_to_nhd_document_copy_nbytes": conversion.dense_document_nbytes,
        "q16_document_payload_nbytes": conversion.document_payload_nbytes,
        "q16_allocated_source_pool_nbytes": conversion.allocated_block_pool_nbytes,
        "full_attention_layers": len(conversion.layer_indices),
        "q16_pool_build_allocator_before": pack_before,
        "q16_pool_build_allocator_after": pack_after,
    }
    return persistent, conversion, {**prefill, **pack}


def _generate_backend(
    model: Any,
    backbone: Any,
    tokenizer: Any,
    request: Any,
    backend_name: str,
    query: torch.Tensor,
    max_new_tokens: int,
) -> dict[str, Any]:
    original = backbone.config._attn_implementation
    current = query
    tokens: list[int] = []
    times: list[float] = []
    logit_sha256: list[str] = []
    allocator_after_first_step: dict[str, Any] | None = None
    try:
        backbone.config._attn_implementation = backend_name
        for _ in range(max_new_tokens):
            _sync()
            started = time.perf_counter()
            output = backbone(input_ids=current, past_key_values=request, use_cache=True)
            logits = model.lm_head(output.last_hidden_state[:, -1, :])
            token = int(logits.argmax(-1).item())
            _sync()
            times.append(time.perf_counter() - started)
            if allocator_after_first_step is None:
                allocator_after_first_step = _allocator_snapshot(
                    "after-first-continuation-model-step"
                )
            # The digest is intentionally outside the timed interval.  It is a
            # bitwise full-vocabulary parity gate, not a serving operation.
            logit_sha256.append(_tensor_sha256(logits))
            tokens.append(token)
            current = torch.tensor([[token]], dtype=torch.long, device=query.device)
    finally:
        backbone.config._attn_implementation = original
    return {
        "generated_token_ids": tokens,
        "generated_text": tokenizer.decode(tokens, skip_special_tokens=True),
        "full_vocab_step_logit_sha256": logit_sha256,
        "continuation_model_first_token_seconds": times[0],
        "median_tpot_seconds": statistics.median(times[1:]),
        "allocator_after_first_continuation_step": allocator_after_first_step,
    }


def _measure_same_kernel_config(
    args: argparse.Namespace,
    model: Any,
    backbone: Any,
    tokenizer: Any,
    plan: Any,
    document: torch.Tensor,
    query: torch.Tensor,
    config: str,
) -> dict[str, Any]:
    _require(config in FAIR_CONFIGS, f"unknown fair config {config}")
    _require(int(document.shape[0]) == int(query.shape[0]) == 1, "batch1 required")
    trial_baseline = _allocator_snapshot("fresh-trial-baseline-before-common-build")
    persistent, conversion, build = _build_q16_source(args, backbone, plan, document)
    kernel = _resolve_vllm_unified_attention()
    _sync()
    torch.cuda.reset_peak_memory_stats()
    request_before = _allocator_snapshot("before-per-request-setup")
    request_started = time.perf_counter()
    if config == FRESH_CONTROL:
        request, request_audit = materialize_qwen35_fresh_full_copy_request(
            persistent, plan
        )
    else:
        request, request_audit = fork_qwen35_vllm_q16_request(persistent, plan)
        request_audit.update(
            {
                "request_policy": SHARED_REUSE,
                "single_request_only": True,
                "same_unified_attention_kernel": True,
                "source_document_storage_shared": True,
                "allocated_request_pool_nbytes": 0,
            }
        )
    gdn_base = linear_gdn_shared_base_contract(
        persistent, request, plan.linear_layer_indices
    )
    _set_production_no_mask(request, tuple(plan.full_attention_layer_indices))
    ledger = Qwen35FairHitLedger(
        plan,
        request,
        request_policy=config,
        expected_calls_per_layer=args.max_new_tokens,
        strict_tail_values=False,
        kernel=kernel,
    )
    backend = register_qwen35_fair_backend(ledger)
    _sync()
    prepare_seconds = time.perf_counter() - request_started
    request_after_setup = _allocator_snapshot("after-per-request-setup-before-continuation")
    prepare_peak = (
        request_after_setup["peak_allocated_bytes"]
        - request_before["current_allocated_bytes"]
    )
    prepare_peak_reserved = (
        request_after_setup["peak_reserved_bytes"]
        - request_before["current_reserved_bytes"]
    )
    prepare_current_allocated = (
        request_after_setup["current_allocated_bytes"]
        - request_before["current_allocated_bytes"]
    )
    prepare_current_reserved = (
        request_after_setup["current_reserved_bytes"]
        - request_before["current_reserved_bytes"]
    )
    memory_before_decode = storage_residency(persistent, request)
    full_attention_storage_before = full_attention_storage_breakdown(
        persistent,
        request,
        plan.full_attention_layer_indices,
        request_policy=config,
    )
    _validate_storage_accounting(
        full_attention_storage_before,
        allocated_source_pool_nbytes=conversion.allocated_block_pool_nbytes,
        request_audit=request_audit,
        request_policy=config,
        document_length=conversion.document_length,
        page_size=conversion.page_size,
        expected_appended_tokens=0,
    )
    try:
        generated = _generate_backend(
            model,
            backbone,
            tokenizer,
            request,
            backend,
            query,
            args.max_new_tokens,
        )
    finally:
        _unregister_backend(backend)
    _sync()
    request_after_generation = _allocator_snapshot("after-continuation-generation")
    total_peak = (
        request_after_generation["peak_allocated_bytes"]
        - request_before["current_allocated_bytes"]
    )
    total_peak_reserved = (
        request_after_generation["peak_reserved_bytes"]
        - request_before["current_reserved_bytes"]
    )
    total_current_allocated = (
        request_after_generation["current_allocated_bytes"]
        - request_before["current_allocated_bytes"]
    )
    total_current_reserved = (
        request_after_generation["current_reserved_bytes"]
        - request_before["current_reserved_bytes"]
    )
    memory_after_decode = storage_residency(persistent, request)
    partial_tail_copy_nbytes = sum(
        int(request.layers[index].sequence.partial_tail_staging_copy_nbytes)
        for index in plan.full_attention_layer_indices
    )
    full_attention_storage_after = full_attention_storage_breakdown(
        persistent,
        request,
        plan.full_attention_layer_indices,
        request_policy=config,
    )
    _validate_storage_accounting(
        full_attention_storage_after,
        allocated_source_pool_nbytes=conversion.allocated_block_pool_nbytes,
        request_audit=request_audit,
        request_policy=config,
        document_length=conversion.document_length,
        page_size=conversion.page_size,
        expected_appended_tokens=(
            int(query.shape[1]) + int(args.max_new_tokens) - 1
        ),
    )
    intercept = ledger.verify_complete()
    _validate_intercept(intercept, config, args.max_new_tokens, strict=False)
    full_copy = int(request_audit["full_document_staging_copy_nbytes"])
    if config == FRESH_CONTROL:
        _require(full_copy > 0, "fresh control did not copy the full document")
        _require(request_audit.get("source_document_storage_shared") is False, "fresh storage shared")
    else:
        _require(full_copy == 0, "reuse path copied the full document")
        _require(request_audit.get("source_document_storage_shared") is True, "reuse storage not shared")
    return {
        "config": config,
        "fair_protocol": FAIR_PROTOCOL,
        "kernel_mode": KERNEL_MODE,
        "kernel_identity": _kernel_identity(kernel),
        "same_unified_attention_kernel": True,
        "quantization": "Q16",
        "single_request_only": True,
        "batch_semantics": "batch-1-equal-length-only",
        "allocator_fresh_trial_baseline": trial_baseline,
        "document_build": build,
        "query_preparation": {
            "seconds": prepare_seconds,
            "cuda_peak_delta_bytes": prepare_peak,
            "cuda_peak_reserved_delta_bytes": prepare_peak_reserved,
            "cuda_current_allocated_delta_bytes": prepare_current_allocated,
            "cuda_current_reserved_delta_bytes": prepare_current_reserved,
            "allocator_before": request_before,
            "allocator_after": request_after_setup,
            "physical_document_block_copy_nbytes_including_padding": full_copy,
            "physical_document_block_copy_included_in_seconds": True,
            "fresh_pool_allocation_included_in_seconds": config == FRESH_CONTROL,
            "partial_document_tail_cow_is_common_append_path": True,
            "partial_document_tail_cow_included_in_request_setup": False,
            "partial_document_tail_cow_occurs_in_first_continuation_step": True,
            "audit": request_audit,
        },
        "memory_before_decode": memory_before_decode,
        "memory_after_decode": memory_after_decode,
        "continuation_append_accounting": {
            "partial_tail_staging_copy_nbytes": partial_tail_copy_nbytes,
            "phase": "first-continuation-model-step-common-append-path",
            "included_in_request_setup": False,
            "included_in_cached_document_request_ttft": True,
        },
        "full_attention_storage_before_decode": full_attention_storage_before,
        "full_attention_storage_after_decode": full_attention_storage_after,
        "linear_gdn_shared_base_at_request_start": gdn_base,
        "arm_difference_scope": "ten-full-attention-cache-ownership-only",
        "cuda_peak_request_delta_bytes": total_peak,
        "cuda_peak_request_reserved_delta_bytes": total_peak_reserved,
        "cuda_current_request_delta_bytes": total_current_allocated,
        "cuda_current_request_reserved_delta_bytes": total_current_reserved,
        "allocator_after_generation": request_after_generation,
        "cached_document_request_ttft_seconds": (
            prepare_seconds + generated["continuation_model_first_token_seconds"]
        ),
        "cached_document_request_ttft_excludes_common_document_build": True,
        **generated,
        "intercept": intercept,
        "conversion": {
            "document_length": conversion.document_length,
            "page_size": conversion.page_size,
            "dense_document_nbytes": conversion.dense_document_nbytes,
            "document_payload_nbytes": conversion.document_payload_nbytes,
            "allocated_source_pool_nbytes": conversion.allocated_block_pool_nbytes,
        },
    }


def _measure_hf_absolute_reference(
    args: argparse.Namespace,
    model: Any,
    backbone: Any,
    tokenizer: Any,
    plan: Any,
    document: torch.Tensor,
    query: torch.Tensor,
) -> dict[str, Any]:
    _sync()
    torch.cuda.reset_peak_memory_stats()
    before = _allocator_snapshot("before-hf-absolute-reference")
    started = time.perf_counter()
    persistent = _build_document_cache(backbone, document, functional=True)
    request = _fork_dense_functional(persistent, plan)
    _sync()
    prepare = time.perf_counter() - started
    generated = _generate_backend(
        model,
        backbone,
        tokenizer,
        request,
        "eager",
        query,
        args.max_new_tokens,
    )
    _sync()
    after = _allocator_snapshot("after-hf-absolute-reference")
    return {
        "comparison_role": "absolute-nonpaired-backend-reference",
        "included_in_primary_abba_ratio": False,
        "backend": "transformers-eager",
        "document_prefill_and_request_prepare_seconds": prepare,
        "raw_prompt_to_first_token_seconds": (
            prepare + generated["continuation_model_first_token_seconds"]
        ),
        "raw_prompt_timing_includes_document_prefill": True,
        "cuda_peak_request_delta_bytes": (
            after["peak_allocated_bytes"] - before["current_allocated_bytes"]
        ),
        "cuda_peak_request_reserved_delta_bytes": (
            after["peak_reserved_bytes"] - before["current_reserved_bytes"]
        ),
        "allocator_before": before,
        "allocator_after": after,
        "residency": storage_residency(persistent, request),
        **generated,
    }


def _record_trial(
    trials: dict[str, list[dict[str, Any]]],
    observed: dict[str, tuple[tuple[int, ...], tuple[str, ...]]],
    config: str,
    trial: dict[str, Any],
) -> None:
    signature = (
        tuple(trial["generated_token_ids"]),
        tuple(trial["full_vocab_step_logit_sha256"]),
    )
    if config in observed and observed[config] != signature:
        raise RuntimeError(f"{config} token/full-logit trajectory changed across trials")
    observed[config] = signature
    if len(observed) == 2 and len(set(observed.values())) != 1:
        raise RuntimeError(
            "same-kernel control/reuse full-logit trajectory diverged; stop paired timing"
        )
    trials[config].append(trial)


def _run_fresh_abba(
    args: argparse.Namespace,
    model: Any,
    backbone: Any,
    tokenizer: Any,
    plan: Any,
    document: torch.Tensor,
    query: torch.Tensor,
):
    pair = FAIR_CONFIGS
    warmup_order = pair if args.rank % 2 == 0 else tuple(reversed(pair))
    order = (
        (FRESH_CONTROL, SHARED_REUSE, SHARED_REUSE, FRESH_CONTROL)
        if args.rank % 2 == 0
        else (SHARED_REUSE, FRESH_CONTROL, FRESH_CONTROL, SHARED_REUSE)
    ) * 2
    for config in warmup_order:
        _fresh_allocator_cleanup(f"before-warmup-{config}")
        warm = _measure_same_kernel_config(
            args, model, backbone, tokenizer, plan, document, query, config
        )
        del warm
        _fresh_allocator_cleanup(f"after-warmup-{config}")
    frozen_baseline = _fresh_allocator_cleanup("post-warmup-frozen-abba-baseline")
    trials = {config: [] for config in pair}
    observed: dict[str, tuple[tuple[int, ...], tuple[str, ...]]] = {}
    cleanup_rows = []
    for trial_index, config in enumerate(order):
        before = _fresh_allocator_cleanup(
            f"before-measurement-{trial_index}-{config}"
        )
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(
                before[field] == frozen_baseline[field],
                f"fresh-state allocator baseline drifted for {field}",
            )
        trial = _measure_same_kernel_config(
            args, model, backbone, tokenizer, plan, document, query, config
        )
        trial["abba_allocator_baseline"] = before
        _record_trial(trials, observed, config, trial)
        del trial
        after = _fresh_allocator_cleanup(
            f"after-measurement-{trial_index}-{config}"
        )
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(
                after[field] == frozen_baseline[field],
                f"allocator did not return to frozen baseline for {field}",
            )
        cleanup_rows.append({"trial_index": trial_index, "config": config, "after": after})
    return warmup_order, order, trials, {
        "frozen_post_warmup_baseline": frozen_baseline,
        "cleanup_after_each_measurement": cleanup_rows,
        "gc_collect_before_empty_cache": True,
        "dynamic_attention_backends_unregistered": True,
        "baseline_exact_fields": ["current_allocated_bytes", "current_reserved_bytes"],
    }


def _median_trials(config: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    _require(len(trials) == 4, f"{config} requires four fresh ABBA trials")
    trajectory = (
        trials[0]["generated_token_ids"],
        trials[0]["full_vocab_step_logit_sha256"],
    )
    _require(
        all(
            (row["generated_token_ids"], row["full_vocab_step_logit_sha256"])
            == trajectory
            for row in trials
        ),
        f"{config} trajectory drift",
    )

    def invariant_projection(row: dict[str, Any]) -> dict[str, Any]:
        preparation = row["query_preparation"]
        audit = preparation["audit"]
        return {
            "config": row["config"],
            "kernel_identity": row["kernel_identity"],
            "same_unified_attention_kernel": row["same_unified_attention_kernel"],
            "arm_difference_scope": row["arm_difference_scope"],
            "document_dense_copy": row["document_build"]
            ["dense_to_nhd_document_copy_nbytes"],
            "physical_document_copy": preparation[
                "physical_document_block_copy_nbytes_including_padding"
            ],
            "request_pool_allocation": audit["allocated_request_pool_nbytes"],
            "source_document_storage_shared": audit[
                "source_document_storage_shared"
            ],
            "memory_before_decode": row["memory_before_decode"],
            "memory_after_decode": row["memory_after_decode"],
            "full_attention_storage_before_decode": row[
                "full_attention_storage_before_decode"
            ],
            "full_attention_storage_after_decode": row[
                "full_attention_storage_after_decode"
            ],
            "continuation_append_accounting": row[
                "continuation_append_accounting"
            ],
            "conversion": row["conversion"],
        }

    invariant = invariant_projection(trials[0])
    _require(
        all(invariant_projection(row) == invariant for row in trials),
        f"{config} non-timing storage/copy invariants drift across fresh trials",
    )
    representative = dict(trials[0])
    representative.update(
        {
            "fresh_trial_count": 4,
            "fresh_trials": trials,
            "cached_document_request_ttft_seconds": statistics.median(
                row["cached_document_request_ttft_seconds"] for row in trials
            ),
            "continuation_model_first_token_seconds": statistics.median(
                row["continuation_model_first_token_seconds"] for row in trials
            ),
            "median_tpot_seconds": statistics.median(
                row["median_tpot_seconds"] for row in trials
            ),
            "cuda_peak_request_delta_bytes": statistics.median(
                row["cuda_peak_request_delta_bytes"] for row in trials
            ),
            "cuda_peak_request_reserved_delta_bytes": statistics.median(
                row["cuda_peak_request_reserved_delta_bytes"] for row in trials
            ),
            "cuda_current_request_delta_bytes": statistics.median(
                row["cuda_current_request_delta_bytes"] for row in trials
            ),
            "cuda_current_request_reserved_delta_bytes": statistics.median(
                row["cuda_current_request_reserved_delta_bytes"] for row in trials
            ),
        }
    )
    representative["query_preparation"] = dict(trials[0]["query_preparation"])
    representative["query_preparation"].update(
        {
            "seconds": statistics.median(
                row["query_preparation"]["seconds"] for row in trials
            ),
            "cuda_peak_delta_bytes": statistics.median(
                row["query_preparation"]["cuda_peak_delta_bytes"] for row in trials
            ),
            "cuda_peak_reserved_delta_bytes": statistics.median(
                row["query_preparation"]["cuda_peak_reserved_delta_bytes"]
                for row in trials
            ),
            "cuda_current_allocated_delta_bytes": statistics.median(
                row["query_preparation"]["cuda_current_allocated_delta_bytes"]
                for row in trials
            ),
            "cuda_current_reserved_delta_bytes": statistics.median(
                row["query_preparation"]["cuda_current_reserved_delta_bytes"]
                for row in trials
            ),
        }
    )
    representative["document_build"] = dict(trials[0]["document_build"])
    for field in (
        "dense_document_prefill_seconds",
        "dense_document_prefill_cuda_peak_delta_bytes",
        "q16_pool_build_seconds",
        "q16_pool_build_cuda_peak_delta_bytes",
        "dense_document_prefill_cuda_peak_reserved_delta_bytes",
        "q16_pool_build_cuda_peak_reserved_delta_bytes",
        "dense_document_prefill_cuda_current_allocated_delta_bytes",
        "dense_document_prefill_cuda_current_reserved_delta_bytes",
        "q16_pool_build_cuda_current_allocated_delta_bytes",
        "q16_pool_build_cuda_current_reserved_delta_bytes",
    ):
        representative["document_build"][field] = statistics.median(
            row["document_build"][field] for row in trials
        )
    return representative


def _validate_raw_trial_derived_fields(
    trial: dict[str, Any], request_policy: str
) -> None:
    """Recompute every timing/copy/allocator scalar from its raw evidence."""

    build = trial["document_build"]
    preparation = trial["query_preparation"]
    audit = preparation["audit"]
    conversion = trial["conversion"]

    def snapshot_deltas(
        before: dict[str, Any], after: dict[str, Any]
    ) -> dict[str, int]:
        return {
            "current_allocated": int(after["current_allocated_bytes"])
            - int(before["current_allocated_bytes"]),
            "current_reserved": int(after["current_reserved_bytes"])
            - int(before["current_reserved_bytes"]),
            "peak_allocated": int(after["peak_allocated_bytes"])
            - int(before["current_allocated_bytes"]),
            "peak_reserved": int(after["peak_reserved_bytes"])
            - int(before["current_reserved_bytes"]),
        }

    prefill = snapshot_deltas(
        build["dense_document_prefill_allocator_before"],
        build["dense_document_prefill_allocator_after"],
    )
    pack = snapshot_deltas(
        build["q16_pool_build_allocator_before"],
        build["q16_pool_build_allocator_after"],
    )
    setup = snapshot_deltas(
        preparation["allocator_before"], preparation["allocator_after"]
    )
    total = snapshot_deltas(
        preparation["allocator_before"], trial["allocator_after_generation"]
    )
    checks = {
        "dense_document_prefill_cuda_current_allocated_delta_bytes": (
            build["dense_document_prefill_cuda_current_allocated_delta_bytes"],
            prefill["current_allocated"],
        ),
        "dense_document_prefill_cuda_current_reserved_delta_bytes": (
            build["dense_document_prefill_cuda_current_reserved_delta_bytes"],
            prefill["current_reserved"],
        ),
        "dense_document_prefill_cuda_peak_delta_bytes": (
            build["dense_document_prefill_cuda_peak_delta_bytes"],
            prefill["peak_allocated"],
        ),
        "dense_document_prefill_cuda_peak_reserved_delta_bytes": (
            build["dense_document_prefill_cuda_peak_reserved_delta_bytes"],
            prefill["peak_reserved"],
        ),
        "q16_pool_build_cuda_current_allocated_delta_bytes": (
            build["q16_pool_build_cuda_current_allocated_delta_bytes"],
            pack["current_allocated"],
        ),
        "q16_pool_build_cuda_current_reserved_delta_bytes": (
            build["q16_pool_build_cuda_current_reserved_delta_bytes"],
            pack["current_reserved"],
        ),
        "q16_pool_build_cuda_peak_delta_bytes": (
            build["q16_pool_build_cuda_peak_delta_bytes"],
            pack["peak_allocated"],
        ),
        "q16_pool_build_cuda_peak_reserved_delta_bytes": (
            build["q16_pool_build_cuda_peak_reserved_delta_bytes"],
            pack["peak_reserved"],
        ),
        "setup_cuda_current_allocated_delta_bytes": (
            preparation["cuda_current_allocated_delta_bytes"],
            setup["current_allocated"],
        ),
        "setup_cuda_current_reserved_delta_bytes": (
            preparation["cuda_current_reserved_delta_bytes"],
            setup["current_reserved"],
        ),
        "setup_cuda_peak_delta_bytes": (
            preparation["cuda_peak_delta_bytes"], setup["peak_allocated"]
        ),
        "setup_cuda_peak_reserved_delta_bytes": (
            preparation["cuda_peak_reserved_delta_bytes"],
            setup["peak_reserved"],
        ),
        "total_cuda_current_allocated_delta_bytes": (
            trial["cuda_current_request_delta_bytes"],
            total["current_allocated"],
        ),
        "total_cuda_current_reserved_delta_bytes": (
            trial["cuda_current_request_reserved_delta_bytes"],
            total["current_reserved"],
        ),
        "total_cuda_peak_delta_bytes": (
            trial["cuda_peak_request_delta_bytes"], total["peak_allocated"]
        ),
        "total_cuda_peak_reserved_delta_bytes": (
            trial["cuda_peak_request_reserved_delta_bytes"],
            total["peak_reserved"],
        ),
    }
    for label, (reported, recomputed) in checks.items():
        _require(
            int(reported) == int(recomputed),
            f"raw-trial derived allocator scalar {label} drift",
        )
    _require(
        int(preparation["physical_document_block_copy_nbytes_including_padding"])
        == int(audit["full_document_staging_copy_nbytes"]),
        "raw-trial physical copy differs from request audit",
    )
    dense_copy = _required_nonnegative_int(
        build.get("dense_to_nhd_document_copy_nbytes"),
        "raw-trial dense-to-NHD document copy bytes",
    )
    dense_conversion = _required_nonnegative_int(
        conversion.get("dense_document_nbytes"),
        "raw-trial conversion dense document bytes",
    )
    payload_build = _required_nonnegative_int(
        build.get("q16_document_payload_nbytes"),
        "raw-trial build Q16 document payload bytes",
    )
    payload_conversion = _required_nonnegative_int(
        conversion.get("document_payload_nbytes"),
        "raw-trial conversion document payload bytes",
    )
    source_pool_build = _required_nonnegative_int(
        build.get("q16_allocated_source_pool_nbytes"),
        "raw-trial build allocated source pool bytes",
    )
    source_pool_conversion = _required_nonnegative_int(
        conversion.get("allocated_source_pool_nbytes"),
        "raw-trial conversion allocated source pool bytes",
    )
    conversion_page_size = _required_positive_int(
        conversion.get("page_size"), "raw-trial conversion page size"
    )
    _require(
        dense_copy == dense_conversion == payload_build == payload_conversion,
        "raw-trial dense copy/Q16 valid payload/conversion bytes drift",
    )
    _require(
        source_pool_build == source_pool_conversion,
        "raw-trial build/conversion source-pool bytes drift",
    )
    _require(
        build.get("full_attention_layers") == 10,
        "raw-trial build full-attention layer count drift",
    )
    for phase in (
        "full_attention_storage_before_decode",
        "full_attention_storage_after_decode",
    ):
        totals = trial[phase]["totals"]
        _require(
            totals.get("valid_document_payload_nbytes") == payload_conversion
            and totals.get("source_total_arena_allocated_nbytes")
            == source_pool_conversion,
            f"raw-trial {phase} payload/source-pool differs from conversion",
        )
        expected_physical_copy = (
            totals.get("fresh_duplicate_document_allocated_nbytes")
            if request_policy == FRESH_CONTROL
            else 0
        )
        _require(
            int(audit["full_document_staging_copy_nbytes"])
            == expected_physical_copy,
            f"raw-trial {phase} physical copy differs from storage ledger",
        )
    after_storage = trial["full_attention_storage_after_decode"]
    recomputed_partial_tail_copy = sum(
        _required_nonnegative_int(
            layer.get("active_request_detached_tail_tokens"),
            f"raw-trial storage layer {layer.get('layer_idx')} detached tail",
        )
        * _required_positive_int(
            layer.get("block_bytes"),
            f"raw-trial storage layer {layer.get('layer_idx')} block bytes",
        )
        // conversion_page_size
        for layer in after_storage["layers"]
    )
    reported_partial_tail_copy = _required_nonnegative_int(
        trial.get("continuation_append_accounting", {}).get(
            "partial_tail_staging_copy_nbytes"
        ),
        "raw-trial partial-tail staging copy bytes",
    )
    _require(
        reported_partial_tail_copy == recomputed_partial_tail_copy,
        "raw-trial partial-tail staging copy differs from storage replay",
    )
    if request_policy == FRESH_CONTROL:
        _require(
            int(audit["allocated_request_pool_nbytes"]) > 0
            and preparation["fresh_pool_allocation_included_in_seconds"] is True,
            "fresh raw trial omitted request-pool materialization",
        )
    else:
        _require(
            int(audit.get("allocated_request_pool_nbytes", 0)) == 0
            and preparation["fresh_pool_allocation_included_in_seconds"] is False,
            "reuse raw trial reports a fresh request pool",
        )
    recomputed_ttft = float(preparation["seconds"]) + float(
        trial["continuation_model_first_token_seconds"]
    )
    _require(
        math.isclose(
            float(trial["cached_document_request_ttft_seconds"]),
            recomputed_ttft,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ),
        "raw-trial cached-document TTFT is not setup plus first model step",
    )


@torch.inference_mode()
def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    # Authorization is read and verified before validation is hashed or opened.
    authorization = _load_frozen_json(
        args.authorization,
        args.expected_authorization_sha256,
        "PG19 authorization",
    )
    _validate_authorization(authorization)
    current_identity = _static_frozen_identity(args.static_audit)
    _require(
        authorization.get("frozen_identity") == current_identity,
        "authorization was produced by different code/model/data/protocol inputs",
    )
    _require(
        authorization.get("protocol_config") == args.static_audit.get("protocol_config"),
        "authorization protocol config differs from validation process",
    )
    _require(args.expected_validation_sha256 != TEST_V2_SHA256, "test-v2 digest refused")
    _require(
        sha256_file(args.validation_data) == args.expected_validation_sha256,
        "validation SHA mismatch",
    )
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.data = args.validation_data
    args.exclude_source_indices = (4, 5)
    args.allow_test_v2 = False
    args.context_lengths = ()
    args.synthetic_repetitions = 0
    workloads, metadata = longbench_workloads(tokenizer, args)
    _require(
        metadata.get("data_sha256") == args.expected_validation_sha256,
        "validation loader reopened bytes with a different SHA256",
    )
    observed_pairs = {
        (str(row["dataset"]), int(row["source_index"])) for row in workloads
    }
    _require(len(workloads) == 8, "validation workload count drift")
    _require(observed_pairs == EXPECTED_VALIDATION_PAIRS, "validation source6-9 isolation failed")
    _require(len({str(row["workload_id"]) for row in workloads}) == 8, "workload IDs repeat")
    _require(metadata.get("test_v2_consumed") is False, "test-v2 was consumed")
    _require(metadata.get("source_revisions") == [args.expected_source_revision], "revision drift")
    _require(metadata.get("datasets") == ["2wikimqa", "qasper"], "dataset drift")
    assigned = workloads[args.rank :: args.world_size]
    _require(len(assigned) == 1, "formal validation requires one workload per rank")
    torch.cuda.set_device(0)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    _require(tuple(plan.full_attention_layer_indices) == FULL_LAYERS, "layer geometry drift")
    workload = assigned[0]
    document = _tokens(workload["document_tokens"], args.max_input_tokens).cuda()
    query = _tokens(workload["query_tokens"], args.max_query_tokens).cuda()
    hf_reference = _measure_hf_absolute_reference(
        args, model, backbone, tokenizer, plan, document, query
    )
    torch.cuda.empty_cache()
    warmup_order, order, trials, allocator_fresh_state = _run_fresh_abba(
        args, model, backbone, tokenizer, plan, document, query
    )
    measurements = {
        config: _median_trials(config, trials[config]) for config in FAIR_CONFIGS
    }
    fresh = measurements[FRESH_CONTROL]
    reuse = measurements[SHARED_REUSE]
    _require(
        fresh["generated_token_ids"] == reuse["generated_token_ids"]
        and fresh["full_vocab_step_logit_sha256"]
        == reuse["full_vocab_step_logit_sha256"],
        "same-kernel ABBA trajectories differ",
    )
    _require(
        fresh["kernel_identity"] == reuse["kernel_identity"],
        "ABBA arms used different unified_attention identities",
    )
    return {
        "status": "completed_fair_v2_validation_shard",
        "rank": args.rank,
        "world_size": args.world_size,
        "fair_protocol": FAIR_PROTOCOL,
        "kernel_mode": KERNEL_MODE,
        "quantization": "Q16",
        "single_request_only": True,
        "batch_semantics": "batch-1-equal-length-only",
        "authorization_sha256": args.expected_authorization_sha256,
        "workload_metadata": metadata,
        "workload": {
            "workload_id": workload["workload_id"],
            "dataset": workload["dataset"],
            "source_index": int(workload["source_index"]),
            "document_tokens": int(document.shape[1]),
            "query_tokens": int(query.shape[1]),
        },
        "hf_eager_absolute_reference": hf_reference,
        "warmup_order": warmup_order,
        "measurement_order": order,
        "measurement_protocol": "same-kernel-fresh-state-ABBAx2-four-trials-per-arm",
        "allocator_fresh_state": allocator_fresh_state,
        "warmup_runs_per_config": 1,
        "fresh_measurement_runs_per_config": 4,
        "measurements": measurements,
        "primary_pair": {
            "full_vocab_step_logits_bitwise_exact": True,
            "generated_tokens_exact": True,
            "cached_document_request_ttft_ratio_reuse_vs_full_copy": (
                _safe_ratio(
                    reuse["cached_document_request_ttft_seconds"],
                    fresh["cached_document_request_ttft_seconds"],
                )
            ),
            "continuation_model_first_token_ratio_reuse_vs_full_copy": (
                _safe_ratio(
                    reuse["continuation_model_first_token_seconds"],
                    fresh["continuation_model_first_token_seconds"],
                )
            ),
            "tpot_ratio_reuse_vs_full_copy": (
                _safe_ratio(
                    reuse["median_tpot_seconds"], fresh["median_tpot_seconds"]
                )
            ),
            "cuda_peak_ratio_reuse_vs_full_copy": (
                _safe_ratio(
                    reuse["cuda_peak_request_delta_bytes"],
                    fresh["cuda_peak_request_delta_bytes"],
                )
            ),
            "cuda_peak_reserved_ratio_reuse_vs_full_copy": (
                _safe_ratio(
                    reuse["cuda_peak_request_reserved_delta_bytes"],
                    fresh["cuda_peak_request_reserved_delta_bytes"],
                )
            ),
            "setup_peak_allocated_ratio_reuse_vs_full_copy": (
                _safe_ratio(
                    reuse["query_preparation"]["cuda_peak_delta_bytes"],
                    fresh["query_preparation"]["cuda_peak_delta_bytes"],
                )
            ),
            "setup_peak_reserved_ratio_reuse_vs_full_copy": (
                _safe_ratio(
                    reuse["query_preparation"]["cuda_peak_reserved_delta_bytes"],
                    fresh["query_preparation"]["cuda_peak_reserved_delta_bytes"],
                )
            ),
            "physical_document_block_copy_bytes_saved_including_padding": (
                fresh["query_preparation"]
                ["physical_document_block_copy_nbytes_including_padding"]
                - reuse["query_preparation"]
                ["physical_document_block_copy_nbytes_including_padding"]
            ),
            "cached_document_request_ttft_excludes_common_document_build": True,
            "isolated_kernel_latency_measured": False,
        },
        "backend_compatibility_nonblocking": {
            "is_primary_performance_pair": False,
            "hf_generated_tokens_match_vllm": (
                hf_reference["generated_token_ids"] == reuse["generated_token_ids"]
            ),
            "hf_full_vocab_step_logit_sha_match_vllm": (
                hf_reference["full_vocab_step_logit_sha256"]
                == reuse["full_vocab_step_logit_sha256"]
            ),
        },
        "validation_consumed_after_pg19_authorization": True,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
        "multi_query_serving_completed": False,
    }


def summarize_validation_shards(
    run_dir: Path,
    *,
    authorization_path: Path,
    expected_authorization_sha256: str,
    expected_frozen_identity: dict[str, str],
    expected_code_ledger_sha256: str,
    expected_model_manifest_sha256: str,
    expected_model_artifact_ledger_sha256: str,
    expected_model_weight_ledger_sha256: str,
    expected_source_revision: str,
    expected_calls_per_layer: int,
) -> dict[str, Any]:
    authorization = _load_frozen_json(
        authorization_path,
        expected_authorization_sha256,
        "PG19 authorization",
    )
    _validate_authorization(authorization)
    frozen_identity = authorization["frozen_identity"]
    _require(
        frozen_identity == expected_frozen_identity,
        "PG19 authorization belongs to a different current frozen snapshot",
    )
    _require(frozen_identity["code_ledger_sha256"] == expected_code_ledger_sha256, "auth code ledger drift")
    _require(frozen_identity["model_manifest_sha256"] == expected_model_manifest_sha256, "auth model manifest drift")
    _require(
        frozen_identity["model_artifact_ledger_sha256"]
        == expected_model_artifact_ledger_sha256,
        "auth model artifact ledger drift",
    )
    _require(
        frozen_identity["model_weight_ledger_sha256"]
        == expected_model_weight_ledger_sha256,
        "auth model weight ledger drift",
    )
    shard_paths = sorted((run_dir / "validation-shards").glob("fair-v2-shard-*.json"))
    _require(len(shard_paths) == 8, "validation summary requires eight shards")
    shards = [json.loads(path.read_text()) for path in shard_paths]
    _require(sorted(int(row["rank"]) for row in shards) == list(range(8)), "rank drift")
    pairs = {
        (str(row["workload"]["dataset"]), int(row["workload"]["source_index"]))
        for row in shards
    }
    _require(pairs == EXPECTED_VALIDATION_PAIRS, "summary source6-9 isolation failed")
    request_ratios = []
    model_first_token_ratios = []
    tpot_ratios = []
    peak_ratios = []
    peak_reserved_ratios = []
    setup_peak_ratios = []
    setup_peak_reserved_ratios = []
    copy_saved = []
    fresh_copy_bytes = []
    reuse_copy_bytes = []
    fresh_combined = []
    reuse_combined = []
    fresh_combined_after = []
    reuse_combined_after = []
    valid_document_payload = []
    source_document_allocated = []
    source_document_padding = []
    source_private_reservation = []
    source_total_arena_allocated = []
    fresh_duplicate_document = []
    fresh_duplicate_document_padding = []
    fresh_private_reservation = []
    fresh_active_private_payload = []
    reuse_active_private_payload = []
    fresh_active_private_blocks = []
    reuse_active_private_blocks = []
    fresh_active_private_allocated = []
    reuse_active_private_allocated = []
    fresh_private_reserved_unused = []
    reuse_private_reserved_unused = []
    source_document_table_metadata = []
    source_cpu_reservation_metadata = []
    fresh_cpu_reservation_metadata = []
    fresh_block_table_metadata = []
    reuse_block_table_metadata = []
    fresh_partial_tail_copy = []
    reuse_partial_tail_copy = []
    fresh_setup_peak_allocated = []
    reuse_setup_peak_allocated = []
    fresh_setup_peak_reserved = []
    reuse_setup_peak_reserved = []
    fresh_total_peak_allocated = []
    reuse_total_peak_allocated = []
    fresh_total_peak_reserved = []
    reuse_total_peak_reserved = []
    common_prefill_seconds = []
    common_pack_seconds = []
    common_prefill_peak_allocated = []
    common_prefill_peak_reserved = []
    common_pack_peak_allocated = []
    common_pack_peak_reserved = []
    hf_token_agreement = []
    cross_rank_kernel_descriptors: set[tuple[str, str, str]] = set()
    allocator_absolute_samples: dict[str, dict[str, list[int]]] = {
        phase: {
            field: []
            for field in (
                "current_allocated_bytes",
                "current_reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "current_allocated_delta_bytes",
                "current_reserved_delta_bytes",
                "peak_allocated_delta_bytes",
                "peak_reserved_delta_bytes",
            )
        }
        for phase in (
            "common_dense_document_prefill_after",
            "common_q16_pack_after",
            "fresh_request_setup_after",
            "reuse_request_setup_after",
            "fresh_setup_plus_first_step_after",
            "reuse_setup_plus_first_step_after",
            "fresh_setup_plus_generation_after",
            "reuse_setup_plus_generation_after",
        )
    }

    def record_allocator(
        phase: str,
        before: dict[str, Any],
        after: dict[str, Any],
    ) -> None:
        _require(
            isinstance(before, dict) and isinstance(after, dict),
            f"{phase} allocator snapshots must be dicts",
        )
        before_values = {
            field: _required_nonnegative_int(
                before.get(field), f"{phase} before allocator {field}"
            )
            for field in (
                "current_allocated_bytes",
                "current_reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        }
        after_values = {
            field: _required_nonnegative_int(
                after.get(field), f"{phase} after allocator {field}"
            )
            for field in (
                "current_allocated_bytes",
                "current_reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        }
        values = {
            "current_allocated_bytes": after_values["current_allocated_bytes"],
            "current_reserved_bytes": after_values["current_reserved_bytes"],
            "peak_allocated_bytes": after_values["peak_allocated_bytes"],
            "peak_reserved_bytes": after_values["peak_reserved_bytes"],
            "current_allocated_delta_bytes": (
                after_values["current_allocated_bytes"]
                - before_values["current_allocated_bytes"]
            ),
            "current_reserved_delta_bytes": (
                after_values["current_reserved_bytes"]
                - before_values["current_reserved_bytes"]
            ),
            "peak_allocated_delta_bytes": (
                after_values["peak_allocated_bytes"]
                - before_values["current_allocated_bytes"]
            ),
            "peak_reserved_delta_bytes": (
                after_values["peak_reserved_bytes"]
                - before_values["current_reserved_bytes"]
            ),
        }
        for field, samples in allocator_absolute_samples[phase].items():
            samples.append(values[field])

    def summarize_ratio(samples: list[float]) -> dict[str, Any]:
        return {
            "defined_count": len(samples),
            "undefined_zero_baseline_count": 8 - len(samples),
            "median_when_defined": statistics.median(samples) if samples else None,
        }

    for shard in shards:
        _require(shard.get("status") == "completed_fair_v2_validation_shard", "shard failed")
        _require(shard.get("fair_protocol") == FAIR_PROTOCOL, "protocol drift")
        _require(shard.get("kernel_mode") == KERNEL_MODE, "kernel drift")
        _require(shard.get("quantization") == "Q16", "non-Q16 shard")
        _require(shard.get("single_request_only") is True, "serving scope drift")
        _require(
            _static_frozen_identity(shard.get("static", {})) == frozen_identity,
            "validation shard static identity differs from PG19 authorization",
        )
        _require(shard.get("authorization_sha256") == expected_authorization_sha256, "auth drift")
        _require(
            shard.get("validation_consumed_after_pg19_authorization") is True
            and shard.get("source_68_99_consumed") is False
            and shard.get("test_v2_consumed") is False,
            "evaluation governance drift",
        )
        metadata = shard["workload_metadata"]
        _require(metadata.get("test_v2_consumed") is False, "metadata says test-v2")
        _require(
            metadata.get("data_sha256")
            == frozen_identity[
                "validation_expected_sha256_recorded_but_not_hashed"
            ],
            "validation shard data SHA differs from PG19 authorization",
        )
        _require(metadata.get("source_revisions") == [expected_source_revision], "source revision drift")
        _require(shard.get("warmup_runs_per_config") == 1, "warmup count drift")
        _require(shard.get("fresh_measurement_runs_per_config") == 4, "trial count drift")
        expected_warmup_order = (
            (FRESH_CONTROL, SHARED_REUSE)
            if int(shard["rank"]) % 2 == 0
            else (SHARED_REUSE, FRESH_CONTROL)
        )
        _require(
            tuple(shard.get("warmup_order", ())) == expected_warmup_order,
            "warmup order drift",
        )
        allocator = shard.get("allocator_fresh_state", {})
        frozen_allocator = allocator.get("frozen_post_warmup_baseline", {})
        cleanup_rows = allocator.get("cleanup_after_each_measurement", ())
        _require(
            allocator.get("gc_collect_before_empty_cache") is True
            and allocator.get("dynamic_attention_backends_unregistered") is True
            and allocator.get("baseline_exact_fields")
            == ["current_allocated_bytes", "current_reserved_bytes"]
            and len(cleanup_rows) == 8,
            "fresh-state allocator governance drift",
        )
        _require(
            isinstance(frozen_allocator, dict),
            "frozen allocator baseline must be a dict",
        )
        frozen_allocator_values = {
            field: _required_nonnegative_int(
                frozen_allocator.get(field), f"frozen allocator baseline {field}"
            )
            for field in ("current_allocated_bytes", "current_reserved_bytes")
        }
        expected_order = (
            (FRESH_CONTROL, SHARED_REUSE, SHARED_REUSE, FRESH_CONTROL)
            if int(shard["rank"]) % 2 == 0
            else (SHARED_REUSE, FRESH_CONTROL, FRESH_CONTROL, SHARED_REUSE)
        ) * 2
        _require(tuple(shard["measurement_order"]) == expected_order, "ABBA order drift")
        for trial_index, (cleanup, config) in enumerate(
            zip(cleanup_rows, expected_order)
        ):
            _require(
                int(cleanup.get("trial_index", -1)) == trial_index
                and cleanup.get("config") == config,
                "allocator cleanup row does not match ABBA order",
            )
            for field in ("current_allocated_bytes", "current_reserved_bytes"):
                cleanup_after = cleanup.get("after")
                _require(
                    isinstance(cleanup_after, dict),
                    "cleanup allocator after snapshot must be a dict",
                )
                cleanup_value = _required_nonnegative_int(
                    cleanup_after.get(field), f"cleanup allocator {field}"
                )
                _require(
                    cleanup_value == frozen_allocator_values[field],
                    f"cleanup allocator {field} differs from frozen baseline",
                )
        reported_measurements = shard["measurements"]
        _require(set(reported_measurements) == set(FAIR_CONFIGS), "primary arms drift")
        reported_fresh = reported_measurements[FRESH_CONTROL]
        reported_reuse = reported_measurements[SHARED_REUSE]
        _require(
            len(reported_fresh["fresh_trials"])
            == len(reported_reuse["fresh_trials"])
            == 4,
            "fresh trials drift",
        )
        shard_raw_kernel_identities = []
        for config, measurement in (
            (FRESH_CONTROL, reported_fresh),
            (SHARED_REUSE, reported_reuse),
        ):
            for trial in measurement["fresh_trials"]:
                _require(isinstance(trial, dict), "raw trial must be a dict")
                _validate_raw_trial_derived_fields(trial, config)
                _validate_intercept(
                    trial["intercept"], config, expected_calls_per_layer, strict=False
                )
                appended_by_layer = {
                    layer: sum(
                        call["current_append_delta_tokens"]
                        for call in trial["intercept"]["calls"]
                        if call["layer_idx"] == layer
                    )
                    for layer in FULL_LAYERS
                }
                _require(
                    len(set(appended_by_layer.values())) == 1,
                    "raw-trial appended-token totals differ across layers",
                )
                total_appended_tokens = appended_by_layer[FULL_LAYERS[0]]
                trial_kernel_identity = _validate_kernel_identity_schema(
                    trial.get("kernel_identity"), "raw-trial top-level"
                )
                _require(
                    trial_kernel_identity
                    == trial["intercept"].get("kernel_identity"),
                    "raw-trial top/intercept kernel identity differs",
                )
                shard_raw_kernel_identities.append(trial_kernel_identity)
                _require(trial.get("same_unified_attention_kernel") is True, "trial kernel flag absent")
                _require(
                    trial.get("arm_difference_scope")
                    == "ten-full-attention-cache-ownership-only",
                    "arm scope drift",
                )
                gdn = trial.get("linear_gdn_shared_base_at_request_start", {})
                _require(
                    gdn.get("passed") is True
                    and gdn.get("linear_layer_count") == 30
                    and gdn.get("persistent_tensor_base_shared_at_request_start") is True
                    and gdn.get("request_updates_are_functional_rebind") is True,
                    "timed GDN base contract drift",
                )
                for baseline_key in (
                    "allocator_fresh_trial_baseline",
                    "abba_allocator_baseline",
                ):
                    baseline = trial.get(baseline_key)
                    _require(
                        isinstance(baseline, dict),
                        f"raw-trial {baseline_key} must be a dict",
                    )
                    for field in (
                        "current_allocated_bytes",
                        "current_reserved_bytes",
                    ):
                        baseline_value = _required_nonnegative_int(
                            baseline.get(field),
                            f"raw-trial {baseline_key} {field}",
                        )
                        _require(
                            baseline_value == frozen_allocator_values[field],
                            f"raw-trial {baseline_key} {field} drift",
                        )
                tokens = trial.get("generated_token_ids")
                logit_hashes = trial.get("full_vocab_step_logit_sha256")
                _require(
                    isinstance(tokens, list)
                    and isinstance(logit_hashes, list)
                    and len(tokens) == len(logit_hashes) == expected_calls_per_layer,
                    "raw-trial trajectory cardinality drift",
                )
                for token_index, token in enumerate(tokens):
                    _required_nonnegative_int(
                        token, f"raw-trial generated token {token_index}"
                    )
                for step_index, digest in enumerate(logit_hashes):
                    _validate_sha256_hex(
                        digest, f"raw-trial full-vocab logit step {step_index}"
                    )
                record_allocator(
                    "common_dense_document_prefill_after",
                    trial["document_build"]
                    ["dense_document_prefill_allocator_before"],
                    trial["document_build"]
                    ["dense_document_prefill_allocator_after"],
                )
                record_allocator(
                    "common_q16_pack_after",
                    trial["document_build"]["q16_pool_build_allocator_before"],
                    trial["document_build"]["q16_pool_build_allocator_after"],
                )
                prefix = "fresh" if config == FRESH_CONTROL else "reuse"
                record_allocator(
                    f"{prefix}_request_setup_after",
                    trial["query_preparation"]["allocator_before"],
                    trial["query_preparation"]["allocator_after"],
                )
                record_allocator(
                    f"{prefix}_setup_plus_first_step_after",
                    trial["query_preparation"]["allocator_before"],
                    trial.get("allocator_after_first_continuation_step"),
                )
                record_allocator(
                    f"{prefix}_setup_plus_generation_after",
                    trial["query_preparation"]["allocator_before"],
                    trial["allocator_after_generation"],
                )
                for phase in (
                    "full_attention_storage_before_decode",
                    "full_attention_storage_after_decode",
                ):
                    trial_storage = trial[phase]
                    _require(
                        trial_storage.get("request_policy") == config
                        and trial_storage.get("full_attention_layer_count") == 10,
                        "raw-trial storage policy/layer count drift",
                    )
                    _validate_storage_accounting(
                        trial_storage,
                        allocated_source_pool_nbytes=trial["conversion"]
                        ["allocated_source_pool_nbytes"],
                        request_audit=trial["query_preparation"]["audit"],
                        request_policy=config,
                        document_length=trial["conversion"]["document_length"],
                        page_size=trial["conversion"]["page_size"],
                        expected_appended_tokens=(
                            0
                            if phase == "full_attention_storage_before_decode"
                            else total_appended_tokens
                        ),
                    )
                _require(
                    trial["query_preparation"].get(
                        "partial_document_tail_cow_included_in_request_setup"
                    )
                    is False
                    and trial["query_preparation"].get(
                        "partial_document_tail_cow_occurs_in_first_continuation_step"
                    )
                    is True,
                    "raw-trial partial-tail phase drift",
                )
                trial_append = trial.get("continuation_append_accounting", {})
                _require(
                    trial_append.get("phase")
                    == "first-continuation-model-step-common-append-path"
                    and trial_append.get("included_in_request_setup") is False
                    and trial_append.get(
                        "included_in_cached_document_request_ttft"
                    )
                    is True,
                    "raw-trial continuation append accounting drift",
                )
        _require(
            len(shard_raw_kernel_identities) == 8
            and all(
                identity == shard_raw_kernel_identities[0]
                for identity in shard_raw_kernel_identities
            ),
            "raw trials within one rank used different kernel identities",
        )
        descriptor = shard_raw_kernel_identities[0]
        cross_rank_kernel_descriptors.add(
            (
                str(descriptor.get("module")),
                str(descriptor.get("qualname")),
                str(descriptor.get("signature")),
            )
        )
        # Every reported top-level median/representative field is untrusted.
        # Replay the four raw trials per arm and use only the replayed values
        # for parity, ratio, copy, storage and allocator aggregation.
        replayed_measurements = {
            config: _median_trials(
                config, reported_measurements[config]["fresh_trials"]
            )
            for config in FAIR_CONFIGS
        }
        for config in FAIR_CONFIGS:
            _require(
                reported_measurements[config] == replayed_measurements[config],
                f"reported {config} measurement differs from raw-trial replay",
            )
        fresh = replayed_measurements[FRESH_CONTROL]
        reuse = replayed_measurements[SHARED_REUSE]
        _require(fresh["kernel_identity"] == reuse["kernel_identity"], "arm kernel identity differs")
        _require(
            fresh["generated_token_ids"] == reuse["generated_token_ids"]
            and fresh["full_vocab_step_logit_sha256"]
            == reuse["full_vocab_step_logit_sha256"],
            "full-logit trajectory parity failed",
        )
        hf_reference = shard.get("hf_eager_absolute_reference")
        _require(
            isinstance(hf_reference, dict),
            "HF eager absolute-reference artifact is missing",
        )
        hf_tokens = hf_reference.get("generated_token_ids")
        hf_logit_hashes = hf_reference.get("full_vocab_step_logit_sha256")
        _require(
            isinstance(hf_tokens, list)
            and isinstance(hf_logit_hashes, list)
            and len(hf_tokens) == len(hf_logit_hashes) == expected_calls_per_layer,
            "HF eager diagnostic trajectory cardinality drift",
        )
        for token_index, token in enumerate(hf_tokens):
            _required_nonnegative_int(
                token, f"HF eager generated token {token_index}"
            )
        for step_index, digest in enumerate(hf_logit_hashes):
            _validate_sha256_hex(
                digest, f"HF eager full-vocab logit step {step_index}"
            )
        recomputed_backend_compatibility = {
            "is_primary_performance_pair": False,
            "hf_generated_tokens_match_vllm": (
                hf_tokens == reuse["generated_token_ids"]
            ),
            "hf_full_vocab_step_logit_sha_match_vllm": (
                hf_logit_hashes == reuse["full_vocab_step_logit_sha256"]
            ),
        }
        _require(
            shard.get("backend_compatibility_nonblocking")
            == recomputed_backend_compatibility,
            "reported HF/vLLM backend compatibility differs from raw trajectories",
        )
        hf_token_agreement.append(
            recomputed_backend_compatibility["hf_generated_tokens_match_vllm"]
        )
        fresh_copy = int(
            fresh["query_preparation"]
            ["physical_document_block_copy_nbytes_including_padding"]
        )
        reuse_copy = int(
            reuse["query_preparation"]
            ["physical_document_block_copy_nbytes_including_padding"]
        )
        _require(fresh_copy > 0 and reuse_copy == 0, "full-copy/reuse byte contract drift")
        _require(
            fresh["query_preparation"]
            ["physical_document_block_copy_included_in_seconds"]
            is True
            and fresh["query_preparation"]["fresh_pool_allocation_included_in_seconds"] is True,
            "fresh materialization cost excluded",
        )
        for measurement in (fresh, reuse):
            _require(
                measurement["query_preparation"].get(
                    "partial_document_tail_cow_included_in_request_setup"
                )
                is False
                and measurement["query_preparation"].get(
                    "partial_document_tail_cow_occurs_in_first_continuation_step"
                )
                is True,
                "partial-tail COW phase accounting drift",
            )
            append_accounting = measurement.get(
                "continuation_append_accounting", {}
            )
            _require(
                append_accounting.get("phase")
                == "first-continuation-model-step-common-append-path"
                and append_accounting.get("included_in_request_setup") is False
                and append_accounting.get("included_in_cached_document_request_ttft")
                is True,
                "continuation append accounting drift",
            )
        _require(
            fresh["document_build"]["dense_to_nhd_document_copy_nbytes"]
            == reuse["document_build"]["dense_to_nhd_document_copy_nbytes"],
            "common document-build accounting differs",
        )
        primary = shard["primary_pair"]
        _require(
            primary.get("full_vocab_step_logits_bitwise_exact") is True
            and primary.get("generated_tokens_exact") is True,
            "primary pair parity flags failed",
        )
        _require(
            primary.get("cached_document_request_ttft_excludes_common_document_build") is True
            and primary.get("isolated_kernel_latency_measured") is False,
            "TTFT/kernel timing scope drift",
        )
        recomputed_ratios = {
            "cached_document_request_ttft_ratio_reuse_vs_full_copy": _safe_ratio(
                reuse["cached_document_request_ttft_seconds"],
                fresh["cached_document_request_ttft_seconds"],
            ),
            "continuation_model_first_token_ratio_reuse_vs_full_copy": _safe_ratio(
                reuse["continuation_model_first_token_seconds"],
                fresh["continuation_model_first_token_seconds"],
            ),
            "tpot_ratio_reuse_vs_full_copy": _safe_ratio(
                reuse["median_tpot_seconds"], fresh["median_tpot_seconds"]
            ),
            "cuda_peak_ratio_reuse_vs_full_copy": _safe_ratio(
                reuse["cuda_peak_request_delta_bytes"],
                fresh["cuda_peak_request_delta_bytes"],
            ),
            "cuda_peak_reserved_ratio_reuse_vs_full_copy": _safe_ratio(
                reuse["cuda_peak_request_reserved_delta_bytes"],
                fresh["cuda_peak_request_reserved_delta_bytes"],
            ),
            "setup_peak_allocated_ratio_reuse_vs_full_copy": _safe_ratio(
                reuse["query_preparation"]["cuda_peak_delta_bytes"],
                fresh["query_preparation"]["cuda_peak_delta_bytes"],
            ),
            "setup_peak_reserved_ratio_reuse_vs_full_copy": _safe_ratio(
                reuse["query_preparation"]["cuda_peak_reserved_delta_bytes"],
                fresh["query_preparation"]["cuda_peak_reserved_delta_bytes"],
            ),
        }
        for key, recomputed in recomputed_ratios.items():
            reported = primary.get(key)
            _require(
                (reported is None and recomputed is None)
                or (
                    reported is not None
                    and recomputed is not None
                    and math.isclose(
                        float(reported),
                        float(recomputed),
                        rel_tol=1e-12,
                        abs_tol=1e-15,
                    )
                ),
                f"primary ratio {key} differs from raw measurements",
            )
        for destination, key in (
            (
                request_ratios,
                "cached_document_request_ttft_ratio_reuse_vs_full_copy",
            ),
            (
                model_first_token_ratios,
                "continuation_model_first_token_ratio_reuse_vs_full_copy",
            ),
            (tpot_ratios, "tpot_ratio_reuse_vs_full_copy"),
            (peak_ratios, "cuda_peak_ratio_reuse_vs_full_copy"),
            (peak_reserved_ratios, "cuda_peak_reserved_ratio_reuse_vs_full_copy"),
            (setup_peak_ratios, "setup_peak_allocated_ratio_reuse_vs_full_copy"),
            (
                setup_peak_reserved_ratios,
                "setup_peak_reserved_ratio_reuse_vs_full_copy",
            ),
        ):
            value = recomputed_ratios[key]
            if value is not None:
                destination.append(float(value))
        copy_saved.append(
            int(
                primary[
                    "physical_document_block_copy_bytes_saved_including_padding"
                ]
            )
        )
        recomputed_copy_saved = fresh_copy - reuse_copy
        _require(
            int(
                primary[
                    "physical_document_block_copy_bytes_saved_including_padding"
                ]
            )
            == recomputed_copy_saved,
            "primary physical-copy saving differs from raw measurements",
        )
        copy_saved[-1] = recomputed_copy_saved
        fresh_tail = int(
            fresh["continuation_append_accounting"]
            ["partial_tail_staging_copy_nbytes"]
        )
        reuse_tail = int(
            reuse["continuation_append_accounting"]
            ["partial_tail_staging_copy_nbytes"]
        )
        _require(
            fresh_tail == reuse_tail,
            "common first-step partial-tail copy differs between arms",
        )
        fresh_partial_tail_copy.append(fresh_tail)
        reuse_partial_tail_copy.append(reuse_tail)
        fresh_copy_bytes.append(fresh_copy)
        reuse_copy_bytes.append(reuse_copy)
        fresh_combined.append(
            int(fresh["memory_before_decode"]["combined_unique_accelerator_nbytes"])
        )
        reuse_combined.append(
            int(reuse["memory_before_decode"]["combined_unique_accelerator_nbytes"])
        )
        fresh_combined_after.append(
            int(fresh["memory_after_decode"]["combined_unique_accelerator_nbytes"])
        )
        reuse_combined_after.append(
            int(reuse["memory_after_decode"]["combined_unique_accelerator_nbytes"])
        )
        fresh_setup_peak_allocated.append(
            fresh["query_preparation"]["cuda_peak_delta_bytes"]
        )
        reuse_setup_peak_allocated.append(
            reuse["query_preparation"]["cuda_peak_delta_bytes"]
        )
        fresh_setup_peak_reserved.append(
            fresh["query_preparation"]["cuda_peak_reserved_delta_bytes"]
        )
        reuse_setup_peak_reserved.append(
            reuse["query_preparation"]["cuda_peak_reserved_delta_bytes"]
        )
        fresh_total_peak_allocated.append(fresh["cuda_peak_request_delta_bytes"])
        reuse_total_peak_allocated.append(reuse["cuda_peak_request_delta_bytes"])
        fresh_total_peak_reserved.append(
            fresh["cuda_peak_request_reserved_delta_bytes"]
        )
        reuse_total_peak_reserved.append(
            reuse["cuda_peak_request_reserved_delta_bytes"]
        )
        for measurement in (fresh, reuse):
            build = measurement["document_build"]
            common_prefill_seconds.append(build["dense_document_prefill_seconds"])
            common_pack_seconds.append(build["q16_pool_build_seconds"])
            common_prefill_peak_allocated.append(
                build["dense_document_prefill_cuda_peak_delta_bytes"]
            )
            common_prefill_peak_reserved.append(
                build["dense_document_prefill_cuda_peak_reserved_delta_bytes"]
            )
            common_pack_peak_allocated.append(
                build["q16_pool_build_cuda_peak_delta_bytes"]
            )
            common_pack_peak_reserved.append(
                build["q16_pool_build_cuda_peak_reserved_delta_bytes"]
            )
        fresh_storage = fresh["full_attention_storage_after_decode"]
        reuse_storage = reuse["full_attention_storage_after_decode"]
        for measurement, policy in (
            (fresh, FRESH_CONTROL),
            (reuse, SHARED_REUSE),
        ):
            representative_appended_by_layer = {
                layer: sum(
                    call["current_append_delta_tokens"]
                    for call in measurement["intercept"]["calls"]
                    if call["layer_idx"] == layer
                )
                for layer in FULL_LAYERS
            }
            _require(
                len(set(representative_appended_by_layer.values())) == 1,
                "representative appended-token totals differ across layers",
            )
            representative_appended = representative_appended_by_layer[
                FULL_LAYERS[0]
            ]
            for phase in (
                "full_attention_storage_before_decode",
                "full_attention_storage_after_decode",
            ):
                _validate_storage_accounting(
                    measurement[phase],
                    allocated_source_pool_nbytes=measurement["conversion"]
                    ["allocated_source_pool_nbytes"],
                    request_audit=measurement["query_preparation"]["audit"],
                    request_policy=policy,
                    document_length=measurement["conversion"]["document_length"],
                    page_size=measurement["conversion"]["page_size"],
                    expected_appended_tokens=(
                        0
                        if phase == "full_attention_storage_before_decode"
                        else representative_appended
                    ),
                )
        for storage, policy in (
            (fresh_storage, FRESH_CONTROL),
            (reuse_storage, SHARED_REUSE),
        ):
            _require(storage.get("request_policy") == policy, "storage policy drift")
            _require(storage.get("full_attention_layer_count") == 10, "storage layer count drift")
            _require(
                storage.get("source_arena_includes_preallocated_private_reservation") is True,
                "source private capacity hidden",
            )
        fresh_totals = fresh_storage["totals"]
        reuse_totals = reuse_storage["totals"]
        for field in (
            "valid_document_payload_nbytes",
            "source_document_allocated_nbytes",
            "source_document_padding_nbytes",
            "source_private_reservation_nbytes",
            "source_document_table_accelerator_nbytes",
            "source_cpu_reservation_metadata_nbytes",
        ):
            _require(fresh_totals[field] == reuse_totals[field], f"common storage {field} drift")
        _require(
            fresh_totals["fresh_duplicate_document_allocated_nbytes"] > 0
            and fresh_totals["fresh_private_reservation_nbytes"] > 0
            and reuse_totals["fresh_duplicate_document_allocated_nbytes"] == 0
            and reuse_totals["fresh_private_reservation_nbytes"] == 0,
            "fresh duplicate pool ledger drift",
        )
        valid_document_payload.append(fresh_totals["valid_document_payload_nbytes"])
        source_document_allocated.append(
            fresh_totals["source_document_allocated_nbytes"]
        )
        source_document_padding.append(fresh_totals["source_document_padding_nbytes"])
        source_private_reservation.append(fresh_totals["source_private_reservation_nbytes"])
        source_total_arena_allocated.append(
            fresh_totals["source_total_arena_allocated_nbytes"]
        )
        source_document_table_metadata.append(
            fresh_totals["source_document_table_accelerator_nbytes"]
        )
        fresh_duplicate_document.append(
            fresh_totals["fresh_duplicate_document_allocated_nbytes"]
        )
        fresh_duplicate_document_padding.append(
            fresh_totals["fresh_duplicate_document_padding_nbytes"]
        )
        fresh_private_reservation.append(fresh_totals["fresh_private_reservation_nbytes"])
        source_cpu_reservation_metadata.append(
            fresh_totals["source_cpu_reservation_metadata_nbytes"]
        )
        fresh_cpu_reservation_metadata.append(
            fresh_totals["fresh_cpu_reservation_metadata_nbytes"]
        )
        fresh_active_private_payload.append(
            fresh_totals["active_request_private_payload_nbytes"]
        )
        reuse_active_private_payload.append(
            reuse_totals["active_request_private_payload_nbytes"]
        )
        fresh_active_private_blocks.append(
            fresh_totals["active_request_private_blocks"]
        )
        reuse_active_private_blocks.append(
            reuse_totals["active_request_private_blocks"]
        )
        fresh_active_private_allocated.append(
            fresh_totals["active_request_private_allocated_page_nbytes"]
        )
        reuse_active_private_allocated.append(
            reuse_totals["active_request_private_allocated_page_nbytes"]
        )
        fresh_private_reserved_unused.append(
            fresh_totals["request_private_reserved_unused_nbytes"]
        )
        reuse_private_reserved_unused.append(
            reuse_totals["request_private_reserved_unused_nbytes"]
        )
        fresh_block_table_metadata.append(
            fresh_totals["request_block_table_accelerator_nbytes"]
            + fresh_totals["fresh_document_table_accelerator_nbytes"]
        )
        reuse_block_table_metadata.append(
            reuse_totals["request_block_table_accelerator_nbytes"]
        )
    _require(
        len(cross_rank_kernel_descriptors) == 1,
        "validation ranks used different kernel descriptors",
    )
    return {
        "status": "completed_fair_v2_summary",
        "fair_protocol": FAIR_PROTOCOL,
        "kernel_mode": KERNEL_MODE,
        "quantization": "Q16",
        "primary_comparison": "same-kernel-full-copy-control-vs-shared-document-reuse",
        "primary_full_logit_parity_fraction": 1.0,
        "workloads": 8,
        "source_indices": [6, 7, 8, 9],
        "datasets": ["2wikimqa", "qasper"],
        "fresh_measurement_runs_per_config": 4,
        "cached_document_request_ttft_ratio_reuse_vs_full_copy": summarize_ratio(
            request_ratios
        ),
        "continuation_model_first_token_ratio_reuse_vs_full_copy": summarize_ratio(
            model_first_token_ratios
        ),
        "isolated_kernel_latency_measured": False,
        "tpot_ratio_reuse_vs_full_copy": summarize_ratio(tpot_ratios),
        "cuda_peak_allocated_ratio_reuse_vs_full_copy": {
            "defined_count": len(peak_ratios),
            "undefined_zero_baseline_count": 8 - len(peak_ratios),
            "median_when_defined": (
                statistics.median(peak_ratios) if peak_ratios else None
            ),
        },
        "cuda_peak_reserved_ratio_reuse_vs_full_copy": {
            "defined_count": len(peak_reserved_ratios),
            "undefined_zero_baseline_count": 8 - len(peak_reserved_ratios),
            "median_when_defined": (
                statistics.median(peak_reserved_ratios)
                if peak_reserved_ratios
                else None
            ),
        },
        "setup_peak_allocated_ratio_reuse_vs_full_copy": {
            "defined_count": len(setup_peak_ratios),
            "undefined_zero_baseline_count": 8 - len(setup_peak_ratios),
            "median_when_defined": (
                statistics.median(setup_peak_ratios)
                if setup_peak_ratios
                else None
            ),
        },
        "setup_peak_reserved_ratio_reuse_vs_full_copy": {
            "defined_count": len(setup_peak_reserved_ratios),
            "undefined_zero_baseline_count": 8
            - len(setup_peak_reserved_ratios),
            "median_when_defined": (
                statistics.median(setup_peak_reserved_ratios)
                if setup_peak_reserved_ratios
                else None
            ),
        },
        "median_physical_document_block_copy_bytes_saved_including_padding": statistics.median(copy_saved),
        "median_fresh_physical_document_block_copy_nbytes_including_padding": statistics.median(fresh_copy_bytes),
        "median_reuse_physical_document_block_copy_nbytes_including_padding": statistics.median(reuse_copy_bytes),
        "median_fresh_combined_unique_accelerator_nbytes_before_continuation": statistics.median(fresh_combined),
        "median_reuse_combined_unique_accelerator_nbytes_before_continuation": statistics.median(reuse_combined),
        "median_fresh_combined_unique_accelerator_nbytes_after_decode": statistics.median(
            fresh_combined_after
        ),
        "median_reuse_combined_unique_accelerator_nbytes_after_decode": statistics.median(
            reuse_combined_after
        ),
        "full_attention_storage_medians": {
            "valid_document_payload_nbytes": statistics.median(
                valid_document_payload
            ),
            "source_document_allocated_nbytes": statistics.median(
                source_document_allocated
            ),
            "source_document_padding_nbytes": statistics.median(
                source_document_padding
            ),
            "source_preallocated_private_reservation_nbytes": statistics.median(
                source_private_reservation
            ),
            "source_total_arena_allocated_nbytes": statistics.median(
                source_total_arena_allocated
            ),
            "fresh_duplicate_document_allocation_nbytes": statistics.median(
                fresh_duplicate_document
            ),
            "fresh_duplicate_document_padding_nbytes": statistics.median(
                fresh_duplicate_document_padding
            ),
            "fresh_duplicate_private_reservation_nbytes": statistics.median(
                fresh_private_reservation
            ),
            "fresh_active_tail_and_append_payload_nbytes": statistics.median(
                fresh_active_private_payload
            ),
            "reuse_active_tail_and_append_payload_nbytes": statistics.median(
                reuse_active_private_payload
            ),
            "fresh_active_private_blocks": statistics.median(
                fresh_active_private_blocks
            ),
            "reuse_active_private_blocks": statistics.median(
                reuse_active_private_blocks
            ),
            "fresh_active_private_allocated_page_nbytes": statistics.median(
                fresh_active_private_allocated
            ),
            "reuse_active_private_allocated_page_nbytes": statistics.median(
                reuse_active_private_allocated
            ),
            "fresh_private_reserved_unused_nbytes": statistics.median(
                fresh_private_reserved_unused
            ),
            "reuse_private_reserved_unused_nbytes": statistics.median(
                reuse_private_reserved_unused
            ),
            "common_source_document_table_accelerator_nbytes": statistics.median(
                source_document_table_metadata
            ),
            "common_source_cpu_reservation_metadata_nbytes": statistics.median(
                source_cpu_reservation_metadata
            ),
            "fresh_cpu_reservation_metadata_nbytes": statistics.median(
                fresh_cpu_reservation_metadata
            ),
            "fresh_block_table_metadata_accelerator_nbytes": statistics.median(
                fresh_block_table_metadata
            ),
            "reuse_block_table_metadata_accelerator_nbytes": statistics.median(
                reuse_block_table_metadata
            ),
            "fresh_first_step_partial_tail_staging_copy_nbytes": statistics.median(
                fresh_partial_tail_copy
            ),
            "reuse_first_step_partial_tail_staging_copy_nbytes": statistics.median(
                reuse_partial_tail_copy
            ),
            "partial_tail_copy_phase": "first-continuation-model-step-common-append-path",
        },
        "allocator_phase_medians": {
            "common_dense_document_prefill_seconds": statistics.median(
                common_prefill_seconds
            ),
            "common_q16_pack_seconds": statistics.median(common_pack_seconds),
            "common_prefill_peak_allocated_delta_bytes": statistics.median(
                common_prefill_peak_allocated
            ),
            "common_prefill_peak_reserved_delta_bytes": statistics.median(
                common_prefill_peak_reserved
            ),
            "common_pack_peak_allocated_delta_bytes": statistics.median(
                common_pack_peak_allocated
            ),
            "common_pack_peak_reserved_delta_bytes": statistics.median(
                common_pack_peak_reserved
            ),
            "fresh_setup_peak_allocated_delta_bytes": statistics.median(
                fresh_setup_peak_allocated
            ),
            "reuse_setup_peak_allocated_delta_bytes": statistics.median(
                reuse_setup_peak_allocated
            ),
            "fresh_setup_peak_reserved_delta_bytes": statistics.median(
                fresh_setup_peak_reserved
            ),
            "reuse_setup_peak_reserved_delta_bytes": statistics.median(
                reuse_setup_peak_reserved
            ),
            "fresh_setup_plus_generation_peak_allocated_delta_bytes": statistics.median(
                fresh_total_peak_allocated
            ),
            "reuse_setup_plus_generation_peak_allocated_delta_bytes": statistics.median(
                reuse_total_peak_allocated
            ),
            "fresh_setup_plus_generation_peak_reserved_delta_bytes": statistics.median(
                fresh_total_peak_reserved
            ),
            "reuse_setup_plus_generation_peak_reserved_delta_bytes": statistics.median(
                reuse_total_peak_reserved
            ),
        },
        "allocator_current_peak_absolute_and_delta_medians": {
            phase: {
                "sample_count": len(next(iter(fields.values()))),
                **{
                    field: statistics.median(samples)
                    for field, samples in fields.items()
                },
            }
            for phase, fields in allocator_absolute_samples.items()
        },
        "combined_unique_storage_is_diagnostic_not_pure_document_storage": True,
        "hf_eager_absolute_reference_only": True,
        "hf_generated_token_agreement_fraction": sum(hf_token_agreement) / len(hf_token_agreement),
        "backend_compatibility_used_for_primary_speedup": False,
        "single_request_only": True,
        "ragged_batch_claimed": False,
        "multi_query_serving_completed": False,
        "validation_consumed_after_pg19_authorization": True,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
        "authorization_sha256": expected_authorization_sha256,
        "code_ledger_sha256": expected_code_ledger_sha256,
        "model_manifest_sha256": expected_model_manifest_sha256,
        "model_artifact_ledger_sha256": expected_model_artifact_ledger_sha256,
        "model_weight_ledger_sha256": expected_model_weight_ledger_sha256,
        "protocol_manifest_sha256": frozen_identity["protocol_manifest_sha256"],
        "protocol_config": authorization["protocol_config"],
        "protocol_config_sha256": frozen_identity["protocol_config_sha256"],
        "claim_boundaries_zh": [
            "主性能结论仅比较同一vLLM unified_attention kernel的完整复制control与文档复用路径。",
            "HF eager只作绝对后端兼容性诊断，不进入主speedup分母。",
            "当前仅验证Q16、单请求、batch=1等长输入；不声称多query或ragged batch。",
            "cached-document request TTFT只含request setup与continuation首步，不含公共document prefill/Q16 pack。",
            "未单独测量unified_attention kernel latency；continuation首步是完整模型步，不能称kernel speedup。",
            "source arena含预分配private reservation；combined unique字节不能表述为纯document存储。",
            "PG19 train门禁通过后才读取validation source 6--9；未读取68--99或test-v2。",
        ],
    }


def _validate_static(args: argparse.Namespace) -> dict[str, Any]:
    # Q8/Q4 must fail before any file hash, model config or package inspection.
    _require(args.bits == 16, "fair v2 fused path supports Q16 only")
    _require(args.world_size == 8, "formal fair v2 requires eight ranks")
    _require(0 <= args.rank < args.world_size, "rank is outside world size")
    _require(args.page_size == 128, "formal page size is frozen at 128")
    _require(args.source_index_start == 6 and args.source_index_end == 9, "validation must be source6-9")
    _require(args.limit_per_dataset == 4, "validation requires four examples per dataset")
    _require(args.min_input_tokens == 1, "validation minimum input token filter drift")
    _require(
        len(args.expected_source_revision) == 40
        and all(ch in "0123456789abcdef" for ch in args.expected_source_revision),
        "expected source revision must be one frozen git SHA1",
    )
    _require(args.max_new_tokens >= 2, "decode needs TTFT and TPOT")
    formal_values = {
        "pg19_books": 8,
        "pg19_document_tokens": 1025,
        "pg19_query_tokens": 32,
        "pg19_window_stride": 257,
        "pg19_candidate_windows": 8,
        "pg19_seed": 20260814,
        "max_input_tokens": 4096,
        "max_query_tokens": 64,
        "max_new_tokens": 8,
    }
    for field, expected in formal_values.items():
        _require(
            int(getattr(args, field)) == expected,
            f"formal v2 freezes {field}={expected}",
        )
    _require(args.expected_validation_sha256 != TEST_V2_SHA256, "test-v2 digest refused")
    _require("test-v2" not in str(args.validation_data).lower(), "test-v2 path refused")
    _require("longbench" not in str(args.pg19_data).lower(), "PG19 gate path is not train-only")
    _require(args.pg19_document_tokens % args.page_size != 0, "PG19 must exercise partial tail")
    _require(0 < args.pg19_query_tokens <= args.max_query_tokens, "query limit drift")
    for value in (
        args.expected_pg19_sha256,
        args.expected_pg19_manifest_sha256,
        args.expected_pg19_windows_sha256,
        args.expected_validation_sha256,
        args.expected_model_manifest_sha256,
        args.expected_code_ledger_sha256,
        args.expected_model_artifact_ledger_sha256,
        args.expected_model_weight_ledger_sha256,
        args.expected_protocol_manifest_sha256,
    ):
        _require(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), "invalid SHA256")
    for path, expected, label in (
        (args.code_ledger, args.expected_code_ledger_sha256, "code"),
        (
            args.model_artifact_ledger,
            args.expected_model_artifact_ledger_sha256,
            "model artifact",
        ),
        (
            args.model_weight_ledger,
            args.expected_model_weight_ledger_sha256,
            "model weight",
        ),
    ):
        _require(path is not None and path.is_file(), f"{label} ledger missing")
        _require(sha256_file(path) == expected, f"{label} ledger SHA mismatch")
    _require(
        args.protocol_manifest is not None and args.protocol_manifest.is_file(),
        "runtime protocol manifest missing",
    )
    protocol_manifest_sha256 = args.expected_protocol_manifest_sha256
    # Static preflight intentionally does not hash/open validation.
    _require(sha256_file(args.pg19_data) == args.expected_pg19_sha256, "PG19 data SHA mismatch")
    _require(
        sha256_file(args.pg19_manifest) == args.expected_pg19_manifest_sha256,
        "PG19 manifest SHA mismatch",
    )
    manifest_sha, manifest_rows = _model_manifest_sha(args.model)
    _require(manifest_sha == args.expected_model_manifest_sha256, "model manifest SHA mismatch")
    config = json.loads((args.model / "config.json").read_text())
    text = config.get("text_config", config)
    layer_types = list(text.get("layer_types", ()))
    full = tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention")
    geometry = {
        "num_hidden_layers": int(text.get("num_hidden_layers", -1)),
        "num_query_heads": int(text.get("num_attention_heads", -1)),
        "num_key_value_heads": int(text.get("num_key_value_heads", -1)),
        "head_dim": int(text.get("head_dim", -1)),
        "full_attention_layer_indices": list(full),
    }
    _require(geometry["num_hidden_layers"] == 40, "model layer count drift")
    _require(full == FULL_LAYERS, "full-attention layer indices drift")
    for key in ("num_query_heads", "num_key_value_heads", "head_dim"):
        _require(geometry[key] == QWEN35_AUDITED_GEOMETRY[key], f"{key} drift")
    environment = audit_frozen_kernel_environment()
    _require(environment["matches_frozen_environment"], f"environment drift: {environment['mismatches']}")
    protocol_config = _protocol_config(args)
    protocol_config_sha256 = _protocol_config_sha256(protocol_config)
    expected_protocol_manifest = {
        "schema_version": 1,
        "fair_protocol": FAIR_PROTOCOL,
        "quantization": "Q16",
        "code_ledger_sha256": args.expected_code_ledger_sha256,
        "model_manifest_sha256": args.expected_model_manifest_sha256,
        "model_artifact_ledger_sha256": (
            args.expected_model_artifact_ledger_sha256
        ),
        "model_weight_ledger_sha256": args.expected_model_weight_ledger_sha256,
        "pg19_data_sha256": args.expected_pg19_sha256,
        "pg19_manifest_sha256": args.expected_pg19_manifest_sha256,
        "pg19_windows_sha256": args.expected_pg19_windows_sha256,
        "validation_expected_sha256_recorded_but_not_hashed": (
            args.expected_validation_sha256
        ),
        "protocol_config": protocol_config,
    }
    protocol_manifest = _load_frozen_json(
        args.protocol_manifest,
        args.expected_protocol_manifest_sha256,
        "runtime protocol manifest",
    )
    _require(
        protocol_manifest == expected_protocol_manifest,
        "runtime protocol manifest fields differ from effective formal CLI",
    )
    return {
        "status": "fair_v2_static_dry_run_passed",
        "fair_protocol": FAIR_PROTOCOL,
        "quantization": "Q16",
        "single_request_only": True,
        "batch_semantics": "batch-1-equal-length-only",
        "page_size": args.page_size,
        "geometry": geometry,
        "model_manifest_sha256": manifest_sha,
        "model_manifest": manifest_rows,
        "environment": environment,
        "pg19_data_sha256": args.expected_pg19_sha256,
        "pg19_manifest_sha256": args.expected_pg19_manifest_sha256,
        "pg19_windows_sha256": args.expected_pg19_windows_sha256,
        "code_ledger_sha256": args.expected_code_ledger_sha256,
        "model_artifact_ledger_sha256": args.expected_model_artifact_ledger_sha256,
        "model_weight_ledger_sha256": args.expected_model_weight_ledger_sha256,
        "protocol_manifest_sha256": protocol_manifest_sha256,
        "protocol_manifest": protocol_manifest,
        "protocol_config": protocol_config,
        "protocol_config_sha256": protocol_config_sha256,
        "validation_expected_sha256_recorded_but_not_hashed": args.expected_validation_sha256,
        "validation_consumed": False,
        "validation_hashed": False,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
        "gpu_initialized": torch.cuda.is_initialized(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=(
            "static-dry-run",
            "pg19-gate",
            "aggregate-pg19",
            "validation",
            "aggregate-validation",
        ),
        required=True,
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pg19-data", type=Path, required=True)
    parser.add_argument("--pg19-manifest", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--expected-pg19-sha256", required=True)
    parser.add_argument("--expected-pg19-manifest-sha256", required=True)
    parser.add_argument("--expected-pg19-windows-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--expected-authorization-sha256", default="0" * 64)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--expected-protocol-manifest-sha256", required=True)
    parser.add_argument("--code-ledger", type=Path)
    parser.add_argument("--model-artifact-ledger", type=Path)
    parser.add_argument("--model-weight-ledger", type=Path)
    parser.add_argument("--expected-code-ledger-sha256", default="0" * 64)
    parser.add_argument("--expected-model-artifact-ledger-sha256", default="0" * 64)
    parser.add_argument("--expected-model-weight-ledger-sha256", default="0" * 64)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-query-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--source-index-start", type=int, default=6)
    parser.add_argument("--source-index-end", type=int, default=9)
    parser.add_argument("--pg19-books", type=int, default=8)
    parser.add_argument("--pg19-document-tokens", type=int, default=1025)
    parser.add_argument("--pg19-query-tokens", type=int, default=32)
    parser.add_argument("--pg19-window-stride", type=int, default=257)
    parser.add_argument("--pg19-candidate-windows", type=int, default=8)
    parser.add_argument("--pg19-seed", type=int, default=20260814)
    # Compatibility fields consumed by longbench_workloads.
    parser.add_argument("--min-input-tokens", type=int, default=1)
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    static = _validate_static(args)
    args.static_audit = static
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "static-dry-run":
        result = static
    elif args.stage == "pg19-gate":
        result = {"static": static, **run_pg19_gate(args)}
    elif args.stage == "aggregate-pg19":
        paths = sorted((args.run_dir / "pg19-gate-shards").glob("pg19-fair-v2-shard-*.json"))
        result = aggregate_pg19_gate_shards(
            paths,
            expected_windows_sha256=args.expected_pg19_windows_sha256,
            expected_frozen_identity=_static_frozen_identity(static),
        )
    elif args.stage == "validation":
        _require(args.authorization is not None, "validation requires authorization")
        result = {"static": static, **run_validation(args)}
    else:
        _require(args.authorization is not None, "summary requires authorization")
        for path, expected, label in (
            (args.code_ledger, args.expected_code_ledger_sha256, "code"),
            (
                args.model_artifact_ledger,
                args.expected_model_artifact_ledger_sha256,
                "model artifact",
            ),
            (
                args.model_weight_ledger,
                args.expected_model_weight_ledger_sha256,
                "model weight",
            ),
        ):
            _require(path is not None and sha256_file(path) == expected, f"{label} ledger drift")
        result = summarize_validation_shards(
            args.run_dir,
            authorization_path=args.authorization,
            expected_authorization_sha256=args.expected_authorization_sha256,
            expected_frozen_identity=_static_frozen_identity(static),
            expected_code_ledger_sha256=args.expected_code_ledger_sha256,
            expected_model_manifest_sha256=args.expected_model_manifest_sha256,
            expected_model_artifact_ledger_sha256=(
                args.expected_model_artifact_ledger_sha256
            ),
            expected_model_weight_ledger_sha256=args.expected_model_weight_ledger_sha256,
            expected_source_revision=args.expected_source_revision,
            expected_calls_per_layer=args.max_new_tokens,
        )
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
