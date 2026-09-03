from __future__ import annotations

"""Formal Q16 fused-kernel gate and source-6--9 deployment benchmark."""

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
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
    KERNEL_MODE,
    POST_ROPE_POSITION_IDS_CONTRACT,
    Qwen35VllmPagedHitLedger,
    convert_all_qwen35_full_layers_to_vllm_q16,
    fork_qwen35_vllm_q16_request,
    full_vocab_forward_kl,
    register_qwen35_vllm_q16_backend,
    validate_qwen35_post_rope_position_ids,
)
from qcomem_torch import cache_nbytes, clone_cache
from qcomem_vllm_paged_kernel import (
    Q16KernelPagedLayer,
    audit_frozen_kernel_environment,
    vllm_triton_q16_paged_attention_forward,
)
from run_deployment_bench import longbench_workloads
from run_downstream import atomic_json


def _sync() -> None:
    torch.cuda.synchronize()


def _resolve_backbone(model: Any) -> Any:
    if hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model.model, "layers"):
        return model.model
    raise RuntimeError("cannot resolve Qwen3.5 text backbone")


def _tokens(value: torch.Tensor, limit: int) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise RuntimeError("token tensor must be rank two")
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

    def seed(value: Any):
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


def _eager_qwen_attention(module, query, key, value, attention_mask, *, scaling, dropout=0.0):
    if dropout != 0.0:
        raise RuntimeError("formal inference requires dropout=0")
    batch, kv_heads, length, dim = key.shape
    groups = module.num_key_value_groups
    key = key[:, :, None].expand(batch, kv_heads, groups, length, dim).reshape(
        batch, kv_heads * groups, length, dim
    )
    value = value[:, :, None].expand(batch, kv_heads, groups, length, dim).reshape(
        batch, kv_heads * groups, length, dim
    )
    if isinstance(attention_mask, dict):
        attention_mask = attention_mask["full_attention"]
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous(), weights


class IsolatedKernelGate:
    """Runs eager and fused kernels on identical post-RoPE Q/full-K/V/mask."""

    def __init__(self, indices: tuple[int, ...], page_size: int, rtol: float, atol: float):
        self.indices = indices
        self.page_size = page_size
        self.rtol = rtol
        self.atol = atol
        self.counts = {index: 0 for index in indices}
        self.rows: list[dict[str, Any]] = []

    def attention_forward(self, module, query, key, value, attention_mask, *args, **kwargs):
        del args
        index = module.layer_idx
        if index not in self.counts or self.counts[index]:
            raise RuntimeError(f"isolated gate unexpected/repeated layer {index}")
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
        eager, weights = _eager_qwen_attention(
            module, query, key, value, attention_mask, scaling=scaling, dropout=dropout
        )
        query_length = int(query.shape[-2])
        prefix_length = int(key.shape[-2]) - query_length
        if prefix_length < 1:
            raise RuntimeError("isolated gate has no document prefix")
        layer = Q16KernelPagedLayer.from_dense_document(
            key[..., :prefix_length, :],
            value[..., :prefix_length, :],
            page_size=self.page_size,
            max_append_tokens=query_length,
        )
        key_view, value_view = layer.update(
            key[..., prefix_length:, :], value[..., prefix_length:, :]
        )
        audit: dict[str, Any] = {}
        candidate, _ = vllm_triton_q16_paged_attention_forward(
            module,
            query,
            key_view,
            value_view,
            attention_mask,
            scaling=scaling,
            audit=audit,
        )
        audit.update(position_audit)
        error = candidate.float() - eager.float()
        denominator = eager.float().square().sum().clamp_min(1e-30).sqrt()
        close = torch.allclose(candidate, eager, rtol=self.rtol, atol=self.atol)
        finite = bool(torch.isfinite(candidate).all())
        row = {
            "layer_idx": index,
            "close": bool(close),
            "finite": finite,
            "max_abs": float(error.abs().max().item()),
            "mean_abs": float(error.abs().mean().item()),
            "relative_l2": float(error.square().sum().sqrt().div(denominator).item()),
            "rtol": self.rtol,
            "atol": self.atol,
            "output_shape": tuple(candidate.shape),
            "output_contiguous": candidate.is_contiguous(),
            "audit": audit,
        }
        self.rows.append(row)
        self.counts[index] += 1
        if not close or not finite:
            raise RuntimeError(f"isolated fused parity failed at layer {index}: {row}")
        return eager, weights

    def verify(self):
        if self.counts != {index: 1 for index in self.indices}:
            raise RuntimeError(f"isolated layer coverage failed: {self.counts}")
        return {
            "passed": True,
            "layer_indices": self.indices,
            "layer_count": len(self.indices),
            "dense_fallback_calls": 0,
            "rows": self.rows,
        }


def _register_isolated_backend(gate: IsolatedKernelGate) -> str:
    from transformers.masking_utils import AttentionMaskInterface, eager_mask
    from transformers.modeling_utils import AttentionInterface

    name = f"qcomem_isolated_vllm_q16_{uuid.uuid4().hex}"
    AttentionInterface.register(name, gate.attention_forward)
    AttentionMaskInterface.register(name, eager_mask)
    return name


def _last_logits(model: Any, output: Any) -> torch.Tensor:
    logits = model.lm_head(output.last_hidden_state[:, -1, :]).float()
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise RuntimeError("invalid final logits")
    return logits


@torch.inference_mode()
def run_pg19_gate(args: argparse.Namespace) -> dict[str, Any]:
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
    if windows_sha != args.expected_pg19_windows_sha256:
        raise RuntimeError(f"PG19 windows SHA mismatch: {windows_sha}")
    torch.cuda.set_device(0)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    if len(plan.full_attention_layer_indices) != 10:
        raise RuntimeError("formal gate requires config-derived 10 full layers")
    assigned = list(enumerate(windows))[args.rank :: args.world_size]
    if len(assigned) != 1:
        raise RuntimeError("formal 8-rank PG19 gate requires one window per rank")
    isolated_rows = []
    semantic_rows = []
    original = backbone.config._attn_implementation
    for window_index, window in assigned:
        document = window.document_ids.unsqueeze(0).cuda()
        query = window.query_ids.unsqueeze(0).cuda()
        stock = _build_document_cache(backbone, document, functional=False)
        baseline = clone_cache(stock)
        isolated = IsolatedKernelGate(
            plan.full_attention_layer_indices, args.page_size, args.isolated_rtol, args.isolated_atol
        )
        isolated_name = _register_isolated_backend(isolated)
        try:
            backbone.config._attn_implementation = isolated_name
            backbone(input_ids=query, past_key_values=stock, use_cache=True)
        finally:
            backbone.config._attn_implementation = original
        isolated_rows.append(
            {"window_index": window_index, "source_object": window.source_object, **isolated.verify()}
        )
        try:
            backbone.config._attn_implementation = "eager"
            baseline_output = backbone(
                input_ids=query, past_key_values=baseline, use_cache=True
            )
        finally:
            backbone.config._attn_implementation = original
        baseline_logits = _last_logits(model, baseline_output)
        persistent = _build_document_cache(backbone, document, functional=True)
        conversion = convert_all_qwen35_full_layers_to_vllm_q16(
            persistent,
            plan,
            page_size=args.page_size,
            max_append_tokens=int(query.shape[1]),
            max_request_forks=1,
        )
        request, fork_audit = fork_qwen35_vllm_q16_request(persistent, plan)
        ledger = Qwen35VllmPagedHitLedger(plan, conversion)
        backend = register_qwen35_vllm_q16_backend(ledger)
        try:
            backbone.config._attn_implementation = backend.name
            candidate_output = backbone(
                input_ids=query, past_key_values=request, use_cache=True
            )
        finally:
            backbone.config._attn_implementation = original
        candidate_logits = _last_logits(model, candidate_output)
        kl = float(full_vocab_forward_kl(baseline_logits, candidate_logits).mean().item())
        token_exact = bool(
            torch.equal(baseline_logits.argmax(-1), candidate_logits.argmax(-1))
        )
        semantic_rows.append(
            {
                "window_index": window_index,
                "source_object": window.source_object,
                "full_vocab_forward_kl": kl,
                "top1_exact": token_exact,
                "max_abs_logit_error": float(
                    (baseline_logits - candidate_logits).abs().max().item()
                ),
                "intercept": ledger.verify_complete(),
                "fork": fork_audit,
            }
        )
        del stock, baseline, persistent, request, document, query
        torch.cuda.empty_cache()
    mean_kl = statistics.fmean(row["full_vocab_forward_kl"] for row in semantic_rows)
    top1 = all(row["top1_exact"] for row in semantic_rows)
    # KL is authorized only after all eight independent shards are aggregated.
    # A single shard must still be finite and top-1 exact.
    passed = bool(top1 and math.isfinite(mean_kl))
    result = {
        "status": "completed_pg19_gate_shard" if passed else "pg19_gate_shard_failed",
        "passed": passed,
        "rank": args.rank,
        "world_size": args.world_size,
        "kernel_mode": KERNEL_MODE,
        "data_audit": data_audit,
        "windows_sha256": windows_sha,
        "isolated_gate": {"passed": True, "windows": isolated_rows},
        "semantic_gate": {
            "passed": passed,
            "top1_agreement": 1.0 if top1 else sum(row["top1_exact"] for row in semantic_rows) / len(semantic_rows),
            "example_equal_mean_full_vocab_forward_kl": mean_kl,
            "mean_kl_threshold": args.semantic_mean_kl_threshold,
            "rows": semantic_rows,
        },
        "validation_consumed": False,
        "test_v2_consumed": False,
    }
    if not passed:
        raise RuntimeError(f"PG19 shard semantic gate failed: mean_kl={mean_kl}, top1={top1}")
    return result


@torch.inference_mode()
def _generate(model, backbone, tokenizer, request, backend_name, query, max_new_tokens):
    original = backbone.config._attn_implementation
    current = query
    tokens, times = [], []
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
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
            tokens.append(token)
            current = torch.tensor([[token]], dtype=torch.long, device=query.device)
    finally:
        backbone.config._attn_implementation = original
    return {
        "generated_token_ids": tokens,
        "generated_text": tokenizer.decode(tokens, skip_special_tokens=True),
        "ttft_seconds": times[0],
        "median_tpot_seconds": statistics.median(times[1:]),
        "cuda_peak_request_delta_bytes": torch.cuda.max_memory_allocated() - before,
    }


def _measure_config(args, model, backbone, tokenizer, plan, document, query, config):
    persistent = _build_document_cache(backbone, document, functional=True)
    q16_build = {
        "performed": False,
        "document_build_pack_seconds": 0.0,
        "document_build_copy_nbytes": 0,
        "document_build_cuda_peak_delta_bytes": 0,
    }
    if config == "stock-eager":
        request = _fork_dense_functional(persistent, plan)
        backend = "eager"
        conversion = None
        ledger = None
        fork = {"full_document_staging_copy_nbytes": 0}
    elif config == "vllm-paged-q16":
        _sync()
        torch.cuda.reset_peak_memory_stats()
        build_allocated_before = torch.cuda.memory_allocated()
        build_started = time.perf_counter()
        conversion = convert_all_qwen35_full_layers_to_vllm_q16(
            persistent,
            plan,
            page_size=args.page_size,
            max_append_tokens=int(query.shape[1]) + args.max_new_tokens,
            max_request_forks=1,
        )
        _sync()
        q16_build = {
            "performed": True,
            "document_build_pack_seconds": time.perf_counter() - build_started,
            # This is the explicit, one-time dense-K/V -> NHD block-pool pack.
            # It is intentionally separate from the zero-copy request fork.
            "document_build_copy_nbytes": conversion.dense_document_nbytes,
            "document_build_cuda_peak_delta_bytes": (
                torch.cuda.max_memory_allocated() - build_allocated_before
            ),
        }
        request, fork = fork_qwen35_vllm_q16_request(persistent, plan)
        # Validation inputs are unpadded batch-1 token tensors. This one-time
        # caller contract lets the timed backend avoid materializing/validating
        # a 4D QxKV mask (and ten GPU->CPU .item() synchronizations) per step.
        if document.ndim != 2 or query.ndim != 2 or document.shape[0] != 1 or query.shape[0] != 1:
            raise RuntimeError("production no-mask contract requires unpadded batch-1 token tensors")
        for index in plan.full_attention_layer_indices:
            request.layers[index].sequence.strict_mask_check = False
        ledger = Qwen35VllmPagedHitLedger(
            plan,
            conversion,
            expected_calls_per_layer=args.max_new_tokens,
            mask_contract="prevalidated-no-padding-tail-causal",
        )
        backend = register_qwen35_vllm_q16_backend(ledger).name
    else:
        raise RuntimeError(f"unknown config {config}")
    persistent_bytes = cache_nbytes(persistent)
    result = _generate(
        model, backbone, tokenizer, request, backend, query, args.max_new_tokens
    )
    result.update(
        {
            "config": config,
            "configuration_scope": (
                "transformers-eager-full-attention-with-functional-gdn-fork-control"
                if config == "stock-eager"
                else "vllm-q16-full-attention-with-functional-gdn-fork"
            ),
            "persistent_total_resident_nbytes": persistent_bytes,
            "persistent_resident_nbytes": persistent_bytes,
            "query_fork_full_document_staging_copy_nbytes": fork[
                "full_document_staging_copy_nbytes"
            ],
            # Backwards-readable alias whose scope is explicitly the query fork.
            "full_document_staging_copy_nbytes": fork[
                "full_document_staging_copy_nbytes"
            ],
            "q16_document_build": q16_build,
            "fork": fork,
            "kernel_mode": KERNEL_MODE if ledger else "transformers-eager",
            "intercept": ledger.verify_complete() if ledger else None,
        }
    )
    if conversion is not None:
        result["dense_document_kv_nbytes"] = conversion.dense_document_nbytes
        result["persistent_q16_allocated_block_pool_nbytes"] = (
            conversion.allocated_block_pool_nbytes
        )
        result["persistent_q16_document_payload_nbytes"] = (
            conversion.document_payload_nbytes
        )
        # Compatibility aliases; the persistent/build scope is unambiguous above.
        result["allocated_block_pool_nbytes"] = conversion.allocated_block_pool_nbytes
        result["document_payload_nbytes"] = conversion.document_payload_nbytes
    return result


def _median_measurements(config: str, trials: list[dict[str, Any]]) -> dict[str, Any]:
    _formal_require(len(trials) == 4, f"{config} requires four fresh ABBA trials")
    tokens = trials[0]["generated_token_ids"]
    _formal_require(
        all(trial["generated_token_ids"] == tokens for trial in trials),
        f"{config} generated tokens changed across fresh trials",
    )
    representative = dict(trials[0])
    representative.update(
        {
            "fresh_trial_count": 4,
            "fresh_trials": trials,
            "ttft_seconds": statistics.median(
                trial["ttft_seconds"] for trial in trials
            ),
            "median_tpot_seconds": statistics.median(
                trial["median_tpot_seconds"] for trial in trials
            ),
            "cuda_peak_request_delta_bytes": statistics.median(
                trial["cuda_peak_request_delta_bytes"] for trial in trials
            ),
            "persistent_total_resident_nbytes": statistics.median(
                trial["persistent_total_resident_nbytes"] for trial in trials
            ),
            "persistent_resident_nbytes": statistics.median(
                trial["persistent_resident_nbytes"] for trial in trials
            ),
        }
    )
    if config == "vllm-paged-q16":
        representative["q16_document_build"] = {
            "performed": True,
            "document_build_pack_seconds": statistics.median(
                trial["q16_document_build"]["document_build_pack_seconds"]
                for trial in trials
            ),
            "document_build_copy_nbytes": trials[0]["q16_document_build"][
                "document_build_copy_nbytes"
            ],
            "document_build_cuda_peak_delta_bytes": statistics.median(
                trial["q16_document_build"]["document_build_cuda_peak_delta_bytes"]
                for trial in trials
            ),
        }
    return representative


def _record_fresh_trial(
    fresh_trials: dict[str, list[dict[str, Any]]],
    observed_tokens: dict[str, list[int]],
    config: str,
    trial: dict[str, Any],
) -> None:
    """Record one ABBA trial, failing before another timed trial can start."""

    tokens = trial["generated_token_ids"]
    if config in observed_tokens and tokens != observed_tokens[config]:
        raise RuntimeError(f"{config} generated tokens changed across fresh ABBA trials")
    observed_tokens[config] = tokens
    if len(observed_tokens) == 2 and len(
        {tuple(value) for value in observed_tokens.values()}
    ) != 1:
        raise RuntimeError(
            "stock/Q16 generated tokens diverged; stop before further paired timing"
        )
    fresh_trials[config].append(trial)


def _run_fresh_abba(
    args: argparse.Namespace,
    model: Any,
    backbone: Any,
    tokenizer: Any,
    plan: Any,
    document: torch.Tensor,
    query: torch.Tensor,
) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, list[dict[str, Any]]]]:
    pair = ("stock-eager", "vllm-paged-q16")
    warmup_order = pair if args.rank % 2 == 0 else tuple(reversed(pair))
    order = (
        ("stock-eager", "vllm-paged-q16", "vllm-paged-q16", "stock-eager")
        if args.rank % 2 == 0
        else ("vllm-paged-q16", "stock-eager", "stock-eager", "vllm-paged-q16")
    ) * 2
    # Exactly one fresh warmup per config; no warmup state is measured/reused.
    for config in warmup_order:
        warm = _measure_config(
            args, model, backbone, tokenizer, plan, document, query, config
        )
        del warm
        torch.cuda.empty_cache()
    fresh_trials = {config: [] for config in pair}
    observed_tokens: dict[str, list[int]] = {}
    for config in order:
        trial = _measure_config(
            args, model, backbone, tokenizer, plan, document, query, config
        )
        _record_fresh_trial(fresh_trials, observed_tokens, config, trial)
        torch.cuda.empty_cache()
    return warmup_order, order, fresh_trials


@torch.inference_mode()
def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    authorization = json.loads(args.authorization.read_text())
    if authorization.get("status") != "pg19_gate_authorized" or not authorization.get("passed"):
        raise RuntimeError("PG19 authorization is absent or failed")
    if sha256_file(args.authorization) != args.expected_authorization_sha256:
        raise RuntimeError("PG19 authorization SHA mismatch")
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    args.data = args.validation_data
    args.exclude_source_indices = (4, 5)
    args.allow_test_v2 = False
    args.context_lengths = ()
    args.synthetic_repetitions = 0
    workloads, metadata = longbench_workloads(tokenizer, args)
    indices = sorted({int(row["source_index"]) for row in workloads})
    expected_pairs = {
        (dataset, source_index)
        for dataset in ("qasper", "2wikimqa")
        for source_index in range(6, 10)
    }
    observed_pairs = {
        (str(row["dataset"]), int(row["source_index"])) for row in workloads
    }
    workload_ids = [str(row["workload_id"]) for row in workloads]
    if (
        indices != [6, 7, 8, 9]
        or len(workloads) != 8
        or observed_pairs != expected_pairs
        or len(set(workload_ids)) != 8
        or metadata.get("test_v2_consumed")
        or metadata.get("source_revisions") != [args.expected_source_revision]
        or metadata.get("datasets") != ["2wikimqa", "qasper"]
    ):
        raise RuntimeError("validation source/workload isolation failed")
    assigned = workloads[args.rank :: args.world_size]
    if len(assigned) != 1:
        raise RuntimeError("formal 8-rank benchmark requires one workload per rank")
    torch.cuda.set_device(0)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    workload = assigned[0]
    document = _tokens(workload["document_tokens"], args.max_input_tokens).cuda()
    query = _tokens(workload["query_tokens"], args.max_input_tokens).cuda()
    warmup_order, order, fresh_trials = _run_fresh_abba(
        args, model, backbone, tokenizer, plan, document, query
    )
    pair = ("stock-eager", "vllm-paged-q16")
    measurements = {
        config: _median_measurements(config, fresh_trials[config]) for config in pair
    }
    stock = measurements["stock-eager"]
    fused = measurements["vllm-paged-q16"]
    token_exact = stock["generated_token_ids"] == fused["generated_token_ids"]
    if not token_exact:
        raise RuntimeError(
            "stock/Q16 generated tokens diverged; paired autoregressive timing is invalid"
        )
    return {
        "status": "completed_shard",
        "rank": args.rank,
        "world_size": args.world_size,
        "kernel_mode": KERNEL_MODE,
        "authorization_sha256": args.expected_authorization_sha256,
        "workload_metadata": metadata,
        "workload": {
            "workload_id": workload["workload_id"],
            "dataset": workload["dataset"],
            "source_index": workload["source_index"],
            "document_tokens": int(document.shape[1]),
            "query_tokens": int(query.shape[1]),
        },
        "warmup_order": warmup_order,
        "measurement_order": order,
        "measurement_protocol": "fresh-state-ABBAx2-four-trials-per-config",
        "warmup_runs_per_config": 1,
        "fresh_measurement_runs_per_config": 4,
        "measurements": measurements,
        "paired": {
            "generated_tokens_exact": token_exact,
            "ttft_ratio_fused_vs_stock": fused["ttft_seconds"] / stock["ttft_seconds"],
            "tpot_ratio_fused_vs_stock": fused["median_tpot_seconds"] / stock["median_tpot_seconds"],
            "persistent_ratio_fused_vs_stock": fused["persistent_total_resident_nbytes"] / stock["persistent_total_resident_nbytes"],
            "cuda_peak_delta_ratio_fused_vs_stock": fused["cuda_peak_request_delta_bytes"] / max(stock["cuda_peak_request_delta_bytes"], 1),
        },
        "test_v2_consumed": False,
    }


def _formal_require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _normalized_counts(value: Any) -> dict[int, int]:
    _formal_require(isinstance(value, dict), "kernel intercept counts must be a mapping")
    try:
        return {int(index): int(count) for index, count in value.items()}
    except (TypeError, ValueError) as error:
        raise RuntimeError("kernel intercept counts are not integer-valued") from error


def _validate_fused_intercept(
    intercept: Any,
    *,
    expected_layers: tuple[int, ...],
    expected_calls_per_layer: int,
    expected_mask_contract: str | None = None,
) -> None:
    _formal_require(isinstance(intercept, dict), "missing fused intercept ledger")
    _formal_require(intercept.get("verified") is True, "fused intercept is unverified")
    _formal_require(intercept.get("kernel_mode") == KERNEL_MODE, "kernel mode drift")
    if expected_mask_contract is not None:
        _formal_require(
            intercept.get("mask_contract") == expected_mask_contract,
            "fused mask contract drift",
        )
    _formal_require(
        tuple(intercept.get("expected_layer_indices", ())) == expected_layers,
        "full-attention layer coverage drift",
    )
    expected_counts = {index: expected_calls_per_layer for index in expected_layers}
    counts = _normalized_counts(intercept.get("counts"))
    _formal_require(counts == expected_counts, f"fused call counts drifted: {counts}")
    _formal_require(
        int(intercept.get("total_calls", -1)) == len(expected_layers) * expected_calls_per_layer,
        "fused total call count drift",
    )
    _formal_require(intercept.get("dense_fallback_calls") == 0, "dense fallback observed")
    _formal_require(intercept.get("full_kv_concatenations") == 0, "full KV cat observed")
    _formal_require(
        intercept.get("position_ids_contract") == POST_ROPE_POSITION_IDS_CONTRACT,
        "position_ids contract drift",
    )
    if intercept.get("mask_contract") == "prevalidated-no-padding-tail-causal":
        _formal_require(
            intercept.get("materialized_attention_mask_nbytes") == 0,
            "production backend materialized an attention mask",
        )
        _formal_require(
            intercept.get("mask_validation_host_syncs") == 0,
            "production backend synchronized for mask validation",
        )
        _formal_require(
            intercept.get("position_ids_validation_host_syncs") == 0,
            "production backend synchronized for position_ids validation",
        )
    calls = intercept.get("calls")
    _formal_require(isinstance(calls, (list, tuple)), "missing per-call fused audit")
    _formal_require(
        len(calls) == len(expected_layers) * expected_calls_per_layer,
        "per-call fused audit length drift",
    )
    observed = Counter()
    for call in calls:
        _formal_require(isinstance(call, dict), "invalid per-call fused audit")
        layer = int(call.get("layer_idx", -1))
        observed[layer] += 1
        _formal_require(call.get("kernel_mode") == KERNEL_MODE, "per-call kernel drift")
        _formal_require(call.get("fused_gpu_kernel_calls") == 1, "non-single fused call")
        _formal_require(call.get("full_kv_concatenations") == 0, "per-call full KV cat")
        _formal_require(
            call.get("full_document_staging_copy_nbytes") == 0,
            "query path staged the full document",
        )
        _formal_require(call.get("quantization") == "Q16", "non-Q16 kernel call")
        _formal_require(
            call.get("position_ids_contract") == POST_ROPE_POSITION_IDS_CONTRACT
            and call.get("position_ids_validated") is True
            and call.get("position_ids_semantically_consumed_upstream") is True,
            "per-call position_ids contract drift",
        )
        if intercept.get("mask_contract") == "prevalidated-no-padding-tail-causal":
            _formal_require(
                call.get("materialized_attention_mask_nbytes") == 0
                and call.get("mask_validation_host_syncs") == 0,
                "production per-call mask overhead is nonzero",
            )
            _formal_require(
                call.get("position_ids_strict_tail_values_checked") is False
                and call.get("position_ids_validation_host_syncs") == 0,
                "production position_ids validation added a host synchronization",
            )
        else:
            _formal_require(
                call.get("position_ids_strict_tail_values_checked") is True
                and call.get("position_ids_validation_host_syncs") == 1,
                "strict gate did not validate position_ids tail values",
            )
    _formal_require(dict(observed) == expected_counts, "per-call layer ledger drift")


def _validate_pg19_authorization(
    authorization: Any,
    *,
    expected_layers: tuple[int, ...],
) -> None:
    _formal_require(isinstance(authorization, dict), "authorization must be JSON object")
    _formal_require(
        authorization.get("status") == "pg19_gate_authorized"
        and authorization.get("passed") is True,
        "PG19 authorization is absent or failed",
    )
    _formal_require(
        authorization.get("validation_consumed") is False
        and authorization.get("test_v2_consumed") is False,
        "PG19 gate violated validation/test-v2 isolation",
    )
    isolated = authorization.get("isolated_gate", {})
    windows = isolated.get("windows") if isinstance(isolated, dict) else None
    _formal_require(isolated.get("passed") is True, "isolated PG19 gate failed")
    _formal_require(isinstance(windows, list) and len(windows) == 8, "PG19 isolated window count drift")
    for window in windows:
        _formal_require(window.get("passed") is True, "isolated window failed")
        _formal_require(
            tuple(window.get("layer_indices", ())) == expected_layers,
            "isolated layer indices drift",
        )
        _formal_require(window.get("layer_count") == 10, "isolated layer count drift")
        _formal_require(window.get("dense_fallback_calls") == 0, "isolated dense fallback")
        _formal_require(len(window.get("rows", ())) == 10, "isolated row count drift")
        for row in window["rows"]:
            _formal_require(row.get("close") is True and row.get("finite") is True, "isolated parity failed")
            audit = row.get("audit", {})
            _formal_require(audit.get("fused_gpu_kernel_calls") == 1, "isolated fused call missing")
            _formal_require(audit.get("full_kv_concatenations") == 0, "isolated full KV cat")
            _formal_require(
                audit.get("position_ids_contract") == POST_ROPE_POSITION_IDS_CONTRACT
                and audit.get("position_ids_validated") is True
                and audit.get("position_ids_semantically_consumed_upstream") is True
                and audit.get("position_ids_strict_tail_values_checked") is True,
                "isolated position_ids tail contract was not strictly validated",
            )
    semantic = authorization.get("semantic_gate", {})
    _formal_require(semantic.get("passed") is True, "semantic PG19 gate failed")
    _formal_require(semantic.get("top1_agreement") == 1.0, "PG19 top-1 is not exact")
    _formal_require(
        float(semantic.get("example_equal_mean_full_vocab_forward_kl", float("inf")))
        <= float(semantic.get("mean_kl_threshold", -1.0)),
        "PG19 mean full-vocab KL exceeds authorization threshold",
    )
    semantic_rows = semantic.get("rows")
    _formal_require(
        isinstance(semantic_rows, list) and len(semantic_rows) == 8,
        "PG19 semantic window count drift",
    )
    for row in semantic_rows:
        _formal_require(row.get("top1_exact") is True, "PG19 semantic top-1 mismatch")
        _validate_fused_intercept(
            row.get("intercept"),
            expected_layers=expected_layers,
            expected_calls_per_layer=1,
            expected_mask_contract="strict-canonical-audit",
        )
        _formal_require(
            row.get("fork", {}).get("full_document_staging_copy_nbytes") == 0,
            "PG19 query fork staged the full document",
        )


def aggregate_pg19_gate_shards(
    shard_paths: list[Path],
    *,
    expected_windows_sha256: str,
    mean_kl_threshold: float,
) -> dict[str, Any]:
    """Aggregate eight one-GPU/one-window gate shards into one authorization."""

    _formal_require(len(shard_paths) == 8, "PG19 aggregation requires eight shards")
    rows = [json.loads(path.read_text()) for path in shard_paths]
    _formal_require({row.get("rank") for row in rows} == set(range(8)), "PG19 rank coverage drift")
    _formal_require(all(row.get("world_size") == 8 for row in rows), "PG19 world-size drift")
    _formal_require(
        all(
            row.get("status") == "completed_pg19_gate_shard"
            and row.get("passed") is True
            and row.get("kernel_mode") == KERNEL_MODE
            for row in rows
        ),
        "one or more PG19 gate shards failed",
    )
    _formal_require(
        all(row.get("windows_sha256") == expected_windows_sha256 for row in rows),
        "PG19 window SHA drift across shards",
    )
    _formal_require(
        all(
            row.get("validation_consumed") is False
            and row.get("test_v2_consumed") is False
            for row in rows
        ),
        "PG19 shard consumed validation/test-v2",
    )
    data_audit_json = {
        json.dumps(row.get("data_audit"), sort_keys=True, separators=(",", ":"))
        for row in rows
    }
    static_json = {
        json.dumps(row.get("static"), sort_keys=True, separators=(",", ":"))
        for row in rows
    }
    _formal_require(len(data_audit_json) == 1, "PG19 data audit drift across ranks")
    _formal_require(len(static_json) == 1, "PG19 static audit drift across ranks")
    isolated_windows = []
    semantic_rows = []
    for row in sorted(rows, key=lambda value: value["rank"]):
        shard_isolated = row.get("isolated_gate", {}).get("windows")
        shard_semantic = row.get("semantic_gate", {}).get("rows")
        _formal_require(
            isinstance(shard_isolated, list) and len(shard_isolated) == 1,
            "PG19 shard must contain one isolated window",
        )
        _formal_require(
            isinstance(shard_semantic, list) and len(shard_semantic) == 1,
            "PG19 shard must contain one semantic window",
        )
        _formal_require(
            shard_isolated[0].get("window_index") == row["rank"]
            and shard_semantic[0].get("window_index") == row["rank"],
            "PG19 rank/window assignment drift",
        )
        _formal_require(
            shard_isolated[0].get("source_object")
            == shard_semantic[0].get("source_object"),
            "PG19 isolated/semantic source mismatch",
        )
        isolated_windows.extend(shard_isolated)
        semantic_rows.extend(shard_semantic)
    source_objects = [row.get("source_object") for row in semantic_rows]
    _formal_require(
        len(set(source_objects)) == 8 and all(source_objects),
        "PG19 source objects are not eight unique train objects",
    )
    top1_agreement = sum(bool(row.get("top1_exact")) for row in semantic_rows) / 8
    mean_kl = statistics.fmean(
        float(row.get("full_vocab_forward_kl", float("inf"))) for row in semantic_rows
    )
    _formal_require(top1_agreement == 1.0, "PG19 global top-1 agreement is not exact")
    _formal_require(
        math.isfinite(mean_kl) and mean_kl <= mean_kl_threshold,
        f"PG19 global mean full-vocab KL {mean_kl} exceeds {mean_kl_threshold}",
    )
    result = {
        "status": "pg19_gate_authorized",
        "passed": True,
        "kernel_mode": KERNEL_MODE,
        "parallel_gate_world_size": 8,
        "one_window_per_rank": True,
        "rank_shards": [str(path) for path in shard_paths],
        "static": rows[0]["static"],
        "data_audit": rows[0]["data_audit"],
        "windows_sha256": expected_windows_sha256,
        "source_objects": source_objects,
        "isolated_gate": {"passed": True, "windows": isolated_windows},
        "semantic_gate": {
            "passed": True,
            "top1_agreement": top1_agreement,
            "example_equal_mean_full_vocab_forward_kl": mean_kl,
            "mean_kl_threshold": mean_kl_threshold,
            "rows": semantic_rows,
        },
        "validation_consumed": False,
        "test_v2_consumed": False,
    }
    _validate_pg19_authorization(
        result, expected_layers=tuple(range(3, 40, 4))
    )
    return result


def summarize_validation_shards(
    run_dir: Path,
    *,
    authorization_path: Path,
    expected_authorization_sha256: str,
    expected_code_ledger_sha256: str,
    expected_model_manifest_sha256: str,
    expected_model_artifact_ledger_sha256: str,
    expected_model_weight_ledger_sha256: str,
    expected_source_revision: str,
    expected_calls_per_layer: int,
) -> dict[str, Any]:
    """Hard-gate and summarize the eight single-request validation shards."""

    _formal_require(
        sha256_file(authorization_path) == expected_authorization_sha256,
        "authorization SHA mismatch during summary",
    )
    ledger_path = run_dir / "code.sha256"
    _formal_require(ledger_path.is_file(), "missing frozen code ledger")
    _formal_require(
        sha256_file(ledger_path) == expected_code_ledger_sha256,
        "code ledger SHA mismatch during summary",
    )
    model_artifact_ledger = run_dir / "model-artifacts.sha256"
    model_weight_ledger = run_dir / "model-weights.sha256"
    _formal_require(
        model_artifact_ledger.is_file()
        and sha256_file(model_artifact_ledger)
        == expected_model_artifact_ledger_sha256,
        "model artifact ledger SHA mismatch during summary",
    )
    _formal_require(
        model_weight_ledger.is_file()
        and sha256_file(model_weight_ledger) == expected_model_weight_ledger_sha256,
        "model weight ledger SHA mismatch during summary",
    )
    authorization = json.loads(authorization_path.read_text())
    expected_layers = tuple(range(3, 40, 4))
    _validate_pg19_authorization(authorization, expected_layers=expected_layers)
    _formal_require(
        authorization.get("static", {}).get("model_manifest_sha256")
        == expected_model_manifest_sha256,
        "authorization model manifest drift",
    )

    paths = sorted((run_dir / "validation").glob("vllm-paged-q16-shard-*.json"))
    _formal_require(len(paths) == 8, f"expected 8 shards, found {len(paths)}")
    rows = [json.loads(path.read_text()) for path in paths]
    _formal_require({row.get("rank") for row in rows} == set(range(8)), "rank coverage drift")
    expected_pairs = {
        (dataset, source_index)
        for dataset in ("qasper", "2wikimqa")
        for source_index in range(6, 10)
    }
    observed_pairs: set[tuple[str, int]] = set()
    workload_ids: set[str] = set()
    orders = Counter()
    build_copy_nbytes = []
    build_pack_seconds = []
    stock_persistent_nbytes = []
    fused_persistent_nbytes = []
    stock_query_peak_nbytes = []
    fused_query_peak_nbytes = []
    query_peak_ratios = []
    stock_ttft_seconds = []
    fused_ttft_seconds = []
    stock_tpot_seconds = []
    fused_tpot_seconds = []
    q16_document_payload_nbytes = []
    q16_allocated_pool_nbytes = []
    q16_build_peak_nbytes = []
    for row in rows:
        _formal_require(row.get("status") == "completed_shard", "incomplete validation shard")
        _formal_require(row.get("world_size") == 8, "validation world-size drift")
        _formal_require(row.get("kernel_mode") == KERNEL_MODE, "shard kernel mode drift")
        _formal_require(
            row.get("static", {}).get("model_manifest_sha256")
            == expected_model_manifest_sha256,
            "shard model manifest drift",
        )
        _formal_require(
            row.get("authorization_sha256") == expected_authorization_sha256,
            "shard authorization SHA drift",
        )
        _formal_require(row.get("test_v2_consumed") is False, "test-v2 was consumed")
        metadata = row.get("workload_metadata", {})
        _formal_require(metadata.get("test_v2_consumed") is False, "metadata reports test-v2")
        _formal_require(
            metadata.get("source_revisions") == [expected_source_revision],
            "source revision drift",
        )
        _formal_require(metadata.get("datasets") == ["2wikimqa", "qasper"], "dataset set drift")
        workload = row.get("workload", {})
        pair = (str(workload.get("dataset")), int(workload.get("source_index", -1)))
        observed_pairs.add(pair)
        workload_id = str(workload.get("workload_id"))
        _formal_require(workload_id not in workload_ids, "duplicate validation workload")
        workload_ids.add(workload_id)
        order = tuple(row.get("measurement_order", ()))
        _formal_require(
            order
            in (
                (
                    "stock-eager",
                    "vllm-paged-q16",
                    "vllm-paged-q16",
                    "stock-eager",
                )
                * 2,
                (
                    "vllm-paged-q16",
                    "stock-eager",
                    "stock-eager",
                    "vllm-paged-q16",
                )
                * 2,
            ),
            "invalid fresh-state ABBAx2 order",
        )
        orders[order] += 1
        _formal_require(row.get("warmup_runs_per_config") == 1, "warmup count drift")
        _formal_require(
            row.get("fresh_measurement_runs_per_config") == 4
            and row.get("measurement_protocol")
            == "fresh-state-ABBAx2-four-trials-per-config",
            "fresh ABBA measurement count/protocol drift",
        )
        measurements = row.get("measurements", {})
        _formal_require(
            set(measurements) == {"stock-eager", "vllm-paged-q16"},
            "measurement config set drift",
        )
        stock, fused = measurements["stock-eager"], measurements["vllm-paged-q16"]
        _formal_require(
            stock.get("fresh_trial_count") == 4
            and fused.get("fresh_trial_count") == 4
            and len(stock.get("fresh_trials", ())) == 4
            and len(fused.get("fresh_trials", ())) == 4,
            "fresh trial evidence drift",
        )
        _formal_require(stock.get("kernel_mode") == "transformers-eager", "stock backend drift")
        _formal_require(fused.get("kernel_mode") == KERNEL_MODE, "fused backend drift")
        _formal_require(
            stock.get("configuration_scope")
            == "transformers-eager-full-attention-with-functional-gdn-fork-control",
            "stock control scope is mislabeled",
        )
        _formal_require(
            fused.get("configuration_scope")
            == "vllm-q16-full-attention-with-functional-gdn-fork",
            "fused control scope is mislabeled",
        )
        _formal_require(row.get("paired", {}).get("generated_tokens_exact") is True, "Q16 token mismatch")
        _formal_require(
            len(stock.get("generated_token_ids", ())) == expected_calls_per_layer
            and stock.get("generated_token_ids") == fused.get("generated_token_ids"),
            "generated token count/content drift",
        )
        for stock_trial, fused_trial in zip(
            stock["fresh_trials"], fused["fresh_trials"]
        ):
            _formal_require(
                stock_trial.get("generated_token_ids")
                == fused_trial.get("generated_token_ids")
                == stock.get("generated_token_ids"),
                "a fresh ABBA replicate has divergent generated tokens",
            )
            _formal_require(
                stock_trial.get("q16_document_build", {}).get("performed") is False,
                "a fresh stock replicate reports a Q16 build",
            )
            _validate_fused_intercept(
                fused_trial.get("intercept"),
                expected_layers=expected_layers,
                expected_calls_per_layer=expected_calls_per_layer,
                expected_mask_contract="prevalidated-no-padding-tail-causal",
            )
        _formal_require(
            fused.get("query_fork_full_document_staging_copy_nbytes") == 0
            and fused.get("fork", {}).get("full_document_staging_copy_nbytes") == 0,
            "validation query fork staged the full document",
        )
        build = fused.get("q16_document_build", {})
        copy_nbytes = int(build.get("document_build_copy_nbytes", 0))
        _formal_require(build.get("performed") is True, "Q16 document pack was not recorded")
        _formal_require(copy_nbytes == fused.get("dense_document_kv_nbytes") > 0, "Q16 build-copy accounting drift")
        _formal_require(
            copy_nbytes == fused.get("persistent_q16_document_payload_nbytes"),
            "Q16 payload/build-copy accounting drift",
        )
        _formal_require(
            fused.get("persistent_q16_allocated_block_pool_nbytes", 0) >= copy_nbytes,
            "Q16 allocated pool is smaller than its document payload",
        )
        _formal_require(
            fused.get("persistent_total_resident_nbytes", 0)
            >= fused.get("persistent_q16_allocated_block_pool_nbytes", 0),
            "persistent total omits the Q16 pool",
        )
        _formal_require(
            float(build.get("document_build_pack_seconds", 0.0)) > 0.0
            and int(build.get("document_build_cuda_peak_delta_bytes", 0)) > 0,
            "Q16 document build timing/peak accounting is missing",
        )
        _formal_require(
            stock.get("q16_document_build", {}).get("performed") is False,
            "stock path incorrectly reports a Q16 pack",
        )
        for trial in fused["fresh_trials"]:
            trial_build = trial.get("q16_document_build", {})
            _formal_require(
                trial_build.get("performed") is True
                and trial_build.get("document_build_copy_nbytes") == copy_nbytes
                and trial.get("query_fork_full_document_staging_copy_nbytes") == 0,
                "fresh Q16 trial build/fork accounting drift",
            )
        build_copy_nbytes.append(copy_nbytes)
        build_pack_seconds.append(float(build["document_build_pack_seconds"]))
        stock_persistent_nbytes.append(float(stock["persistent_total_resident_nbytes"]))
        fused_persistent_nbytes.append(float(fused["persistent_total_resident_nbytes"]))
        stock_query_peak_nbytes.append(float(stock["cuda_peak_request_delta_bytes"]))
        fused_query_peak_nbytes.append(float(fused["cuda_peak_request_delta_bytes"]))
        query_peak_ratios.append(
            float(row["paired"]["cuda_peak_delta_ratio_fused_vs_stock"])
        )
        stock_ttft_seconds.append(float(stock["ttft_seconds"]))
        fused_ttft_seconds.append(float(fused["ttft_seconds"]))
        stock_tpot_seconds.append(float(stock["median_tpot_seconds"]))
        fused_tpot_seconds.append(float(fused["median_tpot_seconds"]))
        q16_document_payload_nbytes.append(
            float(fused["persistent_q16_document_payload_nbytes"])
        )
        q16_allocated_pool_nbytes.append(
            float(fused["persistent_q16_allocated_block_pool_nbytes"])
        )
        q16_build_peak_nbytes.append(
            float(build["document_build_cuda_peak_delta_bytes"])
        )

    _formal_require(observed_pairs == expected_pairs, f"source×dataset coverage drift: {observed_pairs}")
    _formal_require(len(workload_ids) == 8, "workloads are not unique")
    _formal_require(
        sorted(orders.values()) == [4, 4],
        f"A/B order is not balanced: {orders}",
    )
    return {
        "status": "completed",
        "scope": "single-request-per-document-short-validation",
        "stock_control_scope": (
            "transformers-eager-full-attention-with-functional-gdn-fork-control"
        ),
        "multi_query_serving_completed": False,
        "kernel_mode": KERNEL_MODE,
        "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
        "production_position_ids_validation_host_syncs": 0,
        "authorization_sha256": expected_authorization_sha256,
        "code_ledger_sha256": expected_code_ledger_sha256,
        "model_manifest_sha256": expected_model_manifest_sha256,
        "model_artifact_ledger_sha256": expected_model_artifact_ledger_sha256,
        "model_weight_ledger_sha256": expected_model_weight_ledger_sha256,
        "shards": [str(path) for path in paths],
        "source_dataset_pairs": sorted(observed_pairs),
        "generated_tokens_exact_fraction": 1.0,
        "warmup_runs_per_config": 1,
        "fresh_measurement_runs_per_config": 4,
        "measurement_protocol": "fresh-state-ABBAx2-four-trials-per-config",
        "ab_order_counts": {"stock_first": 4, "q16_first": 4},
        "median_ttft_ratio_fused_vs_stock": statistics.median(
            row["paired"]["ttft_ratio_fused_vs_stock"] for row in rows
        ),
        "median_tpot_ratio_fused_vs_stock": statistics.median(
            row["paired"]["tpot_ratio_fused_vs_stock"] for row in rows
        ),
        "median_stock_control_ttft_seconds": statistics.median(stock_ttft_seconds),
        "median_vllm_q16_ttft_seconds": statistics.median(fused_ttft_seconds),
        "median_stock_control_tpot_seconds": statistics.median(stock_tpot_seconds),
        "median_vllm_q16_tpot_seconds": statistics.median(fused_tpot_seconds),
        "median_persistent_ratio_fused_vs_stock": statistics.median(
            row["paired"]["persistent_ratio_fused_vs_stock"] for row in rows
        ),
        "median_stock_control_persistent_total_resident_nbytes": statistics.median(
            stock_persistent_nbytes
        ),
        "median_vllm_q16_persistent_total_resident_nbytes": statistics.median(
            fused_persistent_nbytes
        ),
        "median_stock_control_query_cuda_peak_delta_bytes": statistics.median(
            stock_query_peak_nbytes
        ),
        "median_vllm_q16_query_cuda_peak_delta_bytes": statistics.median(
            fused_query_peak_nbytes
        ),
        "median_query_cuda_peak_delta_ratio_fused_vs_stock": statistics.median(
            query_peak_ratios
        ),
        "median_q16_document_build_pack_seconds": statistics.median(build_pack_seconds),
        "median_q16_document_build_copy_nbytes": statistics.median(build_copy_nbytes),
        "median_q16_document_build_cuda_peak_delta_bytes": statistics.median(
            q16_build_peak_nbytes
        ),
        "median_q16_document_payload_nbytes": statistics.median(
            q16_document_payload_nbytes
        ),
        "median_q16_allocated_block_pool_nbytes": statistics.median(
            q16_allocated_pool_nbytes
        ),
        "dense_fallback_calls": 0,
        "full_kv_concatenations": 0,
        "query_fork_full_document_staging_copy_nbytes": 0,
        "nvml_process_memory_sampled": False,
        "memory_metric_limitations": (
            "This short gate reports PyTorch CUDA allocator deltas only; it does "
            "not replace the prior NVML process-memory protocol."
        ),
        "test_v2_consumed": False,
    }


def _validate_static(args: argparse.Namespace) -> dict[str, Any]:
    if args.bits != 16:
        raise RuntimeError("formal fused runner supports Q16 only; Q8/Q4 fail closed")
    if args.world_size != 8 or not 0 <= args.rank < args.world_size:
        raise RuntimeError("formal protocol requires ranks 0..7 of world size 8")
    if (args.source_index_start, args.source_index_end) != (6, 9):
        raise RuntimeError("formal validation indices are frozen to 6--9")
    if args.page_size < 16 or args.page_size % 16:
        raise RuntimeError("vLLM page size must be a multiple of 16")
    if args.max_new_tokens < 2:
        raise RuntimeError("formal latency protocol requires at least two generated tokens")
    for path in (args.pg19_data, args.pg19_manifest, args.validation_data):
        normalized = str(path).lower().replace("_", "-")
        if "test-v2" in normalized:
            raise RuntimeError("test-v2 path is forbidden")
        if not path.is_file():
            raise RuntimeError(f"missing frozen input: {path}")
    if sha256_file(args.pg19_data) != args.expected_pg19_sha256:
        raise RuntimeError("PG19 data SHA mismatch")
    if sha256_file(args.pg19_manifest) != args.expected_pg19_manifest_sha256:
        raise RuntimeError("PG19 manifest SHA mismatch")
    if sha256_file(args.validation_data) != args.expected_validation_sha256:
        raise RuntimeError("validation SHA mismatch")
    model_manifest_sha256, model_manifest = _model_manifest_sha(args.model)
    if model_manifest_sha256 != args.expected_model_manifest_sha256:
        raise RuntimeError(
            f"frozen model manifest SHA mismatch: {model_manifest_sha256}"
        )
    config = json.loads((args.model / "config.json").read_text())
    text = config.get("text_config", config)
    full = [i for i, kind in enumerate(text.get("layer_types", [])) if kind == "full_attention"]
    geometry = {
        "model_type": text.get("model_type"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "head_dim": text.get("head_dim"),
        "full_attention_layer_indices": full,
    }
    expected_full = list(range(3, 40, 4))
    if (
        geometry["model_type"],
        geometry["num_hidden_layers"],
        geometry["num_attention_heads"],
        geometry["num_key_value_heads"],
        geometry["head_dim"],
        len(full),
    ) != ("qwen3_5_moe_text", 40, 16, 2, 256, 10) or full != expected_full:
        raise RuntimeError(f"Qwen3.5 geometry drifted: {geometry}")
    environment = audit_frozen_kernel_environment()
    if not environment["matches_frozen_environment"]:
        raise RuntimeError(
            f"frozen kernel environment mismatch: {environment['mismatches']}"
        )
    return {
        "status": "static_dry_run_passed",
        "kernel_environment": environment,
        "geometry": geometry,
        "model_manifest_sha256": model_manifest_sha256,
        "model_manifest": model_manifest,
        "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
        "position_ids_semantics": (
            "text positions are consumed by Qwen3.5 mask/RoPE upstream; strict "
            "gate checks tail values and timed production checks metadata only"
        ),
        "bits": args.bits,
        "pg19_train_only": True,
        "validation_indices": [6, 7, 8, 9],
        "test_v2_consumed": False,
        "gpu_initialized": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("static-dry-run", "pg19-gate", "validation"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pg19-data", type=Path, required=True)
    parser.add_argument("--pg19-manifest", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pg19-sha256", required=True)
    parser.add_argument("--expected-pg19-manifest-sha256", required=True)
    parser.add_argument("--expected-pg19-windows-sha256", required=True)
    parser.add_argument("--expected-validation-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--expected-authorization-sha256")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--pg19-books", type=int, default=8)
    parser.add_argument("--pg19-document-tokens", type=int, default=1024)
    parser.add_argument("--pg19-query-tokens", type=int, default=32)
    parser.add_argument("--pg19-window-stride", type=int, default=512)
    parser.add_argument("--pg19-candidate-windows", type=int, default=4)
    parser.add_argument("--pg19-seed", type=int, default=20260813)
    parser.add_argument("--isolated-rtol", type=float, default=0.02)
    parser.add_argument("--isolated-atol", type=float, default=0.05)
    parser.add_argument("--semantic-mean-kl-threshold", type=float, default=1e-3)
    parser.add_argument("--source-index-start", type=int, default=6)
    parser.add_argument("--source-index-end", type=int, default=9)
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    args = parser.parse_args(argv)
    args.expected_source_indices = (6, 7, 8, 9)
    args.expected_workloads = 8
    return args


def main() -> None:
    args = parse_args()
    try:
        static = _validate_static(args)
        if args.stage == "static-dry-run":
            result = static
        else:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required after static gates")
            if args.stage == "pg19-gate":
                result = {"static": static, **run_pg19_gate(args)}
            else:
                if args.authorization is None or args.expected_authorization_sha256 is None:
                    raise RuntimeError("validation requires PG19 authorization and SHA")
                result = {"static": static, **run_validation(args)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output, result)
    except Exception as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            args.output,
            {
                "status": "failed",
                "stage": args.stage,
                "error_type": type(error).__name__,
                "error": str(error),
                "validation_authorized": False,
            },
        )
        raise
    print(json.dumps({"status": result["status"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
