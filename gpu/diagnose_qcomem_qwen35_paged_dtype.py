from __future__ import annotations

"""Single-GPU arithmetic diagnosis for the real Qwen3.5 paged backend.

This intentionally compares three executions of one frozen document/query:

1. the Transformers eager implementation;
2. the rejected one-pass page implementation that kept FP32 probabilities;
3. the two-pass page implementation that follows eager's BF16 weight contract.

It records every full-attention module output and the final logits.  This is a
diagnostic only; it never authorizes a formal benchmark or relaxes its gate.
"""

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import torch

from qcomem_paged_attention import (
    PagedTensorView,
    _mask_page_scores,
    _resolve_pages,
)
from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
from qcomem_qwen35_paged_integration import (
    PagedAttentionHitLedger,
    clone_dense_and_prepare_paged_cache_pair,
    register_qwen35_paged_backend,
    temporary_attention_implementation,
)
from run_deployment_bench import longbench_workloads
from run_downstream import atomic_json
from run_qcomem_qwen35_paged_real import (
    _build_dense_document_cache,
    _final_logits,
    _resolve_backbone,
    _same_query_caller,
)


def _batch_prefix(value: torch.Tensor, limit: int) -> torch.Tensor:
    """Normalize LongBench 1D/2D token tensors to ``[batch, tokens]``."""

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("token limit must be a positive integer")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise ValueError("token tensor must have shape [tokens] or [batch, tokens]")
    return value[:, :limit]


def legacy_fp32_paged_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor | PagedTensorView,
    value: torch.Tensor | PagedTensorView,
    attention_mask: torch.Tensor | dict[str, torch.Tensor] | None,
    dropout: float = 0.0,
    scaling: float | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, None]:
    """Exact arithmetic path used by failed Trial 1833998."""

    if dropout != 0.0:
        raise ValueError("legacy diagnostic supports inference only")
    if isinstance(attention_mask, dict):
        attention_mask = attention_mask["full_attention"]
    pages, _ = _resolve_pages(key, value)
    batch, query_heads, query_length, query_dim = query.shape
    kv_heads = pages[0].num_key_value_heads
    groups = query_heads // kv_heads
    grouped_query = query.reshape(
        batch, kv_heads, groups, query_length, query_dim
    )
    total_length = sum(page.length for page in pages)
    scale = scaling if scaling is not None else float(module.scaling)
    value_dim = pages[0].value_head_dim
    running_max = torch.full(
        (batch, query_heads, query_length),
        -torch.inf,
        dtype=torch.float32,
        device=query.device,
    )
    running_sum = torch.zeros_like(running_max)
    running_value = torch.zeros(
        (batch, query_heads, query_length, value_dim),
        dtype=torch.float32,
        device=query.device,
    )
    offset = 0
    for page in pages:
        page_key, page_value = page.materialize(
            device=query.device, dtype=query.dtype
        )
        scores = torch.einsum(
            "bhgqd,bhkd->bhgqk", grouped_query, page_key
        ).reshape(batch, query_heads, query_length, page.length)
        scores = _mask_page_scores(
            (scores * scale).float(),
            attention_mask=attention_mask,
            start=offset,
            end=offset + page.length,
            total_length=total_length,
            query_length=query_length,
            is_causal=bool(getattr(module, "is_causal", False)),
        )
        page_max = scores.amax(dim=-1)
        merged_max = torch.maximum(running_max, page_max)
        finite_merged = torch.isfinite(merged_max)
        old_scale = torch.where(
            torch.isfinite(running_max) & finite_merged,
            torch.exp(running_max - merged_max),
            torch.zeros_like(merged_max),
        )
        safe_max = torch.where(
            finite_merged, merged_max, torch.zeros_like(merged_max)
        )
        weights = torch.exp(scores - safe_max.unsqueeze(-1))
        weights = torch.where(
            finite_merged.unsqueeze(-1), weights, torch.zeros_like(weights)
        )
        page_sum = weights.sum(dim=-1)
        page_output = torch.einsum(
            "bhgqk,bhkd->bhgqd",
            weights.reshape(
                batch, kv_heads, groups, query_length, page.length
            ),
            page_value.float(),
        ).reshape(batch, query_heads, query_length, value_dim)
        running_value = running_value * old_scale.unsqueeze(-1) + page_output
        running_sum = running_sum * old_scale + page_sum
        running_max = merged_max
        offset += page.length
    output = torch.where(
        (running_sum > 0).unsqueeze(-1),
        running_value
        / running_sum.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1),
        torch.zeros_like(running_value),
    ).to(query.dtype)
    return output.transpose(1, 2).contiguous(), None


def _metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    reference = reference.float()
    candidate = candidate.float()
    error = (reference - candidate).abs()
    denominator = torch.linalg.vector_norm(reference.reshape(-1)).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    return {
        "shape": list(reference.shape),
        "max_abs": float(error.max().item()),
        "mean_abs": float(error.mean().item()),
        "relative_l2": float(
            (torch.linalg.vector_norm(error.reshape(-1)) / denominator).item()
        ),
        "close": bool(torch.allclose(reference, candidate, rtol=rtol, atol=atol)),
        "rtol": rtol,
        "atol": atol,
    }


def _capture_attention(
    backbone: Any,
    indices: tuple[int, ...],
    call: Callable[[], torch.Tensor],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    captures: dict[int, torch.Tensor] = {}
    handles = []
    for index in indices:
        def hook(_module: Any, _inputs: Any, output: Any, *, layer=index) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                raise RuntimeError(f"full-attention layer {layer} returned no tensor")
            captures[layer] = tensor.detach().float().cpu()

        handles.append(backbone.layers[index].self_attn.register_forward_hook(hook))
    try:
        logits = call().detach().float().cpu()
    finally:
        for handle in handles:
            handle.remove()
    if tuple(sorted(captures)) != indices:
        raise RuntimeError(
            f"attention capture coverage mismatch: {tuple(sorted(captures))}"
        )
    return logits, captures


@contextlib.contextmanager
def _replace_ledger_forward(ledger: Any, replacement: Callable[..., Any]):
    """Patch the concrete registered bound method, not its module global.

    ``AttentionInterface.register`` stores ``ledger.attention_forward`` as a
    bound method.  Patching the module-level ``paged_attention_forward`` after
    registration would therefore not affect the already bound callable.
    """

    original = ledger.attention_forward.__func__.__globals__["paged_attention_forward"]
    ledger.attention_forward.__func__.__globals__["paged_attention_forward"] = replacement
    try:
        yield
    finally:
        ledger.attention_forward.__func__.__globals__["paged_attention_forward"] = original


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-source-indices", type=int, nargs="+", required=True)
    parser.add_argument("--expected-workloads", type=int, required=True)
    parser.add_argument("--source-index-start", type=int, default=6)
    parser.add_argument("--source-index-end", type=int, default=9)
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--document-tokens", type=int, default=256)
    parser.add_argument("--query-tokens", type=int, default=32)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--append-page-size", type=int, default=16)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.05)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the real-model diagnostic")
    if args.output.exists():
        raise SystemExit(f"diagnostic output already exists: {args.output}")
    if "test-v2" in str(args.data) or "test_v2" in str(args.data):
        raise SystemExit("test-v2 paths are forbidden")
    if args.source_index_start != 6 or args.source_index_end != 9:
        raise SystemExit("diagnostic freezes validation source indices 6-9")
    if args.source_index_end >= 68:
        raise SystemExit("source >=68/test-v2 is forbidden")
    if args.expected_workloads != 8:
        raise SystemExit("diagnostic freezes eight validation workloads")
    actual_data_sha = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if actual_data_sha != args.expected_data_sha256:
        raise SystemExit("validation data SHA256 mismatch")
    if args.document_tokens != 256 or args.query_tokens != 32:
        raise SystemExit("diagnostic freezes document/query lengths to 256/32")

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    args.exclude_source_indices = (4, 5)
    args.allow_test_v2 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    workloads, workload_metadata = longbench_workloads(tokenizer, args)
    actual_indices = sorted({int(row["source_index"]) for row in workloads})
    if actual_indices != sorted(set(args.expected_source_indices)):
        raise SystemExit("source index hard gate failed")
    if len(workloads) != args.expected_workloads:
        raise SystemExit("workload count hard gate failed")
    if workload_metadata.get("source_revisions") != [args.expected_source_revision]:
        raise SystemExit("source revision hard gate failed")
    if workload_metadata.get("test_v2_consumed"):
        raise SystemExit("test-v2 was consumed")

    torch.cuda.set_device(0)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    workload = workloads[0]
    document = _batch_prefix(
        workload["document_tokens"], args.document_tokens
    ).cuda()
    query = _batch_prefix(workload["query_tokens"], args.query_tokens).cuda()
    if document.shape[-1] != args.document_tokens:
        raise RuntimeError("frozen diagnostic document is shorter than required")
    if query.shape[-1] < 1 or query.shape[-1] > args.query_tokens:
        raise RuntimeError("frozen diagnostic query has invalid length")
    source, install = _build_dense_document_cache(
        backbone, document, functional_linear=False
    )
    if install is not None:
        raise RuntimeError("stock diagnostic unexpectedly installed native cache")

    legacy_pair = clone_dense_and_prepare_paged_cache_pair(
        source,
        plan.full,
        page_size=args.page_size,
        bits=16,
        group_size=64,
        append_page_size=args.append_page_size,
    )
    fixed_pair = clone_dense_and_prepare_paged_cache_pair(
        source,
        plan.full,
        page_size=args.page_size,
        bits=16,
        group_size=64,
        append_page_size=args.append_page_size,
    )
    legacy_ledger = PagedAttentionHitLedger(plan.full, legacy_pair.conversion)
    fixed_ledger = PagedAttentionHitLedger(plan.full, fixed_pair.conversion)
    legacy_backend = register_qwen35_paged_backend(legacy_ledger)
    fixed_backend = register_qwen35_paged_backend(fixed_ledger)
    caller = _same_query_caller(backbone, model, query)
    indices = plan.full_attention_layer_indices

    with torch.inference_mode(), temporary_attention_implementation(
        backbone.config, "eager"
    ):
        eager_logits, eager_layers = _capture_attention(
            backbone,
            indices,
            lambda: _final_logits(caller(legacy_pair.dense_cache)),
        )
    with torch.inference_mode(), _replace_ledger_forward(
        legacy_ledger, legacy_fp32_paged_attention_forward
    ), temporary_attention_implementation(backbone.config, legacy_backend.name):
        legacy_logits, legacy_layers = _capture_attention(
            backbone,
            indices,
            lambda: _final_logits(caller(legacy_pair.paged_cache)),
        )
    legacy_intercept = legacy_ledger.verify_complete()
    with torch.inference_mode(), temporary_attention_implementation(
        backbone.config, fixed_backend.name
    ):
        fixed_logits, fixed_layers = _capture_attention(
            backbone,
            indices,
            lambda: _final_logits(caller(fixed_pair.paged_cache)),
        )
    fixed_intercept = fixed_ledger.verify_complete()

    rows = []
    for index in indices:
        rows.append(
            {
                "layer": index,
                "legacy_fp32_vs_eager": _metrics(
                    eager_layers[index], legacy_layers[index],
                    rtol=args.rtol, atol=args.atol,
                ),
                "two_pass_bf16_vs_eager": _metrics(
                    eager_layers[index], fixed_layers[index],
                    rtol=args.rtol, atol=args.atol,
                ),
            }
        )
    final = {
        "legacy_fp32_vs_eager": _metrics(
            eager_logits, legacy_logits, rtol=args.rtol, atol=args.atol
        ),
        "two_pass_bf16_vs_eager": _metrics(
            eager_logits, fixed_logits, rtol=args.rtol, atol=args.atol
        ),
        "legacy_token_exact": bool(
            torch.equal(eager_logits.argmax(-1), legacy_logits.argmax(-1))
        ),
        "two_pass_token_exact": bool(
            torch.equal(eager_logits.argmax(-1), fixed_logits.argmax(-1))
        ),
    }
    fixed_intercept_complete = bool(
        fixed_intercept.get("verified")
        and fixed_intercept.get("total_calls") == len(indices) == 10
    )
    fixed_gate_passed = bool(
        fixed_intercept_complete
        and final["two_pass_bf16_vs_eager"]["close"]
        and final["two_pass_token_exact"]
    )
    result = {
        "status": "diagnostic_complete",
        "formal_benchmark_authorized": False,
        "next_stage_authorized": fixed_gate_passed,
        "failed_formal_trial": 1833998,
        "model": str(args.model),
        "data": str(args.data),
        "data_sha256": actual_data_sha,
        "workload_metadata": workload_metadata,
        "workload_id": workload["workload_id"],
        "document_tokens": args.document_tokens,
        "query_tokens": args.query_tokens,
        "page_size": args.page_size,
        "append_page_size": args.append_page_size,
        "full_attention_layer_indices": indices,
        "per_layer_attention_output": rows,
        "final_logits": final,
        "legacy_intercept": legacy_intercept,
        "two_pass_intercept": fixed_intercept,
        "fixed_intercept_complete": fixed_intercept_complete,
        "fixed_gate_passed": fixed_gate_passed,
        "root_cause_hypothesis": (
            "legacy kept softmax weights and value accumulation in FP32; Qwen eager "
            "casts normalized softmax weights to BF16 before weight@value"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, result)
    print(json.dumps(final, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
