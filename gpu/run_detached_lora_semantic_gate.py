from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qcomem_lora import (
    PG19WindowDataset,
    assert_replay_adapter_semantics,
    load_inference_lora_checkpoint,
    quant_student_suffix_hidden,
    set_lora_enabled,
)
from qcomem_torch import TorchSplitCausalLM
from run_downstream import atomic_json
from train_qcomem_lora import reject_frozen_test_data


def parse_bits(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in raw.split(","))


@torch.no_grad()
def compare_detached_one(
    adapter,
    packed,
    query_ids: torch.Tensor,
    *,
    projection_block_size: int = 16,
) -> dict[str, Any]:
    if query_ids.shape[1] == 0:
        raise ValueError("query_ids must contain at least one token")
    if projection_block_size < 1:
        raise ValueError("projection_block_size must be positive")
    state = packed.fork()
    query_residual = adapter.continue_lower_replay(state, query_ids)
    detached_hidden, cache_audit = quant_student_suffix_hidden(
        adapter,
        depth=state.depth,
        document_residual=state.document_residual,
        query_residual=query_residual,
        execution="detached-document-cache",
        return_cache_audit=True,
    )

    deployment_cache = adapter.make_cache()
    adapter.run_suffix_cached_last_logits(
        [state.document_residual],
        state.depth,
        deployment_cache,
        position_offset=0,
    )
    deployment_hidden = adapter._run_layers(
        query_residual,
        state.depth,
        adapter.num_layers,
        past_key_values=deployment_cache,
        position_offset=state.document_length,
    )
    deployment_hidden = adapter.language_model.norm(deployment_hidden)
    if detached_hidden.shape != deployment_hidden.shape:
        raise RuntimeError("detached/deployment hidden shapes differ")

    positions = []
    for start in range(0, query_ids.shape[1], projection_block_size):
        end = min(start + projection_block_size, query_ids.shape[1])
        detached_logits = adapter.lm_head(detached_hidden[:, start:end]).float()
        deployment_logits = adapter.lm_head(deployment_hidden[:, start:end]).float()
        probability = torch.softmax(detached_logits, dim=-1)
        kl = F.kl_div(
            torch.log_softmax(deployment_logits, dim=-1),
            probability,
            reduction="none",
        ).sum(dim=-1).mean(dim=0)
        error = (detached_logits - deployment_logits).abs().amax(dim=(0, 2))
        top1 = (
            detached_logits.argmax(dim=-1) == deployment_logits.argmax(dim=-1)
        ).all(dim=0)
        for offset, position in enumerate(range(start, end)):
            positions.append(
                {
                    "position": position,
                    "kl_detached_to_deployment": max(float(kl[offset].item()), 0.0),
                    "max_abs_logit_error": float(error[offset].item()),
                    "top1_match": bool(top1[offset].item()),
                }
            )
    return {
        "query_positions": len(positions),
        "position_top1_match_rate": statistics.fmean(
            row["top1_match"] for row in positions
        ),
        "mean_position_kl": statistics.fmean(
            row["kl_detached_to_deployment"] for row in positions
        ),
        "max_position_kl": max(
            row["kl_detached_to_deployment"] for row in positions
        ),
        "max_abs_logit_error": max(
            row["max_abs_logit_error"] for row in positions
        ),
        "cache_immutability": cache_audit,
        "positions": positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gate detached-document-cache LoRA all-query forward semantics "
            "against mutable deployment continuation"
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--residual-bits", type=int, default=4)
    parser.add_argument("--attention-bits", type=int, default=4)
    parser.add_argument("--linear-bits", type=int, default=8)
    parser.add_argument("--cache-layer-bits", default="8,8,8,4,8,8,8")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--query-tokens", type=int, default=128)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--min-top1-match", type=float, default=1.0)
    parser.add_argument("--max-mean-kl", type=float, default=1e-6)
    parser.add_argument("--max-logit-error", type=float, default=0.0)
    parser.add_argument("--projection-block-size", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("rank must be within world size")
    data_sha256 = reject_frozen_test_data(args.data)
    layer_bits = parse_bits(args.cache_layer_bits)

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = PG19WindowDataset(
        args.data,
        tokenizer,
        context_tokens=args.context_tokens,
        query_tokens=args.query_tokens,
        stride=args.context_tokens + args.query_tokens,
        limit=args.samples,
        max_windows_per_record=1,
    )
    windows = dataset.windows[args.rank :: args.world_size]
    if not windows:
        raise SystemExit("detached semantic rank has no window")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    split = TorchSplitCausalLM(model)
    checkpoint = load_inference_lora_checkpoint(model, split.layers, args.checkpoint)
    assert_replay_adapter_semantics(
        checkpoint,
        depth=args.depth,
        residual_bits=args.residual_bits,
        attention_bits=args.attention_bits,
        linear_bits=args.linear_bits,
        cache_layer_bits=layer_bits,
    )
    if checkpoint["semantics"].get("student_suffix_execution_option") != (
        "detached-document-cache"
    ):
        raise SystemExit("checkpoint is not detached-document-cache LoRA")
    model.eval().cuda()
    set_lora_enabled(model, True)
    adapter = TorchSplitCausalLM(model)
    rows = []
    for window in windows:
        document = window.document_ids.cuda().unsqueeze(0)
        query = window.query_ids.cuda().unsqueeze(0)
        raw = adapter.write_lower_replay(document, args.depth)
        packed = raw.quantize(
            bits=args.residual_bits,
            attention_bits=args.attention_bits,
            linear_bits=args.linear_bits,
            cache_layer_bits=layer_bits,
            group_size=args.group_size,
        )
        rows.append(
            {
                "source_id": window.source_id,
                **compare_detached_one(
                    adapter,
                    packed,
                    query,
                    projection_block_size=args.projection_block_size,
                ),
            }
        )
        del raw, packed, document, query
        torch.cuda.empty_cache()
    positions = [position for row in rows for position in row["positions"]]
    top1 = statistics.fmean(position["top1_match"] for position in positions)
    mean_kl = statistics.fmean(
        position["kl_detached_to_deployment"] for position in positions
    )
    max_error = max(row["max_abs_logit_error"] for row in rows)
    cache_gate = all(
        row["cache_immutability"].get("hard_gate_passed") is True for row in rows
    )
    local_passed = (
        top1 >= args.min_top1_match
        and mean_kl <= args.max_mean_kl
        and max_error <= args.max_logit_error
        and cache_gate
    )
    result = {
        "status": (
            "completed_shard"
            if args.world_size > 1
            else ("passed" if local_passed else "failed")
        ),
        "local_threshold_passed": local_passed,
        "rank": args.rank,
        "world_size": args.world_size,
        "claim": "query-continuation-only detached training forward semantic gate",
        "training_suffix_execution": (
            "cached_document_prefill_detached_then_full_query_continuation"
        ),
        "deployment_suffix_execution": (
            "cached_document_prefill_then_full_query_continuation"
        ),
        "comparison_scope": "all_query_positions",
        "samples": len(rows),
        "global_samples_requested": args.samples,
        "query_positions": len(positions),
        "position_top1_match_rate": top1,
        "mean_position_kl_detached_to_deployment": mean_kl,
        "max_position_kl_detached_to_deployment": max(
            position["kl_detached_to_deployment"] for position in positions
        ),
        "max_abs_logit_error": max_error,
        "cache_immutability_gate_passed": cache_gate,
        "thresholds": {
            "min_top1_match": args.min_top1_match,
            "max_mean_kl": args.max_mean_kl,
            "max_logit_error": args.max_logit_error,
        },
        "data": str(args.data),
        "data_sha256": data_sha256,
        "test_v2_used": False,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "checkpoint_metadata": checkpoint,
        "rows": rows,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False))
    if args.world_size == 1 and result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
