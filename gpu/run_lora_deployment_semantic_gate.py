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
    set_lora_enabled,
)
from qcomem_torch import TorchSplitCausalLM
from run_downstream import atomic_json
from train_qcomem_lora import reject_frozen_test_data


def parse_bits(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in raw.split(","))


@torch.inference_mode()
def compare_one(
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
    # Isolate only the suffix document/query chunk boundary.  Both sides use
    # the same deployment lower-layer semantics: the complete initial query is
    # continued through the lower cache in one call.  Token-wise lower replay
    # belongs to later autoregressive decode and is deliberately not part of
    # this training/deployment gate.
    state = packed.fork()
    query_residual = adapter.continue_lower_replay(state, query_ids)
    training_hidden = torch.cat(
        [state.document_residual, query_residual], dim=1
    )
    training_hidden = adapter._run_layers(
        training_hidden, state.depth, adapter.num_layers
    )
    training_hidden = adapter.language_model.norm(
        training_hidden[:, -query_ids.shape[1] :, :]
    )

    suffix_cache = adapter.make_cache()
    adapter.run_suffix_cached_last_logits(
        [state.document_residual],
        state.depth,
        suffix_cache,
        position_offset=0,
    )
    deployment_hidden = adapter._run_layers(
        query_residual,
        state.depth,
        adapter.num_layers,
        past_key_values=suffix_cache,
        position_offset=state.document_length,
    )
    deployment_hidden = adapter.language_model.norm(deployment_hidden)
    position_rows = []
    for block_start in range(0, query_ids.shape[1], projection_block_size):
        block_end = min(block_start + projection_block_size, query_ids.shape[1])
        deployment_logits = adapter.lm_head(
            deployment_hidden[:, block_start:block_end, :]
        ).float()
        # A small position block amortizes the large-vocabulary projection
        # GEMM while avoiding a materialized [full query, vocab] tensor.
        training_logits = adapter.lm_head(
            training_hidden[:, block_start:block_end, :]
        ).float()
        probability = torch.softmax(training_logits, dim=-1)
        kl_by_batch_position = F.kl_div(
            torch.log_softmax(deployment_logits, dim=-1),
            probability,
            reduction="none",
        ).sum(dim=-1)
        kl_by_position = kl_by_batch_position.mean(dim=0)
        max_error_by_position = (
            (training_logits - deployment_logits).abs().amax(dim=(0, 2))
        )
        top1_by_position = (
            training_logits.argmax(-1) == deployment_logits.argmax(-1)
        ).all(dim=0)
        for block_position, position in enumerate(range(block_start, block_end)):
            position_rows.append(
                {
                    "position": position,
                    "kl_training_to_deployment": max(
                        float(kl_by_position[block_position].item()), 0.0
                    ),
                    "max_abs_logit_error": float(
                        max_error_by_position[block_position].item()
                    ),
                    "top1_match": bool(top1_by_position[block_position].item()),
                }
            )
    return {
        "query_positions": len(position_rows),
        "position_top1_match_rate": statistics.fmean(
            row["top1_match"] for row in position_rows
        ),
        "mean_position_kl": statistics.fmean(
            row["kl_training_to_deployment"] for row in position_rows
        ),
        "max_position_kl": max(
            row["kl_training_to_deployment"] for row in position_rows
        ),
        "max_abs_logit_error": max(
            row["max_abs_logit_error"] for row in position_rows
        ),
        "positions": position_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gate uncached Quant-LoRA training semantics against two-stage deployment"
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
    parser.add_argument("--context-tokens", type=int, default=1792)
    parser.add_argument("--query-tokens", type=int, default=256)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--min-top1-match", type=float, default=1.0)
    parser.add_argument("--max-mean-kl", type=float, default=1e-3)
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
        raise SystemExit("semantic-gate rank has no window shard")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    split = TorchSplitCausalLM(model)
    checkpoint = load_inference_lora_checkpoint(
        model, split.layers, args.checkpoint
    )
    assert_replay_adapter_semantics(
        checkpoint,
        depth=args.depth,
        residual_bits=args.residual_bits,
        attention_bits=args.attention_bits,
        linear_bits=args.linear_bits,
        cache_layer_bits=layer_bits,
    )
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
                **compare_one(
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
    mean_kl = statistics.fmean(
        position["kl_training_to_deployment"] for position in positions
    )
    top1 = statistics.fmean(position["top1_match"] for position in positions)
    local_threshold_passed = (
        top1 >= args.min_top1_match and mean_kl <= args.max_mean_kl
    )
    result = {
        "status": (
            "completed_shard"
            if args.world_size > 1
            else ("passed" if local_threshold_passed else "failed")
        ),
        "local_threshold_passed": local_threshold_passed,
        "rank": args.rank,
        "world_size": args.world_size,
        "claim": "training and deployment suffix execution are not assumed equivalent",
        "training_suffix_execution": "uncached_full_document_plus_query_sequence",
        "deployment_suffix_execution": (
            "cached_document_prefill_then_full_query_continuation"
        ),
        "comparison_scope": "all_query_positions",
        "projection_block_size": args.projection_block_size,
        "samples": len(rows),
        "global_samples_requested": args.samples,
        "query_positions": len(positions),
        "position_top1_match_rate": top1,
        "mean_position_kl_training_to_deployment": mean_kl,
        "max_position_kl_training_to_deployment": max(
            position["kl_training_to_deployment"] for position in positions
        ),
        "max_abs_logit_error": max(row["max_abs_logit_error"] for row in rows),
        "thresholds": {
            "min_top1_match": args.min_top1_match,
            "max_mean_kl": args.max_mean_kl,
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
    print(json.dumps(result, ensure_ascii=False, indent=2))
    # In distributed mode the global aggregate owns the hard decision.  A
    # locally failed shard remains valid evidence and must not disappear.
    if args.world_size == 1 and result["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
