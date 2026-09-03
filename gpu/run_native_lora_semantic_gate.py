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


@torch.no_grad()
def compare_native_one(
    adapter: TorchSplitCausalLM,
    packed: Any,
    query_ids: torch.Tensor,
    *,
    projection_block_size: int,
) -> dict[str, Any]:
    state = packed.fork()
    query_residual = adapter.continue_lower_replay(state, query_ids)
    functional, audit = quant_student_suffix_hidden(
        adapter,
        depth=state.depth,
        document_residual=state.document_residual,
        query_residual=query_residual,
        execution="native-functional-cache",
        return_cache_audit=True,
    )
    mutable = quant_student_suffix_hidden(
        adapter,
        depth=state.depth,
        document_residual=state.document_residual,
        query_residual=query_residual,
        execution="cached-two-stage",
    )
    if functional.shape != mutable.shape:
        raise RuntimeError("functional/mutable hidden shapes differ")
    positions = []
    for start in range(0, query_ids.shape[1], projection_block_size):
        end = min(start + projection_block_size, query_ids.shape[1])
        functional_logits = adapter.lm_head(functional[:, start:end]).float()
        mutable_logits = adapter.lm_head(mutable[:, start:end]).float()
        probability = torch.softmax(functional_logits, dim=-1)
        kl = F.kl_div(
            torch.log_softmax(mutable_logits, dim=-1),
            probability,
            reduction="none",
        ).sum(dim=-1).mean(dim=0)
        error = (functional_logits - mutable_logits).abs().amax(dim=(0, 2))
        top1 = (
            functional_logits.argmax(dim=-1) == mutable_logits.argmax(dim=-1)
        ).all(dim=0)
        for offset, position in enumerate(range(start, end)):
            positions.append(
                {
                    "position": position,
                    "kl_functional_to_mutable": max(float(kl[offset].item()), 0.0),
                    "max_abs_logit_error": float(error[offset].item()),
                    "top1_match": bool(top1[offset].item()),
                }
            )
    return {
        "query_positions": len(positions),
        "cache_audit": audit,
        "positions": positions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--projection-block-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 0 <= args.rank < args.world_size:
        raise SystemExit("CUDA and a valid rank are required")
    data_sha = reject_frozen_test_data(args.data)
    checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    if data_sha != args.expected_data_sha256:
        raise SystemExit("semantic data SHA256 mismatch")
    if checkpoint_sha != args.expected_checkpoint_sha256:
        raise SystemExit("semantic checkpoint SHA256 mismatch")

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = PG19WindowDataset(
        args.data,
        tokenizer,
        context_tokens=1536,
        query_tokens=512,
        stride=2048,
        limit=args.samples,
    )
    windows = dataset.windows[args.rank :: args.world_size]
    if not windows:
        raise SystemExit("semantic rank has no example")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    adapter = TorchSplitCausalLM(model)
    checkpoint = load_inference_lora_checkpoint(model, adapter.layers, args.checkpoint)
    assert_replay_adapter_semantics(
        checkpoint,
        depth=7,
        residual_bits=4,
        attention_bits=4,
        linear_bits=8,
        cache_layer_bits=(8, 8, 8, 4, 8, 8, 8),
    )
    if checkpoint["semantics"].get("student_suffix_execution_option") != (
        "native-functional-cache"
    ):
        raise SystemExit("checkpoint is not native-functional-cache trained")
    model.eval().cuda()
    set_lora_enabled(model, True)
    adapter = TorchSplitCausalLM(model)
    rows = []
    for window in windows:
        document = window.document_ids.cuda().unsqueeze(0)
        query = window.query_ids.cuda().unsqueeze(0)
        raw = adapter.write_lower_replay(document, 7)
        packed = raw.quantize(
            bits=4,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=(8, 8, 8, 4, 8, 8, 8),
            group_size=64,
        )
        rows.append(
            {
                "example_id": window.source_id,
                **compare_native_one(
                    adapter,
                    packed,
                    query,
                    projection_block_size=args.projection_block_size,
                ),
            }
        )
        del document, query, raw, packed
        torch.cuda.empty_cache()
    positions = [position for row in rows for position in row["positions"]]
    result = {
        "status": "completed_shard",
        "rank": args.rank,
        "world_size": args.world_size,
        "comparison_scope": "all_query_positions",
        "functional_execution": "native_functional_same_document_query_boundary",
        "mutable_execution": "standard_mutable_same_document_query_boundary",
        "samples": len(rows),
        "global_samples_requested": args.samples,
        "query_positions": len(positions),
        "position_top1_match_rate": statistics.fmean(
            position["top1_match"] for position in positions
        ),
        "mean_position_kl_functional_to_mutable": statistics.fmean(
            position["kl_functional_to_mutable"] for position in positions
        ),
        "max_position_kl_functional_to_mutable": max(
            position["kl_functional_to_mutable"] for position in positions
        ),
        "max_abs_logit_error": max(
            position["max_abs_logit_error"] for position in positions
        ),
        "cache_gate_passed": all(
            row["cache_audit"].get("hard_gate_passed") is True for row in rows
        ),
        "thresholds": {"min_top1_match": 1.0, "max_mean_kl": 1e-6},
        "data_sha256": data_sha,
        "checkpoint_sha256": checkpoint_sha,
        "test_v2_used": False,
        "single_token_autograd_claimed": False,
        "rows": rows,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}))


if __name__ == "__main__":
    main()
