from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from qcomem_torch import (
    PackedLowerReplayState,
    TorchSplitCausalLM,
    cache_nbytes,
    quantize_residual,
    quantize_transformers_cache,
)
from run_downstream import atomic_json, load_samples, prompt_parts


BITS = (2, 4, 8, 16)


@torch.inference_mode()
def replay_last_logits(
    adapter: TorchSplitCausalLM,
    state,
    query_ids: torch.Tensor,
) -> torch.Tensor:
    local = state.fork()
    query_residual = adapter.continue_lower_replay(local, query_ids)
    return adapter.run_suffix_last_logits(
        [local.document_residual, query_residual], local.depth
    )


def logit_metrics(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | bool]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    squared_error = torch.square(reference_float - candidate_float).sum()
    reference_squared = torch.square(reference_float).sum()
    probability = torch.softmax(reference_float, dim=-1)
    kl = F.kl_div(
        torch.log_softmax(candidate_float, dim=-1),
        probability,
        reduction="batchmean",
    )
    exact_token = int(torch.argmax(reference_float, dim=-1).item())
    candidate_token = int(torch.argmax(candidate_float, dim=-1).item())
    return {
        "relative_logit_mse": float(
            (squared_error / torch.clamp(reference_squared, min=1e-30)).item()
        ),
        "kl_divergence": max(float(kl.item()), 0.0),
        "top1_match": exact_token == candidate_token,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate one Q-CoMem replay component on one H20"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--limit-per-dataset", type=int, default=2)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.world_size != args.depth + 1:
        raise SystemExit("this calibration assigns residual plus one cache layer per GPU")
    if args.rank < 0 or args.rank >= args.world_size:
        raise SystemExit("rank is outside world size")

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    component = "residual" if args.rank == 0 else f"cache.{args.rank - 1}"
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    adapter = TorchSplitCausalLM(model)
    samples = load_samples(args.data, args.limit_per_dataset)
    rows: list[dict[str, Any]] = []
    load_complete = time.perf_counter()

    for sample in samples:
        document_ids, query_ids, *_ = prompt_parts(
            tokenizer, sample, args.max_input_tokens
        )
        document_ids = document_ids.cuda()
        query_ids = query_ids.cuda()
        raw = adapter.write_lower_replay(document_ids, args.depth)
        reference_logits = replay_last_logits(adapter, raw, query_ids)
        layer_count = len(raw.cache.layers)
        cache_index = args.rank - 1
        is_linear = (
            None
            if component == "residual"
            else hasattr(raw.cache.layers[cache_index], "conv_states")
            or hasattr(raw.cache.layers[cache_index], "recurrent_states")
        )
        options = []
        for bits in BITS:
            if component == "residual":
                packed_residual = quantize_residual(
                    raw.document_residual,
                    bits=bits,
                    group_size=args.group_size,
                )
                candidate = PackedLowerReplayState(
                    depth=raw.depth,
                    document_length=raw.document_length,
                    current_length=raw.current_length,
                    document_residual=packed_residual,
                    cache=raw.cache,
                )
                component_nbytes = packed_residual.nbytes
            else:
                layer_bits = [16] * layer_count
                layer_bits[cache_index] = bits
                packed_cache = quantize_transformers_cache(
                    raw.cache,
                    attention_bits=16,
                    linear_bits=16,
                    cache_layer_bits=layer_bits,
                    group_size=args.group_size,
                )
                packed_residual = quantize_residual(
                    raw.document_residual,
                    bits=16,
                    group_size=args.group_size,
                )
                candidate = PackedLowerReplayState(
                    depth=raw.depth,
                    document_length=raw.document_length,
                    current_length=raw.current_length,
                    document_residual=packed_residual,
                    cache=packed_cache,
                )
                component_nbytes = cache_nbytes(
                    packed_cache.cache.layers[cache_index]
                )
            candidate_logits = replay_last_logits(adapter, candidate, query_ids)
            metrics = logit_metrics(reference_logits, candidate_logits)
            options.append(
                {
                    "bits": bits,
                    "component_nbytes": component_nbytes,
                    **metrics,
                }
            )
            del candidate, candidate_logits, packed_residual
            if component != "residual":
                del packed_cache
        rows.append(
            {
                "dataset": sample["dataset"],
                "id": sample.get("_id"),
                "source_index": sample.get("_source_index"),
                "document_tokens": int(document_ids.numel()),
                "query_tokens": int(query_ids.numel()),
                "cache_layers": layer_count,
                "is_linear": is_linear,
                "options": options,
            }
        )
        del raw, reference_logits, document_ids, query_ids
        torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "component": component,
                    "dataset": sample["dataset"],
                    "source_index": sample.get("_source_index"),
                }
            ),
            flush=True,
        )

    summary_options = []
    for bits in BITS:
        selected = [
            option
            for row in rows
            for option in row["options"]
            if option["bits"] == bits
        ]
        summary_options.append(
            {
                "bits": bits,
                "mean_component_nbytes": round(
                    statistics.fmean(item["component_nbytes"] for item in selected)
                ),
                "mean_kl_divergence": statistics.fmean(
                    item["kl_divergence"] for item in selected
                ),
                "mean_relative_logit_mse": statistics.fmean(
                    item["relative_logit_mse"] for item in selected
                ),
                "top1_match_rate": statistics.fmean(
                    item["top1_match"] for item in selected
                ),
            }
        )
    result = {
        "status": "completed",
        "rank": args.rank,
        "world_size": args.world_size,
        "component": component,
        "depth": args.depth,
        "is_linear": rows[0]["is_linear"],
        "samples": len(rows),
        "options": summary_options,
        "rows": rows,
        "model": str(args.model),
        "data": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "max_input_tokens": args.max_input_tokens,
        "group_size": args.group_size,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "elapsed_seconds": time.perf_counter() - load_complete,
    }
    atomic_json(args.run_dir / f"sensitivity-{args.rank}.json", result)
    print(json.dumps({"saved": component, "options": summary_options}), flush=True)


if __name__ == "__main__":
    main()
