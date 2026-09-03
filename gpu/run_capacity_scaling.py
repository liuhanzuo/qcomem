from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from qcomem_torch import (
    PackedCache,
    PackedResidual,
    TorchSplitCausalLM,
    cache_nbytes,
    greedy_generate_dense,
    greedy_generate_full_prefix,
    greedy_generate_replay,
    tensor_nbytes,
)
from run_downstream import atomic_json
from run_replay_diagnostic import timed_gpu


LENGTH_BY_RANK = (4096, 8192, 16384, 4096, 8192, 16384, 4096, 8192)
CONFIG_SUITES = {
    "extreme": (
        "dense",
        "prefix",
        "replay-d7-q16",
        "replay-d7-r4-a4-l4",
        "replay-d10-r4-a4-l4",
        "replay-d13-r4-a4-l4",
    ),
    "quality": (
        "dense",
        "prefix",
        "replay-d7-r4-a8-l8",
        "replay-d10-r4-a8-l8",
        "replay-d13-r4-a8-l8",
    ),
}
CONFIGS = CONFIG_SUITES["extreme"]


def repeated_document(tokenizer, length: int, device: torch.device) -> torch.Tensor:
    paragraph = (
        "This is a reusable on-device knowledge document. It contains stable "
        "technical evidence, measurements, definitions, and implementation notes. "
    )
    base = tokenizer(paragraph, add_special_tokens=False, return_tensors="pt").input_ids
    repeats = (length + base.shape[1] - 1) // base.shape[1]
    return base.repeat(1, repeats)[:, :length].to(device)


def parse_capacity_config(
    name: str,
) -> tuple[str, int | None, int | None, int | None, int | None]:
    if name in {"dense", "prefix"}:
        return name, None, None, None, None
    _, raw_depth, *fields = name.split("-")
    depth = int(raw_depth[1:])
    residual_bits = None
    attention_bits = None
    linear_bits = None
    for field in fields:
        value = int(field[1:])
        if field.startswith(("q", "r")):
            residual_bits = value
        elif field.startswith("a"):
            attention_bits = value
        elif field.startswith("l"):
            linear_bits = value
    return "replay", depth, residual_bits, attention_bits, linear_bits


@torch.inference_mode()
def measure_config(
    *,
    name: str,
    adapter: TorchSplitCausalLM,
    document_ids: torch.Tensor,
    query_ids: torch.Tensor,
    model_allocated_bytes: int,
) -> dict[str, Any]:
    mode, depth, residual_bits, attention_bits, linear_bits = parse_capacity_config(
        name
    )
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    write_seconds = None
    residual_nbytes = None
    state_nbytes = None
    persistent_nbytes = None

    if mode == "dense":
        _, generation_seconds = timed_gpu(
            lambda: greedy_generate_dense(
                adapter,
                torch.cat([document_ids, query_ids], dim=1),
                max_new_tokens=1,
                eos_token_ids=set(),
            )
        )
    elif mode == "prefix":
        state, write_seconds = timed_gpu(
            lambda: adapter.write_full_prefix(document_ids)
        )
        state_nbytes = cache_nbytes(state.cache)
        persistent_nbytes = state.stored_nbytes
        _, generation_seconds = timed_gpu(
            lambda: greedy_generate_full_prefix(
                adapter,
                state,
                query_ids,
                max_new_tokens=1,
                eos_token_ids=set(),
            )
        )
    else:
        assert depth is not None
        raw_state, write_seconds = timed_gpu(
            lambda: adapter.write_lower_replay(document_ids, depth)
        )
        if any(
            value is not None
            for value in (residual_bits, attention_bits, linear_bits)
        ):
            state, quantize_seconds = timed_gpu(
                lambda: raw_state.quantize(
                    bits=residual_bits or 16,
                    attention_bits=attention_bits,
                    linear_bits=linear_bits,
                    group_size=64,
                )
            )
            write_seconds += quantize_seconds
            del raw_state
        else:
            state = raw_state
        residual_nbytes = (
            state.document_residual.nbytes
            if isinstance(state.document_residual, PackedResidual)
            else tensor_nbytes(state.document_residual)
        )
        state_nbytes = (
            state.cache.nbytes
            if isinstance(state.cache, PackedCache)
            else cache_nbytes(state.cache)
        )
        persistent_nbytes = state.stored_nbytes
        _, generation_seconds = timed_gpu(
            lambda: greedy_generate_replay(
                adapter,
                state,
                query_ids,
                max_new_tokens=1,
                eos_token_ids=set(),
            )
        )

    peak = torch.cuda.max_memory_allocated()
    row = {
        "config": name,
        "mode": mode,
        "depth": depth,
        "residual_bits": residual_bits,
        "attention_bits": attention_bits,
        "linear_bits": linear_bits,
        "document_tokens": int(document_ids.numel()),
        "query_tokens": int(query_ids.numel()),
        "write_seconds": write_seconds,
        "generation_seconds": generation_seconds,
        "stored_residual_nbytes": residual_nbytes,
        "stored_state_nbytes": state_nbytes,
        "stored_persistent_nbytes": persistent_nbytes,
        "model_allocated_bytes": model_allocated_bytes,
        "peak_allocated_bytes": peak,
        "incremental_peak_allocated_bytes": max(peak - model_allocated_bytes, 0),
    }
    if mode != "dense":
        del state
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument(
        "--suite", choices=sorted(CONFIG_SUITES), default="extreme"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.rank < 0 or args.rank >= len(LENGTH_BY_RANK):
        raise SystemExit("rank must be in [0, 7]")

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    torch.cuda.synchronize()
    model_allocated_bytes = torch.cuda.memory_allocated()
    length = LENGTH_BY_RANK[args.rank]
    document_ids = repeated_document(tokenizer, length, torch.device("cuda"))
    query_ids = tokenizer(
        "\nQuestion: What kind of document is described?\nAnswer:",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.cuda()
    adapter = TorchSplitCausalLM(model)
    configs = CONFIG_SUITES[args.suite]
    rows = []
    for name in configs:
        row = measure_config(
            name=name,
            adapter=adapter,
            document_ids=document_ids,
            query_ids=query_ids,
            model_allocated_bytes=model_allocated_bytes,
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = {
        "rank": args.rank,
        "document_tokens": length,
        "suite": args.suite,
        "configs": list(configs),
        "rows": rows,
    }
    atomic_json(args.run_dir / f"capacity-rank-{args.rank}.json", result)


def aggregate(run_dir: Path) -> dict[str, Any]:
    shards = [
        json.loads(path.read_text())
        for path in sorted(run_dir.glob("capacity-rank-*.json"))
    ]
    if len(shards) != 8:
        raise ValueError(f"expected 8 shards, found {len(shards)}")
    rows = [row for shard in shards for row in shard["rows"]]
    configs = tuple(shards[0]["configs"])
    if any(tuple(shard["configs"]) != configs for shard in shards):
        raise ValueError("capacity shards disagree on config suite")
    summary = []
    for length in sorted(set(row["document_tokens"] for row in rows)):
        for name in configs:
            matching = [
                row
                for row in rows
                if row["document_tokens"] == length and row["config"] == name
            ]
            state_rows = [
                row for row in matching if row["stored_persistent_nbytes"] is not None
            ]
            summary.append(
                {
                    "document_tokens": length,
                    "config": name,
                    "repetitions": len(matching),
                    "mean_write_seconds": (
                        statistics.fmean(row["write_seconds"] for row in state_rows)
                        if state_rows
                        else None
                    ),
                    "mean_generation_seconds": statistics.fmean(
                        row["generation_seconds"] for row in matching
                    ),
                    "mean_persistent_mib": (
                        statistics.fmean(
                            row["stored_persistent_nbytes"] for row in state_rows
                        )
                        / 2**20
                        if state_rows
                        else None
                    ),
                    "mean_incremental_peak_gib": statistics.fmean(
                        row["incremental_peak_allocated_bytes"] for row in matching
                    )
                    / 2**30,
                }
            )
    return {
        "status": "completed",
        "suite": shards[0].get("suite", "extreme"),
        "rows": rows,
        "summary": summary,
    }


if __name__ == "__main__":
    main()
