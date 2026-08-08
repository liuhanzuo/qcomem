from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
CONTEXT_LENGTHS = (512, 2048, 8192)
CORPUS = (
    "Edge language-model inference has two main phases. During prefill, the "
    "runtime processes input tokens and constructs the attention key-value cache. "
    "During decode, it generates one token at a time while reading model weights "
    "and the existing cache. Apple Silicon gives the CPU and GPU access to unified "
    "memory, but bandwidth, kernel efficiency, cache growth, and synchronization "
    "still determine latency. A reproducible benchmark separates model loading, "
    "time to first token, prompt throughput, decode throughput, and peak memory. "
)


def exact_prompt(tokenizer, target: int) -> list[int]:
    source = tokenizer.encode(CORPUS, add_special_tokens=False)
    repeats = (target + len(source) - 1) // len(source)
    return (source * repeats)[:target]


def run_generation(model, tokenizer, prompt: list[int], output_tokens: int) -> dict:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    started = time.perf_counter()
    first_token_at = None
    final = None
    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=output_tokens,
        sampler=make_sampler(temp=0.0),
    ):
        if first_token_at is None:
            first_token_at = time.perf_counter()
        final = response
    finished = time.perf_counter()
    if final is None or first_token_at is None:
        raise RuntimeError("generation produced no response")
    return {
        "ttft_seconds": first_token_at - started,
        "generation_wall_seconds": finished - started,
        "prompt_tokens": int(final.prompt_tokens),
        "prompt_tokens_per_second": float(final.prompt_tps),
        "generation_tokens": int(final.generation_tokens),
        "generation_tokens_per_second": float(final.generation_tps),
        "peak_memory_gb": float(final.peak_memory),
        "finish_reason": final.finish_reason,
    }


def metric_summary(rows: list[dict]) -> dict:
    metrics = (
        "ttft_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "peak_memory_gb",
    )
    grouped = {}
    for context in CONTEXT_LENGTHS:
        selected = [row for row in rows if row["context_tokens"] == context]
        grouped[str(context)] = {
            metric: {
                "median": statistics.median(row[metric] for row in selected),
                "min": min(row[metric] for row in selected),
                "max": max(row[metric] for row in selected),
            }
            for metric in metrics
        }
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description="Same-process MLX context benchmark")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=128)
    args = parser.parse_args()

    from mlx_lm import load

    load_started = time.perf_counter()
    model, tokenizer = load(args.model)
    load_seconds = time.perf_counter() - load_started

    print("Warm-up: 128 prompt tokens, 32 output tokens")
    run_generation(model, tokenizer, exact_prompt(tokenizer, 128), 32)

    rows = []
    for context in CONTEXT_LENGTHS:
        prompt = exact_prompt(tokenizer, context)
        for run_index in range(args.runs):
            print(f"Context {context}: run {run_index + 1}/{args.runs}")
            result = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "model": args.model,
                "context_tokens": context,
                "run_index": run_index,
                "model_load_seconds": load_seconds,
                **run_generation(model, tokenizer, prompt, args.output_tokens),
            }
            rows.append(result)
            print(json.dumps(result, indent=2))

    output = {
        "model": args.model,
        "model_load_seconds": load_seconds,
        "runs_per_context": args.runs,
        "requested_output_tokens": args.output_tokens,
        "results": rows,
        "summary": metric_summary(rows),
    }
    path = Path("results/context_benchmark.json")
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
