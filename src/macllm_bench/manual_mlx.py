from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlx.core as mx


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
DEFAULT_CONTEXT_LENGTHS = (512, 2048, 8192)
CORPUS = (
    "Edge language-model inference has two main phases. During prefill, the "
    "runtime processes input tokens and constructs the attention key-value cache. "
    "During decode, it generates one token at a time while reading model weights "
    "and the existing cache. Apple Silicon gives the CPU and GPU access to unified "
    "memory, but bandwidth, kernel efficiency, cache growth, and synchronization "
    "still determine latency. A reproducible benchmark separates model loading, "
    "time to first token, prompt throughput, decode throughput, and peak memory. "
)


class ManualKVCache:
    """A block-growing KV cache implemented directly with ``mlx.core`` arrays."""

    step = 256

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0

    def make_mask(
        self,
        length: int,
        return_array: bool = False,
        window_size: int | None = None,
    ) -> str | mx.array | None:
        if length == 1:
            return None
        if not return_array and window_size is None:
            return "causal"

        key_positions = mx.arange(self.offset + length)[None]
        query_positions = mx.arange(self.offset, self.offset + length)[:, None]
        mask = query_positions >= key_positions
        if window_size is not None:
            mask = mask & (query_positions < key_positions + window_size)
        return mask

    def update_and_fetch(
        self, keys: mx.array, values: mx.array
    ) -> tuple[mx.array, mx.array]:
        previous = self.offset
        incoming = keys.shape[2]
        required = previous + incoming

        if self.keys is None or required > self.keys.shape[2]:
            batch, heads, _, key_dim = keys.shape
            value_dim = values.shape[3]
            blocks = (incoming + self.step - 1) // self.step
            key_padding = mx.zeros(
                (batch, heads, blocks * self.step, key_dim), dtype=keys.dtype
            )
            value_padding = mx.zeros(
                (batch, heads, blocks * self.step, value_dim), dtype=values.dtype
            )
            if self.keys is None:
                self.keys = key_padding
                self.values = value_padding
            else:
                self.keys = mx.concatenate(
                    [self.keys[..., :previous, :], key_padding], axis=2
                )
                self.values = mx.concatenate(
                    [self.values[..., :previous, :], value_padding], axis=2
                )

        self.offset = required
        self.keys[..., previous:required, :] = keys
        self.values[..., previous:required, :] = values
        return self.keys[..., :required, :], self.values[..., :required, :]

    @property
    def state(self) -> tuple[mx.array, mx.array] | tuple[()]:
        if self.keys is None or self.values is None:
            return ()
        return self.keys, self.values

    @property
    def nbytes(self) -> int:
        if self.keys is None or self.values is None:
            return 0
        return self.keys.nbytes + self.values.nbytes


def exact_prompt(tokenizer, target: int) -> list[int]:
    source = tokenizer.encode(CORPUS, add_special_tokens=False)
    repeats = (target + len(source) - 1) // len(source)
    return (source * repeats)[:target]


def cache_nbytes(cache: list[ManualKVCache]) -> int:
    return sum(entry.nbytes for entry in cache)


def evaluate_cache(cache: list[ManualKVCache]) -> None:
    states = [entry.state for entry in cache if entry.state]
    if states:
        mx.eval(states)


def model_logits(model, tokens: mx.array, cache: list[ManualKVCache]) -> mx.array:
    logits = model(tokens[None], cache=cache)
    return logits[:, -1, :]


def manual_generate(
    model,
    tokenizer,
    prompt: list[int],
    output_tokens: int,
    prefill_step_size: int,
    gpu_stream: mx.Stream,
) -> dict[str, Any]:
    """Run prefill and greedy decode without MLX-LM generation helpers."""

    if not prompt:
        raise ValueError("prompt must contain at least one token")
    if output_tokens < 1:
        raise ValueError("output_tokens must be at least 1")
    if prefill_step_size < 1:
        raise ValueError("prefill_step_size must be at least 1")

    cache = [ManualKVCache() for _ in model.layers]
    prompt_array = mx.array(prompt)
    eos_token_ids = set(tokenizer.eos_token_ids)

    mx.synchronize(gpu_stream)
    mx.clear_cache()
    mx.reset_peak_memory()
    started = time.perf_counter()

    # Process all but the final prompt token in bounded chunks. Evaluating the
    # cache materializes each chunk while allowing the large logits tensor to be
    # discarded, which keeps long-prompt memory bounded.
    position = 0
    while len(prompt) - position > 1:
        remaining = len(prompt) - position - 1
        chunk_size = min(prefill_step_size, remaining)
        chunk = prompt_array[position : position + chunk_size]
        with mx.stream(gpu_stream):
            model_logits(model, chunk, cache)
        evaluate_cache(cache)
        position += chunk_size
        mx.clear_cache()

    # The last prompt token produces the first generated token. Calling item()
    # is an explicit CPU/GPU synchronization point and therefore defines TTFT.
    final_prompt_token = prompt_array[position:]
    with mx.stream(gpu_stream):
        logits = model_logits(model, final_prompt_token, cache)
        next_token = mx.argmax(logits, axis=-1)
    mx.async_eval(next_token)
    first_token_id = int(next_token.item())
    first_token_at = time.perf_counter()

    generated = [first_token_id]
    finish_reason = "stop" if first_token_id in eos_token_ids else "length"

    while len(generated) < output_tokens and finish_reason != "stop":
        token = mx.array([generated[-1]])
        with mx.stream(gpu_stream):
            logits = model_logits(model, token, cache)
            next_token = mx.argmax(logits, axis=-1)
        mx.async_eval(next_token)
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id in eos_token_ids:
            finish_reason = "stop"
        if len(generated) % 256 == 0:
            mx.clear_cache()

    mx.synchronize(gpu_stream)
    finished = time.perf_counter()
    prefill_seconds = first_token_at - started
    decode_seconds = finished - first_token_at

    return {
        "ttft_seconds": prefill_seconds,
        "generation_wall_seconds": finished - started,
        "prompt_tokens": len(prompt),
        "prompt_tokens_per_second": len(prompt) / prefill_seconds,
        "generation_tokens": len(generated),
        "generation_tokens_per_second": (
            len(generated) / decode_seconds if decode_seconds > 0 else None
        ),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "active_memory_gb": mx.get_active_memory() / 1e9,
        "kv_cache_gb": cache_nbytes(cache) / 1e9,
        "finish_reason": finish_reason,
        "generated_token_ids": generated,
        "generated_text": tokenizer.decode(generated),
    }


def metric_summary(rows: list[dict[str, Any]], context_lengths: list[int]) -> dict:
    metrics = (
        "ttft_seconds",
        "prompt_tokens_per_second",
        "generation_tokens_per_second",
        "peak_memory_gb",
        "kv_cache_gb",
    )
    grouped = {}
    for context in context_lengths:
        selected = [row for row in rows if row["context_tokens"] == context]
        grouped[str(context)] = {}
        for metric in metrics:
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            grouped[str(context)][metric] = {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicit mlx.core Apple-GPU context benchmark"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--prefill-step-size", type=int, default=2048)
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_LENGTHS),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/manual_mlx_context_benchmark.json"),
    )
    parser.add_argument("--skip-warmup", action="store_true")
    args = parser.parse_args()

    if args.runs < 1:
        raise SystemExit("--runs must be at least 1")
    if any(length < 2 for length in args.context_lengths):
        raise SystemExit("all context lengths must be at least 2")
    if not mx.metal.is_available():
        raise SystemExit("the MLX Metal backend is unavailable")

    # Make GPU placement explicit before loading/evaluating model parameters.
    mx.set_default_device(mx.gpu)
    gpu_stream = mx.new_stream(mx.gpu)
    device_info = mx.device_info(mx.gpu)

    from mlx_lm import load

    load_started = time.perf_counter()
    with mx.stream(gpu_stream):
        model, tokenizer = load(args.model)
    mx.synchronize(gpu_stream)
    load_seconds = time.perf_counter() - load_started

    if any(getattr(layer, "use_sliding", False) for layer in model.layers):
        raise SystemExit(
            "ManualKVCache currently supports full attention only, not sliding attention"
        )

    if not args.skip_warmup:
        print("Warm-up: 128 prompt tokens, 32 output tokens")
        manual_generate(
            model,
            tokenizer,
            exact_prompt(tokenizer, 128),
            output_tokens=32,
            prefill_step_size=args.prefill_step_size,
            gpu_stream=gpu_stream,
        )
        mx.clear_cache()

    rows = []
    for context in args.context_lengths:
        prompt = exact_prompt(tokenizer, context)
        for run_index in range(args.runs):
            print(f"Context {context}: run {run_index + 1}/{args.runs}")
            result = {
                "captured_at": datetime.now(timezone.utc).isoformat(),
                "runtime": "manual-mlx-core",
                "device": device_info.get("device_name", "Apple GPU"),
                "model": args.model,
                "context_tokens": context,
                "run_index": run_index,
                "model_load_seconds": load_seconds,
                **manual_generate(
                    model,
                    tokenizer,
                    prompt,
                    output_tokens=args.output_tokens,
                    prefill_step_size=args.prefill_step_size,
                    gpu_stream=gpu_stream,
                ),
            }
            rows.append(result)
            print(json.dumps(result, indent=2))

    output = {
        "runtime": "manual-mlx-core",
        "device": device_info,
        "model": args.model,
        "model_load_seconds": load_seconds,
        "runs_per_context": args.runs,
        "requested_output_tokens": args.output_tokens,
        "prefill_step_size": args.prefill_step_size,
        "context_lengths": args.context_lengths,
        "results": rows,
        "summary": metric_summary(rows, args.context_lengths),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
