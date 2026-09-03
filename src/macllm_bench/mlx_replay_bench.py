from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import mlx.core as mx
import psutil

from .comem_model import SplitCausalLM
from .experiment_env import (
    assess_completed_run,
    assess_preflight,
    collect_experiment_snapshot,
)
from .manual_mlx import CORPUS
from .mlx_replay import (
    FROZEN_REPLAY_CONFIG,
    FullPrefixState,
    LowerReplayState,
    PackedLowerReplayState,
    greedy_generate_dense,
    greedy_generate_full_prefix,
    greedy_generate_replay,
    packed_state_error_sums,
    relative_rmse,
    write_full_prefix,
    write_lower_replay,
)


DEFAULT_MODEL = "mlx-community/Qwen3.5-9B-4bit"
DEFAULT_MODEL_REVISION = "8b2b98c00a6b4d291155e4890773ca8f769aee53"
DEFAULT_CONTEXT_LENGTHS = (512, 2048, 4096)
DEFAULT_QUERY = (
    "\nQuestion: What two phases determine edge language-model inference?\n"
    "Answer:"
)
T = TypeVar("T")


def exact_document(tokenizer, target: int) -> mx.array:
    source = tokenizer.encode(CORPUS, add_special_tokens=False)
    if not source:
        raise ValueError("tokenizer produced an empty reusable document")
    repeats = (target + len(source) - 1) // len(source)
    return mx.array((source * repeats)[:target])


def _evaluate(value: Any) -> None:
    if isinstance(value, (LowerReplayState, FullPrefixState)):
        if isinstance(value, LowerReplayState):
            mx.eval(value.document_residual)
        from .mlx_replay import evaluate_cache

        evaluate_cache(value.cache)
    elif isinstance(value, PackedLowerReplayState):
        value.eval()
    elif isinstance(value, mx.array):
        mx.eval(value)


def timed(operation: Callable[[], T], stream: mx.Stream) -> tuple[T, float]:
    mx.synchronize(stream)
    started = time.perf_counter()
    value = operation()
    _evaluate(value)
    mx.synchronize(stream)
    return value, time.perf_counter() - started


def _eos_ids(tokenizer) -> set[int]:
    values = getattr(tokenizer, "eos_token_ids", None)
    if values is None:
        value = getattr(tokenizer, "eos_token_id", None)
        values = [value] if value is not None else []
    if isinstance(values, int):
        values = [values]
    return {int(value) for value in values}


def _run_generators(
    *,
    adapter: SplitCausalLM,
    full_tokens: mx.array,
    query_tokens: mx.array,
    prefix_state: FullPrefixState,
    raw_state: LowerReplayState,
    packed_state: PackedLowerReplayState,
    max_new_tokens: int,
    eos_ids: set[int],
    gpu_stream: mx.Stream,
    run_index: int,
) -> dict[str, dict[str, Any]]:
    operations = {
        "dense": lambda: greedy_generate_dense(
            adapter,
            full_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
        ),
        "exact_prefix": lambda: greedy_generate_full_prefix(
            adapter,
            prefix_state,
            query_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
        ),
        "replay_q16": lambda: greedy_generate_replay(
            adapter,
            raw_state,
            query_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
            stream=gpu_stream,
        ),
        "replay_q4_q4_q8": lambda: greedy_generate_replay(
            adapter,
            packed_state,
            query_tokens,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_ids,
            stream=gpu_stream,
        ),
    }
    names = list(operations)
    rotation = run_index % len(names)
    order = names[rotation:] + names[:rotation]
    results = {}
    for name in order:
        mx.clear_cache()
        generated, seconds = timed(operations[name], gpu_stream)
        results[name] = {
            "generation_seconds": seconds,
            "generated_token_ids": generated,
        }
    dense = results["dense"]["generated_token_ids"]
    for result in results.values():
        result["tokens_match_dense"] = result["generated_token_ids"] == dense
    return results


def _median_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = {}
    for context in sorted({row["context_tokens"] for row in rows}):
        selected = [row for row in rows if row["context_tokens"] == context]
        paths = {}
        for name in (
            "dense",
            "exact_prefix",
            "replay_q16",
            "replay_q4_q4_q8",
        ):
            values = [row["generation"][name]["generation_seconds"] for row in selected]
            paired_speedups = [
                row["generation"]["dense"]["generation_seconds"]
                / row["generation"][name]["generation_seconds"]
                for row in selected
            ]
            paths[name] = {
                "median_generation_seconds": statistics.median(values),
                "min_generation_seconds": min(values),
                "max_generation_seconds": max(values),
                "generation_cv": (
                    statistics.stdev(values) / statistics.mean(values)
                    if len(values) > 1
                    else 0.0
                ),
                "median_paired_speedup_vs_dense": statistics.median(
                    paired_speedups
                ),
                "all_tokens_match_dense": all(
                    row["generation"][name]["tokens_match_dense"]
                    for row in selected
                ),
            }
        grouped[str(context)] = {
            "runs": len(selected),
            "median_prefix_write_seconds": statistics.median(
                row["prefix_write_seconds"] for row in selected
            ),
            "median_replay_write_seconds": statistics.median(
                row["replay_write_seconds"] for row in selected
            ),
            "median_replay_quantize_seconds": statistics.median(
                row["replay_quantize_seconds"] for row in selected
            ),
            "prefix_persistent_mib": statistics.median(
                row["prefix_persistent_nbytes"] for row in selected
            )
            / 2**20,
            "packed_replay_persistent_mib": statistics.median(
                row["packed_replay_persistent_nbytes"] for row in selected
            )
            / 2**20,
            "persistent_compression_vs_prefix": statistics.median(
                row["persistent_compression_vs_prefix"] for row in selected
            ),
            "paths": paths,
        }
    return grouped


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Frozen Q-CoMem hybrid replay benchmark on Apple MLX/Metal"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        default=list(DEFAULT_CONTEXT_LENGTHS),
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--order-offset",
        type=int,
        default=0,
        help="rotate timed path order across independent benchmark sessions",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-preflight-cpu", type=float, default=35.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=128.0)
    parser.add_argument(
        "--power-policy",
        choices=("require-ac", "record-only"),
        default="require-ac",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/qcomem_mlx_hybrid_replay.json"),
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path("results/qcomem_mlx_hybrid_store"),
    )
    parser.add_argument("--no-save-store", action="store_true")
    args = parser.parse_args()

    if args.runs < 1 or args.max_new_tokens < 1:
        raise SystemExit("--runs and --max-new-tokens must be positive")
    if any(length < 2 for length in args.context_lengths):
        raise SystemExit("all context lengths must be at least 2")

    model_revision = args.model_revision
    if model_revision is None and args.model == DEFAULT_MODEL:
        model_revision = DEFAULT_MODEL_REVISION
    if model_revision is None and not Path(args.model).exists():
        raise SystemExit("remote models require an immutable --model-revision")

    before = collect_experiment_snapshot()
    preflight = assess_preflight(before, max_cpu_percent=args.max_preflight_cpu)
    if args.power_policy == "require-ac" and not preflight["formal_result_eligible"]:
        output = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "status": "preflight_failed",
            "formal_result_eligible": False,
            "model": args.model,
            "model_revision": model_revision,
            "frozen_replay_config": FROZEN_REPLAY_CONFIG,
            "environment_before": before,
            "environment_assessment": preflight,
        }
        _write_json(args.output, output)
        raise SystemExit(
            "formal benchmark preflight failed: " + ", ".join(preflight["reasons"])
        )
    if not mx.metal.is_available():
        raise SystemExit("the MLX Metal backend is unavailable")

    mx.set_default_device(mx.gpu)
    gpu_stream = mx.new_stream(mx.gpu)
    from mlx_lm import load

    load_started = time.perf_counter()
    with mx.stream(gpu_stream):
        model, tokenizer = load(args.model, revision=model_revision)
    mx.synchronize(gpu_stream)
    model_load_seconds = time.perf_counter() - load_started
    adapter = SplitCausalLM(model)
    depth = int(FROZEN_REPLAY_CONFIG["depth"])
    if adapter.num_layers < depth:
        raise SystemExit(
            f"frozen depth {depth} exceeds model depth {adapter.num_layers}"
        )
    lower_layers = adapter.layers[:depth]
    if not any(getattr(layer, "is_linear", False) for layer in lower_layers):
        raise SystemExit("frozen benchmark requires linear-attention lower layers")
    if not any(not getattr(layer, "is_linear", False) for layer in lower_layers):
        raise SystemExit("frozen benchmark requires full-attention lower layers")

    query_ids = tokenizer.encode(args.query, add_special_tokens=False)
    if not query_ids:
        raise SystemExit("query tokenization is empty")
    query_tokens = mx.array(query_ids)
    eos_ids = _eos_ids(tokenizer)
    rows = []
    store_files = {}

    for context in args.context_lengths:
        document_tokens = exact_document(tokenizer, context)
        full_tokens = mx.concatenate([document_tokens, query_tokens])
        for run_index in range(args.runs):
            order_index = args.order_offset + run_index
            print(f"Context {context}: run {run_index + 1}/{args.runs}")
            mx.clear_cache()
            mx.reset_peak_memory()
            with mx.stream(gpu_stream):
                prefix_state, prefix_write_seconds = timed(
                    lambda: write_full_prefix(adapter, document_tokens), gpu_stream
                )
                raw_state, replay_write_seconds = timed(
                    lambda: write_lower_replay(
                        adapter, document_tokens, depth=depth
                    ),
                    gpu_stream,
                )
                packed_state, replay_quantize_seconds = timed(
                    lambda: raw_state.quantize(
                        residual_bits=FROZEN_REPLAY_CONFIG["residual_bits"],
                        attention_bits=FROZEN_REPLAY_CONFIG["attention_bits"],
                        linear_bits=FROZEN_REPLAY_CONFIG["linear_bits"],
                        group_size=FROZEN_REPLAY_CONFIG["group_size"],
                        stream=gpu_stream,
                    ),
                    gpu_stream,
                )

            errors = packed_state_error_sums(
                raw_state, packed_state, stream=gpu_stream
            )
            error_rrmse = {
                category: relative_rmse(metrics)
                for category, metrics in errors.items()
            }

            if run_index == 0:
                # Shape-specific warm-up for all four paths. These tokens are
                # intentionally excluded from the timed rows.
                _run_generators(
                    adapter=adapter,
                    full_tokens=full_tokens,
                    query_tokens=query_tokens,
                    prefix_state=prefix_state,
                    raw_state=raw_state,
                    packed_state=packed_state,
                    max_new_tokens=1,
                    eos_ids=set(),
                    gpu_stream=gpu_stream,
                    run_index=order_index,
                )
            generation = _run_generators(
                adapter=adapter,
                full_tokens=full_tokens,
                query_tokens=query_tokens,
                prefix_state=prefix_state,
                raw_state=raw_state,
                packed_state=packed_state,
                max_new_tokens=args.max_new_tokens,
                eos_ids=eos_ids,
                gpu_stream=gpu_stream,
                run_index=order_index,
            )

            if run_index == 0 and not args.no_save_store:
                store_path = args.store_dir / f"tokens-{context}.safetensors"
                packed_state.save(store_path)
                store_files[str(context)] = {
                    "path": str(store_path),
                    "file_bytes": store_path.stat().st_size,
                }

            row = {
                "context_tokens": context,
                "query_tokens": len(query_ids),
                "run_index": run_index,
                "order_index": order_index,
                "path_order": list(generation),
                "prefix_write_seconds": prefix_write_seconds,
                "replay_write_seconds": replay_write_seconds,
                "replay_quantize_seconds": replay_quantize_seconds,
                "prefix_persistent_nbytes": prefix_state.stored_nbytes,
                "raw_replay_persistent_nbytes": raw_state.stored_nbytes,
                "packed_replay_persistent_nbytes": packed_state.stored_nbytes,
                "persistent_compression_vs_prefix": (
                    prefix_state.stored_nbytes / packed_state.stored_nbytes
                ),
                "relative_rmse": error_rrmse,
                "generation": generation,
                "mlx_peak_memory_bytes": mx.get_peak_memory(),
                "mlx_active_memory_bytes": mx.get_active_memory(),
                "process_rss_bytes": psutil.Process().memory_info().rss,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "context_tokens": context,
                        "run_index": run_index,
                        "compression": row["persistent_compression_vs_prefix"],
                        "packed_matches_dense": generation["replay_q4_q4_q8"][
                            "tokens_match_dense"
                        ],
                    }
                )
            )

    after = collect_experiment_snapshot()
    assessment = assess_completed_run(
        before,
        after,
        max_cpu_percent=args.max_preflight_cpu,
        max_swap_growth_bytes=round(args.max_swap_growth_mb * 1024**2),
    )
    if args.power_policy == "record-only":
        assessment = {
            **assessment,
            "formal_result_eligible": False,
            "reasons": list(
                dict.fromkeys([*assessment["reasons"], "record_only_diagnostic_mode"])
            ),
        }
    output = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "formal_result_eligible": assessment["formal_result_eligible"],
        "runtime": "q-comem-mlx-hybrid-replay",
        "device": mx.device_info(mx.gpu),
        "model": args.model,
        "model_revision": model_revision,
        "model_load_seconds": model_load_seconds,
        "num_layers": adapter.num_layers,
        "frozen_replay_config": FROZEN_REPLAY_CONFIG,
        "experiment": {
            "context_lengths": args.context_lengths,
            "runs": args.runs,
            "order_offset": args.order_offset,
            "max_new_tokens": args.max_new_tokens,
            "query": args.query,
            "configuration_order": "cyclic rotation per run",
            "warmup": "one generated token per path and context length",
            "persistent_bytes": "packed arrays including scales and biases",
        },
        "store_files": store_files,
        "environment_before": before,
        "environment_after": after,
        "environment_assessment": assessment,
        "summary": _median_summary(rows),
        "rows": rows,
    }
    _write_json(args.output, output)
    print(
        json.dumps(
            {
                "formal_result_eligible": output["formal_result_eligible"],
                "invalid_reasons": assessment["reasons"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
