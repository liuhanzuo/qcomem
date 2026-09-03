from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

import mlx.core as mx

from .comem_model import SplitCausalLM
from .comem_quant import (
    DepthBitPolicy,
    StoredResidual,
    quantize_residual,
    select_depth_bit_policy,
)
from .manual_mlx import CORPUS


DEFAULT_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
T = TypeVar("T")


def exact_prompt(tokenizer, target: int) -> list[int]:
    source = tokenizer.encode(CORPUS, add_special_tokens=False)
    repeats = (target + len(source) - 1) // len(source)
    return (source * repeats)[:target]


def _evaluate(value: Any) -> None:
    if isinstance(value, StoredResidual):
        value.eval()
    elif isinstance(value, mx.array):
        mx.eval(value)
    else:
        raise TypeError(f"cannot evaluate value of type {type(value)!r}")


def timed(
    operation: Callable[[], T], gpu_stream: mx.Stream
) -> tuple[T, float]:
    mx.synchronize(gpu_stream)
    started = time.perf_counter()
    value = operation()
    _evaluate(value)
    mx.synchronize(gpu_stream)
    return value, time.perf_counter() - started


def _last_logits(logits: mx.array) -> mx.array:
    return logits[:, -1, :].astype(mx.float32)


def logit_metrics(reference: mx.array, candidate: mx.array) -> dict[str, Any]:
    reference = _last_logits(reference)
    candidate = _last_logits(candidate)
    reference_logp = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
    candidate_logp = candidate - mx.logsumexp(candidate, axis=-1, keepdims=True)
    kl = mx.sum(
        mx.exp(reference_logp) * (reference_logp - candidate_logp), axis=-1
    ).mean()
    max_abs = mx.max(mx.abs(reference - candidate))
    reference_top1 = mx.argmax(reference, axis=-1)
    candidate_top1 = mx.argmax(candidate, axis=-1)
    mx.eval(kl, max_abs, reference_top1, candidate_top1)
    reference_token = int(reference_top1.item())
    candidate_token = int(candidate_top1.item())
    return {
        "kl_divergence": float(kl.item()),
        "max_logit_abs_error": float(max_abs.item()),
        "reference_top1": reference_token,
        "candidate_top1": candidate_token,
        "top1_match": reference_token == candidate_token,
    }


def residual_metrics(reference: mx.array, candidate: mx.array) -> dict[str, float]:
    reference = reference.astype(mx.float32)
    candidate = candidate.astype(mx.float32)
    error = reference - candidate
    rmse = mx.sqrt(mx.mean(mx.square(error)))
    denominator = mx.sqrt(mx.mean(mx.square(reference)))
    relative = rmse / mx.maximum(denominator, mx.array(1e-12))
    max_abs = mx.max(mx.abs(error))
    mx.eval(rmse, relative, max_abs)
    return {
        "residual_rmse": float(rmse.item()),
        "residual_relative_rmse": float(relative.item()),
        "residual_max_abs_error": float(max_abs.item()),
    }


def default_depths(num_layers: int) -> list[int]:
    return sorted(
        {
            max(1, round(num_layers / 6)),
            max(1, round(num_layers / 4)),
            max(1, round(num_layers / 3)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Depth-aware mixed-precision CoMem residual benchmark"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--context-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--depths", type=int, nargs="+")
    parser.add_argument(
        "--bits", type=int, nargs="+", default=[2, 4, 8, 16]
    )
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--kl-threshold", type=float, default=0.02)
    parser.add_argument("--max-relative-rmse", type=float, default=0.05)
    parser.add_argument("--allow-top1-change", action="store_true")
    parser.add_argument(
        "--depth-bits",
        nargs="*",
        default=[],
        metavar="DEPTH:BITS",
        help="manual policy entries overriding automatic calibration",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/q_comem_depth_benchmark.json"),
    )
    parser.add_argument(
        "--store-dir", type=Path, default=Path("results/q_comem_store")
    )
    parser.add_argument("--no-save-store", action="store_true")
    args = parser.parse_args()

    if args.context_tokens < 2:
        raise SystemExit("--context-tokens must be at least 2")
    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    if args.overlap < 0 or args.overlap >= args.chunk_size:
        raise SystemExit("--overlap must satisfy 0 <= overlap < chunk-size")
    if args.kl_threshold < 0:
        raise SystemExit("--kl-threshold must be non-negative")
    if args.max_relative_rmse < 0:
        raise SystemExit("--max-relative-rmse must be non-negative")
    if not mx.metal.is_available():
        raise SystemExit("the MLX Metal backend is unavailable")

    mx.set_default_device(mx.gpu)
    gpu_stream = mx.new_stream(mx.gpu)

    from mlx_lm import load

    load_started = time.perf_counter()
    with mx.stream(gpu_stream):
        model, tokenizer = load(args.model)
    mx.synchronize(gpu_stream)
    load_seconds = time.perf_counter() - load_started

    split_model = SplitCausalLM(model)
    depths = args.depths or default_depths(split_model.num_layers)
    if any(depth < 0 or depth > split_model.num_layers for depth in depths):
        raise SystemExit(f"depths must be in [0, {split_model.num_layers}]")

    prompt = mx.array(exact_prompt(tokenizer, args.context_tokens))
    with mx.stream(gpu_stream):
        dense_logits, dense_seconds = timed(
            lambda: split_model.full_logits(prompt), gpu_stream
        )

    rows: list[dict[str, Any]] = []
    stores: dict[tuple[int, int], StoredResidual] = {}
    depth_summaries = []

    for depth in depths:
        print(f"Depth {depth}/{split_model.num_layers}: chunk-local Write")
        mx.clear_cache()
        with mx.stream(gpu_stream):
            residual, write_seconds = timed(
                lambda depth=depth: split_model.chunk_local_write(
                    prompt,
                    depth,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                ),
                gpu_stream,
            )
            reference_logits, reference_read_seconds = timed(
                lambda residual=residual, depth=depth: split_model.run_suffix(
                    residual, depth
                ),
                gpu_stream,
            )

        interface = logit_metrics(dense_logits, reference_logits)
        depth_summaries.append(
            {
                "depth": depth,
                "write_seconds": write_seconds,
                "bf16_read_seconds": reference_read_seconds,
                "interface_vs_dense": interface,
            }
        )

        for bits in sorted(set(args.bits)):
            print(f"  {bits}-bit residual")
            mx.reset_peak_memory()
            with mx.stream(gpu_stream):
                stored, quantize_seconds = timed(
                    lambda bits=bits, residual=residual, depth=depth: quantize_residual(
                        residual,
                        depth=depth,
                        bits=bits,
                        group_size=args.group_size,
                        stream=gpu_stream,
                    ),
                    gpu_stream,
                )
                restored, dequantize_seconds = timed(
                    lambda stored=stored: stored.dequantize(stream=gpu_stream),
                    gpu_stream,
                )
                candidate_logits, read_seconds = timed(
                    lambda restored=restored, depth=depth: split_model.run_suffix(
                        restored, depth
                    ),
                    gpu_stream,
                )

            stores[(depth, bits)] = stored
            row = {
                "depth": depth,
                "bits": bits,
                "group_size": args.group_size,
                "stored_nbytes": stored.nbytes,
                "dense_nbytes": stored.dense_nbytes,
                "compression_ratio": stored.compression_ratio,
                "quantize_seconds": quantize_seconds,
                "dequantize_seconds": dequantize_seconds,
                "read_seconds": read_seconds,
                "online_dequantize_read_seconds": dequantize_seconds + read_seconds,
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
                **residual_metrics(residual, restored),
                **logit_metrics(reference_logits, candidate_logits),
            }
            rows.append(row)

    automatic = select_depth_bit_policy(
        rows,
        max_kl=args.kl_threshold,
        max_relative_rmse=args.max_relative_rmse,
        require_top1_match=not args.allow_top1_change,
    )
    manual = DepthBitPolicy.from_specs(args.depth_bits) if args.depth_bits else None
    assignments = dict(automatic.assignments)
    if manual is not None:
        unknown_depths = set(manual.assignments) - set(depths)
        if unknown_depths:
            raise SystemExit(
                f"manual policy contains unbenchmarked depths: {sorted(unknown_depths)}"
            )
        assignments.update(manual.assignments)
    policy = DepthBitPolicy(assignments=assignments)

    store_files = {}
    if not args.no_save_store:
        for depth, bits in sorted(policy.assignments.items()):
            try:
                stored = stores[(depth, bits)]
            except KeyError as error:
                raise SystemExit(
                    f"policy selected {bits} bits for depth {depth}, but that "
                    "configuration was not benchmarked"
                ) from error
            path = args.store_dir / f"depth-{depth}-bits-{bits}.safetensors"
            stored.save(path)
            store_files[str(depth)] = {
                "bits": bits,
                "path": str(path),
                "file_bytes": path.stat().st_size,
            }

    output = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "runtime": "q-comem-mlx-core",
        "device": mx.device_info(mx.gpu),
        "model": args.model,
        "model_load_seconds": load_seconds,
        "num_layers": split_model.num_layers,
        "context_tokens": args.context_tokens,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "depths": depths,
        "bits": sorted(set(args.bits)),
        "group_size": args.group_size,
        "dense_full_seconds": dense_seconds,
        "calibration": {
            "scope": "single-prompt smoke; use task-level calibration before deployment",
            "max_kl": args.kl_threshold,
            "max_relative_rmse": args.max_relative_rmse,
            "require_top1_match": not args.allow_top1_change,
        },
        "automatic_policy": automatic.as_dict(),
        "selected_policy": policy.as_dict(),
        "store_files": store_files,
        "depth_summaries": depth_summaries,
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"selected_policy": policy.as_dict()}, indent=2))
    print(f"Saved benchmark: {args.output}")


if __name__ == "__main__":
    main()
