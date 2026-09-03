from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from qcomem_lora import (
    assert_replay_adapter_semantics,
    iter_lora_modules,
    load_inference_lora_checkpoint,
    set_lora_enabled,
)
from qcomem_torch import (
    PackedCache,
    PackedResidual,
    TorchSplitCausalLM,
    cache_nbytes,
    residual_error_sums,
    tensor_nbytes,
)
from run_downstream import (
    answer_f1,
    atomic_json,
    generation_limit,
    load_samples,
    prompt_parts,
)


CONFIG_SUITES = {
    "exact": ("dense", "replay-d7", "replay-d10", "replay-d13"),
    "quant": (
        "dense",
        "prefix",
        "replay-d7-q8",
        "replay-d7-q4",
        "replay-d10-q8",
        "replay-d10-q4",
        "replay-d13-q8",
        "replay-d13-q4",
    ),
    "state": (
        "dense",
        "prefix",
        "replay-d7-r4-a16-l16",
        "replay-d7-r4-a8-l8",
        "replay-d7-r4-a4-l4",
        "replay-d7-r2-a4-l4",
        "replay-d7-r2-a2-l2",
        "replay-d10-r4-a8-l8",
        "replay-d10-r4-a4-l4",
        "replay-d13-r4-a8-l8",
        "replay-d13-r4-a4-l4",
    ),
    "mixed": (
        "dense",
        "prefix",
        "replay-d7-r4-a8-l8",
        "replay-d7-r4-a4-l8",
        "replay-d7-r4-a8-l4",
        "replay-d10-r4-a8-l8",
        "replay-d10-r4-a4-l8",
        "replay-d10-r4-a8-l4",
        "replay-d13-r4-a8-l8",
        "replay-d13-r4-a4-l8",
        "replay-d13-r4-a8-l4",
    ),
    "validation": (
        "dense",
        "prefix",
        "replay-d7-r4-a8-l8",
        "replay-d7-r4-a4-l8",
        "replay-d10-r4-a4-l8",
        "replay-d13-r4-a4-l8",
    ),
    "test": (
        "dense",
        "prefix",
        "replay-d7-r4-a4-l8",
    ),
    # Policies frozen from the four-prompt layer-sensitivity calibration.  The
    # downstream launcher evaluates these jointly during complete autoregressive
    # generation on the disjoint source-index range 6--35.
    "layer-validation": (
        "dense",
        "prefix",
        "replay-d7-layer-q16",
        "replay-d7-layer-q8",
        "replay-d7-frozen-static",
        "replay-d7-same-memory-mixed",
        "replay-d7-minus25-mixed",
    ),
    # Fixed negative-control evaluation for the completed quant checkpoint.
    # The two frozen-static names resolve to identical packed-store bits; only
    # the explicit ``-lora`` alias may enable the installed adapter.
    "quant-lora-validation": (
        "dense",
        "replay-d7-layer-q16",
        "replay-d7-frozen-static",
        "replay-d7-frozen-static-lora",
    ),
}
CONFIGS = CONFIG_SUITES["exact"]

QUANT_LORA_CONFIG = "replay-d7-frozen-static-lora"
VALIDATION_SUITES = frozenset({"layer-validation", "quant-lora-validation"})


LAYER_VALIDATION_POLICIES: dict[
    str, dict[str, int | tuple[int, ...] | None]
] = {
    "replay-d7-layer-q16": {
        "residual_bits": 16,
        "attention_bits": 16,
        "linear_bits": 16,
        "cache_layer_bits": (16, 16, 16, 16, 16, 16, 16),
    },
    "replay-d7-layer-q8": {
        "residual_bits": 8,
        "attention_bits": 8,
        "linear_bits": 8,
        "cache_layer_bits": (8, 8, 8, 8, 8, 8, 8),
    },
    "replay-d7-frozen-static": {
        "residual_bits": 4,
        "attention_bits": 4,
        "linear_bits": 8,
        "cache_layer_bits": (8, 8, 8, 4, 8, 8, 8),
    },
    "replay-d7-frozen-static-lora": {
        "residual_bits": 4,
        "attention_bits": 4,
        "linear_bits": 8,
        "cache_layer_bits": (8, 8, 8, 4, 8, 8, 8),
    },
    "replay-d7-same-memory-mixed": {
        "residual_bits": 4,
        "attention_bits": 4,
        "linear_bits": None,
        "cache_layer_bits": (8, 8, 4, 4, 8, 8, 8),
    },
    "replay-d7-minus25-mixed": {
        "residual_bits": 4,
        "attention_bits": 2,
        "linear_bits": None,
        "cache_layer_bits": (8, 8, 2, 2, 2, 8, 2),
    },
}


@dataclass(frozen=True)
class ResolvedReplayConfig:
    mode: str
    depth: int | None = None
    residual_bits: int | None = None
    attention_bits: int | None = None
    linear_bits: int | None = None
    cache_layer_bits: tuple[int, ...] | None = None
    policy: str | None = None


def timed_gpu(operation):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def lora_parameter_nbytes(model: torch.nn.Module) -> int:
    """Count the resident inference adapter tensors, excluding base weights."""
    seen: set[int] = set()
    total = 0
    for module in iter_lora_modules(model):
        for parameter in (module.lora_a, module.lora_b):
            if id(parameter) not in seen:
                seen.add(id(parameter))
                total += parameter.numel() * parameter.element_size()
    return total


def validate_lora_targets(
    suite: str,
    checkpoint: Path | None,
    apply_to_configs: tuple[str, ...] | list[str],
) -> None:
    """Fail closed for the frozen quant-LoRA paired validation protocol."""
    targets = tuple(apply_to_configs)
    if bool(checkpoint) != bool(targets):
        raise ValueError(
            "a checkpoint and at least one LoRA target config are required together"
        )
    if suite == "quant-lora-validation":
        if checkpoint is None:
            raise ValueError("quant-lora-validation requires a fixed checkpoint")
        if targets != (QUANT_LORA_CONFIG,):
            raise ValueError(
                "quant-lora-validation must enable LoRA only for "
                f"{QUANT_LORA_CONFIG}; got {targets}"
            )


@dataclass
class FreeGenerationTrace:
    generated_token_ids: list[int]
    ttft_seconds: float
    generation_seconds: float


def _finish_incremental_generation(
    logits: torch.Tensor,
    *,
    started: float,
    ttft_seconds: float,
    device: torch.device,
    max_new_tokens: int,
    eos_token_ids: set[int],
    continue_one,
) -> FreeGenerationTrace:
    generated = []
    for step in range(max_new_tokens):
        token = int(torch.argmax(logits, dim=-1).item())
        if token in eos_token_ids:
            break
        generated.append(token)
        if step + 1 < max_new_tokens:
            token_tensor = torch.tensor([[token]], device=device)
            logits = continue_one(token_tensor)
    torch.cuda.synchronize()
    return FreeGenerationTrace(
        generated_token_ids=generated,
        ttft_seconds=ttft_seconds,
        generation_seconds=time.perf_counter() - started,
    )


@torch.inference_mode()
def generate_dense_with_timing(
    adapter: TorchSplitCausalLM,
    tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> FreeGenerationTrace:
    tokens = TorchSplitCausalLM._batch_tokens(tokens)
    torch.cuda.synchronize()
    started = time.perf_counter()
    logits, state = adapter.prefill_full_prefix(tokens)
    torch.cuda.synchronize()
    ttft_seconds = time.perf_counter() - started
    return _finish_incremental_generation(
        logits,
        started=started,
        ttft_seconds=ttft_seconds,
        device=tokens.device,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        continue_one=lambda token: adapter.continue_full_prefix(state, token),
    )


@torch.inference_mode()
def generate_prefix_with_timing(
    adapter: TorchSplitCausalLM,
    document_state,
    query_tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> FreeGenerationTrace:
    query_tokens = TorchSplitCausalLM._batch_tokens(query_tokens)
    torch.cuda.synchronize()
    started = time.perf_counter()
    state = document_state.fork()
    logits = adapter.continue_full_prefix(state, query_tokens)
    torch.cuda.synchronize()
    ttft_seconds = time.perf_counter() - started
    return _finish_incremental_generation(
        logits,
        started=started,
        ttft_seconds=ttft_seconds,
        device=query_tokens.device,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        continue_one=lambda token: adapter.continue_full_prefix(state, token),
    )


@torch.inference_mode()
def generate_replay_with_timing(
    adapter: TorchSplitCausalLM,
    document_state,
    query_tokens: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> FreeGenerationTrace:
    """Mirror ``greedy_generate_replay`` while exposing first-logit latency."""
    query_tokens = TorchSplitCausalLM._batch_tokens(query_tokens)
    torch.cuda.synchronize()
    started = time.perf_counter()
    state = document_state.fork()
    query_residual = adapter.continue_lower_replay(state, query_tokens)
    suffix_cache = adapter.make_cache()
    adapter.run_suffix_cached_last_logits(
        [state.document_residual],
        document_state.depth,
        suffix_cache,
        position_offset=0,
    )
    logits = adapter.run_suffix_cached_last_logits(
        [query_residual],
        document_state.depth,
        suffix_cache,
        position_offset=document_state.document_length,
    )
    suffix_length = state.current_length
    torch.cuda.synchronize()
    ttft_seconds = time.perf_counter() - started

    def continue_one(token: torch.Tensor) -> torch.Tensor:
        nonlocal suffix_length
        token_residual = adapter.continue_lower_replay(state, token)
        next_logits = adapter.run_suffix_cached_last_logits(
            [token_residual],
            document_state.depth,
            suffix_cache,
            position_offset=suffix_length,
        )
        suffix_length += 1
        return next_logits

    return _finish_incremental_generation(
        logits,
        started=started,
        ttft_seconds=ttft_seconds,
        device=query_tokens.device,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        continue_one=continue_one,
    )


def parse_config(
    config_name: str,
) -> tuple[str, int | None, int | None, int | None, int | None]:
    """Parse the legacy public tuple used by older scripts and tests."""
    resolved = resolve_config(config_name)
    return (
        resolved.mode,
        resolved.depth,
        resolved.residual_bits,
        resolved.attention_bits,
        resolved.linear_bits,
    )


def resolve_config(config_name: str) -> ResolvedReplayConfig:
    """Resolve legacy precision fields or a frozen per-layer policy."""
    if config_name == "dense":
        return ResolvedReplayConfig("dense")
    if config_name == "prefix":
        return ResolvedReplayConfig("prefix")
    if config_name in LAYER_VALIDATION_POLICIES:
        policy = LAYER_VALIDATION_POLICIES[config_name]
        raw_depth = config_name.split("-")[1]
        return ResolvedReplayConfig(
            mode="replay",
            depth=int(raw_depth.removeprefix("d")),
            residual_bits=int(policy["residual_bits"]),
            attention_bits=(
                int(policy["attention_bits"])
                if policy["attention_bits"] is not None
                else None
            ),
            linear_bits=(
                int(policy["linear_bits"])
                if policy["linear_bits"] is not None
                else None
            ),
            cache_layer_bits=tuple(policy["cache_layer_bits"]),
            policy=(
                "frozen-static"
                if config_name == QUANT_LORA_CONFIG
                else config_name.removeprefix(f"replay-{raw_depth}-")
            ),
        )
    mode, raw_depth, *precision_fields = config_name.split("-")
    if mode != "replay":
        raise ValueError(f"unknown replay mode: {mode}")
    depth = int(raw_depth.removeprefix("d"))
    residual_bits = None
    attention_bits = None
    linear_bits = None
    for field in precision_fields:
        value = int(field[1:])
        if field.startswith(("q", "r")):
            residual_bits = value
        elif field.startswith("a"):
            attention_bits = value
        elif field.startswith("l"):
            linear_bits = value
        else:
            raise ValueError(f"unknown precision field: {field}")
    return ResolvedReplayConfig(
        mode=mode,
        depth=depth,
        residual_bits=residual_bits,
        attention_bits=attention_bits,
        linear_bits=linear_bits,
    )


def parse_excluded_indices(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(value.strip()) for value in raw.split(",") if value.strip())


def validation_source_range(
    suite: str,
    source_index_start: int | None,
    source_index_end: int | None,
) -> tuple[int | None, int | None]:
    """Apply the preregistered 6--35 range to frozen validation suites."""
    if suite in VALIDATION_SUITES:
        return (
            6 if source_index_start is None else source_index_start,
            35 if source_index_end is None else source_index_end,
        )
    return source_index_start, source_index_end


def validation_excluded_source_indices(
    suite: str, raw: str | None
) -> tuple[int, ...]:
    if raw is None:
        raw = "4,5" if suite in VALIDATION_SUITES else ""
    return parse_excluded_indices(raw)


@torch.inference_mode()
def run_config(
    *,
    config_name: str,
    model,
    tokenizer,
    samples: list[dict[str, Any]],
    model_allocated_bytes: int,
    args,
) -> dict[str, Any]:
    adapter = TorchSplitCausalLM(model)
    resolved = resolve_config(config_name)
    mode = resolved.mode
    depth = resolved.depth
    residual_bits = resolved.residual_bits
    attention_bits = resolved.attention_bits
    linear_bits = resolved.linear_bits
    cache_layer_bits = resolved.cache_layer_bits
    eos_value = tokenizer.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else set(eos_value or [])
    rows = []
    for sample in samples:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        (
            document_ids,
            query_ids,
            prefix_tokens,
            context_tokens,
            original_context_tokens,
        ) = prompt_parts(tokenizer, sample, args.max_input_tokens)
        document_ids = document_ids.cuda()
        query_ids = query_ids.cuda()
        full_ids = torch.cat([document_ids, query_ids])
        max_new_tokens = generation_limit(sample["dataset"], args.max_new_tokens)
        write_seconds = None
        residual_nbytes = None
        lower_cache_nbytes = None
        persistent_nbytes = None
        error_sums = None
        cache_error_sums = None

        if mode == "dense":
            generation = generate_dense_with_timing(
                adapter,
                full_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_ids,
            )
        elif mode == "prefix":
            prefix_state, write_seconds = timed_gpu(
                lambda: adapter.write_full_prefix(document_ids)
            )
            lower_cache_nbytes = cache_nbytes(prefix_state.cache)
            persistent_nbytes = prefix_state.stored_nbytes
            generation = generate_prefix_with_timing(
                adapter,
                prefix_state,
                query_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_ids,
            )
        else:
            assert depth is not None
            raw_state, write_seconds = timed_gpu(
                lambda: adapter.write_lower_replay(document_ids, depth)
            )
            if any(
                value is not None
                for value in (
                    residual_bits,
                    attention_bits,
                    linear_bits,
                    cache_layer_bits,
                )
            ):
                replay_state, quantize_seconds = timed_gpu(
                    lambda: raw_state.quantize(
                        bits=residual_bits or 16,
                        attention_bits=attention_bits,
                        linear_bits=linear_bits,
                        cache_layer_bits=cache_layer_bits,
                        group_size=64,
                    )
                )
                if residual_bits is not None and residual_bits < 16:
                    restored = replay_state.document_residual.dequantize()
                    error_sums = residual_error_sums(
                        raw_state.document_residual, restored
                    )
                    del restored
                cache_error_sums = replay_state.cache_error_sums
                write_seconds += quantize_seconds
                del raw_state
            else:
                replay_state = raw_state
            if isinstance(replay_state.document_residual, PackedResidual):
                residual_nbytes = replay_state.document_residual.nbytes
            else:
                residual_nbytes = tensor_nbytes(replay_state.document_residual)
            lower_cache_nbytes = (
                replay_state.cache.nbytes
                if isinstance(replay_state.cache, PackedCache)
                else cache_nbytes(replay_state.cache)
            )
            persistent_nbytes = replay_state.stored_nbytes
            generation = generate_replay_with_timing(
                adapter,
                replay_state,
                query_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_ids,
            )

        generated = generation.generated_token_ids
        generation_seconds = generation.generation_seconds
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        references = [str(answer) for answer in sample["answers"]]
        f1 = max(answer_f1(prediction, reference) for reference in references)
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        rows.append(
            {
                "dataset": sample["dataset"],
                "id": sample.get("_id"),
                "source_index": sample.get("_source_index"),
                "question": sample["input"],
                "references": references,
                "prediction": prediction,
                "generated_token_ids": generated,
                "f1": f1,
                "prefix_tokens": prefix_tokens,
                "context_tokens": context_tokens,
                "original_context_tokens": original_context_tokens,
                "context_truncated": context_tokens < original_context_tokens,
                "document_tokens": int(document_ids.numel()),
                "query_tokens": int(query_ids.numel()),
                "input_tokens": int(full_ids.numel()),
                "max_new_tokens": max_new_tokens,
                "generated_tokens": len(generated),
                "write_seconds": write_seconds,
                "ttft_seconds": generation.ttft_seconds,
                "generation_seconds": generation_seconds,
                "stored_residual_nbytes": residual_nbytes,
                "stored_lower_cache_nbytes": lower_cache_nbytes,
                "stored_persistent_nbytes": persistent_nbytes,
                "residual_error_sums": error_sums,
                "cache_error_sums": cache_error_sums,
                "model_allocated_bytes": model_allocated_bytes,
                "peak_allocated_bytes": peak_allocated_bytes,
                "incremental_peak_allocated_bytes": max(
                    peak_allocated_bytes - model_allocated_bytes, 0
                ),
            }
        )
        print(
            json.dumps(
                {
                    "config": config_name,
                    "dataset": sample["dataset"],
                    "source_index": sample.get("_source_index"),
                    "f1": f1,
                    "generated_tokens": len(generated),
                    "ttft_seconds": generation.ttft_seconds,
                    "persistent_mib": (
                        persistent_nbytes / 2**20
                        if persistent_nbytes is not None
                        else None
                    ),
                }
            ),
            flush=True,
        )
        del document_ids, query_ids, full_ids
        if mode == "replay":
            del replay_state
        elif mode == "prefix":
            del prefix_state

    return {
        "config": config_name,
        "mode": mode,
        "depth": depth,
        "residual_bits": residual_bits,
        # Legacy alias retained so existing result readers remain compatible.
        "bits": residual_bits,
        "attention_bits": attention_bits,
        "linear_bits": linear_bits,
        "cache_layer_bits": list(cache_layer_bits) if cache_layer_bits else None,
        "policy": resolved.policy,
        "samples": len(rows),
        "mean_f1": statistics.fmean(row["f1"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--suite", choices=sorted(CONFIG_SUITES), default="exact")
    parser.add_argument("--source-index-start", type=int)
    parser.add_argument("--source-index-end", type=int)
    parser.add_argument(
        "--exclude-source-indices",
        default=None,
        help="comma-separated source indices excluded before per-dataset limiting",
    )
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument(
        "--lora-apply-to-configs",
        nargs="*",
        default=(),
        help="exact config names that use the adapter; all others keep it disabled",
    )
    parser.add_argument(
        "--allow-lora-semantic-mismatch",
        action="store_true",
        help=(
            "explicitly allow adapter/config mismatch for a labelled control; "
            "never use this for the primary frozen-policy result"
        ),
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.rank < 0 or args.rank >= args.world_size:
        raise SystemExit("rank is outside world size")
    try:
        validate_lora_targets(
            args.suite, args.lora_checkpoint, args.lora_apply_to_configs
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.suite == "quant-lora-validation" and args.allow_lora_semantic_mismatch:
        raise SystemExit(
            "quant-lora-validation forbids --allow-lora-semantic-mismatch"
        )

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    source_index_start, source_index_end = validation_source_range(
        args.suite, args.source_index_start, args.source_index_end
    )
    excluded_source_indices = validation_excluded_source_indices(
        args.suite, args.exclude_source_indices
    )
    all_samples = load_samples(
        args.data,
        args.limit_per_dataset,
        source_index_start=source_index_start,
        source_index_end=source_index_end,
        exclude_source_indices=excluded_source_indices,
    )
    samples = all_samples[args.rank :: args.world_size]
    if not samples:
        raise SystemExit("rank has no sample shard")
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    configs = CONFIG_SUITES[args.suite]
    unknown_lora_configs = set(args.lora_apply_to_configs) - set(configs)
    if unknown_lora_configs:
        raise SystemExit(
            f"LoRA target configs are outside suite {args.suite}: "
            f"{sorted(unknown_lora_configs)}"
        )
    print(
        json.dumps(
            {
                "rank": args.rank,
                "configs": list(configs),
                "suite": args.suite,
                "samples": len(samples),
                "source_indices": [sample.get("_source_index") for sample in samples],
                "source_index_range": [source_index_start, source_index_end],
                "excluded_source_indices": list(excluded_source_indices),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0),
            }
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    load_started = time.perf_counter()
    base_load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    base_model_deserialize_seconds = time.perf_counter() - base_load_started
    if hasattr(model.model, "visual"):
        model.model.visual = None
    lora_metadata = None
    lora_checkpoint_sha256 = None
    adapter_load_seconds = 0.0
    adapter_parameter_bytes = 0
    adapter_modules = 0
    checkpoint_file_bytes = 0
    if args.lora_checkpoint is not None:
        split = TorchSplitCausalLM(model)
        adapter_load_started = time.perf_counter()
        lora_metadata = load_inference_lora_checkpoint(
            model, split.layers, args.lora_checkpoint
        )
        adapter_load_seconds = time.perf_counter() - adapter_load_started
        lora_checkpoint_sha256 = hashlib.sha256(
            args.lora_checkpoint.read_bytes()
        ).hexdigest()
        checkpoint_file_bytes = args.lora_checkpoint.stat().st_size
        adapter_parameter_bytes = lora_parameter_nbytes(model)
        adapter_modules = sum(1 for _ in iter_lora_modules(model))
        metadata_bytes = lora_metadata.get("adapter", {}).get("trainable_nbytes")
        if metadata_bytes is not None and int(metadata_bytes) != adapter_parameter_bytes:
            raise SystemExit(
                "actual resident LoRA parameter bytes differ from checkpoint metadata: "
                f"actual={adapter_parameter_bytes}, metadata={metadata_bytes}"
            )
    cuda_transfer_started = time.perf_counter()
    model.eval().cuda()
    torch.cuda.synchronize()
    model_to_cuda_seconds = time.perf_counter() - cuda_transfer_started
    model_load_seconds = time.perf_counter() - load_started
    model_allocated_bytes = torch.cuda.memory_allocated()

    for config_name in configs:
        lora_enabled = config_name in set(args.lora_apply_to_configs)
        if lora_enabled and not args.allow_lora_semantic_mismatch:
            resolved = resolve_config(config_name)
            if resolved.mode != "replay" or resolved.depth is None:
                raise SystemExit(
                    f"LoRA target {config_name} is not a replay config; use "
                    "--allow-lora-semantic-mismatch only for a labelled control"
                )
            try:
                assert_replay_adapter_semantics(
                    lora_metadata,
                    depth=resolved.depth,
                    residual_bits=resolved.residual_bits or 16,
                    attention_bits=resolved.attention_bits,
                    linear_bits=resolved.linear_bits,
                    cache_layer_bits=resolved.cache_layer_bits,
                )
            except ValueError as error:
                raise SystemExit(
                    f"LoRA checkpoint is incompatible with {config_name}: {error}"
                ) from error
        set_lora_enabled(model, lora_enabled)
        actual_lora_states = [module.enabled for module in iter_lora_modules(model)]
        if actual_lora_states and any(
            enabled != lora_enabled for enabled in actual_lora_states
        ):
            raise SystemExit(f"failed to set a uniform LoRA state for {config_name}")
        destination = args.run_dir / f"shard-{args.rank}-{config_name}.json"
        if destination.exists():
            print(f"SKIP existing {destination}", flush=True)
            continue
        result = run_config(
            config_name=config_name,
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            model_allocated_bytes=model_allocated_bytes,
            args=args,
        )
        result.update(
            {
                "rank": args.rank,
                "world_size": args.world_size,
                "model": str(args.model),
                "data": str(args.data),
                "data_sha256": data_sha256,
                "prompt_protocol": "longbench-v1-official",
                "model_load_seconds": model_load_seconds,
                "base_model_deserialize_seconds": base_model_deserialize_seconds,
                "model_to_cuda_seconds": model_to_cuda_seconds,
                "model_allocated_bytes": model_allocated_bytes,
                "max_input_tokens": args.max_input_tokens,
                "suite": args.suite,
                "source_index_start": source_index_start,
                "source_index_end": source_index_end,
                "exclude_source_indices": list(excluded_source_indices),
                "lora": {
                    "checkpoint": str(args.lora_checkpoint)
                    if args.lora_checkpoint is not None
                    else None,
                    "checkpoint_sha256": lora_checkpoint_sha256,
                    "checkpoint_file_nbytes": checkpoint_file_bytes,
                    "adapter_load_seconds": adapter_load_seconds,
                    "adapter_parameter_nbytes": adapter_parameter_bytes,
                    "installed_lora_modules": adapter_modules,
                    "enabled_for_this_config": lora_enabled,
                    "apply_to_configs": list(args.lora_apply_to_configs),
                    "checkpoint_metadata": lora_metadata,
                    "semantic_mismatch_explicitly_allowed": (
                        args.allow_lora_semantic_mismatch
                    ),
                },
            }
        )
        atomic_json(destination, result)
        print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
