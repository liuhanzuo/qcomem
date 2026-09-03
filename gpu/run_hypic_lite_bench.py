from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Callable

import torch

from hypic_lite import (
    HYPIC_LITE_CONFIGS,
    build_hypic_lite_store,
    even_segment_lengths,
    hypic_lite_first_token,
    logit_comparison,
    model_suffix_storage_estimate,
    parse_hypic_lite_config,
    qcomem_rebuild_first_token,
    transition_validation_summary,
)
from qcomem_deployment import environment_metadata, run_exactness_gate
from qcomem_torch import TorchSplitCausalLM, active_cache_layer_indices
from run_downstream import atomic_json, load_samples, prompt_parts


PROTOTYPE_STATUS = (
    "HYPIC-inspired reference prototype, not a complete HYPIC reproduction"
)


def _sync() -> None:
    torch.cuda.synchronize()


def _parse_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(field.strip()) for field in value.split(",") if field.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a non-empty comma-separated list")
    return parsed


def _median_trace(operation: Callable[[], Any], repeats: int) -> tuple[Any, dict[str, Any]]:
    traces = [operation() for _ in range(repeats)]
    fields = (
        "ttft_seconds",
        "fork_seconds",
        "lower_query_seconds",
        "suffix_assembly_seconds",
        "suffix_query_seconds",
        "private_suffix_cache_nbytes",
    )
    summary = {
        f"median_{field}": statistics.median(getattr(trace, field) for trace in traces)
        for field in fields
    }
    summary["repeat_ttft_seconds"] = [trace.ttft_seconds for trace in traces]
    return traces[-1], summary


@torch.inference_mode()
def _full_prefix_first_token(
    adapter: TorchSplitCausalLM,
    prefix_state: Any,
    query_tokens: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    _sync()
    started = time.perf_counter()
    local = prefix_state.fork()
    logits = adapter.continue_full_prefix(local, query_tokens)
    _sync()
    return logits, time.perf_counter() - started


def _model_ledger(config: Any, *, depth: int, segments: int) -> dict[str, Any]:
    scenarios = {}
    segment_counts = tuple(dict.fromkeys((1, 4, segments)))
    for segment_count in segment_counts:
        by_seam = {}
        for seam in (0, 8):
            by_seam[f"w{seam}"] = {
                "transformers_runtime_fp32_state": model_suffix_storage_estimate(
                    config,
                    depth=depth,
                    document_tokens=4096,
                    segment_count=segment_count,
                    seam_width=seam,
                    state_bytes=4,
                    cache_bytes=2,
                    transition_bytes=2,
                ),
                "all_bf16_payload_estimate": model_suffix_storage_estimate(
                    config,
                    depth=depth,
                    document_tokens=4096,
                    segment_count=segment_count,
                    seam_width=seam,
                    state_bytes=2,
                    cache_bytes=2,
                    transition_bytes=2,
                ),
            }
        scenarios[f"segments_{segment_count}"] = {"by_seam": by_seam}
    return {
        "depth": depth,
        "document_tokens": 4096,
        "requested_segment_count": segments,
        "scenarios": scenarios,
        "capability_boundary": {
            "public_transformers_cache_exposes_transition": False,
            "reference_extraction": (
                "wrap the internal chunk_gated_delta_rule and run S0=I with "
                "zero values; production requires a fused kernel that emits T_C"
            ),
            "linear_transition_only_can_skip_hybrid_suffix": False,
            "linear_transition_plus_seam_kv_can_skip_hybrid_suffix": False,
            "reason": (
                "Qwen3.5 interleaves linear and full-attention layers. Public cache "
                "APIs expose only each layer's end state/KV, not a cross-layer token "
                "transition, while every full-attention layer needs all-document KV."
            ),
            "compressed_q4_q8_status": (
                "payload-only storage lower bound; new approximate combination, not "
                "HYPIC, and not executable until quantized compose/KV kernels exist"
            ),
        },
    }


def _load_validation_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    samples = load_samples(
        args.data,
        args.limit_per_dataset,
        source_index_start=args.source_index_start,
        source_index_end=args.source_index_end,
        exclude_source_indices=args.exclude_source_indices,
    )
    if not samples:
        raise ValueError("validation slice is empty")
    if any(int(sample.get("_source_index", -1)) >= 68 for sample in samples):
        raise ValueError("test-v2 is frozen and must not be consumed by this prototype")
    if any(int(sample.get("_source_index", -1)) in {4, 5} for sample in samples):
        raise ValueError("calibration indices 4-5 leaked into validation")
    datasets = {sample["dataset"] for sample in samples}
    if datasets - {"qasper", "2wikimqa"}:
        raise ValueError(f"unexpected datasets: {sorted(datasets)}")
    return samples[args.rank :: args.world_size]


def _packed_document_state(
    adapter: TorchSplitCausalLM,
    document: torch.Tensor,
    args: argparse.Namespace,
) -> Any:
    raw = adapter.write_lower_replay(document, args.depth)
    active_layers = active_cache_layer_indices(raw.cache)
    allocated_layers = getattr(raw.cache, "layers", ())
    if len(args.cache_layer_bits) not in {
        len(active_layers),
        len(allocated_layers),
    }:
        raise ValueError(
            f"{len(args.cache_layer_bits)} cache-layer bits for "
            f"{len(active_layers)} active / {len(allocated_layers)} allocated "
            "lower-cache layers"
        )
    return raw.quantize(
        bits=args.residual_bits,
        attention_bits=args.attention_bits,
        linear_bits=args.linear_bits,
        cache_layer_bits=args.cache_layer_bits,
        group_size=args.group_size,
    )


@torch.inference_mode()
def _prototype_gate(
    adapter: TorchSplitCausalLM,
    document: torch.Tensor,
    query: torch.Tensor,
    *,
    depth: int,
    group_size: int,
    eos_ids: set[int],
    logit_atol: float,
    transition_relative_l2: float,
) -> dict[str, Any]:
    exact = run_exactness_gate(
        adapter,
        document,
        query,
        depth=depth,
        group_size=group_size,
        max_new_tokens=2,
        eos_token_ids=eos_ids,
    )
    raw = adapter.write_lower_replay(document, depth)
    config = parse_hypic_lite_config("hypic-lite-transition-w0")
    store = build_hypic_lite_store(
        adapter,
        raw,
        config,
        segment_lengths=(raw.document_length,),
        validate_transitions=True,
    )
    reference = qcomem_rebuild_first_token(adapter, raw, query)
    candidate = hypic_lite_first_token(adapter, store, raw, query)
    logits = logit_comparison(reference.logits, candidate.logits)
    transition = transition_validation_summary(store)
    transition_error = transition["max_relative_l2_error"]
    transition_passed = (
        transition_error is not None
        and float(transition_error) <= transition_relative_l2
    )
    single_segment_passed = bool(logits["top1_match"]) and (
        float(logits["max_abs_logit_error"]) <= logit_atol
    )
    return {
        "passed": bool(exact["passed"] and transition_passed and single_segment_passed),
        "underlying_exact_replay": exact,
        "single_segment_equivalence": {
            **logits,
            "max_abs_logit_error_threshold": logit_atol,
            "passed": single_segment_passed,
            "semantic": (
                "one segment has no cross-segment approximation and must match the "
                "current Q-CoMem suffix rebuild first token"
            ),
        },
        "affine_transition": {
            **transition,
            "max_relative_l2_error_threshold": transition_relative_l2,
            "passed": transition_passed,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HYPIC-inspired suffix-TTFT/bytes reference benchmark"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--segment-count", type=int, default=4)
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--source-index-start", type=int, default=6)
    parser.add_argument("--source-index-end", type=int, default=35)
    parser.add_argument(
        "--exclude-source-indices", type=_parse_ints, default=(4, 5)
    )
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--residual-bits", type=int, default=4)
    parser.add_argument("--attention-bits", type=int, default=4)
    parser.add_argument("--linear-bits", type=int, default=8)
    parser.add_argument(
        "--cache-layer-bits", type=_parse_ints, default=(8, 8, 8, 4, 8, 8, 8)
    )
    parser.add_argument(
        "--configs", nargs="+", choices=HYPIC_LITE_CONFIGS, default=HYPIC_LITE_CONFIGS
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--gate-document-tokens", type=int, default=128)
    parser.add_argument("--gate-query-tokens", type=int, default=64)
    parser.add_argument("--single-segment-logit-atol", type=float, default=0.05)
    parser.add_argument("--transition-relative-l2", type=float, default=0.02)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("rank must be within world size")
    if args.segment_count < 2:
        raise SystemExit("benchmark segment-count must be at least 2")
    if args.warmups < 1 or args.repeats < 1:
        raise SystemExit("at least one warmup and repeat are required")

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    samples = _load_validation_rows(args)
    if not samples:
        raise SystemExit("rank has no validation rows")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    _sync()
    load_seconds = time.perf_counter() - started
    adapter = TorchSplitCausalLM(model)
    eos_value = tokenizer.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else set(eos_value or [])

    first_document, first_query, *_ = prompt_parts(
        tokenizer, samples[0], args.max_input_tokens
    )
    gate = _prototype_gate(
        adapter,
        first_document[: args.gate_document_tokens].unsqueeze(0).cuda(),
        first_query[: args.gate_query_tokens].unsqueeze(0).cuda(),
        depth=args.depth,
        group_size=args.group_size,
        eos_ids=eos_ids,
        logit_atol=args.single_segment_logit_atol,
        transition_relative_l2=args.transition_relative_l2,
    )
    destination = args.run_dir / f"hypic-lite-shard-{args.rank}.json"
    base: dict[str, Any] = {
        "status": "running" if gate["passed"] else "prototype_gate_failed",
        "prototype_status": PROTOTYPE_STATUS,
        "rank": args.rank,
        "world_size": args.world_size,
        "model": str(args.model),
        "model_load_seconds": load_seconds,
        "environment": environment_metadata(model),
        "transformers": transformers.__version__,
        "protocol": {
            "data": str(args.data),
            "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "source_index_start": args.source_index_start,
            "source_index_end": args.source_index_end,
            "exclude_source_indices": list(args.exclude_source_indices),
            "test_v2_consumed": False,
            "first_token_only": True,
            "depth": args.depth,
            "segment_count": args.segment_count,
            "residual_bits": args.residual_bits,
            "attention_bits": args.attention_bits,
            "linear_bits": args.linear_bits,
            "cache_layer_bits": list(args.cache_layer_bits),
            "configs": list(args.configs),
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "depth7_4k_storage_ledger": _model_ledger(
            adapter.config, depth=args.depth, segments=args.segment_count
        ),
        "prototype_gate": gate,
        "rows": [],
    }
    atomic_json(destination, base)
    if not gate["passed"]:
        raise SystemExit("HYPIC-lite correctness gate failed")

    for sample in samples:
        document_cpu, query_cpu, prefix_tokens, context_tokens, original_context = (
            prompt_parts(tokenizer, sample, args.max_input_tokens)
        )
        document = document_cpu.unsqueeze(0).cuda()
        query = query_cpu.unsqueeze(0).cuda()
        packed = _packed_document_state(adapter, document, args)
        full_prefix = adapter.write_full_prefix(document)
        for _ in range(args.warmups):
            qcomem_rebuild_first_token(adapter, packed, query)
            _full_prefix_first_token(adapter, full_prefix, query)
        qcomem_trace, qcomem_timing = _median_trace(
            lambda: qcomem_rebuild_first_token(adapter, packed, query), args.repeats
        )
        prefix_runs = [
            _full_prefix_first_token(adapter, full_prefix, query)
            for _ in range(args.repeats)
        ]
        prefix_logits = prefix_runs[-1][0]
        prefix_ttft = statistics.median(run[1] for run in prefix_runs)
        lengths = even_segment_lengths(packed.document_length, args.segment_count)

        for config_name in args.configs:
            config = parse_hypic_lite_config(config_name)
            store = build_hypic_lite_store(
                adapter,
                packed,
                config,
                segment_lengths=lengths,
                validate_transitions=True,
            )
            for _ in range(args.warmups):
                hypic_lite_first_token(adapter, store, packed, query)
            trace, timing = _median_trace(
                lambda: hypic_lite_first_token(adapter, store, packed, query),
                args.repeats,
            )
            row = {
                "workload_id": f"{sample['dataset']}-{sample['_source_index']}",
                "dataset": sample["dataset"],
                "source_index": int(sample["_source_index"]),
                "prefix_tokens": prefix_tokens,
                "context_tokens": context_tokens,
                "original_context_tokens": original_context,
                "document_tokens": packed.document_length,
                "query_tokens": int(query.shape[1]),
                "config": config_name,
                "composition": config.composition,
                "seam_width": config.seam_width,
                "segment_lengths": list(lengths),
                "offline_store_build_seconds": store.build_seconds,
                **timing,
                "current_qcomem": qcomem_timing,
                "full_prefix_median_ttft_seconds": prefix_ttft,
                "speedup_vs_qcomem_ttft": (
                    qcomem_timing["median_ttft_seconds"]
                    / timing["median_ttft_seconds"]
                ),
                "speedup_vs_full_prefix_ttft": (
                    prefix_ttft / timing["median_ttft_seconds"]
                ),
                "same_packed_qcomem_logits": logit_comparison(
                    qcomem_trace.logits, trace.logits
                ),
                "exact_full_prefix_logits": logit_comparison(prefix_logits, trace.logits),
                "qcomem_vs_full_prefix_logits": logit_comparison(
                    prefix_logits, qcomem_trace.logits
                ),
                "persistent_bytes": store.bytes_ledger(),
                "request_work": store.work_ledger(),
                "transition_validation": transition_validation_summary(store),
                "full_prefix_persistent_nbytes": full_prefix.stored_nbytes,
                "private_suffix_cache_nbytes": trace.private_suffix_cache_nbytes,
            }
            base["rows"].append(row)
            atomic_json(destination, base)
            print(
                json.dumps(
                    {
                        "rank": args.rank,
                        "workload": row["workload_id"],
                        "config": config_name,
                        "ttft_seconds": timing["median_ttft_seconds"],
                        "persistent_mib": row["persistent_bytes"]["profiles"]
                        ["full_suffix_local_cache"]["persistent_nbytes"]
                        / 2**20,
                        "top1_match": row["same_packed_qcomem_logits"]["top1_match"],
                    }
                ),
                flush=True,
            )
            del store
            torch.cuda.empty_cache()
        del packed, full_prefix
        torch.cuda.empty_cache()

    base["status"] = "completed"
    atomic_json(destination, base)
    print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
