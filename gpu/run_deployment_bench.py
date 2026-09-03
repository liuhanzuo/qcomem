from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from qcomem_deployment import (
    DEFAULT_CONFIGS,
    DEFAULT_MIXED_LAYER_BITS,
    MemoryRecorder,
    NvmlProcessSampler,
    build_persistent_state,
    capacity_estimate,
    config_asdict,
    environment_metadata,
    load_mixed_policy,
    parse_deployment_config,
    parse_layer_bits,
    persistent_components,
    run_exactness_gate,
    run_incremental_generation,
    shuffled_config_orders,
)
from qcomem_torch import TorchSplitCausalLM
from run_downstream import (
    answer_f1,
    atomic_json,
    generation_limit,
    load_samples,
    prompt_parts,
)


def _sync() -> None:
    torch.cuda.synchronize()


def batch_prefix(tokens: torch.Tensor, limit: int) -> torch.Tensor:
    """Take a token prefix and normalize it to ``[batch, tokens]``."""
    if tokens.ndim == 1:
        return tokens[:limit].unsqueeze(0)
    if tokens.ndim == 2:
        return tokens[:, :limit]
    raise ValueError("token inputs must be rank 1 or 2")


def timed_gpu(operation):
    _sync()
    started = time.perf_counter()
    value = operation()
    _sync()
    return value, time.perf_counter() - started


def visible_nvml_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",", 1)[0].strip()
        if first.isdigit():
            return int(first)
    return torch.cuda.current_device()


def repeated_document(tokenizer, length: int) -> torch.Tensor:
    paragraph = (
        "This is a reusable deployment document. It contains stable facts, "
        "definitions, measurements, implementation details, and evidence. "
    )
    base = tokenizer(
        paragraph, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    repeats = math.ceil(length / base.shape[1])
    return base.repeat(1, repeats)[:, :length]


def longbench_workloads(tokenizer, args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if args.data is None:
        raise ValueError("--data is required for the longbench workload")
    samples = load_samples(
        args.data,
        args.limit_per_dataset,
        source_index_start=args.source_index_start,
        source_index_end=args.source_index_end,
        exclude_source_indices=tuple(args.exclude_source_indices),
    )
    if not samples:
        raise ValueError("the LongBench input contains no samples")
    datasets = {sample.get("dataset") for sample in samples}
    unexpected = datasets - {"qasper", "2wikimqa"}
    if unexpected:
        raise ValueError(f"unsupported deployment datasets: {sorted(unexpected)}")
    revisions = {sample.get("_source_revision") for sample in samples}
    if None in revisions or len(revisions) != 1:
        raise ValueError("LongBench rows must carry one frozen _source_revision")
    consumed_test_v2 = [
        sample
        for sample in samples
        if int(sample.get("_source_index", -1)) >= 68
    ]
    if consumed_test_v2 and not args.allow_test_v2:
        raise ValueError(
            "source index >=68 is reserved for test-v2; use validation rows or "
            "pass --allow-test-v2 intentionally"
        )

    workloads = []
    for sample in samples:
        document, query, prefix_tokens, context_tokens, original_context = prompt_parts(
            tokenizer, sample, args.max_input_tokens
        )
        workloads.append(
            {
                "workload_id": (
                    f"{sample['dataset']}-{sample.get('_source_index', sample.get('_id'))}"
                ),
                "kind": "longbench",
                "dataset": sample["dataset"],
                "source_index": sample.get("_source_index"),
                "source_id": sample.get("_id"),
                "source_repo": sample.get("_source_repo"),
                "source_revision": sample.get("_source_revision"),
                "document_tokens": document,
                "query_tokens": query,
                "prefix_tokens": prefix_tokens,
                "context_tokens": context_tokens,
                "original_context_tokens": original_context,
                "references": [str(answer) for answer in sample["answers"]],
            }
        )
    metadata = {
        "data": str(args.data),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "source_revisions": sorted(revisions),
        "datasets": sorted(datasets),
        "source_index_start": args.source_index_start,
        "source_index_end": args.source_index_end,
        "excluded_source_indices": list(args.exclude_source_indices),
        "test_v2_consumed": bool(consumed_test_v2),
        "prompt_protocol": "longbench-v1-official",
    }
    return workloads, metadata


def synthetic_workloads(tokenizer, args) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    query = tokenizer(
        "\nQuestion: What kind of document is described?\nAnswer:",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    workloads = []
    for repetition in range(args.synthetic_repetitions):
        for length in args.context_lengths:
            workloads.append(
                {
                    "workload_id": f"synthetic-{length}-r{repetition}",
                    "kind": "synthetic",
                    "dataset": "synthetic-capacity",
                    "source_index": None,
                    "source_id": None,
                    "source_repo": None,
                    "source_revision": "generated-by-run_deployment_bench.py",
                    "document_tokens": repeated_document(tokenizer, length),
                    "query_tokens": query.clone(),
                    "prefix_tokens": 0,
                    "context_tokens": length,
                    "original_context_tokens": length,
                    "references": [],
                }
            )
    return workloads, {
        "generator": "repeated stable deployment paragraph",
        "context_lengths": list(args.context_lengths),
        "synthetic_repetitions": args.synthetic_repetitions,
        "prompt_protocol": "synthetic-capacity-v1",
        "test_v2_consumed": False,
    }


def resolve_configs(args) -> tuple[list[Any], dict[str, Any]]:
    mixed_residual_bits = 4
    mixed_layer_bits = parse_layer_bits(args.mixed_layer_bits)
    policy_metadata: dict[str, Any] = {
        "source": "cli-default",
        "residual_bits": mixed_residual_bits,
        "cache_layer_bits": list(mixed_layer_bits),
    }
    if args.mixed_policy_file is not None:
        mixed_residual_bits, mixed_layer_bits = load_mixed_policy(
            args.mixed_policy_file, args.mixed_policy_name
        )
        policy_metadata = {
            "source": str(args.mixed_policy_file),
            "sha256": hashlib.sha256(args.mixed_policy_file.read_bytes()).hexdigest(),
            "policy_name": args.mixed_policy_name,
            "residual_bits": mixed_residual_bits,
            "cache_layer_bits": list(mixed_layer_bits),
        }

    configs = []
    for name in args.configs:
        config = parse_deployment_config(name, mixed_layer_bits=mixed_layer_bits)
        if name.endswith("-mixed"):
            config = replace(config, residual_bits=mixed_residual_bits)
        configs.append(config)
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("--configs contains duplicate names")
    return configs, policy_metadata


def warmup_config(
    adapter,
    config,
    document,
    query,
    *,
    group_size: int,
    eos_ids: set[int],
    fork_strategy: str = "deep-clone",
) -> None:
    document = batch_prefix(document, 128)
    query = batch_prefix(query, 32)
    state = build_persistent_state(
        adapter,
        config,
        document,
        group_size=group_size,
        fork_strategy=fork_strategy,
    )
    run_incremental_generation(
        adapter,
        config,
        document,
        query,
        state,
        max_new_tokens=2,
        eos_token_ids=eos_ids,
        recorder=MemoryRecorder(),
    )
    del state
    torch.cuda.empty_cache()


@torch.inference_mode()
def measure_one(
    *,
    adapter,
    tokenizer,
    config,
    workload,
    repeat: int,
    order_position: int,
    model_allocated_bytes: int,
    total_device_bytes: int,
    nvml,
    args,
) -> dict[str, Any]:
    document = workload["document_tokens"].cuda()
    query = workload["query_tokens"].cuda()
    eos_value = tokenizer.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else set(eos_value or [])
    max_new_tokens = (
        generation_limit(workload["dataset"], args.max_new_tokens)
        if workload["kind"] == "longbench"
        else args.max_new_tokens
    )

    torch.cuda.empty_cache()
    build_recorder = MemoryRecorder(nvml)
    build_recorder.reset_peak()
    build_start = build_recorder.sample("build_start")
    if config.mode == "dense_recompute":
        persistent = None
        build_seconds = None
    else:
        persistent, build_seconds = timed_gpu(
            lambda: build_persistent_state(
                adapter,
                config,
                document,
                group_size=args.group_size,
                fork_strategy=(
                    args.fork_strategy if config.mode == "qcomem" else "deep-clone"
                ),
            )
        )
    build_recorder.sample("build_end")
    build_memory = build_recorder.summary(steady_prefix="build_end")
    components = persistent_components(persistent)

    torch.cuda.empty_cache()
    request_recorder = MemoryRecorder(nvml)
    trace = run_incremental_generation(
        adapter,
        config,
        document,
        query,
        persistent,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_ids,
        recorder=request_recorder,
    )
    trace_summary = trace.summary()
    fork_memory = trace_summary["fork_memory"]
    request_start_allocated = int(
        trace_summary["memory_samples"][0]["cuda_allocated_bytes"]
    )
    capacity_document_bytes = components.get(
        "persistent_total_resident_nbytes",
        components["persistent_document_nbytes"],
    )
    capacity = capacity_estimate(
        total_device_bytes=total_device_bytes,
        model_allocated_bytes=model_allocated_bytes,
        persistent_document_bytes=capacity_document_bytes,
        request_peak_allocated_bytes=trace_summary["cuda_peak_allocated_bytes"],
        request_start_allocated_bytes=request_start_allocated,
        safety_headroom_bytes=round(args.safety_headroom_gib * 2**30),
    )
    prediction = tokenizer.decode(
        trace.generated_token_ids, skip_special_tokens=True
    ).strip()
    references = workload["references"]
    f1 = (
        max(answer_f1(prediction, reference) for reference in references)
        if references
        else None
    )
    row = {
        "config": config.name,
        "effective_config": config_asdict(config),
        "repeat": repeat,
        "randomized_order_position": order_position,
        "workload_id": workload["workload_id"],
        "workload_kind": workload["kind"],
        "dataset": workload["dataset"],
        "source_index": workload["source_index"],
        "source_id": workload["source_id"],
        "source_repo": workload["source_repo"],
        "source_revision": workload["source_revision"],
        "document_tokens": int(document.numel()),
        "query_tokens": int(query.numel()),
        "prefix_tokens": workload["prefix_tokens"],
        "context_tokens": workload["context_tokens"],
        "original_context_tokens": workload["original_context_tokens"],
        "context_truncated": (
            workload["context_tokens"] < workload["original_context_tokens"]
        ),
        "max_new_tokens": max_new_tokens,
        "prediction": prediction,
        "references": references,
        "f1": f1,
        "write_build_seconds": build_seconds,
        "build_start_cuda_allocated_bytes": int(
            build_start["cuda_allocated_bytes"]
        ),
        "build_memory": build_memory,
        **components,
        "capacity_document_denominator_nbytes": capacity_document_bytes,
        **trace_summary,
        "cow_initial_shared_nbytes": fork_memory.get("initial_shared_nbytes"),
        "cow_initial_private_nbytes": fork_memory.get("initial_private_nbytes"),
        "cow_after_query_shared_nbytes": fork_memory.get(
            "after_query_shared_nbytes"
        ),
        "cow_after_query_private_nbytes": fork_memory.get(
            "after_query_private_nbytes"
        ),
        "cow_final_shared_nbytes": fork_memory.get("final_shared_nbytes"),
        "cow_final_private_nbytes": fork_memory.get("final_private_nbytes"),
        **capacity,
        "model_cuda_allocated_baseline_bytes": model_allocated_bytes,
        "device_total_memory_bytes": total_device_bytes,
        "capacity_safety_headroom_bytes": round(args.safety_headroom_gib * 2**30),
    }
    del persistent, document, query, trace
    torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production-style Q-CoMem KV/deployment benchmark"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--workload", choices=("longbench", "synthetic"), required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--expected-source-revision")
    parser.add_argument("--expected-source-indices", type=int, nargs="*")
    parser.add_argument("--expected-workloads", type=int)
    parser.add_argument("--protocol-label", default="deployment-generic")
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--source-index-start", type=int, default=6)
    parser.add_argument("--source-index-end", type=int, default=35)
    parser.add_argument(
        "--exclude-source-indices", type=int, nargs="*", default=(4, 5)
    )
    parser.add_argument("--allow-test-v2", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=(4096, 8192, 16384, 32768))
    parser.add_argument("--synthetic-repetitions", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument(
        "--mixed-layer-bits",
        default=",".join(str(bit) for bit in DEFAULT_MIXED_LAYER_BITS),
    )
    parser.add_argument("--mixed-policy-file", type=Path)
    parser.add_argument("--mixed-policy-name", default="same_memory_as_frozen")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--fork-strategy",
        choices=("deep-clone", "paged-cow-staging"),
        default="deep-clone",
        help="paged-cow-staging is an audited prototype, not a paged-attention kernel",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--gate-depth", type=int, default=7)
    parser.add_argument("--gate-document-tokens", type=int, default=256)
    parser.add_argument("--gate-query-tokens", type=int, default=64)
    parser.add_argument("--gate-new-tokens", type=int, default=4)
    parser.add_argument(
        "--gate-only",
        action="store_true",
        help="run the per-rank Q16 replay exactness gate and stop before timing",
    )
    parser.add_argument("--require-exact-logits", action="store_true")
    parser.add_argument("--logit-atol", type=float, default=0.0)
    parser.add_argument("--safety-headroom-gib", type=float, default=4.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not (0 <= args.rank < args.world_size):
        raise SystemExit("rank must be within world size")
    if args.warmups < 1 or args.repeats < 1:
        raise SystemExit("formal runs require at least one warmup and one repeat")

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    random.seed(args.seed + args.rank)
    torch.manual_seed(args.seed + args.rank)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.workload == "longbench":
        workloads, workload_metadata = longbench_workloads(tokenizer, args)
    else:
        workloads, workload_metadata = synthetic_workloads(tokenizer, args)
    if args.expected_data_sha256 is not None and (
        workload_metadata.get("data_sha256") != args.expected_data_sha256
    ):
        raise SystemExit("LongBench data SHA256 does not match the frozen protocol")
    if args.expected_source_revision is not None and (
        workload_metadata.get("source_revisions") != [args.expected_source_revision]
    ):
        raise SystemExit("LongBench source revision does not match the frozen protocol")
    if args.expected_source_indices is not None:
        actual_indices = sorted(
            {int(workload["source_index"]) for workload in workloads}
        )
        if actual_indices != sorted(set(args.expected_source_indices)):
            raise SystemExit(
                f"source indices {actual_indices} do not match frozen "
                f"{sorted(set(args.expected_source_indices))}"
            )
    if args.expected_workloads is not None and len(workloads) != args.expected_workloads:
        raise SystemExit(
            f"expected {args.expected_workloads} workloads, found {len(workloads)}"
        )
    workloads = workloads[args.rank :: args.world_size]
    if not workloads:
        raise SystemExit("rank has no workload shard")

    configs, mixed_policy = resolve_configs(args)
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    _sync()
    model_load_seconds = time.perf_counter() - load_started
    model_allocated_bytes = torch.cuda.memory_allocated()
    total_device_bytes = torch.cuda.get_device_properties(0).total_memory
    adapter = TorchSplitCausalLM(model)
    nvml = NvmlProcessSampler(visible_nvml_index())

    first = workloads[0]
    gate_document = batch_prefix(
        first["document_tokens"], args.gate_document_tokens
    ).cuda()
    gate_query = batch_prefix(first["query_tokens"], args.gate_query_tokens).cuda()
    eos_value = tokenizer.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else set(eos_value or [])
    exactness_gate = run_exactness_gate(
        adapter,
        gate_document,
        gate_query,
        depth=args.gate_depth,
        group_size=args.group_size,
        max_new_tokens=args.gate_new_tokens,
        eos_token_ids=eos_ids,
        require_exact_logits=args.require_exact_logits,
        logit_atol=args.logit_atol,
        fork_strategy=args.fork_strategy,
    )
    if not exactness_gate["passed"]:
        destination = args.run_dir / f"deployment-shard-{args.rank}.json"
        atomic_json(
            destination,
            {
                "status": "exactness_gate_failed",
                "rank": args.rank,
                "world_size": args.world_size,
                "workload": args.workload,
                "workload_id": first["workload_id"],
                "source_index": first["source_index"],
                "workload_metadata": workload_metadata,
                "exactness_gate": exactness_gate,
            },
        )
        raise SystemExit("incremental exactness gate failed")
    if args.gate_only:
        destination = args.run_dir / f"deployment-shard-{args.rank}.json"
        atomic_json(
            destination,
            {
                "status": "exactness_gate_passed",
                "rank": args.rank,
                "world_size": args.world_size,
                "model": str(args.model),
                "workload": args.workload,
                "workload_id": first["workload_id"],
                "source_index": first["source_index"],
                "workload_metadata": workload_metadata,
                "exactness_gate": exactness_gate,
                "environment": environment_metadata(model),
                "transformers": transformers.__version__,
            },
        )
        print(f"EXACTNESS_GATE_PASSED {destination}", flush=True)
        return
    del gate_document, gate_query
    torch.cuda.empty_cache()

    for _ in range(args.warmups):
        warmup_order = list(configs)
        random.shuffle(warmup_order)
        for config in warmup_order:
            warmup_config(
                adapter,
                config,
                first["document_tokens"].cuda(),
                first["query_tokens"].cuda(),
                group_size=args.group_size,
                eos_ids=eos_ids,
                fork_strategy=(
                    args.fork_strategy if config.mode == "qcomem" else "deep-clone"
                ),
            )

    config_names = [config.name for config in configs]
    config_by_name = {config.name: config for config in configs}
    orders_by_workload: dict[str, list[list[str]]] = {}
    rows = []
    destination = args.run_dir / f"deployment-shard-{args.rank}.json"
    base_result = {
        "status": "running",
        "rank": args.rank,
        "world_size": args.world_size,
        "model": str(args.model),
        "model_load_seconds": model_load_seconds,
        "model_cuda_allocated_baseline_bytes": model_allocated_bytes,
        "environment": environment_metadata(model),
        "transformers": transformers.__version__,
        "nvml_sampler": nvml.metadata(),
        "workload": args.workload,
        "workload_metadata": workload_metadata,
        "protocol": {
            "label": args.protocol_label,
            "limit_per_dataset": args.limit_per_dataset,
            "source_index_start": args.source_index_start,
            "source_index_end": args.source_index_end,
            "expected_source_indices": args.expected_source_indices,
            "expected_workloads": args.expected_workloads,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens": args.max_new_tokens,
            "gate_document_tokens": args.gate_document_tokens,
            "gate_query_tokens": args.gate_query_tokens,
            "gate_new_tokens": args.gate_new_tokens,
            "expected_data_sha256": args.expected_data_sha256,
            "expected_source_revision": args.expected_source_revision,
        },
        "mixed_policy": mixed_policy,
        "fork_strategy": args.fork_strategy,
        "configs": [config_asdict(config) for config in configs],
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "exactness_gate": exactness_gate,
        "rows": rows,
    }
    atomic_json(destination, base_result)

    for workload_index, workload in enumerate(workloads):
        orders = shuffled_config_orders(
            config_names,
            repeats=args.repeats,
            seed=args.seed + args.rank * 100_000 + workload_index * 1_000,
        )
        orders_by_workload[workload["workload_id"]] = orders
        for repeat, order in enumerate(orders):
            for order_position, name in enumerate(order):
                row = measure_one(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    config=config_by_name[name],
                    workload=workload,
                    repeat=repeat,
                    order_position=order_position,
                    model_allocated_bytes=model_allocated_bytes,
                    total_device_bytes=total_device_bytes,
                    nvml=nvml,
                    args=args,
                )
                rows.append(row)
                base_result["rows"] = rows
                base_result["randomized_orders"] = orders_by_workload
                atomic_json(destination, base_result)
                print(
                    json.dumps(
                        {
                            "rank": args.rank,
                            "workload": workload["workload_id"],
                            "repeat": repeat,
                            "config": name,
                            "persistent_mib": row["persistent_document_nbytes"] / 2**20,
                            "peak_gib": row["cuda_peak_allocated_bytes"] / 2**30,
                            "ttft_seconds": row["ttft_seconds"],
                            "median_tpot_seconds": row["median_tpot_seconds"],
                        }
                    ),
                    flush=True,
                )

    base_result["status"] = "completed"
    base_result["rows"] = rows
    base_result["randomized_orders"] = orders_by_workload
    atomic_json(destination, base_result)
    print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
