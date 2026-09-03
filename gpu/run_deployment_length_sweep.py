"""A4/A5 runner: honest dense baseline, generation-length sweep, quantized exact caches.

This is a new runner beside ``run_deployment_bench.py``, not a replacement for
it.  It reuses that script's workload construction verbatim (identical LongBench
slicing, identical prompt protocol, identical SHA/revision guards) and reuses
``qcomem_deployment``'s timing, memory recorder and capacity estimator.  What it
adds:

A4
  * the ``dense-prefill-once`` arm -- prefill document+query fresh per query,
    then decode against a within-request KV cache;
  * a generation-length sweep, so the same items run at several
    ``max_new_tokens`` values;
  * ``--eos-policy ignore`` so the cap is actually reached instead of being cut
    short by EOS.  With ``--eos-policy stop`` the true per-item token count is
    recorded and nothing is imputed;
  * per-row per-step decode latencies, an end-to-end wall clock, and the
    measured tok/s beside the ``n / (TTFT + n * TPOT)`` model it replaces.

A5
  * ``full-prefix-q8`` / ``full-prefix-q4`` / ``full-prefix-frozen-static`` --
    a full-prefix exact cache put through the same Eq. 3 packer, group size and
    BF16 scale/bias metadata as the split-replay rows.

Both
  * a per-component byte breakdown for every row (packed codes, scales, biases,
    per state type and per layer) plus a dtype-consistent reference count, so a
    native-dtype ratio and an all-BF16 ratio are both computable afterwards;
  * an assertion, on every row, that each quantized component occupies
    ``ceil(n / 64) * (64 b / 8 + 4)`` bytes for its declared width.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from qcomem_deployment import (
    DEFAULT_MIXED_LAYER_BITS,
    MemoryRecorder,
    NvmlProcessSampler,
    capacity_estimate,
    config_asdict,
    environment_metadata,
    load_mixed_policy,
    parse_layer_bits,
    run_exactness_gate,
)
from qcomem_deployment_arms import (
    DEFAULT_CONFIG_LENGTH_LIMITS,
    DEFAULT_GENERATION_LENGTHS,
    DEFAULT_SWEEP_CONFIGS,
    StridedMemoryRecorder,
    auto_decode_stride,
    build_extended_persistent_state,
    parse_extended_deployment_config,
    persistent_components_extended,
    run_dense_semantics_gate,
    run_extended_generation,
    run_full_prefix_quant_gate,
    store_breakdown_for_state,
    warmup_extended_config,
)
from qcomem_eq3_accounting import (
    GROUP_SIZE,
    arm_name,
    decode_latency_summary,
    parse_config_length_limits,
    shuffled_arm_orders,
    summarize_arm,
    sweep_arms,
    throughput_summary,
    validate_row,
)
from qcomem_torch import TorchSplitCausalLM
from run_deployment_bench import (
    batch_prefix,
    longbench_workloads,
    synthetic_workloads,
    visible_nvml_index,
)
from run_downstream import answer_f1, atomic_json, generation_limit


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def resolve_sweep_configs(args) -> tuple[list[Any], dict[str, Any]]:
    """Parse arm names, honouring the same mixed-policy file the bench uses."""

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
        config = parse_extended_deployment_config(
            name, mixed_layer_bits=mixed_layer_bits
        )
        if name.endswith("-mixed"):
            config = replace(config, residual_bits=mixed_residual_bits)
        configs.append(config)
    if len({config.name for config in configs}) != len(configs):
        raise ValueError("--configs contains duplicate names")
    return configs, policy_metadata


@torch.inference_mode()
def measure_arm(
    *,
    adapter,
    tokenizer,
    config,
    workload,
    max_new_tokens: int,
    repeat: int,
    order_position: int,
    model_allocated_bytes: int,
    total_device_bytes: int,
    nvml,
    args,
    eos_ids: set[int],
) -> dict[str, Any]:
    """Measure one (workload, configuration, generation-length) cell."""

    document = workload["document_tokens"].cuda()
    query = workload["query_tokens"].cuda()
    dataset_limit = (
        generation_limit(workload["dataset"], max_new_tokens)
        if workload["kind"] == "longbench"
        else max_new_tokens
    )
    if args.generation_limit_policy == "dataset":
        effective_new_tokens = dataset_limit
        dataset_limit_applied = dataset_limit < max_new_tokens
    else:
        effective_new_tokens = max_new_tokens
        dataset_limit_applied = False
    stop_ids = set() if args.eos_policy == "ignore" else set(eos_ids)

    torch.cuda.empty_cache()
    build_recorder = MemoryRecorder(nvml)
    build_recorder.reset_peak()
    build_start = build_recorder.sample("build_start")
    _sync()
    end_to_end_started = time.perf_counter()
    build_started = time.perf_counter()
    persistent = build_extended_persistent_state(
        adapter,
        config,
        document,
        group_size=args.group_size,
        fork_strategy=(
            args.fork_strategy if config.mode == "qcomem" else "deep-clone"
        ),
    )
    _sync()
    build_seconds = time.perf_counter() - build_started
    build_recorder.sample("build_end")
    build_memory = build_recorder.summary(steady_prefix="build_end")
    components = persistent_components_extended(persistent)
    store_breakdown = store_breakdown_for_state(
        persistent,
        group_size=args.group_size,
        strict=args.strict_accounting,
    )

    torch.cuda.empty_cache()
    stride = (
        auto_decode_stride(effective_new_tokens)
        if args.decode_sample_stride <= 0
        else args.decode_sample_stride
    )
    request_recorder = StridedMemoryRecorder(
        nvml, decode_stride=stride, decode_steps=effective_new_tokens
    )
    _sync()
    request_started = time.perf_counter()
    trace = run_extended_generation(
        adapter,
        config,
        document,
        query,
        persistent,
        max_new_tokens=effective_new_tokens,
        eos_token_ids=stop_ids,
        recorder=request_recorder,
    )
    _sync()
    request_wall_seconds = time.perf_counter() - request_started
    end_to_end_seconds = time.perf_counter() - end_to_end_started

    trace_summary = trace.summary()
    if args.drop_memory_samples:
        trace_summary.pop("memory_samples", None)
        build_memory.pop("memory_samples", None)
    fork_memory = trace_summary["fork_memory"]
    request_start_allocated = int(
        (trace.memory.get("memory_samples") or [{}])[0].get(
            "cuda_allocated_bytes", 0
        )
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
    generated_ids = trace.generated_token_ids
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    references = workload["references"]
    f1 = (
        max(answer_f1(prediction, reference) for reference in references)
        if references
        else None
    )
    natural_eos_step = next(
        (index for index, token in enumerate(generated_ids) if token in eos_ids),
        None,
    )
    decode_latency = decode_latency_summary(trace.tpot_seconds)
    throughput = throughput_summary(
        generated_tokens=len(generated_ids),
        ttft_seconds=trace.ttft_seconds,
        tpot_seconds=trace.tpot_seconds,
        online_seconds=trace.online_seconds,
        request_wall_seconds=request_wall_seconds,
        end_to_end_including_build_seconds=end_to_end_seconds,
    )
    row = {
        "arm": arm_name(config.name, max_new_tokens),
        "config": config.name,
        "mode": config.mode,
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
        # --- generation-length sweep bookkeeping -------------------------
        "max_new_tokens_requested": int(max_new_tokens),
        "max_new_tokens_effective": int(effective_new_tokens),
        "dataset_generation_limit": int(dataset_limit),
        "dataset_generation_limit_applied": bool(dataset_limit_applied),
        "generation_limit_policy": args.generation_limit_policy,
        "eos_policy": args.eos_policy,
        "eos_stopped": bool(len(generated_ids) < effective_new_tokens),
        "reached_cap": bool(len(generated_ids) == effective_new_tokens),
        "natural_eos_step": natural_eos_step,
        "eos_token_ids": sorted(eos_ids),
        # --- timing -------------------------------------------------------
        "decode_latency": decode_latency,
        "throughput": throughput,
        "request_wall_seconds": request_wall_seconds,
        "end_to_end_including_build_seconds": end_to_end_seconds,
        "write_build_seconds": build_seconds,
        # --- prediction ---------------------------------------------------
        "prediction": prediction,
        "references": references,
        "f1": f1,
        # --- memory / store ------------------------------------------------
        "build_start_cuda_allocated_bytes": int(build_start["cuda_allocated_bytes"]),
        "build_memory": build_memory,
        **components,
        "store_breakdown": store_breakdown,
        "capacity_document_denominator_nbytes": capacity_document_bytes,
        **trace_summary,
        "cow_initial_shared_nbytes": fork_memory.get("initial_shared_nbytes"),
        "cow_initial_private_nbytes": fork_memory.get("initial_private_nbytes"),
        "cow_after_query_shared_nbytes": fork_memory.get("after_query_shared_nbytes"),
        "cow_after_query_private_nbytes": fork_memory.get(
            "after_query_private_nbytes"
        ),
        "cow_final_shared_nbytes": fork_memory.get("final_shared_nbytes"),
        "cow_final_private_nbytes": fork_memory.get("final_private_nbytes"),
        **capacity,
        "model_cuda_allocated_baseline_bytes": model_allocated_bytes,
        "device_total_memory_bytes": total_device_bytes,
        "capacity_safety_headroom_bytes": round(args.safety_headroom_gib * 2**30),
        "decode_sample_stride": stride,
    }
    problems = validate_row(row)
    if problems:
        row["row_validation_problems"] = problems
        if args.strict_accounting:
            raise SystemExit(
                f"row for {row['arm']} / {row['workload_id']} is invalid: {problems}"
            )
    del persistent, document, query, trace
    torch.cuda.empty_cache()
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "A4/A5: honest dense baseline, generation-length sweep and "
            "quantized exact-cache baselines for the deployment benchmark"
        )
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
    parser.add_argument("--protocol-label", default="a4-a5-length-sweep")
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--source-index-start", type=int, default=6)
    parser.add_argument("--source-index-end", type=int, default=35)
    parser.add_argument("--exclude-source-indices", type=int, nargs="*", default=(4, 5))
    parser.add_argument("--allow-test-v2", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--context-lengths", type=int, nargs="+", default=(4096, 8192, 16384, 32768)
    )
    parser.add_argument("--synthetic-repetitions", type=int, default=2)
    parser.add_argument("--configs", nargs="+", default=list(DEFAULT_SWEEP_CONFIGS))
    parser.add_argument(
        "--max-new-tokens-sweep",
        type=int,
        nargs="+",
        default=list(DEFAULT_GENERATION_LENGTHS),
        help="generation lengths every arm is run at (A4 requirement 3)",
    )
    parser.add_argument(
        "--config-length-limit",
        nargs="*",
        default=list(DEFAULT_CONFIG_LENGTH_LIMITS),
        metavar="NAME=LIMIT",
        help=(
            "cap the generation lengths one configuration runs at; "
            "dense-recompute is quadratic in the length and defaults to 8"
        ),
    )
    parser.add_argument(
        "--eos-policy",
        choices=("ignore", "stop"),
        default="ignore",
        help=(
            "ignore: never stop on EOS, so the requested cap is always reached; "
            "stop: honour EOS and record the true per-item token count"
        ),
    )
    parser.add_argument(
        "--generation-limit-policy",
        choices=("fixed", "dataset"),
        default="fixed",
        help=(
            "fixed: every arm runs at the swept length; dataset: additionally "
            "clamp to the LongBench per-dataset generation limit"
        ),
    )
    parser.add_argument(
        "--mixed-layer-bits",
        default=",".join(str(bit) for bit in DEFAULT_MIXED_LAYER_BITS),
    )
    parser.add_argument("--mixed-policy-file", type=Path)
    parser.add_argument("--mixed-policy-name", default="same_memory_as_frozen")
    parser.add_argument("--group-size", type=int, default=GROUP_SIZE)
    parser.add_argument(
        "--fork-strategy",
        choices=("deep-clone", "paged-cow-staging"),
        default="deep-clone",
    )
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--gate-depth", type=int, default=7)
    parser.add_argument("--gate-document-tokens", type=int, default=256)
    parser.add_argument("--gate-query-tokens", type=int, default=64)
    parser.add_argument("--gate-new-tokens", type=int, default=4)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--skip-published-exactness-gate", action="store_true")
    parser.add_argument("--require-exact-logits", action="store_true")
    parser.add_argument("--logit-atol", type=float, default=0.0)
    parser.add_argument("--safety-headroom-gib", type=float, default=4.0)
    parser.add_argument(
        "--decode-sample-stride",
        type=int,
        default=0,
        help="0 selects a stride keeping at most 64 decode memory samples",
    )
    parser.add_argument("--drop-memory-samples", action="store_true")
    parser.add_argument(
        "--no-strict-accounting",
        dest="strict_accounting",
        action="store_false",
        help=(
            "do not raise on an Eq. 3 byte-identity failure; the violation is "
            "still recorded in every row"
        ),
    )
    parser.set_defaults(strict_accounting=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if not (0 <= args.rank < args.world_size):
        raise SystemExit("rank must be within world size")
    if args.warmups < 1 or args.repeats < 1:
        raise SystemExit("formal runs require at least one warmup and one repeat")
    if args.group_size != GROUP_SIZE:
        print(
            f"WARNING: group size {args.group_size} is not the published 64; "
            "the Eq. 3 identity is checked at the requested group size",
            flush=True,
        )

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
        actual_indices = sorted({int(item["source_index"]) for item in workloads})
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

    configs, mixed_policy = resolve_sweep_configs(args)
    config_by_name = {config.name: config for config in configs}
    # The default length limit names dense-recompute.  A run that does not
    # include that arm must not die on an unknown-config error, so limits are
    # narrowed to the declared configs and the dropped ones are recorded rather
    # than silently discarded.
    requested_limits = parse_config_length_limits(args.config_length_limit)
    ignored_limits = sorted(set(requested_limits) - set(config_by_name))
    applied_limits = {
        name: limit
        for name, limit in requested_limits.items()
        if name in config_by_name
    }
    arms = sweep_arms(
        [config.name for config in configs],
        args.max_new_tokens_sweep,
        config_length_limits=applied_limits,
    )
    arm_by_name = {arm["arm"]: arm for arm in arms}
    arm_names = [arm["arm"] for arm in arms]

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

    destination = args.run_dir / f"length-sweep-shard-{args.rank}.json"
    gates: dict[str, Any] = {}
    if not args.skip_published_exactness_gate:
        gates["published_exactness_gate"] = run_exactness_gate(
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
    gates["dense_semantics_gate"] = run_dense_semantics_gate(
        adapter,
        gate_document,
        gate_query,
        group_size=args.group_size,
        max_new_tokens=args.gate_new_tokens,
        eos_token_ids=eos_ids,
        logit_atol=args.logit_atol,
    )
    quantized_full_prefix_arms = [
        config.name for config in configs if config.mode == "full_prefix_quantized"
    ]
    if quantized_full_prefix_arms:
        gates["full_prefix_quant_gate"] = run_full_prefix_quant_gate(
            adapter,
            gate_document,
            gate_query,
            config_names=quantized_full_prefix_arms,
            group_size=args.group_size,
            max_new_tokens=args.gate_new_tokens,
            eos_token_ids=eos_ids,
        )
    gates_passed = all(
        bool(gate.get("passed")) for gate in gates.values() if isinstance(gate, dict)
    )
    if not gates_passed:
        atomic_json(
            destination,
            {
                "status": "gate_failed",
                "rank": args.rank,
                "world_size": args.world_size,
                "workload": args.workload,
                "workload_id": first["workload_id"],
                "workload_metadata": workload_metadata,
                "gates": gates,
            },
        )
        raise SystemExit("A4/A5 gate failed")
    if args.gate_only:
        atomic_json(
            destination,
            {
                "status": "gate_passed",
                "rank": args.rank,
                "world_size": args.world_size,
                "model": str(args.model),
                "workload": args.workload,
                "workload_id": first["workload_id"],
                "workload_metadata": workload_metadata,
                "gates": gates,
                "environment": environment_metadata(model),
                "transformers": transformers.__version__,
            },
        )
        print(f"A4_A5_GATE_PASSED {destination}", flush=True)
        return
    del gate_document, gate_query
    torch.cuda.empty_cache()

    for _ in range(args.warmups):
        warmup_order = list(configs)
        random.shuffle(warmup_order)
        for config in warmup_order:
            warmup_extended_config(
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

    rows: list[dict[str, Any]] = []
    orders_by_workload: dict[str, list[list[str]]] = {}
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
            "family": "a4-a5",
            "limit_per_dataset": args.limit_per_dataset,
            "source_index_start": args.source_index_start,
            "source_index_end": args.source_index_end,
            "expected_source_indices": args.expected_source_indices,
            "expected_workloads": args.expected_workloads,
            "max_input_tokens": args.max_input_tokens,
            "max_new_tokens_sweep": list(args.max_new_tokens_sweep),
            "config_length_limits_requested": list(args.config_length_limit),
            "config_length_limits_applied": applied_limits,
            "config_length_limits_ignored": ignored_limits,
            "eos_policy": args.eos_policy,
            "generation_limit_policy": args.generation_limit_policy,
            "group_size": args.group_size,
            "strict_accounting": args.strict_accounting,
            "decode_sample_stride": args.decode_sample_stride,
            "expected_data_sha256": args.expected_data_sha256,
            "expected_source_revision": args.expected_source_revision,
        },
        "mixed_policy": mixed_policy,
        "fork_strategy": args.fork_strategy,
        "configs": [config_asdict(config) for config in configs],
        "arms": arms,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "seed": args.seed,
        "gates": gates,
        "rows": rows,
    }
    args.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(destination, base_result)

    for workload_index, workload in enumerate(workloads):
        orders = shuffled_arm_orders(
            arm_names,
            repeats=args.repeats,
            seed=args.seed + args.rank * 100_000 + workload_index * 1_000,
        )
        orders_by_workload[workload["workload_id"]] = orders
        for repeat, order in enumerate(orders):
            for order_position, name in enumerate(order):
                arm = arm_by_name[name]
                row = measure_arm(
                    adapter=adapter,
                    tokenizer=tokenizer,
                    config=config_by_name[arm["config"]],
                    workload=workload,
                    max_new_tokens=arm["max_new_tokens"],
                    repeat=repeat,
                    order_position=order_position,
                    model_allocated_bytes=model_allocated_bytes,
                    total_device_bytes=total_device_bytes,
                    nvml=nvml,
                    args=args,
                    eos_ids=eos_ids,
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
                            "arm": name,
                            "generated_tokens": row["generated_tokens"],
                            "store_mib": row["persistent_document_nbytes"] / 2**20,
                            "bf16_reference_mib": row["store_breakdown"][
                                "bf16_reference_nbytes"
                            ]
                            / 2**20,
                            "ttft_seconds": row["ttft_seconds"],
                            "decode_median_seconds": row["decode_latency"][
                                "decode_seconds_median"
                            ],
                            "wall_tokens_per_second": row["throughput"][
                                "wall_tokens_per_second"
                            ],
                        }
                    ),
                    flush=True,
                )

    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    base_result["arm_summaries"] = [
        summarize_arm(arm_rows) for _, arm_rows in sorted(by_arm.items())
    ]
    base_result["status"] = "completed"
    base_result["rows"] = rows
    base_result["randomized_orders"] = orders_by_workload
    atomic_json(destination, base_result)
    print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
