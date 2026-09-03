from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from qcomem_lora import CoMemLoRADistillation, PG19WindowDataset, ReplayQuantConfig, cached_two_stage_autograd_capability_gate, single_window_collate
from qcomem_suffix_full import (
    aggregate_suffix_gradient_coverage,
    configure_suffix_full_trainability,
    end_to_end_full_model_capability_gate,
    estimate_sharded_training_storage,
    observed_optimizer_state,
    suffix_full_semantics_metadata,
    suffix_gradient_coverage_local,
)
from qcomem_torch import TorchSplitCausalLM
from train_qcomem_lora import cosine_warmup_decay, parse_layer_bits, reject_frozen_test_data


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name, delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args()
    defaults = json.loads(known.config.read_text()) if known.config is not None else {}

    parser = argparse.ArgumentParser(
        description=(
            "FSDP suffix_full_distillation capacity upper bound. This is not "
            "end-to-end full-model SFT."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--training-scope", choices=("suffix_full_distillation", "end_to_end_full_model_sft_qat"), default="suffix_full_distillation")
    parser.add_argument("--model", type=Path, required="model" not in defaults)
    parser.add_argument("--data", type=Path, required="data" not in defaults)
    parser.add_argument("--output-dir", type=Path, required="output_dir" not in defaults)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--teacher-source", choices=("online", "offline"), default="online")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--forward-weight", type=float, default=0.6)
    parser.add_argument("--reverse-weight", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--loss-kind", choices=("teacher_topk_bidirectional_kl", "token_ce_sft"), default="teacher_topk_bidirectional_kl")
    parser.add_argument("--context-tokens", type=int, default=512)
    parser.add_argument("--query-tokens", type=int, default=128)
    parser.add_argument("--stride", type=int, default=640)
    parser.add_argument("--dataset-limit", type=int, default=64)
    parser.add_argument("--max-windows-per-record", type=int, default=1)
    parser.add_argument("--residual-bits", type=int, default=4)
    parser.add_argument("--attention-bits", type=int, default=4)
    parser.add_argument("--linear-bits", type=int, default=8)
    parser.add_argument("--cache-layer-bits", default="8,8,8,4,8,8,8")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--student-suffix-execution", choices=("cached-two-stage", "detached-document-cache"), default="detached-document-cache")
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-trainable-params", type=int, default=27_000_000_000)
    parser.add_argument("--max-trainable-params", type=int, default=29_000_000_000)
    parser.add_argument("--expected-trainable-params", type=int, default=27_751_037_952)
    parser.add_argument("--expected-attention-params", type=int, default=1_054_614_528)
    parser.add_argument("--expected-mlp-moe-params", type=int, default=26_696_288_256)
    parser.add_argument("--expected-world-size", type=int, default=8)
    parser.add_argument("--checkpoint-mode", choices=("metadata-only",), default="metadata-only")
    known_destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - known_destinations)
    if unknown:
        parser.error(f"unknown config keys: {', '.join(unknown)}")
    parser.set_defaults(**defaults)
    return parser


def distributed_setup(expected_world_size: int) -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != expected_world_size or world_size < 2:
        raise SystemExit(
            f"suffix_full_distillation requires exactly {expected_world_size} ranks; "
            "DDP/single-GPU replication is forbidden"
        )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA with native BF16 support is required")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size, torch.device("cuda", local_rank)


def checkpoint_suffix_mlp_modules(layers: torch.nn.ModuleList, depth: int) -> list[str]:
    """Checkpoint stateless suffix MLP/MoE blocks, not mutable cache attention.

    Whole-layer checkpointing would replay DynamicCache/GatedDeltaNet mutations
    during backward.  The MLP/MoE blocks contain almost all suffix parameters
    and have no cache side effects, so this is the safe activation-memory scope.
    """

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )

    wrapped: list[str] = []
    for index in range(depth, len(layers)):
        layer = layers[index]
        if not hasattr(layer, "mlp"):
            raise RuntimeError(f"suffix layer {index} has no MLP/MoE module to checkpoint")
        layer.mlp = checkpoint_wrapper(
            layer.mlp,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        wrapped.append(f"layers.{index}.mlp")
    return wrapped


def all_rank_trainable_count(module: torch.nn.Module, device: torch.device) -> tuple[int, list[int]]:
    local = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    tensor = torch.tensor([local], dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    values = [int(item.item()) for item in gathered]
    return sum(values), values


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.training_scope == "end_to_end_full_model_sft_qat":
        capability = end_to_end_full_model_capability_gate()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / "capability-gate.json", capability)
        print(json.dumps(capability, ensure_ascii=False, indent=2))
        raise SystemExit("end-to-end full-model SFT/QAT capability gate is not passed")
    if args.loss_kind != "teacher_topk_bidirectional_kl":
        parser.error("token CE SFT is a separate unimplemented ablation")
    if args.student_suffix_execution == "cached-two-stage":
        capability = cached_two_stage_autograd_capability_gate()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / "cached-autograd-capability-gate.json", capability)
        raise SystemExit(
            "cached-two-stage mutable-cache backward failed on Trial 1830867; "
            "suffix-full refuses to allocate FSDP resources for this execution"
        )
    if args.steps != 1:
        parser.error(
            "suffix-full is smoke-only until DCP sharded checkpoint/resume is implemented; "
            "require --steps 1 even with offline teacher targets"
        )
    if args.gradient_accumulation < 1:
        parser.error("gradient accumulation must be positive")

    local_rank, rank, world_size, device = distributed_setup(args.expected_world_size)
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data_sha256 = reject_frozen_test_data(args.data)

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = PG19WindowDataset(
        args.data,
        tokenizer,
        context_tokens=args.context_tokens,
        query_tokens=args.query_tokens,
        stride=args.stride,
        limit=args.dataset_limit,
        max_windows_per_record=args.max_windows_per_record,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed, drop_last=False)
    loader = DataLoader(dataset, batch_size=1, sampler=sampler, collate_fn=single_window_collate, num_workers=0)

    # This loader has a transient full BF16 model per H20 before FSDP shards it.
    # It is not DDP training replication; the transient peak is recorded and is
    # an explicit limitation pending a rank-aware sharded safetensors loader.
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().to(device)
    split = TorchSplitCausalLM(model)
    layer_classes = {type(layer) for layer in split.layers}
    plan = configure_suffix_full_trainability(model, split.layers, depth=args.depth)
    trainable_count = int(plan["actual_trainable_parameters"])
    if not args.min_trainable_params <= trainable_count <= args.max_trainable_params:
        raise SystemExit(
            f"suffix trainable count {trainable_count:,} is outside hard gate "
            f"[{args.min_trainable_params:,}, {args.max_trainable_params:,}]"
        )
    category_counts = plan["category_parameter_counts"]
    exact_expectations = {
        "actual_trainable_parameters": args.expected_trainable_params,
        "attention": args.expected_attention_params,
        "mlp_moe": args.expected_mlp_moe_params,
        "normalization_or_other": (
            args.expected_trainable_params
            - args.expected_attention_params
            - args.expected_mlp_moe_params
        ),
    }
    exact_actual = {
        "actual_trainable_parameters": trainable_count,
        **category_counts,
    }
    if exact_actual != exact_expectations:
        raise SystemExit(
            "frozen-model suffix parameter preflight changed: "
            f"expected={exact_expectations}, actual={exact_actual}"
        )
    plan["frozen_model_exact_expectations"] = exact_expectations
    plan["frozen_model_exact_gate_passed"] = True
    activation_checkpoint_modules = checkpoint_suffix_mlp_modules(split.layers, args.depth)

    quant = ReplayQuantConfig(
        residual_bits=args.residual_bits,
        attention_bits=args.attention_bits,
        linear_bits=args.linear_bits,
        cache_layer_bits=parse_layer_bits(args.cache_layer_bits),
        group_size=args.group_size,
    )
    core = CoMemLoRADistillation(
        model,
        mode="quant",
        depth=args.depth,
        top_k=args.top_k,
        chunk_size=args.context_tokens,
        overlap=0,
        teacher_kind="q16_replay",
        teacher_source=args.teacher_source,
        quant=quant,
        forward_weight=args.forward_weight,
        reverse_weight=args.reverse_weight,
        temperature=args.temperature,
        student_suffix_execution=args.student_suffix_execution,
    )

    from torch.distributed.fsdp import BackwardPrefetch, FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=layer_classes,
    )
    fsdp = FSDP(
        core,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        limit_all_gathers=True,
        device_id=device,
    )
    global_sharded_count, per_rank_counts = all_rank_trainable_count(fsdp, device)
    if global_sharded_count != trainable_count:
        raise RuntimeError(
            f"FSDP sharded trainable count {global_sharded_count:,} != preflight {trainable_count:,}"
        )
    trainable = [parameter for parameter in fsdp.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = cosine_warmup_decay(optimizer, warmup_steps=args.warmup_steps, total_steps=args.steps)
    storage = estimate_sharded_training_storage(
        trainable_parameters=trainable_count,
        parameter_bytes=int(plan["trainable_logical_bytes"]),
        world_size=world_size,
    )
    semantics = suffix_full_semantics_metadata(
        depth=args.depth,
        teacher_source=args.teacher_source,
        quant=quant,
        student_suffix_execution=args.student_suffix_execution,
    )
    metadata: dict[str, Any] = {
        "created_unix": time.time(),
        "model": str(args.model),
        "data": str(args.data),
        "data_sha256": data_sha256,
        "test_v2_used": False,
        "smoke_only": True,
        "world_size": world_size,
        "distributed": {
            "kind": "FSDP1_FULL_SHARD",
            "ddp_used": False,
            "use_orig_params": True,
            "mixed_precision": {"parameter": "bfloat16", "reduce": "bfloat16", "buffer": "bfloat16"},
            "per_rank_trainable_shard_numel": per_rank_counts,
            "global_sharded_trainable_numel": global_sharded_count,
            "transient_loader_full_bf16_replica_per_rank": True,
        },
        "activation_checkpoint": {
            "enabled": True,
            "implementation": "NO_REENTRANT checkpoint_wrapper",
            "scope": "suffix MLP/MoE only",
            "modules": activation_checkpoint_modules,
            "reason_not_whole_layer": "attention mutates DynamicCache/GatedDeltaNet state; replaying that side effect in backward is unsafe",
        },
        "parameter_plan": plan,
        "storage_estimate": storage,
        "checkpoint": {
            "mode": args.checkpoint_mode,
            "full_rank0_gather_forbidden": True,
            "artifact_written": False,
            "note": "1-step capacity smoke records a strict byte ledger but does not spend 55+ GiB on a model-only suffix checkpoint",
        },
        "loss": {
            "kind": "teacher_topk_bidirectional_kl_on_query_positions",
            "top_k": args.top_k,
            "forward_weight": args.forward_weight,
            "reverse_weight": args.reverse_weight,
            "temperature": args.temperature,
            "token_ce_sft_is_separate_unimplemented_ablation": True,
        },
        "training": {
            "steps": args.steps,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "context_tokens": args.context_tokens,
            "query_tokens": args.query_tokens,
            "seed": args.seed,
        },
        "semantics": semantics,
        "end_to_end_full_model_capability": end_to_end_full_model_capability_gate(),
    }
    if rank == 0:
        atomic_json(args.output_dir / "metadata.json", metadata)
        print(json.dumps({"phase": "suffix_full_preflight", "parameter_plan": plan, "storage_estimate": storage}, ensure_ascii=False), flush=True)

    sampler.set_epoch(0)
    iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, args.steps + 1):
        metric_sums = {"loss": 0.0, "forward_kl": 0.0, "reverse_kl": 0.0, "persistent_nbytes": 0.0}
        for _ in range(args.gradient_accumulation):
            try:
                window = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                window = next(iterator)
            document = window.document_ids.to(device, non_blocking=True)
            query = window.query_ids.to(device, non_blocking=True)
            indices = window.teacher_topk_indices.to(device) if window.teacher_topk_indices is not None else None
            logits = window.teacher_topk_logits.to(device) if window.teacher_topk_logits is not None else None
            output = fsdp(document, query, indices, logits)
            (output["loss"] / args.gradient_accumulation).backward()
            for key in metric_sums:
                value = output[key]
                metric_sums[key] += (float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)) / args.gradient_accumulation
        local_gradient_coverage = suffix_gradient_coverage_local(
            fsdp,
            depth=args.depth,
            num_layers=split.num_layers,
        )
        gradient_coverage_by_rank: list[Any] = [None] * world_size
        dist.all_gather_object(gradient_coverage_by_rank, local_gradient_coverage)
        gradient_coverage = aggregate_suffix_gradient_coverage(
            gradient_coverage_by_rank,
            depth=args.depth,
            num_layers=split.num_layers,
        )
        metadata["last_gradient_coverage"] = {
            "step": step,
            **gradient_coverage,
        }
        if rank == 0:
            atomic_json(args.output_dir / "metadata.json", metadata)
        if not gradient_coverage["hard_gate_passed"]:
            raise RuntimeError("suffix-full gradient coverage hard gate failed")
        grad_norm = fsdp.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        packed = torch.tensor([metric_sums[key] for key in metric_sums], dtype=torch.float64, device=device)
        dist.all_reduce(packed)
        packed /= world_size
        row = {
            "step": step,
            "loss": packed[0].item(),
            "forward_kl": packed[1].item(),
            "reverse_kl": packed[2].item(),
            "mean_persistent_nbytes": packed[3].item(),
            "grad_norm": float(grad_norm.detach().item()),
            "learning_rate": scheduler.get_last_lr()[0],
            "gradient_coverage": gradient_coverage,
        }
        if rank == 0:
            append_jsonl(args.output_dir / "metrics.jsonl", row)
            print(json.dumps(row), flush=True)

    local_optimizer = observed_optimizer_state(optimizer)
    optimizer_by_rank: list[Any] = [None] * world_size
    dist.all_gather_object(optimizer_by_rank, local_optimizer)
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)
    peaks: list[Any] = [None] * world_size
    dist.all_gather_object(peaks, {"rank": rank, "max_allocated_bytes": peak_allocated, "max_reserved_bytes": peak_reserved})
    if rank == 0:
        metadata["last_step"] = args.steps
        metadata["observed_optimizer_state_by_rank"] = optimizer_by_rank
        metadata["runtime_cuda_peaks_after_fsdp_wrap"] = peaks
        metadata["finished_unix"] = time.time()
        atomic_json(args.output_dir / "metadata.json", metadata)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
