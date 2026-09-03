from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from qcomem_torch import TorchSplitCausalLM
from supervised_sft import (
    DenseSupervisedCausalLM,
    SupervisedSFTDataset,
    configure_dense_full_model_trainability,
    qcomem_suffix_supervised_sft_capability_gate,
    single_example_collate,
    validate_formal_integrity_ledgers,
    validate_prepared_training_manifest,
    validate_runtime_tokenizer_against_manifest,
)


FROZEN_TEXT_PARAMETER_COUNT = 34_660_610_688
FROZEN_TEXT_LAYER_COUNT = 40
FROZEN_WORLD_SIZE = 8
FROZEN_SMOKE_DATASET_COUNTS = {"qasper": 4, "2wikimqa": 4}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name, delete=False
    ) as stream:
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
            "True answer-supervised CE smoke. This is deliberately separate from "
            "LoRA and teacher-logit distillation."
        )
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--training-scope",
        choices=("dense_full_model_sft_smoke", "qcomem_suffix_supervised_sft"),
        default="dense_full_model_sft_smoke",
    )
    parser.add_argument("--model", type=Path, required="model" not in defaults)
    parser.add_argument("--data", type=Path, required="data" not in defaults)
    parser.add_argument(
        "--manifest", type=Path, required="manifest" not in defaults
    )
    parser.add_argument(
        "--expected-data-sha256", required="expected_data_sha256" not in defaults
    )
    parser.add_argument(
        "--expected-manifest-sha256",
        required="expected_manifest_sha256" not in defaults,
    )
    parser.add_argument(
        "--output-dir", type=Path, required="output_dir" not in defaults
    )
    parser.add_argument(
        "--code-ledger", type=Path, required="code_ledger" not in defaults
    )
    parser.add_argument(
        "--expected-code-ledger-sha256",
        required="expected_code_ledger_sha256" not in defaults,
    )
    parser.add_argument(
        "--model-artifact-ledger",
        type=Path,
        required="model_artifact_ledger" not in defaults,
    )
    parser.add_argument(
        "--expected-model-artifact-ledger-sha256",
        required="expected_model_artifact_ledger_sha256" not in defaults,
    )
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--dataset-limit", type=int, default=FROZEN_WORLD_SIZE)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--expected-trainable-params",
        type=int,
        default=FROZEN_TEXT_PARAMETER_COUNT,
    )
    parser.add_argument(
        "--expected-num-layers", type=int, default=FROZEN_TEXT_LAYER_COUNT
    )
    parser.add_argument("--expected-world-size", type=int, default=FROZEN_WORLD_SIZE)
    parser.add_argument(
        "--checkpoint-mode", choices=("metadata-only",), default="metadata-only"
    )
    known_destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - known_destinations)
    if unknown:
        parser.error(f"unknown config keys: {', '.join(unknown)}")
    parser.set_defaults(**defaults)
    return parser


def validate_smoke_protocol(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.steps != 1:
        parser.error("supervised full-model SFT is smoke-only: --steps must equal 1")
    if args.gradient_accumulation != 1:
        parser.error(
            "supervised full-model SFT is smoke-only: gradient accumulation must equal 1"
        )
    if args.expected_world_size != FROZEN_WORLD_SIZE:
        parser.error("only exactly 8-rank FSDP FULL_SHARD is permitted")
    if args.dataset_limit != FROZEN_WORLD_SIZE:
        parser.error("the 1-step smoke consumes exactly one distinct example per rank")
    if args.expected_trainable_params != FROZEN_TEXT_PARAMETER_COUNT:
        parser.error(
            "expected_trainable_params must remain pinned to the audited Qwen3.5 "
            f"text model count {FROZEN_TEXT_PARAMETER_COUNT}"
        )
    if args.expected_num_layers != FROZEN_TEXT_LAYER_COUNT:
        parser.error(
            f"expected_num_layers must remain pinned to {FROZEN_TEXT_LAYER_COUNT}"
        )
    if args.max_sequence_tokens < 512:
        parser.error("max_sequence_tokens must leave useful context for the eval prompt")
    if args.learning_rate <= 0:
        parser.error("learning_rate must be positive")
    if args.max_grad_norm <= 0:
        parser.error("max_grad_norm must be positive")


def distributed_setup() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != FROZEN_WORLD_SIZE:
        raise SystemExit(
            "dense_full_model_sft_smoke requires exactly 8 ranks under FSDP "
            "FULL_SHARD; single-GPU and DDP replication are forbidden"
        )
    if local_rank < 0 or local_rank >= world_size or rank < 0 or rank >= world_size:
        raise SystemExit("invalid distributed rank environment")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("CUDA GPUs with native BF16 support are required")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size, torch.device("cuda", local_rank)


def checkpoint_all_mlp_modules(layers: torch.nn.ModuleList) -> list[str]:
    """Checkpoint stateless MLP/MoE blocks, never mutable attention/cache code."""

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )

    wrapped: list[str] = []
    for index, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            raise RuntimeError(f"transformer layer {index} has no MLP/MoE block")
        layer.mlp = checkpoint_wrapper(
            layer.mlp,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        wrapped.append(f"language_model.layers.{index}.mlp")
    return wrapped


def all_rank_trainable_count(
    module: torch.nn.Module, device: torch.device
) -> tuple[int, list[int]]:
    local = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    tensor = torch.tensor([local], dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    values = [int(item.item()) for item in gathered]
    return sum(values), values


def training_storage_ledger(
    *, trainable_parameters: int, parameter_bytes: int, world_size: int
) -> dict[str, Any]:
    gradient_bytes = parameter_bytes
    bf16_moments = trainable_parameters * 2 * 2
    fp32_moments = trainable_parameters * 2 * 4

    def shard(value: int) -> int:
        return math.ceil(value / world_size)

    global_values = {
        "parameter_bytes": parameter_bytes,
        "gradient_bytes": gradient_bytes,
        "adam_two_moments_if_bf16_bytes": bf16_moments,
        "adam_two_moments_if_fp32_bytes": fp32_moments,
    }
    return {
        "global": global_values,
        "ideal_even_shard_per_rank": {
            key: shard(value) for key, value in global_values.items()
        },
        "excludes": [
            "FSDP padding and metadata",
            "activations and temporary all-gathers",
            "allocator fragmentation",
        ],
    }


def observed_optimizer_state(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    dtype_elements: dict[str, int] = {}
    total_bytes = 0
    for state in optimizer.state.values():
        for value in state.values():
            if not isinstance(value, torch.Tensor):
                continue
            key = str(value.dtype)
            dtype_elements[key] = dtype_elements.get(key, 0) + value.numel()
            total_bytes += value.numel() * value.element_size()
    return {"dtype_elements": dtype_elements, "local_shard_bytes": total_bytes}


def _parameter_ids(module: torch.nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def require_finite_positive_grad_norm(value: torch.Tensor | float) -> float:
    scalar = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(scalar) or scalar <= 0.0:
        raise RuntimeError(
            "one-step supervised smoke produced no valid gradient: "
            f"global_grad_norm={scalar!r}"
        )
    return scalar


def validate_formal_smoke_dataset_counts(counts: dict[str, int]) -> None:
    if counts != FROZEN_SMOKE_DATASET_COUNTS:
        raise ValueError(
            "formal smoke manifest must contain exactly four train examples per "
            f"dataset: expected={FROZEN_SMOKE_DATASET_COUNTS}, actual={counts}"
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.training_scope == "qcomem_suffix_supervised_sft":
        gate = qcomem_suffix_supervised_sft_capability_gate()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(args.output_dir / "capability-gate.json", gate)
        print(json.dumps(gate, ensure_ascii=False, indent=2))
        raise SystemExit(
            "qcomem_suffix_supervised_sft is fail-closed until cached query and "
            "stepwise answer/decode semantics are validated"
        )
    validate_smoke_protocol(args, parser)

    local_rank, rank, world_size, device = distributed_setup()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    integrity_audit = validate_formal_integrity_ledgers(
        code_ledger_path=args.code_ledger,
        expected_code_ledger_sha256=args.expected_code_ledger_sha256,
        model_ledger_path=args.model_artifact_ledger,
        expected_model_ledger_sha256=args.expected_model_artifact_ledger_sha256,
    )
    manifest_audit = validate_prepared_training_manifest(
        args.manifest,
        args.data,
        expected_data_sha256=args.expected_data_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokenizer_audit = validate_runtime_tokenizer_against_manifest(
        tokenizer, args.manifest, model_path=args.model
    )
    dataset = SupervisedSFTDataset(
        args.data,
        tokenizer,
        max_sequence_tokens=args.max_sequence_tokens,
        limit=args.dataset_limit,
    )
    if len(dataset) != world_size:
        raise SystemExit(
            f"smoke dataset must expose exactly {world_size} examples, got {len(dataset)}"
        )
    try:
        validate_formal_smoke_dataset_counts(
            manifest_audit["dataset_written_examples"]
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if dataset.audit["dataset_counts_all_records"] != manifest_audit[
        "dataset_written_examples"
    ]:
        raise SystemExit(
            "JSONL dataset counts do not match converter manifest: "
            f"jsonl={dataset.audit['dataset_counts_all_records']}, "
            f"manifest={manifest_audit['dataset_written_examples']}"
        )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=single_example_collate,
        num_workers=0,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    split = TorchSplitCausalLM(model)
    if hasattr(split.config, "use_cache"):
        split.config.use_cache = False
    if len(split.layers) != args.expected_num_layers:
        raise SystemExit(
            f"text layer count changed: expected {args.expected_num_layers}, "
            f"got {len(split.layers)}"
        )
    model.train().to(device)
    torch.cuda.synchronize(device)
    full_replica_load_seconds = time.perf_counter() - load_started
    pre_fsdp_full_replica_allocated_bytes = torch.cuda.memory_allocated(device)
    pre_fsdp_full_replica_peak_allocated_bytes = torch.cuda.max_memory_allocated(
        device
    )
    pre_fsdp_full_replica_peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

    parameter_plan = configure_dense_full_model_trainability(model)
    actual_count = int(parameter_plan["trainable_parameters"])
    if actual_count != args.expected_trainable_params:
        raise SystemExit(
            "frozen text-model parameter count changed: "
            f"expected {args.expected_trainable_params:,}, got {actual_count:,}"
        )
    if parameter_plan["parameter_dtype_counts"] != {
        "torch.bfloat16": args.expected_trainable_params
    }:
        raise SystemExit(
            "every full-model parameter must be BF16 before FSDP: "
            f"{parameter_plan['parameter_dtype_counts']}"
        )
    parameter_plan["exact_expected_trainable_parameters"] = (
        args.expected_trainable_params
    )
    parameter_plan["exact_parameter_count_gate_passed"] = True

    original_ids = _parameter_ids(model)
    core = DenseSupervisedCausalLM.from_conditional_generation(model)
    if _parameter_ids(core) != original_ids:
        raise RuntimeError(
            "text-only supervised core does not own exactly every post-visual model parameter"
        )
    layer_classes = {type(layer) for layer in split.layers}
    activation_checkpoint_modules = checkpoint_all_mlp_modules(split.layers)
    if _parameter_ids(core) != original_ids:
        raise RuntimeError("MLP checkpoint wrapping changed the parameter identity set")
    del model, split

    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
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
    del core
    global_sharded_count, per_rank_counts = all_rank_trainable_count(fsdp, device)
    if global_sharded_count != args.expected_trainable_params:
        raise RuntimeError(
            f"FSDP shards sum to {global_sharded_count:,}, expected "
            f"{args.expected_trainable_params:,}"
        )
    post_fsdp_wrap_allocated_bytes = torch.cuda.memory_allocated(device)

    trainable = [
        parameter for parameter in fsdp.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    data_sha256 = manifest_audit["output_jsonl_sha256"]
    storage = training_storage_ledger(
        trainable_parameters=args.expected_trainable_params,
        parameter_bytes=int(parameter_plan["trainable_logical_bytes"]),
        world_size=world_size,
    )
    metadata: dict[str, Any] = {
        "created_unix": time.time(),
        "training_scope": "dense_full_model_sft_smoke",
        "objective_family": "supervised_token_cross_entropy",
        "model": str(args.model),
        "data": str(args.data),
        "data_sha256": data_sha256,
        "converter_manifest": manifest_audit,
        "runtime_tokenizer": tokenizer_audit,
        "integrity": integrity_audit,
        "data_audit": dataset.audit,
        "test_or_validation_used": False,
        "smoke_only": True,
        "long_run_allowed": False,
        "parameter_plan": parameter_plan,
        "storage_estimate": storage,
        "distributed": {
            "kind": "FSDP1_FULL_SHARD",
            "world_size": world_size,
            "ddp_used": False,
            "ddp_forbidden": True,
            "single_gpu_forbidden": True,
            "use_orig_params": True,
            "mixed_precision": {
                "parameter": "bfloat16",
                "reduce": "bfloat16",
                "buffer": "bfloat16",
                "selected_target_logits_for_ce": "float32",
            },
            "per_rank_trainable_shard_numel": per_rank_counts,
            "global_sharded_trainable_numel": global_sharded_count,
            "transient_loader_full_bf16_replica_per_rank": True,
            "full_replica_load_seconds": full_replica_load_seconds,
            "pre_fsdp_full_replica_allocated_bytes": (
                pre_fsdp_full_replica_allocated_bytes
            ),
            "pre_fsdp_full_replica_peak_allocated_bytes": (
                pre_fsdp_full_replica_peak_allocated_bytes
            ),
            "pre_fsdp_full_replica_peak_reserved_bytes": (
                pre_fsdp_full_replica_peak_reserved_bytes
            ),
            "post_fsdp_wrap_allocated_bytes": post_fsdp_wrap_allocated_bytes,
        },
        "activation_checkpoint": {
            "enabled": True,
            "implementation": "NO_REENTRANT checkpoint_wrapper",
            "scope": "all text transformer MLP/MoE blocks only",
            "modules": activation_checkpoint_modules,
            "module_count": len(activation_checkpoint_modules),
            "attention_checkpointed": False,
            "reason": (
                "MLP/MoE blocks are stateless; mutable attention/recurrent cache "
                "side effects are excluded from checkpoint recomputation"
            ),
        },
        "loss": {
            "kind": "answer_and_eos_only_causal_cross_entropy",
            "teacher": None,
            "distillation": False,
            "lora": False,
            "qlora": False,
            "prompt_label": -100,
            "causal_shift": "labels[t] are predicted from hidden[t-1]",
            "target_projection": (
                "lm_head runs only on predecessor positions for answer/EOS targets"
            ),
        },
        "sequence": {
            "layout": "document_prompt + query_prompt + selected_answer + EOS",
            "prompt_builder": "run_downstream.prompt_parts",
            "max_sequence_tokens": args.max_sequence_tokens,
            "truncation": "context head+tail only; selected_answer and EOS preserved",
        },
        "training": {
            "steps": 1,
            "gradient_accumulation": 1,
            "examples_global": world_size,
            "examples_per_rank": 1,
            "frozen_dataset_examples": FROZEN_SMOKE_DATASET_COUNTS,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
        },
        "checkpoint": {
            "mode": "metadata-only",
            "model_or_optimizer_artifact_written": False,
            "full_rank0_gather_forbidden": True,
        },
        "qcomem_suffix_supervised_sft_capability": (
            qcomem_suffix_supervised_sft_capability_gate()
        ),
    }
    if rank == 0:
        atomic_json(args.output_dir / "metadata.json", metadata)
        print(
            json.dumps(
                {
                    "phase": "dense_full_model_sft_preflight",
                    "exact_trainable_parameters": args.expected_trainable_params,
                    "data_audit": dataset.audit,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    sampler.set_epoch(0)
    batch = next(iter(loader))
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    target_tokens = int(batch["target_tokens"])
    loss = fsdp(input_ids, labels, attention_mask)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite supervised CE: {float(loss.detach().item())}")
    loss.backward()
    grad_norm = fsdp.clip_grad_norm_(args.max_grad_norm)
    grad_norm_value = require_finite_positive_grad_norm(grad_norm)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    metric_tensor = torch.tensor(
        [float(loss.detach().item()), float(target_tokens)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(metric_tensor)
    mean_rank_ce = float(metric_tensor[0].item() / world_size)
    global_target_tokens = int(metric_tensor[1].item())
    local_row = {
        "rank": rank,
        "dataset": batch["dataset"],
        "source_id": batch["source_id"],
        "sequence_tokens": int(batch["sequence_tokens"]),
        "prompt_tokens": int(batch["prompt_tokens"]),
        "target_tokens": target_tokens,
        "context_was_truncated": bool(batch["context_was_truncated"]),
        "loss": float(loss.detach().item()),
    }
    rows: list[Any] = [None] * world_size
    dist.all_gather_object(rows, local_row)
    peak = {
        "rank": rank,
        "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    peaks: list[Any] = [None] * world_size
    dist.all_gather_object(peaks, peak)
    optimizer_observed = observed_optimizer_state(optimizer)
    optimizer_by_rank: list[Any] = [None] * world_size
    dist.all_gather_object(optimizer_by_rank, optimizer_observed)

    if rank == 0:
        metric = {
            "step": 1,
            "mean_rank_answer_eos_ce": mean_rank_ce,
            "global_target_tokens": global_target_tokens,
            "grad_norm": grad_norm_value,
            "grad_norm_gate_passed": True,
            "learning_rate": args.learning_rate,
            "rank_examples": rows,
        }
        append_jsonl(args.output_dir / "metrics.jsonl", metric)
        metadata["last_step"] = 1
        metadata["finished_unix"] = time.time()
        metadata["runtime_cuda_peaks_after_fsdp_wrap"] = peaks
        metadata["observed_optimizer_state_by_rank"] = optimizer_by_rank
        metadata["one_step_metric"] = metric
        atomic_json(args.output_dir / "metadata.json", metadata)
        print(json.dumps(metric, ensure_ascii=False), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
