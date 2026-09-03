from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import math
import os
import random
import re
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from fp32_master import (
    aggregate_rank_audits,
    audit_adamw_fp32_state,
    audit_fp32_gradients,
    audit_fp32_parameter_delta,
    require_full_gradient_gate,
    require_parameter_delta_gate,
    snapshot_fp32_local_shards,
)
from qcomem_torch import TorchSplitCausalLM
from sft_quality_validation import (
    PreparedQualityDataset,
    evaluate_answer_eos_ce_distributed,
    paired_quality_comparison,
    validate_runtime_tokenizer_metadata,
)
from sft_dcp_checkpoint import save_eval_model_only_fp32
from supervised_sft import (
    FORMAL_MODEL_LEDGER_FILENAMES,
    DenseSupervisedCausalLM,
    configure_dense_full_model_trainability,
    validate_sha256_ledger,
)
from supervised_sft_longrun import (
    FORMAL_FORMAT,
    PreparedScaleDataset,
    balanced_global_indices,
    cosine_warmup_factor,
    global_token_weighted_rank_scale,
    schedule_audit,
    summarize_loss_rows,
    validate_scale_split_manifest,
)


FROZEN_WORLD_SIZE = 8
FROZEN_TEXT_PARAMETER_COUNT = 34_660_610_688
FROZEN_TEXT_LAYER_COUNT = 40
FROZEN_TRAIN_COUNTS = {"qasper": 512, "2wikimqa": 512}
FROZEN_HELDOUT_COUNTS = {"qasper": 64, "2wikimqa": 64}
FORMAL_CODE_FILENAMES = frozenset(
    {
        "train_supervised_sft_longrun.py",
        "supervised_sft_longrun.py",
        "sft_dcp_checkpoint.py",
        "sft_quality_validation.py",
        "test_supervised_sft_longrun.py",
        "test_sft_quality_validation.py",
        "fsdp_dcp_longrun_preflight.py",
        "launch_supervised_sft_longrun_8gpu.sh",
        "dense_full_model_sft_formal_384.json",
        "supervised_sft.py",
        "fp32_master.py",
        "qcomem_torch.py",
        "run_downstream.py",
        "split_supervised_sft_scale.py",
        "audit_supervised_sft_scale.py",
    }
)
FORMAL_MODEL_WEIGHT_FILENAMES = frozenset(
    f"model.safetensors-{index:05d}-of-00014.safetensors"
    for index in range(1, 15)
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name, delete=False
    ) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def build_parser() -> argparse.ArgumentParser:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args()
    defaults = json.loads(known.config.read_text()) if known.config is not None else {}
    parser = argparse.ArgumentParser(
        description="384-step balanced dense full-model supervised SFT formal run"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path, required="model" not in defaults)
    parser.add_argument("--train-data", type=Path, required="train_data" not in defaults)
    parser.add_argument(
        "--heldout-data", type=Path, required="heldout_data" not in defaults
    )
    parser.add_argument(
        "--split-manifest", type=Path, required="split_manifest" not in defaults
    )
    parser.add_argument(
        "--expected-train-sha256", required="expected_train_sha256" not in defaults
    )
    parser.add_argument(
        "--expected-heldout-sha256", required="expected_heldout_sha256" not in defaults
    )
    parser.add_argument(
        "--expected-split-manifest-sha256",
        required="expected_split_manifest_sha256" not in defaults,
    )
    parser.add_argument("--output-dir", type=Path, required="output_dir" not in defaults)
    parser.add_argument("--code-ledger", type=Path, required=True)
    parser.add_argument("--expected-code-ledger-sha256", required=True)
    parser.add_argument("--model-artifact-ledger", type=Path, required=True)
    parser.add_argument("--expected-model-artifact-ledger-sha256", required=True)
    parser.add_argument("--model-weight-ledger", type=Path, required=True)
    parser.add_argument("--expected-model-weight-ledger-sha256", required=True)
    parser.add_argument("--steps", type=int, default=384)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument(
        "--evaluation-steps", type=int, nargs="+", default=[0, 128, 256, 384]
    )
    parser.add_argument("--delta-gate-steps", type=int, nargs="+", default=[1])
    parser.add_argument(
        "--model-only-checkpoint-steps",
        type=int,
        nargs="+",
        default=[128, 256, 384],
    )
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--expected-world-size", type=int, default=FROZEN_WORLD_SIZE)
    parser.add_argument(
        "--expected-trainable-params", type=int, default=FROZEN_TEXT_PARAMETER_COUNT
    )
    parser.add_argument(
        "--expected-num-layers", type=int, default=FROZEN_TEXT_LAYER_COUNT
    )
    known_destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - known_destinations)
    if unknown:
        parser.error(f"unknown config keys: {', '.join(unknown)}")
    parser.set_defaults(**defaults)
    return parser


def validate_formal_protocol(
    args: argparse.Namespace, parser: argparse.ArgumentParser
) -> None:
    frozen = {
        "steps": 384,
        "gradient_accumulation": 1,
        "learning_rate": 1e-6,
        "warmup_steps": 20,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "max_sequence_tokens": 1024,
        "expected_world_size": FROZEN_WORLD_SIZE,
        "expected_trainable_params": FROZEN_TEXT_PARAMETER_COUNT,
        "expected_num_layers": FROZEN_TEXT_LAYER_COUNT,
        "checkpoint_every": 0,
    }
    for key, expected in frozen.items():
        if getattr(args, key) != expected:
            parser.error(
                f"384-step formal run freezes {key}={expected!r}, got {getattr(args, key)!r}"
            )
    if sorted(set(args.evaluation_steps)) != [0, 128, 256, 384]:
        parser.error("formal run freezes heldout evaluations at steps 0/128/256/384")
    if sorted(set(args.delta_gate_steps)) != [1]:
        parser.error("formal run freezes the initial-to-step-1 FP32 delta gate")
    if sorted(set(args.model_only_checkpoint_steps)) != [128, 256, 384]:
        parser.error("formal run must publish FP32 model-only DCPs at 128/256/384")
    if not set(args.model_only_checkpoint_steps) <= set(args.evaluation_steps):
        parser.error("every model-only checkpoint step must also be a heldout eval step")


def distributed_setup() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != FROZEN_WORLD_SIZE:
        raise SystemExit("dense SFT formal run requires exactly 8 FSDP ranks")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("dense SFT formal run requires CUDA GPUs with native BF16")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    return local_rank, rank, world_size, torch.device("cuda", local_rank)


def checkpoint_all_mlp_modules(layers: torch.nn.ModuleList) -> list[str]:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        checkpoint_wrapper,
    )

    wrapped = []
    for index, layer in enumerate(layers):
        if not hasattr(layer, "mlp"):
            raise RuntimeError(f"text layer {index} has no MLP/MoE block")
        layer.mlp = checkpoint_wrapper(
            layer.mlp, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        )
        wrapped.append(f"language_model.layers.{index}.mlp")
    return wrapped


def _parameter_ids(module: torch.nn.Module) -> set[int]:
    return {id(parameter) for parameter in module.parameters()}


def all_rank_trainable_count(
    module: torch.nn.Module, device: torch.device
) -> tuple[int, list[int]]:
    local = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    value = torch.tensor(local, dtype=torch.int64, device=device)
    gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value)
    counts = [int(item.item()) for item in gathered]
    return sum(counts), counts


def gather_objects(value: Any) -> list[Any]:
    result: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def require_positive_finite(value: torch.Tensor | float, label: str) -> float:
    scalar = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
    if not math.isfinite(scalar) or scalar <= 0:
        raise RuntimeError(f"{label} must be finite and positive, got {scalar!r}")
    return scalar


def _no_parameter_gradients(module: torch.nn.Module) -> bool:
    return all(parameter.grad is None for parameter in module.parameters())


def _runtime_code_audit() -> dict[str, str]:
    names = (
        "train_supervised_sft_longrun.py",
        "supervised_sft_longrun.py",
        "sft_quality_validation.py",
        "supervised_sft.py",
        "fp32_master.py",
        "qcomem_torch.py",
        "run_downstream.py",
        "split_supervised_sft_scale.py",
    )
    root = Path(__file__).parent
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names
    }


def validate_launcher_verified_weight_ledger(
    path: Path, *, expected_sha256: str
) -> dict[str, Any]:
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError("model-weight ledger SHA256 mismatch")
    filenames = set()
    entries = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise RuntimeError(f"invalid model-weight ledger line {line_number}")
        digest, raw_path = match.groups()
        artifact = Path(raw_path)
        if not artifact.is_absolute() or not artifact.is_file():
            raise RuntimeError(f"model-weight artifact is missing: {artifact}")
        if artifact.name in filenames:
            raise RuntimeError("model-weight ledger repeats a filename")
        filenames.add(artifact.name)
        entries.append(
            {"filename": artifact.name, "path": str(artifact), "sha256": digest}
        )
    if filenames != set(FORMAL_MODEL_WEIGHT_FILENAMES):
        raise RuntimeError("model-weight ledger does not contain the exact 14 shards")
    return {
        "ledger_path": str(path),
        "ledger_sha256": actual_sha256,
        "entries": entries,
        "all_artifacts_exist": True,
        "contents_verified_by_launcher_before_torchrun": True,
        "trainer_rehashed_129gib_weights": False,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_formal_protocol(args, parser)
    local_rank, rank, world_size, device = distributed_setup()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    # The launcher owns RUN_DIR/logs/stages.  Trainer artifacts live in a
    # dedicated directory that must be absent, preventing stale metrics or
    # checkpoints from being mistaken for this formal run.
    output_error: list[Any] = [None]
    if rank == 0:
        try:
            if args.output_dir.exists():
                raise RuntimeError(
                    f"formal trainer output directory must not exist: {args.output_dir}"
                )
            args.output_dir.mkdir(parents=True, exist_ok=False)
        except Exception as error:
            output_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(output_error, src=0)
    if output_error[0] is not None:
        raise SystemExit(output_error[0])
    dist.barrier()

    split_audit = validate_scale_split_manifest(
        args.split_manifest,
        expected_manifest_sha256=args.expected_split_manifest_sha256,
        train_path=args.train_data,
        expected_train_sha256=args.expected_train_sha256,
        heldout_path=args.heldout_data,
        expected_heldout_sha256=args.expected_heldout_sha256,
    )
    integrity_audit = {
        "code": validate_sha256_ledger(
            args.code_ledger,
            expected_ledger_sha256=args.expected_code_ledger_sha256,
            required_filenames=FORMAL_CODE_FILENAMES,
            ledger_name="formal_longrun_code",
        ),
        "model_artifacts": validate_sha256_ledger(
            args.model_artifact_ledger,
            expected_ledger_sha256=args.expected_model_artifact_ledger_sha256,
            required_filenames=FORMAL_MODEL_LEDGER_FILENAMES,
            ledger_name="formal_longrun_model_artifacts",
        ),
        "model_weights": validate_launcher_verified_weight_ledger(
            args.model_weight_ledger,
            expected_sha256=args.expected_model_weight_ledger_sha256,
        ),
    }
    if split_audit["counts"]["train_jsonl"] != FROZEN_TRAIN_COUNTS:
        raise SystemExit(
            f"formal train counts drifted: {split_audit['counts']['train_jsonl']}"
        )
    if split_audit["counts"]["heldout_ce_jsonl"] != FROZEN_HELDOUT_COUNTS:
        raise SystemExit(
            f"formal heldout counts drifted: {split_audit['counts']['heldout_ce_jsonl']}"
        )

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    manifest_payload = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    tokenizer_audit = validate_runtime_tokenizer_metadata(tokenizer, manifest_payload)
    train_dataset = PreparedScaleDataset(
        args.train_data,
        eos_token_id=int(tokenizer.eos_token_id),
        max_sequence_tokens=args.max_sequence_tokens,
        expected_counts=FROZEN_TRAIN_COUNTS,
    )
    heldout_dataset = PreparedQualityDataset(
        args.heldout_data,
        args.split_manifest,
        expected_heldout_sha256=args.expected_heldout_sha256,
        expected_split_manifest_sha256=args.expected_split_manifest_sha256,
        expected_train_sha256=args.expected_train_sha256,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    split = TorchSplitCausalLM(model)
    if hasattr(split.config, "use_cache"):
        split.config.use_cache = False
    if len(split.layers) != args.expected_num_layers:
        raise RuntimeError("frozen text layer count drifted")
    model.train().to(device)
    torch.cuda.synchronize(device)
    full_replica_load_seconds = time.perf_counter() - load_started
    full_replica_peak_bytes = torch.cuda.max_memory_allocated(device)
    plan = configure_dense_full_model_trainability(model)
    plan["training_scope"] = "dense_full_model_sft_formal_384"
    if plan["trainable_parameters"] != args.expected_trainable_params:
        raise RuntimeError("frozen text parameter count drifted")
    if plan["parameter_dtype_counts"] != {
        "torch.bfloat16": args.expected_trainable_params
    }:
        raise RuntimeError("pre-FSDP full-model parameters must all be BF16")
    original_ids = _parameter_ids(model)
    core = DenseSupervisedCausalLM.from_conditional_generation(model)
    layer_classes = {type(layer) for layer in split.layers}
    activation_checkpoint_modules = checkpoint_all_mlp_modules(split.layers)
    if _parameter_ids(core) != original_ids:
        raise RuntimeError("text core/checkpoint wrapping changed parameter identities")
    del model, split

    from torch.distributed.fsdp import (
        BackwardPrefetch,
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    fsdp = FSDP(
        core,
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls=layer_classes
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            keep_low_precision_grads=False,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        limit_all_gathers=True,
        device_id=device,
    )
    del core
    fsdp.float()
    global_parameters, per_rank_parameters = all_rank_trainable_count(fsdp, device)
    if global_parameters != args.expected_trainable_params:
        raise RuntimeError("FSDP shard parameter count drifted")
    local_fp32 = sum(parameter.numel() for parameter in fsdp.parameters())
    if (
        local_fp32 != per_rank_parameters[rank]
        or {parameter.dtype for parameter in fsdp.parameters()} != {torch.float32}
    ):
        raise RuntimeError("persistent FSDP optimizer shards must all be FP32")
    trainable = [parameter for parameter in fsdp.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step_index: cosine_warmup_factor(
            step_index,
            warmup_steps=args.warmup_steps,
            total_steps=args.steps,
        ),
    )

    metadata: dict[str, Any] = {
        "format": FORMAL_FORMAT,
        "created_unix": time.time(),
        "training_scope": "dense_full_model_sft_formal_384",
        "objective": "answer_and_eos_only_supervised_causal_cross_entropy",
        "model": str(args.model),
        "runtime_code_sha256": _runtime_code_audit(),
        "integrity": integrity_audit,
        "split": split_audit,
        "tokenizer": tokenizer_audit,
        "train_data": train_dataset.audit,
        "heldout_data": heldout_dataset.audit,
        "test_or_validation_used": False,
        "raw_test_v2_read": False,
        "parameter_plan": plan,
        "distributed": {
            "kind": "FSDP1_FULL_SHARD",
            "world_size": world_size,
            "use_orig_params": True,
            "persistent_parameter_shard_dtype": "float32",
            "forward_parameter_dtype": "bfloat16",
            "gradient_reduce_dtype": "float32",
            "per_rank_parameter_numel": per_rank_parameters,
            "global_parameter_numel": global_parameters,
            "process_group_timeout_seconds": 7200,
            "transient_full_bf16_replica_per_rank": True,
            "full_replica_load_seconds": full_replica_load_seconds,
            "full_replica_peak_allocated_bytes": full_replica_peak_bytes,
        },
        "activation_checkpoint": {
            "implementation": "NO_REENTRANT",
            "scope": "MLP/MoE only; attention/cache excluded",
            "modules": activation_checkpoint_modules,
        },
        "training": {
            "steps": args.steps,
            "gradient_accumulation": args.gradient_accumulation,
            "global_examples_per_micro_batch": world_size,
            "dataset_balance_per_global_micro_batch": {
                "qasper": 4,
                "2wikimqa": 4,
            },
            "schedule": "deterministic_per-dataset_shuffle_then_frozen-rank-shuffle-v1",
            "seed": args.seed,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "lr_schedule": "linear_warmup_then_cosine_to_zero",
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "loss_weighting": "global-target-token-weighted",
            "evaluation_steps": sorted(set(args.evaluation_steps)),
            "delta_gate_steps": sorted(set(args.delta_gate_steps)),
        },
        "checkpoint": {
            "mode": "DCP_sharded_eval_model_only_fp32",
            "planned_steps": sorted(set(args.model_only_checkpoint_steps)),
            "observed_completed_steps": [],
            "model_write_planned": True,
            "model_write_observed": False,
            "optimizer_write_planned": False,
            "optimizer_write_observed": False,
            "scheduler_or_rng_write_planned": False,
            "scheduler_or_rng_write_observed": False,
            "full_resume_checkpoint_write_planned": False,
            "full_resume_checkpoint_write_observed": False,
            "contract": "eval_model_only_fp32",
            "rank0_full_gather_used": False,
        },
        "heldout_protocol": {
            "steps": [0, 128, 256, 384],
            "torch_no_grad": True,
            "enters_backward": False,
            "metrics": ["token_weighted_ce", "mean_example_ce"],
            "by_dataset": True,
            "automatic_early_stop": False,
            "final_downstream_evaluation": False,
        },
    }
    if rank == 0:
        atomic_json(args.output_dir / "metadata.json", metadata)
        print(
            json.dumps(
                {
                    "phase": "dense_sft_formal_preflight",
                    "train_sha256": args.expected_train_sha256,
                    "heldout_sha256": args.expected_heldout_sha256,
                    "split_manifest_sha256": args.expected_split_manifest_sha256,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    evaluation_results: dict[int, dict[str, Any]] = {}
    checkpoint_records: dict[int, dict[str, Any]] = {}

    def run_heldout(step: int) -> None:
        if not _no_parameter_gradients(fsdp):
            raise RuntimeError("heldout CE may run only after all training gradients are cleared")
        result = evaluate_answer_eos_ce_distributed(
            fsdp, heldout_dataset, device, phase=f"step-{step:06d}"
        )
        if not _no_parameter_gradients(fsdp):
            raise RuntimeError("heldout no_grad evaluation created parameter gradients")
        evaluation_results[step] = result
        if rank == 0:
            atomic_json(args.output_dir / f"heldout-step-{step:06d}.json", result)
            append_jsonl(
                args.output_dir / "heldout-metrics.jsonl",
                {"step": step, "summary": result["summary"]},
            )
            print(
                json.dumps(
                    {"phase": "heldout_ce", "step": step, **result["summary"]},
                    sort_keys=True,
                ),
                flush=True,
            )

    run_heldout(0)
    cumulative_counts: Counter[str] = Counter()
    gate_records: dict[str, Any] = {}
    optimizer.zero_grad(set_to_none=True)
    for step in range(1, args.steps + 1):
        gate_step = step in set(args.delta_gate_steps)
        snapshots = (
            snapshot_fp32_local_shards(fsdp.named_parameters()) if gate_step else None
        )
        micro_schedules = [
            balanced_global_indices(
                train_dataset.indices_by_dataset,
                micro_batch_index=(step - 1) * args.gradient_accumulation + micro_step,
                seed=args.seed,
            )
            for micro_step in range(args.gradient_accumulation)
        ]
        for indices in micro_schedules:
            audit = schedule_audit(train_dataset.examples, indices)
            if audit["dataset_counts"] != {"2wikimqa": 4, "qasper": 4}:
                raise RuntimeError("global train micro-batch lost its frozen 4+4 balance")
            cumulative_counts.update(audit["dataset_counts"])
        global_step_targets = sum(
            train_dataset[index].target_tokens
            for indices in micro_schedules
            for index in indices
        )
        optimizer.zero_grad(set_to_none=True)
        local_rows = []
        for micro_step, indices in enumerate(micro_schedules):
            example = train_dataset[indices[rank]]
            input_ids = example.input_ids.unsqueeze(0).to(device, non_blocking=True)
            labels = example.labels.unsqueeze(0).to(device, non_blocking=True)
            attention_mask = torch.ones_like(input_ids)
            local_mean_ce = fsdp(input_ids, labels, attention_mask)
            if not torch.isfinite(local_mean_ce):
                raise RuntimeError(f"non-finite train CE at step {step}")
            scale = global_token_weighted_rank_scale(
                local_target_tokens=example.target_tokens,
                global_step_target_tokens=global_step_targets,
                world_size=world_size,
            )
            (local_mean_ce * scale).backward()
            local_rows.append(
                {
                    "rank": rank,
                    "micro_step": micro_step,
                    "dataset": example.dataset,
                    "source_id_sha256": example.source_id_sha256,
                    "sequence_tokens": example.sequence_tokens,
                    "target_tokens": example.target_tokens,
                    "mean_ce": float(local_mean_ce.detach().float().item()),
                    "backward_scale": scale,
                }
            )
        observed_local_targets = torch.tensor(
            sum(row["target_tokens"] for row in local_rows),
            dtype=torch.int64,
            device=device,
        )
        dist.all_reduce(observed_local_targets)
        if int(observed_local_targets.item()) != global_step_targets:
            raise RuntimeError("scheduled and observed global target-token totals differ")
        grad_norm = require_positive_finite(
            fsdp.clip_grad_norm_(args.max_grad_norm), "global gradient norm"
        )
        gate_record: dict[str, Any] | None = None
        if snapshots is not None:
            gradient = aggregate_rank_audits(
                gather_objects(audit_fp32_gradients(snapshots))
            )
            require_full_gradient_gate(
                gradient,
                expected_parameters=args.expected_trainable_params,
                expected_layers=args.expected_num_layers,
            )
            gate_record = {"gradient": gradient}
        lr_used = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if snapshots is not None and gate_record is not None:
            optimizer_state_by_rank = gather_objects(
                audit_adamw_fp32_state(optimizer, snapshots, expected_step=step)
            )
            if sum(row["parameter_elements"] for row in optimizer_state_by_rank) != (
                args.expected_trainable_params
            ):
                raise RuntimeError("FP32 Adam state does not cover the full model")
            local_delta = audit_fp32_parameter_delta(snapshots)
            delta = {
                precision: aggregate_rank_audits(
                    gather_objects(local_delta[precision])
                )
                for precision in ("fp32_logical", "bf16_forward_visible")
            }
            require_parameter_delta_gate(
                delta,
                expected_parameters=args.expected_trainable_params,
                expected_layers=args.expected_num_layers,
            )
            gate_record["optimizer_state_by_rank"] = optimizer_state_by_rank
            gate_record["parameter_delta"] = delta
            gate_records[str(step)] = gate_record
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        del snapshots

        gathered_rows = gather_objects(local_rows)
        rows = [row for rank_rows in gathered_rows for row in rank_rows]
        summary = summarize_loss_rows(rows)
        step_schedule = [
            schedule_audit(train_dataset.examples, indices) for indices in micro_schedules
        ]
        metric = {
            "step": step,
            "train": summary,
            "global_target_tokens": global_step_targets,
            "global_grad_norm_before_clip": grad_norm,
            "learning_rate_used": lr_used,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "schedule": step_schedule,
            "cumulative_dataset_counts": dict(sorted(cumulative_counts.items())),
            "delta_gate_passed": gate_step,
        }
        if rank == 0:
            append_jsonl(args.output_dir / "train-metrics.jsonl", metric)
            print(json.dumps(metric, sort_keys=True), flush=True)
        if step in set(args.evaluation_steps):
            run_heldout(step)

        if step in set(args.model_only_checkpoint_steps):
            # Heldout runs first at these frozen epoch boundaries, so checkpoint
            # provenance binds the exact diagnostic used for later best selection.
            summary = evaluation_results[step]["summary"]
            checkpoint = save_eval_model_only_fp32(
                fsdp,
                args.output_dir / "checkpoints" / f"step-{step:06d}-eval-model-only-fp32",
                step=step,
                expected_parameters=args.expected_trainable_params,
                provenance={
                    "formal_format": FORMAL_FORMAT,
                    "model": str(args.model),
                    "train_jsonl_sha256": args.expected_train_sha256,
                    "heldout_ce_jsonl_sha256": args.expected_heldout_sha256,
                    "split_manifest_sha256": args.expected_split_manifest_sha256,
                    "code_ledger_sha256": args.expected_code_ledger_sha256,
                    "model_artifact_ledger_sha256": (
                        args.expected_model_artifact_ledger_sha256
                    ),
                    "model_weight_ledger_sha256": args.expected_model_weight_ledger_sha256,
                    "heldout_overall_token_weighted_ce": summary["overall"][
                        "token_weighted_ce"
                    ],
                    "heldout_summary": summary,
                    "heldout_is_train_split_diagnostic_only": True,
                    "raw_test_v2_read": False,
                },
            )
            checkpoint_records[step] = checkpoint
            if rank == 0:
                append_jsonl(
                    args.output_dir / "checkpoint-metrics.jsonl",
                    {
                        "step": step,
                        "checkpoint_path": checkpoint["checkpoint_path"],
                        "checkpoint_manifest_sha256": checkpoint[
                            "checkpoint_manifest_sha256"
                        ],
                        "payload_directory_sha256": checkpoint[
                            "payload_directory_sha256"
                        ],
                        "logical_model_bytes": checkpoint["logical_model_bytes"],
                        "actual_payload_bytes": checkpoint["actual_payload_bytes"],
                        "heldout_overall_token_weighted_ce": summary["overall"][
                            "token_weighted_ce"
                        ],
                    },
                )

    final_comparison = paired_quality_comparison(
        evaluation_results[0]["rows"], evaluation_results[args.steps]["rows"]
    )
    if rank == 0:
        expected_checkpoint_steps = sorted(set(args.model_only_checkpoint_steps))
        if sorted(checkpoint_records) != expected_checkpoint_steps:
            raise RuntimeError(
                "not every frozen model-only checkpoint completed: "
                f"expected={expected_checkpoint_steps}, actual={sorted(checkpoint_records)}"
            )
        if not all(
            record.get("payload_integrity_verified_once_at_save") is True
            and record.get("atomic_publish_completed") is True
            and record.get("success_marker_written") is True
            for record in checkpoint_records.values()
        ):
            raise RuntimeError("best-checkpoint candidates include an incomplete DCP publish")
        candidate_ce = {
            step: float(
                evaluation_results[step]["summary"]["overall"]["token_weighted_ce"]
            )
            for step in expected_checkpoint_steps
        }
        if not all(math.isfinite(value) for value in candidate_ce.values()):
            raise RuntimeError(f"best-checkpoint candidates contain non-finite CE: {candidate_ce}")
        best_step = min(
            checkpoint_records,
            key=lambda candidate: (
                candidate_ce[candidate],
                candidate,
            ),
        )
        best_checkpoint = {
            "schema_version": "qcomem-sft-best-checkpoint-v1",
            "selection_metric": "heldout_ce.overall.token_weighted_ce",
            "selection_direction": "min",
            "tie_break": "earliest_step",
            "candidate_steps": sorted(checkpoint_records),
            "selected_step": best_step,
            "checkpoint_path": checkpoint_records[best_step]["checkpoint_path"],
            "checkpoint_manifest_sha256": checkpoint_records[best_step][
                "checkpoint_manifest_sha256"
            ],
            "payload_directory_sha256": checkpoint_records[best_step][
                "payload_directory_sha256"
            ],
            "heldout_overall_token_weighted_ce": evaluation_results[best_step][
                "summary"
            ]["overall"]["token_weighted_ce"],
            "diagnostic_only_not_final_downstream_quality": True,
            "raw_test_v2_read": False,
        }
        atomic_json(args.output_dir / "best-checkpoint.json", best_checkpoint)
        candidate_comparisons = {
            str(step): paired_quality_comparison(
                evaluation_results[0]["rows"], evaluation_results[step]["rows"]
            )
            for step in expected_checkpoint_steps
        }
        best_comparison = candidate_comparisons[str(best_step)]
        atomic_json(
            args.output_dir / "heldout-candidate-comparisons.json",
            {
                "schema_version": "qcomem-sft-heldout-candidate-comparisons-v1",
                "baseline_step": 0,
                "candidate_steps": expected_checkpoint_steps,
                "comparisons": candidate_comparisons,
                "selected_best_step": best_step,
            },
        )
        metadata["last_step"] = args.steps
        metadata["finished_unix"] = time.time()
        metadata["cumulative_dataset_counts"] = dict(sorted(cumulative_counts.items()))
        metadata["fp32_gradient_optimizer_delta_gates"] = gate_records
        metadata["heldout_evaluation_summaries"] = {
            str(step): result["summary"]
            for step, result in sorted(evaluation_results.items())
        }
        metadata["heldout_step0_to_candidate_comparisons"] = candidate_comparisons
        metadata["heldout_step0_to_final_comparison"] = final_comparison
        metadata["checkpoint_records"] = {
            str(step): {
                key: value
                for key, value in record.items()
                if key
                in {
                    "checkpoint_path",
                    "checkpoint_manifest_sha256",
                    "payload_directory_sha256",
                    "logical_model_bytes",
                    "actual_payload_bytes",
                }
            }
            for step, record in sorted(checkpoint_records.items())
        }
        metadata["best_checkpoint"] = best_checkpoint
        metadata["checkpoint"]["observed_completed_steps"] = sorted(checkpoint_records)
        metadata["checkpoint"]["model_write_observed"] = True
        checkpoint_gate = best_comparison["conditional_model_only_checkpoint_gate"]
        metadata["quality_gate"] = {
            **checkpoint_gate,
            "selected_checkpoint_step": best_step,
            "comparison": "step-0 baseline vs selected best checkpoint",
            "long_run_recommended": bool(checkpoint_gate["passed"]),
            "automatic_early_stop_used": False,
            "final_downstream_quality_tested": False,
        }
        metadata["runtime_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        metadata["runtime_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        atomic_json(args.output_dir / "heldout-comparison-final.json", final_comparison)
        atomic_json(args.output_dir / "metadata.json", metadata)
        print(
            json.dumps(
                {
                    "phase": "dense_sft_formal_complete",
                    "step": args.steps,
                    "quality_gate": metadata["quality_gate"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
