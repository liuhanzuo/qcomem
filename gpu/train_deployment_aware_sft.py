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
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from deployment_aware_sft import (
    HELDOUT_COUNTS,
    TRAIN_COUNTS,
    DeploymentAwareCausalLM,
    DeploymentDataset,
    frozen_teacher_targets,
    sha256_file,
    summarize_example_equal,
    validate_manifest,
)
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
from sft_dcp_checkpoint import save_eval_model_only_fp32
from supervised_sft import configure_dense_full_model_trainability, validate_sha256_ledger
from supervised_sft_longrun import cosine_warmup_factor


FORMAT = "qcomem_dense_long_instruction_preservation_full_sft_control_v1"
WORLD_SIZE = 8
TEXT_PARAMETERS = 34_660_610_688
TEXT_LAYERS = 40
STEPS = 128
EVALUATION_STEPS = (0, 64, 128)
CHECKPOINT_STEPS = (64, 128)
CODE_FILES = frozenset(
    {
        "train_deployment_aware_sft.py",
        "deployment_aware_sft.py",
        "build_deployment_aware_sft.py",
        "audit_deployment_aware_sft.py",
        "test_deployment_aware_sft.py",
        "deployment_aware_fsdp_preflight.py",
        "launch_deployment_aware_sft_8gpu.sh",
        "deployment_aware_sft_4k_128.json",
        "fp32_master.py",
        "qcomem_torch.py",
        "sft_dcp_checkpoint.py",
        "supervised_sft.py",
        "supervised_sft_longrun.py",
    }
)
MODEL_ARTIFACT_FILES = frozenset(
    {
        "config.json",
        "model.safetensors.index.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    }
)
MODEL_WEIGHT_FILES = frozenset(
    f"model.safetensors-{index:05d}-of-00014.safetensors" for index in range(1, 15)
)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name, delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parser() -> argparse.ArgumentParser:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args()
    defaults = json.loads(known.config.read_text()) if known.config else {}
    value = argparse.ArgumentParser(
        description="Dense 4K long-instruction/preservation full-SFT control"
    )
    value.add_argument("--config", type=Path)
    value.add_argument("--model", type=Path, required="model" not in defaults)
    value.add_argument("--train-data", type=Path, required="train_data" not in defaults)
    value.add_argument("--heldout-data", type=Path, required="heldout_data" not in defaults)
    value.add_argument("--data-manifest", type=Path, required="data_manifest" not in defaults)
    value.add_argument("--expected-train-sha256", required=True)
    value.add_argument("--expected-heldout-sha256", required=True)
    value.add_argument("--expected-data-manifest-sha256", required=True)
    value.add_argument("--code-ledger", type=Path, required=True)
    value.add_argument("--expected-code-ledger-sha256", required=True)
    value.add_argument("--model-artifact-ledger", type=Path, required=True)
    value.add_argument("--expected-model-artifact-ledger-sha256", required=True)
    value.add_argument("--model-weight-ledger", type=Path, required=True)
    value.add_argument("--expected-model-weight-ledger-sha256", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--steps", type=int, default=STEPS)
    value.add_argument("--learning-rate", type=float, default=5e-7)
    value.add_argument("--warmup-steps", type=int, default=8)
    value.add_argument("--weight-decay", type=float, default=0.0)
    value.add_argument("--max-grad-norm", type=float, default=1.0)
    value.add_argument("--seed", type=int, default=20260813)
    value.add_argument("--max-sequence-tokens", type=int, default=4096)
    value.add_argument("--teacher-topk", type=int, default=32)
    value.add_argument("--hard-weight", type=float, default=0.45)
    value.add_argument("--kl-weight", type=float, default=0.35)
    value.add_argument("--hidden-weight", type=float, default=0.20)
    value.add_argument("--teacher-projection-chunk-tokens", type=int, default=32)
    value.add_argument("--minimum-step1-headroom-bytes", type=int, default=536_870_912)
    unknown = sorted(set(defaults) - {action.dest for action in value._actions})
    if unknown:
        value.error(f"unknown config keys: {unknown}")
    value.set_defaults(**defaults)
    return value


def validate_protocol(args: argparse.Namespace, value: argparse.ArgumentParser) -> None:
    frozen = {
        "steps": STEPS,
        "learning_rate": 5e-7,
        "warmup_steps": 8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 20260813,
        "max_sequence_tokens": 4096,
        "teacher_topk": 32,
        "hard_weight": 0.45,
        "kl_weight": 0.35,
        "hidden_weight": 0.20,
        "teacher_projection_chunk_tokens": 32,
        "minimum_step1_headroom_bytes": 536_870_912,
    }
    for key, expected in frozen.items():
        if getattr(args, key) != expected:
            value.error(f"formal protocol freezes {key}={expected!r}")


def distributed_setup() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != WORLD_SIZE:
        raise SystemExit("formal training requires exactly eight ranks")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("formal training requires native-BF16 CUDA GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))
    return local_rank, rank, torch.device("cuda", local_rank)


def gather(value: Any) -> list[Any]:
    result: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(result, value)
    return result


def all_rank_parameter_count(module: torch.nn.Module, device: torch.device) -> tuple[int, list[int]]:
    local = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    tensor = torch.tensor(local, dtype=torch.int64, device=device)
    values = [torch.zeros_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(values, tensor)
    counts = [int(value.item()) for value in values]
    return sum(counts), counts


def validate_launcher_verified_weight_ledger(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError("model-weight ledger SHA256 mismatch")
    entries = []
    filenames = set()
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
        entries.append({"filename": artifact.name, "path": str(artifact), "sha256": digest})
    if filenames != set(MODEL_WEIGHT_FILES):
        raise RuntimeError("model-weight ledger does not contain exactly 14 shards")
    return {
        "ledger_path": str(path),
        "ledger_sha256": actual,
        "entries": entries,
        "all_artifacts_exist": True,
        "contents_verified_by_launcher_before_torchrun": True,
        "trainer_rehashed_large_weight_files": False,
    }


def _teacher_shard_path(root: Path, rank: int) -> Path:
    return root / f"teacher-targets-rank-{rank:02d}.pt"


def build_teacher_cache(
    model: torch.nn.Module,
    dataset: DeploymentDataset,
    device: torch.device,
    output_root: Path,
    *,
    rank: int,
    topk: int,
    projection_chunk_tokens: int,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    if rank == 0:
        output_root.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    split = TorchSplitCausalLM(model)
    model.eval()
    local_records: dict[str, dict[str, torch.Tensor]] = {}
    local_positions = []
    for schedule_index, example in enumerate(dataset.examples):
        if not example.teacher_required or schedule_index % WORLD_SIZE != rank:
            continue
        input_ids = example.input_ids.unsqueeze(0).to(device)
        labels = example.labels.unsqueeze(0).to(device)
        attention_mask = torch.ones_like(input_ids)
        targets = frozen_teacher_targets(
            split.language_model,
            split.lm_head,
            input_ids,
            labels,
            attention_mask,
            topk=topk,
            projection_chunk_tokens=projection_chunk_tokens,
        )
        expected = labels[:, 1:][labels[:, 1:] != -100].to(torch.int32).cpu()
        if not torch.equal(targets["target_ids"], expected):
            raise RuntimeError("teacher target token alignment failed")
        if not all(torch.isfinite(value).all() for key, value in targets.items() if key != "topk_ids" and key != "target_ids"):
            raise RuntimeError("teacher cache contains non-finite values")
        local_records[example.example_id] = targets
        local_positions.append(schedule_index)
        del input_ids, labels, attention_mask, targets
    shard_path = _teacher_shard_path(output_root, rank)
    temporary = shard_path.with_suffix(".pt.incomplete")
    torch.save(
        {
            "format": "qcomem-frozen-teacher-target-shard-v1",
            "rank": rank,
            "world_size": WORLD_SIZE,
            "topk": topk,
            "records": local_records,
            "schedule_indices": local_positions,
        },
        temporary,
    )
    os.replace(temporary, shard_path)
    local_summary = {
        "rank": rank,
        "basename": shard_path.name,
        "sha256": sha256_file(shard_path),
        "size_bytes": shard_path.stat().st_size,
        "records": len(local_records),
        "target_positions": sum(
            int(record["target_ids"].numel()) for record in local_records.values()
        ),
        "schedule_indices": local_positions,
    }
    summaries = gather(local_summary)
    manifest_error: list[Any] = [None]
    if rank == 0:
        try:
            if sum(item["records"] for item in summaries) != TRAIN_COUNTS["teacher_preservation"]:
                raise RuntimeError("teacher cache does not cover all preservation examples")
            manifest = {
                "schema_version": "qcomem-frozen-teacher-target-manifest-v1",
                "teacher": "frozen_post_trained_initial_checkpoint",
                "generated_before_optimizer_creation": True,
                "student_updates_observed_before_generation": False,
                "topk": topk,
                "tail_probability_bucket": True,
                "normalized_hidden_dtype": "torch.bfloat16",
                "world_size": WORLD_SIZE,
                "records": sum(item["records"] for item in summaries),
                "target_positions": sum(item["target_positions"] for item in summaries),
                "total_shard_bytes": sum(item["size_bytes"] for item in summaries),
                "shards": sorted(summaries, key=lambda item: item["rank"]),
            }
            atomic_json(output_root / "teacher-targets-manifest.json", manifest)
            manifest_error[0] = {
                "manifest": manifest,
                "manifest_sha256": sha256_file(output_root / "teacher-targets-manifest.json"),
            }
        except Exception as error:
            manifest_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(manifest_error, src=0)
    if isinstance(manifest_error[0], str):
        raise RuntimeError(manifest_error[0])
    dist.barrier()
    # Every rank verifies every frozen shard hash, but retains only its own data.
    manifest_record = manifest_error[0]
    assert isinstance(manifest_record, dict)
    for item in manifest_record["manifest"]["shards"]:
        if sha256_file(output_root / item["basename"]) != item["sha256"]:
            raise RuntimeError("teacher shard changed before training")
    reloaded = torch.load(shard_path, map_location="cpu", weights_only=True)
    if reloaded["records"].keys() != local_records.keys():
        raise RuntimeError("reloaded teacher shard record IDs differ")
    del local_records, split
    torch.cuda.empty_cache()
    return reloaded["records"], manifest_record


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataset: DeploymentDataset,
    device: torch.device,
    *,
    rank: int,
    step: int,
) -> dict[str, Any]:
    model.eval()
    rows = []
    for index in range(rank, len(dataset), WORLD_SIZE):
        example = dataset[index]
        input_ids = example.input_ids.unsqueeze(0).to(device)
        labels = example.labels.unsqueeze(0).to(device)
        loss, ce, _, _ = model(input_ids, labels, torch.ones_like(input_ids))
        value = float(ce.float().item())
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite heldout CE at step {step}")
        rows.append(
            {
                "example_id": example.example_id,
                "dataset": example.dataset,
                "stratum": example.stratum,
                "sequence_tokens": example.sequence_tokens,
                "target_tokens": example.target_tokens,
                "ce": value,
            }
        )
        del input_ids, labels, loss, ce
    gathered = gather(rows)
    flat = [row for rank_rows in gathered for row in rank_rows]
    if len(flat) != len(dataset) or len({row["example_id"] for row in flat}) != len(dataset):
        raise RuntimeError("heldout examples were skipped or duplicated")
    model.train()
    return {"step": step, "summary": summarize_example_equal(flat), "rows": flat}


def main() -> None:
    value = parser()
    args = value.parse_args()
    validate_protocol(args, value)
    local_rank, rank, device = distributed_setup()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    if rank == 0:
        if args.output_dir.exists():
            raise RuntimeError(f"trainer output directory already exists: {args.output_dir}")
        args.output_dir.mkdir(parents=True, exist_ok=False)
    dist.barrier()

    data_manifest = validate_manifest(
        args.data_manifest,
        expected_sha256=args.expected_data_manifest_sha256,
        train_path=args.train_data,
        train_sha256=args.expected_train_sha256,
        heldout_path=args.heldout_data,
        heldout_sha256=args.expected_heldout_sha256,
    )
    integrity = {
        "code": validate_sha256_ledger(
            args.code_ledger,
            expected_ledger_sha256=args.expected_code_ledger_sha256,
            required_filenames=CODE_FILES,
            ledger_name="deployment_aware_code",
        ),
        "model_artifacts": validate_sha256_ledger(
            args.model_artifact_ledger,
            expected_ledger_sha256=args.expected_model_artifact_ledger_sha256,
            required_filenames=MODEL_ARTIFACT_FILES,
            ledger_name="deployment_aware_model_artifacts",
        ),
        "model_weights": validate_launcher_verified_weight_ledger(
            args.model_weight_ledger,
            expected_sha256=args.expected_model_weight_ledger_sha256,
        ),
    }
    train = DeploymentDataset(
        args.train_data,
        split="train",
        max_sequence_tokens=4096,
        expected_sha256=args.expected_train_sha256,
        expected_counts=TRAIN_COUNTS,
    )
    heldout = DeploymentDataset(
        args.heldout_data,
        split="heldout",
        max_sequence_tokens=4096,
        expected_sha256=args.expected_heldout_sha256,
        expected_counts=HELDOUT_COUNTS,
    )

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    expected_tokenizer = data_manifest["tokenizer"]
    actual_template_hash = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
    if (
        type(tokenizer).__name__ != expected_tokenizer["class"]
        or int(tokenizer.vocab_size) != expected_tokenizer["vocab_size"]
        or int(tokenizer.eos_token_id) != expected_tokenizer["eos_token_id"]
        or actual_template_hash != expected_tokenizer["chat_template_sha256"]
    ):
        raise RuntimeError("runtime tokenizer differs from the data builder")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.config.use_cache = False
    split = TorchSplitCausalLM(model)
    split.config.use_cache = False
    if len(split.layers) != TEXT_LAYERS:
        raise RuntimeError("text layer count drifted")
    model.to(device)
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started
    load_peak = torch.cuda.max_memory_allocated(device)

    teacher_cache, teacher_manifest_record = build_teacher_cache(
        model,
        train,
        device,
        args.output_dir / "teacher-targets",
        rank=rank,
        topk=args.teacher_topk,
        projection_chunk_tokens=args.teacher_projection_chunk_tokens,
    )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("teacher generation unexpectedly created gradients")

    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    plan = configure_dense_full_model_trainability(model)
    if plan["trainable_parameters"] != TEXT_PARAMETERS:
        raise RuntimeError("full-state parameter count drifted")
    if plan["parameter_dtype_counts"] != {"torch.bfloat16": TEXT_PARAMETERS}:
        raise RuntimeError("pre-FSDP parameters are not all BF16")
    core = DeploymentAwareCausalLM.from_conditional_generation(model)
    layer_classes = {type(layer) for layer in split.layers}
    del split, model

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
    global_parameters, per_rank_parameters = all_rank_parameter_count(fsdp, device)
    if global_parameters != TEXT_PARAMETERS:
        raise RuntimeError("FSDP parameter shard coverage drifted")
    if {parameter.dtype for parameter in fsdp.parameters()} != {torch.float32}:
        raise RuntimeError("persistent FSDP shards are not FP32")
    optimizer = torch.optim.AdamW(
        [parameter for parameter in fsdp.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step_index: cosine_warmup_factor(
            step_index, warmup_steps=args.warmup_steps, total_steps=args.steps
        ),
    )

    metadata: dict[str, Any] = {
        "format": FORMAT,
        "created_unix": time.time(),
        "model": str(args.model),
        "initialization": "post_trained_not_base",
        "experiment_role": "dense_long_instruction_preservation_full_sft_control",
        "algorithm_path": {
            "use_cache": False,
            "qcomem_replay_or_quantization_in_training_forward": False,
            "cache_aware_training": False,
            "claim_boundary": (
                "dense Full SFT control on deployment-oriented data; subsequent "
                "full-state replay evaluation is required"
            ),
        },
        "integrity": integrity,
        "data_manifest_sha256": args.expected_data_manifest_sha256,
        "data_governance": data_manifest["data_governance"],
        "train_data": train.audit,
        "heldout_data": heldout.audit,
        "teacher_targets": {
            **teacher_manifest_record,
            "loss_weights": {
                "hard_ce": args.hard_weight,
                "kl": args.kl_weight,
                "hidden_cosine": args.hidden_weight,
            },
            "temperature": 1.0,
        },
        "parameter_plan": plan,
        "distributed": {
            "kind": "FSDP1_FULL_SHARD",
            "world_size": WORLD_SIZE,
            "persistent_parameter_dtype": "float32",
            "forward_dtype": "bfloat16",
            "gradient_reduce_dtype": "float32",
            "per_rank_parameter_numel": per_rank_parameters,
            "global_parameter_numel": global_parameters,
            "full_replica_load_seconds": load_seconds,
            "full_replica_peak_allocated_bytes": load_peak,
        },
        "activation_checkpoint": {
            "implementation": "Transformers GradientCheckpointingLayer use_reentrant=False",
            "scope": "all_40_complete_decoder_layers_including_token_mixer_and_MoE",
        },
        "training": {
            "steps": STEPS,
            "global_examples_per_step": WORLD_SIZE,
            "examples_seen_once": 1024,
            "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps,
            "lr_schedule": "linear_warmup_then_cosine_to_zero",
            "loss_weighting": "equal_per_example_not_target_token_weighted",
            "evaluation_steps": list(EVALUATION_STEPS),
            "checkpoint_steps": list(CHECKPOINT_STEPS),
        },
        "step1_gate": {
            "planned": True,
            "checks": [
                "4096_token_backward_exercised",
                "finite_loss_gradient_optimizer_state",
                "FP32_parameter_delta_nonzero_all_layers",
                "BF16_forward_visible_delta_nonzero",
                "CUDA_reserved_headroom_at_least_512MiB",
            ],
            "separate_smoke_job_used": False,
        },
        "raw_longbench_validation_or_test_read": False,
    }
    if rank == 0:
        atomic_json(args.output_dir / "metadata.json", metadata)

    evaluations: dict[int, dict[str, Any]] = {}
    checkpoints: dict[int, dict[str, Any]] = {}

    def run_eval(step: int) -> None:
        if any(parameter.grad is not None for parameter in fsdp.parameters()):
            raise RuntimeError("heldout evaluation requires cleared gradients")
        result = evaluate(fsdp, heldout, device, rank=rank, step=step)
        evaluations[step] = result
        if rank == 0:
            atomic_json(args.output_dir / f"heldout-step-{step:06d}.json", result)
            append_jsonl(
                args.output_dir / "heldout-metrics.jsonl",
                {"step": step, "summary": result["summary"]},
            )
            print(json.dumps({"phase": "heldout", **result["summary"], "step": step}), flush=True)

    run_eval(0)
    optimizer.zero_grad(set_to_none=True)
    step1_gate: dict[str, Any] | None = None
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    for step in range(1, STEPS + 1):
        example = train.examples[(step - 1) * WORLD_SIZE + rank]
        snapshots = snapshot_fp32_local_shards(fsdp.named_parameters()) if step == 1 else None
        input_ids = example.input_ids.unsqueeze(0).to(device)
        labels = example.labels.unsqueeze(0).to(device)
        kwargs: dict[str, Any] = {}
        teacher_record = None
        if example.teacher_required:
            teacher_record = teacher_cache.get(example.example_id)
            if teacher_record is None:
                raise RuntimeError("rank-local teacher cache misses its scheduled example")
            actual_targets = labels[:, 1:][labels[:, 1:] != -100].to(torch.int32).cpu()
            if not torch.equal(actual_targets, teacher_record["target_ids"]):
                raise RuntimeError("frozen teacher targets no longer align with training labels")
            kwargs = {
                "teacher_topk_ids": teacher_record["topk_ids"].to(device),
                "teacher_topk_logprobs": teacher_record["topk_logprobs"].to(device),
                "teacher_tail_logprob": teacher_record["tail_logprob"].to(device),
                "teacher_normalized_hidden": teacher_record["normalized_hidden"].to(device),
                "hard_weight": args.hard_weight,
                "kl_weight": args.kl_weight,
                "hidden_weight": args.hidden_weight,
            }
        optimizer.zero_grad(set_to_none=True)
        loss, ce, kl, hidden = fsdp(input_ids, labels, torch.ones_like(input_ids), **kwargs)
        components = torch.stack((loss.float(), ce.float(), kl.float(), hidden.float()))
        if not torch.isfinite(components).all():
            raise RuntimeError(f"non-finite train objective at step {step}")
        loss.backward()
        grad_norm = fsdp.clip_grad_norm_(args.max_grad_norm)
        if not torch.isfinite(grad_norm) or float(grad_norm.item()) <= 0:
            raise RuntimeError(f"invalid gradient norm at step {step}")
        gate_record: dict[str, Any] | None = None
        if snapshots is not None:
            gradient = aggregate_rank_audits(gather(audit_fp32_gradients(snapshots)))
            require_full_gradient_gate(
                gradient, expected_parameters=TEXT_PARAMETERS, expected_layers=TEXT_LAYERS
            )
            gate_record = {"gradient": gradient}
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        if snapshots is not None and gate_record is not None:
            optimizer_states = gather(
                audit_adamw_fp32_state(optimizer, snapshots, expected_step=1)
            )
            if sum(item["parameter_elements"] for item in optimizer_states) != TEXT_PARAMETERS:
                raise RuntimeError("AdamW state does not cover the full model")
            local_delta = audit_fp32_parameter_delta(snapshots)
            delta = {
                precision: aggregate_rank_audits(
                    gather(local_delta[precision])
                )
                for precision in ("fp32_logical", "bf16_forward_visible")
            }
            require_parameter_delta_gate(
                delta, expected_parameters=TEXT_PARAMETERS, expected_layers=TEXT_LAYERS
            )
            total_memory = int(torch.cuda.get_device_properties(device).total_memory)
            memory = gather(
                {
                    "rank": rank,
                    "sequence_tokens": example.sequence_tokens,
                    "allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                    "total_bytes": total_memory,
                    "reserved_headroom_bytes": total_memory
                    - int(torch.cuda.max_memory_reserved(device)),
                }
            )
            if max(item["sequence_tokens"] for item in memory) != 4096:
                raise RuntimeError("step-1 did not exercise a 4096-token backward")
            if min(item["reserved_headroom_bytes"] for item in memory) < args.minimum_step1_headroom_bytes:
                raise RuntimeError("step-1 CUDA reserved-memory headroom gate failed")
            gate_record.update(
                {
                    "optimizer_state_by_rank": optimizer_states,
                    "parameter_delta": delta,
                    "memory_by_rank": memory,
                    "finite_objective": True,
                    "passed": True,
                }
            )
            step1_gate = gate_record
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        local_metric = {
            "rank": rank,
            "example_id": example.example_id,
            "dataset": example.dataset,
            "stratum": example.stratum,
            "sequence_tokens": example.sequence_tokens,
            "target_tokens": example.target_tokens,
            "loss": float(loss.detach().float().item()),
            "ce": float(ce.float().item()),
            "kl": float(kl.float().item()),
            "hidden_cosine_distance": float(hidden.float().item()),
        }
        rows = gather(local_metric)
        metric = {
            "step": step,
            "learning_rate_used": learning_rate,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "global_grad_norm_before_clip": float(grad_norm.item()),
            "rows": rows,
            "summary": summarize_example_equal(rows),
            "step1_gate_event_recorded": step == 1,
        }
        if rank == 0:
            append_jsonl(args.output_dir / "train-metrics.jsonl", metric)
            print(json.dumps(metric, sort_keys=True), flush=True)
        del input_ids, labels, loss, ce, kl, hidden, snapshots, teacher_record
        if step in EVALUATION_STEPS:
            run_eval(step)
        if step in CHECKPOINT_STEPS:
            summary = evaluations[step]["summary"]
            checkpoint = save_eval_model_only_fp32(
                fsdp,
                args.output_dir / "checkpoints" / f"step-{step:06d}-eval-model-only-fp32",
                step=step,
                expected_parameters=TEXT_PARAMETERS,
                provenance={
                    "formal_format": FORMAT,
                    "model_initialization": "post_trained_not_base",
                    "train_sha256": args.expected_train_sha256,
                    "heldout_sha256": args.expected_heldout_sha256,
                    "data_manifest_sha256": args.expected_data_manifest_sha256,
                    "teacher_manifest_sha256": teacher_manifest_record["manifest_sha256"],
                    "heldout_example_equal_mean_ce": summary["overall"][
                        "example_equal_mean_ce"
                    ],
                    "raw_longbench_validation_or_test_read": False,
                },
            )
            checkpoints[step] = checkpoint
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
                        "heldout_example_equal_mean_ce": summary["overall"][
                            "example_equal_mean_ce"
                        ],
                    },
                )

    if step1_gate is None:
        raise RuntimeError("step-1 gate record is missing")
    if rank == 0:
        if sorted(checkpoints) != list(CHECKPOINT_STEPS):
            raise RuntimeError("formal checkpoint set is incomplete")
        candidates = {
            step: evaluations[step]["summary"]["overall"]["example_equal_mean_ce"]
            for step in CHECKPOINT_STEPS
        }
        best_step = min(candidates, key=lambda step: (candidates[step], step))
        best = {
            "schema_version": "qcomem-deployment-aware-best-checkpoint-v1",
            "selection_metric": "heldout.overall.example_equal_mean_ce",
            "selection_direction": "min",
            "candidate_steps": list(CHECKPOINT_STEPS),
            "candidate_values": candidates,
            "selected_step": best_step,
            "checkpoint_path": checkpoints[best_step]["checkpoint_path"],
            "checkpoint_manifest_sha256": checkpoints[best_step][
                "checkpoint_manifest_sha256"
            ],
            "diagnostic_train_source_heldout_not_final_downstream": True,
        }
        atomic_json(args.output_dir / "best-checkpoint.json", best)
        metadata["finished_unix"] = time.time()
        metadata["last_step"] = STEPS
        metadata["step1_gate"]["observed"] = step1_gate
        metadata["step1_gate"]["passed"] = True
        metadata["heldout_summaries"] = {
            str(step): evaluations[step]["summary"] for step in EVALUATION_STEPS
        }
        metadata["checkpoint_records"] = {
            str(step): {
                key: checkpoint[key]
                for key in (
                    "checkpoint_path",
                    "checkpoint_manifest_sha256",
                    "payload_directory_sha256",
                )
            }
            for step, checkpoint in checkpoints.items()
        }
        metadata["best_checkpoint"] = best
        metadata["runtime_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
        metadata["runtime_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        atomic_json(args.output_dir / "metadata.json", metadata)
        print(
            json.dumps(
                {"phase": "dense_long_instruction_preservation_sft_control_complete", "best_step": best_step},
                sort_keys=True,
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
