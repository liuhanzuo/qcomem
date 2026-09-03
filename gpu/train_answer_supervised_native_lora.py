from __future__ import annotations

"""Formal answer-supervised, task-balanced native-cache Q-CoMem LoRA B."""

import argparse
import datetime
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
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from deployment_aware_sft import (
    HELDOUT_COUNTS,
    TRAIN_COUNTS,
    DeploymentDataset,
    DeploymentExample,
    frozen_teacher_targets,
    validate_manifest,
)
from qcomem_answer_supervised_lora import (
    CHECKPOINT_STEPS,
    DEPTH,
    EVALUATION_STEPS,
    EXPECTED_ADAPTER_MODULES,
    EXPECTED_ADAPTER_PARAMETERS,
    EXPECTED_ADAPTER_PARAMETER_TENSORS,
    FORMAT,
    FULL_ATTENTION_TARGET_SUFFIXES,
    GDN_TARGET_SUFFIXES,
    MAX_ADAPTER_PARAMETERS,
    STEPS,
    TEACHER_FORMAT,
    TEACHER_MANIFEST_FORMAT,
    WORLD_SIZE,
    AnswerBoundary,
    AnswerLoRAContractError,
    AnswerSupervisedNativeLoRA,
    adapter_parameter_records,
    answer_adapter_config,
    answer_boundary,
    answer_decode_semantic_diagnostic,
    balance_group,
    balanced_domain_schedule,
    choose_best_checkpoint,
    evaluate_step1_gate,
    evaluate_step2_gate,
    example_balance_weights,
    gradient_records,
    hybrid_initialize_adapters,
    install_and_audit_adapters,
    optimizer_fp32_audit,
    reject_longbench_path_or_digest,
    require_sha256,
    sha256_file,
    stable_json,
    summarize_weighted_examples,
    teacher_target_contract,
    update_records,
)
from qcomem_lora import (
    ReplayQuantConfig,
    iter_lora_modules,
    load_lora_state_dict,
    lora_state_dict,
)
from qcomem_torch import TorchSplitCausalLM
from supervised_sft import validate_sha256_ledger


CODE_FILES = frozenset(
    {
        "qcomem_answer_supervised_lora.py",
        "train_answer_supervised_native_lora.py",
        "test_answer_supervised_native_lora.py",
        "run_answer_lora_full_state_downstream.py",
        "aggregate_answer_lora_full_state_downstream.py",
        "test_answer_lora_full_state_downstream.py",
        "launch_answer_supervised_native_lora_8gpu.sh",
        "lora_answer_supervised_native_128.json",
        "deployment_aware_sft.py",
        "qcomem_lora.py",
        "qcomem_torch.py",
        "qcomem_qwen35_native_cache.py",
        "supervised_sft.py",
        "run_downstream.py",
        "run_replay_diagnostic.py",
        "analyze_validation.py",
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
    f"model.safetensors-{index:05d}-of-00014.safetensors"
    for index in range(1, 15)
)
DOMAIN_TRAIN_COUNTS = {"qasper": 256, "2wikimqa": 154}
DOMAIN_HELDOUT_COUNTS = {"qasper": 12, "2wikimqa": 14}
EXPECTED_TEACHER_RECORDS = 436
EXPECTED_TEACHER_POSITIONS = 5_992


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


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
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
        description="Answer-supervised task-balanced native-cache LoRA B"
    )
    value.add_argument("--config", type=Path)
    value.add_argument("--model", type=Path, required="model" not in defaults)
    value.add_argument("--train-data", type=Path, required="train_data" not in defaults)
    value.add_argument("--heldout-data", type=Path, required="heldout_data" not in defaults)
    value.add_argument("--data-manifest", type=Path, required="data_manifest" not in defaults)
    value.add_argument("--independent-audit", type=Path, required=True)
    value.add_argument("--init-adapter", type=Path, required=True)
    value.add_argument("--expected-train-sha256", required=True)
    value.add_argument("--expected-heldout-sha256", required=True)
    value.add_argument("--expected-data-manifest-sha256", required=True)
    value.add_argument("--expected-independent-audit-sha256", required=True)
    value.add_argument("--expected-init-adapter-sha256", required=True)
    value.add_argument("--code-ledger", type=Path, required=True)
    value.add_argument("--expected-code-ledger-sha256", required=True)
    value.add_argument("--model-artifact-ledger", type=Path, required=True)
    value.add_argument("--expected-model-artifact-ledger-sha256", required=True)
    value.add_argument("--model-weight-ledger", type=Path, required=True)
    value.add_argument("--expected-model-weight-ledger-sha256", required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--steps", type=int, default=STEPS)
    value.add_argument("--depth", type=int, default=DEPTH)
    value.add_argument("--learning-rate", type=float, default=2e-5)
    value.add_argument("--warmup-steps", type=int, default=8)
    value.add_argument("--weight-decay", type=float, default=0.0)
    value.add_argument("--max-grad-norm", type=float, default=1.0)
    value.add_argument("--seed", type=int, default=20260814)
    value.add_argument("--teacher-topk", type=int, default=32)
    value.add_argument("--teacher-projection-chunk-tokens", type=int, default=32)
    value.add_argument("--student-projection-chunk-positions", type=int, default=32)
    value.add_argument("--hard-weight", type=float, default=0.45)
    value.add_argument("--kl-weight", type=float, default=0.35)
    value.add_argument("--hidden-weight", type=float, default=0.20)
    value.add_argument("--lora-rank", type=int, default=32)
    value.add_argument("--lora-alpha", type=float, default=64.0)
    value.add_argument("--lora-dropout", type=float, default=0.0)
    value.add_argument("--residual-bits", type=int, default=4)
    value.add_argument("--attention-bits", type=int, default=4)
    value.add_argument("--linear-bits", type=int, default=8)
    value.add_argument("--cache-layer-bits", default="8,8,8,4,8,8,8")
    value.add_argument("--group-size", type=int, default=64)
    value.add_argument("--max-adapter-parameters", type=int, default=MAX_ADAPTER_PARAMETERS)
    value.add_argument("--minimum-step1-headroom-bytes", type=int, default=4_294_967_296)
    unknown = sorted(set(defaults) - {action.dest for action in value._actions})
    if unknown:
        value.error(f"unknown config keys: {unknown}")
    value.set_defaults(**defaults)
    return value


def validate_protocol(args: argparse.Namespace, value: argparse.ArgumentParser) -> None:
    frozen = {
        "steps": 128,
        "depth": 7,
        "learning_rate": 2e-5,
        "warmup_steps": 8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "seed": 20260814,
        "teacher_topk": 32,
        "teacher_projection_chunk_tokens": 32,
        "student_projection_chunk_positions": 32,
        "hard_weight": 0.45,
        "kl_weight": 0.35,
        "hidden_weight": 0.20,
        "lora_rank": 32,
        "lora_alpha": 64.0,
        "lora_dropout": 0.0,
        "residual_bits": 4,
        "attention_bits": 4,
        "linear_bits": 8,
        "cache_layer_bits": "8,8,8,4,8,8,8",
        "group_size": 64,
        "max_adapter_parameters": 27_000_000,
        "minimum_step1_headroom_bytes": 4_294_967_296,
    }
    for key, expected in frozen.items():
        if getattr(args, key) != expected:
            value.error(f"formal protocol freezes {key}={expected!r}")


def distributed_setup() -> tuple[int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != WORLD_SIZE:
        raise SystemExit("formal LoRA B requires exactly eight ranks")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("formal LoRA B requires native-BF16 CUDA GPUs")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=4))
    return local_rank, rank, torch.device("cuda", local_rank)


def gather(value: Any) -> list[Any]:
    result: list[Any] = [None] * WORLD_SIZE
    dist.all_gather_object(result, value)
    return result


def lightweight_weight_ledger(path: Path, expected_sha256: str) -> dict[str, Any]:
    expected_sha256 = require_sha256(expected_sha256, "model-weight ledger SHA256")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise AnswerLoRAContractError("model-weight ledger SHA256 mismatch")
    entries, names = [], set()
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise AnswerLoRAContractError(f"invalid model-weight ledger line {line_number}")
        digest, raw_path = match.groups()
        artifact = Path(raw_path)
        if not artifact.is_absolute() or not artifact.is_file():
            raise AnswerLoRAContractError(f"model weight is missing: {artifact}")
        names.add(artifact.name)
        entries.append({"filename": artifact.name, "path": str(artifact), "sha256": digest})
    if names != set(MODEL_WEIGHT_FILES):
        raise AnswerLoRAContractError("model-weight ledger must contain exactly 14 shards")
    return {
        "ledger_path": str(path),
        "ledger_sha256": actual,
        "entries": entries,
        "contents_verified_once_by_launcher": True,
        "trainer_did_not_rehash_large_shards": True,
    }


def load_domain_rows(
    path: Path, examples: Sequence[DeploymentExample]
) -> tuple[list[DeploymentExample], dict[str, dict[str, Any]]]:
    raw = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if line != stable_json(row):
            raise AnswerLoRAContractError(f"noncanonical data line {line_number}")
        if row.get("stratum") == "domain":
            raw[row["example_id"]] = row
    domain = [row for row in examples if row.stratum == "domain"]
    if set(raw) != {row.example_id for row in domain}:
        raise AnswerLoRAContractError("raw/domain example binding drifted")
    return domain, raw


def validate_domain_counts(
    examples: Sequence[DeploymentExample], expected: Mapping[str, int], label: str
) -> None:
    counts = Counter(row.dataset for row in examples)
    if dict(counts) != dict(expected):
        raise AnswerLoRAContractError(
            f"{label} domain counts differ: expected={dict(expected)}, actual={dict(counts)}"
        )


def build_teacher_cache(
    model: torch.nn.Module,
    examples: Sequence[DeploymentExample],
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
    local_indices = []
    for index in range(rank, len(examples), WORLD_SIZE):
        example = examples[index]
        input_ids = example.input_ids.unsqueeze(0).to(device)
        labels = example.labels.unsqueeze(0).to(device)
        record = frozen_teacher_targets(
            split.language_model,
            split.lm_head,
            input_ids,
            labels,
            torch.ones_like(input_ids),
            topk=topk,
            projection_chunk_tokens=projection_chunk_tokens,
        )
        targets = labels[:, 1:][labels[:, 1:] != -100].cpu()
        if not torch.equal(record["target_ids"].long(), targets.long()):
            raise AnswerLoRAContractError("teacher answer/EOS alignment failed")
        local_records[example.example_id] = record
        local_indices.append(index)
        del input_ids, labels, record
    shard_path = output_root / f"teacher-rank-{rank:02d}.pt"
    temporary = shard_path.with_suffix(".pt.incomplete")
    torch.save(
        {
            "format": TEACHER_FORMAT,
            "rank": rank,
            "world_size": WORLD_SIZE,
            "records": local_records,
            "source_indices": local_indices,
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
        "answer_positions": sum(
            int(record["target_ids"].numel()) for record in local_records.values()
        ),
        "source_indices": local_indices,
    }
    summaries = gather(local_summary)
    broadcast: list[Any] = [None]
    if rank == 0:
        manifest = {
            "schema_version": TEACHER_MANIFEST_FORMAT,
            "teacher": "frozen_post_trained_qwen3.5_dense_full_sequence",
            "positions": "answer_plus_eos_only",
            "generated_before_adapter_install_and_optimizer_creation": True,
            "student_updates_observed_before_generation": False,
            "domain_only": True,
            "excluded_strata": ["general_replay", "teacher_preservation"],
            "topk": topk,
            "tail_probability_bucket": True,
            "normalized_hidden_dtype": "torch.bfloat16",
            "records": sum(item["records"] for item in summaries),
            "answer_positions": sum(item["answer_positions"] for item in summaries),
            "world_size": WORLD_SIZE,
            "shards": sorted(summaries, key=lambda item: item["rank"]),
            "raw_longbench_validation_or_test_read": False,
        }
        if (
            manifest["records"] != EXPECTED_TEACHER_RECORDS
            or manifest["answer_positions"] != EXPECTED_TEACHER_POSITIONS
        ):
            raise AnswerLoRAContractError("teacher manifest coverage drifted")
        manifest_path = output_root / "teacher-manifest.json"
        atomic_json(manifest_path, manifest)
        ledger_path = output_root / "teacher-artifacts.sha256"
        ledger_path.write_text(
            "".join(
                f"{item['sha256']}  {output_root / item['basename']}\n"
                for item in manifest["shards"]
            )
            + f"{sha256_file(manifest_path)}  {manifest_path}\n"
        )
        broadcast[0] = {
            "manifest": manifest,
            "manifest_path": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "ledger_path": str(ledger_path),
            "ledger_sha256": sha256_file(ledger_path),
        }
    dist.broadcast_object_list(broadcast, src=0)
    manifest_record = broadcast[0]
    if not isinstance(manifest_record, dict):
        raise AnswerLoRAContractError("teacher manifest broadcast failed")
    dist.barrier()
    merged: dict[str, dict[str, torch.Tensor]] = {}
    for item in manifest_record["manifest"]["shards"]:
        path = output_root / item["basename"]
        if sha256_file(path) != item["sha256"]:
            raise AnswerLoRAContractError("teacher shard SHA256 changed")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format") != TEACHER_FORMAT:
            raise AnswerLoRAContractError("teacher shard format drifted")
        overlap = set(merged) & set(payload["records"])
        if overlap:
            raise AnswerLoRAContractError("teacher shards repeat examples")
        merged.update(payload["records"])
    if len(merged) != EXPECTED_TEACHER_RECORDS:
        raise AnswerLoRAContractError("merged teacher cache misses examples")
    del split, local_records
    torch.cuda.empty_cache()
    return merged, manifest_record


def cosine_factor(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(step, 1) / warmup_steps
    progress = min(max((step - warmup_steps) / (total_steps - warmup_steps), 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def checkpoint_payload(
    model: torch.nn.Module, *, step: int, provenance: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "step": step,
        "lora": lora_state_dict(model),
        "metadata": dict(provenance),
    }


def save_checkpoint(
    model: torch.nn.Module,
    output_dir: Path,
    *,
    step: int,
    provenance: Mapping[str, Any],
    rank: int,
) -> dict[str, Any]:
    path = output_dir / f"checkpoint-{step:06d}.pt"
    if rank == 0:
        temporary = path.with_suffix(".pt.incomplete")
        torch.save(checkpoint_payload(model, step=step, provenance=provenance), temporary)
        os.replace(temporary, path)
    dist.barrier()
    local = {"rank": rank, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    records = gather(local)
    if len({row["sha256"] for row in records}) != 1 or len(
        {row["size_bytes"] for row in records}
    ) != 1:
        raise AnswerLoRAContractError("checkpoint differs across rank readers")
    return {
        "step": step,
        "path": str(path),
        "sha256": records[0]["sha256"],
        "size_bytes": records[0]["size_bytes"],
        "verified_by_all_ranks": True,
    }


def teacher_to_device(record: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in record.items()
        if key != "target_ids"
    }


def run_one(
    core: AnswerSupervisedNativeLoRA,
    example: DeploymentExample,
    raw: Mapping[str, Any],
    teacher: Mapping[str, torch.Tensor],
    device: torch.device,
) -> tuple[dict[str, Any], Any]:
    boundary = answer_boundary(example, raw_row=raw)
    teacher_target_contract(teacher, boundary.target_ids)
    moved = teacher_to_device(teacher, device)
    output = core(
        boundary.document_ids.to(device),
        boundary.continuation_input_ids.to(device),
        boundary.target_ids.to(device),
        moved["topk_ids"],
        moved["topk_logprobs"],
        moved["tail_logprob"],
        moved["normalized_hidden"],
    )
    return output, boundary


@torch.no_grad()
def evaluate(
    core: AnswerSupervisedNativeLoRA,
    examples: Sequence[DeploymentExample],
    raw_rows: Mapping[str, Mapping[str, Any]],
    teacher_cache: Mapping[str, Mapping[str, torch.Tensor]],
    balance_weights: Mapping[str, float],
    device: torch.device,
    *,
    rank: int,
    step: int,
) -> dict[str, Any]:
    core.eval()
    local = []
    for index in range(rank, len(examples), WORLD_SIZE):
        example = examples[index]
        output, boundary = run_one(
            core,
            example,
            raw_rows[example.example_id],
            teacher_cache[example.example_id],
            device,
        )
        local.append(
            {
                "example_id": example.example_id,
                "dataset": example.dataset,
                "stratum": example.stratum,
                "balance_group": balance_group(example.stratum, example.dataset),
                "balance_weight": balance_weights[example.example_id],
                "sequence_tokens": example.sequence_tokens,
                "answer_positions": boundary.answer_positions,
                **{
                    key: float(output[key].detach().float().item())
                    for key in ("loss", "ce", "kl", "hidden")
                },
            }
        )
    rows = [row for part in gather(local) for row in part]
    if len(rows) != len(examples) or len({row["example_id"] for row in rows}) != len(examples):
        raise AnswerLoRAContractError("heldout evaluation skipped or duplicated examples")
    core.model.eval()
    for module in iter_lora_modules(core.model):
        module.train()
    return {"step": step, "summary": summarize_weighted_examples(rows), "rows": rows}


def parameter_group(name: str) -> str:
    module_leaf = name.split(".")[-2]
    parameter_leaf = name.split(".")[-1]
    if module_leaf in FULL_ATTENTION_TARGET_SUFFIXES:
        return "full_attention"
    if module_leaf in GDN_TARGET_SUFFIXES and parameter_leaf == "lora_a":
        return "gdn_lora_a"
    if module_leaf in GDN_TARGET_SUFFIXES and parameter_leaf == "lora_b":
        return "gdn_lora_b"
    raise AnswerLoRAContractError(f"unexpected adapter parameter {name}")


def summarize_audit_group(
    group: str,
    gradients: Sequence[Mapping[str, Any]],
    updates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    grad = [row for row in gradients if parameter_group(str(row["name"])) == group]
    update = [row for row in updates if parameter_group(str(row["name"])) == group]
    return {
        "gradient_tensors": len(grad),
        "finite_gradient_tensors": sum(bool(row["finite"]) for row in grad),
        "nonzero_gradient_tensors": sum(bool(row["nonzero"]) for row in grad),
        "update_tensors": len(update),
        "finite_update_tensors": sum(bool(row["finite"]) for row in update),
        "nonzero_update_tensors": sum(bool(row["nonzero"]) for row in update),
    }


def main() -> None:
    value = parser()
    args = value.parse_args()
    validate_protocol(args, value)
    for path, digest in (
        (args.train_data, args.expected_train_sha256),
        (args.heldout_data, args.expected_heldout_sha256),
        (args.data_manifest, args.expected_data_manifest_sha256),
    ):
        reject_longbench_path_or_digest(path, digest)
    local_rank, rank, device = distributed_setup()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if rank == 0:
        if args.output_dir.exists():
            raise AnswerLoRAContractError(f"output directory exists: {args.output_dir}")
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
    if sha256_file(args.independent_audit) != require_sha256(
        args.expected_independent_audit_sha256, "independent audit SHA256"
    ):
        raise AnswerLoRAContractError("independent data audit SHA256 mismatch")
    independent_audit = json.loads(args.independent_audit.read_text())
    if independent_audit.get("passed") is not True:
        raise AnswerLoRAContractError("independent data audit did not pass")
    integrity = {
        "code": validate_sha256_ledger(
            args.code_ledger,
            expected_ledger_sha256=args.expected_code_ledger_sha256,
            required_filenames=CODE_FILES,
            ledger_name="answer_lora_code",
        ),
        "model_artifacts": validate_sha256_ledger(
            args.model_artifact_ledger,
            expected_ledger_sha256=args.expected_model_artifact_ledger_sha256,
            required_filenames=MODEL_ARTIFACT_FILES,
            ledger_name="answer_lora_model_artifacts",
        ),
        "model_weights": lightweight_weight_ledger(
            args.model_weight_ledger, args.expected_model_weight_ledger_sha256
        ),
    }
    train_all = DeploymentDataset(
        args.train_data,
        split="train",
        max_sequence_tokens=4096,
        expected_sha256=args.expected_train_sha256,
        expected_counts=TRAIN_COUNTS,
    )
    heldout_all = DeploymentDataset(
        args.heldout_data,
        split="heldout",
        max_sequence_tokens=4096,
        expected_sha256=args.expected_heldout_sha256,
        expected_counts=HELDOUT_COUNTS,
    )
    train, train_raw = load_domain_rows(args.train_data, train_all.examples)
    heldout, heldout_raw = load_domain_rows(args.heldout_data, heldout_all.examples)
    validate_domain_counts(train, DOMAIN_TRAIN_COUNTS, "train")
    validate_domain_counts(heldout, DOMAIN_HELDOUT_COUNTS, "heldout")
    schedule, schedule_audit = balanced_domain_schedule(
        train, steps=STEPS, world_size=WORLD_SIZE, seed=args.seed
    )
    train_by_id = {row.example_id: row for row in train}
    train_weight_audit = {
        "kind": "equal_scheduled_occurrence_weight_v1",
        "per_global_step": {"qasper": 4, "2wikimqa": 4},
        "task_mass_per_global_step": {"qasper": 0.5, "2wikimqa": 0.5},
        "all_occurrence_weights": 1.0,
        "target_token_weighting_used": False,
    }
    heldout_weights, heldout_weight_audit = example_balance_weights(heldout)

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    expected_tokenizer = data_manifest["tokenizer"]
    if (
        type(tokenizer).__name__ != expected_tokenizer["class"]
        or int(tokenizer.vocab_size) != expected_tokenizer["vocab_size"]
        or int(tokenizer.eos_token_id) != expected_tokenizer["eos_token_id"]
        or hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
        != expected_tokenizer["chat_template_sha256"]
    ):
        raise AnswerLoRAContractError("runtime tokenizer differs from frozen data")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().requires_grad_(False).to(device)
    split = TorchSplitCausalLM(model)
    if len(split.layers) != 40:
        raise AnswerLoRAContractError("Qwen3.5 text layer count drifted")

    # Teacher cache is frozen before adapters or optimizer exist.
    teacher_cache, teacher_manifest = build_teacher_cache(
        model,
        [*train, *heldout],
        device,
        args.output_dir / "teacher-targets",
        rank=rank,
        topk=args.teacher_topk,
        projection_chunk_tokens=args.teacher_projection_chunk_tokens,
    )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AnswerLoRAContractError("teacher generation unexpectedly created gradients")

    installed, surface_audit = install_and_audit_adapters(
        model,
        split,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        initialization_seed=args.seed,
    )
    initialization = hybrid_initialize_adapters(
        model,
        source_checkpoint=args.init_adapter,
        expected_sha256=args.expected_init_adapter_sha256,
        initialization_seed=args.seed,
    )
    parameters = adapter_parameter_records(model)
    trainable = sum(parameter.numel() for parameter in parameters.values())
    if trainable != EXPECTED_ADAPTER_PARAMETERS or trainable > args.max_adapter_parameters:
        raise AnswerLoRAContractError(
            f"adapter parameter gate failed: {trainable} > {args.max_adapter_parameters}"
        )
    parameter_memory = {
        "trainable_parameters": trainable,
        "hard_cap": args.max_adapter_parameters,
        "fp32_parameter_bytes": trainable * 4,
        "fp32_parameter_plus_gradient_plus_adam_m_v_bytes": trainable * 4 * 4,
        "full_attention_parameters": 6_193_152,
        "gdn_parameters": 20_496_384,
    }
    frozen_adapter_config = answer_adapter_config(
        model,
        installed=installed,
        surface_audit=surface_audit,
    )
    model.eval()
    for module in iter_lora_modules(model):
        module.train()
    quant = ReplayQuantConfig(
        residual_bits=args.residual_bits,
        attention_bits=args.attention_bits,
        linear_bits=args.linear_bits,
        cache_layer_bits=tuple(int(item) for item in args.cache_layer_bits.split(",")),
        group_size=args.group_size,
    )
    core = AnswerSupervisedNativeLoRA(
        model,
        depth=DEPTH,
        quant=quant,
        hard_weight=args.hard_weight,
        kl_weight=args.kl_weight,
        hidden_weight=args.hidden_weight,
        projection_chunk_positions=args.student_projection_chunk_positions,
    )
    ddp = DistributedDataParallel(
        core,
        device_ids=[local_rank],
        output_device=local_rank,
        broadcast_buffers=False,
        find_unused_parameters=False,
    )
    optimizer = torch.optim.AdamW(
        list(parameters.values()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        foreach=False,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda index: cosine_factor(
            index, warmup_steps=args.warmup_steps, total_steps=STEPS
        ),
    )
    provenance = {
        "format": FORMAT,
        "initialization": initialization,
        "initialization_attribution": {
            "full_attention_modules_warm_started": 36,
            "gdn_modules_cold_started": 120,
            "warm_start_source_step": 0,
            "warm_start_prior_downstream_point_estimate": "known_negative",
            "prior_downstream_result_used_for_current_checkpoint_selection": False,
            "pure_cold_start_experiment": False,
            "experiment_interpretation": (
                "tests whether answer supervision plus expansion from 36 full-attention "
                "modules to a 156-module full-attention-and-GDN surface can repair the "
                "warm-started system; it is not a pure cold-start LoRA B"
            ),
            "step_zero_remains_eligible_for_official_train_heldout_selection": True,
            "downstream_adapter_disabled_control_required": True,
        },
        "model": str(args.model),
        "data": {
            "train_sha256": args.expected_train_sha256,
            "heldout_sha256": args.expected_heldout_sha256,
            "manifest_sha256": args.expected_data_manifest_sha256,
            "independent_audit_sha256": args.expected_independent_audit_sha256,
            "train_domain_counts": DOMAIN_TRAIN_COUNTS,
            "heldout_domain_counts": DOMAIN_HELDOUT_COUNTS,
            "included_strata": ["domain"],
            "excluded_strata": ["general_replay", "teacher_preservation"],
            "exclusion_reason": (
                "Tulu rows have no unambiguous frozen document/query boundary; no "
                "synthetic boundary is invented in the primary experiment"
            ),
        },
        "schedule": {**schedule_audit, "occurrence_weighting": train_weight_audit},
        "heldout_weighting": heldout_weight_audit,
        "teacher": teacher_manifest,
        "adapter_config": frozen_adapter_config,
        "adapter_surface": surface_audit,
        "adapter_memory": parameter_memory,
        "loss": {
            "positions": "answer_plus_eos_only",
            "hard_ce": args.hard_weight,
            "frozen_dense_teacher_topk_tail_kl": args.kl_weight,
            "frozen_dense_teacher_hidden_cosine": args.hidden_weight,
            "per_example_position_reduction": "mean",
            "student_projection_chunk_positions": args.student_projection_chunk_positions,
            "student_projection_chunk_backward": (
                "non_reentrant_activation_checkpoint_recompute"
            ),
            "all_answer_full_vocab_logits_retained_until_backward": False,
            "target_token_weighting_used": False,
        },
        "student": {
            "execution": "native-functional-cache",
            "boundary": "document_prefill_then_query_plus_answer_continuation",
            "teacher_forced_answer_continuation": (
                "query_plus_answer_without_eos_one_multi_token_block"
            ),
            "deployment_decode": "query_multi_token_then_answer_token_by_token",
            "teacher_forced_answer_continuation_equals_token_by_token_decode": False,
            "chunk_boundary_equivalence_claimed": False,
            "post_training_heldout_semantic_diagnostic_required": True,
            "quantization": {
                "residual_bits": 4,
                "attention_bits": 4,
                "linear_bits": 8,
                "cache_layer_bits": [8, 8, 8, 4, 8, 8, 8],
                "weights_quantized": False,
            },
        },
        "integrity": integrity,
        "governance": {
            "official_train_sources_only": True,
            "validation_6_35_used_for_tuning": False,
            "test_v2_used": False,
            "raw_longbench_validation_or_test_read": False,
        },
    }
    checkpoints: dict[int, dict[str, Any]] = {}
    evaluations: dict[int, dict[str, Any]] = {}
    checkpoints[0] = save_checkpoint(
        model, args.output_dir, step=0, provenance=provenance, rank=rank
    )

    def run_eval(step: int) -> None:
        if any(parameter.grad is not None for parameter in parameters.values()):
            raise AnswerLoRAContractError("heldout evaluation requires cleared gradients")
        result = evaluate(
            core,
            heldout,
            heldout_raw,
            teacher_cache,
            heldout_weights,
            device,
            rank=rank,
            step=step,
        )
        evaluations[step] = result
        if rank == 0:
            atomic_json(args.output_dir / f"heldout-step-{step:06d}.json", result)
            append_jsonl(
                args.output_dir / "heldout-metrics.jsonl",
                {"step": step, "summary": result["summary"]},
            )
            print(json.dumps({"phase": "heldout", "step": step, **result["summary"]}), flush=True)

    run_eval(0)
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    step_gates: dict[int, dict[str, Any]] = {}
    for step in range(1, STEPS + 1):
        example_id = schedule[(step - 1) * WORLD_SIZE + rank]
        example = train_by_id[example_id]
        snapshots = (
            {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()}
            if step in {1, 2}
            else None
        )
        optimizer.zero_grad(set_to_none=True)
        output, boundary = run_one(
            ddp,
            example,
            train_raw[example_id],
            teacher_cache[example_id],
            device,
        )
        balance_weight = 1.0
        # DDP averages the eight rank-local per-example losses.  The frozen
        # schedule supplies four QASPER and four 2Wiki examples every step, so
        # this is already exactly 0.5/0.5 task mass; no inverse-frequency
        # multiplier is applied to the optimization objective.
        weighted_loss = output["loss"]
        if not torch.isfinite(weighted_loss):
            raise AnswerLoRAContractError(f"non-finite train loss at step {step}")
        weighted_loss.backward()
        gradients = gradient_records(parameters) if snapshots is not None else None
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters.values(), args.max_grad_norm)
        if not torch.isfinite(grad_norm) or float(grad_norm.item()) <= 0:
            raise AnswerLoRAContractError(f"invalid gradient norm at step {step}")
        learning_rate = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        updates = update_records(parameters, snapshots) if snapshots is not None else None
        fp32 = optimizer_fp32_audit(optimizer, parameters) if snapshots is not None else None
        if snapshots is not None and gradients is not None and updates is not None:
            total = int(torch.cuda.get_device_properties(device).total_memory)
            local_gate = {
                "rank": rank,
                "example_id": example_id,
                "dataset": example.dataset,
                "sequence_tokens": example.sequence_tokens,
                "answer_positions": boundary.answer_positions,
                "continuation_positions": int(output["continuation_positions"]),
                "gradient_tensors": len(gradients),
                "finite_gradient_tensors": sum(bool(row["finite"]) for row in gradients),
                "nonzero_gradient_tensors": sum(bool(row["nonzero"]) for row in gradients),
                "finite_update_tensors": sum(bool(row["finite"]) for row in updates),
                "nonzero_update_tensors": sum(bool(row["nonzero"]) for row in updates),
                "full_attention": summarize_audit_group("full_attention", gradients, updates),
                "gdn_lora_a": summarize_audit_group("gdn_lora_a", gradients, updates),
                "gdn_lora_b": summarize_audit_group("gdn_lora_b", gradients, updates),
                "optimizer_fp32": fp32,
                "cache": output["cache_audit"],
                "total_memory_bytes": total,
                "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "reserved_headroom_bytes": total - int(torch.cuda.max_memory_reserved(device)),
                "finite_loss": True,
            }
            gate_rows = gather(local_gate)
            gate = (
                evaluate_step1_gate(
                    gate_rows,
                    minimum_headroom_bytes=args.minimum_step1_headroom_bytes,
                )
                if step == 1
                else evaluate_step2_gate(gate_rows)
            )
            if rank == 0:
                atomic_json(args.output_dir / f"step-{step}-hard-gate.json", gate)
            decision: list[Any] = [gate if rank == 0 else None]
            dist.broadcast_object_list(decision, src=0)
            gate = decision[0]
            if gate.get("status") != "passed":
                raise RuntimeError(f"step-{step} adapter/cache hard gate failed")
            step_gates[step] = gate
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        local_metric = {
            "rank": rank,
            "example_id": example_id,
            "dataset": example.dataset,
            "stratum": example.stratum,
            "balance_group": balance_group(example.stratum, example.dataset),
            "balance_weight": balance_weight,
            "sequence_tokens": example.sequence_tokens,
            "answer_positions": boundary.answer_positions,
            **{
                key: float(output[key].detach().float().item())
                for key in ("loss", "ce", "kl", "hidden")
            },
        }
        rows = gather(local_metric)
        metric = {
            "step": step,
            "learning_rate_used": learning_rate,
            "next_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "global_grad_norm_before_clip": float(grad_norm.item()),
            "rows": rows,
            "summary": summarize_weighted_examples(rows),
            "step_gate_recorded": step if step in {1, 2} else None,
        }
        if rank == 0:
            append_jsonl(args.output_dir / "train-metrics.jsonl", metric)
            print(json.dumps(metric, sort_keys=True), flush=True)
        del output, weighted_loss, snapshots, gradients, updates
        if step in EVALUATION_STEPS:
            run_eval(step)
        if step in CHECKPOINT_STEPS and step != 0:
            checkpoints[step] = save_checkpoint(
                model, args.output_dir, step=step, provenance=provenance, rank=rank
            )

    if set(step_gates) != {1, 2}:
        raise AnswerLoRAContractError("step-1/2 gate records are incomplete")
    best_step = choose_best_checkpoint(evaluations)
    selected_payload = torch.load(
        checkpoints[best_step]["path"], map_location="cpu", weights_only=True
    )
    if selected_payload.get("format") != FORMAT or selected_payload.get("step") != best_step:
        raise AnswerLoRAContractError("selected checkpoint payload drifted")
    load_lora_state_dict(model, selected_payload["lora"])
    model.eval()
    local_semantic = []
    for index in range(rank, len(heldout), WORLD_SIZE):
        example = heldout[index]
        boundary = answer_boundary(example, raw_row=heldout_raw[example.example_id])
        row = answer_decode_semantic_diagnostic(
            model,
            AnswerBoundary(
                document_ids=boundary.document_ids.to(device),
                query_ids=boundary.query_ids.to(device),
                answer_ids=boundary.answer_ids.to(device),
                continuation_input_ids=boundary.continuation_input_ids.to(device),
                target_ids=boundary.target_ids.to(device),
                kind=boundary.kind,
            ),
            quant=quant,
            depth=DEPTH,
            projection_chunk_positions=args.student_projection_chunk_positions,
        )
        local_semantic.append(
            {"example_id": example.example_id, "dataset": example.dataset, **row}
        )
    semantic_rows = [row for part in gather(local_semantic) for row in part]
    if len(semantic_rows) != len(heldout):
        raise AnswerLoRAContractError("decode semantic diagnostic coverage drifted")
    semantic_diagnostic = {
        "schema_version": "qcomem_answer_decode_chunk_diagnostic_v1",
        "checkpoint_step": best_step,
        "official_train_heldout_domain_only": True,
        "examples": len(semantic_rows),
        "positions": sum(row["positions"] for row in semantic_rows),
        "top1_equal_positions": sum(row["top1_equal_positions"] for row in semantic_rows),
        "top1_agreement": sum(row["top1_equal_positions"] for row in semantic_rows)
        / sum(row["positions"] for row in semantic_rows),
        "example_mean_kl_whole_to_token": sum(
            row["mean_kl_whole_to_token"] for row in semantic_rows
        )
        / len(semantic_rows),
        "maximum_kl_whole_to_token": max(
            row["max_kl_whole_to_token"] for row in semantic_rows
        ),
        "maximum_abs_logit_difference": max(
            row["max_abs_logit_difference"] for row in semantic_rows
        ),
        "examples_with_top1_divergence": sum(
            row["first_top1_divergence_position"] is not None
            for row in semantic_rows
        ),
        "equivalence_threshold": None,
        "blocking_gate": False,
        "equivalence_claimed": False,
        "rows": semantic_rows,
        "validation_6_35_used": False,
        "test_v2_used": False,
    }
    if rank == 0:
        atomic_json(
            args.output_dir / "answer-decode-semantic-diagnostic.json",
            semantic_diagnostic,
        )
    if rank == 0:
        best = {
            "schema_version": "qcomem_answer_lora_best_checkpoint_v1",
            "selection_source": "independent_official_train_heldout_domain_only",
            "selection_metric": "task_balanced_answer_ce_kl_hidden.overall.loss",
            "selection_direction": "min",
            "candidate_steps": list(EVALUATION_STEPS),
            "candidate_values": {
                str(step): evaluations[step]["summary"]["overall"]["loss"]
                for step in EVALUATION_STEPS
            },
            "selected_step": best_step,
            "checkpoint": checkpoints[best_step],
            "validation_6_35_used_for_selection": False,
            "test_v2_used": False,
        }
        atomic_json(args.output_dir / "best-checkpoint.json", best)
        metadata = {
            **provenance,
            "created_unix": time.time(),
            "last_step": STEPS,
            "step_gates": {str(step): step_gates[step] for step in sorted(step_gates)},
            "heldout_summaries": {
                str(step): evaluations[step]["summary"] for step in EVALUATION_STEPS
            },
            "checkpoints": {str(step): checkpoints[step] for step in CHECKPOINT_STEPS},
            "best_checkpoint": best,
            "answer_decode_semantic_diagnostic": semantic_diagnostic,
            "runtime_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "runtime_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        atomic_json(args.output_dir / "metadata.json", metadata)
        print(json.dumps({"phase": "answer_supervised_native_lora_complete", "best_step": best_step}), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
