from __future__ import annotations

import argparse
import contextlib
import hashlib
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
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from qcomem_lora import (
    CoMemLoRADistillation,
    LoRAConfig,
    PG19WindowDataset,
    ReplayQuantConfig,
    adapter_metadata,
    estimate_suffix_lora_parameters,
    find_suffix_lora_targets,
    install_suffix_lora,
    iter_lora_modules,
    load_lora_state_dict,
    lora_state_dict,
    lora_gradient_coverage,
    single_window_collate,
    training_semantics_metadata,
)
from qcomem_torch import TorchSplitCausalLM


FROZEN_LONGBENCH_TEST_V2_SHA256 = (
    "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f"
)


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


def parse_layer_bits(raw: str | list[int] | None) -> tuple[int, ...] | None:
    if raw is None or raw == "":
        return None
    if isinstance(raw, list):
        return tuple(int(item) for item in raw)
    return tuple(int(item) for item in raw.split(","))


def parse_target_suffixes(raw: str | list[str]) -> tuple[str, ...]:
    if isinstance(raw, list):
        return tuple(raw)
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def audited_cache_query_position_gate(
    audits_by_rank: list[Any],
) -> dict[str, Any]:
    """Validate each rank's actual query length without assuming fixed windows.

    Explicit ``document_ids``/``query_ids`` examples bypass the tokenizer window
    sizes, so ``--query-tokens`` is not an execution-length contract for those
    rows.  The cache itself records the expected and observed continuation
    length; require those values to agree and be positive on every rank.
    """

    positions = [
        row.get("query_positions_observed") if isinstance(row, dict) else None
        for row in audits_by_rank
    ]
    valid_positions = [
        value
        for value in positions
        if isinstance(value, int) and not isinstance(value, bool) and value > 0
    ]
    passed = bool(audits_by_rank) and all(
        isinstance(row, dict)
        and row.get("hard_gate_passed") is True
        and isinstance(row.get("query_positions_expected"), int)
        and not isinstance(row.get("query_positions_expected"), bool)
        and row["query_positions_expected"] > 0
        and row.get("query_positions_observed")
        == row["query_positions_expected"]
        for row in audits_by_rank
    )
    return {
        "hard_gate_passed": passed,
        "query_positions_by_rank": positions,
        "minimum_query_positions": min(valid_positions) if valid_positions else None,
        "maximum_query_positions": max(valid_positions) if valid_positions else None,
    }


def build_parser() -> argparse.ArgumentParser:
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--config", type=Path)
    known, _ = preliminary.parse_known_args()
    defaults: dict[str, Any] = {}
    if known.config is not None:
        defaults = json.loads(known.config.read_text())

    parser = argparse.ArgumentParser(
        description="Suffix-only LoRA distillation for CoMem replay"
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path, required="model" not in defaults)
    parser.add_argument("--data", type=Path, required="data" not in defaults)
    parser.add_argument(
        "--output-dir", type=Path, required="output_dir" not in defaults
    )
    parser.add_argument("--mode", choices=("interface", "quant"), default="interface")
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--teacher-kind", choices=("dense", "q16_replay"), default="dense")
    parser.add_argument("--teacher-source", choices=("online", "offline"), default="online")
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--forward-weight", type=float, default=0.6)
    parser.add_argument("--reverse-weight", type=float, default=0.4)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--context-tokens", type=int, default=1536)
    parser.add_argument("--query-tokens", type=int, default=512)
    parser.add_argument("--stride", type=int, default=2048)
    parser.add_argument("--dataset-limit", type=int)
    parser.add_argument("--max-windows-per-record", type=int)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=0)

    parser.add_argument("--residual-bits", type=int, default=4)
    parser.add_argument("--attention-bits", type=int, default=8)
    parser.add_argument("--linear-bits", type=int, default=8)
    parser.add_argument("--cache-layer-bits")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--student-suffix-execution",
        choices=(
            "merged-uncached",
            "cached-two-stage",
            "detached-document-cache",
            "native-functional-cache",
        ),
        default="merged-uncached",
        help=(
            "quant student suffix boundary; cached-two-stage prefills document "
            "cache then continues the complete query once"
        ),
    )

    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=float, default=64.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument(
        "--target-suffixes",
        default="q_proj,k_proj,v_proj,o_proj",
    )
    parser.add_argument("--max-trainable-params", type=int, default=100000000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--init-adapter",
        type=Path,
        help=(
            "warm-start LoRA weights only and reset optimizer/scheduler/step; "
            "use this, not --resume, for Interface-to-Quant transfer"
        ),
    )
    known_destinations = {action.dest for action in parser._actions}
    unknown = sorted(set(defaults) - known_destinations)
    if unknown:
        parser.error(f"unknown config keys: {', '.join(unknown)}")
    parser.set_defaults(**defaults)
    return parser


def distributed_setup() -> tuple[int, int, int, torch.device]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the 35B LoRA training path")
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    return local_rank, rank, world_size, torch.device("cuda", local_rank)


def is_main(rank: int) -> bool:
    return rank == 0


def reject_frozen_test_data(path: Path) -> str:
    normalized = str(path).lower().replace("_", "-")
    if "qcomem-longbench-test-v2" in normalized or "longbench-test-v2" in normalized:
        raise SystemExit(
            "refusing to train on frozen LongBench test-v2; use PG-19 "
            "train/calibration data and evaluate test-v2 only after freezing"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest == FROZEN_LONGBENCH_TEST_V2_SHA256:
        raise SystemExit(
            "refusing frozen LongBench test-v2 by SHA256 even though the file "
            "was renamed; LoRA training must use an independent PG-19 split"
        )
    return digest


def cosine_warmup_decay(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return max(step, 1) / max(warmup_steps, 1)
        progress = min(
            max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0),
            1.0,
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": "qcomem_suffix_lora_v1",
        "step": step,
        "lora": lora_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state(),
        "metadata": metadata,
    }


def optimizer_state_dtypes(
    optimizer: torch.optim.Optimizer,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for state in optimizer.state.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                key = str(value.dtype)
                counts[key] = counts.get(key, 0) + value.numel()
    return counts


def save_checkpoint(
    output_dir: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    metadata: dict[str, Any],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"checkpoint-{step:06d}.pt"
    temporary = output_dir / f".{path.name}.tmp"
    torch.save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            metadata=metadata,
        ),
        temporary,
    )
    temporary.replace(path)
    (output_dir / "latest").write_text(path.name + "\n")
    return path


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    expected_metadata: dict[str, Any],
) -> int:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "qcomem_suffix_lora_v1":
        raise ValueError(f"unsupported checkpoint format in {path}")
    assert_resume_compatible(payload.get("metadata"), expected_metadata)
    load_lora_state_dict(model, payload["lora"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    torch.cuda.set_rng_state(payload["cuda_rng_state"].cpu(), device=device)
    return int(payload["step"])


def assert_resume_compatible(
    checkpoint_metadata: Any, expected_metadata: dict[str, Any]
) -> None:
    if not isinstance(checkpoint_metadata, dict):
        raise ValueError("resume checkpoint is missing metadata")
    for key in ("model", "data_sha256", "world_size", "training", "semantics"):
        if checkpoint_metadata.get(key) != expected_metadata.get(key):
            raise ValueError(
                f"resume checkpoint {key} differs from the current run; "
                "cross-mode or changed-schedule transfer must use --init-adapter"
            )
    old_adapter = checkpoint_metadata.get("adapter", {}).get("config")
    new_adapter = expected_metadata.get("adapter", {}).get("config")
    if old_adapter != new_adapter:
        raise ValueError("resume checkpoint adapter config differs from current run")


def load_adapter_warm_start(
    path: Path,
    *,
    model: torch.nn.Module,
    target_metadata: dict[str, Any],
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "qcomem_suffix_lora_v1":
        raise ValueError(f"unsupported warm-start checkpoint format in {path}")
    source = payload.get("metadata")
    if not isinstance(source, dict):
        raise ValueError("warm-start checkpoint is missing metadata")
    source_semantics = source.get("semantics", {})
    target_semantics = target_metadata.get("semantics", {})
    if source_semantics.get("depth") != target_semantics.get("depth"):
        raise ValueError("warm-start split depth differs from target depth")
    if source.get("adapter", {}).get("config") != target_metadata.get("adapter", {}).get(
        "config"
    ):
        raise ValueError("warm-start adapter architecture differs from target")
    source_mode = source_semantics.get("mode")
    target_mode = target_semantics.get("mode")
    if (source_mode, target_mode) not in {
        ("interface", "quant"),
        ("interface", "interface"),
        ("quant", "quant"),
    }:
        raise ValueError(
            f"unsupported adapter warm-start transition {source_mode!r}->{target_mode!r}"
        )
    load_lora_state_dict(model, payload["lora"])
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_step": int(payload["step"]),
        "source_mode": source_mode,
        "target_mode": target_mode,
        "optimizer_scheduler_step_restored": False,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    local_rank, rank, world_size, device = distributed_setup()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "quant" and args.teacher_kind != "q16_replay":
        parser.error("quant mode requires --teacher-kind q16_replay")
    if args.mode != "quant" and args.student_suffix_execution != "merged-uncached":
        parser.error("cached-two-stage student suffix execution requires quant mode")
    if args.steps < 1 or args.gradient_accumulation < 1:
        parser.error("steps and gradient accumulation must be positive")
    if args.resume is not None and args.init_adapter is not None:
        parser.error("--resume and --init-adapter are mutually exclusive")
    training_data_sha256 = reject_frozen_test_data(args.data)
    if args.teacher_source == "offline" and args.mode == "quant":
        # The Q16 state is still rebuilt locally; only its logits are offline.
        pass

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
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=args.seed,
            drop_last=False,
        )
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        shuffle=sampler is None,
        collate_fn=single_window_collate,
        num_workers=0,
    )

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().to(device)
    split = TorchSplitCausalLM(model)
    lora_config = LoRAConfig(
        rank=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        target_suffixes=parse_target_suffixes(args.target_suffixes),
    )
    candidates = find_suffix_lora_targets(
        split.layers,
        depth=args.depth,
        target_suffixes=lora_config.target_suffixes,
    )
    estimated_trainable = estimate_suffix_lora_parameters(
        split.layers,
        depth=args.depth,
        target_suffixes=lora_config.target_suffixes,
        rank=lora_config.rank,
    )
    if is_main(rank):
        print(
            json.dumps(
                {
                    "phase": "before_lora_install",
                    "matched_modules": len(candidates),
                    "backbone_parameters_currently_requires_grad": sum(
                        parameter.numel()
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ),
                    "estimated_adapter_trainable_parameters": estimated_trainable,
                    "max_trainable_parameters": args.max_trainable_params,
                    "target_suffixes": list(lora_config.target_suffixes),
                }
            ),
            flush=True,
        )
    if estimated_trainable > args.max_trainable_params:
        raise SystemExit(
            "refusing to install an unexpectedly large adapter: "
            f"estimated {estimated_trainable:,} trainable params > "
            f"--max-trainable-params {args.max_trainable_params:,}. "
            "For Qwen3.5 MoE, keep the default attention-only targets; "
            "MLP/expert LoRA requires an explicit ablation budget."
        )
    installed = install_suffix_lora(
        model, split.layers, depth=args.depth, config=lora_config
    )
    if installed != candidates:
        raise RuntimeError("LoRA installed-module list differs from preflight match list")
    # Preserve eval semantics for the frozen backbone, while enabling optional
    # adapter dropout only in the small trainable modules.
    model.eval()
    for module in iter_lora_modules(model):
        module.train()

    quant_config = ReplayQuantConfig(
        residual_bits=args.residual_bits,
        attention_bits=args.attention_bits,
        linear_bits=args.linear_bits,
        cache_layer_bits=parse_layer_bits(args.cache_layer_bits),
        group_size=args.group_size,
    )
    core = CoMemLoRADistillation(
        model,
        mode=args.mode,
        depth=args.depth,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        teacher_kind=args.teacher_kind,
        teacher_source=args.teacher_source,
        quant=quant_config,
        forward_weight=args.forward_weight,
        reverse_weight=args.reverse_weight,
        temperature=args.temperature,
        student_suffix_execution=args.student_suffix_execution,
    ).to(device)
    trainable = [parameter for parameter in core.parameters() if parameter.requires_grad]
    trainable_count = sum(parameter.numel() for parameter in trainable)
    if trainable_count > args.max_trainable_params:
        raise RuntimeError("post-install trainable parameter hard gate failed")
    if trainable_count != estimated_trainable:
        raise RuntimeError(
            f"estimated {estimated_trainable:,} adapter params but installed "
            f"{trainable_count:,}"
        )
    if is_main(rank):
        print(
            json.dumps(
                {
                    "phase": "after_lora_install",
                    "installed_lora_modules": len(installed),
                    "trainable_parameters": trainable_count,
                    "max_trainable_parameters": args.max_trainable_params,
                    "target_suffixes": list(lora_config.target_suffixes),
                }
            ),
            flush=True,
        )
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = cosine_warmup_decay(
        optimizer, warmup_steps=args.warmup_steps, total_steps=args.steps
    )
    metadata = {
        "created_unix": time.time(),
        "model": str(args.model),
        "data": str(args.data),
        "data_sha256": training_data_sha256,
        "world_size": world_size,
        "ddp_find_unused_parameters": False,
        "effective_windows_per_step": world_size * args.gradient_accumulation,
        "training": {
            "context_tokens": args.context_tokens,
            "query_tokens": args.query_tokens,
            "stride": args.stride,
            "dataset_limit": args.dataset_limit,
            "max_windows_per_record": args.max_windows_per_record,
            "steps": args.steps,
            "gradient_accumulation": args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_steps": args.warmup_steps,
            "schedule": "cosine_to_zero",
            "max_grad_norm": args.max_grad_norm,
            "seed": args.seed,
        },
        "loss": {
            "kind": "teacher_topk_bidirectional_kl_on_query_positions",
            "top_k": args.top_k,
            "forward_weight": args.forward_weight,
            "reverse_weight": args.reverse_weight,
            "temperature": args.temperature,
        },
        "adapter": adapter_metadata(
            model,
            installed_modules=installed,
            config=lora_config,
        ),
        "optimizer_state_dtypes": {},
        "semantics": training_semantics_metadata(
            mode=args.mode,
            depth=args.depth,
            teacher_kind=args.teacher_kind,
            teacher_source=args.teacher_source,
            quant=quant_config,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            student_suffix_execution=args.student_suffix_execution,
        ),
        "test_v2_used": False,
    }

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            expected_metadata=metadata,
        )
        metadata["resume"] = {
            "checkpoint": str(args.resume),
            "checkpoint_sha256": hashlib.sha256(args.resume.read_bytes()).hexdigest(),
            "restored_step": start_step,
            "optimizer_scheduler_step_restored": True,
        }
    elif args.init_adapter is not None:
        metadata["warm_start"] = load_adapter_warm_start(
            args.init_adapter,
            model=model,
            target_metadata=metadata,
        )
    if is_main(rank):
        atomic_json(args.output_dir / "metadata.json", metadata)
    ddp: torch.nn.Module = core
    if world_size > 1:
        ddp = DistributedDataParallel(
            core,
            device_ids=[local_rank],
            output_device=local_rank,
            # Both training modes execute every suffix attention layer, and
            # the real 8-GPU smoke confirmed all 36 installed LoRA modules
            # participate.  Avoid a redundant autograd traversal each step.
            find_unused_parameters=False,
        )

    consumed_micro_batches = start_step * args.gradient_accumulation
    epoch, offset_in_epoch = divmod(consumed_micro_batches, len(loader))
    if sampler is not None:
        sampler.set_epoch(epoch)
    data_iterator = iter(loader)
    for _ in range(offset_in_epoch):
        next(data_iterator)
    running_persistent_bytes = 0.0
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, args.steps + 1):
        metrics_sum: dict[str, float] = {
            "loss": 0.0,
            "forward_kl": 0.0,
            "reverse_kl": 0.0,
            "persistent_nbytes": 0.0,
        }
        for micro_step in range(args.gradient_accumulation):
            try:
                window = next(data_iterator)
            except StopIteration:
                epoch += 1
                if sampler is not None:
                    sampler.set_epoch(epoch)
                data_iterator = iter(loader)
                window = next(data_iterator)
            document = window.document_ids.to(device, non_blocking=True)
            query = window.query_ids.to(device, non_blocking=True)
            indices = (
                window.teacher_topk_indices.to(device, non_blocking=True)
                if window.teacher_topk_indices is not None
                else None
            )
            logits = (
                window.teacher_topk_logits.to(device, non_blocking=True)
                if window.teacher_topk_logits is not None
                else None
            )
            synchronize = micro_step + 1 == args.gradient_accumulation
            sync_context = (
                contextlib.nullcontext()
                if synchronize or not isinstance(ddp, DistributedDataParallel)
                else ddp.no_sync()
            )
            with sync_context:
                outputs = ddp(document, query, indices, logits)
                loss = outputs["loss"] / args.gradient_accumulation
                loss.backward()
            for name in metrics_sum:
                value = outputs[name]
                metrics_sum[name] += (
                    float(value.detach().item())
                    if isinstance(value, torch.Tensor)
                    else float(value)
                ) / args.gradient_accumulation

        local_gradient_coverage = lora_gradient_coverage(model)
        detached_execution = (
            args.student_suffix_execution == "detached-document-cache"
        )
        native_functional_execution = (
            args.student_suffix_execution == "native-functional-cache"
        )
        audited_cache_execution = detached_execution or native_functional_execution
        local_gradient_coverage["document_cache_contribution_isolated"] = (
            detached_execution
        )
        local_gradient_coverage["gradient_scope"] = (
            "query_continuation_only"
            if detached_execution
            else "document_prefill_and_query_continuation"
        )
        local_gradient_coverage["document_cache_contribution_note"] = (
            "Document suffix prefill ran under no_grad and its cache was "
            "detached+cloned; all observed LoRA gradients come from the full "
            "query continuation."
            if detached_execution
            else local_gradient_coverage["document_cache_contribution_note"]
        )
        gradient_coverage_by_rank: list[Any] = [None] * world_size
        if world_size > 1:
            dist.all_gather_object(gradient_coverage_by_rank, local_gradient_coverage)
        else:
            gradient_coverage_by_rank[0] = local_gradient_coverage
        gradient_hard_gate = all(
            row["module_count"] == len(installed)
            and row["all_modules_have_finite_grad"]
            and (
                not audited_cache_execution
                or row["nonzero_module_count"] == len(installed)
            )
            for row in gradient_coverage_by_rank
        )
        detached_cache_audit_by_rank: list[Any] = [None] * world_size
        local_detached_cache_audit = (
            core.last_detached_cache_audit if audited_cache_execution else None
        )
        if world_size > 1:
            dist.all_gather_object(
                detached_cache_audit_by_rank, local_detached_cache_audit
            )
        else:
            detached_cache_audit_by_rank[0] = local_detached_cache_audit
        cache_position_summary = (
            audited_cache_query_position_gate(detached_cache_audit_by_rank)
            if audited_cache_execution
            else None
        )
        detached_cache_hard_gate = (
            not audited_cache_execution
            or cache_position_summary["hard_gate_passed"]
        )
        metadata["last_gradient_coverage"] = {
            "step": step,
            "hard_gate": (
                "all installed LoRA A/B gradients present and finite on every "
                "rank; audited cached execution additionally requires every module "
                "to have a nonzero query-continuation gradient"
            ),
            "hard_gate_passed": gradient_hard_gate,
            "expected_modules_per_rank": len(installed),
            "by_rank": gradient_coverage_by_rank,
            "document_cache_contribution_isolated": detached_execution,
            "gradient_scope": (
                "query_continuation_only"
                if detached_execution
                else "document_prefill_and_query_continuation"
            ),
        }
        if audited_cache_execution:
            metadata["last_detached_capability"] = {
                "step": step,
                "hard_gate": (
                    "all ranks preserve original document-cache tensor versions, "
                    "satisfy the execution-specific detach/rebind contract, and "
                    "process every configured query position"
                ),
                "hard_gate_passed": detached_cache_hard_gate,
                "query_positions_by_rank": cache_position_summary[
                    "query_positions_by_rank"
                ],
                "minimum_query_positions": cache_position_summary[
                    "minimum_query_positions"
                ],
                "maximum_query_positions": cache_position_summary[
                    "maximum_query_positions"
                ],
                "by_rank": detached_cache_audit_by_rank,
            }
        if is_main(rank):
            atomic_json(args.output_dir / "metadata.json", metadata)
        if not gradient_hard_gate:
            raise RuntimeError("LoRA gradient coverage hard gate failed")
        if not detached_cache_hard_gate:
            raise RuntimeError("detached document-cache immutability hard gate failed")

        torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        running_persistent_bytes += metrics_sum["persistent_nbytes"]

        packed_metrics = torch.tensor(
            [
                metrics_sum["loss"],
                metrics_sum["forward_kl"],
                metrics_sum["reverse_kl"],
                metrics_sum["persistent_nbytes"],
            ],
            device=device,
            dtype=torch.float64,
        )
        if world_size > 1:
            dist.all_reduce(packed_metrics, op=dist.ReduceOp.SUM)
            packed_metrics /= world_size
        if is_main(rank):
            metric_row = {
                "step": step,
                "loss": packed_metrics[0].item(),
                "forward_kl": packed_metrics[1].item(),
                "reverse_kl": packed_metrics[2].item(),
                "mean_persistent_nbytes": packed_metrics[3].item(),
                "learning_rate": scheduler.get_last_lr()[0],
                "gradient_coverage": {
                    "hard_gate_passed": gradient_hard_gate,
                    "expected_modules_per_rank": len(installed),
                    "finite_modules_by_rank": [
                        row["finite_module_count"] for row in gradient_coverage_by_rank
                    ],
                    "nonzero_modules_by_rank": [
                        row["nonzero_module_count"] for row in gradient_coverage_by_rank
                    ],
                    "groups_by_rank": [row["groups"] for row in gradient_coverage_by_rank],
                    "document_cache_contribution_isolated": detached_execution,
                    "gradient_scope": (
                        "query_continuation_only"
                        if detached_execution
                        else "document_prefill_and_query_continuation"
                    ),
                },
            }
            if audited_cache_execution:
                metric_row["detached_cache_capability"] = {
                    "hard_gate_passed": detached_cache_hard_gate,
                    "query_positions_by_rank": cache_position_summary[
                        "query_positions_by_rank"
                    ],
                    "minimum_query_positions": cache_position_summary[
                        "minimum_query_positions"
                    ],
                    "maximum_query_positions": cache_position_summary[
                        "maximum_query_positions"
                    ],
                    "cache_tensor_counts_by_rank": [
                        row["document_cache_tensor_count"]
                        for row in detached_cache_audit_by_rank
                    ],
                }
            append_jsonl(args.output_dir / "metrics.jsonl", metric_row)
            if step % args.log_every == 0 or step == 1:
                print(
                    json.dumps(metric_row),
                    flush=True,
                )

        if is_main(rank) and (step % args.save_every == 0 or step == args.steps):
            metadata["last_step"] = step
            metadata["optimizer_state_dtypes"] = optimizer_state_dtypes(optimizer)
            metadata["mean_local_persistent_nbytes"] = running_persistent_bytes / (
                step - start_step
            )
            save_checkpoint(
                args.output_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                metadata=metadata,
            )
            atomic_json(args.output_dir / "metadata.json", metadata)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
