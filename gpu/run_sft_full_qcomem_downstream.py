from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist

from run_downstream import atomic_json
from run_replay_diagnostic import resolve_config, run_config
from sft_dcp_checkpoint import load_eval_model_only_fp32
from sft_quality_validation import (
    FROZEN_LONGBENCH_REVISION,
    FROZEN_TEXT_PARAMETER_COUNT,
    FROZEN_WORLD_SIZE,
)
from supervised_sft import DenseSupervisedCausalLM


CONFIGS = (
    "dense",
    "replay-d7-layer-q16",
    "replay-d7-layer-q8",
    "replay-d7-frozen-static",
)
EXPECTED_POLICIES = {
    "dense": {
        "mode": "dense",
        "depth": None,
        "residual_bits": None,
        "attention_bits": None,
        "linear_bits": None,
        "cache_layer_bits": None,
    },
    "replay-d7-layer-q16": {
        "mode": "replay",
        "depth": 7,
        "residual_bits": 16,
        "attention_bits": 16,
        "linear_bits": 16,
        "cache_layer_bits": (16, 16, 16, 16, 16, 16, 16),
    },
    "replay-d7-layer-q8": {
        "mode": "replay",
        "depth": 7,
        "residual_bits": 8,
        "attention_bits": 8,
        "linear_bits": 8,
        "cache_layer_bits": (8, 8, 8, 8, 8, 8, 8),
    },
    "replay-d7-frozen-static": {
        "mode": "replay",
        "depth": 7,
        "residual_bits": 4,
        "attention_bits": 4,
        "linear_bits": 8,
        "cache_layer_bits": (8, 8, 8, 4, 8, 8, 8),
    },
}


def load_frozen_validation_slice(
    path: Path, *, expected_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind the 64-row parent artifact, then exclude calibration source 4--5."""

    if len(expected_sha256) != 64:
        raise RuntimeError("expected validation SHA256 is invalid")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"validation parent SHA256 mismatch: {actual_sha256} != {expected_sha256}"
        )

    parent_rows: list[dict[str, Any]] = []
    parent_keys: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise RuntimeError(f"validation line {line_number} is not an object")
            dataset = row.get("dataset")
            source_index = row.get("_source_index")
            if dataset not in {"qasper", "2wikimqa"}:
                raise RuntimeError("validation parent contains an unexpected dataset")
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or not 4 <= source_index <= 35
            ):
                raise RuntimeError(
                    "validation parent permits only calibration 4--5 and evaluation 6--35"
                )
            if row.get("_source_revision") != FROZEN_LONGBENCH_REVISION:
                raise RuntimeError("validation source revision drifted")
            key = dataset, source_index
            if key in parent_keys:
                raise RuntimeError(f"duplicate validation row {key}")
            parent_keys.add(key)
            parent_rows.append(row)
    expected_parent_keys = {
        (dataset, source_index)
        for dataset in ("qasper", "2wikimqa")
        for source_index in range(4, 36)
    }
    if parent_keys != expected_parent_keys:
        raise RuntimeError("validation parent must contain exactly source 4--35")
    selected = [
        row for row in parent_rows if 6 <= int(row["_source_index"]) <= 35
    ]
    selected_keys = {
        (str(row["dataset"]), int(row["_source_index"])) for row in selected
    }
    expected_selected_keys = {
        (dataset, source_index)
        for dataset in ("qasper", "2wikimqa")
        for source_index in range(6, 36)
    }
    if selected_keys != expected_selected_keys or len(selected) != 60:
        raise RuntimeError("evaluation slice must contain exactly source 6--35")
    return selected, {
        "path": str(path),
        "sha256": actual_sha256,
        "source_revision": FROZEN_LONGBENCH_REVISION,
        "parent_source_index_start": 4,
        "parent_source_index_end": 35,
        "parent_rows": len(parent_rows),
        "selected_source_index_start": 6,
        "selected_source_index_end": 35,
        "selected_rows": len(selected),
        "excluded_calibration_source_indices": [4, 5],
        "raw_test_v2_read": False,
    }


def validate_frozen_configs() -> None:
    for name, expected in EXPECTED_POLICIES.items():
        resolved = resolve_config(name)
        actual = {
            "mode": resolved.mode,
            "depth": resolved.depth,
            "residual_bits": resolved.residual_bits,
            "attention_bits": resolved.attention_bits,
            "linear_bits": resolved.linear_bits,
            "cache_layer_bits": resolved.cache_layer_bits,
        }
        if actual != expected:
            raise RuntimeError(
                f"frozen replay policy drifted for {name}: {actual} != {expected}"
            )


def sample_parameter_values(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    samples: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        flat = parameter.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        positions = sorted({0, flat.numel() // 2, flat.numel() - 1})
        samples[name] = flat[positions].cpu().clone()
    return samples


def parameter_sample_audit(
    before: dict[str, torch.Tensor], module: torch.nn.Module
) -> dict[str, Any]:
    digest = hashlib.sha256()
    changed = 0
    elements = 0
    names = 0
    for name, parameter in module.named_parameters():
        if name not in before:
            continue
        flat = parameter.detach().reshape(-1)
        positions = sorted({0, flat.numel() // 2, flat.numel() - 1})
        after = flat[positions].cpu()
        changed += int(torch.count_nonzero(after != before[name]).item())
        elements += after.numel()
        names += 1
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(after.contiguous().view(torch.uint8).numpy().tobytes())
    if names != len(before):
        raise RuntimeError(
            f"parameter sample name coverage drifted: {names} != {len(before)}"
        )
    return {
        "sampled_parameter_names": names,
        "sampled_elements": elements,
        "changed_elements": changed,
        "bf16_sample_sha256": digest.hexdigest(),
    }


def gather_replica_audits(
    local: dict[str, Any], *, require_changed: bool
) -> list[dict[str, Any]]:
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    digests = {item["bf16_sample_sha256"] for item in gathered}
    if len(digests) != 1:
        raise RuntimeError("model parameter samples differ across inference ranks")
    changed = [int(item["changed_elements"]) for item in gathered]
    if require_changed and any(value <= 0 for value in changed):
        raise RuntimeError("SFT DCP produced no sampled BF16 change from base")
    if not require_changed and any(value != 0 for value in changed):
        raise RuntimeError("base inference mutated model parameter samples")
    return gathered


def result_metadata(
    *,
    stage: str,
    rank: int,
    world_size: int,
    model_path: Path,
    data_path: Path,
    data_audit: dict[str, Any],
    model_allocated_bytes: int,
    base_model_load_seconds: float,
    args: argparse.Namespace,
    checkpoint_audit: Optional[dict[str, Any]],
    replica_audits: list[dict[str, Any]],
    dcp_load_seconds: Optional[float],
    peak_after_dcp_bytes: Optional[int],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "model_stage": stage,
        "rank": rank,
        "world_size": world_size,
        "model": str(model_path),
        "data": str(data_path),
        "data_sha256": data_audit["sha256"],
        "data_audit": data_audit,
        "prompt_protocol": "longbench-v1-official",
        "source_index_start": 6,
        "source_index_end": 35,
        "excluded_source_indices": [4, 5],
        "base_model_load_seconds": base_model_load_seconds,
        "model_allocated_bytes": model_allocated_bytes,
        "max_input_tokens": args.max_input_tokens,
        "max_new_tokens": args.max_new_tokens,
        "group_size": args.group_size,
        "replicated_destination_dtype": "torch.bfloat16",
        "replica_sample_audits": replica_audits,
        "raw_test_v2_read": False,
        "full_lower_state_qcomem": True,
    }
    if checkpoint_audit is None:
        metadata.update(
            {
                "checkpoint": None,
                "checkpoint_manifest_sha256": None,
                "checkpoint_payload_directory_sha256": None,
                "checkpoint_step": 0,
                "checkpoint_contract": "base_huggingface_bf16_before_dcp_load",
                "checkpoint_payload_integrity_verified": False,
                "dcp_load_seconds": None,
                "peak_after_dcp_load_bytes": None,
            }
        )
    else:
        metadata.update(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_manifest_sha256": checkpoint_audit[
                    "checkpoint_manifest_sha256"
                ],
                "checkpoint_payload_directory_sha256": checkpoint_audit[
                    "payload_directory_sha256"
                ],
                "checkpoint_step": checkpoint_audit["step"],
                "checkpoint_contract": checkpoint_audit["contract"],
                "checkpoint_payload_integrity_verified": checkpoint_audit[
                    "payload_integrity_verified"
                ],
                "dcp_load_seconds": dcp_load_seconds,
                "peak_after_dcp_load_bytes": peak_after_dcp_bytes,
            }
        )
    return metadata


def run_stage(
    *,
    stage: str,
    model: torch.nn.Module,
    tokenizer: Any,
    samples: list[dict[str, Any]],
    rank: int,
    world_size: int,
    model_allocated_bytes: int,
    base_model_load_seconds: float,
    args: argparse.Namespace,
    data_audit: dict[str, Any],
    checkpoint_audit: Optional[dict[str, Any]],
    replica_audits: list[dict[str, Any]],
    dcp_load_seconds: Optional[float],
    peak_after_dcp_bytes: Optional[int],
) -> None:
    common = result_metadata(
        stage=stage,
        rank=rank,
        world_size=world_size,
        model_path=args.model,
        data_path=args.data,
        data_audit=data_audit,
        model_allocated_bytes=model_allocated_bytes,
        base_model_load_seconds=base_model_load_seconds,
        args=args,
        checkpoint_audit=checkpoint_audit,
        replica_audits=replica_audits,
        dcp_load_seconds=dcp_load_seconds,
        peak_after_dcp_bytes=peak_after_dcp_bytes,
    )
    for config_name in CONFIGS:
        destination = args.run_dir / f"{stage}-shard-{rank}-{config_name}.json"
        if destination.exists():
            raise RuntimeError(f"refusing stale result shard {destination}")
        result = run_config(
            config_name=config_name,
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            model_allocated_bytes=model_allocated_bytes,
            args=args,
        )
        result.update(common)
        atomic_json(destination, result)
        print(f"SAVED {destination}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate base and selected dense-SFT models with dense and complete "
            "lower-state Q-CoMem on the frozen LongBench validation slice"
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-manifest-sha256", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != FROZEN_WORLD_SIZE:
        raise SystemExit(f"unified evaluation requires {FROZEN_WORLD_SIZE} ranks")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("unified evaluation requires native-BF16 CUDA GPUs")
    if args.group_size != 64:
        raise SystemExit("unified evaluation freezes quantization group size to 64")
    if (args.max_input_tokens, args.max_new_tokens) != (4096, 128):
        raise SystemExit("unified evaluation freezes max input/new tokens to 4096/128")
    if len(args.expected_checkpoint_manifest_sha256) != 64:
        raise SystemExit("checkpoint manifest SHA256 is invalid")
    validate_frozen_configs()

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    rows, data_audit = load_frozen_validation_slice(
        args.data, expected_sha256=args.expected_data_sha256
    )
    samples = rows[rank::world_size]
    if not samples:
        raise RuntimeError("rank received no frozen validation rows")

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    model.eval().to(device)
    torch.cuda.synchronize(device)
    base_model_load_seconds = time.perf_counter() - load_started
    base_allocated_bytes = torch.cuda.memory_allocated(device)

    core = DenseSupervisedCausalLM.from_conditional_generation(model)
    parameter_count = sum(parameter.numel() for parameter in core.parameters())
    if parameter_count != FROZEN_TEXT_PARAMETER_COUNT:
        raise RuntimeError(
            f"text parameter count drifted: {parameter_count:,} != "
            f"{FROZEN_TEXT_PARAMETER_COUNT:,}"
        )
    if {parameter.dtype for parameter in core.parameters()} != {torch.bfloat16}:
        raise RuntimeError("replicated inference destination must be entirely BF16")
    base_samples = sample_parameter_values(core)
    initial_audit = parameter_sample_audit(base_samples, core)
    initial_replica_audits = gather_replica_audits(
        initial_audit, require_changed=False
    )

    run_stage(
        stage="base",
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        rank=rank,
        world_size=world_size,
        model_allocated_bytes=base_allocated_bytes,
        base_model_load_seconds=base_model_load_seconds,
        args=args,
        data_audit=data_audit,
        checkpoint_audit=None,
        replica_audits=initial_replica_audits,
        dcp_load_seconds=None,
        peak_after_dcp_bytes=None,
    )
    post_base_audit = parameter_sample_audit(base_samples, core)
    post_base_replica_audits = gather_replica_audits(
        post_base_audit, require_changed=False
    )
    if post_base_replica_audits != initial_replica_audits:
        raise RuntimeError("base parameter audit changed during inference")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    dist.barrier()
    dcp_started = time.perf_counter()
    checkpoint_audit = load_eval_model_only_fp32(
        core,
        args.checkpoint,
        expected_manifest_sha256=args.expected_checkpoint_manifest_sha256,
    )
    dist.barrier()
    torch.cuda.synchronize(device)
    dcp_load_seconds = time.perf_counter() - dcp_started
    sft_sample_audit = parameter_sample_audit(base_samples, core)
    sft_replica_audits = gather_replica_audits(
        sft_sample_audit, require_changed=True
    )
    if checkpoint_audit.get("global_parameter_count") != FROZEN_TEXT_PARAMETER_COUNT:
        raise RuntimeError("checkpoint manifest text parameter count drifted")
    del base_samples, core
    model.eval()
    sft_allocated_bytes = torch.cuda.memory_allocated(device)
    peak_after_dcp_bytes = torch.cuda.max_memory_allocated(device)
    if not math.isfinite(dcp_load_seconds) or dcp_load_seconds <= 0:
        raise RuntimeError("DCP load timing is invalid")

    run_stage(
        stage="sft",
        model=model,
        tokenizer=tokenizer,
        samples=samples,
        rank=rank,
        world_size=world_size,
        model_allocated_bytes=sft_allocated_bytes,
        base_model_load_seconds=base_model_load_seconds,
        args=args,
        data_audit=data_audit,
        checkpoint_audit=checkpoint_audit,
        replica_audits=sft_replica_audits,
        dcp_load_seconds=dcp_load_seconds,
        peak_after_dcp_bytes=peak_after_dcp_bytes,
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
