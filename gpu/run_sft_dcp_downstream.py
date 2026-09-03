from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from run_downstream import atomic_json, run_config
from sft_dcp_checkpoint import load_eval_model_only_fp32
from sft_quality_validation import (
    FROZEN_TEXT_PARAMETER_COUNT,
    FROZEN_WORLD_SIZE,
    validate_longbench_validation_rows,
)
from supervised_sft import DenseSupervisedCausalLM


CONFIGS = ("dense",)


def _sample_parameter_values(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    samples: dict[str, torch.Tensor] = {}
    for name, parameter in module.named_parameters():
        flat = parameter.detach().reshape(-1)
        if flat.numel() == 0:
            continue
        positions = sorted({0, flat.numel() // 2, flat.numel() - 1})
        samples[name] = flat[positions].cpu().clone()
    return samples


def _sample_audit(
    before: dict[str, torch.Tensor], module: torch.nn.Module
) -> dict[str, Any]:
    digest = hashlib.sha256()
    changed = 0
    elements = 0
    for name, parameter in module.named_parameters():
        if name not in before:
            continue
        flat = parameter.detach().reshape(-1)
        positions = sorted({0, flat.numel() // 2, flat.numel() - 1})
        after = flat[positions].cpu()
        changed += int(torch.count_nonzero(after != before[name]).item())
        elements += after.numel()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(after.contiguous().view(torch.uint8).numpy().tobytes())
    return {
        "sampled_elements": elements,
        "changed_from_base_elements": changed,
        "bf16_sample_sha256": digest.hexdigest(),
    }


def _require_equal_replica_audits(local: dict[str, Any]) -> list[dict[str, Any]]:
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local)
    digests = {item["bf16_sample_sha256"] for item in gathered}
    if len(digests) != 1:
        raise RuntimeError("BF16 SFT checkpoint samples differ across inference ranks")
    if not all(item["changed_from_base_elements"] > 0 for item in gathered):
        raise RuntimeError("SFT DCP load has no BF16-visible sampled change from base")
    return gathered


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the selected dense-SFT DCP as eight replicated BF16 models on "
            "the frozen LongBench validation source 6--35"
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
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=0)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size != FROZEN_WORLD_SIZE:
        raise SystemExit(f"SFT downstream evaluation requires {FROZEN_WORLD_SIZE} ranks")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("SFT downstream evaluation requires native-BF16 CUDA GPUs")
    if (args.chunk_size, args.overlap, args.group_size) != (512, 0, 64):
        raise SystemExit("unified comparison freezes chunk/overlap/group to 512/0/64")
    if (args.max_input_tokens, args.max_new_tokens) != (4096, 128):
        raise SystemExit("unified comparison freezes max input/new tokens to 4096/128")
    if len(args.expected_checkpoint_manifest_sha256) != 64:
        raise SystemExit("checkpoint manifest SHA256 is invalid")

    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(hours=2))
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    rows, data_audit = validate_longbench_validation_rows(
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
    base_load_seconds = time.perf_counter() - load_started
    base_allocated_bytes = torch.cuda.memory_allocated(device)

    core = DenseSupervisedCausalLM.from_conditional_generation(model)
    parameter_count = sum(parameter.numel() for parameter in core.parameters())
    if parameter_count != FROZEN_TEXT_PARAMETER_COUNT:
        raise RuntimeError(
            f"text parameter count drifted: {parameter_count:,} != "
            f"{FROZEN_TEXT_PARAMETER_COUNT:,}"
        )
    dtypes = {parameter.dtype for parameter in core.parameters()}
    if dtypes != {torch.bfloat16}:
        raise RuntimeError(f"replicated inference destination must be BF16, got {dtypes}")
    before = _sample_parameter_values(core)
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
    sample_audit = _sample_audit(before, core)
    replica_audits = _require_equal_replica_audits(sample_audit)
    if checkpoint_audit.get("global_parameter_count") != FROZEN_TEXT_PARAMETER_COUNT:
        raise RuntimeError("checkpoint manifest text parameter count drifted")
    del before, core
    model.eval()
    model_allocated_bytes = torch.cuda.memory_allocated(device)
    peak_after_dcp_bytes = torch.cuda.max_memory_allocated(device)
    if not math.isfinite(dcp_load_seconds) or dcp_load_seconds <= 0:
        raise RuntimeError("DCP load timing is invalid")

    for config_name in CONFIGS:
        destination = args.run_dir / f"sft-shard-{rank}-{config_name}.json"
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
        result.update(
            {
                "rank": rank,
                "world_size": world_size,
                "model": str(args.model),
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
                "replicated_destination_dtype": "torch.bfloat16",
                "replica_sample_audits": replica_audits,
                "data": str(args.data),
                "data_sha256": data_audit["sha256"],
                "data_audit": data_audit,
                "prompt_protocol": "longbench-v1-official",
                "base_model_load_seconds": base_load_seconds,
                "dcp_load_seconds": dcp_load_seconds,
                "base_model_allocated_bytes": base_allocated_bytes,
                "model_allocated_bytes": model_allocated_bytes,
                "peak_after_dcp_load_bytes": peak_after_dcp_bytes,
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "group_size": args.group_size,
                "raw_test_v2_read": False,
                "static_store_semantics": "dense_no_comem_store",
            }
        )
        atomic_json(destination, result)
        print(f"SAVED {destination}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
