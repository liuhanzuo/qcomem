from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import torch

from qcomem_lora import (
    assert_replay_adapter_semantics,
    iter_lora_modules,
    load_inference_lora_checkpoint,
    load_lora_state_dict,
    set_lora_enabled,
)
from qcomem_torch import TorchSplitCausalLM
from run_downstream import atomic_json
from run_replay_diagnostic import resolve_config, run_config


ATTRIBUTION_CONFIGS = (
    "adapter-disabled",
    "native-lora-step0",
    "native-lora-step64",
    "native-lora-step128",
)
ACTIVE_STEP = {
    "adapter-disabled": None,
    "native-lora-step0": 0,
    "native-lora-step64": 64,
    "native-lora-step128": 128,
}
STORE_CONFIG = "replay-d7-frozen-static"
FROZEN_LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
EXPECTED_STORE_POLICY = {
    "mode": "replay",
    "depth": 7,
    "residual_bits": 4,
    "attention_bits": 4,
    "linear_bits": 8,
    "cache_layer_bits": (8, 8, 8, 4, 8, 8, 8),
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checkpoint(raw: str) -> tuple[int, Path, str]:
    pieces = raw.split("=", 2)
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("checkpoint must be STEP=PATH=SHA256")
    step = int(pieces[0])
    digest = pieces[2]
    if (
        step not in {0, 64, 128}
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise argparse.ArgumentTypeError("checkpoint step/SHA256 is outside protocol")
    return step, Path(pieces[1]), digest


def checkpoint_payload(
    path: Path, expected_sha256: str, step: int
) -> dict[str, Any]:
    actual = file_sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"checkpoint-{step} SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    semantics = metadata.get("semantics", {})
    if payload.get("format") != "qcomem_suffix_lora_v1" or payload.get("step") != step:
        raise ValueError(f"checkpoint-{step} format/step mismatch")
    if semantics.get("student_suffix_execution_option") != "native-functional-cache":
        raise ValueError(f"checkpoint-{step} is not target-semantic native LoRA")
    if semantics.get("mode") != "quant" or semantics.get("depth") != 7:
        raise ValueError(f"checkpoint-{step} quant/depth semantics drifted")
    if metadata.get("test_v2_used") is not False:
        raise ValueError(f"checkpoint-{step} did not prove test-v2 exclusion")
    if step == 0 and metadata.get("native_initialization_control", {}).get(
        "optimizer_steps"
    ) != 0:
        raise ValueError("step-zero optimizer-step provenance is missing")
    return payload


def validate_store_policy() -> None:
    resolved = resolve_config(STORE_CONFIG)
    actual = {
        "mode": resolved.mode,
        "depth": resolved.depth,
        "residual_bits": resolved.residual_bits,
        "attention_bits": resolved.attention_bits,
        "linear_bits": resolved.linear_bits,
        "cache_layer_bits": resolved.cache_layer_bits,
    }
    if actual != EXPECTED_STORE_POLICY:
        raise RuntimeError(
            f"full-state frozen-static Q4/Q8 policy drifted: {actual}"
        )


def load_validation_slice(
    path: Path, *, expected_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Bind the 4--35 validation artifact and expose only source 6--35."""

    if len(expected_sha256) != 64 or file_sha256(path) != expected_sha256:
        raise RuntimeError("frozen validation SHA256 mismatch")
    parent_rows: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            dataset = row.get("dataset")
            source_index = row.get("_source_index")
            if dataset not in {"qasper", "2wikimqa"}:
                raise RuntimeError(
                    f"validation line {line_number} has an unexpected dataset"
                )
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or not 4 <= source_index <= 35
            ):
                raise RuntimeError(
                    "validation artifact may contain only source indices 4--35"
                )
            if row.get("_source_revision") != FROZEN_LONGBENCH_REVISION:
                raise RuntimeError("validation source revision drifted")
            key = str(dataset), source_index
            if key in keys:
                raise RuntimeError(f"duplicate validation row {key}")
            keys.add(key)
            parent_rows.append(row)
    expected_parent = {
        (dataset, source_index)
        for dataset in ("qasper", "2wikimqa")
        for source_index in range(4, 36)
    }
    if keys != expected_parent or len(parent_rows) != 64:
        raise RuntimeError("validation parent must contain exactly source 4--35")
    selected = [
        row for row in parent_rows if 6 <= int(row["_source_index"]) <= 35
    ]
    return selected, {
        "path": str(path),
        "sha256": expected_sha256,
        "source_revision": FROZEN_LONGBENCH_REVISION,
        "parent_source_index_start": 4,
        "parent_source_index_end": 35,
        "parent_rows": 64,
        "selected_source_index_start": 6,
        "selected_source_index_end": 35,
        "selected_rows": 60,
        "excluded_calibration_source_indices": [4, 5],
        "raw_test_v2_read": False,
    }


def load_and_validate_inputs(
    *,
    data: Path,
    expected_data_sha256: str,
    checkpoints: list[tuple[int, Path, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[int, dict[str, Any]]]:
    validate_store_policy()
    if sorted(step for step, _, _ in checkpoints) != [0, 64, 128]:
        raise ValueError("attribution requires exactly checkpoint steps 0,64,128")
    if len({step for step, _, _ in checkpoints}) != 3:
        raise ValueError("attribution checkpoint steps are duplicated")
    rows, data_audit = load_validation_slice(
        data, expected_sha256=expected_data_sha256
    )
    if data_audit.get("raw_test_v2_read") is not False:
        raise RuntimeError("test-v2 exclusion was not proven")
    payloads: dict[int, dict[str, Any]] = {}
    for step, path, digest in checkpoints:
        payload = checkpoint_payload(path, digest, step)
        assert_replay_adapter_semantics(
            payload["metadata"],
            depth=7,
            residual_bits=4,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=(8, 8, 8, 4, 8, 8, 8),
        )
        payloads[step] = payload
    return rows, data_audit, payloads


def condition_checkpoint_audit(
    condition: str,
    checkpoints: dict[int, tuple[Path, str]],
) -> dict[str, Any]:
    step = ACTIVE_STEP[condition]
    if step is None:
        resident_path, resident_sha = checkpoints[0]
        return {
            "adapter_enabled": False,
            "active_checkpoint_step": None,
            "active_checkpoint": None,
            "active_checkpoint_sha256": None,
            "resident_disabled_adapter_checkpoint_step": 0,
            "resident_disabled_adapter_checkpoint": str(resident_path),
            "resident_disabled_adapter_checkpoint_sha256": resident_sha,
        }
    path, digest = checkpoints[step]
    return {
        "adapter_enabled": True,
        "active_checkpoint_step": step,
        "active_checkpoint": str(path),
        "active_checkpoint_sha256": digest,
        "resident_disabled_adapter_checkpoint_step": None,
        "resident_disabled_adapter_checkpoint": None,
        "resident_disabled_adapter_checkpoint_sha256": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preregistered post-hoc attribution of native LoRA checkpoints on the "
            "already-consumed LongBench validation source 6--35 slice"
        )
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument(
        "--checkpoint", action="append", type=parse_checkpoint, required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_data_sha256 == (
        "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f"
    ):
        raise SystemExit("refusing the frozen LongBench test-v2 artifact")
    if (args.max_input_tokens, args.max_new_tokens, args.group_size) != (4096, 128, 64):
        raise SystemExit("attribution freezes input/new/group-size to 4096/128/64")
    try:
        all_rows, data_audit, payloads = load_and_validate_inputs(
            data=args.data,
            expected_data_sha256=args.expected_data_sha256,
            checkpoints=args.checkpoint,
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(str(error)) from error
    checkpoint_map = {
        step: (path, digest) for step, path, digest in args.checkpoint
    }
    if args.preflight_only:
        result = {
            "status": "preflight_passed",
            "rows": len(all_rows),
            "source_indices": [6, 35],
            "data_sha256": data_audit["sha256"],
            "checkpoint_sha256": {
                str(step): checkpoint_map[step][1] for step in (0, 64, 128)
            },
            "store_policy": EXPECTED_STORE_POLICY,
            "validation_already_consumed": True,
            "selection_or_checkpoint_choice_permitted": False,
            "raw_test_v2_read": False,
        }
        print(json.dumps(result, ensure_ascii=False, default=list))
        return

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required outside preflight-only mode")
    if args.world_size != 8 or not 0 <= args.rank < args.world_size:
        raise SystemExit("formal attribution requires rank 0--7 of world size 8")
    samples = all_rows[args.rank :: args.world_size]
    if not samples:
        raise SystemExit("rank has no validation samples")
    stale = [
        args.run_dir / f"attribution-shard-{args.rank}-{condition}.json"
        for condition in ATTRIBUTION_CONFIGS
        if (args.run_dir / f"attribution-shard-{args.rank}-{condition}.json").exists()
    ]
    if stale:
        raise SystemExit(f"refusing stale attribution shards: {stale}")

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    split = TorchSplitCausalLM(model)
    step0_path, _ = checkpoint_map[0]
    installed_metadata = load_inference_lora_checkpoint(model, split.layers, step0_path)
    if int(installed_metadata["step"]) != 0:
        raise RuntimeError("the resident disabled adapter was not initialized from step 0")
    installed_modules = sum(1 for _ in iter_lora_modules(model))
    if installed_modules != 36:
        raise RuntimeError(f"expected 36 LoRA modules, found {installed_modules}")
    model.eval().cuda()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    model_allocated_bytes = torch.cuda.memory_allocated()

    for condition in ATTRIBUTION_CONFIGS:
        active_step = ACTIVE_STEP[condition]
        if active_step is None:
            set_lora_enabled(model, False)
        else:
            load_lora_state_dict(model, payloads[active_step]["lora"])
            set_lora_enabled(model, True)
        states = [module.enabled for module in iter_lora_modules(model)]
        expected_enabled = active_step is not None
        if len(states) != 36 or any(state != expected_enabled for state in states):
            raise RuntimeError(f"nonuniform LoRA activation for {condition}")

        result = run_config(
            config_name=STORE_CONFIG,
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            model_allocated_bytes=model_allocated_bytes,
            args=args,
        )
        result["store_config"] = result.pop("config")
        result["config"] = condition
        result.update(
            {
                "schema_version": "qcomem-native-lora-checkpoint-attribution-shard-v1",
                "rank": args.rank,
                "world_size": args.world_size,
                "model": str(args.model),
                "data": str(args.data),
                "data_sha256": data_audit["sha256"],
                "data_audit": data_audit,
                "source_index_start": 6,
                "source_index_end": 35,
                "excluded_source_indices": [4, 5],
                "prompt_protocol": "longbench-v1-official",
                "caller": "run_replay_diagnostic.run_config/full_state_replay",
                "decoding": "greedy_argmax",
                "dataset_max_new_tokens": {"qasper": 128, "2wikimqa": 32},
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "group_size": args.group_size,
                "model_load_seconds": model_load_seconds,
                "model_allocated_bytes": model_allocated_bytes,
                "installed_lora_modules": installed_modules,
                "checkpoint": condition_checkpoint_audit(condition, checkpoint_map),
                "checkpoint_ledger": {
                    str(step): {"path": str(checkpoint_map[step][0]), "sha256": checkpoint_map[step][1]}
                    for step in (0, 64, 128)
                },
                "attribution_only": True,
                "validation_already_consumed": True,
                "selection_or_checkpoint_choice_permitted": False,
                "raw_test_v2_read": False,
            }
        )
        destination = (
            args.run_dir / f"attribution-shard-{args.rank}-{condition}.json"
        )
        atomic_json(destination, result)
        print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
