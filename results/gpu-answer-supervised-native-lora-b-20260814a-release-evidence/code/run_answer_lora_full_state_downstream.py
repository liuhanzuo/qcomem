from __future__ import annotations

"""Frozen post-selection full-state downstream evaluation for LoRA B.

This process starts only after the independent official-train heldout record
has selected a checkpoint.  LongBench validation 6--35 is attribution-only and
can neither select nor alter a checkpoint, adapter surface, or store policy.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from qcomem_answer_supervised_lora import (
    EXPECTED_ADAPTER_MODULES,
    EXPECTED_ADAPTER_PARAMETERS,
    FORMAT,
    FROZEN_LONGBENCH_TEST_V2_SHA256,
    AnswerLoRAContractError,
    load_answer_lora_checkpoint,
    load_answer_lora_state_into_installed,
    read_answer_lora_checkpoint,
    sha256_file,
    stable_json,
)
from qcomem_lora import iter_lora_modules, set_lora_enabled
from qcomem_torch import TorchSplitCausalLM
from run_downstream import atomic_json
from run_replay_diagnostic import resolve_config, run_config


FROZEN_LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
CONDITIONS: dict[str, tuple[str, bool, int | None]] = {
    "dense-adapter-disabled-control": ("dense", False, None),
    "q16-adapter-disabled-control": ("replay-d7-layer-q16", False, None),
    "frozen-static-adapter-disabled": ("replay-d7-frozen-static", False, None),
    "frozen-static-answer-lora-step0": ("replay-d7-frozen-static", True, 0),
    "frozen-static-answer-lora-step64": ("replay-d7-frozen-static", True, 64),
    "frozen-static-answer-lora-step128": ("replay-d7-frozen-static", True, 128),
}
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
        "cache_layer_bits": [16, 16, 16, 16, 16, 16, 16],
    },
    "replay-d7-frozen-static": {
        "mode": "replay",
        "depth": 7,
        "residual_bits": 4,
        "attention_bits": 4,
        "linear_bits": 8,
        "cache_layer_bits": [8, 8, 8, 4, 8, 8, 8],
    },
}


def validate_condition_policies() -> None:
    for config_name, expected in EXPECTED_POLICIES.items():
        resolved = resolve_config(config_name)
        actual = {
            "mode": resolved.mode,
            "depth": resolved.depth,
            "residual_bits": resolved.residual_bits,
            "attention_bits": resolved.attention_bits,
            "linear_bits": resolved.linear_bits,
            "cache_layer_bits": (
                list(resolved.cache_layer_bits)
                if resolved.cache_layer_bits is not None
                else None
            ),
        }
        if actual != expected:
            raise AnswerLoRAContractError(
                f"full-state downstream policy drifted for {config_name}: {actual}"
            )


def load_validation_slice(
    path: Path, *, expected_sha256: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if expected_sha256 == FROZEN_LONGBENCH_TEST_V2_SHA256:
        raise AnswerLoRAContractError("refusing frozen LongBench test-v2")
    if len(expected_sha256) != 64 or sha256_file(path) != expected_sha256:
        raise AnswerLoRAContractError("frozen validation SHA256 mismatch")
    rows: list[dict[str, Any]] = []
    keys: set[tuple[str, int]] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            dataset = row.get("dataset")
            source_index = row.get("_source_index")
            if dataset not in {"qasper", "2wikimqa"}:
                raise AnswerLoRAContractError(
                    f"validation line {line_number} has an unexpected dataset"
                )
            if (
                isinstance(source_index, bool)
                or not isinstance(source_index, int)
                or not 4 <= source_index <= 35
            ):
                raise AnswerLoRAContractError(
                    "validation artifact may contain only source indices 4--35"
                )
            if row.get("_source_revision") != FROZEN_LONGBENCH_REVISION:
                raise AnswerLoRAContractError("validation source revision drifted")
            key = str(dataset), source_index
            if key in keys:
                raise AnswerLoRAContractError(f"duplicate validation row {key}")
            keys.add(key)
            rows.append(row)
    expected = {
        (dataset, index)
        for dataset in ("qasper", "2wikimqa")
        for index in range(4, 36)
    }
    if keys != expected or len(rows) != 64:
        raise AnswerLoRAContractError(
            "validation parent must contain exactly both tasks at source 4--35"
        )
    selected = [row for row in rows if 6 <= int(row["_source_index"]) <= 35]
    return selected, {
        "path": str(path),
        "sha256": expected_sha256,
        "source_revision": FROZEN_LONGBENCH_REVISION,
        "parent_source_indices": [4, 35],
        "parent_rows": 64,
        "selected_source_indices": [6, 35],
        "selected_rows": 60,
        "excluded_calibration_source_indices": [4, 5],
        "validation_already_consumed": True,
        "selection_or_checkpoint_choice_permitted": False,
        "raw_test_v2_read": False,
    }


def load_frozen_selection(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": "qcomem_answer_lora_best_checkpoint_v1",
        "selection_source": "independent_official_train_heldout_domain_only",
        "selection_direction": "min",
        "candidate_steps": [0, 64, 128],
        "validation_6_35_used_for_selection": False,
        "test_v2_used": False,
    }
    drift = {
        key: {"expected": expected, "actual": record.get(key)}
        for key, expected in required.items()
        if record.get(key) != expected
    }
    if drift:
        raise AnswerLoRAContractError(f"best-checkpoint selection drifted: {drift}")
    metadata_path = path.with_name("metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    step = record.get("selected_step")
    checkpoint = record.get("checkpoint")
    if step not in {0, 64, 128} or not isinstance(checkpoint, Mapping):
        raise AnswerLoRAContractError("best-checkpoint selection is incomplete")
    if checkpoint.get("step") != step:
        raise AnswerLoRAContractError("selected step and checkpoint record disagree")
    if (
        metadata.get("format") != FORMAT
        or metadata.get("last_step") != 128
        or stable_json(metadata.get("best_checkpoint")) != stable_json(record)
    ):
        raise AnswerLoRAContractError(
            "training metadata and frozen best-checkpoint record disagree"
        )
    raw_suite = metadata.get("checkpoints")
    if not isinstance(raw_suite, Mapping) or set(raw_suite) != {"0", "64", "128"}:
        raise AnswerLoRAContractError("training metadata lacks the full 0/64/128 suite")
    suite: dict[int, dict[str, Any]] = {}
    adapter_configs = []
    for candidate_step in (0, 64, 128):
        item = raw_suite[str(candidate_step)]
        if not isinstance(item, Mapping) or item.get("step") != candidate_step:
            raise AnswerLoRAContractError("checkpoint suite record step drifted")
        checkpoint_path = Path(str(item.get("path")))
        if checkpoint_path.parent.resolve() != path.parent.resolve():
            raise AnswerLoRAContractError("checkpoint suite escaped the artifact dir")
        expected_sha = item.get("sha256")
        if not isinstance(expected_sha, str):
            raise AnswerLoRAContractError("checkpoint suite SHA256 is missing")
        checked = read_answer_lora_checkpoint(
            checkpoint_path,
            expected_sha256=expected_sha,
            expected_step=candidate_step,
        )
        adapter_configs.append(checked["adapter_config"])
        suite[candidate_step] = {
            "step": candidate_step,
            "path": checkpoint_path,
            "sha256": expected_sha,
            "size_bytes": item.get("size_bytes"),
        }
        del checked
    if any(
        stable_json(config) != stable_json(adapter_configs[0])
        for config in adapter_configs[1:]
    ):
        raise AnswerLoRAContractError("checkpoint suite adapter configs differ")
    selected = suite[step]
    if (
        Path(str(checkpoint.get("path"))).resolve() != selected["path"].resolve()
        or checkpoint.get("sha256") != selected["sha256"]
    ):
        raise AnswerLoRAContractError("selected checkpoint differs from full suite ledger")
    return {
        "record": record,
        "record_path": str(path),
        "record_sha256": sha256_file(path),
        "training_metadata_path": str(metadata_path),
        "training_metadata_sha256": sha256_file(metadata_path),
        "selected_step": step,
        "checkpoint_path": selected["path"],
        "checkpoint_sha256": selected["sha256"],
        "checkpoints": suite,
        "adapter_config": adapter_configs[0],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--best-checkpoint", type=Path, required=True)
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
    if (args.max_input_tokens, args.max_new_tokens, args.group_size) != (4096, 128, 64):
        raise SystemExit("downstream freezes input/new/group-size to 4096/128/64")
    try:
        validate_condition_policies()
        # Resolve and verify the official-train-heldout checkpoint first.  The
        # validation artifact is deliberately not opened before this freeze.
        selection = load_frozen_selection(args.best_checkpoint)
        rows, data_audit = load_validation_slice(
            args.data, expected_sha256=args.expected_data_sha256
        )
    except (AnswerLoRAContractError, OSError, KeyError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.preflight_only:
        print(
            json.dumps(
                {
                    "status": "preflight_passed",
                    "rows": len(rows),
                    "conditions": list(CONDITIONS),
                    "selected_step": selection["selected_step"],
                    "checkpoint_sha256": selection["checkpoint_sha256"],
                    "checkpoint_suite_sha256": {
                        str(step): selection["checkpoints"][step]["sha256"]
                        for step in (0, 64, 128)
                    },
                    "all_steps_evaluated_regardless_of_validation": True,
                    "validation_already_consumed": True,
                    "selection_or_checkpoint_choice_permitted": False,
                    "raw_test_v2_read": False,
                },
                sort_keys=True,
            )
        )
        return
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required outside preflight-only mode")
    if args.world_size != 8 or not 0 <= args.rank < 8:
        raise SystemExit("formal downstream requires rank 0--7 of world size 8")
    samples = rows[args.rank :: args.world_size]
    if not samples:
        raise SystemExit("rank has no downstream samples")
    stale = [
        args.run_dir / f"answer-downstream-shard-{args.rank}-{condition}.json"
        for condition in CONDITIONS
        if (args.run_dir / f"answer-downstream-shard-{args.rank}-{condition}.json").exists()
    ]
    if stale:
        raise SystemExit(f"refusing stale answer downstream shards: {stale}")

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
    initial_checkpoint = selection["checkpoints"][0]
    installed = load_answer_lora_checkpoint(
        model,
        split,
        initial_checkpoint["path"],
        expected_sha256=initial_checkpoint["sha256"],
        expected_step=0,
    )
    if len(list(iter_lora_modules(model))) != EXPECTED_ADAPTER_MODULES:
        raise RuntimeError("answer downstream did not install all 156 modules")
    if installed["adapter_config"]["trainable_parameters"] != EXPECTED_ADAPTER_PARAMETERS:
        raise RuntimeError("answer downstream adapter parameter count drifted")
    model.eval().cuda()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    model_allocated_bytes = torch.cuda.memory_allocated()
    resident_step = 0

    for condition, (store_config, enable_adapter, checkpoint_step) in CONDITIONS.items():
        if checkpoint_step is not None and checkpoint_step != resident_step:
            target = selection["checkpoints"][checkpoint_step]
            swapped = load_answer_lora_state_into_installed(
                model,
                target["path"],
                expected_sha256=target["sha256"],
                expected_step=checkpoint_step,
                expected_adapter_config=installed["adapter_config"],
            )
            resident_step = int(swapped["step"])
        set_lora_enabled(model, enable_adapter)
        states = [module.enabled for module in iter_lora_modules(model)]
        if len(states) != EXPECTED_ADAPTER_MODULES or any(
            state is not enable_adapter for state in states
        ):
            raise RuntimeError(f"nonuniform 156-module activation for {condition}")
        result = run_config(
            config_name=store_config,
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
                "schema_version": "qcomem-answer-lora-full-state-shard-v1",
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
                "resident_adapter_modules": EXPECTED_ADAPTER_MODULES,
                "resident_adapter_parameters": EXPECTED_ADAPTER_PARAMETERS,
                "resident_adapter_parameter_bytes": 106_758_144,
                "resident_adapter_memory_scope": (
                    "shared_model_resident_per_process_not_per_document"
                ),
                "adapter_enabled": enable_adapter,
                "resident_checkpoint_step": resident_step,
                "resident_checkpoint_sha256": selection["checkpoints"][resident_step][
                    "sha256"
                ],
                "active_checkpoint_step": (
                    checkpoint_step if enable_adapter else None
                ),
                "active_checkpoint_sha256": (
                    selection["checkpoints"][checkpoint_step]["sha256"]
                    if checkpoint_step is not None
                    else None
                ),
                "selected_checkpoint_step": selection["selected_step"],
                "selected_checkpoint_sha256": selection["checkpoint_sha256"],
                "condition_is_heldout_selected_alias": (
                    checkpoint_step is not None
                    and checkpoint_step == selection["selected_step"]
                ),
                "checkpoint_suite_sha256": {
                    str(step): selection["checkpoints"][step]["sha256"]
                    for step in (0, 64, 128)
                },
                "all_checkpoint_steps_evaluated_unconditionally": True,
                "validation_step_results_may_reselect_checkpoint": False,
                "best_checkpoint_record_sha256": selection["record_sha256"],
                "checkpoint_selection_source": (
                    "independent_official_train_heldout_domain_only"
                ),
                "checkpoint_selection_frozen_before_validation_read": True,
                "adapter_config": installed["adapter_config"],
                "full_state_scope": (
                    "residual plus complete lower-layer KV/recurrent/conv state"
                ),
                "validation_already_consumed": True,
                "selection_or_checkpoint_choice_permitted": False,
                "raw_test_v2_read": False,
            }
        )
        destination = (
            args.run_dir / f"answer-downstream-shard-{args.rank}-{condition}.json"
        )
        atomic_json(destination, result)
        print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
