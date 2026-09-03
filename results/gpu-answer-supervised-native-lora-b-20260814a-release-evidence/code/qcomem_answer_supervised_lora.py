from __future__ import annotations

"""Answer-supervised native-cache LoRA primitives.

This module is deliberately independent from the historical query-KL trainer.
It consumes the frozen deployment-oriented official-train split, predicts only
assistant answer/EOS tokens, and keeps every reduction equal per example.
"""

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from deployment_aware_sft import DeploymentExample, IGNORE_INDEX, log1mexp
from qcomem_lora import (
    LoRAConfig,
    LoRALinear,
    ReplayQuantConfig,
    find_suffix_lora_targets,
    install_suffix_lora,
    iter_lora_modules,
    load_lora_state_dict,
    lora_state_dict,
    quant_student_suffix_hidden,
    set_lora_enabled,
)
from qcomem_torch import TorchSplitCausalLM


FORMAT = "qcomem_answer_supervised_native_lora_v1"
TEACHER_FORMAT = "qcomem_answer_teacher_target_shard_v1"
TEACHER_MANIFEST_FORMAT = "qcomem_answer_teacher_target_manifest_v1"
DEPTH = 7
WORLD_SIZE = 8
STEPS = 128
EVALUATION_STEPS = (0, 64, 128)
CHECKPOINT_STEPS = EVALUATION_STEPS
FULL_ATTENTION_TARGET_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
GDN_TARGET_SUFFIXES = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)
TARGET_SUFFIXES = FULL_ATTENTION_TARGET_SUFFIXES + GDN_TARGET_SUFFIXES
MLP_EXCLUDED_SUFFIXES = ("gate_proj", "up_proj", "down_proj")
EXPECTED_SUFFIX_FULL_ATTENTION_LAYERS = 9
EXPECTED_SUFFIX_GDN_LAYERS = 24
EXPECTED_FULL_ATTENTION_MODULES = 36
EXPECTED_GDN_MODULES = 120
EXPECTED_ADAPTER_MODULES = 156
EXPECTED_ADAPTER_PARAMETER_TENSORS = 312
EXPECTED_ADAPTER_PARAMETERS = 26_689_536
EXPECTED_FULL_ATTENTION_PARAMETERS = 6_193_152
EXPECTED_GDN_PARAMETERS = 20_496_384
MAX_ADAPTER_PARAMETERS = 27_000_000
FROZEN_LONGBENCH_TEST_V2_SHA256 = (
    "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f"
)
FROZEN_ANSWER_STORE = {
    "mode": "quant",
    "depth": DEPTH,
    "student_suffix_execution": "native-functional-cache",
    "residual_bits": 4,
    "attention_bits": 4,
    "linear_bits": 8,
    "cache_layer_bits": [8, 8, 8, 4, 8, 8, 8],
    "group_size": 64,
    "weights_quantized": False,
}


class AnswerLoRAContractError(ValueError):
    pass


def stable_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AnswerLoRAContractError(f"{label} must be one lowercase SHA256")
    return value


def reject_longbench_path_or_digest(path: Path, expected_sha256: str) -> None:
    """Fail closed before opening any LongBench validation/test artifact."""

    normalized = str(path).lower().replace("_", "-")
    if "longbench" in normalized or "test-v2" in normalized:
        raise AnswerLoRAContractError(
            "answer-supervised LoRA accepts only frozen official-train derivatives"
        )
    if require_sha256(expected_sha256, "data SHA256") == FROZEN_LONGBENCH_TEST_V2_SHA256:
        raise AnswerLoRAContractError("refusing frozen LongBench test-v2 by SHA256")


def target_start(labels: torch.Tensor) -> int:
    if labels.ndim != 1 or labels.numel() < 2:
        raise AnswerLoRAContractError("labels must be one sequence with at least two tokens")
    active = torch.nonzero(labels != IGNORE_INDEX, as_tuple=False).flatten()
    if active.numel() < 2:
        raise AnswerLoRAContractError("answer supervision must contain answer plus EOS")
    start = int(active[0].item())
    if not torch.equal(active.cpu(), torch.arange(start, labels.numel())):
        raise AnswerLoRAContractError("labels must be one contiguous target suffix")
    return start


@dataclass(frozen=True)
class AnswerBoundary:
    document_ids: torch.Tensor
    query_ids: torch.Tensor
    answer_ids: torch.Tensor
    continuation_input_ids: torch.Tensor
    target_ids: torch.Tensor
    kind: str

    @property
    def answer_positions(self) -> int:
        return int(self.target_ids.numel())


def answer_boundary(
    example: DeploymentExample,
    *,
    raw_row: Mapping[str, Any],
) -> AnswerBoundary:
    """Create the causal document -> query+answer training boundary.

    Only domain rows have an exact frozen reusable-document boundary.  Tulu
    replay rows are rejected rather than assigned a synthetic split.
    """

    start = target_start(example.labels)
    prompt = example.input_ids[:start]
    answer = example.input_ids[start:]
    if not torch.equal(answer, example.labels[start:]):
        raise AnswerLoRAContractError("answer input IDs and active labels differ")
    if prompt.numel() < 2:
        raise AnswerLoRAContractError("prompt must permit non-empty document and query")
    if example.stratum != "domain":
        raise AnswerLoRAContractError(
            "non-domain replay has no unambiguous document/query boundary and is "
            "excluded from the formal answer-supervised native-cache run"
        )
    boundary = raw_row.get("deployment_boundary")
    if not isinstance(boundary, Mapping) or boundary.get("applicable") is not True:
        raise AnswerLoRAContractError("domain row lacks its exact deployment boundary")
    document = torch.tensor(boundary.get("document_input_ids"), dtype=torch.long)
    query = torch.tensor(boundary.get("query_input_ids"), dtype=torch.long)
    if not torch.equal(torch.cat((document, query)), prompt):
        raise AnswerLoRAContractError("domain document/query boundary drifted")
    kind = "frozen_exact_domain_document_query"
    if document.numel() < 1 or query.numel() < 1:
        raise AnswerLoRAContractError("document and query must both be non-empty")
    # The input at the final query position predicts answer[0].  Subsequent
    # answer inputs predict the next answer token, ending with EOS.
    continuation = torch.cat((query, answer[:-1]))
    if continuation.numel() < answer.numel():
        raise AnswerLoRAContractError("continuation cannot expose every answer target")
    return AnswerBoundary(
        document_ids=document,
        query_ids=query,
        answer_ids=answer,
        continuation_input_ids=continuation,
        target_ids=answer,
        kind=kind,
    )


def balance_group(stratum: str, dataset: str) -> str:
    if stratum == "domain":
        if dataset not in {"qasper", "2wikimqa"}:
            raise AnswerLoRAContractError("domain balance requires QASPER or 2WikiMQA")
        return f"domain/{dataset}"
    raise AnswerLoRAContractError(
        f"formal answer-supervised native-cache training is domain-only, got {stratum!r}"
    )


BALANCE_MASS = {
    "domain/qasper": 0.5,
    "domain/2wikimqa": 0.5,
}


def example_balance_weights(
    examples: Sequence[DeploymentExample],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Return mean-one weights with equal stratum and domain-task mass."""

    counts = Counter(balance_group(row.stratum, row.dataset) for row in examples)
    if set(counts) != set(BALANCE_MASS) or any(counts[group] < 1 for group in BALANCE_MASS):
        raise AnswerLoRAContractError(f"balance groups are incomplete: {dict(counts)}")
    total = len(examples)
    by_id = {
        row.example_id: total * BALANCE_MASS[balance_group(row.stratum, row.dataset)]
        / counts[balance_group(row.stratum, row.dataset)]
        for row in examples
    }
    if len(by_id) != total:
        raise AnswerLoRAContractError("balance calculation found duplicate example IDs")
    group_mass = {
        group: sum(
            by_id[row.example_id]
            for row in examples
            if balance_group(row.stratum, row.dataset) == group
        )
        / total
        for group in BALANCE_MASS
    }
    if not math.isclose(sum(by_id.values()), total, rel_tol=0.0, abs_tol=1e-9):
        raise AnswerLoRAContractError("example weights do not have mean one")
    if any(
        not math.isclose(group_mass[group], target, rel_tol=0.0, abs_tol=1e-12)
        for group, target in BALANCE_MASS.items()
    ):
        raise AnswerLoRAContractError("group mass differs from the frozen contract")
    return by_id, {
        "kind": "domain_task_equal_example_mean_v1",
        "target_token_weighting_used": False,
        "stratum_mass": {"domain": 1.0},
        "domain_task_mass": {"qasper": 0.5, "2wikimqa": 0.5},
        "group_counts": dict(sorted(counts.items())),
        "group_mass": group_mass,
        "minimum_example_weight": min(by_id.values()),
        "maximum_example_weight": max(by_id.values()),
        "mean_example_weight": sum(by_id.values()) / total,
    }


def balanced_domain_schedule(
    examples: Sequence[DeploymentExample],
    *,
    steps: int,
    world_size: int,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    """Build a deterministic 4+4 task schedule with near-equal reuse.

    The first mini-batch takes the longest examples from each task to exercise
    the memory gate.  Later cycles use independent seeded permutations.  Every
    global step is exactly task-balanced and per-example exposure differs by at
    most one within either task.
    """

    if world_size != WORLD_SIZE or world_size % 2:
        raise AnswerLoRAContractError("formal schedule requires eight even ranks")
    if steps != STEPS:
        raise AnswerLoRAContractError(f"formal schedule requires {STEPS} steps")
    pools: dict[str, list[DeploymentExample]] = {
        dataset: [row for row in examples if row.dataset == dataset]
        for dataset in ("qasper", "2wikimqa")
    }
    if any(not pool for pool in pools.values()) or sum(map(len, pools.values())) != len(examples):
        raise AnswerLoRAContractError("domain schedule contains an unsupported dataset")
    per_task = steps * (world_size // 2)

    def task_stream(dataset: str, pool: list[DeploymentExample]) -> list[str]:
        longest = sorted(pool, key=lambda row: (-row.sequence_tokens, row.example_id))
        head = longest[: world_size // 2]
        head_ids = {row.example_id for row in head}
        remainder = [row for row in pool if row.example_id not in head_ids]
        generator = torch.Generator().manual_seed(
            seed + (0 if dataset == "qasper" else 1_000_003)
        )
        order = torch.randperm(len(remainder), generator=generator).tolist()
        result = [row.example_id for row in head]
        result.extend(remainder[index].example_id for index in order)
        cycle = 1
        while len(result) < per_task:
            generator = torch.Generator().manual_seed(
                seed + (0 if dataset == "qasper" else 1_000_003) + cycle
            )
            order = torch.randperm(len(pool), generator=generator).tolist()
            result.extend(pool[index].example_id for index in order)
            cycle += 1
        return result[:per_task]

    streams = {dataset: task_stream(dataset, pool) for dataset, pool in pools.items()}
    schedule = []
    for step in range(steps):
        start = step * (world_size // 2)
        end = start + world_size // 2
        schedule.extend(streams["qasper"][start:end])
        schedule.extend(streams["2wikimqa"][start:end])
    counts = Counter(schedule)
    dataset_counts = {
        dataset: Counter(streams[dataset]) for dataset in streams
    }
    checks = {
        "positions": len(schedule) == steps * world_size,
        "unique_source_examples": set(schedule) == {row.example_id for row in examples},
        "four_plus_four_every_step": all(
            Counter(
                next(row.dataset for row in examples if row.example_id == example_id)
                for example_id in schedule[start : start + world_size]
            )
            == {"qasper": 4, "2wikimqa": 4}
            for start in range(0, len(schedule), world_size)
        ),
        "within_task_exposure_spread_at_most_one": all(
            max(values.values()) - min(values.values()) <= 1
            for values in dataset_counts.values()
        ),
        "first_step_exercises_4096": max(
            next(row.sequence_tokens for row in examples if row.example_id == example_id)
            for example_id in schedule[:world_size]
        )
        == 4096,
    }
    if not all(checks.values()):
        raise AnswerLoRAContractError(f"balanced schedule gate failed: {checks}")
    return schedule, {
        "kind": "domain_task_4_plus_4_cyclic_example_balanced_v1",
        "steps": steps,
        "world_size": world_size,
        "positions": len(schedule),
        "source_examples": len(examples),
        "task_positions": {dataset: len(stream) for dataset, stream in streams.items()},
        "source_task_counts": {dataset: len(pool) for dataset, pool in pools.items()},
        "per_example_exposure": {
            dataset: {
                "min": min(values.values()),
                "max": max(values.values()),
            }
            for dataset, values in dataset_counts.items()
        },
        "ordered_example_id_sha256": hashlib.sha256(
            "\n".join(schedule).encode("ascii")
        ).hexdigest(),
        "first_step_example_ids": schedule[:world_size],
        "rank_assignment_per_step": {
            "ranks_0_1_2_3": "qasper_one_example_per_rank",
            "ranks_4_5_6_7": "2wikimqa_one_example_per_rank",
        },
        "checks": checks,
        "target_token_weighting_used": False,
    }


def classify_adapter_module(name: str, layer_types: Sequence[str]) -> str:
    components = name.split(".")
    try:
        layer_marker = components.index("layers")
        layer_index = int(components[layer_marker + 1])
    except (ValueError, IndexError) as error:
        raise AnswerLoRAContractError(f"cannot locate decoder layer in {name}") from error
    if layer_index < DEPTH or layer_index >= len(layer_types):
        raise AnswerLoRAContractError(f"adapter escaped suffix depth in {name}")
    leaf = components[-1]
    kind = layer_types[layer_index]
    if kind == "full_attention" and leaf in FULL_ATTENTION_TARGET_SUFFIXES:
        return "full_attention"
    if kind == "linear_attention" and leaf in GDN_TARGET_SUFFIXES:
        return "gdn"
    if leaf in MLP_EXCLUDED_SUFFIXES or ".mlp." in name:
        return "mlp"
    raise AnswerLoRAContractError(f"unexpected adapter target {name} ({kind=})")


def audit_adapter_surface(
    installed: Sequence[str], layer_types: Sequence[str]
) -> dict[str, Any]:
    if len(layer_types) != 40:
        raise AnswerLoRAContractError("formal Qwen3.5 layer-type plan must have 40 layers")
    counts = Counter(classify_adapter_module(name, layer_types) for name in installed)
    suffix_full_layers = {
        int(name.split("layers.", 1)[1].split(".", 1)[0])
        for name in installed
        if classify_adapter_module(name, layer_types) == "full_attention"
    }
    suffix_gdn_layers = {
        int(name.split("layers.", 1)[1].split(".", 1)[0])
        for name in installed
        if classify_adapter_module(name, layer_types) == "gdn"
    }
    expected = {
        "full_attention": EXPECTED_FULL_ATTENTION_MODULES,
        "gdn": EXPECTED_GDN_MODULES,
    }
    checks = {
        "module_counts": dict(counts) == expected,
        "total_modules": len(installed) == EXPECTED_ADAPTER_MODULES,
        "suffix_full_attention_layers": len(suffix_full_layers)
        == EXPECTED_SUFFIX_FULL_ATTENTION_LAYERS,
        "suffix_gdn_layers": len(suffix_gdn_layers) == EXPECTED_SUFFIX_GDN_LAYERS,
        "mlp_modules": counts.get("mlp", 0) == 0,
        "unique_modules": len(set(installed)) == len(installed),
    }
    if not all(checks.values()):
        raise AnswerLoRAContractError(
            f"adapter surface gate failed: counts={dict(counts)}, checks={checks}"
        )
    return {
        "status": "passed",
        "checks": checks,
        "depth": DEPTH,
        "installed_modules": list(installed),
        "module_counts": dict(counts),
        "suffix_full_attention_layer_indices": sorted(suffix_full_layers),
        "suffix_gdn_layer_indices": sorted(suffix_gdn_layers),
        "covered_projection_suffixes": {
            "full_attention": list(FULL_ATTENTION_TARGET_SUFFIXES),
            "gdn": list(GDN_TARGET_SUFFIXES),
        },
        "mlp": {
            "covered": False,
            "installed_modules": 0,
            "excluded_projection_suffixes": list(MLP_EXCLUDED_SUFFIXES),
            "claim": "explicit_ablation_not_part_of_this_run",
        },
        "claim_boundary": (
            "suffix full-attention plus every GDN projection; MLP/expert adapters "
            "are excluded, so this is not an all-suffix adapter"
        ),
    }


def install_and_audit_adapters(
    model: nn.Module,
    split: TorchSplitCausalLM,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    initialization_seed: int,
) -> tuple[list[str], dict[str, Any]]:
    torch.manual_seed(initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(initialization_seed)
    config = LoRAConfig(
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_suffixes=TARGET_SUFFIXES,
    )
    candidates = find_suffix_lora_targets(
        split.layers, depth=DEPTH, target_suffixes=TARGET_SUFFIXES
    )
    installed = install_suffix_lora(
        model, split.layers, depth=DEPTH, config=config
    )
    if installed != candidates:
        raise AnswerLoRAContractError("adapter installed list differs from preflight")
    layer_types = tuple(split.config.layer_types)
    audit = audit_adapter_surface(installed, layer_types)
    return installed, audit


def answer_adapter_config(
    model: nn.Module,
    *,
    installed: Sequence[str],
    surface_audit: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the complete, frozen inference-installation contract.

    Checkpoints are intentionally self-describing: downstream code must not
    silently fall back to the historical 36-module full-attention surface.
    """

    state = lora_state_dict(model)
    parameter_count = sum(int(value.numel()) for value in state.values())
    checks = {
        "module_count": len(installed) == EXPECTED_ADAPTER_MODULES,
        "parameter_tensor_count": len(state) == EXPECTED_ADAPTER_PARAMETER_TENSORS,
        "parameter_count": parameter_count == EXPECTED_ADAPTER_PARAMETERS,
        "parameter_dtype_fp32": all(value.dtype == torch.float32 for value in state.values()),
        "surface_status": surface_audit.get("status") == "passed",
        "installed_list_matches_surface": list(installed)
        == list(surface_audit.get("installed_modules", [])),
    }
    if not all(checks.values()):
        raise AnswerLoRAContractError(
            f"cannot freeze an invalid answer adapter config: {checks}"
        )
    return {
        "schema_version": "qcomem_answer_adapter_config_v1",
        "depth": DEPTH,
        "rank": 32,
        "alpha": 64.0,
        "dropout": 0.0,
        "target_suffixes": list(TARGET_SUFFIXES),
        "installed_modules": list(installed),
        "installed_module_count": EXPECTED_ADAPTER_MODULES,
        "parameter_tensor_count": EXPECTED_ADAPTER_PARAMETER_TENSORS,
        "trainable_parameters": EXPECTED_ADAPTER_PARAMETERS,
        "parameter_dtype": "float32",
        "module_counts": {"full_attention": 36, "gdn": 120},
        "mlp_or_expert_covered": False,
        "trained_store": dict(FROZEN_ANSWER_STORE),
        "state_key_sha256": hashlib.sha256(
            "\n".join(sorted(state)).encode("utf-8")
        ).hexdigest(),
        "checks": checks,
    }


def read_answer_lora_checkpoint(
    path: Path,
    *,
    expected_sha256: str,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Read and validate one self-describing answer checkpoint on CPU."""

    expected_sha256 = require_sha256(expected_sha256, "answer checkpoint SHA256")
    if sha256_file(path) != expected_sha256:
        raise AnswerLoRAContractError("answer checkpoint SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != FORMAT:
        raise AnswerLoRAContractError("unsupported answer LoRA checkpoint format")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step not in CHECKPOINT_STEPS:
        raise AnswerLoRAContractError("answer checkpoint step is outside 0/64/128")
    if expected_step is not None and step != expected_step:
        raise AnswerLoRAContractError("answer checkpoint step differs from selection ledger")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise AnswerLoRAContractError("answer checkpoint metadata is missing")
    governance = metadata.get("governance")
    if not isinstance(governance, Mapping) or any(
        governance.get(key) is not False
        for key in (
            "validation_6_35_used_for_tuning",
            "test_v2_used",
            "raw_longbench_validation_or_test_read",
        )
    ):
        raise AnswerLoRAContractError("answer checkpoint governance is not fail-closed")
    frozen = metadata.get("adapter_config")
    if not isinstance(frozen, Mapping):
        raise AnswerLoRAContractError("answer checkpoint lacks adapter_config")
    expected_scalars = {
        "schema_version": "qcomem_answer_adapter_config_v1",
        "depth": DEPTH,
        "rank": 32,
        "alpha": 64.0,
        "dropout": 0.0,
        "target_suffixes": list(TARGET_SUFFIXES),
        "installed_module_count": EXPECTED_ADAPTER_MODULES,
        "parameter_tensor_count": EXPECTED_ADAPTER_PARAMETER_TENSORS,
        "trainable_parameters": EXPECTED_ADAPTER_PARAMETERS,
        "parameter_dtype": "float32",
        "module_counts": {"full_attention": 36, "gdn": 120},
        "mlp_or_expert_covered": False,
        "trained_store": FROZEN_ANSWER_STORE,
    }
    drift = {
        key: {"expected": expected, "actual": frozen.get(key)}
        for key, expected in expected_scalars.items()
        if frozen.get(key) != expected
    }
    if drift:
        raise AnswerLoRAContractError(f"answer adapter_config drifted: {drift}")
    state = payload.get("lora")
    if not isinstance(state, Mapping):
        raise AnswerLoRAContractError("answer checkpoint lacks LoRA state")
    if len(state) != EXPECTED_ADAPTER_PARAMETER_TENSORS:
        raise AnswerLoRAContractError("answer checkpoint tensor count drifted")
    if sum(int(value.numel()) for value in state.values()) != EXPECTED_ADAPTER_PARAMETERS:
        raise AnswerLoRAContractError("answer checkpoint parameter count drifted")
    if any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32 for value in state.values()):
        raise AnswerLoRAContractError("answer checkpoint adapters must all be FP32")
    if any(not bool(torch.isfinite(value).all().item()) for value in state.values()):
        raise AnswerLoRAContractError("answer checkpoint adapters must all be finite")
    state_key_sha256 = hashlib.sha256(
        "\n".join(sorted(state)).encode("utf-8")
    ).hexdigest()
    if frozen.get("state_key_sha256") != state_key_sha256:
        raise AnswerLoRAContractError("answer checkpoint state-key ledger drifted")
    return {
        "payload": payload,
        "format": FORMAT,
        "step": step,
        "sha256": expected_sha256,
        "adapter_config": dict(frozen),
        "governance": dict(governance),
    }


def load_answer_lora_checkpoint(
    model: nn.Module,
    split: TorchSplitCausalLM,
    path: Path,
    *,
    expected_sha256: str,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Install and load the first checkpoint in a 156-module suite."""

    checked = read_answer_lora_checkpoint(
        path,
        expected_sha256=expected_sha256,
        expected_step=expected_step,
    )
    payload = checked["payload"]
    frozen = checked["adapter_config"]
    state = payload["lora"]

    installed, surface = install_and_audit_adapters(
        model,
        split,
        rank=32,
        alpha=64.0,
        dropout=0.0,
        initialization_seed=20260814,
    )
    if list(frozen.get("installed_modules", [])) != installed:
        raise AnswerLoRAContractError("installed 156-module list differs from checkpoint")
    runtime_config = answer_adapter_config(
        model, installed=installed, surface_audit=surface
    )
    if stable_json(runtime_config) != stable_json(dict(frozen)):
        raise AnswerLoRAContractError("runtime adapter contract differs from checkpoint")
    load_lora_state_dict(model, dict(state))
    loaded_state = model.state_dict()
    if any(
        not torch.equal(loaded_state[name].detach().cpu(), value)
        for name, value in state.items()
    ):
        raise AnswerLoRAContractError("answer checkpoint state did not load exactly")
    model.eval()
    set_lora_enabled(model, False)
    return {
        "format": FORMAT,
        "step": checked["step"],
        "sha256": checked["sha256"],
        "adapter_config": runtime_config,
        "adapter_surface": surface,
        "governance": checked["governance"],
    }


def load_answer_lora_state_into_installed(
    model: nn.Module,
    path: Path,
    *,
    expected_sha256: str,
    expected_step: int,
    expected_adapter_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Swap a checked 0/64/128 state into an already-installed surface."""

    checked = read_answer_lora_checkpoint(
        path,
        expected_sha256=expected_sha256,
        expected_step=expected_step,
    )
    if stable_json(checked["adapter_config"]) != stable_json(
        dict(expected_adapter_config)
    ):
        raise AnswerLoRAContractError(
            "checkpoint suite adapter configs are not identical"
        )
    state = checked["payload"]["lora"]
    runtime = lora_state_dict(model)
    if set(runtime) != set(state) or len(list(iter_lora_modules(model))) != EXPECTED_ADAPTER_MODULES:
        raise AnswerLoRAContractError(
            "runtime 156-module surface differs from checkpoint suite"
        )
    load_lora_state_dict(model, dict(state))
    loaded_state = model.state_dict()
    if any(
        not torch.equal(loaded_state[name].detach().cpu(), value)
        for name, value in state.items()
    ):
        raise AnswerLoRAContractError("answer checkpoint swap was not exact")
    model.eval()
    set_lora_enabled(model, False)
    return {
        "format": FORMAT,
        "step": checked["step"],
        "sha256": checked["sha256"],
        "adapter_config": checked["adapter_config"],
        "governance": checked["governance"],
    }


def hybrid_initialize_adapters(
    model: nn.Module,
    *,
    source_checkpoint: Path,
    expected_sha256: str,
    initialization_seed: int,
) -> dict[str, Any]:
    """Warm-start full attention and retain standard zero-B GDN cold start.

    Standard zero-B LoRA necessarily gives the new GDN LoRA-A tensors zero
    first-step gradients.  The formal gate therefore checks GDN-B at step one
    and every GDN A/B tensor at step two in the same job.
    """

    expected_sha256 = require_sha256(expected_sha256, "initial adapter SHA256")
    actual = sha256_file(source_checkpoint)
    if actual != expected_sha256:
        raise AnswerLoRAContractError("initial adapter checkpoint SHA256 mismatch")
    payload = torch.load(source_checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "qcomem_suffix_lora_v1" or payload.get("step") != 0:
        raise AnswerLoRAContractError("warm start must be native LoRA step zero")
    metadata = payload.get("metadata")
    semantics = metadata.get("semantics", {}) if isinstance(metadata, dict) else {}
    if (
        semantics.get("depth") != DEPTH
        or semantics.get("student_suffix_execution_option")
        != "native-functional-cache"
    ):
        raise AnswerLoRAContractError("warm-start checkpoint semantics drifted")
    source_state = payload.get("lora")
    if not isinstance(source_state, dict) or len(source_state) != 72:
        raise AnswerLoRAContractError("warm start must contain exactly 36 A/B modules")
    current = model.state_dict()
    missing = sorted(set(source_state) - set(current))
    if missing:
        raise AnswerLoRAContractError(f"warm-start keys are absent: {missing}")
    with torch.no_grad():
        for name, value in source_state.items():
            if not any(
                f".{suffix}." in name for suffix in FULL_ATTENTION_TARGET_SUFFIXES
            ):
                raise AnswerLoRAContractError(f"warm start contains non-attention key {name}")
            current[name].copy_(value.to(device=current[name].device))
        gdn_modules = []
        full_modules = []
        for name, module in model.named_modules():
            if not isinstance(module, LoRALinear):
                continue
            leaf = name.rsplit(".", 1)[-1]
            if leaf in GDN_TARGET_SUFFIXES:
                gdn_modules.append(name)
            elif leaf in FULL_ATTENTION_TARGET_SUFFIXES:
                full_modules.append(name)
            else:  # pragma: no cover - surface audit should already prevent this
                raise AnswerLoRAContractError(f"unexpected LoRA module {name}")
    if len(full_modules) != EXPECTED_FULL_ATTENTION_MODULES:
        raise AnswerLoRAContractError("full-attention hybrid init coverage drifted")
    if len(gdn_modules) != EXPECTED_GDN_MODULES:
        raise AnswerLoRAContractError("GDN hybrid init coverage drifted")
    state = lora_state_dict(model)
    nonzero = {
        "full_attention": sum(
            int(torch.count_nonzero(value).item() > 0)
            for name, value in state.items()
            if any(f".{suffix}." in name for suffix in FULL_ATTENTION_TARGET_SUFFIXES)
        ),
        "gdn": sum(
            int(torch.count_nonzero(value).item() > 0)
            for name, value in state.items()
            if any(f".{suffix}." in name for suffix in GDN_TARGET_SUFFIXES)
        ),
    }
    if nonzero != {"full_attention": 72, "gdn": 120}:
        raise AnswerLoRAContractError(
            f"hybrid initialization differs from warm/zero-B contract: {nonzero}"
        )
    return {
        "kind": "native_step0_full_attention_plus_seeded_gdn_v1",
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": actual,
        "source_step": 0,
        "source_full_attention_modules": len(full_modules),
        "new_gdn_modules": len(gdn_modules),
        "new_gdn_initialization": {
            "lora_a": "kaiming_uniform",
            "lora_b": "zeros",
            "seed": initialization_seed,
            "step1_expected_gradient": {
                "lora_a": "finite_present_zero_is_mathematically_expected",
                "lora_b": "finite_nonzero_required",
            },
            "step2_expected_gradient": "all_gdn_lora_a_and_lora_b_finite_nonzero",
        },
        "all_adapter_parameter_tensors_nonzero": False,
        "nonzero_parameter_tensors_by_surface": nonzero,
        "optimizer_scheduler_state_restored": False,
    }


def teacher_target_contract(record: Mapping[str, torch.Tensor], targets: torch.Tensor) -> None:
    required = {
        "target_ids",
        "topk_ids",
        "topk_logprobs",
        "tail_logprob",
        "normalized_hidden",
    }
    if set(record) != required:
        raise AnswerLoRAContractError("teacher target fields drifted")
    count = int(targets.numel())
    if not torch.equal(record["target_ids"].long().cpu(), targets.long().cpu()):
        raise AnswerLoRAContractError("teacher targets differ from answer/EOS labels")
    if (
        record["topk_ids"].ndim != 2
        or record["topk_ids"].shape[0] != count
        or record["topk_logprobs"].shape != record["topk_ids"].shape
        or tuple(record["tail_logprob"].shape) != (count,)
        or record["normalized_hidden"].ndim != 2
        or record["normalized_hidden"].shape[0] != count
    ):
        raise AnswerLoRAContractError("teacher target shapes drifted")
    finite = (
        record["topk_logprobs"],
        record["tail_logprob"],
        record["normalized_hidden"],
    )
    if not all(torch.isfinite(value).all() for value in finite):
        raise AnswerLoRAContractError("teacher targets contain non-finite values")


def answer_preservation_objective(
    selected_hidden: torch.Tensor,
    lm_head: nn.Module,
    targets: torch.Tensor,
    teacher: Mapping[str, torch.Tensor],
    *,
    hard_weight: float,
    kl_weight: float,
    hidden_weight: float,
    projection_chunk_positions: int = 32,
) -> dict[str, torch.Tensor]:
    """Compute per-example mean CE/KL/hidden loss on answer+EOS only."""

    if selected_hidden.ndim == 3:
        if selected_hidden.shape[0] != 1:
            raise AnswerLoRAContractError("formal objective requires per-rank batch size one")
        selected_hidden = selected_hidden[0]
    if selected_hidden.ndim != 2 or targets.ndim != 1:
        raise AnswerLoRAContractError("hidden/targets must be [answer,H] and [answer]")
    if selected_hidden.shape[0] != targets.numel():
        raise AnswerLoRAContractError("student answer positions differ from targets")
    if not math.isclose(hard_weight + kl_weight + hidden_weight, 1.0):
        raise AnswerLoRAContractError("objective weights must sum to one")
    if projection_chunk_positions < 1:
        raise AnswerLoRAContractError("projection chunk positions must be positive")
    teacher_target_contract(teacher, targets)
    topk_ids = teacher["topk_ids"].long().to(selected_hidden.device)
    teacher_topk_logprobs = teacher["topk_logprobs"].float().to(selected_hidden.device)
    teacher_tail_logprob = teacher["tail_logprob"].float().to(selected_hidden.device)
    ce_sum = selected_hidden.new_zeros((), dtype=torch.float32)
    kl_sum = selected_hidden.new_zeros((), dtype=torch.float32)
    positions = int(targets.numel())

    def projected_chunk_loss(
        hidden_chunk: torch.Tensor,
        target_chunk: torch.Tensor,
        topk_id_chunk: torch.Tensor,
        teacher_topk_chunk: torch.Tensor,
        teacher_tail_chunk: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = lm_head(hidden_chunk).float()
        chunk_ce = F.cross_entropy(logits, target_chunk.long(), reduction="sum")
        logz = torch.logsumexp(logits, dim=-1)
        student_topk = logits.gather(1, topk_id_chunk) - logz[:, None]
        student_topk_logmass = torch.logsumexp(student_topk, dim=-1)
        student_tail = log1mexp(student_topk_logmass)
        chunk_kl = (
            (teacher_topk_chunk.exp() * (teacher_topk_chunk - student_topk)).sum(
                dim=-1
            )
            + teacher_tail_chunk.exp() * (teacher_tail_chunk - student_tail)
        ).sum()
        return chunk_ce, chunk_kl

    for start in range(0, positions, projection_chunk_positions):
        end = min(start + projection_chunk_positions, positions)
        chunk_inputs = (
            selected_hidden[start:end],
            targets[start:end],
            topk_ids[start:end],
            teacher_topk_logprobs[start:end],
            teacher_tail_logprob[start:end],
        )
        if torch.is_grad_enabled() and selected_hidden.requires_grad:
            # Non-reentrant activation checkpointing recomputes only this small
            # projection chunk during backward, so full-vocabulary logits from
            # every answer position are not retained simultaneously.
            chunk_ce, chunk_kl = checkpoint(
                projected_chunk_loss, *chunk_inputs, use_reentrant=False
            )
        else:
            chunk_ce, chunk_kl = projected_chunk_loss(*chunk_inputs)
        ce_sum = ce_sum + chunk_ce
        kl_sum = kl_sum + chunk_kl
    ce = ce_sum / positions
    kl = kl_sum / positions
    teacher_hidden = teacher["normalized_hidden"].float().to(selected_hidden.device)
    hidden = (
        1.0
        - (F.normalize(selected_hidden.float(), dim=-1) * teacher_hidden).sum(dim=-1)
    ).mean()
    loss = hard_weight * ce + kl_weight * kl + hidden_weight * hidden
    return {"loss": loss, "ce": ce, "kl": kl, "hidden": hidden}


class AnswerSupervisedNativeLoRA(nn.Module):
    """Q4 CoMem replay student with native functional suffix cache writes."""

    def __init__(
        self,
        model: nn.Module,
        *,
        depth: int,
        quant: ReplayQuantConfig,
        hard_weight: float,
        kl_weight: float,
        hidden_weight: float,
        projection_chunk_positions: int = 32,
    ) -> None:
        super().__init__()
        if depth != DEPTH:
            raise AnswerLoRAContractError(f"formal depth is frozen to {DEPTH}")
        self.model = model
        self.adapter = TorchSplitCausalLM(model)
        self.depth = depth
        self.quant = quant
        self.hard_weight = hard_weight
        self.kl_weight = kl_weight
        self.hidden_weight = hidden_weight
        if projection_chunk_positions < 1:
            raise AnswerLoRAContractError("projection chunk positions must be positive")
        self.projection_chunk_positions = projection_chunk_positions
        self.last_cache_audit: dict[str, Any] | None = None

    def forward(
        self,
        document_ids: torch.Tensor,
        continuation_input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        teacher_topk_ids: torch.Tensor,
        teacher_topk_logprobs: torch.Tensor,
        teacher_tail_logprob: torch.Tensor,
        teacher_normalized_hidden: torch.Tensor,
    ) -> dict[str, torch.Tensor | int | dict[str, Any]]:
        if document_ids.ndim == 1:
            document_ids = document_ids.unsqueeze(0)
        if continuation_input_ids.ndim == 1:
            continuation_input_ids = continuation_input_ids.unsqueeze(0)
        if target_ids.ndim != 1:
            target_ids = target_ids.flatten()
        with torch.no_grad():
            raw = self.adapter.write_lower_replay(document_ids, self.depth)
            packed = raw.quantize(
                bits=self.quant.residual_bits,
                attention_bits=self.quant.attention_bits,
                linear_bits=self.quant.linear_bits,
                cache_layer_bits=self.quant.cache_layer_bits,
                group_size=self.quant.group_size,
            )
            local = packed.fork()
            continuation_residual = self.adapter.continue_lower_replay(
                local, continuation_input_ids
            )
            document_residual = local.document_residual
            persistent_nbytes = packed.stored_nbytes
        # Convert inference tensors into ordinary frozen activations.  Suffix
        # autograd begins here and covers both document prefill and the complete
        # multi-token query+answer continuation.
        document_residual = document_residual.clone()
        continuation_residual = continuation_residual.clone()
        hidden, cache_audit = quant_student_suffix_hidden(
            self.adapter,
            depth=self.depth,
            document_residual=document_residual,
            query_residual=continuation_residual,
            execution="native-functional-cache",
            return_cache_audit=True,
        )
        if target_ids.numel() > hidden.shape[1]:
            raise AnswerLoRAContractError("continuation has too few answer positions")
        selected_hidden = hidden[:, -target_ids.numel() :, :]
        teacher = {
            "target_ids": target_ids.detach().cpu(),
            "topk_ids": teacher_topk_ids,
            "topk_logprobs": teacher_topk_logprobs,
            "tail_logprob": teacher_tail_logprob,
            "normalized_hidden": teacher_normalized_hidden,
        }
        objective = answer_preservation_objective(
            selected_hidden,
            self.adapter.lm_head,
            target_ids,
            teacher,
            hard_weight=self.hard_weight,
            kl_weight=self.kl_weight,
            hidden_weight=self.hidden_weight,
            projection_chunk_positions=self.projection_chunk_positions,
        )
        self.last_cache_audit = cache_audit
        return {
            **objective,
            "persistent_nbytes": persistent_nbytes,
            "answer_positions": int(target_ids.numel()),
            "continuation_positions": int(continuation_input_ids.shape[1]),
            "cache_audit": cache_audit,
        }


@torch.inference_mode()
def answer_decode_semantic_diagnostic(
    model: nn.Module,
    boundary: AnswerBoundary,
    *,
    quant: ReplayQuantConfig,
    depth: int = DEPTH,
    projection_chunk_positions: int = 32,
) -> dict[str, Any]:
    """Compare whole-block teacher forcing with token-by-token forcing.

    No threshold is asserted: Qwen3.5 GDN can be numerically sensitive to
    chunk boundaries.  This diagnostic records, rather than assumes, whether
    the training execution agrees with deployment-like incremental decode.
    """

    from qcomem_qwen35_native_cache import install_native_functional_linear_cache

    adapter = TorchSplitCausalLM(model)

    def packed_document():
        raw = adapter.write_lower_replay(boundary.document_ids, depth)
        return raw.quantize(
            bits=quant.residual_bits,
            attention_bits=quant.attention_bits,
            linear_bits=quant.linear_bits,
            cache_layer_bits=quant.cache_layer_bits,
            group_size=quant.group_size,
        )

    whole_local = packed_document().fork()
    whole_continuation = adapter.continue_lower_replay(
        whole_local, boundary.continuation_input_ids
    )
    whole_hidden = quant_student_suffix_hidden(
        adapter,
        depth=depth,
        document_residual=whole_local.document_residual,
        query_residual=whole_continuation,
        execution="native-functional-cache",
        return_cache_audit=False,
    )
    assert isinstance(whole_hidden, torch.Tensor)
    whole_hidden = whole_hidden[:, -boundary.answer_positions :, :]

    token_local = packed_document().fork()
    suffix_cache = adapter.make_cache()
    install_native_functional_linear_cache(suffix_cache, adapter.config)
    adapter._run_layers(
        token_local.document_residual,
        depth,
        adapter.num_layers,
        past_key_values=suffix_cache,
        position_offset=0,
    )
    query_residual = adapter.continue_lower_replay(token_local, boundary.query_ids)
    query_hidden = adapter._run_layers(
        query_residual,
        depth,
        adapter.num_layers,
        past_key_values=suffix_cache,
        position_offset=boundary.document_ids.numel(),
    )
    token_hidden = [adapter.language_model.norm(query_hidden[:, -1:, :])]
    offset = boundary.document_ids.numel() + boundary.query_ids.numel()
    for answer_input in boundary.answer_ids[:-1]:
        residual = adapter.continue_lower_replay(token_local, answer_input.view(1))
        hidden = adapter._run_layers(
            residual,
            depth,
            adapter.num_layers,
            past_key_values=suffix_cache,
            position_offset=offset,
        )
        token_hidden.append(adapter.language_model.norm(hidden[:, -1:, :]))
        offset += 1
    token_hidden_tensor = torch.cat(token_hidden, dim=1)
    if token_hidden_tensor.shape != whole_hidden.shape:
        raise AnswerLoRAContractError("decode diagnostic answer positions drifted")

    def project(hidden: torch.Tensor) -> torch.Tensor:
        parts = []
        for start in range(0, hidden.shape[1], projection_chunk_positions):
            parts.append(
                adapter.lm_head(
                    hidden[:, start : start + projection_chunk_positions, :]
                ).float()
            )
        return torch.cat(parts, dim=1)[0]

    whole_logits = project(whole_hidden)
    token_logits = project(token_hidden_tensor)
    whole_logp = F.log_softmax(whole_logits, dim=-1)
    token_logp = F.log_softmax(token_logits, dim=-1)
    kl = (whole_logp.exp() * (whole_logp - token_logp)).sum(dim=-1)
    whole_top1 = whole_logits.argmax(dim=-1)
    token_top1 = token_logits.argmax(dim=-1)
    equal = whole_top1 == token_top1
    divergent = torch.nonzero(~equal, as_tuple=False).flatten()
    return {
        "positions": int(equal.numel()),
        "top1_equal_positions": int(equal.sum().item()),
        "top1_agreement": float(equal.float().mean().item()),
        "first_top1_divergence_position": (
            int(divergent[0].item()) if divergent.numel() else None
        ),
        "mean_kl_whole_to_token": float(kl.mean().item()),
        "max_kl_whole_to_token": float(kl.max().item()),
        "max_abs_logit_difference": float(
            (whole_logits - token_logits).abs().max().item()
        ),
        "whole_execution": "query_plus_answer_prefix_one_multi_token_block",
        "token_execution": "query_multi_token_then_answer_prefix_one_token_at_a_time",
        "equivalence_claimed": False,
    }


def adapter_parameter_records(model: nn.Module) -> dict[str, nn.Parameter]:
    records = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.endswith(("lora_a", "lora_b"))
    }
    if len(records) != EXPECTED_ADAPTER_PARAMETER_TENSORS:
        raise AnswerLoRAContractError(
            f"expected {EXPECTED_ADAPTER_PARAMETER_TENSORS} adapter tensors, "
            f"found {len(records)}"
        )
    if len({name.rsplit(".", 1)[0] for name in records}) != EXPECTED_ADAPTER_MODULES:
        raise AnswerLoRAContractError("adapter parameter/module cardinality drifted")
    return records


def gradient_records(parameters: Mapping[str, nn.Parameter]) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in sorted(parameters.items()):
        gradient = parameter.grad
        rows.append(
            {
                "name": name,
                "dtype": str(parameter.dtype),
                "present": gradient is not None,
                "finite": bool(gradient is not None and torch.isfinite(gradient).all()),
                "nonzero": bool(
                    gradient is not None and torch.count_nonzero(gradient).item() > 0
                ),
                "norm": (
                    float(torch.linalg.vector_norm(gradient.float()).item())
                    if gradient is not None
                    else None
                ),
            }
        )
    return rows


def update_records(
    parameters: Mapping[str, nn.Parameter], before: Mapping[str, torch.Tensor]
) -> list[dict[str, Any]]:
    rows = []
    for name, parameter in sorted(parameters.items()):
        current = parameter.detach().cpu()
        delta = current - before[name]
        rows.append(
            {
                "name": name,
                "finite": bool(torch.isfinite(current).all() and torch.isfinite(delta).all()),
                "nonzero": bool(torch.count_nonzero(delta).item() > 0),
                "delta_norm": float(torch.linalg.vector_norm(delta.float()).item()),
                "max_abs_delta": float(delta.float().abs().max().item()),
            }
        )
    return rows


def optimizer_fp32_audit(
    optimizer: torch.optim.Optimizer,
    parameters: Mapping[str, nn.Parameter],
) -> dict[str, Any]:
    parameter_ids = {id(parameter) for parameter in parameters.values()}
    rows = []
    for parameter, state in optimizer.state.items():
        if id(parameter) not in parameter_ids:
            continue
        tensor_dtypes = {
            name: str(value.dtype)
            for name, value in state.items()
            if isinstance(value, torch.Tensor)
        }
        rows.append(tensor_dtypes)
    passed = (
        len(rows) == len(parameters)
        and all(parameter.dtype == torch.float32 for parameter in parameters.values())
        and all(
            all(dtype == "torch.float32" for dtype in row.values()) for row in rows
        )
    )
    return {
        "passed": passed,
        "adapter_parameters_fp32": all(
            parameter.dtype == torch.float32 for parameter in parameters.values()
        ),
        "optimizer_parameter_states": len(rows),
        "expected_parameter_states": len(parameters),
        "state_dtype_counts": dict(
            Counter(dtype for row in rows for dtype in row.values())
        ),
    }


def evaluate_step1_gate(
    rank_records: Sequence[Mapping[str, Any]],
    *,
    minimum_headroom_bytes: int,
) -> dict[str, Any]:
    checks = {
        "eight_ranks": len(rank_records) == WORLD_SIZE
        and {row.get("rank") for row in rank_records} == set(range(WORLD_SIZE)),
        "all_adapter_gradients_present_finite": all(
            row.get("gradient_tensors") == EXPECTED_ADAPTER_PARAMETER_TENSORS
            and row.get("finite_gradient_tensors")
            == EXPECTED_ADAPTER_PARAMETER_TENSORS
            for row in rank_records
        ),
        "warm_full_attention_all_gradients_updates_nonzero": all(
            row.get("full_attention", {}).get("gradient_tensors") == 72
            and row["full_attention"].get("nonzero_gradient_tensors") == 72
            and row["full_attention"].get("nonzero_update_tensors") == 72
            for row in rank_records
        ),
        "cold_gdn_b_gradients_updates_nonzero": all(
            row.get("gdn_lora_b", {}).get("gradient_tensors") == 120
            and row["gdn_lora_b"].get("nonzero_gradient_tensors") == 120
            and row["gdn_lora_b"].get("nonzero_update_tensors") == 120
            for row in rank_records
        ),
        "cold_gdn_a_expected_zero_first_update": all(
            row.get("gdn_lora_a", {}).get("gradient_tensors") == 120
            and row["gdn_lora_a"].get("finite_gradient_tensors") == 120
            and row["gdn_lora_a"].get("nonzero_gradient_tensors") == 0
            and row["gdn_lora_a"].get("nonzero_update_tensors") == 0
            for row in rank_records
        ),
        "all_adapter_updates_finite": all(
            row.get("finite_update_tensors") == EXPECTED_ADAPTER_PARAMETER_TENSORS
            for row in rank_records
        ),
        "native_functional_cache": all(
            isinstance(row.get("cache"), Mapping)
            and row["cache"].get("execution") == "native-functional-cache"
            and row["cache"].get("hard_gate_passed") is True
            and row["cache"].get("original_cache_versions_unchanged") is True
            and row["cache"].get("all_cache_paths_rebound") is True
            for row in rank_records
        ),
        "multi_token_continuation": all(
            isinstance(row.get("continuation_positions"), int)
            and row["continuation_positions"] > 1
            and row.get("cache", {}).get("query_positions_observed")
            == row["continuation_positions"]
            for row in rank_records
        ),
        "fp32_optimizer": all(
            row.get("optimizer_fp32", {}).get("passed") is True
            for row in rank_records
        ),
        "memory_headroom": all(
            isinstance(row.get("reserved_headroom_bytes"), int)
            and row["reserved_headroom_bytes"] >= minimum_headroom_bytes
            for row in rank_records
        ),
        "finite_loss": all(row.get("finite_loss") is True for row in rank_records),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "step": 1,
        "checks": checks,
        "ranks": list(rank_records),
        "minimum_required_headroom_bytes": minimum_headroom_bytes,
        "separate_smoke_job_used": False,
        "raw_longbench_validation_or_test_read": False,
    }


def evaluate_step2_gate(rank_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    checks = {
        "eight_ranks": len(rank_records) == WORLD_SIZE
        and {row.get("rank") for row in rank_records} == set(range(WORLD_SIZE)),
        "all_adapter_gradients_finite_nonzero": all(
            row.get("gradient_tensors") == EXPECTED_ADAPTER_PARAMETER_TENSORS
            and row.get("finite_gradient_tensors")
            == EXPECTED_ADAPTER_PARAMETER_TENSORS
            and row.get("nonzero_gradient_tensors")
            == EXPECTED_ADAPTER_PARAMETER_TENSORS
            for row in rank_records
        ),
        "all_adapter_updates_finite_nonzero": all(
            row.get("finite_update_tensors") == EXPECTED_ADAPTER_PARAMETER_TENSORS
            and row.get("nonzero_update_tensors")
            == EXPECTED_ADAPTER_PARAMETER_TENSORS
            for row in rank_records
        ),
        "native_functional_cache": all(
            row.get("cache", {}).get("hard_gate_passed") is True
            and row["cache"].get("original_cache_versions_unchanged") is True
            and row["cache"].get("all_cache_paths_rebound") is True
            for row in rank_records
        ),
        "fp32_optimizer": all(
            row.get("optimizer_fp32", {}).get("passed") is True
            for row in rank_records
        ),
        "finite_loss": all(row.get("finite_loss") is True for row in rank_records),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "step": 2,
        "checks": checks,
        "ranks": list(rank_records),
        "same_job_as_step1": True,
        "separate_smoke_job_used": False,
        "raw_longbench_validation_or_test_read": False,
    }
def summarize_weighted_examples(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise AnswerLoRAContractError("cannot summarize zero examples")
    metrics = ("loss", "ce", "kl", "hidden")
    for row in rows:
        if any(not math.isfinite(float(row[key])) for key in metrics):
            raise AnswerLoRAContractError("summary contains non-finite metrics")
    weight_sum = sum(float(row["balance_weight"]) for row in rows)
    if weight_sum <= 0:
        raise AnswerLoRAContractError("summary weights must be positive")

    def selected(part: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        denominator = sum(float(row["balance_weight"]) for row in part)
        return {
            "examples": len(part),
            "answer_positions": sum(int(row["answer_positions"]) for row in part),
            **{
                key: sum(float(row[key]) * float(row["balance_weight"]) for row in part)
                / denominator
                for key in metrics
            },
        }

    groups = sorted({str(row["balance_group"]) for row in rows})
    return {
        "overall": selected(rows),
        "by_balance_group": {
            group: selected([row for row in rows if row["balance_group"] == group])
            for group in groups
        },
        "selection_metric": "overall.loss",
        "selection_direction": "min",
        "example_balanced": True,
        "target_token_weighting_used": False,
        "weight_sum": weight_sum,
    }


def choose_best_checkpoint(evaluations: Mapping[int, Mapping[str, Any]]) -> int:
    if set(evaluations) != set(EVALUATION_STEPS):
        raise AnswerLoRAContractError("checkpoint selection requires steps 0/64/128")
    values = {
        step: float(evaluations[step]["summary"]["overall"]["loss"])
        for step in EVALUATION_STEPS
    }
    if not all(math.isfinite(value) for value in values.values()):
        raise AnswerLoRAContractError("heldout checkpoint objective is non-finite")
    return min(values, key=lambda step: (values[step], step))


__all__ = [
    "AnswerBoundary",
    "AnswerLoRAContractError",
    "AnswerSupervisedNativeLoRA",
    "BALANCE_MASS",
    "CHECKPOINT_STEPS",
    "DEPTH",
    "EVALUATION_STEPS",
    "EXPECTED_ADAPTER_MODULES",
    "EXPECTED_ADAPTER_PARAMETERS",
    "EXPECTED_ADAPTER_PARAMETER_TENSORS",
    "FORMAT",
    "GDN_TARGET_SUFFIXES",
    "MLP_EXCLUDED_SUFFIXES",
    "STEPS",
    "MAX_ADAPTER_PARAMETERS",
    "TARGET_SUFFIXES",
    "TEACHER_FORMAT",
    "TEACHER_MANIFEST_FORMAT",
    "WORLD_SIZE",
    "adapter_parameter_records",
    "answer_adapter_config",
    "answer_decode_semantic_diagnostic",
    "answer_boundary",
    "answer_preservation_objective",
    "audit_adapter_surface",
    "balance_group",
    "balanced_domain_schedule",
    "choose_best_checkpoint",
    "evaluate_step1_gate",
    "evaluate_step2_gate",
    "example_balance_weights",
    "gradient_records",
    "hybrid_initialize_adapters",
    "install_and_audit_adapters",
    "load_answer_lora_checkpoint",
    "load_answer_lora_state_into_installed",
    "optimizer_fp32_audit",
    "reject_longbench_path_or_digest",
    "read_answer_lora_checkpoint",
    "require_sha256",
    "sha256_file",
    "stable_json",
    "summarize_weighted_examples",
    "teacher_target_contract",
    "target_start",
    "update_records",
]
