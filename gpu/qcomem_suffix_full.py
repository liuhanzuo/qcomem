from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import torch
from torch import nn

from qcomem_lora import (
    ReplayQuantConfig,
    cached_two_stage_autograd_capability_gate,
    functional_cache_capability_gate,
)


@dataclass(frozen=True)
class PublishedModelScaleReference:
    """Rounded safetensors audit supplied by the frozen 35B model checkout."""

    total_parameters_rounded: int = 35_952_000_000
    text_parameters_rounded: int = 34_661_000_000
    lower_layers_0_through_6_rounded: int = 6_729_000_000
    suffix_layers_7_through_39_rounded: int = 27_751_000_000
    suffix_attention_rounded: int = 1_055_000_000
    suffix_mlp_moe_rounded: int = 26_696_000_000
    rounding_unit_parameters: int = 1_000_000


def _unique_named_parameters(
    modules: Iterable[tuple[str, nn.Module]],
) -> list[tuple[str, nn.Parameter]]:
    seen: set[int] = set()
    values: list[tuple[str, nn.Parameter]] = []
    for prefix, module in modules:
        for name, parameter in module.named_parameters(recurse=True):
            if id(parameter) in seen:
                continue
            seen.add(id(parameter))
            values.append((f"{prefix}.{name}" if name else prefix, parameter))
    return values


def _parameter_category(name: str) -> str:
    lowered = name.lower()
    if any(token in lowered for token in ("self_attn", "linear_attn", "attention")):
        return "attention"
    if any(token in lowered for token in ("mlp", "experts", "shared_expert")):
        return "mlp_moe"
    return "normalization_or_other"


def configure_suffix_full_trainability(
    model: nn.Module,
    layers: nn.ModuleList,
    *,
    depth: int,
) -> dict[str, Any]:
    """Freeze the model and unfreeze every parameter in transformer layers >= depth.

    Embeddings, lower/write layers, final norm and lm_head remain frozen.  This
    scope is deliberately named ``suffix_full_distillation``; it is not
    end-to-end full-model SFT.
    """

    if depth < 0 or depth > len(layers):
        raise ValueError("depth is outside the model layer range")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    suffix_parameters = _unique_named_parameters(
        (f"layers.{index}", layers[index]) for index in range(depth, len(layers))
    )
    for _, parameter in suffix_parameters:
        parameter.requires_grad_(True)
    estimated = sum(parameter.numel() for _, parameter in suffix_parameters)
    actual = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if estimated != actual:
        raise RuntimeError(
            f"suffix parameter estimate {estimated:,} != actual trainable count {actual:,}"
        )
    categories = {"attention": 0, "mlp_moe": 0, "normalization_or_other": 0}
    logical_bytes = 0
    for name, parameter in suffix_parameters:
        categories[_parameter_category(name)] += parameter.numel()
        logical_bytes += parameter.numel() * parameter.element_size()
    return {
        "scope": "suffix_full_distillation",
        "depth": depth,
        "suffix_layer_indices": list(range(depth, len(layers))),
        "estimated_trainable_parameters": estimated,
        "actual_trainable_parameters": actual,
        "estimate_matches_actual": True,
        "trainable_logical_bytes": logical_bytes,
        "parameter_dtype_counts": _dtype_counts(parameter for _, parameter in suffix_parameters),
        "category_parameter_counts": categories,
        "embeddings_trainable": False,
        "final_norm_trainable": False,
        "lm_head_trainable": False,
        "lower_write_trainable": False,
        "reference_rounded_safetensors_audit": asdict(PublishedModelScaleReference()),
    }


def _dtype_counts(parameters: Iterable[nn.Parameter]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for parameter in parameters:
        key = str(parameter.dtype)
        counts[key] = counts.get(key, 0) + parameter.numel()
    return counts


def estimate_sharded_training_storage(
    *,
    trainable_parameters: int,
    parameter_bytes: int,
    world_size: int,
) -> dict[str, Any]:
    """Strict logical-byte ledger without assuming Adam moment dtype.

    PyTorch optimizer-state dtype is implementation-dependent.  We therefore
    report both BF16- and FP32-moment bounds and record observed state dtypes
    after the real optimizer step.
    """

    if trainable_parameters < 1 or parameter_bytes < 1 or world_size < 1:
        raise ValueError("trainable parameters, bytes, and world size must be positive")
    gradient_bytes = parameter_bytes
    bf16_moment_bytes = trainable_parameters * 2 * 2
    fp32_moment_bytes = trainable_parameters * 2 * 4
    model_only_checkpoint = parameter_bytes
    bf16_training_checkpoint = model_only_checkpoint + bf16_moment_bytes
    fp32_training_checkpoint = model_only_checkpoint + fp32_moment_bytes

    def shard(value: int) -> int:
        return math.ceil(value / world_size)

    return {
        "global": {
            "trainable_parameter_bytes": parameter_bytes,
            "gradient_bytes": gradient_bytes,
            "adam_two_moments_if_bf16_bytes": bf16_moment_bytes,
            "adam_two_moments_if_fp32_bytes": fp32_moment_bytes,
            "model_only_checkpoint_bytes": model_only_checkpoint,
            "training_checkpoint_if_bf16_moments_bytes": bf16_training_checkpoint,
            "training_checkpoint_if_fp32_moments_bytes": fp32_training_checkpoint,
        },
        "ideal_even_shard_per_rank": {
            "trainable_parameter_bytes": shard(parameter_bytes),
            "gradient_bytes": shard(gradient_bytes),
            "adam_two_moments_if_bf16_bytes": shard(bf16_moment_bytes),
            "adam_two_moments_if_fp32_bytes": shard(fp32_moment_bytes),
            "model_only_checkpoint_bytes": shard(model_only_checkpoint),
            "training_checkpoint_if_bf16_moments_bytes": shard(bf16_training_checkpoint),
            "training_checkpoint_if_fp32_moments_bytes": shard(fp32_training_checkpoint),
        },
        "excludes": [
            "FSDP padding and metadata",
            "frozen lower/model shards",
            "activations and temporary all-gathers",
            "allocator fragmentation",
            "scheduler and RNG metadata",
        ],
    }


def end_to_end_full_model_capability_gate() -> dict[str, Any]:
    """Describe why true end-to-end SFT/QAT is intentionally unavailable."""

    blockers = {
        "differentiable_lower_write": False,
        "ste_pack_dequant_for_residual": False,
        "ste_pack_dequant_for_attention_cache": False,
        "ste_pack_dequant_for_linear_recurrent_state": False,
        "cached_recurrent_state_autograd_validated": False,
        "end_to_end_fsdp_checkpoint_roundtrip_validated": False,
    }
    return {
        "requested_scope": "end_to_end_full_model_sft_qat",
        "implemented": False,
        "capability_gate_passed": all(blockers.values()),
        "capabilities": blockers,
        "required_before_implementation": [
            "replace inference-mode lower/write with a differentiable path",
            "define STE fake-quant/dequant for every stored state category",
            "verify gradients across DynamicCache and GatedDeltaNet state boundaries",
            "wrap the complete model in FSDP/ZeRO-3 and validate sharded resume",
        ],
        "loss_plan": {
            "default_fair_comparison": "Q16 teacher top-k bidirectional KL on query positions",
            "token_ce_sft": "separate future ablation; not implemented by this gate",
        },
    }


def suffix_full_semantics_metadata(
    *,
    depth: int,
    teacher_source: str,
    quant: ReplayQuantConfig,
    student_suffix_execution: str,
) -> dict[str, Any]:
    if teacher_source not in {"online", "offline"}:
        raise ValueError("teacher_source must be online or offline")
    if student_suffix_execution not in {
        "cached-two-stage",
        "detached-document-cache",
    }:
        raise ValueError(
            "suffix_full_distillation requires cached-two-stage or "
            "detached-document-cache execution"
        )
    detached_document_cache = student_suffix_execution == "detached-document-cache"
    return {
        "mode": "quant",
        "training_scope": "suffix_full_distillation",
        "depth": depth,
        "teacher_kind": "q16_replay",
        "teacher_source": teacher_source,
        "write_path_trainable": False,
        "lower_layers_trainable": False,
        "suffix_transformer_layers_trainable": True,
        "embeddings_trainable": False,
        "final_norm_trainable": False,
        "lm_head_trainable": False,
        "adapter_kind": None,
        "suffix_only_adapter": False,
        "is_lora": False,
        "is_qlora": False,
        "is_full_model_sft": False,
        "is_end_to_end_qat": False,
        "quantization_training_name": "suffix_full_quantization_conditioned_distillation",
        "store": {
            "kind": "packed_residual_and_lower_replay_state",
            **asdict(quant),
        },
        "student_suffix_execution_option": student_suffix_execution,
        "student_suffix_execution": (
            "cached_document_prefill_detached_then_full_query_continuation"
            if detached_document_cache
            else "cached_document_prefill_then_full_query_continuation"
        ),
        "deployment_suffix_execution": "cached_document_prefill_then_query_continuation",
        "training_deployment_cache_boundary_structurally_aligned": True,
        "training_deployment_gradient_semantics_equivalent": False,
        "document_prefill_parameter_gradients_enabled": not detached_document_cache,
        "document_cache_detached_before_query": detached_document_cache,
        "claim_limit": (
            "query-continuation-only suffix-full distillation; document-prefill "
            "parameter contribution is frozen"
            if detached_document_cache
            else "known backward failure; not trainable with the mutable cache"
        ),
        "real_model_cached_autograd_smoke_passed": False,
        "cached_two_stage_autograd_capability": cached_two_stage_autograd_capability_gate(),
        "functional_cache_capability": functional_cache_capability_gate(),
        "deployment_semantic_eval_gate_required": True,
        "note": (
            "Fake-quant/dequant is applied to persistent residual/KV/recurrent state. "
            "Gradients enter every parameter in transformer suffix layers at and above "
            "the split depth; they do not enter lower/write, embeddings, final norm or "
            "lm_head. This is suffix-full KL distillation, not LoRA, QLoRA, full-model "
            "SFT, or end-to-end QAT. Mutable cached-two-stage backward is known to fail; "
            "the detached-document-cache option clones frozen document state and is only "
            "a query-continuation approximation. Query-side mutable-cache backward still "
            "requires a real-model capability smoke."
        ),
    }
def observed_optimizer_state(model_optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    dtype_elements: dict[str, int] = {}
    total_bytes = 0
    for state in model_optimizer.state.values():
        for value in state.values():
            if not isinstance(value, torch.Tensor):
                continue
            key = str(value.dtype)
            dtype_elements[key] = dtype_elements.get(key, 0) + value.numel()
            total_bytes += value.numel() * value.element_size()
    return {"dtype_elements": dtype_elements, "local_shard_bytes": total_bytes}


def suffix_gradient_coverage_local(
    module: nn.Module,
    *,
    depth: int,
    num_layers: int,
) -> dict[str, Any]:
    """Summarize local FSDP gradient shards by transformer layer.

    Only scalar norms/counts are returned. No 27B-scale gradient tensor is
    gathered. A zero-sized ``use_orig_params`` shard is valid and is excluded
    from the missing-gradient count.
    """

    layer_rows = {
        str(index): {
            "parameter_shard_elements": 0,
            "gradient_elements": 0,
            "missing_gradient_parameter_shard_elements": 0,
            "gradient_tensors": 0,
            "finite_gradient_tensors": 0,
            "nonzero_gradient_tensors": 0,
            "squared_gradient_norm": 0.0,
        }
        for index in range(depth, num_layers)
    }
    unmatched_trainable: list[str] = []
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
        if match is None:
            if parameter.numel() > 0:
                unmatched_trainable.append(name)
            continue
        index = int(match.group(1))
        if not depth <= index < num_layers:
            raise RuntimeError(f"trainable parameter escaped suffix scope: {name}")
        row = layer_rows[str(index)]
        row["parameter_shard_elements"] += parameter.numel()
        if parameter.numel() == 0:
            continue
        gradient = parameter.grad
        if gradient is None:
            row["missing_gradient_parameter_shard_elements"] += parameter.numel()
            continue
        norm = torch.linalg.vector_norm(gradient.detach(), dtype=torch.float32)
        finite = bool(torch.isfinite(norm).item())
        row["gradient_elements"] += gradient.numel()
        row["gradient_tensors"] += 1
        row["finite_gradient_tensors"] += int(finite)
        row["nonzero_gradient_tensors"] += int(finite and float(norm.item()) > 0.0)
        row["squared_gradient_norm"] += (
            float(norm.item()) ** 2 if finite else float("nan")
        )
    return {
        "layers": layer_rows,
        "unmatched_trainable_parameter_shards": unmatched_trainable,
    }


def aggregate_suffix_gradient_coverage(
    coverage_by_rank: list[dict[str, Any]],
    *,
    depth: int,
    num_layers: int,
) -> dict[str, Any]:
    layers: dict[str, Any] = {}
    for index in range(depth, num_layers):
        key = str(index)
        rows = [rank_row["layers"][key] for rank_row in coverage_by_rank]
        finite = all(
            row["finite_gradient_tensors"] == row["gradient_tensors"]
            for row in rows
        )
        squared_norm = sum(float(row["squared_gradient_norm"]) for row in rows)
        parameter_elements = sum(row["parameter_shard_elements"] for row in rows)
        gradient_elements = sum(row["gradient_elements"] for row in rows)
        missing_elements = sum(
            row["missing_gradient_parameter_shard_elements"] for row in rows
        )
        layers[key] = {
            "parameter_shard_elements": parameter_elements,
            "gradient_elements": gradient_elements,
            "missing_gradient_parameter_shard_elements": missing_elements,
            "gradient_tensors": sum(row["gradient_tensors"] for row in rows),
            "finite": finite and math.isfinite(squared_norm),
            "nonzero": math.isfinite(squared_norm) and squared_norm > 0.0,
            "complete": missing_elements == 0 and gradient_elements == parameter_elements,
            "global_gradient_norm": math.sqrt(squared_norm) if math.isfinite(squared_norm) else None,
        }
    unmatched = [
        {"rank": rank, "names": row["unmatched_trainable_parameter_shards"]}
        for rank, row in enumerate(coverage_by_rank)
        if row["unmatched_trainable_parameter_shards"]
    ]
    hard_gate = not unmatched and all(
        row["finite"] and row["nonzero"] and row["complete"]
        for row in layers.values()
    )
    return {
        "hard_gate": (
            "every suffix layer has complete gradient element coverage and a "
            "finite, nonzero aggregate gradient norm"
        ),
        "hard_gate_passed": hard_gate,
        "layers": layers,
        "unmatched_trainable_parameter_shards_by_rank": unmatched,
        "large_gradient_tensors_gathered": False,
        "document_cache_contribution_isolated": False,
    }
