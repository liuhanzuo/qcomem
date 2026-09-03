from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn


@dataclass
class ParameterSnapshot:
    name: str
    group: str
    parameter: nn.Parameter
    parameter_id: int
    shape: tuple[int, ...]
    before: torch.Tensor


def parameter_group(name: str) -> str:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
    if match is not None:
        return f"layer.{int(match.group(1))}"
    if "embed_tokens" in name:
        return "embedding"
    if "lm_head" in name:
        return "lm_head"
    if "language_model.norm" in name:
        return "final_norm"
    return "other"


def _unique_trainable_named_parameters(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> list[tuple[str, nn.Parameter]]:
    seen: set[int] = set()
    values: list[tuple[str, nn.Parameter]] = []
    for name, parameter in named_parameters:
        if not parameter.requires_grad or parameter.numel() == 0:
            continue
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        values.append((name, parameter))
    return values


def snapshot_fp32_local_shards(
    named_parameters: Iterable[tuple[str, nn.Parameter]],
) -> list[ParameterSnapshot]:
    snapshots: list[ParameterSnapshot] = []
    for name, parameter in _unique_trainable_named_parameters(named_parameters):
        if parameter.dtype != torch.float32:
            raise RuntimeError(
                f"optimizer shard {name} must be FP32, got {parameter.dtype}"
            )
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"optimizer shard {name} contains non-finite values")
        snapshots.append(
            ParameterSnapshot(
                name=name,
                group=parameter_group(name),
                parameter=parameter,
                parameter_id=id(parameter),
                shape=tuple(parameter.shape),
                before=parameter.detach().clone(),
            )
        )
    if not snapshots:
        raise RuntimeError("rank owns no non-empty FP32 optimizer shards")
    return snapshots


def _empty_stats() -> dict[str, Any]:
    return {
        "parameter_elements": 0,
        "missing_elements": 0,
        "nonfinite_elements": 0,
        "nonzero_elements": 0,
        "l1": 0.0,
        "l2_sq": 0.0,
        "max_abs": 0.0,
    }


def _accumulate_tensor(stats: dict[str, Any], tensor: torch.Tensor) -> None:
    stats["nonzero_elements"] += int(torch.count_nonzero(tensor).item())
    stats["l1"] += float(tensor.abs().sum(dtype=torch.float64).item())
    stats["l2_sq"] += float(tensor.square().sum(dtype=torch.float64).item())
    stats["max_abs"] = max(stats["max_abs"], float(tensor.abs().max().item()))


def audit_fp32_gradients(snapshots: list[ParameterSnapshot]) -> dict[str, Any]:
    total = _empty_stats()
    groups: dict[str, dict[str, Any]] = {}
    dtype_elements: dict[str, int] = {}
    for snapshot in snapshots:
        parameter = snapshot.parameter
        if id(parameter) != snapshot.parameter_id or tuple(parameter.shape) != snapshot.shape:
            raise RuntimeError(f"FSDP optimizer shard identity drifted for {snapshot.name}")
        elements = parameter.numel()
        group = groups.setdefault(snapshot.group, _empty_stats())
        total["parameter_elements"] += elements
        group["parameter_elements"] += elements
        gradient = parameter.grad
        if gradient is None:
            total["missing_elements"] += elements
            group["missing_elements"] += elements
            continue
        dtype_key = str(gradient.dtype)
        dtype_elements[dtype_key] = dtype_elements.get(dtype_key, 0) + elements
        if gradient.dtype != torch.float32:
            raise RuntimeError(
                f"gradient for {snapshot.name} must be FP32, got {gradient.dtype}"
            )
        finite = torch.isfinite(gradient)
        nonfinite = int((~finite).sum().item())
        total["nonfinite_elements"] += nonfinite
        group["nonfinite_elements"] += nonfinite
        if nonfinite:
            continue
        _accumulate_tensor(total, gradient)
        _accumulate_tensor(group, gradient)
    return {"total": total, "groups": groups, "dtype_elements": dtype_elements}


def audit_adamw_fp32_state(
    optimizer: torch.optim.Optimizer,
    snapshots: list[ParameterSnapshot],
    *,
    expected_step: int,
) -> dict[str, Any]:
    parameter_ids = {snapshot.parameter_id for snapshot in snapshots}
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
        if parameter.numel() > 0
    }
    if optimizer_ids != parameter_ids:
        raise RuntimeError("optimizer parameter set differs from local FP32 shards")
    result = {
        "parameter_elements": 0,
        "moment_elements": 0,
        "nonfinite_moment_elements": 0,
        "nonzero_moment_elements": 0,
        "step_values": {},
        "moment_dtype_elements": {},
        "all_steps_match": True,
    }
    for snapshot in snapshots:
        state = optimizer.state.get(snapshot.parameter)
        if not state:
            raise RuntimeError(f"AdamW state is missing for {snapshot.name}")
        step_value = state.get("step")
        if isinstance(step_value, torch.Tensor):
            step = int(step_value.detach().item())
        else:
            step = int(step_value)
        result["step_values"][str(step)] = result["step_values"].get(str(step), 0) + 1
        result["all_steps_match"] &= step == expected_step
        result["parameter_elements"] += snapshot.parameter.numel()
        for key in ("exp_avg", "exp_avg_sq"):
            moment = state.get(key)
            if not isinstance(moment, torch.Tensor):
                raise RuntimeError(f"AdamW {key} is missing for {snapshot.name}")
            if moment.dtype != torch.float32 or moment.shape != snapshot.parameter.shape:
                raise RuntimeError(
                    f"AdamW {key} for {snapshot.name} has invalid dtype/shape"
                )
            elements = moment.numel()
            dtype_key = str(moment.dtype)
            result["moment_dtype_elements"][dtype_key] = (
                result["moment_dtype_elements"].get(dtype_key, 0) + elements
            )
            result["moment_elements"] += elements
            finite = torch.isfinite(moment)
            result["nonfinite_moment_elements"] += int((~finite).sum().item())
            result["nonzero_moment_elements"] += int(torch.count_nonzero(moment).item())
    if not result["all_steps_match"]:
        raise RuntimeError(
            f"AdamW step gate failed: expected={expected_step}, "
            f"observed={result['step_values']}"
        )
    if result["nonfinite_moment_elements"]:
        raise RuntimeError("AdamW moments contain non-finite values")
    if result["moment_elements"] != 2 * result["parameter_elements"]:
        raise RuntimeError("AdamW moment coverage is not exactly two tensors per shard")
    if result["nonzero_moment_elements"] == 0:
        raise RuntimeError("AdamW moments are all zero")
    return result


def audit_fp32_parameter_delta(
    snapshots: list[ParameterSnapshot],
) -> dict[str, Any]:
    logical = _empty_stats()
    visible = _empty_stats()
    logical_groups: dict[str, dict[str, Any]] = {}
    visible_groups: dict[str, dict[str, Any]] = {}
    for snapshot in snapshots:
        parameter = snapshot.parameter
        if id(parameter) != snapshot.parameter_id or tuple(parameter.shape) != snapshot.shape:
            raise RuntimeError(f"FSDP optimizer shard identity drifted for {snapshot.name}")
        if parameter.dtype != torch.float32 or not torch.isfinite(parameter).all():
            raise RuntimeError(f"updated optimizer shard is invalid for {snapshot.name}")
        elements = parameter.numel()
        logical["parameter_elements"] += elements
        visible["parameter_elements"] += elements
        logical_group = logical_groups.setdefault(snapshot.group, _empty_stats())
        visible_group = visible_groups.setdefault(snapshot.group, _empty_stats())
        logical_group["parameter_elements"] += elements
        visible_group["parameter_elements"] += elements
        delta = parameter.detach() - snapshot.before
        _accumulate_tensor(logical, delta)
        _accumulate_tensor(logical_group, delta)
        visible_delta = (
            parameter.detach().to(torch.bfloat16).float()
            - snapshot.before.to(torch.bfloat16).float()
        )
        _accumulate_tensor(visible, visible_delta)
        _accumulate_tensor(visible_group, visible_delta)
    return {
        "fp32_logical": {"total": logical, "groups": logical_groups},
        "bf16_forward_visible": {"total": visible, "groups": visible_groups},
    }


def aggregate_rank_audits(values: list[dict[str, Any]]) -> dict[str, Any]:
    if not values:
        raise ValueError("rank audit list may not be empty")

    def merge_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        merged = _empty_stats()
        for item in items:
            for key in (
                "parameter_elements",
                "missing_elements",
                "nonfinite_elements",
                "nonzero_elements",
            ):
                merged[key] += int(item.get(key, 0))
            merged["l1"] += float(item.get("l1", 0.0))
            merged["l2_sq"] += float(item.get("l2_sq", 0.0))
            merged["max_abs"] = max(merged["max_abs"], float(item.get("max_abs", 0.0)))
        merged["l2"] = math.sqrt(merged.pop("l2_sq"))
        return merged

    total = merge_stats([value["total"] for value in values])
    group_names = sorted({name for value in values for name in value["groups"]})
    groups = {
        name: merge_stats(
            [value["groups"].get(name, _empty_stats()) for value in values]
        )
        for name in group_names
    }
    return {"total": total, "groups": groups}


def require_full_gradient_gate(
    audit: dict[str, Any], *, expected_parameters: int, expected_layers: int
) -> None:
    total = audit["total"]
    if total["parameter_elements"] != expected_parameters:
        raise RuntimeError("gradient audit parameter coverage mismatch")
    if total["missing_elements"] or total["nonfinite_elements"]:
        raise RuntimeError("gradient audit found missing or non-finite elements")
    if total["nonzero_elements"] == 0 or total["l2"] <= 0:
        raise RuntimeError("gradient audit found no nonzero gradient")
    required = {f"layer.{index}" for index in range(expected_layers)} | {
        "embedding",
        "final_norm",
        "lm_head",
    }
    missing_groups = sorted(required - set(audit["groups"]))
    if missing_groups:
        raise RuntimeError(f"gradient audit is missing groups: {missing_groups}")
    for name in sorted(required):
        group = audit["groups"][name]
        if group["missing_elements"] or group["nonfinite_elements"]:
            raise RuntimeError(f"gradient group {name} is incomplete or non-finite")
        if group["nonzero_elements"] == 0 or group["l2"] <= 0:
            raise RuntimeError(f"gradient group {name} is entirely zero")


def require_parameter_delta_gate(
    audit: dict[str, Any], *, expected_parameters: int, expected_layers: int
) -> None:
    for precision in ("fp32_logical", "bf16_forward_visible"):
        total = audit[precision]["total"]
        if total["parameter_elements"] != expected_parameters:
            raise RuntimeError(f"{precision} delta parameter coverage mismatch")
        if not all(math.isfinite(float(total[key])) for key in ("l1", "l2", "max_abs")):
            raise RuntimeError(f"{precision} parameter delta is non-finite")
    logical_total = audit["fp32_logical"]["total"]
    if (
        logical_total["nonzero_elements"] == 0
        or logical_total["l2"] <= 0
        or logical_total["max_abs"] <= 0
    ):
        raise RuntimeError("FP32 logical parameter delta is entirely zero")
    logical_groups = audit["fp32_logical"]["groups"]
    for index in range(expected_layers):
        name = f"layer.{index}"
        if name not in logical_groups or logical_groups[name]["nonzero_elements"] == 0:
            raise RuntimeError(f"FP32 parameter delta is zero for {name}")
