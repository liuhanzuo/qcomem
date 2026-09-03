from __future__ import annotations

import contextlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset

from qcomem_torch import TorchSplitCausalLM


DEFAULT_TARGET_SUFFIXES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 32
    alpha: float = 64.0
    dropout: float = 0.0
    target_suffixes: tuple[str, ...] = DEFAULT_TARGET_SUFFIXES

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.target_suffixes:
            raise ValueError("at least one LoRA target suffix is required")


class LoRALinear(nn.Module):
    """A small, dependency-free LoRA wrapper for a frozen ``nn.Linear``.

    The base layer is retained verbatim.  Adapter parameters stay FP32; only
    the small rank-r update is computed in FP32 and cast back to the base
    output dtype.  Optimizer-state dtype is implementation-dependent and is
    recorded from the actual checkpoint, not assumed here.
    """

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.enabled = True
        self.lora_a = nn.Parameter(
            torch.empty(
                rank,
                base.in_features,
                device=base.weight.device,
                dtype=torch.float32,
            )
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base.out_features,
                rank,
                device=base.weight.device,
                dtype=torch.float32,
            )
        )
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.base(inputs)
        if not self.enabled:
            return output
        adapter_inputs = self.dropout(inputs).float()
        update = F.linear(F.linear(adapter_inputs, self.lora_a), self.lora_b)
        return output + (update * self.scaling).to(output.dtype)


def _replace_target_linears(
    module: nn.Module,
    *,
    prefix: str,
    config: LoRAConfig,
    installed: list[str],
) -> None:
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, LoRALinear):
            continue
        if isinstance(child, nn.Linear) and child_name.endswith(config.target_suffixes):
            setattr(
                module,
                child_name,
                LoRALinear(
                    child,
                    rank=config.rank,
                    alpha=config.alpha,
                    dropout=config.dropout,
                ),
            )
            installed.append(full_name)
        else:
            _replace_target_linears(
                child,
                prefix=full_name,
                config=config,
                installed=installed,
            )


def find_suffix_lora_targets(
    layers: Sequence[nn.Module],
    *,
    depth: int,
    target_suffixes: tuple[str, ...],
) -> list[str]:
    if depth < 0 or depth > len(layers):
        raise ValueError("depth is outside the model layer range")
    return [
        f"layers.{layer_index}.{name}"
        for layer_index in range(depth, len(layers))
        for name, module in layers[layer_index].named_modules()
        if name
        and isinstance(module, nn.Linear)
        and name.rsplit(".", 1)[-1].endswith(target_suffixes)
    ]


def estimate_suffix_lora_parameters(
    layers: Sequence[nn.Module],
    *,
    depth: int,
    target_suffixes: tuple[str, ...],
    rank: int,
) -> int:
    if rank < 1:
        raise ValueError("LoRA rank must be positive")
    return sum(
        rank * (module.in_features + module.out_features)
        for layer in layers[depth:]
        for name, module in layer.named_modules()
        if name
        and isinstance(module, nn.Linear)
        and name.rsplit(".", 1)[-1].endswith(target_suffixes)
    )


def install_suffix_lora(
    model: nn.Module,
    layers: Sequence[nn.Module],
    *,
    depth: int,
    config: LoRAConfig,
) -> list[str]:
    """Freeze the complete model and install LoRA only in layers >= depth."""

    if depth < 0 or depth > len(layers):
        raise ValueError("depth is outside the model layer range")
    model.requires_grad_(False)
    installed: list[str] = []
    for layer_index in range(depth, len(layers)):
        _replace_target_linears(
            layers[layer_index],
            prefix=f"layers.{layer_index}",
            config=config,
            installed=installed,
        )
    if not installed:
        raise ValueError(
            "no suffix Linear modules matched LoRA targets: "
            + ",".join(config.target_suffixes)
        )
    return installed


def iter_lora_modules(model: nn.Module) -> Iterator[LoRALinear]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


def set_lora_enabled(model: nn.Module, enabled: bool) -> None:
    for module in iter_lora_modules(model):
        module.enabled = enabled


@contextlib.contextmanager
def lora_disabled(model: nn.Module) -> Iterator[None]:
    modules = list(iter_lora_modules(model))
    previous = [module.enabled for module in modules]
    try:
        for module in modules:
            module.enabled = False
        yield
    finally:
        for module, enabled in zip(modules, previous):
            module.enabled = enabled


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name.endswith(("lora_a", "lora_b"))
    }


def load_lora_state_dict(
    model: nn.Module, state: dict[str, torch.Tensor]
) -> None:
    expected = set(lora_state_dict(model))
    provided = set(state)
    if expected != provided:
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        raise ValueError(
            f"LoRA checkpoint mismatch; missing={missing}, unexpected={unexpected}"
        )
    current = model.state_dict()
    with torch.no_grad():
        for name, value in state.items():
            current[name].copy_(value.to(device=current[name].device))


def load_inference_lora_checkpoint(
    model: nn.Module,
    layers: Sequence[nn.Module],
    path: Path,
) -> dict[str, Any]:
    """Install and load a trusted training checkpoint for downstream inference."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "qcomem_suffix_lora_v1":
        raise ValueError(f"unsupported LoRA checkpoint format in {path}")
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("LoRA checkpoint is missing metadata")
    adapter = metadata.get("adapter", {})
    raw_config = adapter.get("config", {})
    semantics = metadata.get("semantics", {})
    depth = semantics.get("depth")
    if not isinstance(depth, int):
        raise ValueError("LoRA checkpoint metadata is missing integer split depth")
    config = LoRAConfig(
        rank=int(raw_config["rank"]),
        alpha=float(raw_config["alpha"]),
        dropout=float(raw_config.get("dropout", 0.0)),
        target_suffixes=tuple(raw_config["target_suffixes"]),
    )
    installed = install_suffix_lora(model, layers, depth=depth, config=config)
    expected_installed = adapter.get("installed_modules")
    if expected_installed is not None and installed != list(expected_installed):
        raise ValueError("installed LoRA modules differ from checkpoint metadata")
    load_lora_state_dict(model, payload["lora"])
    model.eval()
    set_lora_enabled(model, False)
    return {
        "format": payload["format"],
        "step": int(payload["step"]),
        "adapter": adapter,
        "semantics": semantics,
    }


def assert_replay_adapter_semantics(
    checkpoint_metadata: dict[str, Any],
    *,
    depth: int,
    residual_bits: int,
    attention_bits: int | None,
    linear_bits: int | None,
    cache_layer_bits: Sequence[int] | None,
) -> None:
    """Hard-check that a quant adapter is used with its trained replay policy."""

    semantics = checkpoint_metadata.get("semantics", checkpoint_metadata)
    if semantics.get("mode") != "quant":
        raise ValueError(
            "replay quant config requires a quantization-conditioned adapter; "
            f"checkpoint mode is {semantics.get('mode')!r}"
        )
    if semantics.get("depth") != depth:
        raise ValueError(
            f"adapter depth {semantics.get('depth')!r} != target depth {depth}"
        )
    store = semantics.get("store")
    if not isinstance(store, dict):
        raise ValueError("quant adapter checkpoint is missing store semantics")
    expected = {
        "residual_bits": residual_bits,
        "attention_bits": attention_bits,
        "linear_bits": linear_bits,
        "cache_layer_bits": (
            tuple(cache_layer_bits) if cache_layer_bits is not None else None
        ),
    }
    actual = {
        "residual_bits": store.get("residual_bits"),
        "attention_bits": store.get("attention_bits"),
        "linear_bits": store.get("linear_bits"),
        "cache_layer_bits": (
            tuple(store["cache_layer_bits"])
            if store.get("cache_layer_bits") is not None
            else None
        ),
    }
    if actual != expected:
        raise ValueError(
            "adapter replay policy does not match target config: "
            f"checkpoint={actual}, target={expected}"
        )


def assert_interface_adapter_semantics(
    checkpoint_metadata: dict[str, Any],
    *,
    depth: int,
    chunk_size: int,
    overlap: int,
) -> None:
    semantics = checkpoint_metadata.get("semantics", checkpoint_metadata)
    store = semantics.get("store")
    expected = {
        "mode": "interface",
        "depth": depth,
        "kind": "residual_only_chunk_local",
        "residual_bits": 16,
        "lower_cache_stored": False,
        "chunk_size": chunk_size,
        "overlap": overlap,
    }
    actual = {
        "mode": semantics.get("mode"),
        "depth": semantics.get("depth"),
        "kind": store.get("kind") if isinstance(store, dict) else None,
        "residual_bits": store.get("residual_bits") if isinstance(store, dict) else None,
        "lower_cache_stored": (
            store.get("lower_cache_stored") if isinstance(store, dict) else None
        ),
        "chunk_size": store.get("chunk_size") if isinstance(store, dict) else None,
        "overlap": store.get("overlap") if isinstance(store, dict) else None,
    }
    if actual != expected:
        raise ValueError(
            "Interface adapter semantics do not match target chunk config: "
            f"checkpoint={actual}, target={expected}"
        )


def adapter_metadata(
    model: nn.Module,
    *,
    installed_modules: Sequence[str],
    config: LoRAConfig,
) -> dict[str, Any]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return {
        "config": asdict(config),
        "installed_modules": list(installed_modules),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "trainable_nbytes": sum(
            parameter.numel() * parameter.element_size() for parameter in parameters
        ),
    }


def lora_gradient_coverage(model: nn.Module) -> dict[str, Any]:
    """Audit every installed A/B gradient without assuming both are nonzero.

    At initialization LoRA-B is zero, so the first-step LoRA-A gradient is
    expected to be exactly zero.  The hard gate is therefore presence and
    finiteness for both tensors in every module; nonzero coverage is recorded
    separately rather than used to manufacture a false failure.
    """

    modules: dict[str, Any] = {}
    groups: dict[str, dict[str, int]] = {}
    for name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        group = name.rsplit(".", 1)[-1]
        parameter_rows: dict[str, Any] = {}
        for parameter_name, parameter in (("lora_a", module.lora_a), ("lora_b", module.lora_b)):
            gradient = parameter.grad
            present = gradient is not None
            finite = bool(present and torch.isfinite(gradient).all().item())
            nonzero = bool(present and torch.count_nonzero(gradient).item() > 0)
            parameter_rows[parameter_name] = {
                "present": present,
                "finite": finite,
                "nonzero": nonzero,
                "elements": parameter.numel(),
                "grad_norm": (
                    float(torch.linalg.vector_norm(gradient.detach(), dtype=torch.float32).item())
                    if present
                    else None
                ),
            }
        module_finite = all(row["present"] and row["finite"] for row in parameter_rows.values())
        module_any_nonzero = any(row["nonzero"] for row in parameter_rows.values())
        modules[name] = {
            "group": group,
            "parameters": parameter_rows,
            "all_parameters_have_finite_grad": module_finite,
            "any_parameter_nonzero": module_any_nonzero,
        }
        group_row = groups.setdefault(
            group,
            {"modules": 0, "finite_modules": 0, "nonzero_modules": 0},
        )
        group_row["modules"] += 1
        group_row["finite_modules"] += int(module_finite)
        group_row["nonzero_modules"] += int(module_any_nonzero)
    return {
        "module_count": len(modules),
        "finite_module_count": sum(
            int(row["all_parameters_have_finite_grad"]) for row in modules.values()
        ),
        "nonzero_module_count": sum(
            int(row["any_parameter_nonzero"]) for row in modules.values()
        ),
        "all_modules_have_finite_grad": bool(modules)
        and all(row["all_parameters_have_finite_grad"] for row in modules.values()),
        "groups": groups,
        "modules": modules,
        "document_cache_contribution_isolated": False,
        "document_cache_contribution_note": (
            "This end-to-end suffix gradient covers document prefill and query "
            "continuation jointly; it does not isolate the document-cache contribution."
        ),
    }


def bidirectional_topk_kl(
    student_topk_logits: torch.Tensor,
    teacher_topk_logits: torch.Tensor,
    *,
    forward_weight: float = 0.6,
    reverse_weight: float = 0.4,
    temperature: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """KL on the teacher top-k support, averaged over query positions."""

    if student_topk_logits.shape != teacher_topk_logits.shape:
        raise ValueError("student and teacher top-k logits must have equal shape")
    if student_topk_logits.ndim < 2:
        raise ValueError("top-k logits must include positions and candidates")
    if forward_weight < 0 or reverse_weight < 0:
        raise ValueError("KL weights must be non-negative")
    if forward_weight + reverse_weight <= 0:
        raise ValueError("at least one KL direction must have positive weight")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    teacher_logp = F.log_softmax(teacher_topk_logits.float() / temperature, dim=-1)
    student_logp = F.log_softmax(student_topk_logits.float() / temperature, dim=-1)
    teacher_p = teacher_logp.exp()
    student_p = student_logp.exp()
    forward = torch.sum(teacher_p * (teacher_logp - student_logp), dim=-1).mean()
    reverse = torch.sum(student_p * (student_logp - teacher_logp), dim=-1).mean()
    loss = (forward_weight * forward + reverse_weight * reverse) * temperature**2
    return loss, {
        "forward_kl": forward.detach(),
        "reverse_kl": reverse.detach(),
    }


@dataclass
class TrainingWindow:
    document_ids: torch.Tensor
    query_ids: torch.Tensor
    source_id: str
    teacher_topk_indices: torch.Tensor | None = None
    teacher_topk_logits: torch.Tensor | None = None


class PG19WindowDataset(Dataset[TrainingWindow]):
    """Load tokenized or raw-text JSONL and create context/query windows.

    Accepted records are either ``{"text": ...}`` or
    ``{"document_ids": [...], "query_ids": [...]}``.  The latter may carry
    offline ``teacher_topk_indices`` and ``teacher_topk_logits`` arrays.
    """

    def __init__(
        self,
        path: Path,
        tokenizer: Any,
        *,
        context_tokens: int,
        query_tokens: int,
        stride: int,
        limit: int | None = None,
        max_windows_per_record: int | None = None,
    ) -> None:
        if context_tokens < 1 or query_tokens < 1 or stride < 1:
            raise ValueError("context_tokens, query_tokens, and stride must be positive")
        if max_windows_per_record is not None and max_windows_per_record < 1:
            raise ValueError("max_windows_per_record must be positive")
        self.windows: list[TrainingWindow] = []
        with path.open() as stream:
            for line_index, line in enumerate(stream):
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("dataset") in {"qasper", "2wikimqa"} or (
                    record.get("_source_index") is not None
                    and int(record["_source_index"]) >= 68
                ):
                    raise ValueError(
                        "LongBench QA rows are evaluation data, not PG-19 LoRA "
                        "training windows"
                    )
                source_id = str(record.get("id", line_index))
                if "document_ids" in record and "query_ids" in record:
                    indices = record.get("teacher_topk_indices")
                    logits = record.get("teacher_topk_logits")
                    self.windows.append(
                        TrainingWindow(
                            document_ids=torch.tensor(record["document_ids"], dtype=torch.long),
                            query_ids=torch.tensor(record["query_ids"], dtype=torch.long),
                            source_id=source_id,
                            teacher_topk_indices=(
                                torch.tensor(indices, dtype=torch.long)
                                if indices is not None
                                else None
                            ),
                            teacher_topk_logits=(
                                torch.tensor(logits, dtype=torch.float32)
                                if logits is not None
                                else None
                            ),
                        )
                    )
                elif "text" in record:
                    window_length = context_tokens + query_tokens
                    encode_kwargs: dict[str, Any] = {"add_special_tokens": False}
                    if max_windows_per_record is not None:
                        encode_kwargs.update(
                            {
                                "max_length": window_length
                                + stride * (max_windows_per_record - 1),
                                "truncation": True,
                            }
                        )
                    try:
                        ids = tokenizer.encode(record["text"], **encode_kwargs)
                    except TypeError:
                        # Minimal test tokenizers may not expose the Hugging Face
                        # truncation kwargs.  Keep production semantics while
                        # allowing dependency-free unit tests.
                        ids = tokenizer.encode(
                            record["text"], add_special_tokens=False
                        )[: encode_kwargs.get("max_length")]
                    record_windows = 0
                    for start in range(0, max(0, len(ids) - window_length + 1), stride):
                        split = start + context_tokens
                        end = split + query_tokens
                        self.windows.append(
                            TrainingWindow(
                                document_ids=torch.tensor(ids[start:split], dtype=torch.long),
                                query_ids=torch.tensor(ids[split:end], dtype=torch.long),
                                source_id=f"{source_id}:{start}",
                            )
                        )
                        record_windows += 1
                        if limit is not None and len(self.windows) >= limit:
                            break
                        if (
                            max_windows_per_record is not None
                            and record_windows >= max_windows_per_record
                        ):
                            break
                else:
                    raise ValueError(
                        f"record {line_index} needs text or document_ids/query_ids"
                    )
                if limit is not None and len(self.windows) >= limit:
                    break
        if not self.windows:
            raise ValueError(f"no complete training windows found in {path}")

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> TrainingWindow:
        return self.windows[index]


def single_window_collate(batch: list[TrainingWindow]) -> TrainingWindow:
    if len(batch) != 1:
        raise ValueError("Q-CoMem training currently requires per-rank batch size 1")
    return batch[0]


@dataclass(frozen=True)
class ReplayQuantConfig:
    residual_bits: int = 4
    attention_bits: int = 8
    linear_bits: int = 8
    cache_layer_bits: tuple[int, ...] | None = None
    group_size: int = 64


def quant_student_suffix_hidden(
    adapter: TorchSplitCausalLM,
    *,
    depth: int,
    document_residual: torch.Tensor,
    query_residual: torch.Tensor,
    execution: str,
    return_cache_audit: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Return every query-position hidden state for one quant student.

    The helper is intentionally independent of the loss so tiny tests can
    verify cache boundaries and gradient flow without a real language model.
    """

    audited_executions = {
        "detached-document-cache",
        "native-functional-cache",
    }
    if return_cache_audit and execution not in audited_executions:
        raise ValueError(
            "cache immutability audit is only defined for detached-document-cache "
            "or native-functional-cache"
        )
    if execution == "merged-uncached":
        hidden = torch.cat([document_residual, query_residual], dim=1)
        hidden = adapter._run_layers(hidden, depth, adapter.num_layers)
        return adapter.language_model.norm(hidden[:, -query_residual.shape[1] :, :])
    if execution not in {
        "cached-two-stage",
        "detached-document-cache",
        "native-functional-cache",
    }:
        raise ValueError(
            "execution must be merged-uncached, cached-two-stage, or "
            "detached-document-cache, or native-functional-cache"
        )
    document_cache = adapter.make_cache()
    native_install = None
    if execution == "native-functional-cache":
        from qcomem_qwen35_native_cache import (
            install_native_functional_linear_cache,
        )

        native_install = install_native_functional_linear_cache(
            document_cache, adapter.config
        )
    prefill_context = (
        torch.no_grad()
        if execution == "detached-document-cache"
        else contextlib.nullcontext()
    )
    with prefill_context:
        adapter._run_layers(
            document_residual,
            depth,
            adapter.num_layers,
            past_key_values=document_cache,
            position_offset=0,
        )
    cache_audit = None
    if execution == "detached-document-cache":
        before = (
            cache_tensor_records(document_cache) if return_cache_audit else None
        )
        suffix_cache = detach_cache_tensors(document_cache)
        if before is not None:
            detached = cache_tensor_records(suffix_cache)
            original_storage = {
                row["storage_ptr"]
                for row in before.values()
                if row["storage_ptr"] is not None
            }
            detached_storage = {
                row["storage_ptr"]
                for row in detached.values()
                if row["storage_ptr"] is not None
            }
            cache_audit = {
                "execution": "detached-document-cache",
                "document_cache_tensor_count": len(before),
                "detached_cache_tensor_count": len(detached),
                "cache_structure_paths_match": set(before) == set(detached),
                "detached_cache_storage_disjoint": original_storage.isdisjoint(
                    detached_storage
                ),
                "detached_cache_all_tensors_grad_free": all(
                    not row["requires_grad"] and not row["has_grad_fn"]
                    for row in detached.values()
                ),
                "original_version_counters_available": all(
                    row["version"] is not None for row in before.values()
                ),
                "original_versions_before_query": {
                    path: row["version"] for path, row in before.items()
                },
            }
    else:
        suffix_cache = document_cache
        if execution == "native-functional-cache" and return_cache_audit:
            before_refs = cache_tensor_references(document_cache)
            before = cache_tensor_records(document_cache)
            cache_audit = {
                "execution": execution,
                "document_cache_tensor_count": len(before),
                "original_version_counters_available": all(
                    row["version"] is not None for row in before.values()
                ),
                "original_versions_before_query": {
                    path: row["version"] for path, row in before.items()
                },
                "original_storage_before_query": {
                    path: row["storage_ptr"] for path, row in before.items()
                },
            }
    hidden = adapter._run_layers(
        query_residual,
        depth,
        adapter.num_layers,
        past_key_values=suffix_cache,
        position_offset=document_residual.shape[1],
    )
    hidden = adapter.language_model.norm(hidden)
    if cache_audit is None:
        return hidden
    after = cache_tensor_records(document_cache)
    versions_before = cache_audit.pop("original_versions_before_query")
    if execution == "native-functional-cache":
        original_storage = cache_audit.pop("original_storage_before_query")
        old_versions_unchanged = all(
            _safe_tensor_version(before_refs[path]) == version
            for path, version in versions_before.items()
        )
        paths_unchanged = set(after) == set(versions_before)
        cache_audit.update(
            {
                "original_cache_paths_unchanged": paths_unchanged,
                "original_cache_versions_unchanged": old_versions_unchanged,
                "all_cache_paths_rebound": paths_unchanged
                and all(
                    after[path]["storage_ptr"] != original_storage[path]
                    for path in after
                ),
                "native_linear_layer_count": (
                    None
                    if native_install is None
                    else native_install.installed_linear_layers
                ),
                "native_linear_layer_indices": (
                    None
                    if native_install is None
                    else native_install.linear_layer_indices
                ),
                "native_full_attention_layer_indices": (
                    None
                    if native_install is None
                    else native_install.full_attention_layer_indices
                ),
                "query_positions_expected": int(query_residual.shape[1]),
                "query_positions_observed": int(hidden.shape[1]),
            }
        )
    else:
        cache_audit.update(
            {
                "original_cache_paths_unchanged": set(after) == set(versions_before),
                "original_cache_versions_unchanged": set(after) == set(versions_before)
                and all(
                    after[path]["version"] == version
                    for path, version in versions_before.items()
                ),
                "query_positions_expected": int(query_residual.shape[1]),
                "query_positions_observed": int(hidden.shape[1]),
            }
        )
    common_gate = bool(
        cache_audit["document_cache_tensor_count"] > 0
        and cache_audit["original_version_counters_available"]
        and cache_audit["original_cache_paths_unchanged"]
        and cache_audit["original_cache_versions_unchanged"]
        and cache_audit["query_positions_observed"]
        == cache_audit["query_positions_expected"]
    )
    if execution == "native-functional-cache":
        mode_gate = bool(
            cache_audit["all_cache_paths_rebound"]
            and cache_audit["native_linear_layer_count"]
            == len(cache_audit["native_linear_layer_indices"])
        )
    else:
        mode_gate = bool(
            cache_audit["cache_structure_paths_match"]
            and cache_audit["detached_cache_storage_disjoint"]
            and cache_audit["detached_cache_all_tensors_grad_free"]
        )
    cache_audit["hard_gate_passed"] = common_gate and mode_gate
    return hidden, cache_audit


def cache_tensor_records(value: Any) -> dict[str, dict[str, Any]]:
    """Return stable records for each unique tensor reachable from a cache."""

    records: dict[str, dict[str, Any]] = {}
    seen_objects: set[int] = set()
    seen_tensors: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            tensor_id = id(item)
            if tensor_id in seen_tensors:
                return
            seen_tensors.add(tensor_id)
            storage_ptr = (
                int(item.untyped_storage().data_ptr()) if item.numel() else None
            )
            try:
                version = int(item._version)
            except RuntimeError:
                version = None
            records[path] = {
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "device": str(item.device),
                "version": version,
                "storage_ptr": storage_ptr,
                "requires_grad": bool(item.requires_grad),
                "has_grad_fn": item.grad_fn is not None,
            }
            return
        if isinstance(item, (str, bytes, int, float, bool, type(None))):
            return
        object_id = id(item)
        if object_id in seen_objects:
            return
        seen_objects.add(object_id)
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}[{key!r}]")
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
            return
        if hasattr(item, "__dict__"):
            for name, child in vars(item).items():
                visit(child, f"{path}.{name}")

    visit(value, "cache")
    return records


def _safe_tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor._version)
    except RuntimeError:
        return None


def cache_tensor_references(value: Any) -> dict[str, torch.Tensor]:
    """Return the same unique tensor paths as ``cache_tensor_records``.

    Keeping the old object references lets the native-functional gate verify
    their version counters after the cache has rebound its public fields.
    """

    references: dict[str, torch.Tensor] = {}
    seen_objects: set[int] = set()
    seen_tensors: set[int] = set()

    def visit(item: Any, path: str) -> None:
        if isinstance(item, torch.Tensor):
            if id(item) not in seen_tensors:
                seen_tensors.add(id(item))
                references[path] = item
            return
        if isinstance(item, (str, bytes, int, float, bool, type(None))):
            return
        if id(item) in seen_objects:
            return
        seen_objects.add(id(item))
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}[{key!r}]")
        elif isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif hasattr(item, "__dict__"):
            for name, child in vars(item).items():
                visit(child, f"{path}.{name}")

    visit(value, "cache")
    return references


def detach_cache_tensors(value: Any, memo: dict[int, Any] | None = None) -> Any:
    """Detach every tensor in a mutable cache while preserving its structure."""

    if memo is None:
        memo = {}
    object_id = id(value)
    if object_id in memo:
        return memo[object_id]
    if isinstance(value, torch.Tensor):
        # A plain detach still shares storage and a version counter with the
        # mutable prefill cache. Clone so query-side in-place updates cannot
        # invalidate a tensor saved by the document prefill graph.
        result = value.detach().clone()
        memo[object_id] = result
        return result
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        memo[object_id] = result
        result.update(
            (key, detach_cache_tensors(item, memo)) for key, item in value.items()
        )
        return result
    if isinstance(value, list):
        result: list[Any] = []
        memo[object_id] = result
        result.extend(detach_cache_tensors(item, memo) for item in value)
        return result
    if isinstance(value, tuple):
        result = tuple(detach_cache_tensors(item, memo) for item in value)
        memo[object_id] = result
        return result
    if hasattr(value, "__dict__"):
        import copy

        result = copy.copy(value)
        memo[object_id] = result
        for name, item in vars(value).items():
            setattr(result, name, detach_cache_tensors(item, memo))
        return result
    memo[object_id] = value
    return value


def functional_cache_capability_gate() -> dict[str, Any]:
    return {
        "requested_execution": "functional-cache",
        "implemented": True,
        "capability_gate_passed": False,
        "native_qwen35_cache_adapter": {
            "implemented": True,
            "source": "gpu/qcomem_qwen35_native_cache.py",
            "mechanism": (
                "retain native Qwen3.5/FLA model kernels and replace only "
                "linear-cache conv/recurrent copy_ writes with tensor rebinding"
            ),
            "full_attention_training_cache": (
                "Transformers DynamicLayer torch.cat assignment (out of place)"
            ),
            "real_config_structure_gate_passed": True,
            "real_model_backward_gate_passed": False,
        },
        "tiny_reference": {
            "implemented": True,
            "source": "gpu/qcomem_functional_cache.py",
            "test": "gpu/test_qcomem_functional_cache.py",
            "forward_hidden_and_state_parity_passed": True,
            "input_and_parameter_gradient_parity_passed": True,
            "out_of_place_storage_and_version_gate_passed": True,
            "scope": (
                "minimal full-attention KV plus GDN-like conv/recurrent state; "
                "the GDN equations are not a numerical reproduction of Qwen3.5"
            ),
        },
        "qwen35_integration_implemented": True,
        "qwen35_real_model_gate_passed": False,
        "requirements": [
            "suffix layers return a new immutable cache rather than mutating DynamicCache",
            "GatedDeltaNet recurrent/conv states have out-of-place differentiable updates",
            "document-prefill and full-query continuation backward passes are version-safe",
            "real Qwen3.5 all-module gradient coverage passes",
        ],
        "reason": (
            "The native functional-cache adapter and tiny forward/backward tests are "
            "implemented. Historical trials 1830867/1832364 used mutable or detached "
            "copy_ paths and failed; a new real Qwen3.5 8-rank backward, all-module "
            "gradient/update, cache-version, and same-boundary semantic gate is still "
            "required before training results may be claimed."
        ),
    }


def cached_two_stage_autograd_capability_gate() -> dict[str, Any]:
    return {
        "requested_execution": "cached-two-stage",
        "implemented": True,
        "real_model_forward_passed": True,
        "real_model_backward_passed": False,
        "capability_gate_passed": False,
        "evidence_trial": 1830867,
        "failure": (
            "8/8 ranks: mutable cache inplace version mismatch during loss.backward; "
            "CopyBackwards tensor [1,32,128,128] was version 2, expected 1"
        ),
        "not_oom": True,
        "not_nccl_root_cause": True,
    }


class CoMemLoRADistillation(nn.Module):
    """Online one-model teacher/student distillation for CoMem adapters.

    The teacher is evaluated under ``no_grad`` with LoRA disabled, then only
    its top-k logits survive.  The student sees the same frozen Write/lower
    path and gradients enter suffix LoRA parameters only.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        mode: str,
        depth: int,
        top_k: int,
        chunk_size: int,
        overlap: int,
        teacher_kind: str,
        teacher_source: str,
        quant: ReplayQuantConfig,
        forward_weight: float,
        reverse_weight: float,
        temperature: float,
        student_suffix_execution: str = "merged-uncached",
    ) -> None:
        super().__init__()
        if mode not in {"interface", "quant"}:
            raise ValueError("mode must be interface or quant")
        if teacher_kind not in {"dense", "q16_replay"}:
            raise ValueError("teacher_kind must be dense or q16_replay")
        if mode == "quant" and teacher_kind != "q16_replay":
            raise ValueError("quant mode requires a Q16 replay teacher")
        if teacher_source not in {"online", "offline"}:
            raise ValueError("teacher_source must be online or offline")
        if student_suffix_execution not in {
            "merged-uncached",
            "cached-two-stage",
            "detached-document-cache",
            "native-functional-cache",
        }:
            raise ValueError(
                "unsupported student suffix execution"
            )
        if mode != "quant" and student_suffix_execution != "merged-uncached":
            raise ValueError("cached-two-stage student execution is only defined for quant mode")
        self.model = model
        self.adapter = TorchSplitCausalLM(model)
        self.mode = mode
        self.depth = depth
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.teacher_kind = teacher_kind
        self.teacher_source = teacher_source
        self.quant = quant
        self.forward_weight = forward_weight
        self.reverse_weight = reverse_weight
        self.temperature = temperature
        self.student_suffix_execution = student_suffix_execution
        self.last_detached_cache_audit: dict[str, Any] | None = None

    def _lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.adapter.lm_head(hidden)

    def _selected_lm_logits(
        self, hidden: torch.Tensor, indices: torch.Tensor
    ) -> torch.Tensor:
        head = self.adapter.lm_head
        if not isinstance(head, nn.Linear):
            full = head(hidden)
            return torch.gather(full, dim=-1, index=indices)
        weights = head.weight[indices]
        selected = torch.einsum("bqh,bqkh->bqk", hidden, weights)
        if head.bias is not None:
            selected = selected + head.bias[indices]
        return selected

    def _suffix_hidden(self, residuals: Sequence[torch.Tensor], query_length: int) -> torch.Tensor:
        hidden = torch.cat(list(residuals), dim=1)
        hidden = self.adapter._run_layers(
            hidden, self.depth, self.adapter.num_layers
        )
        hidden = self.adapter.language_model.norm(hidden[:, -query_length:, :])
        return hidden

    def _quant_student_suffix_hidden(
        self,
        document_residual: torch.Tensor,
        query_residual: torch.Tensor,
    ) -> torch.Tensor:
        """Run the quant student with an explicitly selected suffix boundary.

        ``cached-two-stage`` matches deployment prefill semantics: the suffix
        first consumes the document residual into a fresh cache, then consumes
        the complete query residual once at ``position_offset=document_length``.
        No lower query token is re-chunked here.  Keeping this path separate
        makes cache/autograd failures visible instead of silently falling back
        to the historical merged uncached approximation.
        """

        result = quant_student_suffix_hidden(
            self.adapter,
            depth=self.depth,
            document_residual=document_residual,
            query_residual=query_residual,
            execution=self.student_suffix_execution,
            return_cache_audit=(
                self.student_suffix_execution
                in {"detached-document-cache", "native-functional-cache"}
            ),
        )
        if isinstance(result, tuple):
            hidden, audit = result
            self.last_detached_cache_audit = audit
            return hidden
        self.last_detached_cache_audit = None
        return result

    def _dense_teacher(self, document: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
        full = torch.cat([document, query], dim=1)
        output = self.adapter.language_model(input_ids=full, use_cache=False)
        return self._lm_logits(output.last_hidden_state[:, -query.shape[1] :, :])

    def _q16_teacher_and_state(
        self, document: torch.Tensor, query: torch.Tensor
    ) -> tuple[torch.Tensor, Any]:
        raw = self.adapter.write_lower_replay(document, self.depth)
        local = raw.fork()
        query_residual = self.adapter.continue_lower_replay(local, query)
        if self.student_suffix_execution == "native-functional-cache":
            # The teacher uses the ordinary mutable inference cache under the
            # surrounding no_grad context, but exactly the same document/query
            # caller boundary as deployment and the functional student.
            hidden = quant_student_suffix_hidden(
                self.adapter,
                depth=self.depth,
                document_residual=local.document_residual,
                query_residual=query_residual,
                execution="cached-two-stage",
            )
        else:
            hidden = self._suffix_hidden(
                [local.document_residual, query_residual], query.shape[1]
            )
        return self._lm_logits(hidden), raw

    @staticmethod
    def _offline_targets(
        indices: torch.Tensor | None,
        logits: torch.Tensor | None,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if indices is None or logits is None:
            raise ValueError("offline teacher requires top-k indices and logits")
        if indices.shape != logits.shape:
            raise ValueError("offline teacher indices/logits shape mismatch")
        if indices.ndim == 2:
            indices = indices.unsqueeze(0)
            logits = logits.unsqueeze(0)
        return indices.to(device=device), logits.to(device=device)

    def forward(
        self,
        document_ids: torch.Tensor,
        query_ids: torch.Tensor,
        teacher_topk_indices: torch.Tensor | None = None,
        teacher_topk_logits: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | int]:
        if document_ids.ndim == 1:
            document_ids = document_ids.unsqueeze(0)
        if query_ids.ndim == 1:
            query_ids = query_ids.unsqueeze(0)
        raw_state = None
        if self.teacher_source == "offline":
            indices, teacher_values = self._offline_targets(
                teacher_topk_indices,
                teacher_topk_logits,
                device=document_ids.device,
            )
        else:
            with torch.no_grad(), lora_disabled(self.model):
                if self.teacher_kind == "dense":
                    teacher_logits = self._dense_teacher(document_ids, query_ids)
                else:
                    teacher_logits, raw_state = self._q16_teacher_and_state(
                        document_ids, query_ids
                    )
                teacher_values, indices = torch.topk(
                    teacher_logits.float(), k=self.top_k, dim=-1
                )
                del teacher_logits

        if self.mode == "interface":
            with torch.no_grad():
                document_residuals = self.adapter.chunk_local_write_parts(
                    document_ids,
                    self.depth,
                    chunk_size=self.chunk_size,
                    overlap=self.overlap,
                )
                query_residual = self.adapter.run_to_depth(query_ids, self.depth)
                persistent_nbytes = sum(
                    item.numel() * item.element_size() for item in document_residuals
                )
            hidden = self._suffix_hidden(
                [*document_residuals, query_residual], query_ids.shape[1]
            )
        else:
            if raw_state is None:
                with torch.no_grad():
                    raw_state = self.adapter.write_lower_replay(document_ids, self.depth)
            with torch.no_grad():
                packed = raw_state.quantize(
                    bits=self.quant.residual_bits,
                    attention_bits=self.quant.attention_bits,
                    linear_bits=self.quant.linear_bits,
                    cache_layer_bits=self.quant.cache_layer_bits,
                    group_size=self.quant.group_size,
                )
                local = packed.fork()
                query_residual = self.adapter.continue_lower_replay(local, query_ids)
                document_residual = local.document_residual
                persistent_nbytes = packed.stored_nbytes
            # Existing replay writes intentionally use torch.inference_mode.
            # Suffix LoRA backward needs to save these activations, which is
            # forbidden for inference tensors.  Clone outside that context to
            # create ordinary frozen activations without opening lower grads.
            document_residual = document_residual.clone()
            query_residual = query_residual.clone()
            hidden = self._quant_student_suffix_hidden(
                document_residual, query_residual
            )
        student_values = self._selected_lm_logits(hidden, indices)
        loss, metrics = bidirectional_topk_kl(
            student_values,
            teacher_values,
            forward_weight=self.forward_weight,
            reverse_weight=self.reverse_weight,
            temperature=self.temperature,
        )
        return {
            "loss": loss,
            "forward_kl": metrics["forward_kl"],
            "reverse_kl": metrics["reverse_kl"],
            "persistent_nbytes": persistent_nbytes,
        }


def training_semantics_metadata(
    *,
    mode: str,
    depth: int,
    teacher_kind: str,
    teacher_source: str,
    quant: ReplayQuantConfig,
    chunk_size: int,
    overlap: int,
    student_suffix_execution: str = "merged-uncached",
) -> dict[str, Any]:
    if mode == "interface":
        store = {
            "kind": "residual_only_chunk_local",
            "residual_bits": 16,
            "lower_cache_stored": False,
            "chunk_size": chunk_size,
            "overlap": overlap,
        }
    elif mode == "quant":
        store = {
            "kind": "packed_residual_and_lower_replay_state",
            **asdict(quant),
        }
    else:
        raise ValueError("mode must be interface or quant")
    if student_suffix_execution not in {
        "merged-uncached",
        "cached-two-stage",
        "detached-document-cache",
        "native-functional-cache",
    }:
        raise ValueError(
            "unsupported student suffix execution"
        )
    if mode != "quant" and student_suffix_execution != "merged-uncached":
        raise ValueError("cached-two-stage student execution is only defined for quant mode")
    deployment_execution = "cached_document_prefill_then_query_continuation"
    training_execution = (
        "cached_document_prefill_then_full_query_continuation"
        if student_suffix_execution == "cached-two-stage"
        else (
            "cached_document_prefill_detached_then_full_query_continuation"
            if student_suffix_execution == "detached-document-cache"
            else (
            "cached_document_prefill_then_full_query_continuation_native_functional_state"
            if student_suffix_execution == "native-functional-cache"
            else (
            "uncached_full_document_plus_query_sequence"
            if mode == "quant"
            else "uncached_chunk_residuals_plus_query_sequence"
            )
            )
        )
    )
    cache_boundary_aligned = mode == "quant" and student_suffix_execution in {
        "cached-two-stage",
        "detached-document-cache",
        "native-functional-cache",
    }
    detached_document_cache = (
        mode == "quant" and student_suffix_execution == "detached-document-cache"
    )
    return {
        "mode": mode,
        "depth": depth,
        "teacher_kind": teacher_kind,
        "teacher_source": teacher_source,
        "teacher_suffix_execution": (
            "cached_document_prefill_then_full_query_continuation_mutable_inference"
            if mode == "quant"
            and student_suffix_execution == "native-functional-cache"
            else "historical_merged_or_mode_specific"
        ),
        "write_path_trainable": False,
        "lower_layers_trainable": False,
        "suffix_only_adapter": True,
        "store": store,
        "quantization_training_name": (
            "quantization_conditioned_lora_distillation"
            if mode == "quant"
            else None
        ),
        "is_qlora": False,
        "student_suffix_execution_option": student_suffix_execution,
        "student_suffix_execution": training_execution,
        "deployment_suffix_execution": (
            deployment_execution
            if mode == "quant"
            else "implementation_dependent_residual_only_replay"
        ),
        "training_deployment_cache_boundary_structurally_aligned": cache_boundary_aligned,
        "training_deployment_suffix_execution_claimed_equivalent": False,
        "cached_two_stage_autograd_capability": (
            cached_two_stage_autograd_capability_gate()
            if mode == "quant"
            else None
        ),
        "functional_cache_capability": (
            functional_cache_capability_gate() if mode == "quant" else None
        ),
        "document_prefill_parameter_gradients_enabled": not detached_document_cache,
        "document_cache_detached_before_query": detached_document_cache,
        "native_model_kernels_with_functional_cache_writes": (
            mode == "quant"
            and student_suffix_execution == "native-functional-cache"
        ),
        "claim_limit": (
            "query-continuation-only suffix adaptation; document-prefill suffix "
            "parameter contribution is frozen and this is not full two-stage training"
            if detached_document_cache
            else None
        ),
        "deployment_semantic_eval_gate_required": mode == "quant",
        "note": (
            "Quant mode fake-quantizes the persistent residual/KV/recurrent state; "
            "model weights are not quantized and gradients enter LoRA only. "
            + (
                "The document suffix prefill runs without gradients and its cache is "
                "detached+cloned before the differentiable full-query continuation. "
                "This trains query-continuation behavior only and must not be called "
                "full two-stage training. Query-side recurrent in-place updates may "
                "still violate autograd versioning; only a real-model smoke can pass it."
                if detached_document_cache
                else (
                "The cached-two-stage student uses the deployment document-prefill "
                "then full-query suffix boundary. A real-model gradient smoke and "
                "the independent all-position semantic gate are still required."
                if student_suffix_execution == "cached-two-stage"
                else (
                "The native functional student retains the model's native kernels and "
                "the deployment document/query boundary, but rebinds recurrent/conv "
                "cache tensors instead of copy_. Its Q16 teacher uses the ordinary "
                "mutable inference cache at the same boundary. Real-model backward, "
                "all-module update, and all-query semantic gates remain mandatory."
                if student_suffix_execution == "native-functional-cache"
                else "The merged-uncached student differs from deployment's document-prefill "
                "then query-continuation suffix boundary. Results must retain this "
                "distinction until a deployment-semantic gate is measured."
                )
                )
            )
        ),
    }
