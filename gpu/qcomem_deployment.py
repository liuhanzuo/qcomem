from __future__ import annotations

import os
import platform
import random
import socket
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import torch

from qcomem_torch import (
    FullPrefixState,
    PackedCache,
    PackedResidual,
    TorchSplitCausalLM,
    active_cache_layer_indices,
    cache_nbytes,
    tensor_nbytes,
)


DEFAULT_CONFIGS = (
    "dense-recompute",
    "full-prefix-q16",
    "qcomem-d7-r16-a16-l16",
    "qcomem-d7-r8-a8-l8",
    "qcomem-d7-r4-a4-l4",
    "qcomem-d7-r4-a4-l8",
    "qcomem-d7-mixed",
)
DEFAULT_MIXED_LAYER_BITS = (8, 8, 4, 4, 8, 8, 8)


@dataclass(frozen=True)
class DeploymentConfig:
    name: str
    mode: str
    depth: int | None = None
    residual_bits: int | None = None
    attention_bits: int | None = None
    linear_bits: int | None = None
    cache_layer_bits: tuple[int, ...] | None = None

    @property
    def is_exact(self) -> bool:
        return self.mode in {"dense_recompute", "full_prefix"} or (
            self.mode == "qcomem"
            and self.residual_bits == 16
            and self.attention_bits == 16
            and self.linear_bits == 16
            and (
                self.cache_layer_bits is None
                or all(bits == 16 for bits in self.cache_layer_bits)
            )
        )


def parse_layer_bits(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        fields = [field.strip() for field in value.split(",") if field.strip()]
        if not fields:
            raise ValueError("layer bits must not be empty")
        bits = tuple(int(field) for field in fields)
    else:
        bits = tuple(int(field) for field in value)
    if any(bit not in (2, 4, 8, 16) for bit in bits):
        raise ValueError("layer bits must be chosen from 2, 4, 8, 16")
    return bits


def parse_deployment_config(
    name: str,
    *,
    mixed_layer_bits: Sequence[int] = DEFAULT_MIXED_LAYER_BITS,
) -> DeploymentConfig:
    if name == "dense-recompute":
        return DeploymentConfig(name=name, mode="dense_recompute")
    if name in {"full-prefix", "full-prefix-q16"}:
        return DeploymentConfig(name=name, mode="full_prefix")
    if name == "qcomem-d7-frozen-static":
        return DeploymentConfig(
            name=name,
            mode="qcomem",
            depth=7,
            residual_bits=4,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=(8, 8, 8, 4, 8, 8, 8),
        )
    if not name.startswith("qcomem-d"):
        raise ValueError(f"unknown deployment config: {name}")

    fields = name.split("-")
    if len(fields) < 3:
        raise ValueError(f"incomplete Q-CoMem config: {name}")
    depth = int(fields[1].removeprefix("d"))
    if fields[2] == "mixed":
        layer_bits = parse_layer_bits(mixed_layer_bits)
        return DeploymentConfig(
            name=name,
            mode="qcomem",
            depth=depth,
            residual_bits=4,
            attention_bits=16,
            linear_bits=16,
            cache_layer_bits=layer_bits,
        )

    residual_bits = attention_bits = linear_bits = None
    layer_bits: tuple[int, ...] | None = None
    for field in fields[2:]:
        if field.startswith("layers="):
            layer_bits = parse_layer_bits(field.partition("=")[2])
            continue
        if len(field) < 2:
            raise ValueError(f"invalid Q-CoMem field: {field}")
        value = int(field[1:])
        if field.startswith("r"):
            residual_bits = value
        elif field.startswith("a"):
            attention_bits = value
        elif field.startswith("l"):
            linear_bits = value
        else:
            raise ValueError(f"invalid Q-CoMem field: {field}")
    if residual_bits is None:
        raise ValueError(f"Q-CoMem config must specify residual bits: {name}")
    if layer_bits is None and (attention_bits is None or linear_bits is None):
        raise ValueError(
            f"Q-CoMem config must specify a/l bits or layer bits: {name}"
        )
    return DeploymentConfig(
        name=name,
        mode="qcomem",
        depth=depth,
        residual_bits=residual_bits,
        attention_bits=attention_bits or 16,
        linear_bits=linear_bits or 16,
        cache_layer_bits=layer_bits,
    )


def load_mixed_policy(path: Path, policy_name: str) -> tuple[int, tuple[int, ...]]:
    import json

    payload = json.loads(path.read_text())
    if "policies" in payload:
        policy = payload["policies"][policy_name]
    elif policy_name in payload:
        policy = payload[policy_name]
    else:
        policy = payload
    return int(policy["residual_bits"]), parse_layer_bits(
        policy["cache_layer_bits"]
    )


def parameter_and_buffer_nbytes(model: torch.nn.Module) -> dict[str, int]:
    seen: set[tuple[str, int, int]] = set()

    def unique_bytes(tensors) -> int:
        total = 0
        for tensor in tensors:
            storage = tensor.untyped_storage()
            key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
        return total

    parameter_bytes = unique_bytes(model.parameters())
    buffer_bytes = unique_bytes(model.buffers())
    return {
        "model_parameter_nbytes": parameter_bytes,
        "model_buffer_nbytes": buffer_bytes,
        "model_weight_and_buffer_nbytes": parameter_bytes + buffer_bytes,
    }


class NvmlProcessSampler:
    """Low-overhead samples of this process' NVML memory, when pynvml exists."""

    def __init__(self, cuda_index: int) -> None:
        self.cuda_index = cuda_index
        self.available = False
        self.error: str | None = None
        self._nvml = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(cuda_index)
            self.available = True
        except Exception as error:  # pragma: no cover - depends on CUDA host
            self.error = f"{type(error).__name__}: {error}"

    def sample(self) -> int | None:
        if not self.available:
            return None
        assert self._nvml is not None and self._handle is not None
        try:
            processes = []
            for getter_name in (
                "nvmlDeviceGetComputeRunningProcesses_v3",
                "nvmlDeviceGetComputeRunningProcesses_v2",
                "nvmlDeviceGetComputeRunningProcesses",
            ):
                getter = getattr(self._nvml, getter_name, None)
                if getter is not None:
                    processes = getter(self._handle)
                    break
            used = [
                int(process.usedGpuMemory)
                for process in processes
                if process.pid == os.getpid()
                and getattr(process, "usedGpuMemory", None) is not None
            ]
            return sum(used) if used else 0
        except Exception as error:  # pragma: no cover - depends on CUDA host
            self.error = f"{type(error).__name__}: {error}"
            return None

    def metadata(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "error": self.error,
            "semantic": "sampled current process memory, not an allocator peak",
        }


class MemoryRecorder:
    def __init__(self, nvml: NvmlProcessSampler | None = None) -> None:
        self.nvml = nvml
        self.samples: list[dict[str, int | str | None]] = []

    def reset_peak(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.samples.clear()

    def sample(self, phase: str) -> dict[str, int | str | None]:
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
        else:
            allocated = reserved = 0
        row: dict[str, int | str | None] = {
            "phase": phase,
            "cuda_allocated_bytes": allocated,
            "cuda_reserved_bytes": reserved,
            "nvml_process_bytes": self.nvml.sample() if self.nvml else None,
        }
        self.samples.append(row)
        return row

    def summary(self, *, steady_prefix: str = "decode_") -> dict[str, Any]:
        steady = [
            sample
            for sample in self.samples
            if str(sample["phase"]).startswith(steady_prefix)
        ]
        if not steady:
            steady = self.samples[-1:]

        def median(field: str) -> int | None:
            values = [
                int(sample[field])
                for sample in steady
                if sample[field] is not None
            ]
            return round(statistics.median(values)) if values else None

        nvml_values = [
            int(sample["nvml_process_bytes"])
            for sample in self.samples
            if sample["nvml_process_bytes"] is not None
        ]
        return {
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
            ),
            "cuda_peak_reserved_bytes": (
                torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
            ),
            "steady_state_cuda_allocated_bytes": median("cuda_allocated_bytes"),
            "steady_state_cuda_reserved_bytes": median("cuda_reserved_bytes"),
            "nvml_sampled_peak_process_bytes": max(nvml_values)
            if nvml_values
            else None,
            "steady_state_nvml_process_bytes": median("nvml_process_bytes"),
            "memory_samples": self.samples,
        }


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(operation: Callable[[], Any]) -> tuple[Any, float]:
    _sync()
    started = time.perf_counter()
    value = operation()
    _sync()
    return value, time.perf_counter() - started


def persistent_components(state: Any | None) -> dict[str, Any]:
    if state is None:
        return {
            "persistent_residual_nbytes": 0,
            "persistent_lower_state_nbytes": 0,
            "persistent_document_nbytes": 0,
            "persistent_materialized_staging_nbytes": 0,
            "persistent_total_resident_nbytes": 0,
        }
    if hasattr(state, "deployment_memory_components"):
        return state.deployment_memory_components()
    if isinstance(state, FullPrefixState) or (
        hasattr(state, "cache") and not hasattr(state, "document_residual")
    ):
        cache_bytes = cache_nbytes(state.cache)
        return {
            "persistent_residual_nbytes": 0,
            "persistent_lower_state_nbytes": cache_bytes,
            "persistent_document_nbytes": cache_bytes,
            "persistent_materialized_staging_nbytes": 0,
            "persistent_total_resident_nbytes": cache_bytes,
        }
    residual = state.document_residual
    residual_bytes = (
        residual.nbytes if isinstance(residual, PackedResidual) else tensor_nbytes(residual)
    )
    lower_bytes = (
        state.cache.nbytes
        if isinstance(state.cache, PackedCache)
        else cache_nbytes(state.cache)
    )
    return {
        "persistent_residual_nbytes": residual_bytes,
        "persistent_lower_state_nbytes": lower_bytes,
        "persistent_document_nbytes": residual_bytes + lower_bytes,
        "persistent_materialized_staging_nbytes": 0,
        "persistent_total_resident_nbytes": residual_bytes + lower_bytes,
    }


def build_persistent_state(
    adapter: TorchSplitCausalLM,
    config: DeploymentConfig,
    document_tokens: torch.Tensor,
    *,
    group_size: int,
    fork_strategy: str = "deep-clone",
) -> Any | None:
    if fork_strategy not in {"deep-clone", "paged-cow-staging"}:
        raise ValueError("fork_strategy must be deep-clone or paged-cow-staging")
    if config.mode == "dense_recompute":
        return None
    if config.mode == "full_prefix":
        return adapter.write_full_prefix(document_tokens)
    if config.mode != "qcomem":
        raise ValueError(f"unsupported mode: {config.mode}")
    assert config.depth is not None and config.residual_bits is not None
    raw = adapter.write_lower_replay(document_tokens, config.depth)
    if config.cache_layer_bits is not None:
        active_layers = active_cache_layer_indices(raw.cache)
        allocated_layers = getattr(raw.cache, "layers", ())
        if len(config.cache_layer_bits) not in {
            len(active_layers),
            len(allocated_layers),
        }:
            raise ValueError(
                f"{config.name}: {len(config.cache_layer_bits)} layer bits for "
                f"{len(active_layers)} active / {len(allocated_layers)} allocated "
                "cache layers"
            )
    packed = raw.quantize(
        bits=config.residual_bits,
        attention_bits=config.attention_bits,
        linear_bits=config.linear_bits,
        cache_layer_bits=config.cache_layer_bits,
        group_size=group_size,
    )
    del raw
    if fork_strategy == "paged-cow-staging":
        from qcomem_paged import prepare_paged_lower_state

        return prepare_paged_lower_state(packed)
    return packed


@dataclass
class GenerationTrace:
    generated_token_ids: list[int]
    logits: list[torch.Tensor]
    ttft_seconds: float
    tpot_seconds: list[float]
    online_seconds: float
    instrumented_wall_seconds: float
    selected_fork_active_state_peak_nbytes: int
    selected_fork_active_state_steady_nbytes: int
    decode_kv_peak_nbytes: int
    decode_kv_steady_nbytes: int
    fork_memory: dict[str, Any]
    memory: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        generated = len(self.generated_token_ids)
        return {
            "generated_token_ids": self.generated_token_ids,
            "generated_tokens": generated,
            "ttft_seconds": self.ttft_seconds,
            "tpot_seconds": self.tpot_seconds,
            "median_tpot_seconds": (
                statistics.median(self.tpot_seconds) if self.tpot_seconds else None
            ),
            "throughput_tokens_per_second": (
                generated / self.online_seconds if self.online_seconds else None
            ),
            "online_seconds": self.online_seconds,
            "instrumented_wall_seconds": self.instrumented_wall_seconds,
            "selected_fork_active_state_peak_nbytes": (
                self.selected_fork_active_state_peak_nbytes
            ),
            "selected_fork_active_state_steady_nbytes": (
                self.selected_fork_active_state_steady_nbytes
            ),
            "decode_kv_peak_nbytes": self.decode_kv_peak_nbytes,
            "decode_kv_steady_nbytes": self.decode_kv_steady_nbytes,
            "fork_memory": self.fork_memory,
            **self.memory,
        }


def _argmax(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


@torch.inference_mode()
def run_incremental_generation(
    adapter: TorchSplitCausalLM,
    config: DeploymentConfig,
    document_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    persistent_state: Any | None,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    recorder: MemoryRecorder,
    collect_logits: bool = False,
) -> GenerationTrace:
    """Execute one request and expose persistent, active and decode-cache bytes.

    ``dense-recompute`` intentionally has no decode cache and recomputes the
    complete document/query/generated sequence for every token.  The other two
    paths consume the document once and then update only the new position.
    """

    document_tokens = TorchSplitCausalLM._batch_tokens(document_tokens)
    query_tokens = TorchSplitCausalLM._batch_tokens(query_tokens)
    recorder.reset_peak()
    recorder.sample("request_start")
    request_started = time.perf_counter()
    generated: list[int] = []
    logits_trace: list[torch.Tensor] = []
    tpot: list[float] = []
    selected_peak = selected_steady = 0
    decode_sizes: list[int] = []
    fork_memory: dict[str, Any] = {
        "strategy_requested": "not-applicable",
        "strategy_effective": "not-applicable",
        "fallback_reason": None,
    }

    if config.mode == "dense_recompute":
        current = torch.cat([document_tokens, query_tokens], dim=1)
        logits, ttft = _timed(lambda: adapter.full_last_logits(current))
        recorder.sample("ttft")

        def advance(token: int) -> torch.Tensor:
            nonlocal current
            token_tensor = torch.tensor([[token]], device=current.device)
            current = torch.cat([current, token_tensor], dim=1)
            return adapter.full_last_logits(current)

    elif config.mode == "full_prefix":
        if persistent_state is None:
            raise ValueError("full-prefix generation requires a persistent state")

        def initial_full():
            state = persistent_state.fork()
            return state, adapter.continue_full_prefix(state, query_tokens)

        (state, logits), ttft = _timed(initial_full)
        selected_peak = selected_steady = persistent_state.stored_nbytes
        initial_cache_bytes = selected_peak
        decode_sizes.append(max(cache_nbytes(state.cache) - initial_cache_bytes, 0))
        recorder.sample("ttft")

        def advance(token: int) -> torch.Tensor:
            token_tensor = torch.tensor([[token]], device=query_tokens.device)
            result = adapter.continue_full_prefix(state, token_tensor)
            decode_sizes.append(max(cache_nbytes(state.cache) - initial_cache_bytes, 0))
            return result

    elif config.mode == "qcomem":
        if persistent_state is None:
            raise ValueError("Q-CoMem generation requires a persistent state")
        depth = int(config.depth)

        def initial_replay():
            local = persistent_state.fork()
            active_initial = local.stored_nbytes
            local_fork_memory: dict[str, Any] = {
                "strategy_requested": getattr(
                    local, "fork_strategy_requested", "deep-clone"
                ),
                "strategy_effective": getattr(
                    local, "fork_strategy_effective", "deep-clone"
                ),
                "fallback_reason": getattr(local, "fallback_reason", None),
                "initial_shared_nbytes": getattr(
                    local, "initial_shared_nbytes", None
                ),
                "initial_private_nbytes": getattr(
                    local, "initial_private_nbytes", None
                ),
            }
            lower_initial = cache_nbytes(local.cache)
            query_residual = adapter.continue_lower_replay(local, query_tokens)
            suffix_cache = adapter.make_cache()
            # Preserve the same document/query chunk boundary used by an exact
            # full-prefix request.  Qwen3.5 linear-attention states are
            # numerically sensitive to chunking: consuming document+query as a
            # single suffix prefill can leave a different recurrent state even
            # when its first-token argmax happens to match.  That difference is
            # exposed at the next decode token.  The persistent document
            # residual therefore seeds the suffix cache first; query positions
            # extend it in a second call at the true document offset.
            adapter.run_suffix_cached_last_logits(
                [local.document_residual],
                depth,
                suffix_cache,
                position_offset=0,
            )
            first_logits = adapter.run_suffix_cached_last_logits(
                [query_residual],
                depth,
                suffix_cache,
                position_offset=local.document_length,
            )
            if hasattr(local, "memory_breakdown"):
                after_query = local.memory_breakdown()
                local_fork_memory.update(
                    {
                        "after_query_shared_nbytes": after_query["shared_nbytes"],
                        "after_query_private_nbytes": after_query["private_nbytes"],
                    }
                )
            active_after_query = tensor_nbytes(local.document_residual) + cache_nbytes(
                local.cache
            )
            suffix_length = local.current_length
            return (
                local,
                suffix_cache,
                first_logits,
                max(active_initial, active_after_query),
                lower_initial,
                suffix_length,
                local_fork_memory,
            )

        (
            local,
            suffix_cache,
            logits,
            selected_peak,
            lower_initial,
            suffix_length,
            fork_memory,
        ), ttft = _timed(initial_replay)
        lower_growth = max(cache_nbytes(local.cache) - lower_initial, 0)
        decode_sizes.append(lower_growth + cache_nbytes(suffix_cache))
        # The boundary residual is consumed by suffix prefill and is not needed
        # during steady-state decode.  Releasing it reflects the production
        # lifecycle instead of retaining a redundant active copy.
        local.document_residual = None
        selected_steady = cache_nbytes(local.cache)
        recorder.sample("ttft")

        def advance(token: int) -> torch.Tensor:
            nonlocal suffix_length, selected_peak, selected_steady
            token_tensor = torch.tensor([[token]], device=query_tokens.device)
            residual = adapter.continue_lower_replay(local, token_tensor)
            result = adapter.run_suffix_cached_last_logits(
                [residual],
                depth,
                suffix_cache,
                position_offset=suffix_length,
            )
            suffix_length += 1
            lower_delta = max(cache_nbytes(local.cache) - lower_initial, 0)
            decode_sizes.append(lower_delta + cache_nbytes(suffix_cache))
            selected_steady = cache_nbytes(local.cache)
            selected_peak = max(selected_peak, selected_steady)
            return result

    else:
        raise ValueError(f"unsupported mode: {config.mode}")

    for step in range(max_new_tokens):
        if collect_logits:
            logits_trace.append(logits.detach().cpu())
        token = _argmax(logits)
        if token in eos_token_ids:
            break
        generated.append(token)
        recorder.sample(f"decode_{step:03d}")
        if step + 1 < max_new_tokens:
            logits, elapsed = _timed(lambda token=token: advance(token))
            tpot.append(elapsed)

    _sync()
    if config.mode == "qcomem" and hasattr(local, "verify_shared_immutable"):
        fork_memory["immutable_audit"] = local.verify_shared_immutable()
        final_breakdown = local.memory_breakdown()
        fork_memory.update(
            {
                "final_shared_nbytes": final_breakdown["shared_nbytes"],
                "final_private_nbytes": final_breakdown["private_nbytes"],
            }
        )
    instrumented_wall_seconds = time.perf_counter() - request_started
    # NVML/allocator sampling happens outside the synchronized phase timers.
    # Use the sum of those phase timers for throughput so instrumentation does
    # not make one cache strategy appear slower merely because it has samples.
    online_seconds = ttft + sum(tpot)
    recorder.sample("request_end")
    memory = recorder.summary()
    decode_peak = max(decode_sizes, default=0)
    decode_steady = round(statistics.median(decode_sizes[1:] or decode_sizes or [0]))
    return GenerationTrace(
        generated_token_ids=generated,
        logits=logits_trace,
        ttft_seconds=ttft,
        tpot_seconds=tpot,
        online_seconds=online_seconds,
        instrumented_wall_seconds=instrumented_wall_seconds,
        selected_fork_active_state_peak_nbytes=selected_peak,
        selected_fork_active_state_steady_nbytes=selected_steady,
        decode_kv_peak_nbytes=decode_peak,
        decode_kv_steady_nbytes=decode_steady,
        fork_memory=fork_memory,
        memory=memory,
    )


def compare_generation_traces(
    reference: GenerationTrace,
    candidate: GenerationTrace,
    *,
    require_bitwise_logits: bool,
) -> dict[str, Any]:
    """Compare every emitted step, retaining auditable numeric diagnostics."""

    reference_tokens = [_argmax(logits) for logits in reference.logits]
    candidate_tokens = [_argmax(logits) for logits in candidate.logits]
    token_sequence_exact = reference_tokens == candidate_tokens
    same_logit_length = len(reference.logits) == len(candidate.logits)
    per_step = []
    all_bitwise = same_logit_length
    max_abs = 0.0
    max_relative_l2 = 0.0
    first_token_divergence_step = None
    first_logit_difference_step = None
    for step, (left, right) in enumerate(zip(reference.logits, candidate.logits)):
        same_shape = tuple(left.shape) == tuple(right.shape)
        bitwise = same_shape and torch.equal(left, right)
        all_bitwise = all_bitwise and bitwise
        if not bitwise and first_logit_difference_step is None:
            first_logit_difference_step = step
        if (
            _argmax(left) != _argmax(right)
            and first_token_divergence_step is None
        ):
            first_token_divergence_step = step
        if same_shape:
            difference = left.float() - right.float()
            absolute = float(difference.abs().max().item())
            difference_l2 = float(torch.linalg.vector_norm(difference).item())
            reference_l2 = float(torch.linalg.vector_norm(left.float()).item())
            relative_l2 = difference_l2 / max(reference_l2, 1e-30)
            max_abs = max(max_abs, absolute)
            max_relative_l2 = max(max_relative_l2, relative_l2)
        else:
            absolute = None
            difference_l2 = None
            reference_l2 = None
            relative_l2 = None
        per_step.append(
            {
                "step": step,
                "reference_token_id": _argmax(left),
                "candidate_token_id": _argmax(right),
                "token_exact": _argmax(left) == _argmax(right),
                "same_shape": same_shape,
                "logits_bitwise_exact": bitwise,
                "max_abs_logit_error": absolute,
                "logit_difference_l2": difference_l2,
                "reference_logit_l2": reference_l2,
                "relative_l2_logit_error": relative_l2,
            }
        )
    return {
        "passed": token_sequence_exact
        and (all_bitwise if require_bitwise_logits else True),
        "require_bitwise_logits": require_bitwise_logits,
        "reference_emitted_token_ids": reference_tokens,
        "candidate_emitted_token_ids": candidate_tokens,
        "reference_generated_token_ids": reference.generated_token_ids,
        "candidate_generated_token_ids": candidate.generated_token_ids,
        "token_sequence_exact": token_sequence_exact,
        "same_logit_trace_length": same_logit_length,
        "logits_bitwise_exact": all_bitwise,
        "first_token_divergence_step": first_token_divergence_step,
        "first_logit_difference_step": first_logit_difference_step,
        "max_abs_logit_error": max_abs,
        "max_relative_l2_logit_error": max_relative_l2,
        "per_step": per_step,
    }


def _tensor_snapshot(value: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Full-content snapshot used only by the small correctness gate."""

    result: list[tuple[torch.Tensor, torch.Tensor]] = []
    visited: set[int] = set()

    def visit(item: Any) -> None:
        object_id = id(item)
        if object_id in visited:
            return
        visited.add(object_id)
        if isinstance(item, torch.Tensor):
            result.append((item, item.detach().clone()))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif hasattr(item, "__dict__"):
            for child in vars(item).values():
                visit(child)

    visit(value)
    return result


def _verify_tensor_snapshot(
    snapshot: list[tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, Any]:
    failures = [
        index
        for index, (current, expected) in enumerate(snapshot)
        if not torch.equal(current, expected)
    ]
    return {
        "verified": not failures,
        "tensor_count": len(snapshot),
        "tensor_nbytes": sum(tensor_nbytes(current) for current, _ in snapshot),
        "changed_tensor_indices": failures,
        "audit": "full tensor torch.equal snapshot; correctness gate only",
    }


@torch.inference_mode()
def run_cow_vs_deep_clone_gate(
    adapter: TorchSplitCausalLM,
    config: DeploymentConfig,
    document_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    persistent_state: Any,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    full_prefix_trace: GenerationTrace | None = None,
    require_reference_logits: bool = False,
    reference_logit_atol: float = 0.0,
) -> dict[str, Any]:
    """Compare incremental full-prefix, eager Q16, and COW Q16 executions."""

    if config.mode != "qcomem" or not config.is_exact:
        raise ValueError("the direct COW gate requires an exact Q16 Q-CoMem config")
    from qcomem_paged import prepare_paged_lower_state

    # Materialize the COW template before the eager request.  If eager replay
    # accidentally mutates the packed source, the template remains an
    # independent pre-request reference and the paired trace exposes it.
    source_snapshot = _tensor_snapshot(persistent_state)
    cow_state = prepare_paged_lower_state(persistent_state)
    persistent_memory = cow_state.deployment_memory_components()
    same_persistent_source = cow_state.source_state is persistent_state
    eager = run_incremental_generation(
        adapter,
        config,
        document_tokens,
        query_tokens,
        persistent_state,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        recorder=MemoryRecorder(),
        collect_logits=True,
    )
    source_after_eager = _verify_tensor_snapshot(source_snapshot)
    cow = run_incremental_generation(
        adapter,
        config,
        document_tokens,
        query_tokens,
        cow_state,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        recorder=MemoryRecorder(),
        collect_logits=True,
    )
    source_after_cow = _verify_tensor_snapshot(source_snapshot)
    eager_vs_cow = compare_generation_traces(
        eager,
        cow,
        require_bitwise_logits=True,
    )
    full_prefix_vs_eager = (
        compare_generation_traces(
            full_prefix_trace,
            eager,
            require_bitwise_logits=False,
        )
        if full_prefix_trace is not None
        else None
    )
    full_prefix_vs_cow = (
        compare_generation_traces(
            full_prefix_trace,
            cow,
            require_bitwise_logits=False,
        )
        if full_prefix_trace is not None
        else None
    )

    def reference_comparison_passes(comparison: dict[str, Any] | None) -> bool:
        if not isinstance(comparison, dict) or not comparison.get("passed"):
            return False
        if not require_reference_logits:
            return True
        if reference_logit_atol == 0:
            return bool(comparison.get("logits_bitwise_exact"))
        return bool(
            comparison.get("same_logit_trace_length")
            and all(
                step.get("same_shape")
                and step.get("max_abs_logit_error", float("inf"))
                <= reference_logit_atol
                for step in comparison.get("per_step", ())
            )
        )

    full_prefix_vs_eager_passed = reference_comparison_passes(
        full_prefix_vs_eager
    )
    full_prefix_vs_cow_passed = reference_comparison_passes(full_prefix_vs_cow)
    incremental_three_way_token_exact = bool(
        isinstance(full_prefix_vs_eager, dict)
        and full_prefix_vs_eager.get("token_sequence_exact")
        and isinstance(full_prefix_vs_cow, dict)
        and full_prefix_vs_cow.get("token_sequence_exact")
        and eager_vs_cow.get("token_sequence_exact")
    )
    strategy_effective = cow.fork_memory.get("strategy_effective")
    immutable_audit = cow.fork_memory.get("immutable_audit")
    cow_was_exercised = strategy_effective == "paged-cow-staging"
    immutable_verified = bool(
        isinstance(immutable_audit, dict) and immutable_audit.get("verified")
    )
    passed = (
        same_persistent_source
        and cow_was_exercised
        and immutable_verified
        and source_after_eager["verified"]
        and source_after_cow["verified"]
        and full_prefix_vs_eager_passed
        and full_prefix_vs_cow_passed
        and eager_vs_cow["passed"]
    )
    return {
        "passed": passed,
        "semantic_version": "incremental-three-way-v1",
        "semantic": (
            "the same caller-visible document/query/decode chunk boundaries feed "
            "incremental full-prefix, eager Q16, and COW Q16; all three emitted "
            "token traces must match, and eager-vs-COW logits must be bitwise equal"
        ),
        "caller_boundary_match": full_prefix_trace is not None,
        "caller_execution_boundaries": {
            "document_write_chunks": [int(document_tokens.shape[-1])],
            "query_prefill_chunks": [int(query_tokens.shape[-1])],
            "decode_chunks": [1] * max(len(eager.logits) - 1, 0),
        },
        "incremental_three_way_token_exact": incremental_three_way_token_exact,
        "require_reference_logits": require_reference_logits,
        "reference_logit_atol": reference_logit_atol,
        "same_persistent_source": same_persistent_source,
        "persistent_memory": persistent_memory,
        "strategy_requested": "paged-cow-staging",
        "strategy_effective": strategy_effective,
        "fallback_reason": cow.fork_memory.get("fallback_reason"),
        "cow_was_exercised": cow_was_exercised,
        "source_after_eager": source_after_eager,
        "source_after_cow": source_after_cow,
        "cow_immutable_audit": immutable_audit,
        "comparisons": {
            "full_prefix_vs_eager_q16": full_prefix_vs_eager,
            "full_prefix_vs_cow_q16": full_prefix_vs_cow,
            "eager_q16_vs_cow_q16": eager_vs_cow,
        },
        # Compatibility alias.  The hard gate is the three-way comparison above.
        "comparison": eager_vs_cow,
        "eager_fork_memory": eager.fork_memory,
        "cow_fork_memory": cow.fork_memory,
    }


@torch.inference_mode()
def run_exactness_gate(
    adapter: TorchSplitCausalLM,
    document_tokens: torch.Tensor,
    query_tokens: torch.Tensor,
    *,
    depth: int,
    group_size: int,
    max_new_tokens: int,
    eos_token_ids: set[int],
    require_exact_logits: bool = False,
    logit_atol: float = 0.0,
    fork_strategy: str = "deep-clone",
) -> dict[str, Any]:
    """Hard gate incremental Q16 paths; dense recompute is diagnostic only."""

    configs = (
        parse_deployment_config("dense-recompute"),
        parse_deployment_config("full-prefix-q16"),
        parse_deployment_config(f"qcomem-d{depth}-r16-a16-l16"),
    )
    traces: dict[str, GenerationTrace] = {}
    for config in configs:
        persistent = build_persistent_state(
            adapter,
            config,
            document_tokens,
            group_size=group_size,
            fork_strategy=(
                fork_strategy if config.mode == "qcomem" else "deep-clone"
            ),
        )
        traces[config.name] = run_incremental_generation(
            adapter,
            config,
            document_tokens,
            query_tokens,
            persistent,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            recorder=MemoryRecorder(),
            collect_logits=True,
        )
        del persistent

    oracle = traces["dense-recompute"]
    comparisons = {}
    for name, trace in traces.items():
        token_matches = [
            left == right
            for left, right in zip(
                oracle.generated_token_ids, trace.generated_token_ids
            )
        ]
        same_length = len(trace.generated_token_ids) == len(
            oracle.generated_token_ids
        )
        token_exact = same_length and all(token_matches)
        max_abs = 0.0
        logit_exact = len(trace.logits) == len(oracle.logits)
        logit_close = logit_exact
        for left, right in zip(oracle.logits, trace.logits):
            difference = (left.float() - right.float()).abs()
            max_abs = max(max_abs, float(difference.max().item()))
            logit_exact = logit_exact and torch.equal(left, right)
            logit_close = logit_close and torch.allclose(
                left.float(), right.float(), atol=logit_atol, rtol=0
            )
        config_passed = token_exact and (
            (logit_exact if logit_atol == 0 else logit_close)
            if require_exact_logits
            else True
        )
        comparisons[name] = {
            "generated_token_ids": trace.generated_token_ids,
            "per_token_match": token_matches,
            "token_sequence_exact": token_exact,
            "logits_bitwise_exact": logit_exact,
            "logits_within_atol": logit_close,
            "max_abs_logit_error": max_abs,
            "fork_memory": trace.fork_memory,
            "passed": config_passed,
        }
    q16_name = configs[-1].name
    pairwise = {
        "dense_vs_full_prefix": compare_generation_traces(
            traces["dense-recompute"],
            traces["full-prefix-q16"],
            require_bitwise_logits=False,
        ),
        "dense_vs_qcomem_q16": compare_generation_traces(
            traces["dense-recompute"],
            traces[q16_name],
            require_bitwise_logits=False,
        ),
        "full_prefix_vs_qcomem_q16": compare_generation_traces(
            traces["full-prefix-q16"],
            traces[q16_name],
            require_bitwise_logits=False,
        ),
    }

    def incremental_comparison_passes(comparison: dict[str, Any]) -> bool:
        if not comparison.get("passed"):
            return False
        if not require_exact_logits:
            return True
        if logit_atol == 0:
            return bool(comparison.get("logits_bitwise_exact"))
        return bool(
            comparison.get("same_logit_trace_length")
            and all(
                step.get("same_shape")
                and step.get("max_abs_logit_error", float("inf")) <= logit_atol
                for step in comparison.get("per_step", ())
            )
        )

    dense_diagnostic_passed = bool(
        incremental_comparison_passes(pairwise["dense_vs_full_prefix"])
        and incremental_comparison_passes(pairwise["dense_vs_qcomem_q16"])
    )
    full_prefix_vs_q16_passed = incremental_comparison_passes(
        pairwise["full_prefix_vs_qcomem_q16"]
    )
    passed = full_prefix_vs_q16_passed
    document_length = int(TorchSplitCausalLM._batch_tokens(document_tokens).shape[1])
    query_length = int(TorchSplitCausalLM._batch_tokens(query_tokens).shape[1])
    execution_boundaries = {
        "semantic": (
            "caller-visible token chunk lengths; internal CUDA/Triton tile sizes "
            "are not represented"
        ),
        "dense_recompute": {
            "full_history_calls": [
                document_length + query_length + step
                for step in range(len(traces["dense-recompute"].logits))
            ],
        },
        "full_prefix": {
            "document_write_chunks": [document_length],
            "query_prefill_chunks": [query_length],
            "decode_chunks": [1]
            * max(len(traces["full-prefix-q16"].logits) - 1, 0),
        },
        "qcomem_q16": {
            "lower_document_write_chunks": [document_length],
            "lower_query_prefill_chunks": [query_length],
            "suffix_document_seed_chunks": [document_length],
            "suffix_query_prefill_chunks": [query_length],
            "lower_decode_chunks": [1]
            * max(len(traces[q16_name].logits) - 1, 0),
            "suffix_decode_chunks": [1]
            * max(len(traces[q16_name].logits) - 1, 0),
        },
    }
    cow_vs_deep_clone = None
    if fork_strategy == "paged-cow-staging":
        q16_config = configs[-1]
        q16_persistent = build_persistent_state(
            adapter,
            q16_config,
            document_tokens,
            group_size=group_size,
            fork_strategy="deep-clone",
        )
        cow_vs_deep_clone = run_cow_vs_deep_clone_gate(
            adapter,
            q16_config,
            document_tokens,
            query_tokens,
            q16_persistent,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            full_prefix_trace=traces["full-prefix-q16"],
            require_reference_logits=require_exact_logits,
            reference_logit_atol=logit_atol,
        )
        passed = cow_vs_deep_clone["passed"]
        del q16_persistent
    return {
        "passed": passed,
        "hard_gate_reference": "incremental-full-prefix-q16",
        "dense_single_chunk_diagnostic_only": True,
        "dense_diagnostic_passed": dense_diagnostic_passed,
        "incremental_hard_gate": {
            "passed": passed,
            "three_way_required": fork_strategy == "paged-cow-staging",
            "full_prefix_vs_q16_passed": full_prefix_vs_q16_passed,
            "reference": "incremental-full-prefix-q16",
        },
        "gate_semantic": (
            "incremental full-prefix vs Q16 is the hard reference; COW additionally "
            "requires a same-boundary full-prefix/eager/COW three-way token gate and "
            "bitwise eager-vs-COW logits. Dense single-chunk is diagnostic only"
        ),
        "require_exact_logits": require_exact_logits,
        "logit_atol": logit_atol,
        "depth": depth,
        "max_new_tokens": max_new_tokens,
        "fork_strategy": fork_strategy,
        "comparisons": comparisons,
        "pairwise": pairwise,
        "execution_boundaries": execution_boundaries,
        "cow_vs_deep_clone_q16": cow_vs_deep_clone,
    }


def capacity_estimate(
    *,
    total_device_bytes: int,
    model_allocated_bytes: int,
    persistent_document_bytes: int,
    request_peak_allocated_bytes: int,
    request_start_allocated_bytes: int,
    safety_headroom_bytes: int,
) -> dict[str, int | None]:
    if persistent_document_bytes <= 0:
        return {
            "max_resident_documents_store_only": None,
            "max_resident_documents_with_one_active_request": None,
            "active_request_allocator_overhead_bytes": max(
                request_peak_allocated_bytes - request_start_allocated_bytes, 0
            ),
        }
    store_budget = max(
        total_device_bytes - model_allocated_bytes - safety_headroom_bytes, 0
    )
    active_overhead = max(
        request_peak_allocated_bytes - request_start_allocated_bytes, 0
    )
    active_budget = max(store_budget - active_overhead, 0)
    return {
        "max_resident_documents_store_only": store_budget
        // persistent_document_bytes,
        "max_resident_documents_with_one_active_request": active_budget
        // persistent_document_bytes,
        "active_request_allocator_overhead_bytes": active_overhead,
    }


def shuffled_config_orders(
    config_names: Sequence[str], *, repeats: int, seed: int
) -> list[list[str]]:
    orders = []
    for repeat in range(repeats):
        order = list(config_names)
        random.Random(seed + repeat).shuffle(order)
        orders.append(order)
    return orders


def environment_metadata(model: torch.nn.Module | None = None) -> dict[str, Any]:
    def command_output(command: list[str]) -> str | None:
        try:
            return subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except Exception:
            return None

    metadata: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status_short": command_output(["git", "status", "--short"]),
        "nvidia_smi": command_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version,memory.total,pstate,power.limit",
                "--format=csv,noheader",
            ]
        ),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata["cuda_device"] = {
            "logical_index": index,
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    if model is not None:
        metadata.update(parameter_and_buffer_nbytes(model))
    return metadata


def config_asdict(config: DeploymentConfig) -> dict[str, Any]:
    return asdict(config)
