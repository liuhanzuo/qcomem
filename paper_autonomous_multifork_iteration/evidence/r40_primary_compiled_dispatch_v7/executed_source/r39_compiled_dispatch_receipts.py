#!/usr/bin/env python3
"""R40 fail-closed post-return dispatch receipts for the ForkAudit adapter.

This is intentionally a narrow, add-on experiment.  It instruments the actual
vLLM unified-attention Python entry point and Triton's ``CompiledKernel.run``
method, then joins each high-level attention call to the selected on-disk Triton
artifact bundle and its compile configuration. Success is sealed only after
the original selected launcher returns on the same CUDA device/stream. The
Qwen3.5 GDN path used by
the audited adapter is an eager PyTorch functional implementation; its receipts
bind that fact and the exact source file, but never pretend to attest individual
ATen/cuBLAS kernels.

The frozen primary entry point does *not* execute the separate qcomem manual GDN
dispatcher.  Its frozen Qwen3.5-35B-A3B checkpoint executes Transformers
``Qwen3_5MoeGatedDeltaNet.forward`` and uses qcomem's native-cache functional
state rebinding.  The MoE modeling module owns a distinct GDN class and four
mutually exclusive eager fallbacks: multi-token calls use
``torch_chunk_gated_delta_rule`` plus a functional conv-state rebind, while
cached single-token calls use ``torch_recurrent_gated_delta_rule`` plus
``torch_causal_conv1d_update``.  Both routes finish with qcomem's functional
recurrent-state rebind.  The runtime hook binds those actual routes and never
presents eager operations as compiled kernels.

The module is dependency-light when used for replay/tests. Runtime hooks import
Triton, Transformers, torch, and vLLM only inside ``install_runtime_hooks``.
"""

from __future__ import annotations

import argparse
import contextvars
import copy
import functools
import hashlib
import inspect
import json
import os
import shutil
import stat
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping


SCHEMA_VERSION = "forkaudit-r40-compiled-dispatch-receipt-v7"
TARGET_VLLM_ENTRYPOINT = (
    "vllm.v1.attention.ops.triton_unified_attention.unified_attention"
)
TARGET_GDN_ENTRYPOINT = (
    "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe."
    "Qwen3_5MoeGatedDeltaNet.forward"
)
GDN_FUNCTION = TARGET_GDN_ENTRYPOINT
GDN_DISPATCH_KIND = (
    "transformers-qwen3.5-moe-native-eager-torch-with-qcomem-functional-cache-rebind"
)
GDN_SOURCE_KEYS = {
    "transformers_accelerate_wrapper",
    "transformers_qwen35_moe_gdn_forward",
    "transformers_qwen35_moe_torch_chunk_gated_delta_rule",
    "transformers_qwen35_moe_torch_recurrent_gated_delta_rule",
    "transformers_qwen35_moe_torch_causal_conv1d_update",
    "qcomem_install_native_functional_linear_cache",
    "qcomem_functional_update_conv_state",
    "qcomem_functional_update_recurrent_state",
}
COMPILED_SOURCE_KEYS = {
    "vllm_unified_attention",
    "vllm_kernel_unified_attention_source",
    "triton_compiled_kernel_run_property_getter",
    "triton_autotuner_run",
}
DISPATCH_SOURCE_KEYS = GDN_SOURCE_KEYS | COMPILED_SOURCE_KEYS
REQUIRED_TRITON_CONFIG_FIELDS = ("num_warps", "num_ctas", "num_stages")
EXPECTED_TRITON_KERNEL_NAME = "kernel_unified_attention"
LAUNCH_CONTEXT_FIELDS = {
    "cuda_visible_devices",
    "torch_device_index",
    "torch_device_type",
    "torch_stream_id",
}
HOOK_INSTALLATION_RECEIPT = {
    "functional_stack_preloaded": False,
    "native_cache_module_preloaded": False,
    "transformers_qwen35_module_preloaded": False,
    "transformers_qwen35_moe_module_preloaded": False,
    "patched_before_entrypoint": True,
    "patched_before_model_instance_binding": True,
    "frozen_fast_path_available": False,
    "vllm_unified_attention_patched": True,
    "triton_compiled_launcher_patched": True,
    "triton_autotuner_patched": True,
    "chunk_route_patched": True,
    "recurrent_route_patched": True,
    "functional_conv_route_patched": True,
    "inplace_conv_route_patched": True,
    "functional_recurrent_rebind_patched": True,
}


class DispatchReceiptError(RuntimeError):
    """Raised for a missing, malformed, or unbound dispatch receipt."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DispatchReceiptError(message)


def _regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise DispatchReceiptError(f"{label} cannot be stat'ed: {error}") from error
    _require(not stat.S_ISLNK(mode), f"{label} must not be a symbolic link")
    _require(stat.S_ISREG(mode), f"{label} must be a regular file")


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError as error:
        raise DispatchReceiptError(f"path escapes declared root: {path}") from error


def _as_mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _exact_fields(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    _require(set(value) == expected, f"{label} fields drift")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _normalise_json_value(value: Any, *, label: str) -> Any:
    """Return an exact, finite JSON value without lossy string coercion."""

    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        import math

        _require(math.isfinite(value), f"{label} contains a non-finite float")
        return value
    if isinstance(value, (list, tuple)):
        return [
            _normalise_json_value(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            _require(type(key) is str and key, f"{label} has a non-string/empty key")
            _require(key not in result, f"{label} contains a duplicate key")
            result[key] = _normalise_json_value(item, label=f"{label}.{key}")
        return result
    raise DispatchReceiptError(f"{label} contains a non-JSON value of type {type(value).__name__}")


def _autotune_event(best: Any) -> dict[str, Any]:
    kwargs = getattr(best, "kwargs", None)
    _require(isinstance(kwargs, Mapping), "Triton best_config.kwargs is unavailable")
    selected_kwargs = _normalise_json_value(dict(kwargs), label="selected_kwargs")
    _require(isinstance(selected_kwargs, dict), "selected_kwargs must be an object")
    event = {
        "selected_kwargs": selected_kwargs,
        "selected_kwargs_sha256": _sha256_bytes(_canonical_bytes(selected_kwargs)),
        "num_warps": getattr(best, "num_warps", None),
        "num_stages": getattr(best, "num_stages", None),
        "num_ctas": getattr(best, "num_ctas", None),
    }
    for field in REQUIRED_TRITON_CONFIG_FIELDS:
        _require(type(event[field]) is int and event[field] > 0, f"autotune {field} invalid")
    return event


def _trusted_cuda_launch_context() -> dict[str, Any]:
    """Capture launcher-local identity inside the trusted PyTorch/CUDA process."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - frozen formal runtime only.
        raise DispatchReceiptError(f"PyTorch unavailable at compiled launch: {error}") from error
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    _require(
        isinstance(visible, str)
        and visible.startswith("GPU-")
        and "," not in visible
        and visible.strip() == visible,
        "compiled launch requires one UUID-valued CUDA_VISIBLE_DEVICES",
    )
    _require(torch.cuda.is_available(), "CUDA unavailable at compiled launch")
    device_index = torch.cuda.current_device()
    stream = torch.cuda.current_stream(device=device_index)
    stream_id = getattr(stream, "cuda_stream", None)
    _require(type(device_index) is int and device_index == 0, "compiled launch device index drift")
    _require(type(stream_id) is int and stream_id > 0, "compiled launch stream identity invalid")
    return {
        "cuda_visible_devices": visible,
        "torch_device_index": device_index,
        "torch_device_type": "cuda",
        "torch_stream_id": stream_id,
    }


def _verify_launch_context(value: Any, *, label: str) -> Mapping[str, Any]:
    context = _as_mapping(value, label)
    _exact_fields(context, LAUNCH_CONTEXT_FIELDS, label)
    visible = context.get("cuda_visible_devices")
    _require(
        isinstance(visible, str) and visible.startswith("GPU-") and "," not in visible,
        f"{label} visible UUID invalid",
    )
    _require(context.get("torch_device_index") == 0, f"{label} device index drift")
    _require(context.get("torch_device_type") == "cuda", f"{label} device type drift")
    _require(
        type(context.get("torch_stream_id")) is int and context["torch_stream_id"] > 0,
        f"{label} stream identity invalid",
    )
    return context


def _normalise_config(metadata: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for field in REQUIRED_TRITON_CONFIG_FIELDS:
        value = metadata.get(field)
        _require(type(value) is int and value > 0, f"compiled metadata lacks {field}")
        config[field] = value
    for field in ("maxnreg", "ptx_version", "ptx_options", "enable_fp_fusion"):
        if field in metadata:
            config[field] = metadata[field]
    return config


@dataclass(frozen=True)
class TritonArtifact:
    """One fully hashed, selected Triton cache directory."""

    artifact_id: str
    relative_dir: str
    compiler_hash: str
    kernel_name: str
    compile_config: dict[str, Any]
    metadata_sha256: str
    files: list[dict[str, str]]

    @classmethod
    def from_metadata_file(cls, cache_root: Path, metadata_path: Path) -> "TritonArtifact":
        _regular_file(metadata_path, "Triton metadata")
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DispatchReceiptError(f"invalid Triton metadata {metadata_path}: {error}") from error
        metadata = _as_mapping(metadata, "Triton metadata")
        kernel_name = metadata.get("name")
        compiler_hash = metadata.get("hash")
        _require(isinstance(kernel_name, str) and kernel_name, "compiled metadata lacks name")
        _require(
            isinstance(compiler_hash, str) and len(compiler_hash) == 64,
            "compiled metadata lacks a SHA-256 compiler hash",
        )
        compile_config = _normalise_config(metadata)
        artifact_dir = metadata_path.parent
        _require(artifact_dir.is_dir(), "Triton metadata parent is not a directory")
        files: list[dict[str, str]] = []
        for child in sorted(artifact_dir.iterdir(), key=lambda item: os.fsencode(item.name)):
            _regular_file(child, f"Triton artifact {child.name}")
            files.append(
                {
                    "relative_path": _relative(child, cache_root),
                    "sha256": _sha256_file(child),
                }
            )
        suffixes = {PurePosixPath(item["relative_path"]).suffix for item in files}
        _require(".cubin" in suffixes, "selected Triton artifact has no cubin")
        _require(".ptx" in suffixes, "selected Triton artifact has no PTX")
        payload = {
            "relative_dir": _relative(artifact_dir, cache_root),
            "compiler_hash": compiler_hash,
            "kernel_name": kernel_name,
            "compile_config": compile_config,
            "metadata_sha256": _sha256_file(metadata_path),
            "files": files,
        }
        return cls(artifact_id=_sha256_bytes(_canonical_bytes(payload)), **payload)

    def as_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "relative_dir": self.relative_dir,
            "compiler_hash": self.compiler_hash,
            "kernel_name": self.kernel_name,
            "compile_config": dict(self.compile_config),
            "metadata_sha256": self.metadata_sha256,
            "files": [dict(item) for item in self.files],
        }


def _candidate_artifacts(cache_root: Path, metadata: Mapping[str, Any]) -> list[TritonArtifact]:
    _require(cache_root.is_dir(), f"Triton cache root is absent: {cache_root}")
    target_name = metadata.get("name")
    target_hash = metadata.get("hash")
    target_config = _normalise_config(metadata)
    candidates: list[TritonArtifact] = []
    for metadata_path in sorted(cache_root.rglob("*.json"), key=lambda item: str(item)):
        try:
            candidate = TritonArtifact.from_metadata_file(cache_root, metadata_path)
        except DispatchReceiptError:
            continue
        if (
            candidate.kernel_name == target_name
            and candidate.compiler_hash == target_hash
            and candidate.compile_config == target_config
        ):
            candidates.append(candidate)
    return candidates


def _compiled_metadata(kernel: Any) -> dict[str, Any]:
    metadata = getattr(kernel, "metadata", None)
    if hasattr(metadata, "_asdict"):
        metadata = dict(metadata._asdict())
    elif hasattr(metadata, "items"):
        metadata = dict(metadata.items())
    elif hasattr(metadata, "__dict__"):
        metadata = dict(vars(metadata))
    _require(isinstance(metadata, dict), "CompiledKernel metadata is unavailable")
    return metadata


def _call_shape(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("q", "k", "v", "out", "block_table"):
        tensor = kwargs.get(name)
        shape = getattr(tensor, "shape", None)
        if shape is not None:
            result[name] = [int(value) for value in shape]
    for name in ("max_seqlen_q", "max_seqlen_k", "softmax_scale"):
        value = kwargs.get(name)
        if isinstance(value, (bool, int, float)):
            result[name] = value
    return result


@dataclass(frozen=True)
class SourceBinding:
    """One source file bound relative to an explicitly declared root."""

    root_kind: str
    relative_source_path: str
    source_sha256: str
    module: str
    qualname: str

    @classmethod
    def from_function(
        cls,
        function: Callable[..., Any],
        *,
        root: Path,
        root_kind: str,
    ) -> "SourceBinding":
        _require(root_kind in {"code", "runtime"}, "invalid source root kind")
        code = getattr(function, "__code__", None)
        _require(code is not None, "source-bound callable has no Python code object")
        source = Path(code.co_filename).resolve()
        _regular_file(source, "dispatch source")
        return cls(
            root_kind=root_kind,
            relative_source_path=_relative(source, root),
            source_sha256=_sha256_file(source),
            module=str(getattr(function, "__module__", "")),
            qualname=str(getattr(function, "__qualname__", "")),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "root_kind": self.root_kind,
            "relative_source_path": self.relative_source_path,
            "source_sha256": self.source_sha256,
            "module": self.module,
            "qualname": self.qualname,
        }


def _closure_function(function: Callable[..., Any], name: str) -> Callable[..., Any]:
    """Resolve one named closure callable without guessing through wrappers."""

    freevars = tuple(getattr(function.__code__, "co_freevars", ()))
    closure = tuple(function.__closure__ or ())
    _require(len(freevars) == len(closure), "callable closure is malformed")
    matches = [
        cell.cell_contents
        for freevar, cell in zip(freevars, closure)
        if freevar == name and callable(cell.cell_contents)
    ]
    _require(len(matches) == 1, f"callable closure lacks exactly one {name!r}")
    return matches[0]


@dataclass
class _AttentionContext:
    call_index: int
    call_id: str
    call_shape: dict[str, Any]
    launches: list[dict[str, Any]] = field(default_factory=list)
    pending_launches: list[dict[str, Any]] = field(default_factory=list)
    autotune_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _GDNContext:
    call_index: int
    layer_idx: int
    sequence_length: int
    cache_has_previous_state: bool
    execution_phase: str
    chunk_kernel_events: int = 0
    recurrent_kernel_events: int = 0
    conv_rebind_events: int = 0
    inplace_conv_update_events: int = 0
    recurrent_rebind_events: int = 0


def _gdn_route_counts(context: _GDNContext) -> tuple[str, tuple[int, int, int, int, int]]:
    """Return the frozen TF-5.14.1 route and exact event-count tuple.

    Tuple order is chunk rule, recurrent rule, functional conv rebind,
    in-place conv update, functional recurrent rebind.
    """

    if context.cache_has_previous_state and context.sequence_length == 1:
        return "cached-single-token", (0, 1, 0, 1, 1)
    return "multi-token", (1, 0, 1, 0, 1)


def _observed_gdn_route_counts(context: _GDNContext) -> tuple[int, int, int, int, int]:
    return (
        context.chunk_kernel_events,
        context.recurrent_kernel_events,
        context.conv_rebind_events,
        context.inplace_conv_update_events,
        context.recurrent_rebind_events,
    )


def _validate_gdn_route_counts(context: _GDNContext) -> str:
    route, expected = _gdn_route_counts(context)
    observed = _observed_gdn_route_counts(context)
    _require(
        observed == expected,
        f"GDN call {context.call_index} {route} route count drift: "
        f"observed={observed}, expected={expected}",
    )
    return route


class DispatchReceiptRecorder:
    """Collect and serialize fail-closed dispatch records for one process."""

    def __init__(
        self,
        *,
        cache_root: Path,
        code_root: Path,
        runtime_root: Path,
        launch_context_provider: Callable[[], Mapping[str, Any]] | None = None,
    ) -> None:
        self.cache_root = cache_root.resolve()
        self.code_root = code_root.resolve()
        self.runtime_root = runtime_root.resolve()
        self.attention_calls: list[dict[str, Any]] = []
        self.gdn_calls: list[dict[str, Any]] = []
        self.dispatch_source_bindings: dict[str, dict[str, str]] = {}
        self.hook_installation: dict[str, Any] = {}
        self.launch_context_provider = (
            launch_context_provider or _trusted_cuda_launch_context
        )
        self._active_attention: contextvars.ContextVar[_AttentionContext | None] = (
            contextvars.ContextVar("forkaudit_r40_attention", default=None)
        )
        self._active_gdn: contextvars.ContextVar[_GDNContext | None] = (
            contextvars.ContextVar("forkaudit_r40_gdn", default=None)
        )

    def begin_attention(self, kwargs: Mapping[str, Any]) -> contextvars.Token[_AttentionContext | None]:
        _require(self._active_attention.get() is None, "nested attention is unsupported")
        call_index = len(self.attention_calls)
        context = _AttentionContext(
            call_index=call_index,
            call_id=f"attention-{call_index}",
            call_shape=_call_shape(kwargs),
        )
        return self._active_attention.set(context)

    def prepare_compiled_kernel(self, kernel: Any) -> dict[str, Any] | None:
        """Bind the exact selected artifact immediately before its launcher call."""

        context = self._active_attention.get()
        if context is None:
            return None
        _require(not context.pending_launches, "a compiled launcher is already pending")
        _require(not context.launches, "attention attempted a duplicate compiled launch")
        metadata = _compiled_metadata(kernel)
        candidates = _candidate_artifacts(self.cache_root, metadata)
        _require(
            len(candidates) == 1,
            f"compiled kernel must bind exactly one cache artifact, found {len(candidates)}",
        )
        artifact = candidates[0]
        _require(
            artifact.kernel_name == EXPECTED_TRITON_KERNEL_NAME,
            "intercepted compiled launcher is not vLLM unified attention",
        )
        launch_context = dict(self.launch_context_provider())
        _verify_launch_context(launch_context, label="pre-launch context")
        pending = {
            "call_id": context.call_id,
            "artifact": artifact.as_dict(),
            "compiled_metadata": {
                "name": artifact.kernel_name,
                "hash": artifact.compiler_hash,
                **artifact.compile_config,
            },
            "launch_context": launch_context,
        }
        context.pending_launches.append(pending)
        return pending

    def seal_compiled_kernel(self, pending: dict[str, Any] | None) -> None:
        """Seal success only after the original Triton launcher returns normally."""

        if pending is None:
            return
        context = self._active_attention.get()
        _require(context is not None, "compiled launcher returned outside attention")
        _require(
            context.pending_launches == [pending],
            "compiled launch pending identity/order drift",
        )
        post = dict(self.launch_context_provider())
        _verify_launch_context(post, label="post-return launch context")
        _require(post == pending["launch_context"], "compiled launch changed device/stream")
        context.pending_launches.clear()
        sealed = copy.deepcopy(pending)
        sealed["post_launcher_returned"] = True
        sealed["post_return_context_matches"] = True
        context.launches.append(sealed)

    def abort_compiled_kernel(self, pending: dict[str, Any] | None) -> None:
        if pending is None:
            return
        context = self._active_attention.get()
        if context is not None and context.pending_launches == [pending]:
            context.pending_launches.clear()

    def record_autotune(self, autotuner: Any) -> None:
        context = self._active_attention.get()
        if context is None:
            return
        best = getattr(autotuner, "best_config", None)
        if best is None:
            return
        _require(not context.autotune_events, "attention observed duplicate autotune selections")
        context.autotune_events.append(_autotune_event(best))

    def abort_attention(self, token: contextvars.Token[_AttentionContext | None]) -> None:
        self._active_attention.reset(token)

    def finish_attention(self, token: contextvars.Token[_AttentionContext | None]) -> None:
        context = self._active_attention.get()
        _require(context is not None, "attention context disappeared")
        self._active_attention.reset(token)
        _require(not context.pending_launches, "attention ended with an unreturned launcher")
        _require(
            len(context.launches) == 1,
            f"attention call {context.call_index} launched {len(context.launches)} compiled kernels, expected one",
        )
        launch = context.launches[0]
        autotune = (
            {"mode": "triton-autotuner", "events": context.autotune_events}
            if context.autotune_events
            else {"mode": "no-autotuner-observed"}
        )
        receipt_core = {
            "call_id": context.call_id,
            "call_shape": context.call_shape,
            "artifact_id": launch["artifact"]["artifact_id"],
            "selected_compile_config": launch["compiled_metadata"],
            "autotune": autotune,
            "launch_context": launch["launch_context"],
            "post_launcher_returned": launch["post_launcher_returned"],
            "post_return_context_matches": launch["post_return_context_matches"],
        }
        self.attention_calls.append(
            {
                "call_index": context.call_index,
                "call_id": context.call_id,
                "vllm_entrypoint": TARGET_VLLM_ENTRYPOINT,
                "call_shape": context.call_shape,
                "selected_compiled_artifact": launch["artifact"],
                "selected_compile_config": launch["compiled_metadata"],
                "autotune": autotune,
                "launch_context": launch["launch_context"],
                "post_launcher_returned": True,
                "post_return_context_matches": True,
                "call_receipt_sha256": _sha256_bytes(_canonical_bytes(receipt_core)),
            }
        )

    def configure_dispatch_hooks(
        self,
        *,
        bindings: Mapping[str, SourceBinding],
        hook_installation: Mapping[str, Any],
    ) -> None:
        _require(not self.dispatch_source_bindings, "dispatch source bindings configured twice")
        _require(set(bindings) == DISPATCH_SOURCE_KEYS, "dispatch source-binding set drift")
        self.dispatch_source_bindings = {
            name: binding.as_dict() for name, binding in bindings.items()
        }
        self.hook_installation = dict(hook_installation)

    def begin_gdn(
        self,
        *,
        layer_idx: int,
        sequence_length: int,
        cache_has_previous_state: bool,
    ) -> contextvars.Token[_GDNContext | None]:
        _require(self._active_gdn.get() is None, "nested GDN forward is unsupported")
        _require(type(layer_idx) is int and layer_idx >= 0, "GDN layer index invalid")
        _require(
            type(sequence_length) is int and sequence_length > 0,
            "GDN sequence length invalid",
        )
        phase = "request-cell" if cache_has_previous_state else "document-prefill"
        context = _GDNContext(
            call_index=len(self.gdn_calls),
            layer_idx=layer_idx,
            sequence_length=sequence_length,
            cache_has_previous_state=cache_has_previous_state,
            execution_phase=phase,
        )
        return self._active_gdn.set(context)

    def abort_gdn(self, token: contextvars.Token[_GDNContext | None]) -> None:
        self._active_gdn.reset(token)

    def record_gdn_event(self, event: str) -> None:
        context = self._active_gdn.get()
        _require(context is not None, f"{event} occurred outside an intercepted GDN call")
        if event == "chunk-kernel":
            context.chunk_kernel_events += 1
        elif event == "recurrent-kernel":
            context.recurrent_kernel_events += 1
        elif event == "conv-rebind":
            context.conv_rebind_events += 1
        elif event == "inplace-conv-update":
            context.inplace_conv_update_events += 1
        elif event == "recurrent-rebind":
            context.recurrent_rebind_events += 1
        else:
            raise DispatchReceiptError(f"unknown GDN event {event!r}")

    def finish_gdn(self, token: contextvars.Token[_GDNContext | None]) -> None:
        context = self._active_gdn.get()
        _require(context is not None, "GDN context disappeared")
        self._active_gdn.reset(token)
        route = _validate_gdn_route_counts(context)
        if route == "cached-single-token":
            selected_function = "torch_recurrent_gated_delta_rule"
            selected_binding = (
                "transformers_qwen35_moe_torch_recurrent_gated_delta_rule"
            )
            selected_calls = context.recurrent_kernel_events
            conv_function = "torch_causal_conv1d_update"
            conv_binding = "transformers_qwen35_moe_torch_causal_conv1d_update"
            conv_calls = context.inplace_conv_update_events
            conv_semantics = "in-place-cache-update"
        else:
            selected_function = "torch_chunk_gated_delta_rule"
            selected_binding = "transformers_qwen35_moe_torch_chunk_gated_delta_rule"
            selected_calls = context.chunk_kernel_events
            conv_function = "qcomem functional update_conv_state"
            conv_binding = "qcomem_functional_update_conv_state"
            conv_calls = context.conv_rebind_events
            conv_semantics = "functional-cache-rebind"
        self.gdn_calls.append(
            {
                "call_index": context.call_index,
                "function": GDN_FUNCTION,
                "layer_idx": context.layer_idx,
                "sequence_length": context.sequence_length,
                "cache_has_previous_state": context.cache_has_previous_state,
                "execution_phase": context.execution_phase,
                "dispatch_kind": GDN_DISPATCH_KIND,
                "forward_source_bindings": [
                    "transformers_accelerate_wrapper",
                    "transformers_qwen35_moe_gdn_forward",
                ],
                "selected_gdn_kernel": {
                    "route": route,
                    "function": selected_function,
                    "source_binding": selected_binding,
                    "observed_calls": selected_calls,
                    "dispatch_kind": "eager-torch-fallback",
                    "compiled_artifact": None,
                    "autotune": {"mode": "not-applicable-to-eager-torch-fallback"},
                },
                "selected_conv_route": {
                    "function": conv_function,
                    "source_binding": conv_binding,
                    "observed_calls": conv_calls,
                    "state_semantics": conv_semantics,
                },
                "functional_cache_rebind": {
                    "install_source_binding": "qcomem_install_native_functional_linear_cache",
                    "conv_update_source_binding": "qcomem_functional_update_conv_state",
                    "recurrent_update_source_binding": "qcomem_functional_update_recurrent_state",
                    "conv_update_calls": context.conv_rebind_events,
                    "inplace_conv_update_calls": context.inplace_conv_update_events,
                    "recurrent_update_calls": context.recurrent_rebind_events,
                },
            }
        )

    def payload(self) -> dict[str, Any]:
        _require(self.attention_calls, "no vLLM/Triton attention calls were captured")
        _require(self.gdn_calls, "no native Transformers GDN calls were captured")
        _require(
            set(self.dispatch_source_bindings) == DISPATCH_SOURCE_KEYS,
            "dispatch source bindings are incomplete",
        )
        _require(self.hook_installation, "hook-installation receipt is absent")
        return {
            "schema_version": SCHEMA_VERSION,
            "scope": {
                "vllm_attention": TARGET_VLLM_ENTRYPOINT,
                "gdn": (
                    "actual Transformers Qwen3.5-MoE native eager GDN plus qcomem "
                    "functional cache rebind; underlying ATen/CUDA libraries are out of scope"
                ),
            },
            "hook_installation": copy.deepcopy(self.hook_installation),
            "dispatch_source_bindings": copy.deepcopy(self.dispatch_source_bindings),
            "attention_calls": copy.deepcopy(self.attention_calls),
            "gdn_calls": copy.deepcopy(self.gdn_calls),
        }


def install_runtime_hooks(recorder: DispatchReceiptRecorder) -> Callable[[], None]:
    """Install hooks before importing/running the audited model adapter.

    The wrapper is intentionally strict: it needs one ``CompiledKernel.run``
    launch inside every intercepted vLLM unified-attention call.  A Triton API
    drift, a second kernel, or an unbound cache artifact is a capture failure.
    """

    preloaded = {
        name: name in sys.modules
        for name in (
            "qcomem_qwen35_functional_stack",
            "qcomem_qwen35_native_cache",
            "transformers.models.qwen3_5.modeling_qwen3_5",
            "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
        )
    }
    _require(
        not any(preloaded.values()),
        "GDN/functional-stack modules were imported before dispatch interception",
    )
    try:
        from triton.compiler.compiler import CompiledKernel
        from triton.runtime.autotuner import Autotuner
        from transformers.models.qwen3_5_moe import (
            modeling_qwen3_5_moe as qwen_module,
        )
        import qcomem_qwen35_native_cache as native_cache_module

        # Import vLLM only after the GDN module globals are available for
        # interception.  The entry point itself has not run and no model/GDN
        # instance can have bound its selected eager route callables yet.
        from vllm.v1.attention.ops import triton_unified_attention as vllm_module
    except ImportError as error:  # pragma: no cover - frozen H20 environment only.
        raise DispatchReceiptError(f"required runtime module unavailable: {error}") from error

    original_unified = vllm_module.unified_attention
    original_unified_kernel = getattr(vllm_module, "kernel_unified_attention", None)
    original_unified_kernel_source = getattr(
        original_unified_kernel, "fn", original_unified_kernel
    )
    unified_signature = inspect.signature(original_unified)
    original_run_descriptor = inspect.getattr_static(CompiledKernel, "run", None)
    original_autotune = getattr(Autotuner, "run", None)
    gdn_class = qwen_module.Qwen3_5MoeGatedDeltaNet
    _require(
        qwen_module.__name__
        == "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
        and gdn_class.__qualname__ == "Qwen3_5MoeGatedDeltaNet",
        "frozen Qwen3.5-MoE GDN class identity drift",
    )
    original_gdn_forward = gdn_class.forward
    core_gdn_forward = _closure_function(original_gdn_forward, "forward_func")
    original_torch_chunk = qwen_module.torch_chunk_gated_delta_rule
    original_torch_recurrent = qwen_module.torch_recurrent_gated_delta_rule
    original_torch_causal_conv_update = qwen_module.torch_causal_conv1d_update
    original_native_install = native_cache_module.install_native_functional_linear_cache
    original_conv_rebind = native_cache_module._functional_update_conv_state
    original_recurrent_rebind = native_cache_module._functional_update_recurrent_state
    _require(
        isinstance(original_run_descriptor, property)
        and callable(original_run_descriptor.fget),
        "Triton CompiledKernel.run property is unavailable",
    )
    _require(callable(original_autotune), "Triton Autotuner.run is unavailable")
    _require(
        callable(original_unified_kernel_source),
        "vLLM unified-attention Triton kernel source is unavailable",
    )
    _require(callable(original_gdn_forward), "Transformers GDN forward is unavailable")
    _require(callable(original_torch_chunk), "Transformers torch chunk GDN is unavailable")
    _require(
        callable(original_torch_recurrent),
        "Transformers torch recurrent GDN is unavailable",
    )
    _require(
        callable(original_torch_causal_conv_update),
        "Transformers torch causal-conv update is unavailable",
    )
    _require(callable(original_conv_rebind), "qcomem conv rebind is unavailable")
    _require(callable(original_recurrent_rebind), "qcomem recurrent rebind is unavailable")
    _require(
        qwen_module.is_fast_path_available is False
        and qwen_module.causal_conv1d_fn is None
        and qwen_module.causal_conv1d_update is None
        and qwen_module.chunk_gated_delta_rule is None
        and qwen_module.fused_recurrent_gated_delta_rule is None,
        "frozen Qwen3.5 eager fallback selection drift",
    )

    bindings = {
        "vllm_unified_attention": SourceBinding.from_function(
            original_unified,
            root=recorder.runtime_root,
            root_kind="runtime",
        ),
        "vllm_kernel_unified_attention_source": SourceBinding.from_function(
            original_unified_kernel_source,
            root=recorder.runtime_root,
            root_kind="runtime",
        ),
        "triton_compiled_kernel_run_property_getter": SourceBinding.from_function(
            original_run_descriptor.fget,
            root=recorder.runtime_root,
            root_kind="runtime",
        ),
        "triton_autotuner_run": SourceBinding.from_function(
            original_autotune,
            root=recorder.runtime_root,
            root_kind="runtime",
        ),
        "transformers_accelerate_wrapper": SourceBinding.from_function(
            original_gdn_forward,
            root=recorder.runtime_root,
            root_kind="runtime",
        ),
        "transformers_qwen35_moe_gdn_forward": SourceBinding.from_function(
            core_gdn_forward,
            root=recorder.runtime_root,
            root_kind="runtime",
        ),
        "transformers_qwen35_moe_torch_chunk_gated_delta_rule": SourceBinding.from_function(
            original_torch_chunk,
            root=recorder.runtime_root,
            root_kind="runtime",
        ),
        "transformers_qwen35_moe_torch_recurrent_gated_delta_rule": SourceBinding.from_function(
            original_torch_recurrent,
            root=recorder.runtime_root,
            root_kind="runtime",
        ),
        "transformers_qwen35_moe_torch_causal_conv1d_update": SourceBinding.from_function(
            original_torch_causal_conv_update,
            root=recorder.runtime_root,
            root_kind="runtime",
        ),
        "qcomem_install_native_functional_linear_cache": SourceBinding.from_function(
            original_native_install,
            root=recorder.code_root,
            root_kind="code",
        ),
        "qcomem_functional_update_conv_state": SourceBinding.from_function(
            original_conv_rebind,
            root=recorder.code_root,
            root_kind="code",
        ),
        "qcomem_functional_update_recurrent_state": SourceBinding.from_function(
            original_recurrent_rebind,
            root=recorder.code_root,
            root_kind="code",
        ),
    }
    recorder.configure_dispatch_hooks(
        bindings=bindings,
        hook_installation={
            "functional_stack_preloaded": preloaded[
                "qcomem_qwen35_functional_stack"
            ],
            "native_cache_module_preloaded": preloaded[
                "qcomem_qwen35_native_cache"
            ],
            "transformers_qwen35_module_preloaded": preloaded[
                "transformers.models.qwen3_5.modeling_qwen3_5"
            ],
            "transformers_qwen35_moe_module_preloaded": preloaded[
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"
            ],
            "patched_before_entrypoint": True,
            "patched_before_model_instance_binding": True,
            "frozen_fast_path_available": False,
            "vllm_unified_attention_patched": True,
            "triton_compiled_launcher_patched": True,
            "triton_autotuner_patched": True,
            "chunk_route_patched": True,
            "recurrent_route_patched": True,
            "functional_conv_route_patched": True,
            "inplace_conv_route_patched": True,
            "functional_recurrent_rebind_patched": True,
        },
    )

    @functools.wraps(original_unified)
    def unified_wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = unified_signature.bind_partial(*args, **kwargs)
        except TypeError as error:
            raise DispatchReceiptError(
                f"cannot bind intercepted unified-attention arguments: {error}"
            ) from error
        token = recorder.begin_attention(bound.arguments)
        try:
            result = original_unified(*args, **kwargs)
        except BaseException:
            recorder.abort_attention(token)
            raise
        recorder.finish_attention(token)
        return result

    def compiled_run_getter(kernel: Any) -> Callable[..., Any]:
        original_launcher = original_run_descriptor.fget(kernel)
        _require(callable(original_launcher), "Triton instance launcher is unavailable")

        @functools.wraps(original_launcher)
        def compiled_run_wrapper(*args: Any, **kwargs: Any) -> Any:
            pending = recorder.prepare_compiled_kernel(kernel)
            try:
                result = original_launcher(*args, **kwargs)
            except BaseException:
                recorder.abort_compiled_kernel(pending)
                raise
            recorder.seal_compiled_kernel(pending)
            return result

        return compiled_run_wrapper

    @functools.wraps(original_autotune)
    def autotune_wrapper(autotuner: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_autotune(autotuner, *args, **kwargs)
        recorder.record_autotune(autotuner)
        return result

    @functools.wraps(original_torch_chunk)
    def torch_chunk_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original_torch_chunk(*args, **kwargs)
        recorder.record_gdn_event("chunk-kernel")
        return result

    @functools.wraps(original_torch_recurrent)
    def torch_recurrent_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original_torch_recurrent(*args, **kwargs)
        recorder.record_gdn_event("recurrent-kernel")
        return result

    @functools.wraps(original_torch_causal_conv_update)
    def torch_causal_conv_update_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original_torch_causal_conv_update(*args, **kwargs)
        recorder.record_gdn_event("inplace-conv-update")
        return result

    @functools.wraps(original_conv_rebind)
    def conv_rebind_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original_conv_rebind(*args, **kwargs)
        recorder.record_gdn_event("conv-rebind")
        return result

    @functools.wraps(original_recurrent_rebind)
    def recurrent_rebind_wrapper(*args: Any, **kwargs: Any) -> Any:
        result = original_recurrent_rebind(*args, **kwargs)
        recorder.record_gdn_event("recurrent-rebind")
        return result

    @functools.wraps(original_gdn_forward)
    def gdn_forward_wrapper(module: Any, *args: Any, **kwargs: Any) -> Any:
        hidden_states = args[0] if args else kwargs.get("hidden_states")
        cache_params = (
            args[1]
            if len(args) > 1
            else kwargs.get("cache_params")
        )
        shape = getattr(hidden_states, "shape", None)
        _require(shape is not None and len(shape) == 3, "GDN hidden-state shape invalid")
        layer_idx = getattr(module, "layer_idx", None)
        _require(type(layer_idx) is int and layer_idx >= 0, "GDN module layer index invalid")
        _require(cache_params is not None, "R29 GDN call lacks cache parameters")
        _require(
            getattr(module, "chunk_gated_delta_rule", None) is torch_chunk_wrapper,
            "GDN instance did not bind the intercepted MoE torch chunk rule",
        )
        _require(
            getattr(module, "recurrent_gated_delta_rule", None)
            is torch_recurrent_wrapper,
            "GDN instance did not bind the intercepted MoE torch recurrent rule",
        )
        _require(
            getattr(module, "causal_conv1d_update", None)
            is torch_causal_conv_update_wrapper,
            "GDN instance did not bind the intercepted MoE torch causal-conv update",
        )
        cache_layers = getattr(cache_params, "layers", None)
        _require(
            isinstance(cache_layers, (list, tuple)) and layer_idx < len(cache_layers),
            "R29 GDN cache layer sequence is invalid",
        )
        cache_layer = cache_layers[layer_idx]
        _require(
            getattr(getattr(cache_layer, "update_conv_state", None), "__func__", None)
            is conv_rebind_wrapper,
            "GDN cache did not bind the intercepted qcomem conv rebind",
        )
        _require(
            getattr(
                getattr(cache_layer, "update_recurrent_state", None),
                "__func__",
                None,
            )
            is recurrent_rebind_wrapper,
            "GDN cache did not bind the intercepted qcomem recurrent rebind",
        )
        has_previous = cache_params.has_previous_state(layer_idx)
        _require(type(has_previous) is bool, "GDN cache previous-state flag is not bool")
        token = recorder.begin_gdn(
            layer_idx=layer_idx,
            sequence_length=int(shape[1]),
            cache_has_previous_state=has_previous,
        )
        try:
            result = original_gdn_forward(module, *args, **kwargs)
        except BaseException:
            recorder.abort_gdn(token)
            raise
        recorder.finish_gdn(token)
        return result

    vllm_module.unified_attention = unified_wrapper
    CompiledKernel.run = property(compiled_run_getter)
    Autotuner.run = autotune_wrapper
    qwen_module.torch_chunk_gated_delta_rule = torch_chunk_wrapper
    qwen_module.torch_recurrent_gated_delta_rule = torch_recurrent_wrapper
    qwen_module.torch_causal_conv1d_update = torch_causal_conv_update_wrapper
    native_cache_module._functional_update_conv_state = conv_rebind_wrapper
    native_cache_module._functional_update_recurrent_state = recurrent_rebind_wrapper
    gdn_class.forward = gdn_forward_wrapper

    def restore() -> None:
        vllm_module.unified_attention = original_unified
        CompiledKernel.run = original_run_descriptor
        Autotuner.run = original_autotune
        qwen_module.torch_chunk_gated_delta_rule = original_torch_chunk
        qwen_module.torch_recurrent_gated_delta_rule = original_torch_recurrent
        qwen_module.torch_causal_conv1d_update = original_torch_causal_conv_update
        native_cache_module._functional_update_conv_state = original_conv_rebind
        native_cache_module._functional_update_recurrent_state = original_recurrent_rebind
        gdn_class.forward = original_gdn_forward

    return restore


def _verify_artifact(cache_root: Path, raw: Any, label: str) -> None:
    artifact = _as_mapping(raw, f"{label} artifact")
    required = {"artifact_id", "relative_dir", "compiler_hash", "kernel_name", "compile_config", "metadata_sha256", "files"}
    _require(set(artifact) == required, f"{label} artifact fields drift")
    relative_dir = artifact["relative_dir"]
    _require(isinstance(relative_dir, str) and relative_dir, f"{label} relative_dir invalid")
    metadata_path = cache_root / relative_dir / f"{artifact['kernel_name']}.json"
    reconstructed = TritonArtifact.from_metadata_file(cache_root, metadata_path).as_dict()
    _require(reconstructed == dict(artifact), f"{label} selected artifact binding fails")


def _binding_source(
    binding: Mapping[str, Any], *, code_root: Path, runtime_root: Path
) -> Path:
    root_kind = binding.get("root_kind")
    _require(root_kind in {"code", "runtime"}, "source-binding root kind invalid")
    relative_source = binding.get("relative_source_path")
    _require(
        isinstance(relative_source, str) and relative_source,
        "source-binding relative path missing",
    )
    relative = PurePosixPath(relative_source)
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        "source-binding path escapes its root",
    )
    root = code_root if root_kind == "code" else runtime_root
    source = root.joinpath(*relative.parts)
    try:
        source.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise DispatchReceiptError("source-binding path escapes its root") from error
    _regular_file(source, "dispatch source")
    return source


def _verify_dispatch_source_bindings(
    payload: Mapping[str, Any], *, code_root: Path, runtime_root: Path
) -> Mapping[str, Any]:
    bindings = _as_mapping(
        payload.get("dispatch_source_bindings"), "dispatch source bindings"
    )
    _require(set(bindings) == DISPATCH_SOURCE_KEYS, "dispatch source-binding set drift")
    for name, raw in bindings.items():
        binding = _as_mapping(raw, f"dispatch source binding {name}")
        _require(
            set(binding)
            == {
                "root_kind",
                "relative_source_path",
                "source_sha256",
                "module",
                "qualname",
            },
            f"dispatch source binding {name} fields drift",
        )
        source = _binding_source(binding, code_root=code_root, runtime_root=runtime_root)
        _require(
            _sha256_file(source) == binding.get("source_sha256"),
            f"dispatch source binding {name} hash drift",
        )
        _require(
            isinstance(binding.get("module"), str) and binding.get("module"),
            f"dispatch source binding {name} module missing",
        )
        _require(
            isinstance(binding.get("qualname"), str) and binding.get("qualname"),
            f"dispatch source binding {name} qualname missing",
        )
    return bindings


def verify_payload(
    payload: Any,
    *,
    cache_root: Path,
    code_root: Path,
    runtime_root: Path,
) -> dict[str, Any]:
    """Replay all receipt bindings without importing the candidate model/runtime."""

    payload = _as_mapping(payload, "receipt")
    _exact_fields(
        payload,
        {
            "schema_version",
            "scope",
            "hook_installation",
            "dispatch_source_bindings",
            "attention_calls",
            "gdn_calls",
        },
        "receipt",
    )
    _require(payload.get("schema_version") == SCHEMA_VERSION, "receipt schema version drift")
    scope = _as_mapping(payload.get("scope"), "receipt scope")
    _exact_fields(scope, {"vllm_attention", "gdn"}, "receipt scope")
    _require(scope.get("vllm_attention") == TARGET_VLLM_ENTRYPOINT, "vLLM scope drift")
    _require(
        scope.get("gdn")
        == (
            "actual Transformers Qwen3.5-MoE native eager GDN plus qcomem "
            "functional cache rebind; underlying ATen/CUDA libraries are out of scope"
        ),
        "GDN scope drift",
    )
    hook = _as_mapping(payload.get("hook_installation"), "hook installation")
    _require(hook == HOOK_INSTALLATION_RECEIPT, "hook-installation ordering receipt drift")
    bindings = _verify_dispatch_source_bindings(
        payload, code_root=code_root, runtime_root=runtime_root
    )
    attention = payload.get("attention_calls")
    gdn = payload.get("gdn_calls")
    _require(isinstance(attention, list) and attention, "attention receipt list missing")
    _require(isinstance(gdn, list) and gdn, "GDN receipt list missing")
    for index, row_raw in enumerate(attention):
        row = _as_mapping(row_raw, f"attention row {index}")
        _exact_fields(
            row,
            {
                "call_index",
                "call_id",
                "vllm_entrypoint",
                "call_shape",
                "selected_compiled_artifact",
                "selected_compile_config",
                "autotune",
                "launch_context",
                "post_launcher_returned",
                "post_return_context_matches",
                "call_receipt_sha256",
            },
            f"attention row {index}",
        )
        _require(row.get("call_index") == index, "attention call index drift")
        _require(row.get("call_id") == f"attention-{index}", "attention call ID drift")
        _require(row.get("vllm_entrypoint") == TARGET_VLLM_ENTRYPOINT, "attention entrypoint drift")
        selected = _as_mapping(row.get("selected_compiled_artifact"), "selected artifact")
        config = _as_mapping(row.get("selected_compile_config"), "selected config")
        _require(selected.get("kernel_name") == EXPECTED_TRITON_KERNEL_NAME, "unexpected compiled kernel identity")
        _exact_fields(
            config,
            {"name", "hash", *REQUIRED_TRITON_CONFIG_FIELDS},
            "selected config",
        )
        _require(config.get("name") == selected.get("kernel_name"), "kernel-name receipt drift")
        _require(config.get("hash") == selected.get("compiler_hash"), "compiler-hash receipt drift")
        _require(config.get("num_warps") == selected.get("compile_config", {}).get("num_warps"), "num_warps receipt drift")
        _require(config.get("num_ctas") == selected.get("compile_config", {}).get("num_ctas"), "num_ctas receipt drift")
        _require(config.get("num_stages") == selected.get("compile_config", {}).get("num_stages"), "num_stages receipt drift")
        autotune = _as_mapping(row.get("autotune"), "autotune receipt")
        _require(autotune.get("mode") in {"triton-autotuner", "no-autotuner-observed"}, "autotune mode drift")
        if autotune.get("mode") == "triton-autotuner":
            _exact_fields(autotune, {"mode", "events"}, "autotune receipt")
            events = autotune.get("events")
            _require(isinstance(events, list) and len(events) == 1, "autotuner receipt must contain one exact selection")
            selected_autotune = _as_mapping(events[0], "selected autotuner configuration")
            _exact_fields(
                selected_autotune,
                {"selected_kwargs", "selected_kwargs_sha256", *REQUIRED_TRITON_CONFIG_FIELDS},
                "selected autotuner configuration",
            )
            selected_kwargs = _normalise_json_value(
                selected_autotune.get("selected_kwargs"), label="selected_kwargs"
            )
            _require(isinstance(selected_kwargs, dict), "selected_kwargs must be an object")
            _require(
                selected_autotune.get("selected_kwargs_sha256")
                == _sha256_bytes(_canonical_bytes(selected_kwargs)),
                "selected_kwargs digest drift",
            )
            for field in REQUIRED_TRITON_CONFIG_FIELDS:
                _require(
                    selected_autotune.get(field) == config.get(field),
                    f"autotuner {field} disagrees with selected compiled configuration",
                )
        else:
            _require(dict(autotune) == {"mode": "no-autotuner-observed"}, "no-autotuner receipt fields drift")
        launch_context = _verify_launch_context(
            row.get("launch_context"), label=f"attention row {index} launch context"
        )
        _require(row.get("post_launcher_returned") is True, "launcher success-return receipt absent")
        _require(row.get("post_return_context_matches") is True, "launcher context-return match absent")
        receipt_core = {
            "call_id": row["call_id"],
            "call_shape": row["call_shape"],
            "artifact_id": selected["artifact_id"],
            "selected_compile_config": dict(config),
            "autotune": dict(autotune),
            "launch_context": dict(launch_context),
            "post_launcher_returned": True,
            "post_return_context_matches": True,
        }
        _require(
            row.get("call_receipt_sha256")
            == _sha256_bytes(_canonical_bytes(receipt_core)),
            "attention call receipt digest drift",
        )
        _verify_artifact(cache_root, selected, f"attention row {index}")
    for index, row_raw in enumerate(gdn):
        row = _as_mapping(row_raw, f"GDN row {index}")
        _require(row.get("call_index") == index, "GDN call index drift")
        _require(row.get("function") == GDN_FUNCTION, "GDN function drift")
        _require(row.get("dispatch_kind") == GDN_DISPATCH_KIND, "GDN dispatch kind drift")
        _require(
            type(row.get("layer_idx")) is int and row["layer_idx"] >= 0,
            "GDN layer index invalid",
        )
        _require(
            type(row.get("sequence_length")) is int and row["sequence_length"] > 0,
            "GDN sequence length invalid",
        )
        has_previous = row.get("cache_has_previous_state")
        _require(type(has_previous) is bool, "GDN cache-state flag invalid")
        expected_phase = "request-cell" if has_previous else "document-prefill"
        _require(row.get("execution_phase") == expected_phase, "GDN phase binding drift")
        _require(
            row.get("forward_source_bindings")
            == [
                "transformers_accelerate_wrapper",
                "transformers_qwen35_moe_gdn_forward",
            ],
            "GDN forward-source binding drift",
        )
        selected_kernel = _as_mapping(
            row.get("selected_gdn_kernel"), "selected GDN kernel"
        )
        single_token = has_previous and row["sequence_length"] == 1
        expected_route = "cached-single-token" if single_token else "multi-token"
        expected_kernel_function = (
            "torch_recurrent_gated_delta_rule"
            if single_token
            else "torch_chunk_gated_delta_rule"
        )
        expected_kernel_binding = (
            "transformers_qwen35_moe_torch_recurrent_gated_delta_rule"
            if single_token
            else "transformers_qwen35_moe_torch_chunk_gated_delta_rule"
        )
        _require(
            selected_kernel
            == {
                "route": expected_route,
                "function": expected_kernel_function,
                "source_binding": expected_kernel_binding,
                "observed_calls": 1,
                "dispatch_kind": "eager-torch-fallback",
                "compiled_artifact": None,
                "autotune": {"mode": "not-applicable-to-eager-torch-fallback"},
            },
            "selected GDN kernel receipt drift",
        )
        selected_conv = _as_mapping(
            row.get("selected_conv_route"), "selected conv route"
        )
        expected_conv = (
            {
                "function": "torch_causal_conv1d_update",
                "source_binding": "transformers_qwen35_moe_torch_causal_conv1d_update",
                "observed_calls": 1,
                "state_semantics": "in-place-cache-update",
            }
            if single_token
            else {
                "function": "qcomem functional update_conv_state",
                "source_binding": "qcomem_functional_update_conv_state",
                "observed_calls": 1,
                "state_semantics": "functional-cache-rebind",
            }
        )
        _require(selected_conv == expected_conv, "selected conv-route receipt drift")
        cache_rebind = _as_mapping(
            row.get("functional_cache_rebind"), "functional cache rebind"
        )
        _require(
            cache_rebind
            == {
                "install_source_binding": "qcomem_install_native_functional_linear_cache",
                "conv_update_source_binding": "qcomem_functional_update_conv_state",
                "recurrent_update_source_binding": "qcomem_functional_update_recurrent_state",
                "conv_update_calls": 0 if single_token else 1,
                "inplace_conv_update_calls": 1 if single_token else 0,
                "recurrent_update_calls": 1,
            },
            "functional-cache rebind receipt drift",
        )
        for source_key in [*row["forward_source_bindings"], selected_kernel["source_binding"], selected_conv["source_binding"], cache_rebind["install_source_binding"], cache_rebind["conv_update_source_binding"], cache_rebind["recurrent_update_source_binding"]]:
            _require(source_key in bindings, "GDN row references an unknown source binding")
    prefill_count = sum(
        row.get("execution_phase") == "document-prefill" for row in gdn
    )
    request_count = sum(row.get("execution_phase") == "request-cell" for row in gdn)
    return {
        "schema_version": SCHEMA_VERSION,
        "replay_verdict": "pass",
        "attention_call_count": len(attention),
        "gdn_call_count": len(gdn),
        "gdn_document_prefill_call_count": prefill_count,
        "gdn_request_cell_call_count": request_count,
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value) + b"\n")


def _demo_fixture(root: Path) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Build a static fake cache solely for unit tests and negative controls."""

    cache = root / "cache"
    bundle = cache / "ABCD"
    bundle.mkdir(parents=True)
    metadata = {
        "hash": "a" * 64,
        "name": "kernel_unified_attention",
        "num_warps": 4,
        "num_ctas": 1,
        "num_stages": 3,
    }
    _write_json(bundle / "kernel_unified_attention.json", metadata)
    (bundle / "kernel_unified_attention.cubin").write_bytes(b"test-cubin")
    (bundle / "kernel_unified_attention.ptx").write_bytes(b"test-ptx")
    code = root / "code"
    code.mkdir()
    native_source = code / "qcomem_qwen35_native_cache.py"
    native_source.write_text(
        "def install_native_functional_linear_cache(): pass\n"
        "def _functional_update_conv_state(): pass\n"
        "def _functional_update_recurrent_state(): pass\n",
        encoding="utf-8",
    )
    runtime = root / "runtime"
    qwen_source = (
        runtime
        / "transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py"
    )
    qwen_source.parent.mkdir(parents=True)
    qwen_source.write_text(
        "def torch_causal_conv1d_update(): pass\n"
        "def torch_chunk_gated_delta_rule(): pass\n"
        "def torch_recurrent_gated_delta_rule(): pass\n"
        "class Qwen3_5MoeGatedDeltaNet:\n    def forward(self): pass\n",
        encoding="utf-8",
    )
    accelerate_source = runtime / "transformers/integrations/accelerate.py"
    accelerate_source.parent.mkdir(parents=True)
    accelerate_source.write_text("def wrapped(): pass\n", encoding="utf-8")
    vllm_source = runtime / "vllm/v1/attention/ops/triton_unified_attention.py"
    vllm_source.parent.mkdir(parents=True)
    vllm_source.write_text("def unified_attention(): pass\n", encoding="utf-8")
    triton_compiler_source = runtime / "triton/compiler/compiler.py"
    triton_compiler_source.parent.mkdir(parents=True)
    triton_compiler_source.write_text("def run_getter(): pass\n", encoding="utf-8")
    triton_autotuner_source = runtime / "triton/runtime/autotuner.py"
    triton_autotuner_source.parent.mkdir(parents=True)
    triton_autotuner_source.write_text("def run(): pass\n", encoding="utf-8")
    artifact = TritonArtifact.from_metadata_file(cache, bundle / "kernel_unified_attention.json")
    source_binding = lambda root_kind, relative, source, module, qualname: {
        "root_kind": root_kind,
        "relative_source_path": relative,
        "source_sha256": _sha256_file(source),
        "module": module,
        "qualname": qualname,
    }
    call_shape = {
        "q": [32, 16, 256],
        "k": [1024, 2, 256],
        "v": [1024, 2, 256],
        "out": [32, 16, 256],
        "block_table": [1, 64],
    }
    selected_config = {
        "name": artifact.kernel_name,
        "hash": artifact.compiler_hash,
        **artifact.compile_config,
    }
    launch_context = {
        "cuda_visible_devices": "GPU-00000000-0000-0000-0000-000000000000",
        "torch_device_index": 0,
        "torch_device_type": "cuda",
        "torch_stream_id": 1234,
    }
    autotune = {"mode": "no-autotuner-observed"}
    call_core = {
        "call_id": "attention-0",
        "call_shape": call_shape,
        "artifact_id": artifact.artifact_id,
        "selected_compile_config": selected_config,
        "autotune": autotune,
        "launch_context": launch_context,
        "post_launcher_returned": True,
        "post_return_context_matches": True,
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "scope": {
            "vllm_attention": TARGET_VLLM_ENTRYPOINT,
            "gdn": (
                "actual Transformers Qwen3.5-MoE native eager GDN plus qcomem "
                "functional cache rebind; underlying ATen/CUDA libraries are out of scope"
            ),
        },
        "hook_installation": dict(HOOK_INSTALLATION_RECEIPT),
        "dispatch_source_bindings": {
            "vllm_unified_attention": source_binding(
                "runtime",
                "vllm/v1/attention/ops/triton_unified_attention.py",
                vllm_source,
                "vllm.v1.attention.ops.triton_unified_attention",
                "unified_attention",
            ),
            "vllm_kernel_unified_attention_source": source_binding(
                "runtime",
                "vllm/v1/attention/ops/triton_unified_attention.py",
                vllm_source,
                "vllm.v1.attention.ops.triton_unified_attention",
                "kernel_unified_attention",
            ),
            "triton_compiled_kernel_run_property_getter": source_binding(
                "runtime",
                "triton/compiler/compiler.py",
                triton_compiler_source,
                "triton.compiler.compiler",
                "CompiledKernel.run",
            ),
            "triton_autotuner_run": source_binding(
                "runtime",
                "triton/runtime/autotuner.py",
                triton_autotuner_source,
                "triton.runtime.autotuner",
                "Autotuner.run",
            ),
            "transformers_accelerate_wrapper": source_binding(
                "runtime",
                "transformers/integrations/accelerate.py",
                accelerate_source,
                "transformers.integrations.accelerate",
                "force_accelerate_hooks.<locals>.decorator.<locals>.wrapped",
            ),
            "transformers_qwen35_moe_gdn_forward": source_binding(
                "runtime",
                "transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
                qwen_source,
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
                "Qwen3_5MoeGatedDeltaNet.forward",
            ),
            "transformers_qwen35_moe_torch_chunk_gated_delta_rule": source_binding(
                "runtime",
                "transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
                qwen_source,
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
                "torch_chunk_gated_delta_rule",
            ),
            "transformers_qwen35_moe_torch_recurrent_gated_delta_rule": source_binding(
                "runtime",
                "transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
                qwen_source,
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
                "torch_recurrent_gated_delta_rule",
            ),
            "transformers_qwen35_moe_torch_causal_conv1d_update": source_binding(
                "runtime",
                "transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py",
                qwen_source,
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
                "torch_causal_conv1d_update",
            ),
            "qcomem_install_native_functional_linear_cache": source_binding(
                "code",
                native_source.name,
                native_source,
                "qcomem_qwen35_native_cache",
                "install_native_functional_linear_cache",
            ),
            "qcomem_functional_update_conv_state": source_binding(
                "code",
                native_source.name,
                native_source,
                "qcomem_qwen35_native_cache",
                "_functional_update_conv_state",
            ),
            "qcomem_functional_update_recurrent_state": source_binding(
                "code",
                native_source.name,
                native_source,
                "qcomem_qwen35_native_cache",
                "_functional_update_recurrent_state",
            ),
        },
        "attention_calls": [
            {
                "call_index": 0,
                "call_id": "attention-0",
                "vllm_entrypoint": TARGET_VLLM_ENTRYPOINT,
                "call_shape": call_shape,
                "selected_compiled_artifact": artifact.as_dict(),
                "selected_compile_config": selected_config,
                "autotune": autotune,
                "launch_context": launch_context,
                "post_launcher_returned": True,
                "post_return_context_matches": True,
                "call_receipt_sha256": _sha256_bytes(_canonical_bytes(call_core)),
            }
        ],
        "gdn_calls": [
            {
                "call_index": 0,
                "function": GDN_FUNCTION,
                "layer_idx": 0,
                "sequence_length": 4033,
                "cache_has_previous_state": False,
                "execution_phase": "document-prefill",
                "dispatch_kind": GDN_DISPATCH_KIND,
                "forward_source_bindings": [
                    "transformers_accelerate_wrapper",
                    "transformers_qwen35_moe_gdn_forward",
                ],
                "selected_gdn_kernel": {
                    "route": "multi-token",
                    "function": "torch_chunk_gated_delta_rule",
                    "source_binding": "transformers_qwen35_moe_torch_chunk_gated_delta_rule",
                    "observed_calls": 1,
                    "dispatch_kind": "eager-torch-fallback",
                    "compiled_artifact": None,
                    "autotune": {"mode": "not-applicable-to-eager-torch-fallback"},
                },
                "selected_conv_route": {
                    "function": "qcomem functional update_conv_state",
                    "source_binding": "qcomem_functional_update_conv_state",
                    "observed_calls": 1,
                    "state_semantics": "functional-cache-rebind",
                },
                "functional_cache_rebind": {
                    "install_source_binding": "qcomem_install_native_functional_linear_cache",
                    "conv_update_source_binding": "qcomem_functional_update_conv_state",
                    "recurrent_update_source_binding": "qcomem_functional_update_recurrent_state",
                    "conv_update_calls": 1,
                    "inplace_conv_update_calls": 0,
                    "recurrent_update_calls": 1,
                },
            }
        ],
    }
    return cache, code, runtime, payload


def run_negative_controls(output: Path) -> dict[str, Any]:
    """Prove detached replay rejects receipt and artifact substitutions."""

    with tempfile.TemporaryDirectory(prefix="r39-dispatch-") as temporary:
        cache, code, runtime, payload = _demo_fixture(Path(temporary))
        verify_payload(
            payload, cache_root=cache, code_root=code, runtime_root=runtime
        )
        controls: dict[str, str] = {}
        for name, mutate in {
            "receipt-config-tamper": lambda item: item["attention_calls"][0]["selected_compile_config"].__setitem__("num_warps", 8),
            "receipt-artifact-id-tamper": lambda item: item["attention_calls"][0]["selected_compiled_artifact"].__setitem__("artifact_id", "0" * 64),
        }.items():
            candidate = copy.deepcopy(payload)
            mutate(candidate)
            try:
                verify_payload(
                    candidate,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                )
            except DispatchReceiptError:
                controls[name] = "rejected"
            else:  # pragma: no cover - guard for regressions.
                raise DispatchReceiptError(f"negative control unexpectedly passed: {name}")
        # Required files and the exact directory file set are both part of the
        # artifact digest.  Missing and newly added cache members therefore
        # fail even if the compiler metadata JSON remains untouched.
        for name, mutate_cache in {
            "missing-required-ptx": lambda directory: next(directory.glob("*.ptx")).unlink(),
            "extra-unreceipted-artifact": lambda directory: (directory / "unexpected.bin").write_bytes(b"extra"),
        }.items():
            isolated = Path(tempfile.mkdtemp(prefix="r39-dispatch-control-"))
            try:
                control_cache, control_code, control_runtime, control_payload = (
                    _demo_fixture(isolated)
                )
                mutate_cache(next(control_cache.iterdir()))
                try:
                    verify_payload(
                        control_payload,
                        cache_root=control_cache,
                        code_root=control_code,
                        runtime_root=control_runtime,
                    )
                except DispatchReceiptError:
                    controls[name] = "rejected"
                else:  # pragma: no cover - guard for regressions.
                    raise DispatchReceiptError(f"negative control unexpectedly passed: {name}")
            finally:
                shutil.rmtree(isolated)
        # A byte-for-byte substitution of the selected cubin must invalidate its
        # enclosing artifact receipt even though the metadata JSON is unchanged.
        cubin = next(cache.rglob("*.cubin"))
        cubin.write_bytes(b"substituted-cubin")
        try:
            verify_payload(
                payload, cache_root=cache, code_root=code, runtime_root=runtime
            )
        except DispatchReceiptError:
            controls["compiled-artifact-substitution"] = "rejected"
        else:  # pragma: no cover - guard for regressions.
            raise DispatchReceiptError("artifact substitution unexpectedly passed")
    result = {
        "schema_version": SCHEMA_VERSION,
        "negative_controls": controls,
        "all_rejected": all(value == "rejected" for value in controls.values()),
    }
    _write_json(output, result)
    return result


def snapshot_bound_sources(
    *,
    payload: Mapping[str, Any],
    code_root: Path,
    runtime_root: Path,
    target: Path,
    output: Path,
) -> dict[str, Any]:
    """Copy only receipt-bound source files into a self-contained snapshot."""

    _require(not target.exists(), "source-snapshot target must be absent")
    bindings = _verify_dispatch_source_bindings(
        payload, code_root=code_root, runtime_root=runtime_root
    )
    copied: dict[tuple[str, str], dict[str, str]] = {}
    for raw in bindings.values():
        binding = _as_mapping(raw, "dispatch source binding")
        root_kind = str(binding["root_kind"])
        relative = str(binding["relative_source_path"])
        key = (root_kind, relative)
        if key in copied:
            continue
        source = _binding_source(
            binding, code_root=code_root, runtime_root=runtime_root
        )
        destination = target.joinpath(root_kind, *PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        _regular_file(destination, "snapshotted dispatch source")
        _require(
            _sha256_file(destination) == binding["source_sha256"],
            "snapshotted dispatch source hash drift",
        )
        copied[key] = {
            "root_kind": root_kind,
            "relative_source_path": relative,
            "sha256": _sha256_file(destination),
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "source_file_count": len(copied),
        "files": sorted(
            copied.values(),
            key=lambda item: (item["root_kind"], item["relative_source_path"]),
        ),
    }
    _write_json(output, result)
    return result


def _copy_bound_inputs(
    *,
    payload: Mapping[str, Any],
    cache_root: Path,
    code_root: Path,
    runtime_root: Path,
    target: Path,
) -> tuple[Path, Path, Path]:
    copied_cache = target / "cache"
    shutil.copytree(cache_root, copied_cache, symlinks=True)
    source_snapshot = target / "sources"
    snapshot_bound_sources(
        payload=payload,
        code_root=code_root,
        runtime_root=runtime_root,
        target=source_snapshot,
        output=target / "source-snapshot.json",
    )
    return copied_cache, source_snapshot / "code", source_snapshot / "runtime"


def _selected_artifact_file(
    payload: Mapping[str, Any], cache_root: Path, suffix: str
) -> Path:
    attention = payload.get("attention_calls")
    _require(isinstance(attention, list) and attention, "attention receipt list missing")
    row = _as_mapping(attention[0], "first attention row")
    artifact = _as_mapping(row.get("selected_compiled_artifact"), "selected artifact")
    files = artifact.get("files")
    _require(isinstance(files, list), "selected artifact file list missing")
    matches: list[Path] = []
    for item_raw in files:
        item = _as_mapping(item_raw, "selected artifact file")
        relative_path = item.get("relative_path")
        if isinstance(relative_path, str) and PurePosixPath(relative_path).suffix == suffix:
            relative = PurePosixPath(relative_path)
            _require(not relative.is_absolute() and ".." not in relative.parts, "artifact path escapes cache root")
            matches.append(cache_root.joinpath(*relative.parts))
    _require(len(matches) == 1, f"expected one selected {suffix} artifact, found {len(matches)}")
    _regular_file(matches[0], f"selected {suffix} artifact")
    return matches[0]


def run_bound_negative_controls(
    *,
    receipt: Path,
    cache_root: Path,
    code_root: Path,
    runtime_root: Path,
    output: Path,
) -> dict[str, Any]:
    """Apply all substitution controls to the actual captured receipt bundle."""

    payload_raw = json.loads(receipt.read_text(encoding="utf-8"))
    payload = _as_mapping(payload_raw, "receipt")
    verify_payload(
        payload,
        cache_root=cache_root,
        code_root=code_root,
        runtime_root=runtime_root,
    )
    controls: dict[str, str] = {}

    for name, mutate in {
        "receipt-config-tamper": lambda item: item["attention_calls"][0]["selected_compile_config"].__setitem__("num_warps", 2**30),
        "receipt-artifact-id-tamper": lambda item: item["attention_calls"][0]["selected_compiled_artifact"].__setitem__("artifact_id", "0" * 64),
    }.items():
        candidate = copy.deepcopy(payload)
        mutate(candidate)
        try:
            verify_payload(
                candidate,
                cache_root=cache_root,
                code_root=code_root,
                runtime_root=runtime_root,
            )
        except DispatchReceiptError:
            controls[name] = "rejected"
        else:  # pragma: no cover - guard for regressions.
            raise DispatchReceiptError(f"bound negative control unexpectedly passed: {name}")

    def mutate_source(
        item: Mapping[str, Any],
        code: Path,
        runtime: Path,
        binding_name: str,
    ) -> None:
        bindings = _as_mapping(
            item.get("dispatch_source_bindings"), "dispatch source bindings"
        )
        binding = _as_mapping(
            bindings.get(binding_name), f"dispatch binding {binding_name}"
        )
        _binding_source(
            binding, code_root=code, runtime_root=runtime
        ).write_bytes(b"# substituted dispatch source\n")

    cache_mutations: dict[
        str, Callable[[Mapping[str, Any], Path, Path, Path], None]
    ] = {
        "missing-required-ptx": lambda item, cache, code, runtime: _selected_artifact_file(item, cache, ".ptx").unlink(),
        "extra-unreceipted-artifact": lambda item, cache, code, runtime: (
            _selected_artifact_file(item, cache, ".cubin").parent / "unexpected.bin"
        ).write_bytes(b"extra"),
        "compiled-artifact-substitution": lambda item, cache, code, runtime: _selected_artifact_file(item, cache, ".cubin").write_bytes(b"substituted-cubin"),
        "gdn-runtime-source-substitution": lambda item, cache, code, runtime: mutate_source(
            item,
            code,
            runtime,
            "transformers_qwen35_moe_torch_chunk_gated_delta_rule",
        ),
        "gdn-cache-rebind-source-substitution": lambda item, cache, code, runtime: mutate_source(
            item, code, runtime, "qcomem_functional_update_conv_state"
        ),
    }
    for name, mutate in cache_mutations.items():
        with tempfile.TemporaryDirectory(prefix=f"r39-bound-{name}-") as temporary:
            isolated_cache, isolated_code, isolated_runtime = _copy_bound_inputs(
                payload=payload,
                cache_root=cache_root,
                code_root=code_root,
                runtime_root=runtime_root,
                target=Path(temporary),
            )
            mutate(payload, isolated_cache, isolated_code, isolated_runtime)
            try:
                verify_payload(
                    payload,
                    cache_root=isolated_cache,
                    code_root=isolated_code,
                    runtime_root=isolated_runtime,
                )
            except DispatchReceiptError:
                controls[name] = "rejected"
            else:  # pragma: no cover - guard for regressions.
                raise DispatchReceiptError(f"bound negative control unexpectedly passed: {name}")

    result = {
        "schema_version": SCHEMA_VERSION,
        "control_basis": "actual-captured-receipt-and-artifacts",
        "receipt_sha256": _sha256_file(receipt),
        "negative_controls": controls,
        "all_rejected": all(value == "rejected" for value in controls.values()),
    }
    _write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--receipt", type=Path, required=True)
    replay.add_argument("--triton-cache-root", type=Path, required=True)
    replay.add_argument("--code-root", type=Path, required=True)
    replay.add_argument("--runtime-root", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    controls = subparsers.add_parser("negative-controls")
    controls.add_argument("--output", type=Path, required=True)
    bound_controls = subparsers.add_parser("bound-negative-controls")
    bound_controls.add_argument("--receipt", type=Path, required=True)
    bound_controls.add_argument("--triton-cache-root", type=Path, required=True)
    bound_controls.add_argument("--code-root", type=Path, required=True)
    bound_controls.add_argument("--runtime-root", type=Path, required=True)
    bound_controls.add_argument("--output", type=Path, required=True)
    snapshot = subparsers.add_parser("snapshot-sources")
    snapshot.add_argument("--receipt", type=Path, required=True)
    snapshot.add_argument("--code-root", type=Path, required=True)
    snapshot.add_argument("--runtime-root", type=Path, required=True)
    snapshot.add_argument("--target", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "replay":
        payload = json.loads(args.receipt.read_text(encoding="utf-8"))
        result = verify_payload(
            payload,
            cache_root=args.triton_cache_root,
            code_root=args.code_root,
            runtime_root=args.runtime_root,
        )
        _write_json(args.output, result)
    elif args.command == "negative-controls":
        run_negative_controls(args.output)
    elif args.command == "bound-negative-controls":
        run_bound_negative_controls(
            receipt=args.receipt,
            cache_root=args.triton_cache_root,
            code_root=args.code_root,
            runtime_root=args.runtime_root,
            output=args.output,
        )
    else:
        payload = _as_mapping(
            json.loads(args.receipt.read_text(encoding="utf-8")), "receipt"
        )
        snapshot_bound_sources(
            payload=payload,
            code_root=args.code_root,
            runtime_root=args.runtime_root,
            target=args.target,
            output=args.output,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
