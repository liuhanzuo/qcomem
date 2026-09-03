#!/usr/bin/env python3
"""Frozen-runtime, CPU-only installation/restoration smoke for R40 hooks.

This preflight imports the real pinned packages and instantiates the actual
Qwen3.5-MoE GDN class on CPU.  It must finish without initializing CUDA.  It
does not claim that a CUDA launcher ran; that is established only by fresh
per-call post-return receipts in the authorized formal run.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import r39_compiled_dispatch_receipts as receipts


SCHEMA_VERSION = "forkaudit-r40-frozen-runtime-no-cuda-smoke-v1"
EXPECTED_PACKAGES = {
    "torch": "2.11.0+cu129",
    "transformers": "5.14.1",
    "vllm": "0.26.0+cu129",
    "triton": "3.6.0",
}
CALLABLE_CHECKS = {
    "vllm_unified_attention_wrapper_installed",
    "triton_compiled_kernel_run_property_installed",
    "triton_autotuner_wrapper_installed",
    "gdn_instance_chunk_route_bound",
    "gdn_instance_recurrent_route_bound",
    "gdn_instance_inplace_conv_route_bound",
    "gdn_forward_wrapper_installed",
    "functional_conv_rebind_instance_bound",
    "functional_recurrent_rebind_instance_bound",
    "vllm_unified_attention_restored",
    "triton_compiled_kernel_run_restored",
    "triton_autotuner_restored",
    "gdn_routes_restored",
    "gdn_forward_restored",
    "functional_cache_routes_restored",
}
CLAIM_BOUNDARY = (
    "callable/source preflight only; CUDA remains uninitialized and no compiled "
    "launch is claimed until an authorized formal shard seals post-return receipts"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise receipts.DispatchReceiptError(message)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_exclusive(path: Path, value: Any) -> None:
    _require(not path.exists(), f"runtime smoke output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    _require(not temporary.exists(), f"runtime smoke temporary exists: {temporary}")
    temporary.write_bytes(receipts._canonical_bytes(value) + b"\n")
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise receipts.DispatchReceiptError(
            f"runtime smoke output already exists: {path}"
        ) from error
    temporary.unlink()


def load_and_verify_runtime_preflight(path: Path) -> dict[str, Any]:
    value = receipts._as_mapping(_load(path), "runtime preflight")
    receipts._exact_fields(
        value,
        {
            "schema_version",
            "python_version",
            "package_versions",
            "cuda_initialized_before",
            "cuda_initialized_after",
            "hook_installation",
            "dispatch_source_bindings",
            "callable_smoke",
            "claim_boundary",
        },
        "runtime preflight",
    )
    _require(value.get("schema_version") == SCHEMA_VERSION, "runtime preflight schema drift")
    python_version = value.get("python_version")
    _require(
        isinstance(python_version, str) and python_version.startswith("3.11."),
        "runtime preflight Python version drift",
    )
    _require(value.get("package_versions") == EXPECTED_PACKAGES, "runtime package versions drift")
    _require(value.get("cuda_initialized_before") is False, "runtime smoke initialized CUDA before hooks")
    _require(value.get("cuda_initialized_after") is False, "runtime smoke initialized CUDA")
    _require(
        value.get("hook_installation") == receipts.HOOK_INSTALLATION_RECEIPT,
        "runtime hook-installation receipt drift",
    )
    bindings = receipts._as_mapping(
        value.get("dispatch_source_bindings"), "runtime dispatch source bindings"
    )
    _require(set(bindings) == receipts.DISPATCH_SOURCE_KEYS, "runtime source-binding key drift")
    for key, raw in bindings.items():
        binding = receipts._as_mapping(raw, f"runtime source binding {key}")
        receipts._exact_fields(
            binding,
            {
                "root_kind",
                "relative_source_path",
                "source_sha256",
                "module",
                "qualname",
            },
            f"runtime source binding {key}",
        )
        _require(receipts._is_sha256(binding.get("source_sha256")), f"runtime source binding {key} SHA drift")
    checks = receipts._as_mapping(value.get("callable_smoke"), "runtime callable smoke")
    _require(set(checks) == CALLABLE_CHECKS, "runtime callable-smoke field drift")
    _require(all(item is True for item in checks.values()), "runtime callable smoke did not fully pass")
    _require(value.get("claim_boundary") == CLAIM_BOUNDARY, "runtime smoke claim boundary drift")
    return dict(value)


class _LinearLayer:
    def __init__(self) -> None:
        self.conv_states: dict[int, Any] = {}
        self.recurrent_states: dict[int, Any] = {}
        self.is_conv_states_initialized: dict[int, bool] = {}
        self.is_recurrent_states_initialized: dict[int, bool] = {}
        self.has_previous_state: dict[int, bool] = {}
        self.conv_kernel_size: dict[int, int] = {}
        self.record_past = False

    def lazy_initialization(self, **_kwargs: Any) -> None:
        return None


def run_smoke(*, code_root: Path, runtime_root: Path, cache_root: Path) -> dict[str, Any]:
    import torch

    _require(sys.version_info[:2] == (3, 11), "frozen runtime requires Python 3.11")
    versions = {
        name: (
            str(torch.__version__)
            if name == "torch"
            else importlib.metadata.version(name)
        )
        for name in EXPECTED_PACKAGES
    }
    _require(versions == EXPECTED_PACKAGES, f"frozen package version drift: {versions}")
    before = torch.cuda.is_initialized()
    _require(before is False, "CUDA was initialized before no-CUDA smoke")
    cache_root.mkdir(parents=True, exist_ok=False)
    os.environ["TRITON_CACHE_DIR"] = str(cache_root)

    def forbidden_launch_context() -> Mapping[str, Any]:
        raise receipts.DispatchReceiptError("CPU smoke unexpectedly reached a CUDA launcher")

    recorder = receipts.DispatchReceiptRecorder(
        cache_root=cache_root,
        code_root=code_root,
        runtime_root=runtime_root,
        launch_context_provider=forbidden_launch_context,
    )
    restore = receipts.install_runtime_hooks(recorder)

    from triton.compiler.compiler import CompiledKernel
    from triton.runtime.autotuner import Autotuner
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as qwen
    import qcomem_qwen35_native_cache as native_cache
    from vllm.v1.attention.ops import triton_unified_attention as vllm_attention

    patched_unified = vllm_attention.unified_attention
    patched_compiled = inspect.getattr_static(CompiledKernel, "run")
    patched_autotune = Autotuner.run
    patched_chunk = qwen.torch_chunk_gated_delta_rule
    patched_recurrent = qwen.torch_recurrent_gated_delta_rule
    patched_conv = qwen.torch_causal_conv1d_update
    patched_native_conv = native_cache._functional_update_conv_state
    patched_native_recurrent = native_cache._functional_update_recurrent_state
    patched_gdn_forward = qwen.Qwen3_5MoeGatedDeltaNet.forward
    _require(hasattr(patched_unified, "__wrapped__"), "vLLM wrapper was not installed")
    _require(
        isinstance(patched_compiled, property)
        and patched_compiled.fget is not None
        and patched_compiled.fget.__module__ == receipts.__name__,
        "Triton CompiledKernel.run property wrapper was not installed",
    )
    _require(hasattr(patched_autotune, "__wrapped__"), "Triton autotuner wrapper was not installed")
    _require(hasattr(patched_gdn_forward, "__wrapped__"), "GDN forward wrapper was not installed")

    tiny_config = SimpleNamespace(
        hidden_size=8,
        linear_num_value_heads=1,
        linear_num_key_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"],
    )
    gdn = qwen.Qwen3_5MoeGatedDeltaNet(tiny_config, 0)
    _require(gdn.chunk_gated_delta_rule is patched_chunk, "GDN chunk route did not bind wrapper")
    _require(gdn.recurrent_gated_delta_rule is patched_recurrent, "GDN recurrent route did not bind wrapper")
    _require(gdn.causal_conv1d_update is patched_conv, "GDN in-place conv route did not bind wrapper")

    linear_layer = _LinearLayer()
    cache = SimpleNamespace(layers=[linear_layer, object()])
    cache_config = SimpleNamespace(layer_types=["linear_attention", "full_attention"])
    native_cache.install_native_functional_linear_cache(cache, cache_config)
    _require(
        getattr(linear_layer.update_conv_state, "__func__", None) is patched_native_conv,
        "functional conv route did not bind wrapper",
    )
    _require(
        getattr(linear_layer.update_recurrent_state, "__func__", None)
        is patched_native_recurrent,
        "functional recurrent route did not bind wrapper",
    )

    original_unified = patched_unified.__wrapped__
    original_autotune = patched_autotune.__wrapped__
    original_chunk = patched_chunk.__wrapped__
    original_recurrent = patched_recurrent.__wrapped__
    original_conv = patched_conv.__wrapped__
    original_native_conv = patched_native_conv.__wrapped__
    original_native_recurrent = patched_native_recurrent.__wrapped__
    original_gdn_forward = patched_gdn_forward.__wrapped__
    original_compiled = recorder.dispatch_source_bindings[
        "triton_compiled_kernel_run_property_getter"
    ]
    restore()

    restored_compiled = inspect.getattr_static(CompiledKernel, "run")
    checks = {
        "vllm_unified_attention_wrapper_installed": True,
        "triton_compiled_kernel_run_property_installed": True,
        "triton_autotuner_wrapper_installed": True,
        "gdn_instance_chunk_route_bound": True,
        "gdn_instance_recurrent_route_bound": True,
        "gdn_instance_inplace_conv_route_bound": True,
        "gdn_forward_wrapper_installed": True,
        "functional_conv_rebind_instance_bound": True,
        "functional_recurrent_rebind_instance_bound": True,
        "vllm_unified_attention_restored": vllm_attention.unified_attention is original_unified,
        "triton_compiled_kernel_run_restored": (
            isinstance(restored_compiled, property)
            and restored_compiled.fget is not None
            and restored_compiled.fget.__module__ == original_compiled["module"]
            and restored_compiled.fget.__qualname__ == original_compiled["qualname"]
        ),
        "triton_autotuner_restored": Autotuner.run is original_autotune,
        "gdn_routes_restored": (
            qwen.torch_chunk_gated_delta_rule is original_chunk
            and qwen.torch_recurrent_gated_delta_rule is original_recurrent
            and qwen.torch_causal_conv1d_update is original_conv
        ),
        "gdn_forward_restored": qwen.Qwen3_5MoeGatedDeltaNet.forward is original_gdn_forward,
        "functional_cache_routes_restored": (
            native_cache._functional_update_conv_state is original_native_conv
            and native_cache._functional_update_recurrent_state is original_native_recurrent
        ),
    }
    _require(set(checks) == CALLABLE_CHECKS and all(checks.values()), "hook restoration smoke failed")
    after = torch.cuda.is_initialized()
    _require(after is False, "no-CUDA smoke initialized CUDA")
    return {
        "schema_version": SCHEMA_VERSION,
        "python_version": sys.version.split()[0],
        "package_versions": versions,
        "cuda_initialized_before": before,
        "cuda_initialized_after": after,
        "hook_installation": recorder.hook_installation,
        "dispatch_source_bindings": recorder.dispatch_source_bindings,
        "callable_smoke": checks,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = run_smoke(
        code_root=args.code_root.resolve(),
        runtime_root=args.runtime_root.resolve(),
        cache_root=args.cache_root.resolve(),
    )
    _write_exclusive(args.output.resolve(), value)
    load_and_verify_runtime_preflight(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
