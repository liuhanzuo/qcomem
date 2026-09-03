from __future__ import annotations

"""Real-Qwen3.5 exactness/capacity runner for functional cache + paged KV.

Stage 1 is a hard capability gate:

* standard mutable eager and Q16 paged use the same document/query caller;
* final logits are close and greedy tokens are exact;
* all config-derived full-attention layers hit the paged backend; and
* all config-derived GDN layers exist in the native functional-cache plan.

Only after that gate succeeds does the same process run the small 4K
multi-query reference benchmark.  The attention kernel is a Python reference,
so TTFT is recorded but never labelled a production speedup.
"""

import argparse
import copy
import hashlib
import os
import statistics
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from qcomem_deployment import MemoryRecorder, NvmlProcessSampler, environment_metadata
from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
from qcomem_qwen35_native_cache import install_native_functional_linear_cache
from qcomem_qwen35_paged_integration import (
    KERNEL_MODE,
    PagedAttentionHitLedger,
    clone_dense_and_prepare_paged_cache_pair,
    register_qwen35_paged_backend,
    require_passed_reference_gate_before_benchmark,
    run_same_caller_eager_paged_gate,
)
from qcomem_torch import cache_nbytes, clone_cache
from run_deployment_bench import longbench_workloads
from run_downstream import atomic_json


def _sync() -> None:
    torch.cuda.synchronize()


def _visible_nvml_index() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        first = visible.split(",", 1)[0].strip()
        if first.isdigit():
            return int(first)
    return torch.cuda.current_device()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_backbone(model: Any) -> Any:
    if hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model.model, "layers"):
        return model.model
    raise RuntimeError("cannot resolve Qwen3.5 text backbone")


def _model_manifest_sha(model_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    names = ["config.json", "generation_config.json", "model.safetensors.index.json"]
    rows = []
    manifest = hashlib.sha256()
    for name in names:
        path = model_dir / name
        if not path.is_file():
            raise RuntimeError(f"frozen model manifest file is missing: {path}")
        sha = _sha256(path)
        size = path.stat().st_size
        rows.append({"name": name, "sha256": sha, "bytes": size})
        manifest.update(f"{name}\0{sha}\0{size}\n".encode())
    return manifest.hexdigest(), rows


def _tokens(value: torch.Tensor, limit: int) -> torch.Tensor:
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2:
        raise RuntimeError("tokens must be rank 1 or 2")
    return value[:, :limit]


def _build_dense_document_cache(
    backbone: Any,
    document: torch.Tensor,
    *,
    functional_linear: bool,
) -> tuple[Any, Any | None]:
    # The standard cache is the exact same production caller used by the eager
    # baseline. Installing functional rebinds before prefill proves the native
    # 30-layer seam against the target Transformers build without changing the
    # real Qwen DecoderLayer/FLA kernels.
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=backbone.config)
    install = (
        install_native_functional_linear_cache(cache, backbone.config)
        if functional_linear
        else None
    )
    with torch.inference_mode():
        output = backbone(input_ids=document, past_key_values=cache, use_cache=True)
    if output.past_key_values is not cache:
        raise RuntimeError("Qwen document prefill returned a different cache")
    return cache, install


def _same_query_caller(
    backbone: Any,
    model: Any,
    query: torch.Tensor,
):
    def caller(cache: Any) -> Any:
        output = backbone(input_ids=query, past_key_values=cache, use_cache=True)
        logits = model.lm_head(output.last_hidden_state[:, -1:, :])
        return type("CallerOutput", (), {"logits": logits})()

    return caller


def _final_logits(output: Any) -> torch.Tensor:
    logits = getattr(output, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("same caller did not return logits")
    if logits.ndim == 3:
        logits = logits[:, -1, :]
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise RuntimeError("same caller logits are malformed")
    return logits


def _tensor_version(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor._version)
    except RuntimeError:
        return None


@torch.inference_mode()
def _native_same_caller_gate(
    *,
    caller: Any,
    text_config: Any,
    standard_cache: Any,
    native_cache: Any,
    install: Any,
    stack_plan: Any,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    standard = clone_cache(standard_cache)
    # Native method bindings target one concrete cache layer object and are not
    # assumed to survive/deepcopy correctly. Reinstall on the clone explicitly.
    native = clone_cache(native_cache)
    native_install = install_native_functional_linear_cache(
        native, text_config
    )
    if native_install.linear_layer_indices != install.linear_layer_indices:
        raise RuntimeError("native clone reinstall changed the linear layer set")
    before = {}
    for index in stack_plan.linear_layer_indices:
        layer = native.layers[index]
        conv = layer.conv_states.get(0)
        recurrent = layer.recurrent_states.get(0)
        if not isinstance(conv, torch.Tensor) or not isinstance(
            recurrent, torch.Tensor
        ):
            raise RuntimeError(f"native linear layer {index} state is missing")
        before[index] = {
            "conv": conv,
            "conv_version": _tensor_version(conv),
            "recurrent": recurrent,
            "recurrent_version": _tensor_version(recurrent),
        }
    original = text_config._attn_implementation
    try:
        text_config._attn_implementation = "eager"
        standard_logits = _final_logits(caller(standard))
        native_logits = _final_logits(caller(native))
    finally:
        text_config._attn_implementation = original

    rows = []
    for index in stack_plan.linear_layer_indices:
        layer = native.layers[index]
        row = before[index]
        after_conv = layer.conv_states.get(0)
        after_recurrent = layer.recurrent_states.get(0)
        old_versions_unchanged = (
            _tensor_version(row["conv"]) == row["conv_version"]
            and _tensor_version(row["recurrent"]) == row["recurrent_version"]
        )
        rebound = after_conv is not row["conv"] and after_recurrent is not row[
            "recurrent"
        ]
        if not rebound or not old_versions_unchanged:
            raise RuntimeError(
                f"native linear layer {index} did not rebind without mutating base"
            )
        rows.append(
            {
                "layer_idx": index,
                "conv_rebound": after_conv is not row["conv"],
                "recurrent_rebound": after_recurrent is not row["recurrent"],
                "old_versions_unchanged": old_versions_unchanged,
            }
        )
    error = (standard_logits.float() - native_logits.float()).abs()
    close = torch.allclose(standard_logits, native_logits, rtol=rtol, atol=atol)
    token_exact = torch.equal(
        torch.argmax(standard_logits, dim=-1),
        torch.argmax(native_logits, dim=-1),
    )
    full_lengths_standard = {
        index: standard.layers[index].get_seq_length()
        for index in stack_plan.full_attention_layer_indices
    }
    full_lengths_native = {
        index: native.layers[index].get_seq_length()
        for index in stack_plan.full_attention_layer_indices
    }
    lengths_exact = full_lengths_standard == full_lengths_native
    passed = bool(close and token_exact and lengths_exact)
    result = {
        "passed": passed,
        "same_caller_object": True,
        "baseline": "stock-transformers-mutable-eager",
        "candidate": "native-qwen-kernels-functional-linear-state-rebind",
        "config_derived": True,
        "final_logits_close": bool(close),
        "final_tokens_exact": bool(token_exact),
        "cache_lengths_exact": bool(lengths_exact),
        "max_abs_logit_error": float(error.max().item()),
        "rtol": rtol,
        "atol": atol,
        "expected_linear_layer_count": len(stack_plan.linear_layer_indices),
        "observed_linear_layer_count": len(rows),
        "linear_layers": tuple(rows),
        "mutable_linear_copy_updates_used": False,
        "fallback_layers": (),
    }
    if not passed:
        raise RuntimeError(
            "stock/native same-caller gate failed: "
            f"close={close}, token_exact={token_exact}, lengths={lengths_exact}, "
            f"max_abs={result['max_abs_logit_error']}"
        )
    return result


def _native_layer_gate(cache: Any, install: Any, stack_plan: Any) -> dict[str, Any]:
    linear = tuple(install.linear_layer_indices)
    full = tuple(install.full_attention_layer_indices)
    if linear != stack_plan.linear_layer_indices:
        raise RuntimeError("native install did not cover every config linear layer")
    if full != stack_plan.full_attention_layer_indices:
        raise RuntimeError("native install full layer set differs from config")
    initialized_linear = []
    for index in linear:
        layer = cache.layers[index]
        if getattr(layer, "_qcomem_update_mode", None) != "functional-state-rebind":
            raise RuntimeError(f"linear cache layer {index} was not intercepted")
        conv = getattr(layer, "conv_states", {}).get(0)
        recurrent = getattr(layer, "recurrent_states", {}).get(0)
        if not isinstance(conv, torch.Tensor) or not isinstance(
            recurrent, torch.Tensor
        ):
            raise RuntimeError(f"linear cache layer {index} did not initialize state")
        initialized_linear.append(index)
    return {
        "passed": True,
        "config_derived": True,
        "expected_linear_layer_indices": stack_plan.linear_layer_indices,
        "observed_linear_layer_indices": tuple(initialized_linear),
        "expected_linear_layer_count": len(stack_plan.linear_layer_indices),
        "observed_linear_layer_count": len(initialized_linear),
        "expected_full_attention_layer_indices": stack_plan.full_attention_layer_indices,
        "observed_full_attention_layer_indices": full,
        "expected_full_attention_layer_count": len(
            stack_plan.full_attention_layer_indices
        ),
        "observed_full_attention_layer_count": len(full),
        "update_mode": install.update_mode,
        "mutable_linear_copy_updates_used": False,
        "fallback_layers": (),
    }


def _run_capability_gate(
    *,
    backbone: Any,
    model: Any,
    document: torch.Tensor,
    query: torch.Tensor,
    stack_plan: Any,
    page_size: int,
    group_size: int,
    append_page_size: int,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    """Run the independent stock/native/paged orchestration fail-closed.

    This helper deliberately owns both prefills so callers cannot accidentally
    compare two clones of one native-functional cache and label that comparison
    stock-vs-native.
    """

    stock_cache, stock_install = _build_dense_document_cache(
        backbone, document, functional_linear=False
    )
    if stock_install is not None:
        raise RuntimeError("stock baseline unexpectedly installed native seam")
    native_cache, native_install = _build_dense_document_cache(
        backbone, document, functional_linear=True
    )
    native_layer_gate = _native_layer_gate(
        native_cache, native_install, stack_plan
    )
    caller = _same_query_caller(backbone, model, query)
    native_exactness = _native_same_caller_gate(
        caller=caller,
        text_config=backbone.config,
        standard_cache=stock_cache,
        native_cache=native_cache,
        install=native_install,
        stack_plan=stack_plan,
        rtol=rtol,
        atol=atol,
    )
    pair = clone_dense_and_prepare_paged_cache_pair(
        stock_cache,
        stack_plan.full,
        page_size=page_size,
        bits=16,
        group_size=group_size,
        append_page_size=append_page_size,
    )
    ledger = PagedAttentionHitLedger(stack_plan.full, pair.conversion)
    backend = register_qwen35_paged_backend(ledger)
    paged_exactness = run_same_caller_eager_paged_gate(
        caller,
        text_config=backbone.config,
        caches=pair,
        backend=backend,
        rtol=rtol,
        atol=atol,
    )
    benchmark_gate = require_passed_reference_gate_before_benchmark(
        paged_exactness
    )
    gate = {
        "passed": bool(
            native_layer_gate["passed"]
            and native_exactness["passed"]
            and paged_exactness["passed"]
        ),
        "native_functional": native_layer_gate,
        "native_same_caller": native_exactness,
        "paged_same_caller": paged_exactness,
        "benchmark_authorization": benchmark_gate,
        "config_derived_counts": stack_plan.metadata(),
    }
    if not gate["passed"]:
        raise RuntimeError("combined native/paged capability gate failed")
    return gate


def _persistent_paged_bytes(cache: Any, indices: tuple[int, ...]) -> int:
    return sum(cache.layers[index].stored_nbytes for index in indices)


def _request_paged_bytes(cache: Any, indices: tuple[int, ...]) -> tuple[int, int]:
    shared = private = 0
    for index in indices:
        store = cache.layers[index].store
        shared += sum(page.stored_nbytes for page in store.document_pages)
        private += sum(page.stored_nbytes for page in store.request_pages)
    return shared, private


def _fork_cache_metadata_sharing_tensors(persistent: Any) -> Any:
    """Deep-copy cache metadata while preserving every document tensor.

    Both Transformers ``DynamicLayer.update`` and our native linear adapter
    replace state tensors out of place.  A request may therefore reference the
    immutable document tensors directly until its first transition.  Seeding
    ``deepcopy``'s memo avoids a transient full-document CUDA clone and makes
    allocator peaks describe request work rather than fork staging.
    """

    memo: dict[int, Any] = {}
    seen: set[int] = set()

    def seed(value: Any) -> None:
        identity = id(value)
        if identity in seen:
            return
        seen.add(identity)
        if isinstance(value, torch.Tensor):
            memo[identity] = value
        elif isinstance(value, dict):
            for item in value.values():
                seed(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                seed(item)
        elif hasattr(value, "__dict__"):
            for item in vars(value).values():
                seed(item)

    seed(persistent)
    request = copy.deepcopy(persistent, memo)
    if request is persistent:
        raise RuntimeError("cache metadata fork returned the persistent object")
    return request


def _fork_native_paged_cache(
    persistent: Any,
    plan: Any,
) -> tuple[Any, dict[str, Any]]:
    """Fork one hybrid cache while sharing immutable document state.

    The cache object and metadata dictionaries are private. Full-attention
    page payloads and GDN document tensors are shared at the fork boundary;
    native functional rebinds make the first query transition replace (not
    mutate) every GDN state tensor.
    """

    request = _fork_cache_metadata_sharing_tensors(persistent)
    install_native_functional_linear_cache(request, plan.gdn)
    shared_linear = 0
    for index in plan.linear_layer_indices:
        source = persistent.layers[index]
        target = request.layers[index]
        if getattr(target, "_qcomem_update_mode", None) != "functional-state-rebind":
            raise RuntimeError(f"forked linear layer {index} lost functional seam")
        for name in ("conv_states", "recurrent_states"):
            source_mapping = getattr(source, name, None)
            target_mapping = getattr(target, name, None)
            if not isinstance(source_mapping, dict) or not isinstance(
                target_mapping, dict
            ):
                raise RuntimeError(f"linear layer {index}.{name} is not a dict")
            for state_idx, tensor in source_mapping.items():
                if isinstance(tensor, torch.Tensor):
                    target_mapping[state_idx] = tensor
                    shared_linear += tensor.untyped_storage().nbytes()
    shared_page_bytes = 0
    for index in plan.full_attention_layer_indices:
        source = persistent.layers[index]
        target = source.fork()
        request.layers[index] = target
        if source.store.storage_keys != target.store.storage_keys:
            raise RuntimeError(f"full layer {index} fork did not share document pages")
        shared_page_bytes += source.stored_nbytes
    return request, {
        "shared_linear_document_state_nbytes": shared_linear,
        "shared_paged_document_nbytes": shared_page_bytes,
        "query_private_nbytes_at_fork": 0,
        "linear_functional_rebind": True,
        "full_page_payload_shared": True,
    }


def _fork_native_dense_cache(persistent: Any, plan: Any) -> tuple[Any, dict[str, Any]]:
    """Dense eager multi-query fork with shared linear document tensors."""

    request = _fork_cache_metadata_sharing_tensors(persistent)
    install_native_functional_linear_cache(request, plan.gdn)
    shared_linear = 0
    for index in plan.linear_layer_indices:
        source = persistent.layers[index]
        target = request.layers[index]
        if getattr(target, "_qcomem_update_mode", None) != "functional-state-rebind":
            raise RuntimeError(f"dense forked linear layer {index} lost seam")
        for name in ("conv_states", "recurrent_states"):
            for state_idx, tensor in getattr(source, name).items():
                if isinstance(tensor, torch.Tensor):
                    getattr(target, name)[state_idx] = tensor
                    shared_linear += tensor.untyped_storage().nbytes()
    shared_full = 0
    for index in plan.full_attention_layer_indices:
        source = persistent.layers[index]
        target = request.layers[index]
        for name in ("keys", "values"):
            source_tensor = getattr(source, name, None)
            target_tensor = getattr(target, name, None)
            if not isinstance(source_tensor, torch.Tensor) or not isinstance(
                target_tensor, torch.Tensor
            ):
                raise RuntimeError(f"dense full layer {index}.{name} is missing")
            if target_tensor is not source_tensor:
                raise RuntimeError(
                    f"dense full layer {index}.{name} was copied during fork"
                )
            shared_full += source_tensor.untyped_storage().nbytes()
    return request, {
        "shared_linear_document_state_nbytes": shared_linear,
        "shared_dense_full_document_nbytes": shared_full,
        "dense_full_kv_shared_at_fork": True,
        "full_document_staging_copy_nbytes": 0,
    }


def _verify_all_query_linear_states_rebound(
    requests: list[Any], persistent: Any, plan: Any
) -> dict[str, Any]:
    per_request = []
    for request_index, request in enumerate(requests):
        rebound = []
        for layer_index in plan.linear_layer_indices:
            base_layer = persistent.layers[layer_index]
            request_layer = request.layers[layer_index]
            for name in ("conv_states", "recurrent_states"):
                base = getattr(base_layer, name).get(0)
                current = getattr(request_layer, name).get(0)
                if not isinstance(base, torch.Tensor) or not isinstance(
                    current, torch.Tensor
                ):
                    raise RuntimeError(
                        f"linear layer {layer_index}.{name} state is missing"
                    )
                if current is base or (
                    current.untyped_storage().data_ptr()
                    == base.untyped_storage().data_ptr()
                ):
                    raise RuntimeError(
                        f"request {request_index} linear layer {layer_index}.{name} "
                        "did not become query-local"
                    )
            rebound.append(layer_index)
        per_request.append(
            {
                "request_index": request_index,
                "rebound_linear_layer_indices": tuple(rebound),
                "rebound_linear_layer_count": len(rebound),
            }
        )
    return {
        "verified": True,
        "expected_linear_layer_indices": plan.linear_layer_indices,
        "expected_linear_layer_count": len(plan.linear_layer_indices),
        "request_count": len(requests),
        "requests": tuple(per_request),
        "fallback_layers": (),
    }


@torch.inference_mode()
def _measure_stock_request(
    *,
    model: Any,
    backbone: Any,
    tokenizer: Any,
    document_cache: Any,
    plan: Any,
    query: torch.Tensor,
    max_new_tokens: int,
    queries_per_document: int,
    model_allocated_baseline_bytes: int,
    device_total_bytes: int,
    safety_headroom_bytes: int,
    warmup_count: int,
    nvml: NvmlProcessSampler,
) -> dict[str, Any]:
    persistent_linear_refs = {
        (index, name, state_idx): tensor
        for index in plan.linear_layer_indices
        for name in ("conv_states", "recurrent_states")
        for state_idx, tensor in getattr(document_cache.layers[index], name).items()
        if isinstance(tensor, torch.Tensor)
    }
    versions = {
        key: _tensor_version(tensor)
        for key, tensor in persistent_linear_refs.items()
    }
    original = backbone.config._attn_implementation
    try:
        backbone.config._attn_implementation = "eager"
        for _ in range(warmup_count):
            warmup_request, _ = _fork_native_dense_cache(document_cache, plan)
            warmup_output = backbone(
                input_ids=query,
                past_key_values=warmup_request,
                use_cache=True,
            )
            model.lm_head(warmup_output.last_hidden_state[:, -1:, :])
            _sync()
            del warmup_request, warmup_output
    finally:
        backbone.config._attn_implementation = original
    torch.cuda.empty_cache()
    _sync()
    recorder = MemoryRecorder(nvml)
    recorder.reset_peak()
    recorder.sample("request_start")
    original = backbone.config._attn_implementation
    requests = []
    forks = []
    per_query = []
    started = time.perf_counter()
    try:
        backbone.config._attn_implementation = "eager"
        for query_index in range(queries_per_document):
            request, fork = _fork_native_dense_cache(document_cache, plan)
            requests.append(request)
            forks.append(fork)
            current = query
            tokens = []
            ttft = None
            step_times = []
            for step in range(max_new_tokens):
                _sync()
                step_started = time.perf_counter()
                output = backbone(
                    input_ids=current,
                    past_key_values=request,
                    use_cache=True,
                )
                logits = model.lm_head(output.last_hidden_state[:, -1:, :])
                token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                _sync()
                elapsed = time.perf_counter() - step_started
                if step == 0:
                    ttft = elapsed
                else:
                    step_times.append(elapsed)
                tokens.append(token)
                current = torch.tensor([[token]], device=query.device)
                recorder.sample(f"query_{query_index}_decode_{step:03d}")
            per_query.append(
                {
                    "query_index": query_index,
                    "generated_token_ids": tokens,
                    "generated_text": tokenizer.decode(
                        tokens, skip_special_tokens=True
                    ),
                    "ttft_seconds": ttft,
                    "median_tpot_seconds": statistics.median(step_times),
                }
            )
    finally:
        backbone.config._attn_implementation = original
    wall = time.perf_counter() - started
    if any(
        _tensor_version(tensor) != versions[key]
        for key, tensor in persistent_linear_refs.items()
    ):
        raise RuntimeError("stock multi-query execution mutated persistent GDN base")
    query_linear_rebind = _verify_all_query_linear_states_rebound(
        requests, document_cache, plan
    )
    persistent_resident = cache_nbytes(document_cache)
    active_resident = cache_nbytes((document_cache, *requests))
    memory = recorder.summary()
    corpus_capacity = max(
        (
            device_total_bytes
            - model_allocated_baseline_bytes
            - safety_headroom_bytes
        )
        // max(persistent_resident, 1),
        0,
    )
    corpus_capacity_with_active_queries = max(
        (
            device_total_bytes
            - model_allocated_baseline_bytes
            - safety_headroom_bytes
            - (active_resident - persistent_resident)
        )
        // max(persistent_resident, 1),
        0,
    )
    return {
        "config": "dense-native-functional",
        "kernel_mode": "transformers-eager-native-qwen-functional-linear-rebind",
        "production_ttft_optimization_claim_allowed": False,
        "queries_per_document": queries_per_document,
        "warmup_count": warmup_count,
        "per_query": per_query,
        "generated_token_ids": per_query[0]["generated_token_ids"],
        "generated_text": per_query[0]["generated_text"],
        "ttft_seconds": statistics.median(
            float(row["ttft_seconds"]) for row in per_query
        ),
        "median_tpot_seconds": statistics.median(
            float(row["median_tpot_seconds"]) for row in per_query
        ),
        "request_wall_seconds": wall,
        "persistent_total_resident_nbytes": persistent_resident,
        "query_private_nbytes": active_resident - persistent_resident,
        "multi_query_active_total_resident_nbytes": active_resident,
        "full_cache_total_resident_nbytes": active_resident,
        "persistent_gdn_base_immutable": True,
        "query_linear_rebind": query_linear_rebind,
        "full_document_staging_copy_nbytes": 0,
        "auditable_corpus_capacity_documents": int(corpus_capacity),
        "auditable_corpus_capacity_with_active_queries": int(
            corpus_capacity_with_active_queries
        ),
        "capacity_model_allocated_baseline_bytes": model_allocated_baseline_bytes,
        "capacity_device_total_bytes": device_total_bytes,
        "capacity_safety_headroom_bytes": safety_headroom_bytes,
        "forks": forks,
        **memory,
    }


@torch.inference_mode()
def _measure_request(
    *,
    model: Any,
    backbone: Any,
    tokenizer: Any,
    document_cache: Any,
    plan: Any,
    query: torch.Tensor,
    bits: int,
    page_size: int,
    group_size: int,
    append_page_size: int,
    max_new_tokens: int,
    queries_per_document: int,
    model_allocated_baseline_bytes: int,
    device_total_bytes: int,
    safety_headroom_bytes: int,
    warmup_count: int,
    nvml: NvmlProcessSampler,
) -> dict[str, Any]:
    # Conversion happens once per document/config. Request forks copy metadata
    # only and share the packed document tensors, so measured allocator/NVML
    # peaks contain no per-request full-document staging clone.
    pair = clone_dense_and_prepare_paged_cache_pair(
        document_cache,
        plan.full,
        page_size=page_size,
        bits=bits,
        group_size=group_size,
        append_page_size=append_page_size,
    )
    persistent = pair.paged_cache
    persistent_linear_refs = {
        (index, name, state_idx): tensor
        for index in plan.linear_layer_indices
        for name in ("conv_states", "recurrent_states")
        for state_idx, tensor in getattr(persistent.layers[index], name).items()
        if isinstance(tensor, torch.Tensor)
    }
    persistent_linear_versions = {
        key: _tensor_version(tensor)
        for key, tensor in persistent_linear_refs.items()
    }
    original_impl = backbone.config._attn_implementation
    for _ in range(warmup_count):
        warmup_request, _ = _fork_native_paged_cache(persistent, plan)
        conversion = replace(
            pair.conversion,
            layer_store_ids={
                index: id(warmup_request.layers[index].store)
                for index in plan.full_attention_layer_indices
            },
        )
        warmup_ledger = PagedAttentionHitLedger(plan.full, conversion)
        warmup_backend = register_qwen35_paged_backend(warmup_ledger)
        try:
            backbone.config._attn_implementation = warmup_backend.name
            warmup_output = backbone(
                input_ids=query,
                past_key_values=warmup_request,
                use_cache=True,
            )
            model.lm_head(warmup_output.last_hidden_state[:, -1:, :])
            _sync()
        finally:
            backbone.config._attn_implementation = original_impl
        warmup_ledger.verify_complete()
        del warmup_request, warmup_output
    torch.cuda.empty_cache()
    _sync()
    recorder = MemoryRecorder(nvml)
    recorder.reset_peak()
    recorder.sample("request_start")
    torch.cuda.empty_cache()
    started = time.perf_counter()
    requests = []
    fork_rows = []
    per_query = []
    try:
        for query_index in range(queries_per_document):
            request, fork = _fork_native_paged_cache(persistent, plan)
            requests.append(request)
            fork_rows.append(fork)
            conversion = replace(
                pair.conversion,
                layer_store_ids={
                    index: id(request.layers[index].store)
                    for index in plan.full_attention_layer_indices
                },
            )
            ledger = PagedAttentionHitLedger(
                plan.full,
                conversion,
                expected_calls_per_layer=max_new_tokens,
            )
            backend = register_qwen35_paged_backend(ledger)
            tokens = []
            ttft = None
            step_times = []
            current = query
            backbone.config._attn_implementation = backend.name
            for step in range(max_new_tokens):
                _sync()
                step_started = time.perf_counter()
                output = backbone(
                    input_ids=current,
                    past_key_values=request,
                    use_cache=True,
                )
                logits = model.lm_head(output.last_hidden_state[:, -1:, :])
                token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
                _sync()
                elapsed = time.perf_counter() - step_started
                if step == 0:
                    ttft = elapsed
                else:
                    step_times.append(elapsed)
                tokens.append(token)
                current = torch.tensor([[token]], device=query.device)
                recorder.sample(f"query_{query_index}_decode_{step:03d}")
            intercept = ledger.verify_complete()
            per_query.append(
                {
                    "query_index": query_index,
                    "generated_token_ids": tokens,
                    "generated_text": tokenizer.decode(
                        tokens, skip_special_tokens=True
                    ),
                    "ttft_seconds": ttft,
                    "median_tpot_seconds": (
                        statistics.median(step_times) if step_times else None
                    ),
                    "intercept": intercept,
                }
            )
    finally:
        backbone.config._attn_implementation = original_impl
    wall = time.perf_counter() - started
    if any(
        _tensor_version(tensor) != persistent_linear_versions[key]
        for key, tensor in persistent_linear_refs.items()
    ):
        raise RuntimeError("multi-query execution mutated persistent GDN base")
    query_linear_rebind = _verify_all_query_linear_states_rebound(
        requests, persistent, plan
    )
    if any(
        request.layers[index].store.document_pages
        != persistent.layers[index].store.document_pages
        for request in requests
        for index in plan.full_attention_layer_indices
    ):
        raise RuntimeError("multi-query request lost persistent document pages")
    persistent_resident = cache_nbytes(persistent)
    active_resident = cache_nbytes((persistent, *requests))
    query_private = active_resident - persistent_resident
    memory = recorder.summary()
    corpus_capacity = max(
        (
            device_total_bytes
            - model_allocated_baseline_bytes
            - safety_headroom_bytes
        )
        // max(persistent_resident, 1),
        0,
    )
    corpus_capacity_with_active_queries = max(
        (
            device_total_bytes
            - model_allocated_baseline_bytes
            - safety_headroom_bytes
            - query_private
        )
        // max(persistent_resident, 1),
        0,
    )
    intercept = {
        "verified": all(row["intercept"]["verified"] for row in per_query),
        "dense_fallback_calls": sum(
            int(row["intercept"]["dense_fallback_calls"])
            for row in per_query
        ),
        "max_single_unpack_page_nbytes": max(
            int(call["materialization"]["max_single_unpack_page_nbytes"])
            for row in per_query
            for call in row["intercept"]["calls"]
        ),
        "max_materialized_kv_tokens": max(
            int(call["materialization"]["max_materialized_kv_tokens"])
            for row in per_query
            for call in row["intercept"]["calls"]
        ),
        "max_dense_full_kv_nbytes": max(
            int(call["materialization"]["dense_full_kv_nbytes"])
            for row in per_query
            for call in row["intercept"]["calls"]
        ),
        "queries": tuple(row["intercept"] for row in per_query),
    }
    return {
        "config": f"paged-q{bits}",
        "kernel_mode": KERNEL_MODE,
        "production_ttft_optimization_claim_allowed": False,
        "queries_per_document": queries_per_document,
        "warmup_count": warmup_count,
        "per_query": per_query,
        "generated_token_ids": per_query[0]["generated_token_ids"],
        "generated_text": per_query[0]["generated_text"],
        "ttft_seconds": statistics.median(
            float(row["ttft_seconds"]) for row in per_query
        ),
        "median_tpot_seconds": statistics.median(
            float(row["median_tpot_seconds"]) for row in per_query
        ),
        "request_wall_seconds": wall,
        "persistent_paged_document_nbytes": pair.conversion.paged_document_nbytes,
        "dense_document_kv_nbytes": pair.conversion.dense_document_nbytes,
        "query_shared_document_nbytes": pair.conversion.paged_document_nbytes,
        "persistent_total_resident_nbytes": persistent_resident,
        "query_private_nbytes": query_private,
        "multi_query_active_total_resident_nbytes": active_resident,
        "full_cache_total_resident_nbytes": active_resident,
        "forks": fork_rows,
        "persistent_gdn_base_immutable": True,
        "query_linear_rebind": query_linear_rebind,
        "persistent_full_pages_shared": True,
        "full_document_staging_copy_nbytes": 0,
        "auditable_corpus_capacity_documents": int(corpus_capacity),
        "auditable_corpus_capacity_with_active_queries": int(
            corpus_capacity_with_active_queries
        ),
        "capacity_model_allocated_baseline_bytes": model_allocated_baseline_bytes,
        "capacity_device_total_bytes": device_total_bytes,
        "capacity_safety_headroom_bytes": safety_headroom_bytes,
        "intercept": intercept,
        **memory,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--expected-source-indices", type=int, nargs="+", required=True)
    parser.add_argument("--expected-workloads", type=int, required=True)
    parser.add_argument("--source-index-start", type=int, default=6)
    parser.add_argument("--source-index-end", type=int, default=9)
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--gate-document-tokens", type=int, default=256)
    parser.add_argument("--gate-query-tokens", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--append-page-size", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--benchmark-bits", type=int, nargs="+", default=(16, 8, 4))
    parser.add_argument("--queries-per-document", type=int, default=2)
    parser.add_argument("--safety-headroom-gib", type=float, default=4.0)
    parser.add_argument("--warmup-count", type=int, default=1)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("rank must be within world size")
    if args.source_index_end >= 68:
        raise SystemExit("source >=68/test-v2 is forbidden")
    if tuple(args.benchmark_bits) != (16, 8, 4):
        raise SystemExit("formal benchmark bits are frozen to Q16,Q8,Q4")
    if args.queries_per_document < 2:
        raise SystemExit("multi-query benchmark requires at least two queries")
    if args.warmup_count != 1:
        raise SystemExit("formal paired benchmark freezes one warmup per config")
    data_sha = _sha256(args.data)
    if data_sha != args.expected_data_sha256:
        raise SystemExit("validation data SHA256 mismatch")
    args.exclude_source_indices = (4, 5)
    args.allow_test_v2 = False
    args.context_lengths = ()
    args.synthetic_repetitions = 0
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    workloads, workload_metadata = longbench_workloads(tokenizer, args)
    actual_indices = sorted({int(row["source_index"]) for row in workloads})
    if actual_indices != sorted(set(args.expected_source_indices)):
        raise SystemExit("source index hard gate failed")
    if len(workloads) != args.expected_workloads:
        raise SystemExit("workload count hard gate failed")
    if workload_metadata.get("source_revisions") != [args.expected_source_revision]:
        raise SystemExit("source revision hard gate failed")
    if workload_metadata.get("test_v2_consumed"):
        raise SystemExit("test-v2 was consumed")
    rank_workloads = workloads[args.rank :: args.world_size]
    if not rank_workloads:
        raise SystemExit("rank has no frozen validation workload")

    torch.cuda.set_device(0)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    _sync()
    model_allocated_baseline_bytes = torch.cuda.memory_allocated()
    device_total_bytes = torch.cuda.get_device_properties(0).total_memory
    safety_headroom_bytes = round(args.safety_headroom_gib * 2**30)
    backbone = _resolve_backbone(model)
    stack_plan = audit_qwen35_functional_stack_plan(model)
    first = rank_workloads[0]
    gate_document = _tokens(first["document_tokens"], args.gate_document_tokens).cuda()
    gate_query = _tokens(first["query_tokens"], args.gate_query_tokens).cuda()
    gate = _run_capability_gate(
        backbone=backbone,
        model=model,
        document=gate_document,
        query=gate_query,
        stack_plan=stack_plan,
        page_size=args.page_size,
        group_size=args.group_size,
        append_page_size=args.append_page_size,
        rtol=args.rtol,
        atol=args.atol,
    )
    model_manifest_sha, model_manifest = _model_manifest_sha(args.model)
    destination = args.run_dir / f"paged-real-shard-{args.rank}.json"
    base = {
        "status": "exactness_gate_passed" if args.gate_only else "running",
        "rank": args.rank,
        "world_size": args.world_size,
        "kernel_mode": KERNEL_MODE,
        "production_ttft_optimization_claim_allowed": False,
        "gate": gate,
        "workload_metadata": workload_metadata,
        "model": str(args.model),
        "model_manifest_sha256": model_manifest_sha,
        "model_manifest": model_manifest,
        "environment": environment_metadata(model),
        "protocol": {
            key: (
                str(value)
                if isinstance(value, Path)
                else list(value)
                if isinstance(value, tuple)
                else value
            )
            for key, value in vars(args).items()
        },
        "rows": [],
    }
    atomic_json(destination, base)
    if args.gate_only:
        print(f"PAGED_REAL_GATE_PASSED {destination}", flush=True)
        return

    del gate_document, gate_query
    torch.cuda.empty_cache()
    nvml = NvmlProcessSampler(_visible_nvml_index())
    rows = []
    for workload in rank_workloads:
        document = _tokens(workload["document_tokens"], args.max_input_tokens).cuda()
        query = _tokens(workload["query_tokens"], args.max_input_tokens).cuda()
        native_cache, native_install = _build_dense_document_cache(
            backbone, document, functional_linear=True
        )
        native = _native_layer_gate(native_cache, native_install, stack_plan)
        config_order = (
            ("dense-native-functional", "paged-q16", "paged-q8", "paged-q4")
            if args.rank % 2 == 0
            else ("paged-q4", "paged-q8", "paged-q16", "dense-native-functional")
        )
        measurements = {}
        for config_name in config_order:
            if config_name == "dense-native-functional":
                measurements[config_name] = _measure_stock_request(
                    model=model,
                    backbone=backbone,
                    tokenizer=tokenizer,
                    document_cache=native_cache,
                    plan=stack_plan,
                    query=query,
                    max_new_tokens=args.max_new_tokens,
                    queries_per_document=args.queries_per_document,
                    model_allocated_baseline_bytes=model_allocated_baseline_bytes,
                    device_total_bytes=device_total_bytes,
                    safety_headroom_bytes=safety_headroom_bytes,
                    warmup_count=args.warmup_count,
                    nvml=nvml,
                )
            else:
                bits = int(config_name.removeprefix("paged-q"))
                measurements[config_name] = _measure_request(
                    model=model,
                    backbone=backbone,
                    tokenizer=tokenizer,
                    document_cache=native_cache,
                    plan=stack_plan,
                    query=query,
                    bits=bits,
                    page_size=args.page_size,
                    group_size=args.group_size,
                    append_page_size=args.append_page_size,
                    max_new_tokens=args.max_new_tokens,
                    queries_per_document=args.queries_per_document,
                    model_allocated_baseline_bytes=model_allocated_baseline_bytes,
                    device_total_bytes=device_total_bytes,
                    safety_headroom_bytes=safety_headroom_bytes,
                    warmup_count=args.warmup_count,
                    nvml=nvml,
                )
        stock = measurements["dense-native-functional"]
        paired = {}
        for name in ("paged-q16", "paged-q8", "paged-q4"):
            candidate = measurements[name]
            token_exact = all(
                left["generated_token_ids"] == right["generated_token_ids"]
                for left, right in zip(stock["per_query"], candidate["per_query"])
            )
            paired[name] = {
                "generated_tokens_exact": token_exact,
                "persistent_total_resident_ratio_vs_stock": (
                    candidate["persistent_total_resident_nbytes"]
                    / stock["persistent_total_resident_nbytes"]
                ),
                "multi_query_active_ratio_vs_stock": (
                    candidate["multi_query_active_total_resident_nbytes"]
                    / stock["multi_query_active_total_resident_nbytes"]
                ),
                "cuda_peak_ratio_vs_stock": (
                    candidate["cuda_peak_allocated_bytes"]
                    / stock["cuda_peak_allocated_bytes"]
                ),
                "nvml_peak_ratio_vs_stock": (
                    candidate["nvml_sampled_peak_process_bytes"]
                    / stock["nvml_sampled_peak_process_bytes"]
                ),
                "ttft_ratio_vs_stock_reference_only": (
                    candidate["ttft_seconds"] / stock["ttft_seconds"]
                ),
                "tpot_ratio_vs_stock_reference_only": (
                    candidate["median_tpot_seconds"]
                    / stock["median_tpot_seconds"]
                ),
            }
        rows.append(
            {
                "rank": args.rank,
                "workload_id": workload["workload_id"],
                "dataset": workload["dataset"],
                "source_index": workload["source_index"],
                "document_tokens": int(document.shape[1]),
                "query_tokens": int(query.shape[1]),
                "native_gate": native,
                "multi_query_semantics": (
                    "two concurrent request states over one document; this short "
                    "protocol intentionally repeats the same frozen query to isolate "
                    "state sharing and is not a diverse-query workload"
                ),
                "measurement_order": config_order,
                "measurements": measurements,
                "paired": paired,
            }
        )
        base["rows"] = rows
        atomic_json(destination, base)
        del native_cache, document, query
        torch.cuda.empty_cache()
    base["status"] = "completed_shard"
    base["rows"] = rows
    atomic_json(destination, base)
    print(f"PAGED_REAL_SHARD_COMPLETE {destination}", flush=True)


if __name__ == "__main__":
    main()
