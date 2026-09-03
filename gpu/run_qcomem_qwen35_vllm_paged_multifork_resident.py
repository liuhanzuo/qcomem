from __future__ import annotations

"""PG-19-only multi-fork resident capacity protocol.

Each rank keeps one approximately-4K PG-19 train document and runs the entire
``N={1,2,4,8,16,32}`` curve on one GPU.  Every N creates all request objects
before the first forward and serves eight generated tokens in deterministic
round-major order.  There is no LongBench path, validation argument, downstream
quality metric, engine scheduler, or concurrent-kernel throughput claim here.
"""

import argparse
import gc
import hashlib
import inspect
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from qcomem_joint_policy import (
    PG19_BUCKET,
    PG19_PREFIX,
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
    sha256_file,
)
from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
from qcomem_qwen35_native_cache import install_native_functional_linear_cache
from qcomem_qwen35_vllm_paged_integration import (
    convert_all_qwen35_full_layers_to_vllm_q16,
)
from qcomem_vllm_paged_fair_control import FRESH_CONTROL, SHARED_REUSE
from qcomem_vllm_paged_kernel import (
    KERNEL_MODE,
    QWEN35_AUDITED_GEOMETRY,
    _resolve_vllm_unified_attention,
    audit_frozen_kernel_environment,
)
from qcomem_vllm_paged_multifork_resident import (
    MULTIFORK_COUNTS,
    MULTIFORK_POLICIES,
    MULTIFORK_PROTOCOL,
    MultiForkHitLedger,
    build_pg19_train_query_bank,
    build_resident_request_group,
    linear_capacity_fit,
    register_multifork_backend,
    resident_storage_breakdown,
)
from run_downstream import atomic_json


FULL_LAYERS = tuple(range(3, 40, 4))
FORMAL_DOCUMENT_TOKENS = 4095
FORMAL_QUERY_TOKENS = 32
FORMAL_NEW_TOKENS = 8
FORMAL_PAGE_SIZE = 128
FORMAL_WORLD_SIZE = 8
FORMAL_BOOKS = 8
FORMAL_WINDOW_STRIDE = 257
FORMAL_CANDIDATES = 8
FORMAL_SEED = 20260814
FORMAL_EXECUTION_ORDER = (1, 32, 2, 16, 4, 8)
FROZEN_IDENTITY_FIELDS = (
    "code_ledger_sha256",
    "model_manifest_sha256",
    "model_artifact_ledger_sha256",
    "model_weight_ledger_sha256",
    "pg19_data_sha256",
    "pg19_manifest_sha256",
    "pg19_windows_sha256",
    "protocol_manifest_sha256",
    "protocol_config_sha256",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_frozen_json(path: Path, expected_sha256: str, label: str) -> Any:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError(f"{label} cannot be read") from error
    _require(_sha256_bytes(payload) == expected_sha256, f"{label} SHA256 mismatch")
    try:
        return json.loads(payload)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"{label} is not valid JSON") from error


def _sha256_tensor(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return _sha256_bytes(raw)


def _sync() -> None:
    torch.cuda.synchronize()


def _allocator_snapshot(label: str) -> dict[str, Any]:
    return {
        "label": label,
        "current_allocated_bytes": int(torch.cuda.memory_allocated()),
        "current_reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _allocator_cleanup(label: str) -> dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    _sync()
    return _allocator_snapshot(label)


def _allocator_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {
        "current_allocated_delta_bytes": (
            int(after["current_allocated_bytes"]) - int(before["current_allocated_bytes"])
        ),
        "current_reserved_delta_bytes": (
            int(after["current_reserved_bytes"]) - int(before["current_reserved_bytes"])
        ),
        "peak_allocated_delta_bytes": (
            int(after["peak_allocated_bytes"]) - int(before["current_allocated_bytes"])
        ),
        "peak_reserved_delta_bytes": (
            int(after["peak_reserved_bytes"]) - int(before["current_reserved_bytes"])
        ),
    }


def _unregister_backend(name: str) -> None:
    from transformers.masking_utils import AttentionMaskInterface
    from transformers.modeling_utils import AttentionInterface

    attention = AttentionInterface._global_mapping.pop(name, None)
    mask = AttentionMaskInterface._global_mapping.pop(name, None)
    _require(attention is not None and mask is not None, "backend registry cleanup failed")


def _resolve_backbone(model: Any) -> Any:
    if hasattr(model.model, "language_model"):
        return model.model.language_model
    if hasattr(model.model, "layers"):
        return model.model
    raise RuntimeError("cannot resolve Qwen3.5 text backbone")


def _model_manifest_sha(model_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    names = ("config.json", "generation_config.json", "model.safetensors.index.json")
    digest = hashlib.sha256()
    rows = []
    for name in names:
        path = model_dir / name
        _require(path.is_file(), f"model manifest file missing: {name}")
        file_digest = sha256_file(path)
        size = path.stat().st_size
        digest.update(f"{name}\0{file_digest}\0{size}\n".encode())
        rows.append({"name": name, "sha256": file_digest, "bytes": size})
    return digest.hexdigest(), rows


def _audit_model_config_geometry(model_dir: Path) -> dict[str, Any]:
    payload = json.loads((model_dir / "config.json").read_text())
    config = payload.get("text_config", payload)
    layer_types = config.get("layer_types")
    _require(isinstance(layer_types, list) and len(layer_types) == 40, "model layer_types drift")
    full = tuple(index for index, value in enumerate(layer_types) if value == "full_attention")
    linear = tuple(index for index, value in enumerate(layer_types) if value == "linear_attention")
    observed = {
        "num_query_heads": config.get("num_attention_heads"),
        "num_key_value_heads": config.get("num_key_value_heads"),
        "num_key_value_groups": (
            int(config.get("num_attention_heads"))
            // int(config.get("num_key_value_heads"))
        ),
        "head_dim": config.get("head_dim"),
        "full_attention_layers": len(full),
    }
    _require(observed == QWEN35_AUDITED_GEOMETRY, "Qwen3.5 attention geometry drift")
    _require(full == FULL_LAYERS and len(linear) == 30, "Qwen3.5 hybrid layer plan drift")
    return {
        "observed": observed,
        "num_hidden_layers": len(layer_types),
        "full_attention_layer_indices": list(full),
        "linear_attention_layer_count": len(linear),
        "matches_frozen_geometry": True,
    }


def _kernel_identity(kernel: Any) -> dict[str, Any]:
    try:
        signature = str(inspect.signature(kernel))
    except (TypeError, ValueError):
        signature = "<signature-unavailable>"
    return {
        "callable_id": id(kernel),
        "module": str(getattr(kernel, "__module__", type(kernel).__module__)),
        "qualname": str(getattr(kernel, "__qualname__", type(kernel).__qualname__)),
        "signature": signature,
    }


def _kernel_descriptor(identity: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(identity.get("module", "")),
        str(identity.get("qualname", "")),
        str(identity.get("signature", "")),
    )


def _validate_kernel_identity(identity: Any, label: str) -> dict[str, Any]:
    _require(isinstance(identity, dict), f"{label} kernel identity missing")
    _require(
        type(identity.get("callable_id")) is int and identity["callable_id"] > 0,
        f"{label} callable_id drift",
    )
    for field in ("module", "qualname", "signature"):
        _require(
            isinstance(identity.get(field), str) and bool(identity[field]),
            f"{label} {field} drift",
        )
    return identity


def _protocol_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": MULTIFORK_PROTOCOL,
        "bits": int(args.bits),
        "page_size": int(args.page_size),
        "world_size": int(args.world_size),
        "resident_counts": list(args.resident_counts),
        "execution_order": list(args.execution_order),
        "pg19_books": int(args.pg19_books),
        "pg19_document_tokens": int(args.pg19_document_tokens),
        "pg19_query_tokens": int(args.pg19_query_tokens),
        "pg19_window_stride": int(args.pg19_window_stride),
        "pg19_candidate_windows": int(args.pg19_candidate_windows),
        "pg19_seed": int(args.pg19_seed),
        "query_bank_stride": int(args.query_bank_stride),
        "max_new_tokens": int(args.max_new_tokens),
        "max_append_tokens": int(args.pg19_query_tokens + args.max_new_tokens),
        "quantization": "Q16",
        "batch_per_request": 1,
        "one_document_per_rank": True,
        "all_n_run_on_same_rank_document": True,
        "round_major_single_stream_service": True,
        "pg19_train_only": True,
        "longbench_consumed": False,
        "source_6_9_consumed": False,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
    }


def _protocol_config_sha256(config: dict[str, Any]) -> str:
    return _sha256_bytes(json.dumps(config, sort_keys=True, separators=(",", ":")).encode())


def _static_frozen_identity(static: dict[str, Any]) -> dict[str, str]:
    identity = {field: str(static.get(field, "")) for field in FROZEN_IDENTITY_FIELDS}
    for field, value in identity.items():
        _require(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"missing frozen {field}")
    return identity


def _build_document_cache(backbone: Any, document: torch.Tensor) -> Any:
    from transformers.cache_utils import DynamicCache

    original = backbone.config._attn_implementation
    cache = DynamicCache(config=backbone.config)
    install_native_functional_linear_cache(cache, backbone.config)
    try:
        backbone.config._attn_implementation = "eager"
        output = backbone(input_ids=document, past_key_values=cache, use_cache=True)
    finally:
        backbone.config._attn_implementation = original
    _require(output.past_key_values is cache, "document prefill returned a different cache")
    return cache


def _set_production_no_mask(group: Any, layer_indices: Sequence[int]) -> None:
    for request in group.requests:
        for index in layer_indices:
            request.layers[index].sequence.strict_mask_check = False


def _source_document_digests(cache: Any, layer_indices: Sequence[int]) -> dict[str, str]:
    rows = {}
    for index in layer_indices:
        arena = cache.layers[index].arena
        digest = hashlib.sha256()
        for batch_index in range(arena.batch_size):
            for logical in range(arena.document_blocks_per_sequence):
                start = logical * arena.page_size
                valid = min(arena.page_size, arena.document_length - start)
                physical = batch_index * arena.document_blocks_per_sequence + logical
                digest.update(
                    arena.key_cache[physical, :valid]
                    .detach()
                    .contiguous()
                    .cpu()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
                digest.update(
                    arena.value_cache[physical, :valid]
                    .detach()
                    .contiguous()
                    .cpu()
                    .view(torch.uint8)
                    .numpy()
                    .tobytes()
                )
        rows[str(index)] = digest.hexdigest()
    return rows


def _request_logical_kv_digests(group: Any, layer_indices: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    for request_index, request in enumerate(group.requests):
        layer_rows = {}
        for index in layer_indices:
            sequence = request.layers[index].sequence
            arena = sequence.arena
            digest = hashlib.sha256()
            logical_blocks = math.ceil(sequence.sequence_length / arena.page_size)
            for batch_index in range(arena.batch_size):
                for logical in range(logical_blocks):
                    valid = min(
                        arena.page_size,
                        sequence.sequence_length - logical * arena.page_size,
                    )
                    physical = int(sequence.active_block_table[batch_index, logical])
                    digest.update(
                        arena.key_cache[physical, :valid]
                        .detach()
                        .contiguous()
                        .cpu()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
                    digest.update(
                        arena.value_cache[physical, :valid]
                        .detach()
                        .contiguous()
                        .cpu()
                        .view(torch.uint8)
                        .numpy()
                        .tobytes()
                    )
            layer_rows[str(index)] = digest.hexdigest()
        rows.append({"request_index": request_index, "layer_sha256": layer_rows})
    return rows


def _linear_state_digest(cache: Any, layer_indices: Sequence[int]) -> dict[str, Any]:
    digest = hashlib.sha256()
    storage_keys = []
    tensor_count = 0
    for index in layer_indices:
        layer = cache.layers[index]
        for family in ("conv_states", "recurrent_states"):
            values = getattr(layer, family)
            _require(isinstance(values, dict), "linear state family is not a dict")
            for state_index in sorted(values):
                tensor = values[state_index]
                _require(isinstance(tensor, torch.Tensor), "linear state is not tensor")
                raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
                digest.update(f"{index}:{family}:{state_index}\0".encode())
                digest.update(raw)
                storage = tensor.untyped_storage()
                storage_keys.append((str(tensor.device), storage.data_ptr(), storage.nbytes()))
                tensor_count += 1
    return {
        "sha256": digest.hexdigest(),
        "tensor_count": tensor_count,
        "storage_keys": storage_keys,
    }


def _resident_linear_states(group: Any, layer_indices: Sequence[int]) -> list[dict[str, Any]]:
    rows = []
    mutable_sets = []
    for request_index, request in enumerate(group.requests):
        state = _linear_state_digest(request, layer_indices)
        mutable = set(tuple(value) for value in state.pop("storage_keys"))
        mutable_sets.append(mutable)
        rows.append({"request_index": request_index, **state})
    for left in range(len(mutable_sets)):
        for right in range(left + 1, len(mutable_sets)):
            _require(not (mutable_sets[left] & mutable_sets[right]), "resident GDN mutable states alias")
    return rows


def _last_logits(model: Any, output: Any) -> torch.Tensor:
    logits = model.lm_head(output.last_hidden_state[:, -1, :])
    _require(logits.ndim == 2, "invalid logits rank")
    return logits


def _round_robin_generate(
    model: Any,
    backbone: Any,
    tokenizer: Any,
    group: Any,
    queries: Sequence[torch.Tensor],
    backends: Sequence[str],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    n = group.resident_count
    _require(len(group.requests) == len(queries) == len(backends) == n, "resident generation cardinality drift")
    currents = list(queries)
    trajectories = [
        {
            "request_index": request_index,
            "query_token_ids_sha256": _sha256_tensor(queries[request_index]),
            "generated_token_ids": [],
            "full_vocab_step_logit_sha256": [],
            "step_seconds": [],
        }
        for request_index in range(n)
    ]
    full_logits_cpu: list[list[torch.Tensor]] = [[] for _ in range(n)]
    schedule = []
    production_allocator_steps = []
    original = backbone.config._attn_implementation
    wall_started = time.perf_counter()
    try:
        for round_index in range(max_new_tokens):
            for request_index in range(n):
                _require(len(group.requests) == n, "resident request list changed during generation")
                backbone.config._attn_implementation = backends[request_index]
                _sync()
                started = time.perf_counter()
                output = backbone(
                    input_ids=currents[request_index],
                    past_key_values=group.requests[request_index],
                    use_cache=True,
                )
                logits = _last_logits(model, output)
                token = int(logits.argmax(-1).item())
                _sync()
                elapsed = time.perf_counter() - started
                allocator = _allocator_snapshot(
                    f"production-step-r{round_index}-q{request_index}-before-exactness"
                )
                production_allocator_steps.append(
                    {
                        "round_index": round_index,
                        "request_index": request_index,
                        **allocator,
                    }
                )
                _require(bool(torch.isfinite(logits).all()), "non-finite logits")
                trajectory = trajectories[request_index]
                trajectory["generated_token_ids"].append(token)
                trajectory["full_vocab_step_logit_sha256"].append(_sha256_tensor(logits))
                full_logits_cpu[request_index].append(
                    logits.detach().contiguous().cpu().clone()
                )
                trajectory["step_seconds"].append(elapsed)
                schedule.append({"round_index": round_index, "request_index": request_index})
                currents[request_index] = torch.tensor(
                    [[token]], dtype=torch.long, device=queries[request_index].device
                )
                # The digest/CPU clone above is a correctness diagnostic.  Do
                # not let any of its transient allocations pollute the next
                # model-step production peak.
                del output, logits
                _sync()
                torch.cuda.reset_peak_memory_stats()
    finally:
        backbone.config._attn_implementation = original
    _sync()
    expected_schedule = [
        {"round_index": round_index, "request_index": request_index}
        for round_index in range(max_new_tokens)
        for request_index in range(n)
    ]
    _require(schedule == expected_schedule, "global round-major schedule drift")
    for row in trajectories:
        row["generated_text"] = tokenizer.decode(
            row["generated_token_ids"], skip_special_tokens=True
        )
    return {
        "scheduler": "single-cuda-stream-sequential-round-major",
        "concurrent_kernel_execution_claimed": False,
        "all_requests_resident_for_entire_schedule": True,
        "rounds": max_new_tokens,
        "resident_count": n,
        "total_model_steps": n * max_new_tokens,
        "wall_seconds": time.perf_counter() - wall_started,
        "schedule": schedule,
        "trajectories": trajectories,
        "production_allocator_before_exactness": {
            "steps": production_allocator_steps,
            "peak_allocated_bytes": max(
                int(row["peak_allocated_bytes"])
                for row in production_allocator_steps
            ),
            "peak_reserved_bytes": max(
                int(row["peak_reserved_bytes"])
                for row in production_allocator_steps
            ),
            "exactness_diagnostics_excluded_from_peak": True,
        },
        "_full_logits_cpu": full_logits_cpu,
    }


def _measure_arm(
    args: argparse.Namespace,
    model: Any,
    backbone: Any,
    tokenizer: Any,
    plan: Any,
    document: torch.Tensor,
    queries: Sequence[torch.Tensor],
    *,
    resident_count: int,
    policy: str,
    kernel: Any,
) -> dict[str, Any]:
    _require(torch.is_inference_mode_enabled(), "multi-fork arm must run under torch.inference_mode")
    baseline = _allocator_snapshot("arm-model-input-baseline")
    torch.cuda.reset_peak_memory_stats()
    prefill_before = _allocator_snapshot("before-common-document-prefill")
    prefill_started = time.perf_counter()
    persistent = _build_document_cache(backbone, document)
    _sync()
    prefill_after = _allocator_snapshot("after-common-document-prefill")
    prefill_seconds = time.perf_counter() - prefill_started
    persistent_gdn_before = _linear_state_digest(persistent, plan.linear_layer_indices)
    persistent_gdn_before.pop("storage_keys")

    torch.cuda.reset_peak_memory_stats()
    pack_before = _allocator_snapshot("before-common-q16-pack")
    pack_started = time.perf_counter()
    conversion = convert_all_qwen35_full_layers_to_vllm_q16(
        persistent,
        plan,
        page_size=args.page_size,
        max_append_tokens=args.pg19_query_tokens + args.max_new_tokens,
        max_request_forks=resident_count,
    )
    _sync()
    pack_after = _allocator_snapshot("after-common-q16-pack")
    pack_seconds = time.perf_counter() - pack_started
    _require(conversion.max_request_forks == resident_count, "conversion max_forks drift")
    source_document_before = _source_document_digests(
        persistent, plan.full_attention_layer_indices
    )

    torch.cuda.reset_peak_memory_stats()
    setup_before = _allocator_snapshot("before-all-resident-request-setup")
    setup_started = time.perf_counter()
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=resident_count,
        policy=policy,
    )
    _set_production_no_mask(group, plan.full_attention_layer_indices)
    ledgers = [
        MultiForkHitLedger(
            plan,
            request,
            request_index=request_index,
            resident_count=resident_count,
            request_policy=policy,
            expected_calls_per_layer=args.max_new_tokens,
            initial_query_tokens=args.pg19_query_tokens,
            kernel=kernel,
        )
        for request_index, request in enumerate(group.requests)
    ]
    _require(all(ledger.kernel is kernel for ledger in ledgers), "resident ledgers changed kernel")
    backends = [register_multifork_backend(ledger) for ledger in ledgers]
    _sync()
    setup_after = _allocator_snapshot("after-all-resident-setup-before-generation")
    setup_seconds = time.perf_counter() - setup_started
    storage_before = resident_storage_breakdown(persistent, group, plan)
    _require(len(group.requests) == resident_count, "requests not all resident at setup snapshot")

    torch.cuda.reset_peak_memory_stats()
    generation_before = _allocator_snapshot("before-generation-after-setup-peak-reset")
    _require(
        generation_before["current_allocated_bytes"]
        == setup_after["current_allocated_bytes"]
        and generation_before["current_reserved_bytes"]
        == setup_after["current_reserved_bytes"],
        "allocator continuity drift between setup and generation",
    )
    try:
        generated = _round_robin_generate(
            model,
            backbone,
            tokenizer,
            group,
            queries,
            backends,
            max_new_tokens=args.max_new_tokens,
        )
    finally:
        for name in backends:
            _unregister_backend(name)
    _sync()
    generation_after = _allocator_snapshot("after-round-robin-generation-all-resident")
    _require(len(group.requests) == resident_count, "requests not all resident at generation snapshot")
    storage_after = resident_storage_breakdown(persistent, group, plan)
    source_document_after = _source_document_digests(
        persistent, plan.full_attention_layer_indices
    )
    _require(source_document_after == source_document_before, "source document K/V mutated")
    persistent_gdn_after = _linear_state_digest(persistent, plan.linear_layer_indices)
    persistent_gdn_after.pop("storage_keys")
    _require(persistent_gdn_after == persistent_gdn_before, "persistent GDN base mutated")
    request_gdn = _resident_linear_states(group, plan.linear_layer_indices)
    logical_kv = _request_logical_kv_digests(group, plan.full_attention_layer_indices)
    intercepts = [ledger.verify_complete() for ledger in ledgers]
    _require(
        sum(int(row["total_calls"]) for row in intercepts)
        == resident_count * args.max_new_tokens * len(FULL_LAYERS),
        "fused call cardinality drift",
    )
    expected_copy = (
        int(storage_before["totals"]["fresh_duplicate_document_allocated_nbytes"])
        if policy == FRESH_CONTROL
        else 0
    )
    _require(
        int(group.audit["physical_document_block_copy_nbytes_including_padding"])
        == expected_copy,
        "physical document copy accounting drift",
    )
    production_peak = generated["production_allocator_before_exactness"]
    combined_peak_allocated = max(
        int(setup_after["peak_allocated_bytes"]),
        int(production_peak["peak_allocated_bytes"]),
    )
    combined_peak_reserved = max(
        int(setup_after["peak_reserved_bytes"]),
        int(production_peak["peak_reserved_bytes"]),
    )
    internal_logits = generated.pop("_full_logits_cpu")
    return {
        "protocol": MULTIFORK_PROTOCOL,
        "policy": policy,
        "resident_count": resident_count,
        "quantization": "Q16",
        "batch_per_request": 1,
        "all_requests_simultaneously_resident": True,
        "same_unified_attention_kernel": True,
        "kernel_identity": _kernel_identity(kernel),
        "baseline_before_common_build": baseline,
        "common_document_prefill": {
            "seconds": prefill_seconds,
            "allocator_before": prefill_before,
            "allocator_after": prefill_after,
            **_allocator_delta(prefill_before, prefill_after),
        },
        "common_q16_pack": {
            "seconds": pack_seconds,
            "allocator_before": pack_before,
            "allocator_after": pack_after,
            **_allocator_delta(pack_before, pack_after),
            "source_document_payload_nbytes": conversion.document_payload_nbytes,
            "source_allocated_pool_nbytes": conversion.allocated_block_pool_nbytes,
            "max_request_forks": conversion.max_request_forks,
        },
        "resident_setup": {
            "seconds": setup_seconds,
            "allocator_before": setup_before,
            "allocator_after": setup_after,
            **_allocator_delta(setup_before, setup_after),
            "all_n_objects_alive_at_snapshot": True,
            "group_audit": group.audit,
        },
        "setup_plus_generation": {
            "allocator_before": setup_before,
            "allocator_after": generation_after,
            "current_allocated_delta_bytes": (
                generation_after["current_allocated_bytes"]
                - setup_before["current_allocated_bytes"]
            ),
            "current_reserved_delta_bytes": (
                generation_after["current_reserved_bytes"]
                - setup_before["current_reserved_bytes"]
            ),
            "peak_allocated_delta_bytes": (
                combined_peak_allocated - setup_before["current_allocated_bytes"]
            ),
            "peak_reserved_delta_bytes": (
                combined_peak_reserved - setup_before["current_reserved_bytes"]
            ),
            "combined_absolute_peak_allocated_bytes": combined_peak_allocated,
            "combined_absolute_peak_reserved_bytes": combined_peak_reserved,
            "all_n_objects_alive_at_snapshot": True,
        },
        "generation_only": {
            "allocator_before": generation_before,
            "allocator_after": generation_after,
            "current_allocated_delta_bytes": (
                generation_after["current_allocated_bytes"]
                - generation_before["current_allocated_bytes"]
            ),
            "current_reserved_delta_bytes": (
                generation_after["current_reserved_bytes"]
                - generation_before["current_reserved_bytes"]
            ),
            "peak_allocated_delta_bytes": (
                production_peak["peak_allocated_bytes"]
                - generation_before["current_allocated_bytes"]
            ),
            "peak_reserved_delta_bytes": (
                production_peak["peak_reserved_bytes"]
                - generation_before["current_reserved_bytes"]
            ),
            "production_absolute_peak_allocated_bytes": production_peak[
                "peak_allocated_bytes"
            ],
            "production_absolute_peak_reserved_bytes": production_peak[
                "peak_reserved_bytes"
            ],
            "exactness_diagnostics_excluded_from_peak": True,
            "setup_to_generation_current_continuity_verified": True,
            "all_n_objects_alive_at_snapshot": True,
        },
        "storage_before_generation": storage_before,
        "storage_after_generation": storage_after,
        "source_document_sha256_before": source_document_before,
        "source_document_sha256_after": source_document_after,
        "source_document_immutable": True,
        "persistent_gdn_before": persistent_gdn_before,
        "persistent_gdn_after": persistent_gdn_after,
        "persistent_gdn_immutable": True,
        "request_gdn_after_generation": request_gdn,
        "request_logical_kv_after_generation": logical_kv,
        "intercepts": intercepts,
        "generation": generated,
        "_full_logits_cpu": internal_logits,
        "claim_boundaries": {
            "single_stream_sequential_round_robin": True,
            "concurrent_kernel_throughput_claimed": False,
            "ttft_speedup_claimed": False,
            "raw_step_timing_is_diagnostic_single_observation_only": True,
            "round_robin_wall_includes_logit_digest_and_cpu_clone": True,
            "nvml_peak_measured": False,
            "downstream_quality_measured": False,
        },
    }


def _compare_arms(
    fresh: dict[str, Any],
    reuse: dict[str, Any],
    query_audit: dict[str, Any],
    runtime_tensor_equal: list[dict[str, Any]],
) -> dict[str, Any]:
    n = int(fresh["resident_count"])
    _require(int(reuse["resident_count"]) == n, "arm resident count differs")
    _require(fresh["kernel_identity"] == reuse["kernel_identity"], "arms used different callable identity")
    fresh_trajectories = fresh["generation"]["trajectories"]
    reuse_trajectories = reuse["generation"]["trajectories"]
    _require(len(fresh_trajectories) == len(reuse_trajectories) == n, "trajectory count drift")
    rows = []
    _require(len(runtime_tensor_equal) == n, "runtime tensor-equality row count drift")
    for request_index, (left, right) in enumerate(zip(fresh_trajectories, reuse_trajectories)):
        expected_query_sha = query_audit["rows"][request_index]["query_token_ids_sha256"]
        _require(
            left["query_token_ids_sha256"]
            == right["query_token_ids_sha256"]
            == expected_query_sha,
            "fresh/reuse query identity drift",
        )
        token_exact = left["generated_token_ids"] == right["generated_token_ids"]
        logit_exact = (
            left["full_vocab_step_logit_sha256"]
            == right["full_vocab_step_logit_sha256"]
        )
        _require(token_exact and logit_exact, "fresh/reuse token or full-logit trajectory diverged")
        tensor_row = runtime_tensor_equal[request_index]
        _require(
            tensor_row.get("request_index") == request_index
            and tensor_row.get("all_steps_torch_equal") is True
            and tensor_row.get("step_torch_equal") == [True] * FORMAL_NEW_TOKENS,
            "fresh/reuse full logits failed runtime torch.equal",
        )
        _require(
            len(left["generated_token_ids"])
            == len(left["full_vocab_step_logit_sha256"])
            == FORMAL_NEW_TOKENS,
            "generation trajectory length drift",
        )
        rows.append(
            {
                "request_index": request_index,
                "query_token_ids_sha256": expected_query_sha,
                "generated_tokens_exact": token_exact,
                "full_vocab_step_logits_exact": logit_exact,
                "full_vocab_step_logits_runtime_torch_equal": True,
            }
        )
    _require(
        fresh["request_gdn_after_generation"] == reuse["request_gdn_after_generation"],
        "fresh/reuse request GDN state differs",
    )
    _require(
        fresh["request_logical_kv_after_generation"]
        == reuse["request_logical_kv_after_generation"],
        "fresh/reuse logical K/V differs",
    )
    return {
        "passed": True,
        "resident_count": n,
        "request_count": n,
        "query_token_ids_pairwise_distinct": query_audit["pairwise_distinct"],
        "all_request_token_trajectories_exact": True,
        "all_request_full_vocab_step_logits_exact": True,
        "all_request_full_vocab_step_logits_runtime_torch_equal": True,
        "all_request_logical_kv_exact": True,
        "all_request_gdn_state_exact": True,
        "same_unified_attention_callable_identity": True,
        "rows": rows,
    }


def _arm_execution_order(count: int) -> tuple[str, str]:
    # Alternating the first arm avoids one systematic warm-cache direction;
    # this remains a single capacity observation, not an ABBA latency study.
    index = FORMAL_EXECUTION_ORDER.index(count)
    return MULTIFORK_POLICIES if index % 2 == 0 else tuple(reversed(MULTIFORK_POLICIES))


def run_resident_shard(args: argparse.Namespace) -> dict[str, Any]:
    records, data_audit = audit_pg19_train_calibration(
        args.pg19_data,
        args.pg19_manifest,
        expected_data_sha256=args.expected_pg19_sha256,
        expected_manifest_sha256=args.expected_pg19_manifest_sha256,
        minimum_books=args.pg19_books,
    )
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    windows, windows_sha = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=args.pg19_books,
        document_tokens=args.pg19_document_tokens,
        query_tokens=args.pg19_query_tokens,
        stride=args.pg19_window_stride,
        candidate_windows_per_book=args.pg19_candidate_windows,
        seed=args.pg19_seed,
    )
    _require(windows_sha == args.expected_pg19_windows_sha256, "PG19 windows SHA mismatch")
    assigned = list(enumerate(windows))[args.rank :: args.world_size]
    _require(len(assigned) == 1, "each rank requires exactly one train-only PG19 window")
    window_index, window = assigned[0]

    torch.cuda.set_device(0)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    _require(tuple(plan.full_attention_layer_indices) == FULL_LAYERS, "full-layer geometry drift")
    document = window.document_ids.unsqueeze(0).cuda()
    base_query = window.query_ids.unsqueeze(0).cuda()
    _require(tuple(document.shape) == (1, args.pg19_document_tokens), "4K document length drift")
    _require(int(document.shape[1]) % args.page_size == 127, "4095 tail stress drift")
    del base_query
    query_bank_cpu, query_audit = build_pg19_train_query_bank(
        records,
        tokenizer,
        window,
        document_tokens=args.pg19_document_tokens,
        query_tokens=args.pg19_query_tokens,
        count=max(args.resident_counts),
        query_stride=args.query_bank_stride,
    )
    expected_query_bank = args.static_audit["frozen_query_banks"][args.rank]
    _require(query_audit == expected_query_bank, "runtime PG19 query bank differs from frozen manifest")
    query_bank = [query.cuda() for query in query_bank_cpu]
    kernel = _resolve_vllm_unified_attention()
    kernel_identity = _kernel_identity(kernel)

    # One max-N warmup closes lazy allocations before freezing the allocator
    # baseline.  It is deliberately not reported as a measured row.
    with torch.inference_mode():
        warmup_queries = query_bank[: max(args.resident_counts)]
        warmup = _measure_arm(
            args,
            model,
            backbone,
            tokenizer,
            plan,
            document,
            warmup_queries,
            resident_count=max(args.resident_counts),
            policy=SHARED_REUSE,
            kernel=kernel,
        )
        warmup.pop("_full_logits_cpu")
        del warmup
        frozen_baseline = _allocator_cleanup("post-max-n-warmup-frozen-baseline")
        rows_by_n: dict[int, dict[str, Any]] = {}
        cleanup_rows = []
        for count in args.execution_order:
            arms: dict[str, dict[str, Any]] = {}
            first_logits: list[list[torch.Tensor]] | None = None
            first_policy: str | None = None
            runtime_tensor_equal: list[dict[str, Any]] | None = None
            for policy in _arm_execution_order(count):
                before = _allocator_cleanup(f"before-N{count}-{policy}")
                for field in ("current_allocated_bytes", "current_reserved_bytes"):
                    _require(before[field] == frozen_baseline[field], f"allocator baseline drift before N={count}")
                arm = _measure_arm(
                    args,
                    model,
                    backbone,
                    tokenizer,
                    plan,
                    document,
                    query_bank[:count],
                    resident_count=count,
                    policy=policy,
                    kernel=kernel,
                )
                logits_cpu = arm.pop("_full_logits_cpu")
                if first_logits is None:
                    first_logits = logits_cpu
                    first_policy = policy
                else:
                    _require(first_policy is not None, "first arm identity missing")
                    _require(len(first_logits) == len(logits_cpu) == count, "runtime logits request count drift")
                    runtime_tensor_equal = []
                    for request_index, (first_steps, second_steps) in enumerate(
                        zip(first_logits, logits_cpu)
                    ):
                        _require(
                            len(first_steps) == len(second_steps) == args.max_new_tokens,
                            "runtime logits step count drift",
                        )
                        step_equal = [
                            bool(torch.equal(left, right))
                            for left, right in zip(first_steps, second_steps)
                        ]
                        _require(all(step_equal), "fresh/reuse runtime full logits diverged")
                        runtime_tensor_equal.append(
                            {
                                "request_index": request_index,
                                "first_policy": first_policy,
                                "second_policy": policy,
                                "step_torch_equal": step_equal,
                                "all_steps_torch_equal": True,
                            }
                        )
                    del first_logits, logits_cpu
                arm["allocator_frozen_baseline"] = before
                arms[policy] = arm
                del arm
                after = _allocator_cleanup(f"after-N{count}-{policy}")
                for field in ("current_allocated_bytes", "current_reserved_bytes"):
                    _require(after[field] == frozen_baseline[field], f"allocator did not recover after N={count}")
                cleanup_rows.append({"resident_count": count, "policy": policy, "after": after})
            _require(runtime_tensor_equal is not None, "runtime tensor equality was not evaluated")
            parity = _compare_arms(
                arms[FRESH_CONTROL],
                arms[SHARED_REUSE],
                query_audit,
                runtime_tensor_equal,
            )
            rows_by_n[count] = {
                "resident_count": count,
                "arm_execution_order": list(_arm_execution_order(count)),
                "query_bank_prefix_sha256": _sha256_bytes(
                    "".join(
                        row["query_token_ids_sha256"]
                        for row in query_audit["rows"][:count]
                    ).encode()
                ),
                "fresh": arms[FRESH_CONTROL],
                "reuse": arms[SHARED_REUSE],
                "parity": parity,
            }
    rows = [rows_by_n[count] for count in MULTIFORK_COUNTS]
    return {
        "status": "completed_multifork_resident_pg19_shard",
        "passed": True,
        "protocol": MULTIFORK_PROTOCOL,
        "rank": args.rank,
        "world_size": args.world_size,
        "window_index": window_index,
        "source_object": window.source_object,
        "document_tokens": int(document.shape[1]),
        "document_tail_tokens": int(document.shape[1]) % args.page_size,
        "query_tokens": args.pg19_query_tokens,
        "generated_tokens": args.max_new_tokens,
        "kernel_mode": KERNEL_MODE,
        "kernel_identity": kernel_identity,
        "quantization": "Q16",
        "resident_counts": list(MULTIFORK_COUNTS),
        "execution_order": list(args.execution_order),
        "query_bank": query_audit,
        "data_audit": data_audit,
        "windows_sha256": windows_sha,
        "rows": rows,
        "allocator_fresh_state": {
            "max_n_warmup_completed": True,
            "frozen_baseline": frozen_baseline,
            "cleanup_after_each_arm": cleanup_rows,
            "exact_recovery_fields": ["current_allocated_bytes", "current_reserved_bytes"],
        },
        "pg19_train_only": True,
        "longbench_consumed": False,
        "source_6_9_consumed": False,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
        "claim_boundaries": {
            "approximately_4k_non_aligned_document": True,
            "aligned_4096_measured": False,
            "single_stream_round_robin": True,
            "real_parallel_kernel_execution": False,
            "vllm_engine_scheduler_tested": False,
            "multi_document_tested": False,
            "downstream_quality_measured": False,
            "ttft_or_throughput_speedup_claimed": False,
            "nvml_peak_measured": False,
        },
    }


def _require_nonnegative_int(value: Any, label: str) -> int:
    _require(type(value) is int and value >= 0, f"{label} must be a present non-bool integer >= 0")
    return value


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be one lowercase SHA256",
    )
    return value


def _require_finite_nonnegative_number(value: Any, label: str) -> float:
    _require(
        type(value) in (int, float) and math.isfinite(value) and value >= 0,
        f"{label} must be one finite nonnegative number",
    )
    return float(value)


def _validate_allocator_snapshot(value: Any, label: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{label} allocator snapshot missing")
    _require(isinstance(value.get("label"), str) and value["label"], f"{label} label missing")
    for field in (
        "current_allocated_bytes",
        "current_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    ):
        _require_nonnegative_int(value.get(field), f"{label}.{field}")
    _require(value["peak_allocated_bytes"] >= value["current_allocated_bytes"], f"{label} allocated peak below current")
    _require(value["peak_reserved_bytes"] >= value["current_reserved_bytes"], f"{label} reserved peak below current")
    _require(value["current_reserved_bytes"] >= value["current_allocated_bytes"], f"{label} current reserved below allocated")
    _require(value["peak_reserved_bytes"] >= value["peak_allocated_bytes"], f"{label} reserved peak below allocated peak")
    return value


def _validate_standard_allocator_phase(value: Any, label: str) -> None:
    _require(isinstance(value, dict), f"{label} phase missing")
    _require_finite_nonnegative_number(value.get("seconds"), f"{label}.seconds")
    before = _validate_allocator_snapshot(value.get("allocator_before"), f"{label}.before")
    after = _validate_allocator_snapshot(value.get("allocator_after"), f"{label}.after")
    _require(
        before["peak_allocated_bytes"] == before["current_allocated_bytes"]
        and before["peak_reserved_bytes"] == before["current_reserved_bytes"],
        f"{label}.before does not reflect a peak reset",
    )
    expected = _allocator_delta(before, after)
    for field, result in expected.items():
        _require(value.get(field) == result, f"{label}.{field} differs from snapshots")


def _validate_group_audit(value: Any, *, count: int, policy: str) -> None:
    _require(isinstance(value, dict), "resident group audit missing")
    _require(value.get("protocol") == MULTIFORK_PROTOCOL, "resident group protocol drift")
    _require(
        value.get("policy") == policy
        and type(value.get("resident_count")) is int
        and value.get("resident_count") == count,
        "resident group identity drift",
    )
    _require(value.get("all_requests_materialized_before_measurement") is True, "requests were not all materialized")
    _require(value.get("strong_reference_count") == count, "resident strong-reference count drift")
    rows = value.get("rows")
    _require(isinstance(rows, list) and [row.get("request_index") for row in rows] == list(range(count)), "resident request rows drift")
    for row in rows:
        expected_copy_row = 83886080 if policy == FRESH_CONTROL else 0
        expected_pool_row = 89128960 if policy == FRESH_CONTROL else 0
        _require(row.get("document_block_copy_nbytes_including_padding") == expected_copy_row, "per-request document copy drift")
        _require(row.get("allocated_request_pool_nbytes") == expected_pool_row, "per-request pool allocation drift")
        _require(
            row.get("source_document_storage_shared")
            is (policy == SHARED_REUSE),
            "per-request source-sharing flag drift",
        )
    ownership = value.get("ownership")
    _require(isinstance(ownership, dict) and ownership.get("passed") is True, "resident ownership failed")
    _require(type(ownership.get("resident_count")) is int and ownership.get("resident_count") == count, "ownership N drift")
    for field in (
        "request_object_ids_pairwise_distinct",
        "request_sequence_ids_pairwise_distinct",
        "all_requests_strongly_referenced",
    ):
        _require(ownership.get(field) is True, f"ownership field failed: {field}")
    if policy == FRESH_CONTROL:
        _require(ownership.get("fresh_request_arena_storages_pairwise_disjoint") is True, "fresh arena storage aliases")
        _require(ownership.get("fresh_private_id_namespace_is_per_arena") is True, "fresh ID namespace drift")
        _require(ownership.get("reuse_requests_share_source_arena") is False, "fresh group claims source sharing")
        _require(ownership.get("private_physical_reservation_ids_pairwise_disjoint") is False, "fresh group made global numeric-ID claim")
    else:
        _require(ownership.get("reuse_requests_share_source_arena") is True, "reuse group did not share source")
        _require(ownership.get("private_physical_reservation_ids_pairwise_disjoint") is True, "reuse private IDs overlap")
        _require(ownership.get("fresh_request_arena_storages_pairwise_disjoint") is False, "reuse group claims fresh arenas")
    copied = sum(
        _require_nonnegative_int(
            row.get("document_block_copy_nbytes_including_padding"),
            "resident row document copy",
        )
        for row in rows
    )
    allocated = sum(
        _require_nonnegative_int(
            row.get("allocated_request_pool_nbytes"),
            "resident row allocated pool",
        )
        for row in rows
    )
    _require(value.get("physical_document_block_copy_nbytes_including_padding") == copied, "group physical copy total drift")
    _require(value.get("allocated_fresh_request_pool_nbytes") == allocated, "group fresh pool total drift")
    expected_copy = count * 83886080 if policy == FRESH_CONTROL else 0
    expected_pool = count * 89128960 if policy == FRESH_CONTROL else 0
    _require(copied == expected_copy and allocated == expected_pool, "group 4095 pool/copy formula drift")


def _validate_storage(
    storage: dict[str, Any], *, count: int, policy: str, phase: str
) -> None:
    _require(phase in ("before", "after"), "unknown storage phase")
    _require(type(storage.get("resident_count")) is int and storage.get("resident_count") == count, "storage N drift")
    _require(storage.get("policy") == policy, "storage policy drift")
    layers = storage.get("layers")
    totals = storage.get("totals")
    _require(isinstance(layers, list) and len(layers) == 10, "storage layer count drift")
    _require([row.get("layer_idx") for row in layers] == list(FULL_LAYERS), "storage layer indices drift")
    _require(isinstance(totals, dict), "storage totals missing")
    _require(storage.get("simultaneous_lifetime") is True, "storage lost simultaneous lifetime")
    _require(storage.get("source_private_reservation_is_common_pack_capacity") is True, "source private capacity phase drift")
    _require(storage.get("active_private_payload_is_subset_not_additive") is True, "active payload double-count boundary drift")
    keys = sorted(
        key
        for key in layers[0]
        if key.endswith("_nbytes")
        or key in (
            "active_request_private_blocks",
            "active_request_appended_tokens_sum",
            "active_request_detached_tail_tokens_sum",
            "physical_document_block_copy_nbytes_including_padding",
        )
    )
    replay = {key: 0 for key in keys}
    for row in layers:
        _require(
            type(row.get("resident_count")) is int
            and row.get("resident_count") == count,
            "storage layer N drift",
        )
        for key in keys:
            replay[key] += _require_nonnegative_int(row.get(key), f"storage {key}")
        document = row["source_document_allocated_nbytes"]
        payload = row["valid_document_payload_nbytes"]
        padding = row["source_document_padding_nbytes"]
        source_private = row["source_private_reservation_nbytes"]
        _require(row["block_bytes"] == 262144, "Q16 block geometry drift")
        _require(document == 8388608, "4095 document allocation per-layer drift")
        _require(payload == 8386560, "4095 valid document payload per-layer drift")
        _require(padding == 2048, "4095 document padding per-layer drift")
        _require(source_private == count * 524288, "two-page private reservation per request drift")
        _require(document - payload == padding, "source padding formula drift")
        _require(row["source_total_arena_allocated_nbytes"] == document + source_private, "source total formula drift")
        expected_fresh_document = count * document if policy == FRESH_CONTROL else 0
        expected_fresh_padding = count * padding if policy == FRESH_CONTROL else 0
        _require(row["fresh_duplicate_document_allocated_nbytes"] == expected_fresh_document, "fresh document multiplier drift")
        _require(row["fresh_duplicate_document_padding_nbytes"] == expected_fresh_padding, "fresh padding multiplier drift")
        expected_fresh_private = source_private if policy == FRESH_CONTROL else 0
        _require(row["fresh_duplicate_private_reservation_nbytes"] == expected_fresh_private, "fresh private multiplier drift")
        _require(row["physical_document_block_copy_nbytes_including_padding"] == expected_fresh_document, "physical copy drift")
        _require(
            row["active_request_private_payload_nbytes"]
            <= row["active_request_private_allocated_page_nbytes"]
            <= source_private,
            "active private pages exceed reservation",
        )
        _require(
            row["request_private_reserved_unused_nbytes"]
            == source_private - row["active_request_private_allocated_page_nbytes"],
            "reserved-unused formula drift",
        )
        _require(row["source_document_table_accelerator_nbytes"] == 128, "source document table bytes drift")
        _require(row["source_cpu_reservation_metadata_nbytes"] == count * 16, "source CPU reservation bytes drift")
        _require(row["request_block_table_accelerator_nbytes"] == count * 136, "request block table bytes drift")
        _require(
            row["fresh_document_table_accelerator_nbytes"]
            == (count * 128 if policy == FRESH_CONTROL else 0),
            "fresh document table bytes drift",
        )
        _require(
            row["fresh_cpu_reservation_metadata_nbytes"]
            == (count * 16 if policy == FRESH_CONTROL else 0),
            "fresh CPU reservation bytes drift",
        )
        if phase == "before":
            _require(row["active_request_appended_tokens_sum"] == 0, "before phase has appended tokens")
            _require(row["active_request_detached_tail_tokens_sum"] == 0, "before phase detached a tail")
            _require(row["active_request_private_payload_nbytes"] == 0, "before phase has active payload")
            _require(row["active_request_private_allocated_page_nbytes"] == 0, "before phase has active pages")
            _require(row["active_request_private_blocks"] == 0, "before phase has active blocks")
            _require(row["partial_tail_staging_copy_nbytes"] == 0, "before phase copied a tail")
            _require(row["request_private_reserved_unused_nbytes"] == source_private, "before unused reservation drift")
        else:
            _require(row["active_request_appended_tokens_sum"] == count * 39, "after appended-token sum drift")
            _require(row["active_request_detached_tail_tokens_sum"] == count * 127, "after detached-tail sum drift")
            _require(row["active_request_private_payload_nbytes"] == count * 339968, "after active payload drift")
            _require(row["active_request_private_allocated_page_nbytes"] == count * 524288, "after active page bytes drift")
            _require(row["active_request_private_blocks"] == count * 2, "after active block count drift")
            _require(row["partial_tail_staging_copy_nbytes"] == count * 260096, "after tail COW bytes drift")
            _require(row["request_private_reserved_unused_nbytes"] == 0, "after unused reservation drift")
    _require({key: totals.get(key) for key in keys} == replay, "storage totals differ from layers")
    unique = storage.get("unique_storage")
    _require(isinstance(unique, dict), "unique storage diagnostic missing")
    for field in (
        "persistent_total_nbytes",
        "persistent_accelerator_nbytes",
        "requests_total_nbytes",
        "requests_accelerator_nbytes",
        "combined_unique_total_nbytes",
        "combined_unique_accelerator_nbytes",
    ):
        _require_nonnegative_int(unique.get(field), f"unique storage {field}")
    _require(unique["persistent_total_nbytes"] >= unique["persistent_accelerator_nbytes"], "persistent total below accelerator bytes")
    _require(unique["requests_total_nbytes"] >= unique["requests_accelerator_nbytes"], "request total below accelerator bytes")
    _require(unique["combined_unique_total_nbytes"] >= unique["combined_unique_accelerator_nbytes"], "combined total below accelerator bytes")
    _require(
        unique["combined_unique_total_nbytes"]
        >= max(unique["persistent_total_nbytes"], unique["requests_total_nbytes"]),
        "combined unique total is smaller than a component",
    )
    _require(
        unique["combined_unique_accelerator_nbytes"]
        >= max(unique["persistent_accelerator_nbytes"], unique["requests_accelerator_nbytes"]),
        "combined unique accelerator bytes are smaller than a component",
    )
    _require(unique["combined_unique_total_nbytes"] <= unique["persistent_total_nbytes"] + unique["requests_total_nbytes"], "unique total exceeds sum")
    _require(unique["combined_unique_accelerator_nbytes"] <= unique["persistent_accelerator_nbytes"] + unique["requests_accelerator_nbytes"], "unique accelerator exceeds sum")
    analytic_pool = (
        totals["source_total_arena_allocated_nbytes"]
        + totals["fresh_duplicate_document_allocated_nbytes"]
        + totals["fresh_duplicate_private_reservation_nbytes"]
    )
    _require(unique["combined_unique_accelerator_nbytes"] >= analytic_pool, "unique accelerator is smaller than analytic Q16 pools")


def _validate_shard_allocator(shard: dict[str, Any]) -> None:
    ledger = shard.get("allocator_fresh_state")
    _require(isinstance(ledger, dict) and ledger.get("max_n_warmup_completed") is True, "allocator warmup ledger missing")
    frozen = _validate_allocator_snapshot(ledger.get("frozen_baseline"), "frozen allocator baseline")
    cleanup = ledger.get("cleanup_after_each_arm")
    expected_order = [
        (count, policy)
        for count in FORMAL_EXECUTION_ORDER
        for policy in _arm_execution_order(count)
    ]
    _require(isinstance(cleanup, list) and len(cleanup) == len(expected_order), "allocator cleanup row count drift")
    rows_by_n = {int(row["resident_count"]): row for row in shard["rows"]}
    for cleanup_row, (count, policy) in zip(cleanup, expected_order):
        _require(cleanup_row.get("resident_count") == count and cleanup_row.get("policy") == policy, "allocator cleanup order drift")
        after = _validate_allocator_snapshot(cleanup_row.get("after"), "allocator cleanup after arm")
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(after[field] == frozen[field], f"allocator cleanup did not recover {field}")
        arm = rows_by_n[count]["fresh" if policy == FRESH_CONTROL else "reuse"]
        injected = _validate_allocator_snapshot(arm.get("allocator_frozen_baseline"), "arm frozen baseline")
        entry = _validate_allocator_snapshot(arm.get("baseline_before_common_build"), "arm entry baseline")
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(injected[field] == entry[field] == frozen[field], f"arm baseline drift for {field}")


def _validate_arm(
    arm: dict[str, Any],
    *,
    count: int,
    policy: str,
    query_rows: Sequence[dict[str, Any]],
) -> None:
    _require(
        type(arm.get("resident_count")) is int
        and arm.get("resident_count") == count
        and arm.get("policy") == policy,
        "arm identity drift",
    )
    _require(arm.get("protocol") == MULTIFORK_PROTOCOL, "arm protocol drift")
    _require(arm.get("quantization") == "Q16" and arm.get("batch_per_request") == 1, "arm geometry drift")
    _require(arm.get("all_requests_simultaneously_resident") is True, "arm did not keep N requests resident")
    _require(arm.get("same_unified_attention_kernel") is True, "arm kernel contract drift")
    _validate_standard_allocator_phase(arm.get("common_document_prefill"), "common prefill")
    _validate_standard_allocator_phase(arm.get("common_q16_pack"), "common Q16 pack")
    arm_entry = _validate_allocator_snapshot(
        arm.get("baseline_before_common_build"), "arm baseline before common build"
    )
    prefill = arm["common_document_prefill"]
    pack = arm["common_q16_pack"]
    for field in ("current_allocated_bytes", "current_reserved_bytes"):
        _require(arm_entry[field] == prefill["allocator_before"][field], f"arm/prefill continuity drift for {field}")
        _require(prefill["allocator_after"][field] == pack["allocator_before"][field], f"prefill/pack continuity drift for {field}")
    _require(pack.get("max_request_forks") == count, "source max_forks drift")
    _require(pack.get("source_document_payload_nbytes") == 83865600, "source valid payload total drift")
    _require(pack.get("source_allocated_pool_nbytes") == 83886080 + count * 5242880, "source pool formula drift")
    setup = arm.get("resident_setup")
    _validate_standard_allocator_phase(setup, "resident setup")
    for field in ("current_allocated_bytes", "current_reserved_bytes"):
        _require(pack["allocator_after"][field] == setup["allocator_before"][field], f"pack/setup continuity drift for {field}")
    _require(setup.get("all_n_objects_alive_at_snapshot") is True, "setup lost requests")
    _validate_group_audit(setup.get("group_audit"), count=count, policy=policy)

    generation = arm.get("generation")
    _require(isinstance(generation, dict), "generation artifact missing")
    generation_wall = _require_finite_nonnegative_number(
        generation.get("wall_seconds"), "generation.wall_seconds"
    )
    _require(
        generation.get("rounds") == FORMAL_NEW_TOKENS
        and type(generation.get("resident_count")) is int
        and generation.get("resident_count") == count,
        "generation geometry drift",
    )
    generation_only = arm.get("generation_only")
    _require(isinstance(generation_only, dict), "generation-only allocator phase missing")
    generation_before = _validate_allocator_snapshot(
        generation_only.get("allocator_before"), "generation-only.before"
    )
    generation_after = _validate_allocator_snapshot(
        generation_only.get("allocator_after"), "generation-only.after"
    )
    _require(generation_only.get("all_n_objects_alive_at_snapshot") is True, "generation snapshot lost requests")
    _require(
        generation_before["peak_allocated_bytes"] == generation_before["current_allocated_bytes"]
        and generation_before["peak_reserved_bytes"] == generation_before["current_reserved_bytes"],
        "generation-only.before does not reflect a peak reset",
    )
    _require(generation_only.get("setup_to_generation_current_continuity_verified") is True, "setup/generation continuity flag drift")
    _require(
        generation_before["current_allocated_bytes"]
        == setup["allocator_after"]["current_allocated_bytes"]
        and generation_before["current_reserved_bytes"]
        == setup["allocator_after"]["current_reserved_bytes"],
        "setup/generation current allocator continuity drift",
    )
    production = generation.get("production_allocator_before_exactness")
    _require(isinstance(production, dict) and production.get("exactness_diagnostics_excluded_from_peak") is True, "production allocator ledger missing")
    production_steps = production.get("steps")
    expected_schedule = [
        {"round_index": round_index, "request_index": request_index}
        for round_index in range(FORMAL_NEW_TOKENS)
        for request_index in range(count)
    ]
    _require(isinstance(production_steps, list) and len(production_steps) == len(expected_schedule), "production allocator step count drift")
    for expected, observed in zip(expected_schedule, production_steps):
        _require(observed.get("round_index") == expected["round_index"] and observed.get("request_index") == expected["request_index"], "production allocator schedule drift")
        _validate_allocator_snapshot(observed, "production step")
    production_peak_allocated = max(row["peak_allocated_bytes"] for row in production_steps)
    production_peak_reserved = max(row["peak_reserved_bytes"] for row in production_steps)
    _require(production.get("peak_allocated_bytes") == production_peak_allocated, "production allocated peak replay drift")
    _require(production.get("peak_reserved_bytes") == production_peak_reserved, "production reserved peak replay drift")
    expected_generation = {
        "current_allocated_delta_bytes": generation_after["current_allocated_bytes"] - generation_before["current_allocated_bytes"],
        "current_reserved_delta_bytes": generation_after["current_reserved_bytes"] - generation_before["current_reserved_bytes"],
        "peak_allocated_delta_bytes": production_peak_allocated - generation_before["current_allocated_bytes"],
        "peak_reserved_delta_bytes": production_peak_reserved - generation_before["current_reserved_bytes"],
    }
    for field, value in expected_generation.items():
        _require(generation_only.get(field) == value, f"generation-only {field} replay drift")
        _require_nonnegative_int(value, f"generation-only {field}")
    _require(generation_only.get("production_absolute_peak_allocated_bytes") == production_peak_allocated, "generation absolute allocated peak drift")
    _require(generation_only.get("production_absolute_peak_reserved_bytes") == production_peak_reserved, "generation absolute reserved peak drift")

    combined = arm.get("setup_plus_generation")
    _require(isinstance(combined, dict) and combined.get("all_n_objects_alive_at_snapshot") is True, "combined allocator phase missing")
    combined_before = _validate_allocator_snapshot(combined.get("allocator_before"), "combined.before")
    combined_after = _validate_allocator_snapshot(combined.get("allocator_after"), "combined.after")
    _require(combined_before == setup["allocator_before"] and combined_after == generation_after, "combined allocator endpoints drift")
    combined_peak_allocated = max(setup["allocator_after"]["peak_allocated_bytes"], production_peak_allocated)
    combined_peak_reserved = max(setup["allocator_after"]["peak_reserved_bytes"], production_peak_reserved)
    expected_combined = {
        "current_allocated_delta_bytes": combined_after["current_allocated_bytes"] - combined_before["current_allocated_bytes"],
        "current_reserved_delta_bytes": combined_after["current_reserved_bytes"] - combined_before["current_reserved_bytes"],
        "peak_allocated_delta_bytes": combined_peak_allocated - combined_before["current_allocated_bytes"],
        "peak_reserved_delta_bytes": combined_peak_reserved - combined_before["current_reserved_bytes"],
    }
    for field, value in expected_combined.items():
        _require(combined.get(field) == value, f"combined {field} replay drift")
        _require_nonnegative_int(value, f"combined {field}")
    _require(combined.get("combined_absolute_peak_allocated_bytes") == combined_peak_allocated, "combined allocated peak drift")
    _require(combined.get("combined_absolute_peak_reserved_bytes") == combined_peak_reserved, "combined reserved peak drift")

    _validate_storage(
        arm["storage_before_generation"], count=count, policy=policy, phase="before"
    )
    _validate_storage(
        arm["storage_after_generation"], count=count, policy=policy, phase="after"
    )
    source_before = arm.get("source_document_sha256_before")
    source_after = arm.get("source_document_sha256_after")
    _require(
        isinstance(source_before, dict)
        and set(source_before) == {str(index) for index in FULL_LAYERS}
        and all(_require_sha256(value, "source document digest") for value in source_before.values()),
        "source document digest schema drift",
    )
    _require(source_after == source_before, "source document mutation")
    _require(arm.get("source_document_immutable") is True, "source immutable flag drift")
    persistent_before = arm.get("persistent_gdn_before")
    persistent_after = arm.get("persistent_gdn_after")
    _require(isinstance(persistent_before, dict), "persistent GDN digest missing")
    _require_sha256(persistent_before.get("sha256"), "persistent GDN")
    _require(persistent_before.get("tensor_count") == 60, "persistent GDN tensor count drift")
    _require(persistent_after == persistent_before, "persistent GDN mutation")
    _require(arm.get("persistent_gdn_immutable") is True, "persistent GDN immutable flag drift")
    request_gdn = arm.get("request_gdn_after_generation")
    _require(isinstance(request_gdn, list) and len(request_gdn) == count, "request GDN rows drift")
    for request_index, row in enumerate(request_gdn):
        _require(row.get("request_index") == request_index, "request GDN index drift")
        _require_sha256(row.get("sha256"), "request GDN")
        _require(row.get("tensor_count") == 60, "request GDN tensor count drift")
    logical_kv = arm.get("request_logical_kv_after_generation")
    _require(isinstance(logical_kv, list) and len(logical_kv) == count, "logical K/V rows drift")
    for request_index, row in enumerate(logical_kv):
        _require(row.get("request_index") == request_index, "logical K/V request index drift")
        layer_sha = row.get("layer_sha256")
        _require(isinstance(layer_sha, dict) and set(layer_sha) == {str(index) for index in FULL_LAYERS}, "logical K/V layer set drift")
        for value in layer_sha.values():
            _require_sha256(value, "logical K/V")

    intercepts = arm.get("intercepts")
    _require(isinstance(intercepts, list) and len(intercepts) == count, "intercept count drift")
    identity = _validate_kernel_identity(arm.get("kernel_identity"), "arm")
    for request_index, intercept in enumerate(intercepts):
        _require(intercept.get("verified") is True, "intercept verified flag drift")
        _require(intercept.get("request_index") == request_index, "intercept request order drift")
        _require(type(intercept.get("resident_count")) is int and intercept.get("resident_count") == count, "intercept N drift")
        _require(intercept.get("request_policy") == policy, "intercept policy drift")
        _require(intercept.get("protocol") == MULTIFORK_PROTOCOL, "intercept protocol drift")
        _require(intercept.get("kernel_identity") == identity, "intercept kernel identity drift")
        _require(intercept.get("same_unified_attention_kernel") is True, "intercept same-kernel flag drift")
        _require(intercept.get("kernel_mode") == KERNEL_MODE, "intercept kernel mode drift")
        _require(intercept.get("initial_query_tokens") == FORMAL_QUERY_TOKENS, "intercept initial-query drift")
        _require(
            intercept.get("counts")
            == {str(index): FORMAL_NEW_TOKENS for index in FULL_LAYERS},
            "intercept per-layer counts drift",
        )
        _require(
            intercept.get("round_major_request_local_layer_order_verified") is True,
            "intercept request-local order flag drift",
        )
        _require(intercept.get("total_calls") == FORMAL_NEW_TOKENS * len(FULL_LAYERS), "intercept call count drift")
        _require(intercept.get("dense_fallback_calls") == 0 and intercept.get("full_kv_concatenations") == 0, "intercept fallback/concat drift")
        _require(intercept.get("mask_contract") == "prevalidated-no-padding-tail-causal", "intercept mask contract drift")
        _require(intercept.get("position_ids_contract") == "qwen3.5-text-tail-post-rope-v1", "intercept position contract drift")
        calls = intercept.get("calls")
        _require(isinstance(calls, list) and len(calls) == FORMAL_NEW_TOKENS * len(FULL_LAYERS), "raw call ledger drift")
        expected_layers = list(FULL_LAYERS) * FORMAL_NEW_TOKENS
        _require([row.get("layer_idx") for row in calls] == expected_layers, "raw call layer order drift")
        for call_index, row in enumerate(calls):
            round_index = call_index // len(FULL_LAYERS)
            query_tokens = FORMAL_QUERY_TOKENS if round_index == 0 else 1
            kv_tokens = 4127 + round_index
            expected_pool_blocks = 34 if policy == FRESH_CONTROL else 32 + 2 * count
            _require(
                row.get("request_index") == request_index
                and type(row.get("resident_count")) is int
                and row.get("resident_count") == count,
                "raw call request identity drift",
            )
            _require(row.get("request_policy") == policy and row.get("protocol") == MULTIFORK_PROTOCOL, "raw call policy/protocol drift")
            _require(row.get("kernel_identity") == identity, "raw call kernel identity drift")
            _require(row.get("current_append_delta_tokens") == query_tokens and row.get("query_tokens") == query_tokens, "raw call append/query delta drift")
            _require(row.get("kv_tokens") == kv_tokens, "raw call K/V length drift")
            _require(row.get("physical_block_pool_shape") == [expected_pool_blocks, 128, 2, 256], "raw call physical pool geometry drift")
            _require(row.get("active_block_table_shape") == [1, 33], "raw call block table shape drift")
            _require(row.get("kernel_mode") == KERNEL_MODE and row.get("quantization") == "Q16", "raw call kernel/quantization drift")
            _require(row.get("fused_gpu_kernel_calls") == 1 and row.get("full_kv_concatenations") == 0, "raw call fused/concat count drift")
            _require(row.get("full_document_staging_copy_nbytes") == 0, "raw call staged a full document")
            _require(row.get("partial_tail_staging_copy_nbytes") == 260096, "raw call tail COW counter drift")
            _require(row.get("gqa_groups") == 8, "raw call GQA drift")
            _require(row.get("mask_contract") == "prevalidated-no-padding-tail-causal", "raw call mask contract drift")
            _require(row.get("materialized_attention_mask_nbytes") == 0 and row.get("mask_validation_host_syncs") == 0, "raw call materialized/synchronized mask")
            _require(row.get("position_ids_contract") == "qwen3.5-text-tail-post-rope-v1", "raw call position contract drift")
            _require(row.get("position_ids_validated") is True and row.get("position_ids_semantically_consumed_upstream") is True, "raw call position validation drift")
            _require(row.get("position_ids_shape") == [1, query_tokens], "raw call position shape drift")
            _require(row.get("position_ids_dtype") == "torch.int64", "raw call position dtype drift")
            _require(row.get("position_ids_expected_tail_start") == kv_tokens - query_tokens, "raw call position start drift")
            _require(row.get("position_ids_expected_tail_end_exclusive") == kv_tokens, "raw call position end drift")
            _require(row.get("position_ids_strict_tail_values_checked") is False and row.get("position_ids_validation_host_syncs") == 0, "raw call timed position sync drift")

    _require(generation.get("schedule") == expected_schedule, "round-major schedule drift")
    _require(generation.get("all_requests_resident_for_entire_schedule") is True, "generation lost requests")
    _require(generation.get("scheduler") == "single-cuda-stream-sequential-round-major", "scheduler label drift")
    _require(generation.get("concurrent_kernel_execution_claimed") is False, "concurrent execution overclaim")
    _require(generation.get("total_model_steps") == count * FORMAL_NEW_TOKENS, "model-step count drift")
    _require(
        arm.get("claim_boundaries")
        == {
            "single_stream_sequential_round_robin": True,
            "concurrent_kernel_throughput_claimed": False,
            "ttft_speedup_claimed": False,
            "raw_step_timing_is_diagnostic_single_observation_only": True,
            "round_robin_wall_includes_logit_digest_and_cpu_clone": True,
            "nvml_peak_measured": False,
            "downstream_quality_measured": False,
        },
        "arm claim-boundary drift",
    )
    trajectories = generation.get("trajectories")
    _require(isinstance(trajectories, list) and len(trajectories) == count, "trajectory row count drift")
    _require(len(query_rows) >= count, "query rows shorter than N")
    for request_index, row in enumerate(trajectories):
        _require(row.get("request_index") == request_index, "trajectory request index drift")
        _require(row.get("query_token_ids_sha256") == query_rows[request_index].get("query_token_ids_sha256"), "trajectory query digest drift")
        tokens = row.get("generated_token_ids")
        logits = row.get("full_vocab_step_logit_sha256")
        step_seconds = row.get("step_seconds")
        _require(isinstance(tokens, list) and len(tokens) == FORMAL_NEW_TOKENS, "token trajectory length drift")
        _require(all(type(token) is int and token >= 0 for token in tokens), "token trajectory schema drift")
        _require(isinstance(logits, list) and len(logits) == FORMAL_NEW_TOKENS, "logit trajectory length drift")
        _require(
            isinstance(step_seconds, list) and len(step_seconds) == FORMAL_NEW_TOKENS,
            "step timing trajectory length drift",
        )
        for value in step_seconds:
            _require_finite_nonnegative_number(value, "diagnostic model-step seconds")
        for value in logits:
            _require_sha256(value, "full-vocab logit")
    _require(
        generation_wall
        >= sum(
            float(value)
            for row in trajectories
            for value in row["step_seconds"]
        ),
        "generation diagnostic wall time is shorter than model-step sum",
    )


def _rank_capacity_rows(rows: list[dict[str, Any]], policy: str) -> list[dict[str, int]]:
    result = []
    for row in rows:
        count = int(row["resident_count"])
        arm = row["fresh" if policy == FRESH_CONTROL else "reuse"]
        totals = arm["storage_after_generation"]["totals"]
        source_total = int(totals["source_total_arena_allocated_nbytes"])
        fresh_document = int(totals["fresh_duplicate_document_allocated_nbytes"])
        fresh_private = int(totals["fresh_duplicate_private_reservation_nbytes"])
        result.append(
            {
                "resident_count": count,
                "analytic_full_attention_pool_nbytes": source_total + fresh_document + fresh_private,
                "physical_document_copy_nbytes": int(totals["physical_document_block_copy_nbytes_including_padding"]),
                "source_private_reservation_nbytes": int(totals["source_private_reservation_nbytes"]),
                "partial_tail_staging_copy_nbytes": int(
                    totals["partial_tail_staging_copy_nbytes"]
                ),
            }
        )
    return result


def aggregate_shards(
    paths: Sequence[Path],
    *,
    expected_frozen_identity: dict[str, str],
    expected_query_banks: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    _require(len(paths) == FORMAL_WORLD_SIZE, "aggregate requires eight shard artifacts")
    payloads = [path.read_bytes() for path in paths]
    shards = [json.loads(payload) for payload in payloads]
    shard_artifacts = [
        {
            "name": path.name,
            "sha256": _sha256_bytes(payload),
            "bytes": len(payload),
        }
        for path, payload in zip(paths, payloads)
    ]
    _require(
        all(type(shard.get("rank")) is int for shard in shards)
        and sorted(shard["rank"] for shard in shards) == list(range(FORMAL_WORLD_SIZE)),
        "rank set drift",
    )
    shards.sort(key=lambda row: row["rank"])
    descriptors = set()
    rank_curves = []
    source_objects = []
    for rank, shard in enumerate(shards):
        _require(shard.get("status") == "completed_multifork_resident_pg19_shard", "shard status drift")
        _require(shard.get("passed") is True and shard.get("rank") == rank, "shard rank failed")
        _require(type(shard.get("window_index")) is int and shard.get("window_index") == rank, "rank/window assignment drift")
        _require(
            shard.get("protocol") == MULTIFORK_PROTOCOL
            and shard.get("world_size") == FORMAL_WORLD_SIZE
            and shard.get("query_tokens") == FORMAL_QUERY_TOKENS
            and shard.get("generated_tokens") == FORMAL_NEW_TOKENS
            and shard.get("kernel_mode") == KERNEL_MODE
            and shard.get("quantization") == "Q16",
            "shard formal schema drift",
        )
        _require(shard.get("resident_counts") == list(MULTIFORK_COUNTS), "resident count matrix drift")
        _require(shard.get("execution_order") == list(FORMAL_EXECUTION_ORDER), "execution order drift")
        _require(shard.get("document_tokens") == FORMAL_DOCUMENT_TOKENS, "document length drift")
        _require(shard.get("document_tail_tokens") == 127, "tail stress drift")
        _require(
            shard.get("windows_sha256")
            == expected_frozen_identity["pg19_windows_sha256"],
            "shard PG19 windows SHA drift",
        )
        _require(shard.get("pg19_train_only") is True, "non-PG19 shard")
        for field in ("longbench_consumed", "source_6_9_consumed", "source_68_99_consumed", "test_v2_consumed"):
            _require(shard.get(field) is False, f"forbidden data consumed: {field}")
        _require(
            shard.get("claim_boundaries")
            == {
                "approximately_4k_non_aligned_document": True,
                "aligned_4096_measured": False,
                "single_stream_round_robin": True,
                "real_parallel_kernel_execution": False,
                "vllm_engine_scheduler_tested": False,
                "multi_document_tested": False,
                "downstream_quality_measured": False,
                "ttft_or_throughput_speedup_claimed": False,
                "nvml_peak_measured": False,
            },
            "shard claim-boundary drift",
        )
        static = shard.get("static")
        _require(_static_frozen_identity(static) == expected_frozen_identity, "shard frozen identity drift")
        data_audit = shard.get("data_audit")
        _require(isinstance(data_audit, dict), "PG19 data audit missing")
        _require(
            data_audit.get("data_sha256") == expected_frozen_identity["pg19_data_sha256"]
            and data_audit.get("manifest_sha256") == expected_frozen_identity["pg19_manifest_sha256"],
            "PG19 data audit digest drift",
        )
        _require(
            data_audit.get("bucket") == PG19_BUCKET
            and data_audit.get("prefix") == PG19_PREFIX
            and data_audit.get("data_role") == "pg19_train_development_calibration_only",
            "PG19 train-only provenance drift",
        )
        _require(
            type(data_audit.get("records")) is int
            and data_audit["records"] >= FORMAL_BOOKS,
            "PG19 data audit record count drift",
        )
        _require(
            data_audit.get("longbench_labels_used") is False
            and data_audit.get("formal_validation_source_6_35_used") is False
            and data_audit.get("frozen_test_v2_source_68_99_used") is False,
            "forbidden evaluation provenance in PG19 data audit",
        )
        identity = _validate_kernel_identity(shard.get("kernel_identity"), "shard")
        descriptor = _kernel_descriptor(identity)
        _require(all(descriptor), "kernel descriptor schema drift")
        descriptors.add(descriptor)
        query_bank = shard.get("query_bank")
        _require(
            len(expected_query_banks) == FORMAL_WORLD_SIZE
            and query_bank
            == static.get("frozen_query_banks", [])[rank]
            == expected_query_banks[rank],
            "shard query bank differs from frozen manifest",
        )
        _require(shard.get("source_object") == query_bank.get("source_object"), "shard/query-bank source object drift")
        _require(
            isinstance(shard.get("source_object"), str)
            and shard["source_object"].startswith("train/")
            and shard["source_object"].endswith(".txt"),
            "shard source is not a PG19 train object",
        )
        source_objects.append(shard["source_object"])
        _require(query_bank.get("pairwise_distinct") is True and query_bank.get("count") == 32, "query bank drift")
        query_digests = [row.get("query_token_ids_sha256") for row in query_bank.get("rows", [])]
        _require(len(query_digests) == len(set(query_digests)) == 32, "query bank digest uniqueness drift")
        rows = shard.get("rows")
        _require(isinstance(rows, list) and [row.get("resident_count") for row in rows] == list(MULTIFORK_COUNTS), "capacity rows drift")
        for row in rows:
            _require(type(row.get("resident_count")) is int, "resident count must be a non-bool integer")
            count = row["resident_count"]
            _require(row.get("arm_execution_order") == list(_arm_execution_order(count)), "row arm execution order drift")
            expected_prefix_sha = _sha256_bytes(
                "".join(query_digests[:count]).encode()
            )
            _require(
                row.get("query_bank_prefix_sha256") == expected_prefix_sha,
                "query-bank prefix SHA drift",
            )
            _validate_arm(
                row["fresh"],
                count=count,
                policy=FRESH_CONTROL,
                query_rows=query_bank["rows"],
            )
            _validate_arm(
                row["reuse"],
                count=count,
                policy=SHARED_REUSE,
                query_rows=query_bank["rows"],
            )
            _require(row["fresh"]["kernel_identity"] == row["reuse"]["kernel_identity"] == identity, "same-process kernel identity drift")
            fresh_trajectories = row["fresh"]["generation"]["trajectories"]
            reuse_trajectories = row["reuse"]["generation"]["trajectories"]
            for request_index, (fresh_row, reuse_row) in enumerate(
                zip(fresh_trajectories, reuse_trajectories)
            ):
                _require(
                    fresh_row["query_token_ids_sha256"]
                    == reuse_row["query_token_ids_sha256"]
                    == query_digests[request_index],
                    "raw fresh/reuse query SHA drift",
                )
                _require(
                    fresh_row["generated_token_ids"]
                    == reuse_row["generated_token_ids"],
                    "raw fresh/reuse token trajectory diverged",
                )
                _require(
                    fresh_row["full_vocab_step_logit_sha256"]
                    == reuse_row["full_vocab_step_logit_sha256"],
                    "raw fresh/reuse full-logit SHA trajectory diverged",
                )
            _require(
                row["fresh"]["request_gdn_after_generation"]
                == row["reuse"]["request_gdn_after_generation"],
                "raw fresh/reuse GDN digests diverged",
            )
            _require(
                row["fresh"]["request_logical_kv_after_generation"]
                == row["reuse"]["request_logical_kv_after_generation"],
                "raw fresh/reuse logical K/V digests diverged",
            )
            _require(
                row["fresh"]["source_document_sha256_before"]
                == row["reuse"]["source_document_sha256_before"],
                "raw arm source document digests differ",
            )
            _require(
                row["fresh"]["persistent_gdn_before"]
                == row["reuse"]["persistent_gdn_before"],
                "raw arm persistent GDN bases differ",
            )
            parity = row.get("parity")
            _require(
                parity.get("passed") is True
                and parity.get("request_count") == count
                and parity.get("all_request_token_trajectories_exact") is True
                and parity.get("all_request_full_vocab_step_logits_exact") is True
                and parity.get("all_request_full_vocab_step_logits_runtime_torch_equal") is True
                and parity.get("all_request_logical_kv_exact") is True
                and parity.get("all_request_gdn_state_exact") is True,
                "multi-fork parity gate failed",
            )
            parity_rows = parity.get("rows")
            _require(isinstance(parity_rows, list) and len(parity_rows) == count, "parity row count drift")
            for request_index, parity_row in enumerate(parity_rows):
                _require(
                    parity_row
                    == {
                        "request_index": request_index,
                        "query_token_ids_sha256": query_digests[request_index],
                        "generated_tokens_exact": True,
                        "full_vocab_step_logits_exact": True,
                        "full_vocab_step_logits_runtime_torch_equal": True,
                    },
                    "reported parity row differs from raw replay",
                )
        for policy_key in ("fresh", "reuse"):
            prefix_reference: dict[int, dict[str, Any]] = {}
            for row in rows:
                count = row["resident_count"]
                arm = row[policy_key]
                trajectories = arm["generation"]["trajectories"]
                gdn_rows = arm["request_gdn_after_generation"]
                kv_rows = arm["request_logical_kv_after_generation"]
                for request_index in range(count):
                    replayed = {
                        "query_token_ids_sha256": trajectories[request_index][
                            "query_token_ids_sha256"
                        ],
                        "generated_token_ids": trajectories[request_index][
                            "generated_token_ids"
                        ],
                        "full_vocab_step_logit_sha256": trajectories[request_index][
                            "full_vocab_step_logit_sha256"
                        ],
                        "final_gdn": gdn_rows[request_index],
                        "final_logical_kv": kv_rows[request_index],
                    }
                    if request_index not in prefix_reference:
                        prefix_reference[request_index] = replayed
                    else:
                        _require(
                            replayed == prefix_reference[request_index],
                            f"cross-N prefix isolation drift for {policy_key} request {request_index}",
                        )
            _require(len(prefix_reference) == max(MULTIFORK_COUNTS), "cross-N prefix coverage drift")
        _validate_shard_allocator(shard)
        fresh_curve = _rank_capacity_rows(rows, FRESH_CONTROL)
        reuse_curve = _rank_capacity_rows(rows, SHARED_REUSE)
        saving_curve = [
            {
                "resident_count": fresh_row["resident_count"],
                "controlled_pool_bytes_saved": (
                    fresh_row["analytic_full_attention_pool_nbytes"]
                    - reuse_row["analytic_full_attention_pool_nbytes"]
                ),
            }
            for fresh_row, reuse_row in zip(fresh_curve, reuse_curve)
        ]
        fits = {
            "fresh_full_attention_pool": linear_capacity_fit(fresh_curve, "analytic_full_attention_pool_nbytes"),
            "reuse_full_attention_pool": linear_capacity_fit(reuse_curve, "analytic_full_attention_pool_nbytes"),
            "controlled_pool_bytes_saved": linear_capacity_fit(saving_curve, "controlled_pool_bytes_saved"),
            "fresh_physical_document_copy": linear_capacity_fit(fresh_curve, "physical_document_copy_nbytes"),
            "reuse_physical_document_copy": linear_capacity_fit(reuse_curve, "physical_document_copy_nbytes"),
            "source_private_reservation": linear_capacity_fit(reuse_curve, "source_private_reservation_nbytes"),
            "partial_tail_staging_copy": linear_capacity_fit(reuse_curve, "partial_tail_staging_copy_nbytes"),
        }
        expected_fits = {
            "fresh_full_attention_pool": (94371840.0, 83886080.0),
            "reuse_full_attention_pool": (5242880.0, 83886080.0),
            "controlled_pool_bytes_saved": (89128960.0, 0.0),
            "fresh_physical_document_copy": (83886080.0, 0.0),
            "reuse_physical_document_copy": (0.0, 0.0),
            "source_private_reservation": (5242880.0, 0.0),
            "partial_tail_staging_copy": (2600960.0, 0.0),
        }
        for name, (slope, intercept) in expected_fits.items():
            fit = fits[name]
            _require(
                math.isclose(fit["slope_nbytes_per_request"], slope, rel_tol=0, abs_tol=1e-6)
                and math.isclose(fit["intercept_nbytes"], intercept, rel_tol=0, abs_tol=1e-6)
                and math.isclose(fit["r_squared"], 1.0, rel_tol=0, abs_tol=1e-12),
                f"analytic capacity fit drift: {name}",
            )
        rank_curves.append({"rank": rank, "fresh": fresh_curve, "reuse": reuse_curve, "saving": saving_curve, "fits": fits})
    _require(len(descriptors) == 1, "ranks used different unified_attention descriptors")
    _require(len(set(source_objects)) == FORMAL_WORLD_SIZE, "ranks did not use eight distinct PG19 train books")

    matrix = []
    for row_index, count in enumerate(MULTIFORK_COUNTS):
        item: dict[str, Any] = {"resident_count": count}
        for policy, key in ((FRESH_CONTROL, "fresh"), (SHARED_REUSE, "reuse")):
            arms = [shard["rows"][row_index][key] for shard in shards]
            storage = [arm["storage_after_generation"]["totals"] for arm in arms]
            item[key] = {
                "setup_current_allocated_delta_median_bytes": statistics.median(
                    arm["resident_setup"]["current_allocated_delta_bytes"] for arm in arms
                ),
                "setup_peak_allocated_delta_median_bytes": statistics.median(
                    arm["resident_setup"]["peak_allocated_delta_bytes"] for arm in arms
                ),
                "setup_current_reserved_delta_median_bytes": statistics.median(
                    arm["resident_setup"]["current_reserved_delta_bytes"] for arm in arms
                ),
                "setup_peak_reserved_delta_median_bytes": statistics.median(
                    arm["resident_setup"]["peak_reserved_delta_bytes"] for arm in arms
                ),
                "setup_plus_generation_peak_allocated_delta_median_bytes": statistics.median(
                    arm["setup_plus_generation"]["peak_allocated_delta_bytes"] for arm in arms
                ),
                "setup_plus_generation_peak_reserved_delta_median_bytes": statistics.median(
                    arm["setup_plus_generation"]["peak_reserved_delta_bytes"] for arm in arms
                ),
                "setup_plus_generation_current_allocated_delta_median_bytes": statistics.median(
                    arm["setup_plus_generation"]["current_allocated_delta_bytes"] for arm in arms
                ),
                "setup_plus_generation_current_reserved_delta_median_bytes": statistics.median(
                    arm["setup_plus_generation"]["current_reserved_delta_bytes"] for arm in arms
                ),
                "generation_after_current_allocated_median_bytes": statistics.median(
                    arm["generation_only"]["allocator_after"]["current_allocated_bytes"]
                    for arm in arms
                ),
                "generation_after_current_reserved_median_bytes": statistics.median(
                    arm["generation_only"]["allocator_after"]["current_reserved_bytes"]
                    for arm in arms
                ),
                "production_absolute_peak_allocated_median_bytes": statistics.median(
                    arm["generation_only"]["production_absolute_peak_allocated_bytes"]
                    for arm in arms
                ),
                "production_absolute_peak_reserved_median_bytes": statistics.median(
                    arm["generation_only"]["production_absolute_peak_reserved_bytes"]
                    for arm in arms
                ),
                "source_document_allocated_nbytes": statistics.median(
                    value["source_document_allocated_nbytes"] for value in storage
                ),
                "source_private_reservation_nbytes": statistics.median(
                    value["source_private_reservation_nbytes"] for value in storage
                ),
                "fresh_duplicate_document_allocated_nbytes": statistics.median(
                    value["fresh_duplicate_document_allocated_nbytes"] for value in storage
                ),
                "fresh_duplicate_private_reservation_nbytes": statistics.median(
                    value["fresh_duplicate_private_reservation_nbytes"] for value in storage
                ),
                "active_request_private_payload_nbytes": statistics.median(
                    value["active_request_private_payload_nbytes"] for value in storage
                ),
                "physical_document_copy_nbytes": statistics.median(
                    value["physical_document_block_copy_nbytes_including_padding"] for value in storage
                ),
            }
        matrix.append(item)
    return {
        "status": "completed_multifork_resident_pg19_summary",
        "passed": True,
        "protocol": MULTIFORK_PROTOCOL,
        "world_size": FORMAL_WORLD_SIZE,
        "resident_counts": list(MULTIFORK_COUNTS),
        "rank_count": len(shards),
        "raw_shard_artifacts": shard_artifacts,
        "frozen_identity": expected_frozen_identity,
        "same_kernel_full_logit_token_logical_kv_gdn_exact_fraction": 1.0,
        "cross_n_prefix_isolation_exact": True,
        "kernel_descriptor": list(next(iter(descriptors))),
        "capacity_matrix": matrix,
        "rank_capacity_curves_and_fits": rank_curves,
        "primary_capacity_slopes_use_replayed_analytic_q16_pools_only": True,
        "combined_unique_inventory_is_diagnostic_not_fitted_or_claim_authorizing": True,
        "timing_is_raw_validation_instrumented_single_observation_not_aggregated": True,
        "allocator_deltas_are_relative_to_post_pack_request_setup_baseline": True,
        "allocator_absolute_values_are_pytorch_allocator_not_nvml_or_total_model_capacity": True,
        "pg19_train_only": True,
        "longbench_consumed": False,
        "source_6_9_consumed": False,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
        "claim_boundaries_zh": [
            "4095-token约4K非对齐PG19 train-only文档；未实测aligned 4096。",
            "N个请求同时驻留，但模型步在单CUDA stream上按round-major顺序执行；不是并行吞吐实验。",
            "只覆盖Q16、batch1、单文档、10个full-attention层和N<=32。",
            "未测试vLLM engine scheduler、continuous batching、ragged、多文档、回收复用、NVML或下游质量。",
            "未读取LongBench、source6-9、source68-99或test-v2。",
        ],
    }


def _validate_static(args: argparse.Namespace) -> dict[str, Any]:
    # Q8/Q4 fail before any path is opened, package inspected or GPU queried.
    _require(args.bits == 16, "multi-fork resident fused path supports Q16 only")
    _require(args.world_size == FORMAL_WORLD_SIZE, "formal multi-fork requires eight ranks")
    _require(0 <= args.rank < args.world_size, "rank outside world size")
    _require(tuple(args.resident_counts) == MULTIFORK_COUNTS, "resident count matrix drift")
    _require(tuple(args.execution_order) == FORMAL_EXECUTION_ORDER, "execution order drift")
    _require(args.page_size == FORMAL_PAGE_SIZE, "page size drift")
    _require(args.pg19_books == FORMAL_BOOKS, "PG19 book count drift")
    _require(args.pg19_document_tokens == FORMAL_DOCUMENT_TOKENS, "document token count drift")
    _require(args.pg19_query_tokens == FORMAL_QUERY_TOKENS, "query token count drift")
    _require(args.max_new_tokens == FORMAL_NEW_TOKENS, "generation length drift")
    _require(args.pg19_window_stride == FORMAL_WINDOW_STRIDE, "window stride drift")
    _require(args.pg19_candidate_windows == FORMAL_CANDIDATES, "candidate count drift")
    _require(args.pg19_seed == FORMAL_SEED, "PG19 seed drift")
    _require(args.query_bank_stride == 64, "formal query bank stride drift")
    _require(args.pg19_document_tokens % args.page_size == 127, "formal 4095 tail stress drift")
    _require("longbench" not in str(args.pg19_data).lower(), "PG19 path must be train-only")
    _require("test-v2" not in str(args.pg19_data).lower(), "test-v2 path refused")
    _require("68-99" not in str(args.pg19_data).lower(), "source68-99 path refused")
    expected_digests = (
        args.expected_pg19_sha256,
        args.expected_pg19_manifest_sha256,
        args.expected_pg19_windows_sha256,
        args.expected_model_manifest_sha256,
        args.expected_code_ledger_sha256,
        args.expected_model_artifact_ledger_sha256,
        args.expected_model_weight_ledger_sha256,
        args.expected_protocol_manifest_sha256,
    )
    _require(
        all(len(value) == 64 and all(character in "0123456789abcdef" for character in value) for value in expected_digests),
        "invalid frozen SHA256",
    )
    for path, expected, label in (
        (args.code_ledger, args.expected_code_ledger_sha256, "code ledger"),
        (args.model_artifact_ledger, args.expected_model_artifact_ledger_sha256, "model artifact ledger"),
        (args.model_weight_ledger, args.expected_model_weight_ledger_sha256, "model weight ledger"),
    ):
        _require(path.is_file() and sha256_file(path) == expected, f"{label} drift")
    _require(sha256_file(args.pg19_data) == args.expected_pg19_sha256, "PG19 data drift")
    _require(sha256_file(args.pg19_manifest) == args.expected_pg19_manifest_sha256, "PG19 manifest drift")
    model_manifest_sha, model_manifest = _model_manifest_sha(args.model)
    _require(model_manifest_sha == args.expected_model_manifest_sha256, "model manifest drift")
    protocol_config = _protocol_config(args)
    protocol_config_sha = _protocol_config_sha256(protocol_config)
    protocol_manifest = _load_frozen_json(
        args.protocol_manifest,
        args.expected_protocol_manifest_sha256,
        "multi-fork runtime protocol manifest",
    )
    _require(protocol_manifest.get("protocol") == MULTIFORK_PROTOCOL, "protocol manifest name drift")
    _require(protocol_manifest.get("protocol_config") == protocol_config, "protocol manifest config drift")
    frozen_query_banks = protocol_manifest.get("frozen_query_banks")
    _require(
        isinstance(frozen_query_banks, list)
        and len(frozen_query_banks) == FORMAL_WORLD_SIZE,
        "protocol manifest must freeze eight query banks",
    )
    source_objects = []
    for rank, bank in enumerate(frozen_query_banks):
        _require(isinstance(bank, dict), f"query bank {rank} is not an object")
        _require(bank.get("synthetic_markers_used") is False, "formal query bank used synthetic markers")
        _require(
            bank.get("source_role")
            == "same-pg19-train-book-raw-nonoverlapping-query-chunks",
            "formal query bank source role drift",
        )
        _require(bank.get("count") == 32 and bank.get("query_tokens") == 32, "query bank geometry drift")
        _require(bank.get("query_stride_tokens") == args.query_bank_stride, "query bank stride drift")
        _require(bank.get("pairwise_nonoverlapping") is True, "query bank overlap")
        _require(bank.get("pairwise_distinct") is True, "query bank duplicate")
        rows = bank.get("rows")
        _require(isinstance(rows, list) and len(rows) == 32, "query bank row count drift")
        document_start = bank.get("document_start_token")
        document_end = bank.get("document_end_token_exclusive")
        bank_start = bank.get("query_bank_start_token")
        _require(type(document_start) is int and document_start >= 0, "query bank document offset drift")
        _require(document_end == document_start + FORMAL_DOCUMENT_TOKENS, "query bank document end drift")
        _require(bank_start == document_end + FORMAL_QUERY_TOKENS, "query bank start drift")
        _require(
            [row.get("request_index") for row in rows] == list(range(32))
            and [row.get("source_token_offset") for row in rows]
            == [bank_start + index * args.query_bank_stride for index in range(32)]
            and all(row.get("query_tokens") == FORMAL_QUERY_TOKENS for row in rows),
            "query bank row offset/geometry drift",
        )
        digests = [row.get("query_token_ids_sha256") for row in rows]
        _require(
            len(set(digests)) == 32
            and all(
                isinstance(value, str)
                and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in digests
            ),
            "query bank digest schema/uniqueness drift",
        )
        _require(
            isinstance(bank.get("query_bank_sha256"), str)
            and len(bank["query_bank_sha256"]) == 64,
            "query bank SHA missing",
        )
        source_objects.append(bank.get("source_object"))
    _require(len(set(source_objects)) == FORMAL_WORLD_SIZE, "ranks do not use distinct PG19 train books")
    frozen = protocol_manifest.get("frozen_identity")
    _require(isinstance(frozen, dict), "protocol manifest frozen identity missing")
    expected_identity = {
        "code_ledger_sha256": args.expected_code_ledger_sha256,
        "model_manifest_sha256": args.expected_model_manifest_sha256,
        "model_artifact_ledger_sha256": args.expected_model_artifact_ledger_sha256,
        "model_weight_ledger_sha256": args.expected_model_weight_ledger_sha256,
        "pg19_data_sha256": args.expected_pg19_sha256,
        "pg19_manifest_sha256": args.expected_pg19_manifest_sha256,
        "pg19_windows_sha256": args.expected_pg19_windows_sha256,
        "protocol_config_sha256": protocol_config_sha,
    }
    _require(frozen == expected_identity, "protocol manifest frozen identity differs from CLI")
    environment = audit_frozen_kernel_environment()
    _require(environment["matches_frozen_environment"] is True, "frozen kernel environment drift")
    model_geometry = _audit_model_config_geometry(args.model)
    return {
        "status": "multifork_resident_static_dry_run_passed",
        "protocol": MULTIFORK_PROTOCOL,
        **expected_identity,
        "protocol_manifest_sha256": args.expected_protocol_manifest_sha256,
        "protocol_config": protocol_config,
        "model_manifest": model_manifest,
        "environment": environment,
        "model_geometry": model_geometry,
        "frozen_query_banks": frozen_query_banks,
        "pg19_train_only": True,
        "longbench_consumed": False,
        "source_6_9_consumed": False,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
        "gpu_initialized": torch.cuda.is_initialized(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("static-dry-run", "resident-shard", "aggregate"),
        required=True,
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=FORMAL_WORLD_SIZE)
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pg19-data", type=Path, required=True)
    parser.add_argument("--pg19-manifest", type=Path, required=True)
    parser.add_argument("--expected-pg19-sha256", required=True)
    parser.add_argument("--expected-pg19-manifest-sha256", required=True)
    parser.add_argument("--expected-pg19-windows-sha256", required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--expected-protocol-manifest-sha256", required=True)
    parser.add_argument("--code-ledger", type=Path, required=True)
    parser.add_argument("--model-artifact-ledger", type=Path, required=True)
    parser.add_argument("--model-weight-ledger", type=Path, required=True)
    parser.add_argument("--expected-code-ledger-sha256", required=True)
    parser.add_argument("--expected-model-artifact-ledger-sha256", required=True)
    parser.add_argument("--expected-model-weight-ledger-sha256", required=True)
    parser.add_argument("--resident-counts", type=int, nargs="+", default=list(MULTIFORK_COUNTS))
    parser.add_argument("--execution-order", type=int, nargs="+", default=list(FORMAL_EXECUTION_ORDER))
    parser.add_argument("--page-size", type=int, default=FORMAL_PAGE_SIZE)
    parser.add_argument("--pg19-books", type=int, default=FORMAL_BOOKS)
    parser.add_argument("--pg19-document-tokens", type=int, default=FORMAL_DOCUMENT_TOKENS)
    parser.add_argument("--pg19-query-tokens", type=int, default=FORMAL_QUERY_TOKENS)
    parser.add_argument("--pg19-window-stride", type=int, default=FORMAL_WINDOW_STRIDE)
    parser.add_argument("--pg19-candidate-windows", type=int, default=FORMAL_CANDIDATES)
    parser.add_argument("--pg19-seed", type=int, default=FORMAL_SEED)
    parser.add_argument("--query-bank-stride", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=FORMAL_NEW_TOKENS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    static = _validate_static(args)
    args.static_audit = static
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.stage == "static-dry-run":
        result = static
    elif args.stage == "resident-shard":
        result = {"static": static, **run_resident_shard(args)}
    else:
        paths = sorted((args.run_dir / "resident-shards").glob("multifork-resident-shard-*.json"))
        result = aggregate_shards(
            paths,
            expected_frozen_identity=_static_frozen_identity(static),
            expected_query_banks=static["frozen_query_banks"],
        )
    atomic_json(args.output, result)
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
