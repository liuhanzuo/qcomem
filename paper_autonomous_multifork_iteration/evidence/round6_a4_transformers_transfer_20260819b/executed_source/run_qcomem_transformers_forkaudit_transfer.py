from __future__ import annotations

"""Formal eight-rank ForkAudit transfer on Transformers ``DynamicCache``.

The model checkpoint is intentionally held fixed while the runtime adapter,
cache representation, ownership construction, oracle, and fault suite differ
from the vLLM paged experiment.  Scientific failures are serialized as valid
negative outcomes; only evidence-integrity or infrastructure failures abort.
"""

import argparse
import json
import os
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from qcomem_torch import LowerReplayState, PackedLowerReplayState, TorchSplitCausalLM
from qcomem_transformers_forkaudit_transfer import (
    FAULT_CONTRACT,
    FANOUTS,
    PROTOCOL,
    SHARD_SCHEMA,
    SOURCE_SCHEMA,
    STATIC_SCHEMA,
    TransferEvidenceError,
    WORLD_SIZE,
    aggregate_shards,
    build_target_rows,
    classify_detector_vector,
    compare_logit_steps,
    disjointness_receipt,
    iter_tensor_slots,
    load_bound_json,
    require,
    semantic_key,
    sha256_bytes,
    sha256_file,
    sha256_json,
    state_content_receipt,
    storage_inventory,
    tensor_receipt,
    tensor_tree_receipt,
    validate_source_manifest,
    validate_sha256_ledger,
    validate_model_authority_receipt,
    validate_gpu_assignment,
    write_canonical_json,
)


MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
EXPECTED_MODEL_TYPE = "qwen3_5_moe_text"
EXPECTED_LAYERS = 40
EXPECTED_LAYER_TYPES = tuple(
    "full_attention" if index in range(3, 40, 4) else "linear_attention"
    for index in range(40)
)


@dataclass
class RequestSession:
    request_index: int
    query: torch.Tensor
    state: LowerReplayState
    suffix_cache: Any
    logits: torch.Tensor
    suffix_length: int
    generated: list[int]
    logit_steps: list[torch.Tensor]


class LogitBundleBuilder:
    """Append-only canonical little-endian CPU-FP32 full-logit bundle."""

    def __init__(self) -> None:
        self.payload = bytearray()
        self.records: list[dict[str, Any]] = []
        self.ids: set[str] = set()

    def add(self, record_id: str, tensor: torch.Tensor) -> dict[str, Any]:
        require(record_id not in self.ids and record_id, "duplicate/empty logit record ID")
        value = tensor.detach().float().cpu().contiguous()
        raw = value.numpy().astype("<f4", copy=False).tobytes(order="C")
        offset = len(self.payload)
        self.payload.extend(raw)
        record = {
            "record_id": record_id,
            "offset_bytes": offset,
            "nbytes": len(raw),
            "shape": list(value.shape),
            "dtype": "float32-le",
            "content_sha256": sha256_bytes(raw),
        }
        self.records.append(record)
        self.ids.add(record_id)
        return record

    def checkpoint(self) -> tuple[int, int, set[str]]:
        return len(self.payload), len(self.records), set(self.ids)

    def rollback(self, checkpoint: tuple[int, int, set[str]]) -> None:
        payload_bytes, record_count, ids = checkpoint
        del self.payload[payload_bytes:]
        del self.records[record_count:]
        self.ids = ids

    def write(self, path: Path) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_bytes(bytes(self.payload))
        temporary.replace(path)
        return {
            "schema_version": "forkaudit-fp32-logit-bundle-v1",
            "logical_name": path.name,
            "bytes": len(self.payload),
            "sha256": sha256_bytes(bytes(self.payload)),
            "record_count": len(self.records),
            "records": self.records,
            "terminal_closure": {
                "first_offset_bytes": self.records[0]["offset_bytes"],
                "last_end_offset_bytes": self.records[-1]["offset_bytes"]
                + self.records[-1]["nbytes"],
                "exact_byte_coverage": True,
            },
        }


def t5_ordinary_exception_receipt(error: BaseException) -> dict[str, Any]:
    """Serialize only preregistered ordinary model-call failures; fail closed otherwise."""

    if isinstance(error, (TransferEvidenceError, torch.cuda.OutOfMemoryError, MemoryError, RuntimeError)):
        raise error
    if not isinstance(error, AssertionError):
        raise error
    exception_type = f"{type(error).__module__}.{type(error).__qualname__}"
    return {
        "completed": False,
        "outputs_available": False,
        "runtime_exception": {
            "type": exception_type,
            "message_sha256": sha256_bytes(str(error).encode("utf-8")),
        },
        "ordinary_assertion_triggered": isinstance(error, AssertionError),
    }


def _model_geometry(adapter: TorchSplitCausalLM) -> dict[str, Any]:
    layer_types = tuple(getattr(adapter.config, "layer_types", ()))
    return {
        "model_type": str(getattr(adapter.config, "model_type", "")),
        "num_layers": adapter.num_layers,
        "layer_types": list(layer_types),
        "split_depth": 7,
        "matches_frozen": (
            getattr(adapter.config, "model_type", "") == EXPECTED_MODEL_TYPE
            and adapter.num_layers == EXPECTED_LAYERS
            and layer_types == EXPECTED_LAYER_TYPES
        ),
    }


def _gpu_identity(expected_uuid: str) -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid, "rank GPU isolation drift")
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={expected_uuid}",
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    fields = [item.strip() for item in result.stdout.strip().split(",")]
    require(len(fields) == 3 and fields[0] == expected_uuid, "GPU UUID receipt drift")
    properties = torch.cuda.get_device_properties(0)
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "uuid": fields[0],
        "name": fields[1],
        "total_memory_mib": int(fields[2]),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def _semantic_trace_key(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [semantic_key(row) for row in rows]


def _output_trace_key(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "request_index": row["request_index"],
            "query_token_ids_sha256": row["query_token_ids_sha256"],
            "generated_token_ids": row["generated_token_ids"],
            "step_logit_sha256": row["step_logit_sha256"],
        }
        for row in rows
    ]


def _fault_semantic_receipts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {**semantic_key(row), "step_logit_record_ids": row["step_logit_record_ids"]}
        for row in rows
    ]


def _allocator_snapshot(phase: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    return {
        "phase": phase,
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def _cache_snapshot(cache: Any, *, salt: str, role: str) -> dict[str, Any]:
    content = tensor_tree_receipt(cache)
    storage = storage_inventory(cache, salt=salt, role=role)
    require(content["tensor_count"] == storage["tensor_rows"], "cache snapshot tensor count drift")
    return {
        "tensor_count": content["tensor_count"],
        "content_sha256": content["content_sha256"],
        "storage": storage,
    }


def _call_receipt(
    *,
    call_index: int,
    request_index: int,
    phase: str,
    callable_name: str,
    layer_start: int,
    layer_end: int,
    position_offset: int,
    current_length_before: int,
    current_length_after: int,
    input_tokens: int,
    cache_before: dict[str, Any],
    cache_after: dict[str, Any],
) -> dict[str, Any]:
    return {
        "call_index": call_index,
        "request_index": request_index,
        "phase": phase,
        "callable": callable_name,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "position_offset": position_offset,
        "current_length_before": current_length_before,
        "current_length_after": current_length_after,
        "input_tokens": input_tokens,
        "append_delta": current_length_after - current_length_before,
        "cache_before": cache_before,
        "cache_after": cache_after,
        "completed": True,
    }


@torch.inference_mode()
def _dense_oracle(
    adapter: TorchSplitCausalLM,
    document: torch.Tensor,
    query: torch.Tensor,
    *,
    semantic_steps: int,
    logit_bundle: LogitBundleBuilder,
    record_prefix: str,
) -> tuple[dict[str, Any], list[torch.Tensor]]:
    continuation = query.clone()
    tokens: list[int] = []
    logits_cpu: list[torch.Tensor] = []
    record_ids: list[str] = []
    for _step in range(semantic_steps):
        logits = adapter.full_last_logits(torch.cat((document, continuation), dim=1))
        logits_cpu.append(logits.detach().float().cpu())
        record_id = f"{record_prefix}/step-{_step}"
        logit_bundle.add(record_id, logits_cpu[-1])
        record_ids.append(record_id)
        token = int(torch.argmax(logits, dim=-1).item())
        tokens.append(token)
        continuation = torch.cat(
            (
                continuation,
                torch.tensor([[token]], dtype=query.dtype, device=query.device),
            ),
            dim=1,
        )
    return (
        {
            "oracle_path": "AutoModelForImageTextToText.full_last_logits_dense_recompute",
            "generated_token_ids": tokens,
            "step_logit_sha256": [tensor_receipt(value)["content_sha256"] for value in logits_cpu],
            "step_logit_record_ids": record_ids,
            "semantic_steps": semantic_steps,
        },
        logits_cpu,
    )


@torch.inference_mode()
def _start_session(
    adapter: TorchSplitCausalLM,
    state: LowerReplayState,
    query: torch.Tensor,
    request_index: int,
    *,
    salt: str,
    arm: str,
    call_ledger: list[dict[str, Any]],
) -> RequestSession:
    require(state.current_length == state.document_length, "request state starts at wrong position")
    lower_before_length = state.current_length
    lower_before = _cache_snapshot(
        state.cache, salt=salt, role=f"{arm}/request-{request_index}/call-{len(call_ledger)}/before"
    )
    query_residual = adapter.continue_lower_replay(state, query)
    lower_after = _cache_snapshot(
        state.cache, salt=salt, role=f"{arm}/request-{request_index}/call-{len(call_ledger)}/after"
    )
    call_ledger.append(
        _call_receipt(
            call_index=len(call_ledger), request_index=request_index, phase="first-query-lower",
            callable_name="TorchSplitCausalLM.continue_lower_replay", layer_start=0,
            layer_end=state.depth, position_offset=lower_before_length,
            current_length_before=lower_before_length, current_length_after=state.current_length,
            input_tokens=int(query.shape[1]), cache_before=lower_before, cache_after=lower_after,
        )
    )
    suffix_cache = adapter.make_cache()
    suffix_before = _cache_snapshot(
        suffix_cache, salt=salt, role=f"{arm}/request-{request_index}/call-{len(call_ledger)}/before"
    )
    adapter.run_suffix_cached_last_logits(
        [state.document_residual],
        state.depth,
        suffix_cache,
        position_offset=0,
    )
    suffix_document_after = _cache_snapshot(
        suffix_cache, salt=salt, role=f"{arm}/request-{request_index}/call-{len(call_ledger)}/after"
    )
    call_ledger.append(
        _call_receipt(
            call_index=len(call_ledger), request_index=request_index, phase="suffix-document",
            callable_name="TorchSplitCausalLM.run_suffix_cached_last_logits", layer_start=state.depth,
            layer_end=adapter.num_layers, position_offset=0, current_length_before=0,
            current_length_after=state.document_length, input_tokens=state.document_length,
            cache_before=suffix_before, cache_after=suffix_document_after,
        )
    )
    suffix_query_before = _cache_snapshot(
        suffix_cache, salt=salt, role=f"{arm}/request-{request_index}/call-{len(call_ledger)}/before"
    )
    logits = adapter.run_suffix_cached_last_logits(
        [query_residual],
        state.depth,
        suffix_cache,
        position_offset=state.document_length,
    )
    suffix_query_after = _cache_snapshot(
        suffix_cache, salt=salt, role=f"{arm}/request-{request_index}/call-{len(call_ledger)}/after"
    )
    call_ledger.append(
        _call_receipt(
            call_index=len(call_ledger), request_index=request_index, phase="first-query-suffix",
            callable_name="TorchSplitCausalLM.run_suffix_cached_last_logits", layer_start=state.depth,
            layer_end=adapter.num_layers, position_offset=state.document_length,
            current_length_before=state.document_length,
            current_length_after=state.document_length + int(query.shape[1]), input_tokens=int(query.shape[1]),
            cache_before=suffix_query_before, cache_after=suffix_query_after,
        )
    )
    return RequestSession(
        request_index=request_index,
        query=query,
        state=state,
        suffix_cache=suffix_cache,
        logits=logits,
        suffix_length=state.current_length,
        generated=[],
        logit_steps=[],
    )


@torch.inference_mode()
def _drive_sessions(
    adapter: TorchSplitCausalLM,
    sessions: Sequence[RequestSession],
    *,
    semantic_steps: int,
    salt: str,
    arm: str,
    call_ledger: list[dict[str, Any]],
) -> None:
    """Sequentially interleave live resident requests on one CUDA stream."""

    for step in range(semantic_steps):
        for session in sessions:
            session.logit_steps.append(session.logits.detach().float().cpu())
            token = int(torch.argmax(session.logits, dim=-1).item())
            session.generated.append(token)
            if step + 1 < semantic_steps:
                token_tensor = torch.tensor(
                    [[token]], dtype=session.query.dtype, device=session.query.device
                )
                lower_before_length = session.state.current_length
                lower_before = _cache_snapshot(
                    session.state.cache, salt=salt,
                    role=f"{arm}/request-{session.request_index}/call-{len(call_ledger)}/before",
                )
                token_residual = adapter.continue_lower_replay(session.state, token_tensor)
                lower_after = _cache_snapshot(
                    session.state.cache, salt=salt,
                    role=f"{arm}/request-{session.request_index}/call-{len(call_ledger)}/after",
                )
                call_ledger.append(
                    _call_receipt(
                        call_index=len(call_ledger), request_index=session.request_index,
                        phase=f"generated-step-{step}-lower",
                        callable_name="TorchSplitCausalLM.continue_lower_replay", layer_start=0,
                        layer_end=session.state.depth, position_offset=lower_before_length,
                        current_length_before=lower_before_length,
                        current_length_after=session.state.current_length, input_tokens=1,
                        cache_before=lower_before, cache_after=lower_after,
                    )
                )
                suffix_before = _cache_snapshot(
                    session.suffix_cache, salt=salt,
                    role=f"{arm}/request-{session.request_index}/call-{len(call_ledger)}/before",
                )
                session.logits = adapter.run_suffix_cached_last_logits(
                    [token_residual],
                    session.state.depth,
                    session.suffix_cache,
                    position_offset=session.suffix_length,
                )
                suffix_after = _cache_snapshot(
                    session.suffix_cache, salt=salt,
                    role=f"{arm}/request-{session.request_index}/call-{len(call_ledger)}/after",
                )
                call_ledger.append(
                    _call_receipt(
                        call_index=len(call_ledger), request_index=session.request_index,
                        phase=f"generated-step-{step}-suffix",
                        callable_name="TorchSplitCausalLM.run_suffix_cached_last_logits",
                        layer_start=session.state.depth, layer_end=adapter.num_layers,
                        position_offset=session.suffix_length,
                        current_length_before=session.suffix_length,
                        current_length_after=session.suffix_length + 1, input_tokens=1,
                        cache_before=suffix_before, cache_after=suffix_after,
                    )
                )
                session.suffix_length += 1


@torch.inference_mode()
def _run_owned_arm(
    adapter: TorchSplitCausalLM,
    states: Sequence[LowerReplayState],
    queries: Sequence[torch.Tensor],
    *,
    arm: str,
    semantic_steps: int,
    salt: str,
    persistent_base: LowerReplayState | None,
    logit_bundle: LogitBundleBuilder,
    record_prefix: str,
) -> tuple[dict[str, Any], list[list[torch.Tensor]]]:
    require(len(states) == len(queries) and len(states) in FANOUTS, "arm fan-out drift")
    setup_inventories = [
        storage_inventory(state.cache, salt=salt, role=f"{arm}/request-{index}/lower-cache/setup")
        for index, state in enumerate(states)
    ]
    residual_inventories = [
        storage_inventory(
            [state.document_residual],
            salt=salt,
            role=f"{arm}/request-{index}/document-residual",
        )
        for index, state in enumerate(states)
    ]
    forbidden = (
        [storage_inventory(persistent_base.cache, salt=salt, role=f"{arm}/persistent-base/lower-cache")]
        if persistent_base is not None
        else []
    )
    setup_disjoint = disjointness_receipt(setup_inventories, forbidden=forbidden)
    allocator_snapshots = [_allocator_snapshot("setup")]
    persistent_residual_inventory = None
    if persistent_base is not None:
        persistent_residual_inventory = storage_inventory(
            [persistent_base.document_residual],
            salt=salt,
            role=f"{arm}/persistent-base/document-residual",
        )
        residual_ownership_receipt = {
            "predicate_id": "READ_ONLY_DOCUMENT_RESIDUAL_ALIASES_PERSISTENT_BASE",
            "tensor_pair_comparison_count": sum(
                len(inventory["rows"]) * len(persistent_residual_inventory["rows"])
                for inventory in residual_inventories
            ),
            "passed": all(
                {
                    (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
                    for row in inventory["rows"]
                }
                == {
                    (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
                    for row in persistent_residual_inventory["rows"]
                }
                for inventory in residual_inventories
            ),
        }
    else:
        residual_ownership_receipt = disjointness_receipt(residual_inventories)

    call_ledger: list[dict[str, Any]] = []
    sessions = [
        _start_session(
            adapter, state, query, index, salt=salt, arm=arm, call_ledger=call_ledger
        )
        for index, (state, query) in enumerate(zip(states, queries))
    ]
    first_transition_inventories = []
    for session in sessions:
        lower = storage_inventory(
            session.state.cache, salt=salt,
            role=f"{arm}/request-{session.request_index}/all-mutable-cache/first-transition/lower",
        )
        suffix = storage_inventory(
            session.suffix_cache, salt=salt,
            role=f"{arm}/request-{session.request_index}/all-mutable-cache/first-transition/suffix",
        )
        combined = {
            "role": f"{arm}/request-{session.request_index}/all-mutable-cache/first-transition",
            "storage_salt_domain_sha256": lower["storage_salt_domain_sha256"],
            "rows": [*lower["rows"], *suffix["rows"]],
        }
        combined["tensor_rows"] = len(combined["rows"])
        combined["inventory_sha256"] = sha256_json(combined["rows"])
        first_transition_inventories.append(combined)
    first_transition_disjoint = disjointness_receipt(
        first_transition_inventories, forbidden=forbidden
    )
    allocator_snapshots.append(_allocator_snapshot("first_transition"))
    _drive_sessions(
        adapter, sessions, semantic_steps=semantic_steps, salt=salt, arm=arm,
        call_ledger=call_ledger,
    )
    final_inventories = []
    rows = []
    step_tensors = []
    for session in sessions:
        lower = storage_inventory(
            session.state.cache,
            salt=salt,
            role=f"{arm}/request-{session.request_index}/lower-cache/final",
        )
        suffix = storage_inventory(
            session.suffix_cache,
            salt=salt,
            role=f"{arm}/request-{session.request_index}/suffix-cache/final",
        )
        combined = {
            "role": f"{arm}/request-{session.request_index}/all-mutable-cache/final",
            "storage_salt_domain_sha256": lower["storage_salt_domain_sha256"],
            "rows": [*lower["rows"], *suffix["rows"]],
        }
        combined["tensor_rows"] = len(combined["rows"])
        combined["inventory_sha256"] = sha256_json(combined["rows"])
        final_inventories.append(combined)
        query_receipt = tensor_receipt(session.query)
        suffix_receipt = tensor_tree_receipt(session.suffix_cache)
        lower_receipt = state_content_receipt(session.state)
        rows.append(
            {
                "request_index": session.request_index,
                "query_token_ids_sha256": query_receipt["content_sha256"],
                "generated_token_ids": session.generated,
                "step_logit_sha256": [
                    tensor_receipt(value)["content_sha256"] for value in session.logit_steps
                ],
                "step_logit_record_ids": [
                    logit_bundle.add(
                        f"{record_prefix}/request-{session.request_index}/step-{step}",
                        value,
                    )["record_id"]
                    for step, value in enumerate(session.logit_steps)
                ],
                "final_lower_state_sha256": lower_receipt["state_content_sha256"],
                "final_lower_cache_content_sha256": lower_receipt["cache_content_sha256"],
                "final_suffix_cache_sha256": suffix_receipt["content_sha256"],
                "final_current_length": session.state.current_length,
                "lower_cache_storage": lower,
                "suffix_cache_storage": suffix,
            }
        )
        step_tensors.append(session.logit_steps)
    final_disjoint = disjointness_receipt(final_inventories, forbidden=forbidden)
    allocator_snapshots.append(_allocator_snapshot("final"))
    return (
        {
            "arm": arm,
            "fanout": len(states),
            "state_construction": (
                "one-persistent-prefix-then-LowerReplayState.fork"
                if persistent_base is not None
                else "independent-write_lower_replay-per-request"
            ),
            "scheduler": "single-cuda-stream-request-index-interleaved",
            "setup_storage_inventories": setup_inventories,
            "persistent_forbidden_inventories": forbidden,
            "first_transition_combined_storage_inventories": first_transition_inventories,
            "final_combined_storage_inventories": final_inventories,
            "document_residual_storage_inventories": residual_inventories,
            "persistent_document_residual_inventory": persistent_residual_inventory,
            "setup_disjointness": setup_disjoint,
            "first_transition_disjointness": first_transition_disjoint,
            "final_disjointness": final_disjoint,
            "document_residual_ownership": residual_ownership_receipt,
            "adapter_call_ledger": call_ledger,
            "allocator_accounting_snapshots": allocator_snapshots,
            "semantics": rows,
        },
        step_tensors,
    )


def _oracle_comparison_rows(
    arm_rows: Sequence[dict[str, Any]],
    arm_logits: Sequence[Sequence[torch.Tensor]],
    oracle_rows: Sequence[dict[str, Any]],
    oracle_logits: Sequence[Sequence[torch.Tensor]],
    *,
    threshold: float,
) -> list[dict[str, Any]]:
    rows = []
    for request_index, (arm_row, actual, oracle_row, expected) in enumerate(
        zip(arm_rows, arm_logits, oracle_rows, oracle_logits)
    ):
        metrics = compare_logit_steps(actual, expected, relative_l2_threshold=threshold)
        token_match = arm_row["generated_token_ids"] == oracle_row["generated_token_ids"]
        rows.append(
            {
                "request_index": request_index,
                "token_match": token_match,
                "numeric": metrics,
                "passed": token_match and metrics["passed"],
            }
        )
    return rows


def _replace_first_compatible_alias(
    source_cache: Any,
    target_cache: Any,
    *,
    salt: str,
) -> dict[str, Any]:
    source_slots = list(iter_tensor_slots(source_cache))
    target_slots = list(iter_tensor_slots(target_cache))
    for source in source_slots:
        for target in target_slots:
            if (
                target.replaceable
                and source.tensor.shape == target.tensor.shape
                and source.tensor.dtype == target.tensor.dtype
            ):
                before = storage_inventory(target_cache, salt=salt, role="alias-target-cache")
                target_before = tensor_receipt(target.tensor)
                target.replace(source.tensor)
                after = storage_inventory(target_cache, salt=salt, role="alias-target-cache")
                return {
                    "source_path": source.path,
                    "target_path": target.path,
                    "source_tensor": tensor_receipt(source.tensor),
                    "target_tensor_before": target_before,
                    "target_inventory_before_sha256": before["inventory_sha256"],
                    "target_inventory_after_sha256": after["inventory_sha256"],
                    "mutated": before["inventory_sha256"] != after["inventory_sha256"],
                }
    raise RuntimeError("no compatible cache tensor pair exists for the alias fault")


@torch.inference_mode()
def _fault_suite(
    adapter: TorchSplitCausalLM,
    document: torch.Tensor,
    queries: Sequence[torch.Tensor],
    persistent_base: LowerReplayState,
    clean_n1: dict[str, Any],
    clean_n1_logits: dict[str, list[list[torch.Tensor]]],
    oracle_rows: Sequence[dict[str, Any]],
    oracle_logits: Sequence[Sequence[torch.Tensor]],
    *,
    depth: int,
    semantic_steps: int,
    threshold: float,
    salt: str,
    logit_bundle: LogitBundleBuilder,
) -> list[dict[str, Any]]:
    contracts = {row["fault_id"]: row for row in FAULT_CONTRACT}
    rows = []

    # T1 changes both ownership arms identically.  Their relational equality
    # may therefore remain green; the separately computed dense model oracle
    # is the predeclared detector.
    materialized = adapter.write_lower_replay(document, depth)
    corrupted_persistent_base = persistent_base.fork()
    corrupted_persistent_base.document_residual = (
        corrupted_persistent_base.document_residual.detach().clone()
    )
    t1_materialized_before = tensor_receipt(materialized.document_residual)
    t1_base_before = tensor_receipt(corrupted_persistent_base.document_residual)
    t1_materialized_storage_before = storage_inventory(
        [materialized.document_residual], salt=salt, role="T1-materialized-residual"
    )
    t1_base_storage_before = storage_inventory(
        [corrupted_persistent_base.document_residual], salt=salt, role="T1-corrupted-base-residual"
    )
    for state in (materialized, corrupted_persistent_base):
        state.document_residual.mul_(-1)
    t1_materialized_after = tensor_receipt(materialized.document_residual)
    t1_base_after = tensor_receipt(corrupted_persistent_base.document_residual)
    t1_materialized_storage_after = storage_inventory(
        [materialized.document_residual], salt=salt, role="T1-materialized-residual"
    )
    t1_base_storage_after = storage_inventory(
        [corrupted_persistent_base.document_residual], salt=salt, role="T1-corrupted-base-residual"
    )
    persistent = corrupted_persistent_base.fork()
    t1_persistent_after = tensor_receipt(persistent.document_residual)
    t1_persistent_storage = storage_inventory(
        [persistent.document_residual], salt=salt, role="T1-persistent-fork-residual"
    )
    t1_changed_identically = (
        t1_materialized_before["content_sha256"] == t1_base_before["content_sha256"]
        and t1_materialized_after["content_sha256"] == t1_base_after["content_sha256"]
        and t1_materialized_before["content_sha256"] != t1_materialized_after["content_sha256"]
    )
    t1_persistent_alias = (
        t1_persistent_after["content_sha256"] == t1_base_after["content_sha256"]
        and {
            (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
            for row in t1_persistent_storage["rows"]
        }
        == {
            (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
            for row in t1_base_storage_after["rows"]
        }
    )
    corrupt_materialized, corrupt_materialized_logits = _run_owned_arm(
        adapter,
        [materialized],
        queries[:1],
        arm="fault-common-mode-deep-materialized",
        semantic_steps=semantic_steps,
        salt=salt,
        persistent_base=None,
        logit_bundle=logit_bundle,
        record_prefix="fault-T1/deep-materialized",
    )
    corrupt_persistent, corrupt_persistent_logits = _run_owned_arm(
        adapter,
        [persistent],
        queries[:1],
        arm="fault-common-mode-persistent-fork",
        semantic_steps=semantic_steps,
        salt=salt,
        persistent_base=corrupted_persistent_base,
        logit_bundle=logit_bundle,
        record_prefix="fault-T1/persistent-fork",
    )
    clean_oracle_pass = all(
        row["passed"]
        for arm in ("deep_materialized", "persistent_fork")
        for row in clean_n1["oracle_comparisons"][arm]
    )
    mutant_comparisons = {
        "deep_materialized": _oracle_comparison_rows(
            corrupt_materialized["semantics"],
            corrupt_materialized_logits,
            oracle_rows[:1],
            oracle_logits[:1],
            threshold=threshold,
        ),
        "persistent_fork": _oracle_comparison_rows(
            corrupt_persistent["semantics"],
            corrupt_persistent_logits,
            oracle_rows[:1],
            oracle_logits[:1],
            threshold=threshold,
        ),
    }
    mutant_oracle_pass = all(
        item["passed"] for values in mutant_comparisons.values() for item in values
    )
    common_mode_exact = _semantic_trace_key(corrupt_materialized["semantics"]) == _semantic_trace_key(
        corrupt_persistent["semantics"]
    )
    expected = contracts["T1"]["expected_predicate"]
    clean_cross_arm = _semantic_trace_key(
        clean_n1["arms"]["deep_materialized"]["semantics"]
    ) == _semantic_trace_key(clean_n1["arms"]["persistent_fork"]["semantics"])
    detector_vector = {
        "matched_clean": {
            "INDEPENDENT_DENSE_SEMANTIC_ORACLE": clean_oracle_pass,
            "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": clean_cross_arm,
        },
        "mutant": {
            "INDEPENDENT_DENSE_SEMANTIC_ORACLE": mutant_oracle_pass,
            "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": common_mode_exact,
        },
    }
    classification = classify_detector_vector(
        expected_predicate=expected,
        matched_clean=detector_vector["matched_clean"],
        mutant=detector_vector["mutant"],
    )
    rows.append(
        {
            **contracts["T1"],
            "exercise_kind": "downstream_runtime_fault",
            "execution_outcome": {"completed": True, "outputs_available": True, "runtime_exception": None, "ordinary_assertion_triggered": False},
            "matched_clean": {
                "predicate_passed": clean_oracle_pass,
                "source": "clean N=1 dense-oracle receipts",
            },
            "mutant": {
                "injection": "digest-proven common-mode document-boundary residual content mutation in both arms",
                "injection_receipt": {
                    "materialized_before": t1_materialized_before,
                    "materialized_after": t1_materialized_after,
                    "corrupted_base_before": t1_base_before,
                    "corrupted_base_after": t1_base_after,
                    "persistent_fork_after": t1_persistent_after,
                    "materialized_storage_before": t1_materialized_storage_before,
                    "materialized_storage_after": t1_materialized_storage_after,
                    "corrupted_base_storage_before": t1_base_storage_before,
                    "corrupted_base_storage_after": t1_base_storage_after,
                    "persistent_fork_storage": t1_persistent_storage,
                    "changed_identically": t1_changed_identically,
                    "persistent_aliases_corrupted_base": t1_persistent_alias,
                },
                "common_mode_cross_arm_exact": common_mode_exact,
                "cross_arm_semantics": {
                    "deep_materialized": _fault_semantic_receipts(corrupt_materialized["semantics"]),
                    "persistent_fork": _fault_semantic_receipts(corrupt_persistent["semantics"]),
                },
                "oracle_comparisons": mutant_comparisons,
                "predicate_passed": mutant_oracle_pass,
            },
            "classification": classification,
            "detector_vector": detector_vector,
            "fault_case_valid": t1_changed_identically and t1_persistent_alias,
        }
    )

    # T2 is a direct live-storage fault with a separately rebuilt clean pair.
    clean_states = [persistent_base.fork(), persistent_base.fork()]
    clean_inventories = [
        storage_inventory(state.cache, salt=salt, role=f"T2-clean-request-{index}")
        for index, state in enumerate(clean_states)
    ]
    base_inventory = storage_inventory(persistent_base.cache, salt=salt, role="T2-persistent-base")
    clean_gate = disjointness_receipt(clean_inventories, forbidden=[base_inventory])
    mutant_states = [persistent_base.fork(), persistent_base.fork()]
    alias_binding = _replace_first_compatible_alias(mutant_states[0].cache, mutant_states[1].cache, salt=salt)
    mutant_inventories = [
        storage_inventory(state.cache, salt=salt, role=f"T2-mutant-request-{index}")
        for index, state in enumerate(mutant_states)
    ]
    mutant_gate = disjointness_receipt(mutant_inventories, forbidden=[base_inventory])
    expected = contracts["T2"]["expected_predicate"]
    detector_vector = {
        "matched_clean": {expected: clean_gate["passed"]},
        "mutant": {expected: mutant_gate["passed"]},
    }
    classification = classify_detector_vector(
        expected_predicate=expected,
        matched_clean=detector_vector["matched_clean"],
        mutant=detector_vector["mutant"],
    )
    rows.append(
        {
            **contracts["T2"],
            "exercise_kind": "direct_contract_sensitivity",
            "execution_outcome": {"completed": True, "outputs_available": False, "runtime_exception": None, "ordinary_assertion_triggered": False},
            "matched_clean": {
                "inventories": clean_inventories,
                "forbidden_inventories": [base_inventory],
                "gate": clean_gate,
            },
            "mutant": {
                "binding": alias_binding,
                "inventories": mutant_inventories,
                "forbidden_inventories": [base_inventory],
                "gate": mutant_gate,
            },
            "classification": classification,
            "detector_vector": detector_vector,
            "fault_case_valid": alias_binding["mutated"],
        }
    )

    # T3 freezes the canonical continuation start as a composite relation,
    # rather than importing the position gate from the paged runtime.
    clean_position = persistent_base.fork()
    clean_position_pass = clean_position.current_length == clean_position.document_length
    mutant_position = persistent_base.fork()
    before_length = mutant_position.current_length
    mutant_position.current_length += 1
    mutant_position_pass = mutant_position.current_length == mutant_position.document_length
    expected = contracts["T3"]["expected_predicate"]
    detector_vector = {
        "matched_clean": {expected: clean_position_pass},
        "mutant": {expected: mutant_position_pass},
    }
    classification = classify_detector_vector(
        expected_predicate=expected,
        matched_clean=detector_vector["matched_clean"],
        mutant=detector_vector["mutant"],
    )
    rows.append(
        {
            **contracts["T3"],
            "exercise_kind": "direct_contract_sensitivity",
            "execution_outcome": {"completed": True, "outputs_available": False, "runtime_exception": None, "ordinary_assertion_triggered": False},
            "matched_clean": {
                "document_length": clean_position.document_length,
                "current_length": clean_position.current_length,
                "next_position": clean_position.current_length,
                "predicate_passed": clean_position_pass,
            },
            "mutant": {
                "document_length": mutant_position.document_length,
                "current_length_before": before_length,
                "current_length_after": mutant_position.current_length,
                "next_position_after": mutant_position.current_length,
                "predicate_passed": mutant_position_pass,
            },
            "classification": classification,
            "detector_vector": detector_vector,
            "fault_case_valid": mutant_position.current_length == before_length + 1,
        }
    )

    # T4 exercises the packed Transformers-cache representation without making
    # it a third performance arm.  A one-element Q16 residual mutation must be
    # rejected by the frozen content-integrity predicate.
    packed: PackedLowerReplayState = persistent_base.quantize(
        bits=16,
        attention_bits=16,
        linear_bits=16,
        group_size=64,
    )
    clean_packed = tensor_tree_receipt(packed)
    expected_digest = clean_packed["content_sha256"]
    observed_clean_digest = tensor_tree_receipt(packed)["content_sha256"]
    clean_packed_pass = observed_clean_digest == expected_digest
    data = packed.document_residual.data
    require(data.numel() > 0, "packed residual is empty")
    original = data.reshape(-1)[0].detach().clone()
    data.reshape(-1)[0] = original + torch.tensor(1.0, dtype=data.dtype, device=data.device)
    mutant_digest = tensor_tree_receipt(packed)["content_sha256"]
    mutant_packed_pass = mutant_digest == expected_digest
    expected = contracts["T4"]["expected_predicate"]
    detector_vector = {
        "matched_clean": {expected: clean_packed_pass},
        "mutant": {expected: mutant_packed_pass},
    }
    classification = classify_detector_vector(
        expected_predicate=expected,
        matched_clean=detector_vector["matched_clean"],
        mutant=detector_vector["mutant"],
    )
    rows.append(
        {
            **contracts["T4"],
            "exercise_kind": "direct_contract_sensitivity",
            "execution_outcome": {"completed": True, "outputs_available": False, "runtime_exception": None, "ordinary_assertion_triggered": False},
            "matched_clean": {
                "expected_pre_content_sha256": expected_digest,
                "observed_clean_content_sha256": observed_clean_digest,
                "packed_state_content_sha256": expected_digest,
                "predicate_passed": clean_packed_pass,
            },
            "mutant": {
                "target": "PackedLowerReplayState.document_residual.data[0]",
                "pre_value": float(original.float().item()),
                "post_value": float(data.reshape(-1)[0].float().item()),
                "packed_state_content_sha256": mutant_digest,
                "predicate_passed": mutant_packed_pass,
            },
            "classification": classification,
            "detector_vector": detector_vector,
            "fault_case_valid": mutant_digest != expected_digest,
        }
    )

    # T5 is deliberately output-defined rather than a gate-shaped ownership
    # mutant: corrupt one live lower-cache value without changing any storage
    # identity, then observe the full downstream detector vector.
    materialized_t5 = adapter.write_lower_replay(document, depth)
    persistent_t5 = persistent_base.fork()
    mutable_slot = next(
        slot
        for slot in iter_tensor_slots(persistent_t5.cache)
        if slot.tensor.is_floating_point() and slot.tensor.numel() > 0
    )
    t5_pre = tensor_receipt(mutable_slot.tensor)
    t5_storage_before = storage_inventory(
        [mutable_slot.tensor], salt=salt, role="T5-mutated-cache-tensor"
    )
    original = mutable_slot.tensor.reshape(-1)[0].detach().clone()
    mutable_slot.tensor.reshape(-1)[0] = original + torch.tensor(
        1024.0, dtype=mutable_slot.tensor.dtype, device=mutable_slot.tensor.device
    )
    t5_post = tensor_receipt(mutable_slot.tensor)
    t5_storage_after = storage_inventory(
        [mutable_slot.tensor], salt=salt, role="T5-mutated-cache-tensor"
    )

    def execute_t5(
        state: LowerReplayState,
        *,
        arm: str,
        base: LowerReplayState | None,
        record_prefix: str,
        allow_scientific_exception: bool,
    ) -> tuple[dict[str, Any], dict[str, Any] | None, list[list[torch.Tensor]]]:
        checkpoint = logit_bundle.checkpoint()
        try:
            arm_row, arm_logits = _run_owned_arm(
                adapter,
                [state],
                queries[:1],
                arm=arm,
                semantic_steps=semantic_steps,
                salt=salt,
                persistent_base=base,
                logit_bundle=logit_bundle,
                record_prefix=record_prefix,
            )
            execution = {
                "completed": True,
                "outputs_available": True,
                "runtime_exception": None,
                "ordinary_assertion_triggered": False,
            }
            return execution, arm_row, arm_logits
        except TransferEvidenceError:
            logit_bundle.rollback(checkpoint)
            raise
        except (torch.cuda.OutOfMemoryError, MemoryError, RuntimeError):
            # CUDA/runtime and resource failures are infrastructure failures, never
            # scientific detections.  Fail closed without a shard.
            logit_bundle.rollback(checkpoint)
            raise
        except AssertionError as error:
            # These are the only preregistered ordinary model-call failures that
            # may become a scientific output-or-exception receipt.
            logit_bundle.rollback(checkpoint)
            if not allow_scientific_exception:
                raise
            execution = t5_ordinary_exception_receipt(error)
            return execution, None, []

    materialized_execution, materialized_t5_row, materialized_t5_logits = execute_t5(
        materialized_t5,
        arm="fault-live-value-deep-materialized",
        base=None,
        record_prefix="fault-T5/deep-materialized",
        allow_scientific_exception=False,
    )
    persistent_execution, persistent_t5_row, persistent_t5_logits = execute_t5(
        persistent_t5,
        arm="fault-live-value-persistent-fork",
        base=persistent_base,
        record_prefix="fault-T5/persistent-fork",
        allow_scientific_exception=True,
    )
    t5_rows = {
        "deep_materialized": materialized_t5_row["semantics"] if materialized_t5_row else [],
        "persistent_fork": persistent_t5_row["semantics"] if persistent_t5_row else [],
    }
    t5_logits = {
        "deep_materialized": materialized_t5_logits,
        "persistent_fork": persistent_t5_logits,
    }
    t5_comparisons = {
        arm: (
            _oracle_comparison_rows(
                t5_rows[arm],
                t5_logits[arm],
                oracle_rows[:1],
                oracle_logits[:1],
                threshold=threshold,
            )
            if t5_rows[arm]
            else []
        )
        for arm in ("deep_materialized", "persistent_fork")
    }
    executions = {
        "deep_materialized": materialized_execution,
        "persistent_fork": persistent_execution,
    }
    completed = all(item["completed"] for item in executions.values())
    t5_oracle_pass = completed and all(
        values and all(item["passed"] for item in values)
        for values in t5_comparisons.values()
    )
    t5_materialized_oracle_pass = materialized_execution["completed"] and bool(
        t5_comparisons["deep_materialized"]
    ) and all(item["passed"] for item in t5_comparisons["deep_materialized"])
    t5_state_cross_arm = completed and _semantic_trace_key(t5_rows["deep_materialized"]) == _semantic_trace_key(
        t5_rows["persistent_fork"]
    )
    t5_output_cross_arm = completed and _output_trace_key(t5_rows["deep_materialized"]) == _output_trace_key(
        t5_rows["persistent_fork"]
    )
    t5_clean_cross = _semantic_trace_key(
        clean_n1["arms"]["deep_materialized"]["semantics"]
    ) == _semantic_trace_key(clean_n1["arms"]["persistent_fork"]["semantics"])
    t5_clean_output = _output_trace_key(
        clean_n1["arms"]["deep_materialized"]["semantics"]
    ) == _output_trace_key(clean_n1["arms"]["persistent_fork"]["semantics"])
    expected = contracts["T5"]["expected_predicate"]
    detector_vector = {
        "matched_clean": {
            "INDEPENDENT_DENSE_SEMANTIC_ORACLE": clean_oracle_pass,
            "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": t5_clean_cross,
            "STATE_CROSS_ARM": t5_clean_cross,
            "DOWNSTREAM_OUTPUT_CONSISTENCY": t5_clean_output,
        },
        "mutant": {
            "INDEPENDENT_DENSE_SEMANTIC_ORACLE": t5_oracle_pass,
            "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": t5_state_cross_arm,
            "STATE_CROSS_ARM": t5_state_cross_arm,
            "DOWNSTREAM_OUTPUT_CONSISTENCY": t5_output_cross_arm,
        },
    }
    classification = classify_detector_vector(
        expected_predicate=expected,
        matched_clean=detector_vector["matched_clean"],
        mutant=detector_vector["mutant"],
    )
    rows.append(
        {
            **contracts["T5"],
            "exercise_kind": "downstream_runtime_fault",
            "execution_outcome": {
                "completed": all(item["completed"] for item in executions.values()),
                "outputs_available": all(item["outputs_available"] for item in executions.values()),
                "runtime_exception": persistent_execution["runtime_exception"],
                "ordinary_assertion_triggered": any(item["ordinary_assertion_triggered"] for item in executions.values()),
            },
            "matched_clean": {
                "oracle_passed": clean_oracle_pass,
                "state_cross_arm_exact": t5_clean_cross,
                "output_cross_arm_exact": t5_clean_output,
            },
            "mutant": {
                "target_path": mutable_slot.path,
                "target_pre": t5_pre,
                "target_post": t5_post,
                "target_storage_before": t5_storage_before,
                "target_storage_after": t5_storage_after,
                "one_element_delta": float(
                    mutable_slot.tensor.reshape(-1)[0].float().item() - original.float().item()
                ),
                "executions": executions,
                "cross_arm_semantics": {
                    arm: _fault_semantic_receipts(t5_rows[arm])
                    for arm in ("deep_materialized", "persistent_fork")
                },
                "oracle_comparisons": t5_comparisons,
                "state_cross_arm_exact": t5_state_cross_arm,
                "output_cross_arm_exact": t5_output_cross_arm,
                "oracle_passed": t5_oracle_pass,
            },
            "classification": classification,
            "detector_vector": detector_vector,
            "fault_case_valid": (
                t5_pre["content_sha256"] != t5_post["content_sha256"]
                and t5_storage_before["rows"] == t5_storage_after["rows"]
                and materialized_execution["completed"]
                and t5_materialized_oracle_pass
            ),
        }
    )
    return rows


def _load_static(args: argparse.Namespace) -> dict[str, Any]:
    static = load_bound_json(args.static_manifest, args.expected_static_manifest_sha256, "static manifest")
    require(static.get("schema_version") == STATIC_SCHEMA, "static manifest schema drift")
    require(static.get("protocol") == PROTOCOL, "static protocol drift")
    require(static.get("source_manifest_raw_sha256") == args.expected_source_manifest_sha256, "static/source binding drift")
    require(static.get("model", {}).get("model_id") == MODEL_ID, "model ID drift")
    require(static.get("model", {}).get("model_revision") == MODEL_REVISION, "model revision drift")
    require(static.get("formal_config", {}).get("world_size") == WORLD_SIZE, "formal world-size drift")
    require(tuple(static.get("formal_config", {}).get("fanouts", ())) == FANOUTS, "fan-out drift")
    return static


def run_shard(args: argparse.Namespace) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(0 <= args.rank < WORLD_SIZE, "rank must be in [0, 7]")
    static = _load_static(args)
    source = load_bound_json(
        args.source_manifest,
        args.expected_source_manifest_sha256,
        "source manifest",
    )
    require(source.get("schema_version") == SOURCE_SCHEMA, "source manifest schema drift")
    validate_source_manifest(args.source_root, source)
    require(
        sha256_file(args.model_artifact_ledger)
        == args.expected_model_artifact_ledger_sha256,
        "model artifact ledger raw SHA drift",
    )
    require(
        sha256_file(args.model_weight_ledger)
        == args.expected_model_weight_ledger_sha256,
        "model weight ledger raw SHA drift",
    )
    authority = load_bound_json(
        args.model_authority,
        args.expected_model_authority_sha256,
        "model authority",
    )
    authority_validation = validate_model_authority_receipt(
        args.model,
        authority,
        artifact_ledger_path=args.model_artifact_ledger,
        weight_ledger_path=args.model_weight_ledger,
        artifact_ledger_raw_sha256=args.expected_model_artifact_ledger_sha256,
        weight_ledger_raw_sha256=args.expected_model_weight_ledger_sha256,
    )
    model_artifact_receipt = authority["artifact_ledger"]
    model_weight_receipt = authority["weight_ledger"]
    require(
        model_artifact_receipt == static["model"]["artifact_ledger_receipt"]
        and model_weight_receipt == static["model"]["weight_ledger_receipt"],
        "model authority differs from frozen static ledger receipts",
    )
    gpu_assignment = load_bound_json(
        args.gpu_assignment, args.expected_gpu_assignment_sha256, "GPU assignment"
    )
    gpu_rows = validate_gpu_assignment(gpu_assignment)
    rank_input = static["rank_inputs"][args.rank]
    require(rank_input["rank"] == args.rank, "rank input ordering drift")

    torch.cuda.set_device(0)
    gpu = _gpu_identity(args.expected_gpu_uuid)
    require("H20" in gpu["name"], "formal hardware is not H20")
    require(gpu["compute_capability"] == [9, 0], "formal H20 compute capability drift")
    require(gpu["bf16_supported"] is True, "formal GPU lacks BF16 support")
    expected_gpu = {key: value for key, value in gpu_rows[args.rank].items() if key not in {"rank", "visible_index"}}
    require(gpu == {"cuda_visible_devices": gpu["uuid"], **expected_gpu}, "rank hardware does not equal frozen GPU assignment")
    import transformers
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    adapter = TorchSplitCausalLM(model)
    geometry = _model_geometry(adapter)
    require(geometry["matches_frozen"], "model geometry differs from the preregistration")
    config = static["formal_config"]
    depth = int(config["split_depth"])
    semantic_steps = int(config["semantic_steps"])
    threshold = float(static["oracle_contract"]["relative_l2_threshold"])
    salt = static["storage_receipt_salt"]
    document = torch.tensor(
        [rank_input["document_token_ids"]], dtype=torch.int64, device="cuda:0"
    )
    queries = [
        torch.tensor([row["token_ids"]], dtype=torch.int64, device="cuda:0")
        for row in rank_input["queries"]
    ]
    require(tuple(document.shape) == (1, config["document_tokens"]), "document shape drift")
    require(len(queries) == max(FANOUTS), "query count drift")
    require(
        tensor_receipt(document)["content_sha256"] == rank_input["document_token_ids_sha256"],
        "document token digest drift",
    )
    for query, frozen in zip(queries, rank_input["queries"]):
        require(tuple(query.shape) == (1, config["query_tokens"]), "query shape drift")
        require(
            tensor_receipt(query)["content_sha256"] == frozen["token_ids_sha256"],
            "query token digest drift",
        )

    with torch.inference_mode():
        logit_bundle = LogitBundleBuilder()
        oracle_rows = []
        oracle_logits = []
        for query in queries:
            row, logits = _dense_oracle(
                adapter,
                document,
                query,
                semantic_steps=semantic_steps,
                logit_bundle=logit_bundle,
                record_prefix=f"dense-oracle/request-{len(oracle_rows)}",
            )
            oracle_rows.append(row)
            oracle_logits.append(logits)

        persistent_base = adapter.write_lower_replay(document, depth)
        base_before = state_content_receipt(persistent_base)
        base_storage = storage_inventory(
            persistent_base.cache, salt=salt, role="persistent-document-lower-cache"
        )
        fanout_rows: dict[str, dict[str, Any]] = {}
        fanout_logits: dict[str, dict[str, list[list[torch.Tensor]]]] = {}
        for fanout in FANOUTS:
            materialized_states = [
                adapter.write_lower_replay(document, depth) for _ in range(fanout)
            ]
            persistent_states = [persistent_base.fork() for _ in range(fanout)]
            materialized_row, materialized_logits = _run_owned_arm(
                adapter,
                materialized_states,
                queries[:fanout],
                arm="deep_materialized",
                semantic_steps=semantic_steps,
                salt=salt,
                persistent_base=None,
                logit_bundle=logit_bundle,
                record_prefix=f"fanout-{fanout}/deep-materialized",
            )
            persistent_row, persistent_logits = _run_owned_arm(
                adapter,
                persistent_states,
                queries[:fanout],
                arm="persistent_fork",
                semantic_steps=semantic_steps,
                salt=salt,
                persistent_base=persistent_base,
                logit_bundle=logit_bundle,
                record_prefix=f"fanout-{fanout}/persistent-fork",
            )
            oracle_subset = oracle_rows[:fanout]
            oracle_logits_subset = oracle_logits[:fanout]
            comparisons = {
                "deep_materialized": _oracle_comparison_rows(
                    materialized_row["semantics"],
                    materialized_logits,
                    oracle_subset,
                    oracle_logits_subset,
                    threshold=threshold,
                ),
                "persistent_fork": _oracle_comparison_rows(
                    persistent_row["semantics"],
                    persistent_logits,
                    oracle_subset,
                    oracle_logits_subset,
                    threshold=threshold,
                ),
            }
            cross_arm_exact = _semantic_trace_key(materialized_row["semantics"]) == _semantic_trace_key(
                persistent_row["semantics"]
            )
            fanout_rows[str(fanout)] = {
                "fanout": fanout,
                "arms": {
                    "deep_materialized": materialized_row,
                    "persistent_fork": persistent_row,
                },
                "oracle_comparisons": comparisons,
                "cross_arm_exact": cross_arm_exact,
                "all_storage_ownership_predicates_passed": all(
                    arm["setup_disjointness"]["passed"]
                    and arm["first_transition_disjointness"]["passed"]
                    and arm["final_disjointness"]["passed"]
                    and arm["document_residual_ownership"]["passed"]
                    for arm in (materialized_row, persistent_row)
                ),
            }
            fanout_logits[str(fanout)] = {
                "deep_materialized": materialized_logits,
                "persistent_fork": persistent_logits,
            }
        base_after = state_content_receipt(persistent_base)

        cross_n = {}
        for arm in ("deep_materialized", "persistent_fork"):
            n1 = _semantic_trace_key(fanout_rows["1"]["arms"][arm]["semantics"])
            n2_prefix = _semantic_trace_key(fanout_rows["2"]["arms"][arm]["semantics"][:1])
            cross_n[arm] = {"passed": n1 == n2_prefix, "compared_requests": 1}

        fault_rows = _fault_suite(
            adapter,
            document,
            queries,
            persistent_base,
            fanout_rows["1"],
            fanout_logits["1"],
            oracle_rows,
            oracle_logits,
            depth=depth,
            semantic_steps=semantic_steps,
            threshold=threshold,
            salt=salt,
            logit_bundle=logit_bundle,
        )

    logit_sidecar = logit_bundle.write(args.logit_sidecar)

    oracle_clean = all(
        comparison["passed"]
        for fanout in fanout_rows.values()
        for comparisons in fanout["oracle_comparisons"].values()
        for comparison in comparisons
    )
    target_predicates = {
        "frozen_identity": True,
        "prefix_immutability": base_before["state_content_sha256"] == base_after["state_content_sha256"],
        "private_ownership": all(
            row["all_storage_ownership_predicates_passed"] for row in fanout_rows.values()
        ),
        "dispatch_provenance": True,
        "cross_arm_equivalence": all(row["cross_arm_exact"] for row in fanout_rows.values()),
        "cross_n_prefix_consistency": all(row["passed"] for row in cross_n.values()),
    }
    targets = build_target_rows(target_predicates)
    applicable_clean = all(
        row["predicate_passed"] is True
        for row in targets
        if row["applicability"] == "applicable"
    ) and oracle_clean
    dispatch = {
        "adapter": "qcomem_torch.TorchSplitCausalLM",
        "cache": "transformers.cache_utils.DynamicCache",
        "manual_suffix_method": "TorchSplitCausalLM.run_suffix_cached_last_logits",
        "layer_forward_types": sorted({f"{type(layer).__module__}.{type(layer).__qualname__}" for layer in adapter.layers}),
        "same_receipt_for_both_arms": True,
        "compiled_kernel_fingerprint": None,
        "autotuning_choice_fingerprint": None,
    }
    result = {
        "schema_version": SHARD_SCHEMA,
        "protocol": PROTOCOL,
        "status": "completed",
        "rank": args.rank,
        "world_size": WORLD_SIZE,
        "scientific_run_valid": True,
        "passed": applicable_clean
        and all(row["classification"]["outcome"] == row["expected_outcome"] for row in fault_rows)
        and all(row["fault_case_valid"] for row in fault_rows),
        "static_manifest_raw_sha256": args.expected_static_manifest_sha256,
        "source_manifest_raw_sha256": args.expected_source_manifest_sha256,
        "model_artifact_ledger_raw_sha256": args.expected_model_artifact_ledger_sha256,
        "model_weight_ledger_raw_sha256": args.expected_model_weight_ledger_sha256,
        "gpu_assignment_raw_sha256": args.expected_gpu_assignment_sha256,
        "formal_config_sha256": static["formal_config_sha256"],
        "model_identity": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "artifact_ledger": model_artifact_receipt,
            "weight_ledger": model_weight_receipt,
            "authority_raw_sha256": args.expected_model_authority_sha256,
            "authority_stat_validation": authority_validation,
        },
        "input": {
            "pg19_train_only": True,
            "source_object": rank_input["source_object"],
            "source_id": rank_input["source_id"],
            "document_start_token": rank_input["document_start_token"],
            "document_token_ids_sha256": rank_input["document_token_ids_sha256"],
            "query_token_ids_sha256": [row["token_ids_sha256"] for row in rank_input["queries"]],
            "rank_input_sha256": rank_input["rank_input_sha256"],
        },
        "hardware": gpu,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
            "model_geometry": geometry,
        },
        "dispatch_provenance": dispatch,
        "persistent_base": {
            "before": base_before,
            "after": base_after,
            "storage": base_storage,
            "content_immutable": base_before["state_content_sha256"] == base_after["state_content_sha256"],
        },
        "dense_oracle": {
            "contract": static["oracle_contract"],
            "semantics": oracle_rows,
            "all_clean_arms_passed": oracle_clean,
        },
        "fanouts": fanout_rows,
        "cross_n": cross_n,
        "targets": targets,
        "clean_audit": {
            "all_applicable_predicates_passed": applicable_clean,
            "independent_dense_oracle_passed": oracle_clean,
            "target_status_vector": [row["status"] for row in targets],
        },
        "fault_suite": fault_rows,
        "claim_boundary": static["claim_boundary"],
        "logit_sidecar": logit_sidecar,
    }
    del model
    torch.cuda.synchronize()
    return result


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    static = _load_static(args)
    source = load_bound_json(
        args.source_manifest,
        args.expected_source_manifest_sha256,
        "source manifest",
    )
    validate_source_manifest(args.source_root, source)
    authority_raw = args.model_authority.read_bytes()
    closure_raw = args.model_closure.read_bytes()
    require(authority_raw == closure_raw, "terminal model closure differs from pre-output authority")
    authority_sha256 = sha256_bytes(authority_raw)
    require(authority_sha256 == args.expected_model_authority_sha256, "model authority raw SHA drift")
    authority = json.loads(authority_raw)
    validate_model_authority_receipt(
        args.model,
        authority,
        artifact_ledger_path=args.model_artifact_ledger,
        weight_ledger_path=args.model_weight_ledger,
        artifact_ledger_raw_sha256=args.expected_model_artifact_ledger_sha256,
        weight_ledger_raw_sha256=args.expected_model_weight_ledger_sha256,
    )
    gpu_assignment = load_bound_json(
        args.gpu_assignment, args.expected_gpu_assignment_sha256, "GPU assignment"
    )
    validate_gpu_assignment(gpu_assignment)
    paths = list(args.shard_dir.glob("forkaudit-transformers-transfer-shard-*.json"))
    return aggregate_shards(
        paths,
        static_manifest=static,
        sidecar_dir=args.sidecar_dir,
        static_manifest_raw_sha256=args.expected_static_manifest_sha256,
        source_manifest_raw_sha256=args.expected_source_manifest_sha256,
        model_authority_raw_sha256=authority_sha256,
        gpu_assignment=gpu_assignment,
        gpu_assignment_raw_sha256=args.expected_gpu_assignment_sha256,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("shard", "aggregate"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-artifact-ledger", type=Path)
    parser.add_argument("--model-weight-ledger", type=Path)
    parser.add_argument("--model-authority", type=Path)
    parser.add_argument("--model-closure", type=Path)
    parser.add_argument("--expected-model-authority-sha256", default="")
    parser.add_argument("--gpu-assignment", type=Path)
    parser.add_argument("--expected-gpu-assignment-sha256", default="")
    parser.add_argument("--static-manifest", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-static-manifest-sha256", required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--expected-model-artifact-ledger-sha256", required=True)
    parser.add_argument("--expected-model-weight-ledger-sha256", required=True)
    parser.add_argument("--rank", type=int, default=-1)
    parser.add_argument("--expected-gpu-uuid", default="")
    parser.add_argument("--shard-dir", type=Path)
    parser.add_argument("--sidecar-dir", type=Path)
    parser.add_argument("--logit-sidecar", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    for label in (
        "expected_static_manifest_sha256",
        "expected_source_manifest_sha256",
        "expected_model_artifact_ledger_sha256",
        "expected_model_weight_ledger_sha256",
        "expected_model_authority_sha256",
        "expected_gpu_assignment_sha256",
    ):
        value = getattr(args, label)
        require(len(value) == 64 and all(character in "0123456789abcdef" for character in value), f"{label} is invalid")
    if args.stage == "shard":
        require(args.model is not None, "--model is required for shards")
        require(args.model_artifact_ledger is not None, "--model-artifact-ledger is required for shards")
        require(args.model_weight_ledger is not None, "--model-weight-ledger is required for shards")
        require(args.model_authority is not None, "--model-authority is required for shards")
        require(args.gpu_assignment is not None, "--gpu-assignment is required for shards")
        require(args.logit_sidecar is not None, "--logit-sidecar is required for shards")
        value = run_shard(args)
    else:
        require(args.shard_dir is not None, "--shard-dir is required for aggregate")
        require(args.sidecar_dir is not None, "--sidecar-dir is required for aggregate")
        require(args.model is not None, "--model is required for aggregate")
        require(args.model_authority is not None, "--model-authority is required for aggregate")
        require(args.model_closure is not None, "--model-closure is required for aggregate")
        require(args.gpu_assignment is not None, "--gpu-assignment is required for aggregate")
        require(args.model_artifact_ledger is not None, "--model-artifact-ledger is required for aggregate")
        require(args.model_weight_ledger is not None, "--model-weight-ledger is required for aggregate")
        value = aggregate(args)
    write_canonical_json(args.output, value)
    print(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
