#!/usr/bin/env python3
"""R40 compact post-return dispatch receipts for the RR2 primary factorial.

The runtime hooks are inherited from the independently tested R40
compiled-dispatch recorder.  This module changes only serialization and adds a
strict primary-cell context.  It hashes each selected Triton bundle once and
stores integer table references for every intercepted call; this avoids
repeating a complete cubin/PTX manifest hundreds of thousands of times.
"""

from __future__ import annotations

import contextlib
import contextvars
import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import r39_compiled_dispatch_receipts as base


SCHEMA_VERSION = "forkaudit-r40-primary-compiled-dispatch-receipt-v7"
AGGREGATE_SCHEMA_VERSION = "forkaudit-r40-primary-compiled-dispatch-aggregate-v7"
PRIMARY_PROTOCOL = "qcomem-qwen35-forkaudit-review-revision-v1"
PRIMARY_SHARD_SCHEMA = "qcomem-forkaudit-review-shard-v1"
PRIMARY_AGGREGATE_SCHEMA = "qcomem-forkaudit-review-aggregate-v1"
PRIMARY_WORLD_SIZE = 8
PRIMARY_RESIDENT_COUNTS = (1, 8, 32)
PRIMARY_GENERATION_STEPS = 8
PRIMARY_DOCUMENT_TOKENS = 4095
PRIMARY_FULL_LAYERS = tuple(range(3, 40, 4))
PRIMARY_LINEAR_LAYERS = tuple(
    index for index in range(40) if index not in PRIMARY_FULL_LAYERS
)
PRIMARY_KV_POLICIES = (
    "vllm-q16-fresh-full-copy-control",
    "vllm-q16-shared-document-reuse",
)
PRIMARY_GDN_POLICIES = (
    "materialize-request-base-functional-rebind",
    "borrow-immutable-base-functional-rebind",
)
PRIMARY_ARMS = tuple(
    f"kv={kv}|gdn={gdn}"
    for kv in PRIMARY_KV_POLICIES
    for gdn in PRIMARY_GDN_POLICIES
)
PRIMARY_ROLES = ("formal_memory", "ownership_witness")
PRIMARY_RUNNER_RELATIVE_PATH = "run_qcomem_qwen35_forkaudit_review_revision.py"
PRIMARY_RUNNER_SHA256 = "9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775"
PRIMARY_LAUNCHER_RELATIVE_PATH = "launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh"
PRIMARY_LAUNCHER_SHA256 = "077a876b9849661135044c50cfdea272d302a48af0bb4e21ec640eca2ca85460"
PRIMARY_CODE_LEDGER_SHA256 = "837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a"
PRIMARY_MODEL_ARTIFACT_LEDGER_SHA256 = "c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb"
PRIMARY_MODEL_WEIGHT_LEDGER_SHA256 = "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014"
PRIMARY_PROTOCOL_MANIFEST_SHA256 = "975bc6a12f43447024b889889d4156ca71c2f89b68de6157ac609b4a9687e9c0"
PRIMARY_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
PRIMARY_MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
GDN_SCOPE = (
    "actual Transformers Qwen3.5-MoE native eager GDN plus qcomem "
    "functional cache rebind; underlying ATen/CUDA libraries are out of scope"
)
ATTENTION_COLUMNS = [
    "cell_index",
    "local_call_index",
    "call_id",
    "shape_table_index",
    "artifact_table_index",
    "config_table_index",
    "autotune_table_index",
    "cuda_visible_devices",
    "torch_device_index",
    "torch_stream_id",
    "post_launcher_returned",
    "post_return_context_matches",
    "call_receipt_sha256",
]
GDN_COLUMNS = [
    "cell_index",
    "local_call_index",
    "layer_idx",
    "sequence_length",
    "cache_has_previous_state",
    "chunk_rule_calls",
    "recurrent_rule_calls",
    "functional_conv_rebind_calls",
    "inplace_conv_update_calls",
    "recurrent_rebind_calls",
]
class PrimaryDispatchError(base.DispatchReceiptError):
    """A primary-factorial dispatch receipt is absent, ambiguous, or invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PrimaryDispatchError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    return value


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _shape_record(shape: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(shape))
    return {
        "shape": value,
        "shape_sha256": base._sha256_bytes(base._canonical_bytes(value)),
    }


@dataclass(frozen=True)
class Geometry:
    resident_counts: tuple[int, ...]
    arms: tuple[str, ...]
    roles: tuple[str, ...]
    generation_steps: int
    document_tokens: int
    full_layers: tuple[int, ...]
    linear_layers: tuple[int, ...]


PRIMARY_GEOMETRY = Geometry(
    resident_counts=PRIMARY_RESIDENT_COUNTS,
    arms=PRIMARY_ARMS,
    roles=PRIMARY_ROLES,
    generation_steps=PRIMARY_GENERATION_STEPS,
    document_tokens=PRIMARY_DOCUMENT_TOKENS,
    full_layers=PRIMARY_FULL_LAYERS,
    linear_layers=PRIMARY_LINEAR_LAYERS,
)


def expected_cells(geometry: Geometry = PRIMARY_GEOMETRY) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for resident_count in geometry.resident_counts:
        for arm_id in geometry.arms:
            arm_parts = arm_id.split("|", 1)
            _require(
                len(arm_parts) == 2
                and arm_parts[0].startswith("kv=")
                and arm_parts[1].startswith("gdn="),
                "geometry arm spelling is invalid",
            )
            for role in geometry.roles:
                rows.append(
                    {
                        "resident_count": resident_count,
                        "arm_id": arm_id,
                        "kv_policy": arm_parts[0][3:],
                        "gdn_base_policy": arm_parts[1][4:],
                        "cell_role": role,
                    }
                )
    return rows


def expected_rank_counts(geometry: Geometry = PRIMARY_GEOMETRY) -> dict[str, int]:
    cell_count = len(geometry.resident_counts) * len(geometry.arms) * len(geometry.roles)
    resident_sum = sum(geometry.resident_counts) * len(geometry.arms) * len(geometry.roles)
    attention = resident_sum * geometry.generation_steps * len(geometry.full_layers)
    gdn_prefill = cell_count * len(geometry.linear_layers)
    gdn_request = resident_sum * geometry.generation_steps * len(geometry.linear_layers)
    return {
        "cell_count": cell_count,
        "attention_call_count": attention,
        "gdn_document_prefill_call_count": gdn_prefill,
        "gdn_request_call_count": gdn_request,
        "gdn_call_count": gdn_prefill + gdn_request,
    }


@dataclass
class _CompactAttention:
    call_id: str
    call_shape: dict[str, Any]
    launches: list[dict[str, Any]] = field(default_factory=list)
    pending_launches: list[dict[str, Any]] = field(default_factory=list)
    autotune_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _CellState:
    cell_index: int
    metadata: dict[str, Any]
    attention_start: int
    gdn_start: int


class PrimaryDispatchRecorder(base.DispatchReceiptRecorder):
    """R40 hooks with primary-only contexts and compact per-call rows."""

    def __init__(
        self,
        *,
        cache_root: Path,
        code_root: Path,
        runtime_root: Path,
        launch_context_provider: Any | None = None,
    ) -> None:
        super().__init__(
            cache_root=cache_root,
            code_root=code_root,
            runtime_root=runtime_root,
            launch_context_provider=launch_context_provider,
        )
        # The verbose parent lists remain empty by design.
        self.compact_attention_calls: list[list[Any]] = []
        self.compact_gdn_calls: list[list[Any]] = []
        self.cells: list[dict[str, Any]] = []
        self.artifact_table: list[dict[str, Any]] = []
        self.config_table: list[dict[str, Any]] = []
        self.shape_table: list[dict[str, Any]] = []
        self.autotune_table: list[dict[str, Any]] = []
        self._artifact_index: dict[str, int] = {}
        self._config_index: dict[str, int] = {}
        self._shape_index: dict[str, int] = {}
        self._autotune_index: dict[str, int] = {}
        self._compiled_metadata_cache: dict[
            str, tuple[dict[str, Any], dict[str, Any]]
        ] = {}
        self._active_cell: contextvars.ContextVar[_CellState | None] = (
            contextvars.ContextVar("forkaudit_r40_primary_cell", default=None)
        )
        self._factorial_rank: int | None = None
        self._factorial_finished = False

    @staticmethod
    def _intern(
        value: Mapping[str, Any], table: list[dict[str, Any]], index: dict[str, int]
    ) -> int:
        candidate = dict(value)
        key = _canonical(candidate)
        if key not in index:
            index[key] = len(table)
            table.append(candidate)
        return index[key]

    def begin_factorial(self, rank: int) -> None:
        _require(self._factorial_rank is None, "primary factorial entered twice")
        _require(type(rank) is int and 0 <= rank < PRIMARY_WORLD_SIZE, "rank invalid")
        self._factorial_rank = rank

    def finish_factorial(self) -> None:
        _require(self._factorial_rank is not None, "primary factorial was not entered")
        _require(self._active_cell.get() is None, "primary cell remained active")
        _require(
            len(self.cells) == expected_rank_counts()["cell_count"],
            "primary factorial did not execute exactly 24 memory/witness cells",
        )
        self._factorial_finished = True

    @contextlib.contextmanager
    def primary_cell(
        self,
        *,
        rank: int,
        resident_count: int,
        arm_id: str,
        kv_policy: str,
        gdn_base_policy: str,
        cell_role: str,
    ) -> Iterator[None]:
        _require(self._factorial_rank == rank, "cell/rank factorial binding drift")
        _require(not self._factorial_finished, "cell began after factorial closure")
        _require(self._active_cell.get() is None, "nested primary cells are invalid")
        expected = expected_cells()[len(self.cells)]
        observed = {
            "resident_count": resident_count,
            "arm_id": arm_id,
            "kv_policy": kv_policy,
            "gdn_base_policy": gdn_base_policy,
            "cell_role": cell_role,
        }
        _require(observed == expected, f"primary cell order/geometry drift: {observed}")
        state = _CellState(
            cell_index=len(self.cells),
            metadata=observed,
            attention_start=len(self.compact_attention_calls),
            gdn_start=len(self.compact_gdn_calls),
        )
        token = self._active_cell.set(state)
        try:
            yield
        finally:
            active = self._active_cell.get()
            _require(active is state, "primary cell context disappeared")
            self._active_cell.reset(token)
        attention_end = len(self.compact_attention_calls)
        gdn_end = len(self.compact_gdn_calls)
        expected_attention = (
            resident_count * PRIMARY_GENERATION_STEPS * len(PRIMARY_FULL_LAYERS)
        )
        expected_gdn_prefill = len(PRIMARY_LINEAR_LAYERS)
        expected_gdn_request = (
            resident_count * PRIMARY_GENERATION_STEPS * len(PRIMARY_LINEAR_LAYERS)
        )
        _require(
            attention_end - state.attention_start == expected_attention,
            f"primary cell attention count drift for {observed}",
        )
        _require(
            gdn_end - state.gdn_start == expected_gdn_prefill + expected_gdn_request,
            f"primary cell GDN count drift for {observed}",
        )
        self.cells.append(
            {
                "cell_index": state.cell_index,
                "rank": rank,
                **observed,
                "attention_call_range": [state.attention_start, attention_end],
                "gdn_call_range": [state.gdn_start, gdn_end],
                "expected_attention_calls": expected_attention,
                "expected_gdn_document_prefill_calls": expected_gdn_prefill,
                "expected_gdn_request_calls": expected_gdn_request,
            }
        )

    # The base hook calls these methods for the entire shard.  Calls outside a
    # registered primary memory/witness cell are deliberately ignored.
    def begin_attention(self, kwargs: Mapping[str, Any]) -> contextvars.Token[Any]:
        state = self._active_cell.get()
        if state is None:
            return self._active_attention.set(None)
        _require(self._active_attention.get() is None, "nested attention is unsupported")
        _require(self._factorial_rank is not None, "attention began outside factorial")
        local_index = len(self.compact_attention_calls) - state.attention_start
        context = _CompactAttention(
            call_id=(
                f"rank-{self._factorial_rank}/cell-{state.cell_index}/"
                f"attention-{local_index}"
            ),
            call_shape=base._call_shape(kwargs),
        )
        return self._active_attention.set(context)  # type: ignore[arg-type]

    def prepare_compiled_kernel(self, kernel: Any) -> dict[str, Any] | None:
        context = self._active_attention.get()
        if context is None:
            return None
        _require(isinstance(context, _CompactAttention), "attention context type drift")
        _require(not context.pending_launches, "a compiled launcher is already pending")
        _require(not context.launches, "attention attempted a duplicate compiled launch")
        metadata = base._compiled_metadata(kernel)
        normalised = {
            "name": metadata.get("name"),
            "hash": metadata.get("hash"),
            **base._normalise_config(metadata),
        }
        cache_key = _canonical(normalised)
        cached = self._compiled_metadata_cache.get(cache_key)
        if cached is None:
            candidates = base._candidate_artifacts(self.cache_root, metadata)
            _require(
                len(candidates) == 1,
                f"compiled kernel must bind exactly one cache artifact, found {len(candidates)}",
            )
            artifact = candidates[0].as_dict()
            cached = (copy.deepcopy(artifact), copy.deepcopy(normalised))
            self._compiled_metadata_cache[cache_key] = cached
        artifact, normalised = copy.deepcopy(cached)
        _require(
            artifact.get("kernel_name") == base.EXPECTED_TRITON_KERNEL_NAME,
            "intercepted compiled launcher is not vLLM unified attention",
        )
        launch_context = dict(self.launch_context_provider())
        base._verify_launch_context(launch_context, label="pre-launch context")
        pending = {
            "call_id": context.call_id,
            "artifact": artifact,
            "compiled_metadata": normalised,
            "launch_context": launch_context,
        }
        context.pending_launches.append(pending)
        return pending

    def seal_compiled_kernel(self, pending: dict[str, Any] | None) -> None:
        if pending is None:
            return
        context = self._active_attention.get()
        _require(isinstance(context, _CompactAttention), "attention context disappeared")
        _require(context.pending_launches == [pending], "compiled launch pending identity/order drift")
        post = dict(self.launch_context_provider())
        base._verify_launch_context(post, label="post-return launch context")
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
        if isinstance(context, _CompactAttention) and context.pending_launches == [pending]:
            context.pending_launches.clear()

    def record_autotune(self, autotuner: Any) -> None:
        context = self._active_attention.get()
        if context is None:
            return
        _require(isinstance(context, _CompactAttention), "attention context type drift")
        best = getattr(autotuner, "best_config", None)
        if best is None:
            return
        _require(not context.autotune_events, "attention observed duplicate autotune selections")
        context.autotune_events.append(base._autotune_event(best))

    def abort_attention(self, token: contextvars.Token[Any]) -> None:
        self._active_attention.reset(token)

    def finish_attention(self, token: contextvars.Token[Any]) -> None:
        context = self._active_attention.get()
        self._active_attention.reset(token)
        if context is None:
            return
        _require(isinstance(context, _CompactAttention), "attention context type drift")
        _require(not context.pending_launches, "attention ended with an unreturned launcher")
        _require(len(context.launches) == 1, "attention must launch exactly one kernel")
        state = self._active_cell.get()
        _require(state is not None, "primary cell disappeared during attention")
        launch = context.launches[0]
        artifact = launch["artifact"]
        normalised = launch["compiled_metadata"]
        artifact_id = str(artifact["artifact_id"])
        if artifact_id not in self._artifact_index:
            self._artifact_index[artifact_id] = len(self.artifact_table)
            self.artifact_table.append(copy.deepcopy(artifact))
        artifact_index = self._artifact_index[artifact_id]
        config_index = self._intern(normalised, self.config_table, self._config_index)
        autotune = (
            {"mode": "triton-autotuner", "events": context.autotune_events}
            if context.autotune_events
            else {"mode": "no-autotuner-observed"}
        )
        shape_index = self._intern(
            _shape_record(context.call_shape), self.shape_table, self._shape_index
        )
        autotune_index = self._intern(
            autotune, self.autotune_table, self._autotune_index
        )
        local_index = len(self.compact_attention_calls) - state.attention_start
        _require(
            context.call_id
            == f"rank-{self._factorial_rank}/cell-{state.cell_index}/attention-{local_index}",
            "primary attention call ID drift",
        )
        launch_context = launch["launch_context"]
        receipt_core = {
            "call_id": context.call_id,
            "call_shape": context.call_shape,
            "artifact_id": artifact_id,
            "selected_compile_config": normalised,
            "autotune": autotune,
            "launch_context": launch_context,
            "post_launcher_returned": launch["post_launcher_returned"],
            "post_return_context_matches": launch["post_return_context_matches"],
        }
        self.compact_attention_calls.append(
            [
                state.cell_index,
                local_index,
                context.call_id,
                shape_index,
                artifact_index,
                config_index,
                autotune_index,
                launch_context["cuda_visible_devices"],
                launch_context["torch_device_index"],
                launch_context["torch_stream_id"],
                True,
                True,
                base._sha256_bytes(base._canonical_bytes(receipt_core)),
            ]
        )

    def begin_gdn(
        self,
        *,
        layer_idx: int,
        sequence_length: int,
        cache_has_previous_state: bool,
    ) -> contextvars.Token[Any]:
        if self._active_cell.get() is None:
            return self._active_gdn.set(None)
        _require(self._active_gdn.get() is None, "nested GDN forward is unsupported")
        _require(type(layer_idx) is int and layer_idx >= 0, "GDN layer invalid")
        _require(type(sequence_length) is int and sequence_length > 0, "GDN length invalid")
        context = base._GDNContext(
            call_index=len(self.compact_gdn_calls),
            layer_idx=layer_idx,
            sequence_length=sequence_length,
            cache_has_previous_state=cache_has_previous_state,
            execution_phase=(
                "request-cell" if cache_has_previous_state else "document-prefill"
            ),
        )
        return self._active_gdn.set(context)

    def abort_gdn(self, token: contextvars.Token[Any]) -> None:
        self._active_gdn.reset(token)

    def record_gdn_event(self, event: str) -> None:
        context = self._active_gdn.get()
        if context is None:
            return
        super().record_gdn_event(event)

    def finish_gdn(self, token: contextvars.Token[Any]) -> None:
        context = self._active_gdn.get()
        self._active_gdn.reset(token)
        if context is None:
            return
        base._validate_gdn_route_counts(context)
        state = self._active_cell.get()
        _require(state is not None, "primary cell disappeared during GDN")
        local_index = len(self.compact_gdn_calls) - state.gdn_start
        self.compact_gdn_calls.append(
            [
                state.cell_index,
                local_index,
                context.layer_idx,
                context.sequence_length,
                context.cache_has_previous_state,
                context.chunk_kernel_events,
                context.recurrent_kernel_events,
                context.conv_rebind_events,
                context.inplace_conv_update_events,
                context.recurrent_rebind_events,
            ]
        )

    def payload(self) -> dict[str, Any]:
        _require(self._factorial_finished, "primary factorial did not close")
        counts = expected_rank_counts()
        _require(len(self.compact_attention_calls) == counts["attention_call_count"], "rank attention total drift")
        _require(len(self.compact_gdn_calls) == counts["gdn_call_count"], "rank GDN total drift")
        _require(
            set(self.dispatch_source_bindings) == base.DISPATCH_SOURCE_KEYS,
            "dispatch source bindings incomplete",
        )
        _require(self.hook_installation, "hook-installation receipt absent")
        return {
            "schema_version": SCHEMA_VERSION,
            "rank": self._factorial_rank,
            "scope": {
                "primary_protocol": PRIMARY_PROTOCOL,
                "primary_cells_only": True,
                "vllm_attention": base.TARGET_VLLM_ENTRYPOINT,
                "gdn": GDN_SCOPE,
                "trusted_runtime_boundary": (
                    "honest process with trusted pinned PyTorch/CUDA, vLLM, and Triton; "
                    "post-return launcher receipt is not malicious-runtime or device attestation"
                ),
            },
            "hook_installation": copy.deepcopy(self.hook_installation),
            "dispatch_source_bindings": copy.deepcopy(self.dispatch_source_bindings),
            "geometry": {
                "world_size": PRIMARY_WORLD_SIZE,
                "resident_counts": list(PRIMARY_RESIDENT_COUNTS),
                "generation_steps": PRIMARY_GENERATION_STEPS,
                "document_tokens": PRIMARY_DOCUMENT_TOKENS,
                "full_layer_indices": list(PRIMARY_FULL_LAYERS),
                "linear_layer_indices": list(PRIMARY_LINEAR_LAYERS),
                "arm_ids": list(PRIMARY_ARMS),
                "cell_roles": list(PRIMARY_ROLES),
                **counts,
            },
            "tables": {
                "compiled_artifacts": copy.deepcopy(self.artifact_table),
                "selected_compile_configurations": copy.deepcopy(self.config_table),
                "call_shapes": copy.deepcopy(self.shape_table),
                "autotune_observations": copy.deepcopy(self.autotune_table),
            },
            "cells": copy.deepcopy(self.cells),
            "attention_call_columns": list(ATTENTION_COLUMNS),
            "attention_calls": copy.deepcopy(self.compact_attention_calls),
            "gdn_call_columns": list(GDN_COLUMNS),
            "gdn_calls": copy.deepcopy(self.compact_gdn_calls),
        }


def _verify_source_bindings(
    payload: Mapping[str, Any],
    *,
    code_root: Path,
    runtime_root: Path,
    expected_bindings: Mapping[str, Mapping[str, str]],
) -> None:
    observed = base._verify_dispatch_source_bindings(
        payload, code_root=code_root, runtime_root=runtime_root
    )
    _require(set(observed) == set(expected_bindings), "dispatch binding key set drift")
    identities: list[str] = []
    for key, expected in expected_bindings.items():
        binding = _mapping(observed.get(key), f"dispatch binding {key}")
        _require(
            dict(binding) == dict(expected),
            f"dispatch binding identity drift for {key}",
        )
        identities.append(_canonical(binding))
    _require(
        len(set(identities)) == len(identities),
        "two dispatch binding keys resolve to the same callable identity",
    )


def _verify_geometry(raw: Any, geometry: Geometry) -> Mapping[str, Any]:
    value = _mapping(raw, "receipt geometry")
    counts = expected_rank_counts(geometry)
    expected = {
        "world_size": PRIMARY_WORLD_SIZE,
        "resident_counts": list(geometry.resident_counts),
        "generation_steps": geometry.generation_steps,
        "document_tokens": geometry.document_tokens,
        "full_layer_indices": list(geometry.full_layers),
        "linear_layer_indices": list(geometry.linear_layers),
        "arm_ids": list(geometry.arms),
        "cell_roles": list(geometry.roles),
        **counts,
    }
    _require(dict(value) == expected, "receipt geometry/preregistered counts drift")
    return value


def _argv_option(argv: Sequence[str], name: str) -> str:
    matches = [index for index, value in enumerate(argv) if value == name]
    _require(len(matches) == 1 and matches[0] + 1 < len(argv), f"runner argv {name} binding drift")
    return argv[matches[0] + 1]


def verify_gpu_assignment_receipt(value: Any) -> dict[str, Any]:
    receipt = _mapping(value, "GPU assignment receipt")
    _require(
        set(receipt)
        == {
            "schema_version",
            "world_size",
            "inventory_query",
            "rows",
            "unique_visible_indices",
            "unique_uuids",
            "all_h20",
            "all_compute_capability_9_0",
            "generated_before_candidate_outputs",
        },
        "GPU assignment receipt fields drift",
    )
    _require(
        receipt.get("schema_version")
        == "qcomem-forkaudit-gpu-assignment-receipt-v1"
        and receipt.get("world_size") == PRIMARY_WORLD_SIZE
        and receipt.get("inventory_query")
        == "index,uuid,name,memory.total,compute_cap"
        and receipt.get("unique_visible_indices") is True
        and receipt.get("unique_uuids") is True
        and receipt.get("all_h20") is True
        and receipt.get("all_compute_capability_9_0") is True
        and receipt.get("generated_before_candidate_outputs") is True,
        "GPU assignment receipt contract drift",
    )
    rows = receipt.get("rows")
    _require(isinstance(rows, list) and len(rows) == PRIMARY_WORLD_SIZE, "GPU assignment rows drift")
    expected_fields = {
        "rank",
        "visible_index",
        "uuid",
        "name",
        "total_memory_mib",
        "compute_capability",
        "bf16_supported",
    }
    for rank, row_raw in enumerate(rows):
        row = _mapping(row_raw, f"GPU assignment row {rank}")
        _require(set(row) == expected_fields, f"GPU assignment row {rank} fields drift")
        _require(
            row.get("rank") == rank
            and type(row.get("visible_index")) is int
            and row["visible_index"] >= 0
            and isinstance(row.get("uuid"), str)
            and row["uuid"].startswith("GPU-")
            and isinstance(row.get("name"), str)
            and "H20" in row["name"]
            and type(row.get("total_memory_mib")) is int
            and row["total_memory_mib"] > 0
            and row.get("compute_capability") == [9, 0]
            and row.get("bf16_supported") is True,
            f"GPU assignment row {rank} contract drift",
        )
    _require(
        len({row["visible_index"] for row in rows}) == PRIMARY_WORLD_SIZE
        and len({row["uuid"] for row in rows}) == PRIMARY_WORLD_SIZE,
        "GPU assignment is not one-to-one",
    )
    return dict(receipt)


def _verify_rank_identity(
    payload: Mapping[str, Any],
    *,
    rank: int,
    expected_gpu_assignment_receipt: Mapping[str, Any],
    expected_gpu_assignment_raw_sha256: str,
    expected_launcher_identity: Mapping[str, Any],
    expected_launcher_identity_raw_sha256: str,
) -> Mapping[str, Any]:
    assignment = verify_gpu_assignment_receipt(expected_gpu_assignment_receipt)
    _require(base._is_sha256(expected_gpu_assignment_raw_sha256), "GPU assignment raw SHA invalid")
    identity = _mapping(payload.get("rank_identity"), "rank identity")
    _require(
        set(identity)
        == {
            "schema_version",
            "rank",
            "process_id",
            "parent_process_id",
            "cuda_visible_devices",
            "assigned_gpu_uuid",
            "gpu_assignment_receipt_path",
            "gpu_assignment_receipt_raw_sha256",
            "gpu_assignment_row",
            "launcher_identity_path",
            "launcher_identity_raw_sha256",
            "launcher_identity",
        },
        "rank-identity fields drift",
    )
    expected_row = assignment["rows"][rank]
    launcher_identity = _mapping(expected_launcher_identity, "external launcher identity")
    _require(
        set(launcher_identity)
        == {
            "cuda_visible_devices",
            "parent_process_id",
            "process_id",
            "rank",
            "runner",
            "schema_version",
        },
        "external launcher-identity fields drift",
    )
    _require(
        launcher_identity.get("schema_version")
        == "forkaudit-r40-proxy-rank-launch-v1"
        and launcher_identity.get("rank") == rank
        and launcher_identity.get("cuda_visible_devices") == expected_row["uuid"]
        and launcher_identity.get("runner") == PRIMARY_RUNNER_RELATIVE_PATH
        and type(launcher_identity.get("process_id")) is int
        and launcher_identity["process_id"] > 1
        and type(launcher_identity.get("parent_process_id")) is int
        and launcher_identity["parent_process_id"] > 0
        and base._is_sha256(expected_launcher_identity_raw_sha256),
        "external launcher identity contract drift",
    )
    _require(
        identity.get("schema_version") == "forkaudit-r40-rank-launch-identity-v1"
        and identity.get("rank") == rank
        and identity.get("process_id") == launcher_identity["process_id"]
        and identity.get("parent_process_id") == launcher_identity["parent_process_id"]
        and identity.get("cuda_visible_devices") == expected_row["uuid"]
        and identity.get("assigned_gpu_uuid") == expected_row["uuid"]
        and isinstance(identity.get("gpu_assignment_receipt_path"), str)
        and identity["gpu_assignment_receipt_path"].startswith("/")
        and identity.get("gpu_assignment_receipt_raw_sha256")
        == expected_gpu_assignment_raw_sha256
        and identity.get("gpu_assignment_row") == expected_row
        and isinstance(identity.get("launcher_identity_path"), str)
        and identity["launcher_identity_path"].startswith("/")
        and identity.get("launcher_identity_raw_sha256")
        == expected_launcher_identity_raw_sha256
        and identity.get("launcher_identity") == launcher_identity,
        "rank identity does not match external launcher assignment",
    )
    return identity


def _verify_execution_binding(
    payload: Mapping[str, Any],
    *,
    rank: int,
    expected_runtime_preflight_sha256: str,
) -> Mapping[str, Any]:
    binding = _mapping(payload.get("execution_binding"), "execution binding")
    _require(
        set(binding)
        == {
            "runner_relative_path",
            "runner_sha256",
            "runner_argv",
            "runner_argv_sha256",
            "primary_shard_path",
            "primary_shard_sha256",
            "launcher_relative_path",
            "launcher_sha256",
            "code_ledger_path",
            "code_ledger_sha256",
            "model_artifact_ledger_path",
            "model_artifact_ledger_sha256",
            "model_weight_ledger_path",
            "model_weight_ledger_sha256",
            "protocol_manifest_path",
            "protocol_manifest_sha256",
            "model_id",
            "model_revision",
            "runtime_preflight_manifest_path",
            "runtime_preflight_manifest_sha256",
        },
        "execution-binding fields drift",
    )
    _require(binding.get("runner_relative_path") == PRIMARY_RUNNER_RELATIVE_PATH, "runner relative path drift")
    _require(binding.get("runner_sha256") == PRIMARY_RUNNER_SHA256, "runner source hash drift")
    argv = binding.get("runner_argv")
    _require(isinstance(argv, list) and all(isinstance(item, str) for item in argv), "runner argv malformed")
    _require(
        binding.get("runner_argv_sha256")
        == base._sha256_bytes(_canonical(argv).encode("utf-8")),
        "runner argv hash drift",
    )
    _require(_argv_option(argv, "--stage") == "shard", "runner stage is not shard")
    _require(_argv_option(argv, "--rank") == str(rank), "runner argv rank drift")
    output = _argv_option(argv, "--output")
    _require(output == binding.get("primary_shard_path"), "runner argv/output binding drift")
    _require(base._is_sha256(binding.get("primary_shard_sha256")), "primary shard SHA-256 malformed")
    expected_pairs = {
        "launcher_relative_path": PRIMARY_LAUNCHER_RELATIVE_PATH,
        "launcher_sha256": PRIMARY_LAUNCHER_SHA256,
        "code_ledger_sha256": PRIMARY_CODE_LEDGER_SHA256,
        "model_artifact_ledger_sha256": PRIMARY_MODEL_ARTIFACT_LEDGER_SHA256,
        "model_weight_ledger_sha256": PRIMARY_MODEL_WEIGHT_LEDGER_SHA256,
        "protocol_manifest_sha256": PRIMARY_PROTOCOL_MANIFEST_SHA256,
        "model_id": PRIMARY_MODEL_ID,
        "model_revision": PRIMARY_MODEL_REVISION,
        "runtime_preflight_manifest_sha256": expected_runtime_preflight_sha256,
    }
    for field, expected in expected_pairs.items():
        _require(binding.get(field) == expected, f"execution binding {field} drift")
    for option, path_field in (
        ("--code-ledger", "code_ledger_path"),
        ("--model-artifact-ledger", "model_artifact_ledger_path"),
        ("--model-weight-ledger", "model_weight_ledger_path"),
        ("--protocol-manifest", "protocol_manifest_path"),
    ):
        _require(
            _argv_option(argv, option) == binding.get(path_field),
            f"runner argv/{path_field} binding drift",
        )
    _require(
        isinstance(binding.get("runtime_preflight_manifest_path"), str)
        and binding["runtime_preflight_manifest_path"].startswith("/"),
        "runtime preflight manifest path invalid",
    )
    return binding


def verify_payload(
    payload: Any,
    *,
    cache_root: Path,
    code_root: Path,
    runtime_root: Path,
    geometry: Geometry = PRIMARY_GEOMETRY,
    expected_rank: int | None = None,
    expected_source_bindings: Mapping[str, Mapping[str, str]] | None = None,
    expected_gpu_assignment_receipt: Mapping[str, Any] | None = None,
    expected_gpu_assignment_raw_sha256: str | None = None,
    expected_launcher_identity: Mapping[str, Any] | None = None,
    expected_launcher_identity_raw_sha256: str | None = None,
    expected_runtime_preflight_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay one rank's compact receipt without importing candidate runtime."""

    payload = _mapping(payload, "primary dispatch receipt")
    _require(
        set(payload)
        == {
            "schema_version",
            "rank",
            "scope",
            "hook_installation",
            "dispatch_source_bindings",
            "geometry",
            "tables",
            "cells",
            "attention_call_columns",
            "attention_calls",
            "gdn_call_columns",
            "gdn_calls",
            "rank_identity",
            "execution_binding",
        },
        "primary dispatch receipt fields drift",
    )
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schema drift")
    rank = payload.get("rank")
    _require(type(rank) is int and 0 <= rank < PRIMARY_WORLD_SIZE, "rank drift")
    if expected_rank is not None:
        _require(rank == expected_rank, "receipt/caller rank binding drift")
    _require(expected_source_bindings is not None, "external runtime source manifest is required")
    _require(expected_gpu_assignment_receipt is not None, "external GPU assignment receipt is required")
    _require(
        isinstance(expected_gpu_assignment_raw_sha256, str),
        "external GPU assignment raw SHA is required",
    )
    _require(expected_launcher_identity is not None, "external launcher identity is required")
    _require(
        isinstance(expected_launcher_identity_raw_sha256, str),
        "external launcher identity raw SHA is required",
    )
    _require(
        isinstance(expected_runtime_preflight_sha256, str)
        and base._is_sha256(expected_runtime_preflight_sha256),
        "external runtime preflight SHA is required",
    )
    _verify_execution_binding(
        payload,
        rank=rank,
        expected_runtime_preflight_sha256=expected_runtime_preflight_sha256,
    )
    rank_identity = _verify_rank_identity(
        payload,
        rank=rank,
        expected_gpu_assignment_receipt=expected_gpu_assignment_receipt,
        expected_gpu_assignment_raw_sha256=expected_gpu_assignment_raw_sha256,
        expected_launcher_identity=expected_launcher_identity,
        expected_launcher_identity_raw_sha256=expected_launcher_identity_raw_sha256,
    )
    scope = _mapping(payload.get("scope"), "scope")
    _require(
        set(scope)
        == {
            "primary_protocol",
            "primary_cells_only",
            "vllm_attention",
            "gdn",
            "trusted_runtime_boundary",
        },
        "scope fields drift",
    )
    _require(scope.get("primary_protocol") == PRIMARY_PROTOCOL, "protocol drift")
    _require(scope.get("primary_cells_only") is True, "scope is not primary-only")
    _require(scope.get("vllm_attention") == base.TARGET_VLLM_ENTRYPOINT, "attention scope drift")
    _require(scope.get("gdn") == GDN_SCOPE, "GDN eager-scope claim drift")
    _require(
        scope.get("trusted_runtime_boundary")
        == (
            "honest process with trusted pinned PyTorch/CUDA, vLLM, and Triton; "
            "post-return launcher receipt is not malicious-runtime or device attestation"
        ),
        "trusted-runtime claim boundary drift",
    )
    _verify_geometry(payload.get("geometry"), geometry)
    hook = _mapping(payload.get("hook_installation"), "hook installation")
    _require(hook == base.HOOK_INSTALLATION_RECEIPT, "hook-installation ordering drift")
    _verify_source_bindings(
        payload,
        code_root=code_root,
        runtime_root=runtime_root,
        expected_bindings=expected_source_bindings,
    )

    tables = _mapping(payload.get("tables"), "tables")
    _require(
        set(tables)
        == {
            "compiled_artifacts",
            "selected_compile_configurations",
            "call_shapes",
            "autotune_observations",
        },
        "table set drift",
    )
    artifacts = tables["compiled_artifacts"]
    configs = tables["selected_compile_configurations"]
    shapes = tables["call_shapes"]
    autotunes = tables["autotune_observations"]
    _require(isinstance(artifacts, list) and artifacts, "compiled artifact table empty")
    _require(isinstance(configs, list) and configs, "compiled config table empty")
    _require(isinstance(shapes, list) and shapes, "shape table empty")
    _require(isinstance(autotunes, list) and autotunes, "autotune table empty")
    for name, table in (
        ("compiled artifacts", artifacts),
        ("selected configurations", configs),
        ("call shapes", shapes),
        ("autotune observations", autotunes),
    ):
        canonical_rows = [_canonical(row) for row in table]
        _require(
            len(canonical_rows) == len(set(canonical_rows)),
            f"{name} table contains duplicate rows",
        )
    for index, artifact in enumerate(artifacts):
        base._verify_artifact(cache_root, artifact, f"artifact table row {index}")
        _require(
            _mapping(artifact, f"artifact table row {index}").get("kernel_name")
            == base.EXPECTED_TRITON_KERNEL_NAME,
            "compiled artifact is not unified attention",
        )
    for index, config_raw in enumerate(configs):
        config = _mapping(config_raw, f"config table row {index}")
        allowed_config_fields = {
            "name",
            "hash",
            *base.REQUIRED_TRITON_CONFIG_FIELDS,
            "maxnreg",
            "ptx_version",
            "ptx_options",
            "enable_fp_fusion",
        }
        _require(
            set(base.REQUIRED_TRITON_CONFIG_FIELDS).issubset(config)
            and set(config).issubset(allowed_config_fields)
            and isinstance(config.get("name"), str)
            and isinstance(config.get("hash"), str),
            "selected compile configuration incomplete",
        )
        _require(config.get("name") == base.EXPECTED_TRITON_KERNEL_NAME, "selected config kernel identity drift")
        candidates = base._candidate_artifacts(cache_root, config)
        _require(
            len(candidates) == 1,
            f"selected compile configuration {index} is not uniquely bound",
        )
    required_shape_fields = {
        "q",
        "k",
        "v",
        "out",
        "block_table",
        "max_seqlen_q",
        "max_seqlen_k",
        "softmax_scale",
    }
    for index, shape_raw in enumerate(shapes):
        record = _mapping(shape_raw, f"shape table row {index}")
        _require(set(record) == {"shape", "shape_sha256"}, "shape record fields drift")
        shape = _mapping(record.get("shape"), f"shape table row {index} value")
        _require(set(shape) == required_shape_fields, "attention call-shape field set drift")
        _require(
            record.get("shape_sha256")
            == base._sha256_bytes(base._canonical_bytes(dict(shape))),
            "attention call-shape digest drift",
        )
        q = shape.get("q")
        out = shape.get("out")
        k = shape.get("k")
        v = shape.get("v")
        block_table = shape.get("block_table")
        _require(
            isinstance(q, list)
            and q == out
            and len(q) == 3
            and q[0] in {1, 32}
            and q[1:] == [16, 256],
            "attention q/out geometry drift",
        )
        _require(
            isinstance(k, list)
            and k == v
            and len(k) == 4
            and type(k[0]) is int
            and k[0] > 0
            and k[1:] == [128, 2, 256],
            "attention K/V block-pool geometry drift",
        )
        _require(block_table == [1, 33], "attention block-table geometry drift")
        _require(shape.get("max_seqlen_q") == q[0], "attention max_seqlen_q drift")
        _require(
            type(shape.get("max_seqlen_k")) is int
            and 4127 <= shape["max_seqlen_k"] <= 4134,
            "attention max_seqlen_k drift",
        )
        _require(shape.get("softmax_scale") == 0.0625, "attention scale drift")
    for index, autotune_raw in enumerate(autotunes):
        autotune = _mapping(autotune_raw, f"autotune table row {index}")
        _require(autotune.get("mode") in {"triton-autotuner", "no-autotuner-observed"}, "autotune mode drift")
        if autotune.get("mode") == "no-autotuner-observed":
            _require(dict(autotune) == {"mode": "no-autotuner-observed"}, "no-autotuner row fields drift")
        else:
            _require(set(autotune) == {"mode", "events"}, "autotune row fields drift")
            events = autotune.get("events")
            _require(isinstance(events, list) and len(events) == 1, "autotune row must contain one selected event")
            event = _mapping(events[0], "selected autotune event")
            _require(
                set(event)
                == {
                    "selected_kwargs",
                    "selected_kwargs_sha256",
                    *base.REQUIRED_TRITON_CONFIG_FIELDS,
                },
                "selected autotune event fields drift",
            )
            selected_kwargs = base._normalise_json_value(
                event.get("selected_kwargs"), label="selected_kwargs"
            )
            _require(isinstance(selected_kwargs, dict), "selected_kwargs must be an object")
            _require(
                event.get("selected_kwargs_sha256")
                == base._sha256_bytes(base._canonical_bytes(selected_kwargs)),
                "selected_kwargs digest drift",
            )

    cells = payload.get("cells")
    attention = payload.get("attention_calls")
    gdn = payload.get("gdn_calls")
    _require(payload.get("attention_call_columns") == ATTENTION_COLUMNS, "attention column semantics drift")
    _require(payload.get("gdn_call_columns") == GDN_COLUMNS, "GDN column semantics drift")
    _require(isinstance(cells, list), "cell table missing")
    _require(isinstance(attention, list), "attention call table missing")
    _require(isinstance(gdn, list), "GDN call table missing")
    expected_cell_rows = expected_cells(geometry)
    _require(len(cells) == len(expected_cell_rows), "primary cell count drift")
    expected_attention_cursor = 0
    expected_gdn_cursor = 0
    referenced_shapes: set[int] = set()
    referenced_artifacts: set[int] = set()
    referenced_configs: set[int] = set()
    referenced_autotunes: set[int] = set()
    observed_call_ids: set[str] = set()
    observed_call_receipts: set[str] = set()
    for cell_index, (cell_raw, expected) in enumerate(zip(cells, expected_cell_rows)):
        cell = _mapping(cell_raw, f"cell {cell_index}")
        _require(
            set(cell)
            == {
                "cell_index",
                "rank",
                "resident_count",
                "arm_id",
                "kv_policy",
                "gdn_base_policy",
                "cell_role",
                "attention_call_range",
                "gdn_call_range",
                "expected_attention_calls",
                "expected_gdn_document_prefill_calls",
                "expected_gdn_request_calls",
            },
            f"cell {cell_index} fields drift",
        )
        _require(cell.get("cell_index") == cell_index, "cell index drift")
        _require(cell.get("rank") == rank, "cell rank drift")
        for key, value in expected.items():
            _require(cell.get(key) == value, f"cell {cell_index} {key} drift")
        resident_count = expected["resident_count"]
        expected_attention = resident_count * geometry.generation_steps * len(geometry.full_layers)
        expected_prefill = len(geometry.linear_layers)
        expected_request = resident_count * geometry.generation_steps * len(geometry.linear_layers)
        _require(cell.get("expected_attention_calls") == expected_attention, "cell attention expectation drift")
        _require(cell.get("expected_gdn_document_prefill_calls") == expected_prefill, "cell GDN prefill expectation drift")
        _require(cell.get("expected_gdn_request_calls") == expected_request, "cell GDN request expectation drift")
        attention_range = [expected_attention_cursor, expected_attention_cursor + expected_attention]
        gdn_range = [expected_gdn_cursor, expected_gdn_cursor + expected_prefill + expected_request]
        _require(cell.get("attention_call_range") == attention_range, "cell attention range drift")
        _require(cell.get("gdn_call_range") == gdn_range, "cell GDN range drift")

        for local_index, row_raw in enumerate(attention[attention_range[0] : attention_range[1]]):
            _require(
                isinstance(row_raw, list) and len(row_raw) == len(ATTENTION_COLUMNS),
                "attention compact row malformed",
            )
            (
                observed_cell,
                observed_local,
                call_id,
                shape_index,
                artifact_index,
                config_index,
                autotune_index,
                visible_uuid,
                device_index,
                stream_id,
                post_returned,
                context_matches,
                call_receipt_sha256,
            ) = row_raw
            _require(observed_cell == cell_index and observed_local == local_index, "attention call order drift")
            expected_call_id = f"rank-{rank}/cell-{cell_index}/attention-{local_index}"
            _require(call_id == expected_call_id, "attention call ID drift")
            _require(call_id not in observed_call_ids, "duplicate attention call ID")
            observed_call_ids.add(call_id)
            _require(type(shape_index) is int and 0 <= shape_index < len(shapes), "attention shape index drift")
            _require(type(artifact_index) is int and 0 <= artifact_index < len(artifacts), "attention artifact index drift")
            _require(type(config_index) is int and 0 <= config_index < len(configs), "attention config index drift")
            _require(type(autotune_index) is int and 0 <= autotune_index < len(autotunes), "attention autotune index drift")
            referenced_shapes.add(shape_index)
            referenced_artifacts.add(artifact_index)
            referenced_configs.add(config_index)
            referenced_autotunes.add(autotune_index)
            _require(
                visible_uuid == rank_identity["assigned_gpu_uuid"],
                "attention GPU UUID differs from launcher assignment",
            )
            _require(device_index == 0, "attention CUDA device index drift")
            _require(type(stream_id) is int and stream_id >= 0, "attention CUDA stream identity invalid")
            _require(post_returned is True, "attention lacks successful launcher return")
            _require(context_matches is True, "attention post-return launch context drift")
            artifact = _mapping(artifacts[artifact_index], "selected artifact")
            config = _mapping(configs[config_index], "selected configuration")
            _require(config.get("name") == artifact.get("kernel_name"), "artifact/config kernel mismatch")
            _require(config.get("hash") == artifact.get("compiler_hash"), "artifact/config hash mismatch")
            for field in base.REQUIRED_TRITON_CONFIG_FIELDS:
                _require(config.get(field) == artifact.get("compile_config", {}).get(field), f"artifact/config {field} mismatch")
            autotune = _mapping(autotunes[autotune_index], "selected autotune observation")
            if autotune.get("mode") == "triton-autotuner":
                events = autotune.get("events")
                _require(isinstance(events, list) and len(events) == 1, "autotune selection absent")
                selected = _mapping(events[0], "selected autotune event")
                for field in base.REQUIRED_TRITON_CONFIG_FIELDS:
                    _require(selected.get(field) == config.get(field), f"autotune/config {field} mismatch")
            shape = _mapping(shapes[shape_index], "selected shape").get("shape")
            launch_context = {
                "cuda_visible_devices": visible_uuid,
                "torch_device_index": device_index,
                "torch_device_type": "cuda",
                "torch_stream_id": stream_id,
            }
            receipt_core = {
                "call_id": call_id,
                "call_shape": shape,
                "artifact_id": artifact["artifact_id"],
                "selected_compile_config": dict(config),
                "autotune": dict(autotune),
                "launch_context": launch_context,
                "post_launcher_returned": True,
                "post_return_context_matches": True,
            }
            _require(
                call_receipt_sha256
                == base._sha256_bytes(base._canonical_bytes(receipt_core)),
                "attention call receipt digest drift",
            )
            _require(call_receipt_sha256 not in observed_call_receipts, "duplicate attention call receipt")
            observed_call_receipts.add(call_receipt_sha256)

        cell_gdn = gdn[gdn_range[0] : gdn_range[1]]
        for local_index, row_raw in enumerate(cell_gdn):
            _require(
                isinstance(row_raw, list) and len(row_raw) == 10,
                "GDN compact row malformed",
            )
            (
                observed_cell,
                observed_local,
                layer_idx,
                sequence_length,
                has_previous,
                chunk_calls,
                recurrent_rule_calls,
                functional_conv_calls,
                inplace_conv_calls,
                recurrent_rebind_calls,
            ) = row_raw
            _require(observed_cell == cell_index and observed_local == local_index, "GDN call order drift")
            if local_index < expected_prefill:
                expected_layer = geometry.linear_layers[local_index]
                _require(layer_idx == expected_layer, "GDN prefill layer order drift")
                _require(sequence_length == geometry.document_tokens and has_previous is False, "GDN prefill phase geometry drift")
                expected_route_counts = (1, 0, 1, 0, 1)
            else:
                request_local = local_index - expected_prefill
                expected_layer = geometry.linear_layers[request_local % len(geometry.linear_layers)]
                forward_index = request_local // len(geometry.linear_layers)
                round_index = forward_index // resident_count
                _require(layer_idx == expected_layer, "GDN request layer order drift")
                _require(0 <= round_index < geometry.generation_steps, "GDN request round drift")
                expected_length = 32 if round_index == 0 else 1
                _require(sequence_length == expected_length and has_previous is True, "GDN request phase geometry drift")
                expected_route_counts = (
                    (0, 1, 0, 1, 1)
                    if expected_length == 1
                    else (1, 0, 1, 0, 1)
                )
            _require(
                (
                    chunk_calls,
                    recurrent_rule_calls,
                    functional_conv_calls,
                    inplace_conv_calls,
                    recurrent_rebind_calls,
                )
                == expected_route_counts,
                "GDN mutually exclusive route/rebind event count drift",
            )

        expected_attention_cursor = attention_range[1]
        expected_gdn_cursor = gdn_range[1]

    counts = expected_rank_counts(geometry)
    _require(expected_attention_cursor == len(attention) == counts["attention_call_count"], "rank attention closure failed")
    _require(expected_gdn_cursor == len(gdn) == counts["gdn_call_count"], "rank GDN closure failed")
    _require(referenced_shapes == set(range(len(shapes))), "shape table contains an unreferenced row")
    _require(referenced_artifacts == set(range(len(artifacts))), "artifact table contains an unreferenced row")
    _require(referenced_configs == set(range(len(configs))), "config table contains an unreferenced row")
    _require(referenced_autotunes == set(range(len(autotunes))), "autotune table contains an unreferenced row")
    return {
        "schema_version": SCHEMA_VERSION,
        "replay_verdict": "pass",
        "rank": rank,
        **counts,
        "compiled_artifact_count": len(artifacts),
        "selected_compile_configuration_count": len(configs),
    }


def verify_primary_shard(
    shard: Any,
    *,
    expected_rank: int,
    geometry: Geometry = PRIMARY_GEOMETRY,
    receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind compact call ranges to the old runner's emitted primary cells."""

    shard = _mapping(shard, "primary shard")
    _require(shard.get("schema_version") == PRIMARY_SHARD_SCHEMA, "primary shard schema drift")
    _require(shard.get("protocol") == PRIMARY_PROTOCOL, "primary shard protocol drift")
    _require(shard.get("status") == "completed_formal_gpu_shard", "primary shard incomplete")
    _require(shard.get("rank") == expected_rank, "primary shard rank drift")
    _require(shard.get("world_size") == PRIMARY_WORLD_SIZE, "primary shard world-size drift")
    config = _mapping(shard.get("protocol_config"), "primary protocol config")
    _require(config.get("resident_counts") == list(geometry.resident_counts), "primary resident-count drift")
    _require(config.get("generation_steps") == geometry.generation_steps, "primary generation-step drift")
    _require(config.get("document_tokens") == geometry.document_tokens, "primary document-length drift")
    _require(config.get("factorial_arm_ids") == list(geometry.arms), "primary arm order drift")
    rows = shard.get("factorial")
    _require(isinstance(rows, list) and len(rows) == len(geometry.resident_counts), "primary factorial row drift")
    cell_count = 0
    attention_ledger_calls = 0
    expected_call_shapes: list[dict[str, Any]] = []
    for n_row, resident_count in zip(rows, geometry.resident_counts):
        n_row = _mapping(n_row, "primary resident row")
        _require(n_row.get("resident_count") == resident_count, "primary N order drift")
        cells = n_row.get("cells")
        _require(isinstance(cells, list) and len(cells) == len(geometry.arms), "primary arm cell count drift")
        for cell, arm_id in zip(cells, geometry.arms):
            cell = _mapping(cell, "primary factorial cell")
            _require(cell.get("arm_id") == arm_id, "primary arm binding drift")
            for ledger_key in ("memory_kernel_ledgers", "witness_kernel_ledgers"):
                ledgers = cell.get(ledger_key)
                _require(isinstance(ledgers, list) and len(ledgers) == resident_count, f"{ledger_key} count drift")
                for request_index, ledger_raw in enumerate(ledgers):
                    ledger = _mapping(ledger_raw, f"{ledger_key} row")
                    _require(ledger.get("request_index") == request_index, f"{ledger_key} request order drift")
                    _require(ledger.get("total_calls") == geometry.generation_steps * len(geometry.full_layers), f"{ledger_key} call count drift")
                    _require(ledger.get("verified") is True, f"{ledger_key} is unverified")
                    _require(ledger.get("dense_fallback_calls") == 0, f"{ledger_key} used dense fallback")
                    attention_ledger_calls += ledger["total_calls"]
                for round_index in range(geometry.generation_steps):
                    for request_index, ledger_raw in enumerate(ledgers):
                        ledger = _mapping(ledger_raw, f"{ledger_key} row")
                        calls = ledger.get("calls")
                        _require(
                            isinstance(calls, list)
                            and len(calls)
                            == geometry.generation_steps * len(geometry.full_layers),
                            f"{ledger_key} per-call rows drift",
                        )
                        start = round_index * len(geometry.full_layers)
                        for layer_position, layer_idx in enumerate(geometry.full_layers):
                            call = _mapping(calls[start + layer_position], f"{ledger_key} call")
                            _require(call.get("request_index") == request_index, f"{ledger_key} call request drift")
                            _require(call.get("layer_idx") == layer_idx, f"{ledger_key} call layer order drift")
                            query_tokens = call.get("query_tokens")
                            pool_shape = call.get("physical_block_pool_shape")
                            table_shape = call.get("active_block_table_shape")
                            kv_tokens = call.get("kv_tokens")
                            scale = call.get("softmax_scale")
                            _require(query_tokens == (32 if round_index == 0 else 1), f"{ledger_key} query-token schedule drift")
                            expected_call_shapes.append(
                                {
                                    "q": [query_tokens, 16, 256],
                                    "k": pool_shape,
                                    "v": pool_shape,
                                    "out": [query_tokens, 16, 256],
                                    "block_table": table_shape,
                                    "max_seqlen_q": query_tokens,
                                    "max_seqlen_k": kv_tokens,
                                    "softmax_scale": scale,
                                }
                            )
            cell_count += 2
    expected = expected_rank_counts(geometry)
    _require(cell_count == expected["cell_count"], "primary memory/witness cell total drift")
    _require(attention_ledger_calls == expected["attention_call_count"], "primary ledger/compiled receipt attention count mismatch")
    _require(len(expected_call_shapes) == expected["attention_call_count"], "primary shape-ledger call count drift")
    if receipt is not None:
        tables = _mapping(receipt.get("tables"), "receipt tables")
        shapes = tables.get("call_shapes")
        calls = receipt.get("attention_calls")
        _require(isinstance(shapes, list) and isinstance(calls, list), "receipt shape/call table missing")
        _require(len(calls) == len(expected_call_shapes), "receipt/shard shape call count drift")
        for call_index, (call_row, expected_shape) in enumerate(zip(calls, expected_call_shapes)):
            _require(isinstance(call_row, list) and len(call_row) == len(ATTENTION_COLUMNS), "receipt attention row malformed")
            shape_index = call_row[ATTENTION_COLUMNS.index("shape_table_index")]
            _require(type(shape_index) is int and 0 <= shape_index < len(shapes), "receipt shape index drift")
            record = _mapping(shapes[shape_index], f"receipt shape row for call {call_index}")
            _require(record.get("shape") == expected_shape, f"receipt/shard call-shape mismatch at call {call_index}")
    return {
        "schema_version": PRIMARY_SHARD_SCHEMA,
        "rank": expected_rank,
        "status": "pass",
        "cell_count": cell_count,
        "attention_ledger_call_count": attention_ledger_calls,
        "attention_shape_binding_count": len(expected_call_shapes),
    }


def verify_primary_aggregate(value: Any) -> dict[str, Any]:
    aggregate = _mapping(value, "primary aggregate")
    _require(aggregate.get("schema_version") == PRIMARY_AGGREGATE_SCHEMA, "primary aggregate schema drift")
    _require(aggregate.get("protocol") == PRIMARY_PROTOCOL, "primary aggregate protocol drift")
    _require(aggregate.get("formal_ready") is True and aggregate.get("passed") is True, "primary aggregate did not pass")
    _require(aggregate.get("rank_count") == PRIMARY_WORLD_SIZE, "primary aggregate rank count drift")
    _require(aggregate.get("factorial_four_cell_exact") is True, "primary factorial semantics differ")
    _require(aggregate.get("oracle_all_ranks_passed") is True, "primary oracle failed")
    return {
        "schema_version": PRIMARY_AGGREGATE_SCHEMA,
        "status": "pass",
        "rank_count": PRIMARY_WORLD_SIZE,
        "factorial_configuration_count": 96,
    }
