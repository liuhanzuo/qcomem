#!/usr/bin/env python3
"""Compact, fail-closed dispatch receipts for the RR2 primary factorial.

The runtime hooks are inherited from the independently tested Round-39
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


SCHEMA_VERSION = "forkaudit-r39-primary-compiled-dispatch-receipt-v1"
AGGREGATE_SCHEMA_VERSION = "forkaudit-r39-primary-compiled-dispatch-aggregate-v1"
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
    call_shape: dict[str, Any]
    launches: list[tuple[int, int]] = field(default_factory=list)
    autotune_events: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class _CellState:
    cell_index: int
    metadata: dict[str, Any]
    attention_start: int
    gdn_start: int


class PrimaryDispatchRecorder(base.DispatchReceiptRecorder):
    """Round-39 hooks with primary-only contexts and compact per-call rows."""

    def __init__(self, *, cache_root: Path, code_root: Path, runtime_root: Path) -> None:
        super().__init__(
            cache_root=cache_root,
            code_root=code_root,
            runtime_root=runtime_root,
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
        self._compiled_metadata_cache: dict[str, tuple[int, int]] = {}
        self._active_cell: contextvars.ContextVar[_CellState | None] = (
            contextvars.ContextVar("forkaudit_r39_primary_cell", default=None)
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
        if self._active_cell.get() is None:
            return self._active_attention.set(None)
        context = _CompactAttention(call_shape=base._call_shape(kwargs))
        return self._active_attention.set(context)  # type: ignore[arg-type]

    def record_compiled_kernel(self, kernel: Any) -> None:
        context = self._active_attention.get()
        if context is None:
            return
        _require(isinstance(context, _CompactAttention), "attention context type drift")
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
            artifact_id = str(artifact["artifact_id"])
            if artifact_id not in self._artifact_index:
                self._artifact_index[artifact_id] = len(self.artifact_table)
                self.artifact_table.append(artifact)
            artifact_index = self._artifact_index[artifact_id]
            config_index = self._intern(
                normalised, self.config_table, self._config_index
            )
            cached = (artifact_index, config_index)
            self._compiled_metadata_cache[cache_key] = cached
        context.launches.append(cached)

    def record_autotune(self, autotuner: Any) -> None:
        context = self._active_attention.get()
        if context is None:
            return
        _require(isinstance(context, _CompactAttention), "attention context type drift")
        best = getattr(autotuner, "best_config", None)
        if best is None:
            return
        kwargs = getattr(best, "kwargs", {})
        if not isinstance(kwargs, Mapping):
            kwargs = {}
        context.autotune_events.append(
            {
                "selected_kwargs": dict(kwargs),
                "num_warps": getattr(best, "num_warps", None),
                "num_stages": getattr(best, "num_stages", None),
                "num_ctas": getattr(best, "num_ctas", None),
            }
        )

    def finish_attention(self, token: contextvars.Token[Any]) -> None:
        context = self._active_attention.get()
        self._active_attention.reset(token)
        if context is None:
            return
        _require(isinstance(context, _CompactAttention), "attention context type drift")
        _require(len(context.launches) == 1, "attention must launch exactly one kernel")
        state = self._active_cell.get()
        _require(state is not None, "primary cell disappeared during attention")
        artifact_index, config_index = context.launches[0]
        autotune = (
            {"mode": "triton-autotuner", "events": context.autotune_events}
            if context.autotune_events
            else {"mode": "no-autotuner-observed"}
        )
        shape_index = self._intern(context.call_shape, self.shape_table, self._shape_index)
        autotune_index = self._intern(
            autotune, self.autotune_table, self._autotune_index
        )
        local_index = len(self.compact_attention_calls) - state.attention_start
        self.compact_attention_calls.append(
            [
                state.cell_index,
                local_index,
                shape_index,
                artifact_index,
                config_index,
                autotune_index,
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
        _require(context.chunk_kernel_events == 1, "GDN chunk dispatch count drift")
        _require(context.conv_rebind_events == 1, "GDN conv rebind count drift")
        _require(context.recurrent_rebind_events == 1, "GDN recurrent rebind count drift")
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
                context.conv_rebind_events,
                context.recurrent_rebind_events,
            ]
        )

    def payload(self) -> dict[str, Any]:
        _require(self._factorial_finished, "primary factorial did not close")
        counts = expected_rank_counts()
        _require(len(self.compact_attention_calls) == counts["attention_call_count"], "rank attention total drift")
        _require(len(self.compact_gdn_calls) == counts["gdn_call_count"], "rank GDN total drift")
        _require(set(self.gdn_source_bindings) == base.GDN_SOURCE_KEYS, "GDN source bindings incomplete")
        _require(self.hook_installation, "hook-installation receipt absent")
        return {
            "schema_version": SCHEMA_VERSION,
            "rank": self._factorial_rank,
            "scope": {
                "primary_protocol": PRIMARY_PROTOCOL,
                "primary_cells_only": True,
                "vllm_attention": base.TARGET_VLLM_ENTRYPOINT,
                "gdn": (
                    "actual Transformers Qwen3.5-MoE native eager GDN plus qcomem "
                    "functional cache rebind; underlying ATen/CUDA libraries are out of scope"
                ),
            },
            "hook_installation": copy.deepcopy(self.hook_installation),
            "gdn_source_bindings": copy.deepcopy(self.gdn_source_bindings),
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
            "attention_call_columns": [
                "cell_index",
                "local_call_index",
                "shape_table_index",
                "artifact_table_index",
                "config_table_index",
                "autotune_table_index",
            ],
            "attention_calls": copy.deepcopy(self.compact_attention_calls),
            "gdn_call_columns": [
                "cell_index",
                "local_call_index",
                "layer_idx",
                "sequence_length",
                "cache_has_previous_state",
                "chunk_rule_calls",
                "conv_rebind_calls",
                "recurrent_rebind_calls",
            ],
            "gdn_calls": copy.deepcopy(self.compact_gdn_calls),
        }


def _verify_source_bindings(
    payload: Mapping[str, Any], *, code_root: Path, runtime_root: Path
) -> None:
    base._verify_gdn_source_bindings(
        payload, code_root=code_root, runtime_root=runtime_root
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


def verify_payload(
    payload: Any,
    *,
    cache_root: Path,
    code_root: Path,
    runtime_root: Path,
    geometry: Geometry = PRIMARY_GEOMETRY,
) -> dict[str, Any]:
    """Replay one rank's compact receipt without importing candidate runtime."""

    payload = _mapping(payload, "primary dispatch receipt")
    _require(payload.get("schema_version") == SCHEMA_VERSION, "schema drift")
    rank = payload.get("rank")
    _require(type(rank) is int and 0 <= rank < PRIMARY_WORLD_SIZE, "rank drift")
    scope = _mapping(payload.get("scope"), "scope")
    _require(scope.get("primary_protocol") == PRIMARY_PROTOCOL, "protocol drift")
    _require(scope.get("primary_cells_only") is True, "scope is not primary-only")
    _require(scope.get("vllm_attention") == base.TARGET_VLLM_ENTRYPOINT, "attention scope drift")
    _verify_geometry(payload.get("geometry"), geometry)
    hook = _mapping(payload.get("hook_installation"), "hook installation")
    _require(
        hook
        == {
            "functional_stack_preloaded": False,
            "native_cache_module_preloaded": False,
            "transformers_qwen35_module_preloaded": False,
            "transformers_qwen35_moe_module_preloaded": False,
            "patched_before_entrypoint": True,
            "patched_before_model_instance_binding": True,
            "frozen_fast_path_available": False,
        },
        "hook-installation ordering drift",
    )
    _verify_source_bindings(payload, code_root=code_root, runtime_root=runtime_root)

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
    for index, artifact in enumerate(artifacts):
        base._verify_artifact(cache_root, artifact, f"artifact table row {index}")
    for index, config_raw in enumerate(configs):
        config = _mapping(config_raw, f"config table row {index}")
        _require(
            set(base.REQUIRED_TRITON_CONFIG_FIELDS).issubset(config)
            and isinstance(config.get("name"), str)
            and isinstance(config.get("hash"), str),
            "selected compile configuration incomplete",
        )
        candidates = base._candidate_artifacts(cache_root, config)
        _require(
            len(candidates) == 1,
            f"selected compile configuration {index} is not uniquely bound",
        )
    for index, autotune_raw in enumerate(autotunes):
        autotune = _mapping(autotune_raw, f"autotune table row {index}")
        _require(autotune.get("mode") in {"triton-autotuner", "no-autotuner-observed"}, "autotune mode drift")

    cells = payload.get("cells")
    attention = payload.get("attention_calls")
    gdn = payload.get("gdn_calls")
    _require(isinstance(cells, list), "cell table missing")
    _require(isinstance(attention, list), "attention call table missing")
    _require(isinstance(gdn, list), "GDN call table missing")
    expected_cell_rows = expected_cells(geometry)
    _require(len(cells) == len(expected_cell_rows), "primary cell count drift")
    expected_attention_cursor = 0
    expected_gdn_cursor = 0
    for cell_index, (cell_raw, expected) in enumerate(zip(cells, expected_cell_rows)):
        cell = _mapping(cell_raw, f"cell {cell_index}")
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
            _require(isinstance(row_raw, list) and len(row_raw) == 6, "attention compact row malformed")
            observed_cell, observed_local, shape_index, artifact_index, config_index, autotune_index = row_raw
            _require(observed_cell == cell_index and observed_local == local_index, "attention call order drift")
            _require(type(shape_index) is int and 0 <= shape_index < len(shapes), "attention shape index drift")
            _require(type(artifact_index) is int and 0 <= artifact_index < len(artifacts), "attention artifact index drift")
            _require(type(config_index) is int and 0 <= config_index < len(configs), "attention config index drift")
            _require(type(autotune_index) is int and 0 <= autotune_index < len(autotunes), "attention autotune index drift")
            artifact = _mapping(artifacts[artifact_index], "selected artifact")
            config = _mapping(configs[config_index], "selected configuration")
            _require(config.get("name") == artifact.get("kernel_name"), "artifact/config kernel mismatch")
            _require(config.get("hash") == artifact.get("compiler_hash"), "artifact/config hash mismatch")
            for field in base.REQUIRED_TRITON_CONFIG_FIELDS:
                _require(config.get(field) == artifact.get("compile_config", {}).get(field), f"artifact/config {field} mismatch")
            autotune = _mapping(autotunes[autotune_index], "selected autotune observation")
            if autotune.get("mode") == "triton-autotuner":
                events = autotune.get("events")
                _require(isinstance(events, list) and events, "autotune selection absent")
                selected = _mapping(events[-1], "selected autotune event")
                for field in base.REQUIRED_TRITON_CONFIG_FIELDS:
                    _require(selected.get(field) == config.get(field), f"autotune/config {field} mismatch")

        cell_gdn = gdn[gdn_range[0] : gdn_range[1]]
        for local_index, row_raw in enumerate(cell_gdn):
            _require(isinstance(row_raw, list) and len(row_raw) == 8, "GDN compact row malformed")
            observed_cell, observed_local, layer_idx, sequence_length, has_previous, chunk_calls, conv_calls, recurrent_calls = row_raw
            _require(observed_cell == cell_index and observed_local == local_index, "GDN call order drift")
            _require((chunk_calls, conv_calls, recurrent_calls) == (1, 1, 1), "GDN route/rebind event count drift")
            if local_index < expected_prefill:
                expected_layer = geometry.linear_layers[local_index]
                _require(layer_idx == expected_layer, "GDN prefill layer order drift")
                _require(sequence_length == geometry.document_tokens and has_previous is False, "GDN prefill phase geometry drift")
            else:
                request_local = local_index - expected_prefill
                expected_layer = geometry.linear_layers[request_local % len(geometry.linear_layers)]
                forward_index = request_local // len(geometry.linear_layers)
                round_index = forward_index // resident_count
                _require(layer_idx == expected_layer, "GDN request layer order drift")
                _require(0 <= round_index < geometry.generation_steps, "GDN request round drift")
                expected_length = 32 if round_index == 0 else 1
                _require(sequence_length == expected_length and has_previous is True, "GDN request phase geometry drift")

        expected_attention_cursor = attention_range[1]
        expected_gdn_cursor = gdn_range[1]

    counts = expected_rank_counts(geometry)
    _require(expected_attention_cursor == len(attention) == counts["attention_call_count"], "rank attention closure failed")
    _require(expected_gdn_cursor == len(gdn) == counts["gdn_call_count"], "rank GDN closure failed")
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
            cell_count += 2
    expected = expected_rank_counts(geometry)
    _require(cell_count == expected["cell_count"], "primary memory/witness cell total drift")
    _require(attention_ledger_calls == expected["attention_call_count"], "primary ledger/compiled receipt attention count mismatch")
    return {
        "schema_version": PRIMARY_SHARD_SCHEMA,
        "rank": expected_rank,
        "status": "pass",
        "cell_count": cell_count,
        "attention_ledger_call_count": attention_ledger_calls,
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
