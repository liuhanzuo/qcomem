from __future__ import annotations

import hashlib
import hmac
import secrets
from typing import Any, Mapping

import torch

from .fixture import Fixture, get_tensor, set_tensor
from .protocol import (
    LIVE_ITEM_FIELDS,
    coordinate_key,
    require,
    resolve_slot,
    validate_live_items,
    validate_manifest,
)


def _candidate_get(fixture: Fixture, coordinate: Mapping[str, Any]) -> torch.Tensor:
    """Candidate-side traversal, intentionally separate from oracle_adapter."""

    owner = (
        fixture.persistent
        if coordinate["owner_kind"] == "persistent"
        else fixture.requests[int(coordinate["request_index"])]
    )
    layer = owner.layers[int(coordinate["layer_index"])]
    family = str(coordinate["state_family"])
    states = layer.conv_states if family == "conv" else layer.recurrent_states
    tensor = states[int(coordinate["state_index"])]
    require(isinstance(tensor, torch.Tensor), "candidate state is not a tensor")
    return tensor


def bind_candidate_items(
    manifest: Mapping[str, Any],
    fixture: Fixture,
    *,
    overrides: Mapping[str, torch.Tensor] | None = None,
) -> list[dict[str, Any]]:
    rows = validate_manifest(manifest)
    override_map = dict(overrides or {})
    require(set(override_map) <= set(rows), "unknown candidate override")
    items: list[dict[str, Any]] = []
    for slot, coordinate in sorted(rows.items()):
        tensor = override_map.get(slot, _candidate_get(fixture, coordinate))
        require(isinstance(tensor, torch.Tensor), "candidate bound a non-tensor")
        items.append({"slot_id": slot, "tensor": tensor})
    validate_live_items(items, manifest)
    require(set(items[0]) == LIVE_ITEM_FIELDS, "candidate live field drift")
    return items


def _object_token(secret: bytes, tensor: torch.Tensor) -> str:
    storage = tensor.untyped_storage()
    fact = (
        f"{id(tensor)}\0{int(storage.data_ptr())}\0{int(storage.nbytes())}"
        f"\0{tuple(tensor.shape)}\0{tuple(tensor.stride())}\0{tensor.storage_offset()}"
    ).encode("ascii")
    return hmac.new(secret, b"r40-producer-object-v1\0" + fact, hashlib.sha256).hexdigest()


def _mapping(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    return {str(item["slot_id"]): item["tensor"] for item in items}


def apply_fault_and_bind(
    manifest: Mapping[str, Any],
    fixture: Fixture,
    fault_spec: Mapping[str, Any],
    *,
    lane_type: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(lane_type in {"clean", "mutant"}, "lane type drift")
    before_items = bind_candidate_items(manifest, fixture)
    before = _mapping(before_items)
    overrides: dict[str, torch.Tensor] = {}
    expected_target_slots = [
        resolve_slot(manifest, coordinate) for coordinate in fault_spec["targets"]
    ]
    source_slots = [
        resolve_slot(manifest, coordinate)
        for coordinate in fault_spec.get("sources", [])
    ]
    graph_fields_mutated = False

    if lane_type == "mutant":
        kind = fault_spec["kind"]
        if kind == "coherent_slot_swap":
            require(len(fault_spec["targets"]) == 2, "swap target drift")
            left_coordinate, right_coordinate = fault_spec["targets"]
            left_tensor = get_tensor(fixture, left_coordinate)
            right_tensor = get_tensor(fixture, right_coordinate)
            require(left_tensor is not right_tensor, "swap tensors are identical")
            require(tuple(left_tensor.shape) == tuple(right_tensor.shape), "swap shape drift")
            set_tensor(fixture, left_coordinate, right_tensor)
            set_tensor(fixture, right_coordinate, left_tensor)
            source_slots = [expected_target_slots[1], expected_target_slots[0]]
            graph_fields_mutated = True
        elif kind == "stale_handle_after_rebind":
            require(len(expected_target_slots) == 1, "stale target drift")
            coordinate = fault_spec["targets"][0]
            stale = fixture.stale_handles.get(coordinate_key(coordinate))
            require(isinstance(stale, torch.Tensor), "stale handle missing")
            require(stale is not get_tensor(fixture, coordinate), "stale handle still live")
            overrides[expected_target_slots[0]] = stale
        elif kind in {"cross_layer_substitution", "request_base_role_misbinding"}:
            require(
                len(fault_spec["targets"]) == len(fault_spec.get("sources", [])) == 1,
                "substitution target/source drift",
            )
            target = fault_spec["targets"][0]
            source = fault_spec["sources"][0]
            target_tensor = get_tensor(fixture, target)
            source_tensor = get_tensor(fixture, source)
            require(target_tensor is not source_tensor, "substitution is already aliased")
            require(
                tuple(target_tensor.shape) == tuple(source_tensor.shape),
                "substitution shape drift",
            )
            set_tensor(fixture, target, source_tensor)
            graph_fields_mutated = True
        else:
            raise RuntimeError(f"unknown fault kind: {kind}")

    after_items = bind_candidate_items(manifest, fixture, overrides=overrides)
    after = _mapping(after_items)
    changed_slots = sorted(slot for slot in before if after[slot] is not before[slot])
    secret = secrets.token_bytes(32)
    token_rows = [
        {
            "slot_id": slot,
            "before_object_token": _object_token(secret, before[slot]),
            "bound_object_token": _object_token(secret, after[slot]),
            "reference_changed": before[slot] is not after[slot],
        }
        for slot in changed_slots
    ]
    if lane_type == "mutant":
        require(
            changed_slots == sorted(expected_target_slots),
            "actual changed-slot set differs from frozen targets",
        )
    else:
        require(not changed_slots, "clean lane changed a live handle")
    receipt = {
        "schema_version": "forkaudit-r40-live-handle-injection-receipt-v1",
        "fault_id": fault_spec["fault_id"],
        "fault_kind": fault_spec["kind"],
        "lane_type": lane_type,
        "applied": lane_type == "mutant",
        "semantic_manifest_sha256_before": manifest["manifest_sha256"],
        "semantic_manifest_sha256_after": manifest["manifest_sha256"],
        "live_wire_fields_before": sorted(LIVE_ITEM_FIELDS),
        "live_wire_fields_after": sorted(LIVE_ITEM_FIELDS),
        "schema_or_label_row_mutation_used": False,
        "actual_live_tensor_references_changed": bool(changed_slots),
        "graph_fields_mutated": graph_fields_mutated,
        "stale_binder_override_used": bool(overrides),
        "expected_target_slot_ids": sorted(expected_target_slots),
        "source_slot_ids": sorted(source_slots),
        "changed_slot_ids": changed_slots,
        "object_token_rows": token_rows,
        "raw_addresses_serialized": False,
    }
    return after_items, receipt


__all__ = ["apply_fault_and_bind", "bind_candidate_items"]

