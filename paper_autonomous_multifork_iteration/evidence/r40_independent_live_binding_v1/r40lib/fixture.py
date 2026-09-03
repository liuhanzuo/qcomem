from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from .protocol import coordinate_key, require


@dataclass
class LayerState:
    conv_states: dict[int, torch.Tensor]
    recurrent_states: dict[int, torch.Tensor]


@dataclass
class OwnerState:
    layers: list[LayerState]


@dataclass
class Fixture:
    persistent: OwnerState
    requests: list[OwnerState]
    stale_handles: dict[tuple[Any, ...], torch.Tensor]


def _owner_code(owner_kind: str, request_index: int | None) -> int:
    if owner_kind == "persistent":
        require(request_index is None, "persistent request index drift")
        return 1
    require(owner_kind == "request" and request_index is not None, "owner drift")
    return 2 + int(request_index)


def _make_tensor(
    shape: list[int], *, owner_code: int, layer_index: int, family_index: int
) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= int(size)
    offset = owner_code * 10000 + int(layer_index) * 100 + family_index * 30
    tensor = (torch.arange(count, dtype=torch.float32) + float(offset)).reshape(shape)
    return tensor.clone().share_memory_()


def _make_owner(
    preregistration: Mapping[str, Any], owner_kind: str, request_index: int | None
) -> OwnerState:
    fixture = preregistration["fixture"]
    layers: list[LayerState] = []
    maximum_layer = max(int(value) for value in fixture["layer_indices"])
    for layer_index in range(maximum_layer + 1):
        conv = _make_tensor(
            fixture["shapes"]["conv"],
            owner_code=_owner_code(owner_kind, request_index),
            layer_index=layer_index,
            family_index=0,
        )
        recurrent = _make_tensor(
            fixture["shapes"]["recurrent"],
            owner_code=_owner_code(owner_kind, request_index),
            layer_index=layer_index,
            family_index=1,
        )
        layers.append(LayerState({0: conv}, {0: recurrent}))
    return OwnerState(layers)


def get_tensor(fixture: Fixture, coordinate: Mapping[str, Any]) -> torch.Tensor:
    if coordinate["owner_kind"] == "persistent":
        owner = fixture.persistent
    else:
        owner = fixture.requests[int(coordinate["request_index"])]
    layer = owner.layers[int(coordinate["layer_index"])]
    states = (
        layer.conv_states
        if coordinate["state_family"] == "conv"
        else layer.recurrent_states
    )
    return states[int(coordinate["state_index"])]


def set_tensor(
    fixture: Fixture, coordinate: Mapping[str, Any], tensor: torch.Tensor
) -> None:
    if coordinate["owner_kind"] == "persistent":
        owner = fixture.persistent
    else:
        owner = fixture.requests[int(coordinate["request_index"])]
    layer = owner.layers[int(coordinate["layer_index"])]
    states = (
        layer.conv_states
        if coordinate["state_family"] == "conv"
        else layer.recurrent_states
    )
    states[int(coordinate["state_index"])] = tensor


def build_fixture(
    preregistration: Mapping[str, Any], fault_spec: Mapping[str, Any]
) -> Fixture:
    fixture = Fixture(
        persistent=_make_owner(preregistration, "persistent", None),
        requests=[
            _make_owner(preregistration, "request", request_index)
            for request_index in range(int(preregistration["fixture"]["resident_count"]))
        ],
        stale_handles={},
    )
    if fault_spec["kind"] == "stale_handle_after_rebind":
        coordinate = fault_spec["targets"][0]
        stale = get_tensor(fixture, coordinate)
        refreshed = (stale.detach().clone() + 4096.0).share_memory_()
        set_tensor(fixture, coordinate, refreshed)
        fixture.stale_handles[coordinate_key(coordinate)] = stale
    return fixture

