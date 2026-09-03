from __future__ import annotations

from typing import Any, Mapping

from .fixture import Fixture
from .protocol import require, validate_manifest


def bind_oracle_items(
    manifest: Mapping[str, Any], fixture: Fixture
) -> list[dict[str, Any]]:
    """Independent pre-injection traversal used only by the oracle authority."""

    rows = validate_manifest(manifest)
    items: list[dict[str, Any]] = []
    for slot, coordinate in sorted(rows.items()):
        owner_kind = coordinate["owner_kind"]
        if owner_kind == "persistent":
            require(coordinate["request_index"] is None, "oracle persistent index drift")
            owner = fixture.persistent
        else:
            request_index = int(coordinate["request_index"])
            require(0 <= request_index < len(fixture.requests), "oracle request drift")
            owner = fixture.requests[request_index]
        layer = owner.layers[int(coordinate["layer_index"])]
        family = coordinate["state_family"]
        if family == "conv":
            states = layer.conv_states
        elif family == "recurrent":
            states = layer.recurrent_states
        else:
            raise RuntimeError("oracle family drift")
        tensor = states[int(coordinate["state_index"])]
        items.append({"slot_id": slot, "tensor": tensor})
    return items

