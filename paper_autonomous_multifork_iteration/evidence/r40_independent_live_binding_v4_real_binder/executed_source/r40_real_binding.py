from __future__ import annotations

"""Independent checks over the real group mapping and serialized phase rows."""

import hashlib
from itertools import combinations
from typing import Any, Mapping, Sequence

import torch


class RealBindingError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise RealBindingError(message)


def digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


def tensor_at(owner: Any, coordinate: Mapping[str, Any]) -> torch.Tensor:
    layer = owner.layers[int(coordinate["layer_index"])]
    mapping = layer.conv_states if coordinate["state_family"] == "conv" else layer.recurrent_states
    value = mapping[int(coordinate["state_index"])]
    require(isinstance(value, torch.Tensor), "selected live value is not tensor")
    return value


def coordinate_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return int(row["layer_index"]), str(row["state_family"]), int(row["state_index"])


class ActualBindingVerifier:
    """Freeze source semantics before build; verify group and actual serializer."""

    def __init__(self, persistent: Any, selected: Sequence[Mapping[str, Any]]) -> None:
        self.selected = [dict(row) for row in selected]
        source_coordinates = {coordinate_key(row) for row in self.selected}
        self.source = {}
        for layer_index, family, state_index in sorted(source_coordinates):
            coordinate = {"layer_index": layer_index, "state_family": family, "state_index": state_index}
            tensor = tensor_at(persistent, coordinate)
            self.source[(layer_index, family, state_index)] = {
                "content_sha256": digest(tensor), "storage_key": storage_key(tensor),
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            }
        self.persistent = persistent
        self.group: Any = None
        self.initial: dict[tuple[int, int, str, int], tuple[str, int, int]] = {}
        self.phase_order: list[str] = []

    def verify_built_group(self, group: Any) -> None:
        require(self.group is None, "group verified twice")
        self.group = group
        request_tensors: list[tuple[tuple[int, int, str, int], torch.Tensor]] = []
        for row in self.selected:
            if row["owner_kind"] != "request":
                continue
            request_index = int(row["request_index"])
            tensor = tensor_at(group.requests[request_index], row)
            source = self.source[coordinate_key(row)]
            require(digest(tensor) == source["content_sha256"], "real builder coordinate/content mismatch")
            require(list(tensor.shape) == source["shape"] and str(tensor.dtype) == source["dtype"], "real builder descriptor mismatch")
            require(storage_key(tensor) != source["storage_key"], "real builder request/base alias")
            key = (request_index, *coordinate_key(row))
            self.initial[key] = storage_key(tensor)
            request_tensors.append((key, tensor))
        for (left_key, left), (right_key, right) in combinations(request_tensors, 2):
            if left_key[0] != right_key[0]:
                require(storage_key(left) != storage_key(right), "real builder peer alias")

    def verify_serialized_phase(self, gdn: Mapping[str, Any], phase: str) -> dict[str, Any]:
        require(self.group is not None, "phase before real group")
        require(gdn.get("phase") == phase, "returned phase drift")
        rows = gdn.get("storage_witness", {}).get("rows")
        require(isinstance(rows, list), "actual gdn_phase_witness rows missing")
        by_key = {(row["owner_kind"], row["request_index"], *coordinate_key(row)): row for row in rows}
        checked = 0
        for selected in self.selected:
            owner_kind = selected["owner_kind"]
            request_index = selected["request_index"]
            owner = self.persistent if owner_kind == "persistent" else self.group.requests[int(request_index)]
            tensor = tensor_at(owner, selected)
            row = by_key.get((owner_kind, request_index, *coordinate_key(selected)))
            require(isinstance(row, dict), "selected row absent from actual serializer")
            require(row["content_sha256"] == digest(tensor), "actual serializer content/live-object mismatch")
            require(row["shape"] == list(tensor.shape) and row["dtype"] == str(tensor.dtype), "actual serializer descriptor mismatch")
            if owner_kind == "request":
                initial = self.initial[(int(request_index), *coordinate_key(selected))]
                completed = phase == "post_generation" or (phase == "post_transition" and int(request_index) == 0)
                if completed:
                    require(storage_key(tensor) != initial, "real request mapping retained stale pre-transition handle")
                else:
                    require(storage_key(tensor) == initial, "incomplete request mapping changed early")
                source_row = by_key[("persistent", None, *coordinate_key(selected))]
                require(row["storage_id"] != source_row["storage_id"], "actual serializer request/base role alias")
            checked += 1
        self.phase_order.append(phase)
        return {"phase": phase, "selected_rows_verified": checked, "actual_serializer_compared": True}


__all__ = ["ActualBindingVerifier", "RealBindingError", "digest", "storage_key"]
