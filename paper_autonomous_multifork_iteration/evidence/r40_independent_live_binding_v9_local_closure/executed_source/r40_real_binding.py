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

    def __init__(self, persistent: Any, selected: Sequence[Mapping[str, Any]], layer_indices: Sequence[int] | None = None) -> None:
        self.selected = [dict(row) for row in selected]
        source_coordinates = {coordinate_key(row) for row in self.selected}
        self.source = {}
        for layer_index, family, state_index in sorted(source_coordinates):
            coordinate = {"layer_index": layer_index, "state_family": family, "state_index": state_index}
            tensor = tensor_at(persistent, coordinate)
            self.source[(layer_index, family, state_index)] = {
                "content_sha256": digest(tensor), "storage_key": storage_key(tensor),
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                "tensor": tensor,
            }
        self.persistent = persistent
        self.group: Any = None
        self.initial: dict[tuple[int, int, str, int], tuple[str, int, int]] = {}
        self.initial_tensors: dict[tuple[int, int, str, int], torch.Tensor] = {}
        self.phase_order: list[str] = []
        self.layer_indices = tuple(layer_indices) if layer_indices is not None else tuple(sorted({int(r["layer_index"]) for r in selected}))
        self.lineage_receipt: Mapping[str, Any] | None = None

    def attach_lineage_receipt(self, receipt: Mapping[str, Any]) -> None:
        require(self.group is not None, "lineage receipt before built group")
        require(receipt.get("schema_version") == "forkaudit-r40-clone-lineage-v1", "lineage receipt schema drift")
        require(receipt.get("passed") is True and receipt.get("source_independent") is True, "lineage receipt gate unsatisfied")
        edges = receipt.get("edges")
        require(isinstance(edges, list), "lineage edges missing")
        expected = [row for row in self.selected if row["owner_kind"] == "request"]
        require(len(edges) == len(expected), "lineage edge cardinality drift")
        by_key = {(int(edge["request_index"]), int(edge["layer_index"]), str(edge["state_family"]), int(edge["state_index"])): edge for edge in edges}
        require(len(by_key) == len(edges), "duplicate lineage semantic edge")
        for row in expected:
            key=(int(row["request_index"]),*coordinate_key(row)); edge=by_key.get(key)
            require(isinstance(edge, dict), "selected lineage edge missing")
            source=self.source[coordinate_key(row)]; dest=tensor_at(self.group.requests[int(row["request_index"])],row)
            require(tuple(edge["source_storage_key"]) == source["storage_key"], "lineage source interval/storage mismatch")
            require(tuple(edge["destination_storage_key"]) == storage_key(dest), "lineage destination live-object/storage mismatch")
            require(edge["source_content_sha256"] == source["content_sha256"] and edge["destination_content_sha256"] == digest(dest), "lineage content binding mismatch")
            require(edge["clone_operator"] == "aten.clone.default", "lineage operator drift")
        self.lineage_receipt = dict(receipt)

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
            self.initial_tensors[key] = tensor
            request_tensors.append((key, tensor))
        for (left_key, left), (right_key, right) in combinations(request_tensors, 2):
            if left_key[0] != right_key[0]:
                require(storage_key(left) != storage_key(right), "real builder peer alias")

    def verify_serialized_phase(self, gdn: Mapping[str, Any], phase: str, completed_call_args: Sequence[int] | None = None) -> dict[str, Any]:
        require(self.group is not None, "phase before real group")
        require(gdn.get("phase") == phase, "returned phase drift")
        rows = gdn.get("storage_witness", {}).get("rows")
        require(isinstance(rows, list), "actual gdn_phase_witness rows missing")
        witness = gdn["storage_witness"]
        completed_gdn = list(witness.get("completed_request_indices", []))
        require(completed_call_args is not None and completed_gdn == list(completed_call_args), "completed set GDN/call-args mismatch")
        exact_fields = {"owner_kind","request_index","layer_index","state_family","state_index","shape","stride","storage_offset","dtype","device","storage_nbytes","tensor_nbytes","byte_start","byte_end_exclusive","content_sha256","storage_id"}
        require(all(set(row) == exact_fields for row in rows), "serializer row exact schema drift")
        expected_count = (len(self.group.requests)+1)*len(self.layer_indices)*2
        require(len(rows) == expected_count, "serializer row cardinality drift")
        semantic_keys=[(r["owner_kind"],r["request_index"],r["layer_index"],r["state_family"],r["state_index"]) for r in rows]
        require(len(set(semantic_keys)) == len(semantic_keys), "serializer duplicate semantic row")
        # Re-enumerate every live row in the serializer's canonical order and
        # derive normalized storage IDs from live storage keys. This detects a
        # forged-but-well-formed storage_id as well as selected-row tampering.
        normalized: dict[tuple[str, int, int], str] = {}
        storage_owner: dict[tuple[str, int, int], tuple[str, int | None]] = {}
        for row in rows:
            owner = self.persistent if row["owner_kind"] == "persistent" else self.group.requests[int(row["request_index"])]
            tensor = tensor_at(owner, row)
            key = storage_key(tensor)
            if key not in normalized:
                normalized[key] = f"storage-{len(normalized):04d}"
                storage_owner[key] = (row["owner_kind"], row["request_index"])
            else:
                require(storage_owner[key] == (row["owner_kind"], row["request_index"]), "completed/incomplete requests alias or request/base owners alias")
            require(row["storage_id"] == normalized[key], "actual serializer normalized storage_id/live-storage mismatch")
            require(row["content_sha256"] == digest(tensor), "actual serializer content/live-object mismatch")
            require(row["shape"] == list(tensor.shape) and row["dtype"] == str(tensor.dtype), "actual serializer descriptor mismatch")
            require(row["stride"] == list(tensor.stride()) and row["storage_offset"] == int(tensor.storage_offset()), "serializer stride/offset mismatch")
            require(row["storage_nbytes"] == int(tensor.untyped_storage().nbytes()) and row["tensor_nbytes"] == tensor.numel()*tensor.element_size(), "serializer nbytes mismatch")
            start=int(tensor.storage_offset())*tensor.element_size(); end=start
            for size,stride in zip(tensor.shape,tensor.stride()): end += (int(size)-1)*int(stride)*tensor.element_size()
            end += tensor.element_size()
            require(row["byte_start"] == start and row["byte_end_exclusive"] == end, "serializer byte interval mismatch")
        by_key = {(row["owner_kind"], row["request_index"], *coordinate_key(row)): row for row in rows}
        for coordinate, frozen in self.source.items():
            current = tensor_at(self.persistent, {"layer_index":coordinate[0], "state_family":coordinate[1], "state_index":coordinate[2]})
            require(current is frozen["tensor"], "persistent source object changed after freeze")
            require(storage_key(current) == frozen["storage_key"], "persistent source storage changed after freeze")
            require(digest(current) == frozen["content_sha256"], "persistent source content changed after freeze")
            require(list(current.shape) == frozen["shape"] and str(current.dtype) == frozen["dtype"], "persistent source descriptor changed after freeze")
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
                completed = int(request_index) in set(completed_gdn)
                if completed:
                    require(storage_key(tensor) != initial, "real request mapping retained stale pre-transition handle")
                    require(tensor is not self.initial_tensors[(int(request_index), *coordinate_key(selected))], "completed request retained initial tensor object")
                else:
                    require(storage_key(tensor) == initial, "incomplete request mapping changed early")
                    require(tensor is self.initial_tensors[(int(request_index), *coordinate_key(selected))], "incomplete request tensor object changed early")
                source_row = by_key[("persistent", None, *coordinate_key(selected))]
                require(row["storage_id"] != source_row["storage_id"], "actual serializer request/base role alias")
            checked += 1
        request_live = []
        for selected in self.selected:
            if selected["owner_kind"] == "request":
                request_live.append((int(selected["request_index"]), coordinate_key(selected), tensor_at(self.group.requests[int(selected["request_index"])], selected)))
        for (left_owner, left_coord, left), (right_owner, right_coord, right) in combinations(request_live, 2):
            if left_owner != right_owner:
                require(storage_key(left) != storage_key(right), "completed/incomplete or peer requests alias")
        self.phase_order.append(phase)
        return {"phase": phase, "selected_rows_verified": checked, "actual_storage_rows_verified": len(rows), "actual_serializer_compared": True}


__all__ = ["ActualBindingVerifier", "RealBindingError", "digest", "storage_key"]
