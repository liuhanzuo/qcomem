from __future__ import annotations

"""Independent checks over the actual borrowed group and production serializer."""

import hashlib
import json
from itertools import combinations
from typing import Any, Mapping, Sequence

import torch


BORROWED_POLICY = "borrow-immutable-base-functional-rebind"
MATERIALIZED_POLICY = "materialize-request-base-functional-rebind"
PHASES = ("setup_pre_transition", "post_transition", "post_generation")


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


def byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    low = high = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        delta = (int(size) - 1) * int(stride)
        low += min(delta, 0)
        high += max(delta, 0)
    return low * tensor.element_size(), (high + 1) * tensor.element_size()


def tensor_at(owner: Any, coordinate: Mapping[str, Any]) -> torch.Tensor:
    layer = owner.layers[int(coordinate["layer_index"])]
    mapping = layer.conv_states if coordinate["state_family"] == "conv" else layer.recurrent_states
    value = mapping[int(coordinate["state_index"])]
    require(isinstance(value, torch.Tensor), "selected live value is not tensor")
    return value


def coordinate_key(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return int(row["layer_index"]), str(row["state_family"]), int(row["state_index"])


def live_snapshot(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "object": tensor,
        "storage": storage_key(tensor),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_nbytes": int(tensor.untyped_storage().nbytes()),
        "tensor_nbytes": tensor.numel() * tensor.element_size(),
        "interval": byte_interval(tensor),
        "content_sha256": digest(tensor),
    }


def same_snapshot(tensor: torch.Tensor, snapshot: Mapping[str, Any]) -> bool:
    now = live_snapshot(tensor)
    return tensor is snapshot["object"] and all(now[key] == snapshot[key] for key in now if key != "object")


def receipt_snapshot(tensor: torch.Tensor) -> dict[str, Any]:
    snapshot = live_snapshot(tensor)
    return {
        "object_id": id(tensor),
        "storage_key": list(snapshot["storage"]),
        "descriptor": {
            "shape": snapshot["shape"],
            "stride": snapshot["stride"],
            "storage_offset": snapshot["storage_offset"],
            "dtype": snapshot["dtype"],
            "device": snapshot["device"],
            "storage_nbytes": snapshot["storage_nbytes"],
            "tensor_nbytes": snapshot["tensor_nbytes"],
            "byte_interval": list(snapshot["interval"]),
        },
        "content_sha256": snapshot["content_sha256"],
    }


def canonical_compact_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    """Derive the only descriptor authorized after functional rebinding."""
    shape = [int(size) for size in tensor.shape]
    stride = [0] * len(shape)
    running = 1
    for index in range(len(shape) - 1, -1, -1):
        stride[index] = running
        running *= shape[index]
    tensor_nbytes = int(tensor.numel()) * int(tensor.element_size())
    return {
        "shape": shape,
        "stride": stride,
        "storage_offset": 0,
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "storage_nbytes": tensor_nbytes,
        "tensor_nbytes": tensor_nbytes,
        "interval": (0, tensor_nbytes),
    }


class ActualBindingVerifier:
    """Freeze persistent sources before the real builder and follow every rebind."""

    def __init__(
        self,
        persistent: Any,
        selected: Sequence[Mapping[str, Any]],
        layer_indices: Sequence[int] | None = None,
        setup_policy: str = MATERIALIZED_POLICY,
    ) -> None:
        require(setup_policy in {BORROWED_POLICY, MATERIALIZED_POLICY}, "unknown setup policy")
        self.setup_policy = setup_policy
        self.selected = [dict(row) for row in selected]
        self.layer_indices = tuple(layer_indices) if layer_indices is not None else tuple(sorted({int(row["layer_index"]) for row in selected}))
        coordinates = {(int(layer), family, 0) for layer in self.layer_indices for family in ("conv", "recurrent")}
        self.source: dict[tuple[int, str, int], dict[str, Any]] = {}
        for layer_index, family, state_index in sorted(coordinates):
            coordinate = {"layer_index": layer_index, "state_family": family, "state_index": state_index}
            tensor = tensor_at(persistent, coordinate)
            snapshot = live_snapshot(tensor)
            self.source[(layer_index, family, state_index)] = {
                "content_sha256": snapshot["content_sha256"], "storage_key": snapshot["storage"],
                "shape": snapshot["shape"], "stride": snapshot["stride"], "storage_offset": snapshot["storage_offset"],
                "dtype": snapshot["dtype"], "device": snapshot["device"], "storage_nbytes": snapshot["storage_nbytes"],
                "tensor_nbytes": snapshot["tensor_nbytes"], "byte_interval": snapshot["interval"], "tensor": tensor,
            }
        self.persistent = persistent
        self.group: Any = None
        self.initial_tensors: dict[tuple[int, int, str, int], torch.Tensor] = {}
        self.latest_tensors: dict[tuple[int, int, str, int], torch.Tensor] = {}
        self.latest_snapshots: dict[tuple[int, int, str, int], dict[str, Any]] = {}
        self.persistent_snapshots = {coord: live_snapshot(row["tensor"]) for coord, row in self.source.items()}
        self.authorized_descriptors: dict[tuple[int, str, int], dict[str, Any]] = {}
        self.historical_tensors: list[torch.Tensor] = [row["tensor"] for row in self.source.values()]
        self.historical_object_ids = {id(tensor) for tensor in self.historical_tensors}
        self.historical_storage_keys = {storage_key(tensor) for tensor in self.historical_tensors}
        self.completed_seen: set[int] = set()
        self.generation_ledger: list[dict[str, Any]] = []
        self.phase_order: list[str] = []
        self.lineage_receipt: Mapping[str, Any] | None = None

    @staticmethod
    def _descriptor(snapshot: Mapping[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in snapshot.items() if key not in {"object", "storage", "content_sha256"}}

    def attach_lineage_receipt(self, receipt: Mapping[str, Any]) -> None:
        del receipt
        raise RealBindingError("mapping lineage receipts are forbidden; opaque mode capability required")

    def attach_lineage_capability(self, capability: Any) -> None:
        from r40_passive_clone_lineage import _LineageCapability
        require(self.group is not None, "lineage capability before built group")
        require(type(capability) is _LineageCapability, "caller-forged lineage capability rejected")
        capability.consume_into(self)

    def _accept_lineage_edges(self, bound: Sequence[Any]) -> None:
        expected = [row for row in self.selected if row["owner_kind"] == "request"]
        require(len(bound) == len(expected), "lineage binding cardinality drift")
        by_key = {(int(row["request_index"]), *coordinate_key(row)): (event, destination) for row, event, destination in bound}
        require(len(by_key) == len(bound), "duplicate lineage semantic binding")
        clone_edges = exact_aliases = 0
        for row in expected:
            key = (int(row["request_index"]), *coordinate_key(row))
            require(key in by_key, "selected lineage binding missing")
            event, captured_destination = by_key[key]
            source = self.source[coordinate_key(row)]
            destination = tensor_at(self.group.requests[int(row["request_index"])], row)
            require(captured_destination is destination, "lineage destination exact live object mismatch")
            if self.setup_policy == BORROWED_POLICY:
                require(event is None, "borrowed setup unexpectedly used clone lineage")
                require(destination is source["tensor"] and storage_key(destination) == source["storage_key"], "borrowed selected binding is not exact persistent alias")
                require(digest(destination) == source["content_sha256"], "borrowed selected alias content drift")
                exact_aliases += 1
            else:
                require(event is not None and event.source is source["tensor"] and event.destination is destination, "materialized lineage source/destination mismatch")
                require(event.operator == "aten.clone.default", "lineage operator drift")
                require(digest(event.source) == source["content_sha256"] and digest(destination) == digest(event.destination), "lineage content binding mismatch")
                clone_edges += 1
        self.lineage_receipt = {
            "opaque_capability_consumed": True,
            "binding_policy": "borrowed-exact-persistent-alias" if self.setup_policy == BORROWED_POLICY else "materialized-direct-clone",
            "selected_binding_count": len(bound), "selected_exact_alias_count": exact_aliases, "selected_clone_edge_count": clone_edges,
        }

    def verify_built_group(self, group: Any) -> None:
        require(self.group is None, "group verified twice")
        self.group = group
        request_tensors: list[tuple[tuple[int, int, str, int], torch.Tensor]] = []
        for request_index, request in enumerate(group.requests):
            for coord, source in sorted(self.source.items()):
                row = {"layer_index": coord[0], "state_family": coord[1], "state_index": coord[2]}
                tensor = tensor_at(request, row)
                snapshot = live_snapshot(tensor)
                require(snapshot["content_sha256"] == source["content_sha256"], "real builder coordinate/content mismatch")
                require(self._descriptor(snapshot) == self._descriptor(self.persistent_snapshots[coord]), "real builder descriptor mismatch")
                if self.setup_policy == BORROWED_POLICY:
                    require(tensor is source["tensor"] and snapshot["storage"] == source["storage_key"], "real borrowed builder is not exact persistent alias")
                else:
                    require(tensor is not source["tensor"] and snapshot["storage"] != source["storage_key"], "real builder request/base alias")
                key = (request_index, *coord)
                self.initial_tensors[key] = tensor
                self.latest_tensors[key] = tensor
                self.latest_snapshots[key] = snapshot
                descriptor = canonical_compact_descriptor(source["tensor"]) if self.setup_policy == BORROWED_POLICY else self._descriptor(snapshot)
                if coord in self.authorized_descriptors:
                    require(descriptor == self.authorized_descriptors[coord], "initial request descriptor differs across requests")
                else:
                    self.authorized_descriptors[coord] = descriptor
                if self.setup_policy == MATERIALIZED_POLICY:
                    self.historical_tensors.append(tensor)
                    self.historical_object_ids.add(id(tensor))
                    self.historical_storage_keys.add(snapshot["storage"])
                    request_tensors.append((key, tensor))
        if self.setup_policy == MATERIALIZED_POLICY:
            for (left_key, left), (right_key, right) in combinations(request_tensors, 2):
                if left_key[0] != right_key[0]:
                    require(storage_key(left) != storage_key(right), "real builder peer alias")

    def _verify_live_ownership(self) -> tuple[int, int]:
        completed_storage: set[tuple[str, int, int]] = set()
        private_rows = borrowed_rows = 0
        for request_index, request in enumerate(self.group.requests):
            for coord, source in sorted(self.source.items()):
                row = {"layer_index": coord[0], "state_family": coord[1], "state_index": coord[2]}
                tensor = tensor_at(request, row)
                key = (request_index, *coord)
                require(tensor is self.latest_tensors[key] and storage_key(tensor) == storage_key(self.latest_tensors[key]), "live tensor differs from ordered rebind endpoint")
                if request_index in self.completed_seen:
                    require(tensor is not source["tensor"] and storage_key(tensor) != source["storage_key"], "completed request aliases persistent base")
                    require(storage_key(tensor) not in completed_storage, "completed request/peer coordinates alias")
                    completed_storage.add(storage_key(tensor)); private_rows += 1
                else:
                    require(tensor is source["tensor"] and same_snapshot(tensor, self.persistent_snapshots[coord]), "incomplete request is not exact read-only persistent alias")
                    borrowed_rows += 1
        return private_rows, borrowed_rows

    def verify_serialized_phase(self, gdn: Mapping[str, Any], phase: str, completed_call_args: Sequence[int] | None = None) -> dict[str, Any]:
        require(self.group is not None, "phase before real group")
        require(len(self.phase_order) < len(PHASES) and phase == PHASES[len(self.phase_order)], "phase order drift")
        require(gdn.get("phase") == phase, "returned phase drift")
        rows = gdn.get("storage_witness", {}).get("rows")
        require(isinstance(rows, list), "actual gdn_phase_witness rows missing")
        completed_gdn = list(gdn["storage_witness"].get("completed_request_indices", []))
        require(type(gdn["storage_witness"].get("completed_request_indices")) is list and all(type(index) is int for index in completed_gdn), "completed indices exact type drift")
        require(type(completed_call_args) is list and all(type(index) is int for index in completed_call_args), "completed call-args exact type drift")
        derived_completed = [] if phase == PHASES[0] else ([0] if phase == PHASES[1] else list(range(len(self.group.requests))))
        require(completed_gdn == list(completed_call_args) == derived_completed and self.completed_seen == set(derived_completed), "completed set independent lifecycle/GDN/call-args mismatch")
        exact_fields = {"owner_kind", "request_index", "layer_index", "state_family", "state_index", "shape", "stride", "storage_offset", "dtype", "device", "storage_nbytes", "tensor_nbytes", "byte_start", "byte_end_exclusive", "content_sha256", "storage_id"}
        require(all(set(row) == exact_fields for row in rows), "serializer row exact schema drift")
        require(all(type(row["shape"]) is list and type(row["stride"]) is list and all(type(value) is int for value in row["shape"] + row["stride"]) and all(type(row[key]) is int for key in ("layer_index", "state_index", "storage_offset", "storage_nbytes", "tensor_nbytes", "byte_start", "byte_end_exclusive")) and type(row["owner_kind"]) is str and type(row["state_family"]) is str and type(row["dtype"]) is str and type(row["device"]) is str and type(row["content_sha256"]) is str and type(row["storage_id"]) is str for row in rows), "serializer exact field type drift")
        expected_count = (len(self.group.requests) + 1) * len(self.layer_indices) * 2
        require(len(rows) == expected_count, "serializer row cardinality drift")
        semantic_keys = [(row["owner_kind"], row["request_index"], row["layer_index"], row["state_family"], row["state_index"]) for row in rows]
        require(len(set(semantic_keys)) == len(semantic_keys), "serializer duplicate semantic row")
        expected_keys = []
        for owner_kind, request_index in [("persistent", None)] + [("request", index) for index in range(len(self.group.requests))]:
            for layer in self.layer_indices:
                for family in ("conv", "recurrent"):
                    expected_keys.append((owner_kind, request_index, int(layer), family, 0))
        require(semantic_keys == expected_keys, "serializer semantic row universe/order drift")
        normalized: dict[tuple[str, int, int], str] = {}
        by_key = {}
        for row in rows:
            require(row["owner_kind"] in {"persistent", "request"} and row["state_family"] in {"conv", "recurrent"}, "serializer semantic value drift")
            owner = self.persistent if row["owner_kind"] == "persistent" else self.group.requests[int(row["request_index"])]
            tensor = tensor_at(owner, row); key = storage_key(tensor)
            if key not in normalized: normalized[key] = f"storage-{len(normalized):04d}"
            require(row["storage_id"] == normalized[key], "actual serializer normalized storage_id/live-storage mismatch")
            require(row["content_sha256"] == digest(tensor), "actual serializer content/live-object mismatch")
            require(row["shape"] == list(tensor.shape) and row["stride"] == list(tensor.stride()) and row["storage_offset"] == int(tensor.storage_offset()) and row["dtype"] == str(tensor.dtype) and row["device"] == str(tensor.device), "actual serializer descriptor mismatch")
            require(row["storage_nbytes"] == int(tensor.untyped_storage().nbytes()) and row["tensor_nbytes"] == tensor.numel() * tensor.element_size(), "serializer nbytes mismatch")
            start, end = byte_interval(tensor)
            require(row["byte_start"] == start and row["byte_end_exclusive"] == end, "serializer byte interval mismatch")
            by_key[(row["owner_kind"], row["request_index"], *coordinate_key(row))] = row
        for coord, frozen in self.source.items():
            current = tensor_at(self.persistent, {"layer_index": coord[0], "state_family": coord[1], "state_index": coord[2]})
            require(current is frozen["tensor"] and same_snapshot(current, self.persistent_snapshots[coord]), "persistent source object/storage/descriptor/interval/content drift")
        expected_calls = {PHASES[0]: 0, PHASES[1]: 1, PHASES[2]: len(self.group.requests) * 8}[phase]
        require(len(self.generation_ledger) == expected_calls, "generation rebind ledger count/phase drift")
        private_rows, borrowed_rows = self._verify_live_ownership()
        for request_index in range(len(self.group.requests)):
            for coord in sorted(self.source):
                request_row = by_key[("request", request_index, *coord)]
                source_row = by_key[("persistent", None, *coord)]
                if request_index in self.completed_seen:
                    require(request_row["storage_id"] != source_row["storage_id"], "completed serializer row aliases base")
                else:
                    require(request_row["storage_id"] == source_row["storage_id"], "incomplete serializer row is not exact base alias")
        self.phase_order.append(phase)
        counts = [sum(1 for event in self.generation_ledger if event["request_index"] == index) for index in range(len(self.group.requests))]
        isolation = all(event["target_all_new"] and event["non_target_unchanged"] and event["persistent_unchanged"] and event["completed_private"] and event["incomplete_exact_alias"] for event in self.generation_ledger)
        return {"phase": phase, "selected_rows_verified": len(self.selected), "full_live_rows_verified": expected_count, "generation_calls_verified": len(self.generation_ledger), "functional_rebind_edges_verified": len(self.generation_ledger) * len(self.source), "request_rebind_counts": counts, "per_call_isolation_verified": isolation, "private_request_rows_verified": private_rows, "borrowed_request_rows_verified": borrowed_rows, "actual_storage_rows_verified": len(rows), "actual_serializer_compared": True}

    def observe_generation_step(self, round_index: int, request_index: int) -> None:
        require(type(round_index) is int and type(request_index) is int, "generation schedule exact type drift")
        call_index = len(self.generation_ledger); request_count = len(self.group.requests)
        require(call_index < 8 * request_count and round_index == call_index // request_count and request_index == call_index % request_count, "generation round-robin schedule/order drift")
        prior_object_ids = set(self.historical_object_ids); prior_storage = set(self.historical_storage_keys)
        for coord in self.source:
            current = tensor_at(self.persistent, {"layer_index": coord[0], "state_family": coord[1], "state_index": coord[2]})
            require(same_snapshot(current, self.persistent_snapshots[coord]), "callback persistent object/storage/descriptor/interval/content drift")
        for peer_index, peer in enumerate(self.group.requests):
            if peer_index == request_index: continue
            for coord in sorted(self.source):
                current = tensor_at(peer, {"layer_index": coord[0], "state_family": coord[1], "state_index": coord[2]})
                require(same_snapshot(current, self.latest_snapshots[(peer_index, *coord)]), "non-target request changed before its scheduled call")
        edges = []; pending = []; request = self.group.requests[request_index]
        version = sum(1 for event in self.generation_ledger if event["request_index"] == request_index) + 1
        for coord in sorted(self.source):
            row = {"layer_index": coord[0], "state_family": coord[1], "state_index": coord[2]}
            current = tensor_at(request, row); key = (request_index, *coord); previous = self.latest_tensors[key]
            current_snapshot = live_snapshot(current); descriptor = self._descriptor(current_snapshot)
            require(id(current) not in prior_object_ids and current_snapshot["storage"] not in prior_storage, "functional rebind endpoint is not globally fresh; rotation/permutation/reuse detected")
            require(descriptor == self.authorized_descriptors[coord], f"functional rebind descriptor/offset/interval unauthorized; first_coord={coord!r}")
            edges.append({"coordinate": list(coord), "version": version, "pre": receipt_snapshot(previous), "post": receipt_snapshot(current), "new_tensor_object": True, "new_storage": True, "descriptor_authorized": True, "content_recorded": True})
            pending.append((key, current, current_snapshot))
        require(len(edges) == len(self.source), "generation rebind edge cardinality drift")
        for key, current, snapshot in pending:
            self.latest_tensors[key] = current; self.latest_snapshots[key] = snapshot; self.historical_tensors.append(current); self.historical_object_ids.add(id(current)); self.historical_storage_keys.add(snapshot["storage"])
        self.completed_seen.add(request_index)
        private_rows, borrowed_rows = self._verify_live_ownership()
        require(private_rows == len(self.completed_seen) * len(self.source) and borrowed_rows == (request_count - len(self.completed_seen)) * len(self.source), "callback ownership cardinality drift")
        self.generation_ledger.append({"call_index": call_index, "round_index": round_index, "request_index": request_index, "request_version": version, "edge_count": len(edges), "edges": edges, "completed_request_indices_after_call": sorted(self.completed_seen), "private_request_rows_after_call": private_rows, "borrowed_request_rows_after_call": borrowed_rows, "target_all_new": True, "non_target_unchanged": True, "persistent_unchanged": True, "completed_private": True, "incomplete_exact_alias": True})

    def functional_rebind_receipt(self) -> dict[str, Any]:
        value = {"schema_version": "forkaudit-r40-v31-functional-rebind-ledger-v1", "call_count": len(self.generation_ledger), "edge_count": sum(event["edge_count"] for event in self.generation_ledger), "edges_per_call": len(self.source), "all_new_tensor_objects": all(edge["new_tensor_object"] is True for event in self.generation_ledger for edge in event["edges"]), "all_new_storages": all(edge["new_storage"] is True for event in self.generation_ledger for edge in event["edges"]), "all_descriptors_authorized": all(edge["descriptor_authorized"] is True for event in self.generation_ledger for edge in event["edges"]), "all_contents_recorded": all(edge["content_recorded"] is True for event in self.generation_ledger for edge in event["edges"]), "calls": self.generation_ledger, "ledger_sha256": None}
        value["ledger_sha256"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return value


__all__ = ["ActualBindingVerifier", "BORROWED_POLICY", "MATERIALIZED_POLICY", "RealBindingError", "byte_interval", "canonical_compact_descriptor", "digest", "receipt_snapshot", "storage_key"]
