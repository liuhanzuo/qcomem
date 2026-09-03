from __future__ import annotations

"""Passive operator-level provenance for materialized GDN request bases.

The observer records strong Python tensor handles and storage intervals at the
actual ``aten.clone``/``aten.copy_`` dispatch boundary.  It does not tag tensor
values, mutate source or destination tensors, or consume a builder-emitted
coordinate map.
"""

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch.utils._python_dispatch import TorchDispatchMode


Coordinate = tuple[int, str, int]


class LineageViolation(RuntimeError):
    """Raised when a live destination cannot be bound to its expected source."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LineageViolation(message)


def coordinate_key(value: Mapping[str, Any] | Coordinate) -> Coordinate:
    if isinstance(value, tuple):
        require(len(value) == 3, "coordinate tuple must have three fields")
        layer_index, family, state_index = value
    else:
        layer_index = value["layer_index"]
        family = value["state_family"]
        state_index = value["state_index"]
    family = str(family)
    require(family in {"conv", "recurrent"}, "unknown state family")
    return int(layer_index), family, int(state_index)


def tensor_at(owner: Any, coordinate: Mapping[str, Any] | Coordinate) -> torch.Tensor:
    layer_index, family, state_index = coordinate_key(coordinate)
    layer = owner.layers[layer_index]
    mapping = layer.conv_states if family == "conv" else layer.recurrent_states
    tensor = mapping[state_index]
    require(isinstance(tensor, torch.Tensor), "live coordinate is not a tensor")
    return tensor


def tensor_bytes_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes(order="C")
    return hashlib.sha256(raw).hexdigest()


def byte_interval(tensor: torch.Tensor) -> tuple[str, int, int]:
    """Return the inclusive-exclusive byte span touched by a dense strided tensor."""

    require(tensor.layout == torch.strided, "only dense strided tensors are supported")
    storage = tensor.untyped_storage()
    if tensor.numel() == 0:
        pointer = int(storage.data_ptr()) + int(tensor.storage_offset()) * tensor.element_size()
        return str(tensor.device), pointer, pointer
    low = high = int(tensor.storage_offset())
    for size, stride in zip(tensor.shape, tensor.stride()):
        delta = (int(size) - 1) * int(stride)
        low += min(delta, 0)
        high += max(delta, 0)
    base = int(storage.data_ptr())
    element_size = int(tensor.element_size())
    return str(tensor.device), base + low * element_size, base + (high + 1) * element_size


def storage_descriptor(tensor: torch.Tensor) -> tuple[Any, ...]:
    storage = tensor.untyped_storage()
    return (
        str(tensor.device),
        int(storage.data_ptr()),
        int(storage.nbytes()),
        int(tensor.storage_offset()),
        tuple(int(value) for value in tensor.shape),
        tuple(int(value) for value in tensor.stride()),
        str(tensor.dtype),
        byte_interval(tensor),
    )


@dataclass(frozen=True)
class SourceSnapshot:
    coordinate: Coordinate
    tensor: torch.Tensor
    tensor_id: int
    storage: tuple[Any, ...]
    content_sha256: str


class PersistentSourceRegistry:
    """Independently enumerate and freeze persistent semantic coordinates."""

    def __init__(self, persistent: Any, coordinates: Iterable[Mapping[str, Any] | Coordinate]):
        self.persistent = persistent
        normalized = tuple(sorted(coordinate_key(value) for value in coordinates))
        require(len(normalized) == len(set(normalized)), "duplicate source coordinate")
        require(normalized, "source registry is empty")
        self.by_coordinate: dict[Coordinate, SourceSnapshot] = {}
        self.by_tensor_id: dict[int, SourceSnapshot] = {}
        for coordinate in normalized:
            tensor = tensor_at(persistent, coordinate)
            snapshot = SourceSnapshot(
                coordinate=coordinate,
                tensor=tensor,
                tensor_id=id(tensor),
                storage=storage_descriptor(tensor),
                content_sha256=tensor_bytes_sha256(tensor),
            )
            require(
                snapshot.tensor_id not in self.by_tensor_id,
                "two semantic source coordinates share one tensor object; provenance is ambiguous",
            )
            self.by_coordinate[coordinate] = snapshot
            self.by_tensor_id[snapshot.tensor_id] = snapshot

    def verify_unchanged(self) -> None:
        for coordinate, frozen in self.by_coordinate.items():
            current = tensor_at(self.persistent, coordinate)
            require(current is frozen.tensor, f"persistent object changed at {coordinate}")
            require(
                storage_descriptor(current) == frozen.storage,
                f"persistent storage/descriptor changed at {coordinate}",
            )
            require(
                tensor_bytes_sha256(current) == frozen.content_sha256,
                f"persistent content changed at {coordinate}",
            )


@dataclass(frozen=True)
class LineageEvent:
    ordinal: int
    operator: str
    origin_coordinate: Coordinate
    source: torch.Tensor
    destination: torch.Tensor
    source_id: int
    destination_id: int
    source_storage: tuple[Any, ...]
    destination_storage: tuple[Any, ...]


class PassiveCloneLineageMode(TorchDispatchMode):
    """Record clone/copy edges rooted at pre-registered persistent tensors."""

    def __init__(self, registry: PersistentSourceRegistry):
        super().__init__()
        self.registry = registry
        self.events: list[LineageEvent] = []
        self._derived_origin_by_id: dict[int, Coordinate] = {}
        self._derived_tensor_by_id: dict[int, torch.Tensor] = {}

    def _origin(self, tensor: Any) -> Coordinate | None:
        if not isinstance(tensor, torch.Tensor):
            return None
        source = self.registry.by_tensor_id.get(id(tensor))
        if source is not None and source.tensor is tensor:
            return source.coordinate
        derived = self._derived_tensor_by_id.get(id(tensor))
        if derived is tensor:
            return self._derived_origin_by_id[id(tensor)]
        return None

    def _record(self, operator: str, source: torch.Tensor, destination: torch.Tensor) -> None:
        origin = self._origin(source)
        if origin is None:
            return
        event = LineageEvent(
            ordinal=len(self.events),
            operator=operator,
            origin_coordinate=origin,
            source=source,
            destination=destination,
            source_id=id(source),
            destination_id=id(destination),
            source_storage=storage_descriptor(source),
            destination_storage=storage_descriptor(destination),
        )
        self.events.append(event)
        self._derived_origin_by_id[id(destination)] = origin
        self._derived_tensor_by_id[id(destination)] = destination

    def __torch_dispatch__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        del types
        kwargs = kwargs or {}
        result = func(*args, **kwargs)
        operator = str(func)
        if operator == "aten.clone.default":
            if args and isinstance(args[0], torch.Tensor) and isinstance(result, torch.Tensor):
                self._record(operator, args[0], result)
        elif operator == "aten.copy_.default":
            if (
                len(args) >= 2
                and isinstance(args[0], torch.Tensor)
                and isinstance(args[1], torch.Tensor)
                and isinstance(result, torch.Tensor)
            ):
                self._record(operator, args[1], result)
        return result

    def verify_materialized(
        self,
        requests: Sequence[Any],
        coordinates: Iterable[Mapping[str, Any] | Coordinate],
        *,
        require_direct_clone: bool = True,
    ) -> dict[str, Any]:
        self.registry.verify_unchanged()
        normalized = tuple(sorted(coordinate_key(value) for value in coordinates))
        require(set(normalized) == set(self.registry.by_coordinate), "verification coordinate set drift")
        consumed: set[int] = set()
        destination_ids: set[int] = set()
        destination_storage: set[tuple[Any, ...]] = set()
        for request_index, request in enumerate(requests):
            for coordinate in normalized:
                destination = tensor_at(request, coordinate)
                expected_source = self.registry.by_coordinate[coordinate]
                require(destination is not expected_source.tensor, f"request/base object alias at {(request_index, coordinate)}")
                require(
                    storage_descriptor(destination) != expected_source.storage,
                    f"request/base storage alias at {(request_index, coordinate)}",
                )
                candidates = [event for event in self.events if event.destination is destination]
                require(
                    len(candidates) == 1,
                    f"expected exactly one captured clone/copy edge at {(request_index, coordinate)}; got {len(candidates)}",
                )
                event = candidates[0]
                require(
                    event.origin_coordinate == coordinate,
                    f"wrong source at {(request_index, coordinate)}: observed {event.origin_coordinate}",
                )
                require(
                    event.destination_storage == storage_descriptor(destination),
                    f"destination storage changed after captured edge at {(request_index, coordinate)}",
                )
                if require_direct_clone:
                    require(
                        event.operator == "aten.clone.default" and event.source is expected_source.tensor,
                        f"production helper did not directly clone expected source at {(request_index, coordinate)}",
                    )
                require(event.ordinal not in consumed, "one lineage event was consumed twice")
                require(id(destination) not in destination_ids, "two request coordinates share one tensor object")
                descriptor = storage_descriptor(destination)
                require(descriptor not in destination_storage, "two request coordinates share one storage descriptor")
                consumed.add(event.ordinal)
                destination_ids.add(id(destination))
                destination_storage.add(descriptor)
        require(len(consumed) == len(self.events), "captured persistent-rooted lineage contains extra edges")
        return {
            "policy": "materialized",
            "request_count": len(requests),
            "source_coordinate_count": len(normalized),
            "captured_lineage_edges": len(self.events),
            "all_edges_direct_aten_clone": all(
                event.operator == "aten.clone.default" for event in self.events
            ),
            "source_values_rechecked_unchanged": True,
        }

    def verify_borrowed(
        self,
        requests: Sequence[Any],
        coordinates: Iterable[Mapping[str, Any] | Coordinate],
    ) -> dict[str, Any]:
        self.registry.verify_unchanged()
        normalized = tuple(sorted(coordinate_key(value) for value in coordinates))
        require(set(normalized) == set(self.registry.by_coordinate), "verification coordinate set drift")
        for request_index, request in enumerate(requests):
            for coordinate in normalized:
                destination = tensor_at(request, coordinate)
                source = self.registry.by_coordinate[coordinate]
                require(
                    destination is source.tensor,
                    f"borrowed object is not exact expected source at {(request_index, coordinate)}",
                )
                require(
                    storage_descriptor(destination) == source.storage,
                    f"borrowed storage differs from expected source at {(request_index, coordinate)}",
                )
        require(not self.events, "borrowed policy unexpectedly emitted persistent-rooted clone/copy edges")
        return {
            "policy": "borrowed",
            "request_count": len(requests),
            "source_coordinate_count": len(normalized),
            "captured_lineage_edges": 0,
            "all_exact_expected_source_aliases": True,
            "source_values_rechecked_unchanged": True,
        }


__all__ = [
    "Coordinate",
    "LineageViolation",
    "PassiveCloneLineageMode",
    "PersistentSourceRegistry",
    "byte_interval",
    "coordinate_key",
    "storage_descriptor",
    "tensor_at",
    "tensor_bytes_sha256",
]
