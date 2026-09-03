from __future__ import annotations

"""Pointer-free lifecycle leases for the ForkAudit aligned-prefix transfer.

The production Q16 arena preallocates one private reservation per request
slot.  This module adds the scheduler-facing lifetime contract that the
original resident-only experiment intentionally did not exercise: a request
may be cancelled, its slot may be reclaimed, and an old handle must then be
rejected before model execution.  The registry stores no tensor pointers and
is independently replayable from its event receipt.
"""

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


LIFECYCLE_PROTOCOL = "qcomem-forkaudit-aligned-reclamation-transfer-v1"
LIFECYCLE_EVENT_SCHEMA = "qcomem-forkaudit-slot-lifecycle-events-v1"


class LifecycleContractError(RuntimeError):
    """Stable fail-closed lifecycle error used by the transfer adapter."""

    def __init__(self, gate_id: str, message: str) -> None:
        self.gate_id = gate_id
        super().__init__(f"{gate_id}: {message}")


@dataclass(frozen=True)
class SlotLease:
    slot_id: int
    epoch: int
    request_id: str


@dataclass(frozen=True)
class SlotEvent:
    event_index: int
    operation: str
    slot_id: int
    epoch_before: int
    epoch_after: int
    request_id: str


class SlotEpochRegistry:
    """Deterministic finite-slot ownership registry.

    ``request_id`` is a run-local semantic identifier, not ``id(obj)``.  Live
    object identity is checked separately by the caller and is deliberately
    excluded from serialized evidence.
    """

    def __init__(self, slot_count: int) -> None:
        if type(slot_count) is not int or slot_count < 1:
            raise LifecycleContractError("SLOT_CONFIG", "slot_count must be positive")
        self._epochs = [0] * slot_count
        self._owners: list[str | None] = [None] * slot_count
        self._events: list[SlotEvent] = []

    @property
    def slot_count(self) -> int:
        return len(self._owners)

    def _slot(self, slot_id: int) -> None:
        if type(slot_id) is not int or not 0 <= slot_id < self.slot_count:
            raise LifecycleContractError("SLOT_CONFIG", "slot_id is outside the registry")

    def _record(
        self,
        operation: str,
        slot_id: int,
        epoch_before: int,
        epoch_after: int,
        request_id: str,
    ) -> None:
        self._events.append(
            SlotEvent(
                event_index=len(self._events),
                operation=operation,
                slot_id=slot_id,
                epoch_before=epoch_before,
                epoch_after=epoch_after,
                request_id=request_id,
            )
        )

    def acquire(self, slot_id: int, request_id: str) -> SlotLease:
        self._slot(slot_id)
        if not isinstance(request_id, str) or not request_id:
            raise LifecycleContractError("SLOT_CONFIG", "request_id must be non-empty")
        if self._owners[slot_id] is not None:
            raise LifecycleContractError("SLOT_ALREADY_OWNED", "slot already has a live owner")
        epoch = self._epochs[slot_id]
        self._owners[slot_id] = request_id
        self._record("acquire", slot_id, epoch, epoch, request_id)
        return SlotLease(slot_id=slot_id, epoch=epoch, request_id=request_id)

    def cancel(self, lease: SlotLease) -> None:
        self.validate(lease)
        slot_id = lease.slot_id
        before = self._epochs[slot_id]
        self._owners[slot_id] = None
        self._epochs[slot_id] = before + 1
        self._record("cancel", slot_id, before, before + 1, lease.request_id)

    def validate(self, lease: SlotLease) -> None:
        if not isinstance(lease, SlotLease):
            raise LifecycleContractError("STALE_SLOT_LEASE", "request has no typed slot lease")
        self._slot(lease.slot_id)
        owner = self._owners[lease.slot_id]
        epoch = self._epochs[lease.slot_id]
        if owner != lease.request_id or epoch != lease.epoch:
            raise LifecycleContractError(
                "STALE_SLOT_LEASE",
                "request lease is cancelled, superseded, or bound to another owner",
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema_version": LIFECYCLE_EVENT_SCHEMA,
            "slot_count": self.slot_count,
            "final_epochs": list(self._epochs),
            "final_owners": list(self._owners),
            "events": [asdict(event) for event in self._events],
        }


def replay_slot_events(value: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute terminal ownership without trusting serialized final fields."""

    if value.get("schema_version") != LIFECYCLE_EVENT_SCHEMA:
        raise LifecycleContractError("REPLAY_SCHEMA", "lifecycle event schema mismatch")
    slot_count = value.get("slot_count")
    if type(slot_count) is not int or slot_count < 1:
        raise LifecycleContractError("REPLAY_SCHEMA", "invalid slot_count")
    epochs = [0] * slot_count
    owners: list[str | None] = [None] * slot_count
    events = value.get("events")
    if not isinstance(events, list):
        raise LifecycleContractError("REPLAY_SCHEMA", "events must be a list")
    for event_index, row in enumerate(events):
        if not isinstance(row, Mapping) or row.get("event_index") != event_index:
            raise LifecycleContractError("REPLAY_ORDER", "event order is not canonical")
        slot_id = row.get("slot_id")
        if type(slot_id) is not int or not 0 <= slot_id < slot_count:
            raise LifecycleContractError("REPLAY_SCHEMA", "event slot is invalid")
        request_id = row.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise LifecycleContractError("REPLAY_SCHEMA", "event request_id is invalid")
        before = row.get("epoch_before")
        after = row.get("epoch_after")
        if before != epochs[slot_id] or type(after) is not int:
            raise LifecycleContractError("REPLAY_EPOCH", "event epoch chain is invalid")
        operation = row.get("operation")
        if operation == "acquire":
            if owners[slot_id] is not None or after != before:
                raise LifecycleContractError("REPLAY_OWNERSHIP", "invalid acquire transition")
            owners[slot_id] = request_id
        elif operation == "cancel":
            if owners[slot_id] != request_id or after != before + 1:
                raise LifecycleContractError("REPLAY_OWNERSHIP", "invalid cancel transition")
            owners[slot_id] = None
        else:
            raise LifecycleContractError("REPLAY_SCHEMA", "unknown lifecycle operation")
        epochs[slot_id] = after
    result = {"final_epochs": epochs, "final_owners": owners, "event_count": len(events)}
    if value.get("final_epochs") != epochs or value.get("final_owners") != owners:
        raise LifecycleContractError("REPLAY_TERMINAL", "serialized terminal state is incorrect")
    return result


def pairwise_disjoint(sets: Iterable[Iterable[int]]) -> bool:
    normalized = [set(values) for values in sets]
    return all(
        not (normalized[left] & normalized[right])
        for left in range(len(normalized))
        for right in range(left + 1, len(normalized))
    )


__all__ = [
    "LIFECYCLE_EVENT_SCHEMA",
    "LIFECYCLE_PROTOCOL",
    "LifecycleContractError",
    "SlotEpochRegistry",
    "SlotLease",
    "pairwise_disjoint",
    "replay_slot_events",
]
