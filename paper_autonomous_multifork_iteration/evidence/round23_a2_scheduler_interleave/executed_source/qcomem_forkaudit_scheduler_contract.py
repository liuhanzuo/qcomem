from __future__ import annotations

"""Pointer-free guards for the Round-22 scheduler/interleaving case.

The model/cache implementation remains the frozen Qwen3.5 + vLLM-Q16
ForkAudit stack.  This module only adds the scheduler-facing checks needed by
the new affected-only experiment: lease/request binding, live reservation
disjointness, and zero-scrub before a cancelled slot is reassigned.
"""

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from qcomem_forkaudit_lifecycle_transfer import (
    LifecycleContractError,
    SlotEpochRegistry,
    SlotLease,
)


SCHEDULER_PROTOCOL = "qcomem-forkaudit-scheduler-interleave-v1"


class SchedulerGateError(RuntimeError):
    def __init__(self, gate_id: str, message: str) -> None:
        self.gate_id = gate_id
        super().__init__(f"{gate_id}: {message}")


@dataclass(frozen=True)
class DispatchBinding:
    request_id: str
    slot_id: int
    lease: SlotLease
    physical_block_ids: tuple[int, ...]


def require_dispatch(
    registry: SlotEpochRegistry,
    binding: DispatchBinding,
    *,
    request_id: str,
    slot_id: int,
) -> None:
    """Fail closed before a scheduled model call."""

    try:
        registry.validate(binding.lease)
    except LifecycleContractError as error:
        raise SchedulerGateError(error.gate_id, str(error)) from error
    if binding.request_id != request_id or binding.lease.request_id != request_id:
        raise SchedulerGateError(
            "LEASE_REQUEST_MISMATCH", "lease and dispatched request identifiers differ"
        )
    if binding.slot_id != slot_id or binding.lease.slot_id != slot_id:
        raise SchedulerGateError(
            "LEASE_SLOT_MISMATCH", "lease and dispatched slot identifiers differ"
        )


def require_live_reservations_disjoint(bindings: Sequence[DispatchBinding]) -> None:
    seen: dict[int, str] = {}
    for binding in bindings:
        for block_id in binding.physical_block_ids:
            owner = seen.get(block_id)
            if owner is not None:
                raise SchedulerGateError(
                    "LIVE_RESERVATION_OVERLAP",
                    f"physical block {block_id} is live for both {owner} and {binding.request_id}",
                )
            seen[block_id] = binding.request_id


def require_zero_scrubbed(
    blocks: Iterable[Any],
    *,
    expected_physical_block_ids: Sequence[int],
    observed_physical_block_ids: Sequence[int],
) -> None:
    if tuple(observed_physical_block_ids) != tuple(expected_physical_block_ids):
        raise SchedulerGateError(
            "RECLAIM_RESERVATION_MISMATCH",
            "reassigned physical reservation differs from the cancelled slot",
        )
    for block in blocks:
        # Torch tensors and CPU mock blocks both provide count_nonzero().
        value = block.count_nonzero()
        nonzero = int(value.item() if hasattr(value, "item") else value)
        if nonzero:
            raise SchedulerGateError(
                "RECLAIM_NOT_ZERO", "cancelled private storage was not zero-scrubbed"
            )


def observe_gate(callable_: Any) -> str | None:
    try:
        callable_()
    except SchedulerGateError as error:
        return error.gate_id
    return None


def replay_schedule(
    events: Sequence[Mapping[str, Any]],
    *,
    expected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recompute the frozen scheduler order without trusting producer flags."""

    normalized = [
        {
            "event_index": index,
            "phase": row.get("phase"),
            "slot_id": row.get("slot_id"),
            "round_index": row.get("round_index"),
            "request_id": row.get("request_id"),
        }
        for index, row in enumerate(events)
    ]
    target = [dict(row) for row in expected]
    if normalized != target:
        raise SchedulerGateError("SCHEDULE_DRIFT", "observed dispatch order differs from preregistration")
    return {"event_count": len(normalized), "schedule_exact": True}


__all__ = [
    "DispatchBinding",
    "SCHEDULER_PROTOCOL",
    "SchedulerGateError",
    "observe_gate",
    "replay_schedule",
    "require_dispatch",
    "require_live_reservations_disjoint",
    "require_zero_scrubbed",
]
