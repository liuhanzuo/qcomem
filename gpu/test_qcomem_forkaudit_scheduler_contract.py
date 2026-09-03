from __future__ import annotations

import unittest

from qcomem_forkaudit_lifecycle_transfer import SlotEpochRegistry
from qcomem_forkaudit_scheduler_contract import (
    DispatchBinding,
    SchedulerGateError,
    observe_gate,
    replay_schedule,
    require_dispatch,
    require_live_reservations_disjoint,
    require_zero_scrubbed,
)


class MockBlock:
    def __init__(self, nonzero: int) -> None:
        self.nonzero = nonzero

    def count_nonzero(self) -> int:
        return self.nonzero


class SchedulerContractTests(unittest.TestCase):
    def binding(self, registry: SlotEpochRegistry, slot: int, request: str, blocks: tuple[int, ...]) -> DispatchBinding:
        return DispatchBinding(request, slot, registry.acquire(slot, request), blocks)

    def test_clean_dispatch_and_disjoint_reservations(self) -> None:
        registry = SlotEpochRegistry(2)
        left = self.binding(registry, 0, "left", (1, 2))
        right = self.binding(registry, 1, "right", (3, 4))
        require_dispatch(registry, left, request_id="left", slot_id=0)
        require_live_reservations_disjoint((left, right))

    def test_stale_lease_is_rejected_after_reclaim(self) -> None:
        registry = SlotEpochRegistry(1)
        stale = self.binding(registry, 0, "old", (1,))
        registry.cancel(stale.lease)
        self.binding(registry, 0, "new", (1,))
        gate = observe_gate(
            lambda: require_dispatch(registry, stale, request_id="old", slot_id=0)
        )
        self.assertEqual(gate, "STALE_SLOT_LEASE")

    def test_cross_request_dispatch_is_rejected(self) -> None:
        registry = SlotEpochRegistry(1)
        binding = self.binding(registry, 0, "owner", (1,))
        gate = observe_gate(
            lambda: require_dispatch(registry, binding, request_id="other", slot_id=0)
        )
        self.assertEqual(gate, "LEASE_REQUEST_MISMATCH")

    def test_unscrubbed_reclaim_and_overlap_are_rejected(self) -> None:
        gate = observe_gate(
            lambda: require_zero_scrubbed(
                (MockBlock(0), MockBlock(3)),
                expected_physical_block_ids=(4, 5),
                observed_physical_block_ids=(4, 5),
            )
        )
        self.assertEqual(gate, "RECLAIM_NOT_ZERO")
        registry = SlotEpochRegistry(2)
        left = self.binding(registry, 0, "left", (1, 2))
        right = self.binding(registry, 1, "right", (2, 3))
        with self.assertRaisesRegex(SchedulerGateError, "LIVE_RESERVATION_OVERLAP"):
            require_live_reservations_disjoint((left, right))

    def test_schedule_replay_is_exact(self) -> None:
        expected = [
            {"event_index": 0, "phase": "initial", "slot_id": 1, "round_index": 0, "request_id": "r1"},
            {"event_index": 1, "phase": "initial", "slot_id": 0, "round_index": 0, "request_id": "r0"},
        ]
        self.assertTrue(replay_schedule(expected, expected=expected)["schedule_exact"])
        with self.assertRaisesRegex(SchedulerGateError, "SCHEDULE_DRIFT"):
            replay_schedule(list(reversed(expected)), expected=expected)


if __name__ == "__main__":
    unittest.main()
