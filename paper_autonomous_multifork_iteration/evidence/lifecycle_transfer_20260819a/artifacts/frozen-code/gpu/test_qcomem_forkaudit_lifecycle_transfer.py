from __future__ import annotations

import copy
import unittest

from qcomem_forkaudit_lifecycle_transfer import (
    LifecycleContractError,
    SlotEpochRegistry,
    pairwise_disjoint,
    replay_slot_events,
)


class SlotEpochRegistryTest(unittest.TestCase):
    def test_cancel_reclaim_and_stale_handle(self) -> None:
        registry = SlotEpochRegistry(4)
        leases = [registry.acquire(index, f"initial-{index}") for index in range(4)]
        registry.cancel(leases[2])
        registry.cancel(leases[3])
        replacements = [
            registry.acquire(2, "replacement-2"),
            registry.acquire(3, "replacement-3"),
        ]
        for lease in (leases[0], leases[1], *replacements):
            registry.validate(lease)
        with self.assertRaisesRegex(LifecycleContractError, "STALE_SLOT_LEASE"):
            registry.validate(leases[2])
        replay = replay_slot_events(registry.receipt())
        self.assertEqual(replay["final_epochs"], [0, 0, 1, 1])
        self.assertEqual(replay["event_count"], 8)

    def test_replay_rejects_epoch_tamper(self) -> None:
        registry = SlotEpochRegistry(1)
        lease = registry.acquire(0, "a")
        registry.cancel(lease)
        value = copy.deepcopy(registry.receipt())
        value["events"][1]["epoch_after"] = 8
        with self.assertRaises(LifecycleContractError):
            replay_slot_events(value)

    def test_pairwise_disjoint(self) -> None:
        self.assertTrue(pairwise_disjoint(({1, 2}, {3}, {4, 5})))
        self.assertFalse(pairwise_disjoint(({1, 2}, {2, 3})))


if __name__ == "__main__":
    unittest.main()
