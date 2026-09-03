from __future__ import annotations

import unittest

from tests.helpers import allocator_arm
from v2_common import ContractError
from v2_predicates import evaluate_allocator_pair


class AllocatorGateTests(unittest.TestCase):
    def test_paired_endpoints_and_restoration_pass(self) -> None:
        verdict = evaluate_allocator_pair(allocator_arm("reference"), allocator_arm("candidate"))
        self.assertTrue(verdict["passed"])
        self.assertTrue(verdict["reference_restored"])
        self.assertTrue(verdict["candidate_restored"])

    def test_current_peak_and_restoration_mismatches_fail(self) -> None:
        candidate = allocator_arm("candidate", current=[100, 120, 139, 140, 101], peak=[100, 121, 150, 150, 150])
        verdict = evaluate_allocator_pair(allocator_arm("reference"), candidate)
        self.assertFalse(verdict["passed"])
        self.assertFalse(verdict["candidate_restored"])
        self.assertFalse(verdict["comparisons"][1]["peak_exact"])
        self.assertFalse(verdict["comparisons"][2]["current_exact"])

    def test_unsynchronized_or_malformed_phase_is_invalid(self) -> None:
        for mutation in ("sync", "phase"):
            with self.subTest(mutation=mutation):
                candidate = allocator_arm("candidate")
                if mutation == "sync":
                    candidate["endpoints"][2]["synchronized"] = False
                else:
                    candidate["endpoints"][2]["phase"] = "H1"
                with self.assertRaises(ContractError):
                    evaluate_allocator_pair(allocator_arm("reference"), candidate)


if __name__ == "__main__":
    unittest.main()

