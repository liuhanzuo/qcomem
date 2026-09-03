from __future__ import annotations

import unittest

from macllm_bench.replay_policy import (
    ComponentProfile,
    QuantizationOption,
    optimize_bit_policy,
)


class ReplayPolicyTest(unittest.TestCase):
    def test_multiple_choice_budget_is_exact(self) -> None:
        profiles = (
            ComponentProfile(
                "residual",
                (
                    QuantizationOption(2, 2, 8.0),
                    QuantizationOption(4, 4, 2.0),
                    QuantizationOption(8, 8, 0.0),
                ),
            ),
            ComponentProfile(
                "cache.0",
                (
                    QuantizationOption(2, 2, 10.0),
                    QuantizationOption(4, 4, 9.0),
                    QuantizationOption(8, 8, 0.0),
                ),
            ),
        )
        policy = optimize_bit_policy(profiles, budget_bytes=10)
        self.assertEqual(policy.as_dict(), {"residual": 2, "cache.0": 8})
        self.assertEqual(policy.total_nbytes, 10)
        self.assertEqual(policy.total_distortion, 8.0)

    def test_impossible_budget_is_rejected(self) -> None:
        profiles = (
            ComponentProfile("residual", (QuantizationOption(2, 5, 1.0),)),
        )
        with self.assertRaisesRegex(ValueError, "minimum 5"):
            optimize_bit_policy(profiles, budget_bytes=4)


if __name__ == "__main__":
    unittest.main()
