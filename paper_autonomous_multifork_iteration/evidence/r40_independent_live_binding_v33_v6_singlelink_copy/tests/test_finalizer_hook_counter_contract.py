from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "executed_source"))

from r40_finalize import validate_rank_hook_counters  # noqa: E402


class FinalizerHookCounterContract(unittest.TestCase):
    def test_accepts_exact_integer_eight_and_rejects_neighboring_or_bool_values(self):
        expected = {
            "selected_builds": 1,
            "selected_phases": 3,
            "primary_memory_calls_observed": 8,
            "primary_memory_hook_events": 0,
        }
        self.assertIsNone(validate_rank_hook_counters(expected))
        for invalid in (7, 9, True):
            candidate = dict(expected)
            candidate["primary_memory_calls_observed"] = invalid
            with self.subTest(invalid=invalid), self.assertRaisesRegex(RuntimeError, "hook counter drift"):
                validate_rank_hook_counters(candidate)


if __name__ == "__main__":
    unittest.main()
