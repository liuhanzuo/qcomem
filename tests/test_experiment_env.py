from __future__ import annotations

import unittest

from macllm_bench.experiment_env import (
    active_power_mode,
    assess_completed_run,
    assess_preflight,
    parse_battery_status,
    parse_power_profiles,
    parse_swap_bytes,
    parse_thermal_output,
)


BATTERY_OUTPUT = """Now drawing from 'Battery Power'
 -InternalBattery-0 (id=1)\t100%; discharging; 11:09 remaining present: true
"""

POWER_OUTPUT = """Battery Power:
 powermode            0
 sleep                1
AC Power:
 powermode            2
 sleep                1
"""


class ExperimentEnvironmentTest(unittest.TestCase):
    def test_power_and_swap_parsers(self) -> None:
        battery = parse_battery_status(BATTERY_OUTPUT)
        profiles = parse_power_profiles(POWER_OUTPUT)
        self.assertEqual(battery["source"], "battery")
        self.assertEqual(battery["battery_percent"], 100)
        self.assertEqual(battery["battery_state"], "discharging")
        self.assertEqual(active_power_mode(battery, profiles), 0)
        self.assertEqual(profiles["AC Power"]["powermode"], 2)
        swap = parse_swap_bytes(
            "vm.swapusage: total = 1024.00M used = 423.94M free = 600.06M"
        )
        self.assertEqual(swap["total"], 1024**3)
        self.assertGreater(swap["used"], 400 * 1024**2)

    def test_no_thermal_warning_note_is_not_a_warning(self) -> None:
        no_warning = parse_thermal_output(
            "Note: No thermal warning level has been recorded\n"
            "Note: No performance warning level has been recorded"
        )
        warning = parse_thermal_output("Warning: CPU power status reduced")
        self.assertEqual(no_warning["warnings"], [])
        self.assertEqual(len(warning["warnings"]), 1)

    def test_battery_and_swap_growth_invalidate_formal_run(self) -> None:
        before = {
            "power": {"source": "battery", "low_power_mode": True},
            "thermal_state": {"value": 0, "label": "nominal"},
            "pmset_thermal": {"warnings": []},
            "cpu_percent": 4.0,
            "swap_bytes": {"used": 100},
        }
        after = {
            "thermal_state": {"value": 0, "label": "nominal"},
            "pmset_thermal": {"warnings": []},
            "swap_bytes": {"used": 300},
        }
        preflight = assess_preflight(before)
        completed = assess_completed_run(
            before, after, max_swap_growth_bytes=128
        )
        self.assertFalse(preflight["formal_result_eligible"])
        self.assertIn("not_connected_to_ac_power", preflight["reasons"])
        self.assertIn("low_power_mode_enabled", preflight["reasons"])
        self.assertIn("swap_growth_above_limit", completed["reasons"])


if __name__ == "__main__":
    unittest.main()
