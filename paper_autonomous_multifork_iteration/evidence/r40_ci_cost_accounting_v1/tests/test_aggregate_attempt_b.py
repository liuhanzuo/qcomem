from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import aggregate_results_attempt_b as aggregate_b


class AggregateAttemptBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.value = aggregate_b.build_aggregate()

    def test_successful_rows_and_failed_attempt_are_separate(self) -> None:
        self.assertEqual(self.value["attempt_b"]["status"], "18_of_18_component_rows_and_3_of_3_profiles_passed")
        self.assertEqual(self.value["attempt_a"]["failed_primary_rows"], 3)
        self.assertEqual(self.value["attempt_a"]["passed_supporting_rows"], 15)

    def test_minimal_artifact_is_manifest_declared(self) -> None:
        artifact = self.value["profiles"]["minimal_core"]["artifact"]
        self.assertEqual(artifact["manifest_payload_file_count"], 628)
        self.assertEqual(artifact["manifest_payload_logical_bytes"], 892144066)
        self.assertEqual(artifact["raw_trace_file_count"], 536)
        self.assertEqual(artifact["raw_trace_logical_bytes"], 888785811)
        self.assertEqual(artifact["source_unmanifested_file_count"], 13)

    def test_three_rows_per_component(self) -> None:
        self.assertEqual(len(self.value["components"]), 6)
        self.assertTrue(all(row["wall_seconds"]["count"] == 3 for row in self.value["components"].values()))

    def test_unmeasured_boundaries_remain_explicit(self) -> None:
        unmeasured = self.value["explicitly_unmeasured_costs"]
        self.assertIn("current_package_h20_capture_wall_time", unmeasured)
        self.assertIn("gpu_perturbation_or_slowdown", unmeasured)
        self.assertIn("engineering_or_adoption_effort", unmeasured)
        self.assertFalse(self.value["prohibitions"]["local_cpu_replay_is_h20_capture_overhead"])


if __name__ == "__main__":
    unittest.main()

