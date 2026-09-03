from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from r40lib.candidate_binder import apply_fault_and_bind
from r40lib.detector import detect_binding
from r40lib.fixture import build_fixture
from r40lib.lane import execute_lane
from r40lib.observer_worker import observe_candidate
from r40lib.oracle_adapter import bind_oracle_items
from r40lib.oracle_worker import observe_oracle
from r40lib.protocol import build_manifest, sha256_json, validate_manifest


ROOT = Path(__file__).resolve().parents[1]
PREREG = json.loads((ROOT / "preregistration.json").read_text(encoding="utf-8"))


class ProtocolTests(unittest.TestCase):
    def test_manifest_is_independent_and_complete(self) -> None:
        manifest = build_manifest(PREREG)
        rows = validate_manifest(manifest, expected_count=18)
        self.assertEqual(len(rows), 18)
        self.assertEqual(
            len({tuple(shape) for shape in PREREG["fixture"]["shapes"].values()}), 2
        )

    def test_all_faults_change_live_references_not_labels(self) -> None:
        for fault in PREREG["faults"]:
            with self.subTest(fault=fault["fault_id"]):
                manifest = build_manifest(PREREG)
                manifest_sha = manifest["manifest_sha256"]
                fixture = build_fixture(PREREG, fault)
                items, receipt = apply_fault_and_bind(
                    manifest, fixture, fault, lane_type="mutant"
                )
                self.assertEqual(manifest["manifest_sha256"], manifest_sha)
                self.assertEqual(
                    len(receipt["changed_slot_ids"]), fault["expected_changed_slots"]
                )
                self.assertTrue(receipt["actual_live_tensor_references_changed"])
                self.assertFalse(receipt["schema_or_label_row_mutation_used"])
                self.assertTrue(all(set(item) == {"slot_id", "tensor"} for item in items))


class DetectorTests(unittest.TestCase):
    def _direct_observations(self, fault: dict, lane_type: str):
        manifest = build_manifest(PREREG)
        fixture = build_fixture(PREREG, fault)
        oracle = observe_oracle(
            {
                "manifest": manifest,
                "items": bind_oracle_items(manifest, fixture),
                "challenge_seed_sha256": PREREG["fixture"]["challenge_seed_sha256"],
            }
        )
        items, receipt = apply_fault_and_bind(
            manifest, fixture, fault, lane_type=lane_type
        )
        observation = observe_candidate(
            {
                "manifest": manifest,
                "items": items,
                "challenge_seed_sha256": PREREG["fixture"]["challenge_seed_sha256"],
            }
        )
        return detector_result(oracle, observation), receipt

    def test_four_matched_controls_and_faults(self) -> None:
        for fault in PREREG["faults"]:
            with self.subTest(fault=fault["fault_id"], lane="clean"):
                clean, receipt = self._direct_observations(fault, "clean")
                self.assertTrue(clean["passed"])
                self.assertEqual(receipt["changed_slot_ids"], [])
            with self.subTest(fault=fault["fault_id"], lane="mutant"):
                mutant, receipt = self._direct_observations(fault, "mutant")
                self.assertFalse(mutant["passed"])
                self.assertTrue(
                    all(code in mutant["failure_codes"] for code in fault["required_detection_codes"])
                )
                self.assertEqual(
                    len(receipt["changed_slot_ids"]), fault["expected_changed_slots"]
                )

    def test_process_separated_role_misbinding_lane(self) -> None:
        fault = next(row for row in PREREG["faults"] if row["fault_id"] == "R40-LB04")
        lane = execute_lane(
            PREREG,
            fault,
            lane_type="mutant",
            preregistration_sha256="unit-test-prereg",
            source_ledger_sha256="unit-test-source",
        )
        self.assertTrue(lane["acceptance_passed"])
        self.assertEqual(
            len({lane["producer_pid"], lane["oracle_pid"], lane["observer_pid"]}), 3
        )
        self.assertNotEqual(lane["producer_pid"], os.getppid())
        self.assertIn("storage_relation_mismatch", lane["detector"]["failure_codes"])


def detector_result(oracle, observation):
    result = detect_binding(oracle, observation)
    self_hash = sha256_json(result)
    if not self_hash:
        raise AssertionError("unreachable digest guard")
    return result


if __name__ == "__main__":
    unittest.main()
