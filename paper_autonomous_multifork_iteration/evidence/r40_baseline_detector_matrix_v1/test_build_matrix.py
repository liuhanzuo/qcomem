#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("r40_build_matrix", HERE / "build_matrix.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class R40BaselineMatrixTests(unittest.TestCase):
    def test_overlap_rule(self) -> None:
        rows = [
            {
                "owner_kind": "persistent",
                "request_index": None,
                "storage_id": "s0",
                "byte_start": 0,
                "byte_end_exclusive": 16,
                "layer_index": 0,
                "state_family": "conv",
                "state_index": 0,
            },
            {
                "owner_kind": "request",
                "request_index": 0,
                "storage_id": "s0",
                "byte_start": 8,
                "byte_end_exclusive": 24,
                "layer_index": 0,
                "state_family": "conv",
                "state_index": 0,
            },
        ]
        self.assertEqual(len(MODULE.simple_cross_owner_overlaps(rows)), 1)
        rows[1]["storage_id"] = "s1"
        self.assertEqual(MODULE.simple_cross_owner_overlaps(rows), [])

    def test_tail_order_rule(self) -> None:
        clean = {
            "fault_specific_evidence": {
                "ordered_tail_events": [
                    {"kind": "tail_copy", "ordinal": 0},
                    {"kind": "append_write", "ordinal": 1, "premature_shared": False},
                ]
            }
        }
        fault = {
            "fault_specific_evidence": {
                "ordered_tail_events": [
                    {"kind": "append_write", "ordinal": 0, "premature_shared": True},
                    {"kind": "tail_copy", "ordinal": 1},
                ]
            }
        }
        self.assertFalse(MODULE.lifecycle_evidence(clean)[0])
        self.assertTrue(MODULE.lifecycle_evidence(fault)[0])

    def test_full_build_is_deterministic_and_cardinality_bound(self) -> None:
        with tempfile.TemporaryDirectory(prefix="r40-baseline-test-") as temp:
            first = Path(temp) / "first"
            second = Path(temp) / "second"
            command = [
                sys.executable,
                str(HERE / "build_matrix.py"),
                "--repo-root",
                str(REPO_ROOT),
                "--output-dir",
            ]
            subprocess.run([*command, str(first)], check=True, capture_output=True, text=True)
            subprocess.run([*command, str(second)], check=True, capture_output=True, text=True)
            self.assertEqual(
                {path.name: path.read_bytes() for path in first.iterdir()},
                {path.name: path.read_bytes() for path in second.iterdir()},
            )
            matrix = json.loads((first / "baseline_detector_matrix.json").read_text())
            self.assertEqual(matrix["row_count"], 52)
            kinds = [row["case_kind"] for row in matrix["rows"]]
            self.assertEqual(kinds.count("fault"), 22)
            self.assertEqual(kinds.count("clean"), 30)
            faults = [row for row in matrix["rows"] if row["case_kind"] == "fault"]
            cleans = [row for row in matrix["rows"] if row["case_kind"] == "clean"]
            self.assertTrue(all(row["forkaudit"]["detected"] for row in faults))
            self.assertTrue(all(row["forkaudit"]["first_localization"] for row in faults))
            self.assertTrue(
                all(
                    row["baseline"]["first_localization"] is not None
                    for row in faults
                    if row["baseline"]["detected"]
                )
            )
            self.assertEqual(
                [row["case_id"] for row in faults if row["catch_relation"] == "forkaudit_unique"],
                ["primary/M8/fault"],
            )
            self.assertTrue(
                all(not row["baseline"]["detected"] and not row["forkaudit"]["detected"] for row in cleans)
            )
            summary = json.loads((first / "summary.json").read_text())["overall"]
            self.assertEqual(summary["baseline_caught_fault_case_count"], 21)
            self.assertEqual(summary["strict_independent_baseline_caught_fault_case_count"], 11)
            self.assertEqual(summary["forkaudit_caught_fault_case_count"], 22)
            self.assertEqual(summary["baseline_clean_false_positive_case_count"], 0)
            self.assertEqual(summary["forkaudit_clean_false_positive_case_count"], 0)
            subprocess.run([*command, str(first), "--verify-existing"], check=True, capture_output=True, text=True)


if __name__ == "__main__":
    unittest.main()
