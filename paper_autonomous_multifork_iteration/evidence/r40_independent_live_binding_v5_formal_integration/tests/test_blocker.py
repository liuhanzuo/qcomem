from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT.parent / "r40_independent_live_binding_v4_real_binder"


class BlockerAudit(unittest.TestCase):
    def test_v4_production_receipt_has_no_fault_outcome_schema(self):
        hook = (V4 / "executed_source/r40_real_binding_hook.py").read_text()
        self.assertIn('"phase_receipts"', hook)
        self.assertNotIn('"fault_results"', hook)
        self.assertNotIn('"failure_codes"', hook)

    def test_v4_faults_are_local_tests_not_rank_receipts(self):
        tests = (V4 / "tests/test_real_binding.py").read_text()
        self.assertEqual(tests.count("with self.assertRaisesRegex"), 5)
        hook = (V4 / "executed_source/r40_real_binding_hook.py").read_text()
        for fault_id in ("LB01", "LB02", "LB03", "LB04"):
            self.assertNotIn(fault_id, hook)

    def test_blocked_package_contains_no_launcher_or_entrypoint(self):
        self.assertFalse((ROOT / "formal").exists())
        self.assertFalse(any((ROOT / "executed_source").glob("*")) if (ROOT / "executed_source").exists() else False)
        acceptance = json.loads((ROOT / "acceptance.json").read_text())
        self.assertFalse(acceptance["formal_evidence_eligible"])


if __name__ == "__main__":
    unittest.main()
