from __future__ import annotations

import json
from pathlib import Path
import unittest

from tests.helpers import PAPER_ROOT, atomic_receipt, digest, live_snapshot
from v2_predicates import evaluate_atomic_sequence


FORMAL_ROOT = (
    PAPER_ROOT / "evidence/r39_blind_faults/formal_h20/"
    "r39-blind-faults-20260826g-metadata/r39-blind-faults-20260826g"
)


class DevelopmentOnlyRegressionTests(unittest.TestCase):
    def test_prior_campaign_is_read_only_development_evidence(self) -> None:
        summary = json.loads((FORMAL_ROOT / "summary.json").read_text(encoding="utf-8"))
        rows = summary["rows"]
        counts = {status: sum(row["status"] == status for row in rows) for status in {
            "valid_reached", "ineligible_preexecution", "operational_invalid"}}
        self.assertEqual(counts, {"valid_reached": 7, "ineligible_preexecution": 3, "operational_invalid": 1})
        valid = [row for row in rows if row["status"] == "valid_reached"]
        self.assertEqual(sum(row["observer_outcomes"]["forkaudit"]["detected"] for row in valid), 0)

    def test_known_escape_mechanism_is_only_a_development_fixture(self) -> None:
        plan = json.loads((PAPER_ROOT / "evidence/r39_blind_faults/designer_freeze/plan.json").read_text(encoding="utf-8"))
        historical = next(row for row in plan["faults"] if row["id"] == "R39-BF03")
        outcome = json.loads((FORMAL_ROOT / "R39-BF03/outcome.json").read_text(encoding="utf-8"))
        self.assertIn("KV rollback", historical["title"])
        self.assertTrue(outcome["valid_pair"])
        self.assertTrue(outcome["observer_outcomes"]["output_equality"]["complete_fp32_logits_byte_exact"])
        self.assertFalse(outcome["observer_outcomes"]["forkaudit"]["detected"])

        policy_sha = digest("frozen-general-atomic-policy")
        pre = live_snapshot("development-request", 17, 7, 7, "dev-pre")
        post = dict(
            live_snapshot("development-request", 17, 8, 8, "dev-post"),
            kv_version=7,
            kv_commit_epoch=7,
        )
        receipt = atomic_receipt(0, 0, "development-request", pre, post, policy_sha)
        verdict = evaluate_atomic_sequence([receipt], [receipt["call_key"]], policy_sha)
        self.assertFalse(verdict["passed"])
        self.assertEqual(verdict["attribution"], "hybrid_atomic_version_coherence")


if __name__ == "__main__":
    unittest.main()

