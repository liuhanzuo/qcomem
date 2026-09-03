import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReviewerExperimentLoopTest(unittest.TestCase):
    def test_skill_requires_evidence_before_prose_and_rereview(self):
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("Phase 10.5: Reviewer-Triggered Experiment Loop", skill)
        self.assertIn("do not jump directly to prose revision", skill)
        self.assertIn("Do not start blind re-review", skill)
        self.assertIn("templates/reviewer-experiment.schema.json", skill)

    def test_experiment_template_has_outcome_and_verification_fields(self):
        schema = json.loads(
            (ROOT / "templates" / "reviewer-experiment.schema.json").read_text(
                encoding="utf-8"
            )
        )
        item = schema["properties"]["items"]["items"]
        for field in (
            "hypothesis",
            "refutation_condition",
            "primary_endpoint",
            "gates",
            "outcome_policy",
            "attempts",
            "verification_status",
        ):
            self.assertIn(field, item["required"])

    def test_default_panel_models_pdf_only_majority(self):
        skill = (ROOT / "SKILL.md").read_text(encoding="utf-8")
        protocol = (ROOT / "references" / "review-protocol.md").read_text(
            encoding="utf-8"
        )
        schema = json.loads(
            (ROOT / "templates" / "review.schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertIn("3 `pdf_only` reviewers", skill)
        self.assertIn("two `pdf_plus_repository` reviewers", protocol)
        self.assertIn("access_mode", schema["required"])
        self.assertEqual(
            schema["properties"]["access_mode"]["enum"],
            ["pdf_only", "pdf_plus_repository"],
        )


if __name__ == "__main__":
    unittest.main()
