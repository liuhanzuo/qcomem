from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "probes"))

import probe_v4_counterexamples as probe  # noqa: E402


class V4PosthocCounterexampleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = probe.run_all()
        cls.by_id = {row["probe_id"]: row for row in cls.result["probes"]}

    def test_peer_alias_after_transition_is_accepted(self) -> None:
        self.assertEqual(
            self.by_id["peer_alias_after_transition"]["observed_verifier_outcome"],
            "ACCEPTED",
        )

    def test_forged_storage_id_is_accepted(self) -> None:
        self.assertEqual(
            self.by_id["forged_storage_id"]["observed_verifier_outcome"],
            "ACCEPTED",
        )

    def test_persistent_mutation_after_freeze_is_accepted(self) -> None:
        self.assertEqual(
            self.by_id["persistent_mutation_after_freeze"]["observed_verifier_outcome"],
            "ACCEPTED",
        )


if __name__ == "__main__":
    unittest.main()
