from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve()
EVIDENCE = HERE.parents[1]
PAPER = HERE.parents[3]
sys.path.insert(0, str(EVIDENCE / "scripts"))

from audit_independent_slot_census import (  # noqa: E402
    AuditFailure,
    audit_result,
    derive_expected_census,
    derive_linear_layers,
    sha256_file,
)
from run_negative_controls import apply_control  # noqa: E402


class IndependentSlotCensusTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_path = EVIDENCE / "protocol.json"
        cls.protocol = json.loads(cls.protocol_path.read_text(encoding="utf-8"))
        cls.input_path = PAPER / cls.protocol["source_evidence"]["raw_capture_path_from_paper_root"]
        cls.prereg_path = PAPER / cls.protocol["source_evidence"]["preregistration_path_from_paper_root"]
        cls.pristine = json.loads(cls.input_path.read_text(encoding="utf-8"))

    def test_source_bindings(self) -> None:
        self.assertEqual(
            sha256_file(self.input_path),
            self.protocol["source_evidence"]["raw_capture_sha256"],
        )
        self.assertEqual(
            sha256_file(self.prereg_path),
            self.protocol["source_evidence"]["preregistration_sha256"],
        )

    def test_geometry_derives_thirty_linear_layers(self) -> None:
        self.assertEqual(
            derive_linear_layers(self.protocol),
            [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18,
             20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33, 34, 36, 37, 38],
        )

    def test_census_is_complete_and_unique(self) -> None:
        census = derive_expected_census(self.protocol)
        self.assertEqual(len(census), 180)
        self.assertEqual(len(set(census)), 180)
        self.assertEqual(
            {row["owner_kind"] for row in census.values()},
            {"persistent", "request"},
        )

    def test_clean_archived_h20_capture_passes(self) -> None:
        report = audit_result(copy.deepcopy(self.pristine), self.protocol)
        self.assertTrue(report["passed"])
        self.assertFalse(report["producer_manifest_used_as_expectation"])
        self.assertEqual(report["audited_row_observations"], 1080)
        self.assertEqual(report["audited_relation_observations"], 96660)

    def test_resealed_omission_fails_closed(self) -> None:
        tampered, _ = apply_control(self.pristine, "C-OMIT-ONE-SLOT")
        with self.assertRaisesRegex(AuditFailure, "slot set mismatch") as caught:
            audit_result(tampered, self.protocol)
        self.assertEqual(caught.exception.code, "slot_set_mismatch")

    def test_resealed_duplication_fails_closed(self) -> None:
        tampered, _ = apply_control(self.pristine, "C-DUPLICATE-ONE-SLOT")
        with self.assertRaisesRegex(AuditFailure, "duplicates") as caught:
            audit_result(tampered, self.protocol)
        self.assertEqual(caught.exception.code, "duplicate_slot_id")

    def test_resealed_semantic_relabel_fails_closed(self) -> None:
        tampered, _ = apply_control(self.pristine, "C-SEMANTIC-RELABEL")
        with self.assertRaisesRegex(AuditFailure, "relabels") as caught:
            audit_result(tampered, self.protocol)
        self.assertEqual(caught.exception.code, "semantic_binding_mismatch")


if __name__ == "__main__":
    unittest.main()
