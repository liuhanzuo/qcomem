from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from tests.helpers import semantic_arm, write_logits
from v2_common import ContractError
from v2_predicates import evaluate_semantic_pair


EXACT = {"mode": "exact", "max_abs_threshold": 0.0, "relative_l2_threshold": 0.0}


class SemanticGateTests(unittest.TestCase):
    def test_exact_complete_logits_and_tokens_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref_root, cand_root = root / "ref", root / "cand"
            ref = semantic_arm("reference", [write_logits(ref_root, "0.bin", [1, 2, 3, 4])])
            cand = semantic_arm("candidate", [write_logits(cand_root, "0.bin", [1, 2, 3, 4])])
            verdict = evaluate_semantic_pair(ref, cand, ref_root, cand_root, EXACT, 4)
            self.assertTrue(verdict["passed"])
            self.assertEqual(verdict["reference_call_count"], 1)
            self.assertTrue(verdict["comparisons"][0]["logit_bytes_exact"])

    def test_token_logit_and_cardinality_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref_root, cand_root = root / "ref", root / "cand"
            ref_desc = [write_logits(ref_root, "0.bin", [1, 2, 3, 4])]
            cand_desc = [write_logits(cand_root, "0.bin", [1, 2, 3, 5])]
            ref = semantic_arm("reference", ref_desc, [9])
            cand = semantic_arm("candidate", cand_desc, [10])
            verdict = evaluate_semantic_pair(ref, cand, ref_root, cand_root, EXACT, 4)
            self.assertFalse(verdict["passed"])
            self.assertFalse(verdict["comparisons"][0]["token_exact"])
            self.assertFalse(verdict["comparisons"][0]["logit_pass"])
            cand["calls"].append(dict(cand["calls"][0], call_key={
                "call_index": 1, "round_index": 1, "request_id": "request-a"}))
            verdict = evaluate_semantic_pair(ref, cand, ref_root, cand_root, EXACT, 4)
            self.assertFalse(verdict["passed"])
            self.assertFalse(verdict["call_cardinality_and_order_exact"])

    def test_predeclared_tolerance_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ref_root, cand_root = root / "ref", root / "cand"
            ref = semantic_arm("reference", [write_logits(ref_root, "0.bin", [1, 2, 3, 4])])
            cand = semantic_arm("candidate", [write_logits(cand_root, "0.bin", [1, 2, 3, 4.0001])])
            loose = {"mode": "declared_tolerance", "max_abs_threshold": 0.001, "relative_l2_threshold": 0.001}
            strict = {"mode": "declared_tolerance", "max_abs_threshold": 0.00001, "relative_l2_threshold": 0.001}
            self.assertTrue(evaluate_semantic_pair(ref, cand, ref_root, cand_root, loose, 4)["passed"])
            self.assertFalse(evaluate_semantic_pair(ref, cand, ref_root, cand_root, strict, 4)["passed"])

    def test_missing_tampered_and_nonfinite_sidecars_are_invalid(self) -> None:
        for mode in ("missing", "tampered", "nonfinite"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                ref_root, cand_root = root / "ref", root / "cand"
                ref_desc = write_logits(ref_root, "0.bin", [1, 2, 3, 4])
                cand_desc = write_logits(cand_root, "0.bin", [1, 2, 3, 4])
                ref = semantic_arm("reference", [ref_desc])
                cand = semantic_arm("candidate", [cand_desc])
                if mode == "missing":
                    (cand_root / "0.bin").unlink()
                elif mode == "tampered":
                    (cand_root / "0.bin").write_bytes(b"x" * 16)
                else:
                    cand_desc = write_logits(cand_root, "nan.bin", [1, 2, math.nan, 4])
                    cand = semantic_arm("candidate", [cand_desc])
                with self.assertRaises(ContractError):
                    evaluate_semantic_pair(ref, cand, ref_root, cand_root, EXACT, 4)


if __name__ == "__main__":
    unittest.main()

