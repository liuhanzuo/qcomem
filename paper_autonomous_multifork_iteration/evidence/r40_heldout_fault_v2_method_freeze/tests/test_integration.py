from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.helpers import allocator_arm, atomic_receipt, digest, fp32_bytes, live_snapshot
from v2_common import ContractError, sha256_bytes
from v2_integration import AllocatorArmCapture, evaluate_lane_pair


class IntegrationTests(unittest.TestCase):
    def test_fixed_allocator_capture_order(self) -> None:
        readings = iter([
            {"current_allocated_bytes": 100, "peak_allocated_bytes": 100},
            {"current_allocated_bytes": 120, "peak_allocated_bytes": 120},
            {"current_allocated_bytes": 140, "peak_allocated_bytes": 150},
            {"current_allocated_bytes": 140, "peak_allocated_bytes": 150},
            {"current_allocated_bytes": 100, "peak_allocated_bytes": 150},
        ])
        events = iter("sync-%d" % index for index in range(6))
        reset_count = []
        capture = AllocatorArmCapture(
            "reference", lambda: next(events), lambda: reset_count.append(True), lambda: next(readings))
        with self.assertRaises(ContractError):
            capture.capture("H1")
        for phase in ("H0", "H1", "H4", "H6", "H7"):
            capture.capture(phase)
        arm = capture.finish()
        self.assertEqual(len(reset_count), 1)
        self.assertEqual([row["phase"] for row in arm["endpoints"]], ["H0", "H1", "H4", "H6", "H7"])

    def test_reference_candidate_integration_passes_all_three_gates(self) -> None:
        policy_sha = digest("integrated-atomic-policy")
        pre = live_snapshot("a", 10, 0, 0, "pre")
        post = live_snapshot("a", 11, 1, 1, "post")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference_root = root / "reference"
            candidate_root = root / "candidate"
            reference_root.mkdir()
            candidate_root.mkdir()
            raw = fp32_bytes([1, 2, 3, 4])
            (reference_root / "0.bin").write_bytes(raw)
            (candidate_root / "0.bin").write_bytes(raw)
            reference = dict(atomic_receipt(0, 0, "a", pre, post, policy_sha))
            candidate = dict(atomic_receipt(0, 0, "a", pre, post, policy_sha))
            for receipt in (reference, candidate):
                receipt["logits"]["path"] = "0.bin"
                receipt["logits"]["sha256"] = sha256_bytes(raw)
                receipt.pop("payload_sha256")
                from v2_common import seal_payload
                sealed = seal_payload(receipt)
                receipt.clear()
                receipt.update(sealed)
            schedule = [reference["call_key"]]
            verdict = evaluate_lane_pair(
                reference_receipts=[reference], candidate_receipts=[candidate],
                expected_schedule=schedule,
                reference_allocator=allocator_arm("reference"),
                candidate_allocator=allocator_arm("candidate"),
                reference_root=reference_root, candidate_root=candidate_root,
                semantic_policy={"mode": "exact", "max_abs_threshold": 0.0, "relative_l2_threshold": 0.0},
                atomic_policy_sha256=policy_sha, vocab_size=4,
            )
            self.assertTrue(verdict["passed"])
            self.assertEqual(verdict["semantic"]["attribution"], "paired_semantic_baseline")
            self.assertEqual(verdict["candidate_atomic"]["attribution"], "hybrid_atomic_version_coherence")


if __name__ == "__main__":
    unittest.main()

