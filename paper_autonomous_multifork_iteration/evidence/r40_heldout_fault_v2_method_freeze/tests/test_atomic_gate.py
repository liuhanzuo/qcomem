from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tests.helpers import atomic_receipt, digest, fp32_bytes, live_snapshot
from v2_capture import capture_atomic_call
from v2_common import ContractError, seal_payload
from v2_predicates import evaluate_atomic_sequence


POLICY_SHA = digest("frozen-general-atomic-policy")


class AtomicGateTests(unittest.TestCase):
    def test_clean_interleaved_sequence_passes_and_continuity_binds(self) -> None:
        a0 = live_snapshot("a", 10, 0, 0, "a-pre-0")
        a1 = live_snapshot("a", 11, 1, 1, "a-post-0")
        b0 = live_snapshot("b", 20, 0, 0, "b-pre-0")
        b1 = live_snapshot("b", 21, 1, 1, "b-post-0")
        a1_pre = dict(a1, observation_id="a-pre-1")
        a2 = live_snapshot("a", 12, 2, 2, "a-post-1")
        receipts = [
            atomic_receipt(0, 0, "a", a0, a1, POLICY_SHA),
            atomic_receipt(1, 0, "b", b0, b1, POLICY_SHA),
            atomic_receipt(2, 1, "a", a1_pre, a2, POLICY_SHA),
        ]
        schedule = [receipt["call_key"] for receipt in receipts]
        verdict = evaluate_atomic_sequence(receipts, schedule, POLICY_SHA)
        self.assertTrue(verdict["passed"])
        self.assertTrue(verdict["rows"][2]["checks"]["cross_call_continuity"])

    def test_torn_component_commit_fails_uniform_rule(self) -> None:
        pre = live_snapshot("a", 17, 7, 7, "pre")
        post = dict(live_snapshot("a", 17, 8, 8, "post"), kv_version=7, kv_commit_epoch=7)
        receipt = atomic_receipt(0, 0, "a", pre, post, POLICY_SHA)
        verdict = evaluate_atomic_sequence([receipt], [receipt["call_key"]], POLICY_SHA)
        self.assertFalse(verdict["passed"])
        self.assertFalse(verdict["rows"][0]["checks"]["kv_length_delta"])
        self.assertFalse(verdict["rows"][0]["checks"]["kv_version_delta"])
        self.assertFalse(verdict["rows"][0]["checks"]["post_epoch_coherent"])

    def test_schedule_seal_provenance_and_continuity_fail_closed(self) -> None:
        pre = live_snapshot("a", 10, 0, 0, "pre")
        post = live_snapshot("a", 11, 1, 1, "post")
        receipt = atomic_receipt(0, 0, "a", pre, post, POLICY_SHA)
        self.assertFalse(evaluate_atomic_sequence([], [receipt["call_key"]], POLICY_SHA)["passed"])
        tampered = dict(receipt, surfaced_token_id=99)
        with self.assertRaises(ContractError):
            evaluate_atomic_sequence([tampered], [receipt["call_key"]], POLICY_SHA)
        wrong_provenance = dict(receipt)
        wrong_provenance.pop("payload_sha256")
        wrong_provenance["model_reported_state_used_by_gate"] = True
        wrong_provenance = seal_payload(wrong_provenance)
        with self.assertRaises(ContractError):
            evaluate_atomic_sequence([wrong_provenance], [receipt["call_key"]], POLICY_SHA)

    def test_capture_ignores_model_claimed_state_and_is_nonoverwriting(self) -> None:
        pre = live_snapshot("a", 10, 0, 0, "reader-pre")
        post = live_snapshot("a", 11, 1, 1, "reader-post")
        observations = iter([pre, post])
        claimed = live_snapshot("a", 999, 999, 999, "model-claim")
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            receipt = capture_atomic_call(
                call_key={"call_index": 0, "round_index": 0, "request_id": "a"},
                input_token_count=1,
                policy_sha256=POLICY_SHA,
                live_state_reader=lambda: next(observations),
                model_call=lambda: {
                    "token_id": 3,
                    "logits_fp32_le": fp32_bytes([1, 2, 3, 4]),
                    "reported_state": claimed,
                },
                output_root=output,
                logit_relative_path="logits/0.bin",
                receipt_relative_path="receipts/0.json",
                vocab_size=4,
            )
            self.assertEqual(receipt["live_pre"]["observation_id"], "reader-pre")
            self.assertEqual(receipt["live_post"]["observation_id"], "reader-post")
            self.assertFalse(receipt["model_reported_state_used_by_gate"])
            observations = iter([pre, post])
            with self.assertRaises(ContractError):
                capture_atomic_call(
                    call_key={"call_index": 0, "round_index": 0, "request_id": "a"},
                    input_token_count=1,
                    policy_sha256=POLICY_SHA,
                    live_state_reader=lambda: next(observations),
                    model_call=lambda: {"token_id": 3, "logits_fp32_le": fp32_bytes([1, 2, 3, 4])},
                    output_root=output,
                    logit_relative_path="logits/0.bin",
                    receipt_relative_path="receipts/other.json",
                    vocab_size=4,
                )


if __name__ == "__main__":
    unittest.main()

