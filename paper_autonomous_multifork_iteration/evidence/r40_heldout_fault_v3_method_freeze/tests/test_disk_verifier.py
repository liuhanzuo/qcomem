from __future__ import annotations

import math
from pathlib import Path
import tempfile
import unittest

from tests.helpers import fake_authority, fake_formal, fp32, write_lane
from v3_common import ContractError
from v3_formal import FaultBinding
from v3_verifier import LaneExpectation, _DiskCampaignVerifier


class DiskVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.authority = fake_authority()

    def _read_pair(self, root: Path, candidate_mutator=None):
        formal = fake_formal(root / "output")
        fault = formal.faults[0]
        reference_root = root / "reference"
        candidate_root = root / "candidate"
        write_lane(reference_root, self.authority, formal.config_file_sha256, fault, "reference")
        write_lane(candidate_root, self.authority, formal.config_file_sha256, fault, "clean",
                   receipt_mutator=candidate_mutator)
        verifier = _DiskCampaignVerifier(self.authority, formal, vocab_size=4)
        reference = verifier.read_lane(
            reference_root, LaneExpectation(formal.run_id, "reference", fault, formal.config_file_sha256))
        candidate = verifier.read_lane(
            candidate_root, LaneExpectation(formal.run_id, "clean", fault, formal.config_file_sha256))
        return verifier, reference, candidate

    def test_disk_reread_exact_inventory_and_all_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            verifier, reference, candidate = self._read_pair(Path(temporary))
            self.assertTrue(reference.atomic_verdict["passed"])
            self.assertTrue(candidate.atomic_verdict["passed"])
            self.assertTrue(verifier.semantic_pair(reference, candidate)["passed"])
            self.assertTrue(verifier.structural_pair(reference, candidate)["passed"])
            self.assertTrue(verifier.allocator_pair(reference, candidate)["passed"])

    def test_extra_missing_symlink_and_tampered_sidecar_fail(self) -> None:
        mutations = ("extra", "missing", "symlink", "tamper")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                formal = fake_formal(root / "output")
                fault = formal.faults[0]
                lane_root = root / "lane"
                write_lane(lane_root, self.authority, formal.config_file_sha256, fault, "reference")
                if mutation == "extra":
                    (lane_root / "extra.txt").write_text("extra")
                elif mutation == "missing":
                    (lane_root / "receipts/call-015.json").unlink()
                elif mutation == "symlink":
                    (lane_root / "logits/call-015.f32le").unlink()
                    (lane_root / "logits/call-015.f32le").symlink_to(lane_root / "logits/call-014.f32le")
                else:
                    (lane_root / "logits/call-015.f32le").write_bytes(b"x" * 16)
                verifier = _DiskCampaignVerifier(self.authority, formal, vocab_size=4)
                with self.assertRaises(ContractError):
                    verifier.read_lane(
                        lane_root, LaneExpectation(formal.run_id, "reference", fault, formal.config_file_sha256))

    def test_receipt_campaign_lane_fault_gpu_schedule_and_method_bindings_fail(self) -> None:
        fields = ("campaign_id", "lane", "fault_id", "gpu_uuid", "schedule", "method")
        for field in fields:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                formal = fake_formal(root / "output")
                fault = formal.faults[0]

                def mutate(call_index, receipt):
                    if call_index != 0:
                        return
                    if field == "schedule":
                        receipt["call_key"] = dict(receipt["call_key"], input_token_count=1)
                    elif field == "method":
                        receipt["method_core_manifest_sha256"] = "0" * 64
                    else:
                        receipt[field] = "wrong"

                lane_root = root / "lane"
                write_lane(lane_root, self.authority, formal.config_file_sha256, fault, "reference",
                           receipt_mutator=mutate)
                verifier = _DiskCampaignVerifier(self.authority, formal, vocab_size=4)
                with self.assertRaises(ContractError):
                    verifier.read_lane(
                        lane_root, LaneExpectation(formal.run_id, "reference", fault, formal.config_file_sha256))

    def test_nonfinite_complete_logits_fail_even_when_hash_matches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = fake_formal(root / "output")
            fault = formal.faults[0]
            lane_root = root / "lane"
            write_lane(
                lane_root, self.authority, formal.config_file_sha256, fault, "reference",
                logit_mutator=lambda index, raw: fp32([math.nan, 1, 2, 3]) if index == 0 else raw,
            )
            verifier = _DiskCampaignVerifier(self.authority, formal, vocab_size=4)
            with self.assertRaises(ContractError):
                verifier.read_lane(
                    lane_root, LaneExpectation(formal.run_id, "reference", fault, formal.config_file_sha256))

    def test_reference_changes_candidate_rollback_fails_structural_and_atomic(self) -> None:
        def rollback(call_index, receipt):
            if call_index == 14:
                receipt["live_post"] = dict(receipt["live_pre"],
                                            observation_id=receipt["live_post"]["observation_id"],
                                            sync_event_id=receipt["live_post"]["sync_event_id"])

        with tempfile.TemporaryDirectory() as temporary:
            verifier, reference, candidate = self._read_pair(Path(temporary), rollback)
            structural = verifier.structural_pair(reference, candidate)
            self.assertFalse(structural["passed"])
            self.assertFalse(candidate.atomic_verdict["passed"])
            self.assertFalse(structural["rows"][14]["checks"]["live_post.kv_content_sha256"])

    def test_gdn_content_only_drift_fails_paired_structural_gate(self) -> None:
        def mutate(call_index, receipt):
            if call_index == 15:
                receipt["live_post"]["gdn_content_sha256"] = "f" * 64

        with tempfile.TemporaryDirectory() as temporary:
            verifier, reference, candidate = self._read_pair(Path(temporary), mutate)
            self.assertTrue(candidate.atomic_verdict["passed"])
            verdict = verifier.structural_pair(reference, candidate)
            self.assertFalse(verdict["passed"])
            self.assertFalse(verdict["rows"][15]["checks"]["live_post.gdn_content_sha256"])

    def test_observation_and_sync_ids_are_global_not_per_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            formal = fake_formal(root / "output")
            fault = formal.faults[0]
            ref = root / "reference"
            clean = root / "clean"
            write_lane(ref, self.authority, formal.config_file_sha256, fault, "reference", id_namespace="duplicate")
            write_lane(clean, self.authority, formal.config_file_sha256, fault, "clean", id_namespace="duplicate")
            verifier = _DiskCampaignVerifier(self.authority, formal, vocab_size=4)
            verifier.read_lane(ref, LaneExpectation(formal.run_id, "reference", fault, formal.config_file_sha256))
            with self.assertRaises(ContractError):
                verifier.read_lane(clean, LaneExpectation(formal.run_id, "clean", fault, formal.config_file_sha256))

    def test_allocator_peak_monotonicity_binding_and_global_event_uniqueness(self) -> None:
        mutations = ("peak", "lane", "event")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                formal = fake_formal(root / "output")
                fault = formal.faults[0]

                def mutate_allocator(value):
                    if mutation == "peak":
                        value["endpoints"][3]["peak_allocated_bytes"] = 140
                    elif mutation == "lane":
                        value["lane"] = "mutant"
                    else:
                        value["endpoints"][0]["sync_event_id"] = "obs-%s-reference-000-pre" % fault.fault_id

                lane_root = root / "lane"
                write_lane(lane_root, self.authority, formal.config_file_sha256, fault, "reference",
                           allocator_mutator=mutate_allocator)
                verifier = _DiskCampaignVerifier(self.authority, formal, vocab_size=4)
                with self.assertRaises(ContractError):
                    verifier.read_lane(
                        lane_root, LaneExpectation(formal.run_id, "reference", fault, formal.config_file_sha256))

    def test_complete_eight_case_three_lane_campaign_is_disk_enumerated(self) -> None:
        faults = tuple(
            FaultBinding("V3F%02d" % (index + 1), "GPU-test-%d" % index, index)
            for index in range(8)
        )
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            artifacts = output / "artifacts"
            artifacts.mkdir(parents=True)
            formal = fake_formal(output, faults)

            def rollback(call_index, receipt):
                if call_index == 14:
                    receipt["live_post"] = dict(
                        receipt["live_pre"],
                        observation_id=receipt["live_post"]["observation_id"],
                        sync_event_id=receipt["live_post"]["sync_event_id"],
                    )

            for fault_index, fault in enumerate(faults):
                for lane in ("reference", "clean", "mutant"):
                    write_lane(
                        artifacts / fault.fault_id / lane,
                        self.authority, formal.config_file_sha256, fault, lane,
                        receipt_mutator=rollback if fault_index == 0 and lane == "mutant" else None,
                    )
            verdict = _DiskCampaignVerifier(self.authority, formal, vocab_size=4).verify_campaign()
            self.assertEqual(verdict["fault_count"], 8)
            self.assertFalse(verdict["population_detection_rate_computed"])
            self.assertTrue(verdict["faults"][0]["detected_by_frozen_gates"])
            self.assertTrue(all(row["reference_clean_valid"] for row in verdict["faults"]))


if __name__ == "__main__":
    unittest.main()
