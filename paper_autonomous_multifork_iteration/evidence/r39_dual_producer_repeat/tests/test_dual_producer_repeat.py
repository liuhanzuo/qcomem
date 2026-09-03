from __future__ import annotations

import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
from typing import Any, Mapping


HERE = Path(__file__).resolve()
EVIDENCE = HERE.parents[1]
LOCAL_R39 = EVIDENCE / "vendor/r39"
if not LOCAL_R39.is_dir():
    LOCAL_R39 = EVIDENCE.parent / "r39_independent_slot_census/scripts"
sys.path.insert(0, str(LOCAL_R39))
sys.path.insert(0, str(EVIDENCE / "scripts"))

from audit_independent_slot_census import (  # noqa: E402
    audit_result,
    derive_expected_census,
    relation_vector,
    sha256_file,
    sha256_json,
)
from verify_dual_producer_repeat import (  # noqa: E402
    DualRepeatFailure,
    verify_dual_repeat,
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_manifest(protocol: Mapping[str, Any], slots: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = {
        "schema_version": "forkaudit-r33-ipc-slot-manifest-v1",
        "resident_count": 2,
        "layer_indices": [
            index
            for index in range(40)
            if index % 4 != 3
        ],
        "state_index": 0,
        "capture_ids": [row["capture_id"] for row in protocol["schedule"]["captures"]],
        "slots": slots,
        "live_request_disallowed_judgment_fields": [],
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    return manifest


def build_row(
    coordinate: Mapping[str, Any],
    *,
    policy: str,
    capture_id: str,
    producer_label: str,
) -> dict[str, Any]:
    family = str(coordinate["state_family"])
    if family == "conv":
        geometry = {
            "shape": [1, 8192, 4],
            "stride": [32768, 4, 1],
            "storage_offset": 0,
            "dtype": "torch.bfloat16",
            "device": "cuda:0",
            "storage_nbytes": 65536,
            "tensor_nbytes": 65536,
            "byte_start": 0,
            "byte_end_exclusive": 65536,
        }
    else:
        geometry = {
            "shape": [1, 32, 128, 128],
            "stride": [524288, 16384, 128, 1],
            "storage_offset": 0,
            "dtype": "torch.float32",
            "device": "cuda:0",
            "storage_nbytes": 2097152,
            "tensor_nbytes": 2097152,
            "byte_start": 0,
            "byte_end_exclusive": 2097152,
        }
    semantic = {
        field: coordinate[field]
        for field in (
            "owner_kind",
            "request_index",
            "layer_index",
            "state_family",
            "state_index",
        )
    }
    content_key = f"{policy}|{capture_id}|{coordinate['slot_id']}"
    local_key = f"{producer_label}|{content_key}"
    return {
        "slot_id": coordinate["slot_id"],
        **semantic,
        **geometry,
        "content_sha256": digest(content_key),
        "storage_token": digest("storage|" + local_key),
        "view_token": digest("view|" + local_key),
    }


def build_result(
    protocol: Mapping[str, Any],
    preregistration: Mapping[str, Any],
    *,
    producer_label: str,
    producer_pid: int,
    observer_pid_base: int,
) -> dict[str, Any]:
    census = derive_expected_census(protocol)
    manifest_slots = [census[slot_id] for slot_id in sorted(census)]
    manifest = build_manifest(protocol, manifest_slots)
    cells = []
    for cell_index, policy in enumerate(protocol["schedule"]["policy_cells"]):
        observer_pid = observer_pid_base + cell_index
        commitment = digest(f"session|{producer_label}|{policy}")
        captures = []
        for plan in protocol["schedule"]["captures"]:
            capture_id = plan["capture_id"]
            rows = [
                build_row(
                    census[slot_id],
                    policy=policy,
                    capture_id=capture_id,
                    producer_label=producer_label,
                )
                for slot_id in sorted(census)
            ]
            vector = relation_vector(rows)
            captures.append(
                {
                    "schema_version": "forkaudit-r33-out-of-process-capture-v1",
                    "capture_id": capture_id,
                    "observer_pid": observer_pid,
                    "producer_pid": producer_pid,
                    "process_separated": True,
                    "observer_session_commitment_sha256": commitment,
                    "slot_manifest_sha256": manifest["manifest_sha256"],
                    "live_request_fields_received": ["capture_id", "schema_version", "slot_tensors"],
                    "live_slot_fields_received": ["slot_id", "tensor"],
                    "judgment_fields_received": [],
                    "candidate_verdict_fields_received": False,
                    "raw_addresses_serialized": False,
                    "receiver_derived_descriptors": True,
                    "receiver_derived_relations": True,
                    "transport": "torch-cuda-ipc-reduction",
                    "all_cpu_tensors_shared_in_receiver": True,
                    "imported_views_pinned_against_receiver_aba": True,
                    "row_count": len(rows),
                    "relation_count": len(vector),
                    "rows": rows,
                    "rows_sha256": sha256_json(rows),
                    "relation_vector_sha256": sha256_json(vector),
                }
            )
        cells.append(
            {
                "policy": policy,
                "capture_plan": protocol["schedule"]["captures"],
                "slot_manifest": copy.deepcopy(manifest),
                "captures": captures,
            }
        )
    upstream = preregistration["upstream_bindings"]
    return {
        "schema_version": "forkaudit-r33-out-of-process-result-v1",
        "status": "completed_pending_independent_replay",
        "preregistration_sha256": upstream["r33_preregistration_raw_sha256"],
        "source_ledger_raw_sha256": upstream["r33_source_ledger_raw_sha256"],
        "candidate_runtime_code_ledger_raw_sha256": upstream[
            "candidate_runtime_code_ledger_raw_sha256"
        ],
        "source_sha256": upstream["r33_source_sha256"],
        "input_receipt": upstream["input_receipt"],
        "hardware": {
            "uuid": "GPU-SYNTHETIC-SAME",
            "name": "NVIDIA H20-3e",
            "memory_mib": 143771,
            "compute_capability": [9, 0],
            "torch_version": "2.11.0+cu129",
            "torch_cuda": "12.9",
        },
        "claim_authorized": False,
        "independence_boundary": {},
        "cells": cells,
    }


def reseal_capture(capture: dict[str, Any]) -> None:
    capture["rows_sha256"] = sha256_json(capture["rows"])
    vector = relation_vector(capture["rows"])
    capture["relation_count"] = len(vector)
    capture["relation_vector_sha256"] = sha256_json(vector)


class DualProducerRepeatTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol_path = EVIDENCE / "slot_protocol.json"
        cls.preregistration_path = EVIDENCE / "preregistration.json"
        cls.protocol = json.loads(cls.protocol_path.read_text(encoding="utf-8"))
        cls.preregistration = json.loads(
            cls.preregistration_path.read_text(encoding="utf-8")
        )
        cls.census_map = derive_expected_census(cls.protocol)
        cls.census_rows = [
            cls.census_map[slot_id] for slot_id in sorted(cls.census_map)
        ]
        cls.census_semantic_sha = sha256_json(cls.census_rows)
        cls.protocol_sha = sha256_file(cls.protocol_path)
        cls.prereg_sha = sha256_file(cls.preregistration_path)
        cls.source_ledger_sha = digest("synthetic-source-ledger")
        cls.census_sha = digest("synthetic-census-file")
        cls.base_a = build_result(
            cls.protocol,
            cls.preregistration,
            producer_label="a",
            producer_pid=1001,
            observer_pid_base=1101,
        )
        cls.base_b = build_result(
            cls.protocol,
            cls.preregistration,
            producer_label="b",
            producer_pid=2001,
            observer_pid_base=2101,
        )

    def inputs(self) -> dict[str, Any]:
        a = copy.deepcopy(self.base_a)
        b = copy.deepcopy(self.base_b)
        census = {
            "schema_version": "forkaudit-r39-preexecution-slot-census-v1",
            "status": "frozen_before_fresh_h20_producer_start",
            "experiment_id": self.protocol["experiment_id"],
            "protocol_raw_sha256": self.protocol_sha,
            "source_ledger_raw_sha256": self.source_ledger_sha,
            "producer_manifest_used": False,
            "producer_rows_used": False,
            "slot_count": 180,
            "census_semantic_sha256": self.census_semantic_sha,
            "slots": self.census_rows,
        }
        receipt = {
            "schema_version": "forkaudit-r39-preexecution-census-receipt-v1",
            "status": "frozen_before_fresh_h20_producer_start",
            "protocol_raw_sha256": self.protocol_sha,
            "source_ledger_raw_sha256": self.source_ledger_sha,
            "census_file_sha256": self.census_sha,
            "census_semantic_sha256": self.census_semantic_sha,
            "producer_started": False,
            "producer_manifest_available_to_derivation": False,
            "producer_rows_available_to_derivation": False,
        }
        hashes = {
            "preregistration": self.prereg_sha,
            "protocol": self.protocol_sha,
            "source_ledger": self.source_ledger_sha,
            "census": self.census_sha,
            "producer_a": digest("synthetic-producer-a"),
            "producer_b": digest("synthetic-producer-b"),
            "producer_a_audit": digest("synthetic-producer-a-audit"),
            "producer_b_audit": digest("synthetic-producer-b-audit"),
            "producer_a_replay": digest("synthetic-producer-a-replay"),
            "producer_b_replay": digest("synthetic-producer-b-replay"),
        }
        audit_a = audit_result(a, self.protocol)
        audit_b = audit_result(b, self.protocol)
        for audit, raw_sha in (
            (audit_a, hashes["producer_a"]),
            (audit_b, hashes["producer_b"]),
        ):
            audit.update(
                {
                    "input_raw_sha256": raw_sha,
                    "protocol_raw_sha256": self.protocol_sha,
                    "preexecution_census_bound": True,
                }
            )
        replay_a = {
            "passed": True,
            "input_result_sha256": sha256_json(a),
            "cell_count": 2,
            "row_observations": 1080,
            "relation_observations": 96660,
        }
        replay_b = {
            "passed": True,
            "input_result_sha256": sha256_json(b),
            "cell_count": 2,
            "row_observations": 1080,
            "relation_observations": 96660,
        }
        return {
            "preregistration": self.preregistration,
            "protocol": self.protocol,
            "census": census,
            "census_receipt": receipt,
            "producer_a": a,
            "producer_b": b,
            "producer_a_replay": replay_a,
            "producer_b_replay": replay_b,
            "producer_a_audit": audit_a,
            "producer_b_audit": audit_b,
            "file_hashes": hashes,
        }

    def test_clean_repeat_closes_all_preregistered_counts(self) -> None:
        report = verify_dual_repeat(**self.inputs())
        self.assertTrue(report["passed"])
        self.assertEqual(report["matched_semantic_coordinates"], 1080)
        self.assertEqual(report["matched_content_digests"], 1080)
        self.assertEqual(report["matched_stable_descriptors"], 1080)
        self.assertEqual(report["matched_relation_labels"], 96660)
        self.assertEqual(report["numeric_tolerance"], 0)
        self.assertFalse(report["canonical_semantic_fallback"])

    def test_resealed_content_mismatch_fails_closed(self) -> None:
        inputs = self.inputs()
        capture = inputs["producer_b"]["cells"][0]["captures"][0]
        capture["rows"][0]["content_sha256"] = "f" * 64
        reseal_capture(capture)
        inputs["producer_b_audit"] = audit_result(inputs["producer_b"], self.protocol)
        inputs["producer_b_audit"].update(
            {
                "input_raw_sha256": inputs["file_hashes"]["producer_b"],
                "protocol_raw_sha256": self.protocol_sha,
                "preexecution_census_bound": True,
            }
        )
        inputs["producer_b_replay"]["input_result_sha256"] = sha256_json(
            inputs["producer_b"]
        )
        with self.assertRaises(DualRepeatFailure) as caught:
            verify_dual_repeat(**inputs)
        self.assertEqual(caught.exception.code, "cross_producer_content_mismatch")

    def test_resealed_relation_mismatch_fails_closed(self) -> None:
        inputs = self.inputs()
        capture = inputs["producer_b"]["cells"][0]["captures"][0]
        capture["rows"][1]["storage_token"] = capture["rows"][0]["storage_token"]
        capture["rows"][1]["view_token"] = digest("resealed-view")
        reseal_capture(capture)
        inputs["producer_b_audit"] = audit_result(inputs["producer_b"], self.protocol)
        inputs["producer_b_audit"].update(
            {
                "input_raw_sha256": inputs["file_hashes"]["producer_b"],
                "protocol_raw_sha256": self.protocol_sha,
                "preexecution_census_bound": True,
            }
        )
        inputs["producer_b_replay"]["input_result_sha256"] = sha256_json(
            inputs["producer_b"]
        )
        with self.assertRaises(DualRepeatFailure) as caught:
            verify_dual_repeat(**inputs)
        self.assertEqual(caught.exception.code, "cross_producer_relation_mismatch")

    def test_reused_producer_pid_fails_freshness_gate(self) -> None:
        inputs = self.inputs()
        for cell in inputs["producer_b"]["cells"]:
            for capture in cell["captures"]:
                capture["producer_pid"] = 1001
        inputs["producer_b_replay"]["input_result_sha256"] = sha256_json(
            inputs["producer_b"]
        )
        with self.assertRaises(DualRepeatFailure) as caught:
            verify_dual_repeat(**inputs)
        self.assertEqual(caught.exception.code, "fresh_producer_process")

    def test_input_binding_drift_fails_before_cross_comparison(self) -> None:
        inputs = self.inputs()
        inputs["producer_b"]["input_receipt"]["rank"] = 1
        inputs["producer_b_replay"]["input_result_sha256"] = sha256_json(
            inputs["producer_b"]
        )
        with self.assertRaises(DualRepeatFailure) as caught:
            verify_dual_repeat(**inputs)
        self.assertEqual(caught.exception.code, "input_binding_drift")


if __name__ == "__main__":
    unittest.main()
