from __future__ import annotations

import ast
import copy
import unittest
from pathlib import Path

import torch

from r33_ipc_capture_protocol import (
    ProtocolError,
    REQUEST_SCHEMA,
    build_slot_manifest,
    validate_live_request,
)
from r33_out_of_process_capture import OutOfProcessCaptureSession, bind_live_tensors
from r33_replay_independent_capture import (
    PREREG_SCHEMA,
    POLICY_MATERIALIZED,
    ReplayError,
    evaluate_phase,
    replay_result,
)
from r33_run_local_capture_gate import (
    CAPTURE_IDS,
    LAYERS,
    make_persistent,
    make_request,
    run,
)


class R33IndependentCaptureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.result = run()
        cls.replay = replay_result(cls.result)

    def test_worker_imports_no_candidate_or_runtime_module(self) -> None:
        path = Path(__file__).with_name("r33_independent_capture_worker.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        self.assertFalse(
            any(
                term in module.lower()
                for module in modules
                for term in ("qcomem", "forkaudit", "transformers", "vllm")
            ),
            modules,
        )

    def test_full_engineering_gate_is_process_separated_and_replays(self) -> None:
        self.assertTrue(self.replay["passed"])
        self.assertTrue(self.replay["all_observers_process_separated"])
        self.assertEqual(self.replay["cell_count"], 2)
        self.assertEqual(self.replay["row_observations"], 108)
        self.assertEqual(self.replay["relation_observations"], 918)
        producer_pid = self.result["producer_pid"]
        observer_pids = {
            capture["observer_pid"]
            for cell in self.result["cells"]
            for capture in cell["captures"]
        }
        self.assertEqual(len(observer_pids), 2)
        self.assertNotIn(producer_pid, observer_pids)
        for cell in self.result["cells"]:
            for capture in cell["captures"]:
                self.assertEqual(capture["judgment_fields_received"], [])
                self.assertFalse(capture["candidate_verdict_fields_received"])
                self.assertTrue(capture["all_cpu_tensors_shared_in_receiver"])

    def test_live_wire_rejects_phase_policy_and_verdict_fields(self) -> None:
        manifest = build_slot_manifest(
            LAYERS, resident_count=2, capture_ids=CAPTURE_IDS, state_index=0
        )
        persistent = make_persistent()
        requests = [make_request(persistent, borrowed=True) for _ in range(2)]
        base = {
            "schema_version": REQUEST_SCHEMA,
            "capture_id": CAPTURE_IDS[0],
            "slot_tensors": bind_live_tensors(manifest, persistent, requests),
        }
        validate_live_request(base, manifest)
        for field, value in (
            ("phase", "setup"),
            ("policy", "shared-base"),
            ("completed_request_indices", []),
            ("passed", True),
            ("candidate_verdict", True),
        ):
            with self.subTest(field=field):
                bad = dict(base)
                bad[field] = value
                with self.assertRaisesRegex(ProtocolError, "field-set"):
                    validate_live_request(bad, manifest)

    def test_receiver_detects_internal_alias_fault(self) -> None:
        manifest = build_slot_manifest(
            LAYERS, resident_count=2, capture_ids=CAPTURE_IDS, state_index=0
        )
        persistent = make_persistent()
        requests = [make_request(persistent, borrowed=False) for _ in range(2)]
        requests[0].layers[1].conv_states[0] = requests[0].layers[0].conv_states[0]
        with OutOfProcessCaptureSession(manifest, timeout_seconds=60.0) as session:
            capture = session.capture(CAPTURE_IDS[0], persistent, requests)
        with self.assertRaisesRegex(ReplayError, "overlapping"):
            evaluate_phase(
                capture,
                policy=POLICY_MATERIALIZED,
                completed_request_indices=[],
            )

    def test_archived_row_tamper_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.result)
        tampered["cells"][0]["captures"][0]["rows"][0]["content_sha256"] = "0" * 64
        with self.assertRaisesRegex(ReplayError, "row digest"):
            replay_result(tampered)

    def test_replay_binds_frozen_plan_and_rejects_plan_drift(self) -> None:
        first_cell = self.result["cells"][0]
        preregistration = {
            "schema_version": PREREG_SCHEMA,
            "design": {
                "gdn_policy_cells": [cell["policy"] for cell in self.result["cells"]],
                "capture_ids": [
                    capture["capture_id"] for capture in first_cell["captures"]
                ],
                "capture_plan": copy.deepcopy(first_cell["capture_plan"]),
                "linear_layer_indices": list(LAYERS),
                "resident_count": 2,
                "expected_transport": "torch-cpu-shared-memory-reduction",
                "rows_per_phase": 18,
                "unordered_pair_relations_per_phase": 153,
            },
        }
        bound = replay_result(self.result, preregistration)
        self.assertTrue(bound["passed"])
        self.assertTrue(bound["frozen_protocol_bound"])
        bad = copy.deepcopy(preregistration)
        bad["design"]["capture_plan"][1]["completed_request_indices"] = []
        with self.assertRaisesRegex(ReplayError, "frozen capture plan"):
            replay_result(self.result, bad)


if __name__ == "__main__":
    unittest.main()
