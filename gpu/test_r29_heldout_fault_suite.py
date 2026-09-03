from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import r29_heldout_fault_suite as suite


def _fake_group() -> tuple[SimpleNamespace, SimpleNamespace]:
    layer_index = 7
    arena = SimpleNamespace(document_blocks_per_sequence=4)
    requests = []
    for request_index in range(2):
        sequence = SimpleNamespace(
            arena=arena,
            block_table=torch.tensor([[0, 1, 2, 3, -1, -1]], dtype=torch.int32),
            reservations=torch.tensor(
                [[10 + request_index * 2, 11 + request_index * 2]],
                dtype=torch.int64,
            ),
        )
        layer = SimpleNamespace(sequence=sequence)
        layers = [None] * (layer_index + 1)
        layers[layer_index] = layer
        requests.append(SimpleNamespace(layers=layers))
    group = SimpleNamespace(requests=tuple(requests))
    plan = SimpleNamespace(full_attention_layer_indices=(layer_index,))
    return group, plan


class HeldOutFaultSuiteTests(unittest.TestCase):
    def test_h01_swaps_only_selected_route_and_restores(self) -> None:
        group, plan = _fake_group()
        target = group.requests[0].layers[7].sequence.block_table
        original = target.clone()
        handle = suite.apply_state_fault("H01", group, plan)
        self.assertTrue(handle.applied_receipt()["mutation_observed"])
        self.assertEqual(target.tolist(), [[0, 2, 1, 3, -1, -1]])
        receipt = handle.restore()
        self.assertTrue(receipt["restoration_observed"])
        self.assertTrue(torch.equal(target, original))

    def test_h03_reuses_only_future_reservation_and_restores(self) -> None:
        group, plan = _fake_group()
        target = group.requests[0].layers[7].sequence.reservations
        original = target.clone()
        peer = group.requests[1].layers[7].sequence.reservations
        handle = suite.apply_state_fault("H03", group, plan)
        self.assertEqual(int(target[0, 0]), int(original[0, 0]))
        self.assertEqual(int(target[0, 1]), int(peer[0, 1]))
        self.assertNotEqual(int(target[0, 1]), int(original[0, 1]))
        receipt = handle.restore()
        self.assertTrue(receipt["restoration_observed"])
        self.assertTrue(torch.equal(target, original))

    def test_h02_is_exact_same_token_hidden_retry(self) -> None:
        action = suite.h02_action_sequence()
        self.assertEqual(action["advertised_logical_advance_tokens"], 1)
        self.assertEqual(action["actual_model_invocations"], 2)
        self.assertEqual(
            [row["input_coordinate"] for row in action["events"]],
            ["frozen_query_bank[rank][0][31]"] * 2,
        )
        self.assertEqual(
            [row["externally_advertised"] for row in action["events"]],
            [True, False],
        )

    def test_unknown_state_fault_is_rejected(self) -> None:
        group, plan = _fake_group()
        with self.assertRaises(suite.HeldOutFaultConfigurationError):
            suite.apply_state_fault("H02", group, plan)

    def test_validator_rejects_detector_mapping(self) -> None:
        value = {
            "schema_version": suite.SUITE_SCHEMA,
            "faults": [{"fault_id": fault_id} for fault_id in suite.FAULT_IDS],
            "detection_rate_reported": False,
            "author_executor_separation": {"executor_must_not_edit_suite": True},
            "implementation_bindings": suite.implementation_bindings(),
            "expected_gate": "forbidden",
        }
        with self.assertRaises(suite.HeldOutFaultConfigurationError):
            suite.validate_frozen_suite(value)

    def test_frozen_suite_if_present(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "paper_autonomous_multifork_iteration"
            / "evidence"
            / "r29_heldout_faults"
            / "preregistration"
            / "heldout-fault-suite.json"
        )
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            receipt = suite.validate_frozen_suite(value)
            self.assertFalse(receipt["contains_expected_detector_mapping"])


if __name__ == "__main__":
    unittest.main()
