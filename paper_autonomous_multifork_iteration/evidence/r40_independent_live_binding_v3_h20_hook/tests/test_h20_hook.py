from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "executed_source"))

from r40_h20_binding_protocol import REGISTRATION_SCHEMA  # noqa: E402
from r40_h20_hook import (  # noqa: E402
    BindingCampaignSession,
    PHASE_SETUP,
    PHASE_TRANSITION,
    install_h20_live_binding_hooks,
)
from r40_h20_registrar import RegistrationOracle, _payload as registrar_payload  # noqa: E402
from r40_h20_observer import _payload as observer_payload  # noqa: E402


def preregistration() -> dict:
    return json.loads((ROOT / "preregistration.json").read_text(encoding="utf-8"))


def state_owner(owner_number: int, layers: list[int]) -> SimpleNamespace:
    rows = []
    for layer_index in range(max(layers) + 1):
        base = float(1000 * owner_number + 10 * layer_index)
        rows.append(
            SimpleNamespace(
                conv_states={0: torch.tensor([base + 1, base + 2], dtype=torch.bfloat16)},
                recurrent_states={0: torch.tensor([base + 3, base + 4], dtype=torch.bfloat16)},
            )
        )
    return SimpleNamespace(layers=rows)


def fixture_objects():
    layers = [0, 1, 2, 4]
    persistent = state_owner(1, layers)
    requests = [state_owner(index + 2, layers) for index in range(8)]
    # The real materialized request base is a distinct-storage clone with the
    # same values. Freeze that hard LB04 condition in the local fixture.
    requests[1].layers[0].recurrent_states[0] = persistent.layers[0].recurrent_states[0].clone()
    return persistent, SimpleNamespace(requests=requests), SimpleNamespace(linear_layer_indices=layers)


def metadata() -> dict:
    return {
        "cell_role": "ownership_witness",
        "resident_count": 8,
        "kv_policy": "vllm-q16-shared-document-reuse",
        "gdn_base_policy": "materialize-request-base-functional-rebind",
        "arm_id": "unit-test-arm",
    }


def run_campaign(*, process_workers: bool) -> dict:
    persistent, group, plan = fixture_objects()
    session = BindingCampaignSession(
        preregistration(), rank=0, metadata=metadata(), process_workers=process_workers
    )
    try:
        session.register_initial(persistent, group, plan)
        session.run_phase(PHASE_SETUP)
        group.requests[0].layers[2].recurrent_states[0] = torch.tensor([9991.0, 9992.0], dtype=torch.bfloat16)
        session.run_phase(PHASE_TRANSITION)
        return session.payload()
    finally:
        session.close()


class HookTests(unittest.TestCase):
    def test_bfloat16_canonical_bytes_are_cross_path_stable_and_shape_bound(self) -> None:
        tensor = torch.tensor([[1.25, -2.5], [3.75, 4.0]], dtype=torch.bfloat16).t()
        registrar_bytes = registrar_payload(tensor)
        observer_bytes = observer_payload(tensor.clone())
        self.assertEqual(registrar_bytes, observer_bytes)
        self.assertNotEqual(registrar_bytes, registrar_payload(tensor.reshape(4)))
        self.assertIn(b"torch.bfloat16", registrar_bytes)

    def test_four_faults_and_four_matched_clean_controls_fail_closed(self) -> None:
        payload = run_campaign(process_workers=False)
        self.assertEqual(payload["clean_captures_passed"], 4)
        self.assertEqual(payload["mutants_failed_closed"], 4)
        self.assertEqual({row["fault_id"] for row in payload["fault_results"]}, {
            "R40-H20-LB01", "R40-H20-LB02", "R40-H20-LB03", "R40-H20-LB04"
        })
        self.assertTrue(all(not row["semantic_labels_mutated"] for row in payload["fault_results"]))
        self.assertEqual(payload["registration_event_count"], 40)
        self.assertFalse(payload["producer_manifest_sent"])
        self.assertFalse(payload["producer_slot_ids_sent_to_registrar"])
        lb04 = next(row for row in payload["fault_results"] if row["fault_id"] == "R40-H20-LB04")
        self.assertIn("storage_relation_mismatch", lb04["mutant_detector"]["failure_codes"])
        self.assertNotIn("challenge_response_mismatch", lb04["mutant_detector"]["failure_codes"])

    def test_spawned_registrar_and_observer_are_process_separated(self) -> None:
        payload = run_campaign(process_workers=True)
        self.assertTrue(payload["process_separated"])
        self.assertEqual(len({payload["producer_pid"], payload["registrar_pid"], payload["observer_pid"]}), 3)

    def test_registration_wire_rejects_producer_slot_labels(self) -> None:
        oracle = RegistrationOracle(preregistration())
        event = {
            "schema_version": REGISTRATION_SCHEMA, "operation": "initial",
            "owner_kind": "request", "request_index": 7, "layer_index": 4,
            "conv_states": {0: torch.ones(2)}, "recurrent_states": {0: torch.ones(2)},
            "slot_id": "producer-supplied-label",
        }
        with self.assertRaisesRegex(RuntimeError, "forbidden registration field"):
            oracle.register_containers(event)

    def test_hook_runs_reenumeration_before_original_phase_serializer(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        tmp_path = Path(temporary.name)
        persistent, group, plan = fixture_objects()
        order: list[str] = []
        runner = SimpleNamespace()

        def build(cache, received_plan, **kwargs):
            del kwargs
            self.assertIs(cache, persistent)
            self.assertIs(received_plan, plan)
            order.append("group-built")
            return group

        def phase(*args, **kwargs):
            del args
            order.append("producer-serializer:" + kwargs["phase"])

        def witness(**kwargs):
            del kwargs
            built = runner.build_resident_request_group(persistent, plan)
            self.assertIs(built, group)
            runner._write_witness_phase(phase=PHASE_SETUP)
            group.requests[0].layers[2].recurrent_states[0] = torch.tensor([8181.0, 8282.0], dtype=torch.bfloat16)
            runner._write_witness_phase(phase=PHASE_TRANSITION)
            return "unchanged-result"

        runner.build_resident_request_group = build
        runner._write_witness_phase = phase
        runner._run_ownership_witness_cell = witness
        restore = install_h20_live_binding_hooks(
            runner_module=runner, preregistration=preregistration(),
            capture_root=tmp_path, rank=0, process_workers=False,
        )
        try:
            result = runner._run_ownership_witness_cell(
                resident_count=8, kv_policy="vllm-q16-shared-document-reuse",
                gdn_base_policy="materialize-request-base-functional-rebind", arm_id="unit-test-arm",
            )
        finally:
            restore()
        self.assertEqual(result, "unchanged-result")
        self.assertEqual(order, ["group-built", "producer-serializer:setup_pre_transition", "producer-serializer:post_transition"])
        captured = json.loads((tmp_path / "rank-0/raw/independent-live-binding.json").read_text())
        self.assertEqual(captured["clean_captures_passed"], 4)
        self.assertEqual(captured["mutants_failed_closed"], 4)


if __name__ == "__main__":
    unittest.main()
