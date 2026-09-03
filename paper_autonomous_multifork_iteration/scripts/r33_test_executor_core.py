from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import r33_clean_only_dry_run as dry
import r33_executor_core as core


class R33ExecutorCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="r33-executor-test-")
        self.output_dir = Path(self.temporary.name) / "clean"
        dry.run(self.output_dir)
        self.protocol = json.loads((self.output_dir / "protocol.json").read_text())
        self.clean = json.loads((self.output_dir / "clean-result.json").read_text())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_valid_clean_only_aggregate_is_explicitly_non_scientific(self) -> None:
        result = core.aggregate_reports(
            protocol=self.protocol,
            clean_report=self.clean,
            fault_reports=[],
        )
        self.assertTrue(result["clean_gate_passed"])
        self.assertFalse(result["scientific_valid"])
        self.assertFalse(result["fault_module_loaded"])
        self.assertFalse(result["faults_executed"])

    def test_fault_start_is_blocked_until_clean_gate_passes(self) -> None:
        gate = core.CleanGate(
            run_id=self.protocol["run_id"],
            protocol_sha256=core.sha256_json(self.protocol),
            execution_input_sha256=self.protocol["execution_input_sha256"],
        )
        with self.assertRaisesRegex(core.R33ContractError, "clean gate not passed"):
            gate.begin_fault("HF-TEST")
        gate.accept_clean(self.clean)
        receipt = gate.begin_fault("HF-TEST")
        self.assertTrue(receipt["fault_start_authorized"])

    def test_clean_semantic_tamper_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.clean)
        tampered["comparisons"]["canonical_fp32_logits_byte_exact"] = False
        with self.assertRaises(core.R33ContractError):
            core.validate_clean_report(tampered)

    def test_receipt_chain_payload_tamper_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.clean)
        tampered["receipt_chain"][2]["payload"]["all_existing_gates_enabled"] = False
        with self.assertRaisesRegex(core.R33ContractError, "payload digest"):
            core.validate_clean_report(tampered)

    def test_unexpected_pass_sentinel_is_never_a_detector_catch(self) -> None:
        def operation() -> None:
            raise core.UnexpectedFaultPass("sentinel")

        result = core.classify_fault_operation(operation)
        self.assertEqual(result["classification"], "operational_invalid")
        self.assertFalse(result["scientific_outcome_available"])

    def test_only_authenticated_rejection_is_a_catch(self) -> None:
        def operation() -> None:
            raise core.AuthenticatedValidatorRejection(
                "EXISTING_GATE", "rejected", {"authenticated": True}
            )

        result = core.classify_fault_operation(operation)
        self.assertEqual(result["classification"], "caught_by_existing_validator")
        self.assertTrue(result["scientific_outcome_available"])
        self.assertEqual(result["gate_id"], "EXISTING_GATE")

    def test_mutation_restores_when_body_raises(self) -> None:
        state = {"value": 1, "restores": 0}

        def apply() -> dict[str, object]:
            state["value"] = 2
            return {"mutation_observed": True}

        def restore() -> dict[str, object]:
            state["value"] = 1
            state["restores"] += 1
            return {
                "restoration_observed": True,
                "target_restored_exact": True,
                "non_target_preserved_across_undo": True,
            }

        with self.assertRaisesRegex(ValueError, "body failed"):
            with core.MutationTransaction(apply, restore):
                self.assertEqual(state["value"], 2)
                raise ValueError("body failed")
        self.assertEqual(state, {"value": 1, "restores": 1})

    def _formal_protocol_and_clean(self) -> tuple[dict[str, object], dict[str, object]]:
        protocol = copy.deepcopy(self.protocol)
        protocol["mode"] = "formal_fresh_faults"
        protocol["fault_ids"] = ["HF-TEST"]
        protocol["author_freeze_manifest_sha256"] = "a" * 64
        protocol["fault_bindings"] = {
            "HF-TEST": {
                "fault_id": "HF-TEST",
                "rank": 0,
                "expected_primary_gate": "TEST_GATE",
                "fault_definition_sha256": "b" * 64,
            }
        }
        clean = copy.deepcopy(self.clean)
        clean["local_dry_run"] = False
        clean["scientific_result"] = True
        clean["protocol_sha256"] = core.sha256_json(protocol)
        return protocol, clean

    def test_formal_aggregate_refuses_missing_fault_artifact(self) -> None:
        protocol, clean = self._formal_protocol_and_clean()
        with self.assertRaisesRegex(core.R33ContractError, "exact fault artifact count"):
            core.aggregate_reports(
                protocol=protocol,
                clean_report=clean,
                fault_reports=[],
            )

    def test_formal_aggregate_refuses_operationally_invalid_case(self) -> None:
        protocol, clean = self._formal_protocol_and_clean()
        invalid = {
            "schema_version": core.FAULT_SCHEMA,
            "run_id": protocol["run_id"],
            "case_id": "HF-TEST",
            "fault_id": "HF-TEST",
            "status": "operational_invalid",
            "classification": "operational_invalid",
        }
        with self.assertRaises(core.R33ContractError):
            core.aggregate_reports(
                protocol=protocol,
                clean_report=clean,
                fault_reports=[invalid],
            )


if __name__ == "__main__":
    unittest.main()
