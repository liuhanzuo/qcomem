from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from types import SimpleNamespace

from qcomem_forkaudit_mutants import (
    AppliedMutation,
    CampaignPhase,
    EXPECTED_GATE_IDS,
    ExecutionBoundary,
    FaultActivationRegistry,
    FaultCampaignConfigurationError,
    InjectionStage,
    M1_RESERVATION_ALIAS,
    M2_WRONG_SEQUENCE,
    M3_TAIL_COW_OMISSION,
    M4_GDN_BASE_ALIAS,
    M5_GDN_PEER_ALIAS,
    M6_POSITION_OFF_BY_ONE,
    M7_MASK_VIOLATION,
    M8_CALLABLE_SWAP,
    M9_DENSE_FALLBACK,
    MUTANT_IDS,
    MUTANT_SPECS,
    OutcomeClassification,
    RuntimeInvariantError,
    TargetMutationBinding,
    attribute_value_injector,
    callback_injector,
    mapping_value_injector,
    run_clean_case,
    run_mutant_case,
    validate_campaign_outcomes,
    validate_target_mutation_binding,
)


class ForkAuditMutantCampaignTest(unittest.TestCase):
    def test_target_binding_is_generated_inside_same_injector_lifecycle(self):
        state = {"reservation": "request-0", "peer": "request-1"}

        def digest() -> str:
            return hashlib.sha256(state["reservation"].encode("ascii")).hexdigest()

        def apply(context):
            before = digest()
            context["reservation"] = context["peer"]
            mutated = digest()

            def undo():
                context["reservation"] = "request-0"

            return AppliedMutation(
                undo=undo,
                verify_restored=lambda: context["reservation"] == "request-0",
                target_binding=TargetMutationBinding(
                    mutant_id=M1_RESERVATION_ALIAS,
                    case_cell_id="rank0-M1-fresh-cache",
                    capture_id="capture-M1-live-target",
                    target_kind="kv_reservation_table",
                    target_field="physical_block_ids",
                    pre_sha256=before,
                    mutated_sha256=mutated,
                    capture_restored_sha256=digest,
                ),
            )

        outcome = run_mutant_case(
            M1_RESERVATION_ALIAS,
            callback_injector(apply),
            lambda _context: (_ for _ in ()).throw(
                RuntimeInvariantError("KV_RESERVATION_DISJOINT", "alias")
            ),
            context=state,
        )
        self.assertTrue(outcome.detector_satisfied)
        binding = outcome.mutation_receipt.target_mutation_binding
        validated = validate_target_mutation_binding(
            binding,
            mutant_id=M1_RESERVATION_ALIAS,
            case_cell_id="rank0-M1-fresh-cache",
            target_kind="kv_reservation_table",
            target_field="physical_block_ids",
        )
        self.assertEqual(validated["pre_sha256"], validated["restored_sha256"])
        self.assertNotEqual(validated["pre_sha256"], validated["mutated_sha256"])
        self.assertEqual(state["reservation"], "request-0")

    def test_fixed_registry_has_nine_unique_expected_gates_and_stages(self):
        self.assertEqual(
            MUTANT_IDS,
            (
                M1_RESERVATION_ALIAS,
                M2_WRONG_SEQUENCE,
                M3_TAIL_COW_OMISSION,
                M4_GDN_BASE_ALIAS,
                M5_GDN_PEER_ALIAS,
                M6_POSITION_OFF_BY_ONE,
                M7_MASK_VIOLATION,
                M8_CALLABLE_SWAP,
                M9_DENSE_FALLBACK,
            ),
        )
        self.assertEqual(len(MUTANT_SPECS), 9)
        self.assertEqual(set(EXPECTED_GATE_IDS), set(MUTANT_IDS))
        self.assertEqual(
            MUTANT_SPECS[M3_TAIL_COW_OMISSION].injection_stage,
            InjectionStage.BEFORE_FIRST_APPEND,
        )
        self.assertEqual(
            MUTANT_SPECS[M8_CALLABLE_SWAP].expected_gate_id,
            "KERNEL_CALLABLE_ID",
        )
        self.assertEqual(
            MUTANT_SPECS[M6_POSITION_OFF_BY_ONE].expected_gate_id,
            "POSITION_CANONICAL_VALUES",
        )
        self.assertEqual(
            MUTANT_SPECS[M4_GDN_BASE_ALIAS].expected_gate_id,
            "gdn_completed_vs_base_disjoint",
        )
        self.assertEqual(
            MUTANT_SPECS[M5_GDN_PEER_ALIAS].expected_gate_id,
            "gdn_completed_vs_peers_disjoint",
        )

    def test_clean_pass_and_clean_false_positive_are_separate(self):
        clean = run_clean_case(lambda context: context.update(executed=True))
        self.assertEqual(clean.classification, OutcomeClassification.CLEAN_PASS)
        self.assertTrue(clean.aggregate_eligible)

        def false_positive(_context):
            raise RuntimeInvariantError("MASK_CONTRACT", "clean control rejected")

        outcome = run_clean_case(false_positive)
        self.assertEqual(
            outcome.classification, OutcomeClassification.CLEAN_FALSE_POSITIVE
        )
        self.assertEqual(outcome.observed_gate_id, "MASK_CONTRACT")
        self.assertTrue(outcome.scientifically_valid)
        self.assertFalse(outcome.aggregate_eligible)

    def test_expected_detection_restores_mapping_state(self):
        state = {"reservation": 11, "peer_reservation": 7}
        injector = mapping_value_injector(
            "reservation", lambda context, _old: context["peer_reservation"]
        )

        def exercise(context):
            self.assertEqual(context["reservation"], 7)
            raise RuntimeInvariantError(
                EXPECTED_GATE_IDS[M1_RESERVATION_ALIAS], "aliased reservation"
            )

        outcome = run_mutant_case(
            M1_RESERVATION_ALIAS, injector, exercise, context=state
        )
        self.assertEqual(
            outcome.classification, OutcomeClassification.DETECTED_EXPECTED_GATE
        )
        self.assertTrue(outcome.detector_satisfied)
        self.assertTrue(outcome.aggregate_eligible)
        self.assertTrue(outcome.exercise_started)
        self.assertFalse(outcome.exercise_completed)
        self.assertTrue(outcome.restoration_verified)
        self.assertEqual(
            outcome.failure_origin, ExecutionBoundary.DETECTOR_EXERCISE
        )
        self.assertIsNotNone(outcome.mutation_receipt)
        self.assertTrue(outcome.mutation_receipt.mutation_applied)
        self.assertTrue(outcome.mutation_receipt.injector_exit_completed)
        self.assertTrue(outcome.mutation_receipt.restoration_verified)
        self.assertEqual(
            outcome.to_dict()["mutation_receipt"]["injection_stage"],
            InjectionStage.AFTER_REQUEST_CONSTRUCTION.value,
        )
        self.assertEqual(state["reservation"], 11)

    def test_wrong_gate_is_not_expected_detection_and_restores_attribute(self):
        original = object()
        peer = object()
        layer = SimpleNamespace(sequence=original)
        state = {"layer": layer, "peer_sequence": peer}
        injector = attribute_value_injector(
            "layer", "sequence", lambda context, _old: context["peer_sequence"]
        )

        def exercise(context):
            self.assertIs(context["layer"].sequence, peer)
            raise RuntimeInvariantError("KV_PAGED_VIEW", "wrong detector fired")

        outcome = run_mutant_case(
            M2_WRONG_SEQUENCE, injector, exercise, context=state
        )
        self.assertEqual(
            outcome.classification, OutcomeClassification.DETECTED_WRONG_GATE
        )
        self.assertEqual(outcome.expected_gate_id, "KV_SEQUENCE_ID")
        self.assertEqual(outcome.observed_gate_id, "KV_PAGED_VIEW")
        self.assertFalse(outcome.detector_satisfied)
        self.assertIs(layer.sequence, original)

    def test_escape_is_valid_negative_result_not_a_crash(self):
        state = {"position": 20}
        injector = mapping_value_injector(
            "position", lambda _context, old: old + 1
        )
        outcome = run_mutant_case(
            M6_POSITION_OFF_BY_ONE,
            injector,
            lambda context: self.assertEqual(context["position"], 21),
            context=state,
        )
        self.assertEqual(outcome.classification, OutcomeClassification.ESCAPED)
        self.assertTrue(outcome.scientifically_valid)
        self.assertFalse(outcome.detector_satisfied)
        self.assertFalse(outcome.aggregate_eligible)
        self.assertIsNone(outcome.error_type)
        self.assertTrue(outcome.exercise_started)
        self.assertTrue(outcome.exercise_completed)
        self.assertTrue(outcome.restoration_verified)
        self.assertEqual(state["position"], 20)

    def test_unexpected_crash_is_invalid_and_state_is_still_restored(self):
        state = {"mask": None}
        injected_mask = object()
        injector = mapping_value_injector(
            "mask", lambda _context, _old: injected_mask
        )

        def crash(context):
            self.assertIs(context["mask"], injected_mask)
            raise OSError("unrelated runtime failure")

        outcome = run_mutant_case(
            M7_MASK_VIOLATION, injector, crash, context=state
        )
        self.assertEqual(
            outcome.classification, OutcomeClassification.UNEXPECTED_CRASH
        )
        self.assertFalse(outcome.scientifically_valid)
        self.assertEqual(outcome.error_type, "OSError")
        self.assertEqual(
            outcome.failure_origin, ExecutionBoundary.DETECTOR_EXERCISE
        )
        self.assertTrue(outcome.restoration_verified)
        self.assertIsNone(state["mask"])

    def test_expected_gate_from_injector_enter_is_invalid_not_detected(self):
        expected_gate = EXPECTED_GATE_IDS[M1_RESERVATION_ALIAS]

        class EnterSpoof:
            def __enter__(self):
                raise RuntimeInvariantError(expected_gate, "spoof from __enter__")

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                return False

        outcome = run_mutant_case(
            M1_RESERVATION_ALIAS,
            lambda _context: EnterSpoof(),
            lambda _context: self.fail("exercise must not start"),
        )
        self.assertEqual(
            outcome.classification, OutcomeClassification.UNEXPECTED_CRASH
        )
        self.assertFalse(outcome.detector_satisfied)
        self.assertFalse(outcome.aggregate_eligible)
        self.assertFalse(outcome.scientifically_valid)
        self.assertFalse(outcome.exercise_started)
        self.assertFalse(outcome.exercise_completed)
        self.assertFalse(outcome.restoration_verified)
        self.assertIsNone(outcome.observed_gate_id)
        self.assertEqual(outcome.boundary_gate_id, expected_gate)
        self.assertEqual(outcome.failure_origin, ExecutionBoundary.INJECTOR_APPLY)
        self.assertIsNotNone(outcome.mutation_receipt)
        self.assertTrue(outcome.mutation_receipt.injector_enter_started)
        self.assertFalse(outcome.mutation_receipt.injector_enter_completed)

    def test_expected_gate_from_injector_exit_is_invalid_not_detected(self):
        expected_gate = EXPECTED_GATE_IDS[M1_RESERVATION_ALIAS]
        state = {"mutated": False}

        class ExitSpoof:
            def __enter__(self):
                state["mutated"] = True
                return self

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                state["mutated"] = False
                raise RuntimeInvariantError(expected_gate, "spoof from __exit__")

        outcome = run_mutant_case(
            M1_RESERVATION_ALIAS,
            lambda _context: ExitSpoof(),
            lambda _context: self.assertTrue(state["mutated"]),
            context=state,
        )
        self.assertEqual(
            outcome.classification, OutcomeClassification.UNEXPECTED_CRASH
        )
        self.assertFalse(outcome.detector_satisfied)
        self.assertFalse(outcome.aggregate_eligible)
        self.assertFalse(outcome.scientifically_valid)
        self.assertTrue(outcome.exercise_started)
        self.assertTrue(outcome.exercise_completed)
        self.assertFalse(outcome.restoration_verified)
        self.assertIsNone(outcome.observed_gate_id)
        self.assertEqual(outcome.boundary_gate_id, expected_gate)
        self.assertEqual(
            outcome.failure_origin, ExecutionBoundary.INJECTOR_RESTORE
        )
        self.assertFalse(state["mutated"])
        self.assertTrue(outcome.mutation_receipt.injector_exit_started)
        self.assertFalse(outcome.mutation_receipt.injector_exit_completed)

    def test_missing_restoration_verifier_fails_closed(self):
        expected_gate = EXPECTED_GATE_IDS[M1_RESERVATION_ALIAS]

        class NoVerifier:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, traceback):
                del exc_type, exc, traceback
                return False

        outcome = run_mutant_case(
            M1_RESERVATION_ALIAS,
            lambda _context: NoVerifier(),
            lambda _context: (_ for _ in ()).throw(
                RuntimeInvariantError(expected_gate, "real detector interval")
            ),
        )
        self.assertEqual(
            outcome.classification, OutcomeClassification.UNEXPECTED_CRASH
        )
        self.assertEqual(outcome.observed_gate_id, expected_gate)
        self.assertIsNone(outcome.boundary_gate_id)
        self.assertEqual(
            outcome.failure_origin, ExecutionBoundary.INJECTOR_RESTORE
        )
        self.assertFalse(outcome.restoration_verified)
        self.assertFalse(outcome.mutation_receipt.restoration_verifier_present)

    def test_clean_phase_cannot_load_or_activate_mutants(self):
        registry = FaultActivationRegistry()
        injector = mapping_value_injector("value", lambda _context, _old: 2)
        with registry.campaign(CampaignPhase.CLEAN):
            with self.assertRaisesRegex(
                FaultCampaignConfigurationError, "cannot load"
            ):
                registry.load(M1_RESERVATION_ALIAS, injector)
            with self.assertRaisesRegex(
                FaultCampaignConfigurationError, "cannot activate"
            ):
                with registry.activate(
                    M1_RESERVATION_ALIAS,
                    InjectionStage.AFTER_REQUEST_CONSTRUCTION,
                    {"value": 1},
                ):
                    pass
        self.assertEqual(registry.phase, CampaignPhase.IDLE)
        self.assertEqual(registry.loaded_mutant_ids, ())
        self.assertIsNone(registry.active_mutant_id)

    def test_registry_rejects_wrong_stage_and_restores_after_context(self):
        registry = FaultActivationRegistry()
        injector = mapping_value_injector("value", lambda _context, _old: 2)
        with registry.campaign(
            CampaignPhase.MUTANT, mutant_id=M1_RESERVATION_ALIAS
        ):
            registry.load(M1_RESERVATION_ALIAS, injector)
            with self.assertRaisesRegex(
                FaultCampaignConfigurationError, "must activate at"
            ):
                with registry.activate(
                    M1_RESERVATION_ALIAS,
                    InjectionStage.ATTENTION_DISPATCH,
                    {"value": 1},
                ):
                    pass
        self.assertEqual(registry.phase, CampaignPhase.IDLE)
        self.assertEqual(registry.activation_count, 0)

    def test_aggregate_fails_closed_and_preserves_escape(self):
        clean = run_clean_case(lambda _context: None)
        outcomes = {}
        for mutant_id in MUTANT_IDS:
            expected = EXPECTED_GATE_IDS[mutant_id]

            def exercise(_context, gate=expected):
                raise RuntimeInvariantError(gate, "detected")

            outcomes[mutant_id] = run_mutant_case(
                mutant_id,
                mapping_value_injector("armed", lambda _context, _old: True),
                exercise,
                context={"armed": False},
            )
        aggregate = validate_campaign_outcomes(clean, outcomes)
        self.assertTrue(aggregate["passed"])
        self.assertEqual(aggregate["binding_errors"], {})

        correctly_bound = outcomes[M1_RESERVATION_ALIAS]
        outcomes[M1_RESERVATION_ALIAS] = replace(
            correctly_bound, mutant_id=M2_WRONG_SEQUENCE
        )
        aggregate = validate_campaign_outcomes(clean, outcomes)
        self.assertFalse(aggregate["passed"])
        self.assertIn(
            "key_mutant_id", aggregate["binding_errors"][M1_RESERVATION_ALIAS]
        )

        outcomes[M1_RESERVATION_ALIAS] = replace(
            correctly_bound, observed_gate_id="KV_PAGED_VIEW"
        )
        aggregate = validate_campaign_outcomes(clean, outcomes)
        self.assertFalse(aggregate["passed"])
        self.assertIn(
            "expected_detection_binding",
            aggregate["binding_errors"][M1_RESERVATION_ALIAS],
        )

        binding_spoofs = (
            (replace(correctly_bound, phase=CampaignPhase.CLEAN), "phase"),
            (replace(correctly_bound, mutant_name="different"), "spec_name"),
            (
                replace(
                    correctly_bound,
                    injection_stage=InjectionStage.ATTENTION_DISPATCH,
                ),
                "spec_stage",
            ),
            (
                replace(correctly_bound, expected_gate_id="different"),
                "spec_expected_gate",
            ),
            (
                replace(
                    correctly_bound,
                    classification=OutcomeClassification.ESCAPED,
                ),
                "escape_binding",
            ),
        )
        for spoof, expected_error in binding_spoofs:
            with self.subTest(expected_error=expected_error):
                outcomes[M1_RESERVATION_ALIAS] = spoof
                aggregate = validate_campaign_outcomes(clean, outcomes)
                self.assertFalse(aggregate["passed"])
                self.assertIn(
                    expected_error,
                    aggregate["binding_errors"][M1_RESERVATION_ALIAS],
                )

        outcomes[M1_RESERVATION_ALIAS] = correctly_bound

        receipt_fields = (
            "injector_factory_started",
            "injector_factory_completed",
            "injector_enter_started",
            "injector_enter_completed",
            "mutation_applied",
            "injector_exit_started",
            "injector_exit_completed",
            "restoration_verifier_present",
            "restoration_verified",
        )
        for field_name in receipt_fields:
            with self.subTest(receipt_field=field_name):
                spoofed_receipt = replace(
                    correctly_bound.mutation_receipt,
                    **{field_name: False},
                )
                outcomes[M1_RESERVATION_ALIAS] = replace(
                    correctly_bound,
                    mutation_receipt=spoofed_receipt,
                    restoration_verified=(
                        False
                        if field_name == "restoration_verified"
                        else correctly_bound.restoration_verified
                    ),
                )
                aggregate = validate_campaign_outcomes(clean, outcomes)
                self.assertFalse(aggregate["passed"])
                self.assertIn(
                    "expected_detection_binding",
                    aggregate["binding_errors"][M1_RESERVATION_ALIAS],
                )
        outcomes[M1_RESERVATION_ALIAS] = correctly_bound

        outcomes[M9_DENSE_FALLBACK] = run_mutant_case(
            M9_DENSE_FALLBACK,
            mapping_value_injector("dense", lambda _context, _old: True),
            lambda _context: None,
            context={"dense": False},
        )
        aggregate = validate_campaign_outcomes(clean, outcomes)
        self.assertFalse(aggregate["passed"])
        self.assertEqual(aggregate["escaped_mutant_ids"], [M9_DENSE_FALLBACK])


if __name__ == "__main__":
    unittest.main()
