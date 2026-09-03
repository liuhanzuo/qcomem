from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import build_qcomem_qwen35_forkaudit_detector_matrix_v2 as builder
import replay_qcomem_qwen35_forkaudit_detector_matrix_v2 as replay_module
import qcomem_forkaudit_selective_gate_policy as gate_policy
import run_qcomem_qwen35_forkaudit_detector_matrix_v2 as runner_module
from qcomem_qwen35_vllm_paged_integration import Qwen35VllmPagedIntegrationError


class SelectiveGatePolicyRegressionTest(unittest.TestCase):
    @staticmethod
    def _modules(position_validator: object) -> tuple[object, object]:
        def runtime_require(condition: bool, gate_id: str, message: str) -> None:
            if not condition:
                raise RuntimeError(f"{gate_id}:{message}")

        def storage_require(
            condition: bool, message: str, *, gate_id: str = "storage"
        ) -> None:
            if not condition:
                raise RuntimeError(f"{gate_id}:{message}")

        def assert_disjoint(
            _left: object, _right: object, _label: str, *, gate_id: str
        ) -> int:
            if gate_id:
                return 1
            raise AssertionError("unreachable")

        resident = types.SimpleNamespace(
            _runtime_require=runtime_require,
            validate_qwen35_post_rope_position_ids=position_validator,
        )
        storage = types.SimpleNamespace(
            _require=storage_require,
            _assert_sets_disjoint=assert_disjoint,
            _rows_overlap=lambda _left, _right: False,
            _rows_exact_alias=lambda _left, _right: False,
            _coordinate_key=lambda row: tuple(sorted(row.items())),
        )
        return resident, storage

    def test_only_exact_m6_tail_error_replays_relaxed_and_restores_all_identities(self) -> None:
        calls: list[bool] = []

        def validator(
            _position_ids: object,
            *,
            query: object,
            total_length: int,
            strict_tail_values: bool,
        ) -> str:
            del query, total_length
            calls.append(strict_tail_values)
            if strict_tail_values:
                raise Qwen35VllmPagedIntegrationError(
                    "position_ids are not the canonical contiguous causal tail"
                )
            return "relaxed-continuation"

        resident, storage = self._modules(validator)
        originals = (
            resident._runtime_require,
            resident.validate_qwen35_post_rope_position_ids,
            storage._require,
            storage._assert_sets_disjoint,
        )
        policy = gate_policy.SelectiveGatePolicy(
            gate_policy.POSITION_GATE,
            resident_module=resident,
            storage_module=storage,
        )
        with policy:
            result = resident.validate_qwen35_post_rope_position_ids(
                object(), query=object(), total_length=17, strict_tail_values=True
            )
        self.assertEqual(result, "relaxed-continuation")
        self.assertEqual(calls, [True, False])
        self.assertEqual(policy.receipt()["suppressed_gate_ids"], [gate_policy.POSITION_GATE])
        self.assertIs(resident._runtime_require, originals[0])
        self.assertIs(resident.validate_qwen35_post_rope_position_ids, originals[1])
        self.assertIs(storage._require, originals[2])
        self.assertIs(storage._assert_sets_disjoint, originals[3])


    def test_unexpected_attribute_error_propagates_and_restores(self) -> None:
        def validator(*_args: object, **_kwargs: object) -> object:
            raise AttributeError("unexpected integration defect")

        resident, storage = self._modules(validator)
        original = resident.validate_qwen35_post_rope_position_ids
        with self.assertRaisesRegex(AttributeError, "unexpected integration defect"):
            with gate_policy.SelectiveGatePolicy(
                gate_policy.POSITION_GATE,
                resident_module=resident,
                storage_module=storage,
            ):
                resident.validate_qwen35_post_rope_position_ids(
                    object(), query=object(), total_length=1, strict_tail_values=True
                )
        self.assertIs(resident.validate_qwen35_post_rope_position_ids, original)

    def test_non_target_runtime_gate_is_never_suppressed(self) -> None:
        resident, storage = self._modules(lambda *_args, **_kwargs: None)
        policy = gate_policy.SelectiveGatePolicy(
            gate_policy.POSITION_GATE,
            resident_module=resident,
            storage_module=storage,
        )
        with self.assertRaisesRegex(RuntimeError, "MASK_CONTRACT"):
            with policy:
                resident._runtime_require(False, "MASK_CONTRACT", "wrong mask")
        self.assertEqual(policy.receipt()["suppressed_event_count"], 0)


class RunnerPairSemanticsRegressionTest(unittest.TestCase):
    def test_clean_self_semantics_survive_m7_classified_abort(self) -> None:
        clean_semantics = {
            "token_only": runner_module._detector_cell(
                "evaluated",
                False,
                exact_sha=True,
                argmax_equal=True,
                max_abs=None,
                relative_l2=None,
            ),
            "full_logit": runner_module._detector_cell(
                "evaluated",
                False,
                exact_sha=True,
                argmax_equal=True,
                max_abs=0.0,
                relative_l2=0.0,
            ),
        }
        clean = {
            "case_id": "M7:clean",
            "mutant_id": "M7",
            "outcome": {
                "completion_status": "completed",
                "semantics": copy.deepcopy(clean_semantics),
            },
        }
        mutant = {
            "case_id": "M7:target_suppressed",
            "mutant_id": "M7",
            "outcome": {
                "completion_status": "classified_abort",
                "classification": "production_assertion",
                "semantics": {
                    "token_only": {
                        "status": "not_evaluated",
                        "caught": None,
                        "exact_sha": None,
                        "argmax_equal": None,
                        "max_abs": None,
                        "relative_l2": None,
                    },
                    "full_logit": {
                        "status": "not_evaluated",
                        "caught": None,
                        "exact_sha": None,
                        "argmax_equal": None,
                        "max_abs": None,
                        "relative_l2": None,
                    },
                },
                "measured_non_forkaudit_escape": None,
            },
        }

        runner_module._attach_pair_semantics(
            clean,
            mutant,
            sidecars=[],
            rank_root=Path("."),
        )

        self.assertEqual(clean["outcome"]["semantics"], clean_semantics)
        self.assertEqual(
            mutant["outcome"]["semantics"]["token_only"]["status"],
            "not_evaluated",
        )
        self.assertIs(mutant["outcome"]["measured_non_forkaudit_escape"], False)


class DetectorMatrixV2ValidationTest(unittest.TestCase):
    maxDiff = None

    @staticmethod
    def _sha(label: str) -> str:
        return hashlib.sha256(label.encode("utf-8")).hexdigest()

    def _valid_external_pin(self, scope_sha256: str) -> dict[str, object]:
        binding = self.prereg["input_binding"]
        value = {
            "schema_version": builder.EXTERNAL_PIN_SCHEMA,
            "preexecution_revision": builder.EXTERNAL_PIN_PREEXECUTION_REVISION,
            "created_before_candidate_execution": True,
            "candidate_outputs_observed_at_creation": False,
            "experiment_id": "E-R28-FULL-DETECTOR-MATRIX",
            "scope_supersession": {
                "path": (
                    "paper_autonomous_multifork_iteration/evidence/"
                    "r28_full_detector_matrix/scope-supersession.json"
                ),
                "sha256": scope_sha256,
            },
            "inherited_rr2_authority": {
                "original_rr2_run_id": binding["original_rr2_run_id"],
                "original_rr2_receipt_manifest_sha256": binding[
                    "original_rr2_receipt_manifest_sha256"
                ],
                "executed_source_ledger_sha256": binding[
                    "imported_rr2_code_ledger_sha256"
                ],
            },
            "frozen_inputs": {
                "frozen_query_banks_sha256": binding[
                    "frozen_query_banks_sha256"
                ],
                "model": "Qwen/Qwen3.5-35B-A3B",
                "model_artifact_ledger_sha256": binding[
                    "artifact_ledger_raw_sha256"
                ],
                "model_revision": binding["model_revision"],
                "model_weight_ledger_sha256": binding[
                    "weight_ledger_raw_sha256"
                ],
                "pg19_data_sha256": binding["pg19_sha256"],
                "pg19_manifest_sha256": binding["pg19_manifest_sha256"],
                "pg19_windows_sha256": binding["windows_sha256"],
            },
            "fixed_stack": copy.deepcopy(builder.PIN_FIXED_STACK),
            "case_design": {
                "fresh_case_count": 18,
                "clean_cases": 9,
                "target_suppressed_mutant_cases": 9,
                "discarded_warmup_before_allocator_baseline": True,
                "rank_assignment": {
                    str(rank): list(ids)
                    for rank, ids in builder.ASSIGNMENT.items()
                },
                "case_order_per_fault": [
                    "fresh matched-clean with all gates enabled",
                    (
                        "separately rebuilt mutant with only the preregistered "
                        "target gate suppressed"
                    ),
                ],
                "teacher_forced_continuation_fault_ids": ["M3", "M4", "M5"],
                "teacher_forced_continuation_rule": copy.deepcopy(
                    builder.PIN_TEACHER_FORCING_RULE
                ),
            },
            "detector_policy": {
                "classification_precedence": copy.deepcopy(
                    builder.PIN_CLASSIFICATION_PRECEDENCE
                ),
                "r28_detectors": copy.deepcopy(builder.PIN_R28_DETECTORS),
                "production_assertion_allowlist": (
                    builder._pin_production_assertion_allowlist()
                ),
                "fault_payload_abort": copy.deepcopy(
                    builder.PIN_M8_FAULT_PAYLOAD
                ),
                "unallowlisted_exception_policy": (
                    builder.PIN_UNALLOWLISTED_EXCEPTION_POLICY
                ),
            },
        }
        return value

    def _validate_external_pin_fixture(
        self,
        pin: dict[str, object],
        *,
        scope_payload: bytes = b"unit-test-scope-v1\n",
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            scope_path = root / "scope.json"
            scope_path.write_bytes(scope_payload)
            pin_path = root / "pin.json"
            builder.write_json(pin_path, pin)
            return builder.validate_external_pin_payload(
                external_pin_payload=pin_path,
                expected_external_pin_sha256=builder.sha256_file(pin_path),
                scope_supersession=scope_path,
                input_binding=self.prereg["input_binding"],
            )

    def setUp(self) -> None:
        self.prereg_sha = self._sha("preregistration")
        input_binding = {
            "model_revision": "frozen-revision",
            "weight_ledger_raw_sha256": self._sha("weights"),
            "artifact_ledger_raw_sha256": self._sha("artifacts"),
            "pg19_sha256": self._sha("pg19"),
            "pg19_manifest_sha256": self._sha("pg19-manifest"),
            "windows_sha256": self._sha("windows"),
            "frozen_query_banks_sha256": self._sha("queries"),
            "original_rr2_run_id": "rr2-w-run",
            "original_rr2_receipt_manifest_sha256": self._sha("rr2-manifest"),
            "code_ledger_sha256": self._sha("code"),
            "imported_rr2_code_ledger_sha256": self._sha("imported-rr2-code"),
            "external_pin_payload_sha256": self._sha("external-pin"),
        }
        self.prereg = {
            "schema_version": builder.PREREG_SCHEMA,
            "created_before_candidate_outputs": True,
            "workstream_id": "E-R28-FULL-DETECTOR-MATRIX",
            "runner_sha256": self._sha("runner"),
            "builder_sha256": self._sha("builder"),
            "replay_sha256": self._sha("replay"),
            "source_binding": {
                "runner_sha256": self._sha("runner"),
                "builder_sha256": self._sha("builder"),
                "replay_sha256": self._sha("replay"),
                "test_sha256": self._sha("test"),
                "launcher_sha256": self._sha("launcher"),
                "gate_policy_sha256": self._sha("gate-policy"),
                "qs_config_sha256": self._sha("qs-config"),
                "scope_supersession_sha256": self._sha("scope-supersession"),
                "external_pin_payload_sha256": self._sha("external-pin"),
            },
            "rank_schema_version": builder.RANK_SCHEMA,
            "summary_schema_version": builder.SUMMARY_SCHEMA,
            "assignment": {
                str(rank): list(ids) for rank, ids in builder.ASSIGNMENT.items()
            },
            "mutant_ids": list(builder.MUTANT_IDS),
            "lanes": list(builder.LANES),
            "expected_gate_ids": builder.EXPECTED_GATES,
            "gate_predicate_bindings": builder.EXPECTED_GATE_PREDICATES,
            "target_requests": builder.TARGET_REQUESTS,
            "target_contracts": builder.TARGET_CONTRACTS,
            "expected_horizon_stages": {
                mutant_id: list(stages)
                for mutant_id, stages in builder.EXPECTED_HORIZON_STAGES.items()
            },
            "minimum_fresh_cases": 18,
            "required_world_size": 8,
            "required_hardware": {
                "name": "NVIDIA H20-3e",
                "compute_capability": [9, 0],
                "distinct_gpu_uuids": 8,
            },
            "input_binding": input_binding,
            "external_pin_semantic_validation": (
                builder._external_pin_semantic_receipt(
                    payload_sha256=self._sha("external-pin"),
                    scope_sha256=self._sha("scope-supersession"),
                )
            ),
            "production_assertion_allowlist": builder.PRODUCTION_ASSERTION_ALLOWLISTS,
            "teacher_forcing": builder.TEACHER_FORCING,
            "m8_fault_payload_pin": builder.M8_FAULT_PAYLOAD_PIN,
            "classification_precedence": list(builder.CLASSIFICATION_PRECEDENCE),
            "comparison_definitions": builder.COMPARISON_DEFINITIONS,
            "partial_abort_policy": {
                "retain_and_hash_partial_stage_sidecars": True,
                "token_only_until_complete_horizon": "not_evaluated",
                "full_logit_until_complete_horizon": "not_evaluated",
                "not_evaluated_caught_value": None,
            },
            "hard_operational_validity_gates": {
                "discarded_warmup_precedes_post_model_allocator_baseline": True,
                "after_case_allocated_and_reserved_equal_post_warmup_baseline": True,
                "injector_target_restored": True,
                "selective_suppression_hooks_restored": True,
                "case_state_discarded": True,
                "registered_attention_backends_removed": True,
                "attention_implementation_restored": True,
                "traceback_references_cleared": True,
            },
            "policies": {
                "one_fresh_clean_and_one_fresh_target_suppressed_case_per_fault": True,
                "only_preregistered_target_gate_may_be_suppressed": True,
                "valid_scientific_negatives_retained": True,
                "not_evaluated_never_converted_to_not_caught": True,
                "operational_invalidity_rejected": True,
                "separate_rr2_all_gates_on_reference_not_pooled_as_rate": True,
            },
        }
        self.clean_sidecar_payload = np.ones(248320, dtype="<f4").tobytes()
        self.mutant_sidecar_payload = np.full(248320, 2.0, dtype="<f4").tobytes()
        self.sidecar_payloads: dict[str, bytes] = {}
        self.rank_payloads = [self._rank(rank) for rank in range(8)]
        self.rank_receipts = [
            {
                "rank": rank,
                "relative_path": f"detector-matrix-v2-rank-{rank}.json",
                "bytes": 1,
                "sha256": self._sha(f"rank-{rank}"),
            }
            for rank in range(8)
        ]
        self.rr2 = {
            mutant_id: {
                "source": "separate_rr2_all_gates_on_w_run",
                "run_id": "rr2-w-run",
                "rank": next(
                    rank for rank, ids in builder.ASSIGNMENT.items() if mutant_id in ids
                ),
                "shard_relative_path": "shard.json",
                "shard_sha256": self._sha(f"rr2-{mutant_id}"),
                "classification": "detected_expected_gate",
                "expected_gate_id": builder.EXPECTED_GATES[mutant_id],
                "observed_gate_id": builder.EXPECTED_GATES[mutant_id],
                "restoration_verified": True,
                "matched_clean_classification": "clean_pass",
            }
            for mutant_id in builder.MUTANT_IDS
        }

    def _policy_receipt(self, mutant_id: str, lane: str) -> dict[str, object]:
        expected = builder.EXPECTED_GATES[mutant_id]
        events: list[dict[str, object]] = []
        if lane == "target_suppressed":
            predicate = builder.EXPECTED_GATE_PREDICATES[mutant_id]
            callsite: dict[str, object] = {
                "module": "unit_test_bound_source",
                "file": predicate["callsite_file"],
                "function": predicate["callsite_function"],
                "line": 1,
                "source_line_sha256": self._sha(f"callsite-line-{mutant_id}"),
            }
            callsite["callsite_sha256"] = builder.sha256_json(callsite)
            event: dict[str, object] = {
                "schema_version": "forkaudit-selective-gate-event-v2",
                "gate_id": expected,
                "message": "preregistered target predicate was false",
                "predicate_function": predicate["predicate_function"],
                "predicate_source": predicate["predicate_source"],
                "ordinal": 0,
                "callsite": callsite,
            }
            event["event_sha256"] = builder.sha256_json(event)
            events.append(event)
        return {
            "schema_version": "forkaudit-selective-gate-policy-receipt-v2",
            "target_gate_id": None if lane == "clean" else expected,
            "lane": "all-gates-on" if lane == "clean" else "target-only-suppressed",
            "suppressed_event_count": len(events),
            "suppressed_gate_ids": [event["gate_id"] for event in events],
            "events": events,
            "scope_integrity_before_restore": True,
            "original_function_descriptors": {},
            "all_original_function_identities_restored": True,
        }

    @staticmethod
    def _semantic(*, caught: bool, detector: str) -> dict[str, object]:
        return {
            "status": "evaluated",
            "caught": caught,
            "exact_sha": not caught,
            "argmax_equal": True,
            "max_abs": None if detector == "token_only" else (0.0 if not caught else 1.0),
            "relative_l2": None if detector == "token_only" else (0.0 if not caught else 1.0),
        }

    @staticmethod
    def _not_evaluated_semantic() -> dict[str, object]:
        return {
            "status": "not_evaluated",
            "caught": None,
            "exact_sha": None,
            "argmax_equal": None,
            "max_abs": None,
            "relative_l2": None,
        }

    def _teacher(self, mutant_id: str) -> dict[str, object] | None:
        rule = builder.TEACHER_FORCING.get(mutant_id)
        if rule is None:
            return None
        return {
            **rule,
            "token_id": 1234 + int(mutant_id[1:]),
            "source_token_sha256": self._sha(f"{mutant_id}-teacher-token"),
            "source_coordinate": (
                f"frozen_query_bank[rank][{rule['request_index']}]"
                f"[{rule['query_token_index']}]"
            ),
            "independent_of_path_argmax": True,
            "argmax_feedback_used": False,
        }

    def _case(self, mutant_id: str, lane: str) -> dict[str, object]:
        expected = builder.EXPECTED_GATES[mutant_id]
        hook = self._policy_receipt(mutant_id, lane)
        clean = lane == "clean"
        case = {
            "case_id": f"{mutant_id}:{lane}",
            "mutant_id": mutant_id,
            "lane": lane,
            "expected_gate_id": expected,
            "target_request": builder.TARGET_REQUESTS[mutant_id],
            "freshness_receipt": {
                "case_nonce_sha256": self._sha(f"{mutant_id}:{lane}:nonce"),
                "fresh_persistent_cache": True,
                "fresh_request_cache_group": True,
                "prior_case_state_reused": False,
            },
            "outcome": {
                "completion_status": "completed",
                "classification": "completed_semantics",
                "fork_audit": {
                    "target_suppression_events": copy.deepcopy(hook["events"]),
                    "other_gate": {"id": None, "status": "evaluated", "caught": False},
                },
                "production": {
                    "assertion": {
                        "status": "evaluated",
                        "caught": False,
                        "allowlist_id": None,
                        "provenance": None,
                    },
                    "nonassertion_crash": {"status": "evaluated", "caught": False},
                    "fault_payload_abort": {"status": "evaluated", "caught": False},
                },
                "output_availability": {
                    "token": "evaluated",
                    "full_logit": "evaluated",
                    "sidecar": "evaluated",
                },
                "semantics": {
                    "token_only": self._semantic(caught=False, detector="token_only"),
                    "full_logit": self._semantic(caught=not clean, detector="full_logit"),
                },
                "measured_non_forkaudit_escape": None if clean else False,
                "mutation_receipt": (
                    {
                        "applied": False,
                        "mutant_id": mutant_id,
                        "teacher_forcing": self._teacher(mutant_id),
                    }
                    if clean
                    else {
                        "applied": True,
                        "mutant_id": mutant_id,
                        "teacher_forcing": self._teacher(mutant_id),
                        "target_contract": builder.TARGET_CONTRACTS[mutant_id],
                        "pre_descriptor_sha256": self._sha(f"{mutant_id}-pre"),
                        "mutated_descriptor_sha256": self._sha(f"{mutant_id}-mutated"),
                    }
                ),
                "injector_target_restoration": (
                    {"status": "not_applicable", "verified": True}
                    if clean
                    else {
                        "status": "evaluated",
                        "verified": True,
                        "pre_sha256": self._sha(f"{mutant_id}-pre"),
                        "mutated_sha256": self._sha(f"{mutant_id}-mutated"),
                        "restored_sha256": self._sha(f"{mutant_id}-pre"),
                        "target_contract": builder.TARGET_CONTRACTS[mutant_id],
                    }
                ),
                "case_discard_allocator_recovery": {
                    "verified": True,
                    "before_cell": {"allocated_bytes": 100, "reserved_bytes": 200},
                    "after_cleanup": {"allocated_bytes": 100, "reserved_bytes": 200},
                    "frozen_model_query_baseline": {
                        "allocated_bytes": 100,
                        "reserved_bytes": 200,
                    },
                    "gc_collect_completed": True,
                    "cuda_empty_cache_completed": True,
                    "cuda_synchronize_completed": True,
                    "current_allocated_and_reserved_exactly_recovered": True,
                    "disposable_resident_request_group_discarded": True,
                    "registered_attention_backends_removed": True,
                    "attention_implementation_restored": True,
                    "traceback_references_cleared": True,
                },
                "suppression_hook_restoration": hook,
            },
        }
        if mutant_id == "M9" and not clean:
            raw_sha = self._sha("M9-raw-key")
            raw_shape = [1, 8, 4095, 128]
            pre_descriptor = {
                "representation": "q16-paged-key-view",
                "kind": "key",
                "paired_value_remains_q16": True,
            }
            mutated_descriptor = {
                "representation": "raw-torch-tensor",
                "tensor": {
                    "kind": "tensor",
                    "dtype": "torch.bfloat16",
                    "shape": raw_shape,
                    "sha256": raw_sha,
                },
                "paired_value_remains_q16": True,
            }
            restoration = case["outcome"]["injector_target_restoration"]
            restoration.update(
                {
                    "pre_sha256": builder.sha256_json(pre_descriptor),
                    "mutated_sha256": builder.sha256_json(mutated_descriptor),
                    "restored_sha256": builder.sha256_json(pre_descriptor),
                    "pre_descriptor": pre_descriptor,
                    "mutated_descriptor": mutated_descriptor,
                    "restored_descriptor": copy.deepcopy(pre_descriptor),
                }
            )
            mutation = case["outcome"]["mutation_receipt"]
            mutation["pre_descriptor_sha256"] = restoration["pre_sha256"]
            mutation["mutated_descriptor_sha256"] = restoration["mutated_sha256"]
            mutation["raw_tensor_production_receipt"] = {
                "raw_key_type": "torch.Tensor",
                "raw_key_sha256": raw_sha,
                "raw_key_shape": raw_shape,
                "paired_value_type": "Q16KernelPagedTensorView(value)",
                "production_entrypoint": (
                    "qcomem_vllm_paged_kernel."
                    "vllm_triton_q16_paged_attention_forward"
                ),
            }
        return case

    def _rank(self, rank: int) -> dict[str, object]:
        input_receipt = dict(self.prereg["input_binding"])
        input_receipt["preregistration_sha256"] = self.prereg_sha
        cases = [
            self._case(mutant_id, lane)
            for mutant_id in builder.ASSIGNMENT[rank]
            for lane in builder.LANES
        ]
        sidecars = []
        for case in cases:
            payload = (
                self.clean_sidecar_payload
                if case["lane"] == "clean"
                else self.mutant_sidecar_payload
            )
            for stage in builder.EXPECTED_HORIZON_STAGES[case["mutant_id"]]:
                relative = (
                    f"sidecars/rank-{rank}/"
                    f"{case['case_id'].replace(':', '-')}-{stage}.fp32.bin"
                )
                self.sidecar_payloads[relative] = payload
                sidecars.append(
                    {
                        "case_id": case["case_id"],
                        "stage": stage,
                        "relative_path": relative,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "bytes": len(payload),
                        "dtype": "float32",
                        "shape": [1, 248320],
                        "persisted_before_next_stage": True,
                        "atomic_tmp_replace_completed": True,
                        "file_and_directory_fsync_completed": True,
                    }
                )
        for case in cases:
            case_rows = [row for row in sidecars if row["case_id"] == case["case_id"]]
            case["outcome"]["observed_outputs"] = [
                {
                    "stage": row["stage"],
                    "token": 0,
                    "full_logit_sha256": row["sha256"],
                    "sidecar_relative_path": row["relative_path"],
                }
                for row in case_rows
            ]
        return {
            "schema_version": builder.RANK_SCHEMA,
            "rank": rank,
            "assigned_fault_ids": list(builder.ASSIGNMENT[rank]),
            "hardware": {
                "name": "NVIDIA H20-3e",
                "uuid": f"GPU-unit-test-{rank}",
                "compute_capability": [9, 0],
                "memory_mib": 97871,
                "torch_version": builder.PIN_FIXED_STACK["torch"],
                "torch_cuda": builder.PIN_FIXED_STACK["cuda"],
            },
            "input_receipt": input_receipt,
            "discarded_prebaseline_warmup_receipt": {
                "performed": True,
                "discarded": True,
                "completed_before_case_nonces": True,
                "post_warmup_baseline": {
                    "allocated_bytes": 100,
                    "reserved_bytes": 200,
                },
                "gc_collect_completed": True,
                "cuda_empty_cache_completed": True,
                "cuda_synchronize_completed": True,
                "all_gates_on_policy_restored": True,
            },
            "cases": cases,
            "sidecars": sidecars,
            "runner_sha256": self.prereg["source_binding"]["runner_sha256"],
            "gate_policy_sha256": self.prereg["source_binding"][
                "gate_policy_sha256"
            ],
        }

    def _aggregate(self, ranks: list[dict[str, object]] | None = None) -> dict[str, object]:
        return builder.aggregate_rank_payloads(
            prereg=self.prereg,
            prereg_sha=self.prereg_sha,
            rank_payloads=self.rank_payloads if ranks is None else ranks,
            rank_raw_receipts=self.rank_receipts,
            rr2_rows=self.rr2,
            read_sidecar=lambda path: self.sidecar_payloads[path],
        )

    @staticmethod
    def _find_case(ranks: list[dict[str, object]], case_id: str) -> dict[str, object]:
        for rank in ranks:
            for case in rank["cases"]:  # type: ignore[index]
                if case["case_id"] == case_id:
                    return case
        raise AssertionError(case_id)

    def test_valid_scientific_negative_is_accepted(self) -> None:
        value = self._aggregate()
        self.assertTrue(value["scientific_valid"])
        self.assertEqual(value["scientific_outcome"], "negative")
        self.assertEqual(value["operational_invalid_count"], 0)
        self.assertEqual(value["counts"]["cases"], 18)
        self.assertEqual(value["counts"]["clean_cases"], 9)
        self.assertEqual(value["counts"]["target_suppressed_mutant_cases"], 9)
        self.assertEqual(len(value["per_fault_detector_rows"]), 9)

    def test_exact_rev5_external_pin_semantics_are_accepted(self) -> None:
        scope_payload = b"unit-test-scope-v1\n"
        pin = self._valid_external_pin(hashlib.sha256(scope_payload).hexdigest())
        receipt = self._validate_external_pin_fixture(
            pin,
            scope_payload=scope_payload,
        )
        self.assertTrue(receipt["validated"])
        self.assertEqual(
            receipt["checks"],
            {name: True for name in builder.EXTERNAL_PIN_VALIDATION_CHECKS},
        )

    def test_external_pin_tampered_frozen_hash_is_rejected(self) -> None:
        scope_payload = b"unit-test-scope-v1\n"
        pin = self._valid_external_pin(hashlib.sha256(scope_payload).hexdigest())
        pin["frozen_inputs"]["pg19_data_sha256"] = self._sha("tampered-pg19")
        with self.assertRaisesRegex(builder.BuildError, "frozen input drift"):
            self._validate_external_pin_fixture(pin, scope_payload=scope_payload)

    def test_external_pin_scope_mismatch_is_rejected(self) -> None:
        pin = self._valid_external_pin(self._sha("different-scope"))
        with self.assertRaisesRegex(builder.BuildError, "scope raw SHA"):
            self._validate_external_pin_fixture(pin)

    def test_external_pin_old_m9_id_or_missing_m7_is_rejected(self) -> None:
        scope_payload = b"unit-test-scope-v1\n"
        scope_sha = hashlib.sha256(scope_payload).hexdigest()
        for label, mutate in (
            (
                "old-m9-id",
                lambda rows: rows[1].__setitem__(
                    "allowlist_id", "M9_PAIRED_Q16_PAGED_VIEW"
                ),
            ),
            ("missing-m7", lambda rows: rows.pop(0)),
        ):
            with self.subTest(label=label):
                pin = self._valid_external_pin(scope_sha)
                mutate(pin["detector_policy"]["production_assertion_allowlist"])
                with self.assertRaisesRegex(builder.BuildError, "M7/M9 assertion"):
                    self._validate_external_pin_fixture(
                        pin,
                        scope_payload=scope_payload,
                    )

    def test_external_pin_generic_crash_classification_drift_is_rejected(self) -> None:
        scope_payload = b"unit-test-scope-v1\n"
        pin = self._valid_external_pin(hashlib.sha256(scope_payload).hexdigest())
        pin["detector_policy"]["classification_precedence"].insert(
            -1, "generic production nonassertion crash"
        )
        with self.assertRaisesRegex(builder.BuildError, "generic crash/classification"):
            self._validate_external_pin_fixture(pin, scope_payload=scope_payload)

    def test_valid_classified_abort_needs_no_mutant_sidecar(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        case = self._find_case(ranks, "M8:target_suppressed")
        outcome = case["outcome"]
        outcome["completion_status"] = "classified_abort"
        outcome["classification"] = "fault_payload_abort"
        outcome["output_availability"] = {
            "token": "not_evaluated",
            "full_logit": "not_evaluated",
            "sidecar": "not_evaluated",
        }
        outcome["semantics"] = {
            "token_only": self._not_evaluated_semantic(),
            "full_logit": self._not_evaluated_semantic(),
        }
        outcome["observed_outputs"] = []
        outcome["mutation_receipt"]["fault_payload_abort_provenance"] = {
            "exception_type": "AssertionError",
            "exact_message": "matrix M8 sentinel executed",
            "stack_provenance": [
                {
                    "file": "run_qcomem_qwen35_forkaudit_detector_matrix_v2.py",
                    "function": "m8_sentinel",
                }
            ],
        }
        outcome["production"]["assertion"] = {
            "status": "not_evaluated",
            "caught": None,
            "allowlist_id": None,
            "provenance": None,
        }
        outcome["production"]["nonassertion_crash"] = {
            "status": "not_evaluated",
            "caught": None,
        }
        outcome["production"]["fault_payload_abort"] = {
            "status": "evaluated",
            "caught": True,
        }
        ranks[7]["sidecars"] = [
            row
            for row in ranks[7]["sidecars"]
            if row["case_id"] != "M8:target_suppressed"
        ]
        value = self._aggregate(ranks)
        self.assertEqual(value["counts"]["classifications"]["fault_payload_abort"], 1)

    def test_m8_payload_requires_exact_preregistered_provenance(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        case = self._find_case(ranks, "M8:target_suppressed")
        outcome = case["outcome"]
        outcome["completion_status"] = "classified_abort"
        outcome["classification"] = "fault_payload_abort"
        outcome["output_availability"] = {
            "token": "not_evaluated",
            "full_logit": "not_evaluated",
            "sidecar": "not_evaluated",
        }
        outcome["semantics"] = {
            "token_only": self._not_evaluated_semantic(),
            "full_logit": self._not_evaluated_semantic(),
        }
        outcome["observed_outputs"] = []
        outcome["mutation_receipt"]["fault_payload_abort_provenance"] = {
            "exception_type": "AssertionError",
            "exact_message": "wrong sentinel message",
            "stack_provenance": [
                {
                    "file": builder.M8_FAULT_PAYLOAD_PIN["stack_file"],
                    "function": builder.M8_FAULT_PAYLOAD_PIN["stack_function"],
                }
            ],
        }
        outcome["production"]["assertion"] = {
            "status": "not_evaluated",
            "caught": None,
            "allowlist_id": None,
            "provenance": None,
        }
        outcome["production"]["nonassertion_crash"] = {
            "status": "not_evaluated",
            "caught": None,
        }
        outcome["production"]["fault_payload_abort"] = {
            "status": "evaluated",
            "caught": True,
        }
        outcome["measured_non_forkaudit_escape"] = False
        ranks[7]["sidecars"] = [
            row
            for row in ranks[7]["sidecars"]
            if row["case_id"] != "M8:target_suppressed"
        ]
        with self.assertRaisesRegex(builder.BuildError, "payload exact type/message"):
            self._aggregate(ranks)

    def test_valid_classified_abort_retains_partial_sidecar_without_semantic_label(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        case = self._find_case(ranks, "M3:target_suppressed")
        outcome = case["outcome"]
        outcome["completion_status"] = "classified_abort"
        outcome["classification"] = "other_forkaudit_gate"
        outcome["fork_audit"]["other_gate"] = {
            "id": "SOME_OTHER_GATE",
            "status": "evaluated",
            "caught": True,
        }
        outcome["production"] = {
            "assertion": {
                "status": "not_evaluated",
                "caught": None,
                "allowlist_id": None,
                "provenance": None,
            },
            "nonassertion_crash": {"status": "not_evaluated", "caught": None},
            "fault_payload_abort": {"status": "not_evaluated", "caught": None},
        }
        outcome["output_availability"] = {
            "token": "not_evaluated",
            "full_logit": "not_evaluated",
            "sidecar": "evaluated",
        }
        outcome["semantics"] = {
            "token_only": self._not_evaluated_semantic(),
            "full_logit": self._not_evaluated_semantic(),
        }
        outcome["observed_outputs"] = [
            row for row in outcome["observed_outputs"] if row["stage"] == "prefix-r0"
        ]
        ranks[2]["sidecars"] = [
            row
            for row in ranks[2]["sidecars"]
            if row["case_id"] != "M3:target_suppressed"
            or row["stage"] == "prefix-r0"
        ]
        value = self._aggregate(ranks)
        m3_sidecars = [
            row
            for row in value["sidecar_receipts"]
            if row["case_id"] == "M3:target_suppressed"
        ]
        self.assertEqual(len(m3_sidecars), 1)
        m3 = next(
            row
            for row in value["per_fault_detector_rows"]
            if row["mutant_id"] == "M3"
        )
        self.assertEqual(
            m3["r28_target_suppressed"]["token_only"]["status"],
            "not_evaluated",
        )

    def test_missing_discarded_warmup_is_operationally_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        ranks[0].pop("discarded_prebaseline_warmup_receipt")
        with self.assertRaisesRegex(builder.BuildError, "discarded pre-baseline warmup"):
            self._aggregate(ranks)

    def test_duplicate_gpu_uuid_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        ranks[1]["hardware"]["uuid"] = ranks[0]["hardware"]["uuid"]
        with self.assertRaisesRegex(builder.BuildError, "eight distinct H20 UUIDs"):
            self._aggregate(ranks)

    def test_nonexact_h20_name_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        ranks[0]["hardware"]["name"] = "NVIDIA H20"
        with self.assertRaisesRegex(builder.BuildError, "exact H20-3e name"):
            self._aggregate(ranks)

    def test_rank_source_provenance_must_match_preregistration(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        ranks[0]["runner_sha256"] = self._sha("different-runner")
        with self.assertRaisesRegex(builder.BuildError, "runner provenance"):
            self._aggregate(ranks)

    def test_duplicate_fault_or_missing_lane_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        ranks[1]["cases"][1] = copy.deepcopy(ranks[1]["cases"][0])
        with self.assertRaisesRegex(builder.BuildError, "case order|duplicate case"):
            self._aggregate(ranks)

    def test_clean_without_sidecar_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        ranks[0]["sidecars"] = [
            row for row in ranks[0]["sidecars"] if row["case_id"] != "M1:clean"
        ]
        with self.assertRaisesRegex(builder.BuildError, "output/sidecar count|missing nonzero sidecar"):
            self._aggregate(ranks)

    def test_sidecar_digest_mismatch_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        ranks[0]["sidecars"][0]["sha256"] = self._sha("wrong")
        with self.assertRaisesRegex(builder.BuildError, "sidecar 0 SHA"):
            self._aggregate(ranks)

    def test_runner_semantic_claim_cannot_override_identical_sidecars(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        clean_sha = hashlib.sha256(self.clean_sidecar_payload).hexdigest()
        for row in ranks[0]["sidecars"]:
            if row["case_id"] == "M1:target_suppressed":
                row["sha256"] = clean_sha
                self.sidecar_payloads[row["relative_path"]] = self.clean_sidecar_payload
        mutant = self._find_case(ranks, "M1:target_suppressed")
        for row in mutant["outcome"]["observed_outputs"]:
            row["full_logit_sha256"] = clean_sha
        with self.assertRaisesRegex(builder.BuildError, "logit SHA recompute"):
            self._aggregate(ranks)

    def test_partial_sidecars_must_be_an_ordered_horizon_prefix(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        case = self._find_case(ranks, "M3:target_suppressed")
        outcome = case["outcome"]
        outcome["completion_status"] = "classified_abort"
        outcome["classification"] = "other_forkaudit_gate"
        outcome["fork_audit"]["other_gate"] = {
            "id": "SOME_OTHER_GATE",
            "status": "evaluated",
            "caught": True,
        }
        outcome["production"] = {
            "assertion": {
                "status": "not_evaluated",
                "caught": None,
                "allowlist_id": None,
                "provenance": None,
            },
            "nonassertion_crash": {"status": "not_evaluated", "caught": None},
            "fault_payload_abort": {"status": "not_evaluated", "caught": None},
        }
        outcome["output_availability"] = {
            "token": "not_evaluated",
            "full_logit": "not_evaluated",
            "sidecar": "evaluated",
        }
        outcome["semantics"] = {
            "token_only": self._not_evaluated_semantic(),
            "full_logit": self._not_evaluated_semantic(),
        }
        outcome["observed_outputs"] = [
            row for row in outcome["observed_outputs"] if row["stage"] == "prefix-r1"
        ]
        ranks[2]["sidecars"] = [
            row
            for row in ranks[2]["sidecars"]
            if row["case_id"] != "M3:target_suppressed"
            or row["stage"] == "prefix-r1"
        ]
        with self.assertRaisesRegex(builder.BuildError, "ordered horizon prefix"):
            self._aggregate(ranks)

    def test_wrong_suppressed_gate_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        case = self._find_case(ranks, "M1:target_suppressed")
        event = case["outcome"]["suppression_hook_restoration"]["events"][0]
        event["gate_id"] = builder.EXPECTED_GATES["M2"]
        event_copy = dict(event)
        event_copy.pop("event_sha256")
        event["event_sha256"] = builder.sha256_json(event_copy)
        case["outcome"]["suppression_hook_restoration"]["suppressed_gate_ids"] = [
            builder.EXPECTED_GATES["M2"]
        ]
        case["outcome"]["fork_audit"]["target_suppression_events"] = [
            copy.deepcopy(event)
        ]
        with self.assertRaisesRegex(builder.BuildError, "wrong suppressed gate"):
            self._aggregate(ranks)

    def test_not_evaluated_cannot_be_converted_to_not_caught(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        cell = self._find_case(ranks, "M2:target_suppressed")["outcome"]["semantics"]["token_only"]
        cell.update(
            {
                "status": "not_evaluated",
                "caught": False,
                "exact_sha": None,
                "argmax_equal": None,
                "max_abs": None,
                "relative_l2": None,
            }
        )
        with self.assertRaisesRegex(builder.BuildError, "caught must be null"):
            self._aggregate(ranks)

    def test_injector_restoration_mismatch_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        receipt = self._find_case(ranks, "M3:target_suppressed")["outcome"][
            "injector_target_restoration"
        ]
        receipt["restored_sha256"] = self._sha("not-pre")
        with self.assertRaisesRegex(builder.BuildError, "exact restoration"):
            self._aggregate(ranks)

    def test_allocator_cleanup_mismatch_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        receipt = self._find_case(ranks, "M4:target_suppressed")["outcome"][
            "case_discard_allocator_recovery"
        ]
        receipt["after_cleanup"]["allocated_bytes"] += 1
        with self.assertRaisesRegex(builder.BuildError, "allocator baseline recovery"):
            self._aggregate(ranks)

    def test_case_must_start_at_post_warmup_allocator_baseline(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        receipt = self._find_case(ranks, "M4:clean")["outcome"][
            "case_discard_allocator_recovery"
        ]
        receipt["before_cell"]["allocated_bytes"] += 1
        with self.assertRaisesRegex(builder.BuildError, "did not start at allocator baseline"):
            self._aggregate(ranks)

    def test_teacher_forced_token_must_match_across_lanes(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        receipt = self._find_case(ranks, "M5:target_suppressed")["outcome"][
            "mutation_receipt"
        ]["teacher_forcing"]
        receipt["token_id"] += 1
        with self.assertRaisesRegex(builder.BuildError, "teacher-forced token differs"):
            self._aggregate(ranks)

    def test_teacher_source_coordinate_and_independence_are_exact(self) -> None:
        for field, value, message in (
            ("source_coordinate", "frozen_query_bank[rank][0][30]", "source coordinate"),
            ("independent_of_path_argmax", False, "teacher independence"),
        ):
            with self.subTest(field=field):
                ranks = copy.deepcopy(self.rank_payloads)
                teacher = self._find_case(ranks, "M3:target_suppressed")["outcome"][
                    "mutation_receipt"
                ]["teacher_forcing"]
                teacher[field] = value
                with self.assertRaisesRegex(builder.BuildError, message):
                    self._aggregate(ranks)

    def test_gate_predicate_source_and_callsite_are_exact(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        case = self._find_case(ranks, "M4:target_suppressed")
        event = case["outcome"]["suppression_hook_restoration"]["events"][0]
        event["predicate_source"] = "unregistered_generic_wrapper"
        event_copy = dict(event)
        event_copy.pop("event_sha256")
        event["event_sha256"] = builder.sha256_json(event_copy)
        case["outcome"]["fork_audit"]["target_suppression_events"] = [
            copy.deepcopy(event)
        ]
        with self.assertRaisesRegex(builder.BuildError, "predicate binding"):
            self._aggregate(ranks)

    def test_m9_raw_tensor_paired_view_receipt_is_exact(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        receipt = self._find_case(ranks, "M9:target_suppressed")["outcome"][
            "mutation_receipt"
        ]["raw_tensor_production_receipt"]
        receipt["paired_value_type"] = "torch.Tensor"
        with self.assertRaisesRegex(builder.BuildError, "paired value type"):
            self._aggregate(ranks)

    def test_operational_invalid_case_is_rejected_not_reported_negative(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        outcome = self._find_case(ranks, "M5:target_suppressed")["outcome"]
        outcome["completion_status"] = "operational_invalid"
        outcome["classification"] = "operational_invalid"
        with self.assertRaisesRegex(builder.BuildError, "operational invalidity"):
            self._aggregate(ranks)

    def test_unallowlisted_production_assertion_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        case = self._find_case(ranks, "M9:target_suppressed")
        outcome = case["outcome"]
        outcome["completion_status"] = "classified_abort"
        outcome["classification"] = "production_assertion"
        outcome["measured_non_forkaudit_escape"] = False
        outcome["production"]["assertion"] = {
            "status": "evaluated",
            "caught": True,
            "allowlist_id": "PA-UNKNOWN",
            "provenance": {},
        }
        with self.assertRaisesRegex(builder.BuildError, "allowlist"):
            self._aggregate(ranks)

    def test_exact_m7_production_assertion_allowlist_is_accepted(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        case = self._find_case(ranks, "M7:target_suppressed")
        outcome = case["outcome"]
        outcome["completion_status"] = "classified_abort"
        outcome["classification"] = "production_assertion"
        outcome["measured_non_forkaudit_escape"] = False
        outcome["production"]["assertion"] = {
            "status": "evaluated",
            "caught": True,
            "allowlist_id": builder.M7_ASSERTION_ALLOWLIST_ID,
            "provenance": {
                "exception_module": builder.M7_ASSERTION_ALLOWLIST["exception_module"],
                "exception_type": builder.M7_ASSERTION_ALLOWLIST["exception_type"],
                "exact_message": builder.M7_ASSERTION_ALLOWLIST["exact_message"],
                "stack_provenance": [
                    {
                        "file": builder.M7_ASSERTION_ALLOWLIST["stack_file"],
                        "function": builder.M7_ASSERTION_ALLOWLIST["stack_function"],
                    }
                ],
            },
        }
        outcome["production"]["nonassertion_crash"] = {
            "status": "not_evaluated",
            "caught": None,
        }
        outcome["production"]["fault_payload_abort"] = {
            "status": "not_evaluated",
            "caught": None,
        }
        outcome["output_availability"] = {
            "token": "not_evaluated",
            "full_logit": "not_evaluated",
            "sidecar": "not_evaluated",
        }
        outcome["semantics"] = {
            "token_only": self._not_evaluated_semantic(),
            "full_logit": self._not_evaluated_semantic(),
        }
        outcome["observed_outputs"] = []
        ranks[6]["sidecars"] = [
            row
            for row in ranks[6]["sidecars"]
            if row["case_id"] != "M7:target_suppressed"
        ]
        value = self._aggregate(ranks)
        self.assertEqual(value["counts"]["classifications"]["production_assertion"], 1)

    def test_exact_m9_production_assertion_allowlist_is_accepted(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        case = self._find_case(ranks, "M9:target_suppressed")
        outcome = case["outcome"]
        outcome["completion_status"] = "classified_abort"
        outcome["classification"] = "production_assertion"
        outcome["measured_non_forkaudit_escape"] = False
        outcome["production"]["assertion"] = {
            "status": "evaluated",
            "caught": True,
            "allowlist_id": builder.M9_ASSERTION_ALLOWLIST_ID,
            "provenance": {
                "exception_module": builder.M9_ASSERTION_ALLOWLIST["exception_module"],
                "exception_type": builder.M9_ASSERTION_ALLOWLIST["exception_type"],
                "exact_message": builder.M9_ASSERTION_ALLOWLIST["exact_message"],
                "stack_provenance": [
                    {
                        "file": builder.M9_ASSERTION_ALLOWLIST["stack_file"],
                        "function": builder.M9_ASSERTION_ALLOWLIST["stack_function"],
                    }
                ],
            },
        }
        outcome["production"]["nonassertion_crash"] = {
            "status": "not_evaluated",
            "caught": None,
        }
        outcome["production"]["fault_payload_abort"] = {
            "status": "not_evaluated",
            "caught": None,
        }
        outcome["output_availability"] = {
            "token": "not_evaluated",
            "full_logit": "not_evaluated",
            "sidecar": "not_evaluated",
        }
        outcome["semantics"] = {
            "token_only": self._not_evaluated_semantic(),
            "full_logit": self._not_evaluated_semantic(),
        }
        outcome["observed_outputs"] = []
        ranks[0]["sidecars"] = [
            row
            for row in ranks[0]["sidecars"]
            if row["case_id"] != "M9:target_suppressed"
        ]
        value = self._aggregate(ranks)
        self.assertEqual(value["counts"]["classifications"]["production_assertion"], 1)

    def test_generic_nonassertion_crash_classification_is_rejected(self) -> None:
        ranks = copy.deepcopy(self.rank_payloads)
        outcome = self._find_case(ranks, "M2:target_suppressed")["outcome"]
        outcome["completion_status"] = "classified_abort"
        outcome["classification"] = "production_nonassertion_crash"
        with self.assertRaisesRegex(builder.BuildError, "classification"):
            self._aggregate(ranks)

    def test_replay_requires_byte_identical_canonical_summary(self) -> None:
        summary = self._aggregate()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary_path = root / "summary.json"
            output_path = root / "replayed.json"
            builder.write_json(summary_path, summary)
            args = argparse.Namespace(
                summary=summary_path,
                output=output_path,
                preregistration=root / "prereg.json",
                expected_preregistration_sha256=self.prereg_sha,
                rank_root=root / "raw",
                original_receipt_manifest=root / "rr2.json",
                original_rr2_root=root / "rr2",
                expected_runner_sha256=self._sha("runner"),
                runner=root / "runner.py",
                test_file=root / "test.py",
                launcher=root / "launcher.sh",
                gate_policy=root / "policy.py",
                qs_config=root / "qs.yaml",
                scope_supersession=root / "scope.json",
                external_pin_payload=root / "pin.json",
            )

            def fake_aggregate(namespace: argparse.Namespace) -> dict[str, object]:
                builder.write_json(namespace.output, summary)
                return summary

            with mock.patch.object(
                replay_module.builder, "aggregate_from_paths", side_effect=fake_aggregate
            ):
                receipt = replay_module.replay(args)
            self.assertTrue(receipt["byte_identical"])
            self.assertEqual(output_path.read_bytes(), summary_path.read_bytes())

    def test_builder_aggregate_parser_registers_every_source_path(self) -> None:
        args = builder.parser().parse_args(
            [
                "--stage",
                "aggregate",
                "--output",
                "summary.json",
                "--preregistration",
                "prereg.json",
                "--expected-preregistration-sha256",
                self.prereg_sha,
                "--original-receipt-manifest",
                "rr2-manifest.json",
                "--original-rr2-root",
                "rr2-root",
                "--rank-root",
                "raw",
                "--expected-runner-sha256",
                self._sha("runner"),
                "--runner",
                "runner.py",
                "--replay",
                "replay.py",
                "--test-file",
                "test.py",
                "--launcher",
                "launcher.sh",
                "--gate-policy",
                "policy.py",
                "--qs-config",
                "qs.yaml",
                "--scope-supersession",
                "scope.json",
                "--external-pin-payload",
                "external-pin.json",
            ]
        )
        expected_paths = {
            "runner": "runner.py",
            "replay": "replay.py",
            "test_file": "test.py",
            "launcher": "launcher.sh",
            "gate_policy": "policy.py",
            "qs_config": "qs.yaml",
            "scope_supersession": "scope.json",
            "external_pin_payload": "external-pin.json",
        }
        for field, expected in expected_paths.items():
            self.assertEqual(getattr(args, field), Path(expected), field)
        self.assertIsNone(args.external_pin_payload_sha256)

    def test_launcher_has_terminal_integrity_and_negative_acceptance(self) -> None:
        launcher = Path(__file__).with_name(
            "launch_qcomem_qwen35_forkaudit_detector_matrix_v2_8gpu.sh"
        )
        source = launcher.read_text(encoding="utf-8")
        self.assertLess(
            source.index("01_preregistered_before_candidate_outputs"),
            source.index("for rank in $(seq 0 7)"),
        )
        self.assertIn("eight_distinct_h20s_verified", source)
        self.assertIn("discarded_prebaseline_warmup", builder.__file__ and Path(builder.__file__).read_text())
        self.assertGreaterEqual(source.count("sha256sum -c receipts/raw-artifacts.sha256"), 2)
        self.assertIn("cmp \"$RUN_DIR/detector-matrix-v2-summary.json\"", source)
        self.assertIn("COMPLETED_VALID_SCIENTIFIC_OUTCOME", source)
        self.assertIn("(\"positive\", \"negative\", \"mixed\")", source)
        self.assertNotIn("hypothesis_passed", source)
        self.assertNotIn("EXPECTED_QS_CONFIG_SHA256", source)
        self.assertIn('cd "$CODE_DIR"\n  sha256sum -c "$CODE_LEDGER_FILE"', source)
        self.assertIn(
            'cd "$IMPORTED_RR2_CODE_DIR"\n  sha256sum -c "$IMPORTED_RR2_CODE_LEDGER_FILE"',
            source,
        )
        self.assertIn("preflight-unit-tests.log", source)
        self.assertIn("01_preflight_unit_tests_passed", source)
        self.assertIn('assert name == "NVIDIA H20-3e"', source)
        preregister_start = source.index(
            '"$PYTHON" -B "$BUILDER" \\\n  --stage preregister'
        )
        preregister_end = source.index("PREREG_SHA=", preregister_start)
        preregister_options = re.findall(
            r"^\s+(--[a-z0-9-]+)\b",
            source[preregister_start:preregister_end],
            re.M,
        )
        builder_options = set(builder.parser()._option_string_actions)
        self.assertFalse(
            set(preregister_options) - builder_options,
            f"launcher preregister options missing from builder parser: "
            f"{sorted(set(preregister_options) - builder_options)}",
        )
        self.assertIn("--external-pin-payload", preregister_options)
        self.assertIn("--external-pin-payload-sha256", preregister_options)
        aggregate_start = source.index(
            '"$PYTHON" -B "$BUILDER" \\\n  --stage aggregate'
        )
        aggregate_end = source.index('> "$LOGS/aggregate.log"', aggregate_start)
        aggregate_options = re.findall(
            r"^\s+(--[a-z0-9-]+)\b",
            source[aggregate_start:aggregate_end],
            re.M,
        )
        self.assertFalse(
            set(aggregate_options) - builder_options,
            f"launcher aggregate options missing from builder parser: "
            f"{sorted(set(aggregate_options) - builder_options)}",
        )
        self.assertIn("--external-pin-payload", aggregate_options)
        rank_start = source.index('"$PYTHON" -B "$RUNNER"')
        rank_end = source.index('> "$LOGS/rank-$rank.log"', rank_start)
        rank_options = re.findall(r"^\s+(--[a-z0-9-]+)\b", source[rank_start:rank_end], re.M)
        self.assertEqual(
            len(rank_options),
            len(set(rank_options)),
            f"duplicate runner options: {rank_options}",
        )
        subprocess.run(["bash", "-n", str(launcher)], check=True)


if __name__ == "__main__":
    unittest.main()
