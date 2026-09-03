from __future__ import annotations

import copy
import hashlib
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np

import r29_aggregate_heldout_faults as aggregator
import r29_execute_heldout_faults as executor
import r29_heldout_fault_suite as fault_suite


HERE = Path(__file__).resolve().parent
SUITE_PATH = HERE.parent / "paper_autonomous_multifork_iteration" / "evidence" / "r29_heldout_faults" / "preregistration" / "heldout-fault-suite.json"


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cell(status: str, caught: bool | None, reason: str = "synthetic") -> dict[str, Any]:
    return {"status": status, "caught": caught, "reason": reason, "evidence": None}


def _production_catch() -> dict[str, Any]:
    return {
        "status": "evaluated",
        "caught": True,
        "reason": "exact_production_assertion",
        "evidence": {
            "classification": "exact_production_assertion",
            "production_assertion_allowlist_id": "PA-Q16-PAIRED-VIEWS-v1",
            "exception": {
                "module": "qcomem_vllm_paged_kernel",
                "type": "QComemPagedKernelError",
                "message": "fused backend requires paired Q16 paged views",
                "gate_id": None,
                "stack": [
                    {
                        "filename": "qcomem_vllm_paged_kernel.py",
                        "line": 1,
                        "function": "_paired_sequence",
                    }
                ],
            },
        },
    }


class HeldOutExecutorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "run" / "raw"
        self.raw.mkdir(parents=True)
        self.suite_raw = SUITE_PATH.read_bytes()
        self.suite = json.loads(self.suite_raw)
        self.suite_raw_sha = _sha(self.suite_raw)
        self.suite_canonical_sha = fault_suite.validate_frozen_suite(self.suite)["suite_sha256"]
        self.execution_input = self._execution_input()
        self.execution_input_sha = _sha(executor.canonical_bytes(self.execution_input) + b"\n")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _execution_input(self) -> dict[str, Any]:
        h = "0" * 64
        return {
            "schema_version": executor.INPUT_SCHEMA,
            "status": "frozen_before_candidate_outputs",
            "created_at_utc": "2026-08-25T00:00:00Z",
            "run_id": "R29-HELDOUT-TEST",
            "suite_binding": {
                "raw_sha256": self.suite_raw_sha,
                "canonical_sha256": self.suite_canonical_sha,
                "fault_module_sha256": h,
                "launcher_sha256": h,
                "author_test_sha256": h,
            },
            "fixed_protocol": {
                "rank_assignment": {"0": "H01", "1": "H02", "2": "H03"},
                "lane_order": list(executor.LANE_ORDER),
                "document_tokens": 4095,
                "page_size": 128,
                "resident_count": 2,
                "input_token_coordinate": "frozen_query_bank[rank][0][31]",
                "advertised_horizon_tokens": 1,
                "kv_policy": "vllm-q16-shared-document-reuse",
                "gdn_policy": "borrow-immutable-base-functional-rebind",
                "sidecar_shape": [1, 248320],
                "sidecar_dtype": "float32",
                "sidecar_nbytes": 993280,
            },
            "code": {
                "code_dir": str(self.root / "code"),
                "executor_path": str(self.root / "code" / "r29_execute_heldout_faults.py"),
                "executor_sha256": h,
                "aggregator_path": str(self.root / "code" / "r29_aggregate_heldout_faults.py"),
                "aggregator_sha256": h,
                "fault_module_path": str(self.root / "code" / "r29_heldout_fault_suite.py"),
                "imported_rr2_code_dir": str(self.root / "rr2"),
                "imported_rr2_code_ledger_path": str(self.root / "code.sha256"),
                "imported_rr2_code_ledger_raw_sha256": h,
            },
            "model": {
                "model_dir": str(self.root / "model"),
                "model_id": "Qwen/Qwen3.5-35B-A3B",
                "revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
                "dtype": "bfloat16",
                "weight_ledger_path": str(self.root / "weights.sha256"),
                "weight_ledger_raw_sha256": h,
                "artifact_ledger_path": str(self.root / "artifacts.sha256"),
                "artifact_ledger_raw_sha256": h,
            },
            "data": {
                "split": "PG19 train only",
                "pg19_data_path": str(self.root / "pg19.jsonl"),
                "pg19_data_raw_sha256": h,
                "pg19_manifest_path": str(self.root / "pg19.manifest.json"),
                "pg19_manifest_raw_sha256": h,
                "pg19_windows_canonical_sha256": h,
                "frozen_query_banks_path": str(self.root / "banks.json"),
                "frozen_query_banks_raw_sha256": h,
            },
            "environment": {
                "env_dir": str(self.root / "env"),
                "python": "3.11",
                "torch": "2.11.0+cu129",
                "torch_cuda": "12.9",
                "transformers": "5.14.1",
                "vllm": "0.26.0",
                "gpu_name": "NVIDIA H20-3e",
                "compute_capability": [9, 0],
            },
            "output": {"run_root": str(self.root / "run"), "raw_root": str(self.raw)},
            "claim_boundary": {
                "historical_pattern_inspired_only": True,
                "naturally_occurring_claimed": False,
                "upstream_implementation_evaluated": False,
                "detection_rate_reported": False,
            },
        }

    def _sidecar(self, rank: int, lane: str, values: np.ndarray) -> dict[str, Any]:
        relative = Path("sidecars") / f"rank-{rank}" / f"{lane}.bin"
        path = self.raw / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = values.astype(np.float32).tobytes(order="C")
        self.assertEqual(len(payload), executor.SIDE_CAR_NBYTES)
        path.write_bytes(payload)
        return {
            "path": relative.as_posix(),
            "sha256": _sha(payload),
            "dtype": "float32",
            "shape": [1, 248320],
            "nbytes": len(payload),
            "finite": True,
            "contains_absolute_pointer": False,
        }

    def _state_intervention(self, fault_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
        def descriptor(values: list[int]) -> dict[str, Any]:
            return {
                "schema_version": "forkaudit-r29-pointer-free-route-target-v1",
                "request_index": 0,
                "layer_index": 0,
                "field": "synthetic",
                "shape": [1, 3],
                "stride": [3, 1],
                "dtype": "torch.int32",
                "device": "cuda:0",
                "values": values,
                "values_sha256": executor.sha256_json(values),
                "contains_absolute_pointer": False,
            }

        pre_descriptor = descriptor([1, 2, 9])
        mutated_descriptor = descriptor([2, 1, 9])
        pre_restore_descriptor = descriptor([2, 1, 10])
        restored_descriptor = descriptor([1, 2, 10])
        pre = executor.sha256_json(pre_descriptor)
        mutated = executor.sha256_json(mutated_descriptor)
        applied = {
            "schema_version": "forkaudit-r29-heldout-mutation-binding-v1",
            "fault_id": fault_id,
            "target_kind": "synthetic target",
            "pre_sha256": pre,
            "mutated_sha256": mutated,
            "pre_descriptor": pre_descriptor,
            "mutated_descriptor": mutated_descriptor,
            "mutation_observed": True,
            "contains_absolute_pointer": False,
        }
        restored = {
            "schema_version": "forkaudit-r29-heldout-restoration-v2",
            "fault_id": fault_id,
            "target_kind": "synthetic target",
            "applied_pre_sha256": pre,
            "applied_mutated_sha256": mutated,
            "mutation_coordinate_indices": [0, 1],
            "pre_restore_descriptor": pre_restore_descriptor,
            "restored_descriptor": restored_descriptor,
            "pre_restore_sha256": executor.sha256_json(pre_restore_descriptor),
            "restored_sha256": executor.sha256_json(restored_descriptor),
            "target_pre_values_sha256": executor.sha256_json([1, 2]),
            "target_mutated_values_sha256": executor.sha256_json([2, 1]),
            "target_pre_restore_values_sha256": executor.sha256_json([2, 1]),
            "target_restored_values_sha256": executor.sha256_json([1, 2]),
            "target_remained_mutated_through_horizon": True,
            "target_restored_exact": True,
            "non_target_preserved_across_undo": True,
            "restoration_observed": True,
            "contains_absolute_pointer": False,
        }
        return {"kind": "reversible_state_mutation", "fault_active": True, "applied_receipt": applied}, restored

    def _case(self, rank: int, fault_id: str, lane: str, clean_values: np.ndarray, fault_values: np.ndarray) -> dict[str, Any]:
        baseline = {"allocated_bytes": 100, "reserved_bytes": 200}
        values = clean_values if lane == "clean" else fault_values
        action = executor._default_action_sequence(rank)
        restoration = None
        if lane == "clean":
            intervention = {"kind": "none", "fault_active": False, "mutation_observed": False}
        elif fault_id in fault_suite.STATE_MUTATION_FAULT_IDS:
            intervention, restoration = self._state_intervention(fault_id)
        else:
            action = fault_suite.h02_action_sequence(request_index=0)
            intervention = {
                "kind": "immutable_action_sequence",
                "fault_active": True,
                "action_sequence": action,
                "action_sequence_sha256": executor.sha256_json(action),
                "fresh_case_disposal_required": True,
            }
        receipts = [
            {"receipt_id": receipt_id, "status": "passed", "payload": {}}
            for receipt_id in aggregator.RECEIPT_ORDER
        ]
        rejection = None
        forkaudit = _cell("evaluated", False)
        if lane == "fault_conventional":
            receipts = []
            forkaudit = _cell("not_evaluated", None)
        elif lane == "fault_forkaudit" and fault_id == "H02":
            receipts = receipts[:3]
            rejection = {
                "authenticated": True,
                "receipt_id": aggregator.RECEIPT_ORDER[3],
                "predicate_id": "synthetic-generic-action-predicate",
                "exception": {
                    "module": "r29_execute_heldout_faults",
                    "type": "ReceiptPredicateRejection",
                    "message": "synthetic",
                    "gate_id": "synthetic-generic-action-predicate",
                    "stack": [
                        {
                            "filename": "r29_execute_heldout_faults.py",
                            "line": 1,
                            "function": "_schedule_receipt",
                        }
                    ],
                },
            }
            forkaudit = _cell("evaluated", True)
        sidecar = self._sidecar(rank, lane, values)
        return {
            "lane": lane,
            "case_nonce": hashlib.sha256(f"{rank}-{lane}".encode()).hexdigest()[:32],
            "fresh_case": True,
            "state_reused_from_prior_lane": False,
            "allocator_before": baseline,
            "allocator_baseline": baseline,
            "action_sequence": action,
            "intervention": intervention,
            "model_invocations": [],
            "kernel_ledger": {},
            "completion_status": "completed",
            "semantic_horizon_reached": True,
            "advertised_horizon_tokens": 1,
            "greedy_token_id": int(np.argmax(values, axis=-1)[0]),
            "full_logits": sidecar,
            "production_assertion": _cell("evaluated", False),
            "forkaudit": forkaudit,
            "completed_receipts": receipts,
            "first_authenticated_rejection": rejection,
            "restoration_receipt": restoration,
            "operational_invalid": None,
            "cleanup": {
                "fresh_case_disposed": True,
                "registered_backend_restored": True,
                "gc_collect_completed": True,
                "cuda_empty_cache_completed": True,
                "cuda_synchronize_completed": True,
                "allocator_after": baseline,
                "allocator_baseline_exact": True,
                "cleanup_passed": True,
                "cleanup_error": None,
            },
        }

    def _ranks(self) -> list[dict[str, Any]]:
        ranks = []
        for rank, fault_id in enumerate(fault_suite.FAULT_IDS):
            clean_values = np.zeros((1, 248320), dtype=np.float32)
            clean_values[0, rank + 10] = 1.0
            fault_values = clean_values.copy()
            lanes = [self._case(rank, fault_id, lane, clean_values, fault_values) for lane in executor.LANE_ORDER]
            for lane in lanes[1:]:
                lane["semantic_comparisons"] = executor.compare_semantic_outputs(lanes[0], lane, raw_root=self.raw)
            lanes[0]["semantic_comparisons"] = {
                "greedy_token": _cell("not_evaluated", None),
                "full_fp32_logits": _cell("not_evaluated", None),
            }
            ranks.append(
                {
                    "schema_version": aggregator.RANK_SCHEMA,
                    "status": "completed_rank_artifact",
                    "run_id": self.execution_input["run_id"],
                    "rank": rank,
                    "fault_id": fault_id,
                    "suite_raw_sha256": self.suite_raw_sha,
                    "suite_canonical_sha256": self.suite_canonical_sha,
                    "execution_input_raw_sha256": self.execution_input_sha,
                    "source_bindings": {
                        "executor_sha256": "0" * 64,
                        "aggregator_sha256": "0" * 64,
                        "fault_module_sha256": "0" * 64,
                        "launcher_sha256": "0" * 64,
                        "imported_rr2_code_ledger_raw_sha256": "0" * 64,
                    },
                    "hardware": {
                        "uuid": f"GPU-test-{rank}",
                        "name": "NVIDIA H20-3e",
                        "memory_mib": 1000,
                        "compute_capability": [9, 0],
                    },
                    "input_receipt": {
                        "rank": rank,
                        "model_revision": self.execution_input["model"]["revision"],
                        "imported_rr2_code": {"raw_sha256": "0" * 64},
                    },
                    "discarded_warmup": {
                        "performed": True,
                        "discarded": True,
                        "post_warmup_allocator_baseline": {"allocated_bytes": 100, "reserved_bytes": 200},
                    },
                    "lanes": lanes,
                    "operational_invalid_count": 0,
                    "detection_rate_reported": False,
                    "naturally_occurring_claimed": False,
                    "upstream_implementation_evaluated": False,
                }
            )
        return ranks

    def _aggregate(self, ranks: list[dict[str, Any]]) -> dict[str, Any]:
        return aggregator.aggregate_documents(
            suite=self.suite,
            execution_input=self.execution_input,
            execution_input_raw_sha256=self.execution_input_sha,
            suite_raw_sha256=self.suite_raw_sha,
            suite_canonical_sha256=self.suite_canonical_sha,
            ranks=ranks,
            raw_root=self.raw,
        )

    def test_execution_input_is_strict_and_claim_bounded(self) -> None:
        validated = executor.validate_execution_input(self.execution_input)
        self.assertFalse(validated["claim_boundary"]["detection_rate_reported"])
        tampered = copy.deepcopy(self.execution_input)
        tampered["fixed_protocol"]["lane_order"] = ["clean", "fault_forkaudit", "fault_conventional"]
        with self.assertRaises(executor.HeldOutExecutionError):
            executor.validate_execution_input(tampered)

    def test_symlink_venv_identity_accepts_external_base_interpreter(self) -> None:
        base_prefix = self.root / "base"
        base_python = base_prefix / "bin" / "python3.11"
        base_python.parent.mkdir(parents=True)
        base_python.write_bytes(b"synthetic interpreter target")
        env_dir = self.root / "frozen-env"
        env_bin = env_dir / "bin"
        env_bin.mkdir(parents=True)
        (env_bin / "python3").symlink_to(base_python)
        (env_bin / "python").symlink_to("python3")

        receipt = executor.validate_python_environment_identity(
            env_dir,
            executable_value=env_bin / "python",
            prefix_value=env_dir,
            base_prefix_value=base_prefix,
        )
        self.assertTrue(receipt["sys_prefix_exact"])
        self.assertTrue(receipt["lexical_invocation_exact"])
        self.assertTrue(receipt["resolved_interpreter_target_exact"])
        self.assertFalse(receipt["resolved_interpreter_required_below_env"])
        self.assertEqual(receipt["sys_executable_realpath"], str(base_python.resolve()))

        with self.assertRaises(executor.HeldOutExecutionError):
            executor.validate_python_environment_identity(
                env_dir,
                executable_value=env_bin / "python",
                prefix_value=base_prefix,
                base_prefix_value=base_prefix,
            )

    def test_completed_request_append_scope_accepts_unadvanced_live_peer(self) -> None:
        layer_index = 7
        requests = tuple(
            SimpleNamespace(
                layers={
                    layer_index: SimpleNamespace(
                        sequence=SimpleNamespace(appended_tokens=appended)
                    )
                }
            )
            for appended in (1, 0)
        )
        group = SimpleNamespace(requests=requests)
        plan = SimpleNamespace(full_attention_layer_indices=(layer_index,))
        persistent = SimpleNamespace()
        imported_receipt = {
            "passed": True,
            "require_appended_tail_cow": False,
        }
        with mock.patch.object(
            executor.resident,
            "validate_runtime_kv_ownership",
            return_value=imported_receipt,
        ) as validate:
            receipt = executor.validate_completed_request_kv_horizon(
                persistent,
                group,
                plan,
                completed_request_indices=(0,),
            )
        validate.assert_called_once_with(
            persistent,
            group,
            plan,
            require_appended_tail_cow=False,
        )
        self.assertEqual(receipt["completed_request_indices"], [0])
        self.assertEqual(
            [row["appended_tokens"] for row in receipt["appended_rows"]],
            [1, 0],
        )
        self.assertTrue(receipt["completed_requests_appended"])

        requests[0].layers[layer_index].sequence.appended_tokens = 0
        with mock.patch.object(
            executor.resident,
            "validate_runtime_kv_ownership",
            return_value=imported_receipt,
        ):
            with self.assertRaises(executor.ReceiptPredicateRejection):
                executor.validate_completed_request_kv_horizon(
                    persistent,
                    group,
                    plan,
                    completed_request_indices=(0,),
                )

    def test_delta_scope_restoration_preserves_legal_non_target_transition(self) -> None:
        def descriptor(values: list[int]) -> dict[str, Any]:
            return {
                "schema_version": "forkaudit-r29-pointer-free-route-target-v1",
                "request_index": 0,
                "layer_index": 7,
                "field": "synthetic_block_table",
                "shape": [1, 4],
                "stride": [4, 1],
                "dtype": "torch.int32",
                "device": "cuda:0",
                "values": list(values),
                "values_sha256": executor.sha256_json(values),
                "contains_absolute_pointer": False,
            }

        state = [0, 2, 1, 99]

        def capture() -> dict[str, Any]:
            return descriptor(state)

        def undo() -> None:
            state[1], state[2] = 1, 2

        handle = SimpleNamespace(
            fault_id="synthetic",
            target_kind="synthetic route pair",
            pre_descriptor=descriptor([0, 1, 2, -1]),
            mutated_descriptor=descriptor([0, 2, 1, -1]),
            capture=capture,
            undo=undo,
        )
        receipt = executor.restore_mutation_delta_scope(handle)
        self.assertEqual(state, [0, 1, 2, 99])
        self.assertEqual(receipt["mutation_coordinate_indices"], [1, 2])
        self.assertTrue(receipt["target_restored_exact"])
        self.assertTrue(receipt["non_target_preserved_across_undo"])
        applied = {
            "fault_id": handle.fault_id,
            "target_kind": handle.target_kind,
            "pre_sha256": executor.sha256_json(handle.pre_descriptor),
            "mutated_sha256": executor.sha256_json(handle.mutated_descriptor),
            "pre_descriptor": handle.pre_descriptor,
            "mutated_descriptor": handle.mutated_descriptor,
        }
        aggregator._validate_delta_scoped_restoration(
            applied,
            receipt,
            label="synthetic",
        )

    def test_aggregator_rejects_non_target_restoration_tamper(self) -> None:
        intervention, restoration = self._state_intervention("synthetic")
        restoration["restored_descriptor"]["values"][2] = 11
        restoration["restored_descriptor"]["values_sha256"] = executor.sha256_json(
            restoration["restored_descriptor"]["values"]
        )
        restoration["restored_sha256"] = executor.sha256_json(
            restoration["restored_descriptor"]
        )
        with self.assertRaises(aggregator.HeldOutAggregationError):
            aggregator._validate_delta_scoped_restoration(
                intervention["applied_receipt"],
                restoration,
                label="synthetic",
            )

    def test_uniform_schedule_replay_uses_no_fault_identity(self) -> None:
        clean = executor._default_action_sequence(0)
        receipt = executor._schedule_receipt(clean)
        self.assertTrue(receipt["replay"]["schedule_exact"])
        extra = copy.deepcopy(clean)
        extra["events"].append(
            {
                "event_index": 1,
                "request_index": 0,
                "role": "generic-hidden-retry",
                "input_coordinate": "frozen_query_bank[rank][0][31]",
                "externally_advertised": False,
            }
        )
        extra["actual_model_invocations"] = 2
        extra["events_sha256"] = executor.sha256_json(extra["events"])
        with self.assertRaises(executor.ReceiptPredicateRejection):
            executor._schedule_receipt(extra)

    def test_detector_and_scoring_sources_are_fault_id_blind(self) -> None:
        source = "\n".join(
            inspect.getsource(function)
            for function in (
                executor._schedule_receipt,
                executor.validate_completed_request_kv_horizon,
                executor.run_receipt_battery,
                executor.restore_mutation_delta_scope,
                executor.compare_semantic_outputs,
                executor.classify_authenticated_rejection,
                executor.classify_production_assertion,
                aggregator._validate_delta_scoped_restoration,
            )
        )
        for fault_id in fault_suite.FAULT_IDS:
            self.assertNotIn(fault_id, source)
        self.assertNotIn("expected_gate", source)

    def test_full_logits_equal_difference_and_missing_semantics(self) -> None:
        base = np.zeros((1, 248320), dtype=np.float32)
        base[0, 7] = 2.0
        changed = base.copy()
        changed[0, 8] = 3.0
        clean = {"semantic_horizon_reached": True, "greedy_token_id": 7, "full_logits": self._sidecar(0, "cmp-clean", base)}
        same = {"semantic_horizon_reached": True, "greedy_token_id": 7, "full_logits": self._sidecar(0, "cmp-same", base)}
        different = {"semantic_horizon_reached": True, "greedy_token_id": 8, "full_logits": self._sidecar(0, "cmp-diff", changed)}
        self.assertFalse(executor.compare_semantic_outputs(clean, same, raw_root=self.raw)["full_fp32_logits"]["caught"])
        compared = executor.compare_semantic_outputs(clean, different, raw_root=self.raw)
        self.assertTrue(compared["greedy_token"]["caught"])
        self.assertTrue(compared["full_fp32_logits"]["caught"])
        missing = executor.compare_semantic_outputs(clean, {"semantic_horizon_reached": False}, raw_root=self.raw)
        self.assertEqual(missing["full_fp32_logits"]["status"], "not_evaluated")
        self.assertIsNone(missing["full_fp32_logits"]["caught"])

    def test_only_exact_production_kernel_assertion_is_conventional_catch(self) -> None:
        from qcomem_vllm_paged_kernel import QComemPagedKernelError

        namespace = {"Error": QComemPagedKernelError}
        exec(
            compile(
                "def _paired_sequence():\n    raise Error('fused backend requires paired Q16 paged views')\n",
                "qcomem_vllm_paged_kernel.py",
                "exec",
            ),
            namespace,
        )
        try:
            namespace["_paired_sequence"]()
        except QComemPagedKernelError as exc:
            self.assertIsNotNone(executor.classify_production_assertion(exc))
        self.assertIsNone(executor.classify_production_assertion(RuntimeError("generic crash")))

    def test_strict_valid_aggregate_retains_each_fault_and_no_rate(self) -> None:
        result = self._aggregate(self._ranks())
        self.assertTrue(result["scientific_valid"])
        self.assertEqual([row["fault_id"] for row in result["per_fault_rows"]], ["H01", "H02", "H03"])
        self.assertFalse(result["detection_rate_reported"])
        self.assertFalse(result["per_fault_rows"][0]["forkaudit_lane"]["receipt_detector"]["caught"])
        self.assertTrue(result["per_fault_rows"][1]["forkaudit_lane"]["receipt_detector"]["caught"])
        self.assertNotIn('"detection_rate":', json.dumps(result))

    def test_clean_lane_must_complete_and_pass_all_receipts(self) -> None:
        ranks = self._ranks()
        clean = ranks[0]["lanes"][0]
        clean["completion_status"] = "authenticated_forkaudit_rejection_before_horizon"
        clean["semantic_horizon_reached"] = False
        with self.assertRaises(aggregator.HeldOutAggregationError):
            self._aggregate(ranks)

    def test_case_nonces_prove_fresh_lane_identity(self) -> None:
        ranks = self._ranks()
        ranks[0]["lanes"][1]["case_nonce"] = ranks[0]["lanes"][0]["case_nonce"]
        with self.assertRaises(aggregator.HeldOutAggregationError):
            self._aggregate(ranks)

    def test_operational_invalid_is_not_a_scientific_outcome(self) -> None:
        ranks = self._ranks()
        case = ranks[0]["lanes"][1]
        case["operational_invalid"] = {"classification": "synthetic generic crash"}
        case["completion_status"] = "operational_invalid"
        ranks[0]["operational_invalid_count"] = 1
        result = self._aggregate(ranks)
        self.assertFalse(result["scientific_valid"])
        self.assertEqual(result["operational_invalid_count"], 1)

    def test_cleanup_failure_is_operationally_invalid(self) -> None:
        ranks = self._ranks()
        case = ranks[2]["lanes"][2]
        case["cleanup"]["cleanup_passed"] = False
        case["cleanup"]["cleanup_error"] = {"type": "synthetic"}
        case["operational_invalid"] = {"classification": "cleanup"}
        ranks[2]["operational_invalid_count"] = 1
        result = self._aggregate(ranks)
        self.assertFalse(result["scientific_valid"])
        self.assertEqual(result["operational_invalid_count"], 1)

    def test_missing_fault_output_is_not_evaluated_not_not_caught(self) -> None:
        ranks = self._ranks()
        case = ranks[0]["lanes"][1]
        (self.raw / case["full_logits"]["path"]).unlink()
        case["completion_status"] = "production_assertion"
        case["semantic_horizon_reached"] = False
        case["greedy_token_id"] = None
        case["full_logits"] = None
        case["production_assertion"] = _production_catch()
        case["semantic_comparisons"] = executor.compare_semantic_outputs(ranks[0]["lanes"][0], case, raw_root=self.raw)
        result = self._aggregate(ranks)
        row = result["per_fault_rows"][0]["conventional_lane"]
        self.assertTrue(result["scientific_valid"])
        self.assertEqual(row["greedy_token"]["status"], "not_evaluated")
        self.assertIsNone(row["full_fp32_logits"]["caught"])

    def test_nonfinite_sidecar_is_rejected(self) -> None:
        ranks = self._ranks()
        reference = ranks[0]["lanes"][0]["full_logits"]
        path = self.raw / reference["path"]
        values = np.frombuffer(path.read_bytes(), dtype=np.float32).copy()
        values[0] = np.nan
        payload = values.tobytes()
        path.write_bytes(payload)
        reference["sha256"] = _sha(payload)
        with self.assertRaises(aggregator.HeldOutAggregationError):
            self._aggregate(ranks)

    def test_detector_tristate_is_fail_closed(self) -> None:
        ranks = self._ranks()
        ranks[0]["lanes"][1]["forkaudit"]["caught"] = False
        with self.assertRaises(aggregator.HeldOutAggregationError):
            self._aggregate(ranks)

    def test_aggregator_replays_generic_schedule_predicate(self) -> None:
        ranks = self._ranks()
        h02 = ranks[1]["lanes"][2]
        h02["completed_receipts"] = [
            {"receipt_id": receipt_id, "status": "passed", "payload": {}}
            for receipt_id in aggregator.RECEIPT_ORDER
        ]
        h02["first_authenticated_rejection"] = None
        h02["forkaudit"] = _cell("evaluated", False)
        with self.assertRaises(aggregator.HeldOutAggregationError):
            self._aggregate(ranks)


if __name__ == "__main__":
    unittest.main()
