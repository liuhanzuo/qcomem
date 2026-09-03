from __future__ import annotations

import copy
import inspect
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

import r29_live_overhead as runner
import r29_replay_live_overhead as replay


REPO = Path(__file__).resolve().parents[1]
DESIGN = (
    REPO
    / "paper_autonomous_multifork_iteration/evidence/r29_live_overhead/"
    "preregistration.json"
)


def fake_request_gdn_raw_witness() -> dict[str, object]:
    rows = []
    for layer_index in replay.LINEAR_LAYERS:
        for state_family in replay.REQUEST_GDN_STATE_FAMILIES:
            coordinate = f"{layer_index}/{state_family}/0"
            rows.append(
                {
                    "request_index": 0,
                    "layer_index": layer_index,
                    "state_family": state_family,
                    "state_index": 0,
                    "expected_relation": "rebound",
                    "baseline_binding_token": replay.sha256_bytes(
                        f"baseline-binding/{coordinate}".encode("ascii")
                    ),
                    "observed_binding_token": replay.sha256_bytes(
                        f"observed-binding/{coordinate}".encode("ascii")
                    ),
                    "baseline_storage_token": replay.sha256_bytes(
                        f"baseline-storage/{coordinate}".encode("ascii")
                    ),
                    "observed_storage_token": replay.sha256_bytes(
                        f"observed-storage/{coordinate}".encode("ascii")
                    ),
                }
            )
    return {
        "guard_id": "a" * 32,
        "capture_id": None,
        "policy": "shared-base",
        "layer_indices": list(replay.LINEAR_LAYERS),
        "state_index": 0,
        "resident_count": 1,
        "completed_request_indices": [0],
        "expected_tensor_count_per_owner": replay.REQUEST_GDN_TENSORS_PER_OWNER,
        "rows": rows,
        "rows_sha256": replay.sha256_canonical_json(rows),
    }


def fake_live_receipt() -> dict[str, object]:
    return {
        "ledger": {"verified": True, "total_calls": 10},
        "kv_pre": {"passed": True},
        "kv_post": {"passed": True},
        "request_gdn": fake_request_gdn_raw_witness(),
        "persistent_gdn": {
            "baseline_binding_sha256": "1" * 64,
            "observed_binding_sha256": "1" * 64,
            "baseline_content_sha256": "2" * 64,
            "observed_content_sha256": "2" * 64,
        },
        "source_document_sha256_before": {"3": "3" * 64},
        "source_document_sha256_after": {"3": "3" * 64},
        "source_document_immutable": True,
    }


def write_fake_capture(root: Path, sample_id: str) -> dict[str, object]:
    writer = runner.LiveCaptureWriter(root / sample_id, sample_id)
    for layer_index in runner.FULL_LAYERS:
        capture_id = writer.append_observer(
            layer_index,
            {
                "key_states": torch.full((1, 1, 1, 2), layer_index, dtype=torch.float32),
                "value_states": torch.full((1, 1, 1, 2), layer_index + 1, dtype=torch.float32),
                "append_event_index": 0,
                "appended_tokens_before": 0,
                "appended_tokens_after": 1,
                "sequence_length_before": 4,
                "sequence_length_after": 5,
                "source_device": "cuda:0",
                "source_dtype": "torch.bfloat16",
                "source_shape": [1, 2, 1, 256],
            },
        )
        writer.call_observer(
            {
                "observer_schema": "qcomem-forkaudit-call-observer-v2",
                "layer_idx": layer_index,
                "request_index": 0,
                "resident_count": 1,
                "request_policy": "vllm-q16-shared-document-reuse",
                "attention_mask_is_none": True,
                "append_capture_id": capture_id,
                "append_audit": {"capture_id": capture_id},
                "position_audit": {"passed": True},
                "kernel_audit": {"passed": True},
                "effective_scaling": 0.125,
                "query_cpu": torch.full((1, 1, 1, 2), layer_index, dtype=torch.float32),
                "candidate_output_cpu": torch.full(
                    (1, 1, 1, 2), layer_index + 2, dtype=torch.float32
                ),
                "position_ids_cpu": torch.tensor([[4]], dtype=torch.int64),
            }
        )
    return writer.finalize(fake_live_receipt())


def timing_cell(
    arm: str,
    sample_id: str,
    logits: torch.Tensor,
    *,
    wall_ns: int,
    manifest: dict[str, object] | None,
) -> dict[str, object]:
    instrumented = arm == "instrumented"
    before = 1000
    peak = 1100 if not instrumented else 1300
    return {
        "arm": arm,
        "wall_time_ns": wall_ns,
        "wall_time_ms": wall_ns / 1_000_000.0,
        "allocated_before_bytes": before,
        "reserved_before_bytes": 2000,
        "peak_allocated_bytes": peak,
        "incremental_peak_allocated_bytes": peak - before,
        "allocated_after_bytes": 1050,
        "reserved_after_bytes": 2000,
        "audit_artifact_bytes": 0 if manifest is None else manifest["artifact_bytes"],
        "cuda_synchronized_before_start": True,
        "cuda_synchronized_before_stop": True,
        "peak_stats_reset_before_start": True,
        "capture_policy": (
            "full-live-capture-and-ownership-receipt"
            if instrumented
            else "optional-forkaudit-capture-disabled"
        ),
        "mandatory_functional_adapter_checks_retained": True,
        "append_observer_enabled": instrumented,
        "call_observer_enabled": instrumented,
        "ownership_receipt_enabled": instrumented,
        "audit_artifact_manifest": manifest,
        "ledger": {
            "implementation": "MultiForkHitLedger",
            "explicit_frozen_kernel": True,
            "verified": True,
            "total_calls": 10,
            "call_observer_enabled": instrumented,
            "kernel_identity": {
                "module": "frozen.fake_kernel",
                "qualname": "unified_attention",
                "signature": "(*args, **kwargs)",
            },
        },
        "output": {
            "sample_id": sample_id,
            "shape": list(logits.shape),
            "dtype": "float32",
            "full_vocab_logit_sha256": runner.sha256_tensor(logits),
            "token_id": int(logits.argmax(dim=-1).item()),
            "finite": True,
            "copied_after_timed_region": True,
        },
    }


class R29LiveOverheadTest(unittest.TestCase):
    def design(self) -> dict[str, object]:
        return json.loads(DESIGN.read_text(encoding="utf-8"))

    def test_preregistration_freezes_live_pair_boundary(self) -> None:
        design = self.design()
        runner.validate_design(design)
        self.assertEqual(design["timing_population"]["discarded_warmup_pairs"], 1)
        self.assertEqual(design["timing_population"]["measured_pairs"], 5)
        self.assertFalse(design["timing_population"]["document_prefill_in_timed_region"])
        self.assertTrue(
            design["timing_and_memory_protocol"][
                "common_final_full_logit_copy_occurs_after_timer_stop"
            ]
        )
        self.assertIn(
            "throughput, QPS, serving capacity, or concurrency scaling",
            design["claim_boundary"]["not_established"],
        )

    def test_summary_preserves_negative_numeric_delta(self) -> None:
        pairs = []
        for pair_index, order, baseline_slot, instrumented_slot in runner.MEASURED_SCHEDULE:
            baseline_ns = 200 if pair_index == 0 else 100
            instrumented_ns = 100 if pair_index == 0 else 120
            pairs.append(
                {
                    "pair_index": pair_index,
                    "execution_order": list(order),
                    "baseline_slot": baseline_slot,
                    "instrumented_slot": instrumented_slot,
                    "pair_valid": True,
                    "cells": {
                        "baseline": {
                            "wall_time_ns": baseline_ns,
                            "incremental_peak_allocated_bytes": 20,
                            "audit_artifact_bytes": 0,
                        },
                        "instrumented": {
                            "wall_time_ns": instrumented_ns,
                            "incremental_peak_allocated_bytes": 24,
                            "audit_artifact_bytes": 50,
                        },
                    },
                }
            )
        summary = runner.summarize_measured_pairs(pairs)
        self.assertEqual(summary["rows"][0]["paired_wall_delta_ns"], -100)
        self.assertTrue(summary["negative_numeric_deltas_preserved"])
        self.assertEqual(len(summary["rows"]), 5)

    def test_capture_receipt_roundtrip_and_tamper_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = write_fake_capture(root, "sample")
            observed = replay.verify_capture_artifact(root, manifest)
            self.assertTrue(observed["passed"])
            self.assertEqual(observed["tensor_record_count"], 50)
            capture = root / "sample" / "capture.bin"
            raw = bytearray(capture.read_bytes())
            raw[-1] ^= 1
            capture.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError, "capture SHA drift"):
                replay.verify_capture_artifact(root, manifest)

    def test_request_gdn_raw_witness_derivation_and_tamper_gates(self) -> None:
        replay_source = inspect.getsource(replay)
        self.assertNotIn("qcomem_forkaudit_storage_witness", replay_source)
        self.assertNotIn("replay_request_gdn_binding_witness", replay_source)
        witness = fake_request_gdn_raw_witness()
        self.assertNotIn("rebound_tensor_count", witness)
        observed = replay.replay_request_gdn_raw_witness(witness)
        self.assertTrue(observed["passed"])
        self.assertEqual(observed["completed_request_indices"], [0])
        self.assertEqual(observed["rebound_tensor_count"], 60)
        self.assertEqual(observed["unchanged_tensor_count"], 0)

        digest_tamper = copy.deepcopy(witness)
        digest_tamper["rows"][0]["expected_relation"] = "unchanged"
        with self.assertRaisesRegex(RuntimeError, "row digest drift"):
            replay.replay_request_gdn_raw_witness(digest_tamper)

        order_tamper = copy.deepcopy(witness)
        order_tamper["rows"][0], order_tamper["rows"][1] = (
            order_tamper["rows"][1],
            order_tamper["rows"][0],
        )
        order_tamper["rows_sha256"] = replay.sha256_canonical_json(
            order_tamper["rows"]
        )
        with self.assertRaisesRegex(RuntimeError, "coordinate order drift"):
            replay.replay_request_gdn_raw_witness(order_tamper)

        storage_tamper = copy.deepcopy(witness)
        storage_tamper["rows"][0]["observed_storage_token"] = storage_tamper[
            "rows"
        ][0]["baseline_storage_token"]
        storage_tamper["rows_sha256"] = replay.sha256_canonical_json(
            storage_tamper["rows"]
        )
        with self.assertRaisesRegex(RuntimeError, "storage token did not change"):
            replay.replay_request_gdn_raw_witness(storage_tamper)

        token_tamper = copy.deepcopy(witness)
        token_tamper["rows"][0]["observed_binding_token"] = "not-a-sha256"
        token_tamper["rows_sha256"] = replay.sha256_canonical_json(
            token_tamper["rows"]
        )
        with self.assertRaisesRegex(RuntimeError, "token drift"):
            replay.replay_request_gdn_raw_witness(token_tamper)

        schema_tamper = copy.deepcopy(witness)
        schema_tamper["rebound_tensor_count"] = 60
        with self.assertRaisesRegex(RuntimeError, "witness schema drift"):
            replay.replay_request_gdn_raw_witness(schema_tamper)

    def test_semantic_sidecar_roundtrip_and_tamper_gate(self) -> None:
        logits = {
            "baseline": torch.arange(8, dtype=torch.float32).reshape(1, 8),
            "instrumented": torch.arange(8, dtype=torch.float32).reshape(1, 8),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = runner.write_logit_sidecar(root / "semantic.bin", logits)
            observed = replay.read_semantic_sidecar(root, manifest)
            np.testing.assert_array_equal(observed["baseline"], logits["baseline"].numpy())
            path = root / "semantic.bin"
            raw = bytearray(path.read_bytes())
            raw[0] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError, "semantic sidecar SHA drift"):
                replay.read_semantic_sidecar(root, manifest)

    def test_complete_fake_result_replays_independently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logits_by_id: dict[str, torch.Tensor] = {}

            def pair(
                label: str,
                index: int | None,
                warmup: bool,
                order: tuple[str, str],
                baseline_slot: int,
                instrumented_slot: int,
            ) -> dict[str, object]:
                logits = torch.arange(16, dtype=torch.float32).reshape(1, 16)
                baseline_id = f"{label}-baseline"
                instrumented_id = f"{label}-instrumented"
                logits_by_id[baseline_id] = logits.clone()
                logits_by_id[instrumented_id] = logits.clone()
                manifest = write_fake_capture(root, instrumented_id)
                baseline = timing_cell(
                    "baseline",
                    baseline_id,
                    logits,
                    wall_ns=1000 + (index or 0),
                    manifest=None,
                )
                instrumented = timing_cell(
                    "instrumented",
                    instrumented_id,
                    logits,
                    wall_ns=1200 + (index or 0),
                    manifest=manifest,
                )
                return {
                    "pair_label": label,
                    "pair_index": index,
                    "warmup": warmup,
                    "discarded_from_estimands": warmup,
                    "execution_order": list(order),
                    "baseline_slot": baseline_slot,
                    "instrumented_slot": instrumented_slot,
                    "same_persistent_document_within_pair": True,
                    "same_query_tokens_within_pair": True,
                    "distinct_request_objects": True,
                    "source_document_immutable": True,
                    "persistent_gdn_immutable": True,
                    "cells": {"baseline": baseline, "instrumented": instrumented},
                    "semantic_oracle": {
                        "full_vocab_logits_torch_equal": True,
                        "generated_token_equal": True,
                        "baseline_token_id": 15,
                        "instrumented_token_id": 15,
                        "baseline_sha256": runner.sha256_tensor(logits),
                        "instrumented_sha256": runner.sha256_tensor(logits),
                        "max_abs_error": 0.0,
                        "mean_abs_error": 0.0,
                    },
                    "pair_valid": True,
                }

            warmup = pair("warmup-pair", None, True, runner.WARMUP_ORDER, 1, 0)
            measured = [
                pair(
                    f"measured-pair-{pair_index}",
                    pair_index,
                    False,
                    order,
                    baseline_slot,
                    instrumented_slot,
                )
                for pair_index, order, baseline_slot, instrumented_slot in runner.MEASURED_SCHEDULE
            ]
            sidecar = runner.write_logit_sidecar(root / "semantic-logits.fp32.bin", logits_by_id)
            summary = runner.summarize_measured_pairs(measured)
            result = {
                "schema_version": runner.SCHEMA,
                "status": "completed",
                "scientific_run_valid": True,
                "formal_evidence_eligible": True,
                "warmup_pair": warmup,
                "measured_pairs": measured,
                "paired_summary": summary,
                "semantic_sidecar": sidecar,
                "validity": {
                    "warmup_pair_count": 1,
                    "warmup_discarded_from_estimands": True,
                    "measured_pair_count": 5,
                    "alternating_schedule_verified": True,
                    "negative_numeric_deltas_removed": False,
                },
            }
            observed = replay.replay_result(self.design(), result, root)
            self.assertTrue(observed["replay_passed"])
            self.assertTrue(observed["scientific_run_valid_recomputed"])
            self.assertEqual(len(observed["artifact_rows"]), 6)

    def test_timed_regions_make_cuda_boundaries_and_logit_copy_explicit(self) -> None:
        for function in (runner.run_baseline_arm, runner.run_instrumented_arm):
            source = inspect.getsource(function)
            self.assertIn("allocator_before_arm()", source)
            self.assertIn("time.perf_counter_ns()", source)
            self.assertIn("torch.cuda.synchronize()", source)
            self.assertLess(
                source.index("time.perf_counter_ns() - start_ns"),
                source.index("finalize_common_output"),
            )
        instrumented = inspect.getsource(runner.run_instrumented_arm)
        for needle in (
            "capture_persistent_gdn_guard",
            "capture_request_gdn_binding_guard",
            "validate_runtime_kv_ownership",
            "collector.finalize",
        ):
            self.assertIn(needle, instrumented)

    def test_entire_pair_fails_closed_without_unified_inference_mode(self) -> None:
        arguments = (
            None,
            None,
            None,
            torch.empty(0),
            torch.empty(0),
            lambda: None,
            Path("unused"),
        )
        keywords = {
            "pair_label": "test-pair",
            "pair_index": None,
            "warmup": True,
            "execution_order": runner.WARMUP_ORDER,
            "baseline_slot": 1,
            "instrumented_slot": 0,
        }
        with torch.inference_mode(False):
            with self.assertRaisesRegex(RuntimeError, "entire pair requires"):
                runner.run_pair(*arguments, **keywords)

        def guarded_prefill(*_args: object, **_kwargs: object) -> None:
            self.assertTrue(torch.is_inference_mode_enabled())
            raise RuntimeError("guarded-prefill-reached")

        with mock.patch.object(runner, "_build_document_cache", guarded_prefill):
            with torch.inference_mode():
                with self.assertRaisesRegex(RuntimeError, "guarded-prefill-reached"):
                    runner.run_pair(*arguments, **keywords)

        run_gpu_source = inspect.getsource(runner.run_gpu)
        self.assertLess(
            run_gpu_source.index("with torch.inference_mode():"),
            run_gpu_source.index("warmup, warmup_logits = run_pair("),
        )

    def test_both_arms_share_one_explicit_kernel_ledger_factory(self) -> None:
        helper = inspect.getsource(runner.build_arm_ledger)
        baseline = inspect.getsource(runner.run_baseline_arm)
        instrumented = inspect.getsource(runner.run_instrumented_arm)
        self.assertIn("return MultiForkHitLedger(", helper)
        self.assertIn("kernel=kernel", helper)
        self.assertIn("call_observer=call_observer", helper)
        for source in (baseline, instrumented):
            self.assertIn("ledger = build_arm_ledger(", source)
            self.assertNotIn("Qwen35VllmPagedHitLedger", source)
            self.assertIn("unified pair inference-mode scope", source)
        self.assertIn("call_observer=None", baseline)
        self.assertIn("call_observer=collector.call_observer", instrumented)


if __name__ == "__main__":
    unittest.main()
