from __future__ import annotations

import json
import copy
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from qcomem_vllm_paged_fair_control import FRESH_CONTROL, SHARED_REUSE
from run_qcomem_qwen35_vllm_paged_fair_v2 import (
    FAIR_PROTOCOL,
    FULL_LAYERS,
    KERNEL_MODE,
    POST_ROPE_POSITION_IDS_CONTRACT,
    PRODUCTION_MASK_CONTRACT,
    TEST_V2_SHA256,
    _model_manifest_sha,
    _median_trials,
    _parser,
    _protocol_config,
    _protocol_config_sha256,
    _run_fresh_abba,
    _safe_ratio,
    _static_frozen_identity,
    _validate_sha256_hex,
    _validate_static,
    run_validation,
    aggregate_pg19_gate_shards,
    summarize_validation_shards,
    sha256_file,
)


def frozen_protocol_config() -> dict:
    return {
        "bits": 16,
        "page_size": 128,
        "world_size": 8,
        "pg19_books": 8,
        "pg19_document_tokens": 1025,
        "pg19_query_tokens": 32,
        "pg19_window_stride": 257,
        "pg19_candidate_windows": 8,
        "pg19_seed": 20260814,
        "max_input_tokens": 4096,
        "max_query_tokens": 64,
        "max_new_tokens": 8,
        "source_index_start": 6,
        "source_index_end": 9,
        "limit_per_dataset": 4,
        "min_input_tokens": 1,
        "expected_source_revision": "5" * 40,
        "quantization": "Q16",
        "single_request_only": True,
        "batch_semantics": "batch-1-equal-length-only",
    }


def frozen_identity(validation_sha256: str) -> dict[str, str]:
    protocol = frozen_protocol_config()
    return {
        "code_ledger_sha256": "1" * 64,
        "model_manifest_sha256": "2" * 64,
        "model_artifact_ledger_sha256": "3" * 64,
        "model_weight_ledger_sha256": "4" * 64,
        "pg19_data_sha256": "5" * 64,
        "pg19_manifest_sha256": "6" * 64,
        "pg19_windows_sha256": "7" * 64,
        "validation_expected_sha256_recorded_but_not_hashed": validation_sha256,
        "protocol_manifest_sha256": "9" * 64,
        "protocol_config_sha256": _protocol_config_sha256(protocol),
    }


def authorization_value(validation_sha256: str) -> dict:
    return {
        "status": "pg19_fair_v2_authorized",
        "passed": True,
        "fair_protocol": FAIR_PROTOCOL,
        "same_kernel_layout_gate_passed": True,
        "same_kernel_full_vocab_logit_gate_passed": True,
        "validation_consumed": False,
        "validation_hashed": False,
        "source_68_99_consumed": False,
        "test_v2_consumed": False,
        "frozen_identity": frozen_identity(validation_sha256),
        "protocol_config": frozen_protocol_config(),
    }


def fake_intercept(policy: str, calls_per_layer: int = 8) -> dict:
    calls = []
    identity = {
        "callable_id": 123,
        "module": "vllm",
        "qualname": "unified_attention",
        "signature": "(**kwargs)",
    }
    for _step in range(calls_per_layer):
        for layer in FULL_LAYERS:
            calls.append(
                {
                    "layer_idx": layer,
                    "request_policy": policy,
                    "fair_protocol": FAIR_PROTOCOL,
                    "same_unified_attention_kernel": True,
                    "kernel_identity": identity,
                    "kernel_mode": KERNEL_MODE,
                    "fused_gpu_kernel_calls": 1,
                    "full_kv_concatenations": 0,
                    "current_append_delta_tokens": 1,
                    "query_tokens": 1,
                    "quantization": "Q16",
                    "materialized_attention_mask_nbytes": 0,
                    "mask_validation_host_syncs": 0,
                    "position_ids_validation_host_syncs": 0,
                }
            )
    return {
        "verified": True,
        "fair_protocol": FAIR_PROTOCOL,
        "request_policy": policy,
        "kernel_mode": KERNEL_MODE,
        "same_unified_attention_kernel": True,
        "kernel_identity": identity,
        "expected_layer_indices": list(FULL_LAYERS),
        "counts": {index: calls_per_layer for index in FULL_LAYERS},
        "total_calls": len(calls),
        "dense_fallback_calls": 0,
        "full_kv_concatenations": 0,
        "position_ids_contract": POST_ROPE_POSITION_IDS_CONTRACT,
        "mask_contract": PRODUCTION_MASK_CONTRACT,
        "materialized_attention_mask_nbytes": 0,
        "mask_validation_host_syncs": 0,
        "position_ids_validation_host_syncs": 0,
        "calls": calls,
    }


def fake_allocator(label: str, value: int) -> dict:
    return {
        "label": label,
        "current_allocated_bytes": value,
        "current_reserved_bytes": value + 10,
        "peak_allocated_bytes": value + 20,
        "peak_reserved_bytes": value + 30,
    }


def fake_storage(policy: str, appended_tokens: int) -> dict:
    fresh = policy == FRESH_CONTROL
    page_size = 128
    document_length = 1025
    detached_tail = document_length % page_size if appended_tokens else 0
    active_tokens = detached_tail + appended_tokens
    active_blocks = 1 if active_tokens else 0
    layers = []
    for layer in FULL_LAYERS:
        layers.append(
            {
                "layer_idx": layer,
                "block_bytes_formula": (
                    "2*page_size*kv_heads*head_dim*element_size"
                ),
                "block_bytes": 128,
                "valid_document_payload_nbytes": 1025,
                "source_document_allocated_nbytes": 1152,
                "source_document_padding_nbytes": 127,
                "source_private_reservation_nbytes": 128,
                "source_total_arena_allocated_nbytes": 1280,
                "fresh_duplicate_document_allocated_nbytes": 1152 if fresh else 0,
                "fresh_duplicate_document_padding_nbytes": 127 if fresh else 0,
                "fresh_private_reservation_nbytes": 128 if fresh else 0,
                "active_request_private_payload_nbytes": active_tokens,
                "active_request_private_blocks": active_blocks,
                "active_request_private_allocated_page_nbytes": 128 * active_blocks,
                "request_private_reserved_unused_nbytes": 128 * (1 - active_blocks),
                "active_request_appended_tokens": appended_tokens,
                "active_request_detached_tail_tokens": detached_tail,
                "request_block_table_accelerator_nbytes": 4,
                "source_document_table_accelerator_nbytes": 4,
                "fresh_document_table_accelerator_nbytes": 4 if fresh else 0,
                "source_cpu_reservation_metadata_nbytes": 8,
                "fresh_cpu_reservation_metadata_nbytes": 8 if fresh else 0,
                "source_document_storage_shared_by_request": not fresh,
            }
        )
    total_keys = {
        key
        for key in layers[0]
        if key.endswith("_nbytes")
        or key in ("block_bytes", "active_request_private_blocks")
    }
    totals = {key: sum(row[key] for row in layers) for key in total_keys}
    return {
        "request_policy": policy,
        "full_attention_layer_count": 10,
        "scope": "ten-full-attention-layers-only",
        "linear_gdn_included": False,
        "source_arena_includes_preallocated_private_reservation": True,
        "invalid_final_block_padding_is_payload": False,
        "totals": totals,
        "layers": layers,
    }


def refresh_storage_totals(storage: dict) -> None:
    numeric_keys = {
        key
        for key in storage["layers"][0]
        if key.endswith("_nbytes")
        or key in ("block_bytes", "active_request_private_blocks")
    }
    storage["totals"] = {
        key: sum(row[key] for row in storage["layers"]) for key in numeric_keys
    }


def fake_measurement(policy: str) -> dict:
    fresh = policy == FRESH_CONTROL
    copy_bytes = 11520 if fresh else 0
    storage_before = fake_storage(policy, 0)
    storage_after = fake_storage(policy, 8)
    audit = {
        "allocated_request_pool_nbytes": 12800 if fresh else 0,
        "full_document_staging_copy_nbytes": copy_bytes,
        "source_document_storage_shared": not fresh,
    }
    trial = {
        "config": policy,
        "same_unified_attention_kernel": True,
        "allocator_fresh_trial_baseline": {
            "current_allocated_bytes": 10,
            "current_reserved_bytes": 20,
        },
        "abba_allocator_baseline": {
            "current_allocated_bytes": 10,
            "current_reserved_bytes": 20,
        },
        "kernel_identity": {
            "callable_id": 123,
            "module": "vllm",
            "qualname": "unified_attention",
            "signature": "(**kwargs)",
        },
        "generated_token_ids": [1] * 8,
        "full_vocab_step_logit_sha256": ["a" * 64] * 8,
        "intercept": fake_intercept(policy),
        "arm_difference_scope": "ten-full-attention-cache-ownership-only",
        "linear_gdn_shared_base_at_request_start": {
            "passed": True,
            "linear_layer_count": 30,
            "persistent_tensor_base_shared_at_request_start": True,
            "request_updates_are_functional_rebind": True,
        },
        "document_build": {
            "dense_document_prefill_seconds": 1.0,
            "q16_pool_build_seconds": 2.0,
            "dense_document_prefill_cuda_peak_delta_bytes": 70,
            "dense_document_prefill_cuda_peak_reserved_delta_bytes": 70,
            "dense_document_prefill_cuda_current_allocated_delta_bytes": 50,
            "dense_document_prefill_cuda_current_reserved_delta_bytes": 50,
            "q16_pool_build_cuda_peak_delta_bytes": 70,
            "q16_pool_build_cuda_peak_reserved_delta_bytes": 70,
            "q16_pool_build_cuda_current_allocated_delta_bytes": 50,
            "q16_pool_build_cuda_current_reserved_delta_bytes": 50,
            "dense_to_nhd_document_copy_nbytes": 10250,
            "q16_document_payload_nbytes": 10250,
            "q16_allocated_source_pool_nbytes": 12800,
            "full_attention_layers": 10,
            "dense_document_prefill_allocator_before": fake_allocator(
                "before-prefill", 50
            ),
            "dense_document_prefill_allocator_after": fake_allocator(
                "prefill", 100
            ),
            "q16_pool_build_allocator_before": fake_allocator(
                "before-pack", 150
            ),
            "q16_pool_build_allocator_after": fake_allocator("pack", 200),
        },
        "query_preparation": {
            "seconds": 1.0 if fresh else 0.5,
            "cuda_peak_delta_bytes": 0,
            "cuda_peak_reserved_delta_bytes": 0,
            "allocator_before": fake_allocator("before-setup", 200),
            "allocator_after": fake_allocator("setup", 180),
            "cuda_current_allocated_delta_bytes": -20,
            "cuda_current_reserved_delta_bytes": -20,
            "physical_document_block_copy_nbytes_including_padding": copy_bytes,
            "physical_document_block_copy_included_in_seconds": True,
            "fresh_pool_allocation_included_in_seconds": fresh,
            "partial_document_tail_cow_included_in_request_setup": False,
            "partial_document_tail_cow_occurs_in_first_continuation_step": True,
            "audit": audit,
        },
        "memory_before_decode": {"combined_unique_accelerator_nbytes": 3000 if fresh else 1800},
        "memory_after_decode": {"combined_unique_accelerator_nbytes": 3200 if fresh else 2000},
        "continuation_append_accounting": {
            "partial_tail_staging_copy_nbytes": 10,
            "phase": "first-continuation-model-step-common-append-path",
            "included_in_request_setup": False,
            "included_in_cached_document_request_ttft": True,
        },
        "full_attention_storage_before_decode": storage_before,
        "full_attention_storage_after_decode": storage_after,
        "conversion": {
            "document_length": 1025,
            "page_size": 128,
            "dense_document_nbytes": 10250,
            "document_payload_nbytes": 10250,
            "allocated_source_pool_nbytes": 12800,
        },
        "cuda_peak_request_delta_bytes": 0,
        "cuda_peak_request_reserved_delta_bytes": 0,
        "allocator_after_first_continuation_step": fake_allocator(
            "first-step", 180
        ),
        "allocator_after_generation": fake_allocator("generation", 180),
        "cuda_current_request_delta_bytes": -20,
        "cuda_current_request_reserved_delta_bytes": -20,
        "cached_document_request_ttft_seconds": 2.0 if fresh else 1.0,
        "continuation_model_first_token_seconds": 1.0 if fresh else 0.5,
        "median_tpot_seconds": 1.0 if fresh else 0.5,
    }
    trials = [copy.deepcopy(trial) for _ in range(4)]
    return _median_trials(policy, trials)


def make_static_args(root: Path) -> SimpleNamespace:
    model = root / "model"
    model.mkdir()
    config = {
        "text_config": {
            "model_type": "qwen3_5_moe_text",
            "num_hidden_layers": 40,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "layer_types": [
                "full_attention" if index in FULL_LAYERS else "linear_attention"
                for index in range(40)
            ],
        }
    }
    (model / "config.json").write_text(json.dumps(config))
    (model / "generation_config.json").write_text("{}\n")
    (model / "model.safetensors.index.json").write_text("{}\n")
    pg19 = root / "pg19-train.jsonl"
    pg19_manifest = root / "pg19-train-manifest.json"
    validation = root / "longbench-validation.jsonl"
    protocol_manifest = root / "fair-v2-runtime-protocol.json"
    code = root / "code.sha256"
    artifacts = root / "model-artifacts.sha256"
    weights = root / "model-weights.sha256"
    for path, payload in (
        (pg19, "train-only\n"),
        (pg19_manifest, "{}\n"),
        (validation, "validation-never-hashed-in-static\n"),
        (code, "code ledger\n"),
        (artifacts, "artifact ledger\n"),
        (weights, "weight ledger\n"),
    ):
        path.write_text(payload)
    model_manifest, _ = _model_manifest_sha(model)
    args = SimpleNamespace(
        bits=16,
        world_size=8,
        rank=0,
        page_size=128,
        source_index_start=6,
        source_index_end=9,
        limit_per_dataset=4,
        min_input_tokens=1,
        expected_source_revision="5" * 40,
        max_new_tokens=8,
        pg19_document_tokens=1025,
        pg19_query_tokens=32,
        pg19_books=8,
        pg19_window_stride=257,
        pg19_candidate_windows=8,
        pg19_seed=20260814,
        max_input_tokens=4096,
        max_query_tokens=64,
        expected_pg19_sha256=sha256_file(pg19),
        expected_pg19_manifest_sha256=sha256_file(pg19_manifest),
        expected_pg19_windows_sha256="1" * 64,
        expected_validation_sha256=sha256_file(validation),
        expected_model_manifest_sha256=model_manifest,
        expected_code_ledger_sha256=sha256_file(code),
        expected_model_artifact_ledger_sha256=sha256_file(artifacts),
        expected_model_weight_ledger_sha256=sha256_file(weights),
        pg19_data=pg19,
        pg19_manifest=pg19_manifest,
        validation_data=validation,
        model=model,
        code_ledger=code,
        model_artifact_ledger=artifacts,
        model_weight_ledger=weights,
        protocol_manifest=protocol_manifest,
        expected_protocol_manifest_sha256="0" * 64,
    )
    protocol_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "fair_protocol": FAIR_PROTOCOL,
                "quantization": "Q16",
                "code_ledger_sha256": args.expected_code_ledger_sha256,
                "model_manifest_sha256": args.expected_model_manifest_sha256,
                "model_artifact_ledger_sha256": (
                    args.expected_model_artifact_ledger_sha256
                ),
                "model_weight_ledger_sha256": (
                    args.expected_model_weight_ledger_sha256
                ),
                "pg19_data_sha256": args.expected_pg19_sha256,
                "pg19_manifest_sha256": args.expected_pg19_manifest_sha256,
                "pg19_windows_sha256": args.expected_pg19_windows_sha256,
                "validation_expected_sha256_recorded_but_not_hashed": (
                    args.expected_validation_sha256
                ),
                "protocol_config": _protocol_config(args),
            },
            sort_keys=True,
        )
        + "\n"
    )
    args.expected_protocol_manifest_sha256 = sha256_file(protocol_manifest)
    return args


class FairV2StaticAndGovernanceTest(unittest.TestCase):
    def test_q8_q4_fail_before_any_hash_or_environment_access(self):
        for bits in (8, 4):
            args = SimpleNamespace(bits=bits)
            with (
                mock.patch(
                    "run_qcomem_qwen35_vllm_paged_fair_v2.sha256_file"
                ) as digest,
                mock.patch(
                    "run_qcomem_qwen35_vllm_paged_fair_v2.audit_frozen_kernel_environment"
                ) as environment,
            ):
                with self.assertRaisesRegex(RuntimeError, "Q16 only"):
                    _validate_static(args)
            digest.assert_not_called()
            environment.assert_not_called()

    def test_static_binds_code_model_data_protocol_without_hashing_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_static_args(Path(directory))
            calls: list[Path] = []

            def recording_digest(path):
                calls.append(Path(path))
                return sha256_file(path)

            environment = {
                "matches_frozen_environment": True,
                "mismatches": {},
            }
            with (
                mock.patch(
                    "run_qcomem_qwen35_vllm_paged_fair_v2.sha256_file",
                    side_effect=recording_digest,
                ),
                mock.patch(
                    "run_qcomem_qwen35_vllm_paged_fair_v2.audit_frozen_kernel_environment",
                    return_value=environment,
                ),
            ):
                result = _validate_static(args)
            self.assertEqual(result["status"], "fair_v2_static_dry_run_passed")
            self.assertNotIn(args.validation_data, calls)
            self.assertFalse(result["validation_hashed"])
            self.assertFalse(result["validation_consumed"])
            identity = _static_frozen_identity(result)
            self.assertEqual(identity["code_ledger_sha256"], args.expected_code_ledger_sha256)
            self.assertEqual(
                identity["protocol_manifest_sha256"],
                sha256_file(args.protocol_manifest),
            )

    def test_static_rejects_self_consistent_but_non_preregistered_values(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_static_args(Path(directory))
            args.max_new_tokens = 7
            with self.assertRaisesRegex(RuntimeError, "freezes max_new_tokens=8"):
                _validate_static(args)

    def test_static_rejects_semantically_drifted_rehashed_protocol_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            args = make_static_args(Path(directory))
            manifest = json.loads(args.protocol_manifest.read_text())
            manifest["protocol_config"]["pg19_seed"] = 1
            args.protocol_manifest.write_text(json.dumps(manifest) + "\n")
            args.expected_protocol_manifest_sha256 = sha256_file(
                args.protocol_manifest
            )
            with mock.patch(
                "run_qcomem_qwen35_vllm_paged_fair_v2.audit_frozen_kernel_environment",
                return_value={"matches_frozen_environment": True, "mismatches": {}},
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "manifest fields differ"
                ):
                    _validate_static(args)

    def test_parser_exposes_longbench_limit_per_dataset(self):
        parser = _parser()
        actions = {action.dest for action in parser._actions}
        self.assertIn("limit_per_dataset", actions)
        self.assertNotIn("max_examples_per_dataset", actions)

    def test_safe_ratio_does_not_turn_zero_baseline_into_fake_improvement(self):
        self.assertIsNone(_safe_ratio(0, 0))
        self.assertIsNone(_safe_ratio(10, 0))
        self.assertEqual(_safe_ratio(0, 10), 0.0)
        self.assertEqual(_safe_ratio(5, 10), 0.5)

    def test_aggregate_rejects_shards_from_a_different_current_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = {"bits": 16}
            wrong_identity = {
                "code_ledger_sha256": "1" * 64,
                "model_manifest_sha256": "2" * 64,
                "model_artifact_ledger_sha256": "3" * 64,
                "model_weight_ledger_sha256": "4" * 64,
                "pg19_data_sha256": "5" * 64,
                "pg19_manifest_sha256": "6" * 64,
                "pg19_windows_sha256": "7" * 64,
                "validation_expected_sha256_recorded_but_not_hashed": "8" * 64,
                "protocol_manifest_sha256": "9" * 64,
                "protocol_config_sha256": _protocol_config_sha256(protocol),
            }
            expected = dict(wrong_identity)
            expected["code_ledger_sha256"] = "a" * 64
            paths = []
            for rank in range(8):
                path = root / f"pg19-fair-v2-shard-{rank}.json"
                path.write_text(
                    json.dumps(
                        {
                            "status": "completed_pg19_fair_v2_gate_shard",
                            "passed": True,
                            "rank": rank,
                            "world_size": 8,
                            "fair_protocol": FAIR_PROTOCOL,
                            "kernel_mode": "vllm_0_26_triton_unified_attention_q16_block_pool",
                            "quantization": "Q16",
                            "single_request_only": True,
                            "static": {
                                "status": "fair_v2_static_dry_run_passed",
                                "gpu_initialized": False,
                                "validation_consumed": False,
                                "validation_hashed": False,
                                "environment": {"matches_frozen_environment": True},
                                "protocol_config": protocol,
                                **wrong_identity,
                            },
                        }
                    )
                )
                paths.append(path)
            with self.assertRaisesRegex(RuntimeError, "different current frozen"):
                aggregate_pg19_gate_shards(
                    paths,
                    expected_windows_sha256="7" * 64,
                    expected_frozen_identity=expected,
                )

    def test_pg19_aggregate_requires_rank_to_match_exact_window_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = frozen_identity("8" * 64)
            protocol = frozen_protocol_config()
            paths = []
            for rank in range(8):
                path = root / f"pg19-fair-v2-shard-{rank}.json"
                path.write_text(
                    json.dumps(
                        {
                            "status": "completed_pg19_fair_v2_gate_shard",
                            "passed": True,
                            "rank": rank,
                            "world_size": 8,
                            "fair_protocol": FAIR_PROTOCOL,
                            "kernel_mode": KERNEL_MODE,
                            "quantization": "Q16",
                            "single_request_only": True,
                            "static": {
                                **identity,
                                "status": "fair_v2_static_dry_run_passed",
                                "gpu_initialized": False,
                                "validation_consumed": False,
                                "validation_hashed": False,
                                "environment": {
                                    "matches_frozen_environment": True
                                },
                                "protocol_config": protocol,
                            },
                            "windows_sha256": identity["pg19_windows_sha256"],
                            "validation_consumed": False,
                            "validation_hashed": False,
                            "source_68_99_consumed": False,
                            "test_v2_consumed": False,
                            "rows": [
                                {
                                    "window_index": (rank + 1) % 8,
                                    "source_object": f"train/{rank}",
                                    "document_tail_tokens": 1,
                                }
                            ],
                        }
                    )
                )
                paths.append(path)
            with self.assertRaisesRegex(RuntimeError, "rank/window index"):
                aggregate_pg19_gate_shards(
                    paths,
                    expected_windows_sha256=identity["pg19_windows_sha256"],
                    expected_frozen_identity=identity,
                )

    def test_failed_authorization_stops_before_validation_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            authorization = root / "authorization.json"
            authorization.write_text(json.dumps({"status": "failed", "passed": False}))
            validation = root / "validation.jsonl"
            validation.write_text("do not read\n")
            args = SimpleNamespace(
                authorization=authorization,
                expected_authorization_sha256=sha256_file(authorization),
                expected_validation_sha256="b" * 64,
                validation_data=validation,
            )
            with mock.patch(
                "run_qcomem_qwen35_vllm_paged_fair_v2.sha256_file"
            ) as digest:
                with self.assertRaisesRegex(RuntimeError, "authorization"):
                    run_validation(args)
            digest.assert_not_called()

    def test_run_validation_passes_exact_limit_to_longbench_loader(self):
        class StopAfterCompatibilityCheck(Exception):
            pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation.jsonl"
            validation.write_text("frozen validation\n")
            frozen = {field: "c" * 64 for field in (
                "code_ledger_sha256",
                "model_manifest_sha256",
                "model_artifact_ledger_sha256",
                "model_weight_ledger_sha256",
                "pg19_data_sha256",
                "pg19_manifest_sha256",
                "pg19_windows_sha256",
                "validation_expected_sha256_recorded_but_not_hashed",
                "protocol_manifest_sha256",
                "protocol_config_sha256",
            )}
            protocol_config = {
                "bits": 16,
                "page_size": 128,
                "world_size": 8,
                "pg19_books": 8,
                "pg19_document_tokens": 1025,
                "pg19_query_tokens": 32,
                "pg19_window_stride": 257,
                "pg19_candidate_windows": 8,
                "pg19_seed": 20260814,
                "max_input_tokens": 4096,
                "max_query_tokens": 64,
                "max_new_tokens": 8,
                "source_index_start": 6,
                "source_index_end": 9,
                "limit_per_dataset": 4,
                "min_input_tokens": 1,
                "expected_source_revision": "5" * 40,
                "quantization": "Q16",
                "single_request_only": True,
                "batch_semantics": "batch-1-equal-length-only",
            }
            frozen["protocol_config_sha256"] = _protocol_config_sha256(
                protocol_config
            )
            authorization_value = {
                "status": "pg19_fair_v2_authorized",
                "passed": True,
                "fair_protocol": FAIR_PROTOCOL,
                "same_kernel_layout_gate_passed": True,
                "same_kernel_full_vocab_logit_gate_passed": True,
                "validation_consumed": False,
                "validation_hashed": False,
                "source_68_99_consumed": False,
                "test_v2_consumed": False,
                "frozen_identity": frozen,
                "protocol_config": protocol_config,
            }
            authorization = root / "authorization.json"
            authorization.write_text(json.dumps(authorization_value))
            expected_auth = sha256_file(authorization)
            expected_validation = sha256_file(validation)
            frozen["validation_expected_sha256_recorded_but_not_hashed"] = expected_validation
            authorization.write_text(json.dumps(authorization_value))
            expected_auth = sha256_file(authorization)
            static = {**frozen, "protocol_config": protocol_config}
            args = SimpleNamespace(
                authorization=authorization,
                expected_authorization_sha256=expected_auth,
                static_audit=static,
                expected_validation_sha256=expected_validation,
                validation_data=validation,
                model=root / "model",
                data=None,
                exclude_source_indices=(),
                allow_test_v2=False,
                context_lengths=(),
                synthetic_repetitions=0,
                source_index_start=6,
                source_index_end=9,
                limit_per_dataset=4,
                max_input_tokens=4096,
            )

            class FakeTokenizer:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    return object()

            fake_transformers = types.SimpleNamespace(
                AutoTokenizer=FakeTokenizer,
                AutoModelForImageTextToText=object,
            )

            def stop_loader(tokenizer, observed_args):
                self.assertIsNotNone(tokenizer)
                self.assertEqual(observed_args.limit_per_dataset, 4)
                self.assertEqual(observed_args.source_index_start, 6)
                self.assertEqual(observed_args.source_index_end, 9)
                raise StopAfterCompatibilityCheck

            with (
                mock.patch.dict(sys.modules, {"transformers": fake_transformers}),
                mock.patch(
                    "run_qcomem_qwen35_vllm_paged_fair_v2.longbench_workloads",
                    side_effect=stop_loader,
                ),
            ):
                with self.assertRaises(StopAfterCompatibilityCheck):
                    run_validation(args)

    def test_validation_rejects_test_v2_digest_before_hashing_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation.jsonl"
            validation.write_text("must not be hashed\n")
            protocol = {"bits": 16}
            frozen = {
                "code_ledger_sha256": "1" * 64,
                "model_manifest_sha256": "2" * 64,
                "model_artifact_ledger_sha256": "3" * 64,
                "model_weight_ledger_sha256": "4" * 64,
                "pg19_data_sha256": "5" * 64,
                "pg19_manifest_sha256": "6" * 64,
                "pg19_windows_sha256": "7" * 64,
                "validation_expected_sha256_recorded_but_not_hashed": TEST_V2_SHA256,
                "protocol_manifest_sha256": "9" * 64,
                "protocol_config_sha256": _protocol_config_sha256(protocol),
            }
            authorization_value = {
                "status": "pg19_fair_v2_authorized",
                "passed": True,
                "fair_protocol": FAIR_PROTOCOL,
                "same_kernel_layout_gate_passed": True,
                "same_kernel_full_vocab_logit_gate_passed": True,
                "validation_consumed": False,
                "validation_hashed": False,
                "source_68_99_consumed": False,
                "test_v2_consumed": False,
                "frozen_identity": frozen,
                "protocol_config": protocol,
            }
            authorization = root / "authorization.json"
            authorization.write_text(json.dumps(authorization_value))
            auth_sha = sha256_file(authorization)
            args = SimpleNamespace(
                authorization=authorization,
                expected_authorization_sha256=auth_sha,
                static_audit={**frozen, "protocol_config": protocol},
                expected_validation_sha256=TEST_V2_SHA256,
                validation_data=validation,
            )
            reads: list[Path] = []
            original_read_bytes = Path.read_bytes

            def recording_read_bytes(path):
                reads.append(Path(path))
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", recording_read_bytes):
                with self.assertRaisesRegex(RuntimeError, "test-v2 digest"):
                    run_validation(args)
            self.assertEqual(reads, [authorization])

    def test_validation_rejects_loader_reopen_digest_drift_before_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation = root / "validation.jsonl"
            validation.write_text("frozen validation\n")
            validation_sha = sha256_file(validation)
            auth = root / "authorization.json"
            auth.write_text(json.dumps(authorization_value(validation_sha)))
            args = SimpleNamespace(
                authorization=auth,
                expected_authorization_sha256=sha256_file(auth),
                static_audit={
                    **frozen_identity(validation_sha),
                    "protocol_config": frozen_protocol_config(),
                },
                expected_validation_sha256=validation_sha,
                validation_data=validation,
                model=root / "model",
                expected_source_revision="5" * 40,
                source_index_start=6,
                source_index_end=9,
                limit_per_dataset=4,
                max_input_tokens=4096,
            )

            class FakeTokenizer:
                @classmethod
                def from_pretrained(cls, *args, **kwargs):
                    return object()

            fake_transformers = types.SimpleNamespace(
                AutoTokenizer=FakeTokenizer,
                AutoModelForImageTextToText=object,
            )
            with (
                mock.patch.dict(sys.modules, {"transformers": fake_transformers}),
                mock.patch(
                    "run_qcomem_qwen35_vllm_paged_fair_v2.longbench_workloads",
                    return_value=([], {"data_sha256": "0" * 64}),
                ),
                mock.patch(
                    "run_qcomem_qwen35_vllm_paged_fair_v2.torch.cuda.set_device"
                ) as set_device,
            ):
                with self.assertRaisesRegex(RuntimeError, "reopened bytes"):
                    run_validation(args)
            set_device.assert_not_called()


class FairV2LauncherGovernanceTest(unittest.TestCase):
    def test_launcher_orders_authorization_before_validation_and_bounds_children(self):
        launcher = (
            Path(__file__).parent
            / "launch_qcomem_qwen35_vllm_paged_fair_v2_8gpu.sh"
        ).read_text()
        authorized = launcher.index('stages/03_pg19_authorized')
        validation_exists = launcher.index('test -s "$VALIDATION_DATA"')
        validation_hash = launcher.index(
            'verify_sha "$VALIDATION_DATA" "$EXPECTED_VALIDATION_SHA256"'
        )
        validation_stage = launcher.index("--stage validation")
        auth_sha = launcher.index("AUTHORIZATION_SHA256=")
        self.assertLess(authorized, validation_exists)
        self.assertLess(authorized, validation_hash)
        self.assertLess(auth_sha, validation_stage)
        self.assertEqual(launcher.count("timeout --signal=TERM --kill-after=60s 3600s"), 1)
        self.assertEqual(launcher.count("timeout --signal=TERM --kill-after=60s 7200s"), 1)
        self.assertIn("trap 'on_signal 130' INT", launcher)
        self.assertIn("trap 'on_signal 143' TERM", launcher)
        after_traps = launcher.split("trap 'on_signal 143' TERM", 1)[1]
        self.assertNotIn("exit 2", after_traps)
        self.assertEqual(after_traps.count("fail_stage \""), 2)
        self.assertEqual(
            launcher.count('--output "$RUN_DIR/static-dry-run.json"'), 1
        )
        self.assertIn("EXPECTED_PROTOCOL_MANIFEST_SHA256", launcher)
        self.assertNotIn("EXPECTED_YAML_SHA256", launcher)
        self.assertLess(
            launcher.index("scientific-artifact-integrity.log"),
            launcher.index('stages/99_done'),
        )

    def test_launcher_pins_c_locale_for_every_sorted_ledger(self):
        launcher_path = (
            Path(__file__).parent
            / "launch_qcomem_qwen35_vllm_paged_fair_v2_8gpu.sh"
        )
        launcher = launcher_path.read_text()
        self.assertIn("export LC_ALL=C", launcher)
        self.assertEqual(launcher.count("LC_ALL=C sort -z"), 2)
        self.assertNotRegex(launcher, r"(?<!LC_ALL=C )sort -z")

        locales = subprocess.run(
            ["locale", "-a"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        en_us = next(
            (value for value in locales if value.lower().startswith("en_us")),
            None,
        )
        self.assertIsNotNone(en_us, "locale regression requires an en_US locale")
        names = [
            "aggregate_interface_lora.py",
            "aggregate_interface.py",
            "aggregate-interface.py",
            "z.py",
        ]
        payload = b"\0".join(name.encode() for name in names) + b"\0"
        outputs = []
        for inherited_locale in ("C", en_us):
            env = os.environ.copy()
            env["LC_ALL"] = inherited_locale
            result = subprocess.run(
                ["bash", "-c", "LC_ALL=C sort -z"],
                input=payload,
                check=True,
                capture_output=True,
                env=env,
            )
            outputs.append(result.stdout)
        expected = b"\0".join(sorted(name.encode() for name in names)) + b"\0"
        self.assertEqual(outputs, [expected, expected])

    def test_preflight_digest_mismatch_records_failure_markers(self):
        launcher = (
            Path(__file__).parent
            / "launch_qcomem_qwen35_vllm_paged_fair_v2_8gpu.sh"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = root / "code"
            code.mkdir()
            model = root / "model"
            model.mkdir()
            pg19 = root / "pg19.jsonl"
            pg19.write_text("not-the-frozen-pg19\n")
            pg19_manifest = root / "pg19.manifest.json"
            pg19_manifest.write_text("{}\n")
            protocol_manifest = root / "protocol.json"
            protocol_manifest.write_text("{}\n")
            weights = root / "weights.sha256"
            weights.write_text("placeholder\n")
            run_dir = root / "run"
            env = os.environ.copy()
            env.update(
                {
                    "CODE_DIR": str(code),
                    "MODEL_DIR": str(model),
                    "MODEL_WEIGHT_LEDGER_FILE": str(weights),
                    "PG19_DATA": str(pg19),
                    "PG19_MANIFEST": str(pg19_manifest),
                    "VALIDATION_DATA": str(root / "must-not-be-read.jsonl"),
                    "PROTOCOL_MANIFEST_FILE": str(protocol_manifest),
                    "RUN_DIR": str(run_dir),
                    "ENV_DIR": str(root / "env"),
                    "EXPECTED_PG19_SHA256": "0" * 64,
                    "EXPECTED_PG19_MANIFEST_SHA256": "1" * 64,
                    "EXPECTED_PG19_WINDOWS_SHA256": "2" * 64,
                    "EXPECTED_VALIDATION_SHA256": "3" * 64,
                    "EXPECTED_SOURCE_REVISION": "4" * 40,
                    "EXPECTED_MODEL_MANIFEST_SHA256": "5" * 64,
                    "EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256": "6" * 64,
                    "EXPECTED_MODEL_WEIGHT_LEDGER_SHA256": "7" * 64,
                    "EXPECTED_CODE_LEDGER_SHA256": "8" * 64,
                    "EXPECTED_PROTOCOL_MANIFEST_SHA256": "9" * 64,
                }
            )
            result = subprocess.run(
                ["bash", str(launcher)],
                env=env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("pg19-train SHA256 mismatch", result.stderr)
            self.assertTrue((run_dir / "stages" / "FAILED").is_file())
            self.assertEqual(
                (run_dir / "stages" / "FAILED_PHASE").read_text().strip(),
                "preflight",
            )
            self.assertTrue(
                (run_dir / "stages" / "FAILED_preflight").is_file()
            )


class FairV2SyntheticAggregateTest(unittest.TestCase):
    def _write_shards(self, root: Path, validation_sha: str, auth_sha: str) -> None:
        shard_dir = root / "validation-shards"
        shard_dir.mkdir()
        pairs = [
            (dataset, source)
            for dataset in ("qasper", "2wikimqa")
            for source in range(6, 10)
        ]
        identity = frozen_identity(validation_sha)
        for rank, (dataset, source) in enumerate(pairs):
            order = (
                (FRESH_CONTROL, SHARED_REUSE, SHARED_REUSE, FRESH_CONTROL)
                if rank % 2 == 0
                else (SHARED_REUSE, FRESH_CONTROL, FRESH_CONTROL, SHARED_REUSE)
            ) * 2
            shard = {
                "status": "completed_fair_v2_validation_shard",
                "rank": rank,
                "world_size": 8,
                "fair_protocol": FAIR_PROTOCOL,
                "kernel_mode": KERNEL_MODE,
                "quantization": "Q16",
                "single_request_only": True,
                "static": {
                    **identity,
                    "protocol_config": frozen_protocol_config(),
                },
                "authorization_sha256": auth_sha,
                "workload": {"dataset": dataset, "source_index": source},
                "workload_metadata": {
                    "data_sha256": validation_sha,
                    "test_v2_consumed": False,
                    "source_revisions": ["5" * 40],
                },
                "warmup_runs_per_config": 1,
                "warmup_order": (
                    (FRESH_CONTROL, SHARED_REUSE)
                    if rank % 2 == 0
                    else (SHARED_REUSE, FRESH_CONTROL)
                ),
                "fresh_measurement_runs_per_config": 4,
                "measurement_order": order,
                "allocator_fresh_state": {
                    "gc_collect_before_empty_cache": True,
                    "dynamic_attention_backends_unregistered": True,
                    "baseline_exact_fields": [
                        "current_allocated_bytes",
                        "current_reserved_bytes",
                    ],
                    "frozen_post_warmup_baseline": {
                        "current_allocated_bytes": 10,
                        "current_reserved_bytes": 20,
                    },
                    "cleanup_after_each_measurement": [
                        {
                            "trial_index": trial_index,
                            "config": config,
                            "after": {
                                "current_allocated_bytes": 10,
                                "current_reserved_bytes": 20,
                            },
                        }
                        for trial_index, config in enumerate(order)
                    ],
                },
                "measurements": {
                    FRESH_CONTROL: fake_measurement(FRESH_CONTROL),
                    SHARED_REUSE: fake_measurement(SHARED_REUSE),
                },
                "hf_eager_absolute_reference": {
                    "generated_token_ids": [1] * 8,
                    "full_vocab_step_logit_sha256": ["a" * 64] * 8,
                },
                "primary_pair": {
                    "full_vocab_step_logits_bitwise_exact": True,
                    "generated_tokens_exact": True,
                    "cached_document_request_ttft_excludes_common_document_build": True,
                    "isolated_kernel_latency_measured": False,
                    "cached_document_request_ttft_ratio_reuse_vs_full_copy": 0.5,
                    "continuation_model_first_token_ratio_reuse_vs_full_copy": 0.5,
                    "tpot_ratio_reuse_vs_full_copy": 0.5,
                    "cuda_peak_ratio_reuse_vs_full_copy": None,
                    "cuda_peak_reserved_ratio_reuse_vs_full_copy": None,
                    "setup_peak_allocated_ratio_reuse_vs_full_copy": None,
                    "setup_peak_reserved_ratio_reuse_vs_full_copy": None,
                    "physical_document_block_copy_bytes_saved_including_padding": 11520,
                },
                "backend_compatibility_nonblocking": {
                    "is_primary_performance_pair": False,
                    "hf_generated_tokens_match_vllm": True,
                    "hf_full_vocab_step_logit_sha_match_vllm": True,
                },
                "validation_consumed_after_pg19_authorization": True,
                "source_68_99_consumed": False,
                "test_v2_consumed": False,
            }
            (shard_dir / f"fair-v2-shard-{rank}.json").write_text(
                json.dumps(shard)
            )

    def _summarize_fixture(
        self,
        root: Path,
        auth: Path,
        auth_sha: str,
        identity: dict[str, str],
    ) -> dict:
        return summarize_validation_shards(
            root,
            authorization_path=auth,
            expected_authorization_sha256=auth_sha,
            expected_frozen_identity=identity,
            expected_code_ledger_sha256=identity["code_ledger_sha256"],
            expected_model_manifest_sha256=identity["model_manifest_sha256"],
            expected_model_artifact_ledger_sha256=identity[
                "model_artifact_ledger_sha256"
            ],
            expected_model_weight_ledger_sha256=identity[
                "model_weight_ledger_sha256"
            ],
            expected_source_revision="5" * 40,
            expected_calls_per_layer=8,
        )

    def test_eight_shard_summary_preserves_zero_ratio_and_storage_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_sha = "8" * 64
            identity = frozen_identity(validation_sha)
            auth = root / "authorization.json"
            auth.write_text(json.dumps(authorization_value(validation_sha)))
            auth_sha = sha256_file(auth)
            self._write_shards(root, validation_sha, auth_sha)
            summary = summarize_validation_shards(
                root,
                authorization_path=auth,
                expected_authorization_sha256=auth_sha,
                expected_frozen_identity=identity,
                expected_code_ledger_sha256=identity["code_ledger_sha256"],
                expected_model_manifest_sha256=identity["model_manifest_sha256"],
                expected_model_artifact_ledger_sha256=identity[
                    "model_artifact_ledger_sha256"
                ],
                expected_model_weight_ledger_sha256=identity[
                    "model_weight_ledger_sha256"
                ],
                expected_source_revision="5" * 40,
                expected_calls_per_layer=8,
            )
            self.assertEqual(
                summary["cuda_peak_allocated_ratio_reuse_vs_full_copy"],
                {
                    "defined_count": 0,
                    "undefined_zero_baseline_count": 8,
                    "median_when_defined": None,
                },
            )
            self.assertEqual(
                summary[
                    "cached_document_request_ttft_ratio_reuse_vs_full_copy"
                ],
                {
                    "defined_count": 8,
                    "undefined_zero_baseline_count": 0,
                    "median_when_defined": 0.5,
                },
            )
            self.assertEqual(
                summary[
                    "median_physical_document_block_copy_bytes_saved_including_padding"
                ],
                11520,
            )
            self.assertIn(
                "median_fresh_combined_unique_accelerator_nbytes_before_continuation",
                summary,
            )
            self.assertIn(
                "median_fresh_combined_unique_accelerator_nbytes_after_decode",
                summary,
            )
            storage = summary["full_attention_storage_medians"]
            self.assertEqual(
                storage["common_source_document_table_accelerator_nbytes"], 40
            )
            absolute = summary[
                "allocator_current_peak_absolute_and_delta_medians"
            ]
            self.assertEqual(
                absolute["common_dense_document_prefill_after"]["sample_count"],
                64,
            )
            self.assertEqual(
                absolute["fresh_request_setup_after"]["sample_count"], 32
            )
            self.assertEqual(
                absolute["fresh_setup_plus_first_step_after"]["sample_count"],
                32,
            )

            tampered_path = root / "validation-shards" / "fair-v2-shard-0.json"
            tampered = json.loads(tampered_path.read_text())
            pristine = copy.deepcopy(tampered)

            ttft_tampered = copy.deepcopy(pristine)
            reuse = ttft_tampered["measurements"][SHARED_REUSE]
            for trial in reuse["fresh_trials"]:
                trial["cached_document_request_ttft_seconds"] = 3.0
            ttft_tampered["measurements"][SHARED_REUSE] = _median_trials(
                SHARED_REUSE, reuse["fresh_trials"]
            )
            ttft_tampered["primary_pair"][
                "cached_document_request_ttft_ratio_reuse_vs_full_copy"
            ] = 1.5
            tampered_path.write_text(json.dumps(ttft_tampered))
            with self.assertRaisesRegex(RuntimeError, "TTFT is not setup"):
                self._summarize_fixture(root, auth, auth_sha, identity)

            allocator_tampered = copy.deepcopy(pristine)
            reuse = allocator_tampered["measurements"][SHARED_REUSE]
            for trial in reuse["fresh_trials"]:
                trial["cuda_current_request_delta_bytes"] = 999
            allocator_tampered["measurements"][SHARED_REUSE] = _median_trials(
                SHARED_REUSE, reuse["fresh_trials"]
            )
            tampered_path.write_text(json.dumps(allocator_tampered))
            with self.assertRaisesRegex(RuntimeError, "derived allocator scalar"):
                self._summarize_fixture(root, auth, auth_sha, identity)

            copy_tampered = copy.deepcopy(pristine)
            fresh = copy_tampered["measurements"][FRESH_CONTROL]
            for trial in fresh["fresh_trials"]:
                trial["query_preparation"][
                    "physical_document_block_copy_nbytes_including_padding"
                ] = 999
            copy_tampered["measurements"][FRESH_CONTROL] = _median_trials(
                FRESH_CONTROL, fresh["fresh_trials"]
            )
            copy_tampered["primary_pair"][
                "physical_document_block_copy_bytes_saved_including_padding"
            ] = 999
            tampered_path.write_text(json.dumps(copy_tampered))
            with self.assertRaisesRegex(RuntimeError, "physical copy differs"):
                self._summarize_fixture(root, auth, auth_sha, identity)

            call_identity_tampered = copy.deepcopy(pristine)
            call_identity_tampered["measurements"][SHARED_REUSE][
                "fresh_trials"
            ][1]["intercept"]["calls"][0]["kernel_identity"][
                "callable_id"
            ] = 999
            tampered_path.write_text(json.dumps(call_identity_tampered))
            with self.assertRaisesRegex(RuntimeError, "call kernel identity"):
                self._summarize_fixture(root, auth, auth_sha, identity)

            call_layer_tampered = copy.deepcopy(pristine)
            call_layer_tampered["measurements"][SHARED_REUSE]["fresh_trials"][
                1
            ]["intercept"]["calls"][0]["layer_idx"] = 39
            tampered_path.write_text(json.dumps(call_layer_tampered))
            with self.assertRaisesRegex(RuntimeError, "call-layer"):
                self._summarize_fixture(root, auth, auth_sha, identity)

            tampered_path.write_text(json.dumps(pristine))
            tampered = copy.deepcopy(pristine)
            cleanup_tampered = copy.deepcopy(tampered)
            cleanup_tampered["allocator_fresh_state"][
                "cleanup_after_each_measurement"
            ][0]["config"] = SHARED_REUSE
            tampered_path.write_text(json.dumps(cleanup_tampered))
            with self.assertRaisesRegex(RuntimeError, "cleanup row"):
                summarize_validation_shards(
                    root,
                    authorization_path=auth,
                    expected_authorization_sha256=auth_sha,
                    expected_frozen_identity=identity,
                    expected_code_ledger_sha256=identity[
                        "code_ledger_sha256"
                    ],
                    expected_model_manifest_sha256=identity[
                        "model_manifest_sha256"
                    ],
                    expected_model_artifact_ledger_sha256=identity[
                        "model_artifact_ledger_sha256"
                    ],
                    expected_model_weight_ledger_sha256=identity[
                        "model_weight_ledger_sha256"
                    ],
                    expected_source_revision="5" * 40,
                    expected_calls_per_layer=8,
                )
            tampered_path.write_text(json.dumps(tampered))
            tampered["measurements"][SHARED_REUSE][
                "cached_document_request_ttft_seconds"
            ] = 198.0
            tampered["primary_pair"][
                "cached_document_request_ttft_ratio_reuse_vs_full_copy"
            ] = 99.0
            tampered_path.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(
                RuntimeError, "reported .* measurement differs from raw-trial replay"
            ):
                summarize_validation_shards(
                    root,
                    authorization_path=auth,
                    expected_authorization_sha256=auth_sha,
                    expected_frozen_identity=identity,
                    expected_code_ledger_sha256=identity[
                        "code_ledger_sha256"
                    ],
                    expected_model_manifest_sha256=identity[
                        "model_manifest_sha256"
                    ],
                    expected_model_artifact_ledger_sha256=identity[
                        "model_artifact_ledger_sha256"
                    ],
                    expected_model_weight_ledger_sha256=identity[
                        "model_weight_ledger_sha256"
                    ],
                    expected_source_revision="5" * 40,
                    expected_calls_per_layer=8,
                )

            drifted = dict(identity)
            drifted["protocol_manifest_sha256"] = "a" * 64
            with self.assertRaisesRegex(RuntimeError, "different current frozen"):
                summarize_validation_shards(
                    root,
                    authorization_path=auth,
                    expected_authorization_sha256=auth_sha,
                    expected_frozen_identity=drifted,
                    expected_code_ledger_sha256=identity[
                        "code_ledger_sha256"
                    ],
                    expected_model_manifest_sha256=identity[
                        "model_manifest_sha256"
                    ],
                    expected_model_artifact_ledger_sha256=identity[
                        "model_artifact_ledger_sha256"
                    ],
                    expected_model_weight_ledger_sha256=identity[
                        "model_weight_ledger_sha256"
                    ],
                    expected_source_revision="5" * 40,
                    expected_calls_per_layer=8,
                )

            bad_authorization = authorization_value(validation_sha)
            bad_authorization["protocol_config"]["max_new_tokens"] = 7
            auth.write_text(json.dumps(bad_authorization))
            bad_auth_sha = sha256_file(auth)
            with self.assertRaisesRegex(RuntimeError, "protocol config SHA drift"):
                summarize_validation_shards(
                    root,
                    authorization_path=auth,
                    expected_authorization_sha256=bad_auth_sha,
                    expected_frozen_identity=identity,
                    expected_code_ledger_sha256=identity[
                        "code_ledger_sha256"
                    ],
                    expected_model_manifest_sha256=identity[
                        "model_manifest_sha256"
                    ],
                    expected_model_artifact_ledger_sha256=identity[
                        "model_artifact_ledger_sha256"
                    ],
                    expected_model_weight_ledger_sha256=identity[
                        "model_weight_ledger_sha256"
                    ],
                    expected_source_revision="5" * 40,
                    expected_calls_per_layer=8,
                )

    def test_missing_schemas_and_self_consistent_storage_tampering_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            validation_sha = "8" * 64
            identity = frozen_identity(validation_sha)
            auth = root / "authorization.json"
            auth.write_text(json.dumps(authorization_value(validation_sha)))
            auth_sha = sha256_file(auth)
            self._write_shards(root, validation_sha, auth_sha)
            tampered_path = (
                root / "validation-shards" / "fair-v2-shard-0.json"
            )
            pristine = json.loads(tampered_path.read_text())

            def expect_failure(value: dict, pattern: str) -> None:
                tampered_path.write_text(json.dumps(value))
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self._summarize_fixture(
                        root, auth, auth_sha, identity
                    )

            def replay_arm(value: dict, policy: str) -> None:
                measurement = value["measurements"][policy]
                value["measurements"][policy] = _median_trials(
                    policy, measurement["fresh_trials"]
                )

            missing_frozen = copy.deepcopy(pristine)
            missing_frozen["allocator_fresh_state"][
                "frozen_post_warmup_baseline"
            ].pop("current_allocated_bytes")
            expect_failure(missing_frozen, "frozen allocator baseline")

            missing_cleanup = copy.deepcopy(pristine)
            missing_cleanup["allocator_fresh_state"][
                "cleanup_after_each_measurement"
            ][0]["after"].pop("current_reserved_bytes")
            expect_failure(missing_cleanup, "cleanup allocator")

            for baseline_key in (
                "allocator_fresh_trial_baseline",
                "abba_allocator_baseline",
            ):
                missing_trial_baseline = copy.deepcopy(pristine)
                trials = missing_trial_baseline["measurements"][SHARED_REUSE][
                    "fresh_trials"
                ]
                for trial in trials:
                    trial[baseline_key].pop("current_allocated_bytes")
                replay_arm(missing_trial_baseline, SHARED_REUSE)
                expect_failure(missing_trial_baseline, baseline_key)

            missing_first_step = copy.deepcopy(pristine)
            for trial in missing_first_step["measurements"][SHARED_REUSE][
                "fresh_trials"
            ]:
                trial.pop("allocator_after_first_continuation_step")
            replay_arm(missing_first_step, SHARED_REUSE)
            expect_failure(
                missing_first_step, "setup_plus_first_step_after allocator snapshots"
            )

            empty_identity = copy.deepcopy(pristine)
            trial = empty_identity["measurements"][SHARED_REUSE][
                "fresh_trials"
            ][1]
            trial["kernel_identity"] = {}
            trial["intercept"]["kernel_identity"] = {}
            for call in trial["intercept"]["calls"]:
                call["kernel_identity"] = {}
            expect_failure(empty_identity, "kernel identity|callable_id")

            missing_append = copy.deepcopy(pristine)
            call = missing_append["measurements"][SHARED_REUSE][
                "fresh_trials"
            ][1]["intercept"]["calls"][0]
            call.pop("current_append_delta_tokens")
            call.pop("query_tokens")
            expect_failure(missing_append, "current_append_delta_tokens")

            empty_trajectory = copy.deepcopy(pristine)
            for trial in empty_trajectory["measurements"][SHARED_REUSE][
                "fresh_trials"
            ]:
                trial["generated_token_ids"] = []
                trial["full_vocab_step_logit_sha256"] = []
            replay_arm(empty_trajectory, SHARED_REUSE)
            expect_failure(empty_trajectory, "trajectory cardinality")

            source_total = copy.deepcopy(pristine)
            for trial in source_total["measurements"][SHARED_REUSE][
                "fresh_trials"
            ]:
                for phase in (
                    "full_attention_storage_before_decode",
                    "full_attention_storage_after_decode",
                ):
                    storage = trial[phase]
                    for row in storage["layers"]:
                        row["source_total_arena_allocated_nbytes"] = 999
                    refresh_storage_totals(storage)
            replay_arm(source_total, SHARED_REUSE)
            expect_failure(source_total, "source-pool|source payload|arena formula")

            padding = copy.deepcopy(pristine)
            for trial in padding["measurements"][FRESH_CONTROL]["fresh_trials"]:
                for phase in (
                    "full_attention_storage_before_decode",
                    "full_attention_storage_after_decode",
                ):
                    storage = trial[phase]
                    for row in storage["layers"]:
                        row["source_document_padding_nbytes"] = 777
                        row["fresh_duplicate_document_padding_nbytes"] = 777
                    refresh_storage_totals(storage)
            replay_arm(padding, FRESH_CONTROL)
            expect_failure(padding, "payload/padding|padding total")

            active_payload = copy.deepcopy(pristine)
            for trial in active_payload["measurements"][SHARED_REUSE][
                "fresh_trials"
            ]:
                storage = trial["full_attention_storage_after_decode"]
                for row in storage["layers"]:
                    row["active_request_private_payload_nbytes"] = 7
                refresh_storage_totals(storage)
            replay_arm(active_payload, SHARED_REUSE)
            expect_failure(active_payload, "active-private")

            synchronized_copy = copy.deepcopy(pristine)
            fresh = synchronized_copy["measurements"][FRESH_CONTROL]
            for trial in fresh["fresh_trials"]:
                trial["query_preparation"][
                    "physical_document_block_copy_nbytes_including_padding"
                ] = 10
                trial["query_preparation"]["audit"][
                    "full_document_staging_copy_nbytes"
                ] = 10
            replay_arm(synchronized_copy, FRESH_CONTROL)
            synchronized_copy["primary_pair"][
                "physical_document_block_copy_bytes_saved_including_padding"
            ] = 10
            expect_failure(synchronized_copy, "physical copy differs from storage")

            synchronized_tail = copy.deepcopy(pristine)
            for policy in (FRESH_CONTROL, SHARED_REUSE):
                for trial in synchronized_tail["measurements"][policy][
                    "fresh_trials"
                ]:
                    trial["continuation_append_accounting"][
                        "partial_tail_staging_copy_nbytes"
                    ] = 999
                replay_arm(synchronized_tail, policy)
            expect_failure(synchronized_tail, "partial-tail staging copy")

            hf_flag = copy.deepcopy(pristine)
            hf_flag["backend_compatibility_nonblocking"][
                "hf_generated_tokens_match_vllm"
            ] = False
            expect_failure(hf_flag, "backend compatibility differs")

    def test_pg19_semantic_hash_schema_rejects_missing_equal_values(self):
        with self.assertRaisesRegex(RuntimeError, "64-hex"):
            _validate_sha256_hex(None, "PG19 fresh logits")


class FairV2AbbaTest(unittest.TestCase):
    def test_abba_uses_two_warmups_eight_measurements_and_exact_cleanup(self):
        baseline = {
            "label": "mock",
            "current_allocated_bytes": 100,
            "current_reserved_bytes": 200,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        }

        def measured(*args):
            return {
                "config": args[-1],
                "generated_token_ids": [7, 8],
                "full_vocab_step_logit_sha256": ["a" * 64, "b" * 64],
            }

        with (
            mock.patch(
                "run_qcomem_qwen35_vllm_paged_fair_v2._measure_same_kernel_config",
                side_effect=measured,
            ) as measure,
            mock.patch(
                "run_qcomem_qwen35_vllm_paged_fair_v2._fresh_allocator_cleanup",
                return_value=baseline,
            ) as cleanup,
        ):
            warmup, order, trials, allocator = _run_fresh_abba(
                SimpleNamespace(rank=0),
                object(),
                object(),
                object(),
                object(),
                object(),
                object(),
            )
        self.assertEqual(warmup, (FRESH_CONTROL, SHARED_REUSE))
        self.assertEqual(len(order), 8)
        self.assertEqual(measure.call_count, 10)
        self.assertEqual(cleanup.call_count, 21)
        self.assertEqual(len(trials[FRESH_CONTROL]), 4)
        self.assertEqual(len(trials[SHARED_REUSE]), 4)
        self.assertTrue(allocator["gc_collect_before_empty_cache"])
        self.assertTrue(allocator["dynamic_attention_backends_unregistered"])

    def test_rank_one_reverses_warmup_and_uses_baab_order(self):
        baseline = {
            "current_allocated_bytes": 1,
            "current_reserved_bytes": 2,
            "peak_allocated_bytes": 1,
            "peak_reserved_bytes": 2,
        }

        def measured(*args):
            return {
                "config": args[-1],
                "generated_token_ids": [1, 2],
                "full_vocab_step_logit_sha256": ["1" * 64, "2" * 64],
            }

        with (
            mock.patch(
                "run_qcomem_qwen35_vllm_paged_fair_v2._measure_same_kernel_config",
                side_effect=measured,
            ),
            mock.patch(
                "run_qcomem_qwen35_vllm_paged_fair_v2._fresh_allocator_cleanup",
                return_value=baseline,
            ),
        ):
            warmup, order, _, _ = _run_fresh_abba(
                SimpleNamespace(rank=1), *(object() for _ in range(6))
            )
        self.assertEqual(warmup, (SHARED_REUSE, FRESH_CONTROL))
        self.assertEqual(
            order,
            (SHARED_REUSE, FRESH_CONTROL, FRESH_CONTROL, SHARED_REUSE) * 2,
        )

    def test_allocator_baseline_drift_fails_before_measurement(self):
        stable = {
            "current_allocated_bytes": 10,
            "current_reserved_bytes": 20,
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 20,
        }
        drift = dict(stable)
        drift["current_reserved_bytes"] = 21
        cleanup_values = [stable] * 5 + [drift]

        def measured(*args):
            return {
                "config": args[-1],
                "generated_token_ids": [1, 2],
                "full_vocab_step_logit_sha256": ["1" * 64, "2" * 64],
            }

        with (
            mock.patch(
                "run_qcomem_qwen35_vllm_paged_fair_v2._measure_same_kernel_config",
                side_effect=measured,
            ) as measure,
            mock.patch(
                "run_qcomem_qwen35_vllm_paged_fair_v2._fresh_allocator_cleanup",
                side_effect=cleanup_values,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "baseline drifted"):
                _run_fresh_abba(
                    SimpleNamespace(rank=0), *(object() for _ in range(6))
                )
        self.assertEqual(measure.call_count, 2)


if __name__ == "__main__":
    unittest.main()
