from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import r39_compiled_dispatch_receipts as base
from r39_primary_compact_dispatch import (
    ATTENTION_COLUMNS,
    GDN_COLUMNS,
    Geometry,
    PRIMARY_ARMS,
    PrimaryDispatchError,
    expected_cells,
    expected_rank_counts,
    verify_payload,
    verify_primary_shard,
    _shape_record,
)


TINY = Geometry(
    resident_counts=(1,),
    arms=("kv=tiny-kv|gdn=tiny-gdn",),
    roles=("formal_memory", "ownership_witness"),
    generation_steps=2,
    document_tokens=3,
    full_layers=(1,),
    linear_layers=(0,),
)


def build_tiny(root: Path):
    cache, code, runtime, verbose = base._demo_fixture(root)
    artifact = verbose["attention_calls"][0]["selected_compiled_artifact"]
    config = verbose["attention_calls"][0]["selected_compile_config"]
    shape_round_0 = {
        "q": [32, 16, 256],
        "k": [34, 128, 2, 256],
        "v": [34, 128, 2, 256],
        "out": [32, 16, 256],
        "block_table": [1, 33],
        "max_seqlen_q": 32,
        "max_seqlen_k": 4127,
        "softmax_scale": 0.0625,
    }
    shape_round_1 = {
        "q": [1, 16, 256],
        "k": [34, 128, 2, 256],
        "v": [34, 128, 2, 256],
        "out": [1, 16, 256],
        "block_table": [1, 33],
        "max_seqlen_q": 1,
        "max_seqlen_k": 4128,
        "softmax_scale": 0.0625,
    }
    cells = []
    attention = []
    gdn = []
    attention_cursor = 0
    gdn_cursor = 0
    for cell_index, metadata in enumerate(expected_cells(TINY)):
        cells.append(
            {
                "cell_index": cell_index,
                "rank": 0,
                **metadata,
                "attention_call_range": [attention_cursor, attention_cursor + 2],
                "gdn_call_range": [gdn_cursor, gdn_cursor + 3],
                "expected_attention_calls": 2,
                "expected_gdn_document_prefill_calls": 1,
                "expected_gdn_request_calls": 2,
            }
        )
        attention.extend([[cell_index, 0, 0, 0, 0, 0], [cell_index, 1, 1, 0, 0, 0]])
        gdn.extend(
            [
                [cell_index, 0, 0, 3, False, 1, 1, 1],
                [cell_index, 1, 0, 32, True, 1, 1, 1],
                [cell_index, 2, 0, 1, True, 1, 1, 1],
            ]
        )
        attention_cursor += 2
        gdn_cursor += 3
    counts = expected_rank_counts(TINY)
    payload = {
        "schema_version": "forkaudit-r39-primary-compiled-dispatch-receipt-v2",
        "rank": 0,
        "scope": {
            "primary_protocol": "qcomem-qwen35-forkaudit-review-revision-v1",
            "primary_cells_only": True,
            "vllm_attention": base.TARGET_VLLM_ENTRYPOINT,
            "gdn": verbose["scope"]["gdn"],
        },
        "hook_installation": verbose["hook_installation"],
        "gdn_source_bindings": verbose["gdn_source_bindings"],
        "geometry": {
            "world_size": 8,
            "resident_counts": [1],
            "generation_steps": 2,
            "document_tokens": 3,
            "full_layer_indices": [1],
            "linear_layer_indices": [0],
            "arm_ids": ["kv=tiny-kv|gdn=tiny-gdn"],
            "cell_roles": ["formal_memory", "ownership_witness"],
            **counts,
        },
        "tables": {
            "compiled_artifacts": [artifact],
            "selected_compile_configurations": [config],
            "call_shapes": [_shape_record(shape_round_0), _shape_record(shape_round_1)],
            "autotune_observations": [{"mode": "no-autotuner-observed"}],
        },
        "cells": cells,
        "attention_call_columns": ATTENTION_COLUMNS,
        "attention_calls": attention,
        "gdn_call_columns": GDN_COLUMNS,
        "gdn_calls": gdn,
        "execution_binding": {
            "runner_relative_path": "run_qcomem_qwen35_forkaudit_review_revision.py",
            "runner_sha256": "9da619fc037e2c670b146d778fd9f4d5344212b7e525f3d3f26a077f79d67775",
            "runner_argv": ["--stage", "shard", "--rank", "0", "--output", "/frozen/forkaudit-shard-0.json"],
            "runner_argv_sha256": "799243df111d4b163a0a848fcfef7c7982916424bf828225a0eee27ff52fab61",
            "primary_shard_path": "/frozen/forkaudit-shard-0.json",
            "primary_shard_sha256": "a" * 64,
        },
    }
    return cache, code, runtime, payload


def fixture_bindings(payload):
    return copy.deepcopy(payload["gdn_source_bindings"])


class PrimaryCompactDispatchTests(unittest.TestCase):
    def test_frozen_primary_counts(self) -> None:
        self.assertEqual(len(PRIMARY_ARMS), 4)
        self.assertEqual(
            expected_rank_counts(),
            {
                "cell_count": 24,
                "attention_call_count": 26240,
                "gdn_document_prefill_call_count": 720,
                "gdn_request_call_count": 78720,
                "gdn_call_count": 79440,
            },
        )

    def test_tiny_compact_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            result = verify_payload(
                payload,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                geometry=TINY,
                expected_rank=0,
                expected_gdn_bindings=fixture_bindings(payload),
            )
            self.assertEqual(result["replay_verdict"], "pass")
            self.assertEqual(result["attention_call_count"], 4)
            self.assertEqual(result["gdn_call_count"], 6)

    def test_missing_call_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            candidate = copy.deepcopy(payload)
            candidate["attention_calls"].pop()
            with self.assertRaises(PrimaryDispatchError):
                verify_payload(
                    candidate,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                    geometry=TINY,
                    expected_rank=0,
                    expected_gdn_bindings=fixture_bindings(payload),
                )

    def test_artifact_reference_tamper_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            candidate = copy.deepcopy(payload)
            candidate["tables"]["compiled_artifacts"][0]["artifact_id"] = "0" * 64
            with self.assertRaises(base.DispatchReceiptError):
                verify_payload(
                    candidate,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                    geometry=TINY,
                    expected_rank=0,
                    expected_gdn_bindings=fixture_bindings(payload),
                )

    def test_cross_key_callable_substitution_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            frozen = fixture_bindings(payload)
            candidate = copy.deepcopy(payload)
            candidate["gdn_source_bindings"]["transformers_accelerate_wrapper"] = copy.deepcopy(
                candidate["gdn_source_bindings"][
                    "transformers_qwen35_moe_torch_chunk_gated_delta_rule"
                ]
            )
            with self.assertRaises(PrimaryDispatchError):
                verify_payload(
                    candidate,
                    cache_root=cache,
                    code_root=code,
                    runtime_root=runtime,
                    geometry=TINY,
                    expected_rank=0,
                    expected_gdn_bindings=frozen,
                )

    def test_scope_column_shape_and_rank_substitutions_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            frozen = fixture_bindings(payload)
            candidates = []
            candidate = copy.deepcopy(payload)
            candidate["scope"]["gdn"] = "compiled-superkernel-attested"
            candidates.append(candidate)
            candidate = copy.deepcopy(payload)
            candidate["attention_call_columns"] = ["lies"]
            candidates.append(candidate)
            candidate = copy.deepcopy(payload)
            candidate["tables"]["call_shapes"][0]["shape"]["q"] = [999, 999]
            candidates.append(candidate)
            candidate = copy.deepcopy(payload)
            candidate["rank"] = 1
            candidates.append(candidate)
            for candidate in candidates:
                with self.assertRaises(PrimaryDispatchError):
                    verify_payload(
                        candidate,
                        cache_root=cache,
                        code_root=code,
                        runtime_root=runtime,
                        geometry=TINY,
                        expected_rank=0,
                        expected_gdn_bindings=frozen,
                    )

    def test_primary_shard_ledger_binding(self) -> None:
        calls = [
            {
                "request_index": 0,
                "layer_idx": 1,
                "query_tokens": 32,
                "physical_block_pool_shape": [34, 128, 2, 256],
                "active_block_table_shape": [1, 33],
                "kv_tokens": 4127,
                "softmax_scale": 0.0625,
            },
            {
                "request_index": 0,
                "layer_idx": 1,
                "query_tokens": 1,
                "physical_block_pool_shape": [34, 128, 2, 256],
                "active_block_table_shape": [1, 33],
                "kv_tokens": 4128,
                "softmax_scale": 0.0625,
            },
        ]
        shard = {
            "schema_version": "qcomem-forkaudit-review-shard-v1",
            "protocol": "qcomem-qwen35-forkaudit-review-revision-v1",
            "status": "completed_formal_gpu_shard",
            "rank": 0,
            "world_size": 8,
            "protocol_config": {
                "resident_counts": [1],
                "generation_steps": 2,
                "document_tokens": 3,
                "factorial_arm_ids": ["kv=tiny-kv|gdn=tiny-gdn"],
            },
            "factorial": [
                {
                    "resident_count": 1,
                    "cells": [
                        {
                            "arm_id": "kv=tiny-kv|gdn=tiny-gdn",
                            "memory_kernel_ledgers": [
                                {
                                    "request_index": 0,
                                    "total_calls": 2,
                                    "verified": True,
                                    "dense_fallback_calls": 0,
                                    "calls": copy.deepcopy(calls),
                                }
                            ],
                            "witness_kernel_ledgers": [
                                {
                                    "request_index": 0,
                                    "total_calls": 2,
                                    "verified": True,
                                    "dense_fallback_calls": 0,
                                    "calls": copy.deepcopy(calls),
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        result = verify_primary_shard(shard, expected_rank=0, geometry=TINY)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["attention_ledger_call_count"], 4)
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            verify_payload(
                payload,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                geometry=TINY,
                expected_rank=0,
                expected_gdn_bindings=fixture_bindings(payload),
            )
            candidate = copy.deepcopy(payload)
            record = candidate["tables"]["call_shapes"][0]
            record["shape"]["max_seqlen_k"] = 4128
            record["shape_sha256"] = base._sha256_bytes(
                base._canonical_bytes(record["shape"])
            )
            # The altered row remains structurally valid and self-consistent,
            # but it no longer describes the immutable runner's actual call.
            verify_payload(
                candidate,
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                geometry=TINY,
                expected_rank=0,
                expected_gdn_bindings=fixture_bindings(payload),
            )
            with self.assertRaises(PrimaryDispatchError):
                verify_primary_shard(
                    shard,
                    expected_rank=0,
                    geometry=TINY,
                    receipt=candidate,
                )


if __name__ == "__main__":
    unittest.main()
