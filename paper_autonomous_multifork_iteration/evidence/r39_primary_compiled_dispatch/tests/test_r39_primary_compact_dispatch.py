from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import r39_compiled_dispatch_receipts as base
from r39_primary_compact_dispatch import (
    Geometry,
    PRIMARY_ARMS,
    PrimaryDispatchError,
    expected_cells,
    expected_rank_counts,
    verify_payload,
    verify_primary_shard,
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
        attention.extend(
            [[cell_index, local, 0, 0, 0, 0] for local in range(2)]
        )
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
        "schema_version": "forkaudit-r39-primary-compiled-dispatch-receipt-v1",
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
            "call_shapes": [verbose["attention_calls"][0]["call_shape"]],
            "autotune_observations": [{"mode": "no-autotuner-observed"}],
        },
        "cells": cells,
        "attention_call_columns": [],
        "attention_calls": attention,
        "gdn_call_columns": [],
        "gdn_calls": gdn,
    }
    return cache, code, runtime, payload


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
                )

    def test_primary_shard_ledger_binding(self) -> None:
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
                                }
                            ],
                            "witness_kernel_ledgers": [
                                {
                                    "request_index": 0,
                                    "total_calls": 2,
                                    "verified": True,
                                    "dense_fallback_calls": 0,
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


if __name__ == "__main__":
    unittest.main()
