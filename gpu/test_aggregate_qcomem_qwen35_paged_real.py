from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from aggregate_qcomem_qwen35_paged_real import AggregateError, aggregate
from qcomem_qwen35_paged_integration import KERNEL_MODE


DATA_SHA = "a" * 64
REVISION = "b" * 40


def measurement(name: str, *, payload: int | None = None) -> dict:
    paged = name.startswith("paged-")
    value = {
        "config": name,
        "kernel_mode": (
            KERNEL_MODE
            if paged
            else "transformers-eager-native-qwen-functional-linear-rebind"
        ),
        "production_ttft_optimization_claim_allowed": False,
        "queries_per_document": 2,
        "warmup_count": 1,
        "per_query": [
            {"generated_token_ids": [1, 2]},
            {"generated_token_ids": [1, 2]},
        ],
        "persistent_total_resident_nbytes": 500 if paged else 800,
        "query_private_nbytes": 20,
        "multi_query_active_total_resident_nbytes": 540 if paged else 850,
        "cuda_peak_allocated_bytes": 1000,
        "nvml_sampled_peak_process_bytes": 1200,
        "ttft_seconds": 0.5,
        "median_tpot_seconds": 0.2,
        "persistent_gdn_base_immutable": True,
        "query_linear_rebind": {
            "verified": True,
            "fallback_layers": [],
            "request_count": 2,
        },
        "full_document_staging_copy_nbytes": 0,
        "auditable_corpus_capacity_documents": 10,
        "auditable_corpus_capacity_with_active_queries": 9,
    }
    if paged:
        dense = 400
        value.update(
            {
                "persistent_full_pages_shared": True,
                "persistent_paged_document_nbytes": payload,
                "dense_document_kv_nbytes": dense,
                "query_shared_document_nbytes": payload,
                "intercept": {
                    "verified": True,
                    "dense_fallback_calls": 0,
                    "max_single_unpack_page_nbytes": 10,
                    "max_dense_full_kv_nbytes": dense,
                },
            }
        )
    return value


def shard(rank: int) -> dict:
    source_index = 6 + rank
    linear = (0, 1, 2)
    full = (3,)
    measurements = {
        "dense-native-functional": measurement("dense-native-functional"),
        "paged-q16": measurement("paged-q16", payload=400),
        "paged-q8": measurement("paged-q8", payload=240),
        "paged-q4": measurement("paged-q4", payload=140),
    }
    ratio_fields = {
        "persistent_total_resident_ratio_vs_stock": 0.625,
        "multi_query_active_ratio_vs_stock": 0.635,
        "cuda_peak_ratio_vs_stock": 1.0,
        "nvml_peak_ratio_vs_stock": 1.0,
        "ttft_ratio_vs_stock_reference_only": 2.0,
        "tpot_ratio_vs_stock_reference_only": 2.0,
    }
    return {
        "status": "completed_shard",
        "rank": rank,
        "world_size": 2,
        "kernel_mode": KERNEL_MODE,
        "production_ttft_optimization_claim_allowed": False,
        "model_manifest_sha256": "c" * 64,
        "workload_metadata": {
            "data_sha256": DATA_SHA,
            "source_revisions": [REVISION],
            "test_v2_consumed": False,
        },
        "protocol": {"benchmark_bits": [16, 8, 4], "warmup_count": 1},
        "gate": {
            "passed": True,
            "native_functional": {
                "passed": True,
                "config_derived": True,
                "expected_linear_layer_count": len(linear),
                "expected_full_attention_layer_count": len(full),
            },
            "native_same_caller": {
                "passed": True,
                "config_derived": True,
                "baseline": "stock-transformers-mutable-eager",
            },
            "paged_same_caller": {
                "passed": True,
                "intercept": {"verified": True, "dense_fallback_calls": 0},
            },
            "benchmark_authorization": {"benchmark_gate_passed": True},
            "config_derived_counts": {
                "linear_layer_count": len(linear),
                "full_attention_layer_count": len(full),
            },
        },
        "rows": [
            {
                "rank": rank,
                "workload_id": f"qasper-{source_index}",
                "source_index": source_index,
                "multi_query_semantics": (
                    "two concurrent request states; repeats the same frozen query"
                ),
                "measurement_order": (
                    tuple(measurements)
                    if rank % 2 == 0
                    else tuple(reversed(tuple(measurements)))
                ),
                "measurements": measurements,
                "paired": {
                    name: {"generated_tokens_exact": True, **ratio_fields}
                    for name in ("paged-q16", "paged-q8", "paged-q4")
                },
            }
        ],
    }


class AggregatePagedRealTest(unittest.TestCase):
    def write(self, directory: Path, values: list[dict]) -> None:
        for value in values:
            (directory / f"paged-real-shard-{value['rank']}.json").write_text(
                json.dumps(value)
            )

    def call(self, directory: Path):
        return aggregate(
            directory,
            expected_shards=2,
            expected_data_sha256=DATA_SHA,
            expected_source_revision=REVISION,
            expected_source_indices=(6, 7),
            expected_workloads=2,
            expected_max_new_tokens=2,
        )

    def test_nested_paired_matrix_allows_q16_equal_and_requires_q8_q4_smaller(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            self.write(directory, [shard(0), shard(1)])
            result = self.call(directory)
            self.assertEqual(result["status"], "passed")
            self.assertFalse(result["production_ttft_optimization_claim_allowed"])
            self.assertEqual(
                result["median_by_config"]["paged-q16"][
                    "payload_compression_ratio"
                ],
                1.0,
            )
            self.assertGreater(
                result["median_by_config"]["paged-q4"][
                    "payload_compression_ratio"
                ],
                1.0,
            )

    def test_missing_intercept_test_v2_or_uncompressed_q4_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            broken = shard(0)
            broken["gate"]["paged_same_caller"]["intercept"]["verified"] = False
            self.write(directory, [broken, shard(1)])
            with self.assertRaisesRegex(AggregateError, "intercept"):
                self.call(directory)

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            bad = deepcopy(shard(1))
            bad["workload_metadata"]["test_v2_consumed"] = True
            self.write(directory, [shard(0), bad])
            with self.assertRaisesRegex(AggregateError, "test-v2"):
                self.call(directory)

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            bad = shard(0)
            bad["rows"][0]["measurements"]["paged-q4"][
                "persistent_paged_document_nbytes"
            ] = 400
            bad["rows"][0]["measurements"]["paged-q4"][
                "query_shared_document_nbytes"
            ] = 400
            self.write(directory, [bad, shard(1)])
            with self.assertRaisesRegex(AggregateError, "not smaller"):
                self.call(directory)

    def test_warmup_q16_exactness_and_ab_ba_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            bad = shard(0)
            bad["rows"][0]["measurements"]["paged-q8"]["warmup_count"] = 0
            self.write(directory, [bad, shard(1)])
            with self.assertRaisesRegex(AggregateError, "warmup"):
                self.call(directory)

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            bad = shard(0)
            bad["rows"][0]["paired"]["paged-q16"][
                "generated_tokens_exact"
            ] = False
            self.write(directory, [bad, shard(1)])
            with self.assertRaisesRegex(AggregateError, "Q16"):
                self.call(directory)

        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            bad = shard(1)
            bad["rows"][0]["measurement_order"] = tuple(
                bad["rows"][0]["measurements"]
            )
            self.write(directory, [shard(0), bad])
            with self.assertRaisesRegex(AggregateError, "rank parity"):
                self.call(directory)


if __name__ == "__main__":
    unittest.main()
