from __future__ import annotations

import copy
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from aggregate_deployment import direct_cow_gate_passed, validate_complete_protocol
from qcomem_deployment import (
    MemoryRecorder,
    capacity_estimate,
    load_mixed_policy,
    parse_deployment_config,
    run_exactness_gate,
    run_incremental_generation,
    shuffled_config_orders,
)
from run_deployment_bench import batch_prefix, warmup_config


@dataclass
class FakeCache:
    tokens: torch.Tensor


@dataclass
class FakeFullState:
    cache: FakeCache
    document_length: int
    current_length: int

    @property
    def stored_nbytes(self) -> int:
        return self.cache.tokens.numel() * self.cache.tokens.element_size()

    def fork(self) -> "FakeFullState":
        return copy.deepcopy(self)


@dataclass
class FakeLowerState:
    depth: int
    document_length: int
    current_length: int
    document_residual: torch.Tensor | None
    cache: FakeCache

    @property
    def stored_nbytes(self) -> int:
        residual_bytes = (
            0
            if self.document_residual is None
            else self.document_residual.numel()
            * self.document_residual.element_size()
        )
        return residual_bytes + self.cache.tokens.numel() * self.cache.tokens.element_size()

    def fork(self) -> "FakeLowerState":
        return copy.deepcopy(self)

    def quantize(self, **kwargs) -> "FakeLowerState":
        del kwargs
        return self.fork()


class FakeAdapter:
    vocab_size = 32

    def __init__(self) -> None:
        self.suffix_call_lengths: list[int] = []

    @staticmethod
    def _logits(last_token: int) -> torch.Tensor:
        logits = torch.full((1, FakeAdapter.vocab_size), -1000.0)
        logits[0, (last_token + 1) % FakeAdapter.vocab_size] = 1.0
        return logits

    def full_last_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._logits(int(tokens[0, -1]))

    def write_full_prefix(self, tokens: torch.Tensor) -> FakeFullState:
        return FakeFullState(
            cache=FakeCache(tokens.clone()),
            document_length=tokens.shape[1],
            current_length=tokens.shape[1],
        )

    def continue_full_prefix(
        self, state: FakeFullState, tokens: torch.Tensor
    ) -> torch.Tensor:
        state.cache.tokens = torch.cat([state.cache.tokens, tokens], dim=1)
        state.current_length += tokens.shape[1]
        return self._logits(int(tokens[0, -1]))

    def write_lower_replay(self, tokens: torch.Tensor, depth: int) -> FakeLowerState:
        return FakeLowerState(
            depth=depth,
            document_length=tokens.shape[1],
            current_length=tokens.shape[1],
            document_residual=tokens.float().unsqueeze(-1),
            cache=FakeCache(tokens.clone()),
        )

    def continue_lower_replay(
        self, state: FakeLowerState, tokens: torch.Tensor
    ) -> torch.Tensor:
        state.cache.tokens = torch.cat([state.cache.tokens, tokens], dim=1)
        state.current_length += tokens.shape[1]
        return tokens.float().unsqueeze(-1)

    def make_cache(self) -> FakeCache:
        return FakeCache(torch.empty((1, 0), dtype=torch.long))

    def run_suffix_cached_last_logits(
        self,
        residuals,
        depth: int,
        cache: FakeCache,
        *,
        position_offset: int,
    ) -> torch.Tensor:
        del depth, position_offset
        tokens = torch.cat(list(residuals), dim=1).squeeze(-1).long()
        self.suffix_call_lengths.append(tokens.shape[1])
        cache.tokens = torch.cat([cache.tokens, tokens], dim=1)
        return self._logits(int(tokens[0, -1]))


class DenseDiagnosticDivergentAdapter(FakeAdapter):
    def full_last_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self._logits(int(tokens[0, -1]) + 1)


class DeploymentConfigTest(unittest.TestCase):
    @staticmethod
    def _complete_protocol_shards() -> list[dict]:
        configs = [
            {"name": "full-prefix-q16"},
            {
                "name": "qcomem-d7-frozen-static",
                "depth": 7,
                "residual_bits": 4,
                "attention_bits": 4,
                "linear_bits": 8,
                "cache_layer_bits": [8, 8, 8, 4, 8, 8, 8],
            },
        ]
        metadata = {
            "data_sha256": "data-sha",
            "source_revisions": ["revision"],
            "test_v2_consumed": False,
        }
        shards = []
        for rank, source_index in enumerate((6, 7)):
            workload_id = f"qasper-{source_index}"
            rows = []
            for position, config in enumerate(configs):
                row = {
                    "workload_id": workload_id,
                    "dataset": "qasper",
                    "source_index": source_index,
                    "repeat": 0,
                    "config": config["name"],
                    "randomized_order_position": position,
                    "max_new_tokens": 8,
                    "persistent_total_resident_nbytes": 10,
                    "persistent_materialized_staging_nbytes": 0,
                    "capacity_document_denominator_nbytes": 10,
                    "cuda_peak_allocated_bytes": 20,
                    "cuda_peak_reserved_bytes": 30,
                    "nvml_sampled_peak_process_bytes": 40,
                    "ttft_seconds": 0.1,
                }
                if config["name"].startswith("qcomem-"):
                    row.update(
                        {
                            "cow_initial_shared_nbytes": 4,
                            "cow_initial_private_nbytes": 6,
                            "cow_after_query_shared_nbytes": 2,
                            "cow_after_query_private_nbytes": 8,
                            "cow_final_shared_nbytes": 0,
                            "cow_final_private_nbytes": 10,
                            "fork_memory": {
                                "strategy_effective": "paged-cow-staging",
                                "fallback_reason": None,
                            },
                        }
                    )
                rows.append(row)
            shards.append(
                {
                    "rank": rank,
                    "world_size": 2,
                    "workload_metadata": metadata,
                    "configs": configs,
                    "fork_strategy": "paged-cow-staging",
                    "warmups": 1,
                    "repeats": 1,
                    "rows": rows,
                    "randomized_orders": {
                        workload_id: [[config["name"] for config in configs]]
                    },
                }
            )
        return shards

    def test_batch_prefix_accepts_longbench_1d_and_synthetic_2d(self) -> None:
        one_dimensional = torch.tensor([1, 2, 3, 4])
        two_dimensional = one_dimensional.unsqueeze(0)
        expected = torch.tensor([[1, 2, 3]])
        self.assertTrue(torch.equal(batch_prefix(one_dimensional, 3), expected))
        self.assertTrue(torch.equal(batch_prefix(two_dimensional, 3), expected))

    def test_config_parser_covers_uniform_frozen_and_layerwise(self) -> None:
        self.assertEqual(
            parse_deployment_config("dense-recompute").mode, "dense_recompute"
        )
        frozen = parse_deployment_config("qcomem-d7-r4-a4-l8")
        self.assertEqual(
            (frozen.depth, frozen.residual_bits, frozen.attention_bits, frozen.linear_bits),
            (7, 4, 4, 8),
        )
        mixed = parse_deployment_config(
            "qcomem-d7-mixed", mixed_layer_bits=(8, 8, 4, 4, 8, 8, 8)
        )
        self.assertEqual(mixed.cache_layer_bits, (8, 8, 4, 4, 8, 8, 8))
        explicit = parse_deployment_config(
            "qcomem-d3-r4-layers=8,4,8"
        )
        self.assertEqual(explicit.cache_layer_bits, (8, 4, 8))
        frozen_static = parse_deployment_config("qcomem-d7-frozen-static")
        self.assertEqual(
            (
                frozen_static.residual_bits,
                frozen_static.attention_bits,
                frozen_static.linear_bits,
                frozen_static.cache_layer_bits,
            ),
            (4, 4, 8, (8, 8, 8, 4, 8, 8, 8)),
        )

    def test_policy_loader_accepts_layer_sensitivity_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(
                '{"policies":{"same_memory_as_frozen":'
                '{"residual_bits":4,"cache_layer_bits":[8,4,8]}}}'
            )
            self.assertEqual(
                load_mixed_policy(path, "same_memory_as_frozen"),
                (4, (8, 4, 8)),
            )

    def test_orders_are_reproducible_and_change_across_repeats(self) -> None:
        configs = ["a", "b", "c", "d"]
        first = shuffled_config_orders(configs, repeats=3, seed=17)
        second = shuffled_config_orders(configs, repeats=3, seed=17)
        self.assertEqual(first, second)
        self.assertTrue(all(sorted(order) == configs for order in first))
        self.assertGreater(len({tuple(order) for order in first}), 1)

    def test_capacity_reserves_one_active_request(self) -> None:
        result = capacity_estimate(
            total_device_bytes=1_000,
            model_allocated_bytes=200,
            persistent_document_bytes=100,
            request_peak_allocated_bytes=450,
            request_start_allocated_bytes=300,
            safety_headroom_bytes=100,
        )
        self.assertEqual(result["max_resident_documents_store_only"], 7)
        self.assertEqual(
            result["max_resident_documents_with_one_active_request"], 5
        )

    def test_aggregate_fails_closed_on_missing_direct_cow_gate(self) -> None:
        self.assertTrue(direct_cow_gate_passed({"fork_strategy": "deep-clone"}))
        self.assertFalse(
            direct_cow_gate_passed({"fork_strategy": "paged-cow-staging"})
        )
        direct = {
            "passed": True,
            "semantic_version": "incremental-three-way-v1",
            "caller_boundary_match": True,
            "incremental_three_way_token_exact": True,
            "same_persistent_source": True,
            "cow_was_exercised": True,
            "strategy_effective": "paged-cow-staging",
            "source_after_eager": {"verified": True},
            "source_after_cow": {"verified": True},
            "cow_immutable_audit": {"verified": True},
            "comparisons": {
                "full_prefix_vs_eager_q16": {
                    "passed": True,
                    "token_sequence_exact": True,
                },
                "full_prefix_vs_cow_q16": {
                    "passed": True,
                    "token_sequence_exact": True,
                },
                "eager_q16_vs_cow_q16": {
                    "passed": True,
                    "token_sequence_exact": True,
                    "logits_bitwise_exact": True,
                },
            },
        }
        gate = {
            "fork_strategy": "paged-cow-staging",
            "hard_gate_reference": "incremental-full-prefix-q16",
            "dense_single_chunk_diagnostic_only": True,
            "incremental_hard_gate": {"passed": True},
            "cow_vs_deep_clone_q16": direct,
        }
        self.assertTrue(direct_cow_gate_passed(gate))
        direct["comparisons"]["full_prefix_vs_cow_q16"]["passed"] = False
        self.assertFalse(direct_cow_gate_passed(gate))
        direct["comparisons"]["full_prefix_vs_cow_q16"]["passed"] = True
        direct["comparisons"]["eager_q16_vs_cow_q16"][
            "logits_bitwise_exact"
        ] = False
        self.assertFalse(direct_cow_gate_passed(gate))

    def test_protocol_validator_fails_closed_on_missing_measurement(self) -> None:
        shards = self._complete_protocol_shards()
        result = validate_complete_protocol(
            shards,
            expected_shards=2,
            expected_configs=["full-prefix-q16", "qcomem-d7-frozen-static"],
            expected_workloads=2,
            expected_source_indices=[6, 7],
            expected_data_sha256="data-sha",
            expected_source_revision="revision",
            expected_fork_strategy="paged-cow-staging",
            expected_warmups=1,
            expected_repeats=1,
            expected_max_new_tokens=8,
            require_complete_measurements=True,
            require_no_test_v2=True,
        )
        self.assertEqual(result["measurements"], 4)
        broken = copy.deepcopy(shards)
        del broken[1]["rows"][1]["nvml_sampled_peak_process_bytes"]
        with self.assertRaisesRegex(ValueError, "omits measurements"):
            validate_complete_protocol(
                broken,
                expected_shards=2,
                require_complete_measurements=True,
            )

    def test_protocol_validator_preserves_synthetic_workloads(self) -> None:
        shards = self._complete_protocol_shards()
        for shard in shards:
            shard["workload"] = "synthetic"
            for row in shard["rows"]:
                row["source_index"] = None
                row["dataset"] = "synthetic-capacity"
        result = validate_complete_protocol(shards, expected_shards=2)
        self.assertEqual(result["source_indices"], [])

class IncrementalCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = FakeAdapter()
        self.document = torch.tensor([[3, 4, 5, 6]])
        self.query = torch.tensor([[7, 8]])

    def test_per_token_gate_covers_full_prefix_and_q16_replay(self) -> None:
        result = run_exactness_gate(
            self.adapter,
            self.document,
            self.query,
            depth=2,
            group_size=64,
            max_new_tokens=4,
            eos_token_ids=set(),
            require_exact_logits=True,
        )
        self.assertTrue(result["passed"])
        for comparison in result["comparisons"].values():
            self.assertTrue(comparison["token_sequence_exact"])
            self.assertTrue(comparison["logits_bitwise_exact"])
        for comparison in result["pairwise"].values():
            self.assertTrue(comparison["token_sequence_exact"])
            self.assertTrue(comparison["logits_bitwise_exact"])
            self.assertIsNone(comparison["first_token_divergence_step"])
            self.assertIsNone(comparison["first_logit_difference_step"])
        self.assertEqual(
            result["execution_boundaries"]["dense_recompute"][
                "full_history_calls"
            ],
            [6, 7, 8, 9],
        )
        self.assertEqual(
            result["execution_boundaries"]["full_prefix"][
                "document_write_chunks"
            ],
            [4],
        )
        self.assertEqual(
            result["execution_boundaries"]["qcomem_q16"][
                "suffix_query_prefill_chunks"
            ],
            [2],
        )

    def test_dense_single_chunk_divergence_is_diagnostic_only(self) -> None:
        result = run_exactness_gate(
            DenseDiagnosticDivergentAdapter(),
            self.document,
            self.query,
            depth=2,
            group_size=64,
            max_new_tokens=4,
            eos_token_ids=set(),
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["dense_single_chunk_diagnostic_only"])
        self.assertFalse(result["dense_diagnostic_passed"])
        self.assertTrue(result["incremental_hard_gate"]["passed"])
        self.assertTrue(
            result["pairwise"]["full_prefix_vs_qcomem_q16"][
                "token_sequence_exact"
            ]
        )

    def test_decode_cache_grows_without_recomputing_boundary(self) -> None:
        config = parse_deployment_config("qcomem-d2-r16-a16-l16")
        persistent = self.adapter.write_lower_replay(self.document, depth=2)
        trace = run_incremental_generation(
            self.adapter,
            config,
            self.document,
            self.query,
            persistent,
            max_new_tokens=4,
            eos_token_ids=set(),
            recorder=MemoryRecorder(),
        )
        self.assertEqual(trace.generated_token_ids, [9, 10, 11, 12])
        self.assertGreater(trace.selected_fork_active_state_peak_nbytes, 0)
        self.assertGreater(trace.decode_kv_peak_nbytes, trace.decode_kv_steady_nbytes - 1)
        self.assertGreater(trace.decode_kv_peak_nbytes, 0)
        self.assertEqual(self.adapter.suffix_call_lengths[:2], [4, 2])

    def test_warmup_batches_one_dimensional_longbench_tokens(self) -> None:
        config = parse_deployment_config("qcomem-d2-r16-a16-l16")
        warmup_config(
            self.adapter,
            config,
            torch.arange(160),
            torch.arange(40),
            group_size=64,
            eos_ids=set(),
        )
        self.assertEqual(self.adapter.suffix_call_lengths[:2], [128, 32])


if __name__ == "__main__":
    unittest.main()
