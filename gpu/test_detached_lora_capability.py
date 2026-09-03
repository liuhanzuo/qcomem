from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from aggregate_detached_lora_semantic_gate import aggregate_detached_shards
from audit_detached_lora_checkpoint import audit_checkpoint
from run_detached_lora_semantic_gate import compare_detached_one


class TinyDetachedAdapter:
    def __init__(self, width: int = 4) -> None:
        self.num_layers = 3
        self.language_model = SimpleNamespace(norm=nn.Identity())
        self.lm_head = nn.Linear(width, 7, bias=False)

    def make_cache(self):
        return {}

    def _run_layers(
        self,
        hidden,
        start,
        end,
        *,
        past_key_values=None,
        position_offset=0,
    ):
        del start, end
        if position_offset == 0:
            past_key_values["summary"] = hidden.mean(dim=1, keepdim=True)
            return hidden
        return hidden + past_key_values["summary"]

    def continue_lower_replay(self, state, query):
        del state
        return query

    def run_suffix_cached_last_logits(
        self, residuals, depth, cache, *, position_offset
    ):
        hidden = self._run_layers(
            residuals[0],
            depth,
            self.num_layers,
            past_key_values=cache,
            position_offset=position_offset,
        )
        return self.lm_head(hidden[:, -1])


class TinyPacked:
    def __init__(self, document: torch.Tensor) -> None:
        self.document = document

    def fork(self):
        return SimpleNamespace(
            document_residual=self.document.clone(),
            document_length=self.document.shape[1],
            depth=1,
        )


class DetachedCapabilityTest(unittest.TestCase):
    def test_all_query_detached_semantics_and_immutability(self) -> None:
        torch.manual_seed(3)
        adapter = TinyDetachedAdapter()
        packed = TinyPacked(torch.randn(1, 5, 4))
        query = torch.randn(1, 3, 4)
        result = compare_detached_one(
            adapter, packed, query, projection_block_size=2
        )
        self.assertEqual(result["query_positions"], 3)
        self.assertEqual(result["position_top1_match_rate"], 1.0)
        self.assertEqual(result["max_abs_logit_error"], 0.0)
        self.assertTrue(result["cache_immutability"]["hard_gate_passed"])

    def test_global_aggregate_requires_every_query_position(self) -> None:
        shards = []
        for rank in range(2):
            positions = [
                {
                    "position": position,
                    "kl_detached_to_deployment": 0.0,
                    "max_abs_logit_error": 0.0,
                    "top1_match": True,
                }
                for position in range(3)
            ]
            shards.append(
                {
                    "status": "completed_shard",
                    "local_threshold_passed": True,
                    "rank": rank,
                    "world_size": 2,
                    "training_suffix_execution": (
                        "cached_document_prefill_detached_then_full_query_continuation"
                    ),
                    "deployment_suffix_execution": (
                        "cached_document_prefill_then_full_query_continuation"
                    ),
                    "comparison_scope": "all_query_positions",
                    "global_samples_requested": 2,
                    "thresholds": {
                        "min_top1_match": 1.0,
                        "max_mean_kl": 1e-6,
                        "max_logit_error": 0.0,
                    },
                    "data_sha256": "data",
                    "checkpoint_sha256": "checkpoint",
                    "test_v2_used": False,
                    "rows": [
                        {
                            "source_id": f"source-{rank}",
                            "query_positions": 3,
                            "cache_immutability": {"hard_gate_passed": True},
                            "positions": positions,
                        }
                    ],
                }
            )
        result = aggregate_detached_shards(
            shards,
            expected_world_size=2,
            expected_data_sha256="data",
            expected_checkpoint_sha256="checkpoint",
            expected_query_positions=3,
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["query_positions"], 6)

    def test_checkpoint_audit_requires_query_gradients_and_update(self) -> None:
        rank_row = {
            "module_count": 2,
            "finite_module_count": 2,
            "nonzero_module_count": 2,
        }
        cache_row = {
            "hard_gate_passed": True,
            "detached_cache_storage_disjoint": True,
            "detached_cache_all_tensors_grad_free": True,
            "original_cache_versions_unchanged": True,
            "document_cache_tensor_count": 3,
            "query_positions_expected": 3,
            "query_positions_observed": 3,
        }
        metadata = {
            "last_step": 1,
            "world_size": 2,
            "adapter": {
                "installed_modules": ["m0", "m1"],
                "trainable_parameters": 16,
            },
            "semantics": {
                "student_suffix_execution_option": "detached-document-cache",
                "document_cache_detached_before_query": True,
                "document_prefill_parameter_gradients_enabled": False,
            },
            "last_gradient_coverage": {
                "hard_gate_passed": True,
                "gradient_scope": "query_continuation_only",
                "by_rank": [rank_row, rank_row],
            },
            "last_detached_capability": {
                "hard_gate_passed": True,
                "by_rank": [cache_row, cache_row],
            },
            "test_v2_used": False,
        }
        payload = {
            "format": "qcomem_suffix_lora_v1",
            "step": 1,
            "metadata": metadata,
            "lora": {
                "m0.lora_a": torch.ones(2, 2),
                "m0.lora_b": torch.ones(2, 2),
                "m1.lora_a": torch.ones(2, 2),
                "m1.lora_b": torch.ones(2, 2),
            },
            "optimizer": {"state": {0: {"step": torch.tensor(1)}}},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.pt"
            torch.save(payload, path)
            result = audit_checkpoint(
                path,
                expected_sha256=None,
                expected_world_size=2,
                expected_modules=2,
                expected_query_positions=3,
            )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["lora_b_update"]["nonzero_tensors"], 2)


if __name__ == "__main__":
    unittest.main()
