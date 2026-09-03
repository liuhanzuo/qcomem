from __future__ import annotations

import unittest
from dataclasses import dataclass

import torch

from aggregate_lora_deployment_semantic_gate import aggregate_semantic_shards
from run_lora_deployment_semantic_gate import compare_one


@dataclass
class _TinyState:
    depth: int
    document_length: int
    current_length: int
    document_residual: torch.Tensor


class _TinyPacked:
    def __init__(self, document_residual: torch.Tensor) -> None:
        self.document_residual = document_residual

    def fork(self) -> _TinyState:
        length = self.document_residual.shape[1]
        return _TinyState(1, length, length, self.document_residual.clone())


class _TinyLanguageModel:
    norm = torch.nn.Identity()


class _TinyAdapter:
    def __init__(self) -> None:
        self.num_layers = 2
        self.language_model = _TinyLanguageModel()
        self.lm_head = torch.nn.Linear(4, 7, bias=False)
        torch.manual_seed(9)
        torch.nn.init.normal_(self.lm_head.weight)
        self.continue_lengths: list[int] = []

    def continue_lower_replay(
        self, state: _TinyState, tokens: torch.Tensor
    ) -> torch.Tensor:
        self.continue_lengths.append(tokens.shape[1])
        state.current_length += tokens.shape[1]
        values = tokens.to(torch.float32).unsqueeze(-1)
        return torch.cat([values + offset for offset in range(4)], dim=-1)

    def make_cache(self) -> dict[str, bool]:
        return {"document_prefilled": False}

    def _run_layers(
        self,
        hidden: torch.Tensor,
        start: int,
        end: int,
        *,
        past_key_values=None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        del start, end, position_offset
        if past_key_values is not None:
            if not past_key_values["document_prefilled"]:
                raise AssertionError("query suffix ran before document prefill")
        return hidden

    def run_suffix_cached_last_logits(
        self,
        residuals,
        depth: int,
        cache: dict[str, bool],
        *,
        position_offset: int,
    ) -> torch.Tensor:
        del depth
        if position_offset != 0:
            raise AssertionError("tiny prefill must start at position zero")
        hidden = torch.cat(list(residuals), dim=1)
        cache["document_prefilled"] = True
        return self.lm_head(hidden[:, -1, :])


class DeploymentSemanticGateTest(unittest.TestCase):
    def test_full_query_trajectory_uses_one_lower_continuation(self) -> None:
        adapter = _TinyAdapter()
        packed = _TinyPacked(torch.randn(1, 3, 4))
        query = torch.tensor([[2, 5, 3, 7]])

        result = compare_one(adapter, packed, query)

        self.assertEqual(adapter.continue_lengths, [4])
        self.assertEqual(result["query_positions"], 4)
        self.assertEqual(len(result["positions"]), 4)
        self.assertEqual(result["position_top1_match_rate"], 1.0)
        self.assertLess(result["max_position_kl"], 1e-6)
        self.assertEqual(result["max_abs_logit_error"], 0.0)

    def test_projection_blocks_preserve_every_position(self) -> None:
        adapter = _TinyAdapter()
        packed = _TinyPacked(torch.randn(1, 2, 4))
        query = torch.tensor([[1, 2, 3, 4, 5]])

        result = compare_one(adapter, packed, query, projection_block_size=2)

        self.assertEqual(adapter.continue_lengths, [5])
        self.assertEqual(
            [row["position"] for row in result["positions"]], [0, 1, 2, 3, 4]
        )

    def test_projection_block_size_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            compare_one(
                _TinyAdapter(),
                _TinyPacked(torch.randn(1, 2, 4)),
                torch.tensor([[1]]),
                projection_block_size=0,
            )

    @staticmethod
    def _semantic_shard(rank: int, *, top1: bool = True) -> dict:
        position = {
            "position": 0,
            "top1_match": top1,
            "kl_training_to_deployment": 0.0001,
            "max_abs_logit_error": 0.01,
        }
        return {
            "status": "completed_shard",
            "local_threshold_passed": top1,
            "rank": rank,
            "world_size": 2,
            "training_suffix_execution": (
                "uncached_full_document_plus_query_sequence"
            ),
            "deployment_suffix_execution": (
                "cached_document_prefill_then_full_query_continuation"
            ),
            "comparison_scope": "all_query_positions",
            "thresholds": {"min_top1_match": 1.0, "max_mean_kl": 0.001},
            "global_samples_requested": 2,
            "data_sha256": "data",
            "checkpoint_sha256": "checkpoint",
            "test_v2_used": False,
            "rows": [
                {
                    "source_id": f"source-{rank}",
                    "query_positions": 1,
                    "positions": [position],
                }
            ],
        }

    def test_distributed_semantic_gate_uses_global_position_threshold(self) -> None:
        result = aggregate_semantic_shards(
            [self._semantic_shard(0), self._semantic_shard(1)],
            expected_world_size=2,
            expected_data_sha256="data",
            expected_checkpoint_sha256="checkpoint",
        )
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["samples"], 2)
        self.assertEqual(result["query_positions"], 2)

        failed = aggregate_semantic_shards(
            [self._semantic_shard(0), self._semantic_shard(1, top1=False)],
            expected_world_size=2,
            expected_data_sha256="data",
            expected_checkpoint_sha256="checkpoint",
        )
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["position_top1_match_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
