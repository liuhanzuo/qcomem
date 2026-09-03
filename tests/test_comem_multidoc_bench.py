from __future__ import annotations

import unittest

import mlx.core as mx

from macllm_bench.comem_multidoc_bench import (
    quality_metrics,
    select_aggregate_policy,
)


class CoMemMultiDocumentBenchmarkTest(unittest.TestCase):
    def test_teacher_forced_quality_is_exact_for_identical_logits(self) -> None:
        logits = mx.arange(1 * 8 * 16).reshape(1, 8, 16).astype(mx.float32)
        metrics = quality_metrics(
            logits,
            logits,
            document_tokens=3,
            query_prefix_tokens=2,
            answer_tokens=mx.array([4, 5, 6]),
        )
        self.assertEqual(metrics["evaluation_scope"], "teacher_forced_answer")
        self.assertEqual(metrics["evaluated_answer_tokens"], 3)
        self.assertAlmostEqual(metrics["kl_divergence"], 0.0)
        self.assertAlmostEqual(metrics["answer_nll_delta"], 0.0)
        self.assertEqual(metrics["top1_agreement_rate"], 1.0)

    def test_aggregate_policy_uses_all_query_bounds(self) -> None:
        common = {
            "residual_relative_rmse": 0.02,
            "min_query_top1_agreement_rate": 1.0,
        }
        rows = [
            {
                **common,
                "depth": 5,
                "bits": 2,
                "corpus_stored_nbytes": 20,
                "max_position_kl": 0.03,
            },
            {
                **common,
                "depth": 5,
                "bits": 4,
                "corpus_stored_nbytes": 40,
                "max_position_kl": 0.01,
            },
            {
                **common,
                "depth": 5,
                "bits": 8,
                "corpus_stored_nbytes": 80,
                "max_position_kl": 0.001,
            },
        ]
        policy = select_aggregate_policy(
            rows,
            max_kl=0.02,
            max_relative_rmse=0.05,
            min_top1_agreement=1.0,
        )
        self.assertEqual(policy.as_dict(), {"5": 4})


if __name__ == "__main__":
    unittest.main()
