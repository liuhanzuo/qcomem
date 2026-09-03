from __future__ import annotations

import unittest

from analyze_validation import bootstrap_mean_ci, comparison


def config(name: str, scores: list[tuple[str, float]]) -> dict:
    return {
        "config": name,
        "rows": [
            {
                "dataset": dataset,
                "id": f"{dataset}-{index}",
                "source_index": index,
                "f1": score,
                "prediction": str(score),
                "generated_tokens": index + 1,
            }
            for index, (dataset, score) in enumerate(scores)
        ],
    }


class ValidationAnalysisTest(unittest.TestCase):
    def test_bootstrap_is_deterministic(self) -> None:
        values = [-0.1, 0.0, 0.1]
        self.assertEqual(
            bootstrap_mean_ci(values, seed=7, repetitions=100),
            bootstrap_mean_ci(values, seed=7, repetitions=100),
        )

    def test_noninferiority_uses_overall_and_dataset_means(self) -> None:
        reference = config("q16", [("a", 0.5), ("b", 0.5)])
        passing = config("q8", [("a", 0.49), ("b", 0.48)])
        failing = config("q4", [("a", 0.50), ("b", 0.46)])
        self.assertTrue(comparison(passing, reference, seed=1)["passes_preregistered_mean_margins"])
        self.assertFalse(comparison(failing, reference, seed=1)["passes_preregistered_mean_margins"])


if __name__ == "__main__":
    unittest.main()
