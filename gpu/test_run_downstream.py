from __future__ import annotations

import unittest

from run_downstream import CONFIGS, DATASET_PROMPTS, assigned_configs, generation_limit


class LongBenchProtocolTest(unittest.TestCase):
    def test_dataset_specific_generation_limits(self) -> None:
        self.assertEqual(generation_limit("qasper", 256), 128)
        self.assertEqual(generation_limit("2wikimqa", 256), 32)
        self.assertEqual(generation_limit("qasper", 16), 16)

    def test_official_qa_prompts_keep_context_and_question_slots(self) -> None:
        for dataset in ("qasper", "2wikimqa"):
            prompt = DATASET_PROMPTS[dataset]
            self.assertEqual(prompt.count("{context}"), 1)
            self.assertEqual(prompt.count("{input}"), 1)
        self.assertIn("unanswerable", DATASET_PROMPTS["qasper"])
        self.assertIn("given passages", DATASET_PROMPTS["2wikimqa"])

    def test_eight_gpu_assignment_is_complete_and_cost_aware(self) -> None:
        assignments = [assigned_configs(rank, 8) for rank in range(8)]
        self.assertCountEqual(
            [config for rank_configs in assignments for config in rank_configs],
            CONFIGS,
        )
        self.assertEqual(assignments[0], ["dense"])
        self.assertEqual(assignments[1], ["d7-q16"])
        self.assertEqual(len(assignments[4]), 2)
        self.assertEqual(len(assignments[5]), 2)


if __name__ == "__main__":
    unittest.main()
