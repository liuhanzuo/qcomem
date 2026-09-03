from __future__ import annotations

import unittest

import torch

from aggregate_sft_dcp_downstream import merge_config
from run_sft_dcp_downstream import _sample_audit, _sample_parameter_values


class SFTDCPDownstreamTests(unittest.TestCase):
    def _shards(self) -> list[dict]:
        rows = [
            {
                "dataset": dataset,
                "id": f"{dataset}-{index}",
                "source_index": index,
                "prediction": "answer",
                "generated_tokens": 1,
                "f1": 1.0,
            }
            for dataset in ("qasper", "2wikimqa")
            for index in range(6, 36)
        ]
        return [
            {
                "rank": rank,
                "config": "dense",
                "rows": rows[rank::8],
            }
            for rank in range(8)
        ]

    def test_merge_requires_exact_paired_source_6_to_35(self) -> None:
        merged = merge_config(self._shards(), "dense", expected_prefix="sft")
        self.assertEqual(len(merged["rows"]), 60)
        self.assertEqual(merged["mean_f1"], 1.0)

        bad = self._shards()
        bad[0]["rows"][0]["source_index"] = 68
        with self.assertRaisesRegex(ValueError, "indices are not 6--35"):
            merge_config(bad, "dense", expected_prefix="sft")

    def test_parameter_sample_gate_detects_checkpoint_change(self) -> None:
        model = torch.nn.Linear(8, 4, bias=False).to(torch.bfloat16)
        before = _sample_parameter_values(model)
        with torch.no_grad():
            model.weight.add_(1)
        audit = _sample_audit(before, model)
        self.assertGreater(audit["changed_from_base_elements"], 0)
        self.assertEqual(len(audit["bf16_sample_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
