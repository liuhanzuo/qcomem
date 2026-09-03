from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aggregate_replay import paired_lora_summary, token_position_agreement
from run_downstream import load_samples
from run_replay_diagnostic import (
    CONFIG_SUITES,
    QUANT_LORA_CONFIG,
    parse_excluded_indices,
    resolve_config,
    validate_lora_targets,
    validation_excluded_source_indices,
    validation_source_range,
)


class MixedValidationProtocolTest(unittest.TestCase):
    def test_frozen_layer_policies_are_complete(self) -> None:
        expected = {
            "replay-d7-layer-q16": (
                16, 16, 16, (16, 16, 16, 16, 16, 16, 16)
            ),
            "replay-d7-layer-q8": (
                8, 8, 8, (8, 8, 8, 8, 8, 8, 8)
            ),
            "replay-d7-frozen-static": (
                4, 4, 8, (8, 8, 8, 4, 8, 8, 8)
            ),
            "replay-d7-same-memory-mixed": (
                4, 4, None, (8, 8, 4, 4, 8, 8, 8)
            ),
            "replay-d7-minus25-mixed": (
                4, 2, None, (8, 8, 2, 2, 2, 8, 2)
            ),
        }
        self.assertEqual(
            CONFIG_SUITES["layer-validation"][:2], ("dense", "prefix")
        )
        for name, (
            residual_bits,
            attention_bits,
            linear_bits,
            layer_bits,
        ) in expected.items():
            resolved = resolve_config(name)
            self.assertEqual(resolved.mode, "replay")
            self.assertEqual(resolved.depth, 7)
            self.assertEqual(resolved.residual_bits, residual_bits)
            self.assertEqual(resolved.attention_bits, attention_bits)
            self.assertEqual(resolved.linear_bits, linear_bits)
            self.assertEqual(resolved.cache_layer_bits, layer_bits)
            self.assertEqual(len(resolved.cache_layer_bits or ()), resolved.depth)

    def test_layer_validation_defaults_to_disjoint_source_range(self) -> None:
        self.assertEqual(
            validation_source_range("layer-validation", None, None), (6, 35)
        )
        self.assertEqual(validation_source_range("exact", None, None), (None, None))
        self.assertEqual(
            validation_excluded_source_indices("layer-validation", None), (4, 5)
        )
        self.assertEqual(validation_excluded_source_indices("exact", None), ())
        self.assertEqual(parse_excluded_indices("4, 5"), (4, 5))

    def test_quant_lora_suite_pairs_identical_store_policies(self) -> None:
        self.assertEqual(
            CONFIG_SUITES["quant-lora-validation"],
            (
                "dense",
                "replay-d7-layer-q16",
                "replay-d7-frozen-static",
                QUANT_LORA_CONFIG,
            ),
        )
        baseline = resolve_config("replay-d7-frozen-static")
        adapted = resolve_config(QUANT_LORA_CONFIG)
        self.assertEqual(baseline, adapted)
        self.assertEqual(
            validation_source_range("quant-lora-validation", None, None),
            (6, 35),
        )
        self.assertEqual(
            validation_excluded_source_indices("quant-lora-validation", None),
            (4, 5),
        )

    def test_quant_lora_suite_fails_closed_on_adapter_targets(self) -> None:
        checkpoint = Path("checkpoint.pt")
        validate_lora_targets(
            "quant-lora-validation", checkpoint, [QUANT_LORA_CONFIG]
        )
        for targets in (
            [],
            ["replay-d7-frozen-static"],
            ["dense"],
            ["replay-d7-layer-q16"],
            ["replay-d7-frozen-static", QUANT_LORA_CONFIG],
        ):
            with self.assertRaises(ValueError):
                validate_lora_targets(
                    "quant-lora-validation", checkpoint, targets
                )

    def test_paired_lora_summary_includes_quality_trajectory_and_ttft(self) -> None:
        reference = [
            {
                "dataset": dataset,
                "id": str(index),
                "source_index": index,
                "prediction": "base",
                "generated_token_ids": [1, 2],
                "f1": f1,
                "ttft_seconds": 1.0,
            }
            for index, (dataset, f1) in enumerate(
                (("qasper", 0.5), ("2wikimqa", 1.0)), start=6
            )
        ]
        candidate = [
            {
                **row,
                "prediction": "adapted",
                "generated_token_ids": [1, 3],
                "f1": row["f1"] - 0.1,
                "ttft_seconds": 1.2,
            }
            for row in reference
        ]
        result = paired_lora_summary(
            reference,
            candidate,
            catastrophic_delta=-0.5,
            bootstrap_seed=17,
        )
        self.assertAlmostEqual(
            result["mean_f1_delta_lora_minus_untrained"], -0.1
        )
        self.assertEqual(result["token_sequence_exact_agreement_rate"], 0.0)
        self.assertEqual(result["mean_token_position_agreement"], 0.5)
        self.assertEqual(result["catastrophic_regression_rate"], 0.0)
        self.assertAlmostEqual(
            result["ttft"]["median_delta_seconds_lora_minus_untrained"], 0.2
        )

    def test_source_filter_happens_before_per_dataset_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "rows.jsonl"
            rows = [
                {
                    "dataset": dataset,
                    "_source_index": source_index,
                    "input": "question",
                    "context": "context",
                    "answers": ["answer"],
                }
                for dataset in ("qasper", "2wikimqa")
                for source_index in range(40)
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            selected = load_samples(
                path,
                30,
                source_index_start=6,
                source_index_end=35,
                exclude_source_indices=(4, 5),
            )
        self.assertEqual(len(selected), 60)
        for dataset in ("qasper", "2wikimqa"):
            indices = [
                row["_source_index"]
                for row in selected
                if row["dataset"] == dataset
            ]
            self.assertEqual(indices, list(range(6, 36)))

    def test_token_position_agreement_counts_length_mismatch(self) -> None:
        self.assertEqual(token_position_agreement([], []), 1.0)
        self.assertEqual(token_position_agreement([1, 2], [1, 3]), 0.5)
        self.assertEqual(token_position_agreement([1, 2], [1]), 0.5)


if __name__ == "__main__":
    unittest.main()
