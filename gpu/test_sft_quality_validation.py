from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sft_quality_validation import (
    FROZEN_LONGBENCH_REVISION,
    PreparedQualityDataset,
    QualityContractError,
    paired_generation_comparison,
    paired_quality_comparison,
    sha256_file,
    validate_longbench_validation_rows,
)
from split_supervised_sft_scale import MANIFEST_SCHEMA, example_fingerprints, stable_json


class SFTQualityValidationTests(unittest.TestCase):
    def _quality_row(self, dataset: str, source_id: str) -> dict:
        context = f"context {dataset} {source_id}"
        question = f"question {source_id}"
        return {
            "dataset": dataset,
            "source_id": source_id,
            "source_split": "train",
            "context": context,
            "input": question,
            "input_ids": [11, 12, 13, 99],
            "labels": [-100, -100, 13, 99],
            "document_input_ids": [11],
            "query_input_ids": [12],
            "answer_input_ids": [13, 99],
            "token_counts": {
                "total": 4,
                "prompt": 2,
                "answer": 1,
                "answer_with_eos": 2,
                "query": 1,
            },
            "provenance": {
                "source_split": "train",
                "fingerprints": example_fingerprints(source_id, context, question),
            },
        }

    def _write_quality_contract(self, directory: Path):
        heldout = directory / "heldout-ce.jsonl"
        rows = [
            self._quality_row(
                "qasper" if index < 4 else "2wikimqa", f"source-{index}"
            )
            for index in range(8)
        ]
        heldout.write_text(
            "".join(stable_json(row) + "\n" for row in rows), encoding="utf-8"
        )
        train_sha = "a" * 64
        heldout_sha = sha256_file(heldout)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "status": "passed",
            "outputs": {
                "train_jsonl": {"basename": "train.jsonl", "sha256": train_sha},
                "heldout_ce_jsonl": {
                    "basename": heldout.name,
                    "sha256": heldout_sha,
                    "count": 8,
                    "dataset_counts": {"qasper": 4, "2wikimqa": 4},
                },
            },
            "disjoint_audit": {
                "all_zero": True,
                "source_id_intersection_count": 0,
                "component_intersection_count": 0,
                "fingerprint_intersection_counts": {
                    "id_sha256": 0,
                    "context_input_sha256": 0,
                    "context_sha256": 0,
                    "input_sha256": 0,
                },
            },
            "data_governance": {
                "all_rows_top_level_source_split": "train",
                "all_rows_provenance_source_split": "train",
                "validation_or_test_rows_used": False,
                "raw_test_v2_read": False,
                "heldout_ce_usage": "train-split CE diagnostics only",
                "heldout_ce_is_final_downstream_evaluation": False,
            },
            "parent": {
                "raw_test_v2_read_by_converter": False,
                "test_v2_content_hash_check": "deferred_not_read",
            },
            "tokenizer": {
                "class": "FakeTokenizer",
                "vocab_size": 100,
                "eos_token_id": 99,
                "chat_template_sha256": "b" * 64,
            },
            "prompt_protocol": {"max_sequence_tokens": 1024},
        }
        manifest_path = directory / "split-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return heldout, heldout_sha, manifest_path, sha256_file(manifest_path), train_sha

    def test_prepared_heldout_contract_validates_every_row_and_sha(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            paths = self._write_quality_contract(Path(raw_directory))
            heldout, heldout_sha, manifest, manifest_sha, train_sha = paths
            dataset = PreparedQualityDataset(
                heldout,
                manifest,
                expected_heldout_sha256=heldout_sha,
                expected_split_manifest_sha256=manifest_sha,
                expected_train_sha256=train_sha,
            )
            self.assertEqual(len(dataset), 8)
            self.assertEqual(dataset.audit["dataset_counts"], {"qasper": 4, "2wikimqa": 4})
            self.assertFalse(dataset.audit["raw_test_v2_read"])
            self.assertEqual(dataset[0].target_tokens, 2)

            with self.assertRaisesRegex(QualityContractError, "heldout JSONL SHA"):
                PreparedQualityDataset(
                    heldout,
                    manifest,
                    expected_heldout_sha256="0" * 64,
                    expected_split_manifest_sha256=manifest_sha,
                    expected_train_sha256=train_sha,
                )

    def test_paired_ce_uses_token_weighted_checkpoint_gate(self) -> None:
        before = [
            {
                "dataset": "qasper",
                "source_id": "a",
                "target_tokens": 1,
                "ce": 2.0,
                "nll_sum": 2.0,
            },
            {
                "dataset": "2wikimqa",
                "source_id": "b",
                "target_tokens": 3,
                "ce": 1.0,
                "nll_sum": 3.0,
            },
        ]
        after = [
            {**before[0], "ce": 1.0, "nll_sum": 1.0},
            {**before[1], "ce": 0.5, "nll_sum": 1.5},
        ]
        result = paired_quality_comparison(before, after)
        self.assertEqual(result["before"]["overall"]["token_weighted_ce"], 1.25)
        self.assertEqual(result["after"]["overall"]["token_weighted_ce"], 0.625)
        self.assertTrue(result["conditional_model_only_checkpoint_gate"]["passed"])

    def test_longbench_gate_accepts_exact_source_6_to_35_only(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "longbench-validation.jsonl"
            rows = [
                {
                    "dataset": dataset,
                    "_source_index": source_index,
                    "_source_revision": FROZEN_LONGBENCH_REVISION,
                    "context": "context",
                    "input": "question",
                    "answers": ["answer"],
                }
                for dataset in ("qasper", "2wikimqa")
                for source_index in range(6, 36)
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            selected, audit = validate_longbench_validation_rows(
                path, expected_sha256=sha256_file(path)
            )
            self.assertEqual(len(selected), 60)
            self.assertEqual(audit["source_index_start"], 6)
            self.assertEqual(audit["source_index_end"], 35)
            self.assertFalse(audit["raw_test_v2_read"])

            rows[0]["_source_index"] = 68
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(QualityContractError, "only source indices"):
                validate_longbench_validation_rows(
                    path, expected_sha256=sha256_file(path)
                )

    def test_paired_generation_requires_identical_slice(self) -> None:
        base = [
            {
                "dataset": "qasper",
                "source_index": 6,
                "prediction": "base",
                "f1": 0.25,
            },
            {
                "dataset": "2wikimqa",
                "source_index": 6,
                "prediction": "base",
                "f1": 0.5,
            },
        ]
        checkpoint = [
            {**base[0], "prediction": "after", "f1": 0.75},
            {**base[1], "prediction": "after", "f1": 0.5},
        ]
        result = paired_generation_comparison(base, checkpoint)
        self.assertEqual(result["checkpoint_minus_base_mean_f1"], 0.25)
        self.assertFalse(result["raw_test_v2_read"])

        with self.assertRaisesRegex(QualityContractError, "paired slices differ"):
            paired_generation_comparison(base, checkpoint[:1])


if __name__ == "__main__":
    unittest.main()
