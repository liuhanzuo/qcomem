from __future__ import annotations

import copy
import unittest
from pathlib import Path

from split_supervised_sft_scale import (
    DATASETS,
    FINGERPRINT_FIELDS,
    SplitContractError,
    example_fingerprints,
    fingerprint_components,
    load_config,
    select_heldout_components,
    validate_parent_row,
)


class ScaleSplitTests(unittest.TestCase):
    def _row(
        self, dataset: str, source_id: str, context: str, question: str
    ) -> dict:
        fingerprints = example_fingerprints(source_id, context, question)
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
                "fingerprints": fingerprints,
            },
        }

    def test_checked_in_config_is_frozen_and_balanced(self) -> None:
        local_layout = (
            Path(__file__).parents[1]
            / "configs"
            / "supervised_sft_scale_split_512_64.json"
        )
        flattened_remote_layout = (
            Path(__file__).parent
            / "configs"
            / "supervised_sft_scale_split_512_64.json"
        )
        config = load_config(
            local_layout if local_layout.exists() else flattened_remote_layout
        )
        self.assertEqual(set(config["dataset_counts"]), set(DATASETS))
        for counts in config["dataset_counts"].values():
            self.assertEqual(counts["pool"], 576)
            self.assertEqual(counts["train"], 512)
            self.assertEqual(counts["heldout_ce"], 64)

    def test_complete_answer_eos_and_mask_are_validated(self) -> None:
        row = self._row("qasper", "q-1", "paper", "question")
        observed = validate_parent_row(
            row, row_index=0, eos_token_id=99, max_sequence_tokens=1024
        )
        self.assertEqual(set(observed), set(FINGERPRINT_FIELDS))

        broken = copy.deepcopy(row)
        broken["answer_input_ids"][-1] = 98
        broken["input_ids"][-1] = 98
        broken["labels"][-1] = 98
        with self.assertRaisesRegex(SplitContractError, "answer plus EOS"):
            validate_parent_row(
                broken, row_index=0, eos_token_id=99, max_sequence_tokens=1024
            )

    def test_every_fingerprint_collision_is_one_global_component(self) -> None:
        rows = [
            self._row("qasper", "q-0", "q context", "shared question"),
            self._row("2wikimqa", "w-0", "w context", "shared question"),
            self._row("qasper", "q-1", "q other", "q only"),
            self._row("2wikimqa", "w-1", "w other", "w only"),
        ]
        fingerprints = [row["provenance"]["fingerprints"] for row in rows]
        components = fingerprint_components(rows, fingerprints, "frozen-test-salt")
        shared = [component for component in components if set(component["indices"]) == {0, 1}]
        self.assertEqual(len(shared), 1)
        self.assertEqual(shared[0]["counts"], (1, 1))

        selected = select_heldout_components(
            components, qasper_count=1, twowiki_count=1
        )
        heldout_indices = {
            index
            for component in components
            if component["digest"] in selected
            for index in component["indices"]
        }
        self.assertEqual(
            sum(rows[index]["dataset"] == "qasper" for index in heldout_indices), 1
        )
        self.assertEqual(
            sum(rows[index]["dataset"] == "2wikimqa" for index in heldout_indices),
            1,
        )
        for component in components:
            member_splits = {
                index in heldout_indices for index in component["indices"]
            }
            self.assertEqual(len(member_splits), 1)

    def test_non_train_parent_row_is_rejected(self) -> None:
        row = self._row("2wikimqa", "w-2", "context", "question")
        row["source_split"] = "validation"
        with self.assertRaisesRegex(SplitContractError, "official train"):
            validate_parent_row(
                row, row_index=0, eos_token_id=99, max_sequence_tokens=1024
            )


if __name__ == "__main__":
    unittest.main()
