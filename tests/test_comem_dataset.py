from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from macllm_bench.comem_dataset import (
    BM25Selector,
    Document,
    evidence_recall,
    load_dataset,
)


class CoMemDatasetTest(unittest.TestCase):
    def test_demo_dataset_and_frozen_references_are_valid(self) -> None:
        dataset = load_dataset(Path("configs/comem_multidoc_demo.json"))
        self.assertEqual(len(dataset.documents), 6)
        self.assertEqual(len(dataset.queries), 6)
        self.assertTrue(all(query.expected_answer for query in dataset.queries))
        self.assertTrue(
            all(
                evidence_recall(
                    query.selected_document_ids, query.relevant_document_ids
                )
                == 1.0
                for query in dataset.queries
            )
        )

    def test_bm25_prefers_matching_document(self) -> None:
        documents = [
            Document("gpu", "Metal GPU", "unified memory and GPU streams"),
            Document("fruit", "Orchard", "apple pear banana fruit"),
        ]
        selector = BM25Selector(documents)
        self.assertEqual(selector.select("GPU stream memory", 1), ["gpu"])

    def test_unknown_document_reference_is_rejected(self) -> None:
        payload = {
            "name": "broken",
            "documents": [{"id": "a", "title": "A", "text": "text"}],
            "queries": [
                {
                    "id": "q",
                    "query": "question",
                    "relevant_document_ids": ["missing"],
                    "selected_document_ids": ["a"],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dataset.json"
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "unknown documents"):
                load_dataset(path)


if __name__ == "__main__":
    unittest.main()
