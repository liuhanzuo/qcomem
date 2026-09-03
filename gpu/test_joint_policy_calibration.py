from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from qcomem_joint_policy import (
    FROZEN_STATIC_LAYER_BITS,
    JointPolicy,
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
    frozen_static_policy,
    logit_metric_sums,
    merge_metric_sums,
    policy_for_component,
    q16_exactness_passes,
    selected_query_positions,
    top_predicted_policies,
)


class TinyTokenizer:
    def encode(self, text: str, **kwargs):
        ids = [int(value) for value in text.split()]
        maximum = kwargs.get("max_length")
        return ids if maximum is None else ids[:maximum]


def write_pg19_fixture(directory: Path, books: int = 4) -> tuple[Path, Path, str, str]:
    data = directory / "pg19_train.jsonl"
    manifest = directory / "pg19_train.manifest.json"
    rows = []
    objects = []
    for index in range(books):
        name = f"train/{100 + index}.txt"
        md5 = f"md5-{index}"
        rows.append(
            {
                "id": f"book-{index}",
                "text": " ".join(str(value) for value in range(80)),
                "_source_bucket": "deepmind-gutenberg",
                "_source_object": name,
                "_source_md5_base64": md5,
            }
        )
        objects.append({"name": name, "md5_base64": md5})
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
    manifest.write_text(
        json.dumps(
            {
                "bucket": "deepmind-gutenberg",
                "prefix": "train/",
                "test_or_validation_objects_used": False,
                "jsonl_sha256": data_sha,
                "objects": objects,
            }
        )
        + "\n"
    )
    manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return data, manifest, data_sha, manifest_sha


class PG19ProtocolTest(unittest.TestCase):
    def test_valid_pg19_train_fixture_passes_and_selects_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data, manifest, data_sha, manifest_sha = write_pg19_fixture(
                Path(directory)
            )
            records, audit = audit_pg19_train_calibration(
                data,
                manifest,
                expected_data_sha256=data_sha,
                expected_manifest_sha256=manifest_sha,
                minimum_books=4,
            )
            first, first_digest = build_pg19_calibration_windows(
                records,
                TinyTokenizer(),
                books=3,
                document_tokens=8,
                query_tokens=4,
                stride=4,
                candidate_windows_per_book=3,
                seed=7,
            )
            second, second_digest = build_pg19_calibration_windows(
                records,
                TinyTokenizer(),
                books=3,
                document_tokens=8,
                query_tokens=4,
                stride=4,
                candidate_windows_per_book=3,
                seed=7,
            )
        self.assertEqual(first_digest, second_digest)
        self.assertEqual(
            [(item.source_object, item.start_token) for item in first],
            [(item.source_object, item.start_token) for item in second],
        )
        self.assertFalse(audit["longbench_labels_used"])
        self.assertFalse(audit["formal_validation_source_6_35_used"])
        self.assertFalse(audit["frozen_test_v2_source_68_99_used"])

    def test_qa_schema_is_rejected_even_under_pg19_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data, manifest, _, _ = write_pg19_fixture(root, books=1)
            record = json.loads(data.read_text().strip())
            record.update({"dataset": "qasper", "_source_index": 6})
            data.write_text(json.dumps(record) + "\n")
            data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
            payload = json.loads(manifest.read_text())
            payload["jsonl_sha256"] = data_sha
            manifest.write_text(json.dumps(payload) + "\n")
            manifest_sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "evaluation/QA schema"):
                audit_pg19_train_calibration(
                    data,
                    manifest,
                    expected_data_sha256=data_sha,
                    expected_manifest_sha256=manifest_sha,
                    minimum_books=1,
                )

    def test_longbench_named_path_is_rejected_before_schema_use(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, manifest, _, manifest_sha = write_pg19_fixture(root, books=1)
            forbidden = root / "longbench_validation.jsonl"
            forbidden.write_text("{}\n")
            digest = hashlib.sha256(forbidden.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError, "PG-19 train paths only"):
                audit_pg19_train_calibration(
                    forbidden,
                    manifest,
                    expected_data_sha256=digest,
                    expected_manifest_sha256=manifest_sha,
                    minimum_books=1,
                )

    def test_query_positions_never_consume_missing_next_token(self) -> None:
        positions = selected_query_positions(128, 8)
        self.assertEqual(len(positions), 8)
        self.assertGreaterEqual(min(positions), 0)
        self.assertLessEqual(max(positions), 126)


class JointMetricAndPolicyTest(unittest.TestCase):
    def test_exact_logits_have_zero_objective_and_pass_q16_gate(self) -> None:
        torch.manual_seed(1)
        teacher = torch.randn(4, 11)
        targets = torch.tensor([1, 2, 3, 4])
        sums = logit_metric_sums(teacher, teacher.clone(), targets)
        summary = merge_metric_sums([sums])
        self.assertAlmostEqual(summary["joint_objective"], 0.0, places=7)
        self.assertTrue(q16_exactness_passes(summary))

    def test_joint_objective_penalizes_distribution_and_continuation_harm(self) -> None:
        teacher = torch.tensor([[5.0, 0.0, -1.0], [0.0, 4.0, -2.0]])
        candidate = torch.tensor([[0.0, 5.0, -1.0], [4.0, 0.0, -2.0]])
        targets = torch.tensor([0, 1])
        summary = merge_metric_sums(
            [logit_metric_sums(teacher, candidate, targets)]
        )
        self.assertGreater(summary["mean_forward_kl"], 0)
        self.assertGreater(summary["mean_positive_nll_delta"], 0)
        self.assertEqual(summary["top1_agreement"], 0)
        self.assertGreater(summary["joint_objective"], 0.1)

    def test_frozen_and_component_policy_bits_are_explicit(self) -> None:
        frozen = frozen_static_policy(7)
        self.assertEqual(frozen.residual_bits, 4)
        self.assertEqual(frozen.cache_layer_bits, FROZEN_STATIC_LAYER_BITS)
        cache = policy_for_component(3, 4, depth=7)
        self.assertEqual(cache.residual_bits, 16)
        self.assertEqual(cache.cache_layer_bits[2], 4)
        self.assertEqual(sum(bits != 16 for bits in cache.cache_layer_bits), 1)
        with self.assertRaises(ValueError):
            JointPolicy("bad", 3, (16,) * 7, "test")

    def test_candidate_enumeration_respects_budget_and_exclusions(self) -> None:
        profiles = []
        for index in range(8):
            profiles.append(
                {
                    "component": "residual" if index == 0 else f"cache.{index - 1}",
                    "options": [
                        {
                            "bits": bits,
                            "mean_component_nbytes": bits * 10 + index,
                            "metrics": {"joint_objective": 1.0 / bits + index / 1000},
                        }
                        for bits in (2, 4, 8, 16)
                    ],
                }
            )
        excluded = (4,) * 8
        rows = top_predicted_policies(
            profiles,
            budget_bytes=8 * 80 + sum(range(8)),
            limit=5,
            excluded_bits=[excluded],
        )
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(byte_count <= 8 * 80 + sum(range(8)) for _, byte_count, _ in rows))
        self.assertTrue(all(bits != excluded for bits, _, _ in rows))


if __name__ == "__main__":
    unittest.main()
