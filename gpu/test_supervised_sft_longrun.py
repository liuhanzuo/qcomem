from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from sft_dcp_checkpoint import (
    DCPCheckpointError,
    checkpoint_payload_entries,
    payload_directory_sha256,
)
from supervised_sft_longrun import (
    PreparedScaleExample,
    balanced_global_indices,
    cosine_warmup_factor,
    global_token_weighted_rank_scale,
    heldout_quality_gate,
    schedule_audit,
    summarize_loss_rows,
    validate_scale_split_manifest,
)


class LongrunSFTTests(unittest.TestCase):
    def test_three_epochs_are_balanced_and_deterministic(self) -> None:
        indices = {"qasper": tuple(range(512)), "2wikimqa": tuple(range(512, 1024))}
        schedule = [
            balanced_global_indices(indices, micro_batch_index=step, seed=31)
            for step in range(384)
        ]
        self.assertEqual(
            schedule,
            [
                balanced_global_indices(indices, micro_batch_index=step, seed=31)
                for step in range(384)
            ],
        )
        self.assertTrue(all(len(step) == len(set(step)) == 8 for step in schedule))
        for step in schedule:
            self.assertEqual(sum(index < 512 for index in step), 4)
            self.assertEqual(sum(index >= 512 for index in step), 4)
        counts = {index: 0 for index in range(1024)}
        for step in schedule:
            for index in step:
                counts[index] += 1
        self.assertEqual(set(counts.values()), {3})

    def test_scheduler_warmup_and_cosine_boundaries(self) -> None:
        update_factors = [
            cosine_warmup_factor(step, warmup_steps=20, total_steps=384)
            for step in range(384)
        ]
        self.assertAlmostEqual(update_factors[0], 1 / 20)
        self.assertAlmostEqual(update_factors[19], 1.0)
        self.assertAlmostEqual(update_factors[20], 1.0)
        self.assertGreater(update_factors[-1], 0.0)
        self.assertAlmostEqual(
            cosine_warmup_factor(384, warmup_steps=20, total_steps=384), 0.0
        )
        self.assertTrue(all(0.0 < value <= 1.0 for value in update_factors))

    def test_global_token_weighted_rank_objective(self) -> None:
        tokens = [2, 3, 5, 7]
        losses = [1.0, 2.0, 4.0, 3.0]
        scales = [
            global_token_weighted_rank_scale(
                local_target_tokens=count,
                global_step_target_tokens=sum(tokens),
                world_size=len(tokens),
            )
            for count in tokens
        ]
        rank_average = sum(scale * loss for scale, loss in zip(scales, losses)) / len(tokens)
        token_average = sum(count * loss for count, loss in zip(tokens, losses)) / sum(tokens)
        self.assertAlmostEqual(rank_average, token_average)

    def test_loss_summary_and_quality_gate(self) -> None:
        rows = [
            {"dataset": "qasper", "target_tokens": 2, "mean_ce": 2.0},
            {"dataset": "qasper", "target_tokens": 4, "mean_ce": 1.0},
            {"dataset": "2wikimqa", "target_tokens": 1, "mean_ce": 3.0},
            {"dataset": "2wikimqa", "target_tokens": 3, "mean_ce": 1.0},
        ]
        summary = summarize_loss_rows(rows)
        self.assertAlmostEqual(summary["qasper"]["sample_equal_mean_ce"], 1.5)
        self.assertAlmostEqual(summary["qasper"]["token_weighted_ce"], 8 / 6)
        gate = heldout_quality_gate(
            {
                0: {"overall": {"token_weighted_ce": 2.0}},
                384: {"overall": {"token_weighted_ce": 1.5}},
            },
            final_step=384,
        )
        self.assertTrue(gate["passed"])

    def test_schedule_audit_contains_hashes_not_raw_ids(self) -> None:
        examples = [
            PreparedScaleExample(
                input_ids=None,  # audit reads metadata only
                labels=None,
                dataset="qasper" if index < 4 else "2wikimqa",
                source_id_sha256=hashlib.sha256(str(index).encode()).hexdigest(),
                prompt_tokens=2,
                target_tokens=2,
            )
            for index in range(8)
        ]
        audit = schedule_audit(examples, list(range(8)))
        self.assertEqual(audit["dataset_counts"], {"2wikimqa": 4, "qasper": 4})
        self.assertFalse(audit["raw_source_ids_recorded"])
        self.assertTrue(all(len(value) == 64 for value in audit["source_id_sha256"]))

    def test_payload_directory_digest_is_ordered_and_detects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            (root / "a").write_bytes(b"one")
            (root / "b").write_bytes(b"two")
            (root / "checkpoint-manifest.json").write_text("ignored")
            (root / "_SUCCESS").write_text("ignored")
            entries = checkpoint_payload_entries(root)
            first = payload_directory_sha256(entries)
            self.assertEqual([entry["path"] for entry in entries], ["a", "b"])
            (root / "b").write_bytes(b"changed")
            second = payload_directory_sha256(checkpoint_payload_entries(root))
            self.assertNotEqual(first, second)
            bad = list(reversed(copy.deepcopy(entries)))
            with self.assertRaises(DCPCheckpointError):
                payload_directory_sha256(bad)

    def test_manifest_sha_gate_fails_before_data_use(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            train = root / "train.jsonl"
            heldout = root / "heldout.jsonl"
            manifest = root / "manifest.json"
            train.write_text("{}\n")
            heldout.write_text("{}\n")
            manifest.write_text(json.dumps({"schema_version": "wrong"}))
            with self.assertRaisesRegex(Exception, "SHA256 mismatch"):
                validate_scale_split_manifest(
                    manifest,
                    expected_manifest_sha256="0" * 64,
                    train_path=train,
                    expected_train_sha256="1" * 64,
                    heldout_path=heldout,
                    expected_heldout_sha256="2" * 64,
                )


if __name__ == "__main__":
    unittest.main()
