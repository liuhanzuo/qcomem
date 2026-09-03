from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from aggregate_sft_full_qcomem import (
    finite_memory_summary,
    merge_config,
    storage_summary,
    token_sequence_agreement,
)
from run_sft_full_qcomem_downstream import (
    CONFIGS,
    EXPECTED_POLICIES,
    load_frozen_validation_slice,
    parameter_sample_audit,
    sample_parameter_values,
    validate_frozen_configs,
)
from sft_quality_validation import FROZEN_LONGBENCH_REVISION


class SFTFullQComemDownstreamTests(unittest.TestCase):
    def _write_validation_parent(self, path: Path) -> str:
        rows = [
            {
                "dataset": dataset,
                "_source_index": source_index,
                "_source_revision": FROZEN_LONGBENCH_REVISION,
                "input": "question",
                "context": "document",
                "answers": ["answer"],
                "_id": f"{dataset}-{source_index}",
            }
            for dataset in ("qasper", "2wikimqa")
            for source_index in range(4, 36)
        ]
        payload = "".join(
            json.dumps(row, sort_keys=True) + "\n" for row in rows
        ).encode()
        path.write_bytes(payload)
        return hashlib.sha256(payload).hexdigest()

    def _shards(self, stage: str, config: str) -> list[dict]:
        policy = EXPECTED_POLICIES[config]
        rows = [
            {
                "dataset": dataset,
                "id": f"{dataset}-{index}",
                "source_index": index,
                "prediction": "answer",
                "generated_token_ids": [1, 2],
                "generated_tokens": 2,
                "f1": 1.0,
                "stored_persistent_nbytes": 100 if config != "dense" else None,
                "stored_residual_nbytes": 40 if config != "dense" else None,
                "stored_lower_cache_nbytes": 60 if config != "dense" else None,
            }
            for dataset in ("qasper", "2wikimqa")
            for index in range(6, 36)
        ]
        return [
            {
                "rank": rank,
                "world_size": 8,
                "model_stage": stage,
                "config": config,
                "full_lower_state_qcomem": True,
                **policy,
                "cache_layer_bits": (
                    list(policy["cache_layer_bits"])
                    if policy["cache_layer_bits"] is not None
                    else None
                ),
                "rows": rows[rank::8],
            }
            for rank in range(8)
        ]

    def test_frozen_config_resolution_is_exact(self) -> None:
        validate_frozen_configs()
        self.assertEqual(
            EXPECTED_POLICIES["replay-d7-frozen-static"]["cache_layer_bits"],
            (8, 8, 8, 4, 8, 8, 8),
        )
        self.assertEqual(CONFIGS[0], "dense")

    def test_parent_sha_is_bound_before_calibration_rows_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "validation.jsonl"
            sha256 = self._write_validation_parent(path)
            rows, audit = load_frozen_validation_slice(
                path, expected_sha256=sha256
            )
            self.assertEqual(len(rows), 60)
            self.assertEqual(audit["parent_rows"], 64)
            self.assertEqual(audit["excluded_calibration_source_indices"], [4, 5])
            self.assertEqual(
                {row["_source_index"] for row in rows}, set(range(6, 36))
            )
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                load_frozen_validation_slice(path, expected_sha256="0" * 64)

    def test_merge_requires_full_paired_source_6_to_35(self) -> None:
        shards = self._shards("sft", "replay-d7-layer-q8")
        merged = merge_config(shards, "sft", "replay-d7-layer-q8")
        self.assertEqual(len(merged["rows"]), 60)
        self.assertEqual(merged["mean_f1"], 1.0)
        bad = copy.deepcopy(shards)
        bad[0]["rows"][0]["source_index"] = 68
        with self.assertRaisesRegex(ValueError, "indices are not 6--35"):
            merge_config(bad, "sft", "replay-d7-layer-q8")

    def test_policy_drift_is_rejected(self) -> None:
        shards = self._shards("base", "replay-d7-frozen-static")
        shards[0]["attention_bits"] = 8
        with self.assertRaisesRegex(ValueError, "policy drifted"):
            merge_config(shards, "base", "replay-d7-frozen-static")

    def test_parameter_sample_audit_detects_change(self) -> None:
        model = torch.nn.Linear(8, 4, bias=False).to(torch.bfloat16)
        before = sample_parameter_values(model)
        unchanged = parameter_sample_audit(before, model)
        self.assertEqual(unchanged["changed_elements"], 0)
        with torch.no_grad():
            model.weight.add_(1)
        changed = parameter_sample_audit(before, model)
        self.assertGreater(changed["changed_elements"], 0)
        self.assertEqual(len(changed["bf16_sample_sha256"]), 64)

    def test_storage_and_token_agreement_use_complete_state_rows(self) -> None:
        merged = merge_config(
            self._shards("sft", "replay-d7-frozen-static"),
            "sft",
            "replay-d7-frozen-static",
        )
        summary = storage_summary(merged)
        self.assertEqual(summary["mean_persistent_bytes"], 100)
        self.assertEqual(summary["mean_lower_cache_bytes"], 60)
        self.assertEqual(token_sequence_agreement(merged, merged), 1.0)

    def test_memory_summary_deduplicates_four_configs_per_rank(self) -> None:
        shards = [
            {
                "rank": rank,
                "model_allocated_bytes": 1000 + rank,
                "peak_after_dcp_load_bytes": 2000 + rank,
                "dcp_load_seconds": 3.0 + rank,
            }
            for _config in CONFIGS
            for rank in range(8)
        ]
        summary = finite_memory_summary(shards)
        self.assertEqual(len(summary["model_allocated_bytes_per_rank"]), 8)
        self.assertEqual(summary["max_peak_after_dcp_load_bytes"], 2007)
        bad = copy.deepcopy(shards)
        bad[-1]["model_allocated_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "repeated inconsistent"):
            finite_memory_summary(bad)


if __name__ == "__main__":
    unittest.main()
