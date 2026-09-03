from __future__ import annotations

import copy
import unittest

from aggregate_native_lora_checkpoint_attribution import (
    ANSWER_TYPES,
    aggregate,
    answer_type,
    answer_type_confusion,
    paired_comparison,
)
from run_native_lora_checkpoint_attribution import (
    ACTIVE_STEP,
    ATTRIBUTION_CONFIGS,
    EXPECTED_STORE_POLICY,
    STORE_CONFIG,
    validate_store_policy,
)


CHECKPOINTS = {0: "0" * 64, 64: "6" * 64, 128: "8" * 64}
DATA_SHA = "d" * 64


def sample_row(dataset: str, source_index: int, condition: str) -> dict:
    base_f1 = 0.5 if dataset == "qasper" else 0.6
    increment = {
        "adapter-disabled": 0.0,
        "native-lora-step0": 0.01,
        "native-lora-step64": 0.02,
        "native-lora-step128": 0.03,
    }[condition]
    if dataset == "2wikimqa" and source_index == 6:
        reference = "yes"
    elif dataset == "2wikimqa" and source_index == 7:
        reference = "no"
    elif dataset == "2wikimqa" and source_index == 8:
        reference = "unanswerable"
    else:
        reference = f"entity-{source_index}"
    token = 10 + ATTRIBUTION_CONFIGS.index(condition)
    return {
        "dataset": dataset,
        "id": f"{dataset}-{source_index}",
        "source_index": source_index,
        "references": [reference],
        "prediction": reference,
        "generated_token_ids": [token, source_index],
        "generated_tokens": 2,
        "f1": base_f1 + increment,
        "max_new_tokens": 128 if dataset == "qasper" else 32,
    }


def make_shards() -> list[dict]:
    all_rows = [
        sample_row(dataset, source_index, condition)
        for condition in ATTRIBUTION_CONFIGS
        for dataset in ("qasper", "2wikimqa")
        for source_index in range(6, 36)
    ]
    shards = []
    for condition in ATTRIBUTION_CONFIGS:
        condition_rows = [row for row in all_rows if row["generated_token_ids"][0] == 10 + ATTRIBUTION_CONFIGS.index(condition)]
        for rank in range(8):
            step = ACTIVE_STEP[condition]
            checkpoint = {
                "adapter_enabled": step is not None,
                "active_checkpoint_step": step,
                "active_checkpoint": None if step is None else f"step-{step}.pt",
                "active_checkpoint_sha256": None if step is None else CHECKPOINTS[step],
                "resident_disabled_adapter_checkpoint_step": 0 if step is None else None,
                "resident_disabled_adapter_checkpoint": "step-0.pt" if step is None else None,
                "resident_disabled_adapter_checkpoint_sha256": CHECKPOINTS[0] if step is None else None,
            }
            shards.append(
                {
                    "schema_version": "qcomem-native-lora-checkpoint-attribution-shard-v1",
                    "config": condition,
                    "store_config": STORE_CONFIG,
                    **{
                        key: list(value) if key == "cache_layer_bits" else value
                        for key, value in EXPECTED_STORE_POLICY.items()
                    },
                    "rank": rank,
                    "world_size": 8,
                    "rows": condition_rows[rank::8],
                    "data_sha256": DATA_SHA,
                    "raw_test_v2_read": False,
                    "validation_already_consumed": True,
                    "selection_or_checkpoint_choice_permitted": False,
                    "attribution_only": True,
                    "prompt_protocol": "longbench-v1-official",
                    "caller": "run_replay_diagnostic.run_config/full_state_replay",
                    "decoding": "greedy_argmax",
                    "source_index_start": 6,
                    "source_index_end": 35,
                    "excluded_source_indices": [4, 5],
                    "dataset_max_new_tokens": {"qasper": 128, "2wikimqa": 32},
                    "max_input_tokens": 4096,
                    "max_new_tokens": 128,
                    "group_size": 64,
                    "checkpoint": checkpoint,
                    "checkpoint_ledger": {
                        str(step): {"path": f"step-{step}.pt", "sha256": digest}
                        for step, digest in CHECKPOINTS.items()
                    },
                }
            )
    return shards


class NativeLoRACheckpointAttributionTest(unittest.TestCase):
    def test_store_policy_is_frozen_q4_q8(self) -> None:
        validate_store_policy()
        self.assertEqual(EXPECTED_STORE_POLICY["residual_bits"], 4)
        self.assertEqual(EXPECTED_STORE_POLICY["attention_bits"], 4)
        self.assertEqual(EXPECTED_STORE_POLICY["linear_bits"], 8)
        self.assertEqual(
            EXPECTED_STORE_POLICY["cache_layer_bits"],
            (8, 8, 8, 4, 8, 8, 8),
        )

    def test_answer_type_rule_and_confusion(self) -> None:
        self.assertEqual(answer_type(" Yes. "), "yes")
        self.assertEqual(answer_type("NO"), "no")
        self.assertEqual(answer_type("unanswerable"), "abstain")
        self.assertEqual(answer_type("Ada Lovelace"), "entity")
        rows = [
            {"references": [label], "prediction": label}
            for label in ("yes", "no", "entity", "unanswerable")
        ]
        result = answer_type_confusion(rows)
        self.assertEqual(result["labels"], list(ANSWER_TYPES))
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["type_exact_accuracy"], 1.0)

    def test_paired_comparison_reports_f1_and_token_agreement(self) -> None:
        reference = {
            "config": "disabled",
            "rows": [sample_row("qasper", 6, "adapter-disabled")],
        }
        candidate = {
            "config": "step0",
            "rows": [sample_row("qasper", 6, "native-lora-step0")],
        }
        result = paired_comparison(candidate, reference, seed=7)
        self.assertAlmostEqual(result["mean_f1_delta"], 0.01)
        self.assertEqual(result["prediction_exact_agreement"], 1.0)
        self.assertEqual(result["token_sequence_exact_agreement"], 0.0)
        self.assertEqual(result["mean_token_position_agreement"], 0.5)

    def test_aggregate_requires_all_paired_rows_and_marks_consumed_validation(self) -> None:
        result = aggregate(
            make_shards(),
            expected_data_sha256=DATA_SHA,
            expected_checkpoint_sha256=CHECKPOINTS,
            source_job_id=235749,
            source_trial_id=1834056,
            bootstrap_seed=17,
        )
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["samples"], 60)
        self.assertTrue(result["claim_boundaries"]["validation_already_consumed"])
        self.assertFalse(
            result["claim_boundaries"]["validation_may_select_checkpoint_or_policy"]
        )
        self.assertFalse(result["claim_boundaries"]["raw_test_v2_read"])
        self.assertEqual(
            result["two_wiki_answer_type_confusion"]["adapter-disabled"]["samples"],
            30,
        )

    def test_aggregate_fails_closed_on_governance_or_checkpoint_drift(self) -> None:
        for mutate in ("test_v2", "checkpoint"):
            shards = copy.deepcopy(make_shards())
            if mutate == "test_v2":
                shards[0]["raw_test_v2_read"] = True
            else:
                shards[0]["checkpoint_ledger"]["64"]["sha256"] = "f" * 64
            with self.assertRaises(ValueError):
                aggregate(
                    shards,
                    expected_data_sha256=DATA_SHA,
                    expected_checkpoint_sha256=CHECKPOINTS,
                    source_job_id=235749,
                    source_trial_id=1834056,
                    bootstrap_seed=17,
                )


if __name__ == "__main__":
    unittest.main()
