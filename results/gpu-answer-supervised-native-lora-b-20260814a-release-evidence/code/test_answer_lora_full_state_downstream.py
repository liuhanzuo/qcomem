from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import qcomem_answer_supervised_lora as answer_core
from aggregate_answer_lora_full_state_downstream import aggregate
from qcomem_answer_supervised_lora import (
    FORMAT,
    answer_adapter_config,
    install_and_audit_adapters,
    load_answer_lora_checkpoint,
    load_answer_lora_state_into_installed,
    read_answer_lora_checkpoint,
    sha256_file,
)
from qcomem_lora import lora_state_dict
from run_answer_lora_full_state_downstream import (
    CONDITIONS,
    EXPECTED_POLICIES,
    load_frozen_selection,
    load_validation_slice,
    validate_condition_policies,
)
from test_answer_supervised_native_lora import MockModel, MockSplit


def layer_types() -> list[str]:
    return [
        "full_attention" if index % 4 == 3 else "linear_attention"
        for index in range(40)
    ]


class AnswerLoRAFullStateDownstreamTest(unittest.TestCase):
    def test_condition_policies_include_dense_q16_and_paired_frozen_static(self) -> None:
        validate_condition_policies()
        self.assertEqual(
            CONDITIONS,
            {
                "dense-adapter-disabled-control": ("dense", False, None),
                "q16-adapter-disabled-control": (
                    "replay-d7-layer-q16",
                    False,
                    None,
                ),
                "frozen-static-adapter-disabled": (
                    "replay-d7-frozen-static",
                    False,
                    None,
                ),
                "frozen-static-answer-lora-step0": (
                    "replay-d7-frozen-static",
                    True,
                    0,
                ),
                "frozen-static-answer-lora-step64": (
                    "replay-d7-frozen-static",
                    True,
                    64,
                ),
                "frozen-static-answer-lora-step128": (
                    "replay-d7-frozen-static",
                    True,
                    128,
                ),
            },
        )
        self.assertEqual(
            EXPECTED_POLICIES["replay-d7-frozen-static"]["cache_layer_bits"],
            [8, 8, 8, 4, 8, 8, 8],
        )

    def test_answer_checkpoint_156_module_roundtrip(self) -> None:
        types = layer_types()
        source = MockModel(types)
        installed, surface = install_and_audit_adapters(
            source,
            MockSplit(source, types),
            rank=32,
            alpha=64.0,
            dropout=0.0,
            initialization_seed=20260814,
        )
        state = lora_state_dict(source)
        tiny_parameters = sum(value.numel() for value in state.values())
        with mock.patch.object(
            answer_core, "EXPECTED_ADAPTER_PARAMETERS", tiny_parameters
        ):
            adapter_config = answer_adapter_config(
                source, installed=installed, surface_audit=surface
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                states = {
                    step: {
                        name: value.clone() + step / 128_000
                        for name, value in state.items()
                    }
                    for step in (0, 64, 128)
                }
                records = {}
                for step in (0, 64, 128):
                    checkpoint = root / f"checkpoint-{step:06d}.pt"
                    torch.save(
                        {
                            "format": FORMAT,
                            "step": step,
                            "lora": states[step],
                            "metadata": {
                                "adapter_config": adapter_config,
                                "governance": {
                                    "validation_6_35_used_for_tuning": False,
                                    "test_v2_used": False,
                                    "raw_longbench_validation_or_test_read": False,
                                },
                            },
                        },
                        checkpoint,
                    )
                    records[str(step)] = {
                        "step": step,
                        "path": str(checkpoint),
                        "sha256": sha256_file(checkpoint),
                        "size_bytes": checkpoint.stat().st_size,
                    }
                corrupt_state = {
                    name: value.clone() for name, value in states[0].items()
                }
                first_key = next(iter(corrupt_state))
                corrupt_state[first_key].view(-1)[0] = float("nan")
                corrupt = root / "corrupt-nan-checkpoint.pt"
                torch.save(
                    {
                        "format": FORMAT,
                        "step": 0,
                        "lora": corrupt_state,
                        "metadata": {
                            "adapter_config": adapter_config,
                            "governance": {
                                "validation_6_35_used_for_tuning": False,
                                "test_v2_used": False,
                                "raw_longbench_validation_or_test_read": False,
                            },
                        },
                    },
                    corrupt,
                )
                with self.assertRaisesRegex(Exception, "must all be finite"):
                    read_answer_lora_checkpoint(
                        corrupt,
                        expected_sha256=sha256_file(corrupt),
                        expected_step=0,
                    )
                best_record = {
                    "schema_version": "qcomem_answer_lora_best_checkpoint_v1",
                    "selection_source": (
                        "independent_official_train_heldout_domain_only"
                    ),
                    "selection_direction": "min",
                    "candidate_steps": [0, 64, 128],
                    "selected_step": 64,
                    "checkpoint": records["64"],
                    "validation_6_35_used_for_selection": False,
                    "test_v2_used": False,
                }
                best = root / "best-checkpoint.json"
                best.write_text(json.dumps(best_record))
                (root / "metadata.json").write_text(
                    json.dumps(
                        {
                            "format": FORMAT,
                            "last_step": 128,
                            "best_checkpoint": best_record,
                            "checkpoints": records,
                        }
                    )
                )
                selection = load_frozen_selection(best)
                target = MockModel(types)
                loaded = load_answer_lora_checkpoint(
                    target,
                    MockSplit(target, types),
                    selection["checkpoints"][0]["path"],
                    expected_sha256=selection["checkpoints"][0]["sha256"],
                    expected_step=0,
                )
                swapped = load_answer_lora_state_into_installed(
                    target,
                    selection["checkpoints"][64]["path"],
                    expected_sha256=selection["checkpoints"][64]["sha256"],
                    expected_step=64,
                    expected_adapter_config=loaded["adapter_config"],
                )
        self.assertEqual(loaded["adapter_config"]["installed_module_count"], 156)
        self.assertEqual(swapped["step"], 64)
        self.assertEqual(selection["selected_step"], 64)
        self.assertEqual(len(lora_state_dict(target)), 312)
        for name, value in states[64].items():
            self.assertTrue(torch.equal(value, lora_state_dict(target)[name]))

    def test_selection_rejects_validation_based_checkpoint_choice(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint-000064.pt"
            torch.save({"format": FORMAT, "step": 64}, checkpoint)
            record = {
                "schema_version": "qcomem_answer_lora_best_checkpoint_v1",
                "selection_source": "independent_official_train_heldout_domain_only",
                "selection_direction": "min",
                "candidate_steps": [0, 64, 128],
                "selected_step": 64,
                "checkpoint": {
                    "step": 64,
                    "path": str(checkpoint),
                    "sha256": sha256_file(checkpoint),
                },
                "validation_6_35_used_for_selection": True,
                "test_v2_used": False,
            }
            best = root / "best-checkpoint.json"
            best.write_text(json.dumps(record))
            with self.assertRaisesRegex(Exception, "selection drifted"):
                load_frozen_selection(best)

    def test_test_v2_digest_is_rejected_before_open(self) -> None:
        with self.assertRaisesRegex(Exception, "test-v2"):
            load_validation_slice(
                Path("/does/not/exist.jsonl"),
                expected_sha256=(
                    "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f"
                ),
            )

    @staticmethod
    def shard(condition: str, rank: int) -> dict:
        store, enabled, active_step = CONDITIONS[condition]
        policy = EXPECTED_POLICIES[store]
        all_rows = [
            {
                "dataset": dataset,
                "id": f"{dataset}-{index}",
                "source_index": index,
                "prediction": "x",
                "references": ["yes" if index == 6 else "x"],
                "f1": 0.5 + (0.01 if enabled else 0.0),
                "stored_persistent_nbytes": (
                    None if store == "dense" else 1000
                ),
            }
            for dataset in ("qasper", "2wikimqa")
            for index in range(6, 36)
        ]
        rows = all_rows[rank::8]
        return {
            "schema_version": "qcomem-answer-lora-full-state-shard-v1",
            "rank": rank,
            "config": condition,
            "store_config": store,
            **policy,
            "rows": rows,
            "data_sha256": "a" * 64,
            "selected_checkpoint_step": 64,
            "selected_checkpoint_sha256": "b" * 64,
            "best_checkpoint_record_sha256": "c" * 64,
            "checkpoint_suite_sha256": {
                "0": "d" * 64,
                "64": "b" * 64,
                "128": "e" * 64,
            },
            "adapter_enabled": enabled,
            "active_checkpoint_step": active_step,
            "active_checkpoint_sha256": (
                {0: "d" * 64, 64: "b" * 64, 128: "e" * 64}[active_step]
                if active_step is not None
                else None
            ),
            "resident_checkpoint_step": active_step if active_step is not None else 0,
            "resident_checkpoint_sha256": (
                {0: "d" * 64, 64: "b" * 64, 128: "e" * 64}[
                    active_step if active_step is not None else 0
                ]
            ),
            "condition_is_heldout_selected_alias": active_step == 64,
            "all_checkpoint_steps_evaluated_unconditionally": True,
            "validation_step_results_may_reselect_checkpoint": False,
            "raw_test_v2_read": False,
            "validation_already_consumed": True,
            "selection_or_checkpoint_choice_permitted": False,
            "checkpoint_selection_frozen_before_validation_read": True,
            "checkpoint_selection_source": (
                "independent_official_train_heldout_domain_only"
            ),
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
            "resident_adapter_modules": 156,
            "resident_adapter_parameters": 26_689_536,
            "resident_adapter_parameter_bytes": 106_758_144,
            "resident_adapter_memory_scope": (
                "shared_model_resident_per_process_not_per_document"
            ),
            "adapter_config": {"rank": 32, "alpha": 64.0, "dropout": 0.0},
        }

    def test_aggregate_locks_paired_f1_ci_and_governance(self) -> None:
        shards = [
            self.shard(condition, rank)
            for condition in CONDITIONS
            for rank in range(8)
        ]
        with mock.patch(
            "aggregate_answer_lora_full_state_downstream.bootstrap_mean_ci",
            side_effect=lambda values, seed: [min(values), max(values)],
        ):
            result = aggregate(
                shards, expected_data_sha256="a" * 64, bootstrap_seed=17
            )
        self.assertEqual(result["status"], "completed")
        primary = result["paired_comparisons"][
            "frozen-static-answer-lora-step64_vs_frozen-static-adapter-disabled"
        ]
        self.assertAlmostEqual(primary["mean_f1_delta"], 0.01)
        self.assertEqual(primary["samples"], 60)
        self.assertFalse(
            result["claim_boundaries"]["validation_may_select_checkpoint_or_policy"]
        )
        self.assertEqual(
            result["memory_accounting"]["shared_model_resident_answer_adapter"][
                "bytes"
            ],
            106_758_144,
        )
        self.assertFalse(
            result["memory_accounting"]["adapter_bytes_included_in_per_document_state"]
        )
        self.assertEqual(
            result["two_wiki_selected_vs_disabled_type_transition"]["samples"],
            30,
        )
        self.assertEqual(
            result["heldout_selected_alias"]["condition"],
            "frozen-static-answer-lora-step64",
        )
        self.assertFalse(
            result["heldout_selected_alias"]["additional_forward_executed"]
        )


if __name__ == "__main__":
    unittest.main()
