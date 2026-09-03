from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from supervised_sft import (
    IGNORE_INDEX,
    DenseSupervisedCausalLM,
    SupervisedSFTDataset,
    TinyLanguageModel,
    answer_only_causal_ce,
    build_supervised_example,
    configure_dense_full_model_trainability,
    file_sha256,
    qcomem_suffix_supervised_sft_capability_gate,
    validate_prepared_training_manifest,
    validate_runtime_tokenizer_against_manifest,
    validate_supervised_row,
)
from train_supervised_sft import (
    require_finite_positive_grad_norm,
    validate_formal_smoke_dataset_counts,
)
from preflight_supervised_sft import FROZEN_CONFIG, preflight_formal_supervised_smoke


class CharacterTokenizer:
    eos_token_id = 2
    vocab_size = 214
    chat_template = "character-tokenizer-template-v1"

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool = False,
    ) -> str:
        del tokenize, add_generation_prompt, enable_thinking
        return messages[0]["content"] + "\n<assistant>\n"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [3 + (ord(character) % 211) for character in text]


def training_row(
    *,
    source_id: str = "train-0",
    context: str = "context",
    split: str = "train",
    dataset: str = "2wikimqa",
) -> dict:
    return {
        "dataset": dataset,
        "source_split": split,
        "source_id": source_id,
        "context": context,
        "input": "Who wrote the work?",
        "answers": ["Ada", "Ada Lovelace"],
        "selected_answer": "Ada",
        "provenance": {
            "source_repo": "example/dataset",
            "source_split": split,
            "source_record_index": 0,
        },
    }


class SupervisedSFTTest(unittest.TestCase):
    @staticmethod
    def _integrity_ledgers(
        root: Path, config_path: Path
    ) -> tuple[Path, str, Path, str, Path]:
        code_files = [
            Path(__file__).with_name("supervised_sft.py"),
            Path(__file__).with_name("train_supervised_sft.py"),
            Path(__file__).with_name("preflight_supervised_sft.py"),
            Path(__file__).with_name("launch_supervised_sft_8gpu.sh"),
            config_path,
            Path(__file__).with_name("run_downstream.py"),
        ]
        code_ledger = root / "code.sha256"
        code_ledger.write_text(
            "".join(f"{file_sha256(path)}  {path.resolve()}\n" for path in code_files)
        )

        model_path = root / "Qwen3.5-35B-A3B-59d61f3"
        model_path.mkdir()
        model_files = []
        for filename in (
            "config.json",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "chat_template.jinja",
        ):
            path = model_path / filename
            path.write_text(f"synthetic-{filename}\n")
            model_files.append(path)
        model_ledger = root / "model-artifacts.sha256"
        model_ledger.write_text(
            "".join(f"{file_sha256(path)}  {path.resolve()}\n" for path in model_files)
        )
        return (
            code_ledger,
            file_sha256(code_ledger),
            model_ledger,
            file_sha256(model_ledger),
            model_path,
        )

    @staticmethod
    def _manifest(data_path: Path) -> dict:
        dataset_spec = {
            "source_split": "train",
            "source_revision": "frozen-revision",
            "archive_sha256": "a" * 64,
            "extracted_file_sha256": "b" * 64,
            "license": "research",
        }
        return {
            "schema_version": "qcomem-supervised-qa-v1",
            "status": "passed",
            "mode": "build",
            "tokenizer": {
                "requested_name_or_path": (
                    "/models/Qwen3.5-35B-A3B-59d61f3"
                ),
                "requested_revision": "59d61f3",
                "resolved_commit_hash": None,
                "class": "CharacterTokenizer",
                "vocab_size": CharacterTokenizer.vocab_size,
                "eos_token_id": CharacterTokenizer.eos_token_id,
                "chat_template_sha256": hashlib.sha256(
                    CharacterTokenizer.chat_template.encode("utf-8")
                ).hexdigest(),
            },
            "prompt_protocol": {
                "function": "run_downstream.prompt_parts",
                "source_file_sha256": file_sha256(
                    Path(__file__).with_name("run_downstream.py")
                ),
                "target_builder": "supervised_sft.build_supervised_example",
                "target_builder_source_file_sha256": file_sha256(
                    Path(__file__).with_name("supervised_sft.py")
                ),
                "max_sequence_tokens": 1024,
                "answer_tokens_reserved_before_prompt_truncation": True,
                "label_ignore_index": -100,
                "answer_eos_appended": True,
            },
            "output_jsonl": str(data_path),
            "output_jsonl_sha256": file_sha256(data_path),
            "detected_overlap_count": 0,
            "output_overlap_count": 0,
            "overlap_report": [],
            "dataset_stats": {
                "qasper": {
                    "parsed_examples": 4,
                    "overlap_examples": 0,
                    "dropped_examples": 0,
                    "eligible_examples": 4,
                    "full_eligible_examples": 4,
                    "selected_for_output_examples": 4,
                    "written_examples": 4,
                    "output_selection_skipped_answer_over_cap": 0,
                    "output_selection_skipped_answer_over_cap_source_id_sha256": [],
                },
                "2wikimqa": {
                    "parsed_examples": 4,
                    "overlap_examples": 0,
                    "dropped_examples": 0,
                    "eligible_examples": 4,
                    "full_eligible_examples": 4,
                    "selected_for_output_examples": 4,
                    "written_examples": 4,
                    "output_selection_skipped_answer_over_cap": 0,
                    "output_selection_skipped_answer_over_cap_source_id_sha256": [],
                },
            },
            "output_selection": {
                "strategy": (
                    "first_n_target_valid_eligible_in_official_source_order-v1"
                ),
                "requested_max_output_per_dataset": 4,
                "max_output_per_dataset": 4,
                "full_train_scan_completed": True,
                "selection_applied_after_overlap_filter": True,
                "target_validity_checked_before_selection": True,
                "answer_over_cap_policy": "skip_complete_answer_without_truncation",
                "skipped_answer_over_cap": {
                    "qasper": {"count": 0, "source_id_sha256": []},
                    "2wikimqa": {"count": 0, "source_id_sha256": []},
                },
                "written_smoke_count": 8,
                "written_jsonl_count": 8,
            },
            "heldout_protocol": {
                "test_v2_content_hash_check": "deferred_not_read",
                "raw_test_v2_read_by_converter": False,
                "overlap_policy": "fail",
            },
            "source_spec": {
                "datasets": {
                    "qasper": dict(dataset_spec),
                    "2wikimqa": dict(dataset_spec),
                }
            },
        }

    def test_label_mask_preserves_answer_and_one_eos_under_context_truncation(self) -> None:
        tokenizer = CharacterTokenizer()
        row = training_row(context="C" * 4000)
        example = build_supervised_example(
            tokenizer,
            row,
            max_sequence_tokens=1400,
        )
        answer_ids = tokenizer.encode("Ada", add_special_tokens=False)
        expected_targets = torch.tensor(answer_ids + [tokenizer.eos_token_id])

        self.assertTrue(example.context_was_truncated)
        self.assertEqual(example.sequence_tokens, 1400)
        self.assertTrue(
            torch.all(example.labels[: example.prompt_tokens] == IGNORE_INDEX)
        )
        self.assertTrue(
            torch.equal(example.labels[example.prompt_tokens :], expected_targets)
        )
        self.assertTrue(torch.equal(example.input_ids[-len(expected_targets) :], expected_targets))
        self.assertEqual(int((example.labels != IGNORE_INDEX).sum()), len(expected_targets))
        self.assertEqual(int(example.labels[-1]), tokenizer.eos_token_id)
        self.assertEqual(
            int((example.labels == tokenizer.eos_token_id).sum()),
            1,
        )

    def test_causal_ce_uses_predecessor_logits_without_off_by_one(self) -> None:
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4]])
        logits = torch.full((1, 4, 6), -10.0)
        logits[0, 1, 3] = 10.0  # final prompt token predicts first answer token
        logits[0, 2, 4] = 10.0  # answer token predicts EOS
        loss, targets = answer_only_causal_ce(logits, labels)
        self.assertEqual(targets, 2)
        self.assertLess(float(loss), 1e-6)

        unused_final_position = logits.clone()
        unused_final_position[0, 3, :] = torch.arange(6, dtype=torch.float32) * 100
        unchanged, _ = answer_only_causal_ce(unused_final_position, labels)
        self.assertAlmostEqual(float(loss), float(unchanged), places=7)

        wrong_shift = torch.full((1, 4, 6), -10.0)
        wrong_shift[0, 2, 3] = 10.0
        wrong_shift[0, 3, 4] = 10.0
        wrong_loss, _ = answer_only_causal_ce(wrong_shift, labels)
        self.assertGreater(float(wrong_loss), 5.0)

    def test_answer_only_loss_backpropagates_through_full_tiny_model(self) -> None:
        torch.manual_seed(3)
        language_model = TinyLanguageModel(vocabulary=32, width=8)
        lm_head = nn.Linear(8, 32, bias=False)
        core = DenseSupervisedCausalLM(language_model, lm_head)
        labels = torch.tensor(
            [[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 7, 2]]
        )
        loss = core(
            torch.tensor([[4, 5, 6, 7, 2]]),
            labels,
            torch.ones((1, 5), dtype=torch.long),
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        parameters = list(core.parameters())
        self.assertTrue(parameters)
        self.assertTrue(all(parameter.requires_grad for parameter in parameters))
        self.assertTrue(all(parameter.grad is not None for parameter in parameters))
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in parameters))
        self.assertGreater(
            sum(float(parameter.grad.abs().sum()) for parameter in parameters),
            0.0,
        )

    def test_full_model_parameter_plan_is_exact_and_not_lora(self) -> None:
        model = nn.Sequential(nn.Linear(4, 5), nn.LayerNorm(5), nn.Linear(5, 7))
        model[0].weight.requires_grad_(False)
        expected = sum(parameter.numel() for parameter in model.parameters())
        plan = configure_dense_full_model_trainability(model)

        self.assertEqual(plan["trainable_parameters"], expected)
        self.assertEqual(plan["total_parameters"], expected)
        self.assertTrue(plan["all_model_parameters_trainable"])
        self.assertFalse(plan["lora_used"])
        self.assertFalse(plan["distillation_used"])
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

    def test_schema_rejects_nontrain_and_selected_answer_drift(self) -> None:
        row = training_row(split="validation")
        with self.assertRaisesRegex(ValueError, "exactly 'train'"):
            validate_supervised_row(row, line_number=1)

        row = training_row()
        row["selected_answer"] = "not in candidates"
        with self.assertRaisesRegex(ValueError, "occur verbatim"):
            validate_supervised_row(row, line_number=2)

    def test_dataset_validates_heldout_rows_beyond_smoke_limit(self) -> None:
        tokenizer = CharacterTokenizer()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            rows = [
                training_row(source_id="train-0", context="C" * 1000),
                training_row(
                    source_id="heldout-1", context="D" * 1000, split="test"
                ),
            ]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "exactly 'train'"):
                SupervisedSFTDataset(
                    path,
                    tokenizer,
                    max_sequence_tokens=900,
                    limit=1,
                )

    def test_qcomem_suffix_supervised_sft_is_fail_closed(self) -> None:
        gate = qcomem_suffix_supervised_sft_capability_gate()
        self.assertFalse(gate["implemented"])
        self.assertFalse(gate["capability_gate_passed"])
        self.assertTrue(gate["fail_closed"])
        self.assertFalse(
            gate["capabilities"][
                "answer_chunk_then_stepwise_decode_equivalence_validated"
            ]
        )
        self.assertEqual(gate["observed_blocker"]["trial_id"], 1830867)
        self.assertFalse(gate["observed_blocker"]["dense_full_model_sft_affected"])

    def test_one_step_requires_finite_positive_global_grad_norm(self) -> None:
        self.assertEqual(require_finite_positive_grad_norm(torch.tensor(1.25)), 1.25)
        for value in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(
                RuntimeError, "no valid gradient"
            ):
                require_finite_positive_grad_norm(value)

    def test_formal_smoke_requires_four_examples_per_dataset(self) -> None:
        validate_formal_smoke_dataset_counts({"qasper": 4, "2wikimqa": 4})
        for counts in (
            {"qasper": 8},
            {"qasper": 5, "2wikimqa": 3},
            {"qasper": 400, "2wikimqa": 400},
        ):
            with self.subTest(counts=counts), self.assertRaisesRegex(
                ValueError, "exactly four"
            ):
                validate_formal_smoke_dataset_counts(counts)

    def test_formal_manifest_binds_jsonl_and_heldout_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "supervised-train.jsonl"
            data_path.write_text("eight formal rows\n")
            manifest_path = Path(directory) / "supervised-train.manifest.json"
            manifest = self._manifest(data_path)
            manifest_path.write_text(json.dumps(manifest))

            audit = validate_prepared_training_manifest(
                manifest_path,
                data_path,
                expected_data_sha256=file_sha256(data_path),
                expected_manifest_sha256=file_sha256(manifest_path),
            )
            self.assertEqual(audit["detected_overlap_count"], 0)
            self.assertEqual(audit["output_overlap_count"], 0)
            self.assertEqual(audit["test_v2_content_hash_check"], "deferred_not_read")
            self.assertFalse(audit["raw_test_v2_read_by_converter"])
            self.assertEqual(
                audit["dataset_written_examples"],
                {"2wikimqa": 4, "qasper": 4},
            )

            manifest["output_overlap_count"] = 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "integer zero"):
                validate_prepared_training_manifest(
                    manifest_path,
                    data_path,
                    expected_data_sha256=file_sha256(data_path),
                    expected_manifest_sha256=file_sha256(manifest_path),
                )

    def test_drop_policy_preserves_detected_count_but_publishes_clean_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "supervised-train.jsonl"
            data_path.write_text("eight cleaned rows\n")
            manifest_path = Path(directory) / "supervised-train.manifest.json"
            manifest = self._manifest(data_path)
            manifest["heldout_protocol"]["overlap_policy"] = "drop"
            manifest["detected_overlap_count"] = 3
            manifest["overlap_report"] = [
                {
                    "dataset": "qasper",
                    "train_source_id_sha256": str(index) * 64,
                    "matches": [
                        {
                            "fingerprint_kind": "id_sha256",
                            "dataset": "qasper",
                            "split": "validation",
                            "source_index": index,
                        }
                    ],
                }
                for index in (1, 2, 3)
            ]
            manifest["dataset_stats"]["qasper"] = {
                "parsed_examples": 13,
                "overlap_examples": 3,
                "dropped_examples": 3,
                "eligible_examples": 10,
                "full_eligible_examples": 10,
                "selected_for_output_examples": 4,
                "written_examples": 4,
                "output_selection_skipped_answer_over_cap": 1,
                "output_selection_skipped_answer_over_cap_source_id_sha256": [
                    "c" * 64
                ],
            }
            manifest["output_selection"]["skipped_answer_over_cap"]["qasper"] = {
                "count": 1,
                "source_id_sha256": ["c" * 64],
            }
            manifest_path.write_text(json.dumps(manifest))
            audit = validate_prepared_training_manifest(
                manifest_path,
                data_path,
                expected_data_sha256=file_sha256(data_path),
                expected_manifest_sha256=file_sha256(manifest_path),
            )
            self.assertEqual(audit["detected_overlap_count"], 3)
            self.assertEqual(audit["dataset_dropped_examples"]["qasper"], 3)
            self.assertEqual(audit["dataset_full_eligible_examples"]["qasper"], 10)
            self.assertEqual(
                audit["dataset_output_selection_skipped_answer_over_cap"]["qasper"],
                1,
            )
            self.assertEqual(audit["output_overlap_count"], 0)

            manifest["dataset_stats"]["qasper"]["dropped_examples"] = 2
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "dropped_examples"):
                validate_prepared_training_manifest(
                    manifest_path,
                    data_path,
                    expected_data_sha256=file_sha256(data_path),
                    expected_manifest_sha256=file_sha256(manifest_path),
                )

    def test_cpu_preflight_accepts_exact_four_plus_four_formal_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "formal-train.jsonl"
            rows = [
                training_row(source_id=f"qasper-{index}", dataset="qasper")
                for index in range(4)
            ] + [
                training_row(source_id=f"2wiki-{index}", dataset="2wikimqa")
                for index in range(4)
            ]
            data_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            manifest_path = root / "formal-train.manifest.json"
            manifest_path.write_text(json.dumps(self._manifest(data_path)))
            config_path = root / "dense_full_model_sft_smoke_1.json"
            config_path.write_text(json.dumps(FROZEN_CONFIG))
            (
                code_ledger,
                code_ledger_sha256,
                model_ledger,
                model_ledger_sha256,
                model_path,
            ) = self._integrity_ledgers(root, config_path)

            result = preflight_formal_supervised_smoke(
                data_path=data_path,
                manifest_path=manifest_path,
                config_path=config_path,
                expected_data_sha256=file_sha256(data_path),
                expected_manifest_sha256=file_sha256(manifest_path),
                tokenizer=CharacterTokenizer(),
                model_path=model_path,
                code_ledger_path=code_ledger,
                expected_code_ledger_sha256=code_ledger_sha256,
                model_ledger_path=model_ledger,
                expected_model_ledger_sha256=model_ledger_sha256,
            )
            self.assertEqual(result["status"], "passed")
            self.assertEqual(
                result["dataset_counts"], {"qasper": 4, "2wikimqa": 4}
            )
            self.assertTrue(result["labels_rebuilt_at_runtime"])
            self.assertTrue(
                result["runtime_tokenizer"]["runtime_matches_manifest"]
            )

    def test_runtime_tokenizer_drift_and_unlocked_revision_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "formal-train.jsonl"
            data_path.write_text("eight rows\n")
            manifest_path = root / "formal-train.manifest.json"
            manifest = self._manifest(data_path)
            manifest_path.write_text(json.dumps(manifest))
            model_path = Path("/models/Qwen3.5-35B-A3B-59d61f3")

            audit = validate_runtime_tokenizer_against_manifest(
                CharacterTokenizer(), manifest_path, model_path=model_path
            )
            self.assertTrue(audit["runtime_matches_manifest"])

            drifted = CharacterTokenizer()
            drifted.vocab_size = CharacterTokenizer.vocab_size + 1
            with self.assertRaisesRegex(ValueError, "vocab_size drift"):
                validate_runtime_tokenizer_against_manifest(
                    drifted, manifest_path, model_path=model_path
                )

            manifest["tokenizer"]["requested_revision"] = ""
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "requested_revision"):
                validate_runtime_tokenizer_against_manifest(
                    CharacterTokenizer(), manifest_path, model_path=model_path
                )


if __name__ == "__main__":
    unittest.main()
