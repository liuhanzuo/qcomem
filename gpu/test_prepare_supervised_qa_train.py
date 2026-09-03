from __future__ import annotations

import argparse
import json
import tempfile
import unittest
import zipfile
from copy import deepcopy
from pathlib import Path

from build_sft_overlap_ledger import build_ledger, consumed_entries
from prepare_supervised_qa_train import (
    DATASETS,
    EXPECTED_HELDOUT_INDICES,
    FINGERPRINT_FIELDS,
    FROZEN_LONGBENCH_REVISION,
    FROZEN_TEST_V2_FILE_SHA256,
    LEDGER_SCHEMA_VERSION,
    SCHEMA_VERSION,
    DataContractError,
    OutputSelectionAnswerOverCap,
    add_training_target,
    example_fingerprints,
    find_heldout_overlaps,
    iter_json_array,
    iter_qasper_examples,
    normalize_text,
    prepare,
    qasper_context,
    qasper_source_inventory,
    select_qasper_answer,
    sha256_file,
    stable_json,
    twowiki_context,
    validate_heldout_ledger,
    validate_source_spec,
)


class FakeTokenizer:
    eos_token_id = 0
    vocab_size = 512
    chat_template = "fake-template-v1"
    init_kwargs = {"_commit_hash": "f" * 40}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=False,
    ):
        assert not tokenize
        assert add_generation_prompt
        assert enable_thinking is False
        return "<user>" + messages[0]["content"] + "</user><assistant>"

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [ord(character) + 1 for character in text]


def make_deferred_ledger() -> dict:
    return {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "source_revision": FROZEN_LONGBENCH_REVISION,
        "splits": {
            split: {
                "source_revision": FROZEN_LONGBENCH_REVISION,
                "entries": [],
                **(
                    {
                        "status": "deferred_not_read",
                        "source_file_sha256": FROZEN_TEST_V2_FILE_SHA256,
                    }
                    if split == "frozen_test_v2"
                    else {}
                ),
            }
            for split in EXPECTED_HELDOUT_INDICES
        },
    }


def make_protocol_ledger() -> dict:
    payload = make_deferred_ledger()
    for split, source_range in EXPECTED_HELDOUT_INDICES.items():
        entries = []
        for dataset in DATASETS:
            for source_index in source_range:
                fingerprints = example_fingerprints(
                    f"heldout-{dataset}-{source_index}",
                    f"heldout context {dataset} {source_index}",
                    f"heldout question {dataset} {source_index}",
                )
                entries.append(
                    {
                        "dataset": dataset,
                        "source_index": source_index,
                        **fingerprints,
                    }
                )
        payload["splits"][split]["entries"] = entries
    payload["splits"]["frozen_test_v2"]["status"] = "blind_hash_manifest"
    return payload


def qasper_annotation(
    annotation_id: str,
    *,
    free_form: str = "",
    spans: list[str] | None = None,
    yes_no=None,
    unanswerable: bool = False,
) -> dict:
    return {
        "annotation_id": annotation_id,
        "worker_id": "worker-" + annotation_id,
        "answer": {
            "unanswerable": unanswerable,
            "extractive_spans": spans or [],
            "yes_no": yes_no,
            "free_form_answer": free_form,
        },
    }


class ConversionUnitTest(unittest.TestCase):
    def test_normalized_fingerprints_are_format_stable(self) -> None:
        left = example_fingerprints("Ａ-1", "a\n b", " what\t now ")
        right = example_fingerprints("A-1", "a b", "what now")
        self.assertEqual(left, right)
        self.assertEqual(normalize_text(" Ａ\n b "), "A b")

    def test_qasper_context_and_deterministic_multi_answer(self) -> None:
        paper = {
            "full_text": [
                {"section_name": " Intro ", "paragraphs": [" First. ", "Second."]},
                {"section_name": "Methods", "paragraphs": ["Method\n body."]},
                {"section_name": None, "paragraphs": []},
            ]
        }
        self.assertEqual(
            qasper_context(paper), "Intro\nFirst.\nSecond.\nMethods\nMethod body."
        )
        annotations = [
            qasper_annotation("a2", free_form=" alpha "),
            qasper_annotation("a3", free_form="Beta"),
            qasper_annotation("a1", free_form="Alpha"),
        ]
        answers, selected, provenance = select_qasper_answer(annotations)
        _, reordered, reordered_provenance = select_qasper_answer(
            list(reversed(annotations))
        )
        self.assertEqual(selected, "Alpha")
        self.assertEqual(reordered, selected)
        self.assertEqual(provenance["strategy"], "canonical-majority_then_lexicographic-v1")
        self.assertEqual(provenance["selected_annotation_ids"], ["a1", "a2"])
        self.assertEqual(provenance["selected_count"], 2)
        self.assertEqual(reordered_provenance["selected_annotation_ids"], ["a1", "a2"])
        self.assertCountEqual(answers, ["Alpha", "alpha", "Beta"])

    def test_qasper_answer_types_are_explicit(self) -> None:
        annotations = [
            qasper_annotation("no", yes_no=False),
            qasper_annotation("span", spans=["first", "second"]),
            qasper_annotation("ua", unanswerable=True),
        ]
        answers, selected, provenance = select_qasper_answer(annotations)
        self.assertCountEqual(answers, ["no", "first, second", "unanswerable"])
        # Three-way tie is resolved by the canonical lexicographic key.
        self.assertEqual(selected, "first, second")
        self.assertCountEqual(
            [item["answer_type"] for item in provenance["annotations"]],
            ["yes_no", "extractive", "unanswerable"],
        )

    def test_original_qasper_shape_yields_question_records(self) -> None:
        payload = {
            "paper-1": {
                "full_text": [{"section_name": "S", "paragraphs": ["body"]}],
                "qas": [
                    {
                        "question_id": "q-1",
                        "question": "Question?",
                        "answers": [qasper_annotation("a", free_form="Answer")],
                    }
                ],
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qasper.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            records = list(iter_qasper_examples(path))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["source_split"], "train")
        self.assertEqual(records[0]["source_id"], "q-1")
        self.assertEqual(records[0]["selected_answer"], "Answer")

    def test_qasper_empty_full_text_matches_official_reader_skip(self) -> None:
        payload = {
            "empty-paper": {
                "full_text": [],
                "qas": [
                    {
                        "question_id": f"empty-{index}",
                        "question": "Question?",
                        "answers": [qasper_annotation("a", free_form="Answer")],
                    }
                    for index in range(3)
                ],
            },
            "normal-paper": {
                "full_text": [{"section_name": "S", "paragraphs": ["body"]}],
                "qas": [
                    {
                        "question_id": "normal",
                        "question": "Question?",
                        "answers": [qasper_annotation("b", free_form="Answer")],
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qasper.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            inventory = qasper_source_inventory(path)
            records = list(iter_qasper_examples(path))
        self.assertEqual(inventory["raw_source_examples"], 4)
        self.assertEqual(inventory["skipped_empty_full_text_records"], 1)
        self.assertEqual(inventory["skipped_empty_full_text_examples"], 3)
        self.assertEqual(inventory["expected_converted_examples"], 1)
        self.assertEqual([record["source_id"] for record in records], ["normal"])

    def test_twowiki_context_matches_longbench_shape(self) -> None:
        context = twowiki_context(
            [["Title A", ["First. ", "Second."]], ["Title B", ["Third."]]]
        )
        self.assertEqual(
            context,
            "Passage 1:\nTitle A\nFirst. Second.\nPassage 2:\nTitle B\nThird.",
        )

    def test_streaming_json_array_handles_small_chunks(self) -> None:
        rows = [{"value": "unicode-中文"}, {"value": [1, 2, 3]}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.json"
            path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
            self.assertEqual(list(iter_json_array(path, chunk_size=3)), rows)


class LabelAndLeakageTest(unittest.TestCase):
    def test_only_complete_answer_over_cap_has_selection_skip_exception(self) -> None:
        tokenizer = FakeTokenizer()
        record = {
            "dataset": "qasper",
            "source_split": "train",
            "source_id": "over-cap",
            "context": "context",
            "input": "question?",
            "answers": ["X" * 128],
            "selected_answer": "X" * 128,
            "provenance": {"source_split": "train"},
        }
        with self.assertRaises(OutputSelectionAnswerOverCap):
            add_training_target(record, tokenizer, 4096)

        record["dataset"] = "2wikimqa"
        record["answers"] = ["ok"]
        record["selected_answer"] = "ok"
        with self.assertRaises(DataContractError) as captured:
            add_training_target(record, tokenizer, 8)
        self.assertNotIsInstance(captured.exception, OutputSelectionAnswerOverCap)

    def test_answer_ce_labels_reuse_production_prompt_parts(self) -> None:
        tokenizer = FakeTokenizer()
        record = {
            "dataset": "2wikimqa",
            "source_split": "train",
            "source_id": "id-1",
            "context": "short context",
            "input": "question?",
            "answers": ["answer"],
            "selected_answer": "answer",
            "provenance": {"source_split": "train"},
        }
        converted = add_training_target(record, tokenizer, 4096)
        from run_downstream import prompt_parts
        from supervised_sft import build_supervised_example

        target_tokens = len(converted["answer_input_ids"])
        document, query, *_ = prompt_parts(tokenizer, record, 4096 - target_tokens)
        prompt_ids = document.tolist() + query.tolist()
        trainer_example = build_supervised_example(
            tokenizer, record, max_sequence_tokens=4096
        )
        self.assertEqual(converted["input_ids"][: len(prompt_ids)], prompt_ids)
        self.assertEqual(converted["input_ids"], trainer_example.input_ids.tolist())
        self.assertEqual(converted["labels"], trainer_example.labels.tolist())
        self.assertEqual(converted["labels"][: len(prompt_ids)], [-100] * len(prompt_ids))
        self.assertEqual(converted["answer_input_ids"][-1], tokenizer.eos_token_id)
        self.assertEqual(converted["labels"][len(prompt_ids) :], converted["answer_input_ids"])
        self.assertEqual(len(converted["labels"]), len(converted["input_ids"]))

    def test_all_four_overlap_keys_are_checked(self) -> None:
        fingerprints = example_fingerprints("id", "context", "question")
        indexes = {field: {} for field in FINGERPRINT_FIELDS}
        for index, field in enumerate(FINGERPRINT_FIELDS):
            indexes[field][fingerprints[field]] = [
                {"dataset": "qasper", "split": "validation", "source_index": index}
            ]
        matches = find_heldout_overlaps(fingerprints, indexes)
        self.assertEqual(
            {match["fingerprint_kind"] for match in matches}, set(FINGERPRINT_FIELDS)
        )

    def test_ledger_rejects_raw_test_content(self) -> None:
        payload = make_deferred_ledger()
        validate_heldout_ledger(payload, enforce_protocol=False)
        payload["splits"]["frozen_test_v2"]["answers"] = ["secret"]
        with self.assertRaisesRegex(DataContractError, "forbidden raw held-out"):
            validate_heldout_ledger(payload, enforce_protocol=False)

    def test_deferred_test_v2_must_be_empty(self) -> None:
        payload = make_deferred_ledger()
        payload["splits"]["frozen_test_v2"]["entries"] = [
            {
                "dataset": "qasper",
                "source_index": 68,
                **example_fingerprints("id", "ctx", "q"),
            }
        ]
        with self.assertRaisesRegex(DataContractError, "must not contain"):
            validate_heldout_ledger(payload, enforce_protocol=False)


class SourceAndIntegrationTest(unittest.TestCase):
    def _overlap_fixture(self, root: Path, policy: str) -> argparse.Namespace:
        qasper_path = root / "qasper.json"
        qasper_path.write_text(
            json.dumps(
                {
                    "paper": {
                        "full_text": [
                            {"section_name": "Intro", "paragraphs": ["Body."]}
                        ],
                        "qas": [
                            {
                                "question_id": "train-qasper-id",
                                "question": "Same heldout question?",
                                "answers": [qasper_annotation("ann", free_form="Answer")],
                            }
                        ],
                    }
                }
            ),
            encoding="utf-8",
        )
        twowiki_path = root / "twowiki.json"
        twowiki_path.write_text(
            json.dumps(
                [
                    {
                        "_id": "train-wiki-id",
                        "question": "Unique question?",
                        "answer": "Answer",
                        "context": [["Title", ["Sentence."]]],
                    }
                ]
            ),
            encoding="utf-8",
        )
        qasper_archive = root / "qasper.tgz"
        qasper_archive.write_bytes(b"qasper")
        twowiki_archive = root / "twowiki.zip"
        twowiki_archive.write_bytes(b"twowiki")
        spec = {
            "schema_version": SCHEMA_VERSION,
            "datasets": {
                dataset: {
                    "source_repo": "official/repo",
                    "source_url": "https://example.invalid/archive",
                    "source_revision": ("a" if dataset == "qasper" else "b") * 40,
                    "archive_filename": (
                        qasper_archive.name if dataset == "qasper" else twowiki_archive.name
                    ),
                    "archive_sha256": sha256_file(
                        qasper_archive if dataset == "qasper" else twowiki_archive
                    ),
                    "extracted_filename": (
                        qasper_path.name if dataset == "qasper" else twowiki_path.name
                    ),
                    "extracted_file_sha256": sha256_file(
                        qasper_path if dataset == "qasper" else twowiki_path
                    ),
                    "license": (
                        "CC-BY-4.0" if dataset == "qasper" else "Apache-2.0"
                    ),
                    "source_split": "train",
                    "expected_source_records": 1,
                    "expected_examples": 1,
                    "expected_converted_examples": 1,
                    "expected_skipped_empty_full_text_examples": 0,
                }
                for dataset in DATASETS
            },
        }
        spec_path = root / "spec.json"
        spec_path.write_text(stable_json(spec, pretty=True), encoding="utf-8")
        ledger = make_protocol_ledger()
        # Match only input_sha256. This exercises the intentionally strict
        # input-only leakage gate without exposing any held-out text in output.
        train_fingerprints = example_fingerprints(
            "train-qasper-id", "Intro\nBody.", "Same heldout question?"
        )
        ledger["splits"]["validation"]["entries"][0]["input_sha256"] = (
            train_fingerprints["input_sha256"]
        )
        ledger_path = root / "ledger.json"
        ledger_path.write_text(stable_json(ledger, pretty=True), encoding="utf-8")
        return argparse.Namespace(
            qasper_train=qasper_path,
            twowiki_train=twowiki_path,
            qasper_archive=qasper_archive,
            twowiki_archive=twowiki_archive,
            source_spec=spec_path,
            heldout_ledger=ledger_path,
            manifest=root / "manifest.json",
            mode="build",
            output_jsonl=root / "train.jsonl",
            tokenizer="fake",
            tokenizer_revision="f" * 40,
            allow_tokenizer_download=False,
            max_sequence_tokens=4096,
            smoke_count_per_dataset=1,
            overlap_policy=policy,
            overwrite=False,
        )

    def test_source_spec_rejects_non_train_or_wrong_license(self) -> None:
        spec = {
            "schema_version": SCHEMA_VERSION,
            "datasets": {
                dataset: {
                    "source_repo": "repo",
                    "source_url": "https://example.invalid/archive",
                    "source_revision": "c" * 40,
                    "archive_filename": "archive",
                    "archive_sha256": "a" * 64,
                    "extracted_filename": "train.json",
                    "extracted_file_sha256": "b" * 64,
                    "license": ("CC-BY-4.0" if dataset == "qasper" else "Apache-2.0"),
                    "source_split": "train",
                    "expected_source_records": 1,
                    "expected_examples": 1,
                    "expected_converted_examples": 1,
                    "expected_skipped_empty_full_text_examples": 0,
                }
                for dataset in DATASETS
            },
        }
        validate_source_spec(spec)
        bad = deepcopy(spec)
        bad["datasets"]["qasper"]["source_split"] = "validation"
        with self.assertRaisesRegex(DataContractError, "train split"):
            validate_source_spec(bad)
        bad = deepcopy(spec)
        bad["datasets"]["2wikimqa"]["license"] = "unknown"
        with self.assertRaisesRegex(DataContractError, "license mismatch"):
            validate_source_spec(bad)

    def test_ledger_builder_never_reads_physical_row_68(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            directory_path = Path(directory)
            dataset_paths = {}
            for dataset in DATASETS:
                path = directory_path / f"{dataset}.jsonl"
                lines = []
                for index in range(68):
                    lines.append(
                        stable_json(
                            {
                                "dataset": dataset,
                                "_id": f"{dataset}-{index}",
                                "context": f"context-{index}",
                                "input": f"input-{index}",
                                "answers": ["ignored"],
                            }
                        )
                    )
                # This is deliberately invalid JSON. islice(stop=68) must not consume it.
                path.write_text("\n".join(lines) + "\nNOT-JSON\n", encoding="utf-8")
                dataset_paths[dataset] = path
                self.assertEqual(len(consumed_entries(path, dataset)), 68)
            ledger = build_ledger(dataset_paths, None)
            self.assertEqual(
                ledger["splits"]["frozen_test_v2"]["status"], "deferred_not_read"
            )
            self.assertFalse(ledger["raw_test_v2_read"])

    def test_ledger_zip_stream_never_reads_physical_row_68(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive_path = root / "longbench.zip"
            sources = {}
            with zipfile.ZipFile(archive_path, "w") as archive:
                for dataset in DATASETS:
                    lines = [
                        stable_json(
                            {
                                "dataset": dataset,
                                "_id": f"{dataset}-{index}",
                                "context": f"context-{index}",
                                "input": f"input-{index}",
                                "answers": ["ignored"],
                            }
                        )
                        for index in range(68)
                    ]
                    member = f"data/{dataset}.jsonl"
                    archive.writestr(member, "\n".join(lines) + "\nNOT-JSON\n")
                    sources[dataset] = (archive_path, member)
            ledger = build_ledger(sources, None)
            self.assertEqual(len(ledger["splits"]["pilot"]["entries"]), 8)
            self.assertFalse(ledger["raw_test_v2_read"])

    def test_full_build_is_deterministic_and_labels_are_serialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            qasper_path = root / "qasper.json"
            qasper_path.write_text(
                json.dumps(
                    {
                        "paper": {
                            "full_text": [
                                {"section_name": "Intro", "paragraphs": ["Body."]}
                            ],
                            "qas": [
                                {
                                    "question_id": "q-train-over-cap",
                                    "question": "Qasper over-cap question?",
                                    "answers": [
                                        qasper_annotation("ann-over", free_form="X" * 128)
                                    ],
                                },
                                {
                                    "question_id": "q-train-valid",
                                    "question": "Qasper valid question?",
                                    "answers": [
                                        qasper_annotation(
                                            "ann-valid", free_form="Qasper answer"
                                        )
                                    ],
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            twowiki_path = root / "twowiki.json"
            twowiki_path.write_text(
                json.dumps(
                    [
                        {
                            "_id": "w-train",
                            "question": "Wiki question?",
                            "answer": "Wiki answer",
                            "context": [["Title", ["Sentence."]]],
                        },
                        {
                            "_id": "w-train-second",
                            "question": "Second Wiki question?",
                            "answer": "Second Wiki answer",
                            "context": [["Second title", ["Second sentence."]]],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            qasper_archive = root / "qasper.tgz"
            qasper_archive.write_bytes(b"qasper archive")
            twowiki_archive = root / "twowiki.zip"
            twowiki_archive.write_bytes(b"twowiki archive")
            source_spec = {
                "schema_version": SCHEMA_VERSION,
                "datasets": {
                    "qasper": {
                        "source_repo": "allenai/qasper",
                        "source_url": "https://example.invalid/qasper",
                        "source_revision": "a" * 40,
                        "archive_filename": qasper_archive.name,
                        "archive_sha256": sha256_file(qasper_archive),
                        "extracted_filename": qasper_path.name,
                        "extracted_file_sha256": sha256_file(qasper_path),
                        "license": "CC-BY-4.0",
                        "source_split": "train",
                        "expected_source_records": 1,
                        "expected_examples": 2,
                        "expected_converted_examples": 2,
                        "expected_skipped_empty_full_text_examples": 0,
                    },
                    "2wikimqa": {
                        "source_repo": "Alab-NII/2wikimultihop",
                        "source_url": "https://example.invalid/twowiki",
                        "source_revision": "b" * 40,
                        "archive_filename": twowiki_archive.name,
                        "archive_sha256": sha256_file(twowiki_archive),
                        "extracted_filename": twowiki_path.name,
                        "extracted_file_sha256": sha256_file(twowiki_path),
                        "license": "Apache-2.0",
                        "source_split": "train",
                        "expected_source_records": 2,
                        "expected_examples": 2,
                        "expected_converted_examples": 2,
                        "expected_skipped_empty_full_text_examples": 0,
                    },
                },
            }
            source_spec_path = root / "source-spec.json"
            source_spec_path.write_text(stable_json(source_spec, pretty=True), encoding="utf-8")
            ledger_path = root / "ledger.json"
            ledger_path.write_text(
                stable_json(make_protocol_ledger(), pretty=True), encoding="utf-8"
            )
            output_path = root / "train.jsonl"
            manifest_path = root / "manifest.json"
            args = argparse.Namespace(
                qasper_train=qasper_path,
                twowiki_train=twowiki_path,
                qasper_archive=qasper_archive,
                twowiki_archive=twowiki_archive,
                source_spec=source_spec_path,
                heldout_ledger=ledger_path,
                manifest=manifest_path,
                mode="build",
                output_jsonl=output_path,
                tokenizer="fake-tokenizer",
                tokenizer_revision="f" * 40,
                allow_tokenizer_download=False,
                max_sequence_tokens=4096,
                smoke_count_per_dataset=1,
                overlap_policy="fail",
                overwrite=False,
                max_output_per_dataset=1,
            )
            first = prepare(args, tokenizer=FakeTokenizer())
            first_output = output_path.read_bytes()
            first_manifest = manifest_path.read_bytes()
            first_manifest_payload = json.loads(first_manifest)
            self.assertEqual(first_manifest_payload["status"], "passed")
            self.assertEqual(first_manifest_payload["detected_overlap_count"], 0)
            self.assertEqual(first_manifest_payload["output_overlap_count"], 0)
            self.assertEqual(
                first_manifest_payload["heldout_protocol"][
                    "test_v2_content_hash_check"
                ],
                "blind_hash_manifest",
            )
            args.overwrite = True
            second = prepare(args, tokenizer=FakeTokenizer())
            self.assertEqual(first_output, output_path.read_bytes())
            self.assertEqual(first_manifest, manifest_path.read_bytes())
            self.assertEqual(first["output_jsonl_sha256"], second["output_jsonl_sha256"])
            records = [json.loads(line) for line in output_path.read_text().splitlines()]
            self.assertEqual(len(records), 2)
            self.assertEqual(
                first_manifest_payload["dataset_stats"]["2wikimqa"][
                    "full_eligible_examples"
                ],
                2,
            )
            self.assertEqual(
                first_manifest_payload["dataset_stats"]["2wikimqa"][
                    "written_examples"
                ],
                1,
            )
            self.assertEqual(
                first_manifest_payload["output_selection"]["written_smoke_count"],
                2,
            )
            self.assertEqual(first_manifest_payload["token_length_summary"]["count"], 2)
            self.assertEqual(
                first_manifest_payload["output_selection"]["strategy"],
                "first_n_target_valid_eligible_in_official_source_order-v1",
            )
            qasper_stats = first_manifest_payload["dataset_stats"]["qasper"]
            self.assertEqual(qasper_stats["full_eligible_examples"], 2)
            self.assertEqual(qasper_stats["selected_for_output_examples"], 1)
            self.assertEqual(qasper_stats["output_selection_skipped_answer_over_cap"], 1)
            self.assertEqual(
                qasper_stats[
                    "output_selection_skipped_answer_over_cap_source_id_sha256"
                ],
                [example_fingerprints("q-train-over-cap", "unused", "unused")["id_sha256"]],
            )
            self.assertEqual(
                first_manifest_payload["output_selection"][
                    "skipped_answer_over_cap"
                ]["qasper"],
                {
                    "count": 1,
                    "source_id_sha256": qasper_stats[
                        "output_selection_skipped_answer_over_cap_source_id_sha256"
                    ],
                },
            )
            self.assertEqual(
                [record["source_id"] for record in records if record["dataset"] == "qasper"],
                ["q-train-valid"],
            )
            for record in records:
                self.assertEqual(record["source_split"], "train")
                self.assertEqual(len(record["input_ids"]), len(record["labels"]))
                prompt = record["token_counts"]["prompt"]
                self.assertEqual(record["labels"][:prompt], [-100] * prompt)
                self.assertEqual(record["labels"][prompt:], record["answer_input_ids"])
                self.assertIn("archive_sha256", record["provenance"])
                self.assertIn("answer_selection", record["provenance"])

    def test_drop_policy_reports_detected_but_publishes_clean_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._overlap_fixture(Path(directory), "drop")
            manifest = prepare(args, tokenizer=FakeTokenizer())
            self.assertEqual(manifest["status"], "passed")
            self.assertEqual(manifest["detected_overlap_count"], 1)
            self.assertEqual(manifest["detected_overlap_unique_train_examples"], 1)
            self.assertEqual(manifest["detected_overlap_match_entries"], 1)
            self.assertEqual(manifest["output_overlap_count"], 0)
            self.assertEqual(manifest["dataset_stats"]["qasper"]["dropped_examples"], 1)
            self.assertEqual(
                manifest["dataset_stats"]["qasper"]["written_examples"], 0
            )
            self.assertTrue(args.output_jsonl.exists())
            rows = [json.loads(line) for line in args.output_jsonl.read_text().splitlines()]
            self.assertEqual([row["dataset"] for row in rows], ["2wikimqa"])
            self.assertNotIn("Same heldout question?", stable_json(manifest["overlap_report"]))

    def test_fail_policy_writes_hash_audit_but_never_publishes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._overlap_fixture(Path(directory), "fail")
            with self.assertRaisesRegex(DataContractError, "wrote hash-only failure audit"):
                prepare(args, tokenizer=FakeTokenizer())
            manifest = json.loads(args.manifest.read_text())
            self.assertEqual(manifest["status"], "failed_overlap")
            self.assertEqual(manifest["detected_overlap_count"], 1)
            self.assertEqual(manifest["output_overlap_count"], 0)
            self.assertIsNone(manifest["output_jsonl"])
            self.assertIsNone(manifest["output_jsonl_sha256"])
            self.assertFalse(args.output_jsonl.exists())


if __name__ == "__main__":
    unittest.main()
