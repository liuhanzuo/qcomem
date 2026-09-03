from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO


SCHEMA_VERSION = "qcomem-supervised-qa-v1"
LEDGER_SCHEMA_VERSION = "qcomem-sft-heldout-fingerprints-v1"
FROZEN_LONGBENCH_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
FROZEN_TEST_V2_FILE_SHA256 = (
    "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f"
)
DATASETS = ("qasper", "2wikimqa")
EXPECTED_LICENSES = {"qasper": "CC-BY-4.0", "2wikimqa": "Apache-2.0"}
EXPECTED_HELDOUT_INDICES = {
    "pilot": range(0, 4),
    "calibration": range(4, 6),
    "validation": range(6, 36),
    "legacy_test": range(36, 68),
    "frozen_test_v2": range(68, 100),
}
FINGERPRINT_FIELDS = (
    "id_sha256",
    "context_input_sha256",
    "context_sha256",
    "input_sha256",
)
HASH_RE = re.compile(r"[0-9a-f]{64}")
REVISION_RE = re.compile(r"[0-9a-f]{40}")
FORBIDDEN_LEDGER_KEYS = {
    "_id",
    "source_id",
    "context",
    "input",
    "question",
    "answer",
    "answers",
    "selected_answer",
}


class DataContractError(ValueError):
    pass


class HeldoutOverlapError(DataContractError):
    pass


class OutputSelectionAnswerOverCap(DataContractError):
    """A complete train answer is valid data but cannot fit the eval output cap."""

    def __init__(self, record: dict[str, Any], error: ValueError) -> None:
        self.dataset = record["dataset"]
        self.source_id = record["source_id"]
        super().__init__(
            f"{self.dataset}/{self.source_id} selected answer is over the frozen "
            f"generation cap: {error}"
        )


def stable_json(payload: Any, *, pretty: bool = False) -> str:
    if pretty:
        return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise DataContractError(f"expected text, found {type(value).__name__}")
    return " ".join(unicodedata.normalize("NFKC", value).split())


def example_fingerprints(source_id: str, context: str, input_text: str) -> dict[str, str]:
    normalized_id = normalize_text(source_id)
    normalized_context = normalize_text(context)
    normalized_input = normalize_text(input_text)
    pair = stable_json([normalized_context, normalized_input])
    return {
        "id_sha256": sha256_text(normalized_id),
        "context_input_sha256": sha256_text(pair),
        "context_sha256": sha256_text(normalized_context),
        "input_sha256": sha256_text(normalized_input),
    }


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HASH_RE.fullmatch(value) is None:
        raise DataContractError(f"{label} must be a lowercase 64-character SHA256")
    return value


def _reject_raw_heldout_content(value: Any, location: str = "ledger") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            if key in FORBIDDEN_LEDGER_KEYS:
                raise DataContractError(
                    f"{location} contains forbidden raw held-out field {key!r}"
                )
            _reject_raw_heldout_content(nested, f"{location}.{key}")
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            _reject_raw_heldout_content(nested, f"{location}[{index}]")


def validate_heldout_ledger(
    payload: dict[str, Any], *, enforce_protocol: bool = True
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    """Validate a hash-only held-out ledger and return indexes by hash kind.

    The ledger deliberately cannot contain raw IDs, contexts, questions, or answers.
    This lets the training-data job compare against test-v2 without opening test-v2.
    """

    _reject_raw_heldout_content(payload)
    if payload.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise DataContractError(
            f"heldout ledger schema must be {LEDGER_SCHEMA_VERSION!r}"
        )
    if payload.get("source_revision") != FROZEN_LONGBENCH_REVISION:
        raise DataContractError("heldout ledger LongBench revision is not frozen revision")
    splits = payload.get("splits")
    if not isinstance(splits, dict) or set(splits) != set(EXPECTED_HELDOUT_INDICES):
        raise DataContractError(
            "heldout ledger must contain exactly pilot, calibration, validation, "
            "legacy_test, and frozen_test_v2"
        )

    indexes: dict[str, dict[str, list[dict[str, Any]]]] = {
        field: defaultdict(list) for field in FINGERPRINT_FIELDS
    }
    for split_name, expected_range in EXPECTED_HELDOUT_INDICES.items():
        split = splits[split_name]
        if not isinstance(split, dict) or not isinstance(split.get("entries"), list):
            raise DataContractError(f"heldout split {split_name} must contain entries")
        if split.get("source_revision") != FROZEN_LONGBENCH_REVISION:
            raise DataContractError(f"heldout split {split_name} revision mismatch")
        if split_name == "frozen_test_v2":
            if split.get("source_file_sha256") != FROZEN_TEST_V2_FILE_SHA256:
                raise DataContractError("frozen test-v2 source file SHA256 mismatch")
            status = split.get("status")
            if status not in {"blind_hash_manifest", "deferred_not_read"}:
                raise DataContractError(
                    "frozen test-v2 status must be blind_hash_manifest or deferred_not_read"
                )
            if status == "deferred_not_read" and split["entries"]:
                raise DataContractError(
                    "deferred frozen test-v2 must not contain newly derived entries"
                )

        seen_dataset_indices: set[tuple[str, int]] = set()
        for entry_index, entry in enumerate(split["entries"]):
            location = f"splits.{split_name}.entries[{entry_index}]"
            if not isinstance(entry, dict):
                raise DataContractError(f"{location} must be an object")
            allowed = {"dataset", "source_index", *FINGERPRINT_FIELDS}
            unknown = set(entry) - allowed
            if unknown:
                raise DataContractError(f"{location} has unknown fields {sorted(unknown)}")
            dataset = entry.get("dataset")
            if dataset not in DATASETS:
                raise DataContractError(f"{location}.dataset is invalid")
            source_index = entry.get("source_index")
            if not isinstance(source_index, int) or isinstance(source_index, bool):
                raise DataContractError(f"{location}.source_index must be an integer")
            key = (dataset, source_index)
            if key in seen_dataset_indices:
                raise DataContractError(f"duplicate heldout entry {split_name}/{key}")
            seen_dataset_indices.add(key)
            reference = {
                "dataset": dataset,
                "split": split_name,
                "source_index": source_index,
            }
            for field in FINGERPRINT_FIELDS:
                digest = _require_sha256(entry.get(field), f"{location}.{field}")
                indexes[field][digest].append(reference)

        if enforce_protocol and not (
            split_name == "frozen_test_v2"
            and split.get("status") == "deferred_not_read"
        ):
            expected = {
                (dataset, source_index)
                for dataset in DATASETS
                for source_index in expected_range
            }
            if seen_dataset_indices != expected:
                missing = sorted(expected - seen_dataset_indices)
                unexpected = sorted(seen_dataset_indices - expected)
                raise DataContractError(
                    f"heldout split {split_name} does not match frozen protocol; "
                    f"missing={missing[:5]}, unexpected={unexpected[:5]}"
                )
    return indexes


def load_heldout_ledger(
    path: Path, *, enforce_protocol: bool = True
) -> tuple[dict[str, Any], dict[str, dict[str, list[dict[str, Any]]]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataContractError("heldout ledger root must be an object")
    return payload, validate_heldout_ledger(payload, enforce_protocol=enforce_protocol)


def find_heldout_overlaps(
    fingerprints: dict[str, str],
    indexes: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    matches = []
    for field in FINGERPRINT_FIELDS:
        for reference in indexes[field].get(fingerprints[field], []):
            matches.append({"fingerprint_kind": field, **reference})
    return sorted(
        matches,
        key=lambda row: (
            row["split"], row["dataset"], row["source_index"], row["fingerprint_kind"]
        ),
    )


def _as_sequence_rows(value: Any, label: str) -> list[dict[str, Any]]:
    """Accept original list-of-dicts and HF's dict-of-parallel-lists encoding."""

    if isinstance(value, list):
        if not all(isinstance(item, dict) for item in value):
            raise DataContractError(f"{label} must contain objects")
        return value
    if isinstance(value, dict):
        if not value:
            return []
        lengths = {len(item) for item in value.values() if isinstance(item, list)}
        if len(lengths) != 1 or any(not isinstance(item, list) for item in value.values()):
            raise DataContractError(f"{label} parallel fields have inconsistent lengths")
        length = lengths.pop()
        return [{key: item[index] for key, item in value.items()} for index in range(length)]
    raise DataContractError(f"{label} must be a list or parallel-field object")


def qasper_context(paper: dict[str, Any]) -> str:
    sections = _as_sequence_rows(paper.get("full_text"), "qasper.full_text")
    lines: list[str] = []
    for section_index, section in enumerate(sections):
        name = section.get("section_name")
        if name is not None and not isinstance(name, str):
            raise DataContractError(
                f"qasper.full_text[{section_index}].section_name must be text"
            )
        normalized_name = normalize_text(name) if isinstance(name, str) else ""
        if normalized_name:
            lines.append(normalized_name)
        paragraphs = section.get("paragraphs")
        if not isinstance(paragraphs, list) or not all(
            isinstance(paragraph, str) for paragraph in paragraphs
        ):
            raise DataContractError(
                f"qasper.full_text[{section_index}].paragraphs must be text list"
            )
        lines.extend(filter(None, (normalize_text(paragraph) for paragraph in paragraphs)))
    context = "\n".join(lines)
    if not context:
        raise DataContractError("qasper paper has empty full_text")
    return context


def _qasper_answer_text(answer: dict[str, Any]) -> tuple[str, str]:
    if bool(answer.get("unanswerable")):
        return "unanswerable", "unanswerable"
    # This precedence exactly follows the official Qasper LED reader. In valid
    # v0.3 rows only one non-unanswerable field is populated.
    yes_no = answer.get("yes_no")
    if isinstance(yes_no, bool):
        return ("yes" if yes_no else "no"), "yes_no"
    spans = answer.get("extractive_spans")
    if isinstance(spans, list):
        clean_spans = [normalize_text(span) for span in spans if normalize_text(span)]
        if clean_spans:
            # Match Qasper's official LED reader representation.
            return ", ".join(clean_spans), "extractive"
    free_form = answer.get("free_form_answer")
    if isinstance(free_form, str) and normalize_text(free_form):
        return normalize_text(free_form), "free_form"
    raise DataContractError("Qasper annotation has no supported answer value")


def select_qasper_answer(
    raw_annotations: Any,
) -> tuple[list[str], str, dict[str, Any]]:
    annotations = _as_sequence_rows(raw_annotations, "qasper.answers")
    if not annotations:
        raise DataContractError("Qasper question has no answer annotations")
    candidates = []
    for original_index, annotation in enumerate(annotations):
        answer_payload = annotation.get("answer")
        if not isinstance(answer_payload, dict):
            raise DataContractError("Qasper annotation.answer must be an object")
        text, answer_type = _qasper_answer_text(answer_payload)
        annotation_id = str(annotation.get("annotation_id", ""))
        worker_id = str(annotation.get("worker_id", ""))
        candidates.append(
            {
                "text": text,
                "canonical": normalize_text(text).casefold(),
                "answer_type": answer_type,
                "annotation_id": annotation_id,
                "worker_id": worker_id,
                "original_index": original_index,
                "extractive_spans": [
                    normalize_text(span)
                    for span in answer_payload.get("extractive_spans", [])
                    if isinstance(span, str) and normalize_text(span)
                ],
            }
        )

    counts = Counter(candidate["canonical"] for candidate in candidates)
    selected_canonical = min(counts, key=lambda key: (-counts[key], key))
    selected_text = min(
        candidate["text"]
        for candidate in candidates
        if candidate["canonical"] == selected_canonical
    )
    ordered = sorted(
        candidates,
        key=lambda item: (
            item["canonical"],
            item["text"],
            item["annotation_id"],
            item["worker_id"],
            item["original_index"],
        ),
    )
    groups = []
    for canonical in sorted(counts):
        members = [item for item in ordered if item["canonical"] == canonical]
        groups.append(
            {
                "canonical": canonical,
                "count": len(members),
                "surfaces": sorted({item["text"] for item in members}),
                "annotation_ids": sorted(item["annotation_id"] for item in members),
                "answer_types": sorted({item["answer_type"] for item in members}),
            }
        )
    provenance = {
        "strategy": "canonical-majority_then_lexicographic-v1",
        "candidate_count": len(ordered),
        "selected_canonical": selected_canonical,
        "selected_count": counts[selected_canonical],
        "selected_annotation_ids": sorted(
            item["annotation_id"]
            for item in ordered
            if item["canonical"] == selected_canonical
        ),
        "groups": groups,
        "annotations": ordered,
    }
    return [candidate["text"] for candidate in ordered], selected_text, provenance


def iter_qasper_examples(path: Path) -> Iterator[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        papers = [(str(row.get("id", index)), row) for index, row in enumerate(payload["data"])]
    elif isinstance(payload, list):
        papers = [(str(row.get("id", index)), row) for index, row in enumerate(payload)]
    elif isinstance(payload, dict):
        papers = [(str(paper_id), paper) for paper_id, paper in payload.items()]
    else:
        raise DataContractError("Qasper train root must be a paper map or list")

    for paper_index, (fallback_paper_id, paper) in enumerate(papers):
        if not isinstance(paper, dict):
            raise DataContractError(f"Qasper paper {fallback_paper_id!r} is not an object")
        paper_id = str(paper.get("id", fallback_paper_id))
        # The official Qasper LED reader skips papers whose full_text list is
        # empty. v0.3 train contains exactly one such paper (3 questions).
        if not paper.get("full_text"):
            continue
        context = qasper_context(paper)
        questions = _as_sequence_rows(paper.get("qas"), f"qasper.{paper_id}.qas")
        for question_index, question in enumerate(questions):
            source_id = str(question.get("question_id", ""))
            input_text = question.get("question")
            if not source_id or not isinstance(input_text, str) or not normalize_text(input_text):
                raise DataContractError(
                    f"Qasper {paper_id} question {question_index} lacks id/question"
                )
            answers, selected, selection = select_qasper_answer(question.get("answers"))
            yield {
                "dataset": "qasper",
                "source_split": "train",
                "source_id": source_id,
                "context": context,
                "input": input_text,
                "answers": answers,
                "selected_answer": selected,
                "_conversion_provenance": {
                    "source_record_index": paper_index,
                    "source_paper_id": paper_id,
                    "source_question_index": question_index,
                    "answer_selection": selection,
                },
            }


def qasper_source_inventory(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
        papers = payload["data"]
    elif isinstance(payload, list):
        papers = payload
    elif isinstance(payload, dict):
        papers = list(payload.values())
    else:
        raise DataContractError("Qasper train root must be a paper map or list")
    raw_questions = 0
    skipped_papers = 0
    skipped_questions = 0
    for paper in papers:
        if not isinstance(paper, dict):
            raise DataContractError("Qasper source contains a non-object paper")
        questions = _as_sequence_rows(paper.get("qas"), "qasper.qas")
        raw_questions += len(questions)
        if not paper.get("full_text"):
            skipped_papers += 1
            skipped_questions += len(questions)
    return {
        "source_records": len(papers),
        "raw_source_examples": raw_questions,
        "skipped_empty_full_text_records": skipped_papers,
        "skipped_empty_full_text_examples": skipped_questions,
        "expected_converted_examples": raw_questions - skipped_questions,
    }


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    """Stream a top-level JSON array without requiring the optional ijson package."""

    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        cursor = 0
        eof = False

        def fill() -> bool:
            nonlocal buffer, cursor, eof
            if eof:
                return False
            if cursor:
                buffer = buffer[cursor:]
                cursor = 0
            block = handle.read(chunk_size)
            if not block:
                eof = True
                return False
            buffer += block
            return True

        fill()
        while True:
            while cursor < len(buffer) and buffer[cursor].isspace():
                cursor += 1
            if cursor < len(buffer):
                break
            if not fill():
                raise DataContractError(f"{path} is empty")
        if buffer[cursor] != "[":
            raise DataContractError(f"{path} must contain a top-level JSON array")
        cursor += 1
        expect_value = True
        while True:
            while True:
                while cursor < len(buffer) and buffer[cursor].isspace():
                    cursor += 1
                if cursor < len(buffer):
                    break
                if not fill():
                    raise DataContractError(f"{path} ended before closing array")
            if buffer[cursor] == "]":
                cursor += 1
                break
            if not expect_value:
                if buffer[cursor] != ",":
                    raise DataContractError(f"{path} expected comma between array values")
                cursor += 1
                expect_value = True
                continue
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                    cursor = end
                    break
                except json.JSONDecodeError:
                    if not fill():
                        raise DataContractError(f"{path} contains truncated/invalid JSON")
            yield value
            expect_value = False
        while True:
            while cursor < len(buffer) and buffer[cursor].isspace():
                cursor += 1
            if cursor < len(buffer):
                raise DataContractError(f"{path} has content after top-level JSON array")
            if not fill():
                break


def twowiki_context(raw_context: Any) -> str:
    if not isinstance(raw_context, list) or not raw_context:
        raise DataContractError("2Wiki context must be a non-empty passage list")
    passages = []
    for passage_index, passage in enumerate(raw_context, start=1):
        if not isinstance(passage, list) or len(passage) != 2:
            raise DataContractError(f"2Wiki passage {passage_index} must be [title, sentences]")
        title, sentences = passage
        if not isinstance(title, str) or not isinstance(sentences, list) or not all(
            isinstance(sentence, str) for sentence in sentences
        ):
            raise DataContractError(f"2Wiki passage {passage_index} has invalid fields")
        # This intentionally matches the LongBench context representation.
        passages.append(
            f"Passage {passage_index}:\n{title.strip()}\n{''.join(sentences).strip()}"
        )
    return "\n".join(passages)


def iter_twowiki_examples(path: Path) -> Iterator[dict[str, Any]]:
    for source_index, row in enumerate(iter_json_array(path)):
        if not isinstance(row, dict):
            raise DataContractError(f"2Wiki row {source_index} must be an object")
        source_id = str(row.get("_id", ""))
        input_text = row.get("question")
        answer = row.get("answer")
        if not source_id or not isinstance(input_text, str) or not normalize_text(input_text):
            raise DataContractError(f"2Wiki row {source_index} lacks _id/question")
        if not isinstance(answer, str) or not normalize_text(answer):
            raise DataContractError(f"2Wiki train row {source_index} lacks answer")
        selected = normalize_text(answer)
        yield {
            "dataset": "2wikimqa",
            "source_split": "train",
            "source_id": source_id,
            "context": twowiki_context(row.get("context")),
            "input": input_text,
            "answers": [selected],
            "selected_answer": selected,
            "_conversion_provenance": {
                "source_record_index": source_index,
                "answer_selection": {
                    "strategy": "single_official_answer-v1",
                    "candidate_count": 1,
                },
            },
        }


def validate_source_spec(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise DataContractError(f"source spec schema must be {SCHEMA_VERSION!r}")
    datasets = payload.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(DATASETS):
        raise DataContractError(f"source spec must contain exactly {DATASETS}")
    required = {
        "source_repo",
        "source_url",
        "source_revision",
        "archive_filename",
        "archive_sha256",
        "extracted_filename",
        "extracted_file_sha256",
        "license",
        "source_split",
        "expected_source_records",
        "expected_examples",
        "expected_converted_examples",
        "expected_skipped_empty_full_text_examples",
    }
    official_identity = {
        "qasper": {
            "source_repo": "allenai/qasper",
            "source_url": (
                "https://qasper-dataset.s3.us-west-2.amazonaws.com/"
                "qasper-train-dev-v0.3.tgz"
            ),
            "source_revision": "fdc9d8214fbab5dd782958601db4d678e6934a54",
            "archive_filename": "qasper-train-dev-v0.3.tgz",
            "archive_sha256": (
                "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
            ),
            "extracted_filename": "qasper-train-v0.3.json",
            "extracted_file_sha256": (
                "9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4"
            ),
            "expected_source_records": 888,
            "expected_examples": 2593,
            "expected_converted_examples": 2590,
            "expected_skipped_empty_full_text_examples": 3,
        },
        "2wikimqa": {
            "source_repo": "Alab-NII/2wikimultihop",
            "source_url": "https://www.dropbox.com/s/npidmtadreo6df2/data.zip",
            "source_revision": "13800e5be57df1b4040b9b1588c6c811779e69e9",
            "archive_filename": "data.zip",
            "archive_sha256": (
                "e8e57c0aafc4a26d41131e320ebb5afb6f2aca86b8a6e6611b08f52033cb7d04"
            ),
            "extracted_filename": "data/train.json",
            "extracted_file_sha256": (
                "b3fddb4d5bb42cd797919cad67616545be51b24740e0a7dabdae7bf76b8f7bfa"
            ),
            "expected_source_records": 167454,
            "expected_examples": 167454,
            "expected_converted_examples": 167454,
            "expected_skipped_empty_full_text_examples": 0,
        },
    }
    for dataset, spec in datasets.items():
        if not isinstance(spec, dict) or not required.issubset(spec):
            missing = sorted(required - set(spec or {}))
            raise DataContractError(f"source spec {dataset} missing {missing}")
        if spec["source_split"] != "train":
            raise DataContractError(f"source spec {dataset} must use train split")
        if spec["license"] != EXPECTED_LICENSES[dataset]:
            raise DataContractError(f"source spec {dataset} license mismatch")
        _require_sha256(spec["archive_sha256"], f"source spec {dataset}.archive_sha256")
        _require_sha256(
            spec["extracted_file_sha256"],
            f"source spec {dataset}.extracted_file_sha256",
        )
        for count_field in (
            "expected_source_records",
            "expected_examples",
            "expected_converted_examples",
            "expected_skipped_empty_full_text_examples",
        ):
            value = spec[count_field]
            minimum = 0 if count_field == "expected_skipped_empty_full_text_examples" else 1
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise DataContractError(f"source spec {dataset}.{count_field} is invalid")
        revision = spec["source_revision"]
        if not isinstance(revision, str) or REVISION_RE.fullmatch(revision) is None:
            raise DataContractError(
                f"source spec {dataset}.source_revision must be an immutable 40-hex commit"
            )
        if payload.get("official_source_lock", False):
            for field, expected in official_identity[dataset].items():
                if spec[field] != expected:
                    raise DataContractError(
                        f"official source lock mismatch: {dataset}.{field}"
                    )
    return datasets


def load_and_verify_source_spec(
    path: Path,
    train_paths: dict[str, Path],
    archive_paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataContractError("source spec root must be an object")
    specs = validate_source_spec(payload)
    for dataset in DATASETS:
        train_path = train_paths[dataset]
        archive_path = archive_paths[dataset]
        if sha256_file(train_path) != specs[dataset]["extracted_file_sha256"]:
            raise DataContractError(f"{dataset} extracted train file SHA256 mismatch")
        if sha256_file(archive_path) != specs[dataset]["archive_sha256"]:
            raise DataContractError(f"{dataset} source archive SHA256 mismatch")
    return payload, specs


def add_training_target(
    record: dict[str, Any], tokenizer: Any, max_sequence_tokens: int
) -> dict[str, Any]:
    # This is the same builder used by the trainer. It reserves answer+EOS before
    # calling prompt_parts, so serialized targets cannot exceed or drift from the
    # trainer's max-sequence contract.
    from supervised_sft import (
        AnswerTargetOverGenerationCapError,
        build_supervised_example,
    )

    try:
        example = build_supervised_example(
            tokenizer,
            record,
            max_sequence_tokens=max_sequence_tokens,
        )
    except AnswerTargetOverGenerationCapError as error:
        raise OutputSelectionAnswerOverCap(record, error) from error
    except ValueError as error:
        raise DataContractError(
            f"{record['dataset']}/{record['source_id']} supervised target failed: {error}"
        ) from error
    input_ids = example.input_ids.tolist()
    labels = example.labels.tolist()
    prompt_length = example.prompt_tokens
    document_length = example.prefix_tokens + example.context_tokens
    document_ids = input_ids[:document_length]
    query_ids = input_ids[document_length:prompt_length]
    answer_ids = input_ids[prompt_length:]
    return {
        **record,
        "document_input_ids": document_ids,
        "query_input_ids": query_ids,
        "answer_input_ids": answer_ids,
        "input_ids": input_ids,
        "labels": labels,
        "token_counts": {
            "prefix": example.prefix_tokens,
            "context": example.context_tokens,
            "original_context": example.original_context_tokens,
            "query": len(query_ids),
            "prompt": prompt_length,
            "answer": example.answer_tokens,
            "answer_with_eos": example.target_tokens,
            "total": len(input_ids),
        },
    }


def finalized_record(
    raw_record: dict[str, Any], source_spec: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    if raw_record.get("source_split") != "train":
        raise DataContractError("converter refuses every non-train source record")
    fingerprints = example_fingerprints(
        raw_record["source_id"], raw_record["context"], raw_record["input"]
    )
    conversion = raw_record.pop("_conversion_provenance")
    provenance = {
        "source_repo": source_spec["source_repo"],
        "source_url": source_spec["source_url"],
        "source_revision": source_spec["source_revision"],
        "archive_sha256": source_spec["archive_sha256"],
        "extracted_file_sha256": source_spec["extracted_file_sha256"],
        "license": source_spec["license"],
        "source_split": "train",
        "fingerprints": fingerprints,
        **conversion,
    }
    return {**raw_record, "provenance": provenance}, fingerprints


def _iter_dataset(path: Path, dataset: str) -> Iterator[dict[str, Any]]:
    if dataset == "qasper":
        return iter_qasper_examples(path)
    return iter_twowiki_examples(path)


def _source_record_count(record: dict[str, Any]) -> int:
    provenance = record["provenance"]
    return int(provenance["source_record_index"])


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _load_tokenizer(name_or_path: str, revision: str, allow_download: bool) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise DataContractError("build mode requires transformers") from error
    return AutoTokenizer.from_pretrained(
        name_or_path,
        revision=revision,
        local_files_only=not allow_download,
        trust_remote_code=False,
    )


def _tokenizer_metadata(tokenizer: Any, requested: str, revision: str) -> dict[str, Any]:
    chat_template = getattr(tokenizer, "chat_template", None)
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    return {
        "requested_name_or_path": requested,
        "requested_revision": revision,
        "resolved_commit_hash": init_kwargs.get("_commit_hash"),
        "class": type(tokenizer).__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "chat_template_sha256": (
            sha256_text(chat_template) if isinstance(chat_template, str) else None
        ),
    }


def prepare(args: argparse.Namespace, *, tokenizer: Any | None = None) -> dict[str, Any]:
    requested_output_limit = getattr(args, "max_output_per_dataset", None)
    max_output_per_dataset = (
        4
        if args.mode == "smoke-manifest" and requested_output_limit is None
        else requested_output_limit
    )
    train_paths = {"qasper": args.qasper_train, "2wikimqa": args.twowiki_train}
    archive_paths = {"qasper": args.qasper_archive, "2wikimqa": args.twowiki_archive}
    source_payload, source_specs = load_and_verify_source_spec(
        args.source_spec, train_paths, archive_paths
    )
    ledger_payload, heldout_indexes = load_heldout_ledger(args.heldout_ledger)
    if args.mode == "build" and tokenizer is None:
        if not args.tokenizer or not args.tokenizer_revision:
            raise DataContractError(
                "build mode requires --tokenizer and immutable --tokenizer-revision"
            )
        tokenizer = _load_tokenizer(
            args.tokenizer, args.tokenizer_revision, args.allow_tokenizer_download
        )
    if args.mode == "smoke-manifest" and args.output_jsonl is not None:
        raise DataContractError("smoke-manifest mode never writes --output-jsonl")
    if args.mode == "build" and args.output_jsonl is None:
        raise DataContractError("build mode requires --output-jsonl")
    if (
        args.mode == "build"
        and args.output_jsonl.exists()
        and not getattr(args, "overwrite", False)
    ):
        raise DataContractError(
            f"refusing to replace existing {args.output_jsonl}; pass --overwrite explicitly"
        )

    # Build rows into a deterministic staging file. This is intentionally also
    # used by smoke-manifest mode: it exercises the exact serialization path but
    # never publishes a training JSONL.
    staging_directory = args.manifest.parent
    staging_directory.mkdir(parents=True, exist_ok=True)
    output_temporary = staging_directory / (args.manifest.name + ".records.tmp")
    output_handle: TextIO | None = output_temporary.open("w", encoding="utf-8")
    output_digest = hashlib.sha256()

    dataset_stats: dict[str, dict[str, Any]] = {}
    overlap_report: list[dict[str, Any]] = []
    overlap_match_counts: Counter[tuple[str, str]] = Counter()
    smoke_records: list[dict[str, Any]] = []
    seen_ids: dict[str, set[str]] = {dataset: set() for dataset in DATASETS}
    token_lengths: list[int] = []
    try:
        for dataset in DATASETS:
            qasper_inventory = (
                qasper_source_inventory(train_paths[dataset])
                if dataset == "qasper"
                else None
            )
            stats: dict[str, Any] = {
                "source_records": 0,
                "parsed_examples": 0,
                "overlap_examples": 0,
                "dropped_examples": 0,
                "eligible_examples": 0,
                "full_eligible_examples": 0,
                "selected_for_output_examples": 0,
                "written_examples": 0,
                "output_selection_skipped_answer_over_cap": 0,
                "output_selection_skipped_answer_over_cap_source_id_sha256": [],
                "raw_source_examples": (
                    qasper_inventory["raw_source_examples"]
                    if qasper_inventory is not None
                    else 0
                ),
                "skipped_empty_full_text_records": (
                    qasper_inventory["skipped_empty_full_text_records"]
                    if qasper_inventory is not None
                    else 0
                ),
                "skipped_empty_full_text_examples": (
                    qasper_inventory["skipped_empty_full_text_examples"]
                    if qasper_inventory is not None
                    else 0
                ),
            }
            max_source_index = -1
            for raw_record in _iter_dataset(train_paths[dataset], dataset):
                record, fingerprints = finalized_record(raw_record, source_specs[dataset])
                stats["parsed_examples"] += 1
                max_source_index = max(max_source_index, _source_record_count(record))
                source_id = record["source_id"]
                if source_id in seen_ids[dataset]:
                    raise DataContractError(f"duplicate train source id {dataset}/{source_id}")
                seen_ids[dataset].add(source_id)
                matches = find_heldout_overlaps(fingerprints, heldout_indexes)
                if matches:
                    stats["overlap_examples"] += 1
                    report = {
                        "dataset": dataset,
                        "train_source_id_sha256": fingerprints["id_sha256"],
                        "matches": matches,
                    }
                    overlap_report.append(report)
                    overlap_match_counts.update(
                        (match["split"], match["fingerprint_kind"])
                        for match in matches
                    )
                    if args.overlap_policy == "fail":
                        # Keep scanning so the failure manifest contains the complete
                        # hash-only audit. No candidate output is published below.
                        continue
                    stats["dropped_examples"] += 1
                    continue
                stats["eligible_examples"] += 1
                stats["full_eligible_examples"] += 1

                output_record = record
                selected_for_output = (
                    max_output_per_dataset is None
                    or stats["selected_for_output_examples"]
                    < max_output_per_dataset
                )
                if selected_for_output and tokenizer is not None:
                    try:
                        output_record = add_training_target(
                            record, tokenizer, args.max_sequence_tokens
                        )
                    except OutputSelectionAnswerOverCap:
                        if max_output_per_dataset is None:
                            # A full-output build has no selection boundary: every
                            # eligible row is output, so an invalid target is fatal.
                            raise
                        stats["output_selection_skipped_answer_over_cap"] += 1
                        stats[
                            "output_selection_skipped_answer_over_cap_source_id_sha256"
                        ].append(fingerprints["id_sha256"])
                        # The selected answer is never truncated. Continue official
                        # source order until N complete target-valid rows are found.
                        continue
                    token_lengths.append(output_record["token_counts"]["total"])
                if len(smoke_records) < args.smoke_count_per_dataset * len(DATASETS):
                    dataset_smoke_count = sum(
                        sample["dataset"] == dataset for sample in smoke_records
                    )
                    if dataset_smoke_count < args.smoke_count_per_dataset:
                        selection = record["provenance"]["answer_selection"]
                        smoke_records.append(
                            {
                                "dataset": dataset,
                                "source_id_sha256": fingerprints["id_sha256"],
                                "context_input_sha256": fingerprints[
                                    "context_input_sha256"
                                ],
                                "selected_answer_sha256": sha256_text(
                                    normalize_text(record["selected_answer"])
                                ),
                                "answer_selection_strategy": selection["strategy"],
                                "answer_candidate_count": selection["candidate_count"],
                                "token_counts": output_record.get("token_counts"),
                            }
                        )
                if selected_for_output:
                    stats["selected_for_output_examples"] += 1
                    line = stable_json(output_record) + "\n"
                    output_handle.write(line)
                    output_digest.update(line.encode("utf-8"))
                    if args.mode == "build":
                        stats["written_examples"] += 1
            stats["source_records"] = max_source_index + 1
            if qasper_inventory is not None:
                stats["source_records"] = qasper_inventory["source_records"]
            else:
                stats["raw_source_examples"] = stats["parsed_examples"]
            expected_records = source_specs[dataset]["expected_source_records"]
            expected_examples = source_specs[dataset]["expected_examples"]
            expected_converted = source_specs[dataset]["expected_converted_examples"]
            if stats["source_records"] != expected_records:
                raise DataContractError(
                    f"{dataset} source record count {stats['source_records']} != "
                    f"frozen {expected_records}"
                )
            if stats["raw_source_examples"] != expected_examples:
                raise DataContractError(
                    f"{dataset} raw source example count {stats['raw_source_examples']} != "
                    f"frozen {expected_examples}"
                )
            if stats["parsed_examples"] != expected_converted:
                raise DataContractError(
                    f"{dataset} example count {stats['parsed_examples']} != "
                    f"frozen converted count {expected_converted}"
                )
            expected_skipped = source_specs[dataset][
                "expected_skipped_empty_full_text_examples"
            ]
            if stats["skipped_empty_full_text_examples"] != expected_skipped:
                raise DataContractError(
                    f"{dataset} empty-full-text skip count "
                    f"{stats['skipped_empty_full_text_examples']} != frozen "
                    f"{expected_skipped}"
                )
            dataset_stats[dataset] = stats
    except Exception:
        output_handle.close()
        output_temporary.unlink(missing_ok=True)
        raise
    else:
        output_handle.flush()
        os.fsync(output_handle.fileno())
        output_handle.close()
        if args.mode == "smoke-manifest":
            output_temporary.unlink(missing_ok=True)
        elif overlap_report and args.overlap_policy == "fail":
            output_temporary.unlink(missing_ok=True)
            for stats in dataset_stats.values():
                stats["candidate_examples_before_failed_gate"] = stats[
                    "written_examples"
                ]
                stats["written_examples"] = 0
        else:
            os.replace(output_temporary, args.output_jsonl)

    prompt_path = Path(__file__).with_name("run_downstream.py")
    target_builder_path = Path(__file__).with_name("supervised_sft.py")
    smoke_payload_sha256 = sha256_text(stable_json(smoke_records))
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "converter_source_file_sha256": sha256_file(Path(__file__)),
        "status": (
            "failed_overlap"
            if overlap_report and args.overlap_policy == "fail"
            else "passed"
        ),
        "mode": args.mode,
        "source_spec_sha256": sha256_file(args.source_spec),
        "source_spec": source_payload,
        "source_artifacts": {
            dataset: {
                "source_repo": source_specs[dataset]["source_repo"],
                "source_revision": source_specs[dataset]["source_revision"],
                "source_split": source_specs[dataset]["source_split"],
                "license": source_specs[dataset]["license"],
                "archive_filename": source_specs[dataset]["archive_filename"],
                "archive_sha256": source_specs[dataset]["archive_sha256"],
                "train_filename": source_specs[dataset]["extracted_filename"],
                "train_file_sha256": source_specs[dataset][
                    "extracted_file_sha256"
                ],
                "observed_source_records": dataset_stats[dataset]["source_records"],
                "observed_raw_source_examples": dataset_stats[dataset][
                    "raw_source_examples"
                ],
                "observed_converted_examples": dataset_stats[dataset][
                    "parsed_examples"
                ],
                "skipped_empty_full_text_examples": dataset_stats[dataset][
                    "skipped_empty_full_text_examples"
                ],
            }
            for dataset in DATASETS
        },
        "heldout_ledger_sha256": sha256_file(args.heldout_ledger),
        "heldout_ledger_schema": ledger_payload["schema_version"],
        "heldout_protocol": {
            "source_revision": FROZEN_LONGBENCH_REVISION,
            "frozen_test_v2_source_file_sha256": FROZEN_TEST_V2_FILE_SHA256,
            "raw_test_v2_read_by_converter": False,
            "test_v2_content_hash_check": ledger_payload["splits"][
                "frozen_test_v2"
            ]["status"],
            "fingerprint_fields": list(FINGERPRINT_FIELDS),
            "overlap_policy": args.overlap_policy,
        },
        "prompt_protocol": {
            "function": "run_downstream.prompt_parts",
            "source_file_sha256": sha256_file(prompt_path),
            "target_builder": "supervised_sft.build_supervised_example",
            "target_builder_source_file_sha256": sha256_file(target_builder_path),
            "max_sequence_tokens": args.max_sequence_tokens,
            "answer_tokens_reserved_before_prompt_truncation": True,
            "label_ignore_index": -100,
            "answer_eos_appended": True,
        },
        "output_selection": {
            "strategy": (
                "first_n_target_valid_eligible_in_official_source_order-v1"
                if max_output_per_dataset is not None and tokenizer is not None
                else "first_n_eligible_in_official_source_order-v1"
            ),
            "requested_max_output_per_dataset": requested_output_limit,
            "max_output_per_dataset": max_output_per_dataset,
            "full_train_scan_completed": True,
            "selection_applied_after_overlap_filter": True,
            "target_validity_checked_before_selection": (
                max_output_per_dataset is not None and tokenizer is not None
            ),
            "answer_over_cap_policy": "skip_complete_answer_without_truncation",
            "skipped_answer_over_cap": {
                dataset: {
                    "count": dataset_stats[dataset][
                        "output_selection_skipped_answer_over_cap"
                    ],
                    "source_id_sha256": dataset_stats[dataset][
                        "output_selection_skipped_answer_over_cap_source_id_sha256"
                    ],
                }
                for dataset in DATASETS
            },
            "written_smoke_count": sum(
                stats["written_examples"] for stats in dataset_stats.values()
            ) if args.mode == "build" and max_output_per_dataset is not None else 0,
            "smoke_manifest_record_count": len(smoke_records),
            "written_jsonl_count": sum(
                stats["written_examples"] for stats in dataset_stats.values()
            ),
        },
        "answer_target_selection": {
            "qasper": "canonical-majority_then_lexicographic-v1",
            "qasper_extractive_span_join": ", ",
            "2wikimqa": "single_official_answer-v1",
        },
        "dataset_stats": dataset_stats,
        "detected_overlap_count": len(overlap_report),
        "detected_overlap_unique_train_examples": len(overlap_report),
        "detected_overlap_match_entries": sum(overlap_match_counts.values()),
        "overlap_match_breakdown": [
            {
                "split": split,
                "fingerprint_kind": fingerprint_kind,
                "match_entries": count,
            }
            for (split, fingerprint_kind), count in sorted(overlap_match_counts.items())
        ],
        "overlap_example_match_count_histogram": {
            str(match_count): count
            for match_count, count in sorted(
                Counter(len(report["matches"]) for report in overlap_report).items()
            )
        },
        # Every detected row is either excluded by drop policy or blocks output
        # publication under fail policy. Thus no published output contains overlap.
        "output_overlap_count": 0,
        "overlap_report": overlap_report,
        "smoke_count_per_dataset": args.smoke_count_per_dataset,
        "smoke_records": smoke_records,
        "smoke_payload_sha256": smoke_payload_sha256,
        "candidate_records_sha256": output_digest.hexdigest(),
        "requested_output_jsonl": str(args.output_jsonl) if args.output_jsonl else None,
        "output_jsonl": (
            str(args.output_jsonl)
            if args.output_jsonl
            and not (overlap_report and args.overlap_policy == "fail")
            else None
        ),
        "output_jsonl_sha256": (
            output_digest.hexdigest()
            if args.mode == "build"
            and not (overlap_report and args.overlap_policy == "fail")
            else None
        ),
        "tokenizer": (
            _tokenizer_metadata(tokenizer, args.tokenizer, args.tokenizer_revision)
            if tokenizer is not None
            else None
        ),
        "token_length_summary": (
            {
                "count": len(token_lengths),
                "min": min(token_lengths),
                "median": statistics.median(token_lengths),
                "max": max(token_lengths),
            }
            if token_lengths
            else None
        ),
    }
    manifest_preimage = stable_json(manifest)
    manifest["manifest_preimage_sha256"] = sha256_text(manifest_preimage)
    _atomic_write_text(args.manifest, stable_json(manifest, pretty=True))
    if manifest["status"] == "failed_overlap":
        raise HeldoutOverlapError(
            f"detected {len(overlap_report)} held-out overlaps; "
            f"wrote hash-only failure audit to {args.manifest}; no JSONL was published"
        )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline, leakage-audited Qasper/2Wiki supervised SFT converter"
    )
    parser.add_argument("--qasper-train", type=Path, required=True)
    parser.add_argument("--twowiki-train", type=Path, required=True)
    parser.add_argument("--qasper-archive", type=Path, required=True)
    parser.add_argument("--twowiki-archive", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--heldout-ledger", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("smoke-manifest", "build"), default="smoke-manifest"
    )
    parser.add_argument("--output-jsonl", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--tokenizer")
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    parser.add_argument("--max-sequence-tokens", type=int, default=1024)
    parser.add_argument("--smoke-count-per-dataset", type=int, default=4)
    parser.add_argument(
        "--max-output-per-dataset",
        type=int,
        help=(
            "deterministically publish only the first N eligible rows per dataset; "
            "the converter still scans and audits every train row"
        ),
    )
    parser.add_argument("--overlap-policy", choices=("fail", "drop"), default="fail")
    args = parser.parse_args()
    if args.max_sequence_tokens < 512:
        parser.error("--max-sequence-tokens must be at least 512")
    if args.smoke_count_per_dataset < 1:
        parser.error("--smoke-count-per-dataset must be positive")
    if args.max_output_per_dataset is not None and args.max_output_per_dataset < 1:
        parser.error("--max-output-per-dataset must be positive")
    return args


def main() -> None:
    try:
        manifest = prepare(parse_args())
    except DataContractError as error:
        raise SystemExit(f"data contract failed: {error}") from error
    print(stable_json({
        "manifest_preimage_sha256": manifest["manifest_preimage_sha256"],
        "output_jsonl_sha256": manifest["output_jsonl_sha256"],
        "detected_overlap_count": manifest["detected_overlap_count"],
        "output_overlap_count": manifest["output_overlap_count"],
        "dataset_stats": manifest["dataset_stats"],
    }, pretty=True), end="")


if __name__ == "__main__":
    main()
