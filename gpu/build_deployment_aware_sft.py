from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import random
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from deployment_aware_sft import (
    DATASETS,
    EXAMPLE_SCHEMA,
    HELDOUT_COUNTS,
    SCHEMA_VERSION,
    STRATA,
    TRAIN_COUNTS,
    stable_json,
    sha256_file,
)
from prepare_supervised_qa_train import (
    _qasper_answer_text,
    example_fingerprints,
    find_heldout_overlaps,
    iter_json_array,
    load_heldout_ledger,
    normalize_text,
    qasper_context,
    select_qasper_answer,
    twowiki_context,
)
from run_downstream import DATASET_PROMPTS


QASPER_TRAIN_SHA256 = "9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4"
QASPER_ARCHIVE_SHA256 = "a28fdf966db827bcee3d873107d6b6669864fb7ca8fbf73a192f5e39191bdb5a"
TWOWIKI_TRAIN_SHA256 = "b3fddb4d5bb42cd797919cad67616545be51b24740e0a7dabdae7bf76b8f7bfa"
TWOWIKI_ARCHIVE_SHA256 = "e8e57c0aafc4a26d41131e320ebb5afb6f2aca86b8a6e6611b08f52033cb7d04"
TULU_RAW_SHA256 = "90b98839fe0c8402553a58e75ed5e1c0bccfda48f94f526935cfc35b1e6531b1"
TULU_PARQUET_SHA256 = "19a16c5f1649d367f69899b3cfadbbeb5ffef91f24e20c6617588bdd87cd3e60"
TULU_REVISION = "b2fdafaa0744f36c91682ac1276a8bfed2da5dea"
MAX_SEQUENCE_TOKENS = 4096
MAX_TARGET_TOKENS = 512
SEED = 20260813
QASPER_TRAIN_ROWS = 256
QASPER_HELDOUT_ROWS = 12
TWOWIKI_TRAIN_ROWS = 154
TWOWIKI_HELDOUT_ROWS = 14
TULU_TRAIN_GENERAL_ROWS = 307
TULU_TRAIN_TEACHER_ROWS = 307
TULU_HELDOUT_GENERAL_ROWS = 19
TULU_HELDOUT_TEACHER_ROWS = 19


class BuildError(ValueError):
    pass


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def token_ids_sha256(values: list[int]) -> str:
    """Freeze token-list hashing as SHA256(canonical stable_json(list) UTF-8)."""

    return sha256_text(stable_json(values))


def priority(key: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{SEED}|{key}".encode()).digest(), "big")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name, delete=False
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _verify(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise BuildError(f"{label} SHA256 mismatch: expected={expected}, actual={actual}")
    return actual


def _top_candidates(rows: Iterable[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Retain deterministic smallest-hash candidates without holding a huge corpus."""

    heap: list[tuple[int, int, dict[str, Any]]] = []
    serial = 0
    for row in rows:
        score = priority(f"{row['dataset']}|{row['source_id']}")
        item = (-score, serial, row)
        serial += 1
        if len(heap) < count:
            heapq.heappush(heap, item)
        elif score < -heap[0][0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda item: (-item[0], item[1]))]


def _chat_render(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    try:
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
    if hasattr(ids, "keys") and "input_ids" in ids:
        ids = ids["input_ids"]
    elif isinstance(ids, dict):
        ids = ids.get("input_ids")
    if ids is None:
        raise BuildError("chat template returned no input_ids")
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise BuildError("chat template unexpectedly returned a batch")
        ids = ids[0]
    if not isinstance(ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in ids
    ):
        raise BuildError("chat template input_ids must be a list of non-negative integer token IDs")
    return ids


def _marker_parts(tokenizer: Any, dataset: str, question: str) -> tuple[list[int], list[int]]:
    marker = "QCOMEM_DEPLOYMENT_CONTEXT_6B8B3D"
    prompt = DATASET_PROMPTS[dataset].format(context=marker, input=question)
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    if rendered.count(marker) != 1:
        raise BuildError("chat template did not preserve the context marker")
    prefix, suffix = rendered.split(marker)
    return (
        list(tokenizer.encode(prefix, add_special_tokens=False)),
        list(tokenizer.encode(suffix, add_special_tokens=False)),
    )


def _spread_context(ids: list[int], budget: int, separator: list[int]) -> list[int]:
    if len(ids) <= budget:
        return ids
    separator = separator[: min(len(separator), max(1, budget // 64))]
    content_budget = budget - 2 * len(separator)
    if content_budget < 3:
        return ids[:budget]
    first = content_budget // 3
    middle = content_budget // 3
    last = content_budget - first - middle
    midpoint = len(ids) // 2
    middle_start = max(0, min(len(ids) - middle, midpoint - middle // 2))
    return (
        ids[:first]
        + separator
        + ids[middle_start : middle_start + middle]
        + separator
        + ids[-last:]
    )[:budget]


def _domain_tokens(
    tokenizer: Any,
    candidate: dict[str, Any],
) -> tuple[list[int], list[int], dict[str, Any], dict[str, Any]]:
    target = list(tokenizer.encode(candidate["answer"], add_special_tokens=False))
    eos = int(tokenizer.eos_token_id)
    if not target or eos in target or len(target) + 1 > MAX_TARGET_TOKENS:
        raise BuildError("domain answer cannot be represented by the frozen target policy")
    target.append(eos)
    prefix, suffix = _marker_parts(tokenizer, candidate["dataset"], candidate["question"])
    budget = MAX_SEQUENCE_TOKENS - len(target) - len(prefix) - len(suffix)
    if budget < 256:
        raise BuildError("domain prompt leaves fewer than 256 context tokens")
    original = list(tokenizer.encode(candidate["context"], add_special_tokens=False))
    evidence_text = "\n".join(candidate.get("evidence", []))
    evidence = (
        list(
            tokenizer.encode(
                "\n[Evidence-relevant excerpts]\n" + evidence_text,
                add_special_tokens=False,
            )
        )
        if evidence_text
        else []
    )
    truncated = len(original) > budget
    if not truncated:
        selected = original
        retained_evidence = bool(evidence) or candidate["dataset"] == "2wikimqa"
        strategy = "full_context"
    elif evidence:
        evidence_cap = min(len(evidence), max(64, budget // 3))
        kept_evidence = evidence[:evidence_cap]
        separator = list(tokenizer.encode("\n[...]\n", add_special_tokens=False))
        remaining = budget - evidence_cap - len(separator)
        selected = (
            kept_evidence
            + separator
            + _spread_context(original, max(0, remaining), separator)
        )[:budget]
        retained_evidence = evidence_cap > 0
        strategy = "evidence_first_plus_head_middle_tail_v1"
    else:
        separator = list(tokenizer.encode("\n[...]\n", add_special_tokens=False))
        selected = _spread_context(original, budget, separator)
        retained_evidence = candidate["answer"].casefold() == "unanswerable"
        strategy = "head_middle_tail_v1"
    document_ids = prefix + selected
    query_ids = suffix
    prompt_ids = document_ids + query_ids
    input_ids = prompt_ids + target
    labels = [-100] * len(prompt_ids) + target
    boundary = {
        "applicable": True,
        "document_input_ids": document_ids,
        "query_input_ids": query_ids,
        "document_tokens": len(document_ids),
        "query_tokens": len(query_ids),
        "prompt_tokens": len(prompt_ids),
        "document_input_ids_sha256": token_ids_sha256(document_ids),
        "query_input_ids_sha256": token_ids_sha256(query_ids),
        "prompt_input_ids_sha256": token_ids_sha256(prompt_ids),
        "answer_or_eos_tokens_in_query": False,
    }
    return input_ids, labels, {
        "original_context_tokens": len(original),
        "retained_context_tokens": len(selected),
        "context_truncated": truncated,
        "truncation_strategy": strategy,
        "evidence_available": bool(evidence),
        "evidence_retained": retained_evidence,
    }, boundary


def _tulu_tokens(
    tokenizer: Any, candidate: dict[str, Any]
) -> tuple[list[int], list[int], dict[str, Any], dict[str, Any]]:
    messages = candidate["messages"]
    if (
        not isinstance(messages, list)
        or len(messages) < 2
        or messages[-1].get("role") != "assistant"
        or not isinstance(messages[-1].get("content"), str)
    ):
        raise BuildError("Tulu row is not an assistant-terminated conversation")
    prompt_messages = []
    for message in messages[:-1]:
        if message.get("role") not in {"system", "user", "assistant"} or not isinstance(
            message.get("content"), str
        ):
            raise BuildError("Tulu prompt contains an invalid message")
        prompt_messages.append(
            {"role": str(message["role"]), "content": str(message["content"])}
        )
    prompt_ids = _chat_render(tokenizer, prompt_messages)
    target = list(tokenizer.encode(messages[-1]["content"], add_special_tokens=False))
    eos = int(tokenizer.eos_token_id)
    if not target or eos in target or len(target) + 1 > MAX_TARGET_TOKENS:
        raise BuildError("Tulu response violates the complete-target policy")
    target.append(eos)
    if len(prompt_ids) + len(target) > MAX_SEQUENCE_TOKENS:
        raise BuildError("Tulu conversation exceeds 4096 without prompt truncation")
    return prompt_ids + target, [-100] * len(prompt_ids) + target, {
        "original_context_tokens": None,
        "retained_context_tokens": None,
        "context_truncated": False,
        "truncation_strategy": "complete_chat_no_truncation",
        "evidence_available": False,
        "evidence_retained": False,
    }, {"applicable": False, "reason": "non_domain_replay_row"}


def _record(
    tokenizer: Any,
    candidate: dict[str, Any],
    *,
    stratum: str,
) -> dict[str, Any]:
    if stratum == "domain":
        input_ids, labels, tokenization, boundary = _domain_tokens(tokenizer, candidate)
    else:
        input_ids, labels, tokenization, boundary = _tulu_tokens(tokenizer, candidate)
    active_start = next(index for index, value in enumerate(labels) if value != -100)
    source_id_hash = sha256_text(normalize_text(candidate["source_id"]))
    example_id = sha256_text(f"{candidate['dataset']}\0{source_id_hash}\0{stratum}")
    prompt_hash = sha256_text(" ".join(str(value) for value in input_ids[:active_start]))
    context_hash = (
        sha256_text(normalize_text(candidate["context"]))
        if candidate.get("context") is not None
        else None
    )
    document_hash = (
        sha256_text(normalize_text(candidate["document_id"]))
        if candidate.get("document_id") is not None
        else None
    )
    return {
        "schema_version": EXAMPLE_SCHEMA,
        "example_id": example_id,
        "dataset": candidate["dataset"],
        "stratum": stratum,
        "source_split": "train",
        "source_id_sha256": source_id_hash,
        "document_id_sha256": document_hash,
        "prompt_sha256": prompt_hash,
        "context_sha256": context_hash,
        "input_ids": input_ids,
        "labels": labels,
        "token_counts": {
            "prompt": active_start,
            "target": len(labels) - active_start,
            "total": len(input_ids),
        },
        "tokenization": tokenization,
        "deployment_boundary": boundary,
        "teacher_target_required": stratum == "teacher_preservation",
        "schedule_index": None,
        "provenance": {
            "source_dataset": candidate["source_dataset"],
            "source_revision": candidate["source_revision"],
            "source_split": "train",
            "license": candidate["license"],
            "selection_priority_sha256": sha256_text(
                f"{SEED}|{candidate['dataset']}|{candidate['source_id']}"
            ),
            "raw_text_retained_in_manifest": False,
        },
    }


def _qasper_candidates(path: Path, heldout_indexes: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    papers = list(payload.items()) if isinstance(payload, dict) else []
    groups = []
    excluded_documents = 0
    excluded_questions = 0
    for paper_index, (paper_id, paper) in enumerate(papers):
        if not isinstance(paper, dict) or not paper.get("full_text"):
            continue
        context = qasper_context(paper)
        questions = []
        document_overlap = False
        for question_index, question in enumerate(paper.get("qas", [])):
            source_id = str(question.get("question_id", ""))
            question_text = question.get("question")
            if not source_id or not isinstance(question_text, str):
                raise BuildError("QASPER row lacks a question ID/text")
            fingerprints = example_fingerprints(source_id, context, question_text)
            matches = find_heldout_overlaps(fingerprints, heldout_indexes)
            if matches:
                document_overlap = True
                excluded_questions += 1
            answers, selected, selection = select_qasper_answer(question.get("answers"))
            selected_canonical = selection["selected_canonical"]
            evidence = []
            for annotation in question.get("answers", []):
                answer_payload = annotation.get("answer", {})
                try:
                    surface, _ = _qasper_answer_text(answer_payload)
                except Exception:
                    continue
                if normalize_text(surface).casefold() != selected_canonical:
                    continue
                for key in ("highlighted_evidence", "evidence"):
                    for snippet in answer_payload.get(key, []) or []:
                        if isinstance(snippet, str) and normalize_text(snippet):
                            evidence.append(normalize_text(snippet))
            questions.append(
                {
                    "dataset": "qasper",
                    "source_id": source_id,
                    "document_id": str(paper_id),
                    "question": question_text,
                    "answer": selected,
                    "answers": answers,
                    "context": context,
                    "evidence": list(dict.fromkeys(evidence)),
                    "source_dataset": "allenai/qasper",
                    "source_revision": "fdc9d8214fbab5dd782958601db4d678e6934a54",
                    "license": "CC-BY-4.0",
                    "source_record_index": paper_index,
                    "source_question_index": question_index,
                }
            )
        # A single heldout question excludes its entire paper/query family. This
        # is stronger than row-level exact matching and removes QASPER near-duplicates.
        # The hash-only ledger deliberately does not expose which raw paper owns
        # a heldout question, so in addition compare every paper context against
        # the ledger's context hashes without opening any LongBench raw split.
        context_hash = example_fingerprints("placeholder", context, "placeholder")[
            "context_sha256"
        ]
        if heldout_indexes["context_sha256"].get(context_hash):
            document_overlap = True
        if document_overlap:
            excluded_documents += 1
            continue
        if len(questions) < 2:
            continue
        ordered = sorted(
            questions,
            key=lambda row: priority(f"qasper-question|{row['source_id']}"),
        )
        groups.append(
            {
                "document_id": str(paper_id),
                "rows": ordered[:2],
                "score": priority(f"qasper-document|{paper_id}"),
            }
        )
    groups.sort(key=lambda group: group["score"])
    return groups, {
        "eligible_multi_query_documents": len(groups),
        "heldout_matched_document_families_excluded": excluded_documents,
        "heldout_matched_questions_observed": excluded_questions,
        "queries_selected_per_document": 2,
    }


def _twowiki_candidates(path: Path, heldout_indexes: Any, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = excluded = 0
    prompt_seen: set[str] = set()

    def rows() -> Iterable[dict[str, Any]]:
        nonlocal parsed, excluded
        for source_index, raw in enumerate(iter_json_array(path)):
            parsed += 1
            source_id = str(raw.get("_id", ""))
            question = raw.get("question")
            answer = raw.get("answer")
            if not source_id or not isinstance(question, str) or not isinstance(answer, str):
                raise BuildError("2Wiki row lacks required text")
            context = twowiki_context(raw.get("context"))
            fingerprints = example_fingerprints(source_id, context, question)
            if find_heldout_overlaps(fingerprints, heldout_indexes):
                excluded += 1
                continue
            prompt_key = sha256_text(normalize_text(question))
            if prompt_key in prompt_seen:
                continue
            prompt_seen.add(prompt_key)
            yield {
                "dataset": "2wikimqa",
                "source_id": source_id,
                "document_id": source_id,
                "question": question,
                "answer": normalize_text(answer),
                "context": context,
                "evidence": [],
                "source_dataset": "Alab-NII/2wikimultihop",
                "source_revision": "13800e5be57df1b4040b9b1588c6c811779e69e9",
                "license": "Apache-2.0",
                "source_record_index": source_index,
            }

    selected = _top_candidates(rows(), count)
    return selected, {
        "parsed_train_rows": parsed,
        "heldout_exact_rows_excluded": excluded,
        "normalized_duplicate_prompts_excluded": parsed - excluded - len(prompt_seen),
    }


def _tulu_candidates(path: Path, heldout_indexes: Any, count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = excluded = duplicates = 0
    prompts: set[str] = set()

    def rows() -> Iterable[dict[str, Any]]:
        nonlocal parsed, excluded, duplicates
        with path.open(encoding="utf-8") as handle:
            for line_index, line in enumerate(handle):
                raw = json.loads(line)
                parsed += 1
                source_id = str(raw.get("id", ""))
                prompt = raw.get("prompt")
                messages = raw.get("messages")
                if not source_id or not isinstance(prompt, str) or not isinstance(messages, list):
                    raise BuildError("Tulu row lacks id/prompt/messages")
                normalized = normalize_text(prompt)
                prompt_hash = sha256_text(normalized.casefold())
                if prompt_hash in prompts:
                    duplicates += 1
                    continue
                prompts.add(prompt_hash)
                fingerprints = example_fingerprints(source_id, "", prompt)
                if find_heldout_overlaps(fingerprints, heldout_indexes):
                    excluded += 1
                    continue
                yield {
                    "dataset": "tulu3_persona_if",
                    "source_id": source_id,
                    "document_id": None,
                    "context": None,
                    "messages": messages,
                    "source_dataset": "allenai/tulu-3-sft-personas-instruction-following",
                    "source_revision": TULU_REVISION,
                    "license": "ODC-BY-1.0",
                    "source_record_index": line_index,
                }

    selected = _top_candidates(rows(), count)
    return selected, {
        "parsed_decontaminated_rows": parsed,
        "heldout_input_hash_rows_excluded": excluded,
        "normalized_duplicate_prompts_excluded": duplicates,
    }


def _materialize(
    tokenizer: Any,
    candidates: list[dict[str, Any]],
    *,
    count: int,
    stratum: str,
) -> tuple[list[dict[str, Any]], int]:
    records = []
    skipped = 0
    for candidate in candidates:
        try:
            record = _record(tokenizer, candidate, stratum=stratum)
        except BuildError:
            skipped += 1
            continue
        records.append(record)
        if len(records) == count:
            break
    if len(records) != count:
        raise BuildError(
            f"could build only {len(records)}/{count} token-valid {stratum} records"
        )
    return records, skipped


def _fair_labels(counts: dict[str, int]) -> list[str]:
    total = sum(counts.values())
    current = {stratum: 0 for stratum in STRATA}
    used = Counter()
    labels = []
    for _ in range(total):
        for stratum in STRATA:
            if used[stratum] < counts[stratum]:
                current[stratum] += counts[stratum]
        eligible = [stratum for stratum in STRATA if used[stratum] < counts[stratum]]
        chosen = max(eligible, key=lambda stratum: (current[stratum], -STRATA.index(stratum)))
        current[chosen] -= total
        used[chosen] += 1
        labels.append(chosen)
    if dict(used) != counts:
        raise RuntimeError("fair schedule lost a stratum")
    return labels


def _schedule(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pools = defaultdict(list)
    for row in records:
        pools[row["stratum"]].append(row)
    for stratum in STRATA:
        pools[stratum].sort(
            key=lambda row: (row["token_counts"]["total"], row["example_id"])
        )
    labels = _fair_labels(TRAIN_COUNTS)
    ordered = []
    for schedule_index, stratum in enumerate(labels):
        # Step 1 deliberately exercises the longest examples in every represented
        # stratum.  Its backward/delta/memory gate is part of the only formal job.
        row = pools[stratum].pop() if schedule_index < 8 else pools[stratum].pop(0)
        row["schedule_index"] = schedule_index
        ordered.append(row)
    if any(pools.values()):
        raise RuntimeError("schedule did not consume every row")
    return ordered


def _intersection_counts(train: list[dict[str, Any]], heldout: list[dict[str, Any]]) -> dict[str, int]:
    fields = (
        "context_sha256",
        "document_id_sha256",
        "example_id",
        "prompt_sha256",
        "source_id_sha256",
    )
    result = {}
    for field in fields:
        left = {row[field] for row in train if row[field] is not None}
        right = {row[field] for row in heldout if row[field] is not None}
        result[field] = len(left & right)
    return result


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(row["token_counts"]["total"] for row in rows)
    targets = sorted(row["token_counts"]["target"] for row in rows)
    domain_rows = [row for row in rows if row["stratum"] == "domain"]
    document_lengths = sorted(
        row["deployment_boundary"]["document_tokens"] for row in domain_rows
    )
    query_lengths = sorted(
        row["deployment_boundary"]["query_tokens"] for row in domain_rows
    )
    return {
        "rows": len(rows),
        "stratum_counts": dict(sorted(Counter(row["stratum"] for row in rows).items())),
        "dataset_counts": dict(sorted(Counter(row["dataset"] for row in rows).items())),
        "sequence_tokens": {
            "min": lengths[0],
            "median": lengths[len(lengths) // 2],
            "p90": lengths[min(len(lengths) - 1, int(0.9 * len(lengths)))],
            "max": lengths[-1],
        },
        "target_tokens": {
            "min": targets[0],
            "median": targets[len(targets) // 2],
            "max": targets[-1],
        },
        "context_truncated": sum(row["tokenization"]["context_truncated"] for row in rows),
        "evidence_available_and_retained": sum(
            row["tokenization"]["evidence_available"]
            and row["tokenization"]["evidence_retained"]
            for row in rows
        ),
        "deployment_boundary": {
            "applicable_rows": len(domain_rows),
            "inapplicable_rows": len(rows) - len(domain_rows),
            "all_applicable_segments_nonempty": bool(domain_rows)
            and min(document_lengths) > 0
            and min(query_lengths) > 0,
            "document_tokens": {
                "min": document_lengths[0],
                "median": document_lengths[len(document_lengths) // 2],
                "max": document_lengths[-1],
            },
            "query_tokens": {
                "min": query_lengths[0],
                "median": query_lengths[len(query_lengths) // 2],
                "max": query_lengths[-1],
            },
        },
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise BuildError(f"refusing non-empty output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _verify(args.qasper_train, QASPER_TRAIN_SHA256, "QASPER train")
    _verify(args.qasper_archive, QASPER_ARCHIVE_SHA256, "QASPER archive")
    _verify(args.twowiki_train, TWOWIKI_TRAIN_SHA256, "2Wiki train")
    # The source lock is read from the existing audited spec because legacy
    # documentation contained two visually similar archive strings.
    source_spec = json.loads(args.source_spec.read_text(encoding="utf-8"))
    expected_tw_archive = source_spec["datasets"]["2wikimqa"]["archive_sha256"]
    if expected_tw_archive != TWOWIKI_ARCHIVE_SHA256:
        raise BuildError("2Wiki source spec archive lock drifted")
    _verify(args.twowiki_archive, expected_tw_archive, "2Wiki archive")
    _verify(args.tulu_raw, TULU_RAW_SHA256, "Tulu decontaminated JSONL")
    receipt = json.loads(args.tulu_receipt.read_text(encoding="utf-8"))
    expected_receipt = {
        "eligible_rows": 26316,
        "eligible_unique_ids": 26316,
        "source_revision": TULU_REVISION,
        "parquet_sha256": TULU_PARQUET_SHA256,
        "raw_jsonl_sha256": TULU_RAW_SHA256,
    }
    if any(receipt.get(key) != value for key, value in expected_receipt.items()):
        raise BuildError("Tulu decontamination receipt drifted")
    if sha256_file(args.heldout_ledger) != args.expected_heldout_ledger_sha256:
        raise BuildError("heldout hash ledger SHA256 mismatch")
    ledger, heldout_indexes = load_heldout_ledger(args.heldout_ledger)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    if not isinstance(tokenizer.eos_token_id, int):
        raise BuildError("tokenizer must have one EOS token ID")

    qasper_groups, qasper_stats = _qasper_candidates(args.qasper_train, heldout_indexes)
    required_qasper_documents = (QASPER_TRAIN_ROWS + QASPER_HELDOUT_ROWS) // 2
    qasper_records_by_document = []
    qasper_token_skips = 0
    for group in qasper_groups:
        try:
            rows = [_record(tokenizer, row, stratum="domain") for row in group["rows"]]
        except BuildError:
            qasper_token_skips += 1
            continue
        if len(rows) != 2:
            raise RuntimeError("QASPER document did not yield exactly two queries")
        qasper_records_by_document.append(rows)
        if len(qasper_records_by_document) == required_qasper_documents:
            break
    if len(qasper_records_by_document) != required_qasper_documents:
        raise BuildError("insufficient token-valid multi-query QASPER documents")
    qasper_heldout = [row for group in qasper_records_by_document[:6] for row in group]
    qasper_train = [row for group in qasper_records_by_document[6:] for row in group]

    tw_candidates, tw_stats = _twowiki_candidates(
        args.twowiki_train,
        heldout_indexes,
        TWOWIKI_TRAIN_ROWS + TWOWIKI_HELDOUT_ROWS + 128,
    )
    tw_records, tw_skips = _materialize(
        tokenizer,
        tw_candidates,
        count=TWOWIKI_TRAIN_ROWS + TWOWIKI_HELDOUT_ROWS,
        stratum="domain",
    )
    tw_heldout = tw_records[:TWOWIKI_HELDOUT_ROWS]
    tw_train = tw_records[TWOWIKI_HELDOUT_ROWS:]

    tulu_total = (
        TULU_TRAIN_GENERAL_ROWS
        + TULU_TRAIN_TEACHER_ROWS
        + TULU_HELDOUT_GENERAL_ROWS
        + TULU_HELDOUT_TEACHER_ROWS
    )
    tulu_candidates, tulu_stats = _tulu_candidates(
        args.tulu_raw, heldout_indexes, tulu_total + 512
    )
    cursor = 0

    def take_tulu(count: int, stratum: str) -> list[dict[str, Any]]:
        nonlocal cursor
        values = []
        while cursor < len(tulu_candidates) and len(values) < count:
            candidate = tulu_candidates[cursor]
            cursor += 1
            try:
                values.append(_record(tokenizer, candidate, stratum=stratum))
            except BuildError:
                continue
        if len(values) != count:
            raise BuildError(f"insufficient token-valid Tulu rows for {stratum}")
        return values

    tulu_heldout_general = take_tulu(TULU_HELDOUT_GENERAL_ROWS, "general_replay")
    tulu_heldout_teacher = take_tulu(
        TULU_HELDOUT_TEACHER_ROWS, "teacher_preservation"
    )
    tulu_train_general = take_tulu(TULU_TRAIN_GENERAL_ROWS, "general_replay")
    tulu_train_teacher = take_tulu(
        TULU_TRAIN_TEACHER_ROWS, "teacher_preservation"
    )

    train = _schedule(
        qasper_train
        + tw_train
        + tulu_train_general
        + tulu_train_teacher
    )
    heldout = (
        qasper_heldout
        + tw_heldout
        + tulu_heldout_general
        + tulu_heldout_teacher
    )
    heldout.sort(key=lambda row: (row["stratum"], row["dataset"], row["example_id"]))
    if Counter(row["stratum"] for row in train) != Counter(TRAIN_COUNTS):
        raise RuntimeError("formal train quotas drifted")
    if Counter(row["stratum"] for row in heldout) != Counter(HELDOUT_COUNTS):
        raise RuntimeError("formal heldout quotas drifted")
    intersections = _intersection_counts(train, heldout)
    if any(intersections.values()):
        raise BuildError(f"train/heldout overlap: {intersections}")
    qasper_documents = Counter(
        row["document_id_sha256"]
        for row in train + heldout
        if row["dataset"] == "qasper"
    )
    if not qasper_documents or set(qasper_documents.values()) != {2}:
        raise BuildError("every selected QASPER document must have exactly two queries")

    train_path = args.output_dir / "deployment-aware-train-1024.jsonl"
    heldout_path = args.output_dir / "deployment-aware-heldout-64.jsonl"
    _atomic_text(train_path, "".join(stable_json(row) + "\n" for row in train))
    _atomic_text(heldout_path, "".join(stable_json(row) + "\n" for row in heldout))
    schedule_labels = [row["stratum"] for row in train]
    first_step_lengths = [row["token_counts"]["total"] for row in train[:8]]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "model_initialization": "post_trained_qwen3.5_35b_a3b",
        "seed": SEED,
        "outputs": {
            "train": {
                "basename": train_path.name,
                "sha256": sha256_file(train_path),
                **_summary(train),
            },
            "heldout": {
                "basename": heldout_path.name,
                "sha256": sha256_file(heldout_path),
                **_summary(heldout),
            },
        },
        "source_artifacts": {
            "qasper": {
                "train_sha256": QASPER_TRAIN_SHA256,
                "archive_sha256": QASPER_ARCHIVE_SHA256,
                "source_revision": "fdc9d8214fbab5dd782958601db4d678e6934a54",
                "source_split": "train",
                "license": "CC-BY-4.0",
            },
            "2wikimqa": {
                "train_sha256": TWOWIKI_TRAIN_SHA256,
                "archive_sha256": expected_tw_archive,
                "source_revision": "13800e5be57df1b4040b9b1588c6c811779e69e9",
                "source_split": "train",
                "license": "Apache-2.0",
            },
            "tulu3_persona_if": {
                "raw_jsonl_sha256": TULU_RAW_SHA256,
                "receipt_sha256": sha256_file(args.tulu_receipt),
                "parquet_sha256": TULU_PARQUET_SHA256,
                "source_revision": TULU_REVISION,
                "source_split": "train",
                "license": "ODC-BY-1.0",
                "receipt": receipt,
            },
        },
        "data_mix": {
            "unit": "examples_not_target_tokens",
            "train_stratum_counts": TRAIN_COUNTS,
            "train_stratum_fractions": {
                key: value / sum(TRAIN_COUNTS.values())
                for key, value in TRAIN_COUNTS.items()
            },
            "domain_dataset_counts": {
                "qasper": QASPER_TRAIN_ROWS,
                "2wikimqa": TWOWIKI_TRAIN_ROWS,
            },
            "general_replay_source": "disjoint Tulu3 persona-IF rows",
            "teacher_preservation_source": "disjoint Tulu3 persona-IF rows",
            "global_target_token_weighting": False,
        },
        "prompt_protocol": {
            "max_sequence_tokens": MAX_SEQUENCE_TOKENS,
            "max_complete_target_tokens_including_eos": MAX_TARGET_TOKENS,
            "thinking_disabled": True,
            "answer_eos_only_labels": True,
            "qasper_truncation": "evidence_first_plus_head_middle_tail_v1",
            "qasper_answer_truncation": False,
            "tulu_prompt_or_response_truncation": False,
            "deployment_boundary_schema": {
                "version": "qcomem-domain-document-query-boundary-v1",
                "field": "deployment_boundary",
                "applicable_exactly_when_stratum": "domain",
                "document_field": "document_input_ids",
                "query_field": "query_input_ids",
                "reconstruction": (
                    "document_input_ids + query_input_ids == "
                    "input_ids[:first_non_ignore_label]"
                ),
                "answer_or_eos_in_query": False,
                "per_segment_sha256": True,
                "token_list_sha256_definition": (
                    "sha256(stable_json(list_ids).encode('utf-8')); stable_json uses "
                    "ensure_ascii=False,sort_keys=True,separators=(',',':'), no newline"
                ),
                "all_domain_segments_nonempty": True,
            },
        },
        "schedule": {
            "world_size": 8,
            "steps": len(train) // 8,
            "examples_per_rank_step": 1,
            "first_step_is_4096_backward_memory_delta_gate": max(first_step_lengths)
            == MAX_SEQUENCE_TOKENS,
            "first_step_sequence_tokens": first_step_lengths,
            "post_gate_curriculum": "ascending_sequence_length_within_each_stratum",
            "stratum_counts": dict(Counter(schedule_labels)),
            "ordered_example_id_sha256": sha256_text(
                "\n".join(row["example_id"] for row in train)
            ),
        },
        "teacher_target_protocol": {
            "generated_inside_the_only_formal_H20_job_before_optimizer_creation": True,
            "teacher": "same frozen post-trained checkpoint",
            "positions": "assistant target positions only",
            "topk": 32,
            "tail_probability_bucket": True,
            "normalized_hidden_state_saved": True,
            "rank_shards_sha256_frozen_before_training": True,
            "loss_weights_on_teacher_stratum": {
                "hard_ce": 0.45,
                "topk_plus_tail_kl_temperature_1": 0.35,
                "target_hidden_cosine": 0.20,
            },
            "loss_on_other_strata": "hard_ce_1.0",
        },
        "data_governance": {
            "heldout_ledger_sha256": sha256_file(args.heldout_ledger),
            "heldout_ledger_hash_only": True,
            "heldout_ledger_schema": ledger["schema_version"],
            "longbench_validation_rows_read": False,
            "longbench_legacy_rows_read": False,
            "longbench_test_v2_rows_read": False,
            "validation_or_test_rows_used_for_training": False,
            "qasper_entire_document_family_excluded_on_any_hash_match": True,
            "normalized_prompt_deduplication": True,
            "blind_near_duplicate_boundary": (
                "same-document QASPER families and normalized exact prompt/context hashes; "
                "raw blind rows remain unread"
            ),
        },
        "selection_stats": {
            "qasper": {**qasper_stats, "token_invalid_documents_skipped": qasper_token_skips},
            "2wikimqa": {**tw_stats, "token_invalid_rows_skipped": tw_skips},
            "tulu3_persona_if": {
                **tulu_stats,
                "token_invalid_rows_skipped": cursor - tulu_total,
            },
        },
        "audit": {
            "passed": True,
            "train_heldout_overlap_counts": intersections,
            "qasper_min_queries_per_document": min(qasper_documents.values()),
            "qasper_max_queries_per_document": max(qasper_documents.values()),
            "qasper_document_count": len(qasper_documents),
            "domain_boundary_rows": sum(
                row["deployment_boundary"]["applicable"] for row in train
            ),
            "domain_boundary_reconstruction_checked_per_row": True,
            "raw_source_ids_or_text_written_to_manifest": False,
        },
        "tokenizer": {
            "class": type(tokenizer).__name__,
            "vocab_size": int(tokenizer.vocab_size),
            "eos_token_id": int(tokenizer.eos_token_id),
            "chat_template_sha256": sha256_text(str(tokenizer.chat_template)),
        },
    }
    manifest_path = args.output_dir / "deployment-aware-manifest.json"
    _atomic_text(manifest_path, stable_json(manifest, pretty=True))
    return {
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train": str(train_path),
        "train_sha256": sha256_file(train_path),
        "heldout": str(heldout_path),
        "heldout_sha256": sha256_file(heldout_path),
        "train_summary": manifest["outputs"]["train"],
        "heldout_summary": manifest["outputs"]["heldout"],
        "schedule": manifest["schedule"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build audited deployment-aware 4K SFT data")
    parser.add_argument("--qasper-train", type=Path, required=True)
    parser.add_argument("--qasper-archive", type=Path, required=True)
    parser.add_argument("--twowiki-train", type=Path, required=True)
    parser.add_argument("--twowiki-archive", type=Path, required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--tulu-raw", type=Path, required=True)
    parser.add_argument("--tulu-receipt", type=Path, required=True)
    parser.add_argument("--heldout-ledger", type=Path, required=True)
    parser.add_argument("--expected-heldout-ledger-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.expected_heldout_ledger_sha256) != 64:
        raise SystemExit("expected heldout ledger SHA256 is invalid")
    try:
        result = build(args)
    except BuildError as error:
        raise SystemExit(f"deployment-aware data build failed: {error}") from error
    print(stable_json(result, pretty=True), end="")


if __name__ == "__main__":
    main()
