#!/usr/bin/env python3
"""Formal ordered BoolQ experiment for A11.

The online policy observes parsed Yes/No votes only.  Gold labels are kept out
of every acquisition and stopping function and are joined only after a TEST
episode has frozen its delivered answer.  The module's pure policy functions
are stdlib-only; transformers is imported only by the formal entry point.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from math import comb, log, sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA = "a11-boolq-ordered-correctness-cost-v2"
LEDGERS = (
    "task_manifest.jsonl",
    "rollout_trace.jsonl",
    "stop_decision.jsonl",
    "cancellation_ledger.jsonl",
    "episode_result.jsonl",
)
CANONICAL_WORD = re.compile(r"\b(?:yes|no)\b", re.IGNORECASE)


class FormalError(RuntimeError):
    failure_class = "infrastructure_preflight"


class DataError(FormalError):
    failure_class = "data_evaluator"


class IntegrityError(FormalError):
    failure_class = "online_policy_integrity"


class CalibrationRejected(FormalError):
    failure_class = "calibration_screen_rejection"


def require(condition: bool, message: str, exc: type = FormalError) -> None:
    if not condition:
        raise exc(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verify_model_snapshot(model_path: Path, model_spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify the complete staged snapshot, not only its weight shards."""

    require(model_path.is_dir() and not model_path.is_symlink(), "model snapshot root is not a regular directory")
    expected = dict(model_spec["snapshot_files"])
    require(len(expected) == 14, "frozen model snapshot must contain exactly 14 files")
    actual_paths: Dict[str, Path] = {}
    for entry in model_path.iterdir():
        if entry.name == ".cache":
            require(entry.is_dir() and not entry.is_symlink(), "model .cache entry is not a regular directory")
            continue
        require(not entry.is_symlink(), "model snapshot contains symlink: " + entry.name)
        require(entry.is_file(), "model snapshot contains non-regular entry: " + entry.name)
        actual_paths[entry.name] = entry
    require(set(actual_paths) == set(expected), "model snapshot file set mismatch")
    computed = {name: sha256_file(actual_paths[name]) for name in sorted(actual_paths)}
    for name, expected_sha in expected.items():
        require(computed[name] == expected_sha, "model snapshot SHA mismatch: " + name)
    ledger = "".join("%s  %s\n" % (computed[name], name) for name in sorted(computed))
    ledger_sha = sha256_bytes(ledger.encode("utf-8"))
    require(ledger_sha == model_spec["snapshot_manifest_sha256"], "model snapshot manifest SHA mismatch")
    return {
        "contract": model_spec["snapshot_manifest_contract"],
        "file_count": len(computed),
        "files": computed,
        "ledger_sha256": ledger_sha,
    }


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("xb") as handle:
        handle.write(canonical_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def exclusive_text(path: Path, text: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


class LedgerSet:
    """Thread-safe append-only JSONL writers, created with O_EXCL semantics."""

    def __init__(self, root: Path) -> None:
        self._handles: Dict[str, Any] = {}
        self._locks: Dict[str, threading.Lock] = {}
        for name in LEDGERS:
            self._handles[name] = (root / name).open("x", encoding="utf-8", buffering=1)
            self._locks[name] = threading.Lock()

    def append(self, name: str, row: Mapping[str, Any]) -> None:
        require(name in self._handles, "unknown ledger " + name)
        payload = canonical_bytes(dict(row)).decode("utf-8") + "\n"
        with self._locks[name]:
            self._handles[name].write(payload)

    def flush(self) -> None:
        for name, handle in self._handles.items():
            with self._locks[name]:
                handle.flush()
                os.fsync(handle.fileno())

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()


@dataclass(frozen=True)
class Task:
    task_id: str
    passage_id: str
    internal_split: str
    source_split: str
    source_index: int
    passage: str
    question: str
    gold: bool
    shadow: bool = False

    def prompt_only(self, rendered_prompt: str, input_tokens: int) -> "PromptTask":
        return PromptTask(
            task_id=self.task_id,
            passage_id=self.passage_id,
            internal_split=self.internal_split,
            source_split=self.source_split,
            source_index=self.source_index,
            passage=self.passage,
            question=self.question,
            rendered_prompt=rendered_prompt,
            prompt_sha256=sha256_bytes(rendered_prompt.encode("utf-8")),
            input_tokens=input_tokens,
            shadow=self.shadow,
        )


@dataclass(frozen=True)
class PromptTask:
    """Policy-visible task; deliberately has no gold field."""

    task_id: str
    passage_id: str
    internal_split: str
    source_split: str
    source_index: int
    passage: str
    question: str
    rendered_prompt: str
    prompt_sha256: str
    input_tokens: int
    shadow: bool


@dataclass(frozen=True)
class ParsedVote:
    yes: bool
    first_canonical_word: Optional[str]
    strict_compliant: bool


@dataclass(frozen=True)
class Completion:
    request_id: str
    response_id: Optional[str]
    raw_output: str
    vote: ParsedVote
    dispatch_monotonic_ns: int
    first_token_monotonic_ns: int
    final_token_monotonic_ns: int
    response_arrival_monotonic_ns: int
    dispatch_utc: str
    first_token_utc: str
    final_token_utc: str
    response_arrival_utc: str
    local_input_tokens: int
    local_output_tokens: int
    provider_prompt_tokens: Any
    provider_completion_tokens: Any
    provider_total_tokens: Any
    finish_reason: Optional[str]
    seed: int
    server_rank: int


@dataclass
class EpisodeOutcome:
    task_id: str
    source_split: str
    arm: str
    episode_id: str
    votes: List[bool]
    delivered_yes: bool
    stop_k: int
    decision_monotonic_ns: int
    decision_utc: str
    first_dispatch_monotonic_ns: int
    first_token_monotonic_ns: int
    completed_output_tokens: int
    completed_input_tokens: int
    strict_compliant_count: int
    missing_canonical_count: int
    decision_sha256: str
    shadow_full_yes: Optional[bool] = None

    @property
    def time_to_final_answer_ns(self) -> int:
        return self.decision_monotonic_ns - self.first_dispatch_monotonic_ns

    @property
    def time_to_first_token_ns(self) -> int:
        return self.first_token_monotonic_ns - self.first_dispatch_monotonic_ns


def parse_vote(text: str) -> ParsedVote:
    words = CANONICAL_WORD.findall(text)
    first = words[0].casefold() if words else None
    normalized = text.strip().casefold()
    return ParsedVote(
        yes=(first == "yes"),
        first_canonical_word=first,
        strict_compliant=normalized in ("yes", "no"),
    )


def majority_yes(votes: Sequence[bool]) -> bool:
    return sum(bool(v) for v in votes) > len(votes) / 2


def hyper_pmf(N: int, K: int, k: int, x: int) -> float:
    if x < max(0, k - (N - K)) or x > min(k, K):
        return 0.0
    return comb(K, x) * comb(N - K, k - x) / comb(N, k)


def build_cert_table(H: Sequence[float], N: int) -> List[List[Optional[float]]]:
    """Posterior flip table; unsupported zero-denominator states are None."""

    require(len(H) == N + 1, "H has wrong support", IntegrityError)
    table: List[List[Optional[float]]] = []
    for k in range(N + 1):
        row: List[Optional[float]] = []
        for x in range(k + 1):
            den = 0.0
            num = 0.0
            prefix_side = x > k / 2
            for K, mass in enumerate(H):
                weight = mass * hyper_pmf(N, K, k, x)
                den += weight
                if (K > N / 2) != prefix_side:
                    num += weight
            row.append(num / den if den > 0.0 else None)
        table.append(row)
    return table


def fixed_flip_probability(N: int, K: int, k: int) -> float:
    full = K > N / 2
    return sum(
        hyper_pmf(N, K, k, x)
        for x in range(k + 1)
        if (x > k / 2) != full
    )


def dp_adaptive(N: int, K: int, table: Sequence[Sequence[Optional[float]]], alpha: float, minimum_k: int) -> Tuple[float, float]:
    """Exact replay flip probability and expected k for the frozen rule."""

    full = K > N / 2
    reach: Dict[Tuple[int, int], float] = {(0, 0): 1.0}
    flip = 0.0
    expected_k = 0.0
    for k in range(N):
        nxt: Dict[Tuple[int, int], float] = {}
        for (kk, x), probability in reach.items():
            require(kk == k, "DP state clock mismatch", IntegrityError)
            cert = table[k][x]
            stop = k >= minimum_k and cert is not None and cert <= alpha
            if stop:
                expected_k += probability * k
                flip += probability * ((x > k / 2) != full)
                continue
            p1 = (K - x) / (N - k)
            p0 = 1.0 - p1
            if p0 > 0:
                nxt[(k + 1, x)] = nxt.get((k + 1, x), 0.0) + probability * p0
            if p1 > 0:
                nxt[(k + 1, x + 1)] = nxt.get((k + 1, x + 1), 0.0) + probability * p1
        reach = nxt
    for (k, x), probability in reach.items():
        require(k == N and x == K, "terminal DP state invalid", IntegrityError)
        expected_k += probability * N
    return flip, expected_k


def empirical_bernstein_ucb(values: Sequence[float], delta: float) -> float:
    require(len(values) > 1, "empirical Bernstein needs at least two values", IntegrityError)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean + sqrt(2 * variance * log(4 / delta) / len(values)) + 7 * log(4 / delta) / (3 * (len(values) - 1))


def deterministic_seed(base: str, *parts: Any) -> int:
    material = "\0".join([base] + [str(part) for part in parts]).encode("utf-8")
    return int(hashlib.sha256(material).hexdigest()[:8], 16) & 0x7FFFFFFF


def task_identity(row: Mapping[str, Any]) -> str:
    material = (row["passage"].strip() + "\0" + row["question"].strip()).encode("utf-8")
    return sha256_bytes(material)


def passage_identity(row: Mapping[str, Any]) -> str:
    return sha256_bytes(row["passage"].strip().encode("utf-8"))


def selected_manifest_text(tasks: Sequence[Task]) -> str:
    return "\n".join(
        "%s\t%s\t%s\t%s" % (task.task_id, task.passage_id, task.source_split, "true" if task.gold else "false")
        for task in tasks
    )


def shadow_manifest_text(tasks: Sequence[Task], seed: str) -> str:
    ordered = sorted(tasks, key=lambda task: sha256_bytes((seed + ":shadow:" + task.task_id).encode("utf-8")))
    return "\n".join(task.task_id for task in ordered if task.shadow)


def load_and_allocate(dataset: Path, protocol: Mapping[str, Any]) -> Tuple[List[Task], List[Task], List[Task]]:
    carrier = protocol["carrier"]
    require(dataset.stat().st_size == carrier["canonical_jsonl_bytes"], "dataset byte count mismatch", DataError)
    require(sha256_file(dataset) == carrier["canonical_jsonl_sha256"], "dataset SHA-256 mismatch", DataError)
    rows: List[Mapping[str, Any]] = []
    with dataset.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DataError("invalid dataset JSON at line %d: %s" % (line_number, error))
            require(set(row) == {"answer", "passage", "question", "source_index", "source_split"}, "dataset schema mismatch", DataError)
            require(type(row["answer"]) is bool, "non-boolean BoolQ answer", DataError)
            require(isinstance(row["passage"], str) and bool(row["passage"].strip()), "empty passage", DataError)
            require(isinstance(row["question"], str) and bool(row["question"].strip()), "empty question", DataError)
            require(row["source_split"] in ("train", "validation"), "unexpected source split", DataError)
            require(type(row["source_index"]) is int and row["source_index"] >= 0, "invalid source index", DataError)
            rows.append(row)
    require(len(rows) == carrier["rows"], "dataset row count mismatch", DataError)
    require(len({(row["passage"], row["question"]) for row in rows}) == carrier["unique_prompt_pairs"], "duplicate prompt pair", DataError)
    task_ids = [task_identity(row) for row in rows]
    require(len(set(task_ids)) == len(task_ids), "duplicate task identity", DataError)
    passage_ids = [passage_identity(row) for row in rows]
    require(len(set(passage_ids)) == carrier["unique_stripped_passages"], "unique stripped-passage count mismatch", DataError)
    allocation = protocol["allocation"]
    seed = allocation["seed"]
    grouped: Dict[str, List[int]] = defaultdict(list)
    for index, passage_id in enumerate(passage_ids):
        grouped[passage_id].append(index)
    representatives = [
        min(indices, key=lambda index: sha256_bytes((seed + ":within-passage:" + task_ids[index]).encode("utf-8")))
        for indices in grouped.values()
    ]
    require(len(representatives) == allocation["eligible_passage_representatives"], "eligible passage representative count mismatch", DataError)
    require(len(rows) - len(representatives) == allocation["duplicate_passage_rows_discarded"], "discarded duplicate-passage count mismatch", DataError)
    order = sorted(representatives, key=lambda index: sha256_bytes((seed + ":passage:" + passage_ids[index]).encode("utf-8")))
    n_fit, n_cal, n_test = allocation["fit"], allocation["cal"], allocation["test"]
    require(len(order) >= n_fit + n_cal + n_test, "fewer than 9000 eligible tasks", DataError)
    require(len(order) - n_fit - n_cal - n_test == allocation["unused_representatives"], "unused representative count mismatch", DataError)
    shadow_order = sorted(
        order[n_fit + n_cal:n_fit + n_cal + n_test],
        key=lambda index: sha256_bytes((seed + ":shadow:" + task_ids[index]).encode("utf-8")),
    )
    shadow_ids = {task_ids[index] for index in shadow_order[: allocation["shadow_test"]]}

    def build(indices: Sequence[int], split: str) -> List[Task]:
        return [
            Task(
                task_id=task_ids[index],
                passage_id=passage_ids[index],
                internal_split=split,
                source_split=str(rows[index]["source_split"]),
                source_index=int(rows[index]["source_index"]),
                passage=str(rows[index]["passage"]),
                question=str(rows[index]["question"]),
                gold=bool(rows[index]["answer"]),
                shadow=task_ids[index] in shadow_ids,
            )
            for index in indices
        ]

    allocated = (
        build(order[:n_fit], "FIT"),
        build(order[n_fit:n_fit + n_cal], "CAL"),
        build(order[n_fit + n_cal:n_fit + n_cal + n_test], "TEST"),
    )
    selected = allocated[0] + allocated[1] + allocated[2]
    require(sha256_bytes(selected_manifest_text(selected).encode("utf-8")) == allocation["selected_manifest_sha256"], "selected manifest SHA mismatch", DataError)
    for split, tasks in zip(("FIT", "CAL", "TEST"), allocated):
        require(sha256_bytes(selected_manifest_text(tasks).encode("utf-8")) == allocation["split_manifest_sha256"][split], split + " manifest SHA mismatch", DataError)
        require(len({task.passage_id for task in tasks}) == len(tasks), split + " contains duplicate passage", DataError)
    passage_sets = [{task.passage_id for task in tasks} for tasks in allocated]
    require(not (passage_sets[0] & passage_sets[1] or passage_sets[0] & passage_sets[2] or passage_sets[1] & passage_sets[2]), "FIT/CAL/TEST passage overlap", DataError)
    require(sha256_bytes(shadow_manifest_text(allocated[2], seed).encode("utf-8")) == allocation["shadow_manifest_sha256"], "shadow manifest SHA mismatch", DataError)
    return allocated


def render_prompt(task: Task, tokenizer: Any, prompt_spec: Mapping[str, Any]) -> PromptTask:
    messages = [
        {"role": "system", "content": prompt_spec["system"]},
        {"role": "user", "content": prompt_spec["user_template"].format(passage=task.passage, question=task.question)},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_tokens = len(tokenizer.encode(rendered, add_special_tokens=False))
    return task.prompt_only(rendered, input_tokens)


class OpenAICompletionClient:
    def __init__(self, server_url: str, server_rank: int, served_model: str, inference: Mapping[str, Any], tokenizer: Any) -> None:
        self.server_url = server_url.rstrip("/")
        self.server_rank = server_rank
        self.served_model = served_model
        self.inference = inference
        self.tokenizer = tokenizer

    def complete(self, task: PromptTask, request_id: str, seed: int) -> Completion:
        payload = {
            "model": self.served_model,
            "prompt": task.rendered_prompt,
            "temperature": self.inference["temperature"],
            "top_p": self.inference["top_p"],
            "max_tokens": self.inference["max_tokens"],
            "n": 1,
            "seed": seed,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        request = urllib.request.Request(
            self.server_url + "/v1/completions",
            data=canonical_bytes(payload),
            headers={"Content-Type": "application/json", "X-Request-Id": request_id},
            method="POST",
        )
        dispatch_monotonic_ns = time.monotonic_ns()
        dispatch_utc = utc_now()
        chunks: List[str] = []
        first_token_monotonic_ns: Optional[int] = None
        final_token_monotonic_ns: Optional[int] = None
        first_token_utc: Optional[str] = None
        final_token_utc: Optional[str] = None
        response_id: Optional[str] = None
        finish_reason: Optional[str] = None
        usage: Mapping[str, Any] = {}
        saw_done = False
        try:
            with urllib.request.urlopen(request, timeout=float(self.inference["request_timeout_seconds"])) as response:
                require(response.status == 200, "completion HTTP status %s" % response.status)
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="strict").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        saw_done = True
                        break
                    event = json.loads(data)
                    response_id = event.get("id", response_id)
                    if isinstance(event.get("usage"), Mapping):
                        usage = event["usage"]
                    choices = event.get("choices") or []
                    if choices:
                        text = choices[0].get("text") or ""
                        if text:
                            moment = time.monotonic_ns()
                            moment_utc = utc_now()
                            if first_token_monotonic_ns is None:
                                first_token_monotonic_ns = moment
                                first_token_utc = moment_utc
                            final_token_monotonic_ns = moment
                            final_token_utc = moment_utc
                            chunks.append(text)
                        if choices[0].get("finish_reason") is not None:
                            finish_reason = choices[0]["finish_reason"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, UnicodeDecodeError) as error:
            raise FormalError("request %s failed without retry: %s" % (request_id, error))
        require(saw_done, "request %s ended without streaming EOS marker" % request_id)
        require(finish_reason is not None, "request %s has no finish_reason" % request_id)
        arrival_monotonic_ns = time.monotonic_ns()
        arrival_utc = utc_now()
        if first_token_monotonic_ns is None:
            first_token_monotonic_ns = arrival_monotonic_ns
            first_token_utc = arrival_utc
        if final_token_monotonic_ns is None:
            final_token_monotonic_ns = arrival_monotonic_ns
            final_token_utc = arrival_utc
        raw_output = "".join(chunks)
        return Completion(
            request_id=request_id,
            response_id=response_id,
            raw_output=raw_output,
            vote=parse_vote(raw_output),
            dispatch_monotonic_ns=dispatch_monotonic_ns,
            first_token_monotonic_ns=first_token_monotonic_ns,
            final_token_monotonic_ns=final_token_monotonic_ns,
            response_arrival_monotonic_ns=arrival_monotonic_ns,
            dispatch_utc=dispatch_utc,
            first_token_utc=first_token_utc,
            final_token_utc=final_token_utc,
            response_arrival_utc=arrival_utc,
            local_input_tokens=task.input_tokens,
            local_output_tokens=len(self.tokenizer.encode(raw_output, add_special_tokens=False)),
            provider_prompt_tokens=usage.get("prompt_tokens", "unknown"),
            provider_completion_tokens=usage.get("completion_tokens", "unknown"),
            provider_total_tokens=usage.get("total_tokens", "unknown"),
            finish_reason=finish_reason,
            seed=seed,
            server_rank=self.server_rank,
        )


def trace_row(task: PromptTask, episode_id: str, arm: str, rollout_index: int, completion: Completion, segment: str, excluded: bool) -> Dict[str, Any]:
    return {
        "schema": SCHEMA,
        "task_id": task.task_id,
        "passage_id": task.passage_id,
        "internal_split": task.internal_split,
        "source_split": task.source_split,
        "episode_id": episode_id,
        "arm": arm,
        "segment": segment,
        "rollout_index": rollout_index,
        "request_id": completion.request_id,
        "server_rank": completion.server_rank,
        "sampling_seed": completion.seed,
        "prompt_sha256": task.prompt_sha256,
        "dispatch_monotonic_ns": completion.dispatch_monotonic_ns,
        "first_token_monotonic_ns": completion.first_token_monotonic_ns,
        "final_token_monotonic_ns": completion.final_token_monotonic_ns,
        "response_arrival_monotonic_ns": completion.response_arrival_monotonic_ns,
        "dispatch_utc": completion.dispatch_utc,
        "first_token_utc": completion.first_token_utc,
        "final_token_utc": completion.final_token_utc,
        "response_arrival_utc": completion.response_arrival_utc,
        "raw_output": completion.raw_output,
        "raw_output_sha256": sha256_bytes(completion.raw_output.encode("utf-8")),
        "parsed_yes": completion.vote.yes,
        "first_canonical_word": completion.vote.first_canonical_word,
        "strict_compliant": completion.vote.strict_compliant,
        "local_input_tokens": completion.local_input_tokens,
        "local_completed_output_tokens": completion.local_output_tokens,
        "provider_prompt_tokens": completion.provider_prompt_tokens,
        "provider_completion_tokens": completion.provider_completion_tokens,
        "provider_total_tokens": completion.provider_total_tokens,
        "provider_billable_tokens": "unknown",
        "finish_reason": completion.finish_reason,
        "retry_count": 0,
        "status": "completed",
        "excluded_from_primary_cost": excluded,
    }


def episode_summary(task: PromptTask, arm: str, episode_id: str, votes: List[bool], completions: Sequence[Completion], stop_k: int, decision_monotonic_ns: int, decision_utc: str, decision_sha256: str, shadow_full: Optional[bool] = None) -> EpisodeOutcome:
    primary = list(completions[:stop_k])
    require(len(primary) == stop_k and stop_k > 0, "episode primary completion count mismatch", IntegrityError)
    return EpisodeOutcome(
        task_id=task.task_id,
        source_split=task.source_split,
        arm=arm,
        episode_id=episode_id,
        votes=list(votes),
        delivered_yes=majority_yes(votes[:stop_k]),
        stop_k=stop_k,
        decision_monotonic_ns=decision_monotonic_ns,
        decision_utc=decision_utc,
        first_dispatch_monotonic_ns=primary[0].dispatch_monotonic_ns,
        first_token_monotonic_ns=primary[0].first_token_monotonic_ns,
        completed_output_tokens=sum(item.local_output_tokens for item in primary),
        completed_input_tokens=sum(item.local_input_tokens for item in primary),
        strict_compliant_count=sum(item.vote.strict_compliant for item in primary),
        missing_canonical_count=sum(item.vote.first_canonical_word is None for item in primary),
        decision_sha256=decision_sha256,
        shadow_full_yes=shadow_full,
    )


def run_full_episode(task: PromptTask, client: OpenAICompletionClient, ledger: LedgerSet, base_seed: str, N: int, stage: str, arm: str) -> EpisodeOutcome:
    episode_id = "%s:%s:%s" % (stage, arm, task.task_id)
    votes: List[bool] = []
    completions: List[Completion] = []
    for rollout_index in range(1, N + 1):
        request_id = "%s:%02d" % (episode_id, rollout_index)
        seed = deterministic_seed(base_seed, task.task_id, episode_id, rollout_index)
        completion = client.complete(task, request_id, seed)
        completions.append(completion)
        votes.append(completion.vote.yes)
        ledger.append("rollout_trace.jsonl", trace_row(task, episode_id, arm, rollout_index, completion, "primary", False))
    decision_monotonic_ns = time.monotonic_ns()
    decision_utc = utc_now()
    decision = {
        "schema": SCHEMA,
        "task_id": task.task_id,
        "internal_split": task.internal_split,
        "episode_id": episode_id,
        "arm": arm,
        "policy": "FULL-N",
        "stop_k": N,
        "yes_count": sum(votes),
        "prefix_votes": votes,
        "certificate_table_sha256": None,
        "certificate_value": 0.0,
        "posterior_supported": True,
        "delivered_yes": majority_yes(votes),
        "decision_monotonic_ns": decision_monotonic_ns,
        "decision_utc": decision_utc,
    }
    decision_sha = sha256_json(decision)
    ledger.append("stop_decision.jsonl", dict(decision, decision_payload_sha256=decision_sha))
    return episode_summary(task, arm, episode_id, votes, completions, N, decision_monotonic_ns, decision_utc, decision_sha)


def run_online_episode(task: PromptTask, client: OpenAICompletionClient, ledger: LedgerSet, base_seed: str, N: int, minimum_k: int, alpha: float, table: Sequence[Sequence[Optional[float]]], table_sha256: str) -> EpisodeOutcome:
    arm = "BAYES-H-online"
    episode_id = "TEST:%s:%s" % (arm, task.task_id)
    votes: List[bool] = []
    completions: List[Completion] = []
    stop_k: Optional[int] = None
    certificate: Optional[float] = None
    supported = False
    decision_monotonic_ns: Optional[int] = None
    decision_utc: Optional[str] = None
    for rollout_index in range(1, N + 1):
        request_id = "%s:%02d" % (episode_id, rollout_index)
        seed = deterministic_seed(base_seed, task.task_id, episode_id, rollout_index)
        completion = client.complete(task, request_id, seed)
        completions.append(completion)
        votes.append(completion.vote.yes)
        ledger.append("rollout_trace.jsonl", trace_row(task, episode_id, arm, rollout_index, completion, "primary", False))
        x = sum(votes)
        certificate = table[rollout_index][x]
        supported = certificate is not None
        should_stop = rollout_index >= minimum_k and supported and certificate <= alpha
        if should_stop or rollout_index == N:
            stop_k = rollout_index
            decision_monotonic_ns = time.monotonic_ns()
            decision_utc = utc_now()
            break
    require(stop_k is not None and decision_monotonic_ns is not None and decision_utc is not None, "online episode did not terminate", IntegrityError)
    delivered = majority_yes(votes)
    decision = {
        "schema": SCHEMA,
        "task_id": task.task_id,
        "internal_split": task.internal_split,
        "episode_id": episode_id,
        "arm": arm,
        "policy": arm,
        "alpha": alpha,
        "stop_k": stop_k,
        "yes_count": sum(votes),
        "prefix_votes": list(votes),
        "certificate_table_sha256": table_sha256,
        "certificate_value": certificate,
        "posterior_supported": supported,
        "forced_full_fallback": stop_k == N,
        "delivered_yes": delivered,
        "decision_monotonic_ns": decision_monotonic_ns,
        "decision_utc": decision_utc,
    }
    decision_sha = sha256_json(decision)
    ledger.append("stop_decision.jsonl", dict(decision, decision_payload_sha256=decision_sha))
    ledger.append("cancellation_ledger.jsonl", {
        "schema": SCHEMA,
        "task_id": task.task_id,
        "episode_id": episode_id,
        "arm": arm,
        "decision_monotonic_ns": decision_monotonic_ns,
        "decision_utc": decision_utc,
        "in_flight_request_ids": [],
        "cancellation_attempted": False,
        "cancellation_status": "not_applicable_sequential_no_prefetch",
        "cancellation_dispatch_monotonic_ns": None,
        "cancellation_dispatch_utc": None,
        "provider_acknowledgement": "unknown",
        "provider_acknowledgement_monotonic_ns": None,
        "provider_acknowledgement_utc": None,
        "post_stop_completed_output_tokens": 0,
        "provider_billable_tokens_after_stop": "unknown",
    })
    return episode_summary(task, arm, episode_id, votes, completions, stop_k, decision_monotonic_ns, decision_utc, decision_sha)


def continue_shadow_episode(task: PromptTask, outcome: EpisodeOutcome, client: OpenAICompletionClient, ledger: LedgerSet, base_seed: str, N: int, global_primary_seal_monotonic_ns: int) -> None:
    """Continue a preselected online trajectory only after all TEST primaries seal."""

    require(task.shadow and outcome.arm == "BAYES-H-online", "invalid shadow continuation target", IntegrityError)
    require(len(outcome.votes) == outcome.stop_k, "shadow prefix length mismatch", IntegrityError)
    for rollout_index in range(outcome.stop_k + 1, N + 1):
        request_id = "%s:%02d" % (outcome.episode_id, rollout_index)
        seed = deterministic_seed(base_seed, task.task_id, outcome.episode_id, rollout_index)
        completion = client.complete(task, request_id, seed)
        require(completion.dispatch_monotonic_ns > outcome.decision_monotonic_ns, "shadow dispatched before task stop decision", IntegrityError)
        require(completion.dispatch_monotonic_ns > global_primary_seal_monotonic_ns, "shadow dispatched before global TEST primary seal", IntegrityError)
        outcome.votes.append(completion.vote.yes)
        ledger.append("rollout_trace.jsonl", trace_row(task, outcome.episode_id, outcome.arm, rollout_index, completion, "shadow_continuation", True))
    require(len(outcome.votes) == N, "shadow trajectory did not reach FULL-N", IntegrityError)
    outcome.shadow_full_yes = majority_yes(outcome.votes)


def assign_server(task_id: str, server_count: int) -> int:
    return int(task_id[:16], 16) % server_count


def execute_full_stage(tasks: Sequence[PromptTask], clients: Sequence[OpenAICompletionClient], ledger: LedgerSet, protocol: Mapping[str, Any], stage: str, arm: str) -> List[EpisodeOutcome]:
    maximum = int(protocol["inference"]["max_concurrent_tasks_per_server"])
    N = int(protocol["policy"]["N"])
    base_seed = protocol["allocation"]["seed"]
    executors = [concurrent.futures.ThreadPoolExecutor(max_workers=maximum) for _ in clients]
    futures: List[concurrent.futures.Future] = []
    try:
        for task in tasks:
            rank = assign_server(task.task_id, len(clients))
            futures.append(executors[rank].submit(run_full_episode, task, clients[rank], ledger, base_seed, N, stage, arm))
        outcomes = [future.result() for future in concurrent.futures.as_completed(futures)]
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
    return sorted(outcomes, key=lambda item: item.task_id)


def balanced_arm_orders(seed: str, tasks: Sequence[PromptTask]) -> Dict[str, Tuple[str, str]]:
    ranked = sorted(tasks, key=lambda task: sha256_bytes((seed + ":test-arm-order:" + task.task_id).encode("utf-8")))
    require(len(ranked) % 2 == 0, "balanced arm order requires an even TEST size", IntegrityError)
    full_first = {task.task_id for task in ranked[: len(ranked) // 2]}
    return {
        task.task_id: (("FULL-N", "BAYES-H-online") if task.task_id in full_first else ("BAYES-H-online", "FULL-N"))
        for task in tasks
    }


def execute_test(tasks: Sequence[PromptTask], clients: Sequence[OpenAICompletionClient], ledger: LedgerSet, protocol: Mapping[str, Any], table: Sequence[Sequence[Optional[float]]], table_sha256: str, run_dir: Path) -> List[Tuple[EpisodeOutcome, EpisodeOutcome]]:
    maximum = int(protocol["inference"]["max_concurrent_tasks_per_server"])
    N = int(protocol["policy"]["N"])
    minimum_k = int(protocol["policy"]["minimum_stop_k"])
    alpha = float(protocol["policy"]["primary_alpha"])
    base_seed = protocol["allocation"]["seed"]
    arm_orders = balanced_arm_orders(base_seed, tasks)

    def one(task: PromptTask, client: OpenAICompletionClient) -> Tuple[EpisodeOutcome, EpisodeOutcome]:
        outputs: Dict[str, EpisodeOutcome] = {}
        for arm in arm_orders[task.task_id]:
            if arm == "FULL-N":
                outputs[arm] = run_full_episode(task, client, ledger, base_seed, N, "TEST", arm)
            else:
                outputs[arm] = run_online_episode(task, client, ledger, base_seed, N, minimum_k, alpha, table, table_sha256)
        return outputs["FULL-N"], outputs["BAYES-H-online"]

    executors = [concurrent.futures.ThreadPoolExecutor(max_workers=maximum) for _ in clients]
    futures: List[concurrent.futures.Future] = []
    try:
        for task in tasks:
            rank = assign_server(task.task_id, len(clients))
            futures.append(executors[rank].submit(one, task, clients[rank]))
        pairs = [future.result() for future in concurrent.futures.as_completed(futures)]
    finally:
        for executor in executors:
            executor.shutdown(wait=True, cancel_futures=True)
    pairs = sorted(pairs, key=lambda pair: pair[0].task_id)
    ledger.flush()
    primary_seal_monotonic_ns = time.monotonic_ns()
    primary_seal_utc = utc_now()
    decision_pairs = sorted(
        (outcome.episode_id, outcome.decision_sha256)
        for pair in pairs
        for outcome in pair
    )
    primary_seal = {
        "schema": SCHEMA,
        "kind": "all_test_primary_decisions_sealed",
        "test_tasks": len(tasks),
        "decision_count": len(decision_pairs),
        "decision_set_sha256": sha256_json(decision_pairs),
        "seal_monotonic_ns": primary_seal_monotonic_ns,
        "seal_utc": primary_seal_utc,
    }
    atomic_json(run_dir / "locks" / "all_test_primary_decisions_sealed.json", dict(primary_seal, seal_payload_sha256=sha256_json(primary_seal)))

    online_by_task = {online.task_id: online for _, online in pairs}
    shadow_tasks = [task for task in tasks if task.shadow]
    require(len(shadow_tasks) == protocol["allocation"]["shadow_test"], "shadow task count mismatch", IntegrityError)
    shadow_executors = [concurrent.futures.ThreadPoolExecutor(max_workers=maximum) for _ in clients]
    shadow_futures: List[concurrent.futures.Future] = []
    try:
        for task in shadow_tasks:
            rank = assign_server(task.task_id, len(clients))
            shadow_futures.append(shadow_executors[rank].submit(
                continue_shadow_episode,
                task,
                online_by_task[task.task_id],
                clients[rank],
                ledger,
                base_seed,
                N,
                primary_seal_monotonic_ns,
            ))
        for future in concurrent.futures.as_completed(shadow_futures):
            future.result()
    finally:
        for executor in shadow_executors:
            executor.shutdown(wait=True, cancel_futures=True)
    return pairs


def fit_lock(outcomes: Sequence[EpisodeOutcome], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    N = int(protocol["policy"]["N"])
    require(len(outcomes) == protocol["allocation"]["fit"], "FIT outcome count mismatch", IntegrityError)
    H = [0.0] * (N + 1)
    for outcome in outcomes:
        require(len(outcome.votes) == N, "incomplete FIT trace", IntegrityError)
        H[sum(outcome.votes)] += 1.0
    H = [mass / len(outcomes) for mass in H]
    table = build_cert_table(H, N)
    table_sha = sha256_json(table)
    return {
        "schema": SCHEMA,
        "created_at_utc": utc_now(),
        "N": N,
        "fit_tasks": len(outcomes),
        "H": H,
        "certificate_table": table,
        "certificate_table_sha256": table_sha,
        "unsupported_state_count": sum(value is None for row in table for value in row),
        "unsupported_state_action": "continue_to_FULL-N",
    }


def calibration_lock(cal_outcomes: Sequence[EpisodeOutcome], fit: Mapping[str, Any], protocol: Mapping[str, Any]) -> Dict[str, Any]:
    N = int(protocol["policy"]["N"])
    table = fit["certificate_table"]
    counts = [sum(outcome.votes) for outcome in cal_outcomes]
    require(len(counts) == protocol["allocation"]["cal"], "CAL outcome count mismatch", IntegrityError)
    primary_alpha = float(protocol["policy"]["primary_alpha"])
    delta_cal = float(protocol["policy"]["delta_cal"])
    adaptive = [dp_adaptive(N, K, table, primary_alpha, int(protocol["policy"]["minimum_stop_k"])) for K in counts]
    values = [item[0] for item in adaptive]
    primary = {
        "kind": "BAYES-H",
        "alpha_stop": primary_alpha,
        "mean_exact_dp_flip": sum(values) / len(values),
        "mean_exact_dp_k": sum(item[1] for item in adaptive) / len(adaptive),
        "eb_ucb": empirical_bernstein_ucb(values, delta_cal),
        "loss_source": "exact g_r(K) from complete CAL K; not observed chronological flip",
    }
    accepted = primary["eb_ucb"] <= primary_alpha
    return {
        "schema": SCHEMA,
        "created_at_utc": utc_now(),
        "fit_lock_sha256": sha256_json(fit),
        "certificate_table_sha256": fit["certificate_table_sha256"],
        "cal_tasks": len(counts),
        "delta_cal": delta_cal,
        "family_size": 1,
        "alpha_stop": primary_alpha,
        "primary_candidate": primary,
        "primary_accepted": accepted,
        "gate": "eb_ucb <= primary_alpha",
    }


def percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.floor(probability * (len(ordered) - 1)))))
    return ordered[index]


def input_length_summary(values: Sequence[int], max_tokens: int, max_model_len: int) -> Dict[str, Any]:
    require(bool(values), "empty rendered input-length list", DataError)
    summary = {
        "tasks": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "p99_9": percentile(values, 0.999),
        "max": max(values),
        "max_tokens": max_tokens,
        "max_model_len": max_model_len,
        "above_input_budget": sum(value + max_tokens > max_model_len for value in values),
        "quantile_definition": "sorted_values[floor(p*(n-1))]",
    }
    require(summary["above_input_budget"] == 0, "rendered task exceeds frozen model context", DataError)
    return summary


def paired_analysis(pairs: Sequence[Tuple[EpisodeOutcome, EpisodeOutcome]], gold: Mapping[str, bool], source_split: Mapping[str, str], protocol: Mapping[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    for full, online in pairs:
        require(full.task_id == online.task_id, "unpaired TEST task", IntegrityError)
        task_gold = gold[full.task_id]
        rows.append({
            "task_id": full.task_id,
            "source_split": source_split[full.task_id],
            "arm_order": "FULL-N_first" if full.first_dispatch_monotonic_ns < online.first_dispatch_monotonic_ns else "BAYES-H-online_first",
            "gold": task_gold,
            "full_correct": full.delivered_yes == task_gold,
            "online_correct": online.delivered_yes == task_gold,
            "full_decision_sha256": full.decision_sha256,
            "online_decision_sha256": online.decision_sha256,
            "correctness_difference": int(online.delivered_yes == task_gold) - int(full.delivered_yes == task_gold),
            "full_completed_output_tokens": full.completed_output_tokens,
            "online_completed_output_tokens": online.completed_output_tokens,
            "completed_output_tokens_saved": full.completed_output_tokens - online.completed_output_tokens,
            "full_completed_input_tokens": full.completed_input_tokens,
            "online_completed_input_tokens": online.completed_input_tokens,
            "full_requests": full.stop_k,
            "online_requests": online.stop_k,
            "full_time_to_final_answer_ns": full.time_to_final_answer_ns,
            "online_time_to_final_answer_ns": online.time_to_final_answer_ns,
            "full_time_to_first_token_ns": full.time_to_first_token_ns,
            "online_time_to_first_token_ns": online.time_to_first_token_ns,
            "full_strict_compliance_rate": full.strict_compliant_count / full.stop_k,
            "online_strict_compliance_rate": online.strict_compliant_count / online.stop_k,
            "full_missing_canonical_votes": full.missing_canonical_count,
            "online_missing_canonical_votes": online.missing_canonical_count,
            "full_yes_count": sum(full.votes[:full.stop_k]),
            "online_yes_count": sum(online.votes[:online.stop_k]),
            "full_vote_tie": sum(full.votes[:full.stop_k]) * 2 == full.stop_k,
            "online_vote_tie": sum(online.votes[:online.stop_k]) * 2 == online.stop_k,
            "shadow_full_yes": online.shadow_full_yes,
            "shadow_flip": None if online.shadow_full_yes is None else online.delivered_yes != online.shadow_full_yes,
        })
    require(len(rows) == protocol["allocation"]["test"], "TEST pair count mismatch", IntegrityError)
    by_stratum: Dict[str, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_stratum[row["source_split"]].append(index)
    bootstrap_replicates = int(protocol["primary_analysis"]["bootstrap_replicates"])
    rng = random.Random(int(protocol["primary_analysis"]["bootstrap_seed"]))
    correctness_draws: List[float] = []
    token_reduction_draws: List[float] = []
    input_reduction_draws: List[float] = []
    request_reduction_draws: List[float] = []
    latency_reduction_draws: List[float] = []
    ttft_difference_draws: List[float] = []
    for _ in range(bootstrap_replicates):
        sampled: List[int] = []
        for indices in by_stratum.values():
            sampled.extend(indices[rng.randrange(len(indices))] for _ in indices)
        sums = {
            "correctness": 0.0,
            "full_output": 0.0,
            "online_output": 0.0,
            "full_input": 0.0,
            "online_input": 0.0,
            "full_requests": 0.0,
            "online_requests": 0.0,
            "full_latency": 0.0,
            "online_latency": 0.0,
            "full_ttft": 0.0,
            "online_ttft": 0.0,
        }
        for index in sampled:
            row = rows[index]
            sums["correctness"] += row["correctness_difference"]
            sums["full_output"] += row["full_completed_output_tokens"]
            sums["online_output"] += row["online_completed_output_tokens"]
            sums["full_input"] += row["full_completed_input_tokens"]
            sums["online_input"] += row["online_completed_input_tokens"]
            sums["full_requests"] += row["full_requests"]
            sums["online_requests"] += row["online_requests"]
            sums["full_latency"] += row["full_time_to_final_answer_ns"]
            sums["online_latency"] += row["online_time_to_final_answer_ns"]
            sums["full_ttft"] += row["full_time_to_first_token_ns"]
            sums["online_ttft"] += row["online_time_to_first_token_ns"]
        correctness_draws.append(sums["correctness"] / len(sampled))
        token_reduction_draws.append(1.0 - sums["online_output"] / sums["full_output"] if sums["full_output"] else 0.0)
        input_reduction_draws.append(1.0 - sums["online_input"] / sums["full_input"] if sums["full_input"] else 0.0)
        request_reduction_draws.append(1.0 - sums["online_requests"] / sums["full_requests"])
        latency_reduction_draws.append(1.0 - sums["online_latency"] / sums["full_latency"] if sums["full_latency"] else 0.0)
        ttft_difference_draws.append((sums["online_ttft"] - sums["full_ttft"]) / len(sampled))
    accuracy_full = sum(row["full_correct"] for row in rows) / len(rows)
    accuracy_online = sum(row["online_correct"] for row in rows) / len(rows)
    full_tokens = sum(row["full_completed_output_tokens"] for row in rows)
    online_tokens = sum(row["online_completed_output_tokens"] for row in rows)
    shadow = [row for row in rows if row["shadow_flip"] is not None]
    gold_yes = [row for row in rows if row["gold"]]
    gold_no = [row for row in rows if not row["gold"]]
    full_first_rows = [row for row in rows if row["arm_order"] == "FULL-N_first"]
    online_first_rows = [row for row in rows if row["arm_order"] == "BAYES-H-online_first"]
    correctness_lower = percentile(correctness_draws, 0.05)
    cost_lower = percentile(token_reduction_draws, 0.025)
    analysis = {
        "schema": SCHEMA,
        "test_tasks": len(rows),
        "source_split_counts": {key: len(value) for key, value in by_stratum.items()},
        "arm_order_counts": {"FULL-N_first": len(full_first_rows), "BAYES-H-online_first": len(online_first_rows)},
        "accuracy_full": accuracy_full,
        "accuracy_online": accuracy_online,
        "accuracy_difference_online_minus_full": accuracy_online - accuracy_full,
        "accuracy_difference_one_sided_95_lower": correctness_lower,
        "noninferiority_margin": -0.02,
        "noninferiority_pass": correctness_lower >= -0.02,
        "full_completed_output_tokens": full_tokens,
        "online_completed_output_tokens": online_tokens,
        "median_full_completed_output_tokens": statistics.median(row["full_completed_output_tokens"] for row in rows),
        "median_online_completed_output_tokens": statistics.median(row["online_completed_output_tokens"] for row in rows),
        "completed_output_token_reduction": 1.0 - online_tokens / full_tokens if full_tokens else 0.0,
        "completed_output_token_reduction_two_sided_95": [percentile(token_reduction_draws, 0.025), percentile(token_reduction_draws, 0.975)],
        "cost_reduction_pass": cost_lower > 0.0,
        "mean_full_requests": sum(row["full_requests"] for row in rows) / len(rows),
        "mean_online_requests": sum(row["online_requests"] for row in rows) / len(rows),
        "median_full_requests": statistics.median(row["full_requests"] for row in rows),
        "median_online_requests": statistics.median(row["online_requests"] for row in rows),
        "request_reduction_two_sided_95": [percentile(request_reduction_draws, 0.025), percentile(request_reduction_draws, 0.975)],
        "mean_full_time_to_final_answer_ns": sum(row["full_time_to_final_answer_ns"] for row in rows) / len(rows),
        "mean_online_time_to_final_answer_ns": sum(row["online_time_to_final_answer_ns"] for row in rows) / len(rows),
        "median_full_time_to_final_answer_ns": statistics.median(row["full_time_to_final_answer_ns"] for row in rows),
        "median_online_time_to_final_answer_ns": statistics.median(row["online_time_to_final_answer_ns"] for row in rows),
        "time_to_final_answer_reduction_two_sided_95": [percentile(latency_reduction_draws, 0.025), percentile(latency_reduction_draws, 0.975)],
        "mean_full_time_to_first_token_ns": sum(row["full_time_to_first_token_ns"] for row in rows) / len(rows),
        "mean_online_time_to_first_token_ns": sum(row["online_time_to_first_token_ns"] for row in rows) / len(rows),
        "time_to_first_token_online_minus_full_two_sided_95_ns": [percentile(ttft_difference_draws, 0.025), percentile(ttft_difference_draws, 0.975)],
        "mean_full_completed_input_tokens": sum(row["full_completed_input_tokens"] for row in rows) / len(rows),
        "mean_online_completed_input_tokens": sum(row["online_completed_input_tokens"] for row in rows) / len(rows),
        "median_full_completed_input_tokens": statistics.median(row["full_completed_input_tokens"] for row in rows),
        "median_online_completed_input_tokens": statistics.median(row["online_completed_input_tokens"] for row in rows),
        "completed_input_token_reduction_two_sided_95": [percentile(input_reduction_draws, 0.025), percentile(input_reduction_draws, 0.975)],
        "mean_full_strict_compliance_rate": sum(row["full_strict_compliance_rate"] for row in rows) / len(rows),
        "mean_online_strict_compliance_rate": sum(row["online_strict_compliance_rate"] for row in rows) / len(rows),
        "full_missing_canonical_votes": sum(row["full_missing_canonical_votes"] for row in rows),
        "online_missing_canonical_votes": sum(row["online_missing_canonical_votes"] for row in rows),
        "full_vote_ties": sum(row["full_vote_tie"] for row in rows),
        "online_vote_ties": sum(row["online_vote_tie"] for row in rows),
        "gold_yes_tasks": len(gold_yes),
        "gold_no_tasks": len(gold_no),
        "gold_yes_accuracy_full": sum(row["full_correct"] for row in gold_yes) / len(gold_yes) if gold_yes else None,
        "gold_yes_accuracy_online": sum(row["online_correct"] for row in gold_yes) / len(gold_yes) if gold_yes else None,
        "gold_no_accuracy_full": sum(row["full_correct"] for row in gold_no) / len(gold_no) if gold_no else None,
        "gold_no_accuracy_online": sum(row["online_correct"] for row in gold_no) / len(gold_no) if gold_no else None,
        "shadow_tasks": len(shadow),
        "shadow_flip_rate": sum(row["shadow_flip"] for row in shadow) / len(shadow) if shadow else None,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": protocol["primary_analysis"]["bootstrap_seed"],
        "forced_choice_output_token_limitation": protocol["primary_analysis"]["forced_choice_limitation"],
        "ttft_direction_prespecified": False,
    }
    analysis["scientific_endpoints_pass"] = analysis["noninferiority_pass"] and analysis["cost_reduction_pass"]
    return analysis, rows


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise IntegrityError("invalid JSONL %s:%d: %s" % (path.name, line_number, error))


def audit_ledgers(run_dir: Path, protocol: Mapping[str, Any], table: Sequence[Sequence[Optional[float]]]) -> Dict[str, Any]:
    table_sha256 = sha256_json(table)
    require(sha256_file(run_dir / "locks" / "posthoc_selected_with_gold.tsv") == protocol["allocation"]["selected_manifest_sha256"], "posthoc selected task lock SHA mismatch", IntegrityError)
    for split in ("FIT", "CAL", "TEST"):
        require(sha256_file(run_dir / "locks" / ("posthoc_" + split + "_with_gold.tsv")) == protocol["allocation"]["split_manifest_sha256"][split], split + " posthoc task lock SHA mismatch", IntegrityError)
    test_gold: Dict[str, Tuple[str, str, bool]] = {}
    for line in (run_dir / "locks" / "posthoc_TEST_with_gold.tsv").read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        require(len(fields) == 4 and fields[3] in ("true", "false"), "invalid posthoc TEST gold manifest row", IntegrityError)
        task_id, passage_id, source_split, label = fields
        require(task_id not in test_gold, "duplicate posthoc TEST gold task", IntegrityError)
        test_gold[task_id] = (passage_id, source_split, label == "true")
    require(len(test_gold) == protocol["allocation"]["test"], "posthoc TEST gold manifest count mismatch", IntegrityError)
    tasks = list(read_jsonl(run_dir / "task_manifest.jsonl"))
    expected_tasks = protocol["allocation"]["fit"] + protocol["allocation"]["cal"] + protocol["allocation"]["test"]
    require(len(tasks) == expected_tasks, "task manifest row count mismatch", IntegrityError)
    require(len({row["task_id"] for row in tasks}) == len(tasks), "duplicate task manifest ID", IntegrityError)
    require(len({row["passage_id"] for row in tasks}) == len(tasks), "duplicate selected passage", IntegrityError)
    task_by_id = {row["task_id"]: row for row in tasks}
    require(all("gold" not in row and "answer" not in row for row in tasks), "gold leaked into task manifest", IntegrityError)
    split_counts = {split: sum(row["internal_split"] == split for row in tasks) for split in ("FIT", "CAL", "TEST")}
    require(split_counts == {"FIT": protocol["allocation"]["fit"], "CAL": protocol["allocation"]["cal"], "TEST": protocol["allocation"]["test"]}, "task split count mismatch", IntegrityError)
    require(set(test_gold) == {row["task_id"] for row in tasks if row["internal_split"] == "TEST"}, "posthoc gold/task-manifest TEST set mismatch", IntegrityError)
    split_passages = {
        split: {row["passage_id"] for row in tasks if row["internal_split"] == split}
        for split in ("FIT", "CAL", "TEST")
    }
    require(all(len(values) == split_counts[split] for split, values in split_passages.items()), "within-split passage duplication", IntegrityError)
    require(not (split_passages["FIT"] & split_passages["CAL"] or split_passages["FIT"] & split_passages["TEST"] or split_passages["CAL"] & split_passages["TEST"]), "cross-split passage overlap", IntegrityError)
    require(sum(bool(row["shadow"]) for row in tasks) == protocol["allocation"]["shadow_test"], "shadow manifest count mismatch", IntegrityError)
    manifest_test_ids = [row["task_id"] for row in tasks if row["internal_split"] == "TEST"]
    expected_shadow = set(sorted(
        manifest_test_ids,
        key=lambda task_id: sha256_bytes((protocol["allocation"]["seed"] + ":shadow:" + task_id).encode("utf-8")),
    )[: protocol["allocation"]["shadow_test"]])
    require({row["task_id"] for row in tasks if row["shadow"]} == expected_shadow, "shadow set differs from frozen hash allocation", IntegrityError)
    shadow_order = sorted(expected_shadow, key=lambda task_id: sha256_bytes((protocol["allocation"]["seed"] + ":shadow:" + task_id).encode("utf-8")))
    require(sha256_bytes("\n".join(shadow_order).encode("utf-8")) == protocol["allocation"]["shadow_manifest_sha256"], "shadow manifest SHA mismatch", IntegrityError)
    traces: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    request_ids = set()
    sampling_seeds = set()
    trace_count = 0
    for row in read_jsonl(run_dir / "rollout_trace.jsonl"):
        require("gold" not in row and "answer" not in row, "gold leaked into rollout trace", IntegrityError)
        require(row["task_id"] in task_by_id, "trace references unknown task", IntegrityError)
        require(row["passage_id"] == task_by_id[row["task_id"]]["passage_id"], "trace passage_id drift", IntegrityError)
        require(row["internal_split"] == task_by_id[row["task_id"]]["internal_split"], "trace split drift", IntegrityError)
        require(row["request_id"] not in request_ids, "duplicate request_id", IntegrityError)
        request_ids.add(row["request_id"])
        sampling_seeds.add(row["sampling_seed"])
        require(row["request_id"] == "%s:%02d" % (row["episode_id"], row["rollout_index"]), "request_id/index drift", IntegrityError)
        require(
            row["sampling_seed"] == deterministic_seed(protocol["allocation"]["seed"], row["task_id"], row["episode_id"], row["rollout_index"]),
            "sampling seed drift",
            IntegrityError,
        )
        require(row["dispatch_monotonic_ns"] <= row["first_token_monotonic_ns"] <= row["final_token_monotonic_ns"] <= row["response_arrival_monotonic_ns"], "invalid request monotonic timestamp order", IntegrityError)
        require(all(isinstance(row[field], str) and row[field].endswith("Z") for field in ("dispatch_utc", "first_token_utc", "final_token_utc", "response_arrival_utc")), "missing UTC request timestamp", IntegrityError)
        parsed = parse_vote(row["raw_output"])
        require(row["raw_output_sha256"] == sha256_bytes(row["raw_output"].encode("utf-8")), "raw output hash mismatch", IntegrityError)
        require((row["parsed_yes"], row["first_canonical_word"], row["strict_compliant"]) == (parsed.yes, parsed.first_canonical_word, parsed.strict_compliant), "recorded vote parse mismatch", IntegrityError)
        require(row["status"] == "completed" and row["retry_count"] == 0, "trace retry/status drift", IntegrityError)
        traces[row["episode_id"]].append(row)
        trace_count += 1
    stops: Dict[str, Mapping[str, Any]] = {}
    for row in read_jsonl(run_dir / "stop_decision.jsonl"):
        require("gold" not in row and "answer" not in row, "gold leaked into stop decision", IntegrityError)
        decision_payload = dict(row)
        recorded_decision_sha = decision_payload.pop("decision_payload_sha256", None)
        require(recorded_decision_sha == sha256_json(decision_payload), "stop decision seal mismatch", IntegrityError)
        require(row["episode_id"] not in stops, "duplicate stop decision", IntegrityError)
        stops[row["episode_id"]] = row
    expected_episodes = protocol["allocation"]["fit"] + protocol["allocation"]["cal"] + 2 * protocol["allocation"]["test"]
    require(len(stops) == expected_episodes, "episode count mismatch", IntegrityError)
    require(set(traces) == set(stops), "trace/decision episode set mismatch", IntegrityError)
    primary_seal = json.loads((run_dir / "locks" / "all_test_primary_decisions_sealed.json").read_text(encoding="utf-8"))
    primary_seal_payload = dict(primary_seal)
    primary_seal_hash = primary_seal_payload.pop("seal_payload_sha256", None)
    require(primary_seal_hash == sha256_json(primary_seal_payload), "global TEST primary seal hash mismatch", IntegrityError)
    require(primary_seal["kind"] == "all_test_primary_decisions_sealed", "global TEST primary seal kind mismatch", IntegrityError)
    test_decision_pairs = sorted(
        (episode_id, stop["decision_payload_sha256"])
        for episode_id, stop in stops.items()
        if stop["internal_split"] == "TEST"
    )
    require(primary_seal["decision_count"] == 2 * protocol["allocation"]["test"] == len(test_decision_pairs), "global TEST primary decision count mismatch", IntegrityError)
    require(primary_seal["decision_set_sha256"] == sha256_json(test_decision_pairs), "global TEST primary decision set mismatch", IntegrityError)
    require(isinstance(primary_seal.get("seal_utc"), str) and primary_seal["seal_utc"].endswith("Z"), "missing global TEST primary seal UTC", IntegrityError)
    minimum_k = int(protocol["policy"]["minimum_stop_k"])
    alpha = float(protocol["policy"]["primary_alpha"])
    N = int(protocol["policy"]["N"])
    for episode_id, stop in stops.items():
        rows = sorted(traces[episode_id], key=lambda row: row["rollout_index"])
        primary = [row for row in rows if not row["excluded_from_primary_cost"]]
        shadow = [row for row in rows if row["excluded_from_primary_cost"]]
        require(stop["task_id"] in task_by_id, "stop references unknown task", IntegrityError)
        task_record = task_by_id[stop["task_id"]]
        require(all(row["episode_id"] == episode_id and row["arm"] == stop["arm"] and row["task_id"] == stop["task_id"] for row in rows), "episode trace identity drift", IntegrityError)
        require(len(primary) == stop["stop_k"], "stop_k/primary row mismatch", IntegrityError)
        require([row["rollout_index"] for row in primary] == list(range(1, stop["stop_k"] + 1)), "non-contiguous primary order", IntegrityError)
        require(all(row["segment"] == "primary" for row in primary), "primary segment mislabeled", IntegrityError)
        require(all(row["response_arrival_monotonic_ns"] <= stop["decision_monotonic_ns"] for row in primary), "future row used by stop", IntegrityError)
        require(isinstance(stop.get("decision_utc"), str) and stop["decision_utc"].endswith("Z"), "missing UTC decision timestamp", IntegrityError)
        if stop["internal_split"] == "TEST":
            require(stop["decision_monotonic_ns"] <= primary_seal["seal_monotonic_ns"], "TEST decision occurs after global primary seal", IntegrityError)
        for previous, current in zip(rows, rows[1:]):
            require(previous["response_arrival_monotonic_ns"] <= current["dispatch_monotonic_ns"], "more than one in-flight request for task", IntegrityError)
        primary_votes = [row["parsed_yes"] for row in primary]
        require(stop["prefix_votes"] == primary_votes, "stop prefix_votes differ from trace", IntegrityError)
        require(stop["yes_count"] == sum(primary_votes), "stop yes_count differs from trace", IntegrityError)
        require(stop["delivered_yes"] == majority_yes(primary_votes), "delivered vote is not prefix majority", IntegrityError)
        require(all(row["dispatch_monotonic_ns"] > stop["decision_monotonic_ns"] and row["dispatch_monotonic_ns"] > primary_seal["seal_monotonic_ns"] and row["segment"] == "shadow_continuation" for row in shadow), "shadow chronology/cost exclusion violation", IntegrityError)
        if stop["arm"] == "BAYES-H-online":
            require(stop["certificate_table_sha256"] == table_sha256, "TEST table hash drift", IntegrityError)
            first_hit: Optional[int] = None
            running_yes = 0
            for k, vote in enumerate(primary_votes, 1):
                running_yes += bool(vote)
                cert_at_k = table[k][running_yes]
                if k >= minimum_k and cert_at_k is not None and cert_at_k <= alpha:
                    first_hit = k
                    break
            expected_stop = first_hit if first_hit is not None else N
            require(stop["stop_k"] == expected_stop, "BAYES-H did not stop at frozen first hit/FULL fallback", IntegrityError)
            expected_certificate = table[stop["stop_k"]][sum(primary_votes)]
            require(stop["certificate_value"] == expected_certificate, "BAYES-H certificate value drift", IntegrityError)
            require(stop["posterior_supported"] == (expected_certificate is not None), "BAYES-H posterior support flag drift", IntegrityError)
            require(stop["forced_full_fallback"] == (stop["stop_k"] == N), "BAYES-H FULL fallback flag drift", IntegrityError)
            if task_record["shadow"]:
                require([row["rollout_index"] for row in rows] == list(range(1, N + 1)), "selected shadow trajectory did not reach FULL-N", IntegrityError)
                require(len(primary) + len(shadow) == N, "selected shadow row count mismatch", IntegrityError)
            else:
                require(not shadow, "nonshadow task has excluded continuation rows", IntegrityError)
        else:
            require(stop["policy"] == "FULL-N" and stop["stop_k"] == N, "FULL arm did not reach N", IntegrityError)
            require(not shadow, "FULL arm has excluded continuation rows", IntegrityError)
    cancellations = list(read_jsonl(run_dir / "cancellation_ledger.jsonl"))
    require(len({row["episode_id"] for row in cancellations}) == len(cancellations), "duplicate cancellation episode", IntegrityError)
    expected_online_episodes = {episode_id for episode_id, stop in stops.items() if stop["arm"] == "BAYES-H-online"}
    require({row["episode_id"] for row in cancellations} == expected_online_episodes, "cancellation/online-decision episode set mismatch", IntegrityError)
    for row in cancellations:
        stop = stops[row["episode_id"]]
        require(row["task_id"] == stop["task_id"] and row["decision_monotonic_ns"] == stop["decision_monotonic_ns"] and row["decision_utc"] == stop["decision_utc"], "cancellation row does not bind exact stop decision", IntegrityError)
        require(row["in_flight_request_ids"] == [], "nonzero online in-flight set", IntegrityError)
        require(row["cancellation_attempted"] is False and row["post_stop_completed_output_tokens"] == 0, "sequential cancellation accounting drift", IntegrityError)
        require(row["cancellation_status"] == "not_applicable_sequential_no_prefetch", "cancellation mislabeled", IntegrityError)
    require(len(cancellations) == protocol["allocation"]["test"], "cancellation row count mismatch", IntegrityError)
    episode_results = list(read_jsonl(run_dir / "episode_result.jsonl"))
    require(len(episode_results) == protocol["allocation"]["test"], "episode result pair count mismatch", IntegrityError)
    require(len({row["task_id"] for row in episode_results}) == len(episode_results), "duplicate TEST episode result task", IntegrityError)
    for result in episode_results:
        require(result.get("evaluated_after_both_arms") is True, "gold evaluated before both arms sealed", IntegrityError)
        task_id = result["task_id"]
        full_id = "TEST:FULL-N:%s" % task_id
        online_id = "TEST:BAYES-H-online:%s" % task_id
        require(task_id in test_gold, "episode result task absent from posthoc TEST gold manifest", IntegrityError)
        passage_id, gold_source_split, gold_value = test_gold[task_id]
        require(task_by_id[task_id]["passage_id"] == passage_id and task_by_id[task_id]["source_split"] == gold_source_split, "posthoc TEST identity/source drift", IntegrityError)
        require(result["full_decision_sha256"] == stops[full_id]["decision_payload_sha256"], "FULL decision/result seal mismatch", IntegrityError)
        require(result["online_decision_sha256"] == stops[online_id]["decision_payload_sha256"], "online decision/result seal mismatch", IntegrityError)
        full_stop = stops[full_id]
        online_stop = stops[online_id]
        full_primary = sorted((row for row in traces[full_id] if not row["excluded_from_primary_cost"]), key=lambda row: row["rollout_index"])
        online_primary = sorted((row for row in traces[online_id] if not row["excluded_from_primary_cost"]), key=lambda row: row["rollout_index"])
        expected_full_correct = full_stop["delivered_yes"] == gold_value
        expected_online_correct = online_stop["delivered_yes"] == gold_value
        require(result["gold"] is gold_value, "episode result gold mismatch", IntegrityError)
        require(result["source_split"] == gold_source_split, "episode result source split mismatch", IntegrityError)
        require(result["full_correct"] is expected_full_correct and result["online_correct"] is expected_online_correct, "episode result correctness flag mismatch", IntegrityError)
        require(result["correctness_difference"] == int(expected_online_correct) - int(expected_full_correct), "episode result correctness difference mismatch", IntegrityError)
        require(result["full_requests"] == full_stop["stop_k"] == len(full_primary), "FULL result request count mismatch", IntegrityError)
        require(result["online_requests"] == online_stop["stop_k"] == len(online_primary), "online result request count mismatch", IntegrityError)
        require(result["full_yes_count"] == sum(row["parsed_yes"] for row in full_primary), "FULL result Yes count mismatch", IntegrityError)
        require(result["online_yes_count"] == sum(row["parsed_yes"] for row in online_primary), "online result Yes count mismatch", IntegrityError)
        require(result["full_vote_tie"] == (2 * result["full_yes_count"] == result["full_requests"]), "FULL result tie mismatch", IntegrityError)
        require(result["online_vote_tie"] == (2 * result["online_yes_count"] == result["online_requests"]), "online result tie mismatch", IntegrityError)
        require(result["full_completed_output_tokens"] == sum(row["local_completed_output_tokens"] for row in full_primary), "FULL output-token result mismatch", IntegrityError)
        require(result["online_completed_output_tokens"] == sum(row["local_completed_output_tokens"] for row in online_primary), "online output-token result mismatch", IntegrityError)
        require(result["completed_output_tokens_saved"] == result["full_completed_output_tokens"] - result["online_completed_output_tokens"], "saved output-token arithmetic mismatch", IntegrityError)
        require(result["full_completed_input_tokens"] == sum(row["local_input_tokens"] for row in full_primary), "FULL input-token result mismatch", IntegrityError)
        require(result["online_completed_input_tokens"] == sum(row["local_input_tokens"] for row in online_primary), "online input-token result mismatch", IntegrityError)
    full_first = 0
    online_first = 0
    test_seed_overlap = 0
    actual_full_first = set()
    for result in episode_results:
        task_id = result["task_id"]
        full_rows = traces["TEST:FULL-N:%s" % task_id]
        online_rows = traces["TEST:BAYES-H-online:%s" % task_id]
        if min(row["dispatch_monotonic_ns"] for row in full_rows) < min(row["dispatch_monotonic_ns"] for row in online_rows):
            full_first += 1
            actual_full_first.add(task_id)
        else:
            online_first += 1
        full_seeds = {row["sampling_seed"] for row in full_rows}
        online_seeds = {row["sampling_seed"] for row in online_rows}
        require(len(full_seeds) == len(full_rows) and len(online_seeds) == len(online_rows), "within-episode sampling seed collision", IntegrityError)
        test_seed_overlap += len(full_seeds & online_seeds)
    require(full_first == online_first == protocol["allocation"]["test"] // 2, "TEST arm order is not exactly balanced", IntegrityError)
    test_ids = [row["task_id"] for row in tasks if row["internal_split"] == "TEST"]
    ranked_ids = sorted(test_ids, key=lambda task_id: sha256_bytes((protocol["allocation"]["seed"] + ":test-arm-order:" + task_id).encode("utf-8")))
    require(actual_full_first == set(ranked_ids[: len(ranked_ids) // 2]), "TEST arm order differs from frozen hash allocation", IntegrityError)
    require(test_seed_overlap == 0, "FULL/online seed namespaces overlap", IntegrityError)
    return {
        "schema": SCHEMA,
        "passed": True,
        "trace_rows": trace_count,
        "unique_request_ids": len(request_ids),
        "unique_sampling_seeds": len(sampling_seeds),
        "global_sampling_seed_collision_rows": trace_count - len(sampling_seeds),
        "episodes": len(stops),
        "task_split_counts": split_counts,
        "selected_unique_passages": len({row["passage_id"] for row in tasks}),
        "cross_split_passage_overlap": 0,
        "shadow_manifest_tasks": protocol["allocation"]["shadow_test"],
        "online_cancellation_rows": len(cancellations),
        "full_first_tasks": full_first,
        "online_first_tasks": online_first,
        "test_arm_seed_overlap": test_seed_overlap,
        "causality": "passed",
        "strict_task_seriality": "passed",
        "shadow_after_decision": "passed",
        "shadow_after_global_test_primary_seal": "passed",
        "frozen_policy_first_hit_replay": "passed",
        "full_arms_reach_N": "passed",
        "gold_absent_from_policy_ledgers": "passed",
        "decision_hash_seals": "passed",
        "gold_join_after_both_decision_seals": "passed",
    }


def health_check(server_url: str) -> None:
    for suffix in ("/health", "/v1/models"):
        try:
            with urllib.request.urlopen(server_url.rstrip("/") + suffix, timeout=15) as response:
                require(response.status == 200, "server health status %s" % response.status)
        except urllib.error.URLError as error:
            raise FormalError("server health failed %s: %s" % (server_url, error))


def package_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {"python": sys.version, "platform": sys.platform}
    for package in ("vllm", "torch", "transformers", "huggingface-hub"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    try:
        versions["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip().splitlines()
    except Exception as error:
        versions["nvidia_smi"] = "unavailable:%s" % error
    versions["image"] = os.environ.get("QS_IMAGE", "unknown")
    return versions


def mark_stage(run_dir: Path, name: str) -> None:
    exclusive_text(run_dir / "stages" / name, utc_now() + "\n")


def run(args: argparse.Namespace) -> None:
    protocol_path = Path(args.protocol).resolve()
    dataset_path = Path(args.dataset).resolve()
    model_path = Path(args.model).resolve()
    run_dir = Path(args.run_dir).resolve()
    require(not run_dir.exists(), "RUN_DIR already exists: %s" % run_dir)
    protocol_raw = protocol_path.read_bytes()
    require(sha256_bytes(protocol_raw) == args.expected_protocol_sha256, "protocol SHA-256 mismatch")
    protocol = json.loads(protocol_raw)
    require(protocol["schema"] == SCHEMA, "protocol schema mismatch")
    require(str(dataset_path) == protocol["carrier"]["canonical_jsonl"], "dataset path drift")
    require(str(model_path) == protocol["model"]["staged_path"], "model path drift")
    require(str(run_dir) == protocol["execution"]["run_dir"], "RUN_DIR path drift")
    require(len(args.server) == protocol["inference"]["servers"], "server count drift")
    run_dir.mkdir(parents=True)
    (run_dir / "stages").mkdir()
    (run_dir / "locks").mkdir()
    mark_stage(run_dir, "00-started")
    ledger = LedgerSet(run_dir)
    try:
        model_snapshot = verify_model_snapshot(model_path, protocol["model"])
        fit_tasks, cal_tasks, test_tasks = load_and_allocate(dataset_path, protocol)
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, revision=protocol["model"]["revision"])
        prompt_spec = protocol["inference"]["prompt"]
        all_tasks = fit_tasks + cal_tasks + test_tasks
        rendered_by_id: Dict[str, PromptTask] = {}
        for task in all_tasks:
            rendered_by_id[task.task_id] = render_prompt(task, tokenizer, prompt_spec)
            require(
                rendered_by_id[task.task_id].input_tokens + int(protocol["inference"]["max_tokens"]) <= int(protocol["inference"]["max_model_len"]),
                "rendered task exceeds frozen model context: " + task.task_id,
                DataError,
            )
            ledger.append("task_manifest.jsonl", {
                "schema": SCHEMA,
                "task_id": task.task_id,
                "passage_id": task.passage_id,
                "internal_split": task.internal_split,
                "source_split": task.source_split,
                "source_index": task.source_index,
                "prompt_sha256": rendered_by_id[task.task_id].prompt_sha256,
                "rendered_input_tokens": rendered_by_id[task.task_id].input_tokens,
                "eligible": True,
                "shadow": task.shadow,
                "gold_evaluator": protocol["carrier"]["gold_evaluator"],
            })
        prompt_lengths = input_length_summary(
            [rendered_by_id[task.task_id].input_tokens for task in all_tasks],
            int(protocol["inference"]["max_tokens"]),
            int(protocol["inference"]["max_model_len"]),
        )
        N = int(protocol["policy"]["N"])
        base_seed = protocol["allocation"]["seed"]
        test_arm_seed_overlap = 0
        for task in test_tasks:
            full_episode = "TEST:FULL-N:%s" % task.task_id
            online_episode = "TEST:BAYES-H-online:%s" % task.task_id
            full_seeds = {deterministic_seed(base_seed, task.task_id, full_episode, index) for index in range(1, N + 1)}
            online_seeds = {deterministic_seed(base_seed, task.task_id, online_episode, index) for index in range(1, N + 1)}
            require(len(full_seeds) == N and len(online_seeds) == N, "planned within-episode seed collision", IntegrityError)
            test_arm_seed_overlap += len(full_seeds & online_seeds)
        require(test_arm_seed_overlap == 0, "planned per-task FULL/BAYES seed namespace collision", IntegrityError)
        for server in args.server:
            health_check(server)
        clients = [OpenAICompletionClient(server, rank, args.served_model, protocol["inference"], tokenizer) for rank, server in enumerate(args.server)]
        atomic_json(run_dir / "preflight.json", {
            "schema": SCHEMA,
            "passed": True,
            "protocol_sha256": args.expected_protocol_sha256,
            "dataset_sha256": sha256_file(dataset_path),
            "dataset_tasks": len(all_tasks),
            "selected_manifest_sha256_computed_in_memory": sha256_bytes(selected_manifest_text(fit_tasks + cal_tasks + test_tasks).encode("utf-8")),
            "split_manifest_sha256_computed_in_memory": {
                "FIT": sha256_bytes(selected_manifest_text(fit_tasks).encode("utf-8")),
                "CAL": sha256_bytes(selected_manifest_text(cal_tasks).encode("utf-8")),
                "TEST": sha256_bytes(selected_manifest_text(test_tasks).encode("utf-8")),
            },
            "shadow_manifest_sha256_computed_in_memory": sha256_bytes(shadow_manifest_text(test_tasks, protocol["allocation"]["seed"]).encode("utf-8")),
            "selected_unique_passages": len({task.passage_id for task in all_tasks}),
            "cross_split_passage_overlap": 0,
            "planned_test_full_online_same_task_seed_overlap": test_arm_seed_overlap,
            "gold_manifest_materialized_before_test": False,
            "model_revision": protocol["model"]["revision"],
            "model_snapshot": model_snapshot,
            "rendered_input_token_distribution": prompt_lengths,
            "image_contract": {
                "literal_tag": protocol["inference"]["image"],
                "qs_image_id": protocol["inference"]["image_id"],
                "qs_api_created_updated_date": protocol["inference"]["image_api_created_updated_date"],
                "registry_digest": protocol["inference"]["image_digest"],
                "tag_caveat": protocol["inference"]["image_tag_caveat"],
            },
            "versions": package_versions(),
            "servers": args.server,
            "schema_fixture": {name: "created" for name in LEDGERS},
        })
        ledger.flush()
        mark_stage(run_dir, "01-preflight-passed")

        fit_prompts = [rendered_by_id[task.task_id] for task in fit_tasks]
        fit_outcomes = execute_full_stage(fit_prompts, clients, ledger, protocol, "FIT", "FULL-N-training")
        fit = fit_lock(fit_outcomes, protocol)
        atomic_json(run_dir / "locks" / "fit_lock.json", fit)
        ledger.flush()
        mark_stage(run_dir, "02-fit-locked")

        cal_prompts = [rendered_by_id[task.task_id] for task in cal_tasks]
        cal_outcomes = execute_full_stage(cal_prompts, clients, ledger, protocol, "CAL", "FULL-N-calibration")
        cal = calibration_lock(cal_outcomes, fit, protocol)
        atomic_json(run_dir / "locks" / "cal_lock.json", cal)
        ledger.flush()
        mark_stage(run_dir, "03-cal-locked")
        if not cal["primary_accepted"]:
            mark_stage(run_dir, "CAL_REJECTED_NO_TEST")
            raise CalibrationRejected("alpha=.05 BAYES-H failed the frozen CAL empirical-Bernstein gate")

        test_prompts = [rendered_by_id[task.task_id] for task in test_tasks]
        pairs = execute_test(test_prompts, clients, ledger, protocol, fit["certificate_table"], fit["certificate_table_sha256"], run_dir)
        # All paired TEST decisions are now append-only and hash-sealed.  Only
        # at this point may the gold-bound carrier manifests be materialized.
        exclusive_text(run_dir / "locks" / "posthoc_selected_with_gold.tsv", selected_manifest_text(fit_tasks + cal_tasks + test_tasks))
        exclusive_text(run_dir / "locks" / "posthoc_FIT_with_gold.tsv", selected_manifest_text(fit_tasks))
        exclusive_text(run_dir / "locks" / "posthoc_CAL_with_gold.tsv", selected_manifest_text(cal_tasks))
        exclusive_text(run_dir / "locks" / "posthoc_TEST_with_gold.tsv", selected_manifest_text(test_tasks))
        gold = {task.task_id: task.gold for task in test_tasks}
        source_split = {task.task_id: task.source_split for task in test_tasks}
        analysis, episode_rows = paired_analysis(pairs, gold, source_split, protocol)
        # Gold is joined only here, after both arms have returned frozen outcomes.
        for row in episode_rows:
            ledger.append("episode_result.jsonl", dict(row, evaluated_after_both_arms=True))
        ledger.flush()
        mark_stage(run_dir, "04-test-completed")
        audit = audit_ledgers(run_dir, protocol, fit["certificate_table"])
        atomic_json(run_dir / "online_integrity_audit.json", audit)
        atomic_json(run_dir / "analysis.json", analysis)
        mark_stage(run_dir, "05-audit-analysis-completed")
        outputs = {}
        for path in sorted(run_dir.iterdir()):
            if path.is_file() and path.name != "run_manifest.json":
                outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        atomic_json(run_dir / "run_manifest.json", {
            "schema": SCHEMA,
            "status": "completed_valid",
            "completed_at_utc": utc_now(),
            "command": sys.argv,
            "protocol_sha256": args.expected_protocol_sha256,
            "source_sha256": sha256_file(Path(__file__)),
            "dataset_sha256": sha256_file(dataset_path),
            "model_revision": protocol["model"]["revision"],
            "model_snapshot_manifest_sha256": model_snapshot["ledger_sha256"],
            "image_contract": {
                "literal_tag": protocol["inference"]["image"],
                "qs_image_id": protocol["inference"]["image_id"],
                "registry_digest": protocol["inference"]["image_digest"],
            },
            "fit_lock_sha256": sha256_file(run_dir / "locks" / "fit_lock.json"),
            "cal_lock_sha256": sha256_file(run_dir / "locks" / "cal_lock.json"),
            "test_primary_seal_sha256": sha256_file(run_dir / "locks" / "all_test_primary_decisions_sealed.json"),
            "outputs": outputs,
            "versions": package_versions(),
        })
        mark_stage(run_dir, "COMPLETED")
    except Exception as error:
        failure = {
            "schema": SCHEMA,
            "status": "failed_before_valid_completion",
            "failed_at_utc": utc_now(),
            "failure_class": getattr(error, "failure_class", "infrastructure_preflight"),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        failure_path = run_dir / "failure.json"
        if not failure_path.exists():
            atomic_json(failure_path, failure)
        marker = run_dir / "stages" / "FAILED"
        if not marker.exists():
            exclusive_text(marker, utc_now() + "\n")
        raise
    finally:
        ledger.close()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--expected-protocol-sha256", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model", default="a11-qwen25-7b")
    parser.add_argument("--server", action="append", required=True)
    parser.add_argument("--run-dir", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
