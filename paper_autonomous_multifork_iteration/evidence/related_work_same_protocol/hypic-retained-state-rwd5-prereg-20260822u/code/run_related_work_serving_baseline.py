#!/usr/bin/env python3
"""Same-model serving baseline client and blind aggregate.

The client speaks the OpenAI completions streaming protocol so vLLM and SGLang
are timed at the same boundary.  It never compares these HTTP timings directly
with the paper's in-process CoMem adapter timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import string
import time
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


EXPECTED_PAIRS = tuple(
    (dataset, source_index)
    for dataset in ("qasper", "2wikimqa")
    for source_index in range(6, 10)
)
DATA_SHA256 = "1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe"
SOURCE_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
DATASET_PROMPTS = {
    "qasper": (
        "You are given a scientific article and a question. Answer the question as "
        "concisely as you can, using a single phrase or sentence if possible. If the "
        "question cannot be answered based on the information in the article, write "
        '"unanswerable". If the question is a yes/no question, answer "yes", "no", or '
        '"unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n'
        "Answer the question based on the above article as concisely as you can, using "
        "a single phrase or sentence if possible. If the question cannot be answered "
        'based on the information in the article, write "unanswerable". If the question '
        'is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide '
        "any explanation.\n\nQuestion: {input}\n\nAnswer:"
    ),
    "2wikimqa": (
        "Answer the question based on the given passages. Only give me the answer and "
        "do not output any other words.\n\nThe following are given passages.\n"
        "{context}\n\nAnswer the question based on the given passages. Only give me the "
        "answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
    ),
}


class BaselineError(RuntimeError):
    pass


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BaselineError(message)


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    _require(not isinstance(value, bool) and isinstance(value, (int, float)), label)
    result = float(value)
    _require(math.isfinite(result), label)
    if positive:
        _require(result > 0.0, label)
    return result


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def parse_prometheus(text: str) -> dict[str, float]:
    """Collapse unlabeled and labeled numeric samples by metric name."""
    totals: dict[str, float] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(None, 1)
        if len(fields) != 2:
            continue
        name = fields[0].split("{", 1)[0]
        try:
            value = float(fields[1])
        except ValueError:
            continue
        if math.isfinite(value):
            totals[name] = totals.get(name, 0.0) + value
    return totals


def prefix_counter_snapshot(metrics: dict[str, float]) -> dict[str, float]:
    aliases = {
        "queries": (
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_queries_total",
            "sglang:prefix_cache_queries",
            "sglang:prefix_cache_queries_total",
            "sglang:prompt_tokens_total",
        ),
        "hits": (
            "vllm:prefix_cache_hits",
            "vllm:prefix_cache_hits_total",
            "sglang:prefix_cache_hits",
            "sglang:prefix_cache_hits_total",
            "sglang:cached_tokens_total",
        ),
    }
    result: dict[str, float] = {}
    for logical, names in aliases.items():
        present = [metrics[name] for name in names if name in metrics]
        if present:
            result[logical] = max(present)
    for name, value in metrics.items():
        lowered = name.lower()
        if "prefix" not in lowered or "cache" not in lowered:
            continue
        if "quer" in lowered and "queries" not in result:
            result["queries"] = value
        if "hit" in lowered and "rate" not in lowered and "hits" not in result:
            result["hits"] = value
    return result


def counter_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float | None]:
    result: dict[str, float | None] = {"queries": None, "hits": None}
    for key in result:
        if key in before and key in after:
            delta = after[key] - before[key]
            _require(delta >= 0.0, f"prefix counter {key} decreased")
            result[key] = delta
    queries = result["queries"]
    hits = result["hits"]
    result["hit_rate"] = (
        float(hits) / float(queries)
        if isinstance(queries, float) and queries > 0.0 and isinstance(hits, float)
        else None
    )
    return result


def http_get_text(url: str, timeout: float = 10.0) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def stream_completion(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: float,
    require_text: bool = True,
    clock: Callable[[], float] = time.perf_counter,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> dict[str, Any]:
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = clock()
    chunks: list[dict[str, Any]] = []
    text_parts: list[str] = []
    first_text_at: float | None = None
    first_choice_at: float | None = None
    usage: dict[str, Any] | None = None
    finish_reason: str | None = None
    event_count = 0
    with opener(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line or line.startswith(":"):
                continue
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            event = json.loads(body)
            event_count += 1
            if "error" in event:
                raise BaselineError(f"stream returned server error: {event['error']!r}")
            observed = clock()
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0]
            if first_choice_at is None:
                first_choice_at = observed
            part = choice.get("text", "")
            if not isinstance(part, str):
                raise BaselineError("stream choice text is not a string")
            if part:
                if first_text_at is None:
                    first_text_at = observed
                text_parts.append(part)
                chunks.append({"offset_seconds": observed - started, "text": part})
            if choice.get("finish_reason") is not None:
                finish_reason = str(choice["finish_reason"])
    finished = clock()
    if require_text:
        _require(first_text_at is not None, "stream returned no generated text")
    else:
        _require(first_choice_at is not None, "stream returned no completion choice")
    _require(isinstance(usage, dict), "stream omitted authoritative usage")
    completion_tokens = usage.get("completion_tokens")
    _require(
        not isinstance(completion_tokens, bool)
        and isinstance(completion_tokens, int)
        and completion_tokens > 0,
        "invalid completion token count",
    )
    first_token_at = first_text_at if first_text_at is not None else first_choice_at
    _require(first_token_at is not None, "stream returned no timed completion token")
    ttft = first_token_at - started
    e2e = finished - started
    tpot = (
        (finished - first_token_at) / (completion_tokens - 1)
        if completion_tokens > 1
        else None
    )
    return {
        "prediction": "".join(text_parts).strip(),
        "chunks": chunks,
        "usage": usage,
        "finish_reason": finish_reason,
        "stream_event_count": event_count,
        "ttft_seconds": ttft,
        "e2e_seconds": e2e,
        "median_tpot_seconds": tpot,
        "generated_tokens_per_second": completion_tokens / e2e,
    }


def load_workload(model: Path, data: Path, rank: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    selected: dict[str, int] = {}
    samples: list[dict[str, Any]] = []
    with data.open() as handle:
        for line in handle:
            sample = json.loads(line)
            source_index = int(sample["_source_index"])
            if source_index < 6 or source_index > 9 or source_index in {4, 5}:
                continue
            dataset = str(sample["dataset"])
            count = selected.get(dataset, 0)
            if count >= 4:
                continue
            selected[dataset] = count + 1
            samples.append(sample)
    _require(bool(samples), "frozen LongBench slice is empty")
    datasets = {str(sample["dataset"]) for sample in samples}
    _require(datasets == {"qasper", "2wikimqa"}, "frozen dataset set drift")
    revisions = {sample.get("_source_revision") for sample in samples}
    _require(revisions == {SOURCE_REVISION}, "source revision drift")

    workloads: list[dict[str, Any]] = []
    marker = "QCOMEM_CONTEXT_MARKER_8D31F4"
    for sample in samples:
        prompt_format = DATASET_PROMPTS[str(sample["dataset"])]
        user_text = prompt_format.format(context=marker, input=sample["input"])
        messages = [{"role": "user", "content": user_text}]
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        _require(rendered.count(marker) == 1, "chat template marker drift")
        prefix, suffix = rendered.split(marker)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        context_ids = tokenizer.encode(sample["context"], add_special_tokens=False)
        original_context_tokens = len(context_ids)
        available = 4096 - len(prefix_ids) - len(suffix_ids)
        _require(available >= 256, "insufficient frozen context budget")
        if len(context_ids) > available:
            left = available // 2
            context_ids = context_ids[:left] + context_ids[-(available - left) :]
        workloads.append(
            {
                "workload_id": f"{sample['dataset']}-{sample['_source_index']}",
                "dataset": str(sample["dataset"]),
                "source_index": int(sample["_source_index"]),
                "document_token_ids": [int(value) for value in prefix_ids + context_ids],
                "query_token_ids": [int(value) for value in suffix_ids],
                "references": [str(answer) for answer in sample["answers"]],
            }
        )
    metadata = {
        "data": str(data),
        "data_sha256": sha256_file(data),
        "source_revisions": sorted(revisions),
        "datasets": sorted(datasets),
        "source_index_start": 6,
        "source_index_end": 9,
        "excluded_source_indices": [4, 5],
        "test_v2_consumed": False,
        "prompt_protocol": "longbench-v1-official",
    }
    observed = tuple(
        (str(row["dataset"]), int(row["source_index"])) for row in workloads
    )
    _require(observed == EXPECTED_PAIRS, "frozen workload pair order drift")
    _require(metadata["data_sha256"] == DATA_SHA256, "data digest drift")
    _require(metadata["source_revisions"] == [SOURCE_REVISION], "source revision drift")
    workload = workloads[rank]
    return {
        "workload_id": workload["workload_id"],
        "dataset": workload["dataset"],
        "source_index": int(workload["source_index"]),
        "document_token_ids": workload["document_token_ids"],
        "query_token_ids": workload["query_token_ids"],
        "references": list(workload["references"]),
        "metadata": metadata,
    }


def client_stage(args: argparse.Namespace) -> None:
    _require(0 <= args.rank < args.world_size == 8, "formal run requires ranks 0..7")
    _require(args.phase in {"cache_off", "cache_on"}, "invalid phase")
    _require(sha256_file(args.data) == args.expected_data_sha256 == DATA_SHA256, "data hash")
    workload = load_workload(args.model, args.data, args.rank)
    base_url = args.base_url.rstrip("/")
    # SGLang 0.5.17 can keep /health at 503 even after its OpenAI endpoint is
    # serving successful requests.  /model_info is its authoritative startup
    # probe; vLLM retains the ordinary /health probe.
    readiness_endpoint = (
        "/model_info" if args.system.startswith("sglang-") else "/health"
    )
    health = http_get_text(base_url + readiness_endpoint, timeout=10.0)

    common = {
        "model": args.served_model_name,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": True,
        # SGLang 0.5.17 can omit its terminal usage-only event for a cached
        # request. Continuous usage keeps the authoritative server token count
        # available without estimating it from decoded text.
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
    }
    warmup = stream_completion(
        base_url + "/v1/completions",
        {**common, "prompt": workload["query_token_ids"][:8], "max_tokens": 1},
        timeout=args.timeout,
    )
    metrics_before = prefix_counter_snapshot(
        parse_prometheus(http_get_text(base_url + "/metrics", timeout=10.0))
    )
    prime: dict[str, Any] | None = None
    if args.phase == "cache_on":
        prime = stream_completion(
            base_url + "/v1/completions",
            {**common, "prompt": workload["document_token_ids"], "max_tokens": 1},
            timeout=args.timeout,
            # A one-token cache prime may legitimately decode to an empty
            # string. Its server usage receipt, not visible text, proves that
            # the request executed.
            require_text=False,
        )

    metrics_after_prime = prefix_counter_snapshot(
        parse_prometheus(http_get_text(base_url + "/metrics", timeout=10.0))
    )
    measured = stream_completion(
        base_url + "/v1/completions",
        {
            **common,
            "prompt": workload["document_token_ids"] + workload["query_token_ids"],
            "max_tokens": args.max_new_tokens,
        },
        timeout=args.timeout,
    )
    metrics_after = prefix_counter_snapshot(
        parse_prometheus(http_get_text(base_url + "/metrics", timeout=10.0))
    )
    references = workload["references"]
    measured["f1"] = max(
        answer_f1(measured["prediction"], reference) for reference in references
    )
    payload = {
        "schema": "forkaudit-related-serving-shard-v1",
        "status": "completed",
        "system": args.system,
        "phase": args.phase,
        "rank": args.rank,
        "world_size": args.world_size,
        "served_model_name": args.served_model_name,
        "workload": {
            "workload_id": workload["workload_id"],
            "dataset": workload["dataset"],
            "source_index": workload["source_index"],
            "document_tokens": len(workload["document_token_ids"]),
            "query_tokens": len(workload["query_token_ids"]),
            "references": references,
        },
        "protocol": {
            "data_sha256": DATA_SHA256,
            "source_revision": SOURCE_REVISION,
            "max_input_tokens": 4096,
            "max_new_tokens": args.max_new_tokens,
            "greedy": True,
            "timing_boundary": "openai-completions-stream-client-wall-clock",
        },
        "health_response": health,
        "warmup": warmup,
        "prime": prime,
        "measured": measured,
        "prefix_counters": {
            "before": metrics_before,
            "after_prime": metrics_after_prime,
            "after": metrics_after,
            "measured_delta": counter_delta(metrics_after_prime, metrics_after),
            "prime_plus_measured_delta": counter_delta(metrics_before, metrics_after),
        },
    }
    atomic_json(args.output, payload)


def _validate_shard(value: Any, *, phase: str, rank: int, system: str) -> dict[str, Any]:
    _require(isinstance(value, dict), "shard is not an object")
    _require(value.get("schema") == "forkaudit-related-serving-shard-v1", "schema")
    _require(value.get("status") == "completed", "shard incomplete")
    _require(value.get("system") == system, "system drift")
    _require(value.get("phase") == phase, "phase drift")
    _require(value.get("rank") == rank and value.get("world_size") == 8, "rank drift")
    workload = value.get("workload")
    _require(isinstance(workload, dict), "workload missing")
    _require(
        (workload.get("dataset"), workload.get("source_index")) == EXPECTED_PAIRS[rank],
        "workload assignment drift",
    )
    measured = value.get("measured")
    _require(isinstance(measured, dict), "measured missing")
    _require(isinstance(measured.get("prediction"), str), "prediction missing")
    _number(measured.get("f1"), "f1")
    _number(measured.get("ttft_seconds"), "ttft", positive=True)
    _number(measured.get("e2e_seconds"), "e2e", positive=True)
    _number(measured.get("generated_tokens_per_second"), "throughput", positive=True)
    usage = measured.get("usage")
    _require(isinstance(usage, dict), "usage missing")
    _require(
        not isinstance(usage.get("completion_tokens"), bool)
        and isinstance(usage.get("completion_tokens"), int)
        and usage["completion_tokens"] > 0,
        "completion tokens invalid",
    )
    return value


def aggregate_stage(args: argparse.Namespace) -> None:
    phases: dict[str, list[dict[str, Any]]] = {"cache_off": [], "cache_on": []}
    for phase in phases:
        for rank in range(8):
            path = args.input_dir / f"{phase}-rank-{rank}.json"
            value = json.loads(path.read_text())
            phases[phase].append(
                _validate_shard(value, phase=phase, rank=rank, system=args.system)
            )

    token_exact = []
    cache_hit_observed = []
    phase_summaries: dict[str, Any] = {}
    for rank in range(8):
        off = phases["cache_off"][rank]["measured"]
        on = phases["cache_on"][rank]["measured"]
        token_exact.append(off["prediction"] == on["prediction"])
        counters = phases["cache_on"][rank]["prefix_counters"]["measured_delta"]
        usage_details = on["usage"].get("prompt_tokens_details")
        cached_tokens = (
            usage_details.get("cached_tokens")
            if isinstance(usage_details, dict)
            else None
        )
        counter_hits = counters.get("hits")
        cache_hit_observed.append(
            (isinstance(cached_tokens, int) and cached_tokens > 0)
            or (isinstance(counter_hits, (int, float)) and counter_hits > 0)
        )

    for phase, shards in phases.items():
        rows = [value["measured"] for value in shards]
        tpots = [row["median_tpot_seconds"] for row in rows if row["median_tpot_seconds"] is not None]
        phase_summaries[phase] = {
            "mean_f1": statistics.mean(float(row["f1"]) for row in rows),
            "median_ttft_seconds": statistics.median(float(row["ttft_seconds"]) for row in rows),
            "median_tpot_seconds": statistics.median(float(value) for value in tpots),
            "median_generated_tokens_per_second": statistics.median(
                float(row["generated_tokens_per_second"]) for row in rows
            ),
            "predictions": [row["prediction"] for row in rows],
        }

    all_exact = all(token_exact)
    all_hits = all(cache_hit_observed)
    payload = {
        "schema": "forkaudit-related-serving-summary-v1",
        "scientific_run_valid": True,
        "hypothesis_passed": all_exact and all_hits,
        "scientific_outcome": (
            "valid_positive" if all_exact and all_hits else "valid_negative"
        ),
        "system": args.system,
        "comparison_boundary": "same-model-same-slice-openai-streaming-serving-only",
        "not_comparable_to": "in-process CoMem direct-adapter wall-clock",
        "pairs": [list(value) for value in EXPECTED_PAIRS],
        "cache_off_vs_on_prediction_exact": token_exact,
        "cache_hit_observed": cache_hit_observed,
        "phases": phase_summaries,
        "raw_shards": {
            phase: [f"{phase}-rank-{rank}.json" for rank in range(8)]
            for phase in phases
        },
    }
    atomic_json(args.output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("client", "aggregate"), required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--expected-data-sha256", default=DATA_SHA256)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--phase", choices=("cache_off", "cache_on"))
    parser.add_argument("--base-url")
    parser.add_argument("--served-model-name", default="qwen35-related-baseline")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    if args.stage == "client":
        _require(args.model is not None and args.data is not None, "client inputs missing")
        _require(args.phase is not None and args.base_url is not None, "client endpoint missing")
        client_stage(args)
    else:
        _require(args.input_dir is not None, "aggregate input dir missing")
        aggregate_stage(args)


if __name__ == "__main__":
    main()
