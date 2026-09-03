from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import string
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, TypeVar

import torch

from qcomem_torch import (
    TorchSplitCausalLM,
    greedy_generate_comem,
    greedy_generate_dense,
    quantize_residual,
    residual_error_sums,
)


T = TypeVar("T")
CONFIGS = [
    "dense",
    "d7-q16",
    "d7-q8",
    "d7-q4",
    "d10-q16",
    "d10-q8",
    "d10-q4",
    "d13-q16",
    "d13-q8",
    "d13-q4",
]
# Cost-aware static assignment for the live 8-GPU resource pack.  Dense and
# shallow splits do more online work, so the two extra configurations are paired
# with depth-10 workers instead of blindly landing on ranks 0 and 1.
EIGHT_GPU_ASSIGNMENTS = (
    ("dense",),
    ("d7-q16",),
    ("d7-q8",),
    ("d7-q4",),
    ("d10-q16", "d13-q8"),
    ("d10-q8", "d13-q4"),
    ("d10-q4",),
    ("d13-q16",),
)
DATASET_MAX_NEW_TOKENS = {"qasper": 128, "2wikimqa": 32}
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


def timed_gpu(operation: Callable[[], T]) -> tuple[T, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


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


def max_reference_f1(prediction: str, references: list[str]) -> float:
    return max(answer_f1(prediction, reference) for reference in references)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_samples(
    path: Path,
    limit_per_dataset: int,
    *,
    source_index_start: int | None = None,
    source_index_end: int | None = None,
    exclude_source_indices: tuple[int, ...] = (),
) -> list[dict[str, Any]]:
    """Load a deterministic per-dataset slice of the JSONL benchmark.

    Source indices are filtered before applying ``limit_per_dataset``.  This is
    important for the mixed-bit validation protocol: calibration uses indices
    4--5, while downstream validation is frozen to the disjoint 6--35 range.
    Existing callers keep their old first-N behaviour when no range is given.
    """
    if limit_per_dataset < 1:
        raise ValueError("limit_per_dataset must be positive")
    if (
        source_index_start is not None
        and source_index_end is not None
        and source_index_start > source_index_end
    ):
        raise ValueError("source_index_start must not exceed source_index_end")

    excluded = set(exclude_source_indices)
    selected: dict[str, int] = {}
    rows = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if source_index_start is not None or source_index_end is not None:
                source_index = row.get("_source_index")
                if source_index is None:
                    raise ValueError(
                        "source-index filtering requires _source_index on every row"
                    )
                source_index = int(source_index)
                if source_index_start is not None and source_index < source_index_start:
                    continue
                if source_index_end is not None and source_index > source_index_end:
                    continue
            else:
                source_index = row.get("_source_index")
            if source_index is not None and int(source_index) in excluded:
                continue
            dataset = row["dataset"]
            count = selected.get(dataset, 0)
            if count >= limit_per_dataset:
                continue
            selected[dataset] = count + 1
            rows.append(row)
    return rows


def generation_limit(dataset: str, global_cap: int) -> int:
    return min(global_cap, DATASET_MAX_NEW_TOKENS.get(dataset, global_cap))


def assigned_configs(rank: int, world_size: int) -> list[str]:
    if world_size == 8:
        return list(EIGHT_GPU_ASSIGNMENTS[rank])
    return [
        config for index, config in enumerate(CONFIGS) if index % world_size == rank
    ]


def prompt_parts(tokenizer, sample: dict[str, Any], max_input_tokens: int):
    marker = "QCOMEM_CONTEXT_MARKER_8D31F4"
    prompt_format = DATASET_PROMPTS.get(
        sample["dataset"],
        "Use only the following context to answer the question.\n\n{context}\n\n"
        "Question: {input}\nReturn only a concise answer without explanation.",
    )
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
    if rendered.count(marker) != 1:
        raise ValueError("chat template did not preserve the context marker")
    prefix, suffix = rendered.split(marker)
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    context_ids = tokenizer.encode(sample["context"], add_special_tokens=False)
    original_context_tokens = len(context_ids)
    available = max_input_tokens - len(prefix_ids) - len(suffix_ids)
    if available < 256:
        raise ValueError("max_input_tokens leaves too little room for context")
    if len(context_ids) > available:
        left = available // 2
        context_ids = context_ids[:left] + context_ids[-(available - left) :]
    document_ids = torch.tensor(prefix_ids + context_ids, dtype=torch.long)
    query_ids = torch.tensor(suffix_ids, dtype=torch.long)
    return (
        document_ids,
        query_ids,
        len(prefix_ids),
        len(context_ids),
        original_context_tokens,
    )


def parse_config(name: str) -> tuple[int | None, int | None]:
    if name == "dense":
        return None, None
    match = re.fullmatch(r"d(\d+)-q(\d+)", name)
    if not match:
        raise ValueError(f"invalid config name: {name}")
    return int(match.group(1)), int(match.group(2))


@torch.inference_mode()
def run_config(
    *,
    config_name: str,
    model,
    tokenizer,
    samples: list[dict[str, Any]],
    model_allocated_bytes: int,
    args,
) -> dict[str, Any]:
    adapter = TorchSplitCausalLM(model)
    depth, bits = parse_config(config_name)
    eos_values = tokenizer.eos_token_id
    eos_ids = {int(eos_values)} if isinstance(eos_values, int) else set(eos_values or [])
    rows = []

    for sample_index, sample in enumerate(samples):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        (
            document_ids,
            query_ids,
            prefix_tokens,
            context_tokens,
            original_context_tokens,
        ) = prompt_parts(tokenizer, sample, args.max_input_tokens)
        document_ids = document_ids.cuda()
        query_ids = query_ids.cuda()
        full_ids = torch.cat([document_ids, query_ids])
        max_new_tokens = generation_limit(sample["dataset"], args.max_new_tokens)

        if config_name == "dense":
            generated, generation_seconds = timed_gpu(
                lambda: greedy_generate_dense(
                    adapter,
                    full_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_ids,
                )
            )
            storage: dict[str, Any] = {}
        else:
            assert depth is not None and bits is not None
            parts, write_seconds = timed_gpu(
                lambda: adapter.chunk_local_write_parts(
                    document_ids,
                    depth,
                    chunk_size=args.chunk_size,
                    overlap=args.overlap,
                )
            )
            stores = []
            restored = []
            error_sums = {
                "squared_error_sum": 0.0,
                "reference_squared_sum": 0.0,
                "max_abs_error": 0.0,
                "elements": 0,
            }
            quantize_started = time.perf_counter()
            for part in parts:
                store = quantize_residual(
                    part, bits=bits, group_size=args.group_size
                )
                candidate = store.dequantize(part.dtype)
                metrics = residual_error_sums(part, candidate)
                stores.append(store)
                restored.append(candidate)
                for key in ("squared_error_sum", "reference_squared_sum", "elements"):
                    error_sums[key] += metrics[key]
                error_sums["max_abs_error"] = max(
                    error_sums["max_abs_error"], metrics["max_abs_error"]
                )
            torch.cuda.synchronize()
            quantize_seconds = time.perf_counter() - quantize_started
            dense_nbytes = sum(store.dense_nbytes for store in stores)
            stored_nbytes = sum(store.nbytes for store in stores)
            del parts, stores, part, store, candidate
            generated, generation_seconds = timed_gpu(
                lambda: greedy_generate_comem(
                    adapter,
                    restored,
                    query_ids,
                    depth=depth,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_ids,
                )
            )
            rmse = math.sqrt(
                error_sums["squared_error_sum"] / max(error_sums["elements"], 1)
            )
            denominator = math.sqrt(
                error_sums["reference_squared_sum"]
                / max(error_sums["elements"], 1)
            )
            storage = {
                "write_seconds": write_seconds,
                "quantize_dequantize_seconds": quantize_seconds,
                "dense_residual_nbytes": dense_nbytes,
                "stored_residual_nbytes": stored_nbytes,
                "compression_ratio": dense_nbytes / stored_nbytes,
                "residual_rmse": rmse,
                "residual_relative_rmse": rmse / max(denominator, 1e-12),
                "residual_max_abs_error": error_sums["max_abs_error"],
            }

        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        references = [str(answer) for answer in sample["answers"]]
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        rows.append(
            {
                "sample_index": sample_index,
                "id": sample.get("_id", f"{sample['dataset']}-{sample_index}"),
                "dataset": sample["dataset"],
                "question": sample["input"],
                "references": references,
                "prediction": prediction,
                "f1": max_reference_f1(prediction, references),
                "source_length": sample.get("length"),
                "source_repo": sample.get("_source_repo"),
                "source_revision": sample.get("_source_revision"),
                "source_index": sample.get("_source_index"),
                "prefix_tokens": prefix_tokens,
                "context_tokens": context_tokens,
                "original_context_tokens": original_context_tokens,
                "context_truncated": context_tokens < original_context_tokens,
                "input_tokens": int(full_ids.numel()),
                "max_new_tokens": max_new_tokens,
                "generated_tokens": len(generated),
                "generation_seconds": generation_seconds,
                "model_allocated_bytes": model_allocated_bytes,
                "peak_allocated_bytes": peak_allocated_bytes,
                "incremental_peak_allocated_bytes": max(
                    peak_allocated_bytes - model_allocated_bytes, 0
                ),
                **storage,
            }
        )
        print(
            json.dumps(
                {
                    "config": config_name,
                    "sample": sample_index,
                    "dataset": sample["dataset"],
                    "f1": rows[-1]["f1"],
                    "generated_tokens": len(generated),
                }
            ),
            flush=True,
        )
        del document_ids, query_ids, full_ids
        if config_name != "dense":
            del restored

    return {
        "config": config_name,
        "depth": depth,
        "bits": bits,
        "samples": len(rows),
        "mean_f1": statistics.fmean(row["f1"] for row in rows),
        "mean_generation_seconds": statistics.fmean(
            row["generation_seconds"] for row in rows
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--limit-per-dataset", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--group-size", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.rank < 0 or args.rank >= args.world_size:
        raise SystemExit("rank is outside world size")
    assigned = assigned_configs(args.rank, args.world_size)
    if not assigned:
        raise SystemExit("rank has no assigned configuration")

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    samples = load_samples(args.data, args.limit_per_dataset)
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    print(
        json.dumps(
            {
                "rank": args.rank,
                "assigned": assigned,
                "samples": len(samples),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "gpu": torch.cuda.get_device_name(0),
            }
        ),
        flush=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    load_started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=True,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    model_allocated_bytes = torch.cuda.memory_allocated()

    for config_name in assigned:
        destination = args.run_dir / f"config-{config_name}.json"
        if destination.exists():
            print(f"SKIP existing {destination}", flush=True)
            continue
        result = run_config(
            config_name=config_name,
            model=model,
            tokenizer=tokenizer,
            samples=samples,
            model_allocated_bytes=model_allocated_bytes,
            args=args,
        )
        result.update(
            {
                "rank": args.rank,
                "model": str(args.model),
                "data": str(args.data),
                "data_sha256": data_sha256,
                "model_load_seconds": model_load_seconds,
                "model_allocated_bytes": model_allocated_bytes,
                "prompt_protocol": "longbench-v1-official",
                "dataset_max_new_tokens": {
                    dataset: generation_limit(dataset, args.max_new_tokens)
                    for dataset in sorted({sample["dataset"] for sample in samples})
                },
                "max_input_tokens": args.max_input_tokens,
                "max_new_tokens": args.max_new_tokens,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "group_size": args.group_size,
            }
        )
        atomic_json(destination, result)
        print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
