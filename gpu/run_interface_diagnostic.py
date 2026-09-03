from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from qcomem_lora import (
    assert_interface_adapter_semantics,
    load_inference_lora_checkpoint,
    set_lora_enabled,
)
from qcomem_torch import (
    TorchSplitCausalLM,
    greedy_generate_comem,
    greedy_generate_dense,
    greedy_generate_oracle,
    tensor_nbytes,
)
from run_downstream import (
    answer_f1,
    atomic_json,
    generation_limit,
    load_samples,
    prompt_parts,
)


STANDARD_CONFIGS = (
    "dense",
    "oracle-d10",
    "document-d7",
    "chunk-d7",
    "document-d10",
    "chunk-d10",
    "document-d13",
    "chunk-d13",
)
LORA_VALIDATION_CONFIGS = (
    "dense",
    "chunk-d7",
    "chunk-lora-d7",
)
CONFIG_SUITES = {
    "standard": STANDARD_CONFIGS,
    "lora-validation": LORA_VALIDATION_CONFIGS,
}
CONFIGS = STANDARD_CONFIGS
FROZEN_LONGBENCH_TEST_V2_SHA256 = (
    "fe046477bf5b39629e9f66fd4def7a55c2d5d1f073c8bb601ee3833f08eaaa5f"
)


def timed_gpu(operation):
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def parse_config(config_name: str) -> tuple[str, int | None]:
    if config_name == "dense":
        return "dense", None
    normalized = config_name.replace("chunk-lora-", "chunk-")
    mode, raw_depth = normalized.split("-d", 1)
    if mode not in {"oracle", "document", "chunk"}:
        raise ValueError(f"unknown interface mode: {mode}")
    return mode, int(raw_depth)


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
    mode, depth = parse_config(config_name)
    eos_value = tokenizer.eos_token_id
    eos_ids = {int(eos_value)} if isinstance(eos_value, int) else set(eos_value or [])
    rows = []
    for sample in samples:
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
        write_seconds = None
        stored_residual_nbytes = None

        if mode == "dense":
            generated, generation_seconds = timed_gpu(
                lambda: greedy_generate_dense(
                    adapter,
                    full_ids,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_ids,
                )
            )
        elif mode == "oracle":
            assert depth is not None
            generated, generation_seconds = timed_gpu(
                lambda: greedy_generate_oracle(
                    adapter,
                    document_ids,
                    query_ids,
                    depth=depth,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_ids,
                )
            )
        else:
            assert depth is not None
            if mode == "document":
                parts, write_seconds = timed_gpu(
                    lambda: [adapter.run_to_depth(document_ids, depth)]
                )
            else:
                parts, write_seconds = timed_gpu(
                    lambda: adapter.chunk_local_write_parts(
                        document_ids,
                        depth,
                        chunk_size=args.chunk_size,
                        overlap=args.overlap,
                    )
                )
            stored_residual_nbytes = sum(tensor_nbytes(part) for part in parts)
            generated, generation_seconds = timed_gpu(
                lambda: greedy_generate_comem(
                    adapter,
                    parts,
                    query_ids,
                    depth=depth,
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_ids,
                )
            )

        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()
        references = [str(answer) for answer in sample["answers"]]
        f1 = max(answer_f1(prediction, reference) for reference in references)
        peak_allocated_bytes = torch.cuda.max_memory_allocated()
        rows.append(
            {
                "dataset": sample["dataset"],
                "id": sample.get("_id"),
                "source_index": sample.get("_source_index"),
                "question": sample["input"],
                "references": references,
                "prediction": prediction,
                "f1": f1,
                "prefix_tokens": prefix_tokens,
                "context_tokens": context_tokens,
                "original_context_tokens": original_context_tokens,
                "context_truncated": context_tokens < original_context_tokens,
                "input_tokens": int(full_ids.numel()),
                "max_new_tokens": max_new_tokens,
                "generated_tokens": len(generated),
                "write_seconds": write_seconds,
                "generation_seconds": generation_seconds,
                "stored_residual_nbytes": stored_residual_nbytes,
                "model_allocated_bytes": model_allocated_bytes,
                "peak_allocated_bytes": peak_allocated_bytes,
                "incremental_peak_allocated_bytes": max(
                    peak_allocated_bytes - model_allocated_bytes, 0
                ),
            }
        )
        print(
            json.dumps(
                {
                    "config": config_name,
                    "dataset": sample["dataset"],
                    "source_index": sample.get("_source_index"),
                    "f1": f1,
                    "generated_tokens": len(generated),
                }
            ),
            flush=True,
        )
        del document_ids, query_ids, full_ids
        if mode in {"document", "chunk"}:
            del parts

    return {
        "config": config_name,
        "mode": mode,
        "depth": depth,
        "samples": len(rows),
        "mean_f1": statistics.fmean(row["f1"] for row in rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--limit-per-dataset", type=int, default=32)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--suite", choices=sorted(CONFIG_SUITES), default="standard")
    parser.add_argument("--source-index-start", type=int)
    parser.add_argument("--source-index-end", type=int)
    parser.add_argument("--exclude-source-indices", default="")
    parser.add_argument("--lora-checkpoint", type=Path)
    parser.add_argument("--expected-data-sha256")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.rank < 0 or args.rank >= args.world_size:
        raise SystemExit("rank is outside world size")

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    if args.suite == "lora-validation" and args.lora_checkpoint is None:
        raise SystemExit("lora-validation requires --lora-checkpoint")
    if args.suite == "standard" and args.lora_checkpoint is not None:
        raise SystemExit("standard suite does not accept an Interface LoRA checkpoint")
    if args.suite == "lora-validation":
        normalized_path = str(args.data).lower().replace("_", "-")
        if "longbench-test-v2" in normalized_path:
            raise SystemExit("lora-validation refuses frozen LongBench test-v2 by path")
        if args.expected_data_sha256 is None:
            raise SystemExit("lora-validation requires --expected-data-sha256")
        if (
            args.limit_per_dataset,
            args.source_index_start,
            args.source_index_end,
        ) != (30, 6, 35):
            raise SystemExit(
                "lora-validation is frozen to 30 samples/dataset at source indices 6-35"
            )
    data_sha256 = hashlib.sha256(args.data.read_bytes()).hexdigest()
    if args.suite == "lora-validation":
        if data_sha256 == FROZEN_LONGBENCH_TEST_V2_SHA256:
            raise SystemExit("lora-validation refuses frozen LongBench test-v2 by SHA256")
        if data_sha256 != args.expected_data_sha256:
            raise SystemExit("lora-validation data SHA256 does not match the frozen input")
    excluded = tuple(
        int(value) for value in args.exclude_source_indices.split(",") if value.strip()
    )
    all_samples = load_samples(
        args.data,
        args.limit_per_dataset,
        source_index_start=args.source_index_start,
        source_index_end=args.source_index_end,
        exclude_source_indices=excluded,
    )
    if args.suite == "lora-validation":
        expected_indices = set(range(6, 36))
        datasets = {sample["dataset"] for sample in all_samples}
        if datasets != {"qasper", "2wikimqa"} or len(all_samples) != 60:
            raise SystemExit("lora-validation requires exactly 30 Qasper + 30 2Wiki rows")
        for dataset in datasets:
            indices = {
                int(sample["_source_index"])
                for sample in all_samples
                if sample["dataset"] == dataset
            }
            if indices != expected_indices:
                raise SystemExit(
                    f"{dataset}: lora-validation requires source indices 6-35"
                )
    samples = all_samples[args.rank :: args.world_size]
    if not samples:
        raise SystemExit("rank has no sample shard")
    print(
        json.dumps(
            {
                "rank": args.rank,
                "configs": list(CONFIG_SUITES[args.suite]),
                "samples": len(samples),
                "source_indices": [sample.get("_source_index") for sample in samples],
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
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    lora_metadata = None
    lora_checkpoint_sha256 = None
    if args.lora_checkpoint is not None:
        split = TorchSplitCausalLM(model)
        lora_metadata = load_inference_lora_checkpoint(
            model, split.layers, args.lora_checkpoint
        )
        try:
            assert_interface_adapter_semantics(
                lora_metadata,
                depth=7,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        lora_checkpoint_sha256 = hashlib.sha256(
            args.lora_checkpoint.read_bytes()
        ).hexdigest()
    model.eval().cuda()
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    model_allocated_bytes = torch.cuda.memory_allocated()

    for config_name in CONFIG_SUITES[args.suite]:
        lora_enabled = config_name == "chunk-lora-d7"
        set_lora_enabled(model, lora_enabled)
        destination = args.run_dir / f"shard-{args.rank}-{config_name}.json"
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
                "world_size": args.world_size,
                "model": str(args.model),
                "data": str(args.data),
                "data_sha256": data_sha256,
                "prompt_protocol": "longbench-v1-official",
                "model_load_seconds": model_load_seconds,
                "model_allocated_bytes": model_allocated_bytes,
                "max_input_tokens": args.max_input_tokens,
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "suite": args.suite,
                "source_index_start": args.source_index_start,
                "source_index_end": args.source_index_end,
                "exclude_source_indices": list(excluded),
                "lora": {
                    "checkpoint": str(args.lora_checkpoint)
                    if args.lora_checkpoint is not None
                    else None,
                    "checkpoint_sha256": lora_checkpoint_sha256,
                    "enabled_for_this_config": lora_enabled,
                    "checkpoint_metadata": lora_metadata,
                },
            }
        )
        atomic_json(destination, result)
        print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
