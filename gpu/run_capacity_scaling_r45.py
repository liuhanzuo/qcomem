"""R45 deep-j / long-document capacity sweep.

Thin wrapper over ``run_capacity_scaling`` that makes the split depth and the
document length sweepable from the command line instead of fixed constants.

The question this answers is the one the R44 panel and the A4 results raised and
the existing suites cannot: the published operating point ``j = 7`` skips only
7 of 40 layers, so it structurally caps the prefill saving at 17.5% and, at
4,096-token documents, that saving is smaller than the dequantization overhead.
Both a deeper split and a longer document should move that balance, because the
saved fraction is ``j / L`` of a document cost that grows with length while the
per-request reconstruction overhead does not.

Nothing here changes any existing measurement path: ``measure_config`` and
``repeated_document`` are imported unchanged from ``run_capacity_scaling``, so a
row produced here is directly comparable to a row from the published suites.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import torch

from qcomem_torch import TorchSplitCausalLM
from run_capacity_scaling import measure_config, repeated_document
from run_downstream import atomic_json


def parse_int_list(raw: str) -> tuple[int, ...]:
    values = tuple(int(part) for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("expected at least one integer")
    return values


def build_configs(depths: tuple[int, ...], bits: str) -> tuple[str, ...]:
    return ("dense", "prefix") + tuple(f"replay-d{d}-{bits}" for d in depths)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument(
        "--depths",
        type=str,
        default="7,13,20,26",
        help="split depths to sweep, comma separated",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        default="4096,8192,16384,32768",
        help="document token lengths to sweep, comma separated",
    )
    parser.add_argument(
        "--bits",
        type=str,
        default="r4-a8-l8",
        help="bit-width suffix appended to each replay config name",
    )
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")

    depths = parse_int_list(args.depths)
    lengths = parse_int_list(args.lengths)
    configs = build_configs(depths, args.bits)

    # Each rank owns one document length; ranks beyond the length list repeat it
    # so that an eight-way launch still covers every length at least once.
    if args.rank < 0 or args.rank >= args.world_size:
        raise SystemExit(f"rank must be in [0, {args.world_size - 1}]")
    length = lengths[args.rank % len(lengths)]

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    torch.cuda.synchronize()
    model_allocated_bytes = torch.cuda.memory_allocated()

    num_layers = getattr(model.config, "num_hidden_layers", None)
    if num_layers is not None:
        too_deep = [d for d in depths if d >= num_layers]
        if too_deep:
            raise SystemExit(
                f"depths {too_deep} are not below the model's {num_layers} layers"
            )

    document_ids = repeated_document(tokenizer, length, torch.device("cuda"))
    query_ids = tokenizer(
        "\nQuestion: What kind of document is described?\nAnswer:",
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids.cuda()
    adapter = TorchSplitCausalLM(model)

    rows: list[dict[str, Any]] = []
    for name in configs:
        for repeat in range(args.repeats):
            row = measure_config(
                name=name,
                adapter=adapter,
                document_ids=document_ids,
                query_ids=query_ids,
                model_allocated_bytes=model_allocated_bytes,
            )
            row["repeat"] = repeat
            row["document_tokens"] = length
            row["num_hidden_layers"] = num_layers
            rows.append(row)

    summary: list[dict[str, Any]] = []
    for name in configs:
        matching = [r for r in rows if r.get("config") == name]
        if not matching:
            continue

        def mean_of(key: str) -> float | None:
            values = [
                r[key] for r in matching if isinstance(r.get(key), (int, float))
            ]
            return statistics.fmean(values) if values else None

        def median_of(key: str) -> float | None:
            values = [
                r[key] for r in matching if isinstance(r.get(key), (int, float))
            ]
            return statistics.median(values) if values else None

        persistent = mean_of("stored_persistent_nbytes")
        summary.append(
            {
                "config": name,
                "document_tokens": length,
                "depth": matching[0].get("depth"),
                "repetitions": len(matching),
                "mean_write_seconds": mean_of("write_seconds"),
                "mean_generation_seconds": mean_of("generation_seconds"),
                "median_generation_seconds": median_of("generation_seconds"),
                "mean_persistent_mib": (
                    persistent / 2**20 if persistent is not None else None
                ),
                "mean_residual_mib": (
                    (mean_of("stored_residual_nbytes") or 0) / 2**20
                    if mean_of("stored_residual_nbytes") is not None
                    else None
                ),
                "mean_state_mib": (
                    (mean_of("stored_state_nbytes") or 0) / 2**20
                    if mean_of("stored_state_nbytes") is not None
                    else None
                ),
            }
        )

    args.run_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.run_dir / f"deepj-rank-{args.rank}.json",
        {
            "status": "completed",
            "rank": args.rank,
            "document_tokens": length,
            "depths": list(depths),
            "lengths": list(lengths),
            "bits": args.bits,
            "configs": list(configs),
            "num_hidden_layers": num_layers,
            "model_allocated_bytes": model_allocated_bytes,
            "repeats": args.repeats,
            "rows": rows,
            "summary": summary,
        },
    )
    print(
        f"DEEPJ_RANK_DONE rank={args.rank} length={length} "
        f"configs={len(configs)} rows={len(rows)}"
    )


if __name__ == "__main__":
    main()
