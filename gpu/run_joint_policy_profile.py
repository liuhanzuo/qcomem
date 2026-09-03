from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from qcomem_joint_policy import (
    SUPPORTED_BITS,
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
    component_nbytes,
    logit_metric_sums,
    merge_metric_sums,
    policy_for_component,
    q16_exactness_passes,
    quantized_policy_state,
    replay_selected_logits,
    selected_query_positions,
)
from qcomem_torch import TorchSplitCausalLM, active_cache_layer_indices
from run_downstream import atomic_json


def component_name(rank: int) -> str:
    return "residual" if rank == 0 else f"cache.{rank - 1}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile one replay component on PG-19 train-only calibration"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-windows-sha256", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--calibration-books", type=int, default=32)
    parser.add_argument("--document-tokens", type=int, default=1024)
    parser.add_argument("--query-tokens", type=int, default=128)
    parser.add_argument("--query-positions", type=int, default=8)
    parser.add_argument("--window-stride", type=int, default=512)
    parser.add_argument("--candidate-windows-per-book", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.world_size != args.depth + 1 or args.world_size != 8:
        raise SystemExit("depth-7 profiling requires residual plus seven cache ranks")
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("rank is outside world size")

    records, data_audit = audit_pg19_train_calibration(
        args.data,
        args.manifest,
        expected_data_sha256=args.expected_data_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        minimum_books=args.calibration_books,
    )

    import transformers
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    windows, windows_sha256 = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=args.calibration_books,
        document_tokens=args.document_tokens,
        query_tokens=args.query_tokens,
        stride=args.window_stride,
        candidate_windows_per_book=args.candidate_windows_per_book,
        seed=args.seed,
    )
    if windows_sha256 != args.expected_windows_sha256:
        raise SystemExit(
            f"PG-19 calibration windows SHA256 mismatch: {windows_sha256}"
        )
    positions = selected_query_positions(args.query_tokens, args.query_positions)
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    adapter = TorchSplitCausalLM(model)
    rows: list[dict[str, Any]] = []
    detected_kind: str | None = None

    for window_index, window in enumerate(windows):
        document = window.document_ids.unsqueeze(0).cuda()
        query = window.query_ids.unsqueeze(0).cuda()
        raw = adapter.write_lower_replay(document, args.depth)
        active = active_cache_layer_indices(raw.cache)
        if len(active) != args.depth:
            raise RuntimeError(
                f"expected {args.depth} active lower cache layers, found {len(active)}"
            )
        if args.rank == 0:
            current_kind = "residual"
        else:
            layer = raw.cache.layers[active[args.rank - 1]]
            current_kind = (
                "linear"
                if hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")
                else "attention"
            )
        if detected_kind is None:
            detected_kind = current_kind
        elif detected_kind != current_kind:
            raise RuntimeError("cache component type changed across calibration windows")

        teacher_logits = replay_selected_logits(adapter, raw, query, positions)
        targets = query[0, [position + 1 for position in positions]]
        for bits in SUPPORTED_BITS:
            policy = policy_for_component(args.rank, bits, depth=args.depth)
            packed = quantized_policy_state(raw, policy, group_size=args.group_size)
            candidate_logits = replay_selected_logits(
                adapter, packed, query, positions
            )
            metrics = logit_metric_sums(teacher_logits, candidate_logits, targets)
            row = {
                "source_id": window.source_id,
                "source_object": window.source_object,
                "start_token": window.start_token,
                "window_index": window_index,
                "component": component_name(args.rank),
                "component_kind": current_kind,
                "bits": bits,
                "component_nbytes": component_nbytes(packed, args.rank),
                "total_policy_nbytes": packed.stored_nbytes,
                "metrics": metrics,
            }
            rows.append(row)
            print(
                json.dumps(
                    {
                        "rank": args.rank,
                        "window": window_index,
                        "bits": bits,
                        "forward_kl": metrics["forward_kl_sum"]
                        / metrics["positions"],
                        "top1": metrics["top1_matches"] / metrics["positions"],
                    }
                ),
                flush=True,
            )
            del packed, candidate_logits
        del raw, teacher_logits, document, query
        torch.cuda.empty_cache()

    options = []
    for bits in SUPPORTED_BITS:
        selected = [row for row in rows if row["bits"] == bits]
        summary = merge_metric_sums(row["metrics"] for row in selected)
        options.append(
            {
                "bits": bits,
                "mean_component_nbytes": round(
                    statistics.fmean(row["component_nbytes"] for row in selected)
                ),
                "mean_total_policy_nbytes": round(
                    statistics.fmean(row["total_policy_nbytes"] for row in selected)
                ),
                "metrics": summary,
            }
        )
    q16_summary = next(item["metrics"] for item in options if item["bits"] == 16)
    if not q16_exactness_passes(q16_summary):
        raise SystemExit(f"rank {args.rank} Q16 replay exactness gate failed: {q16_summary}")

    result = {
        "status": "completed",
        "stage": "expanded_pg19_component_profile",
        "rank": args.rank,
        "world_size": args.world_size,
        "component": component_name(args.rank),
        "component_kind": detected_kind,
        "depth": args.depth,
        "options": options,
        "rows": rows,
        "protocol": {
            **data_audit,
            "manifest": str(args.manifest),
            "calibration_books": args.calibration_books,
            "one_window_per_book": True,
            "document_tokens": args.document_tokens,
            "query_tokens": args.query_tokens,
            "query_positions": list(positions),
            "window_stride": args.window_stride,
            "candidate_windows_per_book": args.candidate_windows_per_book,
            "selection_seed": args.seed,
            "selected_windows_sha256": windows_sha256,
            "objective_data": (
                "teacher logits and natural next-token continuation from official "
                "PG-19 train objects; no downstream QA labels"
            ),
            "formal_longbench_validation_labels_reusable_for_selection": False,
        },
        "model": str(args.model),
        "model_load_seconds": load_seconds,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.perf_counter() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    destination = args.run_dir / f"joint-profile-{args.rank}.json"
    atomic_json(destination, result)
    print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
