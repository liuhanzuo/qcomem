from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import mlx.core as mx
import psutil

from .comem_bench import DEFAULT_MODEL, default_depths, residual_metrics
from .comem_dataset import (
    BM25Selector,
    MultiDocumentDataset,
    Query,
    evidence_recall,
    load_dataset,
    render_document,
    render_query_prefix,
)
from .comem_model import SplitCausalLM
from .comem_quant import DepthBitPolicy, StoredResidual, quantize_residual
from .experiment_env import (
    assess_completed_run,
    assess_preflight,
    collect_experiment_snapshot,
)


T = TypeVar("T")
DEFAULT_MODEL_REVISION = "7f0dc925e0d0afb0322d96f9255cfddf2ba5636e"


def _evaluate(value: Any) -> None:
    if isinstance(value, mx.array):
        mx.eval(value)
    elif isinstance(value, StoredResidual):
        value.eval()
    elif isinstance(value, (list, tuple)):
        for item in value:
            _evaluate(item)
    else:
        raise TypeError(f"cannot evaluate value of type {type(value)!r}")


def timed(operation: Callable[[], T], stream: mx.Stream) -> tuple[T, float]:
    mx.synchronize(stream)
    started = time.perf_counter()
    value = operation()
    _evaluate(value)
    mx.synchronize(stream)
    return value, time.perf_counter() - started


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _encode(tokenizer, text: str) -> list[int]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        raise ValueError("tokenizer produced an empty sequence")
    return tokens


def _query_tokens(tokenizer, query: Query) -> tuple[mx.array, int, mx.array | None]:
    prefix = _encode(tokenizer, render_query_prefix(query))
    if query.expected_answer is None:
        return mx.array(prefix), len(prefix), None
    answer = _encode(tokenizer, query.expected_answer)
    return mx.array(prefix + answer), len(prefix), mx.array(answer)


def quality_metrics(
    reference: mx.array,
    candidate: mx.array,
    *,
    document_tokens: int,
    query_prefix_tokens: int,
    answer_tokens: mx.array | None,
) -> dict[str, Any]:
    """Compare either all teacher-forced answer positions or the final logit."""

    reference = reference.astype(mx.float32)
    candidate = candidate.astype(mx.float32)
    if answer_tokens is None:
        reference = reference[:, -1:, :]
        candidate = candidate[:, -1:, :]
        targets = None
        scope = "next_token"
    else:
        answer_length = answer_tokens.shape[0]
        start = document_tokens + query_prefix_tokens - 1
        reference = reference[:, start : start + answer_length, :]
        candidate = candidate[:, start : start + answer_length, :]
        targets = answer_tokens[None, :, None]
        scope = "teacher_forced_answer"

    reference_logp = reference - mx.logsumexp(reference, axis=-1, keepdims=True)
    candidate_logp = candidate - mx.logsumexp(candidate, axis=-1, keepdims=True)
    position_kl = mx.sum(
        mx.exp(reference_logp) * (reference_logp - candidate_logp), axis=-1
    )
    max_logit_error = mx.max(mx.abs(reference - candidate))
    reference_top1 = mx.argmax(reference, axis=-1)
    candidate_top1 = mx.argmax(candidate, axis=-1)
    agreement = reference_top1 == candidate_top1

    arrays = [position_kl, max_logit_error, agreement]
    target_metrics: dict[str, Any] = {}
    if targets is not None:
        reference_target_logp = mx.take_along_axis(
            reference_logp, targets, axis=-1
        ).squeeze(-1)
        candidate_target_logp = mx.take_along_axis(
            candidate_logp, targets, axis=-1
        ).squeeze(-1)
        target_accuracy_reference = reference_top1 == answer_tokens[None]
        target_accuracy_candidate = candidate_top1 == answer_tokens[None]
        arrays.extend(
            [
                reference_target_logp,
                candidate_target_logp,
                target_accuracy_reference,
                target_accuracy_candidate,
            ]
        )
    mx.eval(*arrays)

    if targets is not None:
        target_metrics = {
            "reference_answer_nll": float((-reference_target_logp.mean()).item()),
            "candidate_answer_nll": float((-candidate_target_logp.mean()).item()),
            "answer_nll_delta": float(
                (reference_target_logp.mean() - candidate_target_logp.mean()).item()
            ),
            "reference_target_accuracy": float(
                target_accuracy_reference.astype(mx.float32).mean().item()
            ),
            "candidate_target_accuracy": float(
                target_accuracy_candidate.astype(mx.float32).mean().item()
            ),
            "evaluated_answer_tokens": int(answer_tokens.shape[0]),
        }
    agreement_rate = float(agreement.astype(mx.float32).mean().item())
    return {
        "evaluation_scope": scope,
        "kl_divergence": float(position_kl.mean().item()),
        "max_position_kl": float(position_kl.max().item()),
        "max_logit_abs_error": float(max_logit_error.item()),
        "top1_agreement_rate": agreement_rate,
        "top1_match": agreement_rate == 1.0,
        **target_metrics,
    }


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    depth: int,
    bits: int,
    corpus_stored_nbytes: int,
    corpus_dense_nbytes: int,
    quantize_seconds: float,
    residual: Mapping[str, float],
) -> dict[str, Any]:
    kls = [float(row["kl_divergence"]) for row in rows]
    position_kls = [float(row["max_position_kl"]) for row in rows]
    agreements = [float(row["top1_agreement_rate"]) for row in rows]
    selected_bytes = [float(row["selected_stored_nbytes"]) for row in rows]
    result = {
        "depth": depth,
        "bits": bits,
        "queries": len(rows),
        "corpus_stored_nbytes": corpus_stored_nbytes,
        "corpus_dense_nbytes": corpus_dense_nbytes,
        "corpus_compression_ratio": corpus_dense_nbytes / corpus_stored_nbytes,
        "mean_selected_stored_nbytes": _mean(selected_bytes),
        "quantize_corpus_seconds": quantize_seconds,
        "mean_kl": _mean(kls),
        "p95_query_mean_kl": _percentile(kls, 0.95),
        "max_query_mean_kl": max(kls),
        "max_position_kl": max(position_kls),
        "mean_top1_agreement_rate": _mean(agreements),
        "min_query_top1_agreement_rate": min(agreements),
        "all_queries_top1_match": all(bool(row["top1_match"]) for row in rows),
        "mean_dequantize_seconds": _mean(
            [float(row["dequantize_seconds"]) for row in rows]
        ),
        "mean_suffix_read_seconds": _mean(
            [float(row["suffix_read_seconds"]) for row in rows]
        ),
        "mean_online_dequantize_read_seconds": _mean(
            [float(row["online_dequantize_read_seconds"]) for row in rows]
        ),
        **residual,
    }
    if all("answer_nll_delta" in row for row in rows):
        result["mean_answer_nll_delta"] = _mean(
            [float(row["answer_nll_delta"]) for row in rows]
        )
        result["max_abs_answer_nll_delta"] = max(
            abs(float(row["answer_nll_delta"])) for row in rows
        )
    return result


def select_aggregate_policy(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    max_kl: float,
    max_relative_rmse: float,
    min_top1_agreement: float,
) -> DepthBitPolicy:
    assignments = {}
    by_depth: dict[int, list[Mapping[str, Any]]] = {}
    for row in aggregates:
        by_depth.setdefault(int(row["depth"]), []).append(row)
    for depth, candidates in by_depth.items():
        eligible = [
            row
            for row in candidates
            if float(row["max_position_kl"]) <= max_kl
            and float(row["residual_relative_rmse"]) <= max_relative_rmse
            and float(row["min_query_top1_agreement_rate"])
            >= min_top1_agreement
        ]
        if eligible:
            selected = min(
                eligible,
                key=lambda row: (
                    int(row["corpus_stored_nbytes"]),
                    int(row["bits"]),
                ),
            )
            assignments[depth] = int(selected["bits"])
        else:
            assignments[depth] = 16
    return DepthBitPolicy(assignments)


def _safe_id(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )
    return safe or "document"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _selected_documents(
    dataset: MultiDocumentDataset,
    *,
    method: str,
    top_k: int,
) -> tuple[dict[str, list[str]], dict[str, list[tuple[str, float]]] | None]:
    if method == "frozen":
        selected = {}
        for query in dataset.queries:
            if not query.selected_document_ids:
                raise ValueError(
                    f"query {query.id!r} has no frozen selected_document_ids"
                )
            selected[query.id] = list(query.selected_document_ids)
        return selected, None
    selector = BM25Selector(dataset.documents)
    rankings = {query.id: selector.score(query.text) for query in dataset.queries}
    return (
        {
            query.id: [document_id for document_id, _ in rankings[query.id][:top_k]]
            for query in dataset.queries
        },
        rankings,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-document, multi-query Q-CoMem calibration benchmark"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("configs/comem_multidoc_demo.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--model-revision",
        help="immutable Hugging Face commit; the built-in model has a pinned default",
    )
    parser.add_argument("--depths", type=int, nargs="+")
    parser.add_argument("--bits", type=int, nargs="+", default=[2, 4, 8, 16])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=16)
    parser.add_argument("--selection", choices=["frozen", "bm25"], default="frozen")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--kl-threshold", type=float, default=0.02)
    parser.add_argument("--max-relative-rmse", type=float, default=0.05)
    parser.add_argument("--min-top1-agreement", type=float, default=1.0)
    parser.add_argument("--max-preflight-cpu", type=float, default=35.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=128.0)
    parser.add_argument(
        "--power-policy",
        choices=["require-ac", "record-only"],
        default="require-ac",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/q_comem_multidoc_benchmark.json"),
    )
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=Path("results/q_comem_multidoc_store"),
    )
    parser.add_argument("--no-save-store", action="store_true")
    args = parser.parse_args()

    if args.chunk_size < 1:
        raise SystemExit("--chunk-size must be positive")
    if args.overlap < 0 or args.overlap >= args.chunk_size:
        raise SystemExit("--overlap must satisfy 0 <= overlap < chunk-size")
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    if not 0 <= args.min_top1_agreement <= 1:
        raise SystemExit("--min-top1-agreement must be between zero and one")
    if args.kl_threshold < 0 or args.max_relative_rmse < 0:
        raise SystemExit("quality thresholds must be non-negative")

    dataset = load_dataset(args.dataset)
    model_revision = args.model_revision
    if model_revision is None and args.model == DEFAULT_MODEL:
        model_revision = DEFAULT_MODEL_REVISION
    if model_revision is None and not Path(args.model).exists():
        raise SystemExit(
            "remote models require --model-revision so the experiment is reproducible"
        )
    selected_by_query, rankings = _selected_documents(
        dataset, method=args.selection, top_k=args.top_k
    )
    before = collect_experiment_snapshot()
    preflight = assess_preflight(
        before, max_cpu_percent=args.max_preflight_cpu
    )
    if args.power_policy == "require-ac" and not preflight["formal_result_eligible"]:
        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "status": "preflight_failed",
            "formal_result_eligible": False,
            "power_policy": args.power_policy,
            "dataset": str(args.dataset),
            "model": args.model,
            "model_revision": model_revision,
            "environment_before": before,
            "environment_assessment": preflight,
        }
        _write_json(args.output, payload)
        reasons = ", ".join(preflight["reasons"])
        raise SystemExit(
            f"formal benchmark preflight failed: {reasons}; details saved to {args.output}"
        )
    if not mx.metal.is_available():
        raise SystemExit("the MLX Metal backend is unavailable")

    mx.set_default_device(mx.gpu)
    gpu_stream = mx.new_stream(mx.gpu)
    from mlx_lm import load

    load_started = time.perf_counter()
    with mx.stream(gpu_stream):
        model, tokenizer = load(args.model, revision=model_revision)
    mx.synchronize(gpu_stream)
    model_load_seconds = time.perf_counter() - load_started
    split_model = SplitCausalLM(model)
    depths = args.depths or default_depths(split_model.num_layers)
    if any(depth < 0 or depth > split_model.num_layers for depth in depths):
        raise SystemExit(f"depths must be in [0, {split_model.num_layers}]")

    document_tokens = {
        document.id: mx.array(_encode(tokenizer, render_document(document)))
        for document in dataset.documents
    }
    query_inputs = {
        query.id: _query_tokens(tokenizer, query) for query in dataset.queries
    }

    # Compile/warm the common full-model path. This output is intentionally not
    # included in measurements.
    first_query = dataset.queries[0]
    warm_ids = selected_by_query[first_query.id]
    warm_tokens = mx.concatenate(
        [*(document_tokens[document_id] for document_id in warm_ids), query_inputs[first_query.id][0]]
    )
    with mx.stream(gpu_stream):
        warm_logits = split_model.full_logits(warm_tokens)
        mx.eval(warm_logits)
    mx.synchronize(gpu_stream)

    dense_baselines = {}
    query_metadata = {}
    for query in dataset.queries:
        selected_ids = selected_by_query[query.id]
        query_tokens, prefix_length, answer_tokens = query_inputs[query.id]
        dense_tokens = mx.concatenate(
            [
                *(document_tokens[document_id] for document_id in selected_ids),
                query_tokens,
            ]
        )
        with mx.stream(gpu_stream):
            dense_logits, dense_seconds = timed(
                lambda dense_tokens=dense_tokens: split_model.full_logits(dense_tokens),
                gpu_stream,
            )
        dense_baselines[query.id] = dense_logits
        query_metadata[query.id] = {
            "query_id": query.id,
            "selected_document_ids": selected_ids,
            "relevant_document_ids": list(query.relevant_document_ids),
            "evidence_recall": evidence_recall(
                selected_ids, query.relevant_document_ids
            ),
            "selected_document_tokens": sum(
                document_tokens[document_id].shape[0] for document_id in selected_ids
            ),
            "query_tokens": int(query_tokens.shape[0]),
            "query_prefix_tokens": prefix_length,
            "answer_tokens": int(answer_tokens.shape[0]) if answer_tokens is not None else 0,
            "dense_full_seconds": dense_seconds,
        }

    stores: dict[tuple[int, int, str], StoredResidual] = {}
    all_query_rows: list[dict[str, Any]] = []
    aggregate_results: list[dict[str, Any]] = []
    depth_results: list[dict[str, Any]] = []

    for depth in depths:
        print(f"Depth {depth}/{split_model.num_layers}: writing corpus once")
        residuals: dict[str, mx.array] = {}
        write_times = {}
        for document in dataset.documents:
            with mx.stream(gpu_stream):
                residual, seconds = timed(
                    lambda document=document: split_model.chunk_local_write(
                        document_tokens[document.id],
                        depth,
                        chunk_size=args.chunk_size,
                        overlap=args.overlap,
                    ),
                    gpu_stream,
                )
            residuals[document.id] = residual
            write_times[document.id] = seconds

        bf16_query_rows = []
        reference_logits_by_query = {}
        query_residual_by_query = {}
        for query in dataset.queries:
            selected_ids = selected_by_query[query.id]
            query_tokens, prefix_length, answer_tokens = query_inputs[query.id]
            with mx.stream(gpu_stream):
                query_residual, query_write_seconds = timed(
                    lambda query_tokens=query_tokens: split_model.run_to_depth(
                        query_tokens, depth
                    ),
                    gpu_stream,
                )
                combined = mx.concatenate(
                    [*(residuals[document_id] for document_id in selected_ids), query_residual],
                    axis=1,
                )
                reference_logits, suffix_seconds = timed(
                    lambda combined=combined: split_model.run_suffix(combined, depth),
                    gpu_stream,
                )
            reference_logits_by_query[query.id] = reference_logits
            query_residual_by_query[query.id] = query_residual
            metadata = query_metadata[query.id]
            interface = quality_metrics(
                dense_baselines[query.id],
                reference_logits,
                document_tokens=int(metadata["selected_document_tokens"]),
                query_prefix_tokens=prefix_length,
                answer_tokens=answer_tokens,
            )
            bf16_query_rows.append(
                {
                    **metadata,
                    "query_write_seconds": query_write_seconds,
                    "suffix_read_seconds": suffix_seconds,
                    "interface_vs_dense": interface,
                }
            )

        depth_results.append(
            {
                "depth": depth,
                "documents": len(dataset.documents),
                "corpus_tokens": sum(
                    tokens.shape[0] for tokens in document_tokens.values()
                ),
                "write_corpus_seconds": sum(write_times.values()),
                "write_document_seconds": write_times,
                "mean_interface_kl_vs_dense": _mean(
                    [
                        float(row["interface_vs_dense"]["kl_divergence"])
                        for row in bf16_query_rows
                    ]
                ),
                "max_interface_position_kl_vs_dense": max(
                    float(row["interface_vs_dense"]["max_position_kl"])
                    for row in bf16_query_rows
                ),
                "mean_interface_top1_agreement_vs_dense": _mean(
                    [
                        float(row["interface_vs_dense"]["top1_agreement_rate"])
                        for row in bf16_query_rows
                    ]
                ),
                "queries": bf16_query_rows,
            }
        )

        corpus_reference = mx.concatenate(list(residuals.values()), axis=1)
        for bits in sorted(set(args.bits)):
            print(f"  {bits}-bit corpus; evaluating {len(dataset.queries)} queries")
            mx.reset_peak_memory()
            quantize_seconds = 0.0
            bit_stores = {}
            for document in dataset.documents:
                with mx.stream(gpu_stream):
                    stored, seconds = timed(
                        lambda document=document, bits=bits: quantize_residual(
                            residuals[document.id],
                            depth=depth,
                            bits=bits,
                            group_size=args.group_size,
                            stream=gpu_stream,
                        ),
                        gpu_stream,
                    )
                stores[(depth, bits, document.id)] = stored
                bit_stores[document.id] = stored
                quantize_seconds += seconds

            with mx.stream(gpu_stream):
                restored_corpus, _ = timed(
                    lambda: [
                        bit_stores[document.id].dequantize(stream=gpu_stream)
                        for document in dataset.documents
                    ],
                    gpu_stream,
                )
            corpus_residual_metrics = residual_metrics(
                corpus_reference, mx.concatenate(restored_corpus, axis=1)
            )

            query_rows = []
            for query in dataset.queries:
                selected_ids = selected_by_query[query.id]
                _, prefix_length, answer_tokens = query_inputs[query.id]
                with mx.stream(gpu_stream):
                    restored_documents, dequantize_seconds = timed(
                        lambda selected_ids=selected_ids: [
                            bit_stores[document_id].dequantize(stream=gpu_stream)
                            for document_id in selected_ids
                        ],
                        gpu_stream,
                    )
                    combined = mx.concatenate(
                        [*restored_documents, query_residual_by_query[query.id]],
                        axis=1,
                    )
                    candidate_logits, suffix_seconds = timed(
                        lambda combined=combined: split_model.run_suffix(combined, depth),
                        gpu_stream,
                    )
                quality = quality_metrics(
                    reference_logits_by_query[query.id],
                    candidate_logits,
                    document_tokens=int(
                        query_metadata[query.id]["selected_document_tokens"]
                    ),
                    query_prefix_tokens=prefix_length,
                    answer_tokens=answer_tokens,
                )
                row = {
                    "depth": depth,
                    "bits": bits,
                    **query_metadata[query.id],
                    "selected_stored_nbytes": sum(
                        bit_stores[document_id].nbytes for document_id in selected_ids
                    ),
                    "dequantize_seconds": dequantize_seconds,
                    "suffix_read_seconds": suffix_seconds,
                    "online_dequantize_read_seconds": dequantize_seconds + suffix_seconds,
                    **quality,
                }
                query_rows.append(row)
                all_query_rows.append(row)

            aggregate = aggregate_rows(
                query_rows,
                depth=depth,
                bits=bits,
                corpus_stored_nbytes=sum(
                    stored.nbytes for stored in bit_stores.values()
                ),
                corpus_dense_nbytes=sum(
                    stored.dense_nbytes for stored in bit_stores.values()
                ),
                quantize_seconds=quantize_seconds,
                residual=corpus_residual_metrics,
            )
            aggregate.update(
                {
                    "mlx_peak_memory_bytes": mx.get_peak_memory(),
                    "mlx_active_memory_bytes": mx.get_active_memory(),
                    "mlx_cache_memory_bytes": mx.get_cache_memory(),
                    "process_rss_bytes": psutil.Process().memory_info().rss,
                }
            )
            aggregate_results.append(aggregate)

    policy = select_aggregate_policy(
        aggregate_results,
        max_kl=args.kl_threshold,
        max_relative_rmse=args.max_relative_rmse,
        min_top1_agreement=args.min_top1_agreement,
    )
    policy_sensitivity = [
        {
            "max_position_kl": args.kl_threshold,
            "max_relative_rmse": args.max_relative_rmse,
            "min_query_top1_agreement_rate": threshold,
            "policy": select_aggregate_policy(
                aggregate_results,
                max_kl=args.kl_threshold,
                max_relative_rmse=args.max_relative_rmse,
                min_top1_agreement=threshold,
            ).as_dict(),
        }
        for threshold in (1.0, 0.99, 0.95, 0.90)
    ]
    store_files: dict[str, Any] = {}
    if not args.no_save_store:
        for depth, bits in policy.assignments.items():
            depth_files = {}
            for document in dataset.documents:
                path = (
                    args.store_dir
                    / f"depth-{depth}-bits-{bits}"
                    / f"{_safe_id(document.id)}.safetensors"
                )
                stores[(depth, bits, document.id)].save(path)
                depth_files[document.id] = {
                    "path": str(path),
                    "file_bytes": path.stat().st_size,
                }
            store_files[str(depth)] = {"bits": bits, "documents": depth_files}

    after = collect_experiment_snapshot()
    completed_assessment = assess_completed_run(
        before,
        after,
        max_cpu_percent=args.max_preflight_cpu,
        max_swap_growth_bytes=round(args.max_swap_growth_mb * 1024**2),
    )
    if args.power_policy == "record-only":
        completed_assessment = {
            **completed_assessment,
            "formal_result_eligible": False,
            "reasons": list(
                dict.fromkeys(
                    [
                        *completed_assessment["reasons"],
                        "record_only_diagnostic_mode",
                    ]
                )
            ),
        }
    output = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "formal_result_eligible": completed_assessment["formal_result_eligible"],
        "power_policy": args.power_policy,
        "runtime": "q-comem-multidoc-mlx-core",
        "device": mx.device_info(mx.gpu),
        "model": args.model,
        "model_revision": model_revision,
        "model_load_seconds": model_load_seconds,
        "num_layers": split_model.num_layers,
        "dataset": {
            "name": dataset.name,
            "path": str(args.dataset),
            "documents": len(dataset.documents),
            "queries": len(dataset.queries),
            "teacher_forced_queries": sum(
                query.expected_answer is not None for query in dataset.queries
            ),
        },
        "selection": {
            "method": args.selection,
            "top_k": args.top_k,
            "selected_by_query": selected_by_query,
            "bm25_rankings": rankings,
            "mean_evidence_recall": _mean(
                [
                    float(query_metadata[query.id]["evidence_recall"])
                    for query in dataset.queries
                    if query_metadata[query.id]["evidence_recall"] is not None
                ]
            ),
        },
        "experiment": {
            "depths": depths,
            "bits": sorted(set(args.bits)),
            "group_size": args.group_size,
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
            "write_reuse": "each document residual is written once per depth and reused by every query",
            "quality_scope": "teacher-forced answer tokens when expected_answer is present",
            "timing_note": "single measurements after one dense-path warm-up; use repeated fresh-process runs for publication",
        },
        "calibration": {
            "scope": "aggregate over all dataset queries",
            "max_position_kl": args.kl_threshold,
            "max_relative_rmse": args.max_relative_rmse,
            "min_query_top1_agreement_rate": args.min_top1_agreement,
        },
        "selected_policy": policy.as_dict(),
        "policy_sensitivity": policy_sensitivity,
        "store_files": store_files,
        "environment_before": before,
        "environment_after": after,
        "environment_assessment": completed_assessment,
        "depth_results": depth_results,
        "aggregate_results": aggregate_results,
        "query_results": all_query_rows,
    }
    _write_json(args.output, output)
    print(
        json.dumps(
            {
                "formal_result_eligible": output["formal_result_eligible"],
                "invalid_reasons": completed_assessment["reasons"],
                "selected_policy": policy.as_dict(),
            },
            indent=2,
        )
    )
    print(f"Saved benchmark: {args.output}")


if __name__ == "__main__":
    main()
