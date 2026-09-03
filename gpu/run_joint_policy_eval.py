from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from qcomem_joint_policy import (
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
    logit_metric_sums,
    merge_metric_sums,
    policy_from_dict,
    q16_exactness_passes,
    quantized_policy_state,
    replay_selected_logits,
    selected_query_positions,
)
from qcomem_torch import TorchSplitCausalLM, active_cache_layer_indices
from run_downstream import atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate jointly quantized policies on PG-19 train calibration"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-windows-sha256", required=True)
    parser.add_argument("--candidate-file", type=Path, required=True)
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
    if args.world_size != 8 or args.depth != 7:
        raise SystemExit("the frozen expanded calibration protocol is depth 7 / 8 GPUs")
    if not 0 <= args.rank < args.world_size:
        raise SystemExit("rank is outside world size")

    records, data_audit = audit_pg19_train_calibration(
        args.data,
        args.manifest,
        expected_data_sha256=args.expected_data_sha256,
        expected_manifest_sha256=args.expected_manifest_sha256,
        minimum_books=args.calibration_books,
    )
    candidate_bytes = args.candidate_file.read_bytes()
    candidate_payload = json.loads(candidate_bytes)
    if candidate_payload.get("status") != "candidates_frozen_before_joint_evaluation":
        raise SystemExit("joint policy candidate file is not frozen")
    contract = candidate_payload.get("selection_contract", {})
    required_false = (
        "longbench_validation_labels_used",
        "formal_validation_source_6_35_may_select_policy",
        "frozen_test_v2_source_68_99_used",
        "automatic_policy_reuses_formal_validation_label",
        "legacy_layer_validation_names_reused",
    )
    if any(contract.get(field) is not False for field in required_false):
        raise SystemExit("candidate contract does not fail closed on LongBench selection")
    all_policies = [policy_from_dict(item) for item in candidate_payload["evaluation_policies"]]
    if len({policy.name for policy in all_policies}) != len(all_policies):
        raise SystemExit("candidate policy names are not unique")
    if any(policy.name.startswith("replay-d7-") for policy in all_policies):
        raise SystemExit("legacy formal-validation labels cannot be reused")
    assigned = [
        policy for index, policy in enumerate(all_policies) if index % args.world_size == args.rank
    ]
    if not assigned:
        raise SystemExit("rank has no joint policy assignment")

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
    if windows_sha256 != candidate_payload["protocol"]["selected_windows_sha256"]:
        raise SystemExit("joint evaluation windows differ from component profiling")
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

    for window_index, window in enumerate(windows):
        document = window.document_ids.unsqueeze(0).cuda()
        query = window.query_ids.unsqueeze(0).cuda()
        raw = adapter.write_lower_replay(document, args.depth)
        active = active_cache_layer_indices(raw.cache)
        if len(active) != args.depth:
            raise RuntimeError(
                f"expected {args.depth} active lower cache layers, found {len(active)}"
            )
        teacher_logits = replay_selected_logits(adapter, raw, query, positions)
        targets = query[0, [position + 1 for position in positions]]
        for policy in assigned:
            packed = quantized_policy_state(raw, policy, group_size=args.group_size)
            candidate_logits = replay_selected_logits(adapter, packed, query, positions)
            metrics = logit_metric_sums(teacher_logits, candidate_logits, targets)
            rows.append(
                {
                    "source_id": window.source_id,
                    "source_object": window.source_object,
                    "start_token": window.start_token,
                    "window_index": window_index,
                    "policy": policy.name,
                    "selection_group": policy.selection_group,
                    "residual_bits": policy.residual_bits,
                    "cache_layer_bits": list(policy.cache_layer_bits),
                    "persistent_nbytes": packed.stored_nbytes,
                    "metrics": metrics,
                }
            )
            print(
                json.dumps(
                    {
                        "rank": args.rank,
                        "window": window_index,
                        "policy": policy.name,
                        "forward_kl": metrics["forward_kl_sum"]
                        / metrics["positions"],
                        "top1": metrics["top1_matches"] / metrics["positions"],
                        "persistent_mib": packed.stored_nbytes / 2**20,
                    }
                ),
                flush=True,
            )
            del packed, candidate_logits
        del raw, teacher_logits, document, query
        torch.cuda.empty_cache()

    summaries = []
    for policy in assigned:
        selected = [row for row in rows if row["policy"] == policy.name]
        metrics = merge_metric_sums(row["metrics"] for row in selected)
        summaries.append(
            {
                **policy.as_dict(),
                "samples": len(selected),
                "mean_persistent_nbytes": round(
                    statistics.fmean(row["persistent_nbytes"] for row in selected)
                ),
                "metrics": metrics,
            }
        )
        if policy.name == "q16-control" and not q16_exactness_passes(metrics):
            raise SystemExit(f"joint Q16 exactness gate failed: {metrics}")

    protocol = {
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
    }
    if protocol != candidate_payload["protocol"]:
        raise SystemExit("joint evaluation protocol drifted from candidate generation")
    result = {
        "status": "completed",
        "stage": "expanded_pg19_joint_quantization_evaluation",
        "rank": args.rank,
        "world_size": args.world_size,
        "assigned_policies": [policy.name for policy in assigned],
        "summaries": summaries,
        "rows": rows,
        "protocol": protocol,
        "candidate_file": str(args.candidate_file),
        "candidate_file_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "selection_contract": contract,
        "model": str(args.model),
        "model_load_seconds": load_seconds,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "elapsed_seconds": time.perf_counter() - started,
        "max_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    destination = args.run_dir / f"joint-eval-{args.rank}.json"
    atomic_json(destination, result)
    print(f"SAVED {destination}", flush=True)


if __name__ == "__main__":
    main()
