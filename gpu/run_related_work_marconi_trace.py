#!/usr/bin/env python3
"""Run a bounded Marconi policy trace on the frozen LongBench workloads.

This is deliberately a policy-simulator experiment.  It does not report model
quality or wall-clock serving throughput, and it must not be merged into the
same-protocol Qwen3.5 serving table.  The simulator keeps Marconi's published
native Attention--Mamba2 geometry while replacing only the request trace with
the eight frozen LongBench workloads used by the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any


DEBUG_SCHEMA = "forkaudit-related-marconi-policy-trace-debug-v1"
FORMAL_SCHEMA = "forkaudit-related-marconi-policy-trace-formal-v1"
PREREG_SCHEMA = "forkaudit-related-marconi-policy-preregistration-v1"
DATA_SHA256 = "1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe"
MARCONI_COMMIT = "08016617b1524e6bf6ac29b680641cc945bda7f0"
EXPECTED_PAIRS = tuple(
    (dataset, source_index)
    for dataset in ("qasper", "2wikimqa")
    for source_index in range(6, 10)
)
REQUEST_COUNTS = (32, 24, 20, 16, 12, 10, 8, 6)
# A 4K request occupies roughly 3.7 GB in the official vLLM+ hybrid-state
# simulator.  Budgets below that point cannot hold even one request and the
# artifact's approximate eviction accounting can finish above budget.  These
# three budgets are the smallest useful shared sweep for this long trace.
CAPACITIES_BYTES = (5_000_000_000, 10_000_000_000, 20_000_000_000)
NATIVE_GEOMETRY = {
    "num_ssm_layers": 24,
    "num_attn_layers": 4,
    "num_mlp_layers": 28,
    "d": 4096,
    "n": 128,
}


class TraceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TraceError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    temporary.replace(path)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smooth_weighted_schedule(counts: tuple[int, ...]) -> list[int]:
    """Return a deterministic interleaving with the exact requested counts."""
    require(bool(counts) and all(value > 0 for value in counts), "invalid counts")
    total = sum(counts)
    current = [0] * len(counts)
    remaining = list(counts)
    schedule: list[int] = []
    for _ in range(total):
        for index, weight in enumerate(counts):
            if remaining[index] > 0:
                current[index] += weight
        candidates = [index for index, value in enumerate(remaining) if value > 0]
        selected = max(candidates, key=lambda index: (current[index], -index))
        schedule.append(selected)
        current[selected] -= total
        remaining[selected] -= 1
    require(
        tuple(schedule.count(index) for index in range(len(counts))) == counts,
        "schedule count drift",
    )
    return schedule


def collect_requests(
    *, model: Path, data: Path, serving_run: Path, loader_path: Path
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    require(sha256_file(data) == DATA_SHA256, "frozen LongBench digest drift")
    loader = load_module("forkaudit_related_serving_loader", loader_path)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    requests: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    for rank in range(8):
        workload = loader.load_workload(model, data, rank)
        observed_pair = (str(workload["dataset"]), int(workload["source_index"]))
        require(observed_pair == EXPECTED_PAIRS[rank], "workload order drift")
        raw_path = serving_run / "raw" / f"cache_on-rank-{rank}.json"
        raw = json.loads(raw_path.read_text())
        raw_hashes[raw_path.name] = sha256_file(raw_path)
        require(raw.get("status") == "completed", f"serving rank {rank} incomplete")
        require(raw.get("workload", {}).get("workload_id") == workload["workload_id"],
                f"serving workload mismatch at rank {rank}")
        measured = raw.get("measured")
        require(isinstance(measured, dict), f"missing measured result at rank {rank}")
        prediction = measured.get("prediction")
        require(isinstance(prediction, str) and prediction, f"empty prediction at rank {rank}")
        output_ids = tokenizer.encode(prediction, add_special_tokens=False)
        require(bool(output_ids), f"empty output tokenization at rank {rank}")
        input_ids = [
            int(value)
            for value in workload["document_token_ids"] + workload["query_token_ids"]
        ]
        requests.append(
            {
                "workload_id": workload["workload_id"],
                "dataset": workload["dataset"],
                "source_index": int(workload["source_index"]),
                "input_tokens": input_ids,
                "output_tokens": [int(value) for value in output_ids],
                "input_token_count": len(input_ids),
                "output_token_count": len(output_ids),
                "input_sha256": hashlib.sha256(canonical_bytes(input_ids)).hexdigest(),
                "output_sha256": hashlib.sha256(canonical_bytes(output_ids)).hexdigest(),
            }
        )
    followup_ids = tokenizer.encode(
        "\nFollow-up: repeat only the same short answer.\nAnswer:",
        add_special_tokens=False,
    )
    require(bool(followup_ids), "empty synthetic follow-up tokenization")
    return requests, [int(value) for value in followup_ids], {
        "data_sha256": DATA_SHA256,
        "serving_raw_sha256": raw_hashes,
        "loader_sha256": sha256_file(loader_path),
    }


def build_multiturn_events(
    base_requests: list[dict[str, Any]],
    schedule: list[int],
    followup_ids: list[int],
) -> list[dict[str, Any]]:
    """Extend each frozen workload into a deterministic synthetic session.

    Marconi's artifact models chat sessions in which the next input contains
    the previous input and output.  A repeated identical single-turn prompt is
    therefore not a valid Marconi state-reuse trace.  This construction keeps
    the eight real document/question prefixes and measured answers, and adds a
    fixed, disclosed follow-up suffix after each turn.
    """
    current = [list(request["input_tokens"]) for request in base_requests]
    turns = [0] * len(base_requests)
    events: list[dict[str, Any]] = []
    for event_id, workload_index in enumerate(schedule):
        request = base_requests[workload_index]
        input_ids = list(current[workload_index])
        output_ids = list(request["output_tokens"])
        events.append(
            {
                "event_id": event_id,
                "workload_index": workload_index,
                "workload_id": request["workload_id"],
                "turn_index": turns[workload_index],
                "input_tokens": input_ids,
                "output_tokens": output_ids,
                "input_token_count": len(input_ids),
                "output_token_count": len(output_ids),
            }
        )
        current[workload_index] = input_ids + output_ids + followup_ids
        turns[workload_index] += 1
    require(tuple(turns) == REQUEST_COUNTS, "multiturn count drift")
    return events


def import_marconi(repo: Path):
    import subprocess

    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    require(commit == MARCONI_COMMIT, "Marconi commit drift")
    sys.path.insert(0, str(repo))
    import radix_cache_hybrid
    import radix_cache_vllm

    return radix_cache_hybrid, radix_cache_vllm, {
        "commit": commit,
        "radix_cache_hybrid_sha256": sha256_file(repo / "radix_cache_hybrid.py"),
        "radix_cache_vllm_sha256": sha256_file(repo / "radix_cache_vllm.py"),
        "config_tuner_sha256": sha256_file(repo / "config_tuner.py"),
        "utils_sha256": sha256_file(repo / "utils.py"),
    }


def run_policy(
    *,
    policy: str,
    capacity_bytes: int,
    events: list[dict[str, Any]],
    hybrid_module,
    vllm_module,
) -> dict[str, Any]:
    common = dict(capacity_bytes=capacity_bytes, use_logical_ts=True, **NATIVE_GEOMETRY)
    if policy == "vllm_plus":
        tree = vllm_module.RadixCache(block_size=32, **common)
    elif policy == "sglang_plus":
        tree = hybrid_module.RadixCache(
            evict_policy_version=1, eff_weight=0.0, bootstrap_multiplier=2, **common
        )
    elif policy == "marconi":
        tree = hybrid_module.RadixCache(
            evict_policy_version=2, eff_weight=0.0, bootstrap_multiplier=2, **common
        )
    else:
        raise TraceError(f"unknown policy {policy}")

    started = time.perf_counter()
    capacity_closure_count = 0
    for request_id, event in enumerate(events):
        input_ids = event["input_tokens"]
        output_ids = event["output_tokens"]
        tree.match_prefix(input_ids)
        all_tokens = input_ids + output_ids
        if policy == "vllm_plus":
            tree.insert(token_ids=all_tokens)
        else:
            tree.insert(
                token_ids=all_tokens,
                state_at_leaf=request_id,
                state_at_branchoff=request_id,
            )
        # The artifact estimates the required bytes before insertion and can
        # under-evict for long requests.  Close the same published eviction
        # policy against the exact post-insert tree size instead of accepting
        # an over-budget simulator state.
        observed_bytes = int(tree.get_tree_size())
        if observed_bytes > capacity_bytes:
            tree.evict(bytes_to_remove=observed_bytes - capacity_bytes)
            capacity_closure_count += 1
            require(
                int(tree.get_tree_size()) <= capacity_bytes,
                "post-insert capacity closure failed",
            )
    elapsed = time.perf_counter() - started
    request_hit, token_hit, mamba_flops, attention_flops, mlp_flops = (
        tree.get_cache_stats(verbose=False)
    )
    history = [
        {"cache_hit": bool(row[0]), "input_tokens": int(row[1]), "tokens_reused": int(row[2])}
        for row in tree.request_history
    ]
    require(len(history) == len(events), "request history length drift")
    total_input = sum(row["input_tokens"] for row in history)
    total_reused = sum(row["tokens_reused"] for row in history)
    require(total_input > 0, "empty trace")
    require(abs(token_hit - total_reused / total_input) < 1e-12, "hit-rate replay drift")
    final_cache_bytes = int(tree.get_tree_size())
    require(final_cache_bytes <= capacity_bytes, "simulator cache exceeds budget")
    return {
        "policy": policy,
        "capacity_bytes": capacity_bytes,
        "request_count": len(history),
        "request_hit_rate": request_hit,
        "token_hit_rate": token_hit,
        "total_input_tokens": total_input,
        "total_reused_tokens": total_reused,
        "simulator_wall_seconds_context_only": elapsed,
        "post_insert_capacity_closure_count": capacity_closure_count,
        "predicted_flops_saved_native_geometry": {
            "mamba": int(mamba_flops),
            "attention": int(attention_flops),
            "mlp": int(mlp_flops),
            "total": int(mamba_flops + attention_flops + mlp_flops),
        },
        "final_cache_bytes": final_cache_bytes,
        "request_history_sha256": hashlib.sha256(canonical_bytes(history)).hexdigest(),
        "marconi_eff_weight_history": (
            list(getattr(tree, "eff_weight_history", [])) if policy == "marconi" else None
        ),
    }


def self_test() -> None:
    schedule = smooth_weighted_schedule(REQUEST_COUNTS)
    require(len(schedule) == 128, "schedule length")
    require(schedule[:8] == [0, 1, 2, 3, 4, 0, 5, 1], "schedule prefix drift")
    require(canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}\n', "canonical JSON")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--serving-run", type=Path)
    parser.add_argument("--loader", type=Path)
    parser.add_argument("--marconi-repo", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--debug-only", action="store_true")
    parser.add_argument("--formal-prereg", type=Path)
    parser.add_argument("--expected-prereg-sha256")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("SELF_TEST_PASS")
        return
    require(
        args.debug_only != (args.formal_prereg is not None),
        "select exactly one of --debug-only or --formal-prereg",
    )
    formal = args.formal_prereg is not None
    if formal:
        require(
            isinstance(args.expected_prereg_sha256, str)
            and len(args.expected_prereg_sha256) == 64,
            "formal run requires expected preregistration SHA-256",
        )
        require(
            sha256_file(args.formal_prereg) == args.expected_prereg_sha256,
            "preregistration byte hash drift",
        )
    required = (args.model, args.data, args.serving_run, args.loader, args.marconi_repo, args.output_dir)
    require(all(value is not None for value in required), "missing required path")
    self_test()

    requests, followup_ids, source = collect_requests(
        model=args.model,
        data=args.data,
        serving_run=args.serving_run,
        loader_path=args.loader,
    )
    schedule = smooth_weighted_schedule(REQUEST_COUNTS)
    events = build_multiturn_events(requests, schedule, followup_ids)
    hybrid, vllm, marconi_source = import_marconi(args.marconi_repo)
    observed_prereg = {
        "schema": PREREG_SCHEMA,
        "runner_sha256": sha256_file(Path(__file__)),
        "data_sha256": DATA_SHA256,
        "loader_sha256": source["loader_sha256"],
        "serving_raw_sha256": source["serving_raw_sha256"],
        "marconi_source": marconi_source,
        "request_counts": list(REQUEST_COUNTS),
        "capacities_bytes": list(CAPACITIES_BYTES),
        "native_marconi_geometry": NATIVE_GEOMETRY,
        "schedule_sha256": hashlib.sha256(canonical_bytes(schedule)).hexdigest(),
        "followup_sha256": hashlib.sha256(canonical_bytes(followup_ids)).hexdigest(),
        "event_inputs_sha256": hashlib.sha256(canonical_bytes(events)).hexdigest(),
        "claim_boundary": "policy_simulator_only_not_qwen_serving_or_longbench_quality",
    }
    if formal:
        prereg = json.loads(args.formal_prereg.read_text())
        require(prereg == observed_prereg, "formal preregistration content drift")
    schema = FORMAL_SCHEMA if formal else DEBUG_SCHEMA
    trace = {
        "schema": schema,
        "debug_only": not formal,
        "formal_evidence_eligible": formal,
        "preregistration_sha256": (
            args.expected_prereg_sha256 if formal else None
        ),
        "request_counts": list(REQUEST_COUNTS),
        "schedule": schedule,
        "base_requests": requests,
        "followup_tokens": followup_ids,
        "followup_sha256": hashlib.sha256(canonical_bytes(followup_ids)).hexdigest(),
        "events": events,
        "source": source,
        "scope": {
            "claimable": (
                "policy-simulator token reuse on a disclosed synthetic multi-turn "
                "extension of eight frozen LongBench workloads"
            ),
            "not_claimable": [
                "Qwen3.5 end-to-end throughput",
                "LongBench quality",
                "same-stack serving latency",
                "CoMem-vs-Marconi direct comparison",
            ],
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output_dir / "trace.json", trace)

    rows: list[dict[str, Any]] = []
    for capacity_bytes in CAPACITIES_BYTES:
        for policy in ("vllm_plus", "sglang_plus", "marconi"):
            rows.append(
                run_policy(
                    policy=policy,
                    capacity_bytes=capacity_bytes,
                    events=events,
                    hybrid_module=hybrid,
                    vllm_module=vllm,
                )
            )
    summary = {
        "schema": schema,
        "debug_only": not formal,
        "formal_evidence_eligible": formal,
        "preregistration_sha256": (
            args.expected_prereg_sha256 if formal else None
        ),
        "status": "completed",
        "trace_sha256": sha256_file(args.output_dir / "trace.json"),
        "native_marconi_geometry": NATIVE_GEOMETRY,
        "marconi_source": marconi_source,
        "rows": rows,
        "interpretation_boundary": (
            "The official simulator uses its native Attention--Mamba2 geometry. "
            "A deterministic post-insert call to the simulator's own eviction policy "
            "closes its approximate pre-insert accounting to the declared byte budget. "
            "Only token-hit rates are portable to this frozen trace; simulator wall time "
            "and predicted FLOPs are context-only and are not model serving measurements."
        ),
    }
    atomic_json(args.output_dir / "summary.json", summary)
    digest_lines = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            digest_lines.append(f"{sha256_file(path)}  {path.name}\n")
    (args.output_dir / "SHA256SUMS").write_text("".join(digest_lines))
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
