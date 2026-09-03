#!/usr/bin/env python3
"""Affected-only Prefix/HYPIC retained-document byte run for RW-D5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from run_hypic_same_protocol import (
    DATA_SHA256,
    HYPIC_COMMIT,
    SEAM_TOKENS,
    _server_configuration_receipt,
    cached_tokens_from_completion,
    canonical_json_sha256 as formal_canonical_json_sha256,
    load_segmented_workload,
    request_prompts,
)
from run_related_work_serving_baseline import (
    _require,
    answer_f1,
    atomic_json,
    http_get_text,
    sha256_file,
    stream_completion,
)


MODES = ("prefix_cache", "transition_rope_recompute")
TARGET_SCHEMA = "forkaudit-hypic-retained-state-target-v2"
SERVER_SCHEMA = "hypic-rwd5-server-launch-receipt-v2"
PREREG_SCHEMA = "hypic-rwd5-retained-state-preregistration-v2"
WORKER_SCHEMA = "forkaudit-hypic-scheduler-worker-v2"


def rwd5_canonical_json_bytes(value: Any) -> bytes:
    """RW-D5 authority canonicalization; deliberately distinct from legacy formal helper semantics."""
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def rwd5_canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(rwd5_canonical_json_bytes(value)).hexdigest()


def _wait_for_json(path: Path, timeout: float = 90.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.is_file():
            try:
                return json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                pass
        time.sleep(0.05)
    raise TimeoutError(f"timed out waiting for receipt: {path}")


def _post_flush(base_url: str, timeout: float) -> str:
    deadline = time.monotonic() + min(timeout, 90.0)
    last_body = ""
    while time.monotonic() < deadline:
        request = urllib.request.Request(
            base_url.rstrip("/") + "/flush_cache", data=b"", method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=min(timeout, 30.0)) as response:
                body = response.read().decode("utf-8")
                _require(
                    response.status == 200 and "Cache flushed" in body,
                    "flush cache response",
                )
                return body
        except urllib.error.HTTPError as error:
            last_body = error.read().decode("utf-8", errors="replace")
            if error.code != 400:
                raise
        time.sleep(0.1)
    raise TimeoutError(f"flush cache did not reach idle state: {last_body}")


def _target(
    workload: dict[str, Any], mode: str, rank: int, authority: dict[str, Any]
) -> dict[str, Any]:
    value = {
        "schema": TARGET_SCHEMA,
        "snapshot_id": f"{mode}-rank-{rank}",
        "official_commit": HYPIC_COMMIT,
        "mode": mode,
        "rank": rank,
        "workload_id": workload["workload_id"],
        "document_token_ids": workload["document_token_ids"],
        "document_token_sha256": workload["document_token_sha256"],
        "seam_tokens": SEAM_TOKENS if mode == "transition_rope_recompute" else 0,
        "authority": authority,
        "workload_binding": {
            "workload_id": workload["workload_id"],
            "dataset": workload["dataset"],
            "source_index": workload["source_index"],
            "document_tokens": len(workload["document_token_ids"]),
            "query_tokens": len(workload["query_token_ids"]),
            "prompt_token_sha256": workload["prompt_token_sha256"],
            "document_token_sha256": workload["document_token_sha256"],
            "segment_offsets": workload["segment_offsets"],
            "token_identity_verified": True,
        },
    }
    if mode == "transition_rope_recompute":
        value["segment_token_ids"] = [
            workload["direct_token_ids"][start:end]
            for start, end in workload["segment_offsets"][:2]
        ]
        _require(
            value["segment_token_ids"][0] + value["segment_token_ids"][1]
            == workload["document_token_ids"],
            "HYPIC target segment coverage",
        )
    return value


def _validate_live_receipt(
    receipt: dict[str, Any], target: dict[str, Any], measured_cached_tokens: int | None = None
) -> None:
    _require(
        receipt.get("schema") == "forkaudit-hypic-retained-state-receipt-v2"
        and receipt.get("status") == "owned_state_snapshot_complete"
        and receipt.get("official_commit") == HYPIC_COMMIT,
        "store receipt authority",
    )
    observed = receipt["target"]
    for key in ("snapshot_id", "mode", "rank", "workload_id", "document_token_sha256"):
        _require(observed.get(key) == target.get(key), f"store target binding: {key}")
    _require(receipt["authority"]["bindings"]["target_sha256"], "receipt target SHA")
    selection = receipt["selection"]
    _require(
        selection["owned_document_token_sha256"] == target["document_token_sha256"]
        if target["mode"] == "transition_rope_recompute"
        else selection["owned_document_tokens"] <= len(target["document_token_ids"]),
        "owned document boundary",
    )
    payload = receipt["tensor_payload"]["union"]
    _require(
        isinstance(payload["unique_overlap_aware_bytes"], int)
        and payload["unique_overlap_aware_bytes"] > 0
        and payload["unique_overlap_aware_bytes"] <= payload["naive_range_bytes"],
        "payload byte sum",
    )
    _require(receipt["metadata"]["excluded_from_store_mib"] is True, "metadata boundary")
    if measured_cached_tokens is not None:
        _require(
            selection["expected_measured_cached_tokens"] == measured_cached_tokens,
            "receipt/cache-hit coverage",
        )


def client_stage(args: argparse.Namespace) -> None:
    _require(args.mode in MODES, "affected-only mode")
    _require(0 <= args.rank < 8 and args.world_size == 8, "rank contract")
    _require(sha256_file(args.data) == DATA_SHA256, "data digest drift")
    _require(args.server_receipt.is_file(), "server receipt")
    server_receipt = json.loads(args.server_receipt.read_text())
    _require(
        server_receipt.get("schema") == SERVER_SCHEMA
        and server_receipt.get("mode") == args.mode
        and server_receipt.get("rank") == args.rank
        and server_receipt.get("official_commit") == HYPIC_COMMIT,
        "server launch binding",
    )
    _require(args.preregistration.is_file(), "client preregistration")
    prereg = json.loads(args.preregistration.read_text())
    _require(prereg.get("schema") == PREREG_SCHEMA, "client preregistration schema")
    _require(args.freeze_manifest.is_file(), "client external manifest")
    _require(
        sha256_file(args.freeze_manifest) == args.expected_freeze_manifest_sha256,
        "client external manifest SHA",
    )
    authority = dict(server_receipt["authority"])
    authority.update(
        {
            "server_launch_receipt_sha256": sha256_file(args.server_receipt),
            "scheduler_worker_receipt_sha256": server_receipt["scheduler_worker"]["receipt_sha256"],
            "server_configuration_sha256": server_receipt["server_configuration_sha256"],
        }
    )
    workload = load_segmented_workload(args.model, args.data, args.rank)
    base = args.base_url.rstrip("/")
    server_info = json.loads(http_get_text(base + "/server_info", timeout=30.0))
    server_configuration = _server_configuration_receipt(
        server_info, args.mode, 1, model_path=args.model, rank=args.rank
    )
    _require(
        formal_canonical_json_sha256(server_info) == server_receipt["server_info_sha256"],
        "live server info drift",
    )
    prompts = request_prompts(workload, args.mode)
    common = {
        "model": args.served_model_name,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }

    # Same prefix-disjoint warm path as the completed HYPIC formal run.
    warm_prime = stream_completion(
        base + "/v1/completions",
        {**common, "prompt": prompts["warm_prime"], "max_tokens": 1},
        timeout=args.timeout,
        require_text=False,
    )
    warmup = stream_completion(
        base + "/v1/completions",
        {**common, "prompt": prompts["warm_measured"], "max_tokens": 1},
        timeout=args.timeout,
        require_text=False,
    )
    warm_cached = cached_tokens_from_completion(warmup)
    _require(isinstance(warm_cached, int) and warm_cached > 0, "warm cache path")

    target = _target(workload, args.mode, args.rank, authority)
    _require(not args.target_file.exists(), "target file must be absent before formal prime")
    _require(not args.store_receipt.exists(), "store receipt must be absent before formal prime")
    atomic_json(args.target_file, target)
    target_sha = sha256_file(args.target_file)

    prime = stream_completion(
        base + "/v1/completions",
        {**common, "prompt": prompts["formal_prime"], "max_tokens": 1},
        timeout=args.timeout,
        require_text=False,
    )
    receipt = _wait_for_json(args.store_receipt)
    _validate_live_receipt(receipt, target)
    pre_measured_receipt_sha = sha256_file(args.store_receipt)

    measured = stream_completion(
        base + "/v1/completions",
        {**common, "prompt": prompts["formal_measured"], "max_tokens": 32},
        timeout=args.timeout,
    )
    measured["f1"] = max(
        answer_f1(measured["prediction"], reference) for reference in workload["references"]
    )
    cached_tokens = cached_tokens_from_completion(measured)
    _require(isinstance(cached_tokens, int) and cached_tokens > 0, "formal cache hit")
    _validate_live_receipt(receipt, target, cached_tokens)
    _require(
        sha256_file(args.store_receipt) == pre_measured_receipt_sha,
        "pre-measured receipt mutated after measured query",
    )

    flush_response = _post_flush(base, args.timeout)
    terminal = _wait_for_json(args.terminal_receipt)
    _require(
        terminal.get("schema") == "forkaudit-hypic-retained-state-terminal-v2"
        and terminal.get("passed") is True
        and terminal.get("prior_receipt_sha256") == pre_measured_receipt_sha,
        "terminal ownership receipt",
    )
    payload = {
        "schema": "forkaudit-hypic-retained-state-shard-v2",
        "status": "completed",
        "official_commit": HYPIC_COMMIT,
        "mode": args.mode,
        "rank": args.rank,
        "world_size": args.world_size,
        "server_configuration": server_configuration,
        "server_info_sha256": formal_canonical_json_sha256(server_info),
        "server_launch_receipt_sha256": sha256_file(args.server_receipt),
        "authority": receipt["authority"],
        "preregistration_sha256": sha256_file(args.preregistration),
        "freeze_manifest_sha256": sha256_file(args.freeze_manifest),
        "workload": {
            "workload_id": workload["workload_id"],
            "dataset": workload["dataset"],
            "source_index": workload["source_index"],
            "document_tokens": len(workload["document_token_ids"]),
            "query_tokens": len(workload["query_token_ids"]),
            "document_token_sha256": workload["document_token_sha256"],
            "prompt_token_sha256": workload["prompt_token_sha256"],
            "segment_offsets": workload["segment_offsets"],
            "token_identity_verified": True,
        },
        "target_sha256": target_sha,
        "target": target,
        "warm_prime": warm_prime,
        "warmup": warmup,
        "prime": prime,
        "measured": measured,
        "cache_observation": {
            "cached_tokens": cached_tokens,
            "authority": "openai-completion-usage.cached_tokens",
        },
        "store_receipt": {
            "path": str(args.store_receipt),
            "sha256": pre_measured_receipt_sha,
            "payload_bytes": receipt["tensor_payload"]["union"]["unique_overlap_aware_bytes"],
            "metadata_excluded": True,
            "captured_after_prime_before_measured": True,
        },
        "terminal_receipt": {
            "path": str(args.terminal_receipt),
            "sha256": sha256_file(args.terminal_receipt),
            "flush_response": flush_response,
        },
    }
    atomic_json(args.output, payload)


def _process_receipt(pid: int, expected_pythonpath: str, expected_uuid: str) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    _require(root.is_dir(), "server process live")
    cmdline = [part.decode() for part in (root / "cmdline").read_bytes().split(b"\0") if part]
    environment = {}
    wanted = {
        "CUDA_VISIBLE_DEVICES",
        "PYTHONPATH",
        "PIC_SEAM_SINK",
        "FORKAUDIT_RWD5_TARGET_PATH",
        "FORKAUDIT_RWD5_RECEIPT_DIR",
        "FORKAUDIT_RWD5_WORKER_RECEIPT_PATH",
        "FORKAUDIT_RWD5_SERVER_RECEIPT_PATH",
        "FORKAUDIT_RWD5_PREREGISTRATION_PATH",
        "FORKAUDIT_RWD5_FREEZE_MANIFEST_PATH",
        "FORKAUDIT_RWD5_FREEZE_MANIFEST_SHA256",
        "FORKAUDIT_RWD5_FRONTEND_PID",
        "FORKAUDIT_RWD5_MODE",
        "FORKAUDIT_RWD5_RANK",
    }
    for item in (root / "environ").read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        name = key.decode()
        if name in wanted:
            environment[name] = value.decode()
    _require(environment.get("CUDA_VISIBLE_DEVICES") == expected_uuid, "server GPU UUID")
    _require(environment.get("PYTHONPATH") == expected_pythonpath, "instrumented PYTHONPATH")
    _require(environment.get("PIC_SEAM_SINK") == "8", "seam environment")
    _require(
        environment.get("FORKAUDIT_RWD5_TARGET_PATH")
        and environment.get("FORKAUDIT_RWD5_RECEIPT_DIR"),
        "receipt environment",
    )
    return {
        "pid": pid,
        "ppid": int((root / "stat").read_text().rsplit(")", 1)[1].split()[1]),
        "cmdline": cmdline,
        "cmdline_sha256": rwd5_canonical_json_sha256(cmdline),
        "environment": environment,
        "environment_sha256": rwd5_canonical_json_sha256(environment),
    }


def server_receipt_stage(args: argparse.Namespace) -> None:
    import torch

    _require(
        subprocess.check_output(["git", "-C", str(args.official_repo), "rev-parse", "HEAD"], text=True).strip()
        == HYPIC_COMMIT,
        "official commit",
    )
    _require(
        not subprocess.check_output(
            ["git", "-C", str(args.official_repo), "status", "--porcelain", "--untracked-files=all"],
            text=True,
        ).strip(),
        "official worktree cleanliness",
    )
    _require(
        subprocess.check_output(["git", "-C", str(args.instrumented_repo), "rev-parse", "HEAD"], text=True).strip()
        == HYPIC_COMMIT,
        "instrumented base commit",
    )
    subprocess.check_call(
        ["git", "-C", str(args.instrumented_repo), "apply", "--reverse", "--check", str(args.patch)]
    )
    module = args.instrumented_repo / "python/sglang/srt/retained_state_receipt.py"
    _require(module.is_file() and sha256_file(module) == sha256_file(args.receipt_module), "receipt module closure")
    server_info = json.loads(http_get_text(args.base_url.rstrip("/") + "/server_info", timeout=30.0))
    configuration = _server_configuration_receipt(
        server_info, args.mode, 1, model_path=args.model, rank=args.rank
    )
    extra_expected = {"enable_int8_mamba_checkpoint": False, "page_size": 1}
    extra_observed = {key: server_info.get(key) for key in extra_expected}
    _require(extra_observed == extra_expected, "RW-D5 resolved server extras")
    configuration = {**configuration, "rwd5_expected": extra_expected, "rwd5_observed": extra_observed}
    process = _process_receipt(
        args.server_pid,
        f"{args.instrumented_repo}/python:{args.code_dir}",
        args.expected_gpu_uuid,
    )
    process_environment = process["environment"]
    _require(
        process_environment.get("FORKAUDIT_RWD5_WORKER_RECEIPT_PATH") == str(args.worker_receipt)
        and process_environment.get("FORKAUDIT_RWD5_SERVER_RECEIPT_PATH") == str(args.output)
        and process_environment.get("FORKAUDIT_RWD5_PREREGISTRATION_PATH") == str(args.preregistration)
        and process_environment.get("FORKAUDIT_RWD5_FREEZE_MANIFEST_PATH") == str(args.freeze_manifest)
        and process_environment.get("FORKAUDIT_RWD5_FREEZE_MANIFEST_SHA256") == args.expected_freeze_manifest_sha256
        and process_environment.get("FORKAUDIT_RWD5_MODE") == args.mode
        and process_environment.get("FORKAUDIT_RWD5_RANK") == str(args.rank)
        and process_environment.get("FORKAUDIT_RWD5_FRONTEND_PID") == str(args.server_pid),
        "server receipt environment bindings",
    )
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one visible GPU")
    props = torch.cuda.get_device_properties(0)
    _require(props.name == "NVIDIA H20-3e", "H20-3e")
    prereg = json.loads(args.preregistration.read_text())
    _require(prereg.get("schema") == PREREG_SCHEMA, "server preregistration")
    _require(sha256_file(args.freeze_manifest) == args.expected_freeze_manifest_sha256, "server frozen manifest")
    worker = json.loads(args.worker_receipt.read_text())
    _require(worker.get("schema") == WORKER_SCHEMA, "scheduler worker schema")
    _require(worker["mode"] == args.mode and worker["rank"] == args.rank, "scheduler worker cell")
    _require(worker["frontend_pid"] == args.server_pid, "scheduler/frontend PID binding")
    _require(args.server_pid in worker["process"]["ancestry_pids"], "scheduler child lineage")
    status = subprocess.check_output(
        ["git", "-C", str(args.instrumented_repo), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    ).splitlines()
    expected_status = sorted([
        " M python/sglang/srt/managers/scheduler.py",
        " M python/sglang/srt/mem_cache/common.py",
        "?? python/sglang/srt/retained_state_receipt.py",
    ])
    _require(sorted(status) == expected_status, "server exact overlay status")
    overlay_diff = subprocess.check_output(
        ["git", "-C", str(args.instrumented_repo), "diff", "--binary", "--no-ext-diff", "--full-index", "--",
         "python/sglang/srt/managers/scheduler.py", "python/sglang/srt/mem_cache/common.py"]
    )
    overlay_sha = hashlib.sha256(overlay_diff).hexdigest()
    _require(overlay_sha == prereg["instrumentation"]["overlay"]["canonical_diff_sha256"], "server overlay diff")
    authority = {
        "official_commit": HYPIC_COMMIT,
        "preregistration_sha256": sha256_file(args.preregistration),
        "freeze_manifest_sha256": args.expected_freeze_manifest_sha256,
        "official_source_ledger_sha256": prereg["official_source_ledger_sha256"],
        "environment_ledger_sha256": prereg["environment_ledger_sha256"],
        "data_sha256": prereg["data"]["sha256"],
        "model_weight_ledger_sha256": prereg["model"]["weight_ledger_raw_sha256"],
        "model_artifact_ledger_sha256": prereg["model"]["artifact_ledger_raw_sha256"],
        "model_config_sha256": prereg["model"]["config_sha256"],
        "storage_contract_sha256": prereg["model"]["storage_contract_sha256"],
        "overlay_diff_sha256": overlay_sha,
        "code_sha256": {key: row["sha256"] for key, row in sorted(prereg["code"].items())},
    }
    payload = {
        "schema": SERVER_SCHEMA,
        "official_commit": HYPIC_COMMIT,
        "official_worktree_clean": True,
        "instrumentation_only_overlay": True,
        "mode": args.mode,
        "rank": args.rank,
        "tp_size": 1,
        "data_sha256": sha256_file(args.data),
        "model_weight_ledger_sha256": sha256_file(args.model_weight_ledger),
        "model_artifact_ledger_sha256": sha256_file(args.model_artifact_ledger),
        "patch_sha256": sha256_file(args.patch),
        "receipt_module_sha256": sha256_file(args.receipt_module),
        "client_sha256": sha256_file(Path(__file__)),
        "static_preregistration_sha256": sha256_file(args.preregistration),
        "server_info_sha256": formal_canonical_json_sha256(server_info),
        "server_configuration": configuration,
        "server_configuration_sha256": rwd5_canonical_json_sha256(configuration),
        "server_process": process,
        "frontend_process": process,
        "scheduler_worker": {"receipt_sha256": sha256_file(args.worker_receipt), "identity": worker},
        "authority": authority,
        "instrumented_overlay": {"porcelain_v1": sorted(status), "canonical_diff_sha256": overlay_sha},
        "hardware": {
            "gpu_name": props.name,
            "torch_visible_memory_mib": int(props.total_memory // (1024 * 1024)),
            "gpu_uuid": args.expected_gpu_uuid,
        },
    }
    atomic_json(args.output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("client", "server_receipt"), required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model-name", default="qwen35-hypic")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--server-receipt", type=Path)
    parser.add_argument("--target-file", type=Path)
    parser.add_argument("--store-receipt", type=Path)
    parser.add_argument("--terminal-receipt", type=Path)
    parser.add_argument("--official-repo", type=Path)
    parser.add_argument("--instrumented-repo", type=Path)
    parser.add_argument("--patch", type=Path)
    parser.add_argument("--receipt-module", type=Path)
    parser.add_argument("--code-dir", type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--model-weight-ledger", type=Path)
    parser.add_argument("--model-artifact-ledger", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--worker-receipt", type=Path)
    parser.add_argument("--freeze-manifest", type=Path)
    parser.add_argument("--expected-freeze-manifest-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "client":
        for name in (
            "server_receipt", "target_file", "store_receipt", "terminal_receipt",
            "preregistration", "freeze_manifest", "expected_freeze_manifest_sha256",
        ):
            _require(getattr(args, name) is not None, f"client {name}")
        client_stage(args)
    else:
        for name in (
            "official_repo",
            "instrumented_repo",
            "patch",
            "receipt_module",
            "code_dir",
            "server_pid",
            "expected_gpu_uuid",
            "model_weight_ledger",
            "model_artifact_ledger",
            "preregistration",
            "worker_receipt",
            "freeze_manifest",
            "expected_freeze_manifest_sha256",
        ):
            _require(getattr(args, name) is not None, f"server receipt {name}")
        server_receipt_stage(args)


if __name__ == "__main__":
    main()
