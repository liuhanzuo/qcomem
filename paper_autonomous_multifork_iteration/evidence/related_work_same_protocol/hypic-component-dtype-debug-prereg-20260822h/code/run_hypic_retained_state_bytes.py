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
SERVER_INFO_RECEIPT_TIMEOUT_SECONDS = 120.0


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


def wait_for_server_info(
    base_url: str,
    output: Path,
    *,
    mode: str,
    rank: int,
    server_pid: int,
    total_timeout: float = 300.0,
    single_timeout: float = 3.0,
    poll_interval: float = 1.0,
    fetch=http_get_text,
    monotonic=time.monotonic,
    sleeper=time.sleep,
) -> dict[str, Any]:
    """Poll the real readiness endpoint and retain every attempt as evidence."""
    _require(total_timeout > 30.0, "server_info total timeout exceeds legacy one-shot window")
    _require(0.0 < single_timeout <= 10.0, "short server_info attempt timeout")
    _require(0.0 <= poll_interval <= 10.0, "bounded server_info poll interval")
    _require(mode in MODES and 0 <= rank < 8, "server_info readiness cell")
    _require(server_pid > 1, "server_info readiness frontend PID")
    endpoint = base_url.rstrip("/") + "/server_info"
    started = monotonic()
    deadline = started + total_timeout
    attempts: list[dict[str, Any]] = []
    while monotonic() < deadline:
        attempt_started = monotonic()
        attempt_timeout = min(single_timeout, max(0.001, deadline - attempt_started))
        row: dict[str, Any] = {
            "attempt": len(attempts) + 1,
            "elapsed_before_seconds": round(attempt_started - started, 6),
            "request_timeout_seconds": round(attempt_timeout, 6),
        }
        try:
            value = json.loads(fetch(endpoint, timeout=attempt_timeout))
            _require(isinstance(value, dict) and value, "nonempty server_info object")
            row.update({
                "outcome": "ready",
                "response_sha256": formal_canonical_json_sha256(value),
                "elapsed_after_seconds": round(monotonic() - started, 6),
            })
            attempts.append(row)
            payload = {
                "schema": "hypic-rwd5-server-info-readiness-v1",
                "status": "ready",
                "endpoint": endpoint,
                "mode": mode,
                "rank": rank,
                "server_pid": server_pid,
                "total_timeout_seconds": total_timeout,
                "single_timeout_seconds": single_timeout,
                "poll_interval_seconds": poll_interval,
                "attempt_count": len(attempts),
                "elapsed_seconds": round(monotonic() - started, 6),
                "attempts": attempts,
                "server_info_sha256": row["response_sha256"],
            }
            atomic_json(output, payload)
            return payload
        except Exception as error:
            row.update({
                "outcome": "not_ready",
                "error_type": type(error).__name__,
                "error": str(error),
                "elapsed_after_seconds": round(monotonic() - started, 6),
            })
            attempts.append(row)
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        sleeper(min(poll_interval, remaining))
    payload = {
        "schema": "hypic-rwd5-server-info-readiness-v1",
        "status": "failed_timeout",
        "endpoint": endpoint,
        "mode": mode,
        "rank": rank,
        "server_pid": server_pid,
        "total_timeout_seconds": total_timeout,
        "single_timeout_seconds": single_timeout,
        "poll_interval_seconds": poll_interval,
        "attempt_count": len(attempts),
        "elapsed_seconds": round(monotonic() - started, 6),
        "attempts": attempts,
    }
    atomic_json(output, payload)
    raise RuntimeError(f"server_info readiness deadline expired after {len(attempts)} attempts")


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


def _debug_c_contiguous(shape: list[int], stride: list[int]) -> bool:
    expected = 1
    for size, observed in zip(reversed(shape), reversed(stride)):
        if size != 1 and observed != expected:
            return False
        expected *= size
    return True


def _debug_expected_recurrent_layers(model: Path) -> int:
    config = json.loads((model / "config.json").read_text())
    text = config.get("text_config", config)
    count = int(text["num_hidden_layers"])
    layer_types = text.get("layer_types")
    if layer_types is None:
        interval = int(text["full_attention_interval"])
        _require(interval > 0, "debug model full-attention interval")
        layer_types = [
            "attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(count)
        ]
    _require(isinstance(layer_types, list) and len(layer_types) == count, "debug layer types")
    recurrent = sum(str(value) == "linear_attention" for value in layer_types)
    _require(recurrent > 0, "debug recurrent layer count")
    return recurrent


def validate_dtype_debug_receipt(
    debug: dict[str, Any], *, mode: str, model: Path
) -> dict[str, Any]:
    """Fail-closed consumer validation of a debug-only live pool inventory."""
    expected_names = {
        "prefix_cache": ["conv[0]", "temporal"],
        "transition_rope_recompute": [
            "conv[0]", "temporal", "transition", "conv_tails[0]",
        ],
    }
    _require(mode in expected_names, "debug validation mode")
    _require(
        set(debug) == {
            "schema", "status", "official_commit", "mode", "tree_cache_class",
            "mamba_pool_class", "mamba_allocator_size", "mamba_capacity_axis",
            "runtime_environment", "components", "formal_receipt_emitted",
        },
        "exact debug receipt keys",
    )
    _require(
        debug["schema"] == "hypic-rwd5-component-dtype-debug-v1"
        and debug["status"] == "debug_only_not_formal_evidence"
        and debug["official_commit"] == HYPIC_COMMIT
        and debug["mode"] == mode
        and debug["formal_receipt_emitted"] is False,
        "debug identity/nonformal status",
    )
    expected_cache = "MambaRadixCache" if mode == "prefix_cache" else "PICache"
    _require(debug["tree_cache_class"] == expected_cache, "debug tree-cache class")
    _require(debug["mamba_pool_class"] == "MambaPool", "debug MambaPool class")
    _require(
        debug["runtime_environment"] == {
            "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
            "SGLANG_MAMBA_SSM_DTYPE": "float32",
        },
        "debug exact dtype environment",
    )
    allocator_size = int(debug["mamba_allocator_size"])
    capacity_axis = int(debug["mamba_capacity_axis"])
    _require(allocator_size > 0 and capacity_axis == allocator_size + 1, "debug slot capacity")
    recurrent_layers = _debug_expected_recurrent_layers(model)
    components = debug["components"]
    _require(
        isinstance(components, dict) and list(components) == expected_names[mode],
        "debug exact ordered component topology",
    )
    expected_dtype = {
        "conv[0]": ("torch.bfloat16", 2),
        "temporal": ("torch.float32", 4),
        "transition": ("torch.float32", 4),
        "conv_tails[0]": ("torch.bfloat16", 2),
    }
    for name in expected_names[mode]:
        row = components[name]
        _require(
            isinstance(row, dict)
            and set(row) == {"dtype", "element_size", "shape", "stride", "device", "c_contiguous"},
            f"debug exact component fields {name}",
        )
        dtype, element = expected_dtype[name]
        _require(row["dtype"] == dtype and int(row["element_size"]) == element, f"debug dtype {name}")
        shape = row["shape"]
        stride = row["stride"]
        expected_rank = 4 if name in {"conv[0]", "conv_tails[0]"} else 5
        _require(
            isinstance(shape, list) and isinstance(stride, list)
            and len(shape) == len(stride) == expected_rank
            and all(isinstance(value, int) and value > 0 for value in shape)
            and all(isinstance(value, int) and value > 0 for value in stride),
            f"debug shape/stride {name}",
        )
        _require(
            shape[0] == recurrent_layers and shape[1] == capacity_axis,
            f"debug recurrent layer/slot axes {name}",
        )
        _require(
            row["c_contiguous"] is True and _debug_c_contiguous(shape, stride),
            f"debug C-contiguous layout {name}",
        )
        _require(row["device"] == "cuda:0", f"debug single visible GPU device {name}")
    if mode == "transition_rope_recompute":
        _require(
            components["conv_tails[0]"]["shape"] == components["conv[0]"]["shape"]
            and components["conv_tails[0]"]["stride"] == components["conv[0]"]["stride"],
            "debug conv-tail mirrors conv layout",
        )
        temporal_shape = components["temporal"]["shape"]
        transition_shape = components["transition"]["shape"]
        _require(
            transition_shape[:3] == temporal_shape[:3]
            and transition_shape[3] == temporal_shape[3]
            and transition_shape[4] == temporal_shape[3],
            "debug transition follows temporal H/K axes",
        )
    return {
        "schema": "hypic-rwd5-component-dtype-debug-validation-v1",
        "status": "passed_exact_live_component_contract",
        "official_commit": HYPIC_COMMIT,
        "mode": mode,
        "paper_evidence": False,
        "expected_recurrent_layers": recurrent_layers,
        "mamba_capacity_axis": capacity_axis,
        "components": components,
    }


def dtype_debug_stage(args: argparse.Namespace) -> None:
    """Prime one frozen workload and collect only the read-only pool inventory."""
    _require(args.mode in MODES and args.rank == 0, "single-GPU affected dtype debug cell")
    _require(sha256_file(args.data) == DATA_SHA256, "debug data digest drift")
    _require(not args.target_file.exists(), "debug target must be absent")
    _require(not args.dtype_debug_receipt.exists(), "debug receipt must be absent")
    workload = load_segmented_workload(args.model, args.data, args.rank)
    target = _target(workload, args.mode, args.rank, {})
    target["snapshot_id"] = f"dtype-debug-{args.mode}-rank-0"
    atomic_json(args.target_file, target)
    prompt = request_prompts(workload, args.mode)["formal_prime"]
    prime = stream_completion(
        args.base_url.rstrip("/") + "/v1/completions",
        {
            "model": args.served_model_name,
            "temperature": 0.0,
            "seed": args.seed,
            "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": True},
            "prompt": prompt,
            "max_tokens": 1,
        },
        timeout=args.timeout,
        require_text=False,
    )
    debug = _wait_for_json(args.dtype_debug_receipt, timeout=90.0)
    validation = validate_dtype_debug_receipt(debug, mode=args.mode, model=args.model)
    flush_response = _post_flush(args.base_url, args.timeout)
    atomic_json(args.output, {
        "schema": "hypic-rwd5-component-dtype-debug-run-v1",
        "status": "completed_debug_only_not_formal_evidence",
        "official_commit": HYPIC_COMMIT,
        "mode": args.mode,
        "rank": 0,
        "workload_id": workload["workload_id"],
        "target_sha256": sha256_file(args.target_file),
        "debug_receipt_sha256": sha256_file(args.dtype_debug_receipt),
        "validation": validation,
        "prime": prime,
        "flush_response": flush_response,
        "raw_formal_receipt_emitted": False,
        "store_formal_receipt_emitted": False,
        "paper_evidence": False,
    })


def dtype_debug_validate_stage(args: argparse.Namespace) -> None:
    debug = json.loads(args.dtype_debug_receipt.read_text())
    validation = validate_dtype_debug_receipt(debug, mode=args.mode, model=args.model)
    validation["debug_receipt_sha256"] = sha256_file(args.dtype_debug_receipt)
    atomic_json(args.output, validation)


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
    server_info = json.loads(
        http_get_text(base + "/server_info", timeout=SERVER_INFO_RECEIPT_TIMEOUT_SECONDS)
    )
    server_configuration = _server_configuration_receipt(
        server_info, args.mode, 1, model_path=args.model, rank=args.rank
    )
    _require(
        formal_canonical_json_sha256(server_info) == server_receipt["server_info_sha256"],
        "live server info drift",
    )
    _require(
        server_receipt["server_info_readiness"]["identity"]["server_info_sha256"]
        == server_receipt["server_info_sha256"],
        "client server_info readiness binding",
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
        "SGLANG_MAMBA_CONV_DTYPE",
        "SGLANG_MAMBA_SSM_DTYPE",
        "FORKAUDIT_RWD5_DTYPE_DEBUG_PATH",
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
    _require(environment.get("SGLANG_MAMBA_CONV_DTYPE") == "bfloat16", "conv dtype environment")
    _require(environment.get("SGLANG_MAMBA_SSM_DTYPE") == "float32", "temporal dtype environment")
    _require("FORKAUDIT_RWD5_DTYPE_DEBUG_PATH" not in environment, "formal server excludes dtype debug mode")
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
    server_info = json.loads(
        http_get_text(
            args.base_url.rstrip("/") + "/server_info",
            timeout=SERVER_INFO_RECEIPT_TIMEOUT_SECONDS,
        )
    )
    readiness = json.loads(args.server_info_readiness.read_text())
    expected_base_url = f"http://127.0.0.1:{33400 + args.rank}"
    expected_endpoint = expected_base_url + "/server_info"
    _require(
        readiness.get("schema") == "hypic-rwd5-server-info-readiness-v1"
        and readiness.get("status") == "ready"
        and readiness.get("attempt_count") == len(readiness.get("attempts", []))
        and readiness["attempts"][-1].get("outcome") == "ready"
        and all(row.get("outcome") == "not_ready" for row in readiness["attempts"][:-1])
        and readiness.get("mode") == args.mode
        and readiness.get("rank") == args.rank
        and readiness.get("server_pid") == args.server_pid
        and readiness.get("endpoint") == expected_endpoint
        and args.base_url.rstrip("/") == expected_base_url
        and readiness.get("total_timeout_seconds") == 300.0
        and readiness.get("single_timeout_seconds") == 3.0
        and readiness.get("poll_interval_seconds") == 1.0
        and [row.get("attempt") for row in readiness["attempts"]]
        == list(range(1, len(readiness["attempts"]) + 1))
        and readiness["attempts"][-1].get("response_sha256")
        == readiness.get("server_info_sha256")
        and readiness.get("server_info_sha256") == formal_canonical_json_sha256(server_info),
        "server_info readiness evidence binding",
    )
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
        "base_url": expected_base_url,
        "server_info_endpoint": expected_endpoint,
        "server_info_readiness": {
            "sha256": sha256_file(args.server_info_readiness),
            "identity": readiness,
        },
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
    parser.add_argument(
        "--stage", choices=("client", "server_receipt", "wait_server_info", "dtype_debug", "dtype_debug_validate"), required=True
    )
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--served-model-name", default="qwen35-hypic")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--server-receipt", type=Path)
    parser.add_argument("--server-info-readiness", type=Path)
    parser.add_argument("--dtype-debug-receipt", type=Path)
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
    parser.add_argument("--server-info-total-timeout", type=float, default=300.0)
    parser.add_argument("--server-info-single-timeout", type=float, default=3.0)
    parser.add_argument("--server-info-poll-interval", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "wait_server_info":
        _require(args.server_pid is not None, "server_info readiness server_pid")
        wait_for_server_info(
            args.base_url,
            args.output,
            mode=args.mode,
            rank=args.rank,
            server_pid=args.server_pid,
            total_timeout=args.server_info_total_timeout,
            single_timeout=args.server_info_single_timeout,
            poll_interval=args.server_info_poll_interval,
        )
    elif args.stage == "dtype_debug_validate":
        for name in ("model", "dtype_debug_receipt"):
            _require(getattr(args, name) is not None, f"dtype debug validation {name}")
        dtype_debug_validate_stage(args)
    elif args.stage == "dtype_debug":
        for name in ("model", "data", "target_file", "dtype_debug_receipt"):
            _require(getattr(args, name) is not None, f"dtype debug {name}")
        dtype_debug_stage(args)
    elif args.stage == "client":
        for name in (
            "model", "data",
            "server_receipt", "target_file", "store_receipt", "terminal_receipt",
            "preregistration", "freeze_manifest", "expected_freeze_manifest_sha256",
        ):
            _require(getattr(args, name) is not None, f"client {name}")
        client_stage(args)
    else:
        for name in (
            "model",
            "data",
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
            "server_info_readiness",
            "freeze_manifest",
            "expected_freeze_manifest_sha256",
        ):
            _require(getattr(args, name) is not None, f"server receipt {name}")
        server_receipt_stage(args)


if __name__ == "__main__":
    main()
