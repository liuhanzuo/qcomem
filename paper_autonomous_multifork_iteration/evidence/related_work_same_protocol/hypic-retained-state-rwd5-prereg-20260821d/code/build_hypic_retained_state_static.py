#!/usr/bin/env python3
"""Pre-output/terminal static authority for affected-only HYPIC RW-D5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from build_hypic_formal_static import (
    DATA_SHA256,
    HYPIC_COMMIT,
    MODEL_ARTIFACT_LEDGER_SHA256,
    MODEL_LEDGER_SHA256,
    atomic_json,
    build_environment_ledger,
    build_source_ledger,
    canonical_bytes,
    parse_model_ledger,
    require,
    sha256_file,
)


MODES = ["prefix_cache", "transition_rope_recompute"]


def model_storage_contract(model: Path) -> dict[str, Any]:
    """Derive the exact, model-specific RW-D5 tensor topology."""
    config_path = model / "config.json"
    require(config_path.is_file(), "model config.json")
    raw = json.loads(config_path.read_text())
    text = raw.get("text_config", raw)
    require(isinstance(text, dict), "text model config")
    model_type = str(text.get("model_type", ""))
    require(model_type in {"qwen3_5_text", "qwen3_5_moe_text"}, "Qwen3.5 text config")
    num_layers = int(text["num_hidden_layers"])
    require(num_layers > 0, "num_hidden_layers")
    layer_types = text.get("layer_types")
    if layer_types is None:
        interval = int(text["full_attention_interval"])
        require(interval > 0, "full_attention_interval")
        layer_types = [
            "attention" if (index + 1) % interval == 0 else "linear_attention"
            for index in range(num_layers)
        ]
    require(isinstance(layer_types, list) and len(layer_types) == num_layers, "layer_types")
    normalized = [str(value) for value in layer_types]
    full_ids = [
        index
        for index, value in enumerate(normalized)
        if value in {"attention", "full_attention"}
    ]
    recurrent_ids = [
        index for index, value in enumerate(normalized) if value == "linear_attention"
    ]
    require(full_ids and recurrent_ids, "hybrid layer partition")
    require(len(full_ids) + len(recurrent_ids) == num_layers, "known layer types only")
    return {
        "schema": "hypic-rwd5-model-storage-contract-v2",
        "config_sha256": sha256_file(config_path),
        "model_type": model_type,
        "num_hidden_layers": num_layers,
        "full_attention_layer_ids": full_ids,
        "recurrent_layer_ids": recurrent_ids,
        "full_attention_layer_count": len(full_ids),
        "recurrent_layer_count": len(recurrent_ids),
        "conv_tensor_count": 1,
        "temporal_tensor_count": 1,
        "dtype": "torch.bfloat16",
        "kv_layout": "nhd",
        "kv_slot_axis": 0,
        "mamba_layer_axis": 0,
        "mamba_slot_axis": 1,
        "page_size": 1,
        "enable_int8_mamba_checkpoint": False,
        "mode_components": {
            "prefix_cache": {
                "transition_tensor_count": 0,
                "conv_tails_tensor_count": 0,
            },
            "transition_rope_recompute": {
                "transition_tensor_count": 1,
                "conv_tails_tensor_count": 1,
            },
        },
    }


def overlay_material(args: argparse.Namespace) -> tuple[dict[str, Any], bytes]:
    expected_status = [
        " M python/sglang/srt/managers/scheduler.py",
        " M python/sglang/srt/mem_cache/common.py",
        "?? python/sglang/srt/retained_state_receipt.py",
    ]
    status = subprocess.check_output(
        [
            "git", "-C", str(args.instrumented_repo), "status",
            "--porcelain=v1", "--untracked-files=all",
        ],
        text=True,
    ).splitlines()
    require(sorted(status) == sorted(expected_status), f"instrumented overlay status: {status!r}")
    diff = subprocess.check_output(
        [
            "git", "-C", str(args.instrumented_repo), "diff", "--binary",
            "--no-ext-diff", "--full-index", "--",
            "python/sglang/srt/managers/scheduler.py",
            "python/sglang/srt/mem_cache/common.py",
        ]
    )
    require(diff, "canonical tracked overlay diff")
    module = args.instrumented_repo / "python/sglang/srt/retained_state_receipt.py"
    ledger = {
        "schema": "hypic-rwd5-instrumentation-overlay-v2",
        "base_commit": HYPIC_COMMIT,
        "porcelain_v1": sorted(status),
        "tracked_files": [
            "python/sglang/srt/managers/scheduler.py",
            "python/sglang/srt/mem_cache/common.py",
        ],
        "untracked_files": ["python/sglang/srt/retained_state_receipt.py"],
        "canonical_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "receipt_module_sha256": sha256_file(module),
        "no_other_tracked_or_untracked": True,
    }
    return ledger, diff


def code_row(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"missing code file: {path}")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def materialize(args: argparse.Namespace, verify_model_bytes: bool) -> dict[str, bytes]:
    require(sha256_file(args.data) == DATA_SHA256, "data SHA")
    parse_model_ledger(
        args.model_weight_ledger,
        args.model,
        expected_raw_sha256=MODEL_LEDGER_SHA256,
        expected_count=14,
        label="model weight",
        verify_bytes=verify_model_bytes,
    )
    parse_model_ledger(
        args.model_artifact_ledger,
        args.model,
        expected_raw_sha256=MODEL_ARTIFACT_LEDGER_SHA256,
        expected_count=9,
        label="model artifact",
        verify_bytes=verify_model_bytes,
    )
    source = build_source_ledger(args.official_repo)
    environment = build_environment_ledger()
    subprocess.check_call(
        ["git", "-C", str(args.official_repo), "apply", "--check", str(args.patch)]
    )
    subprocess.check_call(
        ["git", "-C", str(args.instrumented_repo), "apply", "--reverse", "--check", str(args.patch)]
    )
    installed_module = args.instrumented_repo / "python/sglang/srt/retained_state_receipt.py"
    require(
        installed_module.is_file()
        and sha256_file(installed_module) == sha256_file(args.receipt_module),
        "installed receipt module",
    )
    require(args.freeze_manifest.is_file(), "external frozen manifest")
    require(
        sha256_file(args.freeze_manifest) == args.expected_freeze_manifest_sha256,
        "external frozen manifest SHA",
    )
    subprocess.check_call(
        ["sha256sum", "-c", str(args.freeze_manifest)], cwd=args.freeze_manifest.parent
    )
    storage_contract = model_storage_contract(args.model)
    overlay, overlay_diff = overlay_material(args)
    code = {
        name: code_row(getattr(args, name))
        for name in (
            "client",
            "formal_helper",
            "formal_static_helper",
            "serving_helper",
            "receipt_module",
            "patch",
            "receipt_test",
            "replay",
            "launcher",
            "static_builder",
        )
    }
    prereg = {
        "schema": "hypic-rwd5-retained-state-preregistration-v2",
        "status": "frozen_before_outputs",
        "research_question": "What physical tensor payload is owned by the cached document in Prefix Cache and HYPIC under the completed same-protocol run?",
        "official_repository": "https://github.com/redai-infra/HYPIC",
        "official_commit": HYPIC_COMMIT,
        "official_source_ledger_sha256": hashlib.sha256(canonical_bytes(source)).hexdigest(),
        "environment_ledger_sha256": hashlib.sha256(canonical_bytes(environment)).hexdigest(),
        "instrumentation": {
            "read_only": True,
            "official_checkout_remains_clean": True,
            "temporary_copy_base_commit": HYPIC_COMMIT,
            "patch_sha256": sha256_file(args.patch),
            "module_sha256": sha256_file(args.receipt_module),
            "forbidden": ["NVML", "process allocation delta", "pool capacity delta"],
            "overlay": overlay,
        },
        "external_freeze": {
            "manifest_path": str(args.freeze_manifest),
            "manifest_sha256": args.expected_freeze_manifest_sha256,
            "all_entries_verified": True,
        },
        "code": code,
        "model": {
            "path": str(args.model),
            "weight_ledger_raw_sha256": MODEL_LEDGER_SHA256,
            "artifact_ledger_raw_sha256": MODEL_ARTIFACT_LEDGER_SHA256,
            "config_sha256": storage_contract["config_sha256"],
            "storage_contract": storage_contract,
            "storage_contract_sha256": hashlib.sha256(canonical_bytes(storage_contract)).hexdigest(),
        },
        "data": {"path": str(args.data), "sha256": DATA_SHA256, "frozen_rows": 8},
        "design": {
            "modes": MODES,
            "cells": 16,
            "excluded_modes": [
                "full_recompute",
                "CoMem",
                "RR2",
                "GDN",
                "vLLM",
                "SGLang serving controls",
            ],
            "model": "Qwen3.5-35B-A3B",
            "hardware": "one H20-3e per frozen row",
            "tp_size": 1,
            "input_cap": 4096,
            "decode": "greedy max 32 tokens",
            "hypic_seam_tokens": 8,
            "server_readiness": "model_info success followed by evidence-bearing short /server_info polls; 300 second total deadline",
            "failure_lifecycle": "set -E plus idempotent EXIT/ERR/INT/TERM process-group cleanup; unsuccessful exit removes COMPLETED and writes FAILED",
            "snapshot_timing": "after formal prime completes and before measured query begins",
            "payload_denominator": "unique overlap-aware backing-storage byte ranges selected by exact cache-owned KV/Mamba slots",
            "payload_components": [
                "full-attention key",
                "full-attention value",
                "conv",
                "temporal",
                "transition exactly once for HYPIC and absent for Prefix Cache",
                "conv_tails exactly once for HYPIC and absent for Prefix Cache",
            ],
            "metadata": "reported separately and excluded from Store MiB",
            "terminal_gate": "flush removes exact target entries and returns every old KV/Mamba slot to its allocator",
        },
        "acceptance": {
            "all_16_cells": True,
            "cache_hit_coverage": True,
            "range_inside_backing_storage": True,
            "blind_union_replay_exact": True,
            "terminal_ownership_removal": True,
            "median_only_after_all_rows_pass": True,
            "server_info_ready_before_server_receipt": True,
            "failure_terminal_has_no_completed_marker": True,
        },
        "invalid_attempts": [{
            "trial_id": 1876986,
            "freeze": "C",
            "status": "invalid_before_outputs",
            "completed_cells": 0,
            "reason": "one-shot /server_info 30 second timeout during approximately 46 second scheduler internal warmup after model_info readiness",
            "paper_evidence": False,
        }],
    }
    return {
        "official-source-ledger.json": canonical_bytes(source),
        "environment-ledger.json": canonical_bytes(environment),
        "model-storage-contract.json": canonical_bytes(storage_contract),
        "instrumentation-overlay.json": canonical_bytes(overlay),
        "instrumentation-overlay.diff": overlay_diff,
        "preregistration.json": canonical_bytes(prereg),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build", "verify"), required=True)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--instrumented-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-weight-ledger", type=Path, required=True)
    parser.add_argument("--model-artifact-ledger", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--freeze-manifest", type=Path, required=True)
    parser.add_argument("--expected-freeze-manifest-sha256", required=True)
    for name in (
        "client",
        "formal_helper",
        "formal_static_helper",
        "serving_helper",
        "receipt_module",
        "patch",
        "receipt_test",
        "replay",
        "launcher",
        "static_builder",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--verify-model-bytes", action="store_true")
    args = parser.parse_args()
    outputs = materialize(args, verify_model_bytes=args.stage == "build" or args.verify_model_bytes)
    if args.stage == "build":
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, data in outputs.items():
            (args.output_dir / name).write_bytes(data)
        atomic_json(
            args.output_dir / "preoutput-validation.json",
            {
                "schema": "hypic-rwd5-preoutput-validation-v1",
                "passed": True,
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(outputs.items())},
            },
        )
    else:
        require(args.validation_output is not None, "validation output")
        for name, expected in outputs.items():
            require((args.output_dir / name).read_bytes() == expected, f"terminal static drift: {name}")
        atomic_json(
            args.validation_output,
            {
                "schema": "hypic-rwd5-terminal-static-verification-v1",
                "passed": True,
                "model_bytes_rehashed": bool(args.verify_model_bytes),
                "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(outputs.items())},
            },
        )


if __name__ == "__main__":
    main()
