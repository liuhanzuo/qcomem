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
        "schema": "hypic-rwd5-retained-state-preregistration-v1",
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
        },
        "code": code,
        "model": {
            "path": str(args.model),
            "weight_ledger_raw_sha256": MODEL_LEDGER_SHA256,
            "artifact_ledger_raw_sha256": MODEL_ARTIFACT_LEDGER_SHA256,
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
            "snapshot_timing": "after formal prime completes and before measured query begins",
            "payload_denominator": "unique overlap-aware backing-storage byte ranges selected by exact cache-owned KV/Mamba slots",
            "payload_components": [
                "full-attention key",
                "full-attention value",
                "conv",
                "temporal",
                "transition when present",
                "conv_tails when present",
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
        },
    }
    return {
        "official-source-ledger.json": canonical_bytes(source),
        "environment-ledger.json": canonical_bytes(environment),
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
