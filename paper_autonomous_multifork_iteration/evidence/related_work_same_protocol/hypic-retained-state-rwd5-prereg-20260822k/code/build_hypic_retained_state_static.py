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
LIVE_DEBUG_REMOTE_ROOT = (
    "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/"
    "runs/qcomem/hypic-component-dtype-debug-trial1879097-20260822j/"
)
LIVE_DEBUG_MIRROR_MANIFEST_SHA256 = (
    "59530c0c8bc10cedbf4b0bde51d04e5490adeaf369e8738d9df363fc83941026"
)
LIVE_DEBUG_FILES = {
    "COMPLETED_DEBUG_ONLY",
    "all-debug-artifacts.sha256",
    "commands/prefix_cache.txt",
    "commands/transition_rope_recompute.txt",
    "debug-receipts/prefix_cache-rank-0.json",
    "debug-receipts/prefix_cache-server-info.json",
    "debug-receipts/prefix_cache-validation.json",
    "debug-receipts/transition_rope_recompute-rank-0.json",
    "debug-receipts/transition_rope_recompute-server-info.json",
    "debug-receipts/transition_rope_recompute-validation.json",
    "logs/prefix_cache.log",
    "logs/transition_rope_recompute.log",
    "nvidia-smi-after.txt",
    "nvidia-smi-before.txt",
    "run-summaries/prefix_cache-rank-0.json",
    "run-summaries/transition_rope_recompute-rank-0.json",
    "targets/prefix_cache-rank-0.json",
    "targets/transition_rope_recompute-rank-0.json",
}


def _sha_manifest(path: Path, *, relative: bool) -> dict[str, str]:
    require(path.is_file(), f"manifest exists: {path}")
    rows: dict[str, str] = {}
    for line in path.read_text().splitlines():
        fields = line.split(maxsplit=1)
        require(len(fields) == 2 and len(fields[0]) == 64, "manifest row")
        digest, raw_name = fields
        require(all(value in "0123456789abcdef" for value in digest), "manifest digest")
        name = raw_name.strip()
        if relative:
            require(name.startswith("./") and not name.startswith("../"), "relative manifest path")
            name = name[2:]
            candidate = Path(name)
            require(not candidate.is_absolute() and ".." not in candidate.parts, "confined manifest path")
        else:
            require(name.startswith(LIVE_DEBUG_REMOTE_ROOT), "remote debug root")
            name = name[len(LIVE_DEBUG_REMOTE_ROOT):]
            require(name and not Path(name).is_absolute() and ".." not in Path(name).parts, "remote manifest path")
        require(name not in rows, "duplicate manifest path")
        rows[name] = digest
    return rows


def _is_c_contiguous(shape: list[int], stride: list[int]) -> bool:
    expected = 1
    for size, observed in zip(reversed(shape), reversed(stride)):
        if size != 1 and observed != expected:
            return False
        expected *= size
    return True


def validate_live_component_debug(
    root: Path, *, expected_manifest_sha256: str, recurrent_layers: int
) -> dict[str, Any]:
    """Bind the successful debug-only GPU inventory into formal preregistration."""
    require(
        expected_manifest_sha256 == LIVE_DEBUG_MIRROR_MANIFEST_SHA256,
        "frozen live debug manifest identity",
    )
    manifest = root / "mirror-files.sha256"
    require(sha256_file(manifest) == expected_manifest_sha256, "live debug mirror manifest SHA")
    mirror_rows = _sha_manifest(manifest, relative=True)
    require(set(mirror_rows) == LIVE_DEBUG_FILES, "exact live debug mirror file set")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.name != "mirror-files.sha256"
    }
    require(actual == LIVE_DEBUG_FILES, "live debug mirror tree")
    for name, digest in mirror_rows.items():
        require(sha256_file(root / name) == digest, f"live debug mirror digest: {name}")
    require((root / "COMPLETED_DEBUG_ONLY").read_bytes() == b"", "debug terminal marker")
    require(not (root / "FAILED_DEBUG_ONLY").exists(), "debug failed marker absent")

    remote_ledger = root / "all-debug-artifacts.sha256"
    remote_rows = _sha_manifest(remote_ledger, relative=False)
    expected_remote = LIVE_DEBUG_FILES - {"COMPLETED_DEBUG_ONLY", "all-debug-artifacts.sha256"}
    require(set(remote_rows) == expected_remote, "remote debug artifact file set")
    for name, digest in remote_rows.items():
        require(digest == mirror_rows[name] == sha256_file(root / name), f"remote/local debug digest: {name}")

    expected_components = {
        "prefix_cache": {
            "conv[0]": ("torch.bfloat16", 2, 4),
            "temporal": ("torch.float32", 4, 5),
        },
        "transition_rope_recompute": {
            "conv[0]": ("torch.bfloat16", 2, 4),
            "temporal": ("torch.float32", 4, 5),
            "transition": ("torch.float32", 4, 5),
            "conv_tails[0]": ("torch.bfloat16", 2, 4),
        },
    }
    mode_summaries = {}
    for mode in MODES:
        raw_path = root / f"debug-receipts/{mode}-rank-0.json"
        validation_path = root / f"debug-receipts/{mode}-validation.json"
        run_path = root / f"run-summaries/{mode}-rank-0.json"
        target_path = root / f"targets/{mode}-rank-0.json"
        raw = json.loads(raw_path.read_text())
        validation = json.loads(validation_path.read_text())
        run = json.loads(run_path.read_text())
        expected_cache = "MambaRadixCache" if mode == "prefix_cache" else "PICache"
        require(
            raw.get("schema") == "hypic-rwd5-component-dtype-debug-v1"
            and raw.get("status") == "debug_only_not_formal_evidence"
            and raw.get("official_commit") == HYPIC_COMMIT
            and raw.get("mode") == mode
            and raw.get("tree_cache_class") == expected_cache
            and raw.get("mamba_pool_class") == "MambaPool"
            and raw.get("formal_receipt_emitted") is False,
            f"live debug raw identity: {mode}",
        )
        require(
            raw.get("runtime_environment") == {
                "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
                "SGLANG_MAMBA_SSM_DTYPE": "float32",
            },
            f"live debug environment: {mode}",
        )
        capacity = int(raw["mamba_capacity_axis"])
        require(capacity == int(raw["mamba_allocator_size"]) + 1, f"live debug capacity: {mode}")
        components = raw.get("components")
        require(isinstance(components, dict) and set(components) == set(expected_components[mode]), f"live component keys: {mode}")
        for name, (dtype, element_size, rank) in expected_components[mode].items():
            row = components[name]
            require(set(row) == {"dtype", "element_size", "shape", "stride", "device", "c_contiguous"}, f"live component fields: {mode}/{name}")
            shape = row["shape"]
            stride = row["stride"]
            require(
                row["dtype"] == dtype and int(row["element_size"]) == element_size
                and row["device"] == "cuda:0" and row["c_contiguous"] is True
                and isinstance(shape, list) and isinstance(stride, list)
                and len(shape) == len(stride) == rank
                and all(isinstance(value, int) and value > 0 for value in shape + stride)
                and shape[0] == recurrent_layers and shape[1] == capacity
                and _is_c_contiguous(shape, stride),
                f"live component contract: {mode}/{name}",
            )
        if mode == "transition_rope_recompute":
            require(components["conv_tails[0]"] == components["conv[0]"], "live conv tail topology")
            require(components["transition"] == components["temporal"], "live transition topology")

        raw_sha = sha256_file(raw_path)
        require(
            validation.get("schema") == "hypic-rwd5-component-dtype-debug-validation-v1"
            and validation.get("status") == "passed_exact_live_component_contract"
            and validation.get("official_commit") == HYPIC_COMMIT
            and validation.get("mode") == mode
            and validation.get("paper_evidence") is False
            and validation.get("debug_receipt_sha256") == raw_sha
            and validation.get("expected_recurrent_layers") == recurrent_layers
            and validation.get("mamba_capacity_axis") == capacity
            and validation.get("components") == components,
            f"live validation binding: {mode}",
        )
        expected_run_validation = dict(validation)
        expected_run_validation.pop("debug_receipt_sha256")
        require(
            run.get("schema") == "hypic-rwd5-component-dtype-debug-run-v1"
            and run.get("status") == "completed_debug_only_not_formal_evidence"
            and run.get("official_commit") == HYPIC_COMMIT
            and run.get("mode") == mode and run.get("rank") == 0
            and run.get("workload_id") == "qasper-6"
            and run.get("paper_evidence") is False
            and run.get("raw_formal_receipt_emitted") is False
            and run.get("store_formal_receipt_emitted") is False
            and run.get("debug_receipt_sha256") == raw_sha
            and run.get("target_sha256") == sha256_file(target_path)
            and run.get("validation") == expected_run_validation,
            f"live debug run binding: {mode}",
        )
        mode_summaries[mode] = {
            "raw_receipt_sha256": raw_sha,
            "validation_receipt_sha256": sha256_file(validation_path),
            "run_summary_sha256": sha256_file(run_path),
            "target_sha256": sha256_file(target_path),
            "cache_class": expected_cache,
            "mamba_capacity_axis": capacity,
            "components": {
                name: {
                    "dtype": row["dtype"], "element_size": row["element_size"],
                    "shape": row["shape"], "stride": row["stride"],
                }
                for name, row in components.items()
            },
        }
    return {
        "schema": "hypic-rwd5-live-component-debug-binding-v1",
        "status": "passed_debug_only_not_paper_evidence",
        "paper_evidence": False,
        "platform_job_id": 247512,
        "platform_trial_id": 1879097,
        "official_commit": HYPIC_COMMIT,
        "mirror_manifest_sha256": expected_manifest_sha256,
        "remote_artifact_ledger_sha256": sha256_file(remote_ledger),
        "terminal_completed_debug_only": True,
        "formal_receipts_emitted": 0,
        "modes": mode_summaries,
    }


def model_storage_contract(
    model: Path, live_debug_validation: dict[str, Any] | None = None
) -> dict[str, Any]:
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
        "schema": "hypic-rwd5-model-storage-contract-v3",
        "config_sha256": sha256_file(config_path),
        "model_type": model_type,
        "num_hidden_layers": num_layers,
        "full_attention_layer_ids": full_ids,
        "recurrent_layer_ids": recurrent_ids,
        "full_attention_layer_count": len(full_ids),
        "recurrent_layer_count": len(recurrent_ids),
        "conv_tensor_count": 1,
        "temporal_tensor_count": 1,
        "kv_dtype": "torch.bfloat16",
        "mamba_component_dtypes": {
            "conv": "torch.bfloat16",
            "temporal": "torch.float32",
            "transition": "torch.float32",
            "conv_tails": "torch.bfloat16",
        },
        "dtype_authority": {
            "runtime_environment": {
                "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
                "SGLANG_MAMBA_SSM_DTYPE": "float32",
            },
            "official_pool_rule": "conv and conv_tails allocate cache_params.dtype.conv; temporal and transition allocate cache_params.dtype.temporal",
            "legacy_unified_dtype_forbidden": True,
            **({"live_debug_validation": live_debug_validation} if live_debug_validation else {}),
        },
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
    live_debug_validation = validate_live_component_debug(
        args.live_debug_root,
        expected_manifest_sha256=args.expected_live_debug_manifest_sha256,
        recurrent_layers=int(storage_contract["recurrent_layer_count"]),
    )
    storage_contract = model_storage_contract(args.model, live_debug_validation)
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
            "server_readiness": "each exact mode/rank/base-URL/server-PID cell binds model_info success to evidence-bearing short /server_info polls; 300 second total deadline",
            "failure_lifecycle": "set -E plus idempotent EXIT/ERR/INT/TERM process-group cleanup; TERM is followed by bounded liveness polling, KILL escalation, and final reap; unsuccessful exit removes COMPLETED and writes FAILED before cleanup",
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
        "invalid_attempts": [
            {
                "trial_id": 1876986,
                "freeze": "C",
                "status": "invalid_before_outputs",
                "completed_cells": 0,
                "reason": "one-shot /server_info 30 second timeout during approximately 46 second scheduler internal warmup after model_info readiness",
                "paper_evidence": False,
            },
            {
                "trial_id": None,
                "freeze": "D",
                "status": "retired_before_gpu",
                "completed_cells": 0,
                "reason": "independent audit found an unbounded TERM-to-wait ordering and insufficient mode/rank/endpoint readiness binding",
                "paper_evidence": False,
            },
            {
                "trial_id": None,
                "freeze": "E",
                "status": "retired_before_gpu",
                "completed_cells": 0,
                "reason": "independent audit proved a completely re-signed rank-1 receipt chain could be placed in the rank-0 file slot because blind replay lacked an external expected-cell anchor",
                "paper_evidence": False,
            },
            {
                "trial_id": 1876986,
                "freeze": "F",
                "status": "invalid_before_raw_outputs",
                "completed_cells": 0,
                "reason": "unified Mamba dtype contract rejected the live FP32 temporal state on all Prefix servers",
                "paper_evidence": False,
            },
        ],
        "debug_validation": live_debug_validation,
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
    parser.add_argument("--live-debug-root", type=Path, required=True)
    parser.add_argument("--expected-live-debug-manifest-sha256", required=True)
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
