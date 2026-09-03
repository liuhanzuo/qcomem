#!/usr/bin/env python3
"""Run the preregistered Hydragen operator transfer on captured Qwen3.5 Q/K/V."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable

import torch

from hydragen_vllm_flash_compat import (
    install_flash_attn_compat,
    install_hydragen_runtime_compat,
)
from qcomem_hydragen_transfer import (
    HydragenTransferError,
    build_replicated_dense_kv,
    build_transfer_case,
    canonical_json_bytes,
    cpu_fp32_oracle,
    error_metrics,
    load_rr2_capture,
    read_json,
    require,
    sha256_file,
    storage_accounting,
    timing_statistics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--capture-metadata", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--official-hydragen-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _source_closure_receipt(
    manifest_path: Path,
    expected_manifest_sha256: str,
    official_root: Path,
    preregistration_path: Path,
) -> dict[str, Any]:
    manifest_sha = sha256_file(manifest_path)
    require(manifest_sha == expected_manifest_sha256, "source manifest SHA drift")
    manifest = read_json(manifest_path)
    require(
        manifest.get("schema_version")
        == "qcomem-related-hydragen-transfer-source-manifest-v1",
        "source manifest schema drift",
    )
    rows = manifest.get("files")
    require(isinstance(rows, list) and len(rows) == 8, "source manifest row drift")
    code_root = Path(__file__).resolve().parent
    local_paths = {
        "third_party/hydragen/hydragen/attention.py": official_root / "hydragen" / "attention.py",
        "third_party/hydragen/hydragen/flash.py": official_root / "hydragen" / "flash.py",
        "third_party/hydragen/hydragen/xformers_stuff.py": official_root / "hydragen" / "xformers_stuff.py",
        "gpu/hydragen_vllm_flash_compat.py": code_root / "hydragen_vllm_flash_compat.py",
        "gpu/qcomem_hydragen_transfer.py": code_root / "qcomem_hydragen_transfer.py",
        "gpu/run_qcomem_hydragen_transfer.py": Path(__file__).resolve(),
        "gpu/test_qcomem_hydragen_transfer.py": code_root / "test_qcomem_hydragen_transfer.py",
        "paper_autonomous_multifork_iteration/evidence/round24_related_work_transfer/hydragen_preregistration.json": preregistration_path,
    }
    require(
        {row.get("path") for row in rows if isinstance(row, dict)} == set(local_paths),
        "source manifest path-set drift",
    )
    receipt_rows = []
    for row in rows:
        require(set(row) == {"path", "sha256"}, "source manifest row schema drift")
        path = local_paths[row["path"]]
        require(path.is_file(), f"missing frozen source {path}")
        digest = sha256_file(path)
        require(digest == row["sha256"], f"source hash drift: {row['path']}")
        receipt_rows.append(
            {
                "logical_path": row["path"],
                "executed_path": str(path),
                "sha256": digest,
                "bytes": path.stat().st_size,
            }
        )
    return {"manifest_sha256": manifest_sha, "files": receipt_rows}


def _write_raw_tensor(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu().float().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    path.write_bytes(raw)
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "dtype": "torch.float32",
        "shape": list(tensor.shape),
        "encoding": "torch-contiguous-raw-little-endian-v1",
    }


def _official_source_receipt(root: Path, prereg: dict[str, Any]) -> dict[str, Any]:
    method = prereg["method"]
    paths = {
        "attention": root / "hydragen" / "attention.py",
        "flash": root / "hydragen" / "flash.py",
        "xformers_kernel": root / "hydragen" / "xformers_stuff.py",
    }
    expected = {
        "attention": method["official_attention_sha256"],
        "flash": method["official_flash_sha256"],
        "xformers_kernel": method["official_xformers_kernel_sha256"],
    }
    receipt = {}
    for name, path in paths.items():
        require(path.is_file(), f"missing official Hydragen source {path}")
        digest = sha256_file(path)
        require(digest == expected[name], f"official {name} source hash drift")
        receipt[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    adapter_path = Path(__file__).resolve().parent / "hydragen_vllm_flash_compat.py"
    adapter_sha = sha256_file(adapter_path)
    require(adapter_sha == method["compatibility_adapter_sha256"], "FlashAttention compatibility adapter hash drift")
    receipt["flashattention_compatibility_adapter"] = {
        "path": str(adapter_path),
        "sha256": adapter_sha,
        "bytes": adapter_path.stat().st_size,
        "backend": "vllm.vllm_flash_attn.flash_attn_varlen_func",
    }
    return receipt


def _time_official(
    fn: Callable[[], torch.Tensor],
    timed_with_graphs: Callable[..., list[float]],
    cache: torch.Tensor,
) -> dict[str, float | int]:
    times = timed_with_graphs(
        fn,
        num_iters=200,
        num_warmup=50,
        return_type="times",
        verbose=False,
        unit="ms",
        between_fn=lambda: cache.random_(0, 1),
    )
    return timing_statistics(times)


def _environment_receipt(device: torch.device) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is unavailable")
    properties = torch.cuda.get_device_properties(device)
    name = torch.cuda.get_device_name(device)
    capability = torch.cuda.get_device_capability(device)
    require(name == "NVIDIA H20-3e", f"GPU name drift: {name}")
    require(capability == (9, 0), f"compute capability drift: {capability}")
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": name,
        "gpu_uuid": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_bytes": int(properties.total_memory),
        "multiprocessor_count": int(properties.multi_processor_count),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    prereg_sha = sha256_file(args.preregistration)
    require(prereg_sha == args.expected_preregistration_sha256, "preregistration SHA drift")
    prereg = read_json(args.preregistration)
    require(prereg.get("schema_version") == "qcomem-related-hydragen-transfer-prereg-v1", "prereg schema drift")
    require(prereg.get("status") == "frozen_before_new_gpu_outputs", "prereg status drift")
    require(not args.output_dir.exists(), "output directory already exists")

    source_closure = _source_closure_receipt(
        args.source_manifest,
        args.expected_source_manifest_sha256,
        args.official_hydragen_root,
        args.preregistration,
    )
    official_receipt = _official_source_receipt(args.official_hydragen_root, prereg)
    require(sha256_file(args.capture_metadata) == prereg["captured_model_boundary"]["source_metadata_sha256"], "capture metadata SHA drift")
    capture = load_rr2_capture(args.capture_metadata, args.raw_root)

    install_flash_attn_compat()
    sys.path.insert(0, str(args.official_hydragen_root))
    attention = importlib.import_module("hydragen.attention")
    flash = importlib.import_module("hydragen.flash")
    benchmark = importlib.import_module("hydragen.benchmark_utils")
    install_hydragen_runtime_compat(attention, flash)
    require(attention.__file__ is not None and Path(attention.__file__).resolve() == (args.official_hydragen_root / "hydragen" / "attention.py").resolve(), "Hydragen import path drift")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    environment = _environment_receipt(device)
    args.output_dir.mkdir(parents=True, mode=0o700)
    cells = []

    for resident_requests in (8, 32):
        case_cpu = build_transfer_case(capture, resident_requests)
        oracle = cpu_fp32_oracle(case_cpu)
        case = build_transfer_case(capture, resident_requests)
        q = case.query.to(device=device, non_blocking=False)
        unique_k = case.unique_key.to(device=device, non_blocking=False)
        unique_v = case.unique_value.to(device=device, non_blocking=False)
        shared_k = case.shared_key.to(device=device, non_blocking=False)
        shared_v = case.shared_value.to(device=device, non_blocking=False)
        unique_lens = case.unique_seq_lens.to(device=device)
        dense_k_cpu, dense_v_cpu, total_lens_cpu = build_replicated_dense_kv(case)
        dense_k = dense_k_cpu.to(device=device)
        dense_v = dense_v_cpu.to(device=device)
        total_lens = total_lens_cpu.to(device=device)
        del dense_k_cpu, dense_v_cpu

        dense_fn = lambda: flash.flash_attention_seqlen(q, dense_k, dense_v, seq_len=total_lens)[0]
        hydragen_fn = lambda: attention.hydragen_attention_nopad(
            q,
            unique_k,
            unique_v,
            shared_ks=[shared_k],
            shared_vs=[shared_v],
            seq_len=unique_lens,
        )

        dense_output = dense_fn()
        hydragen_output = hydragen_fn()
        torch.cuda.synchronize(device)
        dense_cpu_metrics = error_metrics(dense_output, oracle)
        hydragen_cpu_metrics = error_metrics(hydragen_output, oracle)
        hydragen_dense_metrics = error_metrics(hydragen_output, dense_output)

        cache = torch.empty(134217728 // 2, dtype=torch.bfloat16, device=device)
        dense_timing = _time_official(dense_fn, benchmark.timed_with_graphs, cache)
        hydragen_timing = _time_official(hydragen_fn, benchmark.timed_with_graphs, cache)

        prefix = f"N{resident_requests}"
        sidecars = {
            "cpu_fp32_oracle": _write_raw_tensor(args.output_dir / f"{prefix}-cpu-fp32-oracle.bin", oracle),
            "replicated_dense_output": _write_raw_tensor(args.output_dir / f"{prefix}-replicated-dense-output.bin", dense_output),
            "hydragen_output": _write_raw_tensor(args.output_dir / f"{prefix}-hydragen-output.bin", hydragen_output),
        }
        gates = {
            "dense_vs_cpu_fp32_relative_l2": dense_cpu_metrics["relative_l2"] <= 0.005,
            "hydragen_vs_cpu_fp32_relative_l2": hydragen_cpu_metrics["relative_l2"] <= 0.005,
            "hydragen_vs_dense_relative_l2": hydragen_dense_metrics["relative_l2"] <= 0.005,
            "all_outputs_finite": bool(
                dense_cpu_metrics["finite"]
                and hydragen_cpu_metrics["finite"]
                and hydragen_dense_metrics["finite"]
            ),
        }
        cell = {
            "resident_requests": resident_requests,
            "query_indices_zero_based": list(case.query_indices),
            "unique_kv_lengths": list(case.unique_lengths),
            "geometry": {
                "shared_prefix_tokens": 4095,
                "padded_unique_tokens": 32,
                "query_tokens_per_request": 1,
                "query_heads": 16,
                "key_value_heads": 2,
                "head_dim": 256,
                "dtype": "torch.bfloat16",
            },
            "metrics": {
                "replicated_dense_vs_cpu_fp32": dense_cpu_metrics,
                "hydragen_vs_cpu_fp32": hydragen_cpu_metrics,
                "hydragen_vs_replicated_dense": hydragen_dense_metrics,
            },
            "timing": {
                "replicated_dense": dense_timing,
                "hydragen_shared_prefix": hydragen_timing,
                "dense_over_hydragen_mean_speedup": dense_timing["mean_ms"] / hydragen_timing["mean_ms"],
            },
            "storage": storage_accounting(case),
            "sidecars": sidecars,
            "gates": gates,
            "passed": all(gates.values()),
        }
        cells.append(cell)
        del q, unique_k, unique_v, shared_k, shared_v, unique_lens, dense_k, dense_v, total_lens
        del dense_output, hydragen_output, oracle, cache
        torch.cuda.empty_cache()

    summary = {
        "schema_version": "qcomem-related-hydragen-transfer-result-v1",
        "scientific_scope": "captured Qwen3.5 layer-3 post-RoPE attention operator only",
        "formal_evidence_eligible": True,
        "preregistration": {
            "path": str(args.preregistration),
            "sha256": prereg_sha,
        },
        "source_closure": source_closure,
        "official_hydragen": {
            "commit": prereg["method"]["official_commit"],
            "source_receipt": official_receipt,
        },
        "capture": {
            "metadata_path": str(args.capture_metadata),
            "metadata_sha256": capture.metadata_sha256,
            "sidecar_sha256": capture.sidecar_sha256,
        },
        "environment": environment,
        "cells": cells,
        "passed": all(cell["passed"] for cell in cells),
        "claim_boundary": prereg["not_authorized"],
    }
    output_path = args.output_dir / "hydragen-transfer-summary.json"
    output_path.write_bytes(canonical_json_bytes(summary))
    ledger_rows = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file():
            ledger_rows.append(f"{sha256_file(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(ledger_rows) + "\n")
    print(json.dumps({"output": str(output_path), "sha256": sha256_file(output_path), "passed": summary["passed"]}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except HydragenTransferError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
