#!/usr/bin/env python3
"""Run the bounded Palu-style Qwen3.5 projection transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from qcomem_palu_transfer import (
    PaluTransferError,
    apply_headwise,
    headwise_svd_factors,
    logical_kv_storage,
    relative_l2,
    require,
    truncate_factors,
)


class CaptureComplete(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-preregistration-sha256", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--expected-source-manifest-sha256", required=True)
    parser.add_argument("--official-palu-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def write_tensor(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    tensor = tensor.detach().cpu().contiguous()
    raw = tensor.view(torch.uint8).numpy().tobytes()
    path.write_bytes(raw)
    return {
        "relative_path": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "encoding": "torch-contiguous-raw-little-endian-v1",
    }


def source_closure(args: argparse.Namespace, prereg: dict[str, Any]) -> dict[str, Any]:
    manifest_sha = sha256(args.source_manifest)
    require(manifest_sha == args.expected_source_manifest_sha256, "source manifest SHA drift")
    manifest = json.loads(args.source_manifest.read_text())
    require(manifest.get("schema_version") == "qcomem-related-palu-transfer-source-manifest-v1", "source manifest schema drift")
    rows = manifest.get("files")
    require(isinstance(rows, list) and len(rows) == 6, "source manifest row drift")
    code = Path(__file__).resolve().parent
    mapping = {
        "third_party/palu/palu/model/modules/svd_linear.py": args.official_palu_root / "palu/model/modules/svd_linear.py",
        "third_party/palu/palu/model/svd_qwen/modeling_palu_qwen.py": args.official_palu_root / "palu/model/svd_qwen/modeling_palu_qwen.py",
        "gpu/qcomem_palu_transfer.py": code / "qcomem_palu_transfer.py",
        "gpu/run_qcomem_palu_transfer.py": Path(__file__).resolve(),
        "gpu/test_qcomem_palu_transfer.py": code / "test_qcomem_palu_transfer.py",
        "paper_autonomous_multifork_iteration/evidence/round24_related_work_transfer/palu_preregistration.json": args.preregistration,
    }
    require({row.get("path") for row in rows} == set(mapping), "source manifest path drift")
    receipt = []
    for row in rows:
        require(set(row) == {"path", "sha256"}, "source row schema drift")
        path = mapping[row["path"]]
        require(path.is_file() and sha256(path) == row["sha256"], f"source drift: {row['path']}")
        receipt.append({"logical_path": row["path"], "executed_path": str(path), "sha256": row["sha256"], "bytes": path.stat().st_size})
    require(receipt[0]["sha256"] == prereg["method"]["official_svd_module_sha256"], "official SVD hash drift")
    require(receipt[1]["sha256"] == prereg["method"]["official_qwen2_wrapper_sha256"], "official Qwen wrapper hash drift")
    return {"manifest_sha256": manifest_sha, "files": receipt}


def model_input_receipt(args: argparse.Namespace, prereg: dict[str, Any]) -> tuple[torch.Tensor, dict[str, Any]]:
    from transformers import AutoTokenizer

    model = prereg["model"]
    checks = {
        "model-weights.sha256": model["model_weight_ledger_sha256"],
        "model-artifacts.sha256": model["model_artifact_ledger_sha256"],
        "model.safetensors.index.json": model["weight_index_sha256"],
        "config.json": model["config_sha256"],
    }
    observed = {}
    for name, expected in checks.items():
        path = args.model_dir / name
        require(path.is_file() and sha256(path) == expected, f"model receipt drift: {name}")
        observed[name] = expected
    require(sha256(args.data_file) == prereg["input"]["dataset_sha256"], "dataset SHA drift")
    row = json.loads(args.data_file.read_text().splitlines()[0])
    require(isinstance(row.get("text"), str), "PG19 row text missing")
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    ids = tokenizer(row["text"], add_special_tokens=False, return_tensors="pt")["input_ids"][:, :256].to(torch.int64).contiguous()
    raw = ids.numpy().tobytes()
    token_sha = hashlib.sha256(raw).hexdigest()
    require(list(ids.shape) == prereg["input"]["token_ids_shape"], "token shape drift")
    require(token_sha == prereg["input"]["token_ids_sha256"], "token byte drift")
    return ids, {"model_files": observed, "dataset_sha256": prereg["input"]["dataset_sha256"], "token_ids_sha256": token_sha, "token_ids_shape": list(ids.shape)}


@torch.no_grad()
def main() -> None:
    args = parse_args()
    require(not args.output_dir.exists(), "output directory already exists")
    prereg_sha = sha256(args.preregistration)
    require(prereg_sha == args.expected_preregistration_sha256, "preregistration SHA drift")
    prereg = json.loads(args.preregistration.read_text())
    require(prereg.get("schema_version") == "qcomem-related-palu-transfer-prereg-v1", "prereg schema drift")
    require(prereg.get("status") == "frozen_before_new_gpu_model_outputs", "prereg status drift")
    source = source_closure(args, prereg)
    token_ids, inputs = model_input_receipt(args, prereg)

    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.cuda.get_device_name(0) == "NVIDIA H20-3e", "GPU name drift")
    require(torch.cuda.get_device_capability(0) == (9, 0), "GPU capability drift")
    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir),
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "visual"):
        outer.visual = None
    model.eval().to(device="cuda:0", dtype=torch.bfloat16)
    backbone = getattr(getattr(model, "model", None), "language_model", None)
    require(backbone is not None and hasattr(backbone, "layers"), "Qwen3.5 text backbone missing")
    attention = backbone.layers[3].self_attn
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, positional: tuple[Any, ...], keywords: dict[str, Any]) -> None:
        hidden = keywords.get("hidden_states")
        if hidden is None and positional:
            hidden = positional[0]
        require(isinstance(hidden, torch.Tensor), "layer-3 hidden input missing")
        captured["hidden"] = hidden.detach().clone()
        raise CaptureComplete

    handle = attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        try:
            backbone(input_ids=token_ids.to("cuda:0"), use_cache=False)
        except CaptureComplete:
            pass
    finally:
        handle.remove()
    require("hidden" in captured, "hidden capture did not fire")
    hidden = captured["hidden"]
    require(tuple(hidden.shape) == (1, 256, 2048), "hidden geometry drift")
    k_weight = attention.k_proj.weight.detach().clone()
    v_weight = attention.v_proj.weight.detach().clone()
    k_bias = attention.k_proj.bias.detach().clone() if attention.k_proj.bias is not None else None
    v_bias = attention.v_proj.bias.detach().clone() if attention.v_proj.bias is not None else None
    original_k = attention.k_proj(hidden).detach()
    original_v = attention.v_proj(hidden).detach()
    del model, backbone, attention, outer
    torch.cuda.empty_cache()

    k_full = headwise_svd_factors(k_weight, heads=2)
    v_full = headwise_svd_factors(v_weight, heads=2)
    rows = []
    candidates: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for rank in prereg["ranks_per_kv_head"]:
        approx_k = apply_headwise(hidden, truncate_factors(k_full, rank=rank, dtype=torch.bfloat16), bias=k_bias)
        approx_v = apply_headwise(hidden, truncate_factors(v_full, rank=rank, dtype=torch.bfloat16), bias=v_bias)
        candidates[rank] = (approx_k, approx_v)
        k_error = relative_l2(approx_k, original_k)
        v_error = relative_l2(approx_v, original_v)
        rows.append({
            "rank_per_kv_head": rank,
            "projection_relative_l2": {"key": k_error, "value": v_error},
            "all_outputs_finite": bool(torch.isfinite(approx_k).all().item() and torch.isfinite(approx_v).all().item()),
            "logical_storage": logical_kv_storage(rank=rank),
        })
    key_errors = [row["projection_relative_l2"]["key"] for row in rows]
    value_errors = [row["projection_relative_l2"]["value"] for row in rows]
    monotone = all(a > b for a, b in zip(key_errors, key_errors[1:])) and all(a > b for a, b in zip(value_errors, value_errors[1:]))
    passed = monotone and all(row["all_outputs_finite"] for row in rows)

    args.output_dir.mkdir(parents=True, mode=0o700)
    sidecars = {
        "token_ids": write_tensor(args.output_dir / "token-ids-int64.bin", token_ids),
        "layer3_hidden_bf16": write_tensor(args.output_dir / "layer3-hidden-bf16.bin", hidden),
        "original_key_bf16": write_tensor(args.output_dir / "original-key-bf16.bin", original_k),
        "original_value_bf16": write_tensor(args.output_dir / "original-value-bf16.bin", original_v),
    }
    for rank, (key, value) in candidates.items():
        sidecars[f"rank{rank}_key_bf16"] = write_tensor(args.output_dir / f"rank{rank}-key-bf16.bin", key)
        sidecars[f"rank{rank}_value_bf16"] = write_tensor(args.output_dir / f"rank{rank}-value-bf16.bin", value)
    summary = {
        "schema_version": "qcomem-related-palu-transfer-result-v1",
        "scientific_scope": "Qwen3.5 layer-3 K/V projection operator on one frozen 256-token PG19 prefix",
        "formal_evidence_eligible": True,
        "passed": passed,
        "preregistration": {"sha256": prereg_sha, "path": str(args.preregistration)},
        "source_closure": source,
        "input_receipt": inputs,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0), "compute_capability": "9.0", "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "geometry": {"layer_index": 3, "hidden_shape": list(hidden.shape), "projection_shape": list(original_k.shape), "key_value_heads": 2, "head_dim": 256},
        "rows": rows,
        "gates": {"all_outputs_finite": all(row["all_outputs_finite"] for row in rows), "projection_error_strictly_decreases_with_rank": monotone},
        "sidecars": sidecars,
        "claim_boundary": prereg["not_authorized"],
    }
    summary_path = args.output_dir / "palu-transfer-summary.json"
    summary_path.write_bytes(canonical_json(summary))
    ledger = [f"{sha256(path)}  {path.name}" for path in sorted(args.output_dir.iterdir()) if path.is_file()]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(ledger) + "\n")
    print(json.dumps({"output": str(summary_path), "sha256": sha256(summary_path), "passed": passed}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except PaluTransferError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
