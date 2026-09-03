#!/usr/bin/env python3
"""Run a bounded held-out Palu whitening transfer on Qwen3.5 layer 3."""

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
    activation_whitener,
    apply_headwise,
    headwise_svd_factors,
    logical_kv_storage,
    relative_l2,
    require,
    truncate_factors,
    whitened_headwise_svd_factors,
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
    observed_manifest_sha = sha256(args.source_manifest)
    require(observed_manifest_sha == args.expected_source_manifest_sha256, "source manifest SHA drift")
    manifest = json.loads(args.source_manifest.read_text())
    require(manifest.get("schema_version") == "qcomem-related-palu-whiten-source-manifest-v1", "source manifest schema drift")
    code = Path(__file__).resolve().parent
    mapping = {
        "third_party/palu/palu/decomposition.py": args.official_palu_root / "palu/decomposition.py",
        "third_party/palu/palu/model/modules/svd_linear.py": args.official_palu_root / "palu/model/modules/svd_linear.py",
        "gpu/qcomem_palu_transfer.py": code / "qcomem_palu_transfer.py",
        "gpu/run_qcomem_palu_whiten_transfer.py": Path(__file__).resolve(),
        "gpu/test_qcomem_palu_transfer.py": code / "test_qcomem_palu_transfer.py",
        "paper_autonomous_multifork_iteration/evidence/round24_related_work_transfer/palu_whiten_preregistration.json": args.preregistration,
    }
    rows = manifest.get("files")
    require(isinstance(rows, list) and len(rows) == len(mapping), "source manifest row drift")
    require({row.get("path") for row in rows} == set(mapping), "source manifest path drift")
    receipt = []
    for row in rows:
        require(set(row) == {"path", "sha256"}, "source row schema drift")
        path = mapping[row["path"]]
        require(path.is_file() and sha256(path) == row["sha256"], f"source drift: {row['path']}")
        receipt.append({"logical_path": row["path"], "executed_path": str(path), "sha256": row["sha256"], "bytes": path.stat().st_size})
    require(manifest["official_commit"] == prereg["method"]["official_commit"], "official commit drift")
    require(next(row for row in rows if row["path"].endswith("decomposition.py"))["sha256"] == prereg["method"]["official_decomposition_sha256"], "official decomposition hash drift")
    require(next(row for row in rows if row["path"].endswith("svd_linear.py"))["sha256"] == prereg["method"]["official_svd_module_sha256"], "official SVD hash drift")
    return {"manifest_sha256": observed_manifest_sha, "files": receipt}


def model_and_tokens(args: argparse.Namespace, prereg: dict[str, Any]) -> tuple[dict[int, torch.Tensor], dict[str, Any]]:
    from transformers import AutoTokenizer

    model = prereg["model"]
    expected_files = {
        "model-weights.sha256": model["model_weight_ledger_sha256"],
        "model-artifacts.sha256": model["model_artifact_ledger_sha256"],
        "model.safetensors.index.json": model["weight_index_sha256"],
        "config.json": model["config_sha256"],
    }
    for name, expected in expected_files.items():
        require((args.model_dir / name).is_file() and sha256(args.model_dir / name) == expected, f"model receipt drift: {name}")
    require(sha256(args.data_file) == prereg["input"]["dataset_sha256"], "dataset SHA drift")
    rows = [json.loads(line) for line in args.data_file.read_text().splitlines()]
    wanted = prereg["input"]["calibration_rows_zero_based"] + [prereg["input"]["heldout_evaluation_row_zero_based"]]
    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    tokens: dict[int, torch.Tensor] = {}
    receipts = {}
    for index in wanted:
        require(index < len(rows) and isinstance(rows[index].get("text"), str), f"PG19 row {index} missing")
        ids = tokenizer(rows[index]["text"], add_special_tokens=False, return_tensors="pt")["input_ids"][:, :256].to(torch.int64).contiguous()
        require(tuple(ids.shape) == (1, 256), f"token geometry drift row {index}")
        digest = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
        require(digest == prereg["input"]["token_ids_sha256_by_row"][str(index)], f"token byte drift row {index}")
        tokens[index] = ids
        receipts[str(index)] = {"sha256": digest, "shape": list(ids.shape), "dtype": str(ids.dtype)}
    return tokens, {"model_files": expected_files, "dataset_sha256": prereg["input"]["dataset_sha256"], "token_rows": receipts}


def capture_layer3_hidden(backbone: Any, attention: Any, token_ids: torch.Tensor) -> torch.Tensor:
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
    require("hidden" in captured and tuple(captured["hidden"].shape) == (1, 256, 2048), "hidden capture drift")
    return captured["hidden"]


def strictly_decreasing(values: list[float]) -> bool:
    return all(left > right for left, right in zip(values, values[1:]))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    require(not args.output_dir.exists(), "output directory already exists")
    prereg_sha = sha256(args.preregistration)
    require(prereg_sha == args.expected_preregistration_sha256, "preregistration SHA drift")
    prereg = json.loads(args.preregistration.read_text())
    require(prereg.get("schema_version") == "qcomem-related-palu-whiten-transfer-prereg-v1", "prereg schema drift")
    require(prereg.get("status") == "frozen_before_new_gpu_model_outputs", "prereg status drift")
    source = source_closure(args, prereg)
    tokens, inputs = model_and_tokens(args, prereg)
    require(torch.cuda.is_available(), "CUDA unavailable")
    require(torch.cuda.get_device_name(0) == "NVIDIA H20-3e", "GPU name drift")
    require(torch.cuda.get_device_capability(0) == (9, 0), "GPU capability drift")

    from transformers import AutoModelForImageTextToText

    model = AutoModelForImageTextToText.from_pretrained(str(args.model_dir), dtype=torch.bfloat16, local_files_only=True, trust_remote_code=False)
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "visual"):
        outer.visual = None
    model.eval().to(device="cuda:0", dtype=torch.bfloat16)
    backbone = getattr(getattr(model, "model", None), "language_model", None)
    require(backbone is not None and hasattr(backbone, "layers"), "Qwen3.5 text backbone missing")
    attention = backbone.layers[3].self_attn
    calibration_hiddens = [capture_layer3_hidden(backbone, attention, tokens[index]) for index in prereg["input"]["calibration_rows_zero_based"]]
    heldout_hidden = capture_layer3_hidden(backbone, attention, tokens[prereg["input"]["heldout_evaluation_row_zero_based"]])
    calibration_hidden = torch.cat(calibration_hiddens, dim=1).contiguous()
    require(tuple(calibration_hidden.shape) == (1, 2048, 2048), "calibration hidden geometry drift")
    k_weight = attention.k_proj.weight.detach().clone()
    v_weight = attention.v_proj.weight.detach().clone()
    k_bias = attention.k_proj.bias.detach().clone() if attention.k_proj.bias is not None else None
    v_bias = attention.v_proj.bias.detach().clone() if attention.v_proj.bias is not None else None
    original_k = attention.k_proj(heldout_hidden).detach()
    original_v = attention.v_proj(heldout_hidden).detach()
    del model, backbone, attention, outer, calibration_hiddens
    torch.cuda.empty_cache()

    whitener, whitener_receipt = activation_whitener(calibration_hidden)
    plain_k = headwise_svd_factors(k_weight, heads=2)
    plain_v = headwise_svd_factors(v_weight, heads=2)
    whiten_k = whitened_headwise_svd_factors(k_weight, heads=2, scale=whitener)
    whiten_v = whitened_headwise_svd_factors(v_weight, heads=2, scale=whitener)
    rows = []
    tensors: dict[str, torch.Tensor] = {}
    for rank in prereg["ranks_per_kv_head"]:
        plain_key = apply_headwise(heldout_hidden, truncate_factors(plain_k, rank=rank, dtype=torch.bfloat16), bias=k_bias)
        plain_value = apply_headwise(heldout_hidden, truncate_factors(plain_v, rank=rank, dtype=torch.bfloat16), bias=v_bias)
        whiten_key = apply_headwise(heldout_hidden, truncate_factors(whiten_k, rank=rank, dtype=torch.bfloat16), bias=k_bias)
        whiten_value = apply_headwise(heldout_hidden, truncate_factors(whiten_v, rank=rank, dtype=torch.bfloat16), bias=v_bias)
        finite = all(bool(torch.isfinite(value).all().item()) for value in (plain_key, plain_value, whiten_key, whiten_value))
        rows.append({
            "rank_per_kv_head": rank,
            "heldout_projection_relative_l2": {
                "plain_svd": {"key": relative_l2(plain_key, original_k), "value": relative_l2(plain_value, original_v)},
                "activation_whitened": {"key": relative_l2(whiten_key, original_k), "value": relative_l2(whiten_value, original_v)},
            },
            "all_outputs_finite": finite,
            "logical_storage": logical_kv_storage(rank=rank),
        })
        tensors.update({f"rank{rank}_plain_key": plain_key, f"rank{rank}_plain_value": plain_value, f"rank{rank}_whiten_key": whiten_key, f"rank{rank}_whiten_value": whiten_value})

    metric_keys = (("plain_svd", "key"), ("plain_svd", "value"), ("activation_whitened", "key"), ("activation_whitened", "value"))
    monotone = all(strictly_decreasing([row["heldout_projection_relative_l2"][method][kind] for row in rows]) for method, kind in metric_keys)
    finite = all(row["all_outputs_finite"] for row in rows)
    hypothesis = all(
        row["heldout_projection_relative_l2"]["activation_whitened"][kind] < row["heldout_projection_relative_l2"]["plain_svd"][kind]
        for row in rows for kind in ("key", "value")
    )
    scientific_run_valid = finite and monotone

    args.output_dir.mkdir(parents=True, mode=0o700)
    sidecars = {
        "calibration_hidden_bf16": write_tensor(args.output_dir / "calibration-hidden-bf16.bin", calibration_hidden),
        "heldout_hidden_bf16": write_tensor(args.output_dir / "heldout-hidden-bf16.bin", heldout_hidden),
        "whitener_fp32": write_tensor(args.output_dir / "whitener-fp32.bin", whitener),
        "original_key_bf16": write_tensor(args.output_dir / "original-key-bf16.bin", original_k),
        "original_value_bf16": write_tensor(args.output_dir / "original-value-bf16.bin", original_v),
    }
    for name, tensor in tensors.items():
        sidecars[name] = write_tensor(args.output_dir / f"{name.replace('_', '-')}-bf16.bin", tensor)
    for index, token_ids in tokens.items():
        sidecars[f"token_ids_row{index}"] = write_tensor(args.output_dir / f"token-ids-row{index}-int64.bin", token_ids)
    summary = {
        "schema_version": "qcomem-related-palu-whiten-transfer-result-v1",
        "scientific_scope": "Qwen3.5 layer-3 K/V projection: eight 256-token calibration rows and one disjoint 256-token held-out PG19 row",
        "formal_evidence_eligible": True,
        "scientific_run_valid": scientific_run_valid,
        "hypothesis_passed": hypothesis,
        "preregistration": {"sha256": prereg_sha, "path": str(args.preregistration)},
        "source_closure": source,
        "input_receipt": inputs,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0), "compute_capability": "9.0", "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "geometry": {"layer_index": 3, "calibration_hidden_shape": list(calibration_hidden.shape), "heldout_hidden_shape": list(heldout_hidden.shape), "projection_shape": list(original_k.shape), "key_value_heads": 2, "head_dim": 256},
        "whitener_receipt": whitener_receipt,
        "rows": rows,
        "gates": {"all_outputs_finite": finite, "all_projection_errors_strictly_decrease_with_rank": monotone},
        "hypothesis": {"activation_whitening_improves_over_plain_svd_at_every_rank_for_both_key_and_value": hypothesis},
        "sidecars": sidecars,
        "claim_boundary": prereg["not_authorized"],
    }
    summary_path = args.output_dir / "palu-whiten-transfer-summary.json"
    summary_path.write_bytes(canonical_json(summary))
    ledger = [f"{sha256(path)}  {path.name}" for path in sorted(args.output_dir.iterdir()) if path.is_file()]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(ledger) + "\n")
    print(json.dumps({"output": str(summary_path), "sha256": sha256(summary_path), "scientific_run_valid": scientific_run_valid, "hypothesis_passed": hypothesis}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except PaluTransferError as error:
        print(json.dumps({"status": "failed_closed", "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2)
