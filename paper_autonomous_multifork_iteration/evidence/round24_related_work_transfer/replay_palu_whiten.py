#!/usr/bin/env python3
"""Blind replay of the bounded Palu whitening transfer from raw sidecars."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "palu_whiten_formal"
SUMMARY = RUN / "palu-whiten-transfer-summary.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def tensor_from_receipt(receipt: dict[str, object]) -> torch.Tensor:
    path = RUN / str(receipt["relative_path"])
    raw = path.read_bytes()
    require(len(raw) == receipt["bytes"], f"byte count drift: {path.name}")
    require(hashlib.sha256(raw).hexdigest() == receipt["sha256"], f"SHA drift: {path.name}")
    dtype = {"torch.bfloat16": torch.bfloat16, "torch.float32": torch.float32, "torch.int64": torch.int64}[str(receipt["dtype"])]
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(tuple(receipt["shape"])).clone()


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    delta = candidate.float() - reference.float()
    return float(torch.linalg.vector_norm(delta).item() / max(torch.linalg.vector_norm(reference.float()).item(), 1e-30))


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    require(summary["schema_version"] == "qcomem-related-palu-whiten-transfer-result-v1", "result schema drift")
    declared = {}
    for line in (RUN / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        declared[name] = digest
        require(sha256(RUN / name) == digest, f"ledger drift: {name}")
    require(set(declared) == {path.name for path in RUN.iterdir() if path.is_file() and path.name != "SHA256SUMS"}, "ledger coverage drift")
    sidecars = summary["sidecars"]
    original_key = tensor_from_receipt(sidecars["original_key_bf16"])
    original_value = tensor_from_receipt(sidecars["original_value_bf16"])
    tensor_from_receipt(sidecars["calibration_hidden_bf16"])
    tensor_from_receipt(sidecars["heldout_hidden_bf16"])
    tensor_from_receipt(sidecars["whitener_fp32"])
    for index in range(9):
        token_ids = tensor_from_receipt(sidecars[f"token_ids_row{index}"])
        require(hashlib.sha256(token_ids.contiguous().numpy().tobytes()).hexdigest() == summary["input_receipt"]["token_rows"][str(index)]["sha256"], f"token replay drift row {index}")

    recomputed = []
    maximum_metric_absolute_drift = 0.0
    for row in summary["rows"]:
        rank = row["rank_per_kv_head"]
        values = {}
        finite = True
        for method, prefix in (("plain_svd", "plain"), ("activation_whitened", "whiten")):
            key = tensor_from_receipt(sidecars[f"rank{rank}_{prefix}_key"])
            value = tensor_from_receipt(sidecars[f"rank{rank}_{prefix}_value"])
            finite = finite and bool(torch.isfinite(key).all() and torch.isfinite(value).all())
            values[method] = {"key": relative_l2(key, original_key), "value": relative_l2(value, original_value)}
            for kind in ("key", "value"):
                drift = abs(values[method][kind] - row["heldout_projection_relative_l2"][method][kind])
                maximum_metric_absolute_drift = max(maximum_metric_absolute_drift, drift)
                require(math.isclose(values[method][kind], row["heldout_projection_relative_l2"][method][kind], rel_tol=0.0, abs_tol=1e-5), f"metric drift rank {rank} {method} {kind}")
        expected_dense = 2048
        expected_latent = 8 * rank
        storage = row["logical_storage"]
        require(storage["dense_kv_bytes_per_token"] == expected_dense, f"dense storage drift rank {rank}")
        require(storage["palu_latent_kv_bytes_per_token"] == expected_latent, f"latent storage drift rank {rank}")
        require(finite and row["all_outputs_finite"], f"nonfinite rank {rank}")
        recomputed.append({"rank_per_kv_head": rank, "metrics": values})

    methods = (("plain_svd", "key"), ("plain_svd", "value"), ("activation_whitened", "key"), ("activation_whitened", "value"))
    monotone = all(all(left > right for left, right in zip(values, values[1:])) for method, kind in methods for values in [[row["metrics"][method][kind] for row in recomputed]])
    hypothesis = all(row["metrics"]["activation_whitened"][kind] < row["metrics"]["plain_svd"][kind] for row in recomputed for kind in ("key", "value"))
    require(monotone, "rank monotonicity replay failed")
    require(hypothesis, "whitening improvement replay failed")
    require(summary["scientific_run_valid"] is True and summary["hypothesis_passed"] is True, "producer classification drift")
    report = {
        "schema_version": "qcomem-related-palu-whiten-replay-v1",
        "passed": True,
        "summary_sha256": sha256(SUMMARY),
        "sidecar_count": len(sidecars),
        "scientific_run_valid_recomputed": True,
        "hypothesis_passed_recomputed": True,
        "portable_cpu_vs_gpu_reduction_absolute_tolerance": 1e-5,
        "maximum_observed_metric_absolute_drift": maximum_metric_absolute_drift,
        "rows_recomputed_from_raw_sidecars": recomputed,
    }
    output = ROOT / "palu_whiten_replay_report.json"
    output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sha256": sha256(output), "passed": True}, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"passed": False, "error": str(error)}, sort_keys=True), file=sys.stderr)
        raise
