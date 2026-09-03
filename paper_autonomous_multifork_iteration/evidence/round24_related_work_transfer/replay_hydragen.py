#!/usr/bin/env python3
"""Blind replay for the bounded Hydragen-on-Qwen3.5 operator result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sidecar(root: Path, record: dict) -> torch.Tensor:
    path = root / record["relative_path"]
    assert path.is_file()
    assert path.stat().st_size == record["bytes"]
    assert sha256(path) == record["sha256"]
    assert record["dtype"] == "torch.float32"
    assert record["encoding"] == "torch-contiguous-raw-little-endian-v1"
    data = bytearray(path.read_bytes())
    tensor = torch.frombuffer(data, dtype=torch.float32).clone()
    return tensor.reshape(record["shape"])


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    delta = candidate - reference
    denominator = torch.linalg.vector_norm(reference).item()
    return {
        "finite": bool(torch.isfinite(candidate).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": float(
            torch.linalg.vector_norm(delta).item() / max(denominator, 1e-30)
        ),
        "argmax_head_dimension_exact": bool(
            torch.equal(candidate.argmax(dim=-1), reference.argmax(dim=-1))
        ),
    }


NUMERIC_REPLAY_ABS_TOLERANCE = 1e-7


def close(recorded: dict, replayed: dict) -> bool:
    for key in ("finite", "argmax_head_dimension_exact"):
        if recorded[key] != replayed[key]:
            return False
    return all(
        abs(recorded[key] - replayed[key]) <= NUMERIC_REPLAY_ABS_TOLERANCE
        for key in ("max_abs", "mean_abs", "relative_l2")
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary_path = args.result_dir / "hydragen-transfer-summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["schema_version"] == "qcomem-related-hydragen-transfer-result-v1"

    ledger_ok = True
    for line in (args.result_dir / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        ledger_ok &= sha256(args.result_dir / name) == expected

    cells = []
    for cell in summary["cells"]:
        sidecars = cell["sidecars"]
        oracle = load_sidecar(args.result_dir, sidecars["cpu_fp32_oracle"])
        dense = load_sidecar(args.result_dir, sidecars["replicated_dense_output"])
        hydragen = load_sidecar(args.result_dir, sidecars["hydragen_output"])
        replayed = {
            "replicated_dense_vs_cpu_fp32": metrics(dense, oracle),
            "hydragen_vs_cpu_fp32": metrics(hydragen, oracle),
            "hydragen_vs_replicated_dense": metrics(hydragen, dense),
        }
        metric_match = all(
            close(cell["metrics"][name], value) for name, value in replayed.items()
        )
        gates = {
            "dense_vs_cpu_fp32_relative_l2": replayed[
                "replicated_dense_vs_cpu_fp32"
            ]["relative_l2"]
            <= 0.005,
            "hydragen_vs_cpu_fp32_relative_l2": replayed[
                "hydragen_vs_cpu_fp32"
            ]["relative_l2"]
            <= 0.005,
            "hydragen_vs_dense_relative_l2": replayed[
                "hydragen_vs_replicated_dense"
            ]["relative_l2"]
            <= 0.005,
            "all_outputs_finite": bool(
                replayed["replicated_dense_vs_cpu_fp32"]["finite"]
                and replayed["hydragen_vs_cpu_fp32"]["finite"]
                and replayed["hydragen_vs_replicated_dense"]["finite"]
            ),
        }
        cells.append(
            {
                "resident_requests": cell["resident_requests"],
                "metric_match": metric_match,
                "gate_match": gates == cell["gates"],
                "passed": metric_match and gates == cell["gates"] and all(gates.values()),
                "replayed_metrics": replayed,
            }
        )

    report = {
        "schema_version": "qcomem-related-hydragen-transfer-replay-v1",
        "summary_sha256": sha256(summary_path),
        "ledger_ok": ledger_ok,
        "cells": cells,
        "passed": ledger_ok and len(cells) == 2 and all(row["passed"] for row in cells),
        "numeric_replay_abs_tolerance": NUMERIC_REPLAY_ABS_TOLERANCE,
        "timing_boundary": "Latency samples are producer-recorded CUDA-event measurements; this replay validates tensor bytes, metrics, and gates, not GPU timing recapture.",
    }
    args.output.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "passed": report["passed"]}))


if __name__ == "__main__":
    main()
