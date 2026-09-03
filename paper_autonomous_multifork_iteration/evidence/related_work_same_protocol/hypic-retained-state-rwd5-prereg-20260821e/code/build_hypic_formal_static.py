#!/usr/bin/env python3
"""Build and replay the pre-output authority for the matched HYPIC run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


HYPIC_COMMIT = "98147c01909004e66d98bcb18b886927d41b0ee5"
DATA_SHA256 = "1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe"
MODEL_LEDGER_SHA256 = "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014"
MODEL_ARTIFACT_LEDGER_SHA256 = "d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd"
EXPECTED_PAIRS = [
    ["qasper", 6],
    ["qasper", 7],
    ["qasper", 8],
    ["qasper", 9],
    ["2wikimqa", 6],
    ["2wikimqa", 7],
    ["2wikimqa", 8],
    ["2wikimqa", 9],
]
MODES = ["full_recompute", "prefix_cache", "transition_rope_recompute"]


class StaticError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise StaticError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_bytes(value))
    os.replace(temporary, path)


def git_lines(repo: Path, *arguments: str) -> list[str]:
    output = subprocess.check_output(["git", "-C", str(repo), *arguments], text=True)
    return [line for line in output.splitlines() if line]


def build_source_ledger(repo: Path) -> dict[str, Any]:
    head = git_lines(repo, "rev-parse", "HEAD")
    require(head == [HYPIC_COMMIT], "HYPIC commit drift")
    require(
        not git_lines(repo, "status", "--porcelain", "--untracked-files=all"),
        "HYPIC source is dirty or has untracked files",
    )
    names = git_lines(repo, "ls-files")
    require(bool(names) and names == sorted(names), "tracked source order")
    rows = []
    for name in names:
        path = repo / name
        require(path.is_file(), f"tracked source is not a file: {name}")
        rows.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "schema": "hypic-official-source-ledger-v1",
        "official_repository": "https://github.com/redai-infra/HYPIC",
        "official_commit": HYPIC_COMMIT,
        "tracked_worktree_clean": True,
        "untracked_files_absent": True,
        "file_count": len(rows),
        "files": rows,
    }


def distribution_row(name: str) -> dict[str, str]:
    distribution = importlib.metadata.distribution(name)
    record_candidates = [
        entry for entry in (distribution.files or []) if str(entry).endswith(".dist-info/RECORD")
    ]
    require(len(record_candidates) == 1, f"{name} RECORD")
    record_path = Path(distribution.locate_file(record_candidates[0]))
    require(record_path.is_file(), f"{name} RECORD file")
    return {
        "name": name,
        "version": distribution.version,
        "base_version": distribution.version.split("+")[0],
        "record_path": str(record_candidates[0]),
        "record_sha256": sha256_file(record_path),
    }


def nvidia_inventory() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows = []
    for line in output.splitlines():
        index, uuid, name, memory, driver = (part.strip() for part in line.split(",", 4))
        rows.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_mib": int(memory),
                "driver_version": driver,
            }
        )
    require(
        len(rows) == 8
        and [row["index"] for row in rows] == list(range(8))
        and len({row["uuid"] for row in rows}) == 8
        and all(row["name"] == "NVIDIA H20-3e" for row in rows)
        and all(row["memory_mib"] == 143771 for row in rows),
        "formal 8xH20-3e inventory drift",
    )
    return rows


def build_environment_ledger() -> dict[str, Any]:
    import torch

    packages = [
        distribution_row(name)
        for name in (
            "torch",
            "sglang-kernel",
            "sgl-deep-gemm",
            "transformers",
            "sglang",
            "flashinfer-python",
            "flashinfer-cubin",
        )
    ]
    expected_bases = {
        "torch": "2.11.0",
        "sglang-kernel": "0.4.4",
        "sgl-deep-gemm": "0.1.3",
        "transformers": "5.8.1",
        "sglang": "0.5.14",
        "flashinfer-python": "0.5.3",
        "flashinfer-cubin": "0.5.3",
    }
    require(
        {row["name"]: row["base_version"] for row in packages} == expected_bases,
        "package base-version drift",
    )
    require(torch.version.cuda == "12.9", "torch CUDA runtime drift")
    require(
        importlib.util.find_spec("flashinfer") is not None,
        "CUDA 12.9-compatible FlashInfer is unavailable",
    )
    subprocess.check_call(
        [
            os.environ.get("PYTHON", os.sys.executable),
            "-c",
            "import sgl_kernel, torch; assert torch.version.cuda == '12.9'",
        ]
    )
    python_path = Path(os.sys.executable).resolve()
    return {
        "schema": "hypic-formal-environment-ledger-v1",
        "python_executable": str(python_path),
        "python_sha256": sha256_file(python_path),
        "torch_cuda": str(torch.version.cuda),
        "packages": packages,
        "hardware": nvidia_inventory(),
        "runtime_environment": {
            "PIC_SEAM_SINK": "8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "SGLANG_NUMA_BIND_V2": "0",
            "SGLANG_IS_FLASHINFER_AVAILABLE": "0",
            "sampling_backend": "pytorch",
            "flashinfer_compatibility_version": "0.5.3",
            "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "unset",
        },
    }


def parse_model_ledger(
    path: Path,
    model: Path,
    *,
    expected_raw_sha256: str,
    expected_count: int,
    label: str,
    verify_bytes: bool,
) -> list[dict[str, Any]]:
    require(sha256_file(path) == expected_raw_sha256, f"{label} ledger raw SHA")
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        name = name.strip()
        target = model / name
        info = target.stat()
        require(stat.S_ISREG(info.st_mode), f"model shard not regular: {name}")
        require(info.st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0, f"writable model shard: {name}")
        if verify_bytes:
            require(sha256_file(target) == digest, f"model shard SHA drift: {name}")
        rows.append(
            {
                "path": name,
                "sha256": digest,
                "bytes": info.st_size,
                "mode": stat.S_IMODE(info.st_mode),
            }
        )
    require(len(rows) == expected_count, f"{label} file count")
    return rows


def build_preregistration(args: argparse.Namespace, source: dict[str, Any], environment: dict[str, Any]) -> dict[str, Any]:
    code = {}
    for name in ("client", "helper", "test", "launcher", "static_builder"):
        path = getattr(args, name)
        require(path.is_file(), f"missing {name}")
        code[name] = {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {
        "schema": "hypic-same-protocol-preregistration-v1",
        "status": "frozen_before_model_outputs",
        "research_question": "How does official HYPIC compare under the same Qwen3.5-35B-A3B LongBench slice and streaming timing boundary?",
        "official_repository": "https://github.com/redai-infra/HYPIC",
        "official_commit": HYPIC_COMMIT,
        "source_ledger_sha256": hashlib.sha256(canonical_bytes(source)).hexdigest(),
        "environment_ledger_sha256": hashlib.sha256(canonical_bytes(environment)).hexdigest(),
        "code": code,
        "model": {
            "path": str(args.model),
            "weight_ledger_raw_sha256": MODEL_LEDGER_SHA256,
            "artifact_ledger_raw_sha256": MODEL_ARTIFACT_LEDGER_SHA256,
            "shard_count": 14,
            "artifact_count": 9,
            "processor_config_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
            "full_bytes_verified_pre_output": True,
        },
        "data": {"path": str(args.data), "sha256": DATA_SHA256, "pairs": EXPECTED_PAIRS},
        "design": {
            "modes": MODES,
            "rank_count": 8,
            "one_distinct_H20_per_rank": True,
            "tensor_parallel_size": 1,
            "input_cap": 4096,
            "max_new_tokens": 32,
            "decoding": "greedy_temperature_0_seed_20260821",
            "timing_boundary": "OpenAI-compatible streaming client wall clock",
            "discarded_warmup": "prefix-disjoint, exact-length; cache modes also warm prime+reuse path",
            "formal_cell": "fresh server/cache, one document prime when applicable, then exactly one measured request",
            "prefix_cache_boundary": "strictly document tokens only",
            "hypic_boundary": "three segments; last dummy/real query diverges immediately; PIC_SEAM_SINK=8",
            "mode_order": MODES,
        },
        "resolved_server_contract": {
            "common": ["--tp=1", "--enable-cache-report", "--disable-overlap-schedule"],
            "full_recompute": ["--disable-radix-cache", "--mamba-radix-cache-strategy=no_buffer"],
            "prefix_cache": ["radix-cache-enabled", "--mamba-radix-cache-strategy=extra_buffer"],
            "transition_rope_recompute": [
                "--page-size=1",
                "--chunked-prefill-size=-1",
                "--mamba-radix-cache-strategy=no_buffer",
                "--pic-enable",
                "--pic-mode=transition_rope_recompute",
                "--pic-separator-str=<<PIC_SEP>>",
                "PIC_SEAM_SINK=8",
            ],
        },
        "outcomes": {
            "protocol_validity": "all bindings, token identities, cache boundaries, and 24 cells complete",
            "accuracy": "report per-row and mean F1; no predeclared pass threshold",
            "performance": "report TTFT, TPOT, and completion_tokens/client_e2e; no predeclared pass threshold",
            "approximation": "HYPIC is approximate; preserve all output differences versus full recompute",
        },
    }


def materialize(args: argparse.Namespace, *, verify_model_bytes: bool) -> dict[str, bytes]:
    require(sha256_file(args.data) == DATA_SHA256, "data SHA drift")
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
    source = build_source_ledger(args.repo)
    environment = build_environment_ledger()
    preregistration = build_preregistration(args, source, environment)
    return {
        "source-ledger.json": canonical_bytes(source),
        "environment-ledger.json": canonical_bytes(environment),
        "preregistration.json": canonical_bytes(preregistration),
    }


def build_stage(args: argparse.Namespace) -> None:
    outputs = materialize(args, verify_model_bytes=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, data in outputs.items():
        (args.output_dir / name).write_bytes(data)
    ledger = {
        name: hashlib.sha256(data).hexdigest() for name, data in sorted(outputs.items())
    }
    atomic_json(
        args.output_dir / "preoutput-validation.json",
        {
            "schema": "hypic-preoutput-validation-v1",
            "passed": True,
            "full_model_bytes_verified": True,
            "files": ledger,
        },
    )


def verify_stage(args: argparse.Namespace) -> None:
    outputs = materialize(args, verify_model_bytes=args.verify_model_bytes)
    for name, expected in outputs.items():
        observed = (args.output_dir / name).read_bytes()
        require(observed == expected, f"terminal static drift: {name}")
    atomic_json(
        args.validation_output,
        {
            "schema": "hypic-terminal-static-verification-v1",
            "passed": True,
            "model_bytes_rehashed": bool(args.verify_model_bytes),
            "files": {
                name: hashlib.sha256(data).hexdigest() for name, data in sorted(outputs.items())
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("build", "verify"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-weight-ledger", type=Path, required=True)
    parser.add_argument("--model-artifact-ledger", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--client", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--static-builder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--verify-model-bytes", action="store_true")
    args = parser.parse_args()
    if args.stage == "build":
        build_stage(args)
    else:
        require(args.validation_output is not None, "validation output")
        verify_stage(args)


if __name__ == "__main__":
    main()
