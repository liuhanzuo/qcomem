#!/usr/bin/env python3
"""Build and verify the frozen, GPU-independent R39 preregistration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "forkaudit-qwen35-dense-transformers-transfer-v1"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
A4_STATIC_SHA256 = "ccd1ef0bcd9dc98c0d8fc871326162bfc0bcb6cf2e6beb7599af04433b023cb0"
A4_AGGREGATE_SHA256 = "33f9acb87baaf15fd62e74e39cd5c57260f626554be35542c122317dffdfc4da"
A4_QCOMEM_SHA256 = "5901f153fcfcabbfab63f756a3c19a04ace56b4985fc02421f2dde4118a7373c"

DOCUMENT_TOKENS = 64
QUERY_TOKENS = 8
WORLD_SIZE = 8
FANOUTS = (1, 2)
DEPTH = 7
SEMANTIC_STEPS = 2
VOCAB_SIZE = 248320
STORAGE_SALT = hashlib.sha256(
    b"r39-qwen35-dense-private-storage-domain-v1"
).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def int64_le_sha256(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        require(0 <= int(value) < VOCAB_SIZE, "token outside registered vocabulary")
        digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def default_paths(repo_root: Path) -> dict[str, Path]:
    evidence = repo_root / "paper_autonomous_multifork_iteration" / "evidence"
    a4 = evidence / "round6_a4_transformers_transfer_20260819b"
    return {
        "a4_static": a4 / "preregistration" / "static-preregistration.json",
        "a4_aggregate": a4 / "results" / "forkaudit-transformers-transfer-aggregate.json",
        "a4_qcomem": a4 / "executed_source" / "qcomem_torch.py",
    }


def build_static(repo_root: Path) -> dict[str, Any]:
    paths = default_paths(repo_root)
    require(sha256_file(paths["a4_static"]) == A4_STATIC_SHA256, "A4 static drift")
    require(
        sha256_file(paths["a4_aggregate"]) == A4_AGGREGATE_SHA256,
        "A4 aggregate drift",
    )
    require(sha256_file(paths["a4_qcomem"]) == A4_QCOMEM_SHA256, "A4 code drift")
    a4 = json.loads(paths["a4_static"].read_text(encoding="utf-8"))
    rank_inputs = a4.get("rank_inputs")
    require(isinstance(rank_inputs, list) and len(rank_inputs) == WORLD_SIZE, "A4 rank count drift")

    selected: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    for expected_rank, row in enumerate(rank_inputs):
        require(row.get("rank") == expected_rank, "A4 rank ordering drift")
        source_id = str(row["source_id"])
        require(source_id not in source_ids, "A4 source reuse")
        source_ids.add(source_id)
        document = [int(value) for value in row["document_token_ids"][:DOCUMENT_TOKENS]]
        require(len(document) == DOCUMENT_TOKENS, "short document unavailable")
        queries = []
        for expected_request, query in enumerate(row["queries"][: max(FANOUTS)]):
            require(query["request_index"] == expected_request, "A4 request ordering drift")
            tokens = [int(value) for value in query["token_ids"][:QUERY_TOKENS]]
            require(len(tokens) == QUERY_TOKENS, "short query unavailable")
            queries.append(
                {
                    "request_index": expected_request,
                    "source_token_offset": int(query["source_token_offset"]),
                    "token_ids": tokens,
                    "token_ids_int64_le_sha256": int64_le_sha256(tokens),
                }
            )
        require(len(queries) == 2, "two queries are required")
        selected.append(
            {
                "rank": expected_rank,
                "source_id": source_id,
                "source_object": row["source_object"],
                "document_start_token": int(row["document_start_token"]),
                "document_token_ids": document,
                "document_token_ids_int64_le_sha256": int64_le_sha256(document),
                "queries": queries,
            }
        )

    rank_inputs_sha256 = sha256_bytes(canonical_bytes(selected))
    formal_config = {
        "world_size": WORLD_SIZE,
        "fanouts": list(FANOUTS),
        "document_tokens": DOCUMENT_TOKENS,
        "query_tokens": QUERY_TOKENS,
        "split_depth": DEPTH,
        "semantic_steps": SEMANTIC_STEPS,
        "scheduler": "one-process-per-gpu-single-stream-step-request-interleave",
        "arms": ["deep_materialized", "persistent_q16"],
        "q16": {
            "residual_bits": 16,
            "attention_bits": 16,
            "linear_bits": 16,
            "group_size": 64,
        },
    }
    return {
        "schema_version": "r39-second-model-transfer-prereg-v1",
        "protocol": PROTOCOL,
        "created_before_gpu_execution": True,
        "model": {
            "repo_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "revision_is_full_commit": True,
            "text_only": True,
            "expected_geometry": {
                "model_type": "qwen3_5_text",
                "num_hidden_layers": 24,
                "hidden_size": 1024,
                "vocab_size": VOCAB_SIZE,
                "dense_no_experts": True,
                "layer_types": [
                    item
                    for _block in range(6)
                    for item in (
                        "linear_attention",
                        "linear_attention",
                        "linear_attention",
                        "full_attention",
                    )
                ],
            },
            "official_revision_tree": (
                "https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/"
                + MODEL_REVISION
            ),
        },
        "runtime": {
            "transformers_version": "5.14.1",
            "cache": "transformers.cache_utils.DynamicCache",
            "weight_dtype": "torch.bfloat16",
            "trust_remote_code": False,
            "primary_vllm_stack_used": False,
        },
        "formal_config": formal_config,
        "formal_config_sha256": sha256_bytes(canonical_bytes(formal_config)),
        "input_authority": {
            "derivation": (
                "first 64 document and first 8 query token IDs from each "
                "immutable A4 rank; no reselection"
            ),
            "a4_static_sha256": A4_STATIC_SHA256,
            "a4_aggregate_sha256": A4_AGGREGATE_SHA256,
            "distinct_source_ids": len(source_ids),
        },
        "rank_inputs": selected,
        "rank_inputs_sha256": rank_inputs_sha256,
        "storage_receipt_salt": STORAGE_SALT,
        "reference_contract": {
            "official_one_shot_path": (
                "AutoModelForImageTextToText text-only forward, use_cache=False"
            ),
            "official_cached_path": (
                "language_model DynamicCache document prefill then query/decode"
            ),
            "manual_one_shot_validation": {
                "top1_exact_required": True,
                "relative_l2_threshold": 0.001,
            },
            "standard_cache_validation": {
                "top1_exact_required": True,
                "relative_l2_threshold": 0.005,
            },
            "split_vs_authorized_reference": {
                "top1_exact_required": True,
                "relative_l2_threshold": 0.005,
            },
            "full_vocabulary_cpu_fp32_sidecars_required": True,
            "reference_is_conditionally_authorized": True,
        },
        "semantic_contract": {
            "cross_arm": "exact generated token and exact FP32 sidecar bytes",
            "cross_n": "exact request-0 trajectory within arm",
            "vocabulary": VOCAB_SIZE,
        },
        "controls": [
            {
                "control_id": "MUTABLE_CACHE_ALIAS",
                "expected_first_failing_predicate": "PRIVATE_MUTABLE_STORAGE",
            },
            {
                "control_id": "PREFIX_CONTENT_MUTATION",
                "expected_first_failing_predicate": "PERSISTENT_PREFIX_IMMUTABLE",
            },
            {
                "control_id": "POSITION_OFFSET_DRIFT",
                "expected_first_failing_predicate": "POSITION_CANONICAL",
            },
            {
                "control_id": "DENSE_MASK_ROUTE_RELABEL",
                "expected_first_failing_predicate": "LAYER_TYPE_MASK_ROUTE",
            },
        ],
        "targets": [
            {"index": 1, "name": "frozen_identity", "maximum_status": "full"},
            {"index": 2, "name": "prefix_immutability", "maximum_status": "full"},
            {"index": 3, "name": "private_ownership", "maximum_status": "full"},
            {
                "index": 4,
                "name": "tail_safe_append",
                "maximum_status": "not_applicable",
                "exact_missingness": [
                    "DynamicCache has no fixed-size paged partial tail",
                    "no page-level copy-before-append event exists",
                ],
            },
            {
                "index": 5,
                "name": "dispatch_provenance",
                "maximum_status": "partial",
                "exact_missingness": [
                    "compiled CUDA/Triton kernel binary fingerprint",
                    "kernel autotuning-choice fingerprint",
                    "hardware instruction trace",
                ],
            },
            {"index": 6, "name": "cross_arm_equivalence", "maximum_status": "full"},
            {"index": 7, "name": "cross_n_consistency", "maximum_status": "full"},
        ],
        "claim_boundary": {
            "authorized_if_positive": (
                "bounded second-model/second-runtime ownership and relational "
                "transfer on the registered short-input cells"
            ),
            "not_authorized": [
                "vision path",
                "paged-tail safety",
                "compiled dispatch",
                "continuous batching or concurrency",
                "latency, throughput, capacity, or memory saving",
                "scheduler or production portability",
                "other models or revisions",
            ],
        },
    }


def verify_static(static: dict[str, Any], rebuilt: dict[str, Any]) -> None:
    require(static == rebuilt, "static preregistration is not a bytewise rebuild")
    require(static["model"]["revision"] == MODEL_REVISION, "model revision drift")
    require(static["formal_config"]["world_size"] == WORLD_SIZE, "world size drift")
    require(static["formal_config"]["fanouts"] == list(FANOUTS), "fanout drift")


def source_rows(repo_root: Path, relative_paths: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for raw in sorted(relative_paths):
        relative = Path(raw)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        text = relative.as_posix()
        require(text not in seen, "duplicate source path")
        seen.add(text)
        path = repo_root / relative
        require(path.is_file() and not path.is_symlink(), f"missing regular source: {text}")
        rows.append(
            {
                "path": text,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def verify_source_manifest(
    repo_root: Path,
    manifest_path: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    require(re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256) is not None, "bad source SHA")
    raw = manifest_path.read_bytes()
    require(sha256_bytes(raw) == expected_manifest_sha256, "source manifest raw drift")
    manifest = json.loads(raw)
    require(manifest.get("schema_version") == "r39-second-model-transfer-source-v1", "source schema drift")
    rows = manifest.get("files")
    require(isinstance(rows, list) and rows, "empty source manifest")
    actual = source_rows(repo_root, [row["path"] for row in rows])
    require(actual == rows, "source file hash/size drift")
    return {
        "schema_version": "r39-second-model-transfer-source-verification-v1",
        "manifest_sha256": expected_manifest_sha256,
        "file_count": len(rows),
        "verified": True,
    }


def gpu_assignment() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    candidates = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        require(len(parts) == 4, "unexpected nvidia-smi assignment row")
        index, uuid, name, memory_mib = parts
        if "H20" in name:
            candidates.append(
                {
                    "visible_index": int(index),
                    "uuid": uuid,
                    "name": name,
                    "total_memory_mib": int(memory_mib),
                }
            )
    candidates.sort(key=lambda row: row["visible_index"])
    require(len(candidates) >= WORLD_SIZE, "fewer than eight H20 devices are available")
    selected = candidates[:WORLD_SIZE]
    require(len({row["uuid"] for row in selected}) == WORLD_SIZE, "GPU UUID reuse")
    rows = [{"rank": rank, **row} for rank, row in enumerate(selected)]
    return {
        "schema_version": "r39-second-model-transfer-gpu-assignment-v1",
        "selection": "first eight H20 devices in numeric visible-index order",
        "available_h20_count": len(candidates),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)

    build = subparsers.add_parser("build-static")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify-static")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--static", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--output", type=Path, required=True)

    source = subparsers.add_parser("build-source")
    source.add_argument("--repo-root", type=Path, required=True)
    source.add_argument("--path", action="append", required=True)
    source.add_argument("--output", type=Path, required=True)

    verify_source = subparsers.add_parser("verify-source")
    verify_source.add_argument("--repo-root", type=Path, required=True)
    verify_source.add_argument("--manifest", type=Path, required=True)
    verify_source.add_argument("--expected-sha256", required=True)
    verify_source.add_argument("--output", type=Path, required=True)

    gpu = subparsers.add_parser("gpu-assignment")
    gpu.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "build-static":
        payload = build_static(args.repo_root.resolve())
        atomic_write(args.output, canonical_bytes(payload))
    elif args.stage == "verify-static":
        raw = args.static.read_bytes()
        require(sha256_bytes(raw) == args.expected_sha256, "static raw SHA drift")
        static = json.loads(raw)
        rebuilt = build_static(args.repo_root.resolve())
        verify_static(static, rebuilt)
        receipt = {
            "schema_version": "r39-second-model-transfer-static-verification-v1",
            "static_sha256": args.expected_sha256,
            "rebuilt_sha256": sha256_bytes(canonical_bytes(rebuilt)),
            "verified": True,
        }
        atomic_write(args.output, canonical_bytes(receipt))
    elif args.stage == "build-source":
        rows = source_rows(args.repo_root.resolve(), args.path)
        manifest = {
            "schema_version": "r39-second-model-transfer-source-v1",
            "protocol": PROTOCOL,
            "files": rows,
            "file_count": len(rows),
            "normalized_files_sha256": sha256_bytes(canonical_bytes(rows)),
        }
        atomic_write(args.output, canonical_bytes(manifest))
    elif args.stage == "verify-source":
        receipt = verify_source_manifest(
            args.repo_root.resolve(),
            args.manifest,
            args.expected_sha256,
        )
        atomic_write(args.output, canonical_bytes(receipt))
    elif args.stage == "gpu-assignment":
        atomic_write(args.output, canonical_bytes(gpu_assignment()))
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()

