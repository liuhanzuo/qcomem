from __future__ import annotations

"""Build and independently rebuild the Transformers transfer preregistration."""

import argparse
import hashlib
import json
import platform
import stat as statlib
import subprocess
from pathlib import Path
from typing import Any, Sequence

import torch

from qcomem_transformers_forkaudit_transfer import (
    FAULT_CONTRACT,
    FANOUTS,
    PROTOCOL,
    SOURCE_SCHEMA,
    STATIC_SCHEMA,
    TARGET_CONTRACT,
    WORLD_SIZE,
    canonical_json_bytes,
    load_bound_json,
    parse_sha256_ledger_metadata,
    require,
    sha256_file,
    sha256_bytes,
    sha256_json,
    tensor_receipt,
    validate_sha256_ledger,
    validate_source_manifest,
    write_canonical_json,
)


MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
PG19_DATA_SHA256 = "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
PG19_MANIFEST_SHA256 = "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"
PG19_BUCKET = "deepmind-gutenberg"
PG19_PREFIX = "train/"
DOCUMENT_TOKENS = 256
QUERY_TOKENS = 24
QUERY_STRIDE = 32
WINDOW_STRIDE = 197
CANDIDATE_WINDOWS = 8
SEED = 20260819
SPLIT_DEPTH = 7
SEMANTIC_STEPS = 2
ORACLE_RELATIVE_L2_THRESHOLD = 0.005
STORAGE_RECEIPT_SALT = "forkaudit-transformers-transfer-storage-domain-20260819a"
LAYER_TYPES = tuple(
    "full_attention" if index in range(3, 40, 4) else "linear_attention"
    for index in range(40)
)

SOURCE_FILES = (
    "qcomem_torch.py",
    "qcomem_transformers_forkaudit_transfer.py",
    "run_qcomem_transformers_forkaudit_transfer.py",
    "build_qcomem_transformers_forkaudit_transfer_prereg.py",
    "test_qcomem_transformers_forkaudit_transfer.py",
    "launch_qcomem_transformers_forkaudit_transfer_8gpu.sh",
    "FORKAUDIT_TRANSFORMERS_TRANSFER_PROTOCOL.md",
)


def _load_pg19(data_path: Path, manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require(sha256_file(data_path) == PG19_DATA_SHA256, "PG-19 JSONL SHA drift")
    require(sha256_file(manifest_path) == PG19_MANIFEST_SHA256, "PG-19 manifest SHA drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(
        manifest.get("bucket") == PG19_BUCKET
        and manifest.get("prefix") == PG19_PREFIX
        and manifest.get("test_or_validation_objects_used") is False
        and manifest.get("jsonl_sha256") == PG19_DATA_SHA256,
        "PG-19 manifest provenance drift",
    )
    listed = {
        row["name"]: row
        for row in manifest.get("objects", [])
        if type(row) is dict and type(row.get("name")) is str
    }
    records = []
    seen = set()
    with data_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            require(
                not any(key in row for key in ("dataset", "context", "input", "answers", "_source_index")),
                f"PG-19 line {line_number} has evaluation schema",
            )
            source = row.get("_source_object")
            require(
                row.get("_source_bucket") == PG19_BUCKET
                and type(source) is str
                and source.startswith(PG19_PREFIX)
                and source.endswith(".txt")
                and source not in seen,
                f"PG-19 line {line_number} source drift",
            )
            require(type(row.get("text")) is str and row["text"], "PG-19 text is missing")
            require(source in listed and listed[source].get("md5_base64") == row.get("_source_md5_base64"), "PG-19 GCS identity drift")
            seen.add(source)
            records.append(row)
    require(set(listed) == seen and len(records) >= WORLD_SIZE, "PG-19 manifest/JSONL object set drift")
    return records, {
        "bucket": PG19_BUCKET,
        "prefix": PG19_PREFIX,
        "records": len(records),
        "data_sha256": PG19_DATA_SHA256,
        "manifest_sha256": PG19_MANIFEST_SHA256,
        "test_or_validation_objects_used": False,
    }


def _book_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(f"{SEED}|transformers-transfer-book|{row['_source_object']}".encode()).hexdigest()


def _window_index(source: str, complete: int) -> int:
    digest = hashlib.sha256(f"{SEED}|transformers-transfer-window|{source}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % complete


def _bounded_tokens(tokenizer: Any, text: str, required: int) -> list[int]:
    guard = required + 64
    character_limit = min(len(text), max(guard * 6, 8192))
    while True:
        ids = tokenizer.encode(text[:character_limit], add_special_tokens=False)
        if len(ids) >= guard or character_limit == len(text):
            return list(ids)
        character_limit = min(len(text), character_limit * 2)


def build_rank_inputs(records: Sequence[dict[str, Any]], tokenizer: Any) -> list[dict[str, Any]]:
    rank_inputs = []
    required_window = (
        DOCUMENT_TOKENS
        + QUERY_TOKENS
        + (max(FANOUTS) - 1) * QUERY_STRIDE
        + QUERY_TOKENS
    )
    maximum = required_window + WINDOW_STRIDE * (CANDIDATE_WINDOWS - 1)
    for record in sorted(records, key=_book_key):
        source = str(record["_source_object"])
        ids = _bounded_tokens(tokenizer, record["text"], maximum)
        complete = min(CANDIDATE_WINDOWS, max(0, (len(ids) - required_window) // WINDOW_STRIDE + 1))
        if complete < 1:
            continue
        start = _window_index(source, complete) * WINDOW_STRIDE
        document_ids = ids[start : start + DOCUMENT_TOKENS]
        bank_start = start + DOCUMENT_TOKENS + QUERY_TOKENS
        query_rows = []
        for request_index in range(max(FANOUTS)):
            offset = bank_start + request_index * QUERY_STRIDE
            token_ids = ids[offset : offset + QUERY_TOKENS]
            tensor = torch.tensor([token_ids], dtype=torch.int64)
            query_rows.append(
                {
                    "request_index": request_index,
                    "source_token_offset": offset,
                    "token_ids": token_ids,
                    "token_ids_sha256": tensor_receipt(tensor)["content_sha256"],
                }
            )
        rank = len(rank_inputs)
        base = {
            "rank": rank,
            "source_id": str(record.get("id", Path(source).stem)),
            "source_object": source,
            "document_start_token": start,
            "document_end_token_exclusive": start + DOCUMENT_TOKENS,
            "document_token_ids": document_ids,
            "document_token_ids_sha256": tensor_receipt(
                torch.tensor([document_ids], dtype=torch.int64)
            )["content_sha256"],
            "queries": query_rows,
        }
        rank_inputs.append({**base, "rank_input_sha256": sha256_json(base)})
        if len(rank_inputs) == WORLD_SIZE:
            break
    require(len(rank_inputs) == WORLD_SIZE, "could not construct eight complete PG-19 windows")
    require(len({row["source_object"] for row in rank_inputs}) == WORLD_SIZE, "PG-19 books are not distinct")
    return rank_inputs


def _formal_config() -> dict[str, Any]:
    return {
        "world_size": WORLD_SIZE,
        "pg19_train_books": WORLD_SIZE,
        "document_tokens": DOCUMENT_TOKENS,
        "query_tokens": QUERY_TOKENS,
        "fanouts": list(FANOUTS),
        "split_depth": SPLIT_DEPTH,
        "semantic_steps": SEMANTIC_STEPS,
        "window_stride": WINDOW_STRIDE,
        "query_stride": QUERY_STRIDE,
        "candidate_windows_per_book": CANDIDATE_WINDOWS,
        "seed": SEED,
        "scheduler": "single-cuda-stream-request-index-interleaved",
        "arms": ["deep_materialized", "persistent_fork"],
    }


def model_authority(
    model_root: Path,
    artifact_ledger: Path,
    weight_ledger: Path,
    *,
    artifact_sha256: str,
    weight_sha256: str,
) -> dict[str, Any]:
    artifact = validate_sha256_ledger(
        model_root,
        artifact_ledger,
        expected_raw_sha256=artifact_sha256,
        label="model artifact ledger",
    )
    weight = validate_sha256_ledger(
        model_root,
        weight_ledger,
        expected_raw_sha256=weight_sha256,
        label="model weight ledger",
    )
    require(weight["file_count"] == 14, "expected fourteen weight shards")
    stats = []
    for entry in [*artifact["entries"], *weight["entries"]]:
        path = model_root / entry["path"]
        stat_result = path.stat()
        no_write_mode_bits = (
            stat_result.st_mode & (statlib.S_IWUSR | statlib.S_IWGRP | statlib.S_IWOTH)
        ) == 0
        require(statlib.S_ISREG(stat_result.st_mode), f"model artifact is not a regular file: {entry['path']}")
        require(no_write_mode_bits, f"model artifact has a write mode bit: {entry['path']}")
        stats.append(
            {
                "path": entry["path"],
                "bytes": stat_result.st_size,
                "device": stat_result.st_dev,
                "inode": stat_result.st_ino,
                "ctime_ns": stat_result.st_ctime_ns,
                "regular_file": True,
                "no_write_mode_bits": no_write_mode_bits,
            }
        )
    return {
        "schema_version": "forkaudit-transformers-model-authority-v2",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "artifact_ledger": artifact,
        "weight_ledger": weight,
        "stat_snapshot": stats,
        "full_file_sha256_verified": True,
    }


def build_static(
    args: argparse.Namespace,
    *,
    frozen_model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    source_raw_sha = sha256_file(args.source_manifest)
    require(source_raw_sha == args.expected_source_manifest_sha256, "source manifest raw SHA drift")
    source = load_bound_json(args.source_manifest, source_raw_sha, "source manifest")
    validate_source_manifest(args.source_root, source)
    records, data_receipt = _load_pg19(args.pg19_data, args.pg19_manifest)
    import transformers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )
    rank_inputs = build_rank_inputs(records, tokenizer)
    if frozen_model is None:
        authority = model_authority(
            args.model,
            args.model_artifact_ledger,
            args.model_weight_ledger,
            artifact_sha256=args.expected_model_artifact_ledger_sha256,
            weight_sha256=args.expected_model_weight_ledger_sha256,
        )
        artifact_receipt = authority["artifact_ledger"]
        weight_receipt = authority["weight_ledger"]
    else:
        require(
            frozen_model["model_artifact_ledger_raw_sha256"]
            == args.expected_model_artifact_ledger_sha256
            and frozen_model["model_weight_ledger_raw_sha256"]
            == args.expected_model_weight_ledger_sha256,
            "frozen model ledger binding drift",
        )
        require(sha256_file(args.model_artifact_ledger) == args.expected_model_artifact_ledger_sha256, "artifact ledger raw SHA drift")
        require(sha256_file(args.model_weight_ledger) == args.expected_model_weight_ledger_sha256, "weight ledger raw SHA drift")
        artifact_receipt = parse_sha256_ledger_metadata(
            args.model, args.model_artifact_ledger,
            expected_raw_sha256=args.expected_model_artifact_ledger_sha256,
            label="model artifact ledger",
        )
        weight_receipt = parse_sha256_ledger_metadata(
            args.model, args.model_weight_ledger,
            expected_raw_sha256=args.expected_model_weight_ledger_sha256,
            label="model weight ledger",
        )
        require(
            artifact_receipt == frozen_model["artifact_ledger_receipt"]
            and weight_receipt == frozen_model["weight_ledger_receipt"],
            "frozen model receipts do not match parsed ledger metadata",
        )
    config = _formal_config()
    return {
        "schema_version": STATIC_SCHEMA,
        "protocol": PROTOCOL,
        "created_before_gpu_execution": True,
        "source_manifest_raw_sha256": source_raw_sha,
        "formal_config": config,
        "formal_config_sha256": sha256_json(config),
        "dataset": data_receipt,
        "window_algorithm": {
            "implementation": "independent-bounded-PG19-raw-token-windows-v1",
            "selection_key": "sha256(seed|transformers-transfer-book|source_object)",
            "window_key": "sha256(seed|transformers-transfer-window|source_object)",
            "raw_int64_token_receipts": True,
        },
        "rank_inputs": rank_inputs,
        "rank_inputs_sha256": sha256_json(rank_inputs),
        "model": {
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "model_artifact_ledger_raw_sha256": args.expected_model_artifact_ledger_sha256,
            "model_weight_ledger_raw_sha256": args.expected_model_weight_ledger_sha256,
            "artifact_ledger_receipt": artifact_receipt,
            "weight_ledger_receipt": weight_receipt,
            "artifact_set_sha256": sha256_json(artifact_receipt["entries"]),
            "layer_types": list(LAYER_TYPES),
            "tokenizer_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        },
        "environment_contract": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "hardware_contract": {
            "world_size": WORLD_SIZE,
            "gpu_name": "NVIDIA H20-3e",
            "compute_capability": [9, 0],
            "bf16_required": True,
            "assignment_frozen_pre_output": True,
        },
        "storage_receipt_salt": STORAGE_RECEIPT_SALT,
        "oracle_contract": {
            "path": "standard Transformers dense full-model recompute from frozen raw token IDs",
            "independent_of_ownership_arms": True,
            "full_vocabulary_cpu_fp32_sidecars_required": True,
            "top1_exact_required": True,
            "relative_l2_threshold": ORACLE_RELATIVE_L2_THRESHOLD,
        },
        "target_contract": list(TARGET_CONTRACT),
        "fault_contract": list(FAULT_CONTRACT),
        "portable_record_mapping": {
            "identity": "adapted: frozen PG19/tokenizer/model/source/config/authority receipts",
            "ownership": "adapted: normalized contiguous Transformers-cache storage byte ranges at setup, first transition, and final",
            "execution": "adapted: per-call lower/suffix layer, position, length, append, content, and storage ledger",
            "accounting": "adapted context-only: CUDA allocated/reserved/max snapshots at setup, first transition, and final",
            "tail_event": "not_applicable: DynamicCache has no fixed-size paged partial-tail event",
            "dispatch": "partial: Python adapter/layer callables; compiled kernel and autotune fingerprints unavailable",
        },
        "claim_boundary": {
            "same_model": f"{MODEL_ID}@{MODEL_REVISION}",
            "different_runtime": "Transformers DynamicCache through qcomem_torch.TorchSplitCausalLM",
            "tail_target": "not_applicable: no paged partial tail in DynamicCache",
            "dispatch_target": "partial: Python callable/class provenance only",
            "not_authorized": [
                "second-model transfer",
                "compiled-kernel identity",
                "paged-tail transfer",
                "production scheduler or concurrency claims",
                "latency, throughput, capacity, or NVML claims",
            ],
        },
    }


def build_source_manifest(source_root: Path) -> dict[str, Any]:
    rows = []
    for relative in SOURCE_FILES:
        path = source_root / relative
        require(path.is_file(), f"source file is missing: {relative}")
        rows.append({"path": relative, "sha256": sha256_file(path), "bytes": path.stat().st_size})
    normalized = [{"path": row["path"], "sha256": row["sha256"]} for row in rows]
    return {
        "schema_version": SOURCE_SCHEMA,
        "protocol": PROTOCOL,
        "files": rows,
        "normalized_files_sha256": sha256_json(normalized),
    }


def build_gpu_assignment() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    rows = []
    for line in result.stdout.strip().splitlines():
        fields = [item.strip() for item in line.split(",")]
        require(len(fields) == 5, "GPU assignment query schema drift")
        visible_index, uuid, name, memory, capability = fields
        major_minor = [int(item) for item in capability.split(".")]
        rows.append(
            {
                "rank": len(rows),
                "visible_index": int(visible_index),
                "uuid": uuid,
                "name": name,
                "total_memory_mib": int(memory),
                "compute_capability": major_minor,
                "bf16_supported": major_minor >= [8, 0],
            }
        )
    require(len(rows) == WORLD_SIZE, "formal node must expose exactly eight GPUs")
    require(len({row["uuid"] for row in rows}) == WORLD_SIZE, "GPU UUIDs are not unique")
    for rank, row in enumerate(rows):
        require(
            row["rank"] == rank
            and row["visible_index"] == rank
            and row["name"] == "NVIDIA H20-3e"
            and row["compute_capability"] == [9, 0]
            and row["bf16_supported"] is True,
            "formal H20 assignment drift",
        )
    return {
        "schema_version": "forkaudit-transformers-gpu-assignment-v1",
        "world_size": WORLD_SIZE,
        "hardware_contract": "NVIDIA H20-3e / compute capability 9.0 / BF16",
        "rows": rows,
        "rows_sha256": sha256_json(rows),
    }


def verify_static_rebuild(frozen: dict[str, Any], rebuilt: dict[str, Any]) -> dict[str, Any]:
    require(
        canonical_json_bytes(rebuilt) == canonical_json_bytes(frozen),
        "static manifest does not bytewise rebuild from PG-19/tokenizer/model inputs",
    )
    return {
        "schema_version": "forkaudit-transformers-static-rebuild-receipt-v1",
        "verified": True,
        "static_manifest_raw_sha256": sha256_bytes(canonical_json_bytes(frozen) + b"\n"),
        "rank_inputs_sha256": frozen["rank_inputs_sha256"],
        "pg19_data_sha256": PG19_DATA_SHA256,
        "pg19_manifest_sha256": PG19_MANIFEST_SHA256,
        "model_artifact_set_sha256": frozen["model"]["artifact_set_sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("source", "verify-source", "static", "verify-static", "model-authority", "gpu-assignment"), required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--expected-source-manifest-sha256", default="")
    parser.add_argument("--pg19-data", type=Path)
    parser.add_argument("--pg19-manifest", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-artifact-ledger", type=Path)
    parser.add_argument("--model-weight-ledger", type=Path)
    parser.add_argument("--expected-model-artifact-ledger-sha256", default="")
    parser.add_argument("--expected-model-weight-ledger-sha256", default="")
    parser.add_argument("--static-manifest", type=Path)
    parser.add_argument("--expected-static-manifest-sha256", default="")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.stage == "gpu-assignment":
        value = build_gpu_assignment()
    elif args.stage == "source":
        require(args.source_root is not None, "source root is required")
        value = build_source_manifest(args.source_root)
    elif args.stage == "verify-source":
        require(args.source_root is not None and args.source_manifest is not None, "source inputs are required")
        source = load_bound_json(args.source_manifest, args.expected_source_manifest_sha256, "source manifest")
        value = validate_source_manifest(args.source_root, source)
    elif args.stage == "model-authority":
        require(all(item is not None for item in (args.model, args.model_artifact_ledger, args.model_weight_ledger)), "model authority inputs are required")
        value = model_authority(
            args.model,
            args.model_artifact_ledger,
            args.model_weight_ledger,
            artifact_sha256=args.expected_model_artifact_ledger_sha256,
            weight_sha256=args.expected_model_weight_ledger_sha256,
        )
    else:
        require(
            all(
                item is not None
                for item in (
                    args.source_root,
                    args.source_manifest,
                    args.pg19_data,
                    args.pg19_manifest,
                    args.model,
                    args.model_artifact_ledger,
                    args.model_weight_ledger,
                )
            ),
            "static build inputs are required",
        )
        if args.stage == "verify-static":
            frozen = load_bound_json(args.static_manifest, args.expected_static_manifest_sha256, "static manifest")
            rebuilt = build_static(args, frozen_model=frozen["model"])
            value = verify_static_rebuild(frozen, rebuilt)
            require(value["static_manifest_raw_sha256"] == args.expected_static_manifest_sha256, "static rebuild raw SHA drift")
        else:
            rebuilt = build_static(args)
            value = rebuilt
    write_canonical_json(args.output, value)
    print(json.dumps(value, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
