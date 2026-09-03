#!/usr/bin/env python3
"""Build and bytewise verify the GPU-independent Falcon-H1 R39 freeze."""

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


PROTOCOL = "forkaudit-falcon-h1-hybrid-transformers-transfer-v1"
MODEL_ID = "tiiuae/Falcon-H1-0.5B-Base"
HF_REVISION = "59fb76e8c5d3fc7441b062be638e1ba0afd5c687"
MS_REVISION = "a475c769e108fd1dc6cfe41e342305d36431ef20"
MODEL_ENDPOINT = "https://modelscope.cn"
WEIGHT_SHA256 = "865a1e864b3fe6495ec37256e1fdec8cd1d254b607eab29141e7263791172ce6"
TOKENIZER_SHA256 = "605c664925653e3fbf2f35ea063847db441ba5b7a6af04378880409c3ab311fc"
HF_TREE_SHA256 = "9ad0d35e2b8824ff0007089055f5cd061bf8e4146fe4a118b8054f9ce458aca6"
MS_TREE_SHA256 = "4e1fb677f62f3830393907729f705955abc6eb90533f876424b42786bb3a65a4"
CROSS_SOURCE_SHA256 = "11dac596848fd338b086cb7cce6dc537dc9ad2f49648eb5ce17ee6511b3a8c80"
PG19_INPUTS_SHA256 = "d4c8341c74e4b0e2ee0969208d7a3912f86cd1fb4a1f1a906b0243944342fd4c"
A4_QCOMEM_SHA256 = "5901f153fcfcabbfab63f756a3c19a04ace56b4985fc02421f2dde4118a7373c"
OFFICIAL_MODELING_SHA256 = "e90bf774524e9b66284ad1c5528c35339271a187f58f16ba2d45c97f4bc6b5bd"
OFFICIAL_CACHE_UTILS_SHA256 = "ee7902fbd031ed332b5e26d07756a33f09b5c90a435b8363b9330876dc33ce0e"
OFFICIAL_MASKING_UTILS_SHA256 = "5f48e428ea02d1b6008acb45c147fcdb4eba89deea69627744662aa05da1b9f2"

WORLD_SIZE = 8
FANOUTS = (1, 2)
DOCUMENT_TOKENS = 64
QUERY_TOKENS = 8
DEPTH = 18
SEMANTIC_STEPS = 2
VOCAB_SIZE = 32784
TOKENIZER_VOCAB_SIZE = 32768
STATE_FAMILIES = ("kv_key", "kv_value", "conv", "mamba2_recurrent")
STORAGE_SALT = hashlib.sha256(b"r39-falcon-h1-private-mutable-storage-v1").hexdigest()
DOWNLOAD_ATTEMPTS = 12


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
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


def load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"{label} absent")
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"{label} raw SHA-256 drift")
    value = json.loads(raw)
    require(isinstance(value, dict), f"{label} root is not an object")
    return value


def int64_le_sha256(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        require(0 <= int(value) < VOCAB_SIZE, "token outside Falcon model vocabulary")
        digest.update(struct.pack("<q", int(value)))
    return digest.hexdigest()


def package_paths(repo_root: Path) -> dict[str, Path]:
    package = (
        repo_root
        / "paper_autonomous_multifork_iteration"
        / "evidence"
        / "r39_falcon_h1_transfer"
    )
    return {
        "package": package,
        "hf_tree": package / "preregistration" / "huggingface-tree.json",
        "ms_tree": package / "preregistration" / "modelscope-tree.json",
        "cross": package / "preregistration" / "cross-source-equivalence.json",
        "inputs": package / "preregistration" / "pg19-tokenized-inputs.json",
        "a4_qcomem": (
            repo_root
            / "paper_autonomous_multifork_iteration"
            / "evidence"
            / "round6_a4_transformers_transfer_20260819b"
            / "executed_source"
            / "qcomem_torch.py"
        ),
    }


def validate_model_authorities(repo_root: Path) -> dict[str, Any]:
    paths = package_paths(repo_root)
    hf = load_bound_json(paths["hf_tree"], HF_TREE_SHA256, "Hugging Face tree")
    ms = load_bound_json(paths["ms_tree"], MS_TREE_SHA256, "ModelScope tree")
    cross = load_bound_json(paths["cross"], CROSS_SOURCE_SHA256, "cross-source manifest")
    require(hf["official_source"]["repo_id"] == MODEL_ID, "HF model ID drift")
    require(hf["official_source"]["revision"] == HF_REVISION, "HF revision drift")
    require(ms["official_source"]["repo_id"] == MODEL_ID, "ModelScope model ID drift")
    require(ms["official_source"]["revision"] == MS_REVISION, "ModelScope revision drift")
    require(hf["file_count"] == len(hf["files"]) == 8, "HF tree count drift")
    require(ms["file_count"] == len(ms["files"]) == 9, "ModelScope tree count drift")
    hf_rows = {row["path"]: row for row in hf["files"]}
    ms_rows = {row["path"]: row for row in ms["files"]}
    require(hf_rows["model.safetensors"]["sha256"] == WEIGHT_SHA256, "HF weight drift")
    require(ms_rows["model.safetensors"]["sha256"] == WEIGHT_SHA256, "MS weight drift")
    require(hf_rows["tokenizer.json"]["sha256"] == TOKENIZER_SHA256, "HF tokenizer drift")
    require(ms_rows["tokenizer.json"]["sha256"] == TOKENIZER_SHA256, "MS tokenizer drift")
    exact = cross["common_exact_files"]
    require(cross["common_exact_file_count"] == len(exact) == 7, "cross-source count drift")
    for row in exact:
        path = row["path"]
        require(path in hf_rows and path in ms_rows, f"cross-source path absent: {path}")
        require(
            hf_rows[path]["size"] == ms_rows[path]["size"] == row["bytes"]
            and hf_rows[path]["sha256"] == ms_rows[path]["sha256"] == row["sha256"],
            f"cross-source equivalence drift: {path}",
        )
    require(cross["scientific_load_files_exact_across_sources"] is True, "scientific file equivalence failed")
    require(sha256_file(paths["a4_qcomem"]) == A4_QCOMEM_SHA256, "A4 Q16 code drift")
    return {"hf": hf, "ms": ms, "cross": cross}


def acquisition_policy() -> dict[str, Any]:
    return {
        "endpoint": MODEL_ENDPOINT,
        "endpoint_is_explicit": True,
        "official_namespace": "tiiuae",
        "token": False,
        "token_policy": "public-no-token",
        "transport": "modelscope-official-revision-pinned-per-file-http200-restart-from-zero",
        "canonical_huggingface_revision": HF_REVISION,
        "canonical_huggingface_revision_is_full_commit": True,
        "modelscope_revision": MS_REVISION,
        "modelscope_revision_is_full_commit": True,
        "weight_equivalence_sha256": WEIGHT_SHA256,
        "tokenizer_equivalence_sha256": TOKENIZER_SHA256,
        "huggingface_tree_sha256": HF_TREE_SHA256,
        "modelscope_tree_sha256": MS_TREE_SHA256,
        "frozen_tree_sha256": MS_TREE_SHA256,
        "cross_source_equivalence_sha256": CROSS_SOURCE_SHA256,
        "remote_tree_must_equal_frozen_manifest": True,
        "per_file_size_and_sha256_required": True,
        "restart_from_zero_per_attempt": True,
        "independent_attempt_temp_files": True,
        "range_requests_forbidden": True,
        "append_to_partial_forbidden": True,
        "full_response_http_status": 200,
        "content_length_exact_total_required": True,
        "max_attempts_per_file": DOWNLOAD_ATTEMPTS,
        "fresh_nonoverwriting_snapshot_required": True,
    }


def build_rank_inputs(repo_root: Path) -> list[dict[str, Any]]:
    manifest = load_bound_json(
        package_paths(repo_root)["inputs"],
        PG19_INPUTS_SHA256,
        "PG-19 Falcon tokenization",
    )
    require(manifest["schema_version"] == "r39-falcon-h1-pg19-tokenized-inputs-v1", "input schema drift")
    tokenizer = manifest["tokenizer"]
    require(tokenizer["repo_id"] == MODEL_ID and tokenizer["revision"] == HF_REVISION, "input tokenizer identity drift")
    require(tokenizer["sha256"] == TOKENIZER_SHA256, "input tokenizer hash drift")
    require(tokenizer["derivation_library_version"] == "0.22.2", "tokenizer derivation version drift")
    require(tokenizer["add_special_tokens"] is False, "special-token policy drift")
    rule = manifest["window_rule"]
    require(rule["document"] == {"start_token": 197, "length": 64}, "document window drift")
    require(
        rule["queries"]
        == [
            {"request_index": 0, "start_token": 477, "length": 8},
            {"request_index": 1, "start_token": 509, "length": 8},
        ],
        "query windows drift",
    )
    require(rule["no_out_of_vocabulary_filtering_or_reselection"] is True, "input reselection drift")
    rows = manifest["rows"]
    require(len(rows) == WORLD_SIZE, "PG-19 source count drift")
    selected = []
    source_ids = set()
    for rank, row in enumerate(rows):
        require(row["rank"] == rank, "input rank ordering drift")
        require(row["source_id"] not in source_ids, "PG-19 source reuse")
        source_ids.add(row["source_id"])
        document = [int(value) for value in row["document_token_ids"]]
        require(len(document) == DOCUMENT_TOKENS, "document token count drift")
        require(all(0 <= value < VOCAB_SIZE for value in document), "Falcon document token OOV")
        queries = []
        for request_index, query in enumerate(row["queries"]):
            tokens = [int(value) for value in query["token_ids"]]
            require(query["request_index"] == request_index, "query ordering drift")
            require(len(tokens) == QUERY_TOKENS, "query token count drift")
            require(all(0 <= value < VOCAB_SIZE for value in tokens), "Falcon query token OOV")
            queries.append(
                {
                    "request_index": request_index,
                    "source_token_offset": int(query["source_token_offset"]),
                    "token_ids": tokens,
                    "token_ids_int64_le_sha256": int64_le_sha256(tokens),
                }
            )
        require(len(queries) == 2, "two query windows required")
        selected.append(
            {
                "rank": rank,
                "source_id": row["source_id"],
                "source_object": row["source_object"],
                "source_url": row["source_url"],
                "source_bytes": row["source_bytes"],
                "source_sha256": row["source_sha256"],
                "source_full_falcon_token_count": row["full_token_count"],
                "document_start_token": row["document_start_token"],
                "document_token_ids": document,
                "document_token_ids_int64_le_sha256": int64_le_sha256(document),
                "queries": queries,
            }
        )
    require(len(source_ids) == WORLD_SIZE, "distinct PG-19 source gate failed")
    return selected


def build_static(repo_root: Path) -> dict[str, Any]:
    validate_model_authorities(repo_root)
    rank_inputs = build_rank_inputs(repo_root)
    formal_config = {
        "world_size": WORLD_SIZE,
        "fanouts": list(FANOUTS),
        "document_tokens": DOCUMENT_TOKENS,
        "query_tokens": QUERY_TOKENS,
        "split_depth": DEPTH,
        "semantic_steps": SEMANTIC_STEPS,
        "chunk_schedule": [64, 8, 1],
        "scheduler": "one-process-per-gpu; reference-then-candidate; fixed request interleave",
        "arms": ["deep_materialized", "persistent_q16"],
        "q16": {
            "residual_bits": 16,
            "attention_bits": 16,
            "linear_bits": 16,
            "group_size": 64,
            "lossless_only": True,
            "no_differential_family_quantization_claim": True,
        },
    }
    controls = [
        {"control_id": "MUTABLE_CACHE_ALIAS", "expected_first_failing_predicate": "PRIVATE_MUTABLE_STORAGE"},
        {"control_id": "STATE_FAMILY_OMISSION", "expected_first_failing_predicate": "STATE_FAMILY_COMPLETENESS"},
        {"control_id": "POSITION_OFFSET_DRIFT", "expected_first_failing_predicate": "POSITION_CANONICAL"},
        {"control_id": "STATE_FAMILY_RELABEL", "expected_first_failing_predicate": "STATE_FAMILY_BINDING"},
        {"control_id": "REFERENCE_CANDIDATE_IMPORT", "expected_first_failing_predicate": "REFERENCE_IMPLEMENTATION_INDEPENDENT"},
    ]
    return {
        "schema_version": "r39-falcon-h1-transfer-prereg-v1",
        "protocol": PROTOCOL,
        "created_before_falcon_gpu_execution": True,
        "model": {
            "repo_id": MODEL_ID,
            "revision": HF_REVISION,
            "revision_is_full_commit": True,
            "expected_geometry": {
                "model_type": "falcon_h1",
                "num_hidden_layers": 36,
                "hidden_size": 1024,
                "model_vocab_size": VOCAB_SIZE,
                "tokenizer_vocab_size": TOKENIZER_VOCAB_SIZE,
                "layer_types": ["hybrid"] * 36,
                "state_families_per_layer": list(STATE_FAMILIES),
                "expected_state_family_count_full_model": 144,
                "kv_shape_template": [1, 2, "sequence_length", 64],
                "conv_shape": [1, 1792, 4],
                "mamba2_recurrent_shape": [1, 24, 64, 128],
            },
        },
        "model_acquisition": acquisition_policy(),
        "runtime": {
            "registered_image_label": "vllm-cu129-v1",
            "transformers_version": "5.14.1",
            "cache": "transformers.cache_utils.DynamicCache",
            "weight_dtype": "torch.bfloat16",
            "attention_implementation": "eager",
            "mamba_dispatch": "official FalconH1Mixer.torch_forward",
            "hub_kernels_environment": "USE_HUB_KERNELS=NO",
            "hub_kernels_disabled_before_transformers_import": True,
            "fast_path_forced_false_after_model_init_before_any_forward": True,
            "package_installs_mamba_causal_conv_or_flash_dependencies": False,
            "trust_remote_code": False,
            "official_source_sha256": {
                "modeling_falcon_h1.py": OFFICIAL_MODELING_SHA256,
                "cache_utils.py": OFFICIAL_CACHE_UTILS_SHA256,
                "masking_utils.py": OFFICIAL_MASKING_UTILS_SHA256,
            },
        },
        "formal_config": formal_config,
        "formal_config_sha256": sha256_bytes(canonical_bytes(formal_config)),
        "input_authority": {
            "dataset": "PG-19 train",
            "distinct_sources": 8,
            "tokenized_inputs_manifest_sha256": PG19_INPUTS_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "tokenizer_revision": HF_REVISION,
            "tokenizers_derivation_version": "0.22.2",
            "fixed_window_rule": {"document": [197, 261], "queries": [[477, 485], [509, 517]]},
            "no_oov_filtering_or_reselection": True,
        },
        "rank_inputs": rank_inputs,
        "rank_inputs_sha256": sha256_bytes(canonical_bytes(rank_inputs)),
        "storage_receipt_salt": STORAGE_SALT,
        "reference_contract": {
            "implementation": "candidate-import-free AutoModelForCausalLM + official DynamicCache",
            "same_chunk_schedule": [64, 8, 1],
            "full_vocabulary_cpu_fp32_sidecars_required": True,
            "generated_token_ids_exact": True,
            "full_fp32_logit_bytes_exact": True,
            "max_abs_threshold": 0.0,
            "relative_l2_threshold": 0.0,
            "all_144_state_family_content_sha256_exact_per_request_step": True,
            "candidate_dynamic_import_exec_eval_forbidden": True,
        },
        "semantic_contract": {
            "cross_arm": "exact tokens, full FP32 logits, and 144 family content hashes at every step",
            "candidate_vs_official": "exact tokens, full FP32 logits, and 144 family content hashes at every step",
            "cross_n": "exact request-0 trajectory and family contents within arm",
            "zero_numeric_tolerance": True,
        },
        "controls": controls,
        "persistent_base_detector": {
            "before_after_full_content_sha256_exact": True,
            "prefix_content_mutation_would_fail": "PERSISTENT_PREFIX_IMMUTABLE",
        },
        "claim_boundary": {
            "authorized_if_positive": (
                "bounded exact relational transfer on Falcon-H1-0.5B-Base under one frozen "
                "Transformers 5.14.1/H20 naive-path configuration"
            ),
            "not_authorized": [
                "runtime independence or model-family generality",
                "optional Mamba/causal-conv/Flash kernel paths",
                "compiled dispatch",
                "latency, throughput, capacity, or memory-saving claims",
                "long-context quality or benchmark effectiveness",
                "production scheduling, concurrency, or continuous batching",
                "other Falcon revisions or architectures",
            ],
        },
    }


def verify_static(static: dict[str, Any], rebuilt: dict[str, Any]) -> None:
    require(static == rebuilt, "static preregistration is not a bytewise rebuild")
    require(static["formal_config"]["split_depth"] == 18, "depth drift")
    require(static["model"]["expected_geometry"]["num_hidden_layers"] == 36, "layer count drift")
    require(len(static["controls"]) == 5, "control count drift")
    for row in static["rank_inputs"]:
        values = [*row["document_token_ids"]]
        for query in row["queries"]:
            values.extend(query["token_ids"])
        require(all(0 <= value < VOCAB_SIZE for value in values), "static contains OOV token")


def source_rows(repo_root: Path, relative_paths: Iterable[str]) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for raw in sorted(relative_paths):
        relative = Path(raw)
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        text = relative.as_posix()
        require(text not in seen, "duplicate source path")
        seen.add(text)
        path = repo_root / relative
        require(path.is_file() and not path.is_symlink(), f"missing source: {text}")
        rows.append({"path": text, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    require(rows, "empty source manifest")
    return rows


def build_source(repo_root: Path, paths: Iterable[str]) -> dict[str, Any]:
    rows = source_rows(repo_root, paths)
    return {
        "schema_version": "r39-falcon-h1-transfer-source-v1",
        "protocol": PROTOCOL,
        "files": rows,
        "file_count": len(rows),
        "normalized_files_sha256": sha256_bytes(canonical_bytes(rows)),
    }


def verify_source(repo_root: Path, manifest: Path, expected_sha256: str) -> dict[str, Any]:
    raw = manifest.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, "source manifest raw drift")
    value = json.loads(raw)
    require(value["schema_version"] == "r39-falcon-h1-transfer-source-v1", "source schema drift")
    actual = source_rows(repo_root, [row["path"] for row in value["files"]])
    require(actual == value["files"] and value["file_count"] == len(actual), "source file drift")
    return {"schema_version": "r39-falcon-h1-source-verification-v1", "manifest_sha256": expected_sha256, "file_count": len(actual), "verified": True}


def build_freeze(package_root: Path) -> dict[str, Any]:
    static = package_root / "preregistration" / "static-preregistration.json"
    source = package_root / "preregistration" / "source-manifest.json"
    require(static.is_file() and source.is_file(), "static/source manifests absent")
    return {
        "schema_version": "r39-falcon-h1-transfer-freeze-v1",
        "protocol": PROTOCOL,
        "package_date": "2026-08-27",
        "static_manifest_sha256": sha256_file(static),
        "source_manifest_sha256": sha256_file(source),
        "huggingface_tree_sha256": HF_TREE_SHA256,
        "modelscope_tree_sha256": MS_TREE_SHA256,
        "cross_source_equivalence_sha256": CROSS_SOURCE_SHA256,
        "pg19_tokenized_inputs_sha256": PG19_INPUTS_SHA256,
        "official_transformers_source_sha256": {
            "modeling_falcon_h1.py": OFFICIAL_MODELING_SHA256,
            "cache_utils.py": OFFICIAL_CACHE_UTILS_SHA256,
            "masking_utils.py": OFFICIAL_MASKING_UTILS_SHA256,
        },
        "remote_paths": {
            "stage": "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench/qcomem_r39_falcon_h1_transfer_20260827a",
            "model": (
                "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/models/"
                "Falcon-H1-0.5B-Base-hf-59fb76e8-ms-a475c769-20260827a"
            ),
            "run": (
                "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/runs/qcomem/"
                "r39-falcon-h1-transfer-20260827a"
            ),
            "nonoverwriting_required": True,
        },
        "gpu_execution_status": "not_run_at_freeze",
        "scientific_outputs_inspected_before_freeze": False,
    }


def gpu_assignment() -> dict[str, Any]:
    output = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total", "--format=csv,noheader,nounits"],
        text=True,
    )
    candidates = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        require(len(parts) == 4, "unexpected nvidia-smi row")
        index, uuid, name, memory_mib = parts
        if "H20" in name:
            candidates.append({"visible_index": int(index), "uuid": uuid, "name": name, "total_memory_mib": int(memory_mib)})
    candidates.sort(key=lambda row: row["visible_index"])
    require(len(candidates) >= WORLD_SIZE, "fewer than eight H20 devices")
    selected = candidates[:WORLD_SIZE]
    require(len({row["uuid"] for row in selected}) == WORLD_SIZE, "GPU UUID reuse")
    return {
        "schema_version": "r39-falcon-h1-gpu-assignment-v1",
        "selection": "first eight H20 devices in numeric visible-index order",
        "rows": [{"rank": rank, **row} for rank, row in enumerate(selected)],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="stage", required=True)
    build = sub.add_parser("build-static")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify-static")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--static", type=Path, required=True)
    verify.add_argument("--expected-sha256", required=True)
    verify.add_argument("--output", type=Path, required=True)
    source = sub.add_parser("build-source")
    source.add_argument("--repo-root", type=Path, required=True)
    source.add_argument("--path", action="append", required=True)
    source.add_argument("--output", type=Path, required=True)
    verify_source_parser = sub.add_parser("verify-source")
    verify_source_parser.add_argument("--repo-root", type=Path, required=True)
    verify_source_parser.add_argument("--manifest", type=Path, required=True)
    verify_source_parser.add_argument("--expected-sha256", required=True)
    verify_source_parser.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("build-freeze")
    freeze.add_argument("--package-root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    verify_freeze = sub.add_parser("verify-freeze")
    verify_freeze.add_argument("--package-root", type=Path, required=True)
    verify_freeze.add_argument("--freeze", type=Path, required=True)
    verify_freeze.add_argument("--expected-sha256", required=True)
    verify_freeze.add_argument("--output", type=Path, required=True)
    gpu = sub.add_parser("gpu-assignment")
    gpu.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage == "build-static":
        atomic_write(args.output, canonical_bytes(build_static(args.repo_root.resolve())))
    elif args.stage == "verify-static":
        raw = args.static.read_bytes()
        require(sha256_bytes(raw) == args.expected_sha256, "static raw SHA drift")
        static = json.loads(raw)
        rebuilt = build_static(args.repo_root.resolve())
        verify_static(static, rebuilt)
        atomic_write(args.output, canonical_bytes({"schema_version": "r39-falcon-h1-static-verification-v1", "static_sha256": args.expected_sha256, "rebuilt_sha256": sha256_bytes(canonical_bytes(rebuilt)), "verified": True}))
    elif args.stage == "build-source":
        atomic_write(args.output, canonical_bytes(build_source(args.repo_root.resolve(), args.path)))
    elif args.stage == "verify-source":
        atomic_write(args.output, canonical_bytes(verify_source(args.repo_root.resolve(), args.manifest, args.expected_sha256)))
    elif args.stage == "build-freeze":
        atomic_write(args.output, canonical_bytes(build_freeze(args.package_root.resolve())))
    elif args.stage == "verify-freeze":
        raw = args.freeze.read_bytes()
        require(sha256_bytes(raw) == args.expected_sha256, "freeze raw SHA drift")
        rebuilt = build_freeze(args.package_root.resolve())
        require(json.loads(raw) == rebuilt, "freeze is not a bytewise rebuild")
        atomic_write(args.output, canonical_bytes({"schema_version": "r39-falcon-h1-freeze-verification-v1", "freeze_sha256": args.expected_sha256, "verified": True}))
    elif args.stage == "gpu-assignment":
        atomic_write(args.output, canonical_bytes(gpu_assignment()))
    else:
        raise AssertionError(args.stage)


if __name__ == "__main__":
    main()
