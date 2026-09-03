#!/usr/bin/env python3
"""Detached, torch/Transformers-free replay for R39 raw shards and logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


PROTOCOL = "forkaudit-qwen35-dense-transformers-transfer-v1"
HF_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODELSCOPE_REVISION = "4d58a7b524cd33ed843d5125be8cd8f0a452d9bf"
WORLD_SIZE = 8
FANOUTS = (1, 2)
VOCAB_SIZE = 248320


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to replace output: {path}")
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


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def verify_source(package_root: Path, freeze: dict[str, Any]) -> None:
    manifest_path = package_root / "preregistration" / "source-manifest.json"
    require(sha256_file(manifest_path) == freeze["source_manifest_sha256"], "source manifest drift")
    manifest = load_json(manifest_path)
    require(manifest["protocol"] == PROTOCOL, "source protocol drift")
    repo_root = package_root.parents[2]
    require(manifest["file_count"] == len(manifest["files"]), "source count drift")
    for row in manifest["files"]:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        path = repo_root / relative
        require(path.is_file() and not path.is_symlink(), f"source file absent: {relative}")
        require(path.stat().st_size == row["bytes"], f"source size drift: {relative}")
        require(sha256_file(path) == row["sha256"], f"source hash drift: {relative}")


def sidecar_records(shard: dict[str, Any], path: Path) -> tuple[bytes, dict[str, dict[str, Any]]]:
    receipt = shard["sidecar"]
    raw = path.read_bytes()
    require(len(raw) == receipt["bytes"], "sidecar byte count drift")
    require(sha256_bytes(raw) == receipt["sha256"], "sidecar terminal hash drift")
    records = receipt["records"]
    require(receipt["record_count"] == len(records) and records, "sidecar record count drift")
    by_id: dict[str, dict[str, Any]] = {}
    cursor = 0
    for row in records:
        require(row["record_id"] not in by_id, "duplicate sidecar record ID")
        require(row["offset_bytes"] == cursor, "sidecar has a gap or overlap")
        require(row["shape"] == [1, VOCAB_SIZE] and row["dtype"] == "float32-le", "sidecar type drift")
        require(row["nbytes"] == VOCAB_SIZE * 4, "sidecar record size drift")
        end = cursor + row["nbytes"]
        require(end <= len(raw), "sidecar record exceeds payload")
        record_raw = raw[cursor:end]
        require(sha256_bytes(record_raw) == row["content_sha256"], "record hash drift")
        values = memoryview(record_raw).cast("f")
        require(len(values) == VOCAB_SIZE, "record float count drift")
        argmax = 0
        maximum = -math.inf
        for index, value in enumerate(values):
            require(math.isfinite(value), "non-finite sidecar value")
            if value > maximum:
                maximum = value
                argmax = index
        require(argmax == row["argmax"], "sidecar argmax drift")
        by_id[row["record_id"]] = row
        cursor = end
    require(cursor == len(raw), "sidecar terminal coverage drift")
    closure = receipt["terminal_closure"]
    require(
        closure["exact_byte_coverage"] is True
        and closure["first_offset_bytes"] == 0
        and closure["last_end_offset_bytes"] == len(raw),
        "sidecar closure receipt drift",
    )
    return raw, by_id


def record_bytes(raw: bytes, row: dict[str, Any]) -> bytes:
    start = row["offset_bytes"]
    return raw[start : start + row["nbytes"]]


def numeric_comparison(
    candidate_raw: bytes,
    candidate: dict[str, Any],
    reference_raw: bytes,
    reference: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    left_bytes = record_bytes(candidate_raw, candidate)
    right_bytes = record_bytes(reference_raw, reference)
    left = memoryview(left_bytes).cast("f")
    right = memoryview(right_bytes).cast("f")
    squared_error = 0.0
    squared_reference = 0.0
    maximum = 0.0
    for a, b in zip(left, right):
        difference = float(a) - float(b)
        squared_error += difference * difference
        squared_reference += float(b) * float(b)
        maximum = max(maximum, abs(difference))
    relative = math.sqrt(squared_error) / max(math.sqrt(squared_reference), 1e-12)
    top1 = candidate["argmax"] == reference["argmax"]
    return {
        "candidate_record_id": candidate["record_id"],
        "reference_record_id": reference["record_id"],
        "top1_equal": top1,
        "max_abs": maximum,
        "relative_l2": relative,
        "relative_l2_threshold": threshold,
        "exact_bytes": left_bytes == right_bytes,
        "passed": top1 and relative <= threshold,
    }


def trajectory_comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    raw: bytes,
    by_id: dict[str, dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    require(candidate["request_index"] == reference["request_index"], "request mismatch")
    candidate_ids = candidate["step_logit_record_ids"]
    reference_ids = reference["step_logit_record_ids"]
    require(len(candidate_ids) == len(reference_ids) == 2, "trajectory step count drift")
    rows = [
        numeric_comparison(raw, by_id[left], raw, by_id[right], threshold=threshold)
        for left, right in zip(candidate_ids, reference_ids)
    ]
    tokens_equal = candidate["generated_token_ids"] == reference["generated_token_ids"]
    return {
        "request_index": candidate["request_index"],
        "generated_tokens_equal": tokens_equal,
        "rows": rows,
        "passed": tokens_equal and all(row["passed"] for row in rows),
    }


def validate_semantic_records(
    row: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
) -> set[str]:
    record_ids = row["step_logit_record_ids"]
    require(len(record_ids) == 2 and len(set(record_ids)) == 2, "semantic record IDs drift")
    require(all(record_id in by_id for record_id in record_ids), "semantic record absent")
    records = [by_id[record_id] for record_id in record_ids]
    require(
        row["step_logit_sha256"] == [record["content_sha256"] for record in records],
        "semantic logit hashes are not sidecar-derived",
    )
    require(
        row["generated_token_ids"] == [record["argmax"] for record in records],
        "semantic tokens are not sidecar-derived",
    )
    return set(record_ids)


def semantic_exact(left: dict[str, Any], right: dict[str, Any]) -> bool:
    keys = (
        "request_index",
        "query_token_ids_int64_le_sha256",
        "generated_token_ids",
        "step_logit_sha256",
        "final_lower_state_sha256",
        "final_lower_cache_content_sha256",
        "final_suffix_cache_content_sha256",
    )
    return all(left.get(key) == right.get(key) for key in keys)


def replay_ownership(receipt: dict[str, Any], *, require_peer: bool) -> bool:
    comparisons = receipt["comparisons"]
    require(receipt["comparison_count"] == len(comparisons) and comparisons, "ownership count drift")
    peer = 0
    tensor_pairs_positive = receipt["tensor_pair_comparison_count"] > 0
    passed = True
    for row in comparisons:
        disjoint = not row["overlap_ranges"]
        require(row["disjoint"] is disjoint, "ownership producer flag drift")
        passed = passed and disjoint
        if row["relation"] == "request_request":
            peer += 1
    require(peer == receipt["peer_comparison_count"], "peer comparison count drift")
    if require_peer:
        require(peer > 0, "N=2 ownership is vacuous")
    recomputed = passed and tensor_pairs_positive
    require(receipt["passed"] is recomputed, "ownership predicate flag drift")
    return recomputed


def validate_controls(shard: dict[str, Any], static: dict[str, Any]) -> bool:
    frozen = static["controls"]
    observed = shard["controls"]
    require(len(frozen) == len(observed) == 4, "control cardinality drift")
    all_passed = True
    for expected, row in zip(frozen, observed):
        require(row["control_id"] == expected["control_id"], "control ordering drift")
        predicate = expected["expected_first_failing_predicate"]
        require(row["expected_first_failing_predicate"] == predicate, "control predicate drift")
        clean_pass = row["matched_clean"] == {predicate: True}
        mutant_fail = row["mutant"] == {predicate: False}
        detected = clean_pass and mutant_fail
        if row["control_id"] == "PREFIX_CONTENT_MUTATION":
            detected = detected and row["storage_identity_stable"] is True
        if row["control_id"] == "MUTABLE_CACHE_ALIAS":
            detected = detected and bool(row["mutant_overlap_ranges"])
        require(
            row["classification"]
            == ("detected_expected_predicate" if detected else "escaped_or_clean_failure"),
            "control classification drift",
        )
        all_passed = all_passed and detected
    return all_passed


def replay_rank(shard: dict[str, Any], raw: bytes, by_id: dict[str, dict[str, Any]], static: dict[str, Any]) -> dict[str, Any]:
    require(shard["schema_version"] == "r39-second-model-transfer-shard-v1", "shard schema drift")
    require(shard["protocol"] == PROTOCOL and shard["scientific_run_valid"] is True, "invalid shard")
    identity = shard["identity"]
    require(identity["model_id"] == static["model"]["repo_id"], "rank model ID drift")
    require(identity["model_revision"] == static["model"]["revision"], "rank revision drift")
    require(identity["transformers_version"] == "5.14.1", "rank runtime drift")
    require(identity["geometry"]["matches_registered"] is True, "rank geometry failed")
    require(identity["geometry"]["layer_types"] == static["model"]["expected_geometry"]["layer_types"], "rank route drift")
    require(identity["dispatch"]["scope"] == "partial_python_source_and_class_provenance", "dispatch scope drift")
    require(identity["dispatch"]["compiled_kernel_binary_fingerprint"] is None, "invented compiled identity")
    dispatch = identity["dispatch"]
    for source_row in (
        dispatch["adapter_source"],
        dispatch["masking_utils_source"],
        *dispatch["layer_sources"],
    ):
        require(
            isinstance(source_row["sha256"], str) and len(source_row["sha256"]) == 64,
            "Python dispatch source hash drift",
        )
    dispatch_fingerprint = sha256_bytes(canonical_bytes(dispatch))

    references = shard["references"]
    official = references["official_one_shot"]
    manual = references["manual_one_shot"]
    cached = references["official_dynamic_cache"]
    require(len(official) == len(manual) == len(cached) == 2, "reference request count drift")
    referenced_ids: set[str] = set()
    for row in (*official, *manual, *cached):
        referenced_ids.update(validate_semantic_records(row, by_id))
    manual_threshold = static["reference_contract"]["manual_one_shot_validation"]["relative_l2_threshold"]
    cache_threshold = static["reference_contract"]["standard_cache_validation"]["relative_l2_threshold"]
    arm_threshold = static["reference_contract"]["split_vs_authorized_reference"]["relative_l2_threshold"]
    manual_rows = [
        trajectory_comparison(candidate, reference, raw, by_id, threshold=manual_threshold)
        for candidate, reference in zip(manual, official)
    ]
    cache_rows = [
        trajectory_comparison(candidate, reference, raw, by_id, threshold=cache_threshold)
        for candidate, reference in zip(cached, official)
    ]
    reference_authorized = all(row["passed"] for row in manual_rows + cache_rows)

    base = shard["persistent_base"]
    prefix_immutable = (
        base["content_immutable"] is True
        and base["before"]["state_content_sha256"] == base["after"]["state_content_sha256"]
    )
    cells = shard["cells"]
    require([row["fanout"] for row in cells] == list(FANOUTS), "fanout ordering drift")
    ownership_passed = True
    cross_arm_passed = True
    arm_reference_passed = True
    cell_summaries = []
    for cell in cells:
        fanout = cell["fanout"]
        arms = cell["arms"]
        deep = arms["deep_materialized"]
        persistent = arms["persistent_q16"]
        require(deep["fanout"] == persistent["fanout"] == fanout, "cell fanout drift")
        for arm in (deep, persistent):
            require(len(arm["semantics"]) == fanout, "arm semantic cardinality drift")
            for semantic in arm["semantics"]:
                referenced_ids.update(validate_semantic_records(semantic, by_id))
            for phase in ("setup", "first_query", "final"):
                ownership_passed = replay_ownership(
                    arm["ownership"][phase], require_peer=fanout == 2
                ) and ownership_passed
        exact_rows = [
            semantic_exact(left, right)
            for left, right in zip(deep["semantics"], persistent["semantics"])
        ]
        require(len(exact_rows) == fanout, "cross-arm request cardinality drift")
        cross_arm_passed = all(exact_rows) and cross_arm_passed
        reference_rows = {}
        for name, arm in (("deep_materialized", deep), ("persistent_q16", persistent)):
            rows = [
                trajectory_comparison(candidate, cached[index], raw, by_id, threshold=arm_threshold)
                for index, candidate in enumerate(arm["semantics"])
            ]
            reference_rows[name] = rows
            arm_reference_passed = all(row["passed"] for row in rows) and arm_reference_passed
        cell_summaries.append(
            {
                "fanout": fanout,
                "cross_arm_exact_rows": exact_rows,
                "cross_arm_exact": all(exact_rows),
                "authorized_reference_comparisons": reference_rows,
            }
        )

    cross_n_passed = True
    for arm_name in ("deep_materialized", "persistent_q16"):
        first = cells[0]["arms"][arm_name]["semantics"][0]
        second = cells[1]["arms"][arm_name]["semantics"][0]
        cross_n_passed = semantic_exact(first, second) and cross_n_passed
    controls_passed = validate_controls(shard, static)
    require(referenced_ids == set(by_id), "unreferenced or missing sidecar records")
    return {
        "rank": shard["rank"],
        "gpu_uuid": identity["hardware"]["uuid"],
        "source_id": identity["input"]["source_id"],
        "prefix_immutable": prefix_immutable,
        "private_ownership": ownership_passed,
        "cross_arm_exact": cross_arm_passed,
        "cross_n_exact": cross_n_passed,
        "manual_wrapper_validation": manual_rows,
        "standard_cache_validation": cache_rows,
        "reference_authorized": reference_authorized,
        "split_vs_authorized_reference_passed": arm_reference_passed,
        "controls_passed": controls_passed,
        "dispatch_fingerprint": dispatch_fingerprint,
        "cells": cell_summaries,
    }


def build_aggregate(package_root: Path, run_root: Path) -> dict[str, Any]:
    require(sys.byteorder == "little", "detached replay requires a little-endian host")
    freeze_path = package_root / "preregistration" / "freeze.json"
    freeze = load_json(freeze_path)
    require(freeze["schema_version"] == "r39-second-model-transfer-freeze-v1", "freeze schema drift")
    static_path = package_root / "preregistration" / "static-preregistration.json"
    require(sha256_file(static_path) == freeze["static_manifest_sha256"], "static manifest drift")
    static = load_json(static_path)
    require(static["protocol"] == PROTOCOL, "static protocol drift")
    require(
        freeze["acquisition_variant"]
        == "modelscope-official-http200-restart-from-zero-d",
        "acquisition variant drift",
    )
    require(freeze["model_acquisition"] == static["model_acquisition"], "freeze acquisition drift")
    require(freeze["predecessors"]["a"]["partial_reused"] is False, "A partial reuse drift")
    require(freeze["predecessors"]["b"]["partial_reused"] is False, "B partial reuse drift")
    require(freeze["predecessors"]["c"]["partial_reused"] is False, "C partial reuse drift")
    verify_source(package_root, freeze)

    pre_path = run_root / "receipts" / "model-authority-pre.json"
    terminal_path = run_root / "receipts" / "model-authority-terminal.json"
    require(pre_path.read_bytes() == terminal_path.read_bytes(), "model terminal closure drift")
    model_authority = load_json(pre_path)
    require(
        model_authority["schema_version"] == "r39-second-model-transfer-model-authority-v1",
        "model authority schema drift",
    )
    require(model_authority["repo_id"] == static["model"]["repo_id"], "authority model ID drift")
    require(model_authority["revision"] == static["model"]["revision"], "authority revision drift")
    acquisition = model_authority["model_acquisition"]
    require(acquisition["policy"] == static["model_acquisition"], "model acquisition binding drift")
    require(
        acquisition["canonical_huggingface_revision"] == HF_MODEL_REVISION,
        "canonical Hugging Face revision drift",
    )
    require(acquisition["modelscope_revision"] == MODELSCOPE_REVISION, "ModelScope revision drift")
    require(acquisition["policy"]["endpoint"] == "https://modelscope.cn", "ModelScope endpoint drift")
    require(acquisition["policy"]["official_namespace"] == "Qwen", "official namespace drift")
    require(acquisition["policy"]["token"] is False, "public token policy drift")
    require(
        acquisition["policy"]["weight_equivalence_sha256"]
        == "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696",
        "weight equivalence drift",
    )
    require(
        acquisition["policy"]["tokenizer_equivalence_sha256"]
        == "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer equivalence drift",
    )
    require(model_authority["all_local_files_hashed"] is True, "model files were not fully hashed")
    require(
        model_authority["all_local_files_and_directories_read_only"] is True,
        "model tree was not fully closed read-only",
    )
    model_authority_sha = sha256_file(pre_path)
    assignment_path = run_root / "receipts" / "gpu-assignment.json"
    assignment_sha = sha256_file(assignment_path)
    assignment = load_json(assignment_path)
    require(len(assignment["rows"]) == WORLD_SIZE, "GPU assignment count drift")
    assigned_uuids = [row["uuid"] for row in assignment["rows"]]
    require(len(set(assigned_uuids)) == WORLD_SIZE, "GPU UUID reuse")

    summaries = []
    shard_ledger = []
    sidecar_ledger = []
    source_ids = []
    for rank in range(WORLD_SIZE):
        shard_path = run_root / "raw" / "shards" / f"r39-second-model-shard-{rank}.json"
        sidecar_path = run_root / "raw" / "logits" / f"r39-second-model-logits-{rank}.bin"
        require(shard_path.is_file() and sidecar_path.is_file(), f"rank {rank} raw files absent")
        shard = load_json(shard_path)
        require(shard["rank"] == rank and shard["world_size"] == WORLD_SIZE, "rank identity drift")
        identity = shard["identity"]
        require(identity["static_manifest_sha256"] == freeze["static_manifest_sha256"], "rank static binding drift")
        require(identity["source_manifest_sha256"] == freeze["source_manifest_sha256"], "rank source binding drift")
        require(identity["model_authority_sha256"] == model_authority_sha, "rank model binding drift")
        require(identity["gpu_assignment_sha256"] == assignment_sha, "rank GPU binding drift")
        require(identity["hardware"]["uuid"] == assigned_uuids[rank], "rank GPU UUID drift")
        raw, by_id = sidecar_records(shard, sidecar_path)
        summary = replay_rank(shard, raw, by_id, static)
        summaries.append(summary)
        source_ids.append(summary["source_id"])
        shard_ledger.append(
            {"rank": rank, "path": shard_path.name, "bytes": shard_path.stat().st_size, "sha256": sha256_file(shard_path)}
        )
        sidecar_ledger.append(
            {"rank": rank, "path": sidecar_path.name, "bytes": sidecar_path.stat().st_size, "sha256": sha256_file(sidecar_path)}
        )
    require(len(set(source_ids)) == WORLD_SIZE, "rank input books are not independent")
    require(
        len({row["dispatch_fingerprint"] for row in summaries}) == 1,
        "Python dispatch provenance differs across ranks",
    )

    prefix = all(row["prefix_immutable"] for row in summaries)
    ownership = all(row["private_ownership"] for row in summaries)
    cross_arm = all(row["cross_arm_exact"] for row in summaries)
    cross_n = all(row["cross_n_exact"] for row in summaries)
    controls = all(row["controls_passed"] for row in summaries)
    reference = all(row["reference_authorized"] for row in summaries)
    split_reference = all(row["split_vs_authorized_reference_passed"] for row in summaries)
    positive = prefix and ownership and cross_arm and cross_n and controls and reference and split_reference
    targets = [
        {"index": 1, "name": "frozen_identity", "status": "full", "predicate_passed": True, "exact_missingness": []},
        {"index": 2, "name": "prefix_immutability", "status": "full", "predicate_passed": prefix, "exact_missingness": []},
        {"index": 3, "name": "private_ownership", "status": "full", "predicate_passed": ownership, "exact_missingness": []},
        {
            "index": 4,
            "name": "tail_safe_append",
            "status": "not_applicable",
            "predicate_passed": None,
            "exact_missingness": [
                "DynamicCache has no fixed-size paged partial tail",
                "no page-level copy-before-append event exists",
            ],
        },
        {
            "index": 5,
            "name": "dispatch_provenance",
            "status": "partial",
            "predicate_passed": True,
            "exact_missingness": [
                "compiled CUDA/Triton kernel binary fingerprint",
                "kernel autotuning-choice fingerprint",
                "hardware instruction trace",
            ],
        },
        {"index": 6, "name": "cross_arm_equivalence", "status": "full", "predicate_passed": cross_arm, "exact_missingness": []},
        {"index": 7, "name": "cross_n_consistency", "status": "full", "predicate_passed": cross_n, "exact_missingness": []},
    ]
    return {
        "schema_version": "r39-second-model-transfer-aggregate-v1",
        "protocol": PROTOCOL,
        "scientific_run_valid": True,
        "passed": positive,
        "scientific_outcome": (
            "positive_bounded_second_model_second_runtime_transfer"
            if positive
            else "valid_negative_second_model_second_runtime_transfer"
        ),
        "rank_count": WORLD_SIZE,
        "distinct_gpu_uuids": len(set(assigned_uuids)),
        "distinct_input_books": len(set(source_ids)),
        "fanouts": list(FANOUTS),
        "reference_authorized": reference,
        "split_vs_authorized_reference_passed": split_reference,
        "all_targeted_controls_detected": controls,
        "targets": targets,
        "status_vector": [row["status"] for row in targets],
        "rank_summaries": summaries,
        "shards": shard_ledger,
        "sidecars": sidecar_ledger,
        "model_acquisition": acquisition,
        "bindings": {
            "static_manifest_sha256": freeze["static_manifest_sha256"],
            "source_manifest_sha256": freeze["source_manifest_sha256"],
            "model_authority_sha256": model_authority_sha,
            "gpu_assignment_sha256": assignment_sha,
        },
        "claim_boundary": static["claim_boundary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-existing", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(bool(args.output) ^ bool(args.verify_existing), "choose exactly one replay mode")
    aggregate = build_aggregate(args.package_root.resolve(), args.run_root.resolve())
    payload = canonical_bytes(aggregate)
    if args.output:
        atomic_write(args.output, payload)
    else:
        require(args.verify_existing.read_bytes() == payload, "detached aggregate differs bytewise")
        print(
            json.dumps(
                {
                    "verified": True,
                    "aggregate_sha256": sha256_bytes(payload),
                    "scientific_outcome": aggregate["scientific_outcome"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
