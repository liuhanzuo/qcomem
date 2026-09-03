#!/usr/bin/env python3
"""Torch-free exact replay for the frozen Falcon-H1 R39 transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any


PROTOCOL = "forkaudit-falcon-h1-hybrid-transformers-transfer-v1"
MODEL_ID = "tiiuae/Falcon-H1-0.5B-Base"
HF_REVISION = "59fb76e8c5d3fc7441b062be638e1ba0afd5c687"
MS_REVISION = "a475c769e108fd1dc6cfe41e342305d36431ef20"
WORLD_SIZE = 8
FANOUTS = (1, 2)
VOCAB_SIZE = 32784
FAMILY_ORDER = ("kv_key", "kv_value", "conv", "mamba2_recurrent")
OFFICIAL_SOURCES = {
    "modeling_falcon_h1.py": "e90bf774524e9b66284ad1c5528c35339271a187f58f16ba2d45c97f4bc6b5bd",
    "cache_utils.py": "ee7902fbd031ed332b5e26d07756a33f09b5c90a435b8363b9330876dc33ce0e",
    "masking_utils.py": "5f48e428ea02d1b6008acb45c147fcdb4eba89deea69627744662aa05da1b9f2",
}


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
    require(manifest["schema_version"] == "r39-falcon-h1-transfer-source-v1", "source schema drift")
    require(manifest["protocol"] == PROTOCOL, "source protocol drift")
    repo_root = package_root.parents[2]
    require(manifest["file_count"] == len(manifest["files"]) and manifest["files"], "source count drift")
    for row in manifest["files"]:
        relative = Path(row["path"])
        require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        path = repo_root / relative
        require(path.is_file() and not path.is_symlink(), f"source absent: {relative}")
        require(path.stat().st_size == row["bytes"], f"source size drift: {relative}")
        require(sha256_file(path) == row["sha256"], f"source hash drift: {relative}")


def sidecar_records(shard: dict[str, Any], path: Path) -> tuple[bytes, dict[str, dict[str, Any]]]:
    receipt = shard["sidecar"]
    raw = path.read_bytes()
    require(len(raw) == receipt["bytes"], "sidecar byte count drift")
    require(sha256_bytes(raw) == receipt["sha256"], "sidecar terminal hash drift")
    rows = receipt["records"]
    require(receipt["record_count"] == len(rows) and rows, "sidecar record count drift")
    by_id = {}
    cursor = 0
    for row in rows:
        require(row["record_id"] not in by_id, "duplicate sidecar record ID")
        require(row["offset_bytes"] == cursor, "sidecar gap/overlap")
        require(row["shape"] == [1, VOCAB_SIZE] and row["dtype"] == "float32-le", "sidecar type drift")
        require(row["nbytes"] == VOCAB_SIZE * 4, "sidecar record size drift")
        end = cursor + row["nbytes"]
        require(end <= len(raw), "sidecar record exceeds payload")
        record_raw = raw[cursor:end]
        require(sha256_bytes(record_raw) == row["content_sha256"], "sidecar record hash drift")
        values = memoryview(record_raw).cast("f")
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
    closure = receipt["terminal_closure"]
    require(
        cursor == len(raw)
        and closure["exact_byte_coverage"] is True
        and closure["first_offset_bytes"] == 0
        and closure["last_end_offset_bytes"] == len(raw),
        "sidecar closure drift",
    )
    return raw, by_id


def record_bytes(raw: bytes, row: dict[str, Any]) -> bytes:
    start = row["offset_bytes"]
    return raw[start : start + row["nbytes"]]


def validate_family_receipt(receipt: dict[str, Any], sequence_length: int) -> None:
    require(receipt["schema_version"] == "r39-falcon-h1-composed-state-family-receipt-v1", "family schema drift")
    require(receipt["split_depth"] == 18, "family depth drift")
    require(receipt["expected_sequence_length"] == sequence_length, "family sequence length drift")
    require(receipt["complete"] is True, "incomplete family receipt")
    require(receipt["expected_family_count"] == receipt["observed_family_count"] == 144, "family count drift")
    rows = receipt["rows"]
    require(len(rows) == 144 and receipt["rows_sha256"] == sha256_bytes(canonical_bytes(rows)), "family rows hash drift")
    expected_pairs = [(layer, family) for layer in range(36) for family in FAMILY_ORDER]
    require([(row["layer_index"], row["family"]) for row in rows] == expected_pairs, "family binding drift")
    for row in rows:
        family = row["family"]
        expected_shape = {
            "kv_key": [1, 2, sequence_length, 64],
            "kv_value": [1, 2, sequence_length, 64],
            "conv": [1, 1792, 4],
            "mamba2_recurrent": [1, 24, 64, 128],
        }[family]
        require(row["shape"] == expected_shape, "family shape drift")
        require(row["dtype"] == ("torch.float32" if family == "mamba2_recurrent" else "torch.bfloat16"), "family dtype drift")
        require(isinstance(row["content_sha256"], str) and len(row["content_sha256"]) == 64, "family content hash drift")


def replay_ownership(receipt: dict[str, Any], require_peer: bool) -> bool:
    comparisons = receipt["comparisons"]
    require(receipt["comparison_count"] == len(comparisons) and comparisons, "ownership count drift")
    peer = 0
    passed = receipt["tensor_pair_comparison_count"] > 0
    for row in comparisons:
        disjoint = not row["overlap_ranges"]
        require(row["disjoint"] is disjoint, "ownership producer flag drift")
        passed = passed and disjoint
        if row["relation"] == "request_request":
            peer += 1
    require(peer == receipt["peer_comparison_count"], "peer count drift")
    if require_peer:
        require(peer > 0, "N=2 peer comparison is vacuous")
    require(receipt["passed"] is passed, "ownership aggregate flag drift")
    return passed


def validate_controls(candidate: dict[str, Any], static: dict[str, Any]) -> bool:
    frozen = static["controls"]
    observed = candidate["controls"]
    require(len(frozen) == len(observed) == 5, "control count drift")
    passed = True
    for expected, row in zip(frozen, observed):
        require(row["control_id"] == expected["control_id"], "control ordering drift")
        predicate = expected["expected_first_failing_predicate"]
        require(row["expected_first_failing_predicate"] == predicate, "control predicate drift")
        detected = row["matched_clean"] == {predicate: True} and row["mutant"] == {predicate: False}
        if row["control_id"] == "MUTABLE_CACHE_ALIAS":
            detected = detected and bool(row["mutant_overlap_ranges"])
        require(
            row["classification"] == ("detected_expected_predicate" if detected else "escaped_or_clean_failure"),
            "control classification drift",
        )
        passed = passed and detected
    return passed


def exact_step_comparison(
    candidate_step: dict[str, Any],
    reference_step: dict[str, Any],
    candidate_raw: bytes,
    candidate_by_id: dict[str, dict[str, Any]],
    reference_raw: bytes,
    reference_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    require(candidate_step["step"] == reference_step["step"], "step index drift")
    candidate_record = candidate_by_id[candidate_step["record_id"]]
    reference_record = reference_by_id[reference_step["record_id"]]
    require(candidate_step["logit_sha256"] == candidate_record["content_sha256"], "candidate logit link drift")
    require(reference_step["logit_sha256"] == reference_record["content_sha256"], "reference logit link drift")
    require(candidate_step["generated_token_id"] == candidate_record["argmax"], "candidate token link drift")
    require(reference_step["generated_token_id"] == reference_record["argmax"], "reference token link drift")
    sequence_length = 72 + candidate_step["step"]
    candidate_family = candidate_step["cache_family_receipt"]
    reference_family = reference_step["cache_family_receipt"]
    validate_family_receipt(candidate_family, sequence_length)
    validate_family_receipt(reference_family, sequence_length)
    logit_exact = record_bytes(candidate_raw, candidate_record) == record_bytes(reference_raw, reference_record)
    family_exact = candidate_family["rows"] == reference_family["rows"]
    token_exact = candidate_record["argmax"] == reference_record["argmax"]
    return {
        "step": candidate_step["step"],
        "candidate_record_id": candidate_record["record_id"],
        "reference_record_id": reference_record["record_id"],
        "full_fp32_logit_bytes_exact": logit_exact,
        "max_abs": 0.0 if logit_exact else None,
        "relative_l2": 0.0 if logit_exact else None,
        "generated_token_exact": token_exact,
        "all_144_family_rows_exact": family_exact,
        "candidate_family_sha256": candidate_family["rows_sha256"],
        "reference_family_sha256": reference_family["rows_sha256"],
        "passed": logit_exact and token_exact and family_exact,
    }


def exact_trajectory_comparison(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    candidate_raw: bytes,
    candidate_by_id: dict[str, dict[str, Any]],
    reference_raw: bytes,
    reference_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    require(candidate["request_index"] == reference["request_index"], "request index drift")
    require(candidate["query_token_ids_int64_le_sha256"] == reference["query_token_ids_int64_le_sha256"], "query digest drift")
    require(len(candidate["steps"]) == len(reference["steps"]) == 2, "trajectory step count drift")
    rows = [
        exact_step_comparison(c_step, r_step, candidate_raw, candidate_by_id, reference_raw, reference_by_id)
        for c_step, r_step in zip(candidate["steps"], reference["steps"])
    ]
    tokens_exact = candidate["generated_token_ids"] == reference["generated_token_ids"]
    return {
        "request_index": candidate["request_index"],
        "generated_tokens_exact": tokens_exact,
        "steps": rows,
        "passed": tokens_exact and all(row["passed"] for row in rows),
    }


def candidate_trajectory_exact(left: dict[str, Any], right: dict[str, Any], raw: bytes, by_id: dict[str, dict[str, Any]]) -> bool:
    if left["request_index"] != right["request_index"] or left["generated_token_ids"] != right["generated_token_ids"]:
        return False
    if left["query_token_ids_int64_le_sha256"] != right["query_token_ids_int64_le_sha256"]:
        return False
    if len(left["steps"]) != len(right["steps"]):
        return False
    for a, b in zip(left["steps"], right["steps"]):
        if record_bytes(raw, by_id[a["record_id"]]) != record_bytes(raw, by_id[b["record_id"]]):
            return False
        if a["cache_family_receipt"]["rows"] != b["cache_family_receipt"]["rows"]:
            return False
    return True


def replay_rank(
    candidate: dict[str, Any],
    reference: dict[str, Any],
    candidate_raw: bytes,
    candidate_by_id: dict[str, dict[str, Any]],
    reference_raw: bytes,
    reference_by_id: dict[str, dict[str, Any]],
    static: dict[str, Any],
    expected_static_sha256: str,
    expected_source_sha256: str,
    expected_model_authority_sha256: str,
    expected_gpu_assignment_sha256: str,
    expected_gpu_row: dict[str, Any],
    expected_input_row: dict[str, Any],
) -> dict[str, Any]:
    require(candidate["schema_version"] == "r39-falcon-h1-candidate-shard-v1", "candidate schema drift")
    require(reference["schema_version"] == "r39-falcon-h1-reference-shard-v1", "reference schema drift")
    require(candidate["protocol"] == reference["protocol"] == PROTOCOL, "rank protocol drift")
    require(candidate["rank"] == reference["rank"], "candidate/reference rank drift")
    require(candidate["world_size"] == reference["world_size"] == WORLD_SIZE, "rank world-size drift")
    require(candidate["scientific_run_valid"] is True and reference["scientific_run_valid"] is True, "invalid rank")
    c_identity = candidate["identity"]
    r_identity = reference["identity"]
    require(
        c_identity["static_manifest_sha256"]
        == r_identity["static_manifest_sha256"]
        == expected_static_sha256,
        "rank static authority drift",
    )
    require(
        c_identity["source_manifest_sha256"]
        == r_identity["source_manifest_sha256"]
        == expected_source_sha256,
        "rank source authority drift",
    )
    require(
        c_identity["model_authority_sha256"]
        == r_identity["model_authority_sha256"]
        == expected_model_authority_sha256,
        "rank model authority drift",
    )
    require(
        c_identity["gpu_assignment_sha256"]
        == r_identity["gpu_assignment_sha256"]
        == expected_gpu_assignment_sha256,
        "rank GPU-assignment authority drift",
    )
    require(c_identity["model_id"] == r_identity["model_id"] == MODEL_ID, "model ID drift")
    require(c_identity["model_revision"] == HF_REVISION and r_identity["hf_revision"] == HF_REVISION, "HF revision drift")
    require(r_identity["modelscope_revision"] == MS_REVISION, "MS revision drift")
    require(c_identity["transformers_version"] == r_identity["transformers_version"] == "5.14.1", "runtime drift")
    require(expected_gpu_row["rank"] == candidate["rank"], "expected GPU rank drift")
    require(
        c_identity["hardware"]["uuid"]
        == r_identity["hardware"]["uuid"]
        == expected_gpu_row["uuid"],
        "reference/candidate/assignment GPU drift",
    )
    require(c_identity["geometry"]["matches_registered"] is True, "candidate geometry failed")
    require(r_identity["geometry"]["matches_registered"] is True, "reference geometry failed")
    require(c_identity["official_source_sha256"] == r_identity["official_source_sha256"] == OFFICIAL_SOURCES, "official source drift")
    require(r_identity["reference_implementation"]["candidate_import_free"] is True, "reference imports candidate")
    require(r_identity["reference_implementation"]["same_chunk_schedule"] == [64, 8, 1], "reference schedule drift")
    require(c_identity["dispatch"]["mamba_dispatch"]["fast_path_observed_after_force"] is False, "candidate fast path enabled")
    require(r_identity["dispatch"]["fast_path_observed_after_force"] is False, "reference fast path enabled")
    require(c_identity["dispatch"]["attention_implementation"] == "eager", "candidate attention route drift")
    c_input = c_identity["input"]
    r_input = r_identity["input"]
    require(expected_input_row["rank"] == candidate["rank"], "expected input rank drift")
    require(
        c_input["source_id"] == r_input["source_id"] == expected_input_row["source_id"],
        "rank PG-19 source ID drift",
    )
    require(
        c_input["source_object"]
        == r_input["source_object"]
        == expected_input_row["source_object"],
        "rank PG-19 source object drift",
    )
    require(
        c_input["document_token_ids_int64_le_sha256"]
        == r_input["document_token_ids_int64_le_sha256"]
        == expected_input_row["document_token_ids_int64_le_sha256"],
        "rank document digest drift",
    )
    expected_query_digests = [
        row["token_ids_int64_le_sha256"] for row in expected_input_row["queries"]
    ]
    require(
        c_input["query_token_ids_int64_le_sha256"]
        == r_input["query_token_ids_int64_le_sha256"]
        == expected_query_digests,
        "rank query digest drift",
    )

    references = reference["trajectories"]
    require(len(references) == 2, "reference request count drift")
    referenced_reference_ids = {
        step["record_id"] for trajectory in references for step in trajectory["steps"]
    }
    require(referenced_reference_ids == set(reference_by_id), "reference sidecar coverage drift")

    base = candidate["persistent_base"]
    prefix_immutable = (
        base["content_immutable"] is True
        and base["before"]["state_content_sha256"] == base["after"]["state_content_sha256"]
    )
    exact_lower = base["exact_lower_family_receipt_before_packing"]
    require(exact_lower["complete"] is True and exact_lower["observed_family_count"] == 72, "lower-base family census drift")

    cells = candidate["cells"]
    require([cell["fanout"] for cell in cells] == list(FANOUTS), "fanout ordering drift")
    referenced_candidate_ids = set()
    ownership_passed = True
    official_passed = True
    cross_arm_passed = True
    summaries = []
    for cell in cells:
        fanout = cell["fanout"]
        deep = cell["arms"]["deep_materialized"]
        persistent = cell["arms"]["persistent_q16"]
        require(len(deep["semantics"]) == len(persistent["semantics"]) == fanout, "arm cardinality drift")
        for arm in (deep, persistent):
            for phase in ("setup", "first_query", "final"):
                ownership_passed = replay_ownership(arm["ownership"][phase], fanout == 2) and ownership_passed
            for trajectory in arm["semantics"]:
                referenced_candidate_ids.update(step["record_id"] for step in trajectory["steps"])
        cross_arm_rows = [
            candidate_trajectory_exact(left, right, candidate_raw, candidate_by_id)
            for left, right in zip(deep["semantics"], persistent["semantics"])
        ]
        cross_arm_passed = all(cross_arm_rows) and cross_arm_passed
        official_rows = {}
        for name, arm in (("deep_materialized", deep), ("persistent_q16", persistent)):
            comparisons = [
                exact_trajectory_comparison(
                    trajectory,
                    references[index],
                    candidate_raw,
                    candidate_by_id,
                    reference_raw,
                    reference_by_id,
                )
                for index, trajectory in enumerate(arm["semantics"])
            ]
            official_rows[name] = comparisons
            official_passed = all(row["passed"] for row in comparisons) and official_passed
        summaries.append(
            {
                "fanout": fanout,
                "cross_arm_exact_rows": cross_arm_rows,
                "cross_arm_exact": all(cross_arm_rows),
                "candidate_vs_independent_official": official_rows,
            }
        )

    cross_n_passed = True
    for arm_name in ("deep_materialized", "persistent_q16"):
        first = cells[0]["arms"][arm_name]["semantics"][0]
        second = cells[1]["arms"][arm_name]["semantics"][0]
        cross_n_passed = candidate_trajectory_exact(first, second, candidate_raw, candidate_by_id) and cross_n_passed
    require(referenced_candidate_ids == set(candidate_by_id), "candidate sidecar coverage drift")
    controls_passed = validate_controls(candidate, static)
    prefix_detector = candidate["prefix_mutation_detector"]
    prefix_detector_passed = (
        prefix_detector["detector_id"] == "PREFIX_CONTENT_MUTATION"
        and prefix_detector["predicate"] == "PERSISTENT_PREFIX_IMMUTABLE"
        and prefix_detector["before_content_sha256"] != prefix_detector["after_content_sha256"]
        and prefix_detector["storage_identity_stable"] is True
        and prefix_detector["detected"] is True
    )
    passed = all(
        (
            prefix_immutable,
            ownership_passed,
            official_passed,
            cross_arm_passed,
            cross_n_passed,
            controls_passed,
            prefix_detector_passed,
        )
    )
    return {
        "rank": candidate["rank"],
        "gpu_uuid": c_identity["hardware"]["uuid"],
        "source_id": c_identity["input"]["source_id"],
        "prefix_immutable": prefix_immutable,
        "private_mutable_ownership": ownership_passed,
        "candidate_vs_independent_official_exact": official_passed,
        "cross_arm_exact": cross_arm_passed,
        "cross_n_exact": cross_n_passed,
        "controls_passed": controls_passed,
        "prefix_mutation_detector_passed": prefix_detector_passed,
        "cells": summaries,
        "passed": passed,
    }


def build_aggregate(package_root: Path, run_root: Path) -> dict[str, Any]:
    require(sys.byteorder == "little", "replay requires a little-endian host")
    freeze_path = package_root / "preregistration" / "freeze.json"
    freeze = load_json(freeze_path)
    require(freeze["schema_version"] == "r39-falcon-h1-transfer-freeze-v1", "freeze schema drift")
    static_path = package_root / "preregistration" / "static-preregistration.json"
    require(sha256_file(static_path) == freeze["static_manifest_sha256"], "static manifest drift")
    static = load_json(static_path)
    require(static["protocol"] == PROTOCOL, "static protocol drift")
    require(static["reference_contract"]["max_abs_threshold"] == 0.0, "nonzero max-abs threshold")
    require(static["reference_contract"]["relative_l2_threshold"] == 0.0, "nonzero L2 threshold")
    verify_source(package_root, freeze)

    pre = run_root / "receipts" / "model-authority-pre.json"
    terminal = run_root / "receipts" / "model-authority-terminal.json"
    require(pre.read_bytes() == terminal.read_bytes(), "model authority terminal drift")
    authority = load_json(pre)
    require(authority["schema_version"] == "r39-falcon-h1-model-authority-v1", "model authority schema drift")
    require(authority["repo_id"] == MODEL_ID and authority["revision"] == HF_REVISION, "model authority identity drift")
    require(authority["model_acquisition"]["policy"] == static["model_acquisition"], "acquisition policy drift")
    verification_receipts = {}
    for name in ("source-verification", "static-verification", "freeze-verification"):
        pre_receipt = run_root / "receipts" / f"{name}-pre.json"
        terminal_receipt = run_root / "receipts" / f"{name}-terminal.json"
        require(pre_receipt.read_bytes() == terminal_receipt.read_bytes(), f"{name} terminal drift")
        receipt = load_json(pre_receipt)
        require(receipt["verified"] is True, f"{name} failed")
        verification_receipts[name] = receipt
    require(
        verification_receipts["source-verification"]["manifest_sha256"]
        == freeze["source_manifest_sha256"],
        "source verification is not bound to the freeze",
    )
    require(
        verification_receipts["static-verification"]["static_sha256"]
        == freeze["static_manifest_sha256"],
        "static verification is not bound to the freeze",
    )
    freeze_sha256 = sha256_file(freeze_path)
    require(
        verification_receipts["freeze-verification"]["freeze_sha256"]
        == freeze_sha256,
        "current freeze is not the pre/terminal verified freeze",
    )

    assignment = load_json(run_root / "receipts" / "gpu-assignment.json")
    require(assignment["schema_version"] == "r39-falcon-h1-gpu-assignment-v1", "GPU assignment schema drift")
    require(len(assignment["rows"]) == WORLD_SIZE, "GPU assignment count drift")
    gpu_assignment_sha256 = sha256_file(run_root / "receipts" / "gpu-assignment.json")
    model_authority_sha256 = sha256_file(pre)
    rank_rows = []
    for rank in range(WORLD_SIZE):
        candidate_path = run_root / "raw" / "candidate" / f"r39-falcon-candidate-{rank}.json"
        reference_path = run_root / "raw" / "reference" / f"r39-falcon-reference-{rank}.json"
        candidate_sidecar_path = run_root / "raw" / "logits" / f"r39-falcon-candidate-logits-{rank}.bin"
        reference_sidecar_path = run_root / "raw" / "logits" / f"r39-falcon-reference-logits-{rank}.bin"
        candidate = load_json(candidate_path)
        reference = load_json(reference_path)
        candidate_raw, candidate_by_id = sidecar_records(candidate, candidate_sidecar_path)
        reference_raw, reference_by_id = sidecar_records(reference, reference_sidecar_path)
        rank_rows.append(
            replay_rank(
                candidate,
                reference,
                candidate_raw,
                candidate_by_id,
                reference_raw,
                reference_by_id,
                static,
                freeze["static_manifest_sha256"],
                freeze["source_manifest_sha256"],
                model_authority_sha256,
                gpu_assignment_sha256,
                assignment["rows"][rank],
                static["rank_inputs"][rank],
            )
        )
    require([row["rank"] for row in rank_rows] == list(range(WORLD_SIZE)), "rank ordering drift")
    require(len({row["gpu_uuid"] for row in rank_rows}) == WORLD_SIZE, "GPU UUID reuse")
    require(len({row["source_id"] for row in rank_rows}) == WORLD_SIZE, "PG-19 source reuse")
    passed = all(row["passed"] for row in rank_rows)
    return {
        "schema_version": "r39-falcon-h1-transfer-aggregate-v1",
        "protocol": PROTOCOL,
        "scientific_run_valid": passed,
        "static_manifest_sha256": freeze["static_manifest_sha256"],
        "source_manifest_sha256": freeze["source_manifest_sha256"],
        "freeze_sha256": freeze_sha256,
        "model_authority_sha256": model_authority_sha256,
        "rank_count": WORLD_SIZE,
        "distinct_gpu_uuid_count": len({row["gpu_uuid"] for row in rank_rows}),
        "distinct_pg19_source_count": len({row["source_id"] for row in rank_rows}),
        "zero_tolerance": True,
        "all_ranks_passed": passed,
        "ranks": rank_rows,
        "claim_boundary": static["claim_boundary"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify-existing", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aggregate = build_aggregate(args.package_root.resolve(), args.run_root.resolve())
    require(aggregate["all_ranks_passed"] is True, "Falcon-H1 exact transfer gates failed")
    payload = canonical_bytes(aggregate)
    if args.output is not None:
        atomic_write(args.output, payload)
    else:
        require(args.verify_existing.read_bytes() == payload, "existing aggregate is not a detached rebuild")
        receipt = {
            "schema_version": "r39-falcon-h1-detached-replay-v1",
            "aggregate_sha256": sha256_bytes(payload),
            "all_ranks_passed": True,
            "verified": True,
        }
        sys.stdout.buffer.write(canonical_bytes(receipt))


if __name__ == "__main__":
    main()
