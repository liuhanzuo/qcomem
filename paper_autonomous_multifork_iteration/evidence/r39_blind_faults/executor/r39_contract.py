#!/usr/bin/env python3
"""Dependency-free contracts for the frozen R39 blind-fault execution.

This module is intentionally usable under ``python -I``.  It owns no detector
predicate and contains no accelerator code.  Its job is to bind every artifact
to the immutable designer files, enforce the fixed eleven-row order, and keep
ineligible selectors distinct from executed scientific pairs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SCHEMA = "forkaudit-r39-blind-fault-executor-contract-v1"
RUN_ID = "R39-BLIND-FAULTS-20260826A"
FAULT_IDS = tuple(f"R39-BF{index:02d}" for index in range(1, 12))
FAULT_SET_SHA256 = "a919c53cda32a1e1089568b340725ff287c3d74ac590e25cf97d124779901ac2"
PROTOCOL_CORE_SHA256 = "2aa9ca0cc5652591bbee5338abe97436657c14f6c4605bdc89cd73cf69c88b9e"
PLAN_CANONICAL_SHA256 = "cfb9f93f5b60377c1db9a3f7cca57d376c657b72e0e9449804166c75a84efe4c"
PROTOCOL_RAW_SHA256 = "1da00f78bdcf80f0e50f658573cbe02e64bbb40a6bbfb030e9d268fc155deb06"
PLAN_RAW_SHA256 = "4bcf173dd97d08dc6b506b38dd31202002789071959516e922a955ed293d9e68"
EXECUTION_INPUT_SHA256 = "5a522e48650e3010621e6e06c7c8bbab67c074bca1b6a6a0c70aa50133b4e98d"
SHA_RE = re.compile(r"[0-9a-f]{64}")

# Fixed physical-GPU assignment.  Jobs in each tuple are serial; tuples run in
# parallel.  The model-input rank is the physical-GPU index, except BF10 whose
# two-document selector is resolved from the frozen eight-book manifest before
# model loading and written into its feasibility amendment.
GPU_ASSIGNMENT = {
    0: ("R39-BF01", "R39-BF09"),
    1: ("R39-BF02", "R39-BF10"),
    2: ("R39-BF03", "R39-BF11"),
    3: ("R39-BF04",),
    4: ("R39-BF05",),
    5: ("R39-BF06",),
    6: ("R39-BF07",),
    7: ("R39-BF08",),
}
FAULT_TO_GPU = {
    fault_id: gpu for gpu, fault_ids in GPU_ASSIGNMENT.items() for fault_id in fault_ids
}

EXPECTED_HORIZON = {
    "R39-BF01": "H6", "R39-BF02": "H6", "R39-BF03": "H6",
    "R39-BF04": "H6", "R39-BF05": "H6", "R39-BF06": "H6",
    "R39-BF07": "H7", "R39-BF08": "H7", "R39-BF09": "H8",
    "R39-BF10": "H6", "R39-BF11": "H6",
}

# These two selectors are absent from the frozen implementation.  The reasons
# are source facts, not outcome-dependent exclusions.  They are still emitted
# as individually bound ineligible rows; no substitute is permitted.
STATIC_INELIGIBLE = {
    "R39-BF02": {
        "code": "NO_CROSS_LAYER_SINGLE_PAGE_BACKING_INDIRECTION",
        "reason": (
            "Q16PagedArena exposes one monolithic per-layer key_cache/value_cache "
            "tensor and Q16PagedSequence page tables contain only local integer "
            "block ids.  The frozen source has no operation that can bind exactly "
            "one layer page to another layer's byte range without also rebinding "
            "other roles."
        ),
    },
    "R39-BF09": {
        "code": "NO_PREEXISTING_LAST_USE_EVENT_FENCE_LOCUS",
        "reason": (
            "The fixed replacement implementation synchronizes the completed "
            "two-stream batch before scrub/reclaim and records no per-request "
            "last-use CUDA event or wait.  Therefore there is no exact wait-only "
            "payload locus to suppress; inventing a new fence would substitute a "
            "different implementation."
        ),
    },
}


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    require(not pending.exists(), f"stale pending path {pending}")
    payload = canonical_bytes(value) + b"\n"
    with pending.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def _jq_canonical_faults(plan: Mapping[str, Any]) -> bytes:
    # jq -S -c emits one trailing LF.  Python's canonical form is byte-identical
    # for this ASCII-only frozen JSON.
    return canonical_bytes(plan["faults"]) + b"\n"


def verify_freeze(protocol_path: Path, plan_path: Path) -> dict[str, Any]:
    require(sha256_file(protocol_path) == PROTOCOL_RAW_SHA256, "PROTOCOL.md raw SHA drift")
    require(sha256_file(plan_path) == PLAN_RAW_SHA256, "plan.json raw SHA drift")
    protocol_raw = protocol_path.read_bytes()
    marker = b"## Integrity block"
    require(marker in protocol_raw, "protocol integrity marker missing")
    protocol_core = protocol_raw[: protocol_raw.index(marker)]
    require(sha256_bytes(protocol_core) == PROTOCOL_CORE_SHA256, "protocol core SHA drift")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    faults = plan.get("faults")
    require(isinstance(faults, list), "frozen fault list missing")
    require(tuple(row.get("id") for row in faults) == FAULT_IDS, "fault order drift")
    require(sha256_bytes(_jq_canonical_faults(plan)) == FAULT_SET_SHA256, "fault-set SHA drift")
    rebound = json.loads(json.dumps(plan))
    rebound["integrity"]["plan_canonical_sha256"] = None
    require(
        sha256_bytes(canonical_bytes(rebound) + b"\n") == PLAN_CANONICAL_SHA256,
        "canonical plan SHA drift",
    )
    return {
        "protocol_raw_sha256": PROTOCOL_RAW_SHA256,
        "plan_raw_sha256": PLAN_RAW_SHA256,
        "protocol_core_sha256": PROTOCOL_CORE_SHA256,
        "fault_set_sha256": FAULT_SET_SHA256,
        "plan_canonical_sha256": PLAN_CANONICAL_SHA256,
        "fault_row_sha256": {row["id"]: sha256_json(row) for row in faults},
        "plan": plan,
    }


def verify_source_manifest(path: Path, root: Path) -> dict[str, str]:
    require(path.is_file(), "source manifest missing")
    rows: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line:
            continue
        parts = line.split("  ", 1)
        require(len(parts) == 2 and SHA_RE.fullmatch(parts[0]) is not None, f"manifest line {lineno}")
        relative = Path(parts[1])
        require(not relative.is_absolute() and ".." not in relative.parts, f"unsafe manifest path {relative}")
        target = root / relative
        require(target.is_file(), f"manifest target absent: {relative}")
        require(sha256_file(target) == parts[0], f"source SHA drift: {relative}")
        require(relative.as_posix() not in rows, f"duplicate manifest path: {relative}")
        rows[relative.as_posix()] = parts[0]
    require(rows, "empty source manifest")
    return rows


def validate_feasibility(value: Any, *, fault_id: str, freeze: Mapping[str, Any]) -> dict[str, Any]:
    require(isinstance(value, Mapping), "feasibility amendment must be an object")
    required = {
        "schema_version", "run_id", "status", "candidate_output_seen",
        "fault_id", "fault_row_sha256", "plan_raw_sha256",
        "protocol_raw_sha256", "selector_resolution", "eligible",
        "ineligible_reason", "source_manifest_sha256", "receipt_sha256",
    }
    require(set(value) == required, "feasibility schema drift")
    require(value["schema_version"] == "forkaudit-r39-preexecution-feasibility-v1", "feasibility schema")
    require(value["run_id"] == RUN_ID and value["status"] == "frozen_before_candidate_outputs", "feasibility status")
    require(value["candidate_output_seen"] is False, "post-output feasibility amendment")
    require(value["fault_id"] == fault_id, "feasibility fault binding")
    require(value["fault_row_sha256"] == freeze["fault_row_sha256"][fault_id], "fault-row SHA drift")
    require(value["plan_raw_sha256"] == PLAN_RAW_SHA256 and value["protocol_raw_sha256"] == PROTOCOL_RAW_SHA256, "freeze binding drift")
    require(type(value["eligible"]) is bool, "eligibility type")
    require(isinstance(value["selector_resolution"], Mapping), "selector resolution")
    if value["eligible"]:
        require(value["ineligible_reason"] is None, "eligible row has reason")
    else:
        require(isinstance(value["ineligible_reason"], Mapping), "ineligible reason missing")
    base = dict(value)
    observed = base.pop("receipt_sha256")
    require(SHA_RE.fullmatch(str(observed)) is not None, "feasibility receipt SHA")
    require(sha256_json(base) == observed, "feasibility receipt digest")
    return dict(value)


def make_feasibility(
    *, fault_id: str, freeze: Mapping[str, Any], selector_resolution: Mapping[str, Any],
    eligible: bool, ineligible_reason: Mapping[str, Any] | None,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    require(fault_id in FAULT_IDS, "unknown fault")
    value = {
        "schema_version": "forkaudit-r39-preexecution-feasibility-v1",
        "run_id": RUN_ID,
        "status": "frozen_before_candidate_outputs",
        "candidate_output_seen": False,
        "fault_id": fault_id,
        "fault_row_sha256": freeze["fault_row_sha256"][fault_id],
        "plan_raw_sha256": PLAN_RAW_SHA256,
        "protocol_raw_sha256": PROTOCOL_RAW_SHA256,
        "selector_resolution": dict(selector_resolution),
        "eligible": bool(eligible),
        "ineligible_reason": None if eligible else dict(ineligible_reason or {}),
        "source_manifest_sha256": source_manifest_sha256,
    }
    value["receipt_sha256"] = sha256_json(value)
    return value


def assignment_rows() -> list[dict[str, Any]]:
    rows = []
    for gpu_index in range(8):
        for serial_index, fault_id in enumerate(GPU_ASSIGNMENT[gpu_index]):
            rows.append({
                "fault_id": fault_id,
                "physical_gpu_index": gpu_index,
                "serial_index_on_gpu": serial_index,
                "default_input_rank": gpu_index,
            })
    require([row["fault_id"] for row in sorted(rows, key=lambda item: FAULT_IDS.index(item["fault_id"]))] == list(FAULT_IDS), "assignment coverage")
    return rows


__all__ = [
    "ContractError", "EXECUTION_INPUT_SHA256", "EXPECTED_HORIZON",
    "FAULT_IDS", "FAULT_SET_SHA256", "FAULT_TO_GPU", "GPU_ASSIGNMENT",
    "PLAN_RAW_SHA256", "PROTOCOL_RAW_SHA256", "RUN_ID", "SCHEMA",
    "STATIC_INELIGIBLE", "assignment_rows", "atomic_json", "canonical_bytes",
    "make_feasibility", "require", "sha256_bytes", "sha256_file",
    "sha256_json", "validate_feasibility", "verify_freeze",
    "verify_source_manifest",
]
