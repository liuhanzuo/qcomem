#!/usr/bin/env python3
"""Offline, raw-first replay for the anonymous RR2 reviewer package.

The replay never trusts the conclusions in ``upstream/forkaudit-summary.json``.
It verifies the raw byte ledger, reconstructs the factorial and allocator
tables, re-evaluates the eight dense FP32 oracles from binary tensor sidecars,
and checks the pointer-free GDN storage timelines and matched mutant pairs.

The exact executed oracle implementation is imported from the hash-bound W-run
source closure.  Numerical comparison to the producer is intentionally made on
the scientific metrics rather than on PyTorch's version-specific backend-audit
serialization.  The scientific threshold remains the preregistered 0.005.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping


ORACLE_RELATIVE_L2_TOLERANCE = 0.005
# Cross-version CPU replay tolerances.  These compare recomputed metrics to the
# producer's CUDA/PyTorch-2.11 dense reference; they do not replace the 0.005
# scientific acceptance threshold.
PORTABLE_METRIC_ATOL = {
    "relative_l2": 5e-7,
    "max_abs": 5e-5,
    "mean_abs": 1e-6,
}
RESIDENT_COUNTS = (1, 8, 32)
ARMS = (
    "kv=vllm-q16-fresh-full-copy-control|gdn=materialize-request-base-functional-rebind",
    "kv=vllm-q16-fresh-full-copy-control|gdn=borrow-immutable-base-functional-rebind",
    "kv=vllm-q16-shared-document-reuse|gdn=materialize-request-base-functional-rebind",
    "kv=vllm-q16-shared-document-reuse|gdn=borrow-immutable-base-functional-rebind",
)
ENDPOINT_FIELDS = (
    "setup_plus_generation_peak_allocated_delta_bytes",
    "setup_plus_generation_peak_reserved_delta_bytes",
    "generation_peak_allocated_delta_bytes",
    "generation_peak_reserved_delta_bytes",
    "after_generation_current_allocated_bytes",
    "after_generation_current_reserved_bytes",
    "after_generation_current_allocated_delta_bytes",
    "after_generation_current_reserved_delta_bytes",
)
DTYPE_BYTES = {
    "torch.bfloat16": 2,
    "torch.float16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.bool": 1,
}
STORAGE_ROW_FIELDS = {
    "byte_end_exclusive",
    "byte_start",
    "content_sha256",
    "device",
    "dtype",
    "layer_index",
    "owner_kind",
    "request_index",
    "shape",
    "state_family",
    "state_index",
    "storage_id",
    "storage_nbytes",
    "storage_offset",
    "stride",
    "tensor_nbytes",
}
GATE_MATERIALIZED_SETUP_BASE_DISJOINT = "gdn_materialized_setup_base_disjoint"
GATE_MATERIALIZED_SETUP_PEERS_DISJOINT = "gdn_materialized_setup_peers_disjoint"
GATE_COMPLETED_VS_BASE_DISJOINT = "gdn_completed_vs_base_disjoint"
GATE_COMPLETED_VS_PEERS_DISJOINT = "gdn_completed_vs_peers_disjoint"


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_reference(root: Path, reference: Mapping[str, Any]) -> Path:
    require(set(reference) == {"relative_path", "sha256", "bytes"}, "artifact reference schema drift")
    rel = Path(str(reference["relative_path"]))
    require(not rel.is_absolute() and ".." not in rel.parts, "unsafe artifact path")
    path = root / rel
    require(path.is_file(), f"missing artifact: {rel}")
    require(path.stat().st_size == reference["bytes"], f"byte count mismatch: {rel}")
    require(file_sha(path) == reference["sha256"], f"SHA-256 mismatch: {rel}")
    return path


def validate_raw_ledger(upstream: Path) -> dict[str, Any]:
    ledger = upstream / "receipts" / "all-raw-artifacts.sha256"
    rows = []
    total = 0
    for line in ledger.read_text(encoding="utf-8").splitlines():
        expected, rel_text = line.split(None, 1)
        rel = Path(rel_text.strip())
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe raw-ledger path")
        path = upstream / rel
        require(path.is_file(), f"raw-ledger file missing: {rel}")
        require(file_sha(path) == expected, f"raw-ledger SHA mismatch: {rel}")
        rows.append({"relative_path": rel.as_posix(), "sha256": expected, "bytes": path.stat().st_size})
        total += path.stat().st_size
    require(len(rows) == 536, f"expected 536 raw artifacts, found {len(rows)}")
    return {
        "artifact_count": len(rows),
        "total_bytes": total,
        "ledger_raw_sha256": file_sha(ledger),
        "all_sha256_verified": True,
    }


def validate_executed_source_closure(package_root: Path, upstream: Path) -> dict[str, Any]:
    ledger = upstream / "preregistration" / "code.sha256"
    source = package_root / "executed_source" / "gpu"
    rows = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        expected, rel_text = line.split(None, 1)
        rel = Path(rel_text.strip())
        require(not rel.is_absolute() and ".." not in rel.parts, "unsafe code-ledger path")
        path = source / rel
        require(path.is_file(), f"executed source missing: {rel}")
        require(file_sha(path) == expected, f"executed source SHA mismatch: {rel}")
        rows.append({"relative_path": rel.as_posix(), "sha256": expected, "bytes": path.stat().st_size})
    require(len(rows) == 34, f"expected 34 executed source files, found {len(rows)}")
    require(file_sha(ledger) == "837f7a488d75cbedbc01e35a236a97f00b85259746e6a368b7aeec873045e94a", "code-ledger byte SHA drift")
    return {
        "source_file_count": len(rows),
        "source_total_bytes": sum(row["bytes"] for row in rows),
        "code_ledger_raw_sha256": file_sha(ledger),
        "all_source_sha256_verified": True,
    }


def byte_interval(row: Mapping[str, Any]) -> tuple[int, int]:
    """Return the conservative half-open byte interval of a strided view.

    This exactly specifies the executed pointer-free algorithm.  For exotic
    interleaved views the bounding interval may conservatively report overlap,
    but it cannot miss a true shared byte.
    """

    shape = row["shape"]
    stride = row["stride"]
    require(isinstance(shape, list) and len(shape) == len(stride) and shape, "shape/stride drift")
    require(all(type(v) is int and v > 0 for v in shape), "invalid shape")
    require(all(type(v) is int for v in stride), "invalid stride")
    offset = row["storage_offset"]
    require(type(offset) is int and offset >= 0, "invalid storage offset")
    require(row["dtype"] in DTYPE_BYTES, "unsupported storage dtype")
    minimum = maximum = offset
    for size, step in zip(shape, stride):
        displacement = (size - 1) * step
        minimum += min(displacement, 0)
        maximum += max(displacement, 0)
    require(minimum >= 0, "strided view begins before storage")
    element_size = DTYPE_BYTES[row["dtype"]]
    return minimum * element_size, (maximum + 1) * element_size


def overlaps(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return bool(
        left["storage_id"] == right["storage_id"]
        and left["byte_start"] < right["byte_end_exclusive"]
        and right["byte_start"] < left["byte_end_exclusive"]
    )


def exact_alias(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "storage_id", "byte_start", "byte_end_exclusive", "shape", "stride",
        "storage_offset", "dtype", "device", "storage_nbytes", "tensor_nbytes",
        "content_sha256",
    )
    return all(left[field] == right[field] for field in fields)


def require_all_pairs_disjoint(
    left_rows: Iterable[Mapping[str, Any]],
    right_rows: Iterable[Mapping[str, Any]],
    label: str,
    gate_id: str,
) -> int:
    """Require cross-owner disjointness for the full Cartesian product.

    Coordinates are deliberately irrelevant here.  The ownership contract is
    violated when any tensor view owned by one side overlaps any tensor view
    owned by the other side, including views at different layer/family/index
    coordinates.  Returning the mechanically evaluated comparison count makes
    the coverage of every frozen phase auditable in the derived summary.
    """

    left = list(left_rows)
    right = list(right_rows)
    comparisons = 0
    for left_row in left:
        for right_row in right:
            comparisons += 1
            require(
                not overlaps(left_row, right_row),
                f"[{gate_id}] {label} overlaps across owners: "
                f"{_coordinate(left_row)} vs {_coordinate(right_row)}",
            )
    return comparisons


def _coordinate(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["layer_index"], row["state_family"], row["state_index"])


def validate_storage_rows(rows: Any) -> dict[tuple[Any, ...], Mapping[str, Any]]:
    require(isinstance(rows, list) and rows, "storage rows missing")
    first_seen_storage: list[str] = []
    storage_geometry: dict[str, tuple[str, int]] = {}
    by_owner_coordinate: dict[tuple[Any, ...], Mapping[str, Any]] = {}
    by_owner_storage: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == STORAGE_ROW_FIELDS, "storage row schema/pointer-field drift")
        sid = row["storage_id"]
        require(isinstance(sid, str) and sid.startswith("storage-"), "non-normalized storage ID")
        if sid not in storage_geometry:
            first_seen_storage.append(sid)
            storage_geometry[sid] = (row["device"], row["storage_nbytes"])
        require(storage_geometry[sid] == (row["device"], row["storage_nbytes"]), "conflicting normalized storage-ID reuse")
        expected_start, expected_end = byte_interval(row)
        require((row["byte_start"], row["byte_end_exclusive"]) == (expected_start, expected_end), "storage byte interval drift")
        require(row["byte_end_exclusive"] <= row["storage_nbytes"], "view exceeds storage")
        require(row["tensor_nbytes"] == math.prod(row["shape"]) * DTYPE_BYTES[row["dtype"]], "tensor byte count drift")
        require(isinstance(row["content_sha256"], str) and len(row["content_sha256"]) == 64, "content SHA drift")
        owner = (row["owner_kind"], row["request_index"])
        key = (*owner, *_coordinate(row))
        require(key not in by_owner_coordinate, "duplicate owner/coordinate row")
        by_owner_coordinate[key] = row
        by_owner_storage.setdefault((*owner, sid), []).append(row)
    require(first_seen_storage == [f"storage-{index:04d}" for index in range(len(first_seen_storage))], "storage IDs are not normalized by first appearance")
    for group in by_owner_storage.values():
        ordered = sorted(group, key=lambda row: row["byte_start"])
        for left, right in zip(ordered, ordered[1:]):
            require(not overlaps(left, right), "one owner has overlapping state views")
    return by_owner_coordinate


def validate_storage_phase(phase: Mapping[str, Any]) -> dict[str, Any]:
    gdn = phase["gdn_phase_witness"]
    witness = gdn["storage_witness"]
    rows = witness["rows"]
    require(canonical_sha(rows) == witness["rows_sha256"], "storage rows canonical SHA drift")
    index = validate_storage_rows(rows)
    n = witness["resident_count"]
    expected_per_owner = witness["expected_tensor_count_per_owner"]
    require(len(rows) == (n + 1) * expected_per_owner, "storage owner cardinality drift")
    completed = set(witness["completed_request_indices"])
    persistent_rows = {
        _coordinate(row): row for row in rows if row["owner_kind"] == "persistent"
    }
    require(len(persistent_rows) == expected_per_owner, "persistent row coverage drift")
    request_rows: dict[int, dict[tuple[Any, ...], Mapping[str, Any]]] = {}
    for row in rows:
        if row["owner_kind"] == "request":
            request_rows.setdefault(row["request_index"], {})[_coordinate(row)] = row
    require(set(request_rows) == set(range(n)), "request row coverage drift")
    request_base_all_pairs_comparisons = 0
    request_peer_all_pairs_comparisons = 0
    exact_alias_coordinate_comparisons = 0
    for request_index, coordinate_rows in request_rows.items():
        require(set(coordinate_rows) == set(persistent_rows), "request coordinate coverage drift")
        if request_index not in completed and witness["policy"] == "shared-base":
            for coordinate, row in coordinate_rows.items():
                persistent = persistent_rows[coordinate]
                require(exact_alias(row, persistent), "incomplete shared request is not an exact alias")
                exact_alias_coordinate_comparisons += 1
        else:
            request_base_all_pairs_comparisons += require_all_pairs_disjoint(
                coordinate_rows.values(),
                persistent_rows.values(),
                f"materialized/completed request[{request_index}]/persistent base",
                (
                    GATE_COMPLETED_VS_BASE_DISJOINT
                    if witness["policy"] == "shared-base"
                    else GATE_MATERIALIZED_SETUP_BASE_DISJOINT
                ),
            )
    for left_index in range(n):
        for right_index in range(left_index + 1, n):
            both_incomplete_shared = (
                witness["policy"] == "shared-base"
                and left_index not in completed
                and right_index not in completed
            )
            if both_incomplete_shared:
                for coordinate in persistent_rows:
                    left = request_rows[left_index][coordinate]
                    right = request_rows[right_index][coordinate]
                    require(exact_alias(left, right), "shared peer aliases disagree")
                    exact_alias_coordinate_comparisons += 1
            else:
                request_peer_all_pairs_comparisons += require_all_pairs_disjoint(
                request_rows[left_index].values(),
                request_rows[right_index].values(),
                f"request[{left_index}]/request[{right_index}]",
                (
                    GATE_COMPLETED_VS_PEERS_DISJOINT
                    if witness["policy"] == "shared-base"
                    else GATE_MATERIALIZED_SETUP_PEERS_DISJOINT
                ),
                )
    guard = witness["persistent_guard"]
    require(
        guard["baseline_binding_sha256"] == guard["observed_binding_sha256"]
        and guard["baseline_content_sha256"] == guard["observed_content_sha256"],
        "persistent guard changed",
    )
    binding = gdn["binding_witness"]
    require(canonical_sha(binding["rows"]) == binding["rows_sha256"], "binding rows canonical SHA drift")
    require(len(binding["rows"]) == n * expected_per_owner, "binding row cardinality drift")
    for row in binding["rows"]:
        is_completed = row["request_index"] in completed
        require(row["expected_relation"] == ("rebound" if is_completed else "unchanged"), "binding relation drift")
        if is_completed:
            require(row["baseline_binding_token"] != row["observed_binding_token"], "completed binding did not change")
            require(row["baseline_storage_token"] != row["observed_storage_token"], "completed storage did not change")
        else:
            require(row["baseline_binding_token"] == row["observed_binding_token"], "incomplete binding changed")
            require(row["baseline_storage_token"] == row["observed_storage_token"], "incomplete storage changed")
    return {
        "phase": witness["phase"],
        "completed_request_indices": sorted(completed),
        "row_count": len(rows),
        "binding_row_count": len(binding["rows"]),
        "capture_id": witness["capture_id"],
        "request_guard_id": witness["request_guard_id"],
        "persistent_guard_id": guard["guard_id"],
        "request_base_all_pairs_comparison_count": request_base_all_pairs_comparisons,
        "request_peer_all_pairs_comparison_count": request_peer_all_pairs_comparisons,
        "exact_alias_coordinate_comparison_count": exact_alias_coordinate_comparisons,
    }


def validate_timeline(raw_root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = checked_reference(raw_root, reference)
    manifest = load_json(manifest_path)
    require(manifest["schema_version"] == "qcomem-gdn-external-timeline-manifest-v1", "timeline schema drift")
    phases = []
    for phase_ref in manifest["phase_artifacts"]:
        phase_path = checked_reference(raw_root, phase_ref)
        phase = load_json(phase_path)
        phase_binding = dict(phase["binding"])
        declared_phase = phase_binding.pop("phase", None)
        require(
            phase_binding == manifest["binding"]
            and declared_phase == phase["gdn_phase_witness"]["phase"],
            "timeline/phase binding drift",
        )
        phases.append(validate_storage_phase(phase))
    require([row["phase"] for row in phases] == ["setup_pre_transition", "post_transition", "post_generation"], "timeline phase order drift")
    require(len({row["capture_id"] for row in phases}) == 3, "phase capture ID reused")
    require(len({row["request_guard_id"] for row in phases}) == 1, "request guard changed across timeline")
    require(len({row["persistent_guard_id"] for row in phases}) == 1, "persistent guard changed across timeline")
    n = manifest["binding"]["resident_count"]
    require(phases[0]["completed_request_indices"] == [], "setup phase already completed")
    require(phases[1]["completed_request_indices"] == [0], "transition phase completion set drift")
    require(phases[2]["completed_request_indices"] == list(range(n)), "generation completion set drift")
    return {"binding": manifest["binding"], "phases": phases}


def allocator_endpoints(receipt: Mapping[str, Any]) -> dict[str, int]:
    baseline = receipt["baseline"]
    setup = receipt["after_setup"]
    generation = receipt["after_generation"]
    result = {
        "setup_plus_generation_peak_allocated_delta_bytes": max(setup["peak_allocated_bytes"], generation["peak_allocated_bytes"]) - baseline["current_allocated_bytes"],
        "setup_plus_generation_peak_reserved_delta_bytes": max(setup["peak_reserved_bytes"], generation["peak_reserved_bytes"]) - baseline["current_reserved_bytes"],
        "generation_peak_allocated_delta_bytes": generation["peak_allocated_bytes"] - setup["current_allocated_bytes"],
        "generation_peak_reserved_delta_bytes": generation["peak_reserved_bytes"] - setup["current_reserved_bytes"],
        "after_generation_current_allocated_bytes": generation["current_allocated_bytes"],
        "after_generation_current_reserved_bytes": generation["current_reserved_bytes"],
        "after_generation_current_allocated_delta_bytes": generation["current_allocated_bytes"] - baseline["current_allocated_bytes"],
        "after_generation_current_reserved_delta_bytes": generation["current_reserved_bytes"] - baseline["current_reserved_bytes"],
    }
    for name in ENDPOINT_FIELDS[:4]:
        require(receipt[name] == result[name], f"allocator endpoint drift: {name}")
    require(all(type(value) is int and value >= 0 for value in result.values()), "invalid allocator endpoint")
    require(canonical_sha(receipt["storage_breakdown"]) == receipt["storage_breakdown_sha256"], "storage breakdown SHA drift")
    return result


def validate_adjacent_cross_n(
    semantics_by_n: Mapping[int, list[Mapping[str, Any]]],
    *,
    rank: int,
    arm_id: str,
) -> list[dict[str, Any]]:
    """Blindly replay nested-prefix semantics for N=1->8 and N=8->32.

    The registered query banks are nested.  Consequently every request in the
    smaller fanout must have the complete semantic record at the same request
    index in the next fanout: generated tokens, every full-vocabulary logit
    hash, logical KV hashes, final GDN hash, and the query-token digest.  This
    is deliberately stronger than comparing only top-1 tokens.
    """

    require(
        list(semantics_by_n) == list(RESIDENT_COUNTS),
        "cross-N resident-count coverage/order drift",
    )
    rows: list[dict[str, Any]] = []
    for lower_n, upper_n in zip(RESIDENT_COUNTS, RESIDENT_COUNTS[1:]):
        lower = semantics_by_n[lower_n]
        upper = semantics_by_n[upper_n]
        require(len(lower) == lower_n and len(upper) == upper_n, "cross-N cardinality drift")
        for request_index, lower_semantics in enumerate(lower):
            require(
                lower_semantics["request_index"] == request_index
                and upper[request_index]["request_index"] == request_index,
                "cross-N request-index order drift",
            )
            require(
                lower_semantics == upper[request_index],
                f"cross-N semantic mismatch: rank={rank} arm={arm_id} "
                f"N={lower_n}->{upper_n} request={request_index}",
            )
            rows.append(
                {
                    "rank": rank,
                    "arm_id": arm_id,
                    "lower_resident_count": lower_n,
                    "upper_resident_count": upper_n,
                    "request_index": request_index,
                    "semantic_sha256": canonical_sha(lower_semantics),
                    "exact": True,
                }
            )
    return rows


def validate_factorial_and_memory(raw_root: Path, shards: list[Mapping[str, Any]], parent: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    configurations = []
    timelines = []
    cross_n_rows = []
    memory_rows: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for rank, shard in enumerate(shards):
        require(shard["rank"] == rank and shard["world_size"] == 8, "shard rank/world drift")
        require(shard["protocol_config_sha256"] == canonical_sha(shard["protocol_config"]), "protocol config SHA drift")
        rows = shard["factorial"]
        require([row["resident_count"] for row in rows] == list(RESIDENT_COUNTS), "resident-count order drift")
        rank_semantics: dict[str, dict[int, list[Mapping[str, Any]]]] = {
            arm: {} for arm in ARMS
        }
        for n_row in rows:
            n = n_row["resident_count"]
            cells = n_row["cells"]
            require([cell["arm_id"] for cell in cells] == list(ARMS), "factorial arm order drift")
            reference_semantics = cells[0]["semantics"]
            for cell in cells:
                require(cell["semantics"] == reference_semantics, "cross-arm semantic mismatch")
                require(cell["witness_semantics"] == cell["semantics"], "memory/witness semantic mismatch")
                require(len(cell["semantics"]) == n, "semantic request cardinality drift")
                rank_semantics[cell["arm_id"]][n] = cell["semantics"]
                endpoint = allocator_endpoints(cell["memory_cell"]["allocator_receipt"])
                memory_rows.setdefault((n, cell["arm_id"]), []).append({"rank": rank, **endpoint})
                timeline = validate_timeline(raw_root, cell["witness_cell"]["timeline_manifest_artifact"])
                timelines.append(timeline)
                configurations.append({
                    "rank": rank,
                    "resident_count": n,
                    "arm_id": cell["arm_id"],
                    "semantic_sha256": canonical_sha(cell["semantics"]),
                    "timeline_manifest_sha256": cell["witness_cell"]["timeline_manifest_artifact"]["sha256"],
                })
        for arm in ARMS:
            cross_n_rows.extend(
                validate_adjacent_cross_n(rank_semantics[arm], rank=rank, arm_id=arm)
            )
    require(len(configurations) == 96 and len(timelines) == 96, "factorial/timeline cardinality drift")
    require(len(cross_n_rows) == 288, "adjacent cross-N comparison cardinality drift")
    cells = []
    parent_cells = {(row["resident_count"], row["arm_id"]): row for row in parent["memory_matrix"]["cells"]}
    for n in RESIDENT_COUNTS:
        for arm in ARMS:
            rows = sorted(memory_rows[(n, arm)], key=lambda row: row["rank"])
            require([row["rank"] for row in rows] == list(range(8)), "memory rank coverage drift")
            medians = {field: statistics.median(row[field] for row in rows) for field in ENDPOINT_FIELDS}
            expected = parent_cells[(n, arm)]
            require(rows == expected["allocator_raw_by_rank"], "raw allocator rows differ from registered aggregate")
            require(medians == expected["allocator_median_across_ranks"], "allocator medians differ from registered aggregate")
            cells.append({
                "resident_count": n,
                "arm_id": arm,
                "kv_policy": expected["kv_policy"],
                "gdn_base_policy": expected["gdn_base_policy"],
                "allocator_raw_by_rank": rows,
                "allocator_median_across_ranks": medians,
            })
    memory = {
        "schema_version": "qcomem-forkaudit-memory-matrix-reviewer-derived-v2",
        "rank_aggregation": "median-with-complete-rank-ordered-raw-list",
        "endpoint_fields": list(ENDPOINT_FIELDS),
        "cells": cells,
    }
    return (
        {
            "configuration_count": 96,
            "cross_arm_exact": True,
            "adjacent_cross_n_exact": True,
            "adjacent_cross_n_comparison_count": len(cross_n_rows),
            "adjacent_cross_n_rows": cross_n_rows,
            "rows": configurations,
        },
        memory,
        {
            "timeline_count": 96,
            "phase_artifact_count": 288,
            "storage_row_count": sum(
                phase["row_count"]
                for timeline in timelines
                for phase in timeline["phases"]
            ),
            "binding_row_count": sum(
                phase["binding_row_count"]
                for timeline in timelines
                for phase in timeline["phases"]
            ),
            "request_base_all_pairs_comparison_count": sum(
                phase["request_base_all_pairs_comparison_count"]
                for timeline in timelines
                for phase in timeline["phases"]
            ),
            "request_peer_all_pairs_comparison_count": sum(
                phase["request_peer_all_pairs_comparison_count"]
                for timeline in timelines
                for phase in timeline["phases"]
            ),
            "exact_alias_coordinate_comparison_count": sum(
                phase["exact_alias_coordinate_comparison_count"]
                for timeline in timelines
                for phase in timeline["phases"]
            ),
            "all_pointer_free_overlap_checks_passed": True,
        },
    )


def _load_executed_oracle_modules(package_root: Path):
    source = package_root / "executed_source" / "gpu"
    sys.path.insert(0, str(source))
    import run_qcomem_qwen35_forkaudit_review_revision as runner  # type: ignore
    from qcomem_forkaudit_oracle import OraclePreregistration, OracleThresholds  # type: ignore
    return runner, OraclePreregistration, OracleThresholds


def recompute_oracle(package_root: Path, raw_root: Path, reference: Mapping[str, Any]) -> dict[str, Any]:
    runner, OraclePreregistration, OracleThresholds = _load_executed_oracle_modules(package_root)
    path = checked_reference(raw_root, reference)
    raw = load_json(path)
    tensors = raw["tensors"]
    bindings = []

    def decode(record: Any, label: str):
        tensor, binding = runner._decode_tensor_record(record, root=raw_root, label=label, require_binary=True)
        if binding is not None:
            bindings.append(binding)
        return tensor

    query = decode(tensors["query"], "oracle query")
    candidate = decode(tensors["candidate_output"], "oracle candidate")
    query_positions = decode(tensors["query_positions"], "oracle query positions")
    key_positions = decode(tensors["key_positions"], "oracle key positions")
    key_pool = decode(tensors["physical_document_key_blocks"], "oracle physical document key blocks")
    value_pool = decode(tensors["physical_document_value_blocks"], "oracle physical document value blocks")
    table = decode(tensors["document_block_table"], "oracle document block table")
    document_length = raw["document_geometry"]["document_length"]
    document_key = runner._logical_document_from_physical(key_pool, table, document_length=document_length)
    document_value = runner._logical_document_from_physical(value_pool, table, document_length=document_length)
    append_keys = [decode(event["key"], f"oracle append key {i}") for i, event in enumerate(raw["append_events"])]
    append_values = [decode(event["value"], f"oracle append value {i}") for i, event in enumerate(raw["append_events"])]
    import torch
    key = torch.cat((document_key, torch.cat(append_keys, dim=2)), dim=2)
    value = torch.cat((document_value, torch.cat(append_values, dim=2)), dim=2)
    visibility = tensors["visibility_mask"]
    visibility_mask = None if visibility is None else decode(visibility, "oracle visibility mask")
    outcome = OraclePreregistration(
        OracleThresholds(max_relative_l2=ORACLE_RELATIVE_L2_TOLERANCE)
    ).evaluate_attention(
        query,
        key,
        value,
        candidate,
        query_positions=query_positions,
        key_positions=key_positions,
        visibility_mask=visibility_mask,
        scaling=float(raw["softmax_scale"]),
    ).to_dict()
    recorded = raw["recorded_outcome"]
    require(recorded["thresholds"]["max_relative_l2"] == ORACLE_RELATIVE_L2_TOLERANCE, "recorded oracle tolerance drift")
    require(outcome["thresholds"]["max_relative_l2"] == ORACLE_RELATIVE_L2_TOLERANCE, "recomputed oracle tolerance drift")
    deltas = {}
    for name, atol in PORTABLE_METRIC_ATOL.items():
        delta = abs(outcome["attention_metrics"][name] - recorded["attention_metrics"][name])
        require(delta <= atol, f"portable oracle metric mismatch: {name} delta={delta}")
        deltas[name] = delta
    require(recorded["attention_metrics"]["relative_l2"] <= ORACLE_RELATIVE_L2_TOLERANCE, "recorded oracle exceeds tolerance")
    require(outcome["attention_metrics"]["relative_l2"] <= ORACLE_RELATIVE_L2_TOLERANCE, "recomputed oracle exceeds tolerance")
    return {
        "rank": raw["selection"]["rank"],
        "sample_id": raw["selection"]["sample_id"],
        "raw_artifact_sha256": reference["sha256"],
        "binary_sidecar_count": len(bindings),
        "recorded_metrics": recorded["attention_metrics"],
        "recomputed_metrics": outcome["attention_metrics"],
        "absolute_metric_deltas": deltas,
        "scientific_tolerance_passed": True,
        "portable_replay_comparison_passed": True,
    }


def validate_oracles(package_root: Path, raw_root: Path, shards: list[Mapping[str, Any]], parent: Mapping[str, Any]) -> dict[str, Any]:
    rows = [recompute_oracle(package_root, raw_root, shard["oracle_raw_artifact"]) for shard in shards]
    rows.sort(key=lambda row: row["rank"])
    require([row["rank"] for row in rows] == list(range(8)), "oracle rank coverage drift")
    recorded = [row["recorded_metrics"]["relative_l2"] for row in rows]
    recomputed = [row["recomputed_metrics"]["relative_l2"] for row in rows]
    require(recorded == parent["oracle_relative_l2_by_rank"], "raw oracle values differ from parent aggregate")
    observed_max = max(recorded)
    require(observed_max == 0.0017432502481433169, "registered observed oracle maximum drift")
    require(observed_max < ORACLE_RELATIVE_L2_TOLERANCE, "oracle maximum does not pass tolerance")
    return {
        "schema_version": "qcomem-forkaudit-oracle-reviewer-derived-v2",
        "oracle_relative_l2_tolerance": ORACLE_RELATIVE_L2_TOLERANCE,
        "oracle_max_relative_l2": observed_max,
        "oracle_relative_l2_by_rank": recorded,
        "recomputed_oracle_relative_l2_by_rank": recomputed,
        "all_eight_recorded_passed": True,
        "all_eight_recomputed_passed": True,
        "portable_metric_comparison_atol": PORTABLE_METRIC_ATOL,
        "rows": rows,
    }


def _cleanup_exact(receipt: Mapping[str, Any]) -> bool:
    return bool(
        receipt["current_allocated_and_reserved_exactly_recovered"] is True
        and receipt["after_cleanup"]["current_allocated_bytes"] == receipt["before_cell"]["current_allocated_bytes"]
        and receipt["after_cleanup"]["current_reserved_bytes"] == receipt["before_cell"]["current_reserved_bytes"]
    )


def validate_mutants(shards: list[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for shard in shards:
        for mutant_id, case in shard["fault_campaign"]["mutants"].items():
            clean = case["matched_clean"]
            mutant = case["outcome"]
            coverage = case["exercise_coverage_receipt"]
            isolation = case["case_isolation"]
            mutation = mutant["mutation_receipt"]
            require(clean["outcome"]["classification"] == "clean_pass", "matched-clean classification drift")
            require(clean["outcome"]["exercise_started"] is True and clean["outcome"]["exercise_completed"] is True, "matched-clean exercise incomplete")
            require(mutant["classification"] == "detected_expected_gate", "mutant classification drift")
            require(mutant["exercise_started"] is True and mutant["detector_satisfied"] is True, "mutant detector was not exercised")
            require(mutant["expected_gate_id"] == mutant["observed_gate_id"], "mutant wrong-gate detection")
            require(
                coverage["exercise_started"] is True
                and coverage["outcome_classification"] == "detected_expected_gate"
                and coverage["observed_gate_id"] == mutant["expected_gate_id"]
                and coverage["detector_path"] == coverage["detector_input"]["detector_path"],
                "mutant detector path did not reach the expected gate",
            )
            # A mutant is expected to short-circuit at the named gate, so the
            # producer truthfully records detector_path_completed=false and
            # exercise_completed=false for these caught expected exceptions.
            require(coverage["detector_path_completed"] is False, "mutant unexpectedly ran beyond its gate")
            require(mutation["mutation_applied"] is True and mutation["restoration_verified"] is True, "mutation apply/restore drift")
            binding = mutation["target_mutation_binding"]
            require(binding["contains_absolute_pointer"] is False, "mutation receipt contains an absolute pointer")
            require(binding["pre_sha256"] != binding["mutated_sha256"] and binding["pre_sha256"] == binding["restored_sha256"], "mutation target was not restored")
            require(all(isolation[name] is expected for name, expected in {
                "cache_discarded_after_case": True,
                "cache_reused_from_prior_case": False,
                "fresh_document_cache_built": True,
                "fresh_request_cache_built": True,
            }.items()), "mutant cache isolation drift")
            require(_cleanup_exact(case["cleanup_receipt"]) and _cleanup_exact(clean["cleanup_receipt"]), "mutant cleanup did not exactly recover allocator state")
            rows.append({
                "mutant_id": mutant_id,
                "rank": shard["rank"],
                "matched_clean_classification": clean["outcome"]["classification"],
                "mutant_classification": mutant["classification"],
                "expected_gate_id": mutant["expected_gate_id"],
                "observed_gate_id": mutant["observed_gate_id"],
                "mutation_applied_and_restored": True,
                "detector_path_reached_expected_gate": True,
                "detector_path_completed_after_gate": False,
                "fresh_case_isolation": True,
                "allocator_cleanup_exact": True,
            })
    rows.sort(key=lambda row: int(row["mutant_id"][1:]))
    require([row["mutant_id"] for row in rows] == [f"M{i}" for i in range(1, 10)], "mutant campaign coverage drift")
    return {"schema_version": "qcomem-forkaudit-matched-mutants-reviewer-derived-v2", "matched_pair_count": 9, "all_matched_clean_passed": True, "all_mutants_detected_at_expected_gate": True, "rows": rows}


def render_tables(memory: Mapping[str, Any], mutants: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    gib = 1024 ** 3
    labels = {
        "vllm-q16-fresh-full-copy-control": "Full-copy KV",
        "vllm-q16-shared-document-reuse": "Shared-doc KV",
        "materialize-request-base-functional-rebind": "Materialized GDN base",
        "borrow-immutable-base-functional-rebind": "Borrowed GDN base",
    }
    lines = [
        r"\begin{table}[H]",
        r"\caption{RR2 allocator deltas at $N=32$ (median across eight ranks; rank values coincide). All values are GiB relative to the frozen post-priming baseline.}",
        r"\label{tab:rr2-memory}",
        r"\centering\scriptsize",
        r"\begin{tabular}{@{}llrrr@{}}", r"\toprule",
        r"KV setup & GDN setup & Final alloc. & Setup+gen. peak & Generation peak \\",
        r"\midrule",
    ]
    cells = [row for row in memory["cells"] if row["resident_count"] == 32]
    cells.sort(key=lambda row: ("shared" in row["kv_policy"], "borrow" in row["gdn_base_policy"]))
    for cell in cells:
        a = cell["allocator_median_across_ranks"]
        lines.append(
            f"{labels[cell['kv_policy']]} & {labels[cell['gdn_base_policy']]} & "
            f"{a['after_generation_current_allocated_delta_bytes']/gib:.3f} & "
            f"{a['setup_plus_generation_peak_allocated_delta_bytes']/gib:.3f} & "
            f"{a['generation_peak_allocated_delta_bytes']/gib:.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "table_5_memory.tex").write_text("\n".join(lines), encoding="utf-8")
    names = {
        "M1": "reservation alias", "M2": "sequence swap", "M3": "omit tail COW",
        "M4": "GDN--base alias", "M5": "GDN peer alias", "M6": "position off-by-one",
        "M7": "materialized mask", "M8": "wrong callable", "M9": "dense KV view",
    }
    lines = [
        r"\begin{table}[H]",
        r"\caption{Executed live faults. Every separately rebuilt matched-clean cell passed; every mutant was rejected at its gate fixed before execution and restored before disposal.}",
        r"\label{tab:rr2-mutants}",
        r"\centering\footnotesize",
        r"\begin{tabular}{@{}clll@{}}", r"\toprule",
        r"ID & Injected live fault & Expected/observed gate & Outcome \\", r"\midrule",
    ]
    for row in mutants["rows"]:
        gate = row["expected_gate_id"].replace("_", r"\_")
        lines.append(f"{row['mutant_id']} & {names[row['mutant_id']]} & \\texttt{{{gate}}} & detected \\\\ ")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    (output / "table_4_mutants.tex").write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def build_release_manifest(package_root: Path, output: Path) -> dict[str, Any]:
    excluded_roots = {output.resolve()}
    excluded_files = {"MANIFEST.json", "MANIFEST.sha256"}
    rows = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or any(root == path.resolve() or root in path.resolve().parents for root in excluded_roots):
            continue
        rel = path.relative_to(package_root).as_posix()
        if rel in excluded_files:
            continue
        rows.append({"relative_path": rel, "bytes": path.stat().st_size, "sha256": file_sha(path)})
    return {
        "schema_version": "anonymous-hash-bound-reviewer-package-v1",
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def build_final_manifest(package_root: Path) -> dict[str, Any]:
    excluded = {"MANIFEST.json", "MANIFEST.sha256"}
    rows = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(package_root).as_posix()
        if rel in excluded:
            continue
        rows.append({"relative_path": rel, "bytes": path.stat().st_size, "sha256": file_sha(path)})
    return {
        "schema_version": "anonymous-hash-bound-reviewer-package-complete-v2",
        "parent_manifest_sha256": "d6a9b71ee078c6d21c90c64ad23d9c4f624e381d262faf36ed812104e8e59633",
        "excludes": sorted(excluded),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }


def replay(
    package_root: Path,
    output: Path,
    *,
    refresh_package_manifest: bool = False,
) -> dict[str, Any]:
    upstream = package_root / "upstream"
    raw_root = upstream / "raw"
    parent_path = upstream / "forkaudit-summary.json"
    parent_raw_sha = file_sha(parent_path)
    require(parent_raw_sha == "8700901ad7423d215e9e9e81a709e976f43963752e1b9f3d64441412b390d2bc", "parent aggregate byte SHA drift")
    parent = load_json(parent_path)
    ledger = validate_raw_ledger(upstream)
    source_closure = validate_executed_source_closure(package_root, upstream)
    detached = load_json(upstream / "receipts" / "detached-receipt-manifest.json")
    shard_paths = [checked_reference(raw_root, ref) for ref in detached["shards"]]
    shards = [load_json(path) for path in shard_paths]
    factorial, memory, storage = validate_factorial_and_memory(raw_root, shards, parent)
    oracles = validate_oracles(package_root, raw_root, shards, parent)
    mutants = validate_mutants(shards)
    summary = {
        "schema_version": "qcomem-forkaudit-reviewer-derived-summary-v2",
        "derivation": "raw-first-offline-replay; parent aggregate used only for registered-value equality invariants",
        "parent_aggregate": {
            "relative_path": "upstream/forkaudit-summary.json",
            "raw_sha256": parent_raw_sha,
            "bytes": parent_path.stat().st_size,
            "left_unchanged": True,
        },
        "oracle_relative_l2_tolerance": ORACLE_RELATIVE_L2_TOLERANCE,
        "oracle_max_relative_l2": oracles["oracle_max_relative_l2"],
        "oracle_relative_l2_by_rank": oracles["oracle_relative_l2_by_rank"],
        "factorial_four_cell_exact": factorial["cross_arm_exact"],
        "factorial_adjacent_cross_n_exact": factorial["adjacent_cross_n_exact"],
        "factorial_adjacent_cross_n_comparison_count": factorial[
            "adjacent_cross_n_comparison_count"
        ],
        "memory_matrix": memory,
        "mutant_campaign": {
            "matched_pair_count": mutants["matched_pair_count"],
            "all_matched_clean_passed": mutants["all_matched_clean_passed"],
            "all_mutants_detected_at_expected_gate": mutants["all_mutants_detected_at_expected_gate"],
        },
        "storage_replay": {
            "timeline_count": storage["timeline_count"],
            "phase_artifact_count": storage["phase_artifact_count"],
            "request_base_all_pairs_comparison_count": storage["request_base_all_pairs_comparison_count"],
            "request_peer_all_pairs_comparison_count": storage["request_peer_all_pairs_comparison_count"],
            "exact_alias_coordinate_comparison_count": storage["exact_alias_coordinate_comparison_count"],
            "all_pointer_free_overlap_checks_passed": storage["all_pointer_free_overlap_checks_passed"],
        },
        "invariants": {
            "raw_artifact_count_is_536": ledger["artifact_count"] == 536,
            "raw_ledger_all_sha256_verified": ledger["all_sha256_verified"],
            "executed_source_file_count_is_34": source_closure["source_file_count"] == 34,
            "executed_source_ledger_all_sha256_verified": source_closure["all_source_sha256_verified"],
            "factorial_configuration_count_is_96": factorial["configuration_count"] == 96,
            "adjacent_cross_n_comparison_count_is_288": (
                factorial["adjacent_cross_n_comparison_count"] == 288
            ),
            "oracle_rank_count_is_8": len(oracles["rows"]) == 8,
            "oracle_observed_max_equals_max_rank_value": oracles["oracle_max_relative_l2"] == max(oracles["oracle_relative_l2_by_rank"]),
            "oracle_observed_max_is_distinct_from_tolerance": oracles["oracle_max_relative_l2"] != ORACLE_RELATIVE_L2_TOLERANCE,
            "oracle_observed_max_below_tolerance": oracles["oracle_max_relative_l2"] < ORACLE_RELATIVE_L2_TOLERANCE,
            "matched_pair_count_is_9": mutants["matched_pair_count"] == 9,
            "timeline_count_is_96": storage["timeline_count"] == 96,
            "phase_artifact_count_is_288": storage["phase_artifact_count"] == 288,
            "pointer_free_storage_overlap_checks_passed": storage["all_pointer_free_overlap_checks_passed"],
        },
    }
    require(all(summary["invariants"].values()), "derived-summary invariant failure")
    write_json(output / "derived_summary_v2.json", summary)
    write_json(output / "factorial_96.json", factorial)
    write_json(output / "oracle_8.json", oracles)
    write_json(output / "matched_pairs_9.json", mutants)
    write_json(output / "storage_timeline_validation.json", storage)
    write_json(output / "figure_2_inputs.json", {
        "schema_version": "rr2-figure-2-inputs-v2",
        "oracle_relative_l2_tolerance": ORACLE_RELATIVE_L2_TOLERANCE,
        "oracle_relative_l2_by_rank": oracles["oracle_relative_l2_by_rank"],
        "factorial_arm_count": 4,
        "resident_counts": list(RESIDENT_COUNTS),
        "book_rank_count": 8,
        "all_factorial_semantics_exact": True,
        "matched_clean_pass_count": 9,
        "mutant_expected_gate_detection_count": 9,
        "parent_aggregate_raw_sha256": parent_raw_sha,
    })
    render_tables(memory, mutants, output)
    write_json(output / "raw_ledger_validation.json", ledger)
    write_json(output / "executed_source_validation.json", source_closure)
    manifest = build_release_manifest(package_root, output)
    write_json(output / "release_manifest.json", manifest)
    if refresh_package_manifest:
        complete_manifest = build_final_manifest(package_root)
        write_json(package_root / "MANIFEST.json", complete_manifest)
        manifest_sha = file_sha(package_root / "MANIFEST.json")
        (package_root / "MANIFEST.sha256").write_text(
            f"{manifest_sha}  MANIFEST.json\n", encoding="utf-8"
        )
    else:
        manifest_sha = file_sha(package_root / "MANIFEST.json")
    result = {
        "passed": True,
        "raw_artifacts": ledger["artifact_count"],
        "raw_bytes": ledger["total_bytes"],
        "executed_source_files": source_closure["source_file_count"],
        "factorial_configurations": factorial["configuration_count"],
        "oracle_values": len(oracles["rows"]),
        "matched_pairs": mutants["matched_pair_count"],
        "timelines": storage["timeline_count"],
        "phase_artifacts": storage["phase_artifact_count"],
        "oracle_relative_l2_tolerance": ORACLE_RELATIVE_L2_TOLERANCE,
        "oracle_observed_max_relative_l2": oracles["oracle_max_relative_l2"],
        "parent_aggregate_raw_sha256": parent_raw_sha,
        "complete_manifest_sha256": manifest_sha,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--refresh-package-manifest", action="store_true")
    args = parser.parse_args()
    root = args.package_root.resolve()
    output = (args.output or root / "derived").resolve()
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    replay(root, output, refresh_package_manifest=args.refresh_package_manifest)


if __name__ == "__main__":
    main()
