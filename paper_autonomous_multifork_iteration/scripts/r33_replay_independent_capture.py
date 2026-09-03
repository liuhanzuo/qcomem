from __future__ import annotations

"""Offline evaluator for R33 raw out-of-process captures.

The live observer emits no verdict.  This replay binds opaque capture ids to a
frozen phase/policy plan, recomputes ownership relations, and evaluates the
lifecycle from the archived receiver-derived rows.
"""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence

from r33_ipc_capture_protocol import (
    CAPTURE_SCHEMA,
    FORBIDDEN_JUDGMENT_FIELDS,
    LIVE_REQUEST_FIELDS,
    sha256_json,
)


REPLAY_SCHEMA = "forkaudit-r33-out-of-process-replay-v1"
RESULT_SCHEMA = "forkaudit-r33-out-of-process-result-v1"
PREREG_SCHEMA = "forkaudit-r33-out-of-process-preregistration-v1"
PHASE_SETUP = "setup_pre_transition"
PHASE_TRANSITION = "post_transition"
PHASE_GENERATION = "post_generation"
PHASES = (PHASE_SETUP, PHASE_TRANSITION, PHASE_GENERATION)
POLICY_SHARED = "shared-base"
POLICY_MATERIALIZED = "materialized"


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def row_key(row: Mapping[str, Any]) -> tuple[str, int, int, str, int]:
    request_index = -1 if row["request_index"] is None else int(row["request_index"])
    return (
        str(row["owner_kind"]),
        request_index,
        int(row["layer_index"]),
        str(row["state_family"]),
        int(row["state_index"]),
    )


def relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    overlap = (
        left["storage_token"] == right["storage_token"]
        and int(left["byte_start"]) < int(right["byte_end_exclusive"])
        and int(right["byte_start"]) < int(left["byte_end_exclusive"])
    )
    if not overlap:
        return "disjoint"
    exact_fields = (
        "byte_start",
        "byte_end_exclusive",
        "shape",
        "stride",
        "storage_offset",
        "dtype",
        "device",
        "storage_nbytes",
        "tensor_nbytes",
        "content_sha256",
    )
    if all(left[field] == right[field] for field in exact_fields):
        return "exact_alias"
    return "partial_overlap"


def relation_vector(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    ordered = sorted(rows, key=row_key)
    return [
        [list(row_key(left)), list(row_key(right)), relation(left, right)]
        for left, right in combinations(ordered, 2)
    ]


def validate_capture(capture: Mapping[str, Any]) -> dict[tuple[str, int, int, str, int], Mapping[str, Any]]:
    require(capture.get("schema_version") == CAPTURE_SCHEMA, "capture schema drift")
    require(capture.get("process_separated") is True, "capture is not process separated")
    require(capture.get("observer_pid") != capture.get("producer_pid"), "PID separation drift")
    require(capture.get("receiver_derived_descriptors") is True, "descriptor source drift")
    require(capture.get("receiver_derived_relations") is True, "relation source drift")
    require(capture.get("raw_addresses_serialized") is False, "raw address boundary drift")
    require(capture.get("candidate_verdict_fields_received") is False, "candidate verdict leaked")
    require(capture.get("judgment_fields_received") == [], "judgment field leaked")
    require(
        set(capture.get("live_request_fields_received", [])) == set(LIVE_REQUEST_FIELDS),
        "live request receipt drift",
    )
    require(
        not (set(capture.get("live_request_fields_received", [])) & FORBIDDEN_JUDGMENT_FIELDS),
        "forbidden live request field",
    )
    rows = capture.get("rows")
    require(isinstance(rows, list) and rows, "capture rows missing")
    require(sha256_json(rows) == capture.get("rows_sha256"), "row digest drift")
    vector = relation_vector(rows)
    require(
        sha256_json(vector) == capture.get("relation_vector_sha256"),
        "relation digest drift",
    )
    require(len(vector) == int(capture["relation_count"]), "relation count drift")
    mapped = {row_key(row): row for row in rows}
    require(len(mapped) == len(rows) == int(capture["row_count"]), "row coordinate drift")
    return mapped


def _owner_rows(
    rows: Mapping[tuple[str, int, int, str, int], Mapping[str, Any]],
    owner_kind: str,
    request_index: int,
) -> list[Mapping[str, Any]]:
    target = -1 if owner_kind == "persistent" else request_index
    return [row for key, row in rows.items() if key[0] == owner_kind and key[1] == target]


def _coordinate_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, int], Mapping[str, Any]]:
    return {
        (int(row["layer_index"]), str(row["state_family"]), int(row["state_index"])): row
        for row in rows
    }


def _all_disjoint(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for lhs in left:
        for rhs in right:
            require(relation(lhs, rhs) == "disjoint", "ownership sets overlap")
            count += 1
    return count


def _coordinate_alias(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> int:
    left_map = _coordinate_map(left)
    right_map = _coordinate_map(right)
    require(set(left_map) == set(right_map), "owner coordinate coverage differs")
    for coordinate in left_map:
        require(
            relation(left_map[coordinate], right_map[coordinate]) == "exact_alias",
            "expected exact alias absent",
        )
    return len(left_map)


def evaluate_phase(
    capture: Mapping[str, Any], *, policy: str, completed_request_indices: Sequence[int]
) -> dict[str, Any]:
    rows = validate_capture(capture)
    resident_count = len({key[1] for key in rows if key[0] == "request"})
    completed = tuple(int(value) for value in completed_request_indices)
    persistent = _owner_rows(rows, "persistent", -1)
    requests = [_owner_rows(rows, "request", index) for index in range(resident_count)]
    internal = 0
    for owner in [persistent, *requests]:
        for left, right in combinations(owner, 2):
            require(relation(left, right) == "disjoint", "one owner has overlapping states")
            internal += 1
    exact_aliases = 0
    disjoint = 0
    for request_index, request_rows in enumerate(requests):
        if policy == POLICY_SHARED and request_index not in completed:
            exact_aliases += _coordinate_alias(request_rows, persistent)
        else:
            disjoint += _all_disjoint(request_rows, persistent)
    for left_index, right_index in combinations(range(resident_count), 2):
        if (
            policy == POLICY_SHARED
            and left_index not in completed
            and right_index not in completed
        ):
            exact_aliases += _coordinate_alias(requests[left_index], requests[right_index])
        else:
            disjoint += _all_disjoint(requests[left_index], requests[right_index])
    return {
        "passed": True,
        "capture_id": capture["capture_id"],
        "policy": policy,
        "completed_request_indices": list(completed),
        "row_count": len(rows),
        "exact_alias_comparisons": exact_aliases,
        "disjoint_comparisons": disjoint,
        "internal_disjoint_comparisons": internal,
    }


def _same_view(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    fields = (
        "storage_token",
        "view_token",
        "shape",
        "stride",
        "storage_offset",
        "dtype",
        "device",
        "storage_nbytes",
        "tensor_nbytes",
        "byte_start",
        "byte_end_exclusive",
        "content_sha256",
    )
    return all(left[field] == right[field] for field in fields)


def _storage_rebound(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    structural = ("shape", "dtype", "device", "tensor_nbytes")
    return (
        left["storage_token"] != right["storage_token"]
        and left["view_token"] != right["view_token"]
        and all(left[field] == right[field] for field in structural)
    )


def evaluate_cell(cell: Mapping[str, Any]) -> dict[str, Any]:
    policy = str(cell["policy"])
    require(policy in {POLICY_SHARED, POLICY_MATERIALIZED}, "policy drift")
    plan = cell.get("capture_plan")
    captures = cell.get("captures")
    require(isinstance(plan, list) and len(plan) == 3, "capture plan drift")
    require(isinstance(captures, list) and len(captures) == 3, "capture coverage drift")
    require([row["phase"] for row in plan] == list(PHASES), "phase order drift")
    by_id = {capture["capture_id"]: capture for capture in captures}
    require(len(by_id) == 3, "duplicate capture id")
    ordered = [by_id[row["capture_id"]] for row in plan]
    ready = cell.get("observer_ready_receipt")
    stopped = cell.get("observer_stop_receipt")
    require(isinstance(ready, dict) and ready.get("kind") == "ready", "ready receipt missing")
    require(
        ready.get("process_separated") is True
        and ready.get("observer_pid") != ready.get("producer_pid"),
        "ready PID separation drift",
    )
    require(ready.get("candidate_modules_imported") is False, "candidate import boundary drift")
    require(ready.get("observer_generates_verdicts") is False, "live observer verdict drift")
    require(
        isinstance(stopped, dict)
        and stopped.get("kind") == "stopped"
        and stopped.get("capture_count") == 3
        and stopped.get("pinned_capture_count") == 3,
        "stop/pinning receipt drift",
    )
    require(
        ready.get("observer_pid") == stopped.get("observer_pid")
        == ordered[0].get("observer_pid"),
        "ready/capture/stop observer PID drift",
    )
    maps = [validate_capture(capture) for capture in ordered]
    require(set(maps[0]) == set(maps[1]) == set(maps[2]), "coordinate coverage changed")
    require(
        len({capture["observer_pid"] for capture in ordered}) == 1,
        "observer process changed within a lifecycle",
    )
    require(
        len({capture["observer_session_commitment_sha256"] for capture in ordered}) == 1,
        "observer session changed within a lifecycle",
    )
    require(
        len({capture["slot_manifest_sha256"] for capture in ordered}) == 1,
        "slot manifest changed within a lifecycle",
    )
    persistent_keys = [key for key in maps[0] if key[0] == "persistent"]
    request_indices = sorted({key[1] for key in maps[0] if key[0] == "request"})
    require(request_indices == list(range(len(request_indices))), "request coverage drift")
    request_keys = {
        index: [key for key in maps[0] if key[0] == "request" and key[1] == index]
        for index in request_indices
    }
    for key in persistent_keys:
        require(_same_view(maps[0][key], maps[1][key]), "persistent changed at transition")
        require(_same_view(maps[1][key], maps[2][key]), "persistent changed at generation")
    require(len(request_indices) == 2, "R33 cell requires N=2")
    for key in request_keys[0]:
        require(_storage_rebound(maps[0][key], maps[1][key]), "request 0 did not rebind")
        require(_same_view(maps[1][key], maps[2][key]), "request 0 changed after completion")
    for key in request_keys[1]:
        require(_same_view(maps[0][key], maps[1][key]), "request 1 changed before completion")
        require(_storage_rebound(maps[1][key], maps[2][key]), "request 1 did not rebind")
    phase_reports = [
        evaluate_phase(
            capture,
            policy=policy,
            completed_request_indices=plan_row["completed_request_indices"],
        )
        for capture, plan_row in zip(ordered, plan)
    ]
    return {
        "passed": True,
        "cell_id": cell["cell_id"],
        "policy": policy,
        "observer_pid": ordered[0]["observer_pid"],
        "producer_pid": ordered[0]["producer_pid"],
        "row_observations": sum(report["row_count"] for report in phase_reports),
        "relation_observations": sum(int(capture["relation_count"]) for capture in ordered),
        "persistent_unchanged_rows": len(persistent_keys),
        "request0_rebound_rows": len(request_keys[0]),
        "request1_rebound_rows": len(request_keys[1]),
        "phase_reports": phase_reports,
    }


def _validate_frozen_protocol(
    result: Mapping[str, Any], preregistration: Mapping[str, Any]
) -> None:
    require(
        preregistration.get("schema_version") == PREREG_SCHEMA,
        "preregistration schema drift",
    )
    design = preregistration.get("design")
    require(isinstance(design, dict), "frozen design missing")
    expected_policies = list(design.get("gdn_policy_cells", []))
    cells = result.get("cells", [])
    require(
        [cell.get("policy") for cell in cells] == expected_policies,
        "frozen policy plan drift",
    )
    expected_ids = list(design.get("capture_ids", []))
    expected_plan = list(design.get("capture_plan", []))
    descriptor_contracts = design.get("allowed_descriptor_geometry_by_family")
    for cell in cells:
        require(cell.get("capture_plan") == expected_plan, "frozen capture plan drift")
        require(
            [capture.get("capture_id") for capture in cell.get("captures", [])]
            == expected_ids,
            "frozen capture id order drift",
        )
        manifest = cell.get("slot_manifest", {})
        require(
            manifest.get("layer_indices") == design.get("linear_layer_indices"),
            "frozen layer plan drift",
        )
        require(
            manifest.get("resident_count") == design.get("resident_count"),
            "frozen resident count drift",
        )
        for capture in cell.get("captures", []):
            require(
                capture.get("transport") == design.get("expected_transport"),
                "frozen transport drift",
            )
            require(
                capture.get("row_count") == design.get("rows_per_phase"),
                "frozen row count drift",
            )
            require(
                capture.get("relation_count")
                == design.get("unordered_pair_relations_per_phase"),
                "frozen relation count drift",
            )
            if descriptor_contracts is not None:
                require(isinstance(descriptor_contracts, dict), "descriptor contract drift")
                for row in capture.get("rows", []):
                    contract = descriptor_contracts.get(row.get("state_family"))
                    require(isinstance(contract, dict), "state-family contract missing")
                    for field in (
                        "shape",
                        "storage_offset",
                        "dtype",
                        "storage_nbytes",
                        "tensor_nbytes",
                        "byte_start",
                        "byte_end_exclusive",
                    ):
                        require(row.get(field) == contract.get(field), f"frozen {field} drift")
                    require(
                        row.get("stride") in contract.get("allowed_strides", []),
                        "frozen stride drift",
                    )
                    require(
                        str(row.get("device", "")).startswith(
                            str(contract.get("device_prefix", ""))
                        ),
                        "frozen device drift",
                    )


def replay_result(
    result: Mapping[str, Any],
    preregistration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    require(result.get("schema_version") == RESULT_SCHEMA, "result schema drift")
    if preregistration is not None:
        _validate_frozen_protocol(result, preregistration)
    cells = result.get("cells")
    require(isinstance(cells, list) and cells, "result cells missing")
    reports = [evaluate_cell(cell) for cell in cells]
    return {
        "schema_version": REPLAY_SCHEMA,
        "passed": all(report["passed"] for report in reports),
        "input_result_sha256": sha256_json(result),
        "cell_count": len(reports),
        "row_observations": sum(report["row_observations"] for report in reports),
        "relation_observations": sum(report["relation_observations"] for report in reports),
        "all_observers_process_separated": all(
            report["observer_pid"] != report["producer_pid"] for report in reports
        ),
        "candidate_verdict_fields_authoritative": False,
        "frozen_protocol_bound": preregistration is not None,
        "cell_reports": reports,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--expected-preregistration-sha256")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == args.expected_input_sha256,
        "input result digest drift",
    )
    result = json.loads(raw)
    preregistration = None
    if result.get("preregistration_sha256") is not None:
        require(args.preregistration is not None, "scientific result requires preregistration")
        require(
            args.expected_preregistration_sha256 is not None,
            "scientific result requires expected preregistration digest",
        )
    if args.preregistration is not None:
        prereg_raw = args.preregistration.read_bytes()
        require(
            hashlib.sha256(prereg_raw).hexdigest()
            == args.expected_preregistration_sha256,
            "preregistration digest drift",
        )
        require(
            result.get("preregistration_sha256")
            == args.expected_preregistration_sha256,
            "result/preregistration binding drift",
        )
        preregistration = json.loads(prereg_raw)
    replay = replay_result(result, preregistration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(replay, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
