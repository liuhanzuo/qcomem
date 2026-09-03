from __future__ import annotations

"""Independent expected-slot census for the archived R33 H20 capture.

This verifier intentionally imports no R33 producer, manifest, observer, model,
or replay module.  The expected slot set is derived only from the separately
frozen R39 protocol's model geometry and request schedule.  The producer's
``cell.slot_manifest`` is checked as an untrusted receipt, never consumed to
construct expected coverage.
"""

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping, Sequence


PROTOCOL_SCHEMA = "forkaudit-r39-independent-slot-census-protocol-v1"
AUDIT_SCHEMA = "forkaudit-r39-independent-slot-census-audit-v1"
SEMANTIC_FIELDS = (
    "owner_kind",
    "request_index",
    "layer_index",
    "state_family",
    "state_index",
)
REQUIRED_ROW_FIELDS = frozenset(
    {
        "slot_id",
        *SEMANTIC_FIELDS,
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
        "storage_token",
        "view_token",
    }
)


class AuditFailure(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise AuditFailure(code, message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_coordinate(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in SEMANTIC_FIELDS)


def row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    request_index = -1 if row["request_index"] is None else int(row["request_index"])
    return (
        str(row["owner_kind"]),
        request_index,
        int(row["layer_index"]),
        str(row["state_family"]),
        int(row["state_index"]),
        str(row["slot_id"]),
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
        [list(row_key(left)[:-1]), list(row_key(right)[:-1]), relation(left, right)]
        for left, right in combinations(ordered, 2)
    ]


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require(
        protocol.get("schema_version") == PROTOCOL_SCHEMA,
        "protocol_schema",
        "R39 protocol schema drift",
    )
    geometry = protocol.get("model_geometry")
    schedule = protocol.get("schedule")
    counts = protocol.get("expected_counts")
    require(isinstance(geometry, dict), "protocol_geometry", "model geometry missing")
    require(isinstance(schedule, dict), "protocol_schedule", "schedule missing")
    require(isinstance(counts, dict), "protocol_counts", "expected counts missing")
    require(
        int(geometry.get("num_hidden_layers", 0)) > 0,
        "protocol_geometry",
        "hidden-layer count missing",
    )
    require(
        int(schedule.get("resident_count", 0)) > 0,
        "protocol_schedule",
        "resident count missing",
    )
    require(
        set(geometry.get("state_families", {})) == {"conv", "recurrent"},
        "protocol_geometry",
        "state-family geometry drift",
    )


def derive_linear_layers(protocol: Mapping[str, Any]) -> list[int]:
    geometry = protocol["model_geometry"]
    layer_count = int(geometry["num_hidden_layers"])
    period = int(geometry["full_attention_period"])
    offset = int(geometry["full_attention_offset"])
    require(period > 0 and 0 <= offset < period, "protocol_geometry", "layer period drift")
    layers = [index for index in range(layer_count) if index % period != offset]
    require(
        len(layers) == int(geometry["expected_linear_layer_count"]),
        "derived_layer_count",
        "derived linear-layer count drift",
    )
    return layers


def opaque_slot_id(protocol: Mapping[str, Any], coordinate: Mapping[str, Any]) -> str:
    derivation = protocol["slot_id_derivation"]
    payload = {"domain": derivation["domain"], **coordinate}
    suffix = sha256_json(payload)[: int(derivation["sha256_hex_characters"])]
    return str(derivation["prefix"]) + suffix


def derive_expected_census(protocol: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Derive semantic slots without reading any producer-emitted row/manifest."""

    validate_protocol(protocol)
    geometry = protocol["model_geometry"]
    schedule = protocol["schedule"]
    layers = derive_linear_layers(protocol)
    resident_count = int(schedule["resident_count"])
    owners: list[tuple[str, int | None]] = [("persistent", None)]
    owners.extend(("request", index) for index in range(resident_count))
    state_index = int(geometry["state_index"])
    census: dict[str, dict[str, Any]] = {}
    for owner_kind, request_index in owners:
        for layer_index in layers:
            for state_family in sorted(geometry["state_families"]):
                coordinate = {
                    "owner_kind": owner_kind,
                    "request_index": request_index,
                    "layer_index": layer_index,
                    "state_family": state_family,
                    "state_index": state_index,
                }
                slot_id = opaque_slot_id(protocol, coordinate)
                require(slot_id not in census, "derived_slot_collision", "opaque slot-id collision")
                census[slot_id] = {"slot_id": slot_id, **coordinate}
    require(
        len(census) == int(protocol["expected_counts"]["slots_per_capture"]),
        "derived_slot_count",
        "derived census cardinality drift",
    )
    return census


def _validate_manifest_as_untrusted_receipt(
    manifest: Mapping[str, Any], expected: Mapping[str, Mapping[str, Any]]
) -> str:
    require(isinstance(manifest, dict), "manifest_missing", "producer manifest missing")
    unsigned = dict(manifest)
    manifest_sha = unsigned.pop("manifest_sha256", None)
    require(
        sha256_json(unsigned) == manifest_sha,
        "manifest_digest",
        "producer manifest self-digest drift",
    )
    rows = manifest.get("slots")
    require(isinstance(rows, list), "manifest_missing", "producer manifest slots missing")
    observed_ids = [str(row.get("slot_id")) for row in rows]
    require(
        len(observed_ids) == len(set(observed_ids)),
        "manifest_duplicate_slot_id",
        "producer manifest duplicates a slot id",
    )
    require(
        set(observed_ids) == set(expected),
        "manifest_slot_set_mismatch",
        "producer manifest differs from the independent census",
    )
    for row in rows:
        slot_id = str(row["slot_id"])
        observed_coordinate = {field: row[field] for field in SEMANTIC_FIELDS}
        expected_coordinate = {field: expected[slot_id][field] for field in SEMANTIC_FIELDS}
        require(
            observed_coordinate == expected_coordinate,
            "manifest_semantic_binding_mismatch",
            f"producer manifest relabels {slot_id}",
        )
    return str(manifest_sha)


def _validate_geometry(row: Mapping[str, Any], protocol: Mapping[str, Any]) -> None:
    family = str(row["state_family"])
    expected = protocol["model_geometry"]["state_families"][family]
    exact_fields = (
        "shape",
        "storage_offset",
        "dtype",
        "storage_nbytes",
        "tensor_nbytes",
        "byte_start",
        "byte_end_exclusive",
    )
    for field in exact_fields:
        require(
            row[field] == expected[field],
            "descriptor_geometry_mismatch",
            f"{row['slot_id']} {field} differs from frozen {family} geometry",
        )
    require(
        row["stride"] in expected["allowed_strides"],
        "descriptor_geometry_mismatch",
        f"{row['slot_id']} stride differs from frozen {family} geometry",
    )
    require(
        str(row["device"]).startswith(str(expected["device_prefix"])),
        "descriptor_geometry_mismatch",
        f"{row['slot_id']} device differs from frozen geometry",
    )
    require(
        isinstance(row["content_sha256"], str) and len(row["content_sha256"]) == 64,
        "content_digest_shape",
        f"{row['slot_id']} content digest malformed",
    )


def validate_capture(
    capture: Mapping[str, Any],
    expected: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
    *,
    expected_capture_id: str,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    require(
        capture.get("capture_id") == expected_capture_id,
        "capture_id_mismatch",
        "capture id differs from independent schedule",
    )
    require(
        capture.get("slot_manifest_sha256") == expected_manifest_sha256,
        "capture_manifest_binding",
        "capture/manifest receipt drift",
    )
    require(capture.get("process_separated") is True, "pid_separation", "capture not process separated")
    require(
        capture.get("observer_pid") != capture.get("producer_pid"),
        "pid_separation",
        "observer and producer PIDs coincide",
    )
    require(
        capture.get("receiver_derived_descriptors") is True,
        "descriptor_source",
        "descriptors were not receiver-derived",
    )
    require(
        capture.get("receiver_derived_relations") is True,
        "relation_source",
        "relations were not receiver-derived",
    )
    rows = capture.get("rows")
    require(isinstance(rows, list), "rows_missing", "capture rows missing")
    require(
        sha256_json(rows) == capture.get("rows_sha256"),
        "rows_digest",
        "capture row digest drift",
    )
    observed_ids = [str(row.get("slot_id")) for row in rows]
    require(
        len(observed_ids) == len(set(observed_ids)),
        "duplicate_slot_id",
        "capture duplicates at least one opaque slot id",
    )
    missing = sorted(set(expected) - set(observed_ids))
    unexpected = sorted(set(observed_ids) - set(expected))
    require(
        not missing and not unexpected,
        "slot_set_mismatch",
        f"capture slot set mismatch: missing={missing[:3]}, unexpected={unexpected[:3]}",
    )
    require(
        len(rows) == int(protocol["expected_counts"]["slots_per_capture"]),
        "row_cardinality",
        "capture row cardinality drift",
    )
    for row in rows:
        require(
            isinstance(row, dict) and set(row) == REQUIRED_ROW_FIELDS,
            "row_field_set",
            "receiver row field set drift",
        )
        slot_id = str(row["slot_id"])
        observed_coordinate = {field: row[field] for field in SEMANTIC_FIELDS}
        expected_coordinate = {field: expected[slot_id][field] for field in SEMANTIC_FIELDS}
        require(
            observed_coordinate == expected_coordinate,
            "semantic_binding_mismatch",
            f"receiver row relabels {slot_id}: observed={observed_coordinate}, expected={expected_coordinate}",
        )
        _validate_geometry(row, protocol)
    vector = relation_vector(rows)
    expected_relation_count = int(
        protocol["expected_counts"]["unordered_relations_per_capture"]
    )
    require(
        len(vector) == expected_relation_count == int(capture.get("relation_count", -1)),
        "relation_cardinality",
        "relation cardinality drift",
    )
    require(
        sha256_json(vector) == capture.get("relation_vector_sha256"),
        "relation_digest",
        "receiver relation digest drift",
    )
    return {
        "capture_id": expected_capture_id,
        "slot_set_complete": True,
        "slot_ids_unique": True,
        "semantic_bindings_exact": True,
        "descriptor_geometry_exact": True,
        "row_count": len(rows),
        "relation_count": len(vector),
    }


def audit_result(result: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    validate_protocol(protocol)
    source = protocol["source_evidence"]
    require(
        result.get("schema_version") == source["source_result_schema"],
        "result_schema",
        "source result schema drift",
    )
    require(
        result.get("preregistration_sha256") == source["preregistration_sha256"],
        "preregistration_binding",
        "source result/preregistration binding drift",
    )
    expected = derive_expected_census(protocol)
    expected_policies = list(protocol["schedule"]["policy_cells"])
    cells = result.get("cells")
    require(isinstance(cells, list), "cells_missing", "source policy cells missing")
    by_policy = {str(cell.get("policy")): cell for cell in cells}
    require(
        len(by_policy) == len(cells),
        "duplicate_policy_cell",
        "duplicate policy cell",
    )
    require(
        set(by_policy) == set(expected_policies),
        "policy_cell_set",
        "policy-cell set differs from independent schedule",
    )
    expected_plan = list(protocol["schedule"]["captures"])
    cell_reports: list[dict[str, Any]] = []
    for policy in expected_policies:
        cell = by_policy[policy]
        require(
            cell.get("capture_plan") == expected_plan,
            "capture_plan_mismatch",
            f"{policy} capture plan differs from independent schedule",
        )
        manifest_sha = _validate_manifest_as_untrusted_receipt(
            cell.get("slot_manifest"), expected
        )
        captures = cell.get("captures")
        require(isinstance(captures, list), "captures_missing", f"{policy} captures missing")
        captures_by_id = {str(capture.get("capture_id")): capture for capture in captures}
        require(
            len(captures_by_id) == len(captures),
            "duplicate_capture_id",
            f"{policy} duplicate capture id",
        )
        require(
            set(captures_by_id) == {row["capture_id"] for row in expected_plan},
            "capture_set_mismatch",
            f"{policy} capture set differs from independent schedule",
        )
        capture_reports = [
            validate_capture(
                captures_by_id[plan_row["capture_id"]],
                expected,
                protocol,
                expected_capture_id=plan_row["capture_id"],
                expected_manifest_sha256=manifest_sha,
            )
            for plan_row in expected_plan
        ]
        cell_reports.append(
            {
                "policy": policy,
                "producer_manifest_used_as_expectation": False,
                "producer_manifest_matches_independent_census": True,
                "capture_reports": capture_reports,
            }
        )
    total_rows = sum(
        report["row_count"]
        for cell in cell_reports
        for report in cell["capture_reports"]
    )
    total_relations = sum(
        report["relation_count"]
        for cell in cell_reports
        for report in cell["capture_reports"]
    )
    require(
        total_rows == int(protocol["expected_counts"]["row_observations"]),
        "total_row_cardinality",
        "total row observation count drift",
    )
    require(
        total_relations == int(protocol["expected_counts"]["unordered_relations"]),
        "total_relation_cardinality",
        "total relation observation count drift",
    )
    census_rows = [expected[slot_id] for slot_id in sorted(expected)]
    return {
        "schema_version": AUDIT_SCHEMA,
        "passed": True,
        "experiment_id": protocol["experiment_id"],
        "protocol_semantic_sha256": sha256_json(protocol),
        "expected_census_sha256": sha256_json(census_rows),
        "derived_linear_layer_indices": derive_linear_layers(protocol),
        "derived_slots_per_capture": len(expected),
        "audited_policy_cells": len(cell_reports),
        "audited_captures": sum(len(cell["capture_reports"]) for cell in cell_reports),
        "audited_row_observations": total_rows,
        "audited_relation_observations": total_relations,
        "producer_manifest_used_as_expectation": False,
        "producer_rows_used_as_expected_census": False,
        "cell_reports": cell_reports,
        "claim_boundary": protocol["independence_contract"]["claim_boundary"],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    path.write_text(payload, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--census-output", type=Path)
    parser.add_argument(
        "--expected-input-sha256",
        help=(
            "Expected raw digest for a fresh formal capture. If omitted, the "
            "archived R33 digest frozen in protocol.json is required."
        ),
    )
    parser.add_argument(
        "--expected-census-sha256",
        help="Preexecution semantic census digest required by a fresh formal run.",
    )
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    source = protocol["source_evidence"]
    expected_input_sha256 = (
        args.expected_input_sha256
        if args.expected_input_sha256 is not None
        else source["raw_capture_sha256"]
    )
    require(
        sha256_file(args.input) == expected_input_sha256,
        "raw_input_digest",
        "H20 raw capture digest drift",
    )
    require(
        sha256_file(args.preregistration) == source["preregistration_sha256"],
        "preregistration_digest",
        "archived preregistration digest drift",
    )
    result = json.loads(args.input.read_text(encoding="utf-8"))
    report = audit_result(result, protocol)
    if args.expected_census_sha256 is not None:
        require(
            report["expected_census_sha256"] == args.expected_census_sha256,
            "preexecution_census_binding",
            "fresh capture audit differs from preexecution census",
        )
    report["protocol_raw_sha256"] = sha256_file(args.protocol)
    report["input_raw_sha256"] = sha256_file(args.input)
    report["preregistration_raw_sha256"] = sha256_file(args.preregistration)
    report["fresh_formal_input_digest_supplied"] = args.expected_input_sha256 is not None
    report["preexecution_census_bound"] = args.expected_census_sha256 is not None
    write_json(args.output, report)
    if args.census_output is not None:
        census = derive_expected_census(protocol)
        write_json(
            args.census_output,
            {
                "schema_version": "forkaudit-r39-derived-slot-census-v1",
                "protocol_raw_sha256": sha256_file(args.protocol),
                "producer_manifest_used": False,
                "slot_count": len(census),
                "slots": [census[slot_id] for slot_id in sorted(census)],
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
