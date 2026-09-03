from __future__ import annotations

"""Execute resealed omission, duplication, and semantic-relabel controls."""

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from audit_independent_slot_census import (
    AuditFailure,
    audit_result,
    relation_vector,
    sha256_file,
    sha256_json,
    validate_protocol,
    write_json,
)


CONTROL_SCHEMA = "forkaudit-r39-independent-slot-negative-controls-v1"


def selected_capture(value: Mapping[str, Any]) -> dict[str, Any]:
    cells = value["cells"]
    cell = next(cell for cell in cells if cell["policy"] == "shared-base")
    capture = next(
        capture
        for capture in cell["captures"]
        if capture["capture_id"] == "c-2d8d91660bc7"
    )
    return capture


def reseal_capture(capture: dict[str, Any]) -> None:
    rows = capture["rows"]
    vector = relation_vector(rows)
    capture["row_count"] = len(rows)
    capture["rows_sha256"] = sha256_json(rows)
    capture["relation_count"] = len(vector)
    capture["relation_vector_sha256"] = sha256_json(vector)


def apply_control(pristine: Mapping[str, Any], control_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value = copy.deepcopy(pristine)
    capture = selected_capture(value)
    rows = capture["rows"]
    ordered = sorted(rows, key=lambda row: row["slot_id"])
    if control_id == "C-OMIT-ONE-SLOT":
        target = ordered[0]
        rows.remove(target)
        mutation = {"removed_slot_id": target["slot_id"]}
    elif control_id == "C-DUPLICATE-ONE-SLOT":
        target = ordered[0]
        rows.append(copy.deepcopy(target))
        mutation = {"duplicated_slot_id": target["slot_id"]}
    elif control_id == "C-SEMANTIC-RELABEL":
        candidates = [row for row in ordered if row["state_family"] == "conv"]
        left = candidates[0]
        right = next(
            row
            for row in candidates[1:]
            if (
                row["owner_kind"],
                row["request_index"],
                row["layer_index"],
            )
            != (
                left["owner_kind"],
                left["request_index"],
                left["layer_index"],
            )
        )
        fields = ("owner_kind", "request_index", "layer_index", "state_family", "state_index")
        left_values = {field: left[field] for field in fields}
        right_values = {field: right[field] for field in fields}
        for field in fields:
            left[field] = right_values[field]
            right[field] = left_values[field]
        mutation = {
            "left_slot_id": left["slot_id"],
            "right_slot_id": right["slot_id"],
            "semantic_labels_swapped": True,
            "tensor_descriptors_and_slot_ids_unchanged": True,
        }
    else:
        raise ValueError(f"unknown control id: {control_id}")
    reseal_capture(capture)
    return value, mutation


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-input-sha256",
        help=(
            "Expected pristine digest for a fresh formal capture. If omitted, "
            "the archived R33 digest in protocol.json is required."
        ),
    )
    args = parser.parse_args(argv)
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    validate_protocol(protocol)
    expected_input_sha = (
        args.expected_input_sha256
        if args.expected_input_sha256 is not None
        else protocol["source_evidence"]["raw_capture_sha256"]
    )
    if sha256_file(args.input) != expected_input_sha:
        raise RuntimeError("pristine archived input digest drift")
    pristine = json.loads(args.input.read_text(encoding="utf-8"))
    results = []
    for control in protocol["negative_controls"]:
        control_id = control["control_id"]
        tampered, mutation = apply_control(pristine, control_id)
        # The tampered document's local row and relation digests are resealed.
        # The expected failure must therefore come from the independent census.
        capture = selected_capture(tampered)
        try:
            audit_result(tampered, protocol)
        except AuditFailure as exc:
            observed_code = exc.code
            observed_message = str(exc)
        else:
            observed_code = None
            observed_message = "audit unexpectedly passed"
        expected_code = control["expected_failure_code"]
        passed = observed_code == expected_code
        results.append(
            {
                "control_id": control_id,
                "operation": control["operation"],
                "mutation": mutation,
                "internal_rows_digest_resealed": sha256_json(capture["rows"])
                == capture["rows_sha256"],
                "internal_relation_digest_resealed": sha256_json(
                    relation_vector(capture["rows"])
                )
                == capture["relation_vector_sha256"],
                "expected_failure_code": expected_code,
                "observed_failure_code": observed_code,
                "observed_failure_message": observed_message,
                "failed_closed_as_expected": passed,
            }
        )
    report = {
        "schema_version": CONTROL_SCHEMA,
        "passed": all(row["failed_closed_as_expected"] for row in results),
        "protocol_raw_sha256": sha256_file(args.protocol),
        "pristine_input_raw_sha256": sha256_file(args.input),
        "control_count": len(results),
        "all_internal_digests_resealed": all(
            row["internal_rows_digest_resealed"]
            and row["internal_relation_digest_resealed"]
            for row in results
        ),
        "all_controls_failed_closed": all(
            row["failed_closed_as_expected"] for row in results
        ),
        "controls": results,
    }
    write_json(args.output, report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
