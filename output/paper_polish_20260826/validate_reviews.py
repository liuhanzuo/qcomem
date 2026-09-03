#!/usr/bin/env python3
"""Validate isolated review JSON files without third-party dependencies."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCHEMA = ROOT.parents[1] / ".agents/skills/autonomous-paper-agent/templates/review.schema.json"
ROLES = {
    "R1": "novelty_positioning",
    "R2": "technical_soundness",
    "R3": "experimental_rigor",
}


def validate(value, schema, path="$"):
    errors = []
    expected_type = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }.get(expected_type, True)
    if not type_ok:
        return [f"{path}: expected {expected_type}, got {type(value).__name__}"]

    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    if expected_type == "string" and len(value) < schema.get("minLength", 0):
        errors.append(f"{path}: shorter than minLength")
    if expected_type == "integer":
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if expected_type == "array":
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{path}: fewer than minItems")
        for index, item in enumerate(value):
            errors.extend(validate(item, schema.get("items", {}), f"{path}[{index}]"))
    if expected_type == "object":
        properties = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required key {key}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    errors.append(f"{path}: unexpected key {key}")
        for key, item in value.items():
            if key in properties:
                errors.extend(validate(item, properties[key], f"{path}.{key}"))
    return errors


def main():
    schema = json.loads(SCHEMA.read_text())
    with (ROOT / "review_snapshot_manifest.tsv").open(newline="") as handle:
        expected_hash = {row["paper_id"]: row["snapshot_sha256"] for row in csv.DictReader(handle, delimiter="\t")}

    errors = []
    files = sorted((ROOT / "reviews").glob("R[123]/*.json"))
    for path in files:
        reviewer = path.parent.name
        paper_id = path.stem
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            errors.append(f"{path}: JSON parse failed: {exc}")
            continue
        for item in validate(payload, schema):
            errors.append(f"{path}: {item}")
        if payload.get("reviewer_id") != reviewer:
            errors.append(f"{path}: reviewer_id does not match directory")
        if payload.get("role") != ROLES.get(reviewer):
            errors.append(f"{path}: role does not match isolated lane")
        if payload.get("round") != 1:
            errors.append(f"{path}: round must be 1")
        if payload.get("snapshot_sha256") != expected_hash.get(paper_id):
            errors.append(f"{path}: snapshot SHA-256 mismatch")

    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print(f"VALID: {len(files)} review JSON files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
