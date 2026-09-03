from __future__ import annotations

"""Frozen wire and slot contracts for the R33 out-of-process observer.

The live producer message deliberately contains only an opaque capture id and
an opaque slot id paired with a tensor.  Owner coordinates are supplied to the
observer at process creation from a separately frozen manifest; policy, phase,
completion state, expected relations, and candidate verdicts are never part of
the live capture message.
"""

import hashlib
import json
from typing import Any, Mapping, Sequence


MANIFEST_SCHEMA = "forkaudit-r33-ipc-slot-manifest-v1"
REQUEST_SCHEMA = "forkaudit-r33-ipc-live-request-v1"
RESPONSE_SCHEMA = "forkaudit-r33-ipc-response-v1"
CAPTURE_SCHEMA = "forkaudit-r33-out-of-process-capture-v1"

STATE_ATTRIBUTES = {"conv": "conv_states", "recurrent": "recurrent_states"}
LIVE_REQUEST_FIELDS = frozenset({"schema_version", "capture_id", "slot_tensors"})
LIVE_SLOT_FIELDS = frozenset({"slot_id", "tensor"})
FORBIDDEN_JUDGMENT_FIELDS = frozenset(
    {
        "phase",
        "policy",
        "completed_request_indices",
        "expected_relation",
        "expected_relations",
        "expected_verdict",
        "candidate_capture",
        "candidate_rows",
        "candidate_verdict",
        "passed",
        "verdict",
    }
)


class ProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ProtocolError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _slot_id(coordinate: Mapping[str, Any]) -> str:
    # Opaque on the wire, deterministic in the frozen manifest.
    return "s-" + sha256_json({"domain": "r33-gdn-slot-v1", **coordinate})[:20]


def build_slot_manifest(
    layer_indices: Sequence[int],
    *,
    resident_count: int,
    capture_ids: Sequence[str],
    state_index: int = 0,
) -> dict[str, Any]:
    layers = tuple(int(value) for value in layer_indices)
    captures = tuple(str(value) for value in capture_ids)
    require(layers and len(layers) == len(set(layers)), "invalid layer plan")
    require(resident_count > 0, "resident_count must be positive")
    require(captures and len(captures) == len(set(captures)), "invalid capture ids")
    owners: list[tuple[str, int | None]] = [("persistent", None)]
    owners.extend(("request", index) for index in range(resident_count))
    slots: list[dict[str, Any]] = []
    for owner_kind, request_index in owners:
        for layer_index in layers:
            for family in STATE_ATTRIBUTES:
                coordinate = {
                    "owner_kind": owner_kind,
                    "request_index": request_index,
                    "layer_index": layer_index,
                    "state_family": family,
                    "state_index": int(state_index),
                }
                slots.append({"slot_id": _slot_id(coordinate), **coordinate})
    slots.sort(key=lambda row: row["slot_id"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "resident_count": int(resident_count),
        "layer_indices": list(layers),
        "state_index": int(state_index),
        "capture_ids": list(captures),
        "slots": slots,
        "live_request_disallowed_judgment_fields": sorted(FORBIDDEN_JUDGMENT_FIELDS),
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema drift")
    unsigned = dict(manifest)
    observed_sha = unsigned.pop("manifest_sha256", None)
    require(sha256_json(unsigned) == observed_sha, "manifest digest drift")
    slots = manifest.get("slots")
    require(isinstance(slots, list) and slots, "manifest slots missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    coordinate_keys: set[tuple[Any, ...]] = set()
    for slot in slots:
        require(isinstance(slot, dict), "manifest slot is not an object")
        coordinate = {
            field: slot[field]
            for field in (
                "owner_kind",
                "request_index",
                "layer_index",
                "state_family",
                "state_index",
            )
        }
        slot_id = str(slot.get("slot_id"))
        require(slot_id == _slot_id(coordinate), "slot id/coordinate binding drift")
        require(slot_id not in by_id, "duplicate slot id")
        key = tuple(coordinate.values())
        require(key not in coordinate_keys, "duplicate semantic coordinate")
        require(coordinate["owner_kind"] in {"persistent", "request"}, "owner kind drift")
        require(coordinate["state_family"] in STATE_ATTRIBUTES, "state family drift")
        by_id[slot_id] = slot
        coordinate_keys.add(key)
    expected = (
        (1 + int(manifest["resident_count"]))
        * len(manifest["layer_indices"])
        * len(STATE_ATTRIBUTES)
    )
    require(len(by_id) == expected, "slot cardinality drift")
    capture_ids = manifest.get("capture_ids")
    require(
        isinstance(capture_ids, list)
        and capture_ids
        and len(capture_ids) == len(set(capture_ids)),
        "capture id plan drift",
    )
    return by_id


def validate_live_request(message: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    require(isinstance(message, dict), "live request is not an object")
    require(set(message) == LIVE_REQUEST_FIELDS, "live request field-set drift")
    require(message.get("schema_version") == REQUEST_SCHEMA, "live request schema drift")
    require(message.get("capture_id") in manifest["capture_ids"], "unknown capture id")
    items = message.get("slot_tensors")
    require(isinstance(items, list) and items, "live slot payload missing")
    expected_slots = set(validate_manifest(manifest))
    observed_slots: set[str] = set()
    for item in items:
        require(isinstance(item, dict), "live slot payload is not an object")
        require(set(item) == LIVE_SLOT_FIELDS, "live slot field-set drift")
        slot_id = item.get("slot_id")
        require(slot_id in expected_slots, "unknown live slot")
        require(slot_id not in observed_slots, "duplicate live slot")
        observed_slots.add(slot_id)
    require(observed_slots == expected_slots, "live slot coverage drift")


__all__ = [
    "CAPTURE_SCHEMA",
    "FORBIDDEN_JUDGMENT_FIELDS",
    "LIVE_REQUEST_FIELDS",
    "MANIFEST_SCHEMA",
    "ProtocolError",
    "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA",
    "STATE_ATTRIBUTES",
    "build_slot_manifest",
    "canonical_bytes",
    "require",
    "sha256_json",
    "validate_live_request",
    "validate_manifest",
]
