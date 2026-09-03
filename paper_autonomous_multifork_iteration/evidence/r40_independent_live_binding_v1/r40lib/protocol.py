from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


PREREG_SCHEMA = "forkaudit-r40-independent-live-binding-preregistration-v1"
MANIFEST_SCHEMA = "forkaudit-r40-semantic-slot-manifest-v1"
LANE_SCHEMA = "forkaudit-r40-independent-live-binding-lane-v1"
CAMPAIGN_SCHEMA = "forkaudit-r40-independent-live-binding-campaign-v1"
OBSERVATION_SCHEMA = "forkaudit-r40-live-binding-observation-v1"
LIVE_ITEM_FIELDS = frozenset({"slot_id", "tensor"})
COORDINATE_FIELDS = (
    "owner_kind",
    "request_index",
    "layer_index",
    "state_family",
    "state_index",
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


def coordinate_of(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: value[field] for field in COORDINATE_FIELDS}


def coordinate_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    coordinate = coordinate_of(value)
    request_index = coordinate["request_index"]
    return (
        coordinate["owner_kind"],
        -1 if request_index is None else int(request_index),
        int(coordinate["layer_index"]),
        coordinate["state_family"],
        int(coordinate["state_index"]),
    )


def slot_id(coordinate: Mapping[str, Any]) -> str:
    return "s-" + sha256_json(
        {"domain": "r40-independent-live-binding-slot-v1", **coordinate_of(coordinate)}
    )[:20]


def challenge_nonce(seed_sha256: str, slot: str) -> bytes:
    return hashlib.sha256(
        b"r40-live-binding-nonce-v1\0"
        + bytes.fromhex(seed_sha256)
        + b"\0"
        + slot.encode("ascii")
    ).digest()


def validate_preregistration(value: Mapping[str, Any]) -> None:
    require(value.get("schema_version") == PREREG_SCHEMA, "preregistration schema drift")
    fixture = value.get("fixture")
    require(isinstance(fixture, dict), "fixture missing")
    require(fixture.get("device") == "cpu", "local fixture device drift")
    require(fixture.get("dtype") == "torch.float32", "local fixture dtype drift")
    require(int(fixture.get("resident_count", 0)) == 2, "resident count drift")
    layers = fixture.get("layer_indices")
    require(layers == [0, 1, 2], "layer plan drift")
    families = fixture.get("state_families")
    require(families == ["conv", "recurrent"], "family plan drift")
    faults = value.get("faults")
    require(isinstance(faults, list) and len(faults) == 4, "fault count drift")
    require(len({row["fault_id"] for row in faults}) == 4, "duplicate fault id")
    require(len({row["kind"] for row in faults}) == 4, "duplicate fault kind")


def build_manifest(preregistration: Mapping[str, Any]) -> dict[str, Any]:
    validate_preregistration(preregistration)
    fixture = preregistration["fixture"]
    owners: list[tuple[str, int | None]] = [("persistent", None)]
    owners.extend(
        ("request", request_index)
        for request_index in range(int(fixture["resident_count"]))
    )
    slots: list[dict[str, Any]] = []
    for owner_kind, request_index in owners:
        for layer_index in fixture["layer_indices"]:
            for state_family in fixture["state_families"]:
                coordinate = {
                    "owner_kind": owner_kind,
                    "request_index": request_index,
                    "layer_index": int(layer_index),
                    "state_family": state_family,
                    "state_index": int(fixture["state_index"]),
                }
                slots.append({"slot_id": slot_id(coordinate), **coordinate})
    slots.sort(key=lambda row: row["slot_id"])
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "experiment_id": preregistration["experiment_id"],
        "slots": slots,
    }
    manifest["manifest_sha256"] = sha256_json(manifest)
    validate_manifest(manifest, expected_count=int(fixture["expected_slots_per_lane"]))
    return manifest


def validate_manifest(
    manifest: Mapping[str, Any], *, expected_count: int | None = None
) -> dict[str, Mapping[str, Any]]:
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "manifest schema drift")
    unsigned = dict(manifest)
    observed = unsigned.pop("manifest_sha256", None)
    require(sha256_json(unsigned) == observed, "manifest digest drift")
    slots = manifest.get("slots")
    require(isinstance(slots, list) and slots, "manifest slots missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    coordinates: set[tuple[Any, ...]] = set()
    for row in slots:
        require(isinstance(row, dict), "manifest row type drift")
        coordinate = coordinate_of(row)
        observed_slot = str(row.get("slot_id"))
        require(observed_slot == slot_id(coordinate), "slot/coordinate binding drift")
        require(observed_slot not in by_id, "duplicate slot id")
        key = coordinate_key(coordinate)
        require(key not in coordinates, "duplicate semantic coordinate")
        by_id[observed_slot] = row
        coordinates.add(key)
    if expected_count is not None:
        require(len(by_id) == expected_count, "manifest cardinality drift")
    return by_id


def resolve_slot(manifest: Mapping[str, Any], coordinate: Mapping[str, Any]) -> str:
    by_id = validate_manifest(manifest)
    target = coordinate_key(coordinate)
    matches = [slot for slot, row in by_id.items() if coordinate_key(row) == target]
    require(len(matches) == 1, "coordinate did not resolve exactly once")
    return matches[0]


def validate_live_items(items: Any, manifest: Mapping[str, Any]) -> None:
    require(isinstance(items, list) and items, "live items missing")
    expected = set(validate_manifest(manifest))
    observed: set[str] = set()
    for item in items:
        require(isinstance(item, dict), "live item type drift")
        require(set(item) == LIVE_ITEM_FIELDS, "live item field-set drift")
        slot = str(item.get("slot_id"))
        require(slot in expected, "unknown live slot")
        require(slot not in observed, "duplicate live slot")
        observed.add(slot)
    require(observed == expected, "live slot coverage drift")


def seal_payload(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    output = dict(value)
    output[field] = None
    output[field] = sha256_json(output)
    return output


def verify_sealed_payload(value: Mapping[str, Any], field: str) -> None:
    unsigned = dict(value)
    observed = unsigned.get(field)
    unsigned[field] = None
    require(observed == sha256_json(unsigned), f"{field} drift")

