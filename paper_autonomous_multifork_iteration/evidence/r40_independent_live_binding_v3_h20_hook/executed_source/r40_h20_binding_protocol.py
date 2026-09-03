from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


PREREG_SCHEMA = "forkaudit-r40-h20-preserialization-preregistration-v1"
REGISTRATION_SCHEMA = "forkaudit-r40-h20-raw-container-event-v1"
OBSERVATION_SCHEMA = "forkaudit-r40-h20-binding-observation-v1"
COORDINATE_FIELDS = (
    "owner_kind",
    "request_index",
    "layer_index",
    "state_family",
    "state_index",
)


class BindingProtocolError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingProtocolError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def coordinate(value: Mapping[str, Any]) -> dict[str, Any]:
    return {field: value[field] for field in COORDINATE_FIELDS}


def coordinate_key(value: Mapping[str, Any]) -> tuple[Any, ...]:
    row = coordinate(value)
    return tuple(row[field] for field in COORDINATE_FIELDS)


def protocol_slot_id(value: Mapping[str, Any]) -> str:
    return "s-" + sha256_json(
        {"domain": "r40-h20-preserialization-slot-v1", **coordinate(value)}
    )[:20]


def challenge_nonce(seed_sha256: str, slot_id: str) -> bytes:
    return hashlib.sha256(
        b"r40-h20-live-binding-nonce-v1\0"
        + bytes.fromhex(seed_sha256)
        + b"\0"
        + slot_id.encode("ascii")
    ).digest()


def validate_preregistration(value: Mapping[str, Any]) -> None:
    require(value.get("schema_version") == PREREG_SCHEMA, "preregistration schema drift")
    selected = value.get("selected_coordinates")
    require(isinstance(selected, list) and len(selected) == 6, "selected coordinate drift")
    keys = [coordinate_key(row) for row in selected]
    require(len(set(keys)) == len(keys), "duplicate selected coordinate")
    for row in selected:
        require(row["owner_kind"] in {"persistent", "request"}, "owner kind drift")
        require(row["state_family"] in {"conv", "recurrent"}, "family drift")
        require(int(row["state_index"]) == 0, "state index drift")
        require(int(row["layer_index"]) % 4 != 3, "selected full-attention layer")
    faults = value.get("faults")
    require(isinstance(faults, list) and len(faults) == 4, "fault plan drift")
    require(len({row["fault_id"] for row in faults}) == 4, "duplicate fault id")


def selected_by_event(
    preregistration: Mapping[str, Any],
    *,
    owner_kind: str,
    request_index: int | None,
    layer_index: int,
) -> list[dict[str, Any]]:
    validate_preregistration(preregistration)
    return [
        coordinate(row)
        for row in preregistration["selected_coordinates"]
        if row["owner_kind"] == owner_kind
        and row["request_index"] == request_index
        and int(row["layer_index"]) == int(layer_index)
    ]


def selected_slot_map(preregistration: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    validate_preregistration(preregistration)
    return {
        protocol_slot_id(row): coordinate(row)
        for row in preregistration["selected_coordinates"]
    }


def seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result[field] = None
    result[field] = sha256_json(result)
    return result


def verify_seal(value: Mapping[str, Any], field: str) -> None:
    unsigned = dict(value)
    observed = unsigned.get(field)
    unsigned[field] = None
    require(observed == sha256_json(unsigned), f"{field} drift")


__all__ = [
    "BindingProtocolError",
    "OBSERVATION_SCHEMA",
    "REGISTRATION_SCHEMA",
    "challenge_nonce",
    "coordinate",
    "coordinate_key",
    "protocol_slot_id",
    "require",
    "seal",
    "selected_by_event",
    "selected_slot_map",
    "sha256_json",
    "validate_preregistration",
    "verify_seal",
]

