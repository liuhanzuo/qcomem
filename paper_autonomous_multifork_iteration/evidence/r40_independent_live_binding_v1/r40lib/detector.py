from __future__ import annotations

from typing import Any, Mapping

from .protocol import require, verify_sealed_payload


DESCRIPTOR_FIELDS = (
    "shape",
    "stride",
    "storage_offset",
    "dtype",
    "device",
    "tensor_nbytes",
    "storage_nbytes",
    "byte_start",
    "byte_end_exclusive",
)


def detect_binding(
    oracle: Mapping[str, Any], observation: Mapping[str, Any]
) -> dict[str, Any]:
    verify_sealed_payload(oracle, "payload_sha256")
    verify_sealed_payload(observation, "payload_sha256")
    require(oracle.get("role") == "pre_injection_oracle", "oracle role drift")
    require(observation.get("role") == "post_binding_observer", "observer role drift")
    codes: list[str] = []
    oracle_rows = {row["slot_id"]: row for row in oracle["rows"]}
    observed_rows = {row["slot_id"]: row for row in observation["rows"]}
    missing = sorted(set(oracle_rows) - set(observed_rows))
    unexpected = sorted(set(observed_rows) - set(oracle_rows))
    if missing or unexpected:
        codes.append("slot_set_mismatch")
    common = sorted(set(oracle_rows) & set(observed_rows))
    descriptor_mismatches = [
        slot
        for slot in common
        if any(oracle_rows[slot][field] != observed_rows[slot][field] for field in DESCRIPTOR_FIELDS)
    ]
    if descriptor_mismatches:
        codes.append("descriptor_mismatch")
    challenge_mismatches = [
        slot
        for slot in common
        if oracle_rows[slot]["challenge_response_sha256"]
        != observed_rows[slot]["challenge_response_sha256"]
    ]
    if challenge_mismatches:
        codes.append("challenge_response_mismatch")
    oracle_relations = {
        (left, right): relation for left, right, relation in oracle["relations"]
    }
    observed_relations = {
        (left, right): relation for left, right, relation in observation["relations"]
    }
    relation_mismatches = [
        [left, right, oracle_relations.get((left, right)), observed_relations.get((left, right))]
        for left, right in sorted(set(oracle_relations) | set(observed_relations))
        if oracle_relations.get((left, right)) != observed_relations.get((left, right))
    ]
    if relation_mismatches:
        codes.append("storage_relation_mismatch")
    return {
        "schema_version": "forkaudit-r40-live-binding-detector-v1",
        "passed": not codes,
        "failure_codes": codes,
        "missing_slot_ids": missing,
        "unexpected_slot_ids": unexpected,
        "descriptor_mismatch_slot_ids": descriptor_mismatches,
        "challenge_mismatch_slot_ids": challenge_mismatches,
        "relation_mismatch_pairs": relation_mismatches,
        "numeric_tolerance": 0,
    }


__all__ = ["detect_binding"]

