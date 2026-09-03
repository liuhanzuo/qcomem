from __future__ import annotations

"""Independent live observer for ForkAudit GDN ownership.

This module deliberately imports no ForkAudit or QComem implementation.  It
traverses the live cache objects through their public ``layers`` and state-map
attributes, then derives tensor descriptors, opaque storage identities, byte
overlap relations, and lifecycle transitions directly from PyTorch tensors.

Raw addresses and the per-process HMAC secret are never serialized.  Setup
tensor objects are retained until the experiment finishes so allocator address
reuse cannot make a real functional rebind look unchanged (an ABA collision).
"""

import hashlib
import hmac
import json
import secrets
from dataclasses import dataclass, field
from itertools import combinations
from math import prod
from typing import Any, Mapping, Sequence

import torch


SCHEMA_VERSION = "forkaudit-independent-gdn-observer-v1"
COMPARISON_SCHEMA_VERSION = "forkaudit-independent-producer-comparison-v1"
LIFECYCLE_SCHEMA_VERSION = "forkaudit-independent-gdn-lifecycle-v1"

PHASE_SETUP = "setup_pre_transition"
PHASE_TRANSITION = "post_transition"
PHASE_GENERATION = "post_generation"
PHASES = (PHASE_SETUP, PHASE_TRANSITION, PHASE_GENERATION)

POLICY_SHARED = "shared-base"
POLICY_MATERIALIZED = "materialized"
_SHARED_ALIASES = {
    POLICY_SHARED,
    "shared_base",
    "borrowed-immutable",
    "borrow-immutable-base-functional-rebind",
}
_MATERIALIZED_ALIASES = {
    POLICY_MATERIALIZED,
    "materialized-copy",
    "materialize-request-base-functional-rebind",
}
_STATE_ATTRIBUTES = {"conv": "conv_states", "recurrent": "recurrent_states"}


class IndependentObserverError(RuntimeError):
    """The independent observation is malformed or contradicts its contract."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise IndependentObserverError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def normalize_policy(value: str) -> str:
    require(isinstance(value, str), "policy must be a string")
    if value in _SHARED_ALIASES:
        return POLICY_SHARED
    if value in _MATERIALIZED_ALIASES:
        return POLICY_MATERIALIZED
    raise IndependentObserverError(f"unsupported policy: {value!r}")


def _tensor_payload(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .contiguous()
        .cpu()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )


def _byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    shape = tuple(int(value) for value in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    require(shape and len(shape) == len(stride), "tensor shape/stride rank drift")
    require(all(value > 0 for value in shape), "state tensor contains an empty axis")
    minimum = int(tensor.storage_offset())
    maximum = minimum
    for size, step in zip(shape, stride):
        displacement = (size - 1) * step
        minimum += min(displacement, 0)
        maximum += max(displacement, 0)
    require(minimum >= 0 and maximum >= minimum, "tensor view lies outside storage")
    element_size = int(tensor.element_size())
    return minimum * element_size, (maximum + 1) * element_size


def row_key(row: Mapping[str, Any]) -> tuple[str, int, int, str, int]:
    request_index = -1 if row["request_index"] is None else int(row["request_index"])
    return (
        str(row["owner_kind"]),
        request_index,
        int(row["layer_index"]),
        str(row["state_family"]),
        int(row["state_index"]),
    )


def _key_json(key: tuple[str, int, int, str, int]) -> list[Any]:
    return list(key)


def _descriptor_without_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "owner_kind",
        "request_index",
        "layer_index",
        "state_family",
        "state_index",
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
    return {field: row[field] for field in fields}


def _identity_field(row: Mapping[str, Any]) -> str:
    if "storage_token" in row:
        return "storage_token"
    if "storage_id" in row:
        return "storage_id"
    raise IndependentObserverError("row lacks opaque storage identity")


def relation(left: Mapping[str, Any], right: Mapping[str, Any]) -> str:
    left_identity = _identity_field(left)
    right_identity = _identity_field(right)
    same_storage = left[left_identity] == right[right_identity]
    overlap = (
        same_storage
        and int(left["byte_start"]) < int(right["byte_end_exclusive"])
        and int(right["byte_start"]) < int(left["byte_end_exclusive"])
    )
    if not overlap:
        return "disjoint"
    same_view_fields = (
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
    if all(left[field] == right[field] for field in same_view_fields):
        return "exact_alias"
    return "partial_overlap"


def _relation_vector(rows: Sequence[Mapping[str, Any]]) -> list[list[Any]]:
    ordered = sorted(rows, key=row_key)
    return [
        [_key_json(row_key(left)), _key_json(row_key(right)), relation(left, right)]
        for left, right in combinations(ordered, 2)
    ]


def _row_map(snapshot: Mapping[str, Any]) -> dict[tuple[str, int, int, str, int], Mapping[str, Any]]:
    rows = snapshot.get("rows")
    require(isinstance(rows, list) and rows, "snapshot rows are missing")
    result = {row_key(row): row for row in rows}
    require(len(result) == len(rows), "snapshot contains duplicate tensor coordinates")
    return result


def _owner_rows(
    rows: Mapping[tuple[str, int, int, str, int], Mapping[str, Any]],
    owner_kind: str,
    request_index: int,
) -> list[Mapping[str, Any]]:
    target = -1 if owner_kind == "persistent" else request_index
    return [
        row
        for key, row in rows.items()
        if key[0] == owner_kind and key[1] == target
    ]


def _coordinate_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str, int], Mapping[str, Any]]:
    return {
        (int(row["layer_index"]), str(row["state_family"]), int(row["state_index"])): row
        for row in rows
    }


@dataclass
class ObserverSession:
    """One-process observer with opaque stable identities across phases."""

    _secret: bytes = field(default_factory=lambda: secrets.token_bytes(32), repr=False)
    _setup_tensor_refs: dict[tuple[str, int, int, str, int], torch.Tensor] = field(
        default_factory=dict, init=False, repr=False
    )

    @property
    def commitment(self) -> str:
        return hashlib.sha256(self._secret).hexdigest()

    def _token(self, domain: str, payload: str) -> str:
        return hmac.new(
            self._secret,
            f"{domain}\0{payload}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def capture(
        self,
        persistent: Any,
        requests: Sequence[Any],
        layer_indices: Sequence[int],
        *,
        phase: str,
        policy: str,
        completed_request_indices: Sequence[int],
        state_index: int = 0,
    ) -> dict[str, Any]:
        require(phase in PHASES, f"unsupported phase: {phase!r}")
        canonical_policy = normalize_policy(policy)
        indices = tuple(int(value) for value in layer_indices)
        require(indices and len(set(indices)) == len(indices), "invalid layer plan")
        require(isinstance(requests, (list, tuple)) and requests, "requests are missing")
        completed = tuple(int(value) for value in completed_request_indices)
        require(
            completed == tuple(sorted(set(completed)))
            and all(0 <= value < len(requests) for value in completed),
            "completed request indices are malformed",
        )
        if phase == PHASE_SETUP:
            require(not completed, "setup cannot contain completed requests")
        elif phase == PHASE_GENERATION:
            require(completed == tuple(range(len(requests))), "generation must complete all requests")
        else:
            require(bool(completed), "transition requires a completed request")

        owners: list[tuple[str, int | None, Any]] = [("persistent", None, persistent)]
        owners.extend(("request", index, value) for index, value in enumerate(requests))
        rows: list[dict[str, Any]] = []
        live_refs: dict[tuple[str, int, int, str, int], torch.Tensor] = {}
        for owner_kind, request_index, owner in owners:
            layers = getattr(owner, "layers", None)
            require(isinstance(layers, (list, tuple)), "owner layers are not a sequence")
            for layer_index in indices:
                require(0 <= layer_index < len(layers), "owner layer is missing")
                layer = layers[layer_index]
                for family, attribute in _STATE_ATTRIBUTES.items():
                    states = getattr(layer, attribute, None)
                    require(isinstance(states, dict) and state_index in states, "state mapping is missing")
                    tensor = states[state_index]
                    require(
                        isinstance(tensor, torch.Tensor)
                        and tensor.is_floating_point()
                        and tensor.numel() > 0,
                        "state value is not a non-empty floating tensor",
                    )
                    storage = tensor.untyped_storage()
                    storage_nbytes = int(storage.nbytes())
                    byte_start, byte_end = _byte_interval(tensor)
                    require(0 <= byte_start < byte_end <= storage_nbytes, "view exceeds storage")
                    storage_payload = (
                        f"{tensor.device}\0{int(storage.data_ptr())}\0{storage_nbytes}"
                    )
                    key = (owner_kind, -1 if request_index is None else request_index, layer_index, family, state_index)
                    live_refs[key] = tensor
                    rows.append(
                        {
                            "owner_kind": owner_kind,
                            "request_index": request_index,
                            "layer_index": layer_index,
                            "state_family": family,
                            "state_index": state_index,
                            "shape": [int(value) for value in tensor.shape],
                            "stride": [int(value) for value in tensor.stride()],
                            "storage_offset": int(tensor.storage_offset()),
                            "dtype": str(tensor.dtype),
                            "device": str(tensor.device),
                            "storage_nbytes": storage_nbytes,
                            "tensor_nbytes": int(prod(tensor.shape)) * int(tensor.element_size()),
                            "byte_start": byte_start,
                            "byte_end_exclusive": byte_end,
                            "content_sha256": hashlib.sha256(_tensor_payload(tensor)).hexdigest(),
                            "storage_token": self._token("storage", storage_payload),
                            "tensor_token": self._token("tensor", str(id(tensor))),
                        }
                    )
        rows.sort(key=row_key)
        if phase == PHASE_SETUP and not self._setup_tensor_refs:
            self._setup_tensor_refs = dict(live_refs)
        expected_rows = (1 + len(requests)) * len(indices) * len(_STATE_ATTRIBUTES)
        require(len(rows) == expected_rows, "observer row cardinality drift")
        snapshot = {
            "schema_version": SCHEMA_VERSION,
            "observer_session_commitment_sha256": self.commitment,
            "phase": phase,
            "policy": canonical_policy,
            "layer_indices": list(indices),
            "state_index": state_index,
            "resident_count": len(requests),
            "completed_request_indices": list(completed),
            "setup_tensor_refs_pinned_against_aba": bool(self._setup_tensor_refs),
            "raw_addresses_serialized": False,
            "rows": rows,
            "rows_sha256": sha256_json(rows),
        }
        return snapshot


def validate_snapshot(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    require(snapshot.get("schema_version") == SCHEMA_VERSION, "observer schema drift")
    require(snapshot.get("raw_addresses_serialized") is False, "raw-address claim drift")
    rows = snapshot.get("rows")
    require(isinstance(rows, list) and sha256_json(rows) == snapshot.get("rows_sha256"), "row digest drift")
    forbidden = ("data_ptr", "storage_ptr", "absolute_address", "pointer")
    serialized = json.dumps(snapshot, sort_keys=True).lower()
    require(not any(term in serialized for term in forbidden), "raw address field leaked")
    row_map = _row_map(snapshot)
    expected = (1 + int(snapshot["resident_count"])) * len(snapshot["layer_indices"]) * 2
    require(len(row_map) == expected, "snapshot row coverage drift")
    require(all(len(row["content_sha256"]) == 64 for row in rows), "content digest drift")
    return {
        "row_count": len(rows),
        "row_descriptor_sha256": sha256_json(
            [_descriptor_without_identity(row) for row in sorted(rows, key=row_key)]
        ),
        "relation_count": len(rows) * (len(rows) - 1) // 2,
        "relation_vector_sha256": sha256_json(_relation_vector(rows)),
    }


def _all_disjoint(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for lhs in left:
        for rhs in right:
            count += 1
            require(relation(lhs, rhs) == "disjoint", "ownership sets overlap")
    return count


def _coordinate_exact_alias(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> int:
    left_map = _coordinate_map(left)
    right_map = _coordinate_map(right)
    require(set(left_map) == set(right_map), "owner coordinate coverage differs")
    count = 0
    for coordinate in sorted(left_map):
        count += 1
        require(relation(left_map[coordinate], right_map[coordinate]) == "exact_alias", "expected exact alias is absent")
    return count


def evaluate_phase(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    summary = validate_snapshot(snapshot)
    phase = str(snapshot["phase"])
    policy = normalize_policy(str(snapshot["policy"]))
    resident_count = int(snapshot["resident_count"])
    completed = tuple(int(value) for value in snapshot["completed_request_indices"])
    rows = _row_map(snapshot)
    persistent = _owner_rows(rows, "persistent", -1)
    requests = [_owner_rows(rows, "request", index) for index in range(resident_count)]
    internal = 0
    for owner in [persistent, *requests]:
        for left, right in combinations(owner, 2):
            internal += 1
            require(relation(left, right) == "disjoint", "one owner contains overlapping states")
    exact_aliases = 0
    disjoint = 0
    for request_index, request_rows in enumerate(requests):
        if policy == POLICY_SHARED and request_index not in completed:
            exact_aliases += _coordinate_exact_alias(request_rows, persistent)
        else:
            disjoint += _all_disjoint(request_rows, persistent)
    for left_index, right_index in combinations(range(resident_count), 2):
        left = requests[left_index]
        right = requests[right_index]
        if policy == POLICY_SHARED and left_index not in completed and right_index not in completed:
            exact_aliases += _coordinate_exact_alias(left, right)
        else:
            disjoint += _all_disjoint(left, right)
    return {
        "passed": True,
        "phase": phase,
        "policy": policy,
        "completed_request_indices": list(completed),
        "row_count": summary["row_count"],
        "exact_alias_comparisons": exact_aliases,
        "disjoint_comparisons": disjoint,
        "internal_disjoint_comparisons": internal,
        "row_descriptor_sha256": summary["row_descriptor_sha256"],
        "relation_vector_sha256": summary["relation_vector_sha256"],
    }


def compare_candidate_snapshot(
    observer_snapshot: Mapping[str, Any], candidate_snapshot: Mapping[str, Any]
) -> dict[str, Any]:
    observer_summary = validate_snapshot(observer_snapshot)
    observer_rows = _row_map(observer_snapshot)
    candidate_rows = _row_map(candidate_snapshot)
    require(set(observer_rows) == set(candidate_rows), "candidate/observer coordinate coverage differs")
    descriptor_mismatches: list[list[Any]] = []
    for key in sorted(observer_rows):
        if _descriptor_without_identity(observer_rows[key]) != _descriptor_without_identity(candidate_rows[key]):
            descriptor_mismatches.append(_key_json(key))
    observer_vector = _relation_vector(list(observer_rows.values()))
    candidate_vector = _relation_vector(list(candidate_rows.values()))
    relation_mismatches = [
        index
        for index, (observed, candidate) in enumerate(zip(observer_vector, candidate_vector))
        if observed != candidate
    ]
    candidate_descriptor_sha = sha256_json(
        [_descriptor_without_identity(candidate_rows[key]) for key in sorted(candidate_rows)]
    )
    candidate_relation_sha = sha256_json(candidate_vector)
    passed = not descriptor_mismatches and not relation_mismatches
    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "passed": passed,
        "row_count": len(observer_rows),
        "descriptor_mismatch_count": len(descriptor_mismatches),
        "descriptor_mismatch_examples": descriptor_mismatches[:8],
        "relation_count": len(observer_vector),
        "relation_mismatch_count": len(relation_mismatches),
        "relation_mismatch_indices": relation_mismatches[:8],
        "observer_row_descriptor_sha256": observer_summary["row_descriptor_sha256"],
        "candidate_row_descriptor_sha256": candidate_descriptor_sha,
        "observer_relation_vector_sha256": observer_summary["relation_vector_sha256"],
        "candidate_relation_vector_sha256": candidate_relation_sha,
    }


def _same_row_identity(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["tensor_token"] == right["tensor_token"]
        and left["storage_token"] == right["storage_token"]
        and _descriptor_without_identity(left) == _descriptor_without_identity(right)
    )


def _out_of_place_rebound(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    structural_fields = ("shape", "dtype", "device", "tensor_nbytes")
    return (
        left["tensor_token"] != right["tensor_token"]
        and left["storage_token"] != right["storage_token"]
        and all(left[field] == right[field] for field in structural_fields)
    )


def evaluate_lifecycle(snapshots: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(snapshots) == 3, "lifecycle requires three snapshots")
    require([row["phase"] for row in snapshots] == list(PHASES), "lifecycle phase order drift")
    require(len({row["observer_session_commitment_sha256"] for row in snapshots}) == 1, "observer session changed")
    require(len({normalize_policy(str(row["policy"])) for row in snapshots}) == 1, "policy changed")
    require(len({int(row["resident_count"]) for row in snapshots}) == 1, "resident count changed")
    maps = [_row_map(snapshot) for snapshot in snapshots]
    require(set(maps[0]) == set(maps[1]) == set(maps[2]), "lifecycle coordinate coverage changed")
    persistent_keys = [key for key in sorted(maps[0]) if key[0] == "persistent"]
    request_keys = {
        request_index: [
            key for key in sorted(maps[0]) if key[0] == "request" and key[1] == request_index
        ]
        for request_index in range(int(snapshots[0]["resident_count"]))
    }
    for key in persistent_keys:
        require(_same_row_identity(maps[0][key], maps[1][key]), "persistent state changed at transition")
        require(_same_row_identity(maps[1][key], maps[2][key]), "persistent state changed at generation")
    changed_0 = unchanged_1 = stable_0 = changed_1 = 0
    for key in request_keys[0]:
        require(
            _out_of_place_rebound(maps[0][key], maps[1][key]),
            "request 0 did not out-of-place rebind",
        )
        require(_same_row_identity(maps[1][key], maps[2][key]), "request 0 changed after its only step")
        changed_0 += 1
        stable_0 += 1
    for key in request_keys[1]:
        require(_same_row_identity(maps[0][key], maps[1][key]), "incomplete request 1 changed early")
        require(
            _out_of_place_rebound(maps[1][key], maps[2][key]),
            "request 1 did not out-of-place rebind",
        )
        unchanged_1 += 1
        changed_1 += 1
    phase_reports = [evaluate_phase(snapshot) for snapshot in snapshots]
    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "passed": True,
        "persistent_unchanged_tensor_count": len(persistent_keys),
        "request0_rebound_tensor_count": changed_0,
        "request0_stable_after_completion_tensor_count": stable_0,
        "request1_unchanged_before_completion_tensor_count": unchanged_1,
        "request1_rebound_tensor_count": changed_1,
        "phase_reports": phase_reports,
    }


__all__ = [
    "COMPARISON_SCHEMA_VERSION",
    "IndependentObserverError",
    "LIFECYCLE_SCHEMA_VERSION",
    "ObserverSession",
    "PHASE_GENERATION",
    "PHASE_SETUP",
    "PHASE_TRANSITION",
    "POLICY_MATERIALIZED",
    "POLICY_SHARED",
    "SCHEMA_VERSION",
    "compare_candidate_snapshot",
    "evaluate_lifecycle",
    "evaluate_phase",
    "relation",
    "sha256_json",
    "validate_snapshot",
]
