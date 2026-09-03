from __future__ import annotations

"""Serializable storage-ownership witnesses for the 30 Qwen3.5 GDN layers.

The witness deliberately separates capture from replay.  Capture is the only
operation that observes a live ``data_ptr``; it replaces every storage with a
snapshot-local identifier before returning.  Replay accepts JSON-compatible
rows and derives every alias/overlap decision from storage IDs and byte
intervals.  A caller-supplied ``passed`` field is consequently irrelevant.

Byte intervals are conservative bounding intervals for strided tensors.  This
is fail-closed for ownership: two exotic interleaved views may be reported as
overlapping, but an actual byte overlap cannot be reported as disjoint.
"""

import hashlib
import hmac
import json
import re
import secrets
from dataclasses import dataclass, field
from itertools import combinations
from math import prod
from typing import Any, Sequence

import torch


SCHEMA_VERSION = "qcomem-gdn-storage-witness-v1"

PHASE_SETUP_PRE_TRANSITION = "setup_pre_transition"
# Backward-compatible symbol for callers written before the phase name was
# generalized; the serialized schema always uses ``setup_pre_transition``.
PHASE_SETUP_BORROWED_IMMUTABLE = PHASE_SETUP_PRE_TRANSITION
PHASE_POST_TRANSITION = "post_transition"
PHASE_POST_GENERATION = "post_generation"
PHASES = (
    PHASE_SETUP_PRE_TRANSITION,
    PHASE_POST_TRANSITION,
    PHASE_POST_GENERATION,
)
TIMELINE_SCHEMA_VERSION = "qcomem-gdn-storage-timeline-v1"
KV_POLICIES = (
    "vllm-q16-fresh-full-copy-control",
    "vllm-q16-shared-document-reuse",
)

POLICY_SHARED_BASE = "shared-base"
POLICY_MATERIALIZED = "materialized"
POLICIES = (POLICY_SHARED_BASE, POLICY_MATERIALIZED)

GATE_STORAGE_SCHEMA = "gdn_storage_witness_schema"
GATE_PERSISTENT_IMMUTABLE = "gdn_persistent_immutable"
GATE_SHARED_SETUP_EXACT_BASE_ALIAS = "gdn_shared_setup_exact_base_alias"
GATE_SHARED_INCOMPLETE_EXACT_BASE_ALIAS = "gdn_shared_incomplete_exact_base_alias"
GATE_MATERIALIZED_SETUP_BASE_DISJOINT = "gdn_materialized_setup_base_disjoint"
GATE_MATERIALIZED_SETUP_PEERS_DISJOINT = "gdn_materialized_setup_peers_disjoint"
GATE_COMPLETED_VS_BASE_DISJOINT = "gdn_completed_vs_base_disjoint"
GATE_COMPLETED_VS_PEERS_DISJOINT = "gdn_completed_vs_peers_disjoint"
GATE_OWNER_INTERNAL_DISJOINT = "gdn_owner_internal_disjoint"
GATE_COMPLETED_BINDING_REBOUND = "gdn_completed_binding_rebound"
GATE_INCOMPLETE_BINDING_UNCHANGED = "gdn_incomplete_binding_unchanged"

EXPECTED_LINEAR_LAYERS = 30
STATE_FAMILIES = ("conv", "recurrent")
EXPECTED_TENSORS_PER_OWNER = EXPECTED_LINEAR_LAYERS * len(STATE_FAMILIES)

_FAMILY_ATTRIBUTES = {
    "conv": "conv_states",
    "recurrent": "recurrent_states",
}
_FLOAT_DTYPE_NBYTES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GUARD_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_STORAGE_ID_RE = re.compile(r"^storage-[0-9]{4,}$")

_SHARED_POLICY_ALIASES = {
    POLICY_SHARED_BASE,
    "shared_base",
    "borrowed-immutable",
    "borrow-immutable-base-functional-rebind",
}
_MATERIALIZED_POLICY_ALIASES = {
    POLICY_MATERIALIZED,
    "materialized-copy",
    "materialize-request-base-functional-rebind",
}


class GDNStorageWitnessError(RuntimeError):
    """Raised when a GDN storage witness cannot prove its ownership claim."""

    def __init__(self, message: str, *, gate_id: str = GATE_STORAGE_SCHEMA) -> None:
        self.gate_id = gate_id
        super().__init__(f"[{gate_id}] {message}")


def _require(
    condition: bool,
    message: str,
    *,
    gate_id: str = GATE_STORAGE_SCHEMA,
) -> None:
    if not condition:
        raise GDNStorageWitnessError(message, gate_id=gate_id)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _is_int(value: Any) -> bool:
    return type(value) is int


def _validate_layer_indices(layer_indices: Sequence[int]) -> tuple[int, ...]:
    _require(
        isinstance(layer_indices, (list, tuple)),
        "layer_indices must be a list or tuple",
    )
    result = tuple(layer_indices)
    _require(
        len(result) == EXPECTED_LINEAR_LAYERS,
        f"exactly {EXPECTED_LINEAR_LAYERS} GDN layer indices are required",
    )
    _require(
        all(_is_int(index) and index >= 0 for index in result),
        "GDN layer indices must be non-negative non-bool integers",
    )
    _require(len(set(result)) == len(result), "GDN layer indices must be unique")
    return result


def _normalize_policy(policy: str) -> str:
    _require(isinstance(policy, str), "policy must be a string")
    if policy in _SHARED_POLICY_ALIASES:
        return POLICY_SHARED_BASE
    if policy in _MATERIALIZED_POLICY_ALIASES:
        return POLICY_MATERIALIZED
    raise GDNStorageWitnessError(f"unsupported GDN ownership policy: {policy!r}")


def _validate_phase(phase: str) -> str:
    _require(phase in PHASES, f"unsupported GDN witness phase: {phase!r}")
    return phase


def _tensor_digest(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


def _byte_interval(
    *,
    shape: Sequence[int],
    stride: Sequence[int],
    storage_offset: int,
    element_size: int,
) -> tuple[int, int]:
    _require(len(shape) == len(stride), "shape/stride rank mismatch")
    _require(bool(shape), "GDN state tensors must have non-zero rank")
    _require(all(_is_int(size) and size > 0 for size in shape), "invalid tensor shape")
    _require(all(_is_int(step) for step in stride), "invalid tensor stride")
    _require(_is_int(storage_offset) and storage_offset >= 0, "invalid storage_offset")
    _require(_is_int(element_size) and element_size > 0, "invalid element size")
    minimum = storage_offset
    maximum = storage_offset
    for size, step in zip(shape, stride):
        displacement = (size - 1) * step
        minimum += min(displacement, 0)
        maximum += max(displacement, 0)
    _require(minimum >= 0 and maximum >= minimum, "tensor view is outside storage")
    return minimum * element_size, (maximum + 1) * element_size


def _coordinate(layer_index: int, state_family: str, state_index: int) -> str:
    return f"layer:{layer_index}/{state_family}/state:{state_index}"


@dataclass(frozen=True)
class _LiveTensor:
    layer_index: int
    state_family: str
    state_index: int
    tensor: torch.Tensor = field(repr=False, compare=False)

    @property
    def coordinate(self) -> str:
        return _coordinate(self.layer_index, self.state_family, self.state_index)


def _collect_owner_tensors(
    owner: Any,
    layer_indices: tuple[int, ...],
    state_index: int,
    label: str,
) -> tuple[_LiveTensor, ...]:
    layers = getattr(owner, "layers", None)
    _require(isinstance(layers, (list, tuple)), f"{label}.layers must be a sequence")
    result: list[_LiveTensor] = []
    for layer_index in layer_indices:
        _require(layer_index < len(layers), f"{label} is missing GDN layer {layer_index}")
        layer = layers[layer_index]
        for state_family in STATE_FAMILIES:
            attribute = _FAMILY_ATTRIBUTES[state_family]
            mapping = getattr(layer, attribute, None)
            _require(
                isinstance(mapping, dict),
                f"{label}.layers[{layer_index}].{attribute} must be a dict",
            )
            _require(
                state_index in mapping,
                f"{label}.layers[{layer_index}].{attribute} is missing state {state_index}",
            )
            tensor = mapping[state_index]
            _require(
                isinstance(tensor, torch.Tensor),
                f"{label} {_coordinate(layer_index, state_family, state_index)} is not a tensor",
            )
            _require(
                tensor.is_floating_point(),
                f"{label} {_coordinate(layer_index, state_family, state_index)} is not floating point",
            )
            _require(
                tensor.numel() > 0,
                f"{label} {_coordinate(layer_index, state_family, state_index)} is empty",
            )
            dtype = str(tensor.dtype)
            _require(dtype in _FLOAT_DTYPE_NBYTES, f"unsupported GDN dtype {dtype}")
            result.append(
                _LiveTensor(
                    layer_index=layer_index,
                    state_family=state_family,
                    state_index=state_index,
                    tensor=tensor,
                )
            )
    _require(
        len(result) == EXPECTED_TENSORS_PER_OWNER,
        f"{label} must expose exactly {EXPECTED_TENSORS_PER_OWNER} GDN tensors",
    )
    return tuple(result)


def _live_descriptor(value: _LiveTensor) -> dict[str, Any]:
    tensor = value.tensor
    shape = [int(item) for item in tensor.shape]
    stride = [int(item) for item in tensor.stride()]
    dtype = str(tensor.dtype)
    element_size = _FLOAT_DTYPE_NBYTES[dtype]
    storage_offset = int(tensor.storage_offset())
    byte_start, byte_end = _byte_interval(
        shape=shape,
        stride=stride,
        storage_offset=storage_offset,
        element_size=element_size,
    )
    storage_nbytes = int(tensor.untyped_storage().nbytes())
    _require(
        0 <= byte_start < byte_end <= storage_nbytes,
        f"{value.coordinate} byte interval is outside its storage",
    )
    return {
        "layer_index": value.layer_index,
        "state_family": value.state_family,
        "state_index": value.state_index,
        "shape": shape,
        "stride": stride,
        "storage_offset": storage_offset,
        "dtype": dtype,
        "device": str(tensor.device),
        "storage_nbytes": storage_nbytes,
        "tensor_nbytes": int(tensor.numel()) * element_size,
        "byte_start": byte_start,
        "byte_end_exclusive": byte_end,
        "content_sha256": _tensor_digest(tensor),
    }


def _content_manifest_from_descriptors(descriptors: Sequence[dict[str, Any]]) -> str:
    fields = (
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
    return _sha256_json([{field: row[field] for field in fields} for row in descriptors])


def _binding_manifest(
    values: Sequence[_LiveTensor], secret: bytes
) -> str:
    digest = hmac.new(secret, digestmod=hashlib.sha256)
    for value in values:
        device, pointer, storage_nbytes = _storage_key(value.tensor)
        # Object and storage addresses never leave this HMAC.  The random key
        # remains only in the in-memory guard, making the serialized digest
        # non-invertible and safe to compare across phases of one experiment.
        record = (
            value.coordinate,
            id(value.tensor),
            device,
            pointer,
            storage_nbytes,
        )
        digest.update(_canonical_json(record))
        digest.update(b"\n")
    return digest.hexdigest()


def _binding_token(value: _LiveTensor, secret: bytes) -> str:
    return _binding_token_from_parts(
        value.coordinate,
        id(value.tensor),
        _storage_key(value.tensor),
        secret,
    )


def _binding_token_from_parts(
    coordinate: str,
    object_id: int,
    storage_key: tuple[str, int, int],
    secret: bytes,
) -> str:
    device, pointer, storage_nbytes = storage_key
    return hmac.new(
        secret,
        _canonical_json(
            (
                coordinate,
                object_id,
                device,
                pointer,
                storage_nbytes,
            )
        ),
        hashlib.sha256,
    ).hexdigest()


def _storage_binding_token(value: _LiveTensor, secret: bytes) -> str:
    return _storage_binding_token_from_parts(
        value.coordinate,
        _storage_key(value.tensor),
        secret,
    )


def _storage_binding_token_from_parts(
    coordinate: str,
    storage_key: tuple[str, int, int],
    secret: bytes,
) -> str:
    device, pointer, storage_nbytes = storage_key
    return hmac.new(
        secret,
        _canonical_json((coordinate, device, pointer, storage_nbytes)),
        hashlib.sha256,
    ).hexdigest()


@dataclass(frozen=True)
class PersistentGDNGuard:
    """In-memory binding/content baseline; sensitive addresses are never exposed."""

    guard_id: str
    layer_indices: tuple[int, ...]
    state_index: int
    baseline_binding_sha256: str
    baseline_content_sha256: str
    _secret: bytes = field(repr=False, compare=False)
    _entries: tuple[_LiveTensor, ...] = field(repr=False, compare=False)
    _descriptors: tuple[dict[str, Any], ...] = field(repr=False, compare=False)


@dataclass(frozen=True)
class _FrozenRequestBinding:
    layer_index: int
    state_family: str
    state_index: int
    object_id: int
    storage_key: tuple[str, int, int] = field(repr=False, compare=False)
    descriptor: dict[str, Any] = field(repr=False, compare=False)
    binding_token: str
    storage_token: str

    @property
    def coordinate(self) -> str:
        return _coordinate(self.layer_index, self.state_family, self.state_index)


@dataclass(frozen=True)
class RequestGDNBindingGuard:
    """Frozen setup metadata used to prove functional rebind timing.

    The guard is used only in the separately rebuilt ownership-witness and
    mutant cells, never in a primary allocator endpoint.  It deliberately
    pins the setup tensors for that diagnostic cell so a freed setup address
    cannot be recycled into a later legal output and create an ABA false
    positive in the out-of-place rebind check.
    """

    guard_id: str
    policy: str
    layer_indices: tuple[int, ...]
    state_index: int
    resident_count: int
    _secret: bytes = field(repr=False, compare=False)
    _baseline_by_request: tuple[tuple[_FrozenRequestBinding, ...], ...] = field(
        repr=False, compare=False
    )
    _baseline_tensors_by_request: tuple[tuple[Any, ...], ...] = field(
        repr=False, compare=False
    )


def capture_request_gdn_binding_guard(
    requests: Sequence[Any],
    layer_indices: Sequence[int],
    *,
    policy: str,
    state_index: int = 0,
) -> RequestGDNBindingGuard:
    """Freeze request setup bindings without serializing pointers or objects."""

    indices = _validate_layer_indices(layer_indices)
    canonical_policy = _normalize_policy(policy)
    _require(isinstance(requests, (list, tuple)), "requests must be a sequence")
    _require(len(requests) >= 1, "at least one resident request is required")
    _require(_is_int(state_index) and state_index >= 0, "state_index must be non-negative")
    live_entries = tuple(
        _collect_owner_tensors(
            request,
            indices,
            state_index,
            f"request[{request_index}]",
        )
        for request_index, request in enumerate(requests)
    )
    secret = secrets.token_bytes(32)
    baselines: list[tuple[_FrozenRequestBinding, ...]] = []
    for request_entries in live_entries:
        request_baselines: list[_FrozenRequestBinding] = []
        for entry in request_entries:
            storage_key = _storage_key(entry.tensor)
            request_baselines.append(
                _FrozenRequestBinding(
                    layer_index=entry.layer_index,
                    state_family=entry.state_family,
                    state_index=entry.state_index,
                    object_id=id(entry.tensor),
                    storage_key=storage_key,
                    descriptor=_live_descriptor(entry),
                    binding_token=_binding_token_from_parts(
                        entry.coordinate,
                        id(entry.tensor),
                        storage_key,
                        secret,
                    ),
                    storage_token=_storage_binding_token_from_parts(
                        entry.coordinate,
                        storage_key,
                        secret,
                    ),
                )
            )
        baselines.append(tuple(request_baselines))
    return RequestGDNBindingGuard(
        guard_id=secrets.token_hex(16),
        policy=canonical_policy,
        layer_indices=indices,
        state_index=state_index,
        resident_count=len(requests),
        _secret=secret,
        _baseline_by_request=tuple(baselines),
        _baseline_tensors_by_request=tuple(
            tuple(entry.tensor for entry in request_entries)
            for request_entries in live_entries
        ),
    )


def verify_request_gdn_binding_guard(
    guard: RequestGDNBindingGuard,
    requests: Sequence[Any],
    *,
    completed_request_indices: Sequence[int],
    capture_id: str | None = None,
) -> dict[str, Any]:
    """Prove completed requests rebound and incomplete requests did not."""

    _require(isinstance(guard, RequestGDNBindingGuard), "request binding guard is required")
    _require(isinstance(requests, (list, tuple)), "requests must be a sequence")
    _require(len(requests) == guard.resident_count, "request binding guard resident count drift")
    completed = tuple(completed_request_indices)
    _require(
        all(_is_int(index) and 0 <= index < guard.resident_count for index in completed),
        "completed request index is outside the resident group",
    )
    _require(len(set(completed)) == len(completed), "completed request indices must be unique")
    _require(tuple(sorted(completed)) == completed, "completed request indices must be sorted")
    completed_set = set(completed)
    if capture_id is not None:
        _require(
            isinstance(capture_id, str)
            and _GUARD_ID_RE.fullmatch(capture_id) is not None,
            "request binding capture_id drift",
        )
    rows: list[dict[str, Any]] = []
    for request_index, (baseline_entries, request) in enumerate(
        zip(guard._baseline_by_request, requests)
    ):
        observed_entries = _collect_owner_tensors(
            request,
            guard.layer_indices,
            guard.state_index,
            f"request[{request_index}]",
        )
        expected_relation = "rebound" if request_index in completed_set else "unchanged"
        for baseline, observed in zip(baseline_entries, observed_entries):
            _require(baseline.coordinate == observed.coordinate, "request coordinate order drift")
            observed_storage_key = _storage_key(observed.tensor)
            observed_descriptor = _live_descriptor(observed)
            same_object_binding = (
                baseline.object_id == id(observed.tensor)
                and baseline.storage_key == observed_storage_key
            )
            same_storage = baseline.storage_key == observed_storage_key
            if expected_relation == "rebound" and (
                same_storage or baseline.object_id == id(observed.tensor)
            ):
                raise GDNStorageWitnessError(
                    f"completed request[{request_index}] did not out-of-place rebind {baseline.coordinate}",
                    gate_id=GATE_COMPLETED_BINDING_REBOUND,
                )
            structural_fields = (
                "shape",
                "dtype",
                "device",
                "tensor_nbytes",
            )
            if expected_relation == "rebound" and any(
                baseline.descriptor[field] != observed_descriptor[field]
                for field in structural_fields
            ):
                raise GDNStorageWitnessError(
                    f"completed request[{request_index}] rebound metadata drift at "
                    f"{baseline.coordinate}",
                    gate_id=GATE_COMPLETED_BINDING_REBOUND,
                )
            if expected_relation == "unchanged" and not same_object_binding:
                raise GDNStorageWitnessError(
                    f"incomplete request[{request_index}] changed binding at {baseline.coordinate}",
                    gate_id=GATE_INCOMPLETE_BINDING_UNCHANGED,
                )
            if expected_relation == "unchanged" and baseline.descriptor != observed_descriptor:
                raise GDNStorageWitnessError(
                    f"incomplete request[{request_index}] metadata or content changed at "
                    f"{baseline.coordinate}",
                    gate_id=GATE_INCOMPLETE_BINDING_UNCHANGED,
                )
            rows.append(
                {
                    "request_index": request_index,
                    "layer_index": baseline.layer_index,
                    "state_family": baseline.state_family,
                    "state_index": baseline.state_index,
                    "expected_relation": expected_relation,
                    "baseline_binding_token": baseline.binding_token,
                    "observed_binding_token": _binding_token(observed, guard._secret),
                    "baseline_storage_token": baseline.storage_token,
                    "observed_storage_token": _storage_binding_token(
                        observed, guard._secret
                    ),
                }
            )
    record = {
        "guard_id": guard.guard_id,
        "capture_id": capture_id,
        "policy": guard.policy,
        "layer_indices": list(guard.layer_indices),
        "state_index": guard.state_index,
        "resident_count": guard.resident_count,
        "completed_request_indices": list(completed),
        "expected_tensor_count_per_owner": EXPECTED_TENSORS_PER_OWNER,
        "rows": rows,
    }
    record["rows_sha256"] = _sha256_json(rows)
    replay_request_gdn_binding_witness(record)
    return record


def replay_request_gdn_binding_witness(record: Any) -> dict[str, Any]:
    """Replay pointer-free equality/inequality relations from one raw record."""

    _require(isinstance(record, dict), "request binding witness must be an object")
    required = {
        "guard_id",
        "policy",
        "layer_indices",
        "state_index",
        "resident_count",
        "completed_request_indices",
        "expected_tensor_count_per_owner",
        "rows",
        "rows_sha256",
    }
    _require(not (required - set(record)), "request binding witness schema fields missing")
    _require(
        isinstance(record["guard_id"], str)
        and _GUARD_ID_RE.fullmatch(record["guard_id"]) is not None,
        "request binding guard_id drift",
    )
    capture_id = record.get("capture_id")
    if capture_id is not None:
        _require(
            isinstance(capture_id, str)
            and _GUARD_ID_RE.fullmatch(capture_id) is not None,
            "request binding capture_id drift",
        )
    policy = _normalize_policy(record["policy"])
    _require(policy == record["policy"], "request binding policy must be canonical")
    layer_indices = _validate_layer_indices(record["layer_indices"])
    state_index = record["state_index"]
    _require(_is_int(state_index) and state_index >= 0, "request binding state_index drift")
    resident_count = record["resident_count"]
    _require(_is_int(resident_count) and resident_count >= 1, "request binding resident_count drift")
    completed = record["completed_request_indices"]
    _require(isinstance(completed, list), "request binding completed indices must be a list")
    _require(
        all(_is_int(index) and 0 <= index < resident_count for index in completed),
        "request binding completed index drift",
    )
    _require(completed == sorted(set(completed)), "request binding completed indices drift")
    _require(
        record["expected_tensor_count_per_owner"] == EXPECTED_TENSORS_PER_OWNER,
        "request binding tensor count drift",
    )
    rows = record["rows"]
    _require(isinstance(rows, list), "request binding rows must be a list")
    _require(
        len(rows) == resident_count * EXPECTED_TENSORS_PER_OWNER,
        "request binding row cardinality drift",
    )
    _require_sha256(record["rows_sha256"], "request binding rows_sha256")
    _require(_sha256_json(rows) == record["rows_sha256"], "request binding row digest drift")
    completed_set = set(completed)
    expected_order = [
        (request_index, layer_index, family, state_index)
        for request_index in range(resident_count)
        for layer_index in layer_indices
        for family in STATE_FAMILIES
    ]
    observed_order: list[tuple[int, int, str, int]] = []
    for row in rows:
        _require(isinstance(row, dict), "request binding row must be an object")
        row_required = {
            "request_index",
            "layer_index",
            "state_family",
            "state_index",
            "expected_relation",
            "baseline_binding_token",
            "observed_binding_token",
            "baseline_storage_token",
            "observed_storage_token",
        }
        _require(not (row_required - set(row)), "request binding row schema fields missing")
        request_index = row["request_index"]
        expected_relation = "rebound" if request_index in completed_set else "unchanged"
        _require(row["expected_relation"] == expected_relation, "request binding relation drift")
        baseline_token = _require_sha256(
            row["baseline_binding_token"], "baseline binding token"
        )
        observed_token = _require_sha256(
            row["observed_binding_token"], "observed binding token"
        )
        baseline_storage_token = _require_sha256(
            row["baseline_storage_token"], "baseline storage token"
        )
        observed_storage_token = _require_sha256(
            row["observed_storage_token"], "observed storage token"
        )
        if expected_relation == "rebound":
            _require(
                baseline_storage_token != observed_storage_token,
                "completed request storage token did not change",
                gate_id=GATE_COMPLETED_BINDING_REBOUND,
            )
        else:
            _require(
                baseline_token == observed_token,
                "incomplete request binding token changed",
                gate_id=GATE_INCOMPLETE_BINDING_UNCHANGED,
            )
            _require(
                baseline_storage_token == observed_storage_token,
                "incomplete request storage token changed",
                gate_id=GATE_INCOMPLETE_BINDING_UNCHANGED,
            )
        observed_order.append(
            (
                request_index,
                row["layer_index"],
                row["state_family"],
                row["state_index"],
            )
        )
    _require(observed_order == expected_order, "request binding row coordinate order drift")
    return {
        "passed": True,
        "guard_id": record["guard_id"],
        "capture_id": capture_id,
        "policy": policy,
        "resident_count": resident_count,
        "completed_request_indices": completed,
        "rebound_tensor_count": len(completed) * EXPECTED_TENSORS_PER_OWNER,
        "unchanged_tensor_count": (resident_count - len(completed))
        * EXPECTED_TENSORS_PER_OWNER,
        "rows_sha256": record["rows_sha256"],
    }


def capture_persistent_gdn_guard(
    persistent: Any,
    layer_indices: Sequence[int],
    *,
    state_index: int = 0,
) -> PersistentGDNGuard:
    """Freeze live persistent bindings and values before resident construction."""

    indices = _validate_layer_indices(layer_indices)
    _require(_is_int(state_index) and state_index >= 0, "state_index must be non-negative")
    entries = _collect_owner_tensors(persistent, indices, state_index, "persistent")
    descriptors = tuple(_live_descriptor(entry) for entry in entries)
    secret = secrets.token_bytes(32)
    return PersistentGDNGuard(
        guard_id=secrets.token_hex(16),
        layer_indices=indices,
        state_index=state_index,
        baseline_binding_sha256=_binding_manifest(entries, secret),
        baseline_content_sha256=_content_manifest_from_descriptors(descriptors),
        _secret=secret,
        _entries=entries,
        _descriptors=descriptors,
    )


def verify_persistent_gdn_guard(
    guard: PersistentGDNGuard,
    persistent: Any,
) -> dict[str, str]:
    """Verify the live persistent cache and return a pointer-free guard record."""

    _require(isinstance(guard, PersistentGDNGuard), "persistent_guard is required")
    current = _collect_owner_tensors(
        persistent,
        guard.layer_indices,
        guard.state_index,
        "persistent",
    )
    current_descriptors = tuple(_live_descriptor(entry) for entry in current)
    for baseline, observed, baseline_descriptor, observed_descriptor in zip(
        guard._entries,
        current,
        guard._descriptors,
        current_descriptors,
    ):
        _require(
            baseline.coordinate == observed.coordinate,
            "persistent coordinate order drift",
        )
        if baseline.tensor is not observed.tensor or _storage_key(baseline.tensor) != _storage_key(
            observed.tensor
        ):
            raise GDNStorageWitnessError(
                f"persistent binding drift at {baseline.coordinate}",
                gate_id=GATE_PERSISTENT_IMMUTABLE,
            )
        metadata_fields = (
            "shape",
            "stride",
            "storage_offset",
            "dtype",
            "device",
            "storage_nbytes",
            "tensor_nbytes",
            "byte_start",
            "byte_end_exclusive",
        )
        if any(
            baseline_descriptor[field] != observed_descriptor[field]
            for field in metadata_fields
        ):
            raise GDNStorageWitnessError(
                f"persistent binding metadata drift at {baseline.coordinate}",
                gate_id=GATE_PERSISTENT_IMMUTABLE,
            )
        if baseline_descriptor["content_sha256"] != observed_descriptor["content_sha256"]:
            raise GDNStorageWitnessError(
                f"persistent digest drift at {baseline.coordinate}",
                gate_id=GATE_PERSISTENT_IMMUTABLE,
            )

    observed_binding = _binding_manifest(current, guard._secret)
    observed_content = _content_manifest_from_descriptors(current_descriptors)
    _require(
        observed_binding == guard.baseline_binding_sha256,
        "persistent binding manifest drift",
        gate_id=GATE_PERSISTENT_IMMUTABLE,
    )
    _require(
        observed_content == guard.baseline_content_sha256,
        "persistent content manifest drift",
        gate_id=GATE_PERSISTENT_IMMUTABLE,
    )
    return {
        "guard_id": guard.guard_id,
        "baseline_binding_sha256": guard.baseline_binding_sha256,
        "observed_binding_sha256": observed_binding,
        "baseline_content_sha256": guard.baseline_content_sha256,
        "observed_content_sha256": observed_content,
    }


def _normalize_completed_indices(
    phase: str,
    resident_count: int,
    completed_request_indices: Sequence[int] | None,
) -> tuple[int, ...]:
    if phase == PHASE_SETUP_BORROWED_IMMUTABLE:
        _require(
            completed_request_indices is None or len(completed_request_indices) == 0,
            "setup phase cannot contain completed requests",
        )
        return ()
    if completed_request_indices is None:
        result = tuple(range(resident_count))
    else:
        _require(
            isinstance(completed_request_indices, (list, tuple)),
            "completed_request_indices must be a list or tuple",
        )
        result = tuple(completed_request_indices)
    _require(bool(result), "post phase requires at least one completed request")
    _require(
        all(_is_int(index) and 0 <= index < resident_count for index in result),
        "completed request index is outside the resident group",
    )
    _require(len(set(result)) == len(result), "completed request indices must be unique")
    _require(tuple(sorted(result)) == result, "completed request indices must be sorted")
    if phase == PHASE_POST_GENERATION:
        _require(
            result == tuple(range(resident_count)),
            "post_generation requires every resident request to be completed",
        )
    return result


def capture_gdn_storage_snapshot(
    persistent: Any,
    requests: Sequence[Any],
    layer_indices: Sequence[int],
    *,
    phase: str,
    policy: str,
    persistent_guard: PersistentGDNGuard,
    completed_request_indices: Sequence[int] | None = None,
    state_index: int = 0,
    capture_id: str | None = None,
    request_guard_id: str | None = None,
) -> dict[str, Any]:
    """Capture JSON-compatible rows without making a policy verdict.

    Call :func:`replay_gdn_storage_witness` on the returned object, preferably
    after a JSON round trip, to enforce the phase-specific ownership rules.
    """

    indices = _validate_layer_indices(layer_indices)
    canonical_phase = _validate_phase(phase)
    canonical_policy = _normalize_policy(policy)
    _require(isinstance(requests, (list, tuple)), "requests must be a sequence")
    resident_count = len(requests)
    _require(resident_count >= 1, "at least one resident request is required")
    _require(_is_int(state_index) and state_index >= 0, "state_index must be non-negative")
    _require(isinstance(persistent_guard, PersistentGDNGuard), "persistent_guard is required")
    _require(persistent_guard.layer_indices == indices, "persistent guard layer plan drift")
    _require(persistent_guard.state_index == state_index, "persistent guard state index drift")
    completed = _normalize_completed_indices(
        canonical_phase,
        resident_count,
        completed_request_indices,
    )
    if capture_id is not None:
        _require(
            isinstance(capture_id, str)
            and _GUARD_ID_RE.fullmatch(capture_id) is not None,
            "storage witness capture_id drift",
        )
    if request_guard_id is not None:
        _require(
            isinstance(request_guard_id, str)
            and _GUARD_ID_RE.fullmatch(request_guard_id) is not None,
            "storage witness request_guard_id drift",
        )

    # Bracket capture so a mutation concurrent with row materialization fails.
    verify_persistent_gdn_guard(persistent_guard, persistent)
    owner_entries: list[tuple[str, int | None, tuple[_LiveTensor, ...]]] = [
        (
            "persistent",
            None,
            _collect_owner_tensors(persistent, indices, state_index, "persistent"),
        )
    ]
    for request_index, request in enumerate(requests):
        owner_entries.append(
            (
                "request",
                request_index,
                _collect_owner_tensors(
                    request,
                    indices,
                    state_index,
                    f"request[{request_index}]",
                ),
            )
        )

    normalized_storages: dict[tuple[str, int, int], str] = {}
    rows: list[dict[str, Any]] = []
    for owner_kind, request_index, entries in owner_entries:
        for entry in entries:
            descriptor = _live_descriptor(entry)
            key = _storage_key(entry.tensor)
            if key not in normalized_storages:
                normalized_storages[key] = f"storage-{len(normalized_storages):04d}"
            rows.append(
                {
                    "owner_kind": owner_kind,
                    "request_index": request_index,
                    **descriptor,
                    "storage_id": normalized_storages[key],
                }
            )
    guard_record = verify_persistent_gdn_guard(persistent_guard, persistent)
    persistent_rows = rows[:EXPECTED_TENSORS_PER_OWNER]
    derived_content = _content_manifest_from_descriptors(persistent_rows)
    _require(
        derived_content == guard_record["observed_content_sha256"],
        "persistent rows changed while the snapshot was captured",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_id": capture_id,
        "request_guard_id": request_guard_id,
        "phase": canonical_phase,
        "policy": canonical_policy,
        "layer_indices": list(indices),
        "state_index": state_index,
        "resident_count": resident_count,
        "completed_request_indices": list(completed),
        "expected_tensor_count_per_owner": EXPECTED_TENSORS_PER_OWNER,
        "persistent_guard": guard_record,
        "rows": rows,
        "rows_sha256": _sha256_json(rows),
    }


def _reject_address_fields(value: Any, path: str = "snapshot") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            _require(
                not (
                    "data_ptr" in lowered
                    or "storage_ptr" in lowered
                    or lowered in {"pointer", "address", "absolute_address"}
                ),
                f"absolute pointer field is forbidden at {path}.{key}",
            )
            _reject_address_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_address_fields(item, f"{path}[{index}]")


def _require_sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, f"{label} is not SHA256")
    return value


def _validate_serialized_row(row: Any, row_index: int) -> dict[str, Any]:
    _require(isinstance(row, dict), f"row {row_index} must be an object")
    required = {
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
        "storage_id",
    }
    missing = sorted(required - set(row))
    _require(not missing, f"row {row_index} is missing schema fields: {missing}")
    _require(row["owner_kind"] in ("persistent", "request"), f"row {row_index} owner_kind drift")
    if row["owner_kind"] == "persistent":
        _require(row["request_index"] is None, f"row {row_index} persistent request_index must be null")
    else:
        _require(_is_int(row["request_index"]), f"row {row_index} request_index must be an integer")
    _require(_is_int(row["layer_index"]) and row["layer_index"] >= 0, f"row {row_index} layer_index drift")
    _require(row["state_family"] in STATE_FAMILIES, f"row {row_index} state_family drift")
    _require(_is_int(row["state_index"]) and row["state_index"] >= 0, f"row {row_index} state_index drift")
    _require(isinstance(row["shape"], list), f"row {row_index} shape must be a list")
    _require(isinstance(row["stride"], list), f"row {row_index} stride must be a list")
    dtype = row["dtype"]
    _require(dtype in _FLOAT_DTYPE_NBYTES, f"row {row_index} has unsupported dtype")
    _require(isinstance(row["device"], str) and bool(row["device"]), f"row {row_index} device drift")
    for field_name in (
        "storage_offset",
        "storage_nbytes",
        "tensor_nbytes",
        "byte_start",
        "byte_end_exclusive",
    ):
        _require(_is_int(row[field_name]), f"row {row_index} {field_name} must be an integer")
    _require(row["storage_nbytes"] > 0, f"row {row_index} storage_nbytes must be positive")
    expected_start, expected_end = _byte_interval(
        shape=row["shape"],
        stride=row["stride"],
        storage_offset=row["storage_offset"],
        element_size=_FLOAT_DTYPE_NBYTES[dtype],
    )
    _require(row["byte_start"] == expected_start, f"row {row_index} byte_start drift")
    _require(row["byte_end_exclusive"] == expected_end, f"row {row_index} byte_end drift")
    _require(
        0 <= expected_start < expected_end <= row["storage_nbytes"],
        f"row {row_index} interval exceeds storage",
    )
    expected_tensor_nbytes = prod(row["shape"]) * _FLOAT_DTYPE_NBYTES[dtype]
    _require(row["tensor_nbytes"] == expected_tensor_nbytes, f"row {row_index} tensor_nbytes drift")
    _require_sha256(row["content_sha256"], f"row {row_index} content digest")
    _require(
        isinstance(row["storage_id"], str)
        and _STORAGE_ID_RE.fullmatch(row["storage_id"]) is not None,
        f"row {row_index} storage_id is not normalized",
    )
    return row


def _rows_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return (
        left["storage_id"] == right["storage_id"]
        and left["byte_start"] < right["byte_end_exclusive"]
        and right["byte_start"] < left["byte_end_exclusive"]
    )


def _rows_exact_alias(left: dict[str, Any], right: dict[str, Any]) -> bool:
    fields = (
        "storage_id",
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
    return all(left[field] == right[field] for field in fields)


def _coordinate_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return row["layer_index"], row["state_family"], row["state_index"]


def _assert_pairwise_internal_disjoint(rows: Sequence[dict[str, Any]], label: str) -> int:
    comparisons = 0
    for left, right in combinations(rows, 2):
        comparisons += 1
        if _rows_overlap(left, right):
            raise GDNStorageWitnessError(
                f"{label} tensors overlap: {_coordinate_key(left)} and {_coordinate_key(right)}",
                gate_id=GATE_OWNER_INTERNAL_DISJOINT,
            )
    return comparisons


def _assert_sets_disjoint(
    left_rows: Sequence[dict[str, Any]],
    right_rows: Sequence[dict[str, Any]],
    label: str,
    *,
    gate_id: str,
) -> int:
    comparisons = 0
    for left in left_rows:
        for right in right_rows:
            comparisons += 1
            if _rows_overlap(left, right):
                relation = "exact alias" if _rows_exact_alias(left, right) else "partial overlap"
                raise GDNStorageWitnessError(
                    f"{label} has {relation}: {_coordinate_key(left)} vs {_coordinate_key(right)}",
                    gate_id=gate_id,
                )
    return comparisons


def replay_gdn_storage_witness(snapshot: Any) -> dict[str, Any]:
    """Recompute a phase verdict solely from serialized witness rows.

    Top-level or nested ``passed`` fields are intentionally ignored.  All
    overlap decisions use ``storage_id`` plus the recomputed byte intervals.
    """

    _require(isinstance(snapshot, dict), "storage witness must be an object")
    _reject_address_fields(snapshot)
    required_top = {
        "schema_version",
        "phase",
        "policy",
        "layer_indices",
        "state_index",
        "resident_count",
        "completed_request_indices",
        "expected_tensor_count_per_owner",
        "persistent_guard",
        "rows",
        "rows_sha256",
    }
    missing_top = sorted(required_top - set(snapshot))
    _require(not missing_top, f"storage witness is missing schema fields: {missing_top}")
    _require(snapshot["schema_version"] == SCHEMA_VERSION, "storage witness schema version drift")
    capture_id = snapshot.get("capture_id")
    request_guard_id = snapshot.get("request_guard_id")
    for value, label in (
        (capture_id, "storage witness capture_id"),
        (request_guard_id, "storage witness request_guard_id"),
    ):
        if value is not None:
            _require(
                isinstance(value, str) and _GUARD_ID_RE.fullmatch(value) is not None,
                f"{label} drift",
            )
    phase = _validate_phase(snapshot["phase"])
    policy = _normalize_policy(snapshot["policy"])
    _require(policy == snapshot["policy"], "serialized policy must use its canonical name")
    indices = _validate_layer_indices(snapshot["layer_indices"])
    state_index = snapshot["state_index"]
    _require(_is_int(state_index) and state_index >= 0, "serialized state_index drift")
    resident_count = snapshot["resident_count"]
    _require(_is_int(resident_count) and resident_count >= 1, "serialized resident_count drift")
    completed_raw = snapshot["completed_request_indices"]
    _require(isinstance(completed_raw, list), "completed_request_indices must be serialized as a list")
    completed = _normalize_completed_indices(phase, resident_count, completed_raw)
    _require(
        snapshot["expected_tensor_count_per_owner"] == EXPECTED_TENSORS_PER_OWNER,
        "expected tensor count drift",
    )

    guard = snapshot["persistent_guard"]
    _require(isinstance(guard, dict), "persistent_guard record is missing")
    guard_fields = (
        "baseline_binding_sha256",
        "observed_binding_sha256",
        "baseline_content_sha256",
        "observed_content_sha256",
    )
    _require(
        isinstance(guard.get("guard_id"), str)
        and _GUARD_ID_RE.fullmatch(guard["guard_id"]) is not None,
        "persistent guard_id drift",
    )
    for field_name in guard_fields:
        _require_sha256(guard.get(field_name), f"persistent guard {field_name}")
    _require(
        guard["baseline_binding_sha256"] == guard["observed_binding_sha256"],
        "persistent binding drift in serialized witness",
        gate_id=GATE_PERSISTENT_IMMUTABLE,
    )
    _require(
        guard["baseline_content_sha256"] == guard["observed_content_sha256"],
        "persistent digest drift in serialized witness",
        gate_id=GATE_PERSISTENT_IMMUTABLE,
    )

    rows_raw = snapshot["rows"]
    _require(isinstance(rows_raw, list), "storage witness rows must be a list")
    expected_row_count = (resident_count + 1) * EXPECTED_TENSORS_PER_OWNER
    _require(len(rows_raw) == expected_row_count, f"storage witness must contain {expected_row_count} rows")
    _require_sha256(snapshot["rows_sha256"], "rows_sha256")
    _require(_sha256_json(rows_raw) == snapshot["rows_sha256"], "serialized row digest drift")
    rows = [_validate_serialized_row(row, index) for index, row in enumerate(rows_raw)]

    expected_order: list[tuple[str, int | None, int, str, int]] = []
    for owner_kind, request_index in [
        ("persistent", None),
        *(("request", index) for index in range(resident_count)),
    ]:
        for layer_index in indices:
            for state_family in STATE_FAMILIES:
                expected_order.append(
                    (owner_kind, request_index, layer_index, state_family, state_index)
                )
    observed_order = [
        (
            row["owner_kind"],
            row["request_index"],
            row["layer_index"],
            row["state_family"],
            row["state_index"],
        )
        for row in rows
    ]
    _require(observed_order == expected_order, "storage witness row owner/coordinate order drift")

    seen_storage_ids: list[str] = []
    storage_metadata: dict[str, tuple[str, int]] = {}
    for row_index, row in enumerate(rows):
        storage_id = row["storage_id"]
        if storage_id not in storage_metadata:
            expected_id = f"storage-{len(seen_storage_ids):04d}"
            _require(storage_id == expected_id, f"row {row_index} storage IDs are not snapshot-local normalized order")
            seen_storage_ids.append(storage_id)
            storage_metadata[storage_id] = (row["device"], row["storage_nbytes"])
        _require(
            storage_metadata[storage_id] == (row["device"], row["storage_nbytes"]),
            f"row {row_index} storage metadata conflicts with its normalized ID",
        )

    persistent_rows = rows[:EXPECTED_TENSORS_PER_OWNER]
    request_rows = {
        request_index: rows[
            (request_index + 1) * EXPECTED_TENSORS_PER_OWNER :
            (request_index + 2) * EXPECTED_TENSORS_PER_OWNER
        ]
        for request_index in range(resident_count)
    }
    derived_persistent_content = _content_manifest_from_descriptors(persistent_rows)
    _require(
        derived_persistent_content == guard["observed_content_sha256"],
        "persistent row content does not match its guard",
    )

    internal_comparisons = _assert_pairwise_internal_disjoint(
        persistent_rows, "persistent base"
    )
    for request_index, owner_rows in request_rows.items():
        internal_comparisons += _assert_pairwise_internal_disjoint(
            owner_rows, f"request[{request_index}]"
        )

    exact_alias_comparisons = 0
    disjoint_comparisons = 0
    if phase == PHASE_SETUP_BORROWED_IMMUTABLE and policy == POLICY_SHARED_BASE:
        persistent_by_coordinate = {
            _coordinate_key(row): row for row in persistent_rows
        }
        for request_index, owner_rows in request_rows.items():
            for row in owner_rows:
                exact_alias_comparisons += 1
                base = persistent_by_coordinate[_coordinate_key(row)]
                if not _rows_exact_alias(row, base):
                    relation = "partial overlap" if _rows_overlap(row, base) else "disjoint storage"
                    raise GDNStorageWitnessError(
                        f"request[{request_index}] must exact alias persistent at "
                        f"{_coordinate_key(row)}; observed {relation}",
                        gate_id=GATE_SHARED_SETUP_EXACT_BASE_ALIAS,
                    )
    elif phase == PHASE_SETUP_BORROWED_IMMUTABLE:
        for request_index, owner_rows in request_rows.items():
            disjoint_comparisons += _assert_sets_disjoint(
                owner_rows,
                persistent_rows,
                f"materialized request[{request_index}]/persistent",
                gate_id=GATE_MATERIALIZED_SETUP_BASE_DISJOINT,
            )
        for left_index, right_index in combinations(range(resident_count), 2):
            disjoint_comparisons += _assert_sets_disjoint(
                request_rows[left_index],
                request_rows[right_index],
                f"materialized request[{left_index}]/request[{right_index}]",
                gate_id=GATE_MATERIALIZED_SETUP_PEERS_DISJOINT,
            )
    elif policy == POLICY_SHARED_BASE:
        # Functional rebind has a temporal ownership contract.  A request is
        # allowed to borrow the immutable persistent tensors only until its
        # first transition.  Completed requests must be private, while every
        # not-yet-completed request must still be an exact coordinate-wise
        # alias of the base; eager cloning before its transition is drift.
        persistent_by_coordinate = {
            _coordinate_key(row): row for row in persistent_rows
        }
        incomplete = sorted(set(range(resident_count)) - set(completed))
        for request_index in incomplete:
            for row in request_rows[request_index]:
                exact_alias_comparisons += 1
                base = persistent_by_coordinate[_coordinate_key(row)]
                if not _rows_exact_alias(row, base):
                    relation = "partial overlap" if _rows_overlap(row, base) else "disjoint storage"
                    raise GDNStorageWitnessError(
                        f"incomplete request[{request_index}] must exact alias persistent at "
                        f"{_coordinate_key(row)}; observed {relation}",
                        gate_id=GATE_SHARED_INCOMPLETE_EXACT_BASE_ALIAS,
                    )
        # Every completed request must be independent of the base and every
        # peer, including a peer that has not transitioned yet.
        for request_index in completed:
            owner_rows = request_rows[request_index]
            disjoint_comparisons += _assert_sets_disjoint(
                owner_rows,
                persistent_rows,
                f"completed request[{request_index}]/persistent",
                gate_id=GATE_COMPLETED_VS_BASE_DISJOINT,
            )
            for peer_index in range(resident_count):
                if peer_index == request_index:
                    continue
                disjoint_comparisons += _assert_sets_disjoint(
                    owner_rows,
                    request_rows[peer_index],
                    f"completed request[{request_index}]/request[{peer_index}]",
                    gate_id=GATE_COMPLETED_VS_PEERS_DISJOINT,
                )
    else:
        # A materialized base is request-owned from setup onward, including
        # requests that have not yet executed their first transition.
        for request_index, owner_rows in request_rows.items():
            disjoint_comparisons += _assert_sets_disjoint(
                owner_rows,
                persistent_rows,
                f"materialized request[{request_index}]/persistent",
                gate_id=GATE_MATERIALIZED_SETUP_BASE_DISJOINT,
            )
        for left_index, right_index in combinations(range(resident_count), 2):
            disjoint_comparisons += _assert_sets_disjoint(
                request_rows[left_index],
                request_rows[right_index],
                f"materialized request[{left_index}]/request[{right_index}]",
                gate_id=GATE_MATERIALIZED_SETUP_PEERS_DISJOINT,
            )

    policy_comparisons = exact_alias_comparisons + disjoint_comparisons
    _require(policy_comparisons > 0, "ownership proof is vacuous")
    evaluated_gate_ids = [GATE_PERSISTENT_IMMUTABLE, GATE_OWNER_INTERNAL_DISJOINT]
    if phase == PHASE_SETUP_BORROWED_IMMUTABLE and policy == POLICY_SHARED_BASE:
        evaluated_gate_ids.append(GATE_SHARED_SETUP_EXACT_BASE_ALIAS)
    elif phase == PHASE_SETUP_BORROWED_IMMUTABLE:
        evaluated_gate_ids.extend(
            (
                GATE_MATERIALIZED_SETUP_BASE_DISJOINT,
            )
        )
        if resident_count > 1:
            evaluated_gate_ids.append(GATE_MATERIALIZED_SETUP_PEERS_DISJOINT)
    elif policy == POLICY_SHARED_BASE:
        evaluated_gate_ids.append(GATE_COMPLETED_VS_BASE_DISJOINT)
        if resident_count > 1:
            evaluated_gate_ids.append(GATE_COMPLETED_VS_PEERS_DISJOINT)
        if len(completed) < resident_count:
            evaluated_gate_ids.append(GATE_SHARED_INCOMPLETE_EXACT_BASE_ALIAS)
    else:
        evaluated_gate_ids.append(GATE_MATERIALIZED_SETUP_BASE_DISJOINT)
        if resident_count > 1:
            evaluated_gate_ids.append(GATE_MATERIALIZED_SETUP_PEERS_DISJOINT)
    return {
        "passed": True,
        "schema_version": SCHEMA_VERSION,
        "phase": phase,
        "capture_id": capture_id,
        "request_guard_id": request_guard_id,
        "policy": policy,
        "resident_count": resident_count,
        "completed_request_indices": list(completed),
        "tensor_count_per_owner": EXPECTED_TENSORS_PER_OWNER,
        "row_count": len(rows),
        "normalized_storage_count": len(seen_storage_ids),
        "persistent_binding_and_digest_immutable": True,
        "exact_alias_comparisons": exact_alias_comparisons,
        "disjoint_comparisons": disjoint_comparisons,
        "internal_disjoint_comparisons": internal_comparisons,
        "ownership_proof_nonvacuous": True,
        "evaluated_gate_ids": evaluated_gate_ids,
        "rows_sha256": snapshot["rows_sha256"],
    }


def capture_gdn_phase_witness(
    persistent: Any,
    requests: Sequence[Any],
    layer_indices: Sequence[int],
    *,
    run_id: str,
    cell_id: str,
    kv_policy: str,
    phase: str,
    policy: str,
    persistent_guard: PersistentGDNGuard,
    request_guard: RequestGDNBindingGuard,
    completed_request_indices: Sequence[int] | None = None,
    state_index: int = 0,
) -> dict[str, Any]:
    """Atomically construct both ownership views for one live lifecycle phase.

    The caller cannot supply capture or request-guard identifiers.  This
    function derives them from the live guard objects, brackets storage capture
    with request-binding verification, and returns the only phase-row shape
    accepted by the formal timeline replay.
    """

    indices = _validate_layer_indices(layer_indices)
    for value, label in ((run_id, "run_id"), (cell_id, "cell_id")):
        _require(isinstance(value, str) and bool(value.strip()), f"phase {label} must be non-empty")
    _require(kv_policy in KV_POLICIES, "phase KV policy drift")
    canonical_phase = _validate_phase(phase)
    canonical_policy = _normalize_policy(policy)
    _require(
        isinstance(persistent_guard, PersistentGDNGuard),
        "persistent guard object is required for unified phase capture",
    )
    _require(
        isinstance(request_guard, RequestGDNBindingGuard),
        "request guard object is required for unified phase capture",
    )
    _require(request_guard.policy == canonical_policy, "request guard policy drift")
    _require(request_guard.layer_indices == indices, "request guard layer plan drift")
    _require(request_guard.state_index == state_index, "request guard state index drift")
    _require(request_guard.resident_count == len(requests), "request guard N drift")
    completed = _normalize_completed_indices(
        canonical_phase,
        len(requests),
        completed_request_indices,
    )
    capture_id = secrets.token_hex(16)
    before = verify_request_gdn_binding_guard(
        request_guard,
        requests,
        completed_request_indices=completed,
        capture_id=capture_id,
    )
    storage = capture_gdn_storage_snapshot(
        persistent,
        requests,
        indices,
        phase=canonical_phase,
        policy=canonical_policy,
        persistent_guard=persistent_guard,
        completed_request_indices=completed,
        state_index=state_index,
        capture_id=capture_id,
        request_guard_id=request_guard.guard_id,
    )
    after = verify_request_gdn_binding_guard(
        request_guard,
        requests,
        completed_request_indices=completed,
        capture_id=capture_id,
    )
    _require(before == after, "request bindings drifted during unified phase capture")
    return {
        "phase": canonical_phase,
        "capture_id": capture_id,
        "capture_protocol": "unified-live-gdn-phase-v1",
        "run_id": run_id,
        "cell_id": cell_id,
        "kv_policy": kv_policy,
        "storage_witness": storage,
        "binding_witness": after,
    }


def replay_gdn_storage_timeline(bundle: Any) -> dict[str, Any]:
    """Bind setup, partial-transition, and generation witnesses into one cell.

    Integrity against an adversarial rewrite comes from the caller's external
    raw-artifact manifest.  This function supplies a fail-closed semantic
    replay over the already frozen bytes; it does not call an internal checksum
    a cryptographic chain of custody.
    """

    _require(isinstance(bundle, dict), "GDN timeline must be an object")
    required = {
        "schema_version",
        "run_id",
        "cell_id",
        "kv_policy",
        "gdn_policy",
        "group_gdn_base_policy",
        "resident_count",
        "layer_indices",
        "state_index",
        "phases",
    }
    _require(not (required - set(bundle)), "GDN timeline schema fields missing")
    _require(
        bundle["schema_version"] == TIMELINE_SCHEMA_VERSION,
        "GDN timeline schema version drift",
    )
    for name in ("run_id", "cell_id"):
        _require(
            isinstance(bundle[name], str) and bool(bundle[name].strip()),
            f"GDN timeline {name} must be non-empty",
        )
    kv_policy = bundle["kv_policy"]
    _require(kv_policy in KV_POLICIES, "GDN timeline KV policy drift")
    gdn_policy = _normalize_policy(bundle["gdn_policy"])
    _require(gdn_policy == bundle["gdn_policy"], "timeline GDN policy must be canonical")
    _require(
        _normalize_policy(bundle["group_gdn_base_policy"]) == gdn_policy,
        "group audit GDN policy does not match timeline policy",
    )
    resident_count = bundle["resident_count"]
    _require(_is_int(resident_count) and resident_count >= 1, "timeline resident_count drift")
    layer_indices = _validate_layer_indices(bundle["layer_indices"])
    state_index = bundle["state_index"]
    _require(_is_int(state_index) and state_index >= 0, "timeline state_index drift")
    phases = bundle["phases"]
    _require(isinstance(phases, list) and len(phases) >= 3, "timeline needs at least three phases")
    phase_names = [row.get("phase") if isinstance(row, dict) else None for row in phases]
    _require(phase_names[0] == PHASE_SETUP_PRE_TRANSITION, "timeline must start at setup")
    _require(phase_names[-1] == PHASE_POST_GENERATION, "timeline must end at generation")
    _require(
        all(name == PHASE_POST_TRANSITION for name in phase_names[1:-1]),
        "timeline middle phases must be post_transition",
    )

    persistent_guard_id: str | None = None
    request_guard_id: str | None = None
    persistent_baseline: tuple[str, str] | None = None
    request_baseline_tokens: tuple[str, ...] | None = None
    previous_completed: tuple[int, ...] = ()
    transition_count = 0
    phase_summaries: list[dict[str, Any]] = []
    seen_capture_ids: set[str] = set()
    for phase_index, phase_row in enumerate(phases):
        _require(isinstance(phase_row, dict), "timeline phase row must be an object")
        _require(
            {
                "phase",
                "capture_id",
                "capture_protocol",
                "run_id",
                "cell_id",
                "kv_policy",
                "storage_witness",
                "binding_witness",
            }
            <= set(phase_row),
            "timeline phase row schema fields missing",
        )
        storage = phase_row["storage_witness"]
        binding = phase_row["binding_witness"]
        storage_replay = replay_gdn_storage_witness(storage)
        binding_replay = replay_request_gdn_binding_witness(binding)
        phase_name = phase_row["phase"]
        capture_id = phase_row["capture_id"]
        _require(
            phase_row["capture_protocol"] == "unified-live-gdn-phase-v1",
            "timeline phase capture protocol drift",
        )
        _require(phase_row["run_id"] == bundle["run_id"], "timeline phase run_id drift")
        _require(phase_row["cell_id"] == bundle["cell_id"], "timeline phase cell_id drift")
        _require(phase_row["kv_policy"] == kv_policy, "timeline phase KV policy drift")
        _require(
            isinstance(capture_id, str)
            and _GUARD_ID_RE.fullmatch(capture_id) is not None,
            "timeline phase capture_id drift",
        )
        _require(capture_id not in seen_capture_ids, "timeline phase capture_id reused")
        seen_capture_ids.add(capture_id)
        _require(storage_replay["phase"] == phase_name, "timeline storage phase drift")
        _require(
            storage_replay["capture_id"] == capture_id
            and binding_replay["capture_id"] == capture_id,
            "timeline storage/binding capture_id disagreement",
        )
        _require(
            storage_replay["request_guard_id"] == binding.get("guard_id"),
            "timeline storage/binding request guard disagreement",
        )
        _require(storage_replay["policy"] == gdn_policy, "timeline storage policy drift")
        _require(binding_replay["policy"] == gdn_policy, "timeline binding policy drift")
        _require(storage.get("layer_indices") == list(layer_indices), "timeline storage layer plan drift")
        _require(binding.get("layer_indices") == list(layer_indices), "timeline binding layer plan drift")
        _require(storage.get("state_index") == state_index, "timeline storage state index drift")
        _require(binding.get("state_index") == state_index, "timeline binding state index drift")
        _require(storage_replay["resident_count"] == resident_count, "timeline storage N drift")
        _require(binding_replay["resident_count"] == resident_count, "timeline binding N drift")
        storage_completed = tuple(storage_replay["completed_request_indices"])
        binding_completed = tuple(binding_replay["completed_request_indices"])
        _require(storage_completed == binding_completed, "timeline completed-set disagreement")
        if phase_index == 0:
            _require(storage_completed == (), "setup timeline phase cannot be completed")
        elif phase_name == PHASE_POST_TRANSITION:
            if transition_count == 0:
                _require(
                    storage_completed == (0,),
                    "first transition witness must complete exactly request 0",
                )
            _require(
                set(previous_completed) < set(storage_completed),
                "post_transition completed set must strictly grow",
            )
            transition_count += 1
        else:
            _require(
                storage_completed == tuple(range(resident_count)),
                "generation timeline phase must cover every request",
            )
            _require(
                set(previous_completed) <= set(storage_completed),
                "generation completed set regressed",
            )
        previous_completed = storage_completed

        guard = storage.get("persistent_guard")
        _require(isinstance(guard, dict), "timeline persistent guard missing")
        current_persistent = (
            guard.get("baseline_binding_sha256"),
            guard.get("baseline_content_sha256"),
        )
        current_request_tokens = tuple(
            row.get("baseline_binding_token") for row in binding.get("rows", [])
        )
        if persistent_guard_id is None:
            persistent_guard_id = guard.get("guard_id")
            request_guard_id = binding.get("guard_id")
            persistent_baseline = current_persistent
            request_baseline_tokens = current_request_tokens
        else:
            _require(guard.get("guard_id") == persistent_guard_id, "persistent guard ID drift across phases")
            _require(binding.get("guard_id") == request_guard_id, "request guard ID drift across phases")
            _require(current_persistent == persistent_baseline, "persistent baseline drift across phases")
            _require(current_request_tokens == request_baseline_tokens, "request setup baseline drift across phases")
        phase_summaries.append(
            {
                "phase": phase_name,
                "capture_id": capture_id,
                "completed_request_indices": list(storage_completed),
                "storage_rows_sha256": storage_replay["rows_sha256"],
                "binding_rows_sha256": binding_replay["rows_sha256"],
            }
        )
    _require(transition_count >= 1, "timeline must contain a partial transition witness")
    return {
        "passed": True,
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "run_id": bundle["run_id"],
        "cell_id": bundle["cell_id"],
        "kv_policy": kv_policy,
        "gdn_policy": gdn_policy,
        "resident_count": resident_count,
        "phase_count": len(phases),
        "transition_count": transition_count,
        "completed_all_requests": previous_completed == tuple(range(resident_count)),
        "persistent_guard_id": persistent_guard_id,
        "request_guard_id": request_guard_id,
        "phase_summaries": phase_summaries,
        "integrity_scope": "self-consistent replay; external raw-artifact SHA required",
    }


__all__ = [
    "EXPECTED_LINEAR_LAYERS",
    "EXPECTED_TENSORS_PER_OWNER",
    "GDNStorageWitnessError",
    "GATE_COMPLETED_VS_BASE_DISJOINT",
    "GATE_COMPLETED_VS_PEERS_DISJOINT",
    "GATE_COMPLETED_BINDING_REBOUND",
    "GATE_INCOMPLETE_BINDING_UNCHANGED",
    "GATE_MATERIALIZED_SETUP_BASE_DISJOINT",
    "GATE_MATERIALIZED_SETUP_PEERS_DISJOINT",
    "GATE_OWNER_INTERNAL_DISJOINT",
    "GATE_PERSISTENT_IMMUTABLE",
    "GATE_SHARED_SETUP_EXACT_BASE_ALIAS",
    "GATE_SHARED_INCOMPLETE_EXACT_BASE_ALIAS",
    "GATE_STORAGE_SCHEMA",
    "PHASE_POST_GENERATION",
    "PHASE_POST_TRANSITION",
    "PHASE_SETUP_BORROWED_IMMUTABLE",
    "PHASE_SETUP_PRE_TRANSITION",
    "POLICY_MATERIALIZED",
    "POLICY_SHARED_BASE",
    "PersistentGDNGuard",
    "RequestGDNBindingGuard",
    "SCHEMA_VERSION",
    "TIMELINE_SCHEMA_VERSION",
    "capture_gdn_storage_snapshot",
    "capture_gdn_phase_witness",
    "capture_persistent_gdn_guard",
    "capture_request_gdn_binding_guard",
    "replay_request_gdn_binding_witness",
    "replay_gdn_storage_timeline",
    "replay_gdn_storage_witness",
    "verify_request_gdn_binding_guard",
    "verify_persistent_gdn_guard",
]
