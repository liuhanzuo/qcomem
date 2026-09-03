from __future__ import annotations

"""Outcome-blind held-out fault definitions for the Round-29 campaign.

This module is owned by the fault author, not by the experiment executor.  It
contains consequence-level mutations derived from upstream serving bug
patterns, but it contains no ForkAudit gate names and no expected detector
outcomes.  The executor must treat the frozen JSON suite and this source file
as immutable inputs.

The mutations are intentionally small and reversible.  Each state mutation
returns a pointer-free receipt that proves ``pre != mutated`` and, after
``restore()``, ``restored == pre``.  H02 is an action-sequence fault rather
than an in-place state mutation and therefore uses disposable fresh cases.
"""

import argparse
import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping


SUITE_SCHEMA = "forkaudit-r29-heldout-fault-suite-v1"
MODULE_BINDING = "r29_heldout_fault_suite"
FAULT_IDS = ("H01", "H02", "H03")
STATE_MUTATION_FAULT_IDS = ("H01", "H03")
ACTION_SEQUENCE_FAULT_IDS = ("H02",)


class HeldOutFaultConfigurationError(RuntimeError):
    """The frozen fault could not be applied to the promised live geometry."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldOutFaultConfigurationError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _flat_int_values(tensor: Any) -> list[int]:
    return [int(value) for value in tensor.detach().reshape(-1).cpu().tolist()]


def _tensor_route_descriptor(
    tensor: Any,
    *,
    request_index: int,
    layer_index: int,
    field: str,
) -> dict[str, Any]:
    values = _flat_int_values(tensor)
    return {
        "schema_version": "forkaudit-r29-pointer-free-route-target-v1",
        "request_index": request_index,
        "layer_index": layer_index,
        "field": field,
        "shape": [int(value) for value in tensor.shape],
        "stride": [int(value) for value in tensor.stride()],
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "values": values,
        "values_sha256": sha256_json(values),
        "contains_absolute_pointer": False,
    }


@dataclass
class MutationHandle:
    """One reversible live mutation and its outcome-blind target binding."""

    fault_id: str
    target_kind: str
    pre_descriptor: Mapping[str, Any]
    mutated_descriptor: Mapping[str, Any]
    capture: Callable[[], Mapping[str, Any]]
    undo: Callable[[], None]
    _restored: bool = False

    def __post_init__(self) -> None:
        _require(self.fault_id in STATE_MUTATION_FAULT_IDS, "unknown state fault")
        _require(
            sha256_json(self.pre_descriptor) != sha256_json(self.mutated_descriptor),
            f"{self.fault_id} mutation is a no-op",
        )

    def applied_receipt(self) -> dict[str, Any]:
        return {
            "schema_version": "forkaudit-r29-heldout-mutation-binding-v1",
            "fault_id": self.fault_id,
            "target_kind": self.target_kind,
            "pre_sha256": sha256_json(self.pre_descriptor),
            "mutated_sha256": sha256_json(self.mutated_descriptor),
            "pre_descriptor": dict(self.pre_descriptor),
            "mutated_descriptor": dict(self.mutated_descriptor),
            "mutation_observed": True,
            "contains_absolute_pointer": False,
        }

    def restore(self) -> dict[str, Any]:
        if not self._restored:
            self.undo()
            self._restored = True
        restored = dict(self.capture())
        pre_sha = sha256_json(self.pre_descriptor)
        restored_sha = sha256_json(restored)
        _require(
            restored_sha == pre_sha,
            f"{self.fault_id} target did not restore byte-for-byte",
        )
        return {
            "schema_version": "forkaudit-r29-heldout-restoration-v1",
            "fault_id": self.fault_id,
            "target_kind": self.target_kind,
            "pre_sha256": pre_sha,
            "restored_sha256": restored_sha,
            "restoration_observed": True,
            "contains_absolute_pointer": False,
        }


def _first_full_layer(plan: Any) -> int:
    indices = tuple(int(value) for value in plan.full_attention_layer_indices)
    _require(bool(indices), "full-attention layer plan is empty")
    return indices[0]


def apply_h01_stale_prefix_page_order(group: Any, plan: Any) -> MutationHandle:
    """Model a stale prefix identity by swapping two live document-page routes.

    Only the request-local block table is changed.  Pool bytes, tensor
    geometry, request identity, and reservation metadata are untouched.
    """

    layer_index = _first_full_layer(plan)
    request_index = 0
    sequence = group.requests[request_index].layers[layer_index].sequence
    document_blocks = int(sequence.arena.document_blocks_per_sequence)
    _require(document_blocks >= 3, "H01 requires at least three document blocks")
    left, right = 1, 2
    saved = sequence.block_table[0, [left, right]].detach().clone()

    def capture() -> Mapping[str, Any]:
        return _tensor_route_descriptor(
            sequence.block_table,
            request_index=request_index,
            layer_index=layer_index,
            field="request_block_table",
        )

    pre = dict(capture())
    sequence.block_table[0, left].copy_(saved[1])
    sequence.block_table[0, right].copy_(saved[0])
    mutated = dict(capture())
    _require(
        _flat_int_values(sequence.block_table[0, [left, right]])
        == list(reversed(_flat_int_values(saved))),
        "H01 selected routes were not swapped",
    )

    def undo() -> None:
        sequence.block_table[0, left].copy_(saved[0])
        sequence.block_table[0, right].copy_(saved[1])

    return MutationHandle(
        fault_id="H01",
        target_kind="request-local prefix page order",
        pre_descriptor=pre,
        mutated_descriptor=mutated,
        capture=capture,
        undo=undo,
    )


def apply_h03_future_reservation_reuse(group: Any, plan: Any) -> MutationHandle:
    """Model latent allocator reuse in a not-yet-consumed private reservation.

    The second private reservation of request 0 is rebound to request 1.  The
    formal H03 horizon is exactly one appended token from a 4095-token prefix,
    so only reservation index 0 may become active; index 1 must remain latent.
    """

    layer_index = _first_full_layer(plan)
    target_request = 0
    peer_request = 1
    target = group.requests[target_request].layers[layer_index].sequence
    peer = group.requests[peer_request].layers[layer_index].sequence
    target_flat = target.reservations.reshape(-1)
    peer_flat = peer.reservations.reshape(-1)
    _require(target_flat.numel() >= 2, "H03 requires two private reservations")
    _require(peer_flat.numel() >= 2, "H03 peer requires two private reservations")
    reservation_index = 1
    saved = target_flat[reservation_index].detach().clone()
    peer_value = peer_flat[reservation_index].detach().clone()
    _require(int(saved.item()) != int(peer_value.item()), "H03 peer route is not distinct")

    def capture() -> Mapping[str, Any]:
        return _tensor_route_descriptor(
            target.reservations,
            request_index=target_request,
            layer_index=layer_index,
            field="private_block_reservations",
        )

    pre = dict(capture())
    target_flat[reservation_index].copy_(peer_value)
    mutated = dict(capture())
    _require(
        int(target_flat[reservation_index].item()) == int(peer_value.item()),
        "H03 target reservation did not adopt the peer value",
    )

    def undo() -> None:
        target_flat[reservation_index].copy_(saved)

    return MutationHandle(
        fault_id="H03",
        target_kind="future private reservation",
        pre_descriptor=pre,
        mutated_descriptor=mutated,
        capture=capture,
        undo=undo,
    )


STATE_MUTATORS: Mapping[str, Callable[[Any, Any], MutationHandle]] = MappingProxyType(
    {
        "H01": apply_h01_stale_prefix_page_order,
        "H03": apply_h03_future_reservation_reuse,
    }
)


def apply_state_fault(fault_id: str, group: Any, plan: Any) -> MutationHandle:
    try:
        mutator = STATE_MUTATORS[fault_id]
    except KeyError as exc:
        raise HeldOutFaultConfigurationError(
            f"{fault_id!r} is not a state-mutation fault"
        ) from exc
    return mutator(group, plan)


def h02_action_sequence(*, request_index: int = 0) -> dict[str, Any]:
    """Return the frozen exact-boundary retry schedule for H02.

    The executor runs the same one-token input twice while reporting a single
    advertised logical advance.  No state is rewound: the second call begins
    from recurrent state that already includes that token.
    """

    _require(request_index == 0, "H02 request index is frozen to zero")
    events = [
        {
            "event_index": 0,
            "request_index": request_index,
            "role": "advertised-boundary-token",
            "input_coordinate": "frozen_query_bank[rank][0][31]",
            "externally_advertised": True,
        },
        {
            "event_index": 1,
            "request_index": request_index,
            "role": "hidden-resume-retry-of-same-boundary-token",
            "input_coordinate": "frozen_query_bank[rank][0][31]",
            "externally_advertised": False,
        },
    ]
    return {
        "schema_version": "forkaudit-r29-heldout-action-sequence-v1",
        "fault_id": "H02",
        "advertised_logical_advance_tokens": 1,
        "actual_model_invocations": 2,
        "events": events,
        "events_sha256": sha256_json(events),
        "fresh_case_disposal_required": True,
    }


def implementation_bindings() -> dict[str, Any]:
    functions = {
        "H01": apply_h01_stale_prefix_page_order,
        "H02": h02_action_sequence,
        "H03": apply_h03_future_reservation_reuse,
    }
    return {
        fault_id: {
            "module": MODULE_BINDING,
            "qualname": function.__qualname__,
            "source_sha256": hashlib.sha256(
                inspect.getsource(function).encode("utf-8")
            ).hexdigest(),
        }
        for fault_id, function in functions.items()
    }


def validate_frozen_suite(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "suite must be a JSON object")
    _require(value.get("schema_version") == SUITE_SCHEMA, "suite schema drift")
    faults = value.get("faults")
    _require(isinstance(faults, list), "suite faults must be a list")
    _require([row.get("fault_id") for row in faults] == list(FAULT_IDS), "fault order drift")
    serialized = canonical_bytes(value).decode("utf-8").lower()
    for forbidden in ("expected_gate", "observed_gate", "gate_id"):
        _require(forbidden not in serialized, f"outcome-leaking field present: {forbidden}")
    _require(value.get("detection_rate_reported") is False, "suite cannot report a rate")
    _require(
        value.get("author_executor_separation", {}).get("executor_must_not_edit_suite")
        is True,
        "author/executor separation is not frozen",
    )
    _require(
        value.get("implementation_bindings") == implementation_bindings(),
        "fault implementation binding drift",
    )
    return {
        "validated": True,
        "schema_version": SUITE_SCHEMA,
        "fault_ids": list(FAULT_IDS),
        "suite_sha256": sha256_json(value),
        "contains_expected_detector_mapping": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-frozen", type=Path)
    parser.add_argument("--print-bindings", action="store_true")
    args = parser.parse_args()
    if args.print_bindings:
        print(json.dumps(implementation_bindings(), indent=2, sort_keys=True))
    if args.verify_frozen is not None:
        value = json.loads(args.verify_frozen.read_text(encoding="utf-8"))
        print(json.dumps(validate_frozen_suite(value), indent=2, sort_keys=True))
    _require(args.print_bindings or args.verify_frozen is not None, "no action requested")


if __name__ == "__main__":
    main()


__all__ = [
    "ACTION_SEQUENCE_FAULT_IDS",
    "FAULT_IDS",
    "HeldOutFaultConfigurationError",
    "MutationHandle",
    "MODULE_BINDING",
    "STATE_MUTATION_FAULT_IDS",
    "STATE_MUTATORS",
    "SUITE_SCHEMA",
    "apply_h01_stale_prefix_page_order",
    "apply_h03_future_reservation_reuse",
    "apply_state_fault",
    "canonical_bytes",
    "h02_action_sequence",
    "implementation_bindings",
    "sha256_file",
    "sha256_json",
    "validate_frozen_suite",
]
