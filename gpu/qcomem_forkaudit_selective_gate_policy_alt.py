from __future__ import annotations

"""Exact, scoped suppression of one ForkAudit gate for detector ablations.

The policy instruments the three failure surfaces exercised by the frozen RR2
mutants.  It never suppresses an exception merely because a target gate was
requested: resident gates must carry the exact gate id, GDN overlap failures
must be observed by the named disjoint helper, and the M6 continuation is
allowed only for the one canonical-tail integration error.
"""

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from qcomem_qwen35_vllm_paged_integration import (
    Qwen35VllmPagedIntegrationError,
)


POSITION_GATE = "POSITION_CANONICAL_VALUES"
POSITION_CANONICAL_TAIL_ERROR = (
    "position_ids are not the canonical contiguous causal tail"
)
GDN_BASE_GATE = "gdn_completed_vs_base_disjoint"
GDN_PEER_GATE = "gdn_completed_vs_peers_disjoint"
GDN_TARGET_GATES = frozenset((GDN_BASE_GATE, GDN_PEER_GATE))


class GatePolicyError(RuntimeError):
    """The suppression policy was misconfigured or leaked an installed hook."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GatePolicyError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _function_descriptor(value: Any) -> dict[str, Any]:
    function = getattr(value, "__func__", value)
    try:
        source_sha256 = hashlib.sha256(
            inspect.getsource(function).encode("utf-8")
        ).hexdigest()
    except (OSError, TypeError):
        source_sha256 = None
    return {
        "module": str(getattr(function, "__module__", type(function).__module__)),
        "qualname": str(
            getattr(function, "__qualname__", type(function).__qualname__)
        ),
        "source_sha256": source_sha256,
    }


def _callsite_descriptor() -> dict[str, Any]:
    """Return pointer-free evidence for the first caller outside this module."""

    frame = inspect.currentframe()
    try:
        frame = None if frame is None else frame.f_back
        while frame is not None and frame.f_globals.get("__name__") == __name__:
            frame = frame.f_back
        if frame is None:
            row: dict[str, Any] = {
                "module": "<unavailable>",
                "function": "<unavailable>",
                "line": -1,
                "source_line_sha256": None,
            }
        else:
            info = inspect.getframeinfo(frame, context=1)
            source_line = "" if not info.code_context else info.code_context[0].strip()
            row = {
                "module": str(frame.f_globals.get("__name__", "<unknown>")),
                "file": Path(info.filename).name,
                "function": info.function,
                "line": int(info.lineno),
                "source_line_sha256": hashlib.sha256(
                    source_line.encode("utf-8")
                ).hexdigest(),
            }
        row["callsite_sha256"] = _sha256_json(row)
        return row
    finally:
        del frame


class SelectiveGatePolicy:
    """Install one target-only policy for one disposable clean/mutant case.

    ``target_gate=None`` is the all-gates-on clean lane.  The wrappers remain
    installed there so clean cases also prove scope integrity and restoration.
    """

    def __init__(
        self,
        target_gate: str | None,
        *,
        resident_module: Any | None = None,
        storage_module: Any | None = None,
    ) -> None:
        if resident_module is None:
            import qcomem_vllm_paged_multifork_resident as resident_module
        if storage_module is None:
            import qcomem_forkaudit_storage_witness as storage_module
        _require(target_gate is None or isinstance(target_gate, str), "target gate type")
        self.target_gate = target_gate
        self.resident = resident_module
        self.storage = storage_module
        self.events: list[dict[str, Any]] = []
        self._originals: dict[str, Any] = {}
        self._wrappers: dict[str, Any] = {}
        self._entered = False
        self._exited = False
        self._scope_integrity_before_restore: bool | None = None
        self._restoration_verified: bool | None = None

    def _record(
        self,
        *,
        gate_id: str,
        message: str,
        predicate_function: str,
        predicate_source: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        _require(gate_id == self.target_gate, "attempted non-target suppression")
        event: dict[str, Any] = {
            "schema_version": "forkaudit-selective-gate-event-alt-v1",
            "gate_id": gate_id,
            "message": str(message),
            "predicate_function": predicate_function,
            "predicate_source": predicate_source,
            "ordinal": len(self.events),
            "callsite": _callsite_descriptor(),
        }
        if details:
            event["details"] = dict(details)
        event["event_sha256"] = _sha256_json(event)
        self.events.append(event)

    def __enter__(self) -> "SelectiveGatePolicy":
        _require(not self._entered, "gate policy cannot be re-entered")
        self._entered = True
        resident = self.resident
        storage = self.storage
        originals = {
            "resident_runtime_require": resident._runtime_require,
            "resident_position_validator": resident.validate_qwen35_post_rope_position_ids,
            "storage_require": storage._require,
            "storage_assert_sets_disjoint": storage._assert_sets_disjoint,
        }
        self._originals = originals

        def selective_runtime_require(
            condition: bool, gate_id: str, message: str
        ) -> None:
            if not condition and gate_id == self.target_gate:
                self._record(
                    gate_id=gate_id,
                    message=message,
                    predicate_function="_runtime_require",
                    predicate_source="qcomem_vllm_paged_multifork_resident",
                )
                return
            originals["resident_runtime_require"](condition, gate_id, message)

        def selective_storage_require(
            condition: bool,
            message: str,
            *,
            gate_id: str = "gdn_storage_schema",
        ) -> None:
            if not condition and gate_id == self.target_gate:
                self._record(
                    gate_id=gate_id,
                    message=message,
                    predicate_function="_require",
                    predicate_source="qcomem_forkaudit_storage_witness",
                )
                return
            originals["storage_require"](condition, message, gate_id=gate_id)

        def selective_assert_sets_disjoint(
            left_rows: Sequence[dict[str, Any]],
            right_rows: Sequence[dict[str, Any]],
            label: str,
            *,
            gate_id: str,
        ) -> int:
            if gate_id != self.target_gate:
                return originals["storage_assert_sets_disjoint"](
                    left_rows, right_rows, label, gate_id=gate_id
                )
            comparisons = 0
            for left in left_rows:
                for right in right_rows:
                    comparisons += 1
                    if storage._rows_overlap(left, right):
                        relation = (
                            "exact alias"
                            if storage._rows_exact_alias(left, right)
                            else "partial overlap"
                        )
                        self._record(
                            gate_id=gate_id,
                            message=f"{label} has {relation}",
                            predicate_function="_assert_sets_disjoint",
                            predicate_source="qcomem_forkaudit_storage_witness",
                            details={
                                "relation": relation,
                                "left_coordinate": list(storage._coordinate_key(left)),
                                "right_coordinate": list(storage._coordinate_key(right)),
                            },
                        )
            return comparisons

        def selective_position_validator(
            position_ids: Any,
            *,
            query: Any,
            total_length: int,
            strict_tail_values: bool,
        ) -> Any:
            original = originals["resident_position_validator"]
            try:
                return original(
                    position_ids,
                    query=query,
                    total_length=total_length,
                    strict_tail_values=strict_tail_values,
                )
            except Qwen35VllmPagedIntegrationError as strict_error:
                # M6 is the only legal validator bypass.  Type, exact message,
                # and the strict invocation bit must all match; AttributeError
                # and every other integration failure escape unchanged.
                if not (
                    self.target_gate == POSITION_GATE
                    and strict_tail_values is True
                    and str(strict_error) == POSITION_CANONICAL_TAIL_ERROR
                ):
                    raise
                self._record(
                    gate_id=POSITION_GATE,
                    message=str(strict_error),
                    predicate_function="validate_qwen35_post_rope_position_ids",
                    predicate_source="qcomem_qwen35_vllm_paged_integration",
                    details={
                        "strict_tail_values_attempted": True,
                        "continuation_replay": (
                            "same input with strict_tail_values=False"
                        ),
                        "strict_error_type": type(strict_error).__name__,
                        "strict_error_message_exact": True,
                    },
                )
                return original(
                    position_ids,
                    query=query,
                    total_length=total_length,
                    strict_tail_values=False,
                )

        wrappers = {
            "resident_runtime_require": selective_runtime_require,
            "resident_position_validator": selective_position_validator,
            "storage_require": selective_storage_require,
            "storage_assert_sets_disjoint": selective_assert_sets_disjoint,
        }
        self._wrappers = wrappers
        resident._runtime_require = wrappers["resident_runtime_require"]
        resident.validate_qwen35_post_rope_position_ids = wrappers[
            "resident_position_validator"
        ]
        storage._require = wrappers["storage_require"]
        storage._assert_sets_disjoint = wrappers["storage_assert_sets_disjoint"]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback_value: Any) -> bool:
        del exc_type, exc, traceback_value
        _require(self._entered and not self._exited, "gate policy exit state")
        self._exited = True
        resident = self.resident
        storage = self.storage
        wrappers = self._wrappers
        originals = self._originals
        self._scope_integrity_before_restore = bool(
            resident._runtime_require is wrappers["resident_runtime_require"]
            and resident.validate_qwen35_post_rope_position_ids
            is wrappers["resident_position_validator"]
            and storage._require is wrappers["storage_require"]
            and storage._assert_sets_disjoint
            is wrappers["storage_assert_sets_disjoint"]
        )
        resident._runtime_require = originals["resident_runtime_require"]
        resident.validate_qwen35_post_rope_position_ids = originals[
            "resident_position_validator"
        ]
        storage._require = originals["storage_require"]
        storage._assert_sets_disjoint = originals["storage_assert_sets_disjoint"]
        self._restoration_verified = bool(
            resident._runtime_require is originals["resident_runtime_require"]
            and resident.validate_qwen35_post_rope_position_ids
            is originals["resident_position_validator"]
            and storage._require is originals["storage_require"]
            and storage._assert_sets_disjoint
            is originals["storage_assert_sets_disjoint"]
        )
        if not self._scope_integrity_before_restore:
            raise GatePolicyError("selective gate hook identity changed inside scope")
        if not self._restoration_verified:
            raise GatePolicyError("selective gate hooks were not restored")
        return False

    def receipt(self) -> dict[str, Any]:
        _require(self._exited, "gate policy receipt requested before scope exit")
        return {
            "schema_version": "forkaudit-selective-gate-policy-receipt-alt-v1",
            "target_gate_id": self.target_gate,
            "lane": (
                "all-gates-on"
                if self.target_gate is None
                else "target-only-suppressed"
            ),
            "suppressed_event_count": len(self.events),
            "suppressed_gate_ids": [event["gate_id"] for event in self.events],
            "events": list(self.events),
            "scope_integrity_before_restore": self._scope_integrity_before_restore,
            "original_function_descriptors": {
                name: _function_descriptor(value)
                for name, value in self._originals.items()
            },
            "all_original_function_identities_restored": self._restoration_verified,
        }


__all__ = [
    "GDN_BASE_GATE",
    "GDN_PEER_GATE",
    "GDN_TARGET_GATES",
    "GatePolicyError",
    "POSITION_CANONICAL_TAIL_ERROR",
    "POSITION_GATE",
    "SelectiveGatePolicy",
]
