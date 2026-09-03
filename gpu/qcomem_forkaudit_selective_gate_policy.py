from __future__ import annotations

"""Scoped, receipt-producing suppression of exactly one ForkAudit gate.

This module is deliberately separate from the detector runner.  It instruments
the three gate surfaces used by the frozen RR2 implementation:

* ``qcomem_vllm_paged_multifork_resident._runtime_require``;
* the serialized GDN witness predicates; and
* the strict post-RoPE position validator used by ``MultiForkHitLedger``.

Every non-target failure is delegated to the original implementation.  The
original function objects are restored on every exit path and their identities
are checked before a case can be considered valid.
"""

import hashlib
import inspect
import json
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence


POSITION_GATE = "POSITION_CANONICAL_VALUES"
GDN_BASE_GATE = "gdn_completed_vs_base_disjoint"
GDN_PEER_GATE = "gdn_completed_vs_peers_disjoint"
GDN_TARGET_GATES = frozenset((GDN_BASE_GATE, GDN_PEER_GATE))


class GatePolicyError(RuntimeError):
    """Raised when the selective policy itself is invalid or leaks a hook."""


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
        source = inspect.getsource(function).encode("utf-8")
        source_sha256 = hashlib.sha256(source).hexdigest()
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
    """Return a pointer-free digest of the first caller outside this module."""

    frame = inspect.currentframe()
    try:
        frame = None if frame is None else frame.f_back
        while frame is not None and frame.f_globals.get("__name__") == __name__:
            frame = frame.f_back
        if frame is None:
            row = {
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
        # Do not keep frame locals (and therefore live CUDA objects) alive.
        del frame


class SelectiveGatePolicy:
    """Install one target-only suppression policy for a disposable case.

    ``target_gate=None`` is the all-gates-on lane.  The same wrappers are
    installed there so clean cases also prove hook installation/restoration,
    but every predicate delegates to the original implementation.
    """

    def __init__(
        self,
        target_gate: str | None,
        *,
        resident_module: ModuleType | Any | None = None,
        storage_module: ModuleType | Any | None = None,
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
            "schema_version": "forkaudit-selective-gate-event-v2",
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
            from qcomem_qwen35_vllm_paged_integration import (
                Qwen35VllmPagedIntegrationError,
            )

            original = originals["resident_position_validator"]
            try:
                return original(
                    position_ids,
                    query=query,
                    total_length=total_length,
                    strict_tail_values=strict_tail_values,
                )
            except Qwen35VllmPagedIntegrationError as strict_error:
                if (
                    self.target_gate != POSITION_GATE
                    or not strict_tail_values
                    or str(strict_error)
                    != "position_ids are not the canonical contiguous causal tail"
                ):
                    raise
                self._record(
                    gate_id=POSITION_GATE,
                    message=f"strict post-RoPE position validation failed: {strict_error}",
                    predicate_function="validate_qwen35_post_rope_position_ids",
                    predicate_source="qcomem_qwen35_vllm_paged_integration",
                    details={
                        "strict_tail_values_attempted": True,
                        "continuation_replay": "same input with strict_tail_values=False",
                        "strict_error_type": type(strict_error).__name__,
                    },
                )
                # This is a second validation pass, not a disabled ledger flag:
                # shape, dtype, device, and causal-position schema still apply.
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
            "schema_version": "forkaudit-selective-gate-policy-receipt-v2",
            "target_gate_id": self.target_gate,
            "lane": "all-gates-on" if self.target_gate is None else "target-only-suppressed",
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
    "POSITION_GATE",
    "SelectiveGatePolicy",
]
