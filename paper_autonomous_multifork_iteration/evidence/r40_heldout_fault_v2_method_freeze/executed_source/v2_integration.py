"""Integration adapters for capturing and evaluating the three method-v2 gates."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from v2_common import require, verify_seal
from v2_predicates import (
    ALLOCATOR_PHASES,
    ALLOCATOR_SCHEMA,
    ATOMIC_RECEIPT_SCHEMA,
    SEMANTIC_SCHEMA,
    evaluate_allocator_pair,
    evaluate_atomic_sequence,
    evaluate_semantic_pair,
)


def semantic_arm_from_receipts(receipts: Sequence[Mapping[str, Any]], arm: str) -> dict[str, Any]:
    """Derive a semantic arm from sealed call receipts without dropping calls."""

    require(arm in ("reference", "candidate"), "semantic arm label")
    require(isinstance(receipts, Sequence) and not isinstance(receipts, (str, bytes)), "semantic receipts")
    require(len(receipts) > 0, "nonempty semantic receipts")
    calls = []
    for receipt in receipts:
        require(receipt.get("schema_version") == ATOMIC_RECEIPT_SCHEMA, "semantic receipt schema")
        verify_seal(receipt, "semantic receipt")
        calls.append({
            "call_key": dict(receipt["call_key"]),
            "token_id": receipt["surfaced_token_id"],
            "logits": dict(receipt["logits"]),
        })
    return {"schema_version": SEMANTIC_SCHEMA, "arm": arm, "calls": calls}


class AllocatorArmCapture:
    """Capture fixed synchronized allocator endpoints exactly once in order."""

    def __init__(
        self,
        arm: str,
        synchronize: Callable[[], str],
        reset_peak: Callable[[], None],
        read_allocator: Callable[[], Mapping[str, int]],
    ) -> None:
        require(arm in ("reference", "candidate"), "allocator arm label")
        self._arm = arm
        self._synchronize = synchronize
        self._reset_peak = reset_peak
        self._read_allocator = read_allocator
        self._rows: list[dict[str, Any]] = []
        self._reset_done = False

    def capture(self, phase: str) -> None:
        expected = ALLOCATOR_PHASES[len(self._rows)] if len(self._rows) < len(ALLOCATOR_PHASES) else None
        require(phase == expected, "allocator capture phase order")
        if phase == "H0":
            self._synchronize()
            self._reset_peak()
            self._reset_done = True
        event_id = self._synchronize()
        require(isinstance(event_id, str) and event_id != "", "allocator sync event")
        reading = self._read_allocator()
        require(isinstance(reading, Mapping), "allocator reading")
        require(set(reading.keys()) == {"current_allocated_bytes", "peak_allocated_bytes"},
                "allocator reading fields")
        current = reading["current_allocated_bytes"]
        peak = reading["peak_allocated_bytes"]
        require(type(current) is int and current >= 0, "allocator current")
        require(type(peak) is int and peak >= current, "allocator peak")
        self._rows.append({
            "phase": phase,
            "synchronized": True,
            "sync_event_id": event_id,
            "current_allocated_bytes": current,
            "peak_allocated_bytes": peak,
        })

    def finish(self) -> dict[str, Any]:
        require(self._reset_done and len(self._rows) == len(ALLOCATOR_PHASES), "allocator capture incomplete")
        return {
            "schema_version": ALLOCATOR_SCHEMA,
            "arm": self._arm,
            "peak_reset_before_h0": True,
            "endpoints": list(self._rows),
        }


def evaluate_lane_pair(
    *,
    reference_receipts: Sequence[Mapping[str, Any]],
    candidate_receipts: Sequence[Mapping[str, Any]],
    expected_schedule: Sequence[Mapping[str, Any]],
    reference_allocator: Mapping[str, Any],
    candidate_allocator: Mapping[str, Any],
    reference_root: Path,
    candidate_root: Path,
    semantic_policy: Mapping[str, Any],
    atomic_policy_sha256: str,
    vocab_size: int,
) -> dict[str, Any]:
    """Run all gates; baselines retain separate attribution in the verdict."""

    reference_semantic = semantic_arm_from_receipts(reference_receipts, "reference")
    candidate_semantic = semantic_arm_from_receipts(candidate_receipts, "candidate")
    semantic = evaluate_semantic_pair(
        reference_semantic, candidate_semantic, reference_root, candidate_root,
        semantic_policy, vocab_size,
    )
    allocator = evaluate_allocator_pair(reference_allocator, candidate_allocator)
    reference_atomic = evaluate_atomic_sequence(reference_receipts, expected_schedule, atomic_policy_sha256)
    candidate_atomic = evaluate_atomic_sequence(candidate_receipts, expected_schedule, atomic_policy_sha256)
    return {
        "schema_version": "forkaudit-method-v2-lane-pair-verdict-v1",
        "semantic": semantic,
        "allocator": allocator,
        "reference_atomic": reference_atomic,
        "candidate_atomic": candidate_atomic,
        "passed": all((semantic["passed"], allocator["passed"],
                       reference_atomic["passed"], candidate_atomic["passed"])),
        "attribution_policy": {
            "semantic": "paired_semantic_baseline",
            "allocator": "paired_allocator_baseline",
            "atomic": "hybrid_atomic_version_coherence",
        },
    }

