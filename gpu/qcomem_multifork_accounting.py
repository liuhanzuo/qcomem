"""Torch-free bookkeeping for the shared-packed multifork experiment (C1).

This module deliberately imports nothing from ``torch``.  Everything here is
pure integer/float arithmetic and set algebra over plain dictionaries, so the
three things the C1 experiment has to get right can be unit tested on a laptop
with no CUDA, no Transformers and no checkpoint:

1. **The ownership ledger.**  Which bytes one resident packed entry shares
   across ``N`` concurrent requests, which bytes each request privately owns,
   and whether any two requests' mutable state overlaps.  The byte-range
   algebra mirrors the storage-witness form already used by the ForkAudit
   transfer receipts (opaque storage id + normalized ``[start, end)`` view
   range), but is re-expressed here without a torch import so the aggregator
   can replay an archived shard on a machine with no GPU stack.

2. **The transient working set, for both arms.**  The manuscript's Eq. 1
   assumes the active workspace is method-independent.  It is not: the Read
   path materializes a dequantized view whose size depends on the entry.  This
   module carries per-request *materialized* bytes, *peak transient*
   allocation and *steady-state resident* bytes as first-class fields for both
   Q-CoMem and full-prefix, and fits the affine resident model
   ``intercept + slope * N`` that those fields imply.

3. **Coverage versus verdict.**  The ForkAudit contract is instantiated here as
   a table of targets, each with an applicability, a maximum status, and a set
   of *mandatory receipt slots*.  ``build_multifork_target_rows`` refuses to
   emit a passing status for a target whose mandatory slots are missing,
   duplicated, modified or unbound; such a target is ``open`` with
   ``predicate_passed = None``, never silently ``full``.

Nothing in this module changes, wraps, or re-implements a published accounting
function.  ``PackedCache.nbytes``, ``qcomem_torch.cache_nbytes``,
``qcomem_deployment.capacity_estimate`` and the Eq. 3 identity in
``qcomem_eq3_accounting`` are untouched and are the authorities for the numbers
this module consumes.

Units are bytes everywhere unless a name ends in ``_seconds`` or ``_ratio``.
"""

from __future__ import annotations

import statistics
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL = "qcomem-shared-packed-multifork-v1"
SHARD_SCHEMA = "qcomem-shared-packed-multifork-shard-v1"
AGGREGATE_SCHEMA = "qcomem-shared-packed-multifork-aggregate-v1"

#: fork modes this experiment can select.  ``private-materialize`` is the mode
#: the published Tables 1/2 measure and it must stay reproducible.
FORK_MODES = (
    "private-materialize",
    "shared-packed-view",
)

#: when the shared mode is selected, when the borrowed mutable base rebinds
REBIND_POLICIES = (
    # clone mutable conv/recurrent leaves at fork time
    "setup",
    # borrow them read-only at fork and rebind to private storage at the
    # registered transition, immediately before the first lower-layer call
    "transition",
)

#: how a shared attention prefix is combined with a request's appended tail
TAIL_POLICIES = (
    # keep the document prefix borrowed for the whole request and retain only
    # the appended tail; the concatenation handed to attention is transient
    "borrowed-prefix",
    # the frozen qcomem_paged behaviour: read the shared prefix and bind a
    # newly concatenated private tensor on the first append
    "materialized-tail",
)

#: arms measured side by side, so the transient working set exists for both
MULTIFORK_ARMS = ("qcomem-shared-packed", "qcomem-private-materialize", "full-prefix")


# ---------------------------------------------------------------------------
# ForkAudit contract: seven inherited targets plus three packed-entry targets
# ---------------------------------------------------------------------------

#: The seven targets are the ones the manuscript already names.  Targets 8--10
#: are the obligations the manuscript names as untested because the existing
#: audit never sees a packed entry.
MULTIFORK_TARGET_CONTRACT: tuple[dict[str, Any], ...] = (
    {
        "target_index": 1,
        "target": "frozen_identity",
        "family": "forkaudit-seven",
        "predicate_id": "FROZEN_ENTRY_POLICY_AND_INPUT_BINDINGS",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 2,
        "target": "prefix_immutability",
        "family": "forkaudit-seven",
        "predicate_id": "PERSISTENT_PREFIX_CONTENT_UNCHANGED",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 3,
        "target": "private_ownership",
        "family": "forkaudit-seven",
        "predicate_id": "ALL_MUTABLE_CACHE_STORAGE_PAIRWISE_DISJOINT",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 4,
        "target": "tail_safe_append",
        "family": "forkaudit-seven",
        "predicate_id": "SHARED_PREFIX_NOT_WRITTEN_ON_APPEND",
        "applicability": "applicable",
        # A Transformers DynamicCache has no fixed-size page and therefore no
        # partial-page tail.  What this path *does* have, and what the earlier
        # Transformers transfer did not, is a genuinely shared attention prefix
        # with an append that must not write it.  That is a real
        # copy-before-append obligation, but it is not page-granular, so the
        # target is capped at partial even when its predicate passes.
        "maximum_status": "partial",
    },
    {
        "target_index": 5,
        "target": "dispatch_provenance",
        "family": "forkaudit-seven",
        "predicate_id": "BOUNDED_HOST_SIDE_CALL_PROVENANCE",
        "applicability": "applicable",
        "maximum_status": "partial",
    },
    {
        "target_index": 6,
        "target": "cross_arm_equivalence",
        "family": "forkaudit-seven",
        "predicate_id": "SHARED_FORK_EQUALS_PRIVATE_MATERIALIZATION",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 7,
        "target": "cross_n_prefix_consistency",
        "family": "forkaudit-seven",
        "predicate_id": "FIRST_REQUEST_PREFIX_INVARIANT_ACROSS_N",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 8,
        "target": "dequantized_view_immutability",
        "family": "packed-entry-obligation",
        "predicate_id": "SHARED_DEQUANTIZED_VIEW_CONTENT_UNCHANGED",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 9,
        "target": "residual_chunk_binding",
        "family": "packed-entry-obligation",
        "predicate_id": "DOCUMENT_AND_QUERY_RESIDUAL_CHUNKS_BOUND_DISTINCTLY",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 10,
        "target": "packed_entry_lifetime",
        "family": "packed-entry-obligation",
        "predicate_id": "PACKED_ENTRY_CONTENT_AND_REFERENCE_LIFETIME",
        "applicability": "applicable",
        "maximum_status": "full",
    },
)

#: Mandatory receipt slots per target.  Completeness quantifies over these:
#: every slot must be present, unique, unmodified, and bound to the live object
#: its receipt names.  This is the ``Coverage_i`` of the manuscript's Eq. 4.
MANDATORY_SLOTS: dict[str, tuple[str, ...]] = {
    "frozen_identity": (
        "entry_identity",
        "policy_identity",
        "document_token_identity",
        "query_token_identity",
        "adapter_identity",
    ),
    "prefix_immutability": (
        "shared_view_setup_digest",
        "shared_view_final_digest",
        "packed_entry_setup_digest",
    ),
    "private_ownership": (
        "setup_inventory",
        "transition_inventory",
        "final_inventory",
    ),
    "tail_safe_append": (
        "append_events",
        "shared_attention_inventory",
        "final_inventory",
    ),
    "dispatch_provenance": (
        "adapter_call_log",
        "layer_class_receipt",
    ),
    "cross_arm_equivalence": (
        "shared_token_traces",
        "private_token_traces",
    ),
    "cross_n_prefix_consistency": ("cross_n_token_traces",),
    "dequantized_view_immutability": (
        "shared_view_setup_digest",
        "shared_view_final_digest",
        "view_alias_inventory",
    ),
    "residual_chunk_binding": ("residual_binding_events",),
    "packed_entry_lifetime": (
        "packed_entry_setup_digest",
        "packed_entry_final_digest",
        "fork_release_ledger",
    ),
}

#: What each target still does not establish even when it passes.  These are
#: printed with the verdict so a reader never has to infer scope from a status.
TARGET_EXACT_MISSINGNESS: dict[str, tuple[str, ...]] = {
    "tail_safe_append": (
        "no fixed-size page granularity; the tail is a whole-tensor concatenation",
        "no partial-page copy-before-append event exists in a Transformers DynamicCache",
    ),
    "dispatch_provenance": (
        "compiled CUDA/Triton kernel binary fingerprint",
        "kernel autotuning-choice fingerprint",
        "hardware instruction trace",
    ),
    "cross_arm_equivalence": (
        "equality is over emitted token ids and per-step logits of the two "
        "Q-CoMem fork modes; it is not an equality with the full-prefix arm",
    ),
    "cross_n_prefix_consistency": (
        "invariance is checked over the declared fanouts only, not for arbitrary N",
    ),
    "packed_entry_lifetime": (
        "reference release is observed through Python-visible references and a "
        "content digest; it is not an allocator-level proof of deallocation",
    ),
}

TARGET_SCOPE_NOTES: dict[str, str] = {
    "tail_safe_append": (
        "Establishes that a shared attention prefix is never written by an "
        "append and that the post-append storage is disjoint from it. It does "
        "not establish paged partial-tail safety."
    ),
    "dispatch_provenance": (
        "Full only at the Python adapter/layer-class receipt level; therefore "
        "frozen as partial even when the predicate passes."
    ),
    "cross_arm_equivalence": (
        "The N>1 shared-packed fork must reproduce, token for token, what the "
        "N=1 private-materialization path produces on the same inputs. That is "
        "this experiment's own correctness gate."
    ),
    "dequantized_view_immutability": (
        "The single dequantized document view is content-identical before the "
        "first fork and after the last request completes, and every request's "
        "immutable document tensors alias that one view."
    ),
    "residual_chunk_binding": (
        "Every request seeds its suffix from the one shared document residual "
        "chunk at offset 0 and prefills its own query residual chunk at the "
        "document offset, in two distinct calls; the two chunks never share "
        "storage."
    ),
    "packed_entry_lifetime": (
        "The durable packed entry is content-unchanged across the whole "
        "multifork execution and outlives every request fork."
    ),
}

VALID_STATUSES = ("full", "partial", "open", "not_applicable")
COVERAGE_STATES = ("complete", "incomplete", "not_applicable")


class MultiforkAccountingError(ValueError):
    """An evidence-integrity failure, not a scientific negative."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MultiforkAccountingError(message)


def target_names() -> tuple[str, ...]:
    return tuple(row["target"] for row in MULTIFORK_TARGET_CONTRACT)


def packed_entry_obligation_names() -> tuple[str, ...]:
    return tuple(
        row["target"]
        for row in MULTIFORK_TARGET_CONTRACT
        if row["family"] == "packed-entry-obligation"
    )


def evaluate_slot_coverage(
    target: str, receipts: Mapping[str, Any]
) -> dict[str, Any]:
    """Check the mandatory receipt slots of one target.

    A slot record is a mapping with boolean ``present``, ``unique``, ``bound``
    and ``modified`` fields.  Coverage is ``complete`` only when every mandatory
    slot is present, unique, bound and unmodified.  A slot that is entirely
    absent from ``receipts`` counts as missing, not as vacuously satisfied.
    """

    _require(target in MANDATORY_SLOTS, f"unknown target: {target}")
    failures: list[dict[str, Any]] = []
    for slot in MANDATORY_SLOTS[target]:
        record = receipts.get(slot)
        if record is None:
            failures.append({"slot": slot, "reason": "missing"})
            continue
        if not isinstance(record, Mapping):
            failures.append({"slot": slot, "reason": "malformed_receipt"})
            continue
        for field, wanted in (
            ("present", True),
            ("unique", True),
            ("bound", True),
            ("modified", False),
        ):
            value = record.get(field)
            if type(value) is not bool:
                failures.append(
                    {"slot": slot, "reason": f"non_boolean_{field}"}
                )
            elif value is not wanted:
                failures.append({"slot": slot, "reason": f"slot_{field}_is_{value}"})
    return {
        "target": target,
        "mandatory_slots": list(MANDATORY_SLOTS[target]),
        "coverage": "complete" if not failures else "incomplete",
        "coverage_failures": failures,
    }


def build_multifork_target_rows(
    *,
    predicates: Mapping[str, Any],
    receipts: Mapping[str, Any],
    scope_overrides: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Emit one contract row per target, coverage separated from verdict.

    A target whose mandatory slots are incomplete is reported ``open`` with
    ``predicate_passed = None``: it cannot pass, and it is not silently
    downgraded to ``not_applicable`` either.  A target whose coverage is
    complete but whose predicate is missing or non-boolean raises, because a
    covered target with no verdict is an evidence defect rather than a result.
    """

    overrides = dict(scope_overrides or {})
    rows: list[dict[str, Any]] = []
    for contract in MULTIFORK_TARGET_CONTRACT:
        target = contract["target"]
        row = dict(contract)
        row["exact_missingness"] = list(TARGET_EXACT_MISSINGNESS.get(target, ()))
        row["scope_note"] = overrides.get(target, TARGET_SCOPE_NOTES.get(target, ""))
        if contract["applicability"] == "not_applicable":
            row.update(
                coverage="not_applicable",
                coverage_failures=[],
                mandatory_slots=list(MANDATORY_SLOTS.get(target, ())),
                status="not_applicable",
                predicate_passed=None,
            )
            rows.append(row)
            continue
        coverage = evaluate_slot_coverage(target, receipts)
        row.update(
            coverage=coverage["coverage"],
            coverage_failures=coverage["coverage_failures"],
            mandatory_slots=coverage["mandatory_slots"],
        )
        if coverage["coverage"] != "complete":
            row.update(status="open", predicate_passed=None)
            rows.append(row)
            continue
        passed = predicates.get(target)
        _require(
            type(passed) is bool,
            f"target {target} has complete coverage but no boolean predicate",
        )
        row.update(
            status=contract["maximum_status"] if passed else "open",
            predicate_passed=passed,
        )
        rows.append(row)
    return rows


def contract_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize contract rows without ever collapsing coverage into verdict."""

    _require(bool(rows), "contract summary requires at least one target row")
    applicable = [row for row in rows if row["applicability"] == "applicable"]
    covered = [row for row in applicable if row["coverage"] == "complete"]
    passed = [row for row in covered if row["predicate_passed"] is True]
    seven = [row for row in rows if row["family"] == "forkaudit-seven"]
    obligations = [
        row for row in rows if row["family"] == "packed-entry-obligation"
    ]
    return {
        "target_count": len(rows),
        "status_vector": [row["status"] for row in rows],
        "coverage_vector": [row["coverage"] for row in rows],
        "applicable_target_count": len(applicable),
        "covered_target_count": len(covered),
        "passed_target_count": len(passed),
        "all_applicable_targets_covered": len(covered) == len(applicable),
        "all_applicable_predicates_passed": (
            len(passed) == len(applicable) and len(applicable) > 0
        ),
        "seven_target_status_vector": [row["status"] for row in seven],
        "packed_entry_obligation_status_vector": [
            row["status"] for row in obligations
        ],
        "packed_entry_obligations_all_passed": bool(obligations)
        and all(row["predicate_passed"] is True for row in obligations),
        "open_targets": [row["target"] for row in rows if row["status"] == "open"],
        "uncovered_targets": [
            row["target"] for row in applicable if row["coverage"] != "complete"
        ],
        "overall_contract_status": (
            "open"
            if any(row["status"] == "open" for row in rows)
            else ("partial" if any(row["status"] == "partial" for row in rows) else "full")
        ),
    }


# ---------------------------------------------------------------------------
# ownership ledger over normalized storage byte ranges
# ---------------------------------------------------------------------------


def normalize_inventory(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate and sort a storage inventory in the receipt schema.

    Each row needs ``path``, ``role``, ``storage_id``, ``storage_nbytes``,
    ``view_start_bytes`` and ``view_end_bytes``.  Ranges must satisfy
    ``0 <= start <= end <= storage_nbytes``.  Empty views (``start == end``)
    are retained but never alias.
    """

    normalized: list[dict[str, Any]] = []
    for row in rows:
        _require(isinstance(row, Mapping), "inventory row is not a mapping")
        missing = {
            "path",
            "role",
            "storage_id",
            "storage_nbytes",
            "view_start_bytes",
            "view_end_bytes",
        } - set(row)
        _require(not missing, f"inventory row is missing fields: {sorted(missing)}")
        start = int(row["view_start_bytes"])
        end = int(row["view_end_bytes"])
        total = int(row["storage_nbytes"])
        _require(
            0 <= start <= end <= total,
            f"inventory row {row['path']!r} has an out-of-bounds view range",
        )
        normalized.append(
            {
                "path": str(row["path"]),
                "role": str(row["role"]),
                "storage_id": str(row["storage_id"]),
                "storage_nbytes": total,
                "view_start_bytes": start,
                "view_end_bytes": end,
                "view_nbytes": end - start,
            }
        )
    normalized.sort(
        key=lambda item: (
            item["role"],
            item["storage_id"],
            item["view_start_bytes"],
            item["view_end_bytes"],
            item["path"],
        )
    )
    return normalized


def unique_storage_nbytes(rows: Iterable[Mapping[str, Any]]) -> int:
    """Sum whole-storage bytes once per distinct storage id.

    This matches ``qcomem_torch.cache_nbytes``' deduplication rule, so a
    ledger built here reconciles with the frozen accountant.
    """

    seen: dict[str, int] = {}
    for row in rows:
        seen[str(row["storage_id"])] = int(row["storage_nbytes"])
    return sum(seen.values())


def range_overlaps(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Non-empty view-range intersections on a common storage id."""

    by_storage: dict[str, list[Mapping[str, Any]]] = {}
    for row in right:
        by_storage.setdefault(row["storage_id"], []).append(row)
    overlaps: list[dict[str, Any]] = []
    for left_row in left:
        for right_row in by_storage.get(left_row["storage_id"], ()):
            start = max(left_row["view_start_bytes"], right_row["view_start_bytes"])
            end = min(left_row["view_end_bytes"], right_row["view_end_bytes"])
            if start < end:
                overlaps.append(
                    {
                        "storage_id": left_row["storage_id"],
                        "left_role": left_row["role"],
                        "left_path": left_row["path"],
                        "right_role": right_row["role"],
                        "right_path": right_row["path"],
                        "intersection_start_bytes": start,
                        "intersection_end_bytes": end,
                        "intersection_nbytes": end - start,
                    }
                )
    overlaps.sort(
        key=lambda row: (
            row["storage_id"],
            row["left_path"],
            row["right_path"],
            row["intersection_start_bytes"],
        )
    )
    return overlaps


def ownership_ledger(
    *,
    shared_inventory: Sequence[Mapping[str, Any]],
    request_inventories: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Split every request's tensors into shared-with-the-entry and private.

    ``shared_inventory`` is the immutable document view owned by the entry.
    A request tensor is *shared* when its storage id appears in that inventory
    and *private* otherwise.  The ledger additionally reports, per ordered
    request pair, the byte ranges on which the two requests' **private**
    tensors intersect; a non-empty intersection is an ownership violation.

    Shared/private byte totals deduplicate by storage id, so a request that
    holds two views of one shared storage is charged that storage once.
    """

    shared_rows = normalize_inventory(shared_inventory)
    shared_ids = {row["storage_id"] for row in shared_rows}
    per_request: dict[str, dict[str, Any]] = {}
    private_rows_by_request: dict[str, list[dict[str, Any]]] = {}
    for request_id in sorted(request_inventories):
        rows = normalize_inventory(request_inventories[request_id])
        shared_part = [row for row in rows if row["storage_id"] in shared_ids]
        private_part = [row for row in rows if row["storage_id"] not in shared_ids]
        private_rows_by_request[request_id] = private_part
        per_request[request_id] = {
            "tensor_count": len(rows),
            "shared_tensor_count": len(shared_part),
            "private_tensor_count": len(private_part),
            "shared_nbytes": unique_storage_nbytes(shared_part),
            "private_nbytes": unique_storage_nbytes(private_part),
            "shared_storage_ids": sorted({row["storage_id"] for row in shared_part}),
            "private_storage_ids": sorted({row["storage_id"] for row in private_part}),
        }

    pair_rows: list[dict[str, Any]] = []
    request_ids = sorted(private_rows_by_request)
    for index, left_id in enumerate(request_ids):
        for right_id in request_ids[index + 1 :]:
            overlap = range_overlaps(
                private_rows_by_request[left_id], private_rows_by_request[right_id]
            )
            pair_rows.append(
                {
                    "left_request": left_id,
                    "right_request": right_id,
                    "overlap_ranges": overlap,
                    "disjoint": not overlap,
                }
            )
    private_union_ids: set[str] = set()
    for rows in private_rows_by_request.values():
        private_union_ids |= {row["storage_id"] for row in rows}
    return {
        "predicate_id": "ALL_MUTABLE_CACHE_STORAGE_PAIRWISE_DISJOINT",
        "passed": all(row["disjoint"] for row in pair_rows) and bool(pair_rows),
        "non_vacuous": bool(pair_rows),
        "request_count": len(request_ids),
        "shared_entry_nbytes": unique_storage_nbytes(shared_rows),
        "shared_entry_tensor_count": len(shared_rows),
        "per_request": per_request,
        "pairwise": pair_rows,
        "pairwise_comparison_count": len(pair_rows),
        "total_private_nbytes": sum(
            row["private_nbytes"] for row in per_request.values()
        ),
        "distinct_private_storage_count": len(private_union_ids),
        "semantic": (
            "shared = storage id present in the entry's immutable document "
            "view; private = every other storage id; totals deduplicate by "
            "storage id exactly as cache_nbytes does"
        ),
    }


def sharing_efficiency(ledger: Mapping[str, Any]) -> dict[str, Any]:
    """How much the shared view saves relative to N private copies.

    ``copies_avoided_nbytes`` is what ``N`` private materializations would have
    charged for the shared portion minus what one shared view actually charges.
    With ``N = 1`` this is zero by construction, which is the correct answer:
    sharing across one request saves nothing.
    """

    request_count = int(ledger["request_count"])
    shared = int(ledger["shared_entry_nbytes"])
    private_total = int(ledger["total_private_nbytes"])
    resident = shared + private_total
    n_private_equivalent = request_count * shared + private_total
    return {
        "request_count": request_count,
        "shared_entry_nbytes": shared,
        "total_private_nbytes": private_total,
        "resident_nbytes": resident,
        "n_private_copies_equivalent_nbytes": n_private_equivalent,
        "copies_avoided_nbytes": n_private_equivalent - resident,
        "resident_ratio_vs_private_copies": (
            n_private_equivalent / resident if resident else None
        ),
    }


# ---------------------------------------------------------------------------
# transient working set, both arms
# ---------------------------------------------------------------------------


def working_set_row(
    *,
    arm: str,
    request_count: int,
    entry_retained_nbytes: int,
    shared_view_nbytes: int,
    per_request_materialized_nbytes: Sequence[int],
    per_request_steady_resident_nbytes: Sequence[int],
    measured_baseline_allocated_nbytes: int,
    measured_peak_allocated_nbytes: int,
    measured_steady_allocated_nbytes: int,
) -> dict[str, Any]:
    """First-class transient working-set fields for one arm at one N.

    ``entry_retained_nbytes`` is the durable store the paper already reports.
    ``shared_view_nbytes`` is the one dequantized document view; it is 0 for an
    arm that materializes privately and for the full-prefix arm, which has
    nothing to dequantize.  ``per_request_materialized_nbytes`` is what each
    request materializes for itself.

    The affine resident model this implies is
    ``intercept = entry_retained + shared_view`` and
    ``slope = mean(per_request_steady_resident)``.  The model is reported
    beside the measured allocator numbers; it never replaces them.
    """

    _require(request_count >= 1, "request_count must be at least 1")
    _require(
        len(per_request_materialized_nbytes) == request_count,
        "per-request materialized bytes must have one entry per request",
    )
    _require(
        len(per_request_steady_resident_nbytes) == request_count,
        "per-request steady resident bytes must have one entry per request",
    )
    materialized = [int(value) for value in per_request_materialized_nbytes]
    steady = [int(value) for value in per_request_steady_resident_nbytes]
    intercept = int(entry_retained_nbytes) + int(shared_view_nbytes)
    slope = statistics.mean(steady) if steady else 0.0
    return {
        "arm": arm,
        "request_count": request_count,
        # --- retained state, the quantity the paper already reports ---------
        "entry_retained_nbytes": int(entry_retained_nbytes),
        # --- transient working set, new -------------------------------------
        "shared_dequantized_view_nbytes": int(shared_view_nbytes),
        "per_request_materialized_nbytes": materialized,
        "transient_materialized_nbytes_total": sum(materialized),
        "transient_materialized_nbytes_max": max(materialized) if materialized else 0,
        "transient_materialized_nbytes_mean": (
            statistics.mean(materialized) if materialized else 0.0
        ),
        "peak_transient_allocation_nbytes": max(
            int(measured_peak_allocated_nbytes)
            - int(measured_baseline_allocated_nbytes),
            0,
        ),
        # --- steady state ----------------------------------------------------
        "per_request_steady_resident_nbytes": steady,
        "steady_state_resident_nbytes": int(measured_steady_allocated_nbytes),
        "steady_state_resident_delta_nbytes": max(
            int(measured_steady_allocated_nbytes)
            - int(measured_baseline_allocated_nbytes),
            0,
        ),
        "measured_baseline_allocated_nbytes": int(measured_baseline_allocated_nbytes),
        "measured_peak_allocated_nbytes": int(measured_peak_allocated_nbytes),
        "measured_steady_allocated_nbytes": int(measured_steady_allocated_nbytes),
        # --- the affine model those fields imply -----------------------------
        "resident_model": {
            "intercept_nbytes": intercept,
            "slope_nbytes_per_request": slope,
            "semantic": (
                "modelled resident bytes = intercept + slope * N; intercept is "
                "the retained entry plus one shared dequantized view, slope is "
                "the mean per-request steady resident state"
            ),
        },
        "modelled_resident_nbytes": intercept + slope * request_count,
        "measurement_semantic": (
            "materialized bytes are the sum of unique tensor storages a request "
            "brings into existence for itself; allocator numbers are "
            "torch.cuda counters and include framework overhead the byte "
            "ledger deliberately excludes"
        ),
    }


def resident_bytes_at_n(
    *, intercept_nbytes: float, slope_nbytes_per_request: float, request_count: int
) -> float:
    _require(request_count >= 0, "request_count must be non-negative")
    return float(intercept_nbytes) + float(slope_nbytes_per_request) * request_count


def crossover_request_count(
    *,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    max_request_count: int = 4096,
) -> dict[str, Any]:
    """Smallest integer N>=1 at which ``left`` resident exceeds ``right``.

    Returns ``None`` for ``crossover_request_count`` when the left arm never
    exceeds the right within ``max_request_count``.  This is the arithmetic the
    meta-review asked for: whether the Read path's transient term makes the
    split lose at high concurrency, answered from measured slopes rather than
    asserted.  It is a property of the fitted affine model only.
    """

    left_intercept = float(left["intercept_nbytes"])
    left_slope = float(left["slope_nbytes_per_request"])
    right_intercept = float(right["intercept_nbytes"])
    right_slope = float(right["slope_nbytes_per_request"])
    crossover: int | None = None
    for count in range(1, max_request_count + 1):
        left_bytes = left_intercept + left_slope * count
        right_bytes = right_intercept + right_slope * count
        if left_bytes > right_bytes:
            crossover = count
            break
    return {
        "left_intercept_nbytes": left_intercept,
        "left_slope_nbytes_per_request": left_slope,
        "right_intercept_nbytes": right_intercept,
        "right_slope_nbytes_per_request": right_slope,
        "smaller_intercept": "left" if left_intercept < right_intercept else "right",
        "smaller_slope": "left" if left_slope < right_slope else "right",
        "crossover_request_count": crossover,
        "searched_up_to": max_request_count,
        "semantic": (
            "smallest integer N in [1, searched_up_to] with left resident > "
            "right resident under both fitted affine models; null means no "
            "crossover was found in that range"
        ),
    }


# ---------------------------------------------------------------------------
# semantic equivalence over emitted token ids
# ---------------------------------------------------------------------------


def compare_token_traces(
    *,
    reference: Mapping[str, Sequence[int]],
    candidate: Mapping[str, Sequence[int]],
    reference_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    """Token-for-token comparison of two request-keyed generation traces.

    Every discrepancy is recorded with its request id, the first differing
    step, and both sequences.  A request present in one mapping and absent from
    the other is a discrepancy, not a skipped comparison.
    """

    request_ids = sorted(set(reference) | set(candidate))
    _require(bool(request_ids), "token trace comparison needs at least one request")
    rows: list[dict[str, Any]] = []
    for request_id in request_ids:
        left = reference.get(request_id)
        right = candidate.get(request_id)
        if left is None or right is None:
            rows.append(
                {
                    "request_id": request_id,
                    "present_in_reference": left is not None,
                    "present_in_candidate": right is not None,
                    "identical": False,
                    "first_divergence_step": None,
                    "reference_token_ids": list(left) if left is not None else None,
                    "candidate_token_ids": list(right) if right is not None else None,
                }
            )
            continue
        left_list = [int(value) for value in left]
        right_list = [int(value) for value in right]
        divergence = next(
            (
                step
                for step, (a, b) in enumerate(zip(left_list, right_list))
                if a != b
            ),
            None,
        )
        if divergence is None and len(left_list) != len(right_list):
            divergence = min(len(left_list), len(right_list))
        rows.append(
            {
                "request_id": request_id,
                "present_in_reference": True,
                "present_in_candidate": True,
                "identical": left_list == right_list,
                "first_divergence_step": divergence,
                "reference_token_count": len(left_list),
                "candidate_token_count": len(right_list),
                "reference_token_ids": left_list,
                "candidate_token_ids": right_list,
            }
        )
    discrepancies = [row for row in rows if not row["identical"]]
    return {
        "reference_label": reference_label,
        "candidate_label": candidate_label,
        "compared_request_count": len(rows),
        "identical_request_count": len(rows) - len(discrepancies),
        "token_sequences_identical": not discrepancies,
        "discrepancies": discrepancies,
        "rows": rows,
    }


def cross_n_prefix_consistency(
    traces_by_n: Mapping[int, Mapping[str, Sequence[int]]],
    *,
    prefix_request_id: str,
) -> dict[str, Any]:
    """The first request must emit the same tokens at every declared N.

    ``traces_by_n`` maps a fanout to that run's request-keyed token traces.
    The predicate is non-vacuous only when at least two fanouts are present;
    with one fanout the result is reported as vacuous and does not pass.
    """

    fanouts = sorted(int(value) for value in traces_by_n)
    _require(bool(fanouts), "cross-N consistency needs at least one fanout")
    sequences: dict[int, list[int] | None] = {}
    for fanout in fanouts:
        trace = traces_by_n[fanout].get(prefix_request_id)
        sequences[fanout] = None if trace is None else [int(v) for v in trace]
    present = [fanout for fanout in fanouts if sequences[fanout] is not None]
    non_vacuous = len(present) >= 2
    baseline = sequences[present[0]] if present else None
    mismatches = [
        {
            "fanout": fanout,
            "token_ids": sequences[fanout],
            "baseline_token_ids": baseline,
        }
        for fanout in present[1:]
        if sequences[fanout] != baseline
    ]
    missing = [fanout for fanout in fanouts if sequences[fanout] is None]
    return {
        "predicate_id": "FIRST_REQUEST_PREFIX_INVARIANT_ACROSS_N",
        "prefix_request_id": prefix_request_id,
        "fanouts": fanouts,
        "non_vacuous": non_vacuous,
        "missing_fanouts": missing,
        "mismatches": mismatches,
        "passed": non_vacuous and not mismatches and not missing,
    }


# ---------------------------------------------------------------------------
# row validation and shard summarization
# ---------------------------------------------------------------------------

REQUIRED_ROW_FIELDS = (
    "arm",
    "fork_mode",
    "request_count",
    "workload_id",
    "entry_retained_nbytes",
    "shared_dequantized_view_nbytes",
    "per_request_materialized_nbytes",
    "transient_materialized_nbytes_total",
    "peak_transient_allocation_nbytes",
    "steady_state_resident_nbytes",
    "per_request_steady_resident_nbytes",
    "resident_model",
    "ownership_ledger",
    "semantic_equivalence",
)


def validate_multifork_row(row: Mapping[str, Any]) -> list[str]:
    """Return every problem with one emitted row; never raise on a bad row.

    This validator checks exactly what the shard schema promises: that the
    required fields are present, that the byte fields are non-negative
    integers, that the per-request lists have one entry per request, and that
    the declared fork mode and arm are known.  It does **not** assert that the
    shared mode saved bytes, that any predicate passed, or that the sequences
    matched -- those are results, not schema properties, and a run that fails
    them must still emit a well-formed row.
    """

    problems: list[str] = []
    for field in REQUIRED_ROW_FIELDS:
        if field not in row:
            problems.append(f"missing field: {field}")
    if problems:
        return problems
    if row["arm"] not in MULTIFORK_ARMS:
        problems.append(f"unknown arm: {row['arm']}")
    if row["fork_mode"] not in FORK_MODES:
        problems.append(f"unknown fork mode: {row['fork_mode']}")
    request_count = row["request_count"]
    if type(request_count) is not int or request_count < 1:
        problems.append("request_count must be a positive integer")
        return problems
    for field in (
        "entry_retained_nbytes",
        "shared_dequantized_view_nbytes",
        "transient_materialized_nbytes_total",
        "peak_transient_allocation_nbytes",
        "steady_state_resident_nbytes",
    ):
        value = row[field]
        if type(value) is not int or value < 0:
            problems.append(f"{field} must be a non-negative integer")
    for field in (
        "per_request_materialized_nbytes",
        "per_request_steady_resident_nbytes",
    ):
        values = row[field]
        if not isinstance(values, (list, tuple)):
            problems.append(f"{field} must be a list")
            continue
        if len(values) != request_count:
            problems.append(
                f"{field} has {len(values)} entries for {request_count} requests"
            )
        if any(type(value) is not int or value < 0 for value in values):
            problems.append(f"{field} must contain non-negative integers")
    model = row["resident_model"]
    if not isinstance(model, Mapping):
        problems.append("resident_model must be a mapping")
    else:
        for field in ("intercept_nbytes", "slope_nbytes_per_request"):
            if field not in model:
                problems.append(f"resident_model is missing {field}")
            elif not isinstance(model[field], (int, float)) or model[field] < 0:
                problems.append(f"resident_model.{field} must be non-negative")
    ledger = row["ownership_ledger"]
    if not isinstance(ledger, Mapping):
        problems.append("ownership_ledger must be a mapping")
    elif ledger.get("request_count") != request_count:
        problems.append("ownership_ledger request_count disagrees with the row")
    equivalence = row["semantic_equivalence"]
    if not isinstance(equivalence, Mapping):
        problems.append("semantic_equivalence must be a mapping")
    elif "token_sequences_identical" not in equivalence:
        problems.append("semantic_equivalence is missing token_sequences_identical")
    return problems


def summarize_multifork_rows(
    rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Aggregate emitted rows by (arm, fork mode, request count).

    Discrepant rows are counted, not dropped.  A cell in which some request
    diverged is reported with ``semantic_equivalence_failures > 0`` and its
    byte statistics are still summarized, because the byte ledger is valid
    evidence about ownership even when the semantic gate fails.
    """

    grouped: dict[tuple[str, str, int], list[Mapping[str, Any]]] = {}
    for row in rows:
        key = (row["arm"], row["fork_mode"], int(row["request_count"]))
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for (arm, fork_mode, request_count), cell in sorted(grouped.items()):
        failures = sum(
            0
            if bool(row["semantic_equivalence"].get("token_sequences_identical"))
            else 1
            for row in cell
        )
        summaries.append(
            {
                "arm": arm,
                "fork_mode": fork_mode,
                "request_count": request_count,
                "row_count": len(cell),
                "workload_ids": sorted({row["workload_id"] for row in cell}),
                "entry_retained_nbytes_median": round(
                    statistics.median(
                        [int(row["entry_retained_nbytes"]) for row in cell]
                    )
                ),
                "shared_dequantized_view_nbytes_median": round(
                    statistics.median(
                        [int(row["shared_dequantized_view_nbytes"]) for row in cell]
                    )
                ),
                "transient_materialized_nbytes_total_median": round(
                    statistics.median(
                        [int(row["transient_materialized_nbytes_total"]) for row in cell]
                    )
                ),
                "peak_transient_allocation_nbytes_median": round(
                    statistics.median(
                        [int(row["peak_transient_allocation_nbytes"]) for row in cell]
                    )
                ),
                "steady_state_resident_nbytes_median": round(
                    statistics.median(
                        [int(row["steady_state_resident_nbytes"]) for row in cell]
                    )
                ),
                "resident_intercept_nbytes_median": statistics.median(
                    [float(row["resident_model"]["intercept_nbytes"]) for row in cell]
                ),
                "resident_slope_nbytes_per_request_median": statistics.median(
                    [
                        float(row["resident_model"]["slope_nbytes_per_request"])
                        for row in cell
                    ]
                ),
                "semantic_equivalence_failures": failures,
                "semantic_equivalence_all_identical": failures == 0,
            }
        )
    return summaries


def format_mib(value: float) -> float:
    """Bytes to MiB, rounded to three decimals, for human-readable tables."""

    return round(float(value) / 2**20, 3)


def fanout_plan(
    fanouts: Sequence[int], *, require_multifork: bool = True
) -> list[int]:
    """Validate and normalize the requested request fanouts.

    ``require_multifork`` enforces what this experiment exists to measure: at
    least one fanout greater than one.  N=1 is retained because it is the
    reference the shared arm is compared against, and because cross-N
    consistency needs at least two distinct fanouts.
    """

    values = sorted({int(value) for value in fanouts})
    _require(bool(values), "at least one fanout is required")
    _require(all(value >= 1 for value in values), "fanouts must be >= 1")
    if require_multifork:
        _require(
            any(value > 1 for value in values),
            "the multifork experiment requires at least one fanout greater than 1",
        )
    return values


def request_ids(count: int) -> list[str]:
    _require(count >= 1, "request count must be positive")
    width = max(2, len(str(count - 1)))
    return [f"r{index:0{width}d}" for index in range(count)]


def ceil_div(numerator: int, denominator: int) -> int:
    _require(denominator > 0, "denominator must be positive")
    return -(-int(numerator) // int(denominator))


__all__ = [
    "AGGREGATE_SCHEMA",
    "COVERAGE_STATES",
    "FORK_MODES",
    "MANDATORY_SLOTS",
    "MULTIFORK_ARMS",
    "MULTIFORK_TARGET_CONTRACT",
    "MultiforkAccountingError",
    "PROTOCOL",
    "REBIND_POLICIES",
    "REQUIRED_ROW_FIELDS",
    "SHARD_SCHEMA",
    "TAIL_POLICIES",
    "TARGET_EXACT_MISSINGNESS",
    "TARGET_SCOPE_NOTES",
    "VALID_STATUSES",
    "build_multifork_target_rows",
    "ceil_div",
    "compare_token_traces",
    "contract_summary",
    "cross_n_prefix_consistency",
    "crossover_request_count",
    "evaluate_slot_coverage",
    "fanout_plan",
    "format_mib",
    "normalize_inventory",
    "ownership_ledger",
    "packed_entry_obligation_names",
    "range_overlaps",
    "request_ids",
    "resident_bytes_at_n",
    "sharing_efficiency",
    "summarize_multifork_rows",
    "target_names",
    "unique_storage_nbytes",
    "validate_multifork_row",
    "working_set_row",
]
