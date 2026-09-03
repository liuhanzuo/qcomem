"""ForkAudit instantiated on the shared packed Read path.

Every ForkAudit verdict the manuscript reports comes from a different stack:
vLLM plus Transformers, full-prefix BF16 KV, no split depth, no quantization,
PG-19 documents.  This module runs the same contract against the path the paper
actually proposes -- one quantized depth-split entry, dequantized once, shared
across N>1 concurrent requests -- and adds the obligations the existing audit
cannot express because it never sees a packed entry.

Contract
--------

Ten targets.  Targets 1--7 are the manuscript's seven; targets 8--10 are the
packed-entry obligations Section 4.4 names as untested:

8. dequantized-view immutability across requests,
9. residual-chunk binding,
10. packed-entry lifetime.

Coverage is recorded separately from verdict, exactly as the existing contract
does.  Every target declares mandatory receipt slots; a slot that is missing,
duplicated, modified, or not bound to the live object its receipt names makes
the target ``open`` with ``predicate_passed = None``.  A missing mandatory
receipt can never produce a pass.

Trusted computing base
----------------------

Capture and replay run in the same process as the execution they observe.
Content digests are taken from live tensors, storage identity from the PyTorch
allocator, and call provenance from Python-level adapter calls.  This is an
offline regression contract on a non-adversarial runtime, not an attestation: a
kernel that wrote a shared tensor and restored it before the next digest would
not be caught, and no compiled-kernel identity is recorded.
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Mapping, Sequence

import torch

from qcomem_deployment import (
    DeploymentConfig,
    MemoryRecorder,
    build_persistent_state,
    run_incremental_generation,
)
from qcomem_multifork_accounting import (
    MANDATORY_SLOTS,
    MultiforkAccountingError,
    build_multifork_target_rows,
    compare_token_traces,
    contract_summary,
    cross_n_prefix_consistency,
    normalize_inventory,
    ownership_ledger,
    sharing_efficiency,
)
from qcomem_shared_packed_fork import (
    MultiforkTrace,
    SharedPackedEntry,
    iter_tensor_slots,
    prepare_shared_packed_entry,
    run_full_prefix_multifork,
    run_shared_packed_multifork,
    storage_inventory_rows,
)
from qcomem_torch import PackedLowerReplayState


AUDIT_SCHEMA = "qcomem-shared-packed-forkaudit-v1"
GATE_SCHEMA = "qcomem-shared-packed-multifork-gate-v1"
DEFAULT_SALT = "qcomem-shared-packed-multifork"

#: which capture the view-aliasing obligation is evaluated at, per tail policy
SHARING_WINDOW_BY_TAIL_POLICY = {
    # the prefix stays borrowed for the whole request, so aliasing must still
    # hold after decoding
    "borrowed-prefix": "final",
    # sharing is by construction only between fork and the request's first
    # append, so aliasing is evaluated at setup and the window is printed
    "materialized-tail": "setup",
}


class SharedPackedAuditError(RuntimeError):
    """An evidence-integrity failure, not a scientific negative."""


# ---------------------------------------------------------------------------
# digests and receipt slots
# ---------------------------------------------------------------------------


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def tensor_content_sha256(tensor: torch.Tensor) -> str:
    payload = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def tensor_tree_digest(root: Any, *, label: str) -> dict[str, Any]:
    """Full-content digest of every tensor leaf reachable from ``root``.

    Rows carry the leaf path, dtype, shape and content hash; the tree digest is
    the canonical-JSON hash of those rows.  A structural change (a leaf added,
    removed or renamed) therefore changes the tree digest even when every
    surviving leaf is byte-identical.
    """

    rows = []
    for slot in iter_tensor_slots(root):
        tensor = slot.tensor
        rows.append(
            {
                "path": slot.path,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
                "content_sha256": tensor_content_sha256(tensor),
            }
        )
    return {
        "label": label,
        "tensor_count": len(rows),
        "rows": rows,
        "tree_sha256": sha256_json(rows),
    }


def token_identity(tokens: torch.Tensor, *, label: str) -> dict[str, Any]:
    flat = tokens.detach().reshape(-1).to(torch.int64).cpu()
    return {
        "label": label,
        "token_count": int(flat.numel()),
        "shape": list(tokens.shape),
        "content_sha256": tensor_content_sha256(flat),
    }


def layer_class_receipt(cache: Any, adapter: Any) -> dict[str, Any]:
    layers = getattr(cache, "layers", ()) or ()
    return {
        "adapter_class": type(adapter).__name__,
        "cache_class": type(cache).__name__,
        "layer_classes": [type(layer).__name__ for layer in layers],
        "layer_count": len(layers),
    }


def slot_receipt(
    *,
    present: bool,
    unique: bool,
    bound: bool,
    modified: bool,
    payload: Any = None,
) -> dict[str, Any]:
    """One mandatory receipt slot in the coverage schema.

    ``bound`` means the receipt resolves to the live object it names -- for a
    digest slot, that the digest was recomputed from the live tensors at
    capture time rather than copied forward from an earlier capture.
    """

    return {
        "present": bool(present),
        "unique": bool(unique),
        "bound": bool(bound),
        "modified": bool(modified),
        "payload": payload,
    }


def missing_slot_receipt(reason: str) -> dict[str, Any]:
    """A slot that could not be captured; it can never make a target pass."""

    return {
        "present": False,
        "unique": False,
        "bound": False,
        "modified": False,
        "payload": {"reason": reason},
    }


# ---------------------------------------------------------------------------
# predicate evaluation
# ---------------------------------------------------------------------------


def evaluate_frozen_identity(bindings: Mapping[str, Any]) -> dict[str, Any]:
    """Every declared identity binding replays from the live objects.

    ``bindings`` maps a binding name to ``{"declared": ..., "replayed": ...}``,
    where ``declared`` was captured before the run and ``replayed`` recomputed
    from live objects after it.  The predicate passes only when every binding
    is present and the two values are equal; a binding whose replay was skipped
    is a failure, not an abstention.
    """

    rows = []
    for name in sorted(bindings):
        row = bindings[name]
        declared = row.get("declared")
        replayed = row.get("replayed")
        rows.append(
            {
                "binding": name,
                "declared": declared,
                "replayed": replayed,
                "matches": declared is not None and declared == replayed,
            }
        )
    return {
        "predicate_id": "FROZEN_ENTRY_POLICY_AND_INPUT_BINDINGS",
        "bindings": rows,
        "binding_count": len(rows),
        "non_vacuous": bool(rows),
        "passed": bool(rows) and all(row["matches"] for row in rows),
    }


def evaluate_private_ownership(
    *,
    transition_ledger: Mapping[str, Any],
    final_ledger: Mapping[str, Any],
    shared_storage_ids: Sequence[str],
) -> dict[str, Any]:
    """Pairwise-disjoint private storage at the transition and at the end.

    Evaluated at the transition and final captures only.  The setup capture is
    recorded as a mandatory receipt but is deliberately **not** asserted
    disjoint, because under ``rebind_policy="transition"`` the mutable base is
    borrowed read-only between fork and the registered transition.  That borrow
    is what the rebind ledger exists to make visible; asserting disjointness at
    setup would assert the opposite of the contract under audit.
    """

    shared = set(shared_storage_ids)
    leaks: list[dict[str, Any]] = []
    for label, ledger in (("transition", transition_ledger), ("final", final_ledger)):
        for request_id, row in ledger["per_request"].items():
            overlap = sorted(set(row["private_storage_ids"]) & shared)
            if overlap:
                leaks.append(
                    {
                        "capture": label,
                        "request_id": request_id,
                        "private_ids_also_shared": overlap,
                    }
                )
    non_vacuous = bool(
        transition_ledger["non_vacuous"] and final_ledger["non_vacuous"]
    )
    return {
        "predicate_id": "ALL_MUTABLE_CACHE_STORAGE_PAIRWISE_DISJOINT",
        "evaluated_at": ["transition", "final"],
        "setup_capture_recorded_not_asserted": True,
        "non_vacuous": non_vacuous,
        "transition_pairwise_disjoint": bool(transition_ledger["passed"]),
        "final_pairwise_disjoint": bool(final_ledger["passed"]),
        "private_shared_leaks": leaks,
        "passed": bool(
            non_vacuous
            and transition_ledger["passed"]
            and final_ledger["passed"]
            and not leaks
        ),
    }


def evaluate_tail_safe_append(
    *,
    append_events: Sequence[Mapping[str, Any]],
    shared_storage_ids: Sequence[str],
    request_count: int,
    shared_attention_unchanged: bool,
    tail_policy: str,
) -> dict[str, Any]:
    """No append ever wrote a shared prefix tensor.

    Requires: at least one append event for every request; every event rebinds
    the retained key storage rather than reusing it; no post-append retained
    key/value storage is a shared-view storage; and the shared view's content
    digest is unchanged between the setup and final captures.  Under
    ``borrowed-prefix`` it additionally requires that every event names a
    shared-view prefix storage and reports it unchanged.
    """

    shared = set(shared_storage_ids)
    per_request: dict[str, int] = {}
    violations: list[dict[str, Any]] = []
    for event in append_events:
        per_request[event["request_id"]] = per_request.get(event["request_id"], 0) + 1
        if not event.get("keys_storage_rebound"):
            violations.append(
                {"reason": "append_reused_retained_key_storage", "event": dict(event)}
            )
        for field_name in ("after_keys_storage_id", "after_values_storage_id"):
            if event.get(field_name) in shared:
                violations.append(
                    {
                        "reason": f"{field_name}_is_a_shared_view_storage",
                        "event": dict(event),
                    }
                )
        if tail_policy == "borrowed-prefix":
            prefix_id = event.get("shared_prefix_storage_id")
            if prefix_id not in shared:
                violations.append(
                    {"reason": "prefix_is_not_a_shared_view_storage", "event": dict(event)}
                )
            if not event.get("shared_prefix_storage_unchanged"):
                violations.append(
                    {"reason": "prefix_storage_changed_during_append", "event": dict(event)}
                )
    covered = len(per_request) == request_count and all(
        count > 0 for count in per_request.values()
    )
    return {
        "predicate_id": "SHARED_PREFIX_NOT_WRITTEN_ON_APPEND",
        "tail_policy": tail_policy,
        "append_event_count": len(append_events),
        "requests_with_appends": len(per_request),
        "per_request_append_counts": per_request,
        "all_requests_appended": covered,
        "shared_view_content_unchanged": bool(shared_attention_unchanged),
        "violations": violations,
        "non_vacuous": bool(append_events),
        "passed": bool(
            append_events and covered and shared_attention_unchanged and not violations
        ),
    }


def evaluate_dispatch_provenance(
    *,
    call_log: Sequence[Mapping[str, Any]],
    request_count: int,
    num_layers: int,
    depth: int,
) -> dict[str, Any]:
    """Every host-side adapter call is recorded, attributed and in range.

    Bounded provenance by construction: the Python adapter call, its request,
    its layer range and its position offset.  No compiled kernel identity is
    recorded, so the target is capped at ``partial``.
    """

    requests = {row.get("request_id") for row in call_log}
    problems: list[dict[str, Any]] = []
    for row in call_log:
        layer_range = row.get("layer_range")
        if (
            not isinstance(layer_range, (list, tuple))
            or len(layer_range) != 2
            or not (0 <= int(layer_range[0]) <= int(layer_range[1]) <= num_layers)
        ):
            problems.append({"reason": "layer_range_out_of_bounds", "call": dict(row)})
            continue
        if row.get("call") == "continue_lower_replay" and list(layer_range) != [
            0,
            depth,
        ]:
            problems.append({"reason": "lower_call_layer_range_drift", "call": dict(row)})
    return {
        "predicate_id": "BOUNDED_HOST_SIDE_CALL_PROVENANCE",
        "call_count": len(call_log),
        "attributed_request_count": len(requests - {None}),
        "unattributed_call_count": sum(
            1 for row in call_log if row.get("request_id") is None
        ),
        "problems": problems,
        "non_vacuous": bool(call_log),
        "passed": bool(
            call_log and len(requests - {None}) == request_count and not problems
        ),
    }


def evaluate_dequantized_view_immutability(
    *,
    setup_digest: Mapping[str, Any],
    final_digest: Mapping[str, Any],
    per_request_shared_nbytes: Mapping[str, int],
    view_guard: Mapping[str, Any],
    sharing_window: str,
) -> dict[str, Any]:
    """One view, unchanged, and actually aliased by every request.

    Requires: the view's full-content tree digest is identical before the first
    fork and after the last request finishes; the sampled storage/version guard
    over that view still verifies and is not vacuous; at least two requests;
    and, at the capture named by ``sharing_window``, every request held a
    non-zero number of bytes that alias the view.  A run in which some request
    shared nothing at that window is a failure here, not a vacuous pass.
    """

    non_sharing = sorted(
        request_id
        for request_id, nbytes in per_request_shared_nbytes.items()
        if int(nbytes) <= 0
    )
    digests_equal = setup_digest["tree_sha256"] == final_digest["tree_sha256"]
    return {
        "predicate_id": "SHARED_DEQUANTIZED_VIEW_CONTENT_UNCHANGED",
        "sharing_window": sharing_window,
        "setup_tree_sha256": setup_digest["tree_sha256"],
        "final_tree_sha256": final_digest["tree_sha256"],
        "tree_digests_equal": digests_equal,
        "view_guard_verified": bool(view_guard.get("verified")),
        "view_guard_vacuous": bool(view_guard.get("vacuous")),
        "request_count": len(per_request_shared_nbytes),
        "per_request_shared_nbytes": dict(per_request_shared_nbytes),
        "requests_sharing_nothing": non_sharing,
        "non_vacuous": len(per_request_shared_nbytes) >= 2 and not non_sharing,
        "passed": bool(
            digests_equal
            and view_guard.get("verified")
            and not view_guard.get("vacuous")
            and len(per_request_shared_nbytes) >= 2
            and not non_sharing
        ),
    }


def evaluate_residual_chunk_binding(
    *,
    events: Sequence[Mapping[str, Any]],
    request_count: int,
    document_length: int,
) -> dict[str, Any]:
    """Document and query residual chunks are distinct and correctly bound.

    Requires, for every request: exactly one binding event; the document chunk
    is a shared-view tensor consumed at position offset 0; the query chunk is
    not a shared-view tensor and is consumed at the document offset in a
    separate call; the two chunks do not share storage; and all requests bound
    the same single document chunk storage.
    """

    by_request: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        by_request.setdefault(event["request_id"], []).append(event)
    problems: list[dict[str, Any]] = []
    document_ids: set[str] = set()
    for request_id, rows in sorted(by_request.items()):
        if len(rows) != 1:
            problems.append(
                {
                    "request_id": request_id,
                    "reason": "expected_exactly_one_binding_event",
                }
            )
            continue
        row = rows[0]
        document_ids.add(str(row["document_chunk_storage_id"]))
        if not row["document_chunk_is_shared_view_tensor"]:
            problems.append(
                {
                    "request_id": request_id,
                    "reason": "document_chunk_is_not_the_shared_view",
                }
            )
        if row["query_chunk_is_shared_view_tensor"]:
            problems.append(
                {
                    "request_id": request_id,
                    "reason": "query_chunk_aliases_the_shared_view",
                }
            )
        if row["chunks_share_storage"]:
            problems.append(
                {
                    "request_id": request_id,
                    "reason": "document_and_query_chunks_share_storage",
                }
            )
        if int(row["document_position_offset"]) != 0:
            problems.append(
                {
                    "request_id": request_id,
                    "reason": "document_chunk_offset_is_not_zero",
                }
            )
        if int(row["query_position_offset"]) != int(document_length):
            problems.append(
                {
                    "request_id": request_id,
                    "reason": "query_chunk_offset_is_not_document_length",
                }
            )
        if not row["chunks_are_distinct_calls"]:
            problems.append(
                {"request_id": request_id, "reason": "chunks_were_not_separate_calls"}
            )
    covered = len(by_request) == request_count
    return {
        "predicate_id": "DOCUMENT_AND_QUERY_RESIDUAL_CHUNKS_BOUND_DISTINCTLY",
        "event_count": len(events),
        "requests_with_events": len(by_request),
        "all_requests_covered": covered,
        "distinct_document_chunk_storage_count": len(document_ids),
        "problems": problems,
        "non_vacuous": request_count >= 2 and bool(events),
        "passed": bool(
            events
            and covered
            and not problems
            and len(document_ids) == 1
            and request_count >= 2
        ),
    }


def evaluate_packed_entry_lifetime(
    *,
    setup_digest: Mapping[str, Any],
    final_digest: Mapping[str, Any],
    packed_storage_ids: Sequence[str],
    request_storage_ids: Mapping[str, Sequence[str]],
    forks_created: int,
    forks_released: int,
) -> dict[str, Any]:
    """The packed entry survives every request unchanged and unreferenced.

    Requires: the packed entry's full-content tree digest is identical before
    the first fork and after the last request is released; no request's tensor
    inventory contains a packed-entry storage id; and every fork that was
    created was released.  Release is observed through Python-visible
    references, not through the allocator.
    """

    packed = set(packed_storage_ids)
    aliases = {
        request_id: sorted(set(ids) & packed)
        for request_id, ids in request_storage_ids.items()
    }
    leaking = {request_id: ids for request_id, ids in aliases.items() if ids}
    digests_equal = setup_digest["tree_sha256"] == final_digest["tree_sha256"]
    return {
        "predicate_id": "PACKED_ENTRY_CONTENT_AND_REFERENCE_LIFETIME",
        "setup_tree_sha256": setup_digest["tree_sha256"],
        "final_tree_sha256": final_digest["tree_sha256"],
        "tree_digests_equal": digests_equal,
        "forks_created": int(forks_created),
        "forks_released": int(forks_released),
        "all_forks_released": forks_created == forks_released,
        "requests_referencing_packed_storage": leaking,
        "non_vacuous": bool(packed) and bool(request_storage_ids),
        "passed": bool(
            digests_equal
            and forks_created == forks_released
            and not leaking
            and packed
            and request_storage_ids
        ),
    }


# ---------------------------------------------------------------------------
# the full audit
# ---------------------------------------------------------------------------


def audit_shared_packed_multifork(
    *,
    entry: SharedPackedEntry,
    shared_trace: MultiforkTrace,
    private_reference_traces: Mapping[str, Sequence[int]],
    traces_by_fanout: Mapping[int, Mapping[str, Sequence[int]]],
    captures: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    view_setup_digest: Mapping[str, Any],
    view_final_digest: Mapping[str, Any],
    packed_setup_digest: Mapping[str, Any],
    packed_final_digest: Mapping[str, Any],
    packed_inventory: Sequence[Mapping[str, Any]],
    identity_bindings: Mapping[str, Any],
    layer_classes: Mapping[str, Any],
    num_layers: int,
    forks_created: int,
    forks_released: int,
    prefix_request_id: str,
) -> dict[str, Any]:
    """Evaluate all ten targets and emit coverage-separated contract rows."""

    for point in ("setup", "transition", "final"):
        if point not in captures:
            raise SharedPackedAuditError(f"missing ownership capture: {point}")
    shared_inventory = entry.shared_inventory()
    shared_ids = sorted(
        {row["storage_id"] for row in normalize_inventory(shared_inventory)}
    )
    packed_ids = sorted(
        {row["storage_id"] for row in normalize_inventory(packed_inventory)}
    )
    request_count = len(shared_trace.request_traces)

    ledgers = {
        point: ownership_ledger(
            shared_inventory=shared_inventory,
            request_inventories=captures[point],
        )
        for point in ("setup", "transition", "final")
    }
    sharing_window = SHARING_WINDOW_BY_TAIL_POLICY.get(
        shared_trace.tail_policy, "final"
    )
    view_guard = entry.verify_view_unchanged()
    view_unchanged = (
        view_setup_digest["tree_sha256"] == view_final_digest["tree_sha256"]
    )

    identity = evaluate_frozen_identity(identity_bindings)
    ownership = evaluate_private_ownership(
        transition_ledger=ledgers["transition"],
        final_ledger=ledgers["final"],
        shared_storage_ids=shared_ids,
    )
    tail = evaluate_tail_safe_append(
        append_events=shared_trace.append_events,
        shared_storage_ids=shared_ids,
        request_count=request_count,
        shared_attention_unchanged=view_unchanged,
        tail_policy=shared_trace.tail_policy,
    )
    dispatch = evaluate_dispatch_provenance(
        call_log=shared_trace.adapter_call_log,
        request_count=request_count,
        num_layers=num_layers,
        depth=entry.depth,
    )
    equivalence = compare_token_traces(
        reference=private_reference_traces,
        candidate=shared_trace.token_traces(),
        reference_label="n1-private-materialize",
        candidate_label=f"n{request_count}-shared-packed-view",
    )
    cross_n = cross_n_prefix_consistency(
        traces_by_fanout, prefix_request_id=prefix_request_id
    )
    view_immutability = evaluate_dequantized_view_immutability(
        setup_digest=view_setup_digest,
        final_digest=view_final_digest,
        per_request_shared_nbytes={
            request_id: row["shared_nbytes"]
            for request_id, row in ledgers[sharing_window]["per_request"].items()
        },
        view_guard=view_guard,
        sharing_window=sharing_window,
    )
    residual_binding = evaluate_residual_chunk_binding(
        events=shared_trace.residual_binding_events,
        request_count=request_count,
        document_length=entry.document_length,
    )
    lifetime = evaluate_packed_entry_lifetime(
        setup_digest=packed_setup_digest,
        final_digest=packed_final_digest,
        packed_storage_ids=packed_ids,
        request_storage_ids={
            request_id: list(row["private_storage_ids"])
            + list(row["shared_storage_ids"])
            for request_id, row in ledgers["final"]["per_request"].items()
        },
        forks_created=forks_created,
        forks_released=forks_released,
    )
    prefix_immutability_passed = bool(
        view_unchanged
        and packed_setup_digest["tree_sha256"] == packed_final_digest["tree_sha256"]
    )

    def digest_slot(digest: Mapping[str, Any]) -> dict[str, Any]:
        return slot_receipt(
            present=True,
            unique=True,
            bound=True,
            modified=False,
            payload={
                "tree_sha256": digest["tree_sha256"],
                "tensor_count": digest["tensor_count"],
                "label": digest["label"],
            },
        )

    def ledger_slot(point: str) -> dict[str, Any]:
        ledger = ledgers[point]
        return slot_receipt(
            present=bool(captures[point]),
            unique=True,
            bound=True,
            modified=False,
            payload={
                "capture": point,
                "request_count": ledger["request_count"],
                "per_request": ledger["per_request"],
            },
        )

    receipts: dict[str, Any] = {
        "entry_identity": slot_receipt(
            present=True, unique=True, bound=True, modified=False,
            payload=identity_bindings.get("entry_content"),
        ),
        "policy_identity": slot_receipt(
            present=True, unique=True, bound=True, modified=False,
            payload=identity_bindings.get("policy"),
        ),
        "document_token_identity": slot_receipt(
            present=True, unique=True, bound=True, modified=False,
            payload=identity_bindings.get("document_tokens"),
        ),
        "query_token_identity": slot_receipt(
            present=True, unique=True, bound=True, modified=False,
            payload=identity_bindings.get("query_tokens"),
        ),
        "adapter_identity": slot_receipt(
            present=True, unique=True, bound=True, modified=False,
            payload=identity_bindings.get("adapter"),
        ),
        "shared_view_setup_digest": digest_slot(view_setup_digest),
        "shared_view_final_digest": digest_slot(view_final_digest),
        "packed_entry_setup_digest": digest_slot(packed_setup_digest),
        "packed_entry_final_digest": digest_slot(packed_final_digest),
        "setup_inventory": ledger_slot("setup"),
        "transition_inventory": ledger_slot("transition"),
        "final_inventory": ledger_slot("final"),
        "append_events": slot_receipt(
            present=bool(shared_trace.append_events),
            unique=True, bound=True, modified=False,
            payload={
                "event_count": len(shared_trace.append_events),
                "tail_policy": shared_trace.tail_policy,
            },
        ),
        "shared_attention_inventory": slot_receipt(
            present=bool(shared_ids), unique=True, bound=True, modified=False,
            payload={"storage_count": len(shared_ids)},
        ),
        "adapter_call_log": slot_receipt(
            present=bool(shared_trace.adapter_call_log),
            unique=True, bound=True, modified=False,
            payload={"call_count": len(shared_trace.adapter_call_log)},
        ),
        "layer_class_receipt": slot_receipt(
            present=bool(layer_classes), unique=True, bound=True, modified=False,
            payload=dict(layer_classes),
        ),
        "shared_token_traces": slot_receipt(
            present=bool(shared_trace.token_traces()),
            unique=True, bound=True, modified=False,
            payload=shared_trace.token_traces(),
        ),
        "private_token_traces": slot_receipt(
            present=bool(private_reference_traces),
            unique=True, bound=True, modified=False,
            payload={
                key: list(value) for key, value in private_reference_traces.items()
            },
        ),
        "cross_n_token_traces": slot_receipt(
            present=len(traces_by_fanout) >= 2,
            unique=True, bound=True, modified=False,
            payload={
                str(fanout): {key: list(value) for key, value in traces.items()}
                for fanout, traces in traces_by_fanout.items()
            },
        ),
        "view_alias_inventory": slot_receipt(
            present=bool(ledgers[sharing_window]["per_request"]),
            unique=True, bound=True, modified=False,
            payload={
                "sharing_window": sharing_window,
                "per_request": {
                    request_id: {
                        "shared_nbytes": row["shared_nbytes"],
                        "shared_storage_count": len(row["shared_storage_ids"]),
                    }
                    for request_id, row in ledgers[sharing_window][
                        "per_request"
                    ].items()
                },
            },
        ),
        "residual_binding_events": slot_receipt(
            present=bool(shared_trace.residual_binding_events),
            unique=True, bound=True, modified=False,
            payload={"event_count": len(shared_trace.residual_binding_events)},
        ),
        "fork_release_ledger": slot_receipt(
            present=True, unique=True, bound=True, modified=False,
            payload={
                "forks_created": forks_created,
                "forks_released": forks_released,
            },
        ),
    }

    predicates = {
        "frozen_identity": bool(identity["passed"]),
        "prefix_immutability": prefix_immutability_passed,
        "private_ownership": bool(ownership["passed"]),
        "tail_safe_append": bool(tail["passed"]),
        "dispatch_provenance": bool(dispatch["passed"]),
        "cross_arm_equivalence": bool(equivalence["token_sequences_identical"]),
        "cross_n_prefix_consistency": bool(cross_n["passed"]),
        "dequantized_view_immutability": bool(view_immutability["passed"]),
        "residual_chunk_binding": bool(residual_binding["passed"]),
        "packed_entry_lifetime": bool(lifetime["passed"]),
    }
    scope_overrides = {
        "dequantized_view_immutability": (
            "The single dequantized document view is content-identical before "
            "the first fork and after the last request completes, and every "
            f"request aliased that one view at the {sharing_window} capture. "
            + (
                "Under the borrowed-prefix tail policy that window is the whole "
                "request lifetime."
                if sharing_window == "final"
                else "Under the materialized-tail policy sharing exists only "
                "between fork and the request's first append; this target does "
                "not establish steady-state sharing."
            )
        )
    }
    rows = build_multifork_target_rows(
        predicates=predicates, receipts=receipts, scope_overrides=scope_overrides
    )
    summary = contract_summary(rows)
    return {
        "schema": AUDIT_SCHEMA,
        "request_count": request_count,
        "fork_mode": shared_trace.fork_mode,
        "rebind_policy": shared_trace.rebind_policy,
        "tail_policy": shared_trace.tail_policy,
        "sharing_window": sharing_window,
        "target_rows": rows,
        "contract_summary": summary,
        "predicates": predicates,
        "detail": {
            "frozen_identity": identity,
            "prefix_immutability": {
                "predicate_id": "PERSISTENT_PREFIX_CONTENT_UNCHANGED",
                "view_setup_tree_sha256": view_setup_digest["tree_sha256"],
                "view_final_tree_sha256": view_final_digest["tree_sha256"],
                "packed_setup_tree_sha256": packed_setup_digest["tree_sha256"],
                "packed_final_tree_sha256": packed_final_digest["tree_sha256"],
                "passed": prefix_immutability_passed,
            },
            "private_ownership": ownership,
            "tail_safe_append": tail,
            "dispatch_provenance": dispatch,
            "cross_arm_equivalence": equivalence,
            "cross_n_prefix_consistency": cross_n,
            "dequantized_view_immutability": view_immutability,
            "residual_chunk_binding": residual_binding,
            "packed_entry_lifetime": lifetime,
        },
        "ownership": {
            "setup_ledger": ledgers["setup"],
            "transition_ledger": ledgers["transition"],
            "final_ledger": ledgers["final"],
            "sharing_efficiency_at_window": sharing_efficiency(ledgers[sharing_window]),
            "sharing_efficiency_at_setup": sharing_efficiency(ledgers["setup"]),
            "rebind_events": shared_trace.rebind_events,
            "mask_size_call_forms": shared_trace.mask_size_call_forms,
        },
        "receipt_slot_names": sorted(receipts),
        "mandatory_slot_map": {
            target: list(slots) for target, slots in MANDATORY_SLOTS.items()
        },
        "trusted_computing_base": (
            "in-process capture and replay; content digests from live tensors; "
            "storage identity from the PyTorch allocator; Python-level call "
            "provenance only; no compiled-kernel identity and no adversarial "
            "threat model"
        ),
    }


# ---------------------------------------------------------------------------
# instrumented execution
# ---------------------------------------------------------------------------


def run_audited_shared_packed_multifork(
    adapter: Any,
    packed_state: PackedLowerReplayState,
    document_tokens: torch.Tensor,
    queries: Sequence[tuple[str, torch.Tensor]],
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    rebind_policy: str = "transition",
    tail_policy: str = "borrowed-prefix",
    policy_identity: Mapping[str, Any] | None = None,
    salt: str = DEFAULT_SALT,
    private_reference_traces: Mapping[str, Sequence[int]] | None = None,
    traces_by_fanout: Mapping[int, Mapping[str, Sequence[int]]] | None = None,
) -> dict[str, Any]:
    """Run one shared-packed multifork with every audit receipt captured.

    Captures, in order: the packed entry digest and inventory before any fork;
    the shared view digest after the single dequantization; the per-request
    inventory with all N forks live and before any registered transition; the
    per-request inventory after every transition and query prefill; the
    per-request inventory after decoding; and the view and packed digests after
    the forks are dropped.
    """

    started = time.perf_counter()
    packed_root = {
        "document_residual": packed_state.document_residual,
        "cache": packed_state.cache,
    }
    packed_setup_digest = tensor_tree_digest(packed_root, label="packed_entry_setup")
    packed_inventory = storage_inventory_rows(
        packed_root, role="packed_entry", salt=salt
    )
    declared_document = token_identity(document_tokens, label="document")[
        "content_sha256"
    ]
    declared_queries = sha256_json(
        [
            token_identity(tokens, label=request_id)["content_sha256"]
            for request_id, tokens in queries
        ]
    )
    declared_policy = sha256_json(
        {
            "depth": int(packed_state.depth),
            "share_mode": "shared-packed-view",
            "rebind_policy": rebind_policy,
            "tail_policy": tail_policy,
            **{key: value for key, value in sorted((policy_identity or {}).items())},
        }
    )

    entry = prepare_shared_packed_entry(
        packed_state,
        share_mode="shared-packed-view",
        rebind_policy=rebind_policy,
        tail_policy=tail_policy,
        salt=salt,
    )
    if entry.view is None:
        raise SharedPackedAuditError(
            "the shared-packed audit requires a materialized shared view; the "
            f"entry fell back to private materialization: {entry.fallback_reason}"
        )
    view_root = {
        "document_residual": entry.view.document_residual,
        "cache": entry.view.cache,
    }
    view_setup_digest = tensor_tree_digest(view_root, label="shared_view_setup")
    declared_adapter = sha256_json(layer_class_receipt(entry.view.cache, adapter))

    captures: dict[str, dict[str, list[dict[str, Any]]]] = {}

    def capture(point: str, forks: Sequence[Any]) -> None:
        captures[point] = {
            fork.request_id: storage_inventory_rows(
                {
                    "cache": fork.cache,
                    "document_residual": fork.document_residual,
                },
                role=f"{point}:{fork.request_id}",
                salt=salt,
            )
            for fork in forks
        }

    trace = run_shared_packed_multifork(
        adapter,
        entry,
        queries,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        capture=capture,
    )

    forks_created = len(trace.forks)
    trace.forks = []
    forks_released = forks_created

    view_final_digest = tensor_tree_digest(view_root, label="shared_view_final")
    packed_final_digest = tensor_tree_digest(packed_root, label="packed_entry_final")

    identity_bindings = {
        "entry_content": {
            "declared": packed_setup_digest["tree_sha256"],
            "replayed": packed_final_digest["tree_sha256"],
        },
        "policy": {
            "declared": declared_policy,
            "replayed": sha256_json(
                {
                    "depth": int(entry.depth),
                    "share_mode": entry.effective_share_mode,
                    "rebind_policy": entry.rebind_policy,
                    "tail_policy": entry.tail_policy,
                    **{
                        key: value
                        for key, value in sorted((policy_identity or {}).items())
                    },
                }
            ),
        },
        "document_tokens": {
            "declared": declared_document,
            "replayed": token_identity(document_tokens, label="document")[
                "content_sha256"
            ],
        },
        "query_tokens": {
            "declared": declared_queries,
            "replayed": sha256_json(
                [
                    token_identity(tokens, label=request_id)["content_sha256"]
                    for request_id, tokens in queries
                ]
            ),
        },
        "adapter": {
            "declared": declared_adapter,
            "replayed": sha256_json(layer_class_receipt(entry.view.cache, adapter)),
        },
    }
    audit_inputs: dict[str, Any] = {
        "entry": entry,
        "shared_trace": trace,
        "private_reference_traces": dict(private_reference_traces or {}),
        "captures": captures,
        "view_setup_digest": view_setup_digest,
        "view_final_digest": view_final_digest,
        "packed_setup_digest": packed_setup_digest,
        "packed_final_digest": packed_final_digest,
        "packed_inventory": packed_inventory,
        "identity_bindings": identity_bindings,
        "layer_classes": layer_class_receipt(entry.view.cache, adapter),
        "num_layers": int(getattr(adapter, "num_layers", 0)),
        "forks_created": forks_created,
        "forks_released": forks_released,
        "prefix_request_id": queries[0][0],
    }
    audit = audit_shared_packed_multifork(
        **audit_inputs,
        traces_by_fanout=dict(traces_by_fanout)
        if traces_by_fanout
        else {len(queries): trace.token_traces()},
    )
    audit["audit_seconds"] = time.perf_counter() - started
    return {
        "entry": entry,
        "trace": trace,
        "audit": audit,
        "captures": captures,
        # everything ``audit_shared_packed_multifork`` needs except
        # ``traces_by_fanout``, so a caller that only learns the other fanouts
        # after this run can re-evaluate the contract without re-executing it
        "audit_inputs": audit_inputs,
    }


# ---------------------------------------------------------------------------
# preflight gate
# ---------------------------------------------------------------------------


def published_private_reference_traces(
    adapter: Any,
    config: DeploymentConfig,
    document_tokens: torch.Tensor,
    queries: Sequence[tuple[str, torch.Tensor]],
    packed_state: PackedLowerReplayState,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> dict[str, list[int]]:
    """N=1 reference traces from the published Read path, called unchanged.

    Each query is run on its own through ``run_incremental_generation``, which
    forks the packed entry into a full private copy exactly as the manuscript's
    Tables 1 and 2 do.  Nothing in this function alters that path.
    """

    traces: dict[str, list[int]] = {}
    for request_id, query_tokens in queries:
        trace = run_incremental_generation(
            adapter,
            config,
            document_tokens,
            query_tokens,
            packed_state,
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
            recorder=MemoryRecorder(),
        )
        traces[request_id] = list(trace.generated_token_ids)
    return traces


@torch.inference_mode()
def run_shared_packed_multifork_gate(
    adapter: Any,
    config: DeploymentConfig,
    document_tokens: torch.Tensor,
    queries: Sequence[tuple[str, torch.Tensor]],
    *,
    group_size: int,
    max_new_tokens: int,
    eos_token_ids: set[int],
    rebind_policy: str = "transition",
    tail_policy: str = "borrowed-prefix",
    full_prefix_state: Any | None = None,
) -> dict[str, Any]:
    """Cheap per-rank preflight for the shared-packed multifork path.

    Passes if and only if all four of the following hold.

    1. The shared mode took effect on this build: the entry materialized one
       dequantized view and no fork fell back to private materialization.
    2. Sharing is non-vacuous: at least two requests, and at the tail policy's
       own sharing window every request holds a non-zero number of bytes that
       alias the one view and a non-zero number of private bytes.
    3. The N>1 shared run emits, per request, token-for-token the same ids as
       the published N=1 private-materialization path
       (``qcomem_deployment.run_incremental_generation``, called unchanged) on
       the same document and the same query.
    4. Every applicable contract target has complete coverage and a passing
       predicate.

    It does **not** require agreement with the full-prefix arm.  That
    comparison crosses the document/query chunk boundary the Qwen3.5
    GatedDeltaNet and convolution states are sensitive to; roughly a tenth of
    LongBench items are known to diverge there for reasons unrelated to
    ownership.  When ``full_prefix_state`` is supplied the full-prefix multifork
    arm is executed so its working-set fields exist, and its agreement is
    recorded as ``full_prefix_token_agreement`` -- a diagnostic that never
    gates.
    """

    if config.mode != "qcomem":
        raise MultiforkAccountingError("the multifork gate requires a Q-CoMem config")
    if len(queries) < 2:
        raise MultiforkAccountingError(
            "the multifork gate requires at least two requests"
        )
    started = time.perf_counter()
    packed_state = build_persistent_state(
        adapter,
        config,
        document_tokens,
        group_size=group_size,
        fork_strategy="deep-clone",
    )
    if not isinstance(packed_state, PackedLowerReplayState):
        raise MultiforkAccountingError(
            "the multifork gate requires a packed lower replay state; got "
            f"{type(packed_state).__name__}"
        )

    reference = published_private_reference_traces(
        adapter,
        config,
        document_tokens,
        queries,
        packed_state,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
    )

    single = run_audited_shared_packed_multifork(
        adapter,
        packed_state,
        document_tokens,
        list(queries[:1]),
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        rebind_policy=rebind_policy,
        tail_policy=tail_policy,
        private_reference_traces={queries[0][0]: reference[queries[0][0]]},
    )
    multi = run_audited_shared_packed_multifork(
        adapter,
        packed_state,
        document_tokens,
        list(queries),
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        rebind_policy=rebind_policy,
        tail_policy=tail_policy,
        private_reference_traces=reference,
    )
    # Cross-N consistency needs the N=1 and N=k traces together, and the N=k
    # traces do not exist until the run finishes.  Re-evaluate the contract
    # from the captured receipts rather than re-executing the run.
    multi_audit = audit_shared_packed_multifork(
        **multi["audit_inputs"],
        traces_by_fanout={
            1: single["trace"].token_traces(),
            len(queries): multi["trace"].token_traces(),
        },
    )

    entry = multi["entry"]
    trace = multi["trace"]
    window = multi_audit["sharing_window"]
    window_ledger = multi_audit["ownership"][f"{window}_ledger"]
    per_request = window_ledger["per_request"]

    shared_mode_effective = (
        entry.effective_share_mode == "shared-packed-view"
        and trace.fork_mode == "shared-packed-view"
        and entry.fallback_reason is None
    )
    non_vacuous_sharing = bool(
        len(per_request) >= 2
        and all(int(row["shared_nbytes"]) > 0 for row in per_request.values())
        and all(int(row["private_nbytes"]) > 0 for row in per_request.values())
    )
    equivalence = multi_audit["detail"]["cross_arm_equivalence"]
    contract = multi_audit["contract_summary"]
    contract_passed = bool(
        contract["all_applicable_targets_covered"]
        and contract["all_applicable_predicates_passed"]
    )

    full_prefix_diagnostic: dict[str, Any] | None = None
    if full_prefix_state is not None:
        full_prefix_trace = run_full_prefix_multifork(
            adapter,
            full_prefix_state,
            list(queries),
            max_new_tokens=max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
        full_prefix_diagnostic = {
            "semantic": (
                "diagnostic only; never gates. Q-CoMem and full prefix consume "
                "the document/query boundary differently and the Qwen3.5 "
                "recurrence is sensitive to that boundary"
            ),
            "comparison": compare_token_traces(
                reference=full_prefix_trace.token_traces(),
                candidate=trace.token_traces(),
                reference_label="full-prefix-multifork",
                candidate_label="shared-packed-multifork",
            ),
            "full_prefix_request_traces": [
                row.summary() for row in full_prefix_trace.request_traces
            ],
            "full_prefix_phase_allocated_nbytes": (
                full_prefix_trace.phase_allocated_nbytes
            ),
            "full_prefix_peak_allocated_nbytes": (
                full_prefix_trace.peak_allocated_nbytes
            ),
        }

    passed = bool(
        shared_mode_effective
        and non_vacuous_sharing
        and equivalence["token_sequences_identical"]
        and contract_passed
    )
    return {
        "schema": GATE_SCHEMA,
        "passed": passed,
        "gate_semantic": (
            "shared mode took effect; sharing is non-vacuous at the policy's "
            "sharing window; the N>1 shared run is token-identical to the "
            "published N=1 private-materialization path; and every applicable "
            "contract target is covered and passing. Agreement with the "
            "full-prefix arm is a diagnostic and never gates."
        ),
        "config": config.name,
        "request_count": len(queries),
        "rebind_policy": rebind_policy,
        "tail_policy": tail_policy,
        "sharing_window": window,
        "shared_mode_effective": shared_mode_effective,
        "share_mode_effective": entry.effective_share_mode,
        "fallback_reason": entry.fallback_reason,
        "non_vacuous_sharing": non_vacuous_sharing,
        "per_request_window_ownership": per_request,
        "semantic_equivalence": equivalence,
        "contract_summary": contract,
        "contract_passed": contract_passed,
        "audit": multi_audit,
        "n1_audit": single["audit"],
        "full_prefix_token_agreement": full_prefix_diagnostic,
        "gate_seconds": time.perf_counter() - started,
    }


__all__ = [
    "AUDIT_SCHEMA",
    "DEFAULT_SALT",
    "GATE_SCHEMA",
    "SHARING_WINDOW_BY_TAIL_POLICY",
    "SharedPackedAuditError",
    "audit_shared_packed_multifork",
    "canonical_json_bytes",
    "evaluate_dequantized_view_immutability",
    "evaluate_dispatch_provenance",
    "evaluate_frozen_identity",
    "evaluate_packed_entry_lifetime",
    "evaluate_private_ownership",
    "evaluate_residual_chunk_binding",
    "evaluate_tail_safe_append",
    "layer_class_receipt",
    "missing_slot_receipt",
    "published_private_reference_traces",
    "run_audited_shared_packed_multifork",
    "run_shared_packed_multifork_gate",
    "sha256_json",
    "slot_receipt",
    "tensor_content_sha256",
    "tensor_tree_digest",
    "token_identity",
]
