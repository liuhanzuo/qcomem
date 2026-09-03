#!/usr/bin/env python3
"""Execute exactly one fresh-process reference, clean, or mutant R39 lane."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback
from types import MethodType
from typing import Any, Mapping, Sequence

import r39_contract as contract
import r39_live_common as live

import torch


LANES = ("reference", "clean", "mutant")
CASE_SCHEMA = "forkaudit-r39-blind-fault-lane-v1"


def write_bytes(path: Path, payload: bytes) -> None:
    contract.require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    contract.require(not pending.exists(), f"stale pending sidecar {pending}")
    with pending.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(path)


def logit_sidecar(path: Path, root: Path, logits: torch.Tensor) -> dict[str, Any]:
    raw = live.tensor_bytes(logits.detach().cpu().float().contiguous())
    write_bytes(path, raw)
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "nbytes": len(raw),
        "shape": [int(item) for item in logits.shape],
        "dtype": "float32-little-endian",
    }


def injection_receipt(
    *, fault_id: str, lane: str, fault_row_sha256: str, event: str,
    selector: Mapping[str, Any], details: Mapping[str, Any], applied: bool,
) -> dict[str, Any]:
    value = {
        "schema_version": "forkaudit-r39-byte-bound-injection-receipt-v1",
        "fault_id": fault_id,
        "lane": lane,
        "fault_row_sha256": fault_row_sha256,
        "event": event,
        "selector_resolution_sha256": contract.sha256_json(selector),
        "selector_resolution": dict(selector),
        "payload_applied": bool(applied),
        "eligible_noop": not applied,
        "exactly_one_named_locus": True,
        "semantic_outputs_used_for_target_selection": False,
        "details": dict(details),
    }
    value["receipt_sha256"] = contract.sha256_json(value)
    return value


class TailCopyController:
    def __init__(self, sequence: Any, *, mutant: bool) -> None:
        self.sequence = sequence
        self.mutant = mutant
        self.original = sequence._detach_partial_document_tail
        self.receipt: dict[str, Any] | None = None

    def install(self) -> None:
        controller = self

        def replacement(sequence: Any, batch_index: int) -> None:
            controller.detach(sequence, batch_index)

        self.sequence._detach_partial_document_tail = MethodType(
            replacement, self.sequence
        )

    def detach(self, sequence: Any, batch_index: int) -> None:
        arena = sequence.arena
        tail = arena.document_length % arena.page_size
        if tail == 0 or sequence._tail_detached[batch_index]:
            return
        logical_tail = arena.document_blocks_per_sequence - 1
        true_source = batch_index * arena.document_blocks_per_sequence + logical_tail
        selected_source = true_source - 1 if self.mutant else true_source
        target = sequence._take_private_block(batch_index)
        true_digest = contract.sha256_json({
            "k": live.tensor_sha(arena.key_cache[true_source, :tail]),
            "v": live.tensor_sha(arena.value_cache[true_source, :tail]),
        })
        selected_digest = contract.sha256_json({
            "k": live.tensor_sha(arena.key_cache[selected_source, :tail]),
            "v": live.tensor_sha(arena.value_cache[selected_source, :tail]),
        })
        contract.require(
            not self.mutant or selected_digest != true_digest,
            "BF01 wrong source is byte-equal to true tail",
        )
        arena.key_cache[target, :tail].copy_(arena.key_cache[selected_source, :tail])
        arena.value_cache[target, :tail].copy_(arena.value_cache[selected_source, :tail])
        destination_digest = contract.sha256_json({
            "k": live.tensor_sha(arena.key_cache[target, :tail]),
            "v": live.tensor_sha(arena.value_cache[target, :tail]),
        })
        sequence.block_table[batch_index, logical_tail] = target
        sequence._logical_physical[batch_index][logical_tail] = target
        sequence._tail_detached[batch_index] = True
        sequence.partial_tail_staging_copy_nbytes += (
            2 * tail * arena.num_key_value_heads * arena.head_dim
            * arena.key_cache.element_size()
        )
        contract.require(self.receipt is None, "BF01 injection locus repeated")
        self.receipt = {
            "batch_index": batch_index,
            "tail_tokens": int(tail),
            "true_source_block": true_source,
            "selected_source_block": selected_source,
            "destination_block": target,
            "true_source_digest": true_digest,
            "selected_source_digest": selected_digest,
            "destination_preappend_digest": destination_digest,
            "destination_matches_selected_source": destination_digest == selected_digest,
            "destination_matches_true_source": destination_digest == true_digest,
            "destination_is_private": target in {
                int(item) for item in sequence.reservations.reshape(-1).tolist()
            },
            "copy_preceded_append_write": True,
        }

    def restore(self) -> None:
        self.sequence._detach_partial_document_tail = self.original


def snapshot_kv_before_call(request: Any, plan: Any) -> list[dict[str, Any]]:
    rows = []
    for layer_index in plan.full_attention_layer_indices:
        sequence = request.layers[layer_index].sequence
        absolute = int(sequence.sequence_length)
        logical = absolute // sequence.arena.page_size
        physical = int(sequence._logical_physical[0][logical])
        contract.require(physical >= 0, "BF03 final append page absent")
        rows.append({
            "layer_index": int(layer_index),
            "sequence": sequence,
            "physical_block": physical,
            "page_key": sequence.arena.key_cache[physical].detach().clone(),
            "page_value": sequence.arena.value_cache[physical].detach().clone(),
            "block_table": sequence.block_table.detach().clone(),
            "sequence_length": int(sequence.sequence_length),
            "appended_tokens": int(sequence.appended_tokens),
            "next_private": list(sequence._next_private),
            "tail_detached": list(sequence._tail_detached),
            "logical_physical": copy.deepcopy(sequence._logical_physical),
            "partial_tail_staging_copy_nbytes": int(sequence.partial_tail_staging_copy_nbytes),
            "last_append_capture_id": sequence.last_append_capture_id,
            "last_append_audit": copy.deepcopy(sequence.last_append_audit),
            "append_event_count": int(sequence._append_event_count),
        })
    return rows


def rollback_kv(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    before_versions = []
    advanced_versions = []
    for row in rows:
        sequence = row["sequence"]
        before_versions.append(int(row["append_event_count"]))
        advanced_versions.append(int(sequence._append_event_count))
        contract.require(
            int(sequence._append_event_count) == int(row["append_event_count"]) + 1,
            "BF03 did not observe one real clean append",
        )
        physical = int(row["physical_block"])
        sequence.arena.key_cache[physical].copy_(row["page_key"])
        sequence.arena.value_cache[physical].copy_(row["page_value"])
        sequence.block_table.copy_(row["block_table"])
        sequence.sequence_length = int(row["sequence_length"])
        sequence.appended_tokens = int(row["appended_tokens"])
        sequence._next_private = list(row["next_private"])
        sequence._tail_detached = list(row["tail_detached"])
        sequence._logical_physical = copy.deepcopy(row["logical_physical"])
        sequence.partial_tail_staging_copy_nbytes = int(row["partial_tail_staging_copy_nbytes"])
        sequence.last_append_capture_id = row["last_append_capture_id"]
        sequence.last_append_audit = copy.deepcopy(row["last_append_audit"])
        sequence._append_event_count = int(row["append_event_count"])
    torch.cuda.synchronize()
    exact = all(
        live.tensor_sha(row["sequence"].arena.key_cache[int(row["physical_block"])])
        == live.tensor_sha(row["page_key"])
        and live.tensor_sha(row["sequence"].arena.value_cache[int(row["physical_block"])])
        == live.tensor_sha(row["page_value"])
        for row in rows
    )
    contract.require(exact, "BF03 KV byte rollback mismatch")
    return {
        "full_attention_layer_count": len(rows),
        "versions_before": before_versions,
        "versions_after_real_append": advanced_versions,
        "versions_after_rollback": [int(row["sequence"]._append_event_count) for row in rows],
        "page_bytes_restored_exact": True,
        "logical_lengths_restored_exact": all(
            int(row["sequence"].sequence_length) == int(row["sequence_length"])
            for row in rows
        ),
    }


def apply_h4_locus(
    *, fault_id: str, lane: str, persistent: Any, group: Any, plan: Any,
) -> dict[str, Any]:
    mutant = lane == "mutant"
    request = group.requests[0]
    if fault_id == "R39-BF04":
        layer, family, state_index, current = live.first_gdn_coordinate(request, plan)
        base = getattr(persistent.layers[layer], family)[state_index]
        before = live.tensor_descriptor(current)
        stale = live.tensor_descriptor(base)
        contract.require(live.storage_key(current) != live.storage_key(base), "BF04 target aliases base")
        contract.require(before["content_sha256"] != stale["content_sha256"], "BF04 sources equal")
        if mutant:
            current.copy_(base)
            torch.cuda.synchronize()
        after = live.tensor_descriptor(current)
        return {
            "coordinate": f"{layer}:{family}:{state_index}",
            "correct_h4_source": before,
            "stale_h1_source": stale,
            "post_locus": after,
            "private_storage_unchanged": live.storage_key(current) != live.storage_key(base),
            "post_matches_stale": after["content_sha256"] == stale["content_sha256"],
            "post_matches_correct": after["content_sha256"] == before["content_sha256"],
        }
    if fault_id == "R39-BF05":
        left, right = live.select_gdn_permutation(request, plan)
        left_map = getattr(request.layers[left[0]], left[1])
        right_map = getattr(request.layers[right[0]], right[1])
        left_before = live.tensor_descriptor(left[3])
        right_before = live.tensor_descriptor(right[3])
        contract.require(live.storage_key(left[3]) != live.storage_key(right[3]), "BF05 pair aliases")
        if mutant:
            left_map[left[2]], right_map[right[2]] = right[3], left[3]
        left_after = live.tensor_descriptor(left_map[left[2]])
        right_after = live.tensor_descriptor(right_map[right[2]])
        return {
            "left_coordinate": f"{left[0]}:{left[1]}:{left[2]}",
            "right_coordinate": f"{right[0]}:{right[1]}:{right[2]}",
            "left_before": left_before,
            "right_before": right_before,
            "left_after": left_after,
            "right_after": right_after,
            "bytes_modified_by_swap": False,
            "bindings_transposed": (
                left_after["content_sha256"] == right_before["content_sha256"]
                and right_after["content_sha256"] == left_before["content_sha256"]
            ),
        }
    if fault_id == "R39-BF06":
        layer_index = min(int(item) for item in plan.full_attention_layer_indices)
        sequence = request.layers[layer_index].sequence
        suffix = live.kv_unused_suffix(sequence)
        selected = live.select_cross_family_coordinate(request, plan, suffix)
        target_map = getattr(request.layers[selected[0]], selected[1])
        original = selected[3]
        before = live.tensor_descriptor(original)
        if mutant:
            view = live.make_kv_backed_view(
                original, sequence.arena.key_cache, int(suffix["byte_start"])
            )
            target_map[selected[2]] = view
            del original
            gc.collect()
        observed = target_map[selected[2]]
        after = live.tensor_descriptor(observed)
        key_storage = live.storage_key(sequence.arena.key_cache)
        overlap = live.storage_key(observed) == key_storage
        return {
            "kv_layer": layer_index,
            "kv_suffix": dict(suffix),
            "gdn_coordinate": f"{selected[0]}:{selected[1]}:{selected[2]}",
            "gdn_before": before,
            "gdn_after": after,
            "logical_gdn_content_preserved_at_injection": before["content_sha256"] == after["content_sha256"],
            "cross_family_storage_overlap": overlap,
            "overlap_outside_terminal_valid_kv": (
                after["byte_start"] >= suffix["byte_start"]
                and after["byte_end_exclusive"] <= suffix["byte_end_exclusive"]
            ) if overlap else False,
        }
    raise live.LiveError(f"not an H4 fault: {fault_id}")


def persistent_bundle(
    persistent: Any, plan: Any, persistent_b: Any | None = None
) -> dict[str, Any]:
    value = {"document_a": live.persistent_digest_bundle(persistent, plan)}
    if persistent_b is not None:
        value["document_b"] = live.persistent_digest_bundle(persistent_b, plan)
    return value


def run_lane(args: argparse.Namespace) -> dict[str, Any]:
    contract.require(args.lane in LANES, "lane name")
    contract.require(args.fault_id in contract.FAULT_IDS, "fault id")
    contract.require(contract.FAULT_TO_GPU[args.fault_id] == args.gpu_index, "fault/GPU map")
    contract.require(not args.lane_dir.exists(), "lane directory already exists")
    freeze = contract.verify_freeze(args.protocol, args.plan)
    contract.require(contract.sha256_file(args.execution_input) == contract.EXECUTION_INPUT_SHA256, "execution input SHA")
    contract.verify_source_manifest(args.source_manifest, args.source_root)
    feasibility = json.loads(args.feasibility.read_text(encoding="utf-8"))
    contract.validate_feasibility(feasibility, fault_id=args.fault_id, freeze=freeze)
    contract.require(feasibility["eligible"] is True, "ineligible fault cannot start a lane")
    contract.require(contract.sha256_file(args.feasibility) == args.expected_feasibility_sha256, "feasibility SHA drift")
    contract.require(args.fault_id not in {"R39-BF02", "R39-BF09", "R39-BF11"}, "unsupported exact selector reached lane")
    selector = feasibility["selector_resolution"]
    input_rank = int(selector.get("input_rank", args.gpu_index))
    execution_input = json.loads(args.execution_input.read_text(encoding="utf-8"))
    runtime = live.load_runtime(
        input_rank=input_rank, expected_gpu_uuid=args.expected_gpu_uuid,
        execution_input=execution_input,
    )
    args.lane_dir.mkdir(parents=True)
    case_path = args.lane_dir / "case.json"
    persistent = persistent_b = group = persistent_guard = persistent_guard_b = request_guard = None
    ledgers: list[Any] = []
    backends: list[str] = []
    tail_controller: TailCopyController | None = None
    retained: list[torch.Tensor] = []
    case: dict[str, Any] | None = None
    try:
        warmup = live.discarded_warmup(runtime)
        h0 = live.allocator_snapshot(reset_peak=True)
        contract.require(h0["allocated_bytes"] == runtime.allocator_baseline["allocated_bytes"], "H0 allocator baseline drift")
        bf10_identity = None
        if args.fault_id == "R39-BF10":
            a_rank = int(selector["document_a_rank"])
            b_rank = int(selector["document_b_rank"])
            (
                persistent, persistent_b, group, persistent_guard,
                persistent_guard_b, request_guard, bf10_identity,
            ) = live.build_two_document_case(
                runtime, execution_input, document_a_rank=a_rank,
                document_b_rank=b_rank, mutant=args.lane == "mutant",
            )
            injection = injection_receipt(
                fault_id=args.fault_id, lane=args.lane,
                fault_row_sha256=freeze["fault_row_sha256"][args.fault_id],
                event="H1", selector=selector, details=bf10_identity,
                applied=args.lane == "mutant",
            )
        else:
            persistent, group, persistent_guard, request_guard = live.build_default_case(runtime)
            injection = None
        h1_base = persistent_bundle(persistent, runtime.plan, persistent_b)
        h1 = live.allocator_snapshot(reset_peak=True)
        target_layer = min(int(item) for item in runtime.plan.full_attention_layer_indices)
        if args.fault_id == "R39-BF01":
            tail_controller = TailCopyController(
                group.requests[0].layers[target_layer].sequence,
                mutant=args.lane == "mutant",
            )
            tail_controller.install()
        ledgers, backends = live.make_ledgers(runtime, group)
        currents = [runtime.queries[0], runtime.queries[1]]
        generated = [[], []]
        sidecars: list[dict[str, Any]] = []
        schedule: list[dict[str, int]] = []
        transition_receipts: list[dict[str, Any]] = []
        phase = None
        h4 = None
        h4_base = None
        rollback_details = None
        for round_index in range(8):
            for request_index in range(2):
                if round_index == 1:
                    transition_receipts.append(
                        live.repair.prepare_borrowed_single_token_conv_transition(
                            persistent, group.requests,
                            runtime.plan.linear_layer_indices,
                            request_index=request_index,
                        )
                    )
                rollback_rows = None
                if args.fault_id == "R39-BF03" and round_index == 7 and request_index == 0:
                    rollback_rows = snapshot_kv_before_call(
                        group.requests[0], runtime.plan
                    )
                output, logits = live.model_call(
                    runtime, group.requests[request_index], backends[request_index],
                    currents[request_index],
                )
                token = int(logits.argmax(dim=-1).item())
                relative = Path("sidecars") / f"call-{len(schedule):02d}-round-{round_index}-request-{request_index}.fp32.bin"
                sidecars.append(logit_sidecar(args.lane_dir / relative, args.lane_dir, logits))
                schedule.append({
                    "call_index": len(schedule),
                    "round_index": round_index,
                    "request_index": request_index,
                })
                generated[request_index].append(token)
                currents[request_index] = torch.tensor(
                    [[token]], dtype=torch.long, device="cuda:0"
                )
                del output, logits
                if rollback_rows is not None:
                    if args.lane == "mutant":
                        rollback_details = rollback_kv(rollback_rows)
                    else:
                        rollback_details = {
                            "full_attention_layer_count": len(rollback_rows),
                            "versions_before": [int(row["append_event_count"]) for row in rollback_rows],
                            "versions_after_real_append": [int(row["sequence"]._append_event_count) for row in rollback_rows],
                            "rollback_applied": False,
                            "clean_advance_exactly_once": all(
                                int(row["sequence"]._append_event_count) == int(row["append_event_count"]) + 1
                                for row in rollback_rows
                            ),
                        }
                    injection = injection_receipt(
                        fault_id=args.fault_id, lane=args.lane,
                        fault_row_sha256=freeze["fault_row_sha256"][args.fault_id],
                        event="H5-final-call", selector=selector,
                        details=rollback_details, applied=args.lane == "mutant",
                    )
                    rollback_rows = None
            if round_index == 0:
                if args.fault_id in {"R39-BF04", "R39-BF05", "R39-BF06"}:
                    details = apply_h4_locus(
                        fault_id=args.fault_id, lane=args.lane,
                        persistent=persistent, group=group, plan=runtime.plan,
                    )
                    injection = injection_receipt(
                        fault_id=args.fault_id, lane=args.lane,
                        fault_row_sha256=freeze["fault_row_sha256"][args.fault_id],
                        event="H4", selector=selector, details=details,
                        applied=args.lane == "mutant",
                    )
                phase = live.storage_witness.capture_gdn_phase_witness(
                    persistent, group.requests, runtime.plan.linear_layer_indices,
                    run_id=contract.RUN_ID,
                    cell_id=f"{args.fault_id}-{args.lane}",
                    kv_policy=live.SHARED,
                    phase=live.storage_witness.PHASE_POST_TRANSITION,
                    policy=live.BORROW,
                    persistent_guard=persistent_guard,
                    request_guard=request_guard,
                    completed_request_indices=[0, 1],
                )
                h4_base = persistent_bundle(persistent, runtime.plan, persistent_b)
                h4 = live.allocator_snapshot()
        contract.require(len(schedule) == 16 and all(len(row) == 8 for row in generated), "H6 call schedule")
        if args.fault_id == "R39-BF01":
            contract.require(tail_controller is not None and tail_controller.receipt is not None, "BF01 locus absent")
            injection = injection_receipt(
                fault_id=args.fault_id, lane=args.lane,
                fault_row_sha256=freeze["fault_row_sha256"][args.fault_id],
                event="H2", selector=selector,
                details=tail_controller.receipt,
                applied=args.lane == "mutant",
            )
        h6_base = persistent_bundle(persistent, runtime.plan, persistent_b)
        h6 = live.allocator_snapshot()
        terminal = {
            "generated_token_ids": generated,
            "logical_kv": json.loads(json.dumps(
                live.resident_runner._request_logical_kv_digests(
                    group, runtime.plan.full_attention_layer_indices
                )
            )),
            "gdn": json.loads(json.dumps(
                live.resident_runner._resident_linear_states(
                    group, runtime.plan.linear_layer_indices
                )
            )),
        }
        h7_base = None
        scrub_receipt = None
        if args.fault_id in {"R39-BF07", "R39-BF08"}:
            scrub_live = live.scrub_private_state(persistent, group, runtime.plan)
            scrub_receipt = live.pointer_free_scrub_receipt(scrub_live)
            private_tensors = scrub_live["live_private_tensors"]
            if args.fault_id == "R39-BF07":
                selected = live.select_nonzero_base_slice(persistent, runtime.plan)
                tensor = getattr(
                    persistent.layers[int(selected["layer_index"])],
                    str(selected["family"]),
                )[int(selected["state_index"])]
                raw = tensor.view(torch.uint8)
                before_digest = live.tensor_sha(tensor)
                if args.lane == "mutant":
                    start = int(selected["byte_offset"])
                    end = start + int(selected["byte_length"])
                    raw[start:end].zero_()
                    torch.cuda.synchronize()
                after_digest = live.tensor_sha(tensor)
                details = {
                    "selected_slice": selected,
                    "persistent_digest_before": before_digest,
                    "persistent_digest_after": after_digest,
                    "digest_changed": before_digest != after_digest,
                    "private_scrub_completed_first": True,
                    "allocation_metadata_changed": False,
                }
                injection = injection_receipt(
                    fault_id=args.fault_id, lane=args.lane,
                    fault_row_sha256=freeze["fault_row_sha256"][args.fault_id],
                    event="H7-post-private-scrub", selector=selector,
                    details=details, applied=args.lane == "mutant",
                )
                tensor = raw = None
            else:
                contract.require(bool(private_tensors), "BF08 private storage missing")
                selected_tensor = private_tensors[0]
                selected_key = live.storage_key(selected_tensor)
                if args.lane == "mutant":
                    retained.append(selected_tensor)
                injection = injection_receipt(
                    fault_id=args.fault_id, lane=args.lane,
                    fault_row_sha256=freeze["fault_row_sha256"][args.fault_id],
                    event="H7-release", selector=selector,
                    details={
                        "selected_backing_nbytes": selected_key[2],
                        "selected_backing_storage_token": hashlib.sha256(repr(selected_key).encode()).hexdigest(),
                        "selected_bytes_zero": bool(torch.count_nonzero(selected_tensor).item() == 0),
                        "retained_backing_storage_count": 1 if args.lane == "mutant" else 0,
                        "reused_by_live_request": False,
                    },
                    applied=args.lane == "mutant",
                )
                selected_tensor = None
            private_tensors = []
            scrub_live = None
            h7_base = persistent_bundle(persistent, runtime.plan, persistent_b)
        contract.require(injection is not None, "exactly one injection/no-op receipt absent")

        # H7 faults are evaluated by the unchanged validators only after their
        # post-terminal mutation.  H6 faults are evaluated at H6.
        forkaudit = live.live_forkaudit_receipts(
            persistent=persistent, group=group, plan=runtime.plan,
            ledgers=ledgers, persistent_guard=persistent_guard,
            request_guard=request_guard, phase=phase,
        )
        if persistent_b is not None:
            try:
                receipt_b = live.storage_witness.verify_persistent_gdn_guard(
                    persistent_guard_b, persistent_b
                )
                extra = {"predicate": "PERSISTENT_GDN_DOCUMENT_B_IMMUTABLE", "passed": True, "receipt": receipt_b}
            except BaseException as exc:
                extra = {"predicate": "PERSISTENT_GDN_DOCUMENT_B_IMMUTABLE", "passed": False, "gate_id": getattr(exc, "gate_id", None), "message": str(exc), "receipt": None}
            forkaudit["predicate_rows"].append(extra)
            if not extra["passed"] and forkaudit["verdict"] == "pass":
                forkaudit["verdict"] = "fail"
                forkaudit["first_failed_predicate"] = extra["predicate"]
                forkaudit["first_failed_gate_id"] = extra.get("gate_id")
        full_trace_path = args.lane_dir / "forkaudit-trace.json"
        contract.atomic_json(full_trace_path, forkaudit)

        if tail_controller is not None:
            tail_controller.restore()
            tail_controller = None
        live.rr2._unregister_backends(backends)
        backends = []
        ledgers = []
        currents = []
        # Drop all request/document caches.  For BF08 mutant, exactly one zeroed
        # tensor remains in ``retained`` until the synchronized H7 endpoint.
        group = request_guard = persistent_guard = persistent_guard_b = None
        persistent = persistent_b = None
        gc.collect()
        torch.cuda.empty_cache()
        h7 = live.allocator_snapshot()
        retained_nbytes = sum(live.storage_key(item)[2] for item in retained)
        retained = []
        gc.collect()
        torch.cuda.empty_cache()
        final_recovery = live.allocator_snapshot()
        cleanup = {
            "h7_allocator_endpoint": h7,
            "h0_allocator_endpoint": h0,
            "h7_restores_h0_exact": h7["allocated_bytes"] == h0["allocated_bytes"],
            "retained_allocator_accounted_nbytes": retained_nbytes,
            "post_measurement_recovery": final_recovery,
            "post_measurement_recovery_exact": final_recovery["allocated_bytes"] == h0["allocated_bytes"],
        }
        case = {
            "schema_version": CASE_SCHEMA,
            "run_id": contract.RUN_ID,
            "status": "expected_horizon_completed",
            "fault_id": args.fault_id,
            "fault_row_sha256": freeze["fault_row_sha256"][args.fault_id],
            "lane": args.lane,
            "gpu_index": args.gpu_index,
            "input_rank": input_rank,
            "expected_gpu_uuid": args.expected_gpu_uuid,
            "expected_horizon": contract.EXPECTED_HORIZON[args.fault_id],
            "reached_horizon": contract.EXPECTED_HORIZON[args.fault_id],
            "plan_raw_sha256": contract.PLAN_RAW_SHA256,
            "protocol_raw_sha256": contract.PROTOCOL_RAW_SHA256,
            "feasibility": {
                "path": str(args.feasibility),
                "sha256": args.expected_feasibility_sha256,
                "receipt_sha256": feasibility["receipt_sha256"],
            },
            "source_manifest_sha256": contract.sha256_file(args.source_manifest),
            "hardware": runtime.hardware,
            "discarded_warmup": warmup,
            "all_production_assertions_enabled": True,
            "selective_gate_suppression": False,
            "ordered_schedule": schedule,
            "logit_sidecars": sidecars,
            "semantic_results": terminal,
            "persistent_base_snapshots": {
                "H1": h1_base, "H4": h4_base, "H6": h6_base,
                "H7_pre_release": h7_base,
            },
            "allocator_endpoints": {
                "H0": h0, "H1": h1, "H4": h4, "H6": h6, "H7": h7,
            },
            "transition_receipts": transition_receipts,
            "byte_bound_injection_receipt": injection,
            "scrub_receipt": scrub_receipt,
            "forkaudit": {
                "trace_path": full_trace_path.relative_to(args.lane_dir).as_posix(),
                "trace_sha256": contract.sha256_file(full_trace_path),
                "verdict": forkaudit["verdict"],
                "first_failed_predicate": forkaudit["first_failed_predicate"],
                "first_failed_gate_id": forkaudit["first_failed_gate_id"],
                "compiled_binary_identity_coverage": "partial",
                "autotuning_choice_coverage": "partial",
            },
            "cleanup": cleanup,
            "bf10_component_identity": bf10_identity,
            "operational_invalid": None,
        }
        contract.atomic_json(case_path, case)
        return case
    except BaseException as exc:
        invalid = {
            "schema_version": "forkaudit-r39-lane-operational-invalid-v1",
            "run_id": contract.RUN_ID,
            "fault_id": args.fault_id,
            "lane": args.lane,
            "status": "operational_invalid",
            "exception_type": type(exc).__name__,
            "gate_id": getattr(exc, "gate_id", None),
            "message": str(exc),
            "traceback": traceback.format_exc(),
            "expected_horizon_completed": False,
            "scientific_pair_eligible": False,
            "exception_allowlisted": False,
            "outcome_preserved": True,
        }
        if args.lane_dir.exists() and not (args.lane_dir / "operational-invalid.json").exists():
            contract.atomic_json(args.lane_dir / "operational-invalid.json", invalid)
        raise
    finally:
        if tail_controller is not None:
            try:
                tail_controller.restore()
            except Exception:
                pass
        if backends:
            try:
                live.rr2._unregister_backends(backends)
            except Exception:
                pass
        retained = []
        ledgers = []
        backends = []
        persistent = persistent_b = group = persistent_guard = persistent_guard_b = request_guard = runtime = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fault-id", choices=contract.FAULT_IDS, required=True)
    value.add_argument("--lane", choices=LANES, required=True)
    value.add_argument("--gpu-index", type=int, choices=range(8), required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    value.add_argument("--lane-dir", type=Path, required=True)
    value.add_argument("--protocol", type=Path, required=True)
    value.add_argument("--plan", type=Path, required=True)
    value.add_argument("--execution-input", type=Path, required=True)
    value.add_argument("--feasibility", type=Path, required=True)
    value.add_argument("--expected-feasibility-sha256", required=True)
    value.add_argument("--source-root", type=Path, required=True)
    value.add_argument("--source-manifest", type=Path, required=True)
    return value


if __name__ == "__main__":
    print(json.dumps(run_lane(parser().parse_args()), sort_keys=True), flush=True)
