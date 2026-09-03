#!/usr/bin/env python3
"""Candidate-import-free replay for the R30 end-to-end semantic control."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import statistics
from typing import Any, Iterable

import numpy as np


SCHEMA = "forkaudit-r30-e2e-independent-replay-v1"
INPUT_SCHEMA = "forkaudit-r30-e2e-input-manifest-v1"
REFERENCE_SCHEMA = "forkaudit-r30-e2e-reference-v1"
CANDIDATE_SCHEMA = "forkaudit-r30-e2e-candidate-v1"
REPAIR_SCHEMA = "qcomem-single-token-gdn-conv-privatization-v1"
ARM_IDS = (
    "fresh-materialized",
    "fresh-borrowed",
    "shared-materialized",
    "shared-borrowed",
)
TRACKS = ("greedy", "teacher_forced_reference_history")
FULL_LAYERS = tuple(range(3, 40, 4))
GREEDY_STEPS = 4
REQUESTS = 2
CASES = 2


class ReplayError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReplayError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def int64_sha256(values: Iterable[int]) -> str:
    array = np.asarray([int(value) for value in values], dtype="<i8")
    require(array.ndim == 1, "token array rank drift")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def load_bound_json(path: Path, expected_sha256: str, schema: str, label: str) -> dict[str, Any]:
    require(sha256_file(path) == expected_sha256, f"{label} SHA drift")
    value = json.loads(path.read_bytes())
    require(value.get("schema_version") == schema, f"{label} schema drift")
    return value


def safe_sidecar_path(root: Path, relative: str) -> Path:
    pure = PurePosixPath(relative)
    require(not pure.is_absolute() and ".." not in pure.parts, "unsafe sidecar path")
    target = root.joinpath(*pure.parts).resolve()
    require(target.is_relative_to(root.resolve()), "sidecar escapes artifact root")
    return target


def load_sidecars(
    root: Path,
    receipts: list[dict[str, Any]],
    *,
    expected_count: int,
    prefix: str,
) -> dict[str, np.ndarray]:
    require(isinstance(receipts, list) and len(receipts) == expected_count, f"{prefix} sidecar count drift")
    result: dict[str, np.ndarray] = {}
    vocab_size = None
    for receipt in receipts:
        record_id = receipt.get("record_id")
        require(isinstance(record_id, str) and record_id and record_id not in result, f"{prefix} record ID drift")
        path = safe_sidecar_path(root, receipt.get("path", ""))
        require(path.is_file(), f"{prefix} sidecar missing: {record_id}")
        require(sha256_file(path) == receipt.get("sha256"), f"{prefix} sidecar SHA drift: {record_id}")
        array = np.load(path, allow_pickle=False)
        require(array.dtype == np.dtype("float32"), f"{prefix} sidecar dtype drift")
        require(array.ndim == 1 and array.size > 1, f"{prefix} sidecar shape drift")
        require(list(array.shape) == receipt.get("shape"), f"{prefix} sidecar receipt shape drift")
        require(bool(np.isfinite(array).all()), f"{prefix} sidecar non-finite")
        require(int(np.argmax(array)) == receipt.get("argmax_token_id"), f"{prefix} argmax receipt drift")
        if vocab_size is None:
            vocab_size = int(array.size)
        require(int(array.size) == vocab_size, f"{prefix} vocabulary size drift")
        result[record_id] = array
    return result


def reference_replay(
    inputs: dict[str, Any], reference: dict[str, Any], sidecars: dict[str, np.ndarray]
) -> dict[tuple[int, int], dict[str, Any]]:
    require(reference.get("reference_source_distinct") is True, "reference source is not distinct")
    require(reference.get("candidate_cache_trace_tensor_objects_imported") is False, "reference imported candidate objects")
    require(reference.get("full_model_recompute_each_step") is True, "reference does not recompute")
    require(reference.get("use_cache") is False, "reference used a cache")
    imports = reference.get("imports", {})
    require(imports.get("observed_forbidden_modules") == [], "reference forbidden import audit failed")
    cases = {int(row["case_index"]): row for row in inputs["cases"]}
    require(set(cases) == {0, 1}, "input case identity drift")
    result = {}
    referenced_sidecars = set()
    for row in reference.get("rows", []):
        key = (int(row["case_index"]), int(row["request_index"]))
        require(key not in result and key[0] in cases and key[1] in (0, 1), "reference row identity drift")
        case = cases[key[0]]
        query = case["queries"][key[1]]
        require(row.get("document_token_ids_sha256") == case["document_token_ids_sha256"], "reference document binding drift")
        require(row.get("query_token_ids_sha256") == query["token_ids_sha256"], "reference query binding drift")
        generated = [int(value) for value in row.get("generated_token_ids", [])]
        steps = row.get("steps")
        require(len(generated) == len(steps) == GREEDY_STEPS, "reference horizon drift")
        history = [int(value) for value in case["document_token_ids"]] + [
            int(value) for value in query["token_ids"]
        ]
        for step_index, step in enumerate(steps):
            require(step.get("step_index") == step_index, "reference step index drift")
            require(step.get("raw_history_token_count") == len(history), "reference history length drift")
            require(step.get("raw_history_token_ids_sha256") == int64_sha256(history), "reference history digest drift")
            record_id = step.get("logit_record_id")
            require(record_id in sidecars and record_id not in referenced_sidecars, "reference sidecar reference drift")
            token = int(np.argmax(sidecars[record_id]))
            require(token == generated[step_index] == step.get("generated_token_id"), "reference token/sidecar drift")
            referenced_sidecars.add(record_id)
            history.append(token)
        result[key] = row
    require(set(result) == {(case, request) for case in range(CASES) for request in range(REQUESTS)}, "reference row denominator drift")
    require(referenced_sidecars == set(sidecars), "reference has orphan sidecars")
    return result


def receipt_overlap(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left["request_storage_id"] != right["request_storage_id"]:
        return False
    left_start, left_end = left["request_byte_interval"]
    right_start, right_end = right["request_byte_interval"]
    return int(left_start) < int(right_end) and int(right_start) < int(left_end)


def verify_gdn_snapshot(snapshot: dict[str, Any], *, borrowed_setup: bool, setup: bool) -> bool:
    rows = snapshot.get("rows")
    require(isinstance(rows, list) and len(rows) == 120, "GDN ownership row count drift")
    expected_keys = {
        (request, layer, family, 0)
        for request in range(REQUESTS)
        for layer in range(40)
        if layer not in FULL_LAYERS
        for family in ("conv_states", "recurrent_states")
    }
    observed = {
        (int(row["request_index"]), int(row["layer_index"]), row["family"], int(row["state_index"]))
        for row in rows
    }
    require(observed == expected_keys, "GDN ownership identity drift")
    for row in rows:
        require(len(row.get("request_storage_id", "")) == 64, "GDN request storage token drift")
        require(len(row.get("base_storage_id", "")) == 64, "GDN base storage token drift")
        interval = row.get("request_byte_interval")
        require(isinstance(interval, list) and len(interval) == 2 and 0 <= interval[0] < interval[1], "GDN byte interval drift")
        require(len(row.get("content_sha256", "")) == 64, "GDN content digest drift")
    for row in rows:
        peers = [
            peer
            for peer in rows
            if peer["request_index"] != row["request_index"]
            and peer["layer_index"] == row["layer_index"]
            and peer["family"] == row["family"]
            and peer["state_index"] == row["state_index"]
        ]
        require(len(peers) == 1, "GDN peer cardinality drift")
        recomputed_peer_overlap = sum(receipt_overlap(row, peer) for peer in peers)
        require(recomputed_peer_overlap == row.get("peer_overlap_count"), "GDN peer overlap receipt drift")
        base_overlap = (
            row["request_storage_id"] == row["base_storage_id"]
            and row["request_byte_interval"][0] < row["base_byte_interval"][1]
            and row["base_byte_interval"][0] < row["request_byte_interval"][1]
        )
        require(base_overlap == row.get("base_overlap"), "GDN base overlap receipt drift")
        if setup and borrowed_setup:
            require(row.get("exact_base_alias") is True and base_overlap, "borrowed GDN setup did not alias base")
        else:
            require(row.get("exact_base_alias") is False and not base_overlap, "mutable/materialized GDN state overlaps base")
            require(recomputed_peer_overlap == 0, "mutable/materialized GDN states overlap peers")
    exact_aliases = sum(row.get("exact_base_alias") is True for row in rows)
    require(exact_aliases == snapshot.get("exact_base_alias_count"), "GDN alias count receipt drift")
    require(snapshot.get("all_request_base_disjoint") is (not (setup and borrowed_setup)), "GDN base-disjoint summary drift")
    require(snapshot.get("all_request_peer_disjoint") is (not (setup and borrowed_setup)), "GDN peer-disjoint summary drift")
    return True


def verify_kv_snapshot(snapshot: dict[str, Any], *, shared: bool, setup: bool, step_index: int | None) -> bool:
    layers = snapshot.get("layers")
    require(isinstance(layers, list) and len(layers) == 10, "KV layer count drift")
    require([row.get("layer_index") for row in layers] == list(FULL_LAYERS), "KV layer identity drift")
    for layer in layers:
        source = set(layer.get("source_storage_ids", []))
        require(len(source) == 2, "source K/V storage token count drift")
        requests = layer.get("requests")
        require(isinstance(requests, list) and len(requests) == REQUESTS, "KV request count drift")
        request_sets = [set(row.get("request_storage_ids", [])) for row in requests]
        require(all(len(value) == 2 for value in request_sets), "request K/V storage token count drift")
        if shared:
            require(all(value == source for value in request_sets), "shared request does not use source arena")
        else:
            require(all(not (value & source) for value in request_sets), "fresh request overlaps source arena")
            require(not (request_sets[0] & request_sets[1]), "fresh request arenas overlap")
        reservations = [set(int(value) for value in row["reservation_ids"]) for row in requests]
        if shared:
            require(not (reservations[0] & reservations[1]), "shared private reservations overlap")
        for request in requests:
            require(request.get("shares_source_storage") is shared, "KV shared-storage summary drift")
            active = request.get("active_block_table")
            require(isinstance(active, list) and len(active) >= 32, "KV active block table drift")
            if setup:
                require(request.get("sequence_length") == 4095, "KV setup length drift")
                require(request.get("append_event_count") == 0, "KV setup append count drift")
                require(request.get("tail_is_source_document_block") is True, "KV setup tail is not document block")
                require(request.get("tail_is_private_reservation") is False, "KV setup tail is already private")
            else:
                require(step_index is not None, "KV round index missing")
                require(request.get("sequence_length") == 4095 + 32 + step_index, "KV round length drift")
                require(request.get("append_event_count") == step_index + 1, "KV append event count drift")
                require(request.get("tail_is_source_document_block") is False, "partial tail did not detach")
                require(request.get("tail_is_private_reservation") is True, "partial tail not in private reservation")
        if not setup and shared:
            require(
                requests[0]["active_tail_physical_id"] != requests[1]["active_tail_physical_id"],
                "shared request private tail blocks overlap",
            )
    return True


def verify_repair_receipts(rows: list[dict[str, Any]]) -> bool:
    require(isinstance(rows, list) and len(rows) == 6, "repair receipt denominator drift")
    expected = {(step, request) for step in range(1, GREEDY_STEPS) for request in range(REQUESTS)}
    observed = {(int(row["step_index"]), int(row["request_index"])) for row in rows}
    require(observed == expected, "repair receipt identity drift")
    for row in rows:
        for label in ("primary", "immediate_repeat"):
            receipt = row[label]
            require(receipt.get("schema_version") == REPAIR_SCHEMA, "repair schema drift")
            require(receipt.get("request_index") == row["request_index"], "repair request drift")
            require(receipt.get("resident_count") == REQUESTS, "repair resident count drift")
            require(receipt.get("conv_tensor_count") == 30, "repair tensor count drift")
            require(receipt.get("cloned_tensor_count") in (0, 30), "repair clone count drift")
            require(receipt.get("ownership_only_change") is True, "repair not ownership-only")
            require(receipt.get("fault_id_specialization") is False, "repair fault-ID specialization")
            receipt_rows = receipt.get("rows")
            require(isinstance(receipt_rows, list) and len(receipt_rows) == 30, "repair row count drift")
            require(all(item.get("base_disjoint") is True for item in receipt_rows), "repair base disjointness failed")
            require(all(item.get("all_peers_disjoint") is True for item in receipt_rows), "repair peer disjointness failed")
        require(row["immediate_repeat"].get("cloned_tensor_count") == 0, "repair immediate repeat not idempotent")
    return True


def verify_intercepts(intercepts: list[dict[str, Any]], kv_policy: str) -> bool:
    require(isinstance(intercepts, list) and len(intercepts) == REQUESTS, "intercept request count drift")
    descriptors = []
    for request_index, intercept in enumerate(intercepts):
        require(intercept.get("verified") is True, "fused intercept incomplete")
        require(intercept.get("request_index") == request_index, "intercept request index drift")
        require(intercept.get("resident_count") == REQUESTS, "intercept resident count drift")
        require(intercept.get("request_policy") == kv_policy, "intercept policy drift")
        require(intercept.get("same_unified_attention_kernel") is True, "kernel identity gate failed")
        require(intercept.get("total_calls") == 40, "fused call denominator drift")
        counts = {int(key): int(value) for key, value in intercept.get("counts", {}).items()}
        require(counts == {layer: GREEDY_STEPS for layer in FULL_LAYERS}, "per-layer call count drift")
        calls = intercept.get("calls")
        require(isinstance(calls, list) and len(calls) == 40, "intercept call row drift")
        expected_deltas = [32] * 10 + [1] * 30
        require([int(row["current_append_delta_tokens"]) for row in calls] == expected_deltas, "append schedule drift")
        require(all(row.get("materialized_attention_mask_nbytes") == 0 for row in calls), "attention mask materialized")
        identity = intercept.get("kernel_identity")
        require(isinstance(identity, dict), "kernel identity missing")
        descriptors.append((identity.get("module"), identity.get("qualname"), identity.get("signature"), identity.get("callable_id")))
    require(len(set(descriptors)) == 1, "requests used different unified-attention callables")
    return True


def metric_row(reference: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    require(reference.shape == candidate.shape, "full-vocabulary shape mismatch")
    left = reference.astype(np.float64, copy=False)
    right = candidate.astype(np.float64, copy=False)
    delta = right - left
    reference_norm = float(np.linalg.norm(left))
    candidate_norm = float(np.linalg.norm(right))
    denominator = reference_norm * candidate_norm
    cosine = float(np.dot(left, right) / denominator) if denominator > 0.0 else 1.0
    result = {
        "max_abs": float(np.max(np.abs(delta))),
        "mean_abs": float(np.mean(np.abs(delta))),
        "relative_l2": float(np.linalg.norm(delta) / max(reference_norm, 1e-12)),
        "cosine_distance": float(1.0 - max(-1.0, min(1.0, cosine))),
        "reference_top1_token_id": int(np.argmax(reference)),
        "candidate_top1_token_id": int(np.argmax(candidate)),
        "top1_equal": int(np.argmax(reference)) == int(np.argmax(candidate)),
    }
    require(all(math.isfinite(result[key]) for key in ("max_abs", "mean_abs", "relative_l2", "cosine_distance")), "non-finite full-vocab metric")
    return result


def numeric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(len(rows) == 64, "secondary metric denominator drift")
    fields = ("max_abs", "mean_abs", "relative_l2", "cosine_distance")
    summary = {}
    for field in fields:
        values = [float(row[field]) for row in rows]
        summary[field] = {
            "min": min(values),
            "median": statistics.median(values),
            "mean": statistics.fmean(values),
            "max": max(values),
        }
    summary["top1_equal_count"] = sum(row["top1_equal"] for row in rows)
    summary["top1_equal_rate"] = summary["top1_equal_count"] / len(rows)
    return summary


def replay(args: argparse.Namespace) -> dict[str, Any]:
    inputs = load_bound_json(args.input_manifest, args.expected_input_sha256, INPUT_SCHEMA, "input")
    reference = load_bound_json(args.reference, args.expected_reference_sha256, REFERENCE_SCHEMA, "reference")
    candidate = load_bound_json(args.candidate, args.expected_candidate_sha256, CANDIDATE_SCHEMA, "candidate")
    require(reference.get("input_manifest_sha256") == args.expected_input_sha256, "reference/input binding drift")
    require(candidate.get("input_manifest_sha256") == args.expected_input_sha256, "candidate/input binding drift")
    require(candidate.get("reference_result_sha256") == args.expected_reference_sha256, "candidate/reference binding drift")
    require(candidate.get("reference_logits_or_candidate_objects_consumed") is False, "candidate consumed reference logits/objects")
    boundary = candidate.get("claim_boundary", {})
    require(boundary.get("fixed_runtime_only") is True and boundary.get("runtime_portability_claimed") is False, "runtime claim boundary drift")
    prereq = candidate.get("repair_prerequisite", {})
    require(prereq.get("source_sha256") == args.expected_repair_sha256, "repair source binding drift")
    require(prereq.get("clean_result_sha256") == args.expected_clean_result_sha256, "clean result binding drift")
    require(prereq.get("detached_replay_sha256") == args.expected_detached_replay_sha256, "detached replay binding drift")
    require(prereq.get("clean_regression_passed_before_execution") is True, "clean regression prerequisite missing")

    reference_sidecars = load_sidecars(args.artifact_root, reference.get("sidecars"), expected_count=16, prefix="reference")
    candidate_sidecars = load_sidecars(args.artifact_root, candidate.get("sidecars"), expected_count=128, prefix="candidate")
    reference_rows = reference_replay(inputs, reference, reference_sidecars)
    candidate_rows = candidate.get("rows")
    require(isinstance(candidate_rows, list) and len(candidate_rows) == 16, "candidate row denominator drift")
    candidate_map = {}
    referenced_candidate_sidecars = set()
    exact_rows = []
    secondary_rows = []
    ownership_checks = []
    for row in candidate_rows:
        key = (int(row["case_index"]), row["arm_id"], row["track"])
        require(key not in candidate_map, "candidate row identity duplicate")
        require(key[0] in (0, 1) and key[1] in ARM_IDS and key[2] in TRACKS, "candidate row identity drift")
        candidate_map[key] = row
        shared = key[1].startswith("shared-")
        borrowed = key[1].endswith("-borrowed")
        expected_kv_policy = (
            "vllm-q16-shared-document-reuse" if shared else "vllm-q16-fresh-full-copy-control"
        )
        expected_gdn_policy = (
            "borrow-immutable-base-functional-rebind"
            if borrowed
            else "materialize-request-base-functional-rebind"
        )
        require(row.get("kv_policy") == expected_kv_policy, "candidate KV policy label drift")
        require(row.get("gdn_base_policy") == expected_gdn_policy, "candidate GDN policy label drift")
        require(row.get("source_document_immutable") is True, "source document immutability gate failed")
        require(row.get("source_document_sha256_before") == row.get("source_document_sha256_after"), "source document digest changed")
        require(row.get("persistent_gdn_immutable") is True, "persistent GDN immutability gate failed")
        require(row.get("persistent_gdn_before") == row.get("persistent_gdn_after"), "persistent GDN digest changed")
        verify_gdn_snapshot(row["setup_ownership"]["gdn"], borrowed_setup=borrowed, setup=True)
        verify_kv_snapshot(row["setup_ownership"]["kv"], shared=shared, setup=True, step_index=None)
        rounds = row.get("round_ownership")
        require(isinstance(rounds, list) and len(rounds) == GREEDY_STEPS, "ownership round denominator drift")
        for step_index, round_row in enumerate(rounds):
            require(round_row.get("step_index") == step_index, "ownership round index drift")
            verify_gdn_snapshot(round_row["gdn"], borrowed_setup=False, setup=False)
            verify_kv_snapshot(round_row["kv"], shared=shared, setup=False, step_index=step_index)
        verify_repair_receipts(row.get("repair_receipts"))
        verify_intercepts(row.get("intercepts"), expected_kv_policy)
        trajectories = row.get("trajectories")
        require(isinstance(trajectories, list) and len(trajectories) == REQUESTS, "candidate trajectory count drift")
        for request_index, trajectory in enumerate(trajectories):
            require(trajectory.get("request_index") == request_index, "candidate request index drift")
            reference_row = reference_rows[(key[0], request_index)]
            reference_tokens = [int(value) for value in reference_row["generated_token_ids"]]
            generated = [int(value) for value in trajectory.get("generated_token_ids", [])]
            steps = trajectory.get("steps")
            require(len(generated) == len(steps) == GREEDY_STEPS, "candidate horizon drift")
            case = inputs["cases"][key[0]]
            query = case["queries"][request_index]
            require(trajectory.get("query_token_ids_sha256") == query["token_ids_sha256"], "candidate query binding drift")
            for step_index, step in enumerate(steps):
                require(step.get("step_index") == step_index, "candidate step index drift")
                expected_input = (
                    [int(value) for value in query["token_ids"]]
                    if step_index == 0
                    else [
                        generated[step_index - 1]
                        if key[2] == "greedy"
                        else reference_tokens[step_index - 1]
                    ]
                )
                require(step.get("input_token_count") == len(expected_input), "candidate input length drift")
                require(step.get("input_token_ids_sha256") == int64_sha256(expected_input), "candidate input history drift")
                require(step.get("single_token_repair_applied") is (step_index > 0), "repair timing drift")
                record_id = step.get("logit_record_id")
                require(record_id in candidate_sidecars and record_id not in referenced_candidate_sidecars, "candidate sidecar reference drift")
                token = int(np.argmax(candidate_sidecars[record_id]))
                require(token == generated[step_index] == step.get("candidate_argmax_token_id"), "candidate token/sidecar drift")
                referenced_candidate_sidecars.add(record_id)
                if key[2] == "teacher_forced_reference_history":
                    reference_record_id = reference_row["steps"][step_index]["logit_record_id"]
                    metric = metric_row(reference_sidecars[reference_record_id], candidate_sidecars[record_id])
                    secondary_rows.append(
                        {
                            "case_index": key[0],
                            "request_index": request_index,
                            "arm_id": key[1],
                            "step_index": step_index,
                            "history_matched_by_reference_teacher_forcing": True,
                            "reference_logit_record_id": reference_record_id,
                            "candidate_logit_record_id": record_id,
                            **metric,
                        }
                    )
            if key[2] == "greedy":
                exact_rows.append(
                    {
                        "case_index": key[0],
                        "request_index": request_index,
                        "arm_id": key[1],
                        "reference_generated_token_ids": reference_tokens,
                        "candidate_generated_token_ids": generated,
                        "step_equal": [left == right for left, right in zip(reference_tokens, generated)],
                        "trajectory_exact": reference_tokens == generated,
                    }
                )
        ownership_checks.append({"case_index": key[0], "arm_id": key[1], "track": key[2], "passed": True})
    expected_keys = {(case, arm, track) for case in range(CASES) for arm in ARM_IDS for track in TRACKS}
    require(set(candidate_map) == expected_keys, "candidate factorial denominator drift")
    require(referenced_candidate_sidecars == set(candidate_sidecars), "candidate has orphan sidecars")
    require(len(exact_rows) == 16, "primary trajectory denominator drift")
    require(len(secondary_rows) == 64, "secondary comparison denominator drift")
    exact_decisions = sum(sum(row["step_equal"]) for row in exact_rows)
    exact_gate = exact_decisions == 64 and all(row["trajectory_exact"] for row in exact_rows)
    ownership_gate = all(row["passed"] for row in ownership_checks)
    primary = exact_gate and ownership_gate
    return {
        "schema_version": SCHEMA,
        "status": "completed_independent_replay",
        "input_manifest_sha256": args.expected_input_sha256,
        "reference_result_sha256": args.expected_reference_sha256,
        "candidate_result_sha256": args.expected_candidate_sha256,
        "candidate_or_reference_modules_imported": False,
        "replay_dependencies": ["python-standard-library", "numpy"],
        "infrastructure_valid": True,
        "ownership_gate_passed": ownership_gate,
        "exact_generated_token_gate": {
            "passed": exact_gate,
            "exact_decisions": exact_decisions,
            "total_decisions": 64,
            "exact_trajectories": sum(row["trajectory_exact"] for row in exact_rows),
            "total_trajectories": 16,
            "rows": exact_rows,
        },
        "primary_gate_passed": primary,
        "scientific_outcome": "passed_primary_semantic_control" if primary else "valid_negative_primary_semantic_control",
        "full_vocabulary_secondary": {
            "history_matched": True,
            "acceptance_threshold": None,
            "used_as_primary_gate": False,
            "reported_even_if_unfavorable": True,
            "comparisons": 64,
            "summary": numeric_summary(secondary_rows),
            "rows": secondary_rows,
        },
        "ownership_replay": {
            "passed": ownership_gate,
            "arm_track_checks": len(ownership_checks),
            "rows": ownership_checks,
        },
        "claim_boundary": {
            "fixed_qwen35_h20_runtime_only": True,
            "runtime_portability_claimed": False,
            "hardware_portability_claimed": False,
            "unseen_fault_rate_claimed": False,
            "native_dynamic_batching_claimed": False,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--input-manifest", type=Path, required=True)
    result.add_argument("--expected-input-sha256", required=True)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--expected-reference-sha256", required=True)
    result.add_argument("--candidate", type=Path, required=True)
    result.add_argument("--expected-candidate-sha256", required=True)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--expected-repair-sha256", required=True)
    result.add_argument("--expected-clean-result-sha256", required=True)
    result.add_argument("--expected-detached-replay-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    value = replay(args)
    atomic_write(args.output, canonical_bytes(value))
    print(
        json.dumps(
            {
                "status": value["status"],
                "primary_gate_passed": value["primary_gate_passed"],
                "scientific_outcome": value["scientific_outcome"],
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
