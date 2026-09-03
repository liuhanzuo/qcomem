#!/usr/bin/env python3
"""Audited split/materialized/shared candidate for the R30 semantic control."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForImageTextToText

from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
from qcomem_qwen35_vllm_paged_integration import convert_all_qwen35_full_layers_to_vllm_q16
from qcomem_single_token_gdn_ownership import (
    SCHEMA_VERSION as REPAIR_SCHEMA,
    byte_interval,
    exact_alias,
    overlaps,
    prepare_borrowed_single_token_conv_transition,
)
from qcomem_vllm_paged_fair_control import FRESH_CONTROL, SHARED_REUSE
from qcomem_vllm_paged_kernel import _resolve_vllm_unified_attention
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    GDN_MATERIALIZE_REQUEST_BASE,
    MultiForkHitLedger,
    build_resident_request_group,
    register_multifork_backend,
)
from run_qcomem_qwen35_vllm_paged_multifork_resident import (
    _build_document_cache,
    _last_logits,
    _linear_state_digest,
    _resolve_backbone,
    _set_production_no_mask,
    _source_document_digests,
    _unregister_backend,
)


SCHEMA = "forkaudit-r30-e2e-candidate-v1"
INPUT_SCHEMA = "forkaudit-r30-e2e-input-manifest-v1"
REFERENCE_SCHEMA = "forkaudit-r30-e2e-reference-v1"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
EXPECTED_GPU_UUID = "GPU-d917fce5-80f1-78ac-3965-0476bf8bd441"
REPAIR_SHA256 = "4a2938cc99503f54abf91f780034e08ae64e4105a51c0736433b84ff363bad7a"
PAGE_SIZE = 128
QUERY_TOKENS = 32
GREEDY_STEPS = 4
RESIDENT_COUNT = 2
FULL_LAYERS = tuple(range(3, 40, 4))
ARM_SPECS = (
    ("fresh-materialized", FRESH_CONTROL, GDN_MATERIALIZE_REQUEST_BASE),
    ("fresh-borrowed", FRESH_CONTROL, GDN_BORROW_IMMUTABLE_BASE),
    ("shared-materialized", SHARED_REUSE, GDN_MATERIALIZE_REQUEST_BASE),
    ("shared-borrowed", SHARED_REUSE, GDN_BORROW_IMMUTABLE_BASE),
)
TRACKS = ("greedy", "teacher_forced_reference_history")


class CandidateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CandidateError(message)


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


def int64_sha256(values: Sequence[int]) -> str:
    array = np.asarray([int(value) for value in values], dtype="<i8")
    require(array.ndim == 1, "token array rank drift")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def storage_id(tensor: torch.Tensor, salt: str) -> str:
    storage = tensor.untyped_storage()
    payload = (
        f"{salt}\0{tensor.device}\0{int(storage.data_ptr())}\0{int(storage.nbytes())}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_sidecar(root: Path, record_id: str, logits: torch.Tensor) -> dict[str, Any]:
    array = logits.detach().float().contiguous().cpu().numpy().astype("<f4", copy=False)
    require(array.ndim == 1 and array.size > 1, "candidate logit vector shape drift")
    require(bool(np.isfinite(array).all()), "candidate logits are non-finite")
    relative = Path("candidate") / "logits" / (record_id + ".npy")
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return {
        "record_id": record_id,
        "path": relative.as_posix(),
        "sha256": sha256_file(target),
        "shape": [int(value) for value in array.shape],
        "dtype": "float32",
        "argmax_token_id": int(np.argmax(array)),
    }


def gpu_receipt() -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "4", "candidate GPU isolation drift")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.device_count() == 1, "candidate must see exactly one GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    uuid = "GPU-" + str(getattr(properties, "uuid", ""))
    require(uuid == EXPECTED_GPU_UUID, f"unexpected visible GPU UUID: {uuid}")
    require("H20" in str(properties.name), "visible device is not H20")
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": torch.cuda.device_count(),
        "visible_index": 0,
        "uuid": uuid,
        "name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def load_bound_json(path: Path, expected_sha256: str, schema: str, label: str) -> dict[str, Any]:
    require(sha256_file(path) == expected_sha256, f"{label} SHA drift")
    value = json.loads(path.read_bytes())
    require(value.get("schema_version") == schema, f"{label} schema drift")
    return value


def gdn_content_receipt(owner: Any, layer_indices: Sequence[int]) -> dict[str, Any]:
    digest = hashlib.sha256()
    tensor_count = 0
    for layer_index in layer_indices:
        for family in ("conv_states", "recurrent_states"):
            values = getattr(owner.layers[layer_index], family)
            require(isinstance(values, dict), "GDN family schema drift")
            for state_index in sorted(values):
                tensor = values[state_index]
                require(isinstance(tensor, torch.Tensor), "GDN state is not tensor")
                digest.update(f"{layer_index}:{family}:{state_index}\0".encode())
                digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
                tensor_count += 1
    require(tensor_count == 60, "GDN tensor count drift")
    return {"tensor_count": tensor_count, "sha256": digest.hexdigest()}


def capture_gdn_ownership(
    persistent: Any,
    requests: Sequence[Any],
    layer_indices: Sequence[int],
    *,
    salt: str,
    phase: str,
) -> dict[str, Any]:
    rows = []
    for request_index, request in enumerate(requests):
        for layer_index in layer_indices:
            for family in ("conv_states", "recurrent_states"):
                base_values = getattr(persistent.layers[layer_index], family)
                request_values = getattr(request.layers[layer_index], family)
                require(sorted(base_values) == sorted(request_values), "GDN key drift")
                for state_index in sorted(base_values):
                    base = base_values[state_index]
                    current = request_values[state_index]
                    peers = [
                        getattr(peer.layers[layer_index], family)[state_index]
                        for peer_index, peer in enumerate(requests)
                        if peer_index != request_index
                    ]
                    start, end = byte_interval(current)
                    rows.append(
                        {
                            "request_index": request_index,
                            "layer_index": int(layer_index),
                            "family": family,
                            "state_index": int(state_index),
                            "request_storage_id": storage_id(current, salt),
                            "request_byte_interval": [start, end],
                            "base_storage_id": storage_id(base, salt),
                            "base_byte_interval": list(byte_interval(base)),
                            "content_sha256": tensor_sha256(current),
                            "base_content_sha256": tensor_sha256(base),
                            "exact_base_alias": bool(exact_alias(current, base)),
                            "base_overlap": bool(overlaps(current, base)),
                            "peer_overlap_count": sum(bool(overlaps(current, peer)) for peer in peers),
                        }
                    )
    expected = len(requests) * len(tuple(layer_indices)) * 2
    require(len(rows) == expected, "serialized GDN ownership denominator drift")
    return {
        "phase": phase,
        "request_count": len(requests),
        "tensor_rows": len(rows),
        "all_request_base_disjoint": all(not row["base_overlap"] for row in rows),
        "all_request_peer_disjoint": all(row["peer_overlap_count"] == 0 for row in rows),
        "exact_base_alias_count": sum(row["exact_base_alias"] for row in rows),
        "rows": rows,
    }


def capture_kv_ownership(
    persistent: Any,
    group: Any,
    layer_indices: Sequence[int],
    *,
    salt: str,
    phase: str,
) -> dict[str, Any]:
    layers = []
    for layer_index in layer_indices:
        source_arena = persistent.layers[layer_index].arena
        source_storage_ids = sorted(
            {
                storage_id(source_arena.key_cache, salt),
                storage_id(source_arena.value_cache, salt),
            }
        )
        source_tail = int(source_arena.document_block_table[0, -1].item())
        requests = []
        for request_index, request in enumerate(group.requests):
            sequence = request.layers[layer_index].sequence
            arena = sequence.arena
            active = [int(value) for value in sequence.active_block_table[0].tolist()]
            reservations = [int(value) for value in sequence.reservations[0].tolist()]
            request_storage_ids = sorted(
                {storage_id(arena.key_cache, salt), storage_id(arena.value_cache, salt)}
            )
            requests.append(
                {
                    "request_index": request_index,
                    "sequence_length": int(sequence.sequence_length),
                    "request_storage_ids": request_storage_ids,
                    "shares_source_storage": request_storage_ids == source_storage_ids,
                    "reservation_ids": reservations,
                    "active_block_table": active,
                    "document_tail_physical_id": source_tail,
                    "active_tail_physical_id": int(active[31]),
                    "tail_is_source_document_block": int(active[31]) == source_tail,
                    "tail_is_private_reservation": int(active[31]) in set(reservations),
                    "append_event_count": int(sequence._append_event_count),
                }
            )
        layers.append(
            {
                "layer_index": int(layer_index),
                "source_storage_ids": source_storage_ids,
                "source_document_tail_physical_id": source_tail,
                "requests": requests,
            }
        )
    return {"phase": phase, "layers": layers}


def validate_repair_receipt(value: dict[str, Any], *, request_index: int, repeat: bool) -> None:
    require(value.get("schema_version") == REPAIR_SCHEMA, "repair receipt schema drift")
    require(value.get("request_index") == request_index, "repair request index drift")
    require(value.get("resident_count") == RESIDENT_COUNT, "repair resident count drift")
    require(value.get("conv_tensor_count") == 30, "repair conv denominator drift")
    require(value.get("cloned_tensor_count") in (0, 30), "repair clone count drift")
    if repeat:
        require(value.get("cloned_tensor_count") == 0, "repeat repair was not idempotent")
    require(value.get("ownership_only_change") is True, "repair changed semantics")
    require(value.get("fault_id_specialization") is False, "repair specialized to a fault ID")
    require(len(value.get("rows", [])) == 30, "repair row count drift")
    require(all(row.get("base_disjoint") is True for row in value["rows"]), "repair/base overlap")
    require(all(row.get("all_peers_disjoint") is True for row in value["rows"]), "repair/peer overlap")


def arm_track(
    *,
    model: Any,
    backbone: Any,
    plan: Any,
    document: torch.Tensor,
    queries: Sequence[torch.Tensor],
    reference_tokens: dict[int, list[int]],
    case_index: int,
    arm_id: str,
    kv_policy: str,
    gdn_policy: str,
    track: str,
    kernel: Any,
    salt: str,
    artifact_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(track in TRACKS, "unknown candidate track")
    persistent = _build_document_cache(backbone, document)
    persistent_gdn_before = gdn_content_receipt(persistent, plan.linear_layer_indices)
    conversion = convert_all_qwen35_full_layers_to_vllm_q16(
        persistent,
        plan,
        page_size=PAGE_SIZE,
        max_append_tokens=QUERY_TOKENS + GREEDY_STEPS,
        max_request_forks=RESIDENT_COUNT,
    )
    source_document_before = _source_document_digests(persistent, plan.full_attention_layer_indices)
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=RESIDENT_COUNT,
        policy=kv_policy,
        gdn_base_policy=gdn_policy,
    )
    _set_production_no_mask(group, plan.full_attention_layer_indices)
    setup_gdn = capture_gdn_ownership(
        persistent,
        group.requests,
        plan.linear_layer_indices,
        salt=salt,
        phase="setup-before-query",
    )
    setup_kv = capture_kv_ownership(
        persistent,
        group,
        plan.full_attention_layer_indices,
        salt=salt,
        phase="setup-before-query",
    )
    expected_aliases = 120 if gdn_policy == GDN_BORROW_IMMUTABLE_BASE else 0
    require(setup_gdn["exact_base_alias_count"] == expected_aliases, "GDN setup policy drift")
    if gdn_policy == GDN_MATERIALIZE_REQUEST_BASE:
        require(setup_gdn["all_request_base_disjoint"], "materialized GDN base overlaps source")
        require(setup_gdn["all_request_peer_disjoint"], "materialized GDN requests overlap")

    ledgers = [
        MultiForkHitLedger(
            plan,
            request,
            request_index=request_index,
            resident_count=RESIDENT_COUNT,
            request_policy=kv_policy,
            expected_calls_per_layer=GREEDY_STEPS,
            initial_query_tokens=QUERY_TOKENS,
            kernel=kernel,
        )
        for request_index, request in enumerate(group.requests)
    ]
    backends = [register_multifork_backend(ledger) for ledger in ledgers]
    trajectories = [
        {
            "request_index": request_index,
            "query_token_ids_sha256": int64_sha256(queries[request_index].reshape(-1).tolist()),
            "generated_token_ids": [],
            "steps": [],
        }
        for request_index in range(RESIDENT_COUNT)
    ]
    sidecars: list[dict[str, Any]] = []
    currents = [query for query in queries]
    round_ownership = []
    repair_rows = []
    original_backend = backbone.config._attn_implementation
    try:
        for step_index in range(GREEDY_STEPS):
            for request_index in range(RESIDENT_COUNT):
                repair = None
                repair_repeat = None
                if step_index > 0:
                    repair = prepare_borrowed_single_token_conv_transition(
                        persistent,
                        group.requests,
                        plan.linear_layer_indices,
                        request_index=request_index,
                    )
                    validate_repair_receipt(repair, request_index=request_index, repeat=False)
                    repair_repeat = prepare_borrowed_single_token_conv_transition(
                        persistent,
                        group.requests,
                        plan.linear_layer_indices,
                        request_index=request_index,
                    )
                    validate_repair_receipt(repair_repeat, request_index=request_index, repeat=True)
                    repair_rows.append(
                        {
                            "step_index": step_index,
                            "request_index": request_index,
                            "primary": repair,
                            "immediate_repeat": repair_repeat,
                        }
                    )
                backbone.config._attn_implementation = backends[request_index]
                output = backbone(
                    input_ids=currents[request_index],
                    past_key_values=group.requests[request_index],
                    use_cache=True,
                )
                logits = _last_logits(model, output)[0]
                candidate_token = int(torch.argmax(logits).item())
                record_id = (
                    f"{arm_id}/{track}/case-{case_index}/request-{request_index}"
                    f"/step-{step_index}"
                )
                receipt = write_sidecar(artifact_root, record_id, logits)
                require(receipt["argmax_token_id"] == candidate_token, "candidate sidecar argmax drift")
                sidecars.append(receipt)
                input_token_ids = [int(value) for value in currents[request_index].reshape(-1).tolist()]
                trajectories[request_index]["steps"].append(
                    {
                        "step_index": step_index,
                        "input_token_count": len(input_token_ids),
                        "input_token_ids_sha256": int64_sha256(input_token_ids),
                        "candidate_argmax_token_id": candidate_token,
                        "logit_record_id": record_id,
                        "single_token_repair_applied": step_index > 0,
                    }
                )
                trajectories[request_index]["generated_token_ids"].append(candidate_token)
                next_token = (
                    candidate_token
                    if track == "greedy"
                    else int(reference_tokens[request_index][step_index])
                )
                currents[request_index] = torch.tensor(
                    [[next_token]], dtype=torch.int64, device=document.device
                )
                del output, logits
            round_gdn = capture_gdn_ownership(
                persistent,
                group.requests,
                plan.linear_layer_indices,
                salt=salt,
                phase=f"after-round-{step_index}",
            )
            round_kv = capture_kv_ownership(
                persistent,
                group,
                plan.full_attention_layer_indices,
                salt=salt,
                phase=f"after-round-{step_index}",
            )
            round_ownership.append({"step_index": step_index, "gdn": round_gdn, "kv": round_kv})
    finally:
        backbone.config._attn_implementation = original_backend
        for name in backends:
            _unregister_backend(name)

    source_document_after = _source_document_digests(persistent, plan.full_attention_layer_indices)
    persistent_gdn_after = gdn_content_receipt(persistent, plan.linear_layer_indices)
    intercepts = [ledger.verify_complete() for ledger in ledgers]
    require(len(sidecars) == RESIDENT_COUNT * GREEDY_STEPS, "candidate track sidecar denominator drift")
    row = {
        "case_index": case_index,
        "arm_id": arm_id,
        "track": track,
        "kv_policy": kv_policy,
        "gdn_base_policy": gdn_policy,
        "resident_count": RESIDENT_COUNT,
        "quantization": "Q16",
        "page_size": PAGE_SIZE,
        "document_tokens": int(document.shape[1]),
        "query_tokens": QUERY_TOKENS,
        "greedy_steps": GREEDY_STEPS,
        "teacher_forced_uses_reference_token_history": track == "teacher_forced_reference_history",
        "conversion": {
            "document_payload_nbytes": int(conversion.document_payload_nbytes),
            "allocated_block_pool_nbytes": int(conversion.allocated_block_pool_nbytes),
            "max_request_forks": int(conversion.max_request_forks),
        },
        "group_audit": group.audit,
        "source_document_sha256_before": source_document_before,
        "source_document_sha256_after": source_document_after,
        "source_document_immutable": source_document_before == source_document_after,
        "persistent_gdn_before": persistent_gdn_before,
        "persistent_gdn_after": persistent_gdn_after,
        "persistent_gdn_immutable": persistent_gdn_before == persistent_gdn_after,
        "setup_ownership": {"gdn": setup_gdn, "kv": setup_kv},
        "round_ownership": round_ownership,
        "repair_receipts": repair_rows,
        "repair_source_sha256": REPAIR_SHA256,
        "intercepts": intercepts,
        "trajectories": trajectories,
        "denominators": {
            "trajectories": 2,
            "model_steps": 8,
            "full_vocab_sidecars": 8,
            "repair_primary_receipts": 6,
            "repair_repeat_receipts": 6,
            "fused_attention_calls": 80,
        },
    }
    del group, persistent, ledgers, currents
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return row, sidecars


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(args.output.resolve().parent == args.artifact_root.resolve(), "output must be at artifact root")
    require(not args.output.exists(), "candidate output already exists")
    require(sha256_file(args.repair_source) == REPAIR_SHA256, "repair source SHA drift")
    inputs = load_bound_json(
        args.input_manifest, args.expected_input_sha256, INPUT_SCHEMA, "input manifest"
    )
    reference = load_bound_json(
        args.reference, args.expected_reference_sha256, REFERENCE_SCHEMA, "reference result"
    )
    require(reference.get("input_manifest_sha256") == args.expected_input_sha256, "reference input binding drift")
    require(reference.get("candidate_cache_trace_tensor_objects_imported") is False, "reference imported candidate state")
    require(reference.get("full_model_recompute_each_step") is True, "reference is not dense recompute")
    reference_map = {
        (int(row["case_index"]), int(row["request_index"])): [int(value) for value in row["generated_token_ids"]]
        for row in reference["rows"]
    }
    require(len(reference_map) == 4, "reference trajectory cardinality drift")
    require(all(len(value) == GREEDY_STEPS for value in reference_map.values()), "reference horizon drift")
    require(len(inputs["cases"]) == 2, "candidate case count drift")
    require(int(inputs["selection"]["greedy_steps"]) == GREEDY_STEPS, "candidate horizon drift")

    gpu = gpu_receipt()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    backbone = _resolve_backbone(model)
    plan = audit_qwen35_functional_stack_plan(model)
    require(tuple(plan.full_attention_layer_indices) == FULL_LAYERS, "full-layer plan drift")
    require(len(tuple(plan.linear_layer_indices)) == 30, "GDN layer plan drift")
    kernel = _resolve_vllm_unified_attention()
    kernel_identity = {
        "module": str(getattr(kernel, "__module__", type(kernel).__module__)),
        "qualname": str(getattr(kernel, "__qualname__", type(kernel).__qualname__)),
    }
    salt = hashlib.sha256(("r30-e2e-storage\0" + args.expected_input_sha256).encode()).hexdigest()
    rows = []
    sidecars = []
    with torch.inference_mode():
        for case in inputs["cases"]:
            case_index = int(case["case_index"])
            document_ids = [int(value) for value in case["document_token_ids"]]
            require(int64_sha256(document_ids) == case["document_token_ids_sha256"], "document digest drift")
            require(len(document_ids) == 4095 and len(document_ids) % PAGE_SIZE == 127, "tail-stress geometry drift")
            document = torch.tensor([document_ids], dtype=torch.int64, device="cuda:0")
            query_tensors = []
            for query in case["queries"]:
                query_ids = [int(value) for value in query["token_ids"]]
                require(int64_sha256(query_ids) == query["token_ids_sha256"], "query digest drift")
                require(len(query_ids) == QUERY_TOKENS, "query length drift")
                query_tensors.append(torch.tensor([query_ids], dtype=torch.int64, device="cuda:0"))
            require(len(query_tensors) == RESIDENT_COUNT, "request count drift")
            reference_tokens = {
                request_index: reference_map[(case_index, request_index)]
                for request_index in range(RESIDENT_COUNT)
            }
            for arm_id, kv_policy, gdn_policy in ARM_SPECS:
                for track in TRACKS:
                    row, receipts = arm_track(
                        model=model,
                        backbone=backbone,
                        plan=plan,
                        document=document,
                        queries=query_tensors,
                        reference_tokens=reference_tokens,
                        case_index=case_index,
                        arm_id=arm_id,
                        kv_policy=kv_policy,
                        gdn_policy=gdn_policy,
                        track=track,
                        kernel=kernel,
                        salt=salt,
                        artifact_root=args.artifact_root,
                    )
                    rows.append(row)
                    sidecars.extend(receipts)
            del document, query_tensors
    require(len(rows) == 16, "candidate arm-track denominator drift")
    require(len(sidecars) == 128, "candidate sidecar denominator drift")
    return {
        "schema_version": SCHEMA,
        "status": "completed_audited_candidate",
        "input_manifest_sha256": args.expected_input_sha256,
        "reference_result_sha256": args.expected_reference_sha256,
        "reference_fields_consumed": ["case_index", "request_index", "generated_token_ids"],
        "reference_logits_or_candidate_objects_consumed": False,
        "repair_prerequisite": {
            "source_sha256": REPAIR_SHA256,
            "clean_result_sha256": args.clean_result_sha256,
            "detached_replay_sha256": args.detached_replay_sha256,
            "clean_regression_passed_before_execution": True,
        },
        "model": {
            "model_id": "Qwen/Qwen3.5-35B-A3B",
            "revision": MODEL_REVISION,
            "dtype": "torch.bfloat16",
            "local_files_only": True,
        },
        "gpu": gpu,
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(sys.modules["transformers"].__version__),
            "numpy": str(np.__version__),
        },
        "kernel_identity": kernel_identity,
        "arm_order": [row[0] for row in ARM_SPECS],
        "track_order": list(TRACKS),
        "rows": rows,
        "sidecars": sidecars,
        "denominators": {
            "cases": 2,
            "candidate_arms": 4,
            "tracks_per_arm": 2,
            "greedy_decisions": 64,
            "history_matched_full_vocab_comparisons": 64,
            "all_candidate_full_vocab_sidecars": 128,
        },
        "claim_boundary": {
            "fixed_runtime_only": True,
            "runtime_portability_claimed": False,
            "hardware_portability_claimed": False,
            "single_stream_round_major": True,
            "native_dynamic_batching_claimed": False,
        },
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--input-manifest", type=Path, required=True)
    result.add_argument("--expected-input-sha256", required=True)
    result.add_argument("--reference", type=Path, required=True)
    result.add_argument("--expected-reference-sha256", required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--repair-source", type=Path, required=True)
    result.add_argument("--clean-result-sha256", required=True)
    result.add_argument("--detached-replay-sha256", required=True)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    value = run(args)
    atomic_write(args.output, canonical_bytes(value))
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "rows": len(value["rows"]),
                "sidecars": len(value["sidecars"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
