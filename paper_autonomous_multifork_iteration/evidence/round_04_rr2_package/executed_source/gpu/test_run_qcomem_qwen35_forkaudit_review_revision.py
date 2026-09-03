from __future__ import annotations

import copy
import functools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import run_qcomem_qwen35_forkaudit_review_revision as runner
import qcomem_forkaudit_model_load_lease as model_lease
from qcomem_forkaudit_mutants import (
    AppliedMutation,
    EXPECTED_GATE_IDS,
    MUTANT_IDS,
    RuntimeInvariantError,
    TargetMutationBinding,
    callback_injector,
    run_clean_case,
    run_mutant_case,
)
from qcomem_forkaudit_oracle import (
    OraclePreregistration,
    OracleThresholds,
    fp32_dense_attention_reference,
)
from qcomem_forkaudit_storage_witness import GDNStorageWitnessError


def digest(label: str) -> str:
    return runner.sha256_json({"label": label})


def allocator_snapshot() -> dict[str, int]:
    return {
        "current_allocated_bytes": 100,
        "current_reserved_bytes": 200,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }


def gpu_assignment_receipt() -> dict[str, object]:
    rows = [
        {
            "rank": rank,
            "visible_index": rank,
            "uuid": f"GPU-00000000-0000-0000-0000-{rank:012d}",
            "name": "NVIDIA H20",
            "total_memory_mib": 97871,
            "compute_capability": [9, 0],
            "bf16_supported": True,
        }
        for rank in range(runner.FORMAL_WORLD_SIZE)
    ]
    return {
        "schema_version": runner.GPU_ASSIGNMENT_RECEIPT_SCHEMA_VERSION,
        "world_size": runner.FORMAL_WORLD_SIZE,
        "inventory_query": "index,uuid,name,memory.total,compute_cap",
        "rows": rows,
        "unique_visible_indices": True,
        "unique_uuids": True,
        "all_h20": True,
        "all_compute_capability_9_0": True,
        "generated_before_candidate_outputs": True,
    }


def model_load_evidence(
    run_id: str = "0" * 32,
) -> dict[str, object]:
    weight_rows = [
        {
            "logical_name": f"model.safetensors-{index:05d}-of-00014.safetensors",
            "sha256": digest(f"weight-{index}"),
        }
        for index in range(1, 15)
    ]
    artifact_rows = [
        {"logical_name": "config.json", "sha256": digest("config")}
    ]

    def ledger_raw(rows):
        return "".join(
            f"{row['sha256']}  {row['logical_name']}\n" for row in rows
        ).encode("utf-8")

    weight_raw = ledger_raw(weight_rows)
    artifact_raw = ledger_raw(artifact_rows)
    manifest_rows = []
    combined = [("config.json", "model_artifact", artifact_rows[0]["sha256"])]
    combined.extend(
        (row["logical_name"], "model_weight", row["sha256"])
        for row in weight_rows
    )
    for offset, (name, role, declared) in enumerate(combined, start=1):
        manifest_rows.append(
            {
                "relative_path": name,
                "ledger_roles": [role],
                "declared_sha256": declared,
                "bytes": 1024 + offset,
                "copy_mode": "ficlone" if offset % 2 else "byte-copy",
                "source_device": 10,
                "source_inode": 1000 + offset,
                "view_device": 20,
                "view_inode": 2000 + offset,
                "source_and_view_inode_distinct": True,
            }
        )
    manifest = {
        "schema_version": runner.PRIVATE_MODEL_VIEW_SCHEMA_VERSION,
        "model_id": runner.FORMAL_MODEL_ID,
        "model_revision": runner.FORMAL_MODEL_REVISION,
        "model_artifact_ledger_raw_sha256": runner.sha256_bytes(artifact_raw),
        "model_weight_ledger_raw_sha256": runner.sha256_bytes(weight_raw),
        "copy_policy": "ficlone-then-byte-copy;hardlink-and-symlink-forbidden",
        "file_count": len(manifest_rows),
        "weight_file_count": 14,
        "all_source_and_view_inodes_distinct": True,
        "all_view_files_regular": True,
        "all_view_files_read_only": True,
        "rows": manifest_rows,
        "generated_before_candidate_outputs": True,
        "cuda_initialized": False,
    }
    manifest_raw = runner.canonical_json_bytes(manifest) + b"\n"
    manifest_sha = runner.sha256_bytes(manifest_raw)
    manifest_by_name = {row["relative_path"]: row for row in manifest_rows}
    authority_rows = []
    for index, weight in enumerate(weight_rows, start=1):
        manifest_row = manifest_by_name[weight["logical_name"]]
        authority_rows.append(
            {
                "logical_name": weight["logical_name"],
                "declared_sha256": weight["sha256"],
                "observed_sha256": weight["sha256"],
                "stat": {
                    "bytes": manifest_row["bytes"],
                    "st_dev": manifest_row["view_device"],
                    "st_ino": manifest_row["view_inode"],
                    "mode": 0o444,
                    "mtime_ns": 3000 + index,
                    "ctime_ns": 4000 + index,
                },
                "lease_state": "read",
            }
        )
    authority = {
        "schema_version": model_lease.AUTHORITY_SCHEMA_VERSION,
        "run_id": run_id,
        "threat_model": model_lease.THREAT_MODEL,
        "view_policy": "private-independent-inode-no-symlink-reflink-or-copy",
        "weight_ledger_raw_sha256": runner.sha256_bytes(weight_raw),
        "model_artifact_ledger_raw_sha256": runner.sha256_bytes(artifact_raw),
        "model_view_manifest_sha256": manifest_sha,
        "entry_count": 14,
        "rows": authority_rows,
        "rows_sha256": model_lease.sha256_json(authority_rows),
        "all_content_matches_ledger": True,
        "all_regular_no_symlink": True,
        "all_read_only": True,
        "all_linux_read_leases_active": True,
        "sigio_handler_installed": True,
        "process_thread_count_at_authority": 1,
        "lease_break_count_at_authority": 0,
    }
    authority_raw = model_lease.canonical_receipt_bytes(authority)
    authority_sha = runner.sha256_bytes(authority_raw)
    closure_rows = [
        {
            "logical_name": row["logical_name"],
            "observed_sha256": row["observed_sha256"],
            "stat": copy.deepcopy(row["stat"]),
            "lease_state": "read",
        }
        for row in authority_rows
    ]
    closure = {
        "schema_version": model_lease.CLOSURE_SCHEMA_VERSION,
        "run_id": run_id,
        "threat_model": model_lease.THREAT_MODEL,
        "authority_raw_sha256": authority_sha,
        "entry_count": 14,
        "rows": closure_rows,
        "rows_sha256": model_lease.sha256_json(closure_rows),
        "lease_break_count": 0,
        "sigio_pending_during_terminal": False,
        "sigio_handler_unchanged_until_release": True,
        "all_leases_remained_read": True,
        "all_final_stats_equal_authority": True,
        "all_final_content_equal_authority": True,
        "all_leases_released": True,
        "all_fds_closed": True,
        "lease_release_error_count": 0,
        "fd_close_error_count": 0,
        "terminal_full_content_rehash_performed": True,
        "invalid_reasons": [],
        "passed": True,
    }
    closure_raw = model_lease.canonical_receipt_bytes(closure)
    return {
        "weight_rows": weight_rows,
        "artifact_rows": artifact_rows,
        "weight_raw": weight_raw,
        "artifact_raw": artifact_raw,
        "manifest": manifest,
        "manifest_raw": manifest_raw,
        "manifest_sha": manifest_sha,
        "authority": authority,
        "authority_raw": authority_raw,
        "authority_sha": authority_sha,
        "closure": closure,
        "closure_raw": closure_raw,
        "closure_sha": runner.sha256_bytes(closure_raw),
    }


def live_input_lifetime_receipts(
    bank: dict[str, object], baseline: dict[str, int] | None = None
) -> list[dict[str, object]]:
    frozen = allocator_snapshot() if baseline is None else baseline
    rows = [
        {
            "request_index": row["request_index"],
            "sha256": row["query_token_ids_sha256"],
        }
        for row in bank["rows"]
    ]
    return [
        {
            "schema_version": runner.LIVE_INPUT_LIFETIME_SCHEMA_VERSION,
            "capture_point": point,
            "device": "cuda:0",
            "dtype": "torch.int64",
            "document_shape": [1, runner.FORMAL_DOCUMENT_TOKENS],
            "document_token_ids_sha256": bank["document_token_ids_sha256"],
            "query_shape": [1, runner.FORMAL_QUERY_TOKENS],
            "query_count": max(runner.FORMAL_RESIDENT_COUNTS),
            "query_rows": copy.deepcopy(rows),
            "allocator_before": copy.deepcopy(frozen),
            "allocator_after": copy.deepcopy(frozen),
            "current_allocator_exactly_equal_to_frozen_baseline": True,
        }
        for point in runner.LIVE_INPUT_CAPTURE_POINTS
    ]


def timeline_replay_fixture() -> dict[str, object]:
    return {
        "completed_all_requests": True,
        "normalized_gdn_factor_projection": {"fixture": "gdn"},
        "normalized_kv_factor_projection": {"fixture": "kv"},
    }


def identity() -> dict[str, object]:
    value = {field: digest(field) for field in runner.FROZEN_SHA256_FIELDS}
    value["pg19_data_sha256"] = runner.FORMAL_PG19_DATA_SHA256
    value["pg19_manifest_sha256"] = runner.FORMAL_PG19_MANIFEST_SHA256
    value["pg19_windows_sha256"] = runner.FORMAL_RR2_WINDOWS_SHA256
    value["prior_fp32_context_manifest_sha256"] = (
        runner.PRIOR_FP32_CONTEXT_RAW_SHA256
    )
    value["review_response_plan_sha256"] = (
        runner.FINAL_REVIEW_RESPONSE_PLAN_SHA256
    )
    value["protocol_config_sha256"] = runner.sha256_json(
        runner.formal_protocol_config()
    )
    value["model_id"] = runner.FORMAL_MODEL_ID
    value["model_revision"] = runner.FORMAL_MODEL_REVISION
    return value


def selection_plan() -> list[dict[str, object]]:
    return [
        {
            "selection_rule_id": "rank-frozen-heldout-post-rope-v1",
            "rank": rank,
            "book_index": rank,
            "source_object": f"train/{10000 + rank}.txt",
            "window_index": rank,
            "document_start_token": rank * runner.FORMAL_WINDOW_STRIDE,
            "document_length": runner.FORMAL_DOCUMENT_TOKENS,
            "document_token_ids_sha256": digest(f"document-{rank}"),
            "layer_index": runner.FORMAL_FULL_LAYERS[rank],
            "request_index": 0,
            "round_index": rank % runner.FORMAL_GENERATION_STEPS,
            "sample_id": f"heldout-rank-{rank}",
            "kv_policy": runner.ORACLE_KV_POLICY,
            "gdn_base_policy": runner.ORACLE_GDN_BASE_POLICY,
            "cell_role": "ownership_witness",
            "arm_id": (
                f"kv={runner.ORACLE_KV_POLICY}|"
                f"gdn={runner.ORACLE_GDN_BASE_POLICY}"
            ),
            "oracle_cell_id": (
                f"rank-{rank}-N-1-kv={runner.ORACLE_KV_POLICY}|"
                f"gdn={runner.ORACLE_GDN_BASE_POLICY}-ownership-witness"
            ),
            "held_out_from_threshold_calibration": True,
            "locked_before_candidate_outputs": True,
        }
        for rank in range(runner.FORMAL_WORLD_SIZE)
    ]


def frozen_query_banks() -> list[dict[str, object]]:
    selections = selection_plan()
    result = []
    for rank, selection in enumerate(selections):
        rows = [
            {
                "request_index": request_index,
                "source_token_offset": (
                    selection["document_start_token"]
                    + runner.FORMAL_DOCUMENT_TOKENS
                    + runner.FORMAL_QUERY_TOKENS
                    + request_index * runner.FORMAL_QUERY_BANK_STRIDE
                ),
                "query_tokens": runner.FORMAL_QUERY_TOKENS,
                "query_token_ids_sha256": digest(f"query-{request_index}"),
            }
            for request_index in range(max(runner.FORMAL_RESIDENT_COUNTS))
        ]
        bank = {
                "rank": rank,
                "book_index": rank,
                "source_id": str(10000 + rank),
                "source_object": selection["source_object"],
                "window_index": selection["window_index"],
                "document_start_token": selection["document_start_token"],
                "document_end_token_exclusive": (
                    selection["document_start_token"]
                    + runner.FORMAL_DOCUMENT_TOKENS
                ),
                "document_token_ids_sha256": selection[
                    "document_token_ids_sha256"
                ],
                "query_bank_start_token": (
                    selection["document_start_token"]
                    + runner.FORMAL_DOCUMENT_TOKENS
                    + runner.FORMAL_QUERY_TOKENS
                ),
                "query_stride_tokens": runner.FORMAL_QUERY_BANK_STRIDE,
                "query_tokens": runner.FORMAL_QUERY_TOKENS,
                "count": max(runner.FORMAL_RESIDENT_COUNTS),
                "query_bank_sha256": digest(f"query-bank-{rank}"),
                "rows": rows,
            }
        bank["manifest_sha256"] = runner.sha256_json(bank)
        result.append(bank)
    return result


def clean_outcome_dict() -> dict[str, object]:
    return run_clean_case(lambda _context: None).to_dict()


@functools.lru_cache(maxsize=None)
def coverage_gdn_snapshot(mutant_id: str, failing: bool) -> dict[str, object]:
    layers = []
    for layer_index in range(40):
        layers.append(
            SimpleNamespace(
                conv_states={0: torch.full((1, 1, 2), float(layer_index))},
                recurrent_states={0: torch.full((1, 1, 1, 2), float(100 + layer_index))},
            )
        )
    persistent = SimpleNamespace(layers=layers)
    guard = runner.capture_persistent_gdn_guard(
        persistent, runner.FORMAL_LINEAR_LAYERS
    )
    requests = []
    for _ in range(2):
        request_layers = []
        for source in persistent.layers:
            request_layers.append(
                SimpleNamespace(
                    conv_states={0: source.conv_states[0]},
                    recurrent_states={0: source.recurrent_states[0]},
                )
            )
        requests.append(SimpleNamespace(layers=request_layers))
    completed = [0] if mutant_id == "M4" else [0, 1]
    for request_index in completed:
        for layer_index in runner.FORMAL_LINEAR_LAYERS:
            source = persistent.layers[layer_index]
            requests[request_index].layers[layer_index].conv_states[0] = (
                source.conv_states[0].clone()
            )
            requests[request_index].layers[layer_index].recurrent_states[0] = (
                source.recurrent_states[0].clone()
            )
    if failing:
        first = runner.FORMAL_LINEAR_LAYERS[0]
        if mutant_id == "M4":
            requests[0].layers[first].conv_states[0] = persistent.layers[
                first
            ].conv_states[0]
        else:
            requests[1].layers[first].conv_states[0] = requests[0].layers[
                first
            ].conv_states[0]
    return runner.capture_gdn_storage_snapshot(
        persistent,
        requests,
        runner.FORMAL_LINEAR_LAYERS,
        phase=runner.PHASE_POST_TRANSITION,
        policy=runner.GDN_BORROW_IMMUTABLE_BASE,
        persistent_guard=guard,
        completed_request_indices=completed,
    )


def coverage_full_forward_completion(
    request_index: int = 0,
) -> dict[str, object]:
    calls = [
        {
            "layer_idx": layer,
            "query_tokens": runner.FORMAL_QUERY_TOKENS,
            "kv_tokens": runner.FORMAL_DOCUMENT_TOKENS
            + runner.FORMAL_QUERY_TOKENS,
            "kernel_mode": runner.FORMAL_KERNEL_MODE,
            "quantization": "Q16",
            "mask_contract": runner.FORMAL_MASK_CONTRACT,
            "position_ids_contract": runner.FORMAL_POSITION_CONTRACT,
            "position_ids_expected_tail_start": runner.FORMAL_DOCUMENT_TOKENS,
            "position_ids_expected_tail_end_exclusive": (
                runner.FORMAL_DOCUMENT_TOKENS + runner.FORMAL_QUERY_TOKENS
            ),
            "softmax_scale": runner.FORMAL_SOFTMAX_SCALE,
            "append_event_index": 0,
            "appended_tokens_before": 0,
            "appended_tokens_after": runner.FORMAL_QUERY_TOKENS,
        }
        for layer in runner.FORMAL_FULL_LAYERS
    ]
    return {
        "schema_version": "forkaudit-mutant-full-forward-coverage-v1",
        "verified": True,
        "request_index": request_index,
        "resident_count": 2,
        "request_policy": runner.SHARED_REUSE,
        "initial_query_tokens": runner.FORMAL_QUERY_TOKENS,
        "total_calls": len(calls),
        "counts": {str(layer): 1 for layer in runner.FORMAL_FULL_LAYERS},
        "same_unified_attention_kernel": True,
        "dense_fallback_calls": 0,
        "full_kv_concatenations": 0,
        "kernel_identity": {
            "module": runner.FORMAL_KERNEL_DESCRIPTOR[0],
            "qualname": runner.FORMAL_KERNEL_DESCRIPTOR[1],
            "signature": runner.FORMAL_KERNEL_DESCRIPTOR[2],
        },
        "calls": calls,
    }


def coverage_direct_completion() -> dict[str, object]:
    return {
        "schema_version": "forkaudit-mutant-direct-call-coverage-v1",
        "call_count_before": 0,
        "call_count_after": 1,
        "layer_index": runner.FORMAL_FULL_LAYERS[0],
        "request_index": 0,
        "resident_count": 2,
        "request_policy": runner.SHARED_REUSE,
        "query_tokens": 1,
        "kv_tokens": runner.FORMAL_DOCUMENT_TOKENS + 1,
        "kernel_mode": runner.FORMAL_KERNEL_MODE,
        "quantization": "Q16",
        "mask_contract": runner.FORMAL_MASK_CONTRACT,
        "position_ids_contract": runner.FORMAL_POSITION_CONTRACT,
        "position_ids_expected_tail_start": runner.FORMAL_DOCUMENT_TOKENS,
        "position_ids_expected_tail_end_exclusive": runner.FORMAL_DOCUMENT_TOKENS
        + 1,
        "softmax_scale": runner.FORMAL_SOFTMAX_SCALE,
        "append_event_index": 0,
        "appended_tokens_before": 0,
        "appended_tokens_after": 1,
        "kernel_identity": {
            "module": runner.FORMAL_KERNEL_DESCRIPTOR[0],
            "qualname": runner.FORMAL_KERNEL_DESCRIPTOR[1],
            "signature": runner.FORMAL_KERNEL_DESCRIPTOR[2],
        },
    }


def coverage_detector_input(
    mutant_id: str, *, mutation_activated: bool, outcome: dict[str, object]
) -> dict[str, object]:
    if mutant_id == "M1":
        target_descriptor = {
            "schema_version": "forkaudit-live-target-tensor-v1",
            "logical_slot": (
                f"request1/layer{runner.FORMAL_FULL_LAYERS[0]}/reservations"
            ),
            "binding_role": (
                "request0-reservation-aliased-into-request1"
                if mutation_activated
                else "request1-construction-reservation"
            ),
            "dtype": "torch.int32",
            "shape": [1, 2],
            "tensor_sha256": digest(
                "M1-request0-reservations"
                if mutation_activated
                else "M1-request1-reservations"
            ),
        }
        evidence = {
            "kind": "live-kv-ownership",
            "require_appended_tail_cow": False,
            "full_attention_layers": list(runner.FORMAL_FULL_LAYERS),
            "target_reservations_sha256": target_descriptor[
                "tensor_sha256"
            ],
            "peer_request0_reservations_sha256": digest(
                "M1-request0-reservations"
            ),
            "target_descriptor": target_descriptor,
            "target_descriptor_sha256": runner.sha256_json(target_descriptor),
            "construction_guard_row_count": 2 * len(runner.FORMAL_FULL_LAYERS),
        }
    elif mutant_id == "M3":
        evidence = {
            "kind": "live-kv-ownership",
            "require_appended_tail_cow": True,
            "full_attention_layers": list(runner.FORMAL_FULL_LAYERS),
            "appended_tokens_by_request_layer": [
                {"request_index": request, "layer_index": layer, "appended_tokens": 1}
                for request in range(2)
                for layer in runner.FORMAL_FULL_LAYERS
            ],
            "all_request_layers_appended_once": True,
        }
    elif mutant_id in ("M2", "M8", "M9"):
        evidence = {
            "kind": "live-full-model-forward-ledger",
            "backend_registered": True,
            "request_index": 0,
            "initial_query_tokens": runner.FORMAL_QUERY_TOKENS,
            "expected_calls_per_layer": 1,
            "expected_full_attention_layers": list(runner.FORMAL_FULL_LAYERS),
            "call_count_before": 0,
            "query_token_ids_sha256": digest("query-0"),
        }
    elif mutant_id in ("M4", "M5"):
        failing = bool(outcome["observed_gate_id"])
        witness = copy.deepcopy(coverage_gdn_snapshot(mutant_id, failing))
        evidence = {
            "kind": "live-gdn-storage-replay",
            "phase": runner.PHASE_POST_TRANSITION,
            "completed_request_indices": [0] if mutant_id == "M4" else [0, 1],
            "storage_witness": witness,
            "storage_witness_sha256": runner.sha256_json(witness),
            "transition_forward_ledgers": [
                coverage_full_forward_completion(request_index)
                for request_index in ([0] if mutant_id == "M4" else [0, 1])
            ],
        }
    else:
        position = runner.FORMAL_DOCUMENT_TOKENS + (
            1 if mutation_activated and mutant_id == "M6" else 0
        )
        materialized = mutation_activated and mutant_id == "M7"
        mask = (
            torch.ones(
                (1, 1, 1, runner.FORMAL_DOCUMENT_TOKENS + 1), dtype=torch.bool
            )
            if materialized
            else None
        )
        evidence = {
            "kind": "live-direct-ledger-call",
            "layer_index": runner.FORMAL_FULL_LAYERS[0],
            "call_count_before": 0,
            "appended_tokens": 1,
            "query_sha256": runner._tensor_digest(
                torch.zeros(
                    (
                        1,
                        runner.FORMAL_NUM_QUERY_HEADS,
                        1,
                        runner.FORMAL_HEAD_DIM,
                    ),
                    dtype=torch.bfloat16,
                )
            ),
            "query_dtype": "torch.bfloat16",
            "query_shape": [1, runner.FORMAL_NUM_QUERY_HEADS, 1, runner.FORMAL_HEAD_DIM],
            "position_ids_values": [position],
            "position_ids_sha256": runner._tensor_digest(
                torch.tensor([[position]], dtype=torch.int64)
            ),
            "attention_mask_representation": (
                "materialized-tensor" if materialized else "none"
            ),
            "attention_mask_sha256": (
                runner._tensor_digest(mask) if materialized else None
            ),
            "attention_mask_dtype": "torch.bool" if materialized else None,
            "attention_mask_shape": list(mask.shape) if materialized else None,
        }
    return {
        "schema_version": "forkaudit-mutant-detector-input-v2",
        "mutant_id": mutant_id,
        "detector_path": runner.MUTANT_EXERCISE_PATHS[mutant_id],
        "expected_gate_id": runner.MUTANT_SPECS[mutant_id].expected_gate_id,
        "resident_count": 2,
        "kv_policy": runner.SHARED_REUSE,
        "gdn_base_policy": runner.GDN_BORROW_IMMUTABLE_BASE,
        "evidence": evidence,
    }


def coverage_receipt(
    mutant_id: str,
    outcome: dict[str, object],
    *,
    mutation_activated: bool = True,
) -> dict[str, object]:
    detector_input = (
        coverage_detector_input(
            mutant_id, mutation_activated=mutation_activated, outcome=outcome
        )
        if outcome["exercise_started"]
        else None
    )
    completion = None
    if outcome["exercise_completed"]:
        if mutant_id in ("M1", "M3"):
            completion = {
                "passed": True,
                "gate_ids": [
                    "KV_SEQUENCE_ID",
                    "KV_RESERVATION_DISJOINT",
                    "KV_TAIL_COW",
                    "KV_ACTIVE_BLOCK_OWNERSHIP",
                ],
                "resident_count": 2,
                "require_appended_tail_cow": mutant_id == "M3",
                "construction_binding_verified": True,
            }
        elif mutant_id in ("M2", "M8", "M9"):
            completion = coverage_full_forward_completion()
        elif mutant_id in ("M4", "M5"):
            completion = runner.replay_gdn_storage_witness(
                detector_input["evidence"]["storage_witness"]
            )
        else:
            completion = coverage_direct_completion()
    return {
        "schema_version": "forkaudit-mutant-exercise-coverage-v2",
        "mutant_id": mutant_id,
        "mutation_activated": mutation_activated,
        "exercise_contract_sha256": runner._mutant_exercise_contract_sha256(
            mutant_id
        ),
        "detector_path": runner.MUTANT_EXERCISE_PATHS[mutant_id],
        "exercise_started": outcome["exercise_started"],
        "detector_input": detector_input,
        "detector_input_sha256": (
            runner.sha256_json(detector_input) if detector_input is not None else None
        ),
        "detector_path_completed": outcome["exercise_completed"],
        "completion_receipt": completion,
        "completion_receipt_sha256": (
            runner.sha256_json(completion) if completion is not None else None
        ),
        "outcome_classification": outcome["classification"],
        "observed_gate_id": outcome["observed_gate_id"],
    }


def mutant_outcome(mutant_id: str, case_cell_id: str, mode: str = "detected"):
    kind, field = runner.MUTANT_TARGET_CONTRACT[mutant_id]
    if mutant_id == "M1":
        context = {
            "target": {
                "schema_version": "forkaudit-live-target-tensor-v1",
                "logical_slot": (
                    f"request1/layer{runner.FORMAL_FULL_LAYERS[0]}/reservations"
                ),
                "binding_role": "request1-construction-reservation",
                "dtype": "torch.int32",
                "shape": [1, 2],
                "tensor_sha256": digest("M1-request1-reservations"),
            }
        }
    else:
        context = {"target": {"kind": kind, "field": field, "value": "pre"}}

    def descriptor_sha(state):
        return runner.sha256_json(state["target"])

    def apply(state):
        before = descriptor_sha(state)
        original = dict(state["target"])
        if mutant_id == "M1":
            state["target"]["binding_role"] = (
                "request0-reservation-aliased-into-request1"
            )
            state["target"]["tensor_sha256"] = digest(
                "M1-request0-reservations"
            )
        else:
            state["target"]["value"] = "mutated"
        mutated = descriptor_sha(state)

        def undo():
            state["target"] = original

        def verify():
            return state["target"] == original

        return AppliedMutation(
            undo,
            verify,
            target_binding=TargetMutationBinding(
                mutant_id=mutant_id,
                case_cell_id=case_cell_id,
                capture_id=f"capture-{case_cell_id}",
                target_kind=kind,
                target_field=field,
                pre_sha256=before,
                mutated_sha256=mutated,
                capture_restored_sha256=lambda: descriptor_sha(state),
            ),
        )

    injector = callback_injector(apply)
    if mode == "detected":
        def exercise(_context):
            raise RuntimeInvariantError(EXPECTED_GATE_IDS[mutant_id], "detected")
    elif mode == "escape":
        def exercise(_context):
            return None
    elif mode == "crash":
        def exercise(_context):
            raise OSError("unrelated crash")
    else:  # pragma: no cover
        raise AssertionError(mode)
    return run_mutant_case(mutant_id, injector, exercise, context=context)


def fault_campaign(rank: int, modes: dict[str, str] | None = None) -> dict[str, object]:
    modes = {} if modes is None else modes
    rows = {}
    for mutant_id in runner.MUTANT_ASSIGNMENT_BY_RANK[rank]:
        cell_id = f"mutant-r{rank}-{mutant_id}"
        matched_cell_id = f"matched-clean-r{rank}-{mutant_id}"
        exercise_sha = runner._mutant_exercise_contract_sha256(mutant_id)
        outcome = mutant_outcome(
            mutant_id, cell_id, modes.get(mutant_id, "detected")
        ).to_dict()
        matched_outcome = clean_outcome_dict()
        rows[mutant_id] = {
            "mutant_id": mutant_id,
            "exercise_mutant_id": mutant_id,
            "exercise_contract_sha256": exercise_sha,
            "exercise_coverage_receipt": coverage_receipt(mutant_id, outcome),
            "case_cell_id": cell_id,
            "case_isolation": {
                "fresh_document_cache_built": True,
                "fresh_request_cache_built": True,
                "cache_reused_from_prior_case": False,
                "cache_discarded_after_case": True,
            },
            "cleanup_receipt": cleanup_receipt(),
            "outcome": outcome,
            "matched_clean": {
                "exercise_mutant_id": mutant_id,
                "exercise_contract_sha256": exercise_sha,
                "exercise_coverage_receipt": coverage_receipt(
                    mutant_id, matched_outcome, mutation_activated=False
                ),
                "case_cell_id": matched_cell_id,
                "case_isolation": {
                    "fresh_document_cache_built": True,
                    "fresh_request_cache_built": True,
                    "cache_reused_from_prior_case": False,
                    "cache_discarded_after_case": True,
                },
                "cleanup_receipt": cleanup_receipt(),
                "outcome": matched_outcome,
            },
            "matched_clean_exercise_passed": True,
        }
    global_clean = clean_outcome_dict()
    return {
        "assignment": list(runner.MUTANT_ASSIGNMENT_BY_RANK[rank]),
        "clean_case": {
            "case_cell_id": f"global-clean-r{rank}",
            "case_isolation": {
                "fresh_document_cache_built": True,
                "fresh_request_cache_built": True,
                "cache_reused_from_prior_case": False,
                "cache_discarded_after_case": True,
            },
            "cleanup_receipt": cleanup_receipt(),
            "outcome": global_clean,
        },
        "mutants": rows,
    }


def semantic_rows(count: int, *, salt: str = "common") -> list[dict[str, object]]:
    rows = []
    for request in range(count):
        rows.append(
            {
                "request_index": request,
                "query_token_ids_sha256": digest(f"query-{request}"),
                "generated_token_ids": list(range(8)),
                "full_vocab_step_logit_sha256": [
                    digest(f"logit-{salt}-{request}-{step}") for step in range(8)
                ],
                "logical_kv_sha256": {
                    str(layer): digest(f"kv-{salt}-{request}-{layer}")
                    for layer in runner.FORMAL_FULL_LAYERS
                },
                "final_gdn_sha256": digest(f"gdn-{salt}-{request}"),
            }
        )
    return rows


def kernel_ledgers(
    count: int, kv_policy: str, *, strict_position_values: bool
) -> list[dict[str, object]]:
    identity_row = {
        "module": runner.FORMAL_KERNEL_DESCRIPTOR[0],
        "qualname": runner.FORMAL_KERNEL_DESCRIPTOR[1],
        "signature": runner.FORMAL_KERNEL_DESCRIPTOR[2],
    }
    calls = []
    for round_index in range(runner.FORMAL_GENERATION_STEPS):
        for layer in runner.FORMAL_FULL_LAYERS:
            calls.append(
                {
                    "layer_idx": layer,
                    "request_index": None,
                    "resident_count": count,
                    "request_policy": kv_policy,
                    "protocol": runner.MULTIFORK_PROTOCOL,
                    "kernel_identity": dict(identity_row),
                    "current_append_delta_tokens": (
                        runner.FORMAL_QUERY_TOKENS if round_index == 0 else 1
                    ),
                    "query_tokens": (
                        runner.FORMAL_QUERY_TOKENS if round_index == 0 else 1
                    ),
                    "kv_tokens": (
                        runner.FORMAL_DOCUMENT_TOKENS
                        + runner.FORMAL_QUERY_TOKENS
                        + round_index
                    ),
                    "fused_gpu_kernel_calls": 1,
                    "full_kv_concatenations": 0,
                    "full_document_staging_copy_nbytes": 0,
                    "kernel_mode": runner.FORMAL_KERNEL_MODE,
                    "quantization": "Q16",
                    "gqa_groups": runner.FORMAL_GQA_GROUPS,
                    "physical_block_pool_shape": (
                        (
                            runner.FORMAL_DOCUMENT_BLOCKS
                            + runner.FORMAL_PRIVATE_BLOCKS_PER_REQUEST
                        )
                        if kv_policy == runner.FRESH_CONTROL
                        else (
                            runner.FORMAL_DOCUMENT_BLOCKS
                            + count * runner.FORMAL_PRIVATE_BLOCKS_PER_REQUEST
                        ),
                        runner.FORMAL_PAGE_SIZE,
                        runner.FORMAL_NUM_KV_HEADS,
                        runner.FORMAL_HEAD_DIM,
                    ),
                    "active_block_table_shape": (1, 33),
                    "partial_tail_staging_copy_nbytes": (
                        runner.FORMAL_PARTIAL_TAIL_COPY_NBYTES
                    ),
                    "mask_contract": runner.FORMAL_MASK_CONTRACT,
                    "materialized_attention_mask_nbytes": 0,
                    "mask_validation_host_syncs": 0,
                    "position_ids_contract": runner.FORMAL_POSITION_CONTRACT,
                    "position_ids_validated": True,
                    "position_ids_semantically_consumed_upstream": True,
                    "position_ids_shape": (
                        1,
                        runner.FORMAL_QUERY_TOKENS if round_index == 0 else 1,
                    ),
                    "position_ids_dtype": "torch.int64",
                    "position_ids_expected_tail_start": (
                        runner.FORMAL_DOCUMENT_TOKENS
                        if round_index == 0
                        else runner.FORMAL_DOCUMENT_TOKENS
                        + runner.FORMAL_QUERY_TOKENS
                        + round_index
                        - 1
                    ),
                    "position_ids_expected_tail_end_exclusive": (
                        runner.FORMAL_DOCUMENT_TOKENS
                        + runner.FORMAL_QUERY_TOKENS
                        + round_index
                    ),
                    "position_ids_strict_tail_values_checked": strict_position_values,
                    "position_ids_validation_host_syncs": (
                        1 if strict_position_values else 0
                    ),
                    "append_capture_id": None,
                    "append_audit": {
                        "append_event_index": round_index,
                        "append_tokens": (
                            runner.FORMAL_QUERY_TOKENS
                            if round_index == 0
                            else 1
                        ),
                        "appended_tokens_before": (
                            0
                            if round_index == 0
                            else runner.FORMAL_QUERY_TOKENS + round_index - 1
                        ),
                        "appended_tokens_after": (
                            runner.FORMAL_QUERY_TOKENS + round_index
                        ),
                        "sequence_length_before": (
                            runner.FORMAL_DOCUMENT_TOKENS
                            + (
                                0
                                if round_index == 0
                                else runner.FORMAL_QUERY_TOKENS
                                + round_index
                                - 1
                            )
                        ),
                        "sequence_length_after": (
                            runner.FORMAL_DOCUMENT_TOKENS
                            + runner.FORMAL_QUERY_TOKENS
                            + round_index
                        ),
                        "capture_id": None,
                    },
                    "softmax_scale": runner.FORMAL_SOFTMAX_SCALE,
                }
            )
    rows = []
    for request_index in range(count):
        request_calls = copy.deepcopy(calls)
        for call_index, call in enumerate(request_calls):
            call["request_index"] = request_index
            if strict_position_values:
                capture_id = (
                    f"fixture-N-{count}-request-{request_index}-call-{call_index}"
                )
                call["append_capture_id"] = capture_id
                call["append_audit"]["capture_id"] = capture_id
                values = list(
                    range(
                        call["position_ids_expected_tail_start"],
                        call["position_ids_expected_tail_end_exclusive"],
                    )
                )
                call["position_ids_values"] = values
                call["position_ids_sha256"] = runner._tensor_digest(
                    torch.tensor([values], dtype=torch.int64)
                )
        rows.append({
            "verified": True,
            "protocol": runner.MULTIFORK_PROTOCOL,
            "request_index": request_index,
            "resident_count": count,
            "request_policy": kv_policy,
            "kernel_mode": runner.FORMAL_KERNEL_MODE,
            "same_unified_attention_kernel": True,
            "strict_position_values": strict_position_values,
            "initial_query_tokens": runner.FORMAL_QUERY_TOKENS,
            "round_major_request_local_layer_order_verified": True,
            "mask_contract": runner.FORMAL_MASK_CONTRACT,
            "position_ids_contract": runner.FORMAL_POSITION_CONTRACT,
            "call_observer_enabled": strict_position_values,
            "kernel_identity": dict(identity_row),
            "calls": request_calls,
            "total_calls": len(calls),
            "counts": {
                layer: runner.FORMAL_GENERATION_STEPS
                for layer in runner.FORMAL_FULL_LAYERS
            },
            "dense_fallback_calls": 0,
            "full_kv_concatenations": 0,
        })
    return rows


def group_audit(kv_policy: str, gdn_policy: str, count: int) -> dict[str, object]:
    borrowed = gdn_policy == runner.GDN_BORROW_IMMUTABLE_BASE
    fresh = kv_policy == runner.FRESH_CONTROL
    document_copy = (
        len(runner.FORMAL_FULL_LAYERS) * runner.FORMAL_DOCUMENT_ALLOCATED_NBYTES
        if fresh
        else 0
    )
    pool_bytes = (
        len(runner.FORMAL_FULL_LAYERS)
        * (
            runner.FORMAL_DOCUMENT_ALLOCATED_NBYTES
            + runner.FORMAL_PRIVATE_BLOCKS_PER_REQUEST
            * runner.FORMAL_BLOCK_NBYTES
        )
        if fresh
        else 0
    )
    rows = []
    for request_index in range(count):
        row = {
            "request_index": request_index,
            "document_block_copy_nbytes_including_padding": document_copy,
            "allocated_request_pool_nbytes": pool_bytes,
            "source_document_storage_shared": not fresh,
            "gdn_base": {
                "policy": gdn_policy,
                "tensor_count": 60,
                "borrowed_immutable_base_alias_count": 60 if borrowed else 0,
                "materialized_request_base_nbytes": 0 if borrowed else 6000,
                "functional_rebind_after_transition": True,
            },
        }
        if fresh:
            row.update({
                "document_payload_nbytes": (
                    len(runner.FORMAL_FULL_LAYERS)
                    * runner.FORMAL_DOCUMENT_PAYLOAD_NBYTES
                ),
                "layers": [
                    {
                        "layer_idx": layer,
                        "document_block_copy_nbytes_including_padding": (
                            runner.FORMAL_DOCUMENT_ALLOCATED_NBYTES
                        ),
                        "document_payload_nbytes": (
                            runner.FORMAL_DOCUMENT_PAYLOAD_NBYTES
                        ),
                        "copied_padding_nbytes": (
                            runner.FORMAL_DOCUMENT_PADDING_NBYTES
                        ),
                        "allocated_request_pool_nbytes": (
                            runner.FORMAL_DOCUMENT_ALLOCATED_NBYTES
                            + runner.FORMAL_PRIVATE_BLOCKS_PER_REQUEST
                            * runner.FORMAL_BLOCK_NBYTES
                        ),
                        "source_storage_shared": False,
                    }
                    for layer in runner.FORMAL_FULL_LAYERS
                ],
            })
        rows.append(row)
    ownership = {
        "passed": True,
        "resident_count": count,
        "request_object_ids_pairwise_distinct": True,
        "request_sequence_ids_pairwise_distinct": True,
        "private_physical_reservation_ids_pairwise_disjoint": not fresh,
        "fresh_private_id_namespace_is_per_arena": fresh,
        "reuse_requests_share_source_arena": not fresh,
        "fresh_request_arena_storages_pairwise_disjoint": fresh,
        "all_requests_strongly_referenced": True,
    }
    return {
        "protocol": runner.MULTIFORK_PROTOCOL,
        "policy": kv_policy,
        "gdn_base_policy": gdn_policy,
        "resident_count": count,
        "all_requests_materialized_before_measurement": True,
        "strong_reference_count": count,
        "rows": rows,
        "ownership": ownership,
        "physical_document_block_copy_nbytes_including_padding": count * document_copy,
        "allocated_fresh_request_pool_nbytes": count * pool_bytes,
    }


def storage_breakdown(kv_policy: str, count: int) -> dict[str, object]:
    rows = []
    for layer in runner.FORMAL_FULL_LAYERS:
        source_document = runner.FORMAL_DOCUMENT_ALLOCATED_NBYTES
        source_padding = runner.FORMAL_DOCUMENT_PADDING_NBYTES
        source_private = (
            count
            * runner.FORMAL_PRIVATE_BLOCKS_PER_REQUEST
            * runner.FORMAL_BLOCK_NBYTES
        )
        fresh = kv_policy == runner.FRESH_CONTROL
        rows.append({
            "layer_idx": layer,
            "resident_count": count,
            "block_bytes": runner.FORMAL_BLOCK_NBYTES,
            "valid_document_payload_nbytes": runner.FORMAL_DOCUMENT_PAYLOAD_NBYTES,
            "source_document_allocated_nbytes": source_document,
            "source_document_padding_nbytes": source_padding,
            "source_private_reservation_nbytes": source_private,
            "source_total_arena_allocated_nbytes": source_document + source_private,
            "fresh_duplicate_document_allocated_nbytes": count * source_document if fresh else 0,
            "fresh_duplicate_document_padding_nbytes": count * source_padding if fresh else 0,
            "fresh_duplicate_private_reservation_nbytes": source_private if fresh else 0,
            "active_request_private_payload_nbytes": (
                2
                * count
                * (
                    runner.FORMAL_DOCUMENT_TOKENS % runner.FORMAL_PAGE_SIZE
                    + runner.FORMAL_FINAL_APPENDED_TOKENS
                )
                * runner.FORMAL_NUM_KV_HEADS
                * runner.FORMAL_HEAD_DIM
                * runner.FORMAL_ELEMENT_BYTES
            ),
            "active_request_private_allocated_page_nbytes": source_private,
            "active_request_private_blocks": count * 2,
            "request_private_reserved_unused_nbytes": 0,
            "active_request_appended_tokens_sum": count * runner.FORMAL_FINAL_APPENDED_TOKENS,
            "active_request_detached_tail_tokens_sum": count * (runner.FORMAL_DOCUMENT_TOKENS % runner.FORMAL_PAGE_SIZE),
            "partial_tail_staging_copy_nbytes": (
                count * runner.FORMAL_PARTIAL_TAIL_COPY_NBYTES
            ),
            "request_block_table_accelerator_nbytes": (
                count
                * (
                    runner.FORMAL_DOCUMENT_BLOCKS
                    + runner.FORMAL_PRIVATE_BLOCKS_PER_REQUEST
                )
                * 4
            ),
            "source_document_table_accelerator_nbytes": (
                runner.FORMAL_DOCUMENT_BLOCKS * 4
            ),
            "fresh_document_table_accelerator_nbytes": (
                count * runner.FORMAL_DOCUMENT_BLOCKS * 4 if fresh else 0
            ),
            "source_cpu_reservation_metadata_nbytes": (
                count * runner.FORMAL_PRIVATE_BLOCKS_PER_REQUEST * 8
            ),
            "fresh_cpu_reservation_metadata_nbytes": (
                count * runner.FORMAL_PRIVATE_BLOCKS_PER_REQUEST * 8
                if fresh
                else 0
            ),
            "physical_document_block_copy_nbytes_including_padding": count * source_document if fresh else 0,
        })
    counted = {
        "active_request_private_blocks",
        "active_request_appended_tokens_sum",
        "active_request_detached_tail_tokens_sum",
        "physical_document_block_copy_nbytes_including_padding",
    }
    totals = {}
    for row in rows:
        for key, value in row.items():
            if key.endswith("_nbytes") or key in counted:
                totals[key] = totals.get(key, 0) + value
    return {
        "protocol": runner.MULTIFORK_PROTOCOL,
        "policy": kv_policy,
        "resident_count": count,
        "simultaneous_lifetime": True,
        "full_attention_layer_count": len(runner.FORMAL_FULL_LAYERS),
        "source_private_reservation_is_common_pack_capacity": True,
        "active_private_payload_is_subset_not_additive": True,
        "fresh_duplicate_pool_is_separate_from_source": kv_policy == runner.FRESH_CONTROL,
        "totals": totals,
        "layers": rows,
    }


def allocator_receipt(kv_policy: str, gdn_policy: str, count: int) -> dict[str, object]:
    baseline = {
        "current_allocated_bytes": 100,
        "current_reserved_bytes": 200,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }
    setup = {
        "current_allocated_bytes": 120,
        "current_reserved_bytes": 220,
        "peak_allocated_bytes": 130,
        "peak_reserved_bytes": 230,
    }
    generation = {
        "current_allocated_bytes": 125,
        "current_reserved_bytes": 225,
        "peak_allocated_bytes": 150,
        "peak_reserved_bytes": 250,
    }
    storage = storage_breakdown(kv_policy, count)
    diagnostic_rows = [
        {
            "round_index": round_index,
            "request_index": request_index,
            "before_current_allocated_bytes": 125,
            "before_current_reserved_bytes": 225,
            "after_current_allocated_bytes": 125,
            "after_current_reserved_bytes": 225,
            "cpu_logits_dtype": "torch.float32",
            "cpu_logits_shape": [1, 100],
            "cpu_logits_sha256": digest(
                f"logit-common-{request_index}-{round_index}"
            ),
            "finite_check_on_cpu": True,
            "allocator_state_exactly_unchanged": True,
        }
        for round_index in range(runner.FORMAL_GENERATION_STEPS)
        for request_index in range(count)
    ]
    return {
        "schema_version": "qcomem-formal-allocator-receipt-v4",
        "baseline": baseline,
        "after_setup": setup,
        "after_setup_diagnostics": copy.deepcopy(setup),
        "after_generation": generation,
        "generation_diagnostics": {
            "schema_version": "qcomem-generation-cpu-diagnostics-v1",
            "resident_count": count,
            "rounds": runner.FORMAL_GENERATION_STEPS,
            "schedule": "round-major-request-minor",
            "single_cpu_clone_per_step": True,
            "gpu_finite_or_hash_kernels_after_endpoint_sample": False,
            "rows": diagnostic_rows,
            "rows_sha256": runner.sha256_json(diagnostic_rows),
        },
        "peak_reset_before_setup": True,
        "peak_reset_before_generation": True,
        "synchronized_before_each_snapshot": True,
        "model_weights_loaded_before_baseline": True,
        "diagnostic_cpu_copies_excluded_from_peak": True,
        "diagnostic_current_allocator_state_unchanged": True,
        "setup_plus_generation_peak_allocated_delta_bytes": 50,
        "setup_plus_generation_peak_reserved_delta_bytes": 50,
        "generation_peak_allocated_delta_bytes": 30,
        "generation_peak_reserved_delta_bytes": 30,
        "storage_breakdown": storage,
        "storage_breakdown_sha256": runner.sha256_json(storage),
        "unique_storage_removed_from_authorizing_payload": True,
    }


def cleanup_receipt() -> dict[str, object]:
    baseline = {
        "current_allocated_bytes": 100,
        "current_reserved_bytes": 200,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }
    return {
        "schema_version": "qcomem-cell-cleanup-receipt-v1",
        "before_cell": copy.deepcopy(baseline),
        "after_cleanup": copy.deepcopy(baseline),
        "frozen_model_query_baseline": copy.deepcopy(baseline),
        "explicit_python_references_dropped_on_return": True,
        "gc_collect_completed": True,
        "cuda_empty_cache_completed": True,
        "cuda_synchronize_completed": True,
        "current_allocated_and_reserved_exactly_recovered": True,
    }


def factorial_cell(rank: int, count: int, arm_index: int) -> dict[str, object]:
    kv_policy = runner.KV_POLICIES[arm_index // 2]
    gdn_policy = runner.GDN_BASE_POLICIES[arm_index % 2]
    arm_id = f"kv={kv_policy}|gdn={gdn_policy}"
    source = {str(layer): digest(f"source-rank-{rank}-{layer}") for layer in runner.FORMAL_FULL_LAYERS}
    source_payload = {
        str(layer): digest(f"source-payload-rank-{rank}-{layer}")
        for layer in runner.FORMAL_FULL_LAYERS
    }
    return {
        "arm_id": arm_id,
        "kv_policy": kv_policy,
        "gdn_base_policy": gdn_policy,
        "memory_cell": {
            "cell_role": "formal_memory",
            "rank": rank,
            "resident_count": count,
            "arm_id": arm_id,
            "cell_id": f"memory-r{rank}-n{count}-a{arm_index}",
            "request_guard_created": False,
            "witness_capture_executed": False,
            "primary_memory_endpoint_eligible": True,
            "cleanup_receipt": cleanup_receipt(),
            "allocator_receipt": allocator_receipt(kv_policy, gdn_policy, count),
            "policy_execution_receipt": {
                "builder": "build_resident_request_group",
                "kv_policy": kv_policy,
                "gdn_base_policy": gdn_policy,
                "resident_count": count,
                "group_audit_sha256": runner.sha256_json(
                    group_audit(kv_policy, gdn_policy, count)
                ),
                "group_audit": group_audit(kv_policy, gdn_policy, count),
                "all_requests_materialized_before_measurement": True,
                "all_requests_alive_through_generation": True,
            },
        },
        "witness_cell": {
            "cell_role": "ownership_witness",
            "rank": rank,
            "resident_count": count,
            "arm_id": arm_id,
            "cell_id": f"witness-r{rank}-n{count}-a{arm_index}",
            "request_guard_created": True,
            "witness_capture_executed": True,
            "primary_memory_endpoint_eligible": False,
            "rebuilt_persistent_cache": True,
            "rebuilt_request_group": True,
            "cleanup_receipt": cleanup_receipt(),
            "timeline_manifest_artifact": {"fixture": True},
        },
        "source_physical_document_sha256_before": source,
        "source_physical_document_sha256_after": source,
        "memory_source_physical_document_sha256_before": copy.deepcopy(source),
        "memory_source_physical_document_sha256_after": copy.deepcopy(source),
        "witness_source_physical_document_sha256_before": copy.deepcopy(source),
        "witness_source_physical_document_sha256_after": copy.deepcopy(source),
        "source_digest_scope": "complete-physical-document-blocks-including-tail-padding",
        "source_physical_payload_sha256": source_payload,
        "memory_source_physical_payload_sha256": copy.deepcopy(source_payload),
        "witness_source_physical_payload_sha256": copy.deepcopy(source_payload),
        "source_payload_digest_scope": (
            "key-value-document-block-bytes-including-tail-padding-"
            "excluding-arena-capacity-metadata"
        ),
        "memory_kernel_ledgers": kernel_ledgers(
            count, kv_policy, strict_position_values=False
        ),
        "witness_kernel_ledgers": kernel_ledgers(
            count, kv_policy, strict_position_values=True
        ),
        "semantics": semantic_rows(count),
        "witness_semantics": semantic_rows(count),
    }


def memory_matrix_rows(rank: int) -> list[dict[str, object]]:
    rows = []
    for count in runner.FORMAL_RESIDENT_COUNTS:
        for arm_index, arm_id in enumerate(runner.ARM_IDS):
            kv_policy = runner.KV_POLICIES[arm_index // 2]
            gdn_policy = runner.GDN_BASE_POLICIES[arm_index % 2]
            normalized_group = runner._validate_group_audit(
                group_audit(kv_policy, gdn_policy, count),
                resident_count=count,
                kv_policy=kv_policy,
                gdn_base_policy=gdn_policy,
            )
            rows.append(
                {
                    "rank": rank,
                    "resident_count": count,
                    "arm_id": arm_id,
                    "kv_policy": kv_policy,
                    "gdn_base_policy": gdn_policy,
                    "allocator_endpoints": {
                        field: 1000 * count + 100 * arm_index + rank
                        for field in runner.MEMORY_ENDPOINT_FIELDS
                    },
                    "q16_analytic_totals": storage_breakdown(
                        kv_policy, count
                    )["totals"],
                    "group_kv_receipt": normalized_group["kv"],
                    "group_gdn_receipt": normalized_group["gdn"],
                }
            )
    return rows


class StaticAndBoundaryTest(unittest.TestCase):
    def test_token_id_digest_matches_rr2_raw_int64_bytes(self):
        tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        expected = runner.sha256_bytes(
            tokens.contiguous().view(torch.uint8).numpy().tobytes()
        )
        self.assertEqual(
            runner._token_id_sha256(tokens, expected_shape=(1, 4)), expected
        )
        self.assertNotEqual(runner._tensor_digest(tokens), expected)
        with self.assertRaisesRegex(runner.ReviewAuditError, "torch.int64"):
            runner._token_id_sha256(tokens.to(torch.int32))

    def test_unique_storage_diagnostic_is_optional_and_non_authorizing(self):
        authoritative = storage_breakdown(runner.SHARED_REUSE, 1)
        absent = copy.deepcopy(authoritative)
        arbitrary = {**copy.deepcopy(authoritative), "unique_storage": "garbage"}
        forged = {
            **copy.deepcopy(authoritative),
            "unique_storage": {
                "fabricated_combined_unique_accelerator_nbytes": 10**30
            },
        }
        self.assertEqual(
            runner._authorizing_storage_breakdown(absent), authoritative
        )
        self.assertEqual(
            runner._authorizing_storage_breakdown(arbitrary), authoritative
        )
        self.assertEqual(
            runner._authorizing_storage_breakdown(forged), authoritative
        )

    def test_static_locks_protocol_identity_and_oracle_selection(self):
        static = runner.make_static_artifact(
            identity(), selection_plan(), frozen_query_banks()
        )
        replay = runner.validate_static_artifact(static)
        self.assertFalse(static["formal_ready"])
        self.assertEqual(
            replay["oracle_selection_plan"][3]["sample_id"], "heldout-rank-3"
        )
        bad = copy.deepcopy(static)
        bad["oracle_selection_plan"][0]["round_index"] = 7
        with self.assertRaisesRegex(runner.ReviewAuditError, "selection-plan SHA"):
            runner.validate_static_artifact(bad)

    def test_formal_static_binds_raw_rr2_prior_context_and_response_plan(self):
        repository = Path(__file__).resolve().parents[1]
        prior_raw = (
            repository
            / "paper_autonomous_multifork_iteration/evidence/"
            "forkaudit_fp32_calibration_manifest.json"
        ).read_bytes()
        response_raw = (
            repository
            / "paper_autonomous_multifork_iteration/review/"
            "experiment_response_plan.json"
        ).read_bytes()
        rr2_payload = {
            "oracle_selection_plan": selection_plan(),
            "frozen_query_banks": frozen_query_banks(),
        }
        rr2_raw = runner.canonical_json_bytes(rr2_payload) + b"\n"
        frozen = identity()
        frozen["pg19_input_manifest_sha256"] = runner.sha256_bytes(rr2_raw)
        with mock.patch(
            "build_qcomem_forkaudit_rr2_input_manifest.validate_rr2_input_manifest",
            return_value=rr2_payload,
        ):
            static = runner.make_static_artifact(
                frozen,
                selection_plan(),
                frozen_query_banks(),
                rr2_input_manifest_raw=rr2_raw,
                prior_fp32_context_manifest_raw=prior_raw,
                review_response_plan_raw=response_raw,
            )
            replay = runner.validate_static_artifact(static)
            self.assertTrue(replay["formal_input_provenance_bound"])
            tampered = copy.deepcopy(static)
            tampered["input_provenance"][
                "rr2_input_manifest_raw_base64"
            ] = tampered["input_provenance"][
                "rr2_input_manifest_raw_base64"
            ][:-4] + "AAAA"
            with self.assertRaisesRegex(runner.ReviewAuditError, "raw-byte SHA"):
                runner.validate_static_artifact(tampered)

    def test_shard_and_formal_aggregate_cli_fail_closed(self):
        with self.assertRaises(runner.ProductionLoopNotImplemented):
            runner.run_shard_not_implemented()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.json"
            with self.assertRaisesRegex(
                runner.ReviewAuditError, "formal shard missing explicit inputs"
            ):
                runner.main(
                    [
                        "--stage",
                        "shard",
                        "--output",
                        str(output),
                        "--run-id",
                        "0123456789abcdef0123456789abcdef",
                    ]
                )
            self.assertFalse(output.exists())
            with self.assertRaisesRegex(runner.ReviewAuditError, "128 bits"):
                runner.main(
                    [
                        "--stage",
                        "shard",
                        "--output",
                        str(output),
                        "--run-id",
                        "not-a-shared-run-id",
                    ]
                )

    def test_gdn_and_kv_axes_are_not_conflated(self):
        self.assertEqual(
            runner.GDN_POLICY_TO_WITNESS[runner.GDN_BORROW_IMMUTABLE_BASE],
            runner.POLICY_SHARED_BASE,
        )
        self.assertEqual(
            runner.GDN_POLICY_TO_WITNESS[runner.GDN_MATERIALIZE_REQUEST_BASE],
            runner.POLICY_MATERIALIZED,
        )
        self.assertFalse(set(runner.KV_POLICIES) & set(runner.GDN_BASE_POLICIES))

    def test_formal_runtime_receipt_uses_real_moe_type_and_executed_kernel(self):
        plan = SimpleNamespace(
            metadata=lambda: {
                "model_type": "qwen3_5_moe_text",
                "kernel_mode": "reference_python_two_pass_paged_softmax",
            }
        )
        observed = runner._formal_functional_stack_metadata(plan)
        self.assertEqual(observed["model_type"], runner.FORMAL_MODEL_TYPE)
        self.assertEqual(observed["kernel_mode"], runner.FORMAL_KERNEL_MODE)
        bad = SimpleNamespace(
            metadata=lambda: {
                "model_type": "qwen3_5_text",
                "kernel_mode": "reference_python_two_pass_paged_softmax",
            }
        )
        with self.assertRaisesRegex(runner.ReviewAuditError, "model type"):
            runner._formal_functional_stack_metadata(bad)

    def test_named_witness_and_oracle_errors_bridge_by_gate_id(self):
        with self.assertRaises(RuntimeInvariantError) as caught:
            runner.bridge_named_gate_error(
                lambda: (_ for _ in ()).throw(
                    GDNStorageWitnessError("alias", gate_id="gdn_gate")
                )
            )
        self.assertEqual(caught.exception.gate_id, "gdn_gate")

    def test_strict_json_rejects_duplicates_and_nonfinite_constants(self):
        with self.assertRaisesRegex(runner.ReviewAuditError, "duplicate JSON key"):
            runner.strict_json_loads(b'{"x":1,"x":2}', label="fixture")
        with self.assertRaisesRegex(runner.ReviewAuditError, "non-finite"):
            runner.strict_json_loads(b'{"x":NaN}', label="fixture")


class FactorialSchemaTest(unittest.TestCase):
    def make_shard(self) -> dict[str, object]:
        return {
            "factorial": [
                {
                    "resident_count": count,
                    "cells": [factorial_cell(0, count, arm) for arm in range(4)],
                }
                for count in runner.FORMAL_RESIDENT_COUNTS
            ]
        }

    def test_strict_position_evidence_and_frozen_scale_fail_closed(self):
        valid = kernel_ledgers(
            1, runner.SHARED_REUSE, strict_position_values=True
        )
        runner._validate_kernel_ledgers(
            valid,
            resident_count=1,
            kv_policy=runner.SHARED_REUSE,
            label="strict fixture",
            strict_position_values=True,
        )
        cases = []
        wrong_values = copy.deepcopy(valid)
        wrong_values[0]["calls"][0]["position_ids_values"] = [-1] * 32
        cases.append((wrong_values, "exact position"))
        missing_digest = copy.deepcopy(valid)
        del missing_digest[0]["calls"][0]["position_ids_sha256"]
        cases.append((missing_digest, "call row schema"))
        extra_field = copy.deepcopy(valid)
        extra_field[0]["calls"][0]["position_ids_cpu"] = [-1] * 32
        cases.append((extra_field, "call row schema"))
        wrong_scale = copy.deepcopy(valid)
        wrong_scale[0]["calls"][0]["softmax_scale"] = 1.0
        cases.append((wrong_scale, "frozen head_dim"))
        wrong_kernel = copy.deepcopy(valid)
        wrong_kernel[0]["kernel_identity"]["qualname"] = "lookalike_kernel"
        for call in wrong_kernel[0]["calls"]:
            call["kernel_identity"]["qualname"] = "lookalike_kernel"
        cases.append((wrong_kernel, "frozen vLLM callable"))
        for payload, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(runner.ReviewAuditError, message):
                    runner._validate_kernel_ledgers(
                        payload,
                        resident_count=1,
                        kv_policy=runner.SHARED_REUSE,
                        label="strict fixture",
                        strict_position_values=True,
                    )

    @mock.patch.object(
        runner,
        "_validate_timeline_manifest",
        return_value=(timeline_replay_fixture(), []),
    )
    def test_four_cells_and_cross_n_are_exact(self, _timeline):
        exact, _bindings, _semantics, _source, _oracle, _baseline, _matrix = runner._validate_factorial(
            self.make_shard(), root=Path("."), rank=0, run_id="run"
        )
        self.assertTrue(exact)

    @mock.patch.object(
        runner,
        "_validate_timeline_manifest",
        return_value=(timeline_replay_fixture(), []),
    )
    def test_cross_cell_semantic_change_is_negative(self, _timeline):
        shard = self.make_shard()
        shard["factorial"][1]["cells"][3]["semantics"][0][
            "full_vocab_step_logit_sha256"
        ][2] = digest("tampered")
        shard["factorial"][1]["cells"][3]["witness_semantics"][0][
            "full_vocab_step_logit_sha256"
        ][2] = digest("tampered")
        diagnostics = shard["factorial"][1]["cells"][3]["memory_cell"][
            "allocator_receipt"
        ]["generation_diagnostics"]
        diagnostics["rows"][2 * 8]["cpu_logits_sha256"] = digest("tampered")
        diagnostics["rows_sha256"] = runner.sha256_json(diagnostics["rows"])
        exact, _bindings, _semantics, _source, _oracle, _baseline, _matrix = runner._validate_factorial(
            shard, root=Path("."), rank=0, run_id="run"
        )
        self.assertFalse(exact)

    @mock.patch.object(
        runner,
        "_validate_timeline_manifest",
        return_value=(timeline_replay_fixture(), []),
    )
    def test_factorial_source_blocks_must_match_across_cells_and_n(self, _timeline):
        shard = self.make_shard()
        shard["factorial"][2]["cells"][0]["source_physical_payload_sha256"][
            "3"
        ] = digest("other-source")
        shard["factorial"][2]["cells"][0]["memory_source_physical_payload_sha256"][
            "3"
        ] = digest("other-source")
        shard["factorial"][2]["cells"][0]["witness_source_physical_payload_sha256"][
            "3"
        ] = digest("other-source")
        with self.assertRaisesRegex(runner.ReviewAuditError, "different physical source"):
            runner._validate_factorial(
                shard, root=Path("."), rank=0, run_id="run"
            )

    @mock.patch.object(
        runner,
        "_validate_timeline_manifest",
        return_value=(timeline_replay_fixture(), []),
    )
    def test_memory_endpoint_and_kernel_ledger_tamper_fail_closed(self, _timeline):
        shard = self.make_shard()
        shard["factorial"][0]["cells"][0]["memory_cell"]["allocator_receipt"][
            "generation_peak_allocated_delta_bytes"
        ] += 1
        with self.assertRaisesRegex(runner.ReviewAuditError, "derived endpoint"):
            runner._validate_factorial(shard, root=Path("."), rank=0, run_id="run")
        shard = self.make_shard()
        shard["factorial"][0]["cells"][0]["memory_kernel_ledgers"][0][
            "strict_position_values"
        ] = True
        with self.assertRaisesRegex(runner.ReviewAuditError, "strict position mode"):
            runner._validate_factorial(shard, root=Path("."), rank=0, run_id="run")

    def test_fabricated_storage_and_gdn_endpoints_are_rejected(self):
        memory = factorial_cell(0, 1, 0)["memory_cell"]
        fabricated_storage = copy.deepcopy(memory)
        storage = fabricated_storage["allocator_receipt"]["storage_breakdown"]
        storage["layers"][0]["fabricated_paper_endpoint_nbytes"] = 123456
        storage["totals"]["fabricated_paper_endpoint_nbytes"] = 123456
        fabricated_storage["allocator_receipt"][
            "storage_breakdown_sha256"
        ] = runner.sha256_json(storage)
        with self.assertRaisesRegex(
            runner.ReviewAuditError, "layer row schema drift"
        ):
            runner._validate_memory_receipt(fabricated_storage)

        fabricated_gdn = copy.deepcopy(memory)
        audit = fabricated_gdn["policy_execution_receipt"]["group_audit"]
        audit["rows"][0]["gdn_base"][
            "fabricated_gdn_savings_nbytes"
        ] = 999999
        fabricated_gdn["policy_execution_receipt"][
            "group_audit_sha256"
        ] = runner.sha256_json(audit)
        with self.assertRaisesRegex(
            runner.ReviewAuditError, "request GDN policy receipt drift"
        ):
            runner._validate_memory_receipt(fabricated_gdn)

        changed_by_diagnostics = copy.deepcopy(memory)
        changed_by_diagnostics["allocator_receipt"]["after_setup_diagnostics"][
            "current_reserved_bytes"
        ] += 1
        with self.assertRaisesRegex(
            runner.ReviewAuditError, "diagnostics changed current allocator state"
        ):
            runner._validate_memory_receipt(changed_by_diagnostics)

        generation_drift = copy.deepcopy(memory)
        generation_rows = generation_drift["allocator_receipt"][
            "generation_diagnostics"
        ]
        generation_rows["rows"][0]["after_current_reserved_bytes"] += 1
        generation_rows["rows_sha256"] = runner.sha256_json(
            generation_rows["rows"]
        )
        with self.assertRaisesRegex(
            runner.ReviewAuditError,
            "generation CPU diagnostics changed CUDA allocator state",
        ):
            runner._validate_memory_receipt(generation_drift)

    @mock.patch.object(
        runner,
        "_validate_timeline_manifest",
        return_value=(timeline_replay_fixture(), []),
    )
    def test_scientific_integer_coordinates_reject_bool(self, _timeline):
        memory = factorial_cell(0, 1, 0)["memory_cell"]
        memory["resident_count"] = True
        with self.assertRaisesRegex(
            runner.ReviewAuditError, "non-bool formal N"
        ):
            runner._validate_memory_receipt(memory)

        shard = self.make_shard()
        shard["factorial"][0]["resident_count"] = True
        with self.assertRaisesRegex(runner.ReviewAuditError, "factorial N"):
            runner._validate_factorial(
                shard, root=Path("."), rank=0, run_id="run"
            )

        semantic = semantic_rows(1)
        semantic[0]["request_index"] = False
        with self.assertRaisesRegex(runner.ReviewAuditError, "request order"):
            runner._validate_semantics(semantic, resident_count=1)

        ledgers = kernel_ledgers(
            1, runner.SHARED_REUSE, strict_position_values=True
        )
        ledgers[0]["resident_count"] = True
        with self.assertRaisesRegex(runner.ReviewAuditError, "policy/N"):
            runner._validate_kernel_ledgers(
                ledgers,
                resident_count=1,
                kv_policy=runner.SHARED_REUSE,
                label="strict fixture",
                strict_position_values=True,
            )

    @mock.patch.object(runner, "_validate_timeline_manifest")
    def test_cross_axis_timeline_content_drift_is_rejected(self, timeline):
        call_index = 0

        def replay(*_args, **_kwargs):
            nonlocal call_index
            value = timeline_replay_fixture()
            if call_index % 4 == 2:
                value["normalized_gdn_factor_projection"] = {
                    "fixture": "cross-kv-content-drift"
                }
            call_index += 1
            return value, []

        timeline.side_effect = replay
        with self.assertRaisesRegex(
            runner.ReviewAuditError,
            "KV axis changed the normalized GDN factor receipt",
        ):
            runner._validate_factorial(
                self.make_shard(), root=Path("."), rank=0, run_id="run"
            )


class RealOrchestratorMockTest(unittest.TestCase):
    def test_formal_gpu_is_selected_by_exact_uuid_not_numeric_index(self):
        assignment = gpu_assignment_receipt()["rows"][3]
        expected_uuid = assignment["uuid"]
        with mock.patch.dict(
            runner.os.environ, {"CUDA_VISIBLE_DEVICES": "3"}, clear=False
        ):
            with self.assertRaisesRegex(
                runner.ReviewAuditError, "equal the assigned GPU UUID"
            ):
                runner._audit_formal_local_gpu(expected_uuid, assignment)

        properties = SimpleNamespace(name="NVIDIA H20")
        nvidia_smi = SimpleNamespace(
            stdout=(
                f"{expected_uuid}, {assignment['name']}, "
                f"{assignment['total_memory_mib']}\n"
            )
        )
        with mock.patch.dict(
            runner.os.environ,
            {"CUDA_VISIBLE_DEVICES": expected_uuid},
            clear=False,
        ), mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "device_count", return_value=1
        ), mock.patch.object(
            torch.cuda, "set_device"
        ) as set_device, mock.patch.object(
            torch.cuda, "get_device_properties", return_value=properties
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ), mock.patch.object(
            torch.cuda, "is_bf16_supported", return_value=True
        ), mock.patch.object(
            runner.subprocess, "run", return_value=nvidia_smi
        ) as nvidia_query:
            audit = runner._audit_formal_local_gpu(expected_uuid, assignment)
        set_device.assert_called_once_with(0)
        self.assertEqual(audit["cuda_visible_devices"], expected_uuid)
        self.assertEqual(audit["physical_visible_index"], 3)
        self.assertEqual(audit["process_local_device"], "cuda:0")
        self.assertIn(f"--id={expected_uuid}", nvidia_query.call_args.args[0])

    def test_model_load_authority_private_view_and_terminal_closure_are_external(self):
        evidence = model_load_evidence()
        identity_fields = {
            "model_weight_ledger_sha256": runner.sha256_bytes(
                evidence["weight_raw"]
            ),
            "model_artifact_ledger_sha256": runner.sha256_bytes(
                evidence["artifact_raw"]
            ),
        }
        authority, closure, manifest, summary = (
            runner._validate_external_model_load_evidence(
                authority_raw=evidence["authority_raw"],
                expected_authority_raw_sha256=evidence["authority_sha"],
                closure_raw=evidence["closure_raw"],
                expected_closure_raw_sha256=evidence["closure_sha"],
                private_model_view_manifest_raw=evidence["manifest_raw"],
                expected_private_model_view_manifest_raw_sha256=(
                    evidence["manifest_sha"]
                ),
                model_weight_ledger_raw=evidence["weight_raw"],
                model_artifact_ledger_raw=evidence["artifact_raw"],
                expected_identity=identity_fields,
                expected_run_id="0" * 32,
            )
        )
        self.assertTrue(summary["passed"])
        self.assertEqual(closure["authority_raw_sha256"], evidence["authority_sha"])
        self.assertEqual(authority["model_view_manifest_sha256"], evidence["manifest_sha"])
        self.assertEqual(manifest["weight_file_count"], 14)

        bad_manifest = copy.deepcopy(evidence["manifest"])
        bad_manifest["rows"][1]["copy_mode"] = "hardlink"
        bad_manifest_raw = runner.canonical_json_bytes(bad_manifest) + b"\n"
        with self.assertRaisesRegex(
            runner.ReviewAuditError, "copy provenance"
        ):
            runner._validate_external_model_load_evidence(
                authority_raw=evidence["authority_raw"],
                expected_authority_raw_sha256=evidence["authority_sha"],
                closure_raw=evidence["closure_raw"],
                expected_closure_raw_sha256=evidence["closure_sha"],
                private_model_view_manifest_raw=bad_manifest_raw,
                expected_private_model_view_manifest_raw_sha256=(
                    runner.sha256_bytes(bad_manifest_raw)
                ),
                model_weight_ledger_raw=evidence["weight_raw"],
                model_artifact_ledger_raw=evidence["artifact_raw"],
                expected_identity=identity_fields,
                expected_run_id="0" * 32,
            )

        bad_closure = copy.deepcopy(evidence["closure"])
        bad_closure["all_fds_closed"] = False
        bad_closure_raw = model_lease.canonical_receipt_bytes(bad_closure)
        with self.assertRaisesRegex(
            runner.ReviewAuditError, "closure rejected"
        ):
            runner._validate_external_model_load_evidence(
                authority_raw=evidence["authority_raw"],
                expected_authority_raw_sha256=evidence["authority_sha"],
                closure_raw=bad_closure_raw,
                expected_closure_raw_sha256=runner.sha256_bytes(bad_closure_raw),
                private_model_view_manifest_raw=evidence["manifest_raw"],
                expected_private_model_view_manifest_raw_sha256=(
                    evidence["manifest_sha"]
                ),
                model_weight_ledger_raw=evidence["weight_raw"],
                model_artifact_ledger_raw=evidence["artifact_raw"],
                expected_identity=identity_fields,
                expected_run_id="0" * 32,
            )

    def test_private_view_authority_rows_cannot_drift_from_manifest(self):
        evidence = model_load_evidence()
        authority = copy.deepcopy(evidence["authority"])
        authority["rows"][0]["stat"]["st_ino"] += 1
        authority["rows_sha256"] = model_lease.sha256_json(authority["rows"])
        authority_raw = model_lease.canonical_receipt_bytes(authority)
        closure = copy.deepcopy(evidence["closure"])
        closure["authority_raw_sha256"] = runner.sha256_bytes(authority_raw)
        closure["rows"][0]["stat"]["st_ino"] += 1
        closure["rows_sha256"] = model_lease.sha256_json(closure["rows"])
        closure_raw = model_lease.canonical_receipt_bytes(closure)
        identity_fields = {
            "model_weight_ledger_sha256": runner.sha256_bytes(
                evidence["weight_raw"]
            ),
            "model_artifact_ledger_sha256": runner.sha256_bytes(
                evidence["artifact_raw"]
            ),
        }
        with self.assertRaisesRegex(
            runner.ReviewAuditError, "authority rows differ from the private view"
        ):
            runner._validate_external_model_load_evidence(
                authority_raw=authority_raw,
                expected_authority_raw_sha256=runner.sha256_bytes(authority_raw),
                closure_raw=closure_raw,
                expected_closure_raw_sha256=runner.sha256_bytes(closure_raw),
                private_model_view_manifest_raw=evidence["manifest_raw"],
                expected_private_model_view_manifest_raw_sha256=(
                    evidence["manifest_sha"]
                ),
                model_weight_ledger_raw=evidence["weight_raw"],
                model_artifact_ledger_raw=evidence["artifact_raw"],
                expected_identity=identity_fields,
                expected_run_id="0" * 32,
            )

    def test_gpu_assignment_receipt_is_external_exact_and_one_to_one(self):
        receipt = gpu_assignment_receipt()
        raw = runner.canonical_json_bytes(receipt) + b"\n"
        replay = runner._validate_gpu_assignment_receipt(
            receipt,
            raw_sha256=runner.sha256_bytes(raw),
            raw_bytes=raw,
        )
        self.assertEqual(replay["rows"][3]["rank"], 3)
        duplicate = copy.deepcopy(receipt)
        duplicate["rows"][1]["uuid"] = duplicate["rows"][0]["uuid"]
        duplicate_raw = runner.canonical_json_bytes(duplicate) + b"\n"
        with self.assertRaisesRegex(runner.ReviewAuditError, "one-to-one"):
            runner._validate_gpu_assignment_receipt(
                duplicate,
                raw_sha256=runner.sha256_bytes(duplicate_raw),
                raw_bytes=duplicate_raw,
            )
        wrong_type = copy.deepcopy(receipt)
        wrong_type["rows"][0]["rank"] = False
        wrong_type_raw = runner.canonical_json_bytes(wrong_type) + b"\n"
        with self.assertRaisesRegex(runner.ReviewAuditError, "row drift"):
            runner._validate_gpu_assignment_receipt(
                wrong_type,
                raw_sha256=runner.sha256_bytes(wrong_type_raw),
                raw_bytes=wrong_type_raw,
            )

    def test_live_input_lifetime_receipts_bind_all_three_boundaries(self):
        bank = frozen_query_banks()[0]
        baseline = allocator_snapshot()
        receipts = live_input_lifetime_receipts(bank, baseline)
        replay = runner._validate_live_input_lifetime_receipts(
            receipts,
            expected_query_bank=bank,
            frozen_baseline=baseline,
        )
        self.assertEqual(
            [row["capture_point"] for row in replay],
            list(runner.LIVE_INPUT_CAPTURE_POINTS),
        )
        changed = copy.deepcopy(receipts)
        changed[1]["query_rows"][7]["sha256"] = digest("post-warmup-drift")
        with self.assertRaisesRegex(runner.ReviewAuditError, "lifetime receipt"):
            runner._validate_live_input_lifetime_receipts(
                changed,
                expected_query_bank=bank,
                frozen_baseline=baseline,
            )
        changed_allocator = copy.deepcopy(receipts)
        changed_allocator[2]["allocator_after"]["current_reserved_bytes"] += 1
        changed_allocator[2]["allocator_after"]["peak_reserved_bytes"] += 1
        with self.assertRaisesRegex(runner.ReviewAuditError, "frozen allocator"):
            runner._validate_live_input_lifetime_receipts(
                changed_allocator,
                expected_query_bank=bank,
                frozen_baseline=baseline,
            )

    def test_producer_self_replay_has_exact_schema_and_blind_values(self):
        expected = runner._make_producer_self_replay_receipt(
            factorial_exact=True,
            oracle_passed=False,
            mutant_rows_replayed=2,
            matched_clean_rows_replayed=2,
            memory_matrix_rows_replayed=12,
            detached_sidecar_references_replayed=73,
        )
        self.assertEqual(
            runner._validate_producer_self_replay_receipt(
                copy.deepcopy(expected), expected=expected
            ),
            expected,
        )
        forged = copy.deepcopy(expected)
        forged["oracle_passed"] = True
        with self.assertRaisesRegex(runner.ReviewAuditError, "blind replay"):
            runner._validate_producer_self_replay_receipt(
                forged, expected=expected
            )
        extra = {**expected, "producer_claim": True}
        with self.assertRaisesRegex(runner.ReviewAuditError, "blind replay"):
            runner._validate_producer_self_replay_receipt(extra, expected=expected)

    def test_formal_aggregate_requires_external_shared_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static = runner.make_static_artifact(
                identity(), selection_plan(), frozen_query_banks()
            )
            static_sha = runner.sha256_json(static)
            paths = []
            for rank in range(runner.FORMAL_WORLD_SIZE):
                path = root / f"formal-{rank}.json"
                path.write_bytes(
                    runner.canonical_json_bytes(
                        {"rank": rank, "artifact_mode": "formal_gpu"}
                    )
                    + b"\n"
                )
                paths.append(path)
            receipts = runner.make_receipt_manifest(
                paths, root=root, static_artifact_sha256=static_sha
            )
            with mock.patch.object(runner, "GPU_LOOP_IMPLEMENTED", True):
                with self.assertRaisesRegex(
                    runner.ReviewAuditError, "shared run/GPU/private-view/model-load"
                ):
                    runner.aggregate_shards(
                        receipts,
                        expected_receipt_manifest_sha256=runner.sha256_json(
                            receipts
                        ),
                        static_artifact=static,
                        static_artifact_sha256=static_sha,
                        artifact_root=root,
                        expected_run_id="0" * 32,
                    )

    def test_run_id_receipt_replays_derivation_and_rejects_bool_bits(self):
        static_sha = digest("static")
        protocol_sha = digest("protocol")
        nonce = bytes(range(32))
        domain = b"qcomem-forkaudit-run-id-v1\0"
        run_id = runner.hashlib.sha256(
            domain
            + bytes.fromhex(static_sha)
            + bytes.fromhex(protocol_sha)
            + nonce
        ).hexdigest()[:32]
        receipt = {
            "schema_version": "qcomem-forkaudit-run-id-receipt-v1",
            "run_id": run_id,
            "run_id_bits": 128,
            "derivation": (
                "sha256(domain || static_sha256 || protocol_sha256 || nonce)[:16]"
            ),
            "domain_hex": domain.hex(),
            "static_artifact_sha256": static_sha,
            "protocol_manifest_sha256": protocol_sha,
            "nonce_hex": nonce.hex(),
            "generated_once_after_static_before_candidate_outputs": True,
        }
        expected = runner.sha256_json(receipt)
        self.assertEqual(
            runner._validate_run_id_receipt(
                runner.canonical_json_bytes(receipt),
                expected_sha256=expected,
                run_id=run_id,
                static_artifact_sha256=static_sha,
                protocol_manifest_sha256=protocol_sha,
            )["run_id"],
            run_id,
        )
        bad = copy.deepcopy(receipt)
        bad["run_id_bits"] = True
        with self.assertRaisesRegex(runner.ReviewAuditError, "binding drift"):
            runner._validate_run_id_receipt(
                runner.canonical_json_bytes(bad),
                expected_sha256=runner.sha256_json(bad),
                run_id=run_id,
                static_artifact_sha256=static_sha,
                protocol_manifest_sha256=protocol_sha,
            )

    def test_weight_ledger_uses_index_and_sizes_without_per_rank_rehash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            weight_map = {}
            for index in range(1, 15):
                name = f"model.safetensors-{index:05d}-of-00014.safetensors"
                payload = bytes([index]) * (16 + index)
                (root / name).write_bytes(payload)
                rows.append({"logical_name": name, "sha256": digest(name)})
                weight_map[f"tensor.{index}"] = name
            total = sum((root / row["logical_name"]).stat().st_size for row in rows)
            (root / "model.safetensors.index.json").write_text(
                json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map})
            )
            audit = runner._verify_weight_ledger_structure(rows, model_dir=root)
            self.assertEqual(audit["entry_count"], 14)
            self.assertFalse(audit["per_rank_full_weight_rehash_performed"])
            bad = copy.deepcopy(rows)
            bad[0]["logical_name"] = "unindexed.safetensors"
            with self.assertRaisesRegex(runner.ReviewAuditError, "loader index"):
                runner._verify_weight_ledger_structure(bad, model_dir=root)

    def test_max_n_warmup_runs_all_four_arms_and_freezes_one_baseline(self):
        baseline = allocator_snapshot()
        runtime = SimpleNamespace(
            model=object(),
            backbone=object(),
            plan=object(),
            document=object(),
            queries=tuple(object() for _ in range(32)),
            kernel=object(),
        )

        def fake_cell(**kwargs):
            return {
                "memory_cell": {
                    "primary_memory_endpoint_eligible": True,
                    "resident_count": kwargs["resident_count"],
                    "arm_id": kwargs["arm_id"],
                }
            }

        with mock.patch.object(
            runner, "_gpu_cleanup", return_value=copy.deepcopy(baseline)
        ), mock.patch.object(
            runner, "_run_clean_memory_cell", side_effect=fake_cell
        ) as cells:
            receipt, frozen = runner._run_max_n_warmup(rank=0, runtime=runtime)
        self.assertEqual(cells.call_count, 5)
        self.assertEqual(receipt["arm_order"], list(runner.ARM_IDS))
        self.assertTrue(receipt["one_discarded_priming_cell_before_baseline_freeze"])
        self.assertEqual(
            [call.kwargs["resident_count"] for call in cells.call_args_list],
            [32, 32, 32, 32, 32],
        )
        self.assertEqual(frozen, baseline)

    def test_factorial_orchestrator_builds_twelve_independent_cell_pairs(self):
        source = {str(layer): digest(f"source-{layer}") for layer in runner.FORMAL_FULL_LAYERS}
        payload = {str(layer): digest(f"payload-{layer}") for layer in runner.FORMAL_FULL_LAYERS}
        runtime = SimpleNamespace(
            model=object(),
            backbone=object(),
            plan=object(),
            document=object(),
            queries=tuple(object() for _ in range(32)),
            kernel=object(),
        )
        selection = selection_plan()[0]

        def memory(**kwargs):
            return {
                "memory_cell": {"role": "memory"},
                "memory_kernel_ledgers": [],
                "source_physical_document_sha256_before": copy.deepcopy(source),
                "source_physical_document_sha256_after": copy.deepcopy(source),
                "source_physical_payload_sha256": copy.deepcopy(payload),
                "semantics": [],
            }

        def witness(**kwargs):
            is_oracle = (
                kwargs["resident_count"] == 1
                and kwargs["kv_policy"] == runner.ORACLE_KV_POLICY
                and kwargs["gdn_base_policy"] == runner.ORACLE_GDN_BASE_POLICY
            )
            return {
                "witness_cell": {"role": "witness"},
                "witness_kernel_ledgers": [],
                "source_physical_document_sha256_before": copy.deepcopy(source),
                "source_physical_document_sha256_after": copy.deepcopy(source),
                "source_physical_payload_sha256": copy.deepcopy(payload),
                "semantics": [],
                "oracle_raw_artifact": ({"oracle": True} if is_oracle else None),
            }

        def execute(run_cell, **_kwargs):
            return run_cell()

        with mock.patch.object(
            runner, "_run_clean_memory_cell", side_effect=memory
        ) as memory_calls, mock.patch.object(
            runner, "_run_ownership_witness_cell", side_effect=witness
        ) as witness_calls, mock.patch.object(
            runner, "_execute_cell_with_cleanup", side_effect=execute
        ) as wrappers:
            factorial, oracle = runner._run_formal_factorial_cells(
                artifact_root=Path("."),
                run_id="0" * 32,
                rank=0,
                runtime=runtime,
                oracle_selection=selection,
                frozen_baseline=allocator_snapshot(),
            )
        self.assertEqual(len(factorial), 3)
        self.assertEqual([len(row["cells"]) for row in factorial], [4, 4, 4])
        self.assertEqual(memory_calls.call_count, 12)
        self.assertEqual(witness_calls.call_count, 12)
        self.assertEqual(wrappers.call_count, 24)
        self.assertEqual(oracle, {"oracle": True})

    def test_cell_cleanup_preserves_primary_exception(self):
        baseline = allocator_snapshot()

        class PrimaryError(RuntimeError):
            pass

        with mock.patch.object(
            runner,
            "_gpu_cleanup",
            side_effect=[copy.deepcopy(baseline), copy.deepcopy(baseline)],
        ):
            with self.assertRaisesRegex(PrimaryError, "primary"):
                runner._execute_cell_with_cleanup(
                    lambda: (_ for _ in ()).throw(PrimaryError("primary")),
                    cell_role_key="memory_cell",
                    frozen_baseline=baseline,
                    label="early failure fixture",
                )

    def test_backend_cleanup_failure_does_not_replace_primary_exception(self):
        class PrimaryError(RuntimeError):
            pass

        with mock.patch.object(
            runner,
            "_unregister_backends",
            side_effect=runner.ReviewAuditError("cleanup failed"),
        ):
            with self.assertRaisesRegex(PrimaryError, "detector primary") as caught:
                with runner._registered_backend_scope(["fixture-backend"]):
                    raise PrimaryError("detector primary")
        self.assertTrue(
            any("secondary backend cleanup" in note for note in caught.exception.__notes__)
        )

    def test_full_shard_orchestrator_commits_only_after_self_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "shards" / "forkaudit-shard-0.json"
            static_replay = {
                "frozen_identity": identity(),
                "oracle_selection_plan": selection_plan(),
                "frozen_query_banks": frozen_query_banks(),
            }
            inputs = SimpleNamespace(
                static_sha256=digest("static"),
                static_replay=static_replay,
                run_id_receipt={"run_id": "0" * 32},
                gpu_assignment_receipt=gpu_assignment_receipt(),
                gpu_assignment_receipt_raw_sha256=digest("gpu-assignment"),
                private_model_view_manifest={"schema_version": "fixture"},
                private_model_view_manifest_raw_sha256=digest("private-view"),
                model_load_authority={"schema_version": "fixture"},
                model_load_authority_raw_sha256=digest("lease-authority"),
                data_usage={"dataset": "pg19"},
                input_rebuild_receipt={"schema_version": "fixture"},
            )
            runtime = SimpleNamespace(
                model=object(),
                backbone=object(),
                plan=object(),
                document=object(),
                queries=tuple(object() for _ in range(32)),
                kernel=object(),
                hardware_audit={"schema_version": "fixture"},
                model_runtime_audit={"schema_version": "fixture"},
            )
            args = SimpleNamespace(
                rank=0,
                run_id="0" * 32,
                artifact_root=root,
                output=output,
                expected_run_id_receipt_sha256=digest("run-receipt"),
            )
            order = []

            def mark(name, value):
                def call(*_args, **_kwargs):
                    order.append(name)
                    return value
                return call

            def capture(*_args, **_kwargs):
                point = _kwargs["capture_point"]
                order.append(f"token-gate:{point}")
                return {"capture_point": point}

            def factorial(*_args, **kwargs):
                order.append("factorial")
                (kwargs["artifact_root"] / "rank-0").mkdir(parents=True)
                return ([{"resident_count": 1}], {"oracle": True})

            def write(path, value):
                order.append("atomic-write")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(runner.canonical_json_bytes(value) + b"\n")

            with mock.patch.object(
                runner, "_load_formal_input_bundle", side_effect=mark("inputs", inputs)
            ), mock.patch.object(
                runner, "_load_formal_model_runtime", side_effect=mark("model", runtime)
            ), mock.patch.object(
                runner,
                "_run_max_n_warmup",
                side_effect=mark("warmup", ({"warm": True}, allocator_snapshot())),
            ), mock.patch.object(
                runner,
                "_capture_live_input_lifetime_receipt",
                side_effect=capture,
            ), mock.patch.object(
                runner,
                "_run_formal_factorial_cells",
                side_effect=factorial,
            ), mock.patch.object(
                runner,
                "_run_live_fault_campaign",
                side_effect=mark("mutants", {"campaign": True}),
            ), mock.patch.object(
                runner,
                "_self_validate_formal_shard",
                side_effect=mark("self-replay", {"schema_replay": True}),
            ), mock.patch.object(
                runner, "_write_json", side_effect=write
            ) as writer:
                shard = runner._run_formal_gpu_shard_impl(args)
            self.assertEqual(
                order,
                [
                    "inputs",
                    "model",
                    "warmup",
                    f"token-gate:{runner.LIVE_INPUT_CAPTURE_POINTS[0]}",
                    "factorial",
                    f"token-gate:{runner.LIVE_INPUT_CAPTURE_POINTS[1]}",
                    "mutants",
                    f"token-gate:{runner.LIVE_INPUT_CAPTURE_POINTS[2]}",
                    "self-replay",
                    "atomic-write",
                ],
            )
            self.assertEqual(writer.call_count, 1)
            self.assertEqual(shard["producer_self_replay"], {"schema_replay": True})
            self.assertTrue(output.exists())
            self.assertTrue((root / "rank-0").is_dir())
            self.assertFalse(list(root.glob(".forkaudit-rank-0-*")))

    def test_full_shard_orchestrator_never_commits_failed_self_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "shards" / "forkaudit-shard-0.json"
            inputs = SimpleNamespace(
                static_sha256=digest("static"),
                static_replay={
                    "frozen_identity": identity(),
                    "oracle_selection_plan": selection_plan(),
                    "frozen_query_banks": frozen_query_banks(),
                },
                run_id_receipt={"run_id": "0" * 32},
                gpu_assignment_receipt=gpu_assignment_receipt(),
                gpu_assignment_receipt_raw_sha256=digest("gpu-assignment"),
                private_model_view_manifest={"schema_version": "fixture"},
                private_model_view_manifest_raw_sha256=digest("private-view"),
                model_load_authority={"schema_version": "fixture"},
                model_load_authority_raw_sha256=digest("lease-authority"),
                data_usage={"dataset": "pg19"},
                input_rebuild_receipt={"schema_version": "fixture"},
            )
            runtime = SimpleNamespace(
                model=object(),
                backbone=object(),
                plan=object(),
                document=object(),
                queries=tuple(object() for _ in range(32)),
                kernel=object(),
                hardware_audit={"schema_version": "fixture"},
                model_runtime_audit={"schema_version": "fixture"},
            )
            args = SimpleNamespace(
                rank=0,
                run_id="0" * 32,
                artifact_root=root,
                output=output,
                expected_run_id_receipt_sha256=digest("run-receipt"),
            )

            def factorial(*_args, **kwargs):
                (kwargs["artifact_root"] / "rank-0").mkdir(parents=True)
                return ([{"resident_count": 1}], {"oracle": True})

            with mock.patch.object(
                runner, "_load_formal_input_bundle", return_value=inputs
            ), mock.patch.object(
                runner, "_load_formal_model_runtime", return_value=runtime
            ), mock.patch.object(
                runner,
                "_run_max_n_warmup",
                return_value=({"warm": True}, allocator_snapshot()),
            ), mock.patch.object(
                runner,
                "_capture_live_input_lifetime_receipt",
                side_effect=lambda *_args, **kwargs: {
                    "capture_point": kwargs["capture_point"]
                },
            ), mock.patch.object(
                runner,
                "_run_formal_factorial_cells",
                side_effect=factorial,
            ), mock.patch.object(
                runner,
                "_run_live_fault_campaign",
                return_value={"campaign": True},
            ), mock.patch.object(
                runner,
                "_self_validate_formal_shard",
                side_effect=runner.ReviewAuditError("blind replay failed"),
            ), mock.patch.object(runner, "_write_json") as writer:
                with self.assertRaisesRegex(
                    runner.ReviewAuditError, "blind replay failed"
                ):
                    runner._run_formal_gpu_shard_impl(args)
            writer.assert_not_called()
            self.assertFalse(output.exists())
            self.assertFalse((root / "rank-0").exists())
            self.assertFalse(list(root.glob(".forkaudit-rank-0-*")))

    def test_formal_shard_cli_contract_is_explicit_and_release_gated(self):
        actions = {
            option
            for action in runner._parser()._actions
            for option in action.option_strings
        }
        required = {
            "--artifact-root",
            "--static-artifact",
            "--expected-static-sha256",
            "--rr2-input-manifest",
            "--expected-rr2-input-manifest-sha256",
            "--pg19-data",
            "--pg19-manifest",
            "--prior-capacity-manifest",
            "--model-dir",
            "--code-ledger",
            "--model-artifact-ledger",
            "--model-weight-ledger",
            "--protocol-manifest",
            "--run-id",
            "--run-id-receipt",
            "--expected-run-id-receipt-sha256",
            "--gpu-assignment-receipt",
            "--expected-gpu-assignment-receipt-raw-sha256",
            "--private-model-view-manifest",
            "--expected-private-model-view-manifest-raw-sha256",
            "--model-load-authority",
            "--expected-model-load-authority-raw-sha256",
            "--model-load-closure",
            "--expected-model-load-closure-raw-sha256",
            "--expected-gpu-uuid",
        }
        self.assertTrue(required <= actions)
        self.assertTrue(runner.GPU_LOOP_IMPLEMENTED)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "must-not-exist.json"
            with self.assertRaisesRegex(
                runner.ReviewAuditError, "formal shard missing explicit inputs"
            ):
                runner.main(
                    [
                        "--stage",
                        "shard",
                        "--output",
                        str(output),
                        "--run-id",
                        "0" * 32,
                    ]
                )
            self.assertFalse(output.exists())


class MutantSchemaTest(unittest.TestCase):
    def test_gdn_matched_clean_gate_is_preserved_as_false_positive(self):
        for rank, mutant_id in ((3, "M4"), (4, "M5")):
            with self.subTest(mutant_id=mutant_id):
                gate_id = runner.MUTANT_SPECS[mutant_id].expected_gate_id
                clean_fp = run_clean_case(
                    lambda _context, gate_id=gate_id: (_ for _ in ()).throw(
                        RuntimeInvariantError(gate_id, "matched clean gate")
                    )
                )
                shard = {"fault_campaign": fault_campaign(rank)}
                matched = shard["fault_campaign"]["mutants"][mutant_id][
                    "matched_clean"
                ]
                matched["outcome"] = clean_fp.to_dict()
                matched["exercise_coverage_receipt"] = coverage_receipt(
                    mutant_id,
                    clean_fp.to_dict(),
                    mutation_activated=False,
                )
                shard["fault_campaign"]["mutants"][mutant_id][
                    "matched_clean_exercise_passed"
                ] = False
                _clean, _mutants, matched_outcomes = (
                    runner._validate_fault_campaign(
                        shard,
                        rank=rank,
                        seen_case_ids=set(),
                        expected_query_sha256=digest("query-0"),
                    )
                )
                self.assertFalse(
                    runner._validate_clean_outcome(matched_outcomes[mutant_id])
                )

    def test_every_mutant_has_detector_specific_replayable_coverage(self):
        for mutant_id in MUTANT_IDS:
            with self.subTest(mutant_id=mutant_id, role="matched-clean"):
                clean_dict = clean_outcome_dict()
                clean = runner._parse_campaign_outcome(clean_dict)
                receipt = coverage_receipt(
                    mutant_id, clean_dict, mutation_activated=False
                )
                runner._validate_exercise_coverage_receipt(
                    receipt,
                    mutant_id=mutant_id,
                    outcome=clean,
                    mutation_activated=False,
                    expected_query_sha256=digest("query-0"),
                )
                bad_input = copy.deepcopy(receipt)
                bad_input["detector_input"]["evidence"]["unrelated_noop"] = True
                bad_input["detector_input_sha256"] = runner.sha256_json(
                    bad_input["detector_input"]
                )
                with self.assertRaisesRegex(
                    runner.ReviewAuditError, "drift"
                ):
                    runner._validate_exercise_coverage_receipt(
                        bad_input,
                        mutant_id=mutant_id,
                        outcome=clean,
                        mutation_activated=False,
                        expected_query_sha256=digest("query-0"),
                    )
                bad_completion = copy.deepcopy(receipt)
                bad_completion["completion_receipt_sha256"] = "0" * 64
                with self.assertRaisesRegex(
                    runner.ReviewAuditError, "completion receipt"
                ):
                    runner._validate_exercise_coverage_receipt(
                        bad_completion,
                        mutant_id=mutant_id,
                        outcome=clean,
                        mutation_activated=False,
                        expected_query_sha256=digest("query-0"),
                    )
            with self.subTest(mutant_id=mutant_id, role="injected"):
                injected = mutant_outcome(mutant_id, f"coverage-{mutant_id}")
                injected_dict = injected.to_dict()
                receipt = coverage_receipt(mutant_id, injected_dict)
                runner._validate_exercise_coverage_receipt(
                    receipt,
                    mutant_id=mutant_id,
                    outcome=injected,
                    mutation_activated=True,
                    expected_query_sha256=digest("query-0"),
                )

    def test_clean_false_positive_is_preserved_as_scientific_negative(self):
        clean = run_clean_case(
            lambda _context: (_ for _ in ()).throw(
                RuntimeInvariantError("KV_SEQUENCE_ID", "false positive")
            )
        )
        self.assertFalse(runner._validate_clean_outcome(clean))
        shard = {"fault_campaign": fault_campaign(1)}
        shard["fault_campaign"]["clean_case"]["outcome"] = clean.to_dict()
        parsed, _outcomes, _matched = runner._validate_fault_campaign(
            shard,
            rank=1,
            seen_case_ids=set(),
            expected_query_sha256=digest("query-0"),
        )
        self.assertEqual(
            parsed.classification.value, "clean_false_positive"
        )

    def test_rank_zero_requires_m1_and_m9_and_key_matches_id(self):
        shard = {"fault_campaign": fault_campaign(0)}
        clean, outcomes, matched = runner._validate_fault_campaign(
            shard,
            rank=0,
            seen_case_ids=set(),
            expected_query_sha256=digest("query-0"),
        )
        self.assertEqual(set(outcomes), {"M1", "M9"})
        self.assertEqual(set(matched), {"M1", "M9"})
        self.assertEqual(clean.classification.value, "clean_pass")
        missing = copy.deepcopy(shard)
        del missing["fault_campaign"]["mutants"]["M9"]
        with self.assertRaisesRegex(runner.ReviewAuditError, "missing/extra"):
            runner._validate_fault_campaign(
                missing,
                rank=0,
                seen_case_ids=set(),
                expected_query_sha256=digest("query-0"),
            )
        copied = copy.deepcopy(shard)
        copied["fault_campaign"]["mutants"]["M9"]["outcome"] = copied[
            "fault_campaign"
        ]["mutants"]["M1"]["outcome"]
        with self.assertRaisesRegex(runner.ReviewAuditError, "key/id"):
            runner._validate_fault_campaign(
                copied,
                rank=0,
                seen_case_ids=set(),
                expected_query_sha256=digest("query-0"),
            )

    def test_target_binding_must_be_inside_same_lifecycle_receipt(self):
        outcome = mutant_outcome("M4", "cell")
        row = {"fault_campaign": {
            "assignment": ["M4"],
            "clean_case": {
                "case_cell_id": "global-clean",
                "case_isolation": {
                    "fresh_document_cache_built": True,
                    "fresh_request_cache_built": True,
                    "cache_reused_from_prior_case": False,
                    "cache_discarded_after_case": True,
                },
                "cleanup_receipt": cleanup_receipt(),
                "outcome": clean_outcome_dict(),
            },
            "mutants": {
                "M4": {
                    "mutant_id": "M4",
                    "exercise_mutant_id": "M4",
                    "exercise_contract_sha256": (
                        runner._mutant_exercise_contract_sha256("M4")
                    ),
                    "exercise_coverage_receipt": coverage_receipt(
                        "M4", outcome.to_dict()
                    ),
                    "case_cell_id": "cell",
                    "case_isolation": {
                        "fresh_document_cache_built": True,
                        "fresh_request_cache_built": True,
                        "cache_reused_from_prior_case": False,
                        "cache_discarded_after_case": True,
                    },
                    "cleanup_receipt": cleanup_receipt(),
                    "outcome": outcome.to_dict(),
                    "matched_clean": {
                        "exercise_mutant_id": "M4",
                        "exercise_contract_sha256": (
                            runner._mutant_exercise_contract_sha256("M4")
                        ),
                        "exercise_coverage_receipt": coverage_receipt(
                            "M4",
                            clean_outcome_dict(),
                            mutation_activated=False,
                        ),
                        "case_cell_id": "matched-cell",
                        "case_isolation": {
                            "fresh_document_cache_built": True,
                            "fresh_request_cache_built": True,
                            "cache_reused_from_prior_case": False,
                            "cache_discarded_after_case": True,
                        },
                        "cleanup_receipt": cleanup_receipt(),
                        "outcome": clean_outcome_dict(),
                    },
                    "matched_clean_exercise_passed": True,
                }
            },
        }}
        runner._validate_fault_campaign(
            row,
            rank=3,
            seen_case_ids=set(),
            expected_query_sha256=digest("query-0"),
        )
        bad = copy.deepcopy(row)
        bad["fault_campaign"]["mutants"]["M4"]["outcome"][
            "mutation_receipt"
        ]["target_mutation_binding"]["target_field"] = "unrelated_dummy"
        with self.assertRaisesRegex(runner.ReviewAuditError, "target mutation binding"):
            runner._validate_fault_campaign(
                bad,
                rank=3,
                seen_case_ids=set(),
                expected_query_sha256=digest("query-0"),
            )
        missing = copy.deepcopy(row)
        missing["fault_campaign"]["mutants"]["M4"]["outcome"][
            "mutation_receipt"
        ]["target_mutation_binding"] = None
        with self.assertRaisesRegex(runner.ReviewAuditError, "target mutation binding"):
            runner._validate_fault_campaign(
                missing,
                rank=3,
                seen_case_ids=set(),
                expected_query_sha256=digest("query-0"),
            )


class OracleReplayTest(unittest.TestCase):
    def write_oracle(
        self,
        root: Path,
        *,
        candidate_delta: float,
    ) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        selection = selection_plan()[0]
        query = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
        document_key = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
        document_value = torch.tensor([[[[2.0, 0.0]]]], dtype=torch.float32)
        append_key = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)
        append_value = torch.tensor([[[[0.0, 4.0]]]], dtype=torch.float32)
        physical_document_key_blocks = document_key.clone()
        physical_document_value_blocks = document_value.clone()
        document_block_table = torch.tensor([[0]], dtype=torch.int32)
        active_block_table = torch.tensor([[0, 1]], dtype=torch.int32)
        key = torch.cat((document_key, append_key), dim=2)
        value = torch.cat((document_value, append_value), dim=2)
        query_positions = torch.tensor([1], dtype=torch.int64)
        key_positions = torch.tensor([0, 1], dtype=torch.int64)
        scale = 2 ** -0.5
        reference = fp32_dense_attention_reference(
            query,
            key,
            value,
            query_positions=query_positions,
            key_positions=key_positions,
            scaling=scale,
        ).output
        candidate = reference + candidate_delta
        outcome = OraclePreregistration(
            OracleThresholds(max_relative_l2=runner.ORACLE_MAX_RELATIVE_L2)
        ).evaluate_attention(
            query,
            key,
            value,
            candidate,
            query_positions=query_positions,
            key_positions=key_positions,
            scaling=scale,
        ).to_dict()
        selection_sha, prereg_sha = runner._oracle_selection_preregistration(selection)
        source_payload_sha = runner._physical_payload_digest_from_tensors(
            physical_document_key_blocks,
            physical_document_value_blocks,
            document_block_table,
            layer_index=selection["layer_index"],
            document_length=1,
            page_size=1,
        )
        digests = {
            "query": runner._tensor_digest(query),
            "key": runner._tensor_digest(key),
            "value": runner._tensor_digest(value),
            "document_key": runner._tensor_digest(document_key),
            "document_value": runner._tensor_digest(document_value),
            "append_key_shadow": runner._tensor_digest(append_key),
            "append_value_shadow": runner._tensor_digest(append_value),
            "candidate_output": runner._tensor_digest(candidate),
            "query_positions": runner._tensor_digest(query_positions),
            "key_positions": runner._tensor_digest(key_positions),
            "visibility_mask": runner.sha256_json(None),
            "softmax_scale": runner.sha256_json(float(scale)),
            "document_key_value_component": runner.sha256_json(
                [runner._tensor_digest(document_key), runner._tensor_digest(document_value)]
            ),
            "append_key_value_shadow_component": runner.sha256_json(
                [runner._tensor_digest(append_key), runner._tensor_digest(append_value)]
            ),
            "physical_document_key_blocks": runner._tensor_digest(
                physical_document_key_blocks
            ),
            "physical_document_value_blocks": runner._tensor_digest(
                physical_document_value_blocks
            ),
            "document_block_table": runner._tensor_digest(document_block_table),
            "candidate_active_block_table": runner._tensor_digest(active_block_table),
            "source_physical_payload": source_payload_sha,
        }
        before = {
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        }
        ledger_call = {
            "layer_idx": selection["layer_index"],
            "request_index": selection["request_index"],
            "resident_count": 1,
            "round_index": selection["round_index"],
            "softmax_scale": scale,
            "append_capture_id": "append-shadow-fixture",
            "append_audit": {
                "append_event_index": 0,
                "append_tokens": 1,
                "appended_tokens_before": 0,
                "appended_tokens_after": 1,
                "sequence_length_before": 1,
                "sequence_length_after": 2,
                "capture_id": "append-shadow-fixture",
            },
            "fixture": True,
        }
        append_manifest = [{
            "capture_id": "append-shadow-fixture",
            "append_event_index": 0,
            "appended_tokens_before": 0,
            "appended_tokens_after": 1,
            "key_sha256": runner._tensor_digest(append_key),
            "value_sha256": runner._tensor_digest(append_value),
            "layer_index": selection["layer_index"],
            "request_index": selection["request_index"],
            "round_index": 0,
            "ledger_call_sha256": runner.sha256_json(ledger_call),
        }]
        raw = {
            "schema_version": runner.ORACLE_RAW_SCHEMA_VERSION,
            "resident_count": 1,
            "selection": selection,
            "selection_sha256": selection_sha,
            "outer_preregistration_sha256": prereg_sha,
            "source_contract": {
                "post_rope_qkv": True,
                "candidate_output_from_live_unified_attention": True,
                "key_value_source": "immutable-document-physical-blocks-plus-independent-append-shadow",
                "key_value_independent_of_candidate_active_block_table": True,
                "document_component_sha256": digests["document_key_value_component"],
                "append_shadow_component_sha256": digests["append_key_value_shadow_component"],
                "concatenation_order": "document-then-append-shadow",
                "candidate_softmax_scale_source": "live-kernel-observer",
            },
            "document_geometry": {"document_length": 1, "page_size": 1},
            "arena_geometry": {
                "total_physical_blocks": 2,
                "document_physical_blocks": 1,
                "private_physical_blocks": 1,
                "document_sidecars_exclude_private_uninitialized_backing": True,
            },
            "reference_precision": {
                "arithmetic": "ieee-fp32",
                "candidate_computed_outside_reference_precision_context": True,
                "before": before,
                "effective": {
                    "float32_matmul_precision": "highest",
                    "cuda_matmul_allow_tf32": False,
                    "cudnn_allow_tf32": False,
                },
                "after": before,
                "restored": True,
            },
            "softmax_scale": scale,
            "tensors": {
                "query": runner.encode_inline_tensor(query),
                "candidate_output": runner.encode_inline_tensor(candidate),
                "query_positions": runner.encode_inline_tensor(query_positions),
                "key_positions": runner.encode_inline_tensor(key_positions),
                "physical_document_key_blocks": runner.encode_inline_tensor(
                    physical_document_key_blocks
                ),
                "physical_document_value_blocks": runner.encode_inline_tensor(
                    physical_document_value_blocks
                ),
                "document_block_table": runner.encode_inline_tensor(document_block_table),
                "candidate_active_block_table": runner.encode_inline_tensor(active_block_table),
                "visibility_mask": None,
            },
            "append_events": [{
                "schema_version": "qcomem-oracle-append-event-v1",
                "capture_id": "append-shadow-fixture",
                "append_event_index": 0,
                "appended_tokens_before": 0,
                "appended_tokens_after": 1,
                "sequence_length_before": 1,
                "sequence_length_after": 2,
                "source_device": "cpu",
                "source_dtype": str(append_key.dtype),
                "source_shape": list(append_key.shape),
                "key_sha256": runner._tensor_digest(append_key),
                "value_sha256": runner._tensor_digest(append_value),
                "key": runner.encode_inline_tensor(append_key),
                "value": runner.encode_inline_tensor(append_value),
            }],
            "input_digests": digests,
            "live_call_observer": {
                "schema_version": "qcomem-live-call-observer-v2",
                "run_id": "synthetic-run",
                "rank": 0,
                "resident_count": 1,
                "cell_id": "witness-oracle-cell",
                "arm_id": (
                    f"kv={selection['kv_policy']}|"
                    f"gdn={selection['gdn_base_policy']}"
                ),
                "kv_policy": selection["kv_policy"],
                "gdn_base_policy": selection["gdn_base_policy"],
                "sample_id": selection["sample_id"],
                "layer_index": selection["layer_index"],
                "request_index": selection["request_index"],
                "round_index": selection["round_index"],
                "ledger_call_sha256": runner.sha256_json(ledger_call),
                "append_capture_id": "append-shadow-fixture",
                "append_event_manifest": append_manifest,
                "kernel_audit": {"softmax_scale": scale},
                "effective_scaling": scale,
                "softmax_scale_source": (
                    "MultiForkHitLedger.call_observer.kernel_audit.softmax_scale"
                ),
                "input_digests": digests,
                "document_capture": {
                    "capture_point": (
                        "persistent-document-arena-via-document-block-table"
                    ),
                    "independent_of_candidate_active_block_table": True,
                    "physical_document_key_blocks_sha256": digests[
                        "physical_document_key_blocks"
                    ],
                    "physical_document_value_blocks_sha256": digests[
                        "physical_document_value_blocks"
                    ],
                    "document_block_table_sha256": digests["document_block_table"],
                    "key_sha256": digests["document_key"],
                    "value_sha256": digests["document_value"],
                    "source_physical_payload_sha256": source_payload_sha,
                },
                "append_shadow_capture": {
                    "capture_point": "cache-layer-update-before-sequence-append",
                    "independent_of_candidate_active_block_table": True,
                    "events": append_manifest,
                    "key_sha256": digests["append_key_shadow"],
                    "value_sha256": digests["append_value_shadow"],
                },
                "active_block_table_sha256": digests["candidate_active_block_table"],
            },
            "recorded_outcome": outcome,
        }
        path = root / "oracle.json"
        path.write_bytes(runner.canonical_json_bytes(raw))
        observer_context = {
            "cell_id": "witness-oracle-cell",
            "ledger_call": ledger_call,
            "selected_ledger": {"calls": [ledger_call]},
            "source_physical_payload_sha256": source_payload_sha,
        }
        return runner.artifact_reference(path, root=root), selection, observer_context

    def test_oracle_recomputes_and_threshold_failure_is_valid_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, selection, observer_context = self.write_oracle(
                root, candidate_delta=1.0
            )
            outcome, _binding = runner._recompute_oracle(
                reference,
                root=root,
                rank=0,
                source_object="train/10000.txt",
                expected_selection=selection,
                observer_context=observer_context,
                expected_run_id="synthetic-run",
                synthetic_geometry=True,
            )
            self.assertEqual(outcome["status"], "completed")
            self.assertFalse(outcome["passed"])
            self.assertGreater(
                outcome["attention_metrics"]["relative_l2"],
                runner.ORACLE_MAX_RELATIVE_L2,
            )

    def test_oracle_cannot_self_select_after_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, selection, observer_context = self.write_oracle(
                root, candidate_delta=0.0
            )
            different = copy.deepcopy(selection)
            different["round_index"] = 7
            with self.assertRaisesRegex(runner.ReviewAuditError, "frozen pre-run"):
                runner._recompute_oracle(
                    reference,
                    root=root,
                    rank=0,
                    source_object="train/10000.txt",
                    expected_selection=different,
                    observer_context=observer_context,
                    expected_run_id="synthetic-run",
                    synthetic_geometry=True,
                )

    def test_oracle_rejects_noncanonical_source_device(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference, selection, observer_context = self.write_oracle(
                root, candidate_delta=0.0
            )
            path = root / reference["relative_path"]
            raw = runner.strict_json_loads(path.read_bytes(), label="oracle fixture")
            raw["append_events"][0]["source_device"] = "cuda-fake"
            path.write_bytes(runner.canonical_json_bytes(raw))
            reference = runner.artifact_reference(path, root=root)
            with self.assertRaisesRegex(runner.ReviewAuditError, "source device"):
                runner._recompute_oracle(
                    reference,
                    root=root,
                    rank=0,
                    source_object="train/10000.txt",
                    expected_selection=selection,
                    observer_context=observer_context,
                    expected_run_id="synthetic-run",
                    synthetic_geometry=True,
                )


class AggregateReceiptTest(unittest.TestCase):
    def build_fixture(self, root: Path):
        static = runner.make_static_artifact(
            identity(), selection_plan(), frozen_query_banks()
        )
        static_sha = runner.sha256_json(static)
        paths = []
        for rank in range(8):
            bank = static["frozen_query_banks"][rank]
            shard = {
                "schema_version": runner.SHARD_SCHEMA_VERSION,
                "protocol": runner.PROTOCOL,
                "rank": rank,
                "world_size": 8,
                "artifact_mode": "synthetic_schema_fixture",
                "status": "completed_synthetic_schema_fixture",
                "static_artifact_sha256": static_sha,
                "protocol_config": runner.formal_protocol_config(),
                "protocol_config_sha256": runner.sha256_json(
                    runner.formal_protocol_config()
                ),
                "frozen_identity": static["frozen_identity"],
                "run_id": "0123456789abcdef0123456789abcdef",
                "data_usage": {
                    "dataset": "pg19",
                    "split": "train",
                    "pg19_train_only": True,
                    "longbench_consumed": False,
                    "validation_consumed": False,
                    "test_v2_consumed": False,
                    "source_id": bank["source_id"],
                    "source_object": bank["source_object"],
                    "book_index": rank,
                    "window_index": bank["window_index"],
                    "document_start_token": bank["document_start_token"],
                    "document_end_token_exclusive": bank[
                        "document_end_token_exclusive"
                    ],
                    "document_length": runner.FORMAL_DOCUMENT_TOKENS,
                    "document_token_ids_sha256": bank[
                        "document_token_ids_sha256"
                    ],
                    "document_input_receipt": {
                        "capture_point": (
                            "immediately-before-persistent-document-prefill"
                        ),
                        "dtype": "torch.int64",
                        "shape": [1, runner.FORMAL_DOCUMENT_TOKENS],
                        "sha256": bank["document_token_ids_sha256"],
                        "rebuilt_from_raw_bound_rr2_manifest": True,
                    },
                    "query_bank_input_receipt": {
                        "capture_point": (
                            "immediately-before-formal-factorial-cells"
                        ),
                        "dtype": "torch.int64",
                        "shape_per_query": [1, runner.FORMAL_QUERY_TOKENS],
                        "count": max(runner.FORMAL_RESIDENT_COUNTS),
                        "rows": [
                            {
                                "request_index": row["request_index"],
                                "sha256": row["query_token_ids_sha256"],
                            }
                            for row in bank["rows"]
                        ],
                        "rebuilt_from_raw_bound_rr2_manifest": True,
                    },
                },
            }
            path = root / f"shard-{rank}.json"
            path.write_bytes(runner.canonical_json_bytes(shard))
            paths.append(path)
        receipts = runner.make_receipt_manifest(
            paths, root=root, static_artifact_sha256=static_sha
        )
        return static, static_sha, receipts

    def aggregate(
        self,
        root: Path,
        static,
        static_sha,
        receipts,
        modes=None,
        *,
        expected_run_id="0123456789abcdef0123456789abcdef",
    ):
        modes = {} if modes is None else modes
        original_fault_validator = runner._validate_fault_campaign

        def campaign_side_effect(
            shard,
            *,
            rank,
            seen_case_ids,
            expected_query_sha256,
            expected_frozen_baseline=None,
        ):
            payload = {"fault_campaign": fault_campaign(rank, modes)}
            return original_fault_validator(
                payload,
                rank=rank,
                seen_case_ids=seen_case_ids,
                expected_query_sha256=expected_query_sha256,
                expected_frozen_baseline=expected_frozen_baseline,
            )

        oracle = {
            "passed": True,
            "attention_metrics": {"relative_l2": 0.001},
        }
        fixture_calls = [
            {"softmax_scale": 2 ** -0.5}
            for _ in range(
                runner.FORMAL_GENERATION_STEPS * len(runner.FORMAL_FULL_LAYERS)
            )
        ]
        oracle_contexts = {
            arm_id: {
                "cell_id": f"fixture-{arm_id}",
                "witness_ledgers": [{"calls": fixture_calls}],
                "source_physical_payload_sha256_by_layer": {
                    str(layer): digest(f"fixture-source-{layer}")
                    for layer in runner.FORMAL_FULL_LAYERS
                },
            }
            for arm_id in runner.ARM_IDS
        }
        def factorial_side_effect(
            _shard, *, root, rank, run_id, expected_query_bank=None
        ):
            return (
                True,
                [],
                {},
                {
                    str(layer): digest(f"fixture-source-{layer}")
                    for layer in runner.FORMAL_FULL_LAYERS
                },
                oracle_contexts,
                cleanup_receipt()["frozen_model_query_baseline"],
                memory_matrix_rows(rank),
            )

        with mock.patch.object(
            runner,
            "_validate_factorial",
            side_effect=factorial_side_effect,
        ), mock.patch.object(
            runner,
            "_recompute_oracle",
            return_value=(oracle, [{"sha256": digest("oracle")}]),
        ), mock.patch.object(
            runner, "_validate_fault_campaign", side_effect=campaign_side_effect
        ):
            return runner.aggregate_shards(
                receipts,
                expected_receipt_manifest_sha256=runner.sha256_json(receipts),
                static_artifact=static,
                static_artifact_sha256=static_sha,
                artifact_root=root,
                expected_run_id=expected_run_id,
                allow_synthetic_schema_fixture=True,
            )

    def test_complete_synthetic_eight_shard_schema_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static, static_sha, receipts = self.build_fixture(root)
            result = self.aggregate(root, static, static_sha, receipts)
            self.assertTrue(result["schema_replay_passed"])
            self.assertTrue(result["hypothesis_passed"])
            self.assertTrue(result["passed"])
            self.assertTrue(result["formal_ready"])
            self.assertEqual(result["scientific_outcome"], "valid_positive")
            matrix = result["memory_matrix"]
            self.assertEqual(len(matrix["cells"]), 12)
            first = matrix["cells"][0]
            self.assertEqual(
                [row["rank"] for row in first["allocator_raw_by_rank"]],
                list(range(8)),
            )
            self.assertEqual(
                first["allocator_median_across_ranks"][
                    "generation_peak_allocated_delta_bytes"
                ],
                1003.5,
            )
            self.assertFalse(matrix["generic_unique_storage_endpoint_included"])

    def test_aggregate_requires_launcher_shared_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static, static_sha, receipts = self.build_fixture(root)
            with self.assertRaisesRegex(runner.ReviewAuditError, "run ID drift"):
                self.aggregate(
                    root,
                    static,
                    static_sha,
                    receipts,
                    expected_run_id="fedcba9876543210fedcba9876543210",
                )

    def test_memory_matrix_raw_coordinate_tamper_is_rejected(self):
        rows = [
            row
            for rank in range(runner.FORMAL_WORLD_SIZE)
            for row in memory_matrix_rows(rank)
        ]
        runner._aggregate_memory_matrix(rows)
        duplicated = copy.deepcopy(rows)
        duplicated[-1]["rank"] = 0
        with self.assertRaisesRegex(
            runner.ReviewAuditError, "rank coverage drift"
        ):
            runner._aggregate_memory_matrix(duplicated)
        boolean_n = copy.deepcopy(rows)
        boolean_n[0]["resident_count"] = True
        with self.assertRaisesRegex(
            runner.ReviewAuditError, "raw coordinate drift"
        ):
            runner._aggregate_memory_matrix(boolean_n)

    def test_escape_is_valid_negative_but_crash_invalidates_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static, static_sha, receipts = self.build_fixture(root)
            escaped = self.aggregate(
                root, static, static_sha, receipts, {"M9": "escape"}
            )
            self.assertTrue(escaped["scientific_run_valid"])
            self.assertFalse(escaped["hypothesis_passed"])
            self.assertEqual(
                escaped["mutant_campaign"]["escaped_mutant_ids"], ["M9"]
            )
            crashed = self.aggregate(
                root, static, static_sha, receipts, {"M3": "crash"}
            )
            self.assertFalse(crashed["scientific_run_valid"])
            self.assertFalse(crashed["hypothesis_passed"])

    def test_raw_shard_sha_drift_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static, static_sha, receipts = self.build_fixture(root)
            path = root / receipts["shards"][0]["relative_path"]
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(runner.ReviewAuditError, "byte count mismatch"):
                self.aggregate(root, static, static_sha, receipts)

    def test_validation_and_test_v2_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static, static_sha, receipts = self.build_fixture(root)
            ref = receipts["shards"][2]
            path = root / ref["relative_path"]
            shard = json.loads(path.read_text())
            shard["data_usage"]["split"] = "validation"
            shard["data_usage"]["test_v2_consumed"] = True
            path.write_bytes(runner.canonical_json_bytes(shard))
            receipts["shards"][2] = runner.artifact_reference(path, root=root)
            with self.assertRaisesRegex(runner.ReviewAuditError, "only PG19 train"):
                self.aggregate(root, static, static_sha, receipts)

    def test_live_document_receipt_must_match_raw_bound_static_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            static, static_sha, receipts = self.build_fixture(root)
            ref = receipts["shards"][0]
            path = root / ref["relative_path"]
            shard = json.loads(path.read_text())
            shard["data_usage"]["document_token_ids_sha256"] = digest(
                "post-hoc-other-document"
            )
            shard["data_usage"]["document_input_receipt"]["sha256"] = shard[
                "data_usage"
            ]["document_token_ids_sha256"]
            path.write_bytes(runner.canonical_json_bytes(shard))
            receipts["shards"][0] = runner.artifact_reference(path, root=root)
            with self.assertRaisesRegex(
                runner.ReviewAuditError, "raw-bound RR2 input"
            ):
                self.aggregate(root, static, static_sha, receipts)


if __name__ == "__main__":
    unittest.main()
