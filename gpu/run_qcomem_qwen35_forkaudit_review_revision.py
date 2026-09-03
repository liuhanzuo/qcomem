from __future__ import annotations

"""Reviewer-triggered ForkAudit protocol (audit skeleton).

This file intentionally implements the CPU/static and blind aggregate side of
the protocol before the production GPU loop.  ``--stage shard`` therefore
fails closed.  A static artifact may validate, and synthetic raw artifacts may
exercise the aggregate replay, but this revision can never emit a formal-ready
result only after ``GPU_LOOP_IMPLEMENTED`` is changed together with a reviewed live
implementation.

The aggregate never trusts producer ``passed`` booleans.  It reloads every raw
artifact through a detached SHA-256 receipt, replays the GDN timeline and FP32
oracle, reconstructs mutant dataclasses, and recomputes the four-cell semantic
equalities.  Importing this module performs no CUDA initialization.
"""

import argparse
import base64
import gc
import hashlib
import hmac
import inspect
import json
import math
import os
import re
import secrets
import shutil
import stat
import statistics
import subprocess
import tempfile
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, Sequence

from qcomem_forkaudit_mutants import (
    AppliedMutation,
    CampaignOutcome,
    CampaignPhase,
    ExecutionBoundary,
    FaultCampaignConfigurationError,
    InjectionStage,
    MUTANT_IDS,
    MUTANT_SPECS,
    MutationReceipt,
    OutcomeClassification,
    RuntimeInvariantError,
    TargetMutationBinding,
    callback_injector,
    run_clean_case,
    run_mutant_case,
    validate_campaign_outcomes,
    validate_target_mutation_binding,
)
from qcomem_forkaudit_oracle import (
    ORACLE_SCHEMA_VERSION,
    OracleGateError,
    OraclePreregistration,
    OracleThresholds,
)
from qcomem_forkaudit_storage_witness import (
    GDNStorageWitnessError,
    PHASE_POST_GENERATION,
    PHASE_POST_TRANSITION,
    PHASE_SETUP_PRE_TRANSITION,
    POLICY_MATERIALIZED,
    POLICY_SHARED_BASE,
    TIMELINE_SCHEMA_VERSION,
    capture_gdn_phase_witness,
    capture_gdn_storage_snapshot,
    capture_persistent_gdn_guard,
    capture_request_gdn_binding_guard,
    replay_gdn_storage_timeline,
    replay_gdn_storage_witness,
    verify_request_gdn_binding_guard,
)
from qcomem_vllm_paged_fair_control import FRESH_CONTROL, SHARED_REUSE
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    GDN_MATERIALIZE_REQUEST_BASE,
    MULTIFORK_PROTOCOL,
    MultiForkHitLedger,
    build_pg19_train_query_bank,
    build_resident_request_group,
    register_multifork_backend,
    resident_storage_breakdown,
    source_document_physical_digests,
    validate_runtime_kv_ownership,
)


PROTOCOL = "qcomem-qwen35-forkaudit-review-revision-v1"
STATIC_SCHEMA_VERSION = "qcomem-forkaudit-review-static-v1"
SHARD_SCHEMA_VERSION = "qcomem-forkaudit-review-shard-v1"
AGGREGATE_SCHEMA_VERSION = "qcomem-forkaudit-review-aggregate-v1"
PHASE_ARTIFACT_SCHEMA_VERSION = "qcomem-forkaudit-live-phase-artifact-v1"
ORACLE_RAW_SCHEMA_VERSION = "qcomem-forkaudit-oracle-raw-v2"
KV_WITNESS_SCHEMA_VERSION = "qcomem-forkaudit-kv-witness-v1"
RECEIPT_SCHEMA_VERSION = "qcomem-forkaudit-detached-receipts-v1"
GPU_ASSIGNMENT_RECEIPT_SCHEMA_VERSION = (
    "qcomem-forkaudit-gpu-assignment-receipt-v1"
)
LIVE_INPUT_LIFETIME_SCHEMA_VERSION = "qcomem-forkaudit-live-input-lifetime-v1"
PRIVATE_MODEL_VIEW_SCHEMA_VERSION = "qcomem-forkaudit-private-model-view-v1"

GPU_LOOP_IMPLEMENTED = True
IMPLEMENTATION_STATUS = "formal_gpu_pipeline_released"

FORMAL_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
FORMAL_MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
FORMAL_PG19_DATA_SHA256 = (
    "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
)
FORMAL_PG19_MANIFEST_SHA256 = (
    "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"
)
FORMAL_RR2_WINDOWS_SHA256 = (
    "39bc36bb2eb04d51122e66caaebfa72367c02b43b073f072a2da240ed068c166"
)
PRIOR_CAPACITY_MANIFEST_SHA256 = (
    "975bc6a12f43447024b889889d4156ca71c2f89b68de6157ac609b4a9687e9c0"
)
PRIOR_CAPACITY_WINDOWS_SHA256 = (
    "27ad6c687e5cab28f361bbd89dd1844788aecbecc6f2d25dbd0c60b7705a55f8"
)
PRIOR_FP32_CONTEXT_RAW_SHA256 = (
    "fa64f663bb74a190a0a5c0898fda2a55528171c77a91af2b1321c24a5f310a1d"
)
PRIOR_FP32_CONTEXT_SEMANTIC_SHA256 = (
    "f475bd11c11d58a12d2b7c2ffcf7a35b3700e29f6e591cf17f8f2a5598bcbbeb"
)
FINAL_REVIEW_RESPONSE_PLAN_SHA256 = (
    "e2be05e198d6c86276f229d4e862c3579c65e311d07c219e9e9c792a390cbfbb"
)

FORMAL_WORLD_SIZE = 8
FORMAL_BOOKS = 8
FORMAL_RESIDENT_COUNTS = (1, 8, 32)
FORMAL_GENERATION_STEPS = 8
FORMAL_QUERY_TOKENS = 32
FORMAL_DOCUMENT_TOKENS = 4095
FORMAL_WINDOW_STRIDE = 257
FORMAL_QUERY_BANK_STRIDE = 64
FORMAL_PAGE_SIZE = 128
FORMAL_FINAL_APPENDED_TOKENS = FORMAL_QUERY_TOKENS + FORMAL_GENERATION_STEPS - 1
FORMAL_FULL_LAYERS = tuple(range(3, 40, 4))
FORMAL_LINEAR_LAYERS = tuple(index for index in range(40) if index not in FORMAL_FULL_LAYERS)
ORACLE_RESIDENT_COUNT = 1
ORACLE_MAX_RELATIVE_L2 = 0.005
ORACLE_KV_POLICY = SHARED_REUSE
ORACLE_GDN_BASE_POLICY = GDN_BORROW_IMMUTABLE_BASE
FORMAL_KERNEL_MODE = "vllm_0_26_triton_unified_attention_q16_block_pool"
FORMAL_MODEL_TYPE = "qwen3_5_moe_text"
FORMAL_KERNEL_DESCRIPTOR = (
    "vllm.v1.attention.ops.triton_unified_attention",
    "unified_attention",
    "(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, "
    "softmax_scale, causal, window_size, block_table, softcap, q_descale, "
    "k_descale, v_descale, seq_threshold_3D=None, num_par_softmax_segments=None, "
    "softmax_segm_output=None, softmax_segm_max=None, softmax_segm_expsum=None, "
    "alibi_slopes=None, output_scale=None, qq_bias=None, sinks=None, "
    "mm_prefix_range=None, rswa_prefix_lens=None, rswa_window: int | None = None, "
    "use_alibi_sqrt=False, kv_quant_mode: "
    "vllm.v1.kv_cache_interface.KVQuantMode = <KVQuantMode.NONE: 0>, "
    "k_scale_cache=None, v_scale_cache=None, chunk_lookback=-1, use_td: bool = "
    "False, mm_prefix_clamp_sliding_window: bool = False)"
)
FORMAL_MASK_CONTRACT = "prevalidated-no-padding-tail-causal"
FORMAL_POSITION_CONTRACT = "qwen3.5-text-tail-post-rope-v1"
FORMAL_NUM_QUERY_HEADS = 16
FORMAL_NUM_KV_HEADS = 2
FORMAL_GQA_GROUPS = FORMAL_NUM_QUERY_HEADS // FORMAL_NUM_KV_HEADS
FORMAL_HEAD_DIM = 256
FORMAL_SOFTMAX_SCALE = FORMAL_HEAD_DIM**-0.5
FORMAL_ELEMENT_BYTES = 2
FORMAL_DOCUMENT_BLOCKS = math.ceil(FORMAL_DOCUMENT_TOKENS / FORMAL_PAGE_SIZE)
FORMAL_PRIVATE_BLOCKS_PER_REQUEST = math.ceil(
    (FORMAL_DOCUMENT_TOKENS % FORMAL_PAGE_SIZE + FORMAL_QUERY_TOKENS + FORMAL_GENERATION_STEPS)
    / FORMAL_PAGE_SIZE
)
FORMAL_PARTIAL_TAIL_COPY_NBYTES = (
    2
    * (FORMAL_DOCUMENT_TOKENS % FORMAL_PAGE_SIZE)
    * FORMAL_NUM_KV_HEADS
    * FORMAL_HEAD_DIM
    * FORMAL_ELEMENT_BYTES
)
FORMAL_BLOCK_NBYTES = (
    2
    * FORMAL_PAGE_SIZE
    * FORMAL_NUM_KV_HEADS
    * FORMAL_HEAD_DIM
    * FORMAL_ELEMENT_BYTES
)
FORMAL_DOCUMENT_PAYLOAD_NBYTES = (
    2
    * FORMAL_DOCUMENT_TOKENS
    * FORMAL_NUM_KV_HEADS
    * FORMAL_HEAD_DIM
    * FORMAL_ELEMENT_BYTES
)
FORMAL_DOCUMENT_ALLOCATED_NBYTES = FORMAL_DOCUMENT_BLOCKS * FORMAL_BLOCK_NBYTES
FORMAL_DOCUMENT_PADDING_NBYTES = (
    FORMAL_DOCUMENT_ALLOCATED_NBYTES - FORMAL_DOCUMENT_PAYLOAD_NBYTES
)
MEMORY_ENDPOINT_FIELDS = (
    "setup_plus_generation_peak_allocated_delta_bytes",
    "setup_plus_generation_peak_reserved_delta_bytes",
    "generation_peak_allocated_delta_bytes",
    "generation_peak_reserved_delta_bytes",
    "after_generation_current_allocated_bytes",
    "after_generation_current_reserved_bytes",
    "after_generation_current_allocated_delta_bytes",
    "after_generation_current_reserved_delta_bytes",
)

KV_POLICIES = (FRESH_CONTROL, SHARED_REUSE)
GDN_BASE_POLICIES = (
    GDN_MATERIALIZE_REQUEST_BASE,
    GDN_BORROW_IMMUTABLE_BASE,
)
# This explicit mapping is a protocol invariant.  KV policy must never be used
# as a proxy for the independent GDN ownership axis.
GDN_POLICY_TO_WITNESS = {
    GDN_MATERIALIZE_REQUEST_BASE: POLICY_MATERIALIZED,
    GDN_BORROW_IMMUTABLE_BASE: POLICY_SHARED_BASE,
}
ARM_IDS = tuple(
    f"kv={kv_policy}|gdn={gdn_policy}"
    for kv_policy in KV_POLICIES
    for gdn_policy in GDN_BASE_POLICIES
)
MUTANT_ASSIGNMENT_BY_RANK = {
    0: ("M1", "M9"),
    1: ("M2",),
    2: ("M3",),
    3: ("M4",),
    4: ("M5",),
    5: ("M6",),
    6: ("M7",),
    7: ("M8",),
}
MUTANT_TARGET_CONTRACT = {
    "M1": ("kv_reservation_table", "physical_block_ids"),
    "M2": ("request_kernel_ledger", "sequence_binding"),
    "M3": ("kv_tail_cow_dispatch", "detach_partial_tail_callable"),
    "M4": ("gdn_request_state", "persistent_base_storage_binding"),
    "M5": ("gdn_request_state", "peer_request_storage_binding"),
    "M6": ("post_rope_position_ids", "canonical_position_values"),
    "M7": ("attention_dispatch", "materialized_mask_argument"),
    "M8": ("attention_dispatch", "unified_attention_callable"),
    "M9": ("attention_dispatch", "paged_kv_representation"),
}
MUTANT_EXERCISE_PATHS = {
    "M1": "live-kv-ownership-post-construction",
    "M2": "live-model-forward-sequence-ledger",
    "M3": "live-tail-cow-ownership-after-all-request-layer-appends",
    "M4": "live-direct-gdn-base-storage-replay",
    "M5": "live-direct-gdn-peer-storage-replay",
    "M6": "live-direct-ledger-position-validation",
    "M7": "live-direct-ledger-mask-contract",
    "M8": "live-model-forward-kernel-identity-ledger",
    "M9": "live-model-forward-paged-view-ledger",
}

FROZEN_SHA256_FIELDS = (
    "code_ledger_sha256",
    "model_manifest_sha256",
    "model_artifact_ledger_sha256",
    "model_weight_ledger_sha256",
    "pg19_data_sha256",
    "pg19_manifest_sha256",
    "pg19_windows_sha256",
    "pg19_input_manifest_sha256",
    "prior_fp32_context_manifest_sha256",
    "review_response_plan_sha256",
    "protocol_manifest_sha256",
    "protocol_config_sha256",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_SOURCE_RE = re.compile(r"^train/[0-9]+\.txt$")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class ReviewAuditError(RuntimeError):
    """A raw artifact cannot support the preregistered conclusion."""


class ProductionLoopNotImplemented(ReviewAuditError):
    """The GPU producer is intentionally unavailable in this audit skeleton."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReviewAuditError(message)


def _add_exception_note(error: BaseException, note: str) -> None:
    """Attach secondary cleanup evidence on Python 3.9 through 3.13."""

    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = list(getattr(error, "__notes__", ()))
    notes.append(note)
    error.__notes__ = notes


def _is_int(value: Any) -> bool:
    return type(value) is int


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _require_run_id(value: Any, label: str = "run ID") -> str:
    _require(
        isinstance(value, str) and _RUN_ID_RE.fullmatch(value) is not None,
        f"{label} must be exactly 128 bits encoded as 32 lowercase hex characters",
    )
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReviewAuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes, *, label: str) -> Any:
    try:
        def reject_constant(value: str) -> None:
            raise ReviewAuditError(f"{label} contains non-finite JSON constant {value}")

        return json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ReviewAuditError):
            raise
        raise ReviewAuditError(f"{label} is not strict JSON") from exc


def _safe_artifact_path(root: Path, relative_path: Any) -> Path:
    _require(isinstance(relative_path, str) and bool(relative_path), "artifact path missing")
    pure = PurePosixPath(relative_path)
    _require(not pure.is_absolute(), "artifact path must be relative")
    _require(".." not in pure.parts and "." not in pure.parts, "artifact path traversal")
    root_resolved = root.resolve()
    path = (root_resolved / Path(*pure.parts)).resolve()
    _require(path == root_resolved or root_resolved in path.parents, "artifact escaped root")
    _require(path.is_file(), f"artifact missing: {relative_path}")
    return path


@dataclass(frozen=True)
class LoadedArtifact:
    payload: Any
    binding: dict[str, Any]


def artifact_reference(path: Path, *, root: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    relative = path.resolve().relative_to(root.resolve()).as_posix()
    return {
        "relative_path": relative,
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
    }


def load_json_artifact(
    reference: Any,
    *,
    root: Path,
    label: str,
) -> LoadedArtifact:
    _require(isinstance(reference, dict), f"{label} reference must be an object")
    _require(set(reference) == {"relative_path", "sha256", "bytes"}, f"{label} reference schema drift")
    digest = _require_sha256(reference["sha256"], f"{label} receipt")
    _require(_is_int(reference["bytes"]) and reference["bytes"] >= 2, f"{label} byte count drift")
    path = _safe_artifact_path(root, reference["relative_path"])
    raw = path.read_bytes()
    _require(len(raw) == reference["bytes"], f"{label} byte count mismatch")
    _require(sha256_bytes(raw) == digest, f"{label} SHA-256 mismatch")
    return LoadedArtifact(
        strict_json_loads(raw, label=label),
        {
            "relative_path": reference["relative_path"],
            "sha256": digest,
            "bytes": len(raw),
        },
    )


def formal_protocol_config() -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "world_size": FORMAL_WORLD_SIZE,
        "pg19_books": FORMAL_BOOKS,
        "resident_counts": list(FORMAL_RESIDENT_COUNTS),
        "generation_steps": FORMAL_GENERATION_STEPS,
        "query_tokens": FORMAL_QUERY_TOKENS,
        "document_tokens": FORMAL_DOCUMENT_TOKENS,
        "page_size": FORMAL_PAGE_SIZE,
        "document_tail_tokens": FORMAL_DOCUMENT_TOKENS % FORMAL_PAGE_SIZE,
        "kv_policies": list(KV_POLICIES),
        "gdn_base_policies": list(GDN_BASE_POLICIES),
        "factorial_arm_ids": list(ARM_IDS),
        "kernel_mode": FORMAL_KERNEL_MODE,
        "kernel_descriptor": {
            "module": FORMAL_KERNEL_DESCRIPTOR[0],
            "qualname": FORMAL_KERNEL_DESCRIPTOR[1],
            "signature": FORMAL_KERNEL_DESCRIPTOR[2],
        },
        "expected_softmax_scale": FORMAL_SOFTMAX_SCALE,
        "run_id_encoding": "128-bit-32-lowercase-hex-launcher-shared",
        "oracle_resident_count": ORACLE_RESIDENT_COUNT,
        "oracle_max_relative_l2": ORACLE_MAX_RELATIVE_L2,
        "oracle_threshold_semantics": "independently-pre-fixed-engineering-tolerance",
        "prior_fp32_context_manifest_sha256": PRIOR_FP32_CONTEXT_RAW_SHA256,
        "review_response_plan_sha256": FINAL_REVIEW_RESPONSE_PLAN_SHA256,
        "oracle_selection_locked_before_outputs": True,
        "pg19_train_only": True,
        "longbench_consumed": False,
        "validation_consumed": False,
        "test_v2_consumed": False,
        "ownership_witness_cell_separate_from_memory_cell": True,
        "mutant_assignment_by_rank": {
            str(rank): list(ids) for rank, ids in MUTANT_ASSIGNMENT_BY_RANK.items()
        },
        "mutant_cache_rebuilt_per_case": True,
        "matched_clean_rebuild_per_mutant": True,
        "mutant_exercise_coverage_schema": "forkaudit-mutant-exercise-coverage-v2",
    }


def validate_frozen_identity(identity: Any) -> dict[str, Any]:
    _require(isinstance(identity, dict), "frozen identity must be an object")
    required = set(FROZEN_SHA256_FIELDS) | {"model_id", "model_revision"}
    _require(not (required - set(identity)), "frozen identity fields missing")
    for field in FROZEN_SHA256_FIELDS:
        _require_sha256(identity[field], f"frozen identity {field}")
    exact_hashes = {
        "pg19_data_sha256": FORMAL_PG19_DATA_SHA256,
        "pg19_manifest_sha256": FORMAL_PG19_MANIFEST_SHA256,
        "pg19_windows_sha256": FORMAL_RR2_WINDOWS_SHA256,
        "prior_fp32_context_manifest_sha256": PRIOR_FP32_CONTEXT_RAW_SHA256,
        "review_response_plan_sha256": FINAL_REVIEW_RESPONSE_PLAN_SHA256,
    }
    for field, expected in exact_hashes.items():
        _require(identity[field] == expected, f"frozen identity {field} drift")
    _require(
        identity["protocol_config_sha256"] == sha256_json(formal_protocol_config()),
        "frozen protocol config SHA drift",
    )
    _require(identity["model_id"] == FORMAL_MODEL_ID, "formal model ID drift")
    _require(
        identity["model_revision"] == FORMAL_MODEL_REVISION,
        "formal model revision drift",
    )
    return dict(identity)


def validate_oracle_selection_plan(value: Any) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and len(value) == FORMAL_WORLD_SIZE, "oracle selection plan must freeze eight ranks")
    result = []
    required = {
        "selection_rule_id",
        "rank",
        "book_index",
        "source_object",
        "window_index",
        "document_start_token",
        "document_length",
        "document_token_ids_sha256",
        "layer_index",
        "request_index",
        "round_index",
        "sample_id",
        "kv_policy",
        "gdn_base_policy",
        "cell_role",
        "arm_id",
        "oracle_cell_id",
        "held_out_from_threshold_calibration",
        "locked_before_candidate_outputs",
    }
    for rank, row in enumerate(value):
        _require(isinstance(row, dict) and set(row) == required, "oracle selection-plan row schema drift")
        _require(row["selection_rule_id"] == "rank-frozen-heldout-post-rope-v1", "oracle selection rule drift")
        _require(
            _is_int(row["rank"])
            and _is_int(row["book_index"])
            and row["rank"] == rank
            and row["book_index"] == rank,
            "oracle selection rank/book drift",
        )
        _require(isinstance(row["source_object"], str) and _SOURCE_RE.fullmatch(row["source_object"]), "oracle selection source drift")
        _require(_is_int(row["window_index"]) and row["window_index"] >= 0, "oracle window index drift")
        _require(
            _is_int(row["document_start_token"])
            and row["document_start_token"] >= 0
            and row["document_start_token"]
            == row["window_index"] * FORMAL_WINDOW_STRIDE
            and _is_int(row["document_length"])
            and row["document_length"] == FORMAL_DOCUMENT_TOKENS,
            "oracle document coordinate drift",
        )
        _require_sha256(
            row["document_token_ids_sha256"], "oracle document token digest"
        )
        _require(
            _is_int(row["layer_index"])
            and row["layer_index"] in FORMAL_FULL_LAYERS,
            "oracle selection layer drift",
        )
        _require(
            _is_int(row["request_index"]) and row["request_index"] == 0,
            "oracle selection request must be zero",
        )
        _require(_is_int(row["round_index"]) and 0 <= row["round_index"] < FORMAL_GENERATION_STEPS, "oracle selection round drift")
        _require(isinstance(row["sample_id"], str) and bool(row["sample_id"]), "oracle sample ID missing")
        _require(row["kv_policy"] == ORACLE_KV_POLICY, "oracle KV arm drift")
        _require(row["gdn_base_policy"] == ORACLE_GDN_BASE_POLICY, "oracle GDN arm drift")
        _require(row["cell_role"] == "ownership_witness", "oracle must come from witness cell")
        expected_arm = (
            f"kv={ORACLE_KV_POLICY}|gdn={ORACLE_GDN_BASE_POLICY}"
        )
        _require(row["arm_id"] == expected_arm, "oracle explicit arm binding drift")
        _require(
            row["oracle_cell_id"]
            == f"rank-{rank}-N-1-{expected_arm}-ownership-witness",
            "oracle deterministic witness-cell binding drift",
        )
        _require(row["held_out_from_threshold_calibration"] is True, "oracle sample is not held out")
        _require(row["locked_before_candidate_outputs"] is True, "oracle plan was not locked pre-run")
        result.append(dict(row))
    _require(len({row["sample_id"] for row in result}) == FORMAL_WORLD_SIZE, "oracle sample IDs reused")
    return result


def validate_frozen_query_banks(
    value: Any,
    oracle_selection_plan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    _require(
        isinstance(value, list) and len(value) == FORMAL_WORLD_SIZE,
        "static artifact must freeze eight PG19 query banks",
    )
    result = []
    for rank, (bank, selection) in enumerate(zip(value, oracle_selection_plan)):
        required = {
            "rank",
            "book_index",
            "source_id",
            "source_object",
            "window_index",
            "document_start_token",
            "document_end_token_exclusive",
            "document_token_ids_sha256",
            "query_bank_start_token",
            "query_stride_tokens",
            "query_tokens",
            "count",
            "query_bank_sha256",
            "rows",
            "manifest_sha256",
        }
        _require(isinstance(bank, dict) and set(bank) == required, "frozen query-bank schema drift")
        _require(
            _is_int(bank["rank"])
            and _is_int(bank["book_index"])
            and bank["rank"] == rank
            and bank["book_index"] == rank
            and bank["source_object"] == selection["source_object"]
            and bank["window_index"] == selection["window_index"],
            "frozen query bank/source/window binding drift",
        )
        _require(
            isinstance(bank["source_id"], str)
            and bool(bank["source_id"])
            and _is_int(bank["window_index"])
            and _is_int(bank["document_start_token"])
            and _is_int(bank["document_end_token_exclusive"])
            and _is_int(bank["query_bank_start_token"])
            and _is_int(bank["query_stride_tokens"])
            and _is_int(bank["query_tokens"])
            and _is_int(bank["count"])
            and bank["document_start_token"]
            == selection["document_start_token"]
            == bank["window_index"] * FORMAL_WINDOW_STRIDE
            and bank["document_end_token_exclusive"]
            == bank["document_start_token"] + FORMAL_DOCUMENT_TOKENS
            and bank["document_token_ids_sha256"]
            == selection["document_token_ids_sha256"]
            and bank["query_bank_start_token"]
            == bank["document_end_token_exclusive"] + FORMAL_QUERY_TOKENS
            and bank["query_stride_tokens"] == FORMAL_QUERY_BANK_STRIDE
            and bank["query_tokens"] == FORMAL_QUERY_TOKENS
            and bank["count"] == max(FORMAL_RESIDENT_COUNTS),
            "frozen query-bank document/query geometry drift",
        )
        _require_sha256(
            bank["document_token_ids_sha256"],
            "frozen document token digest",
        )
        _require_sha256(bank["query_bank_sha256"], "frozen query-bank digest")
        _require(
            bank["manifest_sha256"]
            == sha256_json(
                {
                    key: item
                    for key, item in bank.items()
                    if key != "manifest_sha256"
                }
            ),
            "frozen query-bank self hash drift",
        )
        rows = bank["rows"]
        _require(isinstance(rows, list) and len(rows) == max(FORMAL_RESIDENT_COUNTS), "frozen query bank must contain 32 rows")
        digests = []
        prior_end = None
        normalized_rows = []
        for request_index, row in enumerate(rows):
            _require(
                isinstance(row, dict)
                and set(row)
                == {
                    "request_index",
                    "source_token_offset",
                    "query_tokens",
                    "query_token_ids_sha256",
                },
                "frozen query-bank row schema drift",
            )
            _require(
                _is_int(row["request_index"])
                and row["request_index"] == request_index,
                "frozen query request order drift",
            )
            _require(
                _is_int(row["query_tokens"])
                and row["query_tokens"] == FORMAL_QUERY_TOKENS,
                "frozen query length drift",
            )
            offset = row["source_token_offset"]
            _require(
                _is_int(offset)
                and offset
                == bank["query_bank_start_token"]
                + request_index * FORMAL_QUERY_BANK_STRIDE,
                "frozen query offset drift",
            )
            if prior_end is not None:
                _require(offset >= prior_end, "frozen query chunks overlap")
            prior_end = offset + FORMAL_QUERY_TOKENS
            digest = _require_sha256(row["query_token_ids_sha256"], "frozen query digest")
            digests.append(digest)
            normalized_rows.append(dict(row))
        _require(len(set(digests)) == len(digests), "frozen query bank is not pairwise distinct")
        result.append({**bank, "rows": normalized_rows})
    return result


def _validate_prior_fp32_context_manifest(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, dict)
        and value.get("schema_version")
        == "qcomem.forkaudit.fp32-prior-context-manifest.v1",
        "prior FP32 context manifest schema drift",
    )
    _require(
        sha256_json(value) == PRIOR_FP32_CONTEXT_SEMANTIC_SHA256,
        "prior FP32 context semantic digest drift",
    )
    definition = value.get("diagnostic_definition")
    rows = value.get("diagnostics")
    _require(
        isinstance(definition, dict)
        and _is_int(definition.get("diagnostic_count"))
        and definition.get("diagnostic_count") == 80
        and definition.get("role_in_rr2_threshold_choice")
        == "contextual_validation_only"
        and definition.get("selected_or_tuned_rr2_threshold") is False
        and isinstance(rows, list)
        and len(rows) == 80,
        "prior FP32 contextual-only definition drift",
    )
    relative_l2 = []
    for row in rows:
        _require(
            isinstance(row, dict)
            and _is_int(row.get("document_tokens"))
            and row.get("document_tokens") == 1025
            and _is_int(row.get("query_tokens"))
            and row.get("query_tokens") == FORMAL_QUERY_TOKENS,
            "prior FP32 diagnostic coordinate drift",
        )
        metric = row.get("metrics", {}).get("relative_l2")
        _require(
            type(metric) in (int, float)
            and math.isfinite(metric)
            and metric >= 0,
            "prior FP32 diagnostic relative-L2 drift",
        )
        relative_l2.append(float(metric))
    observed_max = max(relative_l2)
    margin = value.get("pre_fixed_threshold_margin_check")
    _require(isinstance(margin, dict), "prior FP32 margin receipt missing")
    _require(
        margin.get("fixed_preregistered_threshold") == ORACLE_MAX_RELATIVE_L2
        and margin.get("maximum_observed_prior_relative_l2") == observed_max
        and margin.get("required_context_margin_multiplier") == 2.0
        and margin.get("required_margin_boundary_from_prior_maximum")
        == 2.0 * observed_max
        and margin.get("fixed_threshold_to_prior_maximum_ratio")
        == ORACLE_MAX_RELATIVE_L2 / observed_max
        and ORACLE_MAX_RELATIVE_L2 >= 2.0 * observed_max
        and margin.get("fixed_threshold_at_least_twice_prior_maximum") is True
        and margin.get("prior_archive_role") == "contextual_validation_only"
        and margin.get("prior_rows_selected_or_tuned_threshold") is False
        and margin.get("threshold_fixed_before_rr2") is True
        and margin.get("threshold_fixed_independently_of_prior_rows") is True,
        "prior FP32 contextual margin does not replay",
    )
    disjoint = value.get("rr2_disjointness_from_prior_context")
    _require(
        isinstance(disjoint, dict)
        and isinstance(disjoint.get("prior_context_document_token_values"), list)
        and all(
            _is_int(item)
            for item in disjoint.get("prior_context_document_token_values", [])
        )
        and disjoint.get("prior_context_document_token_values") == [1025]
        and _is_int(disjoint.get("rr2_preregistered_document_tokens"))
        and disjoint.get("rr2_preregistered_document_tokens")
        == FORMAL_DOCUMENT_TOKENS
        and disjoint.get("document_length_is_required_coordinate_component")
        is True
        and disjoint.get("document_length_disjoint") is True,
        "RR2/prior FP32 coordinate disjointness drift",
    )
    return dict(value)


def _validate_review_response_plan(value: Any) -> dict[str, Any]:
    _require(
        isinstance(value, dict)
        and value.get("schema_version") == "1.0.0"
        and _is_int(value.get("source_round"))
        and value.get("source_round") == 2,
        "review-response plan schema/round drift",
    )
    rows = value.get("items")
    _require(isinstance(rows, list), "review-response plan items missing")
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("experiment_id") == "RR2-EXP-OWNERSHIP-MUTANTS"
    ]
    _require(len(matches) == 1, "RR2 review experiment is missing/duplicated")
    experiment = matches[0]
    configuration = experiment.get("execution", {}).get("configuration", {})
    _require(
        isinstance(configuration.get("ownership_factorial_resident_counts"), list)
        and all(
            _is_int(item)
            for item in configuration.get(
                "ownership_factorial_resident_counts", []
            )
        )
        and configuration.get("ownership_factorial_resident_counts")
        == list(FORMAL_RESIDENT_COUNTS)
        and _is_int(configuration.get("document_tokens"))
        and configuration.get("document_tokens") == FORMAL_DOCUMENT_TOKENS
        and _is_int(configuration.get("query_tokens"))
        and configuration.get("query_tokens") == FORMAL_QUERY_TOKENS
        and _is_int(configuration.get("semantic_steps"))
        and configuration.get("semantic_steps") == FORMAL_GENERATION_STEPS
        and _is_int(configuration.get("mutants"))
        and configuration.get("mutants") == len(MUTANT_IDS)
        and configuration.get("mutant_rank_assignment")
        == {
            str(rank): list(ids)
            for rank, ids in MUTANT_ASSIGNMENT_BY_RANK.items()
        }
        and "separately rebuilt, no-mutation matched control"
        in configuration.get("mutant_case_isolation", ""),
        "review-response formal configuration drift",
    )
    _require(
        any(
            "independently pre-fixed 0.005" in baseline
            and "does not tune" in baseline
            for baseline in experiment.get("baselines", [])
        ),
        "review plan no longer treats 0.005 as independently pre-fixed",
    )
    return dict(value)


def _strict_bound_json_bytes(
    raw: bytes,
    *,
    expected_sha256: str,
    label: str,
) -> Any:
    _require(
        isinstance(raw, bytes)
        and sha256_bytes(raw) == _require_sha256(expected_sha256, label),
        f"{label} raw-byte SHA drift",
    )
    return strict_json_loads(raw, label=label)


def _formal_input_provenance_from_raw(
    *,
    identity: Mapping[str, Any],
    rr2_input_manifest_raw: bytes,
    prior_fp32_context_manifest_raw: bytes,
    review_response_plan_raw: bytes,
) -> dict[str, Any]:
    rr2 = _strict_bound_json_bytes(
        rr2_input_manifest_raw,
        expected_sha256=identity["pg19_input_manifest_sha256"],
        label="RR2 input manifest",
    )
    try:
        from build_qcomem_forkaudit_rr2_input_manifest import (
            validate_rr2_input_manifest,
        )

        validate_rr2_input_manifest(rr2)
    except Exception as exc:
        raise ReviewAuditError(f"RR2 input manifest replay failed: {exc}") from exc
    prior = _strict_bound_json_bytes(
        prior_fp32_context_manifest_raw,
        expected_sha256=PRIOR_FP32_CONTEXT_RAW_SHA256,
        label="prior FP32 context manifest",
    )
    _validate_prior_fp32_context_manifest(prior)
    plan = _strict_bound_json_bytes(
        review_response_plan_raw,
        expected_sha256=FINAL_REVIEW_RESPONSE_PLAN_SHA256,
        label="review-response experiment plan",
    )
    _validate_review_response_plan(plan)
    return {
        "mode": "formal_preoutput_inputs",
        "rr2_input_manifest_raw_sha256": sha256_bytes(rr2_input_manifest_raw),
        "rr2_input_manifest_raw_bytes": len(rr2_input_manifest_raw),
        "rr2_input_manifest_raw_base64": base64.b64encode(
            rr2_input_manifest_raw
        ).decode("ascii"),
        "prior_fp32_context_manifest_raw_sha256": PRIOR_FP32_CONTEXT_RAW_SHA256,
        "prior_fp32_context_manifest_raw_bytes": len(
            prior_fp32_context_manifest_raw
        ),
        "prior_fp32_context_manifest_raw_base64": base64.b64encode(
            prior_fp32_context_manifest_raw
        ).decode("ascii"),
        "review_response_plan_raw_sha256": FINAL_REVIEW_RESPONSE_PLAN_SHA256,
        "review_response_plan_raw_bytes": len(review_response_plan_raw),
        "review_response_plan_raw_base64": base64.b64encode(
            review_response_plan_raw
        ).decode("ascii"),
        "rr2_input_manifest_semantic_sha256": sha256_json(rr2),
        "prior_fp32_context_semantic_sha256": sha256_json(prior),
        "review_response_plan_semantic_sha256": sha256_json(plan),
    }


def make_static_artifact(
    identity: Mapping[str, Any],
    oracle_selection_plan: Sequence[Mapping[str, Any]],
    frozen_query_banks: Sequence[Mapping[str, Any]],
    *,
    rr2_input_manifest_raw: bytes | None = None,
    prior_fp32_context_manifest_raw: bytes | None = None,
    review_response_plan_raw: bytes | None = None,
) -> dict[str, Any]:
    frozen = validate_frozen_identity(identity)
    plan = validate_oracle_selection_plan(list(oracle_selection_plan))
    banks = validate_frozen_query_banks(list(frozen_query_banks), plan)
    raw_inputs = (
        rr2_input_manifest_raw,
        prior_fp32_context_manifest_raw,
        review_response_plan_raw,
    )
    if all(item is None for item in raw_inputs):
        provenance = {"mode": "synthetic_schema_fixture"}
    else:
        _require(
            all(isinstance(item, bytes) for item in raw_inputs),
            "formal static provenance requires all three raw manifests",
        )
        provenance = _formal_input_provenance_from_raw(
            identity=frozen,
            rr2_input_manifest_raw=rr2_input_manifest_raw,  # type: ignore[arg-type]
            prior_fp32_context_manifest_raw=prior_fp32_context_manifest_raw,  # type: ignore[arg-type]
            review_response_plan_raw=review_response_plan_raw,  # type: ignore[arg-type]
        )
        rr2 = strict_json_loads(
            rr2_input_manifest_raw, label="RR2 input manifest"  # type: ignore[arg-type]
        )
        _require(
            rr2.get("oracle_selection_plan") == plan
            and rr2.get("frozen_query_banks") == banks,
            "RR2 main manifest differs from supplied plan/query sidecars",
        )
    return {
        "schema_version": STATIC_SCHEMA_VERSION,
        "status": "static_protocol_validated",
        "passed": True,
        "formal_ready": False,
        "implementation_status": IMPLEMENTATION_STATUS,
        "protocol": PROTOCOL,
        "protocol_config": formal_protocol_config(),
        "protocol_config_sha256": sha256_json(formal_protocol_config()),
        "frozen_identity": frozen,
        "oracle_selection_plan": plan,
        "oracle_selection_plan_sha256": sha256_json(plan),
        "frozen_query_banks": banks,
        "frozen_query_banks_sha256": sha256_json(banks),
        "input_provenance": provenance,
        "claim_boundary": "static/schema validation only; no GPU evidence produced",
    }


def _replay_static_input_provenance(
    value: Any,
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    _require(isinstance(value, dict), "static input provenance missing")
    if value == {"mode": "synthetic_schema_fixture"}:
        return None
    required = {
        "mode",
        "rr2_input_manifest_raw_sha256",
        "rr2_input_manifest_raw_bytes",
        "rr2_input_manifest_raw_base64",
        "prior_fp32_context_manifest_raw_sha256",
        "prior_fp32_context_manifest_raw_bytes",
        "prior_fp32_context_manifest_raw_base64",
        "review_response_plan_raw_sha256",
        "review_response_plan_raw_bytes",
        "review_response_plan_raw_base64",
        "rr2_input_manifest_semantic_sha256",
        "prior_fp32_context_semantic_sha256",
        "review_response_plan_semantic_sha256",
    }
    _require(
        set(value) == required and value.get("mode") == "formal_preoutput_inputs",
        "formal static input-provenance schema drift",
    )

    def decode(field: str, label: str) -> bytes:
        encoded = value[field]
        _require(isinstance(encoded, str) and bool(encoded), f"{label} base64 missing")
        try:
            return base64.b64decode(encoded.encode("ascii"), validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise ReviewAuditError(f"{label} base64 drift") from exc

    rr2_raw = decode("rr2_input_manifest_raw_base64", "RR2 input manifest")
    prior_raw = decode(
        "prior_fp32_context_manifest_raw_base64", "prior FP32 context manifest"
    )
    plan_raw = decode(
        "review_response_plan_raw_base64", "review-response experiment plan"
    )
    replayed = _formal_input_provenance_from_raw(
        identity=identity,
        rr2_input_manifest_raw=rr2_raw,
        prior_fp32_context_manifest_raw=prior_raw,
        review_response_plan_raw=plan_raw,
    )
    _require(value == replayed, "static raw input-provenance replay drift")
    return strict_json_loads(rr2_raw, label="RR2 input manifest")


def validate_static_artifact(payload: Any) -> dict[str, Any]:
    _require(isinstance(payload, dict), "static artifact must be an object")
    _require(payload.get("schema_version") == STATIC_SCHEMA_VERSION, "static schema drift")
    _require(payload.get("status") == "static_protocol_validated", "static status drift")
    _require(payload.get("passed") is True, "static validation did not pass")
    _require(payload.get("formal_ready") is False, "audit skeleton cannot be formal-ready")
    _require(payload.get("implementation_status") == IMPLEMENTATION_STATUS, "implementation status drift")
    _require(payload.get("protocol") == PROTOCOL, "static protocol drift")
    config = payload.get("protocol_config")
    _require(config == formal_protocol_config(), "formal protocol config drift")
    _require(payload.get("protocol_config_sha256") == sha256_json(config), "static config SHA drift")
    identity = validate_frozen_identity(payload.get("frozen_identity"))
    rr2 = _replay_static_input_provenance(
        payload.get("input_provenance"), identity=identity
    )
    plan = validate_oracle_selection_plan(payload.get("oracle_selection_plan"))
    _require(payload.get("oracle_selection_plan_sha256") == sha256_json(plan), "oracle selection-plan SHA drift")
    banks = validate_frozen_query_banks(payload.get("frozen_query_banks"), plan)
    _require(payload.get("frozen_query_banks_sha256") == sha256_json(banks), "frozen query-bank SHA drift")
    if rr2 is not None:
        _require(
            rr2.get("oracle_selection_plan") == plan
            and rr2.get("frozen_query_banks") == banks,
            "static plan/banks differ from bound RR2 main manifest",
        )
    return {
        "protocol_config": config,
        "frozen_identity": identity,
        "oracle_selection_plan": plan,
        "frozen_query_banks": banks,
        "formal_input_provenance_bound": rr2 is not None,
    }


def bridge_named_gate_error(call: Callable[[], Any]) -> Any:
    """Translate witness/oracle gates into the mutant harness gate type.

    Only errors raised by ``call`` (the detector exercise boundary) should be
    passed here.  Injector construction and restoration must remain outside
    this bridge so they cannot impersonate successful detection.
    """

    try:
        return call()
    except (GDNStorageWitnessError, OracleGateError) as exc:
        raise RuntimeInvariantError(str(exc.gate_id), str(exc)) from exc


def run_shard_not_implemented(*_args: Any, **_kwargs: Any) -> None:
    raise ProductionLoopNotImplemented(
        "formal GPU shard loop is not implemented in this audit skeleton; "
        "no shard or passed=true artifact was written"
    )


def _gpu_allocator_snapshot() -> dict[str, int]:
    """Capture allocator counters without importing/initializing CUDA at import."""

    import torch

    torch.cuda.synchronize()
    return {
        "current_allocated_bytes": int(torch.cuda.memory_allocated()),
        "current_reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def _gpu_cleanup() -> dict[str, int]:
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return _gpu_allocator_snapshot()


def _token_id_sha256(
    tensor: Any, *, expected_shape: Sequence[int] | None = None
) -> str:
    """Match the RR2 builder's SHA over contiguous int64 token bytes."""

    import torch

    _require(
        isinstance(tensor, torch.Tensor) and tensor.dtype == torch.int64,
        "token-ID digest requires a torch.int64 tensor",
    )
    value = tensor.detach().contiguous().cpu()
    if expected_shape is not None:
        _require(
            tuple(value.shape) == tuple(expected_shape),
            "token-ID tensor shape differs from the frozen input geometry",
        )
    return sha256_bytes(value.view(torch.uint8).numpy().tobytes())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(8 * 1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as exc:
        raise ReviewAuditError(f"cannot hash required file: {path.name}") from exc
    return digest.hexdigest()


def _load_expected_bytes(path: Path, expected_sha256: Any, *, label: str) -> bytes:
    expected = _require_sha256(expected_sha256, f"expected {label} SHA")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ReviewAuditError(f"{label} cannot be read") from exc
    _require(sha256_bytes(payload) == expected, f"{label} raw-byte SHA drift")
    return payload


def _parse_sha256_ledger(payload: bytes, *, label: str) -> list[dict[str, str]]:
    """Parse one path-independent, C-sorted sha256sum ledger."""

    try:
        value = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReviewAuditError(f"{label} is not UTF-8") from exc
    _require(value.endswith("\n"), f"{label} must end with one newline")
    rows: list[dict[str, str]] = []
    names: list[str] = []
    line_re = re.compile(r"^([0-9a-f]{64}) ([ *])(.+)$")
    for line_index, line in enumerate(value.splitlines(), start=1):
        match = line_re.fullmatch(line)
        _require(match is not None, f"{label} line {line_index} is not sha256sum format")
        assert match is not None
        digest, marker, raw_name = match.groups()
        _require(marker == " ", f"{label} contains a binary-mode marker")
        _require("\\" not in raw_name and "\x00" not in raw_name, f"{label} contains a non-POSIX path")
        while raw_name.startswith("./"):
            raw_name = raw_name[2:]
        pure = PurePosixPath(raw_name)
        _require(
            raw_name
            and not pure.is_absolute()
            and ".." not in pure.parts
            and "." not in pure.parts
            and pure.as_posix() == raw_name,
            f"{label} contains a non-normalized or escaping path",
        )
        names.append(raw_name)
        rows.append({"logical_name": raw_name, "sha256": digest})
    _require(bool(rows), f"{label} is empty")
    _require(len(names) == len(set(names)), f"{label} repeats a logical name")
    _require(
        names == sorted(names, key=lambda item: item.encode("utf-8")),
        f"{label} is not LC_ALL=C sorted",
    )
    return rows


def _verify_model_ledger(
    rows: Sequence[Mapping[str, str]], *, model_dir: Path, label: str
) -> dict[str, Any]:
    """Hash every listed model file; ledger bytes alone are not authority."""

    verified: list[dict[str, Any]] = []
    for row in rows:
        name = row["logical_name"]
        path = model_dir.joinpath(*PurePosixPath(name).parts)
        _require(path.is_file(), f"{label} file is missing: {name}")
        observed = _sha256_file(path)
        _require(observed == row["sha256"], f"{label} content drift: {name}")
        verified.append(
            {
                "logical_name": name,
                "sha256": observed,
                "bytes": int(path.stat().st_size),
            }
        )
    return {
        "entry_count": len(verified),
        "entries_sha256": sha256_json(verified),
    }


def _verify_weight_ledger_structure(
    rows: Sequence[Mapping[str, str]], *, model_dir: Path
) -> dict[str, Any]:
    """Bind the 14-shard ledger to the loader index without 8x full rehashing.

    Content authority comes from the launcher-owned Linux ModelLoadLease-v1
    guard over a private independent-inode view.  Each rank pins that external
    authority and checks the exact loader index/file sizes without repeating a
    roughly 70 GB payload hash.  This structural receipt is never treated as
    content authority by itself.
    """

    _require(len(rows) == 14, "formal model weight ledger must bind 14 shards")
    index_path = model_dir / "model.safetensors.index.json"
    try:
        index = strict_json_loads(index_path.read_bytes(), label="model weight index")
    except OSError as exc:
        raise ReviewAuditError("model weight index cannot be read") from exc
    _require(isinstance(index, dict), "model weight index must be an object")
    weight_map = index.get("weight_map")
    _require(isinstance(weight_map, dict) and bool(weight_map), "model weight map missing")
    indexed_names = set(weight_map.values())
    ledger_names = {row["logical_name"] for row in rows}
    _require(
        all(isinstance(name, str) for name in indexed_names)
        and indexed_names == ledger_names,
        "weight ledger names differ from the loader index",
    )
    files = []
    for row in rows:
        name = row["logical_name"]
        _require(
            re.fullmatch(
                r"model\.safetensors-[0-9]{5}-of-00014\.safetensors", name
            )
            is not None,
            f"unexpected weight-shard logical name: {name}",
        )
        path = model_dir / name
        _require(path.is_file(), f"indexed weight shard missing: {name}")
        size = int(path.stat().st_size)
        _require(size > 0, f"indexed weight shard is empty: {name}")
        files.append(
            {"logical_name": name, "sha256": row["sha256"], "bytes": size}
        )
    metadata = index.get("metadata")
    if isinstance(metadata, dict) and "total_size" in metadata:
        total_file_bytes = sum(row["bytes"] for row in files)
        _require(
            _is_int(metadata["total_size"])
            and 0 < metadata["total_size"] <= total_file_bytes
            and total_file_bytes - metadata["total_size"] <= 128 * 1024 * 1024,
            "weight index tensor size is inconsistent with shard file sizes",
        )
    return {
        "entry_count": len(files),
        "indexed_files_sha256": sha256_json(files),
        "per_rank_full_weight_rehash_performed": False,
    }


def _load_canonical_lf_receipt(
    path: Path, expected_raw_sha256: Any, *, label: str
) -> tuple[dict[str, Any], bytes, str]:
    payload = _load_expected_bytes(path, expected_raw_sha256, label=label)
    value = strict_json_loads(payload, label=label)
    _require(isinstance(value, dict), f"{label} must be an object")
    _require(
        payload == canonical_json_bytes(value) + b"\n",
        f"{label} must use canonical JSON plus one LF",
    )
    return dict(value), payload, sha256_bytes(payload)


def _validate_gpu_assignment_receipt(
    value: Any, *, raw_sha256: str, raw_bytes: bytes
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "world_size",
        "inventory_query",
        "rows",
        "unique_visible_indices",
        "unique_uuids",
        "all_h20",
        "all_compute_capability_9_0",
        "generated_before_candidate_outputs",
    }
    _require(
        isinstance(value, dict)
        and set(value) == fields
        and raw_bytes == canonical_json_bytes(value) + b"\n"
        and sha256_bytes(raw_bytes)
        == _require_sha256(raw_sha256, "GPU-assignment receipt raw SHA"),
        "GPU-assignment receipt raw/schema drift",
    )
    rows = value["rows"]
    _require(
        value["schema_version"] == GPU_ASSIGNMENT_RECEIPT_SCHEMA_VERSION
        and _is_int(value["world_size"])
        and value["world_size"] == FORMAL_WORLD_SIZE
        and value["inventory_query"]
        == "index,uuid,name,memory.total,compute_cap"
        and value["unique_visible_indices"] is True
        and value["unique_uuids"] is True
        and value["all_h20"] is True
        and value["all_compute_capability_9_0"] is True
        and value["generated_before_candidate_outputs"] is True
        and isinstance(rows, list)
        and len(rows) == FORMAL_WORLD_SIZE,
        "GPU-assignment receipt contract drift",
    )
    row_fields = {
        "rank",
        "visible_index",
        "uuid",
        "name",
        "total_memory_mib",
        "compute_capability",
        "bf16_supported",
    }
    normalized: list[dict[str, Any]] = []
    for rank, row in enumerate(rows):
        _require(
            isinstance(row, dict)
            and set(row) == row_fields
            and _is_int(row["rank"])
            and row["rank"] == rank
            and _is_int(row["visible_index"])
            and row["visible_index"] >= 0
            and isinstance(row["uuid"], str)
            and re.fullmatch(r"GPU-[0-9a-fA-F-]+", row["uuid"]) is not None
            and isinstance(row["name"], str)
            and "H20" in row["name"]
            and _is_int(row["total_memory_mib"])
            and row["total_memory_mib"] > 0
            and row["compute_capability"] == [9, 0]
            and all(_is_int(item) for item in row["compute_capability"])
            and row["bf16_supported"] is True,
            "GPU-assignment row drift",
        )
        normalized.append(dict(row))
    _require(
        len({row["visible_index"] for row in normalized}) == FORMAL_WORLD_SIZE
        and len({row["uuid"] for row in normalized}) == FORMAL_WORLD_SIZE,
        "GPU assignment is not one-to-one",
    )
    return dict(value)


def _validate_private_model_view_manifest(
    value: Any,
    *,
    raw_bytes: bytes,
    expected_raw_sha256: str,
    model_artifact_rows: Sequence[Mapping[str, str]],
    model_weight_rows: Sequence[Mapping[str, str]],
    model_artifact_ledger_raw_sha256: str,
    model_weight_ledger_raw_sha256: str,
    model_view: Path | None = None,
    model_load_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Blindly bind the private view, its source-copy proof, and lease rows.

    The manifest is external pre-output state.  A shard additionally compares
    every live path against its recorded view inode; aggregate replay can omit
    ``model_view`` while retaining the raw-byte, ledger, and authority joins.
    """

    fields = {
        "schema_version",
        "model_id",
        "model_revision",
        "model_artifact_ledger_raw_sha256",
        "model_weight_ledger_raw_sha256",
        "copy_policy",
        "file_count",
        "weight_file_count",
        "all_source_and_view_inodes_distinct",
        "all_view_files_regular",
        "all_view_files_read_only",
        "rows",
        "generated_before_candidate_outputs",
        "cuda_initialized",
    }
    expected_raw_sha = _require_sha256(
        expected_raw_sha256, "private model-view manifest raw SHA"
    )
    _require(
        isinstance(value, dict)
        and set(value) == fields
        and raw_bytes == canonical_json_bytes(value) + b"\n"
        and sha256_bytes(raw_bytes) == expected_raw_sha,
        "private model-view manifest raw/schema drift",
    )
    _require(
        value["schema_version"] == PRIVATE_MODEL_VIEW_SCHEMA_VERSION
        and value["model_id"] == FORMAL_MODEL_ID
        and value["model_revision"] == FORMAL_MODEL_REVISION
        and value["model_artifact_ledger_raw_sha256"]
        == _require_sha256(
            model_artifact_ledger_raw_sha256,
            "private-view model-artifact ledger raw SHA",
        )
        and value["model_weight_ledger_raw_sha256"]
        == _require_sha256(
            model_weight_ledger_raw_sha256,
            "private-view model-weight ledger raw SHA",
        )
        and value["copy_policy"]
        == "ficlone-then-byte-copy;hardlink-and-symlink-forbidden"
        and _is_int(value["file_count"])
        and value["file_count"] > 0
        and _is_int(value["weight_file_count"])
        and value["weight_file_count"] == 14
        and value["all_source_and_view_inodes_distinct"] is True
        and value["all_view_files_regular"] is True
        and value["all_view_files_read_only"] is True
        and value["generated_before_candidate_outputs"] is True
        and value["cuda_initialized"] is False,
        "private model-view manifest binding drift",
    )

    expected_by_name: dict[str, dict[str, Any]] = {}
    for role, ledger_rows in (
        ("model_artifact", model_artifact_rows),
        ("model_weight", model_weight_rows),
    ):
        for ledger_row in ledger_rows:
            name = ledger_row["logical_name"]
            digest = _require_sha256(
                ledger_row["sha256"], f"private-view ledger {name}"
            )
            metadata = expected_by_name.setdefault(
                name, {"declared_sha256": digest, "ledger_roles": []}
            )
            _require(
                metadata["declared_sha256"] == digest,
                f"private-view ledgers disagree for {name}",
            )
            metadata["ledger_roles"].append(role)

    rows = value["rows"]
    row_fields = {
        "relative_path",
        "ledger_roles",
        "declared_sha256",
        "bytes",
        "copy_mode",
        "source_device",
        "source_inode",
        "view_device",
        "view_inode",
        "source_and_view_inode_distinct",
    }
    _require(
        isinstance(rows, list)
        and len(rows) == value["file_count"] == len(expected_by_name),
        "private model-view manifest row count drift",
    )
    resolved_model_view: Path | None = None
    if model_view is not None:
        try:
            model_view_lstat = model_view.lstat()
            resolved_model_view = model_view.resolve(strict=True)
        except OSError as exc:
            raise ReviewAuditError("private model-view root cannot be inspected") from exc
        _require(
            stat.S_ISDIR(model_view_lstat.st_mode)
            and not stat.S_ISLNK(model_view_lstat.st_mode)
            and model_view_lstat.st_mode & 0o222 == 0,
            "private model-view root must be a real directory",
        )
    names: list[str] = []
    weight_rows: list[dict[str, Any]] = []
    for row in rows:
        _require(
            isinstance(row, dict) and set(row) == row_fields,
            "private model-view row schema drift",
        )
        name = row["relative_path"]
        pure = PurePosixPath(name) if isinstance(name, str) else None
        _require(
            isinstance(name, str)
            and bool(name)
            and pure is not None
            and not pure.is_absolute()
            and ".." not in pure.parts
            and "." not in pure.parts
            and pure.as_posix() == name
            and "\\" not in name
            and "\x00" not in name,
            "private model-view path is not normalized",
        )
        expected = expected_by_name.get(name)
        _require(expected is not None, "private model-view row is absent from ledgers")
        roles = row["ledger_roles"]
        _require(
            isinstance(roles, list)
            and roles == sorted(set(roles))
            and roles == sorted(expected["ledger_roles"])
            and row["declared_sha256"] == expected["declared_sha256"],
            "private model-view row differs from frozen ledgers",
        )
        for integer_field in (
            "bytes",
            "source_device",
            "source_inode",
            "view_device",
            "view_inode",
        ):
            _require(
                _is_int(row[integer_field]) and row[integer_field] > 0,
                f"private model-view {integer_field} drift",
            )
        _require(
            row["copy_mode"] in {"ficlone", "byte-copy"}
            and row["source_and_view_inode_distinct"] is True
            and (row["source_device"], row["source_inode"])
            != (row["view_device"], row["view_inode"]),
            "private model-view copy provenance drift",
        )
        if model_view is not None:
            path = model_view.joinpath(*pure.parts)
            try:
                current = model_view
                for component in pure.parts[:-1]:
                    current = current / component
                    component_stat = current.lstat()
                    _require(
                        stat.S_ISDIR(component_stat.st_mode)
                        and not stat.S_ISLNK(component_stat.st_mode)
                        and component_stat.st_mode & 0o222 == 0,
                        f"private model-view directory is not real: {component}",
                    )
                observed = path.lstat()
                resolved_path = path.resolve(strict=True)
            except OSError as exc:
                raise ReviewAuditError(
                    f"private model-view file cannot be inspected: {name}"
                ) from exc
            _require(
                stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and int(observed.st_size) == row["bytes"]
                and int(observed.st_dev) == row["view_device"]
                and int(observed.st_ino) == row["view_inode"]
                and observed.st_mode & 0o222 == 0,
                f"private model-view live file drift: {name}",
            )
            _require(
                resolved_model_view is not None
                and resolved_path.is_relative_to(resolved_model_view),
                f"private model-view file escaped its root: {name}",
            )
        if "model_weight" in roles:
            weight_rows.append(dict(row))
        names.append(name)
    _require(
        names == sorted(names, key=lambda item: item.encode("utf-8"))
        and len(names) == len(set(names))
        and len(weight_rows) == value["weight_file_count"],
        "private model-view row order/weight coverage drift",
    )

    if model_load_authority is not None:
        authority_rows = model_load_authority.get("rows")
        _require(
            model_load_authority.get("model_view_manifest_sha256")
            == expected_raw_sha
            and isinstance(authority_rows, list)
            and len(authority_rows) == len(weight_rows),
            "ModelLoadLease authority/private-view manifest binding drift",
        )
        projected = []
        for row in weight_rows:
            projected.append(
                {
                    "logical_name": row["relative_path"],
                    "declared_sha256": row["declared_sha256"],
                    "bytes": row["bytes"],
                    "st_dev": row["view_device"],
                    "st_ino": row["view_inode"],
                }
            )
        authority_projected = []
        for row in authority_rows:
            row_stat = row.get("stat") if isinstance(row, dict) else None
            _require(
                isinstance(row_stat, dict),
                "ModelLoadLease authority row stat missing",
            )
            authority_projected.append(
                {
                    "logical_name": row.get("logical_name"),
                    "declared_sha256": row.get("declared_sha256"),
                    "bytes": row_stat.get("bytes"),
                    "st_dev": row_stat.get("st_dev"),
                    "st_ino": row_stat.get("st_ino"),
                }
            )
        _require(
            authority_projected == projected,
            "ModelLoadLease authority rows differ from the private view",
        )
    return dict(value)


def _model_manifest_digest(model_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    names = ("config.json", "generation_config.json", "model.safetensors.index.json")
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for name in names:
        path = model_dir / name
        _require(path.is_file(), f"model manifest component missing: {name}")
        observed = _sha256_file(path)
        size = int(path.stat().st_size)
        digest.update(f"{name}\0{observed}\0{size}\n".encode("utf-8"))
        rows.append({"logical_name": name, "sha256": observed, "bytes": size})
    return digest.hexdigest(), rows


def _validate_run_id_receipt(
    payload: bytes,
    *,
    expected_sha256: str,
    run_id: str,
    static_artifact_sha256: str,
    protocol_manifest_sha256: str,
) -> dict[str, Any]:
    value = strict_json_loads(payload, label="run-ID receipt")
    _require(
        sha256_json(value)
        == _require_sha256(expected_sha256, "expected run-ID receipt SHA"),
        "run-ID receipt canonical SHA drift",
    )
    required = {
        "schema_version",
        "run_id",
        "run_id_bits",
        "derivation",
        "domain_hex",
        "static_artifact_sha256",
        "protocol_manifest_sha256",
        "nonce_hex",
        "generated_once_after_static_before_candidate_outputs",
    }
    _require(isinstance(value, dict) and set(value) == required, "run-ID receipt schema drift")
    nonce = value["nonce_hex"]
    _require(
        value["schema_version"] == "qcomem-forkaudit-run-id-receipt-v1"
        and value["run_id"] == _require_run_id(run_id, "formal shard run ID")
        and _is_int(value["run_id_bits"])
        and value["run_id_bits"] == 128
        and value["derivation"]
        == "sha256(domain || static_sha256 || protocol_sha256 || nonce)[:16]"
        and value["domain_hex"] == b"qcomem-forkaudit-run-id-v1\0".hex()
        and value["static_artifact_sha256"] == static_artifact_sha256
        and value["protocol_manifest_sha256"] == protocol_manifest_sha256
        and isinstance(nonce, str)
        and re.fullmatch(r"[0-9a-f]{64}", nonce) is not None
        and value["generated_once_after_static_before_candidate_outputs"] is True,
        "run-ID receipt binding drift",
    )
    recomputed = hashlib.sha256(
        bytes.fromhex(value["domain_hex"])
        + bytes.fromhex(static_artifact_sha256)
        + bytes.fromhex(protocol_manifest_sha256)
        + bytes.fromhex(nonce)
    ).hexdigest()[:32]
    _require(recomputed == run_id, "run-ID receipt derivation drift")
    return dict(value)


@dataclass(frozen=True)
class FormalInputBundle:
    static_payload: dict[str, Any]
    static_replay: dict[str, Any]
    static_sha256: str
    rr2_manifest: dict[str, Any]
    document_cpu: Any
    queries_cpu: tuple[Any, ...]
    data_usage: dict[str, Any]
    input_rebuild_receipt: dict[str, Any]
    run_id_receipt: dict[str, Any]
    gpu_assignment_receipt: dict[str, Any]
    gpu_assignment_receipt_raw_sha256: str
    gpu_assignment_row: dict[str, Any]
    private_model_view_manifest: dict[str, Any]
    private_model_view_manifest_raw_sha256: str
    model_load_authority: dict[str, Any]
    model_load_authority_raw_sha256: str
    model_integrity_before_load: dict[str, Any]
    model_artifact_rows: tuple[dict[str, str], ...]
    model_weight_rows: tuple[dict[str, str], ...]


def _load_formal_input_bundle(args: argparse.Namespace) -> FormalInputBundle:
    """Rebuild formal tokens from raw bytes before any CUDA initialization."""

    import torch
    import build_qcomem_forkaudit_rr2_input_manifest as rr2_builder
    import qcomem_joint_policy as joint_policy
    from qcomem_forkaudit_model_load_lease import (
        ModelLoadLeaseError,
        authority_from_canonical_bytes,
    )

    _require(not torch.cuda.is_initialized(), "formal input rebuild must precede CUDA initialization")
    _require(_is_int(args.rank) and 0 <= args.rank < FORMAL_WORLD_SIZE, "formal shard rank drift")
    static_raw = args.static_artifact.read_bytes()
    static_payload = strict_json_loads(static_raw, label="static artifact")
    static_sha = _require_sha256(args.expected_static_sha256, "expected static artifact SHA")
    _require(sha256_json(static_payload) == static_sha, "static artifact canonical SHA drift")
    static_replay = validate_static_artifact(static_payload)
    _require(
        static_replay["formal_input_provenance_bound"] is True,
        "formal shard requires raw-bound static provenance",
    )
    identity = static_replay["frozen_identity"]

    rr2_raw = _load_expected_bytes(
        args.rr2_input_manifest,
        args.expected_rr2_input_manifest_sha256,
        label="RR2 input manifest",
    )
    _require(
        sha256_bytes(rr2_raw) == identity["pg19_input_manifest_sha256"],
        "RR2 input manifest differs from frozen identity",
    )
    rr2_manifest = rr2_builder.validate_rr2_input_manifest(
        rr2_builder.strict_json_loads(rr2_raw, label="RR2 input manifest")
    )
    _require(
        rr2_manifest["frozen_query_banks"] == static_replay["frozen_query_banks"]
        and rr2_manifest["oracle_selection_plan"]
        == static_replay["oracle_selection_plan"],
        "RR2 main manifest differs from static banks/selection",
    )

    # Every provenance ledger is read explicitly; no ambient launcher env is
    # accepted.  The two model ledgers are also replayed against actual files.
    code_raw = _load_expected_bytes(
        args.code_ledger, identity["code_ledger_sha256"], label="code ledger"
    )
    protocol_raw = _load_expected_bytes(
        args.protocol_manifest,
        identity["protocol_manifest_sha256"],
        label="protocol manifest",
    )
    artifact_raw = _load_expected_bytes(
        args.model_artifact_ledger,
        identity["model_artifact_ledger_sha256"],
        label="model artifact ledger",
    )
    weight_raw = _load_expected_bytes(
        args.model_weight_ledger,
        identity["model_weight_ledger_sha256"],
        label="model weight ledger",
    )
    code_rows = _parse_sha256_ledger(code_raw, label="code ledger")
    artifact_rows = _parse_sha256_ledger(
        artifact_raw, label="model artifact ledger"
    )
    weight_rows = _parse_sha256_ledger(weight_raw, label="model weight ledger")
    _require(len(weight_rows) == 14, "formal model weight ledger must bind 14 shards")
    private_view_value, private_view_raw, private_view_raw_sha = (
        _load_canonical_lf_receipt(
            args.private_model_view_manifest,
            args.expected_private_model_view_manifest_raw_sha256,
            label="private model-view manifest",
        )
    )
    authority_raw = _load_expected_bytes(
        args.model_load_authority,
        args.expected_model_load_authority_raw_sha256,
        label="ModelLoadLease authority",
    )
    authority_raw_sha = sha256_bytes(authority_raw)
    try:
        model_load_authority = authority_from_canonical_bytes(
            authority_raw, authority_raw_sha
        )
    except ModelLoadLeaseError as exc:
        raise ReviewAuditError("ModelLoadLease authority rejected") from exc
    _require(
        model_load_authority["run_id"] == args.run_id
        and model_load_authority["weight_ledger_raw_sha256"]
        == sha256_bytes(weight_raw)
        and model_load_authority["model_artifact_ledger_raw_sha256"]
        == sha256_bytes(artifact_raw)
        and [
            {
                "logical_name": row["logical_name"],
                "sha256": row["declared_sha256"],
            }
            for row in model_load_authority["rows"]
        ]
        == [dict(row) for row in weight_rows],
        "ModelLoadLease authority differs from run/model ledgers",
    )
    private_model_view_manifest = _validate_private_model_view_manifest(
        private_view_value,
        raw_bytes=private_view_raw,
        expected_raw_sha256=private_view_raw_sha,
        model_artifact_rows=artifact_rows,
        model_weight_rows=weight_rows,
        model_artifact_ledger_raw_sha256=sha256_bytes(artifact_raw),
        model_weight_ledger_raw_sha256=sha256_bytes(weight_raw),
        model_view=args.model_dir,
        model_load_authority=model_load_authority,
    )
    artifact_audit = _verify_model_ledger(
        artifact_rows, model_dir=args.model_dir, label="model artifact ledger"
    )
    weight_audit = _verify_weight_ledger_structure(
        weight_rows, model_dir=args.model_dir
    )
    model_manifest_sha, model_manifest_rows = _model_manifest_digest(args.model_dir)
    _require(
        model_manifest_sha == identity["model_manifest_sha256"],
        "model manifest differs from frozen identity",
    )
    independently_observed_artifacts = rr2_builder.audit_model_tokenizer_artifacts(
        args.model_dir
    )
    embedded_artifacts = rr2_manifest["model"][
        "model_and_tokenizer_artifacts"
    ]
    _require(
        embedded_artifacts == independently_observed_artifacts,
        "RR2 tokenizer artifacts differ from the model directory",
    )
    artifact_ledger_map = {
        row["logical_name"]: row["sha256"] for row in artifact_rows
    }
    for row in independently_observed_artifacts["artifacts"]:
        _require(
            artifact_ledger_map.get(row["logical_name"]) == row["sha256"],
            f"model artifact ledger disagrees with RR2 input {row['logical_name']}",
        )

    tokenizer = rr2_builder.load_local_tokenizer(args.model_dir)
    rebuilt = rr2_builder.build_from_paths(
        pg19_data=args.pg19_data,
        pg19_manifest=args.pg19_manifest,
        prior_capacity_manifest=args.prior_capacity_manifest,
        model_dir=args.model_dir,
        tokenizer=tokenizer,
    )
    rebuilt_raw = rr2_builder.canonical_json_bytes(rebuilt) + b"\n"
    _require(rebuilt_raw == rr2_raw, "source-rebuilt RR2 manifest is not byte-identical")

    data_bytes = args.pg19_data.read_bytes()
    pg19_manifest_bytes = args.pg19_manifest.read_bytes()
    records, _data_audit = rr2_builder._audit_pg19_train64_bytes(
        data_bytes,
        pg19_manifest_bytes,
        expectations=rr2_builder.FORMAL_EXPECTATIONS,
    )
    windows, windows_sha = joint_policy.build_pg19_calibration_windows(
        records,
        tokenizer,
        books=FORMAL_BOOKS,
        document_tokens=FORMAL_DOCUMENT_TOKENS,
        query_tokens=FORMAL_QUERY_TOKENS,
        stride=FORMAL_WINDOW_STRIDE,
        candidate_windows_per_book=8,
        seed=20260817,
    )
    _require(windows_sha == FORMAL_RR2_WINDOWS_SHA256, "runtime RR2 window digest drift")
    window = windows[args.rank]
    bank = static_replay["frozen_query_banks"][args.rank]
    _require(
        window.source_object == bank["source_object"]
        and int(window.start_token) == bank["document_start_token"],
        "runtime rank window differs from frozen coordinate",
    )
    document_cpu = window.document_ids.detach().contiguous().unsqueeze(0)
    query_values, query_audit = build_pg19_train_query_bank(
        records,
        tokenizer,
        window,
        document_tokens=FORMAL_DOCUMENT_TOKENS,
        query_tokens=FORMAL_QUERY_TOKENS,
        count=max(FORMAL_RESIDENT_COUNTS),
        query_stride=FORMAL_QUERY_BANK_STRIDE,
    )
    queries_cpu = tuple(query.detach().contiguous() for query in query_values)
    document_sha = _token_id_sha256(
        document_cpu, expected_shape=(1, FORMAL_DOCUMENT_TOKENS)
    )
    query_rows = [
        {
            "request_index": request_index,
            "sha256": _token_id_sha256(
                query, expected_shape=(1, FORMAL_QUERY_TOKENS)
            ),
        }
        for request_index, query in enumerate(queries_cpu)
    ]
    _require(document_sha == bank["document_token_ids_sha256"], "runtime document token bytes drift")
    _require(
        [row["sha256"] for row in query_rows]
        == [row["query_token_ids_sha256"] for row in bank["rows"]],
        "runtime query token bytes differ from frozen bank",
    )
    _require(
        [
            {
                "request_index": int(row["request_index"]),
                "source_token_offset": int(row["source_token_offset"]),
                "query_tokens": int(row["query_tokens"]),
                "query_token_ids_sha256": str(row["query_token_ids_sha256"]),
            }
            for row in query_audit["rows"]
        ]
        == bank["rows"],
        "runtime query helper audit differs from frozen bank",
    )
    data_usage = {
        "dataset": "pg19",
        "split": "train",
        "pg19_train_only": True,
        "longbench_consumed": False,
        "validation_consumed": False,
        "test_v2_consumed": False,
        "source_id": bank["source_id"],
        "source_object": bank["source_object"],
        "book_index": args.rank,
        "window_index": bank["window_index"],
        "document_start_token": bank["document_start_token"],
        "document_end_token_exclusive": bank[
            "document_end_token_exclusive"
        ],
        "document_length": FORMAL_DOCUMENT_TOKENS,
        "document_token_ids_sha256": document_sha,
        "document_input_receipt": {
            "capture_point": "immediately-before-persistent-document-prefill",
            "dtype": "torch.int64",
            "shape": [1, FORMAL_DOCUMENT_TOKENS],
            "sha256": document_sha,
            "rebuilt_from_raw_bound_rr2_manifest": True,
        },
        "query_bank_input_receipt": {
            "capture_point": "immediately-before-formal-factorial-cells",
            "dtype": "torch.int64",
            "shape_per_query": [1, FORMAL_QUERY_TOKENS],
            "count": max(FORMAL_RESIDENT_COUNTS),
            "rows": query_rows,
            "rebuilt_from_raw_bound_rr2_manifest": True,
        },
    }
    run_id_raw = args.run_id_receipt.read_bytes()
    run_id_receipt = _validate_run_id_receipt(
        run_id_raw,
        expected_sha256=args.expected_run_id_receipt_sha256,
        run_id=args.run_id,
        static_artifact_sha256=static_sha,
        protocol_manifest_sha256=identity["protocol_manifest_sha256"],
    )
    gpu_assignment, gpu_assignment_raw, gpu_assignment_raw_sha = (
        _load_canonical_lf_receipt(
            args.gpu_assignment_receipt,
            args.expected_gpu_assignment_receipt_raw_sha256,
            label="GPU-assignment receipt",
        )
    )
    gpu_assignment = _validate_gpu_assignment_receipt(
        gpu_assignment,
        raw_sha256=gpu_assignment_raw_sha,
        raw_bytes=gpu_assignment_raw,
    )
    gpu_assignment_row = dict(gpu_assignment["rows"][args.rank])
    _require(
        gpu_assignment_row["uuid"] == args.expected_gpu_uuid,
        "rank expected GPU UUID differs from shared assignment receipt",
    )
    return FormalInputBundle(
        static_payload=dict(static_payload),
        static_replay=static_replay,
        static_sha256=static_sha,
        rr2_manifest=dict(rr2_manifest),
        document_cpu=document_cpu,
        queries_cpu=queries_cpu,
        data_usage=data_usage,
        input_rebuild_receipt={
            "schema_version": "qcomem-forkaudit-live-input-rebuild-v1",
            "rr2_input_manifest_raw_sha256": sha256_bytes(rr2_raw),
            "source_rebuilt_manifest_byte_identical": True,
            "pg19_windows_sha256": windows_sha,
            "rank": args.rank,
            "document_token_ids_sha256": document_sha,
            "query_token_ids_sha256": [row["sha256"] for row in query_rows],
            "model_manifest_rows": model_manifest_rows,
            "model_artifact_audit": artifact_audit,
            "model_weight_audit": weight_audit,
            "code_ledger_entry_count": len(code_rows),
            "protocol_manifest_raw_sha256": sha256_bytes(protocol_raw),
            "private_model_view_manifest_raw_sha256": private_view_raw_sha,
            "model_load_authority_raw_sha256": authority_raw_sha,
            "cuda_initialized_during_rebuild": torch.cuda.is_initialized(),
        },
        run_id_receipt=run_id_receipt,
        gpu_assignment_receipt=gpu_assignment,
        gpu_assignment_receipt_raw_sha256=gpu_assignment_raw_sha,
        gpu_assignment_row=gpu_assignment_row,
        private_model_view_manifest=private_model_view_manifest,
        private_model_view_manifest_raw_sha256=private_view_raw_sha,
        model_load_authority=model_load_authority,
        model_load_authority_raw_sha256=authority_raw_sha,
        model_integrity_before_load={
            "model_manifest_sha256": model_manifest_sha,
            "model_artifact_audit": artifact_audit,
            "model_weight_audit": weight_audit,
            "private_model_view_manifest_raw_sha256": private_view_raw_sha,
            "model_load_authority_raw_sha256": authority_raw_sha,
        },
        model_artifact_rows=tuple(dict(row) for row in artifact_rows),
        model_weight_rows=tuple(dict(row) for row in weight_rows),
    )


def _audit_formal_local_gpu(
    expected_gpu_uuid: Any, expected_assignment: Mapping[str, Any]
) -> dict[str, Any]:
    """Require one launcher-isolated H20 exposed as process-local cuda:0."""

    import torch

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    expected_uuid = expected_gpu_uuid
    _require(
        isinstance(expected_uuid, str)
        and re.fullmatch(r"GPU-[0-9a-fA-F-]+", expected_uuid) is not None,
        "formal shard expected GPU UUID missing",
    )
    _require(
        visible == expected_uuid,
        "formal shard requires CUDA_VISIBLE_DEVICES to equal the assigned GPU UUID",
    )
    _require(
        isinstance(expected_assignment, Mapping)
        and expected_assignment.get("uuid") == expected_uuid,
        "formal GPU assignment row/UUID drift",
    )
    _require(torch.cuda.is_available(), "CUDA is unavailable for formal shard")
    _require(torch.cuda.device_count() == 1, "formal rank must see exactly one GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    capability = tuple(int(item) for item in torch.cuda.get_device_capability(0))
    _require(capability == (9, 0), "formal GPU compute capability must be 9.0")
    _require("H20" in str(properties.name), "formal GPU is not NVIDIA H20")
    _require(torch.cuda.is_bf16_supported(), "formal GPU lacks BF16 support")
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                f"--id={visible}",
                "--query-gpu=uuid,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise ReviewAuditError("cannot query the isolated formal GPU identity") from exc
    rows = [row.strip() for row in process.stdout.splitlines() if row.strip()]
    _require(len(rows) == 1, "isolated GPU UUID query did not return one row")
    columns = [item.strip() for item in rows[0].split(",")]
    _require(len(columns) == 3, "isolated GPU identity row schema drift")
    observed_uuid, observed_name, memory_mib_raw = columns
    try:
        memory_mib = int(memory_mib_raw)
    except ValueError as exc:
        raise ReviewAuditError("isolated GPU memory value is not integral MiB") from exc
    _require(
        observed_uuid == expected_uuid
        and observed_name == expected_assignment["name"]
        and memory_mib == expected_assignment["total_memory_mib"]
        and list(capability) == expected_assignment["compute_capability"]
        and expected_assignment["bf16_supported"] is True,
        "isolated GPU UUID/name/memory differs from launcher binding",
    )
    return {
        "schema_version": "qcomem-forkaudit-local-gpu-audit-v1",
        "cuda_visible_devices": visible,
        "physical_visible_index": expected_assignment["visible_index"],
        "process_local_device": "cuda:0",
        "visible_device_count": 1,
        "uuid": observed_uuid,
        "name": observed_name,
        "total_memory_mib": memory_mib,
        "compute_capability": [9, 0],
        "bf16_supported": True,
        "torch_cuda_version": str(torch.version.cuda),
    }


@dataclass(frozen=True)
class FormalModelRuntime:
    model: Any
    backbone: Any
    plan: Any
    kernel: Any
    document: Any
    queries: tuple[Any, ...]
    hardware_audit: dict[str, Any]
    model_runtime_audit: dict[str, Any]


LIVE_INPUT_CAPTURE_POINTS = (
    "post-warmup-immediately-before-first-formal-prefill",
    "after-all-formal-factorial-cells",
    "after-fault-campaign-before-shard-commit",
)


def _formal_functional_stack_metadata(plan: Any) -> dict[str, Any]:
    metadata = strict_json_loads(
        canonical_json_bytes(plan.metadata()),
        label="runtime functional stack metadata",
    )
    _require(
        metadata.get("model_type") == FORMAL_MODEL_TYPE,
        "runtime functional-stack model type drift",
    )
    # The shared routing-plan helper predates the vLLM formal path and labels
    # its own reference attention implementation. The plan is used here only
    # to bind layer routing; the separately resolved callable is what executes.
    metadata["kernel_mode"] = FORMAL_KERNEL_MODE
    return metadata


def _capture_live_input_lifetime_receipt(
    *,
    runtime: FormalModelRuntime,
    expected_query_bank: Mapping[str, Any],
    frozen_baseline: Mapping[str, int],
    capture_point: str,
) -> dict[str, Any]:
    """Re-hash the exact live CUDA token tensors without changing allocator state."""

    import torch

    _require(capture_point in LIVE_INPUT_CAPTURE_POINTS, "live input capture point drift")
    _require(
        isinstance(runtime.document, torch.Tensor)
        and runtime.document.device.type == "cuda"
        and runtime.document.device.index == 0
        and len(runtime.queries) == max(FORMAL_RESIDENT_COUNTS)
        and all(
            isinstance(query, torch.Tensor)
            and query.device.type == "cuda"
            and query.device.index == 0
            for query in runtime.queries
        ),
        "live input lifetime receipt requires process-local cuda:0 tensors",
    )
    before = _gpu_allocator_snapshot()
    document_sha = _token_id_sha256(
        runtime.document, expected_shape=(1, FORMAL_DOCUMENT_TOKENS)
    )
    query_rows = [
        {
            "request_index": request_index,
            "sha256": _token_id_sha256(
                query, expected_shape=(1, FORMAL_QUERY_TOKENS)
            ),
        }
        for request_index, query in enumerate(runtime.queries)
    ]
    after = _gpu_allocator_snapshot()
    _require(
        all(
            before[field] == after[field] == frozen_baseline[field]
            for field in ("current_allocated_bytes", "current_reserved_bytes")
        ),
        "live input hashing changed or escaped the frozen allocator baseline",
    )
    _require(
        document_sha == expected_query_bank["document_token_ids_sha256"]
        and [row["sha256"] for row in query_rows]
        == [row["query_token_ids_sha256"] for row in expected_query_bank["rows"]],
        "live CUDA document/query tokens changed after source rebuild",
    )
    return {
        "schema_version": LIVE_INPUT_LIFETIME_SCHEMA_VERSION,
        "capture_point": capture_point,
        "device": "cuda:0",
        "dtype": "torch.int64",
        "document_shape": [1, FORMAL_DOCUMENT_TOKENS],
        "document_token_ids_sha256": document_sha,
        "query_shape": [1, FORMAL_QUERY_TOKENS],
        "query_count": max(FORMAL_RESIDENT_COUNTS),
        "query_rows": query_rows,
        "allocator_before": before,
        "allocator_after": after,
        "current_allocator_exactly_equal_to_frozen_baseline": True,
    }


def _validate_live_input_lifetime_receipts(
    value: Any,
    *,
    expected_query_bank: Mapping[str, Any],
    frozen_baseline: Mapping[str, int],
) -> list[dict[str, Any]]:
    _require(
        isinstance(value, list) and len(value) == len(LIVE_INPUT_CAPTURE_POINTS),
        "live input lifetime receipt count drift",
    )
    fields = {
        "schema_version",
        "capture_point",
        "device",
        "dtype",
        "document_shape",
        "document_token_ids_sha256",
        "query_shape",
        "query_count",
        "query_rows",
        "allocator_before",
        "allocator_after",
        "current_allocator_exactly_equal_to_frozen_baseline",
    }
    expected_rows = [
        {
            "request_index": row["request_index"],
            "sha256": row["query_token_ids_sha256"],
        }
        for row in expected_query_bank["rows"]
    ]
    result: list[dict[str, Any]] = []
    for index, row in enumerate(value):
        _require(
            isinstance(row, dict)
            and set(row) == fields
            and row["schema_version"] == LIVE_INPUT_LIFETIME_SCHEMA_VERSION
            and row["capture_point"] == LIVE_INPUT_CAPTURE_POINTS[index]
            and row["device"] == "cuda:0"
            and row["dtype"] == "torch.int64"
            and row["document_shape"] == [1, FORMAL_DOCUMENT_TOKENS]
            and row["document_token_ids_sha256"]
            == expected_query_bank["document_token_ids_sha256"]
            and row["query_shape"] == [1, FORMAL_QUERY_TOKENS]
            and _is_int(row["query_count"])
            and row["query_count"] == max(FORMAL_RESIDENT_COUNTS)
            and row["query_rows"] == expected_rows
            and row["current_allocator_exactly_equal_to_frozen_baseline"] is True,
            "live input lifetime receipt drift",
        )
        before = _validate_allocator_snapshot(
            row["allocator_before"], label="live input allocator before"
        )
        after = _validate_allocator_snapshot(
            row["allocator_after"], label="live input allocator after"
        )
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(
                before[field] == after[field] == frozen_baseline[field],
                "live input receipt left the frozen allocator baseline",
            )
        result.append(dict(row))
    return result


def _load_formal_model_runtime(
    args: argparse.Namespace, inputs: FormalInputBundle
) -> FormalModelRuntime:
    """Load the exact local text model and move frozen inputs to cuda:0."""

    import torch
    from qcomem_forkaudit_model_load_lease import (
        ModelLoadLeaseError,
        capture_rank_stat_envelope,
    )
    from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
    from qcomem_vllm_paged_kernel import (
        _resolve_vllm_unified_attention,
        audit_frozen_kernel_environment,
    )
    from run_qcomem_qwen35_vllm_paged_multifork_resident import (
        _audit_model_config_geometry,
        _resolve_backbone,
    )
    from transformers import AutoModelForImageTextToText

    hardware = _audit_formal_local_gpu(
        args.expected_gpu_uuid, inputs.gpu_assignment_row
    )
    try:
        pre_load_stat = capture_rank_stat_envelope(
            model_view=args.model_dir,
            authority=inputs.model_load_authority,
            authority_raw_sha256=inputs.model_load_authority_raw_sha256,
            rank=args.rank,
            capture_point="immediately-before-from-pretrained",
        )
        model = AutoModelForImageTextToText.from_pretrained(
            str(args.model_dir),
            revision=FORMAL_MODEL_REVISION,
            dtype=torch.bfloat16,
            local_files_only=True,
            trust_remote_code=False,
        )
        post_load_stat = capture_rank_stat_envelope(
            model_view=args.model_dir,
            authority=inputs.model_load_authority,
            authority_raw_sha256=inputs.model_load_authority_raw_sha256,
            rank=args.rank,
            capture_point="immediately-after-from-pretrained",
        )
    except ModelLoadLeaseError as exc:
        raise ReviewAuditError("rank ModelLoadLease stat envelope rejected") from exc
    outer = getattr(model, "model", None)
    visual_removed = bool(outer is not None and hasattr(outer, "visual"))
    if visual_removed:
        outer.visual = None
    model.eval()

    # Bracket Transformers' load with rank metadata envelopes.  Small
    # artifacts are rehashed here; the large weight payloads remain under the
    # launcher-owned Linux read-lease authority and are terminally rehashed by
    # that same fixed-FD guard.
    artifact_after = _verify_model_ledger(
        inputs.model_artifact_rows,
        model_dir=args.model_dir,
        label="model artifact ledger after load",
    )
    weight_after = _verify_weight_ledger_structure(
        inputs.model_weight_rows, model_dir=args.model_dir
    )
    manifest_after, manifest_rows = _model_manifest_digest(args.model_dir)
    _require(
        {
            "model_manifest_sha256": manifest_after,
            "model_artifact_audit": artifact_after,
            "model_weight_audit": weight_after,
            "private_model_view_manifest_raw_sha256": (
                inputs.private_model_view_manifest_raw_sha256
            ),
            "model_load_authority_raw_sha256": (
                inputs.model_load_authority_raw_sha256
            ),
        }
        == inputs.model_integrity_before_load,
        "model inputs changed across local model loading",
    )
    geometry = _audit_model_config_geometry(args.model_dir)
    plan = audit_qwen35_functional_stack_plan(model)
    _require(
        tuple(plan.full_attention_layer_indices) == FORMAL_FULL_LAYERS
        and tuple(plan.linear_layer_indices) == FORMAL_LINEAR_LAYERS,
        "runtime functional-stack layer plan drift",
    )
    kernel_environment = audit_frozen_kernel_environment()
    _require(
        kernel_environment.get("matches_frozen_environment") is True,
        "runtime vLLM/Triton environment drift",
    )
    normalized_kernel_environment = {
        "expected_versions": dict(kernel_environment["expected_versions"]),
        "observed_versions": dict(kernel_environment["observed_versions"]),
        "matches_frozen_environment": kernel_environment[
            "matches_frozen_environment"
        ],
        "mismatches": dict(kernel_environment["mismatches"]),
        "kernel_entrypoint": kernel_environment["kernel_entrypoint"],
        "kernel_mode": kernel_environment["kernel_mode"],
    }
    kernel = _resolve_vllm_unified_attention()
    try:
        kernel_signature = str(inspect.signature(kernel))
    except (TypeError, ValueError) as exc:
        raise ReviewAuditError("cannot inspect formal unified-attention callable") from exc
    descriptor = (
        str(getattr(kernel, "__module__", "")),
        str(getattr(kernel, "__qualname__", "")),
        kernel_signature,
    )
    _require(
        descriptor == FORMAL_KERNEL_DESCRIPTOR,
        "runtime unified-attention callable differs from frozen descriptor",
    )

    model = model.to(device="cuda:0", dtype=torch.bfloat16)
    backbone = _resolve_backbone(model)
    _require(hasattr(model, "lm_head"), "formal model language-model head missing")
    document = inputs.document_cpu.to(device="cuda:0", dtype=torch.int64)
    queries = tuple(
        query.to(device="cuda:0", dtype=torch.int64)
        for query in inputs.queries_cpu
    )
    _require(
        _token_id_sha256(document, expected_shape=(1, FORMAL_DOCUMENT_TOKENS))
        == inputs.data_usage["document_token_ids_sha256"],
        "CUDA document tensor differs from source-rebuilt bytes",
    )
    _require(
        [
            _token_id_sha256(query, expected_shape=(1, FORMAL_QUERY_TOKENS))
            for query in queries
        ]
        == [
            row["sha256"]
            for row in inputs.data_usage["query_bank_input_receipt"]["rows"]
        ],
        "CUDA query tensors differ from source-rebuilt bytes",
    )
    functional_stack_metadata = _formal_functional_stack_metadata(plan)
    return FormalModelRuntime(
        model=model,
        backbone=backbone,
        plan=plan,
        kernel=kernel,
        document=document,
        queries=queries,
        hardware_audit=hardware,
        model_runtime_audit={
            "schema_version": "qcomem-forkaudit-model-runtime-audit-v1",
            "model_id": FORMAL_MODEL_ID,
            "model_revision": FORMAL_MODEL_REVISION,
            "local_files_only": True,
            "trust_remote_code": False,
            "dtype": "torch.bfloat16",
            "device": "cuda:0",
            "visual_branch_removed_for_direct_text_backbone": visual_removed,
            "model_manifest_sha256": manifest_after,
            "model_manifest_rows": manifest_rows,
            "model_artifact_audit_before_after_equal": True,
            "model_weight_index_and_size_before_after_equal": True,
            "model_weight_full_hash_per_rank": False,
            "private_model_view_manifest_raw_sha256": (
                inputs.private_model_view_manifest_raw_sha256
            ),
            "model_load_authority_raw_sha256": (
                inputs.model_load_authority_raw_sha256
            ),
            "model_load_rank_stat_envelopes": [
                pre_load_stat,
                post_load_stat,
            ],
            "geometry": geometry,
            "functional_stack_plan": functional_stack_metadata,
            "kernel_environment": normalized_kernel_environment,
            "kernel_descriptor": {
                "module": descriptor[0],
                "qualname": descriptor[1],
                "signature": descriptor[2],
            },
        },
    )


def _physical_document_payload_digests(
    persistent: Any,
    layer_indices: Sequence[int],
) -> dict[str, str]:
    """Hash only physical document K/V bytes, including tail padding.

    ``source_document_physical_digests`` also binds arena capacity and is used
    for within-cell mutation checks.  This second digest intentionally omits
    ``max_forks`` and private-pool capacity so the same packed PG19 document
    can be compared across N={1,8,32}.
    """

    import torch

    result: dict[str, str] = {}
    for layer_index in layer_indices:
        arena = persistent.layers[layer_index].arena
        table = arena.document_block_table.detach().reshape(-1).cpu().tolist()
        _require(
            len(table)
            == arena.batch_size * arena.document_blocks_per_sequence,
            "physical document table geometry drift",
        )
        digest = hashlib.sha256()
        digest.update(
            canonical_json_bytes(
                {
                    "schema_version": "qcomem-document-physical-payload-v1",
                    "layer_index": int(layer_index),
                    "document_length": int(arena.document_length),
                    "page_size": int(arena.page_size),
                    "batch_size": int(arena.batch_size),
                    "document_blocks_per_sequence": int(
                        arena.document_blocks_per_sequence
                    ),
                    "num_key_value_heads": int(arena.num_key_value_heads),
                    "head_dim": int(arena.head_dim),
                    "dtype": str(arena.key_cache.dtype),
                    "component_order": "batch-major/block-major/key-then-value",
                    "tail_padding_included": True,
                }
            )
        )
        digest.update(b"\0")
        for physical in table:
            for component in (arena.key_cache, arena.value_cache):
                block = component[int(physical)].detach().contiguous().cpu()
                digest.update(block.view(torch.uint8).numpy().tobytes())
        result[str(int(layer_index))] = digest.hexdigest()
    return result


def _persistent_document_kv(persistent: Any, layer_index: int) -> tuple[Any, Any]:
    """Materialize the oracle document from its immutable document table.

    This path never reads a request's candidate ``active_block_table``.
    """

    import torch

    arena = persistent.layers[layer_index].arena
    table = arena.document_block_table.detach().cpu()
    batches_key = []
    batches_value = []
    for batch_index in range(int(arena.batch_size)):
        block_ids = [int(value) for value in table[batch_index].reshape(-1)]
        key = torch.cat(
            [arena.key_cache[physical].detach() for physical in block_ids], dim=0
        )[: int(arena.document_length)]
        value = torch.cat(
            [arena.value_cache[physical].detach() for physical in block_ids], dim=0
        )[: int(arena.document_length)]
        batches_key.append(key.permute(1, 0, 2).contiguous().cpu())
        batches_value.append(value.permute(1, 0, 2).contiguous().cpu())
    return torch.stack(batches_key, dim=0), torch.stack(batches_value, dim=0)


def _active_block_table_digest(sequence: Any) -> str:
    return _tensor_digest(sequence.active_block_table)


@dataclass
class OracleAppendShadowCollector:
    """Synchronous pre-write append capture for one preregistered request."""

    sample_id: str
    events: list[dict[str, Any]]

    def __call__(self, event: Mapping[str, Any]) -> str:
        import torch

        event_index = event.get("append_event_index")
        _require(
            _is_int(event_index) and event_index == len(self.events),
            "oracle append observer event order drift",
        )
        key = event.get("key_states")
        value = event.get("value_states")
        _require(
            isinstance(key, torch.Tensor)
            and isinstance(value, torch.Tensor)
            and key.shape == value.shape,
            "oracle append observer did not receive matching dense K/V",
        )
        # The synchronous clone is intentionally confined to a witness cell;
        # memory endpoints never install this observer.
        key_cpu = key.detach().contiguous().cpu().clone()
        value_cpu = value.detach().contiguous().cpu().clone()
        capture_id = (
            f"{self.sample_id}-append-{event_index}-"
            f"{secrets.token_hex(8)}"
        )
        self.events.append(
            {
                "capture_id": capture_id,
                "append_event_index": int(event_index),
                "appended_tokens_before": int(event["appended_tokens_before"]),
                "appended_tokens_after": int(event["appended_tokens_after"]),
                "sequence_length_before": int(event["sequence_length_before"]),
                "sequence_length_after": int(event["sequence_length_after"]),
                "source_device": str(event.get("source_device")),
                "source_dtype": str(event.get("source_dtype")),
                "source_shape": list(event.get("source_shape", [])),
                "key": key_cpu,
                "value": value_cpu,
                "key_sha256": _tensor_digest(key_cpu),
                "value_sha256": _tensor_digest(value_cpu),
            }
        )
        return capture_id

    def concatenated_through(self, round_index: int) -> tuple[Any, Any, dict[str, Any]]:
        import torch

        _require(
            0 <= round_index < len(self.events),
            "oracle append-shadow round was not captured",
        )
        selected = self.events[: round_index + 1]
        capture_ids = [row["capture_id"] for row in selected]
        _require(len(set(capture_ids)) == len(capture_ids), "oracle append capture ID reused")
        for event_index, row in enumerate(selected):
            expected_before = (
                0
                if event_index == 0
                else FORMAL_QUERY_TOKENS + event_index - 1
            )
            expected_after = FORMAL_QUERY_TOKENS + event_index
            _require(
                row["append_event_index"] == event_index
                and row["appended_tokens_before"] == expected_before
                and row["appended_tokens_after"] == expected_after,
                "oracle append-shadow event schedule drift",
            )
        key = torch.cat([row["key"] for row in selected], dim=2)
        value = torch.cat([row["value"] for row in selected], dim=2)
        return key, value, dict(selected[-1])


@dataclass
class AppendReceiptCollector:
    """Return unique append IDs without retaining dense K/V CPU clones."""

    prefix: str
    event_count: int = 0

    def __call__(self, event: Mapping[str, Any]) -> str:
        event_index = event.get("append_event_index")
        _require(
            _is_int(event_index) and event_index == self.event_count,
            "strict witness append event order drift",
        )
        capture_id = f"{self.prefix}-append-{event_index}-{secrets.token_hex(8)}"
        self.event_count += 1
        return capture_id


@dataclass(frozen=True)
class ChainedCallObserver:
    """Run pointer-free witness observers without closing over live cache state."""

    observers: tuple[Callable[[Mapping[str, Any]], None], ...]

    def __call__(self, event: Mapping[str, Any]) -> None:
        for observer in self.observers:
            observer(event)


@dataclass
class OracleLiveCallCapture:
    selection: Mapping[str, Any]
    append_collectors: Mapping[int, OracleAppendShadowCollector]
    row: dict[str, Any] | None = None

    def __call__(self, event: Mapping[str, Any]) -> None:
        if (
            int(event["request_index"]) != int(self.selection["request_index"])
            or int(event["layer_idx"]) != int(self.selection["layer_index"])
        ):
            return
        # One selected full layer is called once per generation round.
        collector = self.append_collectors[int(event["layer_idx"])]
        observed_round = len(collector.events) - 1
        if observed_round != int(self.selection["round_index"]):
            return
        _require(self.row is None, "oracle live call selected more than once")
        capture_id = event.get("append_capture_id")
        _require(
            isinstance(capture_id, str)
            and capture_id
            and collector.events[-1]["capture_id"] == capture_id,
            "oracle kernel call did not consume its pre-write append shadow",
        )
        query = event.get("query_cpu")
        candidate = event.get("candidate_output_cpu")
        positions = event.get("position_ids_cpu")
        import torch

        _require(
            event.get("observer_schema") == "qcomem-forkaudit-call-observer-v2"
            and event.get("attention_mask_is_none") is True
            and isinstance(query, torch.Tensor)
            and isinstance(candidate, torch.Tensor)
            and isinstance(positions, torch.Tensor),
            "oracle v2 live Q/candidate/position capture missing",
        )
        append_audit = event.get("append_audit")
        _require(
            isinstance(append_audit, dict)
            and append_audit.get("capture_id") == capture_id
            and append_audit.get("append_event_index") == observed_round,
            "oracle v2 call/append audit binding drift",
        )
        self.row = {
            "query": query.detach().contiguous().clone(),
            "candidate": candidate.detach().contiguous().clone(),
            "query_positions": positions.detach().contiguous().reshape(-1).clone(),
            "kernel_audit": dict(event["kernel_audit"]),
            "effective_scaling": float(event["effective_scaling"]),
            "append_capture_id": capture_id,
            "append_audit": dict(append_audit),
            "active_block_table_sha256": None,
            "active_block_table": None,
        }


@dataclass
class PositionEvidenceCollector:
    """Capture exact strict-witness post-RoPE positions for blind replay."""

    request_index: int
    rows: list[dict[str, Any]]

    def __call__(self, event: Mapping[str, Any]) -> None:
        import torch

        _require(
            event.get("request_index") == self.request_index,
            "position observer request binding drift",
        )
        layer_index = event.get("layer_idx")
        _require(layer_index in FORMAL_FULL_LAYERS, "position observer layer drift")
        prior = [row for row in self.rows if row["layer_idx"] == layer_index]
        round_index = len(prior)
        _require(
            round_index < FORMAL_GENERATION_STEPS,
            "position observer call budget exceeded",
        )
        positions = event.get("position_ids_cpu")
        query = event.get("query_cpu")
        _require(
            isinstance(positions, torch.Tensor)
            and positions.dtype == torch.int64
            and isinstance(query, torch.Tensor),
            "position observer exact tensor evidence missing",
        )
        query_tokens = FORMAL_QUERY_TOKENS if round_index == 0 else 1
        end = FORMAL_DOCUMENT_TOKENS + FORMAL_QUERY_TOKENS + round_index
        expected = list(range(end - query_tokens, end))
        observed = [int(item) for item in positions.reshape(-1).tolist()]
        _require(
            list(positions.shape) == [1, query_tokens]
            and int(query.shape[-2]) == query_tokens
            and observed == expected,
            "position observer values differ from canonical tail",
        )
        self.rows.append(
            {
                "request_index": self.request_index,
                "layer_idx": int(layer_index),
                "round_index": round_index,
                "position_ids_values": observed,
                "position_ids_sha256": _tensor_digest(positions),
            }
        )

    def attach(self, ledger: Mapping[str, Any]) -> dict[str, Any]:
        _require(
            len(self.rows)
            == FORMAL_GENERATION_STEPS * len(FORMAL_FULL_LAYERS),
            "strict position evidence is incomplete",
        )
        result = dict(ledger)
        calls = [dict(row) for row in ledger["calls"]]
        _require(len(calls) == len(self.rows), "position/call cardinality drift")
        for call, evidence in zip(calls, self.rows):
            _require(
                call["request_index"] == evidence["request_index"]
                and call["layer_idx"] == evidence["layer_idx"],
                "position evidence/call order drift",
            )
            call["position_ids_values"] = list(evidence["position_ids_values"])
            call["position_ids_sha256"] = evidence["position_ids_sha256"]
        result["calls"] = calls
        return result


def _gpu_round_robin_generate(
    model: Any,
    backbone: Any,
    group: Any,
    queries: Sequence[Any],
    backends: Sequence[str],
    *,
    after_step: Callable[[int, int], None] | None = None,
    measure_allocator: bool,
) -> tuple[
    list[dict[str, Any]],
    dict[str, int] | None,
    dict[str, Any] | None,
]:
    """Execute the exact eight-step round-major schedule.

    GPU allocator peaks are sampled before full-logit CPU hashing.  Peak
    counters are reset after diagnostics and their per-step maxima are folded
    into the returned generation endpoint.
    """

    import torch
    from run_qcomem_qwen35_vllm_paged_multifork_resident import _last_logits

    resident_count = group.resident_count
    _require(
        len(group.requests) == len(queries) == len(backends) == resident_count,
        "GPU generation resident cardinality drift",
    )
    currents = list(queries)
    trajectories = [
        {
            "request_index": request_index,
            "query_token_ids_sha256": _token_id_sha256(
                queries[request_index], expected_shape=(1, FORMAL_QUERY_TOKENS)
            ),
            "generated_token_ids": [],
            "full_vocab_step_logit_sha256": [],
        }
        for request_index in range(resident_count)
    ]
    observed_allocated: list[int] = []
    observed_reserved: list[int] = []
    diagnostic_rows: list[dict[str, Any]] = []
    original_backend = backbone.config._attn_implementation
    try:
        for round_index in range(FORMAL_GENERATION_STEPS):
            for request_index in range(resident_count):
                backbone.config._attn_implementation = backends[request_index]
                output = backbone(
                    input_ids=currents[request_index],
                    past_key_values=group.requests[request_index],
                    use_cache=True,
                )
                logits = _last_logits(model, output)
                token = int(logits.argmax(dim=-1).item())
                torch.cuda.synchronize()
                if measure_allocator:
                    before_exactness = _gpu_allocator_snapshot()
                    observed_allocated.append(
                        before_exactness["peak_allocated_bytes"]
                    )
                    observed_reserved.append(
                        before_exactness["peak_reserved_bytes"]
                    )
                if after_step is not None:
                    after_step(round_index, request_index)
                # A single detached CPU clone is the only exactness transfer.
                # Finite checking and hashing must not run CUDA kernels after
                # the allocator endpoint has been sampled.
                logits_cpu = logits.detach().cpu()
                _require(
                    bool(torch.isfinite(logits_cpu).all()),
                    "non-finite full logits",
                )
                logit_sha256 = _tensor_digest(logits_cpu)
                if measure_allocator:
                    after_exactness = _gpu_allocator_snapshot()
                    _require(
                        after_exactness["current_allocated_bytes"]
                        == before_exactness["current_allocated_bytes"]
                        and after_exactness["current_reserved_bytes"]
                        == before_exactness["current_reserved_bytes"],
                        "generation CPU diagnostics changed CUDA allocator state",
                    )
                    diagnostic_rows.append(
                        {
                            "round_index": round_index,
                            "request_index": request_index,
                            "before_current_allocated_bytes": before_exactness[
                                "current_allocated_bytes"
                            ],
                            "before_current_reserved_bytes": before_exactness[
                                "current_reserved_bytes"
                            ],
                            "after_current_allocated_bytes": after_exactness[
                                "current_allocated_bytes"
                            ],
                            "after_current_reserved_bytes": after_exactness[
                                "current_reserved_bytes"
                            ],
                            "cpu_logits_dtype": str(logits_cpu.dtype),
                            "cpu_logits_shape": list(logits_cpu.shape),
                            "cpu_logits_sha256": logit_sha256,
                            "finite_check_on_cpu": True,
                            "allocator_state_exactly_unchanged": True,
                        }
                    )
                trajectories[request_index]["generated_token_ids"].append(token)
                trajectories[request_index]["full_vocab_step_logit_sha256"].append(
                    logit_sha256
                )
                currents[request_index] = torch.tensor(
                    [[token]], dtype=torch.long, device=queries[request_index].device
                )
                del output, logits, logits_cpu
                torch.cuda.synchronize()
                if measure_allocator:
                    torch.cuda.reset_peak_memory_stats()
    finally:
        backbone.config._attn_implementation = original_backend
    endpoint = None
    diagnostic_receipt = None
    if measure_allocator:
        _require(bool(observed_allocated) and bool(observed_reserved), "generation peak receipt empty")
        current = _gpu_allocator_snapshot()
        endpoint = {
            "current_allocated_bytes": current["current_allocated_bytes"],
            "current_reserved_bytes": current["current_reserved_bytes"],
            "peak_allocated_bytes": max(
                max(observed_allocated), current["current_allocated_bytes"]
            ),
            "peak_reserved_bytes": max(
                max(observed_reserved), current["current_reserved_bytes"]
            ),
        }
        diagnostic_receipt = {
            "schema_version": "qcomem-generation-cpu-diagnostics-v1",
            "resident_count": resident_count,
            "rounds": FORMAL_GENERATION_STEPS,
            "schedule": "round-major-request-minor",
            "single_cpu_clone_per_step": True,
            "gpu_finite_or_hash_kernels_after_endpoint_sample": False,
            "rows": diagnostic_rows,
            "rows_sha256": sha256_json(diagnostic_rows),
        }
    return trajectories, endpoint, diagnostic_receipt


def _semantic_rows_from_live(
    group: Any,
    plan: Any,
    trajectories: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    from run_qcomem_qwen35_vllm_paged_multifork_resident import (
        _request_logical_kv_digests,
        _resident_linear_states,
    )

    logical = _request_logical_kv_digests(
        group, plan.full_attention_layer_indices
    )
    gdn = _resident_linear_states(group, plan.linear_layer_indices)
    _require(
        len(trajectories) == len(logical) == len(gdn) == group.resident_count,
        "semantic diagnostic cardinality drift",
    )
    return [
        {
            "request_index": request_index,
            "query_token_ids_sha256": trajectory["query_token_ids_sha256"],
            "generated_token_ids": list(trajectory["generated_token_ids"]),
            "full_vocab_step_logit_sha256": list(
                trajectory["full_vocab_step_logit_sha256"]
            ),
            "logical_kv_sha256": dict(logical[request_index]["layer_sha256"]),
            "final_gdn_sha256": gdn[request_index]["sha256"],
        }
        for request_index, trajectory in enumerate(trajectories)
    ]


def _make_ledgers_and_backends(
    group: Any,
    plan: Any,
    *,
    kernel: Any,
    strict_position_values: bool,
    call_observers: Mapping[int, Callable[[Mapping[str, Any]], None]] | None = None,
) -> tuple[list[Any], list[str]]:
    ledgers = []
    backends = []
    try:
        for request_index, request in enumerate(group.requests):
            observer = (
                None
                if call_observers is None
                else call_observers.get(request_index)
            )
            ledger = MultiForkHitLedger(
                plan,
                request,
                request_index=request_index,
                resident_count=group.resident_count,
                request_policy=group.policy,
                expected_calls_per_layer=FORMAL_GENERATION_STEPS,
                initial_query_tokens=FORMAL_QUERY_TOKENS,
                kernel=kernel,
                strict_position_values=strict_position_values,
                call_observer=observer,
            )
            ledgers.append(ledger)
            backends.append(register_multifork_backend(ledger))
    except BaseException as primary:
        try:
            _unregister_backends(backends)
        except BaseException as cleanup:
            _add_exception_note(
                primary, f"secondary partial-backend cleanup failure: {cleanup}"
            )
            raise primary from cleanup
        raise
    return ledgers, backends


def _unregister_backends(backends: Sequence[str]) -> None:
    from run_qcomem_qwen35_vllm_paged_multifork_resident import (
        _unregister_backend,
    )

    errors: list[tuple[str, BaseException]] = []
    for name in backends:
        try:
            _unregister_backend(name)
        except BaseException as exc:  # cleanup must attempt every registration
            errors.append((name, exc))
    if errors:
        names = ", ".join(name for name, _exc in errors)
        raise ReviewAuditError(
            f"backend registry cleanup failed for {len(errors)} entries: {names}"
        ) from errors[0][1]


@contextmanager
def _registered_backend_scope(backends: Sequence[str]):
    """Unregister every backend without replacing a primary execution error."""

    try:
        yield
    except BaseException as primary:
        try:
            _unregister_backends(backends)
        except BaseException as cleanup:
            _add_exception_note(
                primary, f"secondary backend cleanup failure: {cleanup}"
            )
            raise primary from cleanup
        raise
    else:
        _unregister_backends(backends)


def _pointer_free_kernel_ledger(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the process-local callable address before artifact emission."""

    result = dict(value)
    identity = dict(result["kernel_identity"])
    _require(
        set(identity) == {"callable_id", "module", "qualname", "signature"},
        "live kernel identity schema drift",
    )
    identity.pop("callable_id")
    result["kernel_identity"] = identity
    calls = []
    for row in result["calls"]:
        call = dict(row)
        call_identity = dict(call["kernel_identity"])
        _require(
            set(call_identity)
            == {"callable_id", "module", "qualname", "signature"},
            "live per-call kernel identity schema drift",
        )
        call_identity.pop("callable_id")
        call["kernel_identity"] = call_identity
        calls.append(call)
    result["calls"] = calls
    return result


def _convert_persistent(
    backbone: Any,
    plan: Any,
    document: Any,
    *,
    resident_count: int,
) -> tuple[Any, Any]:
    from qcomem_qwen35_vllm_paged_integration import (
        convert_all_qwen35_full_layers_to_vllm_q16,
    )
    from run_qcomem_qwen35_vllm_paged_multifork_resident import (
        _build_document_cache,
    )

    persistent = _build_document_cache(backbone, document)
    conversion = convert_all_qwen35_full_layers_to_vllm_q16(
        persistent,
        plan,
        page_size=FORMAL_PAGE_SIZE,
        max_append_tokens=FORMAL_QUERY_TOKENS + FORMAL_GENERATION_STEPS,
        max_request_forks=resident_count,
    )
    _require(
        int(conversion.max_request_forks) == resident_count,
        "Q16 conversion max-request-forks drift",
    )
    return persistent, conversion


def _memory_receipt(
    *,
    baseline: Mapping[str, int],
    after_setup: Mapping[str, int],
    after_setup_diagnostics: Mapping[str, int],
    after_generation: Mapping[str, int],
    generation_diagnostics: Mapping[str, Any],
    storage: Mapping[str, Any],
) -> dict[str, Any]:
    setup_peak_allocated = max(
        int(after_setup["peak_allocated_bytes"]),
        int(after_generation["peak_allocated_bytes"]),
    )
    setup_peak_reserved = max(
        int(after_setup["peak_reserved_bytes"]),
        int(after_generation["peak_reserved_bytes"]),
    )
    return {
        "schema_version": "qcomem-formal-allocator-receipt-v4",
        "baseline": dict(baseline),
        "after_setup": dict(after_setup),
        "after_setup_diagnostics": dict(after_setup_diagnostics),
        "after_generation": dict(after_generation),
        "generation_diagnostics": dict(generation_diagnostics),
        "peak_reset_before_setup": True,
        "peak_reset_before_generation": True,
        "synchronized_before_each_snapshot": True,
        "model_weights_loaded_before_baseline": True,
        "diagnostic_cpu_copies_excluded_from_peak": True,
        "diagnostic_current_allocator_state_unchanged": True,
        "setup_plus_generation_peak_allocated_delta_bytes": (
            setup_peak_allocated - int(baseline["current_allocated_bytes"])
        ),
        "setup_plus_generation_peak_reserved_delta_bytes": (
            setup_peak_reserved - int(baseline["current_reserved_bytes"])
        ),
        "generation_peak_allocated_delta_bytes": (
            int(after_generation["peak_allocated_bytes"])
            - int(after_setup["current_allocated_bytes"])
        ),
        "generation_peak_reserved_delta_bytes": (
            int(after_generation["peak_reserved_bytes"])
            - int(after_setup["current_reserved_bytes"])
        ),
        "storage_breakdown": dict(storage),
        "storage_breakdown_sha256": sha256_json(storage),
        # Generic Python object-graph storage unions cannot be independently
        # reconstructed from pointer-free raw JSON.  They remain inside the
        # helper breakdown for diagnostics only and do not authorize a result.
        "unique_storage_removed_from_authorizing_payload": True,
    }


def _authorizing_storage_breakdown(value: Any) -> dict[str, Any]:
    """Drop the helper's non-authorizing object-graph diagnostic unconditionally."""

    _require(isinstance(value, dict), "helper resident storage breakdown missing")
    return {key: item for key, item in value.items() if key != "unique_storage"}


def _run_clean_memory_cell(
    *,
    rank: int,
    arm_id: str,
    resident_count: int,
    kv_policy: str,
    gdn_base_policy: str,
    model: Any,
    backbone: Any,
    plan: Any,
    document: Any,
    queries: Sequence[Any],
    kernel: Any,
    expected_frozen_baseline: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Run one allocator-eligible cell with every audit hook disabled."""

    import torch
    from run_qcomem_qwen35_vllm_paged_multifork_resident import (
        _set_production_no_mask,
    )

    _require(torch.is_inference_mode_enabled(), "formal GPU cell requires inference mode")
    baseline = _gpu_cleanup()
    if expected_frozen_baseline is not None:
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(
                baseline[field] == expected_frozen_baseline[field],
                f"memory cell baseline drift: {field}",
            )
    torch.cuda.reset_peak_memory_stats()
    persistent, _conversion = _convert_persistent(
        backbone,
        plan,
        document,
        resident_count=resident_count,
    )
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=resident_count,
        policy=kv_policy,
        gdn_base_policy=gdn_base_policy,
    )
    _set_production_no_mask(group, plan.full_attention_layer_indices)
    # Formal memory cells must not enable either the synchronous append hook or
    # the ledger call observer.  Strict position values are proven in the
    # separate witness rebuild and would otherwise introduce host syncs here.
    for request in group.requests:
        for layer_index in plan.full_attention_layer_indices:
            sequence = request.layers[layer_index].sequence
            _require(sequence.append_observer is None, "memory cell append observer leaked in")
    after_setup = _gpu_allocator_snapshot()
    # Correctness copies begin only after the allocator endpoint is frozen.
    source_before = source_document_physical_digests(
        persistent, plan.full_attention_layer_indices
    )
    source_payload_before = _physical_document_payload_digests(
        persistent, plan.full_attention_layer_indices
    )
    validate_runtime_kv_ownership(
        persistent, group, plan, require_appended_tail_cow=False
    )
    after_setup_diagnostics = _gpu_allocator_snapshot()
    _require(
        all(
            after_setup_diagnostics[field] == after_setup[field]
            for field in ("current_allocated_bytes", "current_reserved_bytes")
        ),
        "setup diagnostics changed current allocator state before generation",
    )
    torch.cuda.reset_peak_memory_stats()
    ledgers, backends = _make_ledgers_and_backends(
        group,
        plan,
        kernel=kernel,
        strict_position_values=False,
        call_observers=None,
    )
    with _registered_backend_scope(backends):
        trajectories, after_generation, generation_diagnostics = _gpu_round_robin_generate(
            model,
            backbone,
            group,
            queries[:resident_count],
            backends,
            measure_allocator=True,
        )
    assert after_generation is not None
    assert generation_diagnostics is not None
    validate_runtime_kv_ownership(
        persistent, group, plan, require_appended_tail_cow=True
    )
    source_after = source_document_physical_digests(
        persistent, plan.full_attention_layer_indices
    )
    source_payload_after = _physical_document_payload_digests(
        persistent, plan.full_attention_layer_indices
    )
    _require(source_before == source_after, "memory cell mutated physical source blocks")
    _require(
        source_payload_before == source_payload_after,
        "memory cell mutated document payload or padding",
    )
    storage_with_diagnostics = resident_storage_breakdown(persistent, group, plan)
    storage = _authorizing_storage_breakdown(storage_with_diagnostics)
    verified_ledgers = [
        _pointer_free_kernel_ledger(ledger.verify_complete())
        for ledger in ledgers
    ]
    semantics = _semantic_rows_from_live(group, plan, trajectories)
    cell_id = f"rank-{rank}-N-{resident_count}-{arm_id}-formal-memory"
    return {
        "memory_cell": {
            "cell_role": "formal_memory",
            "rank": rank,
            "resident_count": resident_count,
            "arm_id": arm_id,
            "cell_id": cell_id,
            "request_guard_created": False,
            "witness_capture_executed": False,
            "primary_memory_endpoint_eligible": True,
            "allocator_receipt": _memory_receipt(
                baseline=baseline,
                after_setup=after_setup,
                after_setup_diagnostics=after_setup_diagnostics,
                after_generation=after_generation,
                generation_diagnostics=generation_diagnostics,
                storage=storage,
            ),
            "policy_execution_receipt": {
                "builder": "build_resident_request_group",
                "kv_policy": kv_policy,
                "gdn_base_policy": gdn_base_policy,
                "resident_count": resident_count,
                "group_audit_sha256": sha256_json(group.audit),
                "group_audit": dict(group.audit),
                "all_requests_materialized_before_measurement": True,
                "all_requests_alive_through_generation": True,
            },
        },
        "memory_kernel_ledgers": verified_ledgers,
        "source_physical_document_sha256_before": source_before,
        "source_physical_document_sha256_after": source_after,
        "source_physical_payload_sha256": source_payload_before,
        "semantics": semantics,
    }


def _write_witness_phase(
    *,
    artifact_root: Path,
    path_prefix: str,
    rank: int,
    run_id: str,
    cell_id: str,
    resident_count: int,
    kv_policy: str,
    gdn_base_policy: str,
    phase: str,
    completed_request_indices: Sequence[int],
    persistent: Any,
    group: Any,
    plan: Any,
    persistent_guard: Any,
    request_guard: Any,
    kv_guard: LiveKVIdentityGuard,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gdn = capture_gdn_phase_witness(
        persistent,
        group.requests,
        plan.linear_layer_indices,
        run_id=run_id,
        cell_id=cell_id,
        kv_policy=kv_policy,
        phase=phase,
        # Pass the real helper constant; storage-witness canonicalization is
        # explicit and must never be derived from the independent KV axis.
        policy=gdn_base_policy,
        persistent_guard=persistent_guard,
        request_guard=request_guard,
        completed_request_indices=list(completed_request_indices),
    )
    kv = capture_live_kv_witness(
        persistent,
        group,
        plan,
        kv_guard,
        phase=phase,
        capture_id=gdn["capture_id"],
        completed_request_indices=list(completed_request_indices),
    )
    binding = {
        "rank": rank,
        "run_id": run_id,
        "cell_id": cell_id,
        "resident_count": resident_count,
        "kv_policy": kv_policy,
        "gdn_base_policy": gdn_base_policy,
        "gdn_policy": GDN_POLICY_TO_WITNESS[gdn_base_policy],
        "phase": phase,
    }
    wrapper = {
        "schema_version": PHASE_ARTIFACT_SCHEMA_VERSION,
        "binding": binding,
        "gdn_phase_witness": gdn,
        "kv_ownership_witness": kv,
    }
    path = artifact_root / path_prefix / f"phase-{phase}.json"
    _write_json(path, wrapper)
    return artifact_reference(path, root=artifact_root), gdn


def _write_witness_timeline_manifest(
    *,
    artifact_root: Path,
    path_prefix: str,
    rank: int,
    run_id: str,
    cell_id: str,
    resident_count: int,
    kv_policy: str,
    gdn_base_policy: str,
    phase_refs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    binding = {
        "rank": rank,
        "run_id": run_id,
        "cell_id": cell_id,
        "resident_count": resident_count,
        "kv_policy": kv_policy,
        "gdn_base_policy": gdn_base_policy,
        "gdn_policy": GDN_POLICY_TO_WITNESS[gdn_base_policy],
    }
    manifest = {
        "schema_version": "qcomem-gdn-external-timeline-manifest-v1",
        "binding": binding,
        "phase_artifacts": [dict(row) for row in phase_refs],
    }
    path = artifact_root / path_prefix / "timeline-manifest.json"
    _write_json(path, manifest)
    return artifact_reference(path, root=artifact_root)


def _run_ownership_witness_cell(
    *,
    artifact_root: Path,
    run_id: str,
    rank: int,
    arm_id: str,
    resident_count: int,
    kv_policy: str,
    gdn_base_policy: str,
    model: Any,
    backbone: Any,
    plan: Any,
    document: Any,
    queries: Sequence[Any],
    kernel: Any,
    oracle_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently rebuild and replay one strict ownership cell."""

    import torch
    from run_qcomem_qwen35_vllm_paged_multifork_resident import (
        _set_production_no_mask,
    )

    _require(torch.is_inference_mode_enabled(), "witness GPU cell requires inference mode")
    persistent, _conversion = _convert_persistent(
        backbone,
        plan,
        document,
        resident_count=resident_count,
    )
    source_before = source_document_physical_digests(
        persistent, plan.full_attention_layer_indices
    )
    source_payload_before = _physical_document_payload_digests(
        persistent, plan.full_attention_layer_indices
    )
    persistent_guard = capture_persistent_gdn_guard(
        persistent, plan.linear_layer_indices
    )
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=resident_count,
        policy=kv_policy,
        gdn_base_policy=gdn_base_policy,
    )
    _set_production_no_mask(group, plan.full_attention_layer_indices)
    request_guard = capture_request_gdn_binding_guard(
        group.requests,
        plan.linear_layer_indices,
        policy=gdn_base_policy,
    )
    kv_guard = capture_live_kv_identity_guard(group, plan)
    cell_id = f"rank-{rank}-N-{resident_count}-{arm_id}-ownership-witness"
    path_prefix = (
        f"rank-{rank}/N-{resident_count}/"
        f"arm-{sha256_json(arm_id)[:16]}/witness"
    )

    is_oracle_cell = (
        resident_count == ORACLE_RESIDENT_COUNT
        and kv_policy == oracle_selection["kv_policy"]
        and gdn_base_policy == oracle_selection["gdn_base_policy"]
    )
    append_collectors: dict[int, OracleAppendShadowCollector] = {}
    position_collectors = {
        request_index: PositionEvidenceCollector(request_index, [])
        for request_index in range(resident_count)
    }
    live_capture: OracleLiveCallCapture | None = None
    if is_oracle_cell:
        _require(oracle_selection["request_index"] == 0, "oracle request drift")
        for layer_index in plan.full_attention_layer_indices:
            collector = OracleAppendShadowCollector(
                sample_id=f"{oracle_selection['sample_id']}-layer-{int(layer_index)}",
                events=[],
            )
            append_collectors[int(layer_index)] = collector
            group.requests[0].layers[layer_index].sequence.append_observer = collector
        live_capture = OracleLiveCallCapture(
            selection=oracle_selection,
            append_collectors=append_collectors,
        )
    # Every strict witness ledger has a pointer-free append receipt.  Only the
    # single preregistered oracle request retains dense append shadows; all
    # other cells use a lightweight unique-ID collector.
    for request_index, request in enumerate(group.requests):
        for layer_index in plan.full_attention_layer_indices:
            sequence = request.layers[layer_index].sequence
            if is_oracle_cell and request_index == 0:
                _require(
                    sequence.append_observer is append_collectors[int(layer_index)],
                    "oracle append observer binding drift",
                )
                continue
            _require(
                sequence.append_observer is None,
                "strict witness inherited an unexpected append observer",
            )
            sequence.append_observer = AppendReceiptCollector(
                prefix=(
                    f"{cell_id}-request-{request_index}-"
                    f"layer-{int(layer_index)}"
                )
            )

    phase_refs: list[dict[str, Any]] = []
    setup_ref, _setup = _write_witness_phase(
        artifact_root=artifact_root,
        path_prefix=path_prefix,
        rank=rank,
        run_id=run_id,
        cell_id=cell_id,
        resident_count=resident_count,
        kv_policy=kv_policy,
        gdn_base_policy=gdn_base_policy,
        phase=PHASE_SETUP_PRE_TRANSITION,
        completed_request_indices=[],
        persistent=persistent,
        group=group,
        plan=plan,
        persistent_guard=persistent_guard,
        request_guard=request_guard,
        kv_guard=kv_guard,
    )
    phase_refs.append(setup_ref)
    transition_written = False

    # Construct the request-bound ledgers after the setup snapshot but before
    # the first append.  From this point onward the generation try/finally owns
    # every registered backend.
    call_observers: dict[int, Callable[[Mapping[str, Any]], None]] = {}
    for request_index, position_collector in position_collectors.items():
        observers: list[Callable[[Mapping[str, Any]], None]] = [
            position_collector
        ]
        if live_capture is not None and request_index == 0:
            observers.append(live_capture)
        call_observers[request_index] = ChainedCallObserver(tuple(observers))
    ledgers, backends = _make_ledgers_and_backends(
        group,
        plan,
        kernel=kernel,
        strict_position_values=True,
        call_observers=call_observers,
    )

    def after_step(round_index: int, request_index: int) -> None:
        nonlocal transition_written
        if round_index == 0 and request_index == 0:
            if live_capture is not None and int(oracle_selection["round_index"]) == 0:
                _require(live_capture.row is not None, "round-zero oracle call was not captured")
                live_capture.row["active_block_table_sha256"] = _active_block_table_digest(
                    group.requests[0].layers[
                        int(oracle_selection["layer_index"])
                    ].sequence
                )
                live_capture.row["active_block_table"] = (
                    group.requests[0]
                    .layers[int(oracle_selection["layer_index"])]
                    .sequence.active_block_table.detach()
                    .contiguous()
                    .cpu()
                    .clone()
                )
            transition_ref, _transition = _write_witness_phase(
                artifact_root=artifact_root,
                path_prefix=path_prefix,
                rank=rank,
                run_id=run_id,
                cell_id=cell_id,
                resident_count=resident_count,
                kv_policy=kv_policy,
                gdn_base_policy=gdn_base_policy,
                phase=PHASE_POST_TRANSITION,
                completed_request_indices=[0],
                persistent=persistent,
                group=group,
                plan=plan,
                persistent_guard=persistent_guard,
                request_guard=request_guard,
                kv_guard=kv_guard,
            )
            phase_refs.append(transition_ref)
            transition_written = True
        if (
            live_capture is not None
            and request_index == int(oracle_selection["request_index"])
            and round_index == int(oracle_selection["round_index"])
            and live_capture.row is not None
            and live_capture.row["active_block_table_sha256"] is None
        ):
            live_capture.row["active_block_table_sha256"] = _active_block_table_digest(
                group.requests[request_index].layers[
                    int(oracle_selection["layer_index"])
                ].sequence
            )
            live_capture.row["active_block_table"] = (
                group.requests[request_index]
                .layers[int(oracle_selection["layer_index"])]
                .sequence.active_block_table.detach()
                .contiguous()
                .cpu()
                .clone()
            )

    with _registered_backend_scope(backends):
        trajectories, _unused_endpoint, _unused_diagnostics = _gpu_round_robin_generate(
            model,
            backbone,
            group,
            queries[:resident_count],
            backends,
            after_step=after_step,
            measure_allocator=False,
        )
    _require(transition_written, "witness missed request-zero transition capture")
    final_ref, _final = _write_witness_phase(
        artifact_root=artifact_root,
        path_prefix=path_prefix,
        rank=rank,
        run_id=run_id,
        cell_id=cell_id,
        resident_count=resident_count,
        kv_policy=kv_policy,
        gdn_base_policy=gdn_base_policy,
        phase=PHASE_POST_GENERATION,
        completed_request_indices=list(range(resident_count)),
        persistent=persistent,
        group=group,
        plan=plan,
        persistent_guard=persistent_guard,
        request_guard=request_guard,
        kv_guard=kv_guard,
    )
    phase_refs.append(final_ref)
    _require(len(phase_refs) == 3, "witness phase artifact count drift")
    timeline_ref = _write_witness_timeline_manifest(
        artifact_root=artifact_root,
        path_prefix=path_prefix,
        rank=rank,
        run_id=run_id,
        cell_id=cell_id,
        resident_count=resident_count,
        kv_policy=kv_policy,
        gdn_base_policy=gdn_base_policy,
        phase_refs=phase_refs,
    )
    validate_runtime_kv_ownership(
        persistent, group, plan, require_appended_tail_cow=True
    )
    source_after = source_document_physical_digests(
        persistent, plan.full_attention_layer_indices
    )
    source_payload_after = _physical_document_payload_digests(
        persistent, plan.full_attention_layer_indices
    )
    _require(source_after == source_before, "witness cell mutated physical source")
    _require(source_payload_after == source_payload_before, "witness cell mutated source padding")
    verified_ledgers = [
        position_collectors[request_index].attach(
            _pointer_free_kernel_ledger(ledger.verify_complete())
        )
        for request_index, ledger in enumerate(ledgers)
    ]
    semantics = _semantic_rows_from_live(group, plan, trajectories)
    oracle_ref = None
    if is_oracle_cell:
        assert live_capture is not None
        _require(live_capture.row is not None, "preregistered oracle call missing")
        _require(
            isinstance(live_capture.row["active_block_table_sha256"], str),
            "oracle active block-table digest missing",
        )
        oracle_ref = _write_live_oracle_artifact(
            artifact_root=artifact_root,
            path_prefix=path_prefix,
            run_id=run_id,
            rank=rank,
            cell_id=cell_id,
            arm_id=arm_id,
            selection=oracle_selection,
            persistent=persistent,
            group=group,
            source_payload_sha256=source_payload_before[
                str(int(oracle_selection["layer_index"]))
            ],
            append_collector=append_collectors[
                int(oracle_selection["layer_index"])
            ],
            live_capture=live_capture.row,
            selected_ledger=verified_ledgers[0],
        )
    return {
        "witness_cell": {
            "cell_role": "ownership_witness",
            "rank": rank,
            "resident_count": resident_count,
            "arm_id": arm_id,
            "cell_id": cell_id,
            "request_guard_created": True,
            "witness_capture_executed": True,
            "primary_memory_endpoint_eligible": False,
            "rebuilt_persistent_cache": True,
            "rebuilt_request_group": True,
            "timeline_manifest_artifact": timeline_ref,
        },
        "witness_kernel_ledgers": verified_ledgers,
        "source_physical_document_sha256_before": source_before,
        "source_physical_document_sha256_after": source_after,
        "source_physical_payload_sha256": source_payload_before,
        "semantics": semantics,
        "oracle_raw_artifact": oracle_ref,
    }


def _execute_cell_with_cleanup(
    run_cell: Callable[[], dict[str, Any]],
    *,
    cell_role_key: str,
    frozen_baseline: Mapping[str, int],
    label: str,
) -> dict[str, Any]:
    """Bracket one disposable cell and prove allocator baseline recovery."""

    before = _gpu_cleanup()
    for field in ("current_allocated_bytes", "current_reserved_bytes"):
        _require(
            before[field] == frozen_baseline[field],
            f"allocator baseline drift before {label}: {field}",
        )
    result: dict[str, Any] | None = None
    try:
        result = run_cell()
    except BaseException as primary:
        # A traceback can otherwise retain the disposable persistent/request
        # caches through its child-frame locals and make the cleanup audit
        # fail for the wrong reason.  Clear those locals before allocator
        # recovery, preserve the primary error, and attach cleanup failure only
        # as secondary evidence.
        primary_traceback = primary.__traceback__
        if primary_traceback is not None:
            traceback.clear_frames(primary_traceback)
        try:
            after_failure = _gpu_cleanup()
            for field in ("current_allocated_bytes", "current_reserved_bytes"):
                _require(
                    after_failure[field] == frozen_baseline[field],
                    f"allocator did not recover after failed {label}: {field}",
                )
        except BaseException as cleanup:
            _add_exception_note(
                primary, f"secondary failed-cell cleanup error: {cleanup}"
            )
            raise primary.with_traceback(primary_traceback) from cleanup
        raise primary.with_traceback(primary_traceback)
    else:
        after = _gpu_cleanup()
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(
                after[field] == frozen_baseline[field],
                f"allocator did not recover after {label}: {field}",
            )
    assert result is not None
    role = result.get(cell_role_key)
    _require(isinstance(role, dict), f"{label} role payload missing")
    role["cleanup_receipt"] = _make_cleanup_receipt(
        before=before,
        after=after,
        frozen_baseline=frozen_baseline,
    )
    return result


def _make_cleanup_receipt(
    *,
    before: Mapping[str, int],
    after: Mapping[str, int],
    frozen_baseline: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "schema_version": "qcomem-cell-cleanup-receipt-v1",
        "before_cell": dict(before),
        "after_cleanup": dict(after),
        "frozen_model_query_baseline": dict(frozen_baseline),
        "explicit_python_references_dropped_on_return": True,
        "gc_collect_completed": True,
        "cuda_empty_cache_completed": True,
        "cuda_synchronize_completed": True,
        "current_allocated_and_reserved_exactly_recovered": True,
    }


def _mutant_kernel_sentinel(*_args: Any, **_kwargs: Any) -> Any:
    """A swapped kernel must be rejected before this callable can execute."""

    raise AssertionError("mutant kernel sentinel escaped KERNEL_CALLABLE_ID")


def _target_binding(
    *,
    mutant_id: str,
    case_cell_id: str,
    pre_descriptor: Mapping[str, Any],
    mutated_descriptor: Mapping[str, Any],
    capture_restored_descriptor: Callable[[], Mapping[str, Any]],
) -> TargetMutationBinding:
    kind, field = MUTANT_TARGET_CONTRACT[mutant_id]
    return TargetMutationBinding(
        mutant_id=mutant_id,
        case_cell_id=case_cell_id,
        capture_id=f"{case_cell_id}-target",
        target_kind=kind,
        target_field=field,
        pre_sha256=sha256_json(pre_descriptor),
        mutated_sha256=sha256_json(mutated_descriptor),
        capture_restored_sha256=lambda: sha256_json(
            capture_restored_descriptor()
        ),
    )


def _mutant_exercise_contract_sha256(mutant_id: str) -> str:
    """Freeze the detector exercise shared by a mutant and its clean twin."""

    _require(mutant_id in MUTANT_SPECS, f"unknown mutant exercise {mutant_id}")
    spec = MUTANT_SPECS[mutant_id]
    target_kind, target_field = MUTANT_TARGET_CONTRACT[mutant_id]
    return sha256_json(
        {
            "schema_version": "forkaudit-mutant-exercise-contract-v1",
            "mutant_id": mutant_id,
            "injection_stage": spec.injection_stage.value,
            "expected_gate_id": spec.expected_gate_id,
            "target_kind": target_kind,
            "target_field": target_field,
            "detector_path": MUTANT_EXERCISE_PATHS[mutant_id],
            "matched_and_injected_share_dispatch_branch": True,
        }
    )


def _tensor_target_descriptor(
    tensor: Any,
    *,
    logical_slot: str,
    binding_role: str,
    include_values: bool = True,
) -> dict[str, Any]:
    descriptor = {
        "schema_version": "forkaudit-live-target-tensor-v1",
        "logical_slot": logical_slot,
        "binding_role": binding_role,
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
    }
    if include_values:
        descriptor["tensor_sha256"] = _tensor_digest(tensor)
    return descriptor


def _callable_target_descriptor(
    value: Any, *, logical_slot: str, binding_role: str
) -> dict[str, Any]:
    import inspect

    function = getattr(value, "__func__", value)
    try:
        signature = str(inspect.signature(function))
    except (TypeError, ValueError):
        signature = "<signature-unavailable>"
    return {
        "schema_version": "forkaudit-live-target-callable-v1",
        "logical_slot": logical_slot,
        "binding_role": binding_role,
        "module": str(getattr(function, "__module__", type(function).__module__)),
        "qualname": str(
            getattr(function, "__qualname__", type(function).__qualname__)
        ),
        "signature": signature,
    }


def _run_mutant_forward_prefix(
    *,
    backbone: Any,
    group: Any,
    plan: Any,
    queries: Sequence[Any],
    kernel: Any,
    request_indices: Sequence[int],
) -> list[dict[str, Any]]:
    """Run one real 32-token transition for selected requests."""

    import torch

    ledgers = []
    backends = []
    original_backend = backbone.config._attn_implementation
    try:
        for request_index in request_indices:
            ledger = MultiForkHitLedger(
                plan,
                group.requests[request_index],
                request_index=request_index,
                resident_count=group.resident_count,
                request_policy=group.policy,
                expected_calls_per_layer=1,
                initial_query_tokens=FORMAL_QUERY_TOKENS,
                kernel=kernel,
                strict_position_values=True,
                call_observer=None,
            )
            backend = register_multifork_backend(ledger)
            ledgers.append(ledger)
            backends.append(backend)
        for request_index, backend in zip(request_indices, backends):
            backbone.config._attn_implementation = backend
            output = backbone(
                input_ids=queries[request_index],
                past_key_values=group.requests[request_index],
                use_cache=True,
            )
            del output
            torch.cuda.synchronize()
        return [ledger.verify_complete() for ledger in ledgers]
    finally:
        backbone.config._attn_implementation = original_backend
        _unregister_backends(backends)


def _make_mutant_live_context(
    *,
    mutant_id: str,
    case_cell_id: str,
    model: Any,
    backbone: Any,
    plan: Any,
    document: Any,
    queries: Sequence[Any],
    kernel: Any,
) -> dict[str, Any]:
    """Build one fresh N=2 document/request cache for exactly one mutant."""

    del model
    from run_qcomem_qwen35_vllm_paged_multifork_resident import (
        _set_production_no_mask,
    )

    persistent, _conversion = _convert_persistent(
        backbone, plan, document, resident_count=2
    )
    persistent_guard = capture_persistent_gdn_guard(
        persistent, plan.linear_layer_indices
    )
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=2,
        policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    _set_production_no_mask(group, plan.full_attention_layer_indices)
    request_guard = capture_request_gdn_binding_guard(
        group.requests,
        plan.linear_layer_indices,
        policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    return {
        "mutant_id": mutant_id,
        "case_cell_id": case_cell_id,
        "backbone": backbone,
        "plan": plan,
        "persistent": persistent,
        "persistent_guard": persistent_guard,
        "request_guard": request_guard,
        "group": group,
        "queries": queries,
        "kernel": kernel,
    }


def _mutant_detector_input(mutant_id: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "forkaudit-mutant-detector-input-v2",
        "mutant_id": mutant_id,
        "detector_path": MUTANT_EXERCISE_PATHS[mutant_id],
        "expected_gate_id": MUTANT_SPECS[mutant_id].expected_gate_id,
        "resident_count": 2,
        "kv_policy": SHARED_REUSE,
        "gdn_base_policy": GDN_BORROW_IMMUTABLE_BASE,
        "evidence": dict(evidence),
    }


def _compact_full_forward_mutant_ledger(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep the exact live fields needed to prove a one-round model traversal."""

    identity = value.get("kernel_identity")
    _require(isinstance(identity, dict), "mutant full-forward kernel identity missing")
    descriptor = {
        field: identity.get(field) for field in ("module", "qualname", "signature")
    }
    calls = value.get("calls")
    _require(isinstance(calls, (list, tuple)), "mutant full-forward calls missing")
    projected_calls = []
    for call in calls:
        audit = call.get("append_audit")
        projected_calls.append(
            {
                "layer_idx": call.get("layer_idx"),
                "query_tokens": call.get("query_tokens"),
                "kv_tokens": call.get("kv_tokens"),
                "kernel_mode": call.get("kernel_mode"),
                "quantization": call.get("quantization"),
                "mask_contract": call.get("mask_contract"),
                "position_ids_contract": call.get("position_ids_contract"),
                "position_ids_expected_tail_start": call.get(
                    "position_ids_expected_tail_start"
                ),
                "position_ids_expected_tail_end_exclusive": call.get(
                    "position_ids_expected_tail_end_exclusive"
                ),
                "softmax_scale": call.get("softmax_scale"),
                "append_event_index": (
                    audit.get("append_event_index") if isinstance(audit, dict) else None
                ),
                "appended_tokens_before": (
                    audit.get("appended_tokens_before") if isinstance(audit, dict) else None
                ),
                "appended_tokens_after": (
                    audit.get("appended_tokens_after") if isinstance(audit, dict) else None
                ),
            }
        )
    counts = value.get("counts")
    _require(isinstance(counts, dict), "mutant full-forward layer counts missing")
    return {
        "schema_version": "forkaudit-mutant-full-forward-coverage-v1",
        "verified": value.get("verified"),
        "request_index": value.get("request_index"),
        "resident_count": value.get("resident_count"),
        "request_policy": value.get("request_policy"),
        "initial_query_tokens": value.get("initial_query_tokens"),
        "total_calls": value.get("total_calls"),
        "counts": {str(key): item for key, item in counts.items()},
        "same_unified_attention_kernel": value.get("same_unified_attention_kernel"),
        "dense_fallback_calls": value.get("dense_fallback_calls"),
        "full_kv_concatenations": value.get("full_kv_concatenations"),
        "kernel_identity": descriptor,
        "calls": projected_calls,
    }


def _compact_direct_mutant_call(
    ledger: Any, *, layer_index: int, call_count_before: int
) -> dict[str, Any]:
    calls = ledger.calls
    _require(
        len(calls) == call_count_before + 1,
        "direct mutant detector did not consume exactly one ledger call",
    )
    call = calls[-1]
    audit = call.get("append_audit")
    identity = call.get("kernel_identity")
    _require(
        isinstance(audit, dict) and isinstance(identity, dict),
        "direct mutant ledger call evidence missing",
    )
    return {
        "schema_version": "forkaudit-mutant-direct-call-coverage-v1",
        "call_count_before": call_count_before,
        "call_count_after": len(calls),
        "layer_index": layer_index,
        "request_index": call.get("request_index"),
        "resident_count": call.get("resident_count"),
        "request_policy": call.get("request_policy"),
        "query_tokens": call.get("query_tokens"),
        "kv_tokens": call.get("kv_tokens"),
        "kernel_mode": call.get("kernel_mode"),
        "quantization": call.get("quantization"),
        "mask_contract": call.get("mask_contract"),
        "position_ids_contract": call.get("position_ids_contract"),
        "position_ids_expected_tail_start": call.get(
            "position_ids_expected_tail_start"
        ),
        "position_ids_expected_tail_end_exclusive": call.get(
            "position_ids_expected_tail_end_exclusive"
        ),
        "softmax_scale": call.get("softmax_scale"),
        "append_event_index": audit.get("append_event_index"),
        "appended_tokens_before": audit.get("appended_tokens_before"),
        "appended_tokens_after": audit.get("appended_tokens_after"),
        "kernel_identity": {
            field: identity.get(field)
            for field in ("module", "qualname", "signature")
        },
    }


def _run_one_live_mutant(
    mutant_id: str,
    *,
    rank: int,
    model: Any,
    backbone: Any,
    plan: Any,
    document: Any,
    queries: Sequence[Any],
    kernel: Any,
    activate_mutation: bool = True,
) -> dict[str, Any]:
    """Execute one preregistered fault in a disposable real cache."""

    import torch

    case_role = "mutant" if activate_mutation else "matched-clean"
    case_cell_id = f"rank-{rank}-{case_role}-{mutant_id}-fresh-N-2"
    context = _make_mutant_live_context(
        mutant_id=mutant_id,
        case_cell_id=case_cell_id,
        model=model,
        backbone=backbone,
        plan=plan,
        document=document,
        queries=queries,
        kernel=kernel,
    )
    coverage_state = {
        "exercise_started": False,
        "detector_path_completed": False,
        "detector_input": None,
        "completion_receipt": None,
    }
    context["exercise_coverage_state"] = coverage_state
    group = context["group"]
    persistent = context["persistent"]
    first_full = int(plan.full_attention_layer_indices[0])
    first_linear = int(plan.linear_layer_indices[0])

    def mark_detector_input(evidence: Mapping[str, Any]) -> dict[str, Any]:
        _require(
            coverage_state["detector_input"] is None,
            f"{mutant_id} detector input was captured more than once",
        )
        receipt = _mutant_detector_input(mutant_id, evidence)
        coverage_state["detector_input"] = receipt
        return receipt

    if mutant_id == "M1":
        m1_target = group.requests[1].layers[first_full].sequence.reservations

        def descriptor_m1(role: str) -> dict[str, Any]:
            return _tensor_target_descriptor(
                m1_target,
                logical_slot=f"request1/layer{first_full}/reservations",
                binding_role=role,
            )

        def apply_m1(state: dict[str, Any]) -> AppliedMutation:
            left = state["group"].requests[0].layers[first_full].sequence
            right = state["group"].requests[1].layers[first_full].sequence
            target = right.reservations
            saved = target.detach().clone()
            replacement = left.reservations.detach().clone()
            _require(not torch.equal(saved, replacement), "M1 reservation mutation is a no-op")

            pre = descriptor_m1("request1-construction-reservation")
            target.copy_(replacement)
            mutated = descriptor_m1("request0-reservation-aliased-into-request1")
            return AppliedMutation(
                undo=lambda: target.copy_(saved),
                verify_restored=lambda: bool(torch.equal(target, saved)),
                target_binding=_target_binding(
                    mutant_id=mutant_id,
                    case_cell_id=case_cell_id,
                    pre_descriptor=pre,
                    mutated_descriptor=mutated,
                    capture_restored_descriptor=lambda: descriptor_m1(
                        "request1-construction-reservation"
                    ),
                ),
            )

        injector = callback_injector(apply_m1)
        def exercise(state: dict[str, Any]) -> dict[str, Any]:
            peer_sequence = state["group"].requests[0].layers[
                first_full
            ].sequence
            target_sequence = state["group"].requests[1].layers[
                first_full
            ].sequence
            target_descriptor = descriptor_m1(
                "request0-reservation-aliased-into-request1"
                if activate_mutation
                else "request1-construction-reservation"
            )
            mark_detector_input(
                {
                    "kind": "live-kv-ownership",
                    "require_appended_tail_cow": False,
                    "full_attention_layers": list(
                        state["plan"].full_attention_layer_indices
                    ),
                    "target_reservations_sha256": _tensor_digest(
                        target_sequence.reservations
                    ),
                    "peer_request0_reservations_sha256": _tensor_digest(
                        peer_sequence.reservations
                    ),
                    "target_descriptor": target_descriptor,
                    "target_descriptor_sha256": sha256_json(target_descriptor),
                    "construction_guard_row_count": len(
                        state["group"].kv_binding_guard.rows
                    ),
                }
            )
            return validate_runtime_kv_ownership(
                state["persistent"],
                state["group"],
                state["plan"],
                require_appended_tail_cow=False,
            )

    elif mutant_id == "M2":
        ledger = MultiForkHitLedger(
            plan,
            group.requests[0],
            request_index=0,
            resident_count=2,
            request_policy=group.policy,
            expected_calls_per_layer=1,
            initial_query_tokens=FORMAL_QUERY_TOKENS,
            kernel=kernel,
            strict_position_values=True,
        )
        context.update({"ledger": ledger, "register_backend_ledger": ledger})
        binding_secret = secrets.token_bytes(32)
        original_sequence_id = ledger.sequence_ids[first_full]
        peer_sequence_id = id(group.requests[1].layers[first_full].sequence)

        def descriptor_m2(role: str) -> dict[str, Any]:
            current = ledger.sequence_ids[first_full]
            return {
                "schema_version": "forkaudit-live-sequence-binding-v1",
                "logical_slot": f"request0/ledger/layer{first_full}/sequence",
                "binding_role": role,
                "opaque_current_binding_token": _opaque_token(
                    binding_secret, case_cell_id, first_full, current
                ),
                "matches_frozen_request0_sequence": current
                == original_sequence_id,
                "matches_live_peer_request1_sequence": current
                == peer_sequence_id,
            }

        def apply_m2(state: dict[str, Any]) -> AppliedMutation:
            target = state["ledger"].sequence_ids
            original = target[first_full]
            peer = state["group"].requests[1].layers[first_full].sequence
            _require(id(peer) != original, "M2 sequence mutation is a no-op")
            pre = descriptor_m2("request0-frozen-sequence")
            target[first_full] = id(peer)
            mutated = descriptor_m2("request1-live-sequence")
            return AppliedMutation(
                undo=lambda: target.__setitem__(first_full, original),
                verify_restored=lambda: target[first_full] == original,
                target_binding=_target_binding(
                    mutant_id=mutant_id,
                    case_cell_id=case_cell_id,
                    pre_descriptor=pre,
                    mutated_descriptor=mutated,
                    capture_restored_descriptor=lambda: descriptor_m2(
                        "request0-frozen-sequence"
                    ),
                ),
            )

        injector = callback_injector(apply_m2)

        def exercise(state: dict[str, Any]) -> None:
            mark_detector_input(
                {
                    "kind": "live-full-model-forward-ledger",
                    "backend_registered": isinstance(state.get("backend"), str),
                    "request_index": 0,
                    "initial_query_tokens": FORMAL_QUERY_TOKENS,
                    "expected_calls_per_layer": 1,
                    "expected_full_attention_layers": list(
                        state["plan"].full_attention_layer_indices
                    ),
                    "call_count_before": len(state["ledger"].calls),
                    "query_token_ids_sha256": _token_id_sha256(
                        queries[0], expected_shape=(1, FORMAL_QUERY_TOKENS)
                    ),
                }
            )
            original_backend = backbone.config._attn_implementation
            try:
                backbone.config._attn_implementation = state["backend"]
                backbone(
                    input_ids=queries[0],
                    past_key_values=state["group"].requests[0],
                    use_cache=True,
                )
                return _compact_full_forward_mutant_ledger(
                    state["ledger"].verify_complete()
                )
            finally:
                backbone.config._attn_implementation = original_backend

    elif mutant_id == "M3":
        import types

        sequence = group.requests[0].layers[first_full].sequence

        def omit_tail_cow(_self: Any, _batch_index: int) -> None:
            return None

        def descriptor_m3(role: str) -> dict[str, Any]:
            return _callable_target_descriptor(
                sequence._detach_partial_document_tail,
                logical_slot=f"request0/layer{first_full}/detach-partial-tail",
                binding_role=role,
            )

        def apply_m3(_state: dict[str, Any]) -> AppliedMutation:
            _require(
                "_detach_partial_document_tail" not in vars(sequence),
                "M3 target already has an instance override",
            )
            pre = descriptor_m3("class-tail-cow-implementation")
            sequence._detach_partial_document_tail = types.MethodType(  # type: ignore[method-assign]
                omit_tail_cow, sequence
            )
            mutated = descriptor_m3("omitted-tail-cow-instance-override")

            def undo() -> None:
                del vars(sequence)["_detach_partial_document_tail"]

            return AppliedMutation(
                undo=undo,
                verify_restored=lambda: (
                    "_detach_partial_document_tail" not in vars(sequence)
                ),
                target_binding=_target_binding(
                    mutant_id=mutant_id,
                    case_cell_id=case_cell_id,
                    pre_descriptor=pre,
                    mutated_descriptor=mutated,
                    capture_restored_descriptor=lambda: descriptor_m3(
                        "class-tail-cow-implementation"
                    ),
                ),
            )

        injector = callback_injector(apply_m3)

        def exercise(state: dict[str, Any]) -> dict[str, Any]:
            # The corresponding clean exercise must pass.  Advance every
            # request/layer once so the whole-group require-tail gate cannot
            # be satisfied merely because an unrelated layer was never
            # appended.  Only request0/first_full has the omitted-COW hook.
            for request in state["group"].requests:
                for layer_index in state["plan"].full_attention_layer_indices:
                    layer = request.layers[layer_index]
                    arena = layer.sequence.arena
                    key = torch.zeros(
                        (1, arena.num_key_value_heads, 1, arena.head_dim),
                        dtype=arena.key_cache.dtype,
                        device=arena.key_cache.device,
                    )
                    layer.update(key, key)
            appended = [
                {
                    "request_index": request_index,
                    "layer_index": int(layer_index),
                    "appended_tokens": int(
                        request.layers[layer_index].sequence.appended_tokens
                    ),
                }
                for request_index, request in enumerate(state["group"].requests)
                for layer_index in state["plan"].full_attention_layer_indices
            ]
            mark_detector_input(
                {
                    "kind": "live-kv-ownership",
                    "require_appended_tail_cow": True,
                    "full_attention_layers": list(
                        state["plan"].full_attention_layer_indices
                    ),
                    "appended_tokens_by_request_layer": appended,
                    "all_request_layers_appended_once": all(
                        row["appended_tokens"] == 1 for row in appended
                    ),
                }
            )
            return validate_runtime_kv_ownership(
                state["persistent"],
                state["group"],
                state["plan"],
                require_appended_tail_cow=True,
            )

    elif mutant_id in ("M4", "M5"):
        completed = [0] if mutant_id == "M4" else [0, 1]
        transition_ledger_rows = _run_mutant_forward_prefix(
            backbone=backbone,
            group=group,
            plan=plan,
            queries=queries,
            kernel=kernel,
            request_indices=completed,
        )
        transition_forward_ledgers = [
            _compact_full_forward_mutant_ledger(row)
            for row in transition_ledger_rows
        ]
        verify_request_gdn_binding_guard(
            context["request_guard"],
            group.requests,
            completed_request_indices=completed,
        )
        target_request = 0 if mutant_id == "M4" else 1
        target_values = group.requests[target_request].layers[
            first_linear
        ].conv_states
        state_index = sorted(target_values)[0]
        original = target_values[state_index]
        alias = (
            persistent.layers[first_linear].conv_states[state_index]
            if mutant_id == "M4"
            else group.requests[0].layers[first_linear].conv_states[state_index]
        )
        _require(original is not alias, f"{mutant_id} GDN mutation is a no-op")

        def descriptor_gdn(role: str) -> dict[str, Any]:
            return _tensor_target_descriptor(
                target_values[state_index],
                logical_slot=(
                    f"request{target_request}/layer{first_linear}/"
                    f"conv_states/{state_index}"
                ),
                binding_role=role,
            )

        def apply_gdn(_state: dict[str, Any]) -> AppliedMutation:
            pre = descriptor_gdn("request-local-transitioned-state")
            target_values[state_index] = alias
            mutated = descriptor_gdn(
                "persistent-base-exact-alias"
                if mutant_id == "M4"
                else "peer-request0-exact-alias"
            )
            return AppliedMutation(
                undo=lambda: target_values.__setitem__(state_index, original),
                verify_restored=lambda: target_values[state_index] is original,
                target_binding=_target_binding(
                    mutant_id=mutant_id,
                    case_cell_id=case_cell_id,
                    pre_descriptor=pre,
                    mutated_descriptor=mutated,
                    capture_restored_descriptor=lambda: descriptor_gdn(
                        "request-local-transitioned-state"
                    ),
                ),
            )

        injector = callback_injector(apply_gdn)

        def exercise(state: dict[str, Any]) -> dict[str, Any]:
            snapshot = capture_gdn_storage_snapshot(
                state["persistent"],
                state["group"].requests,
                state["plan"].linear_layer_indices,
                phase=PHASE_POST_TRANSITION,
                policy=GDN_BORROW_IMMUTABLE_BASE,
                persistent_guard=state["persistent_guard"],
                completed_request_indices=completed,
            )
            mark_detector_input(
                {
                    "kind": "live-gdn-storage-replay",
                    "phase": PHASE_POST_TRANSITION,
                    "completed_request_indices": list(completed),
                    "storage_witness": snapshot,
                    "storage_witness_sha256": sha256_json(snapshot),
                    "transition_forward_ledgers": transition_forward_ledgers,
                }
            )
            return bridge_named_gate_error(
                lambda: replay_gdn_storage_witness(
                    json.loads(json.dumps(snapshot))
                )
            )

    elif mutant_id in ("M6", "M7"):
        layer = group.requests[0].layers[first_full]
        sequence = layer.sequence
        ledger = MultiForkHitLedger(
            plan,
            group.requests[0],
            request_index=0,
            resident_count=2,
            request_policy=group.policy,
            expected_calls_per_layer=1,
            initial_query_tokens=1,
            kernel=kernel,
            strict_position_values=True,
        )
        key = torch.zeros(
            (1, FORMAL_NUM_KV_HEADS, 1, FORMAL_HEAD_DIM),
            dtype=sequence.arena.key_cache.dtype,
            device=sequence.arena.key_cache.device,
        )
        layer.update(key, key)
        query = torch.zeros(
            (1, FORMAL_NUM_QUERY_HEADS, 1, FORMAL_HEAD_DIM),
            dtype=sequence.arena.key_cache.dtype,
            device=sequence.arena.key_cache.device,
        )
        call_args = {
            "position_ids": torch.tensor(
                [[FORMAL_DOCUMENT_TOKENS]],
                dtype=torch.long,
                device=query.device,
            ),
            "attention_mask": None,
        }
        context.update(
            {"ledger": ledger, "layer": layer, "query": query, "call_args": call_args}
        )
        if mutant_id == "M6":
            target = call_args["position_ids"]
            saved = target.detach().clone()

            def descriptor_position(role: str) -> dict[str, Any]:
                return _tensor_target_descriptor(
                    target,
                    logical_slot="request0/direct-ledger/position_ids",
                    binding_role=role,
                )

            def apply_position(_state: dict[str, Any]) -> AppliedMutation:
                pre = descriptor_position("canonical-causal-tail")
                target[0, 0] += 1
                mutated = descriptor_position("perturbed-plus-one")
                return AppliedMutation(
                    undo=lambda: target.copy_(saved),
                    verify_restored=lambda: bool(torch.equal(target, saved)),
                    target_binding=_target_binding(
                        mutant_id=mutant_id,
                        case_cell_id=case_cell_id,
                        pre_descriptor=pre,
                        mutated_descriptor=mutated,
                        capture_restored_descriptor=lambda: descriptor_position(
                            "canonical-causal-tail"
                        ),
                    ),
                )

            injector = callback_injector(apply_position)
        else:
            materialized_mask = torch.ones(
                (1, 1, 1, sequence.sequence_length),
                dtype=torch.bool,
                device=query.device,
            )

            def descriptor_mask(role: str) -> dict[str, Any]:
                value = call_args["attention_mask"]
                result = {
                    "schema_version": "forkaudit-live-mask-argument-v1",
                    "logical_slot": "request0/direct-ledger/attention_mask",
                    "binding_role": role,
                    "representation": "none" if value is None else "materialized-tensor",
                }
                if value is not None:
                    result["tensor_sha256"] = _tensor_digest(value)
                return result

            def apply_mask(_state: dict[str, Any]) -> AppliedMutation:
                pre = descriptor_mask("production-none-mask")
                call_args["attention_mask"] = materialized_mask
                mutated = descriptor_mask("materialized-tail-causal-mask")
                return AppliedMutation(
                    undo=lambda: call_args.__setitem__("attention_mask", None),
                    verify_restored=lambda: call_args["attention_mask"] is None,
                    target_binding=_target_binding(
                        mutant_id=mutant_id,
                        case_cell_id=case_cell_id,
                        pre_descriptor=pre,
                        mutated_descriptor=mutated,
                        capture_restored_descriptor=lambda: descriptor_mask(
                            "production-none-mask"
                        ),
                    ),
                )

            injector = callback_injector(apply_mask)

        def exercise(state: dict[str, Any]) -> dict[str, Any]:
            call_count_before = len(state["ledger"].calls)
            position_ids = state["call_args"]["position_ids"]
            attention_mask = state["call_args"]["attention_mask"]
            mark_detector_input(
                {
                    "kind": "live-direct-ledger-call",
                    "layer_index": first_full,
                    "call_count_before": call_count_before,
                    "appended_tokens": int(
                        state["layer"].sequence.appended_tokens
                    ),
                    "query_sha256": _tensor_digest(state["query"]),
                    "query_dtype": str(state["query"].dtype),
                    "query_shape": list(state["query"].shape),
                    "position_ids_values": position_ids.detach()
                    .cpu()
                    .reshape(-1)
                    .tolist(),
                    "position_ids_sha256": _tensor_digest(position_ids),
                    "attention_mask_representation": (
                        "none" if attention_mask is None else "materialized-tensor"
                    ),
                    "attention_mask_sha256": (
                        None
                        if attention_mask is None
                        else _tensor_digest(attention_mask)
                    ),
                    "attention_mask_dtype": (
                        None if attention_mask is None else str(attention_mask.dtype)
                    ),
                    "attention_mask_shape": (
                        None
                        if attention_mask is None
                        else list(attention_mask.shape)
                    ),
                }
            )
            state["ledger"].attention_forward(
                backbone.layers[first_full].self_attn,
                state["query"],
                state["layer"].keys,
                state["layer"].values,
                state["call_args"]["attention_mask"],
                position_ids=state["call_args"]["position_ids"],
            )
            return _compact_direct_mutant_call(
                state["ledger"],
                layer_index=first_full,
                call_count_before=call_count_before,
            )

    elif mutant_id == "M8":
        ledger = MultiForkHitLedger(
            plan,
            group.requests[0],
            request_index=0,
            resident_count=2,
            request_policy=group.policy,
            expected_calls_per_layer=1,
            initial_query_tokens=FORMAL_QUERY_TOKENS,
            kernel=kernel,
            strict_position_values=True,
        )
        context.update({"ledger": ledger, "register_backend_ledger": ledger})

        def descriptor_kernel(value: Any, role: str) -> dict[str, Any]:
            return _callable_target_descriptor(
                value,
                logical_slot="request0/ledger/unified_attention",
                binding_role=role,
            )

        def apply_kernel(state: dict[str, Any]) -> AppliedMutation:
            target = state["ledger"]
            original = target.kernel
            pre = descriptor_kernel(original, "frozen-production-kernel")
            target.kernel = _mutant_kernel_sentinel
            mutated = descriptor_kernel(target.kernel, "swapped-sentinel-kernel")
            return AppliedMutation(
                undo=lambda: setattr(target, "kernel", original),
                verify_restored=lambda: target.kernel is original,
                target_binding=_target_binding(
                    mutant_id=mutant_id,
                    case_cell_id=case_cell_id,
                    pre_descriptor=pre,
                    mutated_descriptor=mutated,
                    capture_restored_descriptor=lambda: descriptor_kernel(
                        target.kernel, "frozen-production-kernel"
                    ),
                ),
            )

        injector = callback_injector(apply_kernel)

        def exercise(state: dict[str, Any]) -> dict[str, Any]:
            mark_detector_input(
                {
                    "kind": "live-full-model-forward-ledger",
                    "backend_registered": isinstance(state.get("backend"), str),
                    "request_index": 0,
                    "initial_query_tokens": FORMAL_QUERY_TOKENS,
                    "expected_calls_per_layer": 1,
                    "expected_full_attention_layers": list(
                        state["plan"].full_attention_layer_indices
                    ),
                    "call_count_before": len(state["ledger"].calls),
                    "query_token_ids_sha256": _token_id_sha256(
                        queries[0], expected_shape=(1, FORMAL_QUERY_TOKENS)
                    ),
                }
            )
            original_backend = backbone.config._attn_implementation
            try:
                backbone.config._attn_implementation = state["backend"]
                backbone(
                    input_ids=queries[0],
                    past_key_values=state["group"].requests[0],
                    use_cache=True,
                )
                return _compact_full_forward_mutant_ledger(
                    state["ledger"].verify_complete()
                )
            finally:
                backbone.config._attn_implementation = original_backend

    elif mutant_id == "M9":
        from qcomem_vllm_paged_kernel import Q16KernelPagedTensorView

        layer = group.requests[0].layers[first_full]
        sequence = layer.sequence
        table = sequence.active_block_table.to(torch.int64)
        dense_batches = []
        for table_row in table:
            blocks = sequence.arena.key_cache.index_select(0, table_row)
            logical = blocks.reshape(
                -1, FORMAL_NUM_KV_HEADS, FORMAL_HEAD_DIM
            )[: sequence.sequence_length]
            dense_batches.append(logical.permute(1, 0, 2).contiguous())
        dense_key = torch.stack(dense_batches, dim=0)
        original_view = layer.keys
        _require(
            isinstance(original_view, Q16KernelPagedTensorView)
            and isinstance(dense_key, torch.Tensor),
            "M9 target precondition drift",
        )

        def descriptor_paged(role: str) -> dict[str, Any]:
            value = layer.keys
            result = {
                "schema_version": "forkaudit-live-kv-representation-v1",
                "logical_slot": f"request0/layer{first_full}/keys",
                "binding_role": role,
                "representation": (
                    "q16-paged-view"
                    if isinstance(value, Q16KernelPagedTensorView)
                    else "dense-tensor"
                ),
            }
            if isinstance(value, torch.Tensor):
                result["tensor_sha256"] = _tensor_digest(value)
            else:
                result.update(
                    {
                        "kind": value.kind,
                        "page_size": FORMAL_PAGE_SIZE,
                        "kv_heads": FORMAL_NUM_KV_HEADS,
                        "head_dim": FORMAL_HEAD_DIM,
                    }
                )
            return result

        def apply_dense(_state: dict[str, Any]) -> AppliedMutation:
            pre = descriptor_paged("original-q16-paged-key-view")
            layer.keys = dense_key
            mutated = descriptor_paged("dense-materialized-key")
            return AppliedMutation(
                undo=lambda: setattr(layer, "keys", original_view),
                verify_restored=lambda: layer.keys is original_view,
                target_binding=_target_binding(
                    mutant_id=mutant_id,
                    case_cell_id=case_cell_id,
                    pre_descriptor=pre,
                    mutated_descriptor=mutated,
                    capture_restored_descriptor=lambda: descriptor_paged(
                        "original-q16-paged-key-view"
                    ),
                ),
            )

        ledger = MultiForkHitLedger(
            plan,
            group.requests[0],
            request_index=0,
            resident_count=2,
            request_policy=group.policy,
            expected_calls_per_layer=1,
            initial_query_tokens=FORMAL_QUERY_TOKENS,
            kernel=kernel,
            strict_position_values=True,
        )
        context.update({"ledger": ledger, "register_backend_ledger": ledger})
        injector = callback_injector(apply_dense)

        def exercise(state: dict[str, Any]) -> dict[str, Any]:
            mark_detector_input(
                {
                    "kind": "live-full-model-forward-ledger",
                    "backend_registered": isinstance(state.get("backend"), str),
                    "request_index": 0,
                    "initial_query_tokens": FORMAL_QUERY_TOKENS,
                    "expected_calls_per_layer": 1,
                    "expected_full_attention_layers": list(
                        state["plan"].full_attention_layer_indices
                    ),
                    "call_count_before": len(state["ledger"].calls),
                    "query_token_ids_sha256": _token_id_sha256(
                        queries[0], expected_shape=(1, FORMAL_QUERY_TOKENS)
                    ),
                }
            )
            original_backend = backbone.config._attn_implementation
            try:
                backbone.config._attn_implementation = state["backend"]
                backbone(
                    input_ids=queries[0],
                    past_key_values=state["group"].requests[0],
                    use_cache=True,
                )
                return _compact_full_forward_mutant_ledger(
                    state["ledger"].verify_complete()
                )
            finally:
                backbone.config._attn_implementation = original_backend

    else:  # pragma: no cover - assignment is frozen above.
        raise ReviewAuditError(f"unknown live mutant {mutant_id}")

    raw_exercise = exercise

    def tracked_exercise(state: dict[str, Any]) -> None:
        _require(
            coverage_state["exercise_started"] is False,
            f"{mutant_id} exercise executed more than once",
        )
        coverage_state["exercise_started"] = True
        completion = raw_exercise(state)
        _require(
            coverage_state["detector_input"] is not None,
            f"{mutant_id} detector path completed without a live input receipt",
        )
        _require(
            isinstance(completion, dict),
            f"{mutant_id} detector path completed without mechanical evidence",
        )
        coverage_state["completion_receipt"] = completion
        coverage_state["detector_path_completed"] = True

    backend_to_unregister = None
    ledger_to_register = context.get("register_backend_ledger")
    if ledger_to_register is not None:
        backend_to_unregister = register_multifork_backend(
            ledger_to_register
        )
        context["backend"] = backend_to_unregister
    if backend_to_unregister is None:
        outcome = (
            run_mutant_case(mutant_id, injector, tracked_exercise, context=context)
            if activate_mutation
            else run_clean_case(tracked_exercise, context=context)
        )
    else:
        with _registered_backend_scope([backend_to_unregister]):
            outcome = (
                run_mutant_case(
                    mutant_id, injector, tracked_exercise, context=context
                )
                if activate_mutation
                else run_clean_case(tracked_exercise, context=context)
            )
    # No subsequent case may observe this mutated or partially advanced
    # cache.  The caller drops the full context and verifies allocator
    # recovery before constructing the next case.
    result = {
        "exercise_mutant_id": mutant_id,
        "exercise_contract_sha256": _mutant_exercise_contract_sha256(mutant_id),
        "exercise_coverage_receipt": {
            "schema_version": "forkaudit-mutant-exercise-coverage-v2",
            "mutant_id": mutant_id,
            "mutation_activated": activate_mutation,
            "exercise_contract_sha256": _mutant_exercise_contract_sha256(
                mutant_id
            ),
            "detector_path": MUTANT_EXERCISE_PATHS[mutant_id],
            "exercise_started": coverage_state["exercise_started"],
            "detector_input": coverage_state["detector_input"],
            "detector_input_sha256": (
                None
                if coverage_state["detector_input"] is None
                else sha256_json(coverage_state["detector_input"])
            ),
            "detector_path_completed": coverage_state[
                "detector_path_completed"
            ],
            "completion_receipt": coverage_state["completion_receipt"],
            "completion_receipt_sha256": (
                None
                if coverage_state["completion_receipt"] is None
                else sha256_json(coverage_state["completion_receipt"])
            ),
            "outcome_classification": outcome.classification.value,
            "observed_gate_id": outcome.observed_gate_id,
        },
        "case_cell_id": case_cell_id,
        "case_isolation": {
            "fresh_document_cache_built": True,
            "fresh_request_cache_built": True,
            "cache_reused_from_prior_case": False,
            "cache_discarded_after_case": True,
        },
        "outcome": outcome.to_dict(),
    }
    if activate_mutation:
        result["mutant_id"] = mutant_id
    return result


def _run_live_fault_campaign(
    *,
    rank: int,
    model: Any,
    backbone: Any,
    plan: Any,
    document: Any,
    queries: Sequence[Any],
    kernel: Any,
    frozen_baseline: Mapping[str, int],
) -> dict[str, Any]:
    """Run clean control and assigned mutants with fresh cache rebuilds."""

    def assert_recovered(label: str) -> dict[str, int]:
        snapshot = _gpu_cleanup()
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(
                snapshot[field] == frozen_baseline[field],
                f"allocator did not recover after {label}",
            )
        return snapshot

    clean_before = assert_recovered("before global clean mutant control")
    clean_context = _make_mutant_live_context(
        mutant_id="clean",
        case_cell_id=f"rank-{rank}-mutant-clean-fresh-N-2",
        model=model,
        backbone=backbone,
        plan=plan,
        document=document,
        queries=queries,
        kernel=kernel,
    )
    clean = run_clean_case(
        lambda state: validate_runtime_kv_ownership(
            state["persistent"],
            state["group"],
            state["plan"],
            require_appended_tail_cow=False,
        ),
        context=clean_context,
    )
    del clean_context
    clean_after = assert_recovered("global clean mutant control")
    clean_case = {
        "case_cell_id": f"rank-{rank}-mutant-global-clean-fresh-N-2",
        "case_isolation": {
            "fresh_document_cache_built": True,
            "fresh_request_cache_built": True,
            "cache_reused_from_prior_case": False,
            "cache_discarded_after_case": True,
        },
        "cleanup_receipt": _make_cleanup_receipt(
            before=clean_before,
            after=clean_after,
            frozen_baseline=frozen_baseline,
        ),
        "outcome": clean.to_dict(),
    }
    rows = {}
    for mutant_id in MUTANT_ASSIGNMENT_BY_RANK[rank]:
        matched_before = assert_recovered(f"before matched clean {mutant_id}")
        matched = _run_one_live_mutant(
            mutant_id,
            rank=rank,
            model=model,
            backbone=backbone,
            plan=plan,
            document=document,
            queries=queries,
            kernel=kernel,
            activate_mutation=False,
        )
        matched_outcome = _parse_campaign_outcome(matched["outcome"])
        matched_passed = _validate_clean_outcome(matched_outcome)
        matched_after = assert_recovered(f"matched clean {mutant_id}")
        matched["cleanup_receipt"] = _make_cleanup_receipt(
            before=matched_before,
            after=matched_after,
            frozen_baseline=frozen_baseline,
        )
        mutant_before = assert_recovered(f"before mutant {mutant_id}")
        row = _run_one_live_mutant(
            mutant_id,
            rank=rank,
            model=model,
            backbone=backbone,
            plan=plan,
            document=document,
            queries=queries,
            kernel=kernel,
            activate_mutation=True,
        )
        row["matched_clean"] = matched
        row["matched_clean_exercise_passed"] = matched_passed
        mutant_after = assert_recovered(f"mutant {mutant_id}")
        row["cleanup_receipt"] = _make_cleanup_receipt(
            before=mutant_before,
            after=mutant_after,
            frozen_baseline=frozen_baseline,
        )
        rows[mutant_id] = row
        del row
    return {
        "assignment": list(MUTANT_ASSIGNMENT_BY_RANK[rank]),
        "clean_case": clean_case,
        "mutants": rows,
    }


def _run_max_n_warmup(
    *,
    rank: int,
    runtime: FormalModelRuntime,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Prime lazy runtime state, then warm every factor arm at a stable baseline."""

    pre_priming = _gpu_cleanup()
    priming_arm_id = ARM_IDS[0]
    priming = _run_clean_memory_cell(
        rank=rank,
        arm_id=priming_arm_id,
        resident_count=max(FORMAL_RESIDENT_COUNTS),
        kv_policy=KV_POLICIES[0],
        gdn_base_policy=GDN_BASE_POLICIES[0],
        model=runtime.model,
        backbone=runtime.backbone,
        plan=runtime.plan,
        document=runtime.document,
        queries=runtime.queries,
        kernel=runtime.kernel,
    )
    _require(
        priming["memory_cell"]["primary_memory_endpoint_eligible"] is True,
        "discarded priming cell did not exercise the clean production path",
    )
    del priming
    initial = _gpu_cleanup()
    rows: list[dict[str, Any]] = []
    for arm_index, arm_id in enumerate(ARM_IDS):
        kv_policy = KV_POLICIES[arm_index // len(GDN_BASE_POLICIES)]
        gdn_policy = GDN_BASE_POLICIES[arm_index % len(GDN_BASE_POLICIES)]
        before = _gpu_cleanup()
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(
                before[field] == initial[field],
                f"warmup arm baseline drift before {arm_id}: {field}",
            )
        discarded = _run_clean_memory_cell(
            rank=rank,
            arm_id=arm_id,
            resident_count=max(FORMAL_RESIDENT_COUNTS),
            kv_policy=kv_policy,
            gdn_base_policy=gdn_policy,
            model=runtime.model,
            backbone=runtime.backbone,
            plan=runtime.plan,
            document=runtime.document,
            queries=runtime.queries,
            kernel=runtime.kernel,
            expected_frozen_baseline=initial,
        )
        _require(
            discarded["memory_cell"]["primary_memory_endpoint_eligible"] is True,
            "warmup cell did not exercise the clean production path",
        )
        del discarded
        after = _gpu_cleanup()
        for field in ("current_allocated_bytes", "current_reserved_bytes"):
            _require(
                after[field] == initial[field],
                f"warmup arm did not recover after {arm_id}: {field}",
            )
        rows.append(
            {
                "arm_id": arm_id,
                "resident_count": max(FORMAL_RESIDENT_COUNTS),
                "before": before,
                "after_cleanup": after,
                "discarded_non_endpoint": True,
                "witness_hooks_enabled": False,
            }
        )
    frozen = _gpu_cleanup()
    _require(
        all(
            frozen[field] == initial[field]
            for field in ("current_allocated_bytes", "current_reserved_bytes")
        ),
        "post-warmup model/query baseline drift",
    )
    return (
        {
            "schema_version": "qcomem-forkaudit-max-n-warmup-v2",
            "pre_priming_allocator_baseline": pre_priming,
            "post_priming_allocator_baseline": initial,
            "priming_arm_id": priming_arm_id,
            "one_discarded_priming_cell_before_baseline_freeze": True,
            "priming_excluded_from_memory_matrix": True,
            "resident_count": max(FORMAL_RESIDENT_COUNTS),
            "arm_order": list(ARM_IDS),
            "same_model_document_and_query_bank_as_formal_cells": True,
            "candidate_outputs_discarded": True,
            "excluded_from_memory_matrix": True,
            "all_four_factor_arms_warmed": True,
            "rows": rows,
            "frozen_model_query_baseline": frozen,
        },
        frozen,
    )


def _merge_formal_cell_results(
    memory: Mapping[str, Any],
    witness: Mapping[str, Any],
    *,
    arm_id: str,
    kv_policy: str,
    gdn_base_policy: str,
) -> dict[str, Any]:
    memory_before = memory["source_physical_document_sha256_before"]
    memory_after = memory["source_physical_document_sha256_after"]
    witness_before = witness["source_physical_document_sha256_before"]
    witness_after = witness["source_physical_document_sha256_after"]
    memory_payload = memory["source_physical_payload_sha256"]
    witness_payload = witness["source_physical_payload_sha256"]
    _require(
        memory_before == memory_after == witness_before == witness_after,
        "memory/witness rebuild physical source digests differ",
    )
    _require(
        memory_payload == witness_payload,
        "memory/witness rebuild physical payload digests differ",
    )
    return {
        "arm_id": arm_id,
        "kv_policy": kv_policy,
        "gdn_base_policy": gdn_base_policy,
        "memory_cell": dict(memory["memory_cell"]),
        "witness_cell": dict(witness["witness_cell"]),
        "source_physical_document_sha256_before": dict(memory_before),
        "source_physical_document_sha256_after": dict(memory_after),
        "memory_source_physical_document_sha256_before": dict(memory_before),
        "memory_source_physical_document_sha256_after": dict(memory_after),
        "witness_source_physical_document_sha256_before": dict(witness_before),
        "witness_source_physical_document_sha256_after": dict(witness_after),
        "source_digest_scope": "complete-physical-document-blocks-including-tail-padding",
        "source_physical_payload_sha256": dict(memory_payload),
        "memory_source_physical_payload_sha256": dict(memory_payload),
        "witness_source_physical_payload_sha256": dict(witness_payload),
        "source_payload_digest_scope": (
            "key-value-document-block-bytes-including-tail-padding-"
            "excluding-arena-capacity-metadata"
        ),
        "memory_kernel_ledgers": list(memory["memory_kernel_ledgers"]),
        "witness_kernel_ledgers": list(witness["witness_kernel_ledgers"]),
        "semantics": list(memory["semantics"]),
        "witness_semantics": list(witness["semantics"]),
    }


def _run_formal_factorial_cells(
    *,
    artifact_root: Path,
    run_id: str,
    rank: int,
    runtime: FormalModelRuntime,
    oracle_selection: Mapping[str, Any],
    frozen_baseline: Mapping[str, int],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    factorial: list[dict[str, Any]] = []
    oracle_refs: list[dict[str, Any]] = []
    for resident_count in FORMAL_RESIDENT_COUNTS:
        cells: list[dict[str, Any]] = []
        for arm_index, arm_id in enumerate(ARM_IDS):
            kv_policy = KV_POLICIES[arm_index // len(GDN_BASE_POLICIES)]
            gdn_policy = GDN_BASE_POLICIES[
                arm_index % len(GDN_BASE_POLICIES)
            ]
            memory = _execute_cell_with_cleanup(
                lambda: _run_clean_memory_cell(
                    rank=rank,
                    arm_id=arm_id,
                    resident_count=resident_count,
                    kv_policy=kv_policy,
                    gdn_base_policy=gdn_policy,
                    model=runtime.model,
                    backbone=runtime.backbone,
                    plan=runtime.plan,
                    document=runtime.document,
                    queries=runtime.queries,
                    kernel=runtime.kernel,
                    expected_frozen_baseline=frozen_baseline,
                ),
                cell_role_key="memory_cell",
                frozen_baseline=frozen_baseline,
                label=f"rank {rank} N={resident_count} {arm_id} memory cell",
            )
            witness = _execute_cell_with_cleanup(
                lambda: _run_ownership_witness_cell(
                    artifact_root=artifact_root,
                    run_id=run_id,
                    rank=rank,
                    arm_id=arm_id,
                    resident_count=resident_count,
                    kv_policy=kv_policy,
                    gdn_base_policy=gdn_policy,
                    model=runtime.model,
                    backbone=runtime.backbone,
                    plan=runtime.plan,
                    document=runtime.document,
                    queries=runtime.queries,
                    kernel=runtime.kernel,
                    oracle_selection=oracle_selection,
                ),
                cell_role_key="witness_cell",
                frozen_baseline=frozen_baseline,
                label=f"rank {rank} N={resident_count} {arm_id} witness cell",
            )
            oracle_ref = witness.pop("oracle_raw_artifact")
            if oracle_ref is not None:
                oracle_refs.append(dict(oracle_ref))
            cells.append(
                _merge_formal_cell_results(
                    memory,
                    witness,
                    arm_id=arm_id,
                    kv_policy=kv_policy,
                    gdn_base_policy=gdn_policy,
                )
            )
            del memory, witness
        factorial.append({"resident_count": resident_count, "cells": cells})
    _require(
        len(oracle_refs) == 1,
        "formal rank must emit exactly one preregistered oracle sample",
    )
    return factorial, oracle_refs[0]


def _make_producer_self_replay_receipt(
    *,
    factorial_exact: bool,
    oracle_passed: bool,
    mutant_rows_replayed: int,
    matched_clean_rows_replayed: int,
    memory_matrix_rows_replayed: int,
    detached_sidecar_references_replayed: int,
) -> dict[str, Any]:
    _require(
        isinstance(factorial_exact, bool)
        and isinstance(oracle_passed, bool)
        and all(
            _is_int(value) and value >= 0
            for value in (
                mutant_rows_replayed,
                matched_clean_rows_replayed,
                memory_matrix_rows_replayed,
                detached_sidecar_references_replayed,
            )
        ),
        "producer self-replay inputs are malformed",
    )
    return {
        "schema_version": "qcomem-forkaudit-producer-self-replay-v1",
        "factorial_exact": factorial_exact,
        "oracle_passed": oracle_passed,
        "mutant_rows_replayed": mutant_rows_replayed,
        "matched_clean_rows_replayed": matched_clean_rows_replayed,
        "memory_matrix_rows_replayed": memory_matrix_rows_replayed,
        "detached_sidecar_references_replayed": (
            detached_sidecar_references_replayed
        ),
        "schema_replay_completed_before_atomic_shard_commit": True,
    }


def _validate_producer_self_replay_receipt(
    value: Any, *, expected: Mapping[str, Any]
) -> dict[str, Any]:
    _require(
        isinstance(value, dict)
        and set(value) == set(expected)
        and value == expected,
        "producer self-replay receipt differs from aggregate blind replay",
    )
    return dict(value)


@dataclass
class RankArtifactStaging:
    artifact_root: Path
    staging_root: Path
    rank: int
    committed: bool = False
    final_output: Path | None = None

    @property
    def staged_rank_directory(self) -> Path:
        return self.staging_root / f"rank-{self.rank}"

    @property
    def final_rank_directory(self) -> Path:
        return self.artifact_root / f"rank-{self.rank}"

    def commit_shard(self, output: Path, value: Mapping[str, Any]) -> None:
        _require(not self.committed, "rank artifact staging committed twice")
        _require(
            self.staged_rank_directory.is_dir(),
            "rank staging contains no detached sidecar directory",
        )
        _require(
            not self.final_rank_directory.exists() and not output.exists(),
            "rank final artifact target already exists",
        )
        pending_shard = self.staging_root / "pending-shard.json"
        _write_json(pending_shard, value)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.staged_rank_directory.replace(self.final_rank_directory)
            self.committed = True
            pending_shard.replace(output)
            self.final_output = output
        except BaseException:
            if self.committed and self.final_rank_directory.is_dir():
                shutil.rmtree(self.final_rank_directory)
            self.committed = False
            try:
                pending_shard.unlink(missing_ok=True)
            except OSError:
                pass
            raise

    def rollback(self) -> None:
        if self.final_output is not None and self.final_output.exists():
            self.final_output.unlink()
        self.final_output = None
        if self.committed and self.final_rank_directory.is_dir():
            shutil.rmtree(self.final_rank_directory)
            self.committed = False
        if self.staging_root.exists():
            shutil.rmtree(self.staging_root)


@contextmanager
def _rank_artifact_staging(*, artifact_root: Path, rank: int):
    artifact_root.mkdir(parents=True, exist_ok=True)
    final_rank = artifact_root / f"rank-{rank}"
    _require(not final_rank.exists(), "formal rank sidecar directory already exists")
    staging_root = Path(
        tempfile.mkdtemp(prefix=f".forkaudit-rank-{rank}-", dir=artifact_root)
    )
    staging = RankArtifactStaging(
        artifact_root=artifact_root,
        staging_root=staging_root,
        rank=rank,
    )
    try:
        yield staging
    except BaseException:
        staging.rollback()
        raise
    else:
        _require(staging.committed, "rank staging exited without shard commit")
        try:
            if staging.staging_root.exists():
                shutil.rmtree(staging.staging_root)
        except BaseException:
            staging.rollback()
            raise


def _self_validate_formal_shard(
    shard: Mapping[str, Any],
    *,
    inputs: FormalInputBundle,
    artifact_root: Path,
) -> dict[str, Any]:
    """Blindly replay one producer result before the atomic shard commit."""

    rank = shard["rank"]
    run_id = _validate_shard_common(
        shard,
        rank=rank,
        static_sha256=inputs.static_sha256,
        expected_identity=inputs.static_replay["frozen_identity"],
        expected_query_bank=inputs.static_replay["frozen_query_banks"][rank],
        expected_run_id_receipt=inputs.run_id_receipt,
        expected_run_id_receipt_sha256=shard["run_id_receipt_sha256"],
        expected_gpu_assignment_receipt=inputs.gpu_assignment_receipt,
        expected_gpu_assignment_receipt_raw_sha256=(
            inputs.gpu_assignment_receipt_raw_sha256
        ),
        expected_private_model_view_manifest=(
            inputs.private_model_view_manifest
        ),
        expected_private_model_view_manifest_raw_sha256=(
            inputs.private_model_view_manifest_raw_sha256
        ),
        expected_model_load_authority=inputs.model_load_authority,
        expected_model_load_authority_raw_sha256=(
            inputs.model_load_authority_raw_sha256
        ),
    )
    (
        factorial_exact,
        bindings,
        _semantics,
        _source,
        oracle_contexts,
        frozen_baseline,
        memory_rows,
    ) = _validate_factorial(
        shard,
        root=artifact_root,
        rank=rank,
        run_id=run_id,
        expected_query_bank=inputs.static_replay["frozen_query_banks"][rank],
    )
    selection = inputs.static_replay["oracle_selection_plan"][rank]
    context = dict(oracle_contexts[selection["arm_id"]])
    selected_ledger = context["witness_ledgers"][selection["request_index"]]
    call_offset = (
        selection["round_index"] * len(FORMAL_FULL_LAYERS)
        + FORMAL_FULL_LAYERS.index(selection["layer_index"])
    )
    context["ledger_call"] = selected_ledger["calls"][call_offset]
    context["selected_ledger"] = selected_ledger
    context["source_physical_payload_sha256"] = context[
        "source_physical_payload_sha256_by_layer"
    ][str(selection["layer_index"])]
    oracle, oracle_bindings = _recompute_oracle(
        shard["oracle_raw_artifact"],
        root=artifact_root,
        rank=rank,
        source_object=shard["data_usage"]["source_object"],
        expected_selection=selection,
        observer_context=context,
        expected_run_id=run_id,
    )
    _clean, mutants, matched = _validate_fault_campaign(
        shard,
        rank=rank,
        seen_case_ids=set(),
        expected_query_sha256=inputs.static_replay["frozen_query_banks"][rank][
            "rows"
        ][0]["query_token_ids_sha256"],
        expected_frozen_baseline=frozen_baseline,
    )
    return _make_producer_self_replay_receipt(
        factorial_exact=factorial_exact,
        oracle_passed=oracle["passed"],
        mutant_rows_replayed=len(mutants),
        matched_clean_rows_replayed=len(matched),
        memory_matrix_rows_replayed=len(memory_rows),
        detached_sidecar_references_replayed=(
            len(bindings) + len(oracle_bindings)
        ),
    )


def _run_formal_gpu_shard_impl(args: argparse.Namespace) -> dict[str, Any]:
    """Execute one real rank.  Main keeps this unreachable until release."""

    import torch

    inputs = _load_formal_input_bundle(args)
    artifact_root = args.artifact_root.resolve()
    output = args.output.resolve()
    _require(
        output
        == artifact_root
        / "shards"
        / f"forkaudit-shard-{args.rank}.json",
        "formal shard output path differs from the frozen rank-local layout",
    )
    _require(not output.exists(), "formal shard output must not pre-exist")
    runtime: FormalModelRuntime | None = None
    try:
        with _rank_artifact_staging(
            artifact_root=artifact_root, rank=args.rank
        ) as staging:
            runtime = _load_formal_model_runtime(args, inputs)
            with torch.inference_mode():
                warmup, frozen_baseline = _run_max_n_warmup(
                    rank=args.rank, runtime=runtime
                )
                expected_query_bank = inputs.static_replay[
                    "frozen_query_banks"
                ][args.rank]
                live_input_lifetime_receipts = [
                    _capture_live_input_lifetime_receipt(
                        runtime=runtime,
                        expected_query_bank=expected_query_bank,
                        frozen_baseline=frozen_baseline,
                        capture_point=LIVE_INPUT_CAPTURE_POINTS[0],
                    )
                ]
                factorial, oracle_ref = _run_formal_factorial_cells(
                    artifact_root=staging.staging_root,
                    run_id=args.run_id,
                    rank=args.rank,
                    runtime=runtime,
                    oracle_selection=inputs.static_replay[
                        "oracle_selection_plan"
                    ][args.rank],
                    frozen_baseline=frozen_baseline,
                )
                live_input_lifetime_receipts.append(
                    _capture_live_input_lifetime_receipt(
                        runtime=runtime,
                        expected_query_bank=expected_query_bank,
                        frozen_baseline=frozen_baseline,
                        capture_point=LIVE_INPUT_CAPTURE_POINTS[1],
                    )
                )
                fault_campaign = _run_live_fault_campaign(
                    rank=args.rank,
                    model=runtime.model,
                    backbone=runtime.backbone,
                    plan=runtime.plan,
                    document=runtime.document,
                    queries=runtime.queries,
                    kernel=runtime.kernel,
                    frozen_baseline=frozen_baseline,
                )
                live_input_lifetime_receipts.append(
                    _capture_live_input_lifetime_receipt(
                        runtime=runtime,
                        expected_query_bank=expected_query_bank,
                        frozen_baseline=frozen_baseline,
                        capture_point=LIVE_INPUT_CAPTURE_POINTS[2],
                    )
                )
            shard = {
                "schema_version": SHARD_SCHEMA_VERSION,
                "protocol": PROTOCOL,
                "rank": args.rank,
                "world_size": FORMAL_WORLD_SIZE,
                "artifact_mode": "formal_gpu",
                "status": "completed_formal_gpu_shard",
                "static_artifact_sha256": inputs.static_sha256,
                "protocol_config": formal_protocol_config(),
                "protocol_config_sha256": sha256_json(formal_protocol_config()),
                "frozen_identity": inputs.static_replay["frozen_identity"],
                "run_id": args.run_id,
                "run_id_receipt": inputs.run_id_receipt,
                "run_id_receipt_sha256": args.expected_run_id_receipt_sha256,
                "gpu_assignment_receipt": inputs.gpu_assignment_receipt,
                "gpu_assignment_receipt_raw_sha256": (
                    inputs.gpu_assignment_receipt_raw_sha256
                ),
                "private_model_view_manifest": (
                    inputs.private_model_view_manifest
                ),
                "private_model_view_manifest_raw_sha256": (
                    inputs.private_model_view_manifest_raw_sha256
                ),
                "model_load_authority": inputs.model_load_authority,
                "model_load_authority_raw_sha256": (
                    inputs.model_load_authority_raw_sha256
                ),
                "data_usage": inputs.data_usage,
                "input_rebuild_receipt": inputs.input_rebuild_receipt,
                "hardware_audit": runtime.hardware_audit,
                "model_runtime_audit": runtime.model_runtime_audit,
                "warmup_receipt": warmup,
                "live_input_lifetime_receipts": live_input_lifetime_receipts,
                "factorial": factorial,
                "oracle_raw_artifact": oracle_ref,
                "fault_campaign": fault_campaign,
            }
            self_replay = _self_validate_formal_shard(
                shard, inputs=inputs, artifact_root=staging.staging_root
            )
            shard["producer_self_replay"] = self_replay
            staging.commit_shard(output, shard)
            return shard
    finally:
        runtime = None
        gc.collect()
        if torch.cuda.is_initialized():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except BaseException:
                # A primary exception remains authoritative; process teardown
                # is the final cleanup if the CUDA runtime itself has failed.
                pass


def run_formal_gpu_shard(args: argparse.Namespace) -> dict[str, Any]:
    _require(
        GPU_LOOP_IMPLEMENTED,
        "formal GPU shard remains release-gated pending independent audit/release",
    )
    return _run_formal_gpu_shard_impl(args)


@dataclass(frozen=True)
class LiveKVIdentityGuard:
    """In-memory pointer-free tokenization baseline for one request group."""

    guard_id: str
    secret: bytes
    resident_count: int
    kv_policy: str
    sequence_tokens: Mapping[tuple[int, int], str]
    arena_tokens: Mapping[tuple[int, int], str]


def _opaque_token(secret: bytes, *parts: Any) -> str:
    payload = "\0".join(str(part) for part in parts).encode("utf-8")
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def capture_live_kv_identity_guard(group: Any, plan: Any) -> LiveKVIdentityGuard:
    _require(group.resident_count >= 1, "live KV guard requires requests")
    secret = secrets.token_bytes(32)
    sequences = {}
    arenas = {}
    for layer_index in plan.full_attention_layer_indices:
        for request_index, request in enumerate(group.requests):
            sequence = request.layers[layer_index].sequence
            coordinate = (int(layer_index), request_index)
            sequences[coordinate] = _opaque_token(
                secret, "sequence-object", id(sequence)
            )
            arenas[coordinate] = _opaque_token(
                secret, "arena-object", id(sequence.arena)
            )
    return LiveKVIdentityGuard(
        guard_id=secrets.token_hex(16),
        secret=secret,
        resident_count=group.resident_count,
        kv_policy=group.policy,
        sequence_tokens=sequences,
        arena_tokens=arenas,
    )


def capture_live_kv_witness(
    persistent: Any,
    group: Any,
    plan: Any,
    guard: LiveKVIdentityGuard,
    *,
    phase: str,
    capture_id: str,
    completed_request_indices: Sequence[int],
) -> dict[str, Any]:
    _require(group.resident_count == guard.resident_count, "KV guard N drift")
    _require(group.policy == guard.kv_policy, "KV guard policy drift")
    require_tail = phase == PHASE_POST_GENERATION
    live_gate = validate_runtime_kv_ownership(
        persistent,
        group,
        plan,
        require_appended_tail_cow=require_tail,
    )
    rows = []
    completed = set(completed_request_indices)
    for layer_index in plan.full_attention_layer_indices:
        for request_index, request in enumerate(group.requests):
            sequence = request.layers[layer_index].sequence
            arena = sequence.arena
            coordinate = (int(layer_index), request_index)
            sequence_token = _opaque_token(
                guard.secret,
                "sequence-object",
                id(sequence),
            )
            arena_token = _opaque_token(
                guard.secret, "arena-object", id(arena)
            )
            _require(sequence_token == guard.sequence_tokens[coordinate], "request/sequence binding changed after setup")
            _require(arena_token == guard.arena_tokens[coordinate], "request/arena binding changed after setup")

            def block_token(physical: int) -> str:
                return _opaque_token(
                    guard.secret,
                    "physical-block",
                    arena_token,
                    int(physical),
                )

            reservations = [
                block_token(int(item))
                for item in sequence.reservations.reshape(-1).tolist()
            ]
            documents = [
                block_token(int(item))
                for item in arena.document_block_table.reshape(-1).tolist()
            ]
            active: list[str] = []
            if sequence.appended_tokens > 0:
                first_private_logical = arena.document_blocks_per_sequence
                if arena.document_length % arena.page_size:
                    first_private_logical -= 1
                for batch_index in range(arena.batch_size):
                    for logical_index in range(
                        first_private_logical, sequence.logical_block_count
                    ):
                        active.append(
                            block_token(
                                int(
                                    sequence.active_block_table[
                                        batch_index, logical_index
                                    ]
                                )
                            )
                        )
            _require(arena.batch_size == 1, "formal KV witness assumes batch size one")
            rows.append(
                {
                    "layer_index": int(layer_index),
                    "request_index": request_index,
                    "arena_id": arena_token,
                    "sequence_id": sequence_token,
                    "reservation_ids": reservations,
                    "document_block_ids": documents,
                    "active_private_ids": active,
                    "appended_tokens": int(sequence.appended_tokens),
                    "tail_detached": bool(sequence._tail_detached[0]),
                }
            )
    return {
        "schema_version": KV_WITNESS_SCHEMA_VERSION,
        "phase": phase,
        "capture_id": capture_id,
        "kv_guard_id": guard.guard_id,
        "kv_policy": group.policy,
        "resident_count": group.resident_count,
        "completed_request_indices": list(completed_request_indices),
        "live_gate_receipt": {
            "function": "validate_runtime_kv_ownership",
            "called": True,
            "gate_ids": list(live_gate["gate_ids"]),
            "require_appended_tail_cow": require_tail,
        },
        "rows": rows,
    }


def _validate_data_usage(
    shard: Mapping[str, Any],
    rank: int,
    *,
    expected_query_bank: Mapping[str, Any],
) -> dict[str, Any]:
    usage = shard.get("data_usage")
    required = {
        "dataset",
        "split",
        "pg19_train_only",
        "longbench_consumed",
        "validation_consumed",
        "test_v2_consumed",
        "source_id",
        "source_object",
        "book_index",
        "window_index",
        "document_start_token",
        "document_end_token_exclusive",
        "document_length",
        "document_token_ids_sha256",
        "document_input_receipt",
        "query_bank_input_receipt",
    }
    _require(
        isinstance(usage, dict) and set(usage) == required,
        "data usage schema drift",
    )
    _require(usage.get("dataset") == "pg19" and usage.get("split") == "train", "only PG19 train is allowed")
    _require(usage.get("pg19_train_only") is True, "PG19 train-only gate failed")
    for field in ("longbench_consumed", "validation_consumed", "test_v2_consumed"):
        _require(usage.get(field) is False, f"forbidden data consumed: {field}")
    source = usage.get("source_object")
    _require(isinstance(source, str) and _SOURCE_RE.fullmatch(source) is not None, "source is not a PG19 train object")
    lowered = source.lower().replace("-", "_")
    _require("validation" not in lowered and "test_v2" not in lowered and "/test/" not in lowered, "validation/test-v2 source rejected")
    for field in (
        "book_index",
        "window_index",
        "document_start_token",
        "document_end_token_exclusive",
        "document_length",
    ):
        _require(
            _is_int(usage.get(field)),
            f"live data-usage {field} must be a non-bool integer",
        )
    _require(usage.get("book_index") == rank, "rank/book assignment drift")
    expected_document = {
        "source_id": expected_query_bank["source_id"],
        "source_object": expected_query_bank["source_object"],
        "book_index": expected_query_bank["book_index"],
        "window_index": expected_query_bank["window_index"],
        "document_start_token": expected_query_bank["document_start_token"],
        "document_end_token_exclusive": expected_query_bank[
            "document_end_token_exclusive"
        ],
        "document_length": FORMAL_DOCUMENT_TOKENS,
        "document_token_ids_sha256": expected_query_bank[
            "document_token_ids_sha256"
        ],
    }
    _require(
        {field: usage[field] for field in expected_document}
        == expected_document,
        "live document coordinate/digest differs from raw-bound RR2 input",
    )
    document_receipt = usage["document_input_receipt"]
    _require(
        document_receipt
        == {
            "capture_point": "immediately-before-persistent-document-prefill",
            "dtype": "torch.int64",
            "shape": [1, FORMAL_DOCUMENT_TOKENS],
            "sha256": expected_document["document_token_ids_sha256"],
            "rebuilt_from_raw_bound_rr2_manifest": True,
        },
        "live document tensor receipt drift",
    )
    query_receipt = usage["query_bank_input_receipt"]
    expected_query_rows = [
        {
            "request_index": row["request_index"],
            "sha256": row["query_token_ids_sha256"],
        }
        for row in expected_query_bank["rows"]
    ]
    _require(
        query_receipt
        == {
            "capture_point": "immediately-before-formal-factorial-cells",
            "dtype": "torch.int64",
            "shape_per_query": [1, FORMAL_QUERY_TOKENS],
            "count": max(FORMAL_RESIDENT_COUNTS),
            "rows": expected_query_rows,
            "rebuilt_from_raw_bound_rr2_manifest": True,
        },
        "live query-bank tensor receipt differs from raw-bound RR2 input",
    )
    return {
        "document": expected_document,
        "query_rows": expected_query_rows,
    }


def _validate_formal_producer_receipts(
    shard: Mapping[str, Any],
    *,
    rank: int,
    run_id: str,
    static_sha256: str,
    expected_identity: Mapping[str, Any],
    expected_query_bank: Mapping[str, Any],
    expected_run_id_receipt: Mapping[str, Any],
    expected_run_id_receipt_sha256: str,
    expected_gpu_assignment_receipt: Mapping[str, Any],
    expected_gpu_assignment_receipt_raw_sha256: str,
    expected_private_model_view_manifest: Mapping[str, Any],
    expected_private_model_view_manifest_raw_sha256: str,
    expected_model_load_authority: Mapping[str, Any],
    expected_model_load_authority_raw_sha256: str,
) -> dict[str, int]:
    receipt_sha = _require_sha256(
        shard.get("run_id_receipt_sha256"), "shard run-ID receipt SHA"
    )
    run_receipt = shard.get("run_id_receipt")
    _require(
        isinstance(run_receipt, dict)
        and run_receipt == expected_run_id_receipt
        and receipt_sha
        == _require_sha256(
            expected_run_id_receipt_sha256,
            "expected shared run-ID receipt SHA",
        ),
        "shard/shared run-ID receipt binding drift",
    )
    _validate_run_id_receipt(
        canonical_json_bytes(run_receipt),
        expected_sha256=receipt_sha,
        run_id=run_id,
        static_artifact_sha256=static_sha256,
        protocol_manifest_sha256=expected_identity[
            "protocol_manifest_sha256"
        ],
    )
    _require(
        shard.get("gpu_assignment_receipt") == expected_gpu_assignment_receipt
        and shard.get("gpu_assignment_receipt_raw_sha256")
        == _require_sha256(
            expected_gpu_assignment_receipt_raw_sha256,
            "expected GPU-assignment receipt raw SHA",
        ),
        "shard GPU-assignment receipt binding drift",
    )
    _require(
        shard.get("private_model_view_manifest")
        == expected_private_model_view_manifest
        and shard.get("private_model_view_manifest_raw_sha256")
        == expected_private_model_view_manifest_raw_sha256,
        "shard private model-view manifest binding drift",
    )
    _require(
        shard.get("model_load_authority") == expected_model_load_authority
        and shard.get("model_load_authority_raw_sha256")
        == expected_model_load_authority_raw_sha256,
        "shard ModelLoadLease authority binding drift",
    )
    expected_gpu_row = expected_gpu_assignment_receipt["rows"][rank]

    rebuild = shard.get("input_rebuild_receipt")
    rebuild_fields = {
        "schema_version",
        "rr2_input_manifest_raw_sha256",
        "source_rebuilt_manifest_byte_identical",
        "pg19_windows_sha256",
        "rank",
        "document_token_ids_sha256",
        "query_token_ids_sha256",
        "model_manifest_rows",
        "model_artifact_audit",
        "model_weight_audit",
        "code_ledger_entry_count",
        "protocol_manifest_raw_sha256",
        "private_model_view_manifest_raw_sha256",
        "model_load_authority_raw_sha256",
        "cuda_initialized_during_rebuild",
    }
    _require(
        isinstance(rebuild, dict)
        and set(rebuild) == rebuild_fields
        and rebuild["schema_version"]
        == "qcomem-forkaudit-live-input-rebuild-v1"
        and rebuild["rr2_input_manifest_raw_sha256"]
        == expected_identity["pg19_input_manifest_sha256"]
        and rebuild["source_rebuilt_manifest_byte_identical"] is True
        and rebuild["pg19_windows_sha256"] == FORMAL_RR2_WINDOWS_SHA256
        and _is_int(rebuild["rank"])
        and rebuild["rank"] == rank
        and rebuild["document_token_ids_sha256"]
        == expected_query_bank["document_token_ids_sha256"]
        and rebuild["query_token_ids_sha256"]
        == [row["query_token_ids_sha256"] for row in expected_query_bank["rows"]]
        and _is_int(rebuild["code_ledger_entry_count"])
        and rebuild["code_ledger_entry_count"] > 0
        and rebuild["protocol_manifest_raw_sha256"]
        == expected_identity["protocol_manifest_sha256"]
        and rebuild["private_model_view_manifest_raw_sha256"]
        == _require_sha256(
            expected_private_model_view_manifest_raw_sha256,
            "expected private model-view manifest raw SHA",
        )
        and rebuild["model_load_authority_raw_sha256"]
        == _require_sha256(
            expected_model_load_authority_raw_sha256,
            "expected ModelLoadLease authority raw SHA",
        )
        and rebuild["cuda_initialized_during_rebuild"] is False,
        "formal live input-rebuild receipt drift",
    )
    manifest_rows = rebuild["model_manifest_rows"]
    _require(
        isinstance(manifest_rows, list)
        and [row.get("logical_name") for row in manifest_rows]
        == [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        ],
        "formal model manifest row order drift",
    )
    digest = hashlib.sha256()
    for row in manifest_rows:
        _require(
            isinstance(row, dict)
            and set(row) == {"logical_name", "sha256", "bytes"}
            and _is_int(row["bytes"])
            and row["bytes"] > 0,
            "formal model manifest row schema drift",
        )
        _require_sha256(row["sha256"], "formal model manifest component")
        digest.update(
            f"{row['logical_name']}\0{row['sha256']}\0{row['bytes']}\n".encode(
                "utf-8"
            )
        )
    _require(
        digest.hexdigest() == expected_identity["model_manifest_sha256"],
        "formal model manifest rows differ from frozen identity",
    )
    artifact_audit = rebuild["model_artifact_audit"]
    weight_audit = rebuild["model_weight_audit"]
    private_rows = expected_private_model_view_manifest["rows"]
    private_by_name = {row["relative_path"]: row for row in private_rows}
    expected_artifact_rows = [
        {
            "logical_name": row["relative_path"],
            "sha256": row["declared_sha256"],
            "bytes": row["bytes"],
        }
        for row in private_rows
        if "model_artifact" in row["ledger_roles"]
    ]
    expected_weight_rows = [
        {
            "logical_name": row["relative_path"],
            "sha256": row["declared_sha256"],
            "bytes": row["bytes"],
        }
        for row in private_rows
        if "model_weight" in row["ledger_roles"]
    ]
    _require(
        isinstance(artifact_audit, dict)
        and set(artifact_audit) == {"entry_count", "entries_sha256"}
        and _is_int(artifact_audit["entry_count"])
        and artifact_audit["entry_count"] == len(expected_artifact_rows)
        and artifact_audit["entries_sha256"]
        == sha256_json(expected_artifact_rows),
        "formal model artifact audit drift",
    )
    _require(
        isinstance(weight_audit, dict)
        and set(weight_audit)
        == {
            "entry_count",
            "indexed_files_sha256",
            "per_rank_full_weight_rehash_performed",
        }
        and _is_int(weight_audit["entry_count"])
        and weight_audit["entry_count"] == 14
        and weight_audit["indexed_files_sha256"]
        == sha256_json(expected_weight_rows)
        and weight_audit["per_rank_full_weight_rehash_performed"] is False,
        "formal model weight index/size audit drift",
    )
    _require_sha256(weight_audit["indexed_files_sha256"], "weight index/size audit")
    model_manifest_names = (
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
    )
    _require(
        all(name in private_by_name for name in model_manifest_names),
        "private model view omits a frozen model manifest component",
    )
    _require(
        manifest_rows
        == [
            {
                "logical_name": name,
                "sha256": private_by_name[name]["declared_sha256"],
                "bytes": private_by_name[name]["bytes"],
            }
            for name in model_manifest_names
        ],
        "formal model manifest rows differ from the private view",
    )

    hardware = shard.get("hardware_audit")
    hardware_fields = {
        "schema_version",
        "cuda_visible_devices",
        "physical_visible_index",
        "process_local_device",
        "visible_device_count",
        "uuid",
        "name",
        "total_memory_mib",
        "compute_capability",
        "bf16_supported",
        "torch_cuda_version",
    }
    _require(
        isinstance(hardware, dict)
        and set(hardware) == hardware_fields
        and hardware["schema_version"]
        == "qcomem-forkaudit-local-gpu-audit-v1"
        and isinstance(hardware["cuda_visible_devices"], str)
        and re.fullmatch(r"GPU-[0-9a-fA-F-]+", hardware["cuda_visible_devices"])
        is not None
        and _is_int(hardware["physical_visible_index"])
        and hardware["physical_visible_index"]
        == expected_gpu_row["visible_index"]
        and hardware["cuda_visible_devices"]
        == expected_gpu_row["uuid"]
        and hardware["process_local_device"] == "cuda:0"
        and _is_int(hardware["visible_device_count"])
        and hardware["visible_device_count"] == 1
        and isinstance(hardware["uuid"], str)
        and hardware["uuid"] == expected_gpu_row["uuid"]
        and hardware["name"] == expected_gpu_row["name"]
        and _is_int(hardware["total_memory_mib"])
        and hardware["total_memory_mib"]
        == expected_gpu_row["total_memory_mib"]
        and hardware["compute_capability"]
        == expected_gpu_row["compute_capability"]
        and all(_is_int(item) for item in hardware["compute_capability"])
        and hardware["bf16_supported"] == expected_gpu_row["bf16_supported"]
        and isinstance(hardware["torch_cuda_version"], str)
        and bool(hardware["torch_cuda_version"]),
        "formal local H20 audit drift",
    )

    model_audit = shard.get("model_runtime_audit")
    model_audit_fields = {
        "schema_version",
        "model_id",
        "model_revision",
        "local_files_only",
        "trust_remote_code",
        "dtype",
        "device",
        "visual_branch_removed_for_direct_text_backbone",
        "model_manifest_sha256",
        "model_manifest_rows",
        "model_artifact_audit_before_after_equal",
        "model_weight_index_and_size_before_after_equal",
        "model_weight_full_hash_per_rank",
        "private_model_view_manifest_raw_sha256",
        "model_load_authority_raw_sha256",
        "model_load_rank_stat_envelopes",
        "geometry",
        "functional_stack_plan",
        "kernel_environment",
        "kernel_descriptor",
    }
    geometry = {
        "observed": {
            "num_query_heads": FORMAL_NUM_QUERY_HEADS,
            "num_key_value_heads": FORMAL_NUM_KV_HEADS,
            "num_key_value_groups": FORMAL_GQA_GROUPS,
            "head_dim": FORMAL_HEAD_DIM,
            "full_attention_layers": len(FORMAL_FULL_LAYERS),
        },
        "num_hidden_layers": 40,
        "full_attention_layer_indices": list(FORMAL_FULL_LAYERS),
        "linear_attention_layer_count": len(FORMAL_LINEAR_LAYERS),
        "matches_frozen_geometry": True,
    }
    functional = model_audit.get("functional_stack_plan") if isinstance(model_audit, dict) else None
    from qcomem_vllm_paged_kernel import AUDITED_PACKAGES
    from qcomem_forkaudit_model_load_lease import (
        ModelLoadLeaseError,
        validate_rank_stat_envelope,
    )

    expected_environment = {
        "expected_versions": dict(AUDITED_PACKAGES),
        "observed_versions": dict(AUDITED_PACKAGES),
        "matches_frozen_environment": True,
        "mismatches": {},
        "kernel_entrypoint": (
            "vllm.v1.attention.ops.triton_unified_attention.unified_attention"
        ),
        "kernel_mode": FORMAL_KERNEL_MODE,
    }
    _require(
        isinstance(model_audit, dict)
        and set(model_audit) == model_audit_fields
        and model_audit["schema_version"]
        == "qcomem-forkaudit-model-runtime-audit-v1"
        and model_audit["model_id"] == FORMAL_MODEL_ID
        and model_audit["model_revision"] == FORMAL_MODEL_REVISION
        and model_audit["local_files_only"] is True
        and model_audit["trust_remote_code"] is False
        and model_audit["dtype"] == "torch.bfloat16"
        and model_audit["device"] == "cuda:0"
        and model_audit["visual_branch_removed_for_direct_text_backbone"] is True
        and model_audit["model_manifest_sha256"]
        == expected_identity["model_manifest_sha256"]
        and model_audit["model_manifest_rows"] == manifest_rows
        and model_audit["model_artifact_audit_before_after_equal"] is True
        and model_audit["model_weight_index_and_size_before_after_equal"]
        is True
        and model_audit["model_weight_full_hash_per_rank"] is False
        and model_audit["private_model_view_manifest_raw_sha256"]
        == expected_private_model_view_manifest_raw_sha256
        and model_audit["model_load_authority_raw_sha256"]
        == expected_model_load_authority_raw_sha256
        and model_audit["geometry"] == geometry
        and isinstance(functional, dict)
        and set(functional)
        == {
            "total_layers",
            "linear_layer_indices",
            "linear_layer_count",
            "full_attention_layer_indices",
            "full_attention_layer_count",
            "layer_types",
            "model_type",
            "kernel_mode",
            "production_ttft_optimization_claim_allowed",
        }
        and _is_int(functional["total_layers"])
        and functional["total_layers"] == 40
        and functional["linear_layer_indices"] == list(FORMAL_LINEAR_LAYERS)
        and _is_int(functional["linear_layer_count"])
        and functional["linear_layer_count"] == len(FORMAL_LINEAR_LAYERS)
        and functional["full_attention_layer_indices"] == list(FORMAL_FULL_LAYERS)
        and _is_int(functional["full_attention_layer_count"])
        and functional["full_attention_layer_count"] == len(FORMAL_FULL_LAYERS)
        and functional["layer_types"]
        == [
            "full_attention" if index in FORMAL_FULL_LAYERS else "linear_attention"
            for index in range(40)
        ]
        and functional["model_type"] == FORMAL_MODEL_TYPE
        and functional["kernel_mode"] == FORMAL_KERNEL_MODE
        and functional["production_ttft_optimization_claim_allowed"] is False
        and model_audit["kernel_environment"] == expected_environment
        and model_audit["kernel_descriptor"]
        == {
            "module": FORMAL_KERNEL_DESCRIPTOR[0],
            "qualname": FORMAL_KERNEL_DESCRIPTOR[1],
            "signature": FORMAL_KERNEL_DESCRIPTOR[2],
        },
        "formal model runtime audit drift",
    )
    envelopes = model_audit["model_load_rank_stat_envelopes"]
    _require(
        isinstance(envelopes, list)
        and len(envelopes) == 2
        and [row.get("capture_point") for row in envelopes]
        == [
            "immediately-before-from-pretrained",
            "immediately-after-from-pretrained",
        ],
        "rank ModelLoadLease stat envelope coverage drift",
    )
    try:
        validated_envelopes = [
            validate_rank_stat_envelope(
                row, authority=expected_model_load_authority
            )
            for row in envelopes
        ]
    except ModelLoadLeaseError as exc:
        raise ReviewAuditError("rank ModelLoadLease stat envelope rejected") from exc
    _require(
        all(row["rank"] == rank for row in validated_envelopes),
        "rank ModelLoadLease stat envelope rank drift",
    )

    warmup = shard.get("warmup_receipt")
    warmup_fields = {
        "schema_version",
        "pre_priming_allocator_baseline",
        "post_priming_allocator_baseline",
        "priming_arm_id",
        "one_discarded_priming_cell_before_baseline_freeze",
        "priming_excluded_from_memory_matrix",
        "resident_count",
        "arm_order",
        "same_model_document_and_query_bank_as_formal_cells",
        "candidate_outputs_discarded",
        "excluded_from_memory_matrix",
        "all_four_factor_arms_warmed",
        "rows",
        "frozen_model_query_baseline",
    }
    _require(
        isinstance(warmup, dict)
        and set(warmup) == warmup_fields
        and warmup["schema_version"] == "qcomem-forkaudit-max-n-warmup-v2"
        and warmup["priming_arm_id"] == ARM_IDS[0]
        and warmup["one_discarded_priming_cell_before_baseline_freeze"] is True
        and warmup["priming_excluded_from_memory_matrix"] is True
        and _is_int(warmup["resident_count"])
        and warmup["resident_count"] == max(FORMAL_RESIDENT_COUNTS)
        and warmup["arm_order"] == list(ARM_IDS)
        and warmup["same_model_document_and_query_bank_as_formal_cells"] is True
        and warmup["candidate_outputs_discarded"] is True
        and warmup["excluded_from_memory_matrix"] is True
        and warmup["all_four_factor_arms_warmed"] is True,
        "formal max-N warmup receipt drift",
    )
    frozen = _validate_allocator_snapshot(
        warmup["frozen_model_query_baseline"], label="warmup frozen baseline"
    )
    _validate_allocator_snapshot(
        warmup["pre_priming_allocator_baseline"], label="pre-priming baseline"
    )
    post_priming = _validate_allocator_snapshot(
        warmup["post_priming_allocator_baseline"], label="post-priming baseline"
    )
    _require(
        post_priming == frozen,
        "post-priming baseline differs from the frozen measurement baseline",
    )
    rows = warmup["rows"]
    _require(
        isinstance(rows, list)
        and len(rows) == len(ARM_IDS)
        and [row.get("arm_id") for row in rows] == list(ARM_IDS),
        "formal warmup arm coverage drift",
    )
    for row in rows:
        _require(
            isinstance(row, dict)
            and set(row)
            == {
                "arm_id",
                "resident_count",
                "before",
                "after_cleanup",
                "discarded_non_endpoint",
                "witness_hooks_enabled",
            }
            and _is_int(row["resident_count"])
            and row["resident_count"] == max(FORMAL_RESIDENT_COUNTS)
            and row["discarded_non_endpoint"] is True
            and row["witness_hooks_enabled"] is False,
            "formal warmup row drift",
        )
        before = _validate_allocator_snapshot(row["before"], label="warmup before")
        after = _validate_allocator_snapshot(
            row["after_cleanup"], label="warmup after"
        )
        _require(
            before == after == frozen,
            "formal warmup did not recover the exact frozen baseline",
        )
    _validate_live_input_lifetime_receipts(
        shard.get("live_input_lifetime_receipts"),
        expected_query_bank=expected_query_bank,
        frozen_baseline=frozen,
    )
    return frozen


def _validate_cell_separation(cell: Mapping[str, Any], *, rank: int, resident_count: int, arm_id: str) -> dict[str, Any]:
    memory = cell.get("memory_cell")
    witness = cell.get("witness_cell")
    _require(isinstance(memory, dict) and isinstance(witness, dict), "memory/witness cells missing")
    for row, role in ((memory, "formal_memory"), (witness, "ownership_witness")):
        _require(row.get("cell_role") == role, f"{role} cell role drift")
        _require(
            _is_int(row.get("rank"))
            and _is_int(row.get("resident_count"))
            and row.get("rank") == rank
            and row.get("resident_count") == resident_count,
            f"{role} cell binding drift",
        )
        _require(row.get("arm_id") == arm_id, f"{role} arm binding drift")
        _require(isinstance(row.get("cell_id"), str) and bool(row["cell_id"]), f"{role} cell_id missing")
    _require(memory["cell_id"] != witness["cell_id"], "witness and memory endpoints share a cell")
    _require(memory.get("request_guard_created") is False, "memory cell retained a request guard")
    _require(memory.get("witness_capture_executed") is False, "memory cell executed witness hashing")
    _require(memory.get("primary_memory_endpoint_eligible") is True, "memory cell not endpoint eligible")
    _require(witness.get("request_guard_created") is True, "witness cell missed request guard")
    _require(witness.get("witness_capture_executed") is True, "witness cell missed capture")
    _require(witness.get("primary_memory_endpoint_eligible") is False, "witness cell leaked into memory endpoint")
    _require(witness.get("rebuilt_persistent_cache") is True and witness.get("rebuilt_request_group") is True, "witness cell was not rebuilt independently")
    memory_baseline = _validate_cleanup_receipt(
        memory.get("cleanup_receipt"), label="formal memory cell"
    )
    witness_baseline = _validate_cleanup_receipt(
        witness.get("cleanup_receipt"), label="ownership witness cell"
    )
    _require(
        memory_baseline == witness_baseline,
        "memory/witness cleanup baselines differ",
    )
    result = _validate_memory_receipt(memory)
    _require(
        memory["allocator_receipt"]["baseline"] == memory_baseline,
        "memory endpoint baseline differs from lifecycle baseline",
    )
    return {**result, "frozen_baseline": memory_baseline}


def _validate_sha_map(value: Any, *, label: str, keys: Sequence[int]) -> dict[str, str]:
    _require(isinstance(value, dict), f"{label} missing")
    expected = {str(index) for index in keys}
    _require(set(value) == expected, f"{label} layer set drift")
    return {key: _require_sha256(item, f"{label}[{key}]") for key, item in value.items()}


def _validate_semantics(value: Any, *, resident_count: int) -> list[dict[str, Any]]:
    _require(isinstance(value, list) and len(value) == resident_count, "semantic request cardinality drift")
    result = []
    for request_index, row in enumerate(value):
        _require(
            isinstance(row, dict)
            and _is_int(row.get("request_index"))
            and row.get("request_index") == request_index,
            "semantic request order drift",
        )
        query_sha = _require_sha256(row.get("query_token_ids_sha256"), "query digest")
        tokens = row.get("generated_token_ids")
        _require(isinstance(tokens, list) and len(tokens) == FORMAL_GENERATION_STEPS, "token trajectory length drift")
        _require(all(_is_int(token) and token >= 0 for token in tokens), "generated token type drift")
        logits = row.get("full_vocab_step_logit_sha256")
        _require(isinstance(logits, list) and len(logits) == FORMAL_GENERATION_STEPS, "full-logit trajectory length drift")
        logits = [_require_sha256(item, "full-logit digest") for item in logits]
        logical = _validate_sha_map(row.get("logical_kv_sha256"), label="logical KV digest", keys=FORMAL_FULL_LAYERS)
        final_gdn = _require_sha256(row.get("final_gdn_sha256"), "final GDN digest")
        result.append(
            {
                "request_index": request_index,
                "query_token_ids_sha256": query_sha,
                "generated_token_ids": tokens,
                "full_vocab_step_logit_sha256": logits,
                "logical_kv_sha256": logical,
                "final_gdn_sha256": final_gdn,
            }
        )
    return result


def _validate_kv_witness(
    value: Any,
    *,
    phase: str,
    kv_policy: str,
    resident_count: int,
    capture_id: str,
) -> dict[str, Any]:
    _require(isinstance(value, dict), "KV witness missing")
    _require(
        set(value)
        == {
            "schema_version",
            "phase",
            "capture_id",
            "kv_guard_id",
            "kv_policy",
            "resident_count",
            "completed_request_indices",
            "live_gate_receipt",
            "rows",
        },
        "KV witness top-level schema drift",
    )
    _require(value.get("schema_version") == KV_WITNESS_SCHEMA_VERSION, "KV witness schema drift")
    _require(value.get("phase") == phase and value.get("capture_id") == capture_id, "KV phase/capture binding drift")
    _require(
        value.get("kv_policy") == kv_policy
        and _is_int(value.get("resident_count"))
        and value.get("resident_count") == resident_count,
        "KV policy/N binding drift",
    )
    guard_id = value.get("kv_guard_id")
    _require(
        isinstance(guard_id, str)
        and re.fullmatch(r"[0-9a-f]{32}", guard_id) is not None,
        "KV guard ID missing/drifted",
    )
    completed = value.get("completed_request_indices")
    expected_completed = (
        [] if phase == PHASE_SETUP_PRE_TRANSITION
        else [0] if phase == PHASE_POST_TRANSITION
        else list(range(resident_count))
    )
    _require(completed == expected_completed, "KV completed-set drift")
    receipt = value.get("live_gate_receipt")
    _require(isinstance(receipt, dict), "live KV gate receipt missing")
    _require(receipt.get("function") == "validate_runtime_kv_ownership" and receipt.get("called") is True, "live KV ownership gate not called")
    _require(
        receipt.get("gate_ids")
        == ["KV_SEQUENCE_ID", "KV_RESERVATION_DISJOINT", "KV_TAIL_COW", "KV_ACTIVE_BLOCK_OWNERSHIP"],
        "live KV gate set drift",
    )
    _require(receipt.get("require_appended_tail_cow") is (phase == PHASE_POST_GENERATION), "live KV tail-COW mode drift")
    rows = value.get("rows")
    _require(isinstance(rows, list) and len(rows) == len(FORMAL_FULL_LAYERS) * resident_count, "KV witness row cardinality drift")
    expected_order = [(layer, request) for layer in FORMAL_FULL_LAYERS for request in range(resident_count)]
    observed_order = []
    sequence_ids: set[str] = set()
    by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in FORMAL_FULL_LAYERS}
    for row in rows:
        _require(
            isinstance(row, dict)
            and set(row)
            == {
                "layer_index",
                "request_index",
                "arena_id",
                "sequence_id",
                "reservation_ids",
                "document_block_ids",
                "active_private_ids",
                "appended_tokens",
                "tail_detached",
            },
            "KV witness row schema drift",
        )
        layer = row.get("layer_index")
        request = row.get("request_index")
        _require(
            _is_int(layer)
            and layer in FORMAL_FULL_LAYERS
            and _is_int(request)
            and 0 <= request < resident_count,
            "KV witness layer/request coordinate drift",
        )
        observed_order.append((layer, request))
        _require_sha256(row.get("arena_id"), "normalized arena identity token")
        sequence = row.get("sequence_id")
        _require_sha256(sequence, "normalized sequence identity token")
        _require(sequence not in sequence_ids, "sequence identity reused")
        sequence_ids.add(sequence)
        reservations = row.get("reservation_ids")
        active = row.get("active_private_ids")
        documents = row.get("document_block_ids")
        _require(
            isinstance(reservations, list)
            and len(reservations) == 2
            and all(_is_int(item) and item >= 0 for item in reservations)
            and len(set(reservations)) == 2,
            "reservation witness drift",
        )
        _require(
            isinstance(documents, list)
            and len(documents)
            == math.ceil(FORMAL_DOCUMENT_TOKENS / FORMAL_PAGE_SIZE)
            and all(_is_int(item) and item >= 0 for item in documents),
            "document block witness drift",
        )
        _require(len(set(documents)) == len(documents) and not (set(reservations) & set(documents)), "document/reservation overlap")
        _require(
            isinstance(active, list)
            and all(_is_int(item) and item >= 0 for item in active)
            and len(set(active)) == len(active)
            and set(active) <= set(reservations),
            "active private ownership drift",
        )
        is_completed = request in expected_completed
        expected_appended = (
            0 if not is_completed else FORMAL_QUERY_TOKENS if phase == PHASE_POST_TRANSITION else FORMAL_FINAL_APPENDED_TOKENS
        )
        _require(
            _is_int(row.get("appended_tokens"))
            and row.get("appended_tokens") == expected_appended,
            "KV appended-token drift",
        )
        _require(row.get("tail_detached") is is_completed, "KV tail-COW witness drift")
        _require(len(active) == (2 if is_completed else 0), "active private block count drift")
        by_layer[layer].append(row)
    _require(observed_order == expected_order, "KV witness row order drift")
    for layer, layer_rows in by_layer.items():
        _require(len(layer_rows) == resident_count, f"KV layer {layer} missing requests")
        arena_ids = [row["arena_id"] for row in layer_rows]
        if kv_policy == SHARED_REUSE:
            _require(len(set(arena_ids)) == 1, "shared KV arm did not share its arena")
            _require(
                all(
                    row["document_block_ids"]
                    == layer_rows[0]["document_block_ids"]
                    for row in layer_rows[1:]
                ),
                "shared KV peers did not retain one immutable document table",
            )
        else:
            _require(len(set(arena_ids)) == resident_count, "fresh KV arm reused an arena")
        for left_index, left in enumerate(layer_rows):
            for right in layer_rows[left_index + 1 :]:
                if left["arena_id"] == right["arena_id"]:
                    _require(not (set(left["reservation_ids"]) & set(right["reservation_ids"])), "peer reservations overlap")
                    _require(not (set(left["active_private_ids"]) & set(right["active_private_ids"])), "peer active blocks overlap")
    normalized_rows = [
        {
            "layer_index": row["layer_index"],
            "request_index": row["request_index"],
            "arena_id": row["arena_id"],
            "sequence_id": row["sequence_id"],
            "reservation_ids": list(row["reservation_ids"]),
            "document_block_ids": list(row["document_block_ids"]),
            "active_private_ids": list(row["active_private_ids"]),
            "appended_tokens": row["appended_tokens"],
            "tail_detached": row["tail_detached"],
        }
        for row in rows
    ]
    return {
        "passed": True,
        "row_count": len(rows),
        "completed_request_indices": expected_completed,
        "kv_guard_id": guard_id,
        "rows": normalized_rows,
    }


def _normalized_kv_factor_projection(
    phases: Sequence[Mapping[str, Any]],
    *,
    kv_policy: str,
    resident_count: int,
) -> dict[str, Any]:
    """Canonicalize opaque KV tokens while preserving every alias relation."""

    token_maps: dict[str, dict[str, str]] = {
        "arena": {},
        "sequence": {},
        "block": {},
    }

    def normalize(kind: str, token: str) -> str:
        mapping = token_maps[kind]
        if token not in mapping:
            mapping[token] = f"{kind}-{len(mapping):04d}"
        return mapping[token]

    projected_phases = []
    for phase in phases:
        projected_rows = []
        for row in phase["rows"]:
            projected_rows.append(
                {
                    "layer_index": row["layer_index"],
                    "request_index": row["request_index"],
                    "arena_class": normalize("arena", row["arena_id"]),
                    "sequence_class": normalize(
                        "sequence", row["sequence_id"]
                    ),
                    "reservation_classes": [
                        normalize("block", token)
                        for token in row["reservation_ids"]
                    ],
                    "document_block_classes": [
                        normalize("block", token)
                        for token in row["document_block_ids"]
                    ],
                    "active_private_classes": [
                        normalize("block", token)
                        for token in row["active_private_ids"]
                    ],
                    "appended_tokens": row["appended_tokens"],
                    "tail_detached": row["tail_detached"],
                }
            )
        projected_phases.append(
            {
                "completed_request_indices": list(
                    phase["completed_request_indices"]
                ),
                "rows": projected_rows,
            }
        )
    return {
        "schema_version": "qcomem-kv-factor-projection-v1",
        "kv_policy": kv_policy,
        "resident_count": resident_count,
        "phases": projected_phases,
    }


def _normalized_gdn_factor_projection(
    phases: Sequence[Mapping[str, Any]],
    *,
    gdn_policy: str,
    resident_count: int,
) -> dict[str, Any]:
    """Project raw storage/binding rows without per-cell guard/HMAC values."""

    storage_row_fields = {
        "owner_kind",
        "request_index",
        "layer_index",
        "state_family",
        "state_index",
        "shape",
        "stride",
        "storage_offset",
        "dtype",
        "device",
        "storage_nbytes",
        "tensor_nbytes",
        "byte_start",
        "byte_end_exclusive",
        "content_sha256",
        "storage_id",
    }
    binding_row_fields = {
        "request_index",
        "layer_index",
        "state_family",
        "state_index",
        "expected_relation",
        "baseline_binding_token",
        "observed_binding_token",
        "baseline_storage_token",
        "observed_storage_token",
    }
    projected_phases = []
    for phase in phases:
        storage = phase["storage_witness"]
        binding = phase["binding_witness"]
        storage_rows = storage["rows"]
        binding_rows = binding["rows"]
        _require(
            all(
                isinstance(row, dict) and set(row) == storage_row_fields
                for row in storage_rows
            ),
            "GDN storage projection row schema drift",
        )
        _require(
            all(
                isinstance(row, dict) and set(row) == binding_row_fields
                for row in binding_rows
            ),
            "GDN binding projection row schema drift",
        )
        # Snapshot-local storage IDs are deterministically allocated in owner /
        # coordinate order by the witness module.  Keeping them proves exact
        # alias/disjoint equivalence classes across the independent KV arms.
        projected_storage_rows = [dict(row) for row in storage_rows]
        projected_binding_rows = [
            {
                "request_index": row["request_index"],
                "layer_index": row["layer_index"],
                "state_family": row["state_family"],
                "state_index": row["state_index"],
                "expected_relation": row["expected_relation"],
                "binding_equal_to_setup": (
                    row["baseline_binding_token"]
                    == row["observed_binding_token"]
                ),
                "storage_equal_to_setup": (
                    row["baseline_storage_token"]
                    == row["observed_storage_token"]
                ),
            }
            for row in binding_rows
        ]
        persistent_ids = {
            row["storage_id"]
            for row in storage_rows
            if row["owner_kind"] == "persistent"
        }
        request_unique_storage = {
            row["storage_id"]: row["storage_nbytes"]
            for row in storage_rows
            if row["owner_kind"] == "request"
            and row["storage_id"] not in persistent_ids
        }
        projected_phases.append(
            {
                "phase": phase["phase"],
                "completed_request_indices": list(
                    storage["completed_request_indices"]
                ),
                "storage_rows": projected_storage_rows,
                "binding_rows": projected_binding_rows,
                "request_owned_unique_storage_nbytes": sum(
                    request_unique_storage.values()
                ),
            }
        )
    return {
        "schema_version": "qcomem-gdn-factor-projection-v1",
        "gdn_policy": gdn_policy,
        "resident_count": resident_count,
        "phases": projected_phases,
    }


def _validate_timeline_manifest(
    reference: Any,
    *,
    root: Path,
    rank: int,
    run_id: str,
    resident_count: int,
    kv_policy: str,
    gdn_base_policy: str,
    witness_cell_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded = load_json_artifact(reference, root=root, label="GDN timeline manifest")
    manifest = loaded.payload
    _require(isinstance(manifest, dict), "GDN timeline manifest must be object")
    _require(manifest.get("schema_version") == "qcomem-gdn-external-timeline-manifest-v1", "external timeline manifest schema drift")
    binding = manifest.get("binding")
    expected_binding = {
        "rank": rank,
        "run_id": run_id,
        "cell_id": witness_cell_id,
        "resident_count": resident_count,
        "kv_policy": kv_policy,
        "gdn_base_policy": gdn_base_policy,
        "gdn_policy": GDN_POLICY_TO_WITNESS[gdn_base_policy],
    }
    _require(binding == expected_binding, "external timeline binding drift")
    phase_refs = manifest.get("phase_artifacts")
    _require(isinstance(phase_refs, list) and len(phase_refs) == 3, "timeline requires setup/first-transition/generation artifacts")
    expected_phases = (
        PHASE_SETUP_PRE_TRANSITION,
        PHASE_POST_TRANSITION,
        PHASE_POST_GENERATION,
    )
    phase_rows = []
    kv_phase_replays: list[dict[str, Any]] = []
    kv_guard_ids: list[str] = []
    artifact_bindings = [loaded.binding]
    seen_paths: set[str] = set()
    for phase_name, phase_ref in zip(expected_phases, phase_refs):
        phase_loaded = load_json_artifact(
            phase_ref,
            root=root,
            label=f"GDN/KV phase {phase_name}",
        )
        artifact_bindings.append(phase_loaded.binding)
        _require(phase_loaded.binding["relative_path"] not in seen_paths, "phase artifact path reused")
        seen_paths.add(phase_loaded.binding["relative_path"])
        wrapper = phase_loaded.payload
        _require(isinstance(wrapper, dict), "phase artifact must be object")
        _require(wrapper.get("schema_version") == PHASE_ARTIFACT_SCHEMA_VERSION, "phase artifact schema drift")
        phase_binding = wrapper.get("binding")
        _require(isinstance(phase_binding, dict), "phase binding missing")
        _require(
            phase_binding
            == {
                **expected_binding,
                "phase": phase_name,
            },
            "phase artifact binding drift",
        )
        gdn_phase = wrapper.get("gdn_phase_witness")
        _require(isinstance(gdn_phase, dict), "unified GDN phase witness missing")
        capture_id = gdn_phase.get("capture_id")
        _require(isinstance(capture_id, str) and bool(capture_id), "GDN capture ID missing")
        kv_replay = _validate_kv_witness(
            wrapper.get("kv_ownership_witness"),
            phase=phase_name,
            kv_policy=kv_policy,
            resident_count=resident_count,
            capture_id=capture_id,
        )
        kv_phase_replays.append(kv_replay)
        kv_guard_ids.append(kv_replay["kv_guard_id"])
        phase_rows.append(gdn_phase)
    # Arena, sequence, reservation, and immutable document identities are one
    # lifecycle chain.  Independent per-phase validity is insufficient.
    setup_rows = kv_phase_replays[0]["rows"]
    _require(
        len(set(kv_guard_ids)) == 1,
        "KV lifecycle phases do not share one construction guard",
    )
    for later in kv_phase_replays[1:]:
        _require(len(later["rows"]) == len(setup_rows), "KV lifecycle row count drift")
        for setup_row, observed_row in zip(setup_rows, later["rows"]):
            for field in (
                "layer_index",
                "request_index",
                "arena_id",
                "sequence_id",
                "reservation_ids",
                "document_block_ids",
            ):
                _require(
                    observed_row[field] == setup_row[field],
                    f"KV lifecycle changed {field} across phases",
                )
    timeline = {
        "schema_version": TIMELINE_SCHEMA_VERSION,
        "run_id": run_id,
        "cell_id": witness_cell_id,
        "kv_policy": kv_policy,
        "gdn_policy": GDN_POLICY_TO_WITNESS[gdn_base_policy],
        "group_gdn_base_policy": gdn_base_policy,
        "resident_count": resident_count,
        "layer_indices": list(FORMAL_LINEAR_LAYERS),
        "state_index": 0,
        "phases": phase_rows,
    }
    # The module independently replays storage rows, request binding tokens,
    # monotone completion, and shared guard/capture identities.
    replay = replay_gdn_storage_timeline(json.loads(json.dumps(timeline)))
    _require(replay.get("passed") is True, "GDN timeline replay failed")
    summaries = replay.get("phase_summaries")
    _require(
        isinstance(summaries, list)
        and [row.get("completed_request_indices") for row in summaries]
        == [[], [0], list(range(resident_count))],
        "formal timeline must witness setup -> request-0 -> all requests",
    )
    replay = dict(replay)
    replay["normalized_gdn_factor_projection"] = (
        _normalized_gdn_factor_projection(
            phase_rows,
            gdn_policy=GDN_POLICY_TO_WITNESS[gdn_base_policy],
            resident_count=resident_count,
        )
    )
    replay["normalized_kv_factor_projection"] = (
        _normalized_kv_factor_projection(
            kv_phase_replays,
            kv_policy=kv_policy,
            resident_count=resident_count,
        )
    )
    return replay, artifact_bindings


def _validate_allocator_snapshot(value: Any, *, label: str) -> dict[str, int]:
    _require(isinstance(value, dict), f"{label} allocator snapshot missing")
    fields = (
        "current_allocated_bytes",
        "current_reserved_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
    )
    _require(set(value) == set(fields), f"{label} allocator snapshot schema drift")
    result = {}
    for field in fields:
        item = value[field]
        _require(_is_int(item) and item >= 0, f"{label} {field} drift")
        result[field] = item
    _require(result["peak_allocated_bytes"] >= result["current_allocated_bytes"], f"{label} allocated peak drift")
    _require(result["peak_reserved_bytes"] >= result["current_reserved_bytes"], f"{label} reserved peak drift")
    _require(
        result["current_reserved_bytes"] >= result["current_allocated_bytes"]
        and result["peak_reserved_bytes"] >= result["peak_allocated_bytes"],
        f"{label} CUDA reserved/allocated ordering drift",
    )
    return result


def _validate_cleanup_receipt(value: Any, *, label: str) -> dict[str, int]:
    required = {
        "schema_version",
        "before_cell",
        "after_cleanup",
        "frozen_model_query_baseline",
        "explicit_python_references_dropped_on_return",
        "gc_collect_completed",
        "cuda_empty_cache_completed",
        "cuda_synchronize_completed",
        "current_allocated_and_reserved_exactly_recovered",
    }
    _require(
        isinstance(value, dict) and set(value) == required,
        f"{label} cleanup schema drift",
    )
    _require(
        value["schema_version"] == "qcomem-cell-cleanup-receipt-v1",
        f"{label} cleanup version drift",
    )
    for flag in required - {
        "schema_version",
        "before_cell",
        "after_cleanup",
        "frozen_model_query_baseline",
    }:
        _require(value[flag] is True, f"{label} cleanup flag failed: {flag}")
    before = _validate_allocator_snapshot(
        value["before_cell"], label=f"{label} before"
    )
    after = _validate_allocator_snapshot(
        value["after_cleanup"], label=f"{label} after"
    )
    frozen = _validate_allocator_snapshot(
        value["frozen_model_query_baseline"],
        label=f"{label} frozen baseline",
    )
    for field in ("current_allocated_bytes", "current_reserved_bytes"):
        _require(
            before[field] == after[field] == frozen[field],
            f"{label} allocator baseline did not exactly recover",
        )
    return frozen


def _validate_resident_storage_breakdown(
    value: Any,
    *,
    resident_count: int,
    kv_policy: str,
) -> dict[str, Any]:
    _require(isinstance(value, dict), "resident storage breakdown missing")
    required = {
        "protocol",
        "policy",
        "resident_count",
        "simultaneous_lifetime",
        "full_attention_layer_count",
        "source_private_reservation_is_common_pack_capacity",
        "active_private_payload_is_subset_not_additive",
        "fresh_duplicate_pool_is_separate_from_source",
        "totals",
        "layers",
    }
    _require(set(value) == required, "resident storage breakdown schema drift")
    _require(
        value["protocol"] == MULTIFORK_PROTOCOL
        and value["policy"] == kv_policy
        and _is_int(value["resident_count"])
        and value["resident_count"] == resident_count
        and value["simultaneous_lifetime"] is True
        and _is_int(value["full_attention_layer_count"])
        and value["full_attention_layer_count"] == len(FORMAL_FULL_LAYERS),
        "resident storage protocol/policy/N drift",
    )
    _require(
        value["source_private_reservation_is_common_pack_capacity"] is True
        and value["active_private_payload_is_subset_not_additive"] is True
        and value["fresh_duplicate_pool_is_separate_from_source"]
        is (kv_policy == FRESH_CONTROL),
        "resident storage interpretation flags drift",
    )
    layers = value["layers"]
    _require(
        isinstance(layers, list)
        and [row.get("layer_idx") for row in layers] == list(FORMAL_FULL_LAYERS),
        "resident storage layer order drift",
    )
    accumulated: dict[str, int] = {}
    normalized_layers = []
    layer_fields = {
        "layer_idx",
        "resident_count",
        "block_bytes",
        "valid_document_payload_nbytes",
        "source_document_allocated_nbytes",
        "source_document_padding_nbytes",
        "source_private_reservation_nbytes",
        "source_total_arena_allocated_nbytes",
        "fresh_duplicate_document_allocated_nbytes",
        "fresh_duplicate_document_padding_nbytes",
        "fresh_duplicate_private_reservation_nbytes",
        "active_request_private_payload_nbytes",
        "active_request_private_allocated_page_nbytes",
        "active_request_private_blocks",
        "request_private_reserved_unused_nbytes",
        "active_request_appended_tokens_sum",
        "active_request_detached_tail_tokens_sum",
        "partial_tail_staging_copy_nbytes",
        "request_block_table_accelerator_nbytes",
        "source_document_table_accelerator_nbytes",
        "fresh_document_table_accelerator_nbytes",
        "source_cpu_reservation_metadata_nbytes",
        "fresh_cpu_reservation_metadata_nbytes",
        "physical_document_block_copy_nbytes_including_padding",
    }
    counted_names = {
        "active_request_private_blocks",
        "active_request_appended_tokens_sum",
        "active_request_detached_tail_tokens_sum",
        "physical_document_block_copy_nbytes_including_padding",
    }
    total_fields = {
        field
        for field in layer_fields
        if field.endswith("_nbytes") or field in counted_names
    }
    for row in layers:
        _require(
            isinstance(row, dict) and set(row) == layer_fields,
            "resident storage layer row schema drift",
        )
        _require(
            _is_int(row["layer_idx"])
            and row["layer_idx"] in FORMAL_FULL_LAYERS,
            "resident storage layer index drift",
        )
        for key, item in row.items():
            if key == "layer_idx":
                continue
            _require(_is_int(item) and item >= 0, f"resident storage {key} drift")
            if key.endswith("_nbytes") or key in counted_names:
                accumulated[key] = accumulated.get(key, 0) + item
        _require(
            _is_int(row.get("resident_count"))
            and row.get("resident_count") == resident_count,
            "storage layer N drift",
        )
        source_document = row["source_document_allocated_nbytes"]
        source_padding = row["source_document_padding_nbytes"]
        source_private = row["source_private_reservation_nbytes"]
        _require(
            row["valid_document_payload_nbytes"] + source_padding == source_document
            and row["source_total_arena_allocated_nbytes"]
            == source_document + source_private,
            "source arena storage formula drift",
        )
        expected_source_private = (
            resident_count
            * FORMAL_PRIVATE_BLOCKS_PER_REQUEST
            * FORMAL_BLOCK_NBYTES
        )
        expected_active_private_payload = (
            2
            * resident_count
            * (
                FORMAL_DOCUMENT_TOKENS % FORMAL_PAGE_SIZE
                + FORMAL_FINAL_APPENDED_TOKENS
            )
            * FORMAL_NUM_KV_HEADS
            * FORMAL_HEAD_DIM
            * FORMAL_ELEMENT_BYTES
        )
        expected_active_pages = expected_source_private
        _require(
            row["block_bytes"] == FORMAL_BLOCK_NBYTES
            and row["valid_document_payload_nbytes"]
            == FORMAL_DOCUMENT_PAYLOAD_NBYTES
            and source_document == FORMAL_DOCUMENT_ALLOCATED_NBYTES
            and source_padding == FORMAL_DOCUMENT_PADDING_NBYTES
            and source_private == expected_source_private
            and row["source_total_arena_allocated_nbytes"]
            == FORMAL_DOCUMENT_ALLOCATED_NBYTES + expected_source_private,
            "formal Q16 source geometry bytes drift",
        )
        expected_fresh_document = resident_count * source_document if kv_policy == FRESH_CONTROL else 0
        expected_fresh_padding = resident_count * source_padding if kv_policy == FRESH_CONTROL else 0
        expected_fresh_private = source_private if kv_policy == FRESH_CONTROL else 0
        _require(
            row["fresh_duplicate_document_allocated_nbytes"] == expected_fresh_document
            and row["fresh_duplicate_document_padding_nbytes"] == expected_fresh_padding
            and row["fresh_duplicate_private_reservation_nbytes"] == expected_fresh_private
            and row["physical_document_block_copy_nbytes_including_padding"]
            == expected_fresh_document,
            "fresh/shared physical storage formula drift",
        )
        _require(
            row["active_request_private_payload_nbytes"]
            <= row["active_request_private_allocated_page_nbytes"]
            <= source_private
            and row["request_private_reserved_unused_nbytes"]
            == source_private - row["active_request_private_allocated_page_nbytes"],
            "active private-page storage formula drift",
        )
        _require(
            row["active_request_private_payload_nbytes"]
            == expected_active_private_payload
            and row["active_request_private_allocated_page_nbytes"]
            == expected_active_pages
            and row["request_private_reserved_unused_nbytes"] == 0
            and row["partial_tail_staging_copy_nbytes"]
            == resident_count * FORMAL_PARTIAL_TAIL_COPY_NBYTES
            and row["request_block_table_accelerator_nbytes"]
            == resident_count
            * (FORMAL_DOCUMENT_BLOCKS + FORMAL_PRIVATE_BLOCKS_PER_REQUEST)
            * 4
            and row["source_document_table_accelerator_nbytes"]
            == FORMAL_DOCUMENT_BLOCKS * 4
            and row["fresh_document_table_accelerator_nbytes"]
            == (
                resident_count * FORMAL_DOCUMENT_BLOCKS * 4
                if kv_policy == FRESH_CONTROL
                else 0
            )
            and row["source_cpu_reservation_metadata_nbytes"]
            == resident_count * FORMAL_PRIVATE_BLOCKS_PER_REQUEST * 8
            and row["fresh_cpu_reservation_metadata_nbytes"]
            == (
                resident_count * FORMAL_PRIVATE_BLOCKS_PER_REQUEST * 8
                if kv_policy == FRESH_CONTROL
                else 0
            ),
            "formal Q16 active/table/reservation byte formula drift",
        )
        _require(
            row["active_request_appended_tokens_sum"]
            == resident_count * FORMAL_FINAL_APPENDED_TOKENS
            and row["active_request_detached_tail_tokens_sum"]
            == resident_count * (FORMAL_DOCUMENT_TOKENS % FORMAL_PAGE_SIZE)
            and row["active_request_private_blocks"] == resident_count * 2,
            "final resident private-page occupancy drift",
        )
        normalized_layers.append(
            {
                key: row[key]
                for key in (
                    "layer_idx",
                    "block_bytes",
                    "valid_document_payload_nbytes",
                    "source_document_allocated_nbytes",
                    "source_document_padding_nbytes",
                    "source_private_reservation_nbytes",
                    "fresh_duplicate_document_allocated_nbytes",
                    "fresh_duplicate_document_padding_nbytes",
                    "fresh_duplicate_private_reservation_nbytes",
                    "active_request_private_payload_nbytes",
                    "active_request_private_allocated_page_nbytes",
                    "active_request_private_blocks",
                    "active_request_appended_tokens_sum",
                    "active_request_detached_tail_tokens_sum",
                    "partial_tail_staging_copy_nbytes",
                    "physical_document_block_copy_nbytes_including_padding",
                )
            }
        )
    totals = value["totals"]
    _require(
        isinstance(totals, dict)
        and set(totals) == total_fields
        and totals == accumulated,
        "resident storage totals do not replay from the exact layer schema",
    )
    return {"layers": normalized_layers, "totals": dict(totals)}


def _validate_group_audit(
    value: Any,
    *,
    resident_count: int,
    kv_policy: str,
    gdn_base_policy: str,
) -> dict[str, Any]:
    _require(isinstance(value, dict), "resident group audit payload missing")
    required = {
        "protocol",
        "policy",
        "gdn_base_policy",
        "resident_count",
        "all_requests_materialized_before_measurement",
        "strong_reference_count",
        "rows",
        "ownership",
        "physical_document_block_copy_nbytes_including_padding",
        "allocated_fresh_request_pool_nbytes",
    }
    _require(set(value) == required, "resident group audit schema drift")
    _require(
        value["protocol"] == MULTIFORK_PROTOCOL
        and value["policy"] == kv_policy
        and value["gdn_base_policy"] == gdn_base_policy
        and _is_int(value["resident_count"])
        and value["resident_count"] == resident_count
        and _is_int(value["strong_reference_count"])
        and value["strong_reference_count"] == resident_count
        and value["all_requests_materialized_before_measurement"] is True,
        "resident group audit policy/lifetime drift",
    )
    rows = value["rows"]
    _require(
        isinstance(rows, list)
        and [row.get("request_index") for row in rows] == list(range(resident_count)),
        "resident group audit request rows drift",
    )
    copy_sum = pool_sum = 0
    gdn_rows = []
    for row in rows:
        _require(isinstance(row, dict) and isinstance(row.get("gdn_base"), dict), "group request GDN audit missing")
        expected_row_fields = {
            "request_index",
            "document_block_copy_nbytes_including_padding",
            "allocated_request_pool_nbytes",
            "source_document_storage_shared",
            "gdn_base",
        }
        if kv_policy == FRESH_CONTROL:
            expected_row_fields |= {"document_payload_nbytes", "layers"}
        _require(set(row) == expected_row_fields, "group request row schema drift")
        _require(
            _is_int(row["request_index"])
            and 0 <= row["request_index"] < resident_count,
            "group request index drift",
        )
        gdn = row["gdn_base"]
        _require(
            set(gdn)
            == {
                "policy",
                "tensor_count",
                "borrowed_immutable_base_alias_count",
                "materialized_request_base_nbytes",
                "functional_rebind_after_transition",
            }
            and gdn.get("policy") == gdn_base_policy
            and _is_int(gdn.get("tensor_count"))
            and gdn.get("tensor_count") == 60
            and _is_int(gdn.get("borrowed_immutable_base_alias_count"))
            and _is_int(gdn.get("materialized_request_base_nbytes"))
            and gdn.get("functional_rebind_after_transition") is True,
            "group request GDN policy receipt drift",
        )
        if gdn_base_policy == GDN_BORROW_IMMUTABLE_BASE:
            _require(gdn.get("borrowed_immutable_base_alias_count") == 60 and gdn.get("materialized_request_base_nbytes") == 0, "borrowed GDN setup receipt drift")
        else:
            _require(gdn.get("borrowed_immutable_base_alias_count") == 0 and _is_int(gdn.get("materialized_request_base_nbytes")) and gdn["materialized_request_base_nbytes"] > 0, "materialized GDN setup receipt drift")
        copy_value = row.get("document_block_copy_nbytes_including_padding")
        pool_value = row.get("allocated_request_pool_nbytes")
        _require(_is_int(copy_value) and copy_value >= 0 and _is_int(pool_value) and pool_value >= 0, "group KV allocation receipt drift")
        expected_copy = (
            len(FORMAL_FULL_LAYERS) * FORMAL_DOCUMENT_ALLOCATED_NBYTES
            if kv_policy == FRESH_CONTROL
            else 0
        )
        expected_pool = (
            len(FORMAL_FULL_LAYERS)
            * (
                FORMAL_DOCUMENT_ALLOCATED_NBYTES
                + FORMAL_PRIVATE_BLOCKS_PER_REQUEST * FORMAL_BLOCK_NBYTES
            )
            if kv_policy == FRESH_CONTROL
            else 0
        )
        _require(
            row["source_document_storage_shared"]
            is (kv_policy == SHARED_REUSE)
            and copy_value == expected_copy
            and pool_value == expected_pool,
            "group KV policy did not execute the formal fresh/shared allocation",
        )
        if kv_policy == FRESH_CONTROL:
            _require(
                _is_int(row["document_payload_nbytes"])
                and row["document_payload_nbytes"]
                == len(FORMAL_FULL_LAYERS) * FORMAL_DOCUMENT_PAYLOAD_NBYTES,
                "fresh group document payload bytes drift",
            )
            layer_rows = row["layers"]
            _require(
                isinstance(layer_rows, list)
                and [item.get("layer_idx") for item in layer_rows]
                == list(FORMAL_FULL_LAYERS),
                "fresh group layer audit order drift",
            )
            for layer_row in layer_rows:
                _require(
                    isinstance(layer_row, dict)
                    and _is_int(layer_row.get("layer_idx"))
                    and all(
                        _is_int(layer_row.get(field))
                        for field in (
                            "document_block_copy_nbytes_including_padding",
                            "document_payload_nbytes",
                            "copied_padding_nbytes",
                            "allocated_request_pool_nbytes",
                        )
                    )
                    and layer_row
                    == {
                        "layer_idx": layer_row["layer_idx"],
                        "document_block_copy_nbytes_including_padding": FORMAL_DOCUMENT_ALLOCATED_NBYTES,
                        "document_payload_nbytes": FORMAL_DOCUMENT_PAYLOAD_NBYTES,
                        "copied_padding_nbytes": FORMAL_DOCUMENT_PADDING_NBYTES,
                        "allocated_request_pool_nbytes": (
                            FORMAL_DOCUMENT_ALLOCATED_NBYTES
                            + FORMAL_PRIVATE_BLOCKS_PER_REQUEST
                            * FORMAL_BLOCK_NBYTES
                        ),
                        "source_storage_shared": False,
                    },
                    "fresh group per-layer physical allocation drift",
                )
        copy_sum += copy_value
        pool_sum += pool_value
        gdn_rows.append({key: gdn[key] for key in sorted(gdn)})
    _require(
        _is_int(value["physical_document_block_copy_nbytes_including_padding"])
        and _is_int(value["allocated_fresh_request_pool_nbytes"])
        and value["physical_document_block_copy_nbytes_including_padding"]
        == copy_sum
        and value["allocated_fresh_request_pool_nbytes"] == pool_sum,
        "group allocation totals drift",
    )
    ownership = value["ownership"]
    expected_ownership = {
        "passed": True,
        "resident_count": resident_count,
        "request_object_ids_pairwise_distinct": True,
        "request_sequence_ids_pairwise_distinct": True,
        "private_physical_reservation_ids_pairwise_disjoint": kv_policy
        == SHARED_REUSE,
        "fresh_private_id_namespace_is_per_arena": kv_policy == FRESH_CONTROL,
        "reuse_requests_share_source_arena": kv_policy == SHARED_REUSE,
        "fresh_request_arena_storages_pairwise_disjoint": kv_policy
        == FRESH_CONTROL,
        "all_requests_strongly_referenced": True,
    }
    _require(
        isinstance(ownership, dict)
        and _is_int(ownership.get("resident_count")),
        "group ownership resident count must be a non-bool integer",
    )
    _require(ownership == expected_ownership, "group ownership audit failed")
    return {
        "kv": {
            "policy": kv_policy,
            "physical_document_copy_nbytes": copy_sum,
            "allocated_fresh_request_pool_nbytes": pool_sum,
            "ownership": dict(ownership),
        },
        "gdn": {"policy": gdn_base_policy, "rows": gdn_rows},
    }


def _validate_generation_diagnostics(
    value: Any, *, resident_count: int
) -> list[dict[str, Any]]:
    required = {
        "schema_version",
        "resident_count",
        "rounds",
        "schedule",
        "single_cpu_clone_per_step",
        "gpu_finite_or_hash_kernels_after_endpoint_sample",
        "rows",
        "rows_sha256",
    }
    _require(
        isinstance(value, dict)
        and set(value) == required
        and value["schema_version"] == "qcomem-generation-cpu-diagnostics-v1"
        and _is_int(value["resident_count"])
        and value["resident_count"] == resident_count
        and _is_int(value["rounds"])
        and value["rounds"] == FORMAL_GENERATION_STEPS
        and value["schedule"] == "round-major-request-minor"
        and value["single_cpu_clone_per_step"] is True
        and value["gpu_finite_or_hash_kernels_after_endpoint_sample"] is False,
        "generation CPU diagnostic receipt drift",
    )
    rows = value["rows"]
    row_fields = {
        "round_index",
        "request_index",
        "before_current_allocated_bytes",
        "before_current_reserved_bytes",
        "after_current_allocated_bytes",
        "after_current_reserved_bytes",
        "cpu_logits_dtype",
        "cpu_logits_shape",
        "cpu_logits_sha256",
        "finite_check_on_cpu",
        "allocator_state_exactly_unchanged",
    }
    _require(
        isinstance(rows, list)
        and len(rows) == FORMAL_GENERATION_STEPS * resident_count
        and value["rows_sha256"] == sha256_json(rows),
        "generation CPU diagnostic rows/digest drift",
    )
    normalized = []
    expected_vocab_size: int | None = None
    for row_index, row in enumerate(rows):
        expected_round = row_index // resident_count
        expected_request = row_index % resident_count
        _require(
            isinstance(row, dict)
            and set(row) == row_fields
            and _is_int(row["round_index"])
            and row["round_index"] == expected_round
            and _is_int(row["request_index"])
            and row["request_index"] == expected_request
            and row["finite_check_on_cpu"] is True
            and row["allocator_state_exactly_unchanged"] is True,
            "generation CPU diagnostic row identity drift",
        )
        for field in (
            "before_current_allocated_bytes",
            "before_current_reserved_bytes",
            "after_current_allocated_bytes",
            "after_current_reserved_bytes",
        ):
            _require(
                _is_int(row[field]) and row[field] >= 0,
                f"generation diagnostic {field} type/value drift",
            )
        _require(
            row["before_current_allocated_bytes"]
            == row["after_current_allocated_bytes"]
            and row["before_current_reserved_bytes"]
            == row["after_current_reserved_bytes"],
            "generation CPU diagnostics changed CUDA allocator state",
        )
        shape = row["cpu_logits_shape"]
        _require(
            row["cpu_logits_dtype"] == "torch.float32"
            and isinstance(shape, list)
            and len(shape) == 2
            and shape[0] == 1
            and _is_int(shape[1])
            and shape[1] > 1,
            "generation CPU logit tensor geometry drift",
        )
        if expected_vocab_size is None:
            expected_vocab_size = shape[1]
        else:
            _require(shape[1] == expected_vocab_size, "generation vocabulary size drift")
        normalized.append(
            {
                "round_index": expected_round,
                "request_index": expected_request,
                "cpu_logits_sha256": _require_sha256(
                    row["cpu_logits_sha256"], "generation CPU logit digest"
                ),
            }
        )
    return normalized


def _validate_memory_receipt(memory_cell: Mapping[str, Any]) -> dict[str, Any]:
    resident_count = memory_cell.get("resident_count")
    _require(
        _is_int(resident_count) and resident_count in FORMAL_RESIDENT_COUNTS,
        "memory resident count must be one of the non-bool formal N values",
    )
    receipt = memory_cell.get("allocator_receipt")
    _require(isinstance(receipt, dict), "formal memory allocator receipt missing")
    _require(
        set(receipt)
        == {
            "schema_version",
            "baseline",
            "after_setup",
            "after_setup_diagnostics",
            "after_generation",
            "generation_diagnostics",
            "peak_reset_before_setup",
            "peak_reset_before_generation",
            "synchronized_before_each_snapshot",
            "model_weights_loaded_before_baseline",
            "diagnostic_cpu_copies_excluded_from_peak",
            "diagnostic_current_allocator_state_unchanged",
            "setup_plus_generation_peak_allocated_delta_bytes",
            "setup_plus_generation_peak_reserved_delta_bytes",
            "generation_peak_allocated_delta_bytes",
            "generation_peak_reserved_delta_bytes",
            "storage_breakdown",
            "storage_breakdown_sha256",
            "unique_storage_removed_from_authorizing_payload",
        },
        "formal memory receipt schema drift",
    )
    _require(receipt["schema_version"] == "qcomem-formal-allocator-receipt-v4", "allocator receipt version drift")
    for flag in (
        "peak_reset_before_setup",
        "peak_reset_before_generation",
        "synchronized_before_each_snapshot",
        "model_weights_loaded_before_baseline",
        "diagnostic_cpu_copies_excluded_from_peak",
        "diagnostic_current_allocator_state_unchanged",
    ):
        _require(receipt[flag] is True, f"allocator execution receipt failed: {flag}")
    baseline = _validate_allocator_snapshot(receipt["baseline"], label="memory baseline")
    setup = _validate_allocator_snapshot(receipt["after_setup"], label="memory setup")
    diagnostics = _validate_allocator_snapshot(
        receipt["after_setup_diagnostics"], label="memory setup diagnostics"
    )
    generation = _validate_allocator_snapshot(receipt["after_generation"], label="memory generation")
    generation_diagnostics = _validate_generation_diagnostics(
        receipt["generation_diagnostics"],
        resident_count=resident_count,
    )
    _require(
        diagnostics["current_allocated_bytes"] == setup["current_allocated_bytes"]
        and diagnostics["current_reserved_bytes"] == setup["current_reserved_bytes"],
        "setup diagnostics changed current allocator state before generation",
    )
    _require(
        setup["current_allocated_bytes"] >= baseline["current_allocated_bytes"]
        and setup["current_reserved_bytes"] >= baseline["current_reserved_bytes"]
        and generation["current_allocated_bytes"] >= baseline["current_allocated_bytes"]
        and generation["current_reserved_bytes"] >= baseline["current_reserved_bytes"],
        "formal memory cell current allocator state fell below loaded-model baseline",
    )
    expected = {
        "setup_plus_generation_peak_allocated_delta_bytes": max(
            setup["peak_allocated_bytes"], generation["peak_allocated_bytes"]
        )
        - baseline["current_allocated_bytes"],
        "setup_plus_generation_peak_reserved_delta_bytes": max(
            setup["peak_reserved_bytes"], generation["peak_reserved_bytes"]
        )
        - baseline["current_reserved_bytes"],
        "generation_peak_allocated_delta_bytes": generation["peak_allocated_bytes"]
        - setup["current_allocated_bytes"],
        "generation_peak_reserved_delta_bytes": generation["peak_reserved_bytes"]
        - setup["current_reserved_bytes"],
        "after_generation_current_allocated_bytes": generation[
            "current_allocated_bytes"
        ],
        "after_generation_current_reserved_bytes": generation[
            "current_reserved_bytes"
        ],
        "after_generation_current_allocated_delta_bytes": generation[
            "current_allocated_bytes"
        ]
        - baseline["current_allocated_bytes"],
        "after_generation_current_reserved_delta_bytes": generation[
            "current_reserved_bytes"
        ]
        - baseline["current_reserved_bytes"],
    }
    for field in (
        "setup_plus_generation_peak_allocated_delta_bytes",
        "setup_plus_generation_peak_reserved_delta_bytes",
        "generation_peak_allocated_delta_bytes",
        "generation_peak_reserved_delta_bytes",
    ):
        expected_value = expected[field]
        _require(receipt[field] == expected_value and _is_int(receipt[field]), f"allocator derived endpoint drift: {field}")
    _require(all(value >= 0 for value in expected.values()), "allocator endpoint became negative")
    storage = receipt["storage_breakdown"]
    _require(isinstance(storage, dict), "memory storage breakdown missing")
    _require(receipt["storage_breakdown_sha256"] == sha256_json(storage), "memory storage breakdown SHA drift")
    kv_policy = memory_cell.get("policy_execution_receipt", {}).get("kv_policy")
    normalized_storage = _validate_resident_storage_breakdown(
        storage,
        resident_count=resident_count,
        kv_policy=kv_policy,
    )
    _require(
        receipt["unique_storage_removed_from_authorizing_payload"] is True,
        "unreplayable unique-storage diagnostic was used as an endpoint",
    )
    policy = memory_cell.get("policy_execution_receipt")
    _require(isinstance(policy, dict), "memory policy execution receipt missing")
    _require(
        set(policy)
        == {
            "builder",
            "kv_policy",
            "gdn_base_policy",
            "resident_count",
            "group_audit_sha256",
            "group_audit",
            "all_requests_materialized_before_measurement",
            "all_requests_alive_through_generation",
        },
        "memory policy receipt schema drift",
    )
    _require(policy["builder"] == "build_resident_request_group", "memory builder drift")
    _require(policy["kv_policy"] in KV_POLICIES and policy["gdn_base_policy"] in GDN_BASE_POLICIES, "memory policy axes drift")
    _require(
        _is_int(policy["resident_count"])
        and policy["resident_count"] == resident_count,
        "memory policy N drift",
    )
    _require(
        policy["group_audit_sha256"] == sha256_json(policy["group_audit"]),
        "memory group audit digest drift",
    )
    normalized_group = _validate_group_audit(
        policy["group_audit"],
        resident_count=policy["resident_count"],
        kv_policy=policy["kv_policy"],
        gdn_base_policy=policy["gdn_base_policy"],
    )
    storage_totals = normalized_storage["totals"]
    _require(
        normalized_group["kv"]["physical_document_copy_nbytes"]
        == storage_totals[
            "physical_document_block_copy_nbytes_including_padding"
        ]
        and normalized_group["kv"]["allocated_fresh_request_pool_nbytes"]
        == storage_totals["fresh_duplicate_document_allocated_nbytes"]
        + storage_totals["fresh_duplicate_private_reservation_nbytes"],
        "group audit allocation totals differ from resident storage replay",
    )
    _require(policy["all_requests_materialized_before_measurement"] is True and policy["all_requests_alive_through_generation"] is True, "memory resident lifetime receipt failed")
    return {
        **expected,
        "generation_diagnostics": generation_diagnostics,
        "normalized_storage": normalized_storage,
        "normalized_group": normalized_group,
    }


def _validate_kernel_ledgers(
    value: Any,
    *,
    resident_count: int,
    kv_policy: str,
    label: str,
    strict_position_values: bool,
) -> tuple[list[dict[str, Any]], tuple[str, str, str]]:
    import torch

    _require(
        _is_int(resident_count) and resident_count in FORMAL_RESIDENT_COUNTS,
        f"{label} resident count must be one of the non-bool formal N values",
    )
    _require(isinstance(value, list) and len(value) == resident_count, f"{label} ledger cardinality drift")
    descriptor: tuple[str, str, str] | None = None
    normalized = []
    ledger_fields = {
        "verified",
        "protocol",
        "request_index",
        "resident_count",
        "request_policy",
        "kernel_mode",
        "kernel_identity",
        "same_unified_attention_kernel",
        "counts",
        "total_calls",
        "round_major_request_local_layer_order_verified",
        "initial_query_tokens",
        "dense_fallback_calls",
        "full_kv_concatenations",
        "mask_contract",
        "position_ids_contract",
        "strict_position_values",
        "call_observer_enabled",
        "calls",
    }
    base_call_fields = {
        "request_index",
        "resident_count",
        "layer_idx",
        "request_policy",
        "protocol",
        "kernel_identity",
        "current_append_delta_tokens",
        "mask_contract",
        "materialized_attention_mask_nbytes",
        "mask_validation_host_syncs",
        "append_capture_id",
        "append_audit",
        "kernel_mode",
        "fused_gpu_kernel_calls",
        "full_kv_concatenations",
        "full_document_staging_copy_nbytes",
        "partial_tail_staging_copy_nbytes",
        "physical_block_pool_shape",
        "active_block_table_shape",
        "query_tokens",
        "kv_tokens",
        "softmax_scale",
        "gqa_groups",
        "quantization",
        "position_ids_contract",
        "position_ids_validated",
        "position_ids_semantically_consumed_upstream",
        "position_ids_shape",
        "position_ids_dtype",
        "position_ids_expected_tail_start",
        "position_ids_expected_tail_end_exclusive",
        "position_ids_strict_tail_values_checked",
        "position_ids_validation_host_syncs",
    }
    strict_call_fields = base_call_fields | {
        "position_ids_values",
        "position_ids_sha256",
    }
    append_audit_fields = {
        "append_event_index",
        "append_tokens",
        "appended_tokens_before",
        "appended_tokens_after",
        "sequence_length_before",
        "sequence_length_after",
        "capture_id",
    }
    for request_index, ledger in enumerate(value):
        _require(
            isinstance(ledger, dict) and set(ledger) == ledger_fields,
            f"{label} ledger row schema drift",
        )
        _require(
            _is_int(ledger.get("request_index"))
            and ledger.get("request_index") == request_index,
            f"{label} request order drift",
        )
        _require(
            _is_int(ledger.get("resident_count"))
            and ledger.get("resident_count") == resident_count
            and ledger.get("request_policy") == kv_policy,
            f"{label} policy/N drift",
        )
        _require(ledger.get("kernel_mode") == FORMAL_KERNEL_MODE, f"{label} kernel mode drift")
        _require(ledger.get("same_unified_attention_kernel") is True, f"{label} same-kernel gate failed")
        _require(
            ledger.get("strict_position_values") is strict_position_values,
            f"{label} strict position mode drift",
        )
        _require(ledger.get("verified") is True, f"{label} ledger was not verified")
        _require(ledger.get("protocol") == MULTIFORK_PROTOCOL, f"{label} protocol drift")
        _require(
            _is_int(ledger.get("initial_query_tokens"))
            and ledger.get("initial_query_tokens") == FORMAL_QUERY_TOKENS,
            f"{label} initial query length drift",
        )
        _require(
            ledger.get("round_major_request_local_layer_order_verified") is True,
            f"{label} round/layer order receipt missing",
        )
        _require(
            ledger.get("mask_contract") == FORMAL_MASK_CONTRACT
            and ledger.get("position_ids_contract") == FORMAL_POSITION_CONTRACT,
            f"{label} mask/position summary contract drift",
        )
        _require(type(ledger.get("call_observer_enabled")) is bool, f"{label} observer flag type drift")
        if strict_position_values:
            _require(
                ledger["call_observer_enabled"] is True,
                f"{label} strict witness did not enable its evidence observer",
            )
        else:
            _require(ledger["call_observer_enabled"] is False, f"{label} memory cell enabled a call observer")
        identity = ledger.get("kernel_identity")
        _require(
            isinstance(identity, dict)
            and set(identity) == {"module", "qualname", "signature"},
            f"{label} kernel identity missing",
        )
        current_descriptor = tuple(str(identity.get(field, "")) for field in ("module", "qualname", "signature"))
        _require(
            current_descriptor == FORMAL_KERNEL_DESCRIPTOR,
            f"{label} kernel descriptor differs from the frozen vLLM callable",
        )
        if descriptor is None:
            descriptor = current_descriptor
        else:
            _require(current_descriptor == descriptor, f"{label} kernel callable descriptor changed")
        calls = ledger.get("calls")
        _require(isinstance(calls, (list, tuple)) and len(calls) == FORMAL_GENERATION_STEPS * len(FORMAL_FULL_LAYERS), f"{label} call budget drift")
        expected_order = list(FORMAL_FULL_LAYERS) * FORMAL_GENERATION_STEPS
        _require([row.get("layer_idx") for row in calls] == expected_order, f"{label} layer/round order drift")
        observed_capture_ids: list[str] = []
        for call_index, call in enumerate(calls):
            _require(
                isinstance(call, dict)
                and set(call)
                == (strict_call_fields if strict_position_values else base_call_fields),
                f"{label} call row schema drift",
            )
            round_index = call_index // len(FORMAL_FULL_LAYERS)
            expected_delta = FORMAL_QUERY_TOKENS if round_index == 0 else 1
            _require(
                _is_int(call.get("request_index"))
                and call.get("request_index") == request_index
                and _is_int(call.get("resident_count"))
                and call.get("resident_count") == resident_count
                and _is_int(call.get("layer_idx"))
                and call.get("layer_idx") == expected_order[call_index]
                and call.get("request_policy") == kv_policy
                and call.get("protocol") == MULTIFORK_PROTOCOL,
                f"{label} call identity/policy binding drift",
            )
            _require(call.get("kernel_identity") == identity, f"{label} per-call kernel identity drift")
            _require(call.get("current_append_delta_tokens") == expected_delta, f"{label} append schedule drift")
            _require(call.get("query_tokens") == expected_delta, f"{label} query-token schedule drift")
            _require(
                call.get("kv_tokens")
                == FORMAL_DOCUMENT_TOKENS + FORMAL_QUERY_TOKENS + round_index,
                f"{label} KV-token schedule drift",
            )
            append_audit = call.get("append_audit")
            expected_before = (
                0 if round_index == 0 else FORMAL_QUERY_TOKENS + round_index - 1
            )
            expected_after = FORMAL_QUERY_TOKENS + round_index
            _require(
                isinstance(append_audit, dict)
                and set(append_audit) == append_audit_fields
                and all(
                    _is_int(append_audit.get(field))
                    for field in append_audit_fields - {"capture_id"}
                )
                and append_audit.get("append_event_index") == round_index
                and append_audit.get("append_tokens") == expected_delta
                and append_audit.get("appended_tokens_before") == expected_before
                and append_audit.get("appended_tokens_after") == expected_after
                and append_audit.get("sequence_length_before")
                == FORMAL_DOCUMENT_TOKENS + expected_before
                and append_audit.get("sequence_length_after")
                == FORMAL_DOCUMENT_TOKENS + expected_after,
                f"{label} append-event receipt drift",
            )
            _require(
                call.get("append_capture_id") == append_audit.get("capture_id"),
                f"{label} append capture/call binding drift",
            )
            if call.get("append_capture_id") is not None:
                _require(
                    isinstance(call["append_capture_id"], str)
                    and bool(call["append_capture_id"]),
                    f"{label} append capture ID drift",
                )
                observed_capture_ids.append(call["append_capture_id"])
            _require(call.get("fused_gpu_kernel_calls") == 1, f"{label} fused call count drift")
            _require(call.get("full_kv_concatenations") == 0 and call.get("full_document_staging_copy_nbytes") == 0, f"{label} dense/full-KV fallback")
            _require(
                call.get("kernel_mode") == FORMAL_KERNEL_MODE
                and call.get("quantization") == "Q16"
                and call.get("gqa_groups") == FORMAL_GQA_GROUPS,
                f"{label} kernel geometry/quantization drift",
            )
            expected_pool_blocks = (
                FORMAL_DOCUMENT_BLOCKS + FORMAL_PRIVATE_BLOCKS_PER_REQUEST
                if kv_policy == FRESH_CONTROL
                else FORMAL_DOCUMENT_BLOCKS
                + resident_count * FORMAL_PRIVATE_BLOCKS_PER_REQUEST
            )
            _require(
                tuple(call.get("physical_block_pool_shape", ()))
                == (
                    expected_pool_blocks,
                    FORMAL_PAGE_SIZE,
                    FORMAL_NUM_KV_HEADS,
                    FORMAL_HEAD_DIM,
                )
                and tuple(call.get("active_block_table_shape", ())) == (1, 33),
                f"{label} physical pool/active table geometry drift",
            )
            _require(
                all(
                    _is_int(item)
                    for item in call.get("physical_block_pool_shape", ())
                )
                and all(
                    _is_int(item)
                    for item in call.get("active_block_table_shape", ())
                ),
                f"{label} physical geometry contains non-integer values",
            )
            _require(
                call.get("partial_tail_staging_copy_nbytes")
                == FORMAL_PARTIAL_TAIL_COPY_NBYTES,
                f"{label} partial-tail copy receipt drift",
            )
            _require(
                call.get("mask_contract") == FORMAL_MASK_CONTRACT
                and call.get("materialized_attention_mask_nbytes") == 0
                and call.get("mask_validation_host_syncs") == 0,
                f"{label} mask contract drift",
            )
            expected_total = (
                FORMAL_DOCUMENT_TOKENS + FORMAL_QUERY_TOKENS + round_index
            )
            expected_position_start = expected_total - expected_delta
            _require(
                call.get("position_ids_contract") == FORMAL_POSITION_CONTRACT
                and call.get("position_ids_validated") is True
                and call.get("position_ids_semantically_consumed_upstream") is True
                and tuple(call.get("position_ids_shape", ()))
                == (1, expected_delta)
                and all(
                    _is_int(item) for item in call.get("position_ids_shape", ())
                )
                and call.get("position_ids_dtype") == "torch.int64"
                and call.get("position_ids_expected_tail_start")
                == expected_position_start
                and call.get("position_ids_expected_tail_end_exclusive")
                == expected_total
                and call.get("position_ids_validation_host_syncs")
                == (1 if strict_position_values else 0),
                f"{label} post-RoPE position receipt drift",
            )
            for integer_field in (
                "current_append_delta_tokens",
                "query_tokens",
                "kv_tokens",
                "fused_gpu_kernel_calls",
                "full_kv_concatenations",
                "full_document_staging_copy_nbytes",
                "partial_tail_staging_copy_nbytes",
                "gqa_groups",
                "materialized_attention_mask_nbytes",
                "mask_validation_host_syncs",
                "position_ids_expected_tail_start",
                "position_ids_expected_tail_end_exclusive",
                "position_ids_validation_host_syncs",
            ):
                _require(
                    _is_int(call.get(integer_field)),
                    f"{label} {integer_field} must be a non-bool integer",
                )
            _require(
                call.get("position_ids_strict_tail_values_checked")
                is strict_position_values,
                f"{label} per-call position mode drift",
            )
            if strict_position_values:
                expected_values = list(
                    range(expected_position_start, expected_total)
                )
                _require(
                    call.get("position_ids_values") == expected_values
                    and call.get("position_ids_sha256")
                    == _tensor_digest(
                        torch.tensor([expected_values], dtype=torch.int64)
                    ),
                    f"{label} exact position values/digest drift",
                )
            else:
                _require(call.get("append_capture_id") is None, f"{label} memory cell retained append-observer evidence")
            scale = call.get("softmax_scale")
            _require(
                isinstance(scale, (int, float))
                and not isinstance(scale, bool)
                and math.isfinite(scale)
                and float(scale) == FORMAL_SOFTMAX_SCALE,
                f"{label} effective scale differs from frozen head_dim=256 scale",
            )
        _require(
            len(observed_capture_ids) == len(set(observed_capture_ids)),
            f"{label} append capture ID reused",
        )
        if ledger["call_observer_enabled"]:
            _require(
                len(observed_capture_ids) == len(calls),
                f"{label} observer ledger missed append captures",
            )
        else:
            _require(
                not observed_capture_ids,
                f"{label} append observer enabled without call observer",
            )
        _require(
            _is_int(ledger.get("total_calls"))
            and ledger.get("total_calls") == len(calls),
            f"{label} total call receipt drift",
        )
        counts = ledger.get("counts")
        canonical_count_keys = {str(layer) for layer in FORMAL_FULL_LAYERS}
        native_count_keys = set(FORMAL_FULL_LAYERS)
        _require(
            isinstance(counts, dict)
            and set(counts) in (canonical_count_keys, native_count_keys)
            and all(
                _is_int(item) and item == FORMAL_GENERATION_STEPS
                for item in counts.values()
            ),
            f"{label} per-layer call counts drift",
        )
        _require(
            _is_int(ledger.get("dense_fallback_calls"))
            and ledger.get("dense_fallback_calls") == 0
            and _is_int(ledger.get("full_kv_concatenations"))
            and ledger.get("full_kv_concatenations") == 0,
            f"{label} fallback summary drift",
        )
        normalized.append({"request_index": request_index, "calls": list(calls), "kernel_identity": identity})
    assert descriptor is not None
    return normalized, descriptor


_TORCH_DTYPES = {
    "torch.float16": "float16",
    "torch.bfloat16": "bfloat16",
    "torch.float32": "float32",
    "torch.float64": "float64",
    "torch.int32": "int32",
    "torch.int64": "int64",
    "torch.bool": "bool",
}


def _tensor_digest(tensor: Any) -> str:
    import torch

    contiguous = tensor.detach().contiguous().cpu()
    metadata = {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "raw_sha256": sha256_bytes(contiguous.view(torch.uint8).numpy().tobytes()),
    }
    return sha256_json(metadata)


def _decode_inline_tensor(record: Any, *, label: str) -> Any:
    import torch

    _require(isinstance(record, dict), f"{label} tensor record missing")
    _require(set(record) == {"encoding", "dtype", "shape", "values", "sha256"}, f"{label} tensor schema drift")
    _require(record["encoding"] == "inline-json-row-major-v1", f"{label} tensor encoding drift")
    dtype_name = record["dtype"]
    _require(dtype_name in _TORCH_DTYPES, f"{label} tensor dtype unsupported")
    shape = record["shape"]
    _require(isinstance(shape, list) and shape and all(_is_int(item) and item > 0 for item in shape), f"{label} tensor shape drift")
    values = record["values"]
    _require(isinstance(values, list) and len(values) == math.prod(shape), f"{label} tensor value count drift")
    tensor = torch.tensor(values, dtype=getattr(torch, _TORCH_DTYPES[dtype_name])).reshape(shape)
    _require(_tensor_digest(tensor) == _require_sha256(record["sha256"], f"{label} tensor digest"), f"{label} tensor digest mismatch")
    return tensor


def _decode_tensor_record(
    record: Any,
    *,
    root: Path,
    label: str,
    require_binary: bool,
) -> tuple[Any, dict[str, Any] | None]:
    if isinstance(record, dict) and record.get("encoding") == "inline-json-row-major-v1":
        _require(not require_binary, f"{label} formal tensor must use a binary sidecar")
        return _decode_inline_tensor(record, label=label), None
    import torch

    _require(isinstance(record, dict), f"{label} tensor record missing")
    _require(
        set(record) == {"encoding", "dtype", "shape", "artifact", "sha256"},
        f"{label} binary tensor schema drift",
    )
    _require(record["encoding"] == "torch-contiguous-raw-little-endian-v1", f"{label} binary encoding drift")
    dtype_name = record["dtype"]
    _require(dtype_name in _TORCH_DTYPES, f"{label} tensor dtype unsupported")
    shape = record["shape"]
    _require(isinstance(shape, list) and shape and all(_is_int(item) and item > 0 for item in shape), f"{label} tensor shape drift")
    reference = record["artifact"]
    _require(isinstance(reference, dict) and set(reference) == {"relative_path", "sha256", "bytes"}, f"{label} binary receipt drift")
    path = _safe_artifact_path(root, reference["relative_path"])
    raw = path.read_bytes()
    _require(len(raw) == reference["bytes"], f"{label} binary byte count mismatch")
    _require(sha256_bytes(raw) == _require_sha256(reference["sha256"], f"{label} binary SHA"), f"{label} binary SHA mismatch")
    dtype = getattr(torch, _TORCH_DTYPES[dtype_name])
    element_size = torch.empty((), dtype=dtype).element_size()
    _require(len(raw) == math.prod(shape) * element_size, f"{label} binary tensor size mismatch")
    byte_tensor = torch.frombuffer(bytearray(raw), dtype=torch.uint8).clone()
    tensor = byte_tensor.view(dtype).reshape(shape)
    _require(_tensor_digest(tensor) == _require_sha256(record["sha256"], f"{label} tensor digest"), f"{label} tensor digest mismatch")
    return tensor, {
        "relative_path": reference["relative_path"],
        "sha256": reference["sha256"],
        "bytes": reference["bytes"],
    }


def encode_inline_tensor(tensor: Any) -> dict[str, Any]:
    """Encode a small probe tensor for a replayable JSON oracle artifact."""

    contiguous = tensor.detach().contiguous().cpu()
    return {
        "encoding": "inline-json-row-major-v1",
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "values": contiguous.reshape(-1).tolist(),
        "sha256": _tensor_digest(contiguous),
    }


def write_binary_tensor(
    tensor: Any,
    *,
    path: Path,
    root: Path,
) -> dict[str, Any]:
    """Freeze one CPU/GPU tensor as a compact raw binary sidecar."""

    import sys
    import torch

    _require(sys.byteorder == "little", "formal tensor sidecars require little endian")
    contiguous = tensor.detach().contiguous().cpu()
    _require(str(contiguous.dtype) in _TORCH_DTYPES, "unsupported binary tensor dtype")
    raw = contiguous.view(torch.uint8).numpy().tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return {
        "encoding": "torch-contiguous-raw-little-endian-v1",
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "artifact": artifact_reference(path, root=root),
        "sha256": _tensor_digest(contiguous),
    }


def _physical_payload_digest_from_tensors(
    key_pool: Any,
    value_pool: Any,
    document_block_table: Any,
    *,
    layer_index: int,
    document_length: int,
    page_size: int,
) -> str:
    """Blind-replay counterpart of ``_physical_document_payload_digests``."""

    import torch

    _require(
        key_pool.shape == value_pool.shape
        and key_pool.ndim == 4
        and key_pool.dtype == value_pool.dtype,
        "oracle physical document pool geometry drift",
    )
    _require(
        document_block_table.ndim == 2
        and document_block_table.dtype == torch.int32,
        "oracle document block-table schema drift",
    )
    batch_size, document_blocks = document_block_table.shape
    _require(
        document_blocks == math.ceil(document_length / page_size)
        and key_pool.shape[1] == page_size,
        "oracle physical document block geometry drift",
    )
    _require(
        bool((document_block_table >= 0).all())
        and bool((document_block_table < key_pool.shape[0]).all()),
        "oracle document block table points outside physical pool",
    )
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "schema_version": "qcomem-document-physical-payload-v1",
                "layer_index": int(layer_index),
                "document_length": int(document_length),
                "page_size": int(page_size),
                "batch_size": int(batch_size),
                "document_blocks_per_sequence": int(document_blocks),
                "num_key_value_heads": int(key_pool.shape[2]),
                "head_dim": int(key_pool.shape[3]),
                "dtype": str(key_pool.dtype),
                "component_order": "batch-major/block-major/key-then-value",
                "tail_padding_included": True,
            }
        )
    )
    digest.update(b"\0")
    for physical in document_block_table.reshape(-1).tolist():
        for component in (key_pool, value_pool):
            digest.update(
                component[int(physical)]
                .detach()
                .contiguous()
                .cpu()
                .view(torch.uint8)
                .numpy()
                .tobytes()
            )
    return digest.hexdigest()


def _logical_document_from_physical(
    pool: Any,
    document_block_table: Any,
    *,
    document_length: int,
) -> Any:
    import torch

    batches = []
    for row in document_block_table:
        physical = row.to(torch.int64)
        blocks = pool.index_select(0, physical).reshape(
            -1, int(pool.shape[2]), int(pool.shape[3])
        )[:document_length]
        batches.append(blocks.permute(1, 0, 2).contiguous())
    return torch.stack(batches, dim=0)


def _write_live_oracle_artifact(
    *,
    artifact_root: Path,
    path_prefix: str,
    run_id: str,
    rank: int,
    cell_id: str,
    arm_id: str,
    selection: Mapping[str, Any],
    persistent: Any,
    group: Any,
    source_payload_sha256: str,
    append_collector: OracleAppendShadowCollector,
    live_capture: Mapping[str, Any],
    selected_ledger: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    layer_index = int(selection["layer_index"])
    round_index = int(selection["round_index"])
    layer_offset = FORMAL_FULL_LAYERS.index(layer_index)
    call_offset = round_index * len(FORMAL_FULL_LAYERS) + layer_offset
    ledger_call = selected_ledger["calls"][call_offset]
    _require(
        live_capture["append_capture_id"] == ledger_call.get("append_capture_id")
        and live_capture["append_audit"] == ledger_call.get("append_audit"),
        "producer oracle live capture/ledger binding drift",
    )
    arena = persistent.layers[layer_index].arena
    document_table_device = arena.document_block_table.detach().contiguous()
    document_table = document_table_device.cpu().clone()
    document_ids = document_table.reshape(-1).to(torch.int64)
    _require(
        len(set(int(value) for value in document_ids.tolist()))
        == int(document_ids.numel()),
        "oracle document table reused a physical block",
    )
    # Never archive private arena backing: it is allocated with torch.empty
    # and contains unwritten bytes beyond the active append.  Only the fully
    # initialized document-table-referenced blocks (including zeroed tail
    # padding) are safe and relevant to the independent oracle source.
    document_ids_device = document_table_device.reshape(-1).to(
        device=arena.key_cache.device, dtype=torch.int64
    )
    key_pool = (
        arena.key_cache.index_select(0, document_ids_device)
        .detach()
        .contiguous()
        .cpu()
        .clone()
    )
    value_pool = (
        arena.value_cache.index_select(0, document_ids_device)
        .detach()
        .contiguous()
        .cpu()
        .clone()
    )
    normalized_document_table = torch.arange(
        document_ids.numel(), dtype=torch.int32
    ).reshape_as(document_table)
    active_table = live_capture["active_block_table"]
    _require(isinstance(active_table, torch.Tensor), "oracle active table snapshot missing")
    physical_digest = _physical_payload_digest_from_tensors(
        key_pool,
        value_pool,
        normalized_document_table,
        layer_index=layer_index,
        document_length=FORMAL_DOCUMENT_TOKENS,
        page_size=FORMAL_PAGE_SIZE,
    )
    _require(physical_digest == source_payload_sha256, "oracle sidecar/source payload digest drift")
    document_key = _logical_document_from_physical(
        key_pool, normalized_document_table, document_length=FORMAL_DOCUMENT_TOKENS
    )
    document_value = _logical_document_from_physical(
        value_pool, normalized_document_table, document_length=FORMAL_DOCUMENT_TOKENS
    )
    append_key, append_value, selected_append = append_collector.concatenated_through(
        round_index
    )
    query = live_capture["query"]
    candidate = live_capture["candidate"]
    query_positions = live_capture["query_positions"].to(torch.int64)
    key = torch.cat((document_key, append_key), dim=2)
    value = torch.cat((document_value, append_value), dim=2)
    key_positions = torch.arange(key.shape[2], dtype=torch.int64)
    scale = float(live_capture["effective_scaling"])
    digests = {
        "query": _tensor_digest(query),
        "key": _tensor_digest(key),
        "value": _tensor_digest(value),
        "document_key": _tensor_digest(document_key),
        "document_value": _tensor_digest(document_value),
        "append_key_shadow": _tensor_digest(append_key),
        "append_value_shadow": _tensor_digest(append_value),
        "candidate_output": _tensor_digest(candidate),
        "query_positions": _tensor_digest(query_positions),
        "key_positions": _tensor_digest(key_positions),
        "visibility_mask": sha256_json(None),
        "softmax_scale": sha256_json(scale),
        "document_key_value_component": sha256_json(
            [_tensor_digest(document_key), _tensor_digest(document_value)]
        ),
        "append_key_value_shadow_component": sha256_json(
            [_tensor_digest(append_key), _tensor_digest(append_value)]
        ),
        "physical_document_key_blocks": _tensor_digest(key_pool),
        "physical_document_value_blocks": _tensor_digest(value_pool),
        "document_block_table": _tensor_digest(normalized_document_table),
        "candidate_active_block_table": _tensor_digest(active_table),
        "source_physical_payload": physical_digest,
    }
    tensor_dir = artifact_root / path_prefix / "oracle-sidecars"
    tensors = {
        "query": write_binary_tensor(query, path=tensor_dir / "query.bin", root=artifact_root),
        "candidate_output": write_binary_tensor(candidate, path=tensor_dir / "candidate-output.bin", root=artifact_root),
        "query_positions": write_binary_tensor(query_positions, path=tensor_dir / "query-positions.bin", root=artifact_root),
        "key_positions": write_binary_tensor(key_positions, path=tensor_dir / "key-positions.bin", root=artifact_root),
        "physical_document_key_blocks": write_binary_tensor(key_pool, path=tensor_dir / "physical-document-key-blocks.bin", root=artifact_root),
        "physical_document_value_blocks": write_binary_tensor(value_pool, path=tensor_dir / "physical-document-value-blocks.bin", root=artifact_root),
        "document_block_table": write_binary_tensor(normalized_document_table, path=tensor_dir / "document-block-table.bin", root=artifact_root),
        "candidate_active_block_table": write_binary_tensor(active_table, path=tensor_dir / "candidate-active-block-table.bin", root=artifact_root),
        "visibility_mask": None,
    }
    append_events = []
    for event in append_collector.events[: round_index + 1]:
        event_index = int(event["append_event_index"])
        append_events.append(
            {
                "schema_version": "qcomem-oracle-append-event-v1",
                "capture_id": event["capture_id"],
                "append_event_index": event_index,
                "appended_tokens_before": event["appended_tokens_before"],
                "appended_tokens_after": event["appended_tokens_after"],
                "sequence_length_before": event["sequence_length_before"],
                "sequence_length_after": event["sequence_length_after"],
                "source_device": event["source_device"],
                "source_dtype": event["source_dtype"],
                "source_shape": event["source_shape"],
                "key_sha256": event["key_sha256"],
                "value_sha256": event["value_sha256"],
                "key": write_binary_tensor(
                    event["key"],
                    path=tensor_dir / f"append-{event_index}-key.bin",
                    root=artifact_root,
                ),
                "value": write_binary_tensor(
                    event["value"],
                    path=tensor_dir / f"append-{event_index}-value.bin",
                    root=artifact_root,
                ),
            }
        )
    _require(selected_append["capture_id"] == live_capture["append_capture_id"], "selected append event drift")
    recorded = OraclePreregistration(
        OracleThresholds(max_relative_l2=ORACLE_MAX_RELATIVE_L2)
    ).evaluate_attention(
        query,
        key,
        value,
        candidate,
        query_positions=query_positions,
        key_positions=key_positions,
        visibility_mask=None,
        scaling=scale,
    ).to_dict()
    selection_sha, outer_sha = _oracle_selection_preregistration(selection)
    append_manifest = []
    for row in append_events:
        event_index = int(row["append_event_index"])
        event_call = selected_ledger["calls"][
            event_index * len(FORMAL_FULL_LAYERS) + layer_offset
        ]
        _require(
            event_call.get("append_capture_id") == row["capture_id"],
            "oracle append history/ledger capture binding drift",
        )
        append_manifest.append(
            {
                "capture_id": row["capture_id"],
                "append_event_index": event_index,
                "appended_tokens_before": row["appended_tokens_before"],
                "appended_tokens_after": row["appended_tokens_after"],
                "key_sha256": row["key_sha256"],
                "value_sha256": row["value_sha256"],
                "layer_index": layer_index,
                "request_index": int(selection["request_index"]),
                "round_index": event_index,
                "ledger_call_sha256": sha256_json(event_call),
            }
        )
    raw = {
        "schema_version": ORACLE_RAW_SCHEMA_VERSION,
        "resident_count": ORACLE_RESIDENT_COUNT,
        "selection": dict(selection),
        "selection_sha256": selection_sha,
        "outer_preregistration_sha256": outer_sha,
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
        "document_geometry": {
            "document_length": FORMAL_DOCUMENT_TOKENS,
            "page_size": FORMAL_PAGE_SIZE,
        },
        "arena_geometry": {
            "total_physical_blocks": int(arena.key_cache.shape[0]),
            "document_physical_blocks": int(document_ids.numel()),
            "private_physical_blocks": int(
                arena.key_cache.shape[0] - document_ids.numel()
            ),
            "document_sidecars_exclude_private_uninitialized_backing": True,
        },
        "softmax_scale": scale,
        "tensors": tensors,
        "append_events": append_events,
        "input_digests": digests,
        "live_call_observer": {
            "schema_version": "qcomem-live-call-observer-v2",
            "run_id": run_id,
            "rank": rank,
            "resident_count": ORACLE_RESIDENT_COUNT,
            "cell_id": cell_id,
            "arm_id": arm_id,
            "kv_policy": selection["kv_policy"],
            "gdn_base_policy": selection["gdn_base_policy"],
            "sample_id": selection["sample_id"],
            "layer_index": layer_index,
            "request_index": int(selection["request_index"]),
            "round_index": round_index,
            "ledger_call_sha256": sha256_json(ledger_call),
            "append_capture_id": live_capture["append_capture_id"],
            "append_event_manifest": append_manifest,
            "kernel_audit": dict(live_capture["kernel_audit"]),
            "effective_scaling": scale,
            "softmax_scale_source": "MultiForkHitLedger.call_observer.kernel_audit.softmax_scale",
            "input_digests": digests,
            "document_capture": {
                "capture_point": "persistent-document-arena-via-document-block-table",
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
                "source_physical_payload_sha256": physical_digest,
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
        "recorded_outcome": recorded,
    }
    path = artifact_root / path_prefix / "oracle-raw.json"
    _write_json(path, raw)
    return artifact_reference(path, root=artifact_root)


def _oracle_selection_preregistration(selection: Mapping[str, Any]) -> tuple[str, str]:
    selection_sha = sha256_json(selection)
    preregistration = {
        "protocol": PROTOCOL,
        "selection_sha256": selection_sha,
        "selection_locked_before_candidate_outputs": True,
        "resident_count": ORACLE_RESIDENT_COUNT,
        "max_relative_l2": ORACLE_MAX_RELATIVE_L2,
        "reference": "independent-dense-ieee-fp32-post-rope-qkv",
        "key_value_source": "persistent-physical-document-blocks-independent-of-candidate-active-block-table",
        "candidate_softmax_scale_source": "live-kernel-observer",
    }
    return selection_sha, sha256_json(preregistration)


def _validate_oracle_precision_audit(value: Any, *, label: str) -> None:
    _require(isinstance(value, dict), f"{label} precision audit missing")
    required = {
        "policy",
        "device_type",
        "applies_to_cuda_reference",
        "candidate_executed_outside_context",
        "before",
        "effective",
        "effective_ieee_fp32",
        "after",
        "restored",
    }
    _require(set(value) == required, f"{label} precision audit schema drift")
    _require(value["policy"] == "ieee-fp32-reference-only", f"{label} precision policy drift")
    device_type = value["device_type"]
    _require(device_type in ("cpu", "cuda"), f"{label} reference device drift")
    _require(
        value["applies_to_cuda_reference"] is (device_type == "cuda"),
        f"{label} CUDA-reference applicability drift",
    )
    _require(value["candidate_executed_outside_context"] is True, f"{label} candidate/reference context overlap")
    _require(value["effective_ieee_fp32"] is True, f"{label} IEEE FP32 was not effective")
    _require(value["after"] == value["before"] and value["restored"] is True, f"{label} backend precision state not restored")
    effective = value["effective"]
    _require(isinstance(effective, dict), f"{label} effective backend state missing")
    _require(effective.get("float32_matmul_precision") == "highest", f"{label} matmul precision drift")
    if device_type == "cuda":
        control = effective.get("cuda_matmul_control")
        _require(control in ("fp32_precision", "allow_tf32"), f"{label} CUDA precision control drift")
        if control == "fp32_precision":
            _require(effective.get("cuda_matmul_fp32_precision") == "ieee", f"{label} CUDA FP32 policy drift")
        else:
            _require(effective.get("cuda_matmul_allow_tf32") is False, f"{label} TF32 remained enabled")


def _recompute_oracle(
    reference: Any,
    *,
    root: Path,
    rank: int,
    source_object: str,
    expected_selection: Mapping[str, Any],
    observer_context: Mapping[str, Any] | None = None,
    expected_run_id: str | None = None,
    synthetic_geometry: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    loaded = load_json_artifact(reference, root=root, label="oracle raw sample")
    raw = loaded.payload
    _require(isinstance(raw, dict) and raw.get("schema_version") == ORACLE_RAW_SCHEMA_VERSION, "oracle raw schema drift")
    _require(
        _is_int(raw.get("resident_count"))
        and raw.get("resident_count") == ORACLE_RESIDENT_COUNT,
        "oracle must use non-bool N=1",
    )
    selection = raw.get("selection")
    _require(isinstance(selection, dict), "oracle selection missing")
    _require(selection == expected_selection, "oracle raw sample differs from frozen pre-run selection")
    _require(selection["rank"] == rank and selection["book_index"] == rank, "oracle rank/book binding drift")
    _require(selection["source_object"] == source_object, "oracle source binding drift")
    _require(selection["layer_index"] in FORMAL_FULL_LAYERS, "oracle layer drift")
    _require(selection["request_index"] == 0 and 0 <= selection["round_index"] < FORMAL_GENERATION_STEPS, "oracle request/round drift")
    _require(selection["held_out_from_threshold_calibration"] is True and selection["locked_before_candidate_outputs"] is True, "oracle selection was not preregistered")
    selection_sha, outer_prereg_sha = _oracle_selection_preregistration(selection)
    _require(raw.get("selection_sha256") == selection_sha, "oracle selection SHA drift")
    _require(raw.get("outer_preregistration_sha256") == outer_prereg_sha, "oracle outer preregistration SHA drift")

    source_contract = raw.get("source_contract")
    _require(
        source_contract
        == {
            "post_rope_qkv": True,
            "candidate_output_from_live_unified_attention": True,
            "key_value_source": "immutable-document-physical-blocks-plus-independent-append-shadow",
            "key_value_independent_of_candidate_active_block_table": True,
            "document_component_sha256": raw.get("input_digests", {}).get("document_key_value_component"),
            "append_shadow_component_sha256": raw.get("input_digests", {}).get("append_key_value_shadow_component"),
            "concatenation_order": "document-then-append-shadow",
            "candidate_softmax_scale_source": "live-kernel-observer",
        },
        "oracle source independence contract drift",
    )
    scale = raw.get("softmax_scale")
    _require(isinstance(scale, (int, float)) and not isinstance(scale, bool) and math.isfinite(scale) and scale > 0.0, "observed oracle scale missing")

    geometry = raw.get("document_geometry")
    _require(
        isinstance(geometry, dict)
        and set(geometry) == {"document_length", "page_size"},
        "oracle document geometry receipt drift",
    )
    document_length = geometry["document_length"]
    page_size = geometry["page_size"]
    _require(_is_int(document_length) and document_length > 0, "oracle document length drift")
    _require(_is_int(page_size) and page_size > 0, "oracle page size drift")
    if not synthetic_geometry:
        _require(
            document_length == FORMAL_DOCUMENT_TOKENS
            and page_size == FORMAL_PAGE_SIZE,
            "formal oracle document geometry drift",
        )

    tensors = raw.get("tensors")
    expected_tensor_names = {
        "query",
        "candidate_output",
        "query_positions",
        "key_positions",
        "physical_document_key_blocks",
        "physical_document_value_blocks",
        "document_block_table",
        "candidate_active_block_table",
        "visibility_mask",
    }
    _require(
        isinstance(tensors, dict) and set(tensors) == expected_tensor_names,
        "oracle tensor manifest schema drift",
    )
    import torch

    tensor_bindings: list[dict[str, Any]] = []

    def decode_record(record: Any, label: str):
        tensor, binding = _decode_tensor_record(
            record,
            root=root,
            label=label,
            require_binary=not synthetic_geometry,
        )
        if binding is not None:
            tensor_bindings.append(binding)
        return tensor

    def decode(name: str, label: str):
        return decode_record(tensors.get(name), label)

    query = decode("query", "oracle query")
    candidate = decode("candidate_output", "oracle candidate")
    query_positions = decode("query_positions", "oracle query positions")
    key_positions = decode("key_positions", "oracle key positions")
    key_pool = decode(
        "physical_document_key_blocks", "oracle physical document key blocks"
    )
    value_pool = decode(
        "physical_document_value_blocks", "oracle physical document value blocks"
    )
    document_table = decode("document_block_table", "oracle document block table")
    active_table = decode("candidate_active_block_table", "oracle candidate block table")
    visibility_record = tensors["visibility_mask"]
    visibility_mask = None
    if visibility_record is not None:
        visibility_mask = decode_record(visibility_record, "oracle visibility mask")
    arena_geometry = raw.get("arena_geometry")
    _require(
        isinstance(arena_geometry, dict)
        and set(arena_geometry)
        == {
            "total_physical_blocks",
            "document_physical_blocks",
            "private_physical_blocks",
            "document_sidecars_exclude_private_uninitialized_backing",
        }
        and _is_int(arena_geometry["total_physical_blocks"])
        and _is_int(arena_geometry["document_physical_blocks"])
        and _is_int(arena_geometry["private_physical_blocks"])
        and arena_geometry["total_physical_blocks"]
        == arena_geometry["document_physical_blocks"]
        + arena_geometry["private_physical_blocks"]
        and arena_geometry["document_physical_blocks"] == key_pool.shape[0]
        and arena_geometry[
            "document_sidecars_exclude_private_uninitialized_backing"
        ]
        is True,
        "oracle arena/document-only sidecar geometry drift",
    )
    total_physical_blocks = arena_geometry["total_physical_blocks"]
    if not synthetic_geometry:
        _require(
            tuple(key_pool.shape) == (32, FORMAL_PAGE_SIZE, 2, 256)
            and value_pool.shape == key_pool.shape
            and key_pool.dtype == value_pool.dtype == torch.bfloat16,
            "formal oracle physical document-block layout drift",
        )
        _require(
            arena_geometry
            == {
                "total_physical_blocks": 34,
                "document_physical_blocks": 32,
                "private_physical_blocks": 2,
                "document_sidecars_exclude_private_uninitialized_backing": True,
            },
            "formal oracle arena geometry drift",
        )
        _require(
            document_table.dtype == torch.int32
            and tuple(document_table.shape) == (1, 32)
            and torch.equal(
                document_table.cpu().reshape(-1),
                torch.arange(32, dtype=torch.int32),
            ),
            "formal oracle document block table drift",
        )
        expected_active = torch.tensor(
            [list(range(31)) + [32, 33]], dtype=torch.int32
        )
        _require(
            active_table.dtype == torch.int32
            and tuple(active_table.shape) == (1, 33)
            and torch.equal(active_table.cpu(), expected_active)
            and bool((active_table >= 0).all())
            and bool((active_table < total_physical_blocks).all()),
            "formal oracle candidate active block table drift",
        )
    else:
        _require(
            bool((active_table >= 0).all())
            and bool((active_table < total_physical_blocks).all()),
            "synthetic oracle active block table points outside arena",
        )

    physical_digest = _physical_payload_digest_from_tensors(
        key_pool,
        value_pool,
        document_table,
        layer_index=int(selection["layer_index"]),
        document_length=document_length,
        page_size=page_size,
    )
    document_key = _logical_document_from_physical(
        key_pool, document_table, document_length=document_length
    )
    document_value = _logical_document_from_physical(
        value_pool, document_table, document_length=document_length
    )

    append_rows = raw.get("append_events")
    _require(
        isinstance(append_rows, list)
        and len(append_rows) == int(selection["round_index"]) + 1,
        "oracle append-event history is incomplete",
    )
    append_keys = []
    append_values = []
    append_manifest = []
    capture_ids: list[str] = []
    running_tokens = 0
    for event_index, event in enumerate(append_rows):
        required_event = {
            "schema_version",
            "capture_id",
            "append_event_index",
            "appended_tokens_before",
            "appended_tokens_after",
            "sequence_length_before",
            "sequence_length_after",
            "source_device",
            "source_dtype",
            "source_shape",
            "key_sha256",
            "value_sha256",
            "key",
            "value",
        }
        _require(isinstance(event, dict) and set(event) == required_event, "oracle append-event schema drift")
        _require(event["schema_version"] == "qcomem-oracle-append-event-v1", "oracle append-event version drift")
        _require(
            all(
                _is_int(event[field])
                for field in (
                    "append_event_index",
                    "appended_tokens_before",
                    "appended_tokens_after",
                    "sequence_length_before",
                    "sequence_length_after",
                )
            ),
            "oracle append-event integer type drift",
        )
        capture_id = event["capture_id"]
        _require(isinstance(capture_id, str) and bool(capture_id), "oracle append capture ID missing")
        capture_ids.append(capture_id)
        event_key = decode_record(event["key"], f"oracle append key event {event_index}")
        event_value = decode_record(event["value"], f"oracle append value event {event_index}")
        _require(event_key.shape == event_value.shape and event_key.ndim == 4, "oracle append event K/V geometry drift")
        incoming = int(event_key.shape[2])
        expected_incoming = (
            FORMAL_QUERY_TOKENS if event_index == 0 else 1
        ) if not synthetic_geometry else incoming
        _require(incoming == expected_incoming and incoming > 0, "oracle append-event token schedule drift")
        _require(
            event["append_event_index"] == event_index
            and event["appended_tokens_before"] == running_tokens
            and event["appended_tokens_after"] == running_tokens + incoming
            and event["sequence_length_before"] == document_length + running_tokens
            and event["sequence_length_after"] == document_length + running_tokens + incoming,
            "oracle append-event before/after receipt drift",
        )
        _require(
            event["source_dtype"] == str(event_key.dtype)
            and event["source_shape"] == list(event_key.shape)
            and event["key_sha256"] == _tensor_digest(event_key)
            and event["value_sha256"] == _tensor_digest(event_value),
            "oracle append-event source/digest receipt drift",
        )
        expected_source_device = "cpu" if synthetic_geometry else "cuda:0"
        _require(
            event["source_device"] == expected_source_device,
            "oracle append-event source device is not the exact local capture device",
        )
        if not synthetic_geometry:
            _require(
                event_key.dtype == torch.bfloat16,
                "formal append shadow was not captured from BF16 local CUDA K/V",
            )
        append_keys.append(event_key)
        append_values.append(event_value)
        append_manifest.append(
            {
                "capture_id": capture_id,
                "append_event_index": event_index,
                "appended_tokens_before": running_tokens,
                "appended_tokens_after": running_tokens + incoming,
                "key_sha256": _tensor_digest(event_key),
                "value_sha256": _tensor_digest(event_value),
            }
        )
        running_tokens += incoming
    _require(len(set(capture_ids)) == len(capture_ids), "oracle append capture ID reused")
    append_key = torch.cat(append_keys, dim=2)
    append_value = torch.cat(append_values, dim=2)
    _require(
        document_key.shape[:2] == append_key.shape[:2]
        and document_key.shape[3:] == append_key.shape[3:],
        "oracle document/append K/V geometry drift",
    )
    key = torch.cat((document_key, append_key), dim=2)
    value = torch.cat((document_value, append_value), dim=2)
    if not synthetic_geometry:
        expected_query_tokens = FORMAL_QUERY_TOKENS if selection["round_index"] == 0 else 1
        _require(
            query.dtype == candidate.dtype == key.dtype == value.dtype == torch.bfloat16
            and query_positions.dtype == key_positions.dtype == torch.int64,
            "formal oracle dtype contract drift",
        )
        _require(
            tuple(query.shape[:2]) == (1, 16)
            and query.shape[2] == expected_query_tokens
            and query.shape[-1] == 256
            and tuple(document_key.shape[:2]) == (1, 2)
            and document_key.shape[2] == FORMAL_DOCUMENT_TOKENS
            and document_key.shape[-1] == 256
            and tuple(candidate.shape) == (1, expected_query_tokens, 16, 256),
            "formal Qwen3.5 oracle geometry drift",
        )
        _require(running_tokens == FORMAL_QUERY_TOKENS + int(selection["round_index"]), "oracle append shadow length drift")
        _require(visibility_mask is None, "formal no-mask oracle cannot accept a visibility mask")
        expected_key_positions = torch.arange(key.shape[2], dtype=torch.int64)
        expected_query_positions = expected_key_positions[-expected_query_tokens:]
        _require(torch.equal(key_positions.cpu().reshape(-1), expected_key_positions), "formal key positions are not 0..T-1")
        _require(torch.equal(query_positions.cpu().reshape(-1), expected_query_positions), "formal query positions are not the key suffix")
    digests = {
        "query": _tensor_digest(query),
        "key": _tensor_digest(key),
        "value": _tensor_digest(value),
        "document_key": _tensor_digest(document_key),
        "document_value": _tensor_digest(document_value),
        "append_key_shadow": _tensor_digest(append_key),
        "append_value_shadow": _tensor_digest(append_value),
        "candidate_output": _tensor_digest(candidate),
        "query_positions": _tensor_digest(query_positions),
        "key_positions": _tensor_digest(key_positions),
        "visibility_mask": sha256_json(None) if visibility_mask is None else _tensor_digest(visibility_mask),
        "softmax_scale": sha256_json(float(scale)),
        "document_key_value_component": sha256_json([_tensor_digest(document_key), _tensor_digest(document_value)]),
        "append_key_value_shadow_component": sha256_json([_tensor_digest(append_key), _tensor_digest(append_value)]),
        "physical_document_key_blocks": _tensor_digest(key_pool),
        "physical_document_value_blocks": _tensor_digest(value_pool),
        "document_block_table": _tensor_digest(document_table),
        "candidate_active_block_table": _tensor_digest(active_table),
        "source_physical_payload": physical_digest,
    }
    _require(raw.get("input_digests") == digests, "oracle input digest manifest drift")

    observer = raw.get("live_call_observer")
    _require(isinstance(observer, dict), "oracle live call-observer receipt missing")
    observer_required = {
        "schema_version",
        "run_id",
        "rank",
        "resident_count",
        "cell_id",
        "arm_id",
        "kv_policy",
        "gdn_base_policy",
        "sample_id",
        "layer_index",
        "request_index",
        "round_index",
        "ledger_call_sha256",
        "append_capture_id",
        "append_event_manifest",
        "kernel_audit",
        "effective_scaling",
        "softmax_scale_source",
        "input_digests",
        "document_capture",
        "append_shadow_capture",
        "active_block_table_sha256",
    }
    _require(set(observer) == observer_required, "oracle live observer schema drift")
    _require(observer["schema_version"] == "qcomem-live-call-observer-v2", "oracle live observer version drift")
    _require(observer_context is not None, "oracle observer has no validated factorial context")
    expected_arm_id = (
        f"kv={selection['kv_policy']}|gdn={selection['gdn_base_policy']}"
    )
    _require(
        _is_int(observer["rank"])
        and observer["rank"] == rank
        and observer["run_id"] == expected_run_id
        and _is_int(observer["resident_count"])
        and observer["resident_count"] == ORACLE_RESIDENT_COUNT
        and observer["cell_id"] == observer_context.get("cell_id")
        and observer["arm_id"] == expected_arm_id
        and observer["kv_policy"] == selection["kv_policy"]
        and observer["gdn_base_policy"] == selection["gdn_base_policy"]
        and observer["sample_id"] == selection["sample_id"]
        and _is_int(observer["layer_index"])
        and observer["layer_index"] == selection["layer_index"]
        and _is_int(observer["request_index"])
        and observer["request_index"] == selection["request_index"]
        and _is_int(observer["round_index"])
        and observer["round_index"] == selection["round_index"],
        "oracle observer/selection/factorial binding drift",
    )
    ledger_call = observer_context.get("ledger_call")
    _require(isinstance(ledger_call, dict), "validated observer ledger call missing")
    _require(observer["ledger_call_sha256"] == sha256_json(ledger_call), "oracle observer is not the selected ledger call")
    _require(
        isinstance(observer["append_capture_id"], str)
        and bool(observer["append_capture_id"])
        and observer["append_capture_id"] == ledger_call.get("append_capture_id"),
        "oracle observer is not bound to the selected call's pre-write append capture",
    )
    selected_ledger = observer_context.get("selected_ledger")
    _require(isinstance(selected_ledger, dict), "oracle selected request ledger missing")
    selected_calls = selected_ledger.get("calls")
    _require(isinstance(selected_calls, list), "oracle selected ledger calls missing")
    layer_offset = FORMAL_FULL_LAYERS.index(int(selection["layer_index"]))
    expected_append_manifest = []
    for event_index, component in enumerate(append_manifest):
        history_call = selected_calls[
            event_index * len(FORMAL_FULL_LAYERS) + layer_offset
        ]
        _require(
            history_call.get("layer_idx") == selection["layer_index"]
            and history_call.get("request_index") == selection["request_index"]
            and history_call.get("resident_count") == ORACLE_RESIDENT_COUNT
            and history_call.get("append_capture_id") == component["capture_id"],
            "oracle append event is not bound to its historical ledger call",
        )
        history_audit = history_call.get("append_audit")
        _require(
            isinstance(history_audit, dict)
            and history_audit.get("append_event_index") == event_index
            and history_audit.get("appended_tokens_before")
            == component["appended_tokens_before"]
            and history_audit.get("appended_tokens_after")
            == component["appended_tokens_after"]
            and history_audit.get("capture_id") == component["capture_id"],
            "oracle append event/ledger append-audit drift",
        )
        expected_append_manifest.append(
            {
                **component,
                "layer_index": int(selection["layer_index"]),
                "request_index": int(selection["request_index"]),
                "round_index": event_index,
                "ledger_call_sha256": sha256_json(history_call),
            }
        )
    _require(
        observer["append_capture_id"] == capture_ids[-1]
        and observer["append_event_manifest"] == expected_append_manifest,
        "oracle observer did not bind the complete ordered append history",
    )
    kernel_audit = observer["kernel_audit"]
    _require(isinstance(kernel_audit, dict), "oracle observer kernel audit missing")
    # The observer is loaded from oracle-raw.json, while the selected ledger
    # is still the live in-memory receipt during producer self-replay.  JSON
    # turns tuple-valued shape fields into lists, so compare their canonical
    # JSON projections instead of Python container identity/typing.
    ledger_kernel_audit = {
        key: ledger_call.get(key) for key in kernel_audit
    }
    _require(
        canonical_json_bytes(ledger_kernel_audit)
        == canonical_json_bytes(kernel_audit),
        "oracle observer kernel audit differs from the selected ledger call",
    )
    observed_scaling = observer["effective_scaling"]
    _require(
        isinstance(observed_scaling, (int, float))
        and not isinstance(observed_scaling, bool)
        and math.isfinite(observed_scaling)
        and observed_scaling > 0,
        "oracle observer effective scaling is not a finite positive number",
    )
    _require(
        float(observed_scaling)
        == float(scale)
        == float(kernel_audit.get("softmax_scale"))
        == float(ledger_call.get("softmax_scale")),
        "oracle scale is not the live kernel's effective scale",
    )
    _require(
        observer["softmax_scale_source"]
        == "MultiForkHitLedger.call_observer.kernel_audit.softmax_scale",
        "oracle scale source drift",
    )
    _require(observer["input_digests"] == digests, "oracle observer/input digest disagreement")
    document_capture = observer["document_capture"]
    _require(
        document_capture
        == {
            "capture_point": "persistent-document-arena-via-document-block-table",
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
            "source_physical_payload_sha256": physical_digest,
        },
        "oracle persistent document capture binding drift",
    )
    _require(
        physical_digest == observer_context.get("source_physical_payload_sha256"),
        "oracle physical document sidecars differ from factorial source payload",
    )
    append_capture = observer["append_shadow_capture"]
    _require(isinstance(append_capture, dict), "oracle append-shadow capture missing")
    _require(
        append_capture.get("capture_point")
        == "cache-layer-update-before-sequence-append"
        and append_capture.get("independent_of_candidate_active_block_table") is True
        and append_capture.get("events") == expected_append_manifest
        and append_capture.get("key_sha256") == digests["append_key_shadow"]
        and append_capture.get("value_sha256") == digests["append_value_shadow"],
        "oracle append-shadow capture binding drift",
    )
    _require(
        observer["active_block_table_sha256"]
        == digests["candidate_active_block_table"],
        "oracle active block-table sidecar digest drift",
    )

    # The oracle module owns the reference-only IEEE context and proves full
    # backend restoration.  The runner must not nest a second global context.
    prereg = OraclePreregistration(
        OracleThresholds(max_relative_l2=ORACLE_MAX_RELATIVE_L2)
    )
    outcome = prereg.evaluate_attention(
        query,
        key,
        value,
        candidate,
        query_positions=query_positions,
        key_positions=key_positions,
        visibility_mask=visibility_mask,
        scaling=float(scale),
    ).to_dict()
    _require(outcome["status"] == "completed", "oracle replay invalid")
    _require(outcome["thresholds"]["max_relative_l2"] == ORACLE_MAX_RELATIVE_L2, "oracle threshold drift")
    _validate_oracle_precision_audit(
        outcome["reference"]["precision_audit"], label="blind oracle"
    )
    recorded = raw.get("recorded_outcome")
    _require(isinstance(recorded, dict), "producer oracle outcome missing")
    _require(recorded.get("schema_version") == ORACLE_SCHEMA_VERSION, "producer oracle schema drift")
    _require(recorded.get("status") == "completed", "producer oracle did not complete")
    _require(recorded.get("thresholds") == outcome["thresholds"], "producer oracle thresholds drift")
    _require(recorded.get("preregistration_sha256") == outcome["preregistration_sha256"], "producer oracle threshold preregistration drift")
    producer_reference = recorded.get("reference")
    _require(isinstance(producer_reference, dict), "producer oracle reference receipt missing")
    _validate_oracle_precision_audit(
        producer_reference.get("precision_audit"), label="producer oracle"
    )
    _require(
        recorded == outcome,
        "producer oracle outcome differs from blind canonical recomputation",
    )
    return outcome, [loaded.binding, *tensor_bindings]


def _enum_value(enum_type: type[Enum], value: Any, label: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ReviewAuditError(f"{label} enum drift") from exc


def _parse_mutation_receipt(value: Any) -> MutationReceipt:
    _require(isinstance(value, dict), "mutation lifecycle receipt missing")
    required = {
        "mutant_id",
        "injection_stage",
        "injector_factory_started",
        "injector_factory_completed",
        "injector_enter_started",
        "injector_enter_completed",
        "mutation_applied",
        "injector_exit_started",
        "injector_exit_completed",
        "restoration_verifier_present",
        "restoration_verified",
        "target_mutation_binding",
    }
    _require(set(value) == required, "mutation lifecycle receipt schema drift")
    flags = required - {
        "mutant_id",
        "injection_stage",
        "target_mutation_binding",
    }
    _require(all(type(value[field]) is bool for field in flags), "mutation lifecycle flag type drift")
    return MutationReceipt(
        mutant_id=value["mutant_id"],
        injection_stage=_enum_value(InjectionStage, value["injection_stage"], "receipt stage"),
        target_mutation_binding=value["target_mutation_binding"],
        **{field: value[field] for field in flags},
    )


def _parse_campaign_outcome(value: Any) -> CampaignOutcome:
    _require(isinstance(value, dict), "campaign outcome must be object")
    required = {
        "phase",
        "classification",
        "mutant_id",
        "mutant_name",
        "injection_stage",
        "expected_gate_id",
        "observed_gate_id",
        "boundary_gate_id",
        "detector_satisfied",
        "aggregate_eligible",
        "scientifically_valid",
        "mutation_receipt",
        "exercise_started",
        "exercise_completed",
        "restoration_verified",
        "failure_origin",
        "error_type",
        "error_message",
    }
    _require(set(value) == required, "campaign outcome schema drift")
    phase = _enum_value(CampaignPhase, value["phase"], "campaign phase")
    classification = _enum_value(
        OutcomeClassification, value["classification"], "campaign classification"
    )
    stage = None if value["injection_stage"] is None else _enum_value(
        InjectionStage, value["injection_stage"], "campaign stage"
    )
    failure = None if value["failure_origin"] is None else _enum_value(
        ExecutionBoundary, value["failure_origin"], "campaign failure origin"
    )
    receipt = None if value["mutation_receipt"] is None else _parse_mutation_receipt(
        value["mutation_receipt"]
    )
    for field in (
        "detector_satisfied",
        "aggregate_eligible",
        "scientifically_valid",
        "exercise_started",
        "exercise_completed",
    ):
        _require(type(value[field]) is bool, f"campaign {field} type drift")
    _require(
        value["restoration_verified"] is None
        or type(value["restoration_verified"]) is bool,
        "campaign restoration flag type drift",
    )
    return CampaignOutcome(
        phase=phase,
        classification=classification,
        mutant_id=value["mutant_id"],
        mutant_name=value["mutant_name"],
        injection_stage=stage,
        expected_gate_id=value["expected_gate_id"],
        observed_gate_id=value["observed_gate_id"],
        boundary_gate_id=value["boundary_gate_id"],
        detector_satisfied=value["detector_satisfied"],
        aggregate_eligible=value["aggregate_eligible"],
        scientifically_valid=value["scientifically_valid"],
        mutation_receipt=receipt,
        exercise_started=value["exercise_started"],
        exercise_completed=value["exercise_completed"],
        restoration_verified=value["restoration_verified"],
        failure_origin=failure,
        error_type=value["error_type"],
        error_message=value["error_message"],
    )


def _validate_receipt_target_binding(
    outcome: CampaignOutcome,
    *,
    mutant_id: str,
    case_cell_id: str,
) -> dict[str, Any]:
    """Validate target evidence emitted by the injector lifecycle itself.

    Formal cases deliberately have no sibling ``target_witness`` field: that
    would let a producer attach an unrelated descriptor after the mutation.
    The only accepted evidence is the v2 binding finalized by the same
    ``AppliedMutation`` during restoration and embedded in its receipt.
    """

    receipt = outcome.mutation_receipt
    _require(receipt is not None, f"mutant {mutant_id} lifecycle receipt missing")
    expected_kind, expected_field = MUTANT_TARGET_CONTRACT[mutant_id]
    try:
        return validate_target_mutation_binding(
            receipt.target_mutation_binding,
            mutant_id=mutant_id,
            case_cell_id=case_cell_id,
            target_kind=expected_kind,
            target_field=expected_field,
        )
    except FaultCampaignConfigurationError as exc:
        raise ReviewAuditError(
            f"mutant {mutant_id} target mutation binding rejected: {exc}"
        ) from exc


def _validate_clean_outcome(outcome: CampaignOutcome) -> bool:
    report = validate_campaign_outcomes(outcome, {})
    _require("clean" not in report["binding_errors"], "clean campaign outcome binding failed")
    if outcome.classification is OutcomeClassification.CLEAN_PASS:
        _require(
            outcome.observed_gate_id is None
            and outcome.boundary_gate_id is None
            and outcome.failure_origin is None
            and outcome.error_type is None
            and outcome.error_message is None,
            "clean pass carries contradictory error fields",
        )
        return True
    _require(
        outcome.classification is OutcomeClassification.CLEAN_FALSE_POSITIVE
        and isinstance(outcome.observed_gate_id, str)
        and bool(outcome.observed_gate_id)
        and outcome.detector_satisfied is False
        and outcome.aggregate_eligible is False
        and outcome.scientifically_valid is True
        and outcome.exercise_started is True
        and outcome.exercise_completed is False
        and outcome.restoration_verified is None
        and outcome.failure_origin is ExecutionBoundary.DETECTOR_EXERCISE
        and outcome.error_type == "RuntimeInvariantError"
        and isinstance(outcome.error_message, str)
        and bool(outcome.error_message),
        "clean control was neither a valid pass nor a classified false positive",
    )
    return False


def _validate_mutant_error_contract(outcome: CampaignOutcome) -> None:
    """Replay classification-dependent exception/null contracts exactly."""

    classification = outcome.classification
    if classification in {
        OutcomeClassification.DETECTED_EXPECTED_GATE,
        OutcomeClassification.DETECTED_WRONG_GATE,
    }:
        _require(
            outcome.failure_origin is ExecutionBoundary.DETECTOR_EXERCISE
            and outcome.error_type == "RuntimeInvariantError"
            and isinstance(outcome.error_message, str)
            and bool(outcome.error_message)
            and isinstance(outcome.observed_gate_id, str)
            and bool(outcome.observed_gate_id)
            and outcome.boundary_gate_id is None,
            "mutant detector classification/error contract drift",
        )
    elif classification is OutcomeClassification.ESCAPED:
        _require(
            outcome.failure_origin is None
            and outcome.error_type is None
            and outcome.error_message is None
            and outcome.observed_gate_id is None
            and outcome.boundary_gate_id is None,
            "escaped mutant carries contradictory error fields",
        )
    elif classification is OutcomeClassification.UNEXPECTED_CRASH:
        _require(
            outcome.failure_origin is not None
            and isinstance(outcome.error_type, str)
            and bool(outcome.error_type)
            and isinstance(outcome.error_message, str)
            and bool(outcome.error_message),
            "unexpected mutant crash lacks exact failure evidence",
        )
    else:
        raise ReviewAuditError("mutant classification is not admissible")


def _validate_mutant_full_forward_completion(
    value: Any, *, mutant_id: str, expected_request_index: int = 0
) -> None:
    required = {
        "schema_version",
        "verified",
        "request_index",
        "resident_count",
        "request_policy",
        "initial_query_tokens",
        "total_calls",
        "counts",
        "same_unified_attention_kernel",
        "dense_fallback_calls",
        "full_kv_concatenations",
        "kernel_identity",
        "calls",
    }
    _require(
        isinstance(value, dict) and set(value) == required,
        f"mutant {mutant_id} full-forward coverage schema drift",
    )
    _require(
        value["schema_version"] == "forkaudit-mutant-full-forward-coverage-v1"
        and value["verified"] is True
        and _is_int(value["request_index"])
        and value["request_index"] == expected_request_index
        and _is_int(value["resident_count"])
        and value["resident_count"] == 2
        and value["request_policy"] == SHARED_REUSE
        and _is_int(value["initial_query_tokens"])
        and value["initial_query_tokens"] == FORMAL_QUERY_TOKENS
        and value["same_unified_attention_kernel"] is True
        and _is_int(value["dense_fallback_calls"])
        and value["dense_fallback_calls"] == 0
        and _is_int(value["full_kv_concatenations"])
        and value["full_kv_concatenations"] == 0,
        f"mutant {mutant_id} full-forward summary drift",
    )
    expected_count_keys = {str(layer) for layer in FORMAL_FULL_LAYERS}
    _require(
        isinstance(value["counts"], dict)
        and set(value["counts"]) == expected_count_keys
        and all(_is_int(item) and item == 1 for item in value["counts"].values()),
        f"mutant {mutant_id} did not traverse every full-attention layer once",
    )
    _require(
        value["kernel_identity"]
        == {
            "module": FORMAL_KERNEL_DESCRIPTOR[0],
            "qualname": FORMAL_KERNEL_DESCRIPTOR[1],
            "signature": FORMAL_KERNEL_DESCRIPTOR[2],
        },
        f"mutant {mutant_id} full-forward kernel descriptor drift",
    )
    calls = value["calls"]
    call_fields = {
        "layer_idx",
        "query_tokens",
        "kv_tokens",
        "kernel_mode",
        "quantization",
        "mask_contract",
        "position_ids_contract",
        "position_ids_expected_tail_start",
        "position_ids_expected_tail_end_exclusive",
        "softmax_scale",
        "append_event_index",
        "appended_tokens_before",
        "appended_tokens_after",
    }
    _require(
        _is_int(value["total_calls"])
        and value["total_calls"] == len(FORMAL_FULL_LAYERS)
        and isinstance(calls, list)
        and len(calls) == value["total_calls"],
        f"mutant {mutant_id} full-forward call budget drift",
    )
    for layer_index, call in zip(FORMAL_FULL_LAYERS, calls):
        scientific_integer_fields = {
            "layer_idx",
            "query_tokens",
            "kv_tokens",
            "position_ids_expected_tail_start",
            "position_ids_expected_tail_end_exclusive",
            "append_event_index",
            "appended_tokens_before",
            "appended_tokens_after",
        }
        _require(
            isinstance(call, dict)
            and set(call) == call_fields
            and all(_is_int(call[field]) for field in scientific_integer_fields)
            and call["layer_idx"] == layer_index
            and call["query_tokens"] == FORMAL_QUERY_TOKENS
            and call["kv_tokens"] == FORMAL_DOCUMENT_TOKENS + FORMAL_QUERY_TOKENS
            and call["kernel_mode"] == FORMAL_KERNEL_MODE
            and call["quantization"] == "Q16"
            and call["mask_contract"] == FORMAL_MASK_CONTRACT
            and call["position_ids_contract"] == FORMAL_POSITION_CONTRACT
            and call["position_ids_expected_tail_start"] == FORMAL_DOCUMENT_TOKENS
            and call["position_ids_expected_tail_end_exclusive"]
            == FORMAL_DOCUMENT_TOKENS + FORMAL_QUERY_TOKENS
            and call["softmax_scale"] == FORMAL_SOFTMAX_SCALE
            and call["append_event_index"] == 0
            and call["appended_tokens_before"] == 0
            and call["appended_tokens_after"] == FORMAL_QUERY_TOKENS,
            f"mutant {mutant_id} full-forward call projection drift",
        )


def _validate_mutant_direct_completion(value: Any, *, mutant_id: str) -> None:
    required = {
        "schema_version",
        "call_count_before",
        "call_count_after",
        "layer_index",
        "request_index",
        "resident_count",
        "request_policy",
        "query_tokens",
        "kv_tokens",
        "kernel_mode",
        "quantization",
        "mask_contract",
        "position_ids_contract",
        "position_ids_expected_tail_start",
        "position_ids_expected_tail_end_exclusive",
        "softmax_scale",
        "append_event_index",
        "appended_tokens_before",
        "appended_tokens_after",
        "kernel_identity",
    }
    _require(
        isinstance(value, dict)
        and set(value) == required
        and value["schema_version"] == "forkaudit-mutant-direct-call-coverage-v1"
        and all(
            _is_int(value[field])
            for field in (
                "call_count_before",
                "call_count_after",
                "layer_index",
                "request_index",
                "resident_count",
                "query_tokens",
                "kv_tokens",
                "position_ids_expected_tail_start",
                "position_ids_expected_tail_end_exclusive",
                "append_event_index",
                "appended_tokens_before",
                "appended_tokens_after",
            )
        )
        and value["call_count_before"] == 0
        and value["call_count_after"] == 1
        and value["layer_index"] == FORMAL_FULL_LAYERS[0]
        and value["request_index"] == 0
        and value["resident_count"] == 2
        and value["request_policy"] == SHARED_REUSE
        and value["query_tokens"] == 1
        and value["kv_tokens"] == FORMAL_DOCUMENT_TOKENS + 1
        and value["kernel_mode"] == FORMAL_KERNEL_MODE
        and value["quantization"] == "Q16"
        and value["mask_contract"] == FORMAL_MASK_CONTRACT
        and value["position_ids_contract"] == FORMAL_POSITION_CONTRACT
        and value["position_ids_expected_tail_start"] == FORMAL_DOCUMENT_TOKENS
        and value["position_ids_expected_tail_end_exclusive"]
        == FORMAL_DOCUMENT_TOKENS + 1
        and value["softmax_scale"] == FORMAL_SOFTMAX_SCALE
        and value["append_event_index"] == 0
        and value["appended_tokens_before"] == 0
        and value["appended_tokens_after"] == 1
        and value["kernel_identity"]
        == {
            "module": FORMAL_KERNEL_DESCRIPTOR[0],
            "qualname": FORMAL_KERNEL_DESCRIPTOR[1],
            "signature": FORMAL_KERNEL_DESCRIPTOR[2],
        },
        f"mutant {mutant_id} direct-call completion drift",
    )


def _validate_mutant_detector_input(
    value: Any,
    *,
    mutant_id: str,
    mutation_activated: bool,
    outcome: CampaignOutcome,
    expected_query_sha256: str,
) -> dict[str, Any] | None:
    required = {
        "schema_version",
        "mutant_id",
        "detector_path",
        "expected_gate_id",
        "resident_count",
        "kv_policy",
        "gdn_base_policy",
        "evidence",
    }
    _require(
        isinstance(value, dict) and set(value) == required,
        f"mutant {mutant_id} detector input schema drift",
    )
    _require(
        value["schema_version"] == "forkaudit-mutant-detector-input-v2"
        and value["mutant_id"] == mutant_id
        and value["detector_path"] == MUTANT_EXERCISE_PATHS[mutant_id]
        and value["expected_gate_id"] == MUTANT_SPECS[mutant_id].expected_gate_id
        and _is_int(value["resident_count"])
        and value["resident_count"] == 2
        and value["kv_policy"] == SHARED_REUSE
        and value["gdn_base_policy"] == GDN_BORROW_IMMUTABLE_BASE,
        f"mutant {mutant_id} detector input identity drift",
    )
    evidence = value["evidence"]
    _require(isinstance(evidence, dict), f"mutant {mutant_id} detector evidence missing")
    if mutant_id in ("M1", "M3"):
        common = {
            "kind",
            "require_appended_tail_cow",
            "full_attention_layers",
        }
        extra = (
            {
                "target_reservations_sha256",
                "peer_request0_reservations_sha256",
                "target_descriptor",
                "target_descriptor_sha256",
                "construction_guard_row_count",
            }
            if mutant_id == "M1"
            else {
                "appended_tokens_by_request_layer",
                "all_request_layers_appended_once",
            }
        )
        _require(set(evidence) == common | extra, f"mutant {mutant_id} KV input schema drift")
        _require(
            evidence["kind"] == "live-kv-ownership"
            and evidence["require_appended_tail_cow"] is (mutant_id == "M3")
            and evidence["full_attention_layers"] == list(FORMAL_FULL_LAYERS),
            f"mutant {mutant_id} KV detector contract drift",
        )
        if mutant_id == "M1":
            target_sha = _require_sha256(
                evidence["target_reservations_sha256"], "M1 reservation digest"
            )
            peer_sha = _require_sha256(
                evidence["peer_request0_reservations_sha256"],
                "M1 peer reservation digest",
            )
            descriptor = evidence["target_descriptor"]
            expected_role = (
                "request0-reservation-aliased-into-request1"
                if mutation_activated
                else "request1-construction-reservation"
            )
            _require(
                isinstance(descriptor, dict)
                and set(descriptor)
                == {
                    "schema_version",
                    "logical_slot",
                    "binding_role",
                    "dtype",
                    "shape",
                    "tensor_sha256",
                }
                and descriptor["schema_version"]
                == "forkaudit-live-target-tensor-v1"
                and descriptor["logical_slot"]
                == f"request1/layer{FORMAL_FULL_LAYERS[0]}/reservations"
                and descriptor["binding_role"] == expected_role
                and isinstance(descriptor["shape"], list)
                and all(_is_int(item) for item in descriptor["shape"])
                and descriptor["tensor_sha256"] == target_sha
                and evidence["target_descriptor_sha256"]
                == sha256_json(descriptor),
                "M1 target descriptor/digest drift",
            )
            if mutation_activated:
                receipt = outcome.mutation_receipt
                _require(
                    target_sha == peer_sha
                    and receipt is not None
                    and receipt.target_mutation_binding is not None
                    and receipt.target_mutation_binding.get("mutated_sha256")
                    == evidence["target_descriptor_sha256"],
                    "M1 detector input is not the mutation receipt's peer alias",
                )
            else:
                _require(
                    target_sha != peer_sha,
                    "M1 matched control already aliases the peer reservation",
                )
            _require(
                _is_int(evidence["construction_guard_row_count"])
                and evidence["construction_guard_row_count"]
                == 2 * len(FORMAL_FULL_LAYERS),
                "M1 construction guard coverage drift",
            )
        else:
            expected_rows = [
                {"request_index": request, "layer_index": layer, "appended_tokens": 1}
                for request in range(2)
                for layer in FORMAL_FULL_LAYERS
            ]
            _require(
                all(
                    isinstance(row, dict)
                    and all(
                        _is_int(row.get(field))
                        for field in (
                            "request_index",
                            "layer_index",
                            "appended_tokens",
                        )
                    )
                    for row in evidence["appended_tokens_by_request_layer"]
                )
                and evidence["appended_tokens_by_request_layer"] == expected_rows
                and evidence["all_request_layers_appended_once"] is True,
                "M3 matched detector did not append every request/layer once",
            )
    elif mutant_id in ("M2", "M8", "M9"):
        _require(
            set(evidence)
            == {
                "kind",
                "backend_registered",
                "request_index",
                "initial_query_tokens",
                "expected_calls_per_layer",
                "expected_full_attention_layers",
                "call_count_before",
                "query_token_ids_sha256",
            }
            and evidence["kind"] == "live-full-model-forward-ledger"
            and evidence["backend_registered"] is True
            and _is_int(evidence["request_index"])
            and evidence["request_index"] == 0
            and _is_int(evidence["initial_query_tokens"])
            and evidence["initial_query_tokens"] == FORMAL_QUERY_TOKENS
            and _is_int(evidence["expected_calls_per_layer"])
            and evidence["expected_calls_per_layer"] == 1
            and evidence["expected_full_attention_layers"] == list(FORMAL_FULL_LAYERS)
            and _is_int(evidence["call_count_before"])
            and evidence["call_count_before"] == 0,
            f"mutant {mutant_id} full-model detector input drift",
        )
        _require(
            evidence["query_token_ids_sha256"] == expected_query_sha256,
            f"mutant {mutant_id} query differs from frozen query-bank row zero",
        )
    elif mutant_id in ("M4", "M5"):
        _require(
            set(evidence)
            == {
                "kind",
                "phase",
                "completed_request_indices",
                "storage_witness",
                "storage_witness_sha256",
                "transition_forward_ledgers",
            }
            and evidence["kind"] == "live-gdn-storage-replay"
            and evidence["phase"] == PHASE_POST_TRANSITION
            and isinstance(evidence["completed_request_indices"], list)
            and all(
                _is_int(item)
                for item in evidence["completed_request_indices"]
            )
            and evidence["completed_request_indices"]
            == ([0] if mutant_id == "M4" else [0, 1])
            and evidence["storage_witness_sha256"]
            == sha256_json(evidence["storage_witness"]),
            f"mutant {mutant_id} GDN replay input drift",
        )
        transition_ledgers = evidence["transition_forward_ledgers"]
        completed_indices = [0] if mutant_id == "M4" else [0, 1]
        _require(
            isinstance(transition_ledgers, list)
            and len(transition_ledgers) == len(completed_indices),
            f"mutant {mutant_id} transition-forward ledger coverage drift",
        )
        for request_index, ledger in zip(completed_indices, transition_ledgers):
            _validate_mutant_full_forward_completion(
                ledger,
                mutant_id=mutant_id,
                expected_request_index=request_index,
            )
        try:
            replayed = replay_gdn_storage_witness(evidence["storage_witness"])
        except GDNStorageWitnessError as exc:
            _require(
                outcome.observed_gate_id == exc.gate_id
                and outcome.failure_origin is ExecutionBoundary.DETECTOR_EXERCISE,
                f"mutant {mutant_id} raw GDN replay disagrees with detector outcome",
            )
            _require(
                (
                    mutation_activated
                    and outcome.classification
                    in {
                        OutcomeClassification.DETECTED_EXPECTED_GATE,
                        OutcomeClassification.DETECTED_WRONG_GATE,
                    }
                )
                or (
                    not mutation_activated
                    and outcome.classification
                    is OutcomeClassification.CLEAN_FALSE_POSITIVE
                ),
                f"mutant {mutant_id} raw GDN gate has no admissible outcome class",
            )
            return None
        return replayed
    else:
        import torch

        _require(
            set(evidence)
            == {
                "kind",
                "layer_index",
                "call_count_before",
                "appended_tokens",
                "query_sha256",
                "query_dtype",
                "query_shape",
                "position_ids_values",
                "position_ids_sha256",
                "attention_mask_representation",
                "attention_mask_sha256",
                "attention_mask_dtype",
                "attention_mask_shape",
            }
            and evidence["kind"] == "live-direct-ledger-call"
            and _is_int(evidence["layer_index"])
            and evidence["layer_index"] == FORMAL_FULL_LAYERS[0]
            and _is_int(evidence["call_count_before"])
            and evidence["call_count_before"] == 0
            and _is_int(evidence["appended_tokens"])
            and evidence["appended_tokens"] == 1,
            f"mutant {mutant_id} direct detector input drift",
        )
        _require_sha256(evidence["query_sha256"], f"mutant {mutant_id} direct query digest")
        expected_direct_query = torch.zeros(
            (1, FORMAL_NUM_QUERY_HEADS, 1, FORMAL_HEAD_DIM),
            dtype=torch.bfloat16,
        )
        _require(
            evidence["query_dtype"] == "torch.bfloat16"
            and isinstance(evidence["query_shape"], list)
            and all(_is_int(item) for item in evidence["query_shape"])
            and evidence["query_shape"] == list(expected_direct_query.shape)
            and evidence["query_sha256"] == _tensor_digest(expected_direct_query),
            f"mutant {mutant_id} direct query differs from fixed BF16 zero probe",
        )
        expected_position = FORMAL_DOCUMENT_TOKENS + (
            1 if mutation_activated and mutant_id == "M6" else 0
        )
        _require(
            isinstance(evidence["position_ids_values"], list)
            and all(_is_int(item) for item in evidence["position_ids_values"])
            and evidence["position_ids_values"] == [expected_position]
            and evidence["position_ids_sha256"]
            == _tensor_digest(torch.tensor([[expected_position]], dtype=torch.int64)),
            f"mutant {mutant_id} direct position input drift",
        )
        materialized = mutation_activated and mutant_id == "M7"
        _require(
            evidence["attention_mask_representation"]
            == ("materialized-tensor" if materialized else "none")
            and evidence["attention_mask_dtype"]
            == ("torch.bool" if materialized else None)
            and evidence["attention_mask_shape"]
            == ([1, 1, 1, FORMAL_DOCUMENT_TOKENS + 1] if materialized else None),
            f"mutant {mutant_id} direct mask input drift",
        )
        if evidence["attention_mask_shape"] is not None:
            _require(
                isinstance(evidence["attention_mask_shape"], list)
                and all(
                    _is_int(item) for item in evidence["attention_mask_shape"]
                ),
                f"mutant {mutant_id} direct mask shape type drift",
            )
        if materialized:
            expected_mask = torch.ones(
                (1, 1, 1, FORMAL_DOCUMENT_TOKENS + 1), dtype=torch.bool
            )
            _require(
                evidence["attention_mask_sha256"] == _tensor_digest(expected_mask),
                "M7 materialized mask digest drift",
            )
        else:
            _require(evidence["attention_mask_sha256"] is None, f"mutant {mutant_id} unexpected mask digest")
    return None


def _validate_exercise_coverage_receipt(
    value: Any,
    *,
    mutant_id: str,
    outcome: CampaignOutcome,
    mutation_activated: bool,
    expected_query_sha256: str,
) -> None:
    required = {
        "schema_version",
        "mutant_id",
        "mutation_activated",
        "exercise_contract_sha256",
        "detector_path",
        "exercise_started",
        "detector_input",
        "detector_input_sha256",
        "detector_path_completed",
        "completion_receipt",
        "completion_receipt_sha256",
        "outcome_classification",
        "observed_gate_id",
    }
    _require(
        isinstance(value, dict) and set(value) == required,
        f"mutant {mutant_id} exercise coverage schema drift",
    )
    _require(
        value["schema_version"] == "forkaudit-mutant-exercise-coverage-v2"
        and value["mutant_id"] == mutant_id
        and value["mutation_activated"] is mutation_activated
        and value["exercise_contract_sha256"]
        == _mutant_exercise_contract_sha256(mutant_id)
        and value["detector_path"] == MUTANT_EXERCISE_PATHS[mutant_id],
        f"mutant {mutant_id} exercise coverage identity drift",
    )
    _require(
        type(value["exercise_started"]) is bool
        and type(value["detector_path_completed"]) is bool
        and value["exercise_started"] is outcome.exercise_started
        and value["detector_path_completed"] is outcome.exercise_completed
        and value["outcome_classification"] == outcome.classification.value
        and value["observed_gate_id"] == outcome.observed_gate_id,
        f"mutant {mutant_id} exercise coverage/outcome disagreement",
    )
    detector_input = value["detector_input"]
    replayed_gdn = None
    if value["exercise_started"]:
        _require(
            value["detector_input_sha256"] == sha256_json(detector_input),
            f"mutant {mutant_id} detector input SHA drift",
        )
        replayed_gdn = _validate_mutant_detector_input(
            detector_input,
            mutant_id=mutant_id,
            mutation_activated=mutation_activated,
            outcome=outcome,
            expected_query_sha256=expected_query_sha256,
        )
    else:
        _require(
            detector_input is None and value["detector_input_sha256"] is None,
            f"mutant {mutant_id} has detector input without starting exercise",
        )
    completion = value["completion_receipt"]
    if value["detector_path_completed"]:
        _require(
            isinstance(completion, dict)
            and value["completion_receipt_sha256"] == sha256_json(completion),
            f"mutant {mutant_id} completion receipt missing or changed",
        )
        if mutant_id in ("M1", "M3"):
            _require(
                completion
                == {
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
                },
                f"mutant {mutant_id} KV detector completion drift",
            )
        elif mutant_id in ("M2", "M8", "M9"):
            _validate_mutant_full_forward_completion(completion, mutant_id=mutant_id)
        elif mutant_id in ("M4", "M5"):
            _require(
                replayed_gdn is not None and completion == replayed_gdn,
                f"mutant {mutant_id} GDN completion differs from blind replay",
            )
        else:
            _validate_mutant_direct_completion(completion, mutant_id=mutant_id)
    else:
        _require(
            completion is None and value["completion_receipt_sha256"] is None,
            f"mutant {mutant_id} incomplete detector carries a completion receipt",
        )


def _validate_matched_injected_coverage_pair(
    matched: Mapping[str, Any],
    injected: Mapping[str, Any],
    *,
    mutant_id: str,
) -> None:
    """Prove the separately rebuilt cases differ only at the injected target."""

    matched_input = matched.get("detector_input")
    injected_input = injected.get("detector_input")
    _require(
        isinstance(matched_input, dict) and isinstance(injected_input, dict),
        f"mutant {mutant_id} matched/injected detector input missing",
    )
    matched_common = {key: value for key, value in matched_input.items() if key != "evidence"}
    injected_common = {key: value for key, value in injected_input.items() if key != "evidence"}
    _require(
        matched_common == injected_common,
        f"mutant {mutant_id} matched/injected detector contract drift",
    )
    left = dict(matched_input["evidence"])
    right = dict(injected_input["evidence"])
    if mutant_id == "M1":
        differing = {
            "target_reservations_sha256",
            "target_descriptor",
            "target_descriptor_sha256",
        }
    elif mutant_id in ("M4", "M5"):
        differing = {"storage_witness", "storage_witness_sha256"}
    elif mutant_id == "M6":
        differing = {"position_ids_values", "position_ids_sha256"}
    elif mutant_id == "M7":
        differing = {
            "attention_mask_representation",
            "attention_mask_sha256",
            "attention_mask_dtype",
            "attention_mask_shape",
        }
    else:
        differing = set()
    _require(
        {key: value for key, value in left.items() if key not in differing}
        == {key: value for key, value in right.items() if key not in differing},
        f"mutant {mutant_id} matched/injected exercises differ outside the target",
    )
    matched_is_clean_false_positive = (
        matched.get("outcome_classification")
        == OutcomeClassification.CLEAN_FALSE_POSITIVE.value
    )
    if differing and not (
        mutant_id in ("M4", "M5") and matched_is_clean_false_positive
    ):
        _require(
            all(left.get(key) != right.get(key) for key in differing),
            f"mutant {mutant_id} injected target did not differ from matched control",
        )


def _validate_fault_campaign(
    shard: Mapping[str, Any],
    *,
    rank: int,
    seen_case_ids: set[str],
    expected_query_sha256: str,
    expected_frozen_baseline: Mapping[str, int] | None = None,
) -> tuple[
    CampaignOutcome,
    dict[str, CampaignOutcome],
    dict[str, CampaignOutcome],
]:
    expected_query_sha256 = _require_sha256(
        expected_query_sha256, "fault-campaign frozen query digest"
    )
    campaign = shard.get("fault_campaign")
    _require(isinstance(campaign, dict), "fault campaign missing")
    _require(
        set(campaign) == {"assignment", "clean_case", "mutants"},
        "fault campaign schema drift",
    )
    _require(
        campaign.get("assignment") == list(MUTANT_ASSIGNMENT_BY_RANK[rank]),
        "rank mutant assignment drift",
    )
    clean_case = campaign.get("clean_case")
    _require(
        isinstance(clean_case, dict)
        and set(clean_case)
        == {"case_cell_id", "case_isolation", "cleanup_receipt", "outcome"},
        "global clean case schema drift",
    )
    clean_cell_id = clean_case["case_cell_id"]
    _require(
        isinstance(clean_cell_id, str)
        and bool(clean_cell_id)
        and clean_cell_id not in seen_case_ids,
        "global clean case cell reused",
    )
    seen_case_ids.add(clean_cell_id)
    expected_isolation = {
        "fresh_document_cache_built": True,
        "fresh_request_cache_built": True,
        "cache_reused_from_prior_case": False,
        "cache_discarded_after_case": True,
    }
    _require(
        clean_case["case_isolation"] == expected_isolation,
        "global clean control did not use a fresh disposable cache",
    )
    campaign_baseline = _validate_cleanup_receipt(
        clean_case["cleanup_receipt"], label="global clean mutant control"
    )
    if expected_frozen_baseline is not None:
        _require(
            campaign_baseline == expected_frozen_baseline,
            "fault campaign baseline differs from factorial baseline",
        )
    clean = _parse_campaign_outcome(clean_case.get("outcome"))
    _validate_clean_outcome(clean)
    rows = campaign.get("mutants")
    _require(isinstance(rows, dict), "mutant rows must be keyed by mutant ID")
    _require(set(rows) == set(MUTANT_ASSIGNMENT_BY_RANK[rank]), "missing/extra rank mutant row")
    outcomes: dict[str, CampaignOutcome] = {}
    matched_outcomes: dict[str, CampaignOutcome] = {}
    for key, case in rows.items():
        _require(isinstance(case, dict), f"mutant {key} case missing")
        _require(
            set(case)
            == {
                "mutant_id",
                "exercise_mutant_id",
                "exercise_contract_sha256",
                "exercise_coverage_receipt",
                "case_cell_id",
                "case_isolation",
                "cleanup_receipt",
                "outcome",
                "matched_clean",
                "matched_clean_exercise_passed",
            },
            f"mutant {key} case schema drift",
        )
        _require(case["mutant_id"] == key, f"mutant dict key/id mismatch for {key}")
        _require(key in MUTANT_SPECS, f"unknown mutant {key}")
        expected_exercise_sha = _mutant_exercise_contract_sha256(key)
        _require(
            case["exercise_mutant_id"] == key
            and case["exercise_contract_sha256"] == expected_exercise_sha,
            f"mutant {key} exercise contract drift",
        )
        cell_id = case["case_cell_id"]
        _require(isinstance(cell_id, str) and cell_id and cell_id not in seen_case_ids, "mutant case cell reused")
        seen_case_ids.add(cell_id)
        _require(
            case["case_isolation"] == expected_isolation,
            f"mutant {key} did not use a fresh disposable cache",
        )
        _require(
            _validate_cleanup_receipt(
                case["cleanup_receipt"], label=f"mutant {key}"
            )
            == campaign_baseline,
            f"mutant {key} cleanup baseline drift",
        )
        matched = case["matched_clean"]
        _require(isinstance(matched, dict), f"mutant {key} matched clean missing")
        _require(
            set(matched)
            == {
                "exercise_mutant_id",
                "exercise_contract_sha256",
                "exercise_coverage_receipt",
                "case_cell_id",
                "case_isolation",
                "cleanup_receipt",
                "outcome",
            },
            f"mutant {key} matched clean schema drift",
        )
        _require(
            matched["exercise_mutant_id"] == key
            and matched["exercise_contract_sha256"] == expected_exercise_sha
            and matched["exercise_contract_sha256"]
            == case["exercise_contract_sha256"],
            f"mutant {key} matched exercise contract drift",
        )
        matched_cell_id = matched["case_cell_id"]
        _require(
            isinstance(matched_cell_id, str)
            and bool(matched_cell_id)
            and matched_cell_id != cell_id
            and matched_cell_id not in seen_case_ids,
            f"mutant {key} matched-clean case cell reused",
        )
        seen_case_ids.add(matched_cell_id)
        _require(
            matched["case_isolation"] == expected_isolation,
            f"mutant {key} matched clean did not use a fresh disposable cache",
        )
        _require(
            _validate_cleanup_receipt(
                matched["cleanup_receipt"],
                label=f"mutant {key} matched clean",
            )
            == campaign_baseline,
            f"mutant {key} matched-clean cleanup baseline drift",
        )
        matched_outcome = _parse_campaign_outcome(matched["outcome"])
        matched_passed = _validate_clean_outcome(matched_outcome)
        _validate_exercise_coverage_receipt(
            matched["exercise_coverage_receipt"],
            mutant_id=key,
            outcome=matched_outcome,
            mutation_activated=False,
            expected_query_sha256=expected_query_sha256,
        )
        _require(
            type(case["matched_clean_exercise_passed"]) is bool
            and case["matched_clean_exercise_passed"] is matched_passed,
            f"mutant {key} matched-clean pass flag disagrees with blind replay",
        )
        matched_outcomes[key] = matched_outcome
        outcome = _parse_campaign_outcome(case["outcome"])
        _require(outcome.mutant_id == key, f"mutant outcome key/id mismatch for {key}")
        _validate_mutant_error_contract(outcome)
        _validate_exercise_coverage_receipt(
            case["exercise_coverage_receipt"],
            mutant_id=key,
            outcome=outcome,
            mutation_activated=True,
            expected_query_sha256=expected_query_sha256,
        )
        _validate_matched_injected_coverage_pair(
            matched["exercise_coverage_receipt"],
            case["exercise_coverage_receipt"],
            mutant_id=key,
        )
        _validate_receipt_target_binding(
            outcome,
            mutant_id=key,
            case_cell_id=cell_id,
        )
        outcomes[key] = outcome
    return clean, outcomes, matched_outcomes


def _validate_factorial(
    shard: Mapping[str, Any],
    *,
    root: Path,
    rank: int,
    run_id: str,
    expected_query_bank: Mapping[str, Any] | None = None,
) -> tuple[
    bool,
    list[dict[str, Any]],
    dict[int, list[dict[str, Any]]],
    dict[str, str],
    dict[str, dict[str, Any]],
    dict[str, int],
    list[dict[str, Any]],
]:
    rows = shard.get("factorial")
    _require(isinstance(rows, list) and len(rows) == len(FORMAL_RESIDENT_COUNTS), "factorial N rows missing")
    artifact_bindings: list[dict[str, Any]] = []
    semantics_by_n: dict[int, list[dict[str, Any]]] = {}
    exact = True
    shared_source_payload_digest: dict[str, str] | None = None
    frozen_kernel_descriptor: tuple[str, str, str] | None = None
    shared_frozen_baseline: dict[str, int] | None = None
    oracle_contexts: dict[str, dict[str, Any]] = {}
    memory_matrix_rows: list[dict[str, Any]] = []
    for expected_n, n_row in zip(FORMAL_RESIDENT_COUNTS, rows):
        _require(
            isinstance(n_row, dict)
            and _is_int(n_row.get("resident_count"))
            and n_row.get("resident_count") == expected_n,
            "factorial N order drift",
        )
        cells = n_row.get("cells")
        _require(isinstance(cells, list) and len(cells) == 4, "factorial requires four cells")
        _require([cell.get("arm_id") for cell in cells] == list(ARM_IDS), "factorial arm order drift")
        cell_semantics: list[list[dict[str, Any]]] = []
        factor_receipts: list[dict[str, Any]] = []
        for cell in cells:
            arm_id = cell["arm_id"]
            kv_policy = cell.get("kv_policy")
            gdn_policy = cell.get("gdn_base_policy")
            _require(kv_policy in KV_POLICIES and gdn_policy in GDN_BASE_POLICIES, "factorial policy drift")
            _require(arm_id == f"kv={kv_policy}|gdn={gdn_policy}", "factorial arm/policy binding drift")
            memory_audit = _validate_cell_separation(
                cell,
                rank=rank,
                resident_count=expected_n,
                arm_id=arm_id,
            )
            if shared_frozen_baseline is None:
                shared_frozen_baseline = dict(memory_audit["frozen_baseline"])
            else:
                _require(
                    memory_audit["frozen_baseline"] == shared_frozen_baseline,
                    "factorial cells do not share one frozen model/query baseline",
                )
            factor_receipts.append(memory_audit)
            memory_policy = cell["memory_cell"]["policy_execution_receipt"]
            _require(
                memory_policy["kv_policy"] == kv_policy
                and memory_policy["gdn_base_policy"] == gdn_policy,
                "memory receipt/factorial policy binding drift",
            )
            memory_matrix_rows.append(
                {
                    "rank": rank,
                    "resident_count": expected_n,
                    "arm_id": arm_id,
                    "kv_policy": kv_policy,
                    "gdn_base_policy": gdn_policy,
                    "allocator_endpoints": {
                        field: memory_audit[field]
                        for field in MEMORY_ENDPOINT_FIELDS
                    },
                    "q16_analytic_totals": dict(
                        memory_audit["normalized_storage"]["totals"]
                    ),
                    "group_kv_receipt": dict(
                        memory_audit["normalized_group"]["kv"]
                    ),
                    "group_gdn_receipt": dict(
                        memory_audit["normalized_group"]["gdn"]
                    ),
                }
            )
            _memory_ledgers, memory_descriptor = _validate_kernel_ledgers(
                cell.get("memory_kernel_ledgers"),
                resident_count=expected_n,
                kv_policy=kv_policy,
                label="formal memory cell",
                strict_position_values=False,
            )
            witness_ledgers, witness_descriptor = _validate_kernel_ledgers(
                cell.get("witness_kernel_ledgers"),
                resident_count=expected_n,
                kv_policy=kv_policy,
                label="ownership witness cell",
                strict_position_values=True,
            )
            _require(memory_descriptor == witness_descriptor, "memory/witness cells used different kernel descriptors")
            if frozen_kernel_descriptor is None:
                frozen_kernel_descriptor = memory_descriptor
            else:
                _require(memory_descriptor == frozen_kernel_descriptor, "factorial cells used different kernel descriptors")
            before = _validate_sha_map(
                cell.get("source_physical_document_sha256_before"),
                label="physical source digest before",
                keys=FORMAL_FULL_LAYERS,
            )
            after = _validate_sha_map(
                cell.get("source_physical_document_sha256_after"),
                label="physical source digest after",
                keys=FORMAL_FULL_LAYERS,
            )
            _require(cell.get("source_digest_scope") == "complete-physical-document-blocks-including-tail-padding", "physical source digest scope drift")
            _require(before == after, "source physical blocks, including padding, mutated")
            payload_digest = _validate_sha_map(
                cell.get("source_physical_payload_sha256"),
                label="physical source payload digest",
                keys=FORMAL_FULL_LAYERS,
            )
            _require(
                cell.get("source_payload_digest_scope")
                == "key-value-document-block-bytes-including-tail-padding-excluding-arena-capacity-metadata",
                "source payload digest scope drift",
            )
            for role in ("memory", "witness"):
                role_before = _validate_sha_map(
                    cell.get(f"{role}_source_physical_document_sha256_before"),
                    label=f"{role} physical source before",
                    keys=FORMAL_FULL_LAYERS,
                )
                role_after = _validate_sha_map(
                    cell.get(f"{role}_source_physical_document_sha256_after"),
                    label=f"{role} physical source after",
                    keys=FORMAL_FULL_LAYERS,
                )
                role_payload = _validate_sha_map(
                    cell.get(f"{role}_source_physical_payload_sha256"),
                    label=f"{role} source payload",
                    keys=FORMAL_FULL_LAYERS,
                )
                _require(
                    role_before == role_after == before
                    and role_payload == payload_digest,
                    f"{role} rebuild source receipt differs from common cell receipt",
                )
            if shared_source_payload_digest is None:
                shared_source_payload_digest = payload_digest
            else:
                _require(payload_digest == shared_source_payload_digest, "factorial cells/N used different physical source payloads")
            if expected_n == ORACLE_RESIDENT_COUNT:
                oracle_contexts[arm_id] = {
                    "cell_id": cell["witness_cell"]["cell_id"],
                    "witness_ledgers": witness_ledgers,
                    "source_physical_payload_sha256_by_layer": payload_digest,
                }
            semantics = _validate_semantics(cell.get("semantics"), resident_count=expected_n)
            diagnostic_rows = memory_audit["generation_diagnostics"]
            for diagnostic in diagnostic_rows:
                _require(
                    diagnostic["cpu_logits_sha256"]
                    == semantics[diagnostic["request_index"]][
                        "full_vocab_step_logit_sha256"
                    ][diagnostic["round_index"]],
                    "generation CPU diagnostic digest differs from semantic trajectory",
                )
            witness_semantics = _validate_semantics(
                cell.get("witness_semantics"), resident_count=expected_n
            )
            _require(
                witness_semantics == semantics,
                "memory and ownership-witness rebuilds diverged semantically",
            )
            _require(
                len({row["query_token_ids_sha256"] for row in semantics})
                == expected_n,
                "resident requests reused a query instead of a distinct PG19 query-bank prefix",
            )
            cell_semantics.append(semantics)
            replay, bindings = _validate_timeline_manifest(
                cell["witness_cell"].get("timeline_manifest_artifact"),
                root=root,
                rank=rank,
                run_id=run_id,
                resident_count=expected_n,
                kv_policy=kv_policy,
                gdn_base_policy=gdn_policy,
                witness_cell_id=cell["witness_cell"]["cell_id"],
            )
            _require(replay.get("completed_all_requests") is True, "GDN timeline incomplete")
            factor_receipts[-1]["normalized_gdn_timeline"] = replay.get(
                "normalized_gdn_factor_projection"
            )
            factor_receipts[-1]["normalized_kv_timeline"] = replay.get(
                "normalized_kv_factor_projection"
            )
            _require(
                isinstance(factor_receipts[-1]["normalized_gdn_timeline"], dict)
                and isinstance(
                    factor_receipts[-1]["normalized_kv_timeline"], dict
                ),
                "factor timeline projection missing",
            )
            artifact_bindings.extend(bindings)
        # Factor isolation is replayed from the raw group/storage payloads,
        # never from producer booleans.  Holding KV fixed must make its
        # normalized allocation/ownership receipt invariant to GDN policy;
        # holding GDN fixed must make its setup ownership receipt invariant to
        # KV copy/reuse policy.
        for left, right in ((0, 1), (2, 3)):
            _require(
                factor_receipts[left]["normalized_storage"]
                == factor_receipts[right]["normalized_storage"]
                and factor_receipts[left]["normalized_group"]["kv"]
                == factor_receipts[right]["normalized_group"]["kv"]
                and factor_receipts[left]["normalized_kv_timeline"]
                == factor_receipts[right]["normalized_kv_timeline"],
                "GDN axis changed the normalized KV factor receipt",
            )
        for left, right in ((0, 2), (1, 3)):
            _require(
                factor_receipts[left]["normalized_group"]["gdn"]
                == factor_receipts[right]["normalized_group"]["gdn"]
                and factor_receipts[left]["normalized_gdn_timeline"]
                == factor_receipts[right]["normalized_gdn_timeline"],
                "KV axis changed the normalized GDN factor receipt",
            )
        baseline = cell_semantics[0]
        if any(candidate != baseline for candidate in cell_semantics[1:]):
            exact = False
        semantics_by_n[expected_n] = baseline
        if expected_query_bank is not None:
            expected_query_digests = [
                row["query_token_ids_sha256"]
                for row in expected_query_bank["rows"][:expected_n]
            ]
            _require(
                [row["query_token_ids_sha256"] for row in baseline]
                == expected_query_digests,
                "factorial semantics differ from frozen PG19 query-bank prefix",
            )
    # Query i and its eight-step result must be stable when the same prefix is
    # embedded in a larger resident fan-out.
    for smaller, larger in zip(
        FORMAL_RESIDENT_COUNTS, FORMAL_RESIDENT_COUNTS[1:]
    ):
        prefix = semantics_by_n[smaller]
        if semantics_by_n[larger][: len(prefix)] != prefix:
            exact = False
    assert shared_source_payload_digest is not None
    assert shared_frozen_baseline is not None
    if shard.get("artifact_mode") == "formal_gpu":
        warmup = shard.get("warmup_receipt")
        _require(
            isinstance(warmup, dict)
            and warmup.get("frozen_model_query_baseline")
            == shared_frozen_baseline,
            "formal cell baseline differs from the post-warmup frozen baseline",
        )
    _require(set(oracle_contexts) == set(ARM_IDS), "N=1 oracle witness contexts incomplete")
    return (
        exact,
        artifact_bindings,
        semantics_by_n,
        shared_source_payload_digest,
        oracle_contexts,
        shared_frozen_baseline,
        memory_matrix_rows,
    )


def _validate_shard_common(
    shard: Any,
    *,
    rank: int,
    static_sha256: str,
    expected_identity: Mapping[str, Any],
    expected_query_bank: Mapping[str, Any],
    expected_run_id_receipt: Mapping[str, Any] | None = None,
    expected_run_id_receipt_sha256: str | None = None,
    expected_gpu_assignment_receipt: Mapping[str, Any] | None = None,
    expected_gpu_assignment_receipt_raw_sha256: str | None = None,
    expected_private_model_view_manifest: Mapping[str, Any] | None = None,
    expected_private_model_view_manifest_raw_sha256: str | None = None,
    expected_model_load_authority: Mapping[str, Any] | None = None,
    expected_model_load_authority_raw_sha256: str | None = None,
) -> str:
    _require(isinstance(shard, dict), "shard artifact must be object")
    _require(shard.get("schema_version") == SHARD_SCHEMA_VERSION, "shard schema drift")
    _require(shard.get("protocol") == PROTOCOL, "shard protocol drift")
    _require(
        _is_int(shard.get("rank"))
        and _is_int(shard.get("world_size"))
        and shard.get("rank") == rank
        and shard.get("world_size") == FORMAL_WORLD_SIZE,
        "shard rank/world drift",
    )
    _require(shard.get("static_artifact_sha256") == static_sha256, "shard/static binding drift")
    _require(shard.get("protocol_config") == formal_protocol_config(), "shard protocol config drift")
    _require(shard.get("protocol_config_sha256") == sha256_json(formal_protocol_config()), "shard protocol SHA drift")
    _require(shard.get("frozen_identity") == expected_identity, "shard frozen identity drift")
    run_id = _require_run_id(shard.get("run_id"), "shard run ID")
    _validate_data_usage(
        shard, rank, expected_query_bank=expected_query_bank
    )
    if shard.get("artifact_mode") == "formal_gpu":
        _require(
            expected_run_id_receipt is not None
            and expected_run_id_receipt_sha256 is not None
            and expected_gpu_assignment_receipt is not None
            and expected_gpu_assignment_receipt_raw_sha256 is not None
            and expected_private_model_view_manifest is not None
            and expected_private_model_view_manifest_raw_sha256 is not None
            and expected_model_load_authority is not None
            and expected_model_load_authority_raw_sha256 is not None,
            "formal shard validation requires external run/GPU/model authorities",
        )
        _validate_formal_producer_receipts(
            shard,
            rank=rank,
            run_id=run_id,
            static_sha256=static_sha256,
            expected_identity=expected_identity,
            expected_query_bank=expected_query_bank,
            expected_run_id_receipt=expected_run_id_receipt,
            expected_run_id_receipt_sha256=expected_run_id_receipt_sha256,
            expected_gpu_assignment_receipt=expected_gpu_assignment_receipt,
            expected_gpu_assignment_receipt_raw_sha256=(
                expected_gpu_assignment_receipt_raw_sha256
            ),
            expected_private_model_view_manifest=(
                expected_private_model_view_manifest
            ),
            expected_private_model_view_manifest_raw_sha256=(
                expected_private_model_view_manifest_raw_sha256
            ),
            expected_model_load_authority=expected_model_load_authority,
            expected_model_load_authority_raw_sha256=(
                expected_model_load_authority_raw_sha256
            ),
        )
    return run_id


def make_receipt_manifest(
    shard_paths: Sequence[Path],
    *,
    root: Path,
    static_artifact_sha256: str,
) -> dict[str, Any]:
    _require(len(shard_paths) == FORMAL_WORLD_SIZE, "receipt manifest requires eight shards")
    refs = [artifact_reference(path, root=root) for path in shard_paths]
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "static_artifact_sha256": _require_sha256(
            static_artifact_sha256, "receipt static SHA"
        ),
        "shards": refs,
    }


def _aggregate_memory_matrix(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the deterministic 2x2xN result matrix from replayed receipts."""

    required = {
        "rank",
        "resident_count",
        "arm_id",
        "kv_policy",
        "gdn_base_policy",
        "allocator_endpoints",
        "q16_analytic_totals",
        "group_kv_receipt",
        "group_gdn_receipt",
    }
    _require(
        len(rows)
        == FORMAL_WORLD_SIZE * len(FORMAL_RESIDENT_COUNTS) * len(ARM_IDS),
        "memory matrix raw row cardinality drift",
    )
    by_coordinate: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for raw in rows:
        _require(
            isinstance(raw, Mapping) and set(raw) == required,
            "memory matrix raw row schema drift",
        )
        rank = raw["rank"]
        resident_count = raw["resident_count"]
        arm_id = raw["arm_id"]
        _require(
            _is_int(rank)
            and 0 <= rank < FORMAL_WORLD_SIZE
            and _is_int(resident_count)
            and resident_count in FORMAL_RESIDENT_COUNTS
            and arm_id in ARM_IDS,
            "memory matrix raw coordinate drift",
        )
        expected_arm = (
            f"kv={raw['kv_policy']}|gdn={raw['gdn_base_policy']}"
        )
        _require(
            raw["kv_policy"] in KV_POLICIES
            and raw["gdn_base_policy"] in GDN_BASE_POLICIES
            and arm_id == expected_arm,
            "memory matrix raw factor binding drift",
        )
        endpoints = raw["allocator_endpoints"]
        _require(
            isinstance(endpoints, dict)
            and set(endpoints) == set(MEMORY_ENDPOINT_FIELDS)
            and all(_is_int(value) and value >= 0 for value in endpoints.values()),
            "memory matrix allocator endpoint drift",
        )
        for name in (
            "q16_analytic_totals",
            "group_kv_receipt",
            "group_gdn_receipt",
        ):
            _require(isinstance(raw[name], dict), f"memory matrix {name} missing")
        by_coordinate.setdefault((resident_count, arm_id), []).append(dict(raw))

    cells = []
    for resident_count in FORMAL_RESIDENT_COUNTS:
        for arm_id in ARM_IDS:
            coordinate_rows = sorted(
                by_coordinate.get((resident_count, arm_id), []),
                key=lambda row: row["rank"],
            )
            _require(
                [row["rank"] for row in coordinate_rows]
                == list(range(FORMAL_WORLD_SIZE)),
                "memory matrix rank coverage drift",
            )
            first = coordinate_rows[0]
            for analytic_field in (
                "q16_analytic_totals",
                "group_kv_receipt",
                "group_gdn_receipt",
            ):
                _require(
                    all(
                        row[analytic_field] == first[analytic_field]
                        for row in coordinate_rows[1:]
                    ),
                    f"memory matrix {analytic_field} differs across ranks",
                )
            cells.append(
                {
                    "resident_count": resident_count,
                    "arm_id": arm_id,
                    "kv_policy": first["kv_policy"],
                    "gdn_base_policy": first["gdn_base_policy"],
                    "allocator_raw_by_rank": [
                        {
                            "rank": row["rank"],
                            **dict(row["allocator_endpoints"]),
                        }
                        for row in coordinate_rows
                    ],
                    "allocator_median_across_ranks": {
                        field: statistics.median(
                            row["allocator_endpoints"][field]
                            for row in coordinate_rows
                        )
                        for field in MEMORY_ENDPOINT_FIELDS
                    },
                    "q16_analytic_totals": dict(
                        first["q16_analytic_totals"]
                    ),
                    "group_kv_receipt": dict(first["group_kv_receipt"]),
                    "group_gdn_receipt": dict(first["group_gdn_receipt"]),
                }
            )
    _require(
        len(by_coordinate) == len(FORMAL_RESIDENT_COUNTS) * len(ARM_IDS),
        "memory matrix contains an unexpected coordinate",
    )
    return {
        "schema_version": "qcomem-forkaudit-memory-matrix-v1",
        "rank_aggregation": "median-with-complete-rank-ordered-raw-list",
        "confidence_interval_reported": False,
        "generic_unique_storage_endpoint_included": False,
        "endpoint_fields": list(MEMORY_ENDPOINT_FIELDS),
        "cells": cells,
    }


def _validate_external_model_load_evidence(
    *,
    authority_raw: bytes,
    expected_authority_raw_sha256: str,
    closure_raw: bytes,
    expected_closure_raw_sha256: str,
    private_model_view_manifest_raw: bytes,
    expected_private_model_view_manifest_raw_sha256: str,
    model_weight_ledger_raw: bytes,
    model_artifact_ledger_raw: bytes,
    expected_identity: Mapping[str, Any],
    expected_run_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Replay launcher-owned model-load authority without trusting shards."""

    from qcomem_forkaudit_model_load_lease import (
        ModelLoadLeaseError,
        authority_from_canonical_bytes,
        closure_from_canonical_bytes,
    )

    authority_sha = _require_sha256(
        expected_authority_raw_sha256,
        "aggregate expected ModelLoadLease authority raw SHA",
    )
    closure_sha = _require_sha256(
        expected_closure_raw_sha256,
        "aggregate expected ModelLoadLease closure raw SHA",
    )
    _require(
        sha256_bytes(model_weight_ledger_raw)
        == expected_identity["model_weight_ledger_sha256"]
        and sha256_bytes(model_artifact_ledger_raw)
        == expected_identity["model_artifact_ledger_sha256"],
        "aggregate model ledgers differ from frozen identity",
    )
    weight_rows = _parse_sha256_ledger(
        model_weight_ledger_raw, label="aggregate model weight ledger"
    )
    artifact_rows = _parse_sha256_ledger(
        model_artifact_ledger_raw, label="aggregate model artifact ledger"
    )
    _require(len(weight_rows) == 14, "aggregate weight ledger must have 14 rows")
    try:
        authority = authority_from_canonical_bytes(authority_raw, authority_sha)
    except ModelLoadLeaseError as exc:
        raise ReviewAuditError("aggregate ModelLoadLease authority rejected") from exc
    _require(
        authority["run_id"] == expected_run_id
        and authority["weight_ledger_raw_sha256"]
        == sha256_bytes(model_weight_ledger_raw)
        and authority["model_artifact_ledger_raw_sha256"]
        == sha256_bytes(model_artifact_ledger_raw)
        and [
            {
                "logical_name": row["logical_name"],
                "sha256": row["declared_sha256"],
            }
            for row in authority["rows"]
        ]
        == weight_rows,
        "aggregate ModelLoadLease authority/ledger binding drift",
    )

    private_manifest_sha = _require_sha256(
        expected_private_model_view_manifest_raw_sha256,
        "aggregate expected private model-view manifest raw SHA",
    )
    private_value = strict_json_loads(
        private_model_view_manifest_raw,
        label="aggregate private model-view manifest",
    )
    private_manifest = _validate_private_model_view_manifest(
        private_value,
        raw_bytes=private_model_view_manifest_raw,
        expected_raw_sha256=private_manifest_sha,
        model_artifact_rows=artifact_rows,
        model_weight_rows=weight_rows,
        model_artifact_ledger_raw_sha256=sha256_bytes(model_artifact_ledger_raw),
        model_weight_ledger_raw_sha256=sha256_bytes(model_weight_ledger_raw),
        model_load_authority=authority,
    )

    try:
        closure = closure_from_canonical_bytes(
            closure_raw,
            closure_sha,
            authority=authority,
            require_passed=True,
        )
    except ModelLoadLeaseError as exc:
        raise ReviewAuditError("aggregate ModelLoadLease closure rejected") from exc
    _require(
        closure["run_id"] == expected_run_id
        and closure["authority_raw_sha256"] == authority_sha,
        "aggregate ModelLoadLease closure binding drift",
    )
    receipt = {
        "schema_version": "qcomem-forkaudit-model-load-integrity-summary-v1",
        "private_model_view_manifest_raw_sha256": private_manifest_sha,
        "authority_raw_sha256": authority_sha,
        "closure_raw_sha256": closure_sha,
        "threat_model": authority["threat_model"],
        "entry_count": authority["entry_count"],
        "terminal_full_content_rehash_performed": closure[
            "terminal_full_content_rehash_performed"
        ],
        "passed": True,
    }
    return authority, closure, private_manifest, receipt


def aggregate_shards(
    receipt_manifest: Any,
    *,
    expected_receipt_manifest_sha256: str,
    static_artifact: Any,
    static_artifact_sha256: str,
    artifact_root: Path,
    expected_run_id: str | None = None,
    run_id_receipt_raw: bytes | None = None,
    expected_run_id_receipt_sha256: str | None = None,
    gpu_assignment_receipt_raw: bytes | None = None,
    expected_gpu_assignment_receipt_raw_sha256: str | None = None,
    private_model_view_manifest_raw: bytes | None = None,
    expected_private_model_view_manifest_raw_sha256: str | None = None,
    model_load_authority_raw: bytes | None = None,
    expected_model_load_authority_raw_sha256: str | None = None,
    model_load_closure_raw: bytes | None = None,
    expected_model_load_closure_raw_sha256: str | None = None,
    model_weight_ledger_raw: bytes | None = None,
    model_artifact_ledger_raw: bytes | None = None,
    allow_synthetic_schema_fixture: bool = False,
) -> dict[str, Any]:
    """Blindly replay detached raw artifacts.

    ``expected_receipt_manifest_sha256`` is external state (for example, a
    launcher terminal receipt).  Recomputing a fresh receipt after editing a
    shard is deliberately insufficient.
    """

    if expected_run_id is not None:
        expected_run_id = _require_run_id(expected_run_id, "aggregate expected run ID")
    elif not allow_synthetic_schema_fixture:
        raise ReviewAuditError("formal aggregate requires an external expected run ID")
    _require(
        sha256_json(receipt_manifest)
        == _require_sha256(
            expected_receipt_manifest_sha256,
            "detached receipt-manifest SHA",
        ),
        "detached receipt manifest SHA drift",
    )
    static = validate_static_artifact(static_artifact)
    static_sha = _require_sha256(static_artifact_sha256, "static artifact SHA")
    _require(sha256_json(static_artifact) == static_sha, "static artifact SHA mismatch")
    _require(isinstance(receipt_manifest, dict), "receipt manifest must be object")
    _require(
        set(receipt_manifest)
        == {"schema_version", "protocol", "static_artifact_sha256", "shards"},
        "receipt manifest schema drift",
    )
    _require(receipt_manifest["schema_version"] == RECEIPT_SCHEMA_VERSION, "receipt schema drift")
    _require(receipt_manifest["protocol"] == PROTOCOL, "receipt protocol drift")
    _require(receipt_manifest["static_artifact_sha256"] == static_sha, "receipt/static binding drift")
    shard_refs = receipt_manifest["shards"]
    _require(isinstance(shard_refs, list) and len(shard_refs) == FORMAL_WORLD_SIZE, "aggregate requires eight shard receipts")

    loaded_shards = [
        load_json_artifact(ref, root=artifact_root, label=f"raw shard receipt {index}")
        for index, ref in enumerate(shard_refs)
    ]
    shards = [row.payload for row in loaded_shards]
    _require(
        all(isinstance(row, dict) and _is_int(row.get("rank")) for row in shards),
        "shard rank missing",
    )
    shards.sort(key=lambda row: row["rank"])
    _require([row["rank"] for row in shards] == list(range(FORMAL_WORLD_SIZE)), "shard ranks incomplete or duplicated")

    modes = {row.get("artifact_mode") for row in shards}
    _require(len(modes) == 1, "aggregate cannot mix formal and synthetic shard modes")
    formal_mode = modes == {"formal_gpu"}
    external_run_receipt: dict[str, Any] | None = None
    external_gpu_assignment: dict[str, Any] | None = None
    external_gpu_assignment_raw_sha: str | None = None
    external_private_model_view_manifest: dict[str, Any] | None = None
    external_private_model_view_manifest_raw_sha: str | None = None
    external_model_load_authority: dict[str, Any] | None = None
    external_model_load_authority_raw_sha: str | None = None
    external_model_load_integrity: dict[str, Any] | None = None
    if formal_mode:
        _require(
            run_id_receipt_raw is not None
            and expected_run_id_receipt_sha256 is not None
            and gpu_assignment_receipt_raw is not None
            and expected_gpu_assignment_receipt_raw_sha256 is not None
            and private_model_view_manifest_raw is not None
            and expected_private_model_view_manifest_raw_sha256 is not None
            and model_load_authority_raw is not None
            and expected_model_load_authority_raw_sha256 is not None
            and model_load_closure_raw is not None
            and expected_model_load_closure_raw_sha256 is not None
            and model_weight_ledger_raw is not None
            and model_artifact_ledger_raw is not None
            and expected_run_id is not None,
            "formal aggregate requires shared run/GPU/private-view/model-load receipts",
        )
        external_run_receipt_value = strict_json_loads(
            run_id_receipt_raw, label="aggregate run-ID receipt"
        )
        _require(
            isinstance(external_run_receipt_value, dict),
            "aggregate run-ID receipt must be an object",
        )
        external_run_receipt = _validate_run_id_receipt(
            run_id_receipt_raw,
            expected_sha256=expected_run_id_receipt_sha256,
            run_id=expected_run_id,
            static_artifact_sha256=static_sha,
            protocol_manifest_sha256=static["frozen_identity"][
                "protocol_manifest_sha256"
            ],
        )
        external_gpu_value = strict_json_loads(
            gpu_assignment_receipt_raw,
            label="aggregate GPU-assignment receipt",
        )
        external_gpu_assignment_raw_sha = _require_sha256(
            expected_gpu_assignment_receipt_raw_sha256,
            "aggregate expected GPU-assignment receipt raw SHA",
        )
        external_gpu_assignment = _validate_gpu_assignment_receipt(
            external_gpu_value,
            raw_sha256=external_gpu_assignment_raw_sha,
            raw_bytes=gpu_assignment_receipt_raw,
        )
        (
            external_model_load_authority,
            _external_model_load_closure,
            external_private_model_view_manifest,
            external_model_load_integrity,
        ) = _validate_external_model_load_evidence(
            authority_raw=model_load_authority_raw,
            expected_authority_raw_sha256=(
                expected_model_load_authority_raw_sha256
            ),
            closure_raw=model_load_closure_raw,
            expected_closure_raw_sha256=(
                expected_model_load_closure_raw_sha256
            ),
            private_model_view_manifest_raw=private_model_view_manifest_raw,
            expected_private_model_view_manifest_raw_sha256=(
                expected_private_model_view_manifest_raw_sha256
            ),
            model_weight_ledger_raw=model_weight_ledger_raw,
            model_artifact_ledger_raw=model_artifact_ledger_raw,
            expected_identity=static["frozen_identity"],
            expected_run_id=expected_run_id,
        )
        external_private_model_view_manifest_raw_sha = _require_sha256(
            expected_private_model_view_manifest_raw_sha256,
            "aggregate expected private model-view manifest raw SHA",
        )
        external_model_load_authority_raw_sha = _require_sha256(
            expected_model_load_authority_raw_sha256,
            "aggregate expected ModelLoadLease authority raw SHA",
        )

    artifact_bindings = [row.binding for row in loaded_shards]
    run_id: str | None = None
    source_objects: list[str] = []
    factorial_exact = True
    oracle_outcomes: list[dict[str, Any]] = []
    clean_outcomes: list[CampaignOutcome] = []
    matched_clean_outcomes: dict[str, CampaignOutcome] = {}
    mutant_outcomes: dict[str, CampaignOutcome] = {}
    memory_matrix_rows: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for rank, shard in enumerate(shards):
        mode = shard.get("artifact_mode")
        if mode == "synthetic_schema_fixture":
            _require(allow_synthetic_schema_fixture, "synthetic fixture rejected by formal aggregate")
            _require(shard.get("status") == "completed_synthetic_schema_fixture", "synthetic shard status drift")
        else:
            _require(GPU_LOOP_IMPLEMENTED, "formal shard rejected: GPU loop is not implemented")
            _require(
                static["formal_input_provenance_bound"] is True,
                "formal shard lacks raw-bound RR2/prior/response-plan inputs",
            )
            _require(mode == "formal_gpu" and shard.get("status") == "completed_formal_gpu_shard", "formal shard status drift")
        current_run_id = _validate_shard_common(
            shard,
            rank=rank,
            static_sha256=static_sha,
            expected_identity=static["frozen_identity"],
            expected_query_bank=static["frozen_query_banks"][rank],
            expected_run_id_receipt=external_run_receipt,
            expected_run_id_receipt_sha256=expected_run_id_receipt_sha256,
            expected_gpu_assignment_receipt=external_gpu_assignment,
            expected_gpu_assignment_receipt_raw_sha256=(
                external_gpu_assignment_raw_sha
            ),
            expected_private_model_view_manifest=(
                external_private_model_view_manifest
            ),
            expected_private_model_view_manifest_raw_sha256=(
                external_private_model_view_manifest_raw_sha
            ),
            expected_model_load_authority=external_model_load_authority,
            expected_model_load_authority_raw_sha256=(
                external_model_load_authority_raw_sha
            ),
        )
        if run_id is None:
            run_id = current_run_id
        else:
            _require(current_run_id == run_id, "shards belong to different runs")
        source_object = shard["data_usage"]["source_object"]
        source_objects.append(source_object)
        (
            cell_exact,
            cell_bindings,
            _semantics,
            _source_payload,
            oracle_contexts,
            frozen_rank_baseline,
            rank_memory_rows,
        ) = _validate_factorial(
            shard,
            root=artifact_root,
            rank=rank,
            run_id=current_run_id,
            expected_query_bank=static["frozen_query_banks"][rank],
        )
        factorial_exact = factorial_exact and cell_exact
        memory_matrix_rows.extend(rank_memory_rows)
        artifact_bindings.extend(cell_bindings)
        frozen_selection = static["oracle_selection_plan"][rank]
        oracle_arm_id = (
            f"kv={frozen_selection['kv_policy']}|"
            f"gdn={frozen_selection['gdn_base_policy']}"
        )
        oracle_context = dict(oracle_contexts[oracle_arm_id])
        selected_ledger = oracle_context["witness_ledgers"][
            frozen_selection["request_index"]
        ]
        layer_offset = FORMAL_FULL_LAYERS.index(frozen_selection["layer_index"])
        call_offset = (
            frozen_selection["round_index"] * len(FORMAL_FULL_LAYERS)
            + layer_offset
        )
        oracle_context["ledger_call"] = selected_ledger["calls"][call_offset]
        oracle_context["selected_ledger"] = selected_ledger
        oracle_context["source_physical_payload_sha256"] = oracle_context[
            "source_physical_payload_sha256_by_layer"
        ][str(frozen_selection["layer_index"])]
        oracle, oracle_bindings = _recompute_oracle(
            shard.get("oracle_raw_artifact"),
            root=artifact_root,
            rank=rank,
            source_object=source_object,
            expected_selection=frozen_selection,
            observer_context=oracle_context,
            expected_run_id=current_run_id,
            synthetic_geometry=mode == "synthetic_schema_fixture",
        )
        oracle_outcomes.append(oracle)
        artifact_bindings.extend(oracle_bindings)
        clean, mutants, matched_controls = _validate_fault_campaign(
            shard,
            rank=rank,
            seen_case_ids=seen_case_ids,
            expected_query_sha256=static["frozen_query_banks"][rank]["rows"][0][
                "query_token_ids_sha256"
            ],
            expected_frozen_baseline=frozen_rank_baseline,
        )
        if formal_mode:
            expected_self_replay = _make_producer_self_replay_receipt(
                factorial_exact=cell_exact,
                oracle_passed=oracle["passed"],
                mutant_rows_replayed=len(mutants),
                matched_clean_rows_replayed=len(matched_controls),
                memory_matrix_rows_replayed=len(rank_memory_rows),
                detached_sidecar_references_replayed=(
                    len(cell_bindings) + len(oracle_bindings)
                ),
            )
            _validate_producer_self_replay_receipt(
                shard.get("producer_self_replay"),
                expected=expected_self_replay,
            )
        clean_outcomes.append(clean)
        for mutant_id, outcome in matched_controls.items():
            _require(
                mutant_id not in matched_clean_outcomes,
                f"matched clean {mutant_id} duplicated across ranks",
            )
            matched_clean_outcomes[mutant_id] = outcome
        for mutant_id, outcome in mutants.items():
            _require(mutant_id not in mutant_outcomes, f"mutant {mutant_id} duplicated across ranks")
            mutant_outcomes[mutant_id] = outcome
    _require(len(set(source_objects)) == FORMAL_BOOKS, "rank PG19 train books are not distinct")
    if expected_run_id is not None:
        _require(run_id == expected_run_id, "aggregate/shard run ID drift")
    _require(set(mutant_outcomes) == set(MUTANT_IDS), "global mutant campaign is incomplete")
    clean_passes = [_validate_clean_outcome(clean) for clean in clean_outcomes]
    clean_false_positive_ranks = [
        rank for rank, passed in enumerate(clean_passes) if not passed
    ]
    matched_clean_passes = {
        mutant_id: _validate_clean_outcome(outcome)
        for mutant_id, outcome in matched_clean_outcomes.items()
    }
    _require(
        set(matched_clean_passes) == set(MUTANT_IDS),
        "global matched-clean campaign is incomplete",
    )
    matched_clean_false_positive_ids = sorted(
        mutant_id
        for mutant_id, passed in matched_clean_passes.items()
        if not passed
    )
    campaign = validate_campaign_outcomes(clean_outcomes[0], mutant_outcomes)
    _require(not campaign["binding_errors"], "mutant outcome/spec/receipt binding failed")
    _require(not campaign["missing_mutant_ids"] and not campaign["unexpected_mutant_ids"], "mutant ID set drift")

    unexpected_crashes = list(campaign["unexpected_crash_mutant_ids"])
    memory_matrix = _aggregate_memory_matrix(memory_matrix_rows)
    scientific_run_valid = not unexpected_crashes
    oracle_passed = all(row["passed"] is True for row in oracle_outcomes)
    hypothesis_passed = bool(
        scientific_run_valid
        and not clean_false_positive_ranks
        and not matched_clean_false_positive_ids
        and factorial_exact
        and oracle_passed
        and campaign["passed"]
    )
    negative_reasons = []
    if clean_false_positive_ranks:
        negative_reasons.append("clean_false_positive")
    if matched_clean_false_positive_ids:
        negative_reasons.append("matched_clean_false_positive")
    if not factorial_exact:
        negative_reasons.append("factorial_semantic_divergence")
    if not oracle_passed:
        negative_reasons.append("oracle_relative_l2_or_gate_failure")
    if campaign["escaped_mutant_ids"]:
        negative_reasons.append("escaped_mutants")
    if campaign["wrong_gate_mutant_ids"]:
        negative_reasons.append("wrong_gate_mutants")
    if unexpected_crashes:
        negative_reasons.append("unexpected_mutant_crash_invalidates_run")

    scientific_outcome = (
        "invalid"
        if not scientific_run_valid
        else ("valid_positive" if hypothesis_passed else "valid_negative")
    )
    # The normalized summary excludes arbitrary raw extras.  It contains only
    # recomputed facts and detached byte bindings.
    return {
        "schema_version": AGGREGATE_SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "implementation_status": IMPLEMENTATION_STATUS,
        "schema_replay_passed": True,
        "scientific_run_valid": scientific_run_valid,
        "hypothesis_passed": hypothesis_passed,
        "passed": scientific_run_valid,
        "formal_ready": scientific_run_valid,
        "scientific_outcome": scientific_outcome,
        "formal_blocker": (
            None
            if scientific_run_valid
            else "unexpected_mutant_crash_invalidates_run"
        ),
        "run_id": run_id,
        "rank_count": FORMAL_WORLD_SIZE,
        "factorial_four_cell_exact": factorial_exact,
        "memory_matrix": memory_matrix,
        "oracle_all_ranks_passed": oracle_passed,
        "oracle_max_relative_l2": ORACLE_MAX_RELATIVE_L2,
        "oracle_relative_l2_by_rank": [
            row["attention_metrics"]["relative_l2"] for row in oracle_outcomes
        ],
        "mutant_campaign": {
            "passed": bool(
                campaign["passed"]
                and not clean_false_positive_ranks
                and not matched_clean_false_positive_ids
            ),
            "clean_false_positive_ranks": clean_false_positive_ranks,
            "matched_clean_false_positive_mutant_ids": (
                matched_clean_false_positive_ids
            ),
            "escaped_mutant_ids": list(campaign["escaped_mutant_ids"]),
            "wrong_gate_mutant_ids": list(campaign["wrong_gate_mutant_ids"]),
            "unexpected_crash_mutant_ids": unexpected_crashes,
        },
        "negative_reasons": negative_reasons,
        "raw_shard_artifacts": [row.binding for row in loaded_shards],
        "replayed_evidence_artifacts": artifact_bindings[len(loaded_shards) :],
        "detached_receipt_manifest_sha256": expected_receipt_manifest_sha256,
        "static_artifact_sha256": static_sha,
        "model_load_integrity": external_model_load_integrity,
        "integrity_scope": (
            "external SHA-256 binds raw bytes; internal witness digests provide "
            "self-consistent replay, not standalone tamper evidence"
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("static", "shard", "aggregate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-identity", type=Path)
    parser.add_argument("--oracle-selection-plan", type=Path)
    parser.add_argument("--frozen-query-banks", type=Path)
    parser.add_argument("--rr2-input-manifest", type=Path)
    parser.add_argument("--expected-rr2-input-manifest-sha256")
    parser.add_argument("--prior-fp32-context-manifest", type=Path)
    parser.add_argument("--expected-prior-fp32-context-manifest-sha256")
    parser.add_argument("--review-experiment-plan", type=Path)
    parser.add_argument("--expected-review-experiment-plan-sha256")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--static-artifact", type=Path)
    parser.add_argument("--expected-static-sha256")
    parser.add_argument("--receipt-manifest", type=Path)
    parser.add_argument("--expected-receipt-manifest-sha256")
    parser.add_argument("--run-id")
    parser.add_argument("--run-id-receipt", type=Path)
    parser.add_argument("--expected-run-id-receipt-sha256")
    parser.add_argument("--gpu-assignment-receipt", type=Path)
    parser.add_argument("--expected-gpu-assignment-receipt-raw-sha256")
    parser.add_argument("--private-model-view-manifest", type=Path)
    parser.add_argument("--expected-private-model-view-manifest-raw-sha256")
    parser.add_argument("--model-load-authority", type=Path)
    parser.add_argument("--expected-model-load-authority-raw-sha256")
    parser.add_argument("--model-load-closure", type=Path)
    parser.add_argument("--expected-model-load-closure-raw-sha256")
    parser.add_argument("--pg19-data", type=Path)
    parser.add_argument("--pg19-manifest", type=Path)
    parser.add_argument("--prior-capacity-manifest", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--code-ledger", type=Path)
    parser.add_argument("--model-artifact-ledger", type=Path)
    parser.add_argument("--model-weight-ledger", type=Path)
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument("--expected-gpu-uuid")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.stage == "static":
        _require(args.frozen_identity is not None, "static requires --frozen-identity")
        _require(args.oracle_selection_plan is not None, "static requires --oracle-selection-plan")
        _require(args.frozen_query_banks is not None, "static requires --frozen-query-banks")
        _require(args.rr2_input_manifest is not None, "static requires --rr2-input-manifest")
        _require(args.prior_fp32_context_manifest is not None, "static requires --prior-fp32-context-manifest")
        _require(args.review_experiment_plan is not None, "static requires --review-experiment-plan")
        identity = strict_json_loads(args.frozen_identity.read_bytes(), label="frozen identity")
        plan = strict_json_loads(args.oracle_selection_plan.read_bytes(), label="oracle selection plan")
        banks = strict_json_loads(args.frozen_query_banks.read_bytes(), label="frozen query banks")
        rr2_raw = args.rr2_input_manifest.read_bytes()
        prior_raw = args.prior_fp32_context_manifest.read_bytes()
        response_raw = args.review_experiment_plan.read_bytes()
        _require(
            sha256_bytes(rr2_raw)
            == _require_sha256(
                args.expected_rr2_input_manifest_sha256,
                "expected RR2 input-manifest SHA",
            ),
            "RR2 input-manifest CLI SHA drift",
        )
        _require(
            sha256_bytes(prior_raw)
            == _require_sha256(
                args.expected_prior_fp32_context_manifest_sha256,
                "expected prior FP32 context SHA",
            )
            == PRIOR_FP32_CONTEXT_RAW_SHA256,
            "prior FP32 context CLI SHA drift",
        )
        _require(
            sha256_bytes(response_raw)
            == _require_sha256(
                args.expected_review_experiment_plan_sha256,
                "expected review experiment-plan SHA",
            )
            == FINAL_REVIEW_RESPONSE_PLAN_SHA256,
            "review experiment-plan CLI SHA drift",
        )
        _write_json(
            args.output,
            make_static_artifact(
                identity,
                plan,
                banks,
                rr2_input_manifest_raw=rr2_raw,
                prior_fp32_context_manifest_raw=prior_raw,
                review_response_plan_raw=response_raw,
            ),
        )
        return 0
    if args.stage == "shard":
        _require_run_id(args.run_id, "shard --run-id")
        if not GPU_LOOP_IMPLEMENTED:
            run_shard_not_implemented(rank=args.rank)
            return 2  # pragma: no cover - the fail-closed exception always wins.
        required = {
            "--artifact-root": args.artifact_root,
            "--static-artifact": args.static_artifact,
            "--expected-static-sha256": args.expected_static_sha256,
            "--rr2-input-manifest": args.rr2_input_manifest,
            "--expected-rr2-input-manifest-sha256": (
                args.expected_rr2_input_manifest_sha256
            ),
            "--pg19-data": args.pg19_data,
            "--pg19-manifest": args.pg19_manifest,
            "--prior-capacity-manifest": args.prior_capacity_manifest,
            "--model-dir": args.model_dir,
            "--code-ledger": args.code_ledger,
            "--model-artifact-ledger": args.model_artifact_ledger,
            "--model-weight-ledger": args.model_weight_ledger,
            "--protocol-manifest": args.protocol_manifest,
            "--run-id-receipt": args.run_id_receipt,
            "--expected-run-id-receipt-sha256": (
                args.expected_run_id_receipt_sha256
            ),
            "--gpu-assignment-receipt": args.gpu_assignment_receipt,
            "--expected-gpu-assignment-receipt-raw-sha256": (
                args.expected_gpu_assignment_receipt_raw_sha256
            ),
            "--private-model-view-manifest": args.private_model_view_manifest,
            "--expected-private-model-view-manifest-raw-sha256": (
                args.expected_private_model_view_manifest_raw_sha256
            ),
            "--model-load-authority": args.model_load_authority,
            "--expected-model-load-authority-raw-sha256": (
                args.expected_model_load_authority_raw_sha256
            ),
            "--expected-gpu-uuid": args.expected_gpu_uuid,
        }
        missing = [option for option, value in required.items() if value is None]
        _require(not missing, f"formal shard missing explicit inputs: {missing}")
        run_formal_gpu_shard(args)
        return 0

    # A production aggregate is also disabled until the producer exists.  The
    # Python API's explicit ``allow_synthetic_schema_fixture`` is for tests and
    # cannot be enabled from this CLI.
    expected_run_id = _require_run_id(args.run_id, "aggregate --run-id")
    if not GPU_LOOP_IMPLEMENTED:
        raise ProductionLoopNotImplemented(
            "aggregate CLI is disabled until the formal GPU shard loop is implemented"
        )
    _require(args.artifact_root is not None, "aggregate requires --artifact-root")
    _require(args.static_artifact is not None, "aggregate requires --static-artifact")
    _require(args.receipt_manifest is not None, "aggregate requires --receipt-manifest")
    _require(args.run_id_receipt is not None, "aggregate requires --run-id-receipt")
    _require(
        args.expected_run_id_receipt_sha256 is not None,
        "aggregate requires --expected-run-id-receipt-sha256",
    )
    _require(
        args.gpu_assignment_receipt is not None,
        "aggregate requires --gpu-assignment-receipt",
    )
    _require(
        args.expected_gpu_assignment_receipt_raw_sha256 is not None,
        "aggregate requires --expected-gpu-assignment-receipt-raw-sha256",
    )
    _require(
        args.private_model_view_manifest is not None,
        "aggregate requires --private-model-view-manifest",
    )
    _require(
        args.expected_private_model_view_manifest_raw_sha256 is not None,
        "aggregate requires --expected-private-model-view-manifest-raw-sha256",
    )
    _require(
        args.model_load_authority is not None,
        "aggregate requires --model-load-authority",
    )
    _require(
        args.expected_model_load_authority_raw_sha256 is not None,
        "aggregate requires --expected-model-load-authority-raw-sha256",
    )
    _require(
        args.model_load_closure is not None,
        "aggregate requires --model-load-closure",
    )
    _require(
        args.expected_model_load_closure_raw_sha256 is not None,
        "aggregate requires --expected-model-load-closure-raw-sha256",
    )
    _require(
        args.model_weight_ledger is not None
        and args.model_artifact_ledger is not None,
        "aggregate requires model weight/artifact ledgers",
    )
    static = strict_json_loads(args.static_artifact.read_bytes(), label="static artifact")
    receipts = strict_json_loads(args.receipt_manifest.read_bytes(), label="receipt manifest")
    result = aggregate_shards(
        receipts,
        expected_receipt_manifest_sha256=args.expected_receipt_manifest_sha256,
        static_artifact=static,
        static_artifact_sha256=args.expected_static_sha256,
        artifact_root=args.artifact_root,
        expected_run_id=expected_run_id,
        run_id_receipt_raw=args.run_id_receipt.read_bytes(),
        expected_run_id_receipt_sha256=args.expected_run_id_receipt_sha256,
        gpu_assignment_receipt_raw=args.gpu_assignment_receipt.read_bytes(),
        expected_gpu_assignment_receipt_raw_sha256=(
            args.expected_gpu_assignment_receipt_raw_sha256
        ),
        private_model_view_manifest_raw=(
            args.private_model_view_manifest.read_bytes()
        ),
        expected_private_model_view_manifest_raw_sha256=(
            args.expected_private_model_view_manifest_raw_sha256
        ),
        model_load_authority_raw=args.model_load_authority.read_bytes(),
        expected_model_load_authority_raw_sha256=(
            args.expected_model_load_authority_raw_sha256
        ),
        model_load_closure_raw=args.model_load_closure.read_bytes(),
        expected_model_load_closure_raw_sha256=(
            args.expected_model_load_closure_raw_sha256
        ),
        model_weight_ledger_raw=args.model_weight_ledger.read_bytes(),
        model_artifact_ledger_raw=args.model_artifact_ledger.read_bytes(),
    )
    _write_json(args.output, result)
    return 0 if result["formal_ready"] else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ReviewAuditError as exc:
        raise SystemExit(f"ForkAudit protocol rejected: {exc}") from exc


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "ARM_IDS",
    "FORMAL_RESIDENT_COUNTS",
    "GDN_BASE_POLICIES",
    "GDN_POLICY_TO_WITNESS",
    "GPU_LOOP_IMPLEMENTED",
    "IMPLEMENTATION_STATUS",
    "KV_POLICIES",
    "MUTANT_ASSIGNMENT_BY_RANK",
    "MUTANT_TARGET_CONTRACT",
    "ORACLE_MAX_RELATIVE_L2",
    "PROTOCOL",
    "ProductionLoopNotImplemented",
    "ReviewAuditError",
    "SHARD_SCHEMA_VERSION",
    "STATIC_SCHEMA_VERSION",
    "aggregate_shards",
    "artifact_reference",
    "bridge_named_gate_error",
    "canonical_json_bytes",
    "encode_inline_tensor",
    "formal_protocol_config",
    "main",
    "make_receipt_manifest",
    "make_static_artifact",
    "sha256_json",
    "strict_json_loads",
    "validate_oracle_selection_plan",
    "validate_static_artifact",
]
