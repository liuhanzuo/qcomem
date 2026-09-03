from __future__ import annotations

"""Preregister and strictly aggregate the R28 ForkAudit detector matrix.

The GPU runner is intentionally not trusted to summarize its own campaign.
This CPU-only builder checks every rank, case, receipt, detector cell, FP32
sidecar, hardware identity, and the separately captured RR2 all-gates-on
receipt before emitting a canonical summary.  A scientifically negative
campaign is valid; an operationally invalid case is not.
"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np


PREREG_SCHEMA = "forkaudit-detector-matrix-preregistration-v2"
RANK_SCHEMA = "forkaudit-detector-matrix-rank-v2"
SUMMARY_SCHEMA = "forkaudit-detector-matrix-summary-v2"
EXTERNAL_PIN_SCHEMA = "forkaudit-r28-preexecution-external-pin-payload-v1"
EXTERNAL_PIN_PREEXECUTION_REVISION = 5
EXTERNAL_PIN_SEMANTIC_RECEIPT_SCHEMA = (
    "forkaudit-r28-external-pin-semantic-validation-receipt-v1"
)
MUTANT_IDS = tuple(f"M{i}" for i in range(1, 10))
ASSIGNMENT: dict[int, tuple[str, ...]] = {
    0: ("M1", "M9"),
    1: ("M2",),
    2: ("M3",),
    3: ("M4",),
    4: ("M5",),
    5: ("M6",),
    6: ("M7",),
    7: ("M8",),
}
EXPECTED_GATES = {
    "M1": "KV_RESERVATION_DISJOINT",
    "M2": "KV_SEQUENCE_ID",
    "M3": "KV_TAIL_COW",
    "M4": "gdn_completed_vs_base_disjoint",
    "M5": "gdn_completed_vs_peers_disjoint",
    "M6": "POSITION_CANONICAL_VALUES",
    "M7": "MASK_CONTRACT",
    "M8": "KERNEL_CALLABLE_ID",
    "M9": "KV_PAGED_VIEW",
}
EXPECTED_GATE_PREDICATES = {
    "M1": {
        "predicate_function": "_runtime_require",
        "predicate_source": "qcomem_vllm_paged_multifork_resident",
        "callsite_file": "qcomem_vllm_paged_multifork_resident.py",
        "callsite_function": "validate_runtime_kv_ownership",
    },
    "M2": {
        "predicate_function": "_runtime_require",
        "predicate_source": "qcomem_vllm_paged_multifork_resident",
        "callsite_file": "qcomem_vllm_paged_multifork_resident.py",
        "callsite_function": "attention_forward",
    },
    "M3": {
        "predicate_function": "_runtime_require",
        "predicate_source": "qcomem_vllm_paged_multifork_resident",
        "callsite_file": "qcomem_vllm_paged_multifork_resident.py",
        "callsite_function": "validate_runtime_kv_ownership",
    },
    "M4": {
        "predicate_function": "_assert_sets_disjoint",
        "predicate_source": "qcomem_forkaudit_storage_witness",
        "callsite_file": "qcomem_forkaudit_storage_witness.py",
        "callsite_function": "replay_gdn_storage_witness",
    },
    "M5": {
        "predicate_function": "_assert_sets_disjoint",
        "predicate_source": "qcomem_forkaudit_storage_witness",
        "callsite_file": "qcomem_forkaudit_storage_witness.py",
        "callsite_function": "replay_gdn_storage_witness",
    },
    "M6": {
        "predicate_function": "validate_qwen35_post_rope_position_ids",
        "predicate_source": "qcomem_qwen35_vllm_paged_integration",
        "callsite_file": "qcomem_vllm_paged_multifork_resident.py",
        "callsite_function": "attention_forward",
    },
    "M7": {
        "predicate_function": "_runtime_require",
        "predicate_source": "qcomem_vllm_paged_multifork_resident",
        "callsite_file": "qcomem_vllm_paged_multifork_resident.py",
        "callsite_function": "attention_forward",
    },
    "M8": {
        "predicate_function": "_runtime_require",
        "predicate_source": "qcomem_vllm_paged_multifork_resident",
        "callsite_file": "qcomem_vllm_paged_multifork_resident.py",
        "callsite_function": "attention_forward",
    },
    "M9": {
        "predicate_function": "_runtime_require",
        "predicate_source": "qcomem_vllm_paged_multifork_resident",
        "callsite_file": "run_qcomem_qwen35_forkaudit_detector_matrix_v2.py",
        "callsite_function": "attention_forward",
    },
}
TARGET_REQUESTS = {
    "M1": 1,
    "M2": 0,
    "M3": 0,
    "M4": 0,
    "M5": 1,
    "M6": 0,
    "M7": 0,
    "M8": 0,
    "M9": 0,
}
TARGET_CONTRACTS = {
    "M1": "request1.first_full.reservation_values",
    "M2": "request0.ledger.first_full.sequence_binding",
    "M3": "request0.first_full.detach_partial_document_tail_callable",
    "M4": "request0.first_linear.conv_state0.persistent_base_binding",
    "M5": "request1.first_linear.conv_state0.peer_request_binding",
    "M6": "request0.first_full.post_rope_position_ids_plus_one",
    "M7": "request0.first_full.materialized_all_true_mask",
    "M8": "request0.ledger.unified_attention_callable",
    "M9": "request0.first_full.key_cache_representation",
}
EXPECTED_HORIZON_STAGES = {
    "M1": ("measured",),
    "M2": ("measured",),
    "M3": ("prefix-r0", "prefix-r1", "continuation-r0"),
    "M4": ("prefix-r0", "continuation-r0"),
    "M5": ("prefix-r0", "prefix-r1", "continuation-r1"),
    "M6": ("measured",),
    "M7": ("measured",),
    "M8": ("measured",),
    "M9": ("measured",),
}
LANES = ("clean", "target_suppressed")
COMPLETION_STATUSES = ("completed", "classified_abort", "operational_invalid")
CLASSIFICATIONS = (
    "completed_semantics",
    "other_forkaudit_gate",
    "production_assertion",
    "fault_payload_abort",
    "operational_invalid",
)
EVALUATION_STATUSES = ("evaluated", "not_evaluated")
M9_ASSERTION_ALLOWLIST_ID = "PA-M9-Q16-PAIRED-VIEWS-v1"
M9_ASSERTION_ALLOWLIST = {
    "mutant_id": "M9",
    "exception_module": "qcomem_vllm_paged_kernel",
    "exception_type": "QComemPagedKernelError",
    "exact_message": "fused backend requires paired Q16 paged views",
    "stack_file": "qcomem_vllm_paged_kernel.py",
    "stack_function": "_paired_sequence",
}
M7_ASSERTION_ALLOWLIST_ID = "PA-M7-CANONICAL-MASK-v1"
M7_ASSERTION_ALLOWLIST = {
    "mutant_id": "M7",
    "exception_module": "qcomem_vllm_paged_kernel",
    "exception_type": "QComemPagedKernelError",
    "exact_message": "vLLM fused backend cannot replace this non-canonical attention mask",
    "stack_file": "qcomem_vllm_paged_kernel.py",
    "stack_function": "validate_canonical_tail_causal_mask",
}
PRODUCTION_ASSERTION_ALLOWLISTS = {
    M7_ASSERTION_ALLOWLIST_ID: M7_ASSERTION_ALLOWLIST,
    M9_ASSERTION_ALLOWLIST_ID: M9_ASSERTION_ALLOWLIST,
}
M8_FAULT_PAYLOAD_PIN = {
    "mutant_id": "M8",
    "classification": "fault_payload_abort",
    "exception_type": "AssertionError",
    "exact_message": "matrix M8 sentinel executed",
    "stack_file": "run_qcomem_qwen35_forkaudit_detector_matrix_v2.py",
    "stack_function": "m8_sentinel",
    "never_production_assertion": True,
}
TEACHER_FORCING = {
    "M3": {"request_index": 0, "query_token_index": 31},
    "M4": {"request_index": 0, "query_token_index": 31},
    "M5": {"request_index": 1, "query_token_index": 31},
}
CLASSIFICATION_PRECEDENCE = (
    "operational_invalid",
    "other_forkaudit_gate",
    "production_assertion",
    "fault_payload_abort",
    "completed_semantics",
)
PIN_CLASSIFICATION_PRECEDENCE = [
    "operational_invalid",
    "other ForkAudit gate after target suppression",
    "allowlisted production assertion with exact stack provenance",
    "fault-payload abort with exact preregistered message",
    "completed semantics",
]
PIN_R28_DETECTORS = [
    "other ForkAudit gate after target suppression",
    "allowlisted production assertion with stack provenance",
    "fault-payload abort",
    "token-only comparison when evaluated",
    "full-logit exact comparison when evaluated",
]
PIN_UNALLOWLISTED_EXCEPTION_POLICY = (
    "Any exception that is neither a non-target ForkAudit gate, the exact M8 "
    "fault-payload abort, nor an exact M7 or M9 production assertion is "
    "operational_invalid and cannot count as a detector catch."
)
PIN_FIXED_STACK = {
    "cuda": "12.9",
    "document_tokens": 4095,
    "gdn_policy": "borrow-immutable-base-functional-rebind",
    "hardware": "8 distinct NVIDIA H20-3e devices, one original RR2 rank/book per device",
    "kv": "Q16/BF16 paged KV, page size 128",
    "kv_policy": "vllm-q16-shared-document-reuse",
    "query_tokens": 32,
    "resident_count": 2,
    "torch": "2.11.0+cu129",
    "transformers": "5.14.1",
    "vllm": "0.26",
}
PIN_TEACHER_FORCING_RULE = {
    "argmax_feedback_forbidden": True,
    "coordinate_M3": "frozen_query_bank[rank][request=0][token=31]",
    "coordinate_M4": "frozen_query_bank[rank][request=0][token=31]",
    "coordinate_M5": "frozen_query_bank[rank][request=1][token=31]",
    "token_source": "last token of the corresponding frozen 32-token query",
}
PIN_M8_FAULT_PAYLOAD = {
    "fault_id": "M8",
    "message": M8_FAULT_PAYLOAD_PIN["exact_message"],
    "never_classify_as_production_assertion": True,
    "required_provenance": "the preregistered M8 replacement callable",
}
EXTERNAL_PIN_VALIDATION_CHECKS = (
    "raw_payload_sha256",
    "schema_revision_and_prospective_timing",
    "experiment_identity",
    "scope_supersession_raw_sha256",
    "rr2_authority",
    "frozen_model_and_data_inputs",
    "fixed_stack_and_geometry",
    "case_counts_and_rank_assignment",
    "teacher_forcing_coordinates",
    "m7_m9_production_assertion_allowlists",
    "m8_fault_payload_abort",
    "classification_precedence_and_no_generic_crash",
)
PIN_RUNTIME_VERSION_EXECUTION_GATE = {
    "torch": PIN_FIXED_STACK["torch"],
    "cuda": PIN_FIXED_STACK["cuda"],
    "transformers": PIN_FIXED_STACK["transformers"],
    "vllm": PIN_FIXED_STACK["vllm"],
    "runner_gate": "audit_frozen_kernel_environment(matches_frozen_environment=True)",
}
COMPARISON_DEFINITIONS = {
    "token_only": {
        "horizon": "all preregistered stages must complete in clean and mutant",
        "exact_sha": "equality of SHA-256 over canonical ordered argmax token integers",
        "argmax_equal": "all stage-wise argmax token integers are equal",
        "max_abs": "not applicable; null",
        "relative_l2": "not applicable; null",
        "caught": "logical negation of argmax_equal",
    },
    "full_logit": {
        "horizon": "all preregistered stages must complete in clean and mutant",
        "exact_sha": "all corresponding FP32 logit byte SHA-256 values are equal",
        "argmax_equal": "all corresponding FP32 logit argmax indices are equal",
        "max_abs": "maximum absolute FP32 element difference across the full horizon",
        "relative_l2": "L2(mutant-clean)/max(L2(clean),1e-30) across the full horizon",
        "caught": "logical negation of exact_sha",
    },
}
SHA_RE = re.compile(r"[0-9a-f]{64}")


class BuildError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BuildError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(canonical_bytes(value) + b"\n")
    pending.replace(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA_RE.fullmatch(value) is not None, label)
    return value


def _plain_int(value: Any, label: str, *, minimum: int = 0) -> int:
    require(isinstance(value, int) and not isinstance(value, bool), label)
    require(value >= minimum, label)
    return value


def _finite_number(value: Any, label: str, *, minimum: float | None = None) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), label)
    number = float(value)
    require(math.isfinite(number), label)
    if minimum is not None:
        require(number >= minimum, label)
    return number


def _check_file(path: Path, expected: str, label: str) -> bytes:
    _sha(expected, f"{label} expected SHA")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected, f"{label} SHA drift")
    return payload


def _pin_production_assertion_allowlist() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for allowlist_id in (M7_ASSERTION_ALLOWLIST_ID, M9_ASSERTION_ALLOWLIST_ID):
        pin = PRODUCTION_ASSERTION_ALLOWLISTS[allowlist_id]
        result.append(
            {
                "allowlist_id": allowlist_id,
                "exception_type": pin["exception_type"],
                "fault_id": pin["mutant_id"],
                "message": pin["exact_message"],
                "required_provenance": {
                    "module": pin["exception_module"],
                    "stack_file": pin["stack_file"],
                    "stack_function": pin["stack_function"],
                },
            }
        )
    return result


def _external_pin_semantic_receipt(
    *, payload_sha256: str, scope_sha256: str
) -> dict[str, Any]:
    return {
        "schema_version": EXTERNAL_PIN_SEMANTIC_RECEIPT_SCHEMA,
        "validated": True,
        "payload_raw_sha256": payload_sha256,
        "scope_supersession_raw_sha256": scope_sha256,
        "pin_schema_version": EXTERNAL_PIN_SCHEMA,
        "pin_preexecution_revision": EXTERNAL_PIN_PREEXECUTION_REVISION,
        "checks": {name: True for name in EXTERNAL_PIN_VALIDATION_CHECKS},
        "runtime_version_execution_gate": PIN_RUNTIME_VERSION_EXECUTION_GATE,
    }


def validate_external_pin_payload(
    *,
    external_pin_payload: Path,
    expected_external_pin_sha256: str,
    scope_supersession: Path,
    input_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Fail closed if the prospectively frozen rev5 pin drifts semantically."""

    raw = _check_file(
        external_pin_payload,
        expected_external_pin_sha256,
        "external preexecution pin payload",
    )
    try:
        pin = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise BuildError("external pin JSON") from error
    require(isinstance(pin, dict), "external pin object")
    require(
        pin.get("schema_version") == EXTERNAL_PIN_SCHEMA
        and pin.get("preexecution_revision") == EXTERNAL_PIN_PREEXECUTION_REVISION,
        "external pin schema/revision",
    )
    require(
        pin.get("created_before_candidate_execution") is True
        and pin.get("candidate_outputs_observed_at_creation") is False,
        "external pin prospective timing",
    )
    require(
        pin.get("experiment_id") == "E-R28-FULL-DETECTOR-MATRIX",
        "external pin experiment identity",
    )

    scope_sha = sha256_file(scope_supersession)
    scope = pin.get("scope_supersession")
    require(isinstance(scope, dict), "external pin scope binding")
    require(
        scope
        == {
            "path": (
                "paper_autonomous_multifork_iteration/evidence/"
                "r28_full_detector_matrix/scope-supersession.json"
            ),
            "sha256": scope_sha,
        },
        "external pin scope raw SHA",
    )

    rr2 = pin.get("inherited_rr2_authority")
    require(isinstance(rr2, dict), "external pin RR2 authority")
    require(
        rr2.get("original_rr2_run_id") == input_binding.get("original_rr2_run_id")
        and rr2.get("original_rr2_receipt_manifest_sha256")
        == input_binding.get("original_rr2_receipt_manifest_sha256")
        and rr2.get("executed_source_ledger_sha256")
        == input_binding.get("imported_rr2_code_ledger_sha256"),
        "external pin RR2 authority drift",
    )

    frozen = pin.get("frozen_inputs")
    require(isinstance(frozen, dict), "external pin frozen inputs")
    require(
        frozen
        == {
            "frozen_query_banks_sha256": input_binding.get(
                "frozen_query_banks_sha256"
            ),
            "model": "Qwen/Qwen3.5-35B-A3B",
            "model_artifact_ledger_sha256": input_binding.get(
                "artifact_ledger_raw_sha256"
            ),
            "model_revision": input_binding.get("model_revision"),
            "model_weight_ledger_sha256": input_binding.get(
                "weight_ledger_raw_sha256"
            ),
            "pg19_data_sha256": input_binding.get("pg19_sha256"),
            "pg19_manifest_sha256": input_binding.get("pg19_manifest_sha256"),
            "pg19_windows_sha256": input_binding.get("windows_sha256"),
        },
        "external pin frozen input drift",
    )
    require(pin.get("fixed_stack") == PIN_FIXED_STACK, "external pin fixed stack drift")

    design = pin.get("case_design")
    require(isinstance(design, dict), "external pin case design")
    require(
        design.get("fresh_case_count") == 18
        and design.get("clean_cases") == 9
        and design.get("target_suppressed_mutant_cases") == 9
        and design.get("discarded_warmup_before_allocator_baseline") is True,
        "external pin case counts/warmup",
    )
    require(
        design.get("rank_assignment")
        == {str(rank): list(ids) for rank, ids in ASSIGNMENT.items()},
        "external pin rank assignment",
    )
    require(
        design.get("case_order_per_fault")
        == [
            "fresh matched-clean with all gates enabled",
            (
                "separately rebuilt mutant with only the preregistered target "
                "gate suppressed"
            ),
        ],
        "external pin case order",
    )
    require(
        design.get("teacher_forced_continuation_fault_ids") == ["M3", "M4", "M5"]
        and design.get("teacher_forced_continuation_rule")
        == PIN_TEACHER_FORCING_RULE,
        "external pin teacher forcing drift",
    )

    detector = pin.get("detector_policy")
    require(isinstance(detector, dict), "external pin detector policy")
    require(
        detector.get("production_assertion_allowlist")
        == _pin_production_assertion_allowlist(),
        "external pin M7/M9 assertion allowlist drift",
    )
    require(
        detector.get("fault_payload_abort") == PIN_M8_FAULT_PAYLOAD,
        "external pin M8 payload drift",
    )
    require(
        detector.get("classification_precedence") == PIN_CLASSIFICATION_PRECEDENCE
        and detector.get("r28_detectors") == PIN_R28_DETECTORS
        and detector.get("unallowlisted_exception_policy")
        == PIN_UNALLOWLISTED_EXCEPTION_POLICY,
        "external pin generic crash/classification drift",
    )
    return _external_pin_semantic_receipt(
        payload_sha256=expected_external_pin_sha256,
        scope_sha256=scope_sha,
    )


def _validate_external_pin_semantic_receipt(
    value: Any,
    *,
    source_binding: Mapping[str, Any],
    input_binding: Mapping[str, Any],
) -> None:
    require(isinstance(value, dict), "external pin semantic validation receipt")
    require(
        set(value)
        == {
            "schema_version",
            "validated",
            "payload_raw_sha256",
            "scope_supersession_raw_sha256",
            "pin_schema_version",
            "pin_preexecution_revision",
            "checks",
            "runtime_version_execution_gate",
        },
        "external pin semantic validation receipt fields",
    )
    require(
        value.get("schema_version") == EXTERNAL_PIN_SEMANTIC_RECEIPT_SCHEMA
        and value.get("validated") is True
        and value.get("pin_schema_version") == EXTERNAL_PIN_SCHEMA
        and value.get("pin_preexecution_revision")
        == EXTERNAL_PIN_PREEXECUTION_REVISION,
        "external pin semantic validation receipt identity",
    )
    require(
        value.get("payload_raw_sha256")
        == source_binding.get("external_pin_payload_sha256")
        == input_binding.get("external_pin_payload_sha256"),
        "external pin semantic validation payload binding",
    )
    require(
        value.get("scope_supersession_raw_sha256")
        == source_binding.get("scope_supersession_sha256"),
        "external pin semantic validation scope binding",
    )
    require(
        value.get("checks")
        == {name: True for name in EXTERNAL_PIN_VALIDATION_CHECKS},
        "external pin semantic validation checks",
    )
    require(
        value.get("runtime_version_execution_gate")
        == PIN_RUNTIME_VERSION_EXECUTION_GATE,
        "external pin runtime version gate",
    )


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    original_sha = sha256_file(args.original_receipt_manifest)
    require(isinstance(args.model_revision, str) and bool(args.model_revision), "model revision")
    require(
        isinstance(args.original_rr2_run_id, str) and bool(args.original_rr2_run_id),
        "original RR2 run id",
    )
    external_pin_sha = _sha(
        args.external_pin_payload_sha256, "external pin payload SHA"
    )
    input_binding = {
        "model_revision": args.model_revision,
        "weight_ledger_raw_sha256": _sha(args.weight_ledger_sha256, "weight SHA"),
        "artifact_ledger_raw_sha256": _sha(
            args.artifact_ledger_sha256, "artifact SHA"
        ),
        "pg19_sha256": _sha(args.pg19_sha256, "PG19 SHA"),
        "pg19_manifest_sha256": _sha(
            args.pg19_manifest_sha256, "PG19 manifest SHA"
        ),
        "windows_sha256": _sha(args.windows_sha256, "windows SHA"),
        "frozen_query_banks_sha256": _sha(
            args.frozen_query_banks_sha256, "query banks SHA"
        ),
        "original_rr2_run_id": args.original_rr2_run_id,
        "original_rr2_receipt_manifest_sha256": original_sha,
        "code_ledger_sha256": _sha(args.code_ledger_sha256, "code ledger SHA"),
        "imported_rr2_code_ledger_sha256": _sha(
            args.imported_rr2_code_ledger_sha256,
            "imported RR2 code ledger SHA",
        ),
        "external_pin_payload_sha256": external_pin_sha,
    }
    external_pin_semantic_validation = validate_external_pin_payload(
        external_pin_payload=args.external_pin_payload,
        expected_external_pin_sha256=external_pin_sha,
        scope_supersession=args.scope_supersession,
        input_binding=input_binding,
    )
    value = {
        "schema_version": PREREG_SCHEMA,
        "created_before_candidate_outputs": True,
        "workstream_id": "E-R28-FULL-DETECTOR-MATRIX",
        "runner_sha256": sha256_file(args.runner),
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "replay_sha256": sha256_file(args.replay),
        "source_binding": {
            "runner_sha256": sha256_file(args.runner),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
            "replay_sha256": sha256_file(args.replay),
            "test_sha256": sha256_file(args.test_file),
            "launcher_sha256": sha256_file(args.launcher),
            "gate_policy_sha256": sha256_file(args.gate_policy),
            "qs_config_sha256": sha256_file(args.qs_config),
            "scope_supersession_sha256": sha256_file(args.scope_supersession),
            "external_pin_payload_sha256": external_pin_sha,
        },
        "rank_schema_version": RANK_SCHEMA,
        "summary_schema_version": SUMMARY_SCHEMA,
        "assignment": {str(rank): list(ids) for rank, ids in ASSIGNMENT.items()},
        "mutant_ids": list(MUTANT_IDS),
        "lanes": list(LANES),
        "expected_gate_ids": EXPECTED_GATES,
        "gate_predicate_bindings": EXPECTED_GATE_PREDICATES,
        "target_requests": TARGET_REQUESTS,
        "target_contracts": TARGET_CONTRACTS,
        "expected_horizon_stages": {
            mutant_id: list(stages)
            for mutant_id, stages in EXPECTED_HORIZON_STAGES.items()
        },
        "minimum_fresh_cases": 18,
        "required_world_size": 8,
        "required_hardware": {
            "name": "NVIDIA H20-3e",
            "compute_capability": [9, 0],
            "distinct_gpu_uuids": 8,
        },
        "input_binding": input_binding,
        "external_pin_semantic_validation": external_pin_semantic_validation,
        "production_assertion_allowlist": PRODUCTION_ASSERTION_ALLOWLISTS,
        "teacher_forcing": TEACHER_FORCING,
        "m8_fault_payload_pin": M8_FAULT_PAYLOAD_PIN,
        "classification_precedence": list(CLASSIFICATION_PRECEDENCE),
        "comparison_definitions": COMPARISON_DEFINITIONS,
        "partial_abort_policy": {
            "retain_and_hash_partial_stage_sidecars": True,
            "token_only_until_complete_horizon": "not_evaluated",
            "full_logit_until_complete_horizon": "not_evaluated",
            "not_evaluated_caught_value": None,
        },
        "hard_operational_validity_gates": {
            "discarded_warmup_precedes_post_model_allocator_baseline": True,
            "after_case_allocated_and_reserved_equal_post_warmup_baseline": True,
            "injector_target_restored": True,
            "selective_suppression_hooks_restored": True,
            "case_state_discarded": True,
            "registered_attention_backends_removed": True,
            "attention_implementation_restored": True,
            "traceback_references_cleared": True,
        },
        "policies": {
            "one_fresh_clean_and_one_fresh_target_suppressed_case_per_fault": True,
            "only_preregistered_target_gate_may_be_suppressed": True,
            "valid_scientific_negatives_retained": True,
            "not_evaluated_never_converted_to_not_caught": True,
            "operational_invalidity_rejected": True,
            "separate_rr2_all_gates_on_reference_not_pooled_as_rate": True,
        },
    }
    write_json(args.output, value)
    return value


def original_receipts(
    *, original_receipt_manifest: Path, original_rr2_root: Path
) -> dict[str, dict[str, Any]]:
    manifest_raw = original_receipt_manifest.read_bytes()
    manifest = json.loads(manifest_raw)
    require(
        manifest.get("schema_version") == "qcomem-forkaudit-detached-receipts-v1",
        "RR2 manifest schema",
    )
    rows: dict[str, dict[str, Any]] = {}
    for shard_ref in manifest.get("shards", []):
        relative = shard_ref.get("relative_path")
        require(isinstance(relative, str) and relative, "RR2 shard relative path")
        path = original_rr2_root / "raw" / relative
        payload = path.read_bytes()
        require(len(payload) == shard_ref.get("bytes"), "RR2 shard bytes")
        require(sha256_bytes(payload) == shard_ref.get("sha256"), "RR2 shard SHA")
        shard = json.loads(payload)
        for mutant_id, case in shard.get("fault_campaign", {}).get("mutants", {}).items():
            require(mutant_id not in rows, f"duplicate RR2 {mutant_id}")
            outcome = case.get("outcome", {})
            clean = case.get("matched_clean", {}).get("outcome", {})
            rows[mutant_id] = {
                "source": "separate_rr2_all_gates_on_w_run",
                "run_id": manifest.get("run_id"),
                "rank": shard.get("rank"),
                "shard_relative_path": relative,
                "shard_sha256": shard_ref.get("sha256"),
                "classification": outcome.get("classification"),
                "expected_gate_id": outcome.get("expected_gate_id"),
                "observed_gate_id": outcome.get("observed_gate_id"),
                "restoration_verified": outcome.get("restoration_verified"),
                "matched_clean_classification": clean.get("classification"),
            }
    require(set(rows) == set(MUTANT_IDS), "RR2 mutant coverage")
    for mutant_id, row in rows.items():
        expected = EXPECTED_GATES[mutant_id]
        require(row["classification"] == "detected_expected_gate", f"{mutant_id} RR2 class")
        require(
            row["expected_gate_id"] == row["observed_gate_id"] == expected,
            f"{mutant_id} RR2 gate binding",
        )
        require(row["restoration_verified"] is True, f"{mutant_id} RR2 restoration")
        require(
            row["matched_clean_classification"] == "clean_pass",
            f"{mutant_id} RR2 clean",
        )
    return rows


def _validate_hardware(value: Any, rank: int) -> dict[str, Any]:
    require(isinstance(value, dict), f"rank {rank} hardware")
    required = {
        "name",
        "uuid",
        "compute_capability",
        "memory_mib",
        "torch_version",
        "torch_cuda",
    }
    require(required <= set(value), f"rank {rank} hardware fields")
    require(value["name"] == "NVIDIA H20-3e", f"rank {rank} exact H20-3e name")
    require(
        isinstance(value["uuid"], str) and value["uuid"].startswith("GPU-") and len(value["uuid"]) > 8,
        f"rank {rank} GPU UUID",
    )
    require(value["compute_capability"] == [9, 0], f"rank {rank} sm90")
    _plain_int(value["memory_mib"], f"rank {rank} memory", minimum=1)
    require(
        value["torch_version"] == PIN_FIXED_STACK["torch"],
        f"rank {rank} exact torch version",
    )
    require(
        value["torch_cuda"] == PIN_FIXED_STACK["cuda"],
        f"rank {rank} exact CUDA version",
    )
    return dict(value)


def _validate_preregistration(prereg: Mapping[str, Any]) -> None:
    require(prereg.get("schema_version") == PREREG_SCHEMA, "preregistration schema")
    require(prereg.get("created_before_candidate_outputs") is True, "preregistration timing")
    require(prereg.get("workstream_id") == "E-R28-FULL-DETECTOR-MATRIX", "workstream id")
    _sha(prereg.get("runner_sha256"), "preregistered runner SHA")
    _sha(prereg.get("builder_sha256"), "preregistered builder SHA")
    _sha(prereg.get("replay_sha256"), "preregistered replay SHA")
    source_binding = prereg.get("source_binding")
    require(isinstance(source_binding, dict), "source binding")
    require(
        set(source_binding)
        == {
            "runner_sha256",
            "builder_sha256",
            "replay_sha256",
            "test_sha256",
            "launcher_sha256",
            "gate_policy_sha256",
            "qs_config_sha256",
            "scope_supersession_sha256",
            "external_pin_payload_sha256",
        },
        "source binding fields",
    )
    for field, value in source_binding.items():
        _sha(value, f"source binding {field}")
    require(source_binding["runner_sha256"] == prereg["runner_sha256"], "runner source binding")
    require(source_binding["builder_sha256"] == prereg["builder_sha256"], "builder source binding")
    require(source_binding["replay_sha256"] == prereg["replay_sha256"], "replay source binding")
    require(prereg.get("rank_schema_version") == RANK_SCHEMA, "rank schema preregistration")
    require(prereg.get("summary_schema_version") == SUMMARY_SCHEMA, "summary schema preregistration")
    require(
        prereg.get("assignment") == {str(rank): list(ids) for rank, ids in ASSIGNMENT.items()},
        "preregistered assignment",
    )
    require(prereg.get("mutant_ids") == list(MUTANT_IDS), "preregistered mutants")
    require(prereg.get("lanes") == list(LANES), "preregistered lanes")
    require(prereg.get("expected_gate_ids") == EXPECTED_GATES, "preregistered gates")
    require(
        prereg.get("gate_predicate_bindings") == EXPECTED_GATE_PREDICATES,
        "preregistered gate predicate bindings",
    )
    require(prereg.get("target_requests") == TARGET_REQUESTS, "preregistered requests")
    require(prereg.get("target_contracts") == TARGET_CONTRACTS, "preregistered target contracts")
    require(
        prereg.get("expected_horizon_stages")
        == {key: list(value) for key, value in EXPECTED_HORIZON_STAGES.items()},
        "preregistered horizons",
    )
    require(prereg.get("minimum_fresh_cases") == 18, "preregistered case count")
    require(prereg.get("required_world_size") == 8, "preregistered world size")
    require(
        prereg.get("required_hardware")
        == {
            "name": "NVIDIA H20-3e",
            "compute_capability": [9, 0],
            "distinct_gpu_uuids": 8,
        },
        "preregistered hardware",
    )
    require(
        prereg.get("production_assertion_allowlist")
        == PRODUCTION_ASSERTION_ALLOWLISTS,
        "production assertion allowlist",
    )
    require(prereg.get("teacher_forcing") == TEACHER_FORCING, "teacher forcing pin")
    require(
        prereg.get("m8_fault_payload_pin") == M8_FAULT_PAYLOAD_PIN,
        "M8 fault payload pin",
    )
    require(
        prereg.get("classification_precedence") == list(CLASSIFICATION_PRECEDENCE),
        "classification precedence",
    )
    require(
        prereg.get("comparison_definitions") == COMPARISON_DEFINITIONS,
        "comparison definitions",
    )
    require(
        prereg.get("partial_abort_policy")
        == {
            "retain_and_hash_partial_stage_sidecars": True,
            "token_only_until_complete_horizon": "not_evaluated",
            "full_logit_until_complete_horizon": "not_evaluated",
            "not_evaluated_caught_value": None,
        },
        "partial abort policy",
    )
    require(
        prereg.get("hard_operational_validity_gates")
        == {
            "discarded_warmup_precedes_post_model_allocator_baseline": True,
            "after_case_allocated_and_reserved_equal_post_warmup_baseline": True,
            "injector_target_restored": True,
            "selective_suppression_hooks_restored": True,
            "case_state_discarded": True,
            "registered_attention_backends_removed": True,
            "attention_implementation_restored": True,
            "traceback_references_cleared": True,
        },
        "hard operational validity gates",
    )
    input_binding = prereg.get("input_binding")
    require(isinstance(input_binding, dict), "preregistered input binding")
    for field in (
        "model_revision",
        "weight_ledger_raw_sha256",
        "artifact_ledger_raw_sha256",
        "pg19_sha256",
        "pg19_manifest_sha256",
        "windows_sha256",
        "frozen_query_banks_sha256",
        "original_rr2_run_id",
        "original_rr2_receipt_manifest_sha256",
        "code_ledger_sha256",
        "imported_rr2_code_ledger_sha256",
        "external_pin_payload_sha256",
    ):
        require(field in input_binding, f"preregistered input {field}")
    for field in (
        "weight_ledger_raw_sha256",
        "artifact_ledger_raw_sha256",
        "pg19_sha256",
        "pg19_manifest_sha256",
        "windows_sha256",
        "frozen_query_banks_sha256",
        "original_rr2_receipt_manifest_sha256",
        "code_ledger_sha256",
        "imported_rr2_code_ledger_sha256",
        "external_pin_payload_sha256",
    ):
        _sha(input_binding[field], f"preregistered input {field}")
    require(
        source_binding["external_pin_payload_sha256"]
        == input_binding["external_pin_payload_sha256"],
        "external pin source/input binding",
    )
    _validate_external_pin_semantic_receipt(
        prereg.get("external_pin_semantic_validation"),
        source_binding=source_binding,
        input_binding=input_binding,
    )
    policies = prereg.get("policies")
    require(isinstance(policies, dict) and policies, "preregistered policies")
    require(all(value is True for value in policies.values()), "preregistered policy values")


def _validate_input_receipt(
    value: Any, *, prereg: Mapping[str, Any], prereg_sha: str, rank: int
) -> None:
    require(isinstance(value, dict), f"rank {rank} input receipt")
    expected = dict(prereg["input_binding"])
    expected["preregistration_sha256"] = prereg_sha
    require(set(expected) <= set(value), f"rank {rank} input receipt fields")
    for key, expected_value in expected.items():
        require(value.get(key) == expected_value, f"rank {rank} input {key}")


def _validate_eval_cell(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} cell")
    status = value.get("status")
    caught = value.get("caught")
    require(status in EVALUATION_STATUSES, f"{label} status")
    if status == "evaluated":
        require(isinstance(caught, bool), f"{label} evaluated caught")
    else:
        require(caught is None, f"{label} not-evaluated caught must be null")
    return dict(value)


def _validate_semantic_cell(value: Any, label: str, *, detector: str) -> dict[str, Any]:
    row = _validate_eval_cell(value, label)
    metrics = ("exact_sha", "argmax_equal", "max_abs", "relative_l2")
    require(set(metrics) <= set(row), f"{label} metric fields")
    if row["status"] == "not_evaluated":
        require(all(row[name] is None for name in metrics), f"{label} unknown metrics")
    else:
        require(isinstance(row["exact_sha"], bool), f"{label} exact SHA")
        require(isinstance(row["argmax_equal"], bool), f"{label} argmax")
        if detector == "token_only":
            require(
                row["max_abs"] is None and row["relative_l2"] is None,
                f"{label} token-only numeric metrics are not applicable",
            )
            require(row["exact_sha"] is row["argmax_equal"], f"{label} token exactness")
            require(row["caught"] is (not row["argmax_equal"]), f"{label} token caught derivation")
        else:
            require(detector == "full_logit", f"{label} detector definition")
            _finite_number(row["max_abs"], f"{label} max abs", minimum=0.0)
            _finite_number(row["relative_l2"], f"{label} relative L2", minimum=0.0)
            require(row["caught"] is (not row["exact_sha"]), f"{label} logit caught derivation")
    return row


def _validate_production_assertion(value: Any, label: str) -> dict[str, Any]:
    row = _validate_eval_cell(value, label)
    require("allowlist_id" in row and "provenance" in row, f"{label} metadata")
    if row["status"] == "evaluated" and row["caught"] is True:
        allowlist_id = row["allowlist_id"]
        require(allowlist_id in PRODUCTION_ASSERTION_ALLOWLISTS, f"{label} allowlist")
        pin = PRODUCTION_ASSERTION_ALLOWLISTS[allowlist_id]
        require(label.startswith(f"{pin['mutant_id']}:"), f"{label} allowlist applicability")
        provenance = row["provenance"]
        require(isinstance(provenance, dict), f"{label} provenance")
        require(
            provenance.get("exception_module") == pin["exception_module"]
            and provenance.get("exception_type") == pin["exception_type"]
            and provenance.get("exact_message") == pin["exact_message"],
            f"{label} exact assertion provenance",
        )
        stack = provenance.get("stack_provenance")
        require(isinstance(stack, list) and bool(stack), f"{label} stack provenance")
        require(
            any(
                isinstance(frame, dict)
                and frame.get("file") == pin["stack_file"]
                and frame.get("function") == pin["stack_function"]
                for frame in stack
            ),
            f"{label} stack callsite",
        )
    else:
        require(row["allowlist_id"] is None and row["provenance"] is None, f"{label} unused metadata")
    return row


def _validate_freshness(value: Any, label: str) -> None:
    require(isinstance(value, dict), f"{label} freshness")
    _sha(value.get("case_nonce_sha256"), f"{label} case nonce")
    require(value.get("fresh_persistent_cache") is True, f"{label} fresh persistent cache")
    require(value.get("fresh_request_cache_group") is True, f"{label} fresh request group")
    require(value.get("prior_case_state_reused") is False, f"{label} reused state")


def _validate_policy_receipt(value: Any, *, lane: str, expected_gate: str, label: str) -> None:
    require(isinstance(value, dict), f"{label} suppression restoration")
    require(
        value.get("schema_version") == "forkaudit-selective-gate-policy-receipt-v2",
        f"{label} policy schema",
    )
    target = None if lane == "clean" else expected_gate
    require(value.get("target_gate_id") == target, f"{label} policy target")
    require(
        value.get("lane") == ("all-gates-on" if lane == "clean" else "target-only-suppressed"),
        f"{label} policy lane",
    )
    events = value.get("events")
    require(isinstance(events, list), f"{label} policy events")
    require(value.get("suppressed_event_count") == len(events), f"{label} policy count")
    require(value.get("suppressed_gate_ids") == [event.get("gate_id") for event in events], f"{label} policy IDs")
    require(value.get("scope_integrity_before_restore") is True, f"{label} hook scope")
    require(
        value.get("all_original_function_identities_restored") is True,
        f"{label} hook restoration",
    )
    if lane == "clean":
        require(not events, f"{label} clean suppression")
    else:
        require(bool(events), f"{label} target suppression not exercised")
        mutant_id = label.split(":", 1)[0]
        predicate = EXPECTED_GATE_PREDICATES[mutant_id]
        for index, event in enumerate(events):
            require(isinstance(event, dict), f"{label} event {index}")
            require(
                event.get("schema_version")
                == "forkaudit-selective-gate-event-v2",
                f"{label} event schema",
            )
            require(event.get("gate_id") == expected_gate, f"{label} wrong suppressed gate")
            require(
                event.get("predicate_function") == predicate["predicate_function"]
                and event.get("predicate_source") == predicate["predicate_source"],
                f"{label} predicate binding",
            )
            require(event.get("ordinal") == index, f"{label} event ordinal")
            require(
                isinstance(event.get("message"), str) and bool(event["message"]),
                f"{label} event message",
            )
            callsite = event.get("callsite")
            require(isinstance(callsite, dict), f"{label} callsite")
            require(
                callsite.get("file") == predicate["callsite_file"]
                and callsite.get("function") == predicate["callsite_function"],
                f"{label} callsite binding",
            )
            _plain_int(callsite.get("line"), f"{label} callsite line", minimum=1)
            _sha(
                callsite.get("source_line_sha256"),
                f"{label} callsite source-line SHA",
            )
            callsite_copy = dict(callsite)
            observed_callsite_sha = callsite_copy.pop("callsite_sha256", None)
            require(
                observed_callsite_sha == sha256_json(callsite_copy),
                f"{label} callsite hash",
            )
            event_copy = dict(event)
            observed_sha = event_copy.pop("event_sha256", None)
            require(observed_sha == sha256_json(event_copy), f"{label} event hash")


def _validate_restoration(
    value: Any, *, lane: str, expected_gate: str, label: str
) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} injector restoration")
    require(value.get("verified") is True, f"{label} injector restoration verified")
    if lane == "clean":
        require(value.get("status") == "not_applicable", f"{label} clean restoration status")
        return dict(value)
    require(value.get("status") == "evaluated", f"{label} mutant restoration status")
    before = _sha(value.get("pre_sha256"), f"{label} pre SHA")
    mutated = _sha(value.get("mutated_sha256"), f"{label} mutated SHA")
    restored = _sha(value.get("restored_sha256"), f"{label} restored SHA")
    require(before == restored and mutated != before, f"{label} exact restoration")
    mutant_id = label.split(":", 1)[0]
    require(
        value.get("target_contract") == TARGET_CONTRACTS[mutant_id],
        f"{label} target contract",
    )
    return dict(value)


def _validate_m9_raw_tensor_receipt(
    value: Any,
    *,
    restoration: Mapping[str, Any],
    label: str,
) -> None:
    require(isinstance(value, dict), f"{label} M9 raw-tensor receipt")
    require(
        set(value)
        == {
            "raw_key_type",
            "raw_key_sha256",
            "raw_key_shape",
            "paired_value_type",
            "production_entrypoint",
        },
        f"{label} M9 raw-tensor receipt fields",
    )
    require(value.get("raw_key_type") == "torch.Tensor", f"{label} M9 raw key type")
    raw_sha = _sha(value.get("raw_key_sha256"), f"{label} M9 raw key SHA")
    raw_shape = value.get("raw_key_shape")
    require(
        isinstance(raw_shape, list)
        and bool(raw_shape)
        and all(
            isinstance(size, int) and not isinstance(size, bool) and size > 0
            for size in raw_shape
        ),
        f"{label} M9 raw key shape",
    )
    require(
        value.get("paired_value_type") == "Q16KernelPagedTensorView(value)",
        f"{label} M9 paired value type",
    )
    require(
        value.get("production_entrypoint")
        == "qcomem_vllm_paged_kernel.vllm_triton_q16_paged_attention_forward",
        f"{label} M9 production entrypoint",
    )
    pre = restoration.get("pre_descriptor")
    mutated = restoration.get("mutated_descriptor")
    restored = restoration.get("restored_descriptor")
    require(
        isinstance(pre, dict)
        and isinstance(mutated, dict)
        and isinstance(restored, dict),
        f"{label} M9 restoration descriptors",
    )
    require(
        pre
        == restored
        == {
            "representation": "q16-paged-key-view",
            "kind": "key",
            "paired_value_remains_q16": True,
        },
        f"{label} M9 restored paired Q16 key view",
    )
    tensor = mutated.get("tensor")
    require(
        mutated.get("representation") == "raw-torch-tensor"
        and mutated.get("paired_value_remains_q16") is True
        and isinstance(tensor, dict)
        and tensor.get("kind") == "tensor"
        and tensor.get("dtype") == "torch.bfloat16"
        and tensor.get("shape") == raw_shape
        and tensor.get("sha256") == raw_sha,
        f"{label} M9 raw tensor/paired-view binding",
    )
    require(
        sha256_json(pre) == restoration.get("pre_sha256")
        and sha256_json(mutated) == restoration.get("mutated_sha256")
        and sha256_json(restored) == restoration.get("restored_sha256"),
        f"{label} M9 restoration descriptor hashes",
    )


def _memory_cell(value: Any, label: str) -> dict[str, int]:
    require(isinstance(value, dict), label)
    allocated = _plain_int(value.get("allocated_bytes"), f"{label} allocated")
    reserved = _plain_int(value.get("reserved_bytes"), f"{label} reserved")
    return {"allocated_bytes": allocated, "reserved_bytes": reserved}


def _validate_cleanup(
    value: Any, label: str, *, expected_post_warmup_baseline: Mapping[str, int]
) -> None:
    require(isinstance(value, dict), f"{label} cleanup")
    require(value.get("verified") is True, f"{label} cleanup verified")
    before = _memory_cell(value.get("before_cell"), f"{label} before cell")
    after = _memory_cell(value.get("after_cleanup"), f"{label} after cleanup")
    baseline = _memory_cell(value.get("frozen_model_query_baseline"), f"{label} baseline")
    require(after == baseline, f"{label} allocator baseline recovery")
    require(before == baseline, f"{label} case did not start at allocator baseline")
    require(
        baseline == dict(expected_post_warmup_baseline),
        f"{label} cleanup baseline is not the discarded-warmup baseline",
    )
    for field in (
        "gc_collect_completed",
        "cuda_empty_cache_completed",
        "cuda_synchronize_completed",
        "current_allocated_and_reserved_exactly_recovered",
        "disposable_resident_request_group_discarded",
        "registered_attention_backends_removed",
        "attention_implementation_restored",
        "traceback_references_cleared",
    ):
        require(value.get(field) is True, f"{label} {field}")


def _validate_mutation_receipt(value: Any, *, lane: str, expected_gate: str, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label} mutation receipt")
    require(value.get("applied") is (lane == "target_suppressed"), f"{label} mutation applied")
    require(value.get("mutant_id") == label.split(":", 1)[0], f"{label} mutation mutant")
    mutant_id = label.split(":", 1)[0]
    teacher = value.get("teacher_forcing")
    if mutant_id in TEACHER_FORCING:
        require(isinstance(teacher, dict), f"{label} teacher forcing receipt")
        require(
            teacher.get("request_index") == TEACHER_FORCING[mutant_id]["request_index"]
            and teacher.get("query_token_index")
            == TEACHER_FORCING[mutant_id]["query_token_index"],
            f"{label} teacher forcing coordinate",
        )
        _plain_int(teacher.get("token_id"), f"{label} teacher token")
        _sha(teacher.get("source_token_sha256"), f"{label} teacher token SHA")
        require(
            teacher.get("source_coordinate")
            == (
                "frozen_query_bank[rank]"
                f"[{TEACHER_FORCING[mutant_id]['request_index']}]"
                f"[{TEACHER_FORCING[mutant_id]['query_token_index']}]"
            ),
            f"{label} teacher source coordinate",
        )
        require(
            teacher.get("independent_of_path_argmax") is True,
            f"{label} teacher independence",
        )
        require(teacher.get("argmax_feedback_used") is False, f"{label} argmax feedback")
    else:
        require(teacher is None, f"{label} unexpected teacher forcing")
    if lane == "target_suppressed":
        require(
            value.get("target_contract") == TARGET_CONTRACTS[mutant_id],
            f"{label} mutation target contract",
        )
        before = _sha(value.get("pre_descriptor_sha256"), f"{label} mutation pre SHA")
        mutated = _sha(
            value.get("mutated_descriptor_sha256"), f"{label} mutation descriptor SHA"
        )
        require(before != mutated, f"{label} mutation descriptor unchanged")
    return dict(value)


def _validate_warmup(value: Any, rank: int) -> dict[str, int]:
    label = f"rank {rank} discarded pre-baseline warmup"
    require(isinstance(value, dict), label)
    require(value.get("performed") is True, f"{label} performed")
    require(value.get("discarded") is True, f"{label} discarded")
    require(
        value.get("completed_before_case_nonces") is True,
        f"{label} ordering",
    )
    for field in (
        "gc_collect_completed",
        "cuda_empty_cache_completed",
        "cuda_synchronize_completed",
    ):
        require(value.get(field) is True, f"{label} {field}")
    require(
        value.get("all_gates_on_policy_restored") is True,
        f"{label} policy restoration",
    )
    return _memory_cell(value.get("post_warmup_baseline"), f"{label} baseline")


def _validate_case(
    case: Any,
    *,
    mutant_id: str,
    lane: str,
    sidecars: Sequence[Mapping[str, Any]],
    sidecar_payloads: Mapping[str, bytes],
    expected_post_warmup_baseline: Mapping[str, int],
) -> dict[str, Any]:
    label = f"{mutant_id}:{lane}"
    require(isinstance(case, dict), f"{label} case")
    require(case.get("case_id") == label, f"{label} case id")
    require(case.get("mutant_id") == mutant_id, f"{label} mutant id")
    require(case.get("lane") == lane, f"{label} lane")
    expected_gate = EXPECTED_GATES[mutant_id]
    require(case.get("expected_gate_id") == expected_gate, f"{label} expected gate")
    require(case.get("target_request") == TARGET_REQUESTS[mutant_id], f"{label} request")
    _validate_freshness(case.get("freshness_receipt"), label)
    outcome = case.get("outcome")
    require(isinstance(outcome, dict), f"{label} outcome")
    completion = outcome.get("completion_status")
    classification = outcome.get("classification")
    require(completion in COMPLETION_STATUSES, f"{label} completion status")
    require(classification in CLASSIFICATIONS, f"{label} classification")
    require(
        completion != "operational_invalid" and classification != "operational_invalid",
        f"{label} operational invalidity",
    )

    fork_audit = outcome.get("fork_audit")
    require(isinstance(fork_audit, dict), f"{label} ForkAudit receipt")
    target_events = fork_audit.get("target_suppression_events")
    require(isinstance(target_events, list), f"{label} target events")
    other_gate = _validate_eval_cell(fork_audit.get("other_gate"), f"{label} other gate")
    require("id" in other_gate, f"{label} other gate id")
    if other_gate["status"] == "evaluated" and other_gate["caught"]:
        require(
            isinstance(other_gate["id"], str)
            and other_gate["id"]
            and other_gate["id"] != expected_gate,
            f"{label} other gate identity",
        )
    else:
        require(other_gate["id"] is None, f"{label} unused other gate id")

    production = outcome.get("production")
    require(isinstance(production, dict), f"{label} production receipt")
    assertion = _validate_production_assertion(
        production.get("assertion"), f"{label} production assertion"
    )
    crash = _validate_eval_cell(
        production.get("nonassertion_crash"), f"{label} production crash"
    )
    fault_payload = _validate_eval_cell(
        production.get("fault_payload_abort"), f"{label} fault payload"
    )

    availability = outcome.get("output_availability")
    require(isinstance(availability, dict), f"{label} output availability")
    require(set(availability) >= {"token", "full_logit", "sidecar"}, f"{label} output fields")
    require(
        all(availability[name] in EVALUATION_STATUSES for name in ("token", "full_logit", "sidecar")),
        f"{label} output status",
    )
    semantics = outcome.get("semantics")
    require(isinstance(semantics, dict), f"{label} semantics")
    token = _validate_semantic_cell(
        semantics.get("token_only"), f"{label} token", detector="token_only"
    )
    logits = _validate_semantic_cell(
        semantics.get("full_logit"), f"{label} logits", detector="full_logit"
    )
    require(token["status"] == availability["token"], f"{label} token availability binding")
    require(logits["status"] == availability["full_logit"], f"{label} logit availability binding")
    if availability["full_logit"] == "evaluated":
        require(availability["sidecar"] == "evaluated", f"{label} evaluated logits need sidecar")
    case_sidecars = [row for row in sidecars if row.get("case_id") == label]
    observed_stages = [row["stage"] for row in case_sidecars]
    require(len(observed_stages) == len(set(observed_stages)), f"{label} duplicate sidecar stage")
    expected_stages = list(EXPECTED_HORIZON_STAGES[mutant_id])
    require(
        observed_stages == expected_stages[: len(observed_stages)],
        f"{label} sidecars are not an ordered horizon prefix",
    )
    observed_outputs = outcome.get("observed_outputs")
    require(isinstance(observed_outputs, list), f"{label} observed outputs")
    require(len(observed_outputs) == len(case_sidecars), f"{label} output/sidecar count")
    for index, (observed, sidecar) in enumerate(zip(observed_outputs, case_sidecars)):
        require(isinstance(observed, dict), f"{label} observed output {index}")
        require(observed.get("stage") == sidecar["stage"], f"{label} output stage")
        require(
            observed.get("full_logit_sha256") == sidecar["sha256"],
            f"{label} output SHA binding",
        )
        require(
            observed.get("sidecar_relative_path") == sidecar["relative_path"],
            f"{label} output path binding",
        )
        token_id = _plain_int(observed.get("token"), f"{label} observed token")
        vector = np.frombuffer(sidecar_payloads[sidecar["relative_path"]], dtype="<f4")
        require(token_id == int(np.argmax(vector)), f"{label} observed token/FP32 argmax")
    if availability["sidecar"] == "evaluated":
        require(bool(case_sidecars), f"{label} missing nonzero sidecar")
    else:
        require(not case_sidecars, f"{label} unexpected sidecar")

    measured_escape = outcome.get("measured_non_forkaudit_escape")
    require(
        measured_escape is None or isinstance(measured_escape, bool),
        f"{label} measured escape",
    )
    mutation_receipt = _validate_mutation_receipt(
        outcome.get("mutation_receipt"), lane=lane, expected_gate=expected_gate, label=label
    )
    restoration = _validate_restoration(
        outcome.get("injector_target_restoration"),
        lane=lane,
        expected_gate=expected_gate,
        label=label,
    )
    raw_tensor_receipt = mutation_receipt.get("raw_tensor_production_receipt")
    if lane == "target_suppressed":
        require(
            mutation_receipt.get("pre_descriptor_sha256")
            == restoration.get("pre_sha256")
            and mutation_receipt.get("mutated_descriptor_sha256")
            == restoration.get("mutated_sha256"),
            f"{label} mutation/restoration hash binding",
        )
    if mutant_id == "M9" and lane == "target_suppressed":
        if classification in ("completed_semantics", "production_assertion"):
            require(
                raw_tensor_receipt is not None,
                f"{label} M9 raw tensor did not reach production",
            )
        if raw_tensor_receipt is not None:
            _validate_m9_raw_tensor_receipt(
                raw_tensor_receipt,
                restoration=restoration,
                label=label,
            )
    else:
        require(
            raw_tensor_receipt is None,
            f"{label} unexpected M9 raw-tensor receipt",
        )
    _validate_cleanup(
        outcome.get("case_discard_allocator_recovery"),
        label,
        expected_post_warmup_baseline=expected_post_warmup_baseline,
    )
    hook = outcome.get("suppression_hook_restoration")
    _validate_policy_receipt(hook, lane=lane, expected_gate=expected_gate, label=label)
    require(target_events == hook.get("events"), f"{label} suppression event binding")

    if lane == "clean":
        require(completion == "completed", f"{label} clean completion")
        require(classification == "completed_semantics", f"{label} clean class")
        require(not target_events, f"{label} clean target event")
        require(other_gate["status"] == "evaluated" and other_gate["caught"] is False, f"{label} clean gate")
        require(assertion["status"] == "evaluated" and assertion["caught"] is False, f"{label} clean assertion")
        require(crash["status"] == "evaluated" and crash["caught"] is False, f"{label} clean crash")
        require(
            fault_payload["status"] == "evaluated"
            and fault_payload["caught"] is False,
            f"{label} clean payload",
        )
        require(
            all(
                availability[name] == "evaluated"
                for name in ("token", "full_logit", "sidecar")
            ),
            f"{label} clean outputs",
        )
        require(observed_stages == expected_stages, f"{label} incomplete clean horizon")
        require(token["caught"] is False and logits["caught"] is False, f"{label} clean divergence")
        require(measured_escape is None, f"{label} clean escape applicability")
    else:
        require(bool(target_events), f"{label} mutant target not suppressed")
        require(all(event.get("gate_id") == expected_gate for event in target_events), f"{label} wrong suppressed target")
        require(measured_escape is not None, f"{label} mutant escape missing")
        if classification == "completed_semantics":
            require(completion == "completed", f"{label} semantic completion")
            require(other_gate["status"] == "evaluated" and other_gate["caught"] is False, f"{label} semantic gate")
            require(assertion["status"] == "evaluated" and assertion["caught"] is False, f"{label} semantic assertion")
            require(crash["status"] == "evaluated" and crash["caught"] is False, f"{label} semantic crash")
            require(
                fault_payload["status"] == "evaluated"
                and fault_payload["caught"] is False,
                f"{label} semantic payload",
            )
            require(
                all(
                    availability[name] == "evaluated"
                    for name in ("token", "full_logit", "sidecar")
                ),
                f"{label} semantic outputs",
            )
            require(observed_stages == expected_stages, f"{label} incomplete semantic horizon")
            expected_escape = token["caught"] is False and logits["caught"] is False
            require(measured_escape is expected_escape, f"{label} escape derivation")
        else:
            require(completion == "classified_abort", f"{label} classified abort")
            require(
                token["status"] == logits["status"] == "not_evaluated",
                f"{label} partial horizon cannot evaluate semantic detectors",
            )
            if classification == "other_forkaudit_gate":
                require(other_gate["status"] == "evaluated" and other_gate["caught"] is True, f"{label} other gate class")
                require(
                    assertion["status"]
                    == crash["status"]
                    == fault_payload["status"]
                    == "not_evaluated"
                    and assertion["caught"] is None
                    and crash["caught"] is None
                    and fault_payload["caught"] is None,
                    f"{label} post-gate detector missingness",
                )
            elif classification == "production_assertion":
                require(mutant_id in ("M7", "M9"), f"{label} assertion applicability")
                require(assertion["status"] == "evaluated" and assertion["caught"] is True, f"{label} assertion class")
                require(
                    crash["status"] == fault_payload["status"] == "not_evaluated"
                    and crash["caught"] is None
                    and fault_payload["caught"] is None,
                    f"{label} post-assertion detector missingness",
                )
            elif classification == "fault_payload_abort":
                require(mutant_id == "M8", f"{label} payload abort applicability")
                require(assertion.get("allowlist_id") is None, f"{label} payload cannot masquerade as assertion")
                provenance = mutation_receipt.get("fault_payload_abort_provenance")
                require(isinstance(provenance, dict), f"{label} payload provenance")
                require(
                    provenance.get("exception_type")
                    == M8_FAULT_PAYLOAD_PIN["exception_type"]
                    and provenance.get("exact_message")
                    == M8_FAULT_PAYLOAD_PIN["exact_message"],
                    f"{label} payload exact type/message",
                )
                stack = provenance.get("stack_provenance")
                require(isinstance(stack, list) and bool(stack), f"{label} payload stack")
                require(
                    any(
                        isinstance(frame, dict)
                        and frame.get("file")
                        == M8_FAULT_PAYLOAD_PIN["stack_file"]
                        and frame.get("function")
                        == M8_FAULT_PAYLOAD_PIN["stack_function"]
                        for frame in stack
                    ),
                    f"{label} payload sentinel provenance",
                )
                require(
                    assertion["status"] == crash["status"] == "not_evaluated"
                    and assertion["caught"] is None
                    and crash["caught"] is None,
                    f"{label} payload production-detector missingness",
                )
                require(
                    fault_payload["status"] == "evaluated"
                    and fault_payload["caught"] is True,
                    f"{label} payload terminal detector",
                )
            require(measured_escape is False, f"{label} abort cannot escape")

        fault_payload_caught = (
            classification == "fault_payload_abort"
            and isinstance(mutation_receipt.get("fault_payload_abort_provenance"), dict)
        )
        if classification != "fault_payload_abort":
            require(
                mutation_receipt.get("fault_payload_abort_provenance") is None,
                f"{label} unexpected payload provenance",
            )
        caught_classifiers = sum(
            value is True
            for value in (
                other_gate["caught"],
                assertion["caught"],
                fault_payload["caught"],
                crash["caught"],
            )
        )
        require(caught_classifiers <= 1, f"{label} mutually exclusive classifier receipts")
        if other_gate["caught"] is True:
            derived_classification = "other_forkaudit_gate"
        elif assertion["caught"] is True:
            derived_classification = "production_assertion"
        elif fault_payload_caught:
            derived_classification = "fault_payload_abort"
        elif crash["caught"] is True:
            raise BuildError(f"{label} unallowlisted production non-assertion crash")
        else:
            derived_classification = "completed_semantics"
        require(
            classification == derived_classification,
            f"{label} classification precedence",
        )
    return dict(case)


def _validate_sidecars(
    rows: Any,
    *,
    rank: int,
    read_sidecar: Callable[[str], bytes],
    seen_paths: set[str],
    payloads_by_path: dict[str, bytes],
) -> list[dict[str, Any]]:
    require(isinstance(rows, list), f"rank {rank} sidecars")
    result: list[dict[str, Any]] = []
    valid_cases = {
        f"{mutant_id}:{lane}" for mutant_id in ASSIGNMENT[rank] for lane in LANES
    }
    for index, row in enumerate(rows):
        label = f"rank {rank} sidecar {index}"
        require(isinstance(row, dict), label)
        require(row.get("case_id") in valid_cases, f"{label} case")
        require(isinstance(row.get("stage"), str) and row["stage"], f"{label} stage")
        relative = row.get("relative_path")
        require(
            isinstance(relative, str)
            and relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            f"{label} safe path",
        )
        require(relative not in seen_paths, f"{label} duplicate path")
        seen_paths.add(relative)
        payload = read_sidecar(relative)
        payloads_by_path[relative] = payload
        expected_bytes = 1 * 248320 * 4
        require(row.get("bytes") == len(payload) == expected_bytes, f"{label} bytes")
        require(row.get("dtype") == "float32", f"{label} dtype")
        require(row.get("shape") == [1, 248320], f"{label} shape")
        require(row.get("sha256") == sha256_bytes(payload), f"{label} SHA")
        require(
            bool(np.isfinite(np.frombuffer(payload, dtype="<f4")).all()),
            f"{label} nonfinite FP32 values",
        )
        for field in (
            "persisted_before_next_stage",
            "atomic_tmp_replace_completed",
            "file_and_directory_fsync_completed",
        ):
            require(row.get(field) is True, f"{label} {field}")
        result.append(dict(row))
    return result


def _validate_recomputed_semantics(
    *,
    mutant_id: str,
    clean: Mapping[str, Any],
    mutant: Mapping[str, Any],
    sidecars: Sequence[Mapping[str, Any]],
    payloads_by_path: Mapping[str, bytes],
) -> None:
    """Independently derive detector cells from the serialized FP32 evidence."""

    expected_stages = EXPECTED_HORIZON_STAGES[mutant_id]

    def rows(case_id: str) -> dict[str, Mapping[str, Any]]:
        result = {
            str(row["stage"]): row for row in sidecars if row.get("case_id") == case_id
        }
        require(list(result) == list(expected_stages), f"{case_id} recompute horizon")
        return result

    clean_rows = rows(f"{mutant_id}:clean")
    mutant_rows = rows(f"{mutant_id}:target_suppressed")
    clean_argmax: list[int] = []
    mutant_argmax: list[int] = []
    exact_logit_sha = True
    max_abs = 0.0
    diff_sq = 0.0
    clean_sq = 0.0
    for stage in expected_stages:
        clean_row = clean_rows[stage]
        mutant_row = mutant_rows[stage]
        clean_vector = np.frombuffer(
            payloads_by_path[str(clean_row["relative_path"])], dtype="<f4"
        )
        mutant_vector = np.frombuffer(
            payloads_by_path[str(mutant_row["relative_path"])], dtype="<f4"
        )
        require(
            clean_vector.size == mutant_vector.size == 248320,
            f"{mutant_id}:{stage} recompute vector shape",
        )
        require(
            bool(np.isfinite(clean_vector).all())
            and bool(np.isfinite(mutant_vector).all()),
            f"{mutant_id}:{stage} nonfinite FP32 sidecar",
        )
        clean_argmax.append(int(np.argmax(clean_vector)))
        mutant_argmax.append(int(np.argmax(mutant_vector)))
        exact_logit_sha = bool(
            exact_logit_sha and clean_row["sha256"] == mutant_row["sha256"]
        )
        difference = mutant_vector.astype(np.float64) - clean_vector.astype(np.float64)
        if difference.size:
            max_abs = max(max_abs, float(np.max(np.abs(difference))))
        diff_sq += float(np.dot(difference, difference))
        clean64 = clean_vector.astype(np.float64)
        clean_sq += float(np.dot(clean64, clean64))
    argmax_equal = clean_argmax == mutant_argmax
    relative_l2 = math.sqrt(diff_sq) / max(math.sqrt(clean_sq), 1e-30)

    clean_semantics = clean["outcome"]["semantics"]
    clean_token = clean_semantics["token_only"]
    require(
        clean_token["status"] == "evaluated"
        and clean_token["caught"] is False
        and clean_token["exact_sha"] is True
        and clean_token["argmax_equal"] is True
        and clean_token["max_abs"] is None
        and clean_token["relative_l2"] is None,
        f"{mutant_id} clean token self-comparison",
    )
    clean_logits = clean_semantics["full_logit"]
    require(
        clean_logits["status"] == "evaluated"
        and clean_logits["caught"] is False
        and clean_logits["exact_sha"] is True
        and clean_logits["argmax_equal"] is True
        and float(clean_logits["max_abs"]) == 0.0
        and float(clean_logits["relative_l2"]) == 0.0,
        f"{mutant_id} clean logit self-comparison",
    )

    observed = mutant["outcome"]["semantics"]
    token = observed["token_only"]
    require(token["status"] == "evaluated", f"{mutant_id} token recompute status")
    require(token["argmax_equal"] is argmax_equal, f"{mutant_id} token argmax recompute")
    require(token["exact_sha"] is argmax_equal, f"{mutant_id} token SHA recompute")
    require(token["caught"] is (not argmax_equal), f"{mutant_id} token caught recompute")
    logits = observed["full_logit"]
    require(logits["status"] == "evaluated", f"{mutant_id} logit recompute status")
    require(logits["exact_sha"] is exact_logit_sha, f"{mutant_id} logit SHA recompute")
    require(logits["argmax_equal"] is argmax_equal, f"{mutant_id} logit argmax recompute")
    require(logits["caught"] is (not exact_logit_sha), f"{mutant_id} logit caught recompute")
    require(
        math.isclose(float(logits["max_abs"]), max_abs, rel_tol=1e-7, abs_tol=1e-7),
        f"{mutant_id} max-abs recompute",
    )
    require(
        math.isclose(
            float(logits["relative_l2"]), relative_l2, rel_tol=1e-7, abs_tol=1e-12
        ),
        f"{mutant_id} relative-L2 recompute",
    )


def aggregate_rank_payloads(
    *,
    prereg: Mapping[str, Any],
    prereg_sha: str,
    rank_payloads: Sequence[Mapping[str, Any]],
    rank_raw_receipts: Sequence[Mapping[str, Any]],
    rr2_rows: Mapping[str, Mapping[str, Any]],
    read_sidecar: Callable[[str], bytes],
) -> dict[str, Any]:
    _validate_preregistration(prereg)
    require(len(rank_payloads) == len(rank_raw_receipts) == 8, "eight rank payloads")
    require(set(rr2_rows) == set(MUTANT_IDS), "RR2 reference coverage")
    hardware_rows: list[dict[str, Any]] = []
    all_cases: list[dict[str, Any]] = []
    all_sidecars: list[dict[str, Any]] = []
    sidecar_payloads: dict[str, bytes] = {}
    seen_paths: set[str] = set()
    nonces: set[str] = set()
    for rank in range(8):
        payload = rank_payloads[rank]
        rank_receipt = rank_raw_receipts[rank]
        require(isinstance(rank_receipt, dict), f"rank {rank} raw receipt")
        require(rank_receipt.get("rank") == rank, f"rank {rank} raw receipt identity")
        require(
            rank_receipt.get("relative_path")
            == f"detector-matrix-v2-rank-{rank}.json",
            f"rank {rank} raw receipt path",
        )
        _plain_int(rank_receipt.get("bytes"), f"rank {rank} raw receipt bytes", minimum=1)
        _sha(rank_receipt.get("sha256"), f"rank {rank} raw receipt SHA")
        require(payload.get("schema_version") == RANK_SCHEMA, f"rank {rank} schema")
        require(payload.get("rank") == rank, f"rank {rank} identity")
        require(
            payload.get("runner_sha256")
            == prereg["source_binding"]["runner_sha256"],
            f"rank {rank} runner provenance",
        )
        require(
            payload.get("gate_policy_sha256")
            == prereg["source_binding"]["gate_policy_sha256"],
            f"rank {rank} gate-policy provenance",
        )
        require(payload.get("assigned_fault_ids") == list(ASSIGNMENT[rank]), f"rank {rank} assignment")
        hardware_rows.append(_validate_hardware(payload.get("hardware"), rank))
        _validate_input_receipt(
            payload.get("input_receipt"), prereg=prereg, prereg_sha=prereg_sha, rank=rank
        )
        post_warmup_baseline = _validate_warmup(
            payload.get("discarded_prebaseline_warmup_receipt"), rank
        )
        sidecars = _validate_sidecars(
            payload.get("sidecars"),
            rank=rank,
            read_sidecar=read_sidecar,
            seen_paths=seen_paths,
            payloads_by_path=sidecar_payloads,
        )
        all_sidecars.extend(sidecars)
        cases = payload.get("cases")
        require(isinstance(cases, list), f"rank {rank} cases")
        expected_case_order = [
            f"{mutant_id}:{lane}"
            for mutant_id in ASSIGNMENT[rank]
            for lane in LANES
        ]
        require(
            [case.get("case_id") if isinstance(case, dict) else None for case in cases]
            == expected_case_order,
            f"rank {rank} clean-before-mutant case order",
        )
        by_id: dict[str, Mapping[str, Any]] = {}
        for case in cases:
            require(isinstance(case, dict), f"rank {rank} case schema")
            case_id = case.get("case_id")
            require(isinstance(case_id, str) and case_id not in by_id, f"rank {rank} duplicate case")
            by_id[case_id] = case
        expected_ids = {
            f"{mutant_id}:{lane}" for mutant_id in ASSIGNMENT[rank] for lane in LANES
        }
        require(set(by_id) == expected_ids, f"rank {rank} exact clean/mutant coverage")
        for mutant_id in ASSIGNMENT[rank]:
            for lane in LANES:
                case = _validate_case(
                    by_id[f"{mutant_id}:{lane}"],
                    mutant_id=mutant_id,
                    lane=lane,
                    sidecars=sidecars,
                    sidecar_payloads=sidecar_payloads,
                    expected_post_warmup_baseline=post_warmup_baseline,
                )
                nonce = case["freshness_receipt"]["case_nonce_sha256"]
                require(nonce not in nonces, f"duplicate freshness nonce {nonce}")
                nonces.add(nonce)
                all_cases.append(case)
    uuids = [row["uuid"] for row in hardware_rows]
    require(len(set(uuids)) == 8, "eight distinct H20 UUIDs")
    require(len({row["torch_version"] for row in hardware_rows}) == 1, "rank torch version drift")
    require(len({row["torch_cuda"] for row in hardware_rows}) == 1, "rank CUDA version drift")
    case_ids = [row["case_id"] for row in all_cases]
    require(len(case_ids) == 18 and len(set(case_ids)) == 18, "exact 18 unique cases")
    clean_cases = [row for row in all_cases if row["lane"] == "clean"]
    mutant_cases = [row for row in all_cases if row["lane"] == "target_suppressed"]
    require(len(clean_cases) == len(mutant_cases) == 9, "nine clean and nine mutants")
    classifications = Counter(row["outcome"]["classification"] for row in mutant_cases)
    operational_invalid_count = classifications.get("operational_invalid", 0)
    require(operational_invalid_count == 0, "operational invalid count")

    per_fault: list[dict[str, Any]] = []
    escape_count = 0
    for mutant_id in MUTANT_IDS:
        clean = next(row for row in clean_cases if row["mutant_id"] == mutant_id)
        mutant = next(row for row in mutant_cases if row["mutant_id"] == mutant_id)
        outcome = mutant["outcome"]
        if mutant_id in TEACHER_FORCING:
            clean_teacher = clean["outcome"]["mutation_receipt"]["teacher_forcing"]
            mutant_teacher = outcome["mutation_receipt"]["teacher_forcing"]
            require(
                clean_teacher == mutant_teacher,
                f"{mutant_id} teacher-forced token differs across lanes",
            )
        if outcome["classification"] == "completed_semantics":
            _validate_recomputed_semantics(
                mutant_id=mutant_id,
                clean=clean,
                mutant=mutant,
                sidecars=all_sidecars,
                payloads_by_path=sidecar_payloads,
            )
        if outcome["measured_non_forkaudit_escape"] is True:
            escape_count += 1
        rr2 = dict(rr2_rows[mutant_id])
        require(
            rr2.get("run_id") == prereg["input_binding"]["original_rr2_run_id"],
            f"{mutant_id} RR2 run binding",
        )
        require(rr2.get("expected_gate_id") == EXPECTED_GATES[mutant_id], f"{mutant_id} RR2 expected gate")
        require(rr2.get("observed_gate_id") == EXPECTED_GATES[mutant_id], f"{mutant_id} RR2 observed gate")
        require(rr2.get("classification") == "detected_expected_gate", f"{mutant_id} RR2 detected")
        require(rr2.get("restoration_verified") is True, f"{mutant_id} RR2 restoration")
        require(
            rr2.get("matched_clean_classification") == "clean_pass",
            f"{mutant_id} RR2 clean reference",
        )
        per_fault.append(
            {
                "mutant_id": mutant_id,
                "expected_gate_id": EXPECTED_GATES[mutant_id],
                "rr2_forkaudit_reference": rr2,
                "r28_target_suppressed": {
                    "classification": outcome["classification"],
                    "other_forkaudit_gate": outcome["fork_audit"]["other_gate"],
                    "production_assertion": outcome["production"]["assertion"],
                    "production_nonassertion_crash": outcome["production"]["nonassertion_crash"],
                    "fault_payload_abort": outcome["production"]["fault_payload_abort"],
                    "token_only": outcome["semantics"]["token_only"],
                    "full_logit": outcome["semantics"]["full_logit"],
                    "measured_non_forkaudit_escape": outcome["measured_non_forkaudit_escape"],
                },
            }
        )
    if escape_count == 0:
        scientific_outcome = "negative"
    elif escape_count == 9:
        scientific_outcome = "positive"
    else:
        scientific_outcome = "mixed"
    summary = {
        "schema_version": SUMMARY_SCHEMA,
        "workstream_id": "E-R28-FULL-DETECTOR-MATRIX",
        "preregistration_sha256": prereg_sha,
        "rank_receipts": [dict(row) for row in rank_raw_receipts],
        "hardware": hardware_rows,
        "sidecar_receipts": sorted(all_sidecars, key=lambda row: (row["case_id"], row["stage"], row["relative_path"])),
        "counts": {
            "ranks": 8,
            "distinct_h20_gpu_uuids": 8,
            "cases": 18,
            "clean_cases": 9,
            "target_suppressed_mutant_cases": 9,
            "clean_fp32_sidecars": sum(row["case_id"].endswith(":clean") for row in all_sidecars),
            "mutant_fp32_sidecars": sum(row["case_id"].endswith(":target_suppressed") for row in all_sidecars),
            "measured_non_forkaudit_escapes": escape_count,
            "classifications": {name: classifications.get(name, 0) for name in CLASSIFICATIONS},
        },
        "per_fault_detector_rows": per_fault,
        "scientific_valid": True,
        "scientific_outcome": scientific_outcome,
        "operational_invalid_count": 0,
        "interpretation": {
            "scientific_outcome_is_not_validity": True,
            "valid_negative_results_retained": True,
            "not_evaluated_is_unknown": True,
            "rr2_reference_is_a_separate_execution_not_a_pooled_rate": True,
        },
    }
    require(summary["counts"]["clean_fp32_sidecars"] >= 9, "all clean cases need sidecars")
    return summary


def aggregate_from_paths(args: argparse.Namespace) -> dict[str, Any]:
    prereg_raw = _check_file(
        args.preregistration, args.expected_preregistration_sha256, "preregistration"
    )
    prereg = json.loads(prereg_raw)
    require(prereg.get("runner_sha256") == args.expected_runner_sha256, "runner binding")
    source_paths = {
        "runner_sha256": args.runner,
        "builder_sha256": Path(__file__).resolve(),
        "replay_sha256": args.replay,
        "test_sha256": args.test_file,
        "launcher_sha256": args.launcher,
        "gate_policy_sha256": args.gate_policy,
        "qs_config_sha256": args.qs_config,
        "scope_supersession_sha256": args.scope_supersession,
        "external_pin_payload_sha256": args.external_pin_payload,
    }
    source_binding = prereg.get("source_binding", {})
    for field, path in source_paths.items():
        require(path is not None and path.is_file(), f"aggregate source file {field}")
        require(
            sha256_file(path) == source_binding.get(field),
            f"aggregate current source SHA {field}",
        )
    require(
        source_binding["external_pin_payload_sha256"]
        == prereg["input_binding"]["external_pin_payload_sha256"],
        "external pin source/input binding",
    )
    recomputed_pin_receipt = validate_external_pin_payload(
        external_pin_payload=args.external_pin_payload,
        expected_external_pin_sha256=source_binding[
            "external_pin_payload_sha256"
        ],
        scope_supersession=args.scope_supersession,
        input_binding=prereg["input_binding"],
    )
    require(
        recomputed_pin_receipt
        == prereg.get("external_pin_semantic_validation"),
        "external pin semantic validation replay",
    )
    require(
        sha256_file(args.original_receipt_manifest)
        == prereg["input_binding"]["original_rr2_receipt_manifest_sha256"],
        "original RR2 manifest binding",
    )
    rr2 = original_receipts(
        original_receipt_manifest=args.original_receipt_manifest,
        original_rr2_root=args.original_rr2_root,
    )
    rank_payloads: list[Mapping[str, Any]] = []
    rank_receipts: list[dict[str, Any]] = []
    for rank in range(8):
        path = args.rank_root / f"detector-matrix-v2-rank-{rank}.json"
        require(path.is_file() and not path.is_symlink(), f"rank {rank} file integrity")
        payload = path.read_bytes()
        rank_payloads.append(json.loads(payload))
        rank_receipts.append(
            {
                "rank": rank,
                "relative_path": path.name,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
            }
        )
    root = args.rank_root.resolve()

    def read_sidecar(relative: str) -> bytes:
        candidate = args.rank_root / relative
        require(not candidate.is_symlink(), "sidecar symlink rejected")
        path = candidate.resolve()
        require(path.is_relative_to(root), "sidecar escaped rank root")
        require(path.is_file() and not path.is_symlink(), "sidecar file integrity")
        return path.read_bytes()

    result = aggregate_rank_payloads(
        prereg=prereg,
        prereg_sha=args.expected_preregistration_sha256,
        rank_payloads=rank_payloads,
        rank_raw_receipts=rank_receipts,
        rr2_rows=rr2,
        read_sidecar=read_sidecar,
    )
    write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("preregister", "aggregate"), required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--runner", type=Path)
    result.add_argument("--replay", type=Path)
    result.add_argument("--test-file", type=Path)
    result.add_argument("--launcher", type=Path)
    result.add_argument("--gate-policy", type=Path)
    result.add_argument("--qs-config", type=Path)
    result.add_argument("--scope-supersession", type=Path)
    result.add_argument("--original-receipt-manifest", type=Path, required=True)
    result.add_argument("--original-rr2-root", type=Path)
    result.add_argument("--original-rr2-run-id", default="372384bd37cf7640ca210537a4360e1a")
    result.add_argument("--model-revision")
    result.add_argument("--weight-ledger-sha256")
    result.add_argument("--artifact-ledger-sha256")
    result.add_argument("--pg19-sha256")
    result.add_argument("--pg19-manifest-sha256")
    result.add_argument("--windows-sha256")
    result.add_argument("--frozen-query-banks-sha256")
    result.add_argument("--code-ledger-sha256")
    result.add_argument("--imported-rr2-code-ledger-sha256")
    result.add_argument("--external-pin-payload", type=Path)
    result.add_argument("--external-pin-payload-sha256")
    result.add_argument("--preregistration", type=Path)
    result.add_argument("--expected-preregistration-sha256")
    result.add_argument("--rank-root", type=Path)
    result.add_argument("--expected-runner-sha256")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.stage == "preregister":
        require(
            all(
                value is not None
                for value in (
                    args.runner,
                    args.replay,
                    args.test_file,
                    args.launcher,
                    args.gate_policy,
                    args.qs_config,
                    args.scope_supersession,
                    args.external_pin_payload,
                )
            ),
            "preregister source paths",
        )
        preregister(args)
    else:
        require(
            args.original_rr2_root is not None
            and args.preregistration is not None
            and args.rank_root is not None
            and args.expected_preregistration_sha256 is not None
            and args.expected_runner_sha256 is not None,
            "aggregate paths and hashes",
        )
        aggregate_from_paths(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
