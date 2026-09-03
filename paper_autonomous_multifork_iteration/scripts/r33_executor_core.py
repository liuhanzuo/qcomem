from __future__ import annotations

"""Fault-content-independent execution contracts for the R33 fresh-fault lane.

This module deliberately contains no fault definitions and imports no model or
accelerator package.  It supplies the pieces that were missing or ambiguous in
the R29/R30 executors:

* a clean gate that must pass before any fault can start;
* hash-chained, detached-replayable receipts;
* exception classification that never converts an unexpected pass into a
  detector catch;
* exception-safe mutation restoration; and
* an aggregate contract that emits no scientific result for incomplete or
  operationally invalid inputs.

The schemas are strict by design.  A future R33 fault adapter may add opaque
payload fields inside receipts, but it may not weaken these state transitions.
"""

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar


CLEAN_SCHEMA = "forkaudit-r33-clean-gate-v1"
FAULT_SCHEMA = "forkaudit-r33-fault-case-v1"
PROTOCOL_SCHEMA = "forkaudit-r33-execution-protocol-v1"
RECEIPT_SCHEMA = "forkaudit-r33-hash-chained-receipt-v1"
SUMMARY_SCHEMA = "forkaudit-r33-strict-summary-v1"

CLEAN_RECEIPT_ORDER = (
    "input_and_source_binding",
    "ownership_transition",
    "existing_validator_battery",
    "semantic_exactness",
    "lifecycle_cleanup",
)
FAULT_RECEIPT_ORDER = (
    "input_and_source_binding",
    "injector_application",
    "existing_validator_battery",
    "semantic_horizon",
    "mutation_restoration",
    "lifecycle_cleanup",
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")
ZERO_SHA256 = "0" * 64


class R33ContractError(RuntimeError):
    """A schema, provenance, or fail-closed contract violation."""


class AuthenticatedValidatorRejection(RuntimeError):
    """A rejection from an explicitly registered pre-existing validator."""

    def __init__(self, gate_id: str, message: str, evidence: Mapping[str, Any] | None = None):
        super().__init__(message)
        require(isinstance(gate_id, str) and bool(gate_id), "validator gate id missing")
        self.gate_id = gate_id
        self.evidence = dict(evidence or {})


class UnexpectedFaultPass(RuntimeError):
    """A harness sentinel; it is never detector evidence."""


class MutationRestorationError(R33ContractError):
    """A mutation could not be restored exactly."""


T = TypeVar("T")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R33ContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def check_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, f"{label} SHA-256")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    require(set(value) == expected, f"{label} keys")


def _receipt_digest(receipt_without_digest: Mapping[str, Any]) -> str:
    return sha256_json(receipt_without_digest)


def build_receipt_chain(
    *,
    run_id: str,
    case_id: str,
    ordered_payloads: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    """Build a canonical receipt chain whose ordering is cryptographically bound."""

    require(isinstance(run_id, str) and bool(run_id), "receipt run id")
    require(isinstance(case_id, str) and bool(case_id), "receipt case id")
    output: list[dict[str, Any]] = []
    previous = ZERO_SHA256
    for index, (receipt_id, payload_value) in enumerate(ordered_payloads):
        require(isinstance(receipt_id, str) and bool(receipt_id), "receipt id")
        require(isinstance(payload_value, Mapping), f"{receipt_id} payload")
        payload = dict(payload_value)
        base = {
            "schema_version": RECEIPT_SCHEMA,
            "run_id": run_id,
            "case_id": case_id,
            "sequence_index": index,
            "receipt_id": receipt_id,
            "previous_receipt_sha256": previous,
            "payload": payload,
            "payload_sha256": sha256_json(payload),
        }
        receipt = {**base, "receipt_sha256": _receipt_digest(base)}
        output.append(receipt)
        previous = receipt["receipt_sha256"]
    return output


def validate_receipt_chain(
    value: Any,
    *,
    run_id: str,
    case_id: str,
    expected_order: Sequence[str],
) -> dict[str, Any]:
    require(isinstance(value, list), f"{case_id} receipt chain")
    require(len(value) == len(expected_order), f"{case_id} receipt count")
    previous = ZERO_SHA256
    observed_ids: list[str] = []
    for index, receipt_value in enumerate(value):
        require(isinstance(receipt_value, Mapping), f"{case_id} receipt {index}")
        receipt = dict(receipt_value)
        _exact_keys(
            receipt,
            {
                "schema_version",
                "run_id",
                "case_id",
                "sequence_index",
                "receipt_id",
                "previous_receipt_sha256",
                "payload",
                "payload_sha256",
                "receipt_sha256",
            },
            f"{case_id} receipt {index}",
        )
        require(receipt["schema_version"] == RECEIPT_SCHEMA, f"{case_id} receipt schema")
        require(receipt["run_id"] == run_id and receipt["case_id"] == case_id, f"{case_id} receipt binding")
        require(receipt["sequence_index"] == index, f"{case_id} receipt index")
        require(receipt["previous_receipt_sha256"] == previous, f"{case_id} receipt predecessor")
        payload = receipt["payload"]
        require(isinstance(payload, Mapping), f"{case_id} receipt payload")
        require(receipt["payload_sha256"] == sha256_json(payload), f"{case_id} receipt payload digest")
        base = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
        require(receipt["receipt_sha256"] == _receipt_digest(base), f"{case_id} receipt digest")
        observed_ids.append(receipt["receipt_id"])
        previous = receipt["receipt_sha256"]
    require(tuple(observed_ids) == tuple(expected_order), f"{case_id} receipt order")
    return {
        "receipt_count": len(value),
        "receipt_ids": observed_ids,
        "chain_head_sha256": previous,
    }


def validate_transition_receipt(
    value: Any,
    *,
    expected_clone_count: int,
    expected_action: str,
    label: str,
) -> dict[str, Any]:
    """Replay the ownership-only repair receipt without importing its producer."""

    require(isinstance(value, Mapping), f"{label} transition receipt")
    receipt = dict(value)
    require(
        receipt.get("schema_version") == "qcomem-single-token-gdn-conv-privatization-v1",
        f"{label} transition schema",
    )
    require(type(receipt.get("conv_tensor_count")) is int and receipt["conv_tensor_count"] > 0, f"{label} tensor count")
    require(receipt.get("cloned_tensor_count") == expected_clone_count, f"{label} clone count")
    require(
        receipt.get("already_private_tensor_count")
        == receipt["conv_tensor_count"] - expected_clone_count,
        f"{label} private count",
    )
    require(receipt.get("ownership_only_change") is True, f"{label} ownership-only flag")
    require(receipt.get("fault_id_specialization") is False, f"{label} fault specialization")
    rows = receipt.get("rows")
    require(isinstance(rows, list) and len(rows) == receipt["conv_tensor_count"], f"{label} rows")
    require(receipt.get("rows_sha256") == sha256_json(rows), f"{label} row digest")
    for row in rows:
        require(isinstance(row, Mapping), f"{label} row type")
        require(row.get("action") == expected_action, f"{label} action")
        require(row.get("base_disjoint") is True and row.get("all_peers_disjoint") is True, f"{label} disjointness")
        check_sha256(row.get("content_sha256"), f"{label} content")
    return {
        "conv_tensor_count": receipt["conv_tensor_count"],
        "cloned_tensor_count": expected_clone_count,
        "rows_sha256": receipt["rows_sha256"],
    }


def _validate_source_bindings(value: Any, label: str) -> dict[str, str]:
    require(isinstance(value, Mapping) and bool(value), f"{label} source bindings")
    output: dict[str, str] = {}
    for key, digest in sorted(value.items()):
        require(isinstance(key, str) and bool(key), f"{label} source key")
        output[key] = check_sha256(digest, f"{label} source {key}")
    return output


def _validate_lifecycle(value: Any, *, label: str) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"{label} lifecycle")
    lifecycle = dict(value)
    _exact_keys(
        lifecycle,
        {"capability_binding", "capability_binding_sha256", "disposal_receipt"},
        f"{label} lifecycle",
    )
    capability = lifecycle["capability_binding"]
    require(isinstance(capability, Mapping), f"{label} lifecycle capability")
    require(lifecycle["capability_binding_sha256"] == sha256_json(capability), f"{label} lifecycle capability digest")
    operation = capability.get("case_disposal_operation")
    receipt_schema = capability.get("case_disposal_receipt_schema")
    require(isinstance(operation, str) and bool(operation), f"{label} disposal operation")
    require(isinstance(receipt_schema, str) and bool(receipt_schema), f"{label} disposal schema")
    require(capability.get("python_reference_clear_only") is False, f"{label} reference-clear-only capability")
    receipt = lifecycle["disposal_receipt"]
    require(isinstance(receipt, Mapping), f"{label} disposal receipt")
    require(receipt.get("schema_version") == receipt_schema, f"{label} disposal receipt schema")
    require(receipt.get("operation") == operation, f"{label} disposal operation binding")
    require(receipt.get("completed") is True, f"{label} disposal incomplete")
    require(receipt.get("registered_backend_restored") is True, f"{label} backend restoration")
    require(receipt.get("strong_references_released") is True, f"{label} strong references")
    require(receipt.get("gc_collect_completed") is True, f"{label} garbage collection")
    if receipt.get("accelerator_cleanup_required") is True:
        require(receipt.get("accelerator_cache_cleanup_completed") is True, f"{label} accelerator cache cleanup")
        require(receipt.get("accelerator_synchronize_completed") is True, f"{label} accelerator synchronize")
    require(receipt.get("allocator_baseline_exact") is True, f"{label} allocator baseline")
    require(receipt.get("cleanup_error") is None, f"{label} cleanup error")
    return {
        "operation": operation,
        "receipt_schema": receipt_schema,
        "allocator_baseline_exact": True,
    }


def validate_clean_report(
    value: Any,
    *,
    expected_run_id: str | None = None,
    expected_protocol_sha256: str | None = None,
    expected_execution_input_sha256: str | None = None,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), "clean report")
    report = dict(value)
    required = {
        "schema_version",
        "run_id",
        "case_id",
        "status",
        "local_dry_run",
        "scientific_result",
        "protocol_sha256",
        "execution_input_sha256",
        "source_bindings",
        "fault_module_loaded",
        "faults_executed",
        "all_existing_gates_enabled",
        "full_horizon_reached",
        "false_positive",
        "comparisons",
        "transition_receipts",
        "storage_replay",
        "binding_replay",
        "lifecycle",
        "receipt_chain",
    }
    _exact_keys(report, required, "clean report")
    require(report["schema_version"] == CLEAN_SCHEMA, "clean schema")
    require(report["case_id"] == "clean", "clean case id")
    require(report["status"] == "clean_pass", "clean status")
    require(isinstance(report["run_id"], str) and bool(report["run_id"]), "clean run id")
    if expected_run_id is not None:
        require(report["run_id"] == expected_run_id, "clean run binding")
    protocol_sha = check_sha256(report["protocol_sha256"], "clean protocol")
    execution_sha = check_sha256(report["execution_input_sha256"], "clean execution input")
    if expected_protocol_sha256 is not None:
        require(protocol_sha == expected_protocol_sha256, "clean protocol SHA drift")
    if expected_execution_input_sha256 is not None:
        require(execution_sha == expected_execution_input_sha256, "clean execution-input SHA drift")
    sources = _validate_source_bindings(report["source_bindings"], "clean")
    require(report["fault_module_loaded"] is False and report["faults_executed"] is False, "clean fault isolation")
    require(report["all_existing_gates_enabled"] is True, "clean gates disabled")
    require(report["full_horizon_reached"] is True, "clean horizon")
    require(report["false_positive"] is False, "clean false positive")
    comparisons = report["comparisons"]
    require(isinstance(comparisons, Mapping), "clean comparisons")
    exact_fields = (
        "greedy_token_exact",
        "canonical_fp32_logits_byte_exact",
        "terminal_request_0_gdn_exact",
        "terminal_logical_kv_exact",
    )
    require(set(comparisons) == set(exact_fields), "clean comparison keys")
    require(all(comparisons[field] is True for field in exact_fields), "clean semantic mismatch")
    transitions = report["transition_receipts"]
    require(isinstance(transitions, Mapping) and set(transitions) == {"first", "repeat"}, "clean transition receipts")
    first_count = transitions["first"].get("conv_tensor_count") if isinstance(transitions["first"], Mapping) else None
    require(type(first_count) is int and first_count > 0, "clean first transition tensor count")
    first_clone_count = transitions["first"].get("cloned_tensor_count")
    require(
        first_clone_count in (0, first_count),
        "clean first transition must be uniformly borrowed or already private",
    )
    first = validate_transition_receipt(
        transitions["first"],
        expected_clone_count=first_clone_count,
        expected_action=(
            "cloned_borrowed_state"
            if first_clone_count == first_count
            else "already_private_noop"
        ),
        label="clean first",
    )
    repeat = validate_transition_receipt(
        transitions["repeat"],
        expected_clone_count=0,
        expected_action="already_private_noop",
        label="clean repeat",
    )
    require(first["conv_tensor_count"] == repeat["conv_tensor_count"], "clean transition tensor count drift")
    for replay_name in ("storage_replay", "binding_replay"):
        replay = report[replay_name]
        require(isinstance(replay, Mapping) and replay.get("passed") is True, f"clean {replay_name}")
        require(replay.get("candidate_modules_imported") is False, f"clean {replay_name} imports")
    lifecycle = _validate_lifecycle(report["lifecycle"], label="clean")
    chain = validate_receipt_chain(
        report["receipt_chain"],
        run_id=report["run_id"],
        case_id="clean",
        expected_order=CLEAN_RECEIPT_ORDER,
    )
    return {
        "run_id": report["run_id"],
        "protocol_sha256": protocol_sha,
        "execution_input_sha256": execution_sha,
        "source_bindings": sources,
        "transition": {"first": first, "repeat": repeat},
        "lifecycle": lifecycle,
        "receipt_chain": chain,
        "local_dry_run": report["local_dry_run"],
        "scientific_result": report["scientific_result"],
        "clean_gate_passed": True,
    }


def classify_fault_operation(operation: Callable[[], T]) -> dict[str, Any]:
    """Classify a fault operation without any unexpected-pass sentinel loophole."""

    try:
        result = operation()
    except AuthenticatedValidatorRejection as exc:
        return {
            "classification": "caught_by_existing_validator",
            "scientific_outcome_available": True,
            "gate_id": exc.gate_id,
            "evidence": exc.evidence,
            "message": str(exc),
            "operation_result": None,
        }
    except UnexpectedFaultPass as exc:
        return {
            "classification": "operational_invalid",
            "scientific_outcome_available": False,
            "reason": "unexpected_pass_sentinel_is_not_detector_evidence",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "operation_result": None,
        }
    except BaseException as exc:
        return {
            "classification": "operational_invalid",
            "scientific_outcome_available": False,
            "reason": "unregistered_exception",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "operation_result": None,
        }
    return {
        "classification": "escaped_to_full_horizon",
        "scientific_outcome_available": True,
        "gate_id": None,
        "evidence": None,
        "message": None,
        "operation_result": result,
    }


@dataclass
class MutationTransaction:
    """Apply once and restore exactly once, even when the body raises."""

    apply_operation: Callable[[], Mapping[str, Any]]
    restore_operation: Callable[[], Mapping[str, Any]]
    applied_receipt: dict[str, Any] | None = None
    restoration_receipt: dict[str, Any] | None = None
    _restoration_attempted: bool = False

    def __enter__(self) -> "MutationTransaction":
        receipt = self.apply_operation()
        require(isinstance(receipt, Mapping), "mutation application receipt")
        self.applied_receipt = dict(receipt)
        require(self.applied_receipt.get("mutation_observed") is True, "mutation application not observed")
        return self

    def restore_once(self) -> dict[str, Any]:
        if self._restoration_attempted:
            require(self.restoration_receipt is not None, "restoration attempted without receipt")
            return dict(self.restoration_receipt)
        self._restoration_attempted = True
        try:
            receipt = self.restore_operation()
        except BaseException as exc:
            raise MutationRestorationError(f"mutation restoration raised {type(exc).__name__}: {exc}") from exc
        require(isinstance(receipt, Mapping), "mutation restoration receipt")
        self.restoration_receipt = dict(receipt)
        require(self.restoration_receipt.get("restoration_observed") is True, "mutation restoration not observed")
        require(self.restoration_receipt.get("target_restored_exact") is True, "mutation restoration not exact")
        require(self.restoration_receipt.get("non_target_preserved_across_undo") is True, "mutation restoration collateral change")
        return dict(self.restoration_receipt)

    def __exit__(self, exc_type: Any, exc: BaseException | None, traceback: Any) -> bool:
        try:
            self.restore_once()
        except BaseException as restore_exc:
            if exc is not None:
                raise MutationRestorationError(
                    f"mutation body raised {type(exc).__name__}; restoration also failed: {restore_exc}"
                ) from restore_exc
            raise
        return False


class CleanGate:
    """A one-way gate that prohibits fault starts until clean replay passes."""

    def __init__(self, *, run_id: str, protocol_sha256: str, execution_input_sha256: str):
        self.run_id = run_id
        self.protocol_sha256 = check_sha256(protocol_sha256, "gate protocol")
        self.execution_input_sha256 = check_sha256(execution_input_sha256, "gate execution input")
        self._clean_receipt: dict[str, Any] | None = None
        self._started_faults: set[str] = set()

    @property
    def unlocked(self) -> bool:
        return self._clean_receipt is not None

    def accept_clean(self, report: Mapping[str, Any]) -> dict[str, Any]:
        require(not self._started_faults, "clean gate cannot be accepted after a fault start")
        receipt = validate_clean_report(
            report,
            expected_run_id=self.run_id,
            expected_protocol_sha256=self.protocol_sha256,
            expected_execution_input_sha256=self.execution_input_sha256,
        )
        self._clean_receipt = receipt
        return dict(receipt)

    def begin_fault(self, fault_id: str) -> dict[str, Any]:
        require(self._clean_receipt is not None, "fault execution blocked: clean gate not passed")
        require(isinstance(fault_id, str) and bool(fault_id), "fault id")
        require(fault_id not in self._started_faults, "fault case already started")
        self._started_faults.add(fault_id)
        return {
            "run_id": self.run_id,
            "fault_id": fault_id,
            "clean_receipt_chain_head_sha256": self._clean_receipt["receipt_chain"]["chain_head_sha256"],
            "fault_start_authorized": True,
        }


def validate_fault_report(
    value: Any,
    *,
    expected_run_id: str,
    expected_fault_id: str,
    expected_protocol_sha256: str,
    expected_execution_input_sha256: str,
    expected_clean_chain_head_sha256: str,
    expected_fault_definition_sha256: str | None = None,
    expected_primary_gate: str | None = None,
) -> dict[str, Any]:
    require(isinstance(value, Mapping), f"fault {expected_fault_id} report")
    report = dict(value)
    required = {
        "schema_version",
        "run_id",
        "case_id",
        "fault_id",
        "status",
        "classification",
        "scientific_outcome_available",
        "protocol_sha256",
        "execution_input_sha256",
        "clean_receipt_chain_head_sha256",
        "source_bindings",
        "injector_witness",
        "authenticated_validator_rejection",
        "full_horizon_reached",
        "semantic_comparisons",
        "mutation_requires_restoration",
        "restoration_receipt",
        "operational_invalid",
        "lifecycle",
        "receipt_chain",
    }
    _exact_keys(report, required, f"fault {expected_fault_id} report")
    require(report["schema_version"] == FAULT_SCHEMA, f"fault {expected_fault_id} schema")
    require(report["run_id"] == expected_run_id, f"fault {expected_fault_id} run binding")
    require(report["case_id"] == expected_fault_id and report["fault_id"] == expected_fault_id, f"fault {expected_fault_id} id binding")
    require(report["status"] == "completed_scientific_case", f"fault {expected_fault_id} status")
    require(report["scientific_outcome_available"] is True, f"fault {expected_fault_id} scientific outcome")
    require(report["operational_invalid"] is None, f"fault {expected_fault_id} operational invalid")
    require(report["protocol_sha256"] == expected_protocol_sha256, f"fault {expected_fault_id} protocol SHA")
    require(report["execution_input_sha256"] == expected_execution_input_sha256, f"fault {expected_fault_id} execution-input SHA")
    require(
        report["clean_receipt_chain_head_sha256"] == expected_clean_chain_head_sha256,
        f"fault {expected_fault_id} clean-gate binding",
    )
    _validate_source_bindings(report["source_bindings"], f"fault {expected_fault_id}")
    witness = report["injector_witness"]
    require(isinstance(witness, Mapping) and witness.get("fault_id") == expected_fault_id, f"fault {expected_fault_id} injector binding")
    require(witness.get("mutation_observed") is True, f"fault {expected_fault_id} mutation witness")
    definition_sha = check_sha256(witness.get("fault_definition_sha256"), f"fault {expected_fault_id} definition")
    if expected_fault_definition_sha256 is not None:
        require(
            definition_sha == expected_fault_definition_sha256,
            f"fault {expected_fault_id} frozen definition SHA",
        )
    classification = report["classification"]
    require(classification in {"caught_by_existing_validator", "escaped_to_full_horizon"}, f"fault {expected_fault_id} classification")
    rejection = report["authenticated_validator_rejection"]
    if classification == "caught_by_existing_validator":
        require(isinstance(rejection, Mapping), f"fault {expected_fault_id} rejection")
        require(rejection.get("authenticated") is True, f"fault {expected_fault_id} rejection authentication")
        require(isinstance(rejection.get("gate_id"), str) and bool(rejection["gate_id"]), f"fault {expected_fault_id} rejection gate")
        if expected_primary_gate is not None:
            require(
                rejection["gate_id"] == expected_primary_gate,
                f"fault {expected_fault_id} unexpected first gate",
            )
        require(type(report["full_horizon_reached"]) is bool, f"fault {expected_fault_id} caught horizon")
        if report["full_horizon_reached"]:
            semantic = report["semantic_comparisons"]
            require(isinstance(semantic, Mapping), f"fault {expected_fault_id} post-horizon semantic comparison")
            require(semantic.get("full_fp32_logits_evaluated") is True, f"fault {expected_fault_id} post-horizon full logits")
            require(type(semantic.get("greedy_token_equal_to_clean")) is bool, f"fault {expected_fault_id} post-horizon token comparison")
            require(type(semantic.get("full_fp32_logits_byte_exact_to_clean")) is bool, f"fault {expected_fault_id} post-horizon logit comparison")
        else:
            require(report["semantic_comparisons"] is None, f"fault {expected_fault_id} pre-horizon semantic output")
    else:
        require(rejection is None, f"fault {expected_fault_id} escaped rejection")
        require(report["full_horizon_reached"] is True, f"fault {expected_fault_id} escaped horizon")
        semantic = report["semantic_comparisons"]
        require(isinstance(semantic, Mapping), f"fault {expected_fault_id} semantic comparison")
        require(semantic.get("full_fp32_logits_evaluated") is True, f"fault {expected_fault_id} full logits")
        require(type(semantic.get("greedy_token_equal_to_clean")) is bool, f"fault {expected_fault_id} token comparison")
        require(type(semantic.get("full_fp32_logits_byte_exact_to_clean")) is bool, f"fault {expected_fault_id} logit comparison")
    if report["mutation_requires_restoration"] is True:
        restoration = report["restoration_receipt"]
        require(isinstance(restoration, Mapping), f"fault {expected_fault_id} restoration")
        require(restoration.get("restoration_observed") is True, f"fault {expected_fault_id} restoration observed")
        require(restoration.get("target_restored_exact") is True, f"fault {expected_fault_id} restoration exact")
        require(restoration.get("non_target_preserved_across_undo") is True, f"fault {expected_fault_id} restoration scope")
    else:
        require(report["restoration_receipt"] is not None, f"fault {expected_fault_id} disposal receipt missing")
    lifecycle = _validate_lifecycle(report["lifecycle"], label=f"fault {expected_fault_id}")
    chain = validate_receipt_chain(
        report["receipt_chain"],
        run_id=expected_run_id,
        case_id=expected_fault_id,
        expected_order=FAULT_RECEIPT_ORDER,
    )
    return {
        "fault_id": expected_fault_id,
        "classification": classification,
        "lifecycle": lifecycle,
        "receipt_chain": chain,
        "scientific_outcome_available": True,
    }


def validate_protocol(value: Any, *, expected_raw_sha256: str | None = None) -> dict[str, Any]:
    require(isinstance(value, Mapping), "R33 protocol")
    protocol = dict(value)
    required = {
        "schema_version",
        "run_id",
        "mode",
        "candidate_output_seen_when_frozen",
        "fault_ids",
        "execution_input_sha256",
        "source_bindings",
        "author_freeze_manifest_sha256",
        "fault_bindings",
        "lifecycle_capability_binding",
        "claim_boundary",
    }
    _exact_keys(protocol, required, "R33 protocol")
    require(protocol["schema_version"] == PROTOCOL_SCHEMA, "R33 protocol schema")
    require(isinstance(protocol["run_id"], str) and bool(protocol["run_id"]), "R33 protocol run id")
    require(protocol["mode"] in {"local_clean_only_dry_run", "formal_fresh_faults"}, "R33 protocol mode")
    require(protocol["candidate_output_seen_when_frozen"] is False, "R33 protocol outcome blind")
    fault_ids = protocol["fault_ids"]
    require(isinstance(fault_ids, list) and len(fault_ids) == len(set(fault_ids)), "R33 protocol fault ids")
    require(all(isinstance(item, str) and bool(item) for item in fault_ids), "R33 protocol fault id type")
    if protocol["mode"] == "local_clean_only_dry_run":
        require(fault_ids == [], "R33 local dry run must not bind faults")
    else:
        require(bool(fault_ids), "R33 formal protocol has no faults")
    check_sha256(protocol["execution_input_sha256"], "R33 protocol execution input")
    _validate_source_bindings(protocol["source_bindings"], "R33 protocol")
    author_sha = check_sha256(protocol["author_freeze_manifest_sha256"], "R33 author freeze")
    if protocol["mode"] == "formal_fresh_faults":
        require(author_sha != ZERO_SHA256, "R33 formal author freeze not bound")
    fault_bindings = protocol["fault_bindings"]
    require(isinstance(fault_bindings, Mapping), "R33 protocol fault bindings")
    require(set(fault_bindings) == set(fault_ids), "R33 protocol fault-binding coverage")
    for fault_id, binding in fault_bindings.items():
        require(isinstance(binding, Mapping), f"R33 protocol fault binding {fault_id}")
        require(binding.get("fault_id") == fault_id, f"R33 protocol fault self-binding {fault_id}")
        require(type(binding.get("rank")) is int and binding["rank"] >= 0, f"R33 protocol fault rank {fault_id}")
        require(isinstance(binding.get("expected_primary_gate"), str) and bool(binding["expected_primary_gate"]), f"R33 protocol fault gate {fault_id}")
        check_sha256(binding.get("fault_definition_sha256"), f"R33 protocol fault definition {fault_id}")
    capability = protocol["lifecycle_capability_binding"]
    require(isinstance(capability, Mapping), "R33 protocol lifecycle capability")
    require(isinstance(capability.get("case_disposal_operation"), str), "R33 protocol disposal operation")
    require(isinstance(capability.get("case_disposal_receipt_schema"), str), "R33 protocol disposal schema")
    require(capability.get("python_reference_clear_only") is False, "R33 protocol weak lifecycle capability")
    claim = protocol["claim_boundary"]
    require(isinstance(claim, Mapping), "R33 claim boundary")
    require(claim.get("local_dry_run_is_scientific_evidence") is False, "R33 dry-run claim boundary")
    if expected_raw_sha256 is not None:
        require(sha256_json(protocol) == expected_raw_sha256, "R33 protocol canonical SHA drift")
    return protocol


def aggregate_reports(
    *,
    protocol: Mapping[str, Any],
    clean_report: Mapping[str, Any],
    fault_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Strict aggregation.  Any missing/invalid case raises before output exists."""

    protocol_value = validate_protocol(protocol)
    protocol_sha = sha256_json(protocol_value)
    clean = validate_clean_report(
        clean_report,
        expected_run_id=protocol_value["run_id"],
        expected_protocol_sha256=protocol_sha,
        expected_execution_input_sha256=protocol_value["execution_input_sha256"],
    )
    require(clean["source_bindings"] == dict(sorted(protocol_value["source_bindings"].items())), "clean/protocol source binding")
    expected_faults = protocol_value["fault_ids"]
    require(len(fault_reports) == len(expected_faults), "R33 exact fault artifact count")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in fault_reports:
        require(isinstance(row, Mapping) and isinstance(row.get("fault_id"), str), "R33 fault artifact id")
        require(row["fault_id"] not in by_id, "R33 duplicate fault artifact")
        by_id[row["fault_id"]] = row
    require(set(by_id) == set(expected_faults), "R33 fault coverage")
    fault_rows = [
        validate_fault_report(
            by_id[fault_id],
            expected_run_id=protocol_value["run_id"],
            expected_fault_id=fault_id,
            expected_protocol_sha256=protocol_sha,
            expected_execution_input_sha256=protocol_value["execution_input_sha256"],
            expected_clean_chain_head_sha256=clean["receipt_chain"]["chain_head_sha256"],
            expected_fault_definition_sha256=protocol_value["fault_bindings"][fault_id][
                "fault_definition_sha256"
            ],
            expected_primary_gate=protocol_value["fault_bindings"][fault_id][
                "expected_primary_gate"
            ],
        )
        for fault_id in expected_faults
    ]
    local_dry = protocol_value["mode"] == "local_clean_only_dry_run"
    if local_dry:
        require(clean["local_dry_run"] is True and clean["scientific_result"] is False, "R33 local clean flags")
        status = "clean_gate_validated_faults_not_loaded_or_executed"
        scientific_valid = False
    else:
        require(clean["local_dry_run"] is False and clean["scientific_result"] is True, "R33 formal clean flags")
        require(bool(fault_rows), "R33 formal fault rows absent")
        status = "completed_strict_scientific_aggregation"
        scientific_valid = True
    return {
        "schema_version": SUMMARY_SCHEMA,
        "run_id": protocol_value["run_id"],
        "status": status,
        "scientific_valid": scientific_valid,
        "clean_gate_passed": True,
        "clean_receipt_chain_head_sha256": clean["receipt_chain"]["chain_head_sha256"],
        "fault_ids": list(expected_faults),
        "fault_rows": fault_rows,
        "operational_invalid_count": 0,
        "missing_fault_artifact_count": 0,
        "negative_or_escaped_faults_retained": True,
        "local_dry_run": local_dry,
        "fault_module_loaded": False if local_dry else None,
        "faults_executed": False if local_dry else True,
    }


__all__ = [
    "AuthenticatedValidatorRejection",
    "CLEAN_RECEIPT_ORDER",
    "CLEAN_SCHEMA",
    "CleanGate",
    "FAULT_RECEIPT_ORDER",
    "FAULT_SCHEMA",
    "MutationRestorationError",
    "MutationTransaction",
    "PROTOCOL_SCHEMA",
    "R33ContractError",
    "SUMMARY_SCHEMA",
    "UnexpectedFaultPass",
    "ZERO_SHA256",
    "aggregate_reports",
    "build_receipt_chain",
    "canonical_bytes",
    "check_sha256",
    "classify_fault_operation",
    "sha256_bytes",
    "sha256_json",
    "validate_clean_report",
    "validate_fault_report",
    "validate_protocol",
    "validate_receipt_chain",
    "validate_transition_receipt",
]
