from __future__ import annotations

"""Round-29 paired live ForkAudit request-step overhead experiment.

The baseline disables the optional append/call capture, ownership witnesses,
and artifact persistence while retaining the mandatory frozen Q16 functional
adapter.  The instrumented arm enables the existing live hooks and persists a
complete receipt.  Document prefill and model load are deliberately outside
the timed request-step boundary.
"""

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from qcomem_forkaudit_storage_witness import (
    POLICY_SHARED_BASE,
    capture_persistent_gdn_guard,
    capture_request_gdn_binding_guard,
    verify_persistent_gdn_guard,
    verify_request_gdn_binding_guard,
)
from qcomem_joint_policy import (
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
    sha256_file,
)
from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
from qcomem_qwen35_vllm_paged_integration import (
    convert_all_qwen35_full_layers_to_vllm_q16,
    register_qwen35_vllm_q16_backend,
)
from qcomem_vllm_paged_fair_control import SHARED_REUSE
from qcomem_vllm_paged_kernel import (
    AUDITED_PACKAGES,
    KERNEL_MODE,
    _resolve_vllm_unified_attention,
    audit_frozen_kernel_environment,
)
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    MultiForkHitLedger,
    ResidentRequestGroup,
    _capture_kv_binding_guard,
    build_pg19_train_query_bank,
    build_resident_request_group,
    validate_runtime_kv_ownership,
)
from run_downstream import atomic_json
from run_qcomem_qwen35_vllm_paged_multifork_resident import (
    _audit_model_config_geometry,
    _build_document_cache,
    _last_logits,
    _linear_state_digest,
    _resolve_backbone,
    _source_document_digests,
    _unregister_backend,
)


SCHEMA = "qcomem-forkaudit-r29-live-overhead-result-v1"
DESIGN_SCHEMA = "qcomem-forkaudit-r29-live-overhead-preregistration-v1"
PROTOCOL = "qcomem-forkaudit-paired-live-request-overhead-v1"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
EXPECTED_DATA_SHA256 = (
    "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
)
EXPECTED_DATA_MANIFEST_SHA256 = (
    "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"
)
EXPECTED_UPSTREAM_LEDGER_SHA256 = (
    "7620f05821fc5435a9aaa260ae82577988a5a20eff0f42901b66e6c6871fd2b9"
)
EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256 = (
    "c0a23e9d3f9d220257af97b78fd97661f315f0c82a3a010b57a771e3eeefbbfb"
)
EXPECTED_MODEL_WEIGHT_LEDGER_SHA256 = (
    "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014"
)
EXPECTED_WINDOWS_SHA256 = (
    "67e7bc9ff621732e6d65e44c8a4b2fafd7aa3ef2d2f24669704e843c3d153e51"
)
EXPECTED_DOCUMENT_SHA256 = (
    "09ab51882be802887c481d0f54583bd734baccbdd5c9022452772fa3cc49ecc4"
)
EXPECTED_QUERY_SHA256 = (
    "386c16255f673f7286c78d26f3e0eef15a6f073a150876e67344488dfa188d9d"
)
FULL_LAYERS = tuple(range(3, 40, 4))
LINEAR_LAYERS = tuple(index for index in range(40) if index not in FULL_LAYERS)
DOCUMENT_TOKENS = 4033
QUERY_TOKENS = 16
PAGE_SIZE = 128
RESIDENT_COUNT = 2
WINDOW_STRIDE = 263
CANDIDATE_WINDOWS = 8
WINDOW_BOOKS = 8
SEED = 20260821
MEASURED_SCHEDULE = (
    (0, ("baseline", "instrumented"), 0, 1),
    (1, ("instrumented", "baseline"), 1, 0),
    (2, ("baseline", "instrumented"), 1, 0),
    (3, ("instrumented", "baseline"), 0, 1),
    (4, ("baseline", "instrumented"), 0, 1),
)
WARMUP_ORDER = ("instrumented", "baseline")
EXPECTED_APPEND_EVENTS = len(FULL_LAYERS)
EXPECTED_CALL_EVENTS = len(FULL_LAYERS)
EXPECTED_TENSOR_RECORDS = len(FULL_LAYERS) * 5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_tensor(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return sha256_bytes(raw)


def strict_json(value: Any, *, label: str) -> Any:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not strict-JSON serializable: {error}") from error
    return value


def load_bound_json(path: Path, expected_sha256: str, label: str) -> Any:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"{label} raw SHA mismatch")
    return json.loads(raw)


def validate_design(design: Mapping[str, Any]) -> None:
    require(design.get("schema_version") == DESIGN_SCHEMA, "design schema drift")
    require(design.get("created_before_gpu_execution") is True, "design timing drift")
    require(design.get("gpu_execution_had_started") is False, "design execution flag drift")
    require(
        design.get("model", {}).get("revision") == MODEL_REVISION,
        "model revision drift",
    )
    geometry = design.get("geometry", {})
    expected_geometry = {
        "document_tokens": DOCUMENT_TOKENS,
        "query_tokens": QUERY_TOKENS,
        "page_size": PAGE_SIZE,
        "resident_requests_per_pair": RESIDENT_COUNT,
        "generated_tokens_per_arm": 1,
        "window_stride": WINDOW_STRIDE,
        "candidate_windows_per_book": CANDIDATE_WINDOWS,
        "window_books": WINDOW_BOOKS,
        "seed": SEED,
    }
    require(geometry == expected_geometry, "design geometry drift")
    timing = design.get("timing_population", {})
    require(
        timing.get("discarded_warmup_pairs") == 1
        and timing.get("measured_pairs") == len(MEASURED_SCHEDULE)
        and tuple(timing.get("warmup_order", ())) == WARMUP_ORDER,
        "warmup/measurement population drift",
    )
    observed_schedule = tuple(
        (
            row.get("pair_index"),
            tuple(row.get("execution_order", ())),
            row.get("baseline_slot"),
            row.get("instrumented_slot"),
        )
        for row in timing.get("measured_pair_schedule", ())
    )
    require(observed_schedule == MEASURED_SCHEDULE, "measured schedule drift")
    arms = design.get("arms", {})
    require(
        arms.get("baseline", {}).get("label") == "capture-disabled"
        and arms.get("instrumented", {}).get("expected_binary_tensor_records")
        == EXPECTED_TENSOR_RECORDS,
        "arm definition drift",
    )
    require(
        design.get("timing_and_memory_protocol", {}).get(
            "cuda_synchronize_before_each_arm"
        )
        is True
        and design.get("timing_and_memory_protocol", {}).get(
            "cuda_synchronize_before_timer_stop"
        )
        is True,
        "CUDA synchronization contract drift",
    )


def verify_bound_files(args: argparse.Namespace) -> None:
    require(sha256_file(args.pg19_data) == EXPECTED_DATA_SHA256, "PG19 SHA drift")
    require(
        sha256_file(args.pg19_manifest) == EXPECTED_DATA_MANIFEST_SHA256,
        "PG19 manifest SHA drift",
    )
    require(
        sha256_file(args.upstream_code_ledger) == EXPECTED_UPSTREAM_LEDGER_SHA256,
        "upstream code-ledger SHA drift",
    )
    require(
        sha256_file(args.model_artifact_ledger)
        == EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256,
        "model artifact-ledger SHA drift",
    )
    require(
        sha256_file(args.model_weight_ledger) == EXPECTED_MODEL_WEIGHT_LEDGER_SHA256,
        "model weight-ledger SHA drift",
    )


def frozen_input_material(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    records, data_audit = audit_pg19_train_calibration(
        args.pg19_data,
        args.pg19_manifest,
        expected_data_sha256=EXPECTED_DATA_SHA256,
        expected_manifest_sha256=EXPECTED_DATA_MANIFEST_SHA256,
        minimum_books=WINDOW_BOOKS,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    windows, windows_sha256 = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=WINDOW_BOOKS,
        document_tokens=DOCUMENT_TOKENS,
        query_tokens=QUERY_TOKENS,
        stride=WINDOW_STRIDE,
        candidate_windows_per_book=CANDIDATE_WINDOWS,
        seed=SEED,
    )
    require(windows_sha256 == EXPECTED_WINDOWS_SHA256, "windows SHA drift")
    window = windows[0]
    require(
        window.source_object == "train/10047.txt" and str(window.source_id) == "10047",
        "frozen rank-0 source drift",
    )
    queries, query_audit = build_pg19_train_query_bank(
        records,
        tokenizer,
        window,
        document_tokens=DOCUMENT_TOKENS,
        query_tokens=QUERY_TOKENS,
        count=1,
        query_stride=64,
    )
    document = window.document_ids[:DOCUMENT_TOKENS].unsqueeze(0)
    query = queries[0]
    require(sha256_tensor(document) == EXPECTED_DOCUMENT_SHA256, "document SHA drift")
    require(sha256_tensor(query) == EXPECTED_QUERY_SHA256, "query SHA drift")
    require(
        query_audit.get("source_role")
        == "same-pg19-train-book-raw-nonoverlapping-query-chunks"
        and query_audit.get("synthetic_markers_used") is False
        and query_audit.get("rows", [])[0].get("source_token_offset") == 4575,
        "query provenance drift",
    )
    return document, query, {
        "data_audit": data_audit,
        "windows_sha256": windows_sha256,
        "source_object": window.source_object,
        "source_id": str(window.source_id),
        "document_token_ids_sha256": sha256_tensor(document),
        "query_token_ids_sha256": sha256_tensor(query),
        "query_audit": query_audit,
        "same_query_used_for_both_arms": True,
    }


def pointer_free(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): pointer_free(item)
            for key, item in value.items()
            if str(key) != "callable_id"
        }
    if isinstance(value, (list, tuple)):
        return [pointer_free(item) for item in value]
    return value


@dataclass
class LiveCaptureWriter:
    root: Path
    sample_id: str
    binary_path: Path = field(init=False)
    receipt_path: Path = field(init=False)
    records: list[dict[str, Any]] = field(default_factory=list)
    append_events: list[dict[str, Any]] = field(default_factory=list)
    call_events: list[dict[str, Any]] = field(default_factory=list)
    _append_ids: dict[int, str] = field(default_factory=dict)
    _offset: int = 0
    _handle: Any = field(init=False, repr=False)
    _finalized: bool = False

    def __post_init__(self) -> None:
        require(not self.root.exists(), f"audit artifact path already exists: {self.root}")
        self.root.mkdir(parents=True)
        self.binary_path = self.root / "capture.bin"
        self.receipt_path = self.root / "receipt.json"
        self._handle = self.binary_path.open("xb")

    def _write_tensor(
        self,
        *,
        event_type: str,
        layer_index: int,
        tensor_name: str,
        value: torch.Tensor,
    ) -> dict[str, Any]:
        require(isinstance(value, torch.Tensor), "capture tensor missing")
        cpu = value.detach().contiguous().cpu()
        raw = cpu.view(torch.uint8).numpy().tobytes()
        self._handle.write(raw)
        row = {
            "record_index": len(self.records),
            "event_type": event_type,
            "layer_index": int(layer_index),
            "tensor_name": tensor_name,
            "dtype": str(cpu.dtype),
            "shape": [int(item) for item in cpu.shape],
            "offset_bytes": self._offset,
            "nbytes": len(raw),
            "content_sha256": sha256_bytes(raw),
        }
        self.records.append(row)
        self._offset += len(raw)
        return row

    def append_observer(self, layer_index: int, event: Mapping[str, Any]) -> str:
        require(layer_index in FULL_LAYERS, "append layer outside frozen plan")
        require(layer_index not in self._append_ids, "duplicate layer append capture")
        require(event.get("append_event_index") == 0, "append event index drift")
        capture_id = f"{self.sample_id}-layer-{layer_index}-append-0"
        self._write_tensor(
            event_type="append",
            layer_index=layer_index,
            tensor_name="key_states",
            value=event["key_states"],
        )
        self._write_tensor(
            event_type="append",
            layer_index=layer_index,
            tensor_name="value_states",
            value=event["value_states"],
        )
        metadata = {
            "layer_index": int(layer_index),
            "capture_id": capture_id,
            "append_event_index": int(event["append_event_index"]),
            "appended_tokens_before": int(event["appended_tokens_before"]),
            "appended_tokens_after": int(event["appended_tokens_after"]),
            "sequence_length_before": int(event["sequence_length_before"]),
            "sequence_length_after": int(event["sequence_length_after"]),
            "source_device": str(event["source_device"]),
            "source_dtype": str(event["source_dtype"]),
            "source_shape": [int(item) for item in event["source_shape"]],
        }
        self.append_events.append(metadata)
        self._append_ids[layer_index] = capture_id
        return capture_id

    def call_observer(self, event: Mapping[str, Any]) -> None:
        require(
            event.get("observer_schema") == "qcomem-forkaudit-call-observer-v2",
            "call observer schema drift",
        )
        layer_index = int(event["layer_idx"])
        require(layer_index in FULL_LAYERS, "call layer outside frozen plan")
        require(
            event.get("append_capture_id") == self._append_ids.get(layer_index),
            "call did not consume its layer append capture",
        )
        require(
            not any(row["layer_index"] == layer_index for row in self.call_events),
            "duplicate layer call capture",
        )
        position = event.get("position_ids_cpu")
        require(isinstance(position, torch.Tensor), "position IDs were not captured")
        self._write_tensor(
            event_type="call",
            layer_index=layer_index,
            tensor_name="query",
            value=event["query_cpu"],
        )
        self._write_tensor(
            event_type="call",
            layer_index=layer_index,
            tensor_name="candidate_output",
            value=event["candidate_output_cpu"],
        )
        self._write_tensor(
            event_type="call",
            layer_index=layer_index,
            tensor_name="position_ids",
            value=position,
        )
        metadata = pointer_free(
            {
                "layer_index": layer_index,
                "request_index": int(event["request_index"]),
                "resident_count": int(event["resident_count"]),
                "request_policy": str(event["request_policy"]),
                "attention_mask_is_none": bool(event["attention_mask_is_none"]),
                "append_capture_id": str(event["append_capture_id"]),
                "append_audit": dict(event["append_audit"]),
                "position_audit": dict(event["position_audit"]),
                "kernel_audit": dict(event["kernel_audit"]),
                "effective_scaling": float(event["effective_scaling"]),
            }
        )
        strict_json(metadata, label="call capture metadata")
        self.call_events.append(metadata)

    def close_partial(self) -> None:
        if not self._handle.closed:
            self._handle.flush()
            self._handle.close()

    def finalize(self, receipt: Mapping[str, Any]) -> dict[str, Any]:
        require(not self._finalized, "capture writer finalized twice")
        self.close_partial()
        require(len(self.append_events) == EXPECTED_APPEND_EVENTS, "append count drift")
        require(len(self.call_events) == EXPECTED_CALL_EVENTS, "call count drift")
        require(len(self.records) == EXPECTED_TENSOR_RECORDS, "tensor record count drift")
        require(
            tuple(row["layer_index"] for row in self.append_events) == FULL_LAYERS
            and tuple(row["layer_index"] for row in self.call_events) == FULL_LAYERS,
            "capture layer order drift",
        )
        binary_bytes = self.binary_path.stat().st_size
        require(binary_bytes == self._offset and binary_bytes > 0, "capture byte coverage drift")
        binary_manifest = {
            "path": self.binary_path.name,
            "bytes": binary_bytes,
            "sha256": sha256_path(self.binary_path),
            "record_count": len(self.records),
            "terminal_exact_byte_coverage": (
                self.records[0]["offset_bytes"] == 0
                and self.records[-1]["offset_bytes"]
                + self.records[-1]["nbytes"]
                == binary_bytes
            ),
            "records": self.records,
        }
        payload = {
            "schema_version": "qcomem-forkaudit-r29-live-capture-receipt-v1",
            "sample_id": self.sample_id,
            "append_events": self.append_events,
            "call_events": self.call_events,
            "capture_binary": binary_manifest,
            "live_receipts": pointer_free(dict(receipt)),
        }
        strict_json(payload, label="live capture receipt")
        atomic_json(self.receipt_path, payload)
        receipt_manifest = {
            "path": self.receipt_path.name,
            "bytes": self.receipt_path.stat().st_size,
            "sha256": sha256_path(self.receipt_path),
        }
        self._finalized = True
        return {
            "schema_version": "qcomem-forkaudit-r29-live-artifact-manifest-v1",
            "relative_root": self.root.name,
            "capture_binary": binary_manifest,
            "receipt": receipt_manifest,
            "artifact_bytes": binary_manifest["bytes"] + receipt_manifest["bytes"],
            "append_event_count": len(self.append_events),
            "call_event_count": len(self.call_events),
            "tensor_record_count": len(self.records),
        }


def compact_linear_digest(owner: Any, layer_indices: Sequence[int]) -> dict[str, Any]:
    value = dict(_linear_state_digest(owner, layer_indices))
    value.pop("storage_keys", None)
    return value


def active_single_request_group(
    request: Any,
    plan: Any,
) -> ResidentRequestGroup:
    requests = (request,)
    return ResidentRequestGroup(
        policy=SHARED_REUSE,
        resident_count=1,
        requests=requests,
        audit={"overhead_single_request_receipt": True},
        kv_binding_guard=_capture_kv_binding_guard(
            requests,
            plan,
            resident_count=1,
            policy=SHARED_REUSE,
        ),
    )


def allocator_before_arm() -> tuple[int, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved())


def finalize_common_output(
    sample_id: str,
    logits: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor]:
    require(logits.ndim == 2 and int(logits.shape[0]) == 1, "logit shape drift")
    require(bool(torch.isfinite(logits).all().item()), "non-finite logits")
    cpu = logits.detach().contiguous().cpu().float().clone()
    return {
        "sample_id": sample_id,
        "shape": [int(item) for item in cpu.shape],
        "dtype": "float32",
        "full_vocab_logit_sha256": sha256_tensor(cpu),
        "token_id": int(cpu.argmax(dim=-1).item()),
        "finite": True,
        "copied_after_timed_region": True,
    }, cpu


def timing_receipt(
    *,
    arm: str,
    wall_ns: int,
    allocated_before: int,
    reserved_before: int,
    artifact_bytes: int,
) -> dict[str, Any]:
    peak = int(torch.cuda.max_memory_allocated())
    allocated_after = int(torch.cuda.memory_allocated())
    reserved_after = int(torch.cuda.memory_reserved())
    incremental = peak - allocated_before
    require(
        wall_ns > 0
        and allocated_before >= 0
        and peak >= allocated_before
        and incremental >= 0
        and artifact_bytes >= 0,
        "timing/allocator receipt is invalid",
    )
    return {
        "arm": arm,
        "wall_time_ns": int(wall_ns),
        "wall_time_ms": float(wall_ns / 1_000_000.0),
        "allocated_before_bytes": allocated_before,
        "reserved_before_bytes": reserved_before,
        "peak_allocated_bytes": peak,
        "incremental_peak_allocated_bytes": incremental,
        "allocated_after_bytes": allocated_after,
        "reserved_after_bytes": reserved_after,
        "audit_artifact_bytes": int(artifact_bytes),
        "cuda_synchronized_before_start": True,
        "cuda_synchronized_before_stop": True,
        "peak_stats_reset_before_start": True,
    }


def build_arm_ledger(
    plan: Any,
    request: Any,
    kernel: Any,
    *,
    call_observer: Any | None,
) -> MultiForkHitLedger:
    """Build the one common live adapter; only the optional observer differs."""

    require(callable(kernel), "frozen unified-attention kernel is not callable")
    return MultiForkHitLedger(
        plan,
        request,
        request_index=0,
        resident_count=1,
        request_policy=SHARED_REUSE,
        expected_calls_per_layer=1,
        initial_query_tokens=QUERY_TOKENS,
        kernel=kernel,
        strict_position_values=False,
        call_observer=call_observer,
    )


def compact_ledger_receipt(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "implementation": "MultiForkHitLedger",
        "explicit_frozen_kernel": True,
        "verified": bool(receipt["verified"]),
        "protocol": str(receipt["protocol"]),
        "total_calls": int(receipt["total_calls"]),
        "call_observer_enabled": bool(receipt["call_observer_enabled"]),
        "dense_fallback_calls": int(receipt["dense_fallback_calls"]),
        "full_kv_concatenations": int(receipt["full_kv_concatenations"]),
        "mask_contract": str(receipt["mask_contract"]),
        "kernel_identity": pointer_free(dict(receipt["kernel_identity"])),
    }


def run_baseline_arm(
    model: Any,
    backbone: Any,
    plan: Any,
    request: Any,
    query: torch.Tensor,
    kernel: Any,
    *,
    sample_id: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    require(
        torch.is_inference_mode_enabled(),
        "baseline arm requires the unified pair inference-mode scope",
    )
    for layer_index in plan.full_attention_layer_indices:
        sequence = request.layers[layer_index].sequence
        require(sequence.append_observer is None, "baseline append observer was enabled")
        sequence.strict_mask_check = False
    allocated_before, reserved_before = allocator_before_arm()
    registered = None
    output = None
    logits = None
    original_backend = backbone.config._attn_implementation
    start_ns = time.perf_counter_ns()
    try:
        ledger = build_arm_ledger(
            plan,
            request,
            kernel,
            call_observer=None,
        )
        registered = register_qwen35_vllm_q16_backend(ledger)
        backbone.config._attn_implementation = registered.name
        output = backbone(
            input_ids=query,
            past_key_values=request,
            use_cache=True,
        )
        logits = _last_logits(model, output)
        torch.cuda.synchronize()
        ledger_receipt = ledger.verify_complete()
    finally:
        backbone.config._attn_implementation = original_backend
        if registered is not None:
            _unregister_backend(registered.name)
    torch.cuda.synchronize()
    wall_ns = time.perf_counter_ns() - start_ns
    require(logits is not None, "baseline did not produce logits")
    timing = timing_receipt(
        arm="baseline",
        wall_ns=wall_ns,
        allocated_before=allocated_before,
        reserved_before=reserved_before,
        artifact_bytes=0,
    )
    output_receipt, logits_cpu = finalize_common_output(sample_id, logits)
    del output, logits
    return {
        **timing,
        "capture_policy": "optional-forkaudit-capture-disabled",
        "mandatory_functional_adapter_checks_retained": True,
        "append_observer_enabled": False,
        "call_observer_enabled": False,
        "ownership_receipt_enabled": False,
        "audit_artifact_manifest": None,
        "ledger": compact_ledger_receipt(ledger_receipt),
        "output": output_receipt,
    }, logits_cpu


def run_instrumented_arm(
    model: Any,
    backbone: Any,
    plan: Any,
    persistent: Any,
    request: Any,
    query: torch.Tensor,
    kernel: Any,
    artifact_root: Path,
    *,
    sample_id: str,
) -> tuple[dict[str, Any], torch.Tensor]:
    require(
        torch.is_inference_mode_enabled(),
        "instrumented arm requires the unified pair inference-mode scope",
    )
    for layer_index in plan.full_attention_layer_indices:
        request.layers[layer_index].sequence.strict_mask_check = False
    allocated_before, reserved_before = allocator_before_arm()
    registered = None
    collector = None
    output = None
    logits = None
    original_backend = backbone.config._attn_implementation
    start_ns = time.perf_counter_ns()
    try:
        collector = LiveCaptureWriter(artifact_root / sample_id, sample_id)
        source_before = _source_document_digests(
            persistent, plan.full_attention_layer_indices
        )
        persistent_gdn_guard = capture_persistent_gdn_guard(
            persistent, plan.linear_layer_indices
        )
        request_gdn_guard = capture_request_gdn_binding_guard(
            (request,),
            plan.linear_layer_indices,
            policy=POLICY_SHARED_BASE,
        )
        active_group = active_single_request_group(request, plan)
        kv_pre = validate_runtime_kv_ownership(
            persistent,
            active_group,
            plan,
            require_appended_tail_cow=False,
        )
        for layer_index in plan.full_attention_layer_indices:
            request.layers[layer_index].sequence.append_observer = (
                lambda event, index=int(layer_index): collector.append_observer(index, event)
            )
        ledger = build_arm_ledger(
            plan,
            request,
            kernel,
            call_observer=collector.call_observer,
        )
        registered = register_qwen35_vllm_q16_backend(ledger)
        backbone.config._attn_implementation = registered.name
        output = backbone(
            input_ids=query,
            past_key_values=request,
            use_cache=True,
        )
        logits = _last_logits(model, output)
        torch.cuda.synchronize()
        ledger_receipt = ledger.verify_complete()
        kv_post = validate_runtime_kv_ownership(
            persistent,
            active_group,
            plan,
            require_appended_tail_cow=True,
        )
        request_gdn = verify_request_gdn_binding_guard(
            request_gdn_guard,
            (request,),
            completed_request_indices=(0,),
        )
        persistent_gdn = verify_persistent_gdn_guard(
            persistent_gdn_guard,
            persistent,
        )
        source_after = _source_document_digests(
            persistent, plan.full_attention_layer_indices
        )
        require(source_before == source_after, "persistent KV source mutated")
        artifact_manifest = collector.finalize(
            {
                "ledger": ledger_receipt,
                "kv_pre": kv_pre,
                "kv_post": kv_post,
                "request_gdn": request_gdn,
                "persistent_gdn": persistent_gdn,
                "source_document_sha256_before": source_before,
                "source_document_sha256_after": source_after,
                "source_document_immutable": True,
            }
        )
    finally:
        backbone.config._attn_implementation = original_backend
        if registered is not None:
            _unregister_backend(registered.name)
        for layer_index in plan.full_attention_layer_indices:
            request.layers[layer_index].sequence.append_observer = None
        if collector is not None and not collector._finalized:
            collector.close_partial()
    torch.cuda.synchronize()
    wall_ns = time.perf_counter_ns() - start_ns
    require(logits is not None, "instrumented arm did not produce logits")
    timing = timing_receipt(
        arm="instrumented",
        wall_ns=wall_ns,
        allocated_before=allocated_before,
        reserved_before=reserved_before,
        artifact_bytes=int(artifact_manifest["artifact_bytes"]),
    )
    output_receipt, logits_cpu = finalize_common_output(sample_id, logits)
    del output, logits
    return {
        **timing,
        "capture_policy": "full-live-capture-and-ownership-receipt",
        "mandatory_functional_adapter_checks_retained": True,
        "append_observer_enabled": True,
        "call_observer_enabled": True,
        "ownership_receipt_enabled": True,
        "audit_artifact_manifest": artifact_manifest,
        "ledger": compact_ledger_receipt(ledger_receipt),
        "output": output_receipt,
    }, logits_cpu


def compare_pair_logits(
    baseline: torch.Tensor,
    instrumented: torch.Tensor,
) -> dict[str, Any]:
    require(baseline.shape == instrumented.shape, "pair logit shape drift")
    difference = (baseline - instrumented).abs()
    baseline_token = int(baseline.argmax(dim=-1).item())
    instrumented_token = int(instrumented.argmax(dim=-1).item())
    return {
        "full_vocab_logits_torch_equal": bool(torch.equal(baseline, instrumented)),
        "generated_token_equal": baseline_token == instrumented_token,
        "baseline_token_id": baseline_token,
        "instrumented_token_id": instrumented_token,
        "baseline_sha256": sha256_tensor(baseline),
        "instrumented_sha256": sha256_tensor(instrumented),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
    }


def run_pair(
    model: Any,
    backbone: Any,
    plan: Any,
    document: torch.Tensor,
    query: torch.Tensor,
    kernel: Any,
    artifact_root: Path,
    *,
    pair_label: str,
    pair_index: int | None,
    warmup: bool,
    execution_order: Sequence[str],
    baseline_slot: int,
    instrumented_slot: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    require(
        torch.is_inference_mode_enabled(),
        "entire pair requires one unified torch.inference_mode scope",
    )
    require(tuple(sorted((baseline_slot, instrumented_slot))) == (0, 1), "slot drift")
    persistent = _build_document_cache(backbone, document)
    conversion = convert_all_qwen35_full_layers_to_vllm_q16(
        persistent,
        plan,
        page_size=PAGE_SIZE,
        max_append_tokens=QUERY_TOKENS,
        max_request_forks=RESIDENT_COUNT,
    )
    group = build_resident_request_group(
        persistent,
        plan,
        resident_count=RESIDENT_COUNT,
        policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    require(
        group.requests[0] is not group.requests[1],
        "pair request objects alias",
    )
    source_before_pair = _source_document_digests(
        persistent, plan.full_attention_layer_indices
    )
    gdn_before_pair = compact_linear_digest(persistent, plan.linear_layer_indices)
    cells: dict[str, dict[str, Any]] = {}
    logits: dict[str, torch.Tensor] = {}
    for arm in execution_order:
        slot = baseline_slot if arm == "baseline" else instrumented_slot
        sample_id = f"{pair_label}-{arm}"
        if arm == "baseline":
            cell, cell_logits = run_baseline_arm(
                model,
                backbone,
                plan,
                group.requests[slot],
                query,
                kernel,
                sample_id=sample_id,
            )
        elif arm == "instrumented":
            cell, cell_logits = run_instrumented_arm(
                model,
                backbone,
                plan,
                persistent,
                group.requests[slot],
                query,
                kernel,
                artifact_root,
                sample_id=sample_id,
            )
        else:
            raise RuntimeError(f"unknown arm {arm}")
        cell["physical_request_slot"] = slot
        cells[arm] = cell
        logits[arm] = cell_logits
    source_after_pair = _source_document_digests(
        persistent, plan.full_attention_layer_indices
    )
    gdn_after_pair = compact_linear_digest(persistent, plan.linear_layer_indices)
    oracle = compare_pair_logits(logits["baseline"], logits["instrumented"])
    pair_valid = (
        source_before_pair == source_after_pair
        and gdn_before_pair == gdn_after_pair
        and oracle["full_vocab_logits_torch_equal"]
        and oracle["generated_token_equal"]
        and cells["baseline"]["audit_artifact_bytes"] == 0
        and cells["baseline"]["append_observer_enabled"] is False
        and cells["baseline"]["call_observer_enabled"] is False
        and cells["baseline"]["ledger"]["implementation"]
        == cells["instrumented"]["ledger"]["implementation"]
        == "MultiForkHitLedger"
        and cells["baseline"]["ledger"]["explicit_frozen_kernel"] is True
        and cells["instrumented"]["ledger"]["explicit_frozen_kernel"] is True
        and cells["baseline"]["ledger"]["call_observer_enabled"] is False
        and cells["instrumented"]["ledger"]["call_observer_enabled"] is True
        and cells["baseline"]["ledger"]["kernel_identity"]
        == cells["instrumented"]["ledger"]["kernel_identity"]
        and cells["instrumented"]["audit_artifact_manifest"]["append_event_count"]
        == EXPECTED_APPEND_EVENTS
        and cells["instrumented"]["audit_artifact_manifest"]["call_event_count"]
        == EXPECTED_CALL_EVENTS
        and cells["instrumented"]["audit_artifact_manifest"]["tensor_record_count"]
        == EXPECTED_TENSOR_RECORDS
    )
    return {
        "pair_label": pair_label,
        "pair_index": pair_index,
        "warmup": warmup,
        "discarded_from_estimands": warmup,
        "execution_order": list(execution_order),
        "baseline_slot": baseline_slot,
        "instrumented_slot": instrumented_slot,
        "same_persistent_document_within_pair": True,
        "same_query_tokens_within_pair": True,
        "distinct_request_objects": True,
        "source_document_sha256_before": source_before_pair,
        "source_document_sha256_after": source_after_pair,
        "source_document_immutable": source_before_pair == source_after_pair,
        "persistent_gdn_immutable": gdn_before_pair == gdn_after_pair,
        "cells": cells,
        "semantic_oracle": oracle,
        "pair_valid": pair_valid,
        "conversion": {
            "document_length": int(conversion.document_length),
            "page_size": int(conversion.page_size),
            "max_append_tokens": int(conversion.max_append_tokens),
            "max_request_forks": int(conversion.max_request_forks),
            "full_attention_layer_count": len(conversion.layer_indices),
        },
    }, {
        cells["baseline"]["output"]["sample_id"]: logits["baseline"],
        cells["instrumented"]["output"]["sample_id"]: logits["instrumented"],
    }


def summarize_measured_pairs(pairs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    require(len(pairs) == len(MEASURED_SCHEDULE), "measured pair count drift")
    rows = []
    for expected, pair in zip(MEASURED_SCHEDULE, pairs):
        pair_index, order, baseline_slot, instrumented_slot = expected
        require(
            pair.get("pair_index") == pair_index
            and tuple(pair.get("execution_order", ())) == order
            and pair.get("baseline_slot") == baseline_slot
            and pair.get("instrumented_slot") == instrumented_slot,
            "measured pair schedule drift",
        )
        baseline = pair["cells"]["baseline"]
        instrumented = pair["cells"]["instrumented"]
        baseline_ns = int(baseline["wall_time_ns"])
        instrumented_ns = int(instrumented["wall_time_ns"])
        require(baseline_ns > 0 and instrumented_ns > 0, "non-positive wall time")
        rows.append(
            {
                "pair_index": pair_index,
                "execution_order": list(order),
                "baseline_slot": baseline_slot,
                "instrumented_slot": instrumented_slot,
                "baseline_wall_time_ns": baseline_ns,
                "instrumented_wall_time_ns": instrumented_ns,
                "paired_wall_delta_ns": instrumented_ns - baseline_ns,
                "paired_wall_ratio": instrumented_ns / baseline_ns,
                "baseline_incremental_peak_allocated_bytes": int(
                    baseline["incremental_peak_allocated_bytes"]
                ),
                "instrumented_incremental_peak_allocated_bytes": int(
                    instrumented["incremental_peak_allocated_bytes"]
                ),
                "paired_incremental_peak_delta_bytes": int(
                    instrumented["incremental_peak_allocated_bytes"]
                )
                - int(baseline["incremental_peak_allocated_bytes"]),
                "instrumented_audit_artifact_bytes": int(
                    instrumented["audit_artifact_bytes"]
                ),
                "pair_valid": bool(pair["pair_valid"]),
            }
        )
    wall_deltas = [row["paired_wall_delta_ns"] for row in rows]
    wall_ratios = [row["paired_wall_ratio"] for row in rows]
    peak_deltas = [row["paired_incremental_peak_delta_bytes"] for row in rows]
    artifact_bytes = [row["instrumented_audit_artifact_bytes"] for row in rows]
    return {
        "measured_pair_count": len(rows),
        "warmup_pairs_included": 0,
        "rows": rows,
        "median_paired_wall_delta_ns": statistics.median(wall_deltas),
        "min_paired_wall_delta_ns": min(wall_deltas),
        "max_paired_wall_delta_ns": max(wall_deltas),
        "median_paired_wall_ratio": statistics.median(wall_ratios),
        "median_paired_incremental_peak_delta_bytes": statistics.median(peak_deltas),
        "median_instrumented_audit_artifact_bytes": statistics.median(artifact_bytes),
        "negative_numeric_deltas_preserved": True,
        "statistical_significance_claimed": False,
    }


def write_logit_sidecar(
    path: Path,
    logits: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "semantic sidecar path already exists")
    records = []
    offset = 0
    with path.open("xb") as handle:
        for sample_id in sorted(logits):
            tensor = logits[sample_id].detach().contiguous().cpu().float()
            raw = tensor.numpy().astype("<f4", copy=False).tobytes(order="C")
            handle.write(raw)
            records.append(
                {
                    "sample_id": sample_id,
                    "dtype": "float32-le",
                    "shape": [int(item) for item in tensor.shape],
                    "offset_bytes": offset,
                    "nbytes": len(raw),
                    "content_sha256": sha256_bytes(raw),
                    "token_id": int(tensor.argmax(dim=-1).item()),
                }
            )
            offset += len(raw)
    return {
        "schema_version": "qcomem-forkaudit-r29-live-overhead-logits-v1",
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_path(path),
        "record_count": len(records),
        "records": records,
        "terminal_exact_byte_coverage": (
            bool(records)
            and records[0]["offset_bytes"] == 0
            and records[-1]["offset_bytes"] + records[-1]["nbytes"]
            == path.stat().st_size
        ),
    }


def run_gpu(args: argparse.Namespace, design: Mapping[str, Any]) -> dict[str, Any]:
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.device_count() == 1, "formal process requires one visible GPU")
    torch.cuda.set_device(0)
    document_cpu, query_cpu, input_receipt = frozen_input_material(args)
    from transformers import AutoModelForImageTextToText

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
    require(tuple(plan.full_attention_layer_indices) == FULL_LAYERS, "full plan drift")
    require(tuple(plan.linear_layer_indices) == LINEAR_LAYERS, "linear plan drift")
    kernel_environment = audit_frozen_kernel_environment()
    require(
        kernel_environment.get("matches_frozen_environment") is True,
        "frozen vLLM kernel environment drift",
    )
    kernel = _resolve_vllm_unified_attention()
    document = document_cpu.to(device="cuda:0", dtype=torch.long)
    query = query_cpu.to(device="cuda:0", dtype=torch.long)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    require(not any(args.artifact_dir.iterdir()), "formal artifact directory is not empty")

    with torch.inference_mode():
        warmup, warmup_logits = run_pair(
            model,
            backbone,
            plan,
            document,
            query,
            kernel,
            args.artifact_dir,
            pair_label="warmup-pair",
            pair_index=None,
            warmup=True,
            execution_order=WARMUP_ORDER,
            baseline_slot=1,
            instrumented_slot=0,
        )
        all_logits = dict(warmup_logits)
        measured = []
        for pair_index, order, baseline_slot, instrumented_slot in MEASURED_SCHEDULE:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            pair, pair_logits = run_pair(
                model,
                backbone,
                plan,
                document,
                query,
                kernel,
                args.artifact_dir,
                pair_label=f"measured-pair-{pair_index}",
                pair_index=pair_index,
                warmup=False,
                execution_order=order,
                baseline_slot=baseline_slot,
                instrumented_slot=instrumented_slot,
            )
            measured.append(pair)
            require(not (set(all_logits) & set(pair_logits)), "duplicate logit sample ID")
            all_logits.update(pair_logits)
    summary = summarize_measured_pairs(measured)
    semantic_sidecar = write_logit_sidecar(
        args.artifact_dir / "semantic-logits.fp32.bin",
        all_logits,
    )
    all_pairs = [warmup, *measured]
    scientific_run_valid = (
        len(measured) == len(MEASURED_SCHEDULE)
        and warmup["pair_valid"] is True
        and all(pair["pair_valid"] is True for pair in measured)
        and semantic_sidecar["record_count"] == 2 * (1 + len(MEASURED_SCHEDULE))
    )
    properties = torch.cuda.get_device_properties(0)
    uuid = getattr(properties, "uuid", None)
    result = {
        "schema_version": SCHEMA,
        "status": "completed",
        "scientific_execution_completed": True,
        "scientific_run_valid": scientific_run_valid,
        "formal_evidence_eligible": scientific_run_valid,
        "protocol": PROTOCOL,
        "design_preregistration_raw_sha256": args.expected_design_sha256,
        "input": input_receipt,
        "hardware": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_device_count": torch.cuda.device_count(),
            "name": str(torch.cuda.get_device_name(0)),
            "uuid": None if uuid is None else str(uuid),
            "total_memory_bytes": int(properties.total_memory),
            "compute_capability": [int(properties.major), int(properties.minor)],
        },
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "cuda": None if torch.version.cuda is None else str(torch.version.cuda),
            "audited_packages": dict(AUDITED_PACKAGES),
            "kernel_mode": KERNEL_MODE,
            "kernel_environment": kernel_environment,
            "kernel_identity": {
                "module": str(getattr(kernel, "__module__", type(kernel).__module__)),
                "qualname": str(getattr(kernel, "__qualname__", type(kernel).__qualname__)),
                "signature": str(inspect.signature(kernel)),
            },
            "model_geometry": _audit_model_config_geometry(args.model),
        },
        "warmup_pair": warmup,
        "measured_pairs": measured,
        "paired_summary": summary,
        "semantic_sidecar": semantic_sidecar,
        "validity": {
            "warmup_pair_count": 1,
            "warmup_discarded_from_estimands": True,
            "measured_pair_count": len(measured),
            "alternating_schedule_verified": True,
            "all_pair_semantic_oracles_exact": all(
                pair["semantic_oracle"]["full_vocab_logits_torch_equal"]
                and pair["semantic_oracle"]["generated_token_equal"]
                for pair in all_pairs
            ),
            "all_live_receipts_valid": all(pair["pair_valid"] for pair in all_pairs),
            "negative_numeric_deltas_removed": False,
        },
        "claim_boundary": design["claim_boundary"],
    }
    return strict_json(result, label="complete live-overhead formal result")


def mock_result(design: Mapping[str, Any]) -> dict[str, Any]:
    validate_design(design)
    pairs = []
    for pair_index, order, baseline_slot, instrumented_slot in MEASURED_SCHEDULE:
        pairs.append(
            {
                "pair_index": pair_index,
                "execution_order": list(order),
                "baseline_slot": baseline_slot,
                "instrumented_slot": instrumented_slot,
                "pair_valid": True,
                "cells": {
                    "baseline": {
                        "wall_time_ns": 100 + pair_index,
                        "incremental_peak_allocated_bytes": 20,
                        "audit_artifact_bytes": 0,
                    },
                    "instrumented": {
                        "wall_time_ns": 110 + pair_index,
                        "incremental_peak_allocated_bytes": 24,
                        "audit_artifact_bytes": 50,
                    },
                },
            }
        )
    summary = summarize_measured_pairs(pairs)
    return {
        "schema_version": "qcomem-forkaudit-r29-live-overhead-mock-v1",
        "passed": True,
        "gpu_executed": False,
        "schedule_pair_count": len(MEASURED_SCHEDULE),
        "summary_shape_validated": summary["measured_pair_count"] == 5,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("mock", "formal"), required=True)
    result.add_argument("--design-preregistration", type=Path, required=True)
    result.add_argument("--expected-design-sha256", required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--artifact-dir", type=Path)
    result.add_argument("--model", type=Path)
    result.add_argument("--model-artifact-ledger", type=Path)
    result.add_argument("--model-weight-ledger", type=Path)
    result.add_argument("--pg19-data", type=Path)
    result.add_argument("--pg19-manifest", type=Path)
    result.add_argument("--upstream-code-ledger", type=Path)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    design = load_bound_json(
        args.design_preregistration,
        args.expected_design_sha256,
        "design preregistration",
    )
    validate_design(design)
    if args.stage == "mock":
        value = mock_result(design)
    else:
        required = (
            args.artifact_dir,
            args.model,
            args.model_artifact_ledger,
            args.model_weight_ledger,
            args.pg19_data,
            args.pg19_manifest,
            args.upstream_code_ledger,
        )
        require(all(item is not None for item in required), "formal paths are required")
        verify_bound_files(args)
        value = run_gpu(args, design)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
