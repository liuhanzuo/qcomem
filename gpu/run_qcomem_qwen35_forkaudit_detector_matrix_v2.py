from __future__ import annotations

"""Run the independent R28 18-case ForkAudit detector matrix.

Each assigned fault has two disposable N=2 cases: a fresh all-gates-on clean
case and a fresh live mutant case in which only that fault's preregistered
ForkAudit gate is suppressed.  The frozen RR2 W-run remains the authority for
the separate gate-on mutant receipt; this runner measures what happens after
that one target predicate is allowed to continue.

Scientific aborts are serialized.  Integration errors, missing receipts,
cleanup drift, and hook leaks are serialized as ``operational_invalid`` and
must be rejected by aggregation.  Missing outputs are never converted to a
negative detector result.
"""

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import re
import subprocess
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

import qcomem_forkaudit_storage_witness as storage_witness
import qcomem_vllm_paged_multifork_resident as resident
import run_qcomem_qwen35_forkaudit_review_revision as rr2
from qcomem_forkaudit_mutants import MUTANT_IDS, MUTANT_SPECS
from qcomem_forkaudit_selective_gate_policy import (
    GatePolicyError,
    SelectiveGatePolicy,
)
from qcomem_vllm_paged_fair_control import SHARED_REUSE
from qcomem_vllm_paged_kernel import (
    Q16KernelPagedTensorView,
    QComemPagedKernelError,
)
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    MultiForkHitLedger,
    RuntimeInvariantError,
    build_pg19_train_query_bank,
    build_resident_request_group,
    register_multifork_backend,
)
from run_qcomem_qwen35_vllm_paged_multifork_resident import (
    _last_logits as capture_live_last_logits,
)


RANK_SCHEMA = "forkaudit-detector-matrix-rank-v2"
PREREG_SCHEMA = "forkaudit-detector-matrix-preregistration-v2"
POLICY_SCHEMA = "forkaudit-selective-gate-policy-receipt-v2"
PRODUCTION_ASSERTION_ALLOWLIST_ID = "PA-M9-Q16-PAIRED-VIEWS-v1"
M7_PRODUCTION_ASSERTION_ALLOWLIST_ID = "PA-M7-CANONICAL-MASK-v1"
M8_SENTINEL_MESSAGE = "matrix M8 sentinel executed"
M7_PRODUCTION_MESSAGE = (
    "vLLM fused backend cannot replace this non-canonical attention mask"
)
M9_PRODUCTION_MESSAGE = "fused backend requires paired Q16 paged views"
POSITION_CANONICAL_MESSAGE = (
    "position_ids are not the canonical contiguous causal tail"
)

MUTANT_ASSIGNMENT: dict[int, tuple[str, ...]] = {
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
    mutant_id: MUTANT_SPECS[mutant_id].expected_gate_id
    for mutant_id in MUTANT_IDS
}
TARGET_REQUEST = {
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
TARGET_CONTRACT = {
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
EXPECTED_CLEAN_STAGES = {
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
TEACHER_TOKEN_RULE = {
    "M3": {"request_index": 0, "query_token_index": 31},
    "M4": {"request_index": 0, "query_token_index": 31},
    "M5": {"request_index": 1, "query_token_index": 31},
}


class MatrixV2Error(RuntimeError):
    """Operational defect in the R28 runner itself or its inputs."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixV2Error(message)


def m8_sentinel(*_args: Any, **_kwargs: Any) -> Any:
    """Frozen M8 fault payload; it is never a production assertion."""

    raise AssertionError(M8_SENTINEL_MESSAGE)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(canonical_bytes(value) + b"\n")
    pending.replace(path)


def check_file(path: Path, expected: str, label: str) -> bytes:
    require(re.fullmatch(r"[0-9a-f]{64}", expected or "") is not None, f"{label} SHA format")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected, f"{label} SHA drift")
    return payload


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor_bytes(tensor))


def _snapshot_allocator() -> dict[str, int]:
    torch.cuda.synchronize()
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
    }


def _cleanup_allocator() -> dict[str, int]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    return _snapshot_allocator()


def _function_descriptor(value: Any) -> dict[str, Any]:
    function = getattr(value, "__func__", value)
    try:
        source_sha = sha256_bytes(inspect.getsource(function).encode("utf-8"))
    except (OSError, TypeError):
        source_sha = None
    return {
        "module": str(getattr(function, "__module__", type(function).__module__)),
        "qualname": str(getattr(function, "__qualname__", type(function).__qualname__)),
        "source_sha256": source_sha,
    }


def _tensor_descriptor(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "kind": "tensor",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "sha256": tensor_sha(tensor),
    }


def _none_or_tensor_descriptor(value: Any) -> dict[str, Any]:
    if value is None:
        return {"kind": "none"}
    require(isinstance(value, torch.Tensor), "target value must be tensor or None")
    return _tensor_descriptor(value)


def _target_receipt(
    mutant_id: str,
    pre: Mapping[str, Any],
    mutated: Mapping[str, Any],
    restored: Mapping[str, Any],
) -> dict[str, Any]:
    pre_sha = sha256_json(pre)
    mutated_sha = sha256_json(mutated)
    restored_sha = sha256_json(restored)
    require(pre_sha != mutated_sha, f"{mutant_id} mutation is a no-op")
    require(restored_sha == pre_sha, f"{mutant_id} injector target did not restore")
    return {
        "status": "evaluated",
        "verified": True,
        "mutant_id": mutant_id,
        "target_contract": TARGET_CONTRACT[mutant_id],
        "pre_sha256": pre_sha,
        "mutated_sha256": mutated_sha,
        "restored_sha256": restored_sha,
        "pre_descriptor": dict(pre),
        "mutated_descriptor": dict(mutated),
        "restored_descriptor": dict(restored),
    }


@dataclass
class ActiveMutation:
    mutant_id: str
    capture: Callable[[], Mapping[str, Any]]
    undo: Callable[[], None]
    pre: Mapping[str, Any]
    mutated: Mapping[str, Any]
    restored_receipt: dict[str, Any] | None = None

    def restore(self) -> dict[str, Any]:
        if self.restored_receipt is None:
            self.undo()
            self.restored_receipt = _target_receipt(
                self.mutant_id, self.pre, self.mutated, self.capture()
            )
        return self.restored_receipt


def _activate_mutation(
    mutant_id: str,
    *,
    capture: Callable[[], Mapping[str, Any]],
    apply: Callable[[], None],
    undo: Callable[[], None],
) -> ActiveMutation:
    pre = dict(capture())
    apply()
    mutated = dict(capture())
    require(sha256_json(pre) != sha256_json(mutated), f"{mutant_id} mutation no-op")
    return ActiveMutation(mutant_id, capture, undo, pre, mutated)


@dataclass
class Runtime:
    model: Any
    backbone: Any
    plan: Any
    kernel: Any
    document: torch.Tensor
    queries: tuple[torch.Tensor, ...]
    input_receipt: dict[str, Any]
    hardware: dict[str, Any]
    allocator_baseline: dict[str, int] | None = None
    discarded_warmup_receipt: dict[str, Any] | None = None


def _gpu_receipt(expected_uuid: str) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(visible == expected_uuid, "CUDA_VISIBLE_DEVICES/assignment mismatch")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "rank needs one GPU")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    capability = list(torch.cuda.get_device_capability(0))
    require(capability == [9, 0] and "H20" in props.name, "rank is not an H20 sm90 GPU")
    row = subprocess.run(
        [
            "nvidia-smi",
            f"--id={expected_uuid}",
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    columns = [item.strip() for item in row.split(",")]
    require(len(columns) == 3 and columns[0] == expected_uuid, "nvidia-smi UUID drift")
    return {
        "uuid": columns[0],
        "name": columns[1],
        "memory_mib": int(columns[2]),
        "compute_capability": capability,
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


def _load_runtime(args: argparse.Namespace, prereg: Mapping[str, Any]) -> Runtime:
    import build_qcomem_forkaudit_rr2_input_manifest as rr2_builder
    import qcomem_joint_policy as joint_policy
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

    require(not torch.cuda.is_initialized(), "input rebuild must precede CUDA init")
    pg19_raw = check_file(args.pg19_data, args.expected_pg19_sha256, "PG19 data")
    manifest_raw = check_file(
        args.pg19_manifest, args.expected_pg19_manifest_sha256, "PG19 manifest"
    )
    check_file(args.frozen_query_banks, args.expected_query_banks_sha256, "query banks")
    check_file(args.model_weight_ledger, args.expected_weight_ledger_sha256, "weight ledger")
    check_file(args.model_artifact_ledger, args.expected_artifact_ledger_sha256, "artifact ledger")
    banks = load_json(args.frozen_query_banks)
    require(isinstance(banks, list) and len(banks) == 8, "query bank rank coverage")
    bank = banks[args.rank]
    tokenizer = rr2_builder.load_local_tokenizer(args.model_dir)
    records, _audit = rr2_builder._audit_pg19_train64_bytes(
        pg19_raw, manifest_raw, expectations=rr2_builder.FORMAL_EXPECTATIONS
    )
    windows, windows_sha = joint_policy.build_pg19_calibration_windows(
        records,
        tokenizer,
        books=rr2.FORMAL_BOOKS,
        document_tokens=rr2.FORMAL_DOCUMENT_TOKENS,
        query_tokens=rr2.FORMAL_QUERY_TOKENS,
        stride=rr2.FORMAL_WINDOW_STRIDE,
        candidate_windows_per_book=8,
        seed=20260817,
    )
    require(windows_sha == args.expected_windows_sha256, "PG19 window digest drift")
    window = windows[args.rank]
    queries, query_audit = build_pg19_train_query_bank(
        records,
        tokenizer,
        window,
        document_tokens=rr2.FORMAL_DOCUMENT_TOKENS,
        query_tokens=rr2.FORMAL_QUERY_TOKENS,
        count=max(rr2.FORMAL_RESIDENT_COUNTS),
        query_stride=rr2.FORMAL_QUERY_BANK_STRIDE,
    )
    document_cpu = window.document_ids.detach().contiguous().unsqueeze(0)
    require(tensor_sha(document_cpu) == bank["document_token_ids_sha256"], "document digest")
    require(
        [tensor_sha(query) for query in queries]
        == [row["query_token_ids_sha256"] for row in bank["rows"]],
        "query digest drift",
    )
    require(
        [int(row["source_token_offset"]) for row in query_audit["rows"]]
        == [int(row["source_token_offset"]) for row in bank["rows"]],
        "query coordinates drift",
    )
    hardware = _gpu_receipt(args.expected_gpu_uuid)
    weight_rows = rr2._parse_sha256_ledger(
        args.model_weight_ledger.read_bytes(), label="R28 model weight ledger"
    )
    artifact_rows = rr2._parse_sha256_ledger(
        args.model_artifact_ledger.read_bytes(), label="R28 model artifact ledger"
    )
    rr2._verify_weight_ledger_structure(weight_rows, model_dir=args.model_dir)
    rr2._verify_model_ledger(
        artifact_rows, model_dir=args.model_dir, label="R28 model artifact ledger"
    )
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_dir),
        revision=rr2.FORMAL_MODEL_REVISION,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    outer = getattr(model, "model", None)
    if outer is not None and hasattr(outer, "visual"):
        outer.visual = None
    model.eval()
    _audit_model_config_geometry(args.model_dir)
    plan = audit_qwen35_functional_stack_plan(model)
    require(tuple(plan.full_attention_layer_indices) == rr2.FORMAL_FULL_LAYERS, "full layers")
    require(tuple(plan.linear_layer_indices) == rr2.FORMAL_LINEAR_LAYERS, "linear layers")
    environment = audit_frozen_kernel_environment()
    require(environment.get("matches_frozen_environment") is True, "kernel environment drift")
    model = model.to(device="cuda:0", dtype=torch.bfloat16)
    backbone = _resolve_backbone(model)
    kernel = _resolve_vllm_unified_attention()
    document = document_cpu.to(device="cuda:0", non_blocking=False)
    live_queries = tuple(query.to(device="cuda:0", non_blocking=False) for query in queries)
    input_binding = prereg.get("input_binding")
    require(isinstance(input_binding, dict), "preregistration input_binding")
    return Runtime(
        model=model,
        backbone=backbone,
        plan=plan,
        kernel=kernel,
        document=document,
        queries=live_queries,
        input_receipt={
            "model_revision": rr2.FORMAL_MODEL_REVISION,
            "weight_ledger_raw_sha256": args.expected_weight_ledger_sha256,
            "artifact_ledger_raw_sha256": args.expected_artifact_ledger_sha256,
            "pg19_sha256": args.expected_pg19_sha256,
            "pg19_manifest_sha256": args.expected_pg19_manifest_sha256,
            "windows_sha256": windows_sha,
            "frozen_query_banks_sha256": args.expected_query_banks_sha256,
            "original_rr2_run_id": input_binding["original_rr2_run_id"],
            "original_rr2_receipt_manifest_sha256": input_binding[
                "original_rr2_receipt_manifest_sha256"
            ],
            "preregistration_sha256": args.expected_preregistration_sha256,
            "code_ledger_sha256": args.expected_code_ledger_sha256,
            "imported_rr2_code_ledger_sha256": (
                args.expected_imported_rr2_code_ledger_sha256
            ),
            "external_pin_payload_sha256": args.expected_external_pin_payload_sha256,
            "rank": args.rank,
            "document_token_ids_sha256": tensor_sha(document),
            "query_token_ids_sha256": [tensor_sha(query) for query in live_queries],
            "query_bank_manifest_sha256": bank["manifest_sha256"],
            "logit_capture": {
                "callable": "run_qcomem_qwen35_vllm_paged_multifork_resident._last_logits",
                "source_sha256": sha256_bytes(
                    inspect.getsource(capture_live_last_logits).encode("utf-8")
                ),
            },
        },
        hardware=hardware,
    )


def _build_group(runtime: Runtime) -> tuple[Any, Any, Any]:
    persistent, _conversion = rr2._convert_persistent(
        runtime.backbone, runtime.plan, runtime.document, resident_count=2
    )
    persistent_guard = storage_witness.capture_persistent_gdn_guard(
        persistent, runtime.plan.linear_layer_indices
    )
    group = build_resident_request_group(
        persistent,
        runtime.plan,
        resident_count=2,
        policy=SHARED_REUSE,
        gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
    )
    from run_qcomem_qwen35_vllm_paged_multifork_resident import (
        _set_production_no_mask,
    )

    _set_production_no_mask(group, runtime.plan.full_attention_layer_indices)
    return persistent, group, persistent_guard


class ArgumentInjectingLedger(MultiForkHitLedger):
    """Inject M6/M7 once while retaining the strict production ledger."""

    def __init__(self, *args: Any, mutant_id: str, **kwargs: Any) -> None:
        require(mutant_id in ("M6", "M7"), "argument injector mutant")
        super().__init__(*args, **kwargs)
        self.injector_mutant_id = mutant_id
        self.injector_count = 0
        self.injector_target_restoration: dict[str, Any] | None = None

    def attention_forward(
        self,
        module: Any,
        query: torch.Tensor,
        key: Any,
        value: Any,
        attention_mask: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        index = getattr(module, "layer_idx", None)
        if index != self.indices[0] or self.injector_count != 0:
            return super().attention_forward(
                module, query, key, value, attention_mask, *args, **kwargs
            )
        self.injector_count += 1
        if self.injector_mutant_id == "M6":
            original = kwargs.get("position_ids")
            require(isinstance(original, torch.Tensor), "M6 position_ids missing")
            pre = _tensor_descriptor(original)
            mutated = original.detach().clone() + 1
            mutated_descriptor = _tensor_descriptor(mutated)
            kwargs["position_ids"] = mutated
            try:
                return super().attention_forward(
                    module, query, key, value, attention_mask, *args, **kwargs
                )
            finally:
                # The original call argument was never overwritten; prove it
                # remains byte-identical after the scoped replacement.
                self.injector_target_restoration = _target_receipt(
                    "M6", pre, mutated_descriptor, _tensor_descriptor(original)
                )
        original_mask = attention_mask
        pre = _none_or_tensor_descriptor(original_mask)
        total_length = int(value.sequence.sequence_length)
        mutated_mask = torch.ones(
            (1, 1, int(query.shape[-2]), total_length),
            dtype=torch.bool,
            device=query.device,
        )
        mutated_descriptor = _tensor_descriptor(mutated_mask)
        try:
            return super().attention_forward(
                module, query, key, value, mutated_mask, *args, **kwargs
            )
        finally:
            self.injector_target_restoration = _target_receipt(
                "M7",
                pre,
                mutated_descriptor,
                _none_or_tensor_descriptor(original_mask),
            )


class M9RawTensorLedger(MultiForkHitLedger):
    """Keep the M9 key as a raw tensor through production ``_paired_sequence``.

    The base ledger reads ``key.sequence`` immediately after ``KV_PAGED_VIEW``.
    Doing that would create an incidental ``AttributeError`` after selective
    suppression.  This subclass uses the still-paged value only for ledger
    bookkeeping, then passes the untouched raw key to the real production
    adapter, whose allowlisted paired-view assertion remains authoritative.
    """

    raw_key_reached_production: dict[str, Any] | None = None

    def attention_forward(
        self,
        module: Any,
        query: torch.Tensor,
        key: Any,
        value: Any,
        attention_mask: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        index = getattr(module, "layer_idx", None)
        resident._runtime_require(
            index in self.arena_ids,
            "KV_PAGED_VIEW",
            f"unexpected full-attention layer {index}",
        )
        resident._runtime_require(
            self.kernel is self._frozen_kernel,
            "KERNEL_CALLABLE_ID",
            "unified_attention callable changed after ledger construction",
        )
        resident._runtime_require(
            isinstance(key, Q16KernelPagedTensorView)
            and isinstance(value, Q16KernelPagedTensorView),
            "KV_PAGED_VIEW",
            "dense fallback",
        )
        require(isinstance(key, torch.Tensor), "M9 key is not the raw dense tensor")
        require(
            isinstance(value, Q16KernelPagedTensorView) and value.kind == "value",
            "M9 paired value is not the original Q16 paged value",
        )
        sequence = value.sequence
        resident._runtime_require(
            id(sequence) == self.sequence_ids[index],
            "KV_SEQUENCE_ID",
            "request sequence changed or another request was misbound",
        )
        resident._runtime_require(
            id(sequence.arena) == self.arena_ids[index],
            "KV_SEQUENCE_ID",
            "request arena changed",
        )
        require(self.counts[index] < self.expected_calls_per_layer, "call budget exceeded")
        query_length = int(query.shape[-2])
        delta = int(sequence.sequence_length) - self.last_lengths[index]
        require(delta == query_length, "current append delta differs from query tokens")
        append_event_count = int(sequence._append_event_count)
        resident._runtime_require(
            append_event_count == self.last_append_event_counts[index] + 1,
            "KV_APPEND_EVENT",
            "attention must consume exactly one cache append event",
        )
        append_audit = sequence.last_append_audit
        resident._runtime_require(
            isinstance(append_audit, dict)
            and append_audit.get("append_event_index")
            == self.last_append_event_counts[index]
            and append_audit.get("append_tokens") == delta
            and append_audit.get("sequence_length_after")
            == int(sequence.sequence_length),
            "KV_APPEND_EVENT",
            "cache append receipt differs from attention delta",
        )
        resident._runtime_require(
            attention_mask is None and not sequence.strict_mask_check,
            "MASK_CONTRACT",
            "production path materialized a mask",
        )
        position_ids = kwargs.pop("position_ids", None)
        from qcomem_qwen35_vllm_paged_integration import (
            Qwen35VllmPagedIntegrationError,
        )

        try:
            resident.validate_qwen35_post_rope_position_ids(
                position_ids,
                query=query,
                total_length=int(sequence.sequence_length),
                strict_tail_values=self.strict_position_values,
            )
        except Qwen35VllmPagedIntegrationError as exc:
            if str(exc) != POSITION_CANONICAL_MESSAGE:
                raise
            raise RuntimeInvariantError(
                "POSITION_CANONICAL_VALUES",
                f"post-RoPE position validation failed: {exc}",
            ) from exc
        self.raw_key_reached_production = {
            "raw_key_type": "torch.Tensor",
            "raw_key_sha256": tensor_sha(key),
            "raw_key_shape": list(key.shape),
            "paired_value_type": "Q16KernelPagedTensorView(value)",
            "production_entrypoint": (
                "qcomem_vllm_paged_kernel."
                "vllm_triton_q16_paged_attention_forward"
            ),
        }
        # Deliberately call the production adapter with ``key`` unchanged.
        return resident.vllm_triton_q16_paged_attention_forward(
            module,
            query,
            key,
            value,
            attention_mask,
            *args,
            audit={},
            _kernel=self.kernel,
            **kwargs,
        )


def _make_ledgers(
    runtime: Runtime,
    group: Any,
    *,
    calls: Mapping[int, int],
    mutant_id: str,
    activate_mutation: bool,
) -> tuple[list[MultiForkHitLedger], list[str]]:
    ledgers: list[MultiForkHitLedger] = []
    backends: list[str] = []
    try:
        for request_index in range(2):
            expected_calls = int(calls.get(request_index, 0))
            if expected_calls == 0:
                ledgers.append(None)  # type: ignore[arg-type]
                backends.append("")
                continue
            kwargs: dict[str, Any] = {}
            ledger_class: type[MultiForkHitLedger] = MultiForkHitLedger
            if activate_mutation and request_index == TARGET_REQUEST[mutant_id]:
                if mutant_id in ("M6", "M7"):
                    ledger_class = ArgumentInjectingLedger
                    kwargs["mutant_id"] = mutant_id
                elif mutant_id == "M9":
                    ledger_class = M9RawTensorLedger
            ledger = ledger_class(
                runtime.plan,
                group.requests[request_index],
                request_index=request_index,
                resident_count=2,
                request_policy=group.policy,
                expected_calls_per_layer=expected_calls,
                initial_query_tokens=rr2.FORMAL_QUERY_TOKENS,
                kernel=runtime.kernel,
                strict_position_values=True,
                **kwargs,
            )
            ledgers.append(ledger)
            backends.append(register_multifork_backend(ledger))
    except BaseException:
        rr2._unregister_backends([name for name in backends if name])
        raise
    return ledgers, backends


def _call_budget(mutant_id: str) -> dict[int, int]:
    if mutant_id == "M3":
        return {0: 2, 1: 1}
    if mutant_id == "M4":
        return {0: 2}
    if mutant_id == "M5":
        return {0: 1, 1: 2}
    return {TARGET_REQUEST[mutant_id]: 1}


def _model_step(
    runtime: Runtime,
    group: Any,
    backend: str,
    request_index: int,
    tokens: torch.Tensor,
) -> tuple[torch.Tensor, int, dict[str, Any]]:
    require(bool(backend), "model step backend missing")
    original = runtime.backbone.config._attn_implementation
    restored = False
    try:
        runtime.backbone.config._attn_implementation = backend
        output = runtime.backbone(
            input_ids=tokens,
            past_key_values=group.requests[request_index],
            use_cache=True,
        )
        logits = (
            capture_live_last_logits(runtime.model, output)
            .detach()
            .cpu()
            .float()
            .contiguous()
        )
        require(tuple(logits.shape) == (1, 248320), "live logit shape drift")
        require(bool(torch.isfinite(logits).all()), "non-finite logits")
        token = int(logits.argmax(dim=-1).item())
        return logits, token, {
            "backend": backend,
            "request_index": request_index,
            "input_token_ids_sha256": tensor_sha(tokens),
            "input_token_count": int(tokens.numel()),
            "attn_implementation_restored": True,
        }
    finally:
        runtime.backbone.config._attn_implementation = original
        restored = runtime.backbone.config._attn_implementation == original
        if not restored:
            raise MatrixV2Error("_attn_implementation did not restore")


def _teacher_token(runtime: Runtime, mutant_id: str) -> tuple[torch.Tensor, dict[str, Any]]:
    rule = TEACHER_TOKEN_RULE[mutant_id]
    request_index = int(rule["request_index"])
    token_index = int(rule["query_token_index"])
    token = runtime.queries[request_index][:, token_index : token_index + 1].detach().clone()
    require(tuple(token.shape) == (1, 1) and token.dtype == torch.long, "teacher token")
    return token, {
        "rule": "last token of the corresponding frozen 32-token query",
        "source_coordinate": (
            f"frozen_query_bank[rank][{request_index}][{token_index}]"
        ),
        "request_index": request_index,
        "query_token_index": token_index,
        "token_id": int(token.item()),
        "token_sha256": tensor_sha(token),
        "independent_of_path_argmax": True,
    }


def _discarded_warmup(runtime: Runtime) -> dict[str, Any]:
    """Warm the same N=2 full-model path, discard it, then freeze baseline."""

    def execute() -> dict[str, Any]:
        persistent = group = None
        backends: list[str] = []
        policy = SelectiveGatePolicy(None)
        try:
            persistent, group, _persistent_guard = _build_group(runtime)
            ledgers, backends = _make_ledgers(
                runtime,
                group,
                calls={0: 1},
                mutant_id="M2",
                activate_mutation=False,
            )
            with policy:
                logits, _token, _step = _model_step(
                    runtime, group, backends[0], 0, runtime.queries[0]
                )
                require(ledgers[0].verify_complete()["verified"] is True, "warmup ledger")
                del logits
            return {"policy": policy.receipt()}
        finally:
            if backends:
                rr2._unregister_backends([name for name in backends if name])
            persistent = group = None

    # Use the same inference semantics as every measured cell so lazy
    # inference-only allocations cannot first appear in the clean lane.
    with torch.inference_mode():
        detail = execute()
    baseline = _cleanup_allocator()
    runtime.allocator_baseline = baseline
    receipt = {
        "performed": True,
        "discarded": True,
        "completed_before_case_nonces": True,
        "post_warmup_baseline": dict(baseline),
        "gc_collect_completed": True,
        "cuda_empty_cache_completed": True,
        "cuda_synchronize_completed": True,
        "all_gates_on_policy_restored": detail["policy"][
            "all_original_function_identities_restored"
        ],
    }
    runtime.discarded_warmup_receipt = receipt
    return receipt


def _materialize_m9_dense_key(layer: Any) -> tuple[Any, torch.Tensor]:
    original = layer.keys
    require(isinstance(original, Q16KernelPagedTensorView), "M9 original key view")
    require(
        isinstance(layer.values, Q16KernelPagedTensorView)
        and layer.values.kind == "value"
        and layer.values.sequence is original.sequence,
        "M9 original paired value",
    )
    sequence = original.sequence
    dense_batches = []
    for table_row in sequence.active_block_table.to(torch.int64):
        blocks = sequence.arena.key_cache.index_select(0, table_row)
        logical = blocks.reshape(
            -1, rr2.FORMAL_NUM_KV_HEADS, rr2.FORMAL_HEAD_DIM
        )[: sequence.sequence_length]
        dense_batches.append(logical.permute(1, 0, 2).contiguous())
    dense = torch.stack(dense_batches, dim=0)
    require(tuple(dense.shape)[0:2] == (1, rr2.FORMAL_NUM_KV_HEADS), "M9 dense shape")
    return original, dense


def _activate_initial_mutation(
    mutant_id: str,
    *,
    runtime: Runtime,
    group: Any,
    ledgers: Sequence[MultiForkHitLedger | None],
) -> ActiveMutation | None:
    first_full = int(runtime.plan.full_attention_layer_indices[0])
    if mutant_id == "M1":
        peer = group.requests[0].layers[first_full].sequence.reservations
        target = group.requests[1].layers[first_full].sequence.reservations
        saved = target.detach().clone()

        def capture() -> Mapping[str, Any]:
            return {
                "target": _tensor_descriptor(target),
                "equals_peer_values": bool(torch.equal(target, peer)),
            }

        return _activate_mutation(
            mutant_id,
            capture=capture,
            apply=lambda: target.copy_(peer),
            undo=lambda: target.copy_(saved),
        )
    if mutant_id == "M2":
        ledger = ledgers[0]
        require(isinstance(ledger, MultiForkHitLedger), "M2 ledger")
        original = ledger.sequence_ids[first_full]
        peer = id(group.requests[1].layers[first_full].sequence)
        require(peer != original, "M2 peer binding no-op")

        def capture() -> Mapping[str, Any]:
            current = ledger.sequence_ids[first_full]
            return {
                "matches_request0": current == original,
                "matches_request1": current == peer,
            }

        return _activate_mutation(
            mutant_id,
            capture=capture,
            apply=lambda: ledger.sequence_ids.__setitem__(first_full, peer),
            undo=lambda: ledger.sequence_ids.__setitem__(first_full, original),
        )
    if mutant_id == "M3":
        sequence = group.requests[0].layers[first_full].sequence
        require(
            "_detach_partial_document_tail" not in vars(sequence),
            "M3 preexisting instance override",
        )

        def omit_tail_cow(_self: Any, _batch_index: int) -> None:
            return None

        original_descriptor = _function_descriptor(
            sequence._detach_partial_document_tail
        )

        def capture() -> Mapping[str, Any]:
            return {
                "has_instance_override": (
                    "_detach_partial_document_tail" in vars(sequence)
                ),
                "callable": _function_descriptor(
                    sequence._detach_partial_document_tail
                ),
                "matches_original_callable": (
                    _function_descriptor(sequence._detach_partial_document_tail)
                    == original_descriptor
                ),
            }

        def apply() -> None:
            sequence._detach_partial_document_tail = types.MethodType(  # type: ignore[method-assign]
                omit_tail_cow, sequence
            )

        def undo() -> None:
            vars(sequence).pop("_detach_partial_document_tail")

        return _activate_mutation(
            mutant_id, capture=capture, apply=apply, undo=undo
        )
    if mutant_id == "M8":
        ledger = ledgers[0]
        require(isinstance(ledger, MultiForkHitLedger), "M8 ledger")
        original = ledger.kernel

        def capture() -> Mapping[str, Any]:
            return {
                "callable": _function_descriptor(ledger.kernel),
                "matches_frozen_kernel": ledger.kernel is original,
                "matches_frozen_m8_fault_sentinel": ledger.kernel is m8_sentinel,
            }

        return _activate_mutation(
            mutant_id,
            capture=capture,
            apply=lambda: setattr(ledger, "kernel", m8_sentinel),
            undo=lambda: setattr(ledger, "kernel", original),
        )
    if mutant_id == "M9":
        layer = group.requests[0].layers[first_full]
        original, dense = _materialize_m9_dense_key(layer)

        def capture() -> Mapping[str, Any]:
            current = layer.keys
            if isinstance(current, torch.Tensor):
                return {
                    "representation": "raw-torch-tensor",
                    "tensor": _tensor_descriptor(current),
                    "paired_value_remains_q16": isinstance(
                        layer.values, Q16KernelPagedTensorView
                    ),
                }
            return {
                "representation": "q16-paged-key-view",
                "kind": current.kind,
                "paired_value_remains_q16": isinstance(
                    layer.values, Q16KernelPagedTensorView
                ),
            }

        return _activate_mutation(
            mutant_id,
            capture=capture,
            apply=lambda: setattr(layer, "keys", dense),
            undo=lambda: setattr(layer, "keys", original),
        )
    return None


def _activate_gdn_mutation(
    mutant_id: str,
    *,
    runtime: Runtime,
    persistent: Any,
    group: Any,
) -> ActiveMutation:
    require(mutant_id in ("M4", "M5"), "GDN mutation id")
    first_linear = int(runtime.plan.linear_layer_indices[0])
    target_request = 0 if mutant_id == "M4" else 1
    values = group.requests[target_request].layers[first_linear].conv_states
    state_index = sorted(values)[0]
    original = values[state_index]
    source = (
        persistent.layers[first_linear].conv_states[state_index]
        if mutant_id == "M4"
        else group.requests[0].layers[first_linear].conv_states[state_index]
    )
    require(original is not source, f"{mutant_id} source alias no-op")

    def capture() -> Mapping[str, Any]:
        current = values[state_index]
        return {
            "is_original_request_state": current is original,
            "is_fault_alias_source": current is source,
            "tensor": _tensor_descriptor(current),
            "request_index": target_request,
            "layer_index": first_linear,
            "state_index": int(state_index),
        }

    return _activate_mutation(
        mutant_id,
        capture=capture,
        apply=lambda: values.__setitem__(state_index, source),
        undo=lambda: values.__setitem__(state_index, original),
    )


def _stack_provenance(exc: BaseException) -> dict[str, Any]:
    rows = []
    current = exc.__traceback__
    while current is not None:
        frame = current.tb_frame
        filename = Path(frame.f_code.co_filename).name
        rows.append(
            {
                "module": str(frame.f_globals.get("__name__", "<unknown>")),
                "file": filename,
                "function": frame.f_code.co_name,
                "line": int(current.tb_lineno),
            }
        )
        current = current.tb_next
    return {"frames": rows, "frames_sha256": sha256_json(rows)}


def _has_frame(
    provenance: Mapping[str, Any], *, file: str, function: str
) -> bool:
    return any(
        row.get("file") == file and row.get("function") == function
        for row in provenance.get("frames", [])
    )


def _classify_exception(exc: BaseException, mutant_id: str) -> dict[str, Any]:
    provenance = _stack_provenance(exc)
    module = type(exc).__module__
    name = type(exc).__name__
    message = str(exc)
    base = {
        "exception_module": module,
        "exception_type": name,
        "message": message,
        "stack_provenance": provenance,
    }
    gate_id = getattr(exc, "gate_id", None)
    if isinstance(exc, RuntimeInvariantError) and gate_id == "POSITION_CANONICAL_VALUES":
        # The imported RR2 ledger wraps every exception raised by its
        # position validator.  Preserve the v2 policy's exact-type/exact-
        # message boundary here so an AttributeError or integration defect
        # cannot be relabelled as a valid ForkAudit catch by that broad legacy
        # wrapper.
        from qcomem_qwen35_vllm_paged_integration import (
            Qwen35VllmPagedIntegrationError,
        )

        cause = exc.__cause__
        if not (
            type(cause) is Qwen35VllmPagedIntegrationError
            and str(cause) == POSITION_CANONICAL_MESSAGE
        ):
            return {
                **base,
                "completion_status": "operational_invalid",
                "classification": "operational_invalid",
                "other_gate_id": None,
                "valid_scientific_outcome": False,
            }
    if isinstance(exc, RuntimeInvariantError) or isinstance(
        exc, storage_witness.GDNStorageWitnessError
    ):
        return {
            **base,
            "completion_status": "classified_abort",
            "classification": "other_forkaudit_gate",
            "other_gate_id": str(gate_id),
            "valid_scientific_outcome": True,
        }
    if (
        mutant_id == "M8"
        and type(exc) is AssertionError
        and message == M8_SENTINEL_MESSAGE
        and _has_frame(
            provenance,
            file="run_qcomem_qwen35_forkaudit_detector_matrix_v2.py",
            function="m8_sentinel",
        )
    ):
        return {
            **base,
            "completion_status": "classified_abort",
            "classification": "fault_payload_abort",
            "other_gate_id": None,
            "valid_scientific_outcome": True,
        }
    if (
        mutant_id == "M7"
        and type(exc) is QComemPagedKernelError
        and module == "qcomem_vllm_paged_kernel"
        and message == M7_PRODUCTION_MESSAGE
        and _has_frame(
            provenance,
            file="qcomem_vllm_paged_kernel.py",
            function="validate_canonical_tail_causal_mask",
        )
    ):
        return {
            **base,
            "completion_status": "classified_abort",
            "classification": "production_assertion",
            "production_assertion_allowlist_id": M7_PRODUCTION_ASSERTION_ALLOWLIST_ID,
            "other_gate_id": None,
            "valid_scientific_outcome": True,
        }
    if (
        mutant_id == "M9"
        and type(exc) is QComemPagedKernelError
        and module == "qcomem_vllm_paged_kernel"
        and message == M9_PRODUCTION_MESSAGE
        and _has_frame(
            provenance,
            file="qcomem_vllm_paged_kernel.py",
            function="_paired_sequence",
        )
    ):
        return {
            **base,
            "completion_status": "classified_abort",
            "classification": "production_assertion",
            "production_assertion_allowlist_id": PRODUCTION_ASSERTION_ALLOWLIST_ID,
            "other_gate_id": None,
            "valid_scientific_outcome": True,
        }
    if isinstance(exc, (torch.OutOfMemoryError, MatrixV2Error, GatePolicyError)) or type(
        exc
    ) in (AttributeError, NameError, ImportError, KeyError, TypeError, AssertionError):
        return {
            **base,
            "completion_status": "operational_invalid",
            "classification": "operational_invalid",
            "other_gate_id": None,
            "valid_scientific_outcome": False,
        }
    if "out of memory" in message.lower():
        return {
            **base,
            "completion_status": "operational_invalid",
            "classification": "operational_invalid",
            "other_gate_id": None,
            "valid_scientific_outcome": False,
        }
    return {
        **base,
        "completion_status": "operational_invalid",
        "classification": "operational_invalid",
        "other_gate_id": None,
        "valid_scientific_outcome": False,
    }


def _detector_cell(status: str, caught: bool | None, **extra: Any) -> dict[str, Any]:
    require(status in ("evaluated", "not_evaluated"), "detector status")
    require(
        (status == "evaluated" and type(caught) is bool)
        or (status == "not_evaluated" and caught is None),
        "detector tri-state",
    )
    return {"status": status, "caught": caught, **extra}


def _persist_stage_sidecar(
    *,
    case_id: str,
    stage: str,
    logits: torch.Tensor,
    sidecar_root: Path,
    rank_root: Path,
) -> dict[str, Any]:
    """Atomically persist one completed stage before the next stage starts."""

    require(isinstance(logits, torch.Tensor), "sidecar logits")
    require(tuple(logits.shape) == (1, 248320), "sidecar shape")
    payload = logits.numpy().tobytes(order="C")
    target_dir = sidecar_root / case_id
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{stage}.fp32.bin"
    require(not target.exists(), "stage sidecar already exists")
    pending = target.with_suffix(target.suffix + ".tmp")
    require(not pending.exists(), "stage sidecar pending path exists")
    with pending.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(target)
    # Bind directory metadata before execution may proceed to the next stage.
    directory_fd = os.open(target_dir, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    relative = target.resolve().relative_to(rank_root.resolve()).as_posix()
    return {
        "case_id": case_id,
        "stage": stage,
        "relative_path": relative,
        "sha256": sha256_bytes(payload),
        "bytes": len(payload),
        "dtype": "float32",
        "shape": [1, 248320],
        "persisted_before_next_stage": True,
        "atomic_tmp_replace_completed": True,
        "file_and_directory_fsync_completed": True,
    }


def _run_case_impl(
    runtime: Runtime,
    *,
    rank: int,
    mutant_id: str,
    lane: str,
    preregistration_sha256: str,
    sidecar_root: Path,
    rank_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(lane in ("clean", "target_suppressed"), "lane")
    activate_mutation = lane == "target_suppressed"
    expected_gate = EXPECTED_GATES[mutant_id]
    case_id = f"{mutant_id}:{lane}"
    case_nonce = sha256_json(
        {
            "schema": RANK_SCHEMA,
            "preregistration_sha256": preregistration_sha256,
            "rank": rank,
            "mutant_id": mutant_id,
            "lane": lane,
        }
    )
    freshness = {
        "case_nonce_sha256": case_nonce,
        "fresh_persistent_cache": True,
        "fresh_request_cache_group": True,
        "prior_case_state_reused": False,
    }
    persistent = group = persistent_guard = None
    ledgers: list[MultiForkHitLedger | None] = []
    backends: list[str] = []
    active_mutation: ActiveMutation | None = None
    target_restoration: dict[str, Any] | None = None
    outputs: list[dict[str, Any]] = []
    step_receipts: list[dict[str, Any]] = []
    probe_receipts: list[dict[str, Any]] = []
    teacher_forcing: dict[str, Any] | None = None
    teacher_token: torch.Tensor | None = None
    exception_result: dict[str, Any] | None = None
    cleanup_errors: list[str] = []
    policy = SelectiveGatePolicy(expected_gate if activate_mutation else None)
    original_attn = runtime.backbone.config._attn_implementation

    def append_stage(
        stage: str, request_index: int, input_tokens: torch.Tensor
    ) -> None:
        logits, token, step = _model_step(
            runtime,
            group,
            backends[request_index],
            request_index,
            input_tokens,
        )
        require(stage not in {row["stage"] for row in outputs}, "duplicate output stage")
        sidecar = _persist_stage_sidecar(
            case_id=case_id,
            stage=stage,
            logits=logits,
            sidecar_root=sidecar_root,
            rank_root=rank_root,
        )
        outputs.append({"stage": stage, "token": token, "sidecar": sidecar})
        step_receipts.append({"stage": stage, **step})

    try:
        persistent, group, persistent_guard = _build_group(runtime)
        ledger_values, backends = _make_ledgers(
            runtime,
            group,
            calls=_call_budget(mutant_id),
            mutant_id=mutant_id,
            activate_mutation=activate_mutation,
        )
        ledgers = list(ledger_values)
        if mutant_id in TEACHER_TOKEN_RULE:
            teacher_token, teacher_forcing = _teacher_token(runtime, mutant_id)
        if activate_mutation and mutant_id not in ("M4", "M5", "M6", "M7"):
            active_mutation = _activate_initial_mutation(
                mutant_id, runtime=runtime, group=group, ledgers=ledgers
            )
            require(active_mutation is not None, f"{mutant_id} target not activated")
        with policy:
            if mutant_id == "M1":
                probe = resident.validate_runtime_kv_ownership(
                    persistent,
                    group,
                    runtime.plan,
                    require_appended_tail_cow=False,
                )
                probe_receipts.append(
                    {"kind": "live-kv-ownership", "receipt": probe}
                )
                append_stage("measured", 1, runtime.queries[1])
            elif mutant_id == "M2":
                append_stage("measured", 0, runtime.queries[0])
            elif mutant_id == "M3":
                append_stage("prefix-r0", 0, runtime.queries[0])
                append_stage("prefix-r1", 1, runtime.queries[1])
                probe = resident.validate_runtime_kv_ownership(
                    persistent,
                    group,
                    runtime.plan,
                    require_appended_tail_cow=True,
                )
                probe_receipts.append(
                    {"kind": "live-kv-ownership-after-prefix", "receipt": probe}
                )
                require(teacher_token is not None, "M3 teacher token missing")
                append_stage("continuation-r0", 0, teacher_token)
            elif mutant_id in ("M4", "M5"):
                completed = [0] if mutant_id == "M4" else [0, 1]
                append_stage("prefix-r0", 0, runtime.queries[0])
                if mutant_id == "M5":
                    append_stage("prefix-r1", 1, runtime.queries[1])
                if activate_mutation:
                    active_mutation = _activate_gdn_mutation(
                        mutant_id,
                        runtime=runtime,
                        persistent=persistent,
                        group=group,
                    )
                snapshot = storage_witness.capture_gdn_storage_snapshot(
                    persistent,
                    group.requests,
                    runtime.plan.linear_layer_indices,
                    phase=storage_witness.PHASE_POST_TRANSITION,
                    policy=GDN_BORROW_IMMUTABLE_BASE,
                    persistent_guard=persistent_guard,
                    completed_request_indices=completed,
                )
                replay_input = json.loads(json.dumps(snapshot))
                replay = storage_witness.replay_gdn_storage_witness(replay_input)
                probe_receipts.append(
                    {
                        "kind": "live-gdn-storage-replay",
                        "completed_request_indices": completed,
                        "storage_witness": snapshot,
                        "storage_witness_sha256": sha256_json(snapshot),
                        "replay_receipt": replay,
                    }
                )
                require(teacher_token is not None, "GDN teacher token missing")
                request_index = TARGET_REQUEST[mutant_id]
                append_stage(
                    f"continuation-r{request_index}", request_index, teacher_token
                )
            else:
                append_stage(
                    "measured",
                    TARGET_REQUEST[mutant_id],
                    runtime.queries[TARGET_REQUEST[mutant_id]],
                )
        if tuple(row["stage"] for row in outputs) != EXPECTED_CLEAN_STAGES[mutant_id]:
            raise MatrixV2Error(f"{mutant_id} completed stage horizon drift")
        for ledger in ledgers:
            if ledger is not None:
                require(ledger.verify_complete()["verified"] is True, "ledger incomplete")
    except BaseException as exc:
        exception_result = _classify_exception(exc, mutant_id)
        tb = exc.__traceback__
        if tb is not None:
            traceback.clear_frames(tb)
        exc.__traceback__ = None
    finally:
        if active_mutation is not None:
            try:
                target_restoration = active_mutation.restore()
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    f"target restoration: {type(cleanup_error).__name__}: {cleanup_error}"
                )
        if activate_mutation and mutant_id in ("M6", "M7") and ledgers:
            injected = ledgers[0]
            candidate = getattr(injected, "injector_target_restoration", None)
            if isinstance(candidate, dict):
                target_restoration = candidate
        if backends:
            try:
                rr2._unregister_backends([name for name in backends if name])
            except BaseException as cleanup_error:
                cleanup_errors.append(
                    f"backend unregister: {type(cleanup_error).__name__}: {cleanup_error}"
                )
        if runtime.backbone.config._attn_implementation != original_attn:
            runtime.backbone.config._attn_implementation = original_attn
            cleanup_errors.append("_attn_implementation leaked outside model step")

    try:
        policy_receipt = policy.receipt()
    except BaseException as receipt_error:
        policy_receipt = {
            "schema_version": POLICY_SCHEMA,
            "target_gate_id": expected_gate if activate_mutation else None,
            "lane": "target-only-suppressed" if activate_mutation else "all-gates-on",
            "suppressed_event_count": 0,
            "suppressed_gate_ids": [],
            "events": [],
            "scope_integrity_before_restore": False,
            "all_original_function_identities_restored": False,
            "receipt_error": f"{type(receipt_error).__name__}: {receipt_error}",
        }
        cleanup_errors.append("suppression policy did not produce a valid receipt")

    if activate_mutation:
        if target_restoration is None:
            cleanup_errors.append("mutant target restoration receipt missing")
        if policy_receipt.get("suppressed_event_count", 0) < 1:
            cleanup_errors.append("mutant did not reach and suppress its target gate")
        if set(policy_receipt.get("suppressed_gate_ids", [])) != {expected_gate}:
            cleanup_errors.append("suppression receipt contains wrong gate")
        if (
            exception_result is not None
            and exception_result.get("classification") == "other_forkaudit_gate"
            and exception_result.get("other_gate_id") == expected_gate
        ):
            cleanup_errors.append("target gate escaped the target-only policy")
    else:
        if policy_receipt.get("suppressed_event_count") != 0:
            cleanup_errors.append("clean all-gates-on case suppressed a gate")

    sidecars = [dict(row["sidecar"]) for row in outputs]
    observed_outputs = [
        {
            "stage": row["stage"],
            "token": int(row["token"]),
            "full_logit_sha256": sidecar["sha256"],
            "sidecar_relative_path": sidecar["relative_path"],
        }
        for row, sidecar in zip(outputs, sidecars)
    ]

    if exception_result is None:
        exception_result = {
            "completion_status": "completed",
            "classification": "completed_semantics",
            "other_gate_id": None,
            "valid_scientific_outcome": True,
            "exception_module": None,
            "exception_type": None,
            "message": None,
            "stack_provenance": None,
        }
    if cleanup_errors:
        exception_result = {
            **exception_result,
            "completion_status": "operational_invalid",
            "classification": "operational_invalid",
            "valid_scientific_outcome": False,
            "cleanup_errors": cleanup_errors,
        }

    classification = exception_result["classification"]
    if classification == "other_forkaudit_gate":
        other_gate = _detector_cell(
            "evaluated", True, id=exception_result["other_gate_id"]
        )
    elif classification == "operational_invalid":
        other_gate = _detector_cell("not_evaluated", None, id=None)
    else:
        other_gate = _detector_cell("evaluated", False, id=None)

    if classification == "production_assertion":
        stack = exception_result["stack_provenance"]
        production_assertion = _detector_cell(
            "evaluated",
            True,
            allowlist_id=exception_result["production_assertion_allowlist_id"],
            provenance={
                "exception_module": exception_result["exception_module"],
                "exception_type": exception_result["exception_type"],
                "exact_message": exception_result["message"],
                "stack_provenance": stack["frames"],
            },
        )
    elif classification == "completed_semantics":
        production_assertion = _detector_cell(
            "evaluated", False, allowlist_id=None, provenance=None
        )
    else:
        production_assertion = _detector_cell(
            "not_evaluated", None, allowlist_id=None, provenance=None
        )
    if classification == "completed_semantics":
        production_nonassertion = _detector_cell(
            "evaluated", False, provenance=None
        )
    else:
        production_nonassertion = _detector_cell(
            "not_evaluated", None, provenance=None
        )

    if activate_mutation and target_restoration is not None:
        mutation_receipt: dict[str, Any] = {
            "applied": True,
            "mutant_id": mutant_id,
            "target_contract": TARGET_CONTRACT[mutant_id],
            "pre_descriptor_sha256": target_restoration["pre_sha256"],
            "mutated_descriptor_sha256": target_restoration["mutated_sha256"],
        }
    else:
        mutation_receipt = {"applied": False, "mutant_id": mutant_id}
    if teacher_forcing is not None:
        mutation_receipt["teacher_forcing"] = {
            "request_index": teacher_forcing["request_index"],
            "query_token_index": teacher_forcing["query_token_index"],
            "token_id": teacher_forcing["token_id"],
            "source_token_sha256": teacher_forcing["token_sha256"],
            "source_coordinate": teacher_forcing["source_coordinate"],
            "independent_of_path_argmax": True,
            "argmax_feedback_used": False,
        }
    if mutant_id == "M8" and classification == "fault_payload_abort":
        mutation_receipt["fault_payload_abort_provenance"] = {
            "exception_type": "AssertionError",
            "exact_message": M8_SENTINEL_MESSAGE,
            "stack_provenance": [
                {
                    "file": "run_qcomem_qwen35_forkaudit_detector_matrix_v2.py",
                    "function": "m8_sentinel",
                }
            ],
        }
    if mutant_id == "M9" and ledgers:
        raw_receipt = getattr(ledgers[0], "raw_key_reached_production", None)
        if raw_receipt is not None:
            mutation_receipt["raw_tensor_production_receipt"] = raw_receipt

    if not activate_mutation:
        injector_restoration = {"status": "not_applicable", "verified": True}
    else:
        injector_restoration = target_restoration or {
            "status": "missing",
            "verified": False,
        }

    any_output = bool(observed_outputs)
    completed_horizon = bool(
        classification == "completed_semantics"
        and [row["stage"] for row in observed_outputs]
        == list(EXPECTED_CLEAN_STAGES[mutant_id])
    )
    if not activate_mutation and completed_horizon:
        initial_semantics = {
            "token_only": _detector_cell(
                "evaluated",
                False,
                exact_sha=True,
                argmax_equal=True,
                max_abs=None,
                relative_l2=None,
            ),
            "full_logit": _detector_cell(
                "evaluated",
                False,
                exact_sha=True,
                argmax_equal=True,
                max_abs=0.0,
                relative_l2=0.0,
            ),
        }
    else:
        initial_semantics = {
            "token_only": _detector_cell(
                "not_evaluated",
                None,
                exact_sha=None,
                argmax_equal=None,
                max_abs=None,
                relative_l2=None,
            ),
            "full_logit": _detector_cell(
                "not_evaluated",
                None,
                exact_sha=None,
                argmax_equal=None,
                max_abs=None,
                relative_l2=None,
            ),
        }
    outcome = {
        "completion_status": exception_result["completion_status"],
        "classification": classification,
        "exception": (
            None
            if exception_result.get("exception_type") is None
            else {
                key: exception_result.get(key)
                for key in (
                    "exception_module",
                    "exception_type",
                    "message",
                    "stack_provenance",
                )
            }
        ),
        "valid_scientific_outcome": exception_result["valid_scientific_outcome"],
        "fork_audit": {
            "target_suppression_events": policy_receipt.get("events", []),
            "other_gate": other_gate,
        },
        "production": {
            "assertion": production_assertion,
            "nonassertion_crash": production_nonassertion,
            "fault_payload_abort": _detector_cell(
                "evaluated" if classification in ("fault_payload_abort", "completed_semantics") else "not_evaluated",
                True if classification == "fault_payload_abort" else (False if classification == "completed_semantics" else None),
            ),
        },
        "output_availability": {
            "token": "evaluated" if completed_horizon else "not_evaluated",
            "full_logit": "evaluated" if completed_horizon else "not_evaluated",
            "sidecar": "evaluated" if any_output else "not_evaluated",
        },
        "observed_outputs": observed_outputs,
        "semantics": initial_semantics,
        "measured_non_forkaudit_escape": (
            None if not activate_mutation or completed_horizon else False
        ),
        "mutation_receipt": mutation_receipt,
        "injector_target_restoration": injector_restoration,
        "case_discard_allocator_recovery": None,
        "suppression_hook_restoration": policy_receipt,
        "backend_registry_restoration": {
            "verified": not any(
                error.startswith("backend unregister") for error in cleanup_errors
            )
        },
        "attention_backend_restoration": {
            "verified": runtime.backbone.config._attn_implementation == original_attn
        },
        "step_receipts": step_receipts,
        "probe_receipts": probe_receipts,
    }
    case = {
        "case_id": case_id,
        "mutant_id": mutant_id,
        "lane": lane,
        "expected_gate_id": expected_gate,
        "target_request": TARGET_REQUEST[mutant_id],
        "freshness_receipt": freshness,
        "outcome": outcome,
    }
    return case, sidecars


def _run_case_with_cleanup(
    runtime: Runtime,
    **kwargs: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    require(runtime.allocator_baseline is not None, "allocator baseline not frozen")
    before = _cleanup_allocator()
    require(before == runtime.allocator_baseline, "allocator baseline drift before case")
    case, sidecars = _run_case_impl(runtime, **kwargs)
    after = _cleanup_allocator()
    receipt = {
        "verified": after == runtime.allocator_baseline,
        "before_cell": dict(before),
        "after_cleanup": dict(after),
        "frozen_model_query_baseline": dict(runtime.allocator_baseline),
        "gc_collect_completed": True,
        "cuda_empty_cache_completed": True,
        "cuda_synchronize_completed": True,
        "current_allocated_and_reserved_exactly_recovered": (
            after == runtime.allocator_baseline
        ),
        "disposable_resident_request_group_discarded": True,
        "registered_attention_backends_removed": case["outcome"][
            "backend_registry_restoration"
        ]["verified"],
        "attention_implementation_restored": case["outcome"][
            "attention_backend_restoration"
        ]["verified"],
        "traceback_references_cleared": True,
    }
    case["outcome"]["case_discard_allocator_recovery"] = receipt
    require(receipt["verified"], "allocator did not recover after case")
    return case, sidecars


def _read_sidecar(row: Mapping[str, Any], rank_root: Path) -> np.ndarray:
    path = rank_root / str(row["relative_path"])
    payload = path.read_bytes()
    require(len(payload) == row["bytes"] == 248320 * 4, "sidecar bytes")
    require(sha256_bytes(payload) == row["sha256"], "sidecar SHA")
    require(row["dtype"] == "float32" and row["shape"] == [1, 248320], "sidecar schema")
    return np.frombuffer(payload, dtype=np.float32).reshape(1, 248320)


def _attach_pair_semantics(
    clean: dict[str, Any],
    mutant: dict[str, Any],
    *,
    sidecars: Sequence[Mapping[str, Any]],
    rank_root: Path,
) -> None:
    clean_outcome = clean["outcome"]
    mutant_outcome = mutant["outcome"]
    if (
        clean_outcome["completion_status"] != "completed"
        or mutant_outcome["completion_status"] != "completed"
    ):
        mutant_outcome["measured_non_forkaudit_escape"] = (
            None
            if mutant_outcome["completion_status"] == "operational_invalid"
            else False
        )
        return
    clean_rows = clean_outcome["observed_outputs"]
    mutant_rows = mutant_outcome["observed_outputs"]
    expected = list(EXPECTED_CLEAN_STAGES[mutant["mutant_id"]])
    require([row["stage"] for row in clean_rows] == expected, "clean stage horizon")
    require([row["stage"] for row in mutant_rows] == expected, "mutant stage horizon")
    sidecar_map = {
        (str(row["case_id"]), str(row["stage"])): row for row in sidecars
    }
    require(len(sidecar_map) == len(sidecars), "duplicate sidecar binding")
    clean_tokens = [int(row["token"]) for row in clean_rows]
    mutant_tokens = [int(row["token"]) for row in mutant_rows]
    clean_digests = [str(row["full_logit_sha256"]) for row in clean_rows]
    mutant_digests = [str(row["full_logit_sha256"]) for row in mutant_rows]
    exact_logits = clean_digests == mutant_digests
    squared_difference = 0.0
    squared_reference = 0.0
    max_abs = 0.0
    clean_sidecar_argmax: list[int] = []
    mutant_sidecar_argmax: list[int] = []
    for clean_row, mutant_row in zip(clean_rows, mutant_rows):
        clean_array = _read_sidecar(
            sidecar_map[(clean["case_id"], clean_row["stage"])], rank_root
        ).astype(np.float64)
        mutant_array = _read_sidecar(
            sidecar_map[(mutant["case_id"], mutant_row["stage"])], rank_root
        ).astype(np.float64)
        difference = mutant_array - clean_array
        clean_sidecar_argmax.append(int(np.argmax(clean_array)))
        mutant_sidecar_argmax.append(int(np.argmax(mutant_array)))
        max_abs = max(max_abs, float(np.max(np.abs(difference))))
        squared_difference += float(np.sum(difference * difference))
        squared_reference += float(np.sum(clean_array * clean_array))
    require(clean_tokens == clean_sidecar_argmax, "clean token/logit sidecar binding")
    require(mutant_tokens == mutant_sidecar_argmax, "mutant token/logit sidecar binding")
    token_equal = clean_sidecar_argmax == mutant_sidecar_argmax
    relative_l2 = math.sqrt(squared_difference) / max(
        math.sqrt(squared_reference), 1e-30
    )
    mutant_outcome["semantics"] = {
        "token_only": _detector_cell(
            "evaluated",
            not token_equal,
            exact_sha=token_equal,
            argmax_equal=token_equal,
            max_abs=None,
            relative_l2=None,
            clean_tokens=clean_tokens,
            mutant_tokens=mutant_tokens,
        ),
        "full_logit": _detector_cell(
            "evaluated",
            not exact_logits,
            exact_sha=exact_logits,
            argmax_equal=token_equal,
            max_abs=max_abs,
            relative_l2=relative_l2,
            clean_sha256=clean_digests,
            mutant_sha256=mutant_digests,
        ),
    }
    other_gate = mutant_outcome["fork_audit"]["other_gate"]
    assertion = mutant_outcome["production"]["assertion"]
    nonassertion = mutant_outcome["production"]["nonassertion_crash"]
    mutant_outcome["measured_non_forkaudit_escape"] = bool(
        token_equal
        and exact_logits
        and other_gate["status"] == "evaluated"
        and other_gate["caught"] is False
        and assertion["status"] == "evaluated"
        and assertion["caught"] is False
        and nonassertion["status"] == "evaluated"
        and nonassertion["caught"] is False
    )


def run_rank(args: argparse.Namespace) -> dict[str, Any]:
    require(args.rank in MUTANT_ASSIGNMENT, "rank assignment missing")
    prereg_raw = check_file(
        args.preregistration,
        args.expected_preregistration_sha256,
        "preregistration",
    )
    prereg = json.loads(prereg_raw)
    require(prereg.get("schema_version") == PREREG_SCHEMA, "preregistration schema")
    require(
        prereg.get("assignment")
        == {str(rank): list(values) for rank, values in MUTANT_ASSIGNMENT.items()},
        "preregistration assignment drift",
    )
    input_binding = prereg.get("input_binding")
    source_binding = prereg.get("source_binding")
    require(isinstance(input_binding, dict), "prereg input binding")
    require(isinstance(source_binding, dict), "prereg source binding")
    check_file(
        args.original_receipt_manifest,
        input_binding["original_rr2_receipt_manifest_sha256"],
        "original RR2 receipt manifest",
    )
    check_file(
        args.code_ledger,
        args.expected_code_ledger_sha256,
        "code ledger",
    )
    check_file(
        args.imported_rr2_code_ledger,
        args.expected_imported_rr2_code_ledger_sha256,
        "imported RR2 code ledger",
    )
    check_file(
        args.external_pin_payload,
        args.expected_external_pin_payload_sha256,
        "external preexecution pin payload",
    )
    runner_sha = sha256_file(Path(__file__).resolve())
    policy_path = Path(__file__).with_name("qcomem_forkaudit_selective_gate_policy.py")
    policy_sha = sha256_file(policy_path)
    require(runner_sha == source_binding["runner_sha256"], "runner source drift")
    require(policy_sha == args.expected_gate_policy_sha256, "gate policy expected SHA")
    require(policy_sha == source_binding["gate_policy_sha256"], "gate policy source drift")
    require(
        args.expected_code_ledger_sha256 == input_binding["code_ledger_sha256"],
        "code ledger preregistration drift",
    )
    require(
        args.expected_imported_rr2_code_ledger_sha256
        == input_binding["imported_rr2_code_ledger_sha256"],
        "imported RR2 code ledger preregistration drift",
    )
    require(
        args.expected_external_pin_payload_sha256
        == input_binding["external_pin_payload_sha256"],
        "external pin preregistration drift",
    )
    require(
        args.expected_external_pin_payload_sha256
        == source_binding["external_pin_payload_sha256"],
        "external pin source-binding drift",
    )
    expected_cli_binding = {
        "model_revision": rr2.FORMAL_MODEL_REVISION,
        "weight_ledger_raw_sha256": args.expected_weight_ledger_sha256,
        "artifact_ledger_raw_sha256": args.expected_artifact_ledger_sha256,
        "pg19_sha256": args.expected_pg19_sha256,
        "pg19_manifest_sha256": args.expected_pg19_manifest_sha256,
        "windows_sha256": args.expected_windows_sha256,
        "frozen_query_banks_sha256": args.expected_query_banks_sha256,
    }
    for field, expected_value in expected_cli_binding.items():
        require(
            input_binding.get(field) == expected_value,
            f"{field} preregistration drift",
        )
    runtime = _load_runtime(args, prereg)
    warmup = _discarded_warmup(runtime)
    cases: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    with torch.inference_mode():
        for mutant_id in MUTANT_ASSIGNMENT[args.rank]:
            clean, clean_sidecars = _run_case_with_cleanup(
                runtime,
                rank=args.rank,
                mutant_id=mutant_id,
                lane="clean",
                preregistration_sha256=args.expected_preregistration_sha256,
                sidecar_root=args.sidecar_root,
                rank_root=args.rank_root,
            )
            mutant, mutant_sidecars = _run_case_with_cleanup(
                runtime,
                rank=args.rank,
                mutant_id=mutant_id,
                lane="target_suppressed",
                preregistration_sha256=args.expected_preregistration_sha256,
                sidecar_root=args.sidecar_root,
                rank_root=args.rank_root,
            )
            pair_sidecars = [*clean_sidecars, *mutant_sidecars]
            _attach_pair_semantics(
                clean,
                mutant,
                sidecars=pair_sidecars,
                rank_root=args.rank_root,
            )
            cases.extend((clean, mutant))
            sidecars.extend(pair_sidecars)
    expected_case_count = 2 * len(MUTANT_ASSIGNMENT[args.rank])
    require(len(cases) == expected_case_count, "rank case count")
    require(len({case["case_id"] for case in cases}) == len(cases), "case IDs unique")
    result = {
        "schema_version": RANK_SCHEMA,
        "rank": args.rank,
        "assigned_fault_ids": list(MUTANT_ASSIGNMENT[args.rank]),
        "hardware": runtime.hardware,
        "input_receipt": runtime.input_receipt,
        "discarded_prebaseline_warmup_receipt": warmup,
        "cases": cases,
        "sidecars": sidecars,
        "runner_sha256": runner_sha,
        "gate_policy_sha256": policy_sha,
    }
    write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--stage", choices=("rank",), required=True)
    result.add_argument("--rank", type=int, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--sidecar-root", type=Path, required=True)
    result.add_argument("--rank-root", type=Path, required=True)
    result.add_argument("--preregistration", type=Path, required=True)
    result.add_argument("--expected-preregistration-sha256", required=True)
    result.add_argument("--original-receipt-manifest", type=Path, required=True)
    result.add_argument("--code-ledger", type=Path, required=True)
    result.add_argument("--expected-code-ledger-sha256", required=True)
    result.add_argument("--imported-rr2-code-ledger", type=Path, required=True)
    result.add_argument(
        "--expected-imported-rr2-code-ledger-sha256", required=True
    )
    result.add_argument("--external-pin-payload", type=Path, required=True)
    result.add_argument("--expected-external-pin-payload-sha256", required=True)
    result.add_argument("--expected-gate-policy-sha256", required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--model-weight-ledger", type=Path, required=True)
    result.add_argument("--model-artifact-ledger", type=Path, required=True)
    result.add_argument("--expected-weight-ledger-sha256", required=True)
    result.add_argument("--expected-artifact-ledger-sha256", required=True)
    result.add_argument("--pg19-data", type=Path, required=True)
    result.add_argument("--pg19-manifest", type=Path, required=True)
    result.add_argument("--expected-pg19-sha256", required=True)
    result.add_argument("--expected-pg19-manifest-sha256", required=True)
    result.add_argument("--expected-windows-sha256", required=True)
    result.add_argument("--frozen-query-banks", type=Path, required=True)
    result.add_argument("--expected-query-banks-sha256", required=True)
    result.add_argument("--expected-gpu-uuid", required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_rank(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXPECTED_CLEAN_STAGES",
    "EXPECTED_GATES",
    "M8_SENTINEL_MESSAGE",
    "M7_PRODUCTION_ASSERTION_ALLOWLIST_ID",
    "M7_PRODUCTION_MESSAGE",
    "M9_PRODUCTION_MESSAGE",
    "MUTANT_ASSIGNMENT",
    "PRODUCTION_ASSERTION_ALLOWLIST_ID",
    "PREREG_SCHEMA",
    "RANK_SCHEMA",
    "TARGET_CONTRACT",
    "TARGET_REQUEST",
    "TEACHER_TOKEN_RULE",
    "ArgumentInjectingLedger",
    "M9RawTensorLedger",
    "MatrixV2Error",
    "_attach_pair_semantics",
    "_classify_exception",
    "_detector_cell",
    "_target_receipt",
    "m8_sentinel",
    "run_rank",
]
