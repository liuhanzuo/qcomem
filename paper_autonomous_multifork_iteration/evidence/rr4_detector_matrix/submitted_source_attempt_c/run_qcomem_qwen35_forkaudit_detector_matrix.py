from __future__ import annotations

"""RR4 detector-by-mutant measurement for the frozen RR2 live faults.

The RR2 campaign intentionally aborts at the first named ForkAudit gate.  That
is the right fail-closed behavior, but it leaves simpler output detectors
unobserved.  This follow-up suppresses *only* the preregistered target gate for
one mutant at a time, then records whether the same live fault is exposed by a
native/runtime failure, token divergence, full-logit divergence, a clean
cross-arm reference, or a clean cross-N reference.  The original RR2 raw
receipt remains the authority for the named ForkAudit detection.

No absent result is converted to a pass.  Unsupported or aborted comparisons
are serialized as ``not_evaluated`` with a reason.
"""

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import re
import subprocess
import traceback
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import torch

import qcomem_vllm_paged_multifork_resident as resident
import run_qcomem_qwen35_forkaudit_review_revision as rr2
from qcomem_forkaudit_mutants import MUTANT_IDS, MUTANT_SPECS
from qcomem_vllm_paged_fair_control import FRESH_CONTROL, SHARED_REUSE
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    GDN_MATERIALIZE_REQUEST_BASE,
    MultiForkHitLedger,
    RuntimeInvariantError,
    build_pg19_train_query_bank,
    build_resident_request_group,
    register_multifork_backend,
)


SCHEMA = "forkaudit-detector-matrix-rank-v1"
AGGREGATE_SCHEMA = "forkaudit-detector-matrix-aggregate-v1"
PREREG_SCHEMA = "forkaudit-detector-matrix-preregistration-v1"
MUTANT_ASSIGNMENT = {
    0: ("M1", "M9"),
    1: ("M2",),
    2: ("M3",),
    3: ("M4",),
    4: ("M5",),
    5: ("M6",),
    6: ("M7",),
    7: ("M8",),
}
FORKAUDIT_GATES = tuple(MUTANT_SPECS[mid].expected_gate_id for mid in MUTANT_IDS)
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
MEASURED_STEPS = {
    "M1": "one request-1 full-model step after reservation mutation",
    "M2": "one request-0 full-model step after ledger sequence mutation",
    "M3": "request-0, request-1, then request-0 continuation",
    "M4": "request-0 prefix then one request-0 continuation after alias",
    "M5": "request-0/request-1 prefixes then one request-1 continuation after alias",
    "M6": "one request-0 full-model step with first-layer post-RoPE position +1",
    "M7": "one request-0 full-model step with first-layer materialized all-true mask",
    "M8": "one request-0 full-model step after callable swap",
    "M9": "one request-0 full-model step after first-layer dense-key substitution",
}


class MatrixError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(canonical_bytes(value) + b"\n")
    pending.replace(path)


def check_file(path: Path, expected: str, label: str) -> bytes:
    require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, f"{label} expected SHA")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected, f"{label} SHA drift")
    return payload


def tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_sha(tensor: torch.Tensor) -> str:
    return sha256_bytes(tensor_bytes(tensor))


@dataclass
class Runtime:
    model: Any
    backbone: Any
    plan: Any
    kernel: Any
    document: torch.Tensor
    queries: tuple[torch.Tensor, ...]
    input_receipt: dict[str, Any]
    hardware_receipt: dict[str, Any]


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
        "torch_cuda": str(torch.version.cuda),
        "torch_version": str(torch.__version__),
    }


def _load_runtime(args: argparse.Namespace) -> Runtime:
    """Rebuild RR2 rank inputs, then load the same frozen local model."""

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
    query_hashes = [tensor_sha(query) for query in queries]
    require(
        query_hashes == [row["query_token_ids_sha256"] for row in bank["rows"]],
        "query digest drift",
    )
    require(
        [int(row["source_token_offset"]) for row in query_audit["rows"]]
        == [int(row["source_token_offset"]) for row in bank["rows"]],
        "query coordinates drift",
    )
    hardware = _gpu_receipt(args.expected_gpu_uuid)
    # The SHA ledgers were checked above; replay their file bindings before load.
    weight_rows = rr2._parse_sha256_ledger(
        args.model_weight_ledger.read_bytes(), label="matrix model weight ledger"
    )
    artifact_rows = rr2._parse_sha256_ledger(
        args.model_artifact_ledger.read_bytes(), label="matrix model artifact ledger"
    )
    rr2._verify_weight_ledger_structure(weight_rows, model_dir=args.model_dir)
    rr2._verify_model_ledger(
        artifact_rows, model_dir=args.model_dir, label="matrix model artifact ledger"
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
    # Match the frozen RR2 runtime: from_pretrained builds the model on CPU,
    # then the complete text model and the rebuilt token tensors move to the
    # rank-local GPU before any cache construction or forward pass.
    model = model.to(device="cuda:0", dtype=torch.bfloat16)
    backbone = _resolve_backbone(model)
    kernel = _resolve_vllm_unified_attention()
    document = document_cpu.to(device="cuda:0", non_blocking=False)
    live_queries = tuple(query.to(device="cuda:0", non_blocking=False) for query in queries)
    return Runtime(
        model=model,
        backbone=backbone,
        plan=plan,
        kernel=kernel,
        document=document,
        queries=live_queries,
        input_receipt={
            "rank": args.rank,
            "windows_sha256": windows_sha,
            "document_token_ids_sha256": tensor_sha(document),
            "query_token_ids_sha256": [tensor_sha(query) for query in live_queries],
            "query_bank_manifest_sha256": bank["manifest_sha256"],
            "weight_ledger_raw_sha256": args.expected_weight_ledger_sha256,
            "artifact_ledger_raw_sha256": args.expected_artifact_ledger_sha256,
        },
        hardware_receipt=hardware,
    )


def _cleanup() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _build_group(
    runtime: Runtime, *, resident_count: int, kv_policy: str, gdn_policy: str
) -> tuple[Any, Any]:
    persistent, _conversion = rr2._convert_persistent(
        runtime.backbone, runtime.plan, runtime.document, resident_count=resident_count
    )
    group = build_resident_request_group(
        persistent,
        runtime.plan,
        resident_count=resident_count,
        policy=kv_policy,
        gdn_base_policy=gdn_policy,
    )
    from run_qcomem_qwen35_vllm_paged_multifork_resident import _set_production_no_mask

    _set_production_no_mask(group, runtime.plan.full_attention_layer_indices)
    return persistent, group


class InjectingLedger(MultiForkHitLedger):
    """The production ledger with one preregistered M6/M7 argument mutation."""

    def __init__(self, *args: Any, matrix_mutant: str | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.matrix_mutant = matrix_mutant
        self.matrix_injection_count = 0

    def attention_forward(self, module: Any, query: torch.Tensor, key: Any, value: Any,
                          attention_mask: Any, *args: Any, **kwargs: Any) -> Any:
        layer_index = getattr(module, "layer_idx", None)
        if layer_index == self.indices[0] and self.matrix_injection_count == 0:
            if self.matrix_mutant == "M6":
                position_ids = kwargs.get("position_ids")
                require(isinstance(position_ids, torch.Tensor), "M6 position IDs missing")
                kwargs["position_ids"] = position_ids.detach().clone() + 1
                self.strict_position_values = False
                self.matrix_injection_count += 1
            elif self.matrix_mutant == "M7":
                total_length = int(key.sequence.sequence_length)
                attention_mask = torch.ones(
                    (1, 1, int(query.shape[-2]), total_length),
                    dtype=torch.bool,
                    device=query.device,
                )
                self.matrix_injection_count += 1
        return super().attention_forward(
            module, query, key, value, attention_mask, *args, **kwargs
        )


@contextlib.contextmanager
def suppress_gate(gate_id: str | None) -> Iterator[list[dict[str, str]]]:
    """Suppress exactly one runtime gate and retain every suppressed event."""

    events: list[dict[str, str]] = []
    original = resident._runtime_require

    def selective(condition: bool, observed_gate: str, message: str) -> None:
        if not condition and gate_id == observed_gate:
            events.append({"gate_id": observed_gate, "message": message})
            return
        original(condition, observed_gate, message)

    resident._runtime_require = selective
    try:
        yield events
    finally:
        resident._runtime_require = original


def _make_ledgers(
    runtime: Runtime,
    group: Any,
    *,
    calls: Mapping[int, int],
    active_mutant: str | None,
) -> tuple[list[MultiForkHitLedger], list[str]]:
    ledgers: list[MultiForkHitLedger] = []
    backends: list[str] = []
    for request_index in range(group.resident_count):
        expected = int(calls.get(request_index, 1))
        ledger_cls = InjectingLedger if request_index == TARGET_REQUEST.get(active_mutant or "") else MultiForkHitLedger
        kwargs: dict[str, Any] = {}
        if ledger_cls is InjectingLedger:
            kwargs["matrix_mutant"] = active_mutant if active_mutant in ("M6", "M7") else None
        ledger = ledger_cls(
            runtime.plan,
            group.requests[request_index],
            request_index=request_index,
            resident_count=group.resident_count,
            request_policy=group.policy,
            expected_calls_per_layer=expected,
            initial_query_tokens=rr2.FORMAL_QUERY_TOKENS,
            kernel=runtime.kernel,
            strict_position_values=True,
            **kwargs,
        )
        ledgers.append(ledger)
        backends.append(register_multifork_backend(ledger))
    return ledgers, backends


def _model_step(runtime: Runtime, group: Any, backend: str, request_index: int,
                tokens: torch.Tensor) -> tuple[torch.Tensor, int]:
    original = runtime.backbone.config._attn_implementation
    try:
        runtime.backbone.config._attn_implementation = backend
        output = runtime.backbone(
            input_ids=tokens,
            past_key_values=group.requests[request_index],
            use_cache=True,
        )
        logits = rr2._last_logits(runtime.model, output).detach().cpu().float().contiguous()
        require(bool(torch.isfinite(logits).all()), "non-finite logits")
        token = int(logits.argmax(dim=-1).item())
        return logits, token
    finally:
        runtime.backbone.config._attn_implementation = original


def _sidecar(path: Path, logits: Sequence[torch.Tensor]) -> list[dict[str, Any]]:
    rows = []
    path.mkdir(parents=True, exist_ok=True)
    for index, tensor in enumerate(logits):
        payload = tensor.numpy().tobytes(order="C")
        target = path / f"step-{index}.fp32.bin"
        target.write_bytes(payload)
        rows.append(
            {
                "relative_path": target.name,
                "bytes": len(payload),
                "sha256": sha256_bytes(payload),
                "dtype": "float32",
                "shape": list(tensor.shape),
            }
        )
    return rows


def _scenario(
    runtime: Runtime,
    *,
    mutant_id: str,
    activate: bool,
    resident_count: int,
    kv_policy: str,
    gdn_policy: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run one fresh disposable case and serialize observations."""

    require(mutant_id in MUTANT_IDS, "unknown mutant")
    target_request = TARGET_REQUEST[mutant_id]
    if target_request >= resident_count:
        return {
            "status": "not_evaluated",
            "reason": f"target request {target_request} is unavailable at N={resident_count}",
            "resident_count": resident_count,
            "kv_policy": kv_policy,
            "gdn_policy": gdn_policy,
        }
    persistent = group = None
    backends: list[str] = []
    restoration_verified: bool | None = None
    suppressed: list[dict[str, str]] = []
    active_suppression_events: list[dict[str, str]] = []
    logits_rows: list[torch.Tensor] = []
    tokens: list[int] = []
    undo_func = lambda: None
    verify_func = lambda: True
    mutation_is_live = False
    mutation_receipt: dict[str, Any] = {"activated": activate, "target": mutant_id}
    expected_gate = MUTANT_SPECS[mutant_id].expected_gate_id if activate else None
    try:
        persistent, group = _build_group(
            runtime,
            resident_count=resident_count,
            kv_policy=kv_policy,
            gdn_policy=gdn_policy,
        )
        if mutant_id == "M3":
            calls = {0: 2, 1: 1}
        elif mutant_id == "M4":
            calls = {0: 2}
        elif mutant_id == "M5":
            calls = {0: 1, 1: 2}
        else:
            calls = {target_request: 1}
        ledgers, backends = _make_ledgers(
            runtime, group, calls=calls, active_mutant=mutant_id if activate else None
        )
        first_full = int(runtime.plan.full_attention_layer_indices[0])
        first_linear = int(runtime.plan.linear_layer_indices[0])
        if mutant_id == "M1" and activate:
            left = group.requests[0].layers[first_full].sequence
            right = group.requests[1].layers[first_full].sequence
            saved = right.reservations.detach().clone()
            replacement = left.reservations.detach().clone()
            require(not torch.equal(saved, replacement), "M1 no-op")
            right.reservations.copy_(replacement)
            undo_func = lambda: right.reservations.copy_(saved)
            verify_func = lambda: bool(torch.equal(right.reservations, saved))
            mutation_is_live = True
            mutation_receipt["mutated_tensor_sha256"] = tensor_sha(right.reservations)
        elif mutant_id == "M2" and activate:
            ledger = ledgers[0]
            original = ledger.sequence_ids[first_full]
            peer = id(group.requests[1].layers[first_full].sequence)
            require(original != peer, "M2 no-op")
            ledger.sequence_ids[first_full] = peer
            undo_func = lambda: ledger.sequence_ids.__setitem__(first_full, original)
            verify_func = lambda: ledger.sequence_ids[first_full] == original
            mutation_is_live = True
            mutation_receipt["peer_binding_installed"] = True
        elif mutant_id == "M3" and activate:
            sequence = group.requests[0].layers[first_full].sequence
            require("_detach_partial_document_tail" not in vars(sequence), "M3 precondition")
            sequence._detach_partial_document_tail = types.MethodType(  # type: ignore[method-assign]
                lambda _self, _batch: None, sequence
            )
            undo_func = lambda: vars(sequence).pop("_detach_partial_document_tail")
            verify_func = lambda: "_detach_partial_document_tail" not in vars(sequence)
            mutation_is_live = True
            mutation_receipt["tail_detach_override"] = "no-op"
        elif mutant_id == "M8" and activate:
            ledger = ledgers[0]
            original_kernel = ledger.kernel

            def sentinel(*_args: Any, **_kwargs: Any) -> Any:
                raise AssertionError("matrix M8 sentinel executed")

            ledger.kernel = sentinel
            undo_func = lambda: setattr(ledger, "kernel", original_kernel)
            verify_func = lambda: ledger.kernel is original_kernel
            mutation_is_live = True
            mutation_receipt["sentinel_installed"] = True
        elif mutant_id == "M9" and activate:
            from qcomem_vllm_paged_kernel import Q16KernelPagedTensorView

            layer = group.requests[0].layers[first_full]
            sequence = layer.sequence
            original_view = layer.keys
            require(isinstance(original_view, Q16KernelPagedTensorView), "M9 precondition")
            dense_batches = []
            for table_row in sequence.active_block_table.to(torch.int64):
                blocks = sequence.arena.key_cache.index_select(0, table_row)
                logical = blocks.reshape(-1, rr2.FORMAL_NUM_KV_HEADS, rr2.FORMAL_HEAD_DIM)[
                    : sequence.sequence_length
                ]
                dense_batches.append(logical.permute(1, 0, 2).contiguous())
            layer.keys = torch.stack(dense_batches, dim=0)
            undo_func = lambda: setattr(layer, "keys", original_view)
            verify_func = lambda: layer.keys is original_view
            mutation_is_live = True
            mutation_receipt["dense_key_sha256"] = tensor_sha(layer.keys)

        with suppress_gate(expected_gate) as active_suppression_events:
            if mutant_id == "M3":
                l0, t0 = _model_step(runtime, group, backends[0], 0, runtime.queries[0])
                l1, t1 = _model_step(runtime, group, backends[1], 1, runtime.queries[1])
                continuation = torch.tensor([[t0]], dtype=torch.long, device="cuda:0")
                l2, t2 = _model_step(runtime, group, backends[0], 0, continuation)
                logits_rows.extend((l0, l1, l2))
                tokens.extend((t0, t1, t2))
            elif mutant_id in ("M4", "M5"):
                l0, t0 = _model_step(runtime, group, backends[0], 0, runtime.queries[0])
                prefix = [(l0, t0)]
                if mutant_id == "M5":
                    l1, t1 = _model_step(runtime, group, backends[1], 1, runtime.queries[1])
                    prefix.append((l1, t1))
                if activate:
                    request_index = 0 if mutant_id == "M4" else 1
                    values = group.requests[request_index].layers[first_linear].conv_states
                    state_index = sorted(values)[0]
                    original_state = values[state_index]
                    alias = (
                        persistent.layers[first_linear].conv_states[state_index]
                        if mutant_id == "M4"
                        else group.requests[0].layers[first_linear].conv_states[state_index]
                    )
                    require(original_state is not alias, f"{mutant_id} no-op")
                    values[state_index] = alias
                    undo_func = lambda: values.__setitem__(state_index, original_state)
                    verify_func = lambda: values[state_index] is original_state
                    mutation_is_live = True
                    mutation_receipt["gdn_alias_installed"] = True
                request_index = 0 if mutant_id == "M4" else 1
                prefix_token = prefix[-1][1]
                continuation = torch.tensor([[prefix_token]], dtype=torch.long, device="cuda:0")
                measured, token = _model_step(
                    runtime, group, backends[request_index], request_index, continuation
                )
                logits_rows.append(measured)
                tokens.append(token)
            else:
                measured, token = _model_step(
                    runtime,
                    group,
                    backends[target_request],
                    target_request,
                    runtime.queries[target_request],
                )
                logits_rows.append(measured)
                tokens.append(token)
            suppressed.extend(active_suppression_events)
        undo_func()
        mutation_is_live = False
        restoration_verified = bool(verify_func())
        require(restoration_verified, f"{mutant_id} restoration failed")
        sidecars = _sidecar(output_dir, logits_rows)
        return {
            "status": "completed",
            "resident_count": resident_count,
            "kv_policy": kv_policy,
            "gdn_policy": gdn_policy,
            "tokens": tokens,
            "full_logit_sha256": [row["sha256"] for row in sidecars],
            "logit_sidecars": sidecars,
            "suppressed_target_gate_events": suppressed,
            "mutation_receipt": mutation_receipt,
            "restoration_verified": restoration_verified,
        }
    except BaseException as exc:
        if active_suppression_events:
            suppressed.extend(active_suppression_events)
        if mutation_is_live:
            try:
                undo_func()
                mutation_is_live = False
                restoration_verified = bool(verify_func())
            except BaseException:
                restoration_verified = False
        gate_id = exc.gate_id if isinstance(exc, RuntimeInvariantError) else None
        return {
            "status": "runtime_abort",
            "resident_count": resident_count,
            "kv_policy": kv_policy,
            "gdn_policy": gdn_policy,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "forkaudit_gate_id": gate_id,
            "traceback": traceback.format_exc(),
            "suppressed_target_gate_events": suppressed,
            "mutation_receipt": mutation_receipt,
            "restoration_verified": restoration_verified,
        }
    finally:
        if mutation_is_live:
            try:
                undo_func()
                restoration_verified = bool(verify_func())
            except BaseException:
                restoration_verified = False
        if backends:
            try:
                rr2._unregister_backends(backends)
            except BaseException:
                pass
        persistent = group = None
        _cleanup()


def _compare(left: Mapping[str, Any], right: Mapping[str, Any], *, detector: str) -> dict[str, Any]:
    if left.get("status") != "completed" or right.get("status") != "completed":
        return {
            "status": "not_evaluated",
            "caught": None,
            "reason": f"comparison unavailable: {left.get('status')} vs {right.get('status')}",
        }
    field = "tokens" if detector == "token_only" else "full_logit_sha256"
    differs = left.get(field) != right.get(field)
    return {
        "status": "evaluated",
        "caught": differs,
        "relation": "different" if differs else "exact_equal",
        "left": left.get(field),
        "right": right.get(field),
    }


def run_rank(args: argparse.Namespace) -> dict[str, Any]:
    require(args.rank in MUTANT_ASSIGNMENT, "rank assignment missing")
    prereg_raw = check_file(args.preregistration, args.expected_preregistration_sha256, "preregistration")
    prereg = json.loads(prereg_raw)
    require(prereg.get("schema_version") == PREREG_SCHEMA, "preregistration schema")
    require(prereg.get("mutant_assignment") == {str(k): list(v) for k, v in MUTANT_ASSIGNMENT.items()}, "assignment drift")
    runtime = _load_runtime(args)
    rows = []
    with torch.inference_mode():
        for mutant_id in MUTANT_ASSIGNMENT[args.rank]:
            root = args.output.parent / f"rank-{args.rank}-sidecars" / mutant_id
            matched = _scenario(
                runtime,
                mutant_id=mutant_id,
                activate=False,
                resident_count=2,
                kv_policy=SHARED_REUSE,
                gdn_policy=GDN_BORROW_IMMUTABLE_BASE,
                output_dir=root / "matched-clean",
            )
            mutant = _scenario(
                runtime,
                mutant_id=mutant_id,
                activate=True,
                resident_count=2,
                kv_policy=SHARED_REUSE,
                gdn_policy=GDN_BORROW_IMMUTABLE_BASE,
                output_dir=root / "mutant",
            )
            cross_arm = _scenario(
                runtime,
                mutant_id=mutant_id,
                activate=False,
                resident_count=2,
                kv_policy=FRESH_CONTROL,
                gdn_policy=GDN_MATERIALIZE_REQUEST_BASE,
                output_dir=root / "cross-arm-clean",
            )
            cross_n = _scenario(
                runtime,
                mutant_id=mutant_id,
                activate=False,
                resident_count=1,
                kv_policy=SHARED_REUSE,
                gdn_policy=GDN_BORROW_IMMUTABLE_BASE,
                output_dir=root / "cross-n-clean",
            )
            runtime_abort = mutant.get("status") == "runtime_abort"
            other_gate = mutant.get("forkaudit_gate_id")
            token = _compare(mutant, matched, detector="token_only")
            logits = _compare(mutant, matched, detector="full_logit")
            arm = _compare(mutant, cross_arm, detector="full_logit")
            n_cmp = _compare(mutant, cross_n, detector="full_logit")
            if logits["status"] == "evaluated":
                preservation = (
                    "output_changed_within_measured_horizon"
                    if logits["caught"]
                    else "output_preserved_within_measured_horizon"
                )
            else:
                preservation = "not_observable_due_to_abort_or_missing_comparator"
            gate_matrix = {
                gate: (
                    "caught_in_original_rr2_receipt"
                    if gate == MUTANT_SPECS[mutant_id].expected_gate_id
                    else (
                        "caught_after_target_gate_suppression"
                        if gate == other_gate
                        else "not_separately_evaluated"
                    )
                )
                for gate in FORKAUDIT_GATES
            }
            rows.append(
                {
                    "mutant_id": mutant_id,
                    "mutant_name": MUTANT_SPECS[mutant_id].short_name,
                    "expected_gate_id": MUTANT_SPECS[mutant_id].expected_gate_id,
                    "target_request": TARGET_REQUEST[mutant_id],
                    "measured_steps": MEASURED_STEPS[mutant_id],
                    "matched_clean": matched,
                    "mutant_target_gate_suppressed": mutant,
                    "cross_arm_clean_reference": cross_arm,
                    "cross_n_clean_reference": cross_n,
                    "detectors": {
                        "token_only": token,
                        "full_logit": logits,
                        "cross_arm": arm,
                        "cross_n": n_cmp,
                        "existing_runtime_assertions": {
                            "status": "evaluated",
                            "caught": runtime_abort and other_gate is None,
                            "abort_type": mutant.get("error_type") if runtime_abort else None,
                            "abort_message": mutant.get("error_message") if runtime_abort else None,
                        },
                        "other_forkaudit_gate": {
                            "caught": other_gate is not None,
                            "gate_id": other_gate,
                        },
                        "forkaudit_gates": gate_matrix,
                    },
                    "output_preserving_status": preservation,
                }
            )
    result = {
        "schema_version": SCHEMA,
        "rank": args.rank,
        "preregistration_sha256": args.expected_preregistration_sha256,
        "original_rr2_run_id": prereg["original_rr2_run_id"],
        "hardware": runtime.hardware_receipt,
        "input_receipt": runtime.input_receipt,
        "rows": rows,
    }
    write_json(args.output, result)
    return result


def _original_rr2_receipts(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    manifest = load_json(args.original_receipt_manifest)
    require(manifest.get("schema_version") == "qcomem-forkaudit-detached-receipts-v1", "RR2 manifest schema")
    rows: dict[str, dict[str, Any]] = {}
    for shard_ref in manifest["shards"]:
        path = args.original_rr2_root / "raw" / shard_ref["relative_path"]
        payload = path.read_bytes()
        require(len(payload) == shard_ref["bytes"], "RR2 shard byte length")
        require(sha256_bytes(payload) == shard_ref["sha256"], "RR2 shard SHA")
        shard = json.loads(payload)
        for mid, case in shard["fault_campaign"]["mutants"].items():
            outcome = case["outcome"]
            rows[mid] = {
                "rank": shard["rank"],
                "shard_relative_path": shard_ref["relative_path"],
                "shard_sha256": shard_ref["sha256"],
                "classification": outcome["classification"],
                "expected_gate_id": outcome["expected_gate_id"],
                "observed_gate_id": outcome["observed_gate_id"],
                "restoration_verified": outcome["restoration_verified"],
                "matched_clean_classification": case["matched_clean"]["outcome"]["classification"],
            }
    require(set(rows) == set(MUTANT_IDS), "RR2 receipt mutant coverage")
    return rows


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    prereg_raw = check_file(args.preregistration, args.expected_preregistration_sha256, "preregistration")
    prereg = json.loads(prereg_raw)
    original = _original_rr2_receipts(args)
    matrix_rows = []
    rank_receipts = []
    for rank in range(8):
        path = args.rank_root / f"detector-matrix-rank-{rank}.json"
        payload = path.read_bytes()
        shard = json.loads(payload)
        require(shard.get("schema_version") == SCHEMA and shard.get("rank") == rank, "rank schema")
        require(shard.get("preregistration_sha256") == args.expected_preregistration_sha256, "rank prereg")
        rank_receipts.append({"rank": rank, "relative_path": path.name, "bytes": len(payload), "sha256": sha256_bytes(payload)})
        for row in shard["rows"]:
            mid = row["mutant_id"]
            rr2_row = original[mid]
            require(rr2_row["classification"] == "detected_expected_gate", f"{mid} original classification")
            require(rr2_row["expected_gate_id"] == rr2_row["observed_gate_id"] == row["expected_gate_id"], f"{mid} gate binding")
            require(rr2_row["matched_clean_classification"] == "clean_pass", f"{mid} clean binding")
            row = dict(row)
            row["original_rr2_forkaudit_receipt"] = rr2_row
            matrix_rows.append(row)
    require([row["mutant_id"] for row in matrix_rows] == list(MUTANT_IDS), "aggregate order/coverage")
    summary = {
        "mutants": len(matrix_rows),
        "token_only_caught": sum(row["detectors"]["token_only"].get("caught") is True for row in matrix_rows),
        "full_logit_caught": sum(row["detectors"]["full_logit"].get("caught") is True for row in matrix_rows),
        "existing_runtime_caught": sum(row["detectors"]["existing_runtime_assertions"].get("caught") is True for row in matrix_rows),
        "output_preserved": sum(row["output_preserving_status"] == "output_preserved_within_measured_horizon" for row in matrix_rows),
        "output_changed": sum(row["output_preserving_status"] == "output_changed_within_measured_horizon" for row in matrix_rows),
        "output_unobservable": sum(row["output_preserving_status"].startswith("not_observable") for row in matrix_rows),
        "forkaudit_expected_gate_caught": sum(row["original_rr2_forkaudit_receipt"]["classification"] == "detected_expected_gate" for row in matrix_rows),
    }
    result = {
        "schema_version": AGGREGATE_SCHEMA,
        "preregistration": prereg,
        "preregistration_sha256": args.expected_preregistration_sha256,
        "rank_receipts": rank_receipts,
        "rows": matrix_rows,
        "summary": summary,
        "limitations": [
            "Output-preserving labels are bounded to the preregistered per-mutant measured steps.",
            "Non-target ForkAudit gates marked not_separately_evaluated are unknown, not passes.",
            "Cross-N is not applicable when a fault targets request 1 because N=1 has no homologous request.",
            "The original RR2 raw receipt, not this bypass run, is authoritative for named-gate detection.",
        ],
    }
    write_json(args.output, result)
    return result


def preregister(args: argparse.Namespace) -> dict[str, Any]:
    runner_sha = sha256_file(Path(__file__).resolve())
    original_manifest_sha = sha256_file(args.original_receipt_manifest)
    value = {
        "schema_version": PREREG_SCHEMA,
        "created_before_candidate_outputs": True,
        "original_rr2_run_id": args.original_rr2_run_id,
        "original_rr2_receipt_manifest_sha256": original_manifest_sha,
        "runner_sha256": runner_sha,
        "imported_rr2_code_ledger_sha256": args.imported_rr2_code_ledger_sha256,
        "mutant_assignment": {str(k): list(v) for k, v in MUTANT_ASSIGNMENT.items()},
        "mutant_ids": list(MUTANT_IDS),
        "expected_gate_ids": {mid: MUTANT_SPECS[mid].expected_gate_id for mid in MUTANT_IDS},
        "target_requests": TARGET_REQUEST,
        "measured_steps": MEASURED_STEPS,
        "mutant_cell": {"resident_count": 2, "kv_policy": SHARED_REUSE, "gdn_policy": GDN_BORROW_IMMUTABLE_BASE},
        "cross_arm_clean_reference": {"resident_count": 2, "kv_policy": FRESH_CONTROL, "gdn_policy": GDN_MATERIALIZE_REQUEST_BASE},
        "cross_n_clean_reference": {"resident_count": 1, "kv_policy": SHARED_REUSE, "gdn_policy": GDN_BORROW_IMMUTABLE_BASE},
        "target_gate_suppression_rule": "suppress exactly the mutant's preregistered gate; preserve every other runtime/ForkAudit failure",
        "detectors": ["token_only", "full_logit", "cross_arm", "cross_n", "existing_runtime_assertions", "each_named_forkaudit_gate"],
        "missingness_policy": "not_evaluated is never converted to pass or not_caught",
    }
    write_json(args.output, value)
    return value


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("preregister", "rank", "aggregate"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rank", type=int)
    p.add_argument("--preregistration", type=Path)
    p.add_argument("--expected-preregistration-sha256")
    p.add_argument("--original-rr2-run-id", default="372384bd37cf7640ca210537a4360e1a")
    p.add_argument("--original-rr2-root", type=Path)
    p.add_argument("--original-receipt-manifest", type=Path, required=True)
    p.add_argument("--imported-rr2-code-ledger-sha256", default="")
    p.add_argument("--rank-root", type=Path)
    p.add_argument("--model-dir", type=Path)
    p.add_argument("--model-weight-ledger", type=Path)
    p.add_argument("--model-artifact-ledger", type=Path)
    p.add_argument("--expected-weight-ledger-sha256")
    p.add_argument("--expected-artifact-ledger-sha256")
    p.add_argument("--pg19-data", type=Path)
    p.add_argument("--pg19-manifest", type=Path)
    p.add_argument("--expected-pg19-sha256")
    p.add_argument("--expected-pg19-manifest-sha256")
    p.add_argument("--expected-windows-sha256")
    p.add_argument("--frozen-query-banks", type=Path)
    p.add_argument("--expected-query-banks-sha256")
    p.add_argument("--expected-gpu-uuid")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.stage == "preregister":
        preregister(args)
    elif args.stage == "rank":
        require(args.rank is not None and args.preregistration is not None, "rank inputs")
        run_rank(args)
    else:
        require(args.rank_root is not None and args.original_rr2_root is not None, "aggregate inputs")
        aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
