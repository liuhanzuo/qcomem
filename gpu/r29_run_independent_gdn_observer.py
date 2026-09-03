from __future__ import annotations

"""Run a minimal live H20 comparison between two GDN storage producers.

The candidate ForkAudit producer is used only to emit the existing witness.
The independent observer computes its own live tensor/storage facts and verdict
without importing that producer.  Both observe the same N=2 cache object in a
fresh disposable cell at setup, after request 0, and after request 1.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import qcomem_forkaudit_storage_witness as candidate_storage
import qcomem_joint_policy as joint_policy
import qcomem_vllm_paged_multifork_resident as resident
import run_qcomem_qwen35_forkaudit_review_revision as rr2
from qcomem_vllm_paged_fair_control import SHARED_REUSE
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    GDN_MATERIALIZE_REQUEST_BASE,
    MultiForkHitLedger,
    build_pg19_train_query_bank,
    build_resident_request_group,
    register_multifork_backend,
)
from r29_independent_gdn_observer import (
    ObserverSession,
    PHASE_GENERATION,
    PHASE_SETUP,
    PHASE_TRANSITION,
    compare_candidate_snapshot,
    evaluate_lifecycle,
    evaluate_phase,
)
from run_qcomem_qwen35_vllm_paged_multifork_resident import (
    _last_logits,
    _set_production_no_mask,
)


RESULT_SCHEMA = "forkaudit-r29-independent-gdn-observer-result-v1"
PREREG_SCHEMA = "forkaudit-r29-independent-gdn-observer-preregistration-v1"
POLICIES = (
    ("shared-base", GDN_BORROW_IMMUTABLE_BASE),
    ("materialized", GDN_MATERIALIZE_REQUEST_BASE),
)


class R29ObserverRunError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise R29ObserverRunError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha(tensor: torch.Tensor) -> str:
    payload = (
        tensor.detach()
        .contiguous()
        .cpu()
        .view(torch.uint8)
        .numpy()
        .tobytes()
    )
    return sha256_bytes(payload)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(payload)
    pending.replace(path)


def check_file(path: Path, expected: str, label: str) -> bytes:
    require(re.fullmatch(r"[0-9a-f]{64}", expected or "") is not None, f"{label} SHA format")
    payload = path.read_bytes()
    require(sha256_bytes(payload) == expected, f"{label} SHA drift")
    return payload


def _gpu_receipt(expected_uuid: str) -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    require(visible == expected_uuid, "CUDA_VISIBLE_DEVICES/assignment mismatch")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "run requires one GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    capability = list(torch.cuda.get_device_capability(0))
    require(capability == [9, 0] and "H20" in properties.name, "assigned device is not H20 sm90")
    output = subprocess.run(
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
    columns = [item.strip() for item in output.split(",")]
    require(len(columns) == 3 and columns[0] == expected_uuid, "GPU UUID receipt drift")
    return {
        "uuid": columns[0],
        "name": columns[1],
        "memory_mib": int(columns[2]),
        "compute_capability": capability,
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
    }


@dataclass
class Runtime:
    model: Any
    backbone: Any
    plan: Any
    kernel: Any
    document: torch.Tensor
    queries: tuple[torch.Tensor, ...]
    hardware: dict[str, Any]
    input_receipt: dict[str, Any]


def _load_runtime(args: argparse.Namespace, prereg: Mapping[str, Any]) -> Runtime:
    import build_qcomem_forkaudit_rr2_input_manifest as rr2_builder
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

    require(not torch.cuda.is_initialized(), "input rebuild must precede CUDA initialization")
    pg19_raw = check_file(args.pg19_data, args.expected_pg19_sha256, "PG19 data")
    manifest_raw = check_file(
        args.pg19_manifest, args.expected_pg19_manifest_sha256, "PG19 manifest"
    )
    check_file(args.frozen_query_banks, args.expected_query_banks_sha256, "query banks")
    check_file(args.model_weight_ledger, args.expected_weight_ledger_sha256, "weight ledger")
    check_file(args.model_artifact_ledger, args.expected_artifact_ledger_sha256, "artifact ledger")
    banks = load_json(args.frozen_query_banks)
    require(isinstance(banks, list) and len(banks) == 8, "query-bank rank coverage drift")
    bank = banks[0]
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
    window = windows[0]
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
    require(tensor_sha(document_cpu) == bank["document_token_ids_sha256"], "document digest drift")
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
        args.model_weight_ledger.read_bytes(), label="R29 model weight ledger"
    )
    artifact_rows = rr2._parse_sha256_ledger(
        args.model_artifact_ledger.read_bytes(), label="R29 model artifact ledger"
    )
    rr2._verify_weight_ledger_structure(weight_rows, model_dir=args.model_dir)
    rr2._verify_model_ledger(
        artifact_rows, model_dir=args.model_dir, label="R29 model artifact ledger"
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
    require(tuple(plan.full_attention_layer_indices) == rr2.FORMAL_FULL_LAYERS, "full-layer plan drift")
    require(tuple(plan.linear_layer_indices) == rr2.FORMAL_LINEAR_LAYERS, "linear-layer plan drift")
    environment = audit_frozen_kernel_environment()
    require(environment.get("matches_frozen_environment") is True, "kernel environment drift")
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
        hardware=hardware,
        input_receipt={
            "rank": 0,
            "model_revision": rr2.FORMAL_MODEL_REVISION,
            "weight_ledger_raw_sha256": args.expected_weight_ledger_sha256,
            "artifact_ledger_raw_sha256": args.expected_artifact_ledger_sha256,
            "pg19_sha256": args.expected_pg19_sha256,
            "pg19_manifest_sha256": args.expected_pg19_manifest_sha256,
            "windows_sha256": windows_sha,
            "frozen_query_banks_sha256": args.expected_query_banks_sha256,
            "document_token_ids_sha256": tensor_sha(document),
            "query_token_ids_sha256": [tensor_sha(query) for query in live_queries[:2]],
            "original_rr2_run_id": prereg["input_binding"]["original_rr2_run_id"],
            "candidate_code_ledger_raw_sha256": args.expected_candidate_code_ledger_sha256,
        },
    )


def _make_backends(runtime: Runtime, group: Any) -> tuple[list[MultiForkHitLedger], list[str]]:
    ledgers: list[MultiForkHitLedger] = []
    backends: list[str] = []
    try:
        for request_index in range(2):
            ledger = MultiForkHitLedger(
                runtime.plan,
                group.requests[request_index],
                request_index=request_index,
                resident_count=2,
                request_policy=group.policy,
                expected_calls_per_layer=1,
                initial_query_tokens=rr2.FORMAL_QUERY_TOKENS,
                kernel=runtime.kernel,
                strict_position_values=True,
            )
            ledgers.append(ledger)
            backends.append(register_multifork_backend(ledger))
    except BaseException:
        rr2._unregister_backends(backends)
        raise
    return ledgers, backends


def _model_step(runtime: Runtime, group: Any, backend: str, request_index: int) -> dict[str, Any]:
    original = runtime.backbone.config._attn_implementation
    try:
        runtime.backbone.config._attn_implementation = backend
        output = runtime.backbone(
            input_ids=runtime.queries[request_index],
            past_key_values=group.requests[request_index],
            use_cache=True,
        )
        logits = _last_logits(runtime.model, output).detach().cpu().float().contiguous()
        require(tuple(logits.shape) == (1, 248320), "live logit shape drift")
        require(bool(torch.isfinite(logits).all()), "non-finite live logits")
        return {
            "request_index": request_index,
            "query_token_ids_sha256": tensor_sha(runtime.queries[request_index]),
            "generated_token_id": int(logits.argmax(dim=-1).item()),
            "full_logit_sha256": tensor_sha(logits),
        }
    finally:
        runtime.backbone.config._attn_implementation = original
        require(runtime.backbone.config._attn_implementation == original, "attention backend did not restore")


def _capture_phase(
    *,
    session: ObserverSession,
    persistent: Any,
    group: Any,
    runtime: Runtime,
    policy_name: str,
    candidate_policy: str,
    candidate_guard: Any,
    phase: str,
    completed: Sequence[int],
) -> dict[str, Any]:
    torch.cuda.synchronize()
    before = session.capture(
        persistent,
        group.requests,
        runtime.plan.linear_layer_indices,
        phase=phase,
        policy=policy_name,
        completed_request_indices=completed,
    )
    candidate = candidate_storage.capture_gdn_storage_snapshot(
        persistent,
        group.requests,
        runtime.plan.linear_layer_indices,
        phase=phase,
        policy=candidate_policy,
        persistent_guard=candidate_guard,
        completed_request_indices=completed,
    )
    after = session.capture(
        persistent,
        group.requests,
        runtime.plan.linear_layer_indices,
        phase=phase,
        policy=policy_name,
        completed_request_indices=completed,
    )
    require(before == after, "live state changed while candidate producer captured")
    independent_verdict = evaluate_phase(before)
    comparison = compare_candidate_snapshot(before, candidate)
    candidate_replay = candidate_storage.replay_gdn_storage_witness(
        json.loads(json.dumps(candidate))
    )
    require(independent_verdict["passed"] is True, "independent phase verdict failed")
    require(comparison["passed"] is True, "candidate and independent observer disagree")
    require(candidate_replay["passed"] is True, "candidate self-replay failed")
    return {
        "phase": phase,
        "completed_request_indices": list(completed),
        "independent_before_candidate_capture": before,
        "candidate_capture": candidate,
        "independent_after_candidate_capture": after,
        "candidate_capture_nonmutating": True,
        "independent_verdict": independent_verdict,
        "independent_candidate_comparison": comparison,
        "candidate_replay_diagnostic": candidate_replay,
    }


def _run_policy_cell(runtime: Runtime, policy_name: str, candidate_policy: str) -> dict[str, Any]:
    persistent = group = session = None
    backends: list[str] = []
    try:
        persistent, _conversion = rr2._convert_persistent(
            runtime.backbone, runtime.plan, runtime.document, resident_count=2
        )
        candidate_guard = candidate_storage.capture_persistent_gdn_guard(
            persistent, runtime.plan.linear_layer_indices
        )
        group = build_resident_request_group(
            persistent,
            runtime.plan,
            resident_count=2,
            policy=SHARED_REUSE,
            gdn_base_policy=candidate_policy,
        )
        _set_production_no_mask(group, runtime.plan.full_attention_layer_indices)
        session = ObserverSession()
        phases = [
            _capture_phase(
                session=session,
                persistent=persistent,
                group=group,
                runtime=runtime,
                policy_name=policy_name,
                candidate_policy=candidate_policy,
                candidate_guard=candidate_guard,
                phase=PHASE_SETUP,
                completed=[],
            )
        ]
        ledgers, backends = _make_backends(runtime, group)
        steps = [_model_step(runtime, group, backends[0], 0)]
        phases.append(
            _capture_phase(
                session=session,
                persistent=persistent,
                group=group,
                runtime=runtime,
                policy_name=policy_name,
                candidate_policy=candidate_policy,
                candidate_guard=candidate_guard,
                phase=PHASE_TRANSITION,
                completed=[0],
            )
        )
        steps.append(_model_step(runtime, group, backends[1], 1))
        phases.append(
            _capture_phase(
                session=session,
                persistent=persistent,
                group=group,
                runtime=runtime,
                policy_name=policy_name,
                candidate_policy=candidate_policy,
                candidate_guard=candidate_guard,
                phase=PHASE_GENERATION,
                completed=[0, 1],
            )
        )
        ledger_receipts = [
            rr2._pointer_free_kernel_ledger(ledger.verify_complete())
            for ledger in ledgers
        ]
        lifecycle = evaluate_lifecycle(
            [row["independent_before_candidate_capture"] for row in phases]
        )
        require(lifecycle["passed"] is True, "independent lifecycle verdict failed")
        return {
            "cell_id": f"N2-kv-shared-gdn-{policy_name}",
            "resident_count": 2,
            "kv_policy": SHARED_REUSE,
            "gdn_policy": policy_name,
            "fresh_persistent_cache": True,
            "fresh_request_group": True,
            "phases": phases,
            "steps": steps,
            "ledger_receipts": ledger_receipts,
            "independent_lifecycle_verdict": lifecycle,
        }
    finally:
        if backends:
            rr2._unregister_backends(backends)
        persistent = group = session = None
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def run(args: argparse.Namespace) -> dict[str, Any]:
    prereg_raw = check_file(args.preregistration, args.expected_preregistration_sha256, "preregistration")
    prereg = json.loads(prereg_raw)
    require(prereg.get("schema_version") == PREREG_SCHEMA, "preregistration schema drift")
    check_file(args.source_ledger, args.expected_source_ledger_sha256, "R29 source ledger")
    check_file(
        args.candidate_code_ledger,
        args.expected_candidate_code_ledger_sha256,
        "candidate code ledger",
    )
    this_path = Path(__file__).resolve()
    observer_path = this_path.with_name("r29_independent_gdn_observer.py")
    source_binding = prereg.get("source_binding")
    require(isinstance(source_binding, dict), "source binding missing")
    require(sha256_file(this_path) == source_binding["runner_sha256"], "runner source drift")
    require(sha256_file(observer_path) == source_binding["observer_sha256"], "observer source drift")
    require(
        args.expected_candidate_code_ledger_sha256
        == prereg["input_binding"]["candidate_code_ledger_raw_sha256"],
        "candidate code binding drift",
    )
    runtime = _load_runtime(args, prereg)
    with torch.inference_mode():
        cells = [
            _run_policy_cell(runtime, policy_name, candidate_policy)
            for policy_name, candidate_policy in POLICIES
        ]
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed_valid_scientific_execution",
        "preregistration_sha256": args.expected_preregistration_sha256,
        "source_ledger_raw_sha256": args.expected_source_ledger_sha256,
        "candidate_code_ledger_raw_sha256": args.expected_candidate_code_ledger_sha256,
        "runner_sha256": sha256_file(this_path),
        "observer_sha256": sha256_file(observer_path),
        "independence_boundary": prereg["independence_boundary"],
        "hardware": runtime.hardware,
        "input_receipt": runtime.input_receipt,
        "cells": cells,
        "valid_cell_count": len(cells),
        "all_independent_verdicts_passed": all(
            cell["independent_lifecycle_verdict"]["passed"] for cell in cells
        ),
        "all_candidate_comparisons_exact": all(
            phase["independent_candidate_comparison"]["passed"]
            for cell in cells
            for phase in cell["phases"]
        ),
    }
    require(result["valid_cell_count"] == 2, "cell coverage drift")
    require(result["all_independent_verdicts_passed"], "independent verdict failure")
    require(result["all_candidate_comparisons_exact"], "candidate comparison failure")
    write_json(args.output, result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--preregistration", type=Path, required=True)
    value.add_argument("--expected-preregistration-sha256", required=True)
    value.add_argument("--source-ledger", type=Path, required=True)
    value.add_argument("--expected-source-ledger-sha256", required=True)
    value.add_argument("--candidate-code-ledger", type=Path, required=True)
    value.add_argument("--expected-candidate-code-ledger-sha256", required=True)
    value.add_argument("--model-dir", type=Path, required=True)
    value.add_argument("--model-weight-ledger", type=Path, required=True)
    value.add_argument("--model-artifact-ledger", type=Path, required=True)
    value.add_argument("--expected-weight-ledger-sha256", required=True)
    value.add_argument("--expected-artifact-ledger-sha256", required=True)
    value.add_argument("--pg19-data", type=Path, required=True)
    value.add_argument("--pg19-manifest", type=Path, required=True)
    value.add_argument("--expected-pg19-sha256", required=True)
    value.add_argument("--expected-pg19-manifest-sha256", required=True)
    value.add_argument("--expected-windows-sha256", required=True)
    value.add_argument("--frozen-query-banks", type=Path, required=True)
    value.add_argument("--expected-query-banks-sha256", required=True)
    value.add_argument("--expected-gpu-uuid", required=True)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    run(parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
