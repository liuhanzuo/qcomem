from __future__ import annotations

"""Run the registered RR2 clean-package rerun on one visible H20.

This is deliberately smaller than the separate eight-rank formal experiment:
rank 0, N=2, one semantic step, one ownership-witness rebuild, and one live
M1 matched-control/mutant pair.  The output binds the freshly rebuilt PG-19
inputs and contains only package-relative artifact references.
"""

import argparse
import gc
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

import build_qcomem_forkaudit_rr2_input_manifest as input_builder
import qcomem_joint_policy as joint_policy
import qcomem_vllm_paged_multifork_resident as resident
import run_qcomem_qwen35_forkaudit_review_revision as formal


SCHEMA_VERSION = "qcomem-forkaudit-rr2-clean-rerun-v1"
EXPERIMENT_ID = "RR2-REP-CLEAN-RERUN"
RESIDENT_COUNT = 2
SEMANTIC_STEPS = 1
RANK = 0
KV_POLICY = formal.SHARED_REUSE
GDN_POLICY = formal.GDN_BORROW_IMMUTABLE_BASE
ARM_ID = f"kv={KV_POLICY}|gdn={GDN_POLICY}"
RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class CleanRerunError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CleanRerunError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _read_exact(path: Path, expected_sha256: str, label: str) -> bytes:
    payload = path.read_bytes()
    _require(_sha256_bytes(payload) == expected_sha256, f"{label} SHA-256 mismatch")
    return payload


def _code_ledger(code_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(code_dir.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(code_dir).as_posix()
        if "__pycache__" in path.parts or relative.endswith((".pyc", ".pyo")):
            continue
        payload = path.read_bytes()
        rows.append(
            {"relative_path": relative, "sha256": _sha256_bytes(payload), "bytes": len(payload)}
        )
    _require(rows, "clean-rerun source closure is empty")
    return rows


def _assert_path_independent(value: Any) -> None:
    if isinstance(value, str):
        lowered = value.lower()
        _require(not value.startswith("/"), "result leaked an absolute path")
        _require("file://" not in lowered, "result leaked a file URI")
        _require("/users/" not in lowered and "/mnt/" not in lowered, "result leaked a host path")
    elif isinstance(value, list):
        for item in value:
            _assert_path_independent(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            _assert_path_independent(str(key))
            _assert_path_independent(item)


def _rebuild_rank_zero_inputs(
    *,
    pg19_data: Path,
    pg19_manifest: Path,
    prior_capacity_manifest: Path,
    model_dir: Path,
    frozen_rr2_manifest: Path,
    expected_rr2_manifest_sha256: str,
) -> tuple[dict[str, Any], torch.Tensor, tuple[torch.Tensor, ...]]:
    tokenizer = input_builder.load_local_tokenizer(model_dir)
    rebuilt = input_builder.build_from_paths(
        pg19_data=pg19_data,
        pg19_manifest=pg19_manifest,
        prior_capacity_manifest=prior_capacity_manifest,
        model_dir=model_dir,
        tokenizer=tokenizer,
    )
    rebuilt_bytes = input_builder.canonical_json_bytes(rebuilt) + b"\n"
    frozen_bytes = _read_exact(
        frozen_rr2_manifest,
        expected_rr2_manifest_sha256,
        "frozen RR2 input manifest",
    )
    _require(rebuilt_bytes == frozen_bytes, "fresh RR2 input reconstruction differs byte-for-byte")

    data_bytes = pg19_data.read_bytes()
    manifest_bytes = pg19_manifest.read_bytes()
    records, _audit = input_builder._audit_pg19_train64_bytes(
        data_bytes,
        manifest_bytes,
        expectations=input_builder.FORMAL_EXPECTATIONS,
    )
    windows, windows_sha = joint_policy.build_pg19_calibration_windows(
        records,
        tokenizer,
        books=input_builder.FORMAL_BOOKS,
        document_tokens=input_builder.FORMAL_DOCUMENT_TOKENS,
        query_tokens=input_builder.FORMAL_QUERY_TOKENS,
        stride=input_builder.FORMAL_WINDOW_STRIDE,
        candidate_windows_per_book=input_builder.FORMAL_CANDIDATE_WINDOWS,
        seed=input_builder.FORMAL_SEED,
    )
    _require(windows_sha == rebuilt["pg19_windows_sha256"], "fresh window digest drift")
    window = windows[RANK]
    query_tensors, query_audit = resident.build_pg19_train_query_bank(
        records,
        tokenizer,
        window,
        document_tokens=input_builder.FORMAL_DOCUMENT_TOKENS,
        query_tokens=input_builder.FORMAL_QUERY_TOKENS,
        count=input_builder.FORMAL_QUERY_BANK_COUNT,
        query_stride=input_builder.FORMAL_QUERY_BANK_STRIDE,
    )
    bank = rebuilt["frozen_query_banks"][RANK]
    _require(query_audit["query_bank_sha256"] == bank["query_bank_sha256"], "fresh query-bank digest drift")
    document = window.document_ids.detach().contiguous().reshape(1, -1)
    queries = tuple(query.detach().contiguous().reshape(1, -1) for query in query_tensors[:RESIDENT_COUNT])
    _require(formal._token_id_sha256(document, expected_shape=(1, formal.FORMAL_DOCUMENT_TOKENS)) == bank["document_token_ids_sha256"], "fresh document tensor drift")
    _require(
        [formal._token_id_sha256(query, expected_shape=(1, formal.FORMAL_QUERY_TOKENS)) for query in queries]
        == [row["query_token_ids_sha256"] for row in bank["rows"][:RESIDENT_COUNT]],
        "fresh query tensors drift",
    )
    receipt = {
        "rr2_input_manifest_raw_sha256": _sha256_bytes(frozen_bytes),
        "rebuilt_bytes_equal_frozen_bytes": True,
        "pg19_data_sha256": _sha256_bytes(data_bytes),
        "pg19_manifest_sha256": _sha256_bytes(manifest_bytes),
        "pg19_windows_sha256": windows_sha,
        "rank": RANK,
        "source_object": bank["source_object"],
        "document_start_token": bank["document_start_token"],
        "document_token_ids_sha256": bank["document_token_ids_sha256"],
        "query_token_ids_sha256": [row["query_token_ids_sha256"] for row in bank["rows"][:RESIDENT_COUNT]],
        "resident_count": RESIDENT_COUNT,
    }
    del tokenizer
    return receipt, document, queries


def _load_runtime(model_dir: Path, document_cpu: torch.Tensor, queries_cpu: Sequence[torch.Tensor]) -> tuple[Any, Any, Any, Any, torch.Tensor, tuple[torch.Tensor, ...], dict[str, Any]]:
    from qcomem_qwen35_functional_stack import audit_qwen35_functional_stack_plan
    from qcomem_vllm_paged_kernel import _resolve_vllm_unified_attention, audit_frozen_kernel_environment
    from run_qcomem_qwen35_vllm_paged_multifork_resident import _audit_model_config_geometry, _resolve_backbone
    from transformers import AutoModelForImageTextToText

    _require(torch.cuda.device_count() >= 1, "clean rerun requires one visible CUDA device")
    _require(torch.cuda.is_bf16_supported(), "clean rerun requires BF16 support")
    name = torch.cuda.get_device_name(0)
    capability = list(torch.cuda.get_device_capability(0))
    _require("H20" in name and capability == [9, 0], "clean rerun requires an H20 (cc 9.0)")
    model = AutoModelForImageTextToText.from_pretrained(
        str(model_dir),
        revision=formal.FORMAL_MODEL_REVISION,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    outer = getattr(model, "model", None)
    visual_removed = bool(outer is not None and hasattr(outer, "visual"))
    if visual_removed:
        outer.visual = None
    model.eval()
    geometry = _audit_model_config_geometry(model_dir)
    plan = audit_qwen35_functional_stack_plan(model)
    kernel_environment = audit_frozen_kernel_environment()
    _require(kernel_environment.get("matches_frozen_environment") is True, "frozen kernel environment drift")
    kernel = _resolve_vllm_unified_attention()
    model = model.to(device="cuda:0", dtype=torch.bfloat16)
    backbone = _resolve_backbone(model)
    document = document_cpu.to(device="cuda:0", dtype=torch.int64)
    queries = tuple(query.to(device="cuda:0", dtype=torch.int64) for query in queries_cpu)
    audit = {
        "gpu_name": name,
        "compute_capability": capability,
        "visible_device_count": torch.cuda.device_count(),
        "used_device": "cuda:0",
        "bf16_supported": True,
        "model_id": formal.FORMAL_MODEL_ID,
        "model_revision": formal.FORMAL_MODEL_REVISION,
        "local_files_only": True,
        "trust_remote_code": False,
        "visual_branch_removed": visual_removed,
        "geometry": geometry,
        "functional_stack_plan_sha256": _sha256_json(plan.metadata()),
        "kernel_environment": {
            "expected_versions": dict(kernel_environment["expected_versions"]),
            "observed_versions": dict(kernel_environment["observed_versions"]),
            "matches_frozen_environment": True,
            "kernel_entrypoint": kernel_environment["kernel_entrypoint"],
            "kernel_mode": kernel_environment["kernel_mode"],
        },
    }
    return model, backbone, plan, kernel, document, queries, audit


def _run_gpu_smoke(*, artifact_root: Path, run_id: str, model_dir: Path, document_cpu: torch.Tensor, queries_cpu: Sequence[torch.Tensor]) -> dict[str, Any]:
    model, backbone, plan, kernel, document, queries, runtime_audit = _load_runtime(model_dir, document_cpu, queries_cpu)
    old_steps = formal.FORMAL_GENERATION_STEPS
    old_final = formal.FORMAL_FINAL_APPENDED_TOKENS
    formal.FORMAL_GENERATION_STEPS = SEMANTIC_STEPS
    formal.FORMAL_FINAL_APPENDED_TOKENS = formal.FORMAL_QUERY_TOKENS + SEMANTIC_STEPS - 1
    try:
        with torch.inference_mode():
            # The first real paged-attention/GDN pass may initialize process-
            # lifetime CUDA state.  It is not an endpoint.  Freeze the
            # allocator baseline only after this discarded priming cell.
            priming = formal._run_clean_memory_cell(
                rank=RANK,
                arm_id=ARM_ID,
                resident_count=RESIDENT_COUNT,
                kv_policy=KV_POLICY,
                gdn_base_policy=GDN_POLICY,
                model=model,
                backbone=backbone,
                plan=plan,
                document=document,
                queries=queries,
                kernel=kernel,
            )
            _require(
                priming["memory_cell"]["primary_memory_endpoint_eligible"] is True,
                "discarded priming cell did not execute the production path",
            )
            del priming
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            frozen_baseline = formal._gpu_cleanup()
            memory = formal._execute_cell_with_cleanup(
                lambda: formal._run_clean_memory_cell(
                    rank=RANK,
                    arm_id=ARM_ID,
                    resident_count=RESIDENT_COUNT,
                    kv_policy=KV_POLICY,
                    gdn_base_policy=GDN_POLICY,
                    model=model,
                    backbone=backbone,
                    plan=plan,
                    document=document,
                    queries=queries,
                    kernel=kernel,
                    expected_frozen_baseline=frozen_baseline,
                ),
                cell_role_key="memory_cell",
                frozen_baseline=frozen_baseline,
                label="RR2 clean-rerun memory smoke",
            )
            oracle_selection = {
                "kv_policy": KV_POLICY,
                "gdn_base_policy": GDN_POLICY,
                "request_index": 0,
                "round_index": 0,
                "layer_index": int(plan.full_attention_layer_indices[0]),
                "sample_id": "clean-rerun-no-oracle-N2",
            }
            witness = formal._execute_cell_with_cleanup(
                lambda: formal._run_ownership_witness_cell(
                    artifact_root=artifact_root,
                    run_id=run_id,
                    rank=RANK,
                    arm_id=ARM_ID,
                    resident_count=RESIDENT_COUNT,
                    kv_policy=KV_POLICY,
                    gdn_base_policy=GDN_POLICY,
                    model=model,
                    backbone=backbone,
                    plan=plan,
                    document=document,
                    queries=queries,
                    kernel=kernel,
                    oracle_selection=oracle_selection,
                ),
                cell_role_key="witness_cell",
                frozen_baseline=frozen_baseline,
                label="RR2 clean-rerun ownership smoke",
            )
            _require(witness["oracle_raw_artifact"] is None, "N=2 clean rerun unexpectedly emitted an oracle sample")
            _require(memory["semantics"] == witness["semantics"], "memory/witness semantic rows differ")

            before_clean = formal._gpu_cleanup()
            _require(before_clean["current_allocated_bytes"] == frozen_baseline["current_allocated_bytes"] and before_clean["current_reserved_bytes"] == frozen_baseline["current_reserved_bytes"], "baseline drift before M1 matched control")
            matched = formal._run_one_live_mutant(
                "M1", rank=RANK, model=model, backbone=backbone, plan=plan,
                document=document, queries=queries, kernel=kernel, activate_mutation=False,
            )
            matched_outcome = formal._parse_campaign_outcome(matched["outcome"])
            matched_passed = formal._validate_clean_outcome(matched_outcome)
            del matched_outcome
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            mutant = formal._run_one_live_mutant(
                "M1", rank=RANK, model=model, backbone=backbone, plan=plan,
                document=document, queries=queries, kernel=kernel, activate_mutation=True,
            )
            mutant_outcome = formal._parse_campaign_outcome(mutant["outcome"])
            _require(mutant_outcome.classification.value == "detected_expected_gate", "M1 was not detected at its preregistered gate")
            _require(mutant_outcome.observed_gate_id == mutant_outcome.expected_gate_id, "M1 observed/expected gate mismatch")
            del mutant_outcome
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            recovered = formal._gpu_cleanup()
            _require(recovered["current_allocated_bytes"] == frozen_baseline["current_allocated_bytes"] and recovered["current_reserved_bytes"] == frozen_baseline["current_reserved_bytes"], "allocator did not recover after M1 cases")
        return {
            "runtime_audit": runtime_audit,
            "configuration": {"rank": RANK, "resident_count": RESIDENT_COUNT, "semantic_steps": SEMANTIC_STEPS, "arm_id": ARM_ID},
            "priming": {
                "one_discarded_production_cell": True,
                "allocator_baseline_frozen_after_priming": True,
                "excluded_from_reported_endpoints": True,
            },
            "memory": memory,
            "ownership_witness": witness,
            "mutant_smoke": {"mutant_id": "M1", "matched_clean_passed": matched_passed, "matched_clean": matched, "injected": mutant},
            "frozen_allocator_baseline": frozen_baseline,
            "final_allocator_baseline": recovered,
        }
    finally:
        formal.FORMAL_GENERATION_STEPS = old_steps
        formal.FORMAL_FINAL_APPENDED_TOKENS = old_final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--pg19-data", type=Path, required=True)
    parser.add_argument("--pg19-manifest", type=Path, required=True)
    parser.add_argument("--prior-capacity-manifest", type=Path, required=True)
    parser.add_argument("--rr2-input-manifest", type=Path, required=True)
    parser.add_argument("--expected-rr2-input-manifest-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require(RUN_ID_RE.fullmatch(args.run_id) is not None, "run ID must be 128-bit lowercase hex")
    output_root = args.output_root.resolve()
    _require(not output_root.exists(), "clean-rerun output root must not pre-exist")
    output_root.mkdir(parents=True)
    input_receipt, document_cpu, queries_cpu = _rebuild_rank_zero_inputs(
        pg19_data=args.pg19_data,
        pg19_manifest=args.pg19_manifest,
        prior_capacity_manifest=args.prior_capacity_manifest,
        model_dir=args.model_dir,
        frozen_rr2_manifest=args.rr2_input_manifest,
        expected_rr2_manifest_sha256=args.expected_rr2_input_manifest_sha256,
    )
    ledger = _code_ledger(args.code_dir.resolve())
    smoke = _run_gpu_smoke(
        artifact_root=output_root,
        run_id=args.run_id,
        model_dir=args.model_dir,
        document_cpu=document_cpu,
        queries_cpu=queries_cpu,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "status": "completed",
        "run_id": args.run_id,
        "scope": "fresh-process one-H20 smoke; separate from the eight-rank formal cohort",
        "input_reconstruction": input_receipt,
        "source_closure": {"rows": ledger, "rows_sha256": _sha256_json(ledger)},
        "smoke": smoke,
        "package_relative_outputs": True,
        "candidate_outputs_consumed_during_input_selection": False,
    }
    _assert_path_independent(result)
    payload = _canonical_bytes(result) + b"\n"
    result_path = output_root / "clean-rerun-result.json"
    result_path.write_bytes(payload)
    (output_root / "clean-rerun-result.sha256").write_text(
        f"{_sha256_bytes(payload)}  clean-rerun-result.json\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
