from __future__ import annotations

"""Deterministically build the PG19-only multi-fork runtime manifest.

This builder is CPU-only.  It binds the exact tokenizer, 4095-token windows,
32 raw query chunks per rank and all externally frozen code/model/data ledgers.
The outer QS YAML is deliberately not part of the manifest, avoiding a
self-referential digest.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from qcomem_joint_policy import (
    audit_pg19_train_calibration,
    build_pg19_calibration_windows,
)
from qcomem_vllm_paged_multifork_resident import (
    MULTIFORK_COUNTS,
    MULTIFORK_PROTOCOL,
    build_pg19_train_query_bank,
)
from run_downstream import atomic_json
from run_qcomem_qwen35_vllm_paged_multifork_resident import (
    FORMAL_BOOKS,
    FORMAL_CANDIDATES,
    FORMAL_DOCUMENT_TOKENS,
    FORMAL_EXECUTION_ORDER,
    FORMAL_NEW_TOKENS,
    FORMAL_PAGE_SIZE,
    FORMAL_QUERY_TOKENS,
    FORMAL_SEED,
    FORMAL_WINDOW_STRIDE,
    FORMAL_WORLD_SIZE,
    _model_manifest_sha,
    _protocol_config,
    _protocol_config_sha256,
    _require,
)


def _sha256(value: str, label: str) -> str:
    _require(
        len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be one lowercase SHA256",
    )
    return value


def build_manifest(args: argparse.Namespace) -> dict[str, object]:
    for field in (
        "expected_pg19_sha256",
        "expected_pg19_manifest_sha256",
        "expected_pg19_windows_sha256",
        "expected_model_manifest_sha256",
        "expected_code_ledger_sha256",
        "expected_model_artifact_ledger_sha256",
        "expected_model_weight_ledger_sha256",
    ):
        _sha256(getattr(args, field), field)
    _require(not torch.cuda.is_initialized(), "manifest builder must not initialize CUDA")
    records, data_audit = audit_pg19_train_calibration(
        args.pg19_data,
        args.pg19_manifest,
        expected_data_sha256=args.expected_pg19_sha256,
        expected_manifest_sha256=args.expected_pg19_manifest_sha256,
        minimum_books=FORMAL_BOOKS,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    windows, windows_sha256 = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=FORMAL_BOOKS,
        document_tokens=FORMAL_DOCUMENT_TOKENS,
        query_tokens=FORMAL_QUERY_TOKENS,
        stride=FORMAL_WINDOW_STRIDE,
        candidate_windows_per_book=FORMAL_CANDIDATES,
        seed=FORMAL_SEED,
    )
    _require(
        windows_sha256 == args.expected_pg19_windows_sha256,
        "rebuilt PG19 windows SHA differs from preregistration",
    )
    frozen_query_banks = []
    for window in windows:
        _, bank = build_pg19_train_query_bank(
            records,
            tokenizer,
            window,
            document_tokens=FORMAL_DOCUMENT_TOKENS,
            query_tokens=FORMAL_QUERY_TOKENS,
            count=max(MULTIFORK_COUNTS),
            query_stride=64,
        )
        frozen_query_banks.append(bank)
    _require(len(frozen_query_banks) == FORMAL_WORLD_SIZE, "query bank count drift")
    _require(
        len({bank["source_object"] for bank in frozen_query_banks})
        == FORMAL_WORLD_SIZE,
        "query banks do not use eight distinct PG19 train books",
    )
    model_manifest_sha256, model_manifest = _model_manifest_sha(args.model)
    _require(
        model_manifest_sha256 == args.expected_model_manifest_sha256,
        "model manifest SHA differs from preregistration",
    )
    config_args = SimpleNamespace(
        bits=16,
        page_size=FORMAL_PAGE_SIZE,
        world_size=FORMAL_WORLD_SIZE,
        resident_counts=list(MULTIFORK_COUNTS),
        execution_order=list(FORMAL_EXECUTION_ORDER),
        pg19_books=FORMAL_BOOKS,
        pg19_document_tokens=FORMAL_DOCUMENT_TOKENS,
        pg19_query_tokens=FORMAL_QUERY_TOKENS,
        pg19_window_stride=FORMAL_WINDOW_STRIDE,
        pg19_candidate_windows=FORMAL_CANDIDATES,
        pg19_seed=FORMAL_SEED,
        query_bank_stride=64,
        max_new_tokens=FORMAL_NEW_TOKENS,
    )
    protocol_config = _protocol_config(config_args)
    frozen_identity = {
        "code_ledger_sha256": args.expected_code_ledger_sha256,
        "model_manifest_sha256": args.expected_model_manifest_sha256,
        "model_artifact_ledger_sha256": (
            args.expected_model_artifact_ledger_sha256
        ),
        "model_weight_ledger_sha256": args.expected_model_weight_ledger_sha256,
        "pg19_data_sha256": args.expected_pg19_sha256,
        "pg19_manifest_sha256": args.expected_pg19_manifest_sha256,
        "pg19_windows_sha256": args.expected_pg19_windows_sha256,
        "protocol_config_sha256": _protocol_config_sha256(protocol_config),
    }
    _require(not torch.cuda.is_initialized(), "manifest build initialized CUDA")
    return {
        "schema_version": 1,
        "protocol": MULTIFORK_PROTOCOL,
        "protocol_config": protocol_config,
        "frozen_identity": frozen_identity,
        "frozen_query_banks": frozen_query_banks,
        "build_audit": {
            "data_audit": data_audit,
            "model_manifest": model_manifest,
            "window_count": len(windows),
            "query_bank_count": len(frozen_query_banks),
            "all_model_inputs_from_pg19_train": True,
            "synthetic_markers_used": False,
            "longbench_consumed": False,
            "source_6_9_consumed": False,
            "source_68_99_consumed": False,
            "test_v2_consumed": False,
            "gpu_initialized": False,
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--pg19-data", type=Path, required=True)
    parser.add_argument("--pg19-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-pg19-sha256", required=True)
    parser.add_argument("--expected-pg19-manifest-sha256", required=True)
    parser.add_argument("--expected-pg19-windows-sha256", required=True)
    parser.add_argument("--expected-model-manifest-sha256", required=True)
    parser.add_argument("--expected-code-ledger-sha256", required=True)
    parser.add_argument("--expected-model-artifact-ledger-sha256", required=True)
    parser.add_argument("--expected-model-weight-ledger-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = build_manifest(args)
    atomic_json(args.output, result)
    print(
        json.dumps(
            {
                "status": "multifork_runtime_manifest_built",
                "output": str(args.output),
                "pg19_windows_sha256": args.expected_pg19_windows_sha256,
            }
        )
    )


if __name__ == "__main__":
    main()
