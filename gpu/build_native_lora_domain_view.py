from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from deployment_aware_sft import (
    HELDOUT_COUNTS,
    TRAIN_COUNTS,
    validate_example_row,
    validate_manifest,
)
from qcomem_native_lora_protocol import (
    BOUNDARY_SCHEMA,
    VIEW_SCHEMA,
    NativeLoRAProtocolError,
    domain_view_record,
    sampler_scheduled_records,
    sha256_file,
    stable_json,
    view_summary,
)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=path.name,
        delete=False,
    ) as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_parent_rows(path: Path, *, split: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or line != stable_json(row) + "\n":
                raise NativeLoRAProtocolError(
                    f"{split} parent row {row_index} is not canonical JSONL"
                )
            validate_example_row(
                row,
                split=split,
                max_sequence_tokens=4096,
                row_index=row_index,
            )
            rows.append(row)
    return rows


def write_view(path: Path, records: list[dict[str, Any]]) -> str:
    atomic_text(path, "".join(stable_json(record) + "\n" for record in records))
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive answer-free native-cache LoRA views from frozen SFT data"
    )
    parser.add_argument("--parent-train", type=Path, required=True)
    parser.add_argument("--parent-heldout", type=Path, required=True)
    parser.add_argument("--parent-manifest", type=Path, required=True)
    parser.add_argument("--parent-train-sha256", required=True)
    parser.add_argument("--parent-heldout-sha256", required=True)
    parser.add_argument("--parent-manifest-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-document-tokens", type=int, default=1536)
    parser.add_argument("--max-query-tokens", type=int, default=512)
    parser.add_argument("--sampler-seed", type=int, default=17)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--expected-train-view-sha256")
    parser.add_argument("--expected-heldout-view-sha256")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"refusing non-empty output directory {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    validate_manifest(
        args.parent_manifest,
        expected_sha256=args.parent_manifest_sha256,
        train_path=args.parent_train,
        train_sha256=args.parent_train_sha256,
        heldout_path=args.parent_heldout,
        heldout_sha256=args.parent_heldout_sha256,
    )
    train_parent = read_parent_rows(args.parent_train, split="train")
    heldout_parent = read_parent_rows(args.parent_heldout, split="heldout")
    train = [
        domain_view_record(
            row,
            max_document_tokens=args.max_document_tokens,
            max_query_tokens=args.max_query_tokens,
        )
        for row in train_parent
        if row["stratum"] == "domain"
    ]
    heldout = [
        domain_view_record(
            row,
            max_document_tokens=args.max_document_tokens,
            max_query_tokens=args.max_query_tokens,
        )
        for row in heldout_parent
        if row["stratum"] == "domain"
    ]
    if len(train) != TRAIN_COUNTS["domain"]:
        raise SystemExit(
            f"expected {TRAIN_COUNTS['domain']} train domain rows, found {len(train)}"
        )
    if len(heldout) != HELDOUT_COUNTS["domain"]:
        raise SystemExit(
            f"expected {HELDOUT_COUNTS['domain']} heldout domain rows, found {len(heldout)}"
        )
    train, schedule = sampler_scheduled_records(
        train,
        seed=args.sampler_seed,
        world_size=args.world_size,
    )
    train_path = args.output_dir / "native-lora-domain-train.jsonl"
    heldout_path = args.output_dir / "native-lora-domain-heldout.jsonl"
    train_sha = write_view(train_path, train)
    heldout_sha = write_view(heldout_path, heldout)
    if (
        args.expected_train_view_sha256 is not None
        and train_sha != args.expected_train_view_sha256
    ):
        raise SystemExit("derived train-view SHA256 differs from the frozen expectation")
    if (
        args.expected_heldout_view_sha256 is not None
        and heldout_sha != args.expected_heldout_view_sha256
    ):
        raise SystemExit("derived heldout-view SHA256 differs from the frozen expectation")
    manifest = {
        "schema_version": VIEW_SCHEMA,
        "status": "passed",
        "parent": {
            "schema": "qcomem-deployment-aware-sft-v1",
            "train": {"path": str(args.parent_train), "sha256": args.parent_train_sha256},
            "heldout": {
                "path": str(args.parent_heldout),
                "sha256": args.parent_heldout_sha256,
            },
            "manifest": {
                "path": str(args.parent_manifest),
                "sha256": args.parent_manifest_sha256,
            },
        },
        "boundary_schema": BOUNDARY_SCHEMA,
        "view_contract": {
            "included_strata": ["domain"],
            "excluded_strata": ["general_replay", "teacher_preservation"],
            "document_max_tokens": args.max_document_tokens,
            "document_truncation": "symmetric_head_tail_v1_if_needed",
            "query_max_tokens": args.max_query_tokens,
            "query_truncation": "forbidden_fail_closed",
            "answer_or_eos_tokens_in_query": False,
            "teacher": "online_q16_mutable_same_document_query_boundary",
            "student": "q4_native_functional_same_document_query_boundary",
        },
        "outputs": {
            "train": {
                "path": str(train_path),
                "sha256": train_sha,
                "summary": view_summary(train),
            },
            "heldout": {
                "path": str(heldout_path),
                "sha256": heldout_sha,
                "summary": view_summary(heldout),
            },
        },
        "train_sampler_schedule": schedule,
        "governance": {
            "official_train_sources_only": True,
            "longbench_validation_rows_used_for_training": False,
            "longbench_test_v2_rows_read": False,
            "test_v2_used": False,
        },
    }
    manifest_path = args.output_dir / "native-lora-domain-view.manifest.json"
    atomic_text(manifest_path, stable_json(manifest) + "\n")
    result = {
        "status": "passed",
        "train": str(train_path),
        "train_sha256": train_sha,
        "heldout": str(heldout_path),
        "heldout_sha256": heldout_sha,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "train_summary": manifest["outputs"]["train"]["summary"],
        "heldout_summary": manifest["outputs"]["heldout"]["summary"],
        "test_v2_used": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
