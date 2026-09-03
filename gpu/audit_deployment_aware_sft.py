from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from deployment_aware_sft import (
    HELDOUT_COUNTS,
    TRAIN_COUNTS,
    DeploymentDataset,
    stable_json,
    validate_manifest,
)


def audit(args: argparse.Namespace) -> dict[str, Any]:
    manifest = validate_manifest(
        args.manifest,
        expected_sha256=args.expected_manifest_sha256,
        train_path=args.train,
        train_sha256=args.expected_train_sha256,
        heldout_path=args.heldout,
        heldout_sha256=args.expected_heldout_sha256,
    )
    train = DeploymentDataset(
        args.train,
        split="train",
        max_sequence_tokens=4096,
        expected_sha256=args.expected_train_sha256,
        expected_counts=TRAIN_COUNTS,
    )
    heldout = DeploymentDataset(
        args.heldout,
        split="heldout",
        max_sequence_tokens=4096,
        expected_sha256=args.expected_heldout_sha256,
        expected_counts=HELDOUT_COUNTS,
    )
    fields = (
        "example_id",
        "source_id_sha256",
        "document_id_sha256",
        "prompt_sha256",
        "context_sha256",
    )
    intersections = {}
    for field in fields:
        left = {
            getattr(example, field)
            for example in train.examples
            if getattr(example, field) is not None
        }
        right = {
            getattr(example, field)
            for example in heldout.examples
            if getattr(example, field) is not None
        }
        intersections[field] = len(left & right)
    if any(intersections.values()):
        raise SystemExit(f"train/heldout intersection detected: {intersections}")
    qasper_documents = Counter(
        example.document_id_sha256
        for example in train.examples + heldout.examples
        if example.dataset == "qasper"
    )
    if not qasper_documents or set(qasper_documents.values()) != {2}:
        raise SystemExit("QASPER documents do not each have exactly two queries")
    if max(example.sequence_tokens for example in train.examples[:8]) != 4096:
        raise SystemExit("step-1 batch does not exercise one 4096-token sequence")
    if len(train.examples) != 1024 or len(heldout.examples) != 64:
        raise SystemExit("formal dataset sizes drifted")
    if sum(example.teacher_required for example in train.examples) != 307:
        raise SystemExit("teacher-preservation train quota drifted")
    raw_train_rows = [json.loads(line) for line in args.train.read_text().splitlines()]
    domain_rows = [row for row in raw_train_rows if row["stratum"] == "domain"]
    if len(domain_rows) != TRAIN_COUNTS["domain"]:
        raise SystemExit("domain boundary row count drifted")
    for row in domain_rows:
        first_target = next(
            index for index, value in enumerate(row["labels"]) if value != -100
        )
        boundary = row["deployment_boundary"]
        if (
            boundary["applicable"] is not True
            or not boundary["document_input_ids"]
            or not boundary["query_input_ids"]
            or boundary["document_input_ids"] + boundary["query_input_ids"]
            != row["input_ids"][:first_target]
        ):
            raise SystemExit("domain document/query boundary hard audit failed")
    if manifest["schedule"]["ordered_example_id_sha256"] != __import__(
        "hashlib"
    ).sha256(
        "\n".join(example.example_id for example in train.examples).encode("ascii")
    ).hexdigest():
        raise SystemExit("ordered schedule digest drifted")
    return {
        "schema_version": "qcomem-deployment-aware-independent-audit-v1",
        "passed": True,
        "manifest_sha256": args.expected_manifest_sha256,
        "train": train.audit,
        "heldout": heldout.audit,
        "train_heldout_intersections": intersections,
        "qasper_documents": len(qasper_documents),
        "qasper_queries_per_document": 2,
        "teacher_train_rows": 307,
        "domain_boundary_rows": len(domain_rows),
        "domain_boundary_reconstruction_checked_per_row": True,
        "answer_or_eos_tokens_in_query": False,
        "step1_max_sequence_tokens": max(
            example.sequence_tokens for example in train.examples[:8]
        ),
        "global_target_token_weighting": False,
        "longbench_raw_rows_read_by_auditor": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Independently audit deployment-aware SFT data")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--expected-heldout-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    print(stable_json(audit(parse_args()), pretty=True), end="")


if __name__ == "__main__":
    main()
