from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from split_supervised_sft_scale import (
    DATASETS,
    FINGERPRINT_FIELDS,
    MANIFEST_SCHEMA,
    SplitContractError,
    _require_sha256,
    sha256_file,
    stable_json,
    validate_parent_row,
)


ASSIGNMENT_FIELDS = {
    "component_sha256",
    "dataset",
    "parent_row_index",
    "row_sha256",
    "source_id_sha256",
    "split",
}


def _read_canonical_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = []
    row_sha256 = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise SplitContractError(f"{path.name} row {index} is not an object")
            canonical = stable_json(row) + "\n"
            if line != canonical:
                raise SplitContractError(f"{path.name} row {index} is not canonical")
            rows.append(row)
            row_sha256.append(hashlib.sha256(canonical[:-1].encode()).hexdigest())
    return rows, row_sha256


def audit(args: argparse.Namespace) -> dict[str, Any]:
    expected_manifest_sha = _require_sha256(
        args.expected_manifest_sha256, "expected_manifest_sha256"
    )
    actual_manifest_sha = sha256_file(args.manifest)
    if actual_manifest_sha != expected_manifest_sha:
        raise SplitContractError("scale split manifest SHA256 mismatch")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise SplitContractError("scale split manifest schema mismatch")
    if manifest.get("status") != "passed":
        raise SplitContractError("scale split manifest did not pass")
    governance = manifest.get("data_governance")
    if not isinstance(governance, dict) or any(
        governance.get(key) is not False
        for key in (
            "validation_or_test_rows_used",
            "raw_test_v2_read",
            "heldout_ce_is_final_downstream_evaluation",
        )
    ):
        raise SplitContractError("scale data-governance contract failed")
    parent_meta = manifest.get("parent")
    outputs_meta = manifest.get("outputs")
    if not isinstance(parent_meta, dict) or not isinstance(outputs_meta, dict):
        raise SplitContractError("scale manifest artifact bindings are missing")
    bindings = (
        (args.parent_jsonl, parent_meta.get("jsonl_sha256")),
        (args.train_jsonl, outputs_meta.get("train_jsonl", {}).get("sha256")),
        (
            args.heldout_ce_jsonl,
            outputs_meta.get("heldout_ce_jsonl", {}).get("sha256"),
        ),
        (
            args.assignment_ledger,
            outputs_meta.get("assignment_ledger_jsonl", {}).get("sha256"),
        ),
    )
    for path, expected_sha in bindings:
        _require_sha256(expected_sha, f"manifest binding for {path.name}")
        if sha256_file(path) != expected_sha:
            raise SplitContractError(f"artifact SHA256 mismatch for {path.name}")

    parent, parent_hashes = _read_canonical_jsonl(args.parent_jsonl)
    train, train_hashes = _read_canonical_jsonl(args.train_jsonl)
    heldout, heldout_hashes = _read_canonical_jsonl(args.heldout_ce_jsonl)
    assignment, _ = _read_canonical_jsonl(args.assignment_ledger)
    if len(parent) != len(assignment):
        raise SplitContractError("assignment ledger does not cover the parent pool")
    if len(set(parent_hashes)) != len(parent_hashes):
        raise SplitContractError("parent pool contains duplicate canonical rows")
    if set(train_hashes) & set(heldout_hashes):
        raise SplitContractError("the train and CE-heldout row sets overlap")
    if set(train_hashes) | set(heldout_hashes) != set(parent_hashes):
        raise SplitContractError("train/CE-heldout do not exactly partition the parent")

    expected_train_order = []
    expected_heldout_order = []
    for index, (row, row_hash) in enumerate(zip(assignment, parent_hashes)):
        if set(row) != ASSIGNMENT_FIELDS:
            raise SplitContractError(f"assignment row {index} contains raw/unknown fields")
        _require_sha256(
            row.get("component_sha256"),
            f"assignment row {index} component_sha256",
        )
        _require_sha256(row.get("row_sha256"), f"assignment row {index} row_sha256")
        _require_sha256(
            row.get("source_id_sha256"),
            f"assignment row {index} source_id_sha256",
        )
        if row.get("parent_row_index") != index or row.get("row_sha256") != row_hash:
            raise SplitContractError(f"assignment row {index} does not bind parent row")
        if row.get("dataset") != parent[index].get("dataset"):
            raise SplitContractError(f"assignment row {index} dataset binding drifted")
        parent_fingerprints = parent[index].get("provenance", {}).get("fingerprints", {})
        if row.get("source_id_sha256") != parent_fingerprints.get("id_sha256"):
            raise SplitContractError(f"assignment row {index} source-ID binding drifted")
        split_name = row.get("split")
        if split_name == "train":
            expected_train_order.append(row_hash)
        elif split_name == "heldout_ce":
            expected_heldout_order.append(row_hash)
        else:
            raise SplitContractError(f"assignment row {index} has invalid split")
    if train_hashes != expected_train_order or heldout_hashes != expected_heldout_order:
        raise SplitContractError("derived outputs do not preserve parent source order")

    tokenizer = manifest.get("tokenizer")
    prompt = manifest.get("prompt_protocol")
    if not isinstance(tokenizer, dict) or not isinstance(prompt, dict):
        raise SplitContractError("tokenizer/prompt metadata are missing")
    eos_token_id = tokenizer.get("eos_token_id")
    max_sequence_tokens = prompt.get("max_sequence_tokens")
    if not isinstance(eos_token_id, int) or not isinstance(max_sequence_tokens, int):
        raise SplitContractError("tokenizer/prompt token limits are invalid")
    for index, row in enumerate(parent):
        validate_parent_row(
            row,
            row_index=index,
            eos_token_id=eos_token_id,
            max_sequence_tokens=max_sequence_tokens,
        )

    train_counts = Counter(row["dataset"] for row in train)
    heldout_counts = Counter(row["dataset"] for row in heldout)
    expected_train_counts = outputs_meta["train_jsonl"].get("dataset_counts")
    expected_heldout_counts = outputs_meta["heldout_ce_jsonl"].get("dataset_counts")
    if dict(train_counts) != expected_train_counts:
        raise SplitContractError("train dataset counts differ from manifest")
    if dict(heldout_counts) != expected_heldout_counts:
        raise SplitContractError("heldout dataset counts differ from manifest")
    if set(train_counts) != set(DATASETS) or set(heldout_counts) != set(DATASETS):
        raise SplitContractError("both outputs must contain both frozen datasets")

    fingerprint_intersections = {}
    for field in FINGERPRINT_FIELDS:
        train_values = {
            row["provenance"]["fingerprints"][field] for row in train
        }
        heldout_values = {
            row["provenance"]["fingerprints"][field] for row in heldout
        }
        fingerprint_intersections[field] = len(train_values & heldout_values)
    if any(fingerprint_intersections.values()):
        raise SplitContractError("fingerprint leakage crosses train and CE-heldout")
    train_components = {
        row["component_sha256"] for row in assignment if row["split"] == "train"
    }
    heldout_components = {
        row["component_sha256"]
        for row in assignment
        if row["split"] == "heldout_ce"
    }
    if train_components & heldout_components:
        raise SplitContractError("a connected fingerprint component crosses outputs")

    return {
        "status": "passed",
        "manifest_sha256": actual_manifest_sha,
        "parent_partition_exact": True,
        "parent_unique_rows": len(set(parent_hashes)),
        "train_counts": dict(train_counts),
        "heldout_ce_counts": dict(heldout_counts),
        "fingerprint_intersection_counts": fingerprint_intersections,
        "component_intersection_count": 0,
        "assignment_ledger_hash_only": True,
        "all_rows_source_split_train": True,
        "complete_answer_eos_and_ce_mask_rows_validated": len(parent),
        "validation_or_test_rows_used": False,
        "raw_test_v2_read": False,
        "heldout_ce_is_final_downstream_evaluation": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the deterministic train/CE-heldout SFT scale artifacts"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--parent-jsonl", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--heldout-ce-jsonl", type=Path, required=True)
    parser.add_argument("--assignment-ledger", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    try:
        result = audit(parse_args())
    except (OSError, json.JSONDecodeError, SplitContractError) as error:
        raise SystemExit(f"scale audit failed: {error}") from error
    print(stable_json(result, pretty=True), end="")


if __name__ == "__main__":
    main()
