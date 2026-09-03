#!/usr/bin/env python3
"""Freeze output-unseen PG-19 inputs for the R30 end-to-end control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


SCHEMA = "forkaudit-r30-e2e-input-manifest-v1"
PG19_DATA_SHA256 = "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
PG19_MANIFEST_SHA256 = "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
SEED = 2026082504
BOOKS = 8
DOCUMENT_TOKENS = 4095
QUERY_TOKENS = 32
WINDOW_STRIDE = 271
CANDIDATE_WINDOWS = 8
REQUESTS_PER_DOCUMENT = 2
QUERY_STRIDE = 64
SELECTED_DOCUMENTS = 2
GREEDY_STEPS = 4
SELECTION_DOMAIN = b"forkaudit-r30-e2e-output-unseen-selection-v1\0"


class InputFreezeError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise InputFreezeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def tensor_sha256(tensor: Any) -> str:
    raw = tensor.detach().contiguous().cpu().numpy().astype("<i8", copy=False).tobytes()
    return hashlib.sha256(raw).hexdigest()


def load_exclusions(path: Path) -> tuple[set[tuple[str, int, int]], set[str], set[str], dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    require(value.get("schema_version") == "forkaudit-r30-e2e-known-input-exclusions-v1", "exclusion schema drift")
    rows = value.get("rows")
    require(isinstance(rows, list) and rows, "empty exclusion ledger")
    coordinates: set[tuple[str, int, int]] = set()
    documents: set[str] = set()
    queries: set[str] = set()
    for row in rows:
        require(isinstance(row, dict), "malformed exclusion row")
        source = row.get("source_object")
        start = row.get("document_start_token")
        length = row.get("document_tokens")
        document_digest = row.get("document_token_ids_sha256")
        query_digests = row.get("query_token_ids_sha256")
        require(isinstance(source, str) and source.startswith("train/"), "non-train exclusion")
        require(type(length) is int and length > 0, "invalid exclusion length")
        if type(start) is int:
            coordinates.add((source, start, length))
        require(isinstance(document_digest, str) and len(document_digest) == 64, "invalid exclusion document digest")
        require(isinstance(query_digests, list), "invalid exclusion query list")
        documents.add(document_digest)
        queries.update(query_digests)
    return coordinates, documents, queries, {
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "row_count": len(rows),
        "coordinate_count": len(coordinates),
        "document_digest_count": len(documents),
        "query_digest_count": len(queries),
    }


def tokenizer_receipt(model_dir: Path) -> dict[str, Any]:
    names = ("tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja")
    rows = []
    for name in names:
        path = model_dir / name
        require(path.is_file(), f"missing tokenizer artifact: {name}")
        rows.append({"name": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"rows": rows, "rows_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest()}


def selection_score(source_object: str, start: int, document_sha256: str) -> str:
    payload = f"{SEED}\0{source_object}\0{start}\0{DOCUMENT_TOKENS}\0{document_sha256}".encode()
    return hashlib.sha256(SELECTION_DOMAIN + payload).hexdigest()


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    require(args.output.resolve() != args.exclusions.resolve(), "output aliases exclusions")
    require(sha256_file(args.pg19_data) == PG19_DATA_SHA256, "PG-19 data SHA drift")
    require(sha256_file(args.pg19_manifest) == PG19_MANIFEST_SHA256, "PG-19 manifest SHA drift")
    coordinates, excluded_documents, excluded_queries, exclusion_receipt = load_exclusions(args.exclusions)

    sys.path.insert(0, str(args.rr2_code_dir.resolve()))
    from qcomem_joint_policy import (  # pylint: disable=import-error,import-outside-toplevel
        audit_pg19_train_calibration,
        build_pg19_calibration_windows,
    )
    from qcomem_vllm_paged_multifork_resident import (  # pylint: disable=import-error,import-outside-toplevel
        build_pg19_train_query_bank,
    )
    from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

    records, data_audit = audit_pg19_train_calibration(
        args.pg19_data,
        args.pg19_manifest,
        expected_data_sha256=PG19_DATA_SHA256,
        expected_manifest_sha256=PG19_MANIFEST_SHA256,
        minimum_books=64,
    )
    require(len(records) == 64, "PG-19 train64 record count drift")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    windows, windows_sha256 = build_pg19_calibration_windows(
        records,
        tokenizer,
        books=BOOKS,
        document_tokens=DOCUMENT_TOKENS,
        query_tokens=QUERY_TOKENS,
        stride=WINDOW_STRIDE,
        candidate_windows_per_book=CANDIDATE_WINDOWS,
        seed=SEED,
    )
    require(len(windows) == BOOKS, "candidate window cardinality drift")

    pool = []
    for pool_index, window in enumerate(windows):
        document = window.document_ids.detach().contiguous().cpu()
        document_digest = tensor_sha256(document)
        queries, query_audit = build_pg19_train_query_bank(
            records,
            tokenizer,
            window,
            document_tokens=DOCUMENT_TOKENS,
            query_tokens=QUERY_TOKENS,
            count=REQUESTS_PER_DOCUMENT,
            query_stride=QUERY_STRIDE,
        )
        query_digests = [tensor_sha256(query) for query in queries]
        coordinate = (str(window.source_object), int(window.start_token), DOCUMENT_TOKENS)
        prior_match = (
            coordinate in coordinates
            or document_digest in excluded_documents
            or bool(set(query_digests) & excluded_queries)
        )
        pool.append(
            {
                "pool_index": pool_index,
                "source_id": str(window.source_id),
                "source_object": str(window.source_object),
                "document_start_token": int(window.start_token),
                "document_end_token_exclusive": int(window.start_token) + DOCUMENT_TOKENS,
                "document_token_ids_sha256": document_digest,
                "query_bank_sha256": query_audit["query_bank_sha256"],
                "query_token_ids_sha256": query_digests,
                "selection_score_sha256": selection_score(str(window.source_object), int(window.start_token), document_digest),
                "excluded_as_prior_input": prior_match,
                "_document_ids": [int(value) for value in document.tolist()],
                "_queries": [
                    {
                        "request_index": request_index,
                        "source_token_offset": int(query_audit["rows"][request_index]["source_token_offset"]),
                        "token_ids": [int(value) for value in query.reshape(-1).tolist()],
                        "token_ids_sha256": query_digests[request_index],
                    }
                    for request_index, query in enumerate(queries)
                ],
            }
        )
    eligible = sorted(
        (row for row in pool if not row["excluded_as_prior_input"]),
        key=lambda row: (row["selection_score_sha256"], row["pool_index"]),
    )
    require(len(eligible) >= SELECTED_DOCUMENTS, "too few prior-input-free candidates")
    chosen = eligible[:SELECTED_DOCUMENTS]
    selected = []
    for case_index, row in enumerate(chosen):
        selected.append(
            {
                "case_index": case_index,
                "case_id": f"r30-e2e-case-{case_index}",
                "source_id": row["source_id"],
                "source_object": row["source_object"],
                "document_start_token": row["document_start_token"],
                "document_end_token_exclusive": row["document_end_token_exclusive"],
                "document_token_ids": row["_document_ids"],
                "document_token_ids_sha256": row["document_token_ids_sha256"],
                "queries": row["_queries"],
                "query_bank_sha256": row["query_bank_sha256"],
                "selection_score_sha256": row["selection_score_sha256"],
            }
        )
    public_pool = [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in pool
    ]
    manifest = {
        "schema_version": SCHEMA,
        "created_before_any_model_output": True,
        "candidate_or_reference_model_invoked": False,
        "selection_uses_candidate_or_reference_output": False,
        "dataset": {
            "name": "PG-19",
            "split": "train",
            "records": 64,
            "data_sha256": PG19_DATA_SHA256,
            "manifest_sha256": PG19_MANIFEST_SHA256,
            "longbench_consumed": False,
            "test_or_validation_consumed": False,
            "audit": data_audit,
        },
        "model": {
            "model_id": "Qwen/Qwen3.5-35B-A3B",
            "revision": MODEL_REVISION,
            "tokenizer": tokenizer_receipt(args.model_dir),
            "weights_loaded": False,
        },
        "selection": {
            "seed": SEED,
            "domain_hex": SELECTION_DOMAIN.hex(),
            "books": BOOKS,
            "document_tokens": DOCUMENT_TOKENS,
            "query_tokens": QUERY_TOKENS,
            "window_stride_tokens": WINDOW_STRIDE,
            "candidate_windows_per_book": CANDIDATE_WINDOWS,
            "requests_per_document": REQUESTS_PER_DOCUMENT,
            "query_stride_tokens": QUERY_STRIDE,
            "selected_documents": SELECTED_DOCUMENTS,
            "greedy_steps": GREEDY_STEPS,
            "windows_sha256": windows_sha256,
            "rule": "lowest SHA256(domain || seed || source_object || start || length || document_digest) among candidates absent from the frozen prior-input ledger",
            "exclusion_ledger": exclusion_receipt,
            "candidate_pool": public_pool,
        },
        "cases": selected,
        "denominators": {
            "documents": SELECTED_DOCUMENTS,
            "requests": SELECTED_DOCUMENTS * REQUESTS_PER_DOCUMENT,
            "reference_greedy_decisions": SELECTED_DOCUMENTS * REQUESTS_PER_DOCUMENT * GREEDY_STEPS,
            "candidate_arms": 4,
            "candidate_greedy_decisions": SELECTED_DOCUMENTS * REQUESTS_PER_DOCUMENT * GREEDY_STEPS * 4,
            "history_matched_full_vocab_comparisons": SELECTED_DOCUMENTS * REQUESTS_PER_DOCUMENT * GREEDY_STEPS * 4,
        },
    }
    atomic_write(args.output, canonical_bytes(manifest))
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--rr2-code-dir", type=Path, required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--pg19-data", type=Path, required=True)
    result.add_argument("--pg19-manifest", type=Path, required=True)
    result.add_argument("--exclusions", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    value = freeze(args)
    print(
        json.dumps(
            {
                "status": "frozen_output_unseen_inputs",
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "cases": [
                    {
                        "case_id": row["case_id"],
                        "source_object": row["source_object"],
                        "document_start_token": row["document_start_token"],
                        "document_token_ids_sha256": row["document_token_ids_sha256"],
                        "query_token_ids_sha256": [item["token_ids_sha256"] for item in row["queries"]],
                    }
                    for row in value["cases"]
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
