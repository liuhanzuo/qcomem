from __future__ import annotations

"""Build the pre-output PG-19 input manifest for ForkAudit RR2.

The production command is deliberately CPU-only and local-only.  It replays
the already-audited PG-19 window-selection implementation over the frozen
train64 bytes and the tokenizer at the exact public Qwen revision.  The result
binds each document, all 32 raw query chunks, the N={1,8,32} prefixes, the
oracle cell, and disjointness from the earlier capacity cohort.

This module never consumes candidate model outputs.  In particular, the
``pg19_windows_sha256`` field is the digest returned by the historical window
algorithm; it is not the SHA-256 of this JSON file.  A launcher must bind the
canonical output bytes separately as ``pg19_input_manifest_sha256``.
"""

import argparse
import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

import qcomem_joint_policy as joint_policy
import qcomem_vllm_paged_multifork_resident as resident


SCHEMA_VERSION = "qcomem-forkaudit-rr2-input-manifest-v1"
PROTOCOL = "qcomem-qwen35-forkaudit-review-revision-v1"
FORMAL_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
FORMAL_MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"

FORMAL_PG19_DATA_SHA256 = (
    "ef18893b6bfb0f4b8cb29eab85ccf2a0ef1fdb44606e9742a70405cca564e18c"
)
FORMAL_PG19_MANIFEST_SHA256 = (
    "5d789d67aa239f089e92de8a4267b86d2f1d2723d5f1370970883738f5f89a9c"
)
PRIOR_CAPACITY_MANIFEST_SHA256 = (
    "975bc6a12f43447024b889889d4156ca71c2f89b68de6157ac609b4a9687e9c0"
)
PRIOR_CAPACITY_WINDOWS_SHA256 = (
    "27ad6c687e5cab28f361bbd89dd1844788aecbecc6f2d25dbd0c60b7705a55f8"
)
FORMAL_RR2_WINDOWS_SHA256 = (
    "39bc36bb2eb04d51122e66caaebfa72367c02b43b073f072a2da240ed068c166"
)

FORMAL_BOOKS = 8
FORMAL_TRAIN_RECORDS = 64
FORMAL_DOCUMENT_TOKENS = 4095
FORMAL_QUERY_TOKENS = 32
FORMAL_QUERY_BANK_COUNT = 32
FORMAL_QUERY_BANK_STRIDE = 64
FORMAL_RESIDENT_COUNTS = (1, 8, 32)
FORMAL_WINDOW_STRIDE = 257
FORMAL_CANDIDATE_WINDOWS = 8
FORMAL_SEED = 20260817
FORMAL_GENERATION_STEPS = 8
FORMAL_FULL_LAYERS = tuple(range(3, 40, 4))

ORACLE_KV_POLICY = "vllm-q16-shared-document-reuse"
ORACLE_GDN_BASE_POLICY = "borrow-immutable-base-functional-rebind"
ORACLE_CELL_ROLE = "ownership_witness"
ORACLE_ARM_ID = f"kv={ORACLE_KV_POLICY}|gdn={ORACLE_GDN_BASE_POLICY}"

# These coordinates were fixed by the CPU-only input probe before any RR2
# candidate output existed.  The builder independently recomputes them and
# their token content from the frozen bytes; it never reads the probe artifact.
FORMAL_RR2_COORDINATES = (
    ("train/10.txt", 1542),
    ("train/10043.txt", 1028),
    ("train/10021.txt", 514),
    ("train/10009.txt", 514),
    ("train/10026.txt", 1542),
    ("train/10031.txt", 514),
    ("train/10045.txt", 1285),
    ("train/10059.txt", 1799),
)

REQUIRED_MODEL_ARTIFACTS = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "tokenizer_config.json",
)
OPTIONAL_TOKENIZER_ARTIFACTS = (
    "added_tokens.json",
    "chat_template.jinja",
    "special_tokens_map.json",
)
TOKENIZER_JSON_LAYOUT_FILES = ("tokenizer.json",)
TOKENIZER_BPE_LAYOUT_FILES = ("vocab.json", "merges.txt")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_RE = re.compile(r"^train/[0-9]+\.txt$")

QUERY_ROW_FIELDS = {
    "request_index",
    "source_token_offset",
    "query_tokens",
    "query_token_ids_sha256",
}
BANK_FIELDS = {
    "rank",
    "book_index",
    "source_id",
    "source_object",
    "window_index",
    "document_start_token",
    "document_end_token_exclusive",
    "document_token_ids_sha256",
    "query_bank_start_token",
    "query_stride_tokens",
    "query_tokens",
    "count",
    "query_bank_sha256",
    "rows",
    "manifest_sha256",
}
PREFIX_FIELDS = {
    "rank",
    "book_index",
    "source_object",
    "window_index",
    "resident_count",
    "request_indices",
    "query_rows_sha256",
}


class RR2InputManifestError(RuntimeError):
    """An input or output cannot satisfy the frozen RR2 preregistration."""


@dataclass(frozen=True)
class InputExpectations:
    pg19_data_sha256: str = FORMAL_PG19_DATA_SHA256
    pg19_manifest_sha256: str = FORMAL_PG19_MANIFEST_SHA256
    prior_manifest_sha256: str = PRIOR_CAPACITY_MANIFEST_SHA256
    prior_windows_sha256: str = PRIOR_CAPACITY_WINDOWS_SHA256
    rr2_windows_sha256: str | None = FORMAL_RR2_WINDOWS_SHA256
    rr2_coordinates: tuple[tuple[str, int], ...] | None = FORMAL_RR2_COORDINATES


FORMAL_EXPECTATIONS = InputExpectations()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RR2InputManifestError(message)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RR2InputManifestError("value is not canonical-JSON serializable") from exc


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _reject_constant(value: str) -> None:
    raise RR2InputManifestError(f"non-finite JSON constant rejected: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON key rejected: {key}")
        result[key] = value
    return result


def strict_json_loads(payload: bytes | str, *, label: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RR2InputManifestError(f"{label} is not strict JSON") from exc


def _require_sha256(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None, f"{label} is not SHA-256")
    return value


def _is_int(value: Any) -> bool:
    return type(value) is int


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    _require(value.dtype == torch.int64, "token-ID tensor dtype must be torch.int64")
    return sha256_bytes(value.view(torch.uint8).numpy().tobytes())


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_strings(item)


def _assert_path_independent(value: Any) -> None:
    for item in _walk_strings(value):
        lowered = item.lower()
        _require(not item.startswith("/"), "manifest leaked an absolute path")
        _require("file://" not in lowered, "manifest leaked a file URI")
        _require("/users/" not in lowered, "manifest leaked a user path")
        _require("/mnt/" not in lowered, "manifest leaked a mount path")


def _read_bytes(path: Path, *, label: str) -> bytes:
    try:
        return path.read_bytes()
    except OSError as exc:
        raise RR2InputManifestError(f"{label} cannot be read") from exc


def _audit_pg19_train64_bytes(
    data_bytes: bytes,
    manifest_bytes: bytes,
    *,
    expectations: InputExpectations,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_sha = sha256_bytes(data_bytes)
    manifest_sha = sha256_bytes(manifest_bytes)
    _require(data_sha == expectations.pg19_data_sha256, "PG-19 train64 data SHA-256 mismatch")
    _require(manifest_sha == expectations.pg19_manifest_sha256, "PG-19 train64 manifest SHA-256 mismatch")

    manifest = strict_json_loads(manifest_bytes, label="PG-19 train64 manifest")
    _require(isinstance(manifest, dict), "PG-19 manifest must be an object")
    _require(manifest.get("bucket") == joint_policy.PG19_BUCKET, "PG-19 bucket drift")
    _require(manifest.get("prefix") == joint_policy.PG19_PREFIX, "PG-19 split is not train/")
    _require(manifest.get("test_or_validation_objects_used") is False, "PG-19 test/validation exclusion missing")
    _require(manifest.get("jsonl_sha256") == data_sha, "PG-19 manifest does not bind train64 bytes")
    objects = manifest.get("objects")
    _require(isinstance(objects, list) and len(objects) == FORMAL_TRAIN_RECORDS, "PG-19 manifest must contain exactly 64 train objects")
    object_rows: dict[str, Mapping[str, Any]] = {}
    for row in objects:
        _require(isinstance(row, dict), "PG-19 object receipt must be an object")
        name = row.get("name")
        _require(isinstance(name, str) and _SOURCE_RE.fullmatch(name) is not None, "non-train PG-19 object rejected")
        _require(name not in object_rows, "PG-19 manifest repeats an object")
        object_rows[name] = row

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        text = data_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RR2InputManifestError("PG-19 train64 JSONL is not UTF-8") from exc
    for line_index, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = strict_json_loads(line, label=f"PG-19 train64 row {line_index}")
        _require(isinstance(row, dict), "PG-19 JSONL row must be an object")
        _require(
            not ({"dataset", "_source_index", "answers", "input", "context"} & set(row)),
            "evaluation/QA-shaped record rejected from PG-19 train64",
        )
        _require(row.get("_source_bucket") == joint_policy.PG19_BUCKET, "PG-19 row bucket drift")
        source = row.get("_source_object")
        _require(isinstance(source, str) and _SOURCE_RE.fullmatch(source) is not None, "PG-19 row is outside train/")
        _require(source not in seen, "PG-19 train64 repeats a source object")
        listed = object_rows.get(source)
        _require(listed is not None, "PG-19 train64 source is absent from its manifest")
        _require(listed.get("md5_base64") == row.get("_source_md5_base64"), "PG-19 source MD5 provenance drift")
        _require(isinstance(row.get("text"), str) and bool(row["text"]), "PG-19 source text is empty")
        seen.add(source)
        records.append(row)
    _require(len(records) == FORMAL_TRAIN_RECORDS, "PG-19 train64 JSONL must contain exactly 64 records")
    _require(seen == set(object_rows), "PG-19 train64 manifest/JSONL object sets differ")
    return records, {
        "bucket": joint_policy.PG19_BUCKET,
        "prefix": joint_policy.PG19_PREFIX,
        "records": FORMAL_TRAIN_RECORDS,
        "pg19_data_sha256": data_sha,
        "pg19_manifest_sha256": manifest_sha,
        "test_or_validation_objects_used": False,
        "longbench_consumed": False,
    }


def audit_model_tokenizer_artifacts(model_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for name in REQUIRED_MODEL_ARTIFACTS:
        path = model_dir / name
        _require(path.is_file(), f"model/tokenizer artifact missing: {name}")
    tokenizer_json_present = (model_dir / "tokenizer.json").is_file()
    vocab_present = (model_dir / "vocab.json").is_file()
    merges_present = (model_dir / "merges.txt").is_file()
    _require(
        not (tokenizer_json_present and (vocab_present or merges_present)),
        "tokenizer.json and vocab/merges layouts must not be mixed",
    )
    if tokenizer_json_present:
        selected_layout = {
            "layout_id": "consolidated-tokenizer-json-v1",
            "required_files": list(TOKENIZER_JSON_LAYOUT_FILES),
        }
        layout_files = TOKENIZER_JSON_LAYOUT_FILES
    else:
        _require(
            vocab_present and merges_present,
            "tokenizer assets require tokenizer.json or the complete vocab.json+merges.txt layout",
        )
        selected_layout = {
            "layout_id": "vocab-merges-bpe-v1",
            "required_files": list(TOKENIZER_BPE_LAYOUT_FILES),
        }
        layout_files = TOKENIZER_BPE_LAYOUT_FILES
    names = sorted(
        {
            *REQUIRED_MODEL_ARTIFACTS,
            *layout_files,
            *(name for name in OPTIONAL_TOKENIZER_ARTIFACTS if (model_dir / name).is_file()),
        },
        key=lambda value: value.encode("utf-8"),
    )
    for name in names:
        payload = _read_bytes(model_dir / name, label=f"model artifact {name}")
        rows.append({"logical_name": name, "sha256": sha256_bytes(payload), "bytes": len(payload)})
    return {
        "logical_root": "Qwen3.5-35B-A3B@59d61f3ce65a",
        "selected_tokenizer_layout": {
            **selected_layout,
            "mutually_exclusive_layout_gate_passed": True,
        },
        "artifacts": rows,
        "artifact_set_sha256": sha256_json(rows),
    }


def tokenizer_runtime_identity(tokenizer: Any, artifacts: Mapping[str, Any]) -> dict[str, Any]:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    _require(_is_int(vocab_size) and vocab_size > 0, "tokenizer vocab_size missing")
    special_ids: dict[str, int | None] = {}
    for field in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        value = getattr(tokenizer, field, None)
        _require(value is None or (_is_int(value) and value >= 0), f"tokenizer {field} drift")
        special_ids[field] = value
    return {
        "class_module": type(tokenizer).__module__,
        "class_qualname": type(tokenizer).__qualname__,
        "vocab_size": vocab_size,
        "is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "special_token_ids": special_ids,
        "artifact_set_sha256": _require_sha256(artifacts.get("artifact_set_sha256"), "tokenizer artifact-set digest"),
    }


def parse_prior_capacity_manifest(
    payload: bytes,
    *,
    expectations: InputExpectations,
) -> dict[str, Any]:
    raw_sha = sha256_bytes(payload)
    _require(raw_sha == expectations.prior_manifest_sha256, "prior capacity manifest byte SHA-256 mismatch")
    value = strict_json_loads(payload, label="prior capacity manifest")
    _require(isinstance(value, dict), "prior capacity manifest must be an object")
    _require(value.get("schema_version") == 1, "prior capacity manifest schema drift")
    _require(
        value.get("protocol")
        == "same-vllm-unified-attention-q16-multifork-resident-v1",
        "prior capacity protocol drift",
    )
    frozen = value.get("frozen_identity")
    _require(isinstance(frozen, dict), "prior capacity frozen identity missing")
    _require(
        frozen.get("pg19_windows_sha256") == expectations.prior_windows_sha256,
        "prior capacity windows SHA-256 mismatch",
    )
    banks = value.get("frozen_query_banks")
    _require(isinstance(banks, list) and len(banks) == FORMAL_BOOKS, "prior capacity manifest must expose eight coordinates")
    coordinates: list[dict[str, Any]] = []
    for rank, bank in enumerate(banks):
        _require(isinstance(bank, dict), "prior capacity query bank must be an object")
        source = bank.get("source_object")
        start = bank.get("document_start_token")
        end = bank.get("document_end_token_exclusive")
        _require(isinstance(source, str) and _SOURCE_RE.fullmatch(source) is not None, "prior capacity source drift")
        _require(_is_int(start) and start >= 0 and _is_int(end) and end > start, "prior capacity document coordinate drift")
        _require(start % FORMAL_WINDOW_STRIDE == 0, "prior capacity window is off the historical stride")
        coordinates.append(
            {
                "rank": rank,
                "source_object": source,
                "window_index": start // FORMAL_WINDOW_STRIDE,
                "document_start_token": start,
                "document_end_token_exclusive": end,
                "document_length": end - start,
            }
        )
    pairs = {(row["source_object"], row["document_start_token"]) for row in coordinates}
    triples = {
        (row["source_object"], row["document_start_token"], row["document_length"])
        for row in coordinates
    }
    _require(len(pairs) == FORMAL_BOOKS and len(triples) == FORMAL_BOOKS, "prior capacity coordinates are not unique")
    return {
        "cohort_id": "gpu-qwen35-vllm-paged-multifork-resident-20260814a",
        "protocol_manifest_sha256": raw_sha,
        "pg19_windows_sha256": expectations.prior_windows_sha256,
        "coordinate_definition": ["source_object", "document_start_token", "document_length"],
        "coordinates": coordinates,
        "coordinates_sha256": sha256_json(coordinates),
    }


def _algorithm_source_digest() -> dict[str, str]:
    window_source = "\n".join(
        inspect.getsource(value)
        for value in (
            joint_policy._selection_key,
            joint_policy._window_choice,
            joint_policy.build_pg19_calibration_windows,
        )
    )
    query_source = inspect.getsource(resident.build_pg19_train_query_bank)
    return {
        "window_algorithm_source_sha256": sha256_bytes(window_source.encode("utf-8")),
        "query_bank_algorithm_source_sha256": sha256_bytes(query_source.encode("utf-8")),
    }


def _bank_self_hash(bank: Mapping[str, Any]) -> str:
    return sha256_json({key: value for key, value in bank.items() if key != "manifest_sha256"})


def _oracle_selection(rank: int, bank: Mapping[str, Any]) -> dict[str, Any]:
    layer_index = FORMAL_FULL_LAYERS[rank]
    round_index = rank % FORMAL_GENERATION_STEPS
    return {
        "selection_rule_id": "rank-frozen-heldout-post-rope-v1",
        "rank": rank,
        "book_index": rank,
        "source_object": bank["source_object"],
        "window_index": bank["window_index"],
        "document_start_token": bank["document_start_token"],
        "document_length": FORMAL_DOCUMENT_TOKENS,
        "document_token_ids_sha256": bank["document_token_ids_sha256"],
        "layer_index": layer_index,
        "request_index": 0,
        "round_index": round_index,
        "sample_id": f"rr2-rank-{rank}-layer-{layer_index}-round-{round_index}",
        "kv_policy": ORACLE_KV_POLICY,
        "gdn_base_policy": ORACLE_GDN_BASE_POLICY,
        "cell_role": ORACLE_CELL_ROLE,
        "arm_id": ORACLE_ARM_ID,
        "oracle_cell_id": f"rank-{rank}-N-1-{ORACLE_ARM_ID}-ownership-witness",
        "held_out_from_threshold_calibration": True,
        "locked_before_candidate_outputs": True,
    }


def _prefix_rows(bank: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for resident_count in FORMAL_RESIDENT_COUNTS:
        rows = bank["rows"][:resident_count]
        result.append(
            {
                "rank": bank["rank"],
                "book_index": bank["book_index"],
                "source_object": bank["source_object"],
                "window_index": bank["window_index"],
                "resident_count": resident_count,
                "request_indices": list(range(resident_count)),
                "query_rows_sha256": sha256_json(rows),
            }
        )
    return result


def build_rr2_input_manifest(
    records: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    data_audit: Mapping[str, Any],
    model_artifacts: Mapping[str, Any],
    prior_manifest_bytes: bytes,
    expectations: InputExpectations = FORMAL_EXPECTATIONS,
) -> dict[str, Any]:
    _require(not torch.cuda.is_initialized(), "RR2 input builder must start before CUDA initialization")
    _require(len(records) == FORMAL_TRAIN_RECORDS, "RR2 builder requires the audited train64 records")
    prior = parse_prior_capacity_manifest(prior_manifest_bytes, expectations=expectations)

    try:
        windows, windows_sha = joint_policy.build_pg19_calibration_windows(
            records,
            tokenizer,
            books=FORMAL_BOOKS,
            document_tokens=FORMAL_DOCUMENT_TOKENS,
            query_tokens=FORMAL_QUERY_TOKENS,
            stride=FORMAL_WINDOW_STRIDE,
            candidate_windows_per_book=FORMAL_CANDIDATE_WINDOWS,
            seed=FORMAL_SEED,
        )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise RR2InputManifestError(f"RR2 window selection rejected: {exc}") from exc
    _require(windows_sha != prior["pg19_windows_sha256"], "new PG-19 windows SHA repeats the prior capacity cohort")
    if expectations.rr2_windows_sha256 is not None:
        _require(windows_sha == expectations.rr2_windows_sha256, "RR2 PG-19 windows SHA differs from the pre-output probe")

    observed_coordinates = tuple((window.source_object, int(window.start_token)) for window in windows)
    if expectations.rr2_coordinates is not None:
        _require(observed_coordinates == expectations.rr2_coordinates, "RR2 PG-19 coordinate order differs from preregistration")

    prior_pairs = {
        (row["source_object"], row["document_start_token"])
        for row in prior["coordinates"]
    }
    prior_triples = {
        (row["source_object"], row["document_start_token"], row["document_length"])
        for row in prior["coordinates"]
    }
    window_rows: list[dict[str, Any]] = []
    banks: list[dict[str, Any]] = []
    all_query_digests: list[str] = []
    for rank, window in enumerate(windows):
        start = int(window.start_token)
        end = start + FORMAL_DOCUMENT_TOKENS
        window_index = start // FORMAL_WINDOW_STRIDE
        _require(start == window_index * FORMAL_WINDOW_STRIDE, "RR2 window is off the frozen stride")
        document_sha = _tensor_sha256(window.document_ids)
        adjacent_query_sha = _tensor_sha256(window.query_ids)
        _require(tuple(window.document_ids.shape) == (FORMAL_DOCUMENT_TOKENS,), "RR2 document token length drift")
        _require(tuple(window.query_ids.shape) == (FORMAL_QUERY_TOKENS,), "RR2 adjacent query token length drift")
        _require((window.source_object, start) not in prior_pairs, "RR2 window source/start overlaps the prior capacity cohort")
        _require(
            (window.source_object, start, FORMAL_DOCUMENT_TOKENS) not in prior_triples,
            "RR2 window coordinate overlaps the prior capacity cohort",
        )

        try:
            query_tensors, query_audit = resident.build_pg19_train_query_bank(
                records,
                tokenizer,
                window,
                document_tokens=FORMAL_DOCUMENT_TOKENS,
                query_tokens=FORMAL_QUERY_TOKENS,
                count=FORMAL_QUERY_BANK_COUNT,
                query_stride=FORMAL_QUERY_BANK_STRIDE,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RR2InputManifestError(f"RR2 query-bank construction rejected: {exc}") from exc
        rows: list[dict[str, Any]] = []
        for request_index, (query, row) in enumerate(zip(query_tensors, query_audit["rows"])):
            normalized = {
                "request_index": request_index,
                "source_token_offset": int(row["source_token_offset"]),
                "query_tokens": FORMAL_QUERY_TOKENS,
                "query_token_ids_sha256": _tensor_sha256(query),
            }
            _require(normalized == row, "query-bank helper audit/tensor disagreement")
            rows.append(normalized)
            all_query_digests.append(normalized["query_token_ids_sha256"])

        bank: dict[str, Any] = {
            "rank": rank,
            "book_index": rank,
            "source_id": str(window.source_id),
            "source_object": window.source_object,
            "window_index": window_index,
            "document_start_token": start,
            "document_end_token_exclusive": end,
            "document_token_ids_sha256": document_sha,
            "query_bank_start_token": int(query_audit["query_bank_start_token"]),
            "query_stride_tokens": FORMAL_QUERY_BANK_STRIDE,
            "query_tokens": FORMAL_QUERY_TOKENS,
            "count": FORMAL_QUERY_BANK_COUNT,
            "query_bank_sha256": str(query_audit["query_bank_sha256"]),
            "rows": rows,
        }
        bank["manifest_sha256"] = _bank_self_hash(bank)
        banks.append(bank)
        window_rows.append(
            {
                "rank": rank,
                "book_index": rank,
                "source_id": str(window.source_id),
                "source_object": window.source_object,
                "window_index": window_index,
                "document_start_token": start,
                "document_end_token_exclusive": end,
                "document_length": FORMAL_DOCUMENT_TOKENS,
                "document_token_ids_sha256": document_sha,
                "adjacent_calibration_query_start_token": end,
                "adjacent_calibration_query_end_token_exclusive": end + FORMAL_QUERY_TOKENS,
                "adjacent_calibration_query_token_ids_sha256": adjacent_query_sha,
                "query_bank_manifest_sha256": bank["manifest_sha256"],
                "absent_from_prior_capacity_source_start_pairs": True,
                "absent_from_prior_capacity_coordinates": True,
            }
        )

    _require(len({row["source_object"] for row in window_rows}) == FORMAL_BOOKS, "RR2 windows do not use eight unique books")
    _require(len(set(all_query_digests)) == FORMAL_BOOKS * FORMAL_QUERY_BANK_COUNT, "RR2 query chunks are not globally unique")
    oracle_plan = [_oracle_selection(rank, bank) for rank, bank in enumerate(banks)]
    prefixes = [row for bank in banks for row in _prefix_rows(bank)]
    tokenizer_identity = tokenizer_runtime_identity(tokenizer, model_artifacts)

    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "manifest_role": "pre-output deterministic PG19 inputs; contains no candidate outputs",
        "model": {
            "model_id": FORMAL_MODEL_ID,
            "model_revision": FORMAL_MODEL_REVISION,
            "local_files_only": True,
            "model_and_tokenizer_artifacts": dict(model_artifacts),
            "tokenizer_runtime_identity": tokenizer_identity,
        },
        "dataset": dict(data_audit),
        "window_algorithm": {
            "implementation": "qcomem_joint_policy.build_pg19_calibration_windows",
            "books": FORMAL_BOOKS,
            "seed": FORMAL_SEED,
            "document_tokens": FORMAL_DOCUMENT_TOKENS,
            "adjacent_query_tokens": FORMAL_QUERY_TOKENS,
            "window_stride_tokens": FORMAL_WINDOW_STRIDE,
            "candidate_windows_per_book": FORMAL_CANDIDATE_WINDOWS,
            **_algorithm_source_digest(),
        },
        "query_bank_protocol": {
            "implementation": "qcomem_vllm_paged_multifork_resident.build_pg19_train_query_bank",
            "count": FORMAL_QUERY_BANK_COUNT,
            "query_tokens": FORMAL_QUERY_TOKENS,
            "query_stride_tokens": FORMAL_QUERY_BANK_STRIDE,
            "resident_counts": list(FORMAL_RESIDENT_COUNTS),
            "bank_starts_after_adjacent_calibration_query": True,
        },
        "pg19_windows_sha256": windows_sha,
        "pg19_input_manifest_sha256_contract": "SHA-256 of canonical output bytes including one trailing LF; bound externally because a JSON object cannot contain its own byte digest",
        "prior_capacity_cohort": prior,
        "windows": window_rows,
        "frozen_query_banks": banks,
        "n_prefixes_by_rank": prefixes,
        "oracle_selection_plan": oracle_plan,
        "oracle_selection_plan_sha256": sha256_json(oracle_plan),
        "invariants": {
            "eight_unique_train_books": True,
            "all_256_query_token_chunks_globally_distinct": True,
            "query_chunks_pairwise_nonoverlapping_within_book": True,
            "query_chunks_disjoint_from_document_and_adjacent_query": True,
            "resident_prefixes_exact_and_nested": True,
            "new_source_start_pairs_disjoint_from_prior_capacity": True,
            "new_coordinates_disjoint_from_prior_capacity": True,
            "new_windows_sha_differs_from_prior_capacity": True,
            "oracle_selection_locked_before_candidate_outputs": True,
        },
        "build_audit": {
            "candidate_outputs_consumed": False,
            "network_access_required": False,
            "cuda_initialized": torch.cuda.is_initialized(),
            "path_independent_serialization": True,
        },
    }
    validate_rr2_input_manifest(manifest, expectations=expectations)
    _assert_path_independent(manifest)
    _require(not torch.cuda.is_initialized(), "RR2 input builder initialized CUDA")
    return manifest


def validate_rr2_input_manifest(
    value: Any,
    *,
    expectations: InputExpectations = FORMAL_EXPECTATIONS,
) -> dict[str, Any]:
    _require(isinstance(value, dict), "RR2 input manifest must be an object")
    _require(value.get("schema_version") == SCHEMA_VERSION, "RR2 input schema drift")
    _require(value.get("protocol") == PROTOCOL, "RR2 protocol drift")
    _require(value.get("pg19_windows_sha256") != expectations.prior_windows_sha256, "RR2 windows digest repeats the prior cohort")
    if expectations.rr2_windows_sha256 is not None:
        _require(value.get("pg19_windows_sha256") == expectations.rr2_windows_sha256, "RR2 windows digest drift")
    model = value.get("model")
    _require(isinstance(model, dict), "RR2 model binding missing")
    _require(model.get("model_id") == FORMAL_MODEL_ID and model.get("model_revision") == FORMAL_MODEL_REVISION, "RR2 model/revision drift")
    dataset = value.get("dataset")
    _require(isinstance(dataset, dict), "RR2 dataset binding missing")
    _require(dataset.get("pg19_data_sha256") == expectations.pg19_data_sha256, "RR2 PG-19 data binding drift")
    _require(dataset.get("pg19_manifest_sha256") == expectations.pg19_manifest_sha256, "RR2 PG-19 manifest binding drift")
    _require(dataset.get("records") == FORMAL_TRAIN_RECORDS and dataset.get("prefix") == "train/", "RR2 train64 binding drift")

    prior = value.get("prior_capacity_cohort")
    _require(isinstance(prior, dict), "RR2 prior-cohort binding missing")
    _require(prior.get("protocol_manifest_sha256") == expectations.prior_manifest_sha256, "RR2 prior manifest binding drift")
    _require(prior.get("pg19_windows_sha256") == expectations.prior_windows_sha256, "RR2 prior windows binding drift")
    prior_rows = prior.get("coordinates")
    _require(isinstance(prior_rows, list) and len(prior_rows) == FORMAL_BOOKS, "RR2 prior coordinates drift")
    _require(prior.get("coordinates_sha256") == sha256_json(prior_rows), "RR2 prior-coordinate SHA drift")
    prior_pairs = {(row["source_object"], row["document_start_token"]) for row in prior_rows}
    prior_triples = {(row["source_object"], row["document_start_token"], row["document_length"]) for row in prior_rows}

    windows = value.get("windows")
    banks = value.get("frozen_query_banks")
    _require(isinstance(windows, list) and len(windows) == FORMAL_BOOKS, "RR2 must freeze eight windows")
    _require(isinstance(banks, list) and len(banks) == FORMAL_BOOKS, "RR2 must freeze eight query banks")
    all_query_digests: list[str] = []
    for rank, (window, bank) in enumerate(zip(windows, banks)):
        _require(isinstance(window, dict) and isinstance(bank, dict), "RR2 window/bank row must be an object")
        _require(set(bank) == BANK_FIELDS, "RR2 query-bank exact schema drift")
        _require(window.get("rank") == rank and window.get("book_index") == rank, "RR2 window rank order drift")
        _require(bank.get("rank") == rank and bank.get("book_index") == rank, "RR2 bank rank order drift")
        for field in ("source_id", "source_object", "window_index", "document_start_token", "document_end_token_exclusive", "document_token_ids_sha256"):
            _require(window.get(field) == bank.get(field), f"RR2 window/bank {field} binding drift")
        source = bank.get("source_object")
        start = bank.get("document_start_token")
        end = bank.get("document_end_token_exclusive")
        _require(isinstance(source, str) and _SOURCE_RE.fullmatch(source) is not None, "RR2 bank source drift")
        _require(_is_int(start) and _is_int(end) and end - start == FORMAL_DOCUMENT_TOKENS, "RR2 document coordinate drift")
        _require(bank.get("window_index") * FORMAL_WINDOW_STRIDE == start, "RR2 window-index binding drift")
        _require_sha256(bank.get("document_token_ids_sha256"), "RR2 document token digest")
        _require(bank.get("query_bank_start_token") == end + FORMAL_QUERY_TOKENS, "RR2 query bank does not start after the adjacent query")
        _require(bank.get("query_stride_tokens") == FORMAL_QUERY_BANK_STRIDE, "RR2 query stride drift")
        _require(bank.get("query_tokens") == FORMAL_QUERY_TOKENS and bank.get("count") == FORMAL_QUERY_BANK_COUNT, "RR2 query-bank geometry drift")
        _require_sha256(bank.get("query_bank_sha256"), "RR2 query-bank digest")
        _require(bank.get("manifest_sha256") == _bank_self_hash(bank), "RR2 query-bank self hash drift")
        _require(window.get("query_bank_manifest_sha256") == bank.get("manifest_sha256"), "RR2 window/query-bank manifest binding drift")
        _require((source, start) not in prior_pairs, "RR2 source/start overlaps prior capacity")
        _require((source, start, FORMAL_DOCUMENT_TOKENS) not in prior_triples, "RR2 coordinate overlaps prior capacity")
        rows = bank.get("rows")
        _require(isinstance(rows, list) and len(rows) == FORMAL_QUERY_BANK_COUNT, "RR2 query bank must contain 32 rows")
        previous_end = end + FORMAL_QUERY_TOKENS
        for request_index, row in enumerate(rows):
            _require(
                isinstance(row, dict)
                and set(row) == QUERY_ROW_FIELDS,
                "RR2 query row schema drift",
            )
            _require(row["request_index"] == request_index and row["query_tokens"] == FORMAL_QUERY_TOKENS, "RR2 query row order/length drift")
            expected_offset = bank["query_bank_start_token"] + request_index * FORMAL_QUERY_BANK_STRIDE
            _require(row["source_token_offset"] == expected_offset, "RR2 query offset drift")
            _require(row["source_token_offset"] >= previous_end, "RR2 query chunks overlap")
            previous_end = row["source_token_offset"] + FORMAL_QUERY_TOKENS
            all_query_digests.append(_require_sha256(row["query_token_ids_sha256"], "RR2 query token digest"))
    _require(len({bank["source_object"] for bank in banks}) == FORMAL_BOOKS, "RR2 books are not unique")
    _require(len(set(all_query_digests)) == FORMAL_BOOKS * FORMAL_QUERY_BANK_COUNT, "RR2 queries are not globally unique")
    if expectations.rr2_coordinates is not None:
        observed = tuple((bank["source_object"], bank["document_start_token"]) for bank in banks)
        _require(observed == expectations.rr2_coordinates, "RR2 coordinate preregistration drift")

    prefixes = value.get("n_prefixes_by_rank")
    _require(isinstance(prefixes, list) and len(prefixes) == FORMAL_BOOKS * len(FORMAL_RESIDENT_COUNTS), "RR2 N-prefix table drift")
    expected_prefixes = [row for bank in banks for row in _prefix_rows(bank)]
    _require(all(isinstance(row, dict) and set(row) == PREFIX_FIELDS for row in prefixes), "RR2 N-prefix row schema drift")
    _require(prefixes == expected_prefixes, "RR2 N-prefix content drift")

    plan = value.get("oracle_selection_plan")
    expected_plan = [_oracle_selection(rank, bank) for rank, bank in enumerate(banks)]
    _require(plan == expected_plan, "RR2 oracle selection plan drift")
    _require(value.get("oracle_selection_plan_sha256") == sha256_json(expected_plan), "RR2 oracle selection SHA drift")
    _assert_path_independent(value)
    return value


def build_from_paths(
    *,
    pg19_data: Path,
    pg19_manifest: Path,
    prior_capacity_manifest: Path,
    model_dir: Path,
    tokenizer: Any,
    expectations: InputExpectations = FORMAL_EXPECTATIONS,
) -> dict[str, Any]:
    data_bytes = _read_bytes(pg19_data, label="PG-19 train64 data")
    manifest_bytes = _read_bytes(pg19_manifest, label="PG-19 train64 manifest")
    records, data_audit = _audit_pg19_train64_bytes(
        data_bytes,
        manifest_bytes,
        expectations=expectations,
    )
    artifacts = audit_model_tokenizer_artifacts(model_dir)
    prior_bytes = _read_bytes(prior_capacity_manifest, label="prior capacity manifest")
    return build_rr2_input_manifest(
        records,
        tokenizer,
        data_audit=data_audit,
        model_artifacts=artifacts,
        prior_manifest_bytes=prior_bytes,
        expectations=expectations,
    )


def _atomic_json(path: Path, value: Any) -> dict[str, Any]:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return {"logical_name": path.name, "sha256": sha256_bytes(payload), "bytes": len(payload)}


def load_local_tokenizer(model_dir: Path) -> Any:
    """Load only the frozen local tokenizer artifacts; network fallback is forbidden."""

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(model_dir),
        revision=FORMAL_MODEL_REVISION,
        local_files_only=True,
        trust_remote_code=False,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pg19-data", type=Path, required=True)
    parser.add_argument("--pg19-manifest", type=Path, required=True)
    parser.add_argument("--prior-capacity-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-id", default=FORMAL_MODEL_ID)
    parser.add_argument("--model-revision", default=FORMAL_MODEL_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-query-banks-output", type=Path, required=True)
    parser.add_argument("--oracle-selection-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _require(args.model_id == FORMAL_MODEL_ID, "formal RR2 model ID drift")
    _require(args.model_revision == FORMAL_MODEL_REVISION, "formal RR2 model revision drift")
    outputs = {
        args.output.resolve(),
        args.frozen_query_banks_output.resolve(),
        args.oracle_selection_output.resolve(),
    }
    _require(len(outputs) == 3, "RR2 manifest outputs must be three distinct files")
    _require(not torch.cuda.is_initialized(), "RR2 CLI started after CUDA initialization")
    tokenizer = load_local_tokenizer(args.model_dir)
    manifest = build_from_paths(
        pg19_data=args.pg19_data,
        pg19_manifest=args.pg19_manifest,
        prior_capacity_manifest=args.prior_capacity_manifest,
        model_dir=args.model_dir,
        tokenizer=tokenizer,
    )
    receipts = {
        "pg19_input_manifest": _atomic_json(args.output, manifest),
        "frozen_query_banks": _atomic_json(args.frozen_query_banks_output, manifest["frozen_query_banks"]),
        "oracle_selection_plan": _atomic_json(args.oracle_selection_output, manifest["oracle_selection_plan"]),
    }
    _require(not torch.cuda.is_initialized(), "RR2 CLI initialized CUDA")
    print(
        json.dumps(
            {
                "status": "rr2_input_preregistration_built",
                "pg19_windows_sha256": manifest["pg19_windows_sha256"],
                "pg19_input_manifest_sha256": receipts["pg19_input_manifest"]["sha256"],
                "receipts": receipts,
                "cuda_initialized": False,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


__all__ = [
    "FORMAL_EXPECTATIONS",
    "FORMAL_RR2_COORDINATES",
    "FORMAL_RR2_WINDOWS_SHA256",
    "InputExpectations",
    "RR2InputManifestError",
    "build_from_paths",
    "build_rr2_input_manifest",
    "canonical_json_bytes",
    "load_local_tokenizer",
    "parse_prior_capacity_manifest",
    "sha256_bytes",
    "sha256_json",
    "validate_rr2_input_manifest",
]


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RR2InputManifestError as exc:
        raise SystemExit(f"RR2 input preregistration rejected: {exc}") from exc
