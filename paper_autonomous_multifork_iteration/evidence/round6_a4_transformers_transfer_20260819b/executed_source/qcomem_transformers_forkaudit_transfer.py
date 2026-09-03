from __future__ import annotations

"""Runtime-neutral evidence helpers for the Transformers-cache transfer.

This module intentionally does not import the existing ForkAudit producer,
mutant, oracle, storage-witness, or vLLM ownership implementations.  It turns
ordinary Python objects and Torch tensors into canonical, pointer-free
receipts that can be replayed without the GPU runtime.
"""

import hashlib
import json
import math
import os
import re
import stat as statlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, MutableMapping, Sequence

import torch


PROTOCOL = "forkaudit-transformers-cache-transfer-v1"
STATIC_SCHEMA = "forkaudit-transformers-cache-transfer-static-v1"
SHARD_SCHEMA = "forkaudit-transformers-cache-transfer-shard-v1"
AGGREGATE_SCHEMA = "forkaudit-transformers-cache-transfer-aggregate-v1"
SOURCE_SCHEMA = "forkaudit-transformers-cache-transfer-source-manifest-v1"
WORLD_SIZE = 8
FANOUTS = (1, 2)

TARGET_CONTRACT: tuple[dict[str, Any], ...] = (
    {
        "target_index": 1,
        "target": "frozen_identity",
        "predicate_id": "FROZEN_INPUT_SOURCE_MODEL_BINDINGS",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 2,
        "target": "prefix_immutability",
        "predicate_id": "PERSISTENT_PREFIX_CONTENT_UNCHANGED",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 3,
        "target": "private_ownership",
        "predicate_id": "ALL_MUTABLE_CACHE_STORAGE_PAIRWISE_DISJOINT",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 4,
        "target": "tail_safe_append",
        "predicate_id": "PAGED_PARTIAL_TAIL_COPY_BEFORE_APPEND",
        "applicability": "not_applicable",
        "maximum_status": "not_applicable",
    },
    {
        "target_index": 5,
        "target": "dispatch_provenance",
        "predicate_id": "SAME_TRANSFORMERS_ADAPTER_AND_LAYER_CALLABLES",
        "applicability": "applicable",
        "maximum_status": "partial",
    },
    {
        "target_index": 6,
        "target": "cross_arm_equivalence",
        "predicate_id": "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK",
        "applicability": "applicable",
        "maximum_status": "full",
    },
    {
        "target_index": 7,
        "target": "cross_n_prefix_consistency",
        "predicate_id": "NESTED_QUERY_PREFIX_INVARIANT_N1_N2",
        "applicability": "applicable",
        "maximum_status": "full",
    },
)

FAULT_CONTRACT: tuple[dict[str, str], ...] = (
    {
        "fault_id": "T1",
        "fault": "common_mode_prefix_corruption",
        "expected_predicate": "INDEPENDENT_DENSE_SEMANTIC_ORACLE",
        "expected_outcome": "detected_expected_predicate",
        "matched_clean": "the same two arms without residual corruption pass the dense oracle",
    },
    {
        "fault_id": "T2",
        "fault": "cross_request_mutable_cache_alias",
        "expected_predicate": "ALL_MUTABLE_CACHE_STORAGE_PAIRWISE_DISJOINT",
        "expected_outcome": "detected_expected_predicate",
        "matched_clean": "two independently forked request caches are pairwise storage-disjoint",
    },
    {
        "fault_id": "T3",
        "fault": "position_current_length_drift",
        "expected_predicate": "POSITION_CURRENT_LENGTH_CANONICAL",
        "expected_outcome": "detected_expected_predicate",
        "matched_clean": "current_length equals document_length before the query continuation",
    },
    {
        "fault_id": "T4",
        "fault": "packed_state_content_corruption",
        "expected_predicate": "PACKED_STATE_IMMUTABILITY",
        "expected_outcome": "detected_expected_predicate",
        "matched_clean": "an untouched Q16 packed state retains its predeclared content digest",
    },
    {
        "fault_id": "T5",
        "fault": "live_lower_cache_value_corruption",
        "expected_predicate": "DOWNSTREAM_OUTPUT_CONSISTENCY",
        "expected_outcome": "detected_expected_predicate",
        "matched_clean": "the corresponding unmodified persistent fork completes and matches both the materialized arm and dense oracle",
    },
)

_SHA_RE = re.compile(r"[0-9a-f]{64}")


class TransferEvidenceError(RuntimeError):
    """An input or evidence-integrity failure, not a scientific negative."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TransferEvidenceError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_canonical_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(canonical_json_bytes(value) + b"\n")
    temporary.replace(path)


def load_bound_json(path: Path, expected_sha256: str, label: str) -> Any:
    require(_SHA_RE.fullmatch(expected_sha256) is not None, f"{label} expected SHA is invalid")
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"{label} raw SHA-256 mismatch")
    return json.loads(raw)


def tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def tensor_receipt(value: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "content_sha256": sha256_bytes(tensor_bytes(value)),
        "finite": bool(torch.isfinite(value).all().item()) if value.is_floating_point() else True,
    }


@dataclass
class TensorSlot:
    path: str
    parent: Any
    key: Any
    tensor: torch.Tensor

    @property
    def replaceable(self) -> bool:
        return (
            isinstance(self.parent, (MutableMapping, list))
            or (
                hasattr(self.parent, "__dict__")
                and isinstance(self.key, str)
                and self.key in vars(self.parent)
            )
        )

    def replace(self, value: torch.Tensor) -> None:
        require(self.replaceable, f"tensor slot is not replaceable: {self.path}")
        if isinstance(self.parent, MutableMapping):
            self.parent[self.key] = value
        elif isinstance(self.parent, list):
            self.parent[self.key] = value
        else:
            setattr(self.parent, self.key, value)
        self.tensor = value


def iter_tensor_slots(root: Any) -> Iterator[TensorSlot]:
    """Yield replaceable tensor leaves through lists, dicts, and attributes."""

    visited: set[int] = set()

    def visit(value: Any, path: str, parent: Any, key: Any) -> Iterator[TensorSlot]:
        if isinstance(value, torch.Tensor):
            if parent is not None:
                yield TensorSlot(path=path, parent=parent, key=key, tensor=value)
            return
        object_id = id(value)
        if object_id in visited:
            return
        visited.add(object_id)
        if isinstance(value, Mapping):
            for child_key in sorted(value, key=lambda item: str(item)):
                yield from visit(
                    value[child_key], f"{path}/{child_key}", value, child_key
                )
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from visit(child, f"{path}/{index}", value, index)
        elif hasattr(value, "__dict__"):
            for name in sorted(vars(value)):
                yield from visit(getattr(value, name), f"{path}/{name}", value, name)

    yield from visit(root, "root", None, None)


def tensor_tree_receipt(root: Any) -> dict[str, Any]:
    rows = [
        {"path": slot.path, **tensor_receipt(slot.tensor)}
        for slot in iter_tensor_slots(root)
    ]
    return {
        "tensor_count": len(rows),
        "rows": rows,
        "content_sha256": sha256_json(rows),
    }


def state_content_receipt(state: Any) -> dict[str, Any]:
    cache = tensor_tree_receipt(state.cache)
    residual = tensor_receipt(state.document_residual)
    identity = {
        "depth": int(state.depth),
        "document_length": int(state.document_length),
        "current_length": int(state.current_length),
        "document_residual": residual,
        "cache_content_sha256": cache["content_sha256"],
        "cache_tensor_count": cache["tensor_count"],
    }
    return {**identity, "state_content_sha256": sha256_json(identity)}


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


def storage_inventory(root: Any, *, salt: str, role: str) -> dict[str, Any]:
    require(bool(salt), "storage receipt salt must be non-empty")
    rows = []
    for slot in iter_tensor_slots(root):
        tensor = slot.tensor
        require(tensor.is_contiguous(), f"non-contiguous storage witness is unsupported: {slot.path}")
        device, pointer, storage_bytes = _storage_key(tensor)
        opaque_id = sha256_bytes(
            f"{salt}|{device}|{pointer}|{storage_bytes}".encode("utf-8")
        )
        view_start = int(tensor.storage_offset() * tensor.element_size())
        view_nbytes = int(tensor.numel() * tensor.element_size())
        view_end = view_start + view_nbytes
        require(0 <= view_start <= view_end <= storage_bytes, "tensor view exceeds storage bounds")
        rows.append(
            {
                "path": slot.path,
                "role": role,
                "device": device,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "contiguous": True,
                "storage_bytes": storage_bytes,
                "view_offset_bytes": view_start,
                "view_nbytes": view_nbytes,
                "view_start_bytes": view_start,
                "view_end_bytes": view_end,
                "storage_id_sha256": opaque_id,
            }
        )
    return {
        "role": role,
        "storage_salt_domain_sha256": sha256_bytes(salt.encode("utf-8")),
        "tensor_rows": len(rows),
        "rows": rows,
        "inventory_sha256": sha256_json(rows),
    }


def storage_ids(inventory: Mapping[str, Any]) -> set[str]:
    rows = inventory.get("rows")
    require(isinstance(rows, list), "storage inventory rows are missing")
    result = set()
    for row in rows:
        require(isinstance(row, dict), "storage inventory row is not an object")
        digest = row.get("storage_id_sha256")
        require(isinstance(digest, str) and _SHA_RE.fullmatch(digest) is not None, "invalid storage ID")
        result.add(digest)
    return result


def _overlapping_storage_ranges(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Return normalized non-empty view-range intersections on shared storage."""

    overlaps: list[dict[str, Any]] = []
    for left_row in left["rows"]:
        for right_row in right["rows"]:
            if left_row["storage_id_sha256"] != right_row["storage_id_sha256"]:
                continue
            left_start, left_end = left_row["view_start_bytes"], left_row["view_end_bytes"]
            right_start, right_end = right_row["view_start_bytes"], right_row["view_end_bytes"]
            start, end = max(left_start, right_start), min(left_end, right_end)
            # Empty views occupy no bytes and therefore never alias.
            if start < end:
                overlaps.append(
                    {
                        "storage_id_sha256": left_row["storage_id_sha256"],
                        "left_path": left_row["path"],
                        "right_path": right_row["path"],
                        "intersection_start_bytes": start,
                        "intersection_end_bytes": end,
                    }
                )
    return sorted(
        overlaps,
        key=lambda row: (
            row["storage_id_sha256"],
            row["left_path"],
            row["right_path"],
            row["intersection_start_bytes"],
            row["intersection_end_bytes"],
        ),
    )


def disjointness_receipt(
    inventories: Sequence[Mapping[str, Any]],
    *,
    forbidden: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    domains = {
        inventory.get("storage_salt_domain_sha256")
        for inventory in (*inventories, *forbidden)
    }
    require(len(domains) <= 1 and None not in domains, "storage inventories use different salt domains")
    pair_rows = []
    tensor_pair_comparisons = 0
    passed = True
    for left_index, left in enumerate(inventories):
        for right_index in range(left_index + 1, len(inventories)):
            right = inventories[right_index]
            overlap = _overlapping_storage_ranges(left, right)
            tensor_pair_comparisons += len(left["rows"]) * len(right["rows"])
            pair_rows.append(
                {
                    "left_role": left["role"],
                    "right_role": right["role"],
                    "overlap_ranges": overlap,
                    "disjoint": not overlap,
                }
            )
            passed = passed and not overlap
        for right in forbidden:
            overlap = _overlapping_storage_ranges(left, right)
            tensor_pair_comparisons += len(left["rows"]) * len(right["rows"])
            pair_rows.append(
                {
                    "left_role": left["role"],
                    "right_role": right["role"],
                    "overlap_ranges": overlap,
                    "disjoint": not overlap,
                }
            )
            passed = passed and not overlap
    return {
        "predicate_id": "ALL_MUTABLE_CACHE_STORAGE_PAIRWISE_DISJOINT",
        "passed": passed,
        "comparisons": pair_rows,
        "comparison_count": len(pair_rows),
        "tensor_pair_comparison_count": tensor_pair_comparisons,
        "receipt_sha256": sha256_json(pair_rows),
    }


def semantic_key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "request_index",
            "query_token_ids_sha256",
            "generated_token_ids",
            "step_logit_sha256",
            "final_lower_state_sha256",
            "final_lower_cache_content_sha256",
            "final_suffix_cache_sha256",
        )
    }


def compare_logit_steps(
    candidate: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
    *,
    relative_l2_threshold: float,
) -> dict[str, Any]:
    require(len(candidate) == len(reference) and len(candidate) > 0, "logit step cardinality drift")
    require(relative_l2_threshold > 0.0, "oracle threshold must be positive")
    rows = []
    for step, (actual, expected) in enumerate(zip(candidate, reference)):
        actual_fp32 = actual.detach().float().cpu()
        expected_fp32 = expected.detach().float().cpu()
        require(actual_fp32.shape == expected_fp32.shape, "oracle logit shape mismatch")
        delta = actual_fp32 - expected_fp32
        reference_norm = float(torch.linalg.vector_norm(expected_fp32).item())
        relative_l2 = float(torch.linalg.vector_norm(delta).item()) / max(reference_norm, 1e-12)
        maximum = float(delta.abs().max().item())
        top1_equal = int(actual_fp32.argmax().item()) == int(expected_fp32.argmax().item())
        rows.append(
            {
                "step": step,
                "finite": math.isfinite(relative_l2) and math.isfinite(maximum),
                "top1_equal": top1_equal,
                "max_abs": maximum,
                "relative_l2": relative_l2,
                "relative_l2_threshold": relative_l2_threshold,
                "passed": top1_equal and relative_l2 <= relative_l2_threshold,
            }
        )
    return {
        "predicate_id": "INDEPENDENT_DENSE_SEMANTIC_ORACLE",
        "passed": all(row["passed"] for row in rows),
        "rows": rows,
    }


def classify_pair(
    *,
    expected_predicate: str,
    clean_predicate_passed: bool,
    mutant_predicate_passed: bool,
    observed_predicate: str | None,
) -> dict[str, Any]:
    if not clean_predicate_passed:
        outcome = "clean_false_positive"
    elif mutant_predicate_passed:
        outcome = "escaped"
    elif observed_predicate == expected_predicate:
        outcome = "detected_expected_predicate"
    else:
        outcome = "detected_wrong_predicate"
    return {
        "expected_predicate": expected_predicate,
        "observed_predicate": observed_predicate,
        "clean_predicate_passed": clean_predicate_passed,
        "mutant_predicate_passed": mutant_predicate_passed,
        "outcome": outcome,
    }


def classify_detector_vector(
    *,
    expected_predicate: str,
    matched_clean: Mapping[str, bool],
    mutant: Mapping[str, bool],
) -> dict[str, Any]:
    require(set(matched_clean) == set(mutant) and expected_predicate in mutant, "detector vector schema drift")
    require(
        all(type(value) is bool for value in (*matched_clean.values(), *mutant.values())),
        "detector vector values must be booleans",
    )
    clean_failures = sorted(key for key, passed in matched_clean.items() if not passed)
    mutant_failures = sorted(key for key, passed in mutant.items() if not passed)
    if clean_failures:
        outcome = "clean_false_positive"
    elif not mutant_failures:
        outcome = "escaped"
    elif expected_predicate in mutant_failures:
        outcome = "detected_expected_predicate"
    else:
        outcome = "detected_wrong_predicate"
    return {
        "expected_predicate": expected_predicate,
        "failed_clean_predicate_ids": clean_failures,
        "failed_mutant_predicate_ids": mutant_failures,
        "outcome": outcome,
    }


def build_target_rows(predicates: Mapping[str, bool]) -> list[dict[str, Any]]:
    rows = []
    for target in TARGET_CONTRACT:
        row = dict(target)
        target_name = target["target"]
        if target["applicability"] == "not_applicable":
            row.update(
                status="not_applicable",
                predicate_passed=None,
                exact_missingness=[
                    "Transformers DynamicCache has no fixed-size paged partial tail",
                    "no page-level copy-before-append event exists in this adapter",
                ],
                scope_note=(
                    "Append safety is covered only indirectly by prefix content immutability "
                    "and request-cache disjointness; this does not satisfy the paged-tail target."
                ),
            )
        else:
            passed = predicates.get(target_name)
            require(isinstance(passed, bool), f"missing predicate for target {target_name}")
            if not passed:
                status = "open"
            else:
                status = target["maximum_status"]
            missingness: list[str] = []
            scope_note = ""
            if target_name == "dispatch_provenance":
                missingness = [
                    "compiled CUDA/Triton kernel binary fingerprint",
                    "kernel autotuning-choice fingerprint",
                    "hardware instruction trace",
                ]
                scope_note = (
                    "Full only at the Python adapter/layer-class receipt level; "
                    "therefore frozen as partial even when the predicate passes."
                )
            row.update(
                status=status,
                predicate_passed=passed,
                exact_missingness=missingness,
                scope_note=scope_note,
            )
        rows.append(row)
    return rows


def validate_source_manifest(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    require(manifest.get("schema_version") == SOURCE_SCHEMA, "source manifest schema drift")
    require(
        set(manifest) == {"schema_version", "protocol", "files", "normalized_files_sha256"}
        and manifest.get("protocol") == PROTOCOL,
        "source manifest top-level schema/protocol drift",
    )
    files = manifest.get("files")
    require(isinstance(files, list) and files, "source manifest has no files")
    seen = set()
    for row in files:
        require(type(row) is dict and set(row) == {"path", "sha256", "bytes"}, "source manifest row is invalid")
        relative = row.get("path")
        expected = row.get("sha256")
        require(isinstance(relative, str) and relative not in seen, "duplicate/invalid source path")
        require(".." not in Path(relative).parts and not Path(relative).is_absolute(), "unsafe source path")
        require(isinstance(expected, str) and _SHA_RE.fullmatch(expected) is not None, "invalid source SHA")
        path = root / relative
        require(path.is_file(), f"source file is missing: {relative}")
        require(sha256_file(path) == expected, f"source file SHA mismatch: {relative}")
        require(type(row["bytes"]) is int and row["bytes"] == path.stat().st_size, f"source file byte count mismatch: {relative}")
        seen.add(relative)
    normalized = [{"path": row["path"], "sha256": row["sha256"]} for row in files]
    require(
        manifest.get("normalized_files_sha256") == sha256_json(normalized),
        "normalized source digest mismatch",
    )
    return {"verified": True, "file_count": len(files), "normalized_files_sha256": sha256_json(normalized)}


def validate_sha256_ledger(
    root: Path,
    ledger_path: Path,
    *,
    expected_raw_sha256: str,
    label: str,
) -> dict[str, Any]:
    require(sha256_file(ledger_path) == expected_raw_sha256, f"{label} raw SHA mismatch")
    text = ledger_path.read_text(encoding="utf-8")
    require(text.endswith("\n"), f"{label} must end with a newline")
    rows = []
    seen = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00\r\n]+)", line)
        require(match is not None, f"{label} line {line_number} is malformed")
        digest, relative = match.groups()
        path = Path(relative)
        require(not path.is_absolute() and ".." not in path.parts and relative not in seen, f"{label} unsafe/duplicate path")
        target = root / path
        require(target.is_file(), f"{label} target is missing: {relative}")
        require(sha256_file(target) == digest, f"{label} target SHA mismatch: {relative}")
        seen.add(relative)
        rows.append({"path": relative, "sha256": digest, "bytes": target.stat().st_size})
    require(rows, f"{label} is empty")
    return {
        "raw_sha256": expected_raw_sha256,
        "file_count": len(rows),
        "normalized_entries_sha256": sha256_json(rows),
        "entries": rows,
    }


def parse_sha256_ledger_metadata(
    root: Path,
    ledger_path: Path,
    *,
    expected_raw_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Bind a small checksum ledger to file metadata without rereading payloads."""

    require(sha256_file(ledger_path) == expected_raw_sha256, f"{label} raw SHA mismatch")
    text = ledger_path.read_text(encoding="utf-8")
    require(text.endswith("\n"), f"{label} must end with a newline")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\x00\r\n]+)", line)
        require(match is not None, f"{label} line {line_number} is malformed")
        digest, relative = match.groups()
        path = Path(relative)
        require(
            not path.is_absolute() and ".." not in path.parts and relative not in seen,
            f"{label} unsafe/duplicate path",
        )
        target = root / path
        require(target.is_file(), f"{label} target is missing: {relative}")
        rows.append({"path": relative, "sha256": digest, "bytes": target.stat().st_size})
        seen.add(relative)
    require(rows, f"{label} is empty")
    return {
        "raw_sha256": expected_raw_sha256,
        "file_count": len(rows),
        "normalized_entries_sha256": sha256_json(rows),
        "entries": rows,
    }


def validate_model_authority_receipt(
    model_root: Path,
    authority: Mapping[str, Any],
    *,
    artifact_ledger_path: Path,
    weight_ledger_path: Path,
    artifact_ledger_raw_sha256: str,
    weight_ledger_raw_sha256: str,
) -> dict[str, Any]:
    require(
        set(authority)
        == {
            "schema_version",
            "model_id",
            "model_revision",
            "artifact_ledger",
            "weight_ledger",
            "stat_snapshot",
            "full_file_sha256_verified",
        }
        and authority.get("schema_version") == "forkaudit-transformers-model-authority-v2",
        "model authority schema drift",
    )
    require(authority.get("model_id") == "Qwen/Qwen3.5-35B-A3B", "model authority ID drift")
    require(
        authority.get("model_revision") == "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "model authority revision drift",
    )
    artifact = authority.get("artifact_ledger")
    weight = authority.get("weight_ledger")
    parsed_artifact = parse_sha256_ledger_metadata(
        model_root,
        artifact_ledger_path,
        expected_raw_sha256=artifact_ledger_raw_sha256,
        label="model artifact ledger",
    )
    parsed_weight = parse_sha256_ledger_metadata(
        model_root,
        weight_ledger_path,
        expected_raw_sha256=weight_ledger_raw_sha256,
        label="model weight ledger",
    )
    for label, claimed, parsed in (
        ("artifact", artifact, parsed_artifact),
        ("weight", weight, parsed_weight),
    ):
        require(
            type(claimed) is dict
            and set(claimed)
            == {"raw_sha256", "file_count", "normalized_entries_sha256", "entries"},
            f"authority {label} ledger schema drift",
        )
        require(claimed == parsed, f"authority {label} ledger does not match parsed ledger metadata")
    require(weight.get("file_count") == 14, "authority weight count drift")
    require(
        {row["path"] for row in artifact.get("entries", [])}
        == {
            "chat_template.jinja",
            "config.json",
            "generation_config.json",
            "merges.txt",
            "model.safetensors.index.json",
            "tokenizer_config.json",
            "vocab.json",
        },
        "authority artifact file set drift",
    )
    require(
        {row["path"] for row in weight.get("entries", [])}
        == {f"model.safetensors-{index:05d}-of-00014.safetensors" for index in range(1, 15)},
        "authority weight file set drift",
    )
    # This is a redundant producer assertion.  Authority comes from the pre-output
    # full-hash stage and must be byte-identical to the terminal full-hash closure;
    # rank-local validation below never treats this boolean as authorization.
    require(type(authority.get("full_file_sha256_verified")) is bool, "authority verification flag type drift")
    stats = authority.get("stat_snapshot")
    require(type(stats) is list and len(stats) == artifact.get("file_count") + weight.get("file_count"), "authority stat count drift")
    expected_paths = {
        row["path"] for row in (*parsed_artifact["entries"], *parsed_weight["entries"])
    }
    seen = set()
    for index, raw in enumerate(stats):
        require(
            type(raw) is dict
            and set(raw)
            == {
                "path",
                "bytes",
                "device",
                "inode",
                "ctime_ns",
                "regular_file",
                "no_write_mode_bits",
            },
            "authority stat schema drift",
        )
        relative = raw["path"]
        require(type(relative) is str and relative not in seen, "authority stat path drift")
        seen.add(relative)
        path = model_root / relative
        require(path.is_file(), f"authority model file missing: {relative}")
        stat_result = path.stat()
        expected = {
            "path": relative,
            "bytes": stat_result.st_size,
            "device": stat_result.st_dev,
            "inode": stat_result.st_ino,
            "ctime_ns": stat_result.st_ctime_ns,
            "regular_file": statlib.S_ISREG(stat_result.st_mode),
            "no_write_mode_bits": (
                stat_result.st_mode & (statlib.S_IWUSR | statlib.S_IWGRP | statlib.S_IWOTH)
            )
            == 0,
        }
        require(raw == expected, f"authority stat drift: {relative}")
        require(raw["regular_file"] is True, f"authority model path is not a regular file: {relative}")
        require(raw["no_write_mode_bits"] is True, f"authority model file has a write mode bit: {relative}")
    require(seen == expected_paths, "authority stat paths do not equal ledger entry union")
    return {
        "verified": True,
        "file_count": len(stats),
        "stat_snapshot_sha256": sha256_json(stats),
    }


def validate_gpu_assignment(value: Any) -> list[dict[str, Any]]:
    receipt = _exact_object(
        value,
        {"schema_version", "world_size", "hardware_contract", "rows", "rows_sha256"},
        "gpu_assignment",
    )
    require(receipt["schema_version"] == "forkaudit-transformers-gpu-assignment-v1", "GPU assignment schema drift")
    require(_exact_int(receipt["world_size"], "gpu_assignment.world_size", minimum=1) == WORLD_SIZE, "GPU assignment world size drift")
    require(receipt["hardware_contract"] == "NVIDIA H20-3e / compute capability 9.0 / BF16", "GPU hardware contract drift")
    rows = _exact_list(receipt["rows"], WORLD_SIZE, "gpu_assignment.rows")
    for rank, raw in enumerate(rows):
        row = _exact_object(
            raw,
            {"rank", "visible_index", "uuid", "name", "total_memory_mib", "compute_capability", "bf16_supported"},
            f"gpu_assignment.rows[{rank}]",
        )
        require(_exact_int(row["rank"], f"gpu_assignment.rank[{rank}]") == rank, "GPU rank drift")
        require(_exact_int(row["visible_index"], f"gpu_assignment.index[{rank}]") == rank, "GPU visible index drift")
        require(type(row["uuid"]) is str and row["uuid"].startswith("GPU-"), "GPU UUID invalid")
        require(row["name"] == "NVIDIA H20-3e", "GPU name drift")
        _exact_int(row["total_memory_mib"], f"gpu_assignment.memory[{rank}]", minimum=1)
        capability = _exact_list(row["compute_capability"], 2, f"gpu_assignment.capability[{rank}]")
        require(all(type(item) is int for item in capability) and capability == [9, 0], "GPU compute capability drift")
        require(_exact_bool(row["bf16_supported"], f"gpu_assignment.bf16[{rank}]"), "GPU BF16 unavailable")
    require(len({row["uuid"] for row in rows}) == WORLD_SIZE, "GPU assignment UUIDs are not unique")
    require(receipt["rows_sha256"] == sha256_json(rows), "GPU assignment rows digest drift")
    return rows


def _validate_static_ledger_receipt(value: Any, label: str) -> dict[str, Any]:
    receipt = _exact_object(
        value, {"raw_sha256", "file_count", "normalized_entries_sha256", "entries"}, label
    )
    _sha(receipt["raw_sha256"], f"{label}.raw_sha256")
    entries = _exact_list(receipt["entries"], None, f"{label}.entries")
    require(_exact_int(receipt["file_count"], f"{label}.file_count", minimum=1) == len(entries), f"{label} count drift")
    seen = set()
    for index, raw in enumerate(entries):
        row = _exact_object(raw, {"path", "sha256", "bytes"}, f"{label}.entries[{index}]")
        require(type(row["path"]) is str and row["path"] not in seen and not Path(row["path"]).is_absolute() and ".." not in Path(row["path"]).parts, f"{label} entry path drift")
        seen.add(row["path"]); _sha(row["sha256"], f"{label} entry SHA"); _exact_int(row["bytes"], f"{label} entry bytes", minimum=1)
    require(receipt["normalized_entries_sha256"] == sha256_json(entries), f"{label} normalized digest drift")
    return receipt


def validate_static_manifest(value: Any) -> dict[str, Any]:
    static = _exact_object(
        value,
        {
            "schema_version", "protocol", "created_before_gpu_execution", "source_manifest_raw_sha256",
            "formal_config", "formal_config_sha256", "dataset", "window_algorithm", "rank_inputs",
            "rank_inputs_sha256", "model", "environment_contract", "hardware_contract",
            "storage_receipt_salt", "oracle_contract", "target_contract", "fault_contract",
            "portable_record_mapping", "claim_boundary",
        },
        "static_manifest",
    )
    require(static["schema_version"] == STATIC_SCHEMA and static["protocol"] == PROTOCOL, "static identity drift")
    require(_exact_bool(static["created_before_gpu_execution"], "static.created_before_gpu_execution"), "static was not pre-output")
    _sha(static["source_manifest_raw_sha256"], "static.source_manifest_raw_sha256")
    config = _exact_object(
        static["formal_config"],
        {"world_size", "pg19_train_books", "document_tokens", "query_tokens", "fanouts", "split_depth", "semantic_steps", "window_stride", "query_stride", "candidate_windows_per_book", "seed", "scheduler", "arms"},
        "static.formal_config",
    )
    for field, minimum in (
        ("world_size", 1), ("pg19_train_books", 1), ("document_tokens", 1),
        ("query_tokens", 1), ("split_depth", 1), ("semantic_steps", 1),
        ("window_stride", 1), ("query_stride", 1), ("candidate_windows_per_book", 1),
        ("seed", 0),
    ):
        _exact_int(config[field], f"static.formal_config.{field}", minimum=minimum)
    fanouts = _exact_list(config["fanouts"], 2, "static.formal_config.fanouts")
    require(all(type(item) is int for item in fanouts), "static fanout types drift")
    arms = _exact_list(config["arms"], 2, "static.formal_config.arms")
    require(all(type(item) is str for item in arms), "static arm types drift")
    require(
        config["world_size"] == config["pg19_train_books"] == WORLD_SIZE
        and config["document_tokens"] == 256 and config["query_tokens"] == 24
        and config["fanouts"] == list(FANOUTS) and config["split_depth"] == 7
        and config["semantic_steps"] == 2 and config["window_stride"] == 197
        and config["query_stride"] == 32 and config["candidate_windows_per_book"] == 8
        and config["seed"] == 20260819
        and config["scheduler"] == "single-cuda-stream-request-index-interleaved"
        and config["arms"] == ["deep_materialized", "persistent_fork"],
        "static formal configuration drift",
    )
    require(static["formal_config_sha256"] == sha256_json(config), "static config digest drift")
    dataset = _exact_object(static["dataset"], {"bucket", "prefix", "records", "data_sha256", "manifest_sha256", "test_or_validation_objects_used"}, "static.dataset")
    require(dataset["bucket"] == "deepmind-gutenberg" and dataset["prefix"] == "train/" and _exact_int(dataset["records"], "static.dataset.records", minimum=WORLD_SIZE) >= WORLD_SIZE, "static dataset identity drift")
    _sha(dataset["data_sha256"], "static.dataset.data_sha256"); _sha(dataset["manifest_sha256"], "static.dataset.manifest_sha256")
    require(_exact_bool(dataset["test_or_validation_objects_used"], "static.dataset.test_used") is False, "static uses test/validation")
    window = _exact_object(static["window_algorithm"], {"implementation", "selection_key", "window_key", "raw_int64_token_receipts"}, "static.window_algorithm")
    require(
        window["implementation"] == "independent-bounded-PG19-raw-token-windows-v1"
        and window["selection_key"] == "sha256(seed|transformers-transfer-book|source_object)"
        and window["window_key"] == "sha256(seed|transformers-transfer-window|source_object)"
        and _exact_bool(window["raw_int64_token_receipts"], "static.window.raw") is True,
        "static window algorithm drift",
    )
    model = _exact_object(
        static["model"],
        {"model_id", "model_revision", "model_artifact_ledger_raw_sha256", "model_weight_ledger_raw_sha256", "artifact_ledger_receipt", "weight_ledger_receipt", "artifact_set_sha256", "layer_types", "tokenizer_class"},
        "static.model",
    )
    require(model["model_id"] == "Qwen/Qwen3.5-35B-A3B" and model["model_revision"] == "59d61f3ce65a6d9863b86d2e96597125219dc754", "static model identity drift")
    artifact = _validate_static_ledger_receipt(model["artifact_ledger_receipt"], "static.model.artifact_ledger")
    weight = _validate_static_ledger_receipt(model["weight_ledger_receipt"], "static.model.weight_ledger")
    require(artifact["raw_sha256"] == model["model_artifact_ledger_raw_sha256"] and weight["raw_sha256"] == model["model_weight_ledger_raw_sha256"], "static model ledger binding drift")
    require(model["artifact_set_sha256"] == sha256_json(artifact["entries"]), "static artifact-set digest drift")
    layer_types = _exact_list(model["layer_types"], 40, "static.model.layer_types")
    require(all(type(item) is str and item in {"linear_attention", "full_attention"} for item in layer_types) and type(model["tokenizer_class"]) is str and model["tokenizer_class"], "static model geometry/tokenizer drift")
    require(
        layer_types
        == ["full_attention" if index in range(3, 40, 4) else "linear_attention" for index in range(40)],
        "static hybrid layer pattern drift",
    )
    environment = _exact_object(static["environment_contract"], {"python", "torch", "cuda", "transformers"}, "static.environment")
    require(all(type(environment[key]) is str and environment[key] for key in environment), "static environment value drift")
    require(type(static["storage_receipt_salt"]) is str and static["storage_receipt_salt"], "static storage salt invalid")
    oracle = _exact_object(static["oracle_contract"], {"path", "independent_of_ownership_arms", "full_vocabulary_cpu_fp32_sidecars_required", "top1_exact_required", "relative_l2_threshold"}, "static.oracle")
    require(_exact_bool(oracle["independent_of_ownership_arms"], "static.oracle.independent") and _exact_bool(oracle["full_vocabulary_cpu_fp32_sidecars_required"], "static.oracle.sidecars") and _exact_bool(oracle["top1_exact_required"], "static.oracle.top1") and _exact_number(oracle["relative_l2_threshold"], "static.oracle.threshold") == 0.005, "static oracle drift")
    require(static["target_contract"] == list(TARGET_CONTRACT) and static["fault_contract"] == list(FAULT_CONTRACT), "static target/fault contract drift")
    mapping = _exact_object(static["portable_record_mapping"], {"identity", "ownership", "execution", "accounting", "tail_event", "dispatch"}, "static.portable_record_mapping")
    require(all(type(value) is str and value for value in mapping.values()), "static portable record mapping value drift")
    claim = _exact_object(static["claim_boundary"], {"same_model", "different_runtime", "tail_target", "dispatch_target", "not_authorized"}, "static.claim_boundary")
    require(
        claim["same_model"] == "Qwen/Qwen3.5-35B-A3B@59d61f3ce65a6d9863b86d2e96597125219dc754"
        and claim["different_runtime"] == "Transformers DynamicCache through qcomem_torch.TorchSplitCausalLM"
        and claim["tail_target"].startswith("not_applicable:")
        and claim["dispatch_target"].startswith("partial:")
        and type(claim["not_authorized"]) is list
        and all(type(item) is str and item for item in claim["not_authorized"]),
        "static claim boundary drift",
    )
    return static


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    require(type(value) is dict, f"{label} must be an object")
    require(set(value) == keys, f"{label} schema drift: {sorted(set(value) ^ keys)}")
    return value


def _exact_list(value: Any, length: int | None, label: str) -> list[Any]:
    require(type(value) is list, f"{label} must be an array")
    if length is not None:
        require(len(value) == length, f"{label} cardinality drift")
    return value


def _exact_bool(value: Any, label: str) -> bool:
    require(type(value) is bool, f"{label} must be a JSON boolean")
    return value


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    require(type(value) is int and value >= minimum, f"{label} must be an integer >= {minimum}")
    return value


def _exact_number(value: Any, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(float(value)), f"{label} must be finite")
    return float(value)


def _sha(value: Any, label: str) -> str:
    require(type(value) is str and _SHA_RE.fullmatch(value) is not None, f"{label} is not a SHA-256")
    return value


def _validate_tensor_receipt(value: Any, label: str) -> dict[str, Any]:
    row = _exact_object(value, {"shape", "dtype", "content_sha256", "finite"}, label)
    shape = _exact_list(row["shape"], None, f"{label}.shape")
    require(all(type(item) is int and item >= 0 for item in shape), f"{label}.shape is invalid")
    require(type(row["dtype"]) is str and row["dtype"], f"{label}.dtype is invalid")
    _sha(row["content_sha256"], f"{label}.content_sha256")
    _exact_bool(row["finite"], f"{label}.finite")
    return row


def _validate_state_receipt(value: Any, label: str) -> dict[str, Any]:
    row = _exact_object(
        value,
        {
            "depth",
            "document_length",
            "current_length",
            "document_residual",
            "cache_content_sha256",
            "cache_tensor_count",
            "state_content_sha256",
        },
        label,
    )
    identity = {
        "depth": _exact_int(row["depth"], f"{label}.depth"),
        "document_length": _exact_int(row["document_length"], f"{label}.document_length", minimum=1),
        "current_length": _exact_int(row["current_length"], f"{label}.current_length", minimum=1),
        "document_residual": _validate_tensor_receipt(row["document_residual"], f"{label}.document_residual"),
        "cache_content_sha256": _sha(row["cache_content_sha256"], f"{label}.cache_content_sha256"),
        "cache_tensor_count": _exact_int(row["cache_tensor_count"], f"{label}.cache_tensor_count"),
    }
    require(row["state_content_sha256"] == sha256_json(identity), f"{label} state digest does not replay")
    return row


def _validate_inventory(
    value: Any, label: str, *, expected_domain: str, allow_empty: bool = False
) -> dict[str, Any]:
    inventory = _exact_object(
        value,
        {"role", "storage_salt_domain_sha256", "tensor_rows", "rows", "inventory_sha256"},
        label,
    )
    require(type(inventory["role"]) is str and inventory["role"], f"{label}.role is invalid")
    require(inventory["storage_salt_domain_sha256"] == expected_domain, f"{label} salt domain drift")
    rows = _exact_list(inventory["rows"], None, f"{label}.rows")
    require(rows or allow_empty, f"{label} tensor rows are empty")
    require(_exact_int(inventory["tensor_rows"], f"{label}.tensor_rows") == len(rows), f"{label} count drift")
    for index, raw in enumerate(rows):
        row = _exact_object(
            raw,
            {
                "path",
                "role",
                "device",
                "dtype",
                "shape",
                "stride",
                "contiguous",
                "storage_bytes",
                "view_offset_bytes",
                "view_nbytes",
                "view_start_bytes",
                "view_end_bytes",
                "storage_id_sha256",
            },
            f"{label}.rows[{index}]",
        )
        for field in ("path", "role", "device", "dtype"):
            require(type(row[field]) is str and row[field], f"{label}.{field} is invalid")
        shape = _exact_list(row["shape"], None, f"{label}.rows[{index}].shape")
        require(all(type(item) is int and item > 0 for item in shape), f"{label} row shape invalid")
        stride = _exact_list(row["stride"], len(shape), f"{label}.rows[{index}].stride")
        require(all(type(item) is int and item >= 0 for item in stride), f"{label} row stride invalid")
        require(_exact_bool(row["contiguous"], f"{label}.rows[{index}].contiguous"), f"{label} non-contiguous witness")
        expected_stride = 1
        for dimension, observed_stride in reversed(list(zip(shape, stride))):
            if dimension != 1:
                require(observed_stride == expected_stride, f"{label} stride is not contiguous")
            expected_stride *= dimension
        storage_bytes = _exact_int(row["storage_bytes"], f"{label}.rows[{index}].storage_bytes", minimum=1)
        view_offset = _exact_int(row["view_offset_bytes"], f"{label}.rows[{index}].view_offset_bytes")
        view_nbytes = _exact_int(row["view_nbytes"], f"{label}.rows[{index}].view_nbytes", minimum=1)
        view_start = _exact_int(row["view_start_bytes"], f"{label}.rows[{index}].view_start_bytes")
        view_end = _exact_int(row["view_end_bytes"], f"{label}.rows[{index}].view_end_bytes", minimum=1)
        require(view_offset == view_start and view_end == view_start + view_nbytes, f"{label} normalized view range drift")
        require(view_end <= storage_bytes, f"{label} view exceeds storage bounds")
        _sha(row["storage_id_sha256"], f"{label}.rows[{index}].storage_id_sha256")
    require(inventory["inventory_sha256"] == sha256_json(rows), f"{label} inventory digest mismatch")
    return inventory


def _physical_storage_rows(inventory: Mapping[str, Any]) -> list[dict[str, Any]]:
    fields = (
        "path", "device", "dtype", "shape", "stride", "contiguous", "storage_bytes", "view_offset_bytes",
        "view_nbytes", "view_start_bytes", "view_end_bytes", "storage_id_sha256",
    )
    return [{field: row[field] for field in fields} for row in inventory["rows"]]


def _validate_cache_snapshot(
    value: Any, label: str, *, expected_domain: str, allow_empty: bool
) -> dict[str, Any]:
    snapshot = _exact_object(value, {"tensor_count", "content_sha256", "storage"}, label)
    storage = _validate_inventory(
        snapshot["storage"], f"{label}.storage", expected_domain=expected_domain,
        allow_empty=allow_empty,
    )
    require(
        _exact_int(snapshot["tensor_count"], f"{label}.tensor_count") == storage["tensor_rows"],
        f"{label} tensor count drift",
    )
    _sha(snapshot["content_sha256"], f"{label}.content_sha256")
    return snapshot


def _validate_allocator_snapshots(value: Any, label: str) -> list[dict[str, Any]]:
    rows = _exact_list(value, 3, label)
    phases = ("setup", "first_transition", "final")
    for phase, raw in zip(phases, rows):
        row = _exact_object(
            raw,
            {"phase", "allocated_bytes", "reserved_bytes", "max_allocated_bytes", "max_reserved_bytes"},
            f"{label}.{phase}",
        )
        require(row["phase"] == phase, f"{label} phase drift")
        allocated = _exact_int(row["allocated_bytes"], f"{label}.{phase}.allocated")
        reserved = _exact_int(row["reserved_bytes"], f"{label}.{phase}.reserved")
        max_allocated = _exact_int(row["max_allocated_bytes"], f"{label}.{phase}.max_allocated")
        max_reserved = _exact_int(row["max_reserved_bytes"], f"{label}.{phase}.max_reserved")
        require(allocated <= reserved and allocated <= max_allocated and reserved <= max_reserved, f"{label}.{phase} allocator relation drift")
    require(
        all(rows[index]["max_allocated_bytes"] <= rows[index + 1]["max_allocated_bytes"] for index in range(2))
        and all(rows[index]["max_reserved_bytes"] <= rows[index + 1]["max_reserved_bytes"] for index in range(2)),
        f"{label} allocator maxima are not monotone",
    )
    return rows


def _load_logit_bundle(
    sidecar_dir: Path,
    value: Any,
    *,
    expected_rank: int,
) -> dict[str, tuple[torch.Tensor, dict[str, Any]]]:
    receipt = _exact_object(
        value,
        {
            "schema_version",
            "logical_name",
            "bytes",
            "sha256",
            "record_count",
            "records",
            "terminal_closure",
        },
        "logit_sidecar",
    )
    require(receipt["schema_version"] == "forkaudit-fp32-logit-bundle-v1", "logit bundle schema drift")
    expected_name = f"forkaudit-transformers-transfer-logits-rank-{expected_rank}.bin"
    require(receipt["logical_name"] == expected_name, "logit bundle logical name drift")
    path = sidecar_dir / expected_name
    require(path.is_file(), f"logit bundle is missing: {expected_name}")
    raw = path.read_bytes()
    require(_exact_int(receipt["bytes"], "logit_sidecar.bytes") == len(raw), "logit bundle byte count drift")
    require(receipt["sha256"] == sha256_bytes(raw), "logit bundle raw SHA mismatch")
    records = _exact_list(receipt["records"], None, "logit_sidecar.records")
    require(_exact_int(receipt["record_count"], "logit_sidecar.record_count") == len(records), "logit record count drift")
    require(records, "logit bundle has no records")
    parsed: dict[str, tuple[torch.Tensor, dict[str, Any]]] = {}
    expected_offset = 0
    for index, raw_record in enumerate(records):
        record = _exact_object(
            raw_record,
            {"record_id", "offset_bytes", "nbytes", "shape", "dtype", "content_sha256"},
            f"logit_sidecar.records[{index}]",
        )
        record_id = record["record_id"]
        require(type(record_id) is str and record_id and record_id not in parsed, "duplicate/invalid logit record ID")
        offset = _exact_int(record["offset_bytes"], f"logit record {record_id}.offset")
        nbytes = _exact_int(record["nbytes"], f"logit record {record_id}.nbytes", minimum=1)
        require(offset == expected_offset, f"logit record {record_id} is not contiguous")
        shape = _exact_list(record["shape"], None, f"logit record {record_id}.shape")
        require(shape and all(type(item) is int and item > 0 for item in shape), "logit shape invalid")
        require(record["dtype"] == "float32-le", "logit sidecar dtype drift")
        require(math.prod(shape) * 4 == nbytes, "logit record shape/byte mismatch")
        end = offset + nbytes
        require(end <= len(raw), "logit record exceeds sidecar")
        payload = raw[offset:end]
        require(record["content_sha256"] == sha256_bytes(payload), "logit record SHA mismatch")
        tensor = torch.frombuffer(bytearray(payload), dtype=torch.float32).clone().reshape(shape)
        require(bool(torch.isfinite(tensor).all().item()), "logit record is non-finite")
        parsed[record_id] = (tensor, record)
        expected_offset = end
    closure = _exact_object(
        receipt["terminal_closure"],
        {"first_offset_bytes", "last_end_offset_bytes", "exact_byte_coverage"},
        "logit_sidecar.terminal_closure",
    )
    require(_exact_int(closure["first_offset_bytes"], "logit closure first") == 0, "logit first offset drift")
    require(_exact_int(closure["last_end_offset_bytes"], "logit closure end") == len(raw), "logit terminal closure drift")
    require(_exact_bool(closure["exact_byte_coverage"], "logit exact coverage"), "logit bundle coverage false")
    require(expected_offset == len(raw), "logit bundle has unindexed trailing bytes")
    return parsed


def _validate_disjointness(
    claimed: Any,
    inventories: Sequence[Mapping[str, Any]],
    forbidden: Sequence[Mapping[str, Any]],
    label: str,
) -> dict[str, Any]:
    expected = disjointness_receipt(inventories, forbidden=forbidden)
    receipt = _exact_object(
        claimed,
        {
            "predicate_id",
            "passed",
            "comparisons",
            "comparison_count",
            "tensor_pair_comparison_count",
            "receipt_sha256",
        },
        label,
    )
    _exact_bool(receipt["passed"], f"{label}.passed")
    _exact_int(receipt["comparison_count"], f"{label}.comparison_count")
    _exact_int(receipt["tensor_pair_comparison_count"], f"{label}.tensor_pair_comparison_count")
    _sha(receipt["receipt_sha256"], f"{label}.receipt_sha256")
    require(receipt == expected, f"{label} does not match blind all-pairs replay")
    return expected


def _validate_semantic_key(
    value: Any,
    label: str,
    *,
    semantic_steps: int,
    logit_bundle: Mapping[str, tuple[torch.Tensor, dict[str, Any]]],
) -> dict[str, Any]:
    row = _exact_object(
        value,
        {
            "request_index",
            "query_token_ids_sha256",
            "generated_token_ids",
            "step_logit_sha256",
            "step_logit_record_ids",
            "final_lower_state_sha256",
            "final_lower_cache_content_sha256",
            "final_suffix_cache_sha256",
        },
        label,
    )
    _exact_int(row["request_index"], f"{label}.request_index")
    _sha(row["query_token_ids_sha256"], f"{label}.query_token_ids_sha256")
    tokens = _exact_list(row["generated_token_ids"], semantic_steps, f"{label}.generated_token_ids")
    require(all(type(token) is int and token >= 0 for token in tokens), f"{label} token list is invalid")
    digests = _exact_list(row["step_logit_sha256"], semantic_steps, f"{label}.step_logit_sha256")
    record_ids = _exact_list(row["step_logit_record_ids"], semantic_steps, f"{label}.step_logit_record_ids")
    for index, (digest, record_id) in enumerate(zip(digests, record_ids)):
        _sha(digest, f"{label}.step_logit_sha256[{index}]")
        require(type(record_id) is str and record_id in logit_bundle, f"{label} missing logit record")
        tensor, record = logit_bundle[record_id]
        require(record["content_sha256"] == digest, f"{label} logit record/digest drift")
        require(int(tensor.argmax().item()) == tokens[index], f"{label} generated token differs from FP32 sidecar")
    _sha(row["final_lower_state_sha256"], f"{label}.final_lower_state_sha256")
    _sha(row["final_lower_cache_content_sha256"], f"{label}.final_lower_cache_content_sha256")
    _sha(row["final_suffix_cache_sha256"], f"{label}.final_suffix_cache_sha256")
    return row


def _validate_semantic_row(
    value: Any,
    label: str,
    *,
    semantic_steps: int,
    document_tokens: int,
    query_tokens: int,
    expected_domain: str,
    logit_bundle: Mapping[str, tuple[torch.Tensor, dict[str, Any]]],
) -> dict[str, Any]:
    row = _exact_object(
        value,
        {
            "request_index",
            "query_token_ids_sha256",
            "generated_token_ids",
            "step_logit_sha256",
            "step_logit_record_ids",
            "final_lower_state_sha256",
            "final_lower_cache_content_sha256",
            "final_suffix_cache_sha256",
            "final_current_length",
            "lower_cache_storage",
            "suffix_cache_storage",
        },
        label,
    )
    _validate_semantic_key(
        {**semantic_key(row), "step_logit_record_ids": row["step_logit_record_ids"]},
        f"{label}.semantic_key",
        semantic_steps=semantic_steps,
        logit_bundle=logit_bundle,
    )
    require(
        _exact_int(row["final_current_length"], f"{label}.final_current_length")
        == document_tokens + query_tokens + semantic_steps - 1,
        f"{label} final current length is noncanonical",
    )
    _validate_inventory(row["lower_cache_storage"], f"{label}.lower_cache_storage", expected_domain=expected_domain)
    _validate_inventory(row["suffix_cache_storage"], f"{label}.suffix_cache_storage", expected_domain=expected_domain)
    _sha(row["final_lower_cache_content_sha256"], f"{label}.final_lower_cache_content_sha256")
    return row


def _replay_numeric_oracle(
    value: Any,
    label: str,
    *,
    threshold: float,
    candidate: Sequence[torch.Tensor],
    reference: Sequence[torch.Tensor],
) -> bool:
    numeric = _exact_object(value, {"predicate_id", "passed", "rows"}, label)
    require(numeric["predicate_id"] == "INDEPENDENT_DENSE_SEMANTIC_ORACLE", f"{label} predicate drift")
    rows = _exact_list(numeric["rows"], None, f"{label}.rows")
    require(len(rows) == len(candidate) == len(reference) and rows, f"{label} numeric cardinality drift")
    replayed = []
    for index, raw in enumerate(rows):
        row = _exact_object(
            raw,
            {
                "step",
                "finite",
                "top1_equal",
                "max_abs",
                "relative_l2",
                "relative_l2_threshold",
                "passed",
            },
            f"{label}.rows[{index}]",
        )
        require(_exact_int(row["step"], f"{label}.rows[{index}].step") == index, f"{label} step order drift")
        actual = candidate[index]
        expected = reference[index]
        require(actual.shape == expected.shape, f"{label} sidecar shape mismatch")
        delta = actual - expected
        maximum = float(delta.abs().max().item())
        reference_norm = float(torch.linalg.vector_norm(expected).item())
        relative = float(torch.linalg.vector_norm(delta).item()) / max(reference_norm, 1e-12)
        require(_exact_number(row["max_abs"], f"{label}.rows[{index}].max_abs") == maximum, f"{label} max_abs is not sidecar-derived")
        require(_exact_number(row["relative_l2"], f"{label}.rows[{index}].relative_l2") == relative, f"{label} relative_l2 is not sidecar-derived")
        require(_exact_number(row["relative_l2_threshold"], f"{label}.threshold") == threshold, f"{label} threshold drift")
        finite = math.isfinite(maximum) and math.isfinite(relative)
        top1 = int(actual.argmax().item()) == int(expected.argmax().item())
        require(_exact_bool(row["top1_equal"], f"{label}.rows[{index}].top1_equal") == top1, f"{label} top1 is not sidecar-derived")
        passed = top1 and relative <= threshold
        require(_exact_bool(row["finite"], f"{label}.rows[{index}].finite") == finite, f"{label} finite flag drift")
        require(_exact_bool(row["passed"], f"{label}.rows[{index}].passed") == passed, f"{label} row pass drift")
        replayed.append(passed)
    overall = all(replayed)
    require(_exact_bool(numeric["passed"], f"{label}.passed") == overall, f"{label} aggregate pass drift")
    return overall


def _replay_oracle_comparisons(
    value: Any,
    semantics: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    label: str,
    *,
    threshold: float,
    logit_bundle: Mapping[str, tuple[torch.Tensor, dict[str, Any]]],
) -> bool:
    rows = _exact_list(value, len(semantics), label)
    passed_rows = []
    for index, (raw, semantic, oracle) in enumerate(zip(rows, semantics, oracle_rows)):
        row = _exact_object(raw, {"request_index", "token_match", "numeric", "passed"}, f"{label}[{index}]")
        require(_exact_int(row["request_index"], f"{label}[{index}].request_index") == index, f"{label} request order drift")
        token_match = semantic["generated_token_ids"] == oracle["generated_token_ids"]
        require(_exact_bool(row["token_match"], f"{label}[{index}].token_match") == token_match, f"{label} token match drift")
        candidate = [logit_bundle[item][0] for item in semantic["step_logit_record_ids"]]
        reference = [logit_bundle[item][0] for item in oracle["step_logit_record_ids"]]
        numeric_pass = _replay_numeric_oracle(
            row["numeric"],
            f"{label}[{index}].numeric",
            threshold=threshold,
            candidate=candidate,
            reference=reference,
        )
        passed = token_match and numeric_pass
        require(_exact_bool(row["passed"], f"{label}[{index}].passed") == passed, f"{label} pass drift")
        passed_rows.append(passed)
    return all(passed_rows)


def _validate_adapter_call_ledger(
    value: Any,
    label: str,
    *,
    fanout: int,
    semantic_steps: int,
    depth: int,
    document_tokens: int,
    query_tokens: int,
    expected_domain: str,
) -> list[dict[str, Any]]:
    expected_plan: list[dict[str, Any]] = []
    for request in range(fanout):
        expected_plan.extend(
            [
                {"request": request, "phase": "first-query-lower", "kind": "lower", "before": document_tokens, "tokens": query_tokens},
                {"request": request, "phase": "suffix-document", "kind": "suffix", "before": 0, "tokens": document_tokens},
                {"request": request, "phase": "first-query-suffix", "kind": "suffix", "before": document_tokens, "tokens": query_tokens},
            ]
        )
    for step in range(semantic_steps - 1):
        for request in range(fanout):
            before = document_tokens + query_tokens + step
            expected_plan.extend(
                [
                    {"request": request, "phase": f"generated-step-{step}-lower", "kind": "lower", "before": before, "tokens": 1},
                    {"request": request, "phase": f"generated-step-{step}-suffix", "kind": "suffix", "before": before, "tokens": 1},
                ]
            )
    expected_count = len(expected_plan)
    rows = _exact_list(value, expected_count, label)
    last_content: dict[tuple[int, str], str] = {}
    last_storage: dict[tuple[int, str], list[dict[str, Any]]] = {}
    phase_counts = {request: 0 for request in range(fanout)}
    for index, (raw, plan) in enumerate(zip(rows, expected_plan)):
        row = _exact_object(
            raw,
            {
                "call_index", "request_index", "phase", "callable", "layer_start", "layer_end",
                "position_offset", "current_length_before", "current_length_after", "input_tokens",
                "append_delta", "cache_before", "cache_after", "completed",
            },
            f"{label}[{index}]",
        )
        require(_exact_int(row["call_index"], f"{label}[{index}].call_index") == index, f"{label} call order drift")
        request = _exact_int(row["request_index"], f"{label}[{index}].request")
        require(request == plan["request"], f"{label} request/interleave order drift")
        phase_counts[request] += 1
        require(row["phase"] == plan["phase"], f"{label} phase schedule drift")
        callable_name = row["callable"]
        require(
            callable_name in {
                "TorchSplitCausalLM.continue_lower_replay",
                "TorchSplitCausalLM.run_suffix_cached_last_logits",
            },
            f"{label} callable drift",
        )
        layer_start = _exact_int(row["layer_start"], f"{label}[{index}].layer_start")
        layer_end = _exact_int(row["layer_end"], f"{label}[{index}].layer_end", minimum=1)
        cache_kind = "lower" if callable_name.endswith("continue_lower_replay") else "suffix"
        require(cache_kind == plan["kind"], f"{label} callable schedule drift")
        require(
            (layer_start, layer_end) == ((0, depth) if cache_kind == "lower" else (depth, 40)),
            f"{label} layer range drift",
        )
        position = _exact_int(row["position_offset"], f"{label}[{index}].position")
        before_length = _exact_int(row["current_length_before"], f"{label}[{index}].before_length")
        after_length = _exact_int(row["current_length_after"], f"{label}[{index}].after_length")
        input_tokens = _exact_int(row["input_tokens"], f"{label}[{index}].input_tokens", minimum=1)
        append_delta = _exact_int(row["append_delta"], f"{label}[{index}].append_delta", minimum=1)
        require(
            position == before_length == plan["before"]
            and after_length - before_length == input_tokens == append_delta == plan["tokens"],
            f"{label} frozen append/position schedule drift",
        )
        before = _validate_cache_snapshot(
            row["cache_before"], f"{label}[{index}].before", expected_domain=expected_domain,
            allow_empty=cache_kind == "suffix" and before_length == 0,
        )
        after = _validate_cache_snapshot(
            row["cache_after"], f"{label}[{index}].after", expected_domain=expected_domain,
            allow_empty=False,
        )
        key = (request, cache_kind)
        if key in last_content:
            require(before["content_sha256"] == last_content[key], f"{label} cache content chain drift")
            require(
                _physical_storage_rows(before["storage"]) == last_storage[key],
                f"{label} cache storage chain drift",
            )
        last_content[key] = after["content_sha256"]
        last_storage[key] = _physical_storage_rows(after["storage"])
        require(_exact_bool(row["completed"], f"{label}[{index}].completed"), f"{label} incomplete call")
    require(all(count == 3 + 2 * (semantic_steps - 1) for count in phase_counts.values()), f"{label} per-request call count drift")
    return rows


def _replay_arm(
    value: Any,
    label: str,
    *,
    arm_name: str,
    fanout: int,
    semantic_steps: int,
    document_tokens: int,
    query_tokens: int,
    depth: int,
    expected_domain: str,
    logit_bundle: Mapping[str, tuple[torch.Tensor, dict[str, Any]]],
) -> dict[str, Any]:
    arm = _exact_object(
        value,
        {
            "arm",
            "fanout",
            "state_construction",
            "scheduler",
            "setup_storage_inventories",
            "persistent_forbidden_inventories",
            "first_transition_combined_storage_inventories",
            "final_combined_storage_inventories",
            "document_residual_storage_inventories",
            "persistent_document_residual_inventory",
            "setup_disjointness",
            "first_transition_disjointness",
            "final_disjointness",
            "document_residual_ownership",
            "adapter_call_ledger",
            "allocator_accounting_snapshots",
            "semantics",
        },
        label,
    )
    require(arm["arm"] == arm_name, f"{label} arm identity drift")
    require(_exact_int(arm["fanout"], f"{label}.fanout", minimum=1) == fanout, f"{label} fanout drift")
    require(arm["scheduler"] == "single-cuda-stream-request-index-interleaved", f"{label} scheduler drift")
    expected_construction = (
        "one-persistent-prefix-then-LowerReplayState.fork"
        if arm_name == "persistent_fork"
        else "independent-write_lower_replay-per-request"
    )
    require(arm["state_construction"] == expected_construction, f"{label} construction drift")
    setup = [
        _validate_inventory(item, f"{label}.setup[{index}]", expected_domain=expected_domain)
        for index, item in enumerate(_exact_list(arm["setup_storage_inventories"], fanout, f"{label}.setup"))
    ]
    forbidden_count = 1 if arm_name == "persistent_fork" else 0
    forbidden = [
        _validate_inventory(item, f"{label}.forbidden[{index}]", expected_domain=expected_domain)
        for index, item in enumerate(_exact_list(arm["persistent_forbidden_inventories"], forbidden_count, f"{label}.forbidden"))
    ]
    final = [
        _validate_inventory(item, f"{label}.final[{index}]", expected_domain=expected_domain)
        for index, item in enumerate(_exact_list(arm["final_combined_storage_inventories"], fanout, f"{label}.final"))
    ]
    first_transition = [
        _validate_inventory(item, f"{label}.first_transition[{index}]", expected_domain=expected_domain)
        for index, item in enumerate(
            _exact_list(arm["first_transition_combined_storage_inventories"], fanout, f"{label}.first_transition")
        )
    ]
    residuals = [
        _validate_inventory(item, f"{label}.residuals[{index}]", expected_domain=expected_domain)
        for index, item in enumerate(_exact_list(arm["document_residual_storage_inventories"], fanout, f"{label}.residuals"))
    ]
    _validate_disjointness(arm["setup_disjointness"], setup, forbidden, f"{label}.setup_disjointness")
    _validate_disjointness(
        arm["first_transition_disjointness"], first_transition, forbidden,
        f"{label}.first_transition_disjointness",
    )
    _validate_disjointness(arm["final_disjointness"], final, forbidden, f"{label}.final_disjointness")
    if arm_name == "persistent_fork":
        persistent_residual = _validate_inventory(
            arm["persistent_document_residual_inventory"],
            f"{label}.persistent_residual",
            expected_domain=expected_domain,
        )
        ownership = _exact_object(
            arm["document_residual_ownership"],
            {"predicate_id", "tensor_pair_comparison_count", "passed"},
            f"{label}.residual_ownership",
        )
        require(ownership["predicate_id"] == "READ_ONLY_DOCUMENT_RESIDUAL_ALIASES_PERSISTENT_BASE", f"{label} residual predicate drift")
        base_ranges = {
            (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
            for row in persistent_residual["rows"]
        }
        ownership_pass = all(
            {
                (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
                for row in item["rows"]
            }
            == base_ranges
            for item in residuals
        )
        expected_pairs = sum(len(item["rows"]) * len(persistent_residual["rows"]) for item in residuals)
        require(
            _exact_int(ownership["tensor_pair_comparison_count"], f"{label}.residual_ownership.tensor_pairs")
            == expected_pairs,
            f"{label} residual comparison count drift",
        )
        require(_exact_bool(ownership["passed"], f"{label}.residual_ownership.passed") == ownership_pass, f"{label} residual ownership drift")
    else:
        require(arm["persistent_document_residual_inventory"] is None, f"{label} unexpected persistent residual")
        ownership_pass = _validate_disjointness(
            arm["document_residual_ownership"], residuals, [], f"{label}.residual_ownership"
        )["passed"]
    if fanout == 2:
        require(arm["setup_disjointness"]["tensor_pair_comparison_count"] > 0, f"{label} setup comparison is vacuous")
        require(arm["final_disjointness"]["tensor_pair_comparison_count"] > 0, f"{label} final comparison is vacuous")
        require(arm["first_transition_disjointness"]["tensor_pair_comparison_count"] > 0, f"{label} first-transition comparison is vacuous")
        require(
            arm["document_residual_ownership"]["tensor_pair_comparison_count"] > 0,
            f"{label} residual comparison is vacuous",
        )
    semantics = [
        _validate_semantic_row(
            item,
            f"{label}.semantics[{index}]",
            semantic_steps=semantic_steps,
            document_tokens=document_tokens,
            query_tokens=query_tokens,
            expected_domain=expected_domain,
            logit_bundle=logit_bundle,
        )
        for index, item in enumerate(_exact_list(arm["semantics"], fanout, f"{label}.semantics"))
    ]
    require([item["request_index"] for item in semantics] == list(range(fanout)), f"{label} semantic order drift")
    call_ledger = _validate_adapter_call_ledger(
        arm["adapter_call_ledger"], f"{label}.call_ledger", fanout=fanout,
        semantic_steps=semantic_steps, depth=depth, document_tokens=document_tokens,
        query_tokens=query_tokens, expected_domain=expected_domain,
    )
    _validate_allocator_snapshots(arm["allocator_accounting_snapshots"], f"{label}.allocator")
    for request in range(fanout):
        request_calls = [item for item in call_ledger if item["request_index"] == request]
        first_lower = next(item for item in request_calls if item["phase"] == "first-query-lower")
        first_suffix = next(item for item in request_calls if item["phase"] == "first-query-suffix")
        require(
            _physical_storage_rows(setup[request])
            == _physical_storage_rows(first_lower["cache_before"]["storage"]),
            f"{label}.setup[{request}] does not bind first lower call",
        )
        expected_first_rows = [
            *_physical_storage_rows(first_lower["cache_after"]["storage"]),
            *_physical_storage_rows(first_suffix["cache_after"]["storage"]),
        ]
        require(
            _physical_storage_rows(first_transition[request]) == expected_first_rows,
            f"{label}.first_transition[{request}] does not bind call receipts",
        )
        lower_calls = [item for item in request_calls if item["callable"].endswith("continue_lower_replay")]
        suffix_calls = [item for item in request_calls if item["callable"].endswith("run_suffix_cached_last_logits")]
        semantic = semantics[request]
        require(
            _physical_storage_rows(semantic["lower_cache_storage"])
            == _physical_storage_rows(lower_calls[-1]["cache_after"]["storage"]),
            f"{label} final lower storage does not bind call ledger",
        )
        require(
            _physical_storage_rows(semantic["suffix_cache_storage"])
            == _physical_storage_rows(suffix_calls[-1]["cache_after"]["storage"]),
            f"{label} final suffix storage does not bind call ledger",
        )
        require(
            semantic["final_lower_cache_content_sha256"] == lower_calls[-1]["cache_after"]["content_sha256"]
            and semantic["final_suffix_cache_sha256"] == suffix_calls[-1]["cache_after"]["content_sha256"],
            f"{label} final cache content does not bind call ledger",
        )
    for index, (combined, semantic) in enumerate(zip(final, semantics)):
        expected_rows = [*semantic["lower_cache_storage"]["rows"], *semantic["suffix_cache_storage"]["rows"]]
        require(combined["rows"] == expected_rows, f"{label}.final[{index}] does not bind semantic cache rows")
    return {
        "semantics": semantics,
        "ownership_passed": (
            arm["setup_disjointness"]["passed"]
            and arm["first_transition_disjointness"]["passed"]
            and arm["final_disjointness"]["passed"]
            and ownership_pass
        ),
    }


def _replay_faults(
    value: Any,
    *,
    clean_n1: Mapping[str, Any],
    oracle_rows: Sequence[Mapping[str, Any]],
    semantic_steps: int,
    threshold: float,
    expected_domain: str,
    logit_bundle: Mapping[str, tuple[torch.Tensor, dict[str, Any]]],
) -> list[dict[str, Any]]:
    faults = _exact_list(value, len(FAULT_CONTRACT), "fault_suite")
    require([row.get("fault_id") for row in faults] == [row["fault_id"] for row in FAULT_CONTRACT], "fault IDs/order drift")
    replayed = []
    for contract, raw in zip(FAULT_CONTRACT, faults):
        fault = _exact_object(
            raw,
            {
                "fault_id",
                "fault",
                "expected_predicate",
                "expected_outcome",
                "matched_clean",
                "mutant",
                "exercise_kind",
                "execution_outcome",
                "classification",
                "detector_vector",
                "fault_case_valid",
            },
            f"fault {contract['fault_id']}",
        )
        for field in ("fault_id", "fault", "expected_predicate", "expected_outcome"):
            require(fault[field] == contract[field], f"fault {contract['fault_id']} {field} drift")
        fault_id = contract["fault_id"]
        expected_kind = "downstream_runtime_fault" if fault_id in {"T1", "T5"} else "direct_contract_sensitivity"
        require(fault["exercise_kind"] == expected_kind, f"fault {fault_id} exercise kind drift")
        if fault_id == "T1":
            clean = _exact_object(fault["matched_clean"], {"predicate_passed", "source"}, "T1.clean")
            clean_pass = all(
                item["oracle_passed"]
                for item in clean_n1["arms"].values()
            )
            require(_exact_bool(clean["predicate_passed"], "T1.clean.predicate_passed") == clean_pass, "T1 clean drift")
            mutant = _exact_object(
                fault["mutant"],
                {
                    "injection",
                    "injection_receipt",
                    "common_mode_cross_arm_exact",
                    "cross_arm_semantics",
                    "oracle_comparisons",
                    "predicate_passed",
                },
                "T1.mutant",
            )
            require(
                mutant["injection"]
                == "digest-proven common-mode document-boundary residual content mutation in both arms",
                "T1 injection description drift",
            )
            injection = _exact_object(
                mutant["injection_receipt"],
                {
                    "materialized_before", "materialized_after", "corrupted_base_before",
                    "corrupted_base_after", "persistent_fork_after", "materialized_storage_before",
                    "materialized_storage_after", "corrupted_base_storage_before",
                    "corrupted_base_storage_after", "persistent_fork_storage", "changed_identically",
                    "persistent_aliases_corrupted_base",
                },
                "T1.injection_receipt",
            )
            materialized_before = _validate_tensor_receipt(injection["materialized_before"], "T1.materialized_before")
            materialized_after = _validate_tensor_receipt(injection["materialized_after"], "T1.materialized_after")
            base_before = _validate_tensor_receipt(injection["corrupted_base_before"], "T1.base_before")
            base_after = _validate_tensor_receipt(injection["corrupted_base_after"], "T1.base_after")
            persistent_after = _validate_tensor_receipt(injection["persistent_fork_after"], "T1.persistent_after")
            materialized_storage_before = _validate_inventory(injection["materialized_storage_before"], "T1.materialized_storage_before", expected_domain=expected_domain)
            materialized_storage_after = _validate_inventory(injection["materialized_storage_after"], "T1.materialized_storage_after", expected_domain=expected_domain)
            base_storage_before = _validate_inventory(injection["corrupted_base_storage_before"], "T1.base_storage_before", expected_domain=expected_domain)
            base_storage_after = _validate_inventory(injection["corrupted_base_storage_after"], "T1.base_storage_after", expected_domain=expected_domain)
            persistent_storage = _validate_inventory(injection["persistent_fork_storage"], "T1.persistent_storage", expected_domain=expected_domain)
            changed_identically = (
                materialized_before["content_sha256"] == base_before["content_sha256"]
                and materialized_after["content_sha256"] == base_after["content_sha256"]
                and materialized_before["content_sha256"] != materialized_after["content_sha256"]
                and _physical_storage_rows(materialized_storage_before) == _physical_storage_rows(materialized_storage_after)
                and _physical_storage_rows(base_storage_before) == _physical_storage_rows(base_storage_after)
            )
            persistent_alias = (
                persistent_after["content_sha256"] == base_after["content_sha256"]
                and [
                    (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
                    for row in persistent_storage["rows"]
                ]
                == [
                    (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
                    for row in base_storage_after["rows"]
                ]
            )
            require(_exact_bool(injection["changed_identically"], "T1.changed_identically") == changed_identically, "T1 change receipt drift")
            require(_exact_bool(injection["persistent_aliases_corrupted_base"], "T1.persistent_alias") == persistent_alias, "T1 persistent alias receipt drift")
            semantic_map = _exact_object(mutant["cross_arm_semantics"], {"deep_materialized", "persistent_fork"}, "T1.semantics")
            semantic_rows = {
                arm: [
                    _validate_semantic_key(
                        item,
                        f"T1.{arm}[{index}]",
                        semantic_steps=semantic_steps,
                        logit_bundle=logit_bundle,
                    )
                    for index, item in enumerate(_exact_list(semantic_map[arm], 1, f"T1.{arm}"))
                ]
                for arm in ("deep_materialized", "persistent_fork")
            }
            common_exact = [semantic_key(item) for item in semantic_rows["deep_materialized"]] == [
                semantic_key(item) for item in semantic_rows["persistent_fork"]
            ]
            require(_exact_bool(mutant["common_mode_cross_arm_exact"], "T1.common_mode") == common_exact, "T1 common-mode drift")
            comparisons = _exact_object(mutant["oracle_comparisons"], {"deep_materialized", "persistent_fork"}, "T1.oracle")
            mutant_pass = all(
                _replay_oracle_comparisons(
                    comparisons[arm],
                    semantic_rows[arm],
                    oracle_rows[:1],
                    f"T1.oracle.{arm}",
                    threshold=threshold,
                    logit_bundle=logit_bundle,
                )
                for arm in ("deep_materialized", "persistent_fork")
            )
            require(_exact_bool(mutant["predicate_passed"], "T1.mutant.predicate_passed") == mutant_pass, "T1 mutant pass drift")
            case_valid = changed_identically and persistent_alias
        elif fault_id == "T2":
            clean = _exact_object(fault["matched_clean"], {"inventories", "forbidden_inventories", "gate"}, "T2.clean")
            clean_inventories = [
                _validate_inventory(item, f"T2.clean.inventory[{index}]", expected_domain=expected_domain)
                for index, item in enumerate(_exact_list(clean["inventories"], 2, "T2.clean.inventories"))
            ]
            clean_forbidden = [
                _validate_inventory(item, f"T2.clean.forbidden[{index}]", expected_domain=expected_domain)
                for index, item in enumerate(_exact_list(clean["forbidden_inventories"], 1, "T2.clean.forbidden"))
            ]
            clean_pass = _validate_disjointness(clean["gate"], clean_inventories, clean_forbidden, "T2.clean.gate")["passed"]
            mutant = _exact_object(fault["mutant"], {"binding", "inventories", "forbidden_inventories", "gate"}, "T2.mutant")
            binding = _exact_object(
                mutant["binding"],
                {
                    "source_path",
                    "target_path",
                    "source_tensor",
                    "target_tensor_before",
                    "target_inventory_before_sha256",
                    "target_inventory_after_sha256",
                    "mutated",
                },
                "T2.binding",
            )
            _validate_tensor_receipt(binding["source_tensor"], "T2.binding.source_tensor")
            _validate_tensor_receipt(binding["target_tensor_before"], "T2.binding.target_tensor_before")
            before = _sha(binding["target_inventory_before_sha256"], "T2.binding.before")
            after = _sha(binding["target_inventory_after_sha256"], "T2.binding.after")
            binding_mutated = before != after
            require(_exact_bool(binding["mutated"], "T2.binding.mutated") == binding_mutated, "T2 binding drift")
            mutant_inventories = [
                _validate_inventory(item, f"T2.mutant.inventory[{index}]", expected_domain=expected_domain)
                for index, item in enumerate(_exact_list(mutant["inventories"], 2, "T2.mutant.inventories"))
            ]
            mutant_forbidden = [
                _validate_inventory(item, f"T2.mutant.forbidden[{index}]", expected_domain=expected_domain)
                for index, item in enumerate(_exact_list(mutant["forbidden_inventories"], 1, "T2.mutant.forbidden"))
            ]
            mutant_pass = _validate_disjointness(mutant["gate"], mutant_inventories, mutant_forbidden, "T2.mutant.gate")["passed"]
            case_valid = binding_mutated and not mutant_pass
        elif fault_id == "T3":
            clean = _exact_object(fault["matched_clean"], {"document_length", "current_length", "next_position", "predicate_passed"}, "T3.clean")
            document_length = _exact_int(clean["document_length"], "T3.clean.document_length", minimum=1)
            current = _exact_int(clean["current_length"], "T3.clean.current_length", minimum=1)
            next_position = _exact_int(clean["next_position"], "T3.clean.next_position", minimum=1)
            clean_pass = current == document_length and next_position == current
            require(_exact_bool(clean["predicate_passed"], "T3.clean.predicate_passed") == clean_pass, "T3 clean drift")
            mutant = _exact_object(fault["mutant"], {"document_length", "current_length_before", "current_length_after", "next_position_after", "predicate_passed"}, "T3.mutant")
            before = _exact_int(mutant["current_length_before"], "T3.mutant.before", minimum=1)
            after = _exact_int(mutant["current_length_after"], "T3.mutant.after", minimum=1)
            mutant_pass = (
                _exact_int(mutant["document_length"], "T3.mutant.document_length", minimum=1) == after
                and _exact_int(mutant["next_position_after"], "T3.mutant.next_position_after", minimum=1) == after
            )
            require(_exact_bool(mutant["predicate_passed"], "T3.mutant.predicate_passed") == mutant_pass, "T3 mutant drift")
            case_valid = after == before + 1
        elif fault_id == "T4":
            clean = _exact_object(
                fault["matched_clean"],
                {"expected_pre_content_sha256", "observed_clean_content_sha256", "packed_state_content_sha256", "predicate_passed"},
                "T4.clean",
            )
            expected_digest = _sha(clean["expected_pre_content_sha256"], "T4.clean.expected")
            observed_digest = _sha(clean["observed_clean_content_sha256"], "T4.clean.observed")
            require(clean["packed_state_content_sha256"] == expected_digest, "T4 clean digest alias drift")
            clean_pass = expected_digest == observed_digest
            require(_exact_bool(clean["predicate_passed"], "T4.clean.predicate_passed") == clean_pass, "T4 clean pass drift")
            mutant = _exact_object(fault["mutant"], {"target", "pre_value", "post_value", "packed_state_content_sha256", "predicate_passed"}, "T4.mutant")
            require(mutant["target"] == "PackedLowerReplayState.document_residual.data[0]", "T4 target drift")
            pre_value = _exact_number(mutant["pre_value"], "T4.mutant.pre_value")
            post_value = _exact_number(mutant["post_value"], "T4.mutant.post_value")
            mutant_digest = _sha(mutant["packed_state_content_sha256"], "T4.mutant.digest")
            mutant_pass = mutant_digest == expected_digest
            require(_exact_bool(mutant["predicate_passed"], "T4.mutant.predicate_passed") == mutant_pass, "T4 mutant pass drift")
            case_valid = mutant_digest != expected_digest and pre_value != post_value
        elif fault_id == "T5":
            clean = _exact_object(
                fault["matched_clean"],
                {"oracle_passed", "state_cross_arm_exact", "output_cross_arm_exact"},
                "T5.clean",
            )
            clean_oracle = _exact_bool(clean["oracle_passed"], "T5.clean.oracle")
            clean_cross = _exact_bool(clean["state_cross_arm_exact"], "T5.clean.state_cross_arm")
            clean_output = _exact_bool(clean["output_cross_arm_exact"], "T5.clean.output_cross_arm")
            clean_pass = clean_output
            mutant = _exact_object(
                fault["mutant"],
                {
                    "target_path",
                    "target_pre",
                    "target_post",
                    "target_storage_before",
                    "target_storage_after",
                    "one_element_delta",
                    "executions",
                    "cross_arm_semantics",
                    "oracle_comparisons",
                    "state_cross_arm_exact",
                    "output_cross_arm_exact",
                    "oracle_passed",
                },
                "T5.mutant",
            )
            pre = _validate_tensor_receipt(mutant["target_pre"], "T5.target_pre")
            post = _validate_tensor_receipt(mutant["target_post"], "T5.target_post")
            require(type(mutant["target_path"]) is str and mutant["target_path"], "T5 target path invalid")
            require(_exact_number(mutant["one_element_delta"], "T5.delta") != 0.0, "T5 delta is zero")
            storage_before = _validate_inventory(
                mutant["target_storage_before"], "T5.storage_before", expected_domain=expected_domain
            )
            storage_after = _validate_inventory(
                mutant["target_storage_after"], "T5.storage_after", expected_domain=expected_domain
            )
            executions = _exact_object(
                mutant["executions"], {"deep_materialized", "persistent_fork"}, "T5.executions"
            )
            completed_by_arm: dict[str, bool] = {}
            for arm in ("deep_materialized", "persistent_fork"):
                execution = _exact_object(
                    executions[arm],
                    {"completed", "outputs_available", "runtime_exception", "ordinary_assertion_triggered"},
                    f"T5.execution.{arm}",
                )
                completed = _exact_bool(execution["completed"], f"T5.execution.{arm}.completed")
                outputs = _exact_bool(execution["outputs_available"], f"T5.execution.{arm}.outputs")
                ordinary = _exact_bool(
                    execution["ordinary_assertion_triggered"], f"T5.execution.{arm}.ordinary"
                )
                if completed:
                    require(outputs and execution["runtime_exception"] is None and not ordinary, f"T5 {arm} completion receipt drift")
                else:
                    require(not outputs, f"T5 {arm} failed execution claims outputs")
                    exception = _exact_object(
                        execution["runtime_exception"], {"type", "message_sha256"}, f"T5.execution.{arm}.exception"
                    )
                    require(type(exception["type"]) is str and exception["type"], f"T5 {arm} exception type invalid")
                    require(
                        arm == "persistent_fork" and exception["type"] == "builtins.AssertionError",
                        f"T5 {arm} exception is not the preregistered mutated-arm assertion",
                    )
                    _sha(exception["message_sha256"], f"T5.execution.{arm}.message")
                    require(ordinary == exception["type"].endswith(".AssertionError"), f"T5 {arm} assertion classification drift")
                completed_by_arm[arm] = completed
            semantic_map = _exact_object(mutant["cross_arm_semantics"], {"deep_materialized", "persistent_fork"}, "T5.semantics")
            semantic_rows = {
                arm: [
                    _validate_semantic_key(
                        item,
                        f"T5.{arm}[{index}]",
                        semantic_steps=semantic_steps,
                        logit_bundle=logit_bundle,
                    )
                    for index, item in enumerate(
                        _exact_list(semantic_map[arm], 1 if completed_by_arm[arm] else 0, f"T5.{arm}")
                    )
                ]
                for arm in ("deep_materialized", "persistent_fork")
            }
            completed = all(completed_by_arm.values())
            state_cross_arm = completed and [semantic_key(item) for item in semantic_rows["deep_materialized"]] == [
                semantic_key(item) for item in semantic_rows["persistent_fork"]
            ]
            def output_key(item: Mapping[str, Any]) -> dict[str, Any]:
                return {
                    "request_index": item["request_index"],
                    "query_token_ids_sha256": item["query_token_ids_sha256"],
                    "generated_token_ids": item["generated_token_ids"],
                    "step_logit_sha256": item["step_logit_sha256"],
                }
            output_cross_arm = completed and [output_key(item) for item in semantic_rows["deep_materialized"]] == [
                output_key(item) for item in semantic_rows["persistent_fork"]
            ]
            require(_exact_bool(mutant["state_cross_arm_exact"], "T5.state_cross_arm") == state_cross_arm, "T5 state cross-arm drift")
            require(_exact_bool(mutant["output_cross_arm_exact"], "T5.output_cross_arm") == output_cross_arm, "T5 output cross-arm drift")
            comparisons = _exact_object(mutant["oracle_comparisons"], {"deep_materialized", "persistent_fork"}, "T5.oracle")
            oracle_by_arm: dict[str, bool] = {}
            for arm in ("deep_materialized", "persistent_fork"):
                if completed_by_arm[arm]:
                    oracle_by_arm[arm] = _replay_oracle_comparisons(
                        comparisons[arm], semantic_rows[arm], oracle_rows[:1], f"T5.oracle.{arm}",
                        threshold=threshold, logit_bundle=logit_bundle,
                    )
                else:
                    _exact_list(comparisons[arm], 0, f"T5.oracle.{arm}")
                    oracle_by_arm[arm] = False
            oracle_pass = completed and all(oracle_by_arm.values())
            require(_exact_bool(mutant["oracle_passed"], "T5.oracle_passed") == oracle_pass, "T5 oracle summary drift")
            mutant_pass = output_cross_arm
            case_valid = (
                pre["content_sha256"] != post["content_sha256"]
                and storage_before["rows"] == storage_after["rows"]
                and completed_by_arm["deep_materialized"]
                and oracle_by_arm["deep_materialized"]
            )
        execution = _exact_object(
            fault["execution_outcome"],
            {"completed", "outputs_available", "runtime_exception", "ordinary_assertion_triggered"},
            f"fault {fault_id}.execution_outcome",
        )
        if fault_id == "T1":
            expected_execution = {"completed": True, "outputs_available": True, "runtime_exception": None, "ordinary_assertion_triggered": False}
        elif fault_id == "T5":
            expected_execution = {
                "completed": all(completed_by_arm.values()),
                "outputs_available": all(executions[arm]["outputs_available"] for arm in executions),
                "runtime_exception": executions["persistent_fork"]["runtime_exception"],
                "ordinary_assertion_triggered": any(executions[arm]["ordinary_assertion_triggered"] for arm in executions),
            }
        else:
            expected_execution = {"completed": True, "outputs_available": False, "runtime_exception": None, "ordinary_assertion_triggered": False}
        require(execution == expected_execution, f"fault {fault_id} execution outcome drift")
        detector = _exact_object(fault["detector_vector"], {"matched_clean", "mutant"}, f"fault {fault_id}.detector_vector")
        if fault_id == "T1":
            clean_vector = {
                "INDEPENDENT_DENSE_SEMANTIC_ORACLE": clean_pass,
                "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": clean_n1["cross_arm"],
            }
            mutant_vector = {
                "INDEPENDENT_DENSE_SEMANTIC_ORACLE": mutant_pass,
                "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": common_exact,
            }
        elif fault_id == "T5":
            clean_vector = {
                "INDEPENDENT_DENSE_SEMANTIC_ORACLE": clean_oracle,
                "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": clean_cross,
                "STATE_CROSS_ARM": clean_cross,
                "DOWNSTREAM_OUTPUT_CONSISTENCY": clean_output,
            }
            mutant_vector = {
                "INDEPENDENT_DENSE_SEMANTIC_ORACLE": oracle_pass,
                "DEEP_MATERIALIZED_EQUALS_PERSISTENT_FORK": state_cross_arm,
                "STATE_CROSS_ARM": state_cross_arm,
                "DOWNSTREAM_OUTPUT_CONSISTENCY": output_cross_arm,
            }
        else:
            clean_vector = {contract["expected_predicate"]: clean_pass}
            mutant_vector = {contract["expected_predicate"]: mutant_pass}
        require(detector["matched_clean"] == clean_vector, f"fault {fault_id} clean detector vector drift")
        require(detector["mutant"] == mutant_vector, f"fault {fault_id} mutant detector vector drift")
        classification = classify_detector_vector(
            expected_predicate=contract["expected_predicate"],
            matched_clean=clean_vector,
            mutant=mutant_vector,
        )
        require(fault["classification"] == classification, f"fault {fault_id} classification does not replay")
        require(_exact_bool(fault["fault_case_valid"], f"fault {fault_id}.fault_case_valid") == case_valid, f"fault {fault_id} validity drift")
        replayed.append({"fault_id": fault_id, "classification": classification, "fault_case_valid": case_valid})
    return replayed


def replay_shard(
    shard: Any,
    *,
    static_manifest: Mapping[str, Any],
    expected_rank: int,
    static_manifest_raw_sha256: str,
    source_manifest_raw_sha256: str,
    sidecar_dir: Path,
    model_authority_raw_sha256: str,
    gpu_assignment_rows: Sequence[Mapping[str, Any]],
    gpu_assignment_raw_sha256: str,
) -> dict[str, Any]:
    row = _exact_object(
        shard,
        {
            "schema_version",
            "protocol",
            "status",
            "rank",
            "world_size",
            "scientific_run_valid",
            "passed",
            "static_manifest_raw_sha256",
            "source_manifest_raw_sha256",
            "model_artifact_ledger_raw_sha256",
            "model_weight_ledger_raw_sha256",
            "gpu_assignment_raw_sha256",
            "formal_config_sha256",
            "model_identity",
            "input",
            "hardware",
            "environment",
            "dispatch_provenance",
            "persistent_base",
            "dense_oracle",
            "fanouts",
            "cross_n",
            "targets",
            "clean_audit",
            "fault_suite",
            "claim_boundary",
            "logit_sidecar",
        },
        f"rank {expected_rank} shard",
    )
    require(row["schema_version"] == SHARD_SCHEMA and row["protocol"] == PROTOCOL, "shard identity drift")
    require(row["status"] == "completed", "shard is not completed")
    require(_exact_int(row["rank"], "shard.rank") == expected_rank, "rank order drift")
    require(_exact_int(row["world_size"], "shard.world_size", minimum=1) == WORLD_SIZE, "world size drift")
    require(_exact_bool(row["scientific_run_valid"], "shard.scientific_run_valid"), "shard invalid")
    _exact_bool(row["passed"], "shard.passed")
    require(row["static_manifest_raw_sha256"] == static_manifest_raw_sha256, "shard/static binding drift")
    require(row["source_manifest_raw_sha256"] == source_manifest_raw_sha256, "shard/source binding drift")
    require(row["formal_config_sha256"] == static_manifest["formal_config_sha256"], "config digest drift")
    require(row["model_artifact_ledger_raw_sha256"] == static_manifest["model"]["model_artifact_ledger_raw_sha256"], "model artifact ledger drift")
    require(row["model_weight_ledger_raw_sha256"] == static_manifest["model"]["model_weight_ledger_raw_sha256"], "model weight ledger drift")
    require(row["gpu_assignment_raw_sha256"] == gpu_assignment_raw_sha256, "GPU assignment binding drift")
    model_identity = _exact_object(
        row["model_identity"],
        {
            "model_id",
            "model_revision",
            "artifact_ledger",
            "weight_ledger",
            "authority_raw_sha256",
            "authority_stat_validation",
        },
        "model_identity",
    )
    require(model_identity["model_id"] == static_manifest["model"]["model_id"], "model identity ID drift")
    require(model_identity["model_revision"] == static_manifest["model"]["model_revision"], "model identity revision drift")
    require(model_identity["artifact_ledger"] == static_manifest["model"]["artifact_ledger_receipt"], "model artifact receipt drift")
    require(model_identity["weight_ledger"] == static_manifest["model"]["weight_ledger_receipt"], "model weight receipt drift")
    require(model_identity["authority_raw_sha256"] == model_authority_raw_sha256, "model authority binding drift")
    authority_validation = _exact_object(
        model_identity["authority_stat_validation"],
        {"verified", "file_count", "stat_snapshot_sha256"},
        "model authority stat validation",
    )
    require(_exact_bool(authority_validation["verified"], "model authority verified"), "model authority not verified")
    require(_exact_int(authority_validation["file_count"], "model authority file count", minimum=1) == 21, "model authority file count drift")
    _sha(authority_validation["stat_snapshot_sha256"], "model authority stat digest")
    config = static_manifest["formal_config"]
    semantic_steps = _exact_int(config["semantic_steps"], "static.semantic_steps", minimum=1)
    document_tokens = _exact_int(config["document_tokens"], "static.document_tokens", minimum=1)
    query_tokens = _exact_int(config["query_tokens"], "static.query_tokens", minimum=1)
    threshold = _exact_number(static_manifest["oracle_contract"]["relative_l2_threshold"], "static.oracle_threshold")
    expected_domain = sha256_bytes(static_manifest["storage_receipt_salt"].encode("utf-8"))
    logit_bundle = _load_logit_bundle(
        sidecar_dir,
        row["logit_sidecar"],
        expected_rank=expected_rank,
    )

    shard_input = _exact_object(
        row["input"],
        {
            "pg19_train_only",
            "source_object",
            "source_id",
            "document_start_token",
            "document_token_ids_sha256",
            "query_token_ids_sha256",
            "rank_input_sha256",
        },
        "shard.input",
    )
    require(_exact_bool(shard_input["pg19_train_only"], "shard.input.pg19_train_only"), "non-PG19 input")
    frozen_input = static_manifest["rank_inputs"][expected_rank]
    expected_input = {
        "pg19_train_only": True,
        "source_object": frozen_input["source_object"],
        "source_id": frozen_input["source_id"],
        "document_start_token": frozen_input["document_start_token"],
        "document_token_ids_sha256": frozen_input["document_token_ids_sha256"],
        "query_token_ids_sha256": [item["token_ids_sha256"] for item in frozen_input["queries"]],
        "rank_input_sha256": frozen_input["rank_input_sha256"],
    }
    require(shard_input == expected_input, "shard input does not equal frozen rank input")

    hardware = _exact_object(
        row["hardware"],
        {"cuda_visible_devices", "uuid", "name", "total_memory_mib", "compute_capability", "bf16_supported"},
        "hardware",
    )
    require(hardware["cuda_visible_devices"] == hardware["uuid"], "rank CUDA UUID mapping drift")
    require(type(hardware["uuid"]) is str and hardware["uuid"].startswith("GPU-"), "GPU UUID invalid")
    _exact_int(hardware["total_memory_mib"], "hardware.total_memory_mib", minimum=1)
    capability = _exact_list(hardware["compute_capability"], 2, "hardware.compute_capability")
    require(all(type(item) is int and item >= 0 for item in capability), "compute capability invalid")
    require("H20" in hardware["name"] and capability == [9, 0], "formal GPU is not an H20/cc9.0")
    require(_exact_bool(hardware["bf16_supported"], "hardware.bf16_supported"), "formal GPU lacks BF16")
    expected_hardware = {
        "cuda_visible_devices": gpu_assignment_rows[expected_rank]["uuid"],
        **{
            key: value
            for key, value in gpu_assignment_rows[expected_rank].items()
            if key not in {"rank", "visible_index"}
        },
    }
    require(hardware == expected_hardware, "shard hardware differs from pre-output GPU assignment")

    environment = _exact_object(
        row["environment"],
        {"python", "torch", "cuda", "transformers", "model_geometry"},
        "environment",
    )
    for field in ("python", "torch", "cuda", "transformers"):
        require(environment[field] == static_manifest["environment_contract"][field], f"environment {field} drift")
    geometry = _exact_object(
        environment["model_geometry"],
        {"model_type", "num_layers", "layer_types", "split_depth", "matches_frozen"},
        "model_geometry",
    )
    require(geometry["model_type"] == "qwen3_5_moe_text", "model type drift")
    require(_exact_int(geometry["num_layers"], "model_geometry.num_layers", minimum=1) == 40, "layer count drift")
    require(_exact_int(geometry["split_depth"], "model_geometry.split_depth") == config["split_depth"], "split depth drift")
    require(geometry["layer_types"] == static_manifest["model"]["layer_types"], "layer type geometry drift")
    require(_exact_bool(geometry["matches_frozen"], "model_geometry.matches_frozen"), "model geometry mismatch")

    dispatch = _exact_object(
        row["dispatch_provenance"],
        {
            "adapter",
            "cache",
            "manual_suffix_method",
            "layer_forward_types",
            "same_receipt_for_both_arms",
            "compiled_kernel_fingerprint",
            "autotuning_choice_fingerprint",
        },
        "dispatch",
    )
    dispatch_pass = (
        dispatch["adapter"] == "qcomem_torch.TorchSplitCausalLM"
        and dispatch["cache"] == "transformers.cache_utils.DynamicCache"
        and dispatch["manual_suffix_method"] == "TorchSplitCausalLM.run_suffix_cached_last_logits"
        and _exact_bool(dispatch["same_receipt_for_both_arms"], "dispatch.same_receipt")
        and dispatch["compiled_kernel_fingerprint"] is None
        and dispatch["autotuning_choice_fingerprint"] is None
        and type(dispatch["layer_forward_types"]) is list
        and bool(dispatch["layer_forward_types"])
    )

    base = _exact_object(row["persistent_base"], {"before", "after", "storage", "content_immutable"}, "persistent_base")
    before = _validate_state_receipt(base["before"], "persistent_base.before")
    after = _validate_state_receipt(base["after"], "persistent_base.after")
    _validate_inventory(base["storage"], "persistent_base.storage", expected_domain=expected_domain)
    immutable = before["state_content_sha256"] == after["state_content_sha256"]
    require(_exact_bool(base["content_immutable"], "persistent_base.content_immutable") == immutable, "prefix immutability flag drift")

    dense = _exact_object(row["dense_oracle"], {"contract", "semantics", "all_clean_arms_passed"}, "dense_oracle")
    require(dense["contract"] == static_manifest["oracle_contract"], "oracle contract drift")
    oracle_rows = []
    for index, raw_oracle in enumerate(_exact_list(dense["semantics"], max(FANOUTS), "dense_oracle.semantics")):
        oracle = _exact_object(
            raw_oracle,
            {
                "oracle_path",
                "generated_token_ids",
                "step_logit_sha256",
                "step_logit_record_ids",
                "semantic_steps",
            },
            f"oracle[{index}]",
        )
        require(oracle["oracle_path"] == "AutoModelForImageTextToText.full_last_logits_dense_recompute", "oracle path drift")
        require(_exact_int(oracle["semantic_steps"], f"oracle[{index}].steps", minimum=1) == semantic_steps, "oracle step drift")
        tokens = _exact_list(oracle["generated_token_ids"], semantic_steps, f"oracle[{index}].tokens")
        require(all(type(token) is int and token >= 0 for token in tokens), "oracle token type drift")
        digests = _exact_list(oracle["step_logit_sha256"], semantic_steps, f"oracle[{index}].logits")
        record_ids = _exact_list(oracle["step_logit_record_ids"], semantic_steps, f"oracle[{index}].record_ids")
        for step, (digest, record_id) in enumerate(zip(digests, record_ids)):
            _sha(digest, "oracle logit digest")
            require(type(record_id) is str and record_id in logit_bundle, "oracle logit record missing")
            tensor, record = logit_bundle[record_id]
            require(record["content_sha256"] == digest, "oracle logit SHA/record drift")
            require(int(tensor.argmax().item()) == tokens[step], "oracle token differs from FP32 sidecar")
        oracle_rows.append(oracle)

    fanouts = _exact_object(row["fanouts"], {str(item) for item in FANOUTS}, "fanouts")
    replayed_fanouts: dict[str, Any] = {}
    for fanout in FANOUTS:
        cell = _exact_object(
            fanouts[str(fanout)],
            {"fanout", "arms", "oracle_comparisons", "cross_arm_exact", "all_storage_ownership_predicates_passed"},
            f"fanout {fanout}",
        )
        require(_exact_int(cell["fanout"], f"fanout {fanout}.fanout", minimum=1) == fanout, "fanout identity drift")
        arms = _exact_object(cell["arms"], {"deep_materialized", "persistent_fork"}, f"fanout {fanout}.arms")
        comparisons = _exact_object(cell["oracle_comparisons"], {"deep_materialized", "persistent_fork"}, f"fanout {fanout}.oracle")
        replayed_arms = {}
        for arm_name in ("deep_materialized", "persistent_fork"):
            replay = _replay_arm(
                arms[arm_name],
                f"fanout {fanout}.{arm_name}",
                arm_name=arm_name,
                fanout=fanout,
                semantic_steps=semantic_steps,
                document_tokens=document_tokens,
                query_tokens=query_tokens,
                depth=config["split_depth"],
                expected_domain=expected_domain,
                logit_bundle=logit_bundle,
            )
            replay["oracle_passed"] = _replay_oracle_comparisons(
                comparisons[arm_name],
                replay["semantics"],
                oracle_rows[:fanout],
                f"fanout {fanout}.oracle.{arm_name}",
                threshold=threshold,
                logit_bundle=logit_bundle,
            )
            replayed_arms[arm_name] = replay
        cross_arm = [semantic_key(item) for item in replayed_arms["deep_materialized"]["semantics"]] == [
            semantic_key(item) for item in replayed_arms["persistent_fork"]["semantics"]
        ]
        require(_exact_bool(cell["cross_arm_exact"], f"fanout {fanout}.cross_arm_exact") == cross_arm, "cross-arm flag drift")
        ownership = all(item["ownership_passed"] for item in replayed_arms.values())
        require(
            _exact_bool(cell["all_storage_ownership_predicates_passed"], f"fanout {fanout}.ownership") == ownership,
            "ownership summary drift",
        )
        replayed_fanouts[str(fanout)] = {"arms": replayed_arms, "cross_arm": cross_arm, "ownership": ownership}

    cross_n_claimed = _exact_object(row["cross_n"], {"deep_materialized", "persistent_fork"}, "cross_n")
    cross_n_pass = True
    for arm_name in ("deep_materialized", "persistent_fork"):
        claim = _exact_object(cross_n_claimed[arm_name], {"passed", "compared_requests"}, f"cross_n.{arm_name}")
        n1 = [semantic_key(item) for item in replayed_fanouts["1"]["arms"][arm_name]["semantics"]]
        n2_prefix = [semantic_key(item) for item in replayed_fanouts["2"]["arms"][arm_name]["semantics"][:1]]
        passed = n1 == n2_prefix
        require(_exact_bool(claim["passed"], f"cross_n.{arm_name}.passed") == passed, "cross-N flag drift")
        require(_exact_int(claim["compared_requests"], f"cross_n.{arm_name}.count") == 1, "cross-N count drift")
        cross_n_pass = cross_n_pass and passed

    oracle_clean = all(item["oracle_passed"] for fanout in replayed_fanouts.values() for item in fanout["arms"].values())
    require(_exact_bool(dense["all_clean_arms_passed"], "dense_oracle.all_clean") == oracle_clean, "oracle summary drift")
    predicates = {
        "frozen_identity": True,
        "prefix_immutability": immutable,
        "private_ownership": all(item["ownership"] for item in replayed_fanouts.values()),
        "dispatch_provenance": dispatch_pass,
        "cross_arm_equivalence": all(item["cross_arm"] for item in replayed_fanouts.values()),
        "cross_n_prefix_consistency": cross_n_pass,
    }
    expected_targets = build_target_rows(predicates)
    require(row["targets"] == expected_targets, "producer target rows/statuses do not replay")
    applicable_clean = all(
        target["predicate_passed"] is True
        for target in expected_targets
        if target["applicability"] == "applicable"
    ) and oracle_clean
    clean = _exact_object(
        row["clean_audit"],
        {"all_applicable_predicates_passed", "independent_dense_oracle_passed", "target_status_vector"},
        "clean_audit",
    )
    require(_exact_bool(clean["all_applicable_predicates_passed"], "clean_audit.all") == applicable_clean, "clean summary drift")
    require(_exact_bool(clean["independent_dense_oracle_passed"], "clean_audit.oracle") == oracle_clean, "clean oracle drift")
    require(clean["target_status_vector"] == [item["status"] for item in expected_targets], "clean status vector drift")

    replayed_faults = _replay_faults(
        row["fault_suite"],
        clean_n1=replayed_fanouts["1"],
        oracle_rows=oracle_rows,
        semantic_steps=semantic_steps,
        threshold=threshold,
        expected_domain=expected_domain,
        logit_bundle=logit_bundle,
    )
    referenced_record_ids = [
        record_id
        for oracle in oracle_rows
        for record_id in oracle["step_logit_record_ids"]
    ]
    for fanout in FANOUTS:
        for arm_name in ("deep_materialized", "persistent_fork"):
            referenced_record_ids.extend(
                record_id
                for semantic in replayed_fanouts[str(fanout)]["arms"][arm_name]["semantics"]
                for record_id in semantic["step_logit_record_ids"]
            )
    for fault_index in (0, 4):
        fault_semantics = row["fault_suite"][fault_index]["mutant"]["cross_arm_semantics"]
        for arm_name in ("deep_materialized", "persistent_fork"):
            referenced_record_ids.extend(
                record_id
                for semantic in fault_semantics[arm_name]
                for record_id in semantic["step_logit_record_ids"]
            )
    require(len(referenced_record_ids) == len(set(referenced_record_ids)), "a logit sidecar record is referenced more than once")
    require(set(referenced_record_ids) == set(logit_bundle), "logit sidecar has missing or orphan records")
    expected_faults = all(
        item["classification"]["outcome"] == "detected_expected_predicate" and item["fault_case_valid"]
        for item in replayed_faults
    )
    derived_pass = applicable_clean and expected_faults
    require(_exact_bool(row["passed"], "shard.passed") == derived_pass, "producer shard pass does not replay")
    require(row["claim_boundary"] == static_manifest["claim_boundary"], "claim boundary drift")
    return {
        "rank": expected_rank,
        "source_object": shard_input["source_object"],
        "gpu_uuid": hardware["uuid"],
        "logit_sidecar": {
            "rank": expected_rank,
            "path": row["logit_sidecar"]["logical_name"],
            "bytes": row["logit_sidecar"]["bytes"],
            "sha256": row["logit_sidecar"]["sha256"],
            "record_count": row["logit_sidecar"]["record_count"],
        },
        "clean_passed": applicable_clean,
        "faults": replayed_faults,
        "targets": expected_targets,
        "derived_passed": derived_pass,
    }


def aggregate_shards(
    shard_paths: Iterable[Path],
    *,
    static_manifest: Mapping[str, Any],
    sidecar_dir: Path,
    static_manifest_raw_sha256: str,
    source_manifest_raw_sha256: str,
    model_authority_raw_sha256: str,
    gpu_assignment: Mapping[str, Any],
    gpu_assignment_raw_sha256: str,
) -> dict[str, Any]:
    static_manifest = validate_static_manifest(static_manifest)
    require(static_manifest.get("source_manifest_raw_sha256") == source_manifest_raw_sha256, "static/source binding drift")
    require(static_manifest.get("formal_config_sha256") == sha256_json(static_manifest["formal_config"]), "static config digest drift")
    require(static_manifest["formal_config"].get("split_depth") == 7, "formal split depth drift")
    require(
        static_manifest.get("hardware_contract")
        == {
            "world_size": WORLD_SIZE,
            "gpu_name": "NVIDIA H20-3e",
            "compute_capability": [9, 0],
            "bf16_required": True,
            "assignment_frozen_pre_output": True,
        },
        "static hardware contract drift",
    )
    hardware_contract = static_manifest["hardware_contract"]
    _exact_int(hardware_contract["world_size"], "static.hardware.world_size", minimum=1)
    hardware_capability = _exact_list(hardware_contract["compute_capability"], 2, "static.hardware.capability")
    require(all(type(item) is int for item in hardware_capability), "static hardware capability types drift")
    _exact_bool(hardware_contract["bf16_required"], "static.hardware.bf16_required")
    _exact_bool(hardware_contract["assignment_frozen_pre_output"], "static.hardware.assignment_frozen")
    record_mapping = _exact_object(
        static_manifest.get("portable_record_mapping"),
        {"identity", "ownership", "execution", "accounting", "tail_event", "dispatch"},
        "static.portable_record_mapping",
    )
    require(record_mapping["tail_event"].startswith("not_applicable:") and record_mapping["dispatch"].startswith("partial:"), "portable record mapping boundary drift")
    gpu_assignment_rows = validate_gpu_assignment(gpu_assignment)
    rank_inputs = static_manifest.get("rank_inputs")
    require(type(rank_inputs) is list and len(rank_inputs) == WORLD_SIZE, "static rank input cardinality drift")
    for expected_rank, rank_input in enumerate(rank_inputs):
        rank_input = _exact_object(
            rank_input,
            {"rank", "source_id", "source_object", "document_start_token", "document_end_token_exclusive", "document_token_ids", "document_token_ids_sha256", "queries", "rank_input_sha256"},
            f"static.rank_inputs[{expected_rank}]",
        )
        require(_exact_int(rank_input["rank"], "static rank") == expected_rank, "static rank input order drift")
        require(type(rank_input["source_id"]) is str and rank_input["source_id"] and type(rank_input["source_object"]) is str and rank_input["source_object"].startswith("train/") and rank_input["source_object"].endswith(".txt"), "static rank source drift")
        start = _exact_int(rank_input["document_start_token"], "static document start")
        require(_exact_int(rank_input["document_end_token_exclusive"], "static document end", minimum=1) == start + 256, "static document boundary drift")
        document_ids = _exact_list(rank_input["document_token_ids"], 256, "static document IDs")
        require(all(type(token) is int and token >= 0 for token in document_ids), "static document token type drift")
        require(rank_input["document_token_ids_sha256"] == sha256_bytes(torch.tensor([document_ids], dtype=torch.int64).numpy().tobytes()), "static document raw-int64 digest drift")
        queries = _exact_list(rank_input["queries"], 2, "static queries")
        for request, raw_query in enumerate(queries):
            query = _exact_object(raw_query, {"request_index", "source_token_offset", "token_ids", "token_ids_sha256"}, f"static query {request}")
            require(_exact_int(query["request_index"], "static query request") == request, "static query order drift")
            require(
                _exact_int(query["source_token_offset"], "static query offset")
                == start + 256 + 24 + request * 32,
                "static query offset drift",
            )
            token_ids = _exact_list(query["token_ids"], 24, "static query token IDs")
            require(all(type(token) is int and token >= 0 for token in token_ids), "static query token type drift")
            require(query["token_ids_sha256"] == sha256_bytes(torch.tensor([token_ids], dtype=torch.int64).numpy().tobytes()), "static query raw-int64 digest drift")
        payload = {key: value for key, value in rank_input.items() if key != "rank_input_sha256"}
        require(rank_input.get("rank_input_sha256") == sha256_json(payload), "static rank input digest drift")
    require(static_manifest.get("rank_inputs_sha256") == sha256_json(rank_inputs), "static rank input set digest drift")
    require(len({row["source_object"] for row in rank_inputs}) == WORLD_SIZE, "static PG-19 books are not distinct")
    paths = list(shard_paths)
    require(len(paths) == WORLD_SIZE, f"expected {WORLD_SIZE} shards, found {len(paths)}")
    parsed: dict[int, tuple[dict[str, Any], Path, bytes]] = {}
    for path in paths:
        raw = path.read_bytes()
        require(raw == canonical_json_bytes(json.loads(raw)) + b"\n", f"shard is not canonical JSON: {path.name}")
        shard = json.loads(raw)
        rank = shard.get("rank")
        require(type(rank) is int and 0 <= rank < WORLD_SIZE and rank not in parsed, "duplicate/invalid rank")
        parsed[rank] = (shard, path, raw)
    require(set(parsed) == set(range(WORLD_SIZE)), "shard ranks are incomplete")
    replays = [
        replay_shard(
            parsed[rank][0],
            static_manifest=static_manifest,
            expected_rank=rank,
            static_manifest_raw_sha256=static_manifest_raw_sha256,
            source_manifest_raw_sha256=source_manifest_raw_sha256,
            sidecar_dir=sidecar_dir,
            model_authority_raw_sha256=model_authority_raw_sha256,
            gpu_assignment_rows=gpu_assignment_rows,
            gpu_assignment_raw_sha256=gpu_assignment_raw_sha256,
        )
        for rank in range(WORLD_SIZE)
    ]
    source_objects = [item["source_object"] for item in replays]
    gpu_uuids = [item["gpu_uuid"] for item in replays]
    require(len(set(source_objects)) == WORLD_SIZE, "the eight ranks do not use distinct PG-19 books")
    require(len(set(gpu_uuids)) == WORLD_SIZE, "the eight ranks do not use distinct GPU UUIDs")
    require(gpu_uuids == [row["uuid"] for row in gpu_assignment_rows], "aggregate GPU rank assignment drift")
    receipts = [
        {
            "rank": rank,
            "path": parsed[rank][1].name,
            "bytes": len(parsed[rank][2]),
            "sha256": sha256_bytes(parsed[rank][2]),
            "gpu_uuid": replays[rank]["gpu_uuid"],
        }
        for rank in range(WORLD_SIZE)
    ]
    clean_all = all(item["clean_passed"] for item in replays)
    fault_rows = [fault for replay in replays for fault in replay["faults"]]
    expected_faults = all(
        item["classification"]["outcome"] == "detected_expected_predicate" and item["fault_case_valid"]
        for item in fault_rows
    )
    target_rows = []
    for contract in TARGET_CONTRACT:
        observed = [
            next(row for row in replay["targets"] if row["target"] == contract["target"])
            for replay in replays
        ]
        statuses = [row["status"] for row in observed]
        if contract["applicability"] == "not_applicable":
            status = "not_applicable"
        elif "open" in statuses:
            status = "open"
        else:
            status = contract["maximum_status"]
        target_rows.append(
            {
                "target_index": contract["target_index"],
                "target": contract["target"],
                "status": status,
                "rank_statuses": statuses,
                "exact_missingness": observed[0]["exact_missingness"],
            }
        )
    positive = clean_all and expected_faults
    outcomes = (
        "detected_expected_predicate",
        "detected_wrong_predicate",
        "clean_false_positive",
        "escaped",
    )
    return {
        "schema_version": AGGREGATE_SCHEMA,
        "protocol": PROTOCOL,
        "scientific_run_valid": True,
        "passed": positive,
        "scientific_outcome": (
            "valid_positive_transformers_runtime_transfer"
            if positive
            else "valid_negative_transformers_runtime_transfer"
        ),
        "rank_count": WORLD_SIZE,
        "distinct_pg19_train_books": len(set(source_objects)),
        "distinct_gpu_uuids": len(set(gpu_uuids)),
        "rank_gpu_assignment": [{"rank": index, "gpu_uuid": uuid} for index, uuid in enumerate(gpu_uuids)],
        "fanouts": list(FANOUTS),
        "clean_all_applicable_predicates_passed": clean_all,
        "all_faults_detected_at_expected_predicate": expected_faults,
        "fault_case_count": len(fault_rows),
        "fault_outcome_counts": {
            name: sum(item["classification"]["outcome"] == name for item in fault_rows)
            for name in outcomes
        },
        "targets": target_rows,
        "status_vector": [row["status"] for row in target_rows],
        "overall_contract_status": (
            "open" if any(row["status"] == "open" for row in target_rows) else "partial"
        ),
        "static_manifest_raw_sha256": static_manifest_raw_sha256,
        "source_manifest_raw_sha256": source_manifest_raw_sha256,
        "formal_config_sha256": static_manifest["formal_config_sha256"],
        "rank_inputs_sha256": static_manifest["rank_inputs_sha256"],
        "model_artifact_ledger_raw_sha256": static_manifest["model"]["model_artifact_ledger_raw_sha256"],
        "model_weight_ledger_raw_sha256": static_manifest["model"]["model_weight_ledger_raw_sha256"],
        "model_authority_raw_sha256": model_authority_raw_sha256,
        "gpu_assignment_raw_sha256": gpu_assignment_raw_sha256,
        "shards": receipts,
        "logit_sidecars": [item["logit_sidecar"] for item in replays],
        "claim_boundary": static_manifest["claim_boundary"],
    }


__all__ = [
    "AGGREGATE_SCHEMA",
    "FAULT_CONTRACT",
    "FANOUTS",
    "PROTOCOL",
    "SHARD_SCHEMA",
    "SOURCE_SCHEMA",
    "STATIC_SCHEMA",
    "TARGET_CONTRACT",
    "TensorSlot",
    "TransferEvidenceError",
    "WORLD_SIZE",
    "aggregate_shards",
    "build_target_rows",
    "canonical_json_bytes",
    "classify_pair",
    "compare_logit_steps",
    "disjointness_receipt",
    "iter_tensor_slots",
    "load_bound_json",
    "require",
    "semantic_key",
    "sha256_bytes",
    "sha256_file",
    "sha256_json",
    "state_content_receipt",
    "storage_inventory",
    "tensor_bytes",
    "tensor_receipt",
    "tensor_tree_receipt",
    "validate_source_manifest",
    "write_canonical_json",
]
