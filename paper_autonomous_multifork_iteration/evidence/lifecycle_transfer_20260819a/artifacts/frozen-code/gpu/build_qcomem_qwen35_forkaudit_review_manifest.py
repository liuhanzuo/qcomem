from __future__ import annotations

"""Build path-independent ForkAudit preregistration and receipt manifests.

This is a CPU-only release-governance utility.  It deliberately does not run
the GPU producer.  The preregistration mode canonicalizes the two inputs used
by ``run_qcomem_qwen35_forkaudit_review_revision.py --stage static`` and binds
them to path-free logical artifact receipts.  The receipt mode is reserved for
the later, reviewed producer: it binds the raw bytes of exactly eight fixed
rank shards without trusting producer ``passed`` fields.

No source or destination filesystem path is serialized.  Therefore identical
input bytes and logical metadata produce identical output bytes after a tree is
relocated.  Importing this module and all supported modes must leave CUDA
uninitialized.
"""

import argparse
import csv
import errno
import fcntl
import hashlib
import inspect
import json
import math
import os
import re
import secrets
import shutil
import signal
import stat
import sys
import threading
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence, TextIO

import torch

import build_qcomem_forkaudit_fp32_calibration_manifest as fp32_context
import build_qcomem_forkaudit_rr2_input_manifest as rr2_input
import run_qcomem_qwen35_forkaudit_review_revision as runner


RELEASE_MANIFEST_SCHEMA_VERSION = "qcomem-forkaudit-review-release-manifest-v2"
FORMAL_MODEL_ID = "Qwen/Qwen3.5-35B-A3B"
FORMAL_MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
PRIOR_FP32_CONTEXT_MANIFEST_SHA256 = (
    "fa64f663bb74a190a0a5c0898fda2a55528171c77a91af2b1321c24a5f310a1d"
)
PRIOR_FP32_SOURCES_BY_RANK = (
    "train/10034.txt",
    "train/10.txt",
    "train/10005.txt",
    "train/10017.txt",
    "train/10008.txt",
    "train/10016.txt",
    "train/10010.txt",
    "train/1004.txt",
)
PRIOR_FP32_POSITION_IDS_SHA256 = (
    "2cb56af25344e4730bcd40143fe622dde884056042e5deb53f02db0f9c52ef77"
)
PRIOR_FP32_MASK_SHA256 = (
    "3d276299b11a1e61a15f57970225d181bc88dbbe4c0937e40207d00af6fef9b6"
)
RUNNER_RR2_FROZEN_FIELDS = {
    "pg19_input_manifest_sha256",
    "prior_fp32_context_manifest_sha256",
    "review_response_plan_sha256",
}
RUNNER_RR2_STATIC_OPTIONS = {
    "--rr2-input-manifest",
    "--expected-rr2-input-manifest-sha256",
    "--prior-fp32-context-manifest",
    "--expected-prior-fp32-context-manifest-sha256",
    "--review-experiment-plan",
    "--expected-review-experiment-plan-sha256",
    "--frozen-query-banks",
    "--oracle-selection-plan",
}
RUNNER_FORMAL_SHARD_OPTIONS = {
    "--stage",
    "--output",
    "--rank",
    "--run-id",
    "--artifact-root",
    "--static-artifact",
    "--expected-static-sha256",
    "--rr2-input-manifest",
    "--expected-rr2-input-manifest-sha256",
    "--pg19-data",
    "--pg19-manifest",
    "--prior-capacity-manifest",
    "--model-dir",
    "--code-ledger",
    "--model-artifact-ledger",
    "--model-weight-ledger",
    "--protocol-manifest",
    "--run-id-receipt",
    "--expected-run-id-receipt-sha256",
    "--expected-gpu-uuid",
    "--gpu-assignment-receipt",
    "--expected-gpu-assignment-receipt-raw-sha256",
    "--model-load-authority",
    "--expected-model-load-authority-raw-sha256",
    "--private-model-view-manifest",
    "--expected-private-model-view-manifest-raw-sha256",
}
RUNNER_FORMAL_AGGREGATE_OPTIONS = {
    "--stage",
    "--output",
    "--run-id",
    "--artifact-root",
    "--static-artifact",
    "--expected-static-sha256",
    "--receipt-manifest",
    "--expected-receipt-manifest-sha256",
    "--run-id-receipt",
    "--expected-run-id-receipt-sha256",
    "--gpu-assignment-receipt",
    "--expected-gpu-assignment-receipt-raw-sha256",
    "--model-load-authority",
    "--expected-model-load-authority-raw-sha256",
    "--model-load-closure",
    "--expected-model-load-closure-raw-sha256",
    "--private-model-view-manifest",
    "--expected-private-model-view-manifest-raw-sha256",
    "--model-artifact-ledger",
    "--model-weight-ledger",
}
MODEL_MANIFEST_NAMES = (
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
)
RAW_SHARD_PATTERN = "shards/forkaudit-shard-{rank}.json"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RUN_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_LEDGER_RE = re.compile(r"^([0-9a-f]{64}) ([ *])(.+)$")
_PG19_SOURCE_RE = re.compile(r"^train/[0-9]+\.txt$")
_FORBIDDEN_DATA_MARKERS = (
    "longbench",
    "test-v2",
    "test_v2",
    "/validation/",
    "/test/",
)
_MODEL_WEIGHT_NAME_RE = re.compile(
    r"^model\.safetensors-[0-9]{5}-of-00014\.safetensors$"
)
_GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")
_FICLONE = 0x40049409
_WRITE_BITS = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH


class ManifestBuildError(RuntimeError):
    """A release manifest would not satisfy the frozen protocol."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestBuildError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    _require(path.is_file(), f"required input is missing: {path.name}")
    return _sha256_bytes(path.read_bytes())


def _artifact_receipt(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"required input is missing: {path.name}")
    payload = path.read_bytes()
    return {"sha256": _sha256_bytes(payload), "bytes": len(payload)}


def _read_json(path: Path, *, label: str) -> Any:
    _require(path.is_file(), f"{label} is missing")
    return runner.strict_json_loads(path.read_bytes(), label=label)


def _atomic_json(path: Path, value: Any) -> None:
    payload = runner.canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _validate_digest(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} is not a lowercase SHA-256",
    )
    return value


def build_run_id_receipt(
    *,
    static_artifact_sha256: str,
    protocol_manifest_sha256: str,
    nonce: bytes | None = None,
) -> dict[str, Any]:
    """Create one replayable 128-bit run identity bound to static protocol state."""

    static_digest = _validate_digest(
        static_artifact_sha256,
        "run-ID static artifact digest",
    )
    protocol_digest = _validate_digest(
        protocol_manifest_sha256,
        "run-ID protocol manifest digest",
    )
    entropy = secrets.token_bytes(32) if nonce is None else nonce
    _require(
        type(entropy) is bytes and len(entropy) == 32,
        "run-ID nonce must contain exactly 256 bits",
    )
    domain = b"qcomem-forkaudit-run-id-v1\0"
    run_id = hashlib.sha256(
        domain
        + bytes.fromhex(static_digest)
        + bytes.fromhex(protocol_digest)
        + entropy
    ).hexdigest()[:32]
    return {
        "schema_version": "qcomem-forkaudit-run-id-receipt-v1",
        "run_id": run_id,
        "run_id_bits": 128,
        "derivation": "sha256(domain || static_sha256 || protocol_sha256 || nonce)[:16]",
        "domain_hex": domain.hex(),
        "static_artifact_sha256": static_digest,
        "protocol_manifest_sha256": protocol_digest,
        "nonce_hex": entropy.hex(),
        "generated_once_after_static_before_candidate_outputs": True,
    }


def validate_run_id_receipt(
    value: Any,
    *,
    expected_sha256: str,
    run_id: str,
    static_artifact_sha256: str,
    protocol_manifest_sha256: str,
) -> dict[str, Any]:
    """Independently replay the one receipt shared by all shards/aggregate."""

    expected_digest = _validate_digest(
        expected_sha256, "expected run-ID receipt canonical digest"
    )
    static_digest = _validate_digest(
        static_artifact_sha256, "run-ID static artifact digest"
    )
    protocol_digest = _validate_digest(
        protocol_manifest_sha256, "run-ID protocol manifest digest"
    )
    _require(
        isinstance(run_id, str) and _RUN_ID_RE.fullmatch(run_id) is not None,
        "run ID must be exactly 32 lowercase hex characters",
    )
    required = {
        "schema_version",
        "run_id",
        "run_id_bits",
        "derivation",
        "domain_hex",
        "static_artifact_sha256",
        "protocol_manifest_sha256",
        "nonce_hex",
        "generated_once_after_static_before_candidate_outputs",
    }
    _require(isinstance(value, dict) and set(value) == required, "run-ID receipt schema drift")
    nonce = value["nonce_hex"]
    domain = b"qcomem-forkaudit-run-id-v1\0"
    _require(
        runner.sha256_json(value) == expected_digest
        and value["schema_version"] == "qcomem-forkaudit-run-id-receipt-v1"
        and value["run_id"] == run_id
        and type(value["run_id_bits"]) is int
        and value["run_id_bits"] == 128
        and value["derivation"]
        == "sha256(domain || static_sha256 || protocol_sha256 || nonce)[:16]"
        and value["domain_hex"] == domain.hex()
        and value["static_artifact_sha256"] == static_digest
        and value["protocol_manifest_sha256"] == protocol_digest
        and isinstance(nonce, str)
        and _SHA256_RE.fullmatch(nonce) is not None
        and value["generated_once_after_static_before_candidate_outputs"] is True,
        "run-ID receipt binding drift",
    )
    replayed = hashlib.sha256(
        domain
        + bytes.fromhex(static_digest)
        + bytes.fromhex(protocol_digest)
        + bytes.fromhex(nonce)
    ).hexdigest()[:32]
    _require(replayed == run_id, "run-ID receipt derivation drift")
    return dict(value)


def build_gpu_assignment_receipt(inventory_raw: bytes) -> dict[str, Any]:
    """Freeze an eight-rank H20 assignment without initializing CUDA."""

    _require(not torch.cuda.is_initialized(), "GPU assignment receipt initialized CUDA")
    try:
        inventory_text = inventory_raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestBuildError("GPU inventory is not UTF-8") from exc
    _require(inventory_text.endswith("\n"), "GPU inventory must end with a newline")
    parsed = list(csv.reader(inventory_text.splitlines()))
    _require(
        len(parsed) == runner.FORMAL_WORLD_SIZE,
        "GPU inventory must contain exactly eight rows",
    )
    hardware_rows: list[dict[str, Any]] = []
    for row_index, columns in enumerate(parsed):
        _require(len(columns) == 5, f"GPU inventory row {row_index} schema drift")
        raw_index, raw_uuid, raw_name, raw_memory, raw_capability = (
            item.strip() for item in columns
        )
        try:
            visible_index = int(raw_index)
            total_memory_mib = int(raw_memory)
            capability_parts = tuple(int(item) for item in raw_capability.split("."))
        except ValueError as exc:
            raise ManifestBuildError(
                f"GPU inventory row {row_index} contains a non-integral field"
            ) from exc
        _require(
            visible_index >= 0
            and total_memory_mib > 0
            and _GPU_UUID_RE.fullmatch(raw_uuid) is not None
            and "H20" in raw_name
            and capability_parts == (9, 0),
            f"GPU inventory row {row_index} is not one BF16-capable H20",
        )
        hardware_rows.append(
            {
                "visible_index": visible_index,
                "uuid": raw_uuid,
                "name": raw_name,
                "total_memory_mib": total_memory_mib,
                "compute_capability": [9, 0],
                "bf16_supported": True,
            }
        )
    hardware_rows.sort(key=lambda row: row["visible_index"])
    visible_indices = [row["visible_index"] for row in hardware_rows]
    uuids = [row["uuid"] for row in hardware_rows]
    _require(
        len(set(visible_indices)) == runner.FORMAL_WORLD_SIZE,
        "GPU visible indices are not unique",
    )
    _require(len(set(uuids)) == runner.FORMAL_WORLD_SIZE, "GPU UUIDs are not unique")
    rows = [
        {"rank": rank, **hardware}
        for rank, hardware in enumerate(hardware_rows)
    ]
    receipt = {
        "schema_version": "qcomem-forkaudit-gpu-assignment-receipt-v1",
        "world_size": runner.FORMAL_WORLD_SIZE,
        "inventory_query": "index,uuid,name,memory.total,compute_cap",
        "rows": rows,
        "unique_visible_indices": True,
        "unique_uuids": True,
        "all_h20": True,
        "all_compute_capability_9_0": True,
        "generated_before_candidate_outputs": True,
    }
    _require(not torch.cuda.is_initialized(), "GPU assignment receipt initialized CUDA")
    return receipt


def _copy_private_model_file(source: Path, destination: Path) -> tuple[str, os.stat_result, os.stat_result]:
    """Create a distinct-inode private file by reflink or byte copy."""

    nofollow = getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(source, os.O_RDONLY | nofollow)
    destination_fd = -1
    try:
        source_before = os.fstat(source_fd)
        _require(stat.S_ISREG(source_before.st_mode), f"model input is not regular: {source.name}")
        destination_fd = os.open(
            destination,
            os.O_RDWR | os.O_CREAT | os.O_EXCL | nofollow,
            0o400,
        )
        try:
            fcntl.ioctl(destination_fd, _FICLONE, source_fd)
            copy_mode = "ficlone"
        except OSError as exc:
            if exc.errno not in {
                errno.EXDEV,
                errno.EINVAL,
                errno.ENOTTY,
                errno.EOPNOTSUPP,
                errno.EPERM,
                errno.EACCES,
                getattr(errno, "ENOSYS", 38),
            }:
                raise
            os.ftruncate(destination_fd, 0)
            os.lseek(source_fd, 0, os.SEEK_SET)
            os.lseek(destination_fd, 0, os.SEEK_SET)
            while True:
                chunk = os.read(source_fd, 8 * 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    _require(written > 0, "private model byte copy stopped early")
                    view = view[written:]
            copy_mode = "byte-copy"
        os.fsync(destination_fd)
        os.fchmod(destination_fd, 0o444)
        source_after = os.fstat(source_fd)
        destination_stat = os.fstat(destination_fd)
    finally:
        if destination_fd >= 0:
            os.close(destination_fd)
        os.close(source_fd)
    source_fields = (
        source_before.st_dev,
        source_before.st_ino,
        source_before.st_size,
        source_before.st_mtime_ns,
        source_before.st_ctime_ns,
        source_before.st_mode,
    )
    source_after_fields = (
        source_after.st_dev,
        source_after.st_ino,
        source_after.st_size,
        source_after.st_mtime_ns,
        source_after.st_ctime_ns,
        source_after.st_mode,
    )
    _require(source_fields == source_after_fields, f"model input changed while copying: {source.name}")
    _require(
        stat.S_ISREG(destination_stat.st_mode)
        and destination_stat.st_size == source_before.st_size
        and (destination_stat.st_mode & _WRITE_BITS) == 0
        and (destination_stat.st_dev, destination_stat.st_ino)
        != (source_before.st_dev, source_before.st_ino),
        f"private model copy is not a distinct read-only regular inode: {source.name}",
    )
    return copy_mode, source_before, destination_stat


def _thaw_and_remove_private_staging(path: Path) -> None:
    if not path.exists():
        return
    for directory, directories, files in os.walk(path, topdown=False):
        for name in files:
            os.chmod(Path(directory) / name, 0o600)
        for name in directories:
            os.chmod(Path(directory) / name, 0o700)
        os.chmod(directory, 0o700)
    shutil.rmtree(path)


def materialize_private_model_view(
    *,
    source_model_dir: Path,
    private_model_view: Path,
    model_artifact_ledger: Path,
    expected_model_artifact_ledger_raw_sha256: str,
    model_weight_ledger: Path,
    expected_model_weight_ledger_raw_sha256: str,
    model_id: str,
    model_revision: str,
) -> dict[str, Any]:
    """Atomically materialize the exact ledger-bound loader closure."""

    _require(not torch.cuda.is_initialized(), "private model materialization initialized CUDA")
    _require(model_id == FORMAL_MODEL_ID, "private model view model ID drift")
    _require(model_revision == FORMAL_MODEL_REVISION, "private model view revision drift")
    source_lstat = source_model_dir.lstat()
    _require(
        stat.S_ISDIR(source_lstat.st_mode) and not stat.S_ISLNK(source_lstat.st_mode),
        "source model directory must be a real directory",
    )
    source_root = source_model_dir.resolve(strict=True)
    destination = private_model_view.absolute()
    staging = destination.with_name(destination.name + ".staging")
    _require(not destination.exists() and not staging.exists(), "private model view must be absent")

    artifact_raw = model_artifact_ledger.read_bytes()
    weight_raw = model_weight_ledger.read_bytes()
    _require(
        _sha256_bytes(artifact_raw)
        == _validate_digest(
            expected_model_artifact_ledger_raw_sha256,
            "expected model-artifact ledger raw digest",
        ),
        "model-artifact ledger raw SHA drift",
    )
    _require(
        _sha256_bytes(weight_raw)
        == _validate_digest(
            expected_model_weight_ledger_raw_sha256,
            "expected model-weight ledger raw digest",
        ),
        "model-weight ledger raw SHA drift",
    )
    artifact_rows = validate_path_independent_sha_ledger(
        model_artifact_ledger, label="private-view model artifact ledger"
    )
    weight_rows = validate_path_independent_sha_ledger(
        model_weight_ledger, label="private-view model weight ledger"
    )
    _require(len(weight_rows) == 14, "private model view requires exactly 14 weight shards")
    _require(
        all(_MODEL_WEIGHT_NAME_RE.fullmatch(row["logical_name"]) for row in weight_rows),
        "private model view weight-shard name drift",
    )
    combined: dict[str, dict[str, Any]] = {}
    for role, ledger_rows in (("model_artifact", artifact_rows), ("model_weight", weight_rows)):
        for ledger_row in ledger_rows:
            name = ledger_row["logical_name"]
            existing = combined.get(name)
            if existing is None:
                combined[name] = {
                    "declared_sha256": ledger_row["sha256"],
                    "ledger_roles": [role],
                }
            else:
                _require(
                    existing["declared_sha256"] == ledger_row["sha256"],
                    f"private-view ledgers disagree for {name}",
                )
                existing["ledger_roles"].append(role)

    manifest_rows: list[dict[str, Any]] = []
    try:
        staging.mkdir(parents=True, mode=0o700)
        for relative_path in sorted(combined, key=lambda name: name.encode("utf-8")):
            pure = PurePosixPath(relative_path)
            source = source_root.joinpath(*pure.parts)
            target = staging.joinpath(*pure.parts)
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            source_path_lstat = source.lstat()
            _require(
                stat.S_ISREG(source_path_lstat.st_mode)
                and not stat.S_ISLNK(source_path_lstat.st_mode),
                f"private-view source is not a regular non-symlink: {relative_path}",
            )
            _require(
                source.resolve(strict=True).is_relative_to(source_root),
                f"private-view source escaped model root: {relative_path}",
            )
            copy_mode, source_stat, view_stat = _copy_private_model_file(source, target)
            metadata = combined[relative_path]
            if "model_artifact" in metadata["ledger_roles"]:
                _require(
                    _sha256_file(target) == metadata["declared_sha256"],
                    f"private-view model artifact hash drift: {relative_path}",
                )
            manifest_rows.append(
                {
                    "relative_path": relative_path,
                    "ledger_roles": sorted(metadata["ledger_roles"]),
                    "declared_sha256": metadata["declared_sha256"],
                    "bytes": int(view_stat.st_size),
                    "copy_mode": copy_mode,
                    "source_device": int(source_stat.st_dev),
                    "source_inode": int(source_stat.st_ino),
                    "view_device": int(view_stat.st_dev),
                    "view_inode": int(view_stat.st_ino),
                    "source_and_view_inode_distinct": True,
                }
            )
        for directory, directories, _files in os.walk(staging, topdown=False):
            for name in directories:
                os.chmod(Path(directory) / name, 0o555)
            os.chmod(directory, 0o555)
        os.replace(staging, destination)
    except BaseException:
        _thaw_and_remove_private_staging(staging)
        raise

    result = {
        "schema_version": "qcomem-forkaudit-private-model-view-v1",
        "model_id": model_id,
        "model_revision": model_revision,
        "model_artifact_ledger_raw_sha256": _sha256_bytes(artifact_raw),
        "model_weight_ledger_raw_sha256": _sha256_bytes(weight_raw),
        "copy_policy": "ficlone-then-byte-copy;hardlink-and-symlink-forbidden",
        "file_count": len(manifest_rows),
        "weight_file_count": len(weight_rows),
        "all_source_and_view_inodes_distinct": True,
        "all_view_files_regular": True,
        "all_view_files_read_only": True,
        "rows": manifest_rows,
        "generated_before_candidate_outputs": True,
        "cuda_initialized": False,
    }
    _require(not torch.cuda.is_initialized(), "private model materialization initialized CUDA")
    return result


def validate_private_model_view_manifest(
    value: Any,
    *,
    model_view: Path,
    expected_model_artifact_ledger_raw_sha256: str,
    expected_model_weight_ledger_raw_sha256: str,
) -> dict[str, Any]:
    fields = {
        "schema_version",
        "model_id",
        "model_revision",
        "model_artifact_ledger_raw_sha256",
        "model_weight_ledger_raw_sha256",
        "copy_policy",
        "file_count",
        "weight_file_count",
        "all_source_and_view_inodes_distinct",
        "all_view_files_regular",
        "all_view_files_read_only",
        "rows",
        "generated_before_candidate_outputs",
        "cuda_initialized",
    }
    _require(isinstance(value, dict) and set(value) == fields, "private model-view manifest schema drift")
    _require(
        value["schema_version"] == "qcomem-forkaudit-private-model-view-v1"
        and value["model_id"] == FORMAL_MODEL_ID
        and value["model_revision"] == FORMAL_MODEL_REVISION
        and value["model_artifact_ledger_raw_sha256"]
        == expected_model_artifact_ledger_raw_sha256
        and value["model_weight_ledger_raw_sha256"]
        == expected_model_weight_ledger_raw_sha256
        and value["copy_policy"]
        == "ficlone-then-byte-copy;hardlink-and-symlink-forbidden"
        and type(value["file_count"]) is int
        and type(value["weight_file_count"]) is int
        and value["weight_file_count"] == 14
        and value["all_source_and_view_inodes_distinct"] is True
        and value["all_view_files_regular"] is True
        and value["all_view_files_read_only"] is True
        and value["generated_before_candidate_outputs"] is True
        and value["cuda_initialized"] is False,
        "private model-view manifest binding drift",
    )
    rows = value["rows"]
    row_fields = {
        "relative_path",
        "ledger_roles",
        "declared_sha256",
        "bytes",
        "copy_mode",
        "source_device",
        "source_inode",
        "view_device",
        "view_inode",
        "source_and_view_inode_distinct",
    }
    _require(
        isinstance(rows, list) and len(rows) == value["file_count"],
        "private model-view row count drift",
    )
    names: list[str] = []
    weight_count = 0
    for row in rows:
        _require(isinstance(row, dict) and set(row) == row_fields, "private model-view row schema drift")
        name = _normalize_ledger_name(row["relative_path"], label="private model-view row")
        roles = row["ledger_roles"]
        _require(
            isinstance(roles, list)
            and roles == sorted(set(roles))
            and set(roles) <= {"model_artifact", "model_weight"}
            and bool(roles),
            "private model-view ledger roles drift",
        )
        _validate_digest(row["declared_sha256"], f"private model-view {name}")
        for field in (
            "bytes",
            "source_device",
            "source_inode",
            "view_device",
            "view_inode",
        ):
            _require(type(row[field]) is int and row[field] > 0, f"private model-view {field} drift")
        _require(
            row["copy_mode"] in {"ficlone", "byte-copy"}
            and row["source_and_view_inode_distinct"] is True
            and (row["source_device"], row["source_inode"])
            != (row["view_device"], row["view_inode"]),
            "private model-view copy provenance drift",
        )
        path = model_view.joinpath(*PurePosixPath(name).parts)
        observed = path.lstat()
        _require(
            stat.S_ISREG(observed.st_mode)
            and not stat.S_ISLNK(observed.st_mode)
            and int(observed.st_size) == row["bytes"]
            and int(observed.st_dev) == row["view_device"]
            and int(observed.st_ino) == row["view_inode"]
            and (observed.st_mode & _WRITE_BITS) == 0,
            f"private model-view live file drift: {name}",
        )
        if "model_weight" in roles:
            weight_count += 1
        names.append(name)
    _require(
        names == sorted(names, key=lambda item: item.encode("utf-8"))
        and len(set(names)) == len(names)
        and weight_count == 14,
        "private model-view row order/weight coverage drift",
    )
    return dict(value)


def run_model_load_lease_keeper(
    *,
    model_view: Path,
    model_weight_ledger: Path,
    expected_model_weight_ledger_raw_sha256: str,
    expected_model_artifact_ledger_raw_sha256: str,
    model_view_manifest: Path,
    expected_model_view_manifest_raw_sha256: str,
    run_id: str,
    authority_output: Path,
    closure_output: Path,
    control_input: TextIO,
    event_output: TextIO,
    lease_factory: Any | None = None,
) -> dict[str, str]:
    """Hold Linux ModelLoadLease-v1 across all rank model loads."""

    import qcomem_forkaudit_model_load_lease as model_lease

    weight_raw = model_weight_ledger.read_bytes()
    _require(
        _sha256_bytes(weight_raw) == expected_model_weight_ledger_raw_sha256,
        "model-load keeper weight ledger raw SHA drift",
    )
    artifact_digest = _validate_digest(
        expected_model_artifact_ledger_raw_sha256,
        "model-load keeper artifact ledger raw digest",
    )
    manifest_raw = model_view_manifest.read_bytes()
    _require(
        _sha256_bytes(manifest_raw) == expected_model_view_manifest_raw_sha256,
        "model-load keeper private-view manifest raw SHA drift",
    )
    manifest_value = runner.strict_json_loads(
        manifest_raw, label="private model-view manifest"
    )
    _require(
        runner.canonical_json_bytes(manifest_value) + b"\n" == manifest_raw,
        "private model-view manifest is not canonical JSON plus LF",
    )
    validate_private_model_view_manifest(
        manifest_value,
        model_view=model_view,
        expected_model_artifact_ledger_raw_sha256=artifact_digest,
        expected_model_weight_ledger_raw_sha256=(
            expected_model_weight_ledger_raw_sha256
        ),
    )
    ledger_rows = validate_path_independent_sha_ledger(
        model_weight_ledger, label="model-load keeper weight ledger"
    )
    factory = model_lease.ModelLoadLeaseSet if lease_factory is None else lease_factory
    lease_set = factory(
        model_view=model_view,
        ledger_rows=ledger_rows,
        run_id=run_id,
        weight_ledger_raw_sha256=expected_model_weight_ledger_raw_sha256,
        model_artifact_ledger_raw_sha256=artifact_digest,
        model_view_manifest_sha256=expected_model_view_manifest_raw_sha256,
    )
    authority_built = False
    closed = False
    previous_term = signal.getsignal(signal.SIGTERM)

    if lease_factory is None:
        _require(
            hasattr(signal, "pthread_sigmask"),
            "model-load keeper requires pthread signal masks",
        )
        inherited_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
        _require(
            signal.SIGIO in inherited_mask,
            "model-load keeper must inherit SIGIO blocked before Python imports",
        )
        _require(
            threading.active_count() == 1,
            "model-load keeper must remain a single Python thread",
        )
        # The bootstrap blocks SIGIO before this interpreter imports torch or
        # any other module that may create native threads.  Only the main
        # keeper thread is unblocked; all inherited worker masks remain closed.
        signal.pthread_sigmask(signal.SIG_UNBLOCK, {signal.SIGIO})

    def abort_keeper(_signum: int, _frame: Any) -> None:
        if authority_built and not closed:
            lease_set.mark_breach_for_test()
        raise ManifestBuildError("model-load lease keeper terminated")

    signal.signal(signal.SIGTERM, abort_keeper)
    try:
        authority = lease_set.acquire_and_hash()
        authority_built = True
        manifest_weights = {
            row["relative_path"]: row
            for row in manifest_value["rows"]
            if "model_weight" in row["ledger_roles"]
        }
        _require(
            set(manifest_weights)
            == {row["logical_name"] for row in authority["rows"]},
            "model-load authority/private-view weight coverage drift",
        )
        for authority_row in authority["rows"]:
            view_row = manifest_weights[authority_row["logical_name"]]
            authority_stat = authority_row["stat"]
            _require(
                authority_row["declared_sha256"] == view_row["declared_sha256"]
                and authority_row["observed_sha256"]
                == view_row["declared_sha256"]
                and authority_stat["bytes"] == view_row["bytes"]
                and authority_stat["st_dev"] == view_row["view_device"]
                and authority_stat["st_ino"] == view_row["view_inode"]
                and view_row["copy_mode"] in {"ficlone", "byte-copy"}
                and view_row["source_and_view_inode_distinct"] is True
                and (view_row["source_device"], view_row["source_inode"])
                != (view_row["view_device"], view_row["view_inode"]),
                "model-load authority does not replay private-view provenance",
            )
        authority_raw = model_lease.canonical_receipt_bytes(authority)
        _atomic_bytes(authority_output, authority_raw)
        authority_raw_sha256 = _sha256_bytes(authority_raw)
        event_output.write(f"READY {authority_raw_sha256}\n")
        event_output.flush()
        command = control_input.readline()
        _require(
            command == f"CLOSE {authority_raw_sha256}\n",
            "model-load lease keeper received an invalid close command",
        )
        closure = lease_set.close_and_receipt()
        closure_raw = model_lease.canonical_receipt_bytes(closure)
        _atomic_bytes(closure_output, closure_raw)
        closed = True
        model_lease.validate_closure(
            closure, authority=authority, require_passed=True
        )
        closure_raw_sha256 = _sha256_bytes(closure_raw)
        event_output.write(f"CLOSED {closure_raw_sha256}\n")
        event_output.flush()
        return {
            "authority_raw_sha256": authority_raw_sha256,
            "closure_raw_sha256": closure_raw_sha256,
        }
    except BaseException as exc:
        if authority_built and not closed:
            # Best-effort evidence preservation must never mask the primary
            # guard failure.  The helper itself guarantees FD/lease cleanup
            # when terminal I/O raises, so a second close may legitimately be
            # rejected as already closed.
            try:
                lease_set.mark_breach_for_test()
                failed_closure = lease_set.close_and_receipt()
                closed = True
                _atomic_bytes(
                    closure_output,
                    model_lease.canonical_receipt_bytes(failed_closure),
                )
            except BaseException:
                pass
        if isinstance(exc, model_lease.ModelLoadLeaseError):
            raise ManifestBuildError(
                f"model-load lease keeper rejected: {exc}"
            ) from exc
        raise
    finally:
        signal.signal(signal.SIGTERM, previous_term)


def _normalize_ledger_name(raw_name: str, *, label: str) -> str:
    _require("\\" not in raw_name and "\x00" not in raw_name, f"{label} contains a non-POSIX path")
    while raw_name.startswith("./"):
        raw_name = raw_name[2:]
    pure = PurePosixPath(raw_name)
    _require(raw_name != "" and not pure.is_absolute(), f"{label} contains an absolute path")
    _require(".." not in pure.parts and "." not in pure.parts, f"{label} contains path traversal")
    normalized = pure.as_posix()
    _require(normalized == raw_name, f"{label} path is not normalized")
    return normalized


def validate_path_independent_sha_ledger(path: Path, *, label: str) -> list[dict[str, str]]:
    """Validate a C-sorted sha256sum ledger that contains only relative names."""

    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ManifestBuildError(f"{label} is not readable UTF-8") from exc
    _require(text.endswith("\n"), f"{label} must end with one newline")
    rows: list[dict[str, str]] = []
    names: list[str] = []
    for index, line in enumerate(text.splitlines(), start=1):
        match = _LEDGER_RE.fullmatch(line)
        _require(match is not None, f"{label} line {index} is not sha256sum format")
        assert match is not None
        digest, marker, raw_name = match.groups()
        _validate_digest(digest, f"{label} line {index}")
        _require(marker == " ", f"{label} must not contain binary-mode ledger markers")
        name = _normalize_ledger_name(raw_name, label=label)
        names.append(name)
        rows.append({"logical_name": name, "sha256": digest})
    _require(bool(rows), f"{label} is empty")
    _require(len(set(names)) == len(names), f"{label} repeats a logical name")
    _require(names == sorted(names, key=lambda item: item.encode("utf-8")), f"{label} is not LC_ALL=C sorted")
    return rows


def _model_manifest(model_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    digest = hashlib.sha256()
    rows: list[dict[str, Any]] = []
    for name in MODEL_MANIFEST_NAMES:
        path = model_dir / name
        _require(path.is_file(), f"model manifest component is missing: {name}")
        file_digest = _sha256_file(path)
        size = path.stat().st_size
        digest.update(f"{name}\0{file_digest}\0{size}\n".encode("utf-8"))
        rows.append({"logical_name": name, "sha256": file_digest, "bytes": size})
    return digest.hexdigest(), rows


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


def _audit_pg19_manifest(manifest: Any, *, pg19_data_sha256: str) -> set[str]:
    _require(isinstance(manifest, dict), "PG19 manifest must be an object")
    _require(manifest.get("bucket") == "deepmind-gutenberg", "PG19 bucket drift")
    _require(manifest.get("prefix") == "train/", "PG19 manifest is not train-only")
    _require(manifest.get("test_or_validation_objects_used") is False, "PG19 test/validation exclusion is not explicit")
    _require(manifest.get("jsonl_sha256") == pg19_data_sha256, "PG19 manifest/data SHA binding drift")
    objects = manifest.get("objects")
    _require(isinstance(objects, list) and len(objects) >= runner.FORMAL_BOOKS, "PG19 manifest has fewer than eight train books")
    names: list[str] = []
    for row in objects:
        _require(isinstance(row, dict), "PG19 object row must be an object")
        name = row.get("name")
        _require(isinstance(name, str) and _PG19_SOURCE_RE.fullmatch(name) is not None, "non-train PG19 source object rejected")
        names.append(name)
    _require(len(set(names)) == len(names), "PG19 manifest repeats an object")
    lowered = "\n".join(_walk_strings(manifest)).lower()
    for marker in _FORBIDDEN_DATA_MARKERS:
        _require(marker not in lowered, f"forbidden dataset marker in PG19 manifest: {marker}")
    return set(names)


def _assert_no_serialized_host_paths(value: Any) -> None:
    for item in _walk_strings(value):
        _require(not item.startswith("/"), "release manifest leaked an absolute host path")
        _require("file://" not in item.lower(), "release manifest leaked a file URI")
        _require("/users/" not in item.lower(), "release manifest leaked a user path")
        _require("/mnt/" not in item.lower(), "release manifest leaked a mount path")


def _require_receipt_sha(path: Path, expected: str, *, label: str) -> dict[str, Any]:
    _validate_digest(expected, f"expected {label} digest")
    receipt = _artifact_receipt(path)
    _require(receipt["sha256"] == expected, f"{label} raw-byte SHA-256 mismatch")
    return receipt


def _runner_rr2_compatibility_gate() -> dict[str, Any]:
    frozen = set(getattr(runner, "FROZEN_SHA256_FIELDS", ()))
    _require(
        RUNNER_RR2_FROZEN_FIELDS <= frozen,
        "runner RR2 frozen-identity contract is not final; release remains blocked",
    )
    parser_factory = getattr(runner, "_parser", None)
    _require(callable(parser_factory), "runner parser compatibility gate is unavailable")
    options = {
        option
        for action in parser_factory()._actions
        for option in getattr(action, "option_strings", ())
    }
    _require(
        RUNNER_RR2_STATIC_OPTIONS <= options,
        "runner RR2 static CLI contract is not final; release remains blocked",
    )
    _require(
        RUNNER_FORMAL_SHARD_OPTIONS <= options,
        "runner formal shard CLI contract is not final; release remains blocked",
    )
    _require(
        RUNNER_FORMAL_AGGREGATE_OPTIONS <= options,
        "runner formal aggregate CLI contract is not final; release remains blocked",
    )
    main_source = inspect.getsource(runner.main)
    aggregate_marker = "# A production aggregate"
    _require(
        aggregate_marker in main_source,
        "runner aggregate compatibility marker is unavailable; release remains blocked",
    )
    aggregate_source = main_source[main_source.index(aggregate_marker) :]
    aggregate_api = getattr(runner, "aggregate_shards", None)
    _require(
        callable(aggregate_api),
        "runner aggregate API is unavailable; release remains blocked",
    )
    aggregate_api_source = inspect.getsource(aggregate_api)
    _require(
        "run_id_receipt_raw=args.run_id_receipt.read_bytes()" in aggregate_source
        and "expected_run_id_receipt_sha256=" in aggregate_source
        and "_validate_run_id_receipt(" in aggregate_api_source,
        "runner aggregate does not replay the shared run-ID receipt; release remains blocked",
    )
    aggregate_forwarded_inputs = {
        "gpu_assignment_receipt_raw=args.gpu_assignment_receipt.read_bytes()",
        "private_model_view_manifest_raw=(",
        "model_load_authority_raw=args.model_load_authority.read_bytes()",
        "model_load_closure_raw=args.model_load_closure.read_bytes()",
        "model_weight_ledger_raw=args.model_weight_ledger.read_bytes()",
        "model_artifact_ledger_raw=args.model_artifact_ledger.read_bytes()",
    }
    _require(
        all(item in aggregate_source for item in aggregate_forwarded_inputs)
        and "_validate_gpu_assignment_receipt(" in aggregate_api_source
        and "_validate_external_model_load_evidence(" in aggregate_api_source,
        "runner aggregate does not consume the external GPU/model authorities; release remains blocked",
    )
    gpu_audit = getattr(runner, "_audit_formal_local_gpu", None)
    _require(
        callable(gpu_audit),
        "runner local-GPU audit is unavailable; release remains blocked",
    )
    gpu_audit_source = inspect.getsource(gpu_audit)
    _require(
        "visible == expected_uuid" in gpu_audit_source
        and 'f"--id={visible}"' in gpu_audit_source
        and "int(visible)" not in gpu_audit_source,
        "runner does not bind CUDA_VISIBLE_DEVICES by stable GPU UUID; release remains blocked",
    )
    static_parameters = set(inspect.signature(runner.make_static_artifact).parameters)
    required_static_parameters = {
        "rr2_input_manifest_raw",
        "prior_fp32_context_manifest_raw",
        "review_response_plan_raw",
    }
    _require(
        required_static_parameters <= static_parameters,
        "runner static raw-provenance API contract is not final; release remains blocked",
    )
    _require(
        runner.FORMAL_RR2_WINDOWS_SHA256 == rr2_input.FORMAL_RR2_WINDOWS_SHA256
        and runner.PRIOR_CAPACITY_MANIFEST_SHA256
        == rr2_input.PRIOR_CAPACITY_MANIFEST_SHA256
        and runner.PRIOR_CAPACITY_WINDOWS_SHA256
        == rr2_input.PRIOR_CAPACITY_WINDOWS_SHA256
        and runner.PRIOR_FP32_CONTEXT_RAW_SHA256
        == PRIOR_FP32_CONTEXT_MANIFEST_SHA256,
        "runner/release frozen RR2 constants disagree",
    )
    return {
        "required_frozen_identity_fields": sorted(RUNNER_RR2_FROZEN_FIELDS),
        "required_static_cli_options": sorted(RUNNER_RR2_STATIC_OPTIONS),
        "required_formal_shard_cli_options": sorted(RUNNER_FORMAL_SHARD_OPTIONS),
        "required_formal_aggregate_cli_options": sorted(
            RUNNER_FORMAL_AGGREGATE_OPTIONS
        ),
        "aggregate_shared_run_id_receipt_replayed": True,
        "aggregate_external_gpu_and_model_authorities_replayed": True,
        "cuda_visible_devices_uses_receipt_uuid": True,
        "required_static_raw_provenance_parameters": sorted(
            required_static_parameters
        ),
        "runner_contract_matched": True,
    }


def _audit_model_tokenizer_cross_binding(
    model_dir: Path,
    model_artifact_rows: Sequence[Mapping[str, str]],
    rr2_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    model = rr2_manifest.get("model")
    _require(isinstance(model, dict), "RR2 input manifest model binding missing")
    embedded = model.get("model_and_tokenizer_artifacts")
    _require(isinstance(embedded, dict), "RR2 tokenizer artifact binding missing")
    independently_observed = rr2_input.audit_model_tokenizer_artifacts(model_dir)
    _require(
        embedded == independently_observed,
        "RR2 tokenizer artifact set differs from the exact model directory",
    )
    ledger = {row["logical_name"]: row["sha256"] for row in model_artifact_rows}
    artifact_rows = independently_observed["artifacts"]
    for row in artifact_rows:
        name = row["logical_name"]
        _require(name in ledger, f"model artifact ledger omits RR2 tokenizer input {name}")
        _require(ledger[name] == row["sha256"], f"model artifact ledger disagrees on {name}")
        path = model_dir / name
        _require(path.is_file() and path.stat().st_size == row["bytes"], f"model artifact size drift for {name}")
    return {
        "selected_tokenizer_layout": independently_observed["selected_tokenizer_layout"],
        "artifact_set_sha256": independently_observed["artifact_set_sha256"],
        "cross_bound_ledger_entries": len(artifact_rows),
        "every_rr2_tokenizer_artifact_present_in_verified_model_ledger": True,
    }


def _rebuild_rr2_inputs_from_source(
    args: argparse.Namespace,
    *,
    expectations: rr2_input.InputExpectations,
    tokenizer: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Re-tokenize PG19 locally and require exact canonical input bytes.

    Structural/self-hash validation alone cannot establish that document and
    query digests came from the frozen train64 bytes.  This replay is therefore
    mandatory in the production preregistration path and happens before the
    launcher can reach either release gate.
    """

    # ``tokenizer`` exists only for deterministic pure-function tests.  Every
    # production caller omits it and must load the exact local artifact set.
    if tokenizer is None:
        try:
            tokenizer = rr2_input.load_local_tokenizer(args.model_dir)
        except Exception as exc:
            raise ManifestBuildError(
                "exact local RR2 tokenizer could not be loaded; network fallback is forbidden"
            ) from exc
    rebuilt = rr2_input.build_from_paths(
        pg19_data=args.pg19_data,
        pg19_manifest=args.pg19_manifest,
        prior_capacity_manifest=args.prior_capacity_manifest,
        model_dir=args.model_dir,
        tokenizer=tokenizer,
        expectations=expectations,
    )
    rebuilt_main = rr2_input.canonical_json_bytes(rebuilt) + b"\n"
    supplied_main = args.pg19_input_manifest.read_bytes()
    _require(
        rebuilt_main == supplied_main,
        "RR2 main manifest does not exactly replay from PG19 bytes and tokenizer",
    )
    rebuilt_banks = rr2_input.canonical_json_bytes(rebuilt["frozen_query_banks"]) + b"\n"
    rebuilt_oracle = rr2_input.canonical_json_bytes(rebuilt["oracle_selection_plan"]) + b"\n"
    _require(
        rebuilt_banks == args.frozen_query_banks_input.read_bytes(),
        "RR2 query-bank sidecar does not exactly replay from PG19 bytes and tokenizer",
    )
    _require(
        rebuilt_oracle == args.oracle_selection_input.read_bytes(),
        "RR2 oracle sidecar does not exactly replay from PG19 bytes and tokenizer",
    )
    _require(not torch.cuda.is_initialized(), "RR2 source replay initialized CUDA")
    return rebuilt, {
        "source_replay_mode": "exact-pg19-bytes-plus-local-frozen-tokenizer",
        "main_raw_sha256": _sha256_bytes(rebuilt_main),
        "main_raw_bytes": len(rebuilt_main),
        "query_banks_raw_sha256": _sha256_bytes(rebuilt_banks),
        "query_banks_raw_bytes": len(rebuilt_banks),
        "oracle_selection_raw_sha256": _sha256_bytes(rebuilt_oracle),
        "oracle_selection_raw_bytes": len(rebuilt_oracle),
        "document_and_all_query_token_digests_recomputed": True,
        "all_three_supplied_files_byte_exact": True,
        "network_access_allowed": False,
        "cuda_initialized": False,
    }


def _audit_prior_fp32_context(
    value: Any,
    *,
    rr2_selection_plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(isinstance(value, dict), "prior FP32 context manifest must be an object")
    _require(
        value.get("schema_version") == fp32_context.SCHEMA_VERSION,
        "prior FP32 context schema drift",
    )
    _require(isinstance(value.get("purpose"), str) and value["purpose"], "prior FP32 context purpose missing")
    definition = value.get("diagnostic_definition")
    _require(isinstance(definition, dict), "prior FP32 diagnostic definition missing")
    _require(
        set(definition)
        == {
            "comparison_path",
            "candidate",
            "reference",
            "backend_compatibility_was_nonblocking_in_prior_run",
            "fresh_and_reuse_metric_payloads_required_exactly_equal",
            "diagnostic_count",
            "role_in_rr2_threshold_choice",
            "selected_or_tuned_rr2_threshold",
        },
        "prior FP32 diagnostic-definition schema drift",
    )
    _require(definition.get("comparison_path") == fp32_context.COMPARISON_PATH, "prior FP32 comparison path drift")
    _require(
        definition.get("candidate") == "vLLM unified attention over Q16 BF16 K/V",
        "prior FP32 candidate drift",
    )
    _require(definition.get("reference") == "dense FP32 attention", "prior FP32 reference drift")
    _require(
        definition.get("backend_compatibility_was_nonblocking_in_prior_run") is True,
        "prior FP32 compatibility role drift",
    )
    _require(
        definition.get("fresh_and_reuse_metric_payloads_required_exactly_equal") is True,
        "prior FP32 fresh/reuse equality contract drift",
    )
    _require(definition.get("diagnostic_count") == 80, "prior FP32 context must bind 80 diagnostics")
    _require(definition.get("role_in_rr2_threshold_choice") == "contextual_validation_only", "prior FP32 context role drift")
    _require(definition.get("selected_or_tuned_rr2_threshold") is False, "prior rows must not tune RR2 threshold")
    rows = value.get("diagnostics")
    _require(isinstance(rows, list) and len(rows) == 80, "prior FP32 diagnostic rows drift")
    row_fields = {
        "rank",
        "window_index",
        "source_object",
        "document_tokens",
        "query_tokens",
        "layer_idx",
        "query_sha256",
        "position_ids_sha256",
        "mask_sha256",
        "scaling",
        "comparison",
        "metrics",
    }
    metric_fields = {"bitwise_exact", "finite", "max_abs", "mean_abs", "relative_l2"}
    coordinate_fields = (
        "source_object",
        "window_index",
        "document_tokens",
        "query_tokens",
        "query_sha256",
        "layer_idx",
        "scaling",
    )
    coordinate_projection = []
    query_digests = []
    observed_relative_l2 = []
    sources_by_rank: dict[int, str] = {}
    for row_index, row in enumerate(rows):
        _require(isinstance(row, dict) and set(row) == row_fields, "prior FP32 diagnostic row schema drift")
        expected_rank = row_index // len(fp32_context.EXPECTED_LAYERS)
        expected_layer = fp32_context.EXPECTED_LAYERS[
            row_index % len(fp32_context.EXPECTED_LAYERS)
        ]
        _require(type(row["rank"]) is int and row["rank"] == expected_rank, "prior FP32 rank/order drift")
        _require(type(row["window_index"]) is int and row["window_index"] == expected_rank, "prior FP32 window/rank drift")
        source = row["source_object"]
        _require(isinstance(source, str) and _PG19_SOURCE_RE.fullmatch(source) is not None, "prior FP32 source is not PG19 train")
        _require(
            source == PRIOR_FP32_SOURCES_BY_RANK[expected_rank],
            "prior FP32 frozen source/rank drift",
        )
        if expected_rank in sources_by_rank:
            _require(sources_by_rank[expected_rank] == source, "prior FP32 rank uses multiple sources")
        else:
            sources_by_rank[expected_rank] = source
        _require(type(row["document_tokens"]) is int and row["document_tokens"] == fp32_context.PRIOR_CONTEXT_DOCUMENT_TOKENS == 1025, "prior FP32 document length drift")
        _require(type(row["query_tokens"]) is int and row["query_tokens"] == fp32_context.PRIOR_CONTEXT_QUERY_TOKENS == 32, "prior FP32 query length drift")
        _require(type(row["layer_idx"]) is int and row["layer_idx"] == expected_layer, "prior FP32 layer/order drift")
        for field in ("query_sha256", "position_ids_sha256", "mask_sha256"):
            _validate_digest(row[field], f"prior FP32 {field}")
        _require(
            row["position_ids_sha256"] == PRIOR_FP32_POSITION_IDS_SHA256,
            "prior FP32 position-ID SHA drift",
        )
        _require(
            row["mask_sha256"] == PRIOR_FP32_MASK_SHA256,
            "prior FP32 mask SHA drift",
        )
        query_digests.append(row["query_sha256"])
        _require(type(row["scaling"]) is float and row["scaling"] == 0.0625, "prior FP32 scale drift")
        _require(row["comparison"] == "vllm_reuse_vs_fp32_dense", "prior FP32 comparison drift")
        metric = row.get("metrics")
        _require(isinstance(metric, dict) and set(metric) == metric_fields, "prior FP32 metric schema drift")
        _require(type(metric["bitwise_exact"]) is bool, "prior FP32 bitwise flag type drift")
        _require(metric["finite"] is True, "prior FP32 diagnostic is non-finite")
        for field in ("max_abs", "mean_abs", "relative_l2"):
            number = metric[field]
            _require(type(number) is float and math.isfinite(number) and number >= 0.0, f"prior FP32 {field} drift")
        relative_l2 = metric.get("relative_l2")
        observed_relative_l2.append(relative_l2)
        coordinate_projection.append(
            {field: row[field] for field in coordinate_fields}
        )
    coordinate_tuples = [tuple(row[field] for field in coordinate_fields) for row in rows]
    _require(len(set(coordinate_tuples)) == 80, "prior FP32 diagnostic coordinate duplicated")
    _require(len(set(query_digests)) == 80, "prior FP32 query digest duplicated")
    _require(len(set(sources_by_rank.values())) == 8, "prior FP32 sources are not eight unique books")
    margin = value.get("pre_fixed_threshold_margin_check")
    _require(isinstance(margin, dict), "prior FP32 threshold-margin check missing")
    _require(
        set(margin)
        == {
            "threshold_fixed_before_rr2",
            "threshold_fixed_independently_of_prior_rows",
            "prior_rows_selected_or_tuned_threshold",
            "prior_archive_role",
            "fixed_preregistered_threshold",
            "required_context_margin_multiplier",
            "maximum_observed_prior_relative_l2",
            "required_margin_boundary_from_prior_maximum",
            "fixed_threshold_to_prior_maximum_ratio",
            "fixed_threshold_at_least_twice_prior_maximum",
            "maximum_coordinate",
        },
        "prior FP32 threshold-margin schema drift",
    )
    maximum_row = max(
        rows,
        key=lambda row: (
            row["metrics"]["relative_l2"],
            row["rank"],
            row["layer_idx"],
        ),
    )
    maximum = maximum_row["metrics"]["relative_l2"]
    required_boundary = 2.0 * maximum
    expected_maximum_coordinate = {
        "rank": maximum_row["rank"],
        "window_index": maximum_row["window_index"],
        "source_object": maximum_row["source_object"],
        "document_tokens": maximum_row["document_tokens"],
        "query_sha256": maximum_row["query_sha256"],
        "layer_idx": maximum_row["layer_idx"],
        "scaling": maximum_row["scaling"],
    }
    _require(margin.get("threshold_fixed_before_rr2") is True, "RR2 threshold was not fixed before execution")
    _require(margin.get("threshold_fixed_independently_of_prior_rows") is True, "prior context selected the RR2 threshold")
    _require(margin.get("prior_rows_selected_or_tuned_threshold") is False, "prior context tuned the RR2 threshold")
    _require(margin.get("prior_archive_role") == "contextual_validation_only", "prior FP32 archive role drift")
    _require(margin.get("fixed_preregistered_threshold") == runner.ORACLE_MAX_RELATIVE_L2 == 0.005, "RR2 oracle threshold drift")
    _require(margin.get("maximum_observed_prior_relative_l2") == maximum, "prior FP32 maximum was not recomputed")
    _require(margin.get("required_context_margin_multiplier") == 2.0, "prior FP32 margin multiplier drift")
    _require(margin.get("required_margin_boundary_from_prior_maximum") == required_boundary, "prior FP32 two-times boundary drift")
    _require(margin.get("fixed_threshold_to_prior_maximum_ratio") == 0.005 / maximum, "prior FP32 threshold/max ratio drift")
    _require(margin.get("maximum_coordinate") == expected_maximum_coordinate, "prior FP32 maximum coordinate drift")
    _require(0.005 >= required_boundary, "pre-fixed RR2 threshold lacks two-times prior margin")
    _require(margin.get("fixed_threshold_at_least_twice_prior_maximum") is True, "serialized prior FP32 margin flag drift")
    disjoint = value.get("rr2_disjointness_from_prior_context")
    _require(isinstance(disjoint, dict), "prior FP32/RR2 disjointness proof missing")
    _require(
        set(disjoint)
        == {
            "ordered_fields",
            "coordinate_equality",
            "prior_coordinate_count",
            "prior_coordinates_sha256",
            "prior_context_document_token_values",
            "rr2_preregistered_document_tokens",
            "document_length_is_required_coordinate_component",
            "document_length_disjoint",
            "rr2_coordinate_disjointness_rule",
            "prior_document_start_token_available",
            "prior_document_start_token_note",
        },
        "prior FP32 disjointness schema drift",
    )
    _require(disjoint.get("ordered_fields") == list(coordinate_fields), "prior FP32 coordinate field order drift")
    _require(disjoint.get("coordinate_equality") == "exact equality of every ordered field", "prior FP32 coordinate equality drift")
    recomputed_coordinate_sha = _sha256_bytes(
        json.dumps(
            coordinate_projection,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    _require(disjoint.get("prior_coordinate_count") == 80, "prior FP32 coordinate count drift")
    _require(disjoint.get("prior_coordinates_sha256") == recomputed_coordinate_sha, "prior FP32 coordinate SHA was not recomputed")
    prior_document_lengths = sorted({row["document_tokens"] for row in rows})
    _require(disjoint.get("prior_context_document_token_values") == prior_document_lengths == [1025], "prior FP32 document-length set drift")
    _require(disjoint.get("rr2_preregistered_document_tokens") == rr2_input.FORMAL_DOCUMENT_TOKENS, "prior context RR2 length drift")
    _require(disjoint.get("document_length_is_required_coordinate_component") is True, "prior FP32 coordinate omits document length")
    _require(disjoint.get("prior_document_start_token_available") is False, "prior FP32 unexpectedly claims start-token provenance")
    _require(isinstance(disjoint.get("prior_document_start_token_note"), str) and disjoint["prior_document_start_token_note"], "prior FP32 start-token limitation missing")
    _require(all(selection.get("document_length") == 4095 for selection in rr2_selection_plan), "RR2 selection plan document length drift")
    _require(all(selection.get("document_length") not in prior_document_lengths for selection in rr2_selection_plan), "RR2 selection overlaps prior FP32 document-length coordinate")
    _require(disjoint.get("document_length_disjoint") is True, "serialized RR2/prior FP32 disjointness flag drift")
    archive = value.get("archive")
    _require(isinstance(archive, dict), "prior FP32 archive receipt missing")
    _require(archive.get("archive_id") == fp32_context.ARCHIVE_ID, "prior FP32 archive ID drift")
    _require(archive.get("fair_protocol") == fp32_context.FAIR_PROTOCOL, "prior FP32 protocol drift")
    _require(archive.get("quantization") == "Q16" and archive.get("single_request_only") is True, "prior FP32 archive configuration drift")
    _validate_digest(archive.get("pg19_data_sha256"), "prior FP32 PG19 data digest")
    _validate_digest(archive.get("pg19_manifest_sha256"), "prior FP32 PG19 manifest digest")
    _validate_digest(archive.get("windows_sha256"), "prior FP32 window digest")
    raw = archive.get("raw_shards")
    _require(isinstance(raw, list) and len(raw) == 8, "prior FP32 archive must bind eight raw shards")
    for rank, receipt in enumerate(raw):
        _require(isinstance(receipt, dict) and receipt.get("rank") == rank, "prior FP32 raw-shard rank drift")
        _require(set(receipt) == {"rank", "logical_name", "sha256", "bytes"}, "prior FP32 raw-shard receipt schema drift")
        _require(receipt.get("logical_name") == f"pg19-gate-shards/pg19-fair-v2-shard-{rank}.json", "prior FP32 raw-shard logical name drift")
        _validate_digest(receipt.get("sha256"), "prior FP32 raw-shard digest")
        _require(type(receipt.get("bytes")) is int and receipt["bytes"] > 0, "prior FP32 raw-shard byte count drift")
    scientific = archive.get("scientific_artifact_ledger")
    _require(isinstance(scientific, dict), "prior FP32 scientific-ledger receipt missing")
    _require(scientific.get("logical_name") == "scientific-artifacts.sha256", "prior FP32 scientific-ledger name drift")
    _require(scientific.get("verified_raw_shard_entries") == 8, "prior FP32 scientific-ledger cardinality drift")
    _validate_digest(scientific.get("normalized_raw_shard_entries_sha256"), "prior FP32 normalized shard-ledger digest")
    _require(scientific.get("source_path_strings_serialized_or_hashed") is False, "prior FP32 ledger leaks source paths")
    path_independence = value.get("path_independence")
    _require(
        path_independence
        == {
            "absolute_paths_serialized": False,
            "filesystem_metadata_serialized": False,
            "raw_artifacts_named_by_logical_relative_name": True,
            "timestamps_serialized": False,
        },
        "prior FP32 path-independence proof drift",
    )
    return {
        "diagnostic_count": 80,
        "maximum_observed_prior_relative_l2": maximum,
        "fixed_preregistered_threshold": 0.005,
        "required_margin_boundary_from_prior_maximum": required_boundary,
        "fixed_threshold_to_prior_maximum_ratio": 0.005 / maximum,
        "maximum_coordinate": expected_maximum_coordinate,
        "prior_coordinates_sha256": recomputed_coordinate_sha,
        "fixed_threshold_at_least_twice_prior_maximum": True,
        "rr2_document_length_disjoint": True,
        "prior_rows_selected_or_tuned_threshold": False,
    }


def _audit_review_response_plan(value: Any) -> dict[str, Any]:
    _require(isinstance(value, dict), "review-response plan must be an object")
    _require(value.get("schema_version") == "1.0.0", "review-response plan schema drift")
    _require(value.get("source_round") == 2, "review-response source round drift")
    items = value.get("items")
    _require(isinstance(items, list), "review-response experiment items missing")
    matches = [row for row in items if isinstance(row, dict) and row.get("experiment_id") == "RR2-EXP-OWNERSHIP-MUTANTS"]
    _require(len(matches) == 1, "RR2 reviewer-response experiment missing or duplicated")
    item = matches[0]
    _require(item.get("disposition") == "experiment_required", "RR2 reviewer response no longer requires experiment")
    execution = item.get("execution")
    _require(isinstance(execution, dict), "RR2 execution preregistration missing")
    configuration = execution.get("configuration")
    _require(isinstance(configuration, dict), "RR2 execution configuration missing")
    _require(configuration.get("document_tokens") == 4095, "RR2 review plan document length drift")
    _require(configuration.get("query_tokens") == 32, "RR2 review plan query length drift")
    _require(configuration.get("ownership_factorial_resident_counts") == [1, 8, 32], "RR2 review plan resident counts drift")
    _require(configuration.get("mutants") == 9, "RR2 review plan mutant count drift")
    return {
        "source_round": 2,
        "experiment_id": "RR2-EXP-OWNERSHIP-MUTANTS",
        "disposition": "experiment_required",
        "reviewer_feedback_precedes_new_experiment": True,
    }


def build_preregistration(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    _require(not torch.cuda.is_initialized(), "manifest builder must start before CUDA initialization")
    _require(args.model_id == FORMAL_MODEL_ID, "formal model ID drift")
    _require(args.model_revision == FORMAL_MODEL_REVISION, "formal model revision is not the frozen full 40-character commit")
    _require(re.fullmatch(r"[0-9a-f]{40}", args.model_revision) is not None, "model revision must be 40 lowercase hexadecimal characters")
    runner_compatibility = _runner_rr2_compatibility_gate()

    code_rows = validate_path_independent_sha_ledger(args.code_ledger, label="code ledger")
    model_artifact_rows = validate_path_independent_sha_ledger(
        args.model_artifact_ledger, label="model artifact ledger"
    )
    model_weight_rows = validate_path_independent_sha_ledger(
        args.model_weight_ledger, label="model weight ledger"
    )
    _require(len(model_weight_rows) == 14, "Qwen3.5-35B-A3B weight ledger must bind exactly 14 shards")

    model_manifest_sha256, model_manifest_rows = _model_manifest(args.model_dir)
    pg19_data_receipt = _artifact_receipt(args.pg19_data)
    pg19_manifest_receipt = _artifact_receipt(args.pg19_manifest)
    pg19_manifest = _read_json(args.pg19_manifest, label="PG19 manifest")
    pg19_sources = _audit_pg19_manifest(
        pg19_manifest, pg19_data_sha256=pg19_data_receipt["sha256"]
    )

    rr2_receipt = _require_receipt_sha(
        args.pg19_input_manifest,
        args.expected_pg19_input_manifest_sha256,
        label="RR2 PG19 input manifest",
    )
    prior_capacity_receipt = _require_receipt_sha(
        args.prior_capacity_manifest,
        rr2_input.PRIOR_CAPACITY_MANIFEST_SHA256,
        label="prior capacity manifest",
    )
    banks_input_receipt = _require_receipt_sha(
        args.frozen_query_banks_input,
        args.expected_frozen_query_banks_input_sha256,
        label="RR2 frozen-query-bank sidecar",
    )
    oracle_input_receipt = _require_receipt_sha(
        args.oracle_selection_input,
        args.expected_oracle_selection_input_sha256,
        label="RR2 oracle-selection sidecar",
    )
    expectations = rr2_input.InputExpectations(
        pg19_data_sha256=pg19_data_receipt["sha256"],
        pg19_manifest_sha256=pg19_manifest_receipt["sha256"],
        prior_manifest_sha256=prior_capacity_receipt["sha256"],
        prior_windows_sha256=rr2_input.PRIOR_CAPACITY_WINDOWS_SHA256,
        rr2_windows_sha256=rr2_input.FORMAL_RR2_WINDOWS_SHA256,
        rr2_coordinates=rr2_input.FORMAL_RR2_COORDINATES,
    )
    try:
        rr2_manifest, rr2_source_replay = _rebuild_rr2_inputs_from_source(
            args,
            expectations=expectations,
        )
    except rr2_input.RR2InputManifestError as exc:
        raise ManifestBuildError(f"RR2 source replay rejected: {exc}") from exc
    rr2_manifest_raw = _read_json(
        args.pg19_input_manifest,
        label="RR2 PG19 input manifest",
    )
    try:
        validated_supplied_manifest = rr2_input.validate_rr2_input_manifest(
            rr2_manifest_raw,
            expectations=expectations,
        )
    except rr2_input.RR2InputManifestError as exc:
        raise ManifestBuildError(f"RR2 PG19 input manifest rejected: {exc}") from exc
    _require(
        validated_supplied_manifest == rr2_manifest,
        "RR2 supplied/rebuilt manifest semantics differ",
    )
    _require(
        rr2_manifest["prior_capacity_cohort"]["protocol_manifest_sha256"]
        == prior_capacity_receipt["sha256"],
        "RR2 main manifest does not bind the exact prior capacity bytes",
    )

    banks_input = _read_json(args.frozen_query_banks_input, label="RR2 frozen query banks")
    plan_input = _read_json(args.oracle_selection_input, label="RR2 oracle selection")
    _require(banks_input == rr2_manifest["frozen_query_banks"], "query-bank sidecar differs from authoritative RR2 main manifest")
    _require(plan_input == rr2_manifest["oracle_selection_plan"], "oracle sidecar differs from authoritative RR2 main manifest")
    # Outputs are derived from the exact-byte-bound main manifest.  The two
    # external sidecars are independent receipts that must compare equal; they
    # are never promoted to the source of authority.
    plan = runner.validate_oracle_selection_plan(
        rr2_manifest["oracle_selection_plan"]
    )
    banks = runner.validate_frozen_query_banks(
        rr2_manifest["frozen_query_banks"], plan
    )
    _require(
        all(row["source_object"] in pg19_sources for row in plan),
        "oracle selection references a source outside the frozen PG19 train manifest",
    )
    tokenizer_cross_binding = _audit_model_tokenizer_cross_binding(
        args.model_dir,
        model_artifact_rows,
        rr2_manifest,
    )

    _require(
        args.expected_prior_fp32_context_manifest_sha256
        == PRIOR_FP32_CONTEXT_MANIFEST_SHA256,
        "prior FP32 context expected SHA is not the frozen fa64 manifest",
    )
    prior_context_receipt = _require_receipt_sha(
        args.prior_fp32_context_manifest,
        PRIOR_FP32_CONTEXT_MANIFEST_SHA256,
        label="prior FP32 context manifest",
    )
    prior_context = _read_json(
        args.prior_fp32_context_manifest,
        label="prior FP32 context manifest",
    )
    prior_context_audit = _audit_prior_fp32_context(
        prior_context,
        rr2_selection_plan=plan,
    )

    _require(
        args.expected_review_response_plan_sha256
        == runner.FINAL_REVIEW_RESPONSE_PLAN_SHA256,
        "review-response expected SHA is not the frozen round-2 plan",
    )
    review_plan_receipt = _require_receipt_sha(
        args.review_response_plan,
        runner.FINAL_REVIEW_RESPONSE_PLAN_SHA256,
        label="review-response experiment plan",
    )
    review_plan = _read_json(args.review_response_plan, label="review-response experiment plan")
    review_plan_audit = _audit_review_response_plan(review_plan)

    source_artifacts = {
        "code_ledger": _artifact_receipt(args.code_ledger),
        "model_artifact_ledger": _artifact_receipt(args.model_artifact_ledger),
        "model_weight_ledger": _artifact_receipt(args.model_weight_ledger),
        "pg19_data": pg19_data_receipt,
        "pg19_manifest": pg19_manifest_receipt,
        "pg19_input_manifest": rr2_receipt,
        "prior_capacity_manifest": prior_capacity_receipt,
        "frozen_query_banks_input": banks_input_receipt,
        "oracle_selection_input": oracle_input_receipt,
        "prior_fp32_context_manifest": prior_context_receipt,
        "review_response_plan": review_plan_receipt,
        "protocol_source_manifest": _artifact_receipt(args.protocol_source_manifest),
    }
    frozen_identity = {
        "code_ledger_sha256": source_artifacts["code_ledger"]["sha256"],
        "model_manifest_sha256": model_manifest_sha256,
        "model_artifact_ledger_sha256": source_artifacts["model_artifact_ledger"]["sha256"],
        "model_weight_ledger_sha256": source_artifacts["model_weight_ledger"]["sha256"],
        "pg19_data_sha256": pg19_data_receipt["sha256"],
        "pg19_manifest_sha256": pg19_manifest_receipt["sha256"],
        "pg19_windows_sha256": rr2_manifest["pg19_windows_sha256"],
        "pg19_input_manifest_sha256": rr2_receipt["sha256"],
        "prior_fp32_context_manifest_sha256": prior_context_receipt["sha256"],
        "review_response_plan_sha256": review_plan_receipt["sha256"],
        "protocol_manifest_sha256": source_artifacts["protocol_source_manifest"]["sha256"],
        "protocol_config_sha256": runner.sha256_json(runner.formal_protocol_config()),
        "model_id": FORMAL_MODEL_ID,
        "model_revision": FORMAL_MODEL_REVISION,
    }
    runner.validate_frozen_identity(frozen_identity)

    rank_assignments = [
        {
            "rank": rank,
            "book_index": rank,
            "source_object": plan[rank]["source_object"],
            "document_start_token": plan[rank]["document_start_token"],
            "document_token_ids_sha256": plan[rank]["document_token_ids_sha256"],
            "oracle_sample_id": plan[rank]["sample_id"],
            "oracle_layer_index": plan[rank]["layer_index"],
            "oracle_round_index": plan[rank]["round_index"],
            "oracle_arm_id": plan[rank]["arm_id"],
            "oracle_cell_id": plan[rank]["oracle_cell_id"],
            "mutant_ids": list(runner.MUTANT_ASSIGNMENT_BY_RANK[rank]),
        }
        for rank in range(runner.FORMAL_WORLD_SIZE)
    ]
    manifest = {
        "schema_version": RELEASE_MANIFEST_SCHEMA_VERSION,
        "protocol": runner.PROTOCOL,
        "formal_ready": False,
        "producer_gate": {
            "gpu_loop_implemented_at_build": runner.GPU_LOOP_IMPLEMENTED,
            "implementation_status_at_build": runner.IMPLEMENTATION_STATUS,
            "launcher_release_gate_required": True,
            "formal_execution_authorized": runner.GPU_LOOP_IMPLEMENTED,
        },
        "runner_rr2_compatibility": runner_compatibility,
        "protocol_config": runner.formal_protocol_config(),
        "protocol_config_sha256": frozen_identity["protocol_config_sha256"],
        "frozen_identity": frozen_identity,
        "frozen_identity_sha256": runner.sha256_json(frozen_identity),
        "rr2_input_binding": {
            "authoritative_main_raw_sha256": rr2_receipt["sha256"],
            "algorithm_windows_sha256": rr2_manifest["pg19_windows_sha256"],
            "algorithm_and_raw_manifest_digests_are_distinct_semantics": True,
            "query_bank_sidecar_equals_authoritative_main": True,
            "oracle_sidecar_equals_authoritative_main": True,
            "frozen_query_banks_sha256": runner.sha256_json(banks),
            "oracle_selection_plan_sha256": runner.sha256_json(plan),
            "n_prefix_rows": len(rr2_manifest["n_prefixes_by_rank"]),
            "prior_capacity_exact_bytes_bound": True,
        },
        "rr2_source_replay": rr2_source_replay,
        "oracle_selection_plan_sha256": runner.sha256_json(plan),
        "oracle_selection_locked_before_candidate_outputs": True,
        "tokenizer_model_ledger_cross_binding": tokenizer_cross_binding,
        "prior_fp32_context_audit": prior_context_audit,
        "review_response_plan_audit": review_plan_audit,
        "rank_assignments": rank_assignments,
        "mutant_case_isolation": {
            "fresh_document_cache_per_case": True,
            "fresh_request_cache_per_case": True,
            "cache_discarded_after_each_case": True,
            "reuse_across_mutants_forbidden": True,
        },
        "raw_artifact_integrity": {
            "detached_external_sha256_receipts_required": True,
            "raw_phase_oracle_and_mutant_artifacts_bound_transitively_by_shard_bytes": True,
            "aggregate_recomputes_from_raw_artifacts": True,
            "producer_passed_fields_trusted": False,
        },
        "measurement_cell_isolation": {
            "formal_memory_cell": "no request guard and no witness hashing",
            "ownership_witness_cell": "fresh persistent cache and request group",
            "cell_ids_must_differ": True,
            "witness_cell_ineligible_for_primary_memory_endpoint": True,
        },
        "environment_gates": {
            "real_transformers_5_14_1_call_stack_test_required": True,
            "focused_test_skips_allowed": 0,
            "ledger_sort_locale": "C",
            "python_pycache_must_be_outside_frozen_snapshot": True,
            "hard_timeouts_and_signal_traps_required": True,
            "done_marker_after_terminal_receipt_verification": True,
        },
        "data_policy": {
            "dataset": "PG19",
            "split": "train",
            "distinct_books": runner.FORMAL_BOOKS,
            "longbench_consumed": False,
            "validation_consumed": False,
            "test_v2_consumed": False,
        },
        "source_artifacts": source_artifacts,
        "source_inventory": {
            "code_ledger_entries": len(code_rows),
            "model_manifest_components": model_manifest_rows,
            "model_artifact_ledger_entries": len(model_artifact_rows),
            "model_weight_shards": len(model_weight_rows),
            "pg19_train_objects": len(pg19_sources),
            "rr2_query_banks": len(banks),
            "rr2_query_rows": sum(len(bank["rows"]) for bank in banks),
            "prior_fp32_diagnostics": 80,
        },
    }
    _assert_no_serialized_host_paths(manifest)
    _require(not torch.cuda.is_initialized(), "manifest build initialized CUDA")
    return manifest, frozen_identity, plan, banks


def build_receipts(
    *,
    artifact_root: Path,
    static_artifact: Path,
    run_id_receipt: Path,
    expected_run_id_receipt_sha256: str,
    run_id: str,
    protocol_manifest_sha256: str,
) -> dict[str, Any]:
    _require(not torch.cuda.is_initialized(), "receipt builder must not initialize CUDA")
    root = artifact_root.resolve()
    _require(root.is_dir(), "artifact root is missing")
    static = _read_json(static_artifact, label="static artifact")
    static_replay = runner.validate_static_artifact(static)
    static_sha256 = runner.sha256_json(static)
    frozen_identity = static_replay["frozen_identity"]
    _require(
        protocol_manifest_sha256 == frozen_identity["protocol_manifest_sha256"],
        "receipt protocol-manifest SHA differs from frozen identity",
    )
    shared_run_id_receipt = validate_run_id_receipt(
        _read_json(run_id_receipt, label="shared run-ID receipt"),
        expected_sha256=expected_run_id_receipt_sha256,
        run_id=run_id,
        static_artifact_sha256=static_sha256,
        protocol_manifest_sha256=protocol_manifest_sha256,
    )
    shard_paths: list[Path] = []
    for rank in range(runner.FORMAL_WORLD_SIZE):
        relative = RAW_SHARD_PATTERN.format(rank=rank)
        path = (root / relative).resolve()
        _require(root in path.parents, "raw shard escaped artifact root")
        shard = _read_json(path, label=f"raw shard {rank}")
        _require(isinstance(shard, dict), f"raw shard {rank} must be an object")
        _require(shard.get("schema_version") == runner.SHARD_SCHEMA_VERSION, f"raw shard {rank} schema drift")
        _require(shard.get("protocol") == runner.PROTOCOL, f"raw shard {rank} protocol drift")
        _require(shard.get("rank") == rank, f"raw shard {rank} rank drift")
        _require(shard.get("world_size") == runner.FORMAL_WORLD_SIZE, f"raw shard {rank} world-size drift")
        _require(shard.get("static_artifact_sha256") == static_sha256, f"raw shard {rank} static binding drift")
        _require(shard.get("run_id") == run_id, f"raw shard {rank} run-ID drift")
        _require(
            shard.get("run_id_receipt_sha256")
            == expected_run_id_receipt_sha256,
            f"raw shard {rank} run-ID receipt SHA drift",
        )
        _require(
            shard.get("run_id_receipt") == shared_run_id_receipt,
            f"raw shard {rank} does not embed the shared run-ID receipt",
        )
        shard_paths.append(path)
    result = runner.make_receipt_manifest(
        shard_paths,
        root=root,
        static_artifact_sha256=static_sha256,
    )
    _assert_no_serialized_host_paths(result)
    _require(not torch.cuda.is_initialized(), "receipt build initialized CUDA")
    return result


def _add_preregister_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("preregister")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-identity-output", type=Path, required=True)
    parser.add_argument("--frozen-query-banks-output", type=Path, required=True)
    parser.add_argument("--oracle-selection-output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--code-ledger", type=Path, required=True)
    parser.add_argument("--model-artifact-ledger", type=Path, required=True)
    parser.add_argument("--model-weight-ledger", type=Path, required=True)
    parser.add_argument("--pg19-data", type=Path, required=True)
    parser.add_argument("--pg19-manifest", type=Path, required=True)
    parser.add_argument("--pg19-input-manifest", type=Path, required=True)
    parser.add_argument("--expected-pg19-input-manifest-sha256", required=True)
    parser.add_argument("--prior-capacity-manifest", type=Path, required=True)
    parser.add_argument("--frozen-query-banks-input", type=Path, required=True)
    parser.add_argument("--expected-frozen-query-banks-input-sha256", required=True)
    parser.add_argument("--protocol-source-manifest", type=Path, required=True)
    parser.add_argument("--oracle-selection-input", type=Path, required=True)
    parser.add_argument("--expected-oracle-selection-input-sha256", required=True)
    parser.add_argument("--prior-fp32-context-manifest", type=Path, required=True)
    parser.add_argument("--expected-prior-fp32-context-manifest-sha256", required=True)
    parser.add_argument("--review-response-plan", type=Path, required=True)
    parser.add_argument("--expected-review-response-plan-sha256", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_preregister_parser(subparsers)
    receipts = subparsers.add_parser("receipts")
    receipts.add_argument("--artifact-root", type=Path, required=True)
    receipts.add_argument("--static-artifact", type=Path, required=True)
    receipts.add_argument("--run-id-receipt", type=Path, required=True)
    receipts.add_argument("--expected-run-id-receipt-sha256", required=True)
    receipts.add_argument("--run-id", required=True)
    receipts.add_argument("--protocol-manifest-sha256", required=True)
    receipts.add_argument("--output", type=Path, required=True)
    run_id = subparsers.add_parser("run-id-receipt")
    run_id.add_argument("--static-artifact-sha256", required=True)
    run_id.add_argument("--protocol-manifest-sha256", required=True)
    run_id.add_argument("--output", type=Path, required=True)
    gpu_assignment = subparsers.add_parser("gpu-assignment-receipt")
    gpu_assignment.add_argument("--inventory", type=Path, required=True)
    gpu_assignment.add_argument("--output", type=Path, required=True)
    private_view = subparsers.add_parser("materialize-private-model-view")
    private_view.add_argument("--source-model-dir", type=Path, required=True)
    private_view.add_argument("--private-model-view", type=Path, required=True)
    private_view.add_argument("--model-artifact-ledger", type=Path, required=True)
    private_view.add_argument(
        "--expected-model-artifact-ledger-raw-sha256", required=True
    )
    private_view.add_argument("--model-weight-ledger", type=Path, required=True)
    private_view.add_argument(
        "--expected-model-weight-ledger-raw-sha256", required=True
    )
    private_view.add_argument("--model-id", required=True)
    private_view.add_argument("--model-revision", required=True)
    private_view.add_argument("--manifest-output", type=Path, required=True)
    lease_keeper = subparsers.add_parser("model-load-lease-keeper")
    lease_keeper.add_argument("--model-view", type=Path, required=True)
    lease_keeper.add_argument("--model-weight-ledger", type=Path, required=True)
    lease_keeper.add_argument(
        "--expected-model-weight-ledger-raw-sha256", required=True
    )
    lease_keeper.add_argument(
        "--expected-model-artifact-ledger-raw-sha256", required=True
    )
    lease_keeper.add_argument("--model-view-manifest", type=Path, required=True)
    lease_keeper.add_argument(
        "--expected-model-view-manifest-raw-sha256", required=True
    )
    lease_keeper.add_argument("--run-id", required=True)
    lease_keeper.add_argument("--authority-output", type=Path, required=True)
    lease_keeper.add_argument("--closure-output", type=Path, required=True)
    digest = subparsers.add_parser("digest-json")
    digest.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "preregister":
        outputs = {
            args.output.resolve(),
            args.frozen_identity_output.resolve(),
            args.frozen_query_banks_output.resolve(),
            args.oracle_selection_output.resolve(),
        }
        _require(len(outputs) == 4, "preregistration outputs must be four distinct files")
        manifest, identity, plan, banks = build_preregistration(args)
        _atomic_json(args.output, manifest)
        _atomic_json(args.frozen_identity_output, identity)
        _atomic_json(args.frozen_query_banks_output, banks)
        _atomic_json(args.oracle_selection_output, plan)
        print(
            json.dumps(
                {
                    "status": "forkaudit_preregistration_built",
                    "release_manifest_sha256": runner.sha256_json(manifest),
                    "frozen_identity_sha256": runner.sha256_json(identity),
                    "frozen_query_banks_sha256": runner.sha256_json(banks),
                    "oracle_selection_plan_sha256": runner.sha256_json(plan),
                    "gpu_initialized": torch.cuda.is_initialized(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "receipts":
        receipts = build_receipts(
            artifact_root=args.artifact_root,
            static_artifact=args.static_artifact,
            run_id_receipt=args.run_id_receipt,
            expected_run_id_receipt_sha256=args.expected_run_id_receipt_sha256,
            run_id=args.run_id,
            protocol_manifest_sha256=args.protocol_manifest_sha256,
        )
        _atomic_json(args.output, receipts)
        print(
            json.dumps(
                {
                    "status": "forkaudit_detached_receipts_built",
                    "receipt_manifest_sha256": runner.sha256_json(receipts),
                    "shard_count": runner.FORMAL_WORLD_SIZE,
                    "run_id": args.run_id,
                    "run_id_receipt_sha256": args.expected_run_id_receipt_sha256,
                    "gpu_initialized": torch.cuda.is_initialized(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "run-id-receipt":
        receipt = build_run_id_receipt(
            static_artifact_sha256=args.static_artifact_sha256,
            protocol_manifest_sha256=args.protocol_manifest_sha256,
        )
        _atomic_json(args.output, receipt)
        print(receipt["run_id"])
        return 0
    if args.command == "gpu-assignment-receipt":
        try:
            inventory_raw = args.inventory.read_bytes()
        except OSError as exc:
            raise ManifestBuildError("GPU inventory cannot be read") from exc
        receipt = build_gpu_assignment_receipt(inventory_raw)
        _atomic_json(args.output, receipt)
        print(
            json.dumps(
                {
                    "status": "forkaudit_gpu_assignment_receipt_built",
                    "raw_sha256": _sha256_file(args.output),
                    "gpu_count": runner.FORMAL_WORLD_SIZE,
                    "cuda_initialized": torch.cuda.is_initialized(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "materialize-private-model-view":
        view_manifest = materialize_private_model_view(
            source_model_dir=args.source_model_dir,
            private_model_view=args.private_model_view,
            model_artifact_ledger=args.model_artifact_ledger,
            expected_model_artifact_ledger_raw_sha256=(
                args.expected_model_artifact_ledger_raw_sha256
            ),
            model_weight_ledger=args.model_weight_ledger,
            expected_model_weight_ledger_raw_sha256=(
                args.expected_model_weight_ledger_raw_sha256
            ),
            model_id=args.model_id,
            model_revision=args.model_revision,
        )
        _atomic_json(args.manifest_output, view_manifest)
        print(
            json.dumps(
                {
                    "status": "forkaudit_private_model_view_materialized",
                    "manifest_raw_sha256": _sha256_file(args.manifest_output),
                    "file_count": view_manifest["file_count"],
                    "weight_file_count": view_manifest["weight_file_count"],
                    "cuda_initialized": torch.cuda.is_initialized(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if args.command == "model-load-lease-keeper":
        run_model_load_lease_keeper(
            model_view=args.model_view,
            model_weight_ledger=args.model_weight_ledger,
            expected_model_weight_ledger_raw_sha256=(
                args.expected_model_weight_ledger_raw_sha256
            ),
            expected_model_artifact_ledger_raw_sha256=(
                args.expected_model_artifact_ledger_raw_sha256
            ),
            model_view_manifest=args.model_view_manifest,
            expected_model_view_manifest_raw_sha256=(
                args.expected_model_view_manifest_raw_sha256
            ),
            run_id=args.run_id,
            authority_output=args.authority_output,
            closure_output=args.closure_output,
            control_input=sys.stdin,
            event_output=sys.stdout,
        )
        return 0
    value = _read_json(args.input, label="canonical JSON digest input")
    print(runner.sha256_json(value))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestBuildError, runner.ReviewAuditError) as exc:
        raise SystemExit(f"ForkAudit release manifest rejected: {exc}") from exc


__all__ = [
    "FORMAL_MODEL_ID",
    "FORMAL_MODEL_REVISION",
    "ManifestBuildError",
    "RAW_SHARD_PATTERN",
    "RELEASE_MANIFEST_SCHEMA_VERSION",
    "build_preregistration",
    "build_receipts",
    "build_run_id_receipt",
    "main",
    "validate_path_independent_sha_ledger",
]
