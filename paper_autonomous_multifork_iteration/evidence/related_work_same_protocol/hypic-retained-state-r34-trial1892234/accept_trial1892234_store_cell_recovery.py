#!/usr/bin/env python3
"""Recovery-scoped cell validator and exact Store extraction for Trial 1892234.

This is a mechanically copied, narrowly patched version of the original strict
whole-run verifier.  Recovery acceptance imports only its cell-local validation
functions.  The sole cell patch recognizes the producer's two-layer server
configuration representation: the raw shard preserves the exact /server_info
configuration, while the launch receipt extends that same object with exact
``rwd5_expected`` and ``rwd5_observed`` assertions.  It also distinguishes
CUDA-resident retained-state payloads from excluded index metadata: HYPIC
``token_ids`` metadata is exactly CPU-resident, whereas slot metadata and every
Store payload tensor remain exactly ``cuda:0``.  No evidence file is edited.

The standalone whole-run ``main`` below is intentionally not used for recovery
because Trial 1892234 has FAILED markers and lacks COMPLETED/stage 99 closure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import sys
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any


JOB_ID = 247699
TRIAL_ID = 1892234
SCOPE = "ROUND27_HYPIC_STORE_FORMAL_W"
OFFICIAL_COMMIT = "98147c01909004e66d98bcb18b886927d41b0ee5"
EXPECTED_BOOT_ROOT_NFS = (
    "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/"
    "runs/qcomem/hypic-rwd5-autobootstrap-job247699-trial1892234"
)
EXPECTED_FORMAL_ROOT_NFS = f"{EXPECTED_BOOT_ROOT_NFS}/formal-run"
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "38219b146dbe5bf56e74262491aaa3f0f1b023f636278e0986c1e2b18f3dfd40"
)
EXPECTED_INITIAL_FLUSH_RESPONSE = (
    "Cache flushed.\nPlease check backend logs for more details. "
    "(When there are running or waiting requests, the operation will not be performed.)\n"
)
EXPECTED_REPLAY_CODE_SHA256 = (
    "ccff3178045eecb4daf5675721aaa800d59fc5ea989a7822c3130fdb34d0fb27"
)
COMEM_Q8_BYTES = 16_664_352
MIB = 1_048_576
MODES = ("prefix_cache", "transition_rope_recompute")
EXPECTED_PAIRS = (
    ("qasper", 6),
    ("qasper", 7),
    ("qasper", 8),
    ("qasper", 9),
    ("2wikimqa", 6),
    ("2wikimqa", 7),
    ("2wikimqa", 8),
    ("2wikimqa", 9),
)
DTYPE_BYTES = {
    "torch.float32": 4,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.int64": 8,
    "torch.int32": 4,
    "torch.int8": 1,
    "torch.uint8": 1,
}
STORE_DENOMINATOR = (
    "blindly rederived exact target-entry-owned physical tensor-range union"
)
STORE_SCOPE = "exact target-entry-owned physical tensor-range union only"
METRIC_VALIDITY = (
    "valid_for_exact_target_owned_physical_tensor_range_union_only"
)
BLIND_DENOMINATOR = (
    "independently derived exact target-entry-owned physical tensor-range union; "
    "metadata excluded"
)
BLIND_CLAIM_BOUNDARY = (
    "allocator free-list anomalies are preserved and reported; no global allocator "
    "correctness or runtime-safety claim is made"
)
MAMBA_BASIS = (
    "unique in-domain physical slot identities; duplicate multiset anomaly reported "
    "separately"
)
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\n$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
LEDGER_RE = re.compile(r"^([0-9a-f]{64})  (/.+)$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")


class AcceptanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AcceptanceError(message)


def exact_int(value: Any, label: str) -> int:
    require(type(value) is int, f"{label}: exact JSON integer required")
    return value


def _identity(row: os.stat_result) -> tuple[int, ...]:
    return (
        row.st_dev,
        row.st_ino,
        row.st_mode,
        row.st_uid,
        row.st_gid,
        row.st_size,
        row.st_mtime_ns,
        row.st_ctime_ns,
    )


def read_regular_bytes(path: Path, label: str) -> bytes:
    try:
        named_before = path.lstat()
    except OSError as error:
        raise AcceptanceError(f"{label}: missing/unreadable file: {path}: {error}") from error
    require(stat.S_ISREG(named_before.st_mode), f"{label}: regular non-symlink file required")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AcceptanceError(f"{label}: safe open failed: {path}: {error}") from error
    try:
        opened_before = os.fstat(descriptor)
        require(
            _identity(named_before) == _identity(opened_before),
            f"{label}: path/open identity race",
        )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        opened_after = os.fstat(descriptor)
        require(
            _identity(opened_before) == _identity(opened_after),
            f"{label}: file changed while reading",
        )
    finally:
        os.close(descriptor)
    named_after = path.lstat()
    require(
        _identity(opened_after) == _identity(named_after),
        f"{label}: path changed after reading",
    )
    return b"".join(chunks)


def sha256_file(path: Path, label: str) -> str:
    try:
        named_before = path.lstat()
    except OSError as error:
        raise AcceptanceError(f"{label}: missing/unreadable file: {path}: {error}") from error
    require(stat.S_ISREG(named_before.st_mode), f"{label}: regular non-symlink file required")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise AcceptanceError(f"{label}: safe open failed: {path}: {error}") from error
    digest = hashlib.sha256()
    try:
        opened_before = os.fstat(descriptor)
        require(
            _identity(named_before) == _identity(opened_before),
            f"{label}: path/open identity race",
        )
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        opened_after = os.fstat(descriptor)
        require(
            _identity(opened_before) == _identity(opened_after),
            f"{label}: file changed while hashing",
        )
    finally:
        os.close(descriptor)
    named_after = path.lstat()
    require(
        _identity(opened_after) == _identity(named_after),
        f"{label}: path changed after hashing",
    )
    return digest.hexdigest()


def _reject_constant(value: str) -> None:
    raise AcceptanceError(f"non-finite JSON constant forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AcceptanceError(f"non-canonical JSON value: {error}") from error


def load_json(path: Path, label: str) -> dict[str, Any]:
    raw = read_regular_bytes(path, label)
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise AcceptanceError(f"{label}: strict JSON decode failed: {error}") from error
    require(isinstance(value, dict), f"{label}: JSON object required")
    require(raw == canonical_json_bytes(value), f"{label}: canonical JSON bytes required")
    return value


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    require(not missing and not extra, f"{label}: key drift; missing={missing}, extra={extra}")


def require_directory(path: Path, label: str) -> None:
    try:
        row = path.lstat()
    except OSError as error:
        raise AcceptanceError(f"{label}: missing directory: {path}: {error}") from error
    require(stat.S_ISDIR(row.st_mode), f"{label}: real directory required")


def exact_children(directory: Path, expected: set[str], label: str) -> None:
    require_directory(directory, label)
    actual = {path.name for path in directory.iterdir()}
    require(actual == expected, f"{label}: exact file set drift; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
    for name in sorted(expected):
        row = (directory / name).lstat()
        require(stat.S_ISREG(row.st_mode), f"{label}/{name}: regular non-symlink file required")


def parse_timestamp_file(path: Path, label: str) -> datetime:
    raw = read_regular_bytes(path, label)
    try:
        text = raw.decode("ascii", errors="strict")
    except UnicodeError as error:
        raise AcceptanceError(f"{label}: ASCII timestamp required") from error
    require(TIMESTAMP_RE.fullmatch(text) is not None, f"{label}: canonical UTC timestamp required")
    return datetime.strptime(text.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def actual_regular_files(root: Path) -> set[str]:
    rows: set[str] = set()
    for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        retained_dirs = []
        for name in dirnames:
            candidate = base / name
            rel = candidate.relative_to(root).as_posix()
            mode = candidate.lstat().st_mode
            if stat.S_ISDIR(mode):
                retained_dirs.append(name)
            elif rel.split("/", 1)[0] != "caches":
                raise AcceptanceError(f"non-directory entry outside caches: {rel}")
        dirnames[:] = sorted(retained_dirs)
        for name in sorted(filenames):
            candidate = base / name
            rel = candidate.relative_to(root).as_posix()
            mode = candidate.lstat().st_mode
            if stat.S_ISREG(mode):
                rows.add(rel)
            elif rel.split("/", 1)[0] != "caches":
                raise AcceptanceError(f"non-regular entry outside caches: {rel}")
    return rows


def validate_artifact_ledger(formal: Path) -> tuple[dict[str, str], str]:
    ledger_path = formal / "all-artifacts.sha256"
    ledger_raw = read_regular_bytes(ledger_path, "artifact ledger")
    require(ledger_raw.endswith(b"\n") and ledger_raw != b"", "artifact ledger: terminal newline/nonempty")
    try:
        lines = ledger_raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as error:
        raise AcceptanceError("artifact ledger: UTF-8 required") from error
    expected_prefix = EXPECTED_FORMAL_ROOT_NFS + "/"
    rows: dict[str, str] = {}
    absolute_names: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        match = LEDGER_RE.fullmatch(line)
        require(match is not None, f"artifact ledger line {line_number}: canonical sha256sum row required")
        digest, absolute_name = match.groups()
        require(absolute_name.startswith(expected_prefix), f"artifact ledger line {line_number}: NFS authority prefix drift")
        relative = absolute_name[len(expected_prefix):]
        pure = PurePosixPath(relative)
        require(
            relative != ""
            and not pure.is_absolute()
            and "." not in pure.parts
            and ".." not in pure.parts
            and pure.as_posix() == relative,
            f"artifact ledger line {line_number}: confined canonical relative path required",
        )
        require(relative not in rows, f"artifact ledger line {line_number}: duplicate path")
        require(
            relative not in {"all-artifacts.sha256", "COMPLETED", "stages/99_done"},
            f"artifact ledger line {line_number}: excluded terminal file unexpectedly ledgered",
        )
        rows[relative] = digest
        absolute_names.append(absolute_name)
    require(
        absolute_names == sorted(absolute_names, key=lambda item: item.encode("utf-8")),
        "artifact ledger: bytewise path order drift",
    )
    expected_actual = set(rows) | {"all-artifacts.sha256", "COMPLETED", "stages/99_done"}
    actual_before = actual_regular_files(formal)
    require(
        actual_before == expected_actual,
        "artifact ledger: exact regular-file closure failed; "
        f"missing={sorted(expected_actual-actual_before)}, extra={sorted(actual_before-expected_actual)}",
    )
    for relative, expected_digest in rows.items():
        observed = sha256_file(formal.joinpath(*PurePosixPath(relative).parts), f"ledger member {relative}")
        require(observed == expected_digest, f"artifact ledger: SHA drift for {relative}")
    actual_after = actual_regular_files(formal)
    require(actual_after == actual_before, "artifact ledger: file set changed during verification")
    require(
        hashlib.sha256(read_regular_bytes(ledger_path, "artifact ledger final stability")).hexdigest()
        == hashlib.sha256(ledger_raw).hexdigest(),
        "artifact ledger changed during verification",
    )
    return rows, hashlib.sha256(ledger_raw).hexdigest()


def validate_core_topology(formal: Path) -> dict[str, datetime]:
    cell_names = {f"{mode}-rank-{rank}" for mode in MODES for rank in range(8)}
    expected_directories = {
        "raw",
        "targets",
        "store-receipts",
        "server-receipts",
        "scheduler-workers",
        "server-logs",
        "logs",
        "commands",
        "stages",
        "caches",
        "static",
    }
    expected_files = {
        "COMPLETED",
        "all-artifacts.sha256",
        "blind-replay.json",
        "terminal-static-verification.json",
        "terminal-idle-compute.csv",
        "terminal-idle-gpus.csv",
        "terminal-idle-processes.txt",
    }
    top = {path.name for path in formal.iterdir()}
    require(
        top == expected_directories | expected_files,
        f"formal root topology drift; missing={sorted((expected_directories|expected_files)-top)}, extra={sorted(top-(expected_directories|expected_files))}",
    )
    for name in expected_directories:
        require_directory(formal / name, f"formal/{name}")
    for name in expected_files:
        read_regular_bytes(formal / name, f"formal/{name}")
    require(read_regular_bytes(formal / "COMPLETED", "formal COMPLETED") == b"", "formal COMPLETED must be empty")
    require(not (formal / "FAILED").exists(), "formal FAILED marker must be absent")

    exact_children(formal / "raw", {f"{name}.json" for name in cell_names}, "raw")
    exact_children(formal / "targets", {f"{name}.json" for name in cell_names}, "targets")
    exact_children(
        formal / "store-receipts",
        {f"{name}.json" for name in cell_names} | {f"{name}.terminal.json" for name in cell_names},
        "store-receipts",
    )
    exact_children(
        formal / "server-receipts",
        {f"{name}.json" for name in cell_names} | {f"{name}.readiness.json" for name in cell_names},
        "server-receipts",
    )
    exact_children(
        formal / "scheduler-workers",
        {f"{name}.json" for name in cell_names},
        "scheduler-workers",
    )
    exact_children(formal / "commands", {f"{name}.txt" for name in cell_names}, "commands")
    exact_children(
        formal / "server-logs",
        {f"{name}.log" for name in cell_names} | {f"{name}.pid" for name in cell_names},
        "server-logs",
    )
    exact_children(
        formal / "logs",
        {"focused-tests.log", "inherited-same-protocol-tests.log"}
        | {f"client-{name}.log" for name in cell_names},
        "logs",
    )
    static_files = {
        "official-source-ledger.json",
        "environment-ledger.json",
        "model-storage-contract.json",
        "instrumentation-overlay.json",
        "instrumentation-overlay.diff",
        "preregistration.json",
        "preoutput-validation.json",
    }
    exact_children(formal / "static", static_files, "static")
    stage_order = (
        "00_started",
        "01_focused_and_inherited_tests_passed",
        "02_preregistered_before_outputs",
        "10_prefix_cache_server_info_ready",
        "20_prefix_cache_complete",
        "10_transition_rope_recompute_server_info_ready",
        "20_transition_rope_recompute_complete",
        "30_blind_replay_complete",
        "99_done",
    )
    exact_children(formal / "stages", set(stage_order), "stages")
    times = {
        name: parse_timestamp_file(formal / "stages" / name, f"stage {name}")
        for name in stage_order
    }
    ordered = [times[name] for name in stage_order]
    require(ordered == sorted(ordered), "stage timestamps must be nondecreasing in frozen execution order")
    for name in cell_names:
        pid_text = read_regular_bytes(formal / "server-logs" / f"{name}.pid", f"PID {name}")
        require(re.fullmatch(rb"[0-9]+\n", pid_text) is not None and int(pid_text) > 1, f"PID {name}: positive canonical PID required")
        require(read_regular_bytes(formal / "commands" / f"{name}.txt", f"command {name}") != b"", f"command {name}: empty")
    return times


def validate_outer_authority(boot_root: Path, formal: Path, stage_times: dict[str, datetime]) -> tuple[dict[str, Any], dict[str, Any]]:
    require_directory(boot_root, "boot root")
    require(formal == boot_root / "formal-run", "formal root location drift")
    require(not (boot_root / "FAILED").exists(), "outer FAILED marker must be absent")
    bootstrap_time = parse_timestamp_file(boot_root / "BOOTSTRAP_COMPLETED", "BOOTSTRAP_COMPLETED")
    require(bootstrap_time >= stage_times["99_done"], "outer completion precedes formal stage 99")
    bootstrap_log = read_regular_bytes(boot_root / "bootstrap.log", "bootstrap log")
    require(
        bootstrap_log != b""
        and f"bootstrap start job={JOB_ID} trial={TRIAL_ID}".encode() in bootstrap_log,
        "bootstrap log: expected trial start authority absent",
    )
    platform = load_json(boot_root / "dynamic-platform-command-authority.json", "platform authority")
    require_exact_keys(
        platform,
        {
            "schema",
            "platform_job_id",
            "platform_trial_id",
            "scope",
            "status_at_bootstrap",
            "platform_command_pid",
            "platform_command",
            "platform_command_environ_sha256",
            "runtime_platform_command_environ_required",
            "gpu_count",
            "gpu_name",
            "gpu_memory_mib",
            "gpu_uuids",
            "paper_evidence",
            "formal_cells_before_bootstrap",
        },
        "platform authority",
    )
    require(
        platform["schema"] == "hypic-rwd5-dynamic-platform-command-authority-v1"
        and platform["platform_job_id"] == JOB_ID
        and platform["platform_trial_id"] == TRIAL_ID
        and platform["scope"] == SCOPE
        and platform["status_at_bootstrap"] == "Running"
        and exact_int(platform["platform_command_pid"], "platform PID") > 1
        and isinstance(platform["platform_command"], list)
        and bool(platform["platform_command"])
        and SHA_RE.fullmatch(str(platform["platform_command_environ_sha256"])) is not None
        and platform["runtime_platform_command_environ_required"]
        == {"QS_JOB_ID": str(JOB_ID), "QS_TRIAL_ID": str(TRIAL_ID), "QCOMEM_DEBUG_SCOPE": SCOPE}
        and platform["gpu_count"] == 8
        and platform["gpu_name"] == "NVIDIA H20-3e"
        and platform["gpu_memory_mib"] == 143771
        and platform["paper_evidence"] is False
        and platform["formal_cells_before_bootstrap"] == 0,
        "platform authority: frozen identity drift",
    )
    uuids = platform["gpu_uuids"]
    require(
        isinstance(uuids, list)
        and len(uuids) == 8
        and len(set(uuids)) == 8
        and all(isinstance(value, str) and GPU_UUID_RE.fullmatch(value) for value in uuids),
        "platform authority: exact eight unique GPU UUIDs required",
    )

    asset = load_json(boot_root / "model-asset-observation.json", "model asset observation")
    require_exact_keys(
        asset,
        {
            "schema",
            "model_root",
            "entries",
            "cross_node_authority_excludes_inode_device_and_timestamps",
            "same_preflight_requires_exact_observation_equality",
        },
        "model asset observation",
    )
    require(
        asset["schema"] == "hypic-rwd5-model-asset-snapshot-v1"
        and asset["model_root"] == "/tmp/Qwen3.5-35B-A3B-hypic-model-view"
        and asset["cross_node_authority_excludes_inode_device_and_timestamps"] is True
        and asset["same_preflight_requires_exact_observation_equality"] is True,
        "model asset observation: authority boundary drift",
    )
    expected_assets = {
        "model-artifacts.sha256": ("d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd", 778),
        "preprocessor_config.json": ("27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516", 390),
        "video_preprocessor_config.json": ("7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13", 385),
    }
    entries = asset["entries"]
    require(isinstance(entries, list) and [row.get("name") for row in entries] == sorted(expected_assets), "model asset observation: exact entries/order")
    for row in entries:
        digest, size = expected_assets[row["name"]]
        require(
            row.get("sha256") == digest
            and row.get("stable_cross_node_authority")
            == {"regular_non_symlink": True, "mode_octal": "0444", "uid": 0, "gid": 0, "size": size}
            and row.get("physical_identity_fields_are_observation_only") is True
            and row.get("atime_excluded_because_hashing_is_a_read") is True,
            f"model asset observation: stable authority drift for {row['name']}",
        )
    return platform, asset


def validate_static(formal: Path, boot_root: Path, platform: dict[str, Any], asset: dict[str, Any], ledger: dict[str, str]) -> tuple[dict[str, Any], str]:
    static = formal / "static"
    payload_names = {
        "official-source-ledger.json",
        "environment-ledger.json",
        "model-storage-contract.json",
        "instrumentation-overlay.json",
        "instrumentation-overlay.diff",
        "preregistration.json",
    }
    pre = load_json(static / "preoutput-validation.json", "preoutput validation")
    require_exact_keys(pre, {"schema", "passed", "files"}, "preoutput validation")
    require(
        pre["schema"] == "hypic-rwd5-preoutput-validation-v1"
        and pre["passed"] is True
        and isinstance(pre["files"], dict)
        and set(pre["files"]) == payload_names,
        "preoutput validation: terminal contract drift",
    )
    for name in payload_names:
        require(pre["files"][name] == ledger[f"static/{name}"], f"preoutput validation: SHA drift for {name}")

    terminal = load_json(formal / "terminal-static-verification.json", "terminal static verification")
    require_exact_keys(terminal, {"schema", "passed", "model_bytes_rehashed", "files"}, "terminal static verification")
    require(
        terminal["schema"] == "hypic-rwd5-terminal-static-verification-v1"
        and terminal["passed"] is True
        and terminal["model_bytes_rehashed"] is True
        and terminal["files"] == pre["files"],
        "terminal static verification: build/terminal byte closure failed",
    )

    prereg = load_json(static / "preregistration.json", "preregistration")
    prereg_sha = ledger["static/preregistration.json"]
    require(
        prereg.get("schema") == "hypic-rwd5-retained-state-preregistration-v2"
        and prereg.get("status") == "frozen_before_outputs"
        and prereg.get("official_commit") == OFFICIAL_COMMIT
        and prereg.get("external_freeze", {}).get("manifest_sha256") == EXPECTED_RUNTIME_MANIFEST_SHA256
        and prereg.get("external_freeze", {}).get("manifest_member_count") == 77
        and prereg.get("external_freeze", {}).get("all_entries_verified") is True
        and prereg.get("external_freeze", {}).get("single_captured_byte_stream") is True
        and prereg.get("design", {}).get("modes") == list(MODES)
        and prereg.get("design", {}).get("cells") == 16
        and prereg.get("design", {}).get("hardware") == "one H20-3e per frozen row"
        and prereg.get("design", {}).get("tp_size") == 1
        and prereg.get("data", {}).get("frozen_rows") == 8,
        "preregistration: frozen design authority drift",
    )
    replay_row = prereg.get("code", {}).get("replay")
    require(
        isinstance(replay_row, dict)
        and replay_row.get("path") == "replay_hypic_retained_state_bytes.py"
        and replay_row.get("sha256") == EXPECTED_REPLAY_CODE_SHA256,
        "preregistration: blind replay code identity drift",
    )
    platform_binding = prereg.get("platform_execution_authority", {})
    platform_sha = sha256_file(boot_root / "dynamic-platform-command-authority.json", "outer platform authority hash")
    require(
        platform_binding.get("receipt_path") == f"{EXPECTED_BOOT_ROOT_NFS}/dynamic-platform-command-authority.json"
        and platform_binding.get("receipt_sha256") == platform_sha
        and platform_binding.get("platform_job_id") == JOB_ID
        and platform_binding.get("platform_trial_id") == TRIAL_ID
        and platform_binding.get("platform_command_identity_verified") is True,
        "preregistration: platform authority binding drift",
    )
    asset_binding = prereg.get("model", {}).get("asset_identity", {})
    asset_sha = sha256_file(boot_root / "model-asset-observation.json", "outer model asset hash")
    require(
        asset_binding.get("observation_path") == f"{EXPECTED_BOOT_ROOT_NFS}/model-asset-observation.json"
        and asset_binding.get("observation_sha256") == asset_sha
        and asset_binding.get("observation") == asset
        and asset_binding.get("old_node_inode_device_and_timestamps_are_not_authority") is True,
        "preregistration: model asset authority binding drift",
    )
    require(platform["platform_job_id"] == JOB_ID, "platform/preregistration identity")
    storage_contract = load_json(static / "model-storage-contract.json", "model storage contract")
    require(
        storage_contract.get("schema") == "hypic-rwd5-model-storage-contract-v3"
        and prereg.get("model", {}).get("storage_contract") == storage_contract
        and prereg.get("model", {}).get("storage_contract_sha256") == ledger["static/model-storage-contract.json"],
        "model storage contract binding drift",
    )
    return prereg, prereg_sha


def _is_c_contiguous(shape: list[int], stride: list[int]) -> bool:
    expected = 1
    for size, observed in zip(reversed(shape), reversed(stride)):
        if size != 1 and observed != expected:
            return False
        expected *= size
    return True


def _numel(shape: list[int], label: str) -> int:
    total = 1
    for value in shape:
        require(type(value) is int and value > 0, f"{label}: positive integer tensor shape")
        total *= value
    return total


def derive_range(row: dict[str, Any], label: str) -> tuple[tuple[str, int, int], int, int]:
    expected_keys = {
        "device",
        "storage_data_ptr",
        "storage_nbytes",
        "storage_id",
        "dtype",
        "shape",
        "stride",
        "element_size",
        "storage_offset_elements",
        "tensor_data_ptr",
        "tensor_name",
        "component",
        "byte_start",
        "byte_end",
        "absolute_byte_start",
        "absolute_byte_end",
        "range_bytes",
        "selection",
    }
    require_exact_keys(row, expected_keys, label)
    dtype = row["dtype"]
    require(dtype in DTYPE_BYTES, f"{label}: known dtype required")
    element = DTYPE_BYTES[dtype]
    require(exact_int(row["element_size"], f"{label} element_size") == element, f"{label}: dtype/element-size drift")
    shape = row["shape"]
    stride = row["stride"]
    require(isinstance(shape, list) and isinstance(stride, list) and shape, f"{label}: shape/stride lists")
    require(all(type(value) is int for value in shape + stride), f"{label}: integer shape/stride")
    require(len(shape) == len(stride) and _is_c_contiguous(shape, stride), f"{label}: C-contiguous shape/stride")
    storage_offset = exact_int(row["storage_offset_elements"], f"{label} storage offset")
    storage_base = exact_int(row["storage_data_ptr"], f"{label} storage base")
    storage_nbytes = exact_int(row["storage_nbytes"], f"{label} storage bytes")
    require(storage_offset >= 0 and storage_base > 0 and storage_nbytes > 0, f"{label}: positive storage identity")
    require(
        exact_int(row["tensor_data_ptr"], f"{label} tensor pointer")
        == storage_base + storage_offset * element,
        f"{label}: pointer-relative tensor identity drift",
    )
    device = row["device"]
    require(
        isinstance(device, str) and (device == "cpu" or device.startswith("cuda:")),
        f"{label}: exact CPU/CUDA device identity",
    )
    expected_storage_id = hashlib.sha256(f"{device}:{storage_base}:{storage_nbytes}".encode()).hexdigest()
    require(row["storage_id"] == expected_storage_id, f"{label}: storage ID drift")
    full_end = (storage_offset + _numel(shape, label)) * element
    require(full_end <= storage_nbytes, f"{label}: tensor exceeds storage")
    selection = row["selection"]
    require(isinstance(selection, dict), f"{label}: selection object")
    kind = selection.get("kind")
    if kind == "axis0_slots":
        require_exact_keys(selection, {"kind", "slot_start", "slot_end_exclusive"}, f"{label} selection")
        require(len(shape) == 3, f"{label}: KV rank")
        first = exact_int(selection["slot_start"], f"{label} slot start")
        after = exact_int(selection["slot_end_exclusive"], f"{label} slot end")
        require(0 < first < after <= shape[0], f"{label}: KV slot domain")
        start = (storage_offset + first * stride[0]) * element
        end = (storage_offset + after * stride[0]) * element
    elif kind == "axis1_slots_at_layer":
        require_exact_keys(
            selection,
            {"kind", "mamba_layer_index", "slot_start", "slot_end_exclusive"},
            f"{label} selection",
        )
        require(len(shape) >= 3, f"{label}: recurrent tensor rank")
        layer = exact_int(selection["mamba_layer_index"], f"{label} layer")
        first = exact_int(selection["slot_start"], f"{label} slot start")
        after = exact_int(selection["slot_end_exclusive"], f"{label} slot end")
        require(0 <= layer < shape[0] and 0 < first < after <= shape[1], f"{label}: recurrent slot domain")
        start = (storage_offset + layer * stride[0] + first * stride[1]) * element
        end = (storage_offset + layer * stride[0] + after * stride[1]) * element
    elif kind == "whole_tensor":
        require_exact_keys(selection, {"kind"}, f"{label} selection")
        start = storage_offset * element
        end = full_end
    else:
        raise AcceptanceError(f"{label}: unknown range selection {kind!r}")
    require(0 <= start < end <= storage_nbytes, f"{label}: derived range outside storage")
    require(exact_int(row["byte_start"], f"{label} byte_start") == start, f"{label}: byte_start drift")
    require(exact_int(row["byte_end"], f"{label} byte_end") == end, f"{label}: byte_end drift")
    require(exact_int(row["range_bytes"], f"{label} range_bytes") == end - start, f"{label}: range size drift")
    require(exact_int(row["absolute_byte_start"], f"{label} absolute start") == storage_base + start, f"{label}: absolute start drift")
    require(exact_int(row["absolute_byte_end"], f"{label} absolute end") == storage_base + end, f"{label}: absolute end drift")
    require(isinstance(row["tensor_name"], str) and row["tensor_name"], f"{label}: tensor name")
    require(isinstance(row["component"], str) and row["component"], f"{label}: component")
    return (device, storage_base, storage_nbytes), start, end


def replay_union(records: Any, label: str) -> dict[str, Any]:
    require(isinstance(records, list) and records, f"{label}: nonempty range records")
    grouped: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    tensor_metadata: dict[str, tuple[Any, ...]] = {}
    naive = 0
    for index, row in enumerate(records):
        require(isinstance(row, dict), f"{label}[{index}]: object required")
        key, start, end = derive_range(row, f"{label}[{index}]")
        tensor_name = row["tensor_name"]
        signature = (
            row["device"],
            row["storage_data_ptr"],
            row["storage_nbytes"],
            row["storage_id"],
            row["dtype"],
            tuple(row["shape"]),
            tuple(row["stride"]),
            row["element_size"],
            row["storage_offset_elements"],
            row["tensor_data_ptr"],
        )
        if tensor_name in tensor_metadata:
            require(tensor_metadata[tensor_name] == signature, f"{label}: tensor metadata instability")
        tensor_metadata[tensor_name] = signature
        naive += end - start
        grouped.setdefault(key, []).append((start, end))
    total = 0
    storage_rows = []
    for key, intervals in sorted(grouped.items()):
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            else:
                merged[-1][1] = max(merged[-1][1], end)
        unique = sum(end - start for start, end in merged)
        total += unique
        storage_rows.append(
            {
                "device": key[0],
                "storage_data_ptr": key[1],
                "storage_nbytes": key[2],
                "merged_byte_ranges": merged,
                "unique_bytes": unique,
            }
        )
    return {
        "record_count": len(records),
        "storage_count": len(storage_rows),
        "naive_range_bytes": naive,
        "unique_overlap_aware_bytes": total,
        "storages": storage_rows,
    }


def validate_mamba_multiset(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict), f"{label}: object required")
    require_exact_keys(
        value,
        {
            "size",
            "raw_free_slots",
            "raw_count",
            "unique_count",
            "duplicates",
            "duplicate_excess_count",
            "canonical_free_domain",
            "canonical_allocated_domain",
            "consistency_status",
            "physical_ownership_basis",
            "global_allocator_correctness_claimed",
        },
        label,
    )
    size = exact_int(value["size"], f"{label} size")
    raw = value["raw_free_slots"]
    require(size > 0 and isinstance(raw, list) and all(type(slot) is int for slot in raw), f"{label}: raw domain")
    require(all(1 <= slot <= size for slot in raw), f"{label}: out-of-domain slot")
    counts = Counter(raw)
    positions: dict[int, list[int]] = {}
    for index, slot in enumerate(raw):
        positions.setdefault(slot, []).append(index)
    duplicates = [
        {"slot": slot, "count": counts[slot], "positions": positions[slot]}
        for slot in sorted(counts)
        if counts[slot] > 1
    ]
    canonical_free = sorted(counts)
    canonical_allocated = sorted(set(range(1, size + 1)) - set(canonical_free))
    excess = sum(row["count"] - 1 for row in duplicates)
    expected_status = (
        "anomalous_duplicate_free_multiset_physical_ownership_closed"
        if excess
        else "exact_unique_free_domain"
    )
    require(
        value["raw_count"] == len(raw)
        and value["unique_count"] == len(canonical_free)
        and value["duplicates"] == duplicates
        and value["duplicate_excess_count"] == excess
        and value["canonical_free_domain"] == canonical_free
        and value["canonical_allocated_domain"] == canonical_allocated
        and value["consistency_status"] == expected_status
        and value["physical_ownership_basis"] == MAMBA_BASIS
        and value["global_allocator_correctness_claimed"] is False,
        f"{label}: independently replayed multiset drift",
    )
    return value


def token_sha256(values: list[int]) -> str:
    require(all(type(value) is int and -(2**31) <= value < 2**31 for value in values), "token IDs: signed int32 domain")
    return hashlib.sha256(b"".join(struct.pack("<i", value) for value in values)).hexdigest()


def contiguous_runs(slots: list[int], label: str) -> list[tuple[int, int]]:
    require(
        isinstance(slots, list)
        and slots
        and all(type(slot) is int and slot > 0 for slot in slots),
        f"{label}: nonempty positive exact-integer slots required",
    )
    unique = sorted(set(slots))
    require(len(unique) == len(slots), f"{label}: duplicate slot")
    runs: list[tuple[int, int]] = []
    start = previous = unique[0]
    for value in unique[1:]:
        if value != previous + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def validate_selection_complete(
    selection: dict[str, Any],
    target: dict[str, Any],
    prereg: dict[str, Any],
    mode: str,
    snapshot: str,
) -> tuple[list[int], list[int], int, int]:
    require_exact_keys(
        selection,
        {
            "cache_kind",
            "entries",
            "owned_document_token_ids",
            "owned_document_token_sha256",
            "owned_document_tokens",
            "expected_measured_cached_tokens",
            "full_kv_slots",
            "mamba_state_slots",
            "metadata_exact_non_tensor_bytes",
        },
        f"selection {snapshot}",
    )
    owned = selection["owned_document_token_ids"]
    full_slots = selection["full_kv_slots"]
    mamba_slots = selection["mamba_state_slots"]
    require(
        isinstance(owned, list)
        and owned
        and token_sha256(owned) == selection["owned_document_token_sha256"]
        and exact_int(selection["owned_document_tokens"], f"selection owned count {snapshot}")
        == len(owned),
        f"selection {snapshot}: owned token hash/count closure failed",
    )
    require(
        isinstance(full_slots, list)
        and all(type(slot) is int and slot > 0 for slot in full_slots)
        and len(full_slots) == len(set(full_slots)) == len(owned),
        f"selection {snapshot}: exact KV ownership coverage failed",
    )
    require(
        isinstance(mamba_slots, list)
        and bool(mamba_slots)
        and all(type(slot) is int and slot > 0 for slot in mamba_slots)
        and len(mamba_slots) == len(set(mamba_slots)),
        f"selection {snapshot}: exact Mamba ownership coverage failed",
    )
    entries = selection["entries"]
    require(isinstance(entries, list) and entries, f"selection {snapshot}: nonempty entries required")
    entry_tokens: list[int] = []
    entry_kv: list[int] = []
    entry_mamba: list[int] = []
    metadata_non_tensor_bytes = 0
    document = target["document_token_ids"]

    if mode == "prefix_cache":
        expected_entry_keys = {
            "node_id",
            "token_ids",
            "token_sha256",
            "token_count",
            "full_kv_slots",
            "mamba_state_slots",
            "lock_refs",
            "exact_key_array_bytes",
            "python_node_shallow_bytes",
        }
        node_ids: list[int] = []
        for position, entry in enumerate(entries):
            require(isinstance(entry, dict), f"selection {snapshot}: Prefix entry {position} object")
            require_exact_keys(entry, expected_entry_keys, f"selection {snapshot} Prefix entry {position}")
            node_id = exact_int(entry["node_id"], f"selection {snapshot} Prefix node")
            require(node_id > 0, f"selection {snapshot}: positive Prefix node ID")
            node_ids.append(node_id)
            tokens = entry["token_ids"]
            slots = entry["full_kv_slots"]
            recurrent = entry["mamba_state_slots"]
            require(
                isinstance(tokens, list)
                and tokens
                and token_sha256(tokens) == entry["token_sha256"]
                and exact_int(entry["token_count"], f"selection {snapshot} Prefix token count")
                == len(tokens),
                f"selection {snapshot}: Prefix entry token closure failed",
            )
            require(
                isinstance(slots, list)
                and all(type(slot) is int and slot > 0 for slot in slots)
                and len(slots) == len(tokens),
                f"selection {snapshot}: Prefix entry KV coverage failed",
            )
            require(
                isinstance(recurrent, list)
                and all(type(slot) is int and slot > 0 for slot in recurrent),
                f"selection {snapshot}: Prefix entry Mamba slots failed",
            )
            exact_key_bytes = exact_int(
                entry["exact_key_array_bytes"],
                f"selection {snapshot} Prefix exact key bytes",
            )
            require(
                entry["lock_refs"] == {"full": 0, "mamba": 0}
                and exact_key_bytes == 8 * len(tokens)
                and exact_int(
                    entry["python_node_shallow_bytes"],
                    f"selection {snapshot} Prefix shallow bytes",
                )
                > 0,
                f"selection {snapshot}: Prefix lock/metadata closure failed",
            )
            metadata_non_tensor_bytes += exact_key_bytes
            entry_tokens.extend(tokens)
            entry_kv.extend(slots)
            entry_mamba.extend(recurrent)
        require(len(node_ids) == len(set(node_ids)), f"selection {snapshot}: duplicate Prefix node ID")
        require(selection["cache_kind"] == "MambaRadixCache", f"selection {snapshot}: Prefix cache kind")
        require(target["seam_tokens"] == 0, f"selection {snapshot}: Prefix seam")
        require(entry_tokens == owned == document[: len(owned)], f"selection {snapshot}: Prefix target binding")
        independently_cached = len(owned)
    else:
        expected_entry_keys = {
            "segment_index",
            "segment_hash_hex",
            "token_ids",
            "token_sha256",
            "token_count",
            "full_kv_slots",
            "mamba_state_slot",
            "lock_ref",
            "exact_segment_hash_bytes",
            "python_entry_shallow_bytes",
        }
        segments = target.get("segment_token_ids")
        require(
            isinstance(segments, list)
            and len(segments) == 2
            and len(entries) == 2,
            f"selection {snapshot}: exact two HYPIC segments/entries required",
        )
        require(
            [entry.get("segment_index") for entry in entries] == [0, 1],
            f"selection {snapshot}: ordered HYPIC segment indices",
        )
        for position, entry in enumerate(entries):
            require(isinstance(entry, dict), f"selection {snapshot}: HYPIC entry {position} object")
            require_exact_keys(entry, expected_entry_keys, f"selection {snapshot} HYPIC entry {position}")
            require(
                exact_int(entry["segment_index"], f"selection {snapshot} segment index")
                == position,
                f"selection {snapshot}: HYPIC segment index drift",
            )
            tokens = entry["token_ids"]
            slots = entry["full_kv_slots"]
            require(
                isinstance(tokens, list)
                and tokens
                and tokens == segments[position]
                and token_sha256(tokens) == entry["token_sha256"]
                and exact_int(entry["token_count"], f"selection {snapshot} HYPIC token count")
                == len(tokens),
                f"selection {snapshot}: HYPIC entry token/target closure failed",
            )
            require(
                isinstance(slots, list)
                and all(type(slot) is int and slot > 0 for slot in slots)
                and len(slots) == len(tokens),
                f"selection {snapshot}: HYPIC entry KV coverage failed",
            )
            segment_hash = hashlib.sha256(
                b"".join(struct.pack("<i", token) for token in tokens)
            ).digest()[:16].hex()
            mamba_slot = exact_int(
                entry["mamba_state_slot"],
                f"selection {snapshot} HYPIC Mamba slot",
            )
            require(
                entry["segment_hash_hex"] == segment_hash
                and mamba_slot > 0
                and exact_int(entry["lock_ref"], f"selection {snapshot} HYPIC lock ref") == 0
                and exact_int(
                    entry["exact_segment_hash_bytes"],
                    f"selection {snapshot} HYPIC segment hash bytes",
                )
                == 16
                and exact_int(
                    entry["python_entry_shallow_bytes"],
                    f"selection {snapshot} HYPIC shallow bytes",
                )
                > 0,
                f"selection {snapshot}: HYPIC hash/slot/lock/metadata closure failed",
            )
            metadata_non_tensor_bytes += 16
            entry_tokens.extend(tokens)
            entry_kv.extend(slots)
            entry_mamba.append(mamba_slot)
        seam = exact_int(
            prereg["design"]["hypic_seam_tokens"],
            f"selection {snapshot} preregistered HYPIC seam",
        )
        require(
            seam == 8
            and 0 <= seam < len(document)
            and target["seam_tokens"] == seam,
            f"selection {snapshot}: preregistered HYPIC seam drift",
        )
        require(selection["cache_kind"] == "PICache", f"selection {snapshot}: HYPIC cache kind")
        require(entry_tokens == owned == document, f"selection {snapshot}: HYPIC document binding")
        independently_cached = len(document) - seam

    require(
        entry_kv == full_slots and entry_mamba == mamba_slots,
        f"selection {snapshot}: entry/top-level slot reconstruction failed",
    )
    require(
        exact_int(
            selection["expected_measured_cached_tokens"],
            f"selection {snapshot} expected cached tokens",
        )
        == independently_cached,
        f"selection {snapshot}: independently derived cached-token drift",
    )
    require(
        exact_int(
            selection["metadata_exact_non_tensor_bytes"],
            f"selection {snapshot} metadata non-tensor bytes",
        )
        == metadata_non_tensor_bytes,
        f"selection {snapshot}: metadata non-tensor reconstruction failed",
    )
    return full_slots, mamba_slots, independently_cached, metadata_non_tensor_bytes


def validate_payload_structure(
    receipt: dict[str, Any],
    prereg: dict[str, Any],
    mode: str,
    full_slots: list[int],
    mamba_slots: list[int],
    snapshot: str,
) -> None:
    contract = prereg["model"]["storage_contract"]
    require_exact_keys(
        contract,
        {
            "schema",
            "config_sha256",
            "model_type",
            "num_hidden_layers",
            "full_attention_layer_ids",
            "recurrent_layer_ids",
            "full_attention_layer_count",
            "recurrent_layer_count",
            "conv_tensor_count",
            "temporal_tensor_count",
            "kv_dtype",
            "mamba_component_dtypes",
            "dtype_authority",
            "kv_layout",
            "kv_slot_axis",
            "mamba_layer_axis",
            "mamba_slot_axis",
            "page_size",
            "enable_int8_mamba_checkpoint",
            "mode_components",
        },
        f"storage contract {snapshot}",
    )
    component_dtypes = contract["mamba_component_dtypes"]
    require(
        receipt["storage_contract"] == contract
        and contract["schema"] == "hypic-rwd5-model-storage-contract-v3"
        and "dtype" not in contract
        and contract["kv_layout"] == "nhd"
        and contract["kv_slot_axis"] == 0
        and contract["mamba_layer_axis"] == 0
        and contract["mamba_slot_axis"] == 1
        and contract["page_size"] == 1
        and contract["enable_int8_mamba_checkpoint"] is False
        and contract["kv_dtype"] in DTYPE_BYTES
        and isinstance(component_dtypes, dict)
        and set(component_dtypes) == {"conv", "temporal", "transition", "conv_tails"}
        and all(dtype in DTYPE_BYTES for dtype in component_dtypes.values())
        and component_dtypes["conv_tails"] == component_dtypes["conv"]
        and component_dtypes["transition"] == component_dtypes["temporal"],
        f"storage contract {snapshot}: topology/dtype authority drift",
    )
    mode_components = contract["mode_components"]
    require(
        isinstance(mode_components, dict)
        and set(mode_components) == set(MODES)
        and mode_components["prefix_cache"]
        == {"transition_tensor_count": 0, "conv_tails_tensor_count": 0}
        and mode_components["transition_rope_recompute"]
        == {"transition_tensor_count": 1, "conv_tails_tensor_count": 1},
        f"storage contract {snapshot}: exact mode-component topology drift",
    )
    full_count = exact_int(
        contract["full_attention_layer_count"],
        f"storage contract {snapshot} full layer count",
    )
    recurrent_layers = exact_int(
        contract["recurrent_layer_count"],
        f"storage contract {snapshot} recurrent layer count",
    )
    conv_count = exact_int(
        contract["conv_tensor_count"],
        f"storage contract {snapshot} conv count",
    )
    require(
        full_count > 0
        and recurrent_layers > 0
        and conv_count > 0
        and contract["temporal_tensor_count"] == 1
        and len(contract["full_attention_layer_ids"]) == full_count
        and len(contract["recurrent_layer_ids"]) == recurrent_layers
        and full_count + recurrent_layers == contract["num_hidden_layers"],
        f"storage contract {snapshot}: layer cardinality drift",
    )

    allocator = receipt["allocator_observation"]
    require_exact_keys(
        allocator,
        {
            "kv_available_tokens",
            "kv_capacity_tokens",
            "kv_page_size",
            "mamba_available_slots",
            "mamba_capacity_slots",
            "pre_free_ownership",
        },
        f"allocator observation {snapshot}",
    )
    pre = allocator["pre_free_ownership"]
    require(isinstance(pre, dict), f"allocator observation {snapshot}: pre-free object")
    require_exact_keys(pre, {"kv", "mamba"}, f"pre-free ownership {snapshot}")
    kv_capacity = exact_int(allocator["kv_capacity_tokens"], f"KV capacity {snapshot}")
    mamba_capacity = exact_int(allocator["mamba_capacity_slots"], f"Mamba capacity {snapshot}")
    require(
        kv_capacity > 0
        and mamba_capacity > 0
        and allocator["kv_page_size"] == contract["page_size"] == 1,
        f"allocator observation {snapshot}: capacity/page-size drift",
    )
    kv_pre = pre["kv"]
    require(isinstance(kv_pre, dict), f"pre KV {snapshot}: object")
    require_exact_keys(
        kv_pre,
        {
            "page_size",
            "size",
            "free_pages",
            "release_pages",
            "canonical_free_domain",
            "canonical_allocated_domain",
        },
        f"pre KV {snapshot}",
    )
    kv_free = kv_pre["free_pages"]
    kv_release = kv_pre["release_pages"]
    require(
        kv_pre["page_size"] == 1
        and kv_pre["size"] == kv_capacity
        and isinstance(kv_free, list)
        and isinstance(kv_release, list)
        and all(type(slot) is int for slot in kv_free + kv_release)
        and kv_free == sorted(kv_free)
        and kv_release == sorted(kv_release),
        f"pre KV {snapshot}: canonical list/domain type drift",
    )
    kv_combined = kv_free + kv_release
    kv_domain = set(range(1, kv_capacity + 1))
    require(
        len(kv_combined) == len(set(kv_combined))
        and set(kv_combined).issubset(kv_domain)
        and kv_pre["canonical_free_domain"] == sorted(kv_combined)
        and kv_pre["canonical_allocated_domain"]
        == sorted(kv_domain - set(kv_combined))
        and allocator["kv_available_tokens"] == len(kv_combined)
        and set(full_slots).issubset(kv_domain - set(kv_combined)),
        f"pre KV {snapshot}: independently replayed ownership drift",
    )
    pre_mamba = validate_mamba_multiset(pre["mamba"], f"pre Mamba {snapshot}")
    require(
        pre_mamba["size"] == mamba_capacity
        and allocator["mamba_available_slots"] == pre_mamba["raw_count"]
        and set(mamba_slots).isdisjoint(pre_mamba["canonical_free_domain"]),
        f"pre Mamba {snapshot}: independently replayed ownership drift",
    )
    require(
        pre_mamba["duplicates"] == []
        and pre_mamba["duplicate_excess_count"] == 0
        and pre_mamba["raw_count"] == pre_mamba["unique_count"]
        and pre_mamba["consistency_status"] == "exact_unique_free_domain",
        f"pre Mamba {snapshot}: r34 clean-start exact free domain required",
    )
    if mode == "transition_rope_recompute":
        require(
            sorted(mamba_slots) == pre_mamba["canonical_allocated_domain"],
            f"pre Mamba {snapshot}: HYPIC exact target-entry physical ownership closure",
        )

    records = receipt["tensor_payload"]["records"]
    require(isinstance(records, list) and records, f"tensor payload {snapshot}: nonempty records")
    require(
        all(row.get("device") == "cuda:0" for row in records),
        f"tensor payload {snapshot}: every retained-state tensor must be cuda:0",
    )
    by_name: dict[str, list[dict[str, Any]]] = {}
    for index, row in enumerate(records):
        require(isinstance(row, dict), f"tensor payload {snapshot} record {index}: object")
        name = row.get("tensor_name")
        require(isinstance(name, str) and name, f"tensor payload {snapshot} record {index}: tensor name")
        by_name.setdefault(name, []).append(row)
    expected_k = [f"full_kv.key[{index}]" for index in range(full_count)]
    expected_v = [f"full_kv.value[{index}]" for index in range(full_count)]
    expected_mamba = [f"mamba.conv[{index}]" for index in range(conv_count)] + ["mamba.temporal"]
    if mode == "transition_rope_recompute":
        expected_mamba += ["mamba.transition"] + [
            f"mamba.conv_tails[{index}]"
            for index in range(mode_components[mode]["conv_tails_tensor_count"])
        ]
    require(
        set(by_name) == set(expected_k + expected_v + expected_mamba),
        f"tensor payload {snapshot}: exact tensor key set drift",
    )
    kv_expected_selections = [
        {"kind": "axis0_slots", "slot_start": first, "slot_end_exclusive": after}
        for first, after in contiguous_runs(full_slots, f"KV slots {snapshot}")
    ]
    kv_shapes: dict[int, list[int]] = {}
    for name in expected_k + expected_v:
        rows = by_name[name]
        expected_component = (
            "full_attention_key" if name.startswith("full_kv.key") else "full_attention_value"
        )
        require(
            [row.get("selection") for row in rows] == kv_expected_selections
            and all(row.get("component") == expected_component for row in rows),
            f"tensor payload {snapshot}: exact KV run/component coverage failed for {name}",
        )
        shape = rows[0].get("shape")
        require(
            isinstance(shape, list)
            and len(shape) == 3
            and shape[0] == kv_capacity + 1
            and rows[0].get("dtype") == contract["kv_dtype"]
            and rows[0].get("element_size") == DTYPE_BYTES[contract["kv_dtype"]],
            f"tensor payload {snapshot}: KV axis/dtype drift for {name}",
        )
        layer = int(name.rsplit("[", 1)[1][:-1])
        if name.startswith("full_kv.key"):
            kv_shapes[layer] = shape
        else:
            require(shape == kv_shapes[layer], f"tensor payload {snapshot}: K/V shape drift at layer {layer}")

    recurrent_expected_selections = [
        {
            "kind": "axis1_slots_at_layer",
            "mamba_layer_index": layer,
            "slot_start": first,
            "slot_end_exclusive": after,
        }
        for layer in range(recurrent_layers)
        for first, after in contiguous_runs(mamba_slots, f"Mamba slots {snapshot}")
    ]
    presence = receipt["component_presence"]
    require(isinstance(presence, dict), f"component presence {snapshot}: object")
    require(
        set(presence) == {name.removeprefix("mamba.") for name in expected_mamba},
        f"component presence {snapshot}: exact key set drift",
    )
    for name in expected_mamba:
        rows = by_name[name]
        short_name = name.removeprefix("mamba.")
        component = (
            "conv_tails"
            if short_name.startswith("conv_tails[")
            else "conv" if short_name.startswith("conv[") else short_name
        )
        expected_dtype = component_dtypes[component]
        shape = rows[0].get("shape")
        require(
            [row.get("selection") for row in rows] == recurrent_expected_selections
            and all(row.get("component") == component for row in rows),
            f"tensor payload {snapshot}: exact recurrent run/component coverage failed for {name}",
        )
        require(
            isinstance(shape, list)
            and len(shape) >= 3
            and shape[0] == recurrent_layers
            and shape[1] == mamba_capacity + 1
            and rows[0].get("dtype") == expected_dtype
            and rows[0].get("element_size") == DTYPE_BYTES[expected_dtype],
            f"tensor payload {snapshot}: recurrent axes/dtype drift for {name}",
        )
        expected_presence = {
            "present": True,
            "shape": shape,
            "dtype": expected_dtype,
            "element_size": DTYPE_BYTES[expected_dtype],
        }
        require(
            presence[short_name] == expected_presence,
            f"component presence {snapshot}: shape/dtype drift for {short_name}",
        )


def validate_metadata_structure(
    metadata: dict[str, Any],
    selection: dict[str, Any],
    mode: str,
    metadata_non_tensor_bytes: int,
    snapshot: str,
) -> None:
    expected: dict[str, int] = {}
    if mode == "prefix_cache":
        for entry in selection["entries"]:
            node = entry["node_id"]
            expected[f"prefix.node[{node}].full_kv_slots"] = len(entry["full_kv_slots"])
            if entry["mamba_state_slots"]:
                expected[f"prefix.node[{node}].mamba_slots"] = len(entry["mamba_state_slots"])
    else:
        for entry in selection["entries"]:
            index = entry["segment_index"]
            expected[f"hypic.segment[{index}].full_kv_slots"] = len(entry["full_kv_slots"])
            expected[f"hypic.segment[{index}].token_ids"] = len(entry["token_ids"])
    rows = metadata["tensor_records"]
    require(
        isinstance(rows, list) and len(rows) == len(expected),
        f"metadata {snapshot}: exact record count drift",
    )
    by_name: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        require(isinstance(row, dict), f"metadata {snapshot} record {index}: object")
        name = row.get("tensor_name")
        require(isinstance(name, str) and name not in by_name, f"metadata {snapshot}: duplicate/invalid tensor name")
        by_name[name] = row
    require(set(by_name) == set(expected), f"metadata {snapshot}: exact tensor key set drift")
    for name, count in expected.items():
        row = by_name[name]
        expected_device = "cpu" if name.endswith(".token_ids") else "cuda:0"
        require(
            row.get("component") == "cache_index_metadata"
            and row.get("device") == expected_device
            and row.get("selection") == {"kind": "whole_tensor"}
            and row.get("dtype") == "torch.int64"
            and row.get("element_size") == 8
            and row.get("shape") == [count]
            and row.get("stride") == [1],
            f"metadata {snapshot}: dtype/shape/selection drift for {name}",
        )
    require(
        metadata["exact_non_tensor_bytes"] == metadata_non_tensor_bytes
        and selection["metadata_exact_non_tensor_bytes"] == metadata_non_tensor_bytes
        and metadata["excluded_from_store_mib"] is True
        and metadata["python_allocator_overhead"]
        == "not attributed; shallow object sizes retained per entry",
        f"metadata {snapshot}: exact exclusion/non-tensor attribution drift",
    )


def validate_terminal(
    terminal: dict[str, Any],
    receipt: dict[str, Any],
    snapshot: str,
    receipt_sha: str,
    mode: str,
) -> tuple[str, int, int, int]:
    require_exact_keys(
        terminal,
        {"schema", "status", "official_commit", "snapshot_id", "passed", "checks", "authority", "prior_receipt_sha256"},
        f"terminal {snapshot}",
    )
    require(
        terminal["schema"] == "forkaudit-hypic-retained-state-terminal-v2"
        and terminal["status"] == "terminal_ownership_removal_complete"
        and terminal["official_commit"] == OFFICIAL_COMMIT
        and terminal["snapshot_id"] == snapshot
        and terminal["passed"] is True
        and terminal["prior_receipt_sha256"] == receipt_sha
        and terminal["authority"] == receipt["authority"],
        f"terminal {snapshot}: identity/binding drift",
    )
    checks = terminal["checks"]
    require_exact_keys(
        checks,
        {
            "target_entries_after",
            "all_cache_entries_after",
            "old_kv_slots_all_free",
            "old_mamba_slots_all_free",
            "old_kv_slots_preallocated",
            "old_mamba_slots_preallocated",
            "kv_available_tokens",
            "kv_capacity_tokens",
            "mamba_available_slots",
            "mamba_capacity_slots",
            "mamba_unique_physical_domain_closed",
            "mamba_duplicate_anomaly_preserved_without_migration_or_growth",
            "mamba_global_allocator_correctness_claimed",
            "store_metric_scope",
            "kv_free_list",
            "mamba_free_list",
        },
        f"terminal checks {snapshot}",
    )
    require(
        checks["target_entries_after"] == 0
        and checks["all_cache_entries_after"] == 0
        and checks["old_kv_slots_all_free"] is True
        and checks["old_mamba_slots_all_free"] is True
        and checks["old_kv_slots_preallocated"] is True
        and checks["old_mamba_slots_preallocated"] is True
        and checks["kv_available_tokens"] == checks["kv_capacity_tokens"]
        and checks["mamba_unique_physical_domain_closed"] is True
        and checks["mamba_duplicate_anomaly_preserved_without_migration_or_growth"] is True
        and checks["mamba_global_allocator_correctness_claimed"] is False
        and checks["store_metric_scope"] == STORE_SCOPE,
        f"terminal checks {snapshot}: closure failed",
    )
    kv = checks["kv_free_list"]
    require_exact_keys(kv, {"page_size", "size", "free_pages", "release_pages", "exact_domain"}, f"terminal KV {snapshot}")
    kv_size = exact_int(kv["size"], f"terminal KV size {snapshot}")
    expected_kv = list(range(1, kv_size + 1))
    require(
        kv["page_size"] == 1
        and kv["release_pages"] == []
        and isinstance(kv["free_pages"], list)
        and all(type(slot) is int for slot in kv["free_pages"])
        and len(kv["free_pages"]) == len(set(kv["free_pages"]))
        and sorted(kv["free_pages"]) == expected_kv
        and kv["exact_domain"] == expected_kv
        and checks["kv_capacity_tokens"] == kv_size,
        f"terminal KV {snapshot}: exact free domain failed",
    )
    mamba = validate_mamba_multiset(checks["mamba_free_list"], f"terminal Mamba {snapshot}")
    require(
        mamba["canonical_free_domain"] == list(range(1, mamba["size"] + 1))
        and mamba["canonical_allocated_domain"] == []
        and mamba["duplicates"] == []
        and mamba["duplicate_excess_count"] == 0
        and mamba["raw_count"] == mamba["unique_count"] == mamba["size"]
        and mamba["consistency_status"] == "exact_unique_free_domain"
        and checks["mamba_available_slots"] == mamba["raw_count"]
        and checks["mamba_capacity_slots"] == mamba["size"],
        f"terminal Mamba {snapshot}: r34 exact unique free domain not closed",
    )
    selection = receipt["selection"]
    require(all(slot in kv["free_pages"] for slot in selection["full_kv_slots"]), f"terminal {snapshot}: old KV not returned")
    require(all(slot in mamba["canonical_free_domain"] for slot in selection["mamba_state_slots"]), f"terminal {snapshot}: old Mamba not returned")
    pre = receipt["allocator_observation"]["pre_free_ownership"]
    pre_mamba = validate_mamba_multiset(pre["mamba"], f"pre Mamba {snapshot}")
    require(
        mamba["duplicates"] == pre_mamba["duplicates"]
        and mamba["duplicate_excess_count"] == pre_mamba["duplicate_excess_count"],
        f"terminal {snapshot}: Mamba anomaly migration/growth",
    )
    require(
        pre_mamba["duplicates"] == []
        and pre_mamba["duplicate_excess_count"] == 0
        and pre_mamba["consistency_status"] == "exact_unique_free_domain",
        f"terminal {snapshot}: r34 pre-snapshot allocator anomaly forbidden",
    )
    pre_kv = pre["kv"]
    require(
        isinstance(pre_kv, dict)
        and set(selection["full_kv_slots"]).isdisjoint(pre_kv["canonical_free_domain"])
        and set(selection["mamba_state_slots"]).isdisjoint(pre_mamba["canonical_free_domain"]),
        f"terminal {snapshot}: selected slots were not preallocated",
    )
    return (
        pre_mamba["consistency_status"],
        pre_mamba["raw_count"],
        pre_mamba["unique_count"],
        pre_mamba["duplicate_excess_count"],
    )


def validate_blind_replay(formal: Path) -> dict[str, Any]:
    blind = load_json(formal / "blind-replay.json", "blind replay")
    require_exact_keys(
        blind,
        {"schema", "passed", "official_commit", "freeze_manifest_sha256", "denominator", "claim_boundary", "global_allocator_correctness_claimed", "rows", "modes"},
        "blind replay",
    )
    require(
        blind["schema"] == "forkaudit-hypic-retained-state-blind-replay-v2"
        and blind["passed"] is True
        and blind["official_commit"] == OFFICIAL_COMMIT
        and blind["freeze_manifest_sha256"] == EXPECTED_RUNTIME_MANIFEST_SHA256
        and blind["denominator"] == BLIND_DENOMINATOR
        and blind["claim_boundary"] == BLIND_CLAIM_BOUNDARY
        and blind["global_allocator_correctness_claimed"] is False
        and isinstance(blind["rows"], list)
        and len(blind["rows"]) == 16
        and isinstance(blind["modes"], dict)
        and set(blind["modes"]) == set(MODES),
        "blind replay: frozen terminal identity drift",
    )
    row_keys = {
        "mode",
        "rank",
        "snapshot_id",
        "workload_id",
        "owned_document_tokens",
        "measured_cached_tokens",
        "payload_bytes",
        "payload_mib",
        "metadata_tensor_bytes",
        "metadata_non_tensor_bytes",
        "mamba_allocator_consistency_status",
        "mamba_allocator_raw_free_count",
        "mamba_allocator_unique_free_count",
        "mamba_allocator_duplicate_excess_count",
        "metric_validity",
        "global_allocator_correctness_claimed",
        "receipt_sha256",
        "terminal_sha256",
        "raw_sha256",
    }
    for index, row in enumerate(blind["rows"]):
        require(isinstance(row, dict), f"blind row {index}: object")
        require_exact_keys(row, row_keys, f"blind row {index}")
    return blind


def validate_cell(formal: Path, mode: str, rank: int, blind_row: dict[str, Any], prereg: dict[str, Any], prereg_sha: str, platform: dict[str, Any], ledger: dict[str, str]) -> dict[str, Any]:
    dataset, source_index = EXPECTED_PAIRS[rank]
    workload_id = f"{dataset}-{source_index}"
    snapshot = f"{mode}-rank-{rank}"
    rel = {
        "raw": f"raw/{snapshot}.json",
        "target": f"targets/{snapshot}.json",
        "receipt": f"store-receipts/{snapshot}.json",
        "terminal": f"store-receipts/{snapshot}.terminal.json",
        "server": f"server-receipts/{snapshot}.json",
        "readiness": f"server-receipts/{snapshot}.readiness.json",
        "worker": f"scheduler-workers/{snapshot}.json",
    }
    values = {name: load_json(formal / path, f"{name} {snapshot}") for name, path in rel.items()}
    digests = {name: ledger[path] for name, path in rel.items()}
    raw = values["raw"]
    target = values["target"]
    receipt = values["receipt"]
    terminal = values["terminal"]
    server = values["server"]
    readiness = values["readiness"]
    worker = values["worker"]

    target_keys = {
        "schema",
        "snapshot_id",
        "official_commit",
        "mode",
        "rank",
        "workload_id",
        "document_token_ids",
        "document_token_sha256",
        "seam_tokens",
        "authority",
        "workload_binding",
    }
    if mode == "transition_rope_recompute":
        target_keys.add("segment_token_ids")
    require_exact_keys(target, target_keys, f"target {snapshot}")
    require(
        target["schema"] == "forkaudit-hypic-retained-state-target-v2"
        and target["snapshot_id"] == snapshot
        and target["official_commit"] == OFFICIAL_COMMIT
        and target["mode"] == mode
        and target["rank"] == rank
        and target["workload_id"] == workload_id
        and target["workload_binding"]["dataset"] == dataset
        and target["workload_binding"]["source_index"] == source_index
        and target["workload_binding"]["workload_id"] == workload_id,
        f"target {snapshot}: frozen cell identity drift",
    )
    document = target["document_token_ids"]
    require(isinstance(document, list) and document and token_sha256(document) == target["document_token_sha256"], f"target {snapshot}: document token hash")
    if mode == "transition_rope_recompute":
        segments = target["segment_token_ids"]
        require(isinstance(segments, list) and len(segments) == 2 and segments[0] + segments[1] == document and target["seam_tokens"] == 8, f"target {snapshot}: HYPIC segment/seam contract")
    else:
        require(target["seam_tokens"] == 0, f"target {snapshot}: Prefix seam contract")

    require_exact_keys(
        worker,
        {"schema", "official_commit", "mode", "rank", "frontend_pid", "process", "tree_cache_class", "picache_mamba_pool_identity", "int8_mamba_checkpoint_enabled", "preregistration_sha256", "freeze_manifest_sha256", "storage_contract_sha256"},
        f"worker {snapshot}",
    )
    require(
        worker["schema"] == "forkaudit-hypic-scheduler-worker-v2"
        and worker["official_commit"] == OFFICIAL_COMMIT
        and worker["mode"] == mode
        and worker["rank"] == rank
        and worker["preregistration_sha256"] == prereg_sha
        and worker["freeze_manifest_sha256"] == EXPECTED_RUNTIME_MANIFEST_SHA256
        and worker["int8_mamba_checkpoint_enabled"] is False
        and worker["tree_cache_class"] == ("MambaRadixCache" if mode == "prefix_cache" else "PICache")
        and worker["picache_mamba_pool_identity"] == (None if mode == "prefix_cache" else True),
        f"worker {snapshot}: frozen worker identity drift",
    )

    readiness_keys = {"schema", "status", "endpoint", "mode", "rank", "server_pid", "total_timeout_seconds", "single_timeout_seconds", "poll_interval_seconds", "attempt_count", "elapsed_seconds", "attempts", "server_info_sha256"}
    require_exact_keys(readiness, readiness_keys, f"readiness {snapshot}")
    expected_base = f"http://127.0.0.1:{33400 + rank}"
    require(
        readiness["schema"] == "hypic-rwd5-server-info-readiness-v1"
        and readiness["status"] == "ready"
        and readiness["mode"] == mode
        and readiness["rank"] == rank
        and readiness["endpoint"] == f"{expected_base}/server_info"
        and readiness["server_pid"] == worker["frontend_pid"]
        and exact_int(readiness["attempt_count"], f"readiness attempts {snapshot}") >= 1
        and isinstance(readiness["attempts"], list)
        and len(readiness["attempts"]) == readiness["attempt_count"]
        and readiness["attempts"][-1].get("outcome") == "ready"
        and readiness["attempts"][-1].get("response_sha256") == readiness["server_info_sha256"],
        f"readiness {snapshot}: terminal success drift",
    )

    server_keys = {"schema", "official_commit", "official_worktree_clean", "instrumentation_only_overlay", "mode", "rank", "tp_size", "data_sha256", "model_weight_ledger_sha256", "model_artifact_ledger_sha256", "patch_sha256", "receipt_module_sha256", "client_sha256", "static_preregistration_sha256", "server_info_sha256", "base_url", "server_info_endpoint", "server_info_readiness", "server_configuration", "server_configuration_sha256", "server_process", "frontend_process", "scheduler_worker", "authority", "instrumented_overlay", "hardware"}
    require_exact_keys(server, server_keys, f"server {snapshot}")
    require(
        server["schema"] == "hypic-rwd5-server-launch-receipt-v2"
        and server["official_commit"] == OFFICIAL_COMMIT
        and server["official_worktree_clean"] is True
        and server["instrumentation_only_overlay"] is True
        and server["mode"] == mode
        and server["rank"] == rank
        and server["tp_size"] == 1
        and server["base_url"] == expected_base
        and server["server_info_endpoint"] == f"{expected_base}/server_info"
        and server["server_info_sha256"] == readiness["server_info_sha256"]
        and server["static_preregistration_sha256"] == prereg_sha
        and server["server_info_readiness"] == {"sha256": digests["readiness"], "identity": readiness}
        and server["scheduler_worker"] == {"receipt_sha256": digests["worker"], "identity": worker}
        and server["hardware"].get("gpu_name") == "NVIDIA H20-3e"
        and server["hardware"].get("gpu_uuid") == platform["gpu_uuids"][rank],
        f"server {snapshot}: launch/readiness/GPU binding drift",
    )

    receipt_keys = {"schema", "status", "official_commit", "authority", "target", "storage_contract", "selection", "tensor_payload", "metadata", "component_presence", "allocator_observation", "forbidden_denominators"}
    require_exact_keys(receipt, receipt_keys, f"receipt {snapshot}")
    require_exact_keys(
        receipt["target"],
        {
            "snapshot_id",
            "mode",
            "rank",
            "workload_id",
            "document_token_sha256",
            "document_tokens",
            "seam_tokens",
        },
        f"receipt target {snapshot}",
    )
    require(
        receipt["schema"] == "forkaudit-hypic-retained-state-receipt-v2"
        and receipt["status"] == "owned_state_snapshot_complete"
        and receipt["official_commit"] == OFFICIAL_COMMIT
        and receipt["target"]["snapshot_id"] == snapshot
        and receipt["target"]["mode"] == mode
        and receipt["target"]["rank"] == rank
        and receipt["target"]["workload_id"] == workload_id
        and receipt["target"]["document_token_sha256"] == target["document_token_sha256"]
        and receipt["target"]["document_tokens"] == len(document)
        and receipt["target"]["seam_tokens"] == target["seam_tokens"]
        and receipt["storage_contract"] == prereg["model"]["storage_contract"]
        and receipt["forbidden_denominators"] == ["NVML", "process_allocation", "pool_capacity_delta"],
        f"receipt {snapshot}: frozen identity/denominator drift",
    )
    bindings = receipt["authority"]["bindings"]
    require(
        bindings["official_commit"] == OFFICIAL_COMMIT
        and bindings["target_sha256"] == digests["target"]
        and bindings["preregistration_sha256"] == prereg_sha
        and bindings["server_launch_receipt_sha256"] == digests["server"]
        and bindings["scheduler_worker_receipt_sha256"] == digests["worker"]
        and bindings["freeze_manifest_sha256"] == EXPECTED_RUNTIME_MANIFEST_SHA256,
        f"receipt {snapshot}: authority hash closure failed",
    )
    selection = receipt["selection"]
    (
        full_slots,
        mamba_slots,
        independently_cached,
        metadata_non_tensor_bytes,
    ) = validate_selection_complete(
        selection,
        target,
        prereg,
        mode,
        snapshot,
    )
    owned = selection["owned_document_token_ids"]
    tensor_payload = receipt["tensor_payload"]
    require_exact_keys(tensor_payload, {"records", "union", "denominator"}, f"tensor payload {snapshot}")
    validate_payload_structure(
        receipt,
        prereg,
        mode,
        full_slots,
        mamba_slots,
        snapshot,
    )
    payload_replay = replay_union(tensor_payload["records"], f"tensor payload records {snapshot}")
    require(
        tensor_payload["denominator"] == STORE_DENOMINATOR
        and tensor_payload["union"] == payload_replay,
        f"receipt {snapshot}: independent payload union drift",
    )
    metadata = receipt["metadata"]
    require_exact_keys(metadata, {"tensor_records", "tensor_union", "exact_non_tensor_bytes", "excluded_from_store_mib", "python_allocator_overhead"}, f"metadata {snapshot}")
    validate_metadata_structure(
        metadata,
        selection,
        mode,
        metadata_non_tensor_bytes,
        snapshot,
    )
    metadata_replay = replay_union(metadata["tensor_records"], f"metadata records {snapshot}")
    require(
        metadata["tensor_union"] == metadata_replay
        and exact_int(metadata["exact_non_tensor_bytes"], f"metadata bytes {snapshot}") >= 0
        and metadata["excluded_from_store_mib"] is True,
        f"receipt {snapshot}: metadata exclusion/replay drift",
    )

    (
        mamba_status,
        mamba_raw_count,
        mamba_unique_count,
        mamba_excess,
    ) = validate_terminal(terminal, receipt, snapshot, digests["receipt"], mode)

    raw_keys = {"schema", "status", "official_commit", "mode", "rank", "world_size", "server_configuration", "server_info_sha256", "server_launch_receipt_sha256", "authority", "preregistration_sha256", "freeze_manifest_sha256", "initial_flush_response", "workload", "target_sha256", "target", "warm_prime", "warmup", "prime", "measured", "cache_observation", "store_receipt", "terminal_receipt"}
    require_exact_keys(raw, raw_keys, f"raw {snapshot}")
    require(
        raw["schema"] == "forkaudit-hypic-retained-state-shard-v2"
        and raw["status"] == "completed"
        and raw["official_commit"] == OFFICIAL_COMMIT
        and raw["mode"] == mode
        and raw["rank"] == rank
        and raw["world_size"] == 8
        and raw["server_configuration"]
        == {
            key: value
            for key, value in server["server_configuration"].items()
            if key not in {"rwd5_expected", "rwd5_observed"}
        }
        and server["server_configuration"].get("rwd5_expected")
        == {"enable_int8_mamba_checkpoint": False, "page_size": 1}
        and server["server_configuration"].get("rwd5_observed")
        == server["server_configuration"].get("rwd5_expected")
        and raw["server_info_sha256"] == server["server_info_sha256"]
        and raw["server_launch_receipt_sha256"] == digests["server"]
        and raw["authority"] == receipt["authority"]
        and raw["preregistration_sha256"] == prereg_sha
        and raw["freeze_manifest_sha256"] == EXPECTED_RUNTIME_MANIFEST_SHA256
        and raw["initial_flush_response"] == EXPECTED_INITIAL_FLUSH_RESPONSE
        and raw["target_sha256"] == digests["target"]
        and raw["target"] == target
        and raw["workload"] == target["workload_binding"]
        and raw["cache_observation"] == {"cached_tokens": independently_cached, "authority": "openai-completion-usage.cached_tokens"}
        and raw["store_receipt"]["sha256"] == digests["receipt"]
        and raw["store_receipt"]["payload_bytes"] == payload_replay["unique_overlap_aware_bytes"]
        and raw["store_receipt"]["metadata_excluded"] is True
        and raw["store_receipt"]["captured_after_prime_before_measured"] is True
        and raw["store_receipt"]["path"] == f"{EXPECTED_FORMAL_ROOT_NFS}/store-receipts/{snapshot}.json"
        and raw["terminal_receipt"]["sha256"] == digests["terminal"]
        and raw["terminal_receipt"]["path"] == f"{EXPECTED_FORMAL_ROOT_NFS}/store-receipts/{snapshot}.terminal.json",
        f"raw {snapshot}: complete producer/hash/result closure failed",
    )

    require(
        blind_row["mode"] == mode
        and blind_row["rank"] == rank
        and blind_row["snapshot_id"] == snapshot
        and blind_row["workload_id"] == workload_id
        and blind_row["owned_document_tokens"] == len(owned)
        and blind_row["measured_cached_tokens"] == independently_cached
        and blind_row["payload_bytes"] == payload_replay["unique_overlap_aware_bytes"]
        and blind_row["payload_mib"] == payload_replay["unique_overlap_aware_bytes"] / MIB
        and blind_row["metadata_tensor_bytes"] == metadata_replay["unique_overlap_aware_bytes"]
        and blind_row["metadata_non_tensor_bytes"] == metadata["exact_non_tensor_bytes"]
        and blind_row["mamba_allocator_consistency_status"] == mamba_status
        and blind_row["mamba_allocator_raw_free_count"] == mamba_raw_count
        and blind_row["mamba_allocator_unique_free_count"] == mamba_unique_count
        and blind_row["mamba_allocator_duplicate_excess_count"] == mamba_excess
        and blind_row["metric_validity"] == METRIC_VALIDITY
        and blind_row["global_allocator_correctness_claimed"] is False
        and blind_row["receipt_sha256"] == digests["receipt"]
        and blind_row["terminal_sha256"] == digests["terminal"]
        and blind_row["raw_sha256"] == digests["raw"],
        f"blind row {snapshot}: producer-independent cross-binding drift",
    )
    return {
        "snapshot_id": snapshot,
        "payload_bytes": payload_replay["unique_overlap_aware_bytes"],
        "metadata_tensor_bytes": metadata_replay["unique_overlap_aware_bytes"],
        "cached_tokens": independently_cached,
        "terminal_passed": True,
    }


def validate_mode_summaries(blind: dict[str, Any], cells: list[dict[str, Any]]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    summary_keys = {"median_payload_bytes", "median_payload_mib", "payload_bytes", "owned_document_tokens", "measured_cached_tokens", "mamba_allocator_consistency_status", "mamba_allocator_duplicate_excess_count", "metric_validity", "global_allocator_correctness_claimed"}
    for mode in MODES:
        indices = range(0, 8) if mode == "prefix_cache" else range(8, 16)
        rows = [blind["rows"][index] for index in indices]
        mode_cells = [cell for cell in cells if cell["snapshot_id"].startswith(mode + "-rank-")]
        values = [cell["payload_bytes"] for cell in mode_cells]
        require(len(values) == 8, f"mode {mode}: exact 8 cells")
        ordered = sorted(values)
        median = Fraction(ordered[3] + ordered[4], 2)
        require(median.denominator == 1, f"mode {mode}: byte median is half-integral; fail closed")
        median_bytes = median.numerator
        summary = blind["modes"][mode]
        require(isinstance(summary, dict), f"mode {mode}: summary object")
        require_exact_keys(summary, summary_keys, f"mode summary {mode}")
        require(
            summary["median_payload_bytes"] == median_bytes
            and summary["median_payload_mib"] == median_bytes / MIB
            and summary["payload_bytes"] == [row["payload_bytes"] for row in rows] == values
            and summary["owned_document_tokens"] == [row["owned_document_tokens"] for row in rows]
            and summary["measured_cached_tokens"] == [row["measured_cached_tokens"] for row in rows]
            and summary["mamba_allocator_consistency_status"] == [row["mamba_allocator_consistency_status"] for row in rows]
            and summary["mamba_allocator_duplicate_excess_count"] == [row["mamba_allocator_duplicate_excess_count"] for row in rows]
            and summary["metric_validity"] == METRIC_VALIDITY
            and summary["global_allocator_correctness_claimed"] is False,
            f"mode summary {mode}: exact 8-cell aggregation drift",
        )
        result[mode] = values
    return result


def validate_terminal_idle(formal: Path, platform: dict[str, Any]) -> None:
    compute = read_regular_bytes(formal / "terminal-idle-compute.csv", "terminal idle compute")
    require(compute == b"", "terminal idle compute: expected zero compute applications")
    gpu_raw = read_regular_bytes(formal / "terminal-idle-gpus.csv", "terminal idle GPUs")
    try:
        lines = gpu_raw.decode("ascii", errors="strict").splitlines()
    except UnicodeError as error:
        raise AcceptanceError("terminal idle GPUs: ASCII required") from error
    require(len(lines) == 8, "terminal idle GPUs: exact eight rows")
    seen: set[int] = set()
    for line in lines:
        fields = [value.strip() for value in line.split(",")]
        require(len(fields) == 3 and fields[0].isdigit(), f"terminal idle GPUs: malformed row {line!r}")
        index = int(fields[0])
        require(
            0 <= index < 8
            and index not in seen
            and fields[1] == platform["gpu_uuids"][index]
            and fields[2] == "0",
            f"terminal idle GPUs: index/UUID/zero-MiB drift at row {line!r}",
        )
        seen.add(index)
    require(seen == set(range(8)), "terminal idle GPUs: exact index domain")
    processes_raw = read_regular_bytes(formal / "terminal-idle-processes.txt", "terminal idle processes")
    try:
        process_lines = processes_raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeError as error:
        raise AcceptanceError("terminal idle processes: UTF-8 required") from error
    require(process_lines, "terminal idle processes: nonempty ps snapshot required")
    run = EXPECTED_FORMAL_ROOT_NFS
    repo = f"/tmp/HYPIC-98147c0-rwd5-store-{TRIAL_ID}"
    client = f"/tmp/rwd5-hypic-store-runtime-{TRIAL_ID}/code/run_hypic_retained_state_bytes.py"
    for line in process_lines:
        matches_scope = run in line or repo in line or client in line
        matches_process = "sglang.launch_server" in line or "scheduler" in line or client in line
        require(not (matches_scope and matches_process), f"terminal idle processes: formal process remained: {line}")


def exact_median_bytes(values: list[int]) -> int:
    require(len(values) == 8 and all(type(value) is int and value > 0 for value in values), "median: eight positive byte integers required")
    ordered = sorted(values)
    result = Fraction(ordered[3] + ordered[4], 2)
    require(result.denominator == 1, "median: non-integral byte median forbidden")
    return result.numerator


def exact_terminating_decimal(value: Fraction) -> str:
    numerator = value.numerator
    denominator = value.denominator
    twos = fives = 0
    while denominator % 2 == 0:
        denominator //= 2
        twos += 1
    while denominator % 5 == 0:
        denominator //= 5
        fives += 1
    require(denominator == 1, "exact decimal requested for non-terminating fraction")
    places = max(twos, fives)
    scaled = numerator * (2 ** (places - twos)) * (5 ** (places - fives))
    if places == 0:
        return str(scaled)
    sign = "-" if scaled < 0 else ""
    digits = str(abs(scaled)).rjust(places + 1, "0")
    result = f"{sign}{digits[:-places]}.{digits[-places:]}"
    return result.rstrip("0").rstrip(".")


def rational_payload(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact_fraction": f"{value.numerator}/{value.denominator}",
    }


def store_payload(median_bytes: int, cell_bytes: list[int]) -> dict[str, Any]:
    mib = Fraction(median_bytes, MIB)
    return {
        "cell_bytes_rank_0_to_7": cell_bytes,
        "median_bytes": median_bytes,
        "median_mib": {
            **rational_payload(mib),
            "exact_decimal": exact_terminating_decimal(mib),
            "bytes_per_mib": MIB,
        },
    }


def accept(boot_root: Path) -> dict[str, Any]:
    require(boot_root.is_absolute(), "boot root must be absolute")
    formal = boot_root / "formal-run"
    require_directory(formal, "formal run")
    stage_times = validate_core_topology(formal)
    ledger, ledger_sha = validate_artifact_ledger(formal)
    platform, asset = validate_outer_authority(boot_root, formal, stage_times)
    prereg, prereg_sha = validate_static(formal, boot_root, platform, asset, ledger)
    blind = validate_blind_replay(formal)
    cells: list[dict[str, Any]] = []
    expected_row_index = 0
    for mode in MODES:
        for rank in range(8):
            row = blind["rows"][expected_row_index]
            cells.append(validate_cell(formal, mode, rank, row, prereg, prereg_sha, platform, ledger))
            expected_row_index += 1
    require(len(cells) == 16 and all(cell["terminal_passed"] for cell in cells), "terminal cells: strict 16/16 failed")
    mode_values = validate_mode_summaries(blind, cells)
    validate_terminal_idle(formal, platform)

    prefix_bytes = exact_median_bytes(mode_values["prefix_cache"])
    hypic_bytes = exact_median_bytes(mode_values["transition_rope_recompute"])
    return {
        "schema": "hypic-rwd5-trial1892234-strict-store-acceptance-v1",
        "status": "passed_strict_16_of_16",
        "job_id": JOB_ID,
        "trial_id": TRIAL_ID,
        "expected_nfs_boot_root": EXPECTED_BOOT_ROOT_NFS,
        "validated_boot_root": str(boot_root),
        "terminal_cells": {"passed": 16, "expected": 16},
        "artifact_ledger": {
            "sha256": ledger_sha,
            "regular_members_verified": len(ledger),
            "exact_file_set_closed": True,
        },
        "store_denominator": STORE_SCOPE,
        "median_definition": "exact median of 8 frozen ranks per mode",
        "clean_start_allocator_gate": (
            "all 16 pre-snapshot and terminal Mamba allocator domains are exact, "
            "unique, and duplicate-free"
        ),
        "store": {
            "P_prefix_cache": store_payload(prefix_bytes, mode_values["prefix_cache"]),
            "H_hypic_transition_rope_recompute": store_payload(
                hypic_bytes, mode_values["transition_rope_recompute"]
            ),
            "H_over_P": rational_payload(Fraction(hypic_bytes, prefix_bytes)),
            "comparison_to_comem_q8": {
                "comem_q8_bytes": COMEM_Q8_BYTES,
                "P_over_comem_q8": rational_payload(Fraction(prefix_bytes, COMEM_Q8_BYTES)),
                "H_over_comem_q8": rational_payload(Fraction(hypic_bytes, COMEM_Q8_BYTES)),
                "comem_q8_over_P": rational_payload(Fraction(COMEM_Q8_BYTES, prefix_bytes)),
                "comem_q8_over_H": rational_payload(Fraction(COMEM_Q8_BYTES, hypic_bytes)),
            },
        },
        "claim_boundary": BLIND_CLAIM_BOUNDARY,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed 16/16 terminal acceptance and exact P/H Store extraction "
            "for QS Trial 1892234."
        )
    )
    parser.add_argument(
        "--boot-root",
        type=Path,
        default=Path(EXPECTED_BOOT_ROOT_NFS),
        help=(
            "Trial boot root (default: original NFS path). A relocated byte mirror is "
            "allowed, but embedded artifact paths remain pinned to the original NFS root."
        ),
    )
    args = parser.parse_args()
    try:
        result = accept(args.boot_root)
    except (AcceptanceError, OSError, KeyError, IndexError, TypeError, ValueError) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
