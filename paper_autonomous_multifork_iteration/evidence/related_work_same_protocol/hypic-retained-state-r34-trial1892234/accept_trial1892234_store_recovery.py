#!/usr/bin/env python3
"""Audit-honest external recovery acceptance for HYPIC Store Trial 1892234.

This verifier deliberately does *not* reinterpret the failed whole-run as a
successful formal run.  It accepts only the narrower scientific object that
survived the post-cell replay failure: sixteen completed GPU measurement cells
whose raw shards, pre-measurement Store receipts, terminal receipts, launch
authorities, targets, and scheduler-worker receipts can still be revalidated.

The verifier hash-pins and re-executes the corrected external replay, requires
its output to be byte-identical to the preserved replay result, then subjects
every cell to the original strict cell validator.  Whole-run COMPLETED,
99_done, terminal static rehash, terminal-idle, and original frozen-replay
closure are explicitly absent and unclaimed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
import tempfile
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from types import ModuleType
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
EXPECTED_RUNTIME_MANIFEST_PATH = Path(
    "/tmp/rwd5-hypic-store-runtime-1892234/RUNTIME-SHA256SUMS"
)
EXPECTED_RUNTIME_MANIFEST_SHA256 = (
    "38219b146dbe5bf56e74262491aaa3f0f1b023f636278e0986c1e2b18f3dfd40"
)
EXPECTED_ORIGINAL_STRICT_VERIFIER_SHA256 = (
    "3855150ad8b9423977e8ee54ddab43206d968c43095939786fd11ca4d0d1a0e8"
)
EXPECTED_RECOVERY_CELL_VERIFIER_SHA256 = (
    "ce445489133c1c8826979843a9b05fbc28cf03e3395a37e37155c58b547a2ce1"
)
EXPECTED_EXTERNAL_REPLAY_VERIFIER_SHA256 = (
    "5fbbf6d49f492a748777475f31cdeba0775b9a107ffd94b39ee7f8c715766591"
)
EXPECTED_ORIGINAL_REPLAY_SHA256 = (
    "ccff3178045eecb4daf5675721aaa800d59fc5ea989a7822c3130fdb34d0fb27"
)
EXPECTED_EXTERNAL_REPLAY_SHA256 = (
    "79face626fc040fc19bf05e3998de41936c46e0d38b13aee803ecc216bb41ab1"
)
COMEM_Q8_BYTES = 16_664_352
MIB = 1_048_576
MODES = ("prefix_cache", "transition_rope_recompute")
EXPECTED_STAGE_ORDER = (
    "00_started",
    "01_focused_and_inherited_tests_passed",
    "02_preregistered_before_outputs",
    "10_prefix_cache_server_info_ready",
    "20_prefix_cache_complete",
    "10_transition_rope_recompute_server_info_ready",
    "20_transition_rope_recompute_complete",
)
STATIC_PAYLOAD_NAMES = {
    "official-source-ledger.json",
    "environment-ledger.json",
    "model-storage-contract.json",
    "instrumentation-overlay.json",
    "instrumentation-overlay.diff",
    "preregistration.json",
}
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\n$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9A-Fa-f-]+$")


class RecoveryAcceptanceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RecoveryAcceptanceError(message)


def sha256_file(path: Path, label: str) -> str:
    try:
        before = path.lstat()
    except OSError as error:
        raise RecoveryAcceptanceError(f"{label}: missing/unreadable: {path}: {error}") from error
    require(stat.S_ISREG(before.st_mode), f"{label}: regular non-symlink file required")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    digest = hashlib.sha256()
    descriptor = os.open(path, flags)
    try:
        opened_before = os.fstat(descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            == (
                opened_before.st_dev,
                opened_before.st_ino,
                opened_before.st_size,
                opened_before.st_mtime_ns,
                opened_before.st_ctime_ns,
            ),
            f"{label}: path/open identity race",
        )
        while True:
            block = os.read(descriptor, 8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
        opened_after = os.fstat(descriptor)
        require(
            (opened_before.st_size, opened_before.st_mtime_ns, opened_before.st_ctime_ns)
            == (opened_after.st_size, opened_after.st_mtime_ns, opened_after.st_ctime_ns),
            f"{label}: file changed while hashing",
        )
    finally:
        os.close(descriptor)
    after = path.lstat()
    require(
        (opened_after.st_dev, opened_after.st_ino, opened_after.st_size,
         opened_after.st_mtime_ns, opened_after.st_ctime_ns)
        == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
        f"{label}: path changed after hashing",
    )
    return digest.hexdigest()


def read_bytes(path: Path, label: str) -> bytes:
    # Hash before and after so an input cannot change unnoticed during the read.
    before = sha256_file(path, label)
    payload = path.read_bytes()
    after = sha256_file(path, label)
    require(before == after == hashlib.sha256(payload).hexdigest(), f"{label}: unstable read")
    return payload


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    payload = read_bytes(path, label)
    try:
        value = json.loads(payload.decode("utf-8", errors="strict"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RecoveryAcceptanceError(f"{label}: strict JSON decode failed: {error}") from error
    require(isinstance(value, dict), f"{label}: JSON object required")
    require(payload == canonical_json_bytes(value), f"{label}: canonical JSON bytes required")
    return value


def exact_children(directory: Path, expected: set[str], label: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(), f"{label}: real directory required")
    actual = {path.name for path in directory.iterdir()}
    require(
        actual == expected,
        f"{label}: exact file set drift; missing={sorted(expected-actual)}, extra={sorted(actual-expected)}",
    )
    for name in expected:
        row = (directory / name).lstat()
        require(stat.S_ISREG(row.st_mode), f"{label}/{name}: regular non-symlink file required")


def parse_timestamp(path: Path, label: str) -> datetime:
    payload = read_bytes(path, label)
    try:
        text = payload.decode("ascii", errors="strict")
    except UnicodeError as error:
        raise RecoveryAcceptanceError(f"{label}: ASCII required") from error
    require(TIMESTAMP_RE.fullmatch(text) is not None, f"{label}: canonical UTC timestamp required")
    return datetime.strptime(text.strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def import_hash_pinned(path: Path, expected_sha256: str, module_name: str) -> ModuleType:
    require(sha256_file(path, module_name) == expected_sha256, f"{module_name}: SHA-256 drift")
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"{module_name}: import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_trial_status_observation(path: Path) -> tuple[dict[str, Any], str]:
    value = load_canonical_json(path, "QS trial-status observation")
    expected_keys = {
        "schema",
        "source_command",
        "captured_at_utc",
        "job_id",
        "trial_id",
        "trial_name",
        "trial_status",
        "create_time",
        "update_time",
        "completion_time",
    }
    require(set(value) == expected_keys, "QS trial-status observation: exact schema")
    require(
        value["schema"] == "qs-training-trial-status-observation-v1"
        and value["source_command"] == "qs training get 1892234 --format json -q"
        and value["job_id"] == JOB_ID
        and value["trial_id"] == TRIAL_ID
        and value["trial_name"]
        == "liuhanzuo-qcomem-hypic-store-shared-v5-cu129meta-20260824-r34"
        and value["trial_status"] == "Failed"
        and isinstance(value["captured_at_utc"], str)
        and value["captured_at_utc"].endswith("Z"),
        "QS trial-status observation: platform identity/status drift",
    )
    datetime.fromisoformat(value["captured_at_utc"].replace("Z", "+00:00"))
    require(all(value[key] for key in ("create_time", "update_time", "completion_time")),
            "QS trial-status observation: timestamps required")
    return value, sha256_file(path, "QS trial-status observation")


def validate_recovery_topology(boot_root: Path) -> tuple[Path, dict[str, str], dict[str, datetime]]:
    require(boot_root.is_absolute(), "boot root must be absolute")
    require(str(boot_root) == EXPECTED_BOOT_ROOT_NFS, "original NFS boot-root authority required")
    require(boot_root.is_dir() and not boot_root.is_symlink(), "boot root: real directory required")
    formal = boot_root / "formal-run"
    require(formal.is_dir() and not formal.is_symlink(), "formal run: real directory required")

    # Preserve the failed whole-run outcome rather than manufacturing completion.
    outer_failed = load_canonical_json(boot_root / "FAILED", "outer FAILED")
    require(set(outer_failed) == {"exit_code", "timestamp_utc"}, "outer FAILED: exact schema")
    require(outer_failed["exit_code"] == 1, "outer FAILED: exit code must remain 1")
    require(read_bytes(formal / "FAILED", "formal FAILED") == b"1\n", "formal FAILED marker drift")
    for absent in (
        boot_root / "BOOTSTRAP_COMPLETED",
        formal / "COMPLETED",
        formal / "all-artifacts.sha256",
        formal / "blind-replay.json",
        formal / "terminal-static-verification.json",
        formal / "terminal-idle-compute.csv",
        formal / "terminal-idle-gpus.csv",
        formal / "terminal-idle-processes.txt",
    ):
        require(not absent.exists(), f"unexpected whole-run closure artifact: {absent}")

    cell_names = {f"{mode}-rank-{rank}" for mode in MODES for rank in range(8)}
    exact_children(formal / "raw", {f"{name}.json" for name in cell_names}, "raw")
    exact_children(formal / "targets", {f"{name}.json" for name in cell_names}, "targets")
    exact_children(
        formal / "store-receipts",
        {f"{name}.json" for name in cell_names}
        | {f"{name}.terminal.json" for name in cell_names},
        "store-receipts",
    )
    exact_children(
        formal / "server-receipts",
        {f"{name}.json" for name in cell_names}
        | {f"{name}.readiness.json" for name in cell_names},
        "server-receipts",
    )
    exact_children(
        formal / "scheduler-workers",
        {f"{name}.json" for name in cell_names},
        "scheduler-workers",
    )
    exact_children(formal / "static", STATIC_PAYLOAD_NAMES | {"preoutput-validation.json"}, "static")
    exact_children(formal / "stages", set(EXPECTED_STAGE_ORDER), "stages")
    stage_times = {
        name: parse_timestamp(formal / "stages" / name, f"stage {name}")
        for name in EXPECTED_STAGE_ORDER
    }
    ordered = [stage_times[name] for name in EXPECTED_STAGE_ORDER]
    require(ordered == sorted(ordered), "recovery stages: nondecreasing execution order required")
    require(
        outer_failed["timestamp_utc"] == stage_times["20_transition_rope_recompute_complete"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "outer FAILED timestamp must follow the completed sixteenth cell",
    )

    external_dir = boot_root / "external-recovery-r34-attempt2"
    exact_children(external_dir, {"blind-replay.json", "blind-replay.json.sha256"}, "external replay")

    relevant: set[str] = {
        "FAILED",
        "formal-run/FAILED",
        "dynamic-platform-command-authority.json",
        "model-asset-observation.json",
        "external-recovery-r34-attempt2/blind-replay.json",
        "external-recovery-r34-attempt2/blind-replay.json.sha256",
    }
    relevant |= {f"formal-run/stages/{name}" for name in EXPECTED_STAGE_ORDER}
    relevant |= {f"formal-run/static/{name}" for name in STATIC_PAYLOAD_NAMES | {"preoutput-validation.json"}}
    for name in cell_names:
        relevant |= {
            f"formal-run/raw/{name}.json",
            f"formal-run/targets/{name}.json",
            f"formal-run/store-receipts/{name}.json",
            f"formal-run/store-receipts/{name}.terminal.json",
            f"formal-run/server-receipts/{name}.json",
            f"formal-run/server-receipts/{name}.readiness.json",
            f"formal-run/scheduler-workers/{name}.json",
        }
    hashes = {name: sha256_file(boot_root / name, f"recovery member {name}") for name in sorted(relevant)}
    return formal, hashes, stage_times


def validate_platform_and_asset(strict: ModuleType, boot_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    platform = strict.load_json(
        boot_root / "dynamic-platform-command-authority.json", "platform authority"
    )
    strict.require_exact_keys(
        platform,
        {
            "schema", "platform_job_id", "platform_trial_id", "scope",
            "status_at_bootstrap", "platform_command_pid", "platform_command",
            "platform_command_environ_sha256", "runtime_platform_command_environ_required",
            "gpu_count", "gpu_name", "gpu_memory_mib", "gpu_uuids", "paper_evidence",
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
        and type(platform["platform_command_pid"]) is int
        and platform["platform_command_pid"] > 1
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
        isinstance(uuids, list) and len(uuids) == len(set(uuids)) == 8
        and all(isinstance(value, str) and GPU_UUID_RE.fullmatch(value) for value in uuids),
        "platform authority: exact eight unique GPU UUIDs required",
    )

    asset = strict.load_json(boot_root / "model-asset-observation.json", "model asset observation")
    strict.require_exact_keys(
        asset,
        {
            "schema", "model_root", "entries",
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
        "model-artifacts.sha256": (
            "d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd", 778
        ),
        "preprocessor_config.json": (
            "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516", 390
        ),
        "video_preprocessor_config.json": (
            "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13", 385
        ),
    }
    require(
        isinstance(asset["entries"], list)
        and [row.get("name") for row in asset["entries"]] == sorted(expected_assets),
        "model asset observation: exact entries/order",
    )
    for row in asset["entries"]:
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


def validate_preoutput_static(
    strict: ModuleType,
    formal: Path,
    boot_root: Path,
    platform: dict[str, Any],
    asset: dict[str, Any],
    member_hashes: dict[str, str],
) -> tuple[dict[str, Any], str]:
    static = formal / "static"
    pre = strict.load_json(static / "preoutput-validation.json", "preoutput validation")
    strict.require_exact_keys(pre, {"schema", "passed", "files"}, "preoutput validation")
    require(
        pre["schema"] == "hypic-rwd5-preoutput-validation-v1"
        and pre["passed"] is True
        and isinstance(pre["files"], dict)
        and set(pre["files"]) == STATIC_PAYLOAD_NAMES,
        "preoutput validation: frozen contract drift",
    )
    for name in STATIC_PAYLOAD_NAMES:
        require(
            pre["files"][name] == member_hashes[f"formal-run/static/{name}"],
            f"preoutput validation: SHA drift for {name}",
        )

    prereg = strict.load_json(static / "preregistration.json", "preregistration")
    prereg_sha = member_hashes["formal-run/static/preregistration.json"]
    require(
        prereg.get("schema") == "hypic-rwd5-retained-state-preregistration-v2"
        and prereg.get("status") == "frozen_before_outputs"
        and prereg.get("official_commit") == OFFICIAL_COMMIT
        and prereg.get("external_freeze", {}).get("manifest_sha256")
        == EXPECTED_RUNTIME_MANIFEST_SHA256
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
        and replay_row.get("sha256") == EXPECTED_ORIGINAL_REPLAY_SHA256,
        "preregistration: original frozen replay identity drift",
    )
    platform_binding = prereg.get("platform_execution_authority", {})
    platform_sha = member_hashes["dynamic-platform-command-authority.json"]
    require(
        platform_binding.get("receipt_path")
        == f"{EXPECTED_BOOT_ROOT_NFS}/dynamic-platform-command-authority.json"
        and platform_binding.get("receipt_sha256") == platform_sha
        and platform_binding.get("platform_job_id") == JOB_ID
        and platform_binding.get("platform_trial_id") == TRIAL_ID
        and platform_binding.get("platform_command_identity_verified") is True,
        "preregistration: platform authority binding drift",
    )
    asset_binding = prereg.get("model", {}).get("asset_identity", {})
    require(
        asset_binding.get("observation_path")
        == f"{EXPECTED_BOOT_ROOT_NFS}/model-asset-observation.json"
        and asset_binding.get("observation_sha256")
        == member_hashes["model-asset-observation.json"]
        and asset_binding.get("observation") == asset
        and asset_binding.get("old_node_inode_device_and_timestamps_are_not_authority") is True,
        "preregistration: model asset authority binding drift",
    )
    require(platform["platform_job_id"] == JOB_ID, "platform/preregistration identity")
    contract = strict.load_json(static / "model-storage-contract.json", "model storage contract")
    require(
        contract.get("schema") == "hypic-rwd5-model-storage-contract-v3"
        and prereg.get("model", {}).get("storage_contract") == contract
        and prereg.get("model", {}).get("storage_contract_sha256")
        == member_hashes["formal-run/static/model-storage-contract.json"],
        "model storage contract binding drift",
    )
    return prereg, prereg_sha


def validate_external_replay(
    strict: ModuleType,
    replay: ModuleType,
    boot_root: Path,
    formal: Path,
    external_replay_path: Path,
    member_hashes: dict[str, str],
) -> dict[str, Any]:
    require(
        member_hashes["external-recovery-r34-attempt2/blind-replay.json"]
        == EXPECTED_EXTERNAL_REPLAY_SHA256,
        "preserved external replay: pinned SHA-256 drift",
    )
    checksum_line = read_bytes(
        external_replay_path.with_suffix(".json.sha256"), "external replay checksum receipt"
    )
    expected_line = (
        f"{EXPECTED_EXTERNAL_REPLAY_SHA256}  "
        f"{EXPECTED_BOOT_ROOT_NFS}/external-recovery-r34-attempt2/blind-replay.json\n"
    ).encode("ascii")
    require(checksum_line == expected_line, "external replay checksum receipt drift")

    preserved = strict.load_json(external_replay_path, "preserved external replay")
    strict.require_exact_keys(
        preserved,
        {
            "schema", "passed", "official_commit", "freeze_manifest_sha256",
            "denominator", "claim_boundary", "global_allocator_correctness_claimed",
            "rows", "modes",
        },
        "preserved external replay",
    )
    require(
        preserved["schema"] == "forkaudit-hypic-retained-state-blind-replay-v2"
        and preserved["passed"] is True
        and preserved["official_commit"] == OFFICIAL_COMMIT
        and preserved["freeze_manifest_sha256"] == EXPECTED_RUNTIME_MANIFEST_SHA256
        and preserved["global_allocator_correctness_claimed"] is False
        and isinstance(preserved["rows"], list)
        and len(preserved["rows"]) == 16
        and set(preserved["modes"]) == set(MODES),
        "preserved external replay: identity/status drift",
    )

    require(
        sha256_file(EXPECTED_RUNTIME_MANIFEST_PATH, "runtime manifest")
        == EXPECTED_RUNTIME_MANIFEST_SHA256,
        "runtime manifest SHA-256 drift",
    )
    with tempfile.TemporaryDirectory(prefix="r34-recovery-replay-") as temporary:
        replay_output = Path(temporary) / "blind-replay.json"
        replay.replay_all(
            formal,
            replay_output,
            EXPECTED_RUNTIME_MANIFEST_PATH,
            EXPECTED_RUNTIME_MANIFEST_SHA256,
        )
        rerun_bytes = read_bytes(replay_output, "rerun external replay")
    require(
        rerun_bytes == read_bytes(external_replay_path, "preserved external replay final"),
        "external replay rerun is not byte-identical to preserved output",
    )
    return preserved


def validate_exact_process_authority(strict: ModuleType, formal: Path) -> None:
    """Close the only external-replay exception around setproctitle procfs state."""
    exact_worker_cmdline = ["sglang::scheduler"]
    exact_worker_cmdline_sha = hashlib.sha256(
        canonical_json_bytes(exact_worker_cmdline)
    ).hexdigest()
    exact_empty_environment: dict[str, str] = {}
    exact_empty_environment_sha = hashlib.sha256(
        canonical_json_bytes(exact_empty_environment)
    ).hexdigest()
    require(
        exact_worker_cmdline_sha
        == "5e20f8ba2c49e215e8f2289357633577e557e31fcb64536fbd9cf4ae2ca28068"
        and exact_empty_environment_sha
        == "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356",
        "canonical process-identity constants drift",
    )
    for mode in MODES:
        for rank in range(8):
            snapshot = f"{mode}-rank-{rank}"
            target_path = formal / "targets" / f"{snapshot}.json"
            worker_path = formal / "scheduler-workers" / f"{snapshot}.json"
            server_path = formal / "server-receipts" / f"{snapshot}.json"
            receipt_path = formal / "store-receipts" / f"{snapshot}.json"
            worker = strict.load_json(worker_path, f"process worker {snapshot}")
            server = strict.load_json(server_path, f"process server {snapshot}")
            receipt = strict.load_json(receipt_path, f"process receipt {snapshot}")
            frontend = server["frontend_process"]
            process = worker["process"]
            expected_environment = {
                "CUDA_VISIBLE_DEVICES": server["hardware"]["gpu_uuid"],
                "FORKAUDIT_RWD5_FREEZE_MANIFEST_PATH": str(EXPECTED_RUNTIME_MANIFEST_PATH),
                "FORKAUDIT_RWD5_FREEZE_MANIFEST_SHA256": EXPECTED_RUNTIME_MANIFEST_SHA256,
                "FORKAUDIT_RWD5_FRONTEND_PID": str(frontend["pid"]),
                "FORKAUDIT_RWD5_MODE": mode,
                "FORKAUDIT_RWD5_PREREGISTRATION_PATH": str(formal / "static/preregistration.json"),
                "FORKAUDIT_RWD5_RANK": str(rank),
                "FORKAUDIT_RWD5_RECEIPT_DIR": str(formal / "store-receipts"),
                "FORKAUDIT_RWD5_SERVER_RECEIPT_PATH": str(server_path),
                "FORKAUDIT_RWD5_TARGET_PATH": str(target_path),
                "FORKAUDIT_RWD5_WORKER_RECEIPT_PATH": str(worker_path),
                "PIC_SEAM_SINK": "8",
                "PYTHONPATH": (
                    f"/tmp/HYPIC-98147c0-rwd5-store-{TRIAL_ID}/python:"
                    f"/tmp/rwd5-hypic-store-runtime-{TRIAL_ID}/code"
                ),
                "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
                "SGLANG_MAMBA_SSM_DTYPE": "float32",
            }
            require(
                set(frontend)
                == {"pid", "ppid", "cmdline", "cmdline_sha256", "environment", "environment_sha256"},
                f"frontend {snapshot}: exact process schema",
            )
            require(
                frontend["environment"] == expected_environment
                and len(frontend["environment"]) == 15
                and frontend["environment_sha256"]
                == hashlib.sha256(canonical_json_bytes(expected_environment)).hexdigest()
                and frontend["cmdline_sha256"]
                == hashlib.sha256(canonical_json_bytes(frontend["cmdline"])).hexdigest(),
                f"frontend {snapshot}: exact 15-key environment/cmdline hash closure",
            )
            require(
                set(process)
                == {
                    "pid", "ppid", "cmdline", "cmdline_sha256", "environment",
                    "environment_sha256", "ancestry", "ancestry_pids",
                },
                f"worker {snapshot}: exact process schema",
            )
            require(
                process["environment"] == exact_empty_environment
                and process["environment_sha256"] == exact_empty_environment_sha
                and process["cmdline"] == exact_worker_cmdline
                and process["cmdline_sha256"] == exact_worker_cmdline_sha
                and process["ppid"] == frontend["pid"]
                and worker["frontend_pid"] == frontend["pid"],
                f"worker {snapshot}: exact setproctitle fallback/direct-parent closure",
            )
            ancestry = process["ancestry"]
            require(
                isinstance(ancestry, list)
                and len(ancestry) >= 1
                and all(
                    isinstance(row, dict)
                    and set(row) == {"pid", "ppid", "cmdline_sha256"}
                    for row in ancestry
                )
                and ancestry[0]
                == {
                    "pid": frontend["pid"],
                    "ppid": frontend["ppid"],
                    "cmdline_sha256": frontend["cmdline_sha256"],
                }
                and process["ancestry_pids"] == [row["pid"] for row in ancestry]
                and len(process["ancestry_pids"]) == len(set(process["ancestry_pids"]))
                and all(ancestry[index]["ppid"] == ancestry[index + 1]["pid"]
                        for index in range(len(ancestry) - 1))
                and ancestry[-1]["ppid"] == 0,
                f"worker {snapshot}: exact first-row and complete ancestry chain closure",
            )
            require(
                receipt["authority"]["scheduler_process"] == process,
                f"receipt {snapshot}: scheduler process must equal worker process in full",
            )


def artifact_manifest_payload(member_hashes: dict[str, str]) -> tuple[str, str]:
    text = "".join(
        f"{digest}  {EXPECTED_BOOT_ROOT_NFS}/{relative}\n"
        for relative, digest in sorted(member_hashes.items(), key=lambda row: row[0].encode("utf-8"))
    )
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest()


def exact_median_bytes(values: list[int]) -> int:
    require(len(values) == 8 and all(type(value) is int and value > 0 for value in values),
            "median: exact eight positive byte integers required")
    ordered = sorted(values)
    median = Fraction(ordered[3] + ordered[4], 2)
    require(median.denominator == 1, "median: non-integral byte median forbidden")
    return median.numerator


def rational_payload(value: Fraction) -> dict[str, Any]:
    return {
        "numerator": value.numerator,
        "denominator": value.denominator,
        "exact_fraction": f"{value.numerator}/{value.denominator}",
    }


def store_payload(median_bytes: int, cells: list[int]) -> dict[str, Any]:
    mib = Fraction(median_bytes, MIB)
    return {
        "cell_bytes_rank_0_to_7": cells,
        "median_bytes": median_bytes,
        "median_mib": {
            **rational_payload(mib),
            "exact_decimal": str(float(mib)),
            "bytes_per_mib": MIB,
        },
    }


def accept(
    boot_root: Path,
    strict_verifier_path: Path,
    recovery_cell_verifier_path: Path,
    external_replay_verifier_path: Path,
    trial_status_observation_path: Path,
) -> tuple[dict[str, Any], str]:
    require(
        sha256_file(strict_verifier_path, "r34 original strict acceptance")
        == EXPECTED_ORIGINAL_STRICT_VERIFIER_SHA256,
        "r34 original strict acceptance: SHA-256 drift",
    )
    strict = import_hash_pinned(
        recovery_cell_verifier_path,
        EXPECTED_RECOVERY_CELL_VERIFIER_SHA256,
        "r34_recovery_cell_acceptance",
    )
    replay = import_hash_pinned(
        external_replay_verifier_path,
        EXPECTED_EXTERNAL_REPLAY_VERIFIER_SHA256,
        "r34_external_corrected_replay",
    )
    trial_observation, trial_observation_sha = validate_trial_status_observation(
        trial_status_observation_path
    )
    formal, member_hashes, stage_times = validate_recovery_topology(boot_root)
    platform, asset = validate_platform_and_asset(strict, boot_root)
    prereg, prereg_sha = validate_preoutput_static(
        strict, formal, boot_root, platform, asset, member_hashes
    )
    external_replay_path = (
        boot_root / "external-recovery-r34-attempt2" / "blind-replay.json"
    )
    blind = validate_external_replay(
        strict, replay, boot_root, formal, external_replay_path, member_hashes
    )
    validate_exact_process_authority(strict, formal)

    # The original strict validator is retained unchanged and reused below only
    # for cell-local validation.  Its whole-run topology/COMPLETED checks are not
    # called, and this narrower scope is made explicit in the output.
    ledger = {
        relative.removeprefix("formal-run/"): digest
        for relative, digest in member_hashes.items()
        if relative.startswith("formal-run/")
    }
    cells: list[dict[str, Any]] = []
    row_index = 0
    for mode in MODES:
        for rank in range(8):
            cells.append(
                strict.validate_cell(
                    formal,
                    mode,
                    rank,
                    blind["rows"][row_index],
                    prereg,
                    prereg_sha,
                    platform,
                    ledger,
                )
            )
            row_index += 1
    require(
        len(cells) == 16 and all(cell["terminal_passed"] for cell in cells),
        "GPU measurement cells: exact 16/16 terminal closure failed",
    )
    mode_values = strict.validate_mode_summaries(blind, cells)
    require(
        all(row["mamba_allocator_consistency_status"] == "exact_unique_free_domain"
            for row in blind["rows"])
        and all(row["mamba_allocator_duplicate_excess_count"] == 0 for row in blind["rows"]),
        "clean-start gate: all 16 pre-snapshot Mamba domains must be exact and duplicate-free",
    )

    # Rehash every remote member after all validation to close read-time races.
    final_hashes = {
        relative: sha256_file(boot_root / relative, f"final recovery member {relative}")
        for relative in member_hashes
    }
    require(final_hashes == member_hashes, "remote recovery evidence changed during validation")
    manifest_text, manifest_sha = artifact_manifest_payload(member_hashes)

    prefix_cells = mode_values["prefix_cache"]
    hypic_cells = mode_values["transition_rope_recompute"]
    prefix_bytes = exact_median_bytes(prefix_cells)
    hypic_bytes = exact_median_bytes(hypic_cells)
    verifier_sha = sha256_file(Path(__file__).resolve(), "recovery acceptance verifier")
    result = {
        "schema": "hypic-rwd5-trial1892234-external-store-recovery-acceptance-v1",
        "schema_version": "hypic-rwd5-trial1892234-external-store-acceptance-v1",
        "status": "passed_external_replay_16_of_16",
        "job_id": JOB_ID,
        "trial_id": TRIAL_ID,
        "official_commit": OFFICIAL_COMMIT,
        "trial_terminal_status": trial_observation["trial_status"],
        "expected_nfs_boot_root": EXPECTED_BOOT_ROOT_NFS,
        "validated_boot_root": str(boot_root),
        "whole_run": {
            "platform_trial_status": trial_observation["trial_status"],
            "platform_status_observation_sha256": trial_observation_sha,
            "outer_failed_marker_sha256": member_hashes["FAILED"],
            "formal_failed_marker_sha256": member_hashes["formal-run/FAILED"],
            "original_completed_marker_present": False,
            "original_stage_99_present": False,
            "original_blind_replay_present": False,
            "terminal_static_reverification_present": False,
            "terminal_idle_snapshot_present": False,
            "accepted_as_successful_whole_run": False,
        },
        "gpu_measurement_cells": {
            "passed": 16,
            "expected": 16,
            "per_mode": {"prefix_cache": 8, "transition_rope_recompute": 8},
            "last_completed_stage_utc": stage_times[
                "20_transition_rope_recompute_complete"
            ].strftime("%Y-%m-%dT%H:%M:%SZ"),
            "raw_shards_verified": 16,
            "pre_measurement_store_receipts_verified": 16,
            "terminal_receipts_verified": 16,
            "target_server_worker_hash_graph_verified": 16,
        },
        "validation_code": {
            "recovery_acceptance_verifier_sha256": verifier_sha,
            "original_strict_verifier_sha256": EXPECTED_ORIGINAL_STRICT_VERIFIER_SHA256,
            "recovery_cell_verifier_sha256": EXPECTED_RECOVERY_CELL_VERIFIER_SHA256,
            "external_corrected_replay_verifier_sha256": EXPECTED_EXTERNAL_REPLAY_VERIFIER_SHA256,
            "original_frozen_replay_verifier_sha256": EXPECTED_ORIGINAL_REPLAY_SHA256,
            "runtime_manifest_sha256": EXPECTED_RUNTIME_MANIFEST_SHA256,
        },
        "external_replay": {
            "schema": blind["schema"],
            "passed": blind["passed"],
            "rows": len(blind["rows"]),
            "sha256": EXPECTED_EXTERNAL_REPLAY_SHA256,
            "preserved_output_sha256": EXPECTED_EXTERNAL_REPLAY_SHA256,
            "rerun_byte_identical": True,
            "worker_procfs_exception": (
                "worker environment may be exactly empty only when cmdline is exactly "
                "['sglang::scheduler']; frontend environment remains exact"
            ),
            "all_other_cell_and_authority_gates_unchanged": True,
        },
        "terminal_cells": {"passed": 16, "expected": 16},
        "validated_artifacts": {
            "member_count": len(member_hashes),
            "canonical_sha256sum_manifest_sha256": manifest_sha,
            "members": member_hashes,
        },
        "denominator": "exact target-entry-owned physical tensor-range union only",
        "store_denominator": "exact target-entry-owned physical tensor-range union only",
        "median_definition": "exact median of 8 frozen ranks per mode",
        "clean_start_allocator_gate": {
            "pre_snapshot_cells_exact_unique_duplicate_free": 16,
            "terminal_cells_exact_unique_duplicate_free": 16,
            "global_allocator_correctness_claimed": False,
        },
        "store": {
            "P_prefix_cache": store_payload(prefix_bytes, prefix_cells),
            "H_hypic_transition_rope_recompute": store_payload(hypic_bytes, hypic_cells),
            "H_over_P": rational_payload(Fraction(hypic_bytes, prefix_bytes)),
            "comparison_to_comem_q8": {
                "comem_q8_bytes": COMEM_Q8_BYTES,
                "P_over_comem_q8": rational_payload(Fraction(prefix_bytes, COMEM_Q8_BYTES)),
                "H_over_comem_q8": rational_payload(Fraction(hypic_bytes, COMEM_Q8_BYTES)),
                "comem_q8_over_P": rational_payload(Fraction(COMEM_Q8_BYTES, prefix_bytes)),
                "comem_q8_over_H": rational_payload(Fraction(COMEM_Q8_BYTES, hypic_bytes)),
            },
        },
        "modes": {
            "prefix_cache": {
                "payload_bytes": prefix_cells,
                "median_payload_bytes": prefix_bytes,
            },
            "transition_rope_recompute": {
                "payload_bytes": hypic_cells,
                "median_payload_bytes": hypic_bytes,
            },
        },
        "claim_boundary": (
            "cell-level retained-state Store only; no successful whole-run completion, "
            "global allocator correctness, runtime safety, capacity, NVML/process-allocation, "
            "terminal-idle, or post-cell whole-run model-byte-rehash claim"
        ),
        "unverified_or_unclaimed": [
            "original whole-run COMPLETED and stage 99 closure",
            "original frozen replay completion",
            "post-cell whole-run terminal static/model-byte rehash",
            "whole-run terminal GPU/process idle snapshot",
            "global allocator correctness or runtime safety",
            "capacity, NVML, process-allocation, or preallocated-pool memory",
        ],
    }
    return result, manifest_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit-honest external recovery acceptance for r34 Trial 1892234 Store cells."
    )
    parser.add_argument("--boot-root", type=Path, default=Path(EXPECTED_BOOT_ROOT_NFS))
    parser.add_argument("--strict-verifier", type=Path, required=True)
    parser.add_argument("--recovery-cell-verifier", type=Path, required=True)
    parser.add_argument("--external-replay-verifier", type=Path, required=True)
    parser.add_argument("--trial-status-observation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--artifact-manifest-output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result, manifest_text = accept(
            args.boot_root,
            args.strict_verifier,
            args.recovery_cell_verifier,
            args.external_replay_verifier,
            args.trial_status_observation,
        )
        output_bytes = canonical_json_bytes(result)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.artifact_manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(output_bytes)
        args.artifact_manifest_output.write_text(manifest_text, encoding="utf-8")
        require(
            hashlib.sha256(args.artifact_manifest_output.read_bytes()).hexdigest()
            == result["validated_artifacts"]["canonical_sha256sum_manifest_sha256"],
            "written artifact manifest SHA drift",
        )
    except (
        RecoveryAcceptanceError,
        OSError,
        KeyError,
        IndexError,
        TypeError,
        ValueError,
        getattr(sys.modules.get("r34_original_strict_acceptance"), "AcceptanceError", RuntimeError),
        getattr(sys.modules.get("r34_external_corrected_replay"), "ReplayError", RuntimeError),
    ) as error:
        print(f"FAIL_CLOSED: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
