from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


STATUS = "HOLD_PENDING_FRESH_AUDIT_AND_H20"
EXPECTATION_SCHEMA = "forkaudit-r40-v16-terminal-expectation-v1"
TREE_SCHEMA = "forkaudit-r40-v16-terminal-tree-v1"

EXPECTATION_FIELDS = {
    "schema_version",
    "status",
    "root",
    "terminal_tree_output",
    "complete_output",
    "terminal_source_rehash",
    "read_only_staging",
    "source_ledger_sha256",
    "expected_paths",
    "expected_node_count",
    "expected_regular_file_count",
    "expected_directory_count",
    "expected_nodes",
    "payload_sha256",
}
TREE_FIELDS = {
    "schema_version",
    "status",
    "root",
    "excluded_output",
    "expected_paths",
    "expected_node_count",
    "expected_regular_file_count",
    "expected_directory_count",
    "final_node_count",
    "final_regular_file_count",
    "final_directory_count",
    "nodes",
}
CUDA_FIELDS = {
    "schema_version",
    "passed",
    "torch_version",
    "cuda_bf16",
    "noncontiguous_sources",
    "actual_frozen_helper",
    "borrowed_setup_aliases",
    "setup_clone_edges",
    "transition_rebind_edges",
    "transition_private_rows",
    "transition_borrowed_rows",
    "storage_interval_checked",
    "allocator_reuse_probe_completed",
    "allocator_reuse_observed",
    "cuda_synchronized",
    "requires_cuda_smoke_before_science",
}
AGGREGATE_FIELDS = {
    "schema_version",
    "rank_count",
    "rank_results",
    "total_selected_rows",
    "total_storage_rows",
    "total_borrowed_setup_aliases",
    "total_setup_clone_edges",
    "total_functional_rebind_calls",
    "total_functional_rebind_edges",
    "total_phase_artifacts",
    "total_primary_calls_observed",
    "primary_events_by_rank",
    "global_primary_memory_hook_events",
    "execution_bindings",
    "requires_cuda_smoke_before_science",
    "formal_gpu_execution",
}
FORMAL_TOP_LEVELS = {
    "preflight",
    "primary",
    "compiled-dispatch-capture",
    "formal-binding",
    "r40-clean-binding",
    "r40-formal",
}

PRIMARY_CODE_PYCACHE_PREFIX = (
    "primary/pycache/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/"
    "indep-bench/qcomem_r40_v32_finalizer_counter_fix_20260902a/"
    "paper_autonomous_multifork_iteration/evidence/round_04_rr2_package/"
    "executed_source/gpu"
)
PRIMARY_PYCACHE_TAG = ".cpython-311.pyc"

PREFLIGHT_EXACT_PATHS = {
    "preflight",
    "preflight/logs",
    "preflight/logs/detached-focused-tests-162.log",
    "preflight/logs/r39-primary-tests-13.log",
    "preflight/pycache",
    "preflight/stages",
    "preflight/stages/00_upstream_preregistration_authority_ok",
    "preflight/stages/01_detached_focused_tests_162_ok",
    "preflight/stages/02_tf514_gdn_routes_static_ok",
    "preflight/stages/03_r39_primary_tests_13_ok",
    "preflight/stages/04_primary_code_bytecode_absent",
    "preflight/tf514-gdn-route-static.json",
}

PRIMARY_EXACT_LOGS = {
    "aggregate-final-audit.json",
    "aggregate.log",
    "code-integrity.log",
    "code-terminal-integrity.log",
    "focused-tests.log",
    "gpu-assignment-receipt-build.json",
    "gpu-assignment-terminal-build.json",
    "manifest-build.json",
    "model-artifact-integrity.log",
    "model-artifact-terminal-integrity.log",
    "model-load-lease-keeper.log",
    "model-weight-integrity.log",
    "model-weight-terminal-integrity.log",
    "private-model-view-materialization.json",
    "raw-artifact-integrity.log",
    "raw-artifact-terminal-integrity.log",
    "receipt-build.json",
    "rr2-source-rebuild.json",
    "run-id-terminal-audit.json",
    "scientific-artifact-integrity.log",
    "static-final-audit.json",
    "static.log",
    *(f"shard-rank-{rank}.log" for rank in range(8)),
}

PRIMARY_EXACT_STAGES = {
    "00_start",
    "01_focused_tests_ok",
    "02_run_identity_bound",
    "02_static_preregistration_ok",
    "03_formal_gpu_preflight_ok",
    "03_private_model_view_ok",
    "04_eight_rank_shards_ok",
    "05_detached_raw_receipts_ok",
    "06_blind_aggregate_ok",
    "99_done",
}

PRIVATE_MODEL_MANIFEST_FIELDS = {
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
PRIVATE_MODEL_ROW_FIELDS = {
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
PRIMARY_INVOCATION_FIELDS = {
    "schema_version",
    "rank",
    "runner_sha256",
    "runner_argv",
    "primary_shard_sha256",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["payload_sha256"] = None
    payload["payload_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def _seal_ok(value: Mapping[str, Any]) -> bool:
    observed = value.get("payload_sha256")
    payload = dict(value)
    payload["payload_sha256"] = None
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return _is_sha256(observed) and observed == expected


def validate_root(root: Path) -> Path:
    """Reject a symlinked or non-canonical root before resolving it."""

    lexical = Path(os.path.abspath(os.fspath(Path(root))))
    metadata = os.lstat(lexical)  # Deliberately precedes every resolve call.
    if stat.S_ISLNK(metadata.st_mode):
        raise RuntimeError("tree root symlink forbidden")
    if not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("tree root is not an exact directory")
    canonical = lexical.resolve(strict=True)
    if canonical != lexical:
        raise RuntimeError("canonical tree root differs from lexical absolute root")
    return lexical


def _validate_relative(relative: str, *, label: str) -> str:
    if type(relative) is not str or any(character in relative for character in ("\x00", "\n", "\r")):
        raise RuntimeError(f"{label} contains a forbidden control character")
    pure = PurePosixPath(relative)
    if not relative or pure.is_absolute() or relative in {".", ".."} or ".." in pure.parts:
        raise RuntimeError(f"{label} is not a normalized strict relative path")
    normalized = pure.as_posix()
    if normalized != relative or any(part in {"", ".", ".."} for part in pure.parts):
        raise RuntimeError(f"{label} is not a normalized strict relative path")
    return normalized


def normalize_output_path(root: Path, output: Path) -> tuple[Path, str]:
    """Normalize and contain an output before any caller performs a write."""

    canonical_root = validate_root(root)
    raw = Path(output)
    if ".." in raw.parts:
        raise RuntimeError("output path contains forbidden dotdot component")
    lexical = Path(os.path.abspath(os.fspath(raw)))
    try:
        relative_path = lexical.relative_to(canonical_root)
    except ValueError as error:
        raise RuntimeError("output path must be strictly inside tree root") from error
    if not relative_path.parts:
        raise RuntimeError("output path must be strictly inside tree root")
    relative = _validate_relative(relative_path.as_posix(), label="output path")
    parent_metadata = os.lstat(lexical.parent)
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise RuntimeError("output parent must be an exact existing directory")
    canonical_parent = lexical.parent.resolve(strict=True)
    if canonical_parent != lexical.parent or (
        canonical_parent != canonical_root and canonical_root not in canonical_parent.parents
    ):
        raise RuntimeError("output parent canonical containment drift")
    return lexical, relative


def _normalize_existing_regular(root: Path, path: Path, *, label: str) -> tuple[Path, str]:
    lexical, relative = normalize_output_path(root, path)
    metadata = os.lstat(lexical)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RuntimeError(f"{label} must be an exact single-link regular file")
    if lexical.resolve(strict=True) != lexical:
        raise RuntimeError(f"{label} canonical path drift")
    return lexical, relative


def ensure_output_absent(root: Path, output: Path) -> tuple[Path, str]:
    lexical, relative = normalize_output_path(root, output)
    if os.path.lexists(lexical):
        raise FileExistsError("terminal output overwrite or special node")
    return lexical, relative


def _read_regular(path: Path) -> tuple[bytes, os.stat_result]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError("capture/terminal tree hardlink or non-regular file forbidden")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(descriptor)
        identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if identity_before != identity_after or len(data) != after.st_size:
            raise RuntimeError("capture/terminal regular file changed during read")
        return data, after
    finally:
        os.close(descriptor)


def lexical_tree(root: Path, excluded: set[str] | None = None) -> dict[str, dict[str, object]]:
    canonical_root = validate_root(root)
    excluded = excluded or set()
    normalized_excluded = {_validate_relative(item, label="excluded path") for item in excluded}
    result: dict[str, dict[str, object]] = {}

    def walk(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                path = directory / entry.name
                relative = path.relative_to(canonical_root).as_posix()
                if relative in normalized_excluded:
                    continue
                metadata = entry.stat(follow_symlinks=False)
                mode = metadata.st_mode
                if stat.S_ISLNK(mode):
                    raise RuntimeError("capture/terminal tree symlink forbidden")
                if stat.S_ISDIR(mode):
                    result[relative] = {"kind": "directory"}
                    walk(path)
                elif stat.S_ISREG(mode):
                    data, exact = _read_regular(path)
                    if (metadata.st_dev, metadata.st_ino) != (exact.st_dev, exact.st_ino):
                        raise RuntimeError("capture/terminal file identity changed during traversal")
                    result[relative] = {
                        "kind": "regular",
                        "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                else:
                    raise RuntimeError("capture/terminal tree special node forbidden")

    walk(canonical_root)
    return result


def _json_schema(value: object) -> dict[str, object]:
    if type(value) is dict:
        return {
            "format": "json",
            "top_level_type": "object",
            "exact_fields": sorted(value),
            "schema_version": value.get("schema_version"),
        }
    if type(value) is list:
        return {"format": "json", "top_level_type": "array", "length": len(value)}
    return {"format": "json", "top_level_type": type(value).__name__}


def _content_schema(relative: str, data: bytes) -> dict[str, object]:
    if data == b"":
        return {"format": "empty"}
    if relative.endswith(".json"):
        try:
            return _json_schema(json.loads(data.decode("utf-8", errors="strict")))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"JSON terminal file is not strict JSON: {relative}") from error
    if relative.endswith(".sha256"):
        try:
            lines = data.decode("ascii", errors="strict").splitlines()
        except UnicodeDecodeError as error:
            raise RuntimeError(f"SHA ledger is not ASCII: {relative}") from error
        if len(lines) == 1 and _is_sha256(lines[0]):
            return {"format": "sha256-digest"}
        for line in lines:
            parts = line.split("  ", 1)
            if len(parts) != 2 or not _is_sha256(parts[0]):
                raise RuntimeError(f"SHA ledger row schema drift: {relative}")
            if not parts[1] or "\x00" in parts[1]:
                raise RuntimeError(f"SHA ledger target schema drift: {relative}")
        return {"format": "sha256-ledger", "row_count": len(lines)}
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return {"format": "opaque-bytes", "suffix": Path(relative).suffix}
    return {
        "format": "utf8-text",
        "suffix": Path(relative).suffix,
        "line_count": len(text.splitlines()),
    }


def _snapshot_nodes(root: Path, nodes: Mapping[str, Mapping[str, object]]) -> dict[str, dict[str, object]]:
    canonical_root = validate_root(root)
    snapshot: dict[str, dict[str, object]] = {}
    for relative, descriptor in sorted(nodes.items()):
        _validate_relative(relative, label="expected path")
        if descriptor == {"kind": "directory"}:
            snapshot[relative] = {"kind": "directory"}
            continue
        _require(set(descriptor) == {"kind", "bytes", "sha256"}, "regular node exact schema drift")
        data, _ = _read_regular(canonical_root / relative)
        _require(len(data) == descriptor["bytes"], "regular node bytes changed during schema read")
        _require(hashlib.sha256(data).hexdigest() == descriptor["sha256"], "regular node hash changed during schema read")
        snapshot[relative] = {
            "kind": "regular",
            "bytes": len(data),
            "sha256": descriptor["sha256"],
            "content_schema": _content_schema(relative, data),
        }
    return snapshot


def publish_json_exclusive(
    root: Path,
    output: Path,
    value: Mapping[str, Any],
    *,
    expected_fields: set[str] | None = None,
) -> Path:
    path, _ = ensure_output_absent(root, output)
    if expected_fields is not None and set(value) != expected_fields:
        raise RuntimeError("terminal JSON exact schema drift")
    payload = _canonical_json_bytes(value)
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def publish_empty_exclusive(root: Path, output: Path) -> Path:
    path, _ = ensure_output_absent(root, output)
    with path.open("xb") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _with_parent_directories(paths: set[str]) -> set[str]:
    result = set(paths)
    for relative in tuple(paths):
        pure = PurePosixPath(_validate_relative(relative, label="formal expected path"))
        for parent in pure.parents:
            if parent == PurePosixPath("."):
                break
            result.add(parent.as_posix())
    return result


def _regular_json(root: Path, relative: str, *, label: str) -> object:
    normalized = _validate_relative(relative, label=label)
    data, _ = _read_regular(root / normalized)
    try:
        return json.loads(data.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not strict JSON") from error


def _sha_ledger_paths(
    root: Path,
    nodes: Mapping[str, Mapping[str, object]],
    *,
    ledger_relative: str,
    base_relative: str,
    allowed_target: Any,
) -> set[str]:
    data, _ = _read_regular(root / ledger_relative)
    try:
        lines = data.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError(f"formal path authority ledger is not ASCII: {ledger_relative}") from error
    _require(lines, f"formal path authority ledger is empty: {ledger_relative}")
    targets: list[str] = []
    expected: set[str] = set()
    for line in lines:
        parts = line.split("  ", 1)
        _require(len(parts) == 2 and _is_sha256(parts[0]), f"formal path authority ledger row drift: {ledger_relative}")
        target = _validate_relative(parts[1], label=f"{ledger_relative} target")
        _require(allowed_target(target), f"formal path authority ledger target outside exact producer schema: {target}")
        full = f"{base_relative}/{target}"
        _require(full in nodes and nodes[full].get("kind") == "regular", f"formal path authority target absent/non-regular: {full}")
        _require(nodes[full].get("sha256") == parts[0], f"formal path authority target hash drift: {full}")
        targets.append(target)
        expected.add(full)
    _require(len(targets) == len(set(targets)), f"duplicate formal path authority target: {ledger_relative}")
    return expected


def _expected_primary_pycache_paths(
    root: Path, nodes: Mapping[str, Mapping[str, object]]
) -> set[str]:
    """Bind the redirected CPython cache exactly to the frozen primary sources."""

    ledger_relative = "primary/preregistration/code.sha256"
    data, _ = _read_regular(root / ledger_relative)
    try:
        lines = data.decode("ascii", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError("primary code ledger is not ASCII") from error
    _require(len(lines) == 34, "primary code ledger row count drift")
    targets: list[str] = []
    for line in lines:
        parts = line.split("  ", 1)
        _require(len(parts) == 2 and _is_sha256(parts[0]), "primary code ledger row drift")
        target = parts[1]
        _require(target.startswith("./"), "primary code ledger target prefix drift")
        relative = _validate_relative(target[2:], label="primary code ledger target")
        _require("/" not in relative, "primary code ledger target must be flat")
        targets.append(relative)
    _require(len(targets) == len(set(targets)), "duplicate primary code ledger target")
    sources = sorted(target for target in targets if target.endswith(".py"))
    _require(len(sources) == 31, "primary Python source authority count drift")
    expected_files = {
        f"{PRIMARY_CODE_PYCACHE_PREFIX}/{source[:-3]}{PRIMARY_PYCACHE_TAG}"
        for source in sources
    }
    _require(len(expected_files) == 31, "primary pycache projection count drift")
    for relative in expected_files:
        descriptor = nodes.get(relative)
        _require(
            type(descriptor) is dict
            and descriptor.get("kind") == "regular"
            and type(descriptor.get("bytes")) is int
            and descriptor["bytes"] > 16,
            f"primary pycache authorized file absent/invalid: {relative}",
        )
    expected = {
        relative
        for relative in _with_parent_directories(expected_files)
        if relative == "primary/pycache" or relative.startswith("primary/pycache/")
    }
    observed = {
        relative
        for relative in nodes
        if relative == "primary/pycache" or relative.startswith("primary/pycache/")
    }
    _require(observed == expected, "primary pycache exact source projection drift")
    return expected


def _expected_primary_paths(root: Path, nodes: Mapping[str, Mapping[str, object]]) -> set[str]:
    fixed = {
        "primary",
        "primary/logs",
        "primary/model-view",
        "primary/preregistration",
        "primary/pycache",
        "primary/raw",
        "primary/raw/shards",
        "primary/receipts",
        "primary/stages",
        "primary/forkaudit-summary.json",
        "primary/gpus-before.csv",
        "primary/scientific-artifacts.sha256",
        *(f"primary/logs/{name}" for name in PRIMARY_EXACT_LOGS),
        *(f"primary/stages/{name}" for name in PRIMARY_EXACT_STAGES),
    }
    observed_primary_roots = {
        relative
        for relative in nodes
        if relative.startswith("primary/") and relative.count("/") == 1
    }
    permitted_primary_roots = {
        "primary/logs",
        "primary/model-view",
        "primary/preregistration",
        "primary/pycache",
        "primary/raw",
        "primary/receipts",
        "primary/stages",
        "primary/forkaudit-summary.json",
        "primary/gpus-before.csv",
        "primary/scientific-artifacts.sha256",
    }
    _require(
        observed_primary_roots <= permitted_primary_roots,
        "formal result exact expected path whitelist drift in primary root",
    )

    scientific = _sha_ledger_paths(
        root,
        nodes,
        ledger_relative="primary/scientific-artifacts.sha256",
        base_relative="primary",
        allowed_target=lambda target: target == "forkaudit-summary.json"
        or target.startswith("preregistration/")
        or target.startswith("raw/")
        or target.startswith("receipts/"),
    )
    _require("primary/forkaudit-summary.json" in scientific, "primary summary absent from scientific path authority")

    manifest_relative = "primary/preregistration/private-model-view-manifest.json"
    _require(manifest_relative in scientific, "private model manifest absent from scientific path authority")
    manifest = _regular_json(root, manifest_relative, label="private model path authority")
    _require(type(manifest) is dict and set(manifest) == PRIVATE_MODEL_MANIFEST_FIELDS, "private model manifest exact schema drift")
    _require(
        manifest["schema_version"] == "qcomem-forkaudit-private-model-view-v1"
        and manifest["copy_policy"] == "ficlone-then-byte-copy;hardlink-and-symlink-forbidden"
        and manifest["all_source_and_view_inodes_distinct"] is True
        and manifest["all_view_files_regular"] is True
        and manifest["all_view_files_read_only"] is True
        and manifest["generated_before_candidate_outputs"] is True
        and manifest["cuda_initialized"] is False,
        "private model manifest identity/closure drift",
    )
    rows = manifest["rows"]
    _require(type(rows) is list and type(manifest["file_count"]) is int and manifest["file_count"] == len(rows), "private model manifest file count drift")
    model_files: set[str] = set()
    model_relatives: list[str] = []
    weight_rows = 0
    for row in rows:
        _require(type(row) is dict and set(row) == PRIVATE_MODEL_ROW_FIELDS, "private model manifest row exact schema drift")
        relative = _validate_relative(row["relative_path"], label="private model manifest row path")
        _require(type(row["ledger_roles"]) is list and set(row["ledger_roles"]) <= {"model_artifact", "model_weight"} and row["ledger_roles"], "private model manifest ledger role drift")
        _require(_is_sha256(row["declared_sha256"]) and type(row["bytes"]) is int and row["bytes"] >= 0, "private model manifest digest/bytes drift")
        _require(row["copy_mode"] in {"ficlone", "byte-copy"} and row["source_and_view_inode_distinct"] is True, "private model manifest copy closure drift")
        full = f"primary/model-view/{relative}"
        _require(
            full in nodes
            and nodes[full].get("kind") == "regular"
            and nodes[full].get("bytes") == row["bytes"]
            and nodes[full].get("sha256") == row["declared_sha256"],
            f"private model exact file whitelist/hash drift: {full}",
        )
        model_files.add(full)
        model_relatives.append(relative)
        weight_rows += "model_weight" in row["ledger_roles"]
    _require(len(model_relatives) == len(set(model_relatives)), "duplicate private model manifest path")
    _require(type(manifest["weight_file_count"]) is int and manifest["weight_file_count"] == weight_rows == 14, "private model weight count drift")
    pycache = _expected_primary_pycache_paths(root, nodes)
    return _with_parent_directories(fixed | scientific | model_files | pycache)


def _expected_formal_binding_paths(
    root: Path, nodes: Mapping[str, Mapping[str, object]]
) -> set[str]:
    def allowed(target: str) -> bool:
        if target == "formal-aggregate.json":
            return True
        parts = PurePosixPath(target).parts
        if len(parts) < 2 or parts[0] not in {f"rank-{rank}" for rank in range(8)}:
            return False
        return parts[1] in {
            "negative-controls.json",
            "primary-compiled-dispatch-receipt.json",
            "primary-shard-replay.json",
            "replay.json",
            "runtime-cache",
            "source-snapshot",
            "source-snapshot-manifest.json",
        }

    ledger = _sha_ledger_paths(
        root,
        nodes,
        ledger_relative="formal-binding/terminal-files.sha256",
        base_relative="formal-binding",
        allowed_target=allowed,
    )
    mandatory = {"formal-binding/formal-aggregate.json"}
    for rank in range(8):
        mandatory.update(
            {
                f"formal-binding/rank-{rank}/negative-controls.json",
                f"formal-binding/rank-{rank}/primary-compiled-dispatch-receipt.json",
                f"formal-binding/rank-{rank}/primary-shard-replay.json",
                f"formal-binding/rank-{rank}/replay.json",
                f"formal-binding/rank-{rank}/source-snapshot-manifest.json",
            }
        )
    _require(mandatory <= ledger, "formal binding mandatory exact path absent from terminal ledger")
    return _with_parent_directories(
        ledger
        | {
            "formal-binding",
            "formal-binding/COMPLETE",
            "formal-binding/terminal-files.sha256",
        }
    )


def _expected_capture_paths(
    root: Path,
    formal_binding: set[str],
    nodes: Mapping[str, Mapping[str, object]],
) -> set[str]:
    expected: set[str] = {"compiled-dispatch-capture"}
    for rank in range(8):
        prefix = f"formal-binding/rank-{rank}/runtime-cache/"
        mapped = {
            f"compiled-dispatch-capture/rank-{rank}/runtime-cache/{relative[len(prefix):]}"
            for relative in formal_binding
            if relative.startswith(prefix)
        }
        _require(mapped, f"rank {rank} runtime-cache exact path authority absent")
        for capture_relative in mapped:
            suffix = capture_relative.split(f"compiled-dispatch-capture/rank-{rank}/runtime-cache/", 1)[1]
            formal_relative = f"formal-binding/rank-{rank}/runtime-cache/{suffix}"
            _require(
                capture_relative in nodes and nodes[capture_relative] == nodes[formal_relative],
                f"rank {rank} capture/formal runtime-cache projection drift: {suffix}",
            )
        expected.update(mapped)
        capture_receipt = f"compiled-dispatch-capture/rank-{rank}/raw/primary-compiled-dispatch-receipt.json"
        formal_receipt = f"formal-binding/rank-{rank}/primary-compiled-dispatch-receipt.json"
        _require(
            capture_receipt in nodes and nodes[capture_receipt] == nodes[formal_receipt],
            f"rank {rank} capture/formal receipt projection drift",
        )
        receipt_bytes, _ = _read_regular(root / formal_receipt)
        try:
            receipt = json.loads(receipt_bytes.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"rank {rank} formal receipt is not strict JSON") from error
        _require(type(receipt) is dict, f"rank {rank} formal receipt object drift")
        binding = receipt.get("execution_binding")
        _require(type(binding) is dict, f"rank {rank} formal receipt execution binding drift")

        invocation_relative = f"compiled-dispatch-capture/rank-{rank}/invocation.json"
        _require(invocation_relative in nodes, f"rank {rank} invocation path absent")
        invocation_bytes, _ = _read_regular(root / invocation_relative)
        try:
            invocation = json.loads(invocation_bytes.decode("utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f"rank {rank} invocation is not strict JSON") from error
        _require(
            type(invocation) is dict and set(invocation) == PRIMARY_INVOCATION_FIELDS,
            f"rank {rank} invocation exact schema drift",
        )
        _require(
            invocation_bytes == _canonical_json_bytes(invocation),
            f"rank {rank} invocation canonical JSON drift",
        )
        _require(
            invocation["schema_version"] == "forkaudit-r39-primary-rank-invocation-v2",
            f"rank {rank} invocation schema version drift",
        )
        _require(type(invocation["rank"]) is int and invocation["rank"] == rank, f"rank {rank} invocation rank drift")
        _require(_is_sha256(invocation["runner_sha256"]), f"rank {rank} invocation runner SHA drift")
        _require(_is_sha256(invocation["primary_shard_sha256"]), f"rank {rank} invocation shard SHA drift")
        _require(
            type(invocation["runner_argv"]) is list
            and all(type(item) is str for item in invocation["runner_argv"]),
            f"rank {rank} invocation argv type drift",
        )
        argv_sha256 = hashlib.sha256(
            json.dumps(
                invocation["runner_argv"],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("utf-8")
        ).hexdigest()
        _require(
            invocation["runner_sha256"] == binding.get("runner_sha256")
            and invocation["runner_argv"] == binding.get("runner_argv")
            and invocation["primary_shard_sha256"] == binding.get("primary_shard_sha256")
            and argv_sha256 == binding.get("runner_argv_sha256"),
            f"rank {rank} invocation/formal receipt binding drift",
        )
        expected.update({capture_receipt, invocation_relative})
    return _with_parent_directories(expected)


def _expected_r40_paths() -> set[str]:
    controlled = {"r40-clean-binding"}
    for rank in range(8):
        controlled.update(
            {
                f"r40-clean-binding/rank-{rank}",
                f"r40-clean-binding/rank-{rank}/raw",
                f"r40-clean-binding/rank-{rank}/raw/real-binding.json",
                f"r40-clean-binding/rank-{rank}/raw/global-absence.json",
            }
        )
    controlled.update(
        {
            "r40-formal",
            "r40-formal/cuda-smoke.json",
            "r40-formal/aggregate.json",
        }
    )
    return controlled


def _profile_r40_formal(nodes: Mapping[str, Mapping[str, object]], root: Path) -> None:
    top_levels = {relative.split("/", 1)[0] for relative in nodes}
    _require(top_levels == FORMAL_TOP_LEVELS, "formal result top-level exact whitelist drift")
    primary_roots = {
        relative
        for relative in nodes
        if relative.startswith("primary/") and relative.count("/") == 1
    }
    _require(
        primary_roots
        <= {
            "primary/logs",
            "primary/model-view",
            "primary/preregistration",
            "primary/pycache",
            "primary/raw",
            "primary/receipts",
            "primary/stages",
            "primary/forkaudit-summary.json",
            "primary/gpus-before.csv",
            "primary/scientific-artifacts.sha256",
        },
        "formal result exact expected path whitelist drift in primary root",
    )
    controlled = _expected_r40_paths()
    observed_controlled = {
        relative
        for relative in nodes
        if relative == "r40-clean-binding"
        or relative.startswith("r40-clean-binding/")
        or relative == "r40-formal"
        or relative.startswith("r40-formal/")
    }
    _require(observed_controlled == controlled, "R40 controlled terminal path whitelist drift")
    for relative, fields, version in (
        ("r40-formal/cuda-smoke.json", CUDA_FIELDS, "forkaudit-r40-v16-cuda-smoke-v1"),
        ("r40-formal/aggregate.json", AGGREGATE_FIELDS, "forkaudit-r40-v32-borrowed-transition-aggregate-v1"),
    ):
        value = json.loads((root / relative).read_text(encoding="utf-8", errors="strict"))
        _require(type(value) is dict and set(value) == fields, f"{relative} exact schema drift")
        _require(value.get("schema_version") == version, f"{relative} schema version drift")
    formal_binding = _expected_formal_binding_paths(root, nodes)
    expected = (
        PREFLIGHT_EXACT_PATHS
        | _expected_primary_paths(root, nodes)
        | formal_binding
        | _expected_capture_paths(root, formal_binding, nodes)
        | controlled
    )
    _require(set(nodes) == expected, "formal result exact expected path whitelist drift")


def formal_expected_existing_paths(root: Path) -> list[str]:
    """Reconstruct the exact formal path plan from fixed paths and bound manifests."""

    canonical_root = validate_root(root)
    observed = lexical_tree(canonical_root)
    _profile_r40_formal(observed, canonical_root)
    return sorted(observed)


def prepare_terminal_expectation(
    root: Path,
    output: Path,
    *,
    terminal_tree_output: Path,
    complete_output: Path,
    source_ledger_sha256: str,
    profile: str = "r40-v16-formal",
    expected_existing_paths: Sequence[str] | None = None,
) -> dict[str, Any]:
    canonical_root = validate_root(root)
    output_path, output_relative = ensure_output_absent(canonical_root, output)
    _, tree_relative = ensure_output_absent(canonical_root, terminal_tree_output)
    _, complete_relative = ensure_output_absent(canonical_root, complete_output)
    _require(len({output_relative, tree_relative, complete_relative}) == 3, "terminal output paths must be distinct")
    _require(_is_sha256(source_ledger_sha256), "source ledger SHA-256 drift")

    observed = lexical_tree(canonical_root)
    if expected_existing_paths is not None:
        normalized = sorted(_validate_relative(item, label="expected existing path") for item in expected_existing_paths)
        _require(len(normalized) == len(set(normalized)), "duplicate expected existing path")
        _require(set(observed) == set(normalized), "existing terminal path exact whitelist drift")
    if profile == "r40-v16-formal":
        _profile_r40_formal(observed, canonical_root)
    elif profile != "fixture":
        raise RuntimeError("unknown terminal expectation profile")

    expected_nodes = _snapshot_nodes(canonical_root, observed)
    self_schema = {
        "format": "json",
        "top_level_type": "object",
        "exact_fields": sorted(EXPECTATION_FIELDS),
        "schema_version": EXPECTATION_SCHEMA,
    }
    expected_nodes[output_relative] = {
        "kind": "regular",
        "self_sealed": True,
        "content_schema": self_schema,
    }
    expected_nodes[complete_relative] = {
        "kind": "regular",
        "bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
        "content_schema": {"format": "empty"},
    }
    expected_paths = sorted(expected_nodes)
    directory_count = sum(row["kind"] == "directory" for row in expected_nodes.values())
    regular_count = len(expected_nodes) - directory_count
    payload = _seal(
        {
            "schema_version": EXPECTATION_SCHEMA,
            "status": STATUS,
            "root": ".",
            "terminal_tree_output": tree_relative,
            "complete_output": complete_relative,
            "terminal_source_rehash": True,
            "read_only_staging": True,
            "source_ledger_sha256": source_ledger_sha256,
            "expected_paths": expected_paths,
            "expected_node_count": len(expected_nodes),
            "expected_regular_file_count": regular_count,
            "expected_directory_count": directory_count,
            "expected_nodes": expected_nodes,
            "payload_sha256": None,
        }
    )
    publish_json_exclusive(canonical_root, output_path, payload, expected_fields=EXPECTATION_FIELDS)
    return payload


def _validate_expectation(value: object, *, expectation_relative: str, tree_relative: str) -> dict[str, Any]:
    _require(type(value) is dict and set(value) == EXPECTATION_FIELDS, "terminal expectation exact schema drift")
    value = dict(value)
    _require(value["schema_version"] == EXPECTATION_SCHEMA and value["status"] == STATUS, "terminal expectation identity drift")
    _require(value["root"] == "." and value["terminal_tree_output"] == tree_relative, "terminal expectation root/output drift")
    _require(value["terminal_source_rehash"] is True and value["read_only_staging"] is True, "terminal source closure flags drift")
    _require(_is_sha256(value["source_ledger_sha256"]) and _seal_ok(value), "terminal expectation hash/seal drift")
    paths = value["expected_paths"]
    nodes = value["expected_nodes"]
    _require(type(paths) is list and all(type(item) is str for item in paths), "expected path whitelist type drift")
    _require(type(nodes) is dict and paths == sorted(nodes) and len(paths) == len(set(paths)), "expected path whitelist/node mismatch")
    _require(expectation_relative in nodes and value["complete_output"] in nodes, "terminal expectation mandatory path absent")
    for relative, descriptor in nodes.items():
        _validate_relative(relative, label="expected terminal path")
        _require(type(descriptor) is dict, "expected node descriptor type drift")
        if descriptor.get("kind") == "directory":
            _require(set(descriptor) == {"kind"}, "expected directory exact schema drift")
        elif descriptor.get("kind") == "regular":
            if relative == expectation_relative:
                _require(
                    set(descriptor) == {"kind", "self_sealed", "content_schema"}
                    and descriptor["self_sealed"] is True,
                    "self expectation descriptor exact schema drift",
                )
            else:
                _require(set(descriptor) == {"kind", "bytes", "sha256", "content_schema"}, "expected regular exact schema drift")
                _require(type(descriptor["bytes"]) is int and descriptor["bytes"] >= 0, "expected regular byte count type drift")
                _require(_is_sha256(descriptor["sha256"]), "expected regular SHA-256 drift")
        else:
            raise RuntimeError("expected node kind drift")
    directories = sum(row["kind"] == "directory" for row in nodes.values())
    regular = len(nodes) - directories
    _require(type(value["expected_node_count"]) is int and value["expected_node_count"] == len(nodes), "expected node count drift")
    _require(type(value["expected_regular_file_count"]) is int and value["expected_regular_file_count"] == regular, "expected regular-file count drift")
    _require(type(value["expected_directory_count"]) is int and value["expected_directory_count"] == directories, "expected directory count drift")
    return value


def write_terminal_ledger(root: Path, output: Path, expectation: Path) -> dict[str, Any]:
    canonical_root = validate_root(root)
    output_path, output_relative = ensure_output_absent(canonical_root, output)
    expectation_path, expectation_relative = _normalize_existing_regular(
        canonical_root, expectation, label="terminal expectation"
    )
    value = _validate_expectation(
        json.loads(expectation_path.read_text(encoding="utf-8", errors="strict")),
        expectation_relative=expectation_relative,
        tree_relative=output_relative,
    )
    expected_nodes: Mapping[str, Mapping[str, object]] = value["expected_nodes"]
    observed = lexical_tree(canonical_root, {output_relative})
    _require(set(observed) == set(expected_nodes), "terminal path exact whitelist drift")
    for relative, expected in expected_nodes.items():
        actual = observed[relative]
        if expected["kind"] == "directory":
            _require(actual == {"kind": "directory"}, "terminal directory exact schema drift")
            continue
        _require(set(actual) == {"kind", "bytes", "sha256"} and actual["kind"] == "regular", "terminal regular node exact schema drift")
        data, _ = _read_regular(canonical_root / relative)
        schema = _content_schema(relative, data)
        if relative == expectation_relative:
            _require(schema == expected["content_schema"], "terminal expectation per-file exact schema drift")
            _require(_seal_ok(json.loads(data.decode("utf-8", errors="strict"))), "terminal expectation self seal drift")
        else:
            _require(
                actual["bytes"] == expected["bytes"]
                and actual["sha256"] == expected["sha256"]
                and schema == expected["content_schema"],
                f"terminal file bytes/hash/exact schema drift: {relative}",
            )

    snapshotted = _snapshot_nodes(canonical_root, observed)
    directories = sum(row["kind"] == "directory" for row in snapshotted.values())
    regular = len(snapshotted) - directories
    _require(len(snapshotted) == value["expected_node_count"], "terminal observed node count drift")
    _require(regular == value["expected_regular_file_count"], "terminal observed regular-file count drift")
    _require(directories == value["expected_directory_count"], "terminal observed directory count drift")
    payload = {
        "schema_version": TREE_SCHEMA,
        "status": STATUS,
        "root": ".",
        "excluded_output": output_relative,
        "expected_paths": value["expected_paths"],
        "expected_node_count": len(snapshotted),
        "expected_regular_file_count": regular,
        "expected_directory_count": directories,
        "final_node_count": len(snapshotted) + 1,
        "final_regular_file_count": regular + 1,
        "final_directory_count": directories,
        "nodes": snapshotted,
    }
    publish_json_exclusive(canonical_root, output_path, payload, expected_fields=TREE_FIELDS)
    after = lexical_tree(canonical_root)
    expected_after = {
        relative: {key: descriptor[key] for key in ("kind", "bytes", "sha256") if key in descriptor}
        for relative, descriptor in snapshotted.items()
    }
    data, _ = _read_regular(output_path)
    expected_after[output_relative] = {
        "kind": "regular",
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    _require(after == expected_after, "terminal lexical tree changed during exclusive closure publication")
    _require(
        len(after) == payload["final_node_count"]
        and sum(row["kind"] == "regular" for row in after.values()) == payload["final_regular_file_count"]
        and sum(row["kind"] == "directory" for row in after.values()) == payload["final_directory_count"],
        "terminal final exact node/file/directory counts drift",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--root", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--terminal-tree-output", type=Path, required=True)
    prepare.add_argument("--complete-output", type=Path, required=True)
    prepare.add_argument("--source-ledger-sha256", required=True)
    prepare.add_argument("--profile", choices=("r40-v16-formal", "fixture"), default="r40-v16-formal")
    prepare.add_argument("--expected-existing-path", action="append")
    expected_paths = subparsers.add_parser("expected-paths")
    expected_paths.add_argument("--root", type=Path, required=True)
    complete = subparsers.add_parser("complete")
    complete.add_argument("--root", type=Path, required=True)
    complete.add_argument("--output", type=Path, required=True)
    close = subparsers.add_parser("close")
    close.add_argument("--root", type=Path, required=True)
    close.add_argument("--output", type=Path, required=True)
    close.add_argument("--expectation", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_terminal_expectation(
            args.root,
            args.output,
            terminal_tree_output=args.terminal_tree_output,
            complete_output=args.complete_output,
            source_ledger_sha256=args.source_ledger_sha256,
            profile=args.profile,
            expected_existing_paths=args.expected_existing_path,
        )
    elif args.command == "expected-paths":
        for relative in formal_expected_existing_paths(args.root):
            sys.stdout.write(relative + "\n")
    elif args.command == "complete":
        publish_empty_exclusive(args.root, args.output)
    else:
        write_terminal_ledger(args.root, args.output, args.expectation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
