#!/usr/bin/env python3
"""Capture and verify fail-closed ForkAudit execution identity receipts.

The receipt separates immutable execution identity from runtime cache identity.
The former must be byte-identical before and after a run.  The latter may grow,
but every terminal cache artifact is enumerated and hashed.  No project module
is imported while building the receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "qcomem-forkaudit-execution-identity-v1"
WRITE_MASK = stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
COMPILED_SUFFIXES = {
    ".bin",
    ".cubin",
    ".fatbin",
    ".hsaco",
    ".o",
    ".ptx",
    ".so",
}
AUTOTUNE_MARKERS = ("autotun", "best_config", "best-config", "tuning")
SAFE_ENV_KEYS = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_CACHE_PATH",
    "CUDA_MODULE_LOADING",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONPYCACHEPREFIX",
    "TORCHINDUCTOR_CACHE_DIR",
    "TRITON_CACHE_DIR",
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> "None":
    raise SystemExit(f"ForkAudit execution identity rejected: {message}")


def atomic_json_write(path: Path, value: Any) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def parse_named_path(raw: str) -> tuple[str, Path]:
    name, separator, path = raw.partition("=")
    if not separator or not name or not path:
        fail(f"expected NAME=PATH, got {raw!r}")
    if any(character in name for character in ("/", "\\", "\n", "\r")):
        fail(f"unsafe receipt label: {name!r}")
    return name, Path(path)


def checked_tree(root_arg: Path, *, require_readonly: bool) -> dict[str, Any]:
    try:
        root_lstat = root_arg.lstat()
    except OSError as exc:
        fail(f"source root cannot be inspected: {exc}")
    if stat.S_ISLNK(root_lstat.st_mode):
        fail("source root itself is a symbolic link")
    if not stat.S_ISDIR(root_lstat.st_mode):
        fail("source root is not a directory")
    if require_readonly and root_lstat.st_mode & WRITE_MASK:
        fail("source root is writable")
    root = root_arg.resolve(strict=True)
    files: list[dict[str, Any]] = []
    directories: list[dict[str, Any]] = []

    def walk(directory: Path, relative_parent: Path) -> None:
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: os.fsencode(item.name))
        except OSError as exc:
            fail(f"cannot scan source path {relative_parent.as_posix() or '.'}: {exc}")
        for entry in entries:
            relative = relative_parent / entry.name
            relative_text = relative.as_posix()
            if any(c in relative_text for c in ("\\", "\n", "\r")):
                fail(f"unsupported source path spelling: {relative_text!r}")
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                fail(f"cannot stat source path {relative_text}: {exc}")
            mode = metadata.st_mode
            if stat.S_ISLNK(mode):
                fail(f"symbolic link in source tree: {relative_text}")
            if require_readonly and mode & WRITE_MASK:
                fail(f"writable source entry: {relative_text}")
            if "__pycache__" in relative.parts:
                fail(f"Python bytecode cache in source tree: {relative_text}")
            if stat.S_ISDIR(mode):
                directories.append(
                    {"path": relative_text, "mode": stat.S_IMODE(mode)}
                )
                walk(Path(entry.path), relative)
            elif stat.S_ISREG(mode):
                if relative.suffix in {".pyc", ".pyo"}:
                    fail(f"Python bytecode file in source tree: {relative_text}")
                path = Path(entry.path)
                before = path.stat()
                digest = sha256_file(path)
                after = path.stat()
                if (
                    before.st_dev != after.st_dev
                    or before.st_ino != after.st_ino
                    or before.st_size != after.st_size
                    or before.st_mtime_ns != after.st_mtime_ns
                    or before.st_mode != after.st_mode
                ):
                    fail(f"source entry changed while hashing: {relative_text}")
                files.append(
                    {
                        "path": relative_text,
                        "sha256": digest,
                        "size": before.st_size,
                        "mode": stat.S_IMODE(before.st_mode),
                    }
                )
            else:
                fail(f"non-regular entry in source tree: {relative_text}")

    walk(root, Path())
    closure = {
        "root": os.fspath(root),
        "root_mode": stat.S_IMODE(root_lstat.st_mode),
        "directories": directories,
        "files": files,
    }
    return {
        **closure,
        "closure_sha256": sha256_bytes(canonical_json_bytes(closure)),
        "file_count": len(files),
    }


def executable_identity(path_arg: Path) -> dict[str, Any]:
    configured = path_arg.absolute()
    try:
        configured_lstat = configured.lstat()
        resolved = configured.resolve(strict=True)
        resolved_stat = resolved.stat()
    except OSError as exc:
        fail(f"Python executable cannot be inspected: {exc}")
    if not stat.S_ISREG(resolved_stat.st_mode):
        fail("resolved Python executable is not a regular file")
    if not os.access(resolved, os.X_OK):
        fail("resolved Python executable is not executable")
    command = [os.fspath(configured), "-I", "-B", "-c", "import sys;print(sys.version)"]
    try:
        version = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        fail(f"Python executable identity probe failed: {exc}")
    return {
        "configured_path": os.fspath(configured),
        "configured_is_symlink": stat.S_ISLNK(configured_lstat.st_mode),
        "configured_symlink_target": (
            os.readlink(configured) if stat.S_ISLNK(configured_lstat.st_mode) else None
        ),
        "resolved_path": os.fspath(resolved),
        "sha256": sha256_file(resolved),
        "size": resolved_stat.st_size,
        "mode": stat.S_IMODE(resolved_stat.st_mode),
        "version": version,
    }


def distribution_identity() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            fail("installed distribution lacks a Name field")
        metadata_text = distribution.read_text("METADATA")
        record_text = distribution.read_text("RECORD")
        direct_url_text = distribution.read_text("direct_url.json")
        rows.append(
            {
                "name": name,
                "normalized_name": name.lower().replace("_", "-").replace(".", "-"),
                "version": distribution.version,
                "metadata_sha256": (
                    sha256_bytes(metadata_text.encode("utf-8"))
                    if metadata_text is not None
                    else None
                ),
                "record_sha256": (
                    sha256_bytes(record_text.encode("utf-8"))
                    if record_text is not None
                    else None
                ),
                "direct_url_sha256": (
                    sha256_bytes(direct_url_text.encode("utf-8"))
                    if direct_url_text is not None
                    else None
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["normalized_name"].encode("utf-8"),
            row["version"].encode("utf-8"),
            (row["record_sha256"] or "").encode("ascii"),
        )
    )
    return rows


def gpu_identity() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return {"status": "not_available", "rows": []}
    query = "index,uuid,name,memory.total,compute_cap,driver_version"
    try:
        output = subprocess.run(
            [executable, f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        fail(f"nvidia-smi identity probe failed: {exc}")
    rows = [line.strip() for line in output.splitlines() if line.strip()]
    return {"status": "recorded", "query": query, "rows": rows}


def environment_identity() -> dict[str, Any]:
    packages = distribution_identity()
    value = {
        "python_runtime": {
            "version": sys.version,
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "abiflags": getattr(sys, "abiflags", ""),
            "soabi": sysconfig.get_config_var("SOABI"),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "environment": {key: os.environ.get(key) for key in SAFE_ENV_KEYS},
        "installed_distributions": packages,
        "gpu_runtime": gpu_identity(),
    }
    return {**value, "identity_sha256": sha256_bytes(canonical_json_bytes(value))}


def command_identity(
    command_files: Iterable[tuple[str, Path]], command_template: str
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    labels: set[str] = set()
    for label, path_arg in command_files:
        if label in labels:
            fail(f"duplicate command-file label: {label}")
        labels.add(label)
        try:
            path = path_arg.resolve(strict=True)
            metadata = path.stat()
        except OSError as exc:
            fail(f"command file {label!r} cannot be inspected: {exc}")
        if not stat.S_ISREG(metadata.st_mode):
            fail(f"command file {label!r} is not regular")
        rows.append(
            {
                "label": label,
                "path": os.fspath(path),
                "sha256": sha256_file(path),
                "size": metadata.st_size,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
        )
    rows.sort(key=lambda row: row["label"].encode("utf-8"))
    value = {"files": rows, "template": command_template}
    return {**value, "identity_sha256": sha256_bytes(canonical_json_bytes(value))}


def cache_tree(label: str, root_arg: Path) -> dict[str, Any]:
    root = root_arg.absolute()
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        fail(f"cache root {label!r} is not a real directory")
    files: list[dict[str, Any]] = []
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort(key=os.fsencode)
        file_names.sort(key=os.fsencode)
        directory_path = Path(directory)
        for name in directory_names:
            path = directory_path / name
            if path.is_symlink():
                fail(f"symbolic link in cache {label!r}: {path.relative_to(root)}")
        for name in file_names:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            metadata = path.lstat()
            if not stat.S_ISREG(metadata.st_mode):
                fail(f"non-regular entry in cache {label!r}: {relative}")
            lowered = relative.lower()
            files.append(
                {
                    "path": relative,
                    "sha256": sha256_file(path),
                    "size": metadata.st_size,
                    "compiled_kernel_artifact": path.suffix.lower() in COMPILED_SUFFIXES,
                    "autotune_artifact": any(marker in lowered for marker in AUTOTUNE_MARKERS),
                }
            )
    identity = {"label": label, "root": os.fspath(root), "files": files}
    compiled_count = sum(bool(row["compiled_kernel_artifact"]) for row in files)
    autotune_count = sum(bool(row["autotune_artifact"]) for row in files)
    return {
        **identity,
        "identity_sha256": sha256_bytes(canonical_json_bytes(identity)),
        "file_count": len(files),
        "compiled_kernel_artifact_count": compiled_count,
        "autotune_artifact_count": autotune_count,
        "compiled_kernel_identity_status": (
            "recorded" if compiled_count else "no_compiled_artifact_exposed"
        ),
        "autotune_identity_status": (
            "recorded" if autotune_count else "no_explicit_autotune_artifact_exposed"
        ),
    }


def capture(args: argparse.Namespace) -> None:
    caches = [cache_tree(*parse_named_path(raw)) for raw in args.cache]
    if args.require_empty_caches and any(row["file_count"] for row in caches):
        fail("preregistration cache roots are not empty")
    static = {
        "source": checked_tree(
            Path(args.source_root), require_readonly=not args.allow_writable_source
        ),
        "executable": executable_identity(Path(args.python)),
        "environment": environment_identity(),
        "command": command_identity(
            (parse_named_path(raw) for raw in args.command_file),
            args.command_template,
        ),
    }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        **static,
        "static_identity_sha256": sha256_bytes(canonical_json_bytes(static)),
        "runtime_caches": caches,
        "runtime_cache_identity_sha256": sha256_bytes(canonical_json_bytes(caches)),
    }
    atomic_json_write(Path(args.output), receipt)


def load_receipt(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read execution identity receipt {path}: {exc}")
    if value.get("schema_version") != SCHEMA_VERSION:
        fail(f"execution identity schema mismatch in {path}")
    return value


def verify_stable(args: argparse.Namespace) -> None:
    before = load_receipt(Path(args.before))
    after = load_receipt(Path(args.after))
    static_fields = ("source", "executable", "environment", "command")
    mismatches = [name for name in static_fields if before.get(name) != after.get(name)]
    before_cache_files = sum(
        int(row.get("file_count", -1)) for row in before.get("runtime_caches", [])
    )
    if before_cache_files != 0:
        mismatches.append("preregistration_runtime_caches_not_empty")
    if before.get("static_identity_sha256") != after.get("static_identity_sha256"):
        mismatches.append("static_identity_sha256")
    if mismatches:
        fail(f"terminal execution identity drift: {sorted(set(mismatches))}")
    terminal_caches = after.get("runtime_caches")
    if not isinstance(terminal_caches, list) or not terminal_caches:
        fail("terminal runtime cache identity is absent")
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "execution_identity_stable",
        "static_identity_sha256": after["static_identity_sha256"],
        "terminal_runtime_cache_identity_sha256": after[
            "runtime_cache_identity_sha256"
        ],
        "terminal_cache_file_count": sum(row["file_count"] for row in terminal_caches),
        "compiled_kernel_artifact_count": sum(
            row["compiled_kernel_artifact_count"] for row in terminal_caches
        ),
        "autotune_artifact_count": sum(
            row["autotune_artifact_count"] for row in terminal_caches
        ),
        "autotune_identity_limitation": (
            None
            if any(row["autotune_artifact_count"] for row in terminal_caches)
            else "no explicit autotuning artifact was exposed in the bound cache roots"
        ),
    }
    atomic_json_write(Path(args.output), result)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    subparsers = result.add_subparsers(dest="command", required=True)
    capture_parser = subparsers.add_parser("capture")
    capture_parser.add_argument("--source-root", required=True)
    capture_parser.add_argument("--python", required=True)
    capture_parser.add_argument("--cache", action="append", required=True)
    capture_parser.add_argument("--command-file", action="append", default=[])
    capture_parser.add_argument("--command-template", required=True)
    capture_parser.add_argument("--output", required=True)
    capture_parser.add_argument("--require-empty-caches", action="store_true")
    capture_parser.add_argument("--allow-writable-source", action="store_true")
    capture_parser.set_defaults(function=capture)
    verify_parser = subparsers.add_parser("verify-stable")
    verify_parser.add_argument("--before", required=True)
    verify_parser.add_argument("--after", required=True)
    verify_parser.add_argument("--output", required=True)
    verify_parser.set_defaults(function=verify_stable)
    return result


def main() -> None:
    args = parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
