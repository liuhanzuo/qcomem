#!/usr/bin/env python3
"""Prepare and attest one dedicated immutable public Hugging Face snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any


MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODEL_ENDPOINT = "https://hf-mirror.com"
MODEL_TOKEN_POLICY = "public-token-false"
MODEL_TOKEN = False
HF_HUB_DISABLE_XET = "1"
MODEL_DOWNLOAD_TRANSPORT = "huggingface-hub-http-no-xet"
MARKER_NAME = ".r39-public-snapshot-source.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def acquisition_policy() -> dict[str, Any]:
    return {
        "endpoint": MODEL_ENDPOINT,
        "endpoint_is_explicit": True,
        "token": MODEL_TOKEN,
        "token_policy": MODEL_TOKEN_POLICY,
        "hf_hub_disable_xet": HF_HUB_DISABLE_XET,
        "transport": MODEL_DOWNLOAD_TRANSPORT,
        "resolved_revision_must_equal_requested_full_commit": True,
        "fresh_nonoverwriting_snapshot_required": True,
        "a_failed_partial_reuse_forbidden": True,
    }


def snapshot_marker(remote_files: list[str], resolved_revision: str) -> dict[str, Any]:
    require(resolved_revision == MODEL_REVISION, "mirror revision resolution drift")
    require(remote_files == sorted(set(remote_files)) and remote_files, "remote tree is not canonical")
    require(all(path and not Path(path).is_absolute() and ".." not in Path(path).parts for path in remote_files), "unsafe remote path")
    return {
        "repo_id": MODEL_ID,
        "requested_revision": MODEL_REVISION,
        "resolved_revision": resolved_revision,
        "model_acquisition": acquisition_policy(),
        "remote_file_count": len(remote_files),
        "remote_files": remote_files,
        "remote_files_sha256": sha256_bytes(canonical_bytes(remote_files)),
    }


def atomic_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to replace existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def all_file_paths(root: Path) -> list[Path]:
    require(root.is_dir() and not root.is_symlink(), "model root is not a regular directory")
    paths = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        require(not path.is_symlink(), f"model snapshot contains a symlink: {path}")
        if path.is_file():
            paths.append(path)
        else:
            require(path.is_dir(), f"model snapshot contains a non-file object: {path}")
    require(paths, "model snapshot has no files")
    return paths


def authority(root: Path) -> dict[str, Any]:
    marker_path = root / MARKER_NAME
    require(marker_path.is_file() and not marker_path.is_symlink(), "snapshot source marker absent")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    require(isinstance(marker, dict), "snapshot source marker is not an object")
    require(marker.get("repo_id") == MODEL_ID, "snapshot model ID drift")
    require(marker.get("requested_revision") == MODEL_REVISION, "snapshot requested revision drift")
    require(marker.get("resolved_revision") == MODEL_REVISION, "snapshot resolved revision drift")
    require(marker.get("model_acquisition") == acquisition_policy(), "snapshot acquisition policy drift")
    remote_files = marker.get("remote_files")
    require(isinstance(remote_files, list), "snapshot remote tree absent")
    require(
        marker == snapshot_marker([str(path) for path in remote_files], MODEL_REVISION),
        "snapshot source marker drift",
    )
    rows = []
    for path in all_file_paths(root):
        status = path.stat()
        require(stat.S_ISREG(status.st_mode), f"not a regular model file: {path}")
        require(status.st_mode & 0o222 == 0, f"writable model file: {path}")
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": status.st_size,
                "sha256": sha256_file(path),
                "mode": stat.S_IMODE(status.st_mode),
                "device": status.st_dev,
                "inode": status.st_ino,
                "ctime_ns": status.st_ctime_ns,
            }
        )
    repo_rows = [row for row in rows if not row["path"].startswith(".cache/")]
    require(
        {row["path"] for row in repo_rows} == {*remote_files, MARKER_NAME},
        "local repository tree differs from the resolved mirror tree",
    )
    require(any(row["path"] == "config.json" for row in repo_rows), "config.json absent")
    require(
        any(
            re.fullmatch(
                r"model(?:\.safetensors)?(?:-[0-9]+-of-[0-9]+)?\.safetensors",
                row["path"],
            )
            for row in repo_rows),
        "safetensors payload absent",
    )
    require(any(row["path"] == "model.safetensors.index.json" for row in repo_rows), "index absent")
    return {
        "schema_version": "r39-second-model-transfer-model-authority-v1",
        "repo_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "model_acquisition": {
            "policy": acquisition_policy(),
            "requested_revision": MODEL_REVISION,
            "resolved_revision": MODEL_REVISION,
            "remote_file_count": marker["remote_file_count"],
            "remote_files_sha256": marker["remote_files_sha256"],
            "source_marker_sha256": sha256_file(marker_path),
        },
        "all_local_files_hashed": True,
        "file_count": len(rows),
        "repo_file_count": len(repo_rows),
        "files": rows,
        "normalized_files_sha256": hashlib.sha256(canonical_bytes(rows)).hexdigest(),
    }


def make_read_only(root: Path) -> None:
    for path in all_file_paths(root):
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, mode & ~0o222)
    directories = [root, *[path for path in root.rglob("*") if path.is_dir()]]
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, (mode & ~0o222) | 0o555)


def prepare(root: Path) -> dict[str, Any]:
    require(MODEL_REVISION in root.name, "dedicated model directory must contain the full revision")
    require("hf-mirror" in root.name.lower(), "dedicated B model directory must identify mirror provenance")
    downloaded = False
    if not root.exists():
        require(os.environ.get("HF_ENDPOINT") == MODEL_ENDPOINT, "HF_ENDPOINT is not the frozen mirror")
        require(os.environ.get("HF_HUB_DISABLE_XET") == HF_HUB_DISABLE_XET, "Xet is not disabled")
        root.parent.mkdir(parents=True, exist_ok=True)
        partial = root.parent / f".{root.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
        require(not partial.exists(), "unique partial model directory collision")
        partial.mkdir(mode=0o700)
        from huggingface_hub import HfApi, snapshot_download

        info = HfApi(endpoint=MODEL_ENDPOINT, token=False).model_info(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            files_metadata=True,
        )
        require(info.sha == MODEL_REVISION, "mirror revision resolution drift")
        remote_files = sorted(sibling.rfilename for sibling in info.siblings)
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_dir=partial,
            token=False,
        )
        for relative in remote_files:
            path = partial / relative
            require(path.is_file() and not path.is_symlink(), f"download omitted remote file: {relative}")
        marker = snapshot_marker(remote_files, info.sha)
        marker_path = partial / MARKER_NAME
        require(not marker_path.exists(), "unexpected pre-existing source marker")
        marker_path.write_bytes(canonical_bytes(marker))
        make_read_only(partial)
        require(not root.exists(), "model root appeared during download; refusing replacement")
        os.replace(partial, root)
        downloaded = True
    receipt = authority(root)
    return {
        "schema_version": "r39-second-model-transfer-model-prepare-v1",
        "downloaded_by_this_call": downloaded,
        "model_authority": receipt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "authority"):
        child = subparsers.add_parser(name)
        child.add_argument("--model-root", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = prepare(args.model_root) if args.stage == "prepare" else authority(args.model_root)
    atomic_write(args.output, canonical_bytes(result))


if __name__ == "__main__":
    main()
