#!/usr/bin/env python3
"""Acquire and attest one immutable official ModelScope snapshot.

The acquisition source may differ from the canonical Hugging Face identity,
but the frozen ModelScope tree is pinned to a full Git commit and the model
weight and tokenizer hashes must exactly match the corresponding frozen
Hugging Face LFS hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


MODEL_ID = "Qwen/Qwen3.5-0.8B"
HF_MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
MODELSCOPE_REVISION = "4d58a7b524cd33ed843d5125be8cd8f0a452d9bf"
MODEL_ENDPOINT = "https://modelscope.cn"
MODEL_TOKEN_POLICY = "public-no-token"
MODEL_TOKEN = False
MODEL_DOWNLOAD_TRANSPORT = "modelscope-official-revision-pinned-per-file-https-range"
FROZEN_TREE_SHA256 = "c66cfad254b688a5ca350d42326ecb05be5a845050752b927a8dca2d2dcaaadf"
WEIGHT_PATH = "model.safetensors-00001-of-00001.safetensors"
WEIGHT_SHA256 = "04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696"
TOKENIZER_PATH = "tokenizer.json"
TOKENIZER_SHA256 = "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
MARKER_NAME = ".r39-public-snapshot-source.json"
DOWNLOAD_ATTEMPTS = 12
CHUNK_BYTES = 8 * 1024 * 1024


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
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def acquisition_policy() -> dict[str, Any]:
    return {
        "endpoint": MODEL_ENDPOINT,
        "endpoint_is_explicit": True,
        "official_namespace": "Qwen",
        "token": MODEL_TOKEN,
        "token_policy": MODEL_TOKEN_POLICY,
        "transport": MODEL_DOWNLOAD_TRANSPORT,
        "canonical_huggingface_revision": HF_MODEL_REVISION,
        "canonical_huggingface_revision_is_full_commit": True,
        "modelscope_revision": MODELSCOPE_REVISION,
        "modelscope_revision_is_full_commit": True,
        "weight_equivalence_sha256": WEIGHT_SHA256,
        "tokenizer_equivalence_sha256": TOKENIZER_SHA256,
        "frozen_tree_sha256": FROZEN_TREE_SHA256,
        "remote_tree_must_equal_frozen_manifest": True,
        "per_file_size_and_sha256_required": True,
        "resumable_range_download": True,
        "fresh_nonoverwriting_snapshot_required": True,
        "a_failed_partial_reuse_forbidden": True,
        "b_failed_partial_reuse_forbidden": True,
    }


def safe_relative_path(raw: str) -> Path:
    path = Path(raw)
    require(raw != "" and not path.is_absolute(), "unsafe empty or absolute remote path")
    require(".." not in path.parts and "." not in path.parts, "unsafe remote path traversal")
    require(path.as_posix() == raw, "remote path is not canonical POSIX")
    return path


def normalize_file_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    seen: set[str] = set()
    for row in rows:
        require(isinstance(row, dict), "remote file row is not an object")
        path = str(row["path"])
        safe_relative_path(path)
        require(path not in seen, "duplicate remote file path")
        seen.add(path)
        size = int(row["size"])
        sha256 = str(row["sha256"])
        file_revision = str(row["file_revision"])
        require(size >= 0, "negative remote file size")
        require(re.fullmatch(r"[0-9a-f]{64}", sha256) is not None, "bad remote SHA-256")
        require(re.fullmatch(r"[0-9a-f]{40}", file_revision) is not None, "bad file revision")
        normalized.append(
            {
                "path": path,
                "size": size,
                "sha256": sha256,
                "is_lfs": bool(row["is_lfs"]),
                "file_revision": file_revision,
            }
        )
    normalized.sort(key=lambda row: row["path"])
    require(normalized and [row["path"] for row in normalized] == sorted(seen), "tree ordering drift")
    return normalized


def load_frozen_tree(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), "frozen ModelScope tree absent")
    require(sha256_file(path) == FROZEN_TREE_SHA256, "frozen ModelScope tree raw drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema_version") == "r39-second-model-transfer-modelscope-tree-v1", "tree schema drift")
    source = value.get("official_source")
    require(isinstance(source, dict), "official source authority absent")
    require(source.get("endpoint") == MODEL_ENDPOINT, "ModelScope endpoint drift")
    require(source.get("namespace") == "Qwen", "official Qwen namespace drift")
    require(source.get("repo_id") == MODEL_ID, "ModelScope model ID drift")
    require(source.get("revision") == MODELSCOPE_REVISION, "ModelScope revision drift")
    require(source.get("revision_is_full_commit") is True, "ModelScope revision is not full")
    identity = value.get("canonical_huggingface_identity")
    require(isinstance(identity, dict), "canonical Hugging Face identity absent")
    require(identity.get("repo_id") == MODEL_ID, "canonical model ID drift")
    require(identity.get("revision") == HF_MODEL_REVISION, "canonical Hugging Face revision drift")
    require(identity.get("revision_is_full_commit") is True, "Hugging Face revision is not full")
    rows = normalize_file_rows(value.get("files", []))
    require(value.get("file_count") == len(rows) == 14, "frozen ModelScope file count drift")
    by_path = {row["path"]: row for row in rows}
    require(by_path[WEIGHT_PATH]["sha256"] == WEIGHT_SHA256, "weight equivalence drift")
    require(by_path[TOKENIZER_PATH]["sha256"] == TOKENIZER_SHA256, "tokenizer equivalence drift")
    equivalence = value.get("cross_source_equivalence")
    require(isinstance(equivalence, dict), "cross-source equivalence absent")
    for file_path, expected_sha in ((WEIGHT_PATH, WEIGHT_SHA256), (TOKENIZER_PATH, TOKENIZER_SHA256)):
        row = equivalence.get(file_path)
        require(isinstance(row, dict), f"equivalence row absent: {file_path}")
        require(row.get("bytes") == by_path[file_path]["size"], f"equivalence size drift: {file_path}")
        require(row.get("huggingface_lfs_sha256") == expected_sha, f"HF hash drift: {file_path}")
        require(row.get("modelscope_sha256") == expected_sha, f"ModelScope hash drift: {file_path}")
    return {**value, "files": rows}


def snapshot_marker(tree: dict[str, Any]) -> dict[str, Any]:
    rows = tree["files"]
    return {
        "schema_version": "r39-second-model-transfer-modelscope-source-marker-v1",
        "repo_id": MODEL_ID,
        "canonical_huggingface_revision": HF_MODEL_REVISION,
        "modelscope_revision": MODELSCOPE_REVISION,
        "model_acquisition": acquisition_policy(),
        "frozen_tree_sha256": FROZEN_TREE_SHA256,
        "remote_file_count": len(rows),
        "remote_files_sha256": sha256_bytes(canonical_bytes(rows)),
        "cross_source_equivalence": tree["cross_source_equivalence"],
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


def all_directory_paths(root: Path) -> list[Path]:
    return [root, *sorted((path for path in root.rglob("*") if path.is_dir()), key=str)]


def authority(root: Path, tree: dict[str, Any]) -> dict[str, Any]:
    marker_path = root / MARKER_NAME
    require(marker_path.is_file() and not marker_path.is_symlink(), "snapshot source marker absent")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    require(marker == snapshot_marker(tree), "snapshot source marker drift")
    expected = {row["path"]: row for row in tree["files"]}
    rows = []
    for path in all_file_paths(root):
        status = path.stat()
        require(stat.S_ISREG(status.st_mode), f"not a regular model file: {path}")
        require(status.st_mode & 0o222 == 0, f"writable model file: {path}")
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        if relative != MARKER_NAME:
            require(relative in expected, f"unexpected local model file: {relative}")
            require(status.st_size == expected[relative]["size"], f"model size drift: {relative}")
            require(digest == expected[relative]["sha256"], f"model hash drift: {relative}")
        rows.append(
            {
                "path": relative,
                "bytes": status.st_size,
                "sha256": digest,
                "mode": stat.S_IMODE(status.st_mode),
                "device": status.st_dev,
                "inode": status.st_ino,
                "ctime_ns": status.st_ctime_ns,
            }
        )
    require({row["path"] for row in rows} == {*expected, MARKER_NAME}, "local tree differs from frozen tree")
    for path in all_directory_paths(root):
        require(path.stat().st_mode & 0o222 == 0, f"writable model directory: {path}")
    return {
        "schema_version": "r39-second-model-transfer-model-authority-v1",
        "repo_id": MODEL_ID,
        "revision": HF_MODEL_REVISION,
        "model_acquisition": {
            "policy": acquisition_policy(),
            "canonical_huggingface_revision": HF_MODEL_REVISION,
            "modelscope_revision": MODELSCOPE_REVISION,
            "remote_file_count": len(expected),
            "remote_files_sha256": marker["remote_files_sha256"],
            "source_marker_sha256": sha256_file(marker_path),
            "cross_source_equivalence": tree["cross_source_equivalence"],
        },
        "all_local_files_hashed": True,
        "all_local_files_and_directories_read_only": True,
        "file_count": len(rows),
        "repo_file_count": len(expected),
        "files": rows,
        "normalized_files_sha256": sha256_bytes(canonical_bytes(rows)),
    }


def make_read_only(root: Path) -> None:
    for path in all_file_paths(root):
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, mode & ~0o222)
    for path in sorted(all_directory_paths(root), key=lambda item: len(item.parts), reverse=True):
        mode = stat.S_IMODE(path.stat().st_mode)
        os.chmod(path, (mode & ~0o222) | 0o555)


def normalize_api_rows(raw_rows: Any) -> list[dict[str, Any]]:
    require(isinstance(raw_rows, list), "ModelScope file list is not an array")
    rows = []
    for row in raw_rows:
        require(isinstance(row, dict), "ModelScope file metadata row is not an object")
        require(row.get("Type") == "blob", "ModelScope tree contains a non-blob")
        rows.append(
            {
                "path": row.get("Path"),
                "size": row.get("Size"),
                "sha256": row.get("Sha256"),
                "is_lfs": row.get("IsLFS"),
                "file_revision": row.get("Revision"),
            }
        )
    return normalize_file_rows(rows)


def require_official_https_response(response: Any) -> None:
    for item in [*response.history, response]:
        parsed = urlsplit(str(item.url))
        require(parsed.scheme == "https", "model download left HTTPS")
        host = (parsed.hostname or "").lower()
        require(host == "modelscope.cn" or host.endswith(".modelscope.cn"), "model download left official ModelScope hosts")


def fetch_remote_tree(client: Any, tree: dict[str, Any]) -> list[dict[str, Any]]:
    response = client.get(
        f"{MODEL_ENDPOINT}/api/v1/models/{MODEL_ID}/repo/files",
        params={"Revision": MODELSCOPE_REVISION, "Recursive": "true"},
    )
    require_official_https_response(response)
    response.raise_for_status()
    payload = response.json()
    require(payload.get("Code") == 200 and payload.get("Success") is True, "ModelScope tree API failure")
    rows = normalize_api_rows(payload.get("Data", {}).get("Files"))
    require(rows == tree["files"], "live ModelScope tree differs from frozen pinned tree")
    return rows


def validate_range_response(status: int, headers: dict[str, str], offset: int, total: int) -> None:
    require(status in (200, 206), f"unexpected model file HTTP status: {status}")
    if status == 200:
        require(offset == 0, "server ignored a nonzero resume range")
        length = headers.get("content-length")
        if length is not None:
            require(int(length) == total, "full-response length drift")
        return
    content_range = headers.get("content-range")
    match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+)", content_range or "")
    require(match is not None, "missing or malformed Content-Range")
    start, end, reported_total = (int(item) for item in match.groups())
    require(start == offset, "resume range start drift")
    require(end >= start and reported_total == total, "resume range extent drift")
    length = headers.get("content-length")
    if length is not None:
        require(int(length) == end - start + 1, "partial-response length drift")


def finalize_file(part: Path, destination: Path, row: dict[str, Any]) -> None:
    require(part.is_file() and not part.is_symlink(), "download partial absent")
    require(part.stat().st_size == row["size"], f"download size drift: {row['path']}")
    require(sha256_file(part) == row["sha256"], f"download SHA-256 drift: {row['path']}")
    require(not destination.exists(), f"refusing to replace model file: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(part, destination)


def download_file(client: Any, row: dict[str, Any], partial_root: Path) -> dict[str, Any]:
    parts = partial_root / ".r39-independent-parts"
    parts.mkdir(exist_ok=True)
    path_key = sha256_bytes(row["path"].encode("utf-8"))[:16]
    part = parts / f"{path_key}-{row['sha256']}.part"
    destination = partial_root / safe_relative_path(row["path"])
    attempts = 0
    resumed_attempts = 0
    final_host = None
    final_path = None
    while True:
        offset = part.stat().st_size if part.exists() else 0
        require(offset <= row["size"], f"oversized partial: {row['path']}")
        if offset == row["size"]:
            break
        attempts += 1
        require(attempts <= DOWNLOAD_ATTEMPTS, f"download retries exhausted: {row['path']}")
        if offset:
            resumed_attempts += 1
        try:
            with client.stream(
                "GET",
                f"{MODEL_ENDPOINT}/api/v1/models/{MODEL_ID}/repo",
                params={"Revision": MODELSCOPE_REVISION, "FilePath": row["path"]},
                headers={"Range": f"bytes={offset}-", "Accept-Encoding": "identity"},
            ) as response:
                require_official_https_response(response)
                headers = {key.lower(): value for key, value in response.headers.items()}
                validate_range_response(response.status_code, headers, offset, row["size"])
                parsed = urlsplit(str(response.url))
                final_host = parsed.hostname
                final_path = parsed.path
                mode = "ab" if offset else "wb"
                with part.open(mode) as handle:
                    for chunk in response.iter_raw(chunk_size=CHUNK_BYTES):
                        if not chunk:
                            continue
                        require(handle.tell() + len(chunk) <= row["size"], f"download exceeded frozen size: {row['path']}")
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
            if part.stat().st_size < row["size"]:
                raise OSError(f"short response for {row['path']}")
        except (OSError, TimeoutError) as error:
            if attempts >= DOWNLOAD_ATTEMPTS:
                raise RuntimeError(f"download retries exhausted: {row['path']}") from error
            time.sleep(min(2 ** (attempts - 1), 30))
        except Exception as error:
            import httpx

            if isinstance(error, httpx.TransportError):
                if attempts >= DOWNLOAD_ATTEMPTS:
                    raise RuntimeError(f"download retries exhausted: {row['path']}") from error
                time.sleep(min(2 ** (attempts - 1), 30))
                continue
            raise
    finalize_file(part, destination, row)
    return {
        "path": row["path"],
        "bytes": row["size"],
        "sha256": row["sha256"],
        "attempts": attempts,
        "resumed_attempts": resumed_attempts,
        "final_host": final_host,
        "final_path": final_path,
    }


def prepare(root: Path, tree_path: Path) -> dict[str, Any]:
    tree = load_frozen_tree(tree_path)
    require(HF_MODEL_REVISION in root.name, "dedicated model directory must contain the HF revision")
    require(MODELSCOPE_REVISION in root.name, "dedicated model directory must contain the ModelScope revision")
    require("modelscope" in root.name.lower() and "20260826c" in root.name.lower(), "dedicated C model directory provenance drift")
    downloaded = False
    receipts: list[dict[str, Any]] = []
    if not root.exists():
        root.parent.mkdir(parents=True, exist_ok=True)
        partial = root.parent / f".{root.name}.partial-{os.getpid()}-{uuid.uuid4().hex}"
        require(not partial.exists(), "unique C partial directory collision")
        partial.mkdir(mode=0o700)
        import httpx

        timeout = httpx.Timeout(connect=30.0, read=90.0, write=30.0, pool=30.0)
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout,
            headers={"User-Agent": "ForkAudit-r39-modelscope-c/1"},
        ) as client:
            rows = fetch_remote_tree(client, tree)
            for row in rows:
                receipts.append(download_file(client, row, partial))
        parts = partial / ".r39-independent-parts"
        require(parts.is_dir() and not any(parts.iterdir()), "download part closure failed")
        parts.rmdir()
        require(
            {path.relative_to(partial).as_posix() for path in all_file_paths(partial)}
            == {row["path"] for row in tree["files"]},
            "downloaded repository tree differs from frozen tree",
        )
        marker_path = partial / MARKER_NAME
        require(not marker_path.exists(), "unexpected pre-existing source marker")
        marker_path.write_bytes(canonical_bytes(snapshot_marker(tree)))
        make_read_only(partial)
        require(not root.exists(), "model root appeared during download; refusing replacement")
        os.replace(partial, root)
        downloaded = True
    receipt = authority(root, tree)
    return {
        "schema_version": "r39-second-model-transfer-model-prepare-v1",
        "downloaded_by_this_call": downloaded,
        "download_receipts": receipts,
        "model_authority": receipt,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for name in ("prepare", "authority"):
        child = subparsers.add_parser(name)
        child.add_argument("--model-root", type=Path, required=True)
        child.add_argument("--frozen-tree", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tree = load_frozen_tree(args.frozen_tree)
    result = prepare(args.model_root, args.frozen_tree) if args.stage == "prepare" else authority(args.model_root, tree)
    atomic_write(args.output, canonical_bytes(result))


if __name__ == "__main__":
    main()
