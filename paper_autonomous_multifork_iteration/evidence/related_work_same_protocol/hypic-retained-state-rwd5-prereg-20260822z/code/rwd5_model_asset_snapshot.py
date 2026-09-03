#!/usr/bin/env python3
"""Fail-closed model-view identity snapshot for RW-D5 recovery U.

The stable cross-node authority is content, kind, ownership, permissions, and
size.  Node-local inode/device/timestamps are observations.  Within one
preflight, however, the complete observation tuple below must remain identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


SCHEMA = "hypic-rwd5-model-asset-snapshot-v1"
EXPECTED = {
    "model-artifacts.sha256": (
        "d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd",
        778,
    ),
    "preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        390,
    ),
    "video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
        385,
    ),
}


class SnapshotError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SnapshotError(message)


def identity_tuple(row: os.stat_result) -> dict[str, int]:
    """Fields stable under reads and sufficient to identify same-preflight drift.

    atime is intentionally excluded because hashing is itself a read and may
    update atime.  All other identity-relevant lstat/fstat fields available on
    Linux are retained, including physical inode/device and nanosecond times.
    """
    return {
        "mode": int(row.st_mode),
        "uid": int(row.st_uid),
        "gid": int(row.st_gid),
        "size": int(row.st_size),
        "inode": int(row.st_ino),
        "device": int(row.st_dev),
        "nlink": int(row.st_nlink),
        "rdev": int(row.st_rdev),
        "block_size": int(getattr(row, "st_blksize", 0)),
        "blocks": int(getattr(row, "st_blocks", 0)),
        "mtime_ns": int(row.st_mtime_ns),
        "ctime_ns": int(row.st_ctime_ns),
    }


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return identity_tuple(left) == identity_tuple(right)


def snapshot_one(path: Path, expected_sha256: str, expected_size: int) -> dict[str, Any]:
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode), f"not regular: {path.name}")
    require(not path.is_symlink(), f"symlink forbidden: {path.name}")
    require(stat.S_IMODE(before.st_mode) == 0o444, f"mode drift: {path.name}")
    require(before.st_uid == 0 and before.st_gid == 0, f"owner drift: {path.name}")
    require(before.st_size == expected_size, f"size drift: {path.name}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened_before = os.fstat(fd)
        require(_same_identity(before, opened_before), f"open identity race: {path.name}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        opened_after = os.fstat(fd)
        require(_same_identity(opened_before, opened_after), f"open file changed: {path.name}")
    finally:
        os.close(fd)
    after = path.lstat()
    require(_same_identity(opened_after, after), f"path changed after read: {path.name}")
    require(digest.hexdigest() == expected_sha256, f"SHA drift: {path.name}")
    return {
        "name": path.name,
        "sha256": digest.hexdigest(),
        "stable_cross_node_authority": {
            "regular_non_symlink": True,
            "mode_octal": "0444",
            "uid": 0,
            "gid": 0,
            "size": expected_size,
        },
        "same_preflight_observation": identity_tuple(after),
        "physical_identity_fields_are_observation_only": True,
        "atime_excluded_because_hashing_is_a_read": True,
    }


def snapshot(model_root: Path) -> dict[str, Any]:
    require(model_root.is_dir() and not model_root.is_symlink(), "model root")
    rows = [
        snapshot_one(model_root / name, digest, size)
        for name, (digest, size) in sorted(EXPECTED.items())
    ]
    return {
        "schema": SCHEMA,
        "model_root": str(model_root),
        "entries": rows,
        "cross_node_authority_excludes_inode_device_and_timestamps": True,
        "same_preflight_requires_exact_observation_equality": True,
    }


def validate_snapshot(value: Any, *, model_root: Path) -> dict[str, Any]:
    require(isinstance(value, dict), "snapshot object")
    require(value.get("schema") == SCHEMA, "snapshot schema")
    require(value.get("model_root") == str(model_root), "snapshot model root")
    require(value.get("cross_node_authority_excludes_inode_device_and_timestamps") is True, "authority boundary")
    require(value.get("same_preflight_requires_exact_observation_equality") is True, "preflight equality")
    rows = value.get("entries")
    require(isinstance(rows, list) and len(rows) == len(EXPECTED), "snapshot entries")
    require([row.get("name") for row in rows] == sorted(EXPECTED), "snapshot entry names")
    for row in rows:
        name = row["name"]
        digest, size = EXPECTED[name]
        require(row.get("sha256") == digest, f"snapshot SHA: {name}")
        require(row.get("stable_cross_node_authority") == {
            "regular_non_symlink": True,
            "mode_octal": "0444",
            "uid": 0,
            "gid": 0,
            "size": size,
        }, f"snapshot stable authority: {name}")
        observed = row.get("same_preflight_observation")
        require(isinstance(observed, dict) and set(observed) == {
            "mode", "uid", "gid", "size", "inode", "device", "nlink", "rdev",
            "block_size", "blocks", "mtime_ns", "ctime_ns",
        }, f"snapshot observation tuple: {name}")
        require(row.get("physical_identity_fields_are_observation_only") is True, f"snapshot physical boundary: {name}")
        require(row.get("atime_excluded_because_hashing_is_a_read") is True, f"snapshot atime boundary: {name}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        os.write(1, canonical_bytes(snapshot(args.model_root)))
    except (OSError, SnapshotError) as error:
        raise SystemExit(f"asset snapshot failed: {error}") from error


if __name__ == "__main__":
    main()
