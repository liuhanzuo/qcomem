#!/usr/bin/env python3
"""Verify the complete reviewer-package manifest before replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXCLUDED = {"MANIFEST.json", "MANIFEST.sha256"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(package_root: Path) -> None:
    manifest_path = package_root / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = manifest["files"]
    expected_paths = [row["relative_path"] for row in rows]
    if expected_paths != sorted(expected_paths) or len(expected_paths) != len(set(expected_paths)):
        raise ValueError("manifest paths are not unique and sorted")
    actual_paths = sorted(
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path.relative_to(package_root).as_posix() not in EXCLUDED
    )
    if actual_paths != expected_paths:
        missing = sorted(set(expected_paths) - set(actual_paths))
        extra = sorted(set(actual_paths) - set(expected_paths))
        raise ValueError(f"package file-set drift: missing={missing}, extra={extra}")
    total_bytes = 0
    for row in rows:
        path = package_root / row["relative_path"]
        size = path.stat().st_size
        digest = sha256_file(path)
        if size != row["bytes"] or digest != row["sha256"]:
            raise ValueError(f"package file drift: {row['relative_path']}")
        total_bytes += size
    if len(rows) != manifest["file_count"] or total_bytes != manifest["total_bytes"]:
        raise ValueError("manifest summary drift")
    sidecar = (package_root / "MANIFEST.sha256").read_text(encoding="ascii").strip()
    expected_sidecar = f"{sha256_file(manifest_path)}  MANIFEST.json"
    if sidecar != expected_sidecar:
        raise ValueError("manifest sidecar drift")
    print(
        f"PASS package manifest: {len(rows)} files, {total_bytes} bytes, "
        f"manifest_sha256={sha256_file(manifest_path)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    args = parser.parse_args()
    verify(args.package_root.resolve())


if __name__ == "__main__":
    main()
