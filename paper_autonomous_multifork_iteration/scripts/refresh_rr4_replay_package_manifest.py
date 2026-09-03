#!/usr/bin/env python3
"""Regenerate the complete, pointer-free Round-4 replay package manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "evidence" / "round_04_rr2_package"
EXCLUDED = {"MANIFEST.json", "MANIFEST.sha256"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(PACKAGE).as_posix()
        if relative in EXCLUDED:
            continue
        rows.append(
            {
                "bytes": path.stat().st_size,
                "relative_path": relative,
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "excludes": sorted(EXCLUDED),
        "file_count": len(rows),
        "files": rows,
        "parent_manifest_sha256": "d6a9b71ee078c6d21c90c64ad23d9c4f624e381d262faf36ed812104e8e59633",
        "schema_version": "anonymous-hash-bound-reviewer-package-complete-v2",
        "total_bytes": sum(row["bytes"] for row in rows),
    }
    manifest_path = PACKAGE / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    digest = sha256_file(manifest_path)
    (PACKAGE / "MANIFEST.sha256").write_text(
        f"{digest}  MANIFEST.json\n", encoding="ascii"
    )
    print(f"{digest} {len(rows)} {manifest['total_bytes']}")


if __name__ == "__main__":
    main()
