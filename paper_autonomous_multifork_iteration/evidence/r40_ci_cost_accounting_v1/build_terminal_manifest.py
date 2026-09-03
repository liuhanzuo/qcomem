#!/usr/bin/env python3
"""Bind final R40 outputs while representing prepared bulk by frozen manifests."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "TERMINAL_MANIFEST.json"
SIDECAR = HERE / "TERMINAL_MANIFEST.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_bulk_prepared_member(relative: str) -> bool:
    if relative.startswith("prepared_inputs/dual_formal/"):
        return True
    if relative.startswith("prepared_inputs/primary_manifest_view/"):
        return relative not in {
            "prepared_inputs/primary_manifest_view/MANIFEST.json",
            "prepared_inputs/primary_manifest_view/MANIFEST.sha256",
        }
    return False


def selected_files() -> list[Path]:
    excluded = {OUTPUT.name, SIDECAR.name}
    return sorted(
        path
        for path in HERE.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and path.relative_to(HERE).as_posix() not in excluded
        and not is_bulk_prepared_member(path.relative_to(HERE).as_posix())
        and "__pycache__" not in path.relative_to(HERE).parts
    )


def build_manifest() -> dict[str, Any]:
    primary = load_json(HERE / "prepared_inputs/PRIMARY_MANIFEST_VIEW.json")
    dual = load_json(HERE / "prepared_inputs/PREPARATION.json")
    rows = [
        {
            "relative_path": path.relative_to(HERE).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in selected_files()
    ]
    return {
        "schema_version": "forkaudit-r40-terminal-manifest-v1",
        "created_at_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "file_count": len(rows),
        "total_logical_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
        "bulk_prepared_input_bindings": {
            "primary_manifest_view": {
                "bulk_members_omitted_from_terminal_rows": True,
                "manifest_sha256": primary["manifest_sha256"],
                "manifest_payload_file_count": primary["declared_file_count"],
                "manifest_payload_logical_bytes": primary["declared_logical_bytes"],
                "all_members_verified_before_and_after_copy": primary[
                    "all_listed_members_size_and_sha256_verified_before_and_after_copy"
                ],
                "receipt_sha256": sha256_file(HERE / "prepared_inputs/PRIMARY_MANIFEST_VIEW.json"),
            },
            "dual_formal_safe_unpack": {
                "bulk_members_omitted_from_terminal_rows": True,
                "source_archive_sha256": dual["archive_sha256"],
                "source_archive_bytes": dual["archive_bytes"],
                "member_file_count": dual["member_file_count"],
                "member_logical_bytes": dual["member_logical_bytes"],
                "receipt_sha256": sha256_file(HERE / "prepared_inputs/PREPARATION.json"),
            },
        },
        "exclusions": [
            "prepared_inputs/primary_manifest_view payload members are bound by their copied MANIFEST.json and preparation receipt",
            "prepared_inputs/dual_formal members are bound by the source archive SHA-256 and safe-unpack receipt",
            "Python bytecode caches are non-evidence and excluded",
            "this manifest and its sidecar are excluded to avoid recursion",
        ],
    }


def write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def main() -> int:
    if OUTPUT.exists() or SIDECAR.exists():
        raise FileExistsError("refusing to replace terminal manifest outputs")
    value = build_manifest()
    write_new(OUTPUT, json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    write_new(SIDECAR, f"{sha256_file(OUTPUT)}  {OUTPUT.name}\n".encode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

