#!/usr/bin/env python3
"""Verify every frozen member and the path-ordered snapshot root."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = json.loads((HERE / "MANIFEST.json").read_text(encoding="utf-8"))
lines = []
for row in sorted(MANIFEST["files"], key=lambda item: item["snapshot_path"].encode("utf-8")):
    path = HERE / row["snapshot_path"]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != row["sha256"] or path.stat().st_size != row["size_bytes"]:
        raise SystemExit(f"FAIL {row['snapshot_path']}")
    lines.append(f"{actual}  {row['snapshot_path']}\n")
root = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
if root != MANIFEST["snapshot_sha256"]:
    raise SystemExit(f"FAIL root {root} != {MANIFEST['snapshot_sha256']}")
print(f"PASS {len(lines)} files root={root}")
