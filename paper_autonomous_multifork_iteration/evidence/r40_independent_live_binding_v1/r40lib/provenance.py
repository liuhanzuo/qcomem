from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .protocol import require


SOURCE_LEDGER_SCHEMA = "forkaudit-r40-independent-live-binding-source-ledger-v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def verify_source_ledger(root: Path, ledger_path: Path) -> Mapping[str, Any]:
    ledger = load_json(ledger_path)
    require(ledger.get("schema_version") == SOURCE_LEDGER_SCHEMA, "source ledger schema drift")
    entries = ledger.get("files")
    require(isinstance(entries, list) and entries, "source ledger entries missing")
    seen: set[str] = set()
    for row in entries:
        relative = str(row["path"])
        require(relative not in seen, "duplicate source ledger path")
        seen.add(relative)
        path = root / relative
        require(path.is_file(), f"source ledger file missing: {relative}")
        require(path.stat().st_size == int(row["bytes"]), f"source size drift: {relative}")
        require(sha256_file(path) == row["sha256"], f"source hash drift: {relative}")
    return ledger


__all__ = [
    "SOURCE_LEDGER_SCHEMA",
    "load_json",
    "sha256_file",
    "verify_source_ledger",
    "write_json_new",
]

