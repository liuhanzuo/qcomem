"""Standard-library integrity primitives for the method-v2 package."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, label)
    return value


def safe_relative_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value != "", label)
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, label)
    return path


def safe_existing_file(root: Path, relative: Path, label: str) -> Path:
    require(root.is_dir(), label + " root")
    resolved_root = root.resolve()
    path = root / relative
    require(path.is_file() and not path.is_symlink(), label + " file")
    require(path.resolve().is_relative_to(resolved_root), label + " path escape")
    return path


def seal_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    require("payload_sha256" not in result, "payload already sealed")
    result["payload_sha256"] = sha256_json(result)
    return result


def verify_seal(value: Mapping[str, Any], label: str) -> None:
    observed = require_sha256(value.get("payload_sha256"), label + " SHA")
    core = dict(value)
    del core["payload_sha256"]
    require(observed == sha256_json(core), label + " seal drift")


def write_new_bytes(path: Path, payload: bytes) -> None:
    require(not path.exists(), "refusing to overwrite " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + ".pending")
    require(not pending.exists(), "stale pending artifact " + str(pending))
    try:
        with pending.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        pending.replace(path)
    except BaseException:
        if pending.exists():
            pending.unlink()
        raise


def write_new_json(path: Path, value: Any) -> None:
    write_new_bytes(path, canonical_bytes(value) + b"\n")
