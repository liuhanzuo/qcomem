"""Fail-closed integrity primitives for method v3."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ContractError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    require(isinstance(value, Mapping), label + " object")
    require(set(value.keys()) == set(expected), label + " fields")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(value: Any, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None, label + " SHA-256")
    return value


def seal_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    require("payload_sha256" not in result, "payload already sealed")
    result["payload_sha256"] = sha256_bytes(canonical_bytes(result))
    return result


def verify_seal(value: Mapping[str, Any], label: str) -> None:
    observed = require_sha256(value.get("payload_sha256"), label + " payload")
    core = dict(value)
    del core["payload_sha256"]
    require(sha256_bytes(canonical_bytes(core)) == observed, label + " seal drift")


def safe_relative_path(value: Any, label: str) -> Path:
    require(isinstance(value, str) and value != "", label)
    path = Path(value)
    require(not path.is_absolute() and ".." not in path.parts, label)
    return path


def require_regular_file_under(root: Path, relative: Path, label: str) -> Path:
    require(root.is_dir() and not root.is_symlink(), label + " root")
    path = root / relative
    require(path.is_file() and not path.is_symlink(), label + " regular file")
    require(path.resolve().is_relative_to(root.resolve()), label + " path escape")
    return path


def load_json_file(path: Path, label: str) -> Mapping[str, Any]:
    require(path.is_file() and not path.is_symlink(), label + " file")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), label + " JSON object")
    return value


def write_new_bytes(path: Path, payload: bytes) -> None:
    require(not path.exists(), "refusing overwrite " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name("." + path.name + ".pending")
    require(not pending.exists(), "stale pending " + str(pending))
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


def open_exclusive_lock(path: Path, payload: Mapping[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        data = canonical_bytes(payload) + b"\n"
        os.write(descriptor, data)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor

