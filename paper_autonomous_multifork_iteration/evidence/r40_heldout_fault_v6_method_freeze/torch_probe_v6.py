#!/usr/bin/env python3
"""Isolated, fixed-name torch provenance probe; stdout is one canonical JSON value."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.machinery
import json
import os
import re
import stat
import sys
import types
from pathlib import Path


UUID_RE = re.compile(
    r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def fail(message: str) -> None:
    raise SystemExit(f"torch-provenance-reject:{message}")


def require(condition: object, message: str) -> None:
    if not condition:
        fail(message)


def regular_digest(path: Path) -> str:
    before = os.lstat(path)
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "torch init regular unique")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "torch init race")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(fd)
        require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
            "torch init changed",
        )
        return digest.hexdigest()
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--torch-init", required=True)
    parser.add_argument("--torch-sha256", required=True)
    parser.add_argument("--visibility", required=True)
    parser.add_argument("--physical-uuid", required=True)
    parser.add_argument("--index", required=True, type=int)
    args = parser.parse_args()

    require(sys.flags.isolated == 1, "not isolated")
    require(sys.flags.no_site == 1, "site enabled")
    require(sys.flags.ignore_environment == 1, "environment not ignored")
    require("torch" not in sys.modules, "preloaded/fake sys.modules torch")
    preimport_absent = True

    init_path = Path(args.torch_init).resolve(strict=True)
    require(init_path.name == "__init__.py" and init_path.parent.name == "torch", "torch path shape")
    require(
        len(args.torch_sha256) == 64
        and all(c in "0123456789abcdef" for c in args.torch_sha256),
        "torch hash syntax",
    )
    require(regular_digest(init_path) == args.torch_sha256, "torch prehash")
    require(args.index >= 0, "negative index")
    visibility = args.visibility.split(",")
    require(
        bool(visibility)
        and all(UUID_RE.fullmatch(item) is not None for item in visibility)
        and len(set(visibility)) == len(visibility)
        and ",".join(visibility) == args.visibility,
        "visibility",
    )
    require(args.index < len(visibility), "index range")
    require(UUID_RE.fullmatch(args.physical_uuid) is not None, "physical UUID")
    require(visibility[args.index] == args.physical_uuid, "UUID visibility binding")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == args.visibility, "visibility environment")

    search_root = init_path.parent.parent
    before_spec = importlib.machinery.PathFinder.find_spec("torch", [str(search_root)])
    require(before_spec is not None and before_spec.loader is not None, "torch spec")
    require(Path(before_spec.origin).resolve(strict=True) == init_path, "transplanted pre-import spec")
    require(
        before_spec.submodule_search_locations is not None
        and list(before_spec.submodule_search_locations) == [str(init_path.parent)],
        "torch package location",
    )
    sys.path.insert(0, str(search_root))
    importlib.invalidate_caches()
    torch = importlib.import_module("torch")
    require(type(torch) is types.ModuleType, "torch object type")
    require(sys.modules.get("torch") is torch, "detached torch module")
    require(torch.__spec__ is not None and torch.__spec__.loader is torch.__loader__, "torch loader identity")
    require(
        type(torch.__spec__.loader) is type(before_spec.loader),
        "transplanted torch loader",
    )
    require(Path(torch.__file__).resolve(strict=True) == init_path, "torch module file")
    require(Path(torch.__spec__.origin).resolve(strict=True) == init_path, "torch spec origin")
    require(regular_digest(init_path) == args.torch_sha256, "torch posthash")

    properties = torch.cuda.get_device_properties(args.index)
    observed_uuid = str(getattr(properties, "uuid", ""))
    require(UUID_RE.fullmatch(observed_uuid) is not None, "observed UUID syntax")
    require(observed_uuid == args.physical_uuid, "observed physical UUID")

    report = {
        "ignore_environment": True,
        "index": args.index,
        "isolated": True,
        "loader_identity": True,
        "module_file": str(init_path),
        "module_identity": True,
        "module_sha256": args.torch_sha256,
        "no_site": True,
        "physical_uuid": observed_uuid,
        "preimport_absent": preimport_absent,
        "schema_version": "forkaudit-v6-torch-provenance-v1",
        "spec_origin": str(init_path),
        "visibility": args.visibility,
    }
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
