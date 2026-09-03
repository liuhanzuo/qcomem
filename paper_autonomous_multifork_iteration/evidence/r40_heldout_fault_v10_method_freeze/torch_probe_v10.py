#!/usr/bin/env python3
"""Isolated, fixed-name torch provenance probe; stdout is one canonical JSON value."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import importlib
import importlib.machinery
import json
import os
import re
import runpy
import stat
import sys
import types
from pathlib import Path, PurePosixPath


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
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "regular unique")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "file race")
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
            "file changed",
        )
        return digest.hexdigest()
    finally:
        os.close(fd)


def regular_bytes(path: Path) -> bytes:
    before = os.lstat(path)
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1, "regular unique")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(fd)
        require((opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino), "file race")
        chunks = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(fd)
        require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
            "file changed",
        )
        return b"".join(chunks)
    finally:
        os.close(fd)


def _fd_digest(fd: int) -> str:
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def loaded_executable_identity() -> dict:
    """Measure the executable vnode backing this running process.

    On Darwin, the dyld image address is resolved through libproc so a
    rename-and-replace of ``sys.executable`` cannot make a different vnode look
    like the one already mapped.  Linux provides the same property through
    ``/proc/self/exe``.  Unsupported kernels fail closed.
    """
    if sys.platform == "darwin":
        class VinfoStat(ctypes.Structure):
            _fields_ = [
                ("dev", ctypes.c_uint32), ("mode", ctypes.c_uint16),
                ("nlink", ctypes.c_uint16), ("ino", ctypes.c_uint64),
                ("uid", ctypes.c_uint32), ("gid", ctypes.c_uint32),
                ("atime", ctypes.c_int64), ("atimensec", ctypes.c_int64),
                ("mtime", ctypes.c_int64), ("mtimensec", ctypes.c_int64),
                ("ctime", ctypes.c_int64), ("ctimensec", ctypes.c_int64),
                ("birthtime", ctypes.c_int64),
                ("birthtimensec", ctypes.c_int64),
                ("size", ctypes.c_int64), ("blocks", ctypes.c_int64),
                ("blksize", ctypes.c_int32), ("flags", ctypes.c_uint32),
                ("gen", ctypes.c_uint32), ("rdev", ctypes.c_uint32),
                ("qspare", ctypes.c_int64 * 2),
            ]

        class Fsid(ctypes.Structure):
            _fields_ = [("value", ctypes.c_int32 * 2)]

        class VnodeInfo(ctypes.Structure):
            _fields_ = [
                ("stat", VinfoStat), ("type", ctypes.c_int),
                ("pad", ctypes.c_int), ("fsid", Fsid),
            ]

        class VnodeInfoPath(ctypes.Structure):
            _fields_ = [("info", VnodeInfo), ("path", ctypes.c_char * 1024)]

        class RegionInfo(ctypes.Structure):
            _fields_ = [
                ("protection", ctypes.c_uint32),
                ("max_protection", ctypes.c_uint32),
                ("inheritance", ctypes.c_uint32), ("flags", ctypes.c_uint32),
                ("offset", ctypes.c_uint64), ("behavior", ctypes.c_uint32),
                ("wired", ctypes.c_uint32), ("tag", ctypes.c_uint32),
                ("resident", ctypes.c_uint32),
                ("shared_now_private", ctypes.c_uint32),
                ("swapped", ctypes.c_uint32), ("dirtied", ctypes.c_uint32),
                ("ref_count", ctypes.c_uint32),
                ("shadow_depth", ctypes.c_uint32),
                ("share_mode", ctypes.c_uint32),
                ("private_resident", ctypes.c_uint32),
                ("shared_resident", ctypes.c_uint32),
                ("obj_id", ctypes.c_uint32), ("depth", ctypes.c_uint32),
                ("address", ctypes.c_uint64), ("size", ctypes.c_uint64),
            ]

        class RegionWithPath(ctypes.Structure):
            _fields_ = [("region", RegionInfo), ("vnode", VnodeInfoPath)]

        process = ctypes.CDLL(None, use_errno=True)
        image_header = process._dyld_get_image_header
        image_header.argtypes = [ctypes.c_uint32]
        image_header.restype = ctypes.c_void_p
        address = image_header(0)
        require(type(address) is int and address > 0, "loaded image address")
        libproc = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
        proc_pidinfo = libproc.proc_pidinfo
        proc_pidinfo.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_uint64,
            ctypes.c_void_p, ctypes.c_int,
        ]
        proc_pidinfo.restype = ctypes.c_int
        region = RegionWithPath()
        returned = proc_pidinfo(
            os.getpid(), 8, address, ctypes.byref(region), ctypes.sizeof(region)
        )
        require(returned == ctypes.sizeof(region), "loaded image vnode query")
        path_raw = bytes(region.vnode.path).split(b"\0", 1)[0]
        try:
            path = Path(path_raw.decode("utf-8", "strict")).resolve(strict=True)
        except (UnicodeError, OSError) as exc:
            fail(f"loaded image path:{type(exc).__name__}")
        path_stat = os.stat(path, follow_symlinks=False)
        vnode_stat = region.vnode.info.stat
        require(
            (path_stat.st_dev, path_stat.st_ino)
            == (vnode_stat.dev, vnode_stat.ino),
            "loaded image namespace/vnode race",
        )
        return {
            "dev": vnode_stat.dev,
            "ino": vnode_stat.ino,
            "path": str(path),
            "sha256": regular_digest(path),
        }
    if sys.platform.startswith("linux"):
        link = Path("/proc/self/exe")
        fd = os.open(link, os.O_RDONLY)
        try:
            opened = os.fstat(fd)
            require(stat.S_ISREG(opened.st_mode), "loaded executable regular")
            path = Path(os.readlink(link)).resolve(strict=True)
            current = os.stat(path, follow_symlinks=False)
            require(
                (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino),
                "loaded executable namespace/vnode race",
            )
            digest = _fd_digest(fd)
            after = os.fstat(fd)
            require(
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
                "loaded executable changed",
            )
            return {
                "dev": opened.st_dev,
                "ino": opened.st_ino,
                "path": str(path),
                "sha256": digest,
            }
        finally:
            os.close(fd)
    fail("unsupported loaded-executable identity platform")


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    out = {}
    for key, value in pairs:
        require(key not in out, "duplicate JSON key")
        out[key] = value
    return out


def canonical_manifest(path: Path, expected_sha256: str) -> list[dict]:
    raw = regular_bytes(path)
    require(hashlib.sha256(raw).hexdigest() == expected_sha256, "manifest digest")
    try:
        value = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=unique_object,
            parse_constant=lambda value: fail(f"non-JSON constant:{value}"),
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        fail(f"manifest JSON:{type(exc).__name__}")
    require(raw == canonical_bytes(value), "noncanonical manifest")
    require(type(value) is list and bool(value), "manifest list")
    rows = []
    previous = None
    for item in value:
        require(type(item) is dict and set(item) == {"path", "sha256", "size"}, "manifest row")
        name = item["path"]
        digest = item["sha256"]
        size = item["size"]
        require(type(name) is str and PurePosixPath(name).as_posix() == name, "manifest path")
        require(not name.startswith("/") and all(part not in ("", ".", "..") for part in PurePosixPath(name).parts), "manifest traversal")
        require(type(digest) is str and len(digest) == 64 and all(c in "0123456789abcdef" for c in digest), "manifest hash")
        require(type(size) is int and size >= 0, "manifest size")
        require(previous is None or previous < name, "manifest order")
        previous = name
        rows.append({"path": name, "sha256": digest, "size": size})
    return rows


def scan_runner(root: Path, expected: list[dict], expected_inventory_sha256: str) -> None:
    root_stat = os.lstat(root)
    require(stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode), "runner root")
    actual = []
    inodes = set()
    for directory, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        filenames.sort()
        for dirname in dirnames:
            item = Path(directory) / dirname
            item_stat = os.lstat(item)
            require(stat.S_ISDIR(item_stat.st_mode) and not stat.S_ISLNK(item_stat.st_mode), "runner directory")
        for filename in filenames:
            item = Path(directory) / filename
            item_stat = os.lstat(item)
            require(stat.S_ISREG(item_stat.st_mode) and item_stat.st_nlink == 1, "runner file")
            key = (item_stat.st_dev, item_stat.st_ino)
            require(key not in inodes, "runner duplicate inode")
            inodes.add(key)
            actual.append(
                {
                    "path": item.relative_to(root).as_posix(),
                    "sha256": regular_digest(item),
                    "size": item_stat.st_size,
                }
            )
    actual.sort(key=lambda row: row["path"])
    require(actual == expected, "runner inventory mismatch")
    inventory = {"files": actual, "schema_version": "forkaudit-v10-runner-inventory-v1"}
    require(hashlib.sha256(canonical_bytes(inventory)).hexdigest() == expected_inventory_sha256, "runner inventory digest")


def worker_main(encoded_payload: str) -> int:
    """Report child-observed identity, then execute the signed Python worker in-process."""
    try:
        raw = base64.b64decode(encoded_payload.encode("ascii"), validate=True)
        payload = json.loads(raw.decode("utf-8", "strict"), object_pairs_hook=unique_object)
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        fail(f"worker payload:{type(exc).__name__}")
    require(raw == canonical_bytes(payload), "noncanonical worker payload")
    require(
        type(payload) is dict
        and set(payload)
        == {"argv", "argv_schema", "cwd_contract", "env", "env_schema", "fault_id"},
        "worker payload schema",
    )
    argv = payload["argv"]
    env = payload["env"]
    require(type(argv) is list and len(argv) >= 2 and all(type(x) is str and "\x00" not in x for x in argv), "worker argv")
    if type(env) is dict:
        for ambient_key in set(os.environ) - set(env):
            os.environ.pop(ambient_key, None)
    require(type(env) is dict and env == dict(os.environ), "worker environment")
    require(type(payload["argv_schema"]) is list and type(payload["env_schema"]) is list, "worker typed schema")
    require(type(payload["fault_id"]) is str and type(payload["cwd_contract"]) is str, "worker identity")

    invoked_path = Path(sys.executable).resolve(strict=True)
    require(invoked_path == Path(argv[0]).resolve(strict=True), "worker invoked Python path")
    executable = loaded_executable_identity()
    cwd_path = Path.cwd().resolve(strict=True)
    cwd_stat = os.stat(cwd_path, follow_symlinks=False)
    logical_argv = list(argv)
    report = {
        "actual_argv": logical_argv,
        "actual_argv_schema": payload["argv_schema"],
        "actual_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "actual_cwd": {
            "contract": payload["cwd_contract"],
            "dev": cwd_stat.st_dev,
            "ino": cwd_stat.st_ino,
            "path": str(cwd_path),
        },
        "actual_env": dict(os.environ),
        "actual_env_schema": payload["env_schema"],
        "actual_executable": executable,
        "fault_id": payload["fault_id"],
        "pgid": os.getpgid(0),
        "pid": os.getpid(),
        "schema_version": "forkaudit-v10-worker-child-report-v1",
    }
    sys.stdout.buffer.write(canonical_bytes(report))
    sys.stdout.buffer.flush()

    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
    finally:
        if devnull > 2:
            os.close(devnull)
    worker_path = Path(argv[1])
    if not worker_path.is_absolute():
        worker_path = cwd_path / worker_path
    worker_path = worker_path.resolve(strict=True)
    sys.argv = list(argv[1:])
    runpy.run_path(str(worker_path), run_name="__main__")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--torch-init", required=True)
    parser.add_argument("--torch-sha256", required=True)
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--visibility", required=True)
    parser.add_argument("--physical-uuid", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--python-executable", required=True)
    parser.add_argument("--python-sha256", required=True)
    parser.add_argument("--python-loaded-executable", required=True)
    parser.add_argument("--python-loaded-sha256", required=True)
    parser.add_argument("--python-implementation", required=True)
    parser.add_argument("--python-version", required=True)
    parser.add_argument("--python-cache-tag", required=True)
    parser.add_argument("--runner-root", required=True)
    parser.add_argument("--runner-manifest", required=True)
    parser.add_argument("--runner-manifest-sha256", required=True)
    parser.add_argument("--runner-inventory-sha256", required=True)
    args = parser.parse_args()

    require(sys.flags.isolated == 1, "not isolated")
    require(sys.flags.no_site == 1, "site enabled")
    require(sys.flags.ignore_environment == 1, "environment not ignored")
    require("torch" not in sys.modules, "preloaded/fake sys.modules torch")
    preimport_absent = True

    runner_root = Path(args.runner_root).resolve(strict=True)
    manifest_path = Path(args.runner_manifest).resolve(strict=True)
    expected_manifest = canonical_manifest(manifest_path, args.runner_manifest_sha256)
    scan_runner(runner_root, expected_manifest, args.runner_inventory_sha256)
    python_path = Path(args.python_executable).resolve(strict=True)
    require(Path(sys.executable).resolve(strict=True) == python_path, "Python executable identity")
    require(regular_digest(python_path) == args.python_sha256, "Python executable digest")
    loaded_python = loaded_executable_identity()
    require(
        loaded_python["path"]
        == str(Path(args.python_loaded_executable).resolve(strict=True)),
        "loaded Python executable path",
    )
    require(
        loaded_python["sha256"] == args.python_loaded_sha256,
        "loaded Python executable digest",
    )
    observed_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    require(sys.implementation.name == args.python_implementation == "cpython", "Python implementation")
    require(observed_version == args.python_version, "Python version")
    require(sys.implementation.cache_tag == args.python_cache_tag, "Python cache tag")

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
    require(type(getattr(torch, "__version__", None)) is str, "torch version type")
    require(torch.__version__ == args.torch_version, "torch version")

    manifest_by_path = {row["path"]: row for row in expected_manifest}
    observed_torch_files = set()
    for module_name, module in sorted(sys.modules.items()):
        if module_name != "torch" and not module_name.startswith("torch."):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            continue
        resolved_module = Path(module_file).resolve(strict=True)
        try:
            relative_module = resolved_module.relative_to(runner_root).as_posix()
        except ValueError:
            fail("torch import escaped runtime closure")
        require(relative_module in manifest_by_path, "torch import absent from closure")
        require(regular_digest(resolved_module) == manifest_by_path[relative_module]["sha256"], "torch import digest")
        observed_torch_files.add(relative_module)
    require(init_path.relative_to(runner_root).as_posix() in observed_torch_files, "torch root absent from import closure")

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
        "python_cache_tag": sys.implementation.cache_tag,
        "python_executable": str(python_path),
        "python_executable_sha256": args.python_sha256,
        "python_implementation": sys.implementation.name,
        "python_loaded_executable": loaded_python,
        "python_version": observed_version,
        "runner_inventory_sha256": args.runner_inventory_sha256,
        "runner_manifest_sha256": args.runner_manifest_sha256,
        "schema_version": "forkaudit-v10-torch-provenance-v1",
        "spec_origin": str(init_path),
        "torch_version": torch.__version__,
        "visibility": args.visibility,
    }
    sys.stdout.write(json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--worker-payload":
        raise SystemExit(worker_main(sys.argv[2]))
    raise SystemExit(main())
