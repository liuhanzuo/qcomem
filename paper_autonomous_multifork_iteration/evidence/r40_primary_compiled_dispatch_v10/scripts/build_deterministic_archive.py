from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import stat
import tarfile
from pathlib import Path, PurePosixPath


FIXED_MTIME = 1787788800
ARCHIVE_ROOT = "paper_autonomous_multifork_iteration"
EVIDENCE_ROOT = f"{ARCHIVE_ROOT}/evidence"
OLD_PACKAGE = f"{EVIDENCE_ROOT}/r40_primary_compiled_dispatch_v9"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_name(name: str) -> str:
    normalized = name.rstrip("/")
    pure = PurePosixPath(normalized)
    if not normalized or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(f"unsafe archive member: {name!r}")
    if pure.as_posix() != normalized or "\\" in normalized:
        raise ValueError(f"noncanonical archive member: {name!r}")
    return normalized


def normalized_info(name: str, *, directory: bool, size: int = 0) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.size = 0 if directory else size
    info.mode = 0o755 if directory else 0o644
    info.mtime = FIXED_MTIME
    info.uid = 0
    info.gid = 0
    info.uname = "root"
    info.gname = "root"
    info.pax_headers = {}
    return info


def load_dependency_snapshot(v9_archive: Path) -> dict[str, tuple[bool, bytes]]:
    entries: dict[str, tuple[bool, bytes]] = {}
    with tarfile.open(v9_archive, "r:gz") as archive:
        for member in archive.getmembers():
            name = canonical_name(member.name)
            if name == OLD_PACKAGE or name.startswith(f"{OLD_PACKAGE}/"):
                continue
            if name in entries:
                raise ValueError(f"duplicate v9 member: {name}")
            if member.isdir():
                entries[name] = (True, b"")
            elif member.isfile():
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"unreadable v9 member: {name}")
                entries[name] = (False, handle.read())
            else:
                raise ValueError(f"unsupported v9 member type: {name}")
    required = {
        ARCHIVE_ROOT,
        EVIDENCE_ROOT,
        f"{EVIDENCE_ROOT}/round_04_rr2_package",
    }
    if not required.issubset(entries):
        raise ValueError("v9 dependency snapshot lacks required roots")
    return entries


def add_v10_source(
    entries: dict[str, tuple[bool, bytes]], source_root: Path
) -> None:
    source_root = source_root.resolve()
    package_name = source_root.name
    if package_name != "r40_primary_compiled_dispatch_v10":
        raise ValueError("source root must be the exact v10 package directory")
    prefix = f"{EVIDENCE_ROOT}/{package_name}"
    entries[prefix] = (True, b"")
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        if relative.parts and relative.parts[0] == "packages":
            if len(relative.parts) == 1:
                entries[f"{prefix}/packages"] = (True, b"")
            continue
        mode = path.lstat().st_mode
        name = canonical_name(f"{prefix}/{relative.as_posix()}")
        if stat.S_ISDIR(mode):
            entries[name] = (True, b"")
        elif stat.S_ISREG(mode):
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}:
                raise ValueError(f"forbidden bytecode path: {relative}")
            entries[name] = (False, path.read_bytes())
        else:
            raise ValueError(f"unsupported source path type: {relative}")


def build_bytes(entries: dict[str, tuple[bool, bytes]]) -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name in sorted(entries):
                directory, data = entries[name]
                archive.addfile(
                    normalized_info(name, directory=directory, size=len(data)),
                    None if directory else io.BytesIO(data),
                )
    return raw.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--v9-archive", required=True, type=Path)
    parser.add_argument("--expected-v9-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("archive overwrite refused")
    if sha256(args.v9_archive) != args.expected_v9_sha256:
        raise ValueError("v9 archive digest mismatch")
    entries = load_dependency_snapshot(args.v9_archive)
    add_v10_source(entries, args.source_root)
    payload = build_bytes(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        args.output.unlink(missing_ok=True)
        raise
    print(f"members={len(entries)} sha256={sha256(args.output)} bytes={len(payload)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
