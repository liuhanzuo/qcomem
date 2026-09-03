from __future__ import annotations

import argparse
import gzip
import io
import os
import stat
import tarfile
from pathlib import Path


FIXED_MTIME = 1787760000


def canonical_root(root: Path) -> Path:
    lexical = Path(os.path.abspath(os.fspath(root)))
    metadata = os.lstat(lexical)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
        raise RuntimeError("archive root must be an exact directory")
    canonical = lexical.resolve(strict=True)
    if canonical != lexical:
        raise RuntimeError("archive canonical root differs from lexical absolute root")
    return lexical


def normalize_output(output_root: Path, output: Path) -> Path:
    raw = Path(output)
    if ".." in raw.parts:
        raise RuntimeError("archive output contains forbidden dotdot component")
    lexical = Path(os.path.abspath(os.fspath(raw)))
    try:
        relative = lexical.relative_to(output_root)
    except ValueError as error:
        raise RuntimeError("archive output must be strictly inside output root") from error
    if not relative.parts:
        raise RuntimeError("archive output must be strictly inside output root")
    parent_metadata = os.lstat(lexical.parent)
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise RuntimeError("archive output parent must be an exact directory")
    canonical_parent = lexical.parent.resolve(strict=True)
    if canonical_parent != lexical.parent or (
        canonical_parent != output_root and output_root not in canonical_parent.parents
    ):
        raise RuntimeError("archive output parent canonical containment drift")
    if os.path.lexists(lexical):
        raise FileExistsError("archive overwrite or special output node")
    return lexical


def regular_bytes(path: Path, expected: os.stat_result) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise RuntimeError("archive regular file must have exactly one link")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(descriptor)
        before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_id != after_id or (before.st_dev, before.st_ino) != (expected.st_dev, expected.st_ino):
            raise RuntimeError("archive source changed during exact read")
        return data
    finally:
        os.close(descriptor)


def collect(root: Path) -> list[tuple[Path, str, os.stat_result]]:
    files: list[tuple[Path, str, os.stat_result]] = []

    def walk(directory: Path) -> None:
        with os.scandir(directory) as entries:
            for entry in sorted(entries, key=lambda item: item.name):
                path = directory / entry.name
                relative = path.relative_to(root)
                if "packages" in relative.parts or "__pycache__" in relative.parts:
                    continue
                metadata = entry.stat(follow_symlinks=False)
                mode = metadata.st_mode
                if stat.S_ISLNK(mode):
                    raise RuntimeError(f"archive symlink forbidden: {relative.as_posix()}")
                if stat.S_ISDIR(mode):
                    walk(path)
                elif stat.S_ISREG(mode):
                    if path.suffix == ".pyc":
                        continue
                    if metadata.st_nlink != 1:
                        raise RuntimeError(f"archive hardlink forbidden: {relative.as_posix()}")
                    files.append((path, relative.as_posix(), metadata))
                else:
                    raise RuntimeError(f"archive special node forbidden: {relative.as_posix()}")

    walk(root)
    return files


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = canonical_root(args.root)
    output_root = canonical_root(args.output_root if args.output_root is not None else root)
    output = normalize_output(output_root, args.output)
    files = [(path, relative, regular_bytes(path, metadata)) for path, relative, metadata in collect(root)]
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for path, relative, data in files:
                info = tarfile.TarInfo(f"{root.name}/{relative}")
                info.size = len(data)
                info.mtime = FIXED_MTIME
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o755 if path.name == "launch_h20.sh" else 0o644
                info.pax_headers = {}
                archive.addfile(info, io.BytesIO(data))
    with output.open("xb") as raw:
        raw.write(buffer.getvalue())
        raw.flush()
        os.fsync(raw.fileno())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
