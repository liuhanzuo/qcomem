#!/usr/bin/env python3
import argparse
import os
import stat
from pathlib import Path


BLOCKED_IMPORT_SHADOWS = (
    "sglang",
    "sglang.py",
    "test_hypic_retained_state_receipt",
    "test_hypic_retained_state_receipt.py",
    "test_run_hypic_same_protocol",
    "test_run_hypic_same_protocol.py",
)


class SafeCwdError(RuntimeError):
    pass


def validate_root_stat(st: os.stat_result) -> None:
    if not stat.S_ISDIR(st.st_mode):
        raise SafeCwdError("fixed cwd is not a directory")
    if st.st_uid != 0 or st.st_gid != 0:
        raise SafeCwdError("fixed cwd is not owned by uid/gid 0/0")
    if stat.S_IMODE(st.st_mode) & 0o022:
        raise SafeCwdError("fixed cwd is group- or world-writable")


def validate_import_shadows(root: Path) -> None:
    present = [
        name
        for name in BLOCKED_IMPORT_SHADOWS
        if (root / name).exists() or (root / name).is_symlink()
    ]
    if present:
        raise SafeCwdError("fixed cwd has import shadows: " + ",".join(present))


def validate_safe_root(root: Path) -> None:
    if str(root) != "/":
        raise SafeCwdError("only the literal filesystem root is authorized")
    validate_root_stat(os.lstat(root))
    validate_import_shadows(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    args = parser.parse_args()
    validate_safe_root(Path(args.root))


if __name__ == "__main__":
    main()
