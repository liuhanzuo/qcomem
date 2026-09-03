#!/usr/bin/env python3
"""Run an existing audited entry point under mandatory Round-39 hooks."""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

from r39_compiled_dispatch_receipts import (
    DispatchReceiptError,
    DispatchReceiptRecorder,
    _write_json,
    install_runtime_hooks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--entrypoint-root", type=Path, required=True)
    parser.add_argument("--triton-cache-root", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--entrypoint", type=Path, required=True)
    parser.add_argument("--entrypoint-args-json", type=Path, required=True)
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    entrypoint_root = args.entrypoint_root.resolve()
    entrypoint = args.entrypoint.resolve()
    try:
        entrypoint.relative_to(entrypoint_root)
    except ValueError as error:
        raise DispatchReceiptError("entry point escapes its reviewed entrypoint root") from error
    raw_args = json.loads(args.entrypoint_args_json.read_text(encoding="utf-8"))
    if not isinstance(raw_args, list) or not all(isinstance(item, str) for item in raw_args):
        raise DispatchReceiptError("entrypoint argument file must be a JSON string list")
    sys.path.insert(0, str(code_root))
    sys.path.insert(0, str(entrypoint_root))
    recorder = DispatchReceiptRecorder(
        cache_root=args.triton_cache_root,
        code_root=code_root,
        runtime_root=args.runtime_root,
    )
    restore = install_runtime_hooks(recorder)
    old_argv = sys.argv
    try:
        sys.argv = [str(entrypoint), *raw_args]
        try:
            runpy.run_path(str(entrypoint), run_name="__main__")
        except SystemExit as exit_value:
            if exit_value.code not in (None, 0):
                raise
    finally:
        sys.argv = old_argv
        restore()
    _write_json(args.receipt, recorder.payload())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
