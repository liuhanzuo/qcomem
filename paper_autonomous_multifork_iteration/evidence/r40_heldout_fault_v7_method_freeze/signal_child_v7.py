#!/usr/bin/env python3
"""Local CPU-only integration child used to prove real signal finalization."""

from __future__ import annotations

import argparse
import signal
import sys
from pathlib import Path

from v7_guard import TERM_IDS, canonical_bytes
from v7_runtime import Lifecycle, ProtectedParent


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--terminal-root", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--runner-root", required=True)
    parser.add_argument("--runner-manifest", required=True)
    parser.add_argument("--ready", required=True)
    args = parser.parse_args()
    with ProtectedParent(args.terminal_root) as parent:
        lifecycle = Lifecycle(
            parent,
            args.archive,
            args.ledger,
            args.snapshot_root,
            args.runner_root,
            args.runner_manifest,
        )
        lifecycle.install_signal_handlers()
        lifecycle.start()
        child_env = {
            "LC_ALL": "C",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        }
        lifecycle.spawn_workers(
            [
                {
                    "argv": [sys.executable, "-I", "-S", "-B", "-c", "import time;time.sleep(3600)"],
                    "cwd": args.runner_root,
                    "env": child_env,
                    "fault_id": fault_id,
                }
                for fault_id in TERM_IDS
            ]
        )
        Path(args.ready).write_bytes(canonical_bytes({"ready": True}))
        while True:
            signal.pause()


if __name__ == "__main__":
    raise SystemExit(main())
