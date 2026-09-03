#!/usr/bin/env python3
"""Local CPU-only integration child used to prove real signal finalization."""

from __future__ import annotations

import argparse
import json
import signal
from pathlib import Path

from v6_guard import canonical_bytes, canonical_load
from v6_runtime import Lifecycle, ProtectedParent


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
    manifest_raw = Path(args.runner_manifest).read_bytes()
    manifest = canonical_load(manifest_raw, "signal fixture manifest")
    with ProtectedParent(args.terminal_root) as parent:
        lifecycle = Lifecycle(
            parent,
            args.archive,
            args.ledger,
            args.snapshot_root,
            args.runner_root,
            manifest,
        )
        lifecycle.install_signal_handlers()
        lifecycle.start()
        Path(args.ready).write_bytes(canonical_bytes({"ready": True}))
        while True:
            signal.pause()


if __name__ == "__main__":
    raise SystemExit(main())
