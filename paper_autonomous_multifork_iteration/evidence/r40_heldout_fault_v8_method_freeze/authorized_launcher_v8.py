#!/usr/bin/env python3
"""The sole formal v8 entrypoint from signed binding to closed terminals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from v8_guard import Reject, TERM_IDS, need, validate_terminal_tree
from v8_runtime import Lifecycle, ProtectedParent, prepare_authorized_plan


FORMAL_AUTHORIZED_LAUNCHER = True


def run_authorized_campaign(
    *,
    binding_path: str,
    archive_path: str,
    source_ledger_path: str,
    snapshot_root: str,
    execution_contract_path: str,
    runtime_expectation_path: str,
    runner_root: str,
    runner_manifest_path: str,
    terminal_root: str,
) -> int:
    # This entire preflight, including the isolated torch probe, finishes before
    # Lifecycle can create a worker process.
    plan = prepare_authorized_plan(
        binding_path,
        archive_path,
        source_ledger_path,
        snapshot_root,
        execution_contract_path,
        runtime_expectation_path,
        runner_root,
        runner_manifest_path,
    )
    terminal = Path(terminal_root)
    need(terminal.is_dir(), "terminal root must pre-exist")
    need(not any(terminal.iterdir()), "terminal root must be empty")
    with ProtectedParent(terminal) as parent:
        lifecycle = Lifecycle(
            parent,
            plan,
            archive_path,
            source_ledger_path,
            snapshot_root,
        )
        lifecycle.install_signal_handlers()
        try:
            lifecycle.start()
            lifecycle.spawn_workers()
            codes = lifecycle.wait_workers()
            reason = "success" if codes == [0] * len(TERM_IDS) else "worker-nonzero"
            result = lifecycle.finalize(reason)
        except BaseException as exc:
            result = lifecycle.finalize(f"launcher-failure:{type(exc).__name__}")
        finally:
            lifecycle.restore_signal_handlers()
    expected_status = "success" if result == 0 else "failure"
    validate_terminal_tree(
        terminal,
        lifecycle.pre_hashes,
        lifecycle._post_hashes()[0],
        expected_status,
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--binding", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--source-ledger", required=True)
    parser.add_argument("--snapshot-root", required=True)
    parser.add_argument("--execution-contract", required=True)
    parser.add_argument("--runtime-expectation", required=True)
    parser.add_argument("--runner-root", required=True)
    parser.add_argument("--runner-manifest", required=True)
    parser.add_argument("--terminal-root", required=True)
    args = parser.parse_args(argv)
    try:
        return run_authorized_campaign(
            binding_path=args.binding,
            archive_path=args.archive,
            source_ledger_path=args.source_ledger,
            snapshot_root=args.snapshot_root,
            execution_contract_path=args.execution_contract,
            runtime_expectation_path=args.runtime_expectation,
            runner_root=args.runner_root,
            runner_manifest_path=args.runner_manifest,
            terminal_root=args.terminal_root,
        )
    except (Reject, OSError) as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
