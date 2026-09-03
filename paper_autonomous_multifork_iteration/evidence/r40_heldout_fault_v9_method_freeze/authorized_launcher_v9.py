#!/usr/bin/env python3
"""Sole formal v9 entrypoint; process authority never escapes this call frame."""

from __future__ import annotations

import argparse
import copy
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from v9_guard import (
    HASH_KEYS,
    TERM_IDS,
    Reject,
    _read_regular,
    canonical_bytes,
    canonical_load,
    canonical_visibility,
    derive_spawned_specs,
    digest_bytes,
    digest_file,
    isolated_torch_probe,
    lifecycle_gate,
    measure_hashes,
    need,
    validate_execution_contract,
    validate_hash_map,
    validate_runtime_expectation,
    validate_terminal_tree,
    verify_operator_binding,
)
from v9_runtime import ProtectedParent


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
    attempt: int,
    run_nonce: str,
) -> int:
    """Consume one signed authority and run exactly its eight derived workers."""
    binding = Path(binding_path)
    archive = Path(archive_path)
    ledger = Path(source_ledger_path)
    snapshot = Path(snapshot_root)
    contract_path = Path(execution_contract_path)
    expectation_path = Path(runtime_expectation_path)
    runner = Path(runner_root)
    manifest = Path(runner_manifest_path)
    terminal = Path(terminal_root).resolve(strict=True)
    need(type(attempt) is int and attempt > 0, "attempt")
    need(type(run_nonce) is str, "run nonce")
    terminal_stat = os.lstat(terminal)
    need(stat.S_ISDIR(terminal_stat.st_mode) and not stat.S_ISLNK(terminal_stat.st_mode), "terminal root")
    need(not any(terminal.iterdir()), "terminal root must be empty")

    # Full signed preflight. No process-creation primitive has been reached yet.
    binding_raw, _ = _read_regular(binding)
    payload = verify_operator_binding(
        binding_raw,
        archive,
        ledger,
        snapshot,
        contract_path,
        expectation_path,
        runner,
        manifest,
        terminal,
        attempt,
        run_nonce,
    )
    expectation_raw, _ = _read_regular(expectation_path)
    expectation = validate_runtime_expectation(
        expectation_raw,
        payload["approved_runtime_expectation_sha256"],
        runner,
        manifest,
        payload["approved_runner_manifest_sha256"],
        payload["approved_runner_inventory_sha256"],
    )
    contract_raw, _ = _read_regular(contract_path)
    contract = validate_execution_contract(
        contract_raw,
        payload["approved_execution_contract_sha256"],
        expectation_path,
        runner,
        manifest,
        payload["approved_runner_manifest_sha256"],
        payload["approved_runner_inventory_sha256"],
        payload["approved_runtime_expectation_sha256"],
    )
    probe_report = isolated_torch_probe(
        expectation_raw,
        payload["approved_runtime_expectation_sha256"],
        runner,
        manifest,
        payload["approved_runner_manifest_sha256"],
        payload["approved_runner_inventory_sha256"],
    )
    intended_specs = derive_spawned_specs(contract, runner)
    intended_specs_sha = digest_bytes(canonical_bytes(intended_specs))
    signed_hashes = {
        "archive_sha256": payload["approved_archive_sha256"],
        "runner_inventory_sha256": payload["approved_runner_inventory_sha256"],
        "runner_manifest_sha256": payload["approved_runner_manifest_sha256"],
        "snapshot_inventory_sha256": payload["approved_snapshot_inventory_sha256"],
        "snapshot_sha256": payload["approved_snapshot_sha256"],
        "source_ledger_sha256": payload["approved_source_ledger_sha256"],
    }
    pre_hashes = measure_hashes(archive, ledger, snapshot, runner, manifest)
    need(pre_hashes == signed_hashes, "signed preflight hash map")

    consumption = {
        "attempt": attempt,
        "binding_sha256": digest_bytes(binding_raw),
        "run_nonce": run_nonce,
        "schema_version": "forkaudit-v9-authority-consumption-v1",
        "terminal_root": str(terminal),
    }
    consumption_raw = canonical_bytes(consumption)
    consumption_sha = digest_bytes(consumption_raw)
    provenance = {
        "binding_sha256": digest_bytes(binding_raw),
        "consumption_sha256": consumption_sha,
        "execution_contract_sha256": digest_bytes(contract_raw),
        "probe_report_sha256": digest_bytes(canonical_bytes(probe_report)),
        "runtime_expectation_sha256": digest_bytes(expectation_raw),
        "spawned_specs_sha256": intended_specs_sha,
    }

    # All mutable execution state is function-local and inaccessible as a module
    # capability. There is no Lifecycle object or worker-spawn callable to steal.
    processes: dict[str, subprocess.Popen] = {}
    kill_errors: list[str] = []
    receipts = {
        fault_id: {
            "actual_argv": None,
            "actual_argv_schema": None,
            "actual_cuda_visible_devices": None,
            "actual_cwd": None,
            "actual_env": None,
            "actual_env_schema": None,
            "actual_executable": None,
            "actual_physical_uuid": None,
            "authority_match": False,
            "death_confirmed": False,
            "exit_code": None,
            "fault_id": fault_id,
            "kill_completed": True,
            "kill_required": False,
            "kill_sent": False,
            "pgid": None,
            "pid": None,
            "spawned": False,
            "spawned_spec_sha256": None,
            "terminate_sent": False,
            "wait_completed": False,
        }
        for fault_id in TERM_IDS
    }
    consumed = False
    done = False
    exit_code: int | None = None
    handling_signal = False
    old_handlers: dict[int, object] = {}

    def rederive_now() -> tuple[dict, dict]:
        """Re-read and re-verify signed command bytes immediately before Popen."""
        current_binding, _ = _read_regular(binding)
        current_payload = verify_operator_binding(
            current_binding,
            archive,
            ledger,
            snapshot,
            contract_path,
            expectation_path,
            runner,
            manifest,
            terminal,
            attempt,
            run_nonce,
        )
        need(digest_bytes(current_binding) == provenance["binding_sha256"], "binding changed")
        current_expectation_raw, _ = _read_regular(expectation_path)
        current_expectation = validate_runtime_expectation(
            current_expectation_raw,
            current_payload["approved_runtime_expectation_sha256"],
            runner,
            manifest,
            current_payload["approved_runner_manifest_sha256"],
            current_payload["approved_runner_inventory_sha256"],
        )
        current_contract_raw, _ = _read_regular(contract_path)
        current_contract = validate_execution_contract(
            current_contract_raw,
            current_payload["approved_execution_contract_sha256"],
            expectation_path,
            runner,
            manifest,
            current_payload["approved_runner_manifest_sha256"],
            current_payload["approved_runner_inventory_sha256"],
            current_payload["approved_runtime_expectation_sha256"],
        )
        need(digest_bytes(current_contract_raw) == provenance["execution_contract_sha256"], "contract changed")
        need(digest_bytes(current_expectation_raw) == provenance["runtime_expectation_sha256"], "expectation changed")
        current_specs = derive_spawned_specs(current_contract, runner)
        need(digest_bytes(canonical_bytes(current_specs)) == intended_specs_sha, "derived specs changed")
        need(measure_hashes(archive, ledger, snapshot, runner, manifest) == signed_hashes, "artifacts changed")
        return current_specs, current_expectation

    def record_wait(fault_id: str, code: int) -> None:
        need(type(code) is int, "actual worker return code")
        receipt = receipts[fault_id]
        receipt["exit_code"] = code
        receipt["wait_completed"] = True
        receipt["death_confirmed"] = processes[fault_id].poll() is not None

    def cleanup_workers() -> None:
        for fault_id in TERM_IDS:
            process = processes.get(fault_id)
            if process is None:
                continue
            receipt = receipts[fault_id]
            current = process.poll()
            if current is not None:
                record_wait(fault_id, current)
                continue
            receipt["kill_required"] = True
            receipt["kill_completed"] = False
            receipt["terminate_sent"] = True
            pgid = receipt["pgid"] if type(receipt["pgid"]) is int else process.pid
            try:
                os.killpg(pgid, signal.SIGTERM)
            except (OSError, ProcessLookupError) as exc:
                kill_errors.append(f"{fault_id}:terminate:{type(exc).__name__}")
            try:
                code = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                receipt["kill_sent"] = True
                try:
                    os.killpg(pgid, signal.SIGKILL)
                except (OSError, ProcessLookupError) as exc:
                    kill_errors.append(f"{fault_id}:kill:{type(exc).__name__}")
                try:
                    code = process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    kill_errors.append(f"{fault_id}:wait:TimeoutExpired")
                    continue
            record_wait(fault_id, code)
            receipt["kill_completed"] = receipt["death_confirmed"]

    def post_hashes() -> tuple[dict, str | None]:
        try:
            return validate_hash_map(measure_hashes(archive, ledger, snapshot, runner, manifest)), None
        except BaseException as exc:
            return {key: None for key in HASH_KEYS}, f"{type(exc).__name__}:{exc}"

    def provenance_unchanged() -> bool:
        try:
            return (
                digest_file(binding) == provenance["binding_sha256"]
                and digest_file(contract_path) == provenance["execution_contract_sha256"]
                and digest_file(expectation_path) == provenance["runtime_expectation_sha256"]
                and digest_bytes(canonical_bytes(intended_specs)) == intended_specs_sha
                and digest_file(terminal / "AUTHORIZED_CONSUMPTION.json") == consumption_sha
            )
        except BaseException:
            return False

    def finalize(reason: str, signum: int | None = None) -> int:
        nonlocal done, exit_code
        if done:
            return exit_code if exit_code is not None else 1
        need(type(reason) is str and bool(reason), "finalize reason")
        need(signum is None or (type(signum) is int and signum in (2, 15)), "finalize signal")
        blocked = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            cleanup_workers()
            post, rehash_error = post_hashes()
            inventory_verified = pre_hashes == signed_hashes
            post_verified = rehash_error is None and pre_hashes == post
            provenance_verified = provenance_unchanged()
            workers = [copy.deepcopy(receipts[fault_id]) for fault_id in TERM_IDS]
            kill_required = any(row["kill_required"] for row in workers)
            kill_completed = all(row["kill_completed"] for row in workers) and not kill_errors
            receipts_verified = all(
                (not row["spawned"])
                or (row["wait_completed"] and row["death_confirmed"] and type(row["exit_code"]) is int)
                for row in workers
            )
            verified = inventory_verified and post_verified and provenance_verified and receipts_verified and kill_completed
            gate = {
                "attempt": attempt,
                "consumption_sha256": consumption_sha,
                "inventory_verified": inventory_verified,
                "kill_completion": {"completed": kill_completed, "errors": list(kill_errors), "required": kill_required},
                "post_rehash_verified": post_verified,
                "provenance": copy.deepcopy(provenance),
                "provenance_verified": provenance_verified,
                "receipts_verified": receipts_verified,
                "run_nonce": run_nonce,
                "schema_version": "forkaudit-v9-lifecycle-gate-v1",
                "terminal_root": str(terminal),
                "verification_complete": verified,
                "workers": workers,
            }
            lifecycle_gate(gate)
            success = False
            if reason == "success" and signum is None and rehash_error is None:
                try:
                    lifecycle_gate(gate, require_success=True)
                    success = True
                except Reject:
                    success = False
            status = "success" if success else "failure"
            terminal_reason = "success" if success else ("success-gate-rejected" if reason == "success" else reason)
            if rehash_error is not None:
                terminal_reason = f"{reason};post-rehash={rehash_error}"
            provenance_sha = digest_bytes(canonical_bytes(provenance))
            outputs = {
                f"{fault_id}.terminal.json": {
                    "fault_id": fault_id,
                    "lifecycle_receipt": gate,
                    "lifecycle_receipt_sha256": digest_bytes(canonical_bytes(gate)),
                    "post_hashes": post,
                    "pre_hashes": pre_hashes,
                    "provenance": provenance,
                    "provenance_sha256": provenance_sha,
                    "reason": terminal_reason,
                    "schema_version": "forkaudit-v9-terminal-v1",
                    "signal": signum,
                    "status": status,
                }
                for fault_id in TERM_IDS
            }
            with ProtectedParent(terminal) as publisher:
                publisher.publish_json_many(outputs)
            done = True
            exit_code = 128 + signum if signum is not None else (0 if success else 1)
            return exit_code
        finally:
            if signum is None:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal handling_signal
        if handling_signal:
            return
        handling_signal = True
        if not consumed:
            raise SystemExit(128 + signum)
        raise SystemExit(finalize("signal", signum))

    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    try:
        # Consume the signed nonce/terminal-root/attempt exactly once. Signals are
        # blocked until both the no-replace receipt and local consumed flag exist.
        blocked = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            with ProtectedParent(terminal) as publisher:
                publisher.publish_many({"AUTHORIZED_CONSUMPTION.json": consumption_raw})
            consumed = True
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

        for index, fault_id in enumerate(TERM_IDS):
            current_specs, current_expectation = rederive_now()
            selected = canonical_load(
                canonical_bytes(current_specs["workers"][index]), "derived worker spec"
            )
            need(selected["fault_id"] == fault_id, "selected worker identity")
            selected_sha = digest_bytes(canonical_bytes(selected))
            need(selected_sha == digest_bytes(canonical_bytes(intended_specs["workers"][index])), "intended worker mismatch")
            argv = list(selected["argv"])
            env = dict(selected["env"])
            cwd_path = Path(selected["cwd"]).resolve(strict=True)
            cwd_stat = os.lstat(cwd_path)
            need(stat.S_ISDIR(cwd_stat.st_mode) and not stat.S_ISLNK(cwd_stat.st_mode), "actual cwd")
            executable_path = Path(argv[0])
            executable_stat = os.lstat(executable_path)
            need(stat.S_ISREG(executable_stat.st_mode) and executable_stat.st_nlink == 1, "actual executable")
            executable_sha = digest_file(executable_path)
            visibility = env["CUDA_VISIBLE_DEVICES"]
            visible = canonical_visibility(visibility)
            expected_device = current_expectation["device"]
            physical_uuid = expected_device["physical_uuid"]
            authority_match = (
                visibility == expected_device["visibility"]
                and visible[expected_device["index"]] == physical_uuid
                and argv[0] == str(executable_path.resolve(strict=True))
                and str(cwd_path) == selected["cwd"]
                and selected_sha == digest_bytes(canonical_bytes(selected))
            )
            need(authority_match, "immediate actual authority mismatch")

            # Only process-creation site. Both formal signals stay blocked until
            # the process handle, PID, PGID, and complete actual receipt are stored.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
            try:
                process = subprocess.Popen(
                    argv,
                    cwd=str(cwd_path),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    close_fds=True,
                    start_new_session=True,
                )
                processes[fault_id] = process
                receipt = receipts[fault_id]
                receipt["spawned"] = True
                receipt["pid"] = process.pid
                receipt["pgid"] = process.pid
                try:
                    receipt["pgid"] = os.getpgid(process.pid)
                except ProcessLookupError:
                    pass
                receipt["actual_argv"] = argv
                receipt["actual_argv_schema"] = copy.deepcopy(selected["argv_schema"])
                receipt["actual_cuda_visible_devices"] = visibility
                receipt["actual_cwd"] = {
                    "contract": selected["cwd_contract"],
                    "dev": cwd_stat.st_dev,
                    "ino": cwd_stat.st_ino,
                    "path": str(cwd_path),
                }
                receipt["actual_env"] = env
                receipt["actual_env_schema"] = copy.deepcopy(selected["env_schema"])
                receipt["actual_executable"] = {
                    "dev": executable_stat.st_dev,
                    "ino": executable_stat.st_ino,
                    "path": argv[0],
                    "sha256": executable_sha,
                }
                receipt["actual_physical_uuid"] = physical_uuid
                receipt["authority_match"] = True
                receipt["spawned_spec_sha256"] = selected_sha
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

        deadline = time.monotonic() + contract["timeout_seconds"]
        codes = []
        for fault_id in TERM_IDS:
            remaining = max(0.001, deadline - time.monotonic())
            try:
                code = processes[fault_id].wait(timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise Reject("worker wait timeout") from exc
            record_wait(fault_id, code)
            codes.append(code)
        result = finalize("success" if codes == [0] * len(TERM_IDS) else "worker-nonzero")
    except SystemExit:
        raise
    except BaseException as exc:
        result = finalize(f"launcher-failure:{type(exc).__name__}") if consumed else 1
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)

    expected_status = "success" if result == 0 else "failure"
    post, _ = post_hashes()
    validate_terminal_tree(terminal, pre_hashes, post, expected_status)
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
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--run-nonce", required=True)
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
            attempt=args.attempt,
            run_nonce=args.run_nonce,
        )
    except (Reject, OSError) as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
