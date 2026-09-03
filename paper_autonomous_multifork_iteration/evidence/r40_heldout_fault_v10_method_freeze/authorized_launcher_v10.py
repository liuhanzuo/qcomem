#!/usr/bin/env python3
"""Sole formal v10 entrypoint; process authority never escapes this call frame."""

from __future__ import annotations

import argparse
import base64
import copy
import os
import select
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

from v10_guard import (
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
    build_torch_probe_plan,
    lifecycle_gate,
    measure_hashes,
    need,
    validate_execution_contract,
    validate_hash_map,
    validate_runtime_expectation,
    validate_terminal_tree,
    validate_torch_report,
    verify_operator_binding,
)
from v10_runtime import ProtectedParent


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
    consumption_root: str,
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
    durable = Path(consumption_root).resolve(strict=True)
    need(type(attempt) is int and attempt > 0, "attempt")
    need(type(run_nonce) is str, "run nonce")
    terminal_stat = os.lstat(terminal)
    need(stat.S_ISDIR(terminal_stat.st_mode) and not stat.S_ISLNK(terminal_stat.st_mode), "terminal root")
    durable_stat = os.lstat(durable)
    need(stat.S_ISDIR(durable_stat.st_mode) and not stat.S_ISLNK(durable_stat.st_mode), "consumption root")
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
        durable,
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
    probe_plan = build_torch_probe_plan(
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

    binding_sha = digest_bytes(binding_raw)
    run_identity = {
        "attempt": attempt,
        "consumption_root": str(durable),
        "consumption_root_dev": durable_stat.st_dev,
        "consumption_root_ino": durable_stat.st_ino,
        "run_nonce": run_nonce,
        "schema_version": "forkaudit-v10-run-identity-v1",
        "terminal_root": str(terminal),
        "terminal_root_dev": terminal_stat.st_dev,
        "terminal_root_ino": terminal_stat.st_ino,
    }
    run_identity_sha = digest_bytes(canonical_bytes(run_identity))
    consumption_name = f"{run_identity_sha}.consumed.json"
    consumption = {
        "attempt": attempt,
        "binding_sha256": binding_sha,
        "consumption_name": consumption_name,
        "consumption_root": str(durable),
        "consumption_root_dev": durable_stat.st_dev,
        "consumption_root_ino": durable_stat.st_ino,
        "run_nonce": run_nonce,
        "run_identity_sha256": run_identity_sha,
        "schema_version": "forkaudit-v10-authority-consumption-v1",
        "terminal_root": str(terminal),
        "terminal_root_dev": terminal_stat.st_dev,
        "terminal_root_ino": terminal_stat.st_ino,
    }
    consumption_raw = canonical_bytes(consumption)
    consumption_sha = digest_bytes(consumption_raw)
    provenance = {
        "binding_sha256": binding_sha,
        "consumption_sha256": consumption_sha,
        "execution_contract_sha256": digest_bytes(contract_raw),
        "probe_report_sha256": "0" * 64,
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
            "child_report": None,
            "child_report_sha256": None,
            "child_report_verified": False,
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
            durable,
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
        stream = processes[fault_id].stdout
        if stream is not None and not stream.closed:
            stream.close()

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
                and digest_file(durable / consumption_name) == consumption_sha
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
            actual_specs = {
                "schema_version": "forkaudit-v10-spawned-specs-v1",
                "workers": [
                    {
                        "argv": row["actual_argv"],
                        "argv_schema": row["actual_argv_schema"],
                        "cwd": row["actual_cwd"]["path"],
                        "cwd_contract": row["actual_cwd"]["contract"],
                        "env": row["actual_env"],
                        "env_schema": row["actual_env_schema"],
                        "fault_id": row["fault_id"],
                    }
                    for row in workers
                    if row["spawned"] and row["child_report_verified"]
                ],
            }
            actual_specs_verified = (
                len(actual_specs["workers"]) == len(TERM_IDS)
                and digest_bytes(canonical_bytes(actual_specs)) == intended_specs_sha
            )
            kill_required = any(row["kill_required"] for row in workers)
            kill_completed = all(row["kill_completed"] for row in workers) and not kill_errors
            receipts_verified = all(
                (not row["spawned"])
                or (row["wait_completed"] and row["death_confirmed"] and type(row["exit_code"]) is int)
                for row in workers
            )
            verified = actual_specs_verified and inventory_verified and post_verified and provenance_verified and receipts_verified and kill_completed
            gate = {
                "actual_specs_verified": actual_specs_verified,
                "attempt": attempt,
                "consumption_sha256": consumption_sha,
                "inventory_verified": inventory_verified,
                "kill_completion": {"completed": kill_completed, "errors": list(kill_errors), "required": kill_required},
                "post_rehash_verified": post_verified,
                "provenance": copy.deepcopy(provenance),
                "provenance_verified": provenance_verified,
                "receipts_verified": receipts_verified,
                "run_nonce": run_nonce,
                "schema_version": "forkaudit-v10-lifecycle-gate-v1",
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
                    "schema_version": "forkaudit-v10-terminal-v1",
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
        # Burn authority in the separately signed durable root before any worker.
        # Signals remain blocked until durable and local receipts both exist.
        blocked = {signal.SIGINT, signal.SIGTERM}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
        try:
            current_terminal = os.lstat(terminal)
            current_durable = os.lstat(durable)
            need(
                (current_terminal.st_dev, current_terminal.st_ino)
                == (terminal_stat.st_dev, terminal_stat.st_ino),
                "terminal identity changed before consumption",
            )
            need(
                (current_durable.st_dev, current_durable.st_ino)
                == (durable_stat.st_dev, durable_stat.st_ino),
                "consumption identity changed before consumption",
            )
            with ProtectedParent(durable) as publisher:
                publisher.publish_many({consumption_name: consumption_raw}, mode=0o444)
            consumed = True
            with ProtectedParent(terminal) as publisher:
                publisher.publish_many({"AUTHORIZED_CONSUMPTION.json": consumption_raw}, mode=0o444)
            try:
                completed_probe = subprocess.run(
                    probe_plan["argv"],
                    cwd=probe_plan["cwd"],
                    env=probe_plan["env"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    close_fds=True,
                    start_new_session=True,
                    check=False,
                    timeout=probe_plan["timeout_seconds"],
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise Reject("authorized torch probe execution") from exc
            need(
                completed_probe.returncode == 0
                and completed_probe.stderr == b""
                and len(completed_probe.stdout) <= 64 * 1024,
                "authorized torch probe failure",
            )
            probe_report = validate_torch_report(
                completed_probe.stdout,
                probe_plan["expectation"],
                probe_plan["runner_root"],
            )
            provenance["probe_report_sha256"] = digest_bytes(
                canonical_bytes(probe_report)
            )
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
            bootstrap_path = (
                runner / current_expectation["probe"]["manifest_path"]
            ).resolve(strict=True)
            visibility = env["CUDA_VISIBLE_DEVICES"]
            visible = canonical_visibility(visibility)
            expected_device = current_expectation["device"]
            physical_uuid = expected_device["physical_uuid"]
            expected_loaded_path = (
                runner / current_expectation["python"]["loaded_manifest_path"]
            ).resolve(strict=True)
            expected_loaded_stat = os.stat(
                expected_loaded_path, follow_symlinks=False
            )
            worker_payload = {
                "argv": argv,
                "argv_schema": copy.deepcopy(selected["argv_schema"]),
                "cwd_contract": selected["cwd_contract"],
                "env": env,
                "env_schema": copy.deepcopy(selected["env_schema"]),
                "fault_id": fault_id,
            }
            encoded_payload = base64.b64encode(canonical_bytes(worker_payload)).decode("ascii")
            bootstrap_argv = [
                argv[0],
                "-I",
                "-S",
                "-B",
                str(bootstrap_path),
                "--worker-payload",
                encoded_payload,
            ]

            # Only process-creation site. Both formal signals stay blocked until
            # the process handle, PID, PGID, and complete actual receipt are stored.
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
            try:
                process = subprocess.Popen(
                    bootstrap_argv,
                    cwd=str(cwd_path),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
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
                need(process.stdout is not None, "child report pipe")
                ready, _, _ = select.select([process.stdout], [], [], 5.0)
                need(bool(ready), "child report timeout")
                report_raw = process.stdout.readline(256 * 1024)
                process.stdout.close()
                need(bool(report_raw) and len(report_raw) <= 256 * 1024, "child report bytes")
                child_report = canonical_load(report_raw, "worker child report")
                need(type(child_report) is dict, "worker child report object")
                actual_executable = child_report.get("actual_executable")
                actual_cwd = child_report.get("actual_cwd")
                actual_spec = {
                    "argv": child_report.get("actual_argv"),
                    "argv_schema": child_report.get("actual_argv_schema"),
                    "cwd": actual_cwd.get("path") if type(actual_cwd) is dict else None,
                    "cwd_contract": actual_cwd.get("contract") if type(actual_cwd) is dict else None,
                    "env": child_report.get("actual_env"),
                    "env_schema": child_report.get("actual_env_schema"),
                    "fault_id": child_report.get("fault_id"),
                }
                authority_match = (
                    child_report.get("schema_version")
                    == "forkaudit-v10-worker-child-report-v1"
                    and child_report.get("pid") == process.pid
                    and child_report.get("pgid") == receipt["pgid"] == process.pid
                    and child_report.get("actual_argv") == argv
                    and child_report.get("actual_argv_schema") == selected["argv_schema"]
                    and child_report.get("actual_env") == env
                    and child_report.get("actual_env_schema") == selected["env_schema"]
                    and child_report.get("actual_cuda_visible_devices") == visibility
                    and type(actual_executable) is dict
                    and actual_executable.get("path")
                    == str(expected_loaded_path)
                    and actual_executable.get("dev") == expected_loaded_stat.st_dev
                    and actual_executable.get("ino") == expected_loaded_stat.st_ino
                    and actual_executable.get("sha256")
                    == current_expectation["python"]["loaded_sha256"]
                    and type(actual_cwd) is dict
                    and actual_cwd.get("path") == str(cwd_path)
                    and actual_cwd.get("contract") == selected["cwd_contract"]
                    and actual_cwd.get("dev") == cwd_stat.st_dev
                    and actual_cwd.get("ino") == cwd_stat.st_ino
                    and visibility == expected_device["visibility"]
                    and visible[expected_device["index"]] == physical_uuid
                    and digest_bytes(canonical_bytes(actual_spec)) == selected_sha
                )
                need(authority_match, "child actual authority mismatch")
                receipt["actual_argv"] = child_report["actual_argv"]
                receipt["actual_argv_schema"] = child_report["actual_argv_schema"]
                receipt["actual_cuda_visible_devices"] = child_report["actual_cuda_visible_devices"]
                receipt["actual_cwd"] = child_report["actual_cwd"]
                receipt["actual_env"] = child_report["actual_env"]
                receipt["actual_env_schema"] = child_report["actual_env_schema"]
                receipt["actual_executable"] = child_report["actual_executable"]
                receipt["actual_physical_uuid"] = physical_uuid
                receipt["authority_match"] = authority_match
                receipt["child_report"] = child_report
                receipt["child_report_sha256"] = digest_bytes(report_raw)
                receipt["child_report_verified"] = authority_match
                receipt["spawned_spec_sha256"] = digest_bytes(canonical_bytes(actual_spec))
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
    validate_terminal_tree(
        terminal, durable, pre_hashes, post, intended_specs_sha, expected_status
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
    parser.add_argument("--consumption-root", required=True)
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
            consumption_root=args.consumption_root,
            attempt=args.attempt,
            run_nonce=args.run_nonce,
        )
    except (Reject, OSError) as exc:
        print(f"HOLD:{type(exc).__name__}:{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
