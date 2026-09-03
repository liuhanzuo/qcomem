"""Zero-argument, fail-closed eight-H20 one-shot executor skeleton."""

from __future__ import annotations

import atexit
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time
from typing import Any, Mapping, Optional, Sequence

from v3_authority import CAMPAIGN_ID, FORMAL_CONFIG_PATH, LANES, load_authority, load_fixed_formal_config
from v3_common import (
    ContractError, open_exclusive_lock, require, sha256_file, write_new_json,
)
from v3_formal import FaultBinding, FormalView, validate_formal_mapping


AUTHORIZATION_VARIABLE = "R40_V3_H20_AUTHORIZED"


@dataclass(frozen=True)
class GPURecord:
    device_index: int
    name: str
    gpu_uuid: str
    memory_used_mib: int


def parse_gpu_inventory(stdout: str) -> tuple[GPURecord, ...]:
    rows = [row.strip() for row in stdout.splitlines() if row.strip()]
    require(len(rows) == 8, "node must expose exactly eight GPUs")
    result = []
    for row in rows:
        parts = [part.strip() for part in row.split(",")]
        require(len(parts) == 4, "GPU inventory row")
        try:
            index = int(parts[0])
            memory = int(parts[3])
        except ValueError as error:
            raise ContractError("GPU inventory numeric field") from error
        require(index == len(result), "GPU index order")
        require(parts[2].startswith("GPU-"), "GPU UUID format")
        require(memory >= 0, "GPU memory reading")
        result.append(GPURecord(index, parts[1], parts[2], memory))
    return tuple(result)


def validate_empty_h20_node(gpus: Sequence[GPURecord], compute_process_stdout: str,
                            formal: FormalView) -> None:
    require(len(gpus) == 8, "eight H20 records")
    require(tuple(row.device_index for row in gpus) == tuple(range(8)), "GPU indices")
    require(all("H20" in row.name for row in gpus), "GPU family")
    require(tuple(row.gpu_uuid for row in gpus) == tuple(row.gpu_uuid for row in formal.faults),
            "specified GPU UUID binding")
    limit = formal.raw["execution_policy"]["max_idle_memory_mib"]
    require(all(row.memory_used_mib <= limit for row in gpus), "GPU node not idle by memory")
    processes = [row.strip() for row in compute_process_stdout.splitlines() if row.strip()]
    processes = [row for row in processes if "No running processes found" not in row]
    require(not processes, "GPU node has compute processes")


def query_and_validate_node(formal: FormalView) -> tuple[GPURecord, ...]:
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.used", "--format=csv,noheader,nounits"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    require(gpu_query.returncode == 0, "GPU inventory query failed")
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30,
    )
    require(process_query.returncode == 0, "GPU compute-process query failed")
    records = parse_gpu_inventory(gpu_query.stdout)
    validate_empty_h20_node(records, process_query.stdout, formal)
    return records


def rehash_fixed_sources_and_config(formal: FormalView) -> Mapping[str, str]:
    authority = load_authority()
    require(sha256_file(FORMAL_CONFIG_PATH) == formal.config_file_sha256, "formal config rehash")
    reparsed = validate_formal_mapping(load_fixed_formal_config(), formal.config_file_sha256, authority)
    require(reparsed.output_root == formal.output_root and reparsed.faults == formal.faults,
            "formal configuration changed")
    return {
        "authoritative_config_sha256": authority.config_file_sha256,
        "method_core_manifest_sha256": authority.method_core_manifest_sha256,
        "formal_config_sha256": formal.config_file_sha256,
        "fault_set_sha256": formal.fault_set_sha256,
    }


class OneShotLocks:
    """Campaign-global plus config-hash locks; neither is removed."""

    def __init__(self, formal: FormalView) -> None:
        require(formal.campaign_parent.is_dir() and not formal.campaign_parent.is_symlink(),
                "precreated campaign parent")
        payload = {
            "schema_version": "forkaudit-method-v3-one-shot-lock-v1",
            "campaign_id": CAMPAIGN_ID,
            "run_id": formal.run_id,
            "formal_config_sha256": formal.config_file_sha256,
            "sealed_output_root": str(formal.output_root),
            "created_unix_ns": time.time_ns(),
        }
        self.global_path = formal.campaign_parent / ".R40_V3_CAMPAIGN_GLOBAL.lock"
        self.global_fd = open_exclusive_lock(self.global_path, dict(payload, lock_kind="campaign_global"))
        self.config_path = formal.campaign_parent / ".locks" / (formal.config_file_sha256 + ".lock")
        try:
            self.config_fd = open_exclusive_lock(self.config_path, dict(payload, lock_kind="config_sha"))
        except BaseException:
            os.close(self.global_fd)
            raise

    def close_descriptors_only(self) -> None:
        for descriptor in (self.global_fd, self.config_fd):
            try:
                os.close(descriptor)
            except OSError:
                pass


class ExecutionFinalizer:
    """Single idempotent finalizer used by signals, exceptions, and normal exit."""

    def __init__(self, formal: FormalView, pre_hashes: Mapping[str, str]) -> None:
        self.formal = formal
        self.pre_hashes = dict(pre_hashes)
        self._mutex = threading.RLock()
        self._finalized = False
        self._processes: set[subprocess.Popen[Any]] = set()
        self._results: dict[str, Mapping[str, Any]] = {}

    def register_process(self, process: subprocess.Popen[Any]) -> None:
        with self._mutex:
            require(not self._finalized, "cannot register after finalization")
            self._processes.add(process)

    def unregister_process(self, process: subprocess.Popen[Any]) -> None:
        with self._mutex:
            self._processes.discard(process)

    def record_result(self, fault_id: str, result: Mapping[str, Any]) -> None:
        with self._mutex:
            require(fault_id not in self._results, "duplicate fault terminal result")
            self._results[fault_id] = dict(result)

    def _kill_process_groups(self) -> list[int]:
        killed = []
        for process in list(self._processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    killed.append(process.pid)
                except ProcessLookupError:
                    pass
        return killed

    def finalize(self, reason: str, signum: Optional[int] = None) -> None:
        with self._mutex:
            if self._finalized:
                return
            self._finalized = True
            killed = self._kill_process_groups()
            try:
                post_hashes: Optional[Mapping[str, str]] = rehash_fixed_sources_and_config(self.formal)
                post_error = None
            except BaseException as error:
                post_hashes = None
                post_error = type(error).__name__ + ": " + str(error)
            terminal_root = self.formal.output_root / "terminals"
            for fault in self.formal.faults:
                result = dict(self._results.get(fault.fault_id, {
                    "fault_id": fault.fault_id,
                    "gpu_uuid": fault.gpu_uuid,
                    "device_index": fault.device_index,
                    "lanes": [],
                    "terminal_class": "interrupted_or_not_started_retained",
                    "scientific_detection_outcome": None,
                }))
                result.update({
                    "schema_version": "forkaudit-method-v3-fault-terminal-v1",
                    "finalizer_reason": reason,
                    "signal": signum,
                    "pre_hashes": self.pre_hashes,
                    "post_hashes": post_hashes,
                    "post_rehash_error": post_error,
                    "retry_count": 0,
                    "payload_tuned": False,
                    "pending_record_retained": True,
                })
                terminal_path = terminal_root / (fault.fault_id + ".terminal.json")
                if not terminal_path.exists():
                    write_new_json(terminal_path, result)
            summary_path = self.formal.output_root / "execution-terminal.json"
            if not summary_path.exists():
                write_new_json(summary_path, {
                    "schema_version": "forkaudit-method-v3-execution-terminal-v1",
                    "campaign_id": CAMPAIGN_ID,
                    "run_id": self.formal.run_id,
                    "reason": reason,
                    "signal": signum,
                    "registered_process_groups_killed": killed,
                    "fault_terminal_count": 8,
                    "pre_hashes": self.pre_hashes,
                    "post_hashes": post_hashes,
                    "post_rehash_error": post_error,
                })


def _render_command(template: Sequence[str], fault_id: str, lane: str) -> list[str]:
    result = []
    for argument in template:
        value = argument.replace("{fault_id}", fault_id).replace("{lane}", lane)
        require("{" not in value and "}" not in value, "unresolved runner placeholder")
        result.append(value)
    return result


def _run_lane(formal: FormalView, fault: FaultBinding, lane: str, remaining: float,
              finalizer: ExecutionFinalizer) -> Mapping[str, Any]:
    log_path = formal.output_root / "logs" / (fault.fault_id + "-" + lane + ".log")
    command = _render_command(formal.runner_command_template, fault.fault_id, lane)
    environment = dict(os.environ)
    environment.update({
        "R40_V3_FAULT_ID": fault.fault_id,
        "R40_V3_LANE": lane,
        "R40_V3_GPU_UUID": fault.gpu_uuid,
        "R40_V3_DEVICE_INDEX": str(fault.device_index),
        "R40_V3_FORMAL_CONFIG_SHA256": formal.config_file_sha256,
    })
    started = time.monotonic()
    with log_path.open("xb") as log_handle:
        process = subprocess.Popen(
            command, cwd=str(formal.runner_root), env=environment, stdout=log_handle,
            stderr=subprocess.STDOUT, shell=False, start_new_session=True,
        )
        finalizer.register_process(process)
        timed_out = False
        try:
            return_code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait()
        finally:
            finalizer.unregister_process(process)
    return {
        "lane": lane,
        "attempt_count": 1,
        "status": "timeout" if timed_out else ("completed" if return_code == 0 else "nonzero_exit"),
        "return_code": return_code,
        "elapsed_seconds": time.monotonic() - started,
        "log_path": str(log_path.relative_to(formal.output_root)),
    }


def _run_fault(formal: FormalView, fault: FaultBinding, finalizer: ExecutionFinalizer) -> Mapping[str, Any]:
    started = time.monotonic()
    deadline = started + 900.0
    lanes = []
    for lane in LANES:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            lanes.append({
                "lane": lane, "attempt_count": 0,
                "status": "not_started_fault_deadline_exhausted", "return_code": None,
                "elapsed_seconds": 0.0, "log_path": None,
            })
        else:
            lanes.append(_run_lane(formal, fault, lane, remaining, finalizer))
    completed = all(row["status"] == "completed" for row in lanes)
    return {
        "fault_id": fault.fault_id, "gpu_uuid": fault.gpu_uuid,
        "device_index": fault.device_index, "lanes": lanes,
        "elapsed_seconds": time.monotonic() - started,
        "terminal_class": "completed_awaiting_frozen_verifier" if completed else "operational_invalid_retained",
        "scientific_detection_outcome": None,
    }


def _precreate_pending_terminals(formal: FormalView, pre_hashes: Mapping[str, str]) -> None:
    terminal_root = formal.output_root / "terminals"
    terminal_root.mkdir()
    for fault in formal.faults:
        write_new_json(terminal_root / (fault.fault_id + ".pending.json"), {
            "schema_version": "forkaudit-method-v3-pending-terminal-v1",
            "campaign_id": CAMPAIGN_ID, "run_id": formal.run_id,
            "fault_id": fault.fault_id, "gpu_uuid": fault.gpu_uuid,
            "device_index": fault.device_index, "status": "pending_before_worker_start",
            "formal_config_sha256": formal.config_file_sha256,
            "pre_hashes": dict(pre_hashes),
        })


def execute_fixed_campaign() -> None:
    """Sole formal executor entry point; it accepts no path or root argument."""

    require(os.environ.get(AUTHORIZATION_VARIABLE) == "yes", "explicit H20 authorization missing")
    authority = load_authority()
    formal_mapping = load_fixed_formal_config()
    formal_sha = sha256_file(FORMAL_CONFIG_PATH)
    formal = validate_formal_mapping(formal_mapping, formal_sha, authority)
    pre_hashes = rehash_fixed_sources_and_config(formal)
    gpu_inventory = query_and_validate_node(formal)
    locks = OneShotLocks(formal)
    require(not formal.output_root.exists(), "sealed output root already exists")
    formal.output_root.mkdir()
    (formal.output_root / "logs").mkdir()
    (formal.output_root / "artifacts").mkdir()
    _precreate_pending_terminals(formal, pre_hashes)
    write_new_json(formal.output_root / "execution-binding.json", {
        "schema_version": "forkaudit-method-v3-execution-binding-v1",
        "campaign_id": CAMPAIGN_ID, "run_id": formal.run_id,
        "formal_config_sha256": formal.config_file_sha256,
        "sealed_output_root": str(formal.output_root),
        "gpu_inventory": [row.__dict__ for row in gpu_inventory],
        "pre_hashes": dict(pre_hashes), "authorization_present": True,
    })
    finalizer = ExecutionFinalizer(formal, pre_hashes)

    def signal_handler(signum: int, _frame: Any) -> None:
        finalizer.finalize("signal", signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(finalizer.finalize, "process_exit", None)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_run_fault, formal, fault, finalizer): fault.fault_id for fault in formal.faults}
            for future in as_completed(futures):
                fault_id = futures[future]
                try:
                    finalizer.record_result(fault_id, future.result())
                except BaseException as error:
                    fault = next(row for row in formal.faults if row.fault_id == fault_id)
                    finalizer.record_result(fault_id, {
                        "fault_id": fault_id, "gpu_uuid": fault.gpu_uuid,
                        "device_index": fault.device_index, "lanes": [],
                        "terminal_class": "worker_exception_retained",
                        "scientific_detection_outcome": None,
                        "exception_type": type(error).__name__, "exception_message": str(error),
                    })
        finalizer.finalize("normal_exit", None)
    except BaseException:
        finalizer.finalize("caught_exception", None)
        raise
    finally:
        locks.close_descriptors_only()


if __name__ == "__main__":
    execute_fixed_campaign()
