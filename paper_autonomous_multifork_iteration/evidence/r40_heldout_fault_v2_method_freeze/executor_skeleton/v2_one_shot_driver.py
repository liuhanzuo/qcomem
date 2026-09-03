"""Fail-closed one-shot dispatcher for a future frozen eight-H20 campaign."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


METHOD_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(METHOD_ROOT / "executed_source"))
from v2_common import (  # noqa: E402
    ContractError,
    require,
    require_sha256,
    safe_relative_path,
    sha256_file,
    verify_seal,
    write_new_json,
)


SCHEMA = "forkaudit-method-v2-formal-execution-v1"
FAULT_SCHEMA = "forkaudit-method-v2-fault-set-v1"
LANES = ("reference", "clean", "mutant")
FAULT_IDS = tuple("V2F%02d" % index for index in range(1, 9))


def _exact_keys(value: Mapping[str, Any], expected: Sequence[str], label: str) -> None:
    require(set(value.keys()) == set(expected), label + " fields")


def _load_json(path: Path, label: str) -> Mapping[str, Any]:
    require(path.is_absolute(), label + " path must be absolute")
    require(path.is_file() and not path.is_symlink(), label + " missing or symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, Mapping), label + " JSON object")
    return value


def _bound_file(row: Mapping[str, Any], path_key: str, hash_key: str, label: str) -> Path:
    path_value = row.get(path_key)
    require(isinstance(path_value, str) and path_value != "", label + " path")
    path = Path(path_value)
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), label + " file")
    expected = require_sha256(row.get(hash_key), label + " SHA")
    require(sha256_file(path) == expected, label + " hash drift")
    return path


def _validate_fault_set(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    _exact_keys(value, ("schema_version", "designer_attestation", "faults"), "fault set")
    require(value.get("schema_version") == FAULT_SCHEMA, "fault set schema")
    attestation = value.get("designer_attestation")
    require(isinstance(attestation, Mapping), "designer attestation")
    _exact_keys(attestation, (
        "inputs_limited_to_snapshot_sha256",
        "no_prior_campaign_material_seen",
        "no_detector_source_seen",
        "no_execution_outcome_seen",
    ), "designer attestation")
    require_sha256(attestation.get("inputs_limited_to_snapshot_sha256"), "designer snapshot SHA")
    for field in (
        "no_prior_campaign_material_seen",
        "no_detector_source_seen",
        "no_execution_outcome_seen",
    ):
        require(attestation.get(field) is True, "designer attestation " + field)
    faults = value.get("faults")
    require(isinstance(faults, list) and len(faults) == 8, "eight frozen faults")
    expected_fields = (
        "fault_id", "mechanism_family", "implementation_mutation",
        "activation_call", "fixed_payload", "eligibility_witness",
        "scientific_rationale",
    )
    observed_ids = []
    mechanisms = []
    for fault in faults:
        require(isinstance(fault, Mapping), "fault row")
        _exact_keys(fault, expected_fields, "fault row")
        observed_ids.append(fault.get("fault_id"))
        mechanism = fault.get("mechanism_family")
        require(isinstance(mechanism, str) and mechanism != "", "mechanism family")
        mechanisms.append(mechanism)
        for field in ("implementation_mutation", "eligibility_witness", "scientific_rationale"):
            require(isinstance(fault.get(field), str) and fault[field] != "", "fault " + field)
        require(isinstance(fault.get("fixed_payload"), Mapping), "fixed payload")
        activation = fault.get("activation_call")
        require(isinstance(activation, Mapping), "activation call")
        _exact_keys(activation, ("request_index", "round_index"), "activation call")
        require(type(activation.get("request_index")) is int and 0 <= activation["request_index"] <= 1,
                "activation request")
        require(type(activation.get("round_index")) is int and 0 <= activation["round_index"] <= 7,
                "activation round")
    require(tuple(observed_ids) == FAULT_IDS, "fault IDs and order")
    require(len(set(mechanisms)) == 8, "duplicate mechanism family")
    return faults


def _verify_runner_manifest(root: Path, manifest: Path) -> None:
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "runner root")
    require(manifest.is_absolute() and manifest.is_file() and not manifest.is_symlink(), "runner manifest")
    resolved_root = root.resolve()
    rows = [line for line in manifest.read_text(encoding="utf-8").splitlines() if line]
    require(rows, "runner manifest empty")
    observed_paths = []
    for row in rows:
        parts = row.split("  ", 1)
        require(len(parts) == 2, "runner manifest row")
        expected = require_sha256(parts[0], "runner file SHA")
        relative = safe_relative_path(parts[1], "runner relative path")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "runner file")
        require(path.resolve().is_relative_to(resolved_root), "runner path escape")
        require(sha256_file(path) == expected, "runner file hash drift")
        observed_paths.append(relative.as_posix())
    require(len(observed_paths) == len(set(observed_paths)), "duplicate runner manifest path")


def validate_formal_config(config_path: Path) -> dict[str, Any]:
    config = dict(_load_json(config_path, "formal config"))
    _exact_keys(config, (
        "schema_version", "campaign_sealed", "frozen_at_utc", "method_freeze",
        "fault_set", "runner_bundle", "gpu_uuids", "runner_command_template",
        "execution_policy", "payload_sha256",
    ), "formal config")
    require(config.get("schema_version") == SCHEMA, "formal config schema")
    require(config.get("campaign_sealed") is True, "campaign not sealed")
    require(isinstance(config.get("frozen_at_utc"), str) and config["frozen_at_utc"] != "", "freeze time")
    verify_seal(config, "formal config")

    method = config.get("method_freeze")
    require(isinstance(method, Mapping), "method binding")
    _exact_keys(method, ("manifest_path", "sha256"), "method binding")
    _bound_file(method, "manifest_path", "sha256", "method manifest")

    fault_binding = config.get("fault_set")
    require(isinstance(fault_binding, Mapping), "fault binding")
    _exact_keys(fault_binding, ("path", "sha256"), "fault binding")
    fault_path = _bound_file(fault_binding, "path", "sha256", "fault set")
    faults = _validate_fault_set(_load_json(fault_path, "fault set"))

    runner = config.get("runner_bundle")
    require(isinstance(runner, Mapping), "runner binding")
    _exact_keys(runner, ("root", "manifest_path", "manifest_sha256"), "runner binding")
    runner_root_value = runner.get("root")
    require(isinstance(runner_root_value, str) and runner_root_value != "", "runner root")
    runner_root = Path(runner_root_value)
    manifest = _bound_file(runner, "manifest_path", "manifest_sha256", "runner manifest")
    _verify_runner_manifest(runner_root, manifest)

    policy = config.get("execution_policy")
    require(isinstance(policy, Mapping), "execution policy")
    _exact_keys(policy, (
        "gpu_count", "gpu_family_substring", "fault_count", "lanes",
        "timeout_seconds_per_fault", "retry_count", "payload_tuning_allowed",
        "overwrite_allowed",
    ), "execution policy")
    require(policy.get("gpu_count") == 8, "eight-GPU policy")
    require(policy.get("gpu_family_substring") == "H20", "H20 policy")
    require(policy.get("fault_count") == 8 == len(faults), "eight-fault policy")
    require(policy.get("lanes") == list(LANES), "lane policy")
    require(policy.get("timeout_seconds_per_fault") == 900, "fault deadline policy")
    require(policy.get("retry_count") == 0, "retry policy")
    require(policy.get("payload_tuning_allowed") is False, "payload policy")
    require(policy.get("overwrite_allowed") is False, "overwrite policy")

    gpu_uuids = config.get("gpu_uuids")
    require(isinstance(gpu_uuids, list) and len(gpu_uuids) == 8, "eight GPU UUIDs")
    require(all(isinstance(item, str) and item.startswith("GPU-") for item in gpu_uuids), "GPU UUID format")
    require(len(set(gpu_uuids)) == 8, "distinct GPU UUIDs")

    command = config.get("runner_command_template")
    require(isinstance(command, list) and command and all(isinstance(item, str) and item for item in command),
            "runner command")
    joined = "\0".join(command)
    for placeholder in ("{fault_set_path}", "{fault_id}", "{lane}", "{lane_output_dir}"):
        require(placeholder in joined, "runner placeholder " + placeholder)
    config["_faults"] = faults
    config["_fault_path"] = str(fault_path)
    config["_runner_root"] = str(runner_root)
    return config


def detect_h20s() -> dict[str, str]:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader,nounits"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    require(completed.returncode == 0, "nvidia-smi failed")
    rows = [row.strip() for row in completed.stdout.splitlines() if row.strip()]
    require(len(rows) == 8, "node must expose exactly eight GPUs")
    result = {}
    for row in rows:
        parts = [part.strip() for part in row.rsplit(",", 1)]
        require(len(parts) == 2 and parts[1].startswith("GPU-"), "nvidia-smi row")
        require("H20" in parts[0], "non-H20 GPU")
        require(parts[1] not in result, "duplicate detected UUID")
        result[parts[1]] = parts[0]
    return result


def _render_command(template: Sequence[str], replacements: Mapping[str, str]) -> list[str]:
    rendered = []
    for argument in template:
        value = argument
        for marker, replacement in replacements.items():
            value = value.replace(marker, replacement)
        require("{" not in value and "}" not in value, "unresolved runner placeholder")
        rendered.append(value)
    return rendered


def _run_lane(command: Sequence[str], cwd: Path, env: Mapping[str, str], log_path: Path,
              timeout_seconds: float) -> dict[str, Any]:
    started = time.monotonic()
    with log_path.open("xb") as log_handle:
        process = subprocess.Popen(
            list(command), cwd=str(cwd), env=dict(env), stdout=log_handle,
            stderr=subprocess.STDOUT, shell=False, start_new_session=True,
        )
        timed_out = False
        try:
            return_code = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            return_code = process.wait()
    return {
        "attempt_count": 1,
        "status": "timeout" if timed_out else ("completed" if return_code == 0 else "nonzero_exit"),
        "return_code": return_code,
        "elapsed_seconds": time.monotonic() - started,
        "log_path": log_path.name,
    }


def _run_fault(fault: Mapping[str, Any], gpu_uuid: str, config: Mapping[str, Any],
               output_root: Path) -> dict[str, Any]:
    fault_id = fault["fault_id"]
    fault_root = output_root / fault_id
    fault_root.mkdir()
    started = time.monotonic()
    deadline = started + 900.0
    lanes = []
    for lane in LANES:
        lane_root = fault_root / lane
        lane_root.mkdir()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            lanes.append({
                "lane": lane,
                "attempt_count": 0,
                "status": "not_started_fault_deadline_exhausted",
                "return_code": None,
                "elapsed_seconds": 0.0,
                "log_path": None,
            })
            continue
        replacements = {
            "{fault_set_path}": config["_fault_path"],
            "{fault_id}": fault_id,
            "{lane}": lane,
            "{lane_output_dir}": str(lane_root),
        }
        command = _render_command(config["runner_command_template"], replacements)
        env = dict(os.environ)
        env.update({
            "CUDA_VISIBLE_DEVICES": gpu_uuid,
            "R40_FAULT_ID": fault_id,
            "R40_LANE": lane,
            "R40_LANE_OUTPUT_DIR": str(lane_root),
        })
        lane_row = _run_lane(command, Path(config["_runner_root"]), env,
                             fault_root / (lane + ".log"), remaining)
        lane_row["lane"] = lane
        lanes.append(lane_row)
    completed_all = all(row["status"] == "completed" for row in lanes)
    terminal = {
        "schema_version": "forkaudit-method-v2-fault-terminal-v1",
        "fault_id": fault_id,
        "gpu_uuid": gpu_uuid,
        "lane_order": list(LANES),
        "lanes": lanes,
        "elapsed_seconds": time.monotonic() - started,
        "terminal_class": "completed_awaiting_frozen_verifier" if completed_all else "operational_invalid_retained",
        "scientific_detection_outcome": None,
        "retry_count": 0,
        "payload_tuned": False,
    }
    write_new_json(fault_root / "terminal.json", terminal)
    return terminal


def execute(config_path: Path, output_root: Path) -> None:
    require(os.environ.get("R40_H20_EXECUTION_AUTHORIZED") == "yes", "explicit authorization missing")
    config = validate_formal_config(config_path)
    require(output_root.is_absolute(), "output root must be absolute")
    require(not output_root.exists(), "output root already exists")
    require(output_root.parent.is_dir(), "output parent missing")
    detected = detect_h20s()
    configured = config["gpu_uuids"]
    require(set(configured) == set(detected), "configured GPU UUID set differs from node")

    output_root.mkdir()
    lock = output_root / "ONE_SHOT_LOCK"
    with lock.open("x", encoding="utf-8") as handle:
        handle.write("one-shot execution started\n")
        handle.flush()
        os.fsync(handle.fileno())
    write_new_json(output_root / "execution-bindings.json", {
        "schema_version": "forkaudit-method-v2-execution-bindings-v1",
        "formal_config_path": str(config_path),
        "formal_config_sha256": sha256_file(config_path),
        "gpu_inventory": detected,
        "authorization_present": True,
        "timeout_seconds_per_fault": 900,
        "lane_order": list(LANES),
        "retry_count": 0,
    })

    terminals = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {
            pool.submit(_run_fault, fault, configured[index], config, output_root): fault["fault_id"]
            for index, fault in enumerate(config["_faults"])
        }
        for future in as_completed(futures):
            fault_id = futures[future]
            try:
                terminals.append(future.result())
            except BaseException as error:
                terminal = {
                    "schema_version": "forkaudit-method-v2-fault-terminal-v1",
                    "fault_id": fault_id,
                    "gpu_uuid": configured[FAULT_IDS.index(fault_id)],
                    "lane_order": list(LANES),
                    "lanes": [],
                    "elapsed_seconds": None,
                    "terminal_class": "dispatcher_exception_retained",
                    "scientific_detection_outcome": None,
                    "retry_count": 0,
                    "payload_tuned": False,
                    "exception_type": type(error).__name__,
                    "exception_message": str(error),
                }
                fault_terminal = output_root / fault_id / "terminal.json"
                if not fault_terminal.exists():
                    write_new_json(fault_terminal, terminal)
                terminals.append(terminal)
    terminals.sort(key=lambda row: row["fault_id"])
    require(tuple(row["fault_id"] for row in terminals) == FAULT_IDS, "terminal completeness")
    write_new_json(output_root / "terminal-ledger.json", {
        "schema_version": "forkaudit-method-v2-terminal-ledger-v1",
        "fault_count": 8,
        "terminals": terminals,
        "scientific_verification_performed": False,
    })


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.execute, "--execute required")
    execute(args.formal_config, args.output_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, OSError, ValueError, json.JSONDecodeError) as error:
        print("HOLD: " + str(error), file=sys.stderr)
        raise SystemExit(64)

