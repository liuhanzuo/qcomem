"""Strict parsing of the later fixed formal campaign binding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from v3_authority import Authority, CAMPAIGN_ID, FIXED_CAMPAIGN_PARENT, LANES, MODEL_REVISION
from v3_common import exact_keys, load_json_file, require, require_sha256, sha256_file, verify_seal


FAULT_IDS = tuple("V3F%02d" % index for index in range(1, 9))


@dataclass(frozen=True)
class FaultBinding:
    fault_id: str
    gpu_uuid: str
    device_index: int


@dataclass(frozen=True)
class FormalView:
    raw: Mapping[str, Any]
    config_file_sha256: str
    run_id: str
    campaign_parent: Path
    output_root: Path
    fault_set_path: Path
    fault_set_sha256: str
    faults: tuple[FaultBinding, ...]
    runner_root: Path
    runner_command_template: tuple[str, ...]


def _verify_runner_bundle(value: Mapping[str, Any]) -> Path:
    exact_keys(value, ("root", "manifest_path", "manifest_sha256"), "runner bundle")
    root_value = value.get("root")
    manifest_value = value.get("manifest_path")
    require(isinstance(root_value, str) and isinstance(manifest_value, str), "runner paths")
    root = Path(root_value)
    manifest = Path(manifest_value)
    require(root.is_absolute() and root.is_dir() and not root.is_symlink(), "runner root")
    require(manifest.is_absolute() and manifest.is_file() and not manifest.is_symlink(), "runner manifest")
    require(sha256_file(manifest) == require_sha256(value.get("manifest_sha256"), "runner manifest"),
            "runner manifest hash")
    observed = []
    for row in manifest.read_text(encoding="utf-8").splitlines():
        parts = row.split("  ", 1)
        require(len(parts) == 2, "runner manifest row")
        relative = Path(parts[1])
        require(not relative.is_absolute() and ".." not in relative.parts, "runner relative path")
        member = root / relative
        require(member.is_file() and not member.is_symlink(), "runner member")
        require(member.resolve().is_relative_to(root.resolve()), "runner member escape")
        require(sha256_file(member) == require_sha256(parts[0], "runner member"), "runner member hash")
        observed.append(relative.as_posix())
    require(observed and len(observed) == len(set(observed)), "runner manifest inventory")
    return root


def _validate_fault_set(path: Path, expected_sha: str, snapshot_sha: str) -> None:
    require(path.is_absolute() and path.is_file() and not path.is_symlink(), "fault-set file")
    require(sha256_file(path) == expected_sha, "fault-set hash")
    value = load_json_file(path, "fault set")
    exact_keys(value, ("schema_version", "designer_attestation", "faults"), "fault set")
    require(value.get("schema_version") == "forkaudit-method-v3-fault-set-v1", "fault-set schema")
    attestation = value.get("designer_attestation")
    exact_keys(attestation, (
        "snapshot_manifest_sha256", "inputs_limited_to_snapshot",
        "no_private_source_seen", "no_prior_cases_or_outcomes_seen",
    ), "designer attestation")
    require(attestation == {
        "snapshot_manifest_sha256": snapshot_sha,
        "inputs_limited_to_snapshot": True,
        "no_private_source_seen": True,
        "no_prior_cases_or_outcomes_seen": True,
    }, "designer attestation")
    faults = value.get("faults")
    require(isinstance(faults, list) and len(faults) == 8, "eight fault rows")
    fields = (
        "fault_id", "mechanism_family", "implementation_mutation",
        "activation_call_index", "fixed_payload", "eligibility_witness",
        "scientific_rationale",
    )
    ids = []
    mechanisms = []
    for row in faults:
        exact_keys(row, fields, "fault row")
        ids.append(row.get("fault_id"))
        mechanism = row.get("mechanism_family")
        require(isinstance(mechanism, str) and mechanism, "mechanism family")
        mechanisms.append(mechanism)
        for field in ("implementation_mutation", "eligibility_witness", "scientific_rationale"):
            require(isinstance(row.get(field), str) and row[field], "fault " + field)
        call_index = row.get("activation_call_index")
        require(type(call_index) is int and 0 <= call_index < 16, "activation call index")
        require(isinstance(row.get("fixed_payload"), Mapping), "fixed payload")
    require(tuple(ids) == FAULT_IDS, "fault IDs/order")
    require(len(set(mechanisms)) == 8, "unique mechanism families")


def validate_formal_mapping(value: Mapping[str, Any], config_file_sha256: str,
                            authority: Authority) -> FormalView:
    """Internal parser; public callers obtain ``value`` only from the fixed disk path."""

    require_sha256(config_file_sha256, "formal config file")
    exact_keys(value, (
        "schema_version", "campaign_id", "run_id", "campaign_parent", "output_root",
        "method_authoritative_config_sha256", "method_core_manifest_sha256",
        "preregistration_sha256", "schedule_sha256", "atomic_policy_sha256",
        "designer_snapshot_manifest_sha256", "model_revision", "fault_set", "faults",
        "gpu_uuids", "runner_bundle", "runner_command_template", "execution_policy",
        "payload_sha256",
    ), "formal config")
    verify_seal(value, "formal config")
    require(value.get("schema_version") == "forkaudit-method-v3-formal-execution-v1", "formal schema")
    require(value.get("campaign_id") == CAMPAIGN_ID, "formal campaign")
    run_id = value.get("run_id")
    require(isinstance(run_id, str) and run_id.startswith("r40-v3-formal-"), "formal run ID")
    require(value.get("method_authoritative_config_sha256") == authority.config_file_sha256,
            "formal authoritative-config binding")
    require(value.get("method_core_manifest_sha256") == authority.method_core_manifest_sha256,
            "formal method-core binding")
    require(value.get("preregistration_sha256") == authority.preregistration_sha256,
            "formal preregistration binding")
    require(value.get("schedule_sha256") == authority.schedule_sha256, "formal schedule binding")
    require(value.get("atomic_policy_sha256") == authority.atomic_policy_sha256, "formal atomic binding")
    require(value.get("designer_snapshot_manifest_sha256") == authority.designer_snapshot_manifest_sha256,
            "formal designer-snapshot binding")
    require(value.get("model_revision") == MODEL_REVISION, "formal model revision")

    parent_value = value.get("campaign_parent")
    output_value = value.get("output_root")
    require(isinstance(parent_value, str) and isinstance(output_value, str), "formal roots")
    parent = Path(parent_value)
    output = Path(output_value)
    require(parent.is_absolute() and output.is_absolute(), "formal absolute roots")
    require(parent == FIXED_CAMPAIGN_PARENT, "fixed campaign parent binding")
    require(output == parent / "output", "sole sealed output root")
    require(".." not in parent.parts and not parent.is_symlink(), "campaign parent path")

    fault_binding = value.get("fault_set")
    exact_keys(fault_binding, ("path", "sha256"), "fault-set binding")
    fault_path_value = fault_binding.get("path")
    require(isinstance(fault_path_value, str), "fault-set path")
    fault_path = Path(fault_path_value)
    fault_sha = require_sha256(fault_binding.get("sha256"), "fault set")
    _validate_fault_set(fault_path, fault_sha, authority.designer_snapshot_manifest_sha256)

    rows = value.get("faults")
    require(isinstance(rows, list) and len(rows) == 8, "formal fault bindings")
    faults = []
    for index, row in enumerate(rows):
        exact_keys(row, ("fault_id", "gpu_uuid", "device_index"), "formal fault row")
        require(row.get("fault_id") == FAULT_IDS[index], "formal fault order")
        gpu_uuid = row.get("gpu_uuid")
        require(isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-"), "formal GPU UUID")
        require(row.get("device_index") == index, "formal device index")
        faults.append(FaultBinding(FAULT_IDS[index], gpu_uuid, index))
    gpu_uuids = value.get("gpu_uuids")
    require(gpu_uuids == [row.gpu_uuid for row in faults], "formal GPU UUID order")
    require(len(set(gpu_uuids)) == 8, "distinct formal GPU UUIDs")

    runner_root = _verify_runner_bundle(value.get("runner_bundle"))
    command = value.get("runner_command_template")
    require(isinstance(command, list) and command and all(isinstance(item, str) and item for item in command),
            "runner command template")
    joined = "\0".join(command)
    require("{fault_id}" in joined and "{lane}" in joined, "runner command placeholders")
    require("{output_root}" not in joined and "{config_path}" not in joined,
            "caller root/config placeholder forbidden")

    policy = value.get("execution_policy")
    exact_keys(policy, (
        "gpu_count", "gpu_family_substring", "empty_compute_processes_required",
        "max_idle_memory_mib", "lanes", "timeout_seconds_per_fault", "retry_count",
        "payload_tuning_allowed", "campaign_global_lock", "config_sha_lock",
        "pending_terminal_count", "pre_post_rehash",
    ), "execution policy")
    require(policy == {
        "gpu_count": 8, "gpu_family_substring": "H20",
        "empty_compute_processes_required": True, "max_idle_memory_mib": 256,
        "lanes": list(LANES), "timeout_seconds_per_fault": 900,
        "retry_count": 0, "payload_tuning_allowed": False,
        "campaign_global_lock": True, "config_sha_lock": True,
        "pending_terminal_count": 8, "pre_post_rehash": True,
    }, "execution policy")
    return FormalView(
        raw=value, config_file_sha256=config_file_sha256, run_id=run_id,
        campaign_parent=parent, output_root=output, fault_set_path=fault_path,
        fault_set_sha256=fault_sha, faults=tuple(faults), runner_root=runner_root,
        runner_command_template=tuple(command),
    )
