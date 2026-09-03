"""Zero-argument loading of the authoritative method and formal configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from v3_common import (
    exact_keys,
    load_json_file,
    require,
    require_sha256,
    sha256_file,
    verify_seal,
)
from v3_constants import (
    ATOMIC_POLICY_SHA256,
    AUTHORITATIVE_CONFIG_SHA256,
    DESIGNER_SNAPSHOT_MANIFEST_SHA256,
    METHOD_CORE_MANIFEST_SHA256,
    PREREGISTRATION_SHA256,
    SCHEDULE_SHA256,
)


METHOD_ROOT = Path(__file__).resolve().parents[1]
AUTHORITATIVE_CONFIG_PATH = METHOD_ROOT / "authoritative_config.json"
FORMAL_CONFIG_PATH = METHOD_ROOT / "formal/formal-execution.json"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
CAMPAIGN_ID = "R40-V3-HELDOUT-FAULTS"
REQUEST_IDS = ("request-a", "request-b")
ALLOCATOR_PHASES = ("H0", "H1", "H4", "H6", "H7")
LANES = ("reference", "clean", "mutant")
VOCAB_SIZE = 248320
FIXED_CAMPAIGN_PARENT = Path("/var/tmp/forkaudit-r40-v3-campaign")


@dataclass(frozen=True)
class Authority:
    config: Mapping[str, Any]
    schedule: tuple[Mapping[str, Any], ...]
    config_file_sha256: str
    method_core_manifest_sha256: str
    preregistration_sha256: str
    schedule_sha256: str
    atomic_policy_sha256: str
    designer_snapshot_manifest_sha256: str


def _verify_hash_manifest_at(root: Path, path: Path, expected_manifest_sha256: str) -> None:
    require(sha256_file(path) == expected_manifest_sha256, "method-core manifest hash")
    rows = path.read_text(encoding="utf-8").splitlines()
    require(rows, "method-core manifest empty")
    observed = []
    for row in rows:
        parts = row.split("  ", 1)
        require(len(parts) == 2, "method-core manifest row")
        expected = require_sha256(parts[0], "method-core member")
        relative = Path(parts[1])
        require(not relative.is_absolute() and ".." not in relative.parts, "method-core path")
        member = root / relative
        require(member.is_file() and not member.is_symlink(), "method-core member")
        require(sha256_file(member) == expected, "method-core member drift")
        observed.append(relative.as_posix())
    require(len(observed) == len(set(observed)), "method-core duplicate member")


def _validate_schedule(schedule_value: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    exact_keys(schedule_value, ("schema_version", "request_order", "calls"), "schedule")
    require(schedule_value.get("schema_version") == "forkaudit-method-v3-schedule-v1", "schedule schema")
    require(schedule_value.get("request_order") == list(REQUEST_IDS), "request order")
    calls = schedule_value.get("calls")
    require(isinstance(calls, list) and len(calls) == 16, "sixteen-call schedule")
    expected = []
    for round_index in range(8):
        for request_offset, request_id in enumerate(REQUEST_IDS):
            call_index = round_index * 2 + request_offset
            expected.append({
                "call_index": call_index,
                "round_index": round_index,
                "request_id": request_id,
                "input_token_count": 32 if round_index == 0 else 1,
            })
    require(calls == expected, "exact schedule geometry/order/q")
    return tuple(calls)


def _validate_authoritative_config(config: Mapping[str, Any]) -> Authority:
    exact_keys(config, (
        "schema_version", "campaign_id", "model", "geometry", "schedule",
        "semantic_policy", "atomic_policy", "allocator_policy", "preregistration",
        "designer_snapshot", "method_core", "receipt_schema_version",
        "lane_inventory", "formal_config_fixed_path", "formal_campaign_parent",
        "payload_sha256",
    ), "authoritative config")
    verify_seal(config, "authoritative config")
    require(config.get("schema_version") == "forkaudit-method-v3-authoritative-config-v1", "config schema")
    require(config.get("campaign_id") == CAMPAIGN_ID, "campaign ID")
    model = config.get("model")
    exact_keys(model, ("name", "revision"), "model")
    require(model == {"name": "Qwen3.5-35B-A3B", "revision": MODEL_REVISION}, "model binding")
    geometry = config.get("geometry")
    exact_keys(geometry, (
        "request_ids", "request_count", "calls_per_request", "total_call_count",
        "vocab_size", "logit_dtype",
    ), "geometry")
    require(geometry == {
        "request_ids": list(REQUEST_IDS),
        "request_count": 2,
        "calls_per_request": 8,
        "total_call_count": 16,
        "vocab_size": VOCAB_SIZE,
        "logit_dtype": "float32-little-endian",
    }, "frozen geometry")

    schedule_binding = config.get("schedule")
    exact_keys(schedule_binding, ("path", "sha256"), "schedule binding")
    require(schedule_binding == {"path": "schedule.json", "sha256": SCHEDULE_SHA256}, "schedule binding")
    schedule_path = METHOD_ROOT / "schedule.json"
    require(sha256_file(schedule_path) == SCHEDULE_SHA256, "schedule file drift")
    schedule = _validate_schedule(load_json_file(schedule_path, "schedule"))

    semantic = config.get("semantic_policy")
    exact_keys(semantic, (
        "mode", "max_abs_threshold", "relative_l2_threshold", "tokens_exact",
        "complete_vocab_required", "call_order_and_cardinality_exact",
    ), "semantic policy")
    require(semantic == {
        "mode": "exact", "max_abs_threshold": 0.0, "relative_l2_threshold": 0.0,
        "tokens_exact": True, "complete_vocab_required": True,
        "call_order_and_cardinality_exact": True,
    }, "exact semantic policy")

    atomic = config.get("atomic_policy")
    exact_keys(atomic, ("path", "sha256"), "atomic policy binding")
    require(atomic == {"path": "atomic_policy.json", "sha256": ATOMIC_POLICY_SHA256}, "atomic policy binding")
    require(sha256_file(METHOD_ROOT / "atomic_policy.json") == ATOMIC_POLICY_SHA256, "atomic policy drift")

    allocator = config.get("allocator_policy")
    exact_keys(allocator, (
        "phases", "synchronization_required", "sync_event_globally_unique",
        "peak_reset_before_h0", "peak_monotone_nondecreasing", "paired_current_exact",
        "paired_peak_exact", "h7_current_equals_h0",
    ), "allocator policy")
    require(allocator == {
        "phases": list(ALLOCATOR_PHASES), "synchronization_required": True,
        "sync_event_globally_unique": True, "peak_reset_before_h0": True,
        "peak_monotone_nondecreasing": True, "paired_current_exact": True,
        "paired_peak_exact": True, "h7_current_equals_h0": True,
    }, "allocator policy")

    prereg = config.get("preregistration")
    exact_keys(prereg, ("path", "sha256"), "preregistration binding")
    require(prereg == {"path": "preregistration.json", "sha256": PREREGISTRATION_SHA256},
            "preregistration binding")
    require(sha256_file(METHOD_ROOT / "preregistration.json") == PREREGISTRATION_SHA256,
            "preregistration drift")
    snapshot = config.get("designer_snapshot")
    exact_keys(snapshot, ("manifest_path", "manifest_sha256"), "designer snapshot binding")
    require(snapshot == {
        "manifest_path": "designer_snapshot/SHA256SUMS",
        "manifest_sha256": DESIGNER_SNAPSHOT_MANIFEST_SHA256,
    }, "designer snapshot binding")
    require(sha256_file(METHOD_ROOT / "designer_snapshot/SHA256SUMS") == DESIGNER_SNAPSHOT_MANIFEST_SHA256,
            "designer snapshot drift")
    method = config.get("method_core")
    exact_keys(method, ("manifest_path", "manifest_sha256"), "method-core binding")
    require(method == {
        "manifest_path": "method-core.sha256", "manifest_sha256": METHOD_CORE_MANIFEST_SHA256,
    }, "method-core binding")
    _verify_hash_manifest_at(METHOD_ROOT, METHOD_ROOT / "method-core.sha256", METHOD_CORE_MANIFEST_SHA256)
    require(config.get("receipt_schema_version") == "forkaudit-method-v3-call-receipt-v1", "receipt schema")
    inventory = config.get("lane_inventory")
    exact_keys(inventory, ("receipt_count", "logit_sidecar_count", "allocator_file", "binding_file"),
               "lane inventory")
    require(inventory == {
        "receipt_count": 16, "logit_sidecar_count": 16,
        "allocator_file": "allocator.json", "binding_file": "lane-binding.json",
    }, "lane inventory")
    require(config.get("formal_config_fixed_path") == "formal/formal-execution.json", "formal path")
    require(config.get("formal_campaign_parent") == str(FIXED_CAMPAIGN_PARENT), "fixed campaign parent")
    return Authority(
        config=config, schedule=schedule, config_file_sha256=AUTHORITATIVE_CONFIG_SHA256,
        method_core_manifest_sha256=METHOD_CORE_MANIFEST_SHA256,
        preregistration_sha256=PREREGISTRATION_SHA256, schedule_sha256=SCHEDULE_SHA256,
        atomic_policy_sha256=ATOMIC_POLICY_SHA256,
        designer_snapshot_manifest_sha256=DESIGNER_SNAPSHOT_MANIFEST_SHA256,
    )


def load_authority() -> Authority:
    """Load the sole authoritative config from its compiled fixed path."""

    require(sha256_file(AUTHORITATIVE_CONFIG_PATH) == AUTHORITATIVE_CONFIG_SHA256,
            "authoritative config file hash")
    return _validate_authoritative_config(load_json_file(AUTHORITATIVE_CONFIG_PATH, "authoritative config"))


def load_fixed_formal_config() -> Mapping[str, Any]:
    """Load the only permitted formal configuration; no caller path is accepted."""

    authority = load_authority()
    require(FORMAL_CONFIG_PATH.is_file() and not FORMAL_CONFIG_PATH.is_symlink(),
            "formal configuration absent: method freeze remains HOLD")
    config = load_json_file(FORMAL_CONFIG_PATH, "formal config")
    verify_seal(config, "formal config")
    require(config.get("method_authoritative_config_sha256") == authority.config_file_sha256,
            "formal method-config binding")
    require(config.get("method_core_manifest_sha256") == authority.method_core_manifest_sha256,
            "formal method-core binding")
    return config
