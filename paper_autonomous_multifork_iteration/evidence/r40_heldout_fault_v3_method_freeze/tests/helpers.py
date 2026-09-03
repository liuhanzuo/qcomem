from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import struct
import sys
from typing import Any, Callable, Mapping, Optional


METHOD_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(METHOD_ROOT / "executed_source"))

from v3_authority import Authority, CAMPAIGN_ID, MODEL_REVISION  # noqa: E402
from v3_capture import FORMAL_BACKEND_KIND, LIVE_SOURCE_KIND  # noqa: E402
from v3_common import canonical_bytes, seal_payload, sha256_bytes  # noqa: E402
from v3_formal import FaultBinding, FormalView  # noqa: E402


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def schedule() -> tuple[Mapping[str, Any], ...]:
    value = json.loads((METHOD_ROOT / "schedule.json").read_text(encoding="utf-8"))
    return tuple(value["calls"])


def fake_authority() -> Authority:
    return Authority(
        config={}, schedule=schedule(), config_file_sha256=digest("authoritative-config"),
        method_core_manifest_sha256=digest("method-core"),
        preregistration_sha256=digest("preregistration"), schedule_sha256=digest("schedule"),
        atomic_policy_sha256=digest("atomic-policy"),
        designer_snapshot_manifest_sha256=digest("designer-snapshot"),
    )


def fake_formal(output_root: Path, faults: Optional[tuple[FaultBinding, ...]] = None) -> FormalView:
    if faults is None:
        faults = (FaultBinding("V3F01", "GPU-test-0", 0),)
    return FormalView(
        raw={"execution_policy": {"max_idle_memory_mib": 256}},
        config_file_sha256=digest("formal-config"), run_id="r40-v3-formal-test",
        campaign_parent=output_root.parent, output_root=output_root,
        fault_set_path=output_root.parent / "fault-set.json", fault_set_sha256=digest("fault-set"),
        faults=faults, runner_root=output_root.parent,
        runner_command_template=("python3", "runner.py", "{fault_id}", "{lane}"),
    )


def fp32(values: list[float]) -> bytes:
    return struct.pack("<%df" % len(values), *values)


def live_snapshot(request_id: str, length: int, version: int, identity_prefix: str,
                  gpu_uuid: str, device_index: int, backend_kind: str = FORMAL_BACKEND_KIND,
                  content_version: Optional[int] = None) -> Mapping[str, Any]:
    content = version if content_version is None else content_version
    return {
        "request_id": request_id,
        "kv_logical_length": length,
        "kv_content_sha256": digest(request_id + "-kv-" + str(content)),
        "gdn_content_sha256": digest(request_id + "-gdn-" + str(content)),
        "kv_inventory_sha256": digest(request_id + "-kv-inventory"),
        "gdn_inventory_sha256": digest(request_id + "-gdn-inventory"),
        "kv_version": version,
        "gdn_version": version,
        "kv_commit_epoch": version,
        "gdn_commit_epoch": version,
        "observation_id": "obs-" + identity_prefix,
        "sync_event_id": "sync-" + identity_prefix,
        "gpu_uuid": gpu_uuid,
        "device_index": device_index,
        "source_kind": LIVE_SOURCE_KIND,
        "backend_kind": backend_kind,
        "synchronized": True,
    }


def write_lane(
    lane_root: Path,
    authority: Authority,
    formal_sha: str,
    fault: FaultBinding,
    lane: str,
    *,
    vocab_size: int = 4,
    receipt_mutator: Optional[Callable[[int, dict[str, Any]], None]] = None,
    logit_mutator: Optional[Callable[[int, bytes], bytes]] = None,
    allocator_mutator: Optional[Callable[[dict[str, Any]], None]] = None,
    id_namespace: Optional[str] = None,
) -> None:
    lane_root.mkdir(parents=True)
    identity = {
        "campaign_id": CAMPAIGN_ID,
        "run_id": "r40-v3-formal-test",
        "lane": lane,
        "fault_id": fault.fault_id,
        "gpu_uuid": fault.gpu_uuid,
        "device_index": fault.device_index,
        "authoritative_config_sha256": authority.config_file_sha256,
        "formal_config_sha256": formal_sha,
        "method_core_manifest_sha256": authority.method_core_manifest_sha256,
    }
    binding = seal_payload(dict(identity, **{
        "schema_version": "forkaudit-method-v3-lane-binding-v1",
        "model_revision": MODEL_REVISION,
        "preregistration_sha256": authority.preregistration_sha256,
        "schedule_sha256": authority.schedule_sha256,
        "atomic_policy_sha256": authority.atomic_policy_sha256,
        "designer_snapshot_manifest_sha256": authority.designer_snapshot_manifest_sha256,
        "backend_kind": FORMAL_BACKEND_KIND,
    }))
    (lane_root / "lane-binding.json").write_bytes(canonical_bytes(binding) + b"\n")
    (lane_root / "receipts").mkdir()
    (lane_root / "logits").mkdir()
    current = {"request-a": (0, 0), "request-b": (0, 0)}
    namespace = id_namespace or (fault.fault_id + "-" + lane)
    for call_index, call in enumerate(authority.schedule):
        request_id = call["request_id"]
        length, version = current[request_id]
        pre = live_snapshot(
            request_id, length, version, "%s-%03d-pre" % (namespace, call_index),
            fault.gpu_uuid, fault.device_index,
        )
        next_length = length + call["input_token_count"]
        next_version = version + 1
        post = live_snapshot(
            request_id, next_length, next_version, "%s-%03d-post" % (namespace, call_index),
            fault.gpu_uuid, fault.device_index,
        )
        current[request_id] = (next_length, next_version)
        raw = fp32([float(call_index), 1.0, 2.0, 3.0])
        if logit_mutator is not None:
            raw = logit_mutator(call_index, raw)
        relative = "logits/call-%03d.f32le" % call_index
        (lane_root / relative).write_bytes(raw)
        receipt = dict(identity, **{
            "schema_version": "forkaudit-method-v3-call-receipt-v1",
            "call_key": dict(call),
            "model_revision": MODEL_REVISION,
            "preregistration_sha256": authority.preregistration_sha256,
            "schedule_sha256": authority.schedule_sha256,
            "atomic_policy_sha256": authority.atomic_policy_sha256,
            "designer_snapshot_manifest_sha256": authority.designer_snapshot_manifest_sha256,
            "surfaced_token_id": call_index + 10,
            "logits": {
                "path": relative, "sha256": sha256_bytes(raw), "nbytes": len(raw),
                "shape": [1, vocab_size], "dtype": "float32-little-endian",
            },
            "live_pre": pre, "live_post": post,
            "state_source": "wrapper_bound_live_tensors_not_model_result",
        })
        if receipt_mutator is not None:
            receipt_mutator(call_index, receipt)
        receipt = seal_payload(receipt)
        (lane_root / ("receipts/call-%03d.json" % call_index)).write_bytes(canonical_bytes(receipt) + b"\n")
    endpoints = []
    currents = [100, 120, 130, 130, 100]
    peaks = [100, 120, 150, 150, 150]
    for index, phase in enumerate(("H0", "H1", "H4", "H6", "H7")):
        endpoints.append({
            "phase": phase, "current_allocated_bytes": currents[index],
            "peak_allocated_bytes": peaks[index],
            "sync_event_id": "sync-%s-allocator-%s" % (namespace, phase),
            "synchronized": True, "gpu_uuid": fault.gpu_uuid,
            "device_index": fault.device_index,
        })
    allocator = dict(identity, **{
        "schema_version": "forkaudit-method-v3-allocator-v1",
        "peak_reset_before_h0": True, "endpoints": endpoints,
    })
    if allocator_mutator is not None:
        allocator_mutator(allocator)
    allocator = seal_payload(allocator)
    (lane_root / "allocator.json").write_bytes(canonical_bytes(allocator) + b"\n")
