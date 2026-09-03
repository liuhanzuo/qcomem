"""Canonical live-tensor capture; model returns cannot report observer state."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import struct
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid

from v3_authority import (
    ALLOCATOR_PHASES, Authority, CAMPAIGN_ID, FORMAL_CONFIG_PATH, LANES,
    MODEL_REVISION, VOCAB_SIZE, load_authority, load_fixed_formal_config,
)
from v3_common import (
    canonical_bytes, exact_keys, require, seal_payload, sha256_bytes, sha256_file,
    write_new_bytes, write_new_json,
)


LIVE_SOURCE_KIND = "canonical-live-tensor-reread-v1"
FORMAL_BACKEND_KIND = "torch-cuda-live-v1"
TEST_BACKEND_KIND = "cpu-mock-live-v1"


@dataclass(frozen=True)
class TensorMaterial:
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    device: str
    raw_bytes: bytes


class TensorBackend(Protocol):
    kind: str

    def synchronize(self) -> None:
        ...

    def materialize(self, tensor: Any) -> TensorMaterial:
        ...

    def scalar_int(self, tensor: Any) -> int:
        ...

    def reset_peak_allocator(self) -> None:
        ...

    def allocator_current_peak(self) -> tuple[int, int]:
        ...


class TorchCudaBackend:
    """Formal backend: direct torch CUDA tensor reads owned by the wrapper."""

    kind = FORMAL_BACKEND_KIND

    def __init__(self, device_index: int) -> None:
        require(type(device_index) is int and 0 <= device_index < 8, "CUDA device index")
        self.device_index = device_index

    def _torch(self) -> Any:
        import torch
        return torch

    def synchronize(self) -> None:
        self._torch().cuda.synchronize(self.device_index)

    def _validate(self, tensor: Any) -> Any:
        torch = self._torch()
        require(isinstance(tensor, torch.Tensor), "formal state handle must be torch.Tensor")
        require(tensor.device.type == "cuda" and tensor.device.index == self.device_index,
                "formal tensor CUDA device binding")
        return tensor

    def materialize(self, tensor: Any) -> TensorMaterial:
        tensor = self._validate(tensor)
        contiguous = tensor.detach().contiguous()
        raw = contiguous.view(self._torch().uint8).cpu().numpy().tobytes(order="C")
        dtype = str(tensor.dtype)
        if dtype.startswith("torch."):
            dtype = dtype[len("torch."):]
        return TensorMaterial(
            dtype=dtype,
            shape=tuple(int(item) for item in tensor.shape),
            stride=tuple(int(item) for item in tensor.stride()),
            device=str(tensor.device),
            raw_bytes=raw,
        )

    def scalar_int(self, tensor: Any) -> int:
        tensor = self._validate(tensor)
        require(tensor.numel() == 1, "live scalar tensor cardinality")
        value = tensor.detach().item()
        require(type(value) is int and value >= 0, "live scalar integer")
        return value

    def reset_peak_allocator(self) -> None:
        self._torch().cuda.reset_peak_memory_stats(self.device_index)

    def allocator_current_peak(self) -> tuple[int, int]:
        torch = self._torch()
        current = int(torch.cuda.memory_allocated(self.device_index))
        peak = int(torch.cuda.max_memory_allocated(self.device_index))
        require(current >= 0 and peak >= current, "formal allocator reading")
        return current, peak


@dataclass(frozen=True)
class BoundTensor:
    role: str
    tensor: Any


@dataclass(frozen=True)
class RequestTensorBindings:
    request_id: str
    kv_tensors: tuple[BoundTensor, ...]
    gdn_tensors: tuple[BoundTensor, ...]
    kv_logical_length: Any
    kv_version: Any
    gdn_version: Any
    kv_commit_epoch: Any
    gdn_commit_epoch: Any


@dataclass(frozen=True)
class CaptureIdentity:
    campaign_id: str
    run_id: str
    lane: str
    fault_id: str
    gpu_uuid: str
    device_index: int
    authoritative_config_sha256: str
    formal_config_sha256: str
    method_core_manifest_sha256: str
    preregistration_sha256: str
    schedule_sha256: str
    atomic_policy_sha256: str
    designer_snapshot_manifest_sha256: str


def _component_digest(bindings: Sequence[BoundTensor], backend: TensorBackend, label: str) -> tuple[str, str]:
    require(isinstance(bindings, Sequence) and len(bindings) > 0, label + " tensor bindings")
    ordered = sorted(bindings, key=lambda item: item.role)
    roles = [item.role for item in ordered]
    require(all(isinstance(role, str) and role for role in roles), label + " tensor roles")
    require(len(roles) == len(set(roles)), label + " duplicate tensor role")
    content = hashlib.sha256()
    inventory = []
    for binding in ordered:
        material = backend.materialize(binding.tensor)
        metadata = {
            "role": binding.role,
            "dtype": material.dtype,
            "shape": list(material.shape),
            "stride": list(material.stride),
            "device": material.device,
            "nbytes": len(material.raw_bytes),
        }
        encoded = canonical_bytes(metadata)
        content.update(struct.pack("<Q", len(encoded)))
        content.update(encoded)
        content.update(struct.pack("<Q", len(material.raw_bytes)))
        content.update(material.raw_bytes)
        inventory.append(dict(metadata, content_sha256=sha256_bytes(material.raw_bytes)))
    return content.hexdigest(), sha256_bytes(canonical_bytes(inventory))


class CanonicalLiveTensorReader:
    """Reads live bound tensors; it has no API for caller-provided state maps."""

    def __init__(self, bindings: Sequence[RequestTensorBindings], backend: TensorBackend,
                 gpu_uuid: str, device_index: int) -> None:
        require(backend.kind in (FORMAL_BACKEND_KIND, TEST_BACKEND_KIND), "capture backend kind")
        require(isinstance(gpu_uuid, str) and gpu_uuid.startswith("GPU-"), "capture GPU UUID")
        require(type(device_index) is int and device_index >= 0, "capture device index")
        mapping = {binding.request_id: binding for binding in bindings}
        require(len(mapping) == len(bindings) == 2, "exact two request tensor bindings")
        require(set(mapping) == {"request-a", "request-b"}, "request tensor binding IDs")
        self._bindings = mapping
        self._backend = backend
        self._gpu_uuid = gpu_uuid
        self._device_index = device_index

    @property
    def backend_kind(self) -> str:
        return self._backend.kind

    def read(self, request_id: str) -> dict[str, Any]:
        require(request_id in self._bindings, "live request binding")
        self._backend.synchronize()
        binding = self._bindings[request_id]
        kv_digest, kv_inventory = _component_digest(binding.kv_tensors, self._backend, "KV")
        gdn_digest, gdn_inventory = _component_digest(binding.gdn_tensors, self._backend, "GDN")
        snapshot = {
            "request_id": request_id,
            "kv_logical_length": self._backend.scalar_int(binding.kv_logical_length),
            "kv_content_sha256": kv_digest,
            "gdn_content_sha256": gdn_digest,
            "kv_inventory_sha256": kv_inventory,
            "gdn_inventory_sha256": gdn_inventory,
            "kv_version": self._backend.scalar_int(binding.kv_version),
            "gdn_version": self._backend.scalar_int(binding.gdn_version),
            "kv_commit_epoch": self._backend.scalar_int(binding.kv_commit_epoch),
            "gdn_commit_epoch": self._backend.scalar_int(binding.gdn_commit_epoch),
            "observation_id": "obs-" + uuid.uuid4().hex,
            "sync_event_id": "sync-" + uuid.uuid4().hex,
            "gpu_uuid": self._gpu_uuid,
            "device_index": self._device_index,
            "source_kind": LIVE_SOURCE_KIND,
            "backend_kind": self._backend.kind,
            "synchronized": True,
        }
        return snapshot


class CaptureWrapper:
    """Captures one fixed-schedule lane; constructor is internal to formal binding."""

    def __init__(self, authority: Authority, identity: CaptureIdentity, lane_root: Path,
                 reader: CanonicalLiveTensorReader, backend: TensorBackend) -> None:
        require(identity.campaign_id == CAMPAIGN_ID, "capture campaign")
        require(identity.lane in LANES, "capture lane")
        require(identity.authoritative_config_sha256 == authority.config_file_sha256, "capture config hash")
        require(len(identity.formal_config_sha256) == 64, "capture formal config hash")
        require(identity.method_core_manifest_sha256 == authority.method_core_manifest_sha256, "capture method hash")
        require(identity.preregistration_sha256 == authority.preregistration_sha256, "capture prereg hash")
        require(identity.schedule_sha256 == authority.schedule_sha256, "capture schedule hash")
        require(identity.atomic_policy_sha256 == authority.atomic_policy_sha256, "capture atomic hash")
        require(identity.designer_snapshot_manifest_sha256 == authority.designer_snapshot_manifest_sha256,
                "capture designer hash")
        require(reader.backend_kind == backend.kind, "capture backend identity")
        require(not lane_root.exists() or (lane_root.is_dir() and not lane_root.is_symlink()), "capture lane root")
        lane_root.mkdir(parents=True, exist_ok=True)
        self._authority = authority
        self._identity = identity
        self._lane_root = lane_root
        self._reader = reader
        self._backend = backend
        self._next_call_index = 0
        binding = seal_payload({
            "schema_version": "forkaudit-method-v3-lane-binding-v1",
            "campaign_id": identity.campaign_id,
            "run_id": identity.run_id,
            "lane": identity.lane,
            "fault_id": identity.fault_id,
            "gpu_uuid": identity.gpu_uuid,
            "device_index": identity.device_index,
            "model_revision": MODEL_REVISION,
            "authoritative_config_sha256": identity.authoritative_config_sha256,
            "formal_config_sha256": identity.formal_config_sha256,
            "method_core_manifest_sha256": identity.method_core_manifest_sha256,
            "preregistration_sha256": identity.preregistration_sha256,
            "schedule_sha256": identity.schedule_sha256,
            "atomic_policy_sha256": identity.atomic_policy_sha256,
            "designer_snapshot_manifest_sha256": identity.designer_snapshot_manifest_sha256,
            "backend_kind": backend.kind,
        })
        write_new_json(lane_root / "lane-binding.json", binding)

    def allocator_capture(self) -> "AllocatorCapture":
        return AllocatorCapture(self._identity, self._lane_root, self._backend)

    def capture_call(self, call_index: int, model_call: Callable[[], Mapping[str, Any]]) -> Mapping[str, Any]:
        require(call_index == self._next_call_index, "capture call order")
        schedule = self._authority.schedule[call_index]
        request_id = schedule["request_id"]
        pre = self._reader.read(request_id)
        result = model_call()
        require(isinstance(result, Mapping), "model result")
        exact_keys(result, ("token_id", "logits_tensor"), "model result")
        require(type(result.get("token_id")) is int and result["token_id"] >= 0, "model token")
        logits = self._backend.materialize(result["logits_tensor"])
        require(logits.dtype == "float32", "complete logits dtype")
        require(logits.shape == (1, VOCAB_SIZE), "complete logits shape")
        require(len(logits.raw_bytes) == VOCAB_SIZE * 4, "complete logits bytes")
        post = self._reader.read(request_id)
        relative_logit = "logits/call-%03d.f32le" % call_index
        write_new_bytes(self._lane_root / relative_logit, logits.raw_bytes)
        logit_row = {
            "path": relative_logit,
            "sha256": sha256_bytes(logits.raw_bytes),
            "nbytes": len(logits.raw_bytes),
            "shape": [1, VOCAB_SIZE],
            "dtype": "float32-little-endian",
        }
        receipt = seal_payload({
            "schema_version": "forkaudit-method-v3-call-receipt-v1",
            "campaign_id": self._identity.campaign_id,
            "run_id": self._identity.run_id,
            "lane": self._identity.lane,
            "fault_id": self._identity.fault_id,
            "gpu_uuid": self._identity.gpu_uuid,
            "device_index": self._identity.device_index,
            "call_key": dict(schedule),
            "model_revision": MODEL_REVISION,
            "authoritative_config_sha256": self._identity.authoritative_config_sha256,
            "formal_config_sha256": self._identity.formal_config_sha256,
            "method_core_manifest_sha256": self._identity.method_core_manifest_sha256,
            "preregistration_sha256": self._identity.preregistration_sha256,
            "schedule_sha256": self._identity.schedule_sha256,
            "atomic_policy_sha256": self._identity.atomic_policy_sha256,
            "designer_snapshot_manifest_sha256": self._identity.designer_snapshot_manifest_sha256,
            "surfaced_token_id": result["token_id"],
            "logits": logit_row,
            "live_pre": pre,
            "live_post": post,
            "state_source": "wrapper_bound_live_tensors_not_model_result",
        })
        write_new_json(self._lane_root / ("receipts/call-%03d.json" % call_index), receipt)
        self._next_call_index += 1
        return receipt


def create_fixed_formal_capture(fault_id: str, lane: str, bindings: Sequence[RequestTensorBindings],
                                device_index: int) -> CaptureWrapper:
    """Formal factory: fixed configs only; no root/config argument exists."""

    authority = load_authority()
    formal_mapping = load_fixed_formal_config()
    from v3_formal import validate_formal_mapping
    formal = validate_formal_mapping(formal_mapping, sha256_file(FORMAL_CONFIG_PATH), authority)
    require(lane in LANES, "formal lane")
    matches = [row for row in formal.faults if row.fault_id == fault_id]
    require(len(matches) == 1, "formal fault binding")
    row = matches[0]
    require(row.device_index == device_index, "formal device index")
    output_root = formal.output_root
    identity = CaptureIdentity(
        campaign_id=CAMPAIGN_ID, run_id=formal.run_id, lane=lane, fault_id=fault_id,
        gpu_uuid=row.gpu_uuid, device_index=device_index,
        authoritative_config_sha256=authority.config_file_sha256,
        formal_config_sha256=sha256_file(FORMAL_CONFIG_PATH),
        method_core_manifest_sha256=authority.method_core_manifest_sha256,
        preregistration_sha256=authority.preregistration_sha256,
        schedule_sha256=authority.schedule_sha256,
        atomic_policy_sha256=authority.atomic_policy_sha256,
        designer_snapshot_manifest_sha256=authority.designer_snapshot_manifest_sha256,
    )
    backend = TorchCudaBackend(device_index)
    reader = CanonicalLiveTensorReader(bindings, backend, row.gpu_uuid, device_index)
    return CaptureWrapper(authority, identity, output_root / "artifacts" / fault_id / lane, reader, backend)


class AllocatorCapture:
    """Fixed synchronized allocator endpoint capture bound to one lane."""

    def __init__(self, identity: CaptureIdentity, lane_root: Path, backend: TensorBackend) -> None:
        self._identity = identity
        self._lane_root = lane_root
        self._backend = backend
        self._rows: list[dict[str, Any]] = []
        self._reset_done = False

    def capture(self, phase: str) -> None:
        expected = ALLOCATOR_PHASES[len(self._rows)] if len(self._rows) < len(ALLOCATOR_PHASES) else None
        require(phase == expected, "allocator phase order")
        self._backend.synchronize()
        if phase == "H0":
            self._backend.reset_peak_allocator()
            self._reset_done = True
            self._backend.synchronize()
        current, peak = self._backend.allocator_current_peak()
        require(type(current) is int and type(peak) is int and 0 <= current <= peak,
                "allocator current/peak")
        self._rows.append({
            "phase": phase,
            "current_allocated_bytes": current,
            "peak_allocated_bytes": peak,
            "sync_event_id": "sync-" + uuid.uuid4().hex,
            "synchronized": True,
            "gpu_uuid": self._identity.gpu_uuid,
            "device_index": self._identity.device_index,
        })

    def finish(self) -> Mapping[str, Any]:
        require(self._reset_done and len(self._rows) == len(ALLOCATOR_PHASES),
                "allocator endpoint completeness")
        value = seal_payload({
            "schema_version": "forkaudit-method-v3-allocator-v1",
            "campaign_id": self._identity.campaign_id,
            "run_id": self._identity.run_id,
            "lane": self._identity.lane,
            "fault_id": self._identity.fault_id,
            "gpu_uuid": self._identity.gpu_uuid,
            "device_index": self._identity.device_index,
            "authoritative_config_sha256": self._identity.authoritative_config_sha256,
            "formal_config_sha256": self._identity.formal_config_sha256,
            "method_core_manifest_sha256": self._identity.method_core_manifest_sha256,
            "peak_reset_before_h0": True,
            "endpoints": list(self._rows),
        })
        write_new_json(self._lane_root / "allocator.json", value)
        return value
