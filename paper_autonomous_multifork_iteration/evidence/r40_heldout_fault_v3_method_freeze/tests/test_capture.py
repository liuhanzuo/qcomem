from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

from tests.helpers import digest, fake_authority
from v3_capture import (
    BoundTensor, CanonicalLiveTensorReader, CaptureIdentity, CaptureWrapper,
    RequestTensorBindings, TEST_BACKEND_KIND, TensorMaterial,
)
from v3_common import ContractError


@dataclass
class FakeTensor:
    raw: bytes
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    device: str = "cpu-test:0"


@dataclass
class FakeScalar:
    value: int


class FakeBackend:
    kind = TEST_BACKEND_KIND

    def __init__(self) -> None:
        self.synchronize_count = 0
        self.reset_count = 0
        self.allocator_rows = iter([(100, 100), (120, 120), (130, 150), (130, 150), (100, 150)])

    def synchronize(self) -> None:
        self.synchronize_count += 1

    def materialize(self, tensor):
        if not isinstance(tensor, FakeTensor):
            raise ContractError("fake tensor required")
        return TensorMaterial(tensor.dtype, tensor.shape, tensor.stride, tensor.device, tensor.raw)

    def scalar_int(self, tensor):
        if not isinstance(tensor, FakeScalar) or type(tensor.value) is not int or tensor.value < 0:
            raise ContractError("fake scalar required")
        return tensor.value

    def reset_peak_allocator(self) -> None:
        self.reset_count += 1

    def allocator_current_peak(self):
        return next(self.allocator_rows)


def bindings():
    rows = []
    for request_id in ("request-a", "request-b"):
        rows.append(RequestTensorBindings(
            request_id=request_id,
            kv_tensors=(BoundTensor("kv.main", FakeTensor(b"kv0", "uint8", (3,), (1,))),),
            gdn_tensors=(BoundTensor("gdn.main", FakeTensor(b"gdn0", "uint8", (4,), (1,))),),
            kv_logical_length=FakeScalar(0), kv_version=FakeScalar(0),
            gdn_version=FakeScalar(0), kv_commit_epoch=FakeScalar(0),
            gdn_commit_epoch=FakeScalar(0),
        ))
    return rows


def identity(authority):
    return CaptureIdentity(
        campaign_id="R40-V3-HELDOUT-FAULTS", run_id="r40-v3-formal-test",
        lane="reference", fault_id="V3F01", gpu_uuid="GPU-test-0", device_index=0,
        authoritative_config_sha256=authority.config_file_sha256,
        formal_config_sha256=digest("formal-config"),
        method_core_manifest_sha256=authority.method_core_manifest_sha256,
        preregistration_sha256=authority.preregistration_sha256,
        schedule_sha256=authority.schedule_sha256,
        atomic_policy_sha256=authority.atomic_policy_sha256,
        designer_snapshot_manifest_sha256=authority.designer_snapshot_manifest_sha256,
    )


class CaptureTests(unittest.TestCase):
    def test_wrapper_synchronizes_and_reads_mutated_live_tensors_not_model_state(self) -> None:
        authority = fake_authority()
        backend = FakeBackend()
        bound = bindings()
        reader = CanonicalLiveTensorReader(bound, backend, "GPU-test-0", 0)
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = CaptureWrapper(authority, identity(authority), Path(temporary), reader, backend)
            request = bound[0]

            def model_call():
                request.kv_tensors[0].tensor.raw = b"kv1"
                request.gdn_tensors[0].tensor.raw = b"gdn1"
                request.kv_logical_length.value = 32
                request.kv_version.value = 1
                request.gdn_version.value = 1
                request.kv_commit_epoch.value = 1
                request.gdn_commit_epoch.value = 1
                return {
                    "token_id": 7,
                    "logits_tensor": FakeTensor(b"\x00" * (248320 * 4), "float32", (1, 248320), (248320, 1)),
                }

            receipt = wrapper.capture_call(0, model_call)
            self.assertEqual(backend.synchronize_count, 2)
            self.assertNotEqual(receipt["live_pre"]["kv_content_sha256"], receipt["live_post"]["kv_content_sha256"])
            self.assertEqual(receipt["live_post"]["kv_logical_length"], 32)
            self.assertEqual(receipt["state_source"], "wrapper_bound_live_tensors_not_model_result")
            self.assertEqual(receipt["logits"]["nbytes"], 248320 * 4)

    def test_model_result_cannot_supply_or_append_state_mapping(self) -> None:
        authority = fake_authority()
        backend = FakeBackend()
        bound = bindings()
        reader = CanonicalLiveTensorReader(bound, backend, "GPU-test-0", 0)
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = CaptureWrapper(authority, identity(authority), Path(temporary), reader, backend)
            with self.assertRaises(ContractError):
                wrapper.capture_call(0, lambda: {
                    "token_id": 7,
                    "logits_tensor": FakeTensor(b"\x00" * (248320 * 4), "float32", (1, 248320), (248320, 1)),
                    "live_state": {"kv_version": 999},
                })

    def test_component_digest_is_canonical_over_role_order(self) -> None:
        first = bindings()
        second = bindings()
        first[0] = RequestTensorBindings(
            request_id="request-a",
            kv_tensors=(
                BoundTensor("z", FakeTensor(b"z", "uint8", (1,), (1,))),
                BoundTensor("a", FakeTensor(b"a", "uint8", (1,), (1,))),
            ),
            gdn_tensors=first[0].gdn_tensors,
            kv_logical_length=first[0].kv_logical_length, kv_version=first[0].kv_version,
            gdn_version=first[0].gdn_version, kv_commit_epoch=first[0].kv_commit_epoch,
            gdn_commit_epoch=first[0].gdn_commit_epoch,
        )
        second[0] = RequestTensorBindings(
            request_id="request-a", kv_tensors=tuple(reversed(first[0].kv_tensors)),
            gdn_tensors=second[0].gdn_tensors,
            kv_logical_length=second[0].kv_logical_length, kv_version=second[0].kv_version,
            gdn_version=second[0].gdn_version, kv_commit_epoch=second[0].kv_commit_epoch,
            gdn_commit_epoch=second[0].gdn_commit_epoch,
        )
        left = CanonicalLiveTensorReader(first, FakeBackend(), "GPU-test-0", 0).read("request-a")
        right = CanonicalLiveTensorReader(second, FakeBackend(), "GPU-test-0", 0).read("request-a")
        self.assertEqual(left["kv_content_sha256"], right["kv_content_sha256"])
        self.assertEqual(left["kv_inventory_sha256"], right["kv_inventory_sha256"])

    def test_allocator_capture_owns_sync_reset_and_fixed_phase_order(self) -> None:
        authority = fake_authority()
        backend = FakeBackend()
        reader = CanonicalLiveTensorReader(bindings(), backend, "GPU-test-0", 0)
        with tempfile.TemporaryDirectory() as temporary:
            wrapper = CaptureWrapper(authority, identity(authority), Path(temporary), reader, backend)
            capture = wrapper.allocator_capture()
            with self.assertRaises(ContractError):
                capture.capture("H1")
            for phase in ("H0", "H1", "H4", "H6", "H7"):
                capture.capture(phase)
            value = capture.finish()
            self.assertEqual(backend.reset_count, 1)
            self.assertEqual([row["phase"] for row in value["endpoints"]], ["H0", "H1", "H4", "H6", "H7"])
            self.assertEqual(len({row["sync_event_id"] for row in value["endpoints"]}), 5)


if __name__ == "__main__":
    unittest.main()

