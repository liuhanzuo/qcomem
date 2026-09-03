from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import r29_replay_true_concurrent_lifecycle as replay
import r29_true_concurrent_lifecycle as runner


REPO = Path(__file__).resolve().parents[1]
DESIGN = (
    REPO
    / "paper_autonomous_multifork_iteration/evidence/r29_true_concurrency/"
    "design_preregistration.json"
)


class R29TrueConcurrentLifecycleTest(unittest.TestCase):
    def design(self) -> dict[str, object]:
        return json.loads(DESIGN.read_text(encoding="utf-8"))

    def test_preregistration_is_frozen_to_two_stream_bounded_claim(self) -> None:
        design = self.design()
        runner.validate_design(design)
        self.assertTrue(design["created_before_gpu_execution"])
        self.assertEqual(design["resource_requirement"]["gpu_count"], 1)
        self.assertIn(
            "simultaneous execution of individual CUDA kernels",
            design["claim_boundary"]["not_established"],
        )
        self.assertIn(
            "native vLLM-engine scheduling or continuous batching",
            design["claim_boundary"]["not_established"],
        )

    def test_overlap_arithmetic_distinguishes_overlap_from_interleave(self) -> None:
        self.assertEqual(
            runner.interval_overlap_ms(
                (
                    {"start_ms": 1.0, "end_ms": 5.0},
                    {"start_ms": 2.0, "end_ms": 6.0},
                )
            ),
            3.0,
        )
        self.assertLessEqual(
            runner.interval_overlap_ms(
                (
                    {"start_ms": 1.0, "end_ms": 2.0},
                    {"start_ms": 2.5, "end_ms": 3.0},
                )
            ),
            0.0,
        )

    def test_mock_replays_cancel_and_same_slot_replacement(self) -> None:
        result = runner.mock_result(self.design())
        self.assertTrue(result["passed"])
        self.assertEqual(result["lifecycle_replay"]["final_epochs"], [0, 1])
        self.assertEqual(
            result["lifecycle_replay"]["final_owners"],
            ["survivor", "replacement"],
        )

    def test_concurrent_source_uses_threads_barrier_and_distinct_streams(self) -> None:
        source = inspect.getsource(runner.concurrent_batch)
        for needle in (
            "ThreadPoolExecutor",
            "threading.Barrier",
            "torch.cuda.Stream",
            "stream.wait_event(origin)",
            "interval_overlap_ms",
        ):
            self.assertIn(needle, source)
        self.assertNotIn("torch.cuda.default_stream(0))\n            output", source)

    def test_logit_sidecars_roundtrip_through_independent_reader(self) -> None:
        logits = {
            f"sample-{index}": torch.arange(12, dtype=torch.float32).reshape(1, 12)
            + index
            for index in range(4)
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = runner.write_logit_sidecar(root / "logits.bin", logits)
            observed = replay.read_sidecar(root, manifest)
            self.assertEqual(set(observed), set(logits))
            for sample_id, tensor in logits.items():
                np.testing.assert_array_equal(observed[sample_id], tensor.numpy())

    def test_sidecar_tamper_fails_closed(self) -> None:
        logits = {
            f"sample-{index}": torch.zeros(1, 4, dtype=torch.float32) + index
            for index in range(4)
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "logits.bin"
            manifest = runner.write_logit_sidecar(path, logits)
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            with self.assertRaisesRegex(RuntimeError, "sidecar SHA drift"):
                replay.read_sidecar(root, manifest)

    def test_complete_formal_payload_serializes_mocked_cuuid(self) -> None:
        class _CUuuidLike:
            def __str__(self) -> str:
                return "GPU-00000000-1111-2222-3333-444444444444"

        hardware = runner.device_hardware_receipt(
            SimpleNamespace(
                uuid=_CUuuidLike(),
                total_memory=100_000,
                major=9,
                minor=0,
            ),
            cuda_visible_devices="GPU-00000000-1111-2222-3333-444444444444",
            visible_device_count=1,
            name="Mock H20",
        )
        payload = runner.build_formal_result_payload(
            expected_design_sha256="0" * 64,
            design=self.design(),
            input_receipt={"document_token_ids_sha256": "1" * 64},
            hardware=hardware,
            environment={"torch": "mock", "cuda": "mock"},
            serialized={"phase_receipts": []},
            concurrent={"phase_receipts": []},
            oracle={"all_full_vocab_logits_torch_equal": True},
            sidecars={"serialized": {}, "concurrent": {}},
            cross_arm={"full_vocab_logits_torch_equal": True},
            treatment_valid=True,
            primary_success=True,
        )
        encoded = json.dumps(payload, sort_keys=True, allow_nan=False)
        self.assertIn("GPU-00000000-1111-2222-3333-444444444444", encoded)
        self.assertIsInstance(payload["hardware"]["uuid"], str)


if __name__ == "__main__":
    unittest.main()
