from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx

from macllm_bench.comem_quant import (
    DepthBitPolicy,
    StoredResidual,
    quantize_residual,
    select_depth_bit_policy,
)


class CoMemQuantTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = mx.gpu if mx.metal.is_available() else mx.cpu
        mx.set_default_device(cls.device)

    def test_real_packing_and_error_order(self) -> None:
        values = mx.sin(mx.arange(2 * 3 * 128).reshape(2, 3, 128) / 17).astype(
            mx.float16
        )
        errors = {}
        sizes = {}
        for bits in (2, 4, 8):
            stored = quantize_residual(
                values, depth=7, bits=bits, group_size=64, stream=self.device
            )
            restored = stored.dequantize(stream=self.device)
            error = mx.sqrt(
                mx.mean(
                    mx.square(values.astype(mx.float32) - restored.astype(mx.float32))
                )
            )
            mx.eval(error)
            errors[bits] = float(error.item())
            sizes[bits] = stored.nbytes
            self.assertEqual(restored.shape, values.shape)
            self.assertLess(stored.nbytes, stored.dense_nbytes)
            self.assertEqual(stored.data.dtype, mx.uint32)

        self.assertLess(sizes[2], sizes[4])
        self.assertLess(sizes[4], sizes[8])
        self.assertLess(errors[8], errors[4])
        self.assertLess(errors[4], errors[2])

    def test_safetensors_round_trip(self) -> None:
        values = mx.random.normal((1, 4, 128)).astype(mx.float16)
        stored = quantize_residual(values, depth=9, bits=4, group_size=64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "residual.safetensors"
            stored.save(path)
            loaded = StoredResidual.load(path, stream=self.device)
            expected = stored.dequantize(stream=self.device)
            actual = loaded.dequantize(stream=self.device)
            equal = mx.allclose(expected, actual)
            mx.eval(equal)

        self.assertTrue(bool(equal.item()))
        self.assertEqual(loaded.depth, 9)
        self.assertEqual(loaded.bits, 4)
        self.assertEqual(loaded.original_shape, values.shape)
        self.assertEqual(loaded.nbytes, stored.nbytes)

    def test_depth_policy_parser_and_calibration(self) -> None:
        parsed = DepthBitPolicy.from_specs(["6:4,9:4", "12:8"])
        self.assertEqual(parsed.bits_for(6), 4)
        self.assertEqual(parsed.bits_for(12), 8)
        self.assertEqual(parsed.bits_for(18), 16)

        rows = [
            {
                "depth": 6,
                "bits": 2,
                "stored_nbytes": 20,
                "kl_divergence": 0.2,
                "top1_match": False,
            },
            {
                "depth": 6,
                "bits": 4,
                "stored_nbytes": 40,
                "kl_divergence": 0.01,
                "top1_match": True,
            },
            {
                "depth": 6,
                "bits": 8,
                "stored_nbytes": 80,
                "kl_divergence": 0.001,
                "top1_match": True,
            },
            {
                "depth": 12,
                "bits": 4,
                "stored_nbytes": 40,
                "kl_divergence": 0.04,
                "top1_match": True,
            },
            {
                "depth": 12,
                "bits": 8,
                "stored_nbytes": 80,
                "kl_divergence": 0.005,
                "top1_match": True,
            },
        ]
        selected = select_depth_bit_policy(rows, max_kl=0.02)
        self.assertEqual(selected.as_dict(), {"6": 4, "12": 8})


if __name__ == "__main__":
    unittest.main()
