import hashlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from hydragen_vllm_flash_compat import (
    _flash_attn_forward,
    _padded_lse,
    flash_attention_seqlen_compat,
    install_flash_attn_compat,
    install_hydragen_runtime_compat,
)
from qcomem_hydragen_transfer import (
    CapturedAttention,
    HydragenTransferError,
    build_replicated_dense_kv,
    build_transfer_case,
    canonical_json_bytes,
    case_definition,
    cpu_fp32_oracle,
    error_metrics,
    load_bound_tensor,
    storage_accounting,
    timing_statistics,
)


class HydragenTransferTests(unittest.TestCase):
    def capture(self):
        torch.manual_seed(7)
        return CapturedAttention(
            document_key=torch.randn(4095, 2, 256, dtype=torch.bfloat16),
            document_value=torch.randn(4095, 2, 256, dtype=torch.bfloat16),
            suffix_query=torch.randn(32, 16, 256, dtype=torch.bfloat16),
            suffix_key=torch.randn(32, 2, 256, dtype=torch.bfloat16),
            suffix_value=torch.randn(32, 2, 256, dtype=torch.bfloat16),
            metadata_sha256="a" * 64,
            sidecar_sha256={},
        )

    def test_case_definitions_are_frozen(self):
        self.assertEqual(case_definition(8), (tuple(range(3, 32, 4)), tuple(range(4, 33, 4))))
        self.assertEqual(case_definition(32), (tuple(range(32)), tuple(range(1, 33))))
        with self.assertRaises(HydragenTransferError):
            case_definition(16)

    def test_case_build_and_dense_layout(self):
        case = build_transfer_case(self.capture(), 8)
        self.assertEqual(tuple(case.query.shape), (8, 1, 16, 256))
        self.assertEqual(tuple(case.unique_key.shape), (8, 32, 2, 256))
        dense_k, dense_v, lengths = build_replicated_dense_kv(case)
        self.assertEqual(tuple(dense_k.shape), (8, 4128, 2, 256))
        self.assertEqual(tuple(dense_v.shape), tuple(dense_k.shape))
        self.assertEqual(lengths.tolist(), [4099, 4103, 4107, 4111, 4115, 4119, 4123, 4127])
        self.assertTrue(torch.equal(dense_k[:, 4127], torch.zeros_like(dense_k[:, 4127])))
        self.assertTrue(torch.equal(case.unique_key[0, 4:], torch.zeros_like(case.unique_key[0, 4:])))

    def test_cpu_oracle_shape_and_identity_metrics(self):
        capture = self.capture()
        case = build_transfer_case(capture, 8)
        oracle = cpu_fp32_oracle(case)
        self.assertEqual(tuple(oracle.shape), (8, 1, 16, 256))
        metrics = error_metrics(oracle, oracle)
        self.assertEqual(metrics["relative_l2"], 0.0)
        self.assertTrue(metrics["argmax_head_dimension_exact"])

    def test_storage_accounting(self):
        case8 = build_transfer_case(self.capture(), 8)
        case32 = build_transfer_case(self.capture(), 32)
        a8 = storage_accounting(case8)
        a32 = storage_accounting(case32)
        self.assertGreater(a8["bytes_avoided"], 0)
        self.assertGreater(a32["replicated_over_hydragen_ratio"], a8["replicated_over_hydragen_ratio"])

    def test_bound_tensor_rejects_tamper(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tensor = torch.arange(8, dtype=torch.int32)
            path = root / "x.bin"
            path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
            raw = path.read_bytes()
            record = {
                "artifact": {
                    "bytes": len(raw),
                    "relative_path": "x.bin",
                    "sha256": hashlib.sha256(raw).hexdigest(),
                },
                "dtype": "torch.int32",
                "encoding": "torch-contiguous-raw-little-endian-v1",
                "shape": [8],
            }
            self.assertTrue(torch.equal(load_bound_tensor(root, record), tensor))
            path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
            with self.assertRaises(HydragenTransferError):
                load_bound_tensor(root, record)

    def test_timing_and_canonical_json(self):
        stats = timing_statistics([1.0, 2.0, 3.0, 4.0])
        self.assertEqual(stats["median_ms"], 2.0)
        self.assertEqual(stats["p90_ms"], 4.0)
        self.assertEqual(canonical_json_bytes({"b": 2, "a": 1}), b'{"a":1,"b":2}\n')

    def test_flashattention_compat_lse_layout_and_install(self):
        raw = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        cu = torch.tensor([0, 2, 6], dtype=torch.int32)
        padded = _padded_lse(raw, cu, 4)
        self.assertEqual(tuple(padded.shape), (2, 4, 4))
        self.assertTrue(torch.equal(padded[0, :, :2], raw[:, :2]))
        self.assertTrue(torch.equal(padded[1], raw[:, 2:6]))
        self.assertTrue(torch.isneginf(padded[0, :, 2:]).all())
        install_flash_attn_compat()
        from flash_attn.flash_attn_interface import _flash_attn_forward
        self.assertTrue(callable(_flash_attn_forward))
        attention = types.SimpleNamespace(flash_attention_seqlen=None)
        flash = types.SimpleNamespace(flash_attention_seqlen=None)
        install_hydragen_runtime_compat(attention, flash)
        self.assertIs(attention.flash_attention_seqlen, flash_attention_seqlen_compat)
        self.assertIs(flash.flash_attention_seqlen, flash_attention_seqlen_compat)

    def test_flashattention_frontends_bind_dense_and_paged_layouts(self):
        calls = []

        def fake_flash(q, k, v, **kwargs):
            calls.append((q.shape, k.shape, v.shape, kwargs))
            return q.clone(), torch.zeros(
                (q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device
            )

        package = types.ModuleType("vllm")
        backend = types.ModuleType("vllm.vllm_flash_attn")
        backend.flash_attn_varlen_func = fake_flash
        package.vllm_flash_attn = backend
        with patch.dict(
            sys.modules,
            {"vllm": package, "vllm.vllm_flash_attn": backend},
        ):
            q = torch.randn(2, 3, 4, 8)
            k = torch.randn(2, 5, 2, 8)
            v = torch.randn_like(k)
            fixed = _flash_attn_forward(
                q,
                k,
                v,
                dropout_p=0.0,
                causal=False,
                softmax_scale=None,
                window_size=(-1, -1),
                return_softmax=False,
            )
            self.assertEqual(tuple(fixed[0].shape), tuple(q.shape))
            self.assertEqual(tuple(fixed[5].shape), (2, 4, 3))
            self.assertEqual(calls[-1][1], torch.Size([10, 2, 8]))
            self.assertIsNotNone(calls[-1][3]["cu_seqlens_k"])
            self.assertNotIn("block_table", calls[-1][3])

            q = torch.randn(2, 1, 4, 8)
            paged_k = torch.randn(2, 16, 2, 8)
            paged_v = torch.randn_like(paged_k)
            output, lse = flash_attention_seqlen_compat(
                q,
                paged_k,
                paged_v,
                torch.tensor([3, 5], dtype=torch.int32),
            )
            self.assertEqual(tuple(output.shape), tuple(q.shape))
            self.assertEqual(tuple(lse.shape), (2, 1, 4))
            self.assertEqual(calls[-1][1], torch.Size([2, 16, 2, 8]))
            self.assertIsNone(calls[-1][3]["cu_seqlens_k"])
            self.assertEqual(tuple(calls[-1][3]["block_table"].shape), (2, 1))


if __name__ == "__main__":
    unittest.main()
