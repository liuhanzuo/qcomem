from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from qcomem_vllm_paged_kernel import (
    Q16KernelPagedLayer,
    Q16PagedArena,
    QComemPagedKernelError,
    validate_canonical_tail_causal_mask,
    vllm_triton_q16_paged_attention_forward,
)


def causal_mask(query_length: int, total_length: int) -> torch.Tensor:
    past = total_length - query_length
    qpos = torch.arange(query_length) + past
    kpos = torch.arange(total_length)
    allowed = kpos.view(1, -1) <= qpos.view(-1, 1)
    mask = torch.zeros(1, 1, query_length, total_length)
    return mask.masked_fill(~allowed.view(1, 1, query_length, total_length), -torch.inf)


class DenseOracleKernel:
    """CPU stand-in with the exact keyword contract of unified_attention."""

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        q = kwargs["q"]
        k_cache = kwargs["k"]
        v_cache = kwargs["v"]
        table = kwargs["block_table"]
        seq_lens = kwargs["seqused_k"]
        cu_q = kwargs["cu_seqlens_q"]
        scale = kwargs["softmax_scale"]
        out = kwargs["out"]
        for batch_index in range(table.shape[0]):
            q_start = int(cu_q[batch_index])
            q_end = int(cu_q[batch_index + 1])
            query = q[q_start:q_end].transpose(0, 1)
            length = int(seq_lens[batch_index])
            blocks = table[batch_index]
            key = k_cache[blocks].reshape(-1, k_cache.shape[-2], k_cache.shape[-1])[:length]
            value = v_cache[blocks].reshape(-1, v_cache.shape[-2], v_cache.shape[-1])[:length]
            key = key.permute(1, 0, 2)
            value = value.permute(1, 0, 2)
            groups = query.shape[0] // key.shape[0]
            key = key[:, None].expand(-1, groups, -1, -1).reshape(
                query.shape[0], length, query.shape[-1]
            )
            value = value[:, None].expand(-1, groups, -1, -1).reshape(
                query.shape[0], length, query.shape[-1]
            )
            scores = torch.matmul(query, key.transpose(1, 2)) * scale
            qlen = q_end - q_start
            past = length - qlen
            qpos = torch.arange(qlen) + past
            kpos = torch.arange(length)
            scores.masked_fill_(kpos.view(1, 1, -1) > qpos.view(1, -1, 1), -torch.inf)
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out[q_start:q_end].copy_(torch.matmul(weights, value).transpose(0, 1))


class MetadataOnlyKernel:
    def __init__(self) -> None:
        self.call = None

    def __call__(self, **kwargs):
        self.call = kwargs
        kwargs["out"].zero_()


class VllmPagedKernelAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260813)
        self.batch = 1
        self.kv_heads = 2
        self.query_heads = 16
        self.head_dim = 32
        self.document_length = 35
        self.query_length = 7
        self.page_size = 16
        self.document_key = torch.randn(
            self.batch, self.kv_heads, self.document_length, self.head_dim
        )
        self.document_value = torch.randn_like(self.document_key)
        self.new_key = torch.randn(
            self.batch, self.kv_heads, self.query_length, self.head_dim
        )
        self.new_value = torch.randn_like(self.new_key)
        self.query = torch.randn(
            self.batch, self.query_heads, self.query_length, self.head_dim
        )
        self.module = SimpleNamespace(
            is_causal=True,
            num_key_value_groups=self.query_heads // self.kv_heads,
            scaling=self.head_dim**-0.5,
        )

    def test_partial_tail_is_request_private_and_document_pool_stays_immutable(self):
        arena = Q16PagedArena.from_dense_document(
            self.document_key,
            self.document_value,
            page_size=self.page_size,
            max_append_tokens=12,
            max_forks=2,
        )
        last_document_block = int(arena.document_block_table[0, -1])
        key_tail_before = arena.key_cache[last_document_block, :3].clone()
        first = arena.fork()
        second = arena.fork()
        first.append(self.new_key, self.new_value)
        second.append(self.new_key + 1, self.new_value + 1)
        self.assertNotEqual(
            int(first.active_block_table[0, -1]), last_document_block
        )
        self.assertNotEqual(
            int(first.active_block_table[0, -1]),
            int(second.active_block_table[0, -1]),
        )
        torch.testing.assert_close(
            arena.key_cache[last_document_block, :3], key_tail_before
        )
        expected_tail_bytes = (
            2 * 3 * self.kv_heads * self.head_dim * self.document_key.element_size()
        )
        self.assertEqual(first.partial_tail_staging_copy_nbytes, expected_tail_bytes)
        self.assertLess(
            first.partial_tail_staging_copy_nbytes,
            self.document_key.numel() * self.document_key.element_size() * 2,
        )

    def test_append_crosses_blocks_without_torch_cat(self):
        layer = Q16KernelPagedLayer.from_dense_document(
            self.document_key,
            self.document_value,
            page_size=self.page_size,
            max_append_tokens=40,
        )
        key = torch.randn(self.batch, self.kv_heads, 30, self.head_dim)
        value = torch.randn_like(key)
        with mock.patch(
            "qcomem_vllm_paged_kernel.torch.cat",
            side_effect=AssertionError("full KV cat is forbidden"),
            create=True,
        ):
            key_view, value_view = layer.update(key, value)
        self.assertEqual(key_view.shape[-2], self.document_length + 30)
        self.assertEqual(value_view.shape, key_view.shape)
        self.assertEqual(layer.sequence.logical_block_count, math.ceil(65 / 16))

    def test_fused_adapter_matches_dense_oracle_for_gqa_and_uses_block_table(self):
        layer = Q16KernelPagedLayer.from_dense_document(
            self.document_key,
            self.document_value,
            page_size=self.page_size,
            max_append_tokens=16,
        )
        key_view, value_view = layer.update(self.new_key, self.new_value)
        mask = causal_mask(self.query_length, self.document_length + self.query_length)
        kernel = DenseOracleKernel()
        audit = {}
        with mock.patch(
            "qcomem_vllm_paged_kernel.torch.cat",
            side_effect=AssertionError("full KV cat is forbidden"),
            create=True,
        ):
            output, weights = vllm_triton_q16_paged_attention_forward(
                self.module,
                self.query,
                key_view,
                value_view,
                mask,
                scaling=self.module.scaling,
                audit=audit,
                _kernel=kernel,
            )
        self.assertIsNone(weights)
        self.assertEqual(output.shape, (1, self.query_length, self.query_heads, self.head_dim))
        self.assertEqual(len(kernel.calls), 1)
        call = kernel.calls[0]
        self.assertEqual(tuple(call["k"].shape[1:]), (16, 2, self.head_dim))
        self.assertEqual(tuple(call["block_table"].shape), (1, 3))
        self.assertEqual(int(call["seqused_k"][0]), 42)
        self.assertTrue(call["causal"])
        self.assertEqual(audit["full_kv_concatenations"], 0)
        self.assertEqual(audit["fused_gpu_kernel_calls"], 1)
        self.assertEqual(audit["gqa_groups"], 8)
        self.assertEqual(audit["softmax_scale"], self.module.scaling)

        dense_key = torch.empty(1, 2, 42, self.head_dim)
        dense_value = torch.empty_like(dense_key)
        dense_key[..., :35, :] = self.document_key
        dense_value[..., :35, :] = self.document_value
        dense_key[..., 35:, :] = self.new_key
        dense_value[..., 35:, :] = self.new_value
        repeated_key = dense_key[:, :, None].expand(-1, -1, 8, -1, -1).reshape(
            1, 16, 42, self.head_dim
        )
        repeated_value = dense_value[:, :, None].expand(-1, -1, 8, -1, -1).reshape(
            1, 16, 42, self.head_dim
        )
        scores = torch.matmul(self.query, repeated_key.transpose(2, 3)) * self.module.scaling
        scores = scores + mask
        probability = torch.softmax(scores, dim=-1, dtype=torch.float32).to(self.query.dtype)
        expected = torch.matmul(probability, repeated_value).transpose(1, 2).contiguous()
        torch.testing.assert_close(output, expected, rtol=2e-6, atol=2e-6)

    def test_noncanonical_additive_mask_fails_closed(self):
        mask = causal_mask(self.query_length, self.document_length + self.query_length)
        mask[..., 0] -= 0.25
        with self.assertRaisesRegex(QComemPagedKernelError, "non-canonical"):
            validate_canonical_tail_causal_mask(
                mask,
                batch_size=1,
                query_length=self.query_length,
                total_length=self.document_length + self.query_length,
                device=torch.device("cpu"),
            )

    def test_exact_qwen35_4k_plus_32_geometry_reaches_one_kernel_call(self):
        document_length = 4096
        query_length = 32
        head_dim = 256
        document_key = torch.zeros(1, 2, document_length, head_dim)
        document_value = torch.zeros_like(document_key)
        layer = Q16KernelPagedLayer.from_dense_document(
            document_key,
            document_value,
            page_size=128,
            max_append_tokens=40,
        )
        new_key = torch.zeros(1, 2, query_length, head_dim)
        new_value = torch.zeros_like(new_key)
        key_view, value_view = layer.update(new_key, new_value)
        query = torch.zeros(1, 16, query_length, head_dim)
        module = SimpleNamespace(
            is_causal=True,
            num_key_value_groups=8,
            scaling=head_dim**-0.5,
        )
        kernel = MetadataOnlyKernel()
        output, _ = vllm_triton_q16_paged_attention_forward(
            module,
            query,
            key_view,
            value_view,
            causal_mask(query_length, document_length + query_length),
            _kernel=kernel,
        )
        self.assertEqual(output.shape, (1, 32, 16, 256))
        self.assertIsNotNone(kernel.call)
        self.assertEqual(kernel.call["max_seqlen_q"], 32)
        self.assertEqual(kernel.call["max_seqlen_k"], 4128)
        self.assertEqual(tuple(kernel.call["block_table"].shape), (1, 33))
        self.assertEqual(tuple(kernel.call["k"].shape[1:]), (128, 2, 256))

    def test_capacity_and_fork_limits_fail_before_overwrite(self):
        arena = Q16PagedArena.from_dense_document(
            self.document_key,
            self.document_value,
            page_size=self.page_size,
            max_append_tokens=4,
            max_forks=1,
        )
        request = arena.fork()
        with self.assertRaisesRegex(QComemPagedKernelError, "exhausted"):
            arena.fork()
        with self.assertRaisesRegex(QComemPagedKernelError, "max_append_tokens"):
            request.append(self.new_key, self.new_value)

    def test_transformers_use_cache_kwarg_is_whitelisted_but_must_be_true(self):
        layer = Q16KernelPagedLayer.from_dense_document(
            self.document_key,
            self.document_value,
            page_size=self.page_size,
            max_append_tokens=self.query_length,
        )
        key, value = layer.update(self.new_key, self.new_value)
        kernel = MetadataOnlyKernel()
        output, _ = vllm_triton_q16_paged_attention_forward(
            self.module,
            self.query,
            key,
            value,
            None,
            use_cache=True,
            _kernel=kernel,
        )
        self.assertEqual(output.shape, (1, 7, 16, 32))
        with self.assertRaisesRegex(QComemPagedKernelError, "use_cache=True"):
            vllm_triton_q16_paged_attention_forward(
                self.module,
                self.query,
                key,
                value,
                None,
                use_cache=False,
                _kernel=kernel,
            )


if __name__ == "__main__":
    unittest.main()
