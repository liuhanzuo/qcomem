from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from qcomem_paged_attention import (
    PagedKVLayer,
    PagedTensorView,
    paged_attention_forward,
    replace_dynamic_cache_layer,
)
from qcomem_torch import cache_nbytes


def repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    batch, kv_heads, tokens, dim = states.shape
    return (
        states[:, :, None, :, :]
        .expand(batch, kv_heads, groups, tokens, dim)
        .reshape(batch, kv_heads * groups, tokens, dim)
    )


def dense_eager_attention(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    scaling: float,
) -> torch.Tensor:
    """The dense Qwen eager contract used by the same-caller gate."""

    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


def causal_additive_mask(
    *,
    query_length: int,
    total_length: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    past_length = total_length - query_length
    query_positions = torch.arange(query_length) + past_length
    key_positions = torch.arange(total_length)
    allowed = key_positions.view(1, -1) <= query_positions.view(-1, 1)
    mask = torch.zeros((1, 1, query_length, total_length), dtype=dtype)
    mask.masked_fill_(~allowed.view(1, 1, query_length, total_length), -torch.inf)
    # Exercise general additive-mask slicing as well as causality.
    mask[..., 1] -= 0.375
    return mask


class DenseDynamicLayer:
    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.keys = key.clone()
        self.values = value.clone()

    def update(self, key: torch.Tensor, value: torch.Tensor):
        self.keys = torch.cat([self.keys, key], dim=-2)
        self.values = torch.cat([self.values, value], dim=-2)
        return self.keys, self.values


class PagedAttentionReferenceTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260813)
        self.batch = 1
        self.kv_heads = 2
        self.groups = 2
        self.query_heads = self.kv_heads * self.groups
        self.head_dim = 8
        self.document_length = 7
        self.query_length = 3
        self.scaling = self.head_dim**-0.5
        self.module = SimpleNamespace(
            num_key_value_groups=self.groups,
            scaling=self.scaling,
            is_causal=True,
        )
        self.document_key = torch.randn(
            self.batch, self.kv_heads, self.document_length, self.head_dim
        )
        self.document_value = torch.randn_like(self.document_key)
        self.query = torch.randn(
            self.batch, self.query_heads, self.query_length, self.head_dim
        )
        self.new_key = torch.randn(
            self.batch, self.kv_heads, self.query_length, self.head_dim
        )
        self.new_value = torch.randn_like(self.new_key)
        self.mask = causal_additive_mask(
            query_length=self.query_length,
            total_length=self.document_length + self.query_length,
            dtype=self.query.dtype,
        )

    def test_cache_nbytes_counts_pages_once_and_deduplicates_shared_forks(self) -> None:
        key = torch.randn(1, 2, 7, 4)
        value = torch.randn(1, 2, 7, 4)
        layer = PagedKVLayer.from_dense_document(
            key, value, page_size=3, bits=4, group_size=8
        )
        fork = layer.fork()
        self.assertEqual(cache_nbytes(layer), layer.stored_nbytes)
        self.assertEqual(cache_nbytes((layer, fork)), layer.stored_nbytes)

    def test_same_qkv_mask_caller_dense_eager_matches_paged_online(self) -> None:
        dense_layer = DenseDynamicLayer(self.document_key, self.document_value)
        paged_layer = PagedKVLayer.from_dense_document(
            self.document_key,
            self.document_value,
            page_size=3,
            append_page_size=2,
        )

        # This is the Qwen3.5 caller shape: the same projected q/new-k/new-v and
        # the same prepared mask; only cache.update and attention backend differ.
        dense_key, dense_value = dense_layer.update(self.new_key, self.new_value)
        dense_output = dense_eager_attention(
            self.module,
            self.query,
            dense_key,
            dense_value,
            self.mask,
            scaling=self.scaling,
        )

        audit: dict[str, object] = {}
        with mock.patch(
            "qcomem_paged_attention.torch.cat",
            side_effect=AssertionError("paged execution must not call torch.cat"),
        ):
            paged_key, paged_value = paged_layer.update(
                self.new_key, self.new_value
            )
            paged_output, weights = paged_attention_forward(
                self.module,
                self.query,
                paged_key,
                paged_value,
                self.mask,
                dropout=0.0,
                scaling=self.scaling,
                audit=audit,
            )

        self.assertIsNone(weights)
        torch.testing.assert_close(paged_output, dense_output, rtol=2e-6, atol=2e-6)
        self.assertEqual(audit["full_kv_concatenations"], 0)
        self.assertEqual(audit["gqa_kv_repeat_materializations"], 0)
        self.assertLess(
            audit["max_materialized_kv_tokens"], audit["total_kv_tokens"]
        )
        self.assertLess(
            audit["max_materialized_kv_nbytes"], audit["dense_full_kv_nbytes"]
        )

    def test_bfloat16_uses_eager_weight_dtype_and_fp32_partial_accumulator(self) -> None:
        # A float32-only toy misses the production failure mode: Qwen eager
        # casts FP32 softmax probabilities back to BF16 before weight@value.
        # This larger shape makes that rounding contract observable.
        torch.manual_seed(1949)
        batch, kv_heads, groups = 1, 2, 4
        query_heads, query_length, total_length, head_dim = 8, 32, 288, 128
        query = torch.randn(
            batch, query_heads, query_length, head_dim, dtype=torch.bfloat16
        )
        key = torch.randn(
            batch, kv_heads, total_length, head_dim, dtype=torch.bfloat16
        )
        value = torch.randn_like(key)
        module = SimpleNamespace(
            num_key_value_groups=groups,
            scaling=head_dim**-0.5,
            is_causal=True,
        )
        mask = causal_additive_mask(
            query_length=query_length,
            total_length=total_length,
            dtype=torch.bfloat16,
        )
        oracle = dense_eager_attention(
            module, query, key, value, mask, scaling=module.scaling
        )
        layer = PagedKVLayer.from_dense_document(
            key[..., :256, :],
            value[..., :256, :],
            page_size=128,
            bits=16,
            group_size=64,
            append_page_size=16,
        )
        paged_key, paged_value = layer.update(
            key[..., 256:, :], value[..., 256:, :]
        )
        audit: dict[str, object] = {}
        candidate, _ = paged_attention_forward(
            module,
            query,
            paged_key,
            paged_value,
            mask,
            scaling=module.scaling,
            audit=audit,
        )
        torch.testing.assert_close(candidate, oracle, rtol=2e-3, atol=2e-3)
        self.assertEqual(audit["softmax_passes"], 2)
        self.assertEqual(audit["normalized_weight_dtype"], "torch.bfloat16")
        self.assertEqual(audit["value_accumulator_dtype"], "torch.float32")

    def test_q4_pages_match_their_dense_dequantized_oracle_and_bound_peak(self) -> None:
        layer = PagedKVLayer.from_dense_document(
            self.document_key,
            self.document_value,
            page_size=3,
            bits=4,
            group_size=16,
            append_page_size=2,
        )
        dense_document_bytes = (
            self.document_key.numel() * self.document_key.element_size()
            + self.document_value.numel() * self.document_value.element_size()
        )
        packed_document_bytes = layer.stored_nbytes
        document_storage_before = layer.store.storage_keys
        self.assertLess(packed_document_bytes, dense_document_bytes)

        paged_key, paged_value = layer.update(self.new_key, self.new_value)
        # Materialise only for an out-of-path correctness oracle.  The guarded
        # execution below must not use this concatenation route.
        dense_pages = [
            page.materialize(device=self.query.device, dtype=self.query.dtype)
            for page in layer.store.pages
        ]
        dense_key = torch.cat([pair[0] for pair in dense_pages], dim=-2)
        dense_value = torch.cat([pair[1] for pair in dense_pages], dim=-2)
        oracle = dense_eager_attention(
            self.module,
            self.query,
            dense_key,
            dense_value,
            self.mask,
            scaling=self.scaling,
        )

        audit: dict[str, object] = {}
        with mock.patch(
            "qcomem_paged_attention.torch.cat",
            side_effect=AssertionError("packed-page execution must not concatenate"),
        ):
            candidate, _ = paged_attention_forward(
                self.module,
                self.query,
                paged_key,
                paged_value,
                self.mask,
                scaling=self.scaling,
                audit=audit,
            )

        torch.testing.assert_close(candidate, oracle, rtol=2e-6, atol=2e-6)
        self.assertTrue(document_storage_before.issubset(layer.store.storage_keys))
        self.assertEqual(audit["max_materialized_kv_tokens"], 3)
        expected_max_page_bytes = (
            self.batch
            * self.kv_heads
            * 3
            * self.head_dim
            * self.query.element_size()
            * 2
        )
        self.assertEqual(audit["max_materialized_kv_nbytes"], expected_max_page_bytes)
        self.assertEqual(
            audit["max_single_unpack_page_nbytes"], expected_max_page_bytes
        )
        self.assertLess(
            audit["max_materialized_kv_nbytes"], audit["dense_full_kv_nbytes"]
        )

    def test_fork_shares_document_storage_and_keeps_request_pages_private(self) -> None:
        source = PagedKVLayer.from_dense_document(
            self.document_key,
            self.document_value,
            page_size=2,
            bits=4,
            group_size=16,
        )
        source_keys = source.store.storage_keys
        source_length = source.get_seq_length()
        fork = source.fork()

        self.assertEqual(fork.store.storage_keys, source_keys)
        self.assertTrue(
            all(
                left is right
                for left, right in zip(
                    source.store.document_pages, fork.store.document_pages
                )
            )
        )
        fork.update(self.new_key, self.new_value)
        self.assertEqual(source.get_seq_length(), source_length)
        self.assertEqual(fork.get_seq_length(), source_length + self.query_length)
        self.assertTrue(source_keys.issubset(fork.store.storage_keys))
        self.assertGreater(fork.stored_nbytes, source.stored_nbytes)

    def test_non_tensor_transformers_views_fail_closed(self) -> None:
        left = PagedKVLayer.from_dense_document(
            self.document_key, self.document_value, page_size=3
        )
        right = PagedKVLayer.from_dense_document(
            self.document_key, self.document_value, page_size=3
        )
        key_view = PagedTensorView(left.store, "key")
        wrong_store_value = PagedTensorView(right.store, "value")
        with self.assertRaisesRegex(ValueError, "same store"):
            paged_attention_forward(
                self.module,
                self.query,
                key_view,
                wrong_store_value,
                self.mask,
                scaling=self.scaling,
            )
        with self.assertRaisesRegex(TypeError, "both be tensors or paired"):
            paged_attention_forward(
                self.module,
                self.query,
                key_view,
                self.document_value,
                self.mask,
                scaling=self.scaling,
            )

    def test_dense_dynamic_cache_layer_conversion_is_explicit_and_non_sliding(self) -> None:
        dense_storage_keys = {
            self.document_key.untyped_storage().data_ptr(),
            self.document_value.untyped_storage().data_ptr(),
        }
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    keys=self.document_key,
                    values=self.document_value,
                    is_sliding=False,
                )
            ]
        )
        converted = replace_dynamic_cache_layer(
            cache,
            0,
            page_size=3,
            bits=4,
            group_size=16,
        )
        self.assertIs(cache.layers[0], converted)
        self.assertEqual(converted.get_seq_length(), self.document_length)
        self.assertEqual(
            converted.get_mask_sizes(self.query_length),
            (self.document_length + self.query_length, 0),
        )
        converted_ptrs = {key[1] for key in converted.store.storage_keys}
        self.assertTrue(dense_storage_keys.isdisjoint(converted_ptrs))

        sliding_cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    keys=self.document_key,
                    values=self.document_value,
                    is_sliding=True,
                )
            ]
        )
        with self.assertRaisesRegex(ValueError, "sliding attention"):
            replace_dynamic_cache_layer(sliding_cache, 0, page_size=3)

    def test_dropout_and_incompatible_gqa_fail_closed(self) -> None:
        layer = PagedKVLayer.from_dense_document(
            self.document_key, self.document_value, page_size=3
        )
        key, value = layer.update(self.new_key, self.new_value)
        with self.assertRaisesRegex(ValueError, "dropout=0"):
            paged_attention_forward(
                self.module,
                self.query,
                key,
                value,
                self.mask,
                dropout=0.1,
            )
        bad_module = SimpleNamespace(
            num_key_value_groups=1, scaling=math.sqrt(self.head_dim), is_causal=True
        )
        with self.assertRaisesRegex(ValueError, "num_key_value_groups"):
            paged_attention_forward(
                bad_module,
                self.query,
                key,
                value,
                self.mask,
            )


if __name__ == "__main__":
    unittest.main()
