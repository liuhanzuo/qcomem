from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest import mock

import torch

from qcomem_vllm_paged_kernel import Q16PagedArena
from qcomem_vllm_ragged_batch import (
    FROZEN_REQUIRED_PARAMETER_COUNT,
    FROZEN_UNIFIED_ATTENTION_PARAMETERS,
    Q16RaggedRequest,
    QComemRaggedBatchError,
    prepare_q16_ragged_batch,
    q16_ragged_paged_attention,
)


QUERY_HEADS = 16
KV_HEADS = 2
GROUPS = 8
HEAD_DIM = 256
PAGE_SIZE = 128


def canonical_mask(
    query_length: int,
    kv_length: int,
    *,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    query_positions = torch.arange(query_length) + kv_length - query_length
    key_positions = torch.arange(kv_length)
    allowed = key_positions.view(1, -1) <= query_positions.view(-1, 1)
    if dtype is None:
        return allowed.view(1, 1, query_length, kv_length)
    mask = torch.zeros(1, 1, query_length, kv_length, dtype=dtype)
    return mask.masked_fill(~allowed.view(1, 1, query_length, kv_length), -torch.inf)


class DenseRaggedOracle:
    """CPU oracle with the 16 required vLLM 0.26 keyword parameters."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.logical_keys: list[torch.Tensor] = []
        self.logical_values: list[torch.Tensor] = []

    def __call__(self, **kwargs: object) -> None:
        self.calls.append(dict(kwargs))
        q = kwargs["q"]
        key_cache = kwargs["k"]
        value_cache = kwargs["v"]
        out = kwargs["out"]
        cu_q = kwargs["cu_seqlens_q"]
        seq_lens = kwargs["seqused_k"]
        block_table = kwargs["block_table"]
        scale = float(kwargs["softmax_scale"])
        assert isinstance(q, torch.Tensor)
        assert isinstance(key_cache, torch.Tensor)
        assert isinstance(value_cache, torch.Tensor)
        assert isinstance(out, torch.Tensor)
        assert isinstance(cu_q, torch.Tensor)
        assert isinstance(seq_lens, torch.Tensor)
        assert isinstance(block_table, torch.Tensor)
        for sequence_index in range(int(seq_lens.numel())):
            start = int(cu_q[sequence_index])
            stop = int(cu_q[sequence_index + 1])
            query_length = stop - start
            kv_length = int(seq_lens[sequence_index])
            logical_blocks = math.ceil(kv_length / int(key_cache.shape[1]))
            physical = block_table[sequence_index, :logical_blocks].long()
            key = key_cache[physical].reshape(
                -1, int(key_cache.shape[2]), int(key_cache.shape[3])
            )[:kv_length]
            value = value_cache[physical].reshape(
                -1, int(value_cache.shape[2]), int(value_cache.shape[3])
            )[:kv_length]
            key = key.permute(1, 0, 2).contiguous()
            value = value.permute(1, 0, 2).contiguous()
            self.logical_keys.append(key.clone())
            self.logical_values.append(value.clone())

            query = q[start:stop].transpose(0, 1).float()
            repeated_key = key[:, None].expand(-1, GROUPS, -1, -1).reshape(
                QUERY_HEADS, kv_length, HEAD_DIM
            )
            repeated_value = value[:, None].expand(-1, GROUPS, -1, -1).reshape(
                QUERY_HEADS, kv_length, HEAD_DIM
            )
            scores = torch.matmul(query, repeated_key.float().transpose(1, 2)) * scale
            query_positions = torch.arange(query_length) + kv_length - query_length
            key_positions = torch.arange(kv_length)
            scores.masked_fill_(
                key_positions.view(1, 1, -1)
                > query_positions.view(1, -1, 1),
                -torch.inf,
            )
            probability = torch.softmax(scores, dim=-1)
            result = torch.matmul(probability, repeated_value.float()).transpose(0, 1)
            out[start:stop].copy_(result.to(out.dtype))


class MetadataOnlyKernel:
    def __init__(self) -> None:
        self.call: dict[str, object] | None = None

    def __call__(self, **kwargs: object) -> None:
        self.call = dict(kwargs)
        out = kwargs["out"]
        assert isinstance(out, torch.Tensor)
        out.zero_()


class Q16RaggedBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260814)

    def build_fixture(
        self,
        query_lengths: tuple[int, ...] = (1, 3, 5),
    ) -> tuple[
        Q16PagedArena,
        tuple[Q16RaggedRequest, ...],
        torch.Tensor,
        torch.Tensor,
        tuple[torch.Tensor, ...],
        tuple[torch.Tensor, ...],
    ]:
        document_length = 129
        document_key = (
            torch.randn(1, KV_HEADS, document_length, HEAD_DIM) * 0.1
        ).to(torch.float16)
        document_value = torch.randn_like(document_key) * 0.1
        arena = Q16PagedArena.from_dense_document(
            document_key,
            document_value,
            page_size=PAGE_SIZE,
            max_append_tokens=max(query_lengths),
            max_forks=len(query_lengths),
        )
        requests: list[Q16RaggedRequest] = []
        appended_keys: list[torch.Tensor] = []
        appended_values: list[torch.Tensor] = []
        for index, query_length in enumerate(query_lengths):
            sequence = arena.fork()
            key = (
                torch.randn(1, KV_HEADS, query_length, HEAD_DIM) * 0.1
                + index * 0.01
            ).to(torch.float16)
            value = (
                torch.randn(1, KV_HEADS, query_length, HEAD_DIM) * 0.1
                - index * 0.01
            ).to(torch.float16)
            sequence.append(key, value)
            query = (
                torch.randn(query_length, QUERY_HEADS, HEAD_DIM) * 0.1
            ).to(torch.float16)
            position_ids = torch.arange(
                document_length,
                document_length + query_length,
                dtype=torch.long,
            )
            if index == 0:
                mask = None
            elif index == 1:
                mask = canonical_mask(
                    query_length,
                    document_length + query_length,
                    dtype=None,
                )
            else:
                mask = canonical_mask(
                    query_length,
                    document_length + query_length,
                    dtype=torch.float16,
                )
            requests.append(
                Q16RaggedRequest(
                    sequence=sequence,
                    query=query,
                    position_ids=position_ids,
                    attention_mask=mask,
                )
            )
            appended_keys.append(key)
            appended_values.append(value)
        return (
            arena,
            tuple(requests),
            document_key,
            document_value,
            tuple(appended_keys),
            tuple(appended_values),
        )

    def expected_logical(
        self,
        document: torch.Tensor,
        appended: torch.Tensor,
    ) -> torch.Tensor:
        document_length = int(document.shape[-2])
        append_length = int(appended.shape[-2])
        dense = torch.empty(
            KV_HEADS,
            document_length + append_length,
            HEAD_DIM,
            dtype=document.dtype,
        )
        dense[:, :document_length].copy_(document[0])
        dense[:, document_length:].copy_(appended[0])
        return dense

    def test_ragged_metadata_logical_kv_gqa_and_output_match_dense_oracle(self) -> None:
        arena, requests, document_key, document_value, new_keys, new_values = (
            self.build_fixture()
        )
        oracle = DenseRaggedOracle()
        with mock.patch(
            "qcomem_vllm_ragged_batch.torch.cat",
            side_effect=AssertionError("full K/V cat is forbidden"),
            create=True,
        ):
            result = q16_ragged_paged_attention(requests, _kernel=oracle)

        self.assertEqual(len(oracle.calls), 1)
        call = oracle.calls[0]
        self.assertEqual(
            set(call), set(FROZEN_UNIFIED_ATTENTION_PARAMETERS[:16])
        )
        self.assertIs(call["k"], arena.key_cache)
        self.assertIs(call["v"], arena.value_cache)
        self.assertEqual(call["max_seqlen_q"], 5)
        self.assertEqual(call["max_seqlen_k"], 134)
        self.assertEqual(call["causal"], True)
        self.assertEqual(call["window_size"], (-1, -1))
        self.assertEqual(call["softcap"], 0.0)
        torch.testing.assert_close(
            call["cu_seqlens_q"], torch.tensor([0, 1, 4, 9], dtype=torch.int32)
        )
        torch.testing.assert_close(
            call["seqused_k"], torch.tensor([130, 132, 134], dtype=torch.int32)
        )
        self.assertEqual(tuple(call["block_table"].shape), (3, 2))

        for index, request in enumerate(requests):
            expected_key = self.expected_logical(document_key, new_keys[index])
            expected_value = self.expected_logical(document_value, new_values[index])
            torch.testing.assert_close(oracle.logical_keys[index], expected_key)
            torch.testing.assert_close(oracle.logical_values[index], expected_value)

            query = request.query.transpose(0, 1).float()
            kv_length = int(expected_key.shape[1])
            query_length = int(request.query.shape[0])
            repeated_key = expected_key[:, None].expand(
                -1, GROUPS, -1, -1
            ).reshape(QUERY_HEADS, kv_length, HEAD_DIM)
            repeated_value = expected_value[:, None].expand(
                -1, GROUPS, -1, -1
            ).reshape(QUERY_HEADS, kv_length, HEAD_DIM)
            scores = torch.matmul(query, repeated_key.float().transpose(1, 2)) * (
                HEAD_DIM**-0.5
            )
            query_positions = torch.arange(query_length) + kv_length - query_length
            key_positions = torch.arange(kv_length)
            scores.masked_fill_(
                key_positions.view(1, 1, -1)
                > query_positions.view(1, -1, 1),
                -torch.inf,
            )
            probability = torch.softmax(scores, dim=-1)
            expected_output = torch.matmul(
                probability, repeated_value.float()
            ).transpose(0, 1)
            torch.testing.assert_close(
                result.sequence_outputs[index].float(),
                expected_output,
                rtol=4e-3,
                atol=4e-3,
            )

        self.assertEqual(result.audit["query_lengths"], (1, 3, 5))
        self.assertEqual(result.audit["kv_lengths"], (130, 132, 134))
        self.assertEqual(result.audit["cu_seqlens_q"], (0, 1, 4, 9))
        self.assertTrue(result.audit["query_is_ragged"])
        self.assertTrue(result.audit["kv_is_ragged"])
        self.assertEqual(result.audit["gqa_groups"], 8)
        self.assertEqual(result.audit["full_kv_concatenations"], 0)
        self.assertEqual(result.audit["full_kv_materializations"], 0)
        self.assertFalse(result.audit["scheduler_integration_claimed"])
        self.assertFalse(result.audit["throughput_claimed"])
        self.assertTrue(result.audit["h20_kernel_gate_required"])
        self.assertFalse(result.audit["h20_kernel_gate_passed"])

    def test_flattened_queries_preserve_ragged_order_and_cardinality(self) -> None:
        _, requests, *_ = self.build_fixture((4, 1, 2))
        batch = prepare_q16_ragged_batch(requests)
        self.assertEqual(batch.query_lengths, (4, 1, 2))
        self.assertEqual(batch.kv_lengths, (133, 130, 131))
        torch.testing.assert_close(
            batch.cu_seqlens_q, torch.tensor([0, 4, 5, 7], dtype=torch.int32)
        )
        for index, request in enumerate(requests):
            start = int(batch.cu_seqlens_q[index])
            stop = int(batch.cu_seqlens_q[index + 1])
            torch.testing.assert_close(batch.q[start:stop], request.query)

    def test_non_tail_positions_and_noncanonical_masks_fail_closed(self) -> None:
        _, requests, *_ = self.build_fixture()
        bad_positions = requests[1].position_ids.clone()
        bad_positions[-1] += 1
        with self.assertRaisesRegex(QComemRaggedBatchError, "position_ids"):
            prepare_q16_ragged_batch(
                (requests[0], replace(requests[1], position_ids=bad_positions))
            )

        bad_mask = canonical_mask(3, 132, dtype=torch.float16)
        bad_mask[..., 0, 0] = -0.25
        with self.assertRaisesRegex(QComemRaggedBatchError, "not canonical"):
            prepare_q16_ragged_batch(
                (requests[0], replace(requests[1], attention_mask=bad_mask))
            )

        with self.assertRaisesRegex(QComemRaggedBatchError, "only full_attention"):
            prepare_q16_ragged_batch(
                (
                    replace(
                        requests[0],
                        attention_mask={
                            "full_attention": canonical_mask(1, 130, dtype=None),
                            "sliding_attention": canonical_mask(1, 130, dtype=None),
                        },
                    ),
                )
            )

    def test_geometry_dtype_and_device_fail_closed(self) -> None:
        _, requests, *_ = self.build_fixture()
        wrong_heads = torch.zeros(1, 8, HEAD_DIM, dtype=torch.float16)
        with self.assertRaisesRegex(QComemRaggedBatchError, "16Q"):
            prepare_q16_ragged_batch((replace(requests[0], query=wrong_heads),))

        wrong_dtype = requests[0].query.float()
        with self.assertRaisesRegex(QComemRaggedBatchError, "dtype/device"):
            prepare_q16_ragged_batch((replace(requests[0], query=wrong_dtype),))

        wrong_device_positions = torch.empty(
            requests[0].position_ids.shape,
            dtype=torch.long,
            device="meta",
        )
        with self.assertRaisesRegex(QComemRaggedBatchError, "dtype/device"):
            prepare_q16_ragged_batch(
                (replace(requests[0], position_ids=wrong_device_positions),)
            )

        document = torch.zeros(1, KV_HEADS, 17, HEAD_DIM, dtype=torch.float16)
        wrong_page_arena = Q16PagedArena.from_dense_document(
            document,
            document,
            page_size=16,
            max_append_tokens=1,
            max_forks=1,
        )
        wrong_page_sequence = wrong_page_arena.fork()
        new = torch.zeros(1, KV_HEADS, 1, HEAD_DIM, dtype=torch.float16)
        wrong_page_sequence.append(new, new)
        wrong_page_request = Q16RaggedRequest(
            wrong_page_sequence,
            torch.zeros(1, QUERY_HEADS, HEAD_DIM, dtype=torch.float16),
            torch.tensor([17], dtype=torch.long),
        )
        with self.assertRaisesRegex(QComemRaggedBatchError, "page_size=128"):
            prepare_q16_ragged_batch((wrong_page_request,))

    def test_shared_arena_unique_sequence_and_private_ownership_fail_closed(self) -> None:
        _, requests, *_ = self.build_fixture()
        with self.assertRaisesRegex(QComemRaggedBatchError, "cannot appear twice"):
            prepare_q16_ragged_batch((requests[0], requests[0]))

        _, other_requests, *_ = self.build_fixture((1,))
        with self.assertRaisesRegex(QComemRaggedBatchError, "share exactly one"):
            prepare_q16_ragged_batch((requests[0], other_requests[0]))

        _, alias_requests, *_ = self.build_fixture((1, 3))
        first_private = alias_requests[0].sequence.active_block_table[0, -1].clone()
        active_tail_index = int(
            alias_requests[1].sequence.active_block_table.shape[1]
        ) - 1
        alias_requests[1].sequence.block_table[0, active_tail_index].copy_(
            first_private
        )
        with self.assertRaisesRegex(QComemRaggedBatchError, "request-private"):
            prepare_q16_ragged_batch(alias_requests)

    def test_invalid_block_table_and_unappended_query_fail_closed(self) -> None:
        arena, requests, *_ = self.build_fixture((1,))
        requests[0].sequence.block_table[0, 0] = int(arena.key_cache.shape[0])
        with self.assertRaisesRegex(QComemRaggedBatchError, "out-of-pool"):
            prepare_q16_ragged_batch(requests)

        source_arena, source_tail_requests, *_ = self.build_fixture((1,))
        active_tail_index = int(
            source_tail_requests[0].sequence.active_block_table.shape[1]
        ) - 1
        source_tail_requests[0].sequence.block_table[0, active_tail_index].copy_(
            source_arena.document_block_table[0, -1]
        )
        with self.assertRaisesRegex(QComemRaggedBatchError, "entirely request-private"):
            prepare_q16_ragged_batch(source_tail_requests)

        document = torch.zeros(
            1, KV_HEADS, PAGE_SIZE, HEAD_DIM, dtype=torch.float16
        )
        empty_arena = Q16PagedArena.from_dense_document(
            document,
            document.clone(),
            page_size=PAGE_SIZE,
            max_append_tokens=1,
            max_forks=1,
        )
        sequence = empty_arena.fork()
        request = Q16RaggedRequest(
            sequence,
            torch.zeros(1, QUERY_HEADS, HEAD_DIM, dtype=torch.float16),
            torch.tensor([PAGE_SIZE - 1], dtype=torch.long),
        )
        with self.assertRaisesRegex(QComemRaggedBatchError, "already-appended"):
            prepare_q16_ragged_batch((request,))

    def test_cpu_requires_injected_kernel_and_scale_fails_closed(self) -> None:
        _, requests, *_ = self.build_fixture((1,))
        with self.assertRaisesRegex(QComemRaggedBatchError, "requires CUDA"):
            q16_ragged_paged_attention(requests)
        kernel = MetadataOnlyKernel()
        for value in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(scale=value):
                with self.assertRaisesRegex(QComemRaggedBatchError, "softmax_scale"):
                    q16_ragged_paged_attention(
                        requests,
                        softmax_scale=value,
                        _kernel=kernel,
                    )

    def test_frozen_signature_contract_has_sixteen_required_parameters(self) -> None:
        self.assertEqual(FROZEN_REQUIRED_PARAMETER_COUNT, 16)
        self.assertEqual(
            FROZEN_UNIFIED_ATTENTION_PARAMETERS[:16],
            (
                "q",
                "k",
                "v",
                "out",
                "cu_seqlens_q",
                "max_seqlen_q",
                "seqused_k",
                "max_seqlen_k",
                "softmax_scale",
                "causal",
                "window_size",
                "block_table",
                "softcap",
                "q_descale",
                "k_descale",
                "v_descale",
            ),
        )


if __name__ == "__main__":
    unittest.main()
