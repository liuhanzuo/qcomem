from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from qcomem_qwen35_vllm_paged_integration import (
    convert_all_qwen35_full_layers_to_vllm_q16,
    fork_qwen35_vllm_q16_request,
)
from qcomem_vllm_paged_fair_control import (
    FAIR_PROTOCOL,
    FRESH_CONTROL,
    SHARED_REUSE,
    QComemFairControlError,
    Qwen35FairHitLedger,
    build_same_kernel_q16_sequence_pair,
    full_attention_storage_breakdown,
    linear_gdn_shared_base_contract,
    materialize_fresh_q16_request_layer,
    materialize_qwen35_fresh_full_copy_request,
    snapshot_linear_gdn_state,
    storage_residency,
    verify_linear_gdn_state_parity,
)
from qcomem_vllm_paged_kernel import (
    Q16KernelPagedDocumentLayer,
    Q16PagedArena,
    vllm_triton_q16_paged_attention_forward,
)


class DenseOracleKernel:
    """CPU stand-in with vLLM unified_attention's audited keyword API."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

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
        for batch_index in range(int(table.shape[0])):
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
            query_length = q_end - q_start
            past = length - query_length
            qpos = torch.arange(query_length) + past
            kpos = torch.arange(length)
            scores.masked_fill_(
                kpos.view(1, 1, -1) > qpos.view(1, -1, 1), -torch.inf
            )
            weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            out[q_start:q_end].copy_(torch.matmul(weights, value).transpose(0, 1))


class ZeroKernel:
    def __call__(self, **kwargs):
        kwargs["out"].zero_()


class LinearLayer:
    def __init__(self) -> None:
        self.conv_states = {0: torch.zeros(1, 2, 3)}
        self.recurrent_states = {0: torch.zeros(1, 2, 4, 4)}
        self.is_conv_states_initialized = {0: True}
        self.is_recurrent_states_initialized = {0: True}
        self.has_previous_state = {0: True}
        self.conv_kernel_size = {0: 3}
        self.record_past = False

    def lazy_initialization(self, **kwargs):
        del kwargs


class FullLayer:
    is_sliding = False

    def __init__(self, seed: int, length: int = 35) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.keys = torch.randn(1, 2, length, 32, generator=generator)
        self.values = torch.randn(1, 2, length, 32, generator=generator)


def make_cache_and_plan(length: int = 35):
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(40)
    ]
    full = tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention")
    linear = tuple(i for i, kind in enumerate(layer_types) if kind == "linear_attention")
    cache = SimpleNamespace(
        layers=[
            FullLayer(100 + i, length) if kind == "full_attention" else LinearLayer()
            for i, kind in enumerate(layer_types)
        ]
    )
    config = SimpleNamespace(layer_types=layer_types)
    plan = SimpleNamespace(
        full_attention_layer_indices=full,
        linear_layer_indices=linear,
        gdn=config,
    )
    return cache, plan


class FairPagedControlTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260814)
        self.document_key = torch.randn(1, 2, 35, 32)
        self.document_value = torch.randn_like(self.document_key)
        self.append_key = torch.randn(1, 2, 30, 32)
        self.append_value = torch.randn_like(self.append_key)
        self.query = torch.randn(1, 16, 30, 32)
        self.module = SimpleNamespace(
            is_causal=True,
            num_key_value_groups=8,
            scaling=32**-0.5,
        )

    def test_tail_and_cross_block_pair_has_exact_logical_payload_and_output(self):
        fresh, reuse, pair = build_same_kernel_q16_sequence_pair(
            self.document_key,
            self.document_value,
            self.append_key,
            self.append_value,
            page_size=16,
            max_append_tokens=32,
        )
        self.assertTrue(pair["layout"]["canonical_layout_equal"])
        self.assertTrue(pair["layout"]["valid_key_payload_bitwise_exact"])
        self.assertTrue(pair["layout"]["valid_value_payload_bitwise_exact"])
        self.assertTrue(pair["document_source_immutable"])
        self.assertTrue(pair["append_crossed_block_boundary"])
        self.assertEqual(pair["document_tail_tokens"], 3)
        expected_tail = 2 * 3 * 2 * 32 * 4
        self.assertEqual(pair["fresh_partial_tail_copy_nbytes"], expected_tail)
        self.assertEqual(pair["shared_partial_tail_copy_nbytes"], expected_tail)
        self.assertGreater(pair["fresh_document_block_copy_nbytes"], 0)
        self.assertEqual(pair["shared_full_document_copy_nbytes"], 0)
        self.assertFalse(pair["layout"]["raw_physical_block_ids_required_equal"])
        self.assertFalse(pair["layout"]["invalid_final_block_padding_compared"])

        kernel = DenseOracleKernel()
        fresh_output, _ = vllm_triton_q16_paged_attention_forward(
            self.module,
            self.query,
            fresh.keys,
            fresh.values,
            None,
            _kernel=kernel,
        )
        reuse_output, _ = vllm_triton_q16_paged_attention_forward(
            self.module,
            self.query,
            reuse.keys,
            reuse.values,
            None,
            _kernel=kernel,
        )
        self.assertTrue(torch.equal(fresh_output, reuse_output))
        self.assertEqual(len(kernel.calls), 2)
        for call in kernel.calls:
            self.assertTrue(call["causal"])
            self.assertEqual(call["max_seqlen_k"], 65)
            self.assertEqual(call["max_seqlen_q"], 30)

    def test_aligned_document_uses_no_partial_tail_copy(self):
        document_key = self.document_key[..., :32, :]
        document_value = self.document_value[..., :32, :]
        _, _, pair = build_same_kernel_q16_sequence_pair(
            document_key,
            document_value,
            self.append_key[..., :7, :],
            self.append_value[..., :7, :],
            page_size=16,
            max_append_tokens=8,
        )
        self.assertEqual(pair["document_tail_tokens"], 0)
        self.assertEqual(pair["fresh_partial_tail_copy_nbytes"], 0)
        self.assertEqual(pair["shared_partial_tail_copy_nbytes"], 0)
        self.assertEqual(pair["current_append_delta_tokens"], 7)

    def test_fresh_control_rejects_multi_fork_source(self):
        arena = Q16PagedArena.from_dense_document(
            self.document_key,
            self.document_value,
            page_size=16,
            max_append_tokens=4,
            max_forks=2,
        )
        with self.assertRaisesRegex(QComemFairControlError, "single-request"):
            materialize_fresh_q16_request_layer(
                Q16KernelPagedDocumentLayer(arena)
            )

    def test_qwen35_full_copy_and_reuse_have_explicit_storage_ownership(self):
        cache, plan = make_cache_and_plan()
        convert_all_qwen35_full_layers_to_vllm_q16(
            cache,
            plan,
            page_size=16,
            max_append_tokens=4,
            max_request_forks=1,
        )
        fresh, fresh_audit = materialize_qwen35_fresh_full_copy_request(cache, plan)
        reuse, reuse_audit = fork_qwen35_vllm_q16_request(cache, plan)
        self.assertEqual(fresh_audit["request_policy"], FRESH_CONTROL)
        self.assertGreater(fresh_audit["full_document_staging_copy_nbytes"], 0)
        self.assertFalse(fresh_audit["source_document_storage_shared"])
        self.assertEqual(reuse_audit["full_document_staging_copy_nbytes"], 0)
        self.assertEqual(len(fresh_audit["request_arena_ids"]), 10)
        for index in plan.full_attention_layer_indices:
            self.assertIsNot(
                fresh.layers[index].sequence.arena,
                cache.layers[index].arena,
            )
            self.assertIs(
                reuse.layers[index].sequence.arena,
                cache.layers[index].arena,
            )
        fresh_memory = storage_residency(cache, fresh)
        reuse_memory = storage_residency(cache, reuse)
        self.assertGreater(
            fresh_memory["request_unique_total_nbytes"],
            reuse_memory["request_unique_total_nbytes"],
        )
        self.assertGreater(
            reuse_memory["shared_total_nbytes"],
            fresh_memory["shared_total_nbytes"],
        )
        fresh_storage = full_attention_storage_breakdown(
            cache,
            fresh,
            plan.full_attention_layer_indices,
            request_policy=FRESH_CONTROL,
        )
        reuse_storage = full_attention_storage_breakdown(
            cache,
            reuse,
            plan.full_attention_layer_indices,
            request_policy=SHARED_REUSE,
        )
        self.assertEqual(fresh_storage["full_attention_layer_count"], 10)
        self.assertTrue(
            fresh_storage["source_arena_includes_preallocated_private_reservation"]
        )
        self.assertGreater(
            fresh_storage["totals"]["fresh_duplicate_document_allocated_nbytes"],
            0,
        )
        self.assertGreater(
            fresh_storage["totals"]["fresh_private_reservation_nbytes"], 0
        )
        self.assertEqual(
            reuse_storage["totals"]["fresh_duplicate_document_allocated_nbytes"],
            0,
        )
        self.assertEqual(
            fresh_storage["totals"]["valid_document_payload_nbytes"],
            reuse_storage["totals"]["valid_document_payload_nbytes"],
        )
        self.assertGreater(
            fresh_storage["totals"]["fresh_document_table_accelerator_nbytes"],
            0,
        )

        snapshot = snapshot_linear_gdn_state(cache, plan.linear_layer_indices)
        fresh_base = linear_gdn_shared_base_contract(
            cache, fresh, plan.linear_layer_indices
        )
        reuse_base = linear_gdn_shared_base_contract(
            cache, reuse, plan.linear_layer_indices
        )
        self.assertEqual(fresh_base["linear_layer_count"], 30)
        self.assertTrue(reuse_base["persistent_tensor_base_shared_at_request_start"])
        for index in plan.linear_layer_indices:
            fresh.layers[index].conv_states[0] = torch.ones_like(
                fresh.layers[index].conv_states[0]
            )
            reuse.layers[index].conv_states[0] = torch.ones_like(
                reuse.layers[index].conv_states[0]
            )
        parity = verify_linear_gdn_state_parity(
            fresh,
            reuse,
            cache,
            snapshot,
            plan.linear_layer_indices,
        )
        self.assertTrue(parity["fresh_reuse_functional_state_bitwise_exact"])
        self.assertTrue(parity["persistent_tensor_base_unchanged"])

    def test_ledger_requires_current_append_delta_and_all_ten_kernel_hits(self):
        cache, plan = make_cache_and_plan(length=32)
        convert_all_qwen35_full_layers_to_vllm_q16(
            cache,
            plan,
            page_size=16,
            max_append_tokens=2,
            max_request_forks=1,
        )
        request, _ = materialize_qwen35_fresh_full_copy_request(cache, plan)
        for index in plan.full_attention_layer_indices:
            request.layers[index].sequence.strict_mask_check = False
        ledger = Qwen35FairHitLedger(
            plan,
            request,
            request_policy=FRESH_CONTROL,
            expected_calls_per_layer=1,
            strict_tail_values=False,
            kernel=ZeroKernel(),
        )
        first_index = plan.full_attention_layer_indices[0]
        first_layer = request.layers[first_index]
        with self.assertRaisesRegex(QComemFairControlError, "append delta"):
            ledger.attention_forward(
                SimpleNamespace(
                    layer_idx=first_index,
                    is_causal=True,
                    num_key_value_groups=8,
                    scaling=32**-0.5,
                ),
                torch.zeros(1, 16, 2, 32),
                first_layer.keys,
                first_layer.values,
                None,
                position_ids=torch.tensor([[32, 33]]),
            )

        for index in plan.full_attention_layer_indices:
            layer = request.layers[index]
            new_key = torch.zeros(1, 2, 2, 32)
            new_value = torch.zeros_like(new_key)
            key, value = layer.update(new_key, new_value)
            ledger.attention_forward(
                SimpleNamespace(
                    layer_idx=index,
                    is_causal=True,
                    num_key_value_groups=8,
                    scaling=32**-0.5,
                ),
                torch.zeros(1, 16, 2, 32),
                key,
                value,
                None,
                position_ids=torch.tensor([[32, 33]]),
            )
        audit = ledger.verify_complete()
        self.assertEqual(audit["fair_protocol"], FAIR_PROTOCOL)
        self.assertEqual(audit["request_policy"], FRESH_CONTROL)
        self.assertEqual(audit["total_calls"], 10)
        self.assertEqual(audit["dense_fallback_calls"], 0)
        self.assertEqual(audit["materialized_attention_mask_nbytes"], 0)
        self.assertEqual(audit["position_ids_validation_host_syncs"], 0)
        self.assertTrue(
            all(call["current_append_delta_tokens"] == 2 for call in audit["calls"])
        )

    def test_fresh_and_reuse_ledgers_can_share_one_kernel_object(self):
        cache, plan = make_cache_and_plan(length=32)
        convert_all_qwen35_full_layers_to_vllm_q16(
            cache,
            plan,
            page_size=16,
            max_append_tokens=2,
            max_request_forks=1,
        )
        fresh, _ = materialize_qwen35_fresh_full_copy_request(cache, plan)
        reuse, _ = fork_qwen35_vllm_q16_request(cache, plan)
        kernel = ZeroKernel()
        fresh_ledger = Qwen35FairHitLedger(
            plan,
            fresh,
            request_policy=FRESH_CONTROL,
            expected_calls_per_layer=1,
            strict_tail_values=True,
            kernel=kernel,
        )
        reuse_ledger = Qwen35FairHitLedger(
            plan,
            reuse,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            strict_tail_values=True,
            kernel=kernel,
        )
        self.assertIs(fresh_ledger.kernel, reuse_ledger.kernel)
        self.assertEqual(
            fresh_ledger.kernel_identity, reuse_ledger.kernel_identity
        )


if __name__ == "__main__":
    unittest.main()
