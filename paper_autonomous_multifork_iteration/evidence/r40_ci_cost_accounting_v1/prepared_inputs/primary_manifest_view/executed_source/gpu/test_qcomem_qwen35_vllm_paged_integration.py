from __future__ import annotations

import sys
import types
import unittest
import uuid
from types import SimpleNamespace
from unittest import mock

import torch

from qcomem_qwen35_vllm_paged_integration import (
    POST_ROPE_POSITION_IDS_CONTRACT,
    Qwen35VllmPagedHitLedger,
    Qwen35VllmPagedIntegrationError,
    convert_all_qwen35_full_layers_to_vllm_q16,
    fork_qwen35_vllm_q16_request,
    full_vocab_forward_kl,
    register_qwen35_vllm_q16_backend,
    validate_qwen35_post_rope_position_ids,
)
from qcomem_vllm_paged_kernel import Q16KernelPagedDocumentLayer


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

    def __init__(self, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.keys = torch.randn(1, 2, 35, 32, generator=generator)
        self.values = torch.randn(1, 2, 35, 32, generator=generator)

    def get_seq_length(self):
        return self.keys.shape[-2]


def make_cache_and_plan():
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(40)
    ]
    full = tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention")
    linear = tuple(i for i, kind in enumerate(layer_types) if kind == "linear_attention")
    layers = [
        FullLayer(100 + i) if kind == "full_attention" else LinearLayer()
        for i, kind in enumerate(layer_types)
    ]
    config = SimpleNamespace(layer_types=layer_types)
    plan = SimpleNamespace(
        full_attention_layer_indices=full,
        linear_layer_indices=linear,
        gdn=config,
    )
    return SimpleNamespace(layers=layers), plan


class ZeroKernel:
    def __call__(self, **kwargs):
        kwargs["out"].zero_()


class Qwen35VllmPagedIntegrationTest(unittest.TestCase):
    def test_atomic_ten_layer_conversion_and_request_fork(self):
        cache, plan = make_cache_and_plan()
        conversion = convert_all_qwen35_full_layers_to_vllm_q16(
            cache,
            plan,
            page_size=16,
            max_append_tokens=16,
            max_request_forks=2,
        )
        self.assertEqual(conversion.layer_indices, plan.full_attention_layer_indices)
        self.assertEqual(len(conversion.layer_arena_ids), 10)
        self.assertEqual(conversion.document_length, 35)
        self.assertTrue(
            all(
                isinstance(cache.layers[i], Q16KernelPagedDocumentLayer)
                for i in plan.full_attention_layer_indices
            )
        )
        request, audit = fork_qwen35_vllm_q16_request(cache, plan)
        self.assertTrue(audit["linear_functional_rebind"])
        self.assertEqual(audit["full_document_staging_copy_nbytes"], 0)
        self.assertEqual(audit["allocated_request_pool_nbytes"], 0)
        self.assertTrue(audit["source_document_storage_shared"])
        self.assertEqual(audit["request_policy"], "vllm-q16-shared-document-reuse")
        for i in plan.full_attention_layer_indices:
            self.assertIs(request.layers[i].sequence.arena, cache.layers[i].arena)
        for i in plan.linear_layer_indices:
            self.assertEqual(
                request.layers[i]._qcomem_update_mode, "functional-state-rebind"
            )

    def test_q8_q4_fail_before_mutating_dense_layers(self):
        for bits in (8, 4):
            cache, plan = make_cache_and_plan()
            originals = [cache.layers[i] for i in plan.full_attention_layer_indices]
            with self.assertRaisesRegex(Qwen35VllmPagedIntegrationError, "Q16 only"):
                convert_all_qwen35_full_layers_to_vllm_q16(
                    cache,
                    plan,
                    page_size=16,
                    max_append_tokens=4,
                    max_request_forks=1,
                    bits=bits,
                )
            self.assertEqual(
                originals, [cache.layers[i] for i in plan.full_attention_layer_indices]
            )

    def test_ledger_covers_all_ten_layers_without_dense_fallback(self):
        cache, plan = make_cache_and_plan()
        conversion = convert_all_qwen35_full_layers_to_vllm_q16(
            cache,
            plan,
            page_size=16,
            max_append_tokens=2,
            max_request_forks=1,
        )
        request, _ = fork_qwen35_vllm_q16_request(cache, plan)
        ledger = Qwen35VllmPagedHitLedger(plan, conversion)
        with mock.patch(
            "qcomem_vllm_paged_kernel._resolve_vllm_unified_attention",
            return_value=ZeroKernel(),
        ):
            for index in plan.full_attention_layer_indices:
                layer = request.layers[index]
                new_key = torch.zeros(1, 2, 2, 32)
                new_value = torch.zeros_like(new_key)
                key, value = layer.update(new_key, new_value)
                query = torch.zeros(1, 16, 2, 32)
                module = SimpleNamespace(
                    layer_idx=index,
                    is_causal=True,
                    num_key_value_groups=8,
                    scaling=32**-0.5,
                )
                ledger.attention_forward(
                    module,
                    query,
                    key,
                    value,
                    None,
                    position_ids=torch.tensor([[35, 36]], dtype=torch.long),
                )
        result = ledger.verify_complete()
        self.assertEqual(result["total_calls"], 10)
        self.assertEqual(result["dense_fallback_calls"], 0)
        self.assertEqual(
            result["position_ids_contract"], POST_ROPE_POSITION_IDS_CONTRACT
        )
        self.assertEqual(result["position_ids_validation_host_syncs"], 10)

    def test_post_rope_position_ids_are_explicitly_validated(self):
        query = torch.zeros(1, 16, 2, 32)
        strict = validate_qwen35_post_rope_position_ids(
            torch.tensor([[35, 36]], dtype=torch.long),
            query=query,
            total_length=37,
            strict_tail_values=True,
        )
        self.assertEqual(
            strict["position_ids_contract"], POST_ROPE_POSITION_IDS_CONTRACT
        )
        self.assertTrue(strict["position_ids_semantically_consumed_upstream"])
        self.assertTrue(strict["position_ids_strict_tail_values_checked"])
        self.assertEqual(strict["position_ids_validation_host_syncs"], 1)
        metadata_only = validate_qwen35_post_rope_position_ids(
            torch.tensor([[999, 1000]], dtype=torch.long),
            query=query,
            total_length=37,
            strict_tail_values=False,
        )
        self.assertFalse(metadata_only["position_ids_strict_tail_values_checked"])
        self.assertEqual(metadata_only["position_ids_validation_host_syncs"], 0)
        with self.assertRaisesRegex(
            Qwen35VllmPagedIntegrationError, "canonical contiguous causal tail"
        ):
            validate_qwen35_post_rope_position_ids(
                torch.tensor([[34, 36]], dtype=torch.long),
                query=query,
                total_length=37,
                strict_tail_values=True,
            )
        with self.assertRaisesRegex(Qwen35VllmPagedIntegrationError, "torch.long"):
            validate_qwen35_post_rope_position_ids(
                torch.tensor([[35, 36]], dtype=torch.int32),
                query=query,
                total_length=37,
                strict_tail_values=True,
            )

    def test_production_ledger_consumes_position_ids_without_host_sync(self):
        cache, plan = make_cache_and_plan()
        conversion = convert_all_qwen35_full_layers_to_vllm_q16(
            cache,
            plan,
            page_size=16,
            max_append_tokens=1,
            max_request_forks=1,
        )
        request, _ = fork_qwen35_vllm_q16_request(cache, plan)
        ledger = Qwen35VllmPagedHitLedger(
            plan,
            conversion,
            mask_contract="prevalidated-no-padding-tail-causal",
        )
        with mock.patch(
            "qcomem_vllm_paged_kernel._resolve_vllm_unified_attention",
            return_value=ZeroKernel(),
        ):
            for index in plan.full_attention_layer_indices:
                layer = request.layers[index]
                layer.sequence.strict_mask_check = False
                new_key = torch.zeros(1, 2, 1, 32)
                key, value = layer.update(new_key, torch.zeros_like(new_key))
                query = torch.zeros(1, 16, 1, 32)
                module = SimpleNamespace(
                    layer_idx=index,
                    is_causal=True,
                    num_key_value_groups=8,
                    scaling=32**-0.5,
                )
                ledger.attention_forward(
                    module,
                    query,
                    key,
                    value,
                    None,
                    position_ids=torch.tensor([[35]], dtype=torch.long),
                )
        result = ledger.verify_complete()
        self.assertEqual(result["position_ids_validation_host_syncs"], 0)
        self.assertTrue(
            all(
                call["position_ids_validated"]
                and call["position_ids_semantically_consumed_upstream"]
                and not call["position_ids_strict_tail_values_checked"]
                for call in result["calls"]
            )
        )

    def test_real_tf514_qwen_call_consumes_and_advances_position_ids(self):
        try:
            import transformers
        except ImportError:
            self.skipTest("real Transformers is unavailable")
        if transformers.__version__ != "5.14.1":
            self.skipTest("real call-stack test is frozen to Transformers 5.14.1")

        from transformers.masking_utils import AttentionMaskInterface, eager_mask
        from transformers.modeling_utils import AttentionInterface
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeTextConfig,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeTextModel,
            eager_attention_forward,
        )

        captures = []

        def spy(module, query, key, value, attention_mask, *args, **kwargs):
            captures.append(
                {
                    "position_ids": kwargs["position_ids"].detach().cpu().clone(),
                    "query": query.detach().cpu().clone(),
                    "key": key.detach().cpu().clone(),
                }
            )
            return eager_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                *args,
                **kwargs,
            )

        backend = f"qcomem_position_ids_spy_{uuid.uuid4().hex}"
        AttentionInterface.register(backend, spy)
        AttentionMaskInterface.register(backend, eager_mask)
        config = Qwen3_5MoeTextConfig(
            vocab_size=32,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            moe_intermediate_size=16,
            shared_expert_intermediate_size=16,
            num_experts=2,
            num_experts_per_tok=1,
            layer_types=["full_attention"],
            pad_token_id=0,
        )
        model = Qwen3_5MoeTextModel(config).eval()
        model.config._attn_implementation = backend
        tokens = torch.tensor([[1, 2, 3]], dtype=torch.long)
        with torch.inference_mode():
            prefill = model(input_ids=tokens, use_cache=True)
            model(
                input_ids=torch.tensor([[4]], dtype=torch.long),
                past_key_values=prefill.past_key_values,
                use_cache=True,
            )
            model(
                input_ids=tokens,
                position_ids=torch.tensor([[7, 8, 9]], dtype=torch.long),
                use_cache=False,
            )

        self.assertEqual(len(captures), 3)
        self.assertEqual(captures[0]["position_ids"].tolist(), [[0, 1, 2]])
        self.assertEqual(captures[1]["position_ids"].tolist(), [[3]])
        self.assertEqual(captures[2]["position_ids"].tolist(), [[7, 8, 9]])
        self.assertFalse(torch.equal(captures[0]["query"], captures[2]["query"]))
        self.assertFalse(torch.equal(captures[0]["key"], captures[2]["key"]))
        validate_qwen35_post_rope_position_ids(
            captures[0]["position_ids"],
            query=captures[0]["query"],
            total_length=3,
            strict_tail_values=True,
        )
        validate_qwen35_post_rope_position_ids(
            captures[1]["position_ids"],
            query=captures[1]["query"],
            total_length=4,
            strict_tail_values=True,
        )
        with self.assertRaisesRegex(
            Qwen35VllmPagedIntegrationError, "canonical contiguous causal tail"
        ):
            validate_qwen35_post_rope_position_ids(
                captures[2]["position_ids"],
                query=captures[2]["query"],
                total_length=3,
                strict_tail_values=True,
            )

    def test_registry_is_tf514_pinned_and_dual_registered(self):
        cache, plan = make_cache_and_plan()
        conversion = convert_all_qwen35_full_layers_to_vllm_q16(
            cache,
            plan,
            page_size=16,
            max_append_tokens=1,
            max_request_forks=1,
        )
        ledger = Qwen35VllmPagedHitLedger(plan, conversion)
        attention, masks = {}, {}

        class AttentionInterface:
            @classmethod
            def register(cls, name, value):
                attention[name] = value

        class AttentionMaskInterface:
            @classmethod
            def register(cls, name, value):
                masks[name] = value

        transformers = types.ModuleType("transformers")
        transformers.__path__ = []
        transformers.__version__ = "5.14.1"
        modeling = types.ModuleType("transformers.modeling_utils")
        modeling.AttentionInterface = AttentionInterface
        masking = types.ModuleType("transformers.masking_utils")
        masking.AttentionMaskInterface = AttentionMaskInterface
        masking.eager_mask = object()
        modules = {
            "transformers": transformers,
            "transformers.modeling_utils": modeling,
            "transformers.masking_utils": masking,
        }
        with mock.patch.dict(sys.modules, modules):
            registered = register_qwen35_vllm_q16_backend(ledger, name="kernel_test")
            self.assertIn(registered.name, attention)
            self.assertIn(registered.name, masks)
            transformers.__version__ = "5.15.0"
            with self.assertRaisesRegex(Qwen35VllmPagedIntegrationError, "5.14"):
                register_qwen35_vllm_q16_backend(ledger, name="wrong_version")

    def test_full_vocab_kl_is_zero_only_for_identical_logits(self):
        logits = torch.randn(3, 17)
        torch.testing.assert_close(
            full_vocab_forward_kl(logits, logits), torch.zeros(3), atol=1e-7, rtol=0
        )
        shifted = logits.clone()
        shifted[:, 0] += 1
        self.assertTrue((full_vocab_forward_kl(logits, shifted) > 0).all())

    def test_production_registry_uses_fail_closed_zero_materialized_mask(self):
        cache, plan = make_cache_and_plan()
        conversion = convert_all_qwen35_full_layers_to_vllm_q16(
            cache,
            plan,
            page_size=16,
            max_append_tokens=1,
            max_request_forks=1,
        )
        ledger = Qwen35VllmPagedHitLedger(
            plan,
            conversion,
            mask_contract="prevalidated-no-padding-tail-causal",
        )
        attention, masks = {}, {}

        class AttentionInterface:
            @classmethod
            def register(cls, name, value):
                attention[name] = value

        class AttentionMaskInterface:
            @classmethod
            def register(cls, name, value):
                masks[name] = value

        transformers = types.ModuleType("transformers")
        transformers.__path__ = []
        transformers.__version__ = "5.14.1"
        modeling = types.ModuleType("transformers.modeling_utils")
        modeling.AttentionInterface = AttentionInterface
        masking = types.ModuleType("transformers.masking_utils")
        masking.AttentionMaskInterface = AttentionMaskInterface
        masking.eager_mask = object()
        modules = {
            "transformers": transformers,
            "transformers.modeling_utils": modeling,
            "transformers.masking_utils": masking,
        }
        with mock.patch.dict(sys.modules, modules):
            registered = register_qwen35_vllm_q16_backend(
                ledger, name="production_no_mask_test"
            )
        mask_factory = masks[registered.name]
        self.assertIsNone(mask_factory(attention_mask=None))
        with self.assertRaisesRegex(Qwen35VllmPagedIntegrationError, "padding/custom"):
            mask_factory(attention_mask=torch.ones(1, 4))


if __name__ == "__main__":
    unittest.main()
