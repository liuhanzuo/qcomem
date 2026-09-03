from __future__ import annotations

import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from qcomem_paged_attention import PagedKVLayer
from qcomem_qwen35_paged_integration import (
    AutogradPagedKVLayer,
    KERNEL_MODE,
    PagedAttentionHitLedger,
    Qwen35PagedIntegrationError,
    RegisteredPagedBackend,
    audit_qwen35_full_attention_plan,
    clone_dense_and_prepare_paged_cache_pair,
    convert_all_planned_cache_layers,
    convert_all_planned_cache_layers_for_training,
    register_qwen35_paged_backend,
    require_passed_reference_gate_before_benchmark,
    run_same_caller_eager_paged_gate,
    temporary_attention_implementation,
)


def layer_types(count: int) -> list[str]:
    return [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(count)
    ]


class FakeAttention:
    def __init__(self, config, layer_idx: int) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.num_key_value_groups = 2
        self.scaling = 8**-0.5
        self.is_causal = True

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        del position_embeddings, attention_mask, past_key_values, kwargs
        return hidden_states


class FakeDecoder:
    def __init__(self, config, index: int, kind: str) -> None:
        self.block_type = kind
        if kind == "full_attention":
            self.self_attn = FakeAttention(config, index)
        else:
            self.linear_attn = object()


class FakeBackbone:
    def __init__(self, count: int = 40) -> None:
        kinds = layer_types(count)
        self.config = SimpleNamespace(
            model_type="qwen3_5_moe_text",
            num_hidden_layers=count,
            layer_types=kinds,
            _attn_implementation="sdpa",
        )
        self.layers = [
            FakeDecoder(self.config, index, kind)
            for index, kind in enumerate(kinds)
        ]


class FakeDenseLayer:
    is_sliding = False

    def __init__(self, key: torch.Tensor, value: torch.Tensor) -> None:
        self.keys = key.clone()
        self.values = value.clone()
        self.is_initialized = True

    def update(self, key: torch.Tensor, value: torch.Tensor, *args, **kwargs):
        del args, kwargs
        self.keys = torch.cat([self.keys, key], dim=-2)
        self.values = torch.cat([self.values, value], dim=-2)
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.keys.shape[-2]


class FakeLinearCacheLayer:
    def __init__(self) -> None:
        self.recurrent_states = {0: torch.zeros(1, 2, 4, 4)}


class FakeCache:
    def __init__(self, layers) -> None:
        self.layers = list(layers)

    def update(self, key, value, layer_idx, *args, **kwargs):
        return self.layers[layer_idx].update(key, value, *args, **kwargs)


def make_cache(backbone: FakeBackbone, document_length: int = 5) -> FakeCache:
    layers = []
    for index, kind in enumerate(backbone.config.layer_types):
        if kind == "full_attention":
            generator = torch.Generator().manual_seed(1000 + index)
            key = torch.randn(1, 2, document_length, 8, generator=generator)
            value = torch.randn(1, 2, document_length, 8, generator=generator)
            layers.append(FakeDenseLayer(key, value))
        else:
            layers.append(FakeLinearCacheLayer())
    return FakeCache(layers)


def repeat_kv(states: torch.Tensor, groups: int) -> torch.Tensor:
    batch, heads, tokens, dim = states.shape
    return (
        states[:, :, None, :, :]
        .expand(batch, heads, groups, tokens, dim)
        .reshape(batch, heads * groups, tokens, dim)
    )


def dense_eager(
    module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key.transpose(2, 3)) * module.scaling + mask
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value).transpose(1, 2).contiguous()


class SameCaller:
    def __init__(self, backbone, ledger, backend_name: str) -> None:
        self.backbone = backbone
        self.ledger = ledger
        self.backend_name = backend_name
        self.query_length = 2
        self.document_length = 5
        self.projection = torch.randn(32, 13, generator=torch.Generator().manual_seed(4))

    def __call__(self, cache):
        combined = None
        total_length = self.document_length + self.query_length
        query_positions = torch.arange(self.query_length) + self.document_length
        key_positions = torch.arange(total_length)
        allowed = key_positions.view(1, -1) <= query_positions.view(-1, 1)
        mask = torch.zeros(1, 1, self.query_length, total_length)
        mask.masked_fill_(~allowed.view(1, 1, self.query_length, total_length), -torch.inf)
        for index in self.ledger.expected_layer_indices:
            generator = torch.Generator().manual_seed(5000 + index)
            query = torch.randn(1, 4, self.query_length, 8, generator=generator)
            key = torch.randn(1, 2, self.query_length, 8, generator=generator)
            value = torch.randn(1, 2, self.query_length, 8, generator=generator)
            key, value = cache.update(key, value, index)
            module = self.backbone.layers[index].self_attn
            if self.backbone.config._attn_implementation == "eager":
                output = dense_eager(module, query, key, value, mask)
            elif self.backbone.config._attn_implementation == self.backend_name:
                output, _ = self.ledger.attention_forward(
                    module,
                    query,
                    key,
                    value,
                    mask,
                    scaling=module.scaling,
                )
            else:
                raise AssertionError("unexpected attention implementation")
            combined = output if combined is None else combined + output
        features = combined[:, -1, :].reshape(1, -1)
        logits = features @ self.projection
        logits[:, 0] += 20.0
        return SimpleNamespace(logits=logits.unsqueeze(1))


class Qwen35PagedIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260813)
        self.backbone = FakeBackbone()
        self.plan = audit_qwen35_full_attention_plan(
            self.backbone,
            layer_start=0,
            expected_full_attention_layers=sum(
                kind == "full_attention"
                for kind in self.backbone.config.layer_types
            ),
        )

    def test_dynamic_plan_audits_all_configured_full_attention_layers(self) -> None:
        expected = tuple(
            index
            for index, kind in enumerate(self.backbone.config.layer_types)
            if kind == "full_attention"
        )
        self.assertEqual(self.plan.full_attention_layer_indices, expected)
        self.assertEqual(
            self.plan.full_attention_layer_count,
            self.backbone.config.layer_types.count("full_attention"),
        )
        self.assertEqual(
            self.plan.transformers_api["kernel_mode"], KERNEL_MODE
        )
        self.assertFalse(
            self.plan.transformers_api[
                "production_ttft_optimization_claim_allowed"
            ]
        )

    def test_plan_fails_closed_on_count_or_real_module_api_drift(self) -> None:
        with self.assertRaisesRegex(
            Qwen35PagedIntegrationError, "dynamic full-attention count mismatch"
        ):
            audit_qwen35_full_attention_plan(
                self.backbone,
                expected_full_attention_layers=(
                    self.plan.full_attention_layer_count - 1
                ),
            )
        index = self.plan.full_attention_layer_indices[0]
        self.backbone.layers[index].self_attn.layer_idx = index + 1
        with self.assertRaisesRegex(Qwen35PagedIntegrationError, "layer_idx mismatch"):
            audit_qwen35_full_attention_plan(self.backbone)

    def test_all_layer_conversion_is_complete_and_preflight_is_atomic(self) -> None:
        cache = make_cache(self.backbone)
        broken_index = self.plan.full_attention_layer_indices[-1]
        broken_layer = cache.layers[broken_index]
        cache.layers[broken_index] = SimpleNamespace(
            keys=broken_layer.keys,
            values=None,
            is_sliding=False,
        )
        with self.assertRaisesRegex(Qwen35PagedIntegrationError, "dense Tensor"):
            convert_all_planned_cache_layers(cache, self.plan, page_size=2)
        self.assertTrue(
            all(
                not isinstance(cache.layers[index], PagedKVLayer)
                for index in self.plan.full_attention_layer_indices
            )
        )

        cache = make_cache(self.backbone)
        non_targets = {
            index: cache.layers[index]
            for index in range(len(cache.layers))
            if index not in self.plan.full_attention_layer_indices
        }
        conversion = convert_all_planned_cache_layers(
            cache,
            self.plan,
            page_size=2,
            bits=4,
            group_size=16,
        )
        self.assertEqual(conversion.layer_indices, self.plan.full_attention_layer_indices)
        self.assertEqual(conversion.document_length, 5)
        self.assertLess(
            conversion.paged_document_nbytes,
            conversion.dense_document_nbytes,
        )
        self.assertTrue(
            all(
                isinstance(cache.layers[index], PagedKVLayer)
                for index in self.plan.full_attention_layer_indices
            )
        )
        self.assertTrue(
            all(cache.layers[index] is layer for index, layer in non_targets.items())
        )

    def test_hit_ledger_rejects_dense_fallback_and_missing_intercepts(self) -> None:
        pair = clone_dense_and_prepare_paged_cache_pair(
            make_cache(self.backbone), self.plan, page_size=2
        )
        ledger = PagedAttentionHitLedger(self.plan, pair.conversion)
        index = self.plan.full_attention_layer_indices[0]
        module = self.backbone.layers[index].self_attn
        query = torch.randn(1, 4, 1, 8)
        key = torch.randn(1, 2, 1, 8)
        value = torch.randn_like(key)
        with self.assertRaisesRegex(Qwen35PagedIntegrationError, "dense K/V"):
            ledger.attention_forward(module, query, key, value, None)
        with self.assertRaisesRegex(Qwen35PagedIntegrationError, "coverage mismatch"):
            ledger.verify_complete()

        layer = pair.paged_cache.layers[index]
        key_view, value_view = layer.update(key, value)
        unexpected = SimpleNamespace(
            layer_idx=0,
            num_key_value_groups=2,
            scaling=8**-0.5,
            is_causal=True,
        )
        with self.assertRaisesRegex(Qwen35PagedIntegrationError, "unexpected"):
            ledger.attention_forward(
                unexpected, query, key_view, value_view, None
            )

    def test_same_caller_gate_requires_every_layer_and_matches_logits_tokens(self) -> None:
        pair = clone_dense_and_prepare_paged_cache_pair(
            make_cache(self.backbone), self.plan, page_size=2
        )
        ledger = PagedAttentionHitLedger(self.plan, pair.conversion)
        backend = RegisteredPagedBackend(
            name="test_qcomem_paged",
            transformers_version="5.15.0",
            ledger=ledger,
        )
        caller = SameCaller(self.backbone, ledger, backend.name)
        result = run_same_caller_eager_paged_gate(
            caller,
            text_config=self.backbone.config,
            caches=pair,
            backend=backend,
            rtol=2e-5,
            atol=2e-5,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["same_caller_object"])
        self.assertTrue(result["final_logits_close"])
        self.assertTrue(result["final_tokens_exact"])
        self.assertTrue(result["cache_lengths_exact"])
        self.assertEqual(
            result["intercept"]["total_calls"],
            self.plan.full_attention_layer_count,
        )
        self.assertEqual(result["intercept"]["dense_fallback_calls"], 0)
        self.assertEqual(self.backbone.config._attn_implementation, "sdpa")
        benchmark = require_passed_reference_gate_before_benchmark(result)
        self.assertTrue(benchmark["benchmark_gate_passed"])
        self.assertEqual(benchmark["kernel_mode"], KERNEL_MODE)
        self.assertFalse(benchmark["production_ttft_optimization_claim_allowed"])

    def test_attention_implementation_context_restores_after_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "stop"):
            with temporary_attention_implementation(
                self.backbone.config, "temporary"
            ):
                self.assertEqual(
                    self.backbone.config._attn_implementation, "temporary"
                )
                raise RuntimeError("stop")
        self.assertEqual(self.backbone.config._attn_implementation, "sdpa")

    def test_transformers_registry_is_unique_version_pinned_and_dual_registered(
        self,
    ) -> None:
        pair = clone_dense_and_prepare_paged_cache_pair(
            make_cache(self.backbone), self.plan, page_size=2
        )
        ledger = PagedAttentionHitLedger(self.plan, pair.conversion)
        attention_mapping = {}
        mask_mapping = {}

        class AttentionInterface:
            @classmethod
            def register(cls, name, value):
                attention_mapping[name] = value

        class AttentionMaskInterface:
            @classmethod
            def register(cls, name, value):
                mask_mapping[name] = value

        class AllInterfaces:
            def __init__(self, mapping):
                self.mapping = mapping

            def valid_keys(self):
                return list(self.mapping)

        transformers = types.ModuleType("transformers")
        transformers.__path__ = []
        transformers.__version__ = "5.15.0"
        modeling = types.ModuleType("transformers.modeling_utils")
        modeling.AttentionInterface = AttentionInterface
        modeling.ALL_ATTENTION_FUNCTIONS = AllInterfaces(attention_mapping)
        masking = types.ModuleType("transformers.masking_utils")
        masking.AttentionMaskInterface = AttentionMaskInterface
        masking.ALL_MASK_ATTENTION_FUNCTIONS = AllInterfaces(mask_mapping)
        masking.eager_mask = object()
        modules = {
            "transformers": transformers,
            "transformers.modeling_utils": modeling,
            "transformers.masking_utils": masking,
        }
        with mock.patch.dict(sys.modules, modules):
            registered = register_qwen35_paged_backend(
                ledger, name="qcomem_test_unique"
            )
            self.assertEqual(registered.transformers_version, "5.15.0")
            self.assertIn(registered.name, attention_mapping)
            self.assertIn(registered.name, mask_mapping)
            with self.assertRaisesRegex(
                Qwen35PagedIntegrationError, "already registered"
            ):
                register_qwen35_paged_backend(
                    ledger, name="qcomem_test_unique"
                )
            transformers.__version__ = "5.14.1"
            registered_514 = register_qwen35_paged_backend(
                ledger, name="qcomem_test_supported_514"
            )
            self.assertEqual(registered_514.transformers_version, "5.14.1")
            transformers.__version__ = "5.13.4"
            with self.assertRaisesRegex(
                Qwen35PagedIntegrationError, "unsupported Transformers API"
            ):
                register_qwen35_paged_backend(
                    ledger, name="qcomem_test_wrong_version"
                )

    def test_autograd_pages_preserve_qkv_parameter_gradients(self) -> None:
        hidden_dim = 12
        query_heads = 4
        kv_heads = 2
        head_dim = 3
        document_hidden = torch.randn(1, 5, hidden_dim, requires_grad=True)
        query_hidden = torch.randn(1, 2, hidden_dim, requires_grad=True)
        q_proj = torch.nn.Linear(hidden_dim, query_heads * head_dim, bias=False)
        k_proj = torch.nn.Linear(hidden_dim, kv_heads * head_dim, bias=False)
        v_proj = torch.nn.Linear(hidden_dim, kv_heads * head_dim, bias=False)

        document_key = k_proj(document_hidden).view(1, 5, kv_heads, head_dim)
        document_key = document_key.transpose(1, 2)
        document_value = v_proj(document_hidden).view(1, 5, kv_heads, head_dim)
        document_value = document_value.transpose(1, 2)
        layer = AutogradPagedKVLayer.from_dense_document(
            document_key,
            document_value,
            page_size=2,
            bits=16,
            preserve_document_graph=True,
        )
        fork = layer.fork()
        self.assertTrue(
            all(
                left is right
                for left, right in zip(
                    layer.store.document_pages, fork.store.document_pages
                )
            )
        )

        query = q_proj(query_hidden).view(1, 2, query_heads, head_dim)
        query = query.transpose(1, 2)
        query_key = k_proj(query_hidden).view(1, 2, kv_heads, head_dim)
        query_key = query_key.transpose(1, 2)
        query_value = v_proj(query_hidden).view(1, 2, kv_heads, head_dim)
        query_value = query_value.transpose(1, 2)
        key_view, value_view = fork.update(query_key, query_value)
        module = SimpleNamespace(
            layer_idx=3,
            num_key_value_groups=2,
            scaling=head_dim**-0.5,
            is_causal=True,
        )
        from qcomem_paged_attention import paged_attention_forward

        output, _ = paged_attention_forward(
            module,
            query,
            key_view,
            value_view,
            None,
            scaling=module.scaling,
        )
        output.square().mean().backward()
        for projection in (q_proj, k_proj, v_proj):
            gradient = projection.weight.grad
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(torch.count_nonzero(gradient).item(), 0)
        self.assertIsNotNone(document_hidden.grad)
        self.assertGreater(torch.count_nonzero(document_hidden.grad).item(), 0)
        self.assertIsNotNone(query_hidden.grad)
        self.assertGreater(torch.count_nonzero(query_hidden.grad).item(), 0)

    def test_autograd_mode_can_detach_document_but_never_append_graph(self) -> None:
        document_key = torch.randn(1, 2, 4, 3, requires_grad=True)
        document_value = torch.randn(1, 2, 4, 3, requires_grad=True)
        query_key = torch.randn(1, 2, 1, 3, requires_grad=True)
        query_value = torch.randn(1, 2, 1, 3, requires_grad=True)
        with mock.patch(
            "qcomem_paged_attention.quantize_tensor",
            side_effect=AssertionError("autograd q16 must not call quantizer"),
        ):
            layer = AutogradPagedKVLayer.from_dense_document(
                document_key,
                document_value,
                page_size=2,
                bits=16,
                preserve_document_graph=False,
            )
        self.assertTrue(
            all(
                not page.key.requires_grad and not page.value.requires_grad
                for page in layer.store.document_pages
            )
        )
        layer.update(query_key, query_value)
        self.assertTrue(
            all(
                page.key.requires_grad and page.value.requires_grad
                for page in layer.store.request_pages
            )
        )
        fork = layer.fork()
        self.assertEqual(fork.store.request_pages, [])
        self.assertIs(
            fork.store.document_pages[-1], layer.store.request_pages[-1]
        )
        loss = sum(
            page.key.sum() + page.value.sum()
            for page in layer.store.request_pages
        )
        loss.backward()
        self.assertIsNone(document_key.grad)
        self.assertIsNone(document_value.grad)
        self.assertIsNotNone(query_key.grad)
        self.assertIsNotNone(query_value.grad)
        with self.assertRaisesRegex(
            Qwen35PagedIntegrationError, "only dense bits=None or bits=16"
        ):
            AutogradPagedKVLayer.from_dense_document(
                document_key,
                document_value,
                page_size=2,
                bits=4,
                preserve_document_graph=True,
            )

    def test_training_conversion_retains_live_document_graph(self) -> None:
        cache = make_cache(self.backbone)
        index = self.plan.full_attention_layer_indices[0]
        producer = torch.nn.Parameter(torch.tensor(2.0))
        cache.layers[index].keys = cache.layers[index].keys * producer
        cache.layers[index].values = cache.layers[index].values * producer
        conversion = convert_all_planned_cache_layers_for_training(
            cache,
            self.plan,
            page_size=2,
            bits=None,
            preserve_document_graph=True,
        )
        self.assertTrue(conversion.preserve_autograd)
        self.assertTrue(conversion.preserve_document_graph)
        self.assertTrue(
            isinstance(cache.layers[index], AutogradPagedKVLayer)
        )
        loss = sum(
            page.key.sum() + page.value.sum()
            for page in cache.layers[index].store.document_pages
        )
        loss.backward()
        self.assertIsNotNone(producer.grad)
        self.assertNotEqual(float(producer.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
