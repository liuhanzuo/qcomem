from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from qcomem_qwen35_functional_stack import (
    Qwen35FunctionalDocumentState,
    Qwen35FunctionalRequestState,
    Qwen35NativeFunctionalState,
    continue_qwen35_functional,
    native_qwen35_functional_forward,
    prefill_qwen35_functional_document,
    qwen35_functional_logits,
)
from qcomem_qwen35_gdn_functional import (
    AUTOGRAD_PRESERVING,
    INFERENCE_DETACHED,
)
from qcomem_qwen35_paged_integration import AutogradPagedKVLayer
from qcomem_qwen35_native_cache import install_native_functional_linear_cache
from test_qcomem_qwen35_gdn_functional import TinyLinearDecoderLayer


class TinyFullAttention(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = 3
        self.num_key_value_groups = 2
        self.scaling = self.head_dim**-0.5
        self.is_causal = True
        self.q_proj = nn.Linear(12, 4 * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(12, 2 * self.head_dim, bias=False)
        self.v_proj = nn.Linear(12, 2 * self.head_dim, bias=False)
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        self.o_proj = nn.Linear(4 * self.head_dim, 12, bias=False)

    # The integration audit intentionally checks this real Qwen caller surface.
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_values=None,
        **kwargs,
    ):
        del position_embeddings, attention_mask, past_key_values, kwargs
        return hidden_states, None


class TinyFullDecoder(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = 12
        self.block_type = "full_attention"
        self.input_layernorm = nn.LayerNorm(12)
        self.post_attention_layernorm = nn.LayerNorm(12)
        self.self_attn = TinyFullAttention(config, layer_idx)
        self.mlp = nn.Linear(12, 12, bias=False)


class TinyRotary(nn.Module):
    def forward(self, hidden, position_ids):
        batch = hidden.shape[0]
        length = hidden.shape[1]
        cos = torch.ones(batch, length, 3, dtype=hidden.dtype, device=hidden.device)
        sin = torch.zeros_like(cos)
        self.last_position_ids = position_ids
        return cos, sin


class TinyConfig:
    model_type = "qwen3_5_moe_text"
    num_hidden_layers = 4
    hidden_size = 12
    layer_types = [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "full_attention",
    ]
    _attn_implementation = "eager"


class TinyFunctionalModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyConfig()
        self.embed_tokens = nn.Embedding(64, 12)
        self.layers = nn.ModuleList(
            [
                TinyLinearDecoderLayer(0),
                TinyLinearDecoderLayer(1, moe_tuple=True),
                TinyLinearDecoderLayer(2),
                TinyFullDecoder(self.config, 3),
            ]
        )
        self.norm = nn.LayerNorm(12)
        self.rotary_emb = TinyRotary()
        self.lm_head = nn.Linear(12, 64, bias=False)


def page_tensors(layer):
    for page in layer.store.pages:
        for payload in (page.key, page.value):
            if isinstance(payload, torch.Tensor):
                yield payload
            else:
                yield payload.data


class Qwen35FunctionalStackTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260813)
        self.model = TinyFunctionalModel()
        self.document = torch.tensor([[1, 2, 3, 4]])
        self.query = torch.tensor([[11, 12, 13]])

    def test_all_layer_prefill_and_continuation_preserve_document_graph(self) -> None:
        document_pass = prefill_qwen35_functional_document(
            self.model,
            self.document,
            gradient_semantics=AUTOGRAD_PRESERVING,
            training_mode=True,
            attention_bits=16,
            page_size=2,
            append_page_size=2,
        )
        self.assertIsInstance(document_pass.state, Qwen35FunctionalDocumentState)
        document = document_pass.state
        self.assertEqual(
            document_pass.telemetry["coverage"]["observed_linear_layer_count"],
            3,
        )
        self.assertEqual(
            document_pass.telemetry["coverage"][
                "observed_full_attention_layer_count"
            ],
            1,
        )
        full = document.full_layers[3]
        self.assertIsInstance(full, AutogradPagedKVLayer)
        self.assertTrue(all(tensor.requires_grad for tensor in page_tensors(full)))
        document_pointers = {tensor.data_ptr() for tensor in page_tensors(full)}

        query_pass = continue_qwen35_functional(
            self.model, self.query, document
        )
        self.assertIsInstance(query_pass.state, Qwen35FunctionalRequestState)
        request = query_pass.state
        self.assertEqual(request.current_length, 7)
        self.assertEqual(
            query_pass.telemetry["coverage"]["observed_linear_layer_count"],
            3,
        )
        self.assertEqual(
            query_pass.telemetry["coverage"][
                "observed_full_attention_layer_count"
            ],
            1,
        )
        self.assertGreater(
            query_pass.telemetry["memory"]["query_private_nbytes"], 0
        )
        request_full = request.full_layers[3]
        self.assertTrue(
            document_pointers.issubset(
                {tensor.data_ptr() for tensor in page_tensors(request_full)}
            )
        )
        self.assertTrue(
            all(
                tensor.requires_grad
                for page in request_full.store.request_pages
                for tensor in (page.key, page.value)
            )
        )

        loss = qwen35_functional_logits(
            self.model, query_pass, last_token_only=True
        ).float().square().mean()
        loss.backward()
        for name, parameter in self.model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        document.assert_unchanged()

    def test_detached_document_keeps_query_kv_and_all_layers_differentiable(self) -> None:
        document_pass = prefill_qwen35_functional_document(
            self.model,
            self.document,
            gradient_semantics=INFERENCE_DETACHED,
            training_mode=True,
            attention_bits=None,
            page_size=2,
            append_page_size=1,
        )
        document = document_pass.state
        self.assertIsInstance(document, Qwen35FunctionalDocumentState)
        self.assertTrue(
            all(
                not tensor.requires_grad
                for tensor in page_tensors(document.full_layers[3])
            )
        )
        self.assertTrue(
            all(not base.conv_state.requires_grad for base in document.gdn_bases.values())
        )

        self.model.zero_grad(set_to_none=True)
        query_pass = continue_qwen35_functional(self.model, self.query, document)
        request = query_pass.state
        self.assertIsInstance(request, Qwen35FunctionalRequestState)
        request_pages = request.full_layers[3].store.request_pages
        self.assertTrue(request_pages)
        self.assertTrue(
            all(
                tensor.requires_grad
                for page in request_pages
                for tensor in (page.key, page.value)
            )
        )
        logits = qwen35_functional_logits(self.model, query_pass)
        logits.sum().backward()
        for index, layer in enumerate(self.model.layers):
            grads = [parameter.grad for parameter in layer.parameters()]
            self.assertTrue(grads, index)
            self.assertTrue(all(gradient is not None for gradient in grads), index)
            self.assertTrue(
                all(torch.isfinite(gradient).all() for gradient in grads), index
            )
        document.assert_unchanged()

    def test_continuation_is_out_of_place_and_old_request_remains_valid(self) -> None:
        document = prefill_qwen35_functional_document(
            self.model,
            self.document,
            gradient_semantics=INFERENCE_DETACHED,
            training_mode=True,
            page_size=2,
        ).state
        first = continue_qwen35_functional(
            self.model, self.query[:, :1], document
        ).state
        self.assertIsInstance(first, Qwen35FunctionalRequestState)
        first_lengths = {
            index: layer.get_seq_length() for index, layer in first.full_layers.items()
        }
        second = continue_qwen35_functional(
            self.model, self.query[:, 1:2], first
        ).state
        self.assertIsInstance(second, Qwen35FunctionalRequestState)
        self.assertEqual(
            first_lengths,
            {index: layer.get_seq_length() for index, layer in first.full_layers.items()},
        )
        self.assertEqual(second.current_length, first.current_length + 1)
        document.assert_unchanged()

    def test_native_stack_requires_every_linear_rebind_and_full_extension(self) -> None:
        model = TinyFunctionalModel()
        linear_layers = []
        full_layers = []
        for kind in model.config.layer_types:
            if kind == "linear_attention":
                layer = SimpleNamespace(
                    conv_states={0: None},
                    recurrent_states={0: None},
                    is_conv_states_initialized={0: False},
                    is_recurrent_states_initialized={0: False},
                    has_previous_state={0: False},
                    conv_kernel_size={0: None},
                    record_past=False,
                )
                layer.lazy_initialization = lambda **kwargs: None
                linear_layers.append(layer)
                full_layers.append(None)
            else:
                layer = SimpleNamespace(
                    keys=torch.empty(1, 2, 0, 3),
                    values=torch.empty(1, 2, 0, 3),
                )
                layer.get_seq_length = lambda layer=layer: int(
                    layer.keys.shape[-2]
                )
                full_layers.append(layer)
                linear_layers.append(None)
        cache = SimpleNamespace(
            layers=[
                linear_layers[index]
                if kind == "linear_attention"
                else full_layers[index]
                for index, kind in enumerate(model.config.layer_types)
            ]
        )
        # Tiny fake layers only need the installed marker for this integration
        # test; state tensors are assigned by the fake real-backbone caller.
        install = install_native_functional_linear_cache(cache, model.config)
        state = Qwen35NativeFunctionalState(
            plan=prefill_qwen35_functional_document(
                model,
                self.document,
                gradient_semantics=INFERENCE_DETACHED,
                training_mode=True,
            ).state.plan,
            cache=cache,
            install=install,
        )

        def fake_backbone_forward(*, input_ids, past_key_values, use_cache):
            self.assertTrue(use_cache)
            length = input_ids.shape[1]
            for index in state.plan.linear_layer_indices:
                layer = past_key_values.layers[index]
                layer.conv_states[0] = torch.randn(1, 24, 4, requires_grad=True)
                layer.recurrent_states[0] = torch.randn(
                    1, 4, 3, 3, requires_grad=True
                )
            for index in state.plan.full_attention_layer_indices:
                layer = past_key_values.layers[index]
                new = torch.randn(1, 2, length, 3, requires_grad=True)
                layer.keys = torch.cat((layer.keys, new), dim=-2)
                layer.values = torch.cat((layer.values, new + 0.2), dim=-2)
            hidden = model.embed_tokens(input_ids)
            return SimpleNamespace(
                last_hidden_state=hidden,
                past_key_values=past_key_values,
            )

        with mock.patch(
            "qcomem_qwen35_functional_stack.resolve_qwen35_text_backbone",
            return_value=SimpleNamespace(
                layers=model.layers,
                config=model.config,
                embed_tokens=model.embed_tokens,
                norm=model.norm,
                rotary_emb=model.rotary_emb,
                __call__=fake_backbone_forward,
            ),
        ):
            # SimpleNamespace is not callable; use a callable facade.
            class Facade:
                layers = model.layers
                config = model.config
                embed_tokens = model.embed_tokens
                norm = model.norm
                rotary_emb = model.rotary_emb

                def __call__(self, **kwargs):
                    return fake_backbone_forward(**kwargs)

            with mock.patch(
                "qcomem_qwen35_functional_stack.resolve_qwen35_text_backbone",
                return_value=Facade(),
            ):
                result = native_qwen35_functional_forward(
                    model, self.query[:, :2], state
                )
        self.assertTrue(result.telemetry["verified"])
        self.assertEqual(result.telemetry["observed_linear_layer_count"], 3)
        self.assertEqual(
            result.telemetry["observed_full_attention_layer_count"], 1
        )
        self.assertFalse(result.telemetry["mutable_linear_copy_updates_used"])
        self.assertEqual(result.state.current_length, 2)


if __name__ == "__main__":
    unittest.main()
