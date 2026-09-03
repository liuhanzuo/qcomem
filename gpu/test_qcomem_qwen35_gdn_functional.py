from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from qcomem_qwen35_gdn_functional import (
    AUTOGRAD_PRESERVING,
    INFERENCE_DETACHED,
    GDNContractError,
    ImmutableGDNBase,
    QueryLocalGDNState,
    audit_qwen35_gdn_module,
    audit_qwen35_gdn_dispatch_plan,
    dispatch_qwen35_decoder_layer,
    functional_qwen35_linear_decoder_layer_forward,
    functional_qwen35_gdn_forward,
    functional_qwen35_gdn_prefill,
    immutable_base_from_transformers_cache,
    zero_gdn_base,
)


class TinyRMSNormGated(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, hidden: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden.dtype
        hidden = hidden.float()
        hidden = hidden * torch.rsqrt(hidden.square().mean(-1, keepdim=True) + self.eps)
        hidden = self.weight * hidden.to(input_dtype)
        return (hidden * F.silu(gate.float())).to(input_dtype)


class TinyQwen35GatedDeltaNet(nn.Module):
    """The audited public Qwen3.5 GDN surface at tiny but real layouts."""

    def __init__(self, layer_idx: int = 0) -> None:
        super().__init__()
        self.hidden_size = 12
        self.num_v_heads = 4
        self.num_k_heads = 2
        self.head_k_dim = 3
        self.head_v_dim = 3
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = self.num_v_heads * self.head_v_dim
        self.conv_dim = 2 * self.key_dim + self.value_dim
        self.conv_kernel_size = 4
        self.layer_idx = layer_idx
        self.layer_type = "linear_attention"
        self.activation = "silu"

        self.conv1d = nn.Conv1d(
            self.conv_dim,
            self.conv_dim,
            self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
            bias=False,
        )
        self.dt_bias = nn.Parameter(torch.randn(self.num_v_heads) * 0.1)
        self.A_log = nn.Parameter(torch.randn(self.num_v_heads) * 0.1)
        self.in_proj_qkv = nn.Linear(
            self.hidden_size, self.conv_dim, bias=False
        )
        self.in_proj_z = nn.Linear(
            self.hidden_size, self.value_dim, bias=False
        )
        self.in_proj_b = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )
        self.in_proj_a = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )
        self.norm = TinyRMSNormGated(self.head_v_dim)
        self.out_proj = nn.Linear(
            self.value_dim, self.hidden_size, bias=False
        )


class TinyMoeMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False)

    def forward(self, hidden: torch.Tensor):
        router_logits = hidden.mean(dim=-1)
        return self.projection(F.silu(hidden)), router_logits


class TinyLinearDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int, *, moe_tuple: bool = False) -> None:
        super().__init__()
        self.hidden_size = 12
        self.block_type = "linear_attention"
        self.linear_attn = TinyQwen35GatedDeltaNet(layer_idx)
        self.input_layernorm = nn.LayerNorm(self.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size)
        self.mlp = (
            TinyMoeMLP(self.hidden_size)
            if moe_tuple
            else nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        )


class TinyFullDecoderLayer(nn.Module):
    def __init__(self, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = 12
        self.block_type = "full_attention"
        self.input_layernorm = nn.LayerNorm(self.hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(self.hidden_size)
        self.self_attn = nn.Identity()
        self.self_attn.layer_idx = layer_idx
        self.mlp = nn.Linear(self.hidden_size, self.hidden_size, bias=False)


def dense_qwen35_reference(
    module: TinyQwen35GatedDeltaNet,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent no-cache Qwen prefill equations plus final states."""

    batch, length, _ = hidden.shape
    raw_mixed = module.in_proj_qkv(hidden).transpose(1, 2)
    weight = module.conv1d.weight.squeeze(1)
    mixed = F.conv1d(
        raw_mixed.to(weight.dtype),
        weight.unsqueeze(1),
        module.conv1d.bias,
        padding=module.conv_kernel_size - 1,
        groups=module.conv_dim,
    )[:, :, :length]
    mixed = F.silu(mixed).to(raw_mixed.dtype).transpose(1, 2)
    query, key, value = torch.split(
        mixed,
        (module.key_dim, module.key_dim, module.value_dim),
        dim=-1,
    )
    query = query.reshape(
        batch, length, module.num_k_heads, module.head_k_dim
    )
    key = key.reshape(
        batch, length, module.num_k_heads, module.head_k_dim
    )
    value = value.reshape(
        batch, length, module.num_v_heads, module.head_v_dim
    )
    query = query * torch.rsqrt(
        query.square().sum(dim=-1, keepdim=True) + 1e-6
    )
    key = key * torch.rsqrt(key.square().sum(dim=-1, keepdim=True) + 1e-6)
    repeat = module.num_v_heads // module.num_k_heads
    query = query.repeat_interleave(repeat, dim=2)
    key = key.repeat_interleave(repeat, dim=2)
    beta = module.in_proj_b(hidden).sigmoid()
    a = module.in_proj_a(hidden)
    g = -module.A_log.float().exp() * F.softplus(a.float() + module.dt_bias)

    query, key, value, beta, g = (
        tensor.transpose(1, 2).contiguous().float()
        for tensor in (query, key, value, beta, g)
    )
    query = query * (module.head_k_dim**-0.5)
    recurrent = torch.zeros(
        batch,
        module.num_v_heads,
        module.head_k_dim,
        module.head_v_dim,
        dtype=torch.float32,
        device=hidden.device,
    )
    outputs = []
    for position in range(length):
        query_t = query[:, :, position]
        key_t = key[:, :, position]
        value_t = value[:, :, position]
        recurrent = recurrent * g[:, :, position].exp()[:, :, None, None]
        memory = (recurrent * key_t.unsqueeze(-1)).sum(dim=-2)
        delta = (value_t - memory) * beta[:, :, position].unsqueeze(-1)
        recurrent = recurrent + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append((recurrent * query_t.unsqueeze(-1)).sum(dim=-2))
    core = torch.stack(outputs, dim=2).transpose(1, 2).contiguous()
    core = core.to(hidden.dtype).reshape(-1, module.head_v_dim)
    gate = module.in_proj_z(hidden).reshape(-1, module.head_v_dim)
    output = module.out_proj(
        module.norm(core, gate).reshape(batch, length, module.value_dim)
    )

    zeros = raw_mixed.new_zeros(
        batch, module.conv_dim, module.conv_kernel_size
    )
    conv_state = torch.cat((zeros, raw_mixed), dim=-1)[
        :, :, -module.conv_kernel_size :
    ].clone()
    return output, conv_state, recurrent


def dense_decoder_reference(
    decoder_layer: TinyLinearDecoderLayer,
    hidden: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    residual = hidden
    normalized = decoder_layer.input_layernorm(hidden)
    mixed, conv_state, recurrent_state = dense_qwen35_reference(
        decoder_layer.linear_attn, normalized
    )
    hidden = residual + mixed
    residual = hidden
    mlp_output = decoder_layer.mlp(
        decoder_layer.post_attention_layernorm(hidden)
    )
    if isinstance(mlp_output, tuple):
        mlp_output = mlp_output[0]
    return residual + mlp_output, conv_state, recurrent_state


def storage_pointer(tensor: torch.Tensor) -> int:
    return tensor.untyped_storage().data_ptr()


class Qwen35FunctionalGDNTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(731)
        self.module = TinyQwen35GatedDeltaNet()
        self.document = torch.randn(2, 5, self.module.hidden_size)
        self.query = torch.randn(2, 3, self.module.hidden_size)

    def test_same_module_dense_vs_document_base_query_forward_and_state(self) -> None:
        merged = torch.cat((self.document, self.query), dim=1)
        dense_output, dense_conv, dense_recurrent = dense_qwen35_reference(
            self.module, merged
        )
        _, document_base = functional_qwen35_gdn_prefill(
            self.module,
            self.document,
            gradient_semantics=AUTOGRAD_PRESERVING,
        )
        query_output, query_state = functional_qwen35_gdn_forward(
            self.module,
            self.query,
            QueryLocalGDNState.from_base(document_base),
        )

        torch.testing.assert_close(
            query_output, dense_output[:, self.document.shape[1] :], atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            query_state.conv_state, dense_conv, atol=2e-6, rtol=2e-6
        )
        torch.testing.assert_close(
            query_state.recurrent_state,
            dense_recurrent,
            atol=3e-6,
            rtol=3e-6,
        )

    def test_query_only_loss_matches_dense_gradients_and_reaches_document_base(self) -> None:
        dense_document = self.document.clone().requires_grad_(True)
        dense_query = self.query.clone().requires_grad_(True)
        dense_output, _, _ = dense_qwen35_reference(
            self.module, torch.cat((dense_document, dense_query), dim=1)
        )
        weights = torch.linspace(
            0.3, 1.2, dense_query.numel(), dtype=dense_output.dtype
        ).reshape_as(dense_query)
        (dense_output[:, dense_document.shape[1] :] * weights).sum().backward()
        dense_parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in self.module.named_parameters()
        }
        dense_document_grad = dense_document.grad.detach().clone()
        dense_query_grad = dense_query.grad.detach().clone()

        self.module.zero_grad(set_to_none=True)
        split_document = self.document.clone().requires_grad_(True)
        split_query = self.query.clone().requires_grad_(True)
        _, document_base = functional_qwen35_gdn_prefill(
            self.module,
            split_document,
            gradient_semantics=AUTOGRAD_PRESERVING,
        )
        self.assertTrue(document_base.preserves_autograd)
        self.assertIsNotNone(document_base.conv_state.grad_fn)
        self.assertIsNotNone(document_base.recurrent_state.grad_fn)
        document_base.conv_state.retain_grad()
        document_base.recurrent_state.retain_grad()
        split_output, _ = functional_qwen35_gdn_forward(
            self.module, split_query, document_base
        )
        (split_output * weights).sum().backward()

        self.assertIsNotNone(split_document.grad)
        self.assertIsNotNone(split_query.grad)
        self.assertGreater(float(split_document.grad.abs().max()), 0.0)
        self.assertGreater(float(split_query.grad.abs().max()), 0.0)
        self.assertIsNotNone(document_base.conv_state.grad)
        self.assertIsNotNone(document_base.recurrent_state.grad)
        self.assertGreater(float(document_base.conv_state.grad.abs().max()), 0.0)
        self.assertGreater(
            float(document_base.recurrent_state.grad.abs().max()), 0.0
        )
        torch.testing.assert_close(
            split_document.grad, dense_document_grad, atol=8e-6, rtol=8e-6
        )
        torch.testing.assert_close(
            split_query.grad, dense_query_grad, atol=8e-6, rtol=8e-6
        )
        for name, parameter in self.module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().max()), 0.0, name)
            torch.testing.assert_close(
                parameter.grad,
                dense_parameter_grads[name],
                atol=1e-5,
                rtol=1e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_inference_base_metadata_detaches_document_but_query_still_backprops(self) -> None:
        document = self.document.clone().requires_grad_(True)
        query = self.query.clone().requires_grad_(True)
        _, document_base = functional_qwen35_gdn_prefill(
            self.module,
            document,
            gradient_semantics=INFERENCE_DETACHED,
        )
        self.assertFalse(document_base.preserves_autograd)
        self.assertEqual(document_base.gradient_semantics, INFERENCE_DETACHED)
        self.assertIsNone(document_base.conv_state.grad_fn)
        self.assertIsNone(document_base.recurrent_state.grad_fn)

        output, _ = functional_qwen35_gdn_forward(
            self.module, query, document_base
        )
        output.square().mean().backward()
        self.assertIsNone(document.grad)
        self.assertIsNotNone(query.grad)
        self.assertGreater(float(query.grad.abs().max()), 0.0)
        for name, parameter in self.module.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_base_is_immutable_forks_are_isolated_and_private_memory_is_bounded(self) -> None:
        with torch.no_grad():
            _, document_base = functional_qwen35_gdn_prefill(
                self.module,
                self.document,
                gradient_semantics=INFERENCE_DETACHED,
            )
        base_values = (
            document_base.conv_state.clone(),
            document_base.recurrent_state.clone(),
        )
        base_versions = (
            document_base.conv_state._version,
            document_base.recurrent_state._version,
        )
        initial = QueryLocalGDNState.from_base(document_base)
        initial_memory = initial.memory_report()
        self.assertEqual(initial_memory["shared_base_nbytes"], document_base.nbytes)
        self.assertEqual(initial_memory["query_private_nbytes"], 0)
        self.assertEqual(initial_memory["total_referenced_nbytes"], document_base.nbytes)

        _, first = functional_qwen35_gdn_forward(
            self.module, self.query[:, :1], initial
        )
        _, second = functional_qwen35_gdn_forward(
            self.module, self.query, initial
        )
        document_base.assert_unchanged()
        self.assertTrue(torch.equal(document_base.conv_state, base_values[0]))
        self.assertTrue(torch.equal(document_base.recurrent_state, base_values[1]))
        self.assertEqual(document_base.conv_state._version, base_versions[0])
        self.assertEqual(document_base.recurrent_state._version, base_versions[1])

        base_pointers = {
            storage_pointer(document_base.conv_state),
            storage_pointer(document_base.recurrent_state),
        }
        first_pointers = {
            storage_pointer(first.conv_state),
            storage_pointer(first.recurrent_state),
        }
        second_pointers = {
            storage_pointer(second.conv_state),
            storage_pointer(second.recurrent_state),
        }
        self.assertFalse(base_pointers & first_pointers)
        self.assertFalse(base_pointers & second_pointers)
        self.assertFalse(first_pointers & second_pointers)
        self.assertEqual(first.query_private_nbytes, second.query_private_nbytes)
        self.assertEqual(
            first.memory_report()["total_referenced_nbytes"],
            document_base.nbytes + first.query_private_nbytes,
        )

        first_values = (first.conv_state.clone(), first.recurrent_state.clone())
        _, continued = functional_qwen35_gdn_forward(
            self.module, self.query[:, 1:2], first
        )
        self.assertTrue(torch.equal(first.conv_state, first_values[0]))
        self.assertTrue(torch.equal(first.recurrent_state, first_values[1]))
        self.assertNotEqual(
            storage_pointer(continued.conv_state), storage_pointer(first.conv_state)
        )
        self.assertNotEqual(
            storage_pointer(continued.recurrent_state),
            storage_pointer(first.recurrent_state),
        )

    def test_transformers_cache_capture_clones_and_preserves_or_detaches_graph(self) -> None:
        spec = audit_qwen35_gdn_module(self.module)
        source_conv = torch.randn(
            2, spec.conv_dim, spec.conv_kernel_size, requires_grad=True
        )
        source_recurrent = torch.randn(
            2, *spec.recurrent_shape_tail, requires_grad=True
        )
        layer = SimpleNamespace(
            conv_states={0: source_conv},
            recurrent_states={0: source_recurrent},
            is_conv_states_initialized={0: True},
            is_recurrent_states_initialized={0: True},
            has_previous_state={0: True},
            conv_kernel_size={0: spec.conv_kernel_size},
        )
        cache = SimpleNamespace(layers=[layer])
        autograd_base = immutable_base_from_transformers_cache(
            self.module,
            cache,
            gradient_semantics=AUTOGRAD_PRESERVING,
        )
        self.assertNotEqual(
            storage_pointer(autograd_base.conv_state), storage_pointer(source_conv)
        )
        self.assertNotEqual(
            storage_pointer(autograd_base.recurrent_state),
            storage_pointer(source_recurrent),
        )
        (autograd_base.conv_state.sum() + autograd_base.recurrent_state.sum()).backward()
        self.assertIsNotNone(source_conv.grad)
        self.assertIsNotNone(source_recurrent.grad)

        detached_base = immutable_base_from_transformers_cache(
            self.module,
            cache,
            gradient_semantics=INFERENCE_DETACHED,
        )
        self.assertFalse(detached_base.conv_state.requires_grad)
        self.assertFalse(detached_base.recurrent_state.requires_grad)
        self.assertEqual(
            detached_base.source, "transformers-cache-layer:0:state:0"
        )

    def test_module_cache_and_state_shape_mismatches_fail_closed(self) -> None:
        bad_module = TinyQwen35GatedDeltaNet()
        bad_module.layer_type = "full_attention"
        with self.assertRaisesRegex(GDNContractError, "layer_type"):
            audit_qwen35_gdn_module(bad_module)

        bad_module = TinyQwen35GatedDeltaNet()
        bad_module.conv_dim += 1
        with self.assertRaisesRegex(GDNContractError, "conv_dim"):
            functional_qwen35_gdn_forward(bad_module, self.query)

        spec = audit_qwen35_gdn_module(self.module)
        malformed_base = ImmutableGDNBase(
            conv_state=torch.zeros(
                2, spec.conv_dim, spec.conv_kernel_size
            ),
            recurrent_state=torch.zeros(
                2, spec.num_v_heads, spec.head_k_dim + 1, spec.head_v_dim
            ),
            gradient_semantics=INFERENCE_DETACHED,
        )
        with self.assertRaisesRegex(GDNContractError, "recurrent state shape"):
            functional_qwen35_gdn_forward(
                self.module, self.query, malformed_base
            )

        malformed_cache = SimpleNamespace(
            layers=[SimpleNamespace(conv_states=[torch.zeros(1)])]
        )
        with self.assertRaisesRegex(GDNContractError, "must be a dict"):
            immutable_base_from_transformers_cache(
                self.module,
                malformed_cache,
                gradient_semantics=INFERENCE_DETACHED,
            )

    def test_decoder_helper_handles_moe_tuple_and_preserves_query_gradients(self) -> None:
        torch.manual_seed(991)
        decoder = TinyLinearDecoderLayer(0, moe_tuple=True)
        document = torch.randn(2, 4, 12)
        query = torch.randn(2, 3, 12)

        dense_document = document.clone().requires_grad_(True)
        dense_query = query.clone().requires_grad_(True)
        dense_output, _, _ = dense_decoder_reference(
            decoder, torch.cat((dense_document, dense_query), dim=1)
        )
        weights = torch.linspace(0.1, 1.0, query.numel()).reshape_as(query)
        (dense_output[:, document.shape[1] :] * weights).sum().backward()
        dense_parameter_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in decoder.named_parameters()
        }
        dense_document_grad = dense_document.grad.detach().clone()
        dense_query_grad = dense_query.grad.detach().clone()

        decoder.zero_grad(set_to_none=True)
        split_document = document.clone().requires_grad_(True)
        split_query = query.clone().requires_grad_(True)
        zero_base = zero_gdn_base(
            decoder.linear_attn,
            decoder.input_layernorm(split_document),
            gradient_semantics=AUTOGRAD_PRESERVING,
        )
        _, document_state, document_telemetry = (
            functional_qwen35_linear_decoder_layer_forward(
                decoder, split_document, zero_base
            )
        )
        document_base = document_state.promote_to_base(
            gradient_semantics=AUTOGRAD_PRESERVING,
            source="decoder-document-prefill",
        )
        split_output, _, query_telemetry = (
            functional_qwen35_linear_decoder_layer_forward(
                decoder, split_query, document_base
            )
        )
        self.assertEqual(document_telemetry.layer_idx, 0)
        self.assertEqual(query_telemetry.route, "functional-qwen35-gdn")
        self.assertFalse(query_telemetry.mutable_cache_used)
        self.assertFalse(query_telemetry.fallback_used)
        torch.testing.assert_close(
            split_output,
            dense_output[:, document.shape[1] :],
            atol=3e-6,
            rtol=3e-6,
        )
        with torch.autograd.set_detect_anomaly(True):
            (split_output * weights).sum().backward()
        self.assertGreater(float(split_document.grad.abs().max()), 0.0)
        self.assertGreater(float(split_query.grad.abs().max()), 0.0)
        torch.testing.assert_close(
            split_document.grad, dense_document_grad, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            split_query.grad, dense_query_grad, atol=1e-5, rtol=1e-5
        )
        for name, parameter in decoder.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(float(parameter.grad.abs().max()), 0.0, name)
            torch.testing.assert_close(
                parameter.grad,
                dense_parameter_grads[name],
                atol=2e-5,
                rtol=2e-5,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_dynamic_dispatch_plan_intercepts_every_linear_layer(self) -> None:
        layers = [
            TinyLinearDecoderLayer(0),
            TinyLinearDecoderLayer(1, moe_tuple=True),
            TinyFullDecoderLayer(2),
            TinyLinearDecoderLayer(3),
        ]
        config = SimpleNamespace(
            num_hidden_layers=4,
            hidden_size=12,
            layer_types=[
                "linear_attention",
                "linear_attention",
                "full_attention",
                "linear_attention",
            ],
        )
        plan = audit_qwen35_gdn_dispatch_plan(layers, config)
        self.assertEqual(plan.linear_layer_indices, (0, 1, 3))
        self.assertEqual(plan.full_attention_layer_indices, (2,))
        self.assertEqual(plan.linear_layer_count, 3)

        hidden = torch.randn(1, 2, 12)
        states = {
            index: zero_gdn_base(
                layers[index].linear_attn,
                hidden,
                gradient_semantics=INFERENCE_DETACHED,
            )
            for index in plan.linear_layer_indices
        }
        plan.validate_linear_states(states)
        intercepted = []
        for index in plan.linear_layer_indices:
            hidden, states[index], telemetry = dispatch_qwen35_decoder_layer(
                layers[index],
                hidden,
                states[index],
                layer_idx=index,
            )
            intercepted.append(telemetry.layer_idx)
            self.assertEqual(telemetry.route, "functional-qwen35-gdn")
            self.assertFalse(telemetry.mutable_cache_used)
            self.assertFalse(telemetry.fallback_used)
            self.assertGreater(telemetry.query_private_nbytes, 0)
        self.assertEqual(tuple(intercepted), plan.linear_layer_indices)

        with self.assertRaisesRegex(GDNContractError, "missing"):
            plan.validate_linear_states({0: states[0]})

    def test_decoder_dispatch_requires_state_and_explicit_full_attention_callback(self) -> None:
        linear = TinyLinearDecoderLayer(0)
        full = TinyFullDecoderLayer(1)
        hidden = torch.randn(1, 2, 12)
        with self.assertRaisesRegex(GDNContractError, "missing explicit"):
            dispatch_qwen35_decoder_layer(
                linear, hidden, None, layer_idx=0
            )
        with self.assertRaisesRegex(GDNContractError, "out-of-place dispatcher"):
            dispatch_qwen35_decoder_layer(
                full, hidden, object(), layer_idx=1
            )
        base = zero_gdn_base(linear.linear_attn, hidden)
        with self.assertRaisesRegex(GDNContractError, "mutable cache arguments"):
            dispatch_qwen35_decoder_layer(
                linear,
                hidden,
                base,
                layer_idx=0,
                past_key_values=object(),
            )

        calls = []

        def full_dispatch(layer, value, **kwargs):
            calls.append((layer, kwargs["layer_idx"], kwargs["layer_state"]))
            return value + 0.25, {"full": "next"}, {"route": "test-full"}

        full_state = {"full": "base"}
        output, next_state, telemetry = dispatch_qwen35_decoder_layer(
            full,
            hidden,
            full_state,
            layer_idx=1,
            full_attention_dispatch=full_dispatch,
        )
        torch.testing.assert_close(output, hidden + 0.25)
        self.assertEqual(next_state, {"full": "next"})
        self.assertEqual(telemetry, {"route": "test-full"})
        self.assertEqual(calls, [(full, 1, full_state)])

        bad_layers = [TinyLinearDecoderLayer(0), TinyFullDecoderLayer(1)]
        bad_layers[1].block_type = "linear_attention"
        config = SimpleNamespace(
            num_hidden_layers=2,
            hidden_size=12,
            layer_types=["linear_attention", "full_attention"],
        )
        with self.assertRaisesRegex(GDNContractError, "block_type"):
            audit_qwen35_gdn_dispatch_plan(bad_layers, config)


if __name__ == "__main__":
    unittest.main()
