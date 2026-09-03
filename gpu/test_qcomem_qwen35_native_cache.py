from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from qcomem_qwen35_native_cache import (
    NativeFunctionalCacheError,
    functional_linear_cache_telemetry,
    install_native_functional_linear_cache,
)


class TinyLinearLayer:
    def __init__(self) -> None:
        self.conv_states = {}
        self.recurrent_states = {}
        self.is_conv_states_initialized = {0: False}
        self.is_recurrent_states_initialized = {0: False}
        self.has_previous_state = {0: False}
        self.conv_kernel_size = {}
        self.record_past = False

    def lazy_initialization(
        self,
        *,
        conv_states=None,
        recurrent_states=None,
        state_idx=0,
        conv_kernel_size=None,
    ):
        if conv_states is not None:
            width = conv_states.shape[-1] if conv_kernel_size is None else conv_kernel_size
            self.conv_kernel_size[state_idx] = width
            self.conv_states[state_idx] = torch.zeros(
                *conv_states.shape[:-1], width, dtype=conv_states.dtype
            )
            self.is_conv_states_initialized[state_idx] = True
        if recurrent_states is not None:
            self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
            self.is_recurrent_states_initialized[state_idx] = True


class NativeFunctionalCacheTest(unittest.TestCase):
    def make_cache(self):
        return SimpleNamespace(
            layers=[TinyLinearLayer(), SimpleNamespace(), TinyLinearLayer()]
        )

    def test_rebind_preserves_document_graph_and_backward(self) -> None:
        cache = self.make_cache()
        config = SimpleNamespace(
            layer_types=["linear_attention", "full_attention", "linear_attention"]
        )
        install = install_native_functional_linear_cache(cache, config)
        layer = cache.layers[0]
        document = torch.randn(1, 4, 5, requires_grad=True)
        full_document = layer.update_conv_state(
            document, conv_kernel_size=3
        )
        document_state = layer.conv_states[0]
        document_version = document_state._version
        query = torch.randn(1, 4, 2, requires_grad=True)
        full_query = layer.update_conv_state(query)
        self.assertEqual(document_state._version, document_version)
        self.assertIsNot(layer.conv_states[0], document_state)
        (full_document.sum() + full_query.square().sum()).backward()
        self.assertGreater(float(document.grad.abs().max()), 0)
        self.assertGreater(float(query.grad.abs().max()), 0)
        telemetry = functional_linear_cache_telemetry(cache, install)
        self.assertTrue(telemetry["all_linear_layers_intercepted"])
        self.assertFalse(telemetry["mutable_copy_updates_used"])

    def test_recurrent_rebind_does_not_increment_old_version(self) -> None:
        cache = self.make_cache()
        config = SimpleNamespace(
            layer_types=["linear_attention", "full_attention", "linear_attention"]
        )
        install_native_functional_linear_cache(cache, config)
        layer = cache.layers[2]
        document = torch.randn(1, 2, 3, 4, requires_grad=True)
        layer.update_recurrent_state(document)
        document_state = layer.recurrent_states[0]
        version = document_state._version
        query_state = document * 0.5 + 1
        result = layer.update_recurrent_state(query_state)
        self.assertIs(result, query_state)
        self.assertEqual(document_state._version, version)
        result.sum().backward()
        self.assertGreater(float(document.grad.abs().max()), 0)

    def test_install_is_all_or_nothing_and_fail_closed(self) -> None:
        cache = self.make_cache()
        del cache.layers[2].conv_states
        config = SimpleNamespace(
            layer_types=["linear_attention", "full_attention", "linear_attention"]
        )
        with self.assertRaisesRegex(NativeFunctionalCacheError, "conv_states"):
            install_native_functional_linear_cache(cache, config)
        self.assertFalse(hasattr(cache.layers[0], "_qcomem_update_mode"))

    def test_hybrid_shape_and_types_are_required(self) -> None:
        cache = SimpleNamespace(layers=[TinyLinearLayer()])
        with self.assertRaisesRegex(NativeFunctionalCacheError, "both linear and full"):
            install_native_functional_linear_cache(
                cache, SimpleNamespace(layer_types=["linear_attention"])
            )
        with self.assertRaisesRegex(NativeFunctionalCacheError, "match"):
            install_native_functional_linear_cache(
                cache, SimpleNamespace(layer_types=[])
            )


if __name__ == "__main__":
    unittest.main()
