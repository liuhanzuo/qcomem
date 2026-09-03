from __future__ import annotations

import copy
import unittest

import torch

from qcomem_deployment import (
    MemoryRecorder,
    parse_deployment_config,
    run_cow_vs_deep_clone_gate,
    run_incremental_generation,
)
from qcomem_paged import PagedForkLowerReplayState, prepare_paged_lower_state
from qcomem_torch import LowerReplayState


class FakeDynamicAttentionLayer:
    is_sliding = False

    def __init__(self, length: int = 5) -> None:
        self.keys = torch.arange(2 * length * 8, dtype=torch.float32).reshape(
            1, 2, length, 8
        )
        self.values = -self.keys.clone()
        self.is_initialized = True

    def lazy_initialization(self, keys, values) -> None:
        self.keys = torch.empty_like(keys[..., :0, :])
        self.values = torch.empty_like(values[..., :0, :])
        self.is_initialized = True

    def update(self, keys, values, *args, **kwargs):
        # Deliberately unsafe original implementation: the COW prototype must
        # replace this instance method rather than merely assuming it is safe.
        del args, kwargs
        self.keys.add_(1000)
        self.values.sub_(1000)
        self.keys = torch.cat([self.keys, keys], dim=-2)
        self.values = torch.cat([self.values, values], dim=-2)
        return self.keys, self.values


class FakeProductionAttentionLayer(FakeDynamicAttentionLayer):
    def update(self, keys, values, *args, **kwargs):
        del args, kwargs
        self.keys = torch.cat([self.keys, keys], dim=-2)
        self.values = torch.cat([self.values, values], dim=-2)
        return self.keys, self.values


class FakeLinearLayer:
    def __init__(self) -> None:
        self.conv_states = {0: torch.arange(24, dtype=torch.float32).reshape(1, 8, 3)}
        self.recurrent_states = {
            0: torch.arange(128, dtype=torch.float32).reshape(1, 2, 8, 8)
        }
        self.has_previous_state = {0: True}
        self.is_conv_states_initialized = {0: True}
        self.is_recurrent_states_initialized = {0: True}

    def update_conv_state(self, value, state_idx=0, **kwargs):
        del kwargs
        self.conv_states[state_idx].copy_(value)
        return self.conv_states[state_idx]

    def update_recurrent_state(self, value, state_idx=0, **kwargs):
        del kwargs
        self.recurrent_states[state_idx].copy_(value)
        return self.recurrent_states[state_idx]


class FakeHybridCache:
    def __init__(self, layers=None) -> None:
        self.layers = layers or [FakeDynamicAttentionLayer(), FakeLinearLayer()]

    def update(self, keys, values, layer_idx, *args, **kwargs):
        return self.layers[layer_idx].update(keys, values, *args, **kwargs)

    def update_conv_state(self, value, layer_idx, state_idx=0, **kwargs):
        return self.layers[layer_idx].update_conv_state(
            value, state_idx=state_idx, **kwargs
        )

    def update_recurrent_state(self, value, layer_idx, state_idx=0, **kwargs):
        return self.layers[layer_idx].update_recurrent_state(
            value, state_idx=state_idx, **kwargs
        )


class FakeUnknownCacheLayer:
    def __init__(self) -> None:
        self.mystery_state = torch.ones(4)


class FakeSuffixCache:
    def __init__(self) -> None:
        self.tokens = torch.empty((1, 0), dtype=torch.long)


class FakeFullState:
    def __init__(self, tokens: torch.Tensor) -> None:
        self.cache = FakeSuffixCache()
        self.cache.tokens = tokens.clone()

    @property
    def stored_nbytes(self) -> int:
        return self.cache.tokens.numel() * self.cache.tokens.element_size()

    def fork(self):
        return copy.deepcopy(self)


class FakePagedAdapter:
    vocab_size = 64

    @staticmethod
    def _logits(token: int) -> torch.Tensor:
        logits = torch.full((1, FakePagedAdapter.vocab_size), -1000.0)
        logits[0, (token + 1) % FakePagedAdapter.vocab_size] = 1.0
        return logits

    def write_full_prefix(self, tokens):
        return FakeFullState(tokens)

    def continue_full_prefix(self, state, tokens):
        state.cache.tokens = torch.cat([state.cache.tokens, tokens], dim=1)
        return self._logits(int(tokens[0, -1]))

    def continue_lower_replay(self, state, tokens):
        length = tokens.shape[1]
        values = tokens.float().reshape(1, 1, length, 1).expand(1, 2, length, 8)
        state.cache.update(values, -values, 0)
        linear = state.cache.layers[1]
        state.cache.update_conv_state(
            torch.full_like(linear.conv_states[0], float(tokens[0, -1])),
            1,
        )
        state.cache.update_recurrent_state(
            torch.full_like(linear.recurrent_states[0], float(tokens[0, -1])),
            1,
        )
        state.current_length += length
        return tokens.float().unsqueeze(-1).expand(1, length, 64)

    def make_cache(self):
        return FakeSuffixCache()

    def run_suffix_cached_last_logits(
        self, residuals, depth, cache, *, position_offset
    ):
        del depth, position_offset
        residual = torch.cat(list(residuals), dim=1)
        tokens = residual[..., 0].long()
        cache.tokens = torch.cat([cache.tokens, tokens], dim=1)
        return self._logits(int(tokens[0, -1]))


def replay_state() -> LowerReplayState:
    return LowerReplayState(
        depth=2,
        document_length=5,
        current_length=5,
        document_residual=torch.arange(5 * 64, dtype=torch.float32).reshape(1, 5, 64),
        cache=FakeHybridCache(),
    )


class PagedLowerStateTest(unittest.TestCase):
    def test_attention_and_residual_share_but_linear_state_is_private(self) -> None:
        source = replay_state()
        source_keys = source.cache.layers[0].keys.clone()
        source_values = source.cache.layers[0].values.clone()
        source_conv = source.cache.layers[1].conv_states[0].clone()
        source_recurrent = source.cache.layers[1].recurrent_states[0].clone()
        paged = prepare_paged_lower_state(source)
        self.assertTrue(paged.plan.supported)

        local = paged.fork()
        self.assertIsInstance(local, PagedForkLowerReplayState)
        self.assertEqual(local.fork_strategy_effective, "paged-cow-staging")
        self.assertEqual(
            local.cache.layers[0].keys.untyped_storage().data_ptr(),
            source.cache.layers[0].keys.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            local.document_residual.untyped_storage().data_ptr(),
            source.document_residual.untyped_storage().data_ptr(),
        )
        self.assertNotEqual(
            local.cache.layers[1].conv_states[0].untyped_storage().data_ptr(),
            source.cache.layers[1].conv_states[0].untyped_storage().data_ptr(),
        )

        new_keys = torch.full((1, 2, 2, 8), 7.0)
        new_values = torch.full((1, 2, 2, 8), 9.0)
        local.cache.update(new_keys, new_values, 0)
        local.cache.update_conv_state(torch.zeros_like(source_conv), 1)
        local.cache.update_recurrent_state(torch.zeros_like(source_recurrent), 1)

        self.assertTrue(torch.equal(source.cache.layers[0].keys, source_keys))
        self.assertTrue(torch.equal(source.cache.layers[0].values, source_values))
        self.assertTrue(torch.equal(source.cache.layers[1].conv_states[0], source_conv))
        self.assertTrue(
            torch.equal(source.cache.layers[1].recurrent_states[0], source_recurrent)
        )
        self.assertEqual(local.cache.layers[0].keys.shape[-2], 7)
        self.assertTrue(local.verify_shared_immutable()["verified"])
        after = local.memory_breakdown()
        self.assertGreater(after["private_nbytes"], local.initial_private_nbytes)

    def test_packed_source_reports_staging_and_total_resident_bytes(self) -> None:
        source = replay_state()
        packed = source.quantize(
            bits=4,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=(4, 8),
            group_size=64,
        )
        paged = prepare_paged_lower_state(packed)
        components = paged.deployment_memory_components()
        self.assertTrue(components["cow_supported"])
        self.assertEqual(components["persistent_document_nbytes"], packed.stored_nbytes)
        self.assertGreater(components["persistent_materialized_staging_nbytes"], 0)
        self.assertGreater(
            components["persistent_total_resident_nbytes"],
            components["persistent_document_nbytes"],
        )
        local = paged.fork()
        self.assertEqual(local.fork_strategy_effective, "paged-cow-staging")
        self.assertTrue(local.verify_shared_immutable()["verified"])

    def test_unknown_cache_tensor_falls_back_without_claiming_cow(self) -> None:
        source = LowerReplayState(
            depth=1,
            document_length=2,
            current_length=2,
            document_residual=torch.zeros(1, 2, 64),
            cache=FakeHybridCache(layers=[FakeUnknownCacheLayer()]),
        )
        paged = prepare_paged_lower_state(source)
        self.assertFalse(paged.plan.supported)
        local = paged.fork()
        self.assertEqual(local.fork_strategy_effective, "deep-clone-fallback")
        self.assertIn("unclassified", local.fallback_reason)
        self.assertNotEqual(
            local.cache.layers[0].mystery_state.untyped_storage().data_ptr(),
            source.cache.layers[0].mystery_state.untyped_storage().data_ptr(),
        )

    def test_mutation_audit_fails_closed(self) -> None:
        local = prepare_paged_lower_state(replay_state()).fork()
        local.document_residual.add_(1)
        with self.assertRaisesRegex(RuntimeError, "immutable-state audit failed"):
            local.verify_shared_immutable()

    def test_direct_q16_gate_pairs_same_source_and_requires_bitwise_logits(self) -> None:
        source = LowerReplayState(
            depth=2,
            document_length=5,
            current_length=5,
            document_residual=torch.arange(5 * 64, dtype=torch.float32).reshape(
                1, 5, 64
            ),
            cache=FakeHybridCache(
                layers=[FakeProductionAttentionLayer(), FakeLinearLayer()]
            ),
        ).quantize(
            bits=16,
            attention_bits=16,
            linear_bits=16,
            group_size=64,
        )
        adapter = FakePagedAdapter()
        document = torch.arange(5).unsqueeze(0)
        query = torch.tensor([[8, 9]])
        full_prefix_trace = run_incremental_generation(
            adapter,
            parse_deployment_config("full-prefix-q16"),
            document,
            query,
            adapter.write_full_prefix(document),
            max_new_tokens=3,
            eos_token_ids=set(),
            recorder=MemoryRecorder(),
            collect_logits=True,
        )
        result = run_cow_vs_deep_clone_gate(
            adapter,
            parse_deployment_config("qcomem-d2-r16-a16-l16"),
            document,
            query,
            source,
            max_new_tokens=3,
            eos_token_ids=set(),
            full_prefix_trace=full_prefix_trace,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["same_persistent_source"])
        self.assertGreater(
            result["persistent_memory"]["persistent_total_resident_nbytes"],
            result["persistent_memory"]["persistent_document_nbytes"],
        )
        self.assertEqual(result["strategy_effective"], "paged-cow-staging")
        self.assertTrue(result["source_after_eager"]["verified"])
        self.assertTrue(result["source_after_cow"]["verified"])
        self.assertTrue(result["comparison"]["token_sequence_exact"])
        self.assertTrue(result["comparison"]["logits_bitwise_exact"])
        self.assertTrue(result["incremental_three_way_token_exact"])
        self.assertTrue(result["comparisons"]["full_prefix_vs_eager_q16"]["passed"])
        self.assertTrue(result["comparisons"]["full_prefix_vs_cow_q16"]["passed"])
        self.assertTrue(
            all(
                step["max_abs_logit_error"] == 0
                and step["relative_l2_logit_error"] == 0
                for step in result["comparison"]["per_step"]
            )
        )


if __name__ == "__main__":
    unittest.main()
