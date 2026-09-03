from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from qcomem_torch import (
    TorchSplitCausalLM,
    active_cache_layer_indices,
    cache_nbytes,
    greedy_generate_dense,
    greedy_generate_full_prefix,
    greedy_generate_oracle,
    greedy_generate_replay,
    quantize_residual,
    quantize_tensor,
    quantize_transformers_cache,
)
from run_replay_diagnostic import parse_config
from run_capacity_scaling import parse_capacity_config


@dataclass
class FakeReplayState:
    depth: int
    document_length: int
    current_length: int
    document_residual: torch.Tensor
    cache: object

    def fork(self):
        return FakeReplayState(
            self.depth,
            self.document_length,
            self.current_length,
            self.document_residual.clone(),
            object(),
        )


class ChunkRecordingAdapter:
    def __init__(self) -> None:
        self.suffix_calls: list[tuple[int, int]] = []

    def continue_lower_replay(self, state, tokens):
        state.current_length += tokens.shape[1]
        return tokens.unsqueeze(-1).float()

    def make_cache(self):
        return object()

    def run_suffix_cached_last_logits(
        self, residuals, depth, cache, *, position_offset
    ):
        del depth, cache
        values = torch.cat(list(residuals), dim=1)
        self.suffix_calls.append((values.shape[1], position_offset))
        logits = torch.full((1, 64), -1000.0)
        logits[0, (int(values[0, -1, 0]) + 1) % 64] = 1.0
        return logits


class QuantizationTest(unittest.TestCase):
    def test_replay_preserves_document_query_suffix_chunk_boundary(self) -> None:
        adapter = ChunkRecordingAdapter()
        state = FakeReplayState(
            depth=2,
            document_length=4,
            current_length=4,
            document_residual=torch.arange(4).view(1, 4, 1).float(),
            cache=object(),
        )
        generated = greedy_generate_replay(
            adapter,
            state,
            torch.tensor([[8, 9]]),
            max_new_tokens=2,
            eos_token_ids=set(),
        )
        self.assertEqual(generated, [10, 11])
        self.assertEqual(adapter.suffix_calls, [(4, 0), (2, 4), (1, 6)])

    def test_replay_config_parser(self) -> None:
        self.assertEqual(
            parse_config("dense"), ("dense", None, None, None, None)
        )
        self.assertEqual(
            parse_config("prefix"), ("prefix", None, None, None, None)
        )
        self.assertEqual(
            parse_config("replay-d7"), ("replay", 7, None, None, None)
        )
        self.assertEqual(
            parse_config("replay-d13-q4"), ("replay", 13, 4, None, None)
        )
        self.assertEqual(
            parse_config("replay-d7-r2-a4-l8"),
            ("replay", 7, 2, 4, 8),
        )
        self.assertEqual(
            parse_capacity_config("replay-d10-r4-a4-l4"),
            ("replay", 10, 4, 4, 4),
        )

    def test_real_pack_size_and_error(self) -> None:
        values = torch.sin(torch.arange(2 * 5 * 128).reshape(2, 5, 128) / 17).to(
            torch.bfloat16
        )
        stores = {
            bits: quantize_residual(values, bits=bits, group_size=64)
            for bits in (2, 4, 8, 16)
        }
        errors = {}
        for bits, store in stores.items():
            restored = store.dequantize()
            self.assertEqual(restored.shape, values.shape)
            errors[bits] = torch.mean(
                torch.square(restored.float() - values.float())
            ).sqrt()
        self.assertLess(stores[2].nbytes, stores[4].nbytes)
        self.assertLess(stores[4].nbytes, stores[8].nbytes)
        self.assertLess(stores[8].nbytes, stores[16].nbytes)
        self.assertLess(errors[8], errors[4])
        self.assertLess(errors[4], errors[2])

    def test_arbitrary_tensor_and_mixed_cache_packing(self) -> None:
        values = torch.cos(torch.arange(17 * 67).reshape(17, 67) / 23)
        stores = {
            bits: quantize_tensor(values, bits=bits, group_size=64)
            for bits in (2, 4, 8, 16)
        }
        for store in stores.values():
            self.assertEqual(store.dequantize().shape, values.shape)
        self.assertLess(stores[2].nbytes, stores[4].nbytes)
        self.assertLess(stores[4].nbytes, stores[8].nbytes)
        self.assertLess(stores[8].nbytes, stores[16].nbytes)

        attention = SimpleNamespace(
            keys=torch.randn(1, 2, 19, 16),
            values=torch.randn(1, 2, 19, 16),
        )
        linear = SimpleNamespace(
            conv_states=[torch.randn(1, 48, 3)],
            recurrent_states=[torch.randn(1, 4, 16, 16)],
        )
        cache = SimpleNamespace(layers=[attention, linear])
        dense_bytes = cache_nbytes(cache)
        packed = quantize_transformers_cache(
            cache, attention_bits=8, linear_bits=4, group_size=64
        )
        self.assertLess(packed.nbytes, dense_bytes)
        restored = packed.dequantize()
        self.assertEqual(restored.layers[0].keys.shape, attention.keys.shape)
        self.assertEqual(
            restored.layers[1].recurrent_states[0].shape,
            linear.recurrent_states[0].shape,
        )
        self.assertGreater(
            packed.error_sums["attention"]["squared_error_sum"], 0
        )
        self.assertGreater(packed.error_sums["linear"]["squared_error_sum"], 0)

    def test_q16_packed_cache_forks_own_independent_mutable_storage(self) -> None:
        attention = SimpleNamespace(
            keys=torch.randn(1, 2, 5, 8, dtype=torch.bfloat16),
            values=torch.randn(1, 2, 5, 8, dtype=torch.bfloat16),
        )
        linear = SimpleNamespace(
            conv_states={
                0: torch.randn(1, 16, 3, dtype=torch.bfloat16)
            },
            recurrent_states={
                0: torch.randn(1, 2, 8, 8, dtype=torch.bfloat16)
            },
        )
        packed = quantize_transformers_cache(
            SimpleNamespace(layers=[attention, linear]),
            attention_bits=16,
            linear_bits=16,
            group_size=64,
        )
        fork_a = packed.dequantize()
        fork_b = packed.dequantize()

        for layer_index, field, state_index in (
            (0, "keys", None),
            (0, "values", None),
            (1, "conv_states", 0),
            (1, "recurrent_states", 0),
        ):
            packed_value = getattr(packed.cache.layers[layer_index], field)
            source = (
                packed_value
                if state_index is None
                else packed_value[state_index]
            ).data
            fork_a_value = getattr(fork_a.layers[layer_index], field)
            value_a = (
                fork_a_value
                if state_index is None
                else fork_a_value[state_index]
            )
            fork_b_value = getattr(fork_b.layers[layer_index], field)
            value_b = (
                fork_b_value
                if state_index is None
                else fork_b_value[state_index]
            )
            pointers = {
                source.untyped_storage().data_ptr(),
                value_a.untyped_storage().data_ptr(),
                value_b.untyped_storage().data_ptr(),
            }
            self.assertEqual(len(pointers), 3)
            source_before = source.clone()
            fork_b_before = value_b.clone()
            value_a.add_(1)
            self.assertTrue(torch.equal(source, source_before))
            self.assertTrue(torch.equal(value_b, fork_b_before))

    def test_compact_policy_expands_over_preallocated_empty_layers(self) -> None:
        active_attention = SimpleNamespace(
            keys=torch.randn(1, 2, 5, 16),
            values=torch.randn(1, 2, 5, 16),
        )
        active_linear = SimpleNamespace(
            conv_states=[torch.randn(1, 48, 3)],
            recurrent_states=[torch.randn(1, 4, 16, 16)],
        )
        cache = SimpleNamespace(
            layers=[active_attention, active_linear, SimpleNamespace(), SimpleNamespace()]
        )
        self.assertEqual(active_cache_layer_indices(cache), (0, 1))
        packed = quantize_transformers_cache(
            cache,
            attention_bits=16,
            linear_bits=16,
            cache_layer_bits=(4, 8),
            group_size=64,
        )
        self.assertEqual(packed.layer_bits, (4, 8, 16, 16))
        self.assertEqual(active_cache_layer_indices(packed.cache), (0, 1))

    def test_tiny_qwen35_split_is_exact(self) -> None:
        try:
            from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
                Qwen3_5MoeTextConfig,
            )
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeTextModel,
            )
        except ImportError:
            self.skipTest("target Transformers build is not installed")

        config = Qwen3_5MoeTextConfig(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
            moe_intermediate_size=64,
            shared_expert_intermediate_size=64,
            num_experts_per_tok=2,
            num_experts=4,
            layer_types=["full_attention", "linear_attention", "full_attention"],
        )
        language = Qwen3_5MoeTextModel(config).eval()
        head = torch.nn.Linear(64, 128, bias=False)
        class Wrapper:
            def __init__(self) -> None:
                self.model = SimpleNamespace(language_model=language)
                self.lm_head = head

            def __call__(
                self, *, input_ids, use_cache=False, logits_to_keep=0
            ) -> SimpleNamespace:
                del use_cache
                hidden = language(input_ids=input_ids).last_hidden_state
                if logits_to_keep:
                    hidden = hidden[:, -logits_to_keep:, :]
                return SimpleNamespace(logits=head(hidden))

        wrapper = Wrapper()
        adapter = TorchSplitCausalLM(wrapper)
        tokens = torch.arange(16).unsqueeze(0)
        full_hidden = language(input_ids=tokens).last_hidden_state
        expected = head(full_hidden[:, -1, :])
        residual = adapter.run_to_depth(tokens, 2)
        actual = adapter.run_suffix_last_logits([residual], 2)
        self.assertTrue(torch.equal(expected, actual))

        document = tokens[:, :11]
        query = tokens[:, 11:]
        dense_tokens = torch.cat([document, query], dim=1)
        dense_generated = greedy_generate_dense(
            adapter, dense_tokens, max_new_tokens=3, eos_token_ids=set()
        )
        oracle_generated = greedy_generate_oracle(
            adapter,
            document,
            query,
            depth=2,
            max_new_tokens=3,
            eos_token_ids=set(),
        )
        self.assertEqual(dense_generated, oracle_generated)

        replay_state = adapter.write_lower_replay(document, depth=2)
        replay_generated = greedy_generate_replay(
            adapter,
            replay_state,
            query,
            max_new_tokens=3,
            eos_token_ids=set(),
        )
        self.assertEqual(dense_generated, replay_generated)
        self.assertEqual(replay_state.current_length, document.shape[1])
        self.assertGreater(replay_state.stored_nbytes, residual.numel() * 2)
        packed_replay = replay_state.quantize(bits=4, group_size=64)
        self.assertLess(packed_replay.stored_nbytes, replay_state.stored_nbytes)
        self.assertEqual(
            packed_replay.fork().document_residual.shape,
            replay_state.document_residual.shape,
        )
        packed_all = replay_state.quantize(
            bits=8,
            attention_bits=8,
            linear_bits=8,
            group_size=64,
        )
        self.assertLess(packed_all.stored_nbytes, replay_state.stored_nbytes)
        packed_generated = greedy_generate_replay(
            adapter,
            packed_all,
            query,
            max_new_tokens=1,
            eos_token_ids=set(),
        )
        self.assertEqual(len(packed_generated), 1)

        prefix_state = adapter.write_full_prefix(document)
        prefix_generated = greedy_generate_full_prefix(
            adapter,
            prefix_state,
            query,
            max_new_tokens=3,
            eos_token_ids=set(),
        )
        self.assertEqual(dense_generated, prefix_generated)
        self.assertEqual(prefix_state.current_length, document.shape[1])
        self.assertGreater(prefix_state.stored_nbytes, 0)


if __name__ == "__main__":
    unittest.main()
