from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.qwen3_5 import Model, ModelArgs

from macllm_bench.comem_model import SplitCausalLM
from macllm_bench.mlx_replay import (
    PackedLowerReplayState,
    error_sums,
    greedy_generate_dense,
    greedy_generate_full_prefix,
    greedy_generate_replay,
    packed_state_error_sums,
    profile_replay_quantization,
    quantize_replay_with_policy,
    quantize_mlx_tensor,
    read_full_prefix,
    read_lower_replay,
    relative_rmse,
    select_replay_bit_policy,
    write_full_prefix,
    write_lower_replay,
    write_lower_replay_documents,
)


def tiny_qwen35() -> Model:
    return Model(
        ModelArgs(
            model_type="qwen3_5",
            text_config={
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 16,
                "vocab_size": 128,
                "linear_num_value_heads": 4,
                "linear_num_key_heads": 2,
                "linear_key_head_dim": 16,
                "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 4,
                "full_attention_interval": 2,
                "tie_word_embeddings": True,
                "max_position_embeddings": 128,
            },
        )
    )


class MLXReplayTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.device = mx.gpu if mx.metal.is_available() else mx.cpu
        mx.set_default_device(cls.device)

    def test_arbitrary_tensor_real_packing(self) -> None:
        values = mx.cos(mx.arange(17 * 67).reshape(17, 67) / 23).astype(
            mx.float16
        )
        stores = {
            bits: quantize_mlx_tensor(
                values, bits=bits, group_size=64, stream=self.device
            )
            for bits in (2, 4, 8, 16)
        }
        errors = {}
        for bits, store in stores.items():
            restored = store.dequantize(stream=self.device)
            mx.eval(restored)
            self.assertEqual(restored.shape, values.shape)
            errors[bits] = relative_rmse(error_sums(values, restored))
        self.assertLess(stores[2].nbytes, stores[4].nbytes)
        self.assertLess(stores[4].nbytes, stores[8].nbytes)
        self.assertLess(stores[8].nbytes, stores[16].nbytes)
        self.assertLess(errors[8], errors[4])
        self.assertLess(errors[4], errors[2])

    def test_qwen35_hybrid_exact_replay_and_frozen_pack(self) -> None:
        adapter = SplitCausalLM(tiny_qwen35())
        document = mx.arange(11)
        query = mx.arange(11, 16)
        full_tokens = mx.concatenate([document, query])
        full_logits = adapter.full_logits(full_tokens)

        replay = write_lower_replay(adapter, document, depth=3)
        replay_logits = read_lower_replay(adapter, replay, query)
        prefix = write_full_prefix(adapter, document)
        prefix_logits = read_full_prefix(adapter, prefix, query)
        mx.eval(full_logits, replay_logits, prefix_logits)
        self.assertTrue(
            bool(mx.allclose(full_logits, replay_logits, rtol=1e-4, atol=1e-4).item())
        )
        self.assertTrue(
            bool(
                mx.allclose(
                    full_logits[:, -query.shape[0] :],
                    prefix_logits,
                    rtol=1e-4,
                    atol=1e-4,
                ).item()
            )
        )

        dense_tokens = greedy_generate_dense(
            adapter, full_tokens, max_new_tokens=3, eos_token_ids=set()
        )
        prefix_tokens = greedy_generate_full_prefix(
            adapter, prefix, query, max_new_tokens=3, eos_token_ids=set()
        )
        replay_tokens = greedy_generate_replay(
            adapter, replay, query, max_new_tokens=3, eos_token_ids=set()
        )
        self.assertEqual(dense_tokens, prefix_tokens)
        self.assertEqual(dense_tokens, replay_tokens)

        packed = replay.quantize(
            residual_bits=4,
            attention_bits=4,
            linear_bits=8,
            group_size=64,
            stream=self.device,
        )
        packed.eval()
        self.assertLess(packed.stored_nbytes, replay.stored_nbytes)
        self.assertEqual(packed.document_residual.bits, 4)
        self.assertEqual(
            [layer.tensors[0].bits for layer in packed.cache], [8, 4, 8]
        )
        errors = packed_state_error_sums(replay, packed, stream=self.device)
        self.assertLess(relative_rmse(errors["linear"]), 0.02)
        packed_logits = read_lower_replay(
            adapter, packed, query, stream=self.device
        )
        mx.eval(packed_logits)
        self.assertEqual(packed_logits.shape, full_logits.shape)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid-replay.safetensors"
            packed.save(path)
            loaded = PackedLowerReplayState.load(path)
            loaded_logits = read_lower_replay(
                adapter, loaded, query, stream=self.device
            )
            mx.eval(loaded_logits)
            equal = mx.array_equal(packed_logits, loaded_logits)
            mx.eval(equal)
        self.assertTrue(bool(equal.item()))
        self.assertEqual(loaded.stored_nbytes, packed.stored_nbytes)
        self.assertEqual(loaded.depth, 3)
        self.assertEqual(loaded.cache_layer_bits, (8, 4, 8))

    def test_fixed_order_multidoc_and_automatic_layer_bits(self) -> None:
        adapter = SplitCausalLM(tiny_qwen35())
        documents = (mx.arange(7), mx.arange(7, 13), mx.arange(13, 18))
        combined = mx.concatenate(documents)
        query = mx.arange(18, 22)
        state = write_lower_replay_documents(adapter, documents, depth=3)
        single_state = write_lower_replay(adapter, combined, depth=3)
        multi_logits = read_lower_replay(adapter, state, query)
        single_logits = read_lower_replay(adapter, single_state, query)
        mx.eval(multi_logits, single_logits)
        self.assertTrue(
            bool(mx.allclose(multi_logits, single_logits, rtol=1e-4, atol=1e-4).item())
        )
        self.assertEqual(state.current_length, combined.shape[0])

        custom = state.quantize(
            residual_bits=4,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=(8, 2, 4),
            group_size=64,
            stream=self.device,
        )
        self.assertEqual(custom.cache_layer_bits, (8, 2, 4))
        self.assertEqual(
            [layer.tensors[0].bits for layer in custom.cache], [8, 2, 4]
        )

        profiles = profile_replay_quantization(
            state,
            candidate_bits=(2, 4, 8, 16),
            group_size=64,
            stream=self.device,
        )
        q4_budget = sum(
            next(option.nbytes for option in profile.options if option.bits == 4)
            for profile in profiles
        )
        policy = select_replay_bit_policy(profiles, budget_bytes=q4_budget)
        selected = quantize_replay_with_policy(
            state, policy, group_size=64, stream=self.device
        )
        selected.eval()
        self.assertLessEqual(selected.stored_nbytes, q4_budget)
        self.assertEqual(
            set(policy.as_dict()), {"residual", "cache.0", "cache.1", "cache.2"}
        )


if __name__ == "__main__":
    unittest.main()
