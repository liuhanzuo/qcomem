from __future__ import annotations

import unittest
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm.models.qwen3 import Model as Qwen3Model
from mlx_lm.models.qwen3 import ModelArgs as Qwen3ModelArgs

from macllm_bench.comem_model import SplitCausalLM


class FakeEmbedding:
    def __call__(self, tokens: mx.array) -> mx.array:
        offsets = mx.arange(4).astype(mx.float32)
        return tokens.astype(mx.float32)[..., None] + offsets

    def as_linear(self, hidden: mx.array) -> mx.array:
        return hidden


class AddLayer:
    def __init__(self, amount: float) -> None:
        self.amount = amount

    def __call__(self, hidden, mask=None, cache=None):
        return hidden + self.amount


class FakeModel:
    def __init__(self) -> None:
        self.args = SimpleNamespace(tie_word_embeddings=True)
        self.model = SimpleNamespace(
            embed_tokens=FakeEmbedding(),
            layers=[AddLayer(1), AddLayer(2), AddLayer(3)],
            norm=lambda hidden: hidden,
        )

    def __call__(self, tokens: mx.array) -> mx.array:
        hidden = self.model.embed_tokens(tokens)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return self.model.embed_tokens.as_linear(hidden)


class CoMemModelTest(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = SplitCausalLM(FakeModel())
        self.tokens = mx.array([1, 2, 3, 4])

    def test_every_split_reconstructs_full_forward(self) -> None:
        full = self.adapter.full_logits(self.tokens)
        captured = self.adapter.capture_depths(self.tokens, [0, 1, 2, 3])
        for depth, residual in captured.items():
            split = self.adapter.run_suffix(residual, depth)
            equal = mx.allclose(full, split)
            mx.eval(equal)
            self.assertTrue(bool(equal.item()), f"depth {depth} did not match")

    def test_chunk_local_write_trims_overlap(self) -> None:
        continuous = self.adapter.run_to_depth(self.tokens, 2)
        chunked = self.adapter.chunk_local_write(
            self.tokens, 2, chunk_size=2, overlap=1
        )
        equal = mx.allclose(continuous, chunked)
        mx.eval(equal)
        self.assertEqual(chunked.shape, continuous.shape)
        self.assertTrue(bool(equal.item()))

    def test_multiple_documents_are_reused_for_read(self) -> None:
        document_a = mx.array([1, 2])
        document_b = mx.array([3, 4])
        query = mx.array([5, 6])
        depth = 2
        stored = [
            self.adapter.run_to_depth(document_a, depth),
            self.adapter.run_to_depth(document_b, depth),
        ]
        actual, query_residual = self.adapter.read_documents(stored, query, depth)
        expected = self.adapter.full_logits(
            mx.concatenate([document_a, document_b, query])
        )
        equal = mx.allclose(actual, expected)
        query_equal = mx.allclose(
            query_residual, self.adapter.run_to_depth(query, depth)
        )
        mx.eval(equal, query_equal)
        self.assertTrue(bool(equal.item()))
        self.assertTrue(bool(query_equal.item()))

    def test_real_qwen_layout_split_is_exact(self) -> None:
        args = Qwen3ModelArgs(
            model_type="qwen3",
            hidden_size=64,
            num_hidden_layers=3,
            intermediate_size=128,
            num_attention_heads=4,
            rms_norm_eps=1e-6,
            vocab_size=128,
            num_key_value_heads=2,
            max_position_embeddings=128,
            rope_theta=10_000.0,
            head_dim=16,
            tie_word_embeddings=True,
        )
        adapter = SplitCausalLM(Qwen3Model(args))
        tokens = mx.arange(16)[None]
        full = adapter.full_logits(tokens)
        residual = adapter.run_to_depth(tokens, 2)
        split = adapter.run_suffix(residual, 2)
        equal = mx.array_equal(full, split)
        mx.eval(equal)
        self.assertTrue(bool(equal.item()))


if __name__ == "__main__":
    unittest.main()
