from __future__ import annotations

from collections.abc import Iterable

import mlx.core as mx
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import make_prompt_cache


class SplitCausalLM:
    """Expose embedding, prefix layers, suffix layers, norm, and LM head.

    The adapter targets both the common MLX-LM Llama/Qwen layout and the
    Qwen3.5 ``language_model.model`` layout. It does not use MLX-LM generation
    helpers.
    """

    def __init__(self, model) -> None:
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            backbone = model.model
            projection_model = model
        elif (
            hasattr(model, "language_model")
            and hasattr(model.language_model, "model")
            and hasattr(model.language_model.model, "layers")
        ):
            backbone = model.language_model.model
            projection_model = model.language_model
        else:
            raise TypeError("model does not expose the expected MLX-LM backbone")
        if not hasattr(backbone, "embed_tokens") or not hasattr(backbone, "norm"):
            raise TypeError("model backbone is missing embedding or final norm")
        self.model = model
        self.backbone = backbone
        self.projection_model = projection_model
        self.layers = self.backbone.layers
        self.num_layers = len(self.layers)

    def _validate_depth(self, depth: int) -> None:
        if depth < 0 or depth > self.num_layers:
            raise ValueError(
                f"depth must be between 0 and {self.num_layers}, got {depth}"
            )

    @staticmethod
    def _batch_tokens(tokens: mx.array) -> mx.array:
        if tokens.ndim == 1:
            return tokens[None]
        if tokens.ndim != 2:
            raise ValueError("token IDs must have shape [tokens] or [batch, tokens]")
        return tokens

    def embed(self, tokens: mx.array) -> mx.array:
        return self.backbone.embed_tokens(self._batch_tokens(tokens))

    @staticmethod
    def _layer_mask(layer, hidden: mx.array, cache=None):
        if getattr(layer, "is_linear", False):
            return create_ssm_mask(hidden, cache)
        return create_attention_mask(hidden, cache)

    def make_cache(self, depth: int | None = None) -> list:
        """Create MLX-LM caches, optionally keeping only lower layers."""

        if depth is not None:
            self._validate_depth(depth)
        cache = make_prompt_cache(self.model)
        return cache if depth is None else cache[:depth]

    def make_suffix_cache(self, depth: int) -> list:
        """Create empty caches for layers ``[depth, L)`` only."""

        self._validate_depth(depth)
        return make_prompt_cache(self.model)[depth:]

    def run_to_depth(self, tokens: mx.array, depth: int) -> mx.array:
        """Run embedding and layers ``[0, depth)`` with causal attention."""

        self._validate_depth(depth)
        h = self.embed(tokens)
        for layer in self.layers[:depth]:
            h = layer(h, mask=self._layer_mask(layer, h), cache=None)
        return h

    def run_to_depth_cached(
        self,
        tokens: mx.array,
        depth: int,
        cache: list,
    ) -> mx.array:
        """Run lower layers while reading and advancing their exact cache."""

        self._validate_depth(depth)
        if len(cache) != depth:
            raise ValueError(f"expected {depth} lower cache entries, got {len(cache)}")
        h = self.embed(tokens)
        for layer, layer_cache in zip(self.layers[:depth], cache):
            h = layer(
                h,
                mask=self._layer_mask(layer, h, layer_cache),
                cache=layer_cache,
            )
        return h

    def capture_depths(
        self, tokens: mx.array, depths: Iterable[int]
    ) -> dict[int, mx.array]:
        """Capture several split residuals in a single lower-model pass."""

        requested = sorted(set(int(depth) for depth in depths))
        if not requested:
            raise ValueError("at least one split depth is required")
        for depth in requested:
            self._validate_depth(depth)

        h = self.embed(tokens)
        captured = {}
        if 0 in requested:
            captured[0] = h
        for depth, layer in enumerate(self.layers, start=1):
            h = layer(h, mask=self._layer_mask(layer, h), cache=None)
            if depth in requested:
                captured[depth] = h
            if depth == requested[-1]:
                break
        return captured

    def chunk_local_write(
        self,
        tokens: mx.array,
        depth: int,
        *,
        chunk_size: int,
        overlap: int = 0,
    ) -> mx.array:
        """Write chunks independently through the lower ``depth`` layers.

        Each chunk resets lower-layer positions. Optional left overlap is
        computed but trimmed before persistence, matching CoMem's repair idea.
        """

        self._validate_depth(depth)
        batched = self._batch_tokens(tokens)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")

        length = batched.shape[1]
        chunks = []
        for start in range(0, length, chunk_size):
            end = min(start + chunk_size, length)
            context_start = max(0, start - overlap)
            local_tokens = batched[:, context_start:end]
            local_h = self.run_to_depth(local_tokens, depth)
            trim = start - context_start
            chunks.append(local_h[:, trim:, :])
        return mx.concatenate(chunks, axis=1)

    def project_logits(self, hidden: mx.array) -> mx.array:
        hidden = self.backbone.norm(hidden)
        if self.projection_model.args.tie_word_embeddings:
            return self.backbone.embed_tokens.as_linear(hidden)
        return self.projection_model.lm_head(hidden)

    def run_suffix(self, residual: mx.array, depth: int) -> mx.array:
        """Run layers ``[depth, L)`` and return logits for all positions."""

        self._validate_depth(depth)
        if residual.ndim != 3:
            raise ValueError("residual must have shape [batch, tokens, hidden]")
        h = residual
        for layer in self.layers[depth:]:
            h = layer(h, mask=self._layer_mask(layer, h), cache=None)
        return self.project_logits(h)

    def run_suffix_last_logits(self, residual: mx.array, depth: int) -> mx.array:
        """Run the suffix but project only the final token to vocabulary logits."""

        self._validate_depth(depth)
        if residual.ndim != 3:
            raise ValueError("residual must have shape [batch, tokens, hidden]")
        h = residual
        for layer in self.layers[depth:]:
            h = layer(h, mask=self._layer_mask(layer, h), cache=None)
        return self.project_logits(h[:, -1:, :])

    def run_suffix_cached_last_logits(
        self,
        residual: mx.array,
        depth: int,
        cache: list,
    ) -> mx.array:
        """Advance suffix-only caches and project logits for the final token."""

        self._validate_depth(depth)
        if residual.ndim != 3:
            raise ValueError("residual must have shape [batch, tokens, hidden]")
        expected = self.num_layers - depth
        if len(cache) != expected:
            raise ValueError(f"expected {expected} suffix cache entries, got {len(cache)}")
        h = residual
        for layer, layer_cache in zip(self.layers[depth:], cache):
            h = layer(
                h,
                mask=self._layer_mask(layer, h, layer_cache),
                cache=layer_cache,
            )
        return self.project_logits(h[:, -1:, :])

    def run_all_cached(self, tokens: mx.array, cache: list) -> mx.array:
        """Run all layers with a mutable cache and project token logits."""

        if len(cache) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} cache entries, got {len(cache)}"
            )
        h = self.embed(tokens)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(
                h,
                mask=self._layer_mask(layer, h, layer_cache),
                cache=layer_cache,
            )
        return self.project_logits(h)

    def run_all_cached_last_logits(self, tokens: mx.array, cache: list) -> mx.array:
        """Cached full-model forward with vocabulary projection on one token."""

        if len(cache) != self.num_layers:
            raise ValueError(
                f"expected {self.num_layers} cache entries, got {len(cache)}"
            )
        h = self.embed(tokens)
        for layer, layer_cache in zip(self.layers, cache):
            h = layer(
                h,
                mask=self._layer_mask(layer, h, layer_cache),
                cache=layer_cache,
            )
        return self.project_logits(h[:, -1:, :])

    def read_documents(
        self,
        document_residuals: Iterable[mx.array],
        query_tokens: mx.array,
        depth: int,
    ) -> tuple[mx.array, mx.array]:
        """Reuse stored document residuals for one query.

        Documents have already run through lower layers during offline Write.
        The query independently runs through those lower layers, then the
        selected documents and query are concatenated for the shared suffix.
        The returned query residual makes query-Write and suffix-Read timing
        observable separately in the benchmark.
        """

        self._validate_depth(depth)
        documents = list(document_residuals)
        if not documents:
            raise ValueError("at least one document residual must be selected")
        query_residual = self.run_to_depth(query_tokens, depth)
        batch = query_residual.shape[0]
        hidden = query_residual.shape[-1]
        for residual in documents:
            if residual.ndim != 3:
                raise ValueError("document residuals must have shape [batch, tokens, hidden]")
            if residual.shape[0] != batch or residual.shape[-1] != hidden:
                raise ValueError("document and query residual shapes are incompatible")
        combined = mx.concatenate([*documents, query_residual], axis=1)
        return self.run_suffix(combined, depth), query_residual

    def full_logits(self, tokens: mx.array) -> mx.array:
        return self.model(self._batch_tokens(tokens))
