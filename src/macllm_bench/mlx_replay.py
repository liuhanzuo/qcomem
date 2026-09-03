from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from .comem_model import SplitCausalLM
from .replay_policy import ComponentProfile, QuantizationOption, ReplayBitPolicy
from .replay_policy import optimize_bit_policy as _optimize_bit_policy


SUPPORTED_REPLAY_BITS = (2, 4, 8, 16)
REPLAY_STORE_VERSION = 1
FROZEN_REPLAY_CONFIG = {
    "depth": 7,
    "residual_bits": 4,
    "attention_bits": 4,
    "linear_bits": 8,
    "group_size": 64,
}


def _dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).removeprefix("mlx.core.")


def _dtype_from_name(name: str) -> mx.Dtype:
    try:
        return {
            "bfloat16": mx.bfloat16,
            "float16": mx.float16,
            "float32": mx.float32,
        }[name]
    except KeyError as error:
        raise ValueError(f"unsupported stored dtype: {name}") from error


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_REPLAY_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_REPLAY_BITS}")


def _iter_arrays(value: Any) -> Iterator[mx.array]:
    if isinstance(value, mx.array):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_arrays(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_arrays(item)


def evaluate_cache(cache: Iterable[Any]) -> None:
    arrays = [array for entry in cache for array in _iter_arrays(entry.state)]
    if arrays:
        mx.eval(*arrays)


def cache_nbytes(cache: Iterable[Any]) -> int:
    return sum(
        array.nbytes for entry in cache for array in _iter_arrays(entry.state)
    )


def clone_cache(cache: Iterable[Any]) -> list[Any]:
    """Fork immutable document state into query-local mutable cache objects."""

    def clone_value(value: Any) -> Any:
        if isinstance(value, mx.array):
            # MLX arrays are normally immutable, but recurrent-state kernels may
            # reuse input buffers internally. Force distinct query-local storage.
            return value + mx.zeros_like(value)
        if isinstance(value, list):
            return [clone_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_value(item) for item in value)
        return value

    return [
        type(entry).from_state(clone_value(entry.state), entry.meta_state)
        for entry in cache
    ]


@dataclass
class StoredMLXTensor:
    """Affine group quantization for an arbitrary MLX floating tensor."""

    bits: int
    group_size: int
    mode: str
    original_shape: tuple[int, ...]
    source_dtype: str
    padded_elements: int
    data: mx.array
    scales: mx.array | None = None
    biases: mx.array | None = None

    def __post_init__(self) -> None:
        _validate_bits(self.bits)
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self.bits == 16:
            if self.scales is not None or self.biases is not None:
                raise ValueError("dense tensors do not use scales or biases")
        elif self.scales is None or self.biases is None:
            raise ValueError("quantized tensors require scales and biases")

    @property
    def nbytes(self) -> int:
        return sum(
            array.nbytes
            for array in (self.data, self.scales, self.biases)
            if array is not None
        )

    @property
    def elements(self) -> int:
        return math.prod(self.original_shape)

    @property
    def dense_nbytes(self) -> int:
        return self.elements * 2

    def eval(self) -> None:
        arrays = [self.data]
        if self.scales is not None:
            arrays.append(self.scales)
        if self.biases is not None:
            arrays.append(self.biases)
        mx.eval(*arrays)

    def dequantize(
        self,
        *,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        dtype = _dtype_from_name(self.source_dtype)
        if self.bits == 16:
            restored = self.data.astype(dtype, stream=stream)
        else:
            assert self.scales is not None and self.biases is not None
            restored = mx.dequantize(
                self.data,
                self.scales,
                self.biases,
                group_size=self.group_size,
                bits=self.bits,
                mode=self.mode,
                dtype=dtype,
                stream=stream,
            )
        return restored.reshape(-1)[: self.elements].reshape(self.original_shape)

    def descriptor(self, prefix: str, arrays: dict[str, mx.array]) -> dict[str, Any]:
        arrays[f"{prefix}.data"] = self.data
        if self.scales is not None:
            arrays[f"{prefix}.scales"] = self.scales
        if self.biases is not None:
            arrays[f"{prefix}.biases"] = self.biases
        return {
            "prefix": prefix,
            "bits": self.bits,
            "group_size": self.group_size,
            "mode": self.mode,
            "original_shape": self.original_shape,
            "source_dtype": self.source_dtype,
            "padded_elements": self.padded_elements,
        }

    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict[str, Any],
        arrays: dict[str, mx.array],
    ) -> StoredMLXTensor:
        prefix = descriptor["prefix"]
        return cls(
            bits=int(descriptor["bits"]),
            group_size=int(descriptor["group_size"]),
            mode=descriptor["mode"],
            original_shape=tuple(descriptor["original_shape"]),
            source_dtype=descriptor["source_dtype"],
            padded_elements=int(descriptor["padded_elements"]),
            data=arrays[f"{prefix}.data"],
            scales=arrays.get(f"{prefix}.scales"),
            biases=arrays.get(f"{prefix}.biases"),
        )


def quantize_mlx_tensor(
    tensor: mx.array,
    *,
    bits: int,
    group_size: int = 64,
    stream: mx.Stream | mx.Device | None = None,
) -> StoredMLXTensor:
    """Flatten, pad, and genuinely bit-pack any floating MLX tensor."""

    _validate_bits(bits)
    if not mx.issubdtype(tensor.dtype, mx.floating):
        raise ValueError("only floating tensors can be quantized")
    if tensor.size < 1:
        raise ValueError("empty tensors are not supported")
    original_shape = tuple(tensor.shape)
    source_dtype = _dtype_name(tensor.dtype)
    if source_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported floating dtype: {tensor.dtype}")

    if bits == 16:
        data = tensor
        if tensor.dtype not in {mx.bfloat16, mx.float16}:
            data = tensor.astype(mx.float16, stream=stream)
            source_dtype = "float16"
        return StoredMLXTensor(
            bits=16,
            group_size=group_size,
            mode="dense",
            original_shape=original_shape,
            source_dtype=source_dtype,
            padded_elements=tensor.size,
            data=data,
        )

    flat = tensor.reshape(-1)
    groups = (flat.size + group_size - 1) // group_size
    padded_elements = groups * group_size
    if padded_elements != flat.size:
        flat = mx.concatenate(
            [flat, mx.repeat(flat[-1:], padded_elements - flat.size)]
        )
    matrix = flat.reshape(1, padded_elements)
    data, scales, biases = mx.quantize(
        matrix,
        group_size=group_size,
        bits=bits,
        mode="affine",
        stream=stream,
    )
    return StoredMLXTensor(
        bits=bits,
        group_size=group_size,
        mode="affine",
        original_shape=original_shape,
        source_dtype=source_dtype,
        padded_elements=padded_elements,
        data=data,
        scales=scales,
        biases=biases,
    )


@dataclass
class PackedCacheLayer:
    cache_class: str
    is_linear: bool
    tensors: tuple[StoredMLXTensor | None, ...]

    @property
    def nbytes(self) -> int:
        return sum(tensor.nbytes for tensor in self.tensors if tensor is not None)

    def eval(self) -> None:
        for tensor in self.tensors:
            if tensor is not None:
                tensor.eval()

    def to_cache(
        self,
        *,
        stream: mx.Stream | mx.Device | None = None,
    ) -> Any:
        classes = {"ArraysCache": ArraysCache, "KVCache": KVCache}
        try:
            cache_type = classes[self.cache_class]
        except KeyError as error:
            raise ValueError(f"unsupported cache class: {self.cache_class}") from error
        values = [
            tensor.dequantize(stream=stream) if tensor is not None else None
            for tensor in self.tensors
        ]
        state = values if self.cache_class == "ArraysCache" else tuple(values)
        return cache_type.from_state(state, "")


@dataclass
class LowerReplayState:
    depth: int
    current_length: int
    document_residual: mx.array
    cache: list[Any]
    layer_is_linear: tuple[bool, ...]

    @property
    def stored_nbytes(self) -> int:
        return self.document_residual.nbytes + cache_nbytes(self.cache)

    def fork_cache(self) -> list[Any]:
        return clone_cache(self.cache)

    def quantize(
        self,
        *,
        residual_bits: int,
        attention_bits: int,
        linear_bits: int,
        cache_layer_bits: Sequence[int] | None = None,
        group_size: int = 64,
        stream: mx.Stream | mx.Device | None = None,
    ) -> PackedLowerReplayState:
        if len(self.cache) != len(self.layer_is_linear):
            raise ValueError("cache and layer type metadata disagree")
        if cache_layer_bits is None:
            resolved_layer_bits = tuple(
                linear_bits if is_linear else attention_bits
                for is_linear in self.layer_is_linear
            )
        else:
            resolved_layer_bits = tuple(cache_layer_bits)
            if len(resolved_layer_bits) != len(self.cache):
                raise ValueError("cache_layer_bits must have one entry per layer")
        for bits in (residual_bits, attention_bits, linear_bits, *resolved_layer_bits):
            _validate_bits(bits)
        residual = quantize_mlx_tensor(
            self.document_residual,
            bits=residual_bits,
            group_size=group_size,
            stream=stream,
        )
        packed_layers = []
        for entry, is_linear, bits in zip(
            self.cache, self.layer_is_linear, resolved_layer_bits
        ):
            packed_layers.append(
                PackedCacheLayer(
                    cache_class=type(entry).__name__,
                    is_linear=is_linear,
                    tensors=tuple(
                        quantize_mlx_tensor(
                            tensor,
                            bits=bits,
                            group_size=group_size,
                            stream=stream,
                        )
                        if tensor is not None
                        else None
                        for tensor in entry.state
                    ),
                )
            )
        return PackedLowerReplayState(
            depth=self.depth,
            current_length=self.current_length,
            residual_bits=residual_bits,
            attention_bits=attention_bits,
            linear_bits=linear_bits,
            cache_layer_bits=resolved_layer_bits,
            group_size=group_size,
            document_residual=residual,
            cache=tuple(packed_layers),
        )


@dataclass
class PackedLowerReplayState:
    depth: int
    current_length: int
    residual_bits: int
    attention_bits: int
    linear_bits: int
    cache_layer_bits: tuple[int, ...]
    group_size: int
    document_residual: StoredMLXTensor
    cache: tuple[PackedCacheLayer, ...]

    @property
    def stored_nbytes(self) -> int:
        return self.document_residual.nbytes + sum(
            layer.nbytes for layer in self.cache
        )

    def eval(self) -> None:
        self.document_residual.eval()
        for layer in self.cache:
            layer.eval()

    def fork_cache(
        self,
        *,
        stream: mx.Stream | mx.Device | None = None,
    ) -> list[Any]:
        return [layer.to_cache(stream=stream) for layer in self.cache]

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, mx.array] = {}
        manifest: dict[str, Any] = {
            "depth": self.depth,
            "current_length": self.current_length,
            "residual_bits": self.residual_bits,
            "attention_bits": self.attention_bits,
            "linear_bits": self.linear_bits,
            "cache_layer_bits": self.cache_layer_bits,
            "group_size": self.group_size,
            "document_residual": self.document_residual.descriptor(
                "residual", arrays
            ),
            "cache": [],
        }
        for layer_index, layer in enumerate(self.cache):
            descriptors = []
            for tensor_index, tensor in enumerate(layer.tensors):
                descriptors.append(
                    tensor.descriptor(
                        f"cache.{layer_index}.{tensor_index}", arrays
                    )
                    if tensor is not None
                    else None
                )
            manifest["cache"].append(
                {
                    "cache_class": layer.cache_class,
                    "is_linear": layer.is_linear,
                    "tensors": descriptors,
                }
            )
        self.eval()
        mx.save_safetensors(
            destination,
            arrays,
            metadata={
                "format": "q-comem-mlx-hybrid-replay",
                "version": str(REPLAY_STORE_VERSION),
                "manifest": json.dumps(manifest, separators=(",", ":")),
            },
        )
        return destination

    @classmethod
    def load(cls, path: Path | str) -> PackedLowerReplayState:
        arrays, metadata = mx.load(path, return_metadata=True, stream=mx.cpu)
        if metadata.get("format") != "q-comem-mlx-hybrid-replay":
            raise ValueError("not a Q-CoMem MLX hybrid replay store")
        if int(metadata.get("version", -1)) != REPLAY_STORE_VERSION:
            raise ValueError("unsupported Q-CoMem MLX replay store version")
        manifest = json.loads(metadata["manifest"])
        layers = []
        for layer in manifest["cache"]:
            layers.append(
                PackedCacheLayer(
                    cache_class=layer["cache_class"],
                    is_linear=bool(layer["is_linear"]),
                    tensors=tuple(
                        StoredMLXTensor.from_descriptor(tensor, arrays)
                        if tensor is not None
                        else None
                        for tensor in layer["tensors"]
                    ),
                )
            )
        cache_layer_bits = manifest.get("cache_layer_bits")
        if cache_layer_bits is None:
            cache_layer_bits = []
            for layer in layers:
                first = next(
                    (tensor for tensor in layer.tensors if tensor is not None),
                    None,
                )
                cache_layer_bits.append(
                    first.bits
                    if first is not None
                    else (
                        int(manifest["linear_bits"])
                        if layer.is_linear
                        else int(manifest["attention_bits"])
                    )
                )
        return cls(
            depth=int(manifest["depth"]),
            current_length=int(manifest["current_length"]),
            residual_bits=int(manifest["residual_bits"]),
            attention_bits=int(manifest["attention_bits"]),
            linear_bits=int(manifest["linear_bits"]),
            cache_layer_bits=tuple(int(bits) for bits in cache_layer_bits),
            group_size=int(manifest["group_size"]),
            document_residual=StoredMLXTensor.from_descriptor(
                manifest["document_residual"], arrays
            ),
            cache=tuple(layers),
        )


@dataclass
class FullPrefixState:
    current_length: int
    cache: list[Any]

    @property
    def stored_nbytes(self) -> int:
        return cache_nbytes(self.cache)

    def fork_cache(self) -> list[Any]:
        return clone_cache(self.cache)


def write_lower_replay(
    adapter: SplitCausalLM,
    document_tokens: mx.array,
    *,
    depth: int,
) -> LowerReplayState:
    tokens = adapter._batch_tokens(document_tokens)
    cache = adapter.make_cache(depth)
    residual = adapter.run_to_depth_cached(tokens, depth, cache)
    mx.eval(residual)
    evaluate_cache(cache)
    return LowerReplayState(
        depth=depth,
        current_length=tokens.shape[1],
        document_residual=residual,
        cache=cache,
        layer_is_linear=tuple(
            bool(getattr(layer, "is_linear", False))
            for layer in adapter.layers[:depth]
        ),
    )


def write_lower_replay_documents(
    adapter: SplitCausalLM,
    documents: Iterable[mx.array],
    *,
    depth: int,
) -> LowerReplayState:
    """Write an exact fixed-order multi-document lower replay state.

    Documents share one continuous positional/cache history.  Reordering or
    dropping a document requires rebuilding this state; arbitrary segment
    composition is deliberately not implied by this exact baseline.
    """

    token_batches = [adapter._batch_tokens(tokens) for tokens in documents]
    if not token_batches:
        raise ValueError("at least one document is required")
    if any(tokens.shape[1] == 0 for tokens in token_batches):
        raise ValueError("replay documents must contain at least one token")
    batch_sizes = {tokens.shape[0] for tokens in token_batches}
    if len(batch_sizes) != 1:
        raise ValueError("all replay documents must use the same batch size")

    cache = adapter.make_cache(depth)
    residuals = [
        adapter.run_to_depth_cached(tokens, depth, cache)
        for tokens in token_batches
    ]
    residual = mx.concatenate(residuals, axis=1)
    mx.eval(residual)
    evaluate_cache(cache)
    return LowerReplayState(
        depth=depth,
        current_length=sum(tokens.shape[1] for tokens in token_batches),
        document_residual=residual,
        cache=cache,
        layer_is_linear=tuple(
            bool(getattr(layer, "is_linear", False))
            for layer in adapter.layers[:depth]
        ),
    )


def write_full_prefix(
    adapter: SplitCausalLM,
    document_tokens: mx.array,
) -> FullPrefixState:
    tokens = adapter._batch_tokens(document_tokens)
    cache = adapter.make_cache()
    adapter.run_to_depth_cached(tokens, adapter.num_layers, cache)
    evaluate_cache(cache)
    return FullPrefixState(current_length=tokens.shape[1], cache=cache)


def read_lower_replay(
    adapter: SplitCausalLM,
    state: LowerReplayState | PackedLowerReplayState,
    query_tokens: mx.array,
    *,
    stream: mx.Stream | mx.Device | None = None,
) -> mx.array:
    if isinstance(state, PackedLowerReplayState):
        document_residual = state.document_residual.dequantize(stream=stream)
        cache = state.fork_cache(stream=stream)
    else:
        document_residual = state.document_residual
        cache = state.fork_cache()
    query_residual = adapter.run_to_depth_cached(query_tokens, state.depth, cache)
    combined = mx.concatenate([document_residual, query_residual], axis=1)
    return adapter.run_suffix(combined, state.depth)


def read_full_prefix(
    adapter: SplitCausalLM,
    state: FullPrefixState,
    query_tokens: mx.array,
) -> mx.array:
    return adapter.run_all_cached(query_tokens, state.fork_cache())


def greedy_generate_dense(
    adapter: SplitCausalLM,
    tokens: mx.array,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> list[int]:
    cache = adapter.make_cache()
    logits = adapter.run_all_cached_last_logits(tokens, cache)
    generated = []
    for _ in range(max_new_tokens):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(token)
        if token in eos_token_ids:
            break
        logits = adapter.run_all_cached_last_logits(mx.array([[token]]), cache)
    return generated


def greedy_generate_full_prefix(
    adapter: SplitCausalLM,
    state: FullPrefixState,
    query_tokens: mx.array,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
) -> list[int]:
    cache = state.fork_cache()
    logits = adapter.run_all_cached_last_logits(query_tokens, cache)
    generated = []
    for _ in range(max_new_tokens):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(token)
        if token in eos_token_ids:
            break
        logits = adapter.run_all_cached_last_logits(mx.array([[token]]), cache)
    return generated


def greedy_generate_replay(
    adapter: SplitCausalLM,
    state: LowerReplayState | PackedLowerReplayState,
    query_tokens: mx.array,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    stream: mx.Stream | mx.Device | None = None,
) -> list[int]:
    if isinstance(state, PackedLowerReplayState):
        document_residual = state.document_residual.dequantize(stream=stream)
        cache = state.fork_cache(stream=stream)
    else:
        document_residual = state.document_residual
        cache = state.fork_cache()
    query_residual = adapter.run_to_depth_cached(query_tokens, state.depth, cache)
    suffix_cache = adapter.make_suffix_cache(state.depth)
    logits = adapter.run_suffix_cached_last_logits(
        mx.concatenate([document_residual, query_residual], axis=1),
        state.depth,
        suffix_cache,
    )
    generated = []
    for _ in range(max_new_tokens):
        token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        generated.append(token)
        if token in eos_token_ids:
            break
        token_residual = adapter.run_to_depth_cached(
            mx.array([[token]]), state.depth, cache
        )
        logits = adapter.run_suffix_cached_last_logits(
            token_residual,
            state.depth,
            suffix_cache,
        )
    return generated


def error_sums(reference: mx.array, candidate: mx.array) -> dict[str, float | int]:
    error = reference.astype(mx.float32) - candidate.astype(mx.float32)
    squared_error = mx.sum(mx.square(error))
    reference_squared = mx.sum(mx.square(reference.astype(mx.float32)))
    max_abs = mx.max(mx.abs(error))
    mx.eval(squared_error, reference_squared, max_abs)
    return {
        "squared_error_sum": float(squared_error.item()),
        "reference_squared_sum": float(reference_squared.item()),
        "max_abs_error": float(max_abs.item()),
        "elements": reference.size,
    }


def packed_state_error_sums(
    reference: LowerReplayState,
    candidate: PackedLowerReplayState,
    *,
    stream: mx.Stream | mx.Device | None = None,
) -> dict[str, dict[str, float | int]]:
    categories = {
        name: {
            "squared_error_sum": 0.0,
            "reference_squared_sum": 0.0,
            "max_abs_error": 0.0,
            "elements": 0,
        }
        for name in ("residual", "attention", "linear")
    }

    def merge(category: str, metrics: dict[str, float | int]) -> None:
        target = categories[category]
        target["squared_error_sum"] += metrics["squared_error_sum"]
        target["reference_squared_sum"] += metrics["reference_squared_sum"]
        target["max_abs_error"] = max(
            target["max_abs_error"], metrics["max_abs_error"]
        )
        target["elements"] += metrics["elements"]

    merge(
        "residual",
        error_sums(
            reference.document_residual,
            candidate.document_residual.dequantize(stream=stream),
        ),
    )
    for dense_layer, packed_layer in zip(reference.cache, candidate.cache):
        category = "linear" if packed_layer.is_linear else "attention"
        for dense, packed in zip(dense_layer.state, packed_layer.tensors):
            if dense is not None and packed is not None:
                merge(category, error_sums(dense, packed.dequantize(stream=stream)))
    return categories


def relative_rmse(metrics: dict[str, float | int]) -> float:
    denominator = float(metrics["reference_squared_sum"])
    if denominator == 0:
        return 0.0
    return math.sqrt(float(metrics["squared_error_sum"]) / denominator)


def profile_replay_quantization(
    state: LowerReplayState,
    *,
    candidate_bits: Sequence[int] = SUPPORTED_REPLAY_BITS,
    group_size: int = 64,
    stream: mx.Stream | mx.Device | None = None,
) -> tuple[ComponentProfile, ...]:
    """Measure additive storage/error choices for residual and cache layers."""

    bits_to_measure = tuple(candidate_bits)
    if not bits_to_measure:
        raise ValueError("candidate_bits must not be empty")
    for bits in bits_to_measure:
        _validate_bits(bits)

    components: list[tuple[str, tuple[mx.array, ...]]] = [
        ("residual", (state.document_residual,))
    ]
    for layer_index, entry in enumerate(state.cache):
        tensors = tuple(
            tensor
            for tensor in entry.state
            if tensor is not None and mx.issubdtype(tensor.dtype, mx.floating)
        )
        components.append((f"cache.{layer_index}", tensors))

    measured: list[tuple[str, list[tuple[int, int, float]], float]] = []
    global_reference_squared = 0.0
    for name, tensors in components:
        reference_squared = 0.0
        options = []
        for bits in bits_to_measure:
            stored_bytes = 0
            squared_error = 0.0
            current_reference_squared = 0.0
            for tensor in tensors:
                packed = quantize_mlx_tensor(
                    tensor,
                    bits=bits,
                    group_size=group_size,
                    stream=stream,
                )
                restored = packed.dequantize(stream=stream)
                metrics = error_sums(tensor, restored)
                stored_bytes += packed.nbytes
                squared_error += float(metrics["squared_error_sum"])
                current_reference_squared += float(
                    metrics["reference_squared_sum"]
                )
            options.append((bits, stored_bytes, squared_error))
            reference_squared = current_reference_squared
        measured.append((name, options, reference_squared))
        global_reference_squared += reference_squared

    denominator = global_reference_squared or 1.0
    return tuple(
        ComponentProfile(
            name=name,
            options=tuple(
                QuantizationOption(
                    bits=bits,
                    nbytes=stored_bytes,
                    distortion=squared_error / denominator,
                )
                for bits, stored_bytes, squared_error in options
            ),
        )
        for name, options, _ in measured
    )


def select_replay_bit_policy(
    profiles: Iterable[ComponentProfile],
    *,
    budget_bytes: int,
) -> ReplayBitPolicy:
    return _optimize_bit_policy(profiles, budget_bytes=budget_bytes)


def quantize_replay_with_policy(
    state: LowerReplayState,
    policy: ReplayBitPolicy,
    *,
    group_size: int = 64,
    stream: mx.Stream | mx.Device | None = None,
) -> PackedLowerReplayState:
    choices = policy.as_dict()
    required = {"residual", *(f"cache.{index}" for index in range(len(state.cache)))}
    missing = required.difference(choices)
    if missing:
        raise ValueError(f"policy is missing replay components: {sorted(missing)}")
    cache_layer_bits = tuple(
        choices[f"cache.{index}"] for index in range(len(state.cache))
    )
    return state.quantize(
        residual_bits=choices["residual"],
        attention_bits=16,
        linear_bits=16,
        cache_layer_bits=cache_layer_bits,
        group_size=group_size,
        stream=stream,
    )
