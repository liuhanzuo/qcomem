from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mlx.core as mx


SUPPORTED_QUANT_BITS = (2, 3, 4, 5, 6, 8)
SUPPORTED_STORAGE_BITS = (*SUPPORTED_QUANT_BITS, 16)
STORE_FORMAT_VERSION = 1


def _dtype_name(dtype: mx.Dtype) -> str:
    return str(dtype).removeprefix("mlx.core.")


def _dtype_from_name(name: str) -> mx.Dtype:
    dtypes = {
        "bfloat16": mx.bfloat16,
        "float16": mx.float16,
        "float32": mx.float32,
    }
    try:
        return dtypes[name]
    except KeyError as error:
        raise ValueError(f"unsupported residual dtype in store: {name}") from error


def _validate_bits(bits: int) -> None:
    if bits not in SUPPORTED_STORAGE_BITS:
        supported = ", ".join(str(value) for value in SUPPORTED_STORAGE_BITS)
        raise ValueError(f"bits must be one of: {supported}")


@dataclass(frozen=True)
class DepthBitPolicy:
    """A deployment policy assigning one residual bit width to each split depth."""

    assignments: Mapping[int, int]
    default_bits: int = 16

    def __post_init__(self) -> None:
        _validate_bits(self.default_bits)
        normalized = {}
        for depth, bits in self.assignments.items():
            depth = int(depth)
            bits = int(bits)
            if depth < 0:
                raise ValueError("split depths must be non-negative")
            _validate_bits(bits)
            normalized[depth] = bits
        object.__setattr__(self, "assignments", normalized)

    def bits_for(self, depth: int) -> int:
        return self.assignments.get(depth, self.default_bits)

    def as_dict(self) -> dict[str, int]:
        return {
            str(depth): bits for depth, bits in sorted(self.assignments.items())
        }

    @classmethod
    def from_specs(
        cls, specs: Iterable[str], default_bits: int = 16
    ) -> DepthBitPolicy:
        """Parse entries such as ``6:4 9:4 12:8`` or ``6:4,9:4,12:8``."""

        assignments = {}
        for spec in specs:
            for entry in spec.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                try:
                    depth_text, bits_text = entry.split(":", maxsplit=1)
                    depth = int(depth_text)
                    bits = int(bits_text)
                except ValueError as error:
                    raise ValueError(
                        f"invalid depth-bit entry {entry!r}; expected DEPTH:BITS"
                    ) from error
                assignments[depth] = bits
        return cls(assignments=assignments, default_bits=default_bits)


@dataclass
class StoredResidual:
    """A dense or genuinely bit-packed intermediate residual.

    For ``bits < 16``, ``data`` is the packed integer tensor returned by
    ``mlx.core.quantize``. Scales and biases are stored separately per group.
    """

    depth: int
    bits: int
    group_size: int
    mode: str
    original_shape: tuple[int, ...]
    source_dtype: str
    data: mx.array
    scales: mx.array | None = None
    biases: mx.array | None = None

    def __post_init__(self) -> None:
        _validate_bits(self.bits)
        if self.depth < 0:
            raise ValueError("depth must be non-negative")
        if self.group_size < 1:
            raise ValueError("group_size must be positive")
        if self.bits == 16:
            if self.scales is not None or self.biases is not None:
                raise ValueError("dense 16-bit residuals do not use scales/biases")
        elif self.scales is None or self.biases is None:
            raise ValueError("quantized affine residuals require scales and biases")

    @property
    def quantized(self) -> bool:
        return self.bits < 16

    @property
    def nbytes(self) -> int:
        arrays = (self.data, self.scales, self.biases)
        return sum(array.nbytes for array in arrays if array is not None)

    @property
    def dense_nbytes(self) -> int:
        elements = 1
        for dimension in self.original_shape:
            elements *= dimension
        return elements * 2

    @property
    def compression_ratio(self) -> float:
        return self.dense_nbytes / self.nbytes

    def eval(self) -> None:
        arrays = [self.data]
        if self.scales is not None:
            arrays.append(self.scales)
        if self.biases is not None:
            arrays.append(self.biases)
        mx.eval(*arrays)

    def dequantize(
        self,
        dtype: mx.Dtype | None = None,
        stream: mx.Stream | mx.Device | None = None,
    ) -> mx.array:
        output_dtype = dtype or _dtype_from_name(self.source_dtype)
        if not self.quantized:
            if self.data.dtype == output_dtype:
                return self.data
            return self.data.astype(output_dtype, stream=stream)
        assert self.scales is not None
        assert self.biases is not None
        return mx.dequantize(
            self.data,
            self.scales,
            self.biases,
            group_size=self.group_size,
            bits=self.bits,
            mode=self.mode,
            dtype=output_dtype,
            stream=stream,
        )

    def save(self, path: Path | str) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"data": self.data}
        if self.scales is not None:
            arrays["scales"] = self.scales
        if self.biases is not None:
            arrays["biases"] = self.biases
        metadata = {
            "format": "q-comem-residual",
            "version": str(STORE_FORMAT_VERSION),
            "depth": str(self.depth),
            "bits": str(self.bits),
            "group_size": str(self.group_size),
            "mode": self.mode,
            "original_shape": json.dumps(self.original_shape),
            "source_dtype": self.source_dtype,
        }
        self.eval()
        mx.save_safetensors(destination, arrays, metadata=metadata)
        return destination

    @classmethod
    def load(
        cls,
        path: Path | str,
        stream: mx.Stream | mx.Device | None = None,
    ) -> StoredResidual:
        # MLX file loading is a CPU operation. Arrays still live in unified
        # memory, so a later GPU dequantize can consume them without an H2D
        # copy; MLX inserts the cross-stream dependency automatically.
        arrays, metadata = mx.load(path, return_metadata=True, stream=mx.cpu)
        if metadata.get("format") != "q-comem-residual":
            raise ValueError("not a Q-CoMem residual store")
        if int(metadata.get("version", -1)) != STORE_FORMAT_VERSION:
            raise ValueError("unsupported Q-CoMem residual store version")
        return cls(
            depth=int(metadata["depth"]),
            bits=int(metadata["bits"]),
            group_size=int(metadata["group_size"]),
            mode=metadata["mode"],
            original_shape=tuple(json.loads(metadata["original_shape"])),
            source_dtype=metadata["source_dtype"],
            data=arrays["data"],
            scales=arrays.get("scales"),
            biases=arrays.get("biases"),
        )


def quantize_residual(
    residual: mx.array,
    *,
    depth: int,
    bits: int,
    group_size: int = 64,
    mode: str = "affine",
    stream: mx.Stream | mx.Device | None = None,
) -> StoredResidual:
    """Quantize and bit-pack a split residual using the MLX backend."""

    _validate_bits(bits)
    if residual.ndim < 2:
        raise ValueError("residual must have at least two dimensions")
    if residual.shape[-1] % group_size != 0:
        raise ValueError(
            f"hidden dimension {residual.shape[-1]} must be divisible by "
            f"group_size {group_size}"
        )
    source_dtype = _dtype_name(residual.dtype)
    if source_dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"residual must use a floating dtype, got {residual.dtype}")

    if bits == 16:
        dense = residual
        if residual.dtype not in {mx.float16, mx.bfloat16}:
            dense = residual.astype(mx.float16, stream=stream)
            source_dtype = "float16"
        return StoredResidual(
            depth=depth,
            bits=bits,
            group_size=group_size,
            mode="dense",
            original_shape=tuple(residual.shape),
            source_dtype=source_dtype,
            data=dense,
        )

    if mode != "affine":
        raise ValueError("the first Q-CoMem prototype supports affine mode only")
    packed, scales, biases = mx.quantize(
        residual,
        group_size=group_size,
        bits=bits,
        mode=mode,
        stream=stream,
    )
    return StoredResidual(
        depth=depth,
        bits=bits,
        group_size=group_size,
        mode=mode,
        original_shape=tuple(residual.shape),
        source_dtype=source_dtype,
        data=packed,
        scales=scales,
        biases=biases,
    )


def select_depth_bit_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_kl: float,
    max_relative_rmse: float | None = None,
    require_top1_match: bool = True,
    default_bits: int = 16,
) -> DepthBitPolicy:
    """Choose the smallest stored representation meeting calibration bounds."""

    by_depth: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_depth.setdefault(int(row["depth"]), []).append(row)

    assignments = {}
    for depth, candidates in by_depth.items():
        eligible = [
            row
            for row in candidates
            if float(row["kl_divergence"]) <= max_kl
            and (
                max_relative_rmse is None
                or float(row["residual_relative_rmse"]) <= max_relative_rmse
            )
            and (not require_top1_match or bool(row["top1_match"]))
        ]
        if eligible:
            best = min(
                eligible,
                key=lambda row: (int(row["stored_nbytes"]), int(row["bits"])),
            )
            assignments[depth] = int(best["bits"])
        else:
            assignments[depth] = default_bits
    return DepthBitPolicy(assignments=assignments, default_bits=default_bits)
