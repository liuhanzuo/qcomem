"""Frozen Falcon-H1 hybrid-state split adapter for the R39 transfer run.

The immutable A4 implementation remains the authority for lossless Q16
packing, cache cloning, and replay-state containers. This module supplies the
Falcon-H1-specific embedding multiplier, paired attention/Mamba masks, hybrid
decoder call signature, final normalization, LM-head multiplier, and complete
KV/conv/Mamba2 state-family census. It never implements a reference model.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Iterable, Sequence

import torch


A4_QCOMEM_SHA256 = "5901f153fcfcabbfab63f756a3c19a04ace56b4985fc02421f2dde4118a7373c"
OFFICIAL_MODELING_SHA256 = "e90bf774524e9b66284ad1c5528c35339271a187f58f16ba2d45c97f4bc6b5bd"
OFFICIAL_CACHE_UTILS_SHA256 = "ee7902fbd031ed332b5e26d07756a33f09b5c90a435b8363b9330876dc33ce0e"
OFFICIAL_MASKING_UTILS_SHA256 = "5f48e428ea02d1b6008acb45c147fcdb4eba89deea69627744662aa05da1b9f2"
EXPECTED_LAYER_TYPES = ("hybrid",) * 36
EXPECTED_DEPTH = 18
EXPECTED_VOCAB_SIZE = 32784
EXPECTED_HIDDEN_SIZE = 1024
EXPECTED_KV_HEADS = 2
EXPECTED_ATTN_HEAD_DIM = 64
EXPECTED_CONV_DIM = 1792
EXPECTED_CONV_KERNEL = 4
EXPECTED_MAMBA_HEADS = 24
EXPECTED_MAMBA_HEAD_DIM = 64
EXPECTED_MAMBA_STATE = 128
FAMILY_ORDER = ("kv_key", "kv_value", "conv", "mamba2_recurrent")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    import json

    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
    )


def _load_a4_module() -> tuple[ModuleType, Path]:
    evidence_root = Path(__file__).resolve().parents[2]
    path = (
        evidence_root
        / "round6_a4_transformers_transfer_20260819b"
        / "executed_source"
        / "qcomem_torch.py"
    )
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"immutable A4 dependency is absent: {path}")
    observed = _sha256_file(path)
    if observed != A4_QCOMEM_SHA256:
        raise RuntimeError(
            "immutable A4 qcomem_torch.py drift: "
            f"expected {A4_QCOMEM_SHA256}, observed {observed}"
        )
    module_name = "_r39_falcon_immutable_a4_qcomem_torch"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct immutable A4 module loader")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, path


_A4, A4_QCOMEM_PATH = _load_a4_module()

LowerReplayState = _A4.LowerReplayState
PackedLowerReplayState = _A4.PackedLowerReplayState
PackedCache = _A4.PackedCache
PackedResidual = _A4.PackedResidual
PackedTensor = _A4.PackedTensor
FullPrefixState = _A4.FullPrefixState
cache_nbytes = _A4.cache_nbytes
clone_cache = _A4.clone_cache


def force_official_naive_path() -> dict[str, Any]:
    """Disable optional fused Mamba dispatch without replacing official code."""

    if os.environ.get("USE_HUB_KERNELS", "").upper() != "NO":
        raise RuntimeError("USE_HUB_KERNELS=NO was not set before Transformers import")
    from transformers.models.falcon_h1 import modeling_falcon_h1

    if not hasattr(modeling_falcon_h1, "is_fast_path_available"):
        raise RuntimeError("official Falcon-H1 dispatch flag is absent")
    before = getattr(modeling_falcon_h1, "is_fast_path_available")
    if not isinstance(before, bool):
        raise RuntimeError("official Falcon-H1 dispatch flag is not boolean")
    modeling_path = Path(inspect.getsourcefile(modeling_falcon_h1) or "")
    if not modeling_path.is_file() or _sha256_file(modeling_path) != OFFICIAL_MODELING_SHA256:
        raise RuntimeError("official Transformers Falcon-H1 source SHA-256 drift")
    modeling_falcon_h1.is_fast_path_available = False
    return {
        "official_module": "transformers.models.falcon_h1.modeling_falcon_h1",
        "forced_dispatch_flag": "is_fast_path_available=False",
        "official_torch_method": "FalconH1Mixer.torch_forward",
        "pre_force_fast_path_available": before,
        "fast_path_observed_after_force": bool(modeling_falcon_h1.is_fast_path_available),
        "modeling_source_sha256": _sha256_file(modeling_path),
        "optional_module_presence": {
            name: importlib.util.find_spec(name) is not None
            for name in ("mamba_ssm", "causal_conv1d", "flash_attn")
        },
        "package_installs_optional_kernels": False,
        "use_hub_kernels_environment": os.environ["USE_HUB_KERNELS"],
    }


def verify_official_transformers_sources() -> dict[str, str]:
    from transformers import cache_utils, masking_utils
    from transformers.models.falcon_h1 import modeling_falcon_h1

    observed = {}
    for name, module, expected in (
        ("modeling_falcon_h1.py", modeling_falcon_h1, OFFICIAL_MODELING_SHA256),
        ("cache_utils.py", cache_utils, OFFICIAL_CACHE_UTILS_SHA256),
        ("masking_utils.py", masking_utils, OFFICIAL_MASKING_UTILS_SHA256),
    ):
        path = Path(inspect.getsourcefile(module) or "")
        if not path.is_file():
            raise RuntimeError(f"official Transformers source absent: {name}")
        digest = _sha256_file(path)
        if digest != expected:
            raise RuntimeError(
                f"official Transformers source drift for {name}: expected {expected}, observed {digest}"
            )
        observed[name] = digest
    return observed


def _tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous()
    if value.numel() == 0:
        return b""
    return value.view(torch.uint8).cpu().numpy().tobytes(order="C")


def _family_tensor_row(layer_index: int, family: str, tensor: torch.Tensor) -> dict[str, Any]:
    if not isinstance(tensor, torch.Tensor) or tensor.numel() == 0:
        raise RuntimeError(f"missing nonempty tensor for layer {layer_index} family {family}")
    if not tensor.is_contiguous():
        raise RuntimeError(f"non-contiguous tensor for layer {layer_index} family {family}")
    return {
        "layer_index": layer_index,
        "family": family,
        "state_index": 0 if family in {"conv", "mamba2_recurrent"} else None,
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "content_sha256": _sha256_bytes(_tensor_raw_bytes(tensor)),
    }


def _expected_family_shape(family: str, sequence_length: int) -> list[int]:
    if family in {"kv_key", "kv_value"}:
        return [1, EXPECTED_KV_HEADS, sequence_length, EXPECTED_ATTN_HEAD_DIM]
    if family == "conv":
        return [1, EXPECTED_CONV_DIM, EXPECTED_CONV_KERNEL]
    if family == "mamba2_recurrent":
        return [1, EXPECTED_MAMBA_HEADS, EXPECTED_MAMBA_HEAD_DIM, EXPECTED_MAMBA_STATE]
    raise AssertionError(family)


def cache_family_receipt(
    cache: Any,
    *,
    expected_active_layers: Sequence[int],
    expected_sequence_length: int,
) -> dict[str, Any]:
    """Census every mutable family and reject omissions or extra active layers."""

    layers = getattr(cache, "layers", None)
    if not isinstance(layers, list) or len(layers) != len(EXPECTED_LAYER_TYPES):
        raise RuntimeError("Falcon DynamicCache must expose exactly 36 allocated layers")
    active = tuple(int(index) for index in expected_active_layers)
    if len(set(active)) != len(active) or tuple(sorted(active)) != active:
        raise RuntimeError("active layer set is duplicated or unsorted")
    if any(index not in range(36) for index in active):
        raise RuntimeError("active layer outside registered geometry")
    active_set = set(active)
    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(layers):
        if layer_index not in active_set:
            if cache_nbytes(layer) != 0:
                raise RuntimeError(f"unexpected state in inactive layer {layer_index}")
            continue
        if set(getattr(layer, "conv_states", {})) != {0}:
            raise RuntimeError(f"conv state index drift at layer {layer_index}")
        if set(getattr(layer, "recurrent_states", {})) != {0}:
            raise RuntimeError(f"recurrent state index drift at layer {layer_index}")
        if getattr(layer, "is_conv_states_initialized", {}).get(0) is not True:
            raise RuntimeError(f"conv state uninitialized at layer {layer_index}")
        if getattr(layer, "is_recurrent_states_initialized", {}).get(0) is not True:
            raise RuntimeError(f"recurrent state uninitialized at layer {layer_index}")
        if getattr(layer, "has_previous_state", {}).get(0) is not True:
            raise RuntimeError(f"Mamba previous-state flag absent at layer {layer_index}")
        tensors = {
            "kv_key": getattr(layer, "keys", None),
            "kv_value": getattr(layer, "values", None),
            "conv": layer.conv_states[0],
            "mamba2_recurrent": layer.recurrent_states[0],
        }
        for family in FAMILY_ORDER:
            row = _family_tensor_row(layer_index, family, tensors[family])
            if row["shape"] != _expected_family_shape(family, expected_sequence_length):
                raise RuntimeError(
                    f"state shape drift at layer {layer_index} family {family}: {row['shape']}"
                )
            expected_dtype = "torch.float32" if family == "mamba2_recurrent" else "torch.bfloat16"
            if row["dtype"] != expected_dtype:
                raise RuntimeError(
                    f"state dtype drift at layer {layer_index} family {family}: {row['dtype']}"
                )
            rows.append(row)
    expected_pairs = [(layer, family) for layer in active for family in FAMILY_ORDER]
    observed_pairs = [(row["layer_index"], row["family"]) for row in rows]
    if observed_pairs != expected_pairs:
        raise RuntimeError("state-family census is incomplete, duplicated, or relabeled")
    return {
        "schema_version": "r39-falcon-h1-state-family-receipt-v1",
        "expected_active_layers": list(active),
        "expected_sequence_length": expected_sequence_length,
        "expected_family_count": len(expected_pairs),
        "observed_family_count": len(rows),
        "complete": True,
        "rows": rows,
        "rows_sha256": _sha256_bytes(_canonical_bytes(rows)),
    }


def composed_cache_family_receipt(
    lower_cache: Any,
    suffix_cache: Any,
    *,
    depth: int,
    expected_sequence_length: int,
) -> dict[str, Any]:
    if depth != EXPECTED_DEPTH:
        raise RuntimeError("Falcon split depth drift")
    lower = cache_family_receipt(
        lower_cache,
        expected_active_layers=tuple(range(depth)),
        expected_sequence_length=expected_sequence_length,
    )
    suffix = cache_family_receipt(
        suffix_cache,
        expected_active_layers=tuple(range(depth, 36)),
        expected_sequence_length=expected_sequence_length,
    )
    rows = sorted(
        [*lower["rows"], *suffix["rows"]],
        key=lambda row: (row["layer_index"], FAMILY_ORDER.index(row["family"])),
    )
    expected_pairs = [(layer, family) for layer in range(36) for family in FAMILY_ORDER]
    if [(row["layer_index"], row["family"]) for row in rows] != expected_pairs:
        raise RuntimeError("composed lower/suffix state families are not exactly complete")
    return {
        "schema_version": "r39-falcon-h1-composed-state-family-receipt-v1",
        "split_depth": depth,
        "expected_sequence_length": expected_sequence_length,
        "expected_family_count": 144,
        "observed_family_count": len(rows),
        "complete": True,
        "lower_rows_sha256": lower["rows_sha256"],
        "suffix_rows_sha256": suffix["rows_sha256"],
        "rows": rows,
        "rows_sha256": _sha256_bytes(_canonical_bytes(rows)),
    }


class TorchSplitFalconH1(_A4.TorchSplitCausalLM):
    """Manual Falcon-H1 split that follows the official 5.14.1 naive path."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__(model)
        self.official_source_sha256 = verify_official_transformers_sources()
        force = force_official_naive_path()
        if force["fast_path_observed_after_force"]:
            raise RuntimeError("failed to force official Falcon-H1 naive path")
        self.initial_naive_dispatch_receipt = force
        if not hasattr(self.language_model, "final_layernorm"):
            raise RuntimeError("Falcon-H1 final layer normalization is absent")

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.language_model.embed_tokens(tokens) * self.language_model.embedding_multiplier

    def _layer_context(
        self,
        hidden: torch.Tensor,
        *,
        past_key_values: Any = None,
        position_offset: int = 0,
        layer_start: int = 0,
    ) -> SimpleNamespace:
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

        length = int(hidden.shape[1])
        position_ids = torch.arange(
            position_offset,
            position_offset + length,
            device=hidden.device,
        ).unsqueeze(0)
        kwargs = {
            "config": self.config,
            "inputs_embeds": hidden,
            "attention_mask": None,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        return SimpleNamespace(
            masks={
                # A split suffix cache has history only in layers depth..35.
                # Binding mask sizing to layer_start is therefore semantic,
                # not cosmetic: the default layer 0 is empty for that cache.
                "full_attention": create_causal_mask(**kwargs, layer_idx=layer_start),
                "linear_attention": create_recurrent_attention_mask(**kwargs),
            },
            position_ids=position_ids,
            position_embeddings=self.language_model.rotary_emb(hidden, position_ids=position_ids),
        )

    def _run_layers(
        self,
        hidden: torch.Tensor,
        start: int,
        end: int,
        *,
        past_key_values: Any = None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        context = self._layer_context(
            hidden,
            past_key_values=past_key_values,
            position_offset=position_offset,
            layer_start=start,
        )
        for index in range(start, end):
            output = self.layers[index](
                hidden,
                attention_mask=context.masks["full_attention"],
                mamba_attention_mask=context.masks["linear_attention"],
                position_ids=context.position_ids,
                position_embeddings=context.position_embeddings,
                past_key_values=past_key_values,
                use_cache=past_key_values is not None,
            )
            hidden = output[0] if isinstance(output, tuple) else output
        return hidden

    def run_to_depth(self, tokens: torch.Tensor, depth: int) -> torch.Tensor:
        self._validate_depth(depth)
        tokens = self._batch_tokens(tokens)
        return self._run_layers(self._embed(tokens), 0, depth)

    @torch.inference_mode()
    def write_lower_replay(self, tokens: torch.Tensor, depth: int) -> LowerReplayState:
        self._validate_depth(depth)
        tokens = self._batch_tokens(tokens)
        if tokens.shape[1] == 0:
            raise ValueError("the replay document must contain at least one token")
        cache = self.make_cache()
        residual = self._run_layers(
            self._embed(tokens),
            0,
            depth,
            past_key_values=cache,
            position_offset=0,
        )
        length = int(tokens.shape[1])
        cache_family_receipt(
            cache,
            expected_active_layers=tuple(range(depth)),
            expected_sequence_length=length,
        )
        return LowerReplayState(
            depth=depth,
            document_length=length,
            current_length=length,
            document_residual=residual,
            cache=cache,
        )

    @torch.inference_mode()
    def continue_lower_replay(self, state: LowerReplayState, tokens: torch.Tensor) -> torch.Tensor:
        self._validate_depth(state.depth)
        tokens = self._batch_tokens(tokens)
        if tokens.shape[1] == 0:
            raise ValueError("replay continuation must contain at least one token")
        residual = self._run_layers(
            self._embed(tokens),
            0,
            state.depth,
            past_key_values=state.cache,
            position_offset=state.current_length,
        )
        state.current_length += int(tokens.shape[1])
        return residual

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = self.language_model.final_layernorm(hidden[:, -1:, :])
        return self.lm_head(normalized)[:, -1, :] * self.language_model.lm_head_multiplier

    def run_suffix_last_logits(self, residuals: Iterable[torch.Tensor], depth: int) -> torch.Tensor:
        self._validate_depth(depth)
        rows = list(residuals)
        if not rows:
            raise ValueError("at least one residual is required")
        hidden = self._run_layers(torch.cat(rows, dim=1), depth, self.num_layers)
        return self._logits(hidden)

    def run_suffix_cached_last_logits(
        self,
        residuals: Iterable[torch.Tensor],
        depth: int,
        cache: Any,
        *,
        position_offset: int,
    ) -> torch.Tensor:
        self._validate_depth(depth)
        rows = list(residuals)
        if not rows:
            raise ValueError("at least one residual is required")
        hidden = self._run_layers(
            torch.cat(rows, dim=1),
            depth,
            self.num_layers,
            past_key_values=cache,
            position_offset=position_offset,
        )
        return self._logits(hidden)

    def full_last_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self._batch_tokens(tokens)
        output = self.model(input_ids=tokens, use_cache=False, logits_to_keep=1)
        return output.logits[:, -1, :]

    @torch.inference_mode()
    def manual_one_shot_last_logits(self, tokens: torch.Tensor, depth: int) -> torch.Tensor:
        self._validate_depth(depth)
        tokens = self._batch_tokens(tokens)
        lower = self.run_to_depth(tokens, depth)
        return self.run_suffix_last_logits([lower], depth)

    def geometry_receipt(self) -> dict[str, Any]:
        layer_types = tuple(getattr(self.config, "layer_types", ()))
        classes = [f"{layer.__class__.__module__}.{layer.__class__.__qualname__}" for layer in self.layers]
        component_rows = [
            {
                "layer_index": index,
                "class": classes[index],
                "has_self_attention": hasattr(layer, "self_attn"),
                "has_mamba2_mixer": hasattr(layer, "mamba"),
                "has_feed_forward": hasattr(layer, "feed_forward"),
            }
            for index, layer in enumerate(self.layers)
        ]
        matches = (
            getattr(self.config, "model_type", None) == "falcon_h1"
            and self.num_layers == 36
            and getattr(self.config, "hidden_size", None) == EXPECTED_HIDDEN_SIZE
            and getattr(self.config, "vocab_size", None) == EXPECTED_VOCAB_SIZE
            and layer_types == EXPECTED_LAYER_TYPES
            and all(
                row["has_self_attention"] and row["has_mamba2_mixer"] and row["has_feed_forward"]
                for row in component_rows
            )
        )
        return {
            "model_type": getattr(self.config, "model_type", None),
            "num_hidden_layers": self.num_layers,
            "hidden_size": getattr(self.config, "hidden_size", None),
            "vocab_size": getattr(self.config, "vocab_size", None),
            "layer_types": list(layer_types),
            "components": component_rows,
            "split_depth": EXPECTED_DEPTH,
            "expected_state_families_per_layer": list(FAMILY_ORDER),
            "matches_registered": matches,
        }

    def dispatch_receipt(self) -> dict[str, Any]:
        from transformers import cache_utils, masking_utils
        from transformers.models.falcon_h1 import modeling_falcon_h1

        def source_receipt(value: Any) -> dict[str, str]:
            path = Path(inspect.getsourcefile(value) or "")
            if not path.is_file():
                raise RuntimeError(f"Python source is unavailable for {value!r}")
            return {"path": str(path), "sha256": _sha256_file(path)}

        if getattr(modeling_falcon_h1, "is_fast_path_available", None) is not False:
            raise RuntimeError("Falcon-H1 fast path changed after the initial force")
        naive = dict(self.initial_naive_dispatch_receipt)
        naive["fast_path_observed_at_dispatch_receipt"] = False
        return {
            "scope": "official_python_source_and_forced_naive_dispatch",
            "adapter_source": source_receipt(self.__class__),
            "a4_dependency_path": str(A4_QCOMEM_PATH),
            "a4_dependency_sha256": A4_QCOMEM_SHA256,
            "falcon_modeling_source": source_receipt(modeling_falcon_h1),
            "cache_utils_source": source_receipt(cache_utils),
            "masking_utils_source": source_receipt(masking_utils),
            "attention_implementation": getattr(self.config, "_attn_implementation", None),
            "mamba_dispatch": naive,
            "official_source_sha256": self.official_source_sha256,
            "compiled_kernel_binary_fingerprint": None,
            "hardware_instruction_trace": None,
            "exact_missingness": [
                "compiled CUDA/Triton kernel binary fingerprint",
                "hardware instruction trace",
            ],
        }


def registered_geometry_passes(layer_types: Sequence[str]) -> bool:
    return tuple(layer_types) == EXPECTED_LAYER_TYPES


__all__ = [
    "A4_QCOMEM_PATH",
    "A4_QCOMEM_SHA256",
    "EXPECTED_DEPTH",
    "EXPECTED_LAYER_TYPES",
    "FAMILY_ORDER",
    "FullPrefixState",
    "LowerReplayState",
    "PackedCache",
    "PackedLowerReplayState",
    "PackedResidual",
    "PackedTensor",
    "TorchSplitFalconH1",
    "cache_family_receipt",
    "cache_nbytes",
    "clone_cache",
    "composed_cache_family_receipt",
    "force_official_naive_path",
    "registered_geometry_passes",
    "verify_official_transformers_sources",
]
