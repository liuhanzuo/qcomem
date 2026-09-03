"""R39 dense-Qwen3.5 extension of the immutable A4 split adapter.

The A4 module remains the authority for packing, cache cloning, state classes,
and the tested split execution methods.  This file changes only hybrid mask
routing and adds explicit validation helpers for the dense ``qwen3_5_text``
backbone.  The dependency path and SHA-256 are frozen in the R39 source
manifest; the old evidence file is never rewritten.
"""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import torch


A4_QCOMEM_SHA256 = "5901f153fcfcabbfab63f756a3c19a04ace56b4985fc02421f2dde4118a7373c"
HYBRID_TEXT_MODEL_TYPES = frozenset({"qwen3_5_text", "qwen3_5_moe_text"})
EXPECTED_DENSE_LAYER_TYPES = tuple(
    item
    for _block in range(6)
    for item in ("linear_attention", "linear_attention", "linear_attention", "full_attention")
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_a4_module() -> tuple[ModuleType, Path]:
    evidence_root = Path(__file__).resolve().parents[2]
    path = (
        evidence_root
        / "round6_a4_transformers_transfer_20260819b"
        / "executed_source"
        / "qcomem_torch.py"
    )
    if not path.is_file():
        raise RuntimeError(f"immutable A4 dependency is absent: {path}")
    observed = _sha256_file(path)
    if observed != A4_QCOMEM_SHA256:
        raise RuntimeError(
            "immutable A4 qcomem_torch.py drift: "
            f"expected {A4_QCOMEM_SHA256}, observed {observed}"
        )
    module_name = "_r39_immutable_a4_qcomem_torch"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot construct A4 module loader")
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


class TorchSplitCausalLM(_A4.TorchSplitCausalLM):
    """A4 adapter with official hybrid-mask routing for dense Qwen3.5 text.

    The upstream A4 branch recognizes only ``qwen3_5_moe_text``.  Falling
    through its generic decoder path for ``qwen3_5_text`` creates only a
    ``full_attention`` mask and cannot route recurrent layers.  This override
    handles both official Qwen3.5 text config types and otherwise delegates to
    the unchanged A4 implementation.
    """

    def _layer_context(
        self,
        hidden: torch.Tensor,
        *,
        past_key_values: Any = None,
        position_offset: int = 0,
        layer_start: int = 0,
    ) -> SimpleNamespace:
        model_type = getattr(self.config, "model_type", "")
        if model_type not in HYBRID_TEXT_MODEL_TYPES:
            return super()._layer_context(
                hidden,
                past_key_values=past_key_values,
                position_offset=position_offset,
                layer_start=layer_start,
            )

        from transformers.masking_utils import (
            create_causal_mask,
            create_recurrent_attention_mask,
        )

        length = int(hidden.shape[1])
        batch = int(hidden.shape[0])
        device = hidden.device
        layer_types = tuple(self.config.layer_types)
        try:
            active_full_layer = next(
                index
                for index in range(layer_start, self.num_layers)
                if layer_types[index] == "full_attention"
            )
        except StopIteration:
            # The causal mask is unused when the selected range contains only
            # recurrent layers.  Use the first real full-attention layer so
            # DynamicCache never resolves sizing through an invented layer.
            active_full_layer = layer_types.index("full_attention")

        four_way_position_ids = torch.arange(
            position_offset,
            position_offset + length,
            device=device,
        ).view(1, 1, -1).expand(4, batch, -1)
        text_position_ids = four_way_position_ids[0]
        rotary_position_ids = four_way_position_ids[1:]
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": hidden,
            "attention_mask": None,
            "past_key_values": past_key_values,
            "position_ids": text_position_ids,
        }
        masks = {
            "full_attention": create_causal_mask(
                **mask_kwargs,
                layer_idx=active_full_layer,
            ),
            "linear_attention": create_recurrent_attention_mask(**mask_kwargs),
        }
        position_embeddings = self.language_model.rotary_emb(
            hidden,
            rotary_position_ids,
        )
        return SimpleNamespace(
            masks=masks,
            position_ids=text_position_ids,
            position_embeddings=position_embeddings,
            active_full_layer=active_full_layer,
        )

    @torch.inference_mode()
    def manual_one_shot_last_logits(
        self,
        tokens: torch.Tensor,
        depth: int,
    ) -> torch.Tensor:
        """Traverse the same full token sequence across the manual boundary."""

        self._validate_depth(depth)
        tokens = self._batch_tokens(tokens)
        lower = self.run_to_depth(tokens, depth)
        return self.run_suffix_last_logits([lower], depth)

    def dense_geometry_receipt(self) -> dict[str, Any]:
        layer_types = tuple(getattr(self.config, "layer_types", ()))
        layer_class_names = [
            f"{layer.__class__.__module__}.{layer.__class__.__qualname__}"
            for layer in self.layers
        ]
        is_dense = not any(
            "moe" in name.lower()
            or hasattr(layer, "mlp") and hasattr(layer.mlp, "experts")
            for name, layer in zip(layer_class_names, self.layers)
        )
        return {
            "model_type": getattr(self.config, "model_type", None),
            "num_hidden_layers": self.num_layers,
            "hidden_size": getattr(self.config, "hidden_size", None),
            "vocab_size": getattr(self.config, "vocab_size", None),
            "full_attention_interval": getattr(
                self.config,
                "full_attention_interval",
                None,
            ),
            "layer_types": list(layer_types),
            "layer_class_names": layer_class_names,
            "dense_no_experts": is_dense,
            "matches_registered": (
                getattr(self.config, "model_type", None) == "qwen3_5_text"
                and self.num_layers == 24
                and getattr(self.config, "hidden_size", None) == 1024
                and getattr(self.config, "vocab_size", None) == 248320
                and layer_types == EXPECTED_DENSE_LAYER_TYPES
                and is_dense
            ),
        }

    def dispatch_receipt(self) -> dict[str, Any]:
        from transformers import masking_utils

        def source_receipt(value: Any) -> dict[str, str]:
            path = Path(inspect.getsourcefile(value) or "")
            if not path.is_file():
                raise RuntimeError(f"Python source is unavailable for {value!r}")
            return {"path": str(path), "sha256": _sha256_file(path)}

        layer_sources: dict[str, dict[str, str]] = {}
        for layer in self.layers:
            class_name = f"{layer.__class__.__module__}.{layer.__class__.__qualname__}"
            layer_sources[class_name] = source_receipt(layer.__class__)

        return {
            "scope": "partial_python_source_and_class_provenance",
            "adapter_class": (
                f"{self.__class__.__module__}.{self.__class__.__qualname__}"
            ),
            "adapter_source": source_receipt(self.__class__),
            "a4_dependency_path": str(A4_QCOMEM_PATH),
            "a4_dependency_sha256": A4_QCOMEM_SHA256,
            "masking_utils_source": source_receipt(masking_utils),
            "layer_forward_types": sorted(
                {
                    f"{layer.__class__.__module__}.{layer.__class__.__qualname__}"
                    for layer in self.layers
                }
            ),
            "layer_sources": [
                {"class": class_name, **layer_sources[class_name]}
                for class_name in sorted(layer_sources)
            ],
            "mask_routes": {
                "full_attention": "transformers.masking_utils.create_causal_mask",
                "linear_attention": (
                    "transformers.masking_utils.create_recurrent_attention_mask"
                ),
            },
            "compiled_kernel_binary_fingerprint": None,
            "autotuning_choice_fingerprint": None,
            "hardware_instruction_trace": None,
            "exact_missingness": [
                "compiled CUDA/Triton kernel binary fingerprint",
                "kernel autotuning-choice fingerprint",
                "hardware instruction trace",
            ],
        }


def registered_layer_route_passes(layer_types: tuple[str, ...] | list[str]) -> bool:
    """Pure predicate used by both the rank producer and detached controls."""

    return tuple(layer_types) == EXPECTED_DENSE_LAYER_TYPES


__all__ = [
    "A4_QCOMEM_PATH",
    "A4_QCOMEM_SHA256",
    "EXPECTED_DENSE_LAYER_TYPES",
    "FullPrefixState",
    "LowerReplayState",
    "PackedCache",
    "PackedLowerReplayState",
    "PackedResidual",
    "PackedTensor",
    "TorchSplitCausalLM",
    "cache_nbytes",
    "clone_cache",
    "registered_layer_route_passes",
]
