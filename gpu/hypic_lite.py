from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

from qcomem_torch import (
    LowerReplayState,
    PackedLowerReplayState,
    TorchSplitCausalLM,
    cache_nbytes,
    tensor_nbytes,
)


HYPIC_LITE_CONFIGS = (
    "hypic-lite-naive-w0",
    "hypic-lite-naive-w8",
    "hypic-lite-transition-w0",
    "hypic-lite-transition-w8",
)


@dataclass(frozen=True)
class HypicLiteConfig:
    name: str
    composition: str
    seam_width: int
    transition_dtype: torch.dtype = torch.bfloat16

    @property
    def uses_transition(self) -> bool:
        return self.composition == "transition"


def parse_hypic_lite_config(name: str) -> HypicLiteConfig:
    fields = name.split("-")
    if len(fields) != 4 or fields[:2] != ["hypic", "lite"]:
        raise ValueError(f"invalid HYPIC-lite config: {name}")
    composition = fields[2]
    if composition not in {"naive", "transition"}:
        raise ValueError(f"unknown state composition: {composition}")
    if not fields[3].startswith("w"):
        raise ValueError(f"missing seam width: {name}")
    seam_width = int(fields[3][1:])
    if seam_width not in {0, 8}:
        raise ValueError("the preregistered HYPIC-lite seam widths are 0 and 8")
    return HypicLiteConfig(
        name=name,
        composition=composition,
        seam_width=seam_width,
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(operation: Callable[[], Any]) -> tuple[Any, float]:
    _sync()
    started = time.perf_counter()
    result = operation()
    _sync()
    return result, time.perf_counter() - started


def _slot(container: Any, index: int = 0) -> Any:
    if isinstance(container, dict):
        return container[index]
    return container[index]


def _set_slot(container: Any, value: Any, index: int = 0) -> None:
    if isinstance(container, dict):
        container[index] = value
    else:
        container[index] = value


def _is_linear_cache_layer(layer: Any) -> bool:
    return hasattr(layer, "recurrent_states") or hasattr(layer, "conv_states")


def _is_full_attention_cache_layer(layer: Any) -> bool:
    return hasattr(layer, "keys") and hasattr(layer, "values")


def compose_affine_state(
    running_state: torch.Tensor,
    zero_start_end_state: torch.Tensor,
    transition: torch.Tensor | None,
) -> torch.Tensor:
    """Compose ``S_out = T_C S_in + S_{C|0}`` or the naive-addition ablation."""
    if transition is None:
        return running_state + zero_start_end_state.to(running_state)
    transformed = torch.matmul(
        transition.to(device=running_state.device, dtype=torch.float32),
        running_state.float(),
    )
    return (transformed + zero_start_end_state.float()).to(running_state.dtype)


@dataclass
class CapturedAffine:
    zero_start_end_state: torch.Tensor
    transition: torch.Tensor | None
    transition_max_abs_error: float | None
    transition_relative_l2_error: float | None


class _GatedDeltaAffineCapture:
    """Reference hook around the real Qwen3.5 GatedDeltaRule kernel.

    Transformers exposes only the final recurrent state in ``DynamicCache``.
    The transient ``k/g/beta`` values required for the cumulative transition
    are available only inside ``Qwen3_5*MoeGatedDeltaNet.forward``.  This hook
    wraps that module's actual chunk kernel and obtains ``T_C`` by invoking the
    same kernel with ``S0=I`` and a zero value/write stream, exactly matching
    the construction described by HYPIC.  It is intentionally an internal-API
    reference path, not a production claim; a fused serving kernel should emit
    ``(T_C, S_C)`` in one prefill integration.
    """

    def __init__(
        self,
        adapter: TorchSplitCausalLM,
        *,
        depth: int,
        body_start: int,
        capture_transition: bool,
        transition_dtype: torch.dtype,
        validate_transition: bool,
    ) -> None:
        self.adapter = adapter
        self.depth = depth
        self.body_start = body_start
        self.capture_transition = capture_transition
        self.transition_dtype = transition_dtype
        self.validate_transition = validate_transition
        self.affines: dict[int, CapturedAffine] = {}
        # (object, attribute name, original callable).  In the pinned target
        # build the FLA rule is an ``nn.Module``; replacing that registered
        # child with a plain function is invalid, so hook ``forward`` instead.
        self._originals: list[tuple[Any, str, Callable[..., Any]]] = []

    @staticmethod
    def _kernel_call(
        kernel: Callable[..., Any],
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor | None,
        inherited_kwargs: dict[str, Any],
    ) -> torch.Tensor:
        kwargs = dict(inherited_kwargs)
        kwargs.pop("initial_state", None)
        kwargs.pop("output_final_state", None)
        # A sliced seam/body is an ordinary dense sequence. Packed-sequence
        # offsets from the original call no longer describe it.
        kwargs.pop("cu_seqlens", None)
        kwargs.pop("cu_seq_lens_q", None)
        _, final_state = kernel(
            query,
            key,
            value,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=True,
            **kwargs,
        )
        if final_state is None:
            raise RuntimeError("GatedDeltaRule kernel did not return a final state")
        return final_state

    def _capture(
        self,
        layer_index: int,
        kernel: Callable[..., Any],
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        kwargs: dict[str, Any],
    ) -> None:
        start = self.body_start
        query = query[:, start:]
        key = key[:, start:]
        value = value[:, start:]
        g = g[:, start:]
        beta = beta[:, start:]
        if query.shape[1] == 0:
            raise ValueError("cannot capture an affine for an empty segment body")

        zero_state = self._kernel_call(
            kernel,
            query=query,
            key=key,
            value=value,
            g=g,
            beta=beta,
            initial_state=None,
            inherited_kwargs=kwargs,
        ).detach()
        transition = None
        max_abs = relative_l2 = None
        if self.capture_transition:
            batch, _, heads, key_dim = key.shape
            value_dim = value.shape[-1]
            if value_dim != key_dim:
                raise RuntimeError(
                    "identity-kernel transition extraction currently requires "
                    f"key_dim == value_dim, got {key_dim} and {value_dim}; a "
                    "dedicated transition-emission kernel is required otherwise"
                )
            identity = torch.eye(
                key_dim, device=key.device, dtype=torch.float32
            ).view(1, 1, key_dim, key_dim)
            identity = identity.expand(batch, heads, key_dim, key_dim).clone()
            zero_values = torch.zeros(
                (*key.shape[:-1], key_dim),
                device=key.device,
                dtype=value.dtype,
            )
            transition_fp32 = self._kernel_call(
                kernel,
                query=query,
                key=key,
                value=zero_values,
                g=g,
                beta=beta,
                initial_state=identity,
                inherited_kwargs=kwargs,
            ).detach()
            transition = transition_fp32.to(self.transition_dtype)

            if self.validate_transition:
                probe = torch.full_like(zero_state, 0.01)
                expected = self._kernel_call(
                    kernel,
                    query=query,
                    key=key,
                    value=value,
                    g=g,
                    beta=beta,
                    initial_state=probe,
                    inherited_kwargs=kwargs,
                ).detach()
                candidate = compose_affine_state(probe, zero_state, transition)
                difference = candidate.float() - expected.float()
                max_abs = float(difference.abs().max().item())
                relative_l2 = float(
                    torch.linalg.vector_norm(difference).item()
                    / max(torch.linalg.vector_norm(expected.float()).item(), 1e-30)
                )

        if layer_index in self.affines:
            raise RuntimeError(
                f"linear layer {layer_index} invoked more than once during segment prefill"
            )
        self.affines[layer_index] = CapturedAffine(
            zero_start_end_state=zero_state,
            transition=transition,
            transition_max_abs_error=max_abs,
            transition_relative_l2_error=relative_l2,
        )

    def __enter__(self) -> "_GatedDeltaAffineCapture":
        layer_types = getattr(self.adapter.config, "layer_types", ())
        for layer_index in range(self.depth, self.adapter.num_layers):
            if layer_types and layer_types[layer_index] != "linear_attention":
                continue
            mixer = getattr(self.adapter.layers[layer_index], "linear_attn", None)
            if mixer is None:
                continue
            kernel_module = getattr(mixer, "chunk_gated_delta_rule", None)
            if kernel_module is None:
                raise RuntimeError(
                    f"linear layer {layer_index} does not expose chunk_gated_delta_rule"
                )
            if hasattr(kernel_module, "forward"):
                owner = kernel_module
                attribute = "forward"
                kernel = kernel_module.forward
            else:
                # Compatibility path for older builds and CPU fakes.
                owner = mixer
                attribute = "chunk_gated_delta_rule"
                kernel = kernel_module

            def wrapped(
                query,
                key,
                value,
                g,
                beta,
                *args,
                _layer_index=layer_index,
                _kernel=kernel,
                **kwargs,
            ):
                if args:
                    raise RuntimeError(
                        "unsupported positional GatedDeltaRule kernel arguments"
                    )
                output = _kernel(
                    query,
                    key,
                    value,
                    g=g,
                    beta=beta,
                    **kwargs,
                )
                self._capture(
                    _layer_index,
                    _kernel,
                    query,
                    key,
                    value,
                    g,
                    beta,
                    kwargs,
                )
                return output

            self._originals.append((owner, attribute, kernel))
            setattr(owner, attribute, wrapped)
        if not self._originals:
            raise RuntimeError(
                "no suffix GatedDeltaNet chunk kernels are visible; "
                "transition composition is unavailable"
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        for owner, attribute, kernel in self._originals:
            setattr(owner, attribute, kernel)
        self._originals.clear()


@dataclass
class HypicLiteSegment:
    start: int
    length: int
    seam_tokens: int
    body_cache: Any
    transitions: dict[int, torch.Tensor]
    transition_validation: dict[int, dict[str, float | None]]
    full_attention_kv_nbytes: int
    linear_end_state_nbytes: int
    conv_tail_nbytes: int
    transition_nbytes: int

    @property
    def body_start(self) -> int:
        return self.start + self.seam_tokens

    @property
    def body_tokens(self) -> int:
        return self.length - self.seam_tokens

    @property
    def persistent_nbytes(self) -> int:
        return (
            self.full_attention_kv_nbytes
            + self.linear_end_state_nbytes
            + self.conv_tail_nbytes
            + self.transition_nbytes
        )


@dataclass
class HypicLiteStore:
    config: HypicLiteConfig
    depth: int
    document_length: int
    suffix_layers: int
    suffix_full_attention_layers: int
    suffix_linear_layers: int
    segments: list[HypicLiteSegment]
    base_document_nbytes: int
    build_seconds: float
    approximation_notes: tuple[str, ...] = (
        "independent segment suffix hidden states omit cross-segment context",
        "full-attention body KV is naively spliced at preassigned global positions",
        "transition capture uses a Transformers/FLA internal kernel hook",
    )

    @property
    def seam_tokens_recomputed(self) -> int:
        return sum(segment.seam_tokens for segment in self.segments)

    @property
    def full_attention_kv_nbytes(self) -> int:
        return sum(segment.full_attention_kv_nbytes for segment in self.segments)

    @property
    def linear_end_state_nbytes(self) -> int:
        return sum(segment.linear_end_state_nbytes for segment in self.segments)

    @property
    def conv_tail_nbytes(self) -> int:
        return sum(segment.conv_tail_nbytes for segment in self.segments)

    @property
    def transition_nbytes(self) -> int:
        return sum(segment.transition_nbytes for segment in self.segments)

    @property
    def suffix_store_nbytes(self) -> int:
        return sum(segment.persistent_nbytes for segment in self.segments)

    def bytes_ledger(self) -> dict[str, Any]:
        seam_kv_budget = estimate_seam_kv_budget(self)
        transition_only = (
            self.linear_end_state_nbytes
            + self.conv_tail_nbytes
            + self.transition_nbytes
        )
        return {
            "base_qcomem_document_nbytes": self.base_document_nbytes,
            "suffix_linear_zero_start_end_state_nbytes": self.linear_end_state_nbytes,
            "suffix_linear_transition_nbytes": self.transition_nbytes,
            "suffix_conv_tail_nbytes": self.conv_tail_nbytes,
            "suffix_full_attention_local_kv_nbytes": self.full_attention_kv_nbytes,
            "suffix_seam_kv_budget_nbytes": seam_kv_budget,
            "profiles": {
                "qcomem_only": {
                    "persistent_nbytes": self.base_document_nbytes,
                    "executable_for_suffix_ttft": True,
                    "online_suffix_document_rebuild_required": True,
                },
                "linear_transition_only": {
                    "persistent_nbytes": self.base_document_nbytes + transition_only,
                    "incremental_nbytes": transition_only,
                    "executable_for_suffix_ttft": self.suffix_full_attention_layers == 0,
                    "limitation": (
                        None
                        if self.suffix_full_attention_layers == 0
                        else "interleaved full-attention layers still need document KV"
                    ),
                },
                "linear_transition_plus_seam_kv": {
                    "persistent_nbytes": (
                        self.base_document_nbytes + transition_only + seam_kv_budget
                    ),
                    "incremental_nbytes": transition_only + seam_kv_budget,
                    "executable_for_suffix_ttft": self.suffix_full_attention_layers == 0,
                    "limitation": (
                        None
                        if self.suffix_full_attention_layers == 0
                        else "seam-only KV cannot provide full-document causal lookback"
                    ),
                },
                "full_suffix_local_cache": {
                    "persistent_nbytes": self.base_document_nbytes
                    + self.suffix_store_nbytes,
                    "incremental_nbytes": self.suffix_store_nbytes,
                    "executable_for_suffix_ttft": True,
                    "approximate": True,
                },
            },
        }

    def work_ledger(self) -> dict[str, int | float]:
        baseline = self.document_length * self.suffix_layers
        seam = self.seam_tokens_recomputed * self.suffix_layers
        saved = baseline - seam
        body_segments = sum(segment.body_tokens > 0 for segment in self.segments)
        return {
            "qcomem_rebuild_suffix_document_token_layer_forwards": baseline,
            "hypic_lite_online_seam_token_layer_forwards": seam,
            "saved_suffix_document_token_layer_forwards": saved,
            "saved_fraction": saved / baseline if baseline else 0.0,
            "online_linear_state_compositions": body_segments
            * self.suffix_linear_layers,
            "online_full_attention_kv_splices": body_segments
            * self.suffix_full_attention_layers,
        }


def estimate_seam_kv_budget(store: HypicLiteStore) -> int:
    body_tokens = sum(segment.body_tokens for segment in store.segments)
    if body_tokens == 0 or store.full_attention_kv_nbytes == 0:
        return 0
    bytes_per_token = store.full_attention_kv_nbytes / body_tokens
    return round(bytes_per_token * store.seam_tokens_recomputed)


def even_segment_lengths(total_tokens: int, segment_count: int) -> tuple[int, ...]:
    if total_tokens < 1:
        raise ValueError("document must contain at least one token")
    if segment_count < 1 or segment_count > total_tokens:
        raise ValueError("segment_count must be within [1, total_tokens]")
    quotient, remainder = divmod(total_tokens, segment_count)
    return tuple(
        quotient + (index < remainder) for index in range(segment_count)
    )


def _trim_and_measure_segment_cache(
    cache: Any,
    *,
    depth: int,
    seam_tokens: int,
    affines: dict[int, CapturedAffine],
) -> tuple[int, int, int]:
    full_kv = linear_state = conv_tail = 0
    for layer_index in range(depth, len(cache.layers)):
        layer = cache.layers[layer_index]
        if _is_full_attention_cache_layer(layer):
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if isinstance(keys, torch.Tensor) and keys.numel():
                layer.keys = keys[..., seam_tokens:, :].detach().clone()
                layer.values = values[..., seam_tokens:, :].detach().clone()
                full_kv += tensor_nbytes(layer.keys) + tensor_nbytes(layer.values)
        if _is_linear_cache_layer(layer):
            affine = affines.get(layer_index)
            if affine is None:
                continue
            recurrent = affine.zero_start_end_state.detach().clone()
            _set_slot(layer.recurrent_states, recurrent)
            linear_state += tensor_nbytes(recurrent)
            conv = _slot(layer.conv_states)
            if isinstance(conv, torch.Tensor):
                conv_tail += tensor_nbytes(conv)
    return full_kv, linear_state, conv_tail


@torch.inference_mode()
def build_hypic_lite_store(
    adapter: TorchSplitCausalLM,
    document_state: LowerReplayState | PackedLowerReplayState,
    config: HypicLiteConfig,
    *,
    segment_lengths: Sequence[int],
    validate_transitions: bool = True,
) -> HypicLiteStore:
    """Build independently reusable suffix segment caches outside request TTFT."""
    if sum(segment_lengths) != document_state.document_length:
        raise ValueError("segment lengths must exactly cover the document")
    if any(length < 1 for length in segment_lengths):
        raise ValueError("segments must be non-empty")
    depth = document_state.depth
    local_document = document_state.fork()
    residual = local_document.document_residual
    layer_types = getattr(adapter.config, "layer_types", ())
    suffix_types = tuple(layer_types[depth:])
    full_layers = suffix_types.count("full_attention")
    linear_layers = suffix_types.count("linear_attention")
    if full_layers + linear_layers != adapter.num_layers - depth:
        raise ValueError("unsupported suffix layer types")

    _sync()
    started = time.perf_counter()
    segments = []
    offset = 0
    for segment_index, length in enumerate(segment_lengths):
        seam = 0 if segment_index == 0 else min(config.seam_width, length)
        body_tokens = length - seam
        segment_residual = residual[:, offset : offset + length]
        local_cache = adapter.make_cache()
        transitions: dict[int, torch.Tensor] = {}
        validation: dict[int, dict[str, float | None]] = {}

        if body_tokens:
            with _GatedDeltaAffineCapture(
                adapter,
                depth=depth,
                body_start=seam,
                capture_transition=config.uses_transition,
                transition_dtype=config.transition_dtype,
                validate_transition=validate_transitions and config.uses_transition,
            ) as capture:
                adapter.run_suffix_cached_last_logits(
                    [segment_residual],
                    depth,
                    local_cache,
                    position_offset=offset,
                )
            expected_linear = {
                index
                for index in range(depth, adapter.num_layers)
                if layer_types[index] == "linear_attention"
            }
            if set(capture.affines) != expected_linear:
                raise RuntimeError(
                    "did not capture every suffix linear layer: "
                    f"expected={sorted(expected_linear)} actual={sorted(capture.affines)}"
                )
            transitions = {
                index: affine.transition
                for index, affine in capture.affines.items()
                if affine.transition is not None
            }
            validation = {
                index: {
                    "max_abs_error": affine.transition_max_abs_error,
                    "relative_l2_error": affine.transition_relative_l2_error,
                }
                for index, affine in capture.affines.items()
                if affine.transition is not None
            }
            full_kv, linear_state, conv_tail = _trim_and_measure_segment_cache(
                local_cache,
                depth=depth,
                seam_tokens=seam,
                affines=capture.affines,
            )
        else:
            full_kv = linear_state = conv_tail = 0

        transition_bytes = sum(tensor_nbytes(value) for value in transitions.values())
        segments.append(
            HypicLiteSegment(
                start=offset,
                length=length,
                seam_tokens=seam,
                body_cache=local_cache,
                transitions=transitions,
                transition_validation=validation,
                full_attention_kv_nbytes=full_kv,
                linear_end_state_nbytes=linear_state,
                conv_tail_nbytes=conv_tail,
                transition_nbytes=transition_bytes,
            )
        )
        offset += length
    _sync()
    return HypicLiteStore(
        config=config,
        depth=depth,
        document_length=document_state.document_length,
        suffix_layers=adapter.num_layers - depth,
        suffix_full_attention_layers=full_layers,
        suffix_linear_layers=linear_layers,
        segments=segments,
        base_document_nbytes=document_state.stored_nbytes,
        build_seconds=time.perf_counter() - started,
    )


def _append_body_cache(
    assembled: Any,
    segment: HypicLiteSegment,
    *,
    depth: int,
    use_transition: bool,
) -> None:
    for layer_index in range(depth, len(segment.body_cache.layers)):
        source = segment.body_cache.layers[layer_index]
        if _is_full_attention_cache_layer(source):
            keys = getattr(source, "keys", None)
            values = getattr(source, "values", None)
            if isinstance(keys, torch.Tensor) and keys.numel():
                assembled.update(keys, values, layer_index)
        if _is_linear_cache_layer(source):
            zero_state = _slot(source.recurrent_states)
            if not isinstance(zero_state, torch.Tensor):
                continue
            target = assembled.layers[layer_index]
            existing = _slot(target.recurrent_states)
            if not isinstance(existing, torch.Tensor):
                existing = torch.zeros_like(zero_state)
            transition = segment.transitions.get(layer_index) if use_transition else None
            composed = compose_affine_state(existing, zero_state, transition)
            assembled.update_recurrent_state(composed, layer_index)
            conv = _slot(source.conv_states)
            if isinstance(conv, torch.Tensor):
                assembled.update_conv_state(conv, layer_index)


@torch.inference_mode()
def assemble_suffix_cache(
    adapter: TorchSplitCausalLM,
    store: HypicLiteStore,
    document_residual: torch.Tensor,
) -> Any:
    """Assemble the approximate private suffix cache for one request."""
    assembled = adapter.make_cache()
    for segment in store.segments:
        if segment.seam_tokens:
            seam = document_residual[
                :, segment.start : segment.start + segment.seam_tokens
            ]
            adapter.run_suffix_cached_last_logits(
                [seam],
                store.depth,
                assembled,
                position_offset=segment.start,
            )
        if segment.body_tokens:
            _append_body_cache(
                assembled,
                segment,
                depth=store.depth,
                use_transition=store.config.uses_transition,
            )
    return assembled


@dataclass
class FirstTokenTrace:
    logits: torch.Tensor
    ttft_seconds: float
    fork_seconds: float
    lower_query_seconds: float
    suffix_assembly_seconds: float
    suffix_query_seconds: float
    private_suffix_cache_nbytes: int

    def summary(self) -> dict[str, Any]:
        return {
            "ttft_seconds": self.ttft_seconds,
            "fork_seconds": self.fork_seconds,
            "lower_query_seconds": self.lower_query_seconds,
            "suffix_assembly_seconds": self.suffix_assembly_seconds,
            "suffix_query_seconds": self.suffix_query_seconds,
            "private_suffix_cache_nbytes": self.private_suffix_cache_nbytes,
        }


@torch.inference_mode()
def hypic_lite_first_token(
    adapter: TorchSplitCausalLM,
    store: HypicLiteStore,
    document_state: LowerReplayState | PackedLowerReplayState,
    query_tokens: torch.Tensor,
) -> FirstTokenTrace:
    """Measure request-time assembly and first-token logits."""
    request_started = time.perf_counter()
    local, fork_seconds = _timed(document_state.fork)
    query_residual, lower_seconds = _timed(
        lambda: adapter.continue_lower_replay(local, query_tokens)
    )
    suffix_cache, assembly_seconds = _timed(
        lambda: assemble_suffix_cache(adapter, store, local.document_residual)
    )
    logits, suffix_query_seconds = _timed(
        lambda: adapter.run_suffix_cached_last_logits(
            [query_residual],
            store.depth,
            suffix_cache,
            position_offset=store.document_length,
        )
    )
    _sync()
    return FirstTokenTrace(
        logits=logits,
        ttft_seconds=time.perf_counter() - request_started,
        fork_seconds=fork_seconds,
        lower_query_seconds=lower_seconds,
        suffix_assembly_seconds=assembly_seconds,
        suffix_query_seconds=suffix_query_seconds,
        private_suffix_cache_nbytes=cache_nbytes(suffix_cache),
    )


@torch.inference_mode()
def qcomem_rebuild_first_token(
    adapter: TorchSplitCausalLM,
    document_state: LowerReplayState | PackedLowerReplayState,
    query_tokens: torch.Tensor,
) -> FirstTokenTrace:
    """Current exact Q-CoMem path: rebuild every suffix document position."""
    request_started = time.perf_counter()
    local, fork_seconds = _timed(document_state.fork)
    query_residual, lower_seconds = _timed(
        lambda: adapter.continue_lower_replay(local, query_tokens)
    )
    suffix_cache = adapter.make_cache()
    _, rebuild_seconds = _timed(
        lambda: adapter.run_suffix_cached_last_logits(
            [local.document_residual],
            document_state.depth,
            suffix_cache,
            position_offset=0,
        )
    )
    logits, suffix_query_seconds = _timed(
        lambda: adapter.run_suffix_cached_last_logits(
            [query_residual],
            document_state.depth,
            suffix_cache,
            position_offset=document_state.document_length,
        )
    )
    _sync()
    return FirstTokenTrace(
        logits=logits,
        ttft_seconds=time.perf_counter() - request_started,
        fork_seconds=fork_seconds,
        lower_query_seconds=lower_seconds,
        suffix_assembly_seconds=rebuild_seconds,
        suffix_query_seconds=suffix_query_seconds,
        private_suffix_cache_nbytes=cache_nbytes(suffix_cache),
    )


def logit_comparison(
    reference: torch.Tensor, candidate: torch.Tensor
) -> dict[str, float | bool | int]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    difference = candidate_float - reference_float
    probability = torch.softmax(reference_float, dim=-1)
    kl = F.kl_div(
        torch.log_softmax(candidate_float, dim=-1),
        probability,
        reduction="batchmean",
    )
    reference_token = int(torch.argmax(reference_float, dim=-1).item())
    candidate_token = int(torch.argmax(candidate_float, dim=-1).item())
    return {
        "top1_match": reference_token == candidate_token,
        "reference_token": reference_token,
        "candidate_token": candidate_token,
        "max_abs_logit_error": float(difference.abs().max().item()),
        "relative_logit_l2_error": float(
            torch.linalg.vector_norm(difference).item()
            / max(torch.linalg.vector_norm(reference_float).item(), 1e-30)
        ),
        "kl_divergence": max(float(kl.item()), 0.0),
    }


def transition_validation_summary(store: HypicLiteStore) -> dict[str, Any]:
    rows = [
        metrics
        for segment in store.segments
        for metrics in segment.transition_validation.values()
    ]
    relative = [
        float(row["relative_l2_error"])
        for row in rows
        if row["relative_l2_error"] is not None
    ]
    maximum = [
        float(row["max_abs_error"])
        for row in rows
        if row["max_abs_error"] is not None
    ]
    return {
        "validated_layer_segments": len(relative),
        "max_relative_l2_error": max(relative, default=None),
        "mean_relative_l2_error": statistics.fmean(relative) if relative else None,
        "max_abs_error": max(maximum, default=None),
        "semantic": (
            "T_C @ probe + S_C|0 versus the same GatedDeltaRule kernel "
            "continued from probe; BF16 transition storage included"
        ),
    }


def model_suffix_storage_estimate(
    model_config: Any,
    *,
    depth: int,
    document_tokens: int,
    segment_count: int,
    seam_width: int,
    state_bytes: int = 4,
    cache_bytes: int = 2,
    transition_bytes: int = 2,
) -> dict[str, Any]:
    """Analytic bytes boundary using the real hybrid layer layout and shapes."""
    layer_types = tuple(model_config.layer_types)
    suffix_types = layer_types[depth:]
    full_layers = suffix_types.count("full_attention")
    linear_layers = suffix_types.count("linear_attention")
    batch = 1
    full_kv_elements_per_token_layer = (
        2
        * batch
        * int(model_config.num_key_value_heads)
        * int(model_config.head_dim)
    )
    recurrent_elements_per_layer = (
        batch
        * int(model_config.linear_num_value_heads)
        * int(model_config.linear_key_head_dim)
        * int(model_config.linear_value_head_dim)
    )
    transition_elements_per_layer = (
        batch
        * int(model_config.linear_num_value_heads)
        * int(model_config.linear_key_head_dim) ** 2
    )
    conv_elements_per_layer_segment = (
        batch
        * (
            2
            * int(model_config.linear_num_key_heads)
            * int(model_config.linear_key_head_dim)
            + int(model_config.linear_num_value_heads)
            * int(model_config.linear_value_head_dim)
        )
        * int(model_config.linear_conv_kernel_dim)
    )
    seam_tokens = max(segment_count - 1, 0) * seam_width
    body_tokens = max(document_tokens - seam_tokens, 0)
    full_kv_elements = full_layers * body_tokens * full_kv_elements_per_token_layer
    recurrent_elements = (
        linear_layers * segment_count * recurrent_elements_per_layer
    )
    transition_elements = (
        linear_layers * segment_count * transition_elements_per_layer
    )
    conv_elements = (
        linear_layers * segment_count * conv_elements_per_layer_segment
    )
    seam_kv_elements = full_layers * seam_tokens * full_kv_elements_per_token_layer
    full_kv = full_kv_elements * cache_bytes
    recurrent = recurrent_elements * state_bytes
    transition = transition_elements * transition_bytes
    conv = conv_elements * cache_bytes
    seam_kv = seam_kv_elements * cache_bytes

    # Q8/Q4 are only an ideal payload lower bound for a possible compressed
    # HYPIC combination.  No quantized transition/cache kernel is implemented
    # here, and real affine-group metadata must be added before execution.
    compressed_payload_only = {}
    for bits in (8, 4):
        payload = (
            full_kv_elements
            + recurrent_elements
            + transition_elements
            + conv_elements
        ) * bits // 8
        compressed_payload_only[f"q{bits}"] = {
            "suffix_payload_nbytes": payload,
            "executable": False,
            "approximation": True,
            "status": "storage lower bound; quantized compose/KV kernels absent",
        }
    return {
        "suffix_full_attention_layers": full_layers,
        "suffix_linear_attention_layers": linear_layers,
        "full_attention_kv_nbytes": full_kv,
        "linear_zero_start_end_state_nbytes": recurrent,
        "linear_transition_nbytes": transition,
        "conv_tail_nbytes": conv,
        "seam_kv_budget_nbytes": seam_kv,
        "transition_only_nbytes": recurrent + transition + conv,
        "transition_plus_seam_kv_nbytes": recurrent + transition + conv + seam_kv,
        "full_hypic_lite_suffix_nbytes": recurrent + transition + conv + full_kv,
        "compressed_hypic_combination_payload_only": compressed_payload_only,
        "assumptions": {
            "state_bytes": state_bytes,
            "cache_bytes": cache_bytes,
            "transition_bytes": transition_bytes,
            "linear_transition_heads": int(model_config.linear_num_value_heads),
            "note": (
                "GatedDeltaNet repeats keys to value heads and has per-value-head "
                "g/beta, so dense transitions are counted per value head"
            ),
        },
    }
