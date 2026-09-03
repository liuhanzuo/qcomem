from __future__ import annotations

import copy
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterable

import torch

from qcomem_torch import (
    LowerReplayState,
    PackedLowerReplayState,
    cache_nbytes,
    clone_cache,
    tensor_nbytes,
)


ATTENTION_TENSOR_FIELDS = frozenset({"keys", "values", "key_cache", "value_cache"})
LINEAR_MUTABLE_FIELDS = frozenset({"conv_states", "recurrent_states"})


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def _storage_nbytes(tensors: Iterable[torch.Tensor]) -> int:
    storages: dict[tuple[str, int, int], int] = {}
    for tensor in tensors:
        key = _storage_key(tensor)
        storages[key] = key[2]
    return sum(storages.values())


def _iter_tensors(value: Any, *, visited: set[int] | None = None):
    if visited is None:
        visited = set()
    object_id = id(value)
    if object_id in visited:
        return
    visited.add(object_id)
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item, visited=visited)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item, visited=visited)
    elif hasattr(value, "__dict__"):
        for item in vars(value).values():
            yield from _iter_tensors(item, visited=visited)


def _sample_fingerprint(tensor: torch.Tensor) -> tuple[float, ...]:
    """Small audit sample complementing PyTorch's full-tensor version counter."""

    if tensor.numel() == 0:
        return ()
    flat = tensor.detach().reshape(-1)
    count = min(flat.numel(), 16)
    indices = torch.linspace(
        0, flat.numel() - 1, steps=count, device=flat.device
    ).round().long()
    return tuple(float(value) for value in flat[indices].float().cpu().tolist())


@dataclass(frozen=True)
class SharedTensorRecord:
    label: str
    tensor: torch.Tensor
    data_ptr: int
    storage_nbytes: int
    version: int | None
    sample: tuple[float, ...]

    @classmethod
    def capture(cls, label: str, tensor: torch.Tensor) -> "SharedTensorRecord":
        storage = tensor.untyped_storage()
        try:
            version = tensor._version
        except RuntimeError:
            # Tensors created under torch.inference_mode intentionally do not
            # expose version counters.  The safe attention update contract and
            # sampled content guard remain active in that case.
            version = None
        return cls(
            label=label,
            tensor=tensor,
            data_ptr=storage.data_ptr(),
            storage_nbytes=storage.nbytes(),
            version=version,
            sample=_sample_fingerprint(tensor),
        )

    def verify(self) -> str | None:
        storage = self.tensor.untyped_storage()
        if storage.data_ptr() != self.data_ptr or storage.nbytes() != self.storage_nbytes:
            return f"{self.label}: shared storage binding changed"
        if self.version is not None:
            try:
                current_version = self.tensor._version
            except RuntimeError:
                return f"{self.label}: version counter became unavailable"
            if current_version != self.version:
                return f"{self.label}: shared tensor version changed"
        if _sample_fingerprint(self.tensor) != self.sample:
            return f"{self.label}: sampled shared tensor contents changed"
        return None


@dataclass(frozen=True)
class CacheTensorPlan:
    supported: bool
    reason: str | None
    attention_tensor_ids: frozenset[int]
    linear_tensor_ids: frozenset[int]
    attention_nbytes: int
    linear_nbytes: int
    active_attention_layers: tuple[int, ...]
    active_linear_layers: tuple[int, ...]


def analyze_cache_for_cow(cache: Any) -> CacheTensorPlan:
    """Classify cache leaves, rejecting any active tensor with unknown semantics."""

    layers = getattr(cache, "layers", None)
    if layers is None:
        return CacheTensorPlan(False, "cache has no layers attribute", frozenset(), frozenset(), 0, 0, (), ())

    attention: dict[int, torch.Tensor] = {}
    linear: dict[int, torch.Tensor] = {}
    known: set[int] = set()
    active_attention_layers = []
    active_linear_layers = []
    for index, layer in enumerate(layers):
        layer_attention = []
        layer_linear = []
        try:
            layer_fields = vars(layer)
        except TypeError:
            return CacheTensorPlan(
                False,
                f"layer {index} does not expose auditable Python attributes",
                frozenset(),
                frozenset(),
                0,
                0,
                (),
                (),
            )
        for field, value in layer_fields.items():
            tensors = list(_iter_tensors(value))
            if field in ATTENTION_TENSOR_FIELDS:
                for tensor in tensors:
                    attention[id(tensor)] = tensor
                    known.add(id(tensor))
                    if tensor.numel():
                        layer_attention.append(tensor)
            elif field in LINEAR_MUTABLE_FIELDS:
                for tensor in tensors:
                    linear[id(tensor)] = tensor
                    known.add(id(tensor))
                    if tensor.numel():
                        layer_linear.append(tensor)
            elif any(tensor.numel() for tensor in tensors):
                return CacheTensorPlan(
                    False,
                    f"layer {index} has unclassified tensor field {field!r}",
                    frozenset(),
                    frozenset(),
                    0,
                    0,
                    (),
                    (),
                )
        if layer_attention:
            if bool(getattr(layer, "is_sliding", False)):
                return CacheTensorPlan(
                    False,
                    f"layer {index} uses sliding attention; staging update is not implemented",
                    frozenset(),
                    frozenset(),
                    0,
                    0,
                    (),
                    (),
                )
            if not all(hasattr(layer, field) for field in ("keys", "values")):
                return CacheTensorPlan(
                    False,
                    f"layer {index} is not a keys/values DynamicLayer",
                    frozenset(),
                    frozenset(),
                    0,
                    0,
                    (),
                    (),
                )
            active_attention_layers.append(index)
        if layer_linear:
            active_linear_layers.append(index)

    all_cache_tensors = {id(tensor): tensor for tensor in _iter_tensors(cache)}
    unknown = [
        tensor
        for object_id, tensor in all_cache_tensors.items()
        if tensor.numel() and object_id not in known
    ]
    if unknown:
        return CacheTensorPlan(
            False,
            f"cache has {len(unknown)} unclassified tensor leaves",
            frozenset(),
            frozenset(),
            0,
            0,
            (),
            (),
        )
    conflict = set(attention) & set(linear)
    if conflict:
        return CacheTensorPlan(
            False,
            "the same tensor is used by attention and mutable linear state",
            frozenset(),
            frozenset(),
            0,
            0,
            (),
            (),
        )
    return CacheTensorPlan(
        supported=True,
        reason=None,
        attention_tensor_ids=frozenset(attention),
        linear_tensor_ids=frozenset(linear),
        attention_nbytes=_storage_nbytes(attention.values()),
        linear_nbytes=_storage_nbytes(linear.values()),
        active_attention_layers=tuple(active_attention_layers),
        active_linear_layers=tuple(active_linear_layers),
    )


def _safe_dynamic_cow_update(
    layer: Any,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Read shared prefix and bind newly concatenated private KV, never mutate it."""

    del args, kwargs
    if not getattr(layer, "is_initialized", True):
        layer.lazy_initialization(key_states, value_states)
    layer.keys = torch.cat([layer.keys, key_states], dim=-2)
    layer.values = torch.cat([layer.values, value_states], dim=-2)
    return layer.keys, layer.values


def _clone_cache_cow(
    cache: Any, plan: CacheTensorPlan
) -> tuple[Any, tuple[SharedTensorRecord, ...]]:
    memo: dict[int, Any] = {}
    records = []
    for index, tensor in enumerate(_iter_tensors(cache)):
        if id(tensor) in plan.attention_tensor_ids:
            memo[id(tensor)] = tensor
            if tensor.numel():
                records.append(SharedTensorRecord.capture(f"attention.{index}", tensor))
        else:
            # Linear recurrent/conv buffers are updated by copy_ and fused
            # kernels.  Empty/unclassified leaves are also private defensively.
            memo[id(tensor)] = tensor.detach().clone()
    local = copy.deepcopy(cache, memo)
    for index in plan.active_attention_layers:
        layer = local.layers[index]
        try:
            layer.update = MethodType(_safe_dynamic_cow_update, layer)
        except Exception as error:
            raise RuntimeError(
                f"cannot install safe COW update on attention layer {index}: {error}"
            ) from error
    return local, tuple(records)


@dataclass
class PagedForkLowerReplayState:
    depth: int
    document_length: int
    current_length: int
    document_residual: torch.Tensor | None
    cache: Any
    fork_strategy_requested: str
    fork_strategy_effective: str
    fallback_reason: str | None
    initial_shared_nbytes: int
    initial_private_nbytes: int
    _template_storage_keys: frozenset[tuple[str, int, int]]
    _guard: tuple[SharedTensorRecord, ...]

    @property
    def stored_nbytes(self) -> int:
        return tensor_nbytes(self.document_residual) + cache_nbytes(self.cache)

    def memory_breakdown(self) -> dict[str, int]:
        tensors = list(_iter_tensors(self.cache))
        if self.document_residual is not None:
            tensors.append(self.document_residual)
        shared = []
        private = []
        for tensor in tensors:
            if _storage_key(tensor) in self._template_storage_keys:
                shared.append(tensor)
            else:
                private.append(tensor)
        return {
            "shared_nbytes": _storage_nbytes(shared),
            "private_nbytes": _storage_nbytes(private),
        }

    def verify_shared_immutable(self) -> dict[str, Any]:
        failures = [failure for record in self._guard if (failure := record.verify())]
        if failures:
            raise RuntimeError("COW immutable-state audit failed: " + "; ".join(failures))
        return {
            "verified": True,
            "guarded_tensors": len(self._guard),
            "version_guarded_tensors": sum(
                record.version is not None for record in self._guard
            ),
            "audit": (
                "safe read/rebind attention update + storage pointer + available "
                "PyTorch version counters + 16-point sample"
            ),
        }


@dataclass
class PagedLowerReplayState:
    """Auditable COW staging for one persistent lower replay state.

    This is deliberately not advertised as a true paged-attention kernel.
    Attention document tensors are shared at fork and materialized by a safe
    torch.cat on first query write.  Linear conv/recurrent tensors are cloned
    eagerly because Qwen3.5 updates them in-place.
    """

    source_state: LowerReplayState | PackedLowerReplayState
    template: LowerReplayState | None
    plan: CacheTensorPlan
    fallback_reason: str | None = None

    @property
    def depth(self) -> int:
        return self.source_state.depth

    @property
    def document_length(self) -> int:
        return self.source_state.document_length

    @property
    def current_length(self) -> int:
        return self.source_state.current_length

    @property
    def stored_nbytes(self) -> int:
        """Logical durable store bytes (packed bytes for a packed source)."""

        return self.source_state.stored_nbytes

    @property
    def total_resident_nbytes(self) -> int:
        if self.template is None or self.template is self.source_state:
            return cache_nbytes(self.source_state)
        return cache_nbytes([self.source_state, self.template])

    def deployment_memory_components(self) -> dict[str, int | str | bool | None]:
        source_residual = self.source_state.document_residual
        source_residual_bytes = (
            source_residual.nbytes
            if hasattr(source_residual, "nbytes")
            else tensor_nbytes(source_residual)
        )
        source_cache_bytes = self.source_state.stored_nbytes - source_residual_bytes
        staging_bytes = (
            0
            if self.template is None or self.template is self.source_state
            else self.template.stored_nbytes
        )
        return {
            "persistent_residual_nbytes": source_residual_bytes,
            "persistent_lower_state_nbytes": source_cache_bytes,
            "persistent_document_nbytes": self.source_state.stored_nbytes,
            "persistent_materialized_staging_nbytes": staging_bytes,
            "persistent_total_resident_nbytes": self.total_resident_nbytes,
            "cow_template_attention_nbytes": self.plan.attention_nbytes,
            "cow_template_linear_nbytes": self.plan.linear_nbytes,
            "cow_supported": self.plan.supported,
            "cow_fallback_reason": self.fallback_reason,
        }

    def fork(self) -> PagedForkLowerReplayState | LowerReplayState:
        if self.template is None or not self.plan.supported:
            local = self.source_state.fork()
            # Dataclasses are intentionally not slotted, so fallback metadata is
            # visible to deployment measurement without changing core classes.
            local.fork_strategy_requested = "paged-cow-staging"
            local.fork_strategy_effective = "deep-clone-fallback"
            local.fallback_reason = self.fallback_reason or self.plan.reason
            return local

        try:
            local_cache, records = _clone_cache_cow(self.template.cache, self.plan)
        except Exception as error:
            local = self.source_state.fork()
            local.fork_strategy_requested = "paged-cow-staging"
            local.fork_strategy_effective = "deep-clone-fallback"
            local.fallback_reason = f"COW clone failed: {type(error).__name__}: {error}"
            return local

        residual = self.template.document_residual
        records = (
            SharedTensorRecord.capture("document_residual", residual),
            *records,
        )
        template_keys = frozenset(
            _storage_key(tensor)
            for tensor in [residual, *_iter_tensors(self.template.cache)]
        )
        initial_shared = tensor_nbytes(residual) + self.plan.attention_nbytes
        return PagedForkLowerReplayState(
            depth=self.depth,
            document_length=self.document_length,
            current_length=self.current_length,
            document_residual=residual,
            cache=local_cache,
            fork_strategy_requested="paged-cow-staging",
            fork_strategy_effective="paged-cow-staging",
            fallback_reason=None,
            initial_shared_nbytes=initial_shared,
            initial_private_nbytes=self.plan.linear_nbytes,
            _template_storage_keys=template_keys,
            _guard=records,
        )


@torch.inference_mode()
def prepare_paged_lower_state(
    state: LowerReplayState | PackedLowerReplayState,
) -> PagedLowerReplayState:
    """Build one reusable dense staging template, or preserve safe fallback."""

    if not isinstance(state, (LowerReplayState, PackedLowerReplayState)):
        return PagedLowerReplayState(
            source_state=state,
            template=None,
            plan=CacheTensorPlan(False, "unsupported replay state type", frozenset(), frozenset(), 0, 0, (), ()),
            fallback_reason=f"unsupported replay state type: {type(state).__name__}",
        )
    template = state if isinstance(state, LowerReplayState) else state.fork()
    plan = analyze_cache_for_cow(template.cache)
    if not plan.supported:
        return PagedLowerReplayState(
            source_state=state,
            template=None,
            plan=plan,
            fallback_reason=plan.reason,
        )
    return PagedLowerReplayState(
        source_state=state,
        template=template,
        plan=plan,
    )
