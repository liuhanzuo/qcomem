"""One packed entry, dequantized once, genuinely shared by N>1 requests.

This module is purely additive.  It does not modify ``PackedCache.nbytes``,
``qcomem_torch.cache_nbytes``, ``qcomem_torch.tensor_nbytes``,
``qcomem_deployment.capacity_estimate``,
``qcomem_deployment.run_incremental_generation`` (whose three published modes
stay byte-for-byte as published), ``qcomem_deployment.run_exactness_gate``, or
``qcomem_deployment_arms.run_dense_semantics_gate``.  It adds a second *fork
mode* beside the published one and a driver that runs several requests against
one entry at the same time.

Why this exists
---------------

``PackedLowerReplayState.fork`` dequantizes the whole entry into a fresh
private ``LowerReplayState`` for every request.  That is the path behind the
manuscript's headline tables, and Section 4.3 says so in terms: it "shares
nothing and exercises neither borrowing nor copy-on-write".  The ownership
discipline the manuscript audits -- shared immutable document tensors,
request-local mutable state, copy-on-write tails, recurrent rebinding -- is
therefore never exercised on a packed entry.

Fork modes
----------

``private-materialize``
    Exactly the published behaviour: one full dequantized private copy per
    request, produced by ``PackedLowerReplayState.fork`` called unchanged.
    Retained so the published numbers stay reproducible and so the shared mode
    has a semantic reference it must equal token for token.

``shared-packed-view``
    The entry is dequantized **once** into a single ``LowerReplayState`` view.
    Every request forks that view: immutable document tensors (ordinary
    attention K/V and the document boundary residual) are shared by reference;
    mutable state (append reservations and the GatedDeltaNet convolution and
    recurrent buffers) is request-local.

Tail policies
-------------

``borrowed-prefix`` (default)
    The document K/V stays borrowed for the whole request.  Each attention
    layer becomes a :class:`BorrowedPrefixKVLayer` that retains **only** the
    tokens this request appended and returns, per call, a transient dense
    concatenation of the borrowed prefix and its private tail.  The tensor the
    attention module receives is the same single contiguous tensor a stock
    ``DynamicLayer`` would have returned, so the attention arithmetic is
    unchanged; what differs is that the cache does not retain it.  Steady-state
    sharing is therefore real: N resident requests hold one copy of the
    document.  The cost is a transient materialization per attention call,
    which is exactly the working-set term the manuscript's Eq. 1 assumes away.

``materialized-tail``
    The frozen ``qcomem_paged._safe_dynamic_cow_update`` behaviour, kept as a
    conservative fallback: the shared prefix is read and a newly concatenated
    private tensor is bound on the first append.  Sharing is real between fork
    and the request's first write and is zero thereafter.  This policy touches
    no cache-layer interface beyond ``update`` and is the one to select if the
    borrowed-prefix layer fails its preflight gate on a given Transformers
    build.

Rebind policies
---------------

``transition`` (default)
    The mutable convolution/recurrent base is borrowed read-only at fork and
    rebound to private storage at the registered transition -- the call to
    :meth:`SharedPackedFork.rebind_mutable_state` that the driver makes
    immediately before the request's first lower-layer call.  This is the
    discipline Section 4.3 describes.

``setup``
    The mutable base is cloned at fork time.  Equivalent to what the existing
    paged COW staging path does.

Ownership vocabulary is inherited from ``qcomem_paged`` rather than reinvented:
``fork_strategy_requested`` / ``fork_strategy_effective`` / ``fallback_reason``,
``initial_shared_nbytes`` / ``initial_private_nbytes``, ``memory_breakdown()``,
``verify_shared_immutable()``, ``deployment_memory_components()``, the
``CacheTensorPlan`` classification that fails closed on any cache leaf whose
mutation semantics are unknown, and ``_safe_dynamic_cow_update`` itself.

What this module deliberately does not do
-----------------------------------------

It does not implement a paged attention kernel and does not use the page-wise
reference in ``qcomem_paged_attention``: that reference is a per-page online
softmax whose reduction order differs from a single dense GEMM and which is
already recorded as failing the exactness gate on this checkpoint.  It does not
intercept a fused GatedDeltaNet kernel to detect an in-place write; rebinding
happens at a registered transition point recorded in the receipt, not at a
mutation trap.  It does not make forking lazy at page granularity.
"""

from __future__ import annotations

import copy
import hashlib
import statistics
import time
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Callable, Iterator, Sequence

import torch

from qcomem_multifork_accounting import (
    FORK_MODES,
    REBIND_POLICIES,
    TAIL_POLICIES,
    MultiforkAccountingError,
)
from qcomem_paged import (
    CacheTensorPlan,
    SharedTensorRecord,
    _iter_tensors,
    _safe_dynamic_cow_update,
    _storage_key,
    analyze_cache_for_cow,
)
from qcomem_torch import (
    FullPrefixState,
    LowerReplayState,
    PackedLowerReplayState,
    TorchSplitCausalLM,
    cache_nbytes,
    tensor_nbytes,
)


try:  # keep the module importable on a laptop with no Transformers install
    from transformers.cache_utils import CacheLayerMixin as _CacheLayerMixin
except ModuleNotFoundError as error:  # pragma: no cover - local Mac environment
    if error.name != "transformers":
        raise

    class _CacheLayerMixin:  # type: ignore[no-redef]
        supports_early_init = False
        is_compileable = False

        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.keys = None
            self.values = None
            self.is_initialized = False


ATTENTION_TENSOR_FIELDS = frozenset({"keys", "values", "key_cache", "value_cache"})
LINEAR_MUTABLE_FIELDS = frozenset({"conv_states", "recurrent_states"})

#: capture points the driver offers to an observer
CAPTURE_POINTS = ("setup", "transition", "final")


class SharedPackedForkError(RuntimeError):
    """A shared-fork precondition failed in a way that must not be papered over."""


# ---------------------------------------------------------------------------
# tensor slots: a replaceable (parent, key) handle for every tensor leaf
# ---------------------------------------------------------------------------


@dataclass
class TensorSlot:
    """A tensor leaf plus the mutable container position that holds it."""

    path: str
    parent: Any
    key: Any
    tensor: torch.Tensor

    def replace(self, value: torch.Tensor) -> None:
        if isinstance(self.parent, dict):
            self.parent[self.key] = value
        elif isinstance(self.parent, list):
            self.parent[self.key] = value
        elif hasattr(self.parent, "__dict__") and isinstance(self.key, str):
            setattr(self.parent, self.key, value)
        else:
            raise SharedPackedForkError(
                f"tensor slot is not replaceable: {self.path}"
            )
        self.tensor = value


def iter_tensor_slots(root: Any) -> Iterator[TensorSlot]:
    """Yield every tensor leaf reachable through dicts, lists and attributes.

    Tuples are traversed for reading but their slots are not replaceable; such
    a slot raises on ``replace`` rather than silently rebinding nothing.
    """

    visited: set[int] = set()

    def visit(value: Any, path: str, parent: Any, key: Any) -> Iterator[TensorSlot]:
        if isinstance(value, torch.Tensor):
            if parent is not None:
                yield TensorSlot(path=path, parent=parent, key=key, tensor=value)
            return
        object_id = id(value)
        if object_id in visited:
            return
        visited.add(object_id)
        if isinstance(value, dict):
            for child_key in sorted(value, key=str):
                yield from visit(
                    value[child_key], f"{path}/{child_key}", value, child_key
                )
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from visit(child, f"{path}/{index}", value, index)
        elif hasattr(value, "__dict__"):
            for name in sorted(vars(value)):
                yield from visit(getattr(value, name), f"{path}/{name}", value, name)

    yield from visit(root, "root", None, None)


def opaque_storage_id(tensor: torch.Tensor, *, salt: str) -> str:
    """Pointer-free storage identity, in the ForkAudit storage-witness form."""

    device, pointer, storage_bytes = _storage_key(tensor)
    return hashlib.sha256(
        f"{salt}|{device}|{pointer}|{storage_bytes}".encode("utf-8")
    ).hexdigest()


def storage_inventory_rows(
    root: Any, *, role: str, salt: str
) -> list[dict[str, Any]]:
    """Normalized ``[start, end)`` byte ranges for every tensor leaf of ``root``.

    The schema is the one ``qcomem_multifork_accounting.normalize_inventory``
    consumes.  A non-contiguous tensor is recorded with its whole storage as
    its range, which over-approximates the bytes it occupies; the ``contiguous``
    flag marks those rows so an over-approximation is never mistaken for an
    exact view range.
    """

    if not salt:
        raise SharedPackedForkError("storage inventory salt must be non-empty")
    rows: list[dict[str, Any]] = []
    for slot in iter_tensor_slots(root):
        tensor = slot.tensor
        _, _, storage_bytes = _storage_key(tensor)
        contiguous = bool(tensor.is_contiguous())
        if contiguous:
            start = int(tensor.storage_offset() * tensor.element_size())
            nbytes = int(tensor.numel() * tensor.element_size())
        else:
            start = 0
            nbytes = storage_bytes
        end = min(start + nbytes, storage_bytes)
        rows.append(
            {
                "path": slot.path,
                "role": role,
                "storage_id": opaque_storage_id(tensor, salt=salt),
                "storage_nbytes": storage_bytes,
                "view_start_bytes": start,
                "view_end_bytes": end,
                "contiguous": contiguous,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
            }
        )
    return rows


def _classify_cache_tensor_ids(cache: Any) -> tuple[set[int], set[int]]:
    """Split cache tensor leaves into attention and mutable-linear leaves.

    The field names are the ones ``qcomem_paged`` uses, so a cache that
    ``analyze_cache_for_cow`` accepts is classified identically here.  Anything
    not in either set is left out and is cloned defensively by the caller.
    """

    attention: set[int] = set()
    linear: set[int] = set()
    for layer in getattr(cache, "layers", None) or ():
        try:
            fields = vars(layer)
        except TypeError:  # pragma: no cover - rejected earlier by the plan
            continue
        for name, value in fields.items():
            if name in ATTENTION_TENSOR_FIELDS:
                attention.update(id(tensor) for tensor in _iter_tensors(value))
            elif name in LINEAR_MUTABLE_FIELDS:
                linear.update(id(tensor) for tensor in _iter_tensors(value))
    return attention, linear


# ---------------------------------------------------------------------------
# the borrowed-prefix attention layer
# ---------------------------------------------------------------------------


class BorrowedPrefixKVLayer(_CacheLayerMixin):
    """Borrow the shared document K/V; retain only this request's tail.

    ``shared_keys`` / ``shared_values`` are the entry's immutable document
    tensors.  They are read on every call and never written.  ``tail_keys`` /
    ``tail_values`` hold **only** the tokens this request appended.  ``update``
    grows the tail and returns ``torch.cat([prefix, tail], dim=-2)``: the same
    single contiguous tensor a stock ``DynamicLayer`` would have returned, so
    the attention arithmetic is bit-identical to the eager path, but the
    concatenation is transient and is not retained by the cache.

    ``keys`` and ``values`` are deliberately ``None``.  Any caller that reads
    them directly instead of using ``update``'s return value or
    ``get_seq_length`` will see ``None`` rather than a wrong tensor; that is a
    fail-loud choice, and the preflight gate is where it surfaces.
    """

    is_sliding = False
    is_compileable = False
    supports_early_init = False

    def __init__(
        self,
        shared_keys: torch.Tensor,
        shared_values: torch.Tensor,
        *,
        layer_index: int,
        request_id: str,
        events: list[dict[str, Any]],
        salt: str,
    ) -> None:
        super().__init__()
        if shared_keys.shape[-2] != shared_values.shape[-2]:
            raise SharedPackedForkError(
                "borrowed prefix key/value token counts disagree"
            )
        self.shared_keys = shared_keys
        self.shared_values = shared_values
        self.tail_keys: torch.Tensor | None = None
        self.tail_values: torch.Tensor | None = None
        self.layer_index = int(layer_index)
        self.request_id = str(request_id)
        self.keys = None
        self.values = None
        self.is_initialized = True
        self.dtype = shared_keys.dtype
        self.device = shared_keys.device
        self._events = events
        self._salt = salt
        self.mask_size_call_forms: list[str] = []

    # -- lengths -------------------------------------------------------------

    @property
    def prefix_length(self) -> int:
        return int(self.shared_keys.shape[-2])

    @property
    def tail_length(self) -> int:
        return 0 if self.tail_keys is None else int(self.tail_keys.shape[-2])

    def get_seq_length(self) -> int:
        return self.prefix_length + self.tail_length

    def get_max_length(self) -> int:
        return -1

    @staticmethod
    def _query_length(value: Any) -> int:
        """Accept either an integer query length or a ``cache_position`` tensor.

        Raises on anything else instead of guessing, so an unrecognized
        Transformers mask-sizing convention fails loudly at the preflight gate
        rather than producing a silently wrong mask.
        """

        if isinstance(value, int):
            return value
        if isinstance(value, torch.Tensor):
            if value.ndim != 1:
                raise SharedPackedForkError(
                    "cache_position must be one-dimensional"
                )
            return int(value.shape[0])
        raise SharedPackedForkError(
            f"unsupported mask-size argument type: {type(value).__name__}"
        )

    def get_mask_sizes(self, *args: Any, **kwargs: Any) -> tuple[int, int]:
        if args:
            first = args[0]
        elif "cache_position" in kwargs:
            first = kwargs["cache_position"]
        elif "query_length" in kwargs:
            first = kwargs["query_length"]
        else:
            raise SharedPackedForkError(
                "get_mask_sizes was called with no recognized argument"
            )
        form = "int" if isinstance(first, int) else type(first).__name__
        if form not in self.mask_size_call_forms:
            self.mask_size_call_forms.append(form)
        return self.get_seq_length() + self._query_length(first), 0

    # -- the update contract --------------------------------------------------

    def lazy_initialization(
        self, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> None:
        del key_states, value_states
        self.is_initialized = True

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        prefix_id = opaque_storage_id(self.shared_keys, salt=self._salt)
        before_id = (
            opaque_storage_id(self.tail_keys, salt=self._salt)
            if self.tail_keys is not None
            else None
        )
        if self.tail_keys is None:
            self.tail_keys = key_states
            self.tail_values = value_states
        else:
            self.tail_keys = torch.cat([self.tail_keys, key_states], dim=-2)
            self.tail_values = torch.cat([self.tail_values, value_states], dim=-2)
        keys = torch.cat([self.shared_keys, self.tail_keys], dim=-2)
        values = torch.cat([self.shared_values, self.tail_values], dim=-2)
        after_id = opaque_storage_id(self.tail_keys, salt=self._salt)
        self._events.append(
            {
                "request_id": self.request_id,
                "layer_index": self.layer_index,
                "tail_policy": "borrowed-prefix",
                "appended_tokens": int(key_states.shape[-2]),
                "before_keys_storage_id": before_id,
                "after_keys_storage_id": after_id,
                "after_values_storage_id": opaque_storage_id(
                    self.tail_values, salt=self._salt
                ),
                "keys_storage_rebound": before_id != after_id,
                "shared_prefix_storage_id": prefix_id,
                "shared_prefix_storage_unchanged": (
                    opaque_storage_id(self.shared_keys, salt=self._salt) == prefix_id
                ),
                "prefix_tokens": self.prefix_length,
                "tail_tokens": self.tail_length,
                "returned_is_retained": False,
                "transient_concat_nbytes": int(
                    keys.numel() * keys.element_size()
                    + values.numel() * values.element_size()
                ),
            }
        )
        return keys, values

    # -- the rest of the cache-layer surface, deliberately unimplemented ------
    # Mirrors qcomem_paged_attention.PagedKVLayer: these exist so the class is
    # a concrete CacheLayerMixin, and each raises rather than silently doing
    # the wrong thing on a borrowed prefix.

    def offload(self) -> None:
        raise NotImplementedError("the borrowed-prefix layer does not offload")

    def prefetch(self) -> None:
        raise NotImplementedError("the borrowed-prefix layer does not prefetch")

    def reset(self) -> None:
        raise NotImplementedError(
            "resetting would discard a borrowed prefix this layer does not own"
        )

    def reorder_cache(self, beam_idx: Any) -> None:
        del beam_idx
        raise NotImplementedError("the borrowed-prefix layer has no beam search")

    def crop(self, tokens_to_remove: int) -> None:
        del tokens_to_remove
        raise NotImplementedError("the borrowed-prefix layer does not crop")

    def batch_repeat_interleave(self, repeats: int) -> None:
        del repeats
        raise NotImplementedError("the borrowed-prefix layer does not batch repeat")

    def batch_select_indices(self, indices: Any) -> None:
        del indices
        raise NotImplementedError("the borrowed-prefix layer does not batch select")


def _make_recording_cow_update(
    layer_index: int, request_id: str, events: list[dict[str, Any]], salt: str
) -> Callable[..., tuple[torch.Tensor, torch.Tensor]]:
    """Wrap the frozen safe COW update so every append leaves a receipt.

    The append is performed by ``qcomem_paged._safe_dynamic_cow_update``
    unchanged; this records only the storage identity of the retained key/value
    tensors before and after, which is what makes copy-before-append
    observable.
    """

    def update(layer: Any, key_states, value_states, *args, **kwargs):
        before_keys = getattr(layer, "keys", None)
        before_id = (
            opaque_storage_id(before_keys, salt=salt)
            if isinstance(before_keys, torch.Tensor)
            else None
        )
        before_tokens = (
            int(before_keys.shape[-2])
            if isinstance(before_keys, torch.Tensor)
            else 0
        )
        keys, values = _safe_dynamic_cow_update(
            layer, key_states, value_states, *args, **kwargs
        )
        after_id = opaque_storage_id(keys, salt=salt)
        events.append(
            {
                "request_id": request_id,
                "layer_index": layer_index,
                "tail_policy": "materialized-tail",
                "appended_tokens": int(key_states.shape[-2]),
                "before_keys_storage_id": before_id,
                "after_keys_storage_id": after_id,
                "after_values_storage_id": opaque_storage_id(values, salt=salt),
                "keys_storage_rebound": before_id != after_id,
                "shared_prefix_storage_id": None,
                "shared_prefix_storage_unchanged": None,
                "prefix_tokens": before_tokens,
                "tail_tokens": int(keys.shape[-2]) - before_tokens,
                "returned_is_retained": True,
                "transient_concat_nbytes": 0,
            }
        )
        return keys, values

    return update


# ---------------------------------------------------------------------------
# one request's fork of the shared view
# ---------------------------------------------------------------------------


@dataclass
class SharedPackedFork:
    """One request's view of a shared packed entry.

    Field names mirror ``LowerReplayState`` so ``TorchSplitCausalLM``'s
    ``continue_lower_replay`` operates on this object unchanged, and mirror
    ``qcomem_paged.PagedForkLowerReplayState`` so any measurement code that
    reads ``fork_strategy_effective`` / ``memory_breakdown`` /
    ``verify_shared_immutable`` through ``getattr`` sees the same vocabulary.
    """

    request_id: str
    depth: int
    document_length: int
    current_length: int
    document_residual: torch.Tensor | None
    cache: Any
    fork_strategy_requested: str
    fork_strategy_effective: str
    fallback_reason: str | None
    rebind_policy: str
    tail_policy: str
    initial_shared_nbytes: int
    initial_private_nbytes: int
    _shared_storage_keys: frozenset[tuple[str, int, int]]
    _guard: tuple[SharedTensorRecord, ...]
    _borrowed_slots: list[TensorSlot] = field(default_factory=list)
    append_events: list[dict[str, Any]] = field(default_factory=list)
    rebind_events: list[dict[str, Any]] = field(default_factory=list)
    residual_binding_events: list[dict[str, Any]] = field(default_factory=list)
    rebound: bool = False
    released_document_residual: bool = False

    # -- ownership vocabulary shared with qcomem_paged ---------------------

    @property
    def stored_nbytes(self) -> int:
        return tensor_nbytes(self.document_residual) + cache_nbytes(self.cache)

    def memory_breakdown(self) -> dict[str, int]:
        tensors = list(_iter_tensors(self.cache))
        if self.document_residual is not None:
            tensors.append(self.document_residual)
        shared: dict[tuple[str, int, int], int] = {}
        private: dict[tuple[str, int, int], int] = {}
        for tensor in tensors:
            key = _storage_key(tensor)
            if key in self._shared_storage_keys:
                shared[key] = key[2]
            else:
                private[key] = key[2]
        return {
            "shared_nbytes": sum(shared.values()),
            "private_nbytes": sum(private.values()),
            "shared_storage_count": len(shared),
            "private_storage_count": len(private),
        }

    def verify_shared_immutable(self) -> dict[str, Any]:
        """Raise if any guarded shared tensor moved, was resized, or changed.

        The guard covers exactly the tensors this fork borrowed at fork time:
        storage pointer and size, the PyTorch version counter where one is
        available, and a 16-point content sample.  It does not certify parts of
        the view this fork never referenced.
        """

        failures = [
            failure for record in self._guard if (failure := record.verify())
        ]
        if failures:
            raise SharedPackedForkError(
                "shared-view immutability audit failed: " + "; ".join(failures)
            )
        return {
            "verified": True,
            "request_id": self.request_id,
            "guarded_tensors": len(self._guard),
            "version_guarded_tensors": sum(
                record.version is not None for record in self._guard
            ),
            "audit": (
                "safe read/rebind attention update + storage pointer + "
                "available PyTorch version counters + 16-point sample, over "
                "the tensors this fork borrowed at setup"
            ),
        }

    # -- the registered transition ------------------------------------------

    def rebind_mutable_state(self) -> dict[str, Any]:
        """Rebind every borrowed mutable leaf to private storage.

        Under ``rebind_policy="setup"`` nothing was borrowed and this is a
        recorded no-op.  Under ``rebind_policy="transition"`` this is the
        registered transition: the borrowed recurrent and convolution base is
        read-only up to this call and privately owned after it.  Calling it
        twice raises, because a second rebind would discard the state the first
        one made private.
        """

        if self.rebound:
            raise SharedPackedForkError(
                f"{self.request_id}: mutable state was already rebound"
            )
        events: list[dict[str, Any]] = []
        for slot in self._borrowed_slots:
            before = _storage_key(slot.tensor)
            private = slot.tensor.detach().clone()
            slot.replace(private)
            after = _storage_key(private)
            events.append(
                {
                    "request_id": self.request_id,
                    "path": slot.path,
                    "before_device": before[0],
                    "before_storage_nbytes": before[2],
                    "after_device": after[0],
                    "after_storage_nbytes": after[2],
                    "storage_identity_changed": before != after,
                    "nbytes": after[2],
                }
            )
        self.rebind_events.extend(events)
        self.rebound = True
        return {
            "request_id": self.request_id,
            "rebind_policy": self.rebind_policy,
            "rebound_tensor_count": len(events),
            "rebound_nbytes": sum(event["nbytes"] for event in events),
            "events": events,
        }

    def record_residual_binding(
        self,
        *,
        shared_residual: torch.Tensor,
        query_residual: torch.Tensor,
        document_position_offset: int,
        query_position_offset: int,
        salt: str,
    ) -> dict[str, Any]:
        """Record which residual chunk seeded the suffix and which extended it."""

        event = {
            "request_id": self.request_id,
            "document_chunk_storage_id": opaque_storage_id(
                shared_residual, salt=salt
            ),
            "query_chunk_storage_id": opaque_storage_id(query_residual, salt=salt),
            "document_chunk_is_shared_view_tensor": (
                _storage_key(shared_residual) in self._shared_storage_keys
            ),
            "query_chunk_is_shared_view_tensor": (
                _storage_key(query_residual) in self._shared_storage_keys
            ),
            "document_position_offset": int(document_position_offset),
            "query_position_offset": int(query_position_offset),
            "document_chunk_tokens": int(shared_residual.shape[1]),
            "query_chunk_tokens": int(query_residual.shape[1]),
            "chunks_are_distinct_calls": True,
            "chunks_share_storage": (
                _storage_key(shared_residual) == _storage_key(query_residual)
            ),
        }
        self.residual_binding_events.append(event)
        return event

    def release_document_residual(self) -> None:
        """Drop this request's reference to the shared boundary residual.

        The tensor belongs to the entry's shared view and stays alive; only
        this fork stops referencing it, which is what the published
        ``run_incremental_generation`` does for its private copy.
        """

        self.document_residual = None
        self.released_document_residual = True


@dataclass
class PrivateMaterializedFork:
    """One request's fully private dequantized copy of a packed entry.

    Wraps ``PackedLowerReplayState.fork`` -- the published Read path, called
    unchanged -- in the interface the shared fork exposes, so the driver and
    the receipts treat both modes identically and the comparison is like for
    like.
    """

    request_id: str
    state: LowerReplayState
    fork_strategy_requested: str = "private-materialize"
    fork_strategy_effective: str = "private-materialize"
    fallback_reason: str | None = None
    rebind_policy: str = "not-applicable"
    tail_policy: str = "not-applicable"
    initial_shared_nbytes: int = 0
    initial_private_nbytes: int = 0
    append_events: list[dict[str, Any]] = field(default_factory=list)
    rebind_events: list[dict[str, Any]] = field(default_factory=list)
    residual_binding_events: list[dict[str, Any]] = field(default_factory=list)
    rebound: bool = True
    released_document_residual: bool = False

    @property
    def depth(self) -> int:
        return self.state.depth

    @property
    def document_length(self) -> int:
        return self.state.document_length

    @property
    def current_length(self) -> int:
        return self.state.current_length

    @current_length.setter
    def current_length(self, value: int) -> None:
        self.state.current_length = value

    @property
    def cache(self) -> Any:
        return self.state.cache

    @property
    def document_residual(self) -> torch.Tensor | None:
        return self.state.document_residual

    @document_residual.setter
    def document_residual(self, value: torch.Tensor | None) -> None:
        self.state.document_residual = value

    @property
    def stored_nbytes(self) -> int:
        return tensor_nbytes(self.state.document_residual) + cache_nbytes(
            self.state.cache
        )

    def memory_breakdown(self) -> dict[str, int]:
        private = {
            _storage_key(tensor): _storage_key(tensor)[2]
            for tensor in _iter_tensors(self.state.cache)
        }
        if self.state.document_residual is not None:
            key = _storage_key(self.state.document_residual)
            private[key] = key[2]
        return {
            "shared_nbytes": 0,
            "private_nbytes": sum(private.values()),
            "shared_storage_count": 0,
            "private_storage_count": len(private),
        }

    def verify_shared_immutable(self) -> dict[str, Any]:
        """Report that this mode shares nothing, so there is nothing to guard.

        Reported as ``verified`` with zero guarded tensors and an explicit
        ``vacuous`` flag: a green immutability audit on this arm is not
        evidence of safe sharing, because no sharing occurred.
        """

        return {
            "verified": True,
            "vacuous": True,
            "request_id": self.request_id,
            "guarded_tensors": 0,
            "version_guarded_tensors": 0,
            "audit": "private-materialize shares nothing; no shared tensor exists",
        }

    def rebind_mutable_state(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "rebind_policy": "not-applicable",
            "rebound_tensor_count": 0,
            "rebound_nbytes": 0,
            "events": [],
        }

    def record_residual_binding(
        self,
        *,
        shared_residual: torch.Tensor,
        query_residual: torch.Tensor,
        document_position_offset: int,
        query_position_offset: int,
        salt: str,
    ) -> dict[str, Any]:
        event = {
            "request_id": self.request_id,
            "document_chunk_storage_id": opaque_storage_id(
                shared_residual, salt=salt
            ),
            "query_chunk_storage_id": opaque_storage_id(query_residual, salt=salt),
            "document_chunk_is_shared_view_tensor": False,
            "query_chunk_is_shared_view_tensor": False,
            "document_position_offset": int(document_position_offset),
            "query_position_offset": int(query_position_offset),
            "document_chunk_tokens": int(shared_residual.shape[1]),
            "query_chunk_tokens": int(query_residual.shape[1]),
            "chunks_are_distinct_calls": True,
            "chunks_share_storage": (
                _storage_key(shared_residual) == _storage_key(query_residual)
            ),
        }
        self.residual_binding_events.append(event)
        return event

    def release_document_residual(self) -> None:
        self.state.document_residual = None
        self.released_document_residual = True


# ---------------------------------------------------------------------------
# the resident entry
# ---------------------------------------------------------------------------


@dataclass
class SharedPackedEntry:
    """A durable packed entry plus, in shared mode, one dequantized view.

    ``packed_state`` is never mutated.  In ``shared-packed-view`` mode exactly
    one dequantized ``LowerReplayState`` view is created and every request
    forks that view; in ``private-materialize`` mode no view is created and
    every request calls ``PackedLowerReplayState.fork`` for itself.
    """

    packed_state: PackedLowerReplayState | LowerReplayState
    share_mode: str
    rebind_policy: str
    tail_policy: str
    salt: str
    view: LowerReplayState | None = None
    plan: CacheTensorPlan | None = None
    fallback_reason: str | None = None
    materialize_seconds: float = 0.0
    fork_count: int = 0
    #: sampled immutability guard over every tensor of the shared view
    _view_guard: tuple[SharedTensorRecord, ...] = ()

    # -- identity forwarded from the entry ---------------------------------

    @property
    def depth(self) -> int:
        return self.packed_state.depth

    @property
    def document_length(self) -> int:
        return self.packed_state.document_length

    @property
    def current_length(self) -> int:
        return self.packed_state.current_length

    @property
    def entry_retained_nbytes(self) -> int:
        """The durable store the manuscript already reports for this entry."""

        return self.packed_state.stored_nbytes

    @property
    def shared_view_nbytes(self) -> int:
        if self.view is None:
            return 0
        return tensor_nbytes(self.view.document_residual) + cache_nbytes(
            self.view.cache
        )

    @property
    def effective_share_mode(self) -> str:
        return (
            "shared-packed-view" if self.view is not None else "private-materialize"
        )

    def shared_storage_keys(self) -> frozenset[tuple[str, int, int]]:
        if self.view is None:
            return frozenset()
        tensors = [*_iter_tensors(self.view.cache)]
        if self.view.document_residual is not None:
            tensors.append(self.view.document_residual)
        return frozenset(_storage_key(tensor) for tensor in tensors)

    def shared_inventory(self) -> list[dict[str, Any]]:
        """Storage rows for the immutable document view; empty in private mode."""

        if self.view is None:
            return []
        return storage_inventory_rows(
            {
                "document_residual": self.view.document_residual,
                "cache": self.view.cache,
            },
            role="shared_view",
            salt=self.salt,
        )

    def deployment_memory_components(self) -> dict[str, int | str | bool | None]:
        """Retained / staging / resident split in the published field vocabulary.

        ``persistent_document_nbytes`` keeps its published meaning (the durable
        packed entry).  The dequantized view is reported separately as staging
        bytes, exactly as the paged COW template already is, so no published
        column silently changes meaning.
        """

        residual = getattr(self.packed_state, "document_residual", None)
        residual_bytes = (
            residual.nbytes
            if hasattr(residual, "nbytes")
            else tensor_nbytes(residual)
        )
        entry_bytes = self.entry_retained_nbytes
        view_bytes = self.shared_view_nbytes
        return {
            "persistent_residual_nbytes": residual_bytes,
            "persistent_lower_state_nbytes": entry_bytes - residual_bytes,
            "persistent_document_nbytes": entry_bytes,
            "persistent_materialized_staging_nbytes": view_bytes,
            "persistent_total_resident_nbytes": entry_bytes + view_bytes,
            "shared_dequantized_view_nbytes": view_bytes,
            "share_mode_requested": self.share_mode,
            "share_mode_effective": self.effective_share_mode,
            "rebind_policy": self.rebind_policy,
            "tail_policy": self.tail_policy,
            "cow_supported": bool(self.plan.supported) if self.plan else False,
            "cow_fallback_reason": self.fallback_reason,
            # A depth-split entry only has attention document state to share if
            # at least one full-attention layer sits below the split.  On
            # Qwen3.5 the full-attention layers are 3, 7, 11, ..., so j=7
            # leaves exactly one.  A split shallower than the first
            # full-attention layer would share only the boundary residual, and
            # the audit's non-vacuous sharing check is what surfaces that.
            "shared_attention_layer_count": (
                len(self.plan.active_attention_layers) if self.plan else 0
            ),
            "shared_linear_layer_count": (
                len(self.plan.active_linear_layers) if self.plan else 0
            ),
            "shared_attention_nbytes": (
                self.plan.attention_nbytes if self.plan else 0
            ),
            "shared_linear_nbytes": self.plan.linear_nbytes if self.plan else 0,
        }

    # -- forking -------------------------------------------------------------

    def fork(self, request_id: str) -> SharedPackedFork | PrivateMaterializedFork:
        self.fork_count += 1
        if self.view is None:
            return PrivateMaterializedFork(
                request_id=request_id,
                state=self.packed_state.fork(),
                fallback_reason=self.fallback_reason,
            )
        assert self.plan is not None
        attention_ids, linear_ids = _classify_cache_tensor_ids(self.view.cache)
        borrow_attention = self.tail_policy == "materialized-tail"
        borrow_linear = self.rebind_policy == "transition"
        memo: dict[int, Any] = {}
        borrowed_linear_ids: set[int] = set()
        guards: list[SharedTensorRecord] = []
        for index, tensor in enumerate(_iter_tensors(self.view.cache)):
            object_id = id(tensor)
            if object_id in attention_ids:
                # Under both tail policies the deep copy borrows the attention
                # tensors.  Under borrowed-prefix they are then lifted out of
                # the layer and into a BorrowedPrefixKVLayer below.
                memo[object_id] = tensor
                if tensor.numel():
                    guards.append(
                        SharedTensorRecord.capture(f"attention.{index}", tensor)
                    )
            elif object_id in linear_ids and borrow_linear:
                memo[object_id] = tensor
                borrowed_linear_ids.add(object_id)
                if tensor.numel():
                    guards.append(
                        SharedTensorRecord.capture(f"borrowed_linear.{index}", tensor)
                    )
            else:
                # Linear buffers under the setup rebind policy, and every empty
                # or unclassified leaf, are private from the first instant.
                memo[object_id] = tensor.detach().clone()
        local_cache = copy.deepcopy(self.view.cache, memo)
        append_events: list[dict[str, Any]] = []
        if borrow_attention:
            for index in self.plan.active_attention_layers:
                layer = local_cache.layers[index]
                layer.update = MethodType(
                    _make_recording_cow_update(
                        index, request_id, append_events, self.salt
                    ),
                    layer,
                )
        else:
            for index in self.plan.active_attention_layers:
                source = local_cache.layers[index]
                local_cache.layers[index] = BorrowedPrefixKVLayer(
                    source.keys,
                    source.values,
                    layer_index=index,
                    request_id=request_id,
                    events=append_events,
                    salt=self.salt,
                )
        residual = self.view.document_residual
        if residual is not None:
            guards.insert(
                0, SharedTensorRecord.capture("document_residual", residual)
            )
        borrowed_slots = [
            slot
            for slot in iter_tensor_slots(local_cache)
            if id(slot.tensor) in borrowed_linear_ids
        ]
        shared_nbytes = tensor_nbytes(residual) + self.plan.attention_nbytes
        private_nbytes = 0
        if borrow_linear:
            shared_nbytes += self.plan.linear_nbytes
        else:
            private_nbytes = self.plan.linear_nbytes
        return SharedPackedFork(
            request_id=request_id,
            depth=self.view.depth,
            document_length=self.view.document_length,
            current_length=self.view.current_length,
            document_residual=residual,
            cache=local_cache,
            fork_strategy_requested="shared-packed-view",
            fork_strategy_effective="shared-packed-view",
            fallback_reason=None,
            rebind_policy=self.rebind_policy,
            tail_policy=self.tail_policy,
            initial_shared_nbytes=shared_nbytes,
            initial_private_nbytes=private_nbytes,
            _shared_storage_keys=self.shared_storage_keys(),
            _guard=tuple(guards),
            _borrowed_slots=borrowed_slots,
            append_events=append_events,
        )

    def verify_view_unchanged(self) -> dict[str, Any]:
        """Report whether the shared view still holds its fork-time content.

        A fresh 16-point sample plus storage identity over every view tensor.
        This is a sampled guard, not a full content comparison; the
        full-content digest lives in the audit module's receipts.
        """

        if self.view is None:
            return {
                "verified": True,
                "vacuous": True,
                "guarded_tensors": 0,
                "failures": [],
                "audit": "no shared view exists in private-materialize mode",
            }
        failures = [
            failure for record in self._view_guard if (failure := record.verify())
        ]
        return {
            "verified": not failures,
            "vacuous": False,
            "guarded_tensors": len(self._view_guard),
            "failures": failures,
            "audit": "storage identity + version counter + 16-point sample",
        }


@torch.inference_mode()
def prepare_shared_packed_entry(
    packed_state: PackedLowerReplayState | LowerReplayState,
    *,
    share_mode: str,
    rebind_policy: str = "transition",
    tail_policy: str = "borrowed-prefix",
    salt: str = "qcomem-shared-packed-multifork",
) -> SharedPackedEntry:
    """Build the resident entry, dequantizing at most once.

    ``share_mode="private-materialize"`` creates no view; every fork calls the
    published ``PackedLowerReplayState.fork``.  ``share_mode="shared-packed-view"``
    dequantizes once and classifies the resulting cache with
    ``qcomem_paged.analyze_cache_for_cow``; if that classification is not
    supported the entry falls back to private materialization **with a recorded
    reason** rather than sharing a cache whose mutation semantics are unknown.
    """

    if share_mode not in FORK_MODES:
        raise MultiforkAccountingError(f"unknown share mode: {share_mode}")
    if rebind_policy not in REBIND_POLICIES:
        raise MultiforkAccountingError(f"unknown rebind policy: {rebind_policy}")
    if tail_policy not in TAIL_POLICIES:
        raise MultiforkAccountingError(f"unknown tail policy: {tail_policy}")
    if share_mode == "private-materialize":
        return SharedPackedEntry(
            packed_state=packed_state,
            share_mode=share_mode,
            rebind_policy="not-applicable",
            tail_policy="not-applicable",
            salt=salt,
            view=None,
            plan=None,
            fallback_reason=None,
        )
    started = time.perf_counter()
    view = (
        packed_state.fork()
        if isinstance(packed_state, PackedLowerReplayState)
        else packed_state
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    plan = analyze_cache_for_cow(view.cache)
    if not plan.supported:
        return SharedPackedEntry(
            packed_state=packed_state,
            share_mode=share_mode,
            rebind_policy="not-applicable",
            tail_policy="not-applicable",
            salt=salt,
            view=None,
            plan=plan,
            fallback_reason=plan.reason,
            materialize_seconds=elapsed,
        )
    entry = SharedPackedEntry(
        packed_state=packed_state,
        share_mode=share_mode,
        rebind_policy=rebind_policy,
        tail_policy=tail_policy,
        salt=salt,
        view=view,
        plan=plan,
        fallback_reason=None,
        materialize_seconds=elapsed,
    )
    tensors = list(_iter_tensors(view.cache))
    if view.document_residual is not None:
        tensors.insert(0, view.document_residual)
    entry._view_guard = tuple(
        SharedTensorRecord.capture(f"view.{index}", tensor)
        for index, tensor in enumerate(tensors)
        if tensor.numel()
    )
    return entry


# ---------------------------------------------------------------------------
# the N-request driver
# ---------------------------------------------------------------------------


@dataclass
class MultiforkRequestTrace:
    request_id: str
    query_token_count: int
    generated_token_ids: list[int]
    ttft_seconds: float
    tpot_seconds: list[float]
    materialized_nbytes: int
    steady_resident_nbytes: int
    suffix_cache_nbytes: int
    initial_shared_nbytes: int
    initial_private_nbytes: int
    final_shared_nbytes: int
    final_private_nbytes: int
    append_event_count: int
    rebind_event_count: int
    rebind_nbytes: int
    transient_concat_peak_nbytes: int

    def summary(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query_token_count": self.query_token_count,
            "generated_token_ids": list(self.generated_token_ids),
            "generated_tokens": len(self.generated_token_ids),
            "ttft_seconds": self.ttft_seconds,
            "tpot_seconds": list(self.tpot_seconds),
            "median_tpot_seconds": (
                statistics.median(self.tpot_seconds) if self.tpot_seconds else None
            ),
            "materialized_nbytes": self.materialized_nbytes,
            "steady_resident_nbytes": self.steady_resident_nbytes,
            "suffix_cache_nbytes": self.suffix_cache_nbytes,
            "initial_shared_nbytes": self.initial_shared_nbytes,
            "initial_private_nbytes": self.initial_private_nbytes,
            "final_shared_nbytes": self.final_shared_nbytes,
            "final_private_nbytes": self.final_private_nbytes,
            "append_event_count": self.append_event_count,
            "rebind_event_count": self.rebind_event_count,
            "rebind_nbytes": self.rebind_nbytes,
            "transient_concat_peak_nbytes": self.transient_concat_peak_nbytes,
        }


@dataclass
class MultiforkTrace:
    arm: str
    fork_mode: str
    rebind_policy: str
    tail_policy: str
    request_traces: list[MultiforkRequestTrace]
    forks: list[Any]
    entry: SharedPackedEntry | None
    setup_seconds: float
    transition_seconds: float
    decode_seconds: float
    phase_allocated_nbytes: dict[str, int]
    peak_allocated_nbytes: int
    baseline_allocated_nbytes: int
    steady_allocated_nbytes: int
    append_events: list[dict[str, Any]]
    rebind_events: list[dict[str, Any]]
    residual_binding_events: list[dict[str, Any]]
    adapter_call_log: list[dict[str, Any]]
    mask_size_call_forms: list[str]

    def token_traces(self) -> dict[str, list[int]]:
        return {
            trace.request_id: list(trace.generated_token_ids)
            for trace in self.request_traces
        }


def _allocated() -> int:
    return int(torch.cuda.memory_allocated()) if torch.cuda.is_available() else 0


def _peak_allocated() -> int:
    return int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _argmax(logits: torch.Tensor) -> int:
    return int(torch.argmax(logits, dim=-1).item())


def _mask_size_call_forms(forks: Sequence[Any]) -> list[str]:
    forms: list[str] = []
    for fork in forks:
        for layer in getattr(getattr(fork, "cache", None), "layers", ()) or ():
            for form in getattr(layer, "mask_size_call_forms", ()) or ():
                if form not in forms:
                    forms.append(form)
    return forms


@torch.inference_mode()
def run_shared_packed_multifork(
    adapter: TorchSplitCausalLM,
    entry: SharedPackedEntry,
    queries: Sequence[tuple[str, torch.Tensor]],
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    capture: Callable[[str, Sequence[Any]], None] | None = None,
) -> MultiforkTrace:
    """Run N requests concurrently against one entry, interleaved on one stream.

    All N forks are created before any of them executes, every request's suffix
    cache stays live for the whole run, and decode steps are interleaved
    round-robin so the requests are genuinely co-resident rather than run to
    completion one after another.

    The registered transition for request ``r`` is the call to
    ``rebind_mutable_state`` immediately before its first
    ``continue_lower_replay``.  Under ``rebind_policy="transition"`` that is the
    point at which the borrowed recurrent/convolution base becomes private
    storage.

    ``capture`` is invoked with ``("setup", forks)`` after every fork exists and
    before any transition, ``("transition", forks)`` after every registered
    transition and query prefill, and ``("final", forks)`` after decoding.  It
    is the only way to observe the borrow window, because it closes as soon as
    the first request transitions.
    """

    if not queries:
        raise SharedPackedForkError("at least one request is required")
    if len({request_id for request_id, _ in queries}) != len(queries):
        raise SharedPackedForkError("request ids must be unique")
    depth = entry.depth
    call_log: list[dict[str, Any]] = []
    phase_allocated: dict[str, int] = {}

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync()
    baseline = _allocated()
    phase_allocated["entry_resident"] = baseline

    setup_started = time.perf_counter()
    forks = [entry.fork(request_id) for request_id, _ in queries]
    _sync()
    setup_seconds = time.perf_counter() - setup_started
    phase_allocated["after_all_forks"] = _allocated()
    if capture is not None:
        capture("setup", forks)

    transition_started = time.perf_counter()
    suffix_caches: list[Any] = []
    suffix_lengths: list[int] = []
    logits: list[torch.Tensor] = []
    ttfts: list[float] = []
    for fork, (request_id, query_tokens) in zip(forks, queries):
        request_started = time.perf_counter()
        fork.rebind_mutable_state()
        shared_residual = fork.document_residual
        if shared_residual is None:
            raise SharedPackedForkError(
                f"{request_id}: the document residual was released before use"
            )
        query_residual = adapter.continue_lower_replay(fork, query_tokens)
        call_log.append(
            {
                "request_id": request_id,
                "call": "continue_lower_replay",
                "chunk": "query_prefill",
                "layer_range": [0, depth],
                "position_offset": int(fork.document_length),
                "input_tokens": int(query_tokens.shape[-1]),
                "current_length_after": int(fork.current_length),
            }
        )
        suffix_cache = adapter.make_cache()
        adapter.run_suffix_cached_last_logits(
            [shared_residual], depth, suffix_cache, position_offset=0
        )
        call_log.append(
            {
                "request_id": request_id,
                "call": "run_suffix_cached_last_logits",
                "chunk": "document_residual_seed",
                "layer_range": [depth, adapter.num_layers],
                "position_offset": 0,
                "input_tokens": int(shared_residual.shape[1]),
            }
        )
        first_logits = adapter.run_suffix_cached_last_logits(
            [query_residual],
            depth,
            suffix_cache,
            position_offset=fork.document_length,
        )
        call_log.append(
            {
                "request_id": request_id,
                "call": "run_suffix_cached_last_logits",
                "chunk": "query_residual_prefill",
                "layer_range": [depth, adapter.num_layers],
                "position_offset": int(fork.document_length),
                "input_tokens": int(query_residual.shape[1]),
            }
        )
        fork.record_residual_binding(
            shared_residual=shared_residual,
            query_residual=query_residual,
            document_position_offset=0,
            query_position_offset=int(fork.document_length),
            salt=entry.salt,
        )
        fork.release_document_residual()
        suffix_caches.append(suffix_cache)
        suffix_lengths.append(int(fork.current_length))
        logits.append(first_logits)
        _sync()
        ttfts.append(time.perf_counter() - request_started)
    _sync()
    transition_seconds = time.perf_counter() - transition_started
    phase_allocated["after_all_transitions"] = _allocated()
    if capture is not None:
        capture("transition", forks)

    generated: list[list[int]] = [[] for _ in queries]
    tpots: list[list[float]] = [[] for _ in queries]
    finished = [False] * len(queries)
    decode_started = time.perf_counter()
    steady_samples: list[int] = []
    for step in range(max_new_tokens):
        for index, fork in enumerate(forks):
            if finished[index]:
                continue
            token = _argmax(logits[index])
            if token in eos_token_ids:
                finished[index] = True
                continue
            generated[index].append(token)
            if step + 1 >= max_new_tokens:
                continue
            token_tensor = torch.tensor(
                [[token]], device=logits[index].device, dtype=torch.long
            )
            started = time.perf_counter()
            residual = adapter.continue_lower_replay(fork, token_tensor)
            logits[index] = adapter.run_suffix_cached_last_logits(
                [residual],
                depth,
                suffix_caches[index],
                position_offset=suffix_lengths[index],
            )
            suffix_lengths[index] += 1
            _sync()
            tpots[index].append(time.perf_counter() - started)
        steady_samples.append(_allocated())
        if all(finished):
            break
    _sync()
    decode_seconds = time.perf_counter() - decode_started
    phase_allocated["decode_end"] = _allocated()

    initial_breakdowns = [
        {
            "shared_nbytes": fork.initial_shared_nbytes,
            "private_nbytes": fork.initial_private_nbytes,
        }
        for fork in forks
    ]
    final_breakdowns = [fork.memory_breakdown() for fork in forks]
    request_traces = []
    for index, (request_id, query_tokens) in enumerate(queries):
        fork = forks[index]
        final = final_breakdowns[index]
        suffix_bytes = cache_nbytes(suffix_caches[index])
        transient_peak = max(
            (
                int(event.get("transient_concat_nbytes") or 0)
                for event in fork.append_events
            ),
            default=0,
        )
        request_traces.append(
            MultiforkRequestTrace(
                request_id=request_id,
                query_token_count=int(query_tokens.shape[-1]),
                generated_token_ids=generated[index],
                ttft_seconds=ttfts[index],
                tpot_seconds=tpots[index],
                materialized_nbytes=final["private_nbytes"],
                steady_resident_nbytes=final["private_nbytes"] + suffix_bytes,
                suffix_cache_nbytes=suffix_bytes,
                initial_shared_nbytes=initial_breakdowns[index]["shared_nbytes"],
                initial_private_nbytes=initial_breakdowns[index]["private_nbytes"],
                final_shared_nbytes=final["shared_nbytes"],
                final_private_nbytes=final["private_nbytes"],
                append_event_count=len(fork.append_events),
                rebind_event_count=len(fork.rebind_events),
                rebind_nbytes=sum(
                    int(event["nbytes"]) for event in fork.rebind_events
                ),
                transient_concat_peak_nbytes=transient_peak,
            )
        )
    if capture is not None:
        capture("final", forks)
    steady = (
        round(statistics.median(steady_samples))
        if steady_samples
        else phase_allocated["decode_end"]
    )
    return MultiforkTrace(
        arm=(
            "qcomem-shared-packed"
            if entry.effective_share_mode == "shared-packed-view"
            else "qcomem-private-materialize"
        ),
        fork_mode=entry.effective_share_mode,
        rebind_policy=entry.rebind_policy,
        tail_policy=entry.tail_policy,
        request_traces=request_traces,
        forks=forks,
        entry=entry,
        setup_seconds=setup_seconds,
        transition_seconds=transition_seconds,
        decode_seconds=decode_seconds,
        phase_allocated_nbytes=phase_allocated,
        peak_allocated_nbytes=_peak_allocated(),
        baseline_allocated_nbytes=baseline,
        steady_allocated_nbytes=steady,
        append_events=[event for fork in forks for event in fork.append_events],
        rebind_events=[event for fork in forks for event in fork.rebind_events],
        residual_binding_events=[
            event for fork in forks for event in fork.residual_binding_events
        ],
        adapter_call_log=call_log,
        mask_size_call_forms=_mask_size_call_forms(forks),
    )


@torch.inference_mode()
def run_full_prefix_multifork(
    adapter: TorchSplitCausalLM,
    state: FullPrefixState,
    queries: Sequence[tuple[str, torch.Tensor]],
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    capture: Callable[[str, Sequence[Any]], None] | None = None,
) -> MultiforkTrace:
    """The same N-request protocol on the exact full-prefix arm.

    ``FullPrefixState.fork`` deep-clones, so every request owns a full private
    copy of the whole prefix cache.  Nothing is shared, and the trace records
    that as zero shared bytes rather than as an absent measurement.  This arm
    exists so the transient working set is measured for **both** methods, which
    is what Eq. 1's method-independence premise requires and does not have.
    """

    if not queries:
        raise SharedPackedForkError("at least one request is required")
    if len({request_id for request_id, _ in queries}) != len(queries):
        raise SharedPackedForkError("request ids must be unique")
    phase_allocated: dict[str, int] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    _sync()
    baseline = _allocated()
    phase_allocated["entry_resident"] = baseline

    setup_started = time.perf_counter()
    forks = [state.fork() for _ in queries]
    _sync()
    setup_seconds = time.perf_counter() - setup_started
    phase_allocated["after_all_forks"] = _allocated()
    initial_nbytes = [cache_nbytes(fork.cache) for fork in forks]
    if capture is not None:
        capture("setup", forks)

    transition_started = time.perf_counter()
    logits: list[torch.Tensor] = []
    ttfts: list[float] = []
    call_log: list[dict[str, Any]] = []
    for fork, (request_id, query_tokens) in zip(forks, queries):
        started = time.perf_counter()
        logits.append(adapter.continue_full_prefix(fork, query_tokens))
        _sync()
        ttfts.append(time.perf_counter() - started)
        call_log.append(
            {
                "request_id": request_id,
                "call": "continue_full_prefix",
                "chunk": "query_prefill",
                "layer_range": [0, adapter.num_layers],
                "position_offset": int(fork.document_length),
                "input_tokens": int(query_tokens.shape[-1]),
                "current_length_after": int(fork.current_length),
            }
        )
    _sync()
    transition_seconds = time.perf_counter() - transition_started
    phase_allocated["after_all_transitions"] = _allocated()
    if capture is not None:
        capture("transition", forks)

    generated: list[list[int]] = [[] for _ in queries]
    tpots: list[list[float]] = [[] for _ in queries]
    finished = [False] * len(queries)
    steady_samples: list[int] = []
    decode_started = time.perf_counter()
    for step in range(max_new_tokens):
        for index, fork in enumerate(forks):
            if finished[index]:
                continue
            token = _argmax(logits[index])
            if token in eos_token_ids:
                finished[index] = True
                continue
            generated[index].append(token)
            if step + 1 >= max_new_tokens:
                continue
            token_tensor = torch.tensor(
                [[token]], device=logits[index].device, dtype=torch.long
            )
            started = time.perf_counter()
            logits[index] = adapter.continue_full_prefix(fork, token_tensor)
            _sync()
            tpots[index].append(time.perf_counter() - started)
        steady_samples.append(_allocated())
        if all(finished):
            break
    _sync()
    decode_seconds = time.perf_counter() - decode_started
    phase_allocated["decode_end"] = _allocated()

    request_traces = []
    for index, (request_id, query_tokens) in enumerate(queries):
        final_nbytes = cache_nbytes(forks[index].cache)
        request_traces.append(
            MultiforkRequestTrace(
                request_id=request_id,
                query_token_count=int(query_tokens.shape[-1]),
                generated_token_ids=generated[index],
                ttft_seconds=ttfts[index],
                tpot_seconds=tpots[index],
                materialized_nbytes=initial_nbytes[index],
                steady_resident_nbytes=final_nbytes,
                suffix_cache_nbytes=0,
                initial_shared_nbytes=0,
                initial_private_nbytes=initial_nbytes[index],
                final_shared_nbytes=0,
                final_private_nbytes=final_nbytes,
                append_event_count=0,
                rebind_event_count=0,
                rebind_nbytes=0,
                transient_concat_peak_nbytes=0,
            )
        )
    if capture is not None:
        capture("final", forks)
    steady = (
        round(statistics.median(steady_samples))
        if steady_samples
        else phase_allocated["decode_end"]
    )
    return MultiforkTrace(
        arm="full-prefix",
        fork_mode="private-materialize",
        rebind_policy="not-applicable",
        tail_policy="not-applicable",
        request_traces=request_traces,
        forks=forks,
        entry=None,
        setup_seconds=setup_seconds,
        transition_seconds=transition_seconds,
        decode_seconds=decode_seconds,
        phase_allocated_nbytes=phase_allocated,
        peak_allocated_nbytes=_peak_allocated(),
        baseline_allocated_nbytes=baseline,
        steady_allocated_nbytes=steady,
        append_events=[],
        rebind_events=[],
        residual_binding_events=[],
        adapter_call_log=call_log,
        mask_size_call_forms=[],
    )


__all__ = [
    "ATTENTION_TENSOR_FIELDS",
    "BorrowedPrefixKVLayer",
    "CAPTURE_POINTS",
    "LINEAR_MUTABLE_FIELDS",
    "MultiforkRequestTrace",
    "MultiforkTrace",
    "PrivateMaterializedFork",
    "SharedPackedEntry",
    "SharedPackedFork",
    "SharedPackedForkError",
    "TensorSlot",
    "iter_tensor_slots",
    "opaque_storage_id",
    "prepare_shared_packed_entry",
    "run_full_prefix_multifork",
    "run_shared_packed_multifork",
    "storage_inventory_rows",
]
