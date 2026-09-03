from __future__ import annotations

"""R40 v26 producer-side compact functional rebinding.

The immutable RR2 runner and Qwen3.5 cache adapter remain byte-identical.  This
overlay wraps every resident request returned to the rank and installs one
rank-lifetime backbone pre/post-hook pair:

* the post-hook compact-clones recurrent endpoints after the already-installed
  native/primary-dispatch updater has run; and
* a cached single-token call compact-clones each linear layer's convolution
  endpoint before the unchanged Transformers causal-convolution update mutates
  that new request-private buffer in place.

The second step deliberately preserves the frozen single-token dispatch route.
It does not make a post-hoc clone after the previous endpoint has been mutated.
"""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch


EXPECTED_LINEAR_LAYERS = 30
STATE_INDEX = 0
INSTALL_MARKER = "r40-v26-construction-step-receipt-fix"


class CompactRebindError(RuntimeError):
    """Raised when the producer fix cannot be applied without guessing."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompactRebindError(message)


def _storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def fresh_compact_state(tensor: torch.Tensor) -> torch.Tensor:
    """Return a differentiable, fresh, canonical compact clone.

    Runtime checks are metadata-only.  A GPU equality/hash check here would
    add kernels or synchronization to the measured generation path; clone
    value preservation is instead covered by the overlay's unit/lifecycle
    tests and the existing formal exactness gates.
    """

    _require(isinstance(tensor, torch.Tensor), "state endpoint must be a tensor")
    _require(tensor.layout == torch.strided, "state endpoint must use strided layout")
    _require(tensor.numel() > 0, "state endpoint must be non-empty")
    old_storage = _storage_key(tensor)
    fresh = tensor.clone(memory_format=torch.contiguous_format)
    _require(fresh is not tensor, "compact clone reused the source object")
    _require(_storage_key(fresh) != old_storage, "compact clone reused source storage")
    _require(
        tuple(fresh.shape) == tuple(tensor.shape)
        and fresh.dtype == tensor.dtype
        and fresh.device == tensor.device,
        "compact clone changed shape/dtype/device",
    )
    _require(fresh.is_contiguous(), "compact clone is not canonical contiguous")
    _require(int(fresh.storage_offset()) == 0, "compact clone has nonzero offset")
    _require(
        int(fresh.untyped_storage().nbytes()) == _tensor_nbytes(fresh),
        "compact clone retains oversized backing storage",
    )
    return fresh


_COUNTERS: dict[str, int] = {
    "runtime_backbones_hooked": 0,
    "persistent_build_scopes_entered": 0,
    "persistent_build_scopes_completed": 0,
    "persistent_document_pre_hooks_bypassed": 0,
    "persistent_document_post_hooks_bypassed": 0,
    "borrowed_setup_calls_delegated": 0,
    "borrowed_requests_returned": 0,
    "materialized_setup_calls_canonicalized": 0,
    "materialized_setup_states_canonicalized": 0,
    "materialized_requests_returned": 0,
    "groups_wrapped": 0,
    "requests_wrapped": 0,
    "recurrent_layer_methods_preserved": 0,
    "cached_calls_postprocessed": 0,
    "cached_calls_aborted_before_postprocess": 0,
    "recurrent_states_post_rebound": 0,
    "multi_token_cached_calls_observed": 0,
    "single_token_cached_calls_observed": 0,
    "single_token_conv_states_pre_rebound": 0,
}


def _reset_counters() -> None:
    for key in _COUNTERS:
        _COUNTERS[key] = 0


def compact_rebind_receipt() -> dict[str, int | bool | str]:
    value: dict[str, int | bool | str] = dict(_COUNTERS)
    value.update(
        {
            "schema_version": "forkaudit-r40-v26-compact-rebind-runtime-v1",
            "install_marker": INSTALL_MARKER,
            "all_single_token_calls_rebound_exactly_30": (
                _COUNTERS["single_token_conv_states_pre_rebound"]
                == EXPECTED_LINEAR_LAYERS
                * _COUNTERS["single_token_cached_calls_observed"]
            ),
            "all_successful_cached_calls_post_rebound_exactly_30_recurrent_states": (
                _COUNTERS["recurrent_states_post_rebound"]
                == EXPECTED_LINEAR_LAYERS
                * _COUNTERS["cached_calls_postprocessed"]
            ),
            "all_cached_calls_accounted_exactly_once": (
                _COUNTERS["cached_calls_postprocessed"]
                + _COUNTERS["cached_calls_aborted_before_postprocess"]
                == _COUNTERS["multi_token_cached_calls_observed"]
                + _COUNTERS["single_token_cached_calls_observed"]
            ),
            "all_materialized_requests_directly_compact_cloned_60_states": (
                _COUNTERS["materialized_setup_calls_canonicalized"]
                == _COUNTERS["materialized_requests_returned"]
                and _COUNTERS["materialized_setup_states_canonicalized"]
                == 60 * _COUNTERS["materialized_requests_returned"]
            ),
            "all_request_construction_borrow_steps_delegated_exactly_once": (
                _COUNTERS["borrowed_setup_calls_delegated"]
                == _COUNTERS["requests_wrapped"]
            ),
            "all_request_final_policies_accounted_exactly_once": (
                _COUNTERS["borrowed_requests_returned"]
                + _COUNTERS["materialized_requests_returned"]
                == _COUNTERS["requests_wrapped"]
            ),
            "all_persistent_document_builds_scoped_exactly_once": (
                _COUNTERS["persistent_build_scopes_entered"]
                == _COUNTERS["persistent_build_scopes_completed"]
                == _COUNTERS["persistent_document_pre_hooks_bypassed"]
                == _COUNTERS["persistent_document_post_hooks_bypassed"]
            ),
        }
    )
    return value


def _validate_linear_indices(indices: Sequence[int]) -> tuple[int, ...]:
    value = tuple(indices)
    _require(
        len(value) == EXPECTED_LINEAR_LAYERS
        and all(type(index) is int and index >= 0 for index in value)
        and tuple(sorted(set(value))) == value,
        "formal linear-layer index set drift",
    )
    return value


def _require_compact(tensor: torch.Tensor, label: str) -> None:
    _require(tensor.is_contiguous(), f"{label} is not canonical contiguous")
    _require(int(tensor.storage_offset()) == 0, f"{label} has nonzero offset")
    _require(
        int(tensor.untyped_storage().nbytes()) == _tensor_nbytes(tensor),
        f"{label} retains oversized backing storage",
    )


def canonicalize_materialized_request_base(
    persistent: Any,
    request: Any,
    plan: Any,
) -> dict[str, Any]:
    """Directly clone the 60 persistent aliases into final compact requests.

    The call executes inside the existing passive-lineage mode.  Each clone's
    source is therefore the exact persistent tensor and its destination is the
    final live request mapping; there is no derived/post-builder clone edge.
    """

    indices = _validate_linear_indices(plan.linear_layer_indices)
    persistent_layers = getattr(persistent, "layers", None)
    request_layers = getattr(request, "layers", None)
    _require(
        isinstance(persistent_layers, (tuple, list))
        and isinstance(request_layers, (tuple, list)),
        "materialized cache layers must be sequences",
    )
    tensor_count = 0
    materialized_nbytes = 0
    for index in indices:
        for family in ("conv_states", "recurrent_states"):
            source_values = getattr(persistent_layers[index], family, None)
            request_values = getattr(request_layers[index], family, None)
            _require(
                isinstance(source_values, dict)
                and isinstance(request_values, dict)
                and sorted(source_values) == [STATE_INDEX]
                and sorted(request_values) == [STATE_INDEX],
                "materialized GDN state-family schema drift",
            )
            source = source_values[STATE_INDEX]
            request_alias = request_values[STATE_INDEX]
            _require(
                isinstance(source, torch.Tensor)
                and request_alias is source
                and _storage_key(request_alias) == _storage_key(source),
                "materialization source is not the exact persistent alias",
            )
            fresh = fresh_compact_state(request_alias)
            request_values[STATE_INDEX] = fresh
            materialized_nbytes += int(fresh.untyped_storage().nbytes())
            tensor_count += 1
    _require(tensor_count == 60, "formal materialization must clone 60 GDN tensors")
    _COUNTERS["materialized_setup_calls_canonicalized"] += 1
    _COUNTERS["materialized_setup_states_canonicalized"] += tensor_count
    return {
        "policy": "materialize-request-base-functional-rebind",
        "tensor_count": tensor_count,
        "borrowed_immutable_base_alias_count": 0,
        "materialized_request_base_nbytes": materialized_nbytes,
        "functional_rebind_after_transition": True,
    }


def install_compact_materialization_helper(
    original_build: Callable[..., Any],
) -> Callable[[], None]:
    """Replace only the builder global used for request-base materialization."""

    namespace = getattr(original_build, "__globals__", None)
    _require(isinstance(namespace, dict), "resident builder globals are unavailable")
    original_prepare = namespace.get("_prepare_request_gdn_base")
    borrowed_policy = namespace.get("GDN_BORROW_IMMUTABLE_BASE")
    materialized_policy = namespace.get("GDN_MATERIALIZE_REQUEST_BASE")
    _require(callable(original_prepare), "resident GDN prepare helper is absent")
    _require(
        borrowed_policy == "borrow-immutable-base-functional-rebind"
        and materialized_policy == "materialize-request-base-functional-rebind",
        "resident GDN policy constants drift",
    )

    def prepare(
        persistent: Any,
        request: Any,
        plan: Any,
        *,
        policy: str,
    ) -> dict[str, Any]:
        _require(
            namespace.get("_prepare_request_gdn_base") is prepare,
            "compact materialization helper identity drift",
        )
        if policy == borrowed_policy:
            result = original_prepare(
                persistent, request, plan, policy=policy
            )
            _COUNTERS["borrowed_setup_calls_delegated"] += 1
            return result
        _require(policy == materialized_policy, "unknown GDN base policy")
        return canonicalize_materialized_request_base(persistent, request, plan)

    namespace["_prepare_request_gdn_base"] = prepare

    def restore() -> None:
        _require(
            namespace.get("_prepare_request_gdn_base") is prepare,
            "compact materialization helper changed before restore",
        )
        namespace["_prepare_request_gdn_base"] = original_prepare

    return restore


def mark_compact_rebind_requests(
    group: Any,
    linear_layer_indices: Sequence[int],
) -> None:
    """Mark every returned request while preserving native updater identity."""

    indices = _validate_linear_indices(linear_layer_indices)
    requests = getattr(group, "requests", None)
    _require(isinstance(requests, (tuple, list)) and requests, "resident group is empty")
    _require(
        int(getattr(group, "resident_count", -1)) == len(requests),
        "resident group cardinality drift",
    )
    audit = getattr(group, "audit", None)
    gdn_policy = audit.get("gdn_base_policy") if isinstance(audit, dict) else None
    if gdn_policy == "materialize-request-base-functional-rebind":
        _COUNTERS["materialized_requests_returned"] += len(requests)
    elif gdn_policy == "borrow-immutable-base-functional-rebind":
        _COUNTERS["borrowed_requests_returned"] += len(requests)
    else:
        raise CompactRebindError("resident group GDN policy drift")
    for request in requests:
        _require(
            getattr(request, "_r40_v26_compact_rebind_indices", None) is None,
            "request compact rebind installed twice",
        )
        layers = getattr(request, "layers", None)
        _require(isinstance(layers, (tuple, list)), "request layers must be a sequence")
        for index in indices:
            _require(index < len(layers), "linear layer index exceeds request layers")
            layer = layers[index]
            inner = getattr(layer, "update_recurrent_state", None)
            mapping = getattr(layer, "recurrent_states", None)
            _require(callable(inner), "native recurrent updater is absent")
            _require(
                isinstance(mapping, dict)
                and sorted(mapping) == [STATE_INDEX]
                and isinstance(mapping[STATE_INDEX], torch.Tensor),
                "formal recurrent-state mapping drift",
            )
            _require(
                getattr(inner, "__func__", None) is not None,
                "native recurrent updater must remain a bound method",
            )
            layer._r40_v26_compact_rebind_mode = INSTALL_MARKER
            _COUNTERS["recurrent_layer_methods_preserved"] += 1
            if gdn_policy == "materialize-request-base-functional-rebind":
                conv = getattr(layer, "conv_states", {}).get(STATE_INDEX)
                _require(isinstance(conv, torch.Tensor), "materialized conv state absent")
                _require_compact(conv, "materialized conv state")
                _require_compact(mapping[STATE_INDEX], "materialized recurrent state")
        request._r40_v26_compact_rebind_indices = indices
        _COUNTERS["requests_wrapped"] += 1
    _COUNTERS["groups_wrapped"] += 1


def pre_rebind_cached_single_token_conv(
    cache: Any,
    linear_layer_indices: Sequence[int],
) -> None:
    """Give the unchanged in-place causal-conv route a fresh compact target."""

    indices = _validate_linear_indices(linear_layer_indices)
    _require(
        getattr(cache, "_r40_v26_compact_rebind_indices", None) == indices,
        "cached single-token call used an unwrapped request",
    )
    layers = getattr(cache, "layers", None)
    _require(isinstance(layers, (tuple, list)), "cached request layers must be a sequence")
    rebound = 0
    for index in indices:
        layer = layers[index]
        _require(
            getattr(layer, "_r40_v26_compact_rebind_mode", None) == INSTALL_MARKER,
            "cached single-token linear layer lacks producer fix",
        )
        mapping = getattr(layer, "conv_states", None)
        _require(
            isinstance(mapping, dict)
            and sorted(mapping) == [STATE_INDEX]
            and isinstance(mapping[STATE_INDEX], torch.Tensor),
            "formal convolution-state mapping drift",
        )
        prior = mapping[STATE_INDEX]
        fresh = fresh_compact_state(prior)
        mapping[STATE_INDEX] = fresh
        rebound += 1
    _require(rebound == EXPECTED_LINEAR_LAYERS, "single-token conv rebind count drift")
    _COUNTERS["single_token_cached_calls_observed"] += 1
    _COUNTERS["single_token_conv_states_pre_rebound"] += rebound


def _input_and_cache(
    args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> tuple[Any, Any, bool]:
    input_ids = kwargs.get("input_ids")
    if input_ids is None and args:
        input_ids = args[0]
    cache = kwargs.get("past_key_values")
    use_cache = kwargs.get("use_cache", cache is not None)
    return input_ids, cache, bool(use_cache)


@dataclass
class _PersistentBuildScope:
    """Exact one-call authority for the immutable document-cache prefill."""

    active: bool = False
    backbone: Any = None
    document: Any = None
    cache: Any = None
    pre_calls: int = 0
    post_calls: int = 0

    def enter(self, backbone: Any, document: Any) -> None:
        _require(not self.active, "persistent-build scope nested")
        _require(
            self.backbone is None
            and self.document is None
            and self.cache is None
            and self.pre_calls == 0
            and self.post_calls == 0,
            "persistent-build scope retained stale state",
        )
        _require(
            backbone is not None and isinstance(document, torch.Tensor),
            "persistent-build backbone/document schema drift",
        )
        self.active = True
        self.backbone = backbone
        self.document = document
        _COUNTERS["persistent_build_scopes_entered"] += 1

    def observe_pre(self, module: Any, input_ids: Any, cache: Any) -> None:
        _require(self.active, "persistent pre-hook outside build scope")
        _require(
            module is self.backbone
            and input_ids is self.document
            and self.cache is None
            and self.pre_calls == 0
            and self.post_calls == 0,
            "persistent-build scope observed multiple pre-hooks",
        )
        self.cache = cache
        self.pre_calls = 1
        _COUNTERS["persistent_document_pre_hooks_bypassed"] += 1

    def observe_post(self, module: Any, input_ids: Any, cache: Any) -> None:
        _require(self.active, "persistent post-hook outside build scope")
        _require(
            module is self.backbone
            and input_ids is self.document
            and self.cache is cache
            and self.pre_calls == 1
            and self.post_calls == 0,
            "persistent-build pre/post cache identity or count drift",
        )
        self.post_calls = 1
        _COUNTERS["persistent_document_post_hooks_bypassed"] += 1

    def complete(self, persistent: Any) -> None:
        _require(self.active, "persistent-build completion outside scope")
        _require(
            self.cache is persistent and self.pre_calls == 1 and self.post_calls == 1,
            "persistent-build did not return its exact single hooked cache",
        )
        _COUNTERS["persistent_build_scopes_completed"] += 1

    def leave(self) -> None:
        self.active = False
        self.backbone = None
        self.document = None
        self.cache = None
        self.pre_calls = 0
        self.post_calls = 0


def _wrapped_request_or_scoped_persistent(
    module: Any,
    input_ids: Any,
    cache: Any,
    indices: tuple[int, ...],
    scope: _PersistentBuildScope | None,
    *,
    phase: str,
) -> bool:
    """Return true for a wrapped request and false for one exact prefill cache."""

    marker = getattr(cache, "_r40_v26_compact_rebind_indices", None)
    if marker is not None:
        _require(marker == indices, f"cached {phase}-hook request marker drift")
        return True
    _require(
        scope is not None and scope.active,
        f"cached {phase}-hook used an unwrapped cache outside persistent-build scope",
    )
    if phase == "pre":
        scope.observe_pre(module, input_ids, cache)
    else:
        _require(phase == "post", "unknown cached-hook phase")
        scope.observe_post(module, input_ids, cache)
    return False


def install_backbone_pre_hook(
    backbone: Any,
    linear_layer_indices: Sequence[int],
    persistent_build_scope: _PersistentBuildScope | None = None,
) -> Any:
    """Install one rank-lifetime pre-hook over every formal backbone call."""

    indices = _validate_linear_indices(linear_layer_indices)
    register = getattr(backbone, "register_forward_pre_hook", None)
    _require(callable(register), "formal backbone has no forward-pre-hook interface")

    def before_forward(
        module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> None:
        input_ids, cache, use_cache = _input_and_cache(args, kwargs)
        if cache is None or not use_cache:
            return
        _require(
            isinstance(input_ids, torch.Tensor)
            and input_ids.ndim >= 1
            and input_ids.shape[-1] >= 1,
            "cached backbone call has invalid input_ids",
        )
        if not _wrapped_request_or_scoped_persistent(
            module, input_ids, cache, indices, persistent_build_scope, phase="pre"
        ):
            _require(
                int(input_ids.shape[-1]) > 1,
                "persistent document build unexpectedly used one token",
            )
            return
        if int(input_ids.shape[-1]) == 1:
            pre_rebind_cached_single_token_conv(cache, indices)
        else:
            _COUNTERS["multi_token_cached_calls_observed"] += 1

    handle = register(before_forward, with_kwargs=True)
    _COUNTERS["runtime_backbones_hooked"] += 1
    return handle


def post_rebind_cached_recurrent(
    cache: Any,
    linear_layer_indices: Sequence[int],
) -> None:
    """Compact-clone recurrent endpoints after the unchanged native updater."""

    indices = _validate_linear_indices(linear_layer_indices)
    _require(
        getattr(cache, "_r40_v26_compact_rebind_indices", None) == indices,
        "cached post-hook used an unwrapped request",
    )
    layers = getattr(cache, "layers", None)
    _require(isinstance(layers, (tuple, list)), "cached request layers must be a sequence")
    rebound = 0
    for index in indices:
        layer = layers[index]
        _require(
            getattr(layer, "_r40_v26_compact_rebind_mode", None) == INSTALL_MARKER,
            "cached recurrent layer lacks producer fix marker",
        )
        mapping = getattr(layer, "recurrent_states", None)
        _require(
            isinstance(mapping, dict)
            and sorted(mapping) == [STATE_INDEX]
            and isinstance(mapping[STATE_INDEX], torch.Tensor),
            "formal recurrent-state mapping drift after forward",
        )
        mapping[STATE_INDEX] = fresh_compact_state(mapping[STATE_INDEX])
        rebound += 1
    _require(rebound == EXPECTED_LINEAR_LAYERS, "post-forward recurrent rebind count drift")
    _COUNTERS["cached_calls_postprocessed"] += 1
    _COUNTERS["recurrent_states_post_rebound"] += rebound


def install_backbone_post_hook(
    backbone: Any,
    linear_layer_indices: Sequence[int],
    persistent_build_scope: _PersistentBuildScope | None = None,
) -> Any:
    """Install the recurrent endpoint post-hook without changing GDN methods."""

    indices = _validate_linear_indices(linear_layer_indices)
    register = getattr(backbone, "register_forward_hook", None)
    _require(callable(register), "formal backbone has no forward-hook interface")

    def after_forward(
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        _output: Any,
    ) -> None:
        input_ids, cache, use_cache = _input_and_cache(args, kwargs)
        if cache is None or not use_cache:
            return
        _require(
            isinstance(input_ids, torch.Tensor)
            and input_ids.ndim >= 1
            and input_ids.shape[-1] >= 1,
            "cached backbone post-hook has invalid input_ids",
        )
        if not _wrapped_request_or_scoped_persistent(
            module, input_ids, cache, indices, persistent_build_scope, phase="post"
        ):
            _require(
                int(input_ids.shape[-1]) > 1,
                "persistent document build unexpectedly used one token",
            )
            return
        if _output is None:
            _COUNTERS["cached_calls_aborted_before_postprocess"] += 1
            return
        post_rebind_cached_recurrent(cache, indices)

    return register(after_forward, with_kwargs=True, always_call=True)


@dataclass(frozen=True)
class CompactRebindInstall:
    update_mode: str = INSTALL_MARKER
    expected_linear_layers: int = EXPECTED_LINEAR_LAYERS
    scope: str = "all-rank-science-builds-and-cached-backbone-calls"


def install_compact_rebind_fix(runner_module: Any) -> tuple[Callable[[], None], CompactRebindInstall]:
    """Wrap the rank-wide producer interfaces without changing primary bytes."""

    _reset_counters()
    original_load = getattr(runner_module, "_load_formal_model_runtime", None)
    original_build = getattr(runner_module, "build_resident_request_group", None)
    original_convert = getattr(runner_module, "_convert_persistent", None)
    _require(callable(original_load), "formal runtime loader interface is absent")
    _require(callable(original_build), "resident group builder interface is absent")
    _require(callable(original_convert), "persistent conversion interface is absent")
    restore_materialization = install_compact_materialization_helper(original_build)
    hook_handles: list[Any] = []
    persistent_build_scope = _PersistentBuildScope()

    def load(*args: Any, **kwargs: Any) -> Any:
        _require(not hook_handles, "formal model runtime loaded more than once")
        runtime = original_load(*args, **kwargs)
        backbone = getattr(runtime, "backbone", None)
        plan = getattr(runtime, "plan", None)
        indices = getattr(plan, "linear_layer_indices", None)
        _require(backbone is not None and indices is not None, "runtime backbone/plan absent")
        hook_handles.append(
            install_backbone_pre_hook(backbone, indices, persistent_build_scope)
        )
        hook_handles.append(
            install_backbone_post_hook(backbone, indices, persistent_build_scope)
        )
        return runtime

    def convert(
        backbone: Any,
        plan: Any,
        document: torch.Tensor,
        *,
        resident_count: int,
    ) -> Any:
        persistent_build_scope.enter(backbone, document)
        try:
            result = original_convert(
                backbone,
                plan,
                document,
                resident_count=resident_count,
            )
            _require(
                isinstance(result, tuple) and len(result) == 2,
                "persistent conversion result schema drift",
            )
            persistent_build_scope.complete(result[0])
            return result
        finally:
            persistent_build_scope.leave()

    def build(cache: Any, plan: Any, **kwargs: Any) -> Any:
        group = original_build(cache, plan, **kwargs)
        mark_compact_rebind_requests(group, plan.linear_layer_indices)
        return group

    runner_module._load_formal_model_runtime = load
    runner_module.build_resident_request_group = build
    runner_module._convert_persistent = convert

    def restore() -> None:
        runner_module._load_formal_model_runtime = original_load
        runner_module.build_resident_request_group = original_build
        runner_module._convert_persistent = original_convert
        _require(not persistent_build_scope.active, "restore during persistent build")
        while hook_handles:
            handle = hook_handles.pop()
            remove = getattr(handle, "remove", None)
            _require(callable(remove), "backbone pre-hook handle has no remove")
            remove()
        restore_materialization()

    return restore, CompactRebindInstall()


__all__ = [
    "CompactRebindError",
    "CompactRebindInstall",
    "compact_rebind_receipt",
    "canonicalize_materialized_request_base",
    "fresh_compact_state",
    "install_backbone_post_hook",
    "install_backbone_pre_hook",
    "install_compact_rebind_fix",
    "install_compact_materialization_helper",
    "mark_compact_rebind_requests",
    "post_rebind_cached_recurrent",
    "pre_rebind_cached_single_token_conv",
]
