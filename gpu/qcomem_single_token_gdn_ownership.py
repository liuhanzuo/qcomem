from __future__ import annotations

"""Post-discovery ownership repair for cached one-token Qwen3.5 GDN calls.

The Transformers single-token convolution kernel mutates its state argument in
place and bypasses ``Cache.update_conv_state``.  Borrowed-base requests must
therefore privatize their convolution states immediately before that first
call.  This helper changes ownership only; it neither executes the model nor
changes any audit predicate.
"""

import hashlib
import json
from typing import Any, Sequence

import torch


SCHEMA_VERSION = "qcomem-single-token-gdn-conv-privatization-v1"


class SingleTokenGDNOwnershipError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SingleTokenGDNOwnershipError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def storage_key(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


def byte_interval(tensor: torch.Tensor) -> tuple[int, int]:
    shape = tuple(int(value) for value in tensor.shape)
    stride = tuple(int(value) for value in tensor.stride())
    require(bool(shape) and len(shape) == len(stride), "tensor rank drift")
    require(all(size > 0 for size in shape), "empty GDN convolution state")
    minimum = int(tensor.storage_offset())
    maximum = minimum
    for size, step in zip(shape, stride):
        displacement = (size - 1) * step
        minimum += min(displacement, 0)
        maximum += max(displacement, 0)
    require(minimum >= 0 and maximum >= minimum, "tensor interval drift")
    width = int(tensor.element_size())
    start, end = minimum * width, (maximum + 1) * width
    require(end <= int(tensor.untyped_storage().nbytes()), "tensor interval outside storage")
    return start, end


def overlaps(left: torch.Tensor, right: torch.Tensor) -> bool:
    if storage_key(left) != storage_key(right):
        return False
    left_start, left_end = byte_interval(left)
    right_start, right_end = byte_interval(right)
    return left_start < right_end and right_start < left_end


def exact_alias(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        storage_key(left) == storage_key(right)
        and byte_interval(left) == byte_interval(right)
        and tuple(left.shape) == tuple(right.shape)
        and tuple(left.stride()) == tuple(right.stride())
        and int(left.storage_offset()) == int(right.storage_offset())
        and left.dtype == right.dtype
        and left.device == right.device
        and tensor_sha256(left) == tensor_sha256(right)
    )


def _conv_tensor(owner: Any, layer_index: int, state_index: int) -> torch.Tensor:
    layers = getattr(owner, "layers", None)
    require(isinstance(layers, (list, tuple)), "cache layers must be a sequence")
    require(0 <= layer_index < len(layers), "GDN layer index outside cache")
    states = getattr(layers[layer_index], "conv_states", None)
    require(isinstance(states, dict) and state_index in states, "conv state missing")
    tensor = states[state_index]
    require(isinstance(tensor, torch.Tensor), "conv state is not a tensor")
    return tensor


def prepare_borrowed_single_token_conv_transition(
    persistent: Any,
    requests: Sequence[Any],
    layer_indices: Sequence[int],
    *,
    request_index: int,
    state_index: int = 0,
) -> dict[str, Any]:
    """Privatize a borrowed request's conv state before an in-place 1-token call."""

    require(isinstance(requests, (list, tuple)) and len(requests) >= 2, "resident requests missing")
    require(type(request_index) is int and 0 <= request_index < len(requests), "request index drift")
    indices = tuple(int(value) for value in layer_indices)
    require(indices and len(set(indices)) == len(indices), "GDN layer plan drift")
    selected = requests[request_index]
    rows: list[dict[str, Any]] = []
    clones = 0
    for layer_index in indices:
        base = _conv_tensor(persistent, layer_index, state_index)
        current = _conv_tensor(selected, layer_index, state_index)
        peers = [
            _conv_tensor(request, layer_index, state_index)
            for index, request in enumerate(requests)
            if index != request_index
        ]
        before_sha = tensor_sha256(current)
        if exact_alias(current, base):
            clone = current.clone(memory_format=torch.preserve_format)
            selected.layers[layer_index].conv_states[state_index] = clone
            current = clone
            action = "cloned_borrowed_state"
            clones += 1
        else:
            require(not overlaps(current, base), "selected conv state partially overlaps persistent base")
            action = "already_private_noop"
        require(tensor_sha256(current) == before_sha, "conv privatization changed content")
        require(not overlaps(current, base), "privatized conv state overlaps persistent base")
        require(all(not overlaps(current, peer) for peer in peers), "privatized conv state overlaps peer")
        start, end = byte_interval(current)
        rows.append(
            {
                "layer_index": layer_index,
                "state_index": state_index,
                "action": action,
                "shape": [int(value) for value in current.shape],
                "stride": [int(value) for value in current.stride()],
                "storage_offset": int(current.storage_offset()),
                "dtype": str(current.dtype),
                "device": str(current.device),
                "storage_nbytes": int(current.untyped_storage().nbytes()),
                "tensor_nbytes": int(current.numel()) * int(current.element_size()),
                "byte_start": start,
                "byte_end_exclusive": end,
                "content_sha256": before_sha,
                "base_disjoint": True,
                "all_peers_disjoint": True,
            }
        )
    require(clones in (0, len(indices)), "mixed private/borrowed conv ownership state")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "request_index": request_index,
        "resident_count": len(requests),
        "state_index": state_index,
        "layer_indices": list(indices),
        "conv_tensor_count": len(indices),
        "cloned_tensor_count": clones,
        "already_private_tensor_count": len(indices) - clones,
        "ownership_only_change": True,
        "fault_id_specialization": False,
        "rows": rows,
    }
    receipt["rows_sha256"] = hashlib.sha256(canonical_bytes(rows)).hexdigest()
    return receipt


__all__ = [
    "SCHEMA_VERSION",
    "SingleTokenGDNOwnershipError",
    "byte_interval",
    "exact_alias",
    "overlaps",
    "prepare_borrowed_single_token_conv_transition",
]
