from __future__ import annotations

"""Strict Q16 ragged-batch adapter for vLLM 0.26 unified attention.

This module is deliberately separate from the single-request and resident
multi-fork experiments.  It only establishes that several request-local
``Q16PagedSequence`` objects backed by one physical K/V pool can be encoded in
the ragged metadata contract accepted by vLLM's Triton
``unified_attention`` entrypoint.

The adapter accepts post-RoPE query rows.  Position ids are checked, but are
not applied again.  Every request must be an ordinary no-padding causal tail;
custom bias, prefix-LM, sliding-window and soft-cap semantics fail closed.
Packing copies the (small) query tensors and block-table metadata only.  It
never materializes or concatenates logical K/V.

CPU execution is supported only through an injected mock kernel.  A real
kernel call, full-model integration, scheduler integration, concurrency and
throughput remain separate H20 gates.
"""

import ast
import hashlib
import importlib.metadata
import importlib.util
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import torch

from qcomem_vllm_paged_kernel import (
    Q16PagedArena,
    Q16PagedSequence,
    QComemPagedKernelError,
    validate_canonical_tail_causal_mask,
)


RAGGED_ADAPTER_MODE = "vllm_0_26_unified_attention_q16_ragged_shared_pool"
FROZEN_VLLM_VERSION = "0.26.0+cu129"
FROZEN_UNIFIED_ATTENTION_SOURCE_SHA256 = (
    "992a2bc892e2e2b43fbd3c8163816ccf7e97ced56cb542bd827adb0ddb2df9fa"
)
FROZEN_HELPERS_SOURCE_SHA256 = (
    "8c730611e7b3c5fb7579ec7846d56a2ab7e348ce06b39136da22072ecc363c95"
)
FROZEN_UNIFIED_ATTENTION_PARAMETERS = (
    "q",
    "k",
    "v",
    "out",
    "cu_seqlens_q",
    "max_seqlen_q",
    "seqused_k",
    "max_seqlen_k",
    "softmax_scale",
    "causal",
    "window_size",
    "block_table",
    "softcap",
    "q_descale",
    "k_descale",
    "v_descale",
    "seq_threshold_3D",
    "num_par_softmax_segments",
    "softmax_segm_output",
    "softmax_segm_max",
    "softmax_segm_expsum",
    "alibi_slopes",
    "output_scale",
    "qq_bias",
    "sinks",
    "mm_prefix_range",
    "rswa_prefix_lens",
    "rswa_window",
    "use_alibi_sqrt",
    "kv_quant_mode",
    "k_scale_cache",
    "v_scale_cache",
    "chunk_lookback",
    "use_td",
    "mm_prefix_clamp_sliding_window",
)
FROZEN_REQUIRED_PARAMETER_COUNT = 16

QWEN35_QUERY_HEADS = 16
QWEN35_KEY_VALUE_HEADS = 2
QWEN35_GQA_GROUPS = 8
QWEN35_HEAD_DIM = 256
QWEN35_PAGE_SIZE = 128
Q16_DTYPES = frozenset((torch.float16, torch.bfloat16))


class QComemRaggedBatchError(QComemPagedKernelError):
    """Raised before dispatch when a ragged-kernel assumption is violated."""


AttentionMask = Optional[Union[torch.Tensor, Mapping[str, torch.Tensor]]]


@dataclass(frozen=True)
class Q16RaggedRequest:
    """One post-RoPE query tail and its already-appended paged K/V state.

    ``query`` has shape ``[query_tokens, 16, 256]``. ``position_ids`` has
    shape ``[query_tokens]`` and must be the exact contiguous tail
    ``kv_length-query_tokens .. kv_length-1``.
    """

    sequence: Q16PagedSequence
    query: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: AttentionMask = None


@dataclass(frozen=True)
class Q16RaggedBatch:
    """Validated tensors passed directly to ``unified_attention``."""

    arena: Q16PagedArena
    q: torch.Tensor
    cu_seqlens_q: torch.Tensor
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    query_lengths: tuple[int, ...]
    kv_lengths: tuple[int, ...]
    position_starts: tuple[int, ...]
    max_seqlen_q: int
    max_seqlen_k: int


@dataclass(frozen=True)
class Q16RaggedBatchResult:
    """Flat kernel output plus zero-copy per-request views and audit facts."""

    flat_output: torch.Tensor
    sequence_outputs: tuple[torch.Tensor, ...]
    audit: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _function_contract(path: Path) -> tuple[tuple[str, ...], int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "unified_attention"
        ),
        None,
    )
    if function is None:
        raise QComemRaggedBatchError(
            "frozen vLLM source has no unified_attention definition"
        )
    parameters = tuple(argument.arg for argument in function.args.args)
    required = len(parameters) - len(function.args.defaults)
    return parameters, required


def audit_frozen_vllm_ragged_api() -> dict[str, Any]:
    """Audit package/source metadata without importing vLLM CUDA modules."""

    try:
        version: str | None = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        version = None
    spec = importlib.util.find_spec("vllm")
    if spec is None or spec.origin is None:
        return {
            "expected_version": FROZEN_VLLM_VERSION,
            "observed_version": version,
            "source_found": False,
            "matches_frozen_api": False,
            "kernel_entrypoint": (
                "vllm.v1.attention.ops.triton_unified_attention.unified_attention"
            ),
        }
    package_root = Path(spec.origin).resolve().parent
    source = package_root / "v1/attention/ops/triton_unified_attention.py"
    helpers = package_root / "v1/attention/ops/triton_attention_helpers.py"
    if not source.is_file() or not helpers.is_file():
        return {
            "expected_version": FROZEN_VLLM_VERSION,
            "observed_version": version,
            "source_found": False,
            "source_path": str(source),
            "helpers_path": str(helpers),
            "matches_frozen_api": False,
            "kernel_entrypoint": (
                "vllm.v1.attention.ops.triton_unified_attention.unified_attention"
            ),
        }
    parameters, required = _function_contract(source)
    source_sha = _sha256(source)
    helpers_sha = _sha256(helpers)
    matches = (
        version == FROZEN_VLLM_VERSION
        and source_sha == FROZEN_UNIFIED_ATTENTION_SOURCE_SHA256
        and helpers_sha == FROZEN_HELPERS_SOURCE_SHA256
        and parameters == FROZEN_UNIFIED_ATTENTION_PARAMETERS
        and required == FROZEN_REQUIRED_PARAMETER_COUNT
    )
    return {
        "expected_version": FROZEN_VLLM_VERSION,
        "observed_version": version,
        "source_found": True,
        "source_path": str(source),
        "source_sha256": source_sha,
        "expected_source_sha256": FROZEN_UNIFIED_ATTENTION_SOURCE_SHA256,
        "helpers_path": str(helpers),
        "helpers_sha256": helpers_sha,
        "expected_helpers_sha256": FROZEN_HELPERS_SOURCE_SHA256,
        "observed_parameters": parameters,
        "expected_parameters": FROZEN_UNIFIED_ATTENTION_PARAMETERS,
        "observed_required_parameter_count": required,
        "expected_required_parameter_count": FROZEN_REQUIRED_PARAMETER_COUNT,
        "matches_frozen_api": matches,
        "kernel_entrypoint": (
            "vllm.v1.attention.ops.triton_unified_attention.unified_attention"
        ),
    }


def _resolve_vllm_unified_attention() -> Callable[..., Any]:
    try:
        from vllm.v1.attention.ops.triton_unified_attention import (
            unified_attention,
        )
    except ImportError as error:  # pragma: no cover - frozen H20 environment only.
        raise QComemRaggedBatchError(
            "vLLM Triton unified_attention is unavailable"
        ) from error
    return unified_attention


def _validate_arena(arena: Q16PagedArena) -> None:
    if arena.batch_size != 1:
        raise QComemRaggedBatchError(
            "ragged adapter requires one logical sequence per request arena row"
        )
    if arena.page_size != QWEN35_PAGE_SIZE:
        raise QComemRaggedBatchError(
            f"Qwen3.5 ragged path requires page_size={QWEN35_PAGE_SIZE}"
        )
    if arena.num_key_value_heads != QWEN35_KEY_VALUE_HEADS:
        raise QComemRaggedBatchError(
            f"Qwen3.5 ragged path requires {QWEN35_KEY_VALUE_HEADS} KV heads"
        )
    if arena.head_dim != QWEN35_HEAD_DIM:
        raise QComemRaggedBatchError(
            f"Qwen3.5 ragged path requires head_dim={QWEN35_HEAD_DIM}"
        )
    key = arena.key_cache
    value = arena.value_cache
    if key.dtype not in Q16_DTYPES or value.dtype != key.dtype:
        raise QComemRaggedBatchError(
            "Q16 ragged path requires matching float16 or bfloat16 K/V"
        )
    if key.device != value.device:
        raise QComemRaggedBatchError("K/V pools must share a device")
    if key.shape != value.shape or key.ndim != 4:
        raise QComemRaggedBatchError("K/V pools must have one matching rank-four shape")
    expected_tail = (
        QWEN35_PAGE_SIZE,
        QWEN35_KEY_VALUE_HEADS,
        QWEN35_HEAD_DIM,
    )
    if tuple(key.shape[1:]) != expected_tail:
        raise QComemRaggedBatchError("K/V pool is not Qwen3.5 NHD block layout")
    if not key.is_contiguous() or not value.is_contiguous():
        raise QComemRaggedBatchError("K/V NHD block pools must be contiguous")
    if key.untyped_storage().data_ptr() == value.untyped_storage().data_ptr():
        raise QComemRaggedBatchError("K/V pools unexpectedly alias one storage")
    table = arena.document_block_table
    if table.dtype != torch.int32 or table.device != key.device or table.ndim != 2:
        raise QComemRaggedBatchError("document block table dtype/device/shape is invalid")
    expected_document_blocks = math.ceil(arena.document_length / arena.page_size)
    if (
        arena.document_length < 1
        or arena.document_blocks_per_sequence != expected_document_blocks
        or tuple(table.shape) != (arena.batch_size, expected_document_blocks)
    ):
        raise QComemRaggedBatchError("document block table cardinality is invalid")
    if not bool(((table >= 0) & (table < int(key.shape[0]))).all().item()):
        raise QComemRaggedBatchError("document block table contains an out-of-pool id")
    if int(torch.unique(table).numel()) != table.numel():
        raise QComemRaggedBatchError("document block table aliases physical blocks")


def _mask_tensor(mask: AttentionMask) -> torch.Tensor | None:
    if isinstance(mask, Mapping):
        if set(mask) != {"full_attention"}:
            raise QComemRaggedBatchError(
                "ragged mask mapping must contain only full_attention"
            )
        mask = mask["full_attention"]
    if mask is not None and not isinstance(mask, torch.Tensor):
        raise QComemRaggedBatchError("attention mask must be tensor, mapping, or None")
    return mask


def _validate_request(
    request: Q16RaggedRequest,
    *,
    arena: Q16PagedArena,
    request_index: int,
) -> tuple[int, int, int, torch.Tensor, frozenset[int]]:
    if not isinstance(request, Q16RaggedRequest):
        raise QComemRaggedBatchError(
            f"request {request_index} is not Q16RaggedRequest"
        )
    sequence = request.sequence
    if not isinstance(sequence, Q16PagedSequence):
        raise QComemRaggedBatchError(
            f"request {request_index} has no Q16PagedSequence"
        )
    if sequence.arena is not arena:
        raise QComemRaggedBatchError(
            "all ragged requests must share exactly one physical K/V arena"
        )
    if sequence.sequence_length != arena.document_length + sequence.appended_tokens:
        raise QComemRaggedBatchError(
            f"request {request_index} sequence length/cursor is inconsistent"
        )
    query = request.query
    if not isinstance(query, torch.Tensor) or query.ndim != 3:
        raise QComemRaggedBatchError(
            f"request {request_index} query must be [tokens, heads, dim]"
        )
    query_length = int(query.shape[0])
    if tuple(query.shape[1:]) != (QWEN35_QUERY_HEADS, QWEN35_HEAD_DIM):
        raise QComemRaggedBatchError(
            f"request {request_index} query geometry is not 16Q/head_dim256"
        )
    if query_length < 1:
        raise QComemRaggedBatchError(f"request {request_index} query is empty")
    if query.dtype != arena.key_cache.dtype or query.device != arena.key_cache.device:
        raise QComemRaggedBatchError(
            f"request {request_index} query dtype/device differs from K/V pool"
        )
    kv_length = int(sequence.sequence_length)
    if query_length > sequence.appended_tokens or query_length > kv_length:
        raise QComemRaggedBatchError(
            f"request {request_index} query is not an already-appended K/V tail"
        )
    position_start = kv_length - query_length
    position_ids = request.position_ids
    if not isinstance(position_ids, torch.Tensor):
        raise QComemRaggedBatchError(
            f"request {request_index} position_ids must be a tensor"
        )
    if position_ids.dtype != torch.long or position_ids.device != query.device:
        raise QComemRaggedBatchError(
            f"request {request_index} position_ids dtype/device is invalid"
        )
    if tuple(position_ids.shape) != (query_length,):
        raise QComemRaggedBatchError(
            f"request {request_index} position_ids shape is invalid"
        )
    expected_positions = torch.arange(
        position_start,
        kv_length,
        dtype=torch.long,
        device=query.device,
    )
    if not torch.equal(position_ids, expected_positions):
        raise QComemRaggedBatchError(
            f"request {request_index} position_ids are not the contiguous causal tail"
        )
    mask = _mask_tensor(request.attention_mask)
    if mask is not None:
        if mask.device != query.device:
            raise QComemRaggedBatchError(
                f"request {request_index} attention mask is on the wrong device"
            )
        if mask.dtype != torch.bool and not mask.is_floating_point():
            raise QComemRaggedBatchError(
                f"request {request_index} attention mask dtype is invalid"
            )
    try:
        validate_canonical_tail_causal_mask(
            mask,
            batch_size=1,
            query_length=query_length,
            total_length=kv_length,
            device=query.device,
        )
    except (QComemPagedKernelError, RuntimeError) as error:
        raise QComemRaggedBatchError(
            f"request {request_index} mask is not canonical tail-causal: {error}"
        ) from error

    active_table = sequence.active_block_table
    expected_blocks = math.ceil(kv_length / arena.page_size)
    if (
        active_table.dtype != torch.int32
        or active_table.device != arena.key_cache.device
        or tuple(active_table.shape) != (1, expected_blocks)
    ):
        raise QComemRaggedBatchError(
            f"request {request_index} active block table shape/dtype/device is invalid"
        )
    if not bool(
        ((active_table >= 0) & (active_table < int(arena.key_cache.shape[0])))
        .all()
        .item()
    ):
        raise QComemRaggedBatchError(
            f"request {request_index} block table contains an out-of-pool id"
        )
    if int(torch.unique(active_table).numel()) != active_table.numel():
        raise QComemRaggedBatchError(
            f"request {request_index} aliases two logical blocks"
        )

    document_blocks = arena.document_blocks_per_sequence
    immutable_document_blocks = document_blocks
    if arena.document_length % arena.page_size:
        immutable_document_blocks -= 1
    if immutable_document_blocks:
        expected_prefix = arena.document_block_table[:, :immutable_document_blocks]
        if not torch.equal(
            active_table[:, :immutable_document_blocks], expected_prefix
        ):
            raise QComemRaggedBatchError(
                f"request {request_index} does not retain the immutable document prefix"
            )
    document_ids = frozenset(
        int(item)
        for item in arena.document_block_table.detach().to(device="cpu").reshape(-1)
    )
    active_ids = frozenset(
        int(item) for item in active_table.detach().to(device="cpu").reshape(-1)
    )
    private_ids = active_ids.difference(document_ids)
    expected_private_blocks = expected_blocks - immutable_document_blocks
    if len(private_ids) != expected_private_blocks:
        raise QComemRaggedBatchError(
            f"request {request_index} mutable tail is not entirely request-private"
        )
    reservations = sequence.reservations
    if (
        not isinstance(reservations, torch.Tensor)
        or reservations.dtype != torch.int64
        or reservations.device.type != "cpu"
        or tuple(reservations.shape)
        != (arena.batch_size, arena.private_blocks_per_sequence)
    ):
        raise QComemRaggedBatchError(
            f"request {request_index} private reservation metadata is invalid"
        )
    reserved_ids = frozenset(int(item) for item in reservations.reshape(-1))
    if not private_ids.issubset(reserved_ids):
        raise QComemRaggedBatchError(
            f"request {request_index} request-private block is outside its reservation"
        )
    return query_length, kv_length, position_start, active_table, private_ids


def prepare_q16_ragged_batch(
    requests: Sequence[Q16RaggedRequest],
) -> Q16RaggedBatch:
    """Validate and pack ragged query/table metadata without touching full K/V."""

    packed_requests = tuple(requests)
    if not packed_requests:
        raise QComemRaggedBatchError("ragged request list must not be empty")
    first = packed_requests[0]
    if not isinstance(first, Q16RaggedRequest) or not isinstance(
        first.sequence, Q16PagedSequence
    ):
        raise QComemRaggedBatchError("first request has no Q16PagedSequence")
    arena = first.sequence.arena
    _validate_arena(arena)
    if len({id(request.sequence) for request in packed_requests}) != len(
        packed_requests
    ):
        raise QComemRaggedBatchError(
            "one mutable Q16PagedSequence cannot appear twice in a ragged batch"
        )

    rows: list[torch.Tensor] = []
    query_lengths: list[int] = []
    kv_lengths: list[int] = []
    position_starts: list[int] = []
    private_owners: set[int] = set()
    for request_index, request in enumerate(packed_requests):
        query_length, kv_length, position_start, table, private_ids = (
            _validate_request(
                request,
                arena=arena,
                request_index=request_index,
            )
        )
        overlap = private_owners.intersection(private_ids)
        if overlap:
            raise QComemRaggedBatchError(
                "two ragged requests alias request-private physical blocks"
            )
        private_owners.update(private_ids)
        query_lengths.append(query_length)
        kv_lengths.append(kv_length)
        position_starts.append(position_start)
        rows.append(table)

    total_query_tokens = sum(query_lengths)
    q = torch.empty(
        (total_query_tokens, QWEN35_QUERY_HEADS, QWEN35_HEAD_DIM),
        dtype=arena.key_cache.dtype,
        device=arena.key_cache.device,
    )
    cursor = 0
    for request, length in zip(packed_requests, query_lengths):
        q[cursor : cursor + length].copy_(request.query)
        cursor += length

    prefix = [0]
    for length in query_lengths:
        prefix.append(prefix[-1] + length)
    cu_seqlens_q = torch.tensor(
        prefix,
        dtype=torch.int32,
        device=arena.key_cache.device,
    )
    seq_lens = torch.tensor(
        kv_lengths,
        dtype=torch.int32,
        device=arena.key_cache.device,
    )
    max_blocks = max(int(row.shape[1]) for row in rows)
    # Zero is a valid immutable block and is safe padding: seqused_k keeps the
    # kernel from reading logical table entries beyond each request length.
    block_table = torch.zeros(
        (len(rows), max_blocks),
        dtype=torch.int32,
        device=arena.key_cache.device,
    )
    for row_index, row in enumerate(rows):
        block_table[row_index, : row.shape[1]].copy_(row[0])

    if not q.is_contiguous() or not block_table.is_contiguous():
        raise QComemRaggedBatchError("packed query/table tensors are not contiguous")
    return Q16RaggedBatch(
        arena=arena,
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens=seq_lens,
        block_table=block_table,
        query_lengths=tuple(query_lengths),
        kv_lengths=tuple(kv_lengths),
        position_starts=tuple(position_starts),
        max_seqlen_q=max(query_lengths),
        max_seqlen_k=max(kv_lengths),
    )


def q16_ragged_paged_attention(
    requests: Sequence[Q16RaggedRequest],
    *,
    softmax_scale: float = QWEN35_HEAD_DIM**-0.5,
    _kernel: Callable[..., Any] | None = None,
) -> Q16RaggedBatchResult:
    """Dispatch one strict ragged batch through vLLM unified attention."""

    if isinstance(softmax_scale, bool) or not math.isfinite(float(softmax_scale)):
        raise QComemRaggedBatchError("softmax_scale must be finite")
    if float(softmax_scale) <= 0:
        raise QComemRaggedBatchError("softmax_scale must be positive")
    batch = prepare_q16_ragged_batch(requests)
    if _kernel is None:
        if batch.q.device.type != "cuda":
            raise QComemRaggedBatchError(
                "real unified_attention dispatch requires CUDA; inject a CPU mock"
            )
        _kernel = _resolve_vllm_unified_attention()
    output = torch.empty_like(batch.q)
    _kernel(
        q=batch.q,
        k=batch.arena.key_cache,
        v=batch.arena.value_cache,
        out=output,
        cu_seqlens_q=batch.cu_seqlens_q,
        max_seqlen_q=batch.max_seqlen_q,
        seqused_k=batch.seq_lens,
        max_seqlen_k=batch.max_seqlen_k,
        softmax_scale=float(softmax_scale),
        causal=True,
        window_size=(-1, -1),
        block_table=batch.block_table,
        softcap=0.0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
    )
    outputs: list[torch.Tensor] = []
    cursor = 0
    query_offsets = [0]
    for query_length in batch.query_lengths:
        stop = cursor + query_length
        outputs.append(output[cursor:stop])
        cursor = stop
        query_offsets.append(cursor)
    audit = {
        "adapter_mode": RAGGED_ADAPTER_MODE,
        "sequence_count": len(batch.query_lengths),
        "query_lengths": batch.query_lengths,
        "kv_lengths": batch.kv_lengths,
        "position_starts": batch.position_starts,
        "query_is_ragged": len(set(batch.query_lengths)) > 1,
        "kv_is_ragged": len(set(batch.kv_lengths)) > 1,
        "flattened_query_shape": tuple(batch.q.shape),
        "cu_seqlens_q": tuple(query_offsets),
        "seqused_k": batch.kv_lengths,
        "block_table_shape": tuple(batch.block_table.shape),
        "max_seqlen_q": batch.max_seqlen_q,
        "max_seqlen_k": batch.max_seqlen_k,
        "query_heads": QWEN35_QUERY_HEADS,
        "key_value_heads": QWEN35_KEY_VALUE_HEADS,
        "gqa_groups": QWEN35_GQA_GROUPS,
        "head_dim": QWEN35_HEAD_DIM,
        "page_size": QWEN35_PAGE_SIZE,
        "dtype": str(batch.q.dtype),
        "device": str(batch.q.device),
        "kernel_calls": 1,
        "causal": True,
        "window_size": (-1, -1),
        "softcap": 0.0,
        "position_ids_exact_tail_checked": True,
        "canonical_tail_causal_masks_checked": True,
        "full_kv_concatenations": 0,
        "full_kv_materializations": 0,
        "query_flatten_copy_tokens": int(batch.q.shape[0]),
        "shared_physical_kv_pool": True,
        "adapter_feasibility_only": True,
        "scheduler_integration_claimed": False,
        "concurrent_execution_claimed": False,
        "throughput_claimed": False,
        "full_model_correctness_claimed": False,
        "h20_kernel_gate_required": True,
        "h20_kernel_gate_passed": False,
    }
    return Q16RaggedBatchResult(
        flat_output=output,
        sequence_outputs=tuple(outputs),
        audit=audit,
    )


__all__ = [
    "FROZEN_HELPERS_SOURCE_SHA256",
    "FROZEN_REQUIRED_PARAMETER_COUNT",
    "FROZEN_UNIFIED_ATTENTION_PARAMETERS",
    "FROZEN_UNIFIED_ATTENTION_SOURCE_SHA256",
    "FROZEN_VLLM_VERSION",
    "Q16RaggedBatch",
    "Q16RaggedBatchResult",
    "Q16RaggedRequest",
    "QComemRaggedBatchError",
    "RAGGED_ADAPTER_MODE",
    "audit_frozen_vllm_ragged_api",
    "prepare_q16_ragged_batch",
    "q16_ragged_paged_attention",
]
