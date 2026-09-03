"""Evidence helpers for a bounded Hydragen-on-Qwen3.5 operator transfer.

This module deliberately contains no Hydragen implementation.  It verifies and
loads the previously captured Qwen3.5 tensors, constructs the two preregistered
shared-prefix cases, and provides an independent CPU-FP32 attention oracle.
The formal runner imports the pinned official Hydragen checkout separately.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


class HydragenTransferError(RuntimeError):
    """Fail-closed evidence or schema error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HydragenTransferError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


DTYPES = {
    "torch.bfloat16": torch.bfloat16,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
}


def _product(values: Iterable[int]) -> int:
    answer = 1
    for value in values:
        require(isinstance(value, int) and not isinstance(value, bool) and value > 0, "invalid shape")
        answer *= value
    return answer


def load_bound_tensor(raw_root: Path, record: dict[str, Any]) -> torch.Tensor:
    require(set(record) >= {"artifact", "dtype", "encoding", "shape"}, "tensor record schema drift")
    require(record["encoding"] == "torch-contiguous-raw-little-endian-v1", "unsupported encoding")
    dtype_name = record["dtype"]
    require(dtype_name in DTYPES, f"unsupported dtype {dtype_name}")
    shape = tuple(record["shape"])
    numel = _product(shape)
    artifact = record["artifact"]
    require(set(artifact) == {"bytes", "relative_path", "sha256"}, "artifact schema drift")
    path = raw_root / artifact["relative_path"]
    require(path.is_file(), f"missing tensor sidecar {path}")
    require(path.stat().st_size == artifact["bytes"], f"byte count drift for {path}")
    require(sha256_file(path) == artifact["sha256"], f"raw SHA drift for {path}")
    tensor = torch.from_file(str(path), shared=False, size=numel, dtype=DTYPES[dtype_name]).clone()
    return tensor.reshape(shape)


@dataclass(frozen=True)
class CapturedAttention:
    document_key: torch.Tensor  # [4095, 2, 256]
    document_value: torch.Tensor
    suffix_query: torch.Tensor  # [32, 16, 256]
    suffix_key: torch.Tensor  # [32, 2, 256]
    suffix_value: torch.Tensor
    metadata_sha256: str
    sidecar_sha256: dict[str, str]


def load_rr2_capture(metadata_path: Path, raw_root: Path) -> CapturedAttention:
    metadata = read_json(metadata_path)
    require(metadata.get("schema_version") == "qcomem-forkaudit-oracle-raw-v2", "oracle schema drift")
    selection = metadata.get("selection", {})
    require(selection.get("rank") == 0, "capture must be rank 0")
    require(selection.get("layer_index") == 3, "capture must be layer 3")
    require(selection.get("document_length") == 4095, "document length drift")
    contract = metadata.get("source_contract", {})
    require(contract.get("post_rope_qkv") is True, "capture is not post-RoPE Q/K/V")
    require(metadata.get("softmax_scale") == 0.0625, "softmax scale drift")

    tensors = metadata.get("tensors", {})
    names = (
        "physical_document_key_blocks",
        "physical_document_value_blocks",
        "document_block_table",
        "query",
    )
    require(all(name in tensors for name in names), "required capture tensors missing")
    append_events = metadata.get("append_events")
    require(isinstance(append_events, list) and len(append_events) == 1, "expected one suffix append")

    key_blocks = load_bound_tensor(raw_root, tensors["physical_document_key_blocks"])
    value_blocks = load_bound_tensor(raw_root, tensors["physical_document_value_blocks"])
    block_table = load_bound_tensor(raw_root, tensors["document_block_table"])
    query = load_bound_tensor(raw_root, tensors["query"])
    suffix_key = load_bound_tensor(raw_root, append_events[0]["key"])
    suffix_value = load_bound_tensor(raw_root, append_events[0]["value"])

    require(tuple(key_blocks.shape) == (32, 128, 2, 256), "document K geometry drift")
    require(tuple(value_blocks.shape) == tuple(key_blocks.shape), "document V geometry drift")
    require(tuple(block_table.shape) == (1, 32), "block table geometry drift")
    require(tuple(query.shape) == (1, 16, 32, 256), "query geometry drift")
    require(tuple(suffix_key.shape) == (1, 2, 32, 256), "suffix K geometry drift")
    require(tuple(suffix_value.shape) == tuple(suffix_key.shape), "suffix V geometry drift")

    table = block_table[0].to(torch.long)
    require(torch.equal(table, torch.arange(32, dtype=torch.long)), "unexpected document block order")
    document_key = key_blocks.index_select(0, table).reshape(4096, 2, 256)[:4095].contiguous()
    document_value = value_blocks.index_select(0, table).reshape(4096, 2, 256)[:4095].contiguous()
    suffix_query = query[0].transpose(0, 1).contiguous()
    suffix_key = suffix_key[0].transpose(0, 1).contiguous()
    suffix_value = suffix_value[0].transpose(0, 1).contiguous()
    require(torch.isfinite(document_key.float()).all().item(), "nonfinite document K")
    require(torch.isfinite(document_value.float()).all().item(), "nonfinite document V")
    require(torch.isfinite(suffix_query.float()).all().item(), "nonfinite suffix Q")
    require(torch.isfinite(suffix_key.float()).all().item(), "nonfinite suffix K")
    require(torch.isfinite(suffix_value.float()).all().item(), "nonfinite suffix V")

    sidecars = {
        name: tensors[name]["artifact"]["sha256"] for name in names
    }
    sidecars["suffix_key"] = append_events[0]["key"]["artifact"]["sha256"]
    sidecars["suffix_value"] = append_events[0]["value"]["artifact"]["sha256"]
    return CapturedAttention(
        document_key=document_key,
        document_value=document_value,
        suffix_query=suffix_query,
        suffix_key=suffix_key,
        suffix_value=suffix_value,
        metadata_sha256=sha256_file(metadata_path),
        sidecar_sha256=sidecars,
    )


@dataclass(frozen=True)
class TransferCase:
    resident_requests: int
    query_indices: tuple[int, ...]
    unique_lengths: tuple[int, ...]
    query: torch.Tensor  # [N, 1, 16, 256]
    unique_key: torch.Tensor  # [N, 32, 2, 256]
    unique_value: torch.Tensor
    shared_key: torch.Tensor  # [1, 4095, 2, 256]
    shared_value: torch.Tensor
    unique_seq_lens: torch.Tensor


def case_definition(resident_requests: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if resident_requests == 8:
        indices = tuple(range(3, 32, 4))
    elif resident_requests == 32:
        indices = tuple(range(32))
    else:
        raise HydragenTransferError("resident_requests must be 8 or 32")
    return indices, tuple(index + 1 for index in indices)


def build_transfer_case(capture: CapturedAttention, resident_requests: int) -> TransferCase:
    indices, lengths = case_definition(resident_requests)
    query = capture.suffix_query[list(indices)].unsqueeze(1).contiguous()
    unique_key = torch.zeros((resident_requests, 32, 2, 256), dtype=torch.bfloat16)
    unique_value = torch.zeros_like(unique_key)
    for row, length in enumerate(lengths):
        unique_key[row, :length].copy_(capture.suffix_key[:length])
        unique_value[row, :length].copy_(capture.suffix_value[:length])
    return TransferCase(
        resident_requests=resident_requests,
        query_indices=indices,
        unique_lengths=lengths,
        query=query,
        unique_key=unique_key,
        unique_value=unique_value,
        shared_key=capture.document_key.unsqueeze(0),
        shared_value=capture.document_value.unsqueeze(0),
        unique_seq_lens=torch.tensor(lengths, dtype=torch.int32),
    )


def build_replicated_dense_kv(case: TransferCase) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = case.resident_requests
    shared_k = case.shared_key.expand(n, -1, -1, -1)
    shared_v = case.shared_value.expand(n, -1, -1, -1)
    logical_k = torch.cat((shared_k, case.unique_key), dim=1).contiguous()
    logical_v = torch.cat((shared_v, case.unique_value), dim=1).contiguous()
    # The current vLLM paged FA2 frontend requires page sizes divisible by 16.
    # Materialize the one-token physical tail once, outside the measured call;
    # ``total_lens`` keeps it logically invisible to every request.
    physical_tokens = math.ceil(logical_k.shape[1] / 16) * 16
    dense_k = torch.zeros(
        (n, physical_tokens, *logical_k.shape[2:]), dtype=logical_k.dtype
    )
    dense_v = torch.zeros_like(dense_k)
    dense_k[:, : logical_k.shape[1]].copy_(logical_k)
    dense_v[:, : logical_v.shape[1]].copy_(logical_v)
    total_lens = case.unique_seq_lens + case.shared_key.shape[1]
    return dense_k, dense_v, total_lens


@torch.no_grad()
def cpu_fp32_oracle(case: TransferCase, scale: float = 0.0625) -> torch.Tensor:
    outputs = []
    shared_k = case.shared_key[0].float()
    shared_v = case.shared_value[0].float()
    for row, length in enumerate(case.unique_lengths):
        query = case.query[row, 0].float()
        key = torch.cat((shared_k, case.unique_key[row, :length].float()), dim=0)
        value = torch.cat((shared_v, case.unique_value[row, :length].float()), dim=0)
        key = key.repeat_interleave(8, dim=1)
        value = value.repeat_interleave(8, dim=1)
        scores = torch.einsum("hd,lhd->hl", query, key) * scale
        weights = torch.softmax(scores, dim=-1, dtype=torch.float32)
        outputs.append(torch.einsum("hl,lhd->hd", weights, value))
    return torch.stack(outputs).unsqueeze(1).contiguous()


def error_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    candidate = candidate.detach().cpu().float()
    reference = reference.detach().cpu().float()
    require(tuple(candidate.shape) == tuple(reference.shape), "comparison shape drift")
    delta = candidate - reference
    denominator = torch.linalg.vector_norm(reference).item()
    relative_l2 = torch.linalg.vector_norm(delta).item() / max(denominator, 1e-30)
    return {
        "finite": bool(torch.isfinite(candidate).all().item()),
        "max_abs": float(delta.abs().max().item()),
        "mean_abs": float(delta.abs().mean().item()),
        "relative_l2": float(relative_l2),
        "argmax_head_dimension_exact": bool(
            torch.equal(candidate.argmax(dim=-1), reference.argmax(dim=-1))
        ),
    }


def storage_accounting(case: TransferCase) -> dict[str, int | float]:
    n = case.resident_requests
    element_bytes = 2
    shared_tokens = case.shared_key.shape[1]
    padded_unique_tokens = case.unique_key.shape[1]
    kv_width = 2 * 256
    shared_bytes = 2 * shared_tokens * kv_width * element_bytes
    unique_bytes = 2 * n * padded_unique_tokens * kv_width * element_bytes
    hydragen_bytes = shared_bytes + unique_bytes
    replicated_bytes = 2 * n * (shared_tokens + padded_unique_tokens) * kv_width * element_bytes
    return {
        "shared_prefix_kv_bytes": shared_bytes,
        "padded_unique_kv_bytes": unique_bytes,
        "hydragen_total_kv_bytes": hydragen_bytes,
        "replicated_dense_total_kv_bytes": replicated_bytes,
        "replicated_over_hydragen_ratio": replicated_bytes / hydragen_bytes,
        "bytes_avoided": replicated_bytes - hydragen_bytes,
    }


def timing_statistics(times_ms: list[float]) -> dict[str, float | int]:
    require(len(times_ms) > 0, "empty timing sample")
    ordered = sorted(float(value) for value in times_ms)
    mean = sum(ordered) / len(ordered)
    variance = sum((value - mean) ** 2 for value in ordered) / len(ordered)

    def percentile(fraction: float) -> float:
        index = math.ceil(fraction * len(ordered)) - 1
        return ordered[max(0, min(index, len(ordered) - 1))]

    return {
        "sample_count": len(ordered),
        "mean_ms": mean,
        "median_ms": percentile(0.5),
        "p90_ms": percentile(0.9),
        "relative_standard_deviation": math.sqrt(variance) / mean,
    }
