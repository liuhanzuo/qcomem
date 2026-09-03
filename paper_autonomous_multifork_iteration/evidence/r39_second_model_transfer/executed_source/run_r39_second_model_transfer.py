#!/usr/bin/env python3
"""One-rank producer for the frozen R39 second-model transfer."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import struct
import subprocess
import tempfile
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import torch

from qwen35_dense_adapter import (
    EXPECTED_DENSE_LAYER_TYPES,
    LowerReplayState,
    PackedLowerReplayState,
    TorchSplitCausalLM,
    registered_layer_route_passes,
)


PROTOCOL = "forkaudit-qwen35-dense-transformers-transfer-v1"
MODEL_ID = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
WORLD_SIZE = 8
FANOUTS = (1, 2)
VOCAB_SIZE = 248320


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    require(not path.exists(), f"refusing to replace output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_bound_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"{label} raw SHA-256 drift")
    value = json.loads(raw)
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def int64_le_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().to(dtype=torch.int64, device="cpu").contiguous().view(-1)
    digest = hashlib.sha256()
    for item in value.tolist():
        digest.update(struct.pack("<q", int(item)))
    return digest.hexdigest()


def tensor_raw_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous()
    if value.numel() == 0:
        return b""
    return value.view(torch.uint8).cpu().numpy().tobytes(order="C")


def tensor_receipt(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "content_sha256": sha256_bytes(tensor_raw_bytes(tensor)),
        "finite": bool(torch.isfinite(tensor).all().item())
        if tensor.is_floating_point()
        else True,
    }


@dataclass
class TensorSlot:
    path: str
    parent: Any
    key: Any
    tensor: torch.Tensor


def iter_tensor_slots(root: Any) -> Iterator[TensorSlot]:
    visited: set[int] = set()

    def visit(value: Any, path: str, parent: Any, key: Any) -> Iterator[TensorSlot]:
        if isinstance(value, torch.Tensor):
            yield TensorSlot(path, parent, key, value)
            return
        object_id = id(value)
        if object_id in visited:
            return
        visited.add(object_id)
        if isinstance(value, Mapping):
            for child_key in sorted(value, key=lambda item: str(item)):
                yield from visit(value[child_key], f"{path}/{child_key}", value, child_key)
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                yield from visit(child, f"{path}/{index}", value, index)
        elif hasattr(value, "__dict__"):
            for name in sorted(vars(value)):
                yield from visit(getattr(value, name), f"{path}/{name}", value, name)

    yield from visit(root, "root", None, None)


def tensor_tree_receipt(root: Any) -> dict[str, Any]:
    rows = [
        {"path": slot.path, **tensor_receipt(slot.tensor)}
        for slot in iter_tensor_slots(root)
    ]
    return {
        "tensor_count": len(rows),
        "rows": rows,
        "content_sha256": sha256_bytes(canonical_bytes(rows)),
    }


def state_content_receipt(state: LowerReplayState | PackedLowerReplayState) -> dict[str, Any]:
    residual = tensor_tree_receipt(state.document_residual)
    cache = tensor_tree_receipt(state.cache)
    identity = {
        "depth": int(state.depth),
        "document_length": int(state.document_length),
        "current_length": int(state.current_length),
        "document_residual": residual,
        "cache": cache,
    }
    return {**identity, "state_content_sha256": sha256_bytes(canonical_bytes(identity))}


def storage_inventory(root: Any, *, salt: str, role: str) -> dict[str, Any]:
    require(bool(salt), "empty storage salt")
    rows = []
    for slot in iter_tensor_slots(root):
        tensor = slot.tensor
        require(tensor.is_contiguous(), f"non-contiguous authorizing view: {slot.path}")
        storage = tensor.untyped_storage()
        storage_bytes = int(storage.nbytes())
        start = int(tensor.storage_offset() * tensor.element_size())
        nbytes = int(tensor.numel() * tensor.element_size())
        end = start + nbytes
        require(0 <= start <= end <= storage_bytes, "tensor view outside storage")
        device = str(tensor.device)
        opaque = sha256_bytes(
            f"{salt}|{device}|{int(storage.data_ptr())}|{storage_bytes}".encode()
        )
        rows.append(
            {
                "path": slot.path,
                "role": role,
                "device": device,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "contiguous": True,
                "storage_bytes": storage_bytes,
                "view_start_bytes": start,
                "view_end_bytes": end,
                "view_nbytes": nbytes,
                "storage_id_sha256": opaque,
            }
        )
    require(rows, f"empty authorizing storage inventory: {role}")
    return {
        "role": role,
        "storage_salt_domain_sha256": sha256_bytes(salt.encode()),
        "tensor_rows": len(rows),
        "rows": rows,
        "inventory_sha256": sha256_bytes(canonical_bytes(rows)),
    }


def overlapping_ranges(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    overlaps = []
    for a in left["rows"]:
        for b in right["rows"]:
            if a["storage_id_sha256"] != b["storage_id_sha256"]:
                continue
            start = max(a["view_start_bytes"], b["view_start_bytes"])
            end = min(a["view_end_bytes"], b["view_end_bytes"])
            if start < end:
                overlaps.append(
                    {
                        "storage_id_sha256": a["storage_id_sha256"],
                        "left_path": a["path"],
                        "right_path": b["path"],
                        "intersection_start_bytes": start,
                        "intersection_end_bytes": end,
                    }
                )
    return sorted(
        overlaps,
        key=lambda row: (
            row["storage_id_sha256"],
            row["left_path"],
            row["right_path"],
            row["intersection_start_bytes"],
        ),
    )


def disjointness_receipt(
    inventories: Sequence[dict[str, Any]],
    *,
    forbidden: Sequence[dict[str, Any]] = (),
    require_peer_comparison: bool,
) -> dict[str, Any]:
    require(inventories, "no request inventories")
    domains = {
        row["storage_salt_domain_sha256"] for row in (*inventories, *forbidden)
    }
    require(len(domains) == 1, "storage salt-domain mismatch")
    comparisons = []
    tensor_pairs = 0
    peer_count = 0
    for index, left in enumerate(inventories):
        for right in inventories[index + 1 :]:
            overlap = overlapping_ranges(left, right)
            peer_count += 1
            tensor_pairs += len(left["rows"]) * len(right["rows"])
            comparisons.append(
                {
                    "relation": "request_request",
                    "left_role": left["role"],
                    "right_role": right["role"],
                    "overlap_ranges": overlap,
                    "disjoint": not overlap,
                }
            )
        for right in forbidden:
            overlap = overlapping_ranges(left, right)
            tensor_pairs += len(left["rows"]) * len(right["rows"])
            comparisons.append(
                {
                    "relation": "request_persistent_base",
                    "left_role": left["role"],
                    "right_role": right["role"],
                    "overlap_ranges": overlap,
                    "disjoint": not overlap,
                }
            )
    if require_peer_comparison:
        require(peer_count > 0, "N=2 request-peer comparison is vacuous")
    require(comparisons and tensor_pairs > 0, "ownership comparison is vacuous")
    return {
        "predicate_id": "PRIVATE_MUTABLE_STORAGE",
        "passed": all(row["disjoint"] for row in comparisons),
        "peer_comparison_count": peer_count,
        "comparison_count": len(comparisons),
        "tensor_pair_comparison_count": tensor_pairs,
        "comparisons": comparisons,
    }


class LogitBundle:
    def __init__(self) -> None:
        self.payload = bytearray()
        self.records: list[dict[str, Any]] = []
        self.ids: set[str] = set()

    def add(self, record_id: str, logits: torch.Tensor) -> dict[str, Any]:
        require(record_id and record_id not in self.ids, "duplicate logit record ID")
        value = logits.detach().float().cpu().contiguous()
        require(tuple(value.shape) == (1, VOCAB_SIZE), "full-vocabulary logit shape drift")
        require(bool(torch.isfinite(value).all().item()), "non-finite logits")
        raw = value.numpy().astype("<f4", copy=False).tobytes(order="C")
        offset = len(self.payload)
        record = {
            "record_id": record_id,
            "offset_bytes": offset,
            "nbytes": len(raw),
            "shape": [1, VOCAB_SIZE],
            "dtype": "float32-le",
            "content_sha256": sha256_bytes(raw),
            "argmax": int(torch.argmax(value, dim=-1).item()),
        }
        self.payload.extend(raw)
        self.records.append(record)
        self.ids.add(record_id)
        return record

    def write(self, path: Path) -> dict[str, Any]:
        require(self.records, "empty logit sidecar")
        atomic_write(path, bytes(self.payload))
        return {
            "schema_version": "r39-full-vocabulary-fp32-sidecar-v1",
            "logical_name": path.name,
            "bytes": len(self.payload),
            "sha256": sha256_bytes(bytes(self.payload)),
            "record_count": len(self.records),
            "records": self.records,
            "terminal_closure": {
                "first_offset_bytes": self.records[0]["offset_bytes"],
                "last_end_offset_bytes": (
                    self.records[-1]["offset_bytes"] + self.records[-1]["nbytes"]
                ),
                "exact_byte_coverage": True,
            },
        }


def gpu_identity(expected_uuid: str) -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid, "GPU isolation drift")
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={expected_uuid}",
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    fields = [item.strip() for item in result.stdout.strip().split(",")]
    require(len(fields) == 3 and fields[0] == expected_uuid, "GPU UUID query drift")
    properties = torch.cuda.get_device_properties(0)
    receipt = {
        "cuda_visible_devices": expected_uuid,
        "uuid": fields[0],
        "name": fields[1],
        "total_memory_mib": int(fields[2]),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }
    require("H20" in receipt["name"], "rank is not assigned an H20")
    require(receipt["compute_capability"] == [9, 0], "H20 compute capability drift")
    require(receipt["bf16_supported"], "BF16 support absent")
    return receipt


def allocator_snapshot(phase: str) -> dict[str, Any]:
    torch.cuda.synchronize()
    return {
        "phase": phase,
        "allocated_bytes": int(torch.cuda.memory_allocated(0)),
        "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
    }


def reference_trajectory(
    adapter: TorchSplitCausalLM,
    document: torch.Tensor,
    query: torch.Tensor,
    *,
    depth: int,
    steps: int,
    mode: str,
    request_index: int,
    sidecar: LogitBundle,
) -> dict[str, Any]:
    require(mode in {"official_one_shot", "manual_one_shot", "official_dynamic_cache"}, "bad reference mode")
    generated: list[int] = []
    records = []
    continuation = query.clone()
    full_state = None
    logits = None
    if mode == "official_dynamic_cache":
        full_state = adapter.write_full_prefix(document)
        logits = adapter.continue_full_prefix(full_state, continuation)
    for step in range(steps):
        if mode == "official_one_shot":
            logits = adapter.full_last_logits(torch.cat((document, continuation), dim=1))
        elif mode == "manual_one_shot":
            logits = adapter.manual_one_shot_last_logits(
                torch.cat((document, continuation), dim=1), depth
            )
        assert logits is not None
        record = sidecar.add(f"reference/{mode}/request-{request_index}/step-{step}", logits)
        records.append(record)
        token = record["argmax"]
        generated.append(token)
        token_tensor = torch.tensor([[token]], dtype=query.dtype, device=query.device)
        continuation = torch.cat((continuation, token_tensor), dim=1)
        if mode == "official_dynamic_cache" and step + 1 < steps:
            assert full_state is not None
            logits = adapter.continue_full_prefix(full_state, token_tensor)
    return {
        "request_index": request_index,
        "query_token_ids_int64_le_sha256": int64_le_sha256(query),
        "generated_token_ids": generated,
        "step_logit_record_ids": [row["record_id"] for row in records],
        "step_logit_sha256": [row["content_sha256"] for row in records],
        "semantic_steps": steps,
    }


@dataclass
class RequestSession:
    request_index: int
    query: torch.Tensor
    state: LowerReplayState
    suffix_cache: Any
    logits: torch.Tensor
    suffix_length: int
    generated: list[int]
    records: list[dict[str, Any]]


def session_inventory(session: RequestSession, *, salt: str, arm: str, phase: str) -> dict[str, Any]:
    return storage_inventory(
        {"lower_cache": session.state.cache, "suffix_cache": session.suffix_cache},
        salt=salt,
        role=f"{arm}/request-{session.request_index}/{phase}/mutable-cache",
    )


def start_session(
    adapter: TorchSplitCausalLM,
    state: LowerReplayState,
    query: torch.Tensor,
    request_index: int,
) -> RequestSession:
    require(state.current_length == state.document_length, "request position starts non-canonically")
    query_residual = adapter.continue_lower_replay(state, query)
    suffix_cache = adapter.make_cache()
    adapter.run_suffix_cached_last_logits(
        [state.document_residual],
        state.depth,
        suffix_cache,
        position_offset=0,
    )
    logits = adapter.run_suffix_cached_last_logits(
        [query_residual],
        state.depth,
        suffix_cache,
        position_offset=state.document_length,
    )
    return RequestSession(
        request_index=request_index,
        query=query,
        state=state,
        suffix_cache=suffix_cache,
        logits=logits,
        suffix_length=state.current_length,
        generated=[],
        records=[],
    )


def run_arm(
    adapter: TorchSplitCausalLM,
    states: Sequence[LowerReplayState],
    queries: Sequence[torch.Tensor],
    *,
    fanout: int,
    arm: str,
    depth: int,
    steps: int,
    salt: str,
    persistent_cache_inventory: dict[str, Any] | None,
    sidecar: LogitBundle,
) -> dict[str, Any]:
    require(len(states) == len(queries) == fanout, "arm cardinality drift")
    setup = [
        storage_inventory(
            state.cache,
            salt=salt,
            role=f"{arm}/request-{index}/setup/lower-cache",
        )
        for index, state in enumerate(states)
    ]
    setup_ownership = disjointness_receipt(
        setup,
        forbidden=([persistent_cache_inventory] if persistent_cache_inventory else []),
        require_peer_comparison=fanout == 2,
    )
    sessions = [
        start_session(adapter, state, query, index)
        for index, (state, query) in enumerate(zip(states, queries))
    ]
    first = [session_inventory(row, salt=salt, arm=arm, phase="first-query") for row in sessions]
    first_ownership = disjointness_receipt(
        first,
        forbidden=([persistent_cache_inventory] if persistent_cache_inventory else []),
        require_peer_comparison=fanout == 2,
    )

    for step in range(steps):
        for session in sessions:
            record = sidecar.add(
                f"fanout-{fanout}/{arm}/request-{session.request_index}/step-{step}",
                session.logits,
            )
            session.records.append(record)
            token = record["argmax"]
            session.generated.append(token)
            if step + 1 < steps:
                token_tensor = torch.tensor(
                    [[token]], dtype=session.query.dtype, device=session.query.device
                )
                token_residual = adapter.continue_lower_replay(session.state, token_tensor)
                session.logits = adapter.run_suffix_cached_last_logits(
                    [token_residual],
                    depth,
                    session.suffix_cache,
                    position_offset=session.suffix_length,
                )
                session.suffix_length += 1

    final = [session_inventory(row, salt=salt, arm=arm, phase="final") for row in sessions]
    final_ownership = disjointness_receipt(
        final,
        forbidden=([persistent_cache_inventory] if persistent_cache_inventory else []),
        require_peer_comparison=fanout == 2,
    )
    semantics = []
    for session in sessions:
        lower_receipt = state_content_receipt(session.state)
        suffix_receipt = tensor_tree_receipt(session.suffix_cache)
        semantics.append(
            {
                "request_index": session.request_index,
                "query_token_ids_int64_le_sha256": int64_le_sha256(session.query),
                "generated_token_ids": session.generated,
                "step_logit_record_ids": [row["record_id"] for row in session.records],
                "step_logit_sha256": [row["content_sha256"] for row in session.records],
                "final_lower_state_sha256": lower_receipt["state_content_sha256"],
                "final_lower_cache_content_sha256": lower_receipt["cache"]["content_sha256"],
                "final_suffix_cache_content_sha256": suffix_receipt["content_sha256"],
            }
        )
    return {
        "arm": arm,
        "fanout": fanout,
        "construction": (
            "independent-write_lower_replay-per-request"
            if arm == "deep_materialized"
            else "one-persistent-Q16-state-then-fork-per-request"
        ),
        "ownership": {
            "setup": setup_ownership,
            "first_query": first_ownership,
            "final": final_ownership,
        },
        "semantics": semantics,
    }


def run_controls(
    adapter: TorchSplitCausalLM,
    packed_base: PackedLowerReplayState,
    document: torch.Tensor,
    *,
    depth: int,
    salt: str,
) -> list[dict[str, Any]]:
    # C1: mutable-cache alias.  The injected object assignment is deliberate
    # and confined to fresh control forks.
    clean_a, clean_b = packed_base.fork(), packed_base.fork()
    clean_inv = [
        storage_inventory(clean_a.cache, salt=salt, role="control-alias/clean-a"),
        storage_inventory(clean_b.cache, salt=salt, role="control-alias/clean-b"),
    ]
    clean_alias = disjointness_receipt(
        clean_inv, forbidden=(), require_peer_comparison=True
    )
    clean_b.cache = clean_a.cache
    mutant_inv = [
        storage_inventory(clean_a.cache, salt=salt, role="control-alias/mutant-a"),
        storage_inventory(clean_b.cache, salt=salt, role="control-alias/mutant-b"),
    ]
    mutant_alias = disjointness_receipt(
        mutant_inv, forbidden=(), require_peer_comparison=True
    )

    # C2: one raw bit in a fresh persistent Q16 residual.
    prefix_control = adapter.write_lower_replay(document, depth).quantize(
        bits=16, attention_bits=16, linear_bits=16, group_size=64
    )
    data = prefix_control.document_residual.data
    before_storage = storage_inventory(data, salt=salt, role="control-prefix/residual")
    expected_digest = tensor_receipt(data)["content_sha256"]
    clean_prefix = tensor_receipt(data)["content_sha256"] == expected_digest
    raw_words = data.view(torch.int16).reshape(-1)
    require(raw_words.numel() > 0, "prefix control residual is empty")
    raw_words[0].bitwise_xor_(1)
    mutant_digest = tensor_receipt(data)["content_sha256"]
    after_storage = storage_inventory(data, salt=salt, role="control-prefix/residual")
    stable_storage = [
        (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
        for row in before_storage["rows"]
    ] == [
        (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
        for row in after_storage["rows"]
    ]

    # C3: position/current-length canonicality.
    position_control = packed_base.fork()
    expected_length = position_control.document_length
    clean_position = position_control.current_length == expected_length
    position_control.current_length += 1
    mutant_position = position_control.current_length == expected_length

    # C4: exact dense 3:1 route.
    clean_route = list(EXPECTED_DENSE_LAYER_TYPES)
    mutant_route = list(clean_route)
    mutant_route[0] = "full_attention"
    return [
        {
            "control_id": "MUTABLE_CACHE_ALIAS",
            "expected_first_failing_predicate": "PRIVATE_MUTABLE_STORAGE",
            "matched_clean": {"PRIVATE_MUTABLE_STORAGE": clean_alias["passed"]},
            "mutant": {"PRIVATE_MUTABLE_STORAGE": mutant_alias["passed"]},
            "mutant_overlap_ranges": mutant_alias["comparisons"][0]["overlap_ranges"],
            "classification": (
                "detected_expected_predicate"
                if clean_alias["passed"] and not mutant_alias["passed"]
                else "escaped_or_clean_failure"
            ),
        },
        {
            "control_id": "PREFIX_CONTENT_MUTATION",
            "expected_first_failing_predicate": "PERSISTENT_PREFIX_IMMUTABLE",
            "matched_clean": {"PERSISTENT_PREFIX_IMMUTABLE": clean_prefix},
            "mutant": {
                "PERSISTENT_PREFIX_IMMUTABLE": mutant_digest == expected_digest
            },
            "expected_content_sha256": expected_digest,
            "mutant_content_sha256": mutant_digest,
            "storage_identity_stable": stable_storage,
            "classification": (
                "detected_expected_predicate"
                if clean_prefix and mutant_digest != expected_digest and stable_storage
                else "escaped_or_clean_failure"
            ),
        },
        {
            "control_id": "POSITION_OFFSET_DRIFT",
            "expected_first_failing_predicate": "POSITION_CANONICAL",
            "matched_clean": {"POSITION_CANONICAL": clean_position},
            "mutant": {"POSITION_CANONICAL": mutant_position},
            "classification": (
                "detected_expected_predicate"
                if clean_position and not mutant_position
                else "escaped_or_clean_failure"
            ),
        },
        {
            "control_id": "DENSE_MASK_ROUTE_RELABEL",
            "expected_first_failing_predicate": "LAYER_TYPE_MASK_ROUTE",
            "matched_clean": {
                "LAYER_TYPE_MASK_ROUTE": registered_layer_route_passes(clean_route)
            },
            "mutant": {
                "LAYER_TYPE_MASK_ROUTE": registered_layer_route_passes(mutant_route)
            },
            "classification": (
                "detected_expected_predicate"
                if registered_layer_route_passes(clean_route)
                and not registered_layer_route_passes(mutant_route)
                else "escaped_or_clean_failure"
            ),
        },
    ]


def run_rank(args: argparse.Namespace) -> None:
    static = load_bound_json(args.static, args.expected_static_sha256, "static preregistration")
    source = load_bound_json(args.source_manifest, args.expected_source_sha256, "source manifest")
    model_authority = load_bound_json(
        args.model_authority,
        args.expected_model_authority_sha256,
        "model authority",
    )
    gpu_assignment = load_bound_json(
        args.gpu_assignment,
        args.expected_gpu_assignment_sha256,
        "GPU assignment",
    )
    require(static["protocol"] == PROTOCOL, "protocol drift")
    require(static["model"]["repo_id"] == MODEL_ID, "model ID drift")
    require(static["model"]["revision"] == MODEL_REVISION, "model revision drift")
    require(source["protocol"] == PROTOCOL, "source protocol drift")
    require(model_authority["repo_id"] == MODEL_ID, "model authority ID drift")
    require(model_authority["revision"] == MODEL_REVISION, "model authority revision drift")
    require(args.rank in range(WORLD_SIZE), "rank outside registered range")
    gpu_rows = gpu_assignment["rows"]
    require(len(gpu_rows) == WORLD_SIZE, "GPU assignment rank count drift")
    expected_gpu_row = gpu_rows[args.rank]
    require(expected_gpu_row["rank"] == args.rank, "GPU assignment ordering drift")
    require(expected_gpu_row["uuid"] == args.expected_gpu_uuid, "expected GPU UUID drift")

    torch.cuda.set_device(0)
    hardware = gpu_identity(args.expected_gpu_uuid)
    import transformers
    from transformers import AutoModelForImageTextToText

    require(transformers.__version__ == "5.14.1", "Transformers version drift")
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model_root),
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    adapter = TorchSplitCausalLM(model)
    geometry = adapter.dense_geometry_receipt()
    require(geometry["matches_registered"], "dense model geometry drift")

    config = static["formal_config"]
    depth = int(config["split_depth"])
    steps = int(config["semantic_steps"])
    salt = static["storage_receipt_salt"]
    input_row = static["rank_inputs"][args.rank]
    require(input_row["rank"] == args.rank, "input rank ordering drift")
    document = torch.tensor(
        [input_row["document_token_ids"]], dtype=torch.int64, device="cuda:0"
    )
    queries = [
        torch.tensor([row["token_ids"]], dtype=torch.int64, device="cuda:0")
        for row in input_row["queries"]
    ]
    require(tuple(document.shape) == (1, 64), "document shape drift")
    require(len(queries) == 2 and all(tuple(row.shape) == (1, 8) for row in queries), "query shape drift")
    require(
        int64_le_sha256(document) == input_row["document_token_ids_int64_le_sha256"],
        "document token digest drift",
    )
    for query, frozen in zip(queries, input_row["queries"]):
        require(
            int64_le_sha256(query) == frozen["token_ids_int64_le_sha256"],
            "query token digest drift",
        )

    torch.cuda.reset_peak_memory_stats(0)
    allocator = [allocator_snapshot("model_loaded")]
    with torch.inference_mode():
        sidecar = LogitBundle()
        references: dict[str, list[dict[str, Any]]] = {}
        for mode in ("official_one_shot", "manual_one_shot", "official_dynamic_cache"):
            references[mode] = [
                reference_trajectory(
                    adapter,
                    document,
                    query,
                    depth=depth,
                    steps=steps,
                    mode=mode,
                    request_index=index,
                    sidecar=sidecar,
                )
                for index, query in enumerate(queries)
            ]

        exact_base = adapter.write_lower_replay(document, depth)
        packed_base = exact_base.quantize(
            bits=config["q16"]["residual_bits"],
            attention_bits=config["q16"]["attention_bits"],
            linear_bits=config["q16"]["linear_bits"],
            group_size=config["q16"]["group_size"],
        )
        base_before = state_content_receipt(packed_base)
        persistent_cache_inventory = storage_inventory(
            packed_base.cache,
            salt=salt,
            role="persistent-q16-base/mutable-cache-store",
        )
        persistent_residual_inventory = storage_inventory(
            packed_base.document_residual,
            salt=salt,
            role="persistent-q16-base/immutable-boundary",
        )
        allocator.append(allocator_snapshot("persistent_q16_ready"))

        cells = []
        for fanout in FANOUTS:
            materialized_states = [
                adapter.write_lower_replay(document, depth) for _ in range(fanout)
            ]
            persistent_states = [packed_base.fork() for _ in range(fanout)]
            materialized = run_arm(
                adapter,
                materialized_states,
                queries[:fanout],
                fanout=fanout,
                arm="deep_materialized",
                depth=depth,
                steps=steps,
                salt=salt,
                persistent_cache_inventory=persistent_cache_inventory,
                sidecar=sidecar,
            )
            persistent = run_arm(
                adapter,
                persistent_states,
                queries[:fanout],
                fanout=fanout,
                arm="persistent_q16",
                depth=depth,
                steps=steps,
                salt=salt,
                persistent_cache_inventory=persistent_cache_inventory,
                sidecar=sidecar,
            )
            cells.append(
                {
                    "fanout": fanout,
                    "arms": {
                        "deep_materialized": materialized,
                        "persistent_q16": persistent,
                    },
                }
            )

        base_after = state_content_receipt(packed_base)
        controls = run_controls(
            adapter,
            packed_base,
            document,
            depth=depth,
            salt=salt,
        )
        allocator.append(allocator_snapshot("rank_complete"))
        sidecar_receipt = sidecar.write(args.sidecar)

    shard = {
        "schema_version": "r39-second-model-transfer-shard-v1",
        "protocol": PROTOCOL,
        "rank": args.rank,
        "world_size": WORLD_SIZE,
        "scientific_run_valid": True,
        "identity": {
            "static_manifest_sha256": args.expected_static_sha256,
            "source_manifest_sha256": args.expected_source_sha256,
            "model_authority_sha256": args.expected_model_authority_sha256,
            "gpu_assignment_sha256": args.expected_gpu_assignment_sha256,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "hardware": hardware,
            "geometry": geometry,
            "dispatch": adapter.dispatch_receipt(),
            "input": {
                "source_id": input_row["source_id"],
                "source_object": input_row["source_object"],
                "document_token_ids_int64_le_sha256": int64_le_sha256(document),
                "query_token_ids_int64_le_sha256": [
                    int64_le_sha256(query) for query in queries
                ],
            },
        },
        "references": references,
        "persistent_base": {
            "representation": "lossless_q16",
            "before": base_before,
            "after": base_after,
            "content_immutable": (
                base_before["state_content_sha256"] == base_after["state_content_sha256"]
            ),
            "mutable_cache_inventory": persistent_cache_inventory,
            "immutable_boundary_inventory": persistent_residual_inventory,
        },
        "cells": cells,
        "controls": controls,
        "allocator_context": allocator,
        "sidecar": sidecar_receipt,
        "claim_boundary": static["claim_boundary"],
    }
    atomic_write(args.output, canonical_bytes(shard))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--model-authority", type=Path, required=True)
    parser.add_argument("--gpu-assignment", type=Path, required=True)
    parser.add_argument("--expected-static-sha256", required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-model-authority-sha256", required=True)
    parser.add_argument("--expected-gpu-assignment-sha256", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--sidecar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    run_rank(parse_args())
