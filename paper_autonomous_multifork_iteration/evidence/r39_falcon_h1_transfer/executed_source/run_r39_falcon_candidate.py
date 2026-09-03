#!/usr/bin/env python3
"""One-rank candidate producer for the frozen R39 Falcon-H1 transfer."""

from __future__ import annotations

import argparse
import ast
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

from falcon_h1_adapter import (
    EXPECTED_DEPTH,
    EXPECTED_LAYER_TYPES,
    FAMILY_ORDER,
    LowerReplayState,
    PackedLowerReplayState,
    TorchSplitFalconH1,
    cache_family_receipt,
    composed_cache_family_receipt,
    registered_geometry_passes,
)


PROTOCOL = "forkaudit-falcon-h1-hybrid-transformers-transfer-v1"
MODEL_ID = "tiiuae/Falcon-H1-0.5B-Base"
MODEL_REVISION = "59fb76e8c5d3fc7441b062be638e1ba0afd5c687"
MODELSCOPE_REVISION = "a475c769e108fd1dc6cfe41e342305d36431ef20"
WORLD_SIZE = 8
FANOUTS = (1, 2)
VOCAB_SIZE = 32784


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
    step_receipts: list[dict[str, Any]]


def session_inventory(session: RequestSession, *, salt: str, arm: str, phase: str) -> dict[str, Any]:
    return storage_inventory(
        {"lower_cache": session.state.cache, "suffix_cache": session.suffix_cache},
        salt=salt,
        role=f"{arm}/request-{session.request_index}/{phase}/mutable-cache",
    )


def start_session(
    adapter: TorchSplitFalconH1,
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
        step_receipts=[],
    )


def run_arm(
    adapter: TorchSplitFalconH1,
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
            family_receipt = composed_cache_family_receipt(
                session.state.cache,
                session.suffix_cache,
                depth=depth,
                expected_sequence_length=session.state.current_length,
            )
            session.step_receipts.append(
                {
                    "step": step,
                    "record_id": record["record_id"],
                    "logit_sha256": record["content_sha256"],
                    "generated_token_id": token,
                    "cache_family_receipt": family_receipt,
                }
            )
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
                "steps": session.step_receipts,
                "final_lower_state_sha256": lower_receipt["state_content_sha256"],
                "final_lower_cache_content_sha256": lower_receipt["cache"]["content_sha256"],
                "final_suffix_cache_content_sha256": suffix_receipt["content_sha256"],
                "final_composed_cache_family_sha256": session.step_receipts[-1][
                    "cache_family_receipt"
                ]["rows_sha256"],
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


def state_family_binding_passes(rows: Sequence[dict[str, Any]]) -> bool:
    expected = [(layer, family) for layer in range(36) for family in FAMILY_ORDER]
    observed = [(row.get("layer_index"), row.get("family")) for row in rows]
    return observed == expected and len(rows) == 144


def reference_independence_passes(source: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    banned = ("falcon" + "_h1_adapter", "run_r39_falcon_" + "candidate", "qcomem" + "_torch")
    forbidden_import = any(
        any(name == token or name.startswith(token + ".") for token in banned)
        for name in names
    )
    forbidden_call = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in {"__import__", "exec", "eval"}:
            forbidden_call = True
        if isinstance(node.func, ast.Attribute):
            dotted = []
            value: Any = node.func
            while isinstance(value, ast.Attribute):
                dotted.append(value.attr)
                value = value.value
            if isinstance(value, ast.Name):
                dotted.append(value.id)
            name = ".".join(reversed(dotted))
            if name in {"importlib.import_module", "importlib.util.spec_from_file_location"}:
                forbidden_call = True
    return not forbidden_import and not forbidden_call


def run_controls(
    adapter: TorchSplitFalconH1,
    packed_base: PackedLowerReplayState,
    query: torch.Tensor,
    *,
    depth: int,
    salt: str,
) -> list[dict[str, Any]]:
    # C1: deliberately alias fresh child caches.
    clean_a, clean_b = packed_base.fork(), packed_base.fork()
    clean_inv = [
        storage_inventory(clean_a.cache, salt=salt, role="control-alias/clean-a"),
        storage_inventory(clean_b.cache, salt=salt, role="control-alias/clean-b"),
    ]
    clean_alias = disjointness_receipt(clean_inv, forbidden=(), require_peer_comparison=True)
    clean_b.cache = clean_a.cache
    mutant_inv = [
        storage_inventory(clean_a.cache, salt=salt, role="control-alias/mutant-a"),
        storage_inventory(clean_b.cache, salt=salt, role="control-alias/mutant-b"),
    ]
    mutant_alias = disjointness_receipt(mutant_inv, forbidden=(), require_peer_comparison=True)

    # Build one clean, complete 36-layer receipt for omission/relabel controls.
    family_state = packed_base.fork()
    family_session = start_session(adapter, family_state, query, 0)
    family_receipt = composed_cache_family_receipt(
        family_session.state.cache,
        family_session.suffix_cache,
        depth=depth,
        expected_sequence_length=family_session.state.current_length,
    )
    clean_rows = family_receipt["rows"]
    omission_rows = clean_rows[:-1]
    relabel_rows = [dict(row) for row in clean_rows]
    relabel_rows[0]["layer_index"] = 1
    clean_family = state_family_binding_passes(clean_rows)
    mutant_omission = state_family_binding_passes(omission_rows)
    mutant_relabel = state_family_binding_passes(relabel_rows)

    # C3: position/current-length canonicality.
    position_control = packed_base.fork()
    expected_length = position_control.document_length
    clean_position = position_control.current_length == expected_length
    position_control.current_length += 1
    mutant_position = position_control.current_length == expected_length

    # C5: the official reference source may not import candidate/A4 code.
    reference_path = Path(__file__).with_name("run_r39_falcon_reference.py")
    clean_reference_source = reference_path.read_text(encoding="utf-8")
    clean_reference = reference_independence_passes(clean_reference_source)
    future_marker = "from __future__ import annotations\n"
    require(
        clean_reference_source.count(future_marker) == 1,
        "reference future-import marker drift",
    )
    mutant_reference_source = clean_reference_source.replace(
        future_marker,
        future_marker + "from falcon_h1_adapter import TorchSplitFalconH1\n",
        1,
    )
    try:
        compile(mutant_reference_source, "<reference-import-mutant>", "exec")
    except SyntaxError as error:
        raise RuntimeError("reference import mutant is not syntactically valid") from error
    mutant_reference = reference_independence_passes(mutant_reference_source)

    def row(
        control_id: str,
        predicate: str,
        clean: bool,
        mutant: bool,
        **extra: Any,
    ) -> dict[str, Any]:
        detected = clean and not mutant
        return {
            "control_id": control_id,
            "expected_first_failing_predicate": predicate,
            "matched_clean": {predicate: clean},
            "mutant": {predicate: mutant},
            "classification": "detected_expected_predicate" if detected else "escaped_or_clean_failure",
            **extra,
        }

    return [
        row(
            "MUTABLE_CACHE_ALIAS",
            "PRIVATE_MUTABLE_STORAGE",
            clean_alias["passed"],
            mutant_alias["passed"],
            mutant_overlap_ranges=mutant_alias["comparisons"][0]["overlap_ranges"],
        ),
        row(
            "STATE_FAMILY_OMISSION",
            "STATE_FAMILY_COMPLETENESS",
            clean_family,
            mutant_omission,
            omitted_pair=[clean_rows[-1]["layer_index"], clean_rows[-1]["family"]],
        ),
        row(
            "POSITION_OFFSET_DRIFT",
            "POSITION_CANONICAL",
            clean_position,
            mutant_position,
        ),
        row(
            "STATE_FAMILY_RELABEL",
            "STATE_FAMILY_BINDING",
            clean_family,
            mutant_relabel,
            relabeled_from=[clean_rows[0]["layer_index"], clean_rows[0]["family"]],
            relabeled_to=[relabel_rows[0]["layer_index"], relabel_rows[0]["family"]],
        ),
        row(
            "REFERENCE_CANDIDATE_IMPORT",
            "REFERENCE_IMPLEMENTATION_INDEPENDENT",
            clean_reference,
            mutant_reference,
            reference_source_sha256=sha256_file(reference_path),
        ),
    ]


def run_prefix_mutation_detector(
    adapter: TorchSplitFalconH1,
    document: torch.Tensor,
    *,
    depth: int,
    salt: str,
) -> dict[str, Any]:
    fresh = adapter.write_lower_replay(document, depth).quantize(
        bits=16,
        attention_bits=16,
        linear_bits=16,
        group_size=64,
    )
    data = fresh.document_residual.data
    before_inventory = storage_inventory(data, salt=salt, role="prefix-detector/residual")
    before = tensor_receipt(data)["content_sha256"]
    words = data.view(torch.int16).reshape(-1)
    require(words.numel() > 0, "prefix detector residual is empty")
    words[0].bitwise_xor_(1)
    after = tensor_receipt(data)["content_sha256"]
    after_inventory = storage_inventory(data, salt=salt, role="prefix-detector/residual")
    identity_before = [
        (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
        for row in before_inventory["rows"]
    ]
    identity_after = [
        (row["storage_id_sha256"], row["view_start_bytes"], row["view_end_bytes"])
        for row in after_inventory["rows"]
    ]
    detected = before != after and identity_before == identity_after
    return {
        "detector_id": "PREFIX_CONTENT_MUTATION",
        "predicate": "PERSISTENT_PREFIX_IMMUTABLE",
        "before_content_sha256": before,
        "after_content_sha256": after,
        "storage_identity_stable": identity_before == identity_after,
        "detected": detected,
    }


def run_rank(args: argparse.Namespace) -> None:
    require(
        os.environ.get("USE_HUB_KERNELS", "").upper() == "NO",
        "USE_HUB_KERNELS=NO absent before Transformers import",
    )
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
    require(
        model_authority["model_acquisition"]["policy"] == static["model_acquisition"],
        "model acquisition authority drift",
    )
    require(
        model_authority["model_acquisition"]["canonical_huggingface_revision"]
        == MODEL_REVISION,
        "canonical Hugging Face revision drift",
    )
    require(
        model_authority["model_acquisition"]["modelscope_revision"]
        == MODELSCOPE_REVISION,
        "ModelScope source revision drift",
    )
    require(args.rank in range(WORLD_SIZE), "rank outside registered range")
    gpu_rows = gpu_assignment["rows"]
    require(len(gpu_rows) == WORLD_SIZE, "GPU assignment rank count drift")
    expected_gpu_row = gpu_rows[args.rank]
    require(expected_gpu_row["rank"] == args.rank, "GPU assignment ordering drift")
    require(expected_gpu_row["uuid"] == args.expected_gpu_uuid, "expected GPU UUID drift")

    torch.cuda.set_device(0)
    hardware = gpu_identity(args.expected_gpu_uuid)
    torch.use_deterministic_algorithms(True)
    import transformers
    from transformers import AutoModelForCausalLM

    require(transformers.__version__ == "5.14.1", "Transformers version drift")
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_root),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
        trust_remote_code=False,
    )
    model.eval().cuda()
    adapter = TorchSplitFalconH1(model)
    geometry = adapter.geometry_receipt()
    require(geometry["matches_registered"], "Falcon-H1 model geometry drift")

    config = static["formal_config"]
    depth = int(config["split_depth"])
    steps = int(config["semantic_steps"])
    require(depth == EXPECTED_DEPTH == 18, "registered split depth drift")
    require(steps == 2, "semantic step count drift")
    require(
        registered_geometry_passes(geometry["layer_types"])
        and tuple(geometry["layer_types"]) == EXPECTED_LAYER_TYPES,
        "registered hybrid layer route drift",
    )
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
        exact_base = adapter.write_lower_replay(document, depth)
        packed_base = exact_base.quantize(
            bits=config["q16"]["residual_bits"],
            attention_bits=config["q16"]["attention_bits"],
            linear_bits=config["q16"]["linear_bits"],
            group_size=config["q16"]["group_size"],
        )
        base_before = state_content_receipt(packed_base)
        base_family_receipt = cache_family_receipt(
            exact_base.cache,
            expected_active_layers=tuple(range(depth)),
            expected_sequence_length=64,
        )
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
            queries[0],
            depth=depth,
            salt=salt,
        )
        prefix_mutation_detector = run_prefix_mutation_detector(
            adapter,
            document,
            depth=depth,
            salt=salt,
        )
        allocator.append(allocator_snapshot("rank_complete"))
        sidecar_receipt = sidecar.write(args.sidecar)

    shard = {
        "schema_version": "r39-falcon-h1-candidate-shard-v1",
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
            "official_source_sha256": adapter.official_source_sha256,
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
        "persistent_base": {
            "representation": "lossless_q16",
            "before": base_before,
            "after": base_after,
            "content_immutable": (
                base_before["state_content_sha256"] == base_after["state_content_sha256"]
            ),
            "mutable_cache_inventory": persistent_cache_inventory,
            "immutable_boundary_inventory": persistent_residual_inventory,
            "exact_lower_family_receipt_before_packing": base_family_receipt,
        },
        "cells": cells,
        "controls": controls,
        "prefix_mutation_detector": prefix_mutation_detector,
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
