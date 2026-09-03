from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import sys
from typing import Any, Iterable, Mapping, Optional


METHOD_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(METHOD_ROOT / "executed_source"))

from v2_common import seal_payload, sha256_bytes  # noqa: E402
from v2_predicates import (  # noqa: E402
    ALLOCATOR_SCHEMA,
    ATOMIC_RECEIPT_SCHEMA,
    LIVE_SOURCE_KIND,
    SEMANTIC_SCHEMA,
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def fp32_bytes(values: Iterable[float]) -> bytes:
    values = list(values)
    return struct.pack("<%df" % len(values), *values)


def write_logits(root: Path, name: str, values: Iterable[float]) -> Mapping[str, Any]:
    raw = fp32_bytes(values)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": name,
        "sha256": sha256_bytes(raw),
        "nbytes": len(raw),
        "shape": [1, len(raw) // 4],
        "dtype": "float32-little-endian",
    }


def semantic_arm(label: str, descriptors: list[Mapping[str, Any]], tokens: Optional[list[int]] = None) -> Mapping[str, Any]:
    if tokens is None:
        tokens = [index + 10 for index in range(len(descriptors))]
    calls = []
    for index, descriptor in enumerate(descriptors):
        calls.append({
            "call_key": {"call_index": index, "round_index": index, "request_id": "request-a"},
            "token_id": tokens[index],
            "logits": descriptor,
        })
    return {"schema_version": SEMANTIC_SCHEMA, "arm": label, "calls": calls}


def allocator_arm(label: str, current: Optional[list[int]] = None, peak: Optional[list[int]] = None) -> Mapping[str, Any]:
    phases = ["H0", "H1", "H4", "H6", "H7"]
    current = current or [100, 120, 140, 140, 100]
    peak = peak or [100, 120, 150, 150, 150]
    return {
        "schema_version": ALLOCATOR_SCHEMA,
        "arm": label,
        "peak_reset_before_h0": True,
        "endpoints": [
            {
                "phase": phase,
                "synchronized": True,
                "sync_event_id": label + "-sync-" + phase,
                "current_allocated_bytes": current[index],
                "peak_allocated_bytes": peak[index],
            }
            for index, phase in enumerate(phases)
        ],
    }


def live_snapshot(request_id: str, length: int, version: int, epoch: int, observation: str,
                  kv_label: Optional[str] = None, gdn_label: Optional[str] = None) -> Mapping[str, Any]:
    return {
        "request_id": request_id,
        "kv_logical_length": length,
        "kv_content_sha256": digest(kv_label or (request_id + "-kv-" + str(version))),
        "gdn_content_sha256": digest(gdn_label or (request_id + "-gdn-" + str(version))),
        "kv_version": version,
        "gdn_version": version,
        "kv_commit_epoch": epoch,
        "gdn_commit_epoch": epoch,
        "observation_id": observation,
        "source_kind": LIVE_SOURCE_KIND,
        "synchronized": True,
    }


def atomic_receipt(call_index: int, round_index: int, request_id: str, pre: Mapping[str, Any],
                   post: Mapping[str, Any], policy_sha256: str, input_token_count: int = 1) -> Mapping[str, Any]:
    return seal_payload({
        "schema_version": ATOMIC_RECEIPT_SCHEMA,
        "policy_sha256": policy_sha256,
        "call_key": {
            "call_index": call_index,
            "round_index": round_index,
            "request_id": request_id,
        },
        "input_token_count": input_token_count,
        "surfaced_token_id": 17,
        "logits": {
            "path": "logits/%03d.bin" % call_index,
            "sha256": digest("logits-%d" % call_index),
            "nbytes": 16,
            "shape": [1, 4],
            "dtype": "float32-little-endian",
        },
        "live_pre": dict(pre),
        "live_post": dict(post),
        "model_reported_state_used_by_gate": False,
    })
