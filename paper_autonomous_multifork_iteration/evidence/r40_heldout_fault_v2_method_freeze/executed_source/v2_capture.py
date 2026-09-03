"""Generic per-call capture wrapper using independent live-state rereads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from v2_common import (
    require,
    require_sha256,
    safe_relative_path,
    seal_payload,
    sha256_bytes,
    write_new_bytes,
    write_new_json,
)
from v2_predicates import ATOMIC_RECEIPT_SCHEMA, validate_live_snapshot


def capture_atomic_call(
    *,
    call_key: Mapping[str, Any],
    input_token_count: int,
    policy_sha256: str,
    live_state_reader: Callable[[], Mapping[str, Any]],
    model_call: Callable[[], Mapping[str, Any]],
    output_root: Path,
    logit_relative_path: str,
    receipt_relative_path: str,
    vocab_size: int,
) -> dict[str, Any]:
    """Execute one call; state returned by ``model_call`` is intentionally ignored."""

    require(type(input_token_count) is int and input_token_count > 0, "input token count")
    require(type(vocab_size) is int and vocab_size > 0, "vocab size")
    require_sha256(policy_sha256, "policy SHA")
    require(set(call_key.keys()) == {"call_index", "round_index", "request_id"}, "call key fields")
    require(type(call_key.get("call_index")) is int and call_key["call_index"] >= 0, "call index")
    require(type(call_key.get("round_index")) is int and call_key["round_index"] >= 0, "round index")
    request_id = call_key.get("request_id")
    require(isinstance(request_id, str) and request_id != "", "request id")
    pre = dict(live_state_reader())
    validate_live_snapshot(pre, request_id, "capture pre")
    result = model_call()
    require(isinstance(result, Mapping), "model call result")
    post = dict(live_state_reader())
    validate_live_snapshot(post, request_id, "capture post")
    require(pre["observation_id"] != post["observation_id"], "independent pre/post observation IDs")

    logits = result.get("logits_fp32_le")
    token_id = result.get("token_id")
    require(isinstance(logits, bytes) and len(logits) == vocab_size * 4, "complete FP32 logits")
    require(type(token_id) is int, "surfaced token")
    logit_relative = safe_relative_path(logit_relative_path, "logit relative path")
    receipt_relative = safe_relative_path(receipt_relative_path, "receipt relative path")
    logit_path = output_root / logit_relative
    write_new_bytes(logit_path, logits)
    logit_row = {
        "path": logit_relative_path,
        "sha256": sha256_bytes(logits),
        "nbytes": len(logits),
        "shape": [1, vocab_size],
        "dtype": "float32-little-endian",
    }
    receipt = seal_payload({
        "schema_version": ATOMIC_RECEIPT_SCHEMA,
        "policy_sha256": policy_sha256,
        "call_key": dict(call_key),
        "input_token_count": input_token_count,
        "surfaced_token_id": token_id,
        "logits": logit_row,
        "live_pre": pre,
        "live_post": post,
        "model_reported_state_used_by_gate": False,
    })
    write_new_json(output_root / receipt_relative, receipt)
    return receipt
