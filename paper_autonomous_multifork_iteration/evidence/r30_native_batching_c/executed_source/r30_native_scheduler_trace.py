from __future__ import annotations

"""Read-only scheduler observer for the R30 native-vLLM batching run.

The class subclasses vLLM's real V1 Scheduler and delegates every state
transition to the stock implementation.  It only appends pointer-free JSONL
receipts after/before the delegated calls.  It does not implement scheduling,
allocate blocks, mutate model state, or install the in-process ForkAudit cache
adapter.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler


TRACE_SCHEMA = "forkaudit-r30-native-vllm-scheduler-trace-v1"


def _status(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name if name is not None else value)


def _sha256_ints(values: list[int] | None) -> str | None:
    if values is None:
        return None
    raw = json.dumps(values, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "__dict__"):
        return {
            key: _jsonable(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return repr(value)


def _emit(payload: dict[str, Any]) -> None:
    target = os.environ.get("R30_NATIVE_TRACE_PATH")
    if not target:
        raise RuntimeError("R30_NATIVE_TRACE_PATH is required for the formal scheduler")
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": TRACE_SCHEMA,
        "pid": os.getpid(),
        **payload,
    }
    data = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, data)
    finally:
        os.close(descriptor)


class TracingScheduler(Scheduler):
    """Stock vLLM scheduler plus append-only ownership/lifecycle receipts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._r30_event_index = 0
        _emit(
            {
                "kind": "scheduler_init",
                "event_index": self._next_index(),
                "scheduler_class": f"{type(self).__module__}.{type(self).__qualname__}",
                "max_num_seqs": int(self.max_num_running_reqs),
                "max_num_scheduled_tokens": int(self.max_num_scheduled_tokens),
                "async_scheduling": bool(getattr(self, "async_scheduling", False)),
                "defer_block_free": bool(getattr(self, "defer_block_free", False)),
            }
        )

    def _next_index(self) -> int:
        index = self._r30_event_index
        self._r30_event_index += 1
        return index

    def _blocks(self, request_id: str) -> list[list[int]]:
        try:
            groups = self.kv_cache_manager.get_block_ids(request_id)
        except Exception:
            return []
        return [[int(block_id) for block_id in group] for group in groups]

    def _request_state(self, request: Any) -> dict[str, Any]:
        prompt_ids = getattr(request, "prompt_token_ids", None)
        output_ids = list(getattr(request, "output_token_ids", ()) or ())
        if not output_ids:
            output_ids = list(getattr(request, "_output_token_ids", ()) or ())
        params = getattr(request, "sampling_params", None)
        return {
            "request_id": str(request.request_id),
            "status": _status(request.status),
            "prompt_tokens": len(prompt_ids) if prompt_ids is not None else None,
            "prompt_token_ids_sha256": _sha256_ints(prompt_ids),
            "output_tokens": len(output_ids),
            "max_tokens": int(getattr(params, "max_tokens", 0)) if params is not None else None,
            "num_computed_tokens": int(getattr(request, "num_computed_tokens", 0)),
            "num_in_flight_tokens": int(getattr(request, "num_in_flight_tokens", 0)),
            "block_ids_by_group": self._blocks(str(request.request_id)),
        }

    def _active_states(self) -> list[dict[str, Any]]:
        return [
            self._request_state(request)
            for _, request in sorted(self.requests.items(), key=lambda item: item[0])
        ]

    def add_request(self, request: Any) -> None:
        _emit(
            {
                "kind": "add_request_begin",
                "event_index": self._next_index(),
                "request": self._request_state(request),
            }
        )
        super().add_request(request)
        _emit(
            {
                "kind": "add_request_end",
                "event_index": self._next_index(),
                "request": self._request_state(request),
                "request_counts": list(self.get_request_counts()),
            }
        )

    def schedule(self, *args: Any, **kwargs: Any) -> Any:
        output = super().schedule(*args, **kwargs)
        new_requests = [
            {
                "request_id": str(row.req_id),
                "prompt_tokens": (
                    len(row.prompt_token_ids) if row.prompt_token_ids is not None else None
                ),
                "prompt_token_ids_sha256": _sha256_ints(row.prompt_token_ids),
                "num_computed_tokens_before": int(row.num_computed_tokens),
                "block_ids_by_group": [
                    [int(block_id) for block_id in group] for group in row.block_ids
                ],
            }
            for row in output.scheduled_new_reqs
        ]
        cached = output.scheduled_cached_reqs
        _emit(
            {
                "kind": "schedule",
                "event_index": self._next_index(),
                "scheduled_new_requests": new_requests,
                "scheduled_cached_request_ids": [str(value) for value in cached.req_ids],
                "scheduled_cached_new_block_ids": _jsonable(cached.new_block_ids),
                "num_scheduled_tokens": {
                    str(key): int(value)
                    for key, value in output.num_scheduled_tokens.items()
                },
                "total_num_scheduled_tokens": int(output.total_num_scheduled_tokens),
                "finished_request_ids": sorted(str(value) for value in output.finished_req_ids),
                "new_block_ids_to_zero": (
                    [int(value) for value in output.new_block_ids_to_zero]
                    if output.new_block_ids_to_zero is not None
                    else []
                ),
                "kv_cache_block_copies": _jsonable(output.kv_cache_block_copies),
                "request_counts": list(self.get_request_counts()),
                "active_requests": self._active_states(),
            }
        )
        return output

    def update_from_output(self, scheduler_output: Any, model_runner_output: Any) -> Any:
        _emit(
            {
                "kind": "update_begin",
                "event_index": self._next_index(),
                "scheduled_request_ids": [
                    str(value) for value in scheduler_output.num_scheduled_tokens
                ],
                "model_runner_request_ids": [
                    str(value) for value in getattr(model_runner_output, "req_ids", ())
                ],
                "sampled_token_ids": _jsonable(
                    getattr(model_runner_output, "sampled_token_ids", None)
                ),
            }
        )
        result = super().update_from_output(scheduler_output, model_runner_output)
        _emit(
            {
                "kind": "update_end",
                "event_index": self._next_index(),
                "active_requests": self._active_states(),
                "request_counts": list(self.get_request_counts()),
            }
        )
        return result

    def _free_request(self, request: Any, delay_free_blocks: bool = False) -> Any:
        _emit(
            {
                "kind": "free_request_begin",
                "event_index": self._next_index(),
                "delay_free_blocks": bool(delay_free_blocks),
                "request": self._request_state(request),
                "processed_step_seq": int(getattr(self, "processed_step_seq", 0)),
                "scheduler_step_seq": int(getattr(self, "sched_step_seq", 0)),
            }
        )
        result = super()._free_request(request, delay_free_blocks=delay_free_blocks)
        _emit(
            {
                "kind": "free_request_end",
                "event_index": self._next_index(),
                "request_id": str(request.request_id),
                "still_registered": str(request.request_id) in self.requests,
                "request_counts": list(self.get_request_counts()),
            }
        )
        return result


__all__ = ["TRACE_SCHEMA", "TracingScheduler"]
