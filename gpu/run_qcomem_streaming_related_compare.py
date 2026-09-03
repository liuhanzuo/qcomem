#!/usr/bin/env python3
"""Common-client CoMem/serving-baseline comparison.

This module adds only the previously missing CoMem side of the already frozen
OpenAI-completions streaming protocol.  It deliberately imports the exact
``stream_completion`` and workload builder used by the verified vLLM and
SGLang runs.  The HTTP wrapper is part of the measured client-wall interval.

The server accepts token-id prompts.  Each formal rank owns one frozen
LongBench workload and one H20.  A document-only request primes one persistent
state; the subsequent document+query request either reuses that state
(``cache_on``) or builds it inside the measured request (``cache_off``).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterable
from urllib import request as urllib_request


EXPECTED_CONFIGS = (
    "full-prefix-q16",
    "qcomem-d7-r16-a16-l16",
    "qcomem-d7-r8-a8-l8",
    "qcomem-d7-r4-a4-l8",
    "qcomem-d7-mixed",
    "qcomem-d7-r4-a4-l4",
)
CACHED_CONFIGS = EXPECTED_CONFIGS
EXPECTED_PAIRS = tuple(
    (dataset, source_index)
    for dataset in ("qasper", "2wikimqa")
    for source_index in range(6, 10)
)
DATA_SHA256 = "1553739496b3c209988de56a4ccf574d407379b6b7937ddfafacbe66925069fe"
SOURCE_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"


class CompareError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompareError(message)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical_json_bytes(value))
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    require(not isinstance(value, bool) and isinstance(value, (int, float)), label)
    result = float(value)
    require(math.isfinite(result), label)
    if positive:
        require(result > 0.0, label)
    return result


def phases_for_config(config: str) -> tuple[str, ...]:
    require(config in EXPECTED_CONFIGS, f"unknown config {config}")
    return ("cache_off", "cache_on")


@dataclass
class CachedState:
    config: str
    document_ids: tuple[int, ...]
    value: Any
    physical_nbytes: int


class QComemRuntime:
    """One-model, one-workload runtime behind the local HTTP endpoint."""

    def __init__(self, model_path: Path, data_path: Path, rank: int, seed: int) -> None:
        import torch
        import transformers
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        from qcomem_torch import TorchSplitCausalLM
        from run_related_work_serving_baseline import load_workload

        require(torch.cuda.is_available(), "CUDA is unavailable")
        require(0 <= rank < 8, "rank must be 0..7")
        require(sha256_file(data_path) == DATA_SHA256, "data SHA drift")
        torch.cuda.set_device(0)
        torch.manual_seed(seed + rank)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        eos = self.tokenizer.eos_token_id
        self.eos_ids = (
            {int(eos)} if isinstance(eos, int) else {int(value) for value in (eos or [])}
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_path, dtype=torch.bfloat16, local_files_only=True
        )
        if hasattr(model.model, "visual"):
            model.model.visual = None
        model.eval().cuda()
        torch.cuda.synchronize()
        self.model = model
        self.adapter = TorchSplitCausalLM(model)
        self.workload = load_workload(model_path, data_path, rank)
        require(
            (self.workload["dataset"], self.workload["source_index"])
            == EXPECTED_PAIRS[rank],
            "workload assignment drift",
        )
        self.document_ids = tuple(int(value) for value in self.workload["document_token_ids"])
        self.query_ids = tuple(int(value) for value in self.workload["query_token_ids"])
        self.full_ids = self.document_ids + self.query_ids
        self.cached: CachedState | None = None
        self.metrics = {"queries": 0, "hits": 0}
        self.last_receipt: dict[str, Any] = {
            "config": None,
            "cache_hit": False,
            "persistent_state_nbytes": 0,
        }
        self.lock = threading.Lock()
        self.environment = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(0),
            "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "gpu_uuid": os.environ.get("NVIDIA_VISIBLE_DEVICES"),
        }
        require(self.environment["gpu_name"] == "NVIDIA H20-3e", "GPU name drift")
        require(self.environment["compute_capability"] == "9.0", "GPU capability drift")

    def reset(self) -> None:
        with self.lock:
            self.cached = None
            self.last_receipt = {
                "config": None,
                "cache_hit": False,
                "persistent_state_nbytes": 0,
            }
            gc.collect()
            self.torch.cuda.empty_cache()

    def inference_context(self):
        return self.torch.inference_mode()

    def _config(self, name: str):
        from qcomem_deployment import parse_deployment_config

        require(name in EXPECTED_CONFIGS, f"config outside frozen set: {name}")
        return parse_deployment_config(name)

    def _persistent_nbytes(self, state: Any | None) -> int:
        from qcomem_deployment import persistent_components

        return int(persistent_components(state)["persistent_total_resident_nbytes"])

    def _build_state(self, config_name: str) -> CachedState | None:
        from qcomem_deployment import build_persistent_state

        config = self._config(config_name)
        if config.mode == "dense_recompute":
            return None
        document = self.torch.tensor(
            [self.document_ids], dtype=self.torch.long, device="cuda"
        )
        value = build_persistent_state(
            self.adapter,
            config,
            document,
            group_size=64,
            fork_strategy="deep-clone",
        )
        return CachedState(
            config=config_name,
            document_ids=self.document_ids,
            value=value,
            physical_nbytes=self._persistent_nbytes(value),
        )

    def _full_prefix_steps(self, prompt_ids: tuple[int, ...], max_tokens: int):
        torch = self.torch
        current = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        logits = self.adapter.full_last_logits(current)
        for step in range(max_tokens):
            token = int(logits.argmax(-1).item())
            yield token
            if token in self.eos_ids:
                break
            if step + 1 < max_tokens:
                current = torch.cat(
                    (current, torch.tensor([[token]], dtype=torch.long, device="cuda")),
                    dim=1,
                )
                logits = self.adapter.full_last_logits(current)

    def _cached_steps(self, config_name: str, state: Any, max_tokens: int):
        torch = self.torch
        config = self._config(config_name)
        query = torch.tensor([self.query_ids], dtype=torch.long, device="cuda")
        if config.mode == "full_prefix":
            local = state.fork()
            logits = self.adapter.continue_full_prefix(local, query)

            def advance(token: int):
                token_tensor = torch.tensor([[token]], dtype=torch.long, device="cuda")
                return self.adapter.continue_full_prefix(local, token_tensor)

        elif config.mode == "qcomem":
            local = state.fork()
            query_residual = self.adapter.continue_lower_replay(local, query)
            suffix_cache = self.adapter.make_cache()
            self.adapter.run_suffix_cached_last_logits(
                [local.document_residual], int(config.depth), suffix_cache, position_offset=0
            )
            logits = self.adapter.run_suffix_cached_last_logits(
                [query_residual],
                int(config.depth),
                suffix_cache,
                position_offset=local.document_length,
            )
            suffix_length = int(local.current_length)
            local.document_residual = None

            def advance(token: int):
                nonlocal suffix_length
                token_tensor = torch.tensor([[token]], dtype=torch.long, device="cuda")
                residual = self.adapter.continue_lower_replay(local, token_tensor)
                output = self.adapter.run_suffix_cached_last_logits(
                    [residual],
                    int(config.depth),
                    suffix_cache,
                    position_offset=suffix_length,
                )
                suffix_length += 1
                return output

        else:
            raise CompareError("cached generation requires full-prefix or CoMem")

        for step in range(max_tokens):
            token = int(logits.argmax(-1).item())
            yield token
            if token in self.eos_ids:
                break
            if step + 1 < max_tokens:
                logits = advance(token)

    def prime(self, config_name: str, prompt_ids: tuple[int, ...]) -> dict[str, Any]:
        require(prompt_ids == self.document_ids, "prime prompt is not frozen document")
        require(config_name in CACHED_CONFIGS, "dense mode cannot be primed")
        with self.lock:
            self.cached = self._build_state(config_name)
            require(self.cached is not None, "cached state build returned none")
            self.metrics["queries"] += 1
            self.last_receipt = {
                "config": config_name,
                "cache_hit": False,
                "persistent_state_nbytes": self.cached.physical_nbytes,
                "request_role": "prime",
            }
        # Priming is an explicit control-plane operation outside the measured
        # OpenAI request.  Returning a receipt avoids pretending that the
        # document-only cache construction is a generated completion.
        return self.state_receipt()

    def complete(
        self, config_name: str, prompt_ids: tuple[int, ...], max_tokens: int
    ) -> Iterable[int]:
        require(1 <= max_tokens <= 32, "max_tokens outside frozen range")
        if prompt_ids not in (self.full_ids, self.query_ids[:8]):
            raise CompareError("prompt does not match warmup or frozen workload")
        if prompt_ids == self.query_ids[:8]:
            return self._full_prefix_steps(prompt_ids, max_tokens)
        self.metrics["queries"] += 1
        hit = bool(
            self.cached is not None
            and self.cached.config == config_name
            and self.cached.document_ids == self.document_ids
        )
        state = self.cached if hit else self._build_state(config_name)
        require(state is not None, "persistent state missing")
        if hit:
            self.metrics["hits"] += 1
        self.last_receipt = {
            "config": config_name,
            "cache_hit": hit,
            "persistent_state_nbytes": state.physical_nbytes,
            "request_role": "measured",
        }
        return self._cached_steps(config_name, state.value, max_tokens)

    def state_receipt(self) -> dict[str, Any]:
        return {
            **self.last_receipt,
            "queries": int(self.metrics["queries"]),
            "hits": int(self.metrics["hits"]),
            "workload_id": self.workload["workload_id"],
            "environment": self.environment,
        }


def _sse_event(text: str, *, finish_reason: str | None, usage: dict[str, int]) -> bytes:
    value = {
        "choices": [{"text": text, "finish_reason": finish_reason}],
        "usage": usage,
    }
    return b"data: " + json.dumps(value, separators=(",", ":")).encode() + b"\n\n"


def make_handler(runtime: QComemRuntime):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            sys.stderr.write((fmt % args) + "\n")

        def _plain(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self._plain(200, b"ok\n", "text/plain")
            elif self.path == "/metrics":
                body = (
                    f"qcomem:prefix_cache_queries_total {runtime.metrics['queries']}\n"
                    f"qcomem:prefix_cache_hits_total {runtime.metrics['hits']}\n"
                ).encode()
                self._plain(200, body, "text/plain")
            elif self.path == "/qcomem_state":
                self._plain(200, canonical_json_bytes(runtime.state_receipt()), "application/json")
            else:
                self._plain(404, b"not found\n", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path == "/reset":
                    runtime.reset()
                    self._plain(200, b"reset\n", "text/plain")
                    return
                if self.path == "/prime":
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length))
                    config = str(payload.get("qcomem_config"))
                    require(config in CACHED_CONFIGS, "qcomem_config missing or invalid")
                    prompt = payload.get("prompt")
                    require(isinstance(prompt, list) and prompt, "token-id prompt required")
                    with getattr(runtime, "inference_context", contextlib.nullcontext)():
                        receipt = runtime.prime(
                            config, tuple(int(value) for value in prompt)
                        )
                    self._plain(200, canonical_json_bytes(receipt), "application/json")
                    return
                require(self.path == "/v1/completions", "unknown endpoint")
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                require(payload.get("stream") is True, "stream=true required")
                config = str(payload.get("qcomem_config"))
                require(config in EXPECTED_CONFIGS, "qcomem_config missing or invalid")
                prompt = payload.get("prompt")
                require(isinstance(prompt, list) and prompt, "token-id prompt required")
                prompt_ids = tuple(int(value) for value in prompt)
                max_tokens = int(payload.get("max_tokens", 0))
                with getattr(runtime, "inference_context", contextlib.nullcontext)():
                    iterator = runtime.complete(config, prompt_ids, max_tokens)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    generated: list[int] = []
                    decoded = ""
                    for token in iterator:
                        generated.append(int(token))
                        current = runtime.tokenizer.decode(
                            generated, skip_special_tokens=True
                        )
                        part = (
                            current[len(decoded) :]
                            if current.startswith(decoded)
                            else current
                        )
                        decoded = current
                        usage = {
                            "prompt_tokens": len(prompt_ids),
                            "completion_tokens": len(generated),
                            "total_tokens": len(prompt_ids) + len(generated),
                        }
                        self.wfile.write(
                            _sse_event(part, finish_reason=None, usage=usage)
                        )
                        self.wfile.flush()
                usage = {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": len(generated),
                    "total_tokens": len(prompt_ids) + len(generated),
                }
                self.wfile.write(
                    _sse_event("", finish_reason="length", usage=usage)
                    + b"data: [DONE]\n\n"
                )
                self.wfile.flush()
                self.close_connection = True
            except Exception as error:  # fail visibly to the common client
                traceback.print_exc(file=sys.stderr)
                body = canonical_json_bytes({"error": type(error).__name__, "detail": str(error)})
                if not self.wfile.closed:
                    try:
                        self._plain(400, body, "application/json")
                    except Exception:
                        pass

    return Handler


def server_stage(args: argparse.Namespace) -> None:
    require(args.model is not None and args.data is not None, "server inputs missing")
    runtime = QComemRuntime(args.model, args.data, args.rank, args.seed)
    server = HTTPServer((args.host, args.port), make_handler(runtime))
    print(f"QCOMEM_STREAM_SERVER_READY rank={args.rank} port={args.port}", flush=True)
    server.serve_forever()


def _post_text(url: str, body: bytes = b"") -> str:
    request = urllib_request.Request(url, data=body, method="POST")
    with urllib_request.urlopen(request, timeout=30.0) as response:
        return response.read().decode()


def _post_json(url: str, value: dict[str, Any]) -> dict[str, Any]:
    request = urllib_request.Request(
        url,
        data=canonical_json_bytes(value),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib_request.urlopen(request, timeout=1800.0) as response:
        payload = json.loads(response.read())
    require(isinstance(payload, dict), "JSON POST endpoint did not return object")
    return payload


def _get_json(url: str) -> dict[str, Any]:
    with urllib_request.urlopen(url, timeout=30.0) as response:
        value = json.loads(response.read())
    require(isinstance(value, dict), "JSON endpoint did not return object")
    return value


def client_stage(args: argparse.Namespace) -> None:
    from run_related_work_serving_baseline import (
        answer_f1,
        counter_delta,
        http_get_text,
        load_workload,
        parse_prometheus,
        prefix_counter_snapshot,
        stream_completion,
    )

    require(args.model is not None and args.data is not None, "client inputs missing")
    require(0 <= args.rank < args.world_size == 8, "formal rank/world-size drift")
    require(args.config in EXPECTED_CONFIGS, "config drift")
    require(args.phase in phases_for_config(args.config), "phase invalid for config")
    workload = load_workload(args.model, args.data, args.rank)
    base = args.base_url.rstrip("/")
    require(http_get_text(base + "/health").strip() == "ok", "server health drift")
    _post_text(base + "/reset")
    common = {
        "model": "qwen35-comem-common-client",
        "temperature": 0.0,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "qcomem_config": args.config,
    }
    warmup = stream_completion(
        base + "/v1/completions",
        {**common, "prompt": workload["query_token_ids"][:8], "max_tokens": 1},
        timeout=args.timeout,
    )
    before = prefix_counter_snapshot(parse_prometheus(http_get_text(base + "/metrics")))
    prime = None
    if args.phase == "cache_on":
        prime = _post_json(
            base + "/prime",
            {
                "qcomem_config": args.config,
                "prompt": workload["document_token_ids"],
            },
        )
    after_prime = prefix_counter_snapshot(
        parse_prometheus(http_get_text(base + "/metrics"))
    )
    measured = stream_completion(
        base + "/v1/completions",
        {
            **common,
            "prompt": workload["document_token_ids"] + workload["query_token_ids"],
            "max_tokens": args.max_new_tokens,
        },
        timeout=args.timeout,
    )
    after = prefix_counter_snapshot(parse_prometheus(http_get_text(base + "/metrics")))
    state = _get_json(base + "/qcomem_state")
    references = list(workload["references"])
    measured["f1"] = max(answer_f1(measured["prediction"], value) for value in references)
    payload = {
        "schema": "forkaudit-comem-common-streaming-shard-v1",
        "status": "completed",
        "system": "comem-transformers-stream-wrapper-v1",
        "config": args.config,
        "phase": args.phase,
        "rank": args.rank,
        "world_size": args.world_size,
        "workload": {
            "workload_id": workload["workload_id"],
            "dataset": workload["dataset"],
            "source_index": workload["source_index"],
            "document_tokens": len(workload["document_token_ids"]),
            "query_tokens": len(workload["query_token_ids"]),
            "references": references,
        },
        "protocol": {
            "data_sha256": DATA_SHA256,
            "source_revision": SOURCE_REVISION,
            "max_input_tokens": 4096,
            "max_new_tokens": args.max_new_tokens,
            "greedy": True,
            "timing_boundary": "openai-completions-stream-client-wall-clock",
            "wrapper_included_in_timing": True,
        },
        "warmup": warmup,
        "prime": prime,
        "measured": measured,
        "prefix_counters": {
            "before": before,
            "after_prime": after_prime,
            "after": after,
            "measured_delta": counter_delta(after_prime, after),
        },
        "state_receipt": state,
    }
    atomic_json(args.output, payload)


def _load_shard(path: Path, config: str, phase: str, rank: int) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(value.get("schema") == "forkaudit-comem-common-streaming-shard-v1", "schema")
    require(value.get("status") == "completed", "incomplete shard")
    require(value.get("config") == config and value.get("phase") == phase, "cell drift")
    require(value.get("rank") == rank and value.get("world_size") == 8, "rank drift")
    workload = value.get("workload", {})
    require(
        (workload.get("dataset"), workload.get("source_index")) == EXPECTED_PAIRS[rank],
        "workload drift",
    )
    measured = value.get("measured", {})
    _number(measured.get("f1"), "f1")
    _number(measured.get("ttft_seconds"), "ttft", positive=True)
    _number(measured.get("e2e_seconds"), "e2e", positive=True)
    _number(measured.get("generated_tokens_per_second"), "throughput", positive=True)
    state = value.get("state_receipt", {})
    require(state.get("config") == config, "state config drift")
    require(
        isinstance(state.get("persistent_state_nbytes"), int)
        and state["persistent_state_nbytes"] >= 0,
        "state byte receipt invalid",
    )
    return value


def _summary_rows(shards: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [row["measured"] for row in shards]
    state = [row["state_receipt"] for row in shards]
    tpots = [row["median_tpot_seconds"] for row in measured if row["median_tpot_seconds"]]
    return {
        "mean_f1": statistics.fmean(float(row["f1"]) for row in measured),
        "median_ttft_seconds": statistics.median(float(row["ttft_seconds"]) for row in measured),
        "median_tpot_seconds": statistics.median(float(value) for value in tpots),
        "median_generated_tokens_per_second": statistics.median(
            float(row["generated_tokens_per_second"]) for row in measured
        ),
        "median_persistent_state_mib": statistics.median(
            int(row["persistent_state_nbytes"]) for row in state
        )
        / 2**20,
        "predictions": [row["prediction"] for row in measured],
    }


def _import_serving_summary(path: Path, expected_system_prefix: str) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(value.get("schema") == "forkaudit-related-serving-summary-v1", "serving schema")
    require(value.get("scientific_run_valid") is True, "serving summary invalid")
    require(str(value.get("system", "")).startswith(expected_system_prefix), "system drift")
    require(value.get("comparison_boundary") == "same-model-same-slice-openai-streaming-serving-only", "boundary drift")
    return {"sha256": sha256_file(path), "path": str(path), "value": value}


def aggregate_stage(args: argparse.Namespace) -> None:
    require(args.input_dir is not None, "input directory missing")
    require(args.vllm_summary is not None and args.sglang_summary is not None, "baseline summaries missing")
    cells: dict[str, dict[str, Any]] = {}
    raw_files: dict[str, list[str]] = {}
    all_passed = True
    for config in EXPECTED_CONFIGS:
        phase_values: dict[str, Any] = {}
        loaded: dict[str, list[dict[str, Any]]] = {}
        for phase in phases_for_config(config):
            files = [f"{config}-{phase}-rank-{rank}.json" for rank in range(8)]
            shards = [
                _load_shard(args.input_dir / name, config, phase, rank)
                for rank, name in enumerate(files)
            ]
            loaded[phase] = shards
            raw_files[f"{config}:{phase}"] = files
            phase_values[phase] = _summary_rows(shards)
        prediction_exact = None
        hit_observed = None
        if config in CACHED_CONFIGS:
            prediction_exact = [
                loaded["cache_off"][rank]["measured"]["prediction"]
                == loaded["cache_on"][rank]["measured"]["prediction"]
                for rank in range(8)
            ]
            hit_observed = [
                loaded["cache_on"][rank]["state_receipt"].get("cache_hit") is True
                and loaded["cache_on"][rank]["prefix_counters"]["measured_delta"].get("hits") == 1.0
                for rank in range(8)
            ]
            all_passed = all_passed and all(prediction_exact) and all(hit_observed)
        cells[config] = {
            "phases": phase_values,
            "cache_off_vs_on_prediction_exact": prediction_exact,
            "cache_hit_observed": hit_observed,
        }

    vllm = _import_serving_summary(args.vllm_summary, "vllm-")
    sglang = _import_serving_summary(args.sglang_summary, "sglang-")
    leaderboard = []
    vanilla = vllm["value"]["phases"]["cache_off"]
    leaderboard.append(
        {
            "method": "Vanilla vLLM (prefix cache off)",
            "family": "control",
            "mean_f1": vanilla["mean_f1"],
            "median_ttft_seconds": vanilla["median_ttft_seconds"],
            "median_tpot_seconds": vanilla["median_tpot_seconds"],
            "median_generated_tokens_per_second": vanilla[
                "median_generated_tokens_per_second"
            ],
            "median_persistent_state_mib": None,
        }
    )
    for config in CACHED_CONFIGS:
        row = cells[config]["phases"]["cache_on"]
        leaderboard.append({"method": config, "family": "CoMem" if config.startswith("qcomem") else "full-prefix control", **row})
    for name, imported in (("vLLM prefix cache", vllm), ("SGLang RadixAttention", sglang)):
        row = imported["value"]["phases"]["cache_on"]
        leaderboard.append(
            {
                "method": name,
                "family": "related work",
                "mean_f1": row["mean_f1"],
                "median_ttft_seconds": row["median_ttft_seconds"],
                "median_tpot_seconds": row["median_tpot_seconds"],
                "median_generated_tokens_per_second": row["median_generated_tokens_per_second"],
                "median_persistent_state_mib": None,
            }
        )
    payload = {
        "schema": "forkaudit-comem-related-common-streaming-summary-v1",
        "scientific_run_valid": True,
        "hypothesis_passed": all_passed,
        "scientific_outcome": "valid_positive" if all_passed else "valid_negative",
        "comparison_boundary": "same-model-same-slice-openai-streaming-client-wall",
        "timing_wrapper_included": True,
        "pairs": [list(value) for value in EXPECTED_PAIRS],
        "configs": cells,
        "imported_verified_summaries": {
            "vllm": {key: vllm[key] for key in ("path", "sha256")},
            "sglang": {key: sglang[key] for key in ("path", "sha256")},
        },
        "leaderboard": leaderboard,
        "raw_files": raw_files,
    }
    atomic_json(args.output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("server", "client", "aggregate"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18600)
    parser.add_argument("--base-url")
    parser.add_argument("--config", choices=EXPECTED_CONFIGS)
    parser.add_argument("--phase", choices=("cache_off", "cache_on"))
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--vllm-summary", type=Path)
    parser.add_argument("--sglang-summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.stage == "server":
        server_stage(args)
    elif args.stage == "client":
        require(args.base_url and args.config and args.phase and args.output, "client args missing")
        client_stage(args)
    else:
        require(args.output is not None, "aggregate output missing")
        aggregate_stage(args)


if __name__ == "__main__":
    main()
