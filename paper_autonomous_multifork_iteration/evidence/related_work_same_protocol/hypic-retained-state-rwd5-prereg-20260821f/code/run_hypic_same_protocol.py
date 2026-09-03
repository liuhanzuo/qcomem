#!/usr/bin/env python3
"""HYPIC same-model LongBench qualification and aggregation.

This client keeps the benchmark prompt token-identical to the existing
vLLM/SGLang rows while exposing the document boundary to HYPIC through its
control-plane separator.  The separator never enters the model token stream.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import subprocess
import statistics
from pathlib import Path
from typing import Any

from run_related_work_serving_baseline import (
    BaselineError,
    DATASET_PROMPTS,
    DATA_SHA256,
    EXPECTED_PAIRS,
    SOURCE_REVISION,
    _require,
    answer_f1,
    atomic_json,
    http_get_text,
    sha256_file,
    stream_completion,
)


SEP = "<<PIC_SEP>>"
HYPIC_COMMIT = "98147c01909004e66d98bcb18b886927d41b0ee5"
MODES = ("full_recompute", "prefix_cache", "transition_rope_recompute")
SEAM_TOKENS = 8


def token_sha256(token_ids: list[int]) -> str:
    payload = bytearray()
    for token_id in token_ids:
        _require(isinstance(token_id, int) and 0 <= token_id < 2**31, "token id")
        payload.extend(token_id.to_bytes(4, "little", signed=True))
    return hashlib.sha256(payload).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def split_and_tokenize_exact(text: str, tokenizer) -> tuple[list[int], list[list[int]]]:
    """Mirror HYPIC's official ``split_and_tokenize`` implementation."""
    ids: list[int] = []
    offsets: list[list[int]] = []
    for part in text.split(SEP):
        segment = [int(value) for value in tokenizer.encode(part, add_special_tokens=False)]
        if not segment:
            continue
        start = len(ids)
        ids.extend(segment)
        offsets.append([start, len(ids)])
    return ids, offsets


def longest_common_prefix(left: list[int], right: list[int]) -> int:
    size = 0
    for first, second in zip(left, right):
        if first != second:
            break
        size += 1
    return size


def segment_disjointness_receipt(
    warm_segments: dict[str, list[int]], formal_segments: dict[str, list[int]]
) -> dict[str, Any]:
    """Prove cacheable warm segments cannot seed a formal HYPIC segment hit."""
    matrix: list[dict[str, Any]] = []
    for warm_name, warm_ids in sorted(warm_segments.items()):
        _require(bool(warm_ids), f"empty warm segment: {warm_name}")
        warm_sha = token_sha256(warm_ids)
        for formal_name, formal_ids in sorted(formal_segments.items()):
            _require(bool(formal_ids), f"empty formal segment: {formal_name}")
            formal_sha = token_sha256(formal_ids)
            matrix.append(
                {
                    "warm_segment": warm_name,
                    "formal_segment": formal_name,
                    "warm_token_sha256": warm_sha,
                    "formal_token_sha256": formal_sha,
                    "equal": warm_ids == formal_ids,
                }
            )
    _require(matrix and not any(row["equal"] for row in matrix), "warm/formal segment collision")
    return {"passed": True, "comparison_count": len(matrix), "matrix": matrix}


def text_with_exact_token_count(
    tokenizer,
    count: int,
    salt: str,
    *,
    lead_candidates: tuple[str, ...] = (
        "Zebra",
        "Quartz",
        "Mango",
        "Violet",
        "Cobalt",
        "Jigsaw",
        "Nebula",
        "Xylophone",
    ),
    forbidden_first: int | None = None,
) -> tuple[str, list[int]]:
    """Create unrelated ordinary text with an exact, round-trippable token count."""
    _require(count > 0 and salt, "synthetic warmup contract")
    for lead in lead_candidates:
        material = ((f"{lead}-{salt} unrelated material. ") * (count + 8)).strip()
        encoded = [int(value) for value in tokenizer.encode(material, add_special_tokens=False)]
        if len(encoded) < count:
            continue
        expected = encoded[:count]
        if forbidden_first is not None and expected[0] == forbidden_first:
            continue
        text = tokenizer.decode(
            expected, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        observed = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
        if observed == expected:
            return text, observed
    raise BaselineError("could not construct prefix-disjoint exact-token warmup text")


def _server_configuration_receipt(
    server_info: dict[str, Any],
    mode: str,
    tp_size: int,
    *,
    model_path: Path,
    rank: int,
) -> dict[str, Any]:
    _require(isinstance(server_info, dict), "server info schema")
    expected = {
        "tp_size": tp_size,
        "dp_size": 1,
        "quantization": None,
        "speculative_algorithm": None,
        "completion_template": None,
        "enable_cache_report": True,
        "model_path": str(model_path),
        "tokenizer_path": str(model_path),
        "served_model_name": "qwen35-hypic",
        "dtype": "bfloat16",
        "context_length": 8192,
        "max_running_requests": 1,
        "max_total_tokens": 8192,
        "mem_fraction_static": 0.8,
        "random_seed": 20260821 + rank,
        "disable_overlap_schedule": True,
        "disable_cuda_graph": True,
        "enable_metrics": False,
        "sampling_backend": "pytorch",
        "attention_backend": "fa3",
        "kv_cache_dtype": "auto",
        "moe_runner_backend": "auto",
        "moe_a2a_backend": "none",
        "version": "0.5.14",
    }
    if mode == "full_recompute":
        expected.update(
            {
                "pic_enable": False,
                "disable_radix_cache": True,
                "mamba_radix_cache_strategy": "no_buffer",
                "linear_attn_prefill_backend": None,
                "linear_attn_decode_backend": None,
            }
        )
    elif mode == "prefix_cache":
        expected.update(
            {
                "pic_enable": False,
                "disable_radix_cache": False,
                "mamba_radix_cache_strategy": "extra_buffer",
                "linear_attn_prefill_backend": None,
                "linear_attn_decode_backend": None,
            }
        )
    else:
        expected.update(
            {
                "pic_enable": True,
                "pic_mode": "transition_rope_recompute",
                "pic_separator_str": SEP,
                "page_size": 1,
                "chunked_prefill_size": -1,
                "mamba_radix_cache_strategy": "no_buffer",
                "disable_overlap_schedule": True,
                "linear_attn_prefill_backend": "triton",
                "linear_attn_decode_backend": "triton",
            }
        )
    observed = {key: server_info.get(key) for key in expected}
    _require(observed == expected, f"resolved server configuration drift: {observed!r}")
    return {"expected": expected, "observed": observed}


def _server_process_receipt(
    pid: int, *, expected_pythonpath: str, expected_cuda: str
) -> dict[str, Any]:
    _require(pid > 1, "server PID")
    root = Path("/proc") / str(pid)
    _require(root.is_dir(), "server process is not live")
    cmdline = [
        part.decode("utf-8")
        for part in (root / "cmdline").read_bytes().split(b"\0")
        if part
    ]
    environment = {}
    for item in (root / "environ").read_bytes().split(b"\0"):
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        decoded_key = key.decode("utf-8")
        if decoded_key in {
            "CUDA_VISIBLE_DEVICES",
            "PIC_SEAM_SINK",
            "PYTHONPATH",
            "PYTHONDONTWRITEBYTECODE",
            "SGLANG_NUMA_BIND_V2",
            "SGLANG_IS_FLASHINFER_AVAILABLE",
        }:
            environment[decoded_key] = value.decode("utf-8")
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": expected_cuda,
        "PIC_SEAM_SINK": "8",
        "PYTHONPATH": expected_pythonpath,
        "PYTHONDONTWRITEBYTECODE": "1",
        "SGLANG_NUMA_BIND_V2": "0",
        "SGLANG_IS_FLASHINFER_AVAILABLE": "0",
    }
    _require(environment == expected_environment, f"server process environment drift: {environment!r}")
    _require("sglang.launch_server" in cmdline, "server command-line module drift")
    return {
        "schema": "hypic-live-server-process-v1",
        "pid": pid,
        "cmdline": cmdline,
        "cmdline_sha256": hashlib.sha256(
            b"\0".join(part.encode("utf-8") for part in cmdline)
        ).hexdigest(),
        "environment": environment,
    }


def _load_server_receipt(
    path: Path | None,
    *,
    args: argparse.Namespace,
    server_info: dict[str, Any],
) -> dict[str, Any] | None:
    if path is None:
        return None
    receipt = json.loads(path.read_text())
    _require(receipt.get("schema") == "hypic-server-launch-receipt-v1", "launch receipt schema")
    _require(receipt.get("official_commit") == HYPIC_COMMIT, "launch receipt commit")
    _require(receipt.get("git_worktree_clean") is True, "dirty HYPIC checkout")
    _require(receipt.get("git_untracked_absent") is True, "untracked HYPIC source")
    _require(receipt.get("mode") == args.mode, "launch receipt mode")
    _require(receipt.get("rank") == args.rank, "launch receipt rank")
    _require(receipt.get("tp_size") == args.expected_tp_size, "launch receipt TP")
    _require(receipt.get("model_path") == str(args.model), "launch receipt model")
    _require(receipt.get("data_sha256") == DATA_SHA256, "launch receipt data")
    _require(
        receipt.get("client_sha256") == sha256_file(Path(__file__)),
        "launch receipt client source",
    )
    _require(
        receipt.get("model_weight_ledger_raw_sha256")
        == "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014",
        "launch receipt model ledger",
    )
    _require(
        receipt.get("model_artifact_ledger_raw_sha256")
        == "d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd",
        "launch receipt model artifact ledger",
    )
    process = receipt.get("server_process")
    _require(
        isinstance(process, dict)
        and process.get("schema") == "hypic-live-server-process-v1"
        and isinstance(process.get("cmdline_sha256"), str)
        and len(process["cmdline_sha256"]) == 64,
        "launch receipt live server process",
    )
    _require(
        receipt.get("server_info_sha256") == canonical_json_sha256(server_info),
        "launch receipt server info",
    )
    hardware = receipt.get("hardware")
    _require(
        isinstance(hardware, dict)
        and hardware.get("gpu_name") == "NVIDIA H20-3e"
        and isinstance(hardware.get("gpu_uuid"), str)
        and hardware["gpu_uuid"].startswith("GPU-")
        and hardware.get("gpu_memory_mib") == 143771,
        "launch receipt hardware",
    )
    environment = receipt.get("environment")
    _require(isinstance(environment, dict), "launch receipt environment")
    if args.mode == "transition_rope_recompute":
        _require(environment.get("PIC_SEAM_SINK") == "8", "HYPIC seam environment")
    packages = receipt.get("packages")
    _require(
        isinstance(packages, dict)
        and packages.get("transformers_base") == "5.8.1"
        and packages.get("torch_base") == "2.11.0"
        and packages.get("torch_cuda") == "12.9"
        and packages.get("sglang-kernel_base") == "0.4.4"
        and packages.get("sgl-deep-gemm_base") == "0.1.3"
        and packages.get("sglang_base") == "0.5.14"
        and packages.get("flashinfer-python_base") == "0.5.3"
        and packages.get("flashinfer-cubin_base") == "0.5.3"
        and all(
            isinstance(packages.get(key), str) and len(packages[key]) == 64
            for key in (
                "transformers_record_sha256",
                "torch_record_sha256",
                "sglang-kernel_record_sha256",
                "sgl-deep-gemm_record_sha256",
                "sglang_record_sha256",
                "flashinfer-python_record_sha256",
                "flashinfer-cubin_record_sha256",
            )
        ),
        "launch receipt package closure",
    )
    return receipt


def _distribution_receipt(name: str) -> dict[str, str]:
    distribution = importlib.metadata.distribution(name)
    version = distribution.version
    base = version.split("+")[0]
    record_candidates = [
        entry for entry in (distribution.files or []) if str(entry).endswith(".dist-info/RECORD")
    ]
    _require(len(record_candidates) == 1, f"{name} RECORD closure")
    record = Path(distribution.locate_file(record_candidates[0]))
    _require(record.is_file(), f"{name} RECORD file")
    key = name.lower()
    return {
        key: version,
        f"{key}_base": base,
        f"{key}_record_sha256": sha256_file(record),
    }


def _nvidia_inventory() -> list[dict[str, Any]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        uuid, name, memory = (part.strip() for part in line.split(",", 2))
        rows.append({"gpu_uuid": uuid, "gpu_name": name, "gpu_memory_mib": int(memory)})
    _require(bool(rows), "empty NVIDIA inventory")
    return rows


def server_receipt_stage(args: argparse.Namespace) -> None:
    import torch

    _require(args.mode in MODES, "receipt mode")
    _require(args.hypic_repo is not None and args.hypic_repo.is_dir(), "HYPIC repo")
    head = subprocess.check_output(
        ["git", "-C", str(args.hypic_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    _require(head == HYPIC_COMMIT, "HYPIC commit drift")
    worktree_status = subprocess.check_output(
        [
            "git",
            "-C",
            str(args.hypic_repo),
            "status",
            "--porcelain",
            "--untracked-files=all",
        ],
        text=True,
    )
    _require(not worktree_status.strip(), "dirty or untracked HYPIC checkout")
    for path, label in (
        (args.source_ledger, "source ledger"),
        (args.environment_ledger, "environment ledger"),
        (args.preregistration, "preregistration"),
        (args.launch_command_file, "launch command"),
        (args.model_weight_ledger, "model weight ledger"),
        (args.model_artifact_ledger, "model artifact ledger"),
    ):
        _require(path is not None and path.is_file(), label)
    _require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one visible GPU")
    _require(torch.version.cuda == "12.9", "torch CUDA runtime drift")
    properties = torch.cuda.get_device_properties(0)
    inventory = _nvidia_inventory()
    matches = [row for row in inventory if row["gpu_uuid"] == args.expected_gpu_uuid]
    _require(len(matches) == 1, "expected GPU UUID absent")
    hardware = matches[0]
    _require(
        hardware["gpu_name"] == "NVIDIA H20-3e"
        and hardware["gpu_memory_mib"] == 143771
        and properties.name == "NVIDIA H20-3e"
        and properties.total_memory // (1024 * 1024) == 143166,
        "formal H20-3e hardware drift",
    )
    hardware["torch_visible_memory_mib"] = properties.total_memory // (1024 * 1024)
    server_info = json.loads(http_get_text(args.base_url.rstrip("/") + "/server_info", timeout=30.0))
    server_configuration = _server_configuration_receipt(
        server_info,
        args.mode,
        args.expected_tp_size,
        model_path=args.model,
        rank=args.rank,
    )
    server_process = _server_process_receipt(
        args.server_pid,
        expected_pythonpath=f"{args.hypic_repo}/python:{Path(__file__).resolve().parent}",
        expected_cuda=args.expected_gpu_uuid,
    )
    packages: dict[str, str] = {}
    for name in (
        "transformers",
        "torch",
        "sglang-kernel",
        "sgl-deep-gemm",
        "sglang",
        "flashinfer-python",
        "flashinfer-cubin",
    ):
        packages.update(_distribution_receipt(name))
    packages["torch_cuda"] = str(torch.version.cuda)
    _require(
        importlib.util.find_spec("flashinfer") is not None,
        "CUDA 12.9-compatible FlashInfer is unavailable",
    )
    payload = {
        "schema": "hypic-server-launch-receipt-v1",
        "official_commit": HYPIC_COMMIT,
        "git_worktree_clean": True,
        "git_untracked_absent": True,
        "mode": args.mode,
        "rank": args.rank,
        "tp_size": args.expected_tp_size,
        "model_path": str(args.model),
        "data_sha256": sha256_file(args.data),
        "client_sha256": sha256_file(Path(__file__)),
        "source_ledger_raw_sha256": sha256_file(args.source_ledger),
        "environment_ledger_raw_sha256": sha256_file(args.environment_ledger),
        "preregistration_sha256": sha256_file(args.preregistration),
        "launch_command_sha256": sha256_file(args.launch_command_file),
        "model_weight_ledger_raw_sha256": sha256_file(args.model_weight_ledger),
        "model_artifact_ledger_raw_sha256": sha256_file(args.model_artifact_ledger),
        "server_info_sha256": canonical_json_sha256(server_info),
        "server_configuration": server_configuration,
        "hardware": hardware,
        "server_process": server_process,
        "environment": server_process["environment"],
        "packages": packages,
    }
    atomic_json(args.output, payload)


def _render_with_marker(tokenizer, prompt_format: str, context: str, question: str) -> str:
    marker = "QCOMEM_CONTEXT_MARKER_8D31F4"
    user_text = prompt_format.format(context=marker, input=question)
    messages = [{"role": "user", "content": user_text}]
    try:
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    _require(rendered.count(marker) == 1, "chat-template marker drift")
    return rendered


def load_segmented_workload(model: Path, data: Path, rank: int) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    rows: list[dict[str, Any]] = []
    selected: dict[str, int] = {}
    with data.open() as handle:
        for line in handle:
            sample = json.loads(line)
            source_index = int(sample["_source_index"])
            if source_index < 6 or source_index > 9:
                continue
            dataset = str(sample["dataset"])
            if selected.get(dataset, 0) >= 4:
                continue
            selected[dataset] = selected.get(dataset, 0) + 1
            rows.append(sample)
    _require(len(rows) == 8, "frozen row count drift")
    _require(
        tuple((str(row["dataset"]), int(row["_source_index"])) for row in rows)
        == EXPECTED_PAIRS,
        "frozen row order drift",
    )
    sample = rows[rank]
    dataset = str(sample["dataset"])
    prompt_format = DATASET_PROMPTS[dataset]
    rendered = _render_with_marker(
        tokenizer, prompt_format, str(sample["context"]), str(sample["input"])
    )
    marker = "QCOMEM_CONTEXT_MARKER_8D31F4"
    prefix_text, suffix_text = rendered.split(marker)
    prefix_ids = [int(v) for v in tokenizer.encode(prefix_text, add_special_tokens=False)]
    suffix_ids = [int(v) for v in tokenizer.encode(suffix_text, add_special_tokens=False)]
    context_ids = [
        int(v)
        for v in tokenizer.encode(str(sample["context"]), add_special_tokens=False)
    ]
    available = 4096 - len(prefix_ids) - len(suffix_ids)
    _require(available >= 256, "insufficient context budget")
    if len(context_ids) > available:
        left = available // 2
        context_ids = context_ids[:left] + context_ids[-(available - left) :]
    context_text = tokenizer.decode(
        context_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    direct_ids = prefix_ids + context_ids + suffix_ids
    measured_text = SEP.join((prefix_text, context_text, suffix_text))
    measured_ids, offsets = split_and_tokenize_exact(measured_text, tokenizer)
    _require(measured_ids == direct_ids, "HYPIC segmented token stream drift")

    document_ids = prefix_ids + context_ids
    dummy_suffix_text, dummy_suffix_ids = text_with_exact_token_count(
        tokenizer,
        max(8, len(suffix_ids)),
        f"dummy-{rank}",
        lead_candidates=("Xylophone", "Violet", "Cobalt", "Nebula"),
        forbidden_first=suffix_ids[0],
    )
    _require(dummy_suffix_ids[0] != suffix_ids[0], "dummy query must diverge immediately")
    hypic_prime_text = SEP.join((prefix_text, context_text, dummy_suffix_text))
    hypic_prime_ids, hypic_prime_offsets = split_and_tokenize_exact(
        hypic_prime_text, tokenizer
    )
    _require(
        longest_common_prefix(hypic_prime_ids, direct_ids) == len(document_ids),
        "HYPIC prime leaked into the query",
    )
    _require(offsets[:2] == hypic_prime_offsets[:2], "prime segment boundary drift")

    warm_prefix_text, warm_prefix_ids = text_with_exact_token_count(
        tokenizer,
        len(prefix_ids),
        f"prefix-{rank}",
        lead_candidates=("Zebra", "Quartz", "Mango", "Violet"),
        forbidden_first=direct_ids[0],
    )
    warm_context_text, warm_context_ids = text_with_exact_token_count(
        tokenizer,
        len(context_ids),
        f"context-{rank}",
        lead_candidates=("Mango", "Cobalt", "Jigsaw", "Nebula"),
    )
    warm_query_text, warm_query_ids = text_with_exact_token_count(
        tokenizer,
        len(suffix_ids),
        f"query-{rank}",
        lead_candidates=("Quartz", "Jigsaw", "Cobalt", "Nebula"),
    )
    warm_dummy_text, warm_dummy_ids = text_with_exact_token_count(
        tokenizer,
        max(8, len(suffix_ids)),
        f"warm-dummy-{rank}",
        lead_candidates=("Violet", "Xylophone", "Zebra", "Mango"),
        forbidden_first=warm_query_ids[0],
    )
    _require(warm_dummy_ids[0] != warm_query_ids[0], "warm dummy must diverge")
    warm_document_ids = warm_prefix_ids + warm_context_ids
    warm_measured_ids = warm_document_ids + warm_query_ids
    _require(
        len(warm_measured_ids) == len(direct_ids)
        and longest_common_prefix(warm_measured_ids, direct_ids) == 0,
        "warmup must be length-matched and prefix-disjoint",
    )
    warm_measured_text = SEP.join(
        (warm_prefix_text, warm_context_text, warm_query_text)
    )
    observed_warm_ids, warm_offsets = split_and_tokenize_exact(
        warm_measured_text, tokenizer
    )
    _require(observed_warm_ids == warm_measured_ids, "warm segmented token drift")
    warm_prime_text = SEP.join(
        (warm_prefix_text, warm_context_text, warm_dummy_text)
    )
    warm_prime_ids, warm_prime_offsets = split_and_tokenize_exact(
        warm_prime_text, tokenizer
    )
    _require(
        longest_common_prefix(warm_prime_ids, warm_measured_ids)
        == len(warm_document_ids),
        "warm prime query leakage",
    )
    warm_segment_disjointness = segment_disjointness_receipt(
        {"prefix": warm_prefix_ids, "context": warm_context_ids},
        {"prefix": prefix_ids, "context": context_ids},
    )
    return {
        "workload_id": f"{dataset}-{sample['_source_index']}",
        "dataset": dataset,
        "source_index": int(sample["_source_index"]),
        "references": [str(value) for value in sample["answers"]],
        "measured_text": measured_text,
        "hypic_prime_text": hypic_prime_text,
        "hypic_prime_token_ids": hypic_prime_ids,
        "direct_token_ids": direct_ids,
        "document_token_ids": document_ids,
        "query_token_ids": suffix_ids,
        "segment_offsets": offsets,
        "prime_segment_offsets": hypic_prime_offsets,
        "warm_document_token_ids": warm_document_ids,
        "warm_measured_token_ids": warm_measured_ids,
        "warm_measured_text": warm_measured_text,
        "warm_prime_text": warm_prime_text,
        "warm_prime_token_ids": warm_prime_ids,
        "warm_segment_offsets": warm_offsets,
        "warm_prime_segment_offsets": warm_prime_offsets,
        "warm_segment_disjointness": warm_segment_disjointness,
        "prompt_token_sha256": token_sha256(direct_ids),
        "document_token_sha256": token_sha256(document_ids),
        "warm_prompt_token_sha256": token_sha256(warm_measured_ids),
    }


def request_prompts(workload: dict[str, Any], mode: str) -> dict[str, Any]:
    """Return mode-specific warmup and formal prompts without changing model tokens."""
    if mode == "transition_rope_recompute":
        return {
            "warm_prime": workload["warm_prime_text"],
            "warm_measured": workload["warm_measured_text"],
            "formal_prime": workload["hypic_prime_text"],
            "formal_measured": workload["measured_text"],
        }
    return {
        "warm_prime": workload["warm_document_token_ids"],
        "warm_measured": workload["warm_measured_token_ids"],
        "formal_prime": workload["document_token_ids"],
        "formal_measured": workload["direct_token_ids"],
    }


def cached_tokens_from_completion(completion: dict[str, Any]) -> int | None:
    usage = completion.get("usage")
    _require(isinstance(usage, dict), "completion usage schema")
    details = usage.get("prompt_tokens_details")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    _require(cached is None or (isinstance(cached, int) and not isinstance(cached, bool)), "cached token schema")
    return cached


def client_stage(args: argparse.Namespace) -> None:
    _require(args.mode in MODES, "invalid mode")
    _require(0 <= args.rank < 8 and args.world_size == 8, "rank contract")
    _require(sha256_file(args.data) == DATA_SHA256, "data digest drift")
    workload = load_segmented_workload(args.model, args.data, args.rank)
    base = args.base_url.rstrip("/")
    health = http_get_text(base + "/model_info", timeout=10.0)
    server_info = json.loads(http_get_text(base + "/server_info", timeout=30.0))
    server_configuration = _server_configuration_receipt(
        server_info,
        args.mode,
        args.expected_tp_size,
        model_path=args.model,
        rank=args.rank,
    )
    server_receipt = _load_server_receipt(
        args.server_receipt, args=args, server_info=server_info
    )
    prompts = request_prompts(workload, args.mode)
    common = {
        "model": args.served_model_name,
        "temperature": 0.0,
        "seed": args.seed,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    warm_prime = None
    if args.mode != "full_recompute":
        warm_prime = stream_completion(
            base + "/v1/completions",
            {**common, "prompt": prompts["warm_prime"], "max_tokens": 1},
            timeout=args.timeout,
            require_text=False,
        )
        _require(
            warm_prime["usage"].get("prompt_tokens")
            == len(
                workload["warm_prime_token_ids"]
                if args.mode == "transition_rope_recompute"
                else workload["warm_document_token_ids"]
            ),
            "warm prime token count drift",
        )
    warmup = stream_completion(
        base + "/v1/completions",
        {**common, "prompt": prompts["warm_measured"], "max_tokens": 1},
        timeout=args.timeout,
        require_text=False,
    )
    _require(
        warmup["usage"].get("prompt_tokens") == len(workload["warm_measured_token_ids"]),
        "warm measured token count drift",
    )
    warm_cached_tokens = cached_tokens_from_completion(warmup)
    if args.mode == "full_recompute":
        _require(warm_cached_tokens in (None, 0), "full warmup unexpectedly hit cache")
    elif args.mode == "prefix_cache":
        _require(
            isinstance(warm_cached_tokens, int)
            and 0 < warm_cached_tokens <= len(workload["warm_document_token_ids"]),
            "prefix warmup did not exercise document-only cache hit",
        )
    else:
        _require(
            warm_cached_tokens == len(workload["warm_document_token_ids"]) - SEAM_TOKENS,
            "HYPIC warmup seam-adjusted hit drift",
        )
    prime = None
    if args.mode != "full_recompute":
        prime = stream_completion(
            base + "/v1/completions",
            {**common, "prompt": prompts["formal_prime"], "max_tokens": 1},
            timeout=args.timeout,
            require_text=False,
        )
        _require(
            prime["usage"].get("prompt_tokens")
            == len(workload["hypic_prime_token_ids"] if args.mode == "transition_rope_recompute" else workload["document_token_ids"]),
            "formal prime token count drift",
        )
    measured = stream_completion(
        base + "/v1/completions",
        {
            **common,
            "prompt": prompts["formal_measured"],
            "max_tokens": args.max_new_tokens,
        },
        timeout=args.timeout,
    )
    measured["f1"] = max(
        answer_f1(measured["prediction"], reference)
        for reference in workload["references"]
    )
    _require(
        measured["usage"].get("prompt_tokens") == len(workload["direct_token_ids"]),
        "measured prompt token count drift",
    )
    cached_tokens = cached_tokens_from_completion(measured)
    expected_cached_tokens = 0
    if args.mode == "prefix_cache":
        _require(
            isinstance(cached_tokens, int)
            and 0 < cached_tokens <= len(workload["document_token_ids"]),
            "prefix cache hit crossed the document boundary",
        )
    elif args.mode == "transition_rope_recompute":
        expected_cached_tokens = len(workload["document_token_ids"]) - SEAM_TOKENS
        _require(cached_tokens == expected_cached_tokens, "HYPIC seam-adjusted hit drift")
    else:
        _require(cached_tokens in (None, 0), "full recompute unexpectedly hit cache")
    payload = {
        "schema": "forkaudit-hypic-same-protocol-shard-v1",
        "status": "completed",
        "formal_evidence_eligible": server_receipt is not None,
        "mode": args.mode,
        "rank": args.rank,
        "world_size": args.world_size,
        "official_commit": HYPIC_COMMIT,
        "served_model_name": args.served_model_name,
        "health_response": health,
        "server_info_sha256": canonical_json_sha256(server_info),
        "server_configuration": server_configuration,
        "server_launch_receipt": server_receipt,
        "workload": {
            "workload_id": workload["workload_id"],
            "dataset": workload["dataset"],
            "source_index": workload["source_index"],
            "references": workload["references"],
            "document_tokens": len(workload["document_token_ids"]),
            "query_tokens": len(workload["query_token_ids"]),
            "prompt_token_sha256": workload["prompt_token_sha256"],
            "document_token_sha256": workload["document_token_sha256"],
            "segment_offsets": workload["segment_offsets"],
            "token_identity_verified": True,
            "formal_prime_lcp_tokens": len(workload["document_token_ids"]),
            "warm_prompt_token_sha256": workload["warm_prompt_token_sha256"],
            "warm_segment_disjointness": workload["warm_segment_disjointness"],
        },
        "protocol": {
            "data_sha256": DATA_SHA256,
            "source_revision": SOURCE_REVISION,
            "max_input_tokens": 4096,
            "max_new_tokens": args.max_new_tokens,
            "greedy": True,
            "separator_is_control_plane_only": True,
            "seam_tokens": SEAM_TOKENS,
            "expected_tp_size": args.expected_tp_size,
            "timing_boundary": "openai-completions-stream-client-wall-clock",
        },
        "warm_prime": warm_prime,
        "warmup": warmup,
        "warm_cache_observation": {
            "cached_tokens": warm_cached_tokens,
            "document_tokens": len(workload["warm_document_token_ids"]),
            "hit_path_exercised": (
                warm_cached_tokens in (None, 0)
                if args.mode == "full_recompute"
                else isinstance(warm_cached_tokens, int) and warm_cached_tokens > 0
            ),
        },
        "prime": prime,
        "measured": measured,
        "cache_observation": {
            "cached_tokens": cached_tokens,
            "expected_cached_tokens": expected_cached_tokens,
            "cache_authority": "openai-completion-usage.cached_tokens",
            "prometheus_metrics_disabled": True,
        },
    }
    atomic_json(args.output, payload)


def aggregate_stage(args: argparse.Namespace) -> None:
    rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in MODES}
    for mode in MODES:
        for rank in range(8):
            value = json.loads((args.input_dir / f"{mode}-rank-{rank}.json").read_text())
            _require(value.get("schema") == "forkaudit-hypic-same-protocol-shard-v1", "schema")
            _require(value.get("status") == "completed", "status")
            _require(value.get("formal_evidence_eligible") is True, "formal evidence gate")
            launch_receipt = value.get("server_launch_receipt")
            _require(
                isinstance(launch_receipt, dict)
                and launch_receipt.get("schema") == "hypic-server-launch-receipt-v1"
                and launch_receipt.get("official_commit") == HYPIC_COMMIT
                and launch_receipt.get("mode") == mode
                and launch_receipt.get("rank") == rank
                and launch_receipt.get("tp_size") == 1
                and launch_receipt.get("data_sha256") == DATA_SHA256
                and launch_receipt.get("client_sha256") == sha256_file(Path(__file__))
                and launch_receipt.get("model_weight_ledger_raw_sha256")
                == "8314a82c9188b9b817193e039b0b0eb0636b328512f19b0c12455853b7e20014"
                and launch_receipt.get("model_artifact_ledger_raw_sha256")
                == "d78424684a27718b44616c80c788e7f8b726feed674fe015901b87de1993f7dd"
                and launch_receipt.get("server_info_sha256")
                == value.get("server_info_sha256"),
                "launch receipt binding",
            )
            process = launch_receipt.get("server_process")
            _require(
                isinstance(process, dict)
                and process.get("schema") == "hypic-live-server-process-v1"
                and isinstance(process.get("cmdline_sha256"), str)
                and len(process["cmdline_sha256"]) == 64,
                "live server process receipt",
            )
            for key in (
                "source_ledger_raw_sha256",
                "environment_ledger_raw_sha256",
                "preregistration_sha256",
                "launch_command_sha256",
            ):
                _require(
                    isinstance(launch_receipt.get(key), str)
                    and len(launch_receipt[key]) == 64,
                    f"launch receipt {key}",
                )
            hardware = launch_receipt.get("hardware")
            packages = launch_receipt.get("packages")
            _require(
                isinstance(hardware, dict)
                and hardware.get("gpu_name") == "NVIDIA H20-3e"
                and hardware.get("gpu_memory_mib") == 143771
                and isinstance(hardware.get("gpu_uuid"), str)
                and hardware["gpu_uuid"].startswith("GPU-"),
                "launch receipt hardware",
            )
            _require(
                isinstance(packages, dict)
                and packages.get("transformers_base") == "5.8.1"
                and packages.get("torch_base") == "2.11.0"
                and packages.get("torch_cuda") == "12.9"
                and packages.get("sglang-kernel_base") == "0.4.4"
                and packages.get("sgl-deep-gemm_base") == "0.1.3"
                and packages.get("sglang_base") == "0.5.14"
                and packages.get("flashinfer-python_base") == "0.5.3"
                and packages.get("flashinfer-cubin_base") == "0.5.3",
                "launch receipt environment",
            )
            _require(value.get("mode") == mode and value.get("rank") == rank, "cell")
            _require(value.get("official_commit") == HYPIC_COMMIT, "commit")
            workload = value.get("workload")
            _require(isinstance(workload, dict), "workload")
            _require(
                (workload.get("dataset"), workload.get("source_index"))
                == EXPECTED_PAIRS[rank],
                "pair",
            )
            _require(workload.get("token_identity_verified") is True, "token gate")
            disjointness = workload.get("warm_segment_disjointness")
            _require(
                isinstance(disjointness, dict)
                and disjointness.get("passed") is True
                and disjointness.get("comparison_count") == 4
                and len(disjointness.get("matrix", [])) == 4
                and all(row.get("equal") is False for row in disjointness["matrix"]),
                "warm/formal segment disjointness",
            )
            protocol = value.get("protocol")
            _require(isinstance(protocol, dict), "protocol")
            _require(protocol.get("data_sha256") == DATA_SHA256, "data binding")
            _require(protocol.get("source_revision") == SOURCE_REVISION, "revision")
            _require(protocol.get("max_input_tokens") == 4096, "input budget")
            _require(protocol.get("max_new_tokens") == 32, "output budget")
            _require(protocol.get("greedy") is True, "decoding contract")
            configuration = value.get("server_configuration")
            _require(isinstance(configuration, dict), "server configuration")
            expected_configuration = configuration.get("expected")
            observed_configuration = configuration.get("observed")
            _require(
                isinstance(expected_configuration, dict)
                and isinstance(observed_configuration, dict)
                and all(
                    observed_configuration.get(key) == expected
                    for key, expected in expected_configuration.items()
                ),
                "server configuration receipt",
            )
            measured = value.get("measured")
            _require(isinstance(measured, dict), "measured")
            usage = measured.get("usage")
            _require(isinstance(usage, dict), "measured usage")
            _require(
                usage.get("prompt_tokens")
                == workload.get("document_tokens") + workload.get("query_tokens"),
                "measured token count",
            )
            cache = value.get("cache_observation")
            _require(isinstance(cache, dict), "cache receipt")
            warm_cache = value.get("warm_cache_observation")
            _require(
                isinstance(warm_cache, dict)
                and warm_cache.get("hit_path_exercised") is True,
                "warm cache-path receipt",
            )
            cached = cache.get("cached_tokens")
            document_tokens = workload.get("document_tokens")
            _require(isinstance(document_tokens, int) and document_tokens > SEAM_TOKENS, "document tokens")
            if mode == "full_recompute":
                _require(cached in (None, 0), "full recompute cache hit")
                _require(
                    warm_cache.get("cached_tokens") in (None, 0),
                    "full warmup cache hit",
                )
            elif mode == "prefix_cache":
                _require(
                    isinstance(cached, int) and 0 < cached <= document_tokens,
                    "prefix cache boundary",
                )
                _require(
                    isinstance(warm_cache.get("cached_tokens"), int)
                    and 0 < warm_cache["cached_tokens"] <= warm_cache.get("document_tokens"),
                    "prefix warm cache boundary",
                )
            else:
                _require(
                    cached == document_tokens - SEAM_TOKENS
                    and cache.get("expected_cached_tokens") == cached,
                    "HYPIC seam receipt",
                )
                _require(
                    warm_cache.get("cached_tokens")
                    == warm_cache.get("document_tokens") - SEAM_TOKENS,
                    "HYPIC warm seam receipt",
                )
            rows[mode].append(value)
    for rank in range(8):
        authority = rows["full_recompute"][rank]
        authority_workload = authority["workload"]
        for mode in MODES[1:]:
            candidate = rows[mode][rank]
            candidate_workload = candidate["workload"]
            for key in (
                "workload_id",
                "dataset",
                "source_index",
                "references",
                "document_tokens",
                "query_tokens",
                "prompt_token_sha256",
                "document_token_sha256",
                "segment_offsets",
                "warm_prompt_token_sha256",
                "warm_segment_disjointness",
            ):
                _require(
                    candidate_workload.get(key) == authority_workload.get(key),
                    f"cross-mode workload drift: rank={rank} key={key}",
                )
            for key in (
                "data_sha256",
                "source_revision",
                "max_input_tokens",
                "max_new_tokens",
                "greedy",
                "expected_tp_size",
            ):
                _require(
                    candidate["protocol"].get(key) == authority["protocol"].get(key),
                    f"cross-mode protocol drift: rank={rank} key={key}",
                )
            authority_receipt = authority["server_launch_receipt"]
            candidate_receipt = candidate["server_launch_receipt"]
            for key in (
                "model_path",
                "data_sha256",
                "client_sha256",
                "source_ledger_raw_sha256",
                "environment_ledger_raw_sha256",
                "preregistration_sha256",
                "model_weight_ledger_raw_sha256",
                "model_artifact_ledger_raw_sha256",
                "packages",
                "hardware",
            ):
                _require(
                    candidate_receipt.get(key) == authority_receipt.get(key),
                    f"cross-mode launch drift: rank={rank} key={key}",
                )
    for mode in MODES:
        uuids = [row["server_launch_receipt"]["hardware"]["gpu_uuid"] for row in rows[mode]]
        _require(len(set(uuids)) == 8, f"{mode} did not use eight distinct GPUs")
    phase_summaries: dict[str, Any] = {}
    full_predictions = [row["measured"]["prediction"] for row in rows["full_recompute"]]
    for mode, shards in rows.items():
        measured = [row["measured"] for row in shards]
        tpots = [row["median_tpot_seconds"] for row in measured if row["median_tpot_seconds"] is not None]
        cached = [row["cache_observation"]["cached_tokens"] for row in shards]
        phase_summaries[mode] = {
            "mean_f1": statistics.mean(float(row["f1"]) for row in measured),
            "median_ttft_seconds": statistics.median(float(row["ttft_seconds"]) for row in measured),
            "median_of_per_request_mean_post_first_token_seconds": statistics.median(
                float(value) for value in tpots
            ),
            "post_first_token_denominator_rows": len(tpots),
            "median_generated_tokens_per_second": statistics.median(
                float(row["generated_tokens_per_second"]) for row in measured
            ),
            "rate_definition": "completion_tokens / client_wall_e2e_seconds",
            "e2e_seconds": [float(row["e2e_seconds"]) for row in measured],
            "completion_tokens": [
                int(row["usage"]["completion_tokens"]) for row in measured
            ],
            "finish_reasons": [row["finish_reason"] for row in measured],
            "per_row_f1": [float(row["f1"]) for row in measured],
            "paired_f1_delta_vs_full_recompute": [
                float(row["f1"]) - float(rows["full_recompute"][index]["measured"]["f1"])
                for index, row in enumerate(measured)
            ],
            "predictions": [row["prediction"] for row in measured],
            "prediction_text_exact_vs_full_recompute": [
                row["prediction"] == full_predictions[index]
                for index, row in enumerate(measured)
            ],
            "cached_tokens": cached,
        }
    cache_hypothesis_passed = all(
        value in (None, 0) for value in phase_summaries["full_recompute"]["cached_tokens"]
    ) and all(
        isinstance(value, int) and value > 0
        for mode in ("prefix_cache", "transition_rope_recompute")
        for value in phase_summaries[mode]["cached_tokens"]
    )
    payload = {
        "schema": "forkaudit-hypic-same-protocol-summary-v1",
        "scientific_run_valid": True,
        "protocol_validity": "passed",
        "cache_hypothesis_passed": cache_hypothesis_passed,
        "accuracy_outcome": "reported_without_predeclared_pass_threshold",
        "performance_outcome": "reported_without_predeclared_pass_threshold",
        "scientific_outcome": "valid_completed_result",
        "official_commit": HYPIC_COMMIT,
        "comparison_boundary": "same-model-same-slice-hypic-openai-streaming",
        "approximate_method": True,
        "modes": phase_summaries,
    }
    atomic_json(args.output, payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("token_gate", "server_receipt", "client", "aggregate"),
        required=True,
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--mode", choices=MODES)
    parser.add_argument("--base-url")
    parser.add_argument("--served-model-name", default="qwen35-hypic")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--expected-tp-size", type=int, default=1)
    parser.add_argument("--server-receipt", type=Path)
    parser.add_argument("--hypic-repo", type=Path)
    parser.add_argument("--source-ledger", type=Path)
    parser.add_argument("--environment-ledger", type=Path)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--launch-command-file", type=Path)
    parser.add_argument("--model-weight-ledger", type=Path)
    parser.add_argument("--model-artifact-ledger", type=Path)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.stage == "token_gate":
        _require(args.model is not None and args.data is not None, "token gate inputs")
        payload = {
            "schema": "forkaudit-hypic-token-identity-v1",
            "official_commit": HYPIC_COMMIT,
            "data_sha256": sha256_file(args.data),
            "rows": [
                {
                    key: value
                    for key, value in load_segmented_workload(args.model, args.data, rank).items()
                    if key
                    in {
                        "workload_id",
                        "dataset",
                        "source_index",
                        "prompt_token_sha256",
                        "document_token_sha256",
                        "segment_offsets",
                        "prime_segment_offsets",
                        "warm_segment_disjointness",
                    }
                }
                for rank in range(8)
            ],
            "all_token_identical": True,
        }
        atomic_json(args.output, payload)
    elif args.stage == "server_receipt":
        _require(
            args.model is not None
            and args.data is not None
            and args.mode is not None
            and args.base_url is not None
            and args.expected_gpu_uuid is not None
            and args.server_pid is not None
            and args.model_artifact_ledger is not None,
            "server receipt inputs",
        )
        server_receipt_stage(args)
    elif args.stage == "client":
        _require(args.model is not None and args.data is not None, "client inputs")
        _require(args.mode is not None and args.base_url is not None, "client endpoint")
        client_stage(args)
    else:
        _require(args.input_dir is not None, "aggregate input")
        aggregate_stage(args)


if __name__ == "__main__":
    main()
