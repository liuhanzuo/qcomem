from __future__ import annotations

"""Build and execute the R30 fixed-stack native-vLLM batching experiment."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


PREREG_SCHEMA = "forkaudit-r30-native-vllm-preregistration-v1"
INPUT_SCHEMA = "forkaudit-r30-native-vllm-input-manifest-v1"
OUTPUT_SCHEMA = "forkaudit-r30-native-vllm-output-comparison-v1"
RESULT_SCHEMA = "forkaudit-r30-native-vllm-result-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_bound_json(path: Path, expected_sha256: str, schema: str) -> dict[str, Any]:
    raw = path.read_bytes()
    require(sha256_bytes(raw) == expected_sha256, f"raw SHA mismatch: {path}")
    value = json.loads(raw)
    require(value.get("schema_version") == schema, f"schema mismatch: {path}")
    return value


def verify_sources(prereg: dict[str, Any], source_dir: Path) -> None:
    for name, expected in prereg["source_sha256"].items():
        path = source_dir / name
        require(path.is_file(), f"missing executed source: {name}")
        require(sha256_file(path) == expected, f"executed source SHA drift: {name}")


def parse_ledger(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(None, 1)
        rows.append((digest, name.strip()))
    return rows


def verify_model_ledger(model: Path, ledger: Path, expected_raw_sha256: str) -> None:
    require(sha256_file(ledger) == expected_raw_sha256, f"model ledger SHA drift: {ledger.name}")
    for expected, name in parse_ledger(ledger):
        target = model / name
        require(target.is_file(), f"model ledger member missing: {name}")
        require(sha256_file(target) == expected, f"model member SHA drift: {name}")


def load_pg19_rows(data: Path, count: int) -> list[dict[str, Any]]:
    rows = []
    with data.open() as stream:
        for _ in range(count):
            rows.append(json.loads(next(stream)))
    return rows


def build_static(args: argparse.Namespace) -> None:
    root = args.output_dir
    static_dir = root / "static"
    require(not static_dir.exists(), "static output already exists")
    prereg = load_bound_json(args.preregistration, args.expected_prereg_sha256, PREREG_SCHEMA)
    source_dir = Path(__file__).resolve().parent
    verify_sources(prereg, source_dir)
    require(sha256_file(args.data) == prereg["data_sha256"], "PG19 data SHA drift")
    verify_model_ledger(args.model, args.model_artifact_ledger, prereg["model_artifact_ledger_sha256"])
    verify_model_ledger(args.model, args.model_weight_ledger, prereg["model_weight_ledger_sha256"])

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    request_specs = prereg["requests"]
    rows = load_pg19_rows(args.data, len(request_specs))
    manifest_rows = []
    for spec, row in zip(request_specs, rows, strict=True):
        prefix = row["text"][: int(prereg["text_prefix_characters"])]
        token_ids = tokenizer.encode(prefix, add_special_tokens=False)
        target = int(spec["prompt_tokens"])
        require(len(token_ids) >= target, f"PG19 source too short for role {spec['role']}")
        token_ids = [int(value) for value in token_ids[:target]]
        manifest_rows.append(
            {
                "role": spec["role"],
                "source_id": str(row["id"]),
                "source_object": row["_source_object"],
                "prompt_tokens": target,
                "max_output_tokens": int(spec["max_output_tokens"]),
                "prompt_token_ids_sha256": sha256_bytes(canonical_bytes(token_ids)),
                "prompt_token_ids": token_ids,
            }
        )
    require(len({row["source_object"] for row in manifest_rows}) == 3, "PG19 sources are not distinct")
    manifest = {
        "schema_version": INPUT_SCHEMA,
        "created_before_gpu_model_load": True,
        "preregistration_raw_sha256": args.expected_prereg_sha256,
        "data_sha256": prereg["data_sha256"],
        "model_artifact_ledger_sha256": prereg["model_artifact_ledger_sha256"],
        "model_weight_ledger_sha256": prereg["model_weight_ledger_sha256"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "requests": manifest_rows,
    }
    atomic_json(static_dir / "input_manifest.json", manifest)
    (static_dir / "MODEL_LEDGER_VERIFIED").write_text("artifact and weight ledgers verified\n")
    print(json.dumps({"input_manifest_sha256": sha256_file(static_dir / "input_manifest.json")}, sort_keys=True))


def append_marker(trace: Path, phase: str) -> None:
    from r30_native_scheduler_trace import TRACE_SCHEMA

    row = {
        "schema_version": TRACE_SCHEMA,
        "pid": os.getpid(),
        "kind": "phase_marker",
        "phase": phase,
        "monotonic_ns": time.monotonic_ns(),
    }
    descriptor = os.open(trace, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(descriptor, (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode())
    finally:
        os.close(descriptor)


def output_by_prompt_sha(outputs: list[Any]) -> dict[str, Any]:
    result = {}
    for output in outputs:
        prompt_ids = [int(value) for value in output.prompt_token_ids]
        digest = sha256_bytes(canonical_bytes(prompt_ids))
        require(digest not in result, "duplicate prompt output")
        result[digest] = output
    return result


def save_output(
    output: Any,
    *,
    phase: str,
    role: str,
    raw_dir: Path,
    vocab_size: int,
) -> dict[str, Any]:
    require(len(output.outputs) == 1, f"{role} expected one candidate")
    sequence = output.outputs[0]
    token_ids = [int(value) for value in sequence.token_ids]
    require(sequence.logprobs is not None, f"{role} missing output log-probs")
    require(len(sequence.logprobs) == len(token_ids), f"{role} log-prob step count drift")
    vectors = np.empty((len(token_ids), vocab_size), dtype=np.float32)
    for step, mapping in enumerate(sequence.logprobs):
        require(len(mapping) == vocab_size, f"{role} step {step} is not full-vocabulary")
        present = np.zeros(vocab_size, dtype=np.bool_)
        vectors[step].fill(np.nan)
        for token_id, record in mapping.items():
            token = int(token_id)
            require(0 <= token < vocab_size, f"{role} out-of-range log-prob token")
            value = float(getattr(record, "logprob", record))
            vectors[step, token] = value
            present[token] = True
        require(bool(present.all()), f"{role} incomplete full-vocabulary vector")
    sidecar = raw_dir / "logprobs" / f"{phase}-{role}.npy"
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    np.save(sidecar, vectors, allow_pickle=False)
    return {
        "request_id": str(output.request_id),
        "role": role,
        "phase": phase,
        "prompt_tokens": len(output.prompt_token_ids),
        "prompt_token_ids_sha256": sha256_bytes(canonical_bytes([int(value) for value in output.prompt_token_ids])),
        "generated_token_ids": token_ids,
        "generated_text_sha256": sha256_bytes(sequence.text.encode()),
        "finish_reason": str(sequence.finish_reason),
        "full_vocab_logprobs_shape": list(vectors.shape),
        "full_vocab_logprobs_dtype": str(vectors.dtype),
        "full_vocab_logprobs_path": str(sidecar.relative_to(raw_dir.parent)),
        "full_vocab_logprobs_sha256": sha256_file(sidecar),
    }


def compare_output_rows(
    batch: dict[str, Any],
    control: dict[str, Any],
    root: Path,
    tolerance: dict[str, float],
) -> dict[str, Any]:
    left = np.load(root / batch["full_vocab_logprobs_path"], allow_pickle=False)
    right = np.load(root / control["full_vocab_logprobs_path"], allow_pickle=False)
    require(left.shape == right.shape, f"{batch['role']} log-prob shape drift")
    finite = np.isfinite(left) & np.isfinite(right)
    same_nonfinite = bool(np.array_equal(np.isfinite(left), np.isfinite(right)))
    absolute = np.abs(left[finite].astype(np.float64) - right[finite].astype(np.float64))
    max_abs = float(absolute.max()) if absolute.size else 0.0
    mean_abs = float(absolute.mean()) if absolute.size else 0.0
    close = bool(
        same_nonfinite
        and np.allclose(
            left,
            right,
            rtol=float(tolerance["rtol"]),
            atol=float(tolerance["atol"]),
            equal_nan=True,
        )
    )
    return {
        "token_ids_exact": batch["generated_token_ids"] == control["generated_token_ids"],
        "batch_token_ids": batch["generated_token_ids"],
        "sequential_token_ids": control["generated_token_ids"],
        "same_nonfinite_pattern": same_nonfinite,
        "full_vocab_logprobs_max_abs_error": max_abs,
        "full_vocab_logprobs_mean_abs_error": mean_abs,
        "full_vocab_logprobs_atol": float(tolerance["atol"]),
        "full_vocab_logprobs_rtol": float(tolerance["rtol"]),
        "full_vocab_logprobs_within_preregistered_tolerance": close,
    }


def package_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "transformers", "vllm", "triton"):
        result[name] = importlib.metadata.version(name)
    return result


def execute(args: argparse.Namespace) -> None:
    root = args.output_dir
    raw_dir = root / "raw"
    require((root / "static" / "MODEL_LEDGER_VERIFIED").is_file(), "model ledger preflight marker missing")
    require(not raw_dir.exists(), "raw formal output already exists")
    prereg = load_bound_json(args.preregistration, args.expected_prereg_sha256, PREREG_SCHEMA)
    verify_sources(prereg, Path(__file__).resolve().parent)
    manifest_path = root / "static" / "input_manifest.json"
    manifest = load_bound_json(manifest_path, args.expected_input_sha256, INPUT_SCHEMA)
    require(manifest["preregistration_raw_sha256"] == args.expected_prereg_sha256, "input/prereg binding drift")
    raw_dir.mkdir(parents=True)
    trace_path = (raw_dir / "scheduler_trace.jsonl").resolve()
    os.environ["R30_NATIVE_TRACE_PATH"] = str(trace_path)

    import torch
    from vllm import LLM, SamplingParams

    require(torch.cuda.is_available(), "CUDA unavailable")
    physical_gpu_index = int(os.environ["R30_PHYSICAL_GPU_INDEX"])
    require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == str(physical_gpu_index),
        "CUDA_VISIBLE_DEVICES must bind exactly the preregistered physical GPU",
    )
    environment = {
        "schema_version": "forkaudit-r30-native-vllm-environment-v1",
        "python": platform.python_version(),
        "packages": package_versions(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_visible_gpu_count": torch.cuda.device_count(),
        "torch_device_name": torch.cuda.get_device_name(0),
        "torch_device_capability": list(torch.cuda.get_device_capability(0)),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "nvidia_smi_assigned_gpu": subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(physical_gpu_index),
                "--query-gpu=index,uuid,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "engine_config": prereg["engine_config"],
    }
    atomic_json(root / "environment.json", environment)

    engine = prereg["engine_config"]
    llm = LLM(
        model=str(args.model),
        tokenizer=str(args.model),
        dtype="bfloat16",
        seed=int(prereg["seed"]),
        max_model_len=int(engine["max_model_len"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        enforce_eager=True,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        language_model_only=True,
        generation_config="vllm",
        scheduler_cls="r30_native_scheduler_trace.TracingScheduler",
        max_logprobs=-1,
    )

    request_rows = manifest["requests"]
    prompts = [{"prompt_token_ids": row["prompt_token_ids"]} for row in request_rows]
    params = [
        SamplingParams(
            temperature=0.0,
            max_tokens=int(row["max_output_tokens"]),
            ignore_eos=True,
            logprobs=-1,
            seed=int(prereg["seed"]),
        )
        for row in request_rows
    ]

    append_marker(trace_path, "warmup")
    llm.generate(
        [{"prompt_token_ids": request_rows[0]["prompt_token_ids"][:129]}],
        SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True, seed=int(prereg["seed"])),
        use_tqdm=False,
    )

    append_marker(trace_path, "native_batch")
    batch_outputs = llm.generate(prompts, params, use_tqdm=False)
    batch_by_sha = output_by_prompt_sha(batch_outputs)

    sequential_by_role = {}
    for row, prompt, sampling in zip(request_rows, prompts, params, strict=True):
        role = row["role"]
        append_marker(trace_path, f"sequential_{role.lower()}")
        values = llm.generate([prompt], sampling, use_tqdm=False)
        sequential_by_role[role] = values[0]

    output_rows = {"native_batch": {}, "sequential": {}}
    vocab_size = int(manifest["tokenizer_vocab_size"])
    for row in request_rows:
        role = row["role"]
        digest = row["prompt_token_ids_sha256"]
        require(digest in batch_by_sha, f"missing native batch output for {role}")
        output_rows["native_batch"][role] = save_output(
            batch_by_sha[digest], phase="native_batch", role=role, raw_dir=raw_dir, vocab_size=vocab_size
        )
        output_rows["sequential"][role] = save_output(
            sequential_by_role[role], phase="sequential", role=role, raw_dir=raw_dir, vocab_size=vocab_size
        )

    comparisons = {
        role: compare_output_rows(
            output_rows["native_batch"][role],
            output_rows["sequential"][role],
            root,
            prereg["full_vocab_logprob_tolerance"],
        )
        for role in ("A", "B", "C")
    }
    outputs_receipt = {
        "schema_version": OUTPUT_SCHEMA,
        "vocab_size": vocab_size,
        "rows": output_rows,
        "comparisons": comparisons,
    }
    atomic_json(raw_dir / "outputs.json", outputs_receipt)

    source_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(source_dir))
    from r30_replay_native_batching import analyze

    replay = analyze(root)
    atomic_json(root / "replay_report.json", replay)
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": "completed",
        "scientific_run_valid": True,
        "formal_evidence_eligible": True,
        "passed_preregistered_gates": replay["status"] == "passed",
        "job_id": int(os.environ["QS_JOB_ID"]),
        "trial_id": int(os.environ["QS_TRIAL_ID"]),
        "pod": platform.node(),
        "physical_gpu_index": physical_gpu_index,
        "preregistration_raw_sha256": args.expected_prereg_sha256,
        "input_manifest_raw_sha256": args.expected_input_sha256,
        "environment_sha256": sha256_file(root / "environment.json"),
        "trace_sha256": sha256_file(trace_path),
        "outputs_sha256": sha256_file(raw_dir / "outputs.json"),
        "replay_sha256": sha256_file(root / "replay_report.json"),
        "native_engine_evidence": replay["native_engine_evidence"],
        "output_comparisons": comparisons,
        "claim_boundary": replay["claim_boundary"],
    }
    atomic_json(root / "formal_result.json", result)

    checksums = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha256_file(path)}  {path.relative_to(root)}")
    (root / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    (root / "COMPLETED").write_text("formal native-vLLM batching run complete\n")
    print(json.dumps(result, sort_keys=True))
    del llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("build-static", "execute"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-artifact-ledger", type=Path, required=True)
    parser.add_argument("--model-weight-ledger", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--expected-prereg-sha256", required=True)
    parser.add_argument("--expected-input-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "build-static":
        build_static(args)
    else:
        require(bool(args.expected_input_sha256), "execute requires --expected-input-sha256")
        execute(args)


if __name__ == "__main__":
    main()
