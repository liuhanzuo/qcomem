#!/usr/bin/env python3
"""Source-distinct dense Transformers reference for the R30 semantic control."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForImageTextToText


SCHEMA = "forkaudit-r30-e2e-reference-v1"
INPUT_SCHEMA = "forkaudit-r30-e2e-input-manifest-v1"
MODEL_REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
EXPECTED_GPU_UUID = "GPU-d917fce5-80f1-78ac-3965-0476bf8bd441"
FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "qcomem_joint_policy",
        "qcomem_qwen35_functional_stack",
        "qcomem_qwen35_native_cache",
        "qcomem_qwen35_vllm_paged_integration",
        "qcomem_vllm_paged_fair_control",
        "qcomem_vllm_paged_kernel",
        "qcomem_vllm_paged_multifork_resident",
        "qcomem_single_token_gdn_ownership",
        "vllm",
    }
)


class ReferenceError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReferenceError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def int64_sha256(values: list[int]) -> str:
    array = np.asarray(values, dtype="<i8")
    require(array.ndim == 1, "token array rank drift")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def write_sidecar(root: Path, record_id: str, logits: torch.Tensor) -> dict[str, Any]:
    array = logits.detach().float().contiguous().cpu().numpy().astype("<f4", copy=False)
    require(array.ndim == 1 and array.size > 1, "reference logit vector shape drift")
    require(bool(np.isfinite(array).all()), "reference logits are non-finite")
    relative = Path("reference") / "logits" / (record_id + ".npy")
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("xb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return {
        "record_id": record_id,
        "path": relative.as_posix(),
        "sha256": sha256_file(target),
        "shape": [int(value) for value in array.shape],
        "dtype": "float32",
        "argmax_token_id": int(np.argmax(array)),
    }


def gpu_receipt() -> dict[str, Any]:
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "4", "reference GPU isolation drift")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    require(torch.cuda.device_count() == 1, "reference must see exactly one GPU")
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    uuid = "GPU-" + str(getattr(properties, "uuid", ""))
    require(uuid == EXPECTED_GPU_UUID, f"unexpected visible GPU UUID: {uuid}")
    require("H20" in str(properties.name), "visible device is not H20")
    return {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "visible_device_count": torch.cuda.device_count(),
        "visible_index": 0,
        "uuid": uuid,
        "name": str(properties.name),
        "compute_capability": [int(properties.major), int(properties.minor)],
        "total_memory_bytes": int(properties.total_memory),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
    }


def import_receipt() -> dict[str, Any]:
    roots = sorted(name for name in sys.modules if name.split(".", 1)[0] in FORBIDDEN_IMPORT_ROOTS)
    require(not roots, f"reference process imported forbidden roots: {roots}")
    return {
        "forbidden_import_roots": sorted(FORBIDDEN_IMPORT_ROOTS),
        "observed_forbidden_modules": roots,
        "candidate_cache_trace_tensor_objects_imported": False,
    }


def load_input(path: Path, expected_sha256: str) -> dict[str, Any]:
    require(sha256_file(path) == expected_sha256, "input manifest SHA drift")
    value = json.loads(path.read_bytes())
    require(value.get("schema_version") == INPUT_SCHEMA, "input schema drift")
    require(value.get("created_before_any_model_output") is True, "input was not output-unseen")
    require(value.get("candidate_or_reference_model_invoked") is False, "input freeze invoked a model")
    cases = value.get("cases")
    require(isinstance(cases, list) and len(cases) == 2, "reference case count drift")
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    require(args.output.resolve().parent == args.artifact_root.resolve(), "output must be at artifact root")
    require(not args.output.exists(), "reference output already exists")
    require(not args.candidate_sentinel.exists(), "candidate artifact exists before reference")
    inputs = load_input(args.input_manifest, args.expected_input_sha256)
    gpu = gpu_receipt()
    imports = import_receipt()

    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        revision=MODEL_REVISION,
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=False,
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    model.eval().cuda()
    rows = []
    sidecars = []
    expected_steps = int(inputs["selection"]["greedy_steps"])
    with torch.inference_mode():
        for case in inputs["cases"]:
            document_ids = [int(value) for value in case["document_token_ids"]]
            require(int64_sha256(document_ids) == case["document_token_ids_sha256"], "document ID digest drift")
            for query in case["queries"]:
                query_ids = [int(value) for value in query["token_ids"]]
                require(int64_sha256(query_ids) == query["token_ids_sha256"], "query ID digest drift")
                history = document_ids + query_ids
                generated: list[int] = []
                step_rows = []
                for step_index in range(expected_steps):
                    raw = torch.tensor([history], dtype=torch.int64, device="cuda:0")
                    output = model(input_ids=raw, use_cache=False, logits_to_keep=1)
                    logits = output.logits[0, -1, :]
                    token = int(torch.argmax(logits).item())
                    record_id = (
                        f"case-{case['case_index']}/request-{query['request_index']}"
                        f"/step-{step_index}"
                    )
                    receipt = write_sidecar(args.artifact_root, record_id, logits)
                    require(receipt["argmax_token_id"] == token, "sidecar argmax drift")
                    sidecars.append(receipt)
                    step_rows.append(
                        {
                            "step_index": step_index,
                            "raw_history_token_count": len(history),
                            "raw_history_token_ids_sha256": int64_sha256(history),
                            "generated_token_id": token,
                            "logit_record_id": record_id,
                        }
                    )
                    generated.append(token)
                    history.append(token)
                    del raw, output, logits
                rows.append(
                    {
                        "case_index": int(case["case_index"]),
                        "case_id": str(case["case_id"]),
                        "request_index": int(query["request_index"]),
                        "document_token_ids_sha256": case["document_token_ids_sha256"],
                        "query_token_ids_sha256": query["token_ids_sha256"],
                        "generated_token_ids": generated,
                        "steps": step_rows,
                    }
                )
    require(len(rows) == 4 and len(sidecars) == 16, "reference denominator drift")
    return {
        "schema_version": SCHEMA,
        "status": "completed_dense_reference",
        "input_manifest_sha256": args.expected_input_sha256,
        "reference_process_started_before_candidate": True,
        "reference_source_distinct": True,
        "reference_inputs": "frozen raw token IDs and local model weights only",
        "candidate_cache_trace_tensor_objects_imported": False,
        "full_model_recompute_each_step": True,
        "use_cache": False,
        "model": {
            "model_id": "Qwen/Qwen3.5-35B-A3B",
            "revision": MODEL_REVISION,
            "dtype": "torch.bfloat16",
            "local_files_only": True,
        },
        "gpu": gpu,
        "imports": imports,
        "versions": {
            "torch": str(torch.__version__),
            "transformers": str(sys.modules["transformers"].__version__),
            "numpy": str(np.__version__),
        },
        "rows": rows,
        "sidecars": sidecars,
        "denominators": {"trajectories": 4, "greedy_decisions": 16, "full_vocab_sidecars": 16},
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--input-manifest", type=Path, required=True)
    result.add_argument("--expected-input-sha256", required=True)
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--candidate-sentinel", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> None:
    args = parser().parse_args()
    value = run(args)
    atomic_write(args.output, canonical_bytes(value))
    print(
        json.dumps(
            {
                "status": value["status"],
                "output": str(args.output),
                "output_sha256": sha256_file(args.output),
                "generated_token_ids": [row["generated_token_ids"] for row in value["rows"]],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
