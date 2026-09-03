from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from qcomem_lora import (
    CoMemLoRADistillation,
    PG19WindowDataset,
    ReplayQuantConfig,
    load_inference_lora_checkpoint,
    load_lora_state_dict,
    set_lora_enabled,
)
from qcomem_torch import TorchSplitCausalLM
from run_downstream import atomic_json
from train_qcomem_lora import reject_frozen_test_data


def parse_checkpoint(raw: str) -> tuple[int, Path, str]:
    pieces = raw.split("=", 2)
    if len(pieces) != 3:
        raise argparse.ArgumentTypeError("checkpoint must be STEP=PATH=SHA256")
    step = int(pieces[0])
    digest = pieces[2]
    if step not in {0, 64, 128} or len(digest) != 64:
        raise argparse.ArgumentTypeError("checkpoint step/SHA256 is outside the protocol")
    return step, Path(pieces[1]), digest


def checkpoint_payload(path: Path, expected_sha256: str, step: int) -> dict[str, Any]:
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"checkpoint-{step} SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    semantics = metadata.get("semantics", {})
    if payload.get("format") != "qcomem_suffix_lora_v1" or payload.get("step") != step:
        raise ValueError(f"checkpoint-{step} format/step mismatch")
    if semantics.get("student_suffix_execution_option") != "native-functional-cache":
        raise ValueError(f"checkpoint-{step} is not target-semantic native LoRA")
    if semantics.get("mode") != "quant" or semantics.get("depth") != 7:
        raise ValueError(f"checkpoint-{step} quant/depth semantics drifted")
    if metadata.get("test_v2_used") is not False:
        raise ValueError(f"checkpoint-{step} did not prove test-v2 exclusion")
    if step == 0 and metadata.get("native_initialization_control", {}).get(
        "optimizer_steps"
    ) != 0:
        raise ValueError("step-zero control provenance is missing")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Example-equal native-cache LoRA heldout KL evaluation"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--expected-data-sha256", required=True)
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--context-tokens", type=int, default=1536)
    parser.add_argument("--query-tokens", type=int, default=512)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not 0 <= args.rank < args.world_size:
        raise SystemExit("CUDA and a valid shard rank are required")
    if sorted(step for step, _, _ in args.checkpoint) != [0, 64, 128]:
        raise SystemExit("heldout selection requires exactly checkpoints 0,64,128")
    data_sha = reject_frozen_test_data(args.data)
    if data_sha != args.expected_data_sha256:
        raise SystemExit("heldout native-LoRA view SHA256 mismatch")

    from transformers import AutoModelForImageTextToText, AutoTokenizer

    torch.cuda.set_device(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    dataset = PG19WindowDataset(
        args.data,
        tokenizer,
        context_tokens=args.context_tokens,
        query_tokens=args.query_tokens,
        stride=args.context_tokens + args.query_tokens,
    )
    windows = dataset.windows[args.rank :: args.world_size]
    if not windows:
        raise SystemExit("heldout rank has no examples")
    checkpoints = [
        (step, path, digest, checkpoint_payload(path, digest, step))
        for step, path, digest in sorted(args.checkpoint)
    ]
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, local_files_only=True
    )
    if hasattr(model.model, "visual"):
        model.model.visual = None
    split = TorchSplitCausalLM(model)
    load_inference_lora_checkpoint(model, split.layers, checkpoints[0][1])
    model.eval().cuda()
    set_lora_enabled(model, True)
    core = CoMemLoRADistillation(
        model,
        mode="quant",
        depth=7,
        top_k=64,
        chunk_size=512,
        overlap=0,
        teacher_kind="q16_replay",
        teacher_source="online",
        quant=ReplayQuantConfig(
            residual_bits=4,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=(8, 8, 8, 4, 8, 8, 8),
            group_size=64,
        ),
        forward_weight=0.6,
        reverse_weight=0.4,
        temperature=1.0,
        student_suffix_execution="native-functional-cache",
    )
    checkpoint_rows = []
    for step, path, digest, payload in checkpoints:
        load_lora_state_dict(model, payload["lora"])
        set_lora_enabled(model, True)
        rows = []
        for window in windows:
            document = window.document_ids.cuda().unsqueeze(0)
            query = window.query_ids.cuda().unsqueeze(0)
            with torch.no_grad():
                outputs = core(document, query)
            audit = core.last_detached_cache_audit
            if not isinstance(audit, dict) or audit.get("hard_gate_passed") is not True:
                raise RuntimeError("heldout native cache audit failed")
            rows.append(
                {
                    "example_id": window.source_id,
                    "document_tokens": int(document.shape[1]),
                    "query_tokens": int(query.shape[1]),
                    "loss": float(outputs["loss"].item()),
                    "forward_kl": float(outputs["forward_kl"].item()),
                    "reverse_kl": float(outputs["reverse_kl"].item()),
                    "persistent_nbytes": int(outputs["persistent_nbytes"]),
                    "cache_audit": audit,
                }
            )
            del document, query, outputs
            torch.cuda.empty_cache()
        checkpoint_rows.append(
            {
                "training_step": step,
                "checkpoint": str(path),
                "checkpoint_sha256": digest,
                "rows": rows,
            }
        )
    result = {
        "status": "completed_shard",
        "rank": args.rank,
        "world_size": args.world_size,
        "data": str(args.data),
        "data_sha256": data_sha,
        "selection_metric": "arithmetic_mean_per_example_topk_bidirectional_kl",
        "teacher_execution": "q16_mutable_same_document_query_boundary",
        "student_execution": "q4_native_functional_same_document_query_boundary",
        "checkpoints": checkpoint_rows,
        "test_v2_used": False,
    }
    atomic_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "checkpoints"}))


if __name__ == "__main__":
    main()
