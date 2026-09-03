from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from sft_dcp_checkpoint import (
    EVAL_MODEL_ONLY_CONTRACT,
    load_eval_model_only_fp32,
    save_eval_model_only_fp32,
    validate_checkpoint_manifest,
)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(value))


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.head = nn.Linear(8, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return self.head(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-rank CPU round-trip for the formal FP32 model-only DCP contract"
    )
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise SystemExit("DCP preflight requires exactly two torchrun ranks")
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(101)

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )

    model = TinyModel().to(torch.bfloat16)
    expected_parameters = sum(parameter.numel() for parameter in model.parameters())
    fsdp = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            keep_low_precision_grads=False,
        ),
        use_orig_params=True,
        device_id=torch.device("cpu"),
    )
    fsdp.float()
    optimizer = torch.optim.AdamW(fsdp.parameters(), lr=1e-3, foreach=False)
    value = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    loss = fsdp(value).float().square().mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    if args.output_root is None:
        root_holder: list[object] = [
            tempfile.mkdtemp(prefix="qcomem-dcp-preflight-") if rank == 0 else None
        ]
        dist.broadcast_object_list(root_holder, src=0)
        root = Path(str(root_holder[0]))
    else:
        root = args.output_root
        if rank == 0:
            if root.exists():
                raise RuntimeError(f"preflight output root already exists: {root}")
            root.mkdir(parents=True)
        dist.barrier()
    checkpoint = root / "step-000001-eval-model-only-fp32"
    provenance = {
        "formal_format": "tiny_preflight_only",
        "train_jsonl_sha256": "1" * 64,
        "heldout_ce_jsonl_sha256": "2" * 64,
        "split_manifest_sha256": "3" * 64,
    }
    saved = save_eval_model_only_fp32(
        fsdp,
        checkpoint,
        step=1,
        expected_parameters=expected_parameters,
        provenance=provenance,
    )
    before = {
        name: parameter.detach().clone()
        for name, parameter in fsdp.named_parameters()
        if parameter.numel() > 0
    }
    with torch.no_grad():
        for parameter in fsdp.parameters():
            parameter.add_(1.0)
    loaded = load_eval_model_only_fp32(
        fsdp,
        checkpoint,
        expected_manifest_sha256=saved["checkpoint_manifest_sha256"],
    )
    for name, parameter in fsdp.named_parameters():
        if parameter.numel() > 0 and not torch.equal(parameter, before[name]):
            raise RuntimeError(f"DCP round-trip changed local FP32 shard {name}")
    dist.barrier()
    if rank == 0:
        verified = validate_checkpoint_manifest(
            checkpoint,
            expected_contract=EVAL_MODEL_ONLY_CONTRACT,
            expected_manifest_sha256=saved["checkpoint_manifest_sha256"],
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "world_size": 2,
                    "contract": verified["contract"],
                    "global_parameter_count": expected_parameters,
                    "persistent_parameter_dtype": "torch.float32",
                    "rank0_full_gather_used": verified["rank0_full_gather_used"],
                    "checkpoint_manifest_sha256": verified[
                        "checkpoint_manifest_sha256"
                    ],
                    "payload_directory_sha256": verified[
                        "payload_directory_sha256"
                    ],
                    "success_marker": (checkpoint / "_SUCCESS").is_file(),
                    "roundtrip_exact": True,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
