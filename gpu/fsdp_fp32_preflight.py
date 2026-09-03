from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch import nn

from fp32_master import (
    audit_adamw_fp32_state,
    audit_fp32_gradients,
    audit_fp32_parameter_delta,
    snapshot_fp32_local_shards,
)


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.proj(value))


class TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([TinyBlock(), TinyBlock()])
        self.lm_head = nn.Linear(8, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.language_model.layers:
            value = layer(value)
        return self.lm_head(value)


def main() -> None:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )

    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise SystemExit("FP32 FSDP preflight requires exactly two torchrun ranks")
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(101 + rank)
    model = TinyTextModel().to(torch.bfloat16)
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
    if {parameter.dtype for parameter in fsdp.parameters()} != {torch.bfloat16}:
        raise RuntimeError("tiny model was not BF16 immediately after FSDP wrap")
    fsdp.float()
    if {parameter.dtype for parameter in fsdp.parameters()} != {torch.float32}:
        raise RuntimeError("fsdp.float() did not promote persistent local shards")
    snapshots = snapshot_fp32_local_shards(fsdp.named_parameters())
    optimizer = torch.optim.AdamW(
        fsdp.parameters(), lr=1e-3, weight_decay=0.0, foreach=False
    )
    value = torch.randn(2, 3, 8, dtype=torch.bfloat16)
    output = fsdp(value)
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"mixed forward must remain BF16, got {output.dtype}")
    output.float().square().mean().backward()
    gradient = audit_fp32_gradients(snapshots)
    if gradient["total"]["missing_elements"] or gradient["total"]["nonfinite_elements"]:
        raise RuntimeError("tiny FP32 gradient coverage failed")
    if gradient["dtype_elements"] != {
        "torch.float32": gradient["total"]["parameter_elements"]
    }:
        raise RuntimeError(f"tiny gradients were not all FP32: {gradient['dtype_elements']}")
    optimizer.step()
    state = audit_adamw_fp32_state(optimizer, snapshots, expected_step=1)
    delta = audit_fp32_parameter_delta(snapshots)
    local = torch.tensor(
        [
            delta["fp32_logical"]["total"]["nonzero_elements"],
            delta["fp32_logical"]["total"]["l2_sq"],
        ],
        dtype=torch.float64,
    )
    dist.all_reduce(local)
    if local[0].item() <= 0 or local[1].item() <= 0:
        raise RuntimeError("tiny native FP32 FSDP parameter delta gate failed")
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "world_size": 2,
                    "persistent_parameter_dtype": "torch.float32",
                    "forward_dtype": str(output.dtype),
                    "gradient_dtype_elements": gradient["dtype_elements"],
                    "moment_dtype_elements": state["moment_dtype_elements"],
                    "global_changed_elements": int(local[0].item()),
                    "global_delta_l2": float(local[1].sqrt().item()),
                },
                sort_keys=True,
            )
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
