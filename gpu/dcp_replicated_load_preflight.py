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
    load_eval_model_only_fp32,
    save_eval_model_only_fp32,
)


class TinyCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Sequential(
            nn.Linear(8, 8, bias=False),
            nn.Tanh(),
            nn.Linear(8, 8, bias=False),
        )
        self.lm_head = nn.Linear(8, 4, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.language_model(value))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "CPU gate: reshard a two-rank FP32 FSDP DCP into one complete BF16 "
            "replica on every rank"
        )
    )
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise SystemExit("replicated-load DCP preflight requires two torchrun ranks")

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(317)

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
    )

    source = TinyCore().float()
    expected_parameters = sum(parameter.numel() for parameter in source.parameters())
    fsdp = FSDP(
        source,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
        device_id=torch.device("cpu"),
    )
    optimizer = torch.optim.SGD(fsdp.parameters(), lr=0.05)
    value = torch.arange(48, dtype=torch.float32).reshape(2, 3, 8) / 31
    fsdp(value).square().mean().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    root_holder: list[object] = [
        (
            str(args.output_root)
            if args.output_root is not None
            else tempfile.mkdtemp(prefix="qcomem-dcp-replicated-preflight-")
        )
        if rank == 0
        else None
    ]
    dist.broadcast_object_list(root_holder, src=0)
    root = Path(str(root_holder[0]))
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    checkpoint = root / "step-000001-eval-model-only-fp32"
    saved = save_eval_model_only_fp32(
        fsdp,
        checkpoint,
        step=1,
        expected_parameters=expected_parameters,
        provenance={"kind": "replicated_bf16_load_preflight"},
    )

    # The real evaluator follows the same transition: the DCP was written from
    # an FP32 FSDP core, while each inference rank owns one complete BF16 core.
    del optimizer, fsdp
    torch.manual_seed(999 + rank)
    replica = TinyCore().to(torch.bfloat16)
    before = {name: value.detach().clone() for name, value in replica.state_dict().items()}
    loaded = load_eval_model_only_fp32(
        replica,
        checkpoint,
        expected_manifest_sha256=saved["checkpoint_manifest_sha256"],
    )
    changed = sum(
        int(torch.count_nonzero(before[name] != value).item())
        for name, value in replica.state_dict().items()
    )
    if changed < 1:
        raise RuntimeError("DCP load did not change the independently seeded replica")
    flattened = torch.cat(
        [value.detach().float().reshape(-1) for value in replica.state_dict().values()]
    )
    gathered = [torch.empty_like(flattened) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, flattened)
    if any(not torch.equal(gathered[0], candidate) for candidate in gathered[1:]):
        raise RuntimeError("replicated BF16 DCP load differs across ranks")
    if rank == 0:
        print(
            json.dumps(
                {
                    "status": "passed",
                    "save_layout": "FSDP_FULL_SHARD_FP32",
                    "load_layout": "complete_BF16_replica_per_rank",
                    "world_size": dist.get_world_size(),
                    "global_parameter_count": expected_parameters,
                    "replica_parameter_count": flattened.numel(),
                    "replicas_exactly_equal": True,
                    "destination_changed_elements": changed,
                    "checkpoint_manifest_sha256": loaded[
                        "checkpoint_manifest_sha256"
                    ],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
