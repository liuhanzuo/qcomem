from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.checkpoint

from deployment_aware_sft import DeploymentAwareCausalLM
from sft_dcp_checkpoint import save_eval_model_only_fp32, validate_checkpoint_manifest


class TinyDecoderLayer(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + torch.tanh(self.proj(self.norm(value)))


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab: int = 32, hidden: int = 16) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([TinyDecoderLayer(hidden), TinyDecoderLayer(hidden)])
        self.norm = nn.LayerNorm(hidden)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        value = self.embed_tokens(input_ids)
        for layer in self.layers:
            value = torch.utils.checkpoint.checkpoint(layer, value, use_reentrant=False)
        return type("Output", (), {"last_hidden_state": self.norm(value)})()


def main() -> None:
    parser = argparse.ArgumentParser(description="2-rank CPU FSDP/checkpoint/DCP preflight")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 2:
        raise SystemExit("preflight requires two torchrun ranks")
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.manual_seed(7)
    language_model = TinyLanguageModel()
    model = DeploymentAwareCausalLM(language_model, nn.Linear(16, 32, bias=False))
    expected_parameters = sum(parameter.numel() for parameter in model.parameters())

    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    fsdp = FSDP(
        model.to(torch.bfloat16),
        auto_wrap_policy=functools.partial(
            transformer_auto_wrap_policy, transformer_layer_cls={TinyDecoderLayer}
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            keep_low_precision_grads=False,
        ),
        use_orig_params=True,
        limit_all_gathers=True,
        device_id=torch.device("cpu"),
    )
    fsdp.float()
    optimizer = torch.optim.AdamW(fsdp.parameters(), lr=1e-3, foreach=False)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    labels = torch.tensor([[-100, -100, 3, 4]])
    loss, ce, _, _ = fsdp(input_ids, labels, torch.ones_like(input_ids))
    if not torch.isfinite(loss):
        raise RuntimeError("tiny FSDP loss is non-finite")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if rank == 0:
        args.output_root.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    checkpoint = save_eval_model_only_fp32(
        fsdp,
        args.output_root / "checkpoint",
        step=1,
        expected_parameters=expected_parameters,
        provenance={"formal_format": "deployment_aware_tiny_preflight_only"},
    )
    if rank == 0:
        verified = validate_checkpoint_manifest(
            args.output_root / "checkpoint",
            expected_manifest_sha256=checkpoint["checkpoint_manifest_sha256"],
        )
        print(
            json.dumps(
                {
                    "status": "passed",
                    "world_size": 2,
                    "full_decoder_layer_checkpoint": True,
                    "FSDP_transformer_auto_wrap": True,
                    "DCP_model_only_FP32": verified["persistent_parameter_dtype"]
                    == "torch.float32",
                },
                sort_keys=True,
            ),
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
