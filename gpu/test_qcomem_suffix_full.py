from __future__ import annotations

import unittest

import torch
from torch import nn

from qcomem_suffix_full import (
    aggregate_suffix_gradient_coverage,
    configure_suffix_full_trainability,
    end_to_end_full_model_capability_gate,
    estimate_sharded_training_storage,
    suffix_full_semantics_metadata,
    suffix_gradient_coverage_local,
)
from qcomem_lora import ReplayQuantConfig


class TinyAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width, bias=False)


class TinyMLP(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(width, width * 2, bias=False)
        self.down_proj = nn.Linear(width * 2, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.relu(self.up_proj(hidden)))


class TinyLayer(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = TinyAttention(width)
        self.mlp = TinyMLP(width)
        self.input_layernorm = nn.LayerNorm(width)


class TinySuffixModel(nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(10, width)
        self.layers = nn.ModuleList([TinyLayer(width) for _ in range(3)])
        self.norm = nn.LayerNorm(width)
        self.lm_head = nn.Linear(width, 10, bias=False)


class SuffixFullTest(unittest.TestCase):
    def test_exact_suffix_plan_freezes_everything_before_depth(self) -> None:
        model = TinySuffixModel()
        plan = configure_suffix_full_trainability(model, model.layers, depth=1)

        expected = sum(parameter.numel() for layer in model.layers[1:] for parameter in layer.parameters())
        self.assertEqual(plan["estimated_trainable_parameters"], expected)
        self.assertEqual(plan["actual_trainable_parameters"], expected)
        self.assertEqual(sum(plan["category_parameter_counts"].values()), expected)
        self.assertTrue(all(not parameter.requires_grad for parameter in model.layers[0].parameters()))
        self.assertTrue(all(parameter.requires_grad for layer in model.layers[1:] for parameter in layer.parameters()))
        self.assertFalse(model.embed_tokens.weight.requires_grad)
        self.assertFalse(model.norm.weight.requires_grad)
        self.assertFalse(model.lm_head.weight.requires_grad)

    def test_storage_ledger_has_bf16_and_fp32_optimizer_bounds(self) -> None:
        ledger = estimate_sharded_training_storage(
            trainable_parameters=80,
            parameter_bytes=160,
            world_size=8,
        )
        self.assertEqual(ledger["global"]["gradient_bytes"], 160)
        self.assertEqual(ledger["global"]["adam_two_moments_if_bf16_bytes"], 320)
        self.assertEqual(ledger["global"]["adam_two_moments_if_fp32_bytes"], 640)
        self.assertEqual(ledger["ideal_even_shard_per_rank"]["training_checkpoint_if_fp32_moments_bytes"], 100)

    def test_end_to_end_gate_is_explicitly_closed(self) -> None:
        gate = end_to_end_full_model_capability_gate()
        self.assertFalse(gate["implemented"])
        self.assertFalse(gate["capability_gate_passed"])
        self.assertEqual(
            gate["loss_plan"]["token_ce_sft"],
            "separate future ablation; not implemented by this gate",
        )

    def test_suffix_full_metadata_never_claims_lora_or_full_sft(self) -> None:
        semantics = suffix_full_semantics_metadata(
            depth=7,
            teacher_source="online",
            quant=ReplayQuantConfig(
                residual_bits=4,
                attention_bits=4,
                linear_bits=8,
                cache_layer_bits=(8, 8, 8, 4, 8, 8, 8),
            ),
            student_suffix_execution="cached-two-stage",
        )
        self.assertFalse(semantics["is_lora"])
        self.assertFalse(semantics["is_qlora"])
        self.assertFalse(semantics["is_full_model_sft"])
        self.assertFalse(semantics["is_end_to_end_qat"])
        self.assertTrue(semantics["suffix_transformer_layers_trainable"])
        self.assertNotIn("gradients enter LoRA", semantics["note"])

        detached = suffix_full_semantics_metadata(
            depth=7,
            teacher_source="online",
            quant=ReplayQuantConfig(),
            student_suffix_execution="detached-document-cache",
        )
        self.assertTrue(detached["document_cache_detached_before_query"])
        self.assertFalse(detached["document_prefill_parameter_gradients_enabled"])
        self.assertIn("query-continuation-only", detached["claim_limit"])

    def test_real_fsdp_apis_import(self) -> None:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            checkpoint_wrapper,
        )
        from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

        self.assertIsNotNone(FullyShardedDataParallel)
        self.assertIsNotNone(ShardingStrategy.FULL_SHARD)
        self.assertIsNotNone(CheckpointImpl.NO_REENTRANT)
        self.assertTrue(callable(checkpoint_wrapper))

    def test_suffix_gradient_coverage_is_scalar_and_layerwise(self) -> None:
        model = TinySuffixModel()
        configure_suffix_full_trainability(model, model.layers, depth=1)
        inputs = torch.tensor([[1, 2]])
        hidden = model.embed_tokens(inputs).detach()
        for layer in model.layers[1:]:
            normalized = layer.input_layernorm(hidden)
            hidden = layer.mlp(normalized) + layer.self_attn.q_proj(normalized)
        hidden.square().mean().backward()
        local = suffix_gradient_coverage_local(
            model,
            depth=1,
            num_layers=len(model.layers),
        )
        aggregate = aggregate_suffix_gradient_coverage(
            [local],
            depth=1,
            num_layers=len(model.layers),
        )
        self.assertTrue(aggregate["hard_gate_passed"])
        self.assertFalse(aggregate["large_gradient_tensors_gathered"])
        self.assertTrue(all(row["finite"] for row in aggregate["layers"].values()))

    def test_suffix_gradient_gate_rejects_partial_missing_gradients(self) -> None:
        model = TinySuffixModel()
        configure_suffix_full_trainability(model, model.layers, depth=1)
        inputs = torch.tensor([[1, 2]])
        hidden = model.embed_tokens(inputs).detach()
        for layer in model.layers[1:]:
            normalized = layer.input_layernorm(hidden)
            hidden = layer.mlp(normalized) + layer.self_attn.q_proj(normalized)
        hidden.square().mean().backward()
        model.layers[1].mlp.up_proj.weight.grad = None
        local = suffix_gradient_coverage_local(
            model,
            depth=1,
            num_layers=len(model.layers),
        )
        aggregate = aggregate_suffix_gradient_coverage(
            [local],
            depth=1,
            num_layers=len(model.layers),
        )
        self.assertFalse(aggregate["hard_gate_passed"])
        self.assertFalse(aggregate["layers"]["1"]["complete"])
        self.assertGreater(
            aggregate["layers"]["1"]["missing_gradient_parameter_shard_elements"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
