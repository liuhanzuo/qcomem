from __future__ import annotations

import unittest

import torch
from torch import nn

from fp32_master import (
    aggregate_rank_audits,
    audit_adamw_fp32_state,
    audit_fp32_gradients,
    audit_fp32_parameter_delta,
    parameter_group,
    require_full_gradient_gate,
    require_parameter_delta_gate,
    snapshot_fp32_local_shards,
)


class TinyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)


class TinyTextModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.embed_tokens = nn.Embedding(8, 4)
        self.language_model.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.language_model.norm = nn.LayerNorm(4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.empty_local_shard = nn.Parameter(torch.empty(0))


def aggregate_delta(local):
    return {
        precision: aggregate_rank_audits([local[precision]])
        for precision in ("fp32_logical", "bf16_forward_visible")
    }


class FP32MasterAuditTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = TinyTextModel().float()
        self.snapshots = snapshot_fp32_local_shards(self.model.named_parameters())
        self.expected = sum(parameter.numel() for parameter in self.model.parameters())

    def test_group_names_and_empty_local_shard(self) -> None:
        groups = {snapshot.group for snapshot in self.snapshots}
        self.assertTrue(
            {"embedding", "layer.0", "layer.1", "final_norm", "lm_head"}
            <= groups
        )
        self.assertEqual(sum(s.parameter.numel() for s in self.snapshots), self.expected)
        self.assertEqual(parameter_group("wrapper.language_model.layers.12.x"), "layer.12")

    def test_gradient_optimizer_and_delta_gates(self) -> None:
        for snapshot in self.snapshots:
            snapshot.parameter.grad = torch.ones_like(snapshot.parameter)
        gradient = aggregate_rank_audits([audit_fp32_gradients(self.snapshots)])
        require_full_gradient_gate(
            gradient, expected_parameters=self.expected, expected_layers=2
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-3, weight_decay=0.0, foreach=False
        )
        optimizer.step()
        state = audit_adamw_fp32_state(optimizer, self.snapshots, expected_step=1)
        self.assertEqual(state["moment_elements"], 2 * self.expected)
        delta = aggregate_delta(audit_fp32_parameter_delta(self.snapshots))
        require_parameter_delta_gate(
            delta, expected_parameters=self.expected, expected_layers=2
        )
        self.assertGreater(delta["fp32_logical"]["total"]["nonzero_elements"], 0)

    def test_missing_and_nonfinite_gradients_fail(self) -> None:
        for snapshot in self.snapshots:
            snapshot.parameter.grad = torch.ones_like(snapshot.parameter)
        self.snapshots[0].parameter.grad = None
        missing = aggregate_rank_audits([audit_fp32_gradients(self.snapshots)])
        with self.assertRaisesRegex(RuntimeError, "missing or non-finite"):
            require_full_gradient_gate(
                missing, expected_parameters=self.expected, expected_layers=2
            )

        self.snapshots[0].parameter.grad = torch.full_like(
            self.snapshots[0].parameter, float("nan")
        )
        nonfinite = aggregate_rank_audits([audit_fp32_gradients(self.snapshots)])
        with self.assertRaisesRegex(RuntimeError, "missing or non-finite"):
            require_full_gradient_gate(
                nonfinite, expected_parameters=self.expected, expected_layers=2
            )

    def test_zero_layer_gradient_fails_layer_gate(self) -> None:
        for snapshot in self.snapshots:
            snapshot.parameter.grad = torch.ones_like(snapshot.parameter)
            if snapshot.group == "layer.1":
                snapshot.parameter.grad.zero_()
        audit = aggregate_rank_audits([audit_fp32_gradients(self.snapshots)])
        with self.assertRaisesRegex(RuntimeError, "layer.1 is entirely zero"):
            require_full_gradient_gate(
                audit, expected_parameters=self.expected, expected_layers=2
            )

    def test_sub_bf16_ulp_update_is_retained_by_fp32_master(self) -> None:
        parameter = nn.Parameter(torch.ones(16, dtype=torch.float32))
        snapshots = snapshot_fp32_local_shards([("language_model.layers.0.w", parameter)])
        parameter.grad = torch.ones_like(parameter)
        optimizer = torch.optim.AdamW(
            [parameter], lr=1e-6, weight_decay=0.0, foreach=False
        )
        optimizer.step()
        delta = aggregate_delta(audit_fp32_parameter_delta(snapshots))
        self.assertGreater(delta["fp32_logical"]["total"]["nonzero_elements"], 0)
        self.assertEqual(
            delta["bf16_forward_visible"]["total"]["nonzero_elements"], 0
        )


if __name__ == "__main__":
    unittest.main()
