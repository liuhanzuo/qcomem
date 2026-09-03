from __future__ import annotations

import copy
import math
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

import qcomem_answer_supervised_lora as answer_core
from deployment_aware_sft import DeploymentExample
from qcomem_answer_supervised_lora import (
    DEPTH,
    EXPECTED_ADAPTER_MODULES,
    AnswerLoRAContractError,
    AnswerBoundary,
    answer_decode_semantic_diagnostic,
    answer_boundary,
    answer_preservation_objective,
    audit_adapter_surface,
    balanced_domain_schedule,
    choose_best_checkpoint,
    evaluate_step1_gate,
    evaluate_step2_gate,
    example_balance_weights,
    install_and_audit_adapters,
    reject_longbench_path_or_digest,
)
from qcomem_lora import ReplayQuantConfig
from train_answer_supervised_native_lora import CODE_FILES as TRAINER_CODE_FILES


def example(
    index: int,
    *,
    dataset: str,
    stratum: str = "domain",
    sequence_tokens: int = 8,
) -> DeploymentExample:
    input_ids = torch.arange(sequence_tokens, dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    labels[-2:] = input_ids[-2:]
    return DeploymentExample(
        input_ids=input_ids,
        labels=labels,
        example_id=f"{dataset}-{index:04d}",
        dataset=dataset,
        stratum=stratum,
        source_id_sha256="1" * 64,
        document_id_sha256="2" * 64,
        prompt_sha256="3" * 64,
        context_sha256="4" * 64,
        schedule_index=index,
    )


class FullAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 8, bias=False)
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)
        self.o_proj = nn.Linear(8, 4, bias=False)


class GDN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_proj_qkv = nn.Linear(4, 8, bias=False)
        self.in_proj_z = nn.Linear(4, 4, bias=False)
        self.in_proj_b = nn.Linear(4, 2, bias=False)
        self.in_proj_a = nn.Linear(4, 2, bias=False)
        self.out_proj = nn.Linear(4, 4, bias=False)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(4, 4, bias=False)
        self.up_proj = nn.Linear(4, 4, bias=False)
        self.down_proj = nn.Linear(4, 4, bias=False)


class Layer(nn.Module):
    def __init__(self, kind: str) -> None:
        super().__init__()
        self.self_attn = FullAttention() if kind == "full_attention" else GDN()
        self.mlp = MLP()


class MockModel(nn.Module):
    def __init__(self, layer_types: list[str]) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Layer(kind) for kind in layer_types])


class MockSplit:
    def __init__(self, model: MockModel, layer_types: list[str]) -> None:
        self.layers = model.layers
        self.config = type("Config", (), {"layer_types": layer_types})()


class TinyLocal:
    def __init__(self, document: torch.Tensor) -> None:
        self.document_residual = document


class TinyPacked:
    def __init__(self, document: torch.Tensor) -> None:
        self.document = document

    def fork(self) -> TinyLocal:
        return TinyLocal(self.document.clone())


class TinyRaw:
    def __init__(self, document: torch.Tensor) -> None:
        self.document = document

    def quantize(self, **kwargs) -> TinyPacked:
        del kwargs
        return TinyPacked(self.document)


class TinyDiagnosticAdapter:
    def __init__(self) -> None:
        self.num_layers = 2
        self.config = SimpleNamespace()
        self.language_model = SimpleNamespace(norm=nn.Identity())
        self.lm_head = nn.Linear(2, 7, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(
                torch.arange(14, dtype=torch.float32).reshape(7, 2) / 13
            )

    @staticmethod
    def encode(tokens: torch.Tensor) -> torch.Tensor:
        values = tokens.reshape(1, -1, 1).float()
        return torch.cat((values, values.square() / 10), dim=-1)

    def write_lower_replay(self, tokens: torch.Tensor, depth: int) -> TinyRaw:
        del depth
        return TinyRaw(self.encode(tokens))

    def continue_lower_replay(self, local: TinyLocal, tokens: torch.Tensor) -> torch.Tensor:
        del local
        return self.encode(tokens)

    @staticmethod
    def make_cache() -> SimpleNamespace:
        return SimpleNamespace()

    @staticmethod
    def _run_layers(
        hidden: torch.Tensor,
        start: int,
        end: int,
        *,
        past_key_values=None,
        position_offset: int = 0,
    ) -> torch.Tensor:
        del start, end, past_key_values, position_offset
        return hidden


class AnswerSupervisedNativeLoRATest(unittest.TestCase):
    def test_causal_answer_shift_and_exact_boundary(self) -> None:
        row = example(0, dataset="qasper", sequence_tokens=6)
        row = DeploymentExample(
            **{
                **row.__dict__,
                "input_ids": torch.tensor([1, 2, 3, 4, 5, 6]),
                "labels": torch.tensor([-100, -100, -100, -100, 5, 6]),
            }
        )
        boundary = answer_boundary(
            row,
            raw_row={
                "deployment_boundary": {
                    "applicable": True,
                    "document_input_ids": [1, 2],
                    "query_input_ids": [3, 4],
                }
            },
        )
        self.assertEqual(boundary.document_ids.tolist(), [1, 2])
        self.assertEqual(boundary.query_ids.tolist(), [3, 4])
        self.assertEqual(boundary.continuation_input_ids.tolist(), [3, 4, 5])
        self.assertEqual(boundary.target_ids.tolist(), [5, 6])
        # Last query hidden predicts answer[0]; answer[:-1] hidden predicts the
        # remaining targets, including EOS.
        predictors = boundary.continuation_input_ids[-boundary.answer_positions :]
        self.assertEqual(predictors.tolist(), [4, 5])

    def test_non_domain_boundary_fails_closed(self) -> None:
        row = example(0, dataset="tulu3_persona_if", stratum="general_replay")
        with self.assertRaisesRegex(AnswerLoRAContractError, "no unambiguous"):
            answer_boundary(row, raw_row={"deployment_boundary": {"applicable": False}})

    def test_real_scale_schedule_is_four_plus_four_and_quota_balanced(self) -> None:
        rows = [
            example(i, dataset="qasper", sequence_tokens=4096 if i == 0 else 10)
            for i in range(256)
        ] + [example(i, dataset="2wikimqa", sequence_tokens=20) for i in range(154)]
        schedule, audit = balanced_domain_schedule(
            rows, steps=128, world_size=8, seed=20260814
        )
        by_id = {row.example_id: row for row in rows}
        for start in range(0, len(schedule), 8):
            counts = Counter(by_id[item].dataset for item in schedule[start : start + 8])
            self.assertEqual(counts, {"qasper": 4, "2wikimqa": 4})
        qasper = Counter(item for item in schedule if by_id[item].dataset == "qasper")
        twowiki = Counter(item for item in schedule if by_id[item].dataset == "2wikimqa")
        self.assertEqual(Counter(qasper.values()), {2: 256})
        self.assertEqual(Counter(twowiki.values()), {3: 104, 4: 50})
        self.assertEqual(
            max(by_id[item].sequence_tokens for item in schedule[:8]), 4096
        )
        self.assertTrue(all(audit["checks"].values()))

    def test_heldout_weights_are_task_equal_not_token_equal(self) -> None:
        rows = [example(i, dataset="qasper") for i in range(12)] + [
            example(i, dataset="2wikimqa") for i in range(14)
        ]
        weights, audit = example_balance_weights(rows)
        for dataset in ("qasper", "2wikimqa"):
            mass = sum(weights[row.example_id] for row in rows if row.dataset == dataset)
            self.assertAlmostEqual(mass / len(rows), 0.5)
        self.assertFalse(audit["target_token_weighting_used"])

    def test_adapter_surface_covers_full_attention_and_gdn_not_mlp(self) -> None:
        layer_types = [
            "full_attention" if index % 4 == 3 else "linear_attention"
            for index in range(40)
        ]
        model = MockModel(layer_types)
        installed, audit = install_and_audit_adapters(
            model,
            MockSplit(model, layer_types),
            rank=2,
            alpha=4.0,
            dropout=0.0,
            initialization_seed=11,
        )
        self.assertEqual(len(installed), EXPECTED_ADAPTER_MODULES)
        self.assertEqual(audit["module_counts"], {"full_attention": 36, "gdn": 120})
        self.assertEqual(audit["mlp"]["installed_modules"], 0)
        self.assertTrue(all(audit["checks"].values()))

    def test_chunked_projection_matches_single_chunk_loss_and_gradients(self) -> None:
        torch.manual_seed(7)
        positions, width, vocab, topk = 11, 6, 23, 5
        raw_a = torch.randn(positions, width)
        raw_b = raw_a.clone()
        adapter_a = nn.Linear(width, width, bias=False)
        adapter_b = copy.deepcopy(adapter_a)
        head_a = nn.Linear(width, vocab, bias=False)
        head_b = copy.deepcopy(head_a)
        targets = torch.randint(vocab, (positions,))
        teacher_logits = torch.randn(positions, vocab)
        teacher_logp = teacher_logits.log_softmax(-1)
        teacher_values, teacher_ids = teacher_logp.topk(topk, dim=-1)
        teacher_tail = torch.log1p(-teacher_values.exp().sum(-1).clamp(max=1 - 1e-7))
        teacher = {
            "target_ids": targets,
            "topk_ids": teacher_ids.to(torch.int32),
            "topk_logprobs": teacher_values,
            "tail_logprob": teacher_tail,
            "normalized_hidden": torch.nn.functional.normalize(
                torch.randn(positions, width), dim=-1
            ),
        }
        full = answer_preservation_objective(
            adapter_a(raw_a),
            head_a,
            targets,
            teacher,
            hard_weight=0.45,
            kl_weight=0.35,
            hidden_weight=0.20,
            projection_chunk_positions=positions,
        )
        with mock.patch(
            "qcomem_answer_supervised_lora.checkpoint",
            wraps=answer_core.checkpoint,
        ) as projection_checkpoint:
            chunked = answer_preservation_objective(
                adapter_b(raw_b),
                head_b,
                targets,
                teacher,
                hard_weight=0.45,
                kl_weight=0.35,
                hidden_weight=0.20,
                projection_chunk_positions=3,
            )
        self.assertEqual(projection_checkpoint.call_count, 4)
        for key in ("loss", "ce", "kl", "hidden"):
            self.assertTrue(torch.allclose(full[key], chunked[key], atol=2e-6, rtol=2e-6))
        full["loss"].backward()
        chunked["loss"].backward()
        self.assertTrue(
            torch.allclose(adapter_a.weight.grad, adapter_b.weight.grad, atol=2e-6, rtol=2e-6)
        )
        self.assertTrue(
            torch.allclose(head_a.weight.grad, head_b.weight.grad, atol=2e-6, rtol=2e-6)
        )

    def test_decode_semantic_diagnostic_whole_vs_incremental_end_to_end(self) -> None:
        adapter = TinyDiagnosticAdapter()
        boundary = AnswerBoundary(
            document_ids=torch.tensor([1, 2]),
            query_ids=torch.tensor([3, 4]),
            answer_ids=torch.tensor([5, 6]),
            continuation_input_ids=torch.tensor([3, 4, 5]),
            target_ids=torch.tensor([5, 6]),
            kind="frozen_exact_domain_document_query",
        )
        quant = ReplayQuantConfig(
            residual_bits=4,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=(8, 8),
            group_size=2,
        )
        with mock.patch(
            "qcomem_answer_supervised_lora.TorchSplitCausalLM",
            return_value=adapter,
        ), mock.patch(
            "qcomem_qwen35_native_cache.install_native_functional_linear_cache",
            return_value=SimpleNamespace(),
        ):
            result = answer_decode_semantic_diagnostic(
                nn.Identity(),
                boundary,
                quant=quant,
                depth=1,
                projection_chunk_positions=1,
            )
        self.assertEqual(result["positions"], 2)
        self.assertEqual(result["top1_equal_positions"], 2)
        self.assertEqual(result["top1_agreement"], 1.0)
        self.assertAlmostEqual(result["mean_kl_whole_to_token"], 0.0, places=7)
        self.assertIsNone(result["first_top1_divergence_position"])
        self.assertFalse(result["equivalence_claimed"])

    def test_launcher_final_audit_requires_all_formal_fields(self) -> None:
        text = Path(__file__).with_name(
            "launch_answer_supervised_native_lora_8gpu.sh"
        ).read_text()
        for field in (
            'm["last_step"]==128',
            'm["adapter_memory"]["trainable_parameters"]==26689536',
            'g1["checks"]["memory_headroom"]',
            'g2["checks"]["all_adapter_gradients_finite_nonzero"]',
            'd["examples"]==26',
            'd["equivalence_claimed"] is False',
            'b["validation_6_35_used_for_selection"] is False',
            'b["test_v2_used"] is False',
            'm["adapter_config"]["installed_module_count"]==156',
            'm["adapter_config"]["parameter_tensor_count"]==312',
            'm["initialization_attribution"]["pure_cold_start_experiment"] is False',
            'stages/02_training_complete',
            'FAILED_${CURRENT_PHASE}',
            'answer-full-state-downstream-analysis.json',
            'r["claim_boundaries"]["validation_may_select_checkpoint_or_policy"] is False',
            'p["samples"]==60 and len(p["paired_bootstrap_95_ci"])==2',
            'timeout --signal=TERM --kill-after=60s 14400s',
            'sha256sum -c "$RUN_DIR/code.sha256"',
            'export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache"',
        ):
            self.assertIn(field, text)

    def test_trainer_and_launcher_freeze_the_same_sixteen_code_files(self) -> None:
        expected = {
            "qcomem_answer_supervised_lora.py",
            "train_answer_supervised_native_lora.py",
            "test_answer_supervised_native_lora.py",
            "run_answer_lora_full_state_downstream.py",
            "aggregate_answer_lora_full_state_downstream.py",
            "test_answer_lora_full_state_downstream.py",
            "launch_answer_supervised_native_lora_8gpu.sh",
            "lora_answer_supervised_native_128.json",
            "deployment_aware_sft.py",
            "qcomem_lora.py",
            "qcomem_torch.py",
            "qcomem_qwen35_native_cache.py",
            "supervised_sft.py",
            "run_downstream.py",
            "run_replay_diagnostic.py",
            "analyze_validation.py",
        }
        self.assertEqual(TRAINER_CODE_FILES, expected)
        launcher = Path(__file__).with_name(
            "launch_answer_supervised_native_lora_8gpu.sh"
        ).read_text()
        launcher_code_files = launcher.split("CODE_FILES=(", 1)[1].split("\n)", 1)[0]
        for name in expected - {"lora_answer_supervised_native_128.json"}:
            self.assertEqual(launcher_code_files.count(name), 1, name)
        self.assertEqual(launcher_code_files.count('"$CONFIG_FILE"'), 1)
        self.assertEqual(
            launcher.count('> "$RUN_DIR/logs/train.log" 2>&1'), 1
        )
        self.assertEqual(
            launcher.count('test -s "$RUN_DIR/logs/train.log"'), 1
        )
        self.assertEqual(
            launcher.count('sha256sum -c "$RUN_DIR/model-weights.sha256"'), 2
        )

    def gate_row(self, rank: int, *, step: int) -> dict:
        nonzero = 192 if step == 1 else 312
        return {
            "rank": rank,
            "gradient_tensors": 312,
            "finite_gradient_tensors": 312,
            "nonzero_gradient_tensors": nonzero,
            "finite_update_tensors": 312,
            "nonzero_update_tensors": nonzero,
            "full_attention": {
                "gradient_tensors": 72,
                "nonzero_gradient_tensors": 72,
                "nonzero_update_tensors": 72,
            },
            "gdn_lora_a": {
                "gradient_tensors": 120,
                "finite_gradient_tensors": 120,
                "nonzero_gradient_tensors": 0 if step == 1 else 120,
                "nonzero_update_tensors": 0 if step == 1 else 120,
            },
            "gdn_lora_b": {
                "gradient_tensors": 120,
                "nonzero_gradient_tensors": 120,
                "nonzero_update_tensors": 120,
            },
            "cache": {
                "execution": "native-functional-cache",
                "hard_gate_passed": True,
                "original_cache_versions_unchanged": True,
                "all_cache_paths_rebound": True,
                "query_positions_observed": 4,
            },
            "continuation_positions": 4,
            "optimizer_fp32": {"passed": True},
            "reserved_headroom_bytes": 5 * 1024**3,
            "finite_loss": True,
        }

    def test_step1_cold_start_and_step2_all_tensor_gates(self) -> None:
        step1 = evaluate_step1_gate(
            [self.gate_row(rank, step=1) for rank in range(8)],
            minimum_headroom_bytes=4 * 1024**3,
        )
        step2 = evaluate_step2_gate(
            [self.gate_row(rank, step=2) for rank in range(8)]
        )
        self.assertEqual(step1["status"], "passed")
        self.assertEqual(step2["status"], "passed")

    def test_selection_uses_exactly_zero_sixty_four_one_twenty_eight(self) -> None:
        values = {0: 1.2, 64: 0.7, 128: 0.8}
        evaluations = {
            step: {"summary": {"overall": {"loss": loss}}}
            for step, loss in values.items()
        }
        self.assertEqual(choose_best_checkpoint(evaluations), 64)

    def test_longbench_path_is_rejected_before_read(self) -> None:
        with self.assertRaisesRegex(AnswerLoRAContractError, "official-train"):
            reject_longbench_path_or_digest(
                Path("/does/not/exist/longbench_validation.jsonl"), "0" * 64
            )


if __name__ == "__main__":
    unittest.main()
