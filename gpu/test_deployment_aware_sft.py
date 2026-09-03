from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from build_deployment_aware_sft import BuildError, _chat_render, _fair_labels, _schedule
from deployment_aware_sft import (
    HELDOUT_COUNTS,
    IGNORE_INDEX,
    TRAIN_COUNTS,
    DeploymentAwareCausalLM,
    DeploymentExample,
    frozen_teacher_targets,
    log1mexp,
    schedule_audit,
    stable_json,
    summarize_example_equal,
    validate_example_row,
)


class TinyLanguageModel(nn.Module):
    def __init__(self, vocab: int = 17, hidden: int = 8) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        hidden = torch.tanh(self.proj(self.embed_tokens(input_ids)))
        return SimpleNamespace(last_hidden_state=hidden)


class DeploymentAwareSFTTest(unittest.TestCase):
    def test_chat_render_accepts_batch_encoding_mapping_and_tensor(self) -> None:
        class MappingTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        class BatchEncodingLike(dict):
            pass

        class EncodingTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return BatchEncodingLike(input_ids=torch.tensor([[4, 5, 6]]))

        self.assertEqual(_chat_render(MappingTokenizer(), []), [1, 2, 3])
        self.assertEqual(_chat_render(EncodingTokenizer(), []), [4, 5, 6])

    def test_chat_render_rejects_missing_or_invalid_input_ids(self) -> None:
        class MissingTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return {"attention_mask": [1]}

        class InvalidTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return ["input_ids"]

        with self.assertRaisesRegex(BuildError, "no input_ids"):
            _chat_render(MissingTokenizer(), [])
        with self.assertRaisesRegex(BuildError, "integer token IDs"):
            _chat_render(InvalidTokenizer(), [])

    def test_log1mexp_matches_probability(self) -> None:
        values = torch.tensor([0.1, 0.4, 0.8]).log()
        result = log1mexp(values).exp()
        self.assertTrue(torch.allclose(result, 1.0 - values.exp(), atol=1e-6))

    def test_teacher_objective_matches_frozen_teacher_at_initialization(self) -> None:
        torch.manual_seed(3)
        language_model = TinyLanguageModel()
        head = nn.Linear(8, 17, bias=False)
        core = DeploymentAwareCausalLM(language_model, head)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 4, 5]])
        teacher = frozen_teacher_targets(
            language_model,
            head,
            input_ids,
            labels,
            torch.ones_like(input_ids),
            topk=5,
            projection_chunk_tokens=1,
        )
        loss, ce, kl, hidden = core(
            input_ids,
            labels,
            torch.ones_like(input_ids),
            teacher_topk_ids=teacher["topk_ids"],
            teacher_topk_logprobs=teacher["topk_logprobs"],
            teacher_tail_logprob=teacher["tail_logprob"],
            teacher_normalized_hidden=teacher["normalized_hidden"],
        )
        self.assertAlmostEqual(float(kl), 0.0, places=5)
        self.assertAlmostEqual(float(hidden), 0.0, places=3)
        self.assertAlmostEqual(float(loss.detach()), 0.45 * float(ce), places=3)
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in core.parameters()))

    def test_non_teacher_path_is_hard_ce(self) -> None:
        torch.manual_seed(5)
        core = DeploymentAwareCausalLM(TinyLanguageModel(), nn.Linear(8, 17, bias=False))
        input_ids = torch.tensor([[1, 2, 3, 4]])
        labels = torch.tensor([[IGNORE_INDEX, IGNORE_INDEX, 3, 4]])
        loss, ce, kl, hidden = core(input_ids, labels, torch.ones_like(input_ids))
        self.assertEqual(float(loss.detach()), float(ce))
        self.assertEqual(float(kl), 0.0)
        self.assertEqual(float(hidden), 0.0)

    def test_example_validation_rejects_non_suffix_labels(self) -> None:
        row = {
            "schema_version": "qcomem-deployment-aware-example-v1",
            "example_id": "0" * 64,
            "dataset": "qasper",
            "stratum": "domain",
            "source_split": "train",
            "source_id_sha256": "1" * 64,
            "document_id_sha256": "2" * 64,
            "prompt_sha256": "3" * 64,
            "context_sha256": "4" * 64,
            "input_ids": [1, 2, 3, 4],
            "labels": [-100, 2, -100, 4],
            "token_counts": {"prompt": 1, "target": 3, "total": 4},
            "teacher_target_required": False,
            "schedule_index": 0,
            "deployment_boundary": {
                "applicable": True,
                "document_input_ids": [1],
                "query_input_ids": [],
            },
        }
        with self.assertRaisesRegex(ValueError, "target suffix"):
            validate_example_row(row, split="train", max_sequence_tokens=4096, row_index=0)

    def test_fair_schedule_has_exact_40_30_30_mix(self) -> None:
        labels = _fair_labels(TRAIN_COUNTS)
        self.assertEqual(len(labels), 1024)
        self.assertEqual(Counter(labels), Counter(TRAIN_COUNTS))
        # No prefix should diverge from its requested fractional quota by more
        # than one example after the first complete 8-way global step.
        for size in range(8, len(labels) + 1):
            counts = Counter(labels[:size])
            for stratum, total in TRAIN_COUNTS.items():
                expected = size * total / 1024
                self.assertLessEqual(abs(counts[stratum] - expected), 1.0)

    def test_summary_is_example_equal_not_target_weighted(self) -> None:
        rows = [
            {"ce": 1.0, "target_tokens": 1, "stratum": "domain", "dataset": "qasper"},
            {"ce": 3.0, "target_tokens": 100, "stratum": "domain", "dataset": "qasper"},
            {"ce": 2.0, "target_tokens": 1, "stratum": "general_replay", "dataset": "tulu3_persona_if"},
            {"ce": 2.0, "target_tokens": 1, "stratum": "teacher_preservation", "dataset": "tulu3_persona_if"},
        ]
        summary = summarize_example_equal(rows)
        self.assertEqual(summary["overall"]["example_equal_mean_ce"], 2.0)
        self.assertFalse(summary["target_token_weighting_used"])

    def test_schedule_audit_requires_rank_divisibility(self) -> None:
        example = DeploymentExample(
            input_ids=torch.tensor([1, 2]),
            labels=torch.tensor([-100, 2]),
            example_id="0" * 64,
            dataset="qasper",
            stratum="domain",
            source_id_sha256="1" * 64,
            document_id_sha256="2" * 64,
            prompt_sha256="3" * 64,
            context_sha256="4" * 64,
            schedule_index=0,
        )
        with self.assertRaisesRegex(ValueError, "divide exactly"):
            schedule_audit([example] * 7)

    def test_frozen_dataset_sizes(self) -> None:
        self.assertEqual(sum(TRAIN_COUNTS.values()), 1024)
        self.assertEqual(sum(HELDOUT_COUNTS.values()), 64)


if __name__ == "__main__":
    unittest.main()
