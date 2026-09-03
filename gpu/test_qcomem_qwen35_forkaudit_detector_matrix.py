from __future__ import annotations

import inspect
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

import build_qcomem_qwen35_forkaudit_detector_matrix as builder
import run_qcomem_qwen35_forkaudit_detector_matrix as runner
from qcomem_vllm_paged_kernel import Q16KernelPagedTensorView


class _Backbone:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden
        self.config = types.SimpleNamespace(_attn_implementation="original")

    def __call__(self, **kwargs: object) -> object:
        assert kwargs["use_cache"] is True
        return types.SimpleNamespace(last_hidden_state=self.hidden)


class _Model:
    def __init__(self) -> None:
        self.lm_head = torch.nn.Identity()


class DetectorMatrixRegressionTest(unittest.TestCase):
    def test_live_capture_reaches_token_logit_and_fp32_sidecar(self) -> None:
        hidden = torch.tensor(
            [[[0.0, 1.0, 2.0], [4.0, -2.0, 7.0]]], dtype=torch.float32
        )
        runtime = types.SimpleNamespace(
            model=_Model(),
            backbone=_Backbone(hidden),
        )
        group = types.SimpleNamespace(requests=[object()])
        logits, token = runner._model_step(
            runtime,
            group,
            "debug-backend",
            0,
            torch.tensor([[1, 2]], dtype=torch.long),
        )
        self.assertEqual(token, 2)
        self.assertTrue(torch.equal(logits, torch.tensor([[4.0, -2.0, 7.0]])))
        self.assertEqual(runtime.backbone.config._attn_implementation, "original")
        self.assertEqual(
            runner.capture_live_last_logits.__module__,
            "run_qcomem_qwen35_vllm_paged_multifork_resident",
        )
        self.assertIn(
            "model.lm_head(output.last_hidden_state[:, -1, :])",
            inspect.getsource(runner.capture_live_last_logits),
        )
        with tempfile.TemporaryDirectory() as temp:
            rows = runner._sidecar(Path(temp), [logits])
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dtype"], "float32")
            self.assertEqual(rows[0]["shape"], [1, 3])
            self.assertEqual(rows[0]["bytes"], 12)
            self.assertEqual(
                runner.sha256_file(Path(temp) / "step-0.fp32.bin"),
                rows[0]["sha256"],
            )

    def test_m8_sentinel_is_mutant_ledger_only(self) -> None:
        clean_kernel = object()
        mutant = types.SimpleNamespace(kernel=clean_kernel)
        clean_peer = types.SimpleNamespace(kernel=clean_kernel)
        original, sentinel = runner._install_m8_sentinel(mutant)
        self.assertIs(original, clean_kernel)
        self.assertIs(mutant.kernel, sentinel)
        self.assertIs(clean_peer.kernel, clean_kernel)
        with self.assertRaisesRegex(AssertionError, "matrix M8 sentinel executed"):
            mutant.kernel()
        mutant.kernel = original
        self.assertIs(mutant.kernel, clean_kernel)
        scenario_source = inspect.getsource(runner._scenario)
        self.assertIn('elif mutant_id == "M8" and activate:', scenario_source)

    def test_m9_uses_dense_key_against_original_paged_value(self) -> None:
        arena = types.SimpleNamespace(
            key_cache=torch.arange(24, dtype=torch.float32).reshape(2, 2, 2, 3)
        )
        sequence = types.SimpleNamespace(
            arena=arena,
            active_block_table=torch.tensor([[0, 1]], dtype=torch.int64),
            sequence_length=4,
        )
        key_view = Q16KernelPagedTensorView(sequence, "key")
        value_view = Q16KernelPagedTensorView(sequence, "value")
        layer = types.SimpleNamespace(
            sequence=sequence,
            keys=key_view,
            values=value_view,
        )
        with mock.patch.object(runner.rr2, "FORMAL_NUM_KV_HEADS", 2), mock.patch.object(
            runner.rr2, "FORMAL_HEAD_DIM", 3
        ):
            original, dense_key, receipt = runner._materialize_m9_dense_key(layer)
        self.assertIs(original, key_view)
        self.assertIsInstance(dense_key, torch.Tensor)
        self.assertEqual(tuple(dense_key.shape), (1, 2, 4, 3))
        self.assertEqual(receipt["cache_slot_after"], "torch.Tensor(dense-key)")
        self.assertEqual(
            receipt["paired_value"], "Q16KernelPagedTensorView(value)"
        )
        ledger = object.__new__(runner.M9DenseKeyLedger)
        ledger.dense_key_bridge_count = 0
        ledger.last_dense_key_sha256 = None
        captured: dict[str, object] = {}

        def base_forward(
            _self: object,
            _module: object,
            _query: torch.Tensor,
            key: object,
            value: object,
            _mask: object,
            *_args: object,
            **_kwargs: object,
        ) -> str:
            captured["key"] = key
            captured["value"] = value
            return "reached-base-ledger"

        with mock.patch.object(runner.MultiForkHitLedger, "attention_forward", base_forward):
            result = ledger.attention_forward(
                object(), torch.zeros(1, 1, 1, 3), dense_key, value_view, None
            )
        self.assertEqual(result, "reached-base-ledger")
        self.assertIsInstance(captured["key"], runner.DenseKeySequenceBridge)
        self.assertIs(captured["key"].dense_key, dense_key)
        self.assertIs(captured["key"].sequence, sequence)
        self.assertIs(captured["value"], value_view)
        self.assertEqual(ledger.dense_key_bridge_count, 1)
        self.assertEqual(ledger.last_dense_key_sha256, receipt["dense_key_sha256"])

    @staticmethod
    def _completed_scenario() -> dict[str, object]:
        digest = "a" * 64
        return {
            "status": "completed",
            "tokens": [1],
            "full_logit_sha256": [digest],
            "logit_sidecars": [
                {
                    "relative_path": "step-0.fp32.bin",
                    "bytes": 16,
                    "sha256": digest,
                    "dtype": "float32",
                    "shape": [1, 4],
                }
            ],
        }

    def test_aggregate_rejects_clean_runtime_abort(self) -> None:
        completed = self._completed_scenario()
        row = {
            "mutant_id": "M8",
            "matched_clean": {"status": "runtime_abort"},
            "cross_arm_clean_reference": completed,
            "cross_n_clean_reference": completed,
            "mutant_target_gate_suppressed": {"status": "runtime_abort"},
        }
        with self.assertRaisesRegex(builder.BuildError, "matched clean did not complete"):
            builder.validate_candidate_campaign([row])

    def test_aggregate_rejects_zero_sidecar_clean(self) -> None:
        zero = {
            "status": "completed",
            "tokens": [],
            "full_logit_sha256": [],
            "logit_sidecars": [],
        }
        row = {
            "mutant_id": "M9",
            "matched_clean": zero,
            "cross_arm_clean_reference": zero,
            "cross_n_clean_reference": zero,
            "mutant_target_gate_suppressed": {"status": "runtime_abort"},
        }
        with self.assertRaisesRegex(
            builder.BuildError, "has no serialized token/logit sidecar"
        ):
            builder.validate_candidate_campaign([row])


if __name__ == "__main__":
    unittest.main()
