from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import math
import unittest

import torch

from qcomem_forkaudit_oracle import (
    DEFAULT_MAX_RELATIVE_L2,
    OracleGateError,
    OraclePreregistration,
    OracleThresholds,
    ThresholdLockedError,
    fp32_dense_attention_reference,
    logit_metrics,
    tensor_error_metrics,
)


class DenseReferenceTest(unittest.TestCase):
    def test_reference_forces_ieee_fp32_and_restores_process_state(self):
        original_precision = torch.get_float32_matmul_precision()
        cuda_matmul = torch.backends.cuda.matmul
        uses_new_cuda_control = hasattr(cuda_matmul, "fp32_precision")
        original_cuda = (
            str(cuda_matmul.fp32_precision)
            if uses_new_cuda_control
            else bool(cuda_matmul.allow_tf32)
        )
        original_cudnn = bool(torch.backends.cudnn.allow_tf32)
        try:
            torch.set_float32_matmul_precision("medium")
            query = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
            key = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
            result = fp32_dense_attention_reference(query, key, key.clone())

            audit = result.precision_audit
            self.assertEqual(
                audit["effective"]["float32_matmul_precision"], "highest"
            )
            self.assertEqual(audit["device_type"], "cpu")
            self.assertFalse(audit["applies_to_cuda_reference"])
            self.assertTrue(audit["restored"])
            self.assertEqual(audit["before"], audit["after"])
            self.assertEqual(torch.get_float32_matmul_precision(), "medium")
        finally:
            torch.set_float32_matmul_precision(original_precision)
            if uses_new_cuda_control:
                cuda_matmul.fp32_precision = original_cuda
            else:
                cuda_matmul.allow_tf32 = original_cuda
            torch.backends.cudnn.allow_tf32 = original_cudnn

    def test_zero_query_is_uniform_over_causal_prefix_and_repeats_kv_heads(self):
        query = torch.zeros((1, 2, 2, 2), dtype=torch.bfloat16)
        key = torch.zeros((1, 1, 3, 2), dtype=torch.bfloat16)
        value = torch.tensor(
            [[[[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]]],
            dtype=torch.bfloat16,
        )

        result = fp32_dense_attention_reference(query, key, value)

        expected = torch.tensor(
            [[[[4.0, 6.0], [4.0, 6.0]], [[6.0, 8.0], [6.0, 8.0]]]],
            dtype=torch.float32,
        )
        self.assertEqual(tuple(result.output.shape), (1, 2, 2, 2))
        self.assertEqual(result.output.dtype, torch.float32)
        self.assertTrue(torch.equal(result.output, expected))
        self.assertEqual(result.grouped_query_factor, 2)
        self.assertEqual(result.position_contract["mode"], "cached_suffix_causal")
        self.assertTrue(result.precision_audit["restored"])

    def test_boolean_visibility_mask_is_additional_restriction(self):
        query = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
        key = torch.zeros((1, 1, 3, 1), dtype=torch.float32)
        value = torch.tensor([[[[1.0], [3.0], [101.0]]]])
        mask = torch.tensor([[[[True, True, False]]]])

        result = fp32_dense_attention_reference(
            query, key, value, visibility_mask=mask
        )

        self.assertEqual(result.output.item(), 2.0)
        self.assertTrue(result.position_contract["additional_visibility_mask"])

    def test_position_off_by_one_is_rejected(self):
        query = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        key = torch.zeros((1, 1, 4, 2), dtype=torch.float32)
        value = key.clone()

        with self.assertRaisesRegex(OracleGateError, "ORACLE_POSITION_CONTRACT"):
            fp32_dense_attention_reference(
                query,
                key,
                value,
                key_positions=torch.tensor([10, 11, 12, 13]),
                query_positions=torch.tensor([11, 12]),
            )

    def test_noncontiguous_positions_are_rejected(self):
        query = torch.zeros((1, 1, 1, 2), dtype=torch.float32)
        key = torch.zeros((1, 1, 3, 2), dtype=torch.float32)

        with self.assertRaisesRegex(OracleGateError, "unit stride"):
            fp32_dense_attention_reference(
                query,
                key,
                key.clone(),
                key_positions=torch.tensor([0, 2, 3]),
                query_positions=torch.tensor([3]),
            )

    def test_shape_dtype_and_finite_gates(self):
        good = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
        with self.assertRaisesRegex(OracleGateError, "ORACLE_INPUT_SHAPE"):
            fp32_dense_attention_reference(good.squeeze(0), good, good)
        with self.assertRaisesRegex(OracleGateError, "ORACLE_INPUT_DTYPE"):
            fp32_dense_attention_reference(good.long(), good.long(), good.long())
        bad = good.clone()
        bad[0, 0, 0, 0] = math.nan
        with self.assertRaisesRegex(OracleGateError, "ORACLE_INPUT_FINITE"):
            fp32_dense_attention_reference(bad, good, good)


class MetricTest(unittest.TestCase):
    def test_attention_error_metrics(self):
        reference = torch.tensor([3.0, 4.0])
        candidate = torch.tensor([0.0, 4.0])

        metrics = tensor_error_metrics(reference, candidate)

        self.assertAlmostEqual(metrics.max_abs, 3.0)
        self.assertAlmostEqual(metrics.mean_abs, 1.5)
        self.assertAlmostEqual(metrics.relative_l2, 0.6)
        self.assertFalse(metrics.bitwise_exact)
        self.assertTrue(metrics.finite)

    def test_zero_reference_relative_l2_is_well_defined(self):
        zeros = torch.zeros(2)
        self.assertEqual(tensor_error_metrics(zeros, zeros).relative_l2, 0.0)
        nonzero = tensor_error_metrics(zeros, torch.ones(2)).relative_l2
        self.assertTrue(math.isfinite(nonzero))
        self.assertGreater(nonzero, 1e20)

    def test_logit_top1_and_forward_kl(self):
        reference = torch.tensor([[3.0, 1.0], [0.0, 2.0]])
        candidate = torch.tensor([[2.0, 1.0], [3.0, 0.0]])

        metrics = logit_metrics(reference, candidate)

        self.assertFalse(metrics.top1_match)
        self.assertEqual(metrics.top1_agreement, 0.5)
        self.assertGreater(metrics.mean_forward_kl, 0.0)
        self.assertGreaterEqual(metrics.max_forward_kl, metrics.mean_forward_kl)


class PreregisteredOutcomeTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        query = torch.tensor([[[[1.0, 0.0]]]], dtype=torch.float32)
        key = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float32)
        value = torch.tensor([[[[2.0, 0.0], [0.0, 4.0]]]], dtype=torch.float32)
        return query, key, value

    def test_default_threshold_success_and_json_outcome(self):
        query, key, value = self._inputs()
        reference = fp32_dense_attention_reference(query, key, value)
        protocol = OraclePreregistration()

        outcome = protocol.evaluate_attention(query, key, value, reference.output)
        payload = outcome.to_dict()

        self.assertTrue(outcome.passed)
        self.assertEqual(outcome.status, "completed")
        self.assertEqual(
            outcome.thresholds["max_relative_l2"], DEFAULT_MAX_RELATIVE_L2
        )
        self.assertEqual(len(outcome.preregistration_sha256), 64)
        self.assertEqual(outcome.evaluation_index, 1)
        self.assertTrue(protocol.locked)
        self.assertTrue(payload["reference"]["precision_audit"]["restored"])
        json.dumps(payload, allow_nan=False)

    def test_relative_l2_failure_is_completed_not_invalid(self):
        query, key, value = self._inputs()
        reference = fp32_dense_attention_reference(query, key, value)
        candidate = reference.output + 1.0
        protocol = OraclePreregistration(OracleThresholds(max_relative_l2=1e-8))

        outcome = protocol.evaluate_attention(query, key, value, candidate)

        self.assertEqual(outcome.status, "completed")
        self.assertFalse(outcome.passed)
        gates = {gate.gate_id: gate for gate in outcome.gates}
        self.assertFalse(gates["ORACLE_RELATIVE_L2"].passed)

    def test_invalid_candidate_returns_named_structured_failure(self):
        query, key, value = self._inputs()
        protocol = OraclePreregistration()
        candidate = torch.zeros((1, 1), dtype=torch.float32)

        outcome = protocol.evaluate_attention(query, key, value, candidate)

        self.assertEqual(outcome.status, "invalid")
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.failure["gate_id"], "ORACLE_CANDIDATE_SHAPE")

    def test_optional_logits_are_measured_and_can_gate(self):
        query, key, value = self._inputs()
        reference = fp32_dense_attention_reference(query, key, value)
        protocol = OraclePreregistration(
            OracleThresholds(require_logits_top1_match=True)
        )

        outcome = protocol.evaluate_attention(
            query,
            key,
            value,
            reference.output,
            reference_logits=torch.tensor([[4.0, 1.0]]),
            candidate_logits=torch.tensor([[1.0, 4.0]]),
        )

        self.assertFalse(outcome.passed)
        self.assertIsNotNone(outcome.logits_metrics)
        self.assertFalse(outcome.logits_metrics.top1_match)

    def test_thresholds_lock_at_first_attempt_and_hash_is_stable(self):
        query, key, value = self._inputs()
        reference = fp32_dense_attention_reference(query, key, value)
        protocol = OraclePreregistration()
        protocol.configure(max_relative_l2=0.004)
        preregistered_hash = protocol.sha256

        first = protocol.evaluate_attention(query, key, value, reference.output)
        second = protocol.evaluate_attention(query, key, value, reference.output)

        self.assertEqual(first.preregistration_sha256, preregistered_hash)
        self.assertEqual(second.preregistration_sha256, preregistered_hash)
        self.assertEqual(second.evaluation_index, 2)
        with self.assertRaises(ThresholdLockedError):
            protocol.configure(max_relative_l2=0.5)

    def test_threshold_dataclass_is_immutable(self):
        thresholds = OracleThresholds()
        with self.assertRaises(FrozenInstanceError):
            thresholds.max_relative_l2 = 1.0


if __name__ == "__main__":
    unittest.main()
