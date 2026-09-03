import unittest

import torch

from qcomem_palu_transfer import (
    PaluTransferError,
    apply_headwise,
    activation_whitener,
    headwise_svd_factors,
    logical_kv_storage,
    relative_l2,
    truncate_factors,
    whitened_headwise_svd_factors,
)


class PaluTransferTests(unittest.TestCase):
    def test_full_rank_reconstructs_two_heads(self):
        torch.manual_seed(3)
        weight = torch.randn(8, 12)
        hidden = torch.randn(2, 5, 12)
        factors = headwise_svd_factors(weight, heads=2)
        truncated = truncate_factors(factors, rank=4, dtype=torch.float32)
        candidate = apply_headwise(hidden, truncated, bias=None)
        reference = torch.nn.functional.linear(hidden, weight)
        self.assertLess(relative_l2(candidate, reference), 1e-5)

    def test_rank_error_is_monotone_for_weight_reconstruction(self):
        torch.manual_seed(5)
        weight = torch.randn(8, 12)
        hidden = torch.eye(12).reshape(1, 12, 12)
        factors = headwise_svd_factors(weight, heads=2)
        errors = []
        for rank in (1, 2, 3):
            candidate = apply_headwise(
                hidden,
                truncate_factors(factors, rank=rank, dtype=torch.float32),
                bias=None,
            )
            errors.append(relative_l2(candidate, torch.nn.functional.linear(hidden, weight)))
        self.assertGreater(errors[0], errors[1])
        self.assertGreater(errors[1], errors[2])

    def test_storage_and_invalid_rank(self):
        self.assertEqual(logical_kv_storage(rank=64)["dense_over_palu_ratio"], 4.0)
        self.assertEqual(logical_kv_storage(rank=128)["dense_over_palu_ratio"], 2.0)
        factors = headwise_svd_factors(torch.randn(8, 12), heads=2)
        with self.assertRaises(PaluTransferError):
            truncate_factors(factors, rank=5, dtype=torch.float32)

    def test_whitened_full_rank_reconstructs(self):
        torch.manual_seed(11)
        weight = torch.randn(8, 12)
        calibration = torch.randn(1, 24, 12)
        scale, receipt = activation_whitener(calibration)
        factors = whitened_headwise_svd_factors(weight, heads=2, scale=scale)
        candidate = apply_headwise(
            calibration,
            truncate_factors(factors, rank=4, dtype=torch.float32),
            bias=None,
        )
        reference = torch.nn.functional.linear(calibration, weight)
        self.assertLess(relative_l2(candidate, reference), 1e-4)
        self.assertEqual(receipt["calibration_tokens"], 24)


if __name__ == "__main__":
    unittest.main()
