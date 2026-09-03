#!/usr/bin/env python3

import unittest
from fractions import Fraction

import run_comparison as rc


class ComparatorMathTests(unittest.TestCase):
    def test_hypergeometric_normalizes(self):
        for N in (4, 8, 32):
            for K in range(N + 1):
                for k in range(N + 1):
                    self.assertEqual(
                        sum(rc.hyper_pmf_fraction(N, K, k, x) for x in range(k + 1)),
                        Fraction(1, 1),
                    )

    def test_uniform_prior_ppr_equivalence(self):
        N = 8
        for alpha in (Fraction(1, 10), Fraction(1, 20)):
            for k in range(N + 1):
                for x in range(k + 1):
                    mask = rc.ppr_current_mask(N, k, x, alpha)
                    likelihood_sum = sum(
                        rc.hyper_pmf_fraction(N, K, k, x) for K in range(N + 1)
                    )
                    for K in range(N + 1):
                        likelihood = rc.hyper_pmf_fraction(N, K, k, x)
                        posterior = likelihood / likelihood_sum if likelihood_sum else 0
                        ratio = Fraction(1, N + 1) / posterior if posterior else None
                        self.assertEqual(bool(mask & (1 << K)), ratio is not None and ratio < 1 / alpha)

    def test_terminal_ppr_singleton(self):
        N = 32
        for alpha in (Fraction(1, 10), Fraction(1, 100)):
            for x in range(N + 1):
                self.assertEqual(rc.ppr_current_mask(N, N, x, alpha), 1 << x)

    def test_empty_mask_does_not_imply_side(self):
        self.assertIsNone(rc.mask_decision(0, 32))

    def test_output_directory_is_frozen(self):
        config = rc.load_and_validate_config()
        self.assertEqual(rc.output_paths(config)["formal_result"].parent, (rc.BASE / "outputs").resolve())


if __name__ == "__main__":
    unittest.main()
