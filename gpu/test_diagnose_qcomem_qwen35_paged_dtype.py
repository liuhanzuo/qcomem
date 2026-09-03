from __future__ import annotations

import unittest

import torch

from diagnose_qcomem_qwen35_paged_dtype import _batch_prefix


class DtypeDiagnosticInputTest(unittest.TestCase):
    def test_one_dimensional_longbench_tokens_gain_batch_dimension(self) -> None:
        tokens = torch.arange(10)
        result = _batch_prefix(tokens, 4)
        self.assertEqual(tuple(result.shape), (1, 4))
        torch.testing.assert_close(result, tokens[:4].unsqueeze(0))

    def test_two_dimensional_tokens_preserve_batch_and_truncate(self) -> None:
        tokens = torch.arange(20).reshape(2, 10)
        result = _batch_prefix(tokens, 6)
        self.assertEqual(tuple(result.shape), (2, 6))
        torch.testing.assert_close(result, tokens[:, :6])

    def test_invalid_rank_or_limit_fails_closed(self) -> None:
        for tokens in (torch.tensor(1), torch.zeros(1, 2, 3)):
            with self.assertRaisesRegex(ValueError, "token tensor"):
                _batch_prefix(tokens, 1)
        for limit in (0, -1, True):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                _batch_prefix(torch.arange(3), limit)


if __name__ == "__main__":
    unittest.main()
