from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from qcomem_single_token_gdn_ownership import (
    SingleTokenGDNOwnershipError,
    exact_alias,
    overlaps,
    prepare_borrowed_single_token_conv_transition,
)


def cache_from_tensors(values: list[torch.Tensor]) -> SimpleNamespace:
    return SimpleNamespace(
        layers=[SimpleNamespace(conv_states={0: value}) for value in values]
    )


class SingleTokenOwnershipTest(unittest.TestCase):
    def setUp(self) -> None:
        self.base_values = [torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + index for index in range(3)]
        self.persistent = cache_from_tensors(self.base_values)
        self.requests = [cache_from_tensors(self.base_values), cache_from_tensors(self.base_values)]

    def test_first_call_clones_selected_only_and_second_is_noop(self) -> None:
        receipt = prepare_borrowed_single_token_conv_transition(
            self.persistent, self.requests, (0, 1, 2), request_index=0
        )
        self.assertEqual(receipt["cloned_tensor_count"], 3)
        for index in range(3):
            selected = self.requests[0].layers[index].conv_states[0]
            base = self.persistent.layers[index].conv_states[0]
            peer = self.requests[1].layers[index].conv_states[0]
            self.assertFalse(overlaps(selected, base))
            self.assertFalse(overlaps(selected, peer))
            self.assertTrue(exact_alias(base, peer))
            self.assertTrue(torch.equal(selected, base))
        second = prepare_borrowed_single_token_conv_transition(
            self.persistent, self.requests, (0, 1, 2), request_index=0
        )
        self.assertEqual(second["cloned_tensor_count"], 0)
        self.assertEqual(second["already_private_tensor_count"], 3)

    def test_both_requests_become_pairwise_private(self) -> None:
        prepare_borrowed_single_token_conv_transition(
            self.persistent, self.requests, (0, 1, 2), request_index=0
        )
        prepare_borrowed_single_token_conv_transition(
            self.persistent, self.requests, (0, 1, 2), request_index=1
        )
        for index in range(3):
            left = self.requests[0].layers[index].conv_states[0]
            right = self.requests[1].layers[index].conv_states[0]
            base = self.persistent.layers[index].conv_states[0]
            self.assertFalse(overlaps(left, right))
            self.assertFalse(overlaps(left, base))
            self.assertFalse(overlaps(right, base))

    def test_partial_base_overlap_fails_closed(self) -> None:
        backing = torch.arange(32, dtype=torch.float32)
        base = backing[:24].reshape(2, 3, 4)
        partial = backing[1:25].reshape(2, 3, 4)
        persistent = cache_from_tensors([base])
        requests = [cache_from_tensors([partial]), cache_from_tensors([base])]
        with self.assertRaises(SingleTokenGDNOwnershipError):
            prepare_borrowed_single_token_conv_transition(
                persistent, requests, (0,), request_index=0
            )


if __name__ == "__main__":
    unittest.main()
