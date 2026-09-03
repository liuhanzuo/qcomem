from __future__ import annotations

import copy
import unittest

import torch

from qcomem_functional_cache import (
    FunctionalHybridState,
    TinyFunctionalHybridLayer,
    assert_out_of_place_transition,
    run_segments,
    state_tensors,
)


class FunctionalCacheReferenceTest(unittest.TestCase):
    def _models_and_inputs(self):
        torch.manual_seed(31)
        merged_model = TinyFunctionalHybridLayer(width=8, heads=2, kernel_size=3)
        segmented_model = copy.deepcopy(merged_model)
        document = torch.randn(2, 5, 8)
        query = torch.randn(2, 4, 8)
        return merged_model, segmented_model, document, query

    def test_document_query_hidden_and_all_states_match_merged(self) -> None:
        merged_model, segmented_model, document, query = self._models_and_inputs()
        merged, merged_state = merged_model(torch.cat((document, query), dim=1))
        segmented, segmented_state = run_segments(
            segmented_model, (document[:, :2], document[:, 2:], query)
        )
        torch.testing.assert_close(segmented, merged, atol=2e-6, rtol=2e-6)
        self.assertEqual(
            len(state_tensors(segmented_state)), len(state_tensors(merged_state))
        )
        for segmented_tensor, merged_tensor in zip(
            state_tensors(segmented_state), state_tensors(merged_state)
        ):
            torch.testing.assert_close(
                segmented_tensor, merged_tensor, atol=2e-6, rtol=2e-6
            )

    def test_document_query_parameter_and_input_gradients_match_merged(self) -> None:
        merged_model, segmented_model, document, query = self._models_and_inputs()
        merged_document = document.clone().requires_grad_(True)
        merged_query = query.clone().requires_grad_(True)
        segmented_document = document.clone().requires_grad_(True)
        segmented_query = query.clone().requires_grad_(True)

        merged, _ = merged_model(torch.cat((merged_document, merged_query), dim=1))
        segmented, _ = run_segments(
            segmented_model, (segmented_document, segmented_query)
        )
        weights = torch.linspace(0.2, 1.1, merged.numel()).view_as(merged)
        (merged * weights).sum().backward()
        (segmented * weights).sum().backward()

        torch.testing.assert_close(
            segmented_document.grad, merged_document.grad, atol=3e-6, rtol=3e-6
        )
        torch.testing.assert_close(
            segmented_query.grad, merged_query.grad, atol=3e-6, rtol=3e-6
        )
        self.assertTrue(torch.isfinite(segmented_document.grad).all())
        self.assertTrue(torch.isfinite(segmented_query.grad).all())
        self.assertGreater(float(segmented_document.grad.abs().max()), 0.0)
        self.assertGreater(float(segmented_query.grad.abs().max()), 0.0)
        merged_parameters = dict(merged_model.named_parameters())
        segmented_parameters = dict(segmented_model.named_parameters())
        self.assertEqual(merged_parameters.keys(), segmented_parameters.keys())
        for name in merged_parameters:
            self.assertIsNotNone(merged_parameters[name].grad, name)
            self.assertIsNotNone(segmented_parameters[name].grad, name)
            self.assertTrue(torch.isfinite(segmented_parameters[name].grad).all(), name)
            self.assertGreater(
                float(segmented_parameters[name].grad.abs().max()), 0.0, name
            )
            torch.testing.assert_close(
                segmented_parameters[name].grad,
                merged_parameters[name].grad,
                atol=5e-6,
                rtol=5e-6,
                msg=lambda message, name=name: f"{name}: {message}",
            )

    def test_query_transition_does_not_mutate_or_alias_document_state(self) -> None:
        model, _, document, query = self._models_and_inputs()
        _, document_state = model(document)
        snapshots = [tensor.detach().clone() for tensor in state_tensors(document_state)]
        versions = [tensor._version for tensor in state_tensors(document_state)]
        _, query_state = model(query, document_state)

        assert_out_of_place_transition(document_state, query_state)
        self.assertEqual(len(state_tensors(document_state)), len(snapshots))
        self.assertEqual(len(state_tensors(document_state)), len(versions))
        for tensor, snapshot, version in zip(
            state_tensors(document_state), snapshots, versions
        ):
            self.assertEqual(tensor._version, version)
            self.assertTrue(torch.equal(tensor, snapshot))

    def test_empty_state_is_valid_and_empty_segments_are_rejected(self) -> None:
        self.assertEqual(state_tensors(FunctionalHybridState()), ())
        model = TinyFunctionalHybridLayer()
        with self.assertRaises(ValueError):
            run_segments(model, ())


if __name__ == "__main__":
    unittest.main()
