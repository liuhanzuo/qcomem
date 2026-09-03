from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from aggregate_hypic_lite import summarize_rows
from hypic_lite import (
    HypicLiteSegment,
    HypicLiteStore,
    _GatedDeltaAffineCapture,
    compose_affine_state,
    even_segment_lengths,
    model_suffix_storage_estimate,
    parse_hypic_lite_config,
    transition_validation_summary,
)


def reference_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    **kwargs,
):
    del kwargs
    if use_qk_l2norm_in_kernel:
        query = torch.nn.functional.normalize(query.float(), dim=-1).to(query)
        key = torch.nn.functional.normalize(key.float(), dim=-1).to(key)
    batch, length, heads, key_dim = key.shape
    value_dim = value.shape[-1]
    state = (
        torch.zeros(batch, heads, key_dim, value_dim, dtype=torch.float32)
        if initial_state is None
        else initial_state.float().clone()
    )
    outputs = []
    for position in range(length):
        state = state * torch.exp(g[:, position]).unsqueeze(-1).unsqueeze(-1)
        key_t = key[:, position].float()
        value_t = value[:, position].float()
        memory = torch.sum(state * key_t.unsqueeze(-1), dim=-2)
        delta = (value_t - memory) * beta[:, position].float().unsqueeze(-1)
        state = state + key_t.unsqueeze(-1) * delta.unsqueeze(-2)
        outputs.append(
            torch.sum(state * query[:, position].float().unsqueeze(-1), dim=-2)
        )
    output = torch.stack(outputs, dim=1).to(value)
    return output, state if output_final_state else None


class FakeAdapter:
    def __init__(self, kernel=reference_gated_delta_rule) -> None:
        self.config = SimpleNamespace(layer_types=["linear_attention"])
        self.layers = [
            SimpleNamespace(
                linear_attn=SimpleNamespace(chunk_gated_delta_rule=kernel)
            )
        ]
        self.num_layers = 1


class ConfigAndCompositionTest(unittest.TestCase):
    def test_preregistered_configs_are_explicit(self) -> None:
        transition = parse_hypic_lite_config("hypic-lite-transition-w8")
        self.assertTrue(transition.uses_transition)
        self.assertEqual(transition.seam_width, 8)
        naive = parse_hypic_lite_config("hypic-lite-naive-w0")
        self.assertFalse(naive.uses_transition)
        with self.assertRaises(ValueError):
            parse_hypic_lite_config("hypic-lite-transition-w4")

    def test_affine_and_naive_composition_are_distinct(self) -> None:
        running = torch.tensor([[[[2.0], [3.0]]]])
        zero_end = torch.tensor([[[[5.0], [7.0]]]])
        transition = torch.tensor([[[[2.0, 0.0], [0.0, 3.0]]]])
        affine = compose_affine_state(running, zero_end, transition)
        naive = compose_affine_state(running, zero_end, None)
        self.assertTrue(torch.equal(affine, torch.tensor([[[[9.0], [16.0]]]])))
        self.assertTrue(torch.equal(naive, torch.tensor([[[[7.0], [10.0]]]])))

    def test_even_segments_cover_every_token(self) -> None:
        self.assertEqual(even_segment_lengths(10, 3), (4, 3, 3))
        with self.assertRaises(ValueError):
            even_segment_lengths(2, 3)


class TransitionExtractionTest(unittest.TestCase):
    def test_internal_kernel_hook_extracts_a_composable_transition(self) -> None:
        torch.manual_seed(7)
        adapter = FakeAdapter()
        query = torch.randn(1, 5, 2, 4)
        key = torch.randn(1, 5, 2, 4)
        value = torch.randn(1, 5, 2, 4)
        g = -torch.rand(1, 5, 2)
        beta = torch.sigmoid(torch.randn(1, 5, 2))
        original = adapter.layers[0].linear_attn.chunk_gated_delta_rule
        with _GatedDeltaAffineCapture(
            adapter,
            depth=0,
            body_start=2,
            capture_transition=True,
            transition_dtype=torch.bfloat16,
            validate_transition=True,
        ) as capture:
            adapter.layers[0].linear_attn.chunk_gated_delta_rule(
                query,
                key,
                value,
                g=g,
                beta=beta,
                initial_state=None,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        self.assertIs(adapter.layers[0].linear_attn.chunk_gated_delta_rule, original)
        affine = capture.affines[0]
        self.assertEqual(tuple(affine.transition.shape), (1, 2, 4, 4))
        self.assertLess(affine.transition_relative_l2_error, 0.01)

    def test_hook_rejects_identity_trick_when_key_value_dims_differ(self) -> None:
        adapter = FakeAdapter()
        with self.assertRaisesRegex(RuntimeError, "key_dim == value_dim"):
            with _GatedDeltaAffineCapture(
                adapter,
                depth=0,
                body_start=0,
                capture_transition=True,
                transition_dtype=torch.bfloat16,
                validate_transition=False,
            ):
                adapter.layers[0].linear_attn.chunk_gated_delta_rule(
                    torch.randn(1, 2, 1, 3),
                    torch.randn(1, 2, 1, 3),
                    torch.randn(1, 2, 1, 2),
                    g=-torch.rand(1, 2, 1),
                    beta=torch.rand(1, 2, 1),
                    output_final_state=True,
                )
        # __exit__ must restore the real callable even after capture failure.
        self.assertIs(
            adapter.layers[0].linear_attn.chunk_gated_delta_rule,
            reference_gated_delta_rule,
        )


class LedgerTest(unittest.TestCase):
    @staticmethod
    def qwen_like_config() -> SimpleNamespace:
        layer_types = [
            "full_attention" if index % 4 == 3 else "linear_attention"
            for index in range(40)
        ]
        return SimpleNamespace(
            layer_types=layer_types,
            num_key_value_heads=2,
            head_dim=256,
            linear_num_value_heads=32,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )

    def test_depth7_4k_ledger_counts_actual_value_head_transitions(self) -> None:
        estimate = model_suffix_storage_estimate(
            self.qwen_like_config(),
            depth=7,
            document_tokens=4096,
            segment_count=4,
            seam_width=8,
        )
        self.assertEqual(estimate["suffix_full_attention_layers"], 9)
        self.assertEqual(estimate["suffix_linear_attention_layers"], 24)
        self.assertEqual(
            estimate["linear_transition_nbytes"],
            24 * 4 * 32 * 128 * 128 * 2,
        )
        self.assertEqual(
            estimate["linear_zero_start_end_state_nbytes"],
            24 * 4 * 32 * 128 * 128 * 4,
        )
        self.assertFalse(
            estimate["compressed_hypic_combination_payload_only"]["q4"][
                "executable"
            ]
        )

    def test_profiles_mark_hybrid_transition_only_as_non_executable(self) -> None:
        config = parse_hypic_lite_config("hypic-lite-transition-w8")
        segments = [
            HypicLiteSegment(
                start=index * 50,
                length=50,
                seam_tokens=0 if index == 0 else 8,
                body_cache=None,
                transitions={},
                transition_validation={},
                full_attention_kv_nbytes=84_000,
                linear_end_state_nbytes=20_000,
                conv_tail_nbytes=1_000,
                transition_nbytes=10_000,
            )
            for index in range(2)
        ]
        store = HypicLiteStore(
            config=config,
            depth=7,
            document_length=100,
            suffix_layers=33,
            suffix_full_attention_layers=9,
            suffix_linear_layers=24,
            segments=segments,
            base_document_nbytes=9_000,
            build_seconds=1.0,
        )
        ledger = store.bytes_ledger()
        self.assertFalse(
            ledger["profiles"]["linear_transition_only"][
                "executable_for_suffix_ttft"
            ]
        )
        self.assertFalse(
            ledger["profiles"]["linear_transition_plus_seam_kv"][
                "executable_for_suffix_ttft"
            ]
        )
        self.assertTrue(
            ledger["profiles"]["full_suffix_local_cache"]["approximate"]
        )
        work = store.work_ledger()
        self.assertEqual(work["hypic_lite_online_seam_token_layer_forwards"], 8 * 33)
        self.assertEqual(work["online_linear_state_compositions"], 2 * 24)

    def test_transition_validation_handles_naive_ablation(self) -> None:
        store = HypicLiteStore(
            config=parse_hypic_lite_config("hypic-lite-naive-w0"),
            depth=7,
            document_length=8,
            suffix_layers=1,
            suffix_full_attention_layers=0,
            suffix_linear_layers=1,
            segments=[],
            base_document_nbytes=1,
            build_seconds=0.0,
        )
        self.assertEqual(
            transition_validation_summary(store)["validated_layer_segments"], 0
        )

    def test_aggregate_keeps_ttft_bytes_and_quality_axes(self) -> None:
        row = {
            "config": "hypic-lite-naive-w0",
            "median_ttft_seconds": 2.0,
            "current_qcomem": {"median_ttft_seconds": 4.0},
            "full_prefix_median_ttft_seconds": 1.0,
            "persistent_bytes": {
                "profiles": {
                    "full_suffix_local_cache": {"persistent_nbytes": 123}
                }
            },
            "same_packed_qcomem_logits": {
                "top1_match": True,
                "relative_logit_l2_error": 0.1,
            },
            "exact_full_prefix_logits": {"top1_match": False},
            "request_work": {"saved_fraction": 0.9},
        }
        summary = summarize_rows([row])["configs"][row["config"]]
        self.assertEqual(summary["ratio_of_medians_vs_qcomem"], 2.0)
        self.assertEqual(summary["median_full_suffix_local_persistent_nbytes"], 123)
        self.assertEqual(summary["same_packed_qcomem_top1_agreement"], 1.0)


if __name__ == "__main__":
    unittest.main()
