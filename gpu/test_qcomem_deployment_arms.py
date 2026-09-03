"""Tests for the A4/A5 deployment arms.

These require torch (they exercise the real Eq. 3 packer and the real decode
loop) but not CUDA and not a model checkpoint: caches are built from
``SimpleNamespace`` layers holding real tensors, and generation runs against a
fake adapter.  On a machine without torch the whole module skips, which is the
expected outcome on the authoring laptop; the byte arithmetic these tests wrap
is covered torch-free in ``test_qcomem_eq3_accounting.py``.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from types import SimpleNamespace

try:  # pragma: no cover - import guard
    import torch

    from qcomem_deployment import DeploymentConfig, MemoryRecorder
    from qcomem_deployment_arms import (
        PackedFullPrefixState,
        StridedMemoryRecorder,
        auto_decode_stride,
        build_extended_persistent_state,
        classify_state_type,
        parse_extended_deployment_config,
        persistent_components_extended,
        run_dense_semantics_gate,
        run_extended_generation,
        run_full_prefix_quant_gate,
        store_breakdown_for_state,
        walk_cache_components,
    )
    from qcomem_torch import (
        FullPrefixState,
        cache_nbytes,
        quantize_transformers_cache,
    )

    TORCH_IMPORT_ERROR: str | None = None
except ImportError as error:  # pragma: no cover - laptop path
    TORCH_IMPORT_ERROR = f"{type(error).__name__}: {error}"


requires_torch = unittest.skipIf(
    TORCH_IMPORT_ERROR is not None,
    f"torch/transformers unavailable ({TORCH_IMPORT_ERROR})",
)


def make_hybrid_cache():
    """A two-layer cache shaped like Qwen3.5: one attention, one GDN layer.

    ``recurrent_states`` is FP32 on purpose: that is what the real model
    produces and it is the component that makes the native-dtype and all-BF16
    references disagree.
    """

    torch.manual_seed(20260902)
    attention = SimpleNamespace(
        keys=torch.randn(1, 2, 64, 16).to(torch.bfloat16),
        values=torch.randn(1, 2, 64, 16).to(torch.bfloat16),
    )
    linear = SimpleNamespace(
        conv_states=[torch.randn(1, 48, 64).to(torch.bfloat16)],
        recurrent_states=[torch.randn(1, 4, 16, 16, dtype=torch.float32)],
        # a deliberately ragged component so the group-padding path is exercised
        ragged_state=torch.randn(100).to(torch.bfloat16),
    )
    return SimpleNamespace(layers=[attention, linear])


@dataclass
class FakeCache:
    tokens: "torch.Tensor"


class FakeAdapter:
    """Minimal adapter exposing the three calls the new modes use."""

    vocab_size = 32

    def __init__(self) -> None:
        self.prefill_calls: list[int] = []
        self.continue_calls: list[int] = []
        self.full_calls: list[int] = []

    @classmethod
    def _logits(cls, last_token: int) -> "torch.Tensor":
        logits = torch.full((1, cls.vocab_size), -1000.0)
        logits[0, (last_token + 1) % cls.vocab_size] = 1.0
        return logits

    def full_last_logits(self, tokens):
        self.full_calls.append(int(tokens.shape[1]))
        return self._logits(int(tokens[0, -1]))

    def prefill_full_prefix(self, tokens):
        self.prefill_calls.append(int(tokens.shape[1]))
        state = FullPrefixState(
            document_length=int(tokens.shape[1]),
            current_length=int(tokens.shape[1]),
            cache=FakeCache(tokens.clone()),
        )
        return self._logits(int(tokens[0, -1])), state

    def write_full_prefix(self, tokens):
        return FullPrefixState(
            document_length=int(tokens.shape[1]),
            current_length=int(tokens.shape[1]),
            cache=FakeCache(tokens.clone()),
        )

    def continue_full_prefix(self, state, tokens):
        self.continue_calls.append(int(tokens.shape[1]))
        state.cache.tokens = torch.cat([state.cache.tokens, tokens], dim=1)
        state.current_length += int(tokens.shape[1])
        return self._logits(int(tokens[0, -1]))


@requires_torch
class ConfigParsingTest(unittest.TestCase):
    def test_dense_prefill_once(self) -> None:
        config = parse_extended_deployment_config("dense-prefill-once")
        self.assertEqual(config.mode, "dense_prefill_once")
        self.assertIsNone(config.depth)

    def test_uniform_quantized_full_prefix(self) -> None:
        for bits in (2, 4, 8):
            config = parse_extended_deployment_config(f"full-prefix-q{bits}")
            self.assertEqual(config.mode, "full_prefix_quantized")
            self.assertEqual(config.attention_bits, bits)
            self.assertEqual(config.linear_bits, bits)
            self.assertIsNone(config.cache_layer_bits)

    def test_state_type_policy_names(self) -> None:
        frozen = parse_extended_deployment_config("full-prefix-frozen-static")
        self.assertEqual(frozen.mode, "full_prefix_quantized")
        self.assertEqual(frozen.attention_bits, 4)
        self.assertEqual(frozen.linear_bits, 8)
        explicit = parse_extended_deployment_config("full-prefix-a4-l8")
        self.assertEqual(
            (explicit.attention_bits, explicit.linear_bits),
            (frozen.attention_bits, frozen.linear_bits),
        )

    def test_published_names_are_unchanged(self) -> None:
        self.assertEqual(
            parse_extended_deployment_config("full-prefix-q16").mode, "full_prefix"
        )
        self.assertEqual(
            parse_extended_deployment_config("full-prefix").mode, "full_prefix"
        )
        self.assertEqual(
            parse_extended_deployment_config("dense-recompute").mode,
            "dense_recompute",
        )
        frozen = parse_extended_deployment_config("qcomem-d7-frozen-static")
        self.assertEqual(frozen.mode, "qcomem")
        self.assertEqual(frozen.cache_layer_bits, (8, 8, 8, 4, 8, 8, 8))

    def test_bits16_quantized_full_prefix_is_refused(self) -> None:
        config = parse_extended_deployment_config("full-prefix-a16-l16")
        with self.assertRaises(ValueError) as caught:
            build_extended_persistent_state(
                FakeAdapter(),
                config,
                torch.arange(8).unsqueeze(0),
                group_size=64,
            )
        message = str(caught.exception)
        self.assertIn("bits=16 full-prefix packer arm is refused", message)
        self.assertIn("full-prefix-q16", message)


@requires_torch
class StoreBreakdownTest(unittest.TestCase):
    def packed_state(self, *, attention_bits=4, linear_bits=8):
        packed = quantize_transformers_cache(
            make_hybrid_cache(),
            attention_bits=attention_bits,
            linear_bits=linear_bits,
            group_size=64,
        )
        return PackedFullPrefixState(
            document_length=64,
            current_length=64,
            cache=packed,
            attention_bits=attention_bits,
            linear_bits=linear_bits,
            cache_layer_bits=None,
            group_size=64,
            materialized_nbytes=0,
        )

    def test_state_type_classification(self) -> None:
        self.assertEqual(
            classify_state_type("layers[1].recurrent_states[0]"), "recurrent_state"
        )
        self.assertEqual(
            classify_state_type("layers[1].conv_states[0]"), "conv_state"
        )
        self.assertEqual(classify_state_type("layers[0].keys"), "attention_key")
        self.assertEqual(classify_state_type("layers[0].values"), "attention_value")
        self.assertEqual(classify_state_type("document_residual"), "document_residual")
        self.assertEqual(classify_state_type("layers[0].mystery"), "other")

    def test_every_packed_component_satisfies_the_byte_identity(self) -> None:
        breakdown = store_breakdown_for_state(self.packed_state(), strict=True)
        self.assertTrue(breakdown["eq3_identity_ok"])
        self.assertEqual(breakdown["eq3_identity_violations"], [])
        self.assertGreaterEqual(breakdown["checked_components"], 5)
        for component in breakdown["components"]:
            if not component["is_packed_width"]:
                continue
            expected = component["groups"] * (
                component["group_size"] * component["bits"] // 8 + 4
            )
            self.assertEqual(component["total_nbytes"], expected)

    def test_ragged_component_pays_for_a_whole_padded_group(self) -> None:
        breakdown = store_breakdown_for_state(self.packed_state(), strict=True)
        ragged = next(
            component
            for component in breakdown["components"]
            if component["leaf_path"].endswith("ragged_state")
        )
        self.assertEqual(ragged["elements"], 100)
        self.assertEqual(ragged["groups"], 2)
        self.assertEqual(ragged["bits"], 8)
        self.assertEqual(ragged["total_nbytes"], 2 * 68)

    def test_breakdown_reconciles_with_the_frozen_accountant(self) -> None:
        state = self.packed_state()
        breakdown = store_breakdown_for_state(state, strict=True)
        self.assertTrue(breakdown["reconciliation"]["matches"])
        self.assertEqual(
            breakdown["reconciliation"]["frozen_accountant_nbytes"],
            state.cache.nbytes,
        )
        self.assertEqual(
            breakdown["packed_store_storage_nbytes"], state.stored_nbytes
        )

    def test_layer_and_state_type_grouping(self) -> None:
        breakdown = store_breakdown_for_state(self.packed_state(), strict=True)
        self.assertEqual(
            sorted(breakdown["by_state_type"]),
            ["attention_key", "attention_value", "conv_state", "other", "recurrent_state"],
        )
        layers = [entry["layer_index"] for entry in breakdown["by_layer"]]
        self.assertEqual(layers, [0, 1])
        self.assertEqual(breakdown["by_layer"][0]["components"], 2)
        self.assertEqual(breakdown["by_layer"][1]["components"], 3)
        self.assertEqual(
            breakdown["by_state_type"]["attention_key"]["bits"], [4]
        )
        self.assertEqual(
            breakdown["by_state_type"]["recurrent_state"]["bits"], [8]
        )

    def test_both_reference_counts_and_the_fp32_gap(self) -> None:
        breakdown = store_breakdown_for_state(self.packed_state(), strict=True)
        recurrent_elements = 1 * 4 * 16 * 16
        self.assertEqual(
            breakdown["native_dtype_reference_nbytes"]
            - breakdown["bf16_reference_nbytes"],
            recurrent_elements * 2,
        )
        self.assertGreater(breakdown["native_dtype_ratio"], breakdown["bf16_ratio"])
        # nothing is stored at bits=16 here, so no component inherits the
        # verbatim-clone reference defect
        self.assertEqual(breakdown["dtype_inconsistent_components"], 0)

    def test_exact_reference_arm_is_flagged_not_silently_averaged(self) -> None:
        exact = FullPrefixState(
            document_length=64, current_length=64, cache=make_hybrid_cache()
        )
        breakdown = store_breakdown_for_state(exact, strict=True)
        self.assertEqual(breakdown["native_dtype_ratio"], 1.0)
        self.assertLess(breakdown["bf16_ratio"], 1.0)
        self.assertEqual(breakdown["dtype_inconsistent_components"], 1)
        flagged = [
            component["leaf_path"]
            for component in breakdown["components"]
            if component["dtype_inconsistent_reference"]
        ]
        self.assertEqual(flagged, ["layers[1].recurrent_states[0]"])
        self.assertTrue(breakdown["reconciliation"]["matches"])

    def test_quantized_store_is_smaller_than_the_exact_store(self) -> None:
        exact = FullPrefixState(
            document_length=64, current_length=64, cache=make_hybrid_cache()
        )
        exact_breakdown = store_breakdown_for_state(exact, strict=True)
        packed_breakdown = store_breakdown_for_state(self.packed_state(), strict=True)
        self.assertLess(
            packed_breakdown["packed_store_nbytes"],
            exact_breakdown["packed_store_nbytes"],
        )
        # component-by-component the shapes and dtypes are preserved
        self.assertEqual(
            {
                component["leaf_path"]: component["elements"]
                for component in packed_breakdown["components"]
            },
            {
                component["leaf_path"]: component["elements"]
                for component in exact_breakdown["components"]
            },
        )

    def test_no_retained_state_yields_an_empty_breakdown(self) -> None:
        breakdown = store_breakdown_for_state(None)
        self.assertEqual(breakdown["packed_store_nbytes"], 0)
        self.assertTrue(breakdown["eq3_identity_ok"])
        self.assertIn("retains no cross-request state", breakdown["semantic"])

    def test_walker_deduplicates_shared_storage_like_cache_nbytes(self) -> None:
        shared = torch.randn(1, 2, 64, 16).to(torch.bfloat16)
        cache = SimpleNamespace(
            layers=[SimpleNamespace(keys=shared, values=shared)]
        )
        records = walk_cache_components(cache)
        self.assertEqual(
            sum(record["storage_nbytes"] for record in records), cache_nbytes(cache)
        )


@requires_torch
class PersistentComponentsTest(unittest.TestCase):
    def test_packed_full_prefix_components(self) -> None:
        packed = quantize_transformers_cache(
            make_hybrid_cache(), attention_bits=4, linear_bits=8, group_size=64
        )
        state = PackedFullPrefixState(
            document_length=64,
            current_length=64,
            cache=packed,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=None,
            group_size=64,
            materialized_nbytes=0,
        )
        components = persistent_components_extended(state)
        self.assertEqual(
            components["persistent_document_nbytes"], state.stored_nbytes
        )
        self.assertEqual(components["persistent_residual_nbytes"], 0)
        self.assertEqual(
            components["persistent_total_resident_nbytes"], state.stored_nbytes
        )

    def test_none_state_is_delegated_unchanged(self) -> None:
        components = persistent_components_extended(None)
        self.assertEqual(components["persistent_document_nbytes"], 0)

    def test_fork_materializes_the_native_dtypes(self) -> None:
        packed = quantize_transformers_cache(
            make_hybrid_cache(), attention_bits=4, linear_bits=8, group_size=64
        )
        state = PackedFullPrefixState(
            document_length=64,
            current_length=64,
            cache=packed,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=None,
            group_size=64,
            materialized_nbytes=0,
        )
        breakdown = store_breakdown_for_state(state)
        state.materialized_nbytes = breakdown["native_dtype_reference_nbytes"]
        forked = state.fork()
        self.assertIsInstance(forked, FullPrefixState)
        self.assertEqual(cache_nbytes(forked.cache), state.materialized_nbytes)
        self.assertEqual(
            forked.cache.layers[1].recurrent_states[0].dtype, torch.float32
        )
        self.assertEqual(forked.cache.layers[0].keys.dtype, torch.bfloat16)
        self.assertGreater(state.materialized_nbytes, state.stored_nbytes)


@requires_torch
class DensePrefillOnceTest(unittest.TestCase):
    def run_arm(self, name, state=None, *, max_new_tokens=4, eos=frozenset()):
        adapter = FakeAdapter()
        config = parse_extended_deployment_config(name)
        trace = run_extended_generation(
            adapter,
            config,
            torch.arange(6).unsqueeze(0),
            torch.arange(6, 9).unsqueeze(0),
            state,
            max_new_tokens=max_new_tokens,
            eos_token_ids=set(eos),
            recorder=MemoryRecorder(),
            collect_logits=True,
        )
        return adapter, trace

    def test_prefills_document_and_query_once_then_decodes_incrementally(
        self,
    ) -> None:
        adapter, trace = self.run_arm("dense-prefill-once")
        # exactly one prefill over document+query, then one token per step
        self.assertEqual(adapter.prefill_calls, [9])
        self.assertEqual(adapter.continue_calls, [1, 1, 1])
        # and never the full-history recompute path
        self.assertEqual(adapter.full_calls, [])
        self.assertEqual(len(trace.generated_token_ids), 4)
        self.assertEqual(len(trace.tpot_seconds), 3)

    def test_no_persistent_store_is_claimed(self) -> None:
        _, trace = self.run_arm("dense-prefill-once")
        summary = trace.summary()
        self.assertEqual(summary["fork_memory"]["strategy_effective"], "not-applicable")
        self.assertIn("prefill_cache_nbytes", summary["fork_memory"])
        # the request-local prefill cache is reported, not counted as a store
        self.assertGreater(summary["fork_memory"]["prefill_cache_nbytes"], 0)

    def test_refuses_a_persistent_state(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self.run_arm("dense-prefill-once", state=object())
        self.assertIn("prefills document+query fresh", str(caught.exception))

    def test_eos_stops_early_and_the_true_count_is_recorded(self) -> None:
        # the fake adapter emits (last_token + 1) % vocab, so after the
        # document+query prefix ending at 8 the first emitted token is 9
        _, trace = self.run_arm("dense-prefill-once", max_new_tokens=6, eos={11})
        self.assertEqual(trace.generated_token_ids, [9, 10])
        self.assertLess(len(trace.generated_token_ids), 6)

    def test_empty_eos_set_always_reaches_the_cap(self) -> None:
        _, trace = self.run_arm("dense-prefill-once", max_new_tokens=6, eos=set())
        self.assertEqual(len(trace.generated_token_ids), 6)

    def test_dense_recompute_is_delegated_unchanged(self) -> None:
        adapter, trace = self.run_arm("dense-recompute")
        # the published mode still recomputes the whole history per token
        self.assertEqual(adapter.full_calls, [9, 10, 11, 12])
        self.assertEqual(adapter.prefill_calls, [])
        self.assertEqual(len(trace.generated_token_ids), 4)

    def test_the_two_dense_arms_emit_the_same_tokens(self) -> None:
        _, recompute = self.run_arm("dense-recompute", max_new_tokens=5)
        _, prefill_once = self.run_arm("dense-prefill-once", max_new_tokens=5)
        self.assertEqual(
            recompute.generated_token_ids, prefill_once.generated_token_ids
        )


@requires_torch
class QuantizedFullPrefixGenerationTest(unittest.TestCase):
    def build_state(self):
        packed = quantize_transformers_cache(
            make_hybrid_cache(), attention_bits=4, linear_bits=8, group_size=64
        )
        state = PackedFullPrefixState(
            document_length=6,
            current_length=6,
            cache=packed,
            attention_bits=4,
            linear_bits=8,
            cache_layer_bits=None,
            group_size=64,
            materialized_nbytes=0,
        )
        state.materialized_nbytes = store_breakdown_for_state(state)[
            "native_dtype_reference_nbytes"
        ]
        return state

    def test_requires_a_packed_full_prefix_state(self) -> None:
        config = parse_extended_deployment_config("full-prefix-q4")
        with self.assertRaises(ValueError) as caught:
            run_extended_generation(
                FakeAdapter(),
                config,
                torch.arange(6).unsqueeze(0),
                torch.arange(6, 9).unsqueeze(0),
                None,
                max_new_tokens=2,
                eos_token_ids=set(),
                recorder=MemoryRecorder(),
            )
        self.assertIn("PackedFullPrefixState", str(caught.exception))

    def test_read_forks_dequantizes_and_reports_both_sizes(self) -> None:
        state = self.build_state()
        config = parse_extended_deployment_config("full-prefix-q4")

        class DequantAdapter(FakeAdapter):
            def continue_full_prefix(self, forked, tokens):
                self.continue_calls.append(int(tokens.shape[1]))
                forked.current_length += int(tokens.shape[1])
                return self._logits(int(tokens[0, -1]))

        adapter = DequantAdapter()
        trace = run_extended_generation(
            adapter,
            config,
            torch.arange(6).unsqueeze(0),
            torch.arange(6, 9).unsqueeze(0),
            state,
            max_new_tokens=3,
            eos_token_ids=set(),
            recorder=MemoryRecorder(),
        )
        summary = trace.summary()
        self.assertEqual(adapter.continue_calls, [3, 1, 1])
        self.assertEqual(
            summary["fork_memory"]["packed_store_nbytes"], state.stored_nbytes
        )
        self.assertEqual(
            summary["fork_memory"]["materialized_read_nbytes"],
            state.materialized_nbytes,
        )
        # the active state after Read is the dequantized cache, not the store
        self.assertEqual(
            summary["selected_fork_active_state_steady_nbytes"],
            state.materialized_nbytes,
        )
        self.assertEqual(len(trace.generated_token_ids), 3)


@requires_torch
class RecorderTest(unittest.TestCase):
    def test_stride_one_is_the_published_recorder(self) -> None:
        recorder = StridedMemoryRecorder(decode_stride=1)
        for step in range(10):
            recorder.sample(f"decode_{step:03d}")
        self.assertEqual(len(recorder.samples), 10)
        self.assertEqual(recorder.skipped_decode_samples, 0)

    def test_thinning_keeps_the_first_and_last_decode_sample(self) -> None:
        recorder = StridedMemoryRecorder(decode_stride=4, decode_steps=10)
        recorder.sample("request_start")
        for step in range(10):
            recorder.sample(f"decode_{step:03d}")
        recorder.sample("request_end")
        phases = [sample["phase"] for sample in recorder.samples]
        self.assertIn("decode_000", phases)
        self.assertIn("decode_009", phases)
        self.assertNotIn("decode_001", phases)
        self.assertEqual(recorder.skipped_decode_samples, 10 - 4)
        summary = recorder.summary()
        self.assertEqual(summary["decode_sample_stride"], 4)
        self.assertEqual(summary["skipped_decode_samples"], 10 - 4)

    def test_non_decode_phases_are_never_thinned(self) -> None:
        recorder = StridedMemoryRecorder(decode_stride=8)
        for phase in ("request_start", "ttft", "build_end", "request_end"):
            recorder.sample(phase)
        self.assertEqual(len(recorder.samples), 4)

    def test_auto_stride_bounds_the_sample_count(self) -> None:
        self.assertEqual(auto_decode_stride(8), 1)
        self.assertEqual(auto_decode_stride(64), 1)
        self.assertEqual(auto_decode_stride(128), 2)
        self.assertEqual(auto_decode_stride(512), 8)
        for length in (8, 128, 512, 1024):
            stride = auto_decode_stride(length)
            self.assertLessEqual(-(-length // stride), 64 + 1)


@requires_torch
class DelegationTest(unittest.TestCase):
    def test_extended_modes_are_exactly_the_two_new_ones(self) -> None:
        from qcomem_deployment_arms import EXTENDED_MODES

        self.assertEqual(
            set(EXTENDED_MODES), {"dense_prefill_once", "full_prefix_quantized"}
        )

    def test_unknown_mode_still_raises_from_the_published_function(self) -> None:
        config = DeploymentConfig(name="bogus", mode="not-a-mode")
        with self.assertRaises(ValueError):
            run_extended_generation(
                FakeAdapter(),
                config,
                torch.arange(4).unsqueeze(0),
                torch.arange(4, 6).unsqueeze(0),
                None,
                max_new_tokens=1,
                eos_token_ids=set(),
                recorder=MemoryRecorder(),
            )


class HybridFakeAdapter(FakeAdapter):
    """Fake adapter whose prefix cache looks like a hybrid Qwen3.5 cache."""

    def write_full_prefix(self, tokens):
        return FullPrefixState(
            document_length=int(tokens.shape[1]),
            current_length=int(tokens.shape[1]),
            cache=make_hybrid_cache(),
        )

    def prefill_full_prefix(self, tokens):
        self.prefill_calls.append(int(tokens.shape[1]))
        state = self.write_full_prefix(tokens)
        return self._logits(int(tokens[0, -1])), state

    def continue_full_prefix(self, state, tokens):
        self.continue_calls.append(int(tokens.shape[1]))
        state.current_length += int(tokens.shape[1])
        return self._logits(int(tokens[0, -1]))


@requires_torch
class DenseSemanticsGateTest(unittest.TestCase):
    def run_gate(self, *, eos=frozenset(), max_new_tokens=4):
        return run_dense_semantics_gate(
            FakeAdapter(),
            torch.arange(6).unsqueeze(0),
            torch.arange(6, 9).unsqueeze(0),
            group_size=64,
            max_new_tokens=max_new_tokens,
            eos_token_ids=set(eos),
        )

    def test_all_three_arms_agree_on_the_first_token(self) -> None:
        gate = self.run_gate()
        self.assertTrue(gate["first_token_agrees"])
        self.assertEqual(set(gate["first_token_ids"].values()), {9})

    def test_eos_is_disabled_inside_the_gate(self) -> None:
        # 11 would stop generation after two tokens if EOS were honoured
        gate = self.run_gate(eos={11}, max_new_tokens=5)
        self.assertTrue(gate["eos_disabled_in_gate"])
        self.assertEqual(gate["declared_eos_token_ids"], [11])
        for comparison in gate["diagnostics"].values():
            self.assertEqual(len(comparison["reference_emitted_token_ids"]), 5)
            self.assertEqual(len(comparison["candidate_emitted_token_ids"]), 5)

    def test_diagnostics_are_recorded_not_gated(self) -> None:
        gate = self.run_gate()
        self.assertIn("dense_recompute_vs_dense_prefill_once", gate["diagnostics"])
        self.assertIn(
            "dense_prefill_once_vs_full_prefix_q16", gate["diagnostics"]
        )
        self.assertIn("token-sequence equality", gate["semantic"])
        # timing keys exist for every arm so the paper can report them
        for field in ("ttft_seconds", "median_tpot_seconds", "tpot_over_ttft"):
            self.assertEqual(
                sorted(gate[field]),
                ["dense-prefill-once", "dense-recompute", "full-prefix-q16"],
            )


@requires_torch
class QuantizedFullPrefixGateTest(unittest.TestCase):
    def run_gate(self, names=("full-prefix-q8", "full-prefix-q4")):
        return run_full_prefix_quant_gate(
            HybridFakeAdapter(),
            torch.arange(6).unsqueeze(0),
            torch.arange(6, 9).unsqueeze(0),
            config_names=list(names),
            group_size=64,
            max_new_tokens=3,
            eos_token_ids=set(),
        )

    def test_gate_passes_for_conforming_quantized_arms(self) -> None:
        gate = self.run_gate()
        self.assertTrue(gate["passed"])
        for name, arm in gate["arms"].items():
            self.assertTrue(arm["passed"], name)
            self.assertTrue(arm["eq3_identity_ok"])
            self.assertEqual(arm["eq3_identity_violations"], [])
            self.assertTrue(arm["reconciles_with_frozen_accountant"])
            self.assertTrue(arm["store_smaller_than_exact"])
            self.assertTrue(arm["component_shapes_match_exact"])
            self.assertGreater(arm["materialized_read_nbytes"], arm["packed_store_nbytes"])

    def test_q4_stores_strictly_less_than_q8(self) -> None:
        gate = self.run_gate()
        self.assertLess(
            gate["arms"]["full-prefix-q4"]["packed_store_nbytes"],
            gate["arms"]["full-prefix-q8"]["packed_store_nbytes"],
        )

    def test_exact_reference_counts_are_reported_both_ways(self) -> None:
        gate = self.run_gate()
        self.assertGreater(
            gate["exact_native_dtype_reference_nbytes"],
            gate["exact_bf16_reference_nbytes"],
        )
        self.assertEqual(gate["exact_dtype_inconsistent_components"], 1)
        self.assertEqual(
            gate["exact_store_nbytes"], gate["exact_native_dtype_reference_nbytes"]
        )

    def test_token_divergence_is_reported_not_gated(self) -> None:
        gate = self.run_gate()
        for arm in gate["arms"].values():
            self.assertIn("vs_exact_full_prefix", arm)
        self.assertIn("token equality with the exact arm is NOT required", gate["semantic"])

    def test_non_quantized_arm_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.run_gate(names=("full-prefix-q16",))


if __name__ == "__main__":
    unittest.main()
