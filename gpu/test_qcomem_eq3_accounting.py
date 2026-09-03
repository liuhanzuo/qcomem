"""Torch-free unit tests for the Eq. 3 byte accounting used by A4/A5.

Every test in this file runs without torch, without CUDA and without a
Transformers install.  That is deliberate: the byte accounting is the part of
the deployment claim a reviewer is most likely to want to re-check, and it must
be checkable on a laptop.
"""

from __future__ import annotations

import unittest

from qcomem_eq3_accounting import (
    BF16_ITEMSIZE,
    GROUP_SIZE,
    METADATA_NBYTES_PER_GROUP,
    Eq3IdentityError,
    arm_name,
    assert_eq3_component_identity,
    assert_eq3_identities,
    bf16_reference_nbytes,
    component_record,
    decode_latency_summary,
    empty_store_breakdown,
    eq3_code_nbytes,
    eq3_component_nbytes,
    eq3_compression_ratio,
    eq3_group_nbytes,
    eq3_metadata_nbytes,
    group_count,
    identity_violations,
    native_reference_nbytes,
    parse_arm_name,
    parse_config_length_limits,
    shuffled_arm_orders,
    summarize_arm,
    summarize_components,
    sweep_arms,
    throughput_summary,
    validate_row,
    verbatim_component_nbytes,
)


def packed_record(
    *,
    elements: int,
    bits: int,
    state_type: str = "attention_key",
    layer_index: int | None = 0,
    group_size: int = GROUP_SIZE,
    leaf_path: str = "layers[0].keys",
    native_itemsize: int = 2,
) -> dict:
    """A record whose byte fields are exactly what the packer would produce."""

    groups = group_count(elements, group_size)
    return component_record(
        leaf_path=leaf_path,
        layer_index=layer_index,
        state_type=state_type,
        elements=elements,
        bits=bits,
        group_size=group_size,
        code_nbytes=groups * group_size * bits // 8,
        scale_nbytes=groups * BF16_ITEMSIZE,
        bias_nbytes=groups * BF16_ITEMSIZE,
        native_itemsize=native_itemsize,
    )


class FormatArithmeticTest(unittest.TestCase):
    def test_group_count_is_exact_ceiling(self) -> None:
        self.assertEqual(group_count(0), 0)
        self.assertEqual(group_count(1), 1)
        self.assertEqual(group_count(64), 1)
        self.assertEqual(group_count(65), 2)
        self.assertEqual(group_count(128), 2)
        self.assertEqual(group_count(10**12 + 1), 10**12 // 64 + 1)

    def test_group_bytes_match_the_published_table(self) -> None:
        self.assertEqual(eq3_group_nbytes(8), 68)
        self.assertEqual(eq3_group_nbytes(4), 36)
        self.assertEqual(eq3_group_nbytes(2), 20)
        self.assertEqual(METADATA_NBYTES_PER_GROUP, 4)

    def test_sixteen_is_not_a_packed_width(self) -> None:
        with self.assertRaises(ValueError):
            eq3_group_nbytes(16)
        with self.assertRaises(ValueError):
            eq3_component_nbytes(1024, 16)

    def test_rejects_unsupported_widths_and_group_sizes(self) -> None:
        for bits in (0, 1, 3, 5, 6, 7, 9, 32):
            with self.assertRaises(ValueError):
                eq3_group_nbytes(bits)
        with self.assertRaises(ValueError):
            eq3_group_nbytes(4, group_size=0)
        with self.assertRaises(ValueError):
            eq3_group_nbytes(2, group_size=3)

    def test_compression_ceilings_match_the_revision_table(self) -> None:
        self.assertAlmostEqual(eq3_compression_ratio(8), 1.8823529411764706, places=12)
        self.assertAlmostEqual(eq3_compression_ratio(4), 3.5555555555555554, places=12)
        self.assertAlmostEqual(eq3_compression_ratio(2), 6.4, places=12)

    def test_component_identity_splits_into_codes_and_metadata(self) -> None:
        for elements in (64, 128, 4096, 8_388_608):
            for bits in (2, 4, 8):
                total = eq3_component_nbytes(elements, bits)
                self.assertEqual(
                    total,
                    eq3_code_nbytes(elements, bits) + eq3_metadata_nbytes(elements),
                )
                self.assertEqual(
                    total, group_count(elements) * eq3_group_nbytes(bits)
                )

    def test_partial_group_is_charged_a_whole_group(self) -> None:
        # the packer pads the last group with edge values, so 65 elements at Q4
        # cost two full groups, not one and a bit
        self.assertEqual(eq3_component_nbytes(65, 4), 2 * 36)
        self.assertEqual(eq3_component_nbytes(1, 8), 68)

    def test_qwen35_residual_identity_from_the_archived_cohort(self) -> None:
        # 4,096 document tokens at hidden size 2,048, one batch row
        elements = 4096 * 2048
        q16 = verbatim_component_nbytes(elements, BF16_ITEMSIZE)
        q4 = eq3_component_nbytes(elements, 4)
        self.assertEqual(q16, 16_777_216)
        self.assertEqual(q4, 4_718_592)
        self.assertAlmostEqual(q16 / q4, 3.5555555555555554, places=12)


class ReferenceCountTest(unittest.TestCase):
    def test_native_and_bf16_references_differ_only_by_itemsize(self) -> None:
        self.assertEqual(native_reference_nbytes(1000, 4), 4000)
        self.assertEqual(bf16_reference_nbytes(1000), 2000)

    def test_fp32_gdn_recurrent_state_inflation_at_split_depth_seven(self) -> None:
        # Qwen3.5 GatedDeltaNet recurrent_states are FP32 by construction.  A
        # bits=16 "reference" keeps them at 4 bytes/element; a dtype-consistent
        # BF16 reference keeps them at 2.  Six such layers sit below j = 7.
        recurrent_elements = 524_288
        per_layer_inflation = native_reference_nbytes(
            recurrent_elements, 4
        ) - bf16_reference_nbytes(recurrent_elements)
        self.assertEqual(per_layer_inflation, 1 << 20)
        self.assertEqual(6 * per_layer_inflation, 6 * (1 << 20))
        self.assertEqual(30 * per_layer_inflation, 30 * (1 << 20))

    def test_record_flags_a_dtype_inconsistent_reference(self) -> None:
        fp32_reference = component_record(
            leaf_path="layers[1].recurrent_states[0]",
            layer_index=1,
            state_type="recurrent_state",
            elements=524_288,
            bits=16,
            group_size=GROUP_SIZE,
            code_nbytes=524_288 * 4,
            scale_nbytes=0,
            bias_nbytes=0,
            native_itemsize=4,
        )
        self.assertTrue(fp32_reference["dtype_inconsistent_reference"])
        self.assertEqual(fp32_reference["native_reference_nbytes"], 524_288 * 4)
        self.assertEqual(fp32_reference["bf16_reference_nbytes"], 524_288 * 2)
        # the verbatim identity still holds; it is the reference that is wrong,
        # not the byte count of what was stored
        assert_eq3_component_identity(fp32_reference)

        bf16_reference = component_record(
            leaf_path="layers[0].keys",
            layer_index=0,
            state_type="attention_key",
            elements=1024,
            bits=16,
            group_size=GROUP_SIZE,
            code_nbytes=2048,
            scale_nbytes=0,
            bias_nbytes=0,
            native_itemsize=2,
        )
        self.assertFalse(bf16_reference["dtype_inconsistent_reference"])

    def test_non_floating_leaf_is_not_flagged_and_keeps_its_own_reference(
        self,
    ) -> None:
        counter = component_record(
            leaf_path="layers[0].cumulative_length",
            layer_index=0,
            state_type="other",
            elements=1,
            bits=None,
            group_size=GROUP_SIZE,
            code_nbytes=8,
            scale_nbytes=0,
            bias_nbytes=0,
            native_itemsize=8,
            floating=False,
        )
        self.assertFalse(counter["dtype_inconsistent_reference"])
        self.assertEqual(counter["bf16_reference_nbytes"], 8)
        self.assertFalse(counter["quantized"])


class IdentityAssertionTest(unittest.TestCase):
    def test_conforming_component_passes(self) -> None:
        record = packed_record(elements=8192, bits=4)
        self.assertTrue(record["eq3_identity_ok"])
        self.assertTrue(record["eq3_identity_checked"])
        assert_eq3_component_identity(record)
        self.assertEqual(identity_violations([record]), [])

    def test_missing_metadata_is_caught(self) -> None:
        # this is precisely the failure mode A1 hypothesised: a counter that
        # omits the 4 bytes per group of BF16 scale and bias
        groups = group_count(8192)
        record = component_record(
            leaf_path="layers[0].keys",
            layer_index=0,
            state_type="attention_key",
            elements=8192,
            bits=4,
            group_size=GROUP_SIZE,
            code_nbytes=groups * GROUP_SIZE * 4 // 8,
            scale_nbytes=0,
            bias_nbytes=0,
            native_itemsize=2,
        )
        self.assertFalse(record["eq3_identity_ok"])
        with self.assertRaises(Eq3IdentityError) as caught:
            assert_eq3_component_identity(record)
        self.assertIn("format identity requires", str(caught.exception))
        violations = identity_violations([record])
        self.assertEqual(len(violations), 1)
        self.assertEqual(violations[0]["delta_nbytes"], -groups * 4)

    def test_assert_over_a_collection_reports_the_first_offender(self) -> None:
        good = packed_record(elements=4096, bits=8)
        bad = dict(packed_record(elements=4096, bits=8))
        bad["total_nbytes"] += 1
        bad["leaf_path"] = "layers[3].values"
        with self.assertRaises(Eq3IdentityError) as caught:
            assert_eq3_identities([good, bad])
        self.assertIn("layers[3].values", str(caught.exception))


class SummaryTest(unittest.TestCase):
    def build_records(self) -> list[dict]:
        return [
            component_record(
                leaf_path="document_residual",
                layer_index=None,
                state_type="document_residual",
                elements=4096 * 2048,
                bits=4,
                group_size=GROUP_SIZE,
                code_nbytes=group_count(4096 * 2048) * 32,
                scale_nbytes=group_count(4096 * 2048) * 2,
                bias_nbytes=group_count(4096 * 2048) * 2,
                native_itemsize=2,
            ),
            packed_record(
                elements=4096 * 512,
                bits=8,
                state_type="attention_key",
                layer_index=3,
                leaf_path="layers[3].keys",
            ),
            packed_record(
                elements=4096 * 512,
                bits=8,
                state_type="attention_value",
                layer_index=3,
                leaf_path="layers[3].values",
            ),
            packed_record(
                elements=524_288,
                bits=8,
                state_type="recurrent_state",
                layer_index=0,
                leaf_path="layers[0].recurrent_states[0]",
                native_itemsize=4,
            ),
            packed_record(
                elements=32_768,
                bits=8,
                state_type="conv_state",
                layer_index=0,
                leaf_path="layers[0].conv_states[0]",
            ),
        ]

    def test_totals_and_grouping(self) -> None:
        records = self.build_records()
        summary = summarize_components(records)
        self.assertTrue(summary["eq3_identity_ok"])
        self.assertEqual(summary["checked_components"], 5)
        self.assertEqual(
            summary["packed_store_nbytes"],
            sum(record["total_nbytes"] for record in records),
        )
        self.assertEqual(
            set(summary["by_state_type"]),
            {
                "document_residual",
                "attention_key",
                "attention_value",
                "recurrent_state",
                "conv_state",
            },
        )
        layers = [entry["layer_index"] for entry in summary["by_layer"]]
        self.assertEqual(layers, [0, 3, None])
        self.assertEqual(summary["by_layer"][0]["components"], 2)

    def test_both_reference_counts_are_emitted(self) -> None:
        records = self.build_records()
        summary = summarize_components(records)
        # the recurrent state is FP32, so the two references differ by exactly
        # its element count
        self.assertEqual(
            summary["native_dtype_reference_nbytes"]
            - summary["bf16_reference_nbytes"],
            524_288 * 2,
        )
        self.assertGreater(summary["native_dtype_ratio"], summary["bf16_ratio"])
        self.assertEqual(summary["dtype_inconsistent_components"], 0)

    def test_reconciliation_against_the_frozen_accountant(self) -> None:
        records = self.build_records()
        total = sum(record["storage_nbytes"] for record in records)
        summary = summarize_components(records, reconciliation_nbytes=total)
        self.assertTrue(summary["reconciliation"]["matches"])
        self.assertEqual(summary["reconciliation"]["delta_nbytes"], 0)
        drifted = summarize_components(records, reconciliation_nbytes=total + 64)
        self.assertFalse(drifted["reconciliation"]["matches"])
        self.assertEqual(drifted["reconciliation"]["delta_nbytes"], -64)

    def test_empty_breakdown_is_well_formed(self) -> None:
        summary = empty_store_breakdown()
        self.assertEqual(summary["packed_store_nbytes"], 0)
        self.assertTrue(summary["eq3_identity_ok"])
        self.assertIsNone(summary["bf16_ratio"])
        self.assertEqual(summary["components"], [])


class LatencyAndThroughputTest(unittest.TestCase):
    def test_decode_summary_reports_the_shape_of_the_distribution(self) -> None:
        values = [0.01] * 8 + [0.02] * 8
        summary = decode_latency_summary(values)
        self.assertEqual(summary["decode_steps"], 16)
        self.assertAlmostEqual(summary["decode_seconds_total"], 0.24)
        self.assertAlmostEqual(summary["decode_seconds_first"], 0.01)
        self.assertAlmostEqual(summary["decode_seconds_last"], 0.02)
        self.assertAlmostEqual(summary["decode_seconds_first_quarter_mean"], 0.01)
        self.assertAlmostEqual(summary["decode_seconds_last_quarter_mean"], 0.02)
        self.assertAlmostEqual(summary["decode_seconds_max"], 0.02)

    def test_empty_decode_summary(self) -> None:
        summary = decode_latency_summary([])
        self.assertEqual(summary["decode_steps"], 0)
        self.assertIsNone(summary["decode_seconds_median"])

    def test_reconstructed_model_is_reported_beside_the_measurement(self) -> None:
        # the model Table 2's tok/s was derived from: n / (TTFT + n * TPOT).
        # It ignores the last step never being paid and any instrumentation.
        summary = throughput_summary(
            generated_tokens=8,
            ttft_seconds=0.5,
            tpot_seconds=[0.1] * 7,
            online_seconds=0.5 + 0.7,
            request_wall_seconds=1.4,
            end_to_end_including_build_seconds=2.0,
        )
        self.assertAlmostEqual(summary["online_tokens_per_second"], 8 / 1.2)
        self.assertAlmostEqual(summary["wall_tokens_per_second"], 8 / 1.4)
        self.assertAlmostEqual(summary["end_to_end_tokens_per_second"], 4.0)
        self.assertAlmostEqual(summary["reconstructed_model_seconds"], 0.5 + 0.8)
        self.assertAlmostEqual(summary["reconstructed_tokens_per_second"], 8 / 1.3)
        self.assertAlmostEqual(
            summary["reconstructed_over_measured"], (8 / 1.3) / (8 / 1.2)
        )
        self.assertAlmostEqual(
            summary["instrumentation_overhead_seconds"], 1.4 - 1.2
        )

    def test_throughput_summary_tolerates_missing_timings(self) -> None:
        summary = throughput_summary(
            generated_tokens=0,
            ttft_seconds=None,
            tpot_seconds=[],
            online_seconds=None,
            request_wall_seconds=None,
        )
        self.assertIsNone(summary["online_tokens_per_second"])
        self.assertIsNone(summary["reconstructed_tokens_per_second"])
        self.assertIsNone(summary["reconstructed_over_measured"])


class SweepPlanningTest(unittest.TestCase):
    def test_arm_names_round_trip(self) -> None:
        self.assertEqual(arm_name("full-prefix-q4", 128), "full-prefix-q4@n128")
        self.assertEqual(
            parse_arm_name("qcomem-d7-frozen-static@n512"),
            ("qcomem-d7-frozen-static", 512),
        )
        with self.assertRaises(ValueError):
            parse_arm_name("no-length-here")

    def test_full_cross_product(self) -> None:
        arms = sweep_arms(["a", "b"], [8, 128, 512])
        self.assertEqual(len(arms), 6)
        self.assertEqual(
            [arm["arm"] for arm in arms],
            ["a@n8", "a@n128", "a@n512", "b@n8", "b@n128", "b@n512"],
        )

    def test_config_length_limits_exclude_only_the_named_config(self) -> None:
        limits = parse_config_length_limits(["dense-recompute=8"])
        self.assertEqual(limits, {"dense-recompute": 8})
        arms = sweep_arms(
            ["dense-recompute", "dense-prefill-once"],
            [8, 128, 512],
            config_length_limits=limits,
        )
        self.assertEqual(
            [arm["arm"] for arm in arms],
            [
                "dense-recompute@n8",
                "dense-prefill-once@n8",
                "dense-prefill-once@n128",
                "dense-prefill-once@n512",
            ],
        )

    def test_limits_must_name_a_declared_config(self) -> None:
        with self.assertRaises(ValueError):
            sweep_arms(["a"], [8], config_length_limits={"b": 8})
        with self.assertRaises(ValueError):
            parse_config_length_limits(["no-equals-sign"])

    def test_rejects_degenerate_sweeps(self) -> None:
        with self.assertRaises(ValueError):
            sweep_arms([], [8])
        with self.assertRaises(ValueError):
            sweep_arms(["a"], [])
        with self.assertRaises(ValueError):
            sweep_arms(["a", "a"], [8])
        with self.assertRaises(ValueError):
            sweep_arms(["a"], [8, 8])
        with self.assertRaises(ValueError):
            sweep_arms(["a"], [0])
        with self.assertRaises(ValueError):
            sweep_arms(["a"], [8], config_length_limits={"a": 4})

    def test_orders_are_deterministic_permutations(self) -> None:
        names = ["a@n8", "b@n8", "c@n8", "d@n8"]
        first = shuffled_arm_orders(names, repeats=3, seed=20260902)
        second = shuffled_arm_orders(names, repeats=3, seed=20260902)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        for order in first:
            self.assertEqual(sorted(order), sorted(names))
        self.assertNotEqual(
            first, shuffled_arm_orders(names, repeats=3, seed=20260903)
        )
        with self.assertRaises(ValueError):
            shuffled_arm_orders(names, repeats=0, seed=1)


def sample_row(**overrides) -> dict:
    row = {
        "arm": "full-prefix-q4@n128",
        "config": "full-prefix-q4",
        "mode": "full_prefix_quantized",
        "workload_id": "qasper-6",
        "dataset": "qasper",
        "repeat": 0,
        "max_new_tokens_requested": 128,
        "max_new_tokens_effective": 128,
        "generated_tokens": 128,
        "ttft_seconds": 0.4,
        "tpot_seconds": [0.01] * 127,
        "decode_latency": decode_latency_summary([0.01] * 127),
        "throughput": throughput_summary(
            generated_tokens=128,
            ttft_seconds=0.4,
            tpot_seconds=[0.01] * 127,
            online_seconds=0.4 + 1.27,
            request_wall_seconds=1.8,
            end_to_end_including_build_seconds=2.4,
        ),
        "store_breakdown": summarize_components(
            [packed_record(elements=4096 * 512, bits=4)]
        ),
        "persistent_document_nbytes": eq3_component_nbytes(4096 * 512, 4),
        "eos_policy": "ignore",
        "eos_stopped": False,
        "f1": 0.5,
    }
    row.update(overrides)
    return row


class RowValidationTest(unittest.TestCase):
    def test_a_well_formed_row_passes(self) -> None:
        self.assertEqual(validate_row(sample_row()), [])

    def test_missing_field_is_reported(self) -> None:
        row = sample_row()
        del row["store_breakdown"]
        self.assertIn("missing field store_breakdown", validate_row(row))

    def test_ignore_policy_must_reach_the_cap(self) -> None:
        row = sample_row(generated_tokens=40, tpot_seconds=[0.01] * 39)
        row["decode_latency"] = decode_latency_summary(row["tpot_seconds"])
        problems = validate_row(row)
        self.assertTrue(
            any("must always reach the cap" in problem for problem in problems)
        )

    def test_stop_policy_records_a_short_generation_honestly(self) -> None:
        row = sample_row(
            eos_policy="stop",
            generated_tokens=40,
            tpot_seconds=[0.01] * 40,
            eos_stopped=True,
        )
        row["decode_latency"] = decode_latency_summary(row["tpot_seconds"])
        self.assertEqual(validate_row(row), [])
        row["eos_stopped"] = False
        self.assertIn(
            "short generation is not marked eos_stopped", validate_row(row)
        )

    def test_decode_step_count_must_match_the_token_count(self) -> None:
        row = sample_row(tpot_seconds=[0.01] * 3)
        row["decode_latency"] = decode_latency_summary(row["tpot_seconds"])
        problems = validate_row(row)
        self.assertTrue(any("inconsistent" in problem for problem in problems))

    def test_generation_beyond_the_cap_is_rejected(self) -> None:
        row = sample_row(generated_tokens=200, tpot_seconds=[0.01] * 199)
        row["decode_latency"] = decode_latency_summary(row["tpot_seconds"])
        problems = validate_row(row)
        self.assertTrue(any("exceeds effective cap" in p for p in problems))

    def test_identity_violation_in_the_breakdown_fails_the_row(self) -> None:
        row = sample_row()
        broken = dict(row["store_breakdown"]["components"][0])
        broken["total_nbytes"] += 4
        row["store_breakdown"] = summarize_components([broken])
        problems = validate_row(row)
        self.assertTrue(any("eq3 identity violations" in p for p in problems))

    def test_dataset_generation_limit_is_honoured_by_the_validator(self) -> None:
        row = sample_row(
            max_new_tokens_requested=512,
            max_new_tokens_effective=128,
        )
        self.assertEqual(validate_row(row), [])


class ArmSummaryTest(unittest.TestCase):
    def test_summary_reports_cap_attainment_and_both_references(self) -> None:
        rows = [sample_row(workload_id=f"qasper-{index}") for index in range(4)]
        rows[3]["generated_tokens"] = 100
        summary = summarize_arm(rows)
        self.assertEqual(summary["rows"], 4)
        self.assertEqual(summary["arm"], "full-prefix-q4@n128")
        self.assertEqual(summary["reached_cap_fraction"], 0.75)
        self.assertEqual(summary["generated_tokens_min"], 100)
        self.assertEqual(summary["generated_tokens_max"], 128)
        self.assertIsNotNone(summary["store_bf16_reference_nbytes_median"])
        self.assertIsNotNone(summary["store_native_reference_nbytes_median"])
        self.assertAlmostEqual(summary["f1_mean"], 0.5)

    def test_summary_needs_at_least_one_row(self) -> None:
        with self.assertRaises(ValueError):
            summarize_arm([])


if __name__ == "__main__":
    unittest.main()
