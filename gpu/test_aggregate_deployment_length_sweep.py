"""Torch-free tests for the A4/A5 aggregator.

The aggregator re-derives every check from the raw per-row fields, so these
tests build shard JSON by hand and assert that a good run passes and that each
specific way of being wrong is caught.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aggregate_deployment_length_sweep import (
    aggregate,
    check_coverage,
    check_gates,
    check_rows,
    check_shard_consistency,
    load_shards,
    paired_against_reference,
    store_reference_audit,
    throughput_model_audit,
)
from qcomem_eq3_accounting import (
    BF16_ITEMSIZE,
    GROUP_SIZE,
    component_record,
    decode_latency_summary,
    eq3_component_nbytes,
    group_count,
    summarize_components,
    sweep_arms,
    throughput_summary,
)


CONFIGS = ("full-prefix-q16", "full-prefix-q4")
LENGTHS = (8, 128)
ARMS = sweep_arms(list(CONFIGS), list(LENGTHS))
ELEMENTS = 4096 * 512


def breakdown_for(config: str) -> dict:
    if config == "full-prefix-q16":
        record = component_record(
            leaf_path="layers[0].keys",
            layer_index=0,
            state_type="attention_key",
            elements=ELEMENTS,
            bits=None,
            group_size=GROUP_SIZE,
            code_nbytes=ELEMENTS * BF16_ITEMSIZE,
            scale_nbytes=0,
            bias_nbytes=0,
            native_itemsize=BF16_ITEMSIZE,
        )
        recurrent = component_record(
            leaf_path="layers[1].recurrent_states[0]",
            layer_index=1,
            state_type="recurrent_state",
            elements=524_288,
            bits=None,
            group_size=GROUP_SIZE,
            code_nbytes=524_288 * 4,
            scale_nbytes=0,
            bias_nbytes=0,
            native_itemsize=4,
        )
        records = [record, recurrent]
    else:
        groups = group_count(ELEMENTS)
        recurrent_groups = group_count(524_288)
        records = [
            component_record(
                leaf_path="layers[0].keys",
                layer_index=0,
                state_type="attention_key",
                elements=ELEMENTS,
                bits=4,
                group_size=GROUP_SIZE,
                code_nbytes=groups * 32,
                scale_nbytes=groups * 2,
                bias_nbytes=groups * 2,
                native_itemsize=BF16_ITEMSIZE,
            ),
            component_record(
                leaf_path="layers[1].recurrent_states[0]",
                layer_index=1,
                state_type="recurrent_state",
                elements=524_288,
                bits=4,
                group_size=GROUP_SIZE,
                code_nbytes=recurrent_groups * 32,
                scale_nbytes=recurrent_groups * 2,
                bias_nbytes=recurrent_groups * 2,
                native_itemsize=4,
            ),
        ]
    total = sum(record["storage_nbytes"] for record in records)
    return summarize_components(records, reconciliation_nbytes=total)


def make_row(*, config: str, length: int, workload_id: str, repeat: int) -> dict:
    generated = length
    steps = generated - 1
    tpot = [0.01 if config == "full-prefix-q16" else 0.011] * steps
    ttft = 0.4 if config == "full-prefix-q16" else 0.45
    online = ttft + sum(tpot)
    breakdown = breakdown_for(config)
    return {
        "arm": f"{config}@n{length}",
        "config": config,
        "mode": (
            "full_prefix"
            if config == "full-prefix-q16"
            else "full_prefix_quantized"
        ),
        "workload_id": workload_id,
        "dataset": "qasper",
        "repeat": repeat,
        "max_new_tokens_requested": length,
        "max_new_tokens_effective": length,
        "generated_tokens": generated,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "decode_latency": decode_latency_summary(tpot),
        "throughput": throughput_summary(
            generated_tokens=generated,
            ttft_seconds=ttft,
            tpot_seconds=tpot,
            online_seconds=online,
            request_wall_seconds=online * 1.05,
            end_to_end_including_build_seconds=online * 1.3,
        ),
        "store_breakdown": breakdown,
        "persistent_document_nbytes": breakdown["packed_store_nbytes"],
        "eos_policy": "ignore",
        "eos_stopped": False,
        "f1": 0.5 if config == "full-prefix-q16" else 0.45,
    }


def make_shard(rank: int, workload_ids, *, repeats: int = 2) -> dict:
    rows = []
    for workload_id in workload_ids:
        for repeat in range(repeats):
            for arm in ARMS:
                rows.append(
                    make_row(
                        config=arm["config"],
                        length=arm["max_new_tokens"],
                        workload_id=workload_id,
                        repeat=repeat,
                    )
                )
    return {
        "status": "completed",
        "rank": rank,
        "world_size": 2,
        "arms": ARMS,
        "protocol": {"label": "a4-a5-length-sweep", "eos_policy": "ignore"},
        "gates": {
            "dense_semantics_gate": {"passed": True},
            "full_prefix_quant_gate": {"passed": True},
        },
        "rows": rows,
    }


def write_run(directory: Path, *, repeats: int = 2, mutate=None) -> Path:
    shards = [
        make_shard(0, ["qasper-6", "qasper-7"], repeats=repeats),
        make_shard(1, ["qasper-8", "qasper-9"], repeats=repeats),
    ]
    if mutate is not None:
        mutate(shards)
    for shard in shards:
        path = directory / f"length-sweep-shard-{shard['rank']}.json"
        path.write_text(json.dumps(shard, indent=2) + "\n")
    return directory


class ShardLoadingTest(unittest.TestCase):
    def test_requires_shards(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                load_shards(Path(directory))

    def test_expected_shard_count_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory))
            with self.assertRaises(ValueError):
                load_shards(Path(directory), expected_shards=8)
            self.assertEqual(len(load_shards(Path(directory), expected_shards=2)), 2)

    def test_incomplete_shard_is_rejected(self) -> None:
        def mutate(shards):
            shards[1]["status"] = "running"

        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory), mutate=mutate)
            with self.assertRaises(ValueError):
                check_shard_consistency(load_shards(Path(directory)))

    def test_divergent_arm_list_is_rejected(self) -> None:
        def mutate(shards):
            shards[1]["arms"] = shards[1]["arms"][:-1]

        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory), mutate=mutate)
            with self.assertRaises(ValueError):
                check_shard_consistency(load_shards(Path(directory)))

    def test_divergent_protocol_is_rejected(self) -> None:
        def mutate(shards):
            shards[1]["protocol"]["eos_policy"] = "stop"

        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory), mutate=mutate)
            with self.assertRaises(ValueError):
                check_shard_consistency(load_shards(Path(directory)))


class GateTest(unittest.TestCase):
    def test_failed_gate_stops_the_aggregate(self) -> None:
        def mutate(shards):
            shards[0]["gates"]["dense_semantics_gate"]["passed"] = False

        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory), mutate=mutate)
            with self.assertRaises(ValueError):
                check_gates(load_shards(Path(directory)))

    def test_missing_gate_block_stops_the_aggregate(self) -> None:
        def mutate(shards):
            shards[1]["gates"] = {}

        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory), mutate=mutate)
            with self.assertRaises(ValueError):
                check_gates(load_shards(Path(directory)))


class RowAuditTest(unittest.TestCase):
    def test_good_rows_pass_and_are_counted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory))
            rows, audit = check_rows(load_shards(Path(directory)))
            self.assertEqual(len(rows), 4 * 2 * len(ARMS))
            self.assertEqual(audit["eq3_identity_violations"], 0)
            # only the Q4 arm has genuinely packed components
            self.assertEqual(
                audit["eq3_components_checked"], 2 * len(rows) // 2
            )

    def test_byte_identity_violation_is_caught_in_aggregate(self) -> None:
        def mutate(shards):
            row = next(
                row for row in shards[0]["rows"] if row["config"] == "full-prefix-q4"
            )
            row["store_breakdown"]["components"][0]["total_nbytes"] -= 4

        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory), mutate=mutate)
            with self.assertRaises(ValueError) as caught:
                check_rows(load_shards(Path(directory)))
            self.assertIn("Eq. 3 byte identity violated", str(caught.exception))

    def test_short_generation_under_ignore_policy_is_caught(self) -> None:
        def mutate(shards):
            row = shards[0]["rows"][0]
            row["generated_tokens"] = 3
            row["tpot_seconds"] = [0.01, 0.01]
            row["decode_latency"] = decode_latency_summary(row["tpot_seconds"])

        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory), mutate=mutate)
            with self.assertRaises(ValueError) as caught:
                check_rows(load_shards(Path(directory)))
            self.assertIn("invalid rows", str(caught.exception))


class CoverageTest(unittest.TestCase):
    def test_missing_cell_is_caught(self) -> None:
        def mutate(shards):
            shards[0]["rows"] = [
                row for row in shards[0]["rows"] if row["arm"] != "full-prefix-q4@n128"
            ]

        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory), mutate=mutate)
            rows, _ = check_rows(load_shards(Path(directory)))
            with self.assertRaises(ValueError):
                check_coverage(rows, [arm["arm"] for arm in ARMS])


class AnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        write_run(Path(self.directory.name))
        self.rows, _ = check_rows(load_shards(Path(self.directory.name)))

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_pairs_every_arm_against_the_exact_full_prefix_arm(self) -> None:
        comparisons = paired_against_reference(self.rows)
        self.assertEqual(len(comparisons), len(LENGTHS))
        for comparison in comparisons:
            self.assertEqual(comparison["config"], "full-prefix-q4")
            self.assertEqual(comparison["pairs"], 8)
            # the Q4 store is smaller, so the reduction ratio exceeds one
            self.assertGreater(comparison["store_reduction_ratio_median"], 1.0)
            # the native-dtype reference counts the FP32 recurrent state at
            # 4 bytes/element and is therefore strictly the larger numerator
            self.assertGreater(
                comparison["store_reduction_ratio_native_reference_median"],
                comparison["store_reduction_ratio_bf16_reference_median"],
            )
            self.assertAlmostEqual(comparison["f1_delta_mean"], -0.05, places=9)
            self.assertGreater(comparison["ttft_ratio_median"], 1.0)

    def test_throughput_audit_exposes_the_reconstructed_model(self) -> None:
        audit = throughput_model_audit(self.rows)
        self.assertEqual(len(audit), len(CONFIGS) * len(LENGTHS))
        for entry in audit:
            # n / (TTFT + n * TPOT) charges one decode step that never happens,
            # so it always understates the measured online rate
            self.assertLess(entry["reconstructed_over_measured_median"], 1.0)
            self.assertGreater(
                entry["measured_online_tokens_per_second_median"],
                entry["measured_wall_tokens_per_second_median"],
            )

    def test_store_audit_reports_both_references_and_reconciles(self) -> None:
        audit = {entry["config"]: entry for entry in store_reference_audit(self.rows)}
        self.assertEqual(set(audit), set(CONFIGS))
        for entry in audit.values():
            self.assertTrue(entry["reconciles_with_frozen_accountant"])
            self.assertGreaterEqual(
                entry["native_dtype_reference_nbytes_median"],
                entry["bf16_reference_nbytes_median"],
            )
            self.assertIn("recurrent_state", entry["state_types"])
        exact = audit["full-prefix-q16"]
        # the exact arm is its own store, so the native ratio is exactly one
        self.assertAlmostEqual(exact["native_dtype_ratio_median"], 1.0)
        # and its FP32 recurrent leaf is flagged as a dtype-inconsistent
        # reference rather than silently averaged in
        self.assertEqual(exact["dtype_inconsistent_components_max"], 1)
        quantized = audit["full-prefix-q4"]
        self.assertAlmostEqual(
            quantized["bf16_ratio_median"], 3.5555555555555554, places=9
        )
        self.assertEqual(quantized["dtype_inconsistent_components_max"], 0)


class EndToEndTest(unittest.TestCase):
    def test_aggregate_produces_every_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory))
            result = aggregate(Path(directory), expected_shards=2)
            self.assertEqual(result["coverage"]["arms_per_cell"], len(ARMS))
            self.assertEqual(result["coverage"]["workloads"], 4)
            self.assertEqual(result["lengths"], list(LENGTHS))
            self.assertEqual(len(result["arm_summary"]), len(ARMS))
            self.assertTrue(result["gates"]["all_passed"])
            self.assertEqual(result["row_audit"]["eq3_identity_violations"], 0)
            names = {entry["arm"] for entry in result["arm_summary"]}
            self.assertEqual(names, {arm["arm"] for arm in ARMS})
            for entry in result["arm_summary"]:
                self.assertEqual(entry["reached_cap_fraction"], 1.0)

    def test_expected_arms_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory))
            with self.assertRaises(ValueError):
                aggregate(Path(directory), expected_arms=["nope@n8"])

    def test_store_ratio_matches_the_format_ceiling(self) -> None:
        # end-to-end: the Q4 attention component must occupy exactly
        # ceil(n / 64) * 36 bytes, so its BF16 ratio is exactly 32/9
        with tempfile.TemporaryDirectory() as directory:
            write_run(Path(directory))
            result = aggregate(Path(directory), expected_shards=2)
            audit = {
                entry["config"]: entry for entry in result["store_reference_audit"]
            }
            self.assertEqual(
                audit["full-prefix-q4"]["packed_store_nbytes_median"],
                eq3_component_nbytes(ELEMENTS, 4)
                + eq3_component_nbytes(524_288, 4),
            )


if __name__ == "__main__":
    unittest.main()
