"""Torch-free tests for the C1 multifork aggregator.

The aggregator is the blind replay: it must re-derive every contract status
from the shard's own coverage and predicate rather than trusting the producer,
must fail closed on an incomplete record, and must still aggregate a run whose
science came out negative.
"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from aggregate_shared_packed_multifork import (
    AggregateError,
    aggregate,
    arm_coverage,
    contract_report,
    crossover_report,
    replay_contract_rows,
    semantic_equivalence_report,
    validate_shard,
    working_set_table,
)
from qcomem_multifork_accounting import (
    AGGREGATE_SCHEMA,
    MULTIFORK_TARGET_CONTRACT,
    PROTOCOL,
    SHARD_SCHEMA,
    build_multifork_target_rows,
    contract_summary,
    MANDATORY_SLOTS,
)


def passing_target_rows(**predicate_overrides):
    receipts = {}
    for slots in MANDATORY_SLOTS.values():
        for slot in slots:
            receipts[slot] = {
                "present": True,
                "unique": True,
                "bound": True,
                "modified": False,
            }
    predicates = {row["target"]: True for row in MULTIFORK_TARGET_CONTRACT}
    predicates.update(predicate_overrides)
    return build_multifork_target_rows(predicates=predicates, receipts=receipts)


def make_row(
    *,
    arm="qcomem-shared-packed",
    fork_mode="shared-packed-view",
    request_count=2,
    workload_id="qasper-6",
    identical=True,
    entry=1000,
    view=4000,
    slope=200.0,
    with_audit=None,
):
    if with_audit is None:
        with_audit = arm == "qcomem-shared-packed"
    row = {
        "protocol": PROTOCOL,
        "arm": arm,
        "fork_mode": fork_mode,
        "request_count": request_count,
        "workload_id": workload_id,
        "repeat": 0,
        "entry_retained_nbytes": entry,
        "shared_dequantized_view_nbytes": view,
        "per_request_materialized_nbytes": [100] * request_count,
        "transient_materialized_nbytes_total": 100 * request_count,
        "peak_transient_allocation_nbytes": 900,
        "steady_state_resident_nbytes": 5000,
        "per_request_steady_resident_nbytes": [int(slope)] * request_count,
        "resident_model": {
            "intercept_nbytes": entry + view,
            "slope_nbytes_per_request": slope,
        },
        "ownership_ledger": {"request_count": request_count},
        "semantic_equivalence": {
            "token_sequences_identical": identical,
            "discrepancies": []
            if identical
            else [{"request_id": "r01", "first_divergence_step": 2}],
        },
    }
    if with_audit:
        target_rows = passing_target_rows()
        row["forkaudit"] = {
            "target_rows": target_rows,
            "contract_summary": contract_summary(target_rows),
            "sharing_window": "final",
        }
    return row


def make_shard(rank=0, rows=None, **overrides):
    payload = {
        "schema": SHARD_SCHEMA,
        "aggregate_schema": AGGREGATE_SCHEMA,
        "protocol": PROTOCOL,
        "status": "completed",
        "rank": rank,
        "world_size": 2,
        "protocol_settings": {
            "label": "c1-shared-packed-multifork",
            "tail_policy": "borrowed-prefix",
            "rebind_policy": "transition",
        },
        "gates": {"shared_packed_multifork_gate": {"passed": True}},
        "rows": rows
        if rows is not None
        else [
            make_row(arm="qcomem-shared-packed"),
            make_row(
                arm="qcomem-private-materialize",
                fork_mode="private-materialize",
                view=0,
                slope=2000.0,
            ),
            make_row(
                arm="full-prefix",
                fork_mode="private-materialize",
                view=0,
                entry=60_000,
                slope=4000.0,
            ),
        ],
    }
    payload.update(overrides)
    return payload


def write_run(shards, directory: Path):
    for index, payload in enumerate(shards):
        (directory / f"multifork-shard-{index}.json").write_text(
            json.dumps(payload)
        )


class ReplayContractTest(unittest.TestCase):
    def test_clean_rows_replay(self):
        self.assertEqual(replay_contract_rows(passing_target_rows()), [])

    def test_status_inflated_above_the_predicate_is_caught(self):
        rows = passing_target_rows(residual_chunk_binding=False)
        # the producer claims a pass anyway
        for row in rows:
            if row["target"] == "residual_chunk_binding":
                row["status"] = "full"
        problems = replay_contract_rows(rows)
        self.assertTrue(
            any("does not replay to 'open'" in message for message in problems)
        )

    def test_incomplete_coverage_with_a_predicate_is_caught(self):
        rows = copy.deepcopy(passing_target_rows())
        for row in rows:
            if row["target"] == "packed_entry_lifetime":
                row["coverage"] = "incomplete"
                row["status"] = "full"
        problems = replay_contract_rows(rows)
        self.assertTrue(
            any(
                "incomplete coverage carries a non-null predicate" in message
                for message in problems
            )
        )

    def test_missing_target_is_caught(self):
        rows = [
            row
            for row in passing_target_rows()
            if row["target"] != "dequantized_view_immutability"
        ]
        problems = replay_contract_rows(rows)
        self.assertTrue(
            any("contract rows missing" in message for message in problems)
        )

    def test_duplicate_target_is_caught(self):
        rows = passing_target_rows()
        rows = rows + [rows[0]]
        problems = replay_contract_rows(rows)
        self.assertIn("duplicate target row: frozen_identity", problems)

    def test_contract_drift_is_caught(self):
        rows = copy.deepcopy(passing_target_rows())
        rows[0]["maximum_status"] = "partial"
        problems = replay_contract_rows(rows)
        self.assertTrue(
            any("maximum_status drifted" in message for message in problems)
        )

    def test_unknown_target_is_caught(self):
        rows = passing_target_rows() + [{"target": "invented"}]
        self.assertIn("unknown target in shard: 'invented'", replay_contract_rows(rows))


class ShardValidationTest(unittest.TestCase):
    def test_good_shard(self):
        result = validate_shard(Path("s.json"), make_shard())
        self.assertTrue(result["valid"], result["problems"])

    def test_gate_failure_is_a_defect(self):
        shard = make_shard(
            status="gate_failed",
            gates={"shared_packed_multifork_gate": {"passed": False}},
            rows=[],
        )
        result = validate_shard(Path("s.json"), shard)
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("did not pass" in message for message in result["problems"])
        )

    def test_running_shard_is_a_defect(self):
        result = validate_shard(Path("s.json"), make_shard(status="running"))
        self.assertFalse(result["valid"])

    def test_missing_gate_record_is_a_defect(self):
        result = validate_shard(Path("s.json"), make_shard(gates={}))
        self.assertIn("shard carries no gate record", result["problems"])

    def test_schema_drift_is_a_defect(self):
        result = validate_shard(Path("s.json"), make_shard(schema="other"))
        self.assertFalse(result["valid"])

    def test_shared_arm_without_contract_rows_is_a_defect(self):
        row = make_row(with_audit=False)
        result = validate_shard(Path("s.json"), make_shard(rows=[row]))
        self.assertTrue(
            any(
                "shared arm carries no contract rows" in message
                for message in result["problems"]
            )
        )

    def test_contract_summary_drift_is_a_defect(self):
        row = make_row()
        row["forkaudit"]["contract_summary"]["all_applicable_predicates_passed"] = False
        result = validate_shard(Path("s.json"), make_shard(rows=[row]))
        self.assertTrue(
            any("contract summary field" in message for message in result["problems"])
        )

    def test_malformed_row_is_a_defect(self):
        row = make_row()
        row["per_request_materialized_nbytes"] = [1]
        result = validate_shard(Path("s.json"), make_shard(rows=[row]))
        self.assertFalse(result["valid"])


class ReportTest(unittest.TestCase):
    def test_arm_coverage_reports_gaps(self):
        rows = [make_row(arm="qcomem-shared-packed")]
        coverage = arm_coverage(rows)
        self.assertFalse(coverage["complete"])
        self.assertEqual(
            coverage["missing_arms_per_fanout"][2],
            ["full-prefix", "qcomem-private-materialize"],
        )

    def test_arm_coverage_complete(self):
        coverage = arm_coverage(make_shard()["rows"])
        self.assertTrue(coverage["complete"])
        self.assertEqual(coverage["fanouts"], [2])

    def test_semantic_equivalence_separates_gating_from_diagnostic(self):
        rows = [
            make_row(arm="qcomem-shared-packed", identical=False),
            make_row(
                arm="full-prefix",
                fork_mode="private-materialize",
                identical=False,
            ),
        ]
        report = semantic_equivalence_report(rows)
        self.assertEqual(report["gating_failure_count"], 1)
        self.assertFalse(report["gating_all_identical"])
        self.assertEqual(len(report["discrepancies"]), 2)

    def test_full_prefix_discrepancy_alone_is_not_a_gating_failure(self):
        rows = [
            make_row(arm="qcomem-shared-packed"),
            make_row(
                arm="full-prefix",
                fork_mode="private-materialize",
                identical=False,
            ),
        ]
        report = semantic_equivalence_report(rows)
        self.assertEqual(report["gating_failure_count"], 0)
        self.assertTrue(report["gating_all_identical"])

    def test_contract_report_counts_coverage_and_verdict(self):
        rows = [make_row(), make_row(workload_id="qasper-7")]
        report = contract_report(rows)
        self.assertEqual(report["shared_arm_row_count"], 2)
        self.assertTrue(report["all_targets_covered_everywhere"])
        self.assertTrue(report["all_targets_passed_everywhere"])
        self.assertTrue(report["packed_entry_obligations_passed_everywhere"])
        self.assertEqual(
            report["per_target"]["residual_chunk_binding"]["passed"], 2
        )

    def test_contract_report_flags_a_failing_obligation(self):
        row = make_row()
        failing = passing_target_rows(packed_entry_lifetime=False)
        row["forkaudit"]["target_rows"] = failing
        row["forkaudit"]["contract_summary"] = contract_summary(failing)
        report = contract_report([row])
        self.assertFalse(report["packed_entry_obligations_passed_everywhere"])
        self.assertEqual(report["per_target"]["packed_entry_lifetime"]["open"], 1)

    def test_working_set_table_has_the_transient_columns(self):
        table = working_set_table(make_shard()["rows"])
        self.assertEqual(len(table), 3)
        for row in table:
            for column in (
                "shared_view_mib",
                "transient_materialized_total_mib",
                "peak_transient_allocation_mib",
                "steady_state_resident_mib",
                "resident_slope_mib_per_request",
            ):
                self.assertIn(column, row)

    def test_crossover_excludes_the_reference_arm(self):
        table = working_set_table(make_shard()["rows"])
        report = crossover_report(table, search_limit=100)
        self.assertEqual(len(report), 2)
        self.assertNotIn("full-prefix", [row["arm"] for row in report])
        shared = next(
            row for row in report if row["arm"] == "qcomem-shared-packed"
        )
        # smaller intercept and smaller slope than full prefix: never crosses
        self.assertIsNone(shared["crossover_request_count"])


class AggregateTest(unittest.TestCase):
    def test_clean_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_run([make_shard(0), make_shard(1)], path)
            result = aggregate(path, expected_shards=2)
        self.assertTrue(result["record_complete"], result["defects"])
        self.assertEqual(result["shard_count"], 2)
        self.assertEqual(result["row_count"], 6)
        self.assertTrue(result["semantic_equivalence"]["gating_all_identical"])
        self.assertEqual(result["schema"], AGGREGATE_SCHEMA)

    def test_missing_shard_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_run([make_shard(0)], path)
            with self.assertRaises(AggregateError):
                aggregate(path, expected_shards=2)

    def test_no_shards_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(AggregateError):
                aggregate(Path(directory), expected_shards=None)

    def test_gate_failed_shard_makes_the_record_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            write_run(
                [
                    make_shard(0),
                    make_shard(
                        1,
                        status="gate_failed",
                        gates={"shared_packed_multifork_gate": {"passed": False}},
                        rows=[],
                    ),
                ],
                path,
            )
            result = aggregate(path, expected_shards=2)
        self.assertFalse(result["record_complete"])

    def test_tail_policy_disagreement_is_a_defect(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            second = make_shard(1)
            second["protocol_settings"]["tail_policy"] = "materialized-tail"
            write_run([make_shard(0), second], path)
            result = aggregate(path, expected_shards=2)
        self.assertFalse(result["record_complete"])
        self.assertTrue(
            any("tail policy" in message for message in result["defects"])
        )

    def test_a_negative_scientific_result_still_aggregates(self):
        """A failed ownership predicate is a result, not a broken record."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            shard = make_shard(0)
            failing = passing_target_rows(private_ownership=False)
            shard["rows"][0]["forkaudit"]["target_rows"] = failing
            shard["rows"][0]["forkaudit"]["contract_summary"] = contract_summary(
                failing
            )
            write_run([shard], path)
            result = aggregate(path, expected_shards=1)
        self.assertTrue(result["record_complete"], result["defects"])
        self.assertFalse(
            result["forkaudit_contract"]["all_targets_passed_everywhere"]
        )
        self.assertEqual(
            result["forkaudit_contract"]["per_target"]["private_ownership"]["open"],
            1,
        )

    def test_semantic_discrepancy_is_surfaced_not_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            shard = make_shard(0)
            shard["rows"][0]["semantic_equivalence"] = {
                "token_sequences_identical": False,
                "discrepancies": [
                    {"request_id": "r01", "first_divergence_step": 3}
                ],
            }
            write_run([shard], path)
            result = aggregate(path, expected_shards=1)
        self.assertTrue(result["record_complete"], result["defects"])
        self.assertFalse(result["semantic_equivalence"]["gating_all_identical"])
        self.assertEqual(result["semantic_equivalence"]["gating_failure_count"], 1)

    def test_missing_arm_is_a_defect(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            shard = make_shard(0, rows=[make_row()])
            write_run([shard], path)
            result = aggregate(path, expected_shards=1)
        self.assertFalse(result["record_complete"])
        self.assertTrue(
            any("arm coverage is incomplete" in message for message in result["defects"])
        )


if __name__ == "__main__":
    unittest.main()
