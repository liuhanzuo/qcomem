"""Torch-free tests for the C1 multifork bookkeeping.

Every test here runs on a laptop with no torch, no CUDA and no checkpoint.
They cover the three things the experiment's correctness rests on: the
ownership ledger's byte-range algebra, the transient working-set arithmetic for
both arms, and the coverage-versus-verdict rule that stops a target with a
missing mandatory receipt from silently passing.
"""

from __future__ import annotations

import unittest

from qcomem_multifork_accounting import (
    FORK_MODES,
    MANDATORY_SLOTS,
    MULTIFORK_ARMS,
    MULTIFORK_TARGET_CONTRACT,
    MultiforkAccountingError,
    REBIND_POLICIES,
    TAIL_POLICIES,
    VALID_STATUSES,
    build_multifork_target_rows,
    ceil_div,
    compare_token_traces,
    contract_summary,
    cross_n_prefix_consistency,
    crossover_request_count,
    evaluate_slot_coverage,
    fanout_plan,
    format_mib,
    normalize_inventory,
    ownership_ledger,
    packed_entry_obligation_names,
    range_overlaps,
    request_ids,
    resident_bytes_at_n,
    sharing_efficiency,
    summarize_multifork_rows,
    target_names,
    unique_storage_nbytes,
    validate_multifork_row,
    working_set_row,
)


def good_slot(**overrides):
    slot = {"present": True, "unique": True, "bound": True, "modified": False}
    slot.update(overrides)
    return slot


def all_receipts(**overrides):
    """A receipt map in which every mandatory slot of every target is good."""

    receipts = {}
    for slots in MANDATORY_SLOTS.values():
        for slot in slots:
            receipts[slot] = good_slot()
    receipts.update(overrides)
    return receipts


def all_predicates(value: bool = True, **overrides):
    predicates = {name: value for name in target_names()}
    predicates.update(overrides)
    return predicates


def inventory_row(path, role, storage_id, storage_nbytes, start, end):
    return {
        "path": path,
        "role": role,
        "storage_id": storage_id,
        "storage_nbytes": storage_nbytes,
        "view_start_bytes": start,
        "view_end_bytes": end,
    }


class ContractTableTest(unittest.TestCase):
    def test_ten_targets_seven_inherited_three_packed_entry(self):
        self.assertEqual(len(MULTIFORK_TARGET_CONTRACT), 10)
        self.assertEqual(
            [row["target_index"] for row in MULTIFORK_TARGET_CONTRACT],
            list(range(1, 11)),
        )
        families = [row["family"] for row in MULTIFORK_TARGET_CONTRACT]
        self.assertEqual(families.count("forkaudit-seven"), 7)
        self.assertEqual(families.count("packed-entry-obligation"), 3)

    def test_packed_entry_obligations_are_the_three_the_paper_names(self):
        self.assertEqual(
            packed_entry_obligation_names(),
            (
                "dequantized_view_immutability",
                "residual_chunk_binding",
                "packed_entry_lifetime",
            ),
        )

    def test_every_target_declares_mandatory_slots_and_a_valid_status(self):
        for row in MULTIFORK_TARGET_CONTRACT:
            self.assertIn(row["target"], MANDATORY_SLOTS, row["target"])
            self.assertTrue(MANDATORY_SLOTS[row["target"]], row["target"])
            self.assertIn(row["maximum_status"], VALID_STATUSES)
            self.assertTrue(row["predicate_id"].isupper())

    def test_tail_and_dispatch_targets_are_capped_at_partial(self):
        capped = {
            row["target"]: row["maximum_status"] for row in MULTIFORK_TARGET_CONTRACT
        }
        self.assertEqual(capped["tail_safe_append"], "partial")
        self.assertEqual(capped["dispatch_provenance"], "partial")

    def test_mode_constants(self):
        self.assertEqual(
            FORK_MODES, ("private-materialize", "shared-packed-view")
        )
        self.assertEqual(REBIND_POLICIES, ("setup", "transition"))
        self.assertEqual(TAIL_POLICIES, ("borrowed-prefix", "materialized-tail"))
        self.assertIn("qcomem-shared-packed", MULTIFORK_ARMS)
        self.assertIn("full-prefix", MULTIFORK_ARMS)


class SlotCoverageTest(unittest.TestCase):
    def test_complete_coverage(self):
        result = evaluate_slot_coverage("residual_chunk_binding", all_receipts())
        self.assertEqual(result["coverage"], "complete")
        self.assertEqual(result["coverage_failures"], [])

    def test_missing_slot_is_incomplete(self):
        receipts = all_receipts()
        del receipts["residual_binding_events"]
        result = evaluate_slot_coverage("residual_chunk_binding", receipts)
        self.assertEqual(result["coverage"], "incomplete")
        self.assertEqual(
            result["coverage_failures"],
            [{"slot": "residual_binding_events", "reason": "missing"}],
        )

    def test_unbound_slot_is_incomplete(self):
        receipts = all_receipts(
            residual_binding_events=good_slot(bound=False)
        )
        result = evaluate_slot_coverage("residual_chunk_binding", receipts)
        self.assertEqual(result["coverage"], "incomplete")
        self.assertIn(
            {"slot": "residual_binding_events", "reason": "slot_bound_is_False"},
            result["coverage_failures"],
        )

    def test_modified_slot_is_incomplete(self):
        receipts = all_receipts(setup_inventory=good_slot(modified=True))
        result = evaluate_slot_coverage("private_ownership", receipts)
        self.assertEqual(result["coverage"], "incomplete")

    def test_duplicated_slot_is_incomplete(self):
        receipts = all_receipts(setup_inventory=good_slot(unique=False))
        self.assertEqual(
            evaluate_slot_coverage("private_ownership", receipts)["coverage"],
            "incomplete",
        )

    def test_non_boolean_slot_field_is_incomplete(self):
        receipts = all_receipts(setup_inventory=good_slot(present="yes"))
        result = evaluate_slot_coverage("private_ownership", receipts)
        self.assertEqual(result["coverage"], "incomplete")
        self.assertIn(
            {"slot": "setup_inventory", "reason": "non_boolean_present"},
            result["coverage_failures"],
        )

    def test_unknown_target_raises(self):
        with self.assertRaises(MultiforkAccountingError):
            evaluate_slot_coverage("not_a_target", all_receipts())


class TargetRowTest(unittest.TestCase):
    def test_all_pass_gives_each_target_its_maximum_status(self):
        rows = build_multifork_target_rows(
            predicates=all_predicates(True), receipts=all_receipts()
        )
        by_target = {row["target"]: row for row in rows}
        self.assertEqual(by_target["frozen_identity"]["status"], "full")
        self.assertEqual(by_target["tail_safe_append"]["status"], "partial")
        self.assertEqual(by_target["dispatch_provenance"]["status"], "partial")
        self.assertTrue(all(row["coverage"] == "complete" for row in rows))

    def test_failing_predicate_opens_the_target(self):
        rows = build_multifork_target_rows(
            predicates=all_predicates(True, residual_chunk_binding=False),
            receipts=all_receipts(),
        )
        row = next(r for r in rows if r["target"] == "residual_chunk_binding")
        self.assertEqual(row["status"], "open")
        self.assertIs(row["predicate_passed"], False)
        self.assertEqual(row["coverage"], "complete")

    def test_missing_receipt_cannot_pass_even_with_a_true_predicate(self):
        receipts = all_receipts()
        del receipts["view_alias_inventory"]
        rows = build_multifork_target_rows(
            predicates=all_predicates(True), receipts=receipts
        )
        row = next(
            r for r in rows if r["target"] == "dequantized_view_immutability"
        )
        self.assertEqual(row["coverage"], "incomplete")
        self.assertEqual(row["status"], "open")
        self.assertIsNone(row["predicate_passed"])

    def test_unbound_receipt_cannot_pass(self):
        receipts = all_receipts(append_events=good_slot(bound=False))
        rows = build_multifork_target_rows(
            predicates=all_predicates(True), receipts=receipts
        )
        row = next(r for r in rows if r["target"] == "tail_safe_append")
        self.assertEqual(row["status"], "open")
        self.assertIsNone(row["predicate_passed"])

    def test_covered_target_with_no_predicate_raises(self):
        predicates = all_predicates(True)
        del predicates["packed_entry_lifetime"]
        with self.assertRaises(MultiforkAccountingError):
            build_multifork_target_rows(
                predicates=predicates, receipts=all_receipts()
            )

    def test_non_boolean_predicate_raises(self):
        predicates = all_predicates(True, packed_entry_lifetime="yes")
        with self.assertRaises(MultiforkAccountingError):
            build_multifork_target_rows(
                predicates=predicates, receipts=all_receipts()
            )

    def test_scope_override_reaches_the_row(self):
        rows = build_multifork_target_rows(
            predicates=all_predicates(True),
            receipts=all_receipts(),
            scope_overrides={"residual_chunk_binding": "measured at setup only"},
        )
        row = next(r for r in rows if r["target"] == "residual_chunk_binding")
        self.assertEqual(row["scope_note"], "measured at setup only")

    def test_exact_missingness_is_carried_for_capped_targets(self):
        rows = build_multifork_target_rows(
            predicates=all_predicates(True), receipts=all_receipts()
        )
        row = next(r for r in rows if r["target"] == "dispatch_provenance")
        self.assertIn(
            "compiled CUDA/Triton kernel binary fingerprint",
            row["exact_missingness"],
        )


class ContractSummaryTest(unittest.TestCase):
    def test_all_pass_summary(self):
        rows = build_multifork_target_rows(
            predicates=all_predicates(True), receipts=all_receipts()
        )
        summary = contract_summary(rows)
        self.assertTrue(summary["all_applicable_targets_covered"])
        self.assertTrue(summary["all_applicable_predicates_passed"])
        self.assertTrue(summary["packed_entry_obligations_all_passed"])
        self.assertEqual(summary["overall_contract_status"], "partial")
        self.assertEqual(summary["open_targets"], [])
        self.assertEqual(len(summary["seven_target_status_vector"]), 7)
        self.assertEqual(len(summary["packed_entry_obligation_status_vector"]), 3)

    def test_coverage_gap_is_reported_separately_from_verdict(self):
        receipts = all_receipts()
        del receipts["cross_n_token_traces"]
        rows = build_multifork_target_rows(
            predicates=all_predicates(True), receipts=receipts
        )
        summary = contract_summary(rows)
        self.assertFalse(summary["all_applicable_targets_covered"])
        self.assertFalse(summary["all_applicable_predicates_passed"])
        self.assertEqual(summary["uncovered_targets"], ["cross_n_prefix_consistency"])
        self.assertEqual(summary["open_targets"], ["cross_n_prefix_consistency"])
        self.assertEqual(summary["overall_contract_status"], "open")

    def test_one_failing_obligation_clears_the_obligation_flag(self):
        rows = build_multifork_target_rows(
            predicates=all_predicates(True, packed_entry_lifetime=False),
            receipts=all_receipts(),
        )
        summary = contract_summary(rows)
        self.assertFalse(summary["packed_entry_obligations_all_passed"])
        self.assertFalse(summary["all_applicable_predicates_passed"])
        self.assertTrue(summary["all_applicable_targets_covered"])

    def test_empty_rows_raise(self):
        with self.assertRaises(MultiforkAccountingError):
            contract_summary([])


class InventoryTest(unittest.TestCase):
    def test_normalize_computes_view_nbytes_and_sorts(self):
        rows = normalize_inventory(
            [
                inventory_row("b", "r1", "s2", 100, 0, 40),
                inventory_row("a", "r1", "s1", 100, 10, 50),
            ]
        )
        self.assertEqual([row["path"] for row in rows], ["a", "b"])
        self.assertEqual(rows[0]["view_nbytes"], 40)

    def test_out_of_bounds_range_raises(self):
        with self.assertRaises(MultiforkAccountingError):
            normalize_inventory([inventory_row("a", "r", "s", 10, 0, 20)])

    def test_inverted_range_raises(self):
        with self.assertRaises(MultiforkAccountingError):
            normalize_inventory([inventory_row("a", "r", "s", 10, 8, 4)])

    def test_missing_field_raises(self):
        with self.assertRaises(MultiforkAccountingError):
            normalize_inventory([{"path": "a", "role": "r"}])

    def test_unique_storage_nbytes_deduplicates_like_cache_nbytes(self):
        rows = normalize_inventory(
            [
                inventory_row("a", "r", "s1", 100, 0, 50),
                inventory_row("b", "r", "s1", 100, 50, 100),
                inventory_row("c", "r", "s2", 30, 0, 30),
            ]
        )
        self.assertEqual(unique_storage_nbytes(rows), 130)

    def test_empty_views_never_alias(self):
        left = normalize_inventory([inventory_row("a", "l", "s", 100, 10, 10)])
        right = normalize_inventory([inventory_row("b", "r", "s", 100, 0, 100)])
        self.assertEqual(range_overlaps(left, right), [])

    def test_partial_overlap_is_reported_with_its_range(self):
        left = normalize_inventory([inventory_row("a", "l", "s", 100, 0, 60)])
        right = normalize_inventory([inventory_row("b", "r", "s", 100, 40, 100)])
        overlap = range_overlaps(left, right)
        self.assertEqual(len(overlap), 1)
        self.assertEqual(overlap[0]["intersection_start_bytes"], 40)
        self.assertEqual(overlap[0]["intersection_end_bytes"], 60)
        self.assertEqual(overlap[0]["intersection_nbytes"], 20)

    def test_different_storages_never_alias(self):
        left = normalize_inventory([inventory_row("a", "l", "s1", 100, 0, 100)])
        right = normalize_inventory([inventory_row("b", "r", "s2", 100, 0, 100)])
        self.assertEqual(range_overlaps(left, right), [])

    def test_adjacent_ranges_do_not_alias(self):
        left = normalize_inventory([inventory_row("a", "l", "s", 100, 0, 50)])
        right = normalize_inventory([inventory_row("b", "r", "s", 100, 50, 100)])
        self.assertEqual(range_overlaps(left, right), [])


class OwnershipLedgerTest(unittest.TestCase):
    def shared(self):
        return [inventory_row("view/keys", "shared_view", "doc", 1000, 0, 1000)]

    def requests(self, overlap: bool = False):
        left = [
            inventory_row("r0/keys", "r0", "doc", 1000, 0, 1000),
            inventory_row("r0/tail", "r0", "tail0", 200, 0, 200),
        ]
        right = [
            inventory_row("r1/keys", "r1", "doc", 1000, 0, 1000),
            inventory_row(
                "r1/tail", "r1", "tail0" if overlap else "tail1", 200, 0, 200
            ),
        ]
        return {"r00": left, "r01": right}

    def test_shared_and_private_split(self):
        ledger = ownership_ledger(
            shared_inventory=self.shared(), request_inventories=self.requests()
        )
        self.assertEqual(ledger["shared_entry_nbytes"], 1000)
        self.assertEqual(ledger["per_request"]["r00"]["shared_nbytes"], 1000)
        self.assertEqual(ledger["per_request"]["r00"]["private_nbytes"], 200)
        self.assertEqual(ledger["total_private_nbytes"], 400)
        self.assertTrue(ledger["passed"])
        self.assertTrue(ledger["non_vacuous"])

    def test_overlapping_private_state_fails(self):
        ledger = ownership_ledger(
            shared_inventory=self.shared(),
            request_inventories=self.requests(overlap=True),
        )
        self.assertFalse(ledger["passed"])
        self.assertEqual(len(ledger["pairwise"]), 1)
        self.assertEqual(len(ledger["pairwise"][0]["overlap_ranges"]), 1)

    def test_single_request_is_vacuous_and_does_not_pass(self):
        ledger = ownership_ledger(
            shared_inventory=self.shared(),
            request_inventories={"r00": self.requests()["r00"]},
        )
        self.assertFalse(ledger["non_vacuous"])
        self.assertFalse(ledger["passed"])
        self.assertEqual(ledger["pairwise_comparison_count"], 0)

    def test_private_only_arm_reports_zero_shared(self):
        ledger = ownership_ledger(
            shared_inventory=[], request_inventories=self.requests()
        )
        self.assertEqual(ledger["shared_entry_nbytes"], 0)
        self.assertEqual(ledger["per_request"]["r00"]["shared_nbytes"], 0)
        # both requests now claim the same "doc" storage privately
        self.assertFalse(ledger["passed"])

    def test_sharing_efficiency_counts_copies_avoided(self):
        ledger = ownership_ledger(
            shared_inventory=self.shared(), request_inventories=self.requests()
        )
        efficiency = sharing_efficiency(ledger)
        self.assertEqual(efficiency["resident_nbytes"], 1400)
        self.assertEqual(efficiency["n_private_copies_equivalent_nbytes"], 2400)
        self.assertEqual(efficiency["copies_avoided_nbytes"], 1000)

    def test_sharing_one_request_avoids_nothing(self):
        ledger = ownership_ledger(
            shared_inventory=self.shared(),
            request_inventories={"r00": self.requests()["r00"]},
        )
        self.assertEqual(sharing_efficiency(ledger)["copies_avoided_nbytes"], 0)


class WorkingSetTest(unittest.TestCase):
    def row(self, **overrides):
        kwargs = dict(
            arm="qcomem-shared-packed",
            request_count=2,
            entry_retained_nbytes=1000,
            shared_view_nbytes=4000,
            per_request_materialized_nbytes=[100, 300],
            per_request_steady_resident_nbytes=[150, 250],
            measured_baseline_allocated_nbytes=10_000,
            measured_peak_allocated_nbytes=18_000,
            measured_steady_allocated_nbytes=12_000,
        )
        kwargs.update(overrides)
        return working_set_row(**kwargs)

    def test_transient_fields_are_first_class(self):
        row = self.row()
        for field in (
            "shared_dequantized_view_nbytes",
            "per_request_materialized_nbytes",
            "transient_materialized_nbytes_total",
            "peak_transient_allocation_nbytes",
            "steady_state_resident_nbytes",
        ):
            self.assertIn(field, row)
        self.assertEqual(row["transient_materialized_nbytes_total"], 400)
        self.assertEqual(row["transient_materialized_nbytes_max"], 300)
        self.assertEqual(row["peak_transient_allocation_nbytes"], 8000)
        self.assertEqual(row["steady_state_resident_delta_nbytes"], 2000)

    def test_resident_model_intercept_and_slope(self):
        row = self.row()
        self.assertEqual(row["resident_model"]["intercept_nbytes"], 5000)
        self.assertEqual(row["resident_model"]["slope_nbytes_per_request"], 200)
        self.assertEqual(row["modelled_resident_nbytes"], 5400)

    def test_full_prefix_arm_has_no_shared_view(self):
        row = self.row(arm="full-prefix", shared_view_nbytes=0)
        self.assertEqual(row["shared_dequantized_view_nbytes"], 0)
        self.assertEqual(row["resident_model"]["intercept_nbytes"], 1000)

    def test_peak_below_baseline_clamps_to_zero(self):
        row = self.row(measured_peak_allocated_nbytes=5_000)
        self.assertEqual(row["peak_transient_allocation_nbytes"], 0)

    def test_length_mismatch_raises(self):
        with self.assertRaises(MultiforkAccountingError):
            self.row(per_request_materialized_nbytes=[100])

    def test_zero_requests_raise(self):
        with self.assertRaises(MultiforkAccountingError):
            self.row(
                request_count=0,
                per_request_materialized_nbytes=[],
                per_request_steady_resident_nbytes=[],
            )

    def test_resident_bytes_at_n(self):
        self.assertEqual(
            resident_bytes_at_n(
                intercept_nbytes=100, slope_nbytes_per_request=10, request_count=4
            ),
            140,
        )
        with self.assertRaises(MultiforkAccountingError):
            resident_bytes_at_n(
                intercept_nbytes=0, slope_nbytes_per_request=0, request_count=-1
            )


class CrossoverTest(unittest.TestCase):
    def test_smaller_intercept_and_slope_never_crosses(self):
        result = crossover_request_count(
            left={"intercept_nbytes": 10, "slope_nbytes_per_request": 1},
            right={"intercept_nbytes": 100, "slope_nbytes_per_request": 5},
            max_request_count=1000,
        )
        self.assertIsNone(result["crossover_request_count"])
        self.assertEqual(result["smaller_intercept"], "left")
        self.assertEqual(result["smaller_slope"], "left")

    def test_larger_slope_crosses_at_the_right_request_count(self):
        # left = 10 + 10N, right = 100 + 1N -> left exceeds right at N = 11
        result = crossover_request_count(
            left={"intercept_nbytes": 10, "slope_nbytes_per_request": 10},
            right={"intercept_nbytes": 100, "slope_nbytes_per_request": 1},
            max_request_count=1000,
        )
        self.assertEqual(result["crossover_request_count"], 11)

    def test_search_limit_is_reported(self):
        result = crossover_request_count(
            left={"intercept_nbytes": 10, "slope_nbytes_per_request": 10},
            right={"intercept_nbytes": 100_000, "slope_nbytes_per_request": 1},
            max_request_count=5,
        )
        self.assertIsNone(result["crossover_request_count"])
        self.assertEqual(result["searched_up_to"], 5)


class TokenTraceTest(unittest.TestCase):
    def test_identical_traces(self):
        result = compare_token_traces(
            reference={"r00": [1, 2, 3], "r01": [4, 5]},
            candidate={"r00": [1, 2, 3], "r01": [4, 5]},
            reference_label="a",
            candidate_label="b",
        )
        self.assertTrue(result["token_sequences_identical"])
        self.assertEqual(result["discrepancies"], [])
        self.assertEqual(result["identical_request_count"], 2)

    def test_divergence_step_is_recorded(self):
        result = compare_token_traces(
            reference={"r00": [1, 2, 3]},
            candidate={"r00": [1, 9, 3]},
            reference_label="a",
            candidate_label="b",
        )
        self.assertFalse(result["token_sequences_identical"])
        self.assertEqual(result["discrepancies"][0]["first_divergence_step"], 1)

    def test_length_difference_is_a_divergence(self):
        result = compare_token_traces(
            reference={"r00": [1, 2, 3]},
            candidate={"r00": [1, 2]},
            reference_label="a",
            candidate_label="b",
        )
        self.assertFalse(result["token_sequences_identical"])
        self.assertEqual(result["discrepancies"][0]["first_divergence_step"], 2)

    def test_missing_request_is_a_discrepancy_not_a_skip(self):
        result = compare_token_traces(
            reference={"r00": [1], "r01": [2]},
            candidate={"r00": [1]},
            reference_label="a",
            candidate_label="b",
        )
        self.assertFalse(result["token_sequences_identical"])
        row = next(
            r for r in result["discrepancies"] if r["request_id"] == "r01"
        )
        self.assertFalse(row["present_in_candidate"])

    def test_empty_comparison_raises(self):
        with self.assertRaises(MultiforkAccountingError):
            compare_token_traces(
                reference={},
                candidate={},
                reference_label="a",
                candidate_label="b",
            )


class CrossNTest(unittest.TestCase):
    def test_two_matching_fanouts_pass(self):
        result = cross_n_prefix_consistency(
            {1: {"r00": [1, 2]}, 4: {"r00": [1, 2], "r01": [9]}},
            prefix_request_id="r00",
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["non_vacuous"])

    def test_single_fanout_is_vacuous_and_does_not_pass(self):
        result = cross_n_prefix_consistency(
            {2: {"r00": [1, 2]}}, prefix_request_id="r00"
        )
        self.assertFalse(result["non_vacuous"])
        self.assertFalse(result["passed"])

    def test_mismatch_is_reported(self):
        result = cross_n_prefix_consistency(
            {1: {"r00": [1, 2]}, 2: {"r00": [1, 3]}}, prefix_request_id="r00"
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["mismatches"][0]["fanout"], 2)

    def test_missing_prefix_request_fails(self):
        result = cross_n_prefix_consistency(
            {1: {"r00": [1]}, 2: {"r01": [1]}}, prefix_request_id="r00"
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["missing_fanouts"], [2])

    def test_no_fanouts_raise(self):
        with self.assertRaises(MultiforkAccountingError):
            cross_n_prefix_consistency({}, prefix_request_id="r00")


def valid_row(**overrides):
    row = {
        "arm": "qcomem-shared-packed",
        "fork_mode": "shared-packed-view",
        "request_count": 2,
        "workload_id": "qasper-6",
        "entry_retained_nbytes": 1000,
        "shared_dequantized_view_nbytes": 4000,
        "per_request_materialized_nbytes": [100, 200],
        "transient_materialized_nbytes_total": 300,
        "peak_transient_allocation_nbytes": 900,
        "steady_state_resident_nbytes": 5000,
        "per_request_steady_resident_nbytes": [150, 250],
        "resident_model": {
            "intercept_nbytes": 5000,
            "slope_nbytes_per_request": 200,
        },
        "ownership_ledger": {"request_count": 2},
        "semantic_equivalence": {"token_sequences_identical": True},
    }
    row.update(overrides)
    return row


class RowValidationTest(unittest.TestCase):
    def test_valid_row(self):
        self.assertEqual(validate_multifork_row(valid_row()), [])

    def test_missing_field(self):
        row = valid_row()
        del row["peak_transient_allocation_nbytes"]
        self.assertIn(
            "missing field: peak_transient_allocation_nbytes",
            validate_multifork_row(row),
        )

    def test_unknown_arm_and_mode(self):
        problems = validate_multifork_row(
            valid_row(arm="mystery", fork_mode="mystery")
        )
        self.assertIn("unknown arm: mystery", problems)
        self.assertIn("unknown fork mode: mystery", problems)

    def test_per_request_list_length_must_match(self):
        problems = validate_multifork_row(
            valid_row(per_request_materialized_nbytes=[1])
        )
        self.assertTrue(
            any("has 1 entries for 2 requests" in message for message in problems)
        )

    def test_negative_bytes_rejected(self):
        problems = validate_multifork_row(valid_row(entry_retained_nbytes=-1))
        self.assertIn(
            "entry_retained_nbytes must be a non-negative integer", problems
        )

    def test_ledger_request_count_must_agree(self):
        problems = validate_multifork_row(
            valid_row(ownership_ledger={"request_count": 3})
        )
        self.assertIn(
            "ownership_ledger request_count disagrees with the row", problems
        )

    def test_validator_does_not_require_a_passing_result(self):
        """A failed run must still emit a well-formed row.

        The validator checks the schema, not the science: a row whose semantic
        equivalence failed and whose sharing saved nothing is still valid.
        """

        row = valid_row(
            semantic_equivalence={"token_sequences_identical": False},
            shared_dequantized_view_nbytes=0,
        )
        self.assertEqual(validate_multifork_row(row), [])


class SummaryTest(unittest.TestCase):
    def test_grouping_and_failure_counts(self):
        rows = [
            valid_row(workload_id="a"),
            valid_row(
                workload_id="b",
                semantic_equivalence={"token_sequences_identical": False},
            ),
            valid_row(arm="full-prefix", fork_mode="private-materialize"),
        ]
        summaries = summarize_multifork_rows(rows)
        by_arm = {row["arm"]: row for row in summaries}
        self.assertEqual(by_arm["qcomem-shared-packed"]["row_count"], 2)
        self.assertEqual(
            by_arm["qcomem-shared-packed"]["semantic_equivalence_failures"], 1
        )
        self.assertFalse(
            by_arm["qcomem-shared-packed"]["semantic_equivalence_all_identical"]
        )
        self.assertEqual(by_arm["full-prefix"]["row_count"], 1)

    def test_discrepant_rows_are_summarized_not_dropped(self):
        rows = [
            valid_row(semantic_equivalence={"token_sequences_identical": False})
        ]
        summaries = summarize_multifork_rows(rows)
        self.assertEqual(summaries[0]["row_count"], 1)
        self.assertEqual(summaries[0]["steady_state_resident_nbytes_median"], 5000)


class HelperTest(unittest.TestCase):
    def test_fanout_plan_requires_a_multifork_cell(self):
        self.assertEqual(fanout_plan([4, 1, 2]), [1, 2, 4])
        with self.assertRaises(MultiforkAccountingError):
            fanout_plan([1])
        self.assertEqual(fanout_plan([1], require_multifork=False), [1])

    def test_fanout_plan_rejects_bad_values(self):
        with self.assertRaises(MultiforkAccountingError):
            fanout_plan([])
        with self.assertRaises(MultiforkAccountingError):
            fanout_plan([0, 2])

    def test_request_ids_are_stable_and_sorted(self):
        self.assertEqual(request_ids(3), ["r00", "r01", "r02"])
        self.assertEqual(request_ids(11)[10], "r10")
        self.assertEqual(sorted(request_ids(11)), request_ids(11))

    def test_ceil_div_and_format_mib(self):
        self.assertEqual(ceil_div(65, 64), 2)
        self.assertEqual(ceil_div(64, 64), 1)
        self.assertEqual(format_mib(2**20), 1.0)
        with self.assertRaises(MultiforkAccountingError):
            ceil_div(1, 0)


if __name__ == "__main__":
    unittest.main()
