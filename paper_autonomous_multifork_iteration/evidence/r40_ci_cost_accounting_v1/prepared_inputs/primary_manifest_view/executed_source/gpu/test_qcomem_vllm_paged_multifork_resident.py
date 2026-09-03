from __future__ import annotations

import unittest
from types import MethodType
from types import SimpleNamespace

import torch

from qcomem_qwen35_vllm_paged_integration import (
    convert_all_qwen35_full_layers_to_vllm_q16,
)
from qcomem_vllm_paged_fair_control import FRESH_CONTROL, SHARED_REUSE
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    GDN_MATERIALIZE_REQUEST_BASE,
    MULTIFORK_COUNTS,
    MULTIFORK_PROTOCOL,
    MultiForkHitLedger,
    QComemMultiForkError,
    RuntimeInvariantError,
    build_deterministic_distinct_queries,
    build_pg19_train_query_bank,
    build_resident_request_group,
    linear_capacity_fit,
    resident_storage_breakdown,
    source_document_physical_digests,
    strict_group_logical_parity,
    validate_resident_group_ownership,
    validate_runtime_kv_ownership,
)


class LinearLayer:
    def __init__(self) -> None:
        self.conv_states = {0: torch.zeros(1, 2, 3)}
        self.recurrent_states = {0: torch.zeros(1, 2, 4, 4)}
        self.is_conv_states_initialized = {0: True}
        self.is_recurrent_states_initialized = {0: True}
        self.has_previous_state = {0: True}
        self.conv_kernel_size = {0: 3}
        self.record_past = False

    def lazy_initialization(self, **kwargs):
        del kwargs


class DenseFullLayer:
    is_sliding = False

    def __init__(self, seed: int, length: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.keys = torch.randn(1, 2, length, 32, generator=generator)
        self.values = torch.randn(1, 2, length, 32, generator=generator)


def make_cache_and_plan(*, length: int, forks: int, max_append_tokens: int = 8):
    layer_types = [
        "full_attention" if (index + 1) % 4 == 0 else "linear_attention"
        for index in range(40)
    ]
    full = tuple(index for index, kind in enumerate(layer_types) if kind == "full_attention")
    linear = tuple(index for index, kind in enumerate(layer_types) if kind == "linear_attention")
    cache = SimpleNamespace(
        layers=[
            DenseFullLayer(1000 + index, length)
            if kind == "full_attention"
            else LinearLayer()
            for index, kind in enumerate(layer_types)
        ]
    )
    config = SimpleNamespace(layer_types=layer_types)
    plan = SimpleNamespace(
        full_attention_layer_indices=full,
        linear_layer_indices=linear,
        gdn=config,
    )
    conversion = convert_all_qwen35_full_layers_to_vllm_q16(
        cache,
        plan,
        page_size=16,
        max_append_tokens=max_append_tokens,
        max_request_forks=forks,
    )
    return cache, plan, conversion


class MarkerTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        digits = [ord(character) - ord("0") for character in text if character.isdigit()]
        return [71, 72, *(100 + digit for digit in digits), 73]


class ConstantTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del text, add_special_tokens
        return [1, 2, 3]


class RawTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False):
        del add_special_tokens
        return [index % 997 for index, _character in enumerate(text)]


class ZeroKernel:
    def __call__(self, **kwargs):
        kwargs["out"].zero_()


class MultiForkResidentTest(unittest.TestCase):
    def test_runtime_kv_guard_rejects_whole_request_sequence_swap(self):
        cache, plan, _ = make_cache_and_plan(length=32, forks=2)
        group = build_resident_request_group(
            cache, plan, resident_count=2, policy=SHARED_REUSE
        )
        layer_index = plan.full_attention_layer_indices[0]
        left = group.requests[0].layers[layer_index]
        right = group.requests[1].layers[layer_index]
        left.sequence, right.sequence = right.sequence, left.sequence

        with self.assertRaisesRegex(RuntimeInvariantError, "KV_SEQUENCE_ID"):
            validate_runtime_kv_ownership(
                cache,
                group,
                plan,
                require_appended_tail_cow=False,
            )

        prefix_cache, prefix_plan, _ = make_cache_and_plan(length=32, forks=2)
        prefix_group = build_resident_request_group(
            prefix_cache, prefix_plan, resident_count=2, policy=SHARED_REUSE
        )
        prefix_sequence = prefix_group.requests[0].layers[layer_index].sequence
        prefix_sequence.active_block_table[0, 0] = (
            prefix_sequence.active_block_table[0, 1]
        )
        with self.assertRaisesRegex(RuntimeInvariantError, "KV_SEQUENCE_ID"):
            validate_runtime_kv_ownership(
                prefix_cache,
                prefix_group,
                prefix_plan,
                require_appended_tail_cow=False,
            )

    @staticmethod
    def _storage_key(tensor: torch.Tensor):
        storage = tensor.untyped_storage()
        return str(tensor.device), storage.data_ptr(), storage.nbytes()

    def test_queries_are_deterministic_fixed_length_and_pairwise_distinct(self):
        base = torch.arange(24, dtype=torch.long).unsqueeze(0)
        first, audit = build_deterministic_distinct_queries(
            base, MarkerTokenizer(), count=32, query_tokens=32
        )
        second, second_audit = build_deterministic_distinct_queries(
            base, MarkerTokenizer(), count=32, query_tokens=32
        )
        self.assertTrue(audit["pairwise_distinct"])
        self.assertEqual(audit, second_audit)
        self.assertEqual(len(first), 32)
        self.assertTrue(all(tuple(value.shape) == (1, 32) for value in first))
        self.assertTrue(all(torch.equal(left, right) for left, right in zip(first, second)))
        self.assertEqual(len({row["query_token_ids_sha256"] for row in audit["rows"]}), 32)

    def test_duplicate_query_content_fails_closed(self):
        with self.assertRaisesRegex(QComemMultiForkError, "pairwise distinct"):
            build_deterministic_distinct_queries(
                torch.arange(32).unsqueeze(0),
                ConstantTokenizer(),
                count=4,
                query_tokens=32,
            )

    def test_formal_query_bank_uses_nonoverlapping_raw_pg19_train_chunks(self):
        source = "train/123.txt"
        records = [{"id": "123", "_source_object": source, "text": "x" * 10000}]
        window = SimpleNamespace(
            source_id="123",
            source_object=source,
            start_token=17,
        )
        queries, audit = build_pg19_train_query_bank(
            records,
            RawTokenizer(),
            window,
            document_tokens=4095,
            query_tokens=32,
            count=32,
            query_stride=64,
        )
        self.assertEqual(len(queries), 32)
        self.assertFalse(audit["synthetic_markers_used"])
        self.assertTrue(audit["pairwise_nonoverlapping"])
        self.assertEqual(audit["query_bank_start_token"], 17 + 4095 + 32)
        self.assertEqual(
            [row["source_token_offset"] for row in audit["rows"]],
            [17 + 4095 + 32 + index * 64 for index in range(32)],
        )
        self.assertEqual(len({row["query_token_ids_sha256"] for row in audit["rows"]}), 32)

    def test_fresh_and_reuse_hold_all_requests_with_exact_ownership(self):
        cache, plan, conversion = make_cache_and_plan(length=35, forks=4)
        fresh = build_resident_request_group(
            cache, plan, resident_count=4, policy=FRESH_CONTROL
        )
        reuse = build_resident_request_group(
            cache, plan, resident_count=4, policy=SHARED_REUSE
        )
        self.assertEqual(len(fresh.requests), 4)
        self.assertEqual(len(reuse.requests), 4)
        self.assertTrue(fresh.audit["all_requests_materialized_before_measurement"])
        self.assertTrue(reuse.audit["all_requests_materialized_before_measurement"])
        self.assertTrue(
            fresh.audit["ownership"]["fresh_request_arena_storages_pairwise_disjoint"]
        )
        self.assertTrue(reuse.audit["ownership"]["reuse_requests_share_source_arena"])
        self.assertTrue(
            reuse.audit["ownership"]["private_physical_reservation_ids_pairwise_disjoint"]
        )
        expected_document_copy = 4 * sum(
            cache.layers[index].arena.batch_size
            * cache.layers[index].arena.document_blocks_per_sequence
            * 2
            * cache.layers[index].arena.page_size
            * cache.layers[index].arena.num_key_value_heads
            * cache.layers[index].arena.head_dim
            * cache.layers[index].arena.key_cache.element_size()
            for index in plan.full_attention_layer_indices
        )
        self.assertEqual(
            fresh.audit["physical_document_block_copy_nbytes_including_padding"],
            expected_document_copy,
        )
        self.assertEqual(reuse.audit["physical_document_block_copy_nbytes_including_padding"], 0)
        self.assertEqual(conversion.max_request_forks, 4)

    def test_gdn_base_factor_changes_only_setup_ownership(self):
        borrowed_cache, borrowed_plan, _ = make_cache_and_plan(length=35, forks=2)
        borrowed = build_resident_request_group(
            borrowed_cache,
            borrowed_plan,
            resident_count=2,
            policy=SHARED_REUSE,
            gdn_base_policy=GDN_BORROW_IMMUTABLE_BASE,
        )
        materialized_cache, materialized_plan, _ = make_cache_and_plan(
            length=35, forks=2
        )
        materialized = build_resident_request_group(
            materialized_cache,
            materialized_plan,
            resident_count=2,
            policy=SHARED_REUSE,
            gdn_base_policy=GDN_MATERIALIZE_REQUEST_BASE,
        )
        self.assertEqual(
            borrowed.audit["gdn_base_policy"], GDN_BORROW_IMMUTABLE_BASE
        )
        self.assertEqual(
            materialized.audit["gdn_base_policy"], GDN_MATERIALIZE_REQUEST_BASE
        )
        for cache, plan, group, should_alias in (
            (borrowed_cache, borrowed_plan, borrowed, True),
            (materialized_cache, materialized_plan, materialized, False),
        ):
            for request in group.requests:
                for layer_index in plan.linear_layer_indices:
                    for family in ("conv_states", "recurrent_states"):
                        source = getattr(cache.layers[layer_index], family)[0]
                        target = getattr(request.layers[layer_index], family)[0]
                        self.assertEqual(
                            self._storage_key(source) == self._storage_key(target),
                            should_alias,
                        )
        self.assertTrue(
            all(
                row["gdn_base"]["borrowed_immutable_base_alias_count"] == 60
                for row in borrowed.audit["rows"]
            )
        )
        self.assertTrue(
            all(
                row["gdn_base"]["borrowed_immutable_base_alias_count"] == 0
                and row["gdn_base"]["materialized_request_base_nbytes"] > 0
                for row in materialized.audit["rows"]
            )
        )

    def test_true_four_cell_helper_factor_isolation(self):
        cells = {}
        for kv_policy in (FRESH_CONTROL, SHARED_REUSE):
            for gdn_policy in (
                GDN_BORROW_IMMUTABLE_BASE,
                GDN_MATERIALIZE_REQUEST_BASE,
            ):
                cache, plan, _ = make_cache_and_plan(length=35, forks=2)
                group = build_resident_request_group(
                    cache,
                    plan,
                    resident_count=2,
                    policy=kv_policy,
                    gdn_base_policy=gdn_policy,
                )
                cells[(kv_policy, gdn_policy)] = (cache, plan, group)
                expect_alias = gdn_policy == GDN_BORROW_IMMUTABLE_BASE
                observed_aliases = []
                for request in group.requests:
                    for layer_index in plan.linear_layer_indices:
                        for family in ("conv_states", "recurrent_states"):
                            observed_aliases.append(
                                self._storage_key(
                                    getattr(cache.layers[layer_index], family)[0]
                                )
                                == self._storage_key(
                                    getattr(request.layers[layer_index], family)[0]
                                )
                            )
                self.assertEqual(observed_aliases, [expect_alias] * 120)
                self.assertEqual(group.audit["gdn_base_policy"], gdn_policy)

        for gdn_policy in (
            GDN_BORROW_IMMUTABLE_BASE,
            GDN_MATERIALIZE_REQUEST_BASE,
        ):
            fresh_cache, fresh_plan, fresh = cells[(FRESH_CONTROL, gdn_policy)]
            reuse_cache, reuse_plan, reuse = cells[(SHARED_REUSE, gdn_policy)]
            self.assertGreater(
                fresh.audit["physical_document_block_copy_nbytes_including_padding"],
                0,
            )
            self.assertEqual(
                reuse.audit["physical_document_block_copy_nbytes_including_padding"],
                0,
            )
            self.assertTrue(
                fresh.audit["ownership"][
                    "fresh_request_arena_storages_pairwise_disjoint"
                ]
            )
            self.assertTrue(
                reuse.audit["ownership"]["reuse_requests_share_source_arena"]
            )
            self.assertEqual(
                [row["gdn_base"] for row in fresh.audit["rows"]],
                [row["gdn_base"] for row in reuse.audit["rows"]],
            )
            for request_index in range(2):
                for layer_index in fresh_plan.linear_layer_indices:
                    self.assertEqual(
                        layer_index in reuse_plan.linear_layer_indices,
                        True,
                    )
                    for family in ("conv_states", "recurrent_states"):
                        self.assertTrue(
                            torch.equal(
                                getattr(
                                    fresh.requests[request_index].layers[layer_index],
                                    family,
                                )[0],
                                getattr(
                                    reuse.requests[request_index].layers[layer_index],
                                    family,
                                )[0],
                            )
                        )
                for layer_index in fresh_plan.linear_layer_indices:
                    for family in ("conv_states", "recurrent_states"):
                        self.assertTrue(
                            torch.equal(
                                getattr(fresh_cache.layers[layer_index], family)[0],
                                getattr(reuse_cache.layers[layer_index], family)[0],
                            )
                        )

        for kv_policy in (FRESH_CONTROL, SHARED_REUSE):
            borrowed_group = cells[(kv_policy, GDN_BORROW_IMMUTABLE_BASE)][2]
            materialized_group = cells[
                (kv_policy, GDN_MATERIALIZE_REQUEST_BASE)
            ][2]

            def kv_only_audit(group):
                return {
                    "protocol": group.audit["protocol"],
                    "policy": group.audit["policy"],
                    "resident_count": group.audit["resident_count"],
                    "ownership": group.audit["ownership"],
                    "physical_copy": group.audit[
                        "physical_document_block_copy_nbytes_including_padding"
                    ],
                    "allocated_pool": group.audit[
                        "allocated_fresh_request_pool_nbytes"
                    ],
                    "rows": [
                        {key: value for key, value in row.items() if key != "gdn_base"}
                        for row in group.audit["rows"]
                    ],
                }

            self.assertEqual(
                kv_only_audit(borrowed_group),
                kv_only_audit(materialized_group),
            )
    def test_append_keeps_per_request_logical_payload_exact_and_storage_formula_closed(self):
        cache, plan, conversion = make_cache_and_plan(length=35, forks=4)
        fresh = build_resident_request_group(
            cache, plan, resident_count=4, policy=FRESH_CONTROL
        )
        reuse = build_resident_request_group(
            cache, plan, resident_count=4, policy=SHARED_REUSE
        )
        for request_index in range(4):
            generator = torch.Generator().manual_seed(8000 + request_index)
            for layer_index in plan.full_attention_layer_indices:
                key = torch.randn(1, 2, 7, 32, generator=generator)
                value = torch.randn(1, 2, 7, 32, generator=generator)
                fresh.requests[request_index].layers[layer_index].update(key, value)
                reuse.requests[request_index].layers[layer_index].update(key, value)
        parity = strict_group_logical_parity(fresh, reuse, plan.full_attention_layer_indices)
        self.assertTrue(parity["passed"])
        self.assertEqual(parity["row_count"], 40)
        fresh_storage = resident_storage_breakdown(cache, fresh, plan)
        reuse_storage = resident_storage_breakdown(cache, reuse, plan)
        fresh_totals = fresh_storage["totals"]
        reuse_totals = reuse_storage["totals"]
        self.assertEqual(
            fresh_totals["source_total_arena_allocated_nbytes"],
            conversion.allocated_block_pool_nbytes,
        )
        self.assertEqual(
            reuse_totals["source_total_arena_allocated_nbytes"],
            conversion.allocated_block_pool_nbytes,
        )
        self.assertEqual(
            fresh_totals["fresh_duplicate_document_allocated_nbytes"],
            fresh.audit["physical_document_block_copy_nbytes_including_padding"],
        )
        self.assertEqual(reuse_totals["fresh_duplicate_document_allocated_nbytes"], 0)
        self.assertEqual(
            fresh_totals["active_request_private_payload_nbytes"],
            reuse_totals["active_request_private_payload_nbytes"],
        )
        self.assertLessEqual(
            reuse_totals["active_request_private_payload_nbytes"],
            reuse_totals["active_request_private_allocated_page_nbytes"],
        )
        self.assertTrue(fresh_storage["active_private_payload_is_subset_not_additive"])

    def test_reuse_overlap_and_missing_strong_reference_fail_closed(self):
        cache, plan, _ = make_cache_and_plan(length=35, forks=4)
        reuse = build_resident_request_group(
            cache, plan, resident_count=4, policy=SHARED_REUSE
        )
        with self.assertRaisesRegex(QComemMultiForkError, "lost a strong"):
            validate_resident_group_ownership(
                cache,
                reuse.requests[:-1],
                plan,
                resident_count=4,
                policy=SHARED_REUSE,
            )
        layer_index = plan.full_attention_layer_indices[0]
        reuse.requests[1].layers[layer_index].sequence.reservations = (
            reuse.requests[0].layers[layer_index].sequence.reservations
        )
        with self.assertRaisesRegex(QComemMultiForkError, "reservations overlap"):
            validate_resident_group_ownership(
                cache,
                reuse.requests,
                plan,
                resident_count=4,
                policy=SHARED_REUSE,
            )

    def test_multifork_ledger_uses_one_kernel_and_exact_call_budget(self):
        cache, plan, _ = make_cache_and_plan(length=32, forks=2)
        group = build_resident_request_group(
            cache, plan, resident_count=2, policy=SHARED_REUSE
        )
        for request in group.requests:
            for layer_index in plan.full_attention_layer_indices:
                request.layers[layer_index].sequence.strict_mask_check = False
        kernel = ZeroKernel()
        ledgers = [
            MultiForkHitLedger(
                plan,
                request,
                request_index=request_index,
                resident_count=2,
                request_policy=SHARED_REUSE,
                expected_calls_per_layer=1,
                initial_query_tokens=1,
                kernel=kernel,
            )
            for request_index, request in enumerate(group.requests)
        ]
        self.assertTrue(all(ledger.kernel is kernel for ledger in ledgers))
        for request_index, (request, ledger) in enumerate(zip(group.requests, ledgers)):
            for layer_index in plan.full_attention_layer_indices:
                layer = request.layers[layer_index]
                key = torch.full((1, 2, 1, 32), float(request_index + 1))
                value = key.clone()
                layer.update(key, value)
                module = SimpleNamespace(
                    layer_idx=layer_index,
                    is_causal=True,
                    num_key_value_groups=8,
                    scaling=32**-0.5,
                )
                query = torch.randn(1, 16, 1, 32)
                position = torch.tensor([[32]], dtype=torch.long)
                output, _ = ledger.attention_forward(
                    module,
                    query,
                    layer.keys,
                    layer.values,
                    None,
                    position_ids=position,
                )
                self.assertEqual(tuple(output.shape), (1, 1, 16, 32))
            verified = ledger.verify_complete()
            self.assertEqual(verified["protocol"], MULTIFORK_PROTOCOL)
            self.assertEqual(verified["request_index"], request_index)
            self.assertEqual(verified["total_calls"], 10)

    def test_ledger_rejects_wrong_request_sequence_and_kernel_swap(self):
        cache, plan, _ = make_cache_and_plan(length=32, forks=2)
        group = build_resident_request_group(
            cache, plan, resident_count=2, policy=SHARED_REUSE
        )
        for request in group.requests:
            for layer_index in plan.full_attention_layer_indices:
                request.layers[layer_index].sequence.strict_mask_check = False
        kernel = ZeroKernel()
        wrong_request_ledger = MultiForkHitLedger(
            plan,
            group.requests[1],
            request_index=1,
            resident_count=2,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            initial_query_tokens=1,
            kernel=kernel,
        )
        layer_index = plan.full_attention_layer_indices[0]
        wrong_layer = group.requests[0].layers[layer_index]
        key = torch.ones(1, 2, 1, 32)
        wrong_layer.update(key, key.clone())
        module = SimpleNamespace(
            layer_idx=layer_index,
            is_causal=True,
            num_key_value_groups=8,
            scaling=32**-0.5,
        )
        with self.assertRaisesRegex(RuntimeInvariantError, "KV_SEQUENCE_ID"):
            wrong_request_ledger.attention_forward(
                module,
                torch.randn(1, 16, 1, 32),
                wrong_layer.keys,
                wrong_layer.values,
                None,
                position_ids=torch.tensor([[32]], dtype=torch.long),
            )

        correct_ledger = MultiForkHitLedger(
            plan,
            group.requests[1],
            request_index=1,
            resident_count=2,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            initial_query_tokens=1,
            kernel=kernel,
        )
        correct_layer = group.requests[1].layers[layer_index]
        correct_layer.update(key, key.clone())
        correct_ledger.kernel = ZeroKernel()
        with self.assertRaisesRegex(RuntimeInvariantError, "KERNEL_CALLABLE_ID"):
            correct_ledger.attention_forward(
                module,
                torch.randn(1, 16, 1, 32),
                correct_layer.keys,
                correct_layer.values,
                None,
                position_ids=torch.tensor([[32]], dtype=torch.long),
            )

    def test_strict_position_audit_rejects_off_by_one_and_observes_clean_call(self):
        cache, plan, _ = make_cache_and_plan(length=32, forks=1)
        group = build_resident_request_group(
            cache, plan, resident_count=1, policy=SHARED_REUSE
        )
        layer_index = plan.full_attention_layer_indices[0]
        layer = group.requests[0].layers[layer_index]
        layer.sequence.strict_mask_check = False
        key = torch.ones(1, 2, 1, 32)
        module = SimpleNamespace(
            layer_idx=layer_index,
            is_causal=True,
            num_key_value_groups=8,
            scaling=32**-0.5,
        )
        observed = []
        ledger = MultiForkHitLedger(
            plan,
            group.requests[0],
            request_index=0,
            resident_count=1,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            initial_query_tokens=1,
            kernel=ZeroKernel(),
            strict_position_values=True,
            call_observer=observed.append,
        )
        layer.update(key, key.clone())
        with self.assertRaisesRegex(
            RuntimeInvariantError, "POSITION_CANONICAL_VALUES"
        ):
            ledger.attention_forward(
                module,
                torch.randn(1, 16, 1, 32),
                layer.keys,
                layer.values,
                None,
                position_ids=torch.tensor([[31]], dtype=torch.long),
            )
        self.assertEqual(observed, [])

        clean_cache, clean_plan, _ = make_cache_and_plan(length=32, forks=1)
        clean_group = build_resident_request_group(
            clean_cache, clean_plan, resident_count=1, policy=SHARED_REUSE
        )
        clean_layer = clean_group.requests[0].layers[layer_index]
        clean_layer.sequence.strict_mask_check = False
        clean_observed = []
        append_events = []

        def capture_append(event):
            append_events.append(event)
            self.assertTrue(torch.equal(event["key_states"], key))
            self.assertTrue(torch.equal(event["value_states"], key))
            return "oracle-append-layer3-round0-request0"

        clean_layer.sequence.append_observer = capture_append
        clean_ledger = MultiForkHitLedger(
            clean_plan,
            clean_group.requests[0],
            request_index=0,
            resident_count=1,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            initial_query_tokens=1,
            kernel=ZeroKernel(),
            strict_position_values=True,
            call_observer=clean_observed.append,
        )
        clean_layer.update(key, key.clone())
        clean_ledger.attention_forward(
            module,
            torch.randn(1, 16, 1, 32),
            clean_layer.keys,
            clean_layer.values,
            None,
            position_ids=torch.tensor([[32]], dtype=torch.long),
            scaling=0.125,
        )
        self.assertEqual(len(clean_observed), 1)
        self.assertEqual(len(append_events), 1)
        self.assertEqual(append_events[0]["append_event_index"], 0)
        self.assertEqual(append_events[0]["appended_tokens_before"], 0)
        self.assertEqual(append_events[0]["appended_tokens_after"], 1)
        self.assertEqual(
            clean_observed[0]["append_capture_id"],
            "oracle-append-layer3-round0-request0",
        )
        self.assertEqual(
            clean_ledger.calls[0]["append_capture_id"],
            "oracle-append-layer3-round0-request0",
        )
        self.assertEqual(clean_observed[0]["position_audit"]["position_ids_strict_tail_values_checked"], True)
        self.assertEqual(clean_observed[0]["effective_scaling"], 0.125)
        self.assertEqual(
            clean_observed[0]["kernel_audit"]["softmax_scale"], 0.125
        )

    def test_attention_consumes_exactly_one_isolated_append_event(self):
        cache, plan, _ = make_cache_and_plan(length=32, forks=1)
        group = build_resident_request_group(
            cache, plan, resident_count=1, policy=SHARED_REUSE
        )
        layer_index = plan.full_attention_layer_indices[0]
        layer = group.requests[0].layers[layer_index]
        layer.sequence.strict_mask_check = False
        module = SimpleNamespace(
            layer_idx=layer_index,
            is_causal=True,
            num_key_value_groups=8,
            scaling=32**-0.5,
        )
        key = torch.ones(1, 2, 1, 32)
        ledger = MultiForkHitLedger(
            plan,
            group.requests[0],
            request_index=0,
            resident_count=1,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            initial_query_tokens=2,
            kernel=ZeroKernel(),
        )
        layer.update(key, key.clone())
        layer.update(key, key.clone())
        with self.assertRaisesRegex(RuntimeInvariantError, "KV_APPEND_EVENT"):
            ledger.attention_forward(
                module,
                torch.randn(1, 16, 2, 32),
                layer.keys,
                layer.values,
                None,
                position_ids=torch.tensor([[32, 33]], dtype=torch.long),
            )
        self.assertEqual(ledger.counts[layer_index], 0)
        self.assertEqual(ledger.last_append_event_counts[layer_index], 0)

        isolated_cache, isolated_plan, _ = make_cache_and_plan(
            length=32, forks=1
        )
        isolated_group = build_resident_request_group(
            isolated_cache,
            isolated_plan,
            resident_count=1,
            policy=SHARED_REUSE,
        )
        isolated_layer = isolated_group.requests[0].layers[layer_index]
        isolated_layer.sequence.strict_mask_check = False
        original = torch.arange(64, dtype=torch.float32).reshape(1, 2, 1, 32)

        def destructive_observer(event):
            event["key_states"].fill_(1234)
            event["value_states"].fill_(-1234)
            return "isolated-append"

        isolated_layer.sequence.append_observer = destructive_observer
        isolated_ledger = MultiForkHitLedger(
            isolated_plan,
            isolated_group.requests[0],
            request_index=0,
            resident_count=1,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            initial_query_tokens=1,
            kernel=ZeroKernel(),
        )
        isolated_layer.update(original, original.clone())
        physical = int(isolated_layer.sequence.active_block_table[0, -1])
        offset = 32 % isolated_layer.sequence.arena.page_size
        self.assertTrue(
            torch.equal(
                isolated_layer.sequence.arena.key_cache[physical, offset],
                original[0, :, 0, :],
            )
        )
        isolated_ledger.attention_forward(
            module,
            torch.randn(1, 16, 1, 32),
            isolated_layer.keys,
            isolated_layer.values,
            None,
            position_ids=torch.tensor([[32]], dtype=torch.long),
        )
        self.assertEqual(isolated_ledger.counts[layer_index], 1)

        guarded_cache, guarded_plan, _ = make_cache_and_plan(
            length=32, forks=1
        )
        guarded_group = build_resident_request_group(
            guarded_cache,
            guarded_plan,
            resident_count=1,
            policy=SHARED_REUSE,
        )
        guarded_layer = guarded_group.requests[0].layers[layer_index]
        guarded_layer.sequence.strict_mask_check = False
        capture_counter = []

        def capture_for_guard(_event):
            capture_id = f"guard-capture-{len(capture_counter)}"
            capture_counter.append(capture_id)
            return capture_id

        guarded_layer.sequence.append_observer = capture_for_guard

        def malicious_call_observer(_event):
            # The payload itself has no live sequence reference.  This closure
            # nevertheless tries to mutate the live request during audit.
            guarded_layer.update(key, key.clone())

        guarded_ledger = MultiForkHitLedger(
            guarded_plan,
            guarded_group.requests[0],
            request_index=0,
            resident_count=1,
            request_policy=SHARED_REUSE,
            expected_calls_per_layer=1,
            initial_query_tokens=1,
            kernel=ZeroKernel(),
            call_observer=malicious_call_observer,
        )
        guarded_layer.update(key, key.clone())
        with self.assertRaisesRegex(RuntimeInvariantError, "KV_APPEND_EVENT"):
            guarded_ledger.attention_forward(
                module,
                torch.randn(1, 16, 1, 32),
                guarded_layer.keys,
                guarded_layer.values,
                None,
                position_ids=torch.tensor([[32]], dtype=torch.long),
            )
        self.assertEqual(guarded_ledger.counts[layer_index], 0)
        self.assertEqual(guarded_ledger.last_lengths[layer_index], 32)
        self.assertEqual(guarded_ledger.last_append_event_counts[layer_index], 0)
        self.assertEqual(guarded_layer.sequence._append_event_count, 2)

    def test_call_observer_guard_supports_inference_tensors(self):
        with torch.inference_mode():
            cache, plan, _ = make_cache_and_plan(length=32, forks=1)
            group = build_resident_request_group(
                cache, plan, resident_count=1, policy=SHARED_REUSE
            )
            layer_index = plan.full_attention_layer_indices[0]
            layer = group.requests[0].layers[layer_index]
            layer.sequence.strict_mask_check = False
            layer.sequence.append_observer = lambda _event: "inference-capture"
            observed = []
            ledger = MultiForkHitLedger(
                plan,
                group.requests[0],
                request_index=0,
                resident_count=1,
                request_policy=SHARED_REUSE,
                expected_calls_per_layer=1,
                initial_query_tokens=1,
                kernel=ZeroKernel(),
                strict_position_values=True,
                call_observer=observed.append,
            )
            key = torch.ones(1, 2, 1, 32)
            layer.update(key, key.clone())
            module = SimpleNamespace(
                layer_idx=layer_index,
                is_causal=True,
                num_key_value_groups=8,
                scaling=32**-0.5,
            )
            ledger.attention_forward(
                module,
                torch.randn(1, 16, 1, 32),
                layer.keys,
                layer.values,
                None,
                position_ids=torch.tensor([[32]], dtype=torch.long),
            )
            self.assertEqual(len(observed), 1)
            self.assertEqual(ledger.counts[layer_index], 1)

    def test_tail_cow_gate_and_physical_document_digest_cover_padding(self):
        cache, plan, _ = make_cache_and_plan(length=35, forks=1)
        rebuilt_cache, rebuilt_plan, _ = make_cache_and_plan(length=35, forks=1)
        self.assertEqual(
            source_document_physical_digests(
                cache, plan.full_attention_layer_indices
            ),
            source_document_physical_digests(
                rebuilt_cache, rebuilt_plan.full_attention_layer_indices
            ),
        )
        group = build_resident_request_group(
            cache, plan, resident_count=1, policy=SHARED_REUSE
        )
        before = source_document_physical_digests(
            cache, plan.full_attention_layer_indices
        )
        for layer_index in plan.full_attention_layer_indices:
            layer = group.requests[0].layers[layer_index]
            key = torch.ones(1, 2, 1, 32)
            layer.update(key, key.clone())
        audit = validate_runtime_kv_ownership(
            cache,
            group,
            plan,
            require_appended_tail_cow=True,
        )
        self.assertTrue(audit["passed"])
        self.assertEqual(
            before,
            source_document_physical_digests(cache, plan.full_attention_layer_indices),
        )

        layer_index = plan.full_attention_layer_indices[0]
        arena = cache.layers[layer_index].arena
        source_tail_block = arena.document_blocks_per_sequence - 1
        arena.key_cache[source_tail_block, 15].add_(1)
        self.assertNotEqual(
            before[str(layer_index)],
            source_document_physical_digests(cache, (layer_index,))[str(layer_index)],
        )

        layout_cache, layout_plan, _ = make_cache_and_plan(length=35, forks=1)
        layout_arena = layout_cache.layers[layer_index].arena
        key_backing = torch.empty(
            (*layout_arena.key_cache.shape[:-1], layout_arena.key_cache.shape[-1] * 2),
            dtype=layout_arena.key_cache.dtype,
        )
        noncontiguous_key = key_backing[..., ::2]
        noncontiguous_key.copy_(layout_arena.key_cache)
        self.assertFalse(noncontiguous_key.is_contiguous())
        layout_arena.key_cache = noncontiguous_key
        with self.assertRaisesRegex(QComemMultiForkError, "layout must be contiguous"):
            source_document_physical_digests(
                layout_cache, layout_plan.full_attention_layer_indices
            )

        table_cache, table_plan, _ = make_cache_and_plan(length=35, forks=1)
        table_cache.layers[layer_index].arena.document_block_table = (
            table_cache.layers[layer_index].arena.document_block_table.to(torch.int64)
        )
        with self.assertRaisesRegex(QComemMultiForkError, "block-table physical schema"):
            source_document_physical_digests(
                table_cache, table_plan.full_attention_layer_indices
            )

        table_storage_cache, table_storage_plan, _ = make_cache_and_plan(
            length=35, forks=1
        )
        table_storage_arena = table_storage_cache.layers[layer_index].arena
        original_table = table_storage_arena.document_block_table
        larger_table_storage = torch.empty(
            (original_table.shape[0], original_table.shape[1] + 1),
            dtype=torch.int32,
        )
        larger_table_storage[:, : original_table.shape[1]].copy_(original_table)
        table_storage_arena.document_block_table = larger_table_storage[
            :, : original_table.shape[1]
        ]
        self.assertTrue(table_storage_arena.document_block_table.is_contiguous())
        with self.assertRaisesRegex(QComemMultiForkError, "block-table physical schema"):
            source_document_physical_digests(
                table_storage_cache,
                table_storage_plan.full_attention_layer_indices,
            )

        for field in (
            "document_length",
            "max_append_tokens",
            "max_forks",
            "page_size",
            "num_key_value_heads",
            "head_dim",
            "batch_size",
            "document_blocks_per_sequence",
            "private_blocks_per_sequence",
        ):
            scalar_cache, scalar_plan, _ = make_cache_and_plan(length=35, forks=1)
            setattr(scalar_cache.layers[layer_index].arena, field, True)
            with self.assertRaisesRegex(QComemMultiForkError, "scalar schema"):
                source_document_physical_digests(
                    scalar_cache, scalar_plan.full_attention_layer_indices
                )

        append_cache, append_plan, _ = make_cache_and_plan(length=35, forks=1)
        append_arena = append_cache.layers[layer_index].arena
        append_before = source_document_physical_digests(
            append_cache, (layer_index,)
        )[str(layer_index)]
        append_arena.max_append_tokens += 1
        self.assertEqual(append_arena.private_blocks_per_sequence, 1)
        self.assertNotEqual(
            append_before,
            source_document_physical_digests(append_cache, (layer_index,))[
                str(layer_index)
            ],
        )

        corrupt_table_cache, corrupt_table_plan, _ = make_cache_and_plan(
            length=35, forks=1
        )
        corrupt_table_arena = corrupt_table_cache.layers[layer_index].arena
        corrupt_table_arena.document_block_table[0, 1] = 0
        with self.assertRaisesRegex(QComemMultiForkError, "block-table IDs drift"):
            source_document_physical_digests(
                corrupt_table_cache,
                corrupt_table_plan.full_attention_layer_indices,
            )

        corrupt_private_cache, corrupt_private_plan, _ = make_cache_and_plan(
            length=35, forks=1
        )
        corrupt_private_arena = corrupt_private_cache.layers[layer_index].arena
        corrupt_private_arena.private_block_reservations[0, 0, 0] = 0
        with self.assertRaisesRegex(QComemMultiForkError, "reservation IDs drift"):
            source_document_physical_digests(
                corrupt_private_cache,
                corrupt_private_plan.full_attention_layer_indices,
            )

        pool_cache, pool_plan, _ = make_cache_and_plan(length=35, forks=1)
        pool_arena = pool_cache.layers[layer_index].arena
        document_blocks = (
            pool_arena.batch_size * pool_arena.document_blocks_per_sequence
        )
        pool_arena.key_cache = pool_arena.key_cache[:document_blocks]
        pool_arena.value_cache = pool_arena.value_cache[:document_blocks]
        with self.assertRaisesRegex(QComemMultiForkError, "full storage|omits"):
            source_document_physical_digests(
                pool_cache, pool_plan.full_attention_layer_indices
            )

        mutant_cache, mutant_plan, _ = make_cache_and_plan(length=35, forks=1)
        mutant_group = build_resident_request_group(
            mutant_cache, mutant_plan, resident_count=1, policy=SHARED_REUSE
        )
        mutant_layer_index = mutant_plan.full_attention_layer_indices[0]
        mutant_sequence = mutant_group.requests[0].layers[
            mutant_layer_index
        ].sequence
        mutant_sequence._detach_partial_document_tail = MethodType(
            lambda self, batch_index: None,
            mutant_sequence,
        )
        key = torch.ones(1, 2, 1, 32)
        mutant_group.requests[0].layers[mutant_layer_index].update(key, key.clone())
        with self.assertRaisesRegex(RuntimeInvariantError, "KV_TAIL_COW"):
            validate_runtime_kv_ownership(
                mutant_cache,
                mutant_group,
                mutant_plan,
                require_appended_tail_cow=True,
            )

    def test_capacity_fit_locks_all_six_counts_and_linear_slope(self):
        rows = [
            {"resident_count": count, "physical_copy_nbytes": 1024 + count * 4096}
            for count in MULTIFORK_COUNTS
        ]
        fit = linear_capacity_fit(rows, "physical_copy_nbytes")
        self.assertEqual(fit["slope_nbytes_per_request"], 4096)
        self.assertEqual(fit["intercept_nbytes"], 1024)
        self.assertEqual(fit["r_squared"], 1.0)
        with self.assertRaisesRegex(QComemMultiForkError, "every N"):
            linear_capacity_fit(rows[:-1], "physical_copy_nbytes")
        bad = list(rows)
        bad[0] = {**bad[0], "resident_count": True}
        with self.assertRaisesRegex(QComemMultiForkError, "non-bool integer"):
            linear_capacity_fit(bad, "physical_copy_nbytes")

    def test_only_registered_resident_counts_are_accepted(self):
        cache, plan, _ = make_cache_and_plan(length=35, forks=3)
        with self.assertRaisesRegex(QComemMultiForkError, "unsupported"):
            build_resident_request_group(
                cache, plan, resident_count=3, policy=SHARED_REUSE
            )


if __name__ == "__main__":
    unittest.main()
