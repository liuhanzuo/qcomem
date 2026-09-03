"""Tests for the shared-packed fork mode and its ForkAudit instantiation.

These need torch but not CUDA, not Transformers and not a checkpoint: caches
are ``SimpleNamespace`` layers holding real tensors, and execution runs against
a fake adapter that emulates a hybrid backbone -- attention layers that append
through the cache-layer ``update`` contract, and GatedDeltaNet-style
convolution/recurrent buffers that are mutated in place.  On a machine without
torch the whole module skips, which is the expected outcome on the authoring
laptop; the byte and contract arithmetic these tests wrap is covered torch-free
in ``test_qcomem_multifork_accounting.py``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

try:  # pragma: no cover - import guard
    import torch

    from qcomem_multifork_accounting import (
        normalize_inventory,
        ownership_ledger,
        range_overlaps,
    )
    from qcomem_paged import analyze_cache_for_cow
    from qcomem_shared_packed_fork import (
        BorrowedPrefixKVLayer,
        PrivateMaterializedFork,
        SharedPackedFork,
        SharedPackedForkError,
        iter_tensor_slots,
        opaque_storage_id,
        prepare_shared_packed_entry,
        run_full_prefix_multifork,
        run_shared_packed_multifork,
        storage_inventory_rows,
    )
    from qcomem_shared_packed_forkaudit import (
        audit_shared_packed_multifork,
        evaluate_dequantized_view_immutability,
        evaluate_frozen_identity,
        evaluate_packed_entry_lifetime,
        evaluate_private_ownership,
        evaluate_residual_chunk_binding,
        evaluate_tail_safe_append,
        layer_class_receipt,
        run_audited_shared_packed_multifork,
        tensor_tree_digest,
    )
    from qcomem_torch import (
        FullPrefixState,
        LowerReplayState,
        PackedLowerReplayState,
        cache_nbytes,
    )

    TORCH_IMPORT_ERROR: str | None = None
except ImportError as error:  # pragma: no cover - laptop path
    TORCH_IMPORT_ERROR = f"{type(error).__name__}: {error}"


requires_torch = unittest.skipIf(
    TORCH_IMPORT_ERROR is not None,
    f"torch unavailable ({TORCH_IMPORT_ERROR})",
)

DEPTH = 2
DOCUMENT_TOKENS = 8
HIDDEN = 64
HEADS = 2
HEAD_DIM = 16


def make_hybrid_cache():
    """A two-layer cache shaped like Qwen3.5: one attention, one GDN layer."""

    torch.manual_seed(20260903)
    attention = SimpleNamespace(
        keys=torch.randn(1, HEADS, DOCUMENT_TOKENS, HEAD_DIM).to(torch.bfloat16),
        values=torch.randn(1, HEADS, DOCUMENT_TOKENS, HEAD_DIM).to(torch.bfloat16),
    )
    linear = SimpleNamespace(
        conv_states=[torch.randn(1, 8, 8).to(torch.bfloat16)],
        recurrent_states=[torch.randn(1, 2, 8, 8, dtype=torch.float32)],
    )
    return SimpleNamespace(layers=[attention, linear])


def make_lower_state():
    return LowerReplayState(
        depth=DEPTH,
        document_length=DOCUMENT_TOKENS,
        current_length=DOCUMENT_TOKENS,
        document_residual=torch.randn(1, DOCUMENT_TOKENS, HIDDEN).to(torch.bfloat16),
        cache=make_hybrid_cache(),
    )


def make_packed_state() -> "PackedLowerReplayState":
    return make_lower_state().quantize(
        bits=4, attention_bits=8, linear_bits=8, group_size=64
    )


class FakeAdapter:
    """Emulates the calls the multifork driver makes on a hybrid backbone.

    Attention layers grow through whatever ``update`` the fork installed (the
    borrowed-prefix layer, the recording COW update, or -- for a private fork
    with no hook -- a plain concatenation, which is what a stock
    ``DynamicLayer`` does).  Recurrent and convolution buffers are mutated in
    place, which is exactly why they must be private before the first call.
    """

    num_layers = 4
    vocab_size = 16

    def __init__(self) -> None:
        self.step = 0
        self.suffix_calls: list[tuple[int, int]] = []

    def _kv(self, tokens: int):
        self.step += 1
        value = float(self.step)
        keys = torch.full(
            (1, HEADS, tokens, HEAD_DIM), value, dtype=torch.bfloat16
        )
        values = torch.full(
            (1, HEADS, tokens, HEAD_DIM), -value, dtype=torch.bfloat16
        )
        return keys, values

    def continue_lower_replay(self, state, tokens):
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        count = int(tokens.shape[1])
        for layer in state.cache.layers:
            if isinstance(layer, BorrowedPrefixKVLayer):
                layer.update(*self._kv(count))
            elif hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor):
                keys, values = self._kv(count)
                update = getattr(layer, "update", None)
                if callable(update):
                    update(keys, values)
                else:
                    layer.keys = torch.cat([layer.keys, keys], dim=-2)
                    layer.values = torch.cat([layer.values, values], dim=-2)
            if hasattr(layer, "recurrent_states"):
                for buffer in layer.recurrent_states:
                    buffer.add_(1.0)
                for buffer in layer.conv_states:
                    buffer.add_(1.0)
        state.current_length = int(state.current_length) + count
        # a deterministic residual whose content depends only on the tokens
        signature = float(int(tokens.reshape(-1)[-1].item()))
        return torch.full((1, count, HIDDEN), signature, dtype=torch.bfloat16)

    def make_cache(self):
        return SimpleNamespace(layers=[])

    def run_suffix_cached_last_logits(
        self, residuals, depth, cache, *, position_offset
    ):
        residual = residuals[0]
        self.suffix_calls.append((int(position_offset), int(residual.shape[1])))
        cache.layers.append(
            SimpleNamespace(
                keys=torch.zeros(1, 1, int(residual.shape[1]), 4),
            )
        )
        base = float(residual.reshape(-1)[0].item())
        logits = torch.arange(self.vocab_size, dtype=torch.float32)
        # argmax lands on a token determined by the residual signature only, so
        # the same query produces the same tokens at every fanout
        index = int(abs(base)) % self.vocab_size
        return (logits * 0).index_fill_(0, torch.tensor([index]), 1.0).unsqueeze(0)

    def continue_full_prefix(self, state, tokens):
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        count = int(tokens.shape[1])
        for layer in state.cache.layers:
            if hasattr(layer, "keys") and isinstance(layer.keys, torch.Tensor):
                keys, values = self._kv(count)
                layer.keys = torch.cat([layer.keys, keys], dim=-2)
                layer.values = torch.cat([layer.values, values], dim=-2)
        state.current_length = int(state.current_length) + count
        signature = float(int(tokens.reshape(-1)[-1].item()))
        logits = torch.zeros(self.vocab_size, dtype=torch.float32)
        logits[int(abs(signature)) % self.vocab_size] = 1.0
        return logits.unsqueeze(0)


def queries(count: int):
    return [
        (f"r{index:02d}", torch.tensor([[3 + index, 5 + index]], dtype=torch.long))
        for index in range(count)
    ]


@requires_torch
class TensorSlotTest(unittest.TestCase):
    def test_slots_reach_list_and_attribute_leaves(self):
        cache = make_hybrid_cache()
        paths = {slot.path for slot in iter_tensor_slots(cache)}
        self.assertTrue(any(path.endswith("/keys") for path in paths))
        self.assertTrue(any("conv_states" in path for path in paths))
        self.assertTrue(any("recurrent_states" in path for path in paths))

    def test_replace_rebinds_a_list_entry(self):
        cache = make_hybrid_cache()
        slot = next(
            slot for slot in iter_tensor_slots(cache) if "conv_states" in slot.path
        )
        replacement = torch.zeros_like(slot.tensor)
        slot.replace(replacement)
        self.assertIs(cache.layers[1].conv_states[0], replacement)

    def test_replace_rebinds_an_attribute(self):
        cache = make_hybrid_cache()
        slot = next(
            slot for slot in iter_tensor_slots(cache) if slot.path.endswith("/keys")
        )
        replacement = torch.zeros_like(slot.tensor)
        slot.replace(replacement)
        self.assertIs(cache.layers[0].keys, replacement)

    def test_replace_on_a_tuple_slot_raises(self):
        holder = SimpleNamespace(pair=(torch.zeros(4), torch.zeros(4)))
        slot = next(iter_tensor_slots(holder))
        with self.assertRaises(SharedPackedForkError):
            slot.replace(torch.ones(4))


@requires_torch
class StorageInventoryTest(unittest.TestCase):
    def test_rows_validate_against_the_torch_free_schema(self):
        cache = make_hybrid_cache()
        rows = storage_inventory_rows(cache, role="probe", salt="s")
        normalized = normalize_inventory(rows)
        self.assertEqual(len(normalized), len(rows))
        self.assertTrue(all(row["view_nbytes"] > 0 for row in normalized))

    def test_same_tensor_gets_the_same_opaque_id(self):
        tensor = torch.zeros(8)
        self.assertEqual(
            opaque_storage_id(tensor, salt="s"), opaque_storage_id(tensor, salt="s")
        )
        self.assertNotEqual(
            opaque_storage_id(tensor, salt="s"),
            opaque_storage_id(torch.zeros(8), salt="s"),
        )

    def test_salt_changes_the_id(self):
        tensor = torch.zeros(8)
        self.assertNotEqual(
            opaque_storage_id(tensor, salt="a"), opaque_storage_id(tensor, salt="b")
        )

    def test_empty_salt_raises(self):
        with self.assertRaises(SharedPackedForkError):
            storage_inventory_rows(make_hybrid_cache(), role="r", salt="")


@requires_torch
class PublishedPathRegressionTest(unittest.TestCase):
    """The published private-materialization Read path must be untouched."""

    def test_packed_fork_still_produces_fully_private_copies(self):
        packed = make_packed_state()
        left = packed.fork()
        right = packed.fork()
        left_rows = normalize_inventory(
            storage_inventory_rows(left.cache, role="l", salt="s")
        )
        right_rows = normalize_inventory(
            storage_inventory_rows(right.cache, role="r", salt="s")
        )
        self.assertEqual(range_overlaps(left_rows, right_rows), [])
        self.assertGreater(cache_nbytes(left.cache), 0)

    def test_private_materialize_entry_shares_nothing(self):
        packed = make_packed_state()
        entry = prepare_shared_packed_entry(
            packed, share_mode="private-materialize"
        )
        self.assertIsNone(entry.view)
        self.assertEqual(entry.shared_view_nbytes, 0)
        self.assertEqual(entry.shared_inventory(), [])
        fork = entry.fork("r00")
        self.assertIsInstance(fork, PrivateMaterializedFork)
        self.assertEqual(fork.memory_breakdown()["shared_nbytes"], 0)
        self.assertGreater(fork.memory_breakdown()["private_nbytes"], 0)

    def test_private_materialize_immutability_audit_declares_itself_vacuous(self):
        entry = prepare_shared_packed_entry(
            make_packed_state(), share_mode="private-materialize"
        )
        audit = entry.fork("r00").verify_shared_immutable()
        self.assertTrue(audit["verified"])
        self.assertTrue(audit["vacuous"])
        self.assertEqual(audit["guarded_tensors"], 0)


@requires_torch
class SharedForkTest(unittest.TestCase):
    def entry(self, **kwargs):
        options = {
            "share_mode": "shared-packed-view",
            "rebind_policy": "transition",
            "tail_policy": "borrowed-prefix",
        }
        options.update(kwargs)
        return prepare_shared_packed_entry(make_packed_state(), **options)

    def test_one_dequantization_serves_every_fork(self):
        entry = self.entry()
        self.assertIsNotNone(entry.view)
        view_id = id(entry.view)
        forks = [entry.fork(f"r{index:02d}") for index in range(3)]
        self.assertEqual(entry.fork_count, 3)
        self.assertEqual(id(entry.view), view_id)
        for fork in forks:
            self.assertIsInstance(fork, SharedPackedFork)
            self.assertIs(fork.document_residual, entry.view.document_residual)

    def test_shared_and_private_bytes_at_fork(self):
        entry = self.entry()
        fork = entry.fork("r00")
        breakdown = fork.memory_breakdown()
        self.assertGreater(breakdown["shared_nbytes"], 0)
        # under the transition rebind policy the mutable base is borrowed, so
        # nothing is private until the registered transition
        self.assertEqual(breakdown["private_nbytes"], 0)
        self.assertEqual(fork.initial_private_nbytes, 0)

    def test_setup_rebind_policy_privatizes_at_fork(self):
        entry = self.entry(rebind_policy="setup")
        fork = entry.fork("r00")
        self.assertGreater(fork.memory_breakdown()["private_nbytes"], 0)
        self.assertEqual(fork.initial_private_nbytes, entry.plan.linear_nbytes)
        self.assertEqual(fork.rebind_mutable_state()["rebound_tensor_count"], 0)

    def test_transition_rebind_changes_storage_identity(self):
        entry = self.entry()
        fork = entry.fork("r00")
        before = fork.memory_breakdown()
        result = fork.rebind_mutable_state()
        after = fork.memory_breakdown()
        self.assertGreater(result["rebound_tensor_count"], 0)
        self.assertTrue(
            all(event["storage_identity_changed"] for event in result["events"])
        )
        self.assertGreater(after["private_nbytes"], before["private_nbytes"])

    def test_rebinding_twice_raises(self):
        fork = self.entry().fork("r00")
        fork.rebind_mutable_state()
        with self.assertRaises(SharedPackedForkError):
            fork.rebind_mutable_state()

    def test_two_forks_have_disjoint_mutable_state_after_the_transition(self):
        entry = self.entry()
        left = entry.fork("r00")
        right = entry.fork("r01")
        left.rebind_mutable_state()
        right.rebind_mutable_state()
        ledger = ownership_ledger(
            shared_inventory=entry.shared_inventory(),
            request_inventories={
                "r00": storage_inventory_rows(
                    left.cache, role="r00", salt=entry.salt
                ),
                "r01": storage_inventory_rows(
                    right.cache, role="r01", salt=entry.salt
                ),
            },
        )
        self.assertTrue(ledger["passed"], ledger["pairwise"])
        self.assertTrue(ledger["non_vacuous"])
        self.assertGreater(ledger["shared_entry_nbytes"], 0)

    def test_borrowed_prefix_forks_alias_one_document(self):
        entry = self.entry()
        left = entry.fork("r00")
        right = entry.fork("r01")
        left_layer = left.cache.layers[0]
        right_layer = right.cache.layers[0]
        self.assertIsInstance(left_layer, BorrowedPrefixKVLayer)
        self.assertIs(left_layer.shared_keys, right_layer.shared_keys)

    def test_materialized_tail_keeps_the_dynamic_layer(self):
        entry = self.entry(tail_policy="materialized-tail")
        fork = entry.fork("r00")
        self.assertNotIsInstance(fork.cache.layers[0], BorrowedPrefixKVLayer)
        self.assertIs(fork.cache.layers[0].keys, entry.view.cache.layers[0].keys)

    def test_release_document_residual_only_drops_the_reference(self):
        entry = self.entry()
        fork = entry.fork("r00")
        residual = entry.view.document_residual
        fork.release_document_residual()
        self.assertIsNone(fork.document_residual)
        self.assertTrue(fork.released_document_residual)
        self.assertIsNotNone(entry.view.document_residual)
        self.assertIs(entry.view.document_residual, residual)

    def test_unsupported_cache_falls_back_with_a_recorded_reason(self):
        state = make_lower_state()
        state.cache.layers[0].mystery = torch.ones(4)
        entry = prepare_shared_packed_entry(
            state, share_mode="shared-packed-view"
        )
        self.assertIsNone(entry.view)
        self.assertEqual(entry.effective_share_mode, "private-materialize")
        self.assertIsNotNone(entry.fallback_reason)
        self.assertIn("mystery", entry.fallback_reason)

    def test_unknown_modes_raise(self):
        packed = make_packed_state()
        with self.assertRaises(Exception):
            prepare_shared_packed_entry(packed, share_mode="nonsense")
        with self.assertRaises(Exception):
            prepare_shared_packed_entry(
                packed, share_mode="shared-packed-view", rebind_policy="nonsense"
            )
        with self.assertRaises(Exception):
            prepare_shared_packed_entry(
                packed, share_mode="shared-packed-view", tail_policy="nonsense"
            )

    def test_deployment_memory_components_keeps_published_field_meanings(self):
        entry = self.entry()
        components = entry.deployment_memory_components()
        self.assertEqual(
            components["persistent_document_nbytes"], entry.entry_retained_nbytes
        )
        self.assertEqual(
            components["persistent_materialized_staging_nbytes"],
            entry.shared_view_nbytes,
        )
        self.assertEqual(components["share_mode_effective"], "shared-packed-view")

    def test_mutating_a_shared_tensor_is_caught(self):
        state = make_lower_state()
        entry = prepare_shared_packed_entry(
            state, share_mode="shared-packed-view", tail_policy="materialized-tail"
        )
        fork = entry.fork("r00")
        self.assertTrue(fork.verify_shared_immutable()["verified"])
        entry.view.cache.layers[0].keys.add_(1.0)
        with self.assertRaises(SharedPackedForkError):
            fork.verify_shared_immutable()
        self.assertFalse(entry.verify_view_unchanged()["verified"])


@requires_torch
class CachePlanTest(unittest.TestCase):
    def test_the_synthetic_cache_is_classified_the_way_the_frozen_plan_expects(self):
        plan = analyze_cache_for_cow(make_hybrid_cache())
        self.assertTrue(plan.supported, plan.reason)
        self.assertEqual(plan.active_attention_layers, (0,))
        self.assertEqual(plan.active_linear_layers, (1,))
        self.assertGreater(plan.attention_nbytes, 0)
        self.assertGreater(plan.linear_nbytes, 0)

    def test_an_unclassified_leaf_fails_closed(self):
        cache = make_hybrid_cache()
        cache.layers[0].mystery = torch.ones(4)
        plan = analyze_cache_for_cow(cache)
        self.assertFalse(plan.supported)
        self.assertIn("mystery", plan.reason)

    def test_entry_reports_how_much_attention_state_is_shareable(self):
        entry = prepare_shared_packed_entry(
            make_packed_state(),
            share_mode="shared-packed-view",
            tail_policy="borrowed-prefix",
        )
        components = entry.deployment_memory_components()
        self.assertEqual(components["shared_attention_layer_count"], 1)
        self.assertEqual(components["shared_linear_layer_count"], 1)
        self.assertGreater(components["shared_attention_nbytes"], 0)


@requires_torch
class BorrowedPrefixLayerTest(unittest.TestCase):
    def layer(self):
        events: list[dict] = []
        prefix_keys = torch.arange(HEADS * 4 * HEAD_DIM, dtype=torch.float32).reshape(
            1, HEADS, 4, HEAD_DIM
        )
        prefix_values = prefix_keys.clone()
        return (
            BorrowedPrefixKVLayer(
                prefix_keys,
                prefix_values,
                layer_index=0,
                request_id="r00",
                events=events,
                salt="s",
            ),
            events,
        )

    def test_update_returns_the_same_tensor_a_plain_cat_would(self):
        layer, _ = self.layer()
        new_keys = torch.ones(1, HEADS, 1, HEAD_DIM)
        new_values = torch.full((1, HEADS, 1, HEAD_DIM), 2.0)
        expected_keys = torch.cat([layer.shared_keys, new_keys], dim=-2)
        keys, values = layer.update(new_keys, new_values)
        self.assertTrue(torch.equal(keys, expected_keys))
        self.assertEqual(values.shape[-2], 5)

    def test_only_the_tail_is_retained(self):
        layer, _ = self.layer()
        layer.update(torch.ones(1, HEADS, 1, HEAD_DIM), torch.ones(1, HEADS, 1, HEAD_DIM))
        self.assertEqual(layer.tail_keys.shape[-2], 1)
        self.assertEqual(layer.prefix_length, 4)
        self.assertEqual(layer.get_seq_length(), 5)

    def test_prefix_is_never_written(self):
        layer, _ = self.layer()
        before = layer.shared_keys.clone()
        for _ in range(3):
            layer.update(
                torch.ones(1, HEADS, 1, HEAD_DIM), torch.ones(1, HEADS, 1, HEAD_DIM)
            )
        self.assertTrue(torch.equal(layer.shared_keys, before))
        self.assertEqual(layer.tail_length, 3)

    def test_append_events_record_the_prefix_and_the_rebind(self):
        layer, events = self.layer()
        layer.update(torch.ones(1, HEADS, 1, HEAD_DIM), torch.ones(1, HEADS, 1, HEAD_DIM))
        layer.update(torch.ones(1, HEADS, 1, HEAD_DIM), torch.ones(1, HEADS, 1, HEAD_DIM))
        self.assertEqual(len(events), 2)
        for event in events:
            self.assertEqual(event["tail_policy"], "borrowed-prefix")
            self.assertTrue(event["keys_storage_rebound"])
            self.assertTrue(event["shared_prefix_storage_unchanged"])
            self.assertFalse(event["returned_is_retained"])
            self.assertGreater(event["transient_concat_nbytes"], 0)
        self.assertEqual(
            events[0]["shared_prefix_storage_id"],
            events[1]["shared_prefix_storage_id"],
        )

    def test_mask_sizes_accept_an_int_and_a_cache_position(self):
        layer, _ = self.layer()
        self.assertEqual(layer.get_mask_sizes(3), (7, 0))
        self.assertEqual(
            layer.get_mask_sizes(torch.arange(2, dtype=torch.long)), (6, 0)
        )
        self.assertEqual(sorted(layer.mask_size_call_forms), ["Tensor", "int"])

    def test_mask_sizes_reject_an_unrecognized_argument(self):
        layer, _ = self.layer()
        with self.assertRaises(SharedPackedForkError):
            layer.get_mask_sizes("three")
        with self.assertRaises(SharedPackedForkError):
            layer.get_mask_sizes()

    def test_mismatched_prefix_lengths_raise(self):
        with self.assertRaises(SharedPackedForkError):
            BorrowedPrefixKVLayer(
                torch.zeros(1, HEADS, 4, HEAD_DIM),
                torch.zeros(1, HEADS, 3, HEAD_DIM),
                layer_index=0,
                request_id="r00",
                events=[],
                salt="s",
            )


@requires_torch
class MultiforkDriverTest(unittest.TestCase):
    def run_shared(self, count=2, **kwargs):
        entry = prepare_shared_packed_entry(
            make_packed_state(),
            share_mode="shared-packed-view",
            rebind_policy=kwargs.pop("rebind_policy", "transition"),
            tail_policy=kwargs.pop("tail_policy", "borrowed-prefix"),
        )
        captures: dict[str, dict] = {}

        def capture(point, forks):
            captures[point] = {
                fork.request_id: storage_inventory_rows(
                    {
                        "cache": fork.cache,
                        "document_residual": fork.document_residual,
                    },
                    role=point,
                    salt=entry.salt,
                )
                for fork in forks
            }

        trace = run_shared_packed_multifork(
            FakeAdapter(),
            entry,
            queries(count),
            max_new_tokens=kwargs.pop("max_new_tokens", 3),
            eos_token_ids=set(),
            capture=capture,
        )
        return entry, trace, captures

    def test_all_requests_run_and_emit_tokens(self):
        _, trace, _ = self.run_shared(count=3)
        self.assertEqual(len(trace.request_traces), 3)
        self.assertEqual(trace.arm, "qcomem-shared-packed")
        self.assertEqual(trace.fork_mode, "shared-packed-view")
        for row in trace.request_traces:
            self.assertEqual(len(row.generated_token_ids), 3)

    def test_capture_points_are_all_taken(self):
        _, _, captures = self.run_shared(count=2)
        self.assertEqual(sorted(captures), ["final", "setup", "transition"])
        self.assertEqual(sorted(captures["setup"]), ["r00", "r01"])

    def test_setup_capture_shows_the_borrow_and_final_shows_privacy(self):
        entry, _, captures = self.run_shared(count=2)
        setup = ownership_ledger(
            shared_inventory=entry.shared_inventory(),
            request_inventories=captures["setup"],
        )
        final = ownership_ledger(
            shared_inventory=entry.shared_inventory(),
            request_inventories=captures["final"],
        )
        self.assertEqual(setup["per_request"]["r00"]["private_nbytes"], 0)
        self.assertGreater(final["per_request"]["r00"]["private_nbytes"], 0)
        self.assertGreater(final["per_request"]["r00"]["shared_nbytes"], 0)
        self.assertTrue(final["passed"], final["pairwise"])

    def test_borrowed_prefix_keeps_sharing_through_decode(self):
        entry, trace, captures = self.run_shared(count=2)
        final = ownership_ledger(
            shared_inventory=entry.shared_inventory(),
            request_inventories=captures["final"],
        )
        for row in final["per_request"].values():
            self.assertGreater(row["shared_nbytes"], 0)
        self.assertGreater(len(trace.append_events), 0)

    def test_materialized_tail_stops_sharing_after_the_first_append(self):
        entry, _, captures = self.run_shared(
            count=2, tail_policy="materialized-tail"
        )
        setup = ownership_ledger(
            shared_inventory=entry.shared_inventory(),
            request_inventories=captures["setup"],
        )
        final = ownership_ledger(
            shared_inventory=entry.shared_inventory(),
            request_inventories=captures["final"],
        )
        self.assertGreater(setup["per_request"]["r00"]["shared_nbytes"], 0)
        self.assertEqual(final["per_request"]["r00"]["shared_nbytes"], 0)

    def test_residual_binding_events_are_recorded_once_per_request(self):
        _, trace, _ = self.run_shared(count=2)
        self.assertEqual(len(trace.residual_binding_events), 2)
        for event in trace.residual_binding_events:
            self.assertTrue(event["document_chunk_is_shared_view_tensor"])
            self.assertFalse(event["query_chunk_is_shared_view_tensor"])
            self.assertEqual(event["document_position_offset"], 0)
            self.assertEqual(event["query_position_offset"], DOCUMENT_TOKENS)
            self.assertFalse(event["chunks_share_storage"])

    def test_call_log_covers_every_request_and_both_chunks(self):
        _, trace, _ = self.run_shared(count=2)
        chunks = {row.get("chunk") for row in trace.adapter_call_log}
        self.assertIn("document_residual_seed", chunks)
        self.assertIn("query_residual_prefill", chunks)
        self.assertEqual(
            {row["request_id"] for row in trace.adapter_call_log}, {"r00", "r01"}
        )

    def test_shared_view_survives_the_whole_run(self):
        entry, _, _ = self.run_shared(count=2)
        self.assertTrue(entry.verify_view_unchanged()["verified"])

    def test_duplicate_request_ids_raise(self):
        entry = prepare_shared_packed_entry(
            make_packed_state(), share_mode="shared-packed-view"
        )
        with self.assertRaises(SharedPackedForkError):
            run_shared_packed_multifork(
                FakeAdapter(),
                entry,
                [("r00", torch.tensor([[1, 2]])), ("r00", torch.tensor([[3, 4]]))],
                max_new_tokens=1,
                eos_token_ids=set(),
            )

    def test_no_requests_raise(self):
        entry = prepare_shared_packed_entry(
            make_packed_state(), share_mode="shared-packed-view"
        )
        with self.assertRaises(SharedPackedForkError):
            run_shared_packed_multifork(
                FakeAdapter(), entry, [], max_new_tokens=1, eos_token_ids=set()
            )

    def test_private_mode_runs_through_the_same_driver(self):
        entry = prepare_shared_packed_entry(
            make_packed_state(), share_mode="private-materialize"
        )
        trace = run_shared_packed_multifork(
            FakeAdapter(),
            entry,
            queries(2),
            max_new_tokens=2,
            eos_token_ids=set(),
        )
        self.assertEqual(trace.arm, "qcomem-private-materialize")
        for row in trace.request_traces:
            self.assertEqual(row.final_shared_nbytes, 0)
            self.assertGreater(row.final_private_nbytes, 0)

    def test_shared_and_private_modes_agree_token_for_token(self):
        adapter = FakeAdapter()
        shared_entry = prepare_shared_packed_entry(
            make_packed_state(), share_mode="shared-packed-view"
        )
        shared = run_shared_packed_multifork(
            adapter, shared_entry, queries(2), max_new_tokens=3, eos_token_ids=set()
        )
        private_entry = prepare_shared_packed_entry(
            make_packed_state(), share_mode="private-materialize"
        )
        private = run_shared_packed_multifork(
            FakeAdapter(),
            private_entry,
            queries(2),
            max_new_tokens=3,
            eos_token_ids=set(),
        )
        self.assertEqual(shared.token_traces(), private.token_traces())

    def test_full_prefix_multifork_reports_zero_sharing(self):
        state = FullPrefixState(
            document_length=DOCUMENT_TOKENS,
            current_length=DOCUMENT_TOKENS,
            cache=make_hybrid_cache(),
        )
        trace = run_full_prefix_multifork(
            FakeAdapter(), state, queries(2), max_new_tokens=2, eos_token_ids=set()
        )
        self.assertEqual(trace.arm, "full-prefix")
        for row in trace.request_traces:
            self.assertEqual(row.initial_shared_nbytes, 0)
            self.assertGreater(row.materialized_nbytes, 0)

    def test_eos_stops_a_request_without_stopping_the_others(self):
        adapter = FakeAdapter()
        entry = prepare_shared_packed_entry(
            make_packed_state(), share_mode="shared-packed-view"
        )
        # r00's query ends with token 5 and r01's with 6; stopping on the token
        # the fake adapter emits for r00 must not truncate r01
        first = run_shared_packed_multifork(
            adapter, entry, queries(2), max_new_tokens=2, eos_token_ids=set()
        )
        stop = {first.request_traces[0].generated_token_ids[0]}
        entry2 = prepare_shared_packed_entry(
            make_packed_state(), share_mode="shared-packed-view"
        )
        second = run_shared_packed_multifork(
            FakeAdapter(), entry2, queries(2), max_new_tokens=2, eos_token_ids=stop
        )
        self.assertEqual(second.request_traces[0].generated_token_ids, [])
        self.assertGreater(len(second.request_traces[1].generated_token_ids), 0)


@requires_torch
class PredicateTest(unittest.TestCase):
    def test_frozen_identity_requires_every_binding_to_replay(self):
        good = evaluate_frozen_identity(
            {"a": {"declared": "x", "replayed": "x"}}
        )
        self.assertTrue(good["passed"])
        bad = evaluate_frozen_identity(
            {"a": {"declared": "x", "replayed": "y"}}
        )
        self.assertFalse(bad["passed"])
        missing = evaluate_frozen_identity({"a": {"replayed": "y"}})
        self.assertFalse(missing["passed"])
        self.assertFalse(evaluate_frozen_identity({})["passed"])

    def test_private_ownership_does_not_assert_the_setup_borrow(self):
        ledger = {
            "per_request": {"r00": {"private_storage_ids": ["p0"]}},
            "passed": True,
            "non_vacuous": True,
        }
        result = evaluate_private_ownership(
            transition_ledger=ledger,
            final_ledger=ledger,
            shared_storage_ids=["s0"],
        )
        self.assertTrue(result["passed"])
        self.assertEqual(result["evaluated_at"], ["transition", "final"])
        self.assertTrue(result["setup_capture_recorded_not_asserted"])

    def test_private_ownership_catches_a_private_id_that_is_shared(self):
        ledger = {
            "per_request": {"r00": {"private_storage_ids": ["s0"]}},
            "passed": True,
            "non_vacuous": True,
        }
        result = evaluate_private_ownership(
            transition_ledger=ledger,
            final_ledger=ledger,
            shared_storage_ids=["s0"],
        )
        self.assertFalse(result["passed"])
        self.assertTrue(result["private_shared_leaks"])

    def test_tail_safe_append_requires_every_request_to_append(self):
        events = [
            {
                "request_id": "r00",
                "keys_storage_rebound": True,
                "after_keys_storage_id": "t0",
                "after_values_storage_id": "t1",
                "shared_prefix_storage_id": "s0",
                "shared_prefix_storage_unchanged": True,
            }
        ]
        result = evaluate_tail_safe_append(
            append_events=events,
            shared_storage_ids=["s0"],
            request_count=2,
            shared_attention_unchanged=True,
            tail_policy="borrowed-prefix",
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["all_requests_appended"])

    def test_tail_safe_append_catches_a_write_to_the_shared_prefix(self):
        events = [
            {
                "request_id": "r00",
                "keys_storage_rebound": True,
                "after_keys_storage_id": "s0",
                "after_values_storage_id": "t1",
                "shared_prefix_storage_id": "s0",
                "shared_prefix_storage_unchanged": True,
            }
        ]
        result = evaluate_tail_safe_append(
            append_events=events,
            shared_storage_ids=["s0"],
            request_count=1,
            shared_attention_unchanged=True,
            tail_policy="borrowed-prefix",
        )
        self.assertFalse(result["passed"])

    def test_tail_safe_append_is_vacuous_without_events(self):
        result = evaluate_tail_safe_append(
            append_events=[],
            shared_storage_ids=["s0"],
            request_count=2,
            shared_attention_unchanged=True,
            tail_policy="borrowed-prefix",
        )
        self.assertFalse(result["non_vacuous"])
        self.assertFalse(result["passed"])

    def test_view_immutability_fails_when_a_request_shared_nothing(self):
        digest = {"tree_sha256": "a", "tensor_count": 1}
        result = evaluate_dequantized_view_immutability(
            setup_digest=digest,
            final_digest=digest,
            per_request_shared_nbytes={"r00": 100, "r01": 0},
            view_guard={"verified": True, "vacuous": False},
            sharing_window="final",
        )
        self.assertFalse(result["passed"])
        self.assertEqual(result["requests_sharing_nothing"], ["r01"])

    def test_view_immutability_fails_on_a_vacuous_guard(self):
        digest = {"tree_sha256": "a", "tensor_count": 1}
        result = evaluate_dequantized_view_immutability(
            setup_digest=digest,
            final_digest=digest,
            per_request_shared_nbytes={"r00": 1, "r01": 1},
            view_guard={"verified": True, "vacuous": True},
            sharing_window="final",
        )
        self.assertFalse(result["passed"])

    def test_residual_binding_requires_one_shared_chunk_for_all_requests(self):
        def event(request_id, document_id):
            return {
                "request_id": request_id,
                "document_chunk_storage_id": document_id,
                "document_chunk_is_shared_view_tensor": True,
                "query_chunk_is_shared_view_tensor": False,
                "chunks_share_storage": False,
                "document_position_offset": 0,
                "query_position_offset": 8,
                "chunks_are_distinct_calls": True,
            }

        good = evaluate_residual_chunk_binding(
            events=[event("r00", "d"), event("r01", "d")],
            request_count=2,
            document_length=8,
        )
        self.assertTrue(good["passed"])
        split = evaluate_residual_chunk_binding(
            events=[event("r00", "d0"), event("r01", "d1")],
            request_count=2,
            document_length=8,
        )
        self.assertFalse(split["passed"])
        self.assertEqual(split["distinct_document_chunk_storage_count"], 2)

    def test_residual_binding_catches_a_wrong_offset(self):
        result = evaluate_residual_chunk_binding(
            events=[
                {
                    "request_id": "r00",
                    "document_chunk_storage_id": "d",
                    "document_chunk_is_shared_view_tensor": True,
                    "query_chunk_is_shared_view_tensor": False,
                    "chunks_share_storage": False,
                    "document_position_offset": 1,
                    "query_position_offset": 8,
                    "chunks_are_distinct_calls": True,
                }
            ],
            request_count=1,
            document_length=8,
        )
        self.assertFalse(result["passed"])

    def test_packed_lifetime_catches_a_leaked_reference(self):
        digest = {"tree_sha256": "a", "tensor_count": 2}
        result = evaluate_packed_entry_lifetime(
            setup_digest=digest,
            final_digest=digest,
            packed_storage_ids=["p0"],
            request_storage_ids={"r00": ["p0", "t0"]},
            forks_created=1,
            forks_released=1,
        )
        self.assertFalse(result["passed"])
        self.assertIn("r00", result["requests_referencing_packed_storage"])

    def test_packed_lifetime_catches_an_unreleased_fork(self):
        digest = {"tree_sha256": "a", "tensor_count": 2}
        result = evaluate_packed_entry_lifetime(
            setup_digest=digest,
            final_digest=digest,
            packed_storage_ids=["p0"],
            request_storage_ids={"r00": ["t0"]},
            forks_created=2,
            forks_released=1,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["all_forks_released"])

    def test_packed_lifetime_catches_a_content_change(self):
        result = evaluate_packed_entry_lifetime(
            setup_digest={"tree_sha256": "a", "tensor_count": 2},
            final_digest={"tree_sha256": "b", "tensor_count": 2},
            packed_storage_ids=["p0"],
            request_storage_ids={"r00": ["t0"]},
            forks_created=1,
            forks_released=1,
        )
        self.assertFalse(result["passed"])


@requires_torch
class DigestTest(unittest.TestCase):
    def test_tree_digest_is_stable_and_content_sensitive(self):
        cache = make_hybrid_cache()
        first = tensor_tree_digest(cache, label="a")
        second = tensor_tree_digest(cache, label="b")
        self.assertEqual(first["tree_sha256"], second["tree_sha256"])
        cache.layers[0].keys.add_(1.0)
        self.assertNotEqual(
            first["tree_sha256"], tensor_tree_digest(cache, label="c")["tree_sha256"]
        )

    def test_layer_class_receipt_names_every_layer(self):
        receipt = layer_class_receipt(make_hybrid_cache(), FakeAdapter())
        self.assertEqual(receipt["layer_count"], 2)
        self.assertEqual(receipt["adapter_class"], "FakeAdapter")


@requires_torch
class EndToEndAuditTest(unittest.TestCase):
    def audited(self, count=2, **kwargs):
        adapter = FakeAdapter()
        packed = make_packed_state()
        document = torch.arange(DOCUMENT_TOKENS, dtype=torch.long).unsqueeze(0)
        request_queries = queries(count)
        reference_entry = prepare_shared_packed_entry(
            make_packed_state(), share_mode="private-materialize"
        )
        reference = run_shared_packed_multifork(
            FakeAdapter(),
            reference_entry,
            request_queries,
            max_new_tokens=3,
            eos_token_ids=set(),
        ).token_traces()
        single = run_audited_shared_packed_multifork(
            adapter,
            packed,
            document,
            request_queries[:1],
            max_new_tokens=3,
            eos_token_ids=set(),
            **kwargs,
        )
        result = run_audited_shared_packed_multifork(
            FakeAdapter(),
            packed,
            document,
            request_queries,
            max_new_tokens=3,
            eos_token_ids=set(),
            private_reference_traces=reference,
            **kwargs,
        )
        rerun = audit_shared_packed_multifork(
            **result["audit_inputs"],
            traces_by_fanout={
                1: single["trace"].token_traces(),
                count: result["trace"].token_traces(),
            },
        )
        return result, rerun

    def test_every_target_is_covered(self):
        _, audit = self.audited(count=2)
        summary = audit["contract_summary"]
        self.assertTrue(
            summary["all_applicable_targets_covered"], summary["uncovered_targets"]
        )
        self.assertEqual(len(audit["target_rows"]), 10)

    def test_every_predicate_passes_on_a_clean_run(self):
        _, audit = self.audited(count=2)
        failing = [
            row["target"]
            for row in audit["target_rows"]
            if row["predicate_passed"] is not True
        ]
        self.assertEqual(failing, [], audit["detail"])
        self.assertTrue(audit["contract_summary"]["packed_entry_obligations_all_passed"])

    def test_audit_records_the_sharing_window(self):
        _, audit = self.audited(count=2)
        self.assertEqual(audit["sharing_window"], "final")
        self.assertIn(
            "whole request lifetime",
            next(
                row["scope_note"]
                for row in audit["target_rows"]
                if row["target"] == "dequantized_view_immutability"
            ),
        )

    def test_materialized_tail_policy_names_its_narrower_window(self):
        _, audit = self.audited(count=2, tail_policy="materialized-tail")
        self.assertEqual(audit["sharing_window"], "setup")
        self.assertIn(
            "does not establish steady-state sharing",
            next(
                row["scope_note"]
                for row in audit["target_rows"]
                if row["target"] == "dequantized_view_immutability"
            ),
        )

    def test_sharing_efficiency_counts_avoided_copies(self):
        _, audit = self.audited(count=3)
        efficiency = audit["ownership"]["sharing_efficiency_at_window"]
        self.assertEqual(efficiency["request_count"], 3)
        self.assertGreater(efficiency["copies_avoided_nbytes"], 0)

    def test_cross_n_consistency_is_non_vacuous(self):
        _, audit = self.audited(count=2)
        cross_n = audit["detail"]["cross_n_prefix_consistency"]
        self.assertTrue(cross_n["non_vacuous"])
        self.assertEqual(cross_n["fanouts"], [1, 2])

    def test_single_request_run_cannot_claim_sharing(self):
        adapter = FakeAdapter()
        packed = make_packed_state()
        document = torch.arange(DOCUMENT_TOKENS, dtype=torch.long).unsqueeze(0)
        result = run_audited_shared_packed_multifork(
            adapter,
            packed,
            document,
            queries(1),
            max_new_tokens=2,
            eos_token_ids=set(),
        )
        audit = result["audit"]
        opened = set(audit["contract_summary"]["open_targets"])
        self.assertIn("dequantized_view_immutability", opened)
        self.assertIn("residual_chunk_binding", opened)
        self.assertIn("private_ownership", opened)
        self.assertIn("cross_n_prefix_consistency", opened)

    def test_missing_capture_raises(self):
        result, _ = self.audited(count=2)
        inputs = dict(result["audit_inputs"])
        inputs["captures"] = {
            key: value
            for key, value in inputs["captures"].items()
            if key != "transition"
        }
        with self.assertRaises(Exception):
            audit_shared_packed_multifork(
                **inputs, traces_by_fanout={2: result["trace"].token_traces()}
            )


if __name__ == "__main__":
    unittest.main()
