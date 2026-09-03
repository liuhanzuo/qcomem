from __future__ import annotations

import copy
import gc
import json
import unittest
import weakref
from types import SimpleNamespace

import torch

from qcomem_forkaudit_storage_witness import (
    EXPECTED_TENSORS_PER_OWNER,
    GATE_COMPLETED_VS_BASE_DISJOINT,
    GATE_COMPLETED_VS_PEERS_DISJOINT,
    GATE_COMPLETED_BINDING_REBOUND,
    GATE_INCOMPLETE_BINDING_UNCHANGED,
    GATE_MATERIALIZED_SETUP_PEERS_DISJOINT,
    GATE_SHARED_INCOMPLETE_EXACT_BASE_ALIAS,
    GDNStorageWitnessError,
    PHASE_POST_GENERATION,
    PHASE_POST_TRANSITION,
    PHASE_SETUP_BORROWED_IMMUTABLE,
    TIMELINE_SCHEMA_VERSION,
    POLICY_MATERIALIZED,
    POLICY_SHARED_BASE,
    capture_gdn_storage_snapshot,
    capture_gdn_phase_witness,
    capture_persistent_gdn_guard,
    capture_request_gdn_binding_guard,
    replay_request_gdn_binding_witness,
    replay_gdn_storage_timeline,
    replay_gdn_storage_witness,
    verify_request_gdn_binding_guard,
)
from qcomem_vllm_paged_multifork_resident import (
    GDN_BORROW_IMMUTABLE_BASE,
    GDN_MATERIALIZE_REQUEST_BASE,
)


LINEAR_LAYERS = tuple(range(30))


def make_persistent() -> SimpleNamespace:
    layers = []
    for layer_index in LINEAR_LAYERS:
        conv = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4) + layer_index
        recurrent = (
            torch.arange(20, dtype=torch.float32).reshape(1, 2, 2, 5)
            + 1000
            + layer_index
        )
        layers.append(
            SimpleNamespace(
                conv_states={0: conv},
                recurrent_states={0: recurrent},
            )
        )
    return SimpleNamespace(layers=layers)


def make_requests(
    persistent: SimpleNamespace,
    count: int,
    *,
    borrowed: bool,
) -> list[SimpleNamespace]:
    requests = []
    for _request_index in range(count):
        layers = []
        for source in persistent.layers:
            conv = source.conv_states[0] if borrowed else source.conv_states[0].clone()
            recurrent = (
                source.recurrent_states[0]
                if borrowed
                else source.recurrent_states[0].clone()
            )
            layers.append(
                SimpleNamespace(
                    conv_states={0: conv},
                    recurrent_states={0: recurrent},
                )
            )
        requests.append(SimpleNamespace(layers=layers))
    return requests


def json_round_trip(value):
    return json.loads(json.dumps(value))


def capture(
    persistent,
    requests,
    *,
    phase,
    policy,
    completed=None,
    guard=None,
    capture_id=None,
    request_guard_id=None,
):
    if guard is None:
        guard = capture_persistent_gdn_guard(persistent, LINEAR_LAYERS)
    return capture_gdn_storage_snapshot(
        persistent,
        requests,
        LINEAR_LAYERS,
        phase=phase,
        policy=policy,
        persistent_guard=guard,
        completed_request_indices=completed,
        capture_id=capture_id,
        request_guard_id=request_guard_id,
    )


class GDNStorageWitnessTest(unittest.TestCase):
    def test_request_binding_guard_pins_setup_tensor_against_aba_and_freezes_content(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=False)
        setup_tensor = requests[0].layers[0].conv_states[0]
        setup_ref = weakref.ref(setup_tensor)
        guard = capture_request_gdn_binding_guard(
            requests,
            LINEAR_LAYERS,
            policy=GDN_MATERIALIZE_REQUEST_BASE,
        )
        requests[0] = make_requests(persistent, 1, borrowed=False)[0]
        del setup_tensor
        gc.collect()
        self.assertIsNotNone(setup_ref())
        del guard
        gc.collect()
        self.assertIsNone(setup_ref())

        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=False)
        guard = capture_request_gdn_binding_guard(
            requests,
            LINEAR_LAYERS,
            policy=GDN_MATERIALIZE_REQUEST_BASE,
        )
        requests[0] = make_requests(persistent, 1, borrowed=False)[0]
        requests[1].layers[0].conv_states[0].add_(1)
        with self.assertRaises(GDNStorageWitnessError) as raised:
            verify_request_gdn_binding_guard(
                guard, requests, completed_request_indices=[0]
            )
        self.assertEqual(raised.exception.gate_id, GATE_INCOMPLETE_BINDING_UNCHANGED)

        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=False)
        guard = capture_request_gdn_binding_guard(
            requests,
            LINEAR_LAYERS,
            policy=GDN_MATERIALIZE_REQUEST_BASE,
        )
        requests[0] = make_requests(persistent, 1, borrowed=False)[0]
        replacement = torch.ones_like(requests[1].layers[0].conv_states[0])
        requests[1].layers[0].conv_states[0].set_(replacement)
        with self.assertRaises(GDNStorageWitnessError) as raised:
            verify_request_gdn_binding_guard(
                guard, requests, completed_request_indices=[0]
            )
        self.assertEqual(raised.exception.gate_id, GATE_INCOMPLETE_BINDING_UNCHANGED)

    def test_four_factor_cells_replay_one_bound_timeline(self):
        for kv_policy in (
            "vllm-q16-fresh-full-copy-control",
            "vllm-q16-shared-document-reuse",
        ):
            for runtime_policy, canonical_policy, borrowed in (
                (GDN_BORROW_IMMUTABLE_BASE, POLICY_SHARED_BASE, True),
                (GDN_MATERIALIZE_REQUEST_BASE, POLICY_MATERIALIZED, False),
            ):
                persistent = make_persistent()
                requests = make_requests(persistent, 2, borrowed=borrowed)
                persistent_guard = capture_persistent_gdn_guard(
                    persistent, LINEAR_LAYERS
                )
                request_guard = capture_request_gdn_binding_guard(
                    requests, LINEAR_LAYERS, policy=runtime_policy
                )

                def phase_row(phase, completed):
                    return capture_gdn_phase_witness(
                        persistent,
                        requests,
                        LINEAR_LAYERS,
                        run_id="unit-run",
                        cell_id=f"{kv_policy}/{canonical_policy}",
                        kv_policy=kv_policy,
                        phase=phase,
                        policy=runtime_policy,
                        persistent_guard=persistent_guard,
                        request_guard=request_guard,
                        completed_request_indices=completed,
                    )

                phases = [
                    phase_row(PHASE_SETUP_BORROWED_IMMUTABLE, None),
                ]
                requests[0] = make_requests(persistent, 1, borrowed=False)[0]
                phases.append(phase_row(PHASE_POST_TRANSITION, [0]))
                requests[1] = make_requests(persistent, 1, borrowed=False)[0]
                phases.append(phase_row(PHASE_POST_GENERATION, [0, 1]))
                bundle = {
                    "schema_version": TIMELINE_SCHEMA_VERSION,
                    "run_id": "unit-run",
                    "cell_id": f"{kv_policy}/{canonical_policy}",
                    "kv_policy": kv_policy,
                    "gdn_policy": canonical_policy,
                    "group_gdn_base_policy": runtime_policy,
                    "resident_count": 2,
                    "layer_indices": list(LINEAR_LAYERS),
                    "state_index": 0,
                    "phases": phases,
                }
                replay = replay_gdn_storage_timeline(json_round_trip(bundle))
                self.assertTrue(replay["passed"])
                self.assertTrue(replay["completed_all_requests"])

                all_at_once = {
                    **bundle,
                    "phases": [
                        phases[0],
                        phase_row(PHASE_POST_TRANSITION, [0, 1]),
                        phase_row(PHASE_POST_GENERATION, [0, 1]),
                    ],
                }
                with self.assertRaisesRegex(
                    GDNStorageWitnessError,
                    "first transition witness must complete exactly request 0",
                ):
                    replay_gdn_storage_timeline(json_round_trip(all_at_once))

                bad = json_round_trip(bundle)
                bad["phases"][1]["binding_witness"]["guard_id"] = "0" * 32
                with self.assertRaisesRegex(
                    GDNStorageWitnessError,
                    "request guard disagreement|guard ID drift",
                ):
                    replay_gdn_storage_timeline(bad)

    def test_request_binding_guard_proves_rebound_and_unchanged_timing(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=True)
        guard = capture_request_gdn_binding_guard(
            requests,
            LINEAR_LAYERS,
            policy=GDN_BORROW_IMMUTABLE_BASE,
        )
        setup = verify_request_gdn_binding_guard(
            guard, requests, completed_request_indices=[]
        )
        self.assertEqual(
            replay_request_gdn_binding_witness(json_round_trip(setup))[
                "unchanged_tensor_count"
            ],
            120,
        )
        requests[0] = make_requests(persistent, 1, borrowed=False)[0]
        transition = verify_request_gdn_binding_guard(
            guard, requests, completed_request_indices=[0]
        )
        replay = replay_request_gdn_binding_witness(json_round_trip(transition))
        self.assertEqual(replay["rebound_tensor_count"], 60)
        self.assertEqual(replay["unchanged_tensor_count"], 60)

        requests[1] = make_requests(persistent, 1, borrowed=False)[0]
        with self.assertRaises(GDNStorageWitnessError) as raised:
            verify_request_gdn_binding_guard(
                guard, requests, completed_request_indices=[0]
            )
        self.assertEqual(raised.exception.gate_id, GATE_INCOMPLETE_BINDING_UNCHANGED)

    def test_request_binding_guard_rejects_completed_without_rebind_and_tamper(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 1, borrowed=False)
        guard = capture_request_gdn_binding_guard(
            requests,
            LINEAR_LAYERS,
            policy=GDN_MATERIALIZE_REQUEST_BASE,
        )
        with self.assertRaises(GDNStorageWitnessError) as raised:
            verify_request_gdn_binding_guard(
                guard, requests, completed_request_indices=[0]
            )
        self.assertEqual(raised.exception.gate_id, GATE_COMPLETED_BINDING_REBOUND)

        setup = verify_request_gdn_binding_guard(
            guard, requests, completed_request_indices=[]
        )

        # A fresh Tensor object that is only a view of the old storage is not
        # an out-of-place functional rebind.
        for layer in requests[0].layers:
            layer.conv_states[0] = layer.conv_states[0].view_as(layer.conv_states[0])
            layer.recurrent_states[0] = layer.recurrent_states[0].view_as(
                layer.recurrent_states[0]
            )
        with self.assertRaises(GDNStorageWitnessError) as raised:
            verify_request_gdn_binding_guard(
                guard, requests, completed_request_indices=[0]
            )
        self.assertEqual(raised.exception.gate_id, GATE_COMPLETED_BINDING_REBOUND)

        requests[0] = make_requests(persistent, 1, borrowed=False)[0]
        requests[0].layers[0].conv_states[0] = torch.zeros(1, 1, 12)
        with self.assertRaises(GDNStorageWitnessError) as raised:
            verify_request_gdn_binding_guard(
                guard, requests, completed_request_indices=[0]
            )
        self.assertEqual(raised.exception.gate_id, GATE_COMPLETED_BINDING_REBOUND)

        setup["rows"][0]["observed_binding_token"] = "0" * 64
        setup["rows_sha256"] = hashlib_sha256_json(setup["rows"])
        with self.assertRaises(GDNStorageWitnessError) as raised:
            replay_request_gdn_binding_witness(setup)
        self.assertEqual(raised.exception.gate_id, GATE_INCOMPLETE_BINDING_UNCHANGED)

    def test_completed_rebind_allows_different_private_backing_capacity(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 1, borrowed=False)
        guard = capture_request_gdn_binding_guard(
            requests,
            LINEAR_LAYERS,
            policy=GDN_MATERIALIZE_REQUEST_BASE,
        )
        for layer in requests[0].layers:
            for family in (layer.conv_states, layer.recurrent_states):
                original = family[0]
                expanded_stride = tuple(
                    int(item) * 2 for item in original.stride()
                )
                replacement = torch.empty_strided(
                    tuple(original.shape),
                    expanded_stride,
                    dtype=original.dtype,
                    device=original.device,
                )
                replacement.copy_(original)
                family[0] = replacement
                self.assertGreater(
                    family[0].untyped_storage().nbytes(),
                    family[0].numel() * family[0].element_size(),
                )
        record = verify_request_gdn_binding_guard(
            guard, requests, completed_request_indices=[0]
        )
        self.assertEqual(
            replay_request_gdn_binding_witness(json_round_trip(record))[
                "rebound_tensor_count"
            ],
            60,
        )

    def test_real_gdn_factor_names_are_accepted_but_kv_factor_names_are_rejected(self):
        persistent = make_persistent()
        borrowed = make_requests(persistent, 1, borrowed=True)
        materialized = make_requests(persistent, 1, borrowed=False)
        for policy, requests in (
            (GDN_BORROW_IMMUTABLE_BASE, borrowed),
            (GDN_MATERIALIZE_REQUEST_BASE, materialized),
        ):
            snapshot = capture(
                persistent,
                requests,
                phase=PHASE_SETUP_BORROWED_IMMUTABLE,
                policy=policy,
            )
            self.assertTrue(replay_gdn_storage_witness(snapshot)["passed"])
        for kv_policy in (
            "vllm-q16-shared-document-reuse",
            "vllm-q16-fresh-full-copy-control",
        ):
            with self.assertRaisesRegex(GDNStorageWitnessError, "unsupported GDN ownership policy"):
                capture(
                    persistent,
                    borrowed,
                    phase=PHASE_SETUP_BORROWED_IMMUTABLE,
                    policy=kv_policy,
                )

    def test_clean_shared_setup_has_60_exact_aliases_per_request_and_no_pointer(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=True)
        snapshot = capture(
            persistent,
            requests,
            phase=PHASE_SETUP_BORROWED_IMMUTABLE,
            policy=POLICY_SHARED_BASE,
        )
        serialized = json.dumps(snapshot, sort_keys=True)
        self.assertNotIn("data_ptr", serialized)
        self.assertNotIn("storage_ptr", serialized)
        replay = replay_gdn_storage_witness(json_round_trip(snapshot))
        self.assertTrue(replay["passed"])
        self.assertEqual(replay["exact_alias_comparisons"], 2 * 60)
        self.assertEqual(replay["tensor_count_per_owner"], EXPECTED_TENSORS_PER_OWNER)

    def test_clean_materialized_setup_and_post_phases_are_disjoint(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=False)
        setup = capture(
            persistent,
            requests,
            phase=PHASE_SETUP_BORROWED_IMMUTABLE,
            policy=POLICY_MATERIALIZED,
        )
        setup_replay = replay_gdn_storage_witness(json_round_trip(setup))
        self.assertGreater(setup_replay["disjoint_comparisons"], 0)

        transition = capture(
            persistent,
            requests,
            phase=PHASE_POST_TRANSITION,
            policy=POLICY_MATERIALIZED,
            completed=[0],
        )
        transition_replay = replay_gdn_storage_witness(json_round_trip(transition))
        self.assertEqual(transition_replay["completed_request_indices"], [0])

        generation = capture(
            persistent,
            requests,
            phase=PHASE_POST_GENERATION,
            policy=POLICY_MATERIALIZED,
        )
        generation_replay = replay_gdn_storage_witness(json_round_trip(generation))
        self.assertEqual(generation_replay["completed_request_indices"], [0, 1])

    def test_shared_post_transition_requires_incomplete_request_to_still_borrow_base(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=True)
        # Request 0 completed its first transition and became private.  Request
        # 1 has not run yet and must still borrow the exact immutable base.
        private = make_requests(persistent, 1, borrowed=False)[0]
        requests[0] = private
        clean = capture(
            persistent,
            requests,
            phase=PHASE_POST_TRANSITION,
            policy=GDN_BORROW_IMMUTABLE_BASE,
            completed=[0],
        )
        self.assertTrue(replay_gdn_storage_witness(clean)["passed"])

        # Prematurely materializing the incomplete request is temporal drift,
        # even though it remains disjoint from every other owner.
        requests[1] = make_requests(persistent, 1, borrowed=False)[0]
        bad = capture(
            persistent,
            requests,
            phase=PHASE_POST_TRANSITION,
            policy=GDN_BORROW_IMMUTABLE_BASE,
            completed=[0],
        )
        with self.assertRaises(GDNStorageWitnessError) as raised:
            replay_gdn_storage_witness(bad)
        self.assertEqual(
            raised.exception.gate_id,
            GATE_SHARED_INCOMPLETE_EXACT_BASE_ALIAS,
        )

    def test_request_request_exact_alias_fails_materialized_setup(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=False)
        requests[1].layers[7].conv_states[0] = requests[0].layers[7].conv_states[0]
        snapshot = capture(
            persistent,
            requests,
            phase=PHASE_SETUP_BORROWED_IMMUTABLE,
            policy=POLICY_MATERIALIZED,
        )
        snapshot["passed"] = True
        with self.assertRaisesRegex(GDNStorageWitnessError, r"request\[0\]/request\[1\].*exact alias") as raised:
            replay_gdn_storage_witness(snapshot)
        self.assertEqual(raised.exception.gate_id, GATE_MATERIALIZED_SETUP_PEERS_DISJOINT)

    def test_completed_request_peer_alias_uses_stable_mutant_gate(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=False)
        requests[1].layers[7].conv_states[0] = requests[0].layers[7].conv_states[0]
        snapshot = capture(
            persistent,
            requests,
            phase=PHASE_POST_TRANSITION,
            policy=POLICY_SHARED_BASE,
            completed=[0, 1],
        )
        with self.assertRaises(GDNStorageWitnessError) as raised:
            replay_gdn_storage_witness(snapshot)
        self.assertEqual(raised.exception.gate_id, GATE_COMPLETED_VS_PEERS_DISJOINT)

    def test_completed_request_base_exact_alias_fails_even_when_passed_is_true(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=True)
        requests[0] = make_requests(persistent, 1, borrowed=False)[0]
        requests[0].layers[3].recurrent_states[0] = persistent.layers[3].recurrent_states[0]
        snapshot = capture(
            persistent,
            requests,
            phase=PHASE_POST_TRANSITION,
            policy=POLICY_SHARED_BASE,
            completed=[0],
        )
        snapshot["passed"] = True
        with self.assertRaisesRegex(GDNStorageWitnessError, r"completed request\[0\]/persistent.*exact alias") as raised:
            replay_gdn_storage_witness(snapshot)
        self.assertEqual(raised.exception.gate_id, GATE_COMPLETED_VS_BASE_DISJOINT)

    def test_partial_byte_overlap_is_not_mistaken_for_disjoint_or_exact_alias(self):
        persistent = make_persistent()
        backing = torch.arange(16, dtype=torch.float32)
        persistent.layers[0].conv_states[0] = backing[:8]
        guard = capture_persistent_gdn_guard(persistent, LINEAR_LAYERS)
        requests = make_requests(persistent, 1, borrowed=False)
        requests[0].layers[0].conv_states[0] = backing[4:12]
        snapshot = capture(
            persistent,
            requests,
            phase=PHASE_POST_TRANSITION,
            policy=POLICY_SHARED_BASE,
            completed=[0],
            guard=guard,
        )
        with self.assertRaisesRegex(GDNStorageWitnessError, "partial overlap"):
            replay_gdn_storage_witness(snapshot)

    def test_n_equals_one_post_transition_proof_is_nonvacuous(self):
        persistent = make_persistent()
        clean_requests = make_requests(persistent, 1, borrowed=False)
        clean = capture(
            persistent,
            clean_requests,
            phase=PHASE_POST_TRANSITION,
            policy=POLICY_SHARED_BASE,
            completed=[0],
        )
        clean["passed"] = False
        report = replay_gdn_storage_witness(clean)
        self.assertTrue(report["passed"])
        self.assertTrue(report["ownership_proof_nonvacuous"])
        self.assertEqual(report["disjoint_comparisons"], 60 * 60)

        borrowed_request = make_requests(persistent, 1, borrowed=True)
        bad = capture(
            persistent,
            borrowed_request,
            phase=PHASE_POST_TRANSITION,
            policy=POLICY_SHARED_BASE,
            completed=[0],
        )
        with self.assertRaisesRegex(GDNStorageWitnessError, r"completed request\[0\]/persistent"):
            replay_gdn_storage_witness(bad)

    def test_missing_tensor_and_missing_serialized_schema_fail_closed(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 1, borrowed=True)
        del requests[0].layers[11].recurrent_states[0]
        with self.assertRaisesRegex(GDNStorageWitnessError, "missing state 0"):
            capture(
                persistent,
                requests,
                phase=PHASE_SETUP_BORROWED_IMMUTABLE,
                policy=POLICY_SHARED_BASE,
            )

        requests = make_requests(persistent, 1, borrowed=True)
        snapshot = capture(
            persistent,
            requests,
            phase=PHASE_SETUP_BORROWED_IMMUTABLE,
            policy=POLICY_SHARED_BASE,
        )
        malformed = copy.deepcopy(snapshot)
        malformed["rows"][0].pop("shape")
        malformed["rows_sha256"] = hashlib_sha256_json(malformed["rows"])
        malformed["passed"] = True
        with self.assertRaisesRegex(GDNStorageWitnessError, "missing schema fields.*shape"):
            replay_gdn_storage_witness(malformed)

    def test_persistent_binding_and_digest_drift_are_distinct_failures(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 1, borrowed=True)
        guard = capture_persistent_gdn_guard(persistent, LINEAR_LAYERS)
        persistent.layers[0].conv_states[0].add_(1)
        with self.assertRaisesRegex(GDNStorageWitnessError, "persistent digest drift"):
            capture(
                persistent,
                requests,
                phase=PHASE_SETUP_BORROWED_IMMUTABLE,
                policy=POLICY_SHARED_BASE,
                guard=guard,
            )

        persistent = make_persistent()
        requests = make_requests(persistent, 1, borrowed=True)
        guard = capture_persistent_gdn_guard(persistent, LINEAR_LAYERS)
        persistent.layers[0].conv_states[0] = persistent.layers[0].conv_states[0].clone()
        with self.assertRaisesRegex(GDNStorageWitnessError, "persistent binding drift"):
            capture(
                persistent,
                requests,
                phase=PHASE_SETUP_BORROWED_IMMUTABLE,
                policy=POLICY_SHARED_BASE,
                guard=guard,
            )

    def test_post_generation_requires_all_requests_completed(self):
        persistent = make_persistent()
        requests = make_requests(persistent, 2, borrowed=False)
        with self.assertRaisesRegex(GDNStorageWitnessError, "every resident request"):
            capture(
                persistent,
                requests,
                phase=PHASE_POST_GENERATION,
                policy=POLICY_SHARED_BASE,
                completed=[0],
            )


def hashlib_sha256_json(value) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()


if __name__ == "__main__":
    unittest.main()
