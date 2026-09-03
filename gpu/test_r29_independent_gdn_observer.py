from __future__ import annotations

import ast
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from r29_independent_gdn_observer import (
    IndependentObserverError,
    ObserverSession,
    PHASE_GENERATION,
    PHASE_SETUP,
    PHASE_TRANSITION,
    POLICY_MATERIALIZED,
    POLICY_SHARED,
    compare_candidate_snapshot,
    evaluate_lifecycle,
    evaluate_phase,
)
from r29_replay_independent_gdn_observer import replay_result


LAYERS = tuple(range(3))


def make_persistent() -> SimpleNamespace:
    layers = []
    for index in LAYERS:
        layers.append(
            SimpleNamespace(
                conv_states={
                    0: torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
                    + index
                },
                recurrent_states={
                    0: torch.arange(20, dtype=torch.float32).reshape(1, 2, 2, 5)
                    + 100
                    + index
                },
            )
        )
    return SimpleNamespace(layers=layers)


def make_request(persistent: SimpleNamespace, *, borrowed: bool) -> SimpleNamespace:
    layers = []
    for source in persistent.layers:
        layers.append(
            SimpleNamespace(
                conv_states={
                    0: source.conv_states[0]
                    if borrowed
                    else source.conv_states[0].clone()
                },
                recurrent_states={
                    0: source.recurrent_states[0]
                    if borrowed
                    else source.recurrent_states[0].clone()
                },
            )
        )
    return SimpleNamespace(layers=layers)


def rebind(request: SimpleNamespace, delta: float) -> None:
    for layer in request.layers:
        layer.conv_states[0] = layer.conv_states[0].clone() + delta
        layer.recurrent_states[0] = layer.recurrent_states[0].clone() + delta


def candidate_from_observer(snapshot):
    value = copy.deepcopy(snapshot)
    identities = {}
    for row in value["rows"]:
        token = row.pop("storage_token")
        row.pop("tensor_token")
        if token not in identities:
            identities[token] = f"storage-{len(identities):04d}"
        row["storage_id"] = identities[token]
    return value


class IndependentObserverTest(unittest.TestCase):
    def _clean_lifecycle(self, policy: str):
        persistent = make_persistent()
        borrowed = policy == POLICY_SHARED
        requests = [
            make_request(persistent, borrowed=borrowed),
            make_request(persistent, borrowed=borrowed),
        ]
        session = ObserverSession()
        setup = session.capture(
            persistent,
            requests,
            LAYERS,
            phase=PHASE_SETUP,
            policy=policy,
            completed_request_indices=[],
        )
        rebind(requests[0], 1)
        transition = session.capture(
            persistent,
            requests,
            LAYERS,
            phase=PHASE_TRANSITION,
            policy=policy,
            completed_request_indices=[0],
        )
        rebind(requests[1], 2)
        generation = session.capture(
            persistent,
            requests,
            LAYERS,
            phase=PHASE_GENERATION,
            policy=policy,
            completed_request_indices=[0, 1],
        )
        return setup, transition, generation

    def test_module_has_no_candidate_import(self):
        source = Path(__file__).with_name("r29_independent_gdn_observer.py").read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)
        modules = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.append(node.module or "")
        self.assertFalse(
            any(
                "qcomem" in module.lower() or "forkaudit" in module.lower()
                for module in modules
            ),
            modules,
        )

    def test_clean_shared_and_materialized_lifecycles(self):
        for policy in (POLICY_SHARED, POLICY_MATERIALIZED):
            with self.subTest(policy=policy):
                snapshots = self._clean_lifecycle(policy)
                report = evaluate_lifecycle(snapshots)
                self.assertTrue(report["passed"])
                self.assertEqual(report["persistent_unchanged_tensor_count"], 6)
                self.assertEqual(report["request0_rebound_tensor_count"], 6)
                self.assertEqual(report["request1_rebound_tensor_count"], 6)

    def test_pointer_free_snapshot_and_setup_aba_pin(self):
        persistent = make_persistent()
        requests = [make_request(persistent, borrowed=True) for _ in range(2)]
        session = ObserverSession()
        setup = session.capture(
            persistent,
            requests,
            LAYERS,
            phase=PHASE_SETUP,
            policy=POLICY_SHARED,
            completed_request_indices=[],
        )
        raw = json.dumps(setup, sort_keys=True).lower()
        self.assertNotIn("data_ptr", raw)
        self.assertNotIn("storage_ptr", raw)
        self.assertNotIn("absolute_address", raw)
        self.assertTrue(setup["setup_tensor_refs_pinned_against_aba"])

    def test_independent_candidate_relation_comparison(self):
        setup, _transition, _generation = self._clean_lifecycle(POLICY_SHARED)
        candidate = candidate_from_observer(setup)
        report = compare_candidate_snapshot(setup, candidate)
        self.assertTrue(report["passed"])
        self.assertEqual(report["descriptor_mismatch_count"], 0)
        self.assertEqual(report["relation_mismatch_count"], 0)

        tampered = copy.deepcopy(candidate)
        tampered["rows"][6]["storage_id"] = "storage-9999"
        bad = compare_candidate_snapshot(setup, tampered)
        self.assertFalse(bad["passed"])
        self.assertGreater(bad["relation_mismatch_count"], 0)

    def test_internal_alias_fault_is_rejected(self):
        persistent = make_persistent()
        request = make_request(persistent, borrowed=False)
        request.layers[1].conv_states[0] = request.layers[0].conv_states[0]
        session = ObserverSession()
        snapshot = session.capture(
            persistent,
            [request],
            LAYERS,
            phase=PHASE_SETUP,
            policy=POLICY_MATERIALIZED,
            completed_request_indices=[],
        )
        with self.assertRaisesRegex(IndependentObserverError, "overlapping"):
            evaluate_phase(snapshot)

    def test_in_place_content_change_is_not_mislabeled_as_rebind(self):
        persistent = make_persistent()
        requests = [make_request(persistent, borrowed=False) for _ in range(2)]
        session = ObserverSession()
        setup = session.capture(
            persistent,
            requests,
            LAYERS,
            phase=PHASE_SETUP,
            policy=POLICY_MATERIALIZED,
            completed_request_indices=[],
        )
        for layer in requests[0].layers:
            layer.conv_states[0].add_(1)
            layer.recurrent_states[0].add_(1)
        transition = session.capture(
            persistent,
            requests,
            LAYERS,
            phase=PHASE_TRANSITION,
            policy=POLICY_MATERIALIZED,
            completed_request_indices=[0],
        )
        rebind(requests[1], 2)
        generation = session.capture(
            persistent,
            requests,
            LAYERS,
            phase=PHASE_GENERATION,
            policy=POLICY_MATERIALIZED,
            completed_request_indices=[0, 1],
        )
        with self.assertRaisesRegex(IndependentObserverError, "out-of-place"):
            evaluate_lifecycle((setup, transition, generation))

    def test_cpu_replay_ignores_candidate_passed_fields(self):
        cells = []
        for policy in (POLICY_SHARED, POLICY_MATERIALIZED):
            snapshots = self._clean_lifecycle(policy)
            phases = []
            for snapshot in snapshots:
                candidate = candidate_from_observer(snapshot)
                candidate["passed"] = False
                phases.append(
                    {
                        "phase": snapshot["phase"],
                        "independent_before_candidate_capture": snapshot,
                        "independent_after_candidate_capture": copy.deepcopy(snapshot),
                        "candidate_capture": candidate,
                        "independent_verdict": evaluate_phase(snapshot),
                        "independent_candidate_comparison": compare_candidate_snapshot(
                            snapshot, candidate
                        ),
                    }
                )
            cells.append(
                {
                    "cell_id": f"unit-{policy}",
                    "gdn_policy": policy,
                    "phases": phases,
                    "independent_lifecycle_verdict": evaluate_lifecycle(snapshots),
                }
            )
        replay = replay_result(
            {
                "schema_version": "forkaudit-r29-independent-gdn-observer-result-v1",
                "status": "completed_valid_scientific_execution",
                "cells": cells,
            }
        )
        self.assertTrue(replay["passed"])
        self.assertFalse(replay["candidate_passed_booleans_authoritative"])


if __name__ == "__main__":
    unittest.main()
