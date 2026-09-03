#!/usr/bin/env python3
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path

import run_boolq_ordered as mod


HERE = Path(__file__).resolve().parent


def load_protocol():
    return json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))


class FakeClient:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.index = 0

    def complete(self, task, request_id, seed):
        text = self.outputs[self.index]
        self.index += 1
        dispatch = time.monotonic_ns()
        wall = mod.utc_now()
        vote = mod.parse_vote(text)
        return mod.Completion(
            request_id=request_id,
            response_id="fake-%d" % self.index,
            raw_output=text,
            vote=vote,
            dispatch_monotonic_ns=dispatch,
            first_token_monotonic_ns=dispatch + 1,
            final_token_monotonic_ns=dispatch + 2,
            response_arrival_monotonic_ns=dispatch + 3,
            dispatch_utc=wall,
            first_token_utc=wall,
            final_token_utc=wall,
            response_arrival_utc=wall,
            local_input_tokens=10,
            local_output_tokens=1,
            provider_prompt_tokens=10,
            provider_completion_tokens=1,
            provider_total_tokens=11,
            finish_reason="stop",
            seed=seed,
            server_rank=0,
        )


def prompt_task(shadow=False):
    return mod.PromptTask(
        task_id="1" * 64,
        passage_id="2" * 64,
        internal_split="TEST",
        source_split="train",
        source_index=0,
        passage="A passage.",
        question="A question?",
        rendered_prompt="prompt",
        prompt_sha256=mod.sha256_bytes(b"prompt"),
        input_tokens=10,
        shadow=shadow,
    )


class ParserTests(unittest.TestCase):
    def test_first_canonical_word_and_strictness(self):
        self.assertEqual(mod.parse_vote("Yes"), mod.ParsedVote(True, "yes", True))
        self.assertEqual(mod.parse_vote(" no "), mod.ParsedVote(False, "no", True))
        self.assertEqual(mod.parse_vote("Maybe YES, then no."), mod.ParsedVote(True, "yes", False))
        self.assertEqual(mod.parse_vote("No, actually yes"), mod.ParsedVote(False, "no", False))
        self.assertEqual(mod.parse_vote("unknown"), mod.ParsedVote(False, None, False))

    def test_majority_tie_is_no(self):
        self.assertFalse(mod.majority_yes([True, False]))
        self.assertTrue(mod.majority_yes([True, True, False]))


class TheoryTests(unittest.TestCase):
    def test_zero_denominator_is_unsupported(self):
        H = [0.0] * 33
        H[0] = 1.0
        table = mod.build_cert_table(H, 32)
        self.assertIsNone(table[1][1])
        self.assertEqual(table[1][0], 0.0)

    def test_unsupported_states_cannot_stop(self):
        H = [0.0] * 33
        H[0] = 1.0
        table = mod.build_cert_table(H, 32)
        flip, expected_k = mod.dp_adaptive(32, 32, table, 0.05, 3)
        self.assertEqual(flip, 0.0)
        self.assertEqual(expected_k, 32.0)

    def test_empirical_bernstein(self):
        self.assertGreater(mod.empirical_bernstein_ucb([0.0] * 3000, 0.05), 0.0)
        self.assertLess(mod.empirical_bernstein_ucb([0.0] * 3000, 0.05), 0.01)

    def test_input_length_gate(self):
        summary = mod.input_length_summary([53, 160, 1295], 4, 2048)
        self.assertEqual(summary["max"], 1295)
        self.assertEqual(summary["above_input_budget"], 0)
        with self.assertRaises(mod.DataError):
            mod.input_length_summary([2045], 4, 2048)


class ProtocolTests(unittest.TestCase):
    def test_frozen_constants(self):
        protocol = load_protocol()
        self.assertEqual(protocol["schema"], mod.SCHEMA)
        self.assertEqual(protocol["carrier"]["rows"], 12697)
        self.assertEqual(protocol["carrier"]["canonical_jsonl_sha256"], "13c2f4143ae320a0191c6de5be919248a20c15515f58c6deb7d3732068f2d31a")
        self.assertEqual(protocol["allocation"]["seed"], "20260822-A11-BOOLQ-v2-passage")
        self.assertEqual(protocol["allocation"]["eligible_passage_representatives"], 10144)
        self.assertEqual(protocol["allocation"]["selected_manifest_sha256"], "c1cb98d45600db7c234396c73161921905c4fa414a0cdd57f02b8d304d5505d8")
        self.assertEqual(protocol["allocation"]["shadow_manifest_sha256"], "89fd7fb6b42dad981b564116e755aecbc3ab2c41dcd14e5e0a65b7bd1053d013")
        self.assertEqual(protocol["model"]["revision"], "a09a35458c702b33eeacc393d103063234e8bc28")
        self.assertEqual(len(protocol["model"]["snapshot_files"]), 14)
        self.assertEqual(protocol["model"]["snapshot_manifest_sha256"], "3ee6c9510b7e50bfcd46d6df33cafa3e2019f13a6a09bf1d2f9e80cdfe1164e8")
        self.assertEqual(protocol["policy"]["calibration_family"]["family_size"], 1)
        self.assertEqual(protocol["policy"]["primary_alpha"], 0.05)
        self.assertEqual(protocol["inference"]["max_tokens"], 4)
        self.assertTrue(protocol["inference"]["strictly_one_inflight_request_per_task"])

    def test_task_identity_uses_only_prompt_pair(self):
        base = {"passage": "p", "question": "q", "source_split": "train", "source_index": 1}
        changed = dict(base, source_split="validation", source_index=999)
        self.assertEqual(mod.task_identity(base), mod.task_identity(changed))
        self.assertEqual(mod.task_identity(base), mod.sha256_bytes(b"p\0q"))
        padded = dict(base, passage=" p ", question=" q\n")
        self.assertEqual(mod.task_identity(base), mod.task_identity(padded))
        self.assertEqual(mod.passage_identity(base), mod.passage_identity(padded))

    def test_prompt_task_has_no_gold(self):
        self.assertNotIn("gold", mod.PromptTask.__dataclass_fields__)

    def test_balanced_arm_order(self):
        tasks = [copy.copy(prompt_task()) for _ in range(10)]
        tasks = [mod.dataclasses.replace(task, task_id=("%064x" % index)) for index, task in enumerate(tasks)]
        orders = mod.balanced_arm_orders("seed", tasks)
        self.assertEqual(sum(order[0] == "FULL-N" for order in orders.values()), 5)
        self.assertEqual(sum(order[0] == "BAYES-H-online" for order in orders.values()), 5)

    def test_complete_model_snapshot_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = {}
            for index in range(14):
                name = "file-%02d" % index
                payload = ("payload-%02d" % index).encode("ascii")
                (root / name).write_bytes(payload)
                expected[name] = mod.sha256_bytes(payload)
            (root / ".cache").mkdir()
            ledger = "".join("%s  %s\n" % (expected[name], name) for name in sorted(expected))
            spec = {
                "snapshot_manifest_contract": "synthetic fixture",
                "snapshot_manifest_sha256": mod.sha256_bytes(ledger.encode("utf-8")),
                "snapshot_files": expected,
            }
            receipt = mod.verify_model_snapshot(root, spec)
            self.assertEqual(receipt["file_count"], 14)
            self.assertEqual(receipt["ledger_sha256"], spec["snapshot_manifest_sha256"])
            (root / "unexpected").write_text("x", encoding="utf-8")
            with self.assertRaises(mod.FormalError):
                mod.verify_model_snapshot(root, spec)


class EpisodeTests(unittest.TestCase):
    def test_online_stop_then_globally_deferred_shadow_continuation(self):
        H = [0.0] * 33
        H[32] = 1.0
        table = mod.build_cert_table(H, 32)
        table_sha = mod.sha256_json(table)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = mod.LedgerSet(root)
            try:
                client = FakeClient(["Yes"] * 32)
                outcome = mod.run_online_episode(
                    prompt_task(shadow=True),
                    client,
                    ledger,
                    "seed",
                    32,
                    3,
                    0.05,
                    table,
                    table_sha,
                )
                global_primary_seal = time.monotonic_ns()
                mod.continue_shadow_episode(
                    prompt_task(shadow=True),
                    outcome,
                    client,
                    ledger,
                    "seed",
                    32,
                    global_primary_seal,
                )
                ledger.flush()
            finally:
                ledger.close()
            self.assertEqual(outcome.stop_k, 3)
            self.assertTrue(outcome.delivered_yes)
            self.assertTrue(outcome.shadow_full_yes)
            traces = list(mod.read_jsonl(root / "rollout_trace.jsonl"))
            self.assertEqual(len(traces), 32)
            self.assertTrue(all(not row["excluded_from_primary_cost"] for row in traces[:3]))
            self.assertTrue(all(row["excluded_from_primary_cost"] for row in traces[3:]))
            stop = next(mod.read_jsonl(root / "stop_decision.jsonl"))
            payload = dict(stop)
            seal = payload.pop("decision_payload_sha256")
            self.assertEqual(seal, mod.sha256_json(payload))
            self.assertTrue(all(row["dispatch_monotonic_ns"] > stop["decision_monotonic_ns"] for row in traces[3:]))
            self.assertTrue(all(row["dispatch_monotonic_ns"] > global_primary_seal for row in traces[3:]))
            self.assertFalse(any("gold" in row for row in traces))
            cancellation = next(mod.read_jsonl(root / "cancellation_ledger.jsonl"))
            self.assertEqual(cancellation["in_flight_request_ids"], [])
            self.assertEqual(cancellation["cancellation_status"], "not_applicable_sequential_no_prefetch")

    def test_execute_test_seals_all_primaries_before_any_shadow(self):
        protocol = load_protocol()
        protocol["policy"]["N"] = 3
        protocol["policy"]["minimum_stop_k"] = 1
        protocol["allocation"]["shadow_test"] = 1
        protocol["inference"]["max_concurrent_tasks_per_server"] = 1
        first = prompt_task(shadow=True)
        second = mod.dataclasses.replace(prompt_task(shadow=False), task_id="3" * 64, passage_id="4" * 64)
        H = [0.0] * 4
        H[3] = 1.0
        table = mod.build_cert_table(H, 3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "locks").mkdir()
            ledger = mod.LedgerSet(root)
            try:
                pairs = mod.execute_test(
                    [first, second],
                    [FakeClient(["Yes"] * 10)],
                    ledger,
                    protocol,
                    table,
                    mod.sha256_json(table),
                    root,
                )
                ledger.flush()
            finally:
                ledger.close()
            self.assertEqual(len(pairs), 2)
            seal = json.loads((root / "locks" / "all_test_primary_decisions_sealed.json").read_text(encoding="utf-8"))
            self.assertEqual(seal["decision_count"], 4)
            traces = list(mod.read_jsonl(root / "rollout_trace.jsonl"))
            shadow = [row for row in traces if row["excluded_from_primary_cost"]]
            self.assertEqual(len(shadow), 2)
            self.assertTrue(all(row["dispatch_monotonic_ns"] > seal["seal_monotonic_ns"] for row in shadow))

    def test_calibration_uses_single_exact_dp_candidate(self):
        protocol = load_protocol()
        protocol["allocation"]["cal"] = 3000
        H = [0.0] * 33
        H[32] = 1.0
        table = mod.build_cert_table(H, 32)
        fit = {"certificate_table": table, "certificate_table_sha256": mod.sha256_json(table)}
        outcomes = []
        for index in range(3000):
            outcomes.append(mod.EpisodeOutcome(
                task_id="%064x" % index,
                source_split="train",
                arm="FULL-N-calibration",
                episode_id="CAL:%d" % index,
                votes=[True] * 32,
                delivered_yes=True,
                stop_k=32,
                decision_monotonic_ns=10,
                decision_utc="2026-08-22T00:00:00Z",
                first_dispatch_monotonic_ns=1,
                first_token_monotonic_ns=2,
                completed_output_tokens=32,
                completed_input_tokens=320,
                strict_compliant_count=32,
                missing_canonical_count=0,
                decision_sha256="x",
            ))
        lock = mod.calibration_lock(outcomes, fit, protocol)
        self.assertEqual(lock["family_size"], 1)
        self.assertTrue(lock["primary_accepted"])
        self.assertIn("exact g_r(K)", lock["primary_candidate"]["loss_source"])


if __name__ == "__main__":
    unittest.main()
