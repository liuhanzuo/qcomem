from __future__ import annotations

"""Static and synthetic replay tests for the frozen R33 five-fault map."""

import ast
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import r33_fault_replay as replay
import r33_aggregate_fresh_faults as aggregate


ROOT = Path(__file__).resolve().parents[2]
AUTHOR = ROOT / "paper_autonomous_multifork_iteration" / "evidence" / "r33_fresh_faults" / "author_freeze"
FAULTS_RAW_SHA256 = "b1f4d6c544c30fccc32370a03e170aee38596a370d02c0db4a6748c83cc34dff"
PROTOCOL_RAW_SHA256 = "b85995e180732588ac6ee09fc33181d9c276980795a7288a23feb4c94ad3925c"
ROW_SHA256 = {
    replay.FAULT_IDS[0]: "2ee27893a09cc9198f227422ec9fda1de1bebf97cc31b35fc1cfce67f773b8f2",
    replay.FAULT_IDS[1]: "20bbf518f3d2f66577db3e850400407658c8029975e03f6509a3e08f75d18970",
    replay.FAULT_IDS[2]: "6e2b0b4cca4f8a3b72d26e2f13aa6a2a47c5791dd8df44c452fb99bd7d42f282",
    replay.FAULT_IDS[3]: "24c88a88ea2991d16f4e7e63c457fcf92d2a95650ef23232fcf2a1c24d7a64f7",
    replay.FAULT_IDS[4]: "6dfbea24d869efeb4881155dfae1d710109a40017cb25bb3e80f621c266ec80a",
}


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding_rows(*, stale_r1: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for request_index in range(2):
        for coordinate in range(60):
            base_binding = hashlib.sha256(f"bb:{request_index}:{coordinate}".encode()).hexdigest()
            observed_binding = hashlib.sha256(f"ob:{request_index}:{coordinate}".encode()).hexdigest()
            if stale_r1 and request_index == 1:
                observed_binding = base_binding
            rows.append(
                {
                    "request_index": request_index,
                    "layer_index": coordinate // 2,
                    "state_family": "conv_states" if coordinate % 2 == 0 else "recurrent_states",
                    "state_index": 0,
                    "expected_relation": "rebound",
                    "baseline_binding_token": base_binding,
                    "observed_binding_token": observed_binding,
                    "baseline_storage_token": hashlib.sha256(f"bs:{request_index}:{coordinate}".encode()).hexdigest(),
                    "observed_storage_token": hashlib.sha256(f"os:{request_index}:{coordinate}".encode()).hexdigest(),
                }
            )
    return rows


def source_digests(seed: str) -> dict[str, str]:
    return {str(index): hashlib.sha256(f"{seed}:{index}".encode()).hexdigest() for index in range(10)}


def case_fixture(root: Path, fault_id: str, lane: str) -> dict[str, object]:
    source = source_digests("same")
    rows = binding_rows()
    sidecars = []
    for index in range(16):
        relative = Path("sidecars") / fault_id / lane / f"logits-{index:02d}.bin"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"{fault_id}:{lane}:{index}".encode()
        path.write_bytes(payload)
        sidecars.append(
            {"path": relative.as_posix(), "sha256": hashlib.sha256(payload).hexdigest(), "nbytes": len(payload)}
        )
    policy = replay.FAULT_POLICIES[fault_id]
    case: dict[str, object] = {
        "schema_version": "forkaudit-r33-executed-case-v1",
        "fault_id": fault_id,
        "lane": lane,
        "status": "full_horizon_completed",
        "operational_invalid": None,
        "all_existing_gates_enabled": True,
        "mandatory_coverage_complete": True,
        "byte_binding_passed": True,
        "dispatch_scope": {
            "python_call_scope": "full",
            "compiled_binary_identity": "partial",
            "autotuning_choice": "partial",
        },
        "rank": policy["rank"],
        "kv_policy": policy["kv_policy"],
        "gdn_policy": policy["gdn_policy"],
        "source_physical_digests": {"setup": dict(source), "transition": dict(source), "final": dict(source)},
        "gdn_binding_witness": {"rows": rows, "rows_sha256": replay.sha256_json(rows)},
        "logit_sidecars": sidecars,
        "ordered_model_schedule": replay.expected_clean_schedule(),
        "fault_specific_evidence": {},
        "injection_witness": None,
        "earlier_predicates": [],
        "cleanup": {
            "completed": True,
            "registered_backend_restored": True,
            "strong_references_released": True,
            "gc_collect_completed": True,
            "accelerator_cache_cleanup_completed": True,
            "accelerator_synchronize_completed": True,
            "allocator_baseline_exact": True,
            "cleanup_error": None,
        },
    }
    if fault_id == replay.FAULT_IDS[0]:
        case["fault_specific_evidence"] = {
            "ordered_tail_events": [
                {"ordinal": 0, "kind": "tail_copy"},
                {"ordinal": 1, "kind": "append_write"},
            ]
        }
    elif fault_id == replay.FAULT_IDS[3]:
        scale = float.hex(256 ** -0.5)
        case["fault_specific_evidence"] = {
            "target_call": {"frozen_scale_hex": scale, "observed_scale_hex": scale}
        }
    return case


def mutate_fixture(case: dict[str, object], fault_id: str) -> None:
    if fault_id == replay.FAULT_IDS[0]:
        case["fault_specific_evidence"] = {
            "ordered_tail_events": [
                {"ordinal": 0, "kind": "append_write"},
                {"ordinal": 1, "kind": "tail_copy"},
            ]
        }
    elif fault_id == replay.FAULT_IDS[1]:
        source = case["source_physical_digests"]
        assert isinstance(source, dict)
        transition = dict(source["setup"])
        transition["0"] = hashlib.sha256(b"one inactive byte changed").hexdigest()
        source["transition"] = transition
        source["final"] = dict(transition)
        case["fault_specific_evidence"] = {"layer_index": 0, "xor_mask": 1}
    elif fault_id == replay.FAULT_IDS[2]:
        case["ordered_model_schedule"] = replay.expected_hf03_schedule()
        case["fault_specific_evidence"] = {"extra_committed_call_count": 1}
    elif fault_id == replay.FAULT_IDS[3]:
        frozen = 256 ** -0.5
        case["fault_specific_evidence"] = {
            "target_call": {"frozen_scale_hex": float.hex(frozen), "observed_scale_hex": float.hex(2 * frozen)}
        }
    else:
        rows = binding_rows(stale_r1=True)
        case["gdn_binding_witness"] = {"rows": rows, "rows_sha256": replay.sha256_json(rows)}
        case["fault_specific_evidence"] = {"stale_request_index": 1, "stale_binding_token_count": 60}


class R33FaultMappingTest(unittest.TestCase):
    def test_warmup_releases_loop_aliases_before_allocator_freeze(self) -> None:
        source = ROOT / "paper_autonomous_multifork_iteration" / "scripts" / "r33_execute_fresh_faults.py"
        module = ast.parse(source.read_text(encoding="utf-8"))
        warmup = next(
            node
            for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "discarded_warmup"
        )
        cleanup_call_line = min(
            node.lineno
            for node in ast.walk(warmup)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_cleanup_allocator"
        )
        released_before_cleanup = {
            target.id
            for node in ast.walk(warmup)
            if isinstance(node, ast.Assign) and node.lineno < cleanup_call_line
            for target in node.targets
            if isinstance(target, ast.Name)
            and isinstance(node.value, ast.Constant)
            and node.value.value is None
        }
        self.assertTrue(
            {"request", "ledger", "logits"}.issubset(released_before_cleanup),
            "warm-up loop aliases must be released before freezing the allocator baseline",
        )

    def test_author_freeze_and_all_exact_bindings(self) -> None:
        self.assertEqual(file_sha(AUTHOR / "FAULTS.json"), FAULTS_RAW_SHA256)
        self.assertEqual(file_sha(AUTHOR / "PROTOCOL.md"), PROTOCOL_RAW_SHA256)
        frozen = json.loads((AUTHOR / "FAULTS.json").read_text())
        self.assertEqual(tuple(row["id"] for row in frozen["faults"]), replay.FAULT_IDS)
        self.assertEqual([replay.FAULT_POLICIES[item]["rank"] for item in replay.FAULT_IDS], list(range(5)))
        self.assertEqual(len(set(replay.EXPECTED_PRIMARY_GATES.values())), 5)
        for row in frozen["faults"]:
            self.assertEqual(replay.sha256_json(row), ROW_SHA256[row["id"]])

    def test_hf03_discarded_first_commit_precedes_surfaced_retry(self) -> None:
        rows = replay.expected_hf03_schedule()
        locus = [row for row in rows if row["round_index"] == 1 and row["request_index"] == 0]
        self.assertEqual([row["duplicate_discarded_output"] for row in locus], [True, False])
        self.assertEqual(len(rows), 17)

    def test_each_matched_clean_unlocks_and_expected_primary_is_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for fault_id in replay.FAULT_IDS:
                clean = case_fixture(root, fault_id, "clean")
                mutant = case_fixture(root, fault_id, "mutant")
                mutate_fixture(mutant, fault_id)
                mutant["injection_witness"] = {
                    "fault_id": fault_id,
                    "fault_definition_sha256": ROW_SHA256[fault_id],
                    "mutation_observed": True,
                    "exactly_one_named_injection": True,
                }
                mutant["earlier_predicates"] = [
                    {"predicate_id": predicate, "passed": True}
                    for predicate in replay.PREDICATE_PREFIX
                ]
                gate = replay.validate_clean_case(fault_id=fault_id, clean_case=clean, artifact_root=root)
                self.assertEqual(gate["status"], "clean_gate_passed")
                result = replay.replay_pair(
                    fault_id=fault_id,
                    clean_case=clean,
                    mutant_case=mutant,
                    artifact_root=root,
                    expected_fault_definition_sha256=ROW_SHA256[fault_id],
                )
                self.assertEqual(result["classification"], "caught_by_expected_primary_gate")
                self.assertEqual(result["first_failed_predicate"], replay.EXPECTED_PRIMARY_GATES[fault_id])

    def test_missing_sidecar_and_earlier_failure_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fault_id = replay.FAULT_IDS[1]
            clean = case_fixture(root, fault_id, "clean")
            Path(root, clean["logit_sidecars"][0]["path"]).unlink()
            with self.assertRaises(replay.R33ReplayError):
                replay.validate_clean_case(fault_id=fault_id, clean_case=clean, artifact_root=root)
            clean = case_fixture(root, fault_id, "clean")
            mutant = case_fixture(root, fault_id, "mutant")
            mutate_fixture(mutant, fault_id)
            mutant["injection_witness"] = {
                "fault_id": fault_id,
                "fault_definition_sha256": ROW_SHA256[fault_id],
                "mutation_observed": True,
                "exactly_one_named_injection": True,
            }
            mutant["earlier_predicates"] = [
                {"predicate_id": predicate, "passed": predicate != replay.PREDICATE_PREFIX[-1]}
                for predicate in replay.PREDICATE_PREFIX
            ]
            with self.assertRaises(replay.R33ReplayError):
                replay.replay_pair(
                    fault_id=fault_id,
                    clean_case=clean,
                    mutant_case=mutant,
                    artifact_root=root,
                    expected_fault_definition_sha256=ROW_SHA256[fault_id],
                )

    def test_five_pair_aggregate_emits_nothing_when_any_rank_is_missing(self) -> None:
        protocol = ROOT / "paper_autonomous_multifork_iteration" / "evidence" / "r33_fresh_faults" / "executor_attempt_b" / "formal-protocol.json"
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "summary.json"
            args = type(
                "Args",
                (),
                {
                    "protocol": protocol,
                    "expected_protocol_sha256": file_sha(protocol),
                    "rank_run_root": Path(temporary) / "missing-ranks",
                    "output": output,
                },
            )()
            with self.assertRaises(RuntimeError):
                aggregate.aggregate(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
