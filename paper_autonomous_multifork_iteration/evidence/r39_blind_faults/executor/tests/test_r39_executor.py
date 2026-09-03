from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


EXECUTOR = Path(__file__).resolve().parents[1]
REPO = EXECUTOR.parents[3]
FREEZE = REPO / "paper_autonomous_multifork_iteration/evidence/r39_blind_faults/designer_freeze"
sys.path.insert(0, str(EXECUTOR))

import r39_contract as contract
import r39_replay as replay


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def injection(fault_id: str, lane: str, details: dict[str, object]) -> dict[str, object]:
    value: dict[str, object] = {
        "schema_version": "forkaudit-r39-byte-bound-injection-receipt-v1",
        "fault_id": fault_id,
        "lane": lane,
        "fault_row_sha256": "a" * 64,
        "event": "H7" if fault_id in {"R39-BF07", "R39-BF08"} else "H2",
        "selector_resolution_sha256": contract.sha256_json({"frozen": True}),
        "selector_resolution": {"frozen": True},
        "payload_applied": lane == "mutant",
        "eligible_noop": lane != "mutant",
        "exactly_one_named_locus": True,
        "semantic_outputs_used_for_target_selection": False,
        "details": details,
    }
    value["receipt_sha256"] = contract.sha256_json(value)
    return value


def make_case(
    root: Path, fault_id: str, lane: str, feasibility_sha: str,
    *, logit_byte: bytes = b"x", persistent_h7_changed: bool = False,
    h7_alloc_delta: int = 0, trace_fail: bool = False,
) -> dict[str, object]:
    sidecars = []
    for index in range(16):
        raw = logit_byte + bytes([index])
        path = root / "sidecars" / f"{index}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        sidecars.append({
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "nbytes": len(raw),
            "shape": [1, 248320],
            "dtype": "float32-little-endian",
        })
    trace = {
        "implementation": "unmodified frozen RR2 ForkAudit validators",
        "predicate_rows": [
            {"predicate": f"P{index}", "passed": not (trace_fail and index == 2)}
            for index in range(4)
        ],
        "verdict": "fail" if trace_fail else "pass",
        "first_failed_predicate": "P2" if trace_fail else None,
        "first_failed_gate_id": "PERSISTENT_GDN_IMMUTABLE" if trace_fail else None,
        "compiled_binary_identity_coverage": "partial",
        "autotuning_choice_coverage": "partial",
        "new_r39_predicates_added": False,
    }
    trace_path = root / "forkaudit-trace.json"
    write_json(trace_path, trace)
    base = {"document_a": {"digest": "base"}}
    h7 = {"document_a": {"digest": "changed"}} if persistent_h7_changed else copy.deepcopy(base)
    alloc = {
        phase: {
            "allocated_bytes": 100 + (h7_alloc_delta if phase == "H7" else 0),
            "reserved_bytes": 200,
            "peak_allocated_bytes": 120,
            "peak_reserved_bytes": 220,
        }
        for phase in ("H0", "H1", "H4", "H6", "H7")
    }
    if fault_id == "R39-BF01":
        details = {
            "destination_matches_selected_source": lane == "mutant",
            "destination_matches_true_source": lane != "mutant",
            "destination_is_private": True,
        }
    elif fault_id == "R39-BF07":
        details = {
            "digest_changed": lane == "mutant",
            "private_scrub_completed_first": True,
        }
    elif fault_id == "R39-BF08":
        details = {
            "retained_backing_storage_count": 1 if lane == "mutant" else 0,
            "selected_bytes_zero": True,
            "reused_by_live_request": False,
        }
    else:
        raise AssertionError(fault_id)
    return {
        "schema_version": "forkaudit-r39-blind-fault-lane-v1",
        "run_id": contract.RUN_ID,
        "status": "expected_horizon_completed",
        "fault_id": fault_id,
        "fault_row_sha256": "a" * 64,
        "lane": lane,
        "gpu_index": contract.FAULT_TO_GPU[fault_id],
        "input_rank": contract.FAULT_TO_GPU[fault_id],
        "expected_gpu_uuid": "GPU-test",
        "expected_horizon": contract.EXPECTED_HORIZON[fault_id],
        "reached_horizon": contract.EXPECTED_HORIZON[fault_id],
        "plan_raw_sha256": contract.PLAN_RAW_SHA256,
        "protocol_raw_sha256": contract.PROTOCOL_RAW_SHA256,
        "feasibility": {"sha256": feasibility_sha},
        "source_manifest_sha256": "b" * 64,
        "hardware": {},
        "discarded_warmup": {},
        "all_production_assertions_enabled": True,
        "selective_gate_suppression": False,
        "ordered_schedule": [
            {"call_index": r * 2 + q, "round_index": r, "request_index": q}
            for r in range(8) for q in range(2)
        ],
        "logit_sidecars": sidecars,
        "semantic_results": {
            "generated_token_ids": [[1] * 8, [2] * 8],
            "logical_kv": {}, "gdn": {},
        },
        "persistent_base_snapshots": {
            "H1": base, "H4": copy.deepcopy(base), "H6": copy.deepcopy(base),
            "H7_pre_release": h7 if contract.EXPECTED_HORIZON[fault_id] == "H7" else None,
        },
        "allocator_endpoints": alloc,
        "transition_receipts": [],
        "byte_bound_injection_receipt": injection(fault_id, lane, details),
        "scrub_receipt": None,
        "forkaudit": {
            "trace_path": trace_path.relative_to(root).as_posix(),
            "trace_sha256": contract.sha256_file(trace_path),
            "verdict": trace["verdict"],
            "first_failed_predicate": trace["first_failed_predicate"],
            "first_failed_gate_id": trace["first_failed_gate_id"],
            "compiled_binary_identity_coverage": "partial",
            "autotuning_choice_coverage": "partial",
        },
        "cleanup": {}, "bf10_component_identity": None,
        "operational_invalid": None,
    }


class ContractTests(unittest.TestCase):
    def test_frozen_integrity_and_exact_fault_order(self) -> None:
        value = contract.verify_freeze(FREEZE / "PROTOCOL.md", FREEZE / "plan.json")
        self.assertEqual(tuple(value["fault_row_sha256"]), contract.FAULT_IDS)

    def test_assignment_is_exact_and_deterministic(self) -> None:
        rows = contract.assignment_rows()
        self.assertEqual(len(rows), 11)
        self.assertEqual({row["fault_id"] for row in rows}, set(contract.FAULT_IDS))
        self.assertEqual(contract.GPU_ASSIGNMENT[0], ("R39-BF01", "R39-BF09"))
        self.assertEqual(contract.GPU_ASSIGNMENT[2], ("R39-BF03", "R39-BF11"))

    def test_static_ineligible_faults_are_exact_not_substitutes(self) -> None:
        self.assertEqual(set(contract.STATIC_INELIGIBLE), {"R39-BF02", "R39-BF09"})
        for value in contract.STATIC_INELIGIBLE.values():
            self.assertIn("reason", value)

    def test_feasibility_tamper_fails_closed(self) -> None:
        freeze = contract.verify_freeze(FREEZE / "PROTOCOL.md", FREEZE / "plan.json")
        value = contract.make_feasibility(
            fault_id="R39-BF02", freeze=freeze,
            selector_resolution={"source_aware_static_preflight": True},
            eligible=False, ineligible_reason=contract.STATIC_INELIGIBLE["R39-BF02"],
            source_manifest_sha256="c" * 64,
        )
        contract.validate_feasibility(value, fault_id="R39-BF02", freeze=freeze)
        value["eligible"] = True
        with self.assertRaises(contract.ContractError):
            contract.validate_feasibility(value, fault_id="R39-BF02", freeze=freeze)


class ReplayTests(unittest.TestCase):
    def _triple(self, fault_id: str, mutant_kwargs: dict[str, object]) -> tuple[Path, dict[str, object], dict[str, object], dict[str, object]]:
        temporary = Path(tempfile.mkdtemp(prefix="r39-replay-test-"))
        sha = "d" * 64
        reference = make_case(temporary / "reference", fault_id, "reference", sha)
        clean = make_case(temporary / "clean", fault_id, "clean", sha)
        mutant = make_case(temporary / "mutant", fault_id, "mutant", sha, **mutant_kwargs)
        return temporary, reference, clean, mutant

    def test_bf01_escape_and_logit_difference_are_retained(self) -> None:
        root, reference, clean, mutant = self._triple("R39-BF01", {"logit_byte": b"y"})
        value = replay.replay_pair(
            fault_id="R39-BF01", reference=reference, clean=clean, mutant=mutant,
            reference_root=root / "reference", clean_root=root / "clean",
            mutant_root=root / "mutant", feasibility_sha256="d" * 64,
        )
        self.assertTrue(value["valid_pair"])
        self.assertFalse(value["observers"]["output_equality"]["complete_fp32_logits_byte_exact"])
        self.assertFalse(value["observers"]["forkaudit"]["detected"])
        self.assertTrue(value["negative_or_escape_retained"])

    def test_bf07_persistent_and_forkaudit_detection(self) -> None:
        root, reference, clean, mutant = self._triple(
            "R39-BF07", {"persistent_h7_changed": True, "trace_fail": True}
        )
        value = replay.replay_pair(
            fault_id="R39-BF07", reference=reference, clean=clean, mutant=mutant,
            reference_root=root / "reference", clean_root=root / "clean",
            mutant_root=root / "mutant", feasibility_sha256="d" * 64,
        )
        self.assertTrue(value["fault_reached"])
        self.assertTrue(value["observers"]["persistent_base_invariant"]["detected"])
        self.assertTrue(value["observers"]["forkaudit"]["detected"])

    def test_bf08_allocation_detection(self) -> None:
        root, reference, clean, mutant = self._triple(
            "R39-BF08", {"h7_alloc_delta": 4096}
        )
        value = replay.replay_pair(
            fault_id="R39-BF08", reference=reference, clean=clean, mutant=mutant,
            reference_root=root / "reference", clean_root=root / "clean",
            mutant_root=root / "mutant", feasibility_sha256="d" * 64,
        )
        self.assertTrue(value["fault_reached"])
        self.assertTrue(value["observers"]["allocation_assertions"]["detected"])

    def test_detached_replay_import_boundary(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-I", str(EXECUTOR / "r39_replay.py"), "--help"],
            capture_output=True, text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)


class SourceTests(unittest.TestCase):
    def test_launcher_contains_exact_eleven_mapping(self) -> None:
        source = (EXECUTOR / "r39_launch_8gpu.sh").read_text(encoding="utf-8")
        for fault_id in contract.FAULT_IDS:
            self.assertIn(fault_id, source)
        self.assertIn("exactly eight visible physical GPUs", source)

    def test_lane_has_no_old_fault_ids_or_gate_suppression(self) -> None:
        source = (EXECUTOR / "r39_lane.py").read_text(encoding="utf-8")
        self.assertNotIn("HF01_", source)
        self.assertNotIn("PRODUCTION_ASSERTION_ALLOWLIST", source)
        self.assertIn('"exception_allowlisted": False', source)
        self.assertIn('"selective_gate_suppression": False', source)


if __name__ == "__main__":
    unittest.main()
