from __future__ import annotations

"""Synthetic contract tests for the R35 replay and eight-rank aggregator."""

from array import array
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from typing import Any, Mapping


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import r35_aggregate_historical_alias_regression as aggregate
import r35_replay_historical_alias_regression as replay


RUN_ID = "R35-HISTORICAL-ALIAS-REGRESSION-20260826A"
PREREG_SHA = hashlib.sha256(b"synthetic-r35-preregistration").hexdigest()
AMENDMENT_SHA = hashlib.sha256(b"synthetic-r35-resource-amendment").hexdigest()
EXECUTION_INPUT_SHA = hashlib.sha256(b"synthetic-r35-execution-input").hexdigest()
SOURCE_LEDGER_SHA = hashlib.sha256(b"synthetic-r35-source-ledger").hexdigest()
PACKAGE_SHA = hashlib.sha256(b"synthetic-r35-package").hexdigest()

REQUIRED_SCIENCE_MODULES = (
    "build_qcomem_forkaudit_rr2_input_manifest",
    "qcomem_forkaudit_storage_witness",
    "qcomem_joint_policy",
    "qcomem_qwen35_functional_stack",
    "qcomem_single_token_gdn_ownership",
    "qcomem_vllm_paged_fair_control",
    "qcomem_vllm_paged_kernel",
    "qcomem_vllm_paged_multifork_resident",
    "run_qcomem_qwen35_forkaudit_review_revision",
    "run_qcomem_qwen35_vllm_paged_multifork_resident",
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def protocol_fixture() -> dict[str, Any]:
    return {
        "schema_version": replay.PROTOCOL_SCHEMA,
        "run_id": RUN_ID,
        "rank_count": 8,
        "lanes": list(replay.LANES),
        "lane_order_by_rank": {
            "even": list(replay.LANES),
            "odd": list(reversed(replay.LANES)),
        },
        "lane_audit_mode": {
            "historical_pre_fix": "unified_storage_and_binding",
            "repaired_borrowed": "unified_storage_and_binding",
            "materialized_control": "policy_aware_storage_only",
        },
        "sidecar": {
            "dtype": "float32-little-endian",
            "shape": [1, 4],
            "nbytes": 16,
        },
        "receipt_order": list(replay.RECEIPT_ORDER),
        "historical_first_failure": {
            "model_step_index": 0,
            "receipt_id": replay.RECEIPT_ORDER[2],
            "predicate_id": replay.GATE_COMPLETED_REBOUND,
        },
        "expected": {"resident_count": 2, "linear_state_count": 60},
        "content_digest_formula": "sha256_json(ordered_content_digests)",
        "coordinate_classes": {
            "archived_coordinate_ranks": [0, 1, 2],
            "additional_frozen_input_ranks": [3, 4, 5, 6, 7],
            "statistical_independence_claimed": False,
        },
        "comparison_matrix": {
            "pair_mappings": [
                "historical_pre_fix_vs_materialized_control",
                "historical_pre_fix_vs_repaired_borrowed",
                "repaired_borrowed_vs_materialized_control",
            ],
            "output_only": ["greedy_token_exact", "full_fp32_logits_exact"],
            "state_differential": [
                "request0_terminal_gdn_content_exact",
                "logical_kv_content_exact",
            ],
            "state_invariant": ["persistent_base_content_only_invariant"],
            "forkaudit": [
                "lane_local_storage_intervals",
                "owner_relations",
                "setup_to_transition_binding",
            ],
            "normalized_storage_ids_comparable_across_lanes": False,
        },
        "source_bindings": {
            "upstream_r29_execution_input_raw_sha256": replay.UPSTREAM_R29_RAW_SHA256,
            "source_ledger_raw_sha256": SOURCE_LEDGER_SHA,
            "runner_sha256": digest("runner"),
            "repair_sha256": digest("repair"),
            "storage_witness_sha256": digest("storage-witness"),
        },
        "resource_amendment_binding": "external_preexecution",
    }


def amendment_fixture(*, duplicate_uuid: bool = False) -> dict[str, Any]:
    assignments = {
        str(rank): {
            "physical_index": rank,
            "uuid": f"GPU-{digest(f'gpu:{rank}')[:32]}",
        }
        for rank in range(8)
    }
    if duplicate_uuid:
        assignments["1"]["uuid"] = assignments["0"]["uuid"]
    return {
        "schema_version": replay.AMENDMENT_SCHEMA,
        "status": "frozen_after_resource_creation_before_candidate_outputs",
        "created_at_utc": "2026-08-26T00:00:00Z",
        "run_id": RUN_ID,
        "preregistration_raw_sha256": PREREG_SHA,
        "execution_input_raw_sha256": EXECUTION_INPUT_SHA,
        "source_ledger_raw_sha256": SOURCE_LEDGER_SHA,
        "execution_package_sha256": PACKAGE_SHA,
        "science_design_changed": False,
        "candidate_output_seen_when_frozen": False,
        "job_id": "synthetic-job",
        "trial_id": "synthetic-trial",
        "pod": "synthetic-pod",
        "gpu_assignments": assignments,
    }


def imported_rr2_code_fixture() -> dict[str, Any]:
    file_sha256 = {
        f"{module}.py": digest(f"rr2-source:{module}")
        for module in REQUIRED_SCIENCE_MODULES
    }
    return {
        "raw_sha256": digest("rr2-code-ledger"),
        "file_count": len(file_sha256),
        "rows_sha256": digest("rr2-code-ledger-rows"),
        "file_sha256": file_sha256,
    }


def module_closure_fixture() -> dict[str, Any]:
    rows = [
        {
            "module": module,
            "source_class": (
                "r35_package_override"
                if module
                in {
                    "qcomem_forkaudit_storage_witness",
                    "qcomem_single_token_gdn_ownership",
                    "qcomem_vllm_paged_multifork_resident",
                    "run_qcomem_qwen35_vllm_paged_multifork_resident",
                }
                else "imported_rr2_ledger"
            ),
            "path": f"{module}.py",
            "sha256": digest(f"loaded-module:{module}"),
        }
        for module in REQUIRED_SCIENCE_MODULES
    ]
    return {
        "module_count": len(rows),
        "modules": rows,
        "modules_sha256": replay.sha256_json(rows),
        "shadowed_module_count": 0,
    }


def _state_coordinate(layer: int, family: str) -> tuple[int, str]:
    family_index = 0 if family == "conv_states" else 1
    return layer * 2 + family_index, f"layer:{layer}/{family}/state:0"


def _binding_and_content(
    *,
    lane: str,
    phase: str,
    owner: str,
    coordinate_index: int,
    family: str,
    historical_reproduces: bool,
) -> tuple[int, str]:
    base = digest(f"base-content:{coordinate_index}")
    updated = digest(f"updated-content:{coordinate_index}")
    if phase == "setup":
        if lane == "materialized_control":
            offset = {"persistent": 0, "request_0": 1000, "request_1": 2000}[owner]
            return offset + coordinate_index, base
        return coordinate_index, base
    if lane == "historical_pre_fix":
        if not historical_reproduces:
            if owner == "request_0":
                return 1000 + coordinate_index, updated
            return coordinate_index, base
        if family == "conv_states":
            return coordinate_index, updated
        if owner == "request_0":
            return 1000 + coordinate_index, updated
        return coordinate_index, base
    if lane == "repaired_borrowed":
        if owner == "request_0":
            return 1000 + coordinate_index, updated
        return coordinate_index, base
    if owner == "persistent":
        return coordinate_index, base
    if owner == "request_1":
        return 2000 + coordinate_index, base
    if family == "conv_states":
        return 1000 + coordinate_index, updated
    return 3000 + coordinate_index, updated


def snapshot_fixture(
    lane: str, phase: str, *, historical_reproduces: bool
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for owner in replay.OWNERS:
        for layer in range(30):
            for family in replay.FAMILIES:
                coordinate_index, coordinate = _state_coordinate(layer, family)
                identity, content = _binding_and_content(
                    lane=lane,
                    phase=phase,
                    owner=owner,
                    coordinate_index=coordinate_index,
                    family=family,
                    historical_reproduces=historical_reproduces,
                )
                rows.append(
                    {
                        "owner": owner,
                        "layer_index": layer,
                        "state_family": family,
                        "state_index": 0,
                        "coordinate": coordinate,
                        "tensor_id": f"tensor-{identity:04d}",
                        "storage_id": f"storage-{identity:04d}",
                        "storage_nbytes": 4,
                        "byte_start": 0,
                        "byte_end_exclusive": 4,
                        "tensor_nbytes": 4,
                        "shape": [1],
                        "stride": [1],
                        "dtype": "torch.float32",
                        "device": "cuda:0",
                        "dense_contiguous": True,
                        "content_sha256": content,
                        "contains_absolute_pointer": False,
                        "contains_python_object_id": False,
                    }
                )
    return {
        "row_count": 180,
        "rows": rows,
        "rows_sha256": replay.sha256_json(rows),
        "absolute_pointers_persisted": False,
        "python_object_ids_persisted": False,
    }


def owner_content(snapshot: Mapping[str, Any], owner: str) -> dict[str, Any]:
    ordered = [row["content_sha256"] for row in snapshot["rows"] if row["owner"] == owner]
    return {
        "sha256": replay.sha256_json(ordered),
        "tensor_count": 60,
        "ordered_content_digests": ordered,
    }


def terminal_content(
    setup: Mapping[str, Any], post: Mapping[str, Any], *, rank: int
) -> dict[str, Any]:
    logical_kv = [
        {
            "request_index": request,
            "layer_sha256": {
                str(layer): digest(f"logical-kv:{rank}:{request}:{layer}")
                for layer in range(10)
            },
        }
        for request in range(2)
    ]
    return {
        "request_gdn": [
            {"request_index": request, **owner_content(post, f"request_{request}")}
            for request in range(2)
        ],
        "logical_kv": logical_kv,
        "logical_kv_sha256": replay.sha256_json(logical_kv),
        "persistent_gdn": {
            "setup": owner_content(setup, "persistent"),
            "post": owner_content(post, "persistent"),
        },
        "storage_or_pointer_fields_persisted": False,
    }


def receipt_rows(lane: str, count: int) -> list[dict[str, Any]]:
    rows = []
    for ordinal, receipt_id in enumerate(replay.RECEIPT_ORDER[:count]):
        payload = {"lane": lane, "ordinal": ordinal, "synthetic": True}
        rows.append(
            {
                "receipt_id": receipt_id,
                "status": "passed",
                "payload": payload,
                "payload_sha256": replay.sha256_json(payload),
            }
        )
    return rows


def rejection_fixture(predicate: str) -> dict[str, Any]:
    return {
        "authenticated": True,
        "receipt_id": replay.RECEIPT_ORDER[2],
        "predicate_id": predicate,
        "exception": {
            "module": "qcomem_forkaudit_storage_witness",
            "type": "GDNStorageWitnessError",
            "message": "synthetic authenticated rejection",
            "gate_id": predicate,
            "stack": [
                {
                    "filename": "qcomem_forkaudit_storage_witness.py",
                    "line": 1,
                    "function": "verify_request_gdn_binding_guard",
                }
            ],
        },
    }


def audit_fixture(
    lane: str,
    *,
    historical_reproduces: bool,
    wrong_gate: bool,
) -> dict[str, Any]:
    mode = (
        "policy_aware_storage_only"
        if lane == "materialized_control"
        else "unified_storage_and_binding"
    )
    if lane == "historical_pre_fix" and historical_reproduces:
        predicate = (
            replay.GATE_PERSISTENT_IMMUTABLE
            if wrong_gate
            else replay.GATE_COMPLETED_REBOUND
        )
        rejection = rejection_fixture(predicate)
        completed = receipt_rows(lane, 2)
    else:
        rejection = None
        completed = receipt_rows(lane, 6)
    storage_passed = len(completed) > 2
    return {
        "audit_mode": mode,
        "completed_receipts": completed,
        "first_authenticated_rejection": rejection,
        "unified_witness_passed": storage_passed if mode == "unified_storage_and_binding" else None,
        "storage_witness_passed": storage_passed,
        "expected_historical_rejection_observed": bool(
            lane == "historical_pre_fix"
            and rejection is not None
            and rejection["predicate_id"] == replay.GATE_COMPLETED_REBOUND
        ),
    }


def repair_receipt_fixture(post: Mapping[str, Any]) -> dict[str, Any]:
    request_rows = [
        row
        for row in post["rows"]
        if row["owner"] == "request_0" and row["state_family"] == "conv_states"
    ]
    rows = [
        {
            "layer_index": row["layer_index"],
            "state_index": 0,
            "action": "cloned_borrowed_state",
            "content_sha256": row["content_sha256"],
            "base_disjoint": True,
            "all_peers_disjoint": True,
        }
        for row in request_rows
    ]
    return {
        "schema_version": "qcomem-single-token-gdn-conv-privatization-v1",
        "request_index": 0,
        "resident_count": 2,
        "state_index": 0,
        "layer_indices": list(range(30)),
        "conv_tensor_count": 30,
        "cloned_tensor_count": 30,
        "already_private_tensor_count": 0,
        "ownership_only_change": True,
        "fault_id_specialization": False,
        "rows": rows,
        "rows_sha256": replay.sha256_json(rows),
    }


def write_sidecar(root: Path, lane: str, values: tuple[float, ...]) -> dict[str, Any]:
    payload = struct.pack("<4f", *values)
    path = root / f"{lane}-full-fp32-logits.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "dtype": "float32-little-endian",
        "shape": [1, 4],
        "nbytes": len(payload),
        "finite": True,
    }


def lane_fixture(
    root: Path,
    lane: str,
    *,
    rank: int,
    process_id: str,
    nonce: str,
    historical_reproduces: bool,
    repair_logits_exact: bool,
    wrong_gate: bool,
) -> dict[str, Any]:
    setup = snapshot_fixture(lane, "setup", historical_reproduces=historical_reproduces)
    post = snapshot_fixture(lane, "post", historical_reproduces=historical_reproduces)
    values = (0.1, 0.9, 0.2, -0.1)
    if lane == "repaired_borrowed" and not repair_logits_exact:
        values = (0.1, 0.2, 0.95, -0.1)
    sidecar = write_sidecar(root, lane, values)
    decoded = array("f")
    decoded.frombytes((root / sidecar["path"]).read_bytes())
    token = max(range(len(decoded)), key=decoded.__getitem__)
    audit = audit_fixture(
        lane,
        historical_reproduces=historical_reproduces,
        wrong_gate=wrong_gate,
    )
    baseline = {"current_allocated_bytes": 0, "current_reserved_bytes": 0}
    return {
        "lane": lane,
        "rank": rank,
        "status": (
            "authenticated_forkaudit_rejection_after_model_step"
            if audit["first_authenticated_rejection"] is not None
            else "completed_clean"
        ),
        "fresh_case": True,
        "case_nonce": nonce,
        "process_instance_id": process_id,
        "state_reused_from_prior_lane": False,
        "allocator_before": baseline,
        "allocator_baseline": baseline,
        "mutation_receipt": {
            "r29_heldout_fault_module_loaded": False,
            "generic_mutant_definition_module_passively_loaded": True,
            "mutation_requested": False,
            "mutation_applied": False,
            "mutation_event_count": 0,
        },
        "repair_transition_receipt": (
            repair_receipt_fixture(post) if lane == "repaired_borrowed" else None
        ),
        "model_step": {
            "step_index": 0,
            "semantic_horizon_reached": True,
            "full_logit_sha256": sidecar["sha256"],
            "greedy_token_id": token,
        },
        "full_logits": sidecar,
        "setup_snapshot": setup,
        "post_snapshot": post,
        "terminal_content": terminal_content(setup, post, rank=rank),
        "audit": audit,
        "operational_invalid": None,
        "cleanup": {
            "fresh_case_disposed": True,
            "registered_backend_restored": True,
            "strong_references_released": True,
            "gc_collect_completed": True,
            "cuda_empty_cache_completed": True,
            "cuda_synchronize_completed": True,
            "allocator_after": baseline,
            "allocator_baseline_exact": True,
            "cleanup_passed": True,
            "cleanup_error": None,
        },
    }


def comparison_fixture(
    candidate: Mapping[str, Any], control: Mapping[str, Any], root: Path
) -> dict[str, bool]:
    candidate_payload = (root / candidate["full_logits"]["path"]).read_bytes()
    control_payload = (root / control["full_logits"]["path"]).read_bytes()
    return {
        "greedy_token_exact": candidate["model_step"]["greedy_token_id"]
        == control["model_step"]["greedy_token_id"],
        "full_fp32_logits_exact": candidate_payload == control_payload,
        "request0_terminal_gdn_content_exact": candidate["terminal_content"]["request_gdn"][0]["sha256"]
        == control["terminal_content"]["request_gdn"][0]["sha256"],
        "logical_kv_content_exact": candidate["terminal_content"]["logical_kv"]
        == control["terminal_content"]["logical_kv"],
        "persistent_base_content_only_invariant": candidate["terminal_content"]["persistent_gdn"]["setup"]["sha256"]
        == candidate["terminal_content"]["persistent_gdn"]["post"]["sha256"],
    }


def rank_fixture(
    root: Path,
    *,
    rank: int,
    protocol: Mapping[str, Any],
    amendment: Mapping[str, Any],
    historical_reproduces: bool = True,
    repair_logits_exact: bool = True,
    wrong_gate: bool = False,
) -> dict[str, Any]:
    process_id = digest(f"process:{rank}")
    lanes = {
        lane: lane_fixture(
            root,
            lane,
            rank=rank,
            process_id=process_id,
            nonce=digest(f"case:{rank}:{lane}")[:32],
            historical_reproduces=historical_reproduces,
            repair_logits_exact=repair_logits_exact,
            wrong_gate=wrong_gate,
        )
        for lane in replay.LANES
    }
    historical = lanes["historical_pre_fix"]
    repaired = lanes["repaired_borrowed"]
    control = lanes["materialized_control"]
    comparisons = {
        "historical_pre_fix_vs_materialized_control": comparison_fixture(
            historical, control, root
        ),
        "historical_pre_fix_vs_repaired_borrowed": comparison_fixture(
            historical, repaired, root
        ),
        "repaired_borrowed_vs_materialized_control": comparison_fixture(
            repaired, control, root
        ),
    }
    assignment = amendment["gpu_assignments"][str(rank)]
    return {
        "schema_version": replay.RANK_SCHEMA,
        "run_id": RUN_ID,
        "status": "rank_completed",
        "rank": rank,
        "operational_invalid": None,
        "process_instance_id": process_id,
        "protocol": dict(protocol),
        "execution_input_raw_sha256": EXECUTION_INPUT_SHA,
        "preregistration_raw_sha256": PREREG_SHA,
        "amendment_raw_sha256": AMENDMENT_SHA,
        "resource": {
            "job_id": amendment["job_id"],
            "trial_id": amendment["trial_id"],
            "pod": amendment["pod"],
            "gpu_assignment": assignment,
            "preregistration_raw_sha256": PREREG_SHA,
            "execution_package_sha256": PACKAGE_SHA,
        },
        "source_bindings": dict(protocol["source_bindings"]),
        "upstream_r29_execution_input": {
            "raw_sha256": replay.UPSTREAM_R29_RAW_SHA256,
            "run_id": replay.UPSTREAM_R29_RUN_ID,
            "copied_fields_exact": True,
        },
        "fault_isolation": {
            "r29_heldout_fault_suite_import_blocked": True,
            "r29_heldout_fault_suite_in_sys_modules": False,
            "generic_mutant_definition_module_passively_loaded": True,
            "mutation_requested": False,
            "mutation_applied": False,
        },
        "input_receipt": {
            "rank": rank,
            "coordinate_class": "archived" if rank < 3 else "additional_frozen",
            "imported_rr2_code": imported_rr2_code_fixture(),
            "loaded_science_module_closure": module_closure_fixture(),
        },
        "lane_order": list(
            protocol["lane_order_by_rank"]["even" if rank % 2 == 0 else "odd"]
        ),
        "lanes": lanes,
        "comparisons": comparisons,
    }


def replay_fixture(
    root: Path,
    *,
    rank: int,
    historical_reproduces: bool = True,
    repair_logits_exact: bool = True,
    wrong_gate: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol = protocol_fixture()
    amendment = amendment_fixture()
    result = rank_fixture(
        root,
        rank=rank,
        protocol=protocol,
        amendment=amendment,
        historical_reproduces=historical_reproduces,
        repair_logits_exact=repair_logits_exact,
        wrong_gate=wrong_gate,
    )
    receipt = replay.replay_rank(
        result=result,
        raw_root=root,
        protocol=protocol,
        protocol_raw_sha256=PREREG_SHA,
        amendment_raw_sha256=AMENDMENT_SHA,
        amendment=amendment,
        result_raw_sha256=digest(f"rank-result:{rank}"),
    )
    return protocol, amendment, result, receipt


class R35HistoricalAliasContractTest(unittest.TestCase):
    def test_valid_protocol_and_external_resource_amendment(self) -> None:
        protocol = protocol_fixture()
        amendment = amendment_fixture()
        self.assertEqual(replay.validate_protocol(protocol), protocol)
        self.assertEqual(
            replay.validate_amendment(
                amendment,
                protocol_raw_sha256=PREREG_SHA,
                amendment_raw_sha256=AMENDMENT_SHA,
                expected_run_id=RUN_ID,
            ),
            amendment,
        )

    def test_full_three_lane_snapshot_and_fp32_happy_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, receipt = replay_fixture(Path(temporary), rank=0)
        self.assertTrue(receipt["operational_valid"])
        self.assertEqual(receipt["verified_lane_count"], 3)
        self.assertEqual(receipt["verified_sidecar_count"], 3)
        self.assertTrue(
            receipt["hypothesis_outcomes"][
                "historical_expected_authenticated_first_gate_reproduced"
            ]
        )
        self.assertTrue(
            receipt["hypothesis_outcomes"][
                "repaired_semantic_and_terminal_exact_to_materialized"
            ]
        )
        self.assertFalse(
            receipt["conventional_baseline_matrix"][
                "historical_pre_fix_vs_materialized_control"
            ]["persistent_base_content_only_invariant"]
        )

    def test_nonreproduction_and_repair_mismatch_are_valid_negative(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            _, _, _, receipt = replay_fixture(
                Path(temporary),
                rank=3,
                historical_reproduces=False,
                repair_logits_exact=False,
            )
        self.assertTrue(receipt["operational_valid"])
        self.assertFalse(
            receipt["hypothesis_outcomes"][
                "historical_expected_authenticated_first_gate_reproduced"
            ]
        )
        self.assertFalse(
            receipt["hypothesis_outcomes"][
                "repaired_semantic_and_terminal_exact_to_materialized"
            ]
        )

    def test_tampered_sidecar_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = protocol_fixture()
            amendment = amendment_fixture()
            result = rank_fixture(
                root, rank=0, protocol=protocol, amendment=amendment
            )
            path = root / "historical_pre_fix-full-fp32-logits.bin"
            payload = bytearray(path.read_bytes())
            payload[-1] ^= 1
            path.write_bytes(payload)
            with self.assertRaisesRegex(replay.ReplayError, "sidecar hash"):
                replay.replay_rank(
                    result=result,
                    raw_root=root,
                    protocol=protocol,
                    protocol_raw_sha256=PREREG_SHA,
                    amendment_raw_sha256=AMENDMENT_SHA,
                    amendment=amendment,
                    result_raw_sha256=digest("tampered-result"),
                )

    def test_duplicate_rank_local_nonce_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = protocol_fixture()
            amendment = amendment_fixture()
            result = rank_fixture(
                root, rank=0, protocol=protocol, amendment=amendment
            )
            result["lanes"]["repaired_borrowed"]["case_nonce"] = result["lanes"][
                "historical_pre_fix"
            ]["case_nonce"]
            with self.assertRaisesRegex(replay.ReplayError, "fresh case nonce"):
                replay.replay_rank(
                    result=result,
                    raw_root=root,
                    protocol=protocol,
                    protocol_raw_sha256=PREREG_SHA,
                    amendment_raw_sha256=AMENDMENT_SHA,
                    amendment=amendment,
                    result_raw_sha256=digest("duplicate-nonce-result"),
                )

    def test_wrong_authenticated_gate_fails_row_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protocol = protocol_fixture()
            amendment = amendment_fixture()
            result = rank_fixture(
                root,
                rank=0,
                protocol=protocol,
                amendment=amendment,
                wrong_gate=True,
            )
            with self.assertRaisesRegex(replay.ReplayError, "gate not reproduced"):
                replay.replay_rank(
                    result=result,
                    raw_root=root,
                    protocol=protocol,
                    protocol_raw_sha256=PREREG_SHA,
                    amendment_raw_sha256=AMENDMENT_SHA,
                    amendment=amendment,
                    result_raw_sha256=digest("wrong-gate-result"),
                )

    def test_duplicate_gpu_uuid_fails_amendment(self) -> None:
        with self.assertRaisesRegex(replay.ReplayError, "GPU UUID uniqueness"):
            replay.validate_amendment(
                amendment_fixture(duplicate_uuid=True),
                protocol_raw_sha256=PREREG_SHA,
                amendment_raw_sha256=AMENDMENT_SHA,
                expected_run_id=RUN_ID,
            )

    def test_eight_rank_aggregate_happy_path_and_global_nonce_rejection(self) -> None:
        protocol = protocol_fixture()
        amendment = amendment_fixture()
        receipts: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for rank in range(8):
                rank_root = root / f"rank-{rank}"
                result = rank_fixture(
                    rank_root,
                    rank=rank,
                    protocol=protocol,
                    amendment=amendment,
                )
                receipt = replay.replay_rank(
                    result=result,
                    raw_root=rank_root,
                    protocol=protocol,
                    protocol_raw_sha256=PREREG_SHA,
                    amendment_raw_sha256=AMENDMENT_SHA,
                    amendment=amendment,
                    result_raw_sha256=digest(f"rank-result:{rank}"),
                )
                receipts.append(
                    aggregate.validate_replay(
                        receipt,
                        expected_rank=rank,
                        expected_preregistration_sha256=PREREG_SHA,
                        expected_amendment_sha256=AMENDMENT_SHA,
                        expected_execution_input_sha256=EXECUTION_INPUT_SHA,
                        expected_source_ledger_sha256=SOURCE_LEDGER_SHA,
                    )
                )
        replay_hashes = {rank: digest(f"rank-replay:{rank}") for rank in range(8)}
        summary = aggregate.aggregate(
            receipts,
            replay_raw_sha256=replay_hashes,
            preregistration_sha256=PREREG_SHA,
            amendment_sha256=AMENDMENT_SHA,
            execution_input_sha256=EXECUTION_INPUT_SHA,
            source_ledger_sha256=SOURCE_LEDGER_SHA,
        )
        self.assertTrue(summary["operational_valid"])
        self.assertEqual(summary["operational_cardinality"]["lane_count"], 24)
        self.assertEqual(
            summary["coordinate_classes"]["archived_coordinates"]["ranks"],
            [0, 1, 2],
        )
        duplicate = copy.deepcopy(receipts)
        duplicate[1]["lanes"]["historical_pre_fix"]["case_nonce"] = duplicate[0][
            "lanes"
        ]["historical_pre_fix"]["case_nonce"]
        with self.assertRaisesRegex(aggregate.AggregateError, "duplicate case nonce"):
            aggregate.aggregate(
                duplicate,
                replay_raw_sha256=replay_hashes,
                preregistration_sha256=PREREG_SHA,
                amendment_sha256=AMENDMENT_SHA,
                execution_input_sha256=EXECUTION_INPUT_SHA,
                source_ledger_sha256=SOURCE_LEDGER_SHA,
            )


if __name__ == "__main__":
    unittest.main()
