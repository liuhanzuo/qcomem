from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "executed_source"))

from r40_finalize import finalize  # noqa: E402


def seal(value):
    value = dict(value)
    value["payload_sha256"] = None
    value["payload_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return value


def functional_ledger():
    coordinates = [[layer, family, 0] for layer in range(30) for family in ("conv", "recurrent")]
    calls = []
    for call_index in range(64):
        round_index, request_index = divmod(call_index, 8)
        completed = list(range(request_index + 1)) if round_index == 0 else list(range(8))
        edges = []
        for coordinate_index, coordinate in enumerate(coordinates):
            endpoint_index = call_index * len(coordinates) + coordinate_index
            descriptor = {
                "shape": [1],
                "stride": [1],
                "storage_offset": 0,
                "dtype": "torch.bfloat16",
                "device": "cuda:0",
                "storage_nbytes": 2,
                "tensor_nbytes": 2,
                "byte_interval": [0, 2],
            }
            pre = {
                "object_id": 10_000_000 + endpoint_index,
                "storage_key": ["cuda:0", 20_000_000 + endpoint_index, 2],
                "descriptor": descriptor,
                "content_sha256": hashlib.sha256(f"pre-{endpoint_index}".encode()).hexdigest(),
            }
            post = {
                "object_id": 30_000_000 + endpoint_index,
                "storage_key": ["cuda:0", 40_000_000 + endpoint_index, 2],
                "descriptor": descriptor,
                "content_sha256": hashlib.sha256(f"post-{endpoint_index}".encode()).hexdigest(),
            }
            edges.append(
                {
                    "coordinate": coordinate,
                    "version": round_index + 1,
                    "pre": pre,
                    "post": post,
                    "new_tensor_object": True,
                    "new_storage": True,
                    "descriptor_authorized": True,
                    "content_recorded": True,
                }
            )
        calls.append(
            {
                "call_index": call_index,
                "round_index": round_index,
                "request_index": request_index,
                "request_version": round_index + 1,
                "edge_count": 60,
                "edges": edges,
                "completed_request_indices_after_call": completed,
                "private_request_rows_after_call": len(completed) * 60,
                "borrowed_request_rows_after_call": (8 - len(completed)) * 60,
                "target_all_new": True,
                "non_target_unchanged": True,
                "persistent_unchanged": True,
                "completed_private": True,
                "incomplete_exact_alias": True,
            }
        )
    ledger = {
        "schema_version": "forkaudit-r40-v32-functional-rebind-ledger-v1",
        "call_count": 64,
        "edge_count": 3840,
        "edges_per_call": 60,
        "all_new_tensor_objects": True,
        "all_new_storages": True,
        "all_descriptors_authorized": True,
        "all_contents_recorded": True,
        "calls": calls,
        "ledger_sha256": None,
    }
    ledger["ledger_sha256"] = hashlib.sha256(
        json.dumps(ledger, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ledger


class Finalizer(unittest.TestCase):
    def fixture(self, ranks=1):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        artifact_temporary = tempfile.TemporaryDirectory(dir=root.parent)
        self.addCleanup(artifact_temporary.cleanup)
        artifact_root = Path(artifact_temporary.name).resolve()
        pre = json.loads((ROOT / "preregistration.json").read_text())
        bindings = {key: "a" * 64 for key in pre["required_execution_binding_fields"]}
        summary = {
            "policy": "borrowed",
            "request_count": 8,
            "source_coordinate_count": 60,
            "captured_lineage_edges": 0,
            "all_exact_expected_source_aliases": True,
            "source_values_rechecked_unchanged": True,
        }
        ledger = functional_ledger()
        for rank in range(ranks):
            phase_receipts = []
            for phase_index, phase_name in enumerate(
                ["setup_pre_transition", "post_transition", "post_generation"]
            ):
                artifact = artifact_root / f"rank-{rank}-phase-{phase_name}.json"
                gdn = {"phase": phase_name}
                binding = {
                    "rank": rank,
                    "resident_count": 8,
                    "kv_policy": pre["selected_cell"]["kv_policy"],
                    "gdn_base_policy": pre["selected_cell"]["gdn_base_policy"],
                    "phase": phase_name,
                    "cell_id": f"rank-{rank}-N-8-fixture-ownership-witness",
                    "run_id": "fixture",
                    "gdn_policy": "fixture",
                }
                payload = (
                    json.dumps(
                        {
                            "schema_version": "fixture",
                            "binding": binding,
                            "gdn_phase_witness": gdn,
                            "kv_ownership_witness": {},
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
                artifact.write_bytes(payload)
                phase_receipts.append(
                    {
                        "phase": phase_name,
                        "selected_rows_verified": 6,
                        "full_live_rows_verified": 540,
                        "generation_calls_verified": [0, 1, 64][phase_index],
                        "functional_rebind_edges_verified": [0, 60, 3840][phase_index],
                        "request_rebind_counts": [
                            [0] * 8,
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [8] * 8,
                        ][phase_index],
                        "per_call_isolation_verified": True,
                        "private_request_rows_verified": [0, 60, 480][phase_index],
                        "borrowed_request_rows_verified": [480, 420, 0][phase_index],
                        "actual_serializer_compared": True,
                        "actual_storage_rows_verified": 540,
                        "artifact_relative_path": artifact.relative_to(root.parent).as_posix(),
                        "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                        "artifact_bytes": len(payload),
                        "gdn_sha256": hashlib.sha256(
                            json.dumps(gdn, sort_keys=True, separators=(",", ":")).encode()
                        ).hexdigest(),
                    }
                )
            value = seal(
                {
                    "schema_version": "forkaudit-r40-v32-borrowed-transition-rank-v1",
                    "experiment_id": pre["experiment_id"],
                    "rank": rank,
                    "selected_cell": pre["selected_cell"],
                    "phase_order": ["setup_pre_transition", "post_transition", "post_generation"],
                    "phase_receipts": phase_receipts,
                    "source_reference_coordinate_count": 60,
                    "actual_selected_rows_verified": 18,
                    "actual_storage_rows_verified": 1620,
                    "count_vector": {
                        "source_reference_coordinates": 60,
                        "selected_rows_by_phase": [6, 6, 6],
                        "storage_rows_by_phase": [540, 540, 540],
                        "full_live_rows_by_phase": [540, 540, 540],
                        "generation_calls_by_phase": [0, 1, 64],
                        "functional_rebind_edges_by_phase": [0, 60, 3840],
                        "request_rebind_counts_by_phase": [
                            [0] * 8,
                            [1, 0, 0, 0, 0, 0, 0, 0],
                            [8] * 8,
                        ],
                        "private_request_rows_by_phase": [0, 60, 480],
                        "borrowed_request_rows_by_phase": [480, 420, 0],
                        "primary_memory_hook_events": 0,
                    },
                    "real_builder_verified": True,
                    "actual_phase_serializer_verified": True,
                    "off_path_candidate_detector_used": False,
                    "producer_coverage": {
                        "prebuild_reference_frozen": True,
                        "real_group_observed": True,
                        "borrowed_setup_exact_aliases_observed": True,
                        "functional_rebind_endpoints_observed": True,
                        "actual_serializer_rows_observed": True,
                        "persistent_rechecked_each_phase": True,
                        "all_storage_rows_normalized_against_live_keys": True,
                    },
                    "primary_memory_hook_events": 0,
                    "global_hook_counters": {
                        "selected_builds": 1,
                        "selected_phases": 3,
                        "primary_memory_calls_observed": 8,
                        "primary_memory_hook_events": 0,
                    },
                    "execution_bindings": bindings,
                    "lineage_summary": summary,
                    "lineage_summary_sha256": hashlib.sha256(
                        json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "lineage_receipt": {
                        "opaque_capability_consumed": True,
                        "binding_policy": "borrowed-exact-persistent-alias",
                        "selected_binding_count": 5,
                        "selected_exact_alias_count": 5,
                        "selected_clone_edge_count": 0,
                    },
                    "functional_rebind_ledger": ledger,
                    "formal_gpu_execution": "fixture",
                }
            )
            path = root / f"rank-{rank}/raw"
            path.mkdir(parents=True)
            (path / "real-binding.json").write_text(json.dumps(value))
            zero = seal(
                {
                    "schema_version": "forkaudit-r40-v16-global-absence-v1",
                    "rank": rank,
                    "selected_builds": 1,
                    "selected_phases": 3,
                    "expected_primary_memory_calls": 12,
                    "primary_memory_calls_observed": 12,
                    "primary_call_coverage_proof": True,
                    "primary_absence_proof": True,
                    "primary_memory_hook_events": 0,
                }
            )
            (path / "global-absence.json").write_text(json.dumps(zero))
        return root, pre

    def test_exact_aggregate_derived(self):
        root, pre = self.fixture()
        output = finalize(root, pre, ranks=1)
        self.assertEqual(
            (
                output["total_selected_rows"],
                output["total_storage_rows"],
                output["total_borrowed_setup_aliases"],
                output["total_setup_clone_edges"],
                output["total_functional_rebind_edges"],
                output["total_phase_artifacts"],
                output["total_primary_calls_observed"],
            ),
            (18, 1620, 480, 0, 3840, 3, 12),
        )
        self.assertEqual(output["global_primary_memory_hook_events"], 0)

    def test_orphan_and_nonzero_primary_rejected(self):
        root, pre = self.fixture()
        (root / "orphan.tmp").write_text("x")
        with self.assertRaisesRegex(RuntimeError, "closure"):
            finalize(root, pre, ranks=1)
        (root / "orphan.tmp").unlink()
        path = root / "rank-0/raw/global-absence.json"
        value = json.loads(path.read_text())
        value["primary_memory_hook_events"] = 1
        path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError, "absence proof"):
            finalize(root, pre, ranks=1)

    def test_extra_rank_and_extra_raw_file_rejected(self):
        root, pre = self.fixture()
        (root / "rank-1").mkdir()
        with self.assertRaisesRegex(RuntimeError, "closure"):
            finalize(root, pre, ranks=1)
        (root / "rank-1").rmdir()
        (root / "rank-0/raw/alien.json").write_text("{}")
        with self.assertRaisesRegex(RuntimeError, "closure"):
            finalize(root, pre, ranks=1)

    def test_resealed_authorizing_flag_and_bool_rank_counter_rejected(self):
        root, pre = self.fixture()
        path = root / "rank-0/raw/real-binding.json"
        value = json.loads(path.read_text())
        value["real_builder_verified"] = 1
        path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError, "authorizing flag"):
            finalize(root, pre, ranks=1)
        root, pre = self.fixture()
        path = root / "rank-0/raw/real-binding.json"
        value = json.loads(path.read_text())
        value["global_hook_counters"]["primary_memory_calls_observed"] = True
        path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError, "hook counter drift"):
            finalize(root, pre, ranks=1)

    def test_bool_rank_uppercase_binding_and_cross_rank_artifact_reuse_rejected(self):
        root, pre = self.fixture()
        path = root / "rank-0/raw/real-binding.json"
        value = json.loads(path.read_text())
        value["rank"] = True
        path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError, "rank/cell"):
            finalize(root, pre, ranks=1)
        root, pre = self.fixture()
        path = root / "rank-0/raw/real-binding.json"
        value = json.loads(path.read_text())
        key = next(iter(value["execution_bindings"]))
        value["execution_bindings"][key] = "A" * 64
        path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError, "binding fields/SHA"):
            finalize(root, pre, ranks=1)
        root, pre = self.fixture(ranks=2)
        path0 = root / "rank-0/raw/real-binding.json"
        path1 = root / "rank-1/raw/real-binding.json"
        value0 = json.loads(path0.read_text())
        value1 = json.loads(path1.read_text())
        value1["phase_receipts"][0] = dict(value0["phase_receipts"][0])
        path1.write_text(json.dumps(seal(value1)))
        with self.assertRaisesRegex(RuntimeError, "artifact receipt drift|reused|rank-cell-phase"):
            finalize(root, pre, ranks=2)

    def test_benign_extra_capture_file_rejected(self):
        root, pre = self.fixture()
        (root / "README.txt").write_text("extra")
        with self.assertRaisesRegex(RuntimeError, "closure"):
            finalize(root, pre, ranks=1)

    def test_symlink_extra_lineage_field_and_phase_order_mismatch_rejected(self):
        root, pre = self.fixture()
        (root / "link").symlink_to(root / "rank-0")
        with self.assertRaisesRegex(RuntimeError, "symlink|directory closure"):
            finalize(root, pre, ranks=1)
        root, pre = self.fixture()
        path = root / "rank-0/raw/real-binding.json"
        value = json.loads(path.read_text())
        value["lineage_summary"]["extra"] = True
        value["lineage_summary_sha256"] = hashlib.sha256(
            json.dumps(value["lineage_summary"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError, "lineage summary drift"):
            finalize(root, pre, ranks=1)
        root, pre = self.fixture()
        path = root / "rank-0/raw/real-binding.json"
        value = json.loads(path.read_text())
        value["phase_receipts"][0]["phase"] = "post_transition"
        path.write_text(json.dumps(seal(value)))
        with self.assertRaisesRegex(RuntimeError, "phase receipt/order"):
            finalize(root, pre, ranks=1)

    def test_fifo_socket_and_hardlink_nodes_rejected_by_lstat_closure(self):
        root, pre = self.fixture()
        os.mkfifo(root / "alien.fifo")
        with self.assertRaisesRegex(RuntimeError, "special node"):
            finalize(root, pre, ranks=1)
        root, pre = self.fixture()
        sock = socket.socket(socket.AF_UNIX)
        self.addCleanup(sock.close)
        sock.bind(str(root / "alien.socket"))
        with self.assertRaisesRegex(RuntimeError, "special node"):
            finalize(root, pre, ranks=1)
        root, pre = self.fixture()
        os.link(root / "rank-0/raw/real-binding.json", root / "alien-hardlink.json")
        with self.assertRaisesRegex(RuntimeError, "hardlink"):
            finalize(root, pre, ranks=1)

    def test_aggregate_command_refuses_preexisting_terminal_output_without_mutation(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        terminal_root = Path(temporary.name).resolve()
        formal = terminal_root / "r40-formal"
        formal.mkdir()
        output = formal / "aggregate.json"
        sentinel = b"do-not-overwrite\n"
        output.write_bytes(sentinel)
        preregistration = ROOT / "preregistration.json"
        expected = hashlib.sha256(preregistration.read_bytes()).hexdigest()
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "executed_source/r40_finalize.py"),
                "--terminal-root",
                str(terminal_root),
                "--capture-root",
                str(terminal_root / "missing-capture"),
                "--preregistration",
                str(preregistration),
                "--expected-prereg-sha256",
                expected,
                "--output",
                str(output),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("overwrite", result.stderr)
        self.assertEqual(output.read_bytes(), sentinel)


if __name__ == "__main__":
    unittest.main()
