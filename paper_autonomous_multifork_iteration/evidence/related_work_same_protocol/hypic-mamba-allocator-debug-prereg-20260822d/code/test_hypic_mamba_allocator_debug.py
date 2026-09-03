#!/usr/bin/env python3
import json
import inspect
import copy
import re
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

import hypic_retained_state_receipt as receipt
import run_hypic_mamba_allocator_debug as runner


class Allocator:
    def __init__(self):
        self.size = 5
        self.device = "cpu"
        self._alloc_iter = None
        self.free_slots = torch.tensor([1, 4, 4, 5], dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)


Allocator.__name__ = "MambaSlotAllocator"
Allocator.__module__ = "sglang.srt.mem_cache.allocator.mamba"


def entry(slot, token_ids, seg_hash):
    return SimpleNamespace(
        seg_hash=seg_hash,
        mamba_state_slot=slot,
        token_ids=torch.tensor(token_ids, dtype=torch.int64),
        lock_ref=0,
    )


class PICache:
    def __init__(self, allocator, entries):
        self.mamba_allocator = allocator
        self.req_to_token_pool = SimpleNamespace(mamba_allocator=allocator)
        self._entries = entries


PICache.__module__ = "sglang.srt.pic.picache"


class MambaAllocatorDebugInventoryTest(unittest.TestCase):
    def make_case(self):
        allocator = Allocator()
        segments = [[10, 11], [12, 13]]
        hash_a = bytes.fromhex(receipt.segment_hash_hex(segments[0]))
        hash_b = bytes.fromhex(receipt.segment_hash_hex(segments[1]))
        tree = PICache(
            allocator,
            {
                hash_a: entry(2, segments[0], hash_a),
                hash_b: entry(3, segments[1], hash_b),
            },
        )
        document = segments[0] + segments[1]
        target = {
            "authority": {},
            "document_token_ids": document,
            "document_token_sha256": receipt.token_sha256(document),
            "mode": "transition_rope_recompute",
            "official_commit": runner.HYPIC_COMMIT,
            "rank": 0,
            "snapshot_id": "allocator-debug-transition_rope_recompute-rank-0",
            "workload_id": "qasper-6",
            "schema": "forkaudit-hypic-retained-state-target-v2",
            "seam_tokens": 8,
            "segment_token_ids": segments,
            "workload_binding": {
                "dataset": "qasper",
                "document_token_sha256": receipt.token_sha256(document),
                "document_tokens": len(document),
                "prompt_token_sha256": "33" * 32,
                "query_tokens": 99,
                "segment_offsets": [[0, 2], [2, 4], [4, 103]],
                "source_index": 6,
                "token_identity_verified": True,
                "workload_id": "qasper-6",
            },
        }
        selection = {
            "mamba_state_slots": [2, 3],
            "entries": [
                {"segment_hash_hex": hash_a.hex()},
                {"segment_hash_hex": hash_b.hex()},
            ],
        }
        return allocator, tree, target, selection

    def make_row(self):
        allocator, tree, target, selection = self.make_case()
        temp = tempfile.TemporaryDirectory()
        target_path = Path(temp.name) / "target.json"
        target_path.write_text(json.dumps(target, sort_keys=True) + "\n")
        row = receipt._mamba_allocator_debug_inventory(
            tree, target, target_path, selection
        )
        target_sha = receipt._sha256_file(target_path)
        return temp, target, row, target_sha

    def validate_mock(self, row, target, target_sha):
        return runner.validate(
            row,
            target=target,
            target_sha256=target_sha,
            expected_allocator_device="cpu",
            expected_tensor_device="cpu",
            expected_allocator_size=5,
        )

    def test_raw_duplicate_representation_is_preserved_and_derived(self):
        allocator, tree, target, selection = self.make_case()
        before = allocator.free_slots.clone()
        with tempfile.TemporaryDirectory() as temp:
            target_path = Path(temp) / "target.json"
            target_path.write_text(json.dumps(target, sort_keys=True) + "\n")
            row = receipt._mamba_allocator_debug_inventory(
                tree, target, target_path, selection
            )
        self.assertTrue(torch.equal(before, allocator.free_slots))
        self.assertEqual(row["schema"], "hypic-rwd5-mamba-allocator-debug-v1")
        self.assertFalse(row["mutation_performed"])
        observed = row["allocator"]
        self.assertEqual(observed["raw_free_slots"], [1, 4, 4, 5])
        self.assertEqual(observed["raw_count"], 4)
        self.assertEqual(observed["unique_count"], 3)
        self.assertEqual(
            observed["duplicates"], [{"slot": 4, "count": 2, "positions": [1, 2]}]
        )
        self.assertEqual(observed["duplicate_excess_count"], 1)
        self.assertEqual(observed["missing_from_raw_unique_domain"], [2, 3])
        self.assertEqual(observed["out_of_domain"], [])
        self.assertFalse(
            observed["release_representations"]["release_slots"]["attribute_present"]
        )
        self.assertEqual(row["cache"]["target_mamba_state_slots"], [2, 3])
        self.assertEqual(
            row["cache"]["target_slots_present_in_raw_free_slots"],
            {"2": [], "3": []},
        )
        self.assertEqual(observed["free_slots_tensor"]["dtype"], "torch.int64")
        self.assertEqual(observed["free_slots_tensor"]["shape"], [4])

    def test_alias_mismatch_is_rejected(self):
        _allocator, tree, target, selection = self.make_case()
        tree.mamba_allocator = Allocator()
        with tempfile.TemporaryDirectory() as temp:
            target_path = Path(temp) / "target.json"
            target_path.write_text("{}\n")
            with self.assertRaisesRegex(receipt.ReceiptError, "PIC allocator alias"):
                receipt._mamba_allocator_debug_inventory(
                    tree, target, target_path, selection
                )

    def test_independent_validator_rederives_duplicate_and_domain(self):
        temp, target, row, target_sha = self.make_row()
        self.addCleanup(temp.cleanup)
        validation = self.validate_mock(row, target, target_sha)
        self.assertEqual(
            validation["status"],
            "passed_exact_duplicate_representation_capture",
        )
        self.assertEqual(validation["duplicate_excess_count"], 1)

    def test_validator_rejects_forged_duplicate_positions(self):
        temp, target, row, target_sha = self.make_row()
        self.addCleanup(temp.cleanup)
        row["allocator"]["duplicates"][0]["positions"] = [0, 1]
        with self.assertRaisesRegex(RuntimeError, "duplicate derivation"):
            self.validate_mock(row, target, target_sha)

    def test_combined_fully_resigned_allocator_tensor_cache_target_tamper_rejected(self):
        temp, target, row, target_sha = self.make_row()
        self.addCleanup(temp.cleanup)
        forged = copy.deepcopy(row)
        allocator = forged["allocator"]
        allocator["module"] = "attacker.allocator"
        allocator["device"] = "cuda:7"
        allocator["alloc_iter_is_none"] = False
        allocator["fields"]["device"] = {"kind": "str", "value": "cuda:7"}
        allocator["fields"]["_alloc_iter"] = {"kind": "object", "module": "builtins", "qualname": "iter"}
        tensor = allocator["free_slots_tensor"]
        tensor["storage_data_ptr"] += 4096
        tensor["storage_id"] = "00" * 32
        tensor["tensor_data_ptr"] += 4096
        tensor["absolute_byte_start"] += 4096
        tensor["absolute_byte_end"] += 4096
        cache = forged["cache"]
        cache["module"] = "attacker.cache"
        cache["entry_count"] += 1
        cache["entries"][0]["lock_ref"] = 9
        cache["entries"][0]["token_sha256"] = "44" * 32
        cache["target_slots_present_in_raw_free_slots"] = {"2": [0], "3": []}
        forged["target"]["document_token_sha256"] = "55" * 32
        with self.assertRaises(RuntimeError):
            self.validate_mock(forged, target, target_sha)

    def test_each_storage_and_membership_tamper_is_rejected(self):
        temp, target, row, target_sha = self.make_row()
        self.addCleanup(temp.cleanup)
        paths = (
            ("storage_data_ptr", lambda value: value + 1),
            ("storage_nbytes", lambda value: value + 8),
            ("storage_id", lambda _value: "00" * 32),
            ("tensor_data_ptr", lambda value: value + 8),
            ("byte_start", lambda value: value + 8),
            ("absolute_byte_end", lambda value: value + 8),
        )
        for name, mutate in paths:
            with self.subTest(name=name):
                forged = copy.deepcopy(row)
                tensor = forged["allocator"]["free_slots_tensor"]
                tensor[name] = mutate(tensor[name])
                with self.assertRaises(RuntimeError):
                    self.validate_mock(forged, target, target_sha)
        forged = copy.deepcopy(row)
        forged["cache"]["target_slots_present_in_raw_free_slots"]["2"] = [0]
        with self.assertRaisesRegex(RuntimeError, "membership replay"):
            self.validate_mock(forged, target, target_sha)

    def test_each_allocator_cache_and_target_tamper_is_rejected(self):
        temp, target, row, target_sha = self.make_row()
        self.addCleanup(temp.cleanup)
        mutations = {
            "allocator_module": lambda value: value["allocator"].__setitem__("module", "attacker.module"),
            "allocator_device": lambda value: value["allocator"].__setitem__("device", "cuda:7"),
            "allocator_iter": lambda value: value["allocator"].__setitem__("alloc_iter_is_none", False),
            "allocator_field": lambda value: value["allocator"]["fields"]["size"].__setitem__("value", 6),
            "cache_module": lambda value: value["cache"].__setitem__("module", "attacker.cache"),
            "cache_count": lambda value: value["cache"].__setitem__("entry_count", 3),
            "cache_token_hash": lambda value: value["cache"]["entries"][0].__setitem__("token_sha256", "66" * 32),
            "cache_lock": lambda value: value["cache"]["entries"][0].__setitem__("lock_ref", 1),
            "target_document": lambda value: value["target"].__setitem__("document_token_sha256", "77" * 32),
            "target_segments": lambda value: value["target"].__setitem__("segment_token_counts", [1, 3]),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                forged = copy.deepcopy(row)
                mutate(forged)
                with self.assertRaises(RuntimeError):
                    self.validate_mock(forged, target, target_sha)

    def test_inventory_source_has_no_allocator_mutation_call(self):
        source = inspect.getsource(receipt._mamba_allocator_debug_inventory)
        self.assertNotIn(".free(", source)
        self.assertNotIn(".alloc(", source)
        self.assertNotIn("torch.unique", source)

    def test_launcher_is_one_cell_closed_environment_and_formal_empty(self):
        launcher = Path(__file__).with_name(
            "launch_hypic_mamba_allocator_debug_1gpu.sh"
        ).read_text()
        self.assertIn("MODE=transition_rope_recompute", launcher)
        self.assertNotIn("MODE=prefix_cache", launcher)
        self.assertIn("GPU_INDEX=0", launcher)
        self.assertIn("setsid /usr/bin/env -i", launcher)
        self.assertIn("PYTHONSAFEPATH=1", launcher)
        self.assertIn("formal-receipts-disabled", launcher)
        self.assertIn('[[ -z "$(find "$OUTPUT_ROOT/formal-receipts-disabled"', launcher)
        self.assertIn("-mindepth 1 -print -quit", launcher)
        self.assertIn('sha256sum -c "$MODEL_DIR/model-weights.sha256"', launcher)
        self.assertIn('sha256sum -c "$MODEL_DIR/model-artifacts.sha256"', launcher)
        self.assertIn("duplicate_excess_count", launcher)
        self.assertIn("completed_debug_only_not_formal_evidence", launcher)
        self.assertIn("allocator_debug_receipt_sha256", launcher)
        self.assertIn("nvidia-smi-after.txt", launcher)

    def test_cleanup_is_bounded_term_poll_kill_wait(self):
        launcher = Path(__file__).with_name(
            "launch_hypic_mamba_allocator_debug_1gpu.sh"
        ).read_text()
        begin = launcher.index("cleanup_server()")
        end = launcher.index("\n}\n\non_exit()", begin)
        cleanup = launcher[begin:end]
        term = cleanup.index("kill -TERM")
        poll = cleanup.index("for attempt in")
        kill = cleanup.index("kill -KILL")
        wait = cleanup.index('wait "${old_pid}"')
        absence = cleanup.index("server PID/PGID survived cleanup")
        self.assertLess(term, poll)
        self.assertLess(poll, kill)
        self.assertLess(kill, wait)
        self.assertLess(wait, absence)

    def test_cleanup_reaps_real_sigterm_ignoring_pid_and_process_group(self):
        launcher = Path(__file__).with_name(
            "launch_hypic_mamba_allocator_debug_1gpu.sh"
        ).read_text()
        match = re.search(r"(?ms)^cleanup_server\(\) \{\n.*?^\}\n", launcher)
        self.assertIsNotNone(match)
        script = "\n".join(
            [
                "set -Eeuo pipefail",
                match.group(0),
                f"setsid {sys.executable!r} -c \"import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)\" &",
                "SERVER_PID=$!",
                "command sleep 0.2",
                "sleep() { command sleep 0.01; }",
                "OLD_PID=$SERVER_PID",
                "cleanup_server",
                "! kill -0 $OLD_PID 2>/dev/null",
                "! kill -0 -- -$OLD_PID 2>/dev/null",
            ]
        )
        started = time.monotonic()
        result = subprocess.run(
            ["bash", "-c", script], text=True, capture_output=True, timeout=5
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertLess(time.monotonic() - started, 3.0)


if __name__ == "__main__":
    unittest.main()
