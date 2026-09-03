#!/usr/bin/env python3
import json
import inspect
import tempfile
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


def entry(slot, token_ids):
    return SimpleNamespace(
        mamba_state_slot=slot,
        token_ids=torch.tensor(token_ids, dtype=torch.int64),
        lock_ref=0,
    )


class MambaAllocatorDebugInventoryTest(unittest.TestCase):
    def make_case(self):
        allocator = Allocator()
        hash_a = bytes.fromhex("11" * 32)
        hash_b = bytes.fromhex("22" * 32)
        tree = SimpleNamespace(
            mamba_allocator=allocator,
            req_to_token_pool=SimpleNamespace(mamba_allocator=allocator),
            _entries={hash_a: entry(2, [10, 11]), hash_b: entry(3, [12, 13])},
        )
        target = {
            "mode": "transition_rope_recompute",
            "rank": 0,
            "snapshot_id": "allocator-debug-transition_rope_recompute-rank-0",
            "workload_id": "qasper-6",
            "document_token_sha256": "33" * 32,
        }
        selection = {
            "mamba_state_slots": [2, 3],
            "entries": [
                {"segment_hash_hex": hash_a.hex()},
                {"segment_hash_hex": hash_b.hex()},
            ],
        }
        return allocator, tree, target, selection

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
        allocator, tree, target, selection = self.make_case()
        with tempfile.TemporaryDirectory() as temp:
            target_path = Path(temp) / "target.json"
            target_path.write_text(json.dumps(target, sort_keys=True) + "\n")
            row = receipt._mamba_allocator_debug_inventory(
                tree, target, target_path, selection
            )
            target_sha = receipt._sha256_file(target_path)
        row["allocator"]["class"] = "MambaSlotAllocator"
        row["cache"]["class"] = "PICache"
        validation = runner.validate(
            row, workload_id="qasper-6", target_sha256=target_sha
        )
        self.assertEqual(
            validation["status"],
            "passed_exact_duplicate_representation_capture",
        )
        self.assertEqual(validation["duplicate_excess_count"], 1)

    def test_validator_rejects_forged_duplicate_positions(self):
        allocator, tree, target, selection = self.make_case()
        with tempfile.TemporaryDirectory() as temp:
            target_path = Path(temp) / "target.json"
            target_path.write_text(json.dumps(target, sort_keys=True) + "\n")
            row = receipt._mamba_allocator_debug_inventory(
                tree, target, target_path, selection
            )
            target_sha = receipt._sha256_file(target_path)
        row["allocator"]["class"] = "MambaSlotAllocator"
        row["cache"]["class"] = "PICache"
        row["allocator"]["duplicates"][0]["positions"] = [0, 1]
        with self.assertRaisesRegex(RuntimeError, "duplicate derivation"):
            runner.validate(row, workload_id="qasper-6", target_sha256=target_sha)

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
        self.assertIn("duplicate_excess_count", launcher)
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
        wait = cleanup.index('wait "${SERVER_PID}"')
        self.assertLess(term, poll)
        self.assertLess(poll, kill)
        self.assertLess(kill, wait)


if __name__ == "__main__":
    unittest.main()
