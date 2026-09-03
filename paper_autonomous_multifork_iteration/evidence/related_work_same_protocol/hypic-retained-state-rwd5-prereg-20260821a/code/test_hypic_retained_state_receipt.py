import json
import os
import tempfile
import unittest
from pathlib import Path

import torch

import hypic_retained_state_receipt as receipt
import replay_hypic_retained_state_bytes as blind
import run_hypic_retained_state_bytes as runner


class Key:
    def __init__(self, ids):
        self.ids = list(ids)

    def raw_token_ids(self):
        return list(self.ids)


class Node:
    def __init__(self, node_id, ids=(), slots=(), mamba=()):
        self.id = node_id
        self.key = Key(ids)
        self.value = torch.tensor(slots, dtype=torch.int64)
        self.mamba_value = (
            torch.tensor(mamba, dtype=torch.int64) if mamba else None
        )
        self.full_lock_ref = 0
        self.mamba_lock_ref = 0
        self.parent = None
        self.children = {}


class FullPool:
    kv_cache_layout = "nhd"

    def __init__(self):
        self.k_buffer = [torch.zeros((16, 2, 3), dtype=torch.float32)]
        self.v_buffer = [torch.zeros((16, 2, 3), dtype=torch.float32)]


class Allocator:
    def __init__(self):
        self.size = 15
        self.page_size = 1
        self.free_pages = torch.arange(5, 16, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)
        self.pool = FullPool()

    def get_kvcache(self):
        return self.pool

    def available_size(self):
        return len(self.free_pages) + len(self.release_pages)


class MambaAllocator:
    def __init__(self):
        self.size = 5
        self.free_slots = torch.tensor([1, 3, 4, 5], dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)


class MambaCache:
    def __init__(self, pic):
        self.conv = [torch.zeros((2, 6, 3, 2), dtype=torch.float32)]
        self.temporal = torch.zeros((2, 6, 2, 2, 2), dtype=torch.float32)
        self.transition = (
            torch.zeros((2, 6, 2, 2, 2), dtype=torch.float32) if pic else None
        )
        self.conv_tails = (
            [torch.zeros((2, 6, 3, 2), dtype=torch.float32)] if pic else None
        )


class ReqPool:
    def __init__(self, pic=False):
        self.mamba_allocator = MambaAllocator()
        self.mamba_pool = type("Pool", (), {"mamba_cache": MambaCache(pic)})()


class MambaRadixCache:
    def __init__(self):
        self.root_node = Node(0)
        child = Node(1, [1, 2, 3], [1, 2, 3], [2])
        child.parent = self.root_node
        self.root_node.children[1] = child
        self.token_to_kv_pool_allocator = Allocator()
        self.req_to_token_pool = ReqPool(pic=False)


class SegmentEntry:
    def __init__(self, ids, slots, mamba_slot):
        self.seg_hash = bytes.fromhex(receipt.segment_hash_hex(ids))
        self.token_ids = torch.tensor(ids, dtype=torch.int64)
        self.full_kv_slots = torch.tensor(slots, dtype=torch.int64)
        self.mamba_state_slot = mamba_slot
        self.lock_ref = 0


class PICache:
    def __init__(self):
        first = SegmentEntry([1, 2], [1, 2], 2)
        second = SegmentEntry([3, 4], [3, 4], 3)
        self._entries = {first.seg_hash: first, second.seg_hash: second}
        self.token_to_kv_pool_allocator = Allocator()
        self.req_to_token_pool = ReqPool(pic=True)


class ReceiptTest(unittest.TestCase):
    def run_snapshot(self, cache, target):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target_path = root / "target.json"
            target_path.write_text(json.dumps(target))
            old = dict(os.environ)
            try:
                os.environ[receipt.TARGET_ENV] = str(target_path)
                os.environ[receipt.OUTPUT_ENV] = str(root)
                receipt.maybe_emit_owned_state_snapshot(cache)
                output = json.loads((root / f"{target['snapshot_id']}.json").read_text())
                return output
            finally:
                os.environ.clear()
                os.environ.update(old)

    @staticmethod
    def target(mode, snapshot_id="unit"):
        value = {
            "schema": receipt.TARGET_SCHEMA,
            "snapshot_id": snapshot_id,
            "official_commit": receipt.OFFICIAL_COMMIT,
            "mode": mode,
            "rank": 0,
            "workload_id": "qasper-6",
            "document_token_ids": [1, 2, 3, 4],
            "document_token_sha256": receipt.token_sha256([1, 2, 3, 4]),
            "seam_tokens": 1,
        }
        if mode == "transition_rope_recompute":
            value["segment_token_ids"] = [[1, 2], [3, 4]]
        return value

    def test_prefix_receipt_counts_only_cached_owned_prefix(self):
        output = self.run_snapshot(MambaRadixCache(), self.target("prefix_cache"))
        self.assertEqual(output["selection"]["owned_document_tokens"], 3)
        self.assertEqual(output["selection"]["full_kv_slots"], [1, 2, 3])
        self.assertEqual(output["selection"]["mamba_state_slots"], [2])
        self.assertFalse(output["component_presence"]["transition"]["present"])
        self.assertEqual(
            output["tensor_payload"]["union"]["unique_overlap_aware_bytes"],
            output["tensor_payload"]["union"]["naive_range_bytes"],
        )

    def test_hypic_receipt_binds_both_segments_and_pic_components(self):
        output = self.run_snapshot(
            PICache(), self.target("transition_rope_recompute")
        )
        self.assertEqual(output["selection"]["owned_document_tokens"], 4)
        self.assertEqual(output["selection"]["expected_measured_cached_tokens"], 3)
        self.assertEqual(output["selection"]["mamba_state_slots"], [2, 3])
        self.assertTrue(output["component_presence"]["transition"]["present"])
        self.assertTrue(output["component_presence"]["conv_tails[0]"]["present"])

    def test_overlap_union_deduplicates_ranges(self):
        tensor = torch.zeros(16, dtype=torch.uint8)
        first = receipt._record_range(
            tensor,
            tensor_name="x",
            component="test",
            start=2,
            end=10,
            selection={},
        )
        second = receipt._record_range(
            tensor,
            tensor_name="x",
            component="test",
            start=6,
            end=14,
            selection={},
        )
        summary = receipt._union_summary([first, second])
        self.assertEqual(summary["naive_range_bytes"], 16)
        self.assertEqual(summary["unique_overlap_aware_bytes"], 12)

    def test_target_absence_is_retryable_but_does_not_emit_a_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.target("prefix_cache", "not-ready")
            target_path = root / "target.json"
            target_path.write_text(json.dumps(target))
            cache = MambaRadixCache()
            cache.root_node.children.clear()
            old = dict(os.environ)
            try:
                os.environ[receipt.TARGET_ENV] = str(target_path)
                os.environ[receipt.OUTPUT_ENV] = str(root)
                receipt.maybe_emit_owned_state_snapshot(cache)
                self.assertFalse((root / "not-ready.json").exists())
            finally:
                os.environ.clear()
                os.environ.update(old)

    def test_terminal_check_requires_slot_return_and_empty_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.target("prefix_cache", "terminal")
            target_path = root / "target.json"
            target_path.write_text(json.dumps(target))
            cache = MambaRadixCache()
            old = dict(os.environ)
            try:
                os.environ[receipt.TARGET_ENV] = str(target_path)
                os.environ[receipt.OUTPUT_ENV] = str(root)
                receipt.maybe_emit_owned_state_snapshot(cache)
                cache.root_node.children.clear()
                cache.token_to_kv_pool_allocator.free_pages = torch.arange(
                    1, 16, dtype=torch.int64
                )
                cache.req_to_token_pool.mamba_allocator.free_slots = torch.arange(
                    1, 6, dtype=torch.int64
                )
                scheduler = type(
                    "Scheduler",
                    (),
                    {
                        "tree_cache": cache,
                        "token_to_kv_pool_allocator": cache.token_to_kv_pool_allocator,
                        "req_to_token_pool": cache.req_to_token_pool,
                    },
                )()
                receipt.maybe_emit_terminal_ownership_snapshot(scheduler)
                terminal = json.loads((root / "terminal.terminal.json").read_text())
                self.assertTrue(terminal["passed"])
            finally:
                os.environ.clear()
                os.environ.update(old)

    def test_target_builder_binds_only_two_document_segments_for_hypic(self):
        workload = {
            "workload_id": "qasper-6",
            "document_token_ids": [1, 2, 3, 4],
            "document_token_sha256": receipt.token_sha256([1, 2, 3, 4]),
            "direct_token_ids": [1, 2, 3, 4, 5],
            "segment_offsets": [[0, 2], [2, 4], [4, 5]],
        }
        target = runner._target(workload, "transition_rope_recompute", 0)
        self.assertEqual(target["segment_token_ids"], [[1, 2], [3, 4]])
        self.assertNotIn(5, target["document_token_ids"])

    def test_blind_replay_recomputes_receipt_and_terminal_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = self.target("prefix_cache", "blind")
            target_path = root / "target.json"
            receipt_path = root / "blind.json"
            terminal_path = root / "blind.terminal.json"
            raw_path = root / "raw.json"
            target_path.write_text(json.dumps(target))
            cache = MambaRadixCache()
            old = dict(os.environ)
            try:
                os.environ[receipt.TARGET_ENV] = str(target_path)
                os.environ[receipt.OUTPUT_ENV] = str(root)
                receipt.maybe_emit_owned_state_snapshot(cache)
                cache.root_node.children.clear()
                cache.token_to_kv_pool_allocator.free_pages = torch.arange(
                    1, 16, dtype=torch.int64
                )
                cache.req_to_token_pool.mamba_allocator.free_slots = torch.arange(
                    1, 6, dtype=torch.int64
                )
                scheduler = type(
                    "Scheduler",
                    (),
                    {
                        "tree_cache": cache,
                        "token_to_kv_pool_allocator": cache.token_to_kv_pool_allocator,
                        "req_to_token_pool": cache.req_to_token_pool,
                    },
                )()
                receipt.maybe_emit_terminal_ownership_snapshot(scheduler)
            finally:
                os.environ.clear()
                os.environ.update(old)
            raw_path.write_text(
                json.dumps(
                    {
                        "schema": "forkaudit-hypic-retained-state-shard-v1",
                        "status": "completed",
                        "mode": "prefix_cache",
                        "rank": 0,
                        "workload": {
                            "workload_id": "qasper-6",
                            "document_token_sha256": target["document_token_sha256"],
                        },
                        "cache_observation": {"cached_tokens": 3},
                    }
                )
            )
            result = blind.replay_one(receipt_path, terminal_path, raw_path)
            self.assertGreater(result["payload_bytes"], 0)

    def test_formal_launcher_contains_no_unaffected_gpu_arm(self):
        launcher = Path(__file__).with_name(
            "launch_hypic_retained_state_bytes_8gpu.sh"
        ).read_text()
        self.assertIn("run_mode prefix_cache", launcher)
        self.assertIn("run_mode transition_rope_recompute", launcher)
        self.assertNotIn("run_mode full_recompute", launcher)
        self.assertNotIn("run_mode comem", launcher.lower())


if __name__ == "__main__":
    unittest.main()
