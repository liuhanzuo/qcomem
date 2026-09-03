import copy
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

import build_hypic_retained_state_static as static_builder
import hypic_retained_state_receipt as receipt
import replay_hypic_retained_state_bytes as blind
import run_hypic_retained_state_bytes as runner


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(value))


def storage_contract(dtype="torch.float32"):
    value = {
        "schema": "hypic-rwd5-model-storage-contract-v2",
        "config_sha256": "c" * 64,
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 3,
        "full_attention_layer_ids": [2],
        "recurrent_layer_ids": [0, 1],
        "full_attention_layer_count": 1,
        "recurrent_layer_count": 2,
        "conv_tensor_count": 1,
        "temporal_tensor_count": 1,
        "dtype": dtype,
        "kv_layout": "nhd",
        "kv_slot_axis": 0,
        "mamba_layer_axis": 0,
        "mamba_slot_axis": 1,
        "page_size": 1,
        "enable_int8_mamba_checkpoint": False,
        "mode_components": {
            "prefix_cache": {"transition_tensor_count": 0, "conv_tails_tensor_count": 0},
            "transition_rope_recompute": {"transition_tensor_count": 1, "conv_tails_tensor_count": 1},
        },
    }
    return value


class Key:
    def __init__(self, ids): self.ids = list(ids)
    def raw_token_ids(self): return list(self.ids)


class Node:
    def __init__(self, node_id, ids=(), slots=(), mamba=()):
        self.id = node_id; self.key = Key(ids)
        self.value = torch.tensor(slots, dtype=torch.int64)
        self.mamba_value = torch.tensor(mamba, dtype=torch.int64) if mamba else None
        self.full_lock_ref = 0; self.mamba_lock_ref = 0
        self.parent = None; self.children = {}


class FullPool:
    kv_cache_layout = "nhd"
    def __init__(self, mismatched=False):
        self.k_buffer = [torch.zeros((16, 2, 3), dtype=torch.float32)]
        value_shape = (16, 2, 4) if mismatched else (16, 2, 3)
        self.v_buffer = [torch.zeros(value_shape, dtype=torch.float32)]


class Allocator:
    def __init__(self, mismatched=False):
        self.size = 15; self.page_size = 1
        self.free_pages = torch.arange(5, 16, dtype=torch.int64)
        self.release_pages = torch.empty(0, dtype=torch.int64)
        self.pool = FullPool(mismatched)
    def get_kvcache(self): return self.pool
    def available_size(self): return len(self.free_pages) + len(self.release_pages)


class MambaAllocator:
    def __init__(self):
        self.size = 5; self.free_slots = torch.tensor([1, 3, 4, 5], dtype=torch.int64)
    def available_size(self): return len(self.free_slots)


class MambaCache:
    def __init__(self, pic, *, missing_transition=False, missing_tails=False):
        self.conv = [torch.zeros((2, 6, 3, 2), dtype=torch.float32)]
        self.temporal = torch.zeros((2, 6, 2, 2, 2), dtype=torch.float32)
        self.transition = (
            None if not pic or missing_transition
            else torch.zeros((2, 6, 2, 2, 2), dtype=torch.float32)
        )
        self.conv_tails = (
            None if not pic or missing_tails
            else [torch.zeros((2, 6, 3, 2), dtype=torch.float32)]
        )


class ReqPool:
    def __init__(self, pic=False, **kwargs):
        self.mamba_allocator = MambaAllocator()
        self.mamba_pool = type("Pool", (), {"mamba_cache": MambaCache(pic, **kwargs)})()
        self.mamba_ckpt_pool = None


class MambaRadixCache:
    def __init__(self, *, mismatched=False):
        self.root_node = Node(0)
        child = Node(1, [1, 2, 3], [1, 2, 3], [2]); child.parent = self.root_node
        self.root_node.children[1] = child
        self.token_to_kv_pool_allocator = Allocator(mismatched)
        self.req_to_token_pool = ReqPool(pic=False)
        self.int8_ckpt_pool = None


class SegmentEntry:
    def __init__(self, ids, slots, mamba_slot):
        self.seg_hash = bytes.fromhex(receipt.segment_hash_hex(ids))
        self.token_ids = torch.tensor(ids, dtype=torch.int64)
        self.full_kv_slots = torch.tensor(slots, dtype=torch.int64)
        self.mamba_state_slot = mamba_slot; self.lock_ref = 0


class PICache:
    def __init__(self, **kwargs):
        first = SegmentEntry([1, 2], [1, 2], 2)
        second = SegmentEntry([3, 4], [3, 4], 3)
        self._entries = {first.seg_hash: first, second.seg_hash: second}
        self.token_to_kv_pool_allocator = Allocator()
        self.req_to_token_pool = ReqPool(pic=True, **kwargs)
        self.mamba_pool = self.req_to_token_pool.mamba_pool


class ReceiptTest(unittest.TestCase):
    @staticmethod
    def target(mode, snapshot_id="unit", authority=None):
        value = {
            "schema": receipt.TARGET_SCHEMA, "snapshot_id": snapshot_id,
            "official_commit": receipt.OFFICIAL_COMMIT, "mode": mode, "rank": 0,
            "workload_id": "qasper-6", "document_token_ids": [1, 2, 3, 4],
            "document_token_sha256": receipt.token_sha256([1, 2, 3, 4]),
            "seam_tokens": 1, "authority": {} if authority is None else authority,
            "workload_binding": {
                "workload_id": "qasper-6", "dataset": "qasper", "source_index": 6,
                "document_tokens": 4, "query_tokens": 1,
                "prompt_token_sha256": "p" * 64,
                "document_token_sha256": receipt.token_sha256([1, 2, 3, 4]),
                "segment_offsets": [[0, 2], [2, 4]], "token_identity_verified": True,
            },
        }
        if mode == "transition_rope_recompute": value["segment_token_ids"] = [[1, 2], [3, 4]]
        return value

    def run_snapshot(self, cache, target):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); target_path = root / "target.json"
            target_path.write_text(json.dumps(target)); old = dict(os.environ)
            fake = {"bindings": {}, "scheduler_process": {},
                    "storage_contract": storage_contract()}
            try:
                os.environ[receipt.TARGET_ENV] = str(target_path)
                os.environ[receipt.OUTPUT_ENV] = str(root)
                with mock.patch.object(receipt, "_bound_authority", return_value=fake):
                    receipt.maybe_emit_owned_state_snapshot(cache)
                return json.loads((root / f"{target['snapshot_id']}.json").read_text())
            finally:
                os.environ.clear(); os.environ.update(old)

    def test_prefix_receipt_exact_components_and_lock_refs(self):
        output = self.run_snapshot(MambaRadixCache(), self.target("prefix_cache"))
        self.assertEqual(output["selection"]["full_kv_slots"], [1, 2, 3])
        self.assertEqual(output["selection"]["entries"][0]["lock_refs"], {"full": 0, "mamba": 0})
        names = {row["tensor_name"] for row in output["tensor_payload"]["records"]}
        self.assertEqual(names, {"full_kv.key[0]", "full_kv.value[0]", "mamba.conv[0]", "mamba.temporal"})

    def test_hypic_receipt_requires_all_pic_components(self):
        output = self.run_snapshot(PICache(), self.target("transition_rope_recompute"))
        names = {row["tensor_name"] for row in output["tensor_payload"]["records"]}
        self.assertIn("mamba.transition", names); self.assertIn("mamba.conv_tails[0]", names)
        self.assertTrue(all(entry["lock_ref"] == 0 for entry in output["selection"]["entries"]))

    def test_producer_rejects_missing_transition(self):
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(PICache(missing_transition=True), self.target("transition_rope_recompute"))

    def test_producer_rejects_missing_conv_tails(self):
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(PICache(missing_tails=True), self.target("transition_rope_recompute"))

    def test_producer_rejects_kv_shape_mismatch(self):
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(MambaRadixCache(mismatched=True), self.target("prefix_cache"))

    def test_producer_rejects_prefix_int8_checkpoint_pool(self):
        cache = MambaRadixCache(); cache.req_to_token_pool.mamba_ckpt_pool = object()
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(cache, self.target("prefix_cache"))

    def test_producer_rejects_pic_mamba_pool_identity_drift(self):
        cache = PICache(); cache.mamba_pool = object()
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(cache, self.target("transition_rope_recompute"))

    def test_producer_rejects_nonzero_lock_ref(self):
        cache = PICache(); next(iter(cache._entries.values())).lock_ref = 1
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(cache, self.target("transition_rope_recompute"))

    def test_target_absence_is_retryable_without_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); target = self.target("prefix_cache", "not-ready")
            target_path = root / "target.json"; target_path.write_text(json.dumps(target))
            cache = MambaRadixCache(); cache.root_node.children.clear(); old = dict(os.environ)
            try:
                os.environ[receipt.TARGET_ENV] = str(target_path); os.environ[receipt.OUTPUT_ENV] = str(root)
                receipt.maybe_emit_owned_state_snapshot(cache)
                self.assertFalse((root / "not-ready.json").exists())
            finally:
                os.environ.clear(); os.environ.update(old)

    def test_terminal_producer_requires_exact_duplicate_free_domain(self):
        allocator = Allocator(); allocator.free_pages = torch.tensor([1, 2, 2] + list(range(4, 16)))
        with self.assertRaises(receipt.ReceiptError): receipt._kv_free_snapshot(allocator)

    def test_target_builder_binds_only_document_segments(self):
        workload = {"workload_id": "qasper-6", "document_token_ids": [1, 2, 3, 4],
                    "document_token_sha256": receipt.token_sha256([1, 2, 3, 4]),
                    "dataset": "qasper", "source_index": 6, "query_token_ids": [5],
                    "prompt_token_sha256": "p" * 64,
                    "direct_token_ids": [1, 2, 3, 4, 5],
                    "segment_offsets": [[0, 2], [2, 4], [4, 5]]}
        target = runner._target(workload, "transition_rope_recompute", 0, {"x": "y"})
        self.assertEqual(target["segment_token_ids"], [[1, 2], [3, 4]])
        self.assertEqual(target["authority"], {"x": "y"})

    def _build_replay_bundle(self, root: Path, mode="prefix_cache"):
        cache = MambaRadixCache() if mode == "prefix_cache" else PICache()
        snapshot = self.run_snapshot(cache, self.target(mode, "cell"))
        freeze = root / "freeze"; code_dir = freeze / "code"; code_dir.mkdir(parents=True)
        code_file = code_dir / "dummy.py"; code_file.write_bytes(b"# frozen\n")
        code_sha = blind.sha256_file(code_file)
        manifest = freeze / "SHA256SUMS"
        manifest.write_text(f"{code_sha}  code/dummy.py\n")
        manifest_sha = blind.sha256_file(manifest)
        static = root / "static"; static.mkdir()
        source_path = static / "official-source-ledger.json"; source_path.write_bytes(canonical({"source": 1}))
        env_path = static / "environment-ledger.json"; env_path.write_bytes(canonical({"env": 1}))
        contract = storage_contract(); contract_path = static / "model-storage-contract.json"; contract_path.write_bytes(canonical(contract))
        overlay_diff = b"diff --git a/a b/a\n"
        (static / "instrumentation-overlay.diff").write_bytes(overlay_diff)
        overlay = {
            "schema": "hypic-rwd5-instrumentation-overlay-v2",
            "base_commit": receipt.OFFICIAL_COMMIT,
            "porcelain_v1": [" M python/sglang/srt/managers/scheduler.py", " M python/sglang/srt/mem_cache/common.py", "?? python/sglang/srt/retained_state_receipt.py"],
            "tracked_files": ["python/sglang/srt/managers/scheduler.py", "python/sglang/srt/mem_cache/common.py"],
            "untracked_files": ["python/sglang/srt/retained_state_receipt.py"],
            "canonical_diff_sha256": hashlib.sha256(overlay_diff).hexdigest(),
            "receipt_module_sha256": "r" * 64, "no_other_tracked_or_untracked": True,
        }
        write_json(static / "instrumentation-overlay.json", overlay)
        prereg = {
            "schema": "hypic-rwd5-retained-state-preregistration-v2",
            "official_commit": receipt.OFFICIAL_COMMIT,
            "official_source_ledger_sha256": blind.sha256_file(source_path),
            "environment_ledger_sha256": blind.sha256_file(env_path),
            "instrumentation": {"overlay": overlay},
            "external_freeze": {"manifest_sha256": manifest_sha},
            "code": {"dummy": {"path": "dummy.py", "bytes": code_file.stat().st_size, "sha256": code_sha}},
            "model": {"weight_ledger_raw_sha256": "w" * 64,
                      "artifact_ledger_raw_sha256": "a" * 64,
                      "config_sha256": contract["config_sha256"],
                      "storage_contract": contract,
                      "storage_contract_sha256": canonical_sha(contract)},
            "data": {"sha256": "d" * 64},
        }
        prereg_path = static / "preregistration.json"; write_json(prereg_path, prereg)
        target_path = root / "target.json"; server_path = root / "server.json"
        worker_path = root / "worker.json"
        expected_env = {
            "FORKAUDIT_RWD5_TARGET_PATH": str(target_path),
            "FORKAUDIT_RWD5_WORKER_RECEIPT_PATH": str(worker_path),
            "FORKAUDIT_RWD5_SERVER_RECEIPT_PATH": str(server_path),
            "FORKAUDIT_RWD5_PREREGISTRATION_PATH": str(prereg_path),
            "FORKAUDIT_RWD5_FREEZE_MANIFEST_PATH": str(manifest),
            "FORKAUDIT_RWD5_FREEZE_MANIFEST_SHA256": manifest_sha,
            "FORKAUDIT_RWD5_MODE": mode, "FORKAUDIT_RWD5_RANK": "0",
            "FORKAUDIT_RWD5_FRONTEND_PID": "100",
        }
        front = {"pid": 100, "ppid": 1, "cmdline": ["python", "launch_server"],
                 "cmdline_sha256": canonical_sha(["python", "launch_server"]),
                 "environment": expected_env, "environment_sha256": canonical_sha(expected_env)}
        worker_process = {"pid": 200, "ppid": 100, "cmdline": ["python", "scheduler"],
                          "cmdline_sha256": canonical_sha(["python", "scheduler"]),
                          "environment": expected_env, "environment_sha256": canonical_sha(expected_env),
                          "ancestry": [{"pid": 100, "ppid": 1, "cmdline_sha256": front["cmdline_sha256"]}],
                          "ancestry_pids": [100]}
        worker = {"schema": "forkaudit-hypic-scheduler-worker-v2",
                  "official_commit": receipt.OFFICIAL_COMMIT, "mode": mode, "rank": 0,
                  "frontend_pid": 100, "process": worker_process,
                  "tree_cache_class": "PICache" if mode != "prefix_cache" else "MambaRadixCache",
                  "picache_mamba_pool_identity": True if mode != "prefix_cache" else None,
                  "int8_mamba_checkpoint_enabled": False,
                  "preregistration_sha256": blind.sha256_file(prereg_path),
                  "freeze_manifest_sha256": manifest_sha,
                  "storage_contract_sha256": canonical_sha(contract)}
        write_json(worker_path, worker)
        config = {"expected": {}, "observed": {},
                  "rwd5_expected": {"enable_int8_mamba_checkpoint": False, "page_size": 1},
                  "rwd5_observed": {"enable_int8_mamba_checkpoint": False, "page_size": 1}}
        server_authority = {
            "official_commit": receipt.OFFICIAL_COMMIT,
            "preregistration_sha256": blind.sha256_file(prereg_path),
            "freeze_manifest_sha256": manifest_sha,
            "official_source_ledger_sha256": prereg["official_source_ledger_sha256"],
            "environment_ledger_sha256": prereg["environment_ledger_sha256"],
            "data_sha256": prereg["data"]["sha256"],
            "model_weight_ledger_sha256": prereg["model"]["weight_ledger_raw_sha256"],
            "model_artifact_ledger_sha256": prereg["model"]["artifact_ledger_raw_sha256"],
            "model_config_sha256": prereg["model"]["config_sha256"],
            "storage_contract_sha256": prereg["model"]["storage_contract_sha256"],
            "overlay_diff_sha256": overlay["canonical_diff_sha256"],
            "code_sha256": {"dummy": code_sha},
        }
        server = {"schema": "hypic-rwd5-server-launch-receipt-v2",
                  "official_commit": receipt.OFFICIAL_COMMIT, "mode": mode, "rank": 0,
                  "server_configuration": config,
                  "server_configuration_sha256": canonical_sha(config),
                  "frontend_process": front, "server_process": front,
                  "scheduler_worker": {"receipt_sha256": blind.sha256_file(worker_path), "identity": worker},
                  "authority": server_authority,
                  "instrumented_overlay": {"porcelain_v1": overlay["porcelain_v1"],
                                           "canonical_diff_sha256": overlay["canonical_diff_sha256"]}}
        write_json(server_path, server)
        target_authority = dict(server_authority)
        target_authority.update({"server_launch_receipt_sha256": blind.sha256_file(server_path),
                                 "scheduler_worker_receipt_sha256": blind.sha256_file(worker_path),
                                 "server_configuration_sha256": server["server_configuration_sha256"]})
        target = self.target(mode, "cell", target_authority)
        write_json(target_path, target)
        bindings = dict(target_authority); bindings["target_sha256"] = blind.sha256_file(target_path)
        snapshot["authority"] = {"bindings": bindings, "scheduler_process": worker_process}
        snapshot["storage_contract"] = contract
        receipt_path = root / "receipt.json"; write_json(receipt_path, snapshot)
        kv_domain = list(range(1, 16)); mamba_domain = list(range(1, 6))
        terminal = {"schema": "forkaudit-hypic-retained-state-terminal-v2",
                    "official_commit": receipt.OFFICIAL_COMMIT, "passed": True,
                    "authority": snapshot["authority"],
                    "prior_receipt_sha256": blind.sha256_file(receipt_path),
                    "checks": {"target_entries_after": 0, "all_cache_entries_after": 0,
                               "old_kv_slots_all_free": True, "old_mamba_slots_all_free": True,
                               "kv_available_tokens": 15, "kv_capacity_tokens": 15,
                               "mamba_available_slots": 5, "mamba_capacity_slots": 5,
                               "kv_free_list": {"page_size": 1, "size": 15,
                                                "free_pages": kv_domain, "release_pages": [], "exact_domain": kv_domain},
                               "mamba_free_list": {"size": 5, "free_slots": mamba_domain,
                                                    "exact_domain": mamba_domain}}}
        terminal_path = root / "terminal.json"; write_json(terminal_path, terminal)
        raw = {"schema": "forkaudit-hypic-retained-state-shard-v2", "status": "completed",
               "official_commit": receipt.OFFICIAL_COMMIT, "mode": mode, "rank": 0,
               "authority": snapshot["authority"], "target": target,
               "target_sha256": blind.sha256_file(target_path),
               "server_launch_receipt_sha256": blind.sha256_file(server_path),
               "preregistration_sha256": blind.sha256_file(prereg_path),
               "freeze_manifest_sha256": manifest_sha,
               "workload": target["workload_binding"],
               "cache_observation": {"cached_tokens": snapshot["selection"]["expected_measured_cached_tokens"]},
               "store_receipt": {"sha256": blind.sha256_file(receipt_path)},
               "terminal_receipt": {"sha256": blind.sha256_file(terminal_path)}}
        raw_path = root / "raw.json"; write_json(raw_path, raw)
        return {"receipt": receipt_path, "terminal": terminal_path, "raw": raw_path,
                "target": target_path, "server": server_path, "worker": worker_path,
                "prereg": prereg_path, "manifest": manifest, "manifest_sha": manifest_sha,
                "static": static}

    def replay(self, paths):
        return blind.replay_one(paths["receipt"], paths["terminal"], paths["raw"],
            target_path=paths["target"], server_path=paths["server"],
            worker_path=paths["worker"], prereg_path=paths["prereg"],
            manifest_path=paths["manifest"], expected_manifest_sha256=paths["manifest_sha"],
            static_dir=paths["static"])

    def rewrite_receipt_chain(self, paths, mutator):
        value = json.loads(paths["receipt"].read_text()); mutator(value); write_json(paths["receipt"], value)
        terminal = json.loads(paths["terminal"].read_text())
        terminal["prior_receipt_sha256"] = blind.sha256_file(paths["receipt"]); write_json(paths["terminal"], terminal)
        raw = json.loads(paths["raw"].read_text())
        raw["store_receipt"]["sha256"] = blind.sha256_file(paths["receipt"])
        raw["terminal_receipt"]["sha256"] = blind.sha256_file(paths["terminal"]); write_json(paths["raw"], raw)

    def test_blind_replay_accepts_complete_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.replay(self._build_replay_bundle(Path(temporary)))
            self.assertGreater(result["payload_bytes"], 0)

    def test_blind_rejects_equal_width_shifted_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary))
            def mutate(value):
                row = value["tensor_payload"]["records"][0]
                row["byte_start"] += 24; row["byte_end"] += 24
                row["absolute_byte_start"] += 24; row["absolute_byte_end"] += 24
                value["tensor_payload"]["union"] = receipt._union_summary(value["tensor_payload"]["records"])
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_forged_shape_stride_range(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary))
            def mutate(value):
                row = value["tensor_payload"]["records"][0]
                row["stride"][0] += 1
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_missing_temporal_after_retotal(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary))
            def mutate(value):
                value["tensor_payload"]["records"] = [r for r in value["tensor_payload"]["records"] if r["tensor_name"] != "mamba.temporal"]
                value["component_presence"].pop("temporal")
                value["tensor_payload"]["union"] = receipt._union_summary(value["tensor_payload"]["records"])
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_missing_recurrent_layer_slot_coverage(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary))
            def mutate(value):
                rows = value["tensor_payload"]["records"]
                removed = False
                kept = []
                for row in rows:
                    if (not removed and row["tensor_name"] == "mamba.conv[0]"
                            and row["selection"]["mamba_layer_index"] == 1):
                        removed = True
                        continue
                    kept.append(row)
                value["tensor_payload"]["records"] = kept
                value["tensor_payload"]["union"] = receipt._union_summary(kept)
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_missing_hypic_transition_after_retotal(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "transition_rope_recompute")
            def mutate(value):
                value["tensor_payload"]["records"] = [r for r in value["tensor_payload"]["records"] if r["tensor_name"] != "mamba.transition"]
                value["component_presence"].pop("transition")
                value["tensor_payload"]["union"] = receipt._union_summary(value["tensor_payload"]["records"])
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_identity_field_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary))
            def mutate(value): value["authority"]["scheduler_process"].pop("ppid")
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_raw_binding_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary))
            raw = json.loads(paths["raw"].read_text()); raw.pop("preregistration_sha256"); write_json(paths["raw"], raw)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_duplicate_replacing_missing_free_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary))
            terminal = json.loads(paths["terminal"].read_text())
            terminal["checks"]["kv_free_list"]["free_pages"][-1] = 14
            write_json(paths["terminal"], terminal)
            raw = json.loads(paths["raw"].read_text()); raw["terminal_receipt"]["sha256"] = blind.sha256_file(paths["terminal"]); write_json(paths["raw"], raw)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_unrelated_overlay_diff(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary))
            (paths["static"] / "instrumentation-overlay.diff").write_bytes(b"unrelated diff\n")
            with self.assertRaises(Exception): self.replay(paths)

    def test_model_contract_derived_from_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / "config.json", {"text_config": {"model_type": "qwen3_5_moe_text", "num_hidden_layers": 4, "full_attention_interval": 2}})
            contract = static_builder.model_storage_contract(root)
            self.assertEqual(contract["full_attention_layer_ids"], [1, 3])
            self.assertEqual(contract["recurrent_layer_ids"], [0, 2])

    def test_launcher_has_only_affected_arms_and_external_manifest_gate(self):
        launcher = Path(__file__).with_name("launch_hypic_retained_state_bytes_8gpu.sh").read_text()
        self.assertIn("run_mode prefix_cache", launcher)
        self.assertIn("run_mode transition_rope_recompute", launcher)
        self.assertNotIn("run_mode full_recompute", launcher)
        self.assertIn("EXPECTED_FREEZE_MANIFEST_SHA256", launcher)
        self.assertIn("sha256sum -c", launcher)


if __name__ == "__main__": unittest.main()
