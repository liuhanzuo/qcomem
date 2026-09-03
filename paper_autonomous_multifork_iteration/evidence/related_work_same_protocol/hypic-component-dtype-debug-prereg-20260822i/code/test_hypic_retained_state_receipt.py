import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
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


def storage_contract(
    *,
    kv_dtype="torch.bfloat16",
    conv_dtype="torch.bfloat16",
    temporal_dtype="torch.float32",
    transition_dtype="torch.float32",
    conv_tails_dtype="torch.bfloat16",
):
    value = {
        "schema": "hypic-rwd5-model-storage-contract-v3",
        "config_sha256": "c" * 64,
        "model_type": "qwen3_5_moe_text",
        "num_hidden_layers": 3,
        "full_attention_layer_ids": [2],
        "recurrent_layer_ids": [0, 1],
        "full_attention_layer_count": 1,
        "recurrent_layer_count": 2,
        "conv_tensor_count": 1,
        "temporal_tensor_count": 1,
        "kv_dtype": kv_dtype,
        "mamba_component_dtypes": {
            "conv": conv_dtype,
            "temporal": temporal_dtype,
            "transition": transition_dtype,
            "conv_tails": conv_tails_dtype,
        },
        "dtype_authority": {
            "runtime_environment": {
                "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
                "SGLANG_MAMBA_SSM_DTYPE": "float32",
            },
            "official_pool_rule": "conv and conv_tails allocate cache_params.dtype.conv; temporal and transition allocate cache_params.dtype.temporal",
            "legacy_unified_dtype_forbidden": True,
        },
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
        self.k_buffer = [torch.zeros((16, 2, 3), dtype=torch.bfloat16)]
        value_shape = (16, 2, 4) if mismatched else (16, 2, 3)
        self.v_buffer = [torch.zeros(value_shape, dtype=torch.bfloat16)]


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
        self.size = 5; self.free_slots = torch.tensor([1, 4, 5], dtype=torch.int64)
    def available_size(self): return len(self.free_slots)


class MambaCache:
    def __init__(
        self,
        pic,
        *,
        missing_transition=False,
        missing_tails=False,
        conv_dtype=torch.bfloat16,
        temporal_dtype=torch.float32,
        transition_dtype=torch.float32,
        conv_tails_dtype=torch.bfloat16,
    ):
        self.conv = [torch.zeros((2, 6, 3, 2), dtype=conv_dtype)]
        self.temporal = torch.zeros((2, 6, 2, 2, 2), dtype=temporal_dtype)
        self.transition = (
            None if not pic or missing_transition
            else torch.zeros((2, 6, 2, 2, 2), dtype=transition_dtype)
        )
        self.conv_tails = (
            None if not pic or missing_tails
            else [torch.zeros((2, 6, 3, 2), dtype=conv_tails_dtype)]
        )


class ReqPool:
    def __init__(self, pic=False, **kwargs):
        self.mamba_allocator = MambaAllocator()
        self.mamba_pool = type("Pool", (), {"mamba_cache": MambaCache(pic, **kwargs)})()
        self.mamba_ckpt_pool = None


class MambaRadixCache:
    def __init__(self, *, mismatched=False, **kwargs):
        self.root_node = Node(0)
        child = Node(1, [1, 2, 3], [1, 2, 3], [2]); child.parent = self.root_node
        self.root_node.children[1] = child
        self.token_to_kv_pool_allocator = Allocator(mismatched)
        self.req_to_token_pool = ReqPool(pic=False, **kwargs)
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
            "seam_tokens": 1 if mode == "transition_rope_recompute" else 0,
            "authority": {} if authority is None else authority,
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
        dtypes = {row["tensor_name"]: row["dtype"] for row in output["tensor_payload"]["records"]}
        self.assertEqual(dtypes["mamba.conv[0]"], "torch.bfloat16")
        self.assertEqual(dtypes["mamba.temporal"], "torch.float32")
        self.assertEqual(output["tensor_payload"]["union"]["unique_overlap_aware_bytes"], 160)

    def test_hypic_receipt_requires_all_pic_components(self):
        output = self.run_snapshot(PICache(), self.target("transition_rope_recompute"))
        names = {row["tensor_name"] for row in output["tensor_payload"]["records"]}
        self.assertIn("mamba.transition", names); self.assertIn("mamba.conv_tails[0]", names)
        self.assertTrue(all(entry["lock_ref"] == 0 for entry in output["selection"]["entries"]))
        dtypes = {row["tensor_name"]: row["dtype"] for row in output["tensor_payload"]["records"]}
        self.assertEqual(dtypes["mamba.conv[0]"], "torch.bfloat16")
        self.assertEqual(dtypes["mamba.temporal"], "torch.float32")
        self.assertEqual(dtypes["mamba.transition"], "torch.float32")
        self.assertEqual(dtypes["mamba.conv_tails[0]"], "torch.bfloat16")
        self.assertEqual(output["tensor_payload"]["union"]["unique_overlap_aware_bytes"], 448)

    def test_read_only_component_dtype_debug_inventory_is_mixed_and_nonformal(self):
        prefix = receipt._component_dtype_debug_inventory(MambaRadixCache(), "prefix_cache")
        hypic = receipt._component_dtype_debug_inventory(PICache(), "transition_rope_recompute")
        self.assertEqual(prefix["status"], "debug_only_not_formal_evidence")
        self.assertFalse(prefix["formal_receipt_emitted"])
        self.assertEqual(prefix["components"]["conv[0]"]["dtype"], "torch.bfloat16")
        self.assertEqual(prefix["components"]["temporal"]["dtype"], "torch.float32")
        self.assertEqual(hypic["components"]["transition"]["dtype"], "torch.float32")
        self.assertEqual(hypic["components"]["conv_tails[0]"]["dtype"], "torch.bfloat16")

    @staticmethod
    def _dtype_debug_model(root: Path) -> Path:
        model = root / "model"
        model.mkdir()
        write_json(model / "config.json", {
            "text_config": {
                "model_type": "qwen3_5_moe_text",
                "num_hidden_layers": 3,
                "full_attention_interval": 3,
            }
        })
        return model

    @staticmethod
    def _valid_dtype_debug_payload(mode: str) -> dict:
        cache = MambaRadixCache() if mode == "prefix_cache" else PICache()
        old = dict(os.environ)
        try:
            os.environ["SGLANG_MAMBA_CONV_DTYPE"] = "bfloat16"
            os.environ["SGLANG_MAMBA_SSM_DTYPE"] = "float32"
            value = receipt._component_dtype_debug_inventory(cache, mode)
        finally:
            os.environ.clear(); os.environ.update(old)
        # The unit fake deliberately has a generic pool class and CPU tensors;
        # normalize only those live-runtime identities to the frozen GPU cell.
        value["mamba_pool_class"] = "MambaPool"
        for row in value["components"].values():
            row["device"] = "cuda:0"
        return value

    def test_dtype_debug_validator_accepts_exact_prefix_and_hypic_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._dtype_debug_model(Path(temporary))
            for mode in ("prefix_cache", "transition_rope_recompute"):
                value = runner.validate_dtype_debug_receipt(
                    self._valid_dtype_debug_payload(mode), mode=mode, model=model
                )
                self.assertEqual(value["status"], "passed_exact_live_component_contract")
                self.assertFalse(value["paper_evidence"])

    def test_dtype_debug_producer_canonical_serialization_round_trips_to_validator(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self._dtype_debug_model(root)
            for mode in ("prefix_cache", "transition_rope_recompute"):
                output = root / f"{mode}.json"
                receipt._atomic_json(output, self._valid_dtype_debug_payload(mode))
                reloaded = json.loads(output.read_text())
                # Producer canonicalization sorts nested component keys.  The
                # consumer must require the exact set without inventing an
                # ordering condition that JSON does not preserve semantically.
                self.assertEqual(list(reloaded["components"]), sorted(reloaded["components"]))
                validation = runner.validate_dtype_debug_receipt(
                    reloaded, mode=mode, model=model
                )
                self.assertEqual(
                    validation["status"], "passed_exact_live_component_contract"
                )

    def test_dtype_debug_validator_rejects_wrong_mixed_dtype_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._dtype_debug_model(Path(temporary))
            value = self._valid_dtype_debug_payload("prefix_cache")
            value["components"]["conv[0]"]["dtype"] = "torch.float32"
            value["components"]["conv[0]"]["element_size"] = 4
            value["components"]["temporal"]["dtype"] = "torch.bfloat16"
            value["components"]["temporal"]["element_size"] = 2
            with self.assertRaises(Exception):
                runner.validate_dtype_debug_receipt(value, mode="prefix_cache", model=model)

    def test_dtype_debug_validator_rejects_missing_and_extra_components(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._dtype_debug_model(Path(temporary))
            missing = self._valid_dtype_debug_payload("transition_rope_recompute")
            missing["components"].pop("transition")
            with self.assertRaises(Exception):
                runner.validate_dtype_debug_receipt(
                    missing, mode="transition_rope_recompute", model=model
                )
            extra = self._valid_dtype_debug_payload("prefix_cache")
            extra["components"]["transition"] = copy.deepcopy(extra["components"]["temporal"])
            with self.assertRaises(Exception):
                runner.validate_dtype_debug_receipt(extra, mode="prefix_cache", model=model)

    def test_dtype_debug_validator_rejects_layout_identity_and_environment_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = self._dtype_debug_model(Path(temporary))
            mutations = []
            mutations.append(lambda value: value["components"]["temporal"]["stride"].__setitem__(-1, 2))
            mutations.append(lambda value: value["components"]["temporal"]["shape"].__setitem__(0, 3))
            mutations.append(lambda value: value.__setitem__("tree_cache_class", "PICache"))
            mutations.append(lambda value: value.__setitem__("official_commit", "0" * 40))
            mutations.append(lambda value: value["runtime_environment"].__setitem__(
                "SGLANG_MAMBA_SSM_DTYPE", "bfloat16"
            ))
            for mutate in mutations:
                value = self._valid_dtype_debug_payload("prefix_cache")
                mutate(value)
                with self.assertRaises(Exception):
                    runner.validate_dtype_debug_receipt(value, mode="prefix_cache", model=model)

    def test_dtype_debug_validation_stage_writes_only_after_exact_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = self._dtype_debug_model(root)
            source = root / "debug.json"
            output = root / "validation.json"
            write_json(source, self._valid_dtype_debug_payload("prefix_cache"))
            args = type("Args", (), {
                "dtype_debug_receipt": source,
                "mode": "prefix_cache",
                "model": model,
                "output": output,
            })()
            runner.dtype_debug_validate_stage(args)
            self.assertEqual(
                json.loads(output.read_text())["status"],
                "passed_exact_live_component_contract",
            )
            output.unlink()
            def wrong_dtype(value):
                value["components"]["conv[0]"]["dtype"] = "torch.float32"
                value["components"]["conv[0]"]["element_size"] = 4
            def missing_component(value):
                value["components"].pop("temporal")
            def extra_component(value):
                value["components"]["transition"] = copy.deepcopy(
                    value["components"]["temporal"]
                )
            for mutate in (wrong_dtype, missing_component, extra_component):
                invalid = self._valid_dtype_debug_payload("prefix_cache")
                mutate(invalid)
                write_json(source, invalid)
                with self.assertRaises(Exception):
                    runner.dtype_debug_validate_stage(args)
                # This is the exact launcher gate: no validation receipt means
                # the shell cannot reach COMPLETED_DEBUG_ONLY.
                self.assertFalse(output.exists())

    def test_dtype_debug_launcher_gates_completion_on_independent_validation(self):
        launcher = Path(__file__).with_name(
            "launch_hypic_component_dtype_debug_1gpu.sh"
        ).read_text()
        self.assertEqual(launcher.count("--stage dtype_debug_validate"), 1)
        self.assertIn("${mode}-validation.json", launcher)
        self.assertLess(
            launcher.index("--stage dtype_debug_validate"),
            launcher.index('touch "$OUTPUT_ROOT/COMPLETED_DEBUG_ONLY"'),
        )
        self.assertIn("passed_exact_live_component_contract", launcher)
        self.assertIn("formal receipt emitted in debug run", launcher)

    def test_producer_rejects_wrong_temporal_dtype(self):
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(
                MambaRadixCache(temporal_dtype=torch.bfloat16),
                self.target("prefix_cache", "wrong-temporal-dtype"),
            )

    def test_producer_rejects_wrong_transition_dtype(self):
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(
                PICache(transition_dtype=torch.bfloat16),
                self.target("transition_rope_recompute", "wrong-transition-dtype"),
            )

    def test_producer_rejects_wrong_conv_tails_dtype(self):
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(
                PICache(conv_tails_dtype=torch.float32),
                self.target("transition_rope_recompute", "wrong-conv-tails-dtype"),
            )

    def test_storage_contract_rejects_retired_unified_dtype_field(self):
        contract = storage_contract()
        contract["dtype"] = "torch.bfloat16"
        prereg = {
            "model": {
                "storage_contract": contract,
                "storage_contract_sha256": canonical_sha(contract),
            }
        }
        with self.assertRaises(receipt.ReceiptError):
            receipt._storage_contract(prereg, "prefix_cache")

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

    def test_prefix_snapshot_rejects_selected_free_kv_slot(self):
        cache = MambaRadixCache()
        cache.token_to_kv_pool_allocator.free_pages = torch.tensor(
            [1] + list(range(5, 16)), dtype=torch.int64
        )
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(cache, self.target("prefix_cache", "prefix-free-kv"))

    def test_prefix_snapshot_rejects_selected_free_mamba_slot(self):
        cache = MambaRadixCache()
        cache.req_to_token_pool.mamba_allocator.free_slots = torch.tensor(
            [1, 2, 4, 5], dtype=torch.int64
        )
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(cache, self.target("prefix_cache", "prefix-free-mamba"))

    def test_hypic_snapshot_rejects_segment_mamba_slot_already_free(self):
        cache = PICache()
        cache.req_to_token_pool.mamba_allocator.free_slots = torch.tensor(
            [1, 3, 4, 5], dtype=torch.int64
        )
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(cache, self.target("transition_rope_recompute", "hypic-free-mamba"))

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

    def test_real_process_receipt_shape_hashes_frontend_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            fake_proc = Path(temporary) / "proc"
            process_root = fake_proc / "123"
            process_root.mkdir(parents=True)
            cmdline = ["python", "-m", "sglang.launch_server"]
            (process_root / "cmdline").write_bytes(b"\0".join(value.encode() for value in cmdline) + b"\0")
            environment = {
                "CUDA_VISIBLE_DEVICES": "GPU-frozen",
                "PYTHONPATH": "/instrumented/python:/frozen/code",
                "PIC_SEAM_SINK": "8",
                "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
                "SGLANG_MAMBA_SSM_DTYPE": "float32",
                "FORKAUDIT_RWD5_TARGET_PATH": "/run/target.json",
                "FORKAUDIT_RWD5_RECEIPT_DIR": "/run/receipts",
                "FORKAUDIT_RWD5_MODE": "prefix_cache",
                "FORKAUDIT_RWD5_RANK": "0",
            }
            (process_root / "environ").write_bytes(
                b"\0".join(f"{key}={value}".encode() for key, value in environment.items()) + b"\0"
            )
            (process_root / "stat").write_text("123 (python) S 77 0 0 0 0\n")
            real_path = Path
            with mock.patch.object(
                runner,
                "Path",
                side_effect=lambda value: fake_proc if str(value) == "/proc" else real_path(value),
            ):
                value = runner._process_receipt(
                    123,
                    "/instrumented/python:/frozen/code",
                    "GPU-frozen",
                )
            self.assertEqual(
                set(value),
                {"pid", "ppid", "cmdline", "cmdline_sha256", "environment", "environment_sha256"},
            )
            self.assertEqual(value["pid"], 123)
            self.assertEqual(value["ppid"], 77)
            self.assertEqual(value["cmdline"], cmdline)
            self.assertEqual(value["cmdline_sha256"], runner.rwd5_canonical_json_sha256(cmdline))
            self.assertEqual(value["environment"], environment)
            self.assertEqual(value["environment_sha256"], runner.rwd5_canonical_json_sha256(environment))

    def test_server_info_readiness_survives_more_than_legacy_thirty_seconds(self):
        with tempfile.TemporaryDirectory() as temporary:
            clock = {"seconds": 0.0}
            calls = {"count": 0}

            def monotonic():
                return clock["seconds"]

            def fetch(url, timeout):
                self.assertEqual(url, "http://127.0.0.1:33400/server_info")
                self.assertLessEqual(timeout, 3.0)
                calls["count"] += 1
                clock["seconds"] += 1.0
                if clock["seconds"] <= 35.0:
                    raise TimeoutError("scheduler internal warmup still active")
                return json.dumps({"model_path": "/frozen/model", "page_size": 1})

            def sleeper(seconds):
                clock["seconds"] += seconds

            output = Path(temporary) / "server-info-readiness.json"
            result = runner.wait_for_server_info(
                "http://127.0.0.1:33400",
                output,
                mode="prefix_cache",
                rank=0,
                server_pid=123,
                total_timeout=90.0,
                single_timeout=3.0,
                poll_interval=0.0,
                fetch=fetch,
                monotonic=monotonic,
                sleeper=sleeper,
            )
            self.assertEqual(result["status"], "ready")
            self.assertGreater(result["elapsed_seconds"], 30.0)
            self.assertEqual(result["attempt_count"], 36)
            self.assertEqual(sum(row["outcome"] == "not_ready" for row in result["attempts"]), 35)
            self.assertEqual(json.loads(output.read_text()), result)

    def test_exit_trap_reaps_mock_servers_and_marks_failure(self):
        launcher_path = Path(__file__).with_name("launch_hypic_retained_state_bytes_8gpu.sh")
        launcher = launcher_path.read_text()
        function_names = (
            "rwd5_cleanup_servers",
            "rwd5_on_exit",
            "rwd5_install_traps",
        )
        functions = []
        for name in function_names:
            match = re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}\n", launcher)
            self.assertIsNotNone(match, name)
            functions.append(match.group(0))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "server-logs"
            logs.mkdir()
            completed = root / "COMPLETED"
            completed.write_text("stale terminal marker\n")
            servers = [
                subprocess.Popen(["sleep", "300"], start_new_session=True),
                subprocess.Popen(["sleep", "300"], start_new_session=True),
            ]
            try:
                for index, server in enumerate(servers):
                    (logs / f"mock-{index}.pid").write_text(f"{server.pid}\n")
                script = "\n".join([
                    "set -Eeuo pipefail",
                    *functions,
                    f"RUN_DIR={str(root)!r}",
                    "RWD5_RUN_SUCCEEDED=0",
                    "RWD5_LAST_ERROR=0",
                    f"SERVER_PIDS=({' '.join(str(server.pid) for server in servers)})",
                    "rwd5_install_traps",
                    "mock_server_receipt_stage() { return 37; }",
                    "mock_server_receipt_stage",
                ])
                result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=20)
                self.assertEqual(result.returncode, 37, result.stderr)
                self.assertTrue((root / "FAILED").is_file())
                self.assertEqual((root / "FAILED").read_text().strip(), "37")
                self.assertFalse(completed.exists())
                for server in servers:
                    server.wait(timeout=3)
                    self.assertIsNotNone(server.returncode)
            finally:
                for server in servers:
                    if server.poll() is None:
                        os.killpg(server.pid, 9)
                        server.wait(timeout=3)

    def test_exit_trap_escalates_for_sigterm_ignoring_process_group(self):
        launcher = Path(__file__).with_name("launch_hypic_retained_state_bytes_8gpu.sh").read_text()
        functions = []
        for name in ("rwd5_cleanup_servers", "rwd5_on_exit", "rwd5_install_traps"):
            match = re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}\n", launcher)
            self.assertIsNotNone(match, name)
            functions.append(match.group(0))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "server-logs"
            logs.mkdir()
            (root / "COMPLETED").write_text("must be removed\n")
            server = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); print('ready', flush=True); time.sleep(300)",
                ],
                start_new_session=True,
                stdout=subprocess.PIPE,
                text=True,
            )
            try:
                self.assertEqual(server.stdout.readline().strip(), "ready")
                server.stdout.close()
                (logs / "stubborn.pid").write_text(f"{server.pid}\n")
                script = "\n".join([
                    "set -Eeuo pipefail",
                    *functions,
                    f"RUN_DIR={str(root)!r}",
                    "RWD5_RUN_SUCCEEDED=0",
                    "RWD5_LAST_ERROR=0",
                    f"SERVER_PIDS=({server.pid})",
                    "rwd5_install_traps",
                    "mock_server_receipt_stage() { return 37; }",
                    "mock_server_receipt_stage",
                ])
                started = time.monotonic()
                result = subprocess.run(
                    ["bash", "-c", script], text=True, capture_output=True, timeout=18
                )
                elapsed = time.monotonic() - started
                self.assertEqual(result.returncode, 37, result.stderr)
                self.assertLess(elapsed, 15.0)
                self.assertEqual((root / "FAILED").read_text().strip(), "37")
                self.assertFalse((root / "COMPLETED").exists())
                server.wait(timeout=3)
                self.assertIsNotNone(server.returncode)
            finally:
                if server.poll() is None:
                    os.killpg(server.pid, 9)
                    server.wait(timeout=3)

    def _build_replay_bundle(self, root: Path, mode="prefix_cache"):
        cache = MambaRadixCache() if mode == "prefix_cache" else PICache()
        snapshot_id = f"{mode}-rank-0"
        snapshot = self.run_snapshot(cache, self.target(mode, snapshot_id))
        freeze = root / "freeze"; code_dir = freeze / "code"; code_dir.mkdir(parents=True)
        code_file = code_dir / "dummy.py"; code_file.write_bytes(b"# frozen\n")
        code_sha = blind.sha256_file(code_file)
        manifest = freeze / "SHA256SUMS"
        # Match the actual frozen SHA256SUMS spelling.  Replay must canonicalize
        # this to the authority key ``code/dummy.py`` without weakening path
        # confinement.
        manifest.write_text(f"{code_sha}  ./code/dummy.py\n")
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
            "design": {"hypic_seam_tokens": 1},
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
            "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
            "SGLANG_MAMBA_SSM_DTYPE": "float32",
        }
        front = {"pid": 100, "ppid": 1, "cmdline": ["python", "launch_server"],
                 "cmdline_sha256": runner.rwd5_canonical_json_sha256(["python", "launch_server"]),
                 "environment": expected_env,
                 "environment_sha256": runner.rwd5_canonical_json_sha256(expected_env)}
        worker_process = {"pid": 200, "ppid": 100, "cmdline": ["python", "scheduler"],
                          "cmdline_sha256": runner.rwd5_canonical_json_sha256(["python", "scheduler"]),
                          "environment": expected_env,
                          "environment_sha256": runner.rwd5_canonical_json_sha256(expected_env),
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
        server_info_sha = "i" * 64
        readiness = {
            "schema": "hypic-rwd5-server-info-readiness-v1",
            "status": "ready",
            "endpoint": "http://127.0.0.1:33400/server_info",
            "mode": mode,
            "rank": 0,
            "server_pid": 100,
            "total_timeout_seconds": 300.0,
            "single_timeout_seconds": 3.0,
            "poll_interval_seconds": 1.0,
            "attempt_count": 2,
            "elapsed_seconds": 46.0,
            "attempts": [
                {"attempt": 1, "outcome": "not_ready"},
                {"attempt": 2, "outcome": "ready", "response_sha256": server_info_sha},
            ],
            "server_info_sha256": server_info_sha,
        }
        server = {"schema": "hypic-rwd5-server-launch-receipt-v2",
                  "official_commit": receipt.OFFICIAL_COMMIT, "mode": mode, "rank": 0,
                  "base_url": "http://127.0.0.1:33400",
                  "server_info_endpoint": "http://127.0.0.1:33400/server_info",
                  "server_info_sha256": server_info_sha,
                  "server_info_readiness": {"sha256": canonical_sha(readiness), "identity": readiness},
                  "server_configuration": config,
                  "server_configuration_sha256": runner.rwd5_canonical_json_sha256(config),
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
        target = self.target(mode, snapshot_id, target_authority)
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
                               "old_kv_slots_preallocated": True, "old_mamba_slots_preallocated": True,
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
                "static": static, "expected_mode": mode, "expected_rank": 0,
                "expected_snapshot_id": snapshot_id, "expected_workload_id": "qasper-6"}

    def replay(self, paths):
        return blind.replay_one(paths["receipt"], paths["terminal"], paths["raw"],
            target_path=paths["target"], server_path=paths["server"],
            worker_path=paths["worker"], prereg_path=paths["prereg"],
            manifest_path=paths["manifest"], expected_manifest_sha256=paths["manifest_sha"],
            static_dir=paths["static"], expected_mode=paths["expected_mode"],
            expected_rank=paths["expected_rank"],
            expected_snapshot_id=paths["expected_snapshot_id"],
            expected_workload_id=paths["expected_workload_id"])

    def rewrite_receipt_chain(self, paths, mutator):
        value = json.loads(paths["receipt"].read_text()); mutator(value); write_json(paths["receipt"], value)
        terminal = json.loads(paths["terminal"].read_text())
        terminal["prior_receipt_sha256"] = blind.sha256_file(paths["receipt"]); write_json(paths["terminal"], terminal)
        raw = json.loads(paths["raw"].read_text())
        raw["store_receipt"]["sha256"] = blind.sha256_file(paths["receipt"])
        raw["terminal_receipt"]["sha256"] = blind.sha256_file(paths["terminal"]); write_json(paths["raw"], raw)

    def rewrite_server_chain(self, paths, mutator):
        """Re-sign every downstream test fixture after a forged server receipt.

        This deliberately gives a tamper test the strongest possible producer-side
        attacker: all file hashes remain internally self-consistent, so replay must
        reject the forged semantic cell rather than merely noticing a stale digest.
        """
        server = json.loads(paths["server"].read_text())
        mutator(server)
        write_json(paths["server"], server)
        server_sha = blind.sha256_file(paths["server"])

        target = json.loads(paths["target"].read_text())
        target["authority"]["server_launch_receipt_sha256"] = server_sha
        write_json(paths["target"], target)
        target_sha = blind.sha256_file(paths["target"])

        value = json.loads(paths["receipt"].read_text())
        value["authority"]["bindings"]["server_launch_receipt_sha256"] = server_sha
        value["authority"]["bindings"]["target_sha256"] = target_sha
        write_json(paths["receipt"], value)

        terminal = json.loads(paths["terminal"].read_text())
        terminal["authority"] = value["authority"]
        terminal["prior_receipt_sha256"] = blind.sha256_file(paths["receipt"])
        write_json(paths["terminal"], terminal)

        raw = json.loads(paths["raw"].read_text())
        raw["authority"] = value["authority"]
        raw["target"] = target
        raw["target_sha256"] = target_sha
        raw["server_launch_receipt_sha256"] = server_sha
        raw["store_receipt"]["sha256"] = blind.sha256_file(paths["receipt"])
        raw["terminal_receipt"]["sha256"] = blind.sha256_file(paths["terminal"])
        write_json(paths["raw"], raw)

    def resign_entire_bundle_as_rank_one(self, paths):
        """Forge a complete internally valid rank-1 cell in rank-0 file slots."""
        worker = json.loads(paths["worker"].read_text())
        worker["rank"] = 1
        worker["process"]["environment"]["FORKAUDIT_RWD5_RANK"] = "1"
        worker["process"]["environment_sha256"] = runner.rwd5_canonical_json_sha256(
            worker["process"]["environment"]
        )
        write_json(paths["worker"], worker)
        worker_sha = blind.sha256_file(paths["worker"])

        server = json.loads(paths["server"].read_text())
        server["rank"] = 1
        server["base_url"] = "http://127.0.0.1:33401"
        server["server_info_endpoint"] = "http://127.0.0.1:33401/server_info"
        readiness = server["server_info_readiness"]["identity"]
        readiness["rank"] = 1
        readiness["endpoint"] = server["server_info_endpoint"]
        server["server_info_readiness"]["sha256"] = canonical_sha(readiness)
        for process_key in ("frontend_process", "server_process"):
            process = server[process_key]
            process["environment"]["FORKAUDIT_RWD5_RANK"] = "1"
            process["environment_sha256"] = runner.rwd5_canonical_json_sha256(
                process["environment"]
            )
        server["scheduler_worker"] = {"receipt_sha256": worker_sha, "identity": worker}
        write_json(paths["server"], server)
        server_sha = blind.sha256_file(paths["server"])

        target = json.loads(paths["target"].read_text())
        target["rank"] = 1
        target["snapshot_id"] = "prefix_cache-rank-1"
        target["workload_id"] = "qasper-7"
        target["workload_binding"]["workload_id"] = "qasper-7"
        target["workload_binding"]["source_index"] = 7
        target["authority"]["server_launch_receipt_sha256"] = server_sha
        target["authority"]["scheduler_worker_receipt_sha256"] = worker_sha
        write_json(paths["target"], target)
        target_sha = blind.sha256_file(paths["target"])

        value = json.loads(paths["receipt"].read_text())
        value["target"]["rank"] = 1
        value["target"]["workload_id"] = "qasper-7"
        value["authority"]["bindings"]["target_sha256"] = target_sha
        value["authority"]["bindings"]["server_launch_receipt_sha256"] = server_sha
        value["authority"]["bindings"]["scheduler_worker_receipt_sha256"] = worker_sha
        value["authority"]["scheduler_process"] = worker["process"]
        write_json(paths["receipt"], value)

        terminal = json.loads(paths["terminal"].read_text())
        terminal["authority"] = value["authority"]
        terminal["prior_receipt_sha256"] = blind.sha256_file(paths["receipt"])
        write_json(paths["terminal"], terminal)

        raw = json.loads(paths["raw"].read_text())
        raw["rank"] = 1
        raw["authority"] = value["authority"]
        raw["target"] = target
        raw["target_sha256"] = target_sha
        raw["server_launch_receipt_sha256"] = server_sha
        raw["workload"] = target["workload_binding"]
        raw["store_receipt"]["sha256"] = blind.sha256_file(paths["receipt"])
        raw["terminal_receipt"]["sha256"] = blind.sha256_file(paths["terminal"])
        write_json(paths["raw"], raw)

    def test_blind_replay_accepts_complete_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.replay(self._build_replay_bundle(Path(temporary)))
            self.assertGreater(result["payload_bytes"], 0)

    def test_blind_replay_accepts_complete_hypic_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = self.replay(
                self._build_replay_bundle(Path(temporary), "transition_rope_recompute")
            )
            self.assertGreater(result["payload_bytes"], 0)

    def test_blind_replay_rejects_cross_rank_readiness_exchange_after_full_resign(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "prefix_cache")

            def exchange_rank_zero_for_rank_one(server):
                readiness = server["server_info_readiness"]["identity"]
                readiness["rank"] = 1
                readiness["endpoint"] = "http://127.0.0.1:33401/server_info"
                server["server_info_readiness"]["sha256"] = canonical_sha(readiness)

            self.rewrite_server_chain(paths, exchange_rank_zero_for_rank_one)
            with self.assertRaises(Exception):
                self.replay(paths)

    def test_blind_replay_and_replay_all_reject_fully_resigned_rank_one_in_rank_zero_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "prefix_cache")
            self.resign_entire_bundle_as_rank_one(paths)

            forged_expectation = dict(paths)
            forged_expectation.update({
                "expected_rank": 1,
                "expected_snapshot_id": "prefix_cache-rank-1",
                "expected_workload_id": "qasper-7",
            })
            self.assertEqual(self.replay(forged_expectation)["rank"], 1)
            with self.assertRaises(Exception):
                self.replay(paths)

            actual_replay_one = blind.replay_one

            def replay_forged_chain(_receipt, _terminal, _raw, **expected):
                return actual_replay_one(
                    paths["receipt"], paths["terminal"], paths["raw"],
                    target_path=paths["target"], server_path=paths["server"],
                    worker_path=paths["worker"], prereg_path=paths["prereg"],
                    manifest_path=paths["manifest"],
                    expected_manifest_sha256=paths["manifest_sha"],
                    static_dir=paths["static"],
                    expected_mode=expected["expected_mode"],
                    expected_rank=expected["expected_rank"],
                    expected_snapshot_id=expected["expected_snapshot_id"],
                    expected_workload_id=expected["expected_workload_id"],
                )

            output = Path(temporary) / "aggregate.json"
            with mock.patch.object(blind, "replay_one", side_effect=replay_forged_chain):
                with self.assertRaises(Exception):
                    blind.replay_all(
                        Path(temporary) / "rank-zero-layout",
                        output,
                        paths["manifest"],
                        paths["manifest_sha"],
                    )
            self.assertFalse(output.exists())

    def test_replay_all_rechecks_returned_cell_before_append(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "aggregate.json"
            forged_row = {
                "mode": "prefix_cache",
                "rank": 1,
                "snapshot_id": "prefix_cache-rank-1",
                "workload_id": "qasper-7",
            }
            with mock.patch.object(blind, "replay_one", return_value=forged_row):
                with self.assertRaises(Exception):
                    blind.replay_all(
                        Path(temporary), output, Path(temporary) / "manifest", "m" * 64
                    )
            self.assertFalse(output.exists())

    def test_rwd5_canonical_hash_fixed_vector_and_legacy_rejection(self):
        value = {"a": 1}
        expected = "e346432021b04179518d9614f3560ccd71354a4ee101ddcb893d6959a9d6301c"
        legacy_without_newline = "015abd7f5cc57a2dd94b7590f04ad8084273905ee33ec5cebeae62276a97f862"
        self.assertEqual(runner.rwd5_canonical_json_sha256(value), expected)
        self.assertEqual(receipt._process_canonical_sha256(value), expected)
        self.assertEqual(blind.canonical_sha256(value), expected)
        blind.require_rwd5_canonical_sha256(value, expected, "fixed vector")
        with self.assertRaises(Exception):
            blind.require_rwd5_canonical_sha256(value, legacy_without_newline, "legacy hash")

    def test_manifest_accepts_exact_frozen_dot_slash_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "SHA256SUMS"
            path.write_text(f"{'a' * 64}  ./code/dummy.py\n")
            self.assertEqual(blind._manifest_rows(path), {"code/dummy.py": "a" * 64})

    def test_manifest_rejects_absolute_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "SHA256SUMS"
            path.write_text(f"{'a' * 64}  /code/dummy.py\n")
            with self.assertRaises(Exception): blind._manifest_rows(path)

    def test_manifest_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "SHA256SUMS"
            path.write_text(f"{'a' * 64}  code/../dummy.py\n")
            with self.assertRaises(Exception): blind._manifest_rows(path)

    def test_manifest_rejects_empty_canonical_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "SHA256SUMS"
            path.write_text(f"{'a' * 64}  ./\n")
            with self.assertRaises(Exception): blind._manifest_rows(path)

    def test_manifest_rejects_duplicate_canonical_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "SHA256SUMS"
            path.write_text(
                f"{'a' * 64}  code/dummy.py\n"
                f"{'a' * 64}  ./code//./dummy.py\n"
            )
            with self.assertRaises(Exception): blind._manifest_rows(path)

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

    def test_blind_rejects_self_consistent_unrelated_prefix_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "prefix_cache")
            def mutate(value):
                unrelated = [91, 92, 93, 94]
                value["selection"]["owned_document_token_ids"] = unrelated
                value["selection"]["owned_document_token_sha256"] = receipt.token_sha256(unrelated)
                offset = 0
                for entry in value["selection"]["entries"]:
                    count = int(entry["token_count"])
                    tokens = unrelated[offset:offset + count]
                    entry["token_ids"] = tokens
                    entry["token_sha256"] = receipt.token_sha256(tokens)
                    offset += count
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_self_consistent_unrelated_hypic_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "transition_rope_recompute")
            def mutate(value):
                unrelated = [91, 92, 93, 94]
                value["selection"]["owned_document_token_ids"] = unrelated
                value["selection"]["owned_document_token_sha256"] = receipt.token_sha256(unrelated)
                offset = 0
                for entry in value["selection"]["entries"]:
                    count = int(entry["token_count"])
                    tokens = unrelated[offset:offset + count]
                    entry["token_ids"] = tokens
                    entry["token_sha256"] = receipt.token_sha256(tokens)
                    entry["segment_hash_hex"] = receipt.segment_hash_hex(tokens)
                    offset += count
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_prefix_selected_kv_marked_pre_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "prefix_cache")
            def mutate(value):
                observation = value["allocator_observation"]
                pre = observation["pre_free_ownership"]["kv"]
                pre["free_pages"] = sorted(pre["free_pages"] + [1])
                pre["canonical_free_domain"] = sorted(pre["free_pages"] + pre["release_pages"])
                pre["canonical_allocated_domain"] = [2, 3, 4]
                observation["kv_available_tokens"] = len(pre["canonical_free_domain"])
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_hypic_selected_mamba_marked_pre_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "transition_rope_recompute")
            def mutate(value):
                observation = value["allocator_observation"]
                pre = observation["pre_free_ownership"]["mamba"]
                pre["free_slots"] = sorted(pre["free_slots"] + [3])
                pre["canonical_free_domain"] = list(pre["free_slots"])
                pre["canonical_allocated_domain"] = [2]
                observation["mamba_available_slots"] = len(pre["free_slots"])
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

    def test_blind_replay_rejects_wrong_temporal_dtype_with_same_element_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "prefix_cache")
            def mutate(value):
                for row in value["tensor_payload"]["records"]:
                    if row["tensor_name"] == "mamba.temporal":
                        row["dtype"] = "torch.int32"
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_replay_rejects_wrong_transition_dtype_with_same_element_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "transition_rope_recompute")
            def mutate(value):
                for row in value["tensor_payload"]["records"]:
                    if row["tensor_name"] == "mamba.transition":
                        row["dtype"] = "torch.int32"
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_replay_rejects_wrong_conv_tails_dtype_with_same_element_size(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(Path(temporary), "transition_rope_recompute")
            def mutate(value):
                for row in value["tensor_payload"]["records"]:
                    if row["tensor_name"] == "mamba.conv_tails[0]":
                        row["dtype"] = "torch.float16"
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
            self.assertNotIn("dtype", contract)
            self.assertEqual(contract["kv_dtype"], "torch.bfloat16")
            self.assertEqual(contract["mamba_component_dtypes"], {
                "conv": "torch.bfloat16",
                "temporal": "torch.float32",
                "transition": "torch.float32",
                "conv_tails": "torch.bfloat16",
            })

    def test_launcher_has_only_affected_arms_and_external_manifest_gate(self):
        launcher = Path(__file__).with_name("launch_hypic_retained_state_bytes_8gpu.sh").read_text()
        self.assertIn("run_mode prefix_cache", launcher)
        self.assertIn("run_mode transition_rope_recompute", launcher)
        self.assertNotIn("run_mode full_recompute", launcher)
        self.assertIn("EXPECTED_FREEZE_MANIFEST_SHA256", launcher)
        self.assertIn("sha256sum -c", launcher)
        self.assertIn("set -Eeuo pipefail", launcher)
        self.assertIn("SGLANG_MAMBA_CONV_DTYPE=bfloat16", launcher)
        self.assertIn("SGLANG_MAMBA_SSM_DTYPE=float32", launcher)
        self.assertLess(launcher.index("--stage wait_server_info"), launcher.index("--stage server_receipt"))
        self.assertIn("rwd5_install_traps", launcher)
        self.assertIn("rwd5_complete_success", launcher)
        success_function = re.search(r"(?ms)^rwd5_complete_success\(\) \{\n.*?^\}\n", launcher)
        self.assertIsNotNone(success_function)
        self.assertLess(
            success_function.group(0).index("rwd5_cleanup_servers"),
            success_function.group(0).index('touch "${RUN_DIR}/COMPLETED"'),
        )


if __name__ == "__main__": unittest.main()
