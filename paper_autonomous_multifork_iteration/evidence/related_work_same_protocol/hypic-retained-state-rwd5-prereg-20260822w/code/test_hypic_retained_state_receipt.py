import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import torch

import build_hypic_retained_state_static as static_builder
from build_hypic_formal_static import StaticError
import hypic_retained_state_receipt as receipt
import replay_hypic_retained_state_bytes as blind
import run_hypic_retained_state_bytes as runner
import rwd5_model_asset_snapshot as asset_snapshot


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
            "live_debug_validation": {
                "schema": "hypic-rwd5-live-component-debug-binding-v1",
                "status": "passed_debug_only_not_paper_evidence",
                "paper_evidence": False,
                "official_commit": receipt.OFFICIAL_COMMIT,
                "mirror_manifest_sha256": "59530c0c8bc10cedbf4b0bde51d04e5490adeaf369e8738d9df363fc83941026",
                "formal_receipts_emitted": 0,
                "modes": {
                    "prefix_cache": {"components": {"conv[0]": {}, "temporal": {}}},
                    "transition_rope_recompute": {
                        "components": {
                            "conv[0]": {}, "temporal": {}, "transition": {},
                            "conv_tails[0]": {},
                        }
                    },
                },
            },
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
    def _live_debug_mirror() -> Path:
        frozen = Path(__file__).resolve().parent.parent / "live-debug-j-trial-1879097"
        source = (
            Path(__file__).resolve().parents[1]
            / "paper_autonomous_multifork_iteration/evidence/related_work_same_protocol/live-debug-j-trial-1879097"
        )
        path = frozen if frozen.is_dir() else source
        if not path.is_dir():
            raise AssertionError(f"missing frozen live debug mirror: {path}")
        return path

    @staticmethod
    def _allocator_debug_mirror() -> Path:
        frozen = Path(__file__).resolve().parent.parent / "live-allocator-debug-d-trial-1879456"
        source = (
            Path(__file__).resolve().parents[1]
            / "paper_autonomous_multifork_iteration/evidence/related_work_same_protocol/live-allocator-debug-d-trial-1879456"
        )
        path = frozen if frozen.is_dir() else source
        if not path.is_dir():
            raise AssertionError(f"missing frozen allocator debug mirror: {path}")
        return path

    @staticmethod
    def _allocator_debug_authority_root() -> Path:
        frozen = Path(__file__).resolve().parent.parent
        if (frozen / "allocator-debug-d-provenance.json").is_file():
            return frozen
        source = (
            Path(__file__).resolve().parents[1]
            / "paper_autonomous_multifork_iteration/evidence/related_work_same_protocol/hypic-retained-state-rwd5-prereg-20260822w"
        )
        if not (source / "allocator-debug-d-provenance.json").is_file():
            raise AssertionError(f"missing allocator debug authority: {source}")
        return source
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

    def test_hypic_duplicate_allocator_multiset_preserved_but_local_ownership_closes(self):
        cache = PICache()
        cache.req_to_token_pool.mamba_allocator.free_slots = torch.tensor(
            [1, 4, 4, 5], dtype=torch.int64
        )
        output = self.run_snapshot(
            cache, self.target("transition_rope_recompute", "hypic-anomalous-multiset")
        )
        mamba = output["allocator_observation"]["pre_free_ownership"]["mamba"]
        self.assertEqual(mamba["raw_free_slots"], [1, 4, 4, 5])
        self.assertEqual(mamba["raw_count"], 4)
        self.assertEqual(mamba["unique_count"], 3)
        self.assertEqual(
            mamba["duplicates"], [{"slot": 4, "count": 2, "positions": [1, 2]}]
        )
        self.assertEqual(mamba["canonical_allocated_domain"], [2, 3])
        self.assertEqual(
            mamba["consistency_status"],
            "anomalous_duplicate_free_multiset_physical_ownership_closed",
        )
        self.assertFalse(mamba["global_allocator_correctness_claimed"])
        self.assertEqual(output["tensor_payload"]["union"]["unique_overlap_aware_bytes"], 448)

    def test_prefix_duplicate_allocator_multiset_is_rejected(self):
        cache = MambaRadixCache()
        cache.req_to_token_pool.mamba_allocator.free_slots = torch.tensor(
            [1, 4, 4, 5], dtype=torch.int64
        )
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(cache, self.target("prefix_cache", "prefix-duplicate-multiset"))

    def test_hypic_anomaly_missing_domain_must_exactly_equal_target_slots(self):
        cache = PICache()
        cache.req_to_token_pool.mamba_allocator.free_slots = torch.tensor(
            [1, 4, 4], dtype=torch.int64
        )
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(
                cache, self.target("transition_rope_recompute", "hypic-unowned-missing-slot")
            )

    def test_hypic_extra_non_target_cache_entry_is_rejected(self):
        cache = PICache()
        extra = SegmentEntry([8, 9], [6, 7], 4)
        cache._entries[extra.seg_hash] = extra
        with self.assertRaises(receipt.ReceiptError):
            self.run_snapshot(
                cache, self.target("transition_rope_recompute", "hypic-extra-cache-entry")
            )

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

    def test_dtype_debug_run_mode_declarations_execute_under_nounset(self):
        launcher = Path(__file__).with_name(
            "launch_hypic_component_dtype_debug_1gpu.sh"
        ).read_text()
        function = re.search(r"(?ms)^run_debug_mode\(\) \{\n.*?^\}\n", launcher)
        self.assertIsNotNone(function)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = "\n".join([
                "set -Eeuo pipefail",
                function.group(0),
                f"OUTPUT_ROOT={str(root)!r}",
                "RWD5_DTYPE_DEBUG_DECLARATION_SMOKE_ONLY=1",
                "run_debug_mode prefix_cache",
                "run_debug_mode transition_rope_recompute",
            ])
            result = subprocess.run(
                ["bash", "-c", script], text=True, capture_output=True, timeout=5
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertTrue(lines[0].startswith(f"prefix_cache|{root}/targets/prefix_cache"))
            self.assertTrue(lines[1].startswith(
                f"transition_rope_recompute|{root}/targets/transition_rope_recompute"
            ))
        self.assertNotIn('local mode=$1 target=', launcher)
        formal = Path(__file__).with_name(
            "launch_hypic_retained_state_bytes_8gpu.sh"
        ).read_text()
        self.assertNotIn('local rank=$1 port=', formal)

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

    def test_storage_contract_requires_frozen_live_component_debug_authority(self):
        contract = storage_contract()
        contract["dtype_authority"].pop("live_debug_validation")
        prereg = {"model": {
            "storage_contract": contract,
            "storage_contract_sha256": canonical_sha(contract),
        }}
        with self.assertRaises(receipt.ReceiptError):
            receipt._storage_contract(prereg, "prefix_cache")

    def test_live_component_debug_mirror_validates_and_binds_storage_contract(self):
        mirror = self._live_debug_mirror()
        summary = static_builder.validate_live_component_debug(
            mirror,
            expected_manifest_sha256=static_builder.LIVE_DEBUG_MIRROR_MANIFEST_SHA256,
            recurrent_layers=30,
        )
        self.assertEqual(summary["status"], "passed_debug_only_not_paper_evidence")
        self.assertFalse(summary["paper_evidence"])
        self.assertEqual(summary["formal_receipts_emitted"], 0)
        self.assertEqual(
            summary["modes"]["prefix_cache"]["raw_receipt_sha256"],
            "83dbc66e65fdc374014eab62e01a5931b924647c0845c32ee88f7f152028a84f",
        )
        self.assertEqual(
            summary["modes"]["transition_rope_recompute"]["raw_receipt_sha256"],
            "017ee1c6dafaa32c9162c91a36ff38e82744b82a2d2c7d1acd19d32807b99a04",
        )
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            write_json(model / "config.json", {
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_hidden_layers": 32,
                    "full_attention_interval": 16,
                }
            })
            contract = static_builder.model_storage_contract(model, summary)
            self.assertEqual(
                contract["dtype_authority"]["live_debug_validation"], summary
            )

    def test_live_component_debug_mirror_rejects_any_self_consistent_replacement(self):
        mirror = self._live_debug_mirror()
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "mirror"
            shutil.copytree(mirror, copied)
            raw_path = copied / "debug-receipts/prefix_cache-rank-0.json"
            raw = json.loads(raw_path.read_text())
            raw["components"]["conv[0]"]["dtype"] = "torch.float32"
            raw["components"]["conv[0]"]["element_size"] = 4
            write_json(raw_path, raw)
            manifest = copied / "mirror-files.sha256"
            manifest.write_text(manifest.read_text().replace(
                "83dbc66e65fdc374014eab62e01a5931b924647c0845c32ee88f7f152028a84f  ./debug-receipts/prefix_cache-rank-0.json",
                f"{blind.sha256_file(raw_path)}  ./debug-receipts/prefix_cache-rank-0.json",
            ))
            # Even if an attacker supplies the new mirror hash, the frozen J
            # manifest identity is an immutable prerequisite.
            replacement_manifest_sha = blind.sha256_file(manifest)
            with self.assertRaises(Exception):
                static_builder.validate_live_component_debug(
                    copied,
                    expected_manifest_sha256=replacement_manifest_sha,
                    recurrent_layers=30,
                )

    def test_live_allocator_debug_mirror_binds_exact_local_ownership_anomaly(self):
        authority = self._allocator_debug_authority_root()
        summary = static_builder.validate_live_allocator_debug(
            self._allocator_debug_mirror(),
            expected_manifest_sha256=(
                static_builder.ALLOCATOR_DEBUG_MIRROR_MANIFEST_SHA256
            ),
            provenance_path=authority / "allocator-debug-d-provenance.json",
            launch_plan_path=authority / "allocator-debug-d-launch-plan.json",
            frozen_debug_manifest_path=authority / "allocator-debug-d-freeze-SHA256SUMS",
        )
        self.assertEqual(summary["raw_free_count"], 182)
        self.assertEqual(summary["unique_free_count"], 181)
        self.assertEqual(
            summary["duplicates"],
            [{"slot": 3, "count": 2, "positions": [168, 177]}],
        )
        self.assertEqual(summary["unique_allocated_domain"], [14, 15])
        self.assertEqual(summary["target_mamba_state_slots"], [14, 15])
        self.assertEqual(summary["platform_job_id"], 247574)
        self.assertEqual(summary["platform_trial_id"], 1879456)
        self.assertEqual(
            summary["remote_run_dir"],
            "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-mamba-allocator-debug-20260822d",
        )
        self.assertFalse(summary["paper_evidence"])
        self.assertFalse(summary["global_allocator_correctness_claimed"])
        self.assertFalse(summary["launch_plan_contains_platform_job_id"])
        self.assertEqual(
            summary["platform_job_authority"],
            "external execution/submission receipt and post-run platform query supplied by the run coordinator",
        )

    def test_live_allocator_debug_mirror_rejects_self_consistent_replacement(self):
        mirror = self._allocator_debug_mirror()
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "mirror"
            shutil.copytree(mirror, copied)
            raw_path = copied / "debug-receipts/transition_rope_recompute-rank-0.json"
            raw = json.loads(raw_path.read_text())
            raw["allocator"]["duplicates"][0]["positions"] = [167, 177]
            write_json(raw_path, raw)
            manifest = copied / "mirror-files.sha256"
            manifest.write_text(manifest.read_text().replace(
                "0ba574426eef61775acf8f33c8d36249dd9ff421c54845abc7fd77ac443290a9  ./debug-receipts/transition_rope_recompute-rank-0.json",
                f"{blind.sha256_file(raw_path)}  ./debug-receipts/transition_rope_recompute-rank-0.json",
            ))
            with self.assertRaises(Exception):
                static_builder.validate_live_allocator_debug(
                    copied,
                    expected_manifest_sha256=blind.sha256_file(manifest),
                )

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

    def test_terminal_mamba_snapshot_preserves_duplicate_multiset(self):
        allocator = MambaAllocator()
        allocator.free_slots = torch.tensor([1, 4, 4, 5, 2, 3], dtype=torch.int64)
        value = receipt._mamba_free_snapshot(allocator)
        self.assertEqual(value["canonical_free_domain"], [1, 2, 3, 4, 5])
        self.assertEqual(value["canonical_allocated_domain"], [])
        self.assertEqual(
            value["duplicates"], [{"slot": 4, "count": 2, "positions": [1, 2]}]
        )
        self.assertEqual(value["raw_count"], 6)
        self.assertEqual(value["unique_count"], 5)
        self.assertFalse(value["global_allocator_correctness_claimed"])

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
            "rwd5_pid_or_group_alive",
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
            script = "\n".join([
                "set -Eeuo pipefail",
                *functions,
                f"RUN_DIR={str(root)!r}",
                "RWD5_RUN_SUCCEEDED=0",
                "RWD5_LAST_ERROR=0",
                "setsid sleep 300 & p0=$!",
                "setsid sleep 300 & p1=$!",
                'printf "%s\\n" "$p0" > "$RUN_DIR/server-logs/mock-0.pid"',
                'printf "%s\\n" "$p1" > "$RUN_DIR/server-logs/mock-1.pid"',
                "SERVER_PIDS=($p0 $p1)",
                "rwd5_install_traps",
                "mock_server_receipt_stage() { return 37; }",
                "mock_server_receipt_stage",
            ])
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=20)
            self.assertEqual(result.returncode, 37, result.stderr)
            self.assertTrue((root / "FAILED").is_file())
            self.assertEqual((root / "FAILED").read_text().strip(), "37")
            self.assertFalse(completed.exists())
            for pid_file in logs.glob("*.pid"):
                with self.assertRaises(ProcessLookupError):
                    os.kill(int(pid_file.read_text()), 0)

    def test_exit_trap_escalates_for_sigterm_ignoring_process_group(self):
        launcher = Path(__file__).with_name("launch_hypic_retained_state_bytes_8gpu.sh").read_text()
        functions = []
        for name in ("rwd5_pid_or_group_alive", "rwd5_cleanup_servers", "rwd5_on_exit", "rwd5_install_traps"):
            match = re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}\n", launcher)
            self.assertIsNotNone(match, name)
            functions.append(match.group(0))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "server-logs"
            logs.mkdir()
            (root / "COMPLETED").write_text("must be removed\n")
            script = "\n".join([
                "set -Eeuo pipefail",
                *functions,
                f"RUN_DIR={str(root)!r}",
                "RWD5_RUN_SUCCEEDED=0",
                "RWD5_LAST_ERROR=0",
                "setsid bash -c 'trap \"\" TERM; exec sleep 300' & stubborn=$!",
                'printf "%s\\n" "$stubborn" > "$RUN_DIR/server-logs/stubborn.pid"',
                "SERVER_PIDS=($stubborn)",
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
            self.assertLess(elapsed, 14.0)
            self.assertEqual((root / "FAILED").read_text().strip(), "37")
            self.assertFalse((root / "COMPLETED").exists())
            stubborn_pid = int((logs / "stubborn.pid").read_text())
            with self.assertRaises(ProcessLookupError):
                os.killpg(stubborn_pid, 0)

    def _build_replay_bundle(self, root: Path, mode="prefix_cache"):
        cache = MambaRadixCache() if mode == "prefix_cache" else PICache()
        if mode == "transition_rope_recompute":
            # Mirror the validated live allocator anomaly: duplicate free
            # bookkeeping for an unrelated slot, while the exact two target
            # entry slots remain the entire unique allocated physical domain.
            cache.req_to_token_pool.mamba_allocator.free_slots = torch.tensor(
                [1, 4, 4, 5], dtype=torch.int64
            )
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
            "allocator_debug_validation": {
                "schema": "hypic-rwd5-live-allocator-debug-binding-v1",
                "status": "passed_debug_only_local_physical_ownership_authority",
                "paper_evidence": False,
                "official_commit": receipt.OFFICIAL_COMMIT,
                "platform_job_id": 247574,
                "platform_trial_id": 1879456,
                "remote_run_dir": "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-mamba-allocator-debug-20260822d",
                "mirror_manifest_sha256": "d57e3e5436f9b7b586a3788a1f3205d9ee3e4f6403496edc3205c2927a842f7e",
                "allocator_size": 183,
                "raw_free_count": 182,
                "unique_free_count": 181,
                "duplicates": [{"slot": 3, "count": 2, "positions": [168, 177]}],
                "duplicate_excess_count": 1,
                "unique_allocated_domain": [14, 15],
                "target_mamba_state_slots": [14, 15],
                "formal_receipts_emitted": 0,
                "global_allocator_correctness_claimed": False,
            },
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
        kv_domain = list(range(1, 16))
        terminal_mamba_allocator = MambaAllocator()
        terminal_mamba_allocator.free_slots = torch.tensor(
            [1, 4, 4, 5, 2, 3]
            if mode == "transition_rope_recompute"
            else list(range(1, 6)),
            dtype=torch.int64,
        )
        terminal_mamba = receipt._mamba_free_snapshot(terminal_mamba_allocator)
        terminal = {"schema": "forkaudit-hypic-retained-state-terminal-v2",
                    "official_commit": receipt.OFFICIAL_COMMIT, "passed": True,
                    "authority": snapshot["authority"],
                    "prior_receipt_sha256": blind.sha256_file(receipt_path),
                    "checks": {"target_entries_after": 0, "all_cache_entries_after": 0,
                               "old_kv_slots_all_free": True, "old_mamba_slots_all_free": True,
                               "old_kv_slots_preallocated": True, "old_mamba_slots_preallocated": True,
                               "kv_available_tokens": 15, "kv_capacity_tokens": 15,
                               "mamba_available_slots": terminal_mamba["raw_count"],
                               "mamba_capacity_slots": 5,
                               "mamba_unique_physical_domain_closed": True,
                               "mamba_duplicate_anomaly_preserved_without_migration_or_growth": True,
                               "mamba_global_allocator_correctness_claimed": False,
                               "store_metric_scope": "exact target-entry-owned physical tensor-range union only",
                               "kv_free_list": {"page_size": 1, "size": 15,
                                                "free_pages": kv_domain, "release_pages": [], "exact_domain": kv_domain},
                               "mamba_free_list": terminal_mamba}}
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

    @staticmethod
    def resign_mamba_multiset(value, raw_slots):
        """Recompute every producer field to exercise semantic replay checks."""
        counts = Counter(raw_slots)
        positions = {}
        for index, slot in enumerate(raw_slots):
            positions.setdefault(slot, []).append(index)
        duplicates = [
            {"slot": slot, "count": counts[slot], "positions": positions[slot]}
            for slot in sorted(counts)
            if counts[slot] > 1
        ]
        size = int(value["size"])
        expected = set(range(1, size + 1))
        unique = sorted(counts)
        excess = sum(row["count"] - 1 for row in duplicates)
        value.update({
            "raw_free_slots": list(raw_slots),
            "raw_count": len(raw_slots),
            "unique_count": len(unique),
            "duplicates": duplicates,
            "duplicate_excess_count": excess,
            "canonical_free_domain": unique,
            "canonical_allocated_domain": sorted(expected - set(unique)),
            "consistency_status": (
                "anomalous_duplicate_free_multiset_physical_ownership_closed"
                if excess else "exact_unique_free_domain"
            ),
            "physical_ownership_basis": (
                "unique in-domain physical slot identities; duplicate multiset anomaly reported separately"
            ),
            "global_allocator_correctness_claimed": False,
        })

    def rewrite_receipt_chain(self, paths, mutator):
        value = json.loads(paths["receipt"].read_text()); mutator(value); write_json(paths["receipt"], value)
        terminal = json.loads(paths["terminal"].read_text())
        terminal["prior_receipt_sha256"] = blind.sha256_file(paths["receipt"]); write_json(paths["terminal"], terminal)
        raw = json.loads(paths["raw"].read_text())
        raw["store_receipt"]["sha256"] = blind.sha256_file(paths["receipt"])
        raw["terminal_receipt"]["sha256"] = blind.sha256_file(paths["terminal"]); write_json(paths["raw"], raw)

    def rewrite_terminal_chain(self, paths, mutator):
        terminal = json.loads(paths["terminal"].read_text())
        mutator(terminal)
        write_json(paths["terminal"], terminal)
        raw = json.loads(paths["raw"].read_text())
        raw["terminal_receipt"]["sha256"] = blind.sha256_file(paths["terminal"])
        write_json(paths["raw"], raw)

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
            self.assertEqual(
                result["mamba_allocator_consistency_status"],
                "anomalous_duplicate_free_multiset_physical_ownership_closed",
            )
            self.assertEqual(result["mamba_allocator_duplicate_excess_count"], 1)
            self.assertEqual(
                result["metric_validity"],
                "valid_for_exact_target_owned_physical_tensor_range_union_only",
            )
            self.assertFalse(result["global_allocator_correctness_claimed"])

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
                pre["raw_free_slots"].append(3)
                pre["raw_count"] += 1
                pre["unique_count"] += 1
                pre["canonical_free_domain"] = [1, 3, 4, 5]
                pre["canonical_allocated_domain"] = [2]
                observation["mamba_available_slots"] = pre["raw_count"]
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception): self.replay(paths)

    def test_blind_rejects_forged_duplicate_positions(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(
                Path(temporary), "transition_rope_recompute"
            )
            def mutate(value):
                value["allocator_observation"]["pre_free_ownership"]["mamba"][
                    "duplicates"
                ][0]["positions"] = [0, 2]
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception):
                self.replay(paths)

    def test_blind_rejects_fully_resigned_selected_slot_marked_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(
                Path(temporary), "transition_rope_recompute"
            )
            def mutate(value):
                observation = value["allocator_observation"]
                self.resign_mamba_multiset(
                    observation["pre_free_ownership"]["mamba"],
                    [1, 3, 4, 4, 5],
                )
                observation["mamba_available_slots"] = 5
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception):
                self.replay(paths)

    def test_blind_rejects_fully_resigned_unrelated_missing_physical_slot(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(
                Path(temporary), "transition_rope_recompute"
            )
            def mutate(value):
                observation = value["allocator_observation"]
                self.resign_mamba_multiset(
                    observation["pre_free_ownership"]["mamba"], [1, 4, 4]
                )
                observation["mamba_available_slots"] = 3
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception):
                self.replay(paths)

    def test_blind_rejects_hidden_allocator_anomaly_status(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(
                Path(temporary), "transition_rope_recompute"
            )
            def mutate(value):
                value["allocator_observation"]["pre_free_ownership"]["mamba"][
                    "consistency_status"
                ] = "exact_unique_free_domain"
            self.rewrite_receipt_chain(paths, mutate)
            with self.assertRaises(Exception):
                self.replay(paths)

    def test_blind_rejects_terminal_duplicate_migration(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(
                Path(temporary), "transition_rope_recompute"
            )
            def mutate(value):
                mamba = value["checks"]["mamba_free_list"]
                self.resign_mamba_multiset(mamba, [1, 4, 5, 5, 2, 3])
                value["checks"]["mamba_available_slots"] = 6
            self.rewrite_terminal_chain(paths, mutate)
            with self.assertRaises(Exception):
                self.replay(paths)

    def test_blind_rejects_terminal_duplicate_growth(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(
                Path(temporary), "transition_rope_recompute"
            )
            def mutate(value):
                mamba = value["checks"]["mamba_free_list"]
                self.resign_mamba_multiset(mamba, [1, 4, 4, 4, 5, 2, 3])
                value["checks"]["mamba_available_slots"] = 7
            self.rewrite_terminal_chain(paths, mutate)
            with self.assertRaises(Exception):
                self.replay(paths)

    def test_blind_rejects_terminal_selected_slot_not_returned(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._build_replay_bundle(
                Path(temporary), "transition_rope_recompute"
            )
            def mutate(value):
                mamba = value["checks"]["mamba_free_list"]
                self.resign_mamba_multiset(mamba, [1, 4, 4, 5, 2])
                value["checks"]["mamba_available_slots"] = 5
            self.rewrite_terminal_chain(paths, mutate)
            with self.assertRaises(Exception):
                self.replay(paths)

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
        self.assertIn("live-debug-j-trial-1879097", launcher)
        self.assertIn(static_builder.LIVE_DEBUG_MIRROR_MANIFEST_SHA256, launcher)
        self.assertIn("--live-debug-root", launcher)
        self.assertIn("--expected-live-debug-manifest-sha256", launcher)
        self.assertIn('sha256sum -c "$LIVE_DEBUG_MANIFEST"', launcher)
        self.assertLess(launcher.index("--stage wait_server_info"), launcher.index("--stage server_receipt"))
        self.assertIn("rwd5_install_traps", launcher)
        self.assertIn("rwd5_complete_success", launcher)
        self.assertIn("test_run_hypic_same_protocol", launcher)
        self.assertIn("rwd5_verify_terminal_runtime_idle", launcher)
        self.assertIn("process PID/PGID still alive after bounded KILL cleanup", launcher)
        success_function = re.search(r"(?ms)^rwd5_complete_success\(\) \{\n.*?^\}\n", launcher)
        self.assertIsNotNone(success_function)
        self.assertLess(
            success_function.group(0).index("rwd5_cleanup_servers"),
            success_function.group(0).index('touch "${RUN_DIR}/COMPLETED"'),
        )
        self.assertLess(
            success_function.group(0).index("rwd5_verify_terminal_runtime_idle"),
            success_function.group(0).index('touch "${RUN_DIR}/COMPLETED"'),
        )

    def test_cleanup_reaps_sigterm_ignoring_orphan_process_group(self):
        launcher = Path(__file__).with_name("launch_hypic_retained_state_bytes_8gpu.sh").read_text()
        functions = []
        for name in ("rwd5_pid_or_group_alive", "rwd5_cleanup_servers"):
            match = re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}\n", launcher)
            self.assertIsNotNone(match, name)
            functions.append(match.group(0))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs = root / "server-logs"
            logs.mkdir()
            child_file = root / "orphan.pid"
            leader = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import os,signal,sys,time; p=os.fork(); "
                        "(open(sys.argv[1],'w').write(str(os.getpid())), "
                        "signal.signal(signal.SIGTERM,signal.SIG_IGN),time.sleep(300)) "
                        "if p==0 else os._exit(0)"
                    ),
                    str(child_file),
                ],
                start_new_session=True,
            )
            try:
                leader.wait(timeout=3)
                for _ in range(100):
                    if child_file.is_file():
                        break
                    time.sleep(0.02)
                self.assertTrue(child_file.is_file())
                orphan_pid = int(child_file.read_text())
                (logs / "orphan-group.pid").write_text(f"{leader.pid}\n")
                script = "\n".join([
                    "set -Eeuo pipefail",
                    *functions,
                    f"RUN_DIR={str(root)!r}",
                    "RWD5_CLEANED_PIDS=''",
                    "SERVER_PIDS=()",
                    "rwd5_cleanup_servers",
                ])
                result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=18)
                self.assertEqual(result.returncode, 0, result.stderr)
                with self.assertRaises(ProcessLookupError):
                    os.killpg(leader.pid, 0)
                # The child can remain briefly as an init-owned zombie, but it
                # must no longer be signalable/running in the old group.
                try:
                    os.kill(orphan_pid, 0)
                except ProcessLookupError:
                    pass
            finally:
                try:
                    os.killpg(leader.pid, 9)
                except ProcessLookupError:
                    pass

    def test_terminal_gpu_gate_rejects_compute_and_nonzero_memory(self):
        launcher = Path(__file__).with_name("launch_hypic_retained_state_bytes_8gpu.sh").read_text()
        match = re.search(r"(?ms)^rwd5_verify_terminal_runtime_idle\(\) \{\n.*?^\}\n", launcher)
        self.assertIsNotNone(match)
        function = match.group(0)
        uuids = [f"GPU-{index}" for index in range(8)]

        def run_gate(compute_line="", nonzero_index=None, process_line="", duplicate_index=False):
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                if process_line == "matching":
                    process_line = (
                        f"123 123 python -m sglang.launch_server "
                        f"{root / 'repo'} {root / 'formal'}"
                    )
                fake = root / "nvidia-smi"
                rows_data = [
                    [index, uuid, 1 if index == nonzero_index else 0]
                    for index, uuid in enumerate(uuids)
                ]
                if duplicate_index:
                    rows_data[-1][:2] = rows_data[0][:2]
                rows = "\n".join(f"{index}, {uuid}, {memory}" for index, uuid, memory in rows_data)
                fake.write_text(
                    "#!/bin/bash\n"
                    "case \"$*\" in\n"
                    f"  *query-compute-apps*) printf '%s\\n' {compute_line!r} ;;\n"
                    f"  *query-gpu*) printf '%b\\n' {rows!r} ;;\n"
                    "  *) exit 99 ;;\n"
                    "esac\n"
                )
                fake.chmod(0o755)
                fake_ps = root / "ps"
                fake_ps.write_text(
                    "#!/bin/bash\n"
                    f"printf '%s\\n' {process_line!r}\n"
                )
                fake_ps.chmod(0o755)
                script = "\n".join([
                    "set -Eeuo pipefail",
                    function,
                    f"PATH={str(root)!r}:$PATH",
                    f"RUN_DIR={str(root / 'formal')!r}",
                    f"INSTRUMENTED_REPO={str(root / 'repo')!r}",
                    f"CLIENT={str(root / 'client.py')!r}",
                    "GPU_UUIDS=(" + " ".join(uuids) + ")",
                    "rwd5_verify_terminal_runtime_idle",
                ])
                return subprocess.run(["bash", "-c", script], text=True, capture_output=True)

        self.assertEqual(run_gate().returncode, 0)
        self.assertNotEqual(run_gate("GPU-0, 123, python, 1").returncode, 0)
        self.assertNotEqual(run_gate(nonzero_index=5).returncode, 0)
        self.assertNotEqual(run_gate(duplicate_index=True).returncode, 0)
        self.assertNotEqual(
            run_gate(process_line="matching").returncode,
            0,
        )
        self.assertEqual(
            run_gate(process_line="456 456 /usr/bin/python unrelated_worker.py").returncode,
            0,
        )

    def test_frozen_inherited_test_failure_blocks_preflight(self):
        launcher = Path(__file__).with_name("launch_hypic_retained_state_bytes_8gpu.sh").read_text()
        match = re.search(r"(?ms)^rwd5_run_frozen_unit_tests\(\) \{\n.*?^\}\n", launcher)
        self.assertIsNotNone(match)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            fake = root / "python"
            fake.write_text(
                "#!/bin/bash\n"
                "case \" $* \" in *' test_run_hypic_same_protocol '*) exit 29;; *) exit 0;; esac\n"
            )
            fake.chmod(0o755)
            script = "\n".join([
                "set -Eeuo pipefail",
                match.group(0),
                f"CODE_DIR={str(root)!r}",
                f"PYTHON_BIN={str(fake)!r}",
                f"RUN_DIR={str(root)!r}",
                "rwd5_run_frozen_unit_tests",
            ])
            result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
            self.assertEqual(result.returncode, 29)
            self.assertTrue((root / "logs" / "inherited-same-protocol-tests.log").is_file())

    def test_safe_w_wrapper_closes_ambient_and_cwd_authority(self):
        wrapper = Path(__file__).with_name("launch_hypic_retained_state_bytes_safe_w.sh").read_text()
        guard_path = Path(__file__).with_name("rwd5_safe_cwd_guard.py")
        self.assertIn("exec /usr/bin/env -i", wrapper)
        self.assertIn("cd /", wrapper)
        self.assertIn('assert "" not in sys.path; assert "/" not in sys.path', wrapper)
        self.assertIn("test_run_hypic_same_protocol.py", wrapper)
        self.assertIn("/tmp/rwd5-hypic-store-freeze-w/code", wrapper)
        self.assertIn("/tmp/HYPIC-98147c0/python/sglang", wrapper)
        self.assertNotIn("LIBRARY_PATH=/usr/local/cuda/lib64/stubs:", wrapper)
        spec = __import__("importlib.util").util.spec_from_file_location("rwd5_guard", guard_path)
        guard = __import__("importlib.util").util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "test_run_hypic_same_protocol.py").write_text("ATTACKER=True\n")
            with self.assertRaises(guard.SafeCwdError):
                guard.validate_import_shadows(root)
        for key in (
            "PYTHON_BIN", "OFFICIAL_REPO", "FREEZE_ROOT", "CODE_DIR", "FREEZE_MANIFEST",
            "LIVE_DEBUG_ROOT", "ALLOCATOR_DEBUG_ROOT", "MODEL_DIR", "VALIDATION_DATA",
            "RUN_DIR", "INSTRUMENTED_REPO", "INHERITED_TEST", "INHERITED_LAUNCHER",
            "SAFE_WRAPPER", "SAFE_CWD_GUARD",
            "MODEL_ASSET_SNAPSHOT", "ASSET_OBSERVATION_PATH",
            "INVALID_T_RECEIPT", "PLATFORM_AUTHORITY_RECEIPT",
            "RETIRED_W_TRIAL_RECEIPT", "RWD5_PID1_ENV_PATH",
            "RWD5_PLATFORM_JOB_ID", "RWD5_PLATFORM_TRIAL_ID",
        ):
            self.assertIn(f"  {key}=", wrapper)
        block = re.search(
            r"(?ms)^# PINNED_EXEC_ENV_BEGIN.*?^exec /usr/bin/env -i \\\n"
            r"(?P<body>.*?)^  /bin/bash --noprofile --norc ",
            wrapper,
        )
        self.assertIsNotNone(block)
        assignments = []
        for line in block.group("body").splitlines():
            token = line.strip()
            if not token:
                continue
            self.assertTrue(token.endswith("\\"), token)
            token = token[:-1].strip()
            if token.startswith("EXPECTED_FREEZE_MANIFEST_SHA256="):
                token = "EXPECTED_FREEZE_MANIFEST_SHA256=frozen-manifest-sha"
            assignments.append(token)
        malicious = dict(os.environ)
        malicious.update({
            "CODE_DIR": "/tmp/attacker-code",
            "PYTHON_BIN": "/tmp/attacker-python",
            "RUN_DIR": "/tmp/attacker-run",
            "BASH_ENV": "/tmp/attacker-bash-env",
        })
        child = subprocess.run(
            ["/usr/bin/env", "-i", *assignments, "/usr/bin/env"],
            env=malicious,
            text=True,
            capture_output=True,
            check=True,
        )
        closed = dict(line.split("=", 1) for line in child.stdout.splitlines())
        self.assertEqual(closed["CODE_DIR"], "/tmp/rwd5-hypic-store-freeze-w/code")
        self.assertEqual(closed["PYTHON_BIN"], "/tmp/round25-hypic-env/venv/bin/python")
        self.assertEqual(
            closed["RUN_DIR"],
            "/mnt/tidal-alsh-hilab/dataset/diandian/user/liuhanzuo/indep-bench_assets/runs/qcomem/hypic-retained-state-rwd5-20260822w",
        )
        self.assertNotIn("BASH_ENV", closed)
        self.assertEqual(closed["RWD5_PLATFORM_JOB_ID"], "247699")
        self.assertEqual(closed["RWD5_PLATFORM_TRIAL_ID"], "1880085")

    def test_w_wrapper_reads_platform_identity_only_from_pid1_environ(self):
        wrapper = Path(__file__).with_name("launch_hypic_retained_state_bytes_safe_w.sh").read_text()
        match = re.search(r"(?ms)^w_require_pid1_env\(\) \{\n.*?^\}\n", wrapper)
        self.assertIsNotNone(match)
        function = match.group(0).replace("/usr/bin/tr", shutil.which("tr"))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid1 = root / "pid1.environ"
            correct = (
                b"QS_JOB_ID=247699\0QS_TRIAL_ID=1880085\0"
                b"QCOMEM_DEBUG_SCOPE=ROUND27_HYPIC_STORE_FORMAL_W\0"
            )
            pid1.write_bytes(correct)
            setup = "\n".join([
                "set -Eeuo pipefail",
                "die() { printf '%s\\n' \"ERROR: $*\" >&2; exit 1; }",
                function,
                f"W_PID1_ENV_PATH={str(pid1)!r}",
            ])
            caller = dict(os.environ, QS_JOB_ID="ATTACKER", QS_TRIAL_ID="ATTACKER")
            passed = subprocess.run(
                ["bash", "-c", setup + "\nw_require_pid1_env QS_JOB_ID 247699\nw_require_pid1_env QS_TRIAL_ID 1880085"],
                env=caller, text=True, capture_output=True,
            )
            self.assertEqual(passed.returncode, 0, passed.stderr)
            pid1.write_bytes(correct + b"QS_TRIAL_ID=1880085\0")
            duplicate = subprocess.run(
                ["bash", "-c", setup + "\nw_require_pid1_env QS_TRIAL_ID 1880085"],
                env=caller, text=True, capture_output=True,
            )
            self.assertNotEqual(duplicate.returncode, 0)
            pid1.write_bytes(correct.replace(b"1880085", b"1879843"))
            wrong = subprocess.run(
                ["bash", "-c", setup + "\nw_require_pid1_env QS_TRIAL_ID 1880085"],
                env=caller, text=True, capture_output=True,
            )
            self.assertNotEqual(wrong.returncode, 0)

    def test_w_static_platform_authority_binds_receipt_and_pid1_environ(self):
        receipt = Path(__file__).resolve().parent.parent / "platform-execution-authority-w.json"
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            os.environ,
            {"RWD5_PLATFORM_JOB_ID": "247699", "RWD5_PLATFORM_TRIAL_ID": "1880085"},
            clear=False,
        ):
            root = Path(temporary)
            pid1 = root / "pid1.environ"
            exact = (
                b"QS_JOB_ID=247699\0QS_TRIAL_ID=1880085\0"
                b"QCOMEM_DEBUG_SCOPE=ROUND27_HYPIC_STORE_FORMAL_W\0"
            )
            pid1.write_bytes(exact)
            binding = static_builder.validate_platform_execution_authority(receipt, pid1)
            self.assertTrue(binding["pid1_identity_verified"])
            self.assertEqual(binding["platform_trial_id"], 1880085)
            pid1.write_bytes(exact.replace(b"1880085", b"1879843"))
            with self.assertRaises(StaticError):
                static_builder.validate_platform_execution_authority(receipt, pid1)
            pid1.write_bytes(exact + b"QS_JOB_ID=247699\0")
            with self.assertRaises(StaticError):
                static_builder.validate_platform_execution_authority(receipt, pid1)
            tampered = root / "authority.json"
            value = json.loads(receipt.read_text())
            value["platform_trial_id"] = 1879843
            tampered.write_text(json.dumps(value))
            pid1.write_bytes(exact)
            with self.assertRaises(StaticError):
                static_builder.validate_platform_execution_authority(tampered, pid1)

    def test_w_asset_snapshot_open_fstat_hash_and_same_preflight_drift(self):
        data = b"exact frozen bytes"
        digest = hashlib.sha256(data).hexdigest()
        stable = mock.Mock(
            st_mode=(0o100000 | 0o444), st_uid=0, st_gid=0, st_size=len(data),
            st_ino=123, st_dev=456, st_nlink=1, st_rdev=0,
            st_blksize=4096, st_blocks=8, st_mtime_ns=1000, st_ctime_ns=2000,
        )
        path = mock.Mock()
        path.name = "model-artifacts.sha256"
        path.is_symlink.return_value = False
        path.lstat.return_value = stable
        with mock.patch.object(asset_snapshot.os, "open", return_value=17) as opened, \
             mock.patch.object(asset_snapshot.os, "fstat", side_effect=[stable, stable]), \
             mock.patch.object(asset_snapshot.os, "read", side_effect=[data, b""]), \
             mock.patch.object(asset_snapshot.os, "close"):
            row = asset_snapshot.snapshot_one(path, digest, len(data))
        self.assertEqual(row["sha256"], digest)
        self.assertTrue(opened.call_args.args[1] & getattr(os, "O_NOFOLLOW", 0))
        moved = mock.Mock(**vars(stable))
        moved.st_ino = 124
        with mock.patch.object(asset_snapshot.os, "open", return_value=17), \
             mock.patch.object(asset_snapshot.os, "fstat", side_effect=[stable, moved]), \
             mock.patch.object(asset_snapshot.os, "read", side_effect=[data, b""]), \
             mock.patch.object(asset_snapshot.os, "close"):
            with self.assertRaises(asset_snapshot.SnapshotError):
                asset_snapshot.snapshot_one(path, digest, len(data))

    def test_w_cross_node_physical_fields_are_observation_not_authority(self):
        def value(inode, device, stamp):
            entries = []
            for name, (digest, size) in sorted(asset_snapshot.EXPECTED.items()):
                entries.append({
                    "name": name,
                    "sha256": digest,
                    "stable_cross_node_authority": {
                        "regular_non_symlink": True, "mode_octal": "0444",
                        "uid": 0, "gid": 0, "size": size,
                    },
                    "same_preflight_observation": {
                        "mode": 0o100444, "uid": 0, "gid": 0, "size": size,
                        "inode": inode, "device": device, "nlink": 1, "rdev": 0,
                        "block_size": 4096, "blocks": 8,
                        "mtime_ns": stamp, "ctime_ns": stamp,
                    },
                    "physical_identity_fields_are_observation_only": True,
                    "atime_excluded_because_hashing_is_a_read": True,
                })
                inode += 1
            return {
                "schema": asset_snapshot.SCHEMA,
                "model_root": "/model",
                "entries": entries,
                "cross_node_authority_excludes_inode_device_and_timestamps": True,
                "same_preflight_requires_exact_observation_equality": True,
            }
        old_node = value(58755952, 2097177, 1787376685000000000)
        new_node = value(10509879, 1048686, 1787380591000000000)
        asset_snapshot.validate_snapshot(old_node, model_root=Path("/model"))
        asset_snapshot.validate_snapshot(new_node, model_root=Path("/model"))
        self.assertNotEqual(old_node, new_node)
        self.assertEqual(
            [row["stable_cross_node_authority"] for row in old_node["entries"]],
            [row["stable_cross_node_authority"] for row in new_node["entries"]],
        )

    def test_w_wrapper_double_snapshot_and_publication_order(self):
        wrapper = Path(__file__).with_name("launch_hypic_retained_state_bytes_safe_w.sh").read_text()
        verify_calls = [m.start() for m in re.finditer(r"(?m)^w_open_trusted_authority$", wrapper)]
        self.assertEqual(len(verify_calls), 3)
        positions = {
            "weight_bytes": wrapper.index("sha256sum -c model-weights.sha256"),
            "artifact_bytes": wrapper.index("sha256sum -c model-artifacts.sha256"),
            "writable": wrapper.index("writable top-level model assets remain"),
            "pre": wrapper.index("ASSET_PRE_SNAPSHOT="),
            "import": wrapper.index("SGLang import authority is not exact official repository"),
            "post_helper_fd": wrapper.index('exec 8< "$W_ASSET_SNAPSHOT"'),
            "post": wrapper.index("ASSET_POST_SNAPSHOT="),
            "equal": wrapper.index('[[ "$ASSET_PRE_SNAPSHOT" == "$ASSET_POST_SNAPSHOT" ]]'),
            "launcher_fd": wrapper.index('exec 9< "$W_LAUNCHER"'),
            "publish": wrapper.index('/usr/bin/mkdir "$W_PREFLIGHT_DIR"'),
            "exec": wrapper.index("exec /usr/bin/env -i"),
        }
        self.assertLess(positions["weight_bytes"], positions["pre"])
        self.assertLess(positions["artifact_bytes"], positions["pre"])
        self.assertLess(positions["writable"], positions["pre"])
        self.assertLess(positions["writable"], verify_calls[0])
        self.assertLess(verify_calls[0], positions["pre"])
        self.assertLess(positions["pre"], positions["import"])
        self.assertLess(positions["import"], verify_calls[1])
        self.assertLess(verify_calls[1], positions["post_helper_fd"])
        self.assertLess(positions["import"], positions["post"])
        self.assertLess(positions["post"], positions["equal"])
        self.assertLess(positions["equal"], verify_calls[2])
        self.assertLess(verify_calls[2], positions["launcher_fd"])
        self.assertLess(positions["equal"], positions["publish"])
        self.assertLess(positions["publish"], positions["exec"])
        self.assertNotIn("58755972", wrapper)
        self.assertNotIn("2097177", wrapper)

    def test_w_manifest_checks_reject_replacement_and_fd_exec_keeps_exact_inode(self):
        wrapper = Path(__file__).with_name("launch_hypic_retained_state_bytes_safe_w.sh").read_text()
        self.assertEqual(wrapper.count("/usr/bin/cat /proc/self/fd/10"), 1)
        self.assertNotIn("sha256sum /proc/self/fd/10", wrapper)
        functions = []
        for name in (
            "w_open_trusted_authority", "w_close_trusted_authority",
            "w_manifest_member_sha", "w_verify_open_fd",
        ):
            match = re.search(rf"(?ms)^{name}\(\) \{{\n.*?^\}}\n", wrapper)
            self.assertIsNotNone(match, name)
            function = match.group(0).replace("/usr/bin/sha256sum", shutil.which("sha256sum"))
            function = function.replace("/usr/bin/cat", shutil.which("cat"))
            if not Path("/proc/self/fd").is_dir():
                function = function.replace("/proc/self/fd/", "/dev/fd/")
            functions.append(function)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "code").mkdir()
            helper = root / "code/helper.py"
            launcher = root / "code/launcher.sh"
            other = root / "other.txt"
            helper.write_text("print('ORIGINAL_HELPER')\n")
            launcher.write_text("#!/bin/bash\nprintf 'ORIGINAL_LAUNCHER\\n'\n")
            other.write_text("authority\n")
            stop = root / "STOP"
            stop.write_text("stop\n")
            manifest = root / "SHA256SUMS"
            manifest.write_text("".join(
                f"{hashlib.sha256(path.read_bytes()).hexdigest()}  ./{path.relative_to(root)}\n"
                for path in (helper, launcher, other)
            ))
            def setup_for_current_manifest():
                return "\n".join([
                    "set -Eeuo pipefail",
                    "die() { printf '%s\\n' \"ERROR: $*\" >&2; exit 1; }",
                    *functions,
                    f"W_ROOT={str(root)!r}", f"W_MANIFEST={str(manifest)!r}",
                    f"W_STOP={str(stop)!r}", "W_EXPECTED_MANIFEST_MEMBERS=3",
                    f"EXPECTED_W_MANIFEST_SHA256={hashlib.sha256(manifest.read_bytes()).hexdigest()}",
                    f"EXPECTED_W_STOP_SHA256={hashlib.sha256(stop.read_bytes()).hexdigest()}",
                ])
            setup = setup_for_current_manifest()
            good = subprocess.run(
                ["bash", "-c", setup + "\nw_open_trusted_authority\nw_close_trusted_authority"],
                capture_output=True, text=True,
            )
            self.assertEqual(good.returncode, 0, good.stderr)
            helper.write_text("print('FORGED_HELPER')\n")
            changed = subprocess.run(
                ["bash", "-c", setup_for_current_manifest() + "\nw_open_trusted_authority"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(changed.returncode, 0, changed.stdout + changed.stderr)
            helper.write_text("print('ORIGINAL_HELPER')\n")
            exact_manifest = manifest.read_text()
            manifest.write_text(exact_manifest.rstrip("\n"))
            no_newline = subprocess.run(
                ["bash", "-c", setup_for_current_manifest() + "\nv_open_trusted_authority"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(no_newline.returncode, 0)
            manifest.write_text(exact_manifest)
            duplicate_rows = exact_manifest.splitlines()
            duplicate_rows[2] = duplicate_rows[0]
            manifest.write_text("\n".join(duplicate_rows) + "\n")
            duplicate = subprocess.run(
                ["bash", "-c", setup_for_current_manifest() + "\nv_open_trusted_authority"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(duplicate.returncode, 0)
            manifest.write_text(exact_manifest)
            traversal_rows = exact_manifest.splitlines()
            traversal_rows[2] = traversal_rows[2].split("  ", 1)[0] + "  ./../other.txt"
            manifest.write_text("\n".join(traversal_rows) + "\n")
            traversal = subprocess.run(
                ["bash", "-c", setup_for_current_manifest() + "\nv_open_trusted_authority"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(traversal.returncode, 0)
            manifest.write_text(exact_manifest)
            absolute_rows = exact_manifest.splitlines()
            absolute_rows[2] = absolute_rows[2].split("  ", 1)[0] + "  /absolute/other.txt"
            manifest.write_text("\n".join(absolute_rows) + "\n")
            absolute = subprocess.run(
                ["bash", "-c", setup_for_current_manifest() + "\nv_open_trusted_authority"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(absolute.returncode, 0)
            manifest.write_text(exact_manifest)
            empty_rows = exact_manifest.splitlines()
            empty_rows[2] = empty_rows[2].split("  ", 1)[0] + "  ./"
            manifest.write_text("\n".join(empty_rows) + "\n")
            empty = subprocess.run(
                ["bash", "-c", setup_for_current_manifest() + "\nv_open_trusted_authority"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(empty.returncode, 0)
            manifest.write_text(exact_manifest)
            fd_path = "/proc/self/fd/9" if Path("/proc/self/fd").is_dir() else "/dev/fd/9"
            helper_after_capture = setup_for_current_manifest() + "\n" + "\n".join([
                "w_open_trusted_authority",
                f"printf '%s\\n' \"print('FORGED_HELPER')\" > {str(helper)!r}",
                'exec 7< "$W_ROOT/code/helper.py"',
                f"w_verify_open_fd {fd_path.replace('9', '7')} code/helper.py",
            ])
            helper_changed = subprocess.run(
                ["bash", "-c", helper_after_capture], capture_output=True, text=True,
            )
            self.assertNotEqual(helper_changed.returncode, 0)
            helper.write_text("print('ORIGINAL_HELPER')\n")
            replacement = root / "replacement.sh"
            replacement.write_text("#!/bin/bash\nprintf 'FORGED_LAUNCHER\\n'\n")
            fd_script = setup_for_current_manifest() + "\n" + "\n".join([
                "w_open_trusted_authority",
                f"printf '%s\\n' 'FORGED_MANIFEST_PATH' > {str(manifest)!r}",
                f"W_LAUNCHER={str(launcher)!r}",
                'exec 9< "$W_LAUNCHER"',
                f"w_verify_open_fd {fd_path} code/launcher.sh",
                f"mv {str(replacement)!r} {str(launcher)!r}",
                "w_close_trusted_authority",
                f"/bin/bash --noprofile --norc {fd_path}",
            ])
            bound = subprocess.run(["bash", "-c", fd_script], capture_output=True, text=True)
            self.assertEqual(bound.returncode, 0, bound.stderr)
            self.assertTrue(bound.stdout.endswith("ORIGINAL_LAUNCHER\n"), bound.stdout)
            manifest.write_text(exact_manifest)

    def test_w_invalid_t_receipt_separates_failed_and_recovery_nodes(self):
        receipt_path = Path(__file__).resolve().parent.parent / "invalid-formal-t-job247574-trial1879456.json"
        value = json.loads(receipt_path.read_text())
        self.assertEqual((value["platform_job_id"], value["platform_trial_id"]), (247574, 1879456))
        self.assertEqual(value["status"], "invalid_pre_science_asset_authority_rejection")
        self.assertEqual(value["completed_cells"], 0)
        self.assertFalse(value["gpu_server_started"])
        self.assertFalse(value["paper_evidence"])
        self.assertFalse(value["store_result"])
        self.assertEqual(value["exit_code"], 1)
        self.assertEqual(value["later_platform_event"]["shell_exit_code"], 137)
        self.assertIn("not the cause", value["later_platform_event"]["relation_to_attempt"])
        self.assertEqual(
            (value["w_recovery_boundary"]["platform_job_id"], value["w_recovery_boundary"]["platform_trial_id"]),
            (247699, 1880085),
        )
        self.assertEqual(value["w_recovery_boundary"]["queue_id"], 408)
        self.assertEqual(value["retired_unfrozen_w_trial"]["platform_trial_id"], 1879843)
        self.assertFalse(value["retired_unfrozen_w_trial"]["stop_actor_known"])
        self.assertFalse(value["retired_unfrozen_w_trial"]["bundle_staged"])
        self.assertEqual(value["retired_v_execution_node"]["platform_trial_id"], 1879689)
        self.assertFalse(value["retired_v_execution_node"]["pod_created"])
        self.assertFalse(value["retired_v_execution_node"]["gpu_or_science_execution"])
        self.assertFalse(value["retired_v_freeze"]["gpu_or_science_execution"])
        self.assertEqual(value["retired_empty_u_node"]["platform_trial_id"], 1879665)
        self.assertFalse(value["retired_empty_u_node"]["gpu_or_science_execution"])
        self.assertFalse(value["retired_u_freeze"]["gpu_or_science_execution"])
        self.assertFalse(value["closure_observed_before_later_platform_termination"]["t_run_dir_present"])
        self.assertFalse(value["closure_observed_before_later_platform_termination"]["formal_outputs_present"])


if __name__ == "__main__": unittest.main()
