from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
REPO = PACKAGE.parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


builder = load_module("r39_prereg_test", HERE / "build_r39_preregistration.py")
replay = load_module("r39_replay_test", HERE / "replay_r39_second_model_transfer.py")


class R39SecondModelTransferTests(unittest.TestCase):
    def test_a4_authorities_and_observed_negative(self):
        paths = builder.default_paths(REPO)
        self.assertEqual(builder.sha256_file(paths["a4_static"]), builder.A4_STATIC_SHA256)
        self.assertEqual(builder.sha256_file(paths["a4_aggregate"]), builder.A4_AGGREGATE_SHA256)
        self.assertEqual(builder.sha256_file(paths["a4_qcomem"]), builder.A4_QCOMEM_SHA256)
        evidence_root = REPO / "paper_autonomous_multifork_iteration" / "evidence"
        raw_root = (
            evidence_root
            / "round6_a4_transformers_transfer_20260819b"
            / "results"
            / "raw"
            / "shards"
        )
        relative = []
        for rank in range(8):
            shard = json.loads(
                (raw_root / f"forkaudit-transformers-transfer-shard-{rank}.json").read_text()
            )
            cells = list(shard["fanouts"].values())
            self.assertTrue(all(cell["cross_arm_exact"] for cell in cells))
            cell = next(cell for cell in cells if cell["fanout"] == 1)
            rows = cell["oracle_comparisons"]["deep_materialized"][0]["numeric"]["rows"]
            relative.extend(row["relative_l2"] for row in rows)
        self.assertEqual(len(relative), 16)
        self.assertAlmostEqual(min(relative), 0.01580442957741941)
        self.assertAlmostEqual(max(relative), 0.08068267932341613)
        self.assertTrue(all(value > 0.005 for value in relative))

    def test_static_preregistration_is_deterministic_and_bounded(self):
        first = builder.build_static(REPO)
        second = builder.build_static(REPO)
        self.assertEqual(builder.canonical_bytes(first), builder.canonical_bytes(second))
        self.assertEqual(first["model"]["revision"], builder.MODEL_REVISION)
        self.assertEqual(first["formal_config"]["world_size"], 8)
        self.assertEqual(first["formal_config"]["fanouts"], [1, 2])
        self.assertEqual(first["formal_config"]["split_depth"], 7)
        self.assertEqual(first["formal_config"]["document_tokens"], 64)
        self.assertEqual(first["formal_config"]["query_tokens"], 8)
        self.assertEqual(len(first["rank_inputs"]), 8)
        self.assertEqual(len({row["source_id"] for row in first["rank_inputs"]}), 8)
        self.assertEqual(len(first["controls"]), 4)
        self.assertEqual(
            [row["maximum_status"] for row in first["targets"]],
            ["full", "full", "full", "not_applicable", "partial", "full", "full"],
        )

    def test_dense_adapter_generalizes_mask_route_without_old_constants(self):
        path = HERE / "qwen35_dense_adapter.py"
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        self.assertIn('"qwen3_5_text"', text)
        self.assertIn('"qwen3_5_moe_text"', text)
        self.assertIn("create_recurrent_attention_mask", text)
        self.assertIn("layer_idx=active_full_layer", text)
        self.assertTrue(any(isinstance(node, ast.ClassDef) and node.name == "TorchSplitCausalLM" for node in ast.walk(tree)))
        for source in HERE.glob("*"):
            if source.suffix not in {".py", ".sh"}:
                continue
            body = source.read_text(encoding="utf-8")
            self.assertNotIn("Qwen3.5-35" + "B-A3B", body)
            self.assertNotIn("fourteen" + "-shard", body)
            self.assertNotIn("00014" + "-of-00014", body)

    def test_model_download_is_public_pinned_and_every_file_is_hashed(self):
        text = (HERE / "prepare_r39_model_snapshot.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("token=False"), 2)
        self.assertIn("revision=MODEL_REVISION", text)
        self.assertIn("for path in all_file_paths(root)", text)
        self.assertIn("sha256_file(path)", text)
        self.assertNotIn("HF_TOKEN", text)

    def test_detached_replay_has_no_ml_import(self):
        path = HERE / "replay_r39_second_model_transfer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("torch", imported)
        self.assertNotIn("transformers", imported)
        self.assertNotIn("numpy", imported)

    def test_numeric_replay_and_exactness_are_raw_byte_derived(self):
        reference_raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        candidate_raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        row_a = {"record_id": "a", "offset_bytes": 0, "nbytes": 16, "argmax": 3}
        row_b = {"record_id": "b", "offset_bytes": 0, "nbytes": 16, "argmax": 3}
        result = replay.numeric_comparison(
            candidate_raw,
            row_a,
            reference_raw,
            row_b,
            threshold=0.001,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["exact_bytes"])
        changed = struct.pack("<4f", 1.0, 2.0, 3.0, 5.0)
        result = replay.numeric_comparison(changed, row_a, reference_raw, row_b, threshold=0.001)
        self.assertFalse(result["passed"])
        self.assertFalse(result["exact_bytes"])

    def test_all_four_targeted_controls_replay(self):
        static = builder.build_static(REPO)
        rows = []
        for frozen in static["controls"]:
            predicate = frozen["expected_first_failing_predicate"]
            row = {
                **frozen,
                "matched_clean": {predicate: True},
                "mutant": {predicate: False},
                "classification": "detected_expected_predicate",
            }
            if row["control_id"] == "PREFIX_CONTENT_MUTATION":
                row["storage_identity_stable"] = True
            if row["control_id"] == "MUTABLE_CACHE_ALIAS":
                row["mutant_overlap_ranges"] = [{"intersection_start_bytes": 0}]
            rows.append(row)
        self.assertTrue(replay.validate_controls({"controls": rows}, static))
        rows[2]["mutant"] = {"POSITION_CANONICAL": True}
        rows[2]["classification"] = "escaped_or_clean_failure"
        self.assertFalse(replay.validate_controls({"controls": rows}, static))

    def test_frozen_files_exist_and_bootstrap_has_no_placeholder(self):
        for relative in (
            "preregistration/static-preregistration.json",
            "preregistration/source-manifest.json",
            "preregistration/freeze.json",
        ):
            self.assertTrue((PACKAGE / relative).is_file(), relative)
        wrapper = (HERE / "launch_trial_1907358.sh").read_text(encoding="utf-8")
        self.assertNotIn("__R39_", wrapper)
        freeze = json.loads((PACKAGE / "preregistration" / "freeze.json").read_text())
        self.assertEqual(
            hashlib.sha256((PACKAGE / "preregistration" / "static-preregistration.json").read_bytes()).hexdigest(),
            freeze["static_manifest_sha256"],
        )
        self.assertEqual(
            hashlib.sha256((PACKAGE / "preregistration" / "source-manifest.json").read_bytes()).hexdigest(),
            freeze["source_manifest_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
