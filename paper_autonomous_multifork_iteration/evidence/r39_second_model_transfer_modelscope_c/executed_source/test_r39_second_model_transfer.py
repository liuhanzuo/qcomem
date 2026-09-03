from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import stat
import struct
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
REPO = PACKAGE.parents[2]
EVIDENCE = REPO / "paper_autonomous_multifork_iteration" / "evidence"
A_PACKAGE = EVIDENCE / "r39_second_model_transfer"
B_PACKAGE = EVIDENCE / "r39_second_model_transfer_mirror_b"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def without_acquisition(value: dict):
    result = dict(value)
    result.pop("model_acquisition", None)
    return result


def runtime_suffix(path: Path) -> bytes:
    text = path.read_text(encoding="utf-8")
    start = text.index("    torch.cuda.set_device(0)\n")
    end = text.index("\ndef parse_args()", start)
    return text[start:end].encode()


builder = load_module("r39_c_prereg_test", HERE / "build_r39_preregistration.py")
prepare = load_module("r39_c_prepare_test", HERE / "prepare_r39_model_snapshot.py")
replay = load_module("r39_c_replay_test", HERE / "replay_r39_second_model_transfer.py")


class R39SecondModelTransferModelScopeCTests(unittest.TestCase):
    def test_a4_authorities_and_observed_negative(self):
        paths = builder.default_paths(REPO)
        self.assertEqual(builder.sha256_file(paths["a4_static"]), builder.A4_STATIC_SHA256)
        self.assertEqual(builder.sha256_file(paths["a4_aggregate"]), builder.A4_AGGREGATE_SHA256)
        self.assertEqual(builder.sha256_file(paths["a4_qcomem"]), builder.A4_QCOMEM_SHA256)
        raw_root = EVIDENCE / "round6_a4_transformers_transfer_20260819b" / "results" / "raw" / "shards"
        relative = []
        for rank in range(8):
            shard = load_json(raw_root / f"forkaudit-transformers-transfer-shard-{rank}.json")
            cells = list(shard["fanouts"].values())
            self.assertTrue(all(cell["cross_arm_exact"] for cell in cells))
            cell = next(cell for cell in cells if cell["fanout"] == 1)
            rows = cell["oracle_comparisons"]["deep_materialized"][0]["numeric"]["rows"]
            relative.extend(row["relative_l2"] for row in rows)
        self.assertEqual(len(relative), 16)
        self.assertAlmostEqual(min(relative), 0.01580442957741941)
        self.assertAlmostEqual(max(relative), 0.08068267932341613)
        self.assertTrue(all(value > 0.005 for value in relative))

    def test_static_scientific_object_equals_a_and_b_after_acquisition_strip(self):
        current = builder.build_static(REPO)
        a_static = load_json(A_PACKAGE / "preregistration" / "static-preregistration.json")
        b_static = load_json(B_PACKAGE / "preregistration" / "static-preregistration.json")
        self.assertEqual(builder.canonical_bytes(without_acquisition(current)), builder.canonical_bytes(a_static))
        self.assertEqual(builder.canonical_bytes(without_acquisition(current)), builder.canonical_bytes(without_acquisition(b_static)))
        self.assertEqual(current["formal_config"]["world_size"], 8)
        self.assertEqual(current["formal_config"]["fanouts"], [1, 2])
        self.assertEqual(current["formal_config"]["split_depth"], 7)
        self.assertEqual(current["formal_config"]["document_tokens"], 64)
        self.assertEqual(current["formal_config"]["query_tokens"], 8)
        self.assertEqual(current["runtime"]["transformers_version"], "5.14.1")
        self.assertEqual(len(current["rank_inputs"]), 8)
        self.assertEqual(len(current["controls"]), 4)
        self.assertEqual(current["model_acquisition"], prepare.acquisition_policy())

    def test_scientific_execution_suffix_and_adapter_are_unchanged(self):
        c_runner = HERE / "run_r39_second_model_transfer.py"
        a_runner = A_PACKAGE / "executed_source" / "run_r39_second_model_transfer.py"
        b_runner = B_PACKAGE / "executed_source" / "run_r39_second_model_transfer.py"
        self.assertEqual(runtime_suffix(c_runner), runtime_suffix(a_runner))
        self.assertEqual(runtime_suffix(c_runner), runtime_suffix(b_runner))
        c_adapter = HERE / "qwen35_dense_adapter.py"
        self.assertEqual(sha256_file(c_adapter), sha256_file(A_PACKAGE / "executed_source" / c_adapter.name))
        self.assertEqual(sha256_file(c_adapter), sha256_file(B_PACKAGE / "executed_source" / c_adapter.name))

    def test_frozen_official_modelscope_tree_and_cross_source_hashes(self):
        tree_path = PACKAGE / "preregistration" / "modelscope-tree.json"
        tree = prepare.load_frozen_tree(tree_path)
        self.assertEqual(sha256_file(tree_path), prepare.FROZEN_TREE_SHA256)
        self.assertEqual(tree["official_source"]["revision"], prepare.MODELSCOPE_REVISION)
        self.assertEqual(tree["canonical_huggingface_identity"]["revision"], prepare.HF_MODEL_REVISION)
        self.assertEqual(len(tree["files"]), 14)
        by_path = {row["path"]: row for row in tree["files"]}
        self.assertEqual(by_path[prepare.WEIGHT_PATH]["sha256"], prepare.WEIGHT_SHA256)
        self.assertEqual(by_path[prepare.TOKENIZER_PATH]["sha256"], prepare.TOKENIZER_SHA256)
        policy = prepare.acquisition_policy()
        self.assertEqual(policy["endpoint"], "https://modelscope.cn")
        self.assertEqual(policy["official_namespace"], "Qwen")
        self.assertIs(policy["token"], False)
        self.assertTrue(policy["a_failed_partial_reuse_forbidden"])
        self.assertTrue(policy["b_failed_partial_reuse_forbidden"])

    def test_downloader_is_pinned_resumable_per_file_and_not_huggingface(self):
        text = (HERE / "prepare_r39_model_snapshot.py").read_text(encoding="utf-8")
        self.assertIn('params={"Revision": MODELSCOPE_REVISION, "Recursive": "true"}', text)
        self.assertIn('"Range": f"bytes={offset}-"', text)
        self.assertIn("sha256_file(part) == row[\"sha256\"]", text)
        self.assertIn("row[\"size\"]", text)
        self.assertIn("require_official_https_response", text)
        self.assertNotIn("snapshot_download", text)
        self.assertNotIn("huggingface_hub", text)
        self.assertNotIn("HF_TOKEN", text)
        self.assertNotIn("20260826a", text)
        self.assertNotIn("20260826b", text)

    def test_range_contract_fails_closed(self):
        prepare.validate_range_response(206, {"content-range": "bytes 3-9/10", "content-length": "7"}, 3, 10)
        prepare.validate_range_response(200, {"content-length": "10"}, 0, 10)
        with self.assertRaises(ValueError):
            prepare.validate_range_response(200, {"content-length": "10"}, 3, 10)
        with self.assertRaises(ValueError):
            prepare.validate_range_response(206, {"content-range": "bytes 2-9/10"}, 3, 10)
        with self.assertRaises(ValueError):
            prepare.validate_range_response(206, {"content-range": "bytes 3-9/11"}, 3, 10)

    def test_finalize_file_requires_exact_size_and_sha(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            payload = b"immutable-model-byte-test"
            row = {"path": "nested/model.bin", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            part = root / "part"
            part.write_bytes(payload)
            destination = root / row["path"]
            prepare.finalize_file(part, destination, row)
            self.assertEqual(destination.read_bytes(), payload)
            bad = root / "bad"
            bad.write_bytes(payload + b"x")
            with self.assertRaises(ValueError):
                prepare.finalize_file(bad, root / "never", row)

    def test_read_only_closure_covers_files_and_directories(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw) / "model"
            (root / "nested").mkdir(parents=True)
            (root / "nested" / "file").write_bytes(b"x")
            prepare.make_read_only(root)
            for path in [root, root / "nested", root / "nested" / "file"]:
                self.assertEqual(stat.S_IMODE(path.stat().st_mode) & 0o222, 0)

    def test_dense_adapter_generalizes_mask_route_without_old_constants(self):
        text = (HERE / "qwen35_dense_adapter.py").read_text(encoding="utf-8")
        tree = ast.parse(text)
        self.assertIn('"qwen3_5_text"', text)
        self.assertIn("create_recurrent_attention_mask", text)
        self.assertTrue(any(isinstance(node, ast.ClassDef) and node.name == "TorchSplitCausalLM" for node in ast.walk(tree)))

    def test_detached_replay_has_no_ml_import(self):
        tree = ast.parse((HERE / "replay_r39_second_model_transfer.py").read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("torch", imported)
        self.assertNotIn("transformers", imported)
        self.assertNotIn("numpy", imported)

    def test_numeric_replay_and_all_controls(self):
        raw = struct.pack("<4f", 1.0, 2.0, 3.0, 4.0)
        row = {"record_id": "a", "offset_bytes": 0, "nbytes": 16, "argmax": 3}
        result = replay.numeric_comparison(raw, row, raw, row, threshold=0.001)
        self.assertTrue(result["passed"])
        static = builder.build_static(REPO)
        controls = []
        for frozen in static["controls"]:
            predicate = frozen["expected_first_failing_predicate"]
            control = {
                **frozen,
                "matched_clean": {predicate: True},
                "mutant": {predicate: False},
                "classification": "detected_expected_predicate",
            }
            if control["control_id"] == "PREFIX_CONTENT_MUTATION":
                control["storage_identity_stable"] = True
            if control["control_id"] == "MUTABLE_CACHE_ALIAS":
                control["mutant_overlap_ranges"] = [{"intersection_start_bytes": 0}]
            controls.append(control)
        self.assertTrue(replay.validate_controls({"controls": controls}, static))

    def test_frozen_manifests_equivalence_receipt_and_c_paths(self):
        for relative in (
            "preregistration/modelscope-tree.json",
            "preregistration/scientific-equivalence.json",
            "preregistration/static-preregistration.json",
            "preregistration/source-manifest.json",
            "preregistration/freeze.json",
        ):
            self.assertTrue((PACKAGE / relative).is_file(), relative)
        freeze = load_json(PACKAGE / "preregistration" / "freeze.json")
        wrapper = (HERE / "launch_trial_1907355_modelscope_c.sh").read_text(encoding="utf-8")
        self.assertNotIn("__R39_", wrapper)
        self.assertIn("qcomem_r39_second_model_transfer_20260826c", wrapper)
        self.assertIn(prepare.HF_MODEL_REVISION, wrapper)
        self.assertIn(prepare.MODELSCOPE_REVISION, wrapper)
        self.assertNotIn("20260826a", wrapper)
        self.assertNotIn("20260826b", wrapper)
        self.assertEqual(freeze["acquisition_variant"], "modelscope-official-revision-pinned-per-file-c")
        self.assertEqual(freeze["model_acquisition"], prepare.acquisition_policy())
        self.assertEqual(sha256_file(PACKAGE / "preregistration" / "static-preregistration.json"), freeze["static_manifest_sha256"])
        self.assertEqual(sha256_file(PACKAGE / "preregistration" / "source-manifest.json"), freeze["source_manifest_sha256"])
        self.assertEqual(sha256_file(PACKAGE / "preregistration" / "scientific-equivalence.json"), freeze["scientific_equivalence_sha256"])
        self.assertIn(freeze["static_manifest_sha256"], wrapper)
        self.assertIn(freeze["source_manifest_sha256"], wrapper)


if __name__ == "__main__":
    unittest.main()
