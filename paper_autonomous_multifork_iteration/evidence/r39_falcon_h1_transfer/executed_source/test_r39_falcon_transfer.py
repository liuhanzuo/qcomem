#!/usr/bin/env python3
"""GPU-free fail-closed tests for the frozen Falcon-H1 R39 package."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import inspect
import json
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
REPO = PACKAGE.parents[2]
PREREG = PACKAGE / "preregistration"
A4_QCOMEM = (
    REPO
    / "paper_autonomous_multifork_iteration"
    / "evidence"
    / "round6_a4_transformers_transfer_20260819b"
    / "executed_source"
    / "qcomem_torch.py"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


builder = load_module("r39_falcon_prereg_test", HERE / "build_r39_falcon_preregistration.py")
prepare = load_module("r39_falcon_prepare_test", HERE / "prepare_r39_falcon_snapshot.py")
replay = load_module("r39_falcon_replay_test", HERE / "replay_r39_falcon_transfer.py")
adapter = load_module("r39_falcon_adapter_test", HERE / "falcon_h1_adapter.py")


class FakeTransportError(Exception):
    pass


class FakeResponse:
    def __init__(self, body: bytes, *, status: int = 200, content_length: int | None = None):
        self.body = body
        self.status_code = status
        self.headers = {"Content-Length": str(len(body) if content_length is None else content_length)}
        self.history = []
        self.url = "https://modelscope.cn/api/v1/models/tiiuae/Falcon-H1-0.5B-Base/repo"

    def __enter__(self):
        return self

    def __exit__(self, _kind, _value, _traceback):
        return False

    def iter_raw(self, chunk_size: int):
        for offset in range(0, len(self.body), max(1, chunk_size)):
            yield self.body[offset : offset + chunk_size]


class FakeClient:
    def __init__(self, plans):
        self.plans = list(plans)
        self.requests = []

    def stream(self, method, url, *, params, headers):
        self.requests.append({"method": method, "url": url, "params": params, "headers": dict(headers)})
        plan = self.plans.pop(0)
        if isinstance(plan, BaseException):
            raise plan
        return plan


class FalconH1PackageTests(unittest.TestCase):
    maxDiff = None

    def test_static_is_a_bytewise_fresh_rebuild(self):
        static_path = PREREG / "static-preregistration.json"
        rebuilt = builder.build_static(REPO)
        self.assertEqual(static_path.read_bytes(), builder.canonical_bytes(rebuilt))
        builder.verify_static(load_json(static_path), rebuilt)
        self.assertEqual(sha256_file(static_path), "eeac9ac53266e5e3312defc6dc4a96ab069fc99704424566c49a09a497b777fc")

    def test_frozen_falcon_identity_geometry_runtime_and_scope(self):
        static = builder.build_static(REPO)
        self.assertEqual(static["model"]["repo_id"], "tiiuae/Falcon-H1-0.5B-Base")
        self.assertEqual(static["model"]["revision"], "59fb76e8c5d3fc7441b062be638e1ba0afd5c687")
        geometry = static["model"]["expected_geometry"]
        self.assertEqual(geometry["model_type"], "falcon_h1")
        self.assertEqual(geometry["num_hidden_layers"], 36)
        self.assertEqual(geometry["hidden_size"], 1024)
        self.assertEqual(geometry["model_vocab_size"], 32784)
        self.assertEqual(geometry["tokenizer_vocab_size"], 32768)
        self.assertEqual(geometry["layer_types"], ["hybrid"] * 36)
        self.assertEqual(geometry["state_families_per_layer"], ["kv_key", "kv_value", "conv", "mamba2_recurrent"])
        self.assertEqual(geometry["expected_state_family_count_full_model"], 144)
        formal = static["formal_config"]
        self.assertEqual(formal["world_size"], 8)
        self.assertEqual(formal["fanouts"], [1, 2])
        self.assertEqual(formal["split_depth"], 18)
        self.assertEqual(formal["chunk_schedule"], [64, 8, 1])
        self.assertEqual(formal["semantic_steps"], 2)
        self.assertTrue(formal["q16"]["lossless_only"])
        self.assertTrue(formal["q16"]["no_differential_family_quantization_claim"])
        runtime = static["runtime"]
        self.assertEqual(runtime["transformers_version"], "5.14.1")
        self.assertEqual(runtime["registered_image_label"], "vllm-cu129-v1")
        self.assertEqual(runtime["attention_implementation"], "eager")
        self.assertEqual(runtime["mamba_dispatch"], "official FalconH1Mixer.torch_forward")
        self.assertEqual(runtime["hub_kernels_environment"], "USE_HUB_KERNELS=NO")
        self.assertTrue(runtime["hub_kernels_disabled_before_transformers_import"])
        self.assertTrue(runtime["fast_path_forced_false_after_model_init_before_any_forward"])
        self.assertFalse(runtime["package_installs_mamba_causal_conv_or_flash_dependencies"])
        self.assertEqual(runtime["official_source_sha256"], {
            "modeling_falcon_h1.py": "e90bf774524e9b66284ad1c5528c35339271a187f58f16ba2d45c97f4bc6b5bd",
            "cache_utils.py": "ee7902fbd031ed332b5e26d07756a33f09b5c90a435b8363b9330876dc33ce0e",
            "masking_utils.py": "5f48e428ea02d1b6008acb45c147fcdb4eba89deea69627744662aa05da1b9f2",
        })
        self.assertIn("runtime independence", static["claim_boundary"]["not_authorized"][0])
        self.assertIn("compiled dispatch", static["claim_boundary"]["not_authorized"])

    def test_inputs_are_retokenized_falcon_pg19_and_never_reselected(self):
        static = builder.build_static(REPO)
        inputs_path = PREREG / "pg19-tokenized-inputs.json"
        inputs = load_json(inputs_path)
        self.assertEqual(sha256_file(inputs_path), builder.PG19_INPUTS_SHA256)
        self.assertEqual(inputs["tokenizer"]["sha256"], builder.TOKENIZER_SHA256)
        self.assertEqual(inputs["tokenizer"]["derivation_library_version"], "0.22.2")
        self.assertFalse(inputs["tokenizer"]["add_special_tokens"])
        self.assertTrue(inputs["window_rule"]["no_out_of_vocabulary_filtering_or_reselection"])
        self.assertEqual(len(static["rank_inputs"]), 8)
        self.assertEqual(len({row["source_id"] for row in static["rank_inputs"]}), 8)
        all_values = []
        for rank, row in enumerate(static["rank_inputs"]):
            self.assertEqual(row["rank"], rank)
            self.assertEqual(len(row["document_token_ids"]), 64)
            self.assertEqual(len(row["queries"]), 2)
            values = list(row["document_token_ids"])
            for query in row["queries"]:
                self.assertEqual(len(query["token_ids"]), 8)
                values.extend(query["token_ids"])
            self.assertTrue(all(0 <= token < 32784 for token in values))
            all_values.extend(values)
        self.assertLess(max(all_values), 32784)

    def test_model_trees_and_cross_source_hashes_are_bound(self):
        authorities = builder.validate_model_authorities(REPO)
        hf, ms, cross = authorities["hf"], authorities["ms"], authorities["cross"]
        self.assertEqual(sha256_file(PREREG / "huggingface-tree.json"), builder.HF_TREE_SHA256)
        self.assertEqual(sha256_file(PREREG / "modelscope-tree.json"), builder.MS_TREE_SHA256)
        self.assertEqual(sha256_file(PREREG / "cross-source-equivalence.json"), builder.CROSS_SOURCE_SHA256)
        self.assertEqual(hf["file_count"], 8)
        self.assertEqual(ms["file_count"], 9)
        self.assertEqual(cross["common_exact_file_count"], 7)
        self.assertTrue(cross["scientific_load_files_exact_across_sources"])
        for tree in (hf, ms):
            by_path = {row["path"]: row for row in tree["files"]}
            self.assertEqual(by_path["model.safetensors"]["sha256"], builder.WEIGHT_SHA256)
            self.assertEqual(by_path["tokenizer.json"]["sha256"], builder.TOKENIZER_SHA256)
        self.assertEqual(sha256_file(A4_QCOMEM), builder.A4_QCOMEM_SHA256)

    def test_modelscope_downloader_is_zero_origin_http200_and_atomic(self):
        policy = prepare.acquisition_policy()
        self.assertEqual(policy, builder.acquisition_policy())
        self.assertEqual(policy["official_namespace"], "tiiuae")
        self.assertEqual(policy["modelscope_revision"], builder.MS_REVISION)
        self.assertTrue(policy["restart_from_zero_per_attempt"])
        self.assertTrue(policy["range_requests_forbidden"])
        self.assertTrue(policy["append_to_partial_forbidden"])
        self.assertTrue(policy["fresh_nonoverwriting_snapshot_required"])
        source = (HERE / "prepare_r39_falcon_snapshot.py").read_text(encoding="utf-8")
        self.assertIn("tempfile.mkstemp", source)
        self.assertIn('part.open("wb")', source)
        self.assertIn("os.replace(part, destination)", source)
        self.assertNotIn('headers={"Range"', source)
        self.assertNotIn('part.open("ab")', source)
        self.assertNotIn("snapshot_download", source)
        self.assertNotIn("huggingface_hub", source)
        self.assertNotIn("pip install", source)

    def test_complete_http200_contract_fails_closed(self):
        prepare.validate_full_response(200, {"content-length": "10"}, 10)
        invalid = (
            (206, {"content-length": "10"}),
            (200, {}),
            (200, {"content-length": "9"}),
            (200, {"content-length": "ten"}),
            (200, {"content-length": "10", "content-range": "bytes 0-9/10"}),
        )
        for status, headers in invalid:
            with self.subTest(status=status, headers=headers):
                with self.assertRaises(prepare.RetryableDownloadError):
                    prepare.validate_full_response(status, headers, 10)

    def test_failed_download_restarts_from_zero_in_a_new_temp_file(self):
        payload = b"falcon-official-complete-body"
        row = {"path": "nested/model-fragment.bin", "size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        plans = [FakeResponse(payload[:-2], content_length=len(payload)), FakeResponse(payload, content_length=len(payload))]
        with tempfile.TemporaryDirectory() as raw, mock.patch.object(prepare.time, "sleep"):
            root = Path(raw)
            client = FakeClient(plans)
            receipt = prepare.download_file(client, row, root, (FakeTransportError,))
            self.assertEqual((root / row["path"]).read_bytes(), payload)
            self.assertEqual(receipt["attempts"], 2)
            self.assertEqual(receipt["failed_attempts"], 1)
            self.assertEqual(len(client.requests), 2)
            self.assertTrue(all("Range" not in request["headers"] and "range" not in request["headers"] for request in client.requests))
            self.assertEqual(list((root / ".r39-zero-origin-attempts").iterdir()), [])

    def test_suffix_mask_uses_layer18_history_and_matches_direct_official_call(self):
        import torch

        calls: list[int] = []
        masking = types.ModuleType("transformers.masking_utils")

        def create_causal_mask(*, inputs_embeds, past_key_values, layer_idx, **_kwargs):
            calls.append(int(layer_idx))
            query_length = int(inputs_embeds.shape[1])
            history = int(past_key_values.get_seq_length(layer_idx))
            return torch.arange(query_length * (history + query_length), dtype=torch.int64).reshape(1, 1, query_length, history + query_length)

        def create_recurrent_attention_mask(*, inputs_embeds, **_kwargs):
            return torch.ones((1, int(inputs_embeds.shape[1])), dtype=torch.bool)

        masking.create_causal_mask = create_causal_mask
        masking.create_recurrent_attention_mask = create_recurrent_attention_mask
        transformers = types.ModuleType("transformers")
        transformers.__path__ = []
        transformers.masking_utils = masking

        class SuffixCache:
            def get_seq_length(self, layer_idx=0):
                return 64 if int(layer_idx) == 18 else 0

        fake_self = SimpleNamespace(
            config=SimpleNamespace(),
            language_model=SimpleNamespace(rotary_emb=lambda hidden, position_ids: (hidden, position_ids)),
        )
        hidden = torch.zeros((1, 8, 1024), dtype=torch.bfloat16)
        cache = SuffixCache()
        with mock.patch.dict(sys.modules, {"transformers": transformers, "transformers.masking_utils": masking}):
            candidate = adapter.TorchSplitFalconH1._layer_context(
                fake_self, hidden, past_key_values=cache, position_offset=64, layer_start=18
            )
            direct_official = create_causal_mask(
                config=fake_self.config,
                inputs_embeds=hidden,
                attention_mask=None,
                past_key_values=cache,
                position_ids=candidate.position_ids,
                layer_idx=18,
            )
        self.assertEqual(calls, [18, 18])
        self.assertEqual(candidate.masks["full_attention"].shape[-1], 72)
        self.assertTrue(torch.equal(candidate.masks["full_attention"], direct_official))
        self.assertEqual(candidate.position_ids.tolist(), [list(range(64, 72))])

    def test_adapter_ast_binds_mask_to_split_layer_start(self):
        source = (HERE / "falcon_h1_adapter.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        matches = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "create_causal_mask":
                values = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg}
                matches.append(values.get("layer_idx"))
        self.assertEqual(len(matches), 1)
        self.assertIsInstance(matches[0], ast.Name)
        self.assertEqual(matches[0].id, "layer_start")
        self.assertNotIn("del layer_start", source)
        self.assertIn("self.initial_naive_dispatch_receipt = force", source)
        self.assertIn("naive = dict(self.initial_naive_dispatch_receipt)", source)

    def test_reference_is_candidate_import_free_and_dynamic_execution_free(self):
        path = HERE / "run_r39_falcon_reference.py"
        source = path.read_text(encoding="utf-8")
        reference = load_module("r39_falcon_reference_ast_test", path)
        self.assertTrue(reference.source_is_candidate_import_free(source))
        marker = "from __future__ import annotations\n"
        mutant = source.replace(
            marker,
            marker + "from falcon_h1_adapter import TorchSplitFalconH1\n",
            1,
        )
        compile(mutant, "<reference-import-mutant-test>", "exec")
        self.assertFalse(reference.source_is_candidate_import_free(mutant))
        tree = ast.parse(source)
        imports, calls = [], []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module or "")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    parts, current = [], node.func
                    while isinstance(current, ast.Attribute):
                        parts.append(current.attr)
                        current = current.value
                    if isinstance(current, ast.Name):
                        parts.append(current.id)
                    calls.append(".".join(reversed(parts)))
        for banned in ("falcon_h1_adapter", "run_r39_falcon_candidate", "qcomem_torch"):
            self.assertFalse(any(name == banned or name.startswith(banned + ".") for name in imports))
        for banned in ("__import__", "exec", "eval", "importlib.import_module", "importlib.util.spec_from_file_location"):
            self.assertNotIn(banned, calls)

    def test_zero_tolerance_family_complete_and_frozen_controls(self):
        static = builder.build_static(REPO)
        reference = static["reference_contract"]
        self.assertEqual(reference["max_abs_threshold"], 0.0)
        self.assertEqual(reference["relative_l2_threshold"], 0.0)
        self.assertTrue(reference["full_fp32_logit_bytes_exact"])
        self.assertTrue(reference["all_144_state_family_content_sha256_exact_per_request_step"])
        self.assertEqual([row["control_id"] for row in static["controls"]], [
            "MUTABLE_CACHE_ALIAS",
            "STATE_FAMILY_OMISSION",
            "POSITION_OFFSET_DRIFT",
            "STATE_FAMILY_RELABEL",
            "REFERENCE_CANDIDATE_IMPORT",
        ])
        self.assertTrue(static["persistent_base_detector"]["before_after_full_content_sha256_exact"])
        replay_parameters = set(inspect.signature(replay.replay_rank).parameters)
        self.assertTrue(
            {
                "expected_static_sha256",
                "expected_source_sha256",
                "expected_model_authority_sha256",
                "expected_gpu_assignment_sha256",
                "expected_gpu_row",
                "expected_input_row",
            }.issubset(replay_parameters)
        )

    def test_family_validator_requires_all_144_bound_rows(self):
        rows = []
        for layer in range(36):
            for family in replay.FAMILY_ORDER:
                shape = {
                    "kv_key": [1, 2, 72, 64],
                    "kv_value": [1, 2, 72, 64],
                    "conv": [1, 1792, 4],
                    "mamba2_recurrent": [1, 24, 64, 128],
                }[family]
                rows.append({
                    "layer_index": layer,
                    "family": family,
                    "shape": shape,
                    "dtype": "torch.float32" if family == "mamba2_recurrent" else "torch.bfloat16",
                    "content_sha256": hashlib.sha256(f"{layer}:{family}".encode()).hexdigest(),
                })
        receipt = {
            "schema_version": "r39-falcon-h1-composed-state-family-receipt-v1",
            "split_depth": 18,
            "expected_sequence_length": 72,
            "complete": True,
            "expected_family_count": 144,
            "observed_family_count": 144,
            "rows": rows,
            "rows_sha256": replay.sha256_bytes(replay.canonical_bytes(rows)),
        }
        replay.validate_family_receipt(receipt, 72)
        broken = dict(receipt)
        broken["rows"] = rows[:-1]
        with self.assertRaises(ValueError):
            replay.validate_family_receipt(broken, 72)

    def test_launcher_is_nonoverwriting_reference_first_and_has_terminal_closure(self):
        source = (HERE / "launch_r39_falcon_transfer_8gpu.sh").read_text(encoding="utf-8")
        self.assertIn('[[ ! -e "$RUN_ROOT" ]]', source)
        self.assertIn('[[ ! -e "$MODEL_ROOT" ]]', source)
        self.assertIn("PYTHONPYCACHEPREFIX", source)
        self.assertLess(source.index("export USE_HUB_KERNELS=NO"), source.index('"$PYTHON" -B "$TESTS"'))
        self.assertIn("PYTHONPATH= \\\n", source)
        self.assertIn('"$PYTHON" -I -B "$REFERENCE"', source)
        self.assertIn("[[ $status -ne 0 ]] || status=1", source)
        self.assertLess(
            source.index("trap fail_closed ERR INT TERM"),
            source.index("bootstrap-source-verification.json"),
        )
        self.assertLess(
            source.index("bootstrap-source-verification.json"),
            source.index('"$PYTHON" -B "$TESTS"'),
        )
        self.assertEqual(source.count('"$REFERENCE"'), 2)
        self.assertEqual(source.count('"$CANDIDATE"'), 2)
        self.assertLess(source.index('"$REFERENCE" \\\n'), source.index('"$CANDIDATE" \\\n'))
        self.assertLess(source.index("20_reference_complete"), source.index('"$CANDIDATE" \\\n'))
        self.assertLess(source.index("40_replay_complete"), source.index("artifact-ledger.json"))
        self.assertLess(source.index("artifact-ledger.json"), source.index("TERMINAL.sha256"))
        self.assertLess(source.index("TERMINAL.sha256"), source.index('> "$RUN_ROOT/COMPLETE"'))
        lowered = source.lower()
        self.assertIsNone(re.search(r"(?m)^\s*qs(?:\s|$)", lowered))
        self.assertNotIn("pip install", lowered)
        self.assertNotIn("conda install", lowered)

    def test_source_and_freeze_manifests_rebuild_exactly(self):
        source_path = PREREG / "source-manifest.json"
        freeze_path = PREREG / "freeze.json"
        self.assertTrue(source_path.is_file())
        self.assertTrue(freeze_path.is_file())
        source = load_json(source_path)
        rebuilt_source = builder.build_source(REPO, [row["path"] for row in source["files"]])
        self.assertEqual(source_path.read_bytes(), builder.canonical_bytes(rebuilt_source))
        self.assertEqual(source["file_count"], len(source["files"]))
        paths = [row["path"] for row in source["files"]]
        self.assertEqual(paths, sorted(paths))
        self.assertIn("paper_autonomous_multifork_iteration/evidence/r39_falcon_h1_transfer/executed_source/falcon_h1_adapter.py", paths)
        self.assertIn("paper_autonomous_multifork_iteration/evidence/round6_a4_transformers_transfer_20260819b/executed_source/qcomem_torch.py", paths)
        self.assertNotIn("paper_autonomous_multifork_iteration/evidence/r39_falcon_h1_transfer/preregistration/source-manifest.json", paths)
        freeze = load_json(freeze_path)
        self.assertEqual(freeze_path.read_bytes(), builder.canonical_bytes(builder.build_freeze(PACKAGE)))
        self.assertEqual(freeze["static_manifest_sha256"], sha256_file(PREREG / "static-preregistration.json"))
        self.assertEqual(freeze["source_manifest_sha256"], sha256_file(source_path))
        self.assertTrue(freeze["remote_paths"]["nonoverwriting_required"])
        self.assertEqual(freeze["gpu_execution_status"], "not_run_at_freeze")
        self.assertFalse(freeze["scientific_outputs_inspected_before_freeze"])

    def test_no_stale_predecessor_protocol_survives_in_package(self):
        banned = (
            "Q" + "wen/Q" + "wen3.5-0.8B",
            "q" + "wen3_5_text",
            "q" + "wen3_5_moe_text",
            "split depth " + "7",
            "depth=" + "7",
            "248" + "320",
        )
        for path in sorted(PACKAGE.rglob("*")):
            if not path.is_file() or path.suffix not in {".py", ".sh", ".md", ".json"}:
                continue
            text = path.read_text(encoding="utf-8")
            for token in banned:
                self.assertNotIn(token, text, f"stale token {token!r} in {path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
