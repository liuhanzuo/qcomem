from __future__ import annotations

import copy
import hashlib
import importlib.machinery
import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

import build_qcomem_forkaudit_rr2_input_manifest as rr2
import build_qcomem_qwen35_forkaudit_review_manifest as builder
import qcomem_forkaudit_model_load_lease as model_lease
import run_qcomem_qwen35_forkaudit_review_revision as runner
import test_build_qcomem_forkaudit_rr2_input_manifest as rr2_fixture


GPU_DIR = Path(__file__).resolve().parent
LAUNCHER = GPU_DIR / "launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh"
BUILDER = GPU_DIR / "build_qcomem_qwen35_forkaudit_review_manifest.py"
PRIOR_CAPACITY = GPU_DIR / "qcomem_qwen35_vllm_paged_multifork_resident_protocol_manifest.json"
PRIOR_FP32_CONTEXT = (
    GPU_DIR.parent
    / "paper_autonomous_multifork_iteration/evidence/forkaudit_fp32_calibration_manifest.json"
)
REVIEW_RESPONSE_PLAN = (
    GPU_DIR.parent
    / "paper_autonomous_multifork_iteration/review/experiment_response_plan.json"
)


def digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(runner.canonical_json_bytes(value) + b"\n")


def write_ledger(path: Path, names: list[str]) -> None:
    names = sorted(names, key=lambda item: item.encode("utf-8"))
    path.write_text(
        "".join(f"{digest(name)}  {name}\n" for name in names),
        encoding="utf-8",
    )


def write_actual_model_ledger(path: Path, model: Path, names: list[str]) -> None:
    names = sorted(names, key=lambda item: item.encode("utf-8"))
    path.write_text(
        "".join(
            f"{hashlib.sha256((model / name).read_bytes()).hexdigest()}  {name}\n"
            for name in names
        ),
        encoding="utf-8",
    )


def rr2_input_manifest_fixture(
    *,
    model: Path,
    pg19_data: Path,
    pg19_manifest: Path,
    prior_capacity: Path,
) -> dict[str, object]:
    banks = frozen_query_banks()
    plan = selection_plan()
    prior = rr2.parse_prior_capacity_manifest(
        prior_capacity.read_bytes(), expectations=rr2.FORMAL_EXPECTATIONS
    )
    model_artifacts = rr2.audit_model_tokenizer_artifacts(model)
    data_sha = hashlib.sha256(pg19_data.read_bytes()).hexdigest()
    data_manifest_sha = hashlib.sha256(pg19_manifest.read_bytes()).hexdigest()
    windows = []
    for rank, bank in enumerate(banks):
        document_end = bank["document_end_token_exclusive"]
        windows.append(
            {
                "rank": rank,
                "book_index": rank,
                "source_id": bank["source_id"],
                "source_object": bank["source_object"],
                "window_index": bank["window_index"],
                "document_start_token": bank["document_start_token"],
                "document_end_token_exclusive": document_end,
                "document_length": rr2.FORMAL_DOCUMENT_TOKENS,
                "document_token_ids_sha256": bank["document_token_ids_sha256"],
                "adjacent_calibration_query_start_token": document_end,
                "adjacent_calibration_query_end_token_exclusive": (
                    document_end + rr2.FORMAL_QUERY_TOKENS
                ),
                "adjacent_calibration_query_token_ids_sha256": digest(
                    f"adjacent-query-rank-{rank}"
                ),
                "query_bank_manifest_sha256": bank["manifest_sha256"],
                "absent_from_prior_capacity_source_start_pairs": True,
                "absent_from_prior_capacity_coordinates": True,
            }
        )
    prefixes = [row for bank in banks for row in rr2._prefix_rows(bank)]
    value = {
        "schema_version": rr2.SCHEMA_VERSION,
        "protocol": rr2.PROTOCOL,
        "manifest_role": "pre-output deterministic PG19 inputs; contains no candidate outputs",
        "model": {
            "model_id": rr2.FORMAL_MODEL_ID,
            "model_revision": rr2.FORMAL_MODEL_REVISION,
            "local_files_only": True,
            "model_and_tokenizer_artifacts": model_artifacts,
            "tokenizer_runtime_identity": {
                "class_module": "fixture",
                "class_qualname": "FixtureTokenizer",
                "vocab_size": 1000,
                "is_fast": True,
                "special_token_ids": {},
                "artifact_set_sha256": model_artifacts["artifact_set_sha256"],
            },
        },
        "dataset": {
            "bucket": "deepmind-gutenberg",
            "prefix": "train/",
            "records": rr2.FORMAL_TRAIN_RECORDS,
            "pg19_data_sha256": data_sha,
            "pg19_manifest_sha256": data_manifest_sha,
            "test_or_validation_objects_used": False,
            "longbench_consumed": False,
        },
        "window_algorithm": {
            "books": 8,
            "seed": rr2.FORMAL_SEED,
            "document_tokens": rr2.FORMAL_DOCUMENT_TOKENS,
        },
        "query_bank_protocol": {
            "count": 32,
            "query_tokens": 32,
            "query_stride_tokens": 64,
            "resident_counts": [1, 8, 32],
        },
        "pg19_windows_sha256": rr2.FORMAL_RR2_WINDOWS_SHA256,
        "pg19_input_manifest_sha256_contract": "external canonical raw-byte receipt",
        "prior_capacity_cohort": prior,
        "windows": windows,
        "frozen_query_banks": banks,
        "n_prefixes_by_rank": prefixes,
        "oracle_selection_plan": plan,
        "oracle_selection_plan_sha256": rr2.sha256_json(plan),
        "invariants": {"pre_output": True},
        "build_audit": {
            "candidate_outputs_consumed": False,
            "network_access_required": False,
            "cuda_initialized": False,
            "path_independent_serialization": True,
        },
    }
    expectations = rr2.InputExpectations(
        pg19_data_sha256=data_sha,
        pg19_manifest_sha256=data_manifest_sha,
        prior_manifest_sha256=rr2.PRIOR_CAPACITY_MANIFEST_SHA256,
        prior_windows_sha256=rr2.PRIOR_CAPACITY_WINDOWS_SHA256,
        rr2_windows_sha256=rr2.FORMAL_RR2_WINDOWS_SHA256,
        rr2_coordinates=rr2.FORMAL_RR2_COORDINATES,
    )
    rr2.validate_rr2_input_manifest(value, expectations=expectations)
    return value


def selection_plan() -> list[dict[str, object]]:
    arm_id = f"kv={runner.ORACLE_KV_POLICY}|gdn={runner.ORACLE_GDN_BASE_POLICY}"
    return [
        {
            "selection_rule_id": "rank-frozen-heldout-post-rope-v1",
            "rank": rank,
            "book_index": rank,
            "source_object": rr2.FORMAL_RR2_COORDINATES[rank][0],
            "window_index": rr2.FORMAL_RR2_COORDINATES[rank][1] // rr2.FORMAL_WINDOW_STRIDE,
            "document_start_token": rr2.FORMAL_RR2_COORDINATES[rank][1],
            "document_length": rr2.FORMAL_DOCUMENT_TOKENS,
            "document_token_ids_sha256": digest(f"document-rank-{rank}"),
            "layer_index": runner.FORMAL_FULL_LAYERS[rank],
            "request_index": 0,
            "round_index": rank % runner.FORMAL_GENERATION_STEPS,
            "sample_id": (
                f"rr2-rank-{rank}-layer-{runner.FORMAL_FULL_LAYERS[rank]}-"
                f"round-{rank % runner.FORMAL_GENERATION_STEPS}"
            ),
            "kv_policy": runner.ORACLE_KV_POLICY,
            "gdn_base_policy": runner.ORACLE_GDN_BASE_POLICY,
            "cell_role": "ownership_witness",
            "arm_id": arm_id,
            "oracle_cell_id": f"rank-{rank}-N-1-{arm_id}-ownership-witness",
            "held_out_from_threshold_calibration": True,
            "locked_before_candidate_outputs": True,
        }
        for rank in range(runner.FORMAL_WORLD_SIZE)
    ]


def frozen_query_banks() -> list[dict[str, object]]:
    plan = selection_plan()
    result = []
    for rank in range(runner.FORMAL_WORLD_SIZE):
        document_start = plan[rank]["document_start_token"]
        document_end = document_start + rr2.FORMAL_DOCUMENT_TOKENS
        bank = {
            "rank": rank,
            "book_index": rank,
            "source_id": str(plan[rank]["source_object"]).removeprefix("train/").removesuffix(".txt"),
            "source_object": plan[rank]["source_object"],
            "window_index": plan[rank]["window_index"],
            "document_start_token": document_start,
            "document_end_token_exclusive": document_end,
            "document_token_ids_sha256": plan[rank]["document_token_ids_sha256"],
            "query_bank_start_token": document_end + rr2.FORMAL_QUERY_TOKENS,
            "query_stride_tokens": rr2.FORMAL_QUERY_BANK_STRIDE,
            "query_tokens": rr2.FORMAL_QUERY_TOKENS,
            "count": rr2.FORMAL_QUERY_BANK_COUNT,
            "query_bank_sha256": digest(f"query-bank-{rank}"),
            "rows": [
                {
                    "request_index": request_index,
                    "source_token_offset": (
                        document_end
                        + rr2.FORMAL_QUERY_TOKENS
                        + request_index * rr2.FORMAL_QUERY_BANK_STRIDE
                    ),
                    "query_tokens": runner.FORMAL_QUERY_TOKENS,
                    "query_token_ids_sha256": digest(
                        f"query-rank-{rank}-request-{request_index}"
                    ),
                }
                for request_index in range(max(runner.FORMAL_RESIDENT_COUNTS))
            ],
        }
        bank["manifest_sha256"] = rr2.sha256_json(bank)
        result.append(bank)
    return result


def make_fixture(root: Path) -> dict[str, Path]:
    model = root / "model"
    model.mkdir(parents=True)
    model_names = sorted(
        {
            *builder.MODEL_MANIFEST_NAMES,
            *rr2.REQUIRED_MODEL_ARTIFACTS,
            *rr2.TOKENIZER_BPE_LAYOUT_FILES,
            "chat_template.jinja",
        }
    )
    for name in model_names:
        write_json(model / name, {"logical_name": name, "fixture": True})
    code_ledger = root / "inputs" / "code.sha256"
    code_ledger.parent.mkdir(parents=True)
    write_ledger(code_ledger, ["a.py", "z.py"])
    model_artifacts = root / "inputs" / "model-artifacts.sha256"
    write_actual_model_ledger(model_artifacts, model, model_names)
    model_weights = root / "inputs" / "model-weights.sha256"
    write_ledger(
        model_weights,
        [f"model.safetensors-{index:05d}-of-00014.safetensors" for index in range(1, 15)],
    )
    pg19_data = root / "inputs" / "pg19.jsonl"
    pg19_data.write_text("{\"fixture\":true}\n", encoding="utf-8")
    pg19_manifest = root / "inputs" / "pg19.manifest.json"
    formal_sources = [source for source, _start in rr2.FORMAL_RR2_COORDINATES]
    remaining_sources = [
        f"train/{20000 + index}.txt"
        for index in range(rr2.FORMAL_TRAIN_RECORDS - len(formal_sources))
    ]
    write_json(
        pg19_manifest,
        {
            "bucket": "deepmind-gutenberg",
            "prefix": "train/",
            "test_or_validation_objects_used": False,
            "jsonl_sha256": hashlib.sha256(pg19_data.read_bytes()).hexdigest(),
            "objects": [
                {"index": index, "name": source}
                for index, source in enumerate(formal_sources + remaining_sources)
            ],
        },
    )
    prior_capacity = root / "inputs" / "prior-capacity.json"
    shutil.copyfile(PRIOR_CAPACITY, prior_capacity)
    pg19_input = root / "inputs" / "rr2-pg19-input.json"
    rr2_value = rr2_input_manifest_fixture(
        model=model,
        pg19_data=pg19_data,
        pg19_manifest=pg19_manifest,
        prior_capacity=prior_capacity,
    )
    write_json(pg19_input, rr2_value)
    frozen_banks = root / "inputs" / "rr2-frozen-query-banks.json"
    write_json(frozen_banks, rr2_value["frozen_query_banks"])
    protocol = root / "inputs" / "protocol.json"
    write_json(protocol, {"protocol": runner.PROTOCOL, "preregistered": True})
    oracle = root / "inputs" / "oracle.json"
    write_json(oracle, rr2_value["oracle_selection_plan"])
    prior_fp32_context = root / "inputs" / "prior-fp32-context.json"
    shutil.copyfile(PRIOR_FP32_CONTEXT, prior_fp32_context)
    review_response_plan = root / "inputs" / "review-response-plan.json"
    shutil.copyfile(REVIEW_RESPONSE_PLAN, review_response_plan)
    output = root / "outputs"
    return {
        "model": model,
        "code_ledger": code_ledger,
        "model_artifacts": model_artifacts,
        "model_weights": model_weights,
        "pg19_data": pg19_data,
        "pg19_manifest": pg19_manifest,
        "pg19_input": pg19_input,
        "prior_capacity": prior_capacity,
        "frozen_banks": frozen_banks,
        "protocol": protocol,
        "oracle": oracle,
        "prior_fp32_context": prior_fp32_context,
        "review_response_plan": review_response_plan,
        "release": output / "release.json",
        "identity": output / "identity.json",
        "banks": output / "banks.json",
        "selection": output / "selection.json",
    }


def preregister_command(paths: dict[str, Path], *, revision: str | None = None) -> list[str]:
    return [
        os.fspath(BUILDER),
        "preregister",
        "--output",
        os.fspath(paths["release"]),
        "--frozen-identity-output",
        os.fspath(paths["identity"]),
        "--frozen-query-banks-output",
        os.fspath(paths["banks"]),
        "--oracle-selection-output",
        os.fspath(paths["selection"]),
        "--model-id",
        builder.FORMAL_MODEL_ID,
        "--model-revision",
        builder.FORMAL_MODEL_REVISION if revision is None else revision,
        "--model-dir",
        os.fspath(paths["model"]),
        "--code-ledger",
        os.fspath(paths["code_ledger"]),
        "--model-artifact-ledger",
        os.fspath(paths["model_artifacts"]),
        "--model-weight-ledger",
        os.fspath(paths["model_weights"]),
        "--pg19-data",
        os.fspath(paths["pg19_data"]),
        "--pg19-manifest",
        os.fspath(paths["pg19_manifest"]),
        "--pg19-input-manifest",
        os.fspath(paths["pg19_input"]),
        "--expected-pg19-input-manifest-sha256",
        hashlib.sha256(paths["pg19_input"].read_bytes()).hexdigest(),
        "--prior-capacity-manifest",
        os.fspath(paths["prior_capacity"]),
        "--frozen-query-banks-input",
        os.fspath(paths["frozen_banks"]),
        "--expected-frozen-query-banks-input-sha256",
        hashlib.sha256(paths["frozen_banks"].read_bytes()).hexdigest(),
        "--protocol-source-manifest",
        os.fspath(paths["protocol"]),
        "--oracle-selection-input",
        os.fspath(paths["oracle"]),
        "--expected-oracle-selection-input-sha256",
        hashlib.sha256(paths["oracle"].read_bytes()).hexdigest(),
        "--prior-fp32-context-manifest",
        os.fspath(paths["prior_fp32_context"]),
        "--expected-prior-fp32-context-manifest-sha256",
        builder.PRIOR_FP32_CONTEXT_MANIFEST_SHA256,
        "--review-response-plan",
        os.fspath(paths["review_response_plan"]),
        "--expected-review-response-plan-sha256",
        hashlib.sha256(paths["review_response_plan"].read_bytes()).hexdigest(),
    ]


def run_builder(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.fspath(GPU_DIR)
    # The production runner correctly requires the frozen real PG19 byte
    # digests.  Synthetic CPU-only fixtures cannot manufacture those hashes,
    # so this test-only child wrapper derives the two expected hashes from the
    # fixture before importing the release builder.  No production CLI or
    # validation path exposes this override.
    wrapper = r'''
import hashlib,json,sys
from pathlib import Path
import run_qcomem_qwen35_forkaudit_review_revision as runner

args=sys.argv[1:]
def option(name):
    return args[args.index(name)+1]
if args[0] == "preregister":
    runner.FORMAL_PG19_DATA_SHA256=hashlib.sha256(
        Path(option("--pg19-data")).read_bytes()
    ).hexdigest()
    runner.FORMAL_PG19_MANIFEST_SHA256=hashlib.sha256(
        Path(option("--pg19-manifest")).read_bytes()
    ).hexdigest()
elif args[0] == "receipts":
    static=json.loads(Path(option("--static-artifact")).read_text(encoding="utf-8"))
    identity=static["frozen_identity"]
    runner.FORMAL_PG19_DATA_SHA256=identity["pg19_data_sha256"]
    runner.FORMAL_PG19_MANIFEST_SHA256=identity["pg19_manifest_sha256"]

import build_qcomem_qwen35_forkaudit_review_manifest as builder
if args[0] == "preregister":
    # Existing release-governance fixtures intentionally use hand-constructed
    # manifests and do not exercise tokenization.  Keep those tests isolated
    # from Transformers availability; the dedicated source-replay regression
    # below calls the real pure function with a deterministic tokenizer.
    def synthetic_fixture_replay(namespace, *, expectations):
        import build_qcomem_forkaudit_rr2_input_manifest as rr2
        main_raw=namespace.pg19_input_manifest.read_bytes()
        main=rr2.strict_json_loads(main_raw, label="synthetic RR2 main")
        banks_raw=rr2.canonical_json_bytes(main["frozen_query_banks"])+b"\n"
        oracle_raw=rr2.canonical_json_bytes(main["oracle_selection_plan"])+b"\n"
        if banks_raw != namespace.frozen_query_banks_input.read_bytes():
            raise builder.ManifestBuildError(
                "RR2 query-bank sidecar differs from authoritative synthetic fixture"
            )
        if oracle_raw != namespace.oracle_selection_input.read_bytes():
            raise builder.ManifestBuildError(
                "RR2 oracle sidecar differs from authoritative synthetic fixture"
            )
        return main, {
            "source_replay_mode":"synthetic-test-only-explicit-bypass",
            "main_raw_sha256":hashlib.sha256(main_raw).hexdigest(),
            "main_raw_bytes":len(main_raw),
            "query_banks_raw_sha256":hashlib.sha256(banks_raw).hexdigest(),
            "query_banks_raw_bytes":len(banks_raw),
            "oracle_selection_raw_sha256":hashlib.sha256(oracle_raw).hexdigest(),
            "oracle_selection_raw_bytes":len(oracle_raw),
            "document_and_all_query_token_digests_recomputed":False,
            "all_three_supplied_files_byte_exact":True,
            "network_access_allowed":False,
            "cuda_initialized":False,
        }
    builder._rebuild_rr2_inputs_from_source=synthetic_fixture_replay
try:
    raise SystemExit(builder.main(args))
except (builder.ManifestBuildError, runner.ReviewAuditError) as exc:
    raise SystemExit(f"ForkAudit release manifest rejected: {exc}") from exc
'''
    return subprocess.run(
        [os.sys.executable, "-c", wrapper, *command[1:]],
        cwd=GPU_DIR,
        env=env,
        check=check,
        capture_output=True,
        text=True,
    )


def code_snapshot_audit_program() -> str:
    source = LAUNCHER.read_text(encoding="utf-8")
    start_marker = "<<'PY_CODE_SNAPSHOT_AUDIT'\n"
    end_marker = "\nPY_CODE_SNAPSHOT_AUDIT\n"
    start = source.index(start_marker) + len(start_marker)
    end = source.index(end_marker, start)
    return source[start:end]


def make_code_snapshot_fixture(root: Path) -> Path:
    code = root / "code"
    required = {
        "run_qcomem_qwen35_forkaudit_review_revision.py",
        "build_qcomem_qwen35_forkaudit_review_manifest.py",
        "build_qcomem_forkaudit_rr2_input_manifest.py",
        "launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh",
        "FORKAUDIT_REVIEW_REVISION_PROTOCOL_ZH.md",
    }
    for name in required:
        path = code / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{name}\n", encoding="utf-8")
    for index in range(12):
        path = code / "package" / "nested" / f"module_{index:02d}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"VALUE = {index}\n", encoding="utf-8")
    return code


def freeze_code_snapshot(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def thaw_code_snapshot(root: Path) -> None:
    if not root.exists():
        return
    os.chmod(root, 0o755)
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def run_code_snapshot_audit(
    code: Path, output_root: Path, label: str
) -> subprocess.CompletedProcess[str]:
    output_root.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        [
            os.sys.executable,
            "-I",
            "-",
            os.fspath(code),
            os.fspath(output_root / f"{label}.nul"),
            os.fspath(output_root / f"{label}.sha256"),
        ],
        input=code_snapshot_audit_program(),
        capture_output=True,
        text=True,
        check=False,
    )


class ForkAuditLauncherGovernanceTest(unittest.TestCase):
    def test_launcher_syntax_and_fail_closed_release_gate(self) -> None:
        subprocess.run(["bash", "-n", os.fspath(LAUNCHER)], check=True)
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("export LC_ALL=C", source)
        self.assertEqual(source.count("FORMAL_PIPELINE_RELEASED=true"), 1)
        self.assertTrue(runner.GPU_LOOP_IMPLEMENTED)
        self.assertLess(
            source.index("CURRENT_PHASE=producer_release_gate"),
            source.index("CURRENT_PHASE=formal_gpu_preflight"),
        )
        gate = source[
            source.index("CURRENT_PHASE=producer_release_gate") :
            source.index("CURRENT_PHASE=formal_gpu_preflight")
        ]
        self.assertIn("BLOCKED_GPU_PRODUCER_NOT_IMPLEMENTED", gate)
        self.assertIn("exit 3", gate)
        self.assertNotIn("nvidia-smi", gate)
        self.assertNotIn("--stage shard", gate)

    def test_launcher_cannot_create_external_resources(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        for command in (
            "qsctl",
            "kubectl",
            "sbatch",
            "qsub",
            "bsub",
            "docker run",
            "ssh ",
            "scp ",
        ):
            self.assertNotIn(command, source)

    def test_model_load_keeper_bootstrap_isolated_but_imports_code_snapshot(self) -> None:
        bootstrap = (
            "import runpy,signal,sys; "
            "signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGIO}); "
            "script=sys.argv[1]; sys.path.insert(0,sys.argv[2]); "
            "sys.argv=[script,*sys.argv[3:]]; "
            'runpy.run_path(script,run_name="__main__")'
        )
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn('"$PYTHON" -I -c \\', source)
        self.assertIn("sys.path.insert(0,sys.argv[2])", source)
        self.assertNotIn("os.execv(sys.executable", source)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trusted = root / "trusted-code"
            trusted.mkdir()
            script = trusted / "keeper_entry.py"
            script.write_text("import keeper_sibling\nprint(keeper_sibling.VALUE)\n")
            (trusted / "keeper_sibling.py").write_text(
                "VALUE='trusted-sibling-imported'\n", encoding="utf-8"
            )
            poison = root / "poison"
            poison.mkdir()
            marker = poison / "POISON_IMPORTED"
            (poison / "keeper_sibling.py").write_text(
                "from pathlib import Path\n"
                f"Path({os.fspath(marker)!r}).write_text('loaded')\n"
                "raise RuntimeError('poison import loaded')\n",
                encoding="utf-8",
            )
            environment = os.environ.copy()
            environment["PYTHONPATH"] = os.fspath(poison)
            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    bootstrap,
                    os.fspath(script),
                    os.fspath(trusted),
                ],
                cwd=poison,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "trusted-sibling-imported\n")
            self.assertFalse(marker.exists())

    def test_rank_mutant_and_fresh_cache_contract_is_pinned(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            'EXPECTED_MUTANT_ASSIGNMENT="0:M1,M9;1:M2;2:M3;3:M4;4:M5;5:M6;6:M7;7:M8"',
            source,
        )
        self.assertEqual(source.count("for RANK in 0 1 2 3 4 5 6 7"), 1)
        for ambient_name in (
            "FORKAUDIT_STATIC_ARTIFACT",
            "FORKAUDIT_EXPECTED_STATIC_SHA256",
            "FORKAUDIT_ORACLE_SELECTION_PLAN",
            "FORKAUDIT_ASSIGNED_MUTANTS",
            "FORKAUDIT_MUTANT_CACHE_POLICY",
            "FORKAUDIT_ARTIFACT_ROOT",
        ):
            self.assertNotIn(ambient_name, source)
        self.assertIn('release["mutant_case_isolation"]', source)
        self.assertIn('"fresh_document_cache_per_case"', source)
        self.assertIn('"fresh_request_cache_per_case"', source)

    def test_formal_shard_and_aggregate_cli_sets_are_exact_and_explicit(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        shard = source[
            source.index("CURRENT_PHASE=eight_rank_shards") :
            source.index("CURRENT_PHASE=detached_raw_receipts")
        ]
        aggregate = source[
            source.index("CURRENT_PHASE=blind_aggregate") :
            source.index("CURRENT_PHASE=terminal_integrity")
        ]

        def line_options(segment: str) -> set[str]:
            return {
                line.split()[0]
                for line in segment.splitlines()
                if line.strip().startswith("--")
            }

        self.assertEqual(line_options(shard), builder.RUNNER_FORMAL_SHARD_OPTIONS)
        self.assertEqual(
            line_options(aggregate), builder.RUNNER_FORMAL_AGGREGATE_OPTIONS
        )
        self.assertNotIn("FORKAUDIT_", shard)
        self.assertIn('--model-dir "$PRIVATE_MODEL_VIEW"', shard)
        self.assertNotIn('--model-dir "$MODEL_DIR"', shard)
        self.assertIn("materialize-private-model-view \\", source)
        self.assertIn("ficlone-then-byte-copy;hardlink-and-symlink-forbidden", BUILDER.read_text(encoding="utf-8"))
        self.assertIn('--artifact-root "$RUN_DIR/raw"', shard)
        self.assertIn('--artifact-root "$RUN_DIR/raw"', aggregate)
        self.assertEqual(source.count('--artifact-root "$RUN_DIR/raw"'), 3)
        self.assertIn("find raw -type f -print0", source)
        self.assertEqual(
            source.count("sha256sum -c receipts/all-raw-artifacts.sha256"), 2
        )
        self.assertIn("gpu-assignment-receipt \\", source)
        self.assertIn('"${#GPU_UUIDS[@]}" -ne 8', source)
        self.assertIn(
            'CUDA_VISIBLE_DEVICES="${GPU_UUIDS[$RANK]}"', shard
        )
        self.assertNotIn(
            'CUDA_VISIBLE_DEVICES="${GPU_VISIBLE_INDICES[$RANK]}"', shard
        )
        self.assertIn('--expected-gpu-uuid "${GPU_UUIDS[$RANK]}"', shard)
        self.assertIn(
            '--gpu-assignment-receipt \\\n'
            '      "$RUN_DIR/receipts/gpu-assignment-receipt.json"',
            shard,
        )
        self.assertIn(
            '--expected-gpu-assignment-receipt-raw-sha256 \\\n'
            '      "$GPU_ASSIGNMENT_RECEIPT_RAW_SHA256"',
            shard,
        )
        for segment in (shard, aggregate):
            self.assertIn("--private-model-view-manifest", segment)
            self.assertIn(
                '"$RUN_DIR/preregistration/private-model-view-manifest.json"',
                segment,
            )
            self.assertIn(
                "--expected-private-model-view-manifest-raw-sha256", segment
            )
            self.assertIn('"$PRIVATE_MODEL_VIEW_MANIFEST_RAW_SHA256"', segment)
        for segment in (shard, aggregate):
            self.assertIn(
                '--run-id-receipt "$RUN_DIR/receipts/run-id-receipt.json"',
                segment,
            )
            self.assertIn(
                '--expected-run-id-receipt-sha256 "$RUN_ID_RECEIPT_SHA256"',
                segment,
            )

    def test_one_preoutput_run_id_is_bound_and_shared_by_all_ranks(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        builder_source = BUILDER.read_text(encoding="utf-8")
        static_done = source.index("02_static_preregistration_ok")
        run_identity = source.index("CURRENT_PHASE=run_identity")
        producer_gate = source.index("CURRENT_PHASE=producer_release_gate")
        shard_loop = source.index("for RANK in 0 1 2 3 4 5 6 7")
        self.assertLess(static_done, run_identity)
        self.assertLess(run_identity, producer_gate)
        self.assertLess(producer_gate, shard_loop)
        self.assertEqual(builder_source.count("secrets.token_bytes(32)"), 1)
        self.assertEqual(source.count("RUN_ID=$("), 1)
        self.assertIn('[[ ! "$RUN_ID" =~ ^[0-9a-f]{32}$ ]]', source)
        self.assertEqual(source.count("run-id-receipt \\"), 1)
        self.assertIn("run-id-receipt.json", source)
        self.assertIn('--static-artifact-sha256 "$STATIC_ARTIFACT_SHA256"', source)
        self.assertIn(
            '--protocol-manifest-sha256 "$EXPECTED_PROTOCOL_SOURCE_MANIFEST_SHA256"',
            source,
        )
        self.assertIn('--run-id "$RUN_ID"', source)
        self.assertEqual(
            source.count(
                '--run-id-receipt "$RUN_DIR/receipts/run-id-receipt.json"'
            ),
            3,
        )
        self.assertNotIn("RUN_ID=00000000000000000000000000000000", source)

    def test_zero_skip_real_transformers_gate_and_data_boundary(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(
            "test_real_tf514_qwen_call_consumes_and_advances_position_ids", source
        )
        self.assertIn("focused test suite contained a skip", source)
        self.assertIn("test_qcomem_qwen35_vllm_paged_integration", source)
        self.assertIn("PG19 train-only", source)
        self.assertNotIn("--validation-data", source)
        self.assertNotIn("--longbench", source)
        self.assertIn("[Ll][Oo][Nn][Gg][Bb][Ee][Nn][Cc][Hh]", source)
        self.assertIn("[Tt][Ee][Ss][Tt]-[Vv]2", source)

    def test_commit_pycache_timeouts_traps_and_terminal_order(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn(builder.FORMAL_MODEL_REVISION, source)
        self.assertIn("^[0-9a-f]{40}$", source)
        self.assertIn('export PYTHONPYCACHEPREFIX="$RUN_DIR/pycache"', source)
        self.assertIn("RUN_DIR and PYTHONPYCACHEPREFIX must be outside", source)
        self.assertIn("trap on_error ERR", source)
        self.assertIn("trap 'record_failure 130' INT", source)
        self.assertIn("trap 'record_failure 143' TERM", source)
        self.assertIn("timeout --signal=TERM --kill-after=60s 21600s", source)
        self.assertIn("timeout --signal=TERM --kill-after=60s 1800s", source)
        receipt = source.index("detached-receipt-manifest.canonical-json.sha256")
        aggregate = source.index("aggregate-final-audit.json")
        terminal_raw = source.index("raw-artifact-terminal-integrity.log")
        scientific = source.index("scientific-artifact-integrity.log")
        done = source.index('date -u +%FT%TZ > "$RUN_DIR/stages/99_done"')
        self.assertLess(receipt, aggregate)
        self.assertLess(aggregate, terminal_raw)
        self.assertLess(terminal_raw, scientific)
        self.assertLess(scientific, done)

    def test_recursive_code_closure_and_terminal_replay_are_pinned(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertEqual(source.count("audit_code_snapshot \\"), 2)
        self.assertIn("symbolic link present", source)
        self.assertIn("writable code entry present", source)
        self.assertIn("Python bytecode cache present", source)
        self.assertIn("selected = list(regular_files)", source)
        self.assertIn('"$PYTHON" -I - "$code_root"', source)
        self.assertIn('*.py) CODE_FILES+=("$CODE_FILE") ;;', source)
        self.assertIn("code-files-terminal.nul", source)
        self.assertIn("code-terminal.sha256", source)
        self.assertIn("recursive code closure changed after preflight", source)
        self.assertIn("recursive code ledger changed after preflight", source)
        self.assertIn(
            'verify_sha "$RUN_DIR/preregistration/code.sha256" \\\n'
            '  "$EXPECTED_CODE_LEDGER_SHA256" '
            "code-ledger-preregistration-terminal",
            source,
        )
        self.assertIn(
            'verify_sha "$RUN_DIR/receipts/code-terminal.sha256" \\\n'
            '  "$EXPECTED_CODE_LEDGER_SHA256" code-ledger-rebuilt-terminal',
            source,
        )
        self.assertNotIn("find . -maxdepth 1 -type f", source)

    def test_code_snapshot_audit_rejects_nested_symlink_writable_and_pyc(self) -> None:
        cases = (
            (
                "nested-symlink",
                lambda code: (code / "package" / "nested" / "escape.py").symlink_to(
                    code / "module-outside.py"
                ),
                "symbolic link present",
            ),
            (
                "writable-file",
                lambda code: os.chmod(
                    code / "package" / "nested" / "module_00.py", 0o644
                ),
                "writable code entry present",
            ),
            (
                "writable-directory",
                lambda code: os.chmod(code / "package" / "nested", 0o755),
                "writable code entry present",
            ),
            (
                "writable-root",
                lambda code: os.chmod(code, 0o755),
                "CODE_DIR root is writable",
            ),
            (
                "pycache",
                lambda code: (
                    (code / "package" / "__pycache__").mkdir(),
                    (code / "package" / "__pycache__" / "module.pyc").write_bytes(
                        b"fixture-bytecode"
                    ),
                ),
                "Python bytecode cache present",
            ),
            (
                "bytecode-file",
                lambda code: (code / "package" / "stale.pyc").write_bytes(
                    b"fixture-bytecode"
                ),
                "Python bytecode file present",
            ),
        )
        for label, mutate, expected in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                code = make_code_snapshot_fixture(root)
                if label in {"writable-file", "writable-directory", "writable-root"}:
                    freeze_code_snapshot(code)
                    mutate(code)
                else:
                    mutate(code)
                    freeze_code_snapshot(code)
                try:
                    result = run_code_snapshot_audit(code, root / "audit", label)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(expected, result.stderr)
                finally:
                    thaw_code_snapshot(code)

    def test_code_snapshot_ledger_detects_nested_extra_python_and_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = make_code_snapshot_fixture(root)
            freeze_code_snapshot(code)
            try:
                baseline = run_code_snapshot_audit(code, root / "audit", "baseline")
                self.assertEqual(baseline.returncode, 0, baseline.stderr)
                baseline_ledger = (root / "audit" / "baseline.sha256").read_bytes()
                baseline_names = (root / "audit" / "baseline.nul").read_bytes()
                self.assertIn(b"./package/nested/module_11.py\0", baseline_names)

                thaw_code_snapshot(code)
                extra = code / "package" / "deeper" / "extra_import.py"
                extra.parent.mkdir(parents=True)
                extra.write_text("EXTRA = True\n", encoding="utf-8")
                freeze_code_snapshot(code)
                extended = run_code_snapshot_audit(code, root / "audit", "extended")
                self.assertEqual(extended.returncode, 0, extended.stderr)
                extended_ledger = (root / "audit" / "extended.sha256").read_bytes()
                extended_names = (root / "audit" / "extended.nul").read_bytes()
                self.assertIn(b"./package/deeper/extra_import.py\0", extended_names)
                self.assertNotEqual(
                    hashlib.sha256(extended_ledger).hexdigest(),
                    hashlib.sha256(baseline_ledger).hexdigest(),
                )
            finally:
                thaw_code_snapshot(code)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = make_code_snapshot_fixture(root)
            freeze_code_snapshot(code)
            try:
                baseline = run_code_snapshot_audit(code, root / "audit", "baseline")
                self.assertEqual(baseline.returncode, 0, baseline.stderr)
                baseline_ledger = (root / "audit" / "baseline.sha256").read_bytes()
                thaw_code_snapshot(code)
                target = code / "package" / "nested" / "module_05.py"
                target.write_text("VALUE = 'tampered'\n", encoding="utf-8")
                freeze_code_snapshot(code)
                tampered = run_code_snapshot_audit(code, root / "audit", "tampered")
                self.assertEqual(tampered.returncode, 0, tampered.stderr)
                tampered_ledger = (root / "audit" / "tampered.sha256").read_bytes()
                self.assertNotEqual(
                    hashlib.sha256(tampered_ledger).hexdigest(),
                    hashlib.sha256(baseline_ledger).hexdigest(),
                )
            finally:
                thaw_code_snapshot(code)

    def test_all_regular_files_include_unknown_and_extension_shadow_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = make_code_snapshot_fixture(root)
            shadow_py = code / "package" / "nested" / "shadow.py"
            shadow_py.write_text("VALUE = 'python'\n", encoding="utf-8")
            freeze_code_snapshot(code)
            try:
                baseline = run_code_snapshot_audit(code, root / "audit", "baseline")
                self.assertEqual(baseline.returncode, 0, baseline.stderr)
                baseline_ledger = (root / "audit" / "baseline.sha256").read_bytes()

                thaw_code_snapshot(code)
                extension_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
                shadow_extension = shadow_py.with_name("shadow" + extension_suffix)
                shadow_extension.write_bytes(b"synthetic-extension-shadow")
                unknown = code / "package" / "runtime.import-config"
                unknown.write_bytes(b"synthetic-unknown-import-input")
                freeze_code_snapshot(code)
                extended = run_code_snapshot_audit(code, root / "audit", "extended")
                self.assertEqual(extended.returncode, 0, extended.stderr)
                extended_names = (root / "audit" / "extended.nul").read_bytes()
                self.assertIn(
                    ("./package/nested/" + shadow_extension.name).encode("utf-8")
                    + b"\0",
                    extended_names,
                )
                self.assertIn(b"./package/runtime.import-config\0", extended_names)
                extended_ledger = (root / "audit" / "extended.sha256").read_bytes()
                self.assertNotEqual(
                    hashlib.sha256(baseline_ledger).hexdigest(),
                    hashlib.sha256(extended_ledger).hexdigest(),
                )
            finally:
                thaw_code_snapshot(code)

    def test_coordinated_terminal_ledger_rewrite_cannot_escape_external_pin(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = make_code_snapshot_fixture(root)
            audit = root / "audit"
            freeze_code_snapshot(code)
            try:
                baseline = run_code_snapshot_audit(code, audit, "preregistration")
                self.assertEqual(baseline.returncode, 0, baseline.stderr)
                preregistration_list = audit / "preregistration.nul"
                preregistration_ledger = audit / "preregistration.sha256"
                expected_external_sha256 = hashlib.sha256(
                    preregistration_ledger.read_bytes()
                ).hexdigest()

                thaw_code_snapshot(code)
                target = code / "package" / "nested" / "module_05.py"
                target.write_text("VALUE = 'coordinated-tamper'\n", encoding="utf-8")
                freeze_code_snapshot(code)
                terminal = run_code_snapshot_audit(code, audit, "terminal")
                self.assertEqual(terminal.returncode, 0, terminal.stderr)
                terminal_list = audit / "terminal.nul"
                terminal_ledger = audit / "terminal.sha256"

                # Model the adversary rewriting both retained preregistration
                # artifacts to match the fresh terminal snapshot.  The old
                # cmp + sha256sum checks are now all self-consistent.
                shutil.copyfile(terminal_list, preregistration_list)
                shutil.copyfile(terminal_ledger, preregistration_ledger)
                self.assertEqual(
                    preregistration_list.read_bytes(), terminal_list.read_bytes()
                )
                self.assertEqual(
                    preregistration_ledger.read_bytes(), terminal_ledger.read_bytes()
                )
                replay = subprocess.run(
                    ["sha256sum", "-c", os.fspath(preregistration_ledger)],
                    cwd=code,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(replay.returncode, 0, replay.stderr)

                verify_function_start = LAUNCHER.read_text(encoding="utf-8").index(
                    "verify_sha() {"
                )
                launcher_source = LAUNCHER.read_text(encoding="utf-8")
                verify_function_end = launcher_source.index(
                    "\n}\njson_digest()", verify_function_start
                ) + len("\n}")
                terminal_pin_gate = (
                    "set -euo pipefail\n"
                    + launcher_source[verify_function_start:verify_function_end]
                    + '\nverify_sha "$1" "$3" preregistration-terminal\n'
                    + 'verify_sha "$2" "$3" rebuilt-terminal\n'
                )
                rejected = subprocess.run(
                    [
                        "bash",
                        "-c",
                        terminal_pin_gate,
                        "terminal-code-pin-test",
                        os.fspath(preregistration_ledger),
                        os.fspath(terminal_ledger),
                        expected_external_sha256,
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertNotEqual(rejected.returncode, 0)
                self.assertIn("SHA-256 mismatch", rejected.stderr)
            finally:
                thaw_code_snapshot(code)

    def test_external_receipts_and_memory_witness_split_are_audited(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        self.assertIn("all-raw-artifacts.sha256", source)
        self.assertIn("detached-receipt-manifest.json", source)
        self.assertIn("expected-receipt-manifest-sha256", source)
        self.assertIn("fresh_document_cache_per_case", source)
        self.assertIn("fresh_request_cache_per_case", source)
        self.assertIn("cell_ids_must_differ", source)

    def test_rr2_static_raw_input_contract_and_sidecar_rebuild_are_pinned(self) -> None:
        source = LAUNCHER.read_text(encoding="utf-8")
        for option in (
            "--rr2-input-manifest",
            "--expected-rr2-input-manifest-sha256",
            "--prior-fp32-context-manifest",
            "--expected-prior-fp32-context-manifest-sha256",
            "--review-experiment-plan",
            "--expected-review-experiment-plan-sha256",
            "--frozen-query-banks",
            "--oracle-selection-plan",
        ):
            self.assertIn(option, source)
        self.assertIn("--frozen-query-banks-output", source)
        self.assertIn("cmp -s", source)
        self.assertIn("RR2_INPUT_BUILDER", source)
        self.assertIn("source-rebuilt-rr2-input-manifest.json", source)
        self.assertIn("main manifest does not byte-replay from PG19", source)
        self.assertIn("generated query-bank sidecar differs bytewise", source)
        self.assertIn("generated oracle sidecar differs bytewise", source)
        self.assertIn(rr2.FORMAL_RR2_WINDOWS_SHA256, source)
        self.assertIn(rr2.PRIOR_CAPACITY_MANIFEST_SHA256, source)
        self.assertIn(builder.PRIOR_FP32_CONTEXT_MANIFEST_SHA256, source)
        self.assertIn(runner.FINAL_REVIEW_RESPONSE_PLAN_SHA256, source)
        self.assertEqual(
            builder._runner_rr2_compatibility_gate()["runner_contract_matched"],
            True,
        )


class ForkAuditManifestBuilderTest(unittest.TestCase):
    def test_model_load_keeper_binds_private_view_and_emits_valid_closure(self) -> None:
        class FakeLeaseOps:
            f_rdlck = 0

            def __init__(self) -> None:
                self.active: set[int] = set()

            def acquire_read(self, fd: int) -> None:
                self.active.add(fd)

            def get(self, fd: int) -> int:
                if fd not in self.active:
                    raise model_lease.ModelLoadLeaseError("fake lease is inactive")
                return self.f_rdlck

            def release(self, fd: int) -> None:
                self.active.discard(fd)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-model"
            source.mkdir()
            artifact_names = [
                "config.json",
                "tokenizer.json",
                "model.safetensors.index.json",
            ]
            weight_names = [
                f"model.safetensors-{index:05d}-of-00014.safetensors"
                for index in range(1, 15)
            ]
            for name in artifact_names + weight_names:
                (source / name).write_bytes(
                    (f"lease-keeper-fixture:{name}\n").encode("utf-8")
                )
            artifact_ledger = root / "model-artifacts.sha256"
            weight_ledger = root / "model-weights.sha256"
            write_actual_model_ledger(artifact_ledger, source, artifact_names)
            write_actual_model_ledger(weight_ledger, source, weight_names)
            artifact_sha = hashlib.sha256(artifact_ledger.read_bytes()).hexdigest()
            weight_sha = hashlib.sha256(weight_ledger.read_bytes()).hexdigest()
            private_view = root / "private-model-view"
            manifest = builder.materialize_private_model_view(
                source_model_dir=source,
                private_model_view=private_view,
                model_artifact_ledger=artifact_ledger,
                expected_model_artifact_ledger_raw_sha256=artifact_sha,
                model_weight_ledger=weight_ledger,
                expected_model_weight_ledger_raw_sha256=weight_sha,
                model_id=builder.FORMAL_MODEL_ID,
                model_revision=builder.FORMAL_MODEL_REVISION,
            )
            manifest_path = root / "private-model-view-manifest.json"
            write_json(manifest_path, manifest)
            manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            authority_path = root / "model-load-authority.json"
            closure_path = root / "model-load-closure.json"
            holder: dict[str, model_lease.ModelLoadLeaseSet] = {}

            def lease_factory(**kwargs: object) -> model_lease.ModelLoadLeaseSet:
                lease = model_lease.ModelLoadLeaseSet(
                    **kwargs,
                    lease_ops=FakeLeaseOps(),
                )
                holder["lease"] = lease
                return lease

            class BoundCloseInput(io.StringIO):
                def readline(self, *args: object, **kwargs: object) -> str:
                    authority = holder["lease"].authority
                    authority_sha = model_lease.sha256_bytes(
                        model_lease.canonical_receipt_bytes(authority)
                    )
                    return f"CLOSE {authority_sha}\n"

            events = io.StringIO()
            with mock.patch.object(
                model_lease, "_process_thread_count", return_value=1
            ):
                result = builder.run_model_load_lease_keeper(
                    model_view=private_view,
                    model_weight_ledger=weight_ledger,
                    expected_model_weight_ledger_raw_sha256=weight_sha,
                    expected_model_artifact_ledger_raw_sha256=artifact_sha,
                    model_view_manifest=manifest_path,
                    expected_model_view_manifest_raw_sha256=manifest_sha,
                    run_id="0123456789abcdef0123456789abcdef",
                    authority_output=authority_path,
                    closure_output=closure_path,
                    control_input=BoundCloseInput(),
                    event_output=events,
                    lease_factory=lease_factory,
                )
            self.assertEqual(
                result["authority_raw_sha256"],
                hashlib.sha256(authority_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                result["closure_raw_sha256"],
                hashlib.sha256(closure_path.read_bytes()).hexdigest(),
            )
            authority = model_lease.authority_from_canonical_bytes(
                authority_path.read_bytes(), result["authority_raw_sha256"]
            )
            closure = model_lease.closure_from_canonical_bytes(
                closure_path.read_bytes(),
                result["closure_raw_sha256"],
                authority=authority,
                require_passed=True,
            )
            self.assertTrue(closure["passed"])
            self.assertTrue(closure["all_leases_released"])
            self.assertTrue(closure["all_fds_closed"])
            self.assertEqual(
                events.getvalue(),
                "READY %s\nCLOSED %s\n"
                % (
                    result["authority_raw_sha256"],
                    result["closure_raw_sha256"],
                ),
            )

            breached_holder: dict[str, model_lease.ModelLoadLeaseSet] = {}

            def breached_factory(
                **kwargs: object,
            ) -> model_lease.ModelLoadLeaseSet:
                lease = model_lease.ModelLoadLeaseSet(
                    **kwargs, lease_ops=FakeLeaseOps()
                )
                breached_holder["lease"] = lease
                return lease

            class BreachingCloseInput(io.StringIO):
                def readline(self, *args: object, **kwargs: object) -> str:
                    lease = breached_holder["lease"]
                    authority_sha = model_lease.sha256_bytes(
                        model_lease.canonical_receipt_bytes(lease.authority)
                    )
                    lease.mark_breach_for_test()
                    return f"CLOSE {authority_sha}\n"

            failed_authority_path = root / "failed-authority.json"
            failed_closure_path = root / "failed-closure.json"
            with mock.patch.object(
                model_lease, "_process_thread_count", return_value=1
            ):
                with self.assertRaisesRegex(
                    builder.ManifestBuildError,
                    "closure did not pass",
                ):
                    builder.run_model_load_lease_keeper(
                        model_view=private_view,
                        model_weight_ledger=weight_ledger,
                        expected_model_weight_ledger_raw_sha256=weight_sha,
                        expected_model_artifact_ledger_raw_sha256=artifact_sha,
                        model_view_manifest=manifest_path,
                        expected_model_view_manifest_raw_sha256=manifest_sha,
                        run_id="fedcba9876543210fedcba9876543210",
                        authority_output=failed_authority_path,
                        closure_output=failed_closure_path,
                        control_input=BreachingCloseInput(),
                        event_output=io.StringIO(),
                        lease_factory=breached_factory,
                    )
            failed_authority = model_lease.authority_from_canonical_bytes(
                failed_authority_path.read_bytes(),
                hashlib.sha256(failed_authority_path.read_bytes()).hexdigest(),
            )
            failed_closure = model_lease.closure_from_canonical_bytes(
                failed_closure_path.read_bytes(),
                hashlib.sha256(failed_closure_path.read_bytes()).hexdigest(),
                authority=failed_authority,
                require_passed=False,
            )
            self.assertFalse(failed_closure["passed"])
            self.assertIn(
                "sticky_sigio_or_pending_lease_break",
                failed_closure["invalid_reasons"],
            )
            self.assertFalse(torch.cuda.is_initialized())

    def test_private_model_view_has_distinct_read_only_inodes_and_no_links(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-model"
            source.mkdir()
            artifact_names = ["config.json", "tokenizer.json", "model.safetensors.index.json"]
            weight_names = [
                f"model.safetensors-{index:05d}-of-00014.safetensors"
                for index in range(1, 15)
            ]
            for name in artifact_names + weight_names:
                (source / name).write_bytes((f"private-view-fixture:{name}\n").encode("utf-8"))
            artifact_ledger = root / "model-artifacts.sha256"
            weight_ledger = root / "model-weights.sha256"
            write_actual_model_ledger(artifact_ledger, source, artifact_names)
            write_actual_model_ledger(weight_ledger, source, weight_names)
            artifact_sha = hashlib.sha256(artifact_ledger.read_bytes()).hexdigest()
            weight_sha = hashlib.sha256(weight_ledger.read_bytes()).hexdigest()
            private_view = root / "private-model-view"
            manifest = builder.materialize_private_model_view(
                source_model_dir=source,
                private_model_view=private_view,
                model_artifact_ledger=artifact_ledger,
                expected_model_artifact_ledger_raw_sha256=artifact_sha,
                model_weight_ledger=weight_ledger,
                expected_model_weight_ledger_raw_sha256=weight_sha,
                model_id=builder.FORMAL_MODEL_ID,
                model_revision=builder.FORMAL_MODEL_REVISION,
            )
            self.assertEqual(manifest["weight_file_count"], 14)
            self.assertEqual(manifest["file_count"], 17)
            self.assertEqual(
                {row["copy_mode"] for row in manifest["rows"]}
                <= {"ficlone", "byte-copy"},
                True,
            )
            for row in manifest["rows"]:
                source_stat = (source / row["relative_path"]).stat()
                view_path = private_view / row["relative_path"]
                view_stat = view_path.stat()
                self.assertTrue(view_path.is_file())
                self.assertFalse(view_path.is_symlink())
                self.assertNotEqual(
                    (source_stat.st_dev, source_stat.st_ino),
                    (view_stat.st_dev, view_stat.st_ino),
                )
                self.assertEqual(stat.S_IMODE(view_stat.st_mode) & 0o222, 0)
            self.assertEqual(stat.S_IMODE(private_view.stat().st_mode) & 0o222, 0)
            self.assertFalse(torch.cuda.is_initialized())

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-model"
            source.mkdir()
            outside = root / "outside.json"
            outside.write_bytes(b"outside\n")
            (source / "config.json").symlink_to(outside)
            weight_names = [
                f"model.safetensors-{index:05d}-of-00014.safetensors"
                for index in range(1, 15)
            ]
            for name in weight_names:
                (source / name).write_bytes(name.encode("utf-8"))
            artifact_ledger = root / "model-artifacts.sha256"
            weight_ledger = root / "model-weights.sha256"
            write_actual_model_ledger(artifact_ledger, source, ["config.json"])
            write_actual_model_ledger(weight_ledger, source, weight_names)
            with self.assertRaisesRegex(builder.ManifestBuildError, "non-symlink"):
                builder.materialize_private_model_view(
                    source_model_dir=source,
                    private_model_view=root / "private-model-view",
                    model_artifact_ledger=artifact_ledger,
                    expected_model_artifact_ledger_raw_sha256=hashlib.sha256(
                        artifact_ledger.read_bytes()
                    ).hexdigest(),
                    model_weight_ledger=weight_ledger,
                    expected_model_weight_ledger_raw_sha256=hashlib.sha256(
                        weight_ledger.read_bytes()
                    ).hexdigest(),
                    model_id=builder.FORMAL_MODEL_ID,
                    model_revision=builder.FORMAL_MODEL_REVISION,
                )

    def test_gpu_assignment_receipt_is_unique_h20_cpu_only_and_raw_bound(self) -> None:
        inventory = "".join(
            f"{index}, GPU-{index:032x}, NVIDIA H20, 97871, 9.0\n"
            for index in range(runner.FORMAL_WORLD_SIZE)
        ).encode("utf-8")
        receipt = builder.build_gpu_assignment_receipt(inventory)
        self.assertEqual(
            set(receipt),
            {
                "schema_version",
                "world_size",
                "inventory_query",
                "rows",
                "unique_visible_indices",
                "unique_uuids",
                "all_h20",
                "all_compute_capability_9_0",
                "generated_before_candidate_outputs",
            },
        )
        self.assertEqual(
            [row["rank"] for row in receipt["rows"]], list(range(8))
        )
        self.assertEqual(
            [row["visible_index"] for row in receipt["rows"]], list(range(8))
        )
        self.assertTrue(all(row["compute_capability"] == [9, 0] for row in receipt["rows"]))
        self.assertTrue(all(row["bf16_supported"] is True for row in receipt["rows"]))
        self.assertFalse(torch.cuda.is_initialized())

        duplicate = inventory.decode("utf-8").replace(
            "GPU-00000000000000000000000000000001",
            "GPU-00000000000000000000000000000000",
        ).encode("utf-8")
        with self.assertRaisesRegex(builder.ManifestBuildError, "UUIDs are not unique"):
            builder.build_gpu_assignment_receipt(duplicate)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory_path = root / "inventory.csv"
            output = root / "gpu-assignment.json"
            inventory_path.write_bytes(inventory)
            result = run_builder(
                [
                    os.fspath(BUILDER),
                    "gpu-assignment-receipt",
                    "--inventory",
                    os.fspath(inventory_path),
                    "--output",
                    os.fspath(output),
                ]
            )
            status = json.loads(result.stdout)
            self.assertEqual(
                status["raw_sha256"], hashlib.sha256(output.read_bytes()).hexdigest()
            )
            self.assertEqual(output.read_bytes(), runner.canonical_json_bytes(receipt) + b"\n")

    def test_run_id_receipt_is_128_bit_replayable_and_state_bound(self) -> None:
        static_sha = "1" * 64
        protocol_sha = "2" * 64
        nonce = bytes(range(32))
        receipt = builder.build_run_id_receipt(
            static_artifact_sha256=static_sha,
            protocol_manifest_sha256=protocol_sha,
            nonce=nonce,
        )
        replayed = hashlib.sha256(
            bytes.fromhex(receipt["domain_hex"])
            + bytes.fromhex(static_sha)
            + bytes.fromhex(protocol_sha)
            + nonce
        ).hexdigest()[:32]
        self.assertEqual(receipt["run_id"], replayed)
        self.assertRegex(receipt["run_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(receipt["run_id_bits"], 128)
        receipt_sha = runner.sha256_json(receipt)
        self.assertEqual(
            builder.validate_run_id_receipt(
                receipt,
                expected_sha256=receipt_sha,
                run_id=receipt["run_id"],
                static_artifact_sha256=static_sha,
                protocol_manifest_sha256=protocol_sha,
            ),
            receipt,
        )
        forged = dict(receipt)
        forged["run_id"] = "0" * 32
        with self.assertRaisesRegex(builder.ManifestBuildError, "binding drift"):
            builder.validate_run_id_receipt(
                forged,
                expected_sha256=runner.sha256_json(forged),
                run_id=receipt["run_id"],
                static_artifact_sha256=static_sha,
                protocol_manifest_sha256=protocol_sha,
            )
        changed = builder.build_run_id_receipt(
            static_artifact_sha256="3" * 64,
            protocol_manifest_sha256=protocol_sha,
            nonce=nonce,
        )
        self.assertNotEqual(changed["run_id"], receipt["run_id"])
        with self.assertRaisesRegex(builder.ManifestBuildError, "exactly 256 bits"):
            builder.build_run_id_receipt(
                static_artifact_sha256=static_sha,
                protocol_manifest_sha256=protocol_sha,
                nonce=b"short",
            )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run-id-receipt.json"
            result = run_builder(
                [
                    os.fspath(BUILDER),
                    "run-id-receipt",
                    "--static-artifact-sha256",
                    static_sha,
                    "--protocol-manifest-sha256",
                    protocol_sha,
                    "--output",
                    os.fspath(output),
                ]
            )
            cli_receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(result.stdout.strip(), cli_receipt["run_id"])
            self.assertRegex(result.stdout.strip(), r"^[0-9a-f]{32}$")

    def test_source_replay_rejects_self_consistent_forged_token_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root = base / "source-tree"
            data, data_manifest, prior, model, expectations = rr2_fixture.write_inputs(
                root
            )
            tokenizer = rr2_fixture.PositionTokenizer()
            genuine = rr2.build_from_paths(
                pg19_data=data,
                pg19_manifest=data_manifest,
                prior_capacity_manifest=prior,
                model_dir=model,
                tokenizer=tokenizer,
                expectations=expectations,
            )
            main = root / "rr2-main.json"
            banks = root / "rr2-banks.json"
            oracle = root / "rr2-oracle.json"
            write_json(main, genuine)
            write_json(banks, genuine["frozen_query_banks"])
            write_json(oracle, genuine["oracle_selection_plan"])
            args = SimpleNamespace(
                pg19_data=data,
                pg19_manifest=data_manifest,
                prior_capacity_manifest=prior,
                model_dir=model,
                pg19_input_manifest=main,
                frozen_query_banks_input=banks,
                oracle_selection_input=oracle,
            )
            replayed, audit = builder._rebuild_rr2_inputs_from_source(
                args,
                expectations=expectations,
                tokenizer=tokenizer,
            )
            self.assertEqual(replayed, genuine)
            self.assertTrue(audit["document_and_all_query_token_digests_recomputed"])

            relocated = base / "unrelated" / "relocated-tree"
            shutil.copytree(root, relocated)
            relocated_args = SimpleNamespace(
                pg19_data=relocated / data.relative_to(root),
                pg19_manifest=relocated / data_manifest.relative_to(root),
                prior_capacity_manifest=relocated / prior.relative_to(root),
                model_dir=relocated / model.relative_to(root),
                pg19_input_manifest=relocated / main.relative_to(root),
                frozen_query_banks_input=relocated / banks.relative_to(root),
                oracle_selection_input=relocated / oracle.relative_to(root),
            )
            relocated_replay, relocated_audit = (
                builder._rebuild_rr2_inputs_from_source(
                    relocated_args,
                    expectations=expectations,
                    tokenizer=rr2_fixture.PositionTokenizer(),
                )
            )
            self.assertEqual(relocated_replay, replayed)
            self.assertEqual(relocated_audit, audit)

            forged = copy.deepcopy(genuine)
            forged_bank = forged["frozen_query_banks"][0]
            forged_bank["document_token_ids_sha256"] = "f" * 64
            forged_bank["rows"][0]["query_token_ids_sha256"] = "e" * 64
            forged_bank["query_bank_sha256"] = "d" * 64
            forged_bank["manifest_sha256"] = rr2._bank_self_hash(forged_bank)
            forged["windows"][0]["document_token_ids_sha256"] = "f" * 64
            forged["windows"][0]["query_bank_manifest_sha256"] = forged_bank[
                "manifest_sha256"
            ]
            forged["n_prefixes_by_rank"][:3] = rr2._prefix_rows(forged_bank)
            forged["oracle_selection_plan"][0] = rr2._oracle_selection(
                0, forged_bank
            )
            forged["oracle_selection_plan_sha256"] = rr2.sha256_json(
                forged["oracle_selection_plan"]
            )
            # This documents the old vulnerability: the structural validator
            # accepts a fully self-consistent digest rewrite.
            rr2.validate_rr2_input_manifest(forged, expectations=expectations)
            write_json(main, forged)
            write_json(banks, forged["frozen_query_banks"])
            write_json(oracle, forged["oracle_selection_plan"])
            with self.assertRaisesRegex(
                builder.ManifestBuildError,
                "does not exactly replay from PG19 bytes and tokenizer",
            ):
                builder._rebuild_rr2_inputs_from_source(
                    args,
                    expectations=expectations,
                    tokenizer=tokenizer,
                )

    def test_prior_fp32_context_is_blindly_rederived_not_boolean_trusted(self) -> None:
        value = json.loads(PRIOR_FP32_CONTEXT.read_text(encoding="utf-8"))
        audit = builder._audit_prior_fp32_context(
            value,
            rr2_selection_plan=selection_plan(),
        )
        self.assertEqual(audit["prior_coordinates_sha256"], value["rr2_disjointness_from_prior_context"]["prior_coordinates_sha256"])
        self.assertEqual(
            audit["maximum_observed_prior_relative_l2"],
            value["pre_fixed_threshold_margin_check"]["maximum_observed_prior_relative_l2"],
        )

        document_tamper = copy.deepcopy(value)
        document_tamper["diagnostics"][0]["document_tokens"] = 4095
        document_tamper["rr2_disjointness_from_prior_context"]["document_length_disjoint"] = True
        with self.assertRaisesRegex(builder.ManifestBuildError, "document length drift"):
            builder._audit_prior_fp32_context(
                document_tamper,
                rr2_selection_plan=selection_plan(),
            )

        coordinate_tamper = copy.deepcopy(value)
        coordinate_tamper["rr2_disjointness_from_prior_context"]["prior_coordinates_sha256"] = "0" * 64
        with self.assertRaisesRegex(builder.ManifestBuildError, "coordinate SHA"):
            builder._audit_prior_fp32_context(
                coordinate_tamper,
                rr2_selection_plan=selection_plan(),
            )

        source_tamper = copy.deepcopy(value)
        source_tamper["diagnostics"][0]["source_object"] = "train/99999.txt"
        with self.assertRaisesRegex(builder.ManifestBuildError, "frozen source/rank"):
            builder._audit_prior_fp32_context(
                source_tamper,
                rr2_selection_plan=selection_plan(),
            )

        hash_tamper = copy.deepcopy(value)
        hash_tamper["diagnostics"][0]["position_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(builder.ManifestBuildError, "position-ID SHA"):
            builder._audit_prior_fp32_context(
                hash_tamper,
                rr2_selection_plan=selection_plan(),
            )

        comparison_tamper = copy.deepcopy(value)
        comparison_tamper["diagnostics"][0]["comparison"] = "vllm_fresh_vs_fp32_dense"
        with self.assertRaisesRegex(builder.ManifestBuildError, "comparison drift"):
            builder._audit_prior_fp32_context(
                comparison_tamper,
                rr2_selection_plan=selection_plan(),
            )

        layer_tamper = copy.deepcopy(value)
        layer_tamper["diagnostics"][0]["layer_idx"] = 4
        with self.assertRaisesRegex(builder.ManifestBuildError, "layer/order"):
            builder._audit_prior_fp32_context(
                layer_tamper,
                rr2_selection_plan=selection_plan(),
            )

        maximum_tamper = copy.deepcopy(value)
        maximum_tamper["pre_fixed_threshold_margin_check"]["fixed_threshold_to_prior_maximum_ratio"] += 0.01
        with self.assertRaisesRegex(builder.ManifestBuildError, "threshold/max ratio"):
            builder._audit_prior_fp32_context(
                maximum_tamper,
                rr2_selection_plan=selection_plan(),
            )

        selection_tamper = copy.deepcopy(selection_plan())
        selection_tamper[0]["document_length"] = 1025
        with self.assertRaisesRegex(builder.ManifestBuildError, "selection plan document length"):
            builder._audit_prior_fp32_context(
                value,
                rr2_selection_plan=selection_tamper,
            )

    def test_preregistration_is_byte_identical_after_tree_relocation(self) -> None:
        self.assertFalse(torch.cuda.is_initialized())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = make_fixture(root / "different" / "left")
            right = make_fixture(root / "another" / "deep" / "right")
            first = run_builder(preregister_command(left))
            second = run_builder(preregister_command(right))
            self.assertEqual(left["release"].read_bytes(), right["release"].read_bytes())
            self.assertEqual(left["identity"].read_bytes(), right["identity"].read_bytes())
            self.assertEqual(left["banks"].read_bytes(), right["banks"].read_bytes())
            self.assertEqual(left["selection"].read_bytes(), right["selection"].read_bytes())
            self.assertEqual(first.stdout, second.stdout)
            release = json.loads(left["release"].read_text(encoding="utf-8"))
            serialized = json.dumps(release, sort_keys=True)
            self.assertNotIn(os.fspath(root), serialized)
            self.assertEqual(
                [row["mutant_ids"] for row in release["rank_assignments"]],
                [["M1", "M9"], ["M2"], ["M3"], ["M4"], ["M5"], ["M6"], ["M7"], ["M8"]],
            )
            self.assertTrue(
                release["measurement_cell_isolation"]["cell_ids_must_differ"]
            )
            self.assertTrue(
                release["raw_artifact_integrity"][
                    "detached_external_sha256_receipts_required"
                ]
            )
            self.assertTrue(
                release["rr2_input_binding"][
                    "query_bank_sidecar_equals_authoritative_main"
                ]
            )
            self.assertTrue(
                release["tokenizer_model_ledger_cross_binding"][
                    "every_rr2_tokenizer_artifact_present_in_verified_model_ledger"
                ]
            )
            self.assertEqual(
                release["prior_fp32_context_audit"]["diagnostic_count"], 80
            )
        self.assertFalse(torch.cuda.is_initialized())

    def test_builder_rejects_short_revision_absolute_ledger_and_longbench(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = make_fixture(root / "short")
            short = run_builder(
                preregister_command(paths, revision="59d61f3"), check=False
            )
            self.assertNotEqual(short.returncode, 0)
            self.assertIn("full 40-character commit", short.stderr)

            paths = make_fixture(root / "absolute")
            paths["code_ledger"].write_text(
                f"{digest('a')}  /tmp/a.py\n", encoding="utf-8"
            )
            absolute = run_builder(preregister_command(paths), check=False)
            self.assertNotEqual(absolute.returncode, 0)
            self.assertIn("absolute path", absolute.stderr)

            paths = make_fixture(root / "longbench")
            manifest = json.loads(paths["pg19_manifest"].read_text(encoding="utf-8"))
            manifest["forbidden_fallback"] = "LongBench"
            write_json(paths["pg19_manifest"], manifest)
            forbidden = run_builder(preregister_command(paths), check=False)
            self.assertNotEqual(forbidden.returncode, 0)
            self.assertIn("forbidden dataset marker", forbidden.stderr)

    def test_builder_rejects_sidecar_and_tokenizer_ledger_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            paths = make_fixture(root / "bank-sidecar")
            banks = json.loads(paths["frozen_banks"].read_text(encoding="utf-8"))
            banks[0]["rows"][0]["query_token_ids_sha256"] = "0" * 64
            write_json(paths["frozen_banks"], banks)
            bank_result = run_builder(preregister_command(paths), check=False)
            self.assertNotEqual(bank_result.returncode, 0)
            self.assertIn("sidecar differs from authoritative", bank_result.stderr)

            paths = make_fixture(root / "oracle-sidecar")
            plan = json.loads(paths["oracle"].read_text(encoding="utf-8"))
            plan[0]["round_index"] = (plan[0]["round_index"] + 1) % 8
            write_json(paths["oracle"], plan)
            oracle_result = run_builder(preregister_command(paths), check=False)
            self.assertNotEqual(oracle_result.returncode, 0)
            self.assertIn("sidecar differs from authoritative", oracle_result.stderr)

            paths = make_fixture(root / "tokenizer-ledger")
            rows = paths["model_artifacts"].read_text(encoding="utf-8").splitlines()
            paths["model_artifacts"].write_text(
                "\n".join(row for row in rows if not row.endswith("  merges.txt")) + "\n",
                encoding="utf-8",
            )
            ledger_result = run_builder(preregister_command(paths), check=False)
            self.assertNotEqual(ledger_result.returncode, 0)
            self.assertIn("omits RR2 tokenizer input merges.txt", ledger_result.stderr)

    def test_receipt_builder_binds_fixed_eight_raw_shards_without_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = make_fixture(root / "prereg")
            run_builder(preregister_command(paths))
            identity = json.loads(paths["identity"].read_text(encoding="utf-8"))
            plan = json.loads(paths["selection"].read_text(encoding="utf-8"))
            banks = json.loads(paths["banks"].read_text(encoding="utf-8"))
            with mock.patch.object(
                runner,
                "FORMAL_PG19_DATA_SHA256",
                identity["pg19_data_sha256"],
            ), mock.patch.object(
                runner,
                "FORMAL_PG19_MANIFEST_SHA256",
                identity["pg19_manifest_sha256"],
            ):
                static = runner.make_static_artifact(identity, plan, banks)
            artifact_root = root / "artifacts"
            static_path = artifact_root / "preregistration" / "static-artifact.json"
            write_json(static_path, static)
            static_sha = runner.sha256_json(static)
            protocol_sha = identity["protocol_manifest_sha256"]
            shared_run_receipt = builder.build_run_id_receipt(
                static_artifact_sha256=static_sha,
                protocol_manifest_sha256=protocol_sha,
                nonce=bytes(range(32)),
            )
            run_id = shared_run_receipt["run_id"]
            run_receipt_sha = runner.sha256_json(shared_run_receipt)
            run_receipt_path = artifact_root / "receipts" / "run-id-receipt.json"
            write_json(run_receipt_path, shared_run_receipt)
            for rank in range(8):
                write_json(
                    artifact_root / builder.RAW_SHARD_PATTERN.format(rank=rank),
                    {
                        "schema_version": runner.SHARD_SCHEMA_VERSION,
                        "protocol": runner.PROTOCOL,
                        "rank": rank,
                        "world_size": 8,
                        "static_artifact_sha256": static_sha,
                        "run_id": run_id,
                        "run_id_receipt": shared_run_receipt,
                        "run_id_receipt_sha256": run_receipt_sha,
                    },
                )
            receipt_path = artifact_root / "receipts" / "detached.json"
            result = run_builder(
                [
                    os.fspath(BUILDER),
                    "receipts",
                    "--artifact-root",
                    os.fspath(artifact_root),
                    "--static-artifact",
                    os.fspath(static_path),
                    "--run-id-receipt",
                    os.fspath(run_receipt_path),
                    "--expected-run-id-receipt-sha256",
                    run_receipt_sha,
                    "--run-id",
                    run_id,
                    "--protocol-manifest-sha256",
                    protocol_sha,
                    "--output",
                    os.fspath(receipt_path),
                ]
            )
            receipts = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(len(receipts["shards"]), 8)
            self.assertEqual(
                [row["relative_path"] for row in receipts["shards"]],
                [builder.RAW_SHARD_PATTERN.format(rank=rank) for rank in range(8)],
            )
            self.assertNotIn(os.fspath(root), receipt_path.read_text(encoding="utf-8"))
            status = json.loads(result.stdout)
            self.assertEqual(
                status["receipt_manifest_sha256"], runner.sha256_json(receipts)
            )
            self.assertEqual(status["run_id"], run_id)
            self.assertEqual(status["run_id_receipt_sha256"], run_receipt_sha)

            drifted_shard = (
                artifact_root / builder.RAW_SHARD_PATTERN.format(rank=3)
            )
            drifted = json.loads(drifted_shard.read_text(encoding="utf-8"))
            drifted["run_id_receipt_sha256"] = digest("wrong-shared-receipt")
            write_json(drifted_shard, drifted)
            rejected = run_builder(
                [
                    os.fspath(BUILDER),
                    "receipts",
                    "--artifact-root",
                    os.fspath(artifact_root),
                    "--static-artifact",
                    os.fspath(static_path),
                    "--run-id-receipt",
                    os.fspath(run_receipt_path),
                    "--expected-run-id-receipt-sha256",
                    run_receipt_sha,
                    "--run-id",
                    run_id,
                    "--protocol-manifest-sha256",
                    protocol_sha,
                    "--output",
                    os.fspath(artifact_root / "receipts" / "rejected.json"),
                ],
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("raw shard 3 run-ID receipt SHA drift", rejected.stderr)


if __name__ == "__main__":
    unittest.main()
