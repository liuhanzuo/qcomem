from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import r39_compiled_dispatch_receipts as base
from r39_primary_compact_dispatch import PrimaryDispatchError, verify_primary_shard
from test_r39_primary_compact_dispatch import (
    TINY,
    build_tiny,
    fixture_bindings,
    gpu_assignment_fixture,
    launcher_identity_fixture,
    launcher_identity_sha256,
    reseal_attention_calls,
    verify_fixture,
)


LAUNCH_CONTEXT = {
    "cuda_visible_devices": "GPU-00000000-0000-0000-0000-000000000000",
    "torch_device_index": 0,
    "torch_device_type": "cuda",
    "torch_stream_id": 777,
}


def _kernel() -> object:
    return types.SimpleNamespace(
        metadata={
            "hash": "a" * 64,
            "name": "kernel_unified_attention",
            "num_warps": 4,
            "num_ctas": 1,
            "num_stages": 3,
        }
    )


class R40SecurityRegressionTests(unittest.TestCase):
    def test_proxy_closes_aggregate_import_and_pycompile_bytecode_paths(self) -> None:
        source_root = Path(base.__file__).resolve().parent
        proxy = source_root / "python_proxy_env/bin/python"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            code = root / "frozen-code"
            runtime = root / "runtime"
            result = root / "result"
            pycache = result / "primary/pycache"
            capture = result / "capture"
            identity = result / "identities"
            for directory in (code, runtime, pycache, capture, identity):
                directory.mkdir(parents=True)
            (code / "sibling.py").write_text("VALUE = 17\n", encoding="utf-8")
            runner = code / "aggregate.py"
            runner.write_text(
                "import os,subprocess,sys\n"
                "import sibling\n"
                "assert sibling.VALUE == 17\n"
                "assert sys.dont_write_bytecode is True\n"
                "assert sys.pycache_prefix == os.environ['PYTHONPYCACHEPREFIX']\n"
                "child=subprocess.run([sys.executable,'-c','import sibling; "
                "assert sibling.VALUE == 17'],check=False)\n"
                "raise SystemExit(child.returncode)\n",
                encoding="utf-8",
            )
            wrapper = root / "unused-wrapper.py"
            wrapper.write_text("raise SystemExit('unreachable')\n", encoding="utf-8")
            launcher = root / "launcher.sh"
            launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            preflight = root / "preflight.json"
            preflight.write_text("{}\n", encoding="utf-8")
            contract = result / "primary-launch-contract.json"
            contract.write_text(
                json.dumps(
                    {
                        "argv": ["bash", str(launcher)],
                        "environment": {
                            "PYTHONDONTWRITEBYTECODE": "1",
                            "PYTHONPYCACHEPREFIX": str(pycache),
                        },
                        "schema_version": "forkaudit-r40-v11-primary-launch-contract-v1",
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            contract_sha256 = hashlib.sha256(contract.read_bytes()).hexdigest()
            launcher_sha256 = hashlib.sha256(launcher.read_bytes()).hexdigest()
            proxy_sha256 = hashlib.sha256(proxy.read_bytes()).hexdigest()
            python_sha256 = hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest()
            environment = os.environ.copy()
            environment.update(
                {
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONPYCACHEPREFIX": str(pycache),
                    "PYTHONPATH": str(code),
                    "R40_PRIMARY_REAL_PYTHON": sys.executable,
                    "R40_PRIMARY_EXPECTED_RUNNER": str(runner),
                    "R40_PRIMARY_RANK_WRAPPER": str(wrapper),
                    "R40_PRIMARY_CODE_ROOT": str(code),
                    "R40_PRIMARY_RUNTIME_ROOT": str(runtime),
                    "R40_PRIMARY_BASE_ROOT": str(source_root),
                    "R40_PRIMARY_SOURCE_ROOT": str(source_root),
                    "R40_PRIMARY_CAPTURE_ROOT": str(capture),
                    "R40_PRIMARY_RUNTIME_PREFLIGHT": str(preflight),
                    "R40_PRIMARY_RUNTIME_PREFLIGHT_SHA256": "0" * 64,
                    "R40_PRIMARY_LAUNCHER": str(launcher),
                    "R40_PRIMARY_LAUNCH_IDENTITY_ROOT": str(identity),
                    "R40_PRIMARY_PYCACHE_ROOT": str(pycache),
                    "R40_PRIMARY_LAUNCH_CONTRACT": str(contract),
                    "R40_PRIMARY_LAUNCH_CONTRACT_SHA256": contract_sha256,
                    "R40_PRIMARY_EXPECTED_LAUNCHER_SHA256": launcher_sha256,
                    "R40_PRIMARY_EXPECTED_PROXY_SHA256": proxy_sha256,
                    "R40_PRIMARY_EXPECTED_PYTHON_SHA256": python_sha256,
                }
            )
            aggregate = subprocess.run(
                ["bash", str(proxy), str(runner), "--stage", "aggregate"],
                cwd=code,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(aggregate.returncode, 0, aggregate.stderr)
            compile_result = subprocess.run(
                ["bash", str(proxy), "-m", "py_compile", str(code / "sibling.py")],
                cwd=code,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(compile_result.returncode, 0, compile_result.stderr)
            source_bytecode = [
                path
                for path in code.rglob("*")
                if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
            ]
            self.assertEqual(source_bytecode, [])
            self.assertTrue(any(path.suffix == ".pyc" for path in pycache.rglob("*")))

            # Exercise the byte-identical RR2 aggregate entrypoint itself, not
            # only the small successful command fixture above.  A deliberately
            # incomplete aggregate invocation must reach its domain gate after
            # importing the real dependency closure, and still leave the
            # frozen source tree free of bytecode.
            evidence_root = source_root.parent.parent
            actual_code = evidence_root / "round_04_rr2_package/executed_source/gpu"
            actual_runner = actual_code / "run_qcomem_qwen35_forkaudit_review_revision.py"
            self.assertTrue(actual_runner.is_file())
            actual_environment = dict(environment)
            actual_environment.update(
                {
                    "PYTHONPATH": str(actual_code),
                    "R40_PRIMARY_EXPECTED_RUNNER": str(actual_runner),
                    "R40_PRIMARY_CODE_ROOT": str(actual_code),
                }
            )
            actual_aggregate = subprocess.run(
                [
                    "bash",
                    str(proxy),
                    str(actual_runner),
                    "--stage",
                    "aggregate",
                    "--output",
                    str(result / "deliberately-incomplete-aggregate.json"),
                ],
                cwd=actual_code,
                env=actual_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(actual_aggregate.returncode, 0)
            self.assertIn("aggregate --run-id must be exactly 128 bits", actual_aggregate.stderr)
            actual_source_bytecode = [
                path
                for path in actual_code.rglob("*")
                if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
            ]
            self.assertEqual(actual_source_bytecode, [])

            rejected_environment = dict(environment)
            rejected_environment["PYTHONDONTWRITEBYTECODE"] = "0"
            rejected = subprocess.run(
                ["bash", str(proxy), str(runner), "--stage", "aggregate"],
                cwd=code,
                env=rejected_environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("lacks PYTHONDONTWRITEBYTECODE=1", rejected.stderr)

    def test_success_receipt_is_not_visible_until_post_return_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, _ = base._demo_fixture(Path(temporary))
            recorder = base.DispatchReceiptRecorder(
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                launch_context_provider=lambda: dict(LAUNCH_CONTEXT),
            )
            token = recorder.begin_attention({})
            pending = recorder.prepare_compiled_kernel(_kernel())
            context = recorder._active_attention.get()
            self.assertEqual(recorder.attention_calls, [])
            self.assertEqual(context.launches, [])
            self.assertEqual(context.pending_launches, [pending])
            recorder.seal_compiled_kernel(pending)
            self.assertEqual(context.pending_launches, [])
            self.assertEqual(len(context.launches), 1)
            recorder.finish_attention(token)
            row = recorder.attention_calls[0]
            self.assertIs(row["post_launcher_returned"], True)
            self.assertIs(row["post_return_context_matches"], True)
            self.assertEqual(len(row["call_receipt_sha256"]), 64)

    def test_failed_launcher_path_leaves_no_success_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, _ = base._demo_fixture(Path(temporary))
            recorder = base.DispatchReceiptRecorder(
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                launch_context_provider=lambda: dict(LAUNCH_CONTEXT),
            )
            token = recorder.begin_attention({})
            pending = recorder.prepare_compiled_kernel(_kernel())
            recorder.abort_compiled_kernel(pending)
            recorder.abort_attention(token)
            self.assertEqual(recorder.attention_calls, [])
            self.assertIsNone(recorder._active_attention.get())

    def test_device_or_stream_change_before_return_is_rejected(self) -> None:
        contexts = [dict(LAUNCH_CONTEXT), {**LAUNCH_CONTEXT, "torch_stream_id": 778}]
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, _ = base._demo_fixture(Path(temporary))
            recorder = base.DispatchReceiptRecorder(
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                launch_context_provider=lambda: contexts.pop(0),
            )
            token = recorder.begin_attention({})
            pending = recorder.prepare_compiled_kernel(_kernel())
            with self.assertRaisesRegex(base.DispatchReceiptError, "changed device/stream"):
                recorder.seal_compiled_kernel(pending)
            recorder.abort_compiled_kernel(pending)
            recorder.abort_attention(token)
            self.assertEqual(recorder.attention_calls, [])

    def test_trusted_cuda_context_accepts_default_stream_zero_only_as_integer(self) -> None:
        stream = types.SimpleNamespace(cuda_stream=0)
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(
                is_available=lambda: True,
                current_device=lambda: 0,
                current_stream=lambda device: stream,
            )
        )
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": LAUNCH_CONTEXT["cuda_visible_devices"]},
            clear=False,
        ):
            context = base._trusted_cuda_launch_context()
        self.assertEqual(context["torch_stream_id"], 0)
        stream.cuda_stream = -1
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": LAUNCH_CONTEXT["cuda_visible_devices"]},
            clear=False,
        ):
            with self.assertRaisesRegex(base.DispatchReceiptError, "stream identity"):
                base._trusted_cuda_launch_context()
        stream.cuda_stream = True
        with mock.patch.dict(sys.modules, {"torch": fake_torch}), mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": LAUNCH_CONTEXT["cuda_visible_devices"]},
            clear=False,
        ):
            with self.assertRaisesRegex(base.DispatchReceiptError, "stream identity"):
                base._trusted_cuda_launch_context()

    def test_default_stream_zero_is_sealed_after_normal_return(self) -> None:
        zero_stream_context = {**LAUNCH_CONTEXT, "torch_stream_id": 0}
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, _ = base._demo_fixture(Path(temporary))
            recorder = base.DispatchReceiptRecorder(
                cache_root=cache,
                code_root=code,
                runtime_root=runtime,
                launch_context_provider=lambda: dict(zero_stream_context),
            )
            token = recorder.begin_attention({})
            pending = recorder.prepare_compiled_kernel(_kernel())
            recorder.seal_compiled_kernel(pending)
            recorder.finish_attention(token)
            self.assertEqual(
                recorder.attention_calls[0]["launch_context"]["torch_stream_id"],
                0,
            )

    def test_primary_verifier_accepts_zero_and_rejects_negative_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            candidate = copy.deepcopy(payload)
            for row in candidate["attention_calls"]:
                row[9] = 0
            reseal_attention_calls(candidate)
            verify_fixture(
                candidate,
                cache=cache,
                code=code,
                runtime=runtime,
                frozen_bindings=fixture_bindings(payload),
            )
            rejected = copy.deepcopy(candidate)
            rejected["attention_calls"][0][9] = -1
            reseal_attention_calls(rejected)
            with self.assertRaisesRegex(PrimaryDispatchError, "stream identity invalid"):
                verify_fixture(
                    rejected,
                    cache=cache,
                    code=code,
                    runtime=runtime,
                    frozen_bindings=fixture_bindings(payload),
                )

    def test_v6_ce1_forged_autotune_kwargs_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            candidate = copy.deepcopy(payload)
            config = candidate["tables"]["selected_compile_configurations"][0]
            selected_kwargs = {"BLOCK_M": 2**31, "AUDIT_SENTINEL": "forged"}
            candidate["tables"]["autotune_observations"][0] = {
                "mode": "triton-autotuner",
                "events": [
                    {
                        "selected_kwargs": selected_kwargs,
                        "selected_kwargs_sha256": base._sha256_bytes(
                            base._canonical_bytes(selected_kwargs)
                        ),
                        "num_warps": config["num_warps"],
                        "num_stages": config["num_stages"],
                        "num_ctas": config["num_ctas"],
                    }
                ],
            }
            with self.assertRaisesRegex(PrimaryDispatchError, "call receipt digest"):
                verify_fixture(
                    candidate,
                    cache=cache,
                    code=code,
                    runtime=runtime,
                    frozen_bindings=fixture_bindings(payload),
                )

    def test_v6_ce2_self_consistent_decoy_kernel_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            decoy_dir = cache / "DECOY"
            decoy_dir.mkdir()
            metadata = {
                "hash": "b" * 64,
                "name": "unrelated_decoy_kernel",
                "num_warps": 4,
                "num_ctas": 1,
                "num_stages": 3,
            }
            (decoy_dir / "unrelated_decoy_kernel.json").write_text(
                json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            (decoy_dir / "unrelated_decoy_kernel.cubin").write_bytes(b"decoy")
            (decoy_dir / "unrelated_decoy_kernel.ptx").write_bytes(b"decoy")
            decoy = base.TritonArtifact.from_metadata_file(
                cache, decoy_dir / "unrelated_decoy_kernel.json"
            ).as_dict()
            candidate = copy.deepcopy(payload)
            candidate["tables"]["compiled_artifacts"] = [decoy]
            candidate["tables"]["selected_compile_configurations"] = [
                {
                    "name": decoy["kernel_name"],
                    "hash": decoy["compiler_hash"],
                    **decoy["compile_config"],
                }
            ]
            reseal_attention_calls(candidate)
            with self.assertRaisesRegex(PrimaryDispatchError, "not unified attention"):
                verify_fixture(
                    candidate,
                    cache=cache,
                    code=code,
                    runtime=runtime,
                    frozen_bindings=fixture_bindings(payload),
                )

    def test_v6_ce3_unknown_contradictory_provenance_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            candidate = copy.deepcopy(payload)
            candidate["independent_device_launch_count"] = 0
            candidate["scope"]["vllm_source_sha256"] = "0" * 64
            with self.assertRaisesRegex(PrimaryDispatchError, "fields drift"):
                verify_fixture(
                    candidate,
                    cache=cache,
                    code=code,
                    runtime=runtime,
                    frozen_bindings=fixture_bindings(payload),
                )

    def test_v6_ce4_rank_relabel_without_proxy_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            candidate = copy.deepcopy(payload)
            candidate["rank"] = 1
            for cell in candidate["cells"]:
                cell["rank"] = 1
            for row in candidate["attention_calls"]:
                row[2] = row[2].replace("rank-0/", "rank-1/", 1)
                row[7] = gpu_assignment_fixture()["rows"][1]["uuid"]
            argv = candidate["execution_binding"]["runner_argv"]
            argv[argv.index("--rank") + 1] = "1"
            argv[argv.index("--output") + 1] = "/frozen/forkaudit-shard-1.json"
            candidate["execution_binding"]["runner_argv_sha256"] = base._sha256_bytes(
                json.dumps(argv, sort_keys=True, separators=(",", ":")).encode()
            )
            candidate["execution_binding"]["primary_shard_path"] = "/frozen/forkaudit-shard-1.json"
            candidate["rank_identity"].update(
                {
                    "rank": 1,
                    "cuda_visible_devices": gpu_assignment_fixture()["rows"][1]["uuid"],
                    "assigned_gpu_uuid": gpu_assignment_fixture()["rows"][1]["uuid"],
                    "gpu_assignment_row": gpu_assignment_fixture()["rows"][1],
                    # A producer can rewrite its embedded object, but cannot
                    # replace the finalizer's independently opened proxy file.
                    "launcher_identity": {
                        **launcher_identity_fixture(1),
                        "process_id": 12345,
                    },
                    "launcher_identity_raw_sha256": launcher_identity_sha256(1),
                }
            )
            reseal_attention_calls(candidate)
            with self.assertRaisesRegex(PrimaryDispatchError, "external launcher assignment"):
                verify_fixture(
                    candidate,
                    cache=cache,
                    code=code,
                    runtime=runtime,
                    expected_rank=1,
                    frozen_bindings=fixture_bindings(payload),
                )

    def test_v6_ce5_duplicate_or_unreferenced_table_rows_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            for table_name in (
                "compiled_artifacts",
                "selected_compile_configurations",
                "call_shapes",
                "autotune_observations",
            ):
                candidate = copy.deepcopy(payload)
                candidate["tables"][table_name].append(
                    copy.deepcopy(candidate["tables"][table_name][0])
                )
                with self.assertRaisesRegex(PrimaryDispatchError, "duplicate rows"):
                    verify_fixture(
                        candidate,
                        cache=cache,
                        code=code,
                        runtime=runtime,
                        frozen_bindings=fixture_bindings(payload),
                    )

    def test_post_return_flags_cannot_be_self_consistently_cleared(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache, code, runtime, payload = build_tiny(Path(temporary))
            candidate = copy.deepcopy(payload)
            candidate["attention_calls"][0][10] = False
            reseal_attention_calls(candidate)
            with self.assertRaisesRegex(PrimaryDispatchError, "successful launcher return"):
                verify_fixture(
                    candidate,
                    cache=cache,
                    code=code,
                    runtime=runtime,
                    frozen_bindings=fixture_bindings(payload),
                )

    def test_dense_fallback_in_primary_shard_is_rejected(self) -> None:
        calls = [
            {
                "request_index": 0,
                "layer_idx": 1,
                "query_tokens": 32 if round_index == 0 else 1,
                "physical_block_pool_shape": [34, 128, 2, 256],
                "active_block_table_shape": [1, 33],
                "kv_tokens": 4127 + round_index,
                "softmax_scale": 0.0625,
            }
            for round_index in range(2)
        ]
        ledger = {
            "request_index": 0,
            "total_calls": 2,
            "verified": True,
            "dense_fallback_calls": 1,
            "calls": calls,
        }
        shard = {
            "schema_version": "qcomem-forkaudit-review-shard-v1",
            "protocol": "qcomem-qwen35-forkaudit-review-revision-v1",
            "status": "completed_formal_gpu_shard",
            "rank": 0,
            "world_size": 8,
            "protocol_config": {
                "resident_counts": [1],
                "generation_steps": 2,
                "document_tokens": 3,
                "factorial_arm_ids": ["kv=tiny-kv|gdn=tiny-gdn"],
            },
            "factorial": [
                {
                    "resident_count": 1,
                    "cells": [
                        {
                            "arm_id": "kv=tiny-kv|gdn=tiny-gdn",
                            "memory_kernel_ledgers": [copy.deepcopy(ledger)],
                            "witness_kernel_ledgers": [copy.deepcopy(ledger)],
                        }
                    ],
                }
            ],
        }
        with self.assertRaisesRegex(PrimaryDispatchError, "used dense fallback"):
            verify_primary_shard(shard, expected_rank=0, geometry=TINY)

    def test_formal_launcher_and_proxy_contain_fail_closed_gates(self) -> None:
        source_root = Path(base.__file__).resolve().parent
        package_root = source_root.parent
        evidence_root = package_root.parent
        launcher = (source_root / "r39_primary_formal_h20.sh").read_text(encoding="utf-8")
        proxy = (source_root / "python_proxy_env/bin/python").read_text(encoding="utf-8")
        for required in (
            "R40_H20_EXECUTION_AUTHORIZED",
            "rank-launch-identities",
            "runtime-preflight",
            "terminal-files.sha256",
            "COMPLETE",
        ):
            self.assertIn(required, launcher)
        for required in (
            "forkaudit-r40-proxy-rank-launch-v1",
            "launcher-rank-identity",
            "expected-launcher-rank-identity-sha256",
            "ln \"$temporary\" \"$identity\"",
            "R40_PRIMARY_PYCACHE_ROOT",
            "R40_PRIMARY_LAUNCH_CONTRACT_SHA256",
            "R40_PRIMARY_EXPECTED_LAUNCHER_SHA256",
            "R40_PRIMARY_EXPECTED_PROXY_SHA256",
            "R40_PRIMARY_EXPECTED_PYTHON_SHA256",
            '"$REAL_PYTHON" -B "$@"',
        ):
            self.assertIn(required, proxy)

        self.assertIn("qcomem_r40_primary_compiled_dispatch_v11_20260827k", launcher)
        self.assertIn("r40-primary-compiled-dispatch-v11-20260827k", launcher)
        self.assertEqual(launcher.count('find "$PRIMARY_CODE"'), 3)
        self.assertIn('PYTHONDONTWRITEBYTECODE=1 \\\nPYTHONPYCACHEPREFIX="$PRIMARY_PYCACHE_ROOT"', launcher)
        self.assertIn("forkaudit-r40-v11-primary-launch-contract-v1", launcher)
        authorization = launcher.index('[[ "${R40_H20_EXECUTION_AUTHORIZED:-}" == "yes" ]]')
        code_presence = launcher.index('[[ -d "$PRIMARY_CODE" ]]')
        first_bytecode_gate = launcher.index('find "$PRIMARY_CODE"')
        result_root_check = launcher.index('[[ ! -e "$RESULT_ROOT" ]]')
        result_root_creation = launcher.index('mkdir -p "$RESULT_ROOT/supervisor"')
        second_bytecode_gate = launcher.index('find "$PRIMARY_CODE"', first_bytecode_gate + 1)
        primary_launch = launcher.index('bash "$PRIMARY_LAUNCHER"')
        terminal_bytecode_gate = launcher.index(
            'find "$PRIMARY_CODE"', second_bytecode_gate + 1
        )
        self.assertLess(authorization, code_presence)
        self.assertLess(code_presence, first_bytecode_gate)
        self.assertLess(first_bytecode_gate, result_root_check)
        self.assertLess(result_root_check, result_root_creation)
        self.assertLess(result_root_creation, second_bytecode_gate)
        self.assertLess(second_bytecode_gate, primary_launch)
        self.assertLess(primary_launch, terminal_bytecode_gate)

        invocation_audit = json.loads(
            (package_root / "python-invocation-audit.json").read_text(encoding="utf-8")
        )
        primary_launcher = (
            evidence_root
            / "round_04_rr2_package/executed_source/gpu/launch_qcomem_qwen35_forkaudit_review_revision_8gpu.sh"
        ).read_text(encoding="utf-8")
        self.assertEqual(primary_launcher.count('"$PYTHON"'), 21)
        self.assertEqual(invocation_audit["audited_launcher_python_literal_sites"], 21)
        self.assertEqual(
            invocation_audit["audited_launcher_executing_python_command_sites"], 20
        )
        inventory = invocation_audit["direct_invocation_inventory"]
        self.assertEqual(len(inventory), 20)
        self.assertEqual(
            sum(row["successful_path_executions"] for row in inventory), 35
        )

        equivalence = json.loads(
            (package_root / "v10-v11-scientific-equivalence.json").read_text(
                encoding="utf-8"
            )
        )
        for name, expected in equivalence[
            "byte_identical_v10_v11_executed_scientific_files"
        ].items():
            observed = hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            self.assertEqual(observed, expected, name)

        forbidden = sorted(
            str(path.relative_to(package_root))
            for path in package_root.rglob("*")
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        )
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
