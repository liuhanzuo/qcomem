from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import freeze_v9
import static_audit_v9
import v9_guard
import v9_runtime
from authorized_launcher_v9 import run_authorized_campaign
from v9_guard import (
    BINDING_DOMAIN,
    HASH_KEYS,
    TERM_IDS,
    Reject,
    _BASE,
    _L,
    _scalar_mult,
    canonical_bytes,
    canonical_load,
    canonical_tar,
    deterministic_gzip,
    derive_spawned_specs,
    designer_attestation,
    digest_bytes,
    digest_file,
    ed25519_verify,
    isolated_torch_probe,
    lifecycle_gate,
    measure_hashes,
    runner_commitments,
    snapshot_commitments,
    strict_gzip,
    validate_execution_contract,
    validate_runtime_expectation,
    validate_terminal_tree,
    verify_archive,
    verify_operator_binding,
)
from v9_runtime import ProtectedParent


PACKAGE = Path(__file__).resolve().parent
UUID0 = "GPU-01234567-89ab-cdef-0123-456789abcdef"


def sample_provenance() -> dict:
    return {
        "binding_sha256": "1" * 64,
        "execution_contract_sha256": "2" * 64,
        "probe_report_sha256": "3" * 64,
        "runtime_expectation_sha256": "4" * 64,
        "spawned_specs_sha256": "5" * 64,
    }


def tree_manifest(root: Path) -> list[dict]:
    rows = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        rows.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest_file(path),
                "size": path.stat().st_size,
            }
        )
    return rows


def archive_fixture() -> tuple[bytes, bytes]:
    payload = b"frozen\n"
    row = {"path": "x", "sha256": digest_bytes(payload), "size": len(payload)}
    ledger = canonical_bytes(
        {
            "files": [row],
            "freeze_timestamp": v9_guard.FREEZE_TIMESTAMP,
            "schema_version": "forkaudit-v9-source-ledger-v1",
        }
    )
    archive = deterministic_gzip(canonical_tar([("source-ledger.json", ledger), ("x", payload)]))
    return archive, ledger


def custom_tar(entries: list[tuple[str, bytes, int, int]]) -> bytes:
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, raw, mode, mtime in entries:
            info = tarfile.TarInfo(name)
            info.size = len(raw)
            info.mode = mode
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime = mtime
            archive.addfile(info, io.BytesIO(raw))
    return deterministic_gzip(output.getvalue())


def successful_receipt() -> dict:
    return {
        "inventory_verified": True,
        "kill_completion": {"completed": True, "errors": [], "required": False},
        "post_rehash_verified": True,
        "provenance": sample_provenance(),
        "provenance_verified": True,
        "receipts_verified": True,
        "schema_version": "forkaudit-v9-lifecycle-gate-v1",
        "verification_complete": True,
        "workers": [
            {
                "death_confirmed": True,
                "exit_code": 0,
                "fault_id": fault_id,
                "kill_completed": True,
                "kill_required": False,
                "kill_sent": False,
                "pid": index + 1000,
                "spawned": True,
                "terminate_sent": False,
                "wait_completed": True,
            }
            for index, fault_id in enumerate(TERM_IDS)
        ],
    }


def write_runner_manifest(base: Path, runner: Path) -> tuple[Path, str, str]:
    manifest_path = base / "runner-manifest.json"
    manifest_path.write_bytes(canonical_bytes(tree_manifest(runner)))
    manifest_sha, inventory_sha, _ = runner_commitments(runner, manifest_path)
    return manifest_path, manifest_sha, inventory_sha


def lifecycle_fixture(base: Path) -> dict:
    archive = base / "archive.tgz"
    ledger = base / "ledger.json"
    archive.write_bytes(b"archive")
    ledger.write_bytes(b"ledger")
    snapshot = base / "snapshot"
    runner = base / "runner"
    snapshot.mkdir()
    runner.mkdir()
    (snapshot / "public.txt").write_text("public\n")
    (runner / "run.py").write_text("print('cpu-only')\n")
    manifest_path, manifest_sha, inventory_sha = write_runner_manifest(base, runner)
    paths = {
        "archive": archive,
        "ledger": ledger,
        "snapshot": snapshot,
        "runner": runner,
        "runner_manifest": manifest_path,
        "runner_manifest_sha256": manifest_sha,
        "runner_inventory_sha256": inventory_sha,
    }
    return paths


def execution_runtime_fixture(base: Path, *, real_python: bool = False) -> dict:
    runner = base / "runner"
    (runner / "bin").mkdir(parents=True)
    (runner / "cfg").mkdir()
    (runner / "tools").mkdir()
    (runner / "torch").mkdir()
    python_path = runner / "bin" / "python"
    if real_python:
        shutil.copyfile(sys.executable, python_path)
        python_path.chmod(0o755)
        framework_binary = Path(sys.executable).resolve().parent.parent / "Python3"
        if framework_binary.is_file():
            shutil.copyfile(framework_binary, runner / "Python3")
            (runner / "Python3").chmod(0o755)
            app_binary = (
                framework_binary.parent
                / "Resources"
                / "Python.app"
                / "Contents"
                / "MacOS"
                / "Python"
            )
            copied_app = runner / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
            copied_app.parent.mkdir(parents=True)
            shutil.copyfile(app_binary, copied_app)
            copied_app.chmod(0o755)
    else:
        python_path.write_bytes(b"not-executed-python-fixture\n")
        python_path.chmod(0o755)
    shutil.copyfile(PACKAGE / "torch_probe_v9.py", runner / "tools" / "torch_probe_v9.py")
    (runner / "run.py").write_text("print('cpu-only')\n")
    (runner / "cfg" / "formal.json").write_bytes(b"{}\n")
    (runner / "torch" / "__init__.py").write_text(
        "__version__='0.test'\n"
        "class _P:\n"
        f" uuid={UUID0!r}\n"
        "class _C:\n"
        " def get_device_properties(self,index): return _P()\n"
        "cuda=_C()\n"
    )
    manifest_path, manifest_sha, inventory_sha = write_runner_manifest(base, runner)
    rows = tree_manifest(runner)
    role_by_path = {
        "bin/python": "python-executable",
        "tools/torch_probe_v9.py": "provenance-probe",
        "torch/__init__.py": "torch-package-root",
    }
    expectation = {
        "cwd": ".",
        "device": {"index": 0, "physical_uuid": UUID0, "visibility": UUID0},
        "probe": {
            "manifest_path": "tools/torch_probe_v9.py",
            "sha256": digest_file(runner / "tools" / "torch_probe_v9.py"),
        },
        "python": {
            "cache_tag": sys.implementation.cache_tag,
            "implementation": "cpython",
            "manifest_path": "bin/python",
            "sha256": digest_file(python_path),
            "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        },
        "runner_inventory_sha256": inventory_sha,
        "runner_manifest_sha256": manifest_sha,
        "runtime_closure": [
            {
                "manifest_path": row["path"],
                "role": role_by_path.get(row["path"], "python-torch-runtime"),
                "sha256": row["sha256"],
                "size": row["size"],
            }
            for row in rows
        ],
        "schema_version": "forkaudit-v9-runtime-expectation-v1",
        "torch": {
            "init_manifest_path": "torch/__init__.py",
            "init_sha256": digest_file(runner / "torch" / "__init__.py"),
            "version": "0.test",
        },
    }
    expectation_raw = canonical_bytes(expectation)
    expectation_path = base / "runtime-expectation.json"
    expectation_path.write_bytes(expectation_raw)
    expectation_sha = digest_bytes(expectation_raw)
    base_argv = [str(python_path.resolve()), "run.py", "--config=cfg/formal.json", "--dry-run", "unit"]
    workers = []
    for fault_id in TERM_IDS:
        argv = base_argv + ["--fault-id", fault_id]
        workers.append(
            {
                "argv": argv,
                "argv_schema": [
                    {"index": 0, "kind": "path", "manifest_path": "bin/python", "spelling": argv[0]},
                    {"index": 1, "kind": "path", "manifest_path": "run.py", "spelling": "run.py"},
                    {
                        "index": 2,
                        "kind": "option-path",
                        "manifest_path": "cfg/formal.json",
                        "option": "--config",
                        "spelling": "cfg/formal.json",
                    },
                    {"index": 3, "kind": "option", "option": "--dry-run"},
                    {"index": 4, "kind": "literal", "literal": "unit"},
                    {"index": 5, "kind": "option", "option": "--fault-id"},
                    {"index": 6, "kind": "literal", "literal": fault_id},
                ],
                "cwd": ".",
                "env": {
                    "CUDA_VISIBLE_DEVICES": UUID0,
                    "FORKAUDIT_CONFIG_PATH": "cfg/formal.json",
                    "LC_ALL": "C",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "PYTHONNOUSERSITE": "1",
                },
                "env_schema": [
                    {"key": "CUDA_VISIBLE_DEVICES", "kind": "uuid-list", "visibility": UUID0},
                    {
                        "key": "FORKAUDIT_CONFIG_PATH",
                        "kind": "path",
                        "manifest_path": "cfg/formal.json",
                        "spelling": "cfg/formal.json",
                    },
                    {"key": "LC_ALL", "kind": "literal", "literal": "C"},
                    {"key": "PYTHONDONTWRITEBYTECODE", "kind": "literal", "literal": "1"},
                    {"key": "PYTHONHASHSEED", "kind": "literal", "literal": "0"},
                    {"key": "PYTHONNOUSERSITE", "kind": "literal", "literal": "1"},
                ],
                "fault_id": fault_id,
            }
        )
    contract = {
        "runner_inventory_sha256": inventory_sha,
        "runner_manifest_sha256": manifest_sha,
        "runtime_expectation_sha256": expectation_sha,
        "schema_version": "forkaudit-v9-execution-contract-v1",
        "timeout_seconds": 60,
        "workers": workers,
    }
    contract_raw = canonical_bytes(contract)
    contract_path = base / "execution-contract.json"
    contract_path.write_bytes(contract_raw)
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_raw": contract_raw,
        "expectation": expectation,
        "expectation_path": expectation_path,
        "expectation_raw": expectation_raw,
        "expectation_sha256": expectation_sha,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_sha,
        "inventory_sha256": inventory_sha,
        "runner": runner,
    }


def refresh_runtime_expectation(runtime: dict) -> None:
    runner = runtime["runner"]
    runtime["manifest_path"].write_bytes(canonical_bytes(tree_manifest(runner)))
    manifest_sha, inventory_sha, _ = runner_commitments(runner, runtime["manifest_path"])
    expectation = runtime["expectation"]
    expectation["runner_manifest_sha256"] = manifest_sha
    expectation["runner_inventory_sha256"] = inventory_sha
    expectation["python"]["sha256"] = digest_file(runner / expectation["python"]["manifest_path"])
    expectation["probe"]["sha256"] = digest_file(runner / expectation["probe"]["manifest_path"])
    expectation["torch"]["init_sha256"] = digest_file(runner / expectation["torch"]["init_manifest_path"])
    role_by_path = {
        expectation["python"]["manifest_path"]: "python-executable",
        expectation["probe"]["manifest_path"]: "provenance-probe",
        expectation["torch"]["init_manifest_path"]: "torch-package-root",
    }
    expectation["runtime_closure"] = [
        {
            "manifest_path": row["path"],
            "role": role_by_path.get(row["path"], "python-torch-runtime"),
            "sha256": row["sha256"],
            "size": row["size"],
        }
        for row in tree_manifest(runner)
    ]
    raw = canonical_bytes(expectation)
    runtime["expectation_path"].write_bytes(raw)
    runtime["expectation_raw"] = raw
    runtime["expectation_sha256"] = digest_bytes(raw)
    runtime["manifest_sha256"] = manifest_sha
    runtime["inventory_sha256"] = inventory_sha
    contract = runtime["contract"]
    contract["runner_manifest_sha256"] = manifest_sha
    contract["runner_inventory_sha256"] = inventory_sha
    contract["runtime_expectation_sha256"] = runtime["expectation_sha256"]
    python_spelling = str((runner / expectation["python"]["manifest_path"]).resolve(strict=True))
    for worker in contract["workers"]:
        worker["argv"][0] = python_spelling
        worker["argv_schema"][0]["spelling"] = python_spelling
    contract_raw = canonical_bytes(contract)
    runtime["contract_path"].write_bytes(contract_raw)
    runtime["contract_raw"] = contract_raw


def signed_fixture(base: Path, *, real_python: bool = True, sleeping: bool = False,
                   terminal_root: Path | None = None, attempt: int = 1,
                   run_nonce: str = "a" * 64) -> dict:
    archive_raw, ledger_raw = archive_fixture()
    archive = base / "method.tgz"
    ledger = base / "source-ledger.json"
    archive.write_bytes(archive_raw)
    ledger.write_bytes(ledger_raw)
    snapshot = base / "snapshot"
    snapshot.mkdir()
    if terminal_root is None:
        terminal_root = base / "terminals"
        terminal_root.mkdir()
    (snapshot / "public.txt").write_text("sealed\n")
    runtime = execution_runtime_fixture(base, real_python=real_python)
    if sleeping:
        (runtime["runner"] / "run.py").write_text("import time\ntime.sleep(3600)\n")
        refresh_runtime_expectation(runtime)
    snapshot_sha, snapshot_inventory_sha, _ = snapshot_commitments(snapshot)
    payload = {
        "approved_archive_sha256": digest_file(archive),
        "approved_attempt": attempt,
        "approved_execution_contract_sha256": digest_file(runtime["contract_path"]),
        "approved_runner_inventory_sha256": runtime["inventory_sha256"],
        "approved_runner_manifest_sha256": runtime["manifest_sha256"],
        "approved_runtime_expectation_sha256": runtime["expectation_sha256"],
        "approved_run_nonce": run_nonce,
        "approved_snapshot_inventory_sha256": snapshot_inventory_sha,
        "approved_snapshot_sha256": snapshot_sha,
        "approved_source_ledger_sha256": digest_file(ledger),
        "approved_terminal_root": str(terminal_root.resolve(strict=True)),
        "operator_id": "independent-test-operator",
        "published_uri": "https://operator.example/r40/v9-binding.json",
        "schema_version": "forkaudit-v9-operator-binding-payload-v1",
        "trust_root_id": v9_guard.TRUST_ROOT_ID,
    }
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    public, signature = _rfc_test_sign(seed, BINDING_DOMAIN + canonical_bytes(payload))
    binding = {
        "payload": payload,
        "schema_version": "forkaudit-v9-signed-operator-binding-v1",
        "signature": {
            "algorithm": "ed25519",
            "key_id": v9_guard.TRUST_ROOT_ID,
            "signature_hex": signature.hex(),
        },
    }
    binding_path = base / "operator-binding.json"
    binding_path.write_bytes(canonical_bytes(binding))
    return {
        "archive": archive,
        "binding": binding,
        "binding_path": binding_path,
        "ledger": ledger,
        "payload": payload,
        "public": public,
        "runtime": runtime,
        "snapshot": snapshot,
        "terminal_root": terminal_root,
        "attempt": attempt,
        "run_nonce": run_nonce,
    }


def resign_fixture(fixture: dict) -> None:
    runtime = fixture["runtime"]
    payload = fixture["payload"]
    payload["approved_execution_contract_sha256"] = digest_file(runtime["contract_path"])
    payload["approved_runner_inventory_sha256"] = runtime["inventory_sha256"]
    payload["approved_runner_manifest_sha256"] = runtime["manifest_sha256"]
    payload["approved_runtime_expectation_sha256"] = runtime["expectation_sha256"]
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    _, signature = _rfc_test_sign(seed, BINDING_DOMAIN + canonical_bytes(payload))
    fixture["binding"]["payload"] = payload
    fixture["binding"]["signature"]["signature_hex"] = signature.hex()
    fixture["binding_path"].write_bytes(canonical_bytes(fixture["binding"]))


def plan_from_fixture(fixture: dict) -> AuthorizedPlan:
    runtime = fixture["runtime"]
    with mock.patch.object(v9_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
        v9_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", fixture["public"].hex()
    ):
        return prepare_authorized_plan(
            fixture["binding_path"],
            fixture["archive"],
            fixture["ledger"],
            fixture["snapshot"],
            runtime["contract_path"],
            runtime["expectation_path"],
            runtime["runner"],
            runtime["manifest_path"],
        )


def _encode_point(point: tuple[int, int]) -> bytes:
    x, y = point
    return (y | ((x & 1) << 255)).to_bytes(32, "little")


def _rfc_test_sign(seed: bytes, message: bytes) -> tuple[bytes, bytes]:
    expanded = hashlib.sha512(seed).digest()
    scalar_raw = bytearray(expanded[:32])
    scalar_raw[0] &= 248
    scalar_raw[31] &= 63
    scalar_raw[31] |= 64
    scalar = int.from_bytes(scalar_raw, "little")
    public = _encode_point(_scalar_mult(_BASE, scalar))
    nonce = int.from_bytes(hashlib.sha512(expanded[32:] + message).digest(), "little") % _L
    encoded_r = _encode_point(_scalar_mult(_BASE, nonce))
    challenge = int.from_bytes(hashlib.sha512(encoded_r + public + message).digest(), "little") % _L
    signature = encoded_r + ((nonce + challenge * scalar) % _L).to_bytes(32, "little")
    return public, signature


class V9Tests(unittest.TestCase):
    def bad(self, function, *args, **kwargs):
        with self.assertRaises(Reject):
            function(*args, **kwargs)

    def test_01_canonical_json_rejects_duplicate_and_whitespace(self):
        self.assertEqual(canonical_load(b'{"a":1}\n', "x"), {"a": 1})
        self.bad(canonical_load, b'{"a":1,"a":2}\n', "x")
        self.bad(canonical_load, b'{ "a": 1 }\n', "x")
        self.bad(canonical_load, b'{"a":NaN}\n', "x")

    def test_02_ed25519_rfc_vector_and_mutation(self):
        public = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
        signature = bytes.fromhex(
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        )
        self.assertTrue(ed25519_verify(public, b"", signature))
        self.assertFalse(ed25519_verify(public, b"x", signature))
        self.assertFalse(ed25519_verify(b"\0" * 32, b"", signature))

    def test_03_operator_api_has_no_callback_or_key_override(self):
        parameters = tuple(inspect.signature(verify_operator_binding).parameters)
        self.assertEqual(
            parameters,
            (
                "raw_binding",
                "archive_path",
                "source_ledger_path",
                "snapshot_root",
                "execution_contract_path",
                "runtime_expectation_path",
                "runner_root",
                "runner_manifest_path",
            ),
        )
        v9_guard._compiled_trust_root()
        with self.assertRaises(TypeError):
            verify_operator_binding(b"{}\n", "a", "b", "c", "d", "e", "f", "g", lambda _: True)

    def test_04_arbitrary_self_claim_and_noncanonical_binding_rejected(self):
        template = json.loads((PACKAGE / "operator-binding.template.json").read_text())
        template["payload"].update(
            {
                "approved_archive_sha256": "a" * 64,
                "approved_execution_contract_sha256": "b" * 64,
                "approved_runner_inventory_sha256": "c" * 64,
                "approved_runner_manifest_sha256": "d" * 64,
                "approved_runtime_expectation_sha256": "e" * 64,
                "approved_snapshot_inventory_sha256": "f" * 64,
                "approved_snapshot_sha256": "1" * 64,
                "approved_source_ledger_sha256": "2" * 64,
                "operator_id": "self",
                "published_uri": "https://operator.example/r40/v9.json",
            }
        )
        template["signature"]["signature_hex"] = "0" * 128
        self.bad(verify_operator_binding, canonical_bytes(template), "a", "b", "c", "d", "e", "f", "g")
        raw = json.dumps(template, indent=2).encode()
        self.bad(verify_operator_binding, raw, "a", "b", "c", "d", "e", "f", "g")

    def test_05_signed_binding_recomputes_all_artifact_commitments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_raw, ledger_raw = archive_fixture()
            archive = root / "method.tgz"
            ledger = root / "source-ledger.json"
            archive.write_bytes(archive_raw)
            ledger.write_bytes(ledger_raw)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            (snapshot / "public.txt").write_text("sealed\n")
            runtime = execution_runtime_fixture(root)
            snapshot_sha, inventory_sha, _ = snapshot_commitments(snapshot)
            payload = {
                "approved_archive_sha256": digest_file(archive),
                "approved_execution_contract_sha256": digest_file(runtime["contract_path"]),
                "approved_runner_inventory_sha256": runtime["inventory_sha256"],
                "approved_runner_manifest_sha256": runtime["manifest_sha256"],
                "approved_runtime_expectation_sha256": runtime["expectation_sha256"],
                "approved_snapshot_inventory_sha256": inventory_sha,
                "approved_snapshot_sha256": snapshot_sha,
                "approved_source_ledger_sha256": digest_file(ledger),
                "operator_id": "independent-test-operator",
                "published_uri": "https://operator.example/r40/v9-binding.json",
                "schema_version": "forkaudit-v9-operator-binding-payload-v1",
                "trust_root_id": v9_guard.TRUST_ROOT_ID,
            }
            seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
            public, signature = _rfc_test_sign(seed, BINDING_DOMAIN + canonical_bytes(payload))
            binding = {
                "payload": payload,
                "schema_version": "forkaudit-v9-signed-operator-binding-v1",
                "signature": {
                    "algorithm": "ed25519",
                    "key_id": v9_guard.TRUST_ROOT_ID,
                    "signature_hex": signature.hex(),
                },
            }
            with mock.patch.object(v9_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
                v9_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", public.hex()
            ):
                self.assertEqual(
                    verify_operator_binding(
                        canonical_bytes(binding),
                        archive,
                        ledger,
                        snapshot,
                        runtime["contract_path"],
                        runtime["expectation_path"],
                        runtime["runner"],
                        runtime["manifest_path"],
                    ),
                    payload,
                )
                runner_member = runtime["runner"] / "run.py"
                original_runner = runner_member.read_bytes()
                runner_member.write_bytes(b"mutated after operator signature\n")
                self.bad(
                    verify_operator_binding,
                    canonical_bytes(binding),
                    archive,
                    ledger,
                    snapshot,
                    runtime["contract_path"],
                    runtime["expectation_path"],
                    runtime["runner"],
                    runtime["manifest_path"],
                )
                runner_member.write_bytes(original_runner)
                original_manifest = runtime["manifest_path"].read_bytes()
                runtime["manifest_path"].write_bytes(canonical_bytes([]))
                self.bad(
                    verify_operator_binding,
                    canonical_bytes(binding),
                    archive,
                    ledger,
                    snapshot,
                    runtime["contract_path"],
                    runtime["expectation_path"],
                    runtime["runner"],
                    runtime["manifest_path"],
                )
                runtime["manifest_path"].write_bytes(original_manifest)
                (snapshot / "extra.txt").write_text("not signed\n")
                self.bad(
                    verify_operator_binding,
                    canonical_bytes(binding),
                    archive,
                    ledger,
                    snapshot,
                    runtime["contract_path"],
                    runtime["expectation_path"],
                    runtime["runner"],
                    runtime["manifest_path"],
                )

    def test_06_snapshot_dual_commitments_and_attestation_exactness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_bytes(b"x")
            snapshot_sha, inventory_sha, _ = snapshot_commitments(root)
            value = {
                "inputs_limited_to_snapshot": True,
                "no_prior_faults_seen": True,
                "no_private_source_seen": True,
                "snapshot_inventory_sha256": inventory_sha,
                "snapshot_sha256": snapshot_sha,
            }
            designer_attestation(value, snapshot_sha, inventory_sha)
            changed = dict(value)
            changed["snapshot_inventory_sha256"] = "0" * 64
            self.bad(designer_attestation, changed, snapshot_sha, inventory_sha)
            changed = dict(value)
            changed["inputs_limited_to_snapshot"] = 1
            self.bad(designer_attestation, changed, snapshot_sha, inventory_sha)

    def test_07_snapshot_rejects_symlink_and_hardlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "a"
            source.write_bytes(b"x")
            (root / "s").symlink_to(source)
            self.bad(snapshot_commitments, root)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "a"
            source.write_bytes(b"x")
            os.link(source, root / "b")
            self.bad(snapshot_commitments, root)

    def test_08_canonical_archive_accepts_exact_ledger(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "method.tgz"
            archive, ledger = archive_fixture()
            path.write_bytes(archive)
            result = verify_archive(path, digest_bytes(archive), digest_bytes(ledger))
            self.assertEqual(result["members"], 2)
            self.assertEqual(strict_gzip(archive), canonical_tar([
                ("source-ledger.json", ledger), ("x", b"frozen\n")
            ]))

    def test_09_archive_rejects_traversal_member(self):
        _, ledger = archive_fixture()
        bad_archive = custom_tar(
            [("../x", b"x", 0o444, 0), ("source-ledger.json", ledger, 0o444, 0)]
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.tgz"
            path.write_bytes(bad_archive)
            self.bad(verify_archive, path, digest_file(path), digest_bytes(ledger))

    def test_10_archive_rejects_noncanonical_mode_and_mtime(self):
        _, ledger = archive_fixture()
        for mode, mtime in ((0o644, 0), (0o444, 1)):
            with self.subTest(mode=mode, mtime=mtime), tempfile.TemporaryDirectory() as temporary:
                archive = custom_tar(
                    [("source-ledger.json", ledger, 0o444, 0), ("x", b"frozen\n", mode, mtime)]
                )
                path = Path(temporary) / "bad.tgz"
                path.write_bytes(archive)
                self.bad(verify_archive, path, digest_file(path), digest_bytes(ledger))

    def test_11_archive_rejects_multiple_gzip_streams_and_trailing_bytes(self):
        archive, ledger = archive_fixture()
        for bad_archive in (archive + archive, archive + b"trailing"):
            with self.subTest(size=len(bad_archive)), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "bad.tgz"
                path.write_bytes(bad_archive)
                self.bad(verify_archive, path, digest_file(path), digest_bytes(ledger))

    def test_12_archive_rejects_nonexact_ledger_inventory_and_hardlink(self):
        _, ledger = archive_fixture()
        archive = deterministic_gzip(
            canonical_tar(
                [("source-ledger.json", ledger), ("x", b"frozen\n"), ("y", b"extra")]
            )
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "bad.tgz"
            path.write_bytes(archive)
            self.bad(verify_archive, path, digest_file(path), digest_bytes(ledger))
            link = root / "link.tgz"
            os.link(path, link)
            self.bad(verify_archive, path, digest_bytes(archive), digest_bytes(ledger))

    def contract_call(self, runtime: dict, contract: dict | None = None):
        value = runtime["contract"] if contract is None else contract
        raw = canonical_bytes(value)
        return validate_execution_contract(
            raw,
            digest_bytes(raw),
            runtime["expectation_path"],
            runtime["runner"],
            runtime["manifest_path"],
            runtime["manifest_sha256"],
            runtime["inventory_sha256"],
            runtime["expectation_sha256"],
        )

    def test_13_execution_contract_uses_exact_typed_schema_for_every_index_and_env(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = execution_runtime_fixture(root)
            self.assertEqual(self.contract_call(runtime), runtime["contract"])
            self.assertEqual(
                [row["kind"] for row in runtime["contract"]["workers"][0]["argv_schema"]],
                ["path", "path", "option-path", "option", "literal", "option", "literal"],
            )

    def test_14_execution_contract_binds_separate_path_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = execution_runtime_fixture(root)
            contract = json.loads(json.dumps(runtime["contract"]))
            for worker in contract["workers"]:
                worker["argv"] = worker["argv"][:2] + ["--config", "cfg/formal.json"] + worker["argv"][3:]
                worker["argv_schema"] = worker["argv_schema"][:2] + [
                    {"index": 2, "kind": "option", "option": "--config"},
                    {"index": 3, "kind": "path", "manifest_path": "cfg/formal.json", "spelling": "cfg/formal.json"},
                    {"index": 4, "kind": "option", "option": "--dry-run"},
                    {"index": 5, "kind": "literal", "literal": "unit"},
                    {"index": 6, "kind": "option", "option": "--fault-id"},
                    {"index": 7, "kind": "literal", "literal": worker["fault_id"]},
                ]
            self.contract_call(runtime, contract)

    def test_15_execution_contract_rejects_missing_wrong_and_unanchored_typed_rows(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = execution_runtime_fixture(root)
            base = runtime["contract"]
            mutants = []
            missing = json.loads(json.dumps(base))
            missing["workers"][0]["argv_schema"] = missing["workers"][0]["argv_schema"][:-1]
            mutants.append(missing)
            wrong_index = json.loads(json.dumps(base))
            wrong_index["workers"][0]["argv_schema"][2]["index"] = 3
            mutants.append(wrong_index)
            wrong_kind = json.loads(json.dumps(base))
            wrong_kind["workers"][0]["argv_schema"][0] = {
                "index": 0,
                "kind": "literal",
                "literal": wrong_kind["workers"][0]["argv"][0],
            }
            mutants.append(wrong_kind)
            unbound = json.loads(json.dumps(base))
            unbound["workers"][0]["argv"].append("weights.safetensors")
            unbound["workers"][0]["argv_schema"].append(
                {"index": 7, "kind": "path", "manifest_path": "weights.safetensors", "spelling": "weights.safetensors"}
            )
            mutants.append(unbound)
            for contract in mutants:
                with self.subTest(argv=contract["workers"][0]["argv"]):
                    self.bad(self.contract_call, runtime, contract)
            raw = runtime["contract_raw"]
            self.bad(
                validate_execution_contract,
                raw,
                digest_bytes(raw),
                runtime["expectation_path"],
                runtime["runner"],
                tree_manifest(runtime["runner"]),
                runtime["manifest_sha256"],
                runtime["inventory_sha256"],
                runtime["expectation_sha256"],
            )

    def test_16_execution_contract_rejects_escape_ambient_env_and_cwd_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = execution_runtime_fixture(root)
            contract = runtime["contract"]
            escape = json.loads(json.dumps(contract))
            escape["workers"][0]["argv"][2] = "--config=../outside.json"
            escape["workers"][0]["argv_schema"][2]["spelling"] = "../outside.json"
            self.bad(self.contract_call, runtime, escape)
            ambient = json.loads(json.dumps(contract))
            ambient["workers"][0]["env"]["PYTHONPATH"] = "/tmp"
            self.bad(self.contract_call, runtime, ambient)
            env_escape = json.loads(json.dumps(contract))
            env_escape["workers"][0]["env"]["FORKAUDIT_CONFIG_PATH"] = "../outside.json"
            env_escape["workers"][0]["env_schema"][1]["spelling"] = "../outside.json"
            self.bad(self.contract_call, runtime, env_escape)
            (runtime["runner"] / "linked-cwd").symlink_to(runtime["runner"] / "cfg", target_is_directory=True)
            linked = json.loads(json.dumps(contract))
            linked["workers"][0]["cwd"] = "linked-cwd"
            self.bad(self.contract_call, runtime, linked)

    def test_17_lifecycle_gate_requires_kill_completion_and_all_worker_statuses(self):
        lifecycle_gate(successful_receipt(), require_success=True)
        missing = successful_receipt()
        del missing["kill_completion"]
        self.bad(lifecycle_gate, missing)
        kill_error = successful_receipt()
        kill_error["kill_completion"] = {
            "completed": False,
            "errors": ["kill failed"],
            "required": False,
        }
        kill_error["verification_complete"] = False
        lifecycle_gate(kill_error)
        self.bad(lifecycle_gate, kill_error, True)
        inconsistent = successful_receipt()
        inconsistent["kill_completion"]["required"] = True
        self.bad(lifecycle_gate, inconsistent)
        worker_bad = successful_receipt()
        worker_bad["workers"][3]["death_confirmed"] = False
        self.bad(lifecycle_gate, worker_bad)
        worker_missing = successful_receipt()
        worker_missing["workers"] = worker_missing["workers"][:-1]
        self.bad(lifecycle_gate, worker_missing)

    def test_18_success_requires_start_and_exact_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = signed_fixture(base)
            plan = plan_from_fixture(fixture)
            runtime = fixture["runtime"]
            terminal_root = base / "terminals"
            terminal_root.mkdir()
            with ProtectedParent(terminal_root) as parent:
                lifecycle = Lifecycle(
                    parent,
                    plan,
                    fixture["archive"],
                    fixture["ledger"],
                    fixture["snapshot"],
                )
                self.assertEqual(lifecycle.finalize("success"), 1)
            self.assertEqual(len(list(terminal_root.iterdir())), 8)
            hashes = measure_hashes(
                fixture["archive"],
                fixture["ledger"],
                fixture["snapshot"],
                runtime["runner"],
                runtime["manifest_path"],
            )
            validate_terminal_tree(terminal_root, hashes, hashes, "failure")

    def test_19_started_success_emits_exact_eight_success_terminals(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = signed_fixture(base)
            plan = plan_from_fixture(fixture)
            terminal_root = base / "terminals"
            terminal_root.mkdir()
            before = plan.artifact_hashes
            with ProtectedParent(terminal_root) as parent:
                lifecycle = Lifecycle(
                    parent,
                    plan,
                    fixture["archive"],
                    fixture["ledger"],
                    fixture["snapshot"],
                )
                lifecycle.install_signal_handlers()
                try:
                    lifecycle.start()
                    self.assertFalse(hasattr(lifecycle, "set_gate"))
                    lifecycle.spawn_workers()
                    self.assertEqual(lifecycle.wait_workers(), [0] * 8)
                    self.assertEqual(lifecycle.finalize("success"), 0)
                finally:
                    lifecycle.restore_signal_handlers()
            validate_terminal_tree(terminal_root, before, before, "success")

    def test_20_premature_success_kills_owned_workers_and_cannot_pass(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = signed_fixture(base, sleeping=True)
            plan = plan_from_fixture(fixture)
            terminal_root = base / "terminals"
            terminal_root.mkdir()
            with ProtectedParent(terminal_root) as parent:
                lifecycle = Lifecycle(
                    parent,
                    plan,
                    fixture["archive"],
                    fixture["ledger"],
                    fixture["snapshot"],
                )
                lifecycle.install_signal_handlers()
                try:
                    lifecycle.start()
                    lifecycle.spawn_workers()
                    self.assertEqual(lifecycle.finalize("success"), 1)
                    self.assertTrue(all(process.poll() is not None for process in lifecycle._processes.values()))
                finally:
                    lifecycle.restore_signal_handlers()
            hashes = plan.artifact_hashes
            validate_terminal_tree(terminal_root, hashes, hashes, "failure")

    def test_21_terminal_tree_rejects_extra_symlink_hardlink_and_schema_mutants(self):
        hashes = {key: "a" * 64 for key in HASH_KEYS}
        for mutant in ("extra", "symlink", "hardlink", "schema", "receipt"):
            with self.subTest(mutant=mutant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                receipt = successful_receipt()
                receipt_sha = digest_bytes(canonical_bytes(receipt))
                for fault_id in TERM_IDS:
                    (root / f"{fault_id}.terminal.json").write_bytes(
                        canonical_bytes(
                            {
                                "fault_id": fault_id,
                                "lifecycle_receipt": receipt,
                                "lifecycle_receipt_sha256": receipt_sha,
                                "post_hashes": hashes,
                                "pre_hashes": hashes,
                                "provenance": sample_provenance(),
                                "provenance_sha256": digest_bytes(canonical_bytes(sample_provenance())),
                                "reason": "success",
                                "schema_version": "forkaudit-v9-terminal-v1",
                                "signal": None,
                                "status": "success",
                            }
                        )
                    )
                if mutant == "extra":
                    (root / "extra").write_bytes(b"x")
                elif mutant == "symlink":
                    (root / "V9F01.terminal.json").unlink()
                    (root / "V9F01.terminal.json").symlink_to(root / "V9F02.terminal.json")
                elif mutant == "hardlink":
                    (root / "V9F01.terminal.json").unlink()
                    os.link(root / "V9F02.terminal.json", root / "V9F01.terminal.json")
                elif mutant == "schema":
                    value = json.loads((root / "V9F01.terminal.json").read_text())
                    value["pre_hashes"]["extra"] = "b" * 64
                    (root / "V9F01.terminal.json").write_bytes(canonical_bytes(value))
                else:
                    value = json.loads((root / "V9F01.terminal.json").read_text())
                    value["lifecycle_receipt"]["workers"][0]["pid"] += 1
                    value["lifecycle_receipt_sha256"] = digest_bytes(
                        canonical_bytes(value["lifecycle_receipt"])
                    )
                    (root / "V9F01.terminal.json").write_bytes(canonical_bytes(value))
                self.bad(validate_terminal_tree, root, hashes, hashes, "success")

    def test_22_real_sigint_and_sigterm_emit_eight_and_exit_130_143(self):
        for signum, expected_exit in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with self.subTest(signum=signum), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                fixture = signed_fixture(base, sleeping=True)
                plan = plan_from_fixture(fixture)
                terminal_root = base / "terminals"
                terminal_root.mkdir()
                ready = base / "ready.json"
                expected_hashes = plan.artifact_hashes
                pid = os.fork()
                if pid == 0:
                    try:
                        with ProtectedParent(terminal_root) as parent:
                            lifecycle = Lifecycle(
                                parent,
                                plan,
                                fixture["archive"],
                                fixture["ledger"],
                                fixture["snapshot"],
                            )
                            lifecycle.install_signal_handlers()
                            lifecycle.start()
                            lifecycle.spawn_workers()
                            ready.write_bytes(canonical_bytes({"ready": True}))
                            while True:
                                signal.pause()
                    except SystemExit as exc:
                        os._exit(int(exc.code))
                    except BaseException:
                        os._exit(1)
                deadline = time.monotonic() + 10
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if not ready.exists():
                    os.kill(pid, signal.SIGKILL)
                    os.waitpid(pid, 0)
                    self.fail("signal child not ready")
                os.kill(pid, signum)
                _, wait_status = os.waitpid(pid, 0)
                self.assertEqual(os.waitstatus_to_exitcode(wait_status), expected_exit)
                validate_terminal_tree(terminal_root, expected_hashes, expected_hashes, "failure")

    def test_23_signal_exit_is_exact_and_rejects_other_values(self):
        self.assertEqual(signal_exit(2), 130)
        self.assertEqual(signal_exit(15), 143)
        for value in (0, 9, True, "15", 2.0, 15.0):
            self.bad(signal_exit, value)
        hashes = {key: "a" * 64 for key in HASH_KEYS}
        receipt = successful_receipt()
        terminal = {
            "fault_id": TERM_IDS[0],
            "lifecycle_receipt": receipt,
            "lifecycle_receipt_sha256": digest_bytes(canonical_bytes(receipt)),
            "post_hashes": hashes,
            "pre_hashes": hashes,
            "provenance": sample_provenance(),
            "provenance_sha256": digest_bytes(canonical_bytes(sample_provenance())),
            "reason": "signal",
            "schema_version": "forkaudit-v9-terminal-v1",
            "signal": 2.0,
            "status": "failure",
        }
        self.bad(v9_guard.terminal_record, terminal, TERM_IDS[0])
        terminal["signal"] = 15.0
        self.bad(v9_guard.terminal_record, terminal, TERM_IDS[0])

    def test_24_protected_parent_no_replace_preserves_bytes_and_inode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "x"
            existing.write_bytes(b"original")
            before = (existing.stat().st_dev, existing.stat().st_ino, existing.read_bytes())
            with ProtectedParent(root) as parent:
                self.bad(parent.publish_many, {"x": b"replacement", "y": b"new"})
            after = (existing.stat().st_dev, existing.stat().st_ino, existing.read_bytes())
            self.assertEqual(after, before)
            self.assertFalse((root / "y").exists())

    def test_25_retained_parent_final_failure_rolls_back_linked_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "parent"
            moved = base / "retained"
            root.mkdir()

            class LateReplacement(ProtectedParent):
                checks = 0

                def check(self):
                    self.checks += 1
                    if self.checks == 2:
                        self.path.rename(moved)
                        self.path.mkdir()
                    return super().check()

            with LateReplacement(root) as parent:
                self.bad(parent.publish_many, {"linked.json": b"{}\n"})
            self.assertFalse((moved / "linked.json").exists())
            self.assertFalse((root / "linked.json").exists())
            self.assertEqual(list(moved.iterdir()), [])

    def test_26_batch_publish_rolls_back_on_mid_link_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_link = os.link
            calls = 0

            def failing_link(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected link failure")
                return real_link(*args, **kwargs)

            with ProtectedParent(root) as parent, mock.patch.object(
                v9_runtime.os, "link", side_effect=failing_link
            ):
                with self.assertRaises(OSError):
                    parent.publish_many({"a": b"a", "b": b"b", "c": b"c"})
            self.assertEqual(list(root.iterdir()), [])

    def test_27_freeze_is_deterministic_transactional_and_rerun_immutable(self):
        first, first_method = freeze_v9.build_outputs(PACKAGE)
        second, second_method = freeze_v9.build_outputs(PACKAGE)
        self.assertEqual(first, second)
        self.assertEqual(first_method, second_method)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            freeze_v9.freeze_package(PACKAGE, target)
            before = {
                path.name: (
                    path.stat().st_dev,
                    path.stat().st_ino,
                    path.stat().st_mode,
                    digest_file(path),
                )
                for path in target.iterdir()
            }
            self.bad(freeze_v9.freeze_package, PACKAGE, target)
            after = {
                path.name: (
                    path.stat().st_dev,
                    path.stat().st_ino,
                    path.stat().st_mode,
                    digest_file(path),
                )
                for path in target.iterdir()
            }
            self.assertEqual(after, before)
            self.assertEqual(set(after), {"METHOD_FROZEN.json", freeze_v9.ARCHIVE_NAME, "source-ledger.json"})

    def test_28_runtime_expectation_is_exact_anchored_and_derives_probe_parameters(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = execution_runtime_fixture(Path(temporary))
            self.assertEqual(
                validate_runtime_expectation(
                    runtime["expectation_raw"],
                    runtime["expectation_sha256"],
                    runtime["runner"],
                    runtime["manifest_path"],
                    runtime["manifest_sha256"],
                    runtime["inventory_sha256"],
                ),
                runtime["expectation"],
            )
            self.assertEqual(
                tuple(inspect.signature(isolated_torch_probe).parameters),
                (
                    "raw_expectation",
                    "expected_expectation_sha256",
                    "runner_root",
                    "runner_manifest_path",
                    "expected_runner_manifest_sha256",
                    "expected_runner_inventory_sha256",
                    "timeout_seconds",
                ),
            )
            extra = json.loads(json.dumps(runtime["expectation"]))
            extra["caller_python"] = "/bin/sh"
            raw = canonical_bytes(extra)
            self.bad(
                validate_runtime_expectation,
                raw,
                digest_bytes(raw),
                runtime["runner"],
                runtime["manifest_path"],
                runtime["manifest_sha256"],
                runtime["inventory_sha256"],
            )
            closure = json.loads(json.dumps(runtime["expectation"]))
            closure["runtime_closure"] = closure["runtime_closure"][:-1]
            raw = canonical_bytes(closure)
            self.bad(
                validate_runtime_expectation,
                raw,
                digest_bytes(raw),
                runtime["runner"],
                runtime["manifest_path"],
                runtime["manifest_sha256"],
                runtime["inventory_sha256"],
            )

    def test_29_isolated_clean_process_imports_only_inventory_bound_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = execution_runtime_fixture(Path(temporary), real_python=True)
            report = isolated_torch_probe(
                runtime["expectation_raw"],
                runtime["expectation_sha256"],
                runtime["runner"],
                runtime["manifest_path"],
                runtime["manifest_sha256"],
                runtime["inventory_sha256"],
            )
            self.assertTrue(report["isolated"])
            self.assertTrue(report["preimport_absent"])
            self.assertEqual(report["runner_manifest_sha256"], runtime["manifest_sha256"])
            self.assertEqual(report["runner_inventory_sha256"], runtime["inventory_sha256"])

    def test_30_transplanted_probe_and_fake_bin_sh_python_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = execution_runtime_fixture(Path(temporary))
            shutil.copyfile("/bin/sh", runtime["runner"] / "bin" / "python")
            (runtime["runner"] / "bin" / "python").chmod(0o755)
            refresh_runtime_expectation(runtime)
            self.bad(
                isolated_torch_probe,
                runtime["expectation_raw"],
                runtime["expectation_sha256"],
                runtime["runner"],
                runtime["manifest_path"],
                runtime["manifest_sha256"],
                runtime["inventory_sha256"],
            )
        with tempfile.TemporaryDirectory() as temporary:
            runtime = execution_runtime_fixture(Path(temporary))
            (runtime["runner"] / "tools" / "torch_probe_v9.py").write_text("print('{}')\n")
            refresh_runtime_expectation(runtime)
            self.bad(
                validate_runtime_expectation,
                runtime["expectation_raw"],
                runtime["expectation_sha256"],
                runtime["runner"],
                runtime["manifest_path"],
                runtime["manifest_sha256"],
                runtime["inventory_sha256"],
            )

    def test_31_counterexample_caller_worker_specs_has_no_lifecycle_entry(self):
        self.assertEqual(tuple(inspect.signature(Lifecycle.spawn_workers).parameters), ("self",))
        self.assertEqual(tuple(inspect.signature(Lifecycle.wait_workers).parameters), ("self",))
        source = (PACKAGE / "v9_runtime.py").read_text()
        self.assertNotIn("worker_specs", source)
        with self.assertRaises(Reject):
            AuthorizedPlan(
                object(),
                artifact_hashes={key: "a" * 64 for key in HASH_KEYS},
                binding_path=Path("binding"),
                contract_path=Path("contract"),
                expectation_path=Path("expectation"),
                provenance=sample_provenance(),
                runner_manifest_path=Path("manifest"),
                runner_root=Path("runner"),
                spawned_specs={"schema_version": "forkaudit-v9-spawned-specs-v1", "workers": []},
                timeout_seconds=60,
            )
        formal_markers = []
        for source_path in PACKAGE.glob("*.py"):
            tree = ast.parse(source_path.read_text(), filename=str(source_path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id == "FORMAL_AUTHORIZED_LAUNCHER" for target in node.targets)
                    and isinstance(node.value, ast.Constant)
                    and node.value.value is True
                ):
                    formal_markers.append(source_path.name)
        self.assertEqual(formal_markers, ["authorized_launcher_v9.py"])

    def test_32_counterexample_signed_semantic_drift_rejected_before_any_spawn(self):
        for mutant in ("python", "visibility"):
            with self.subTest(mutant=mutant), tempfile.TemporaryDirectory() as temporary:
                fixture = signed_fixture(Path(temporary), real_python=False)
                runtime = fixture["runtime"]
                contract = runtime["contract"]
                if mutant == "python":
                    replacement = str((runtime["runner"] / "run.py").resolve(strict=True))
                    contract["workers"][0]["argv"][0] = replacement
                    contract["workers"][0]["argv_schema"][0] = {
                        "index": 0,
                        "kind": "path",
                        "manifest_path": "run.py",
                        "spelling": replacement,
                    }
                else:
                    other = "GPU-fedcba98-7654-3210-fedc-ba9876543210"
                    contract["workers"][3]["env"]["CUDA_VISIBLE_DEVICES"] = other
                    contract["workers"][3]["env_schema"][0]["visibility"] = other
                contract_raw = canonical_bytes(contract)
                runtime["contract_path"].write_bytes(contract_raw)
                runtime["contract_raw"] = contract_raw
                resign_fixture(fixture)
                with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("spawned")) as popen:
                    self.bad(plan_from_fixture, fixture)
                    popen.assert_not_called()

    def test_33_missing_binding_unsigned_manifest_and_bin_sh_fail_pre_spawn(self):
        for mutant in ("missing-binding", "unsigned-manifest", "bin-sh"):
            with self.subTest(mutant=mutant), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                fixture = signed_fixture(base, real_python=False)
                runtime = fixture["runtime"]
                terminal_root = base / "terminals"
                terminal_root.mkdir()
                if mutant == "missing-binding":
                    fixture["binding_path"] = base / "absent-binding.json"
                elif mutant == "unsigned-manifest":
                    runtime["manifest_path"].write_bytes(canonical_bytes([]))
                else:
                    shutil.copyfile("/bin/sh", runtime["runner"] / "bin" / "python")
                    (runtime["runner"] / "bin" / "python").chmod(0o755)
                    refresh_runtime_expectation(runtime)
                    resign_fixture(fixture)
                kwargs = {
                    "binding_path": str(fixture["binding_path"]),
                    "archive_path": str(fixture["archive"]),
                    "source_ledger_path": str(fixture["ledger"]),
                    "snapshot_root": str(fixture["snapshot"]),
                    "execution_contract_path": str(runtime["contract_path"]),
                    "runtime_expectation_path": str(runtime["expectation_path"]),
                    "runner_root": str(runtime["runner"]),
                    "runner_manifest_path": str(runtime["manifest_path"]),
                    "terminal_root": str(terminal_root),
                }
                with mock.patch.object(v9_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
                    v9_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", fixture["public"].hex()
                ), mock.patch.object(subprocess, "Popen", side_effect=AssertionError("spawned")) as popen:
                    self.bad(run_authorized_campaign, **kwargs)
                    popen.assert_not_called()
                self.assertEqual(list(terminal_root.iterdir()), [])

    def test_34_unique_authorized_launcher_runs_eight_real_workers_and_commits_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = signed_fixture(base)
            runtime = fixture["runtime"]
            terminal_root = base / "terminals"
            terminal_root.mkdir()
            with mock.patch.object(v9_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
                v9_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", fixture["public"].hex()
            ):
                result = run_authorized_campaign(
                    binding_path=str(fixture["binding_path"]),
                    archive_path=str(fixture["archive"]),
                    source_ledger_path=str(fixture["ledger"]),
                    snapshot_root=str(fixture["snapshot"]),
                    execution_contract_path=str(runtime["contract_path"]),
                    runtime_expectation_path=str(runtime["expectation_path"]),
                    runner_root=str(runtime["runner"]),
                    runner_manifest_path=str(runtime["manifest_path"]),
                    terminal_root=str(terminal_root),
                )
            self.assertEqual(result, 0)
            self.assertEqual(len(list(terminal_root.iterdir())), 8)
            provenance_digests = set()
            for fault_id in TERM_IDS:
                value = canonical_load(
                    (terminal_root / f"{fault_id}.terminal.json").read_bytes(), "terminal"
                )
                self.assertEqual(value["status"], "success")
                self.assertEqual(value["provenance"], value["lifecycle_receipt"]["provenance"])
                self.assertEqual(
                    value["provenance_sha256"],
                    digest_bytes(canonical_bytes(value["provenance"])),
                )
                provenance_digests.add(value["provenance_sha256"])
            self.assertEqual(len(provenance_digests), 1)

    def test_35_authorized_plan_is_immutable_and_specs_match_all_signed_tokens(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = signed_fixture(Path(temporary))
            plan = plan_from_fixture(fixture)
            runtime = fixture["runtime"]
            expected = derive_spawned_specs(runtime["contract"], runtime["runner"])
            self.assertEqual(plan.spawned_specs, expected)
            self.assertEqual(
                plan.provenance["spawned_specs_sha256"],
                digest_bytes(canonical_bytes(expected)),
            )
            with self.assertRaises(Reject):
                plan.timeout_seconds = 1
            leaked = plan.spawned_specs
            leaked["workers"][0]["argv"][0] = "/bin/sh"
            self.assertNotEqual(leaked, plan.spawned_specs)

    def test_36_static_audit_passes(self):
        result = static_audit_v9.audit(PACKAGE)
        self.assertEqual(result["status"], "PASS", result)
        self.assertTrue(all(result["gates"].values()), result)


if __name__ == "__main__":
    unittest.main(verbosity=2)
