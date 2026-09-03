from __future__ import annotations

import hashlib
import io
import os
import shutil
import sys
import tarfile
from pathlib import Path

import v10_guard
from v10_guard import (
    BINDING_DOMAIN,
    TERM_IDS,
    _BASE,
    _L,
    _scalar_mult,
    canonical_bytes,
    canonical_tar,
    deterministic_gzip,
    digest_bytes,
    digest_file,
    runner_commitments,
    snapshot_commitments,
)


PACKAGE = Path(__file__).resolve().parent
UUID0 = "GPU-01234567-89ab-cdef-0123-456789abcdef"


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
            "freeze_timestamp": v10_guard.FREEZE_TIMESTAMP,
            "schema_version": "forkaudit-v10-source-ledger-v1",
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


def write_runner_manifest(base: Path, runner: Path) -> tuple[Path, str, str]:
    manifest_path = base / "runner-manifest.json"
    manifest_path.write_bytes(canonical_bytes(tree_manifest(runner)))
    manifest_sha, inventory_sha, _ = runner_commitments(runner, manifest_path)
    return manifest_path, manifest_sha, inventory_sha


def execution_runtime_fixture(base: Path, *, real_python: bool = False) -> dict:
    runner = base / "runner"
    (runner / "bin").mkdir(parents=True)
    (runner / "cfg").mkdir()
    (runner / "tools").mkdir()
    (runner / "torch").mkdir()
    python_path = runner / "bin" / "python"
    loaded_manifest_path = "bin/python"
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
            loaded_manifest_path = "Resources/Python.app/Contents/MacOS/Python"
    else:
        python_path.write_bytes(b"not-executed-python-fixture\n")
        python_path.chmod(0o755)
    launch_manifest_path = loaded_manifest_path
    shutil.copyfile(PACKAGE / "torch_probe_v10.py", runner / "tools" / "torch_probe_v10.py")
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
        launch_manifest_path: "python-executable",
        "tools/torch_probe_v10.py": "provenance-probe",
        "torch/__init__.py": "torch-package-root",
    }
    expectation = {
        "cwd": ".",
        "device": {"index": 0, "physical_uuid": UUID0, "visibility": UUID0},
        "probe": {
            "manifest_path": "tools/torch_probe_v10.py",
            "sha256": digest_file(runner / "tools" / "torch_probe_v10.py"),
        },
        "python": {
            "cache_tag": sys.implementation.cache_tag,
            "implementation": "cpython",
            "loaded_manifest_path": loaded_manifest_path,
            "loaded_sha256": digest_file(runner / loaded_manifest_path),
            "manifest_path": launch_manifest_path,
            "sha256": digest_file(runner / launch_manifest_path),
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
        "schema_version": "forkaudit-v10-runtime-expectation-v1",
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
    base_argv = [
        str((runner / launch_manifest_path).resolve()),
        "run.py", "--config=cfg/formal.json", "--dry-run", "unit",
    ]
    workers = []
    for fault_id in TERM_IDS:
        argv = base_argv + ["--fault-id", fault_id]
        workers.append(
            {
                "argv": argv,
                "argv_schema": [
                    {
                        "index": 0, "kind": "path",
                        "manifest_path": launch_manifest_path,
                        "spelling": argv[0],
                    },
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
        "schema_version": "forkaudit-v10-execution-contract-v1",
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
    expectation["python"]["loaded_sha256"] = digest_file(
        runner / expectation["python"]["loaded_manifest_path"]
    )
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
                   run_nonce: str = "a" * 64,
                   consumption_root: Path | None = None) -> dict:
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
    if consumption_root is None:
        consumption_root = base / "consumptions"
        consumption_root.mkdir()
    (snapshot / "public.txt").write_text("sealed\n")
    runtime = execution_runtime_fixture(base, real_python=real_python)
    if sleeping:
        (runtime["runner"] / "run.py").write_text("import time\ntime.sleep(3600)\n")
        refresh_runtime_expectation(runtime)
    snapshot_sha, snapshot_inventory_sha, _ = snapshot_commitments(snapshot)
    terminal_stat = os.lstat(terminal_root)
    consumption_stat = os.lstat(consumption_root)
    payload = {
        "approved_archive_sha256": digest_file(archive),
        "approved_attempt": attempt,
        "approved_consumption_root": str(consumption_root.resolve(strict=True)),
        "approved_consumption_root_dev": consumption_stat.st_dev,
        "approved_consumption_root_ino": consumption_stat.st_ino,
        "approved_execution_contract_sha256": digest_file(runtime["contract_path"]),
        "approved_runner_inventory_sha256": runtime["inventory_sha256"],
        "approved_runner_manifest_sha256": runtime["manifest_sha256"],
        "approved_runtime_expectation_sha256": runtime["expectation_sha256"],
        "approved_run_nonce": run_nonce,
        "approved_snapshot_inventory_sha256": snapshot_inventory_sha,
        "approved_snapshot_sha256": snapshot_sha,
        "approved_source_ledger_sha256": digest_file(ledger),
        "approved_terminal_root": str(terminal_root.resolve(strict=True)),
        "approved_terminal_root_dev": terminal_stat.st_dev,
        "approved_terminal_root_ino": terminal_stat.st_ino,
        "operator_id": "independent-test-operator",
        "published_uri": "https://operator.example/r40/v10-binding.json",
        "schema_version": "forkaudit-v10-operator-binding-payload-v1",
        "trust_root_id": v10_guard.TRUST_ROOT_ID,
    }
    seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
    public, signature = _rfc_test_sign(seed, BINDING_DOMAIN + canonical_bytes(payload))
    binding = {
        "payload": payload,
        "schema_version": "forkaudit-v10-signed-operator-binding-v1",
        "signature": {
            "algorithm": "ed25519",
            "key_id": v10_guard.TRUST_ROOT_ID,
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
        "consumption_root": consumption_root,
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
