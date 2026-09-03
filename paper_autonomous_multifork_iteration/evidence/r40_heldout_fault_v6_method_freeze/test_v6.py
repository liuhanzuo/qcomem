from __future__ import annotations

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

import freeze_v6
import v6_guard
import v6_runtime
from v6_guard import (
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
    designer_attestation,
    digest_bytes,
    digest_file,
    ed25519_verify,
    isolated_torch_probe,
    lifecycle_gate,
    measure_hashes,
    snapshot_commitments,
    strict_gzip,
    validate_execution_contract,
    validate_runner_tree,
    validate_terminal_tree,
    validate_torch_report,
    verify_archive,
    verify_operator_binding,
)
from v6_runtime import Lifecycle, ProtectedParent, signal_exit


PACKAGE = Path(__file__).resolve().parent
UUID0 = "GPU-01234567-89ab-cdef-0123-456789abcdef"
UUID1 = "GPU-fedcba98-7654-3210-fedc-ba9876543210"


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
            "freeze_timestamp": v6_guard.FREEZE_TIMESTAMP,
            "schema_version": "forkaudit-v6-source-ledger-v1",
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


def good_gate() -> dict:
    return {
        "inventory_verified": True,
        "kill_completion": {"completed": True, "errors": [], "required": False},
        "post_rehash_verified": True,
        "receipts_verified": True,
        "schema_version": "forkaudit-v6-lifecycle-gate-v1",
        "verification_complete": True,
        "workers": [
            {
                "death_confirmed": True,
                "exit_code": 0,
                "fault_id": fault_id,
                "kill_completed": True,
                "kill_required": False,
                "spawned": True,
            }
            for fault_id in TERM_IDS
        ],
    }


def lifecycle_fixture(base: Path) -> tuple[dict, list[dict]]:
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
    manifest = tree_manifest(runner)
    paths = {
        "archive": archive,
        "ledger": ledger,
        "snapshot": snapshot,
        "runner": runner,
    }
    return paths, manifest


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


class V6Tests(unittest.TestCase):
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
                "torch_expectation_path",
            ),
        )
        v6_guard._compiled_trust_root()
        with self.assertRaises(TypeError):
            verify_operator_binding(b"{}\n", "a", "b", "c", "d", "e", lambda _: True)

    def test_04_arbitrary_self_claim_and_noncanonical_binding_rejected(self):
        template = json.loads((PACKAGE / "operator-binding.template.json").read_text())
        template["payload"].update(
            {
                "approved_archive_sha256": "a" * 64,
                "approved_execution_contract_sha256": "b" * 64,
                "approved_snapshot_inventory_sha256": "c" * 64,
                "approved_snapshot_sha256": "d" * 64,
                "approved_source_ledger_sha256": "e" * 64,
                "approved_torch_expectation_sha256": "f" * 64,
                "operator_id": "self",
                "published_uri": "https://operator.example/r40/v6.json",
            }
        )
        template["signature"]["signature_hex"] = "0" * 128
        self.bad(verify_operator_binding, canonical_bytes(template), "a", "b", "c", "d", "e")
        raw = json.dumps(template, indent=2).encode()
        self.bad(verify_operator_binding, raw, "a", "b", "c", "d", "e")

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
            contract = root / "execution.json"
            expectation = root / "torch.json"
            contract.write_bytes(canonical_bytes({"sealed": True}))
            expectation.write_bytes(canonical_bytes({"sealed": True}))
            snapshot_sha, inventory_sha, _ = snapshot_commitments(snapshot)
            payload = {
                "approved_archive_sha256": digest_file(archive),
                "approved_execution_contract_sha256": digest_file(contract),
                "approved_snapshot_inventory_sha256": inventory_sha,
                "approved_snapshot_sha256": snapshot_sha,
                "approved_source_ledger_sha256": digest_file(ledger),
                "approved_torch_expectation_sha256": digest_file(expectation),
                "operator_id": "independent-test-operator",
                "published_uri": "https://operator.example/r40/v6-binding.json",
                "schema_version": "forkaudit-v6-operator-binding-payload-v1",
                "trust_root_id": v6_guard.TRUST_ROOT_ID,
            }
            seed = bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60")
            public, signature = _rfc_test_sign(seed, BINDING_DOMAIN + canonical_bytes(payload))
            binding = {
                "payload": payload,
                "schema_version": "forkaudit-v6-signed-operator-binding-v1",
                "signature": {
                    "algorithm": "ed25519",
                    "key_id": v6_guard.TRUST_ROOT_ID,
                    "signature_hex": signature.hex(),
                },
            }
            with mock.patch.object(v6_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
                v6_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", public.hex()
            ):
                self.assertEqual(
                    verify_operator_binding(
                        canonical_bytes(binding), archive, ledger, snapshot, contract, expectation
                    ),
                    payload,
                )
                (snapshot / "extra.txt").write_text("not signed\n")
                self.bad(
                    verify_operator_binding,
                    canonical_bytes(binding),
                    archive,
                    ledger,
                    snapshot,
                    contract,
                    expectation,
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

    def _execution_fixture(self, root: Path) -> tuple[dict, list[dict]]:
        (root / "bin").mkdir()
        (root / "cfg").mkdir()
        (root / "bin" / "python").write_bytes(b"python")
        (root / "run.py").write_bytes(b"run")
        (root / "cfg" / "formal.json").write_bytes(b"{}\n")
        manifest = tree_manifest(root)
        contract = {
            "argv": [
                str((root / "bin" / "python").resolve()),
                "run.py",
                "--config=cfg/formal.json",
            ],
            "cwd": ".",
            "env": {
                "CUDA_VISIBLE_DEVICES": UUID0,
                "PYTHONHASHSEED": "0",
                "PYTHONNOUSERSITE": "1",
            },
            "path_bindings": [
                {"argv_index": 0, "manifest_path": "bin/python", "option": None},
                {"argv_index": 1, "manifest_path": "run.py", "option": None},
                {"argv_index": 2, "manifest_path": "cfg/formal.json", "option": "--config"},
            ],
            "schema_version": "forkaudit-v6-execution-contract-v1",
        }
        return contract, manifest

    def test_13_execution_contract_binds_relative_and_equals_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, manifest = self._execution_fixture(root)
            raw = canonical_bytes(contract)
            self.assertEqual(
                validate_execution_contract(raw, digest_bytes(raw), root, manifest), contract
            )

    def test_14_execution_contract_binds_separate_path_option(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, manifest = self._execution_fixture(root)
            contract["argv"] = contract["argv"][:2] + ["--config", "cfg/formal.json"]
            contract["path_bindings"][-1] = {
                "argv_index": 3,
                "manifest_path": "cfg/formal.json",
                "option": "--config",
            }
            raw = canonical_bytes(contract)
            validate_execution_contract(raw, digest_bytes(raw), root, manifest)

    def test_15_execution_contract_rejects_every_unbound_path_form(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base, manifest = self._execution_fixture(root)
            mutants = []
            equals = json.loads(json.dumps(base))
            equals["path_bindings"] = equals["path_bindings"][:-1]
            mutants.append(equals)
            separate = json.loads(json.dumps(base))
            separate["argv"] = separate["argv"][:2] + ["--config", "cfg/formal.json"]
            separate["path_bindings"] = separate["path_bindings"][:-1]
            mutants.append(separate)
            bare = json.loads(json.dumps(base))
            bare["argv"].append("weights.safetensors")
            mutants.append(bare)
            absolute = json.loads(json.dumps(base))
            absolute["argv"].append("/etc/passwd")
            mutants.append(absolute)
            for contract in mutants:
                with self.subTest(argv=contract["argv"]):
                    raw = canonical_bytes(contract)
                    self.bad(validate_execution_contract, raw, digest_bytes(raw), root, manifest)

    def test_16_execution_contract_rejects_escape_ambient_env_and_cwd_symlink(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract, manifest = self._execution_fixture(root)
            escape = json.loads(json.dumps(contract))
            escape["argv"][2] = "--config=../outside.json"
            raw = canonical_bytes(escape)
            self.bad(validate_execution_contract, raw, digest_bytes(raw), root, manifest)
            ambient = json.loads(json.dumps(contract))
            ambient["env"]["PYTHONPATH"] = "/tmp"
            raw = canonical_bytes(ambient)
            self.bad(validate_execution_contract, raw, digest_bytes(raw), root, manifest)
            (root / "linked-cwd").symlink_to(root / "cfg", target_is_directory=True)
            contract["cwd"] = "linked-cwd"
            raw = canonical_bytes(contract)
            self.bad(validate_execution_contract, raw, digest_bytes(raw), root, tree_manifest(root))

    def test_17_lifecycle_gate_requires_kill_completion_and_all_worker_statuses(self):
        lifecycle_gate(good_gate())
        missing = good_gate()
        del missing["kill_completion"]
        self.bad(lifecycle_gate, missing)
        kill_error = good_gate()
        kill_error["kill_completion"] = {
            "completed": True,
            "errors": ["kill failed"],
            "required": True,
        }
        self.bad(lifecycle_gate, kill_error)
        worker_bad = good_gate()
        worker_bad["workers"][3]["death_confirmed"] = False
        self.bad(lifecycle_gate, worker_bad)
        worker_missing = good_gate()
        worker_missing["workers"] = worker_missing["workers"][:-1]
        self.bad(lifecycle_gate, worker_missing)

    def test_18_success_requires_start_and_exact_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths, manifest = lifecycle_fixture(base)
            terminal_root = base / "terminals"
            terminal_root.mkdir()
            with ProtectedParent(terminal_root) as parent:
                lifecycle = Lifecycle(
                    parent,
                    paths["archive"],
                    paths["ledger"],
                    paths["snapshot"],
                    paths["runner"],
                    manifest,
                )
                self.assertEqual(lifecycle.finalize("success"), 1)
            self.assertEqual(len(list(terminal_root.iterdir())), 8)
            hashes = measure_hashes(
                paths["archive"], paths["ledger"], paths["snapshot"], paths["runner"], manifest
            )
            validate_terminal_tree(terminal_root, hashes, hashes, "failure")

    def test_19_started_success_emits_exact_eight_success_terminals(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            paths, manifest = lifecycle_fixture(base)
            terminal_root = base / "terminals"
            terminal_root.mkdir()
            before = measure_hashes(
                paths["archive"], paths["ledger"], paths["snapshot"], paths["runner"], manifest
            )
            with ProtectedParent(terminal_root) as parent:
                lifecycle = Lifecycle(
                    parent,
                    paths["archive"],
                    paths["ledger"],
                    paths["snapshot"],
                    paths["runner"],
                    manifest,
                )
                lifecycle.install_signal_handlers()
                try:
                    lifecycle.start()
                    lifecycle.set_gate(good_gate())
                    self.assertEqual(lifecycle.finalize("success"), 0)
                finally:
                    lifecycle.restore_signal_handlers()
            validate_terminal_tree(terminal_root, before, before, "success")

    def test_20_terminal_tree_rejects_extra_symlink_hardlink_and_schema_mutants(self):
        hashes = {key: "a" * 64 for key in HASH_KEYS}
        for mutant in ("extra", "symlink", "hardlink", "schema"):
            with self.subTest(mutant=mutant), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                for fault_id in TERM_IDS:
                    (root / f"{fault_id}.terminal.json").write_bytes(
                        canonical_bytes(
                            {
                                "fault_id": fault_id,
                                "post_hashes": hashes,
                                "pre_hashes": hashes,
                                "reason": "success",
                                "schema_version": "forkaudit-v6-terminal-v1",
                                "signal": None,
                                "status": "success",
                            }
                        )
                    )
                if mutant == "extra":
                    (root / "extra").write_bytes(b"x")
                elif mutant == "symlink":
                    (root / "V6F01.terminal.json").unlink()
                    (root / "V6F01.terminal.json").symlink_to(root / "V6F02.terminal.json")
                elif mutant == "hardlink":
                    (root / "V6F01.terminal.json").unlink()
                    os.link(root / "V6F02.terminal.json", root / "V6F01.terminal.json")
                else:
                    value = json.loads((root / "V6F01.terminal.json").read_text())
                    value["pre_hashes"]["extra"] = "b" * 64
                    (root / "V6F01.terminal.json").write_bytes(canonical_bytes(value))
                self.bad(validate_terminal_tree, root, hashes, hashes, "success")

    def test_21_real_sigint_and_sigterm_emit_eight_and_exit_130_143(self):
        for signum, expected_exit in ((signal.SIGINT, 130), (signal.SIGTERM, 143)):
            with self.subTest(signum=signum), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                paths, manifest = lifecycle_fixture(base)
                terminal_root = base / "terminals"
                terminal_root.mkdir()
                manifest_path = base / "manifest.json"
                manifest_path.write_bytes(canonical_bytes(manifest))
                ready = base / "ready.json"
                expected_hashes = measure_hashes(
                    paths["archive"],
                    paths["ledger"],
                    paths["snapshot"],
                    paths["runner"],
                    manifest,
                )
                env = dict(os.environ)
                env["PYTHONDONTWRITEBYTECODE"] = "1"
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(PACKAGE / "signal_child_v6.py"),
                        "--terminal-root",
                        str(terminal_root),
                        "--archive",
                        str(paths["archive"]),
                        "--ledger",
                        str(paths["ledger"]),
                        "--snapshot-root",
                        str(paths["snapshot"]),
                        "--runner-root",
                        str(paths["runner"]),
                        "--runner-manifest",
                        str(manifest_path),
                        "--ready",
                        str(ready),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                )
                deadline = time.monotonic() + 10
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists(), process.stderr.read().decode() if process.poll() is not None else "not ready")
                process.send_signal(signum)
                stdout, stderr = process.communicate(timeout=10)
                self.assertEqual(stdout, b"")
                self.assertEqual(stderr, b"")
                self.assertEqual(process.returncode, expected_exit)
                validate_terminal_tree(terminal_root, expected_hashes, expected_hashes, "failure")

    def test_22_signal_exit_is_exact_and_rejects_other_values(self):
        self.assertEqual(signal_exit(2), 130)
        self.assertEqual(signal_exit(15), 143)
        for value in (0, 9, True, "15"):
            self.bad(signal_exit, value)

    def test_23_protected_parent_no_replace_preserves_bytes_and_inode(self):
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

    def test_24_retained_parent_final_failure_rolls_back_linked_output(self):
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

    def test_25_batch_publish_rolls_back_on_mid_link_failure(self):
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
                v6_runtime.os, "link", side_effect=failing_link
            ):
                with self.assertRaises(OSError):
                    parent.publish_many({"a": b"a", "b": b"b", "c": b"c"})
            self.assertEqual(list(root.iterdir()), [])

    def test_26_freeze_is_deterministic_transactional_and_rerun_immutable(self):
        first, first_method = freeze_v6.build_outputs(PACKAGE)
        second, second_method = freeze_v6.build_outputs(PACKAGE)
        self.assertEqual(first, second)
        self.assertEqual(first_method, second_method)
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            freeze_v6.freeze_package(PACKAGE, target)
            before = {
                path.name: (
                    path.stat().st_dev,
                    path.stat().st_ino,
                    path.stat().st_mode,
                    digest_file(path),
                )
                for path in target.iterdir()
            }
            self.bad(freeze_v6.freeze_package, PACKAGE, target)
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
            self.assertEqual(set(after), {"METHOD_FROZEN.json", freeze_v6.ARCHIVE_NAME, "source-ledger.json"})

    def test_27_torch_report_exact_provenance_uuid_visibility_and_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            package = Path(temporary) / "torch"
            package.mkdir()
            init = package / "__init__.py"
            init.write_text("# fixture\n")
            report = {
                "ignore_environment": True,
                "index": 0,
                "isolated": True,
                "loader_identity": True,
                "module_file": str(init.resolve()),
                "module_identity": True,
                "module_sha256": digest_file(init),
                "no_site": True,
                "physical_uuid": UUID0,
                "preimport_absent": True,
                "schema_version": "forkaudit-v6-torch-provenance-v1",
                "spec_origin": str(init.resolve()),
                "visibility": f"{UUID0},{UUID1}",
            }
            raw = canonical_bytes(report)
            validate_torch_report(raw, init, digest_file(init), report["visibility"], UUID0, 0)
            for key in ("preimport_absent", "module_identity", "loader_identity"):
                mutant = dict(report)
                mutant[key] = False
                self.bad(
                    validate_torch_report,
                    canonical_bytes(mutant),
                    init,
                    digest_file(init),
                    report["visibility"],
                    UUID0,
                    0,
                )
            self.bad(validate_torch_report, raw, init, digest_file(init), "", UUID0, 0)
            self.bad(validate_torch_report, raw, init, digest_file(init), report["visibility"], UUID0, -1)

    def test_28_isolated_clean_process_imports_only_inventory_bound_torch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "tools").mkdir()
            (root / "torch").mkdir()
            python_copy = root / "bin" / "python"
            shutil.copyfile(sys.executable, python_copy)
            python_copy.chmod(0o755)
            framework_binary = Path(sys.executable).resolve().parent.parent / "Python3"
            if framework_binary.is_file():
                shutil.copyfile(framework_binary, root / "Python3")
                (root / "Python3").chmod(0o755)
                app_binary = (
                    framework_binary.parent
                    / "Resources"
                    / "Python.app"
                    / "Contents"
                    / "MacOS"
                    / "Python"
                )
                copied_app = root / "Resources" / "Python.app" / "Contents" / "MacOS" / "Python"
                copied_app.parent.mkdir(parents=True)
                shutil.copyfile(app_binary, copied_app)
                copied_app.chmod(0o755)
            probe_copy = root / "tools" / "torch_probe_v6.py"
            shutil.copyfile(PACKAGE / "torch_probe_v6.py", probe_copy)
            init = root / "torch" / "__init__.py"
            init.write_text(
                "class _P:\n"
                f" uuid={UUID0!r}\n"
                "class _C:\n"
                " def get_device_properties(self,index): return _P()\n"
                "cuda=_C()\n"
            )
            manifest = tree_manifest(root)
            report = isolated_torch_probe(
                python_copy,
                probe_copy,
                init,
                digest_file(init),
                UUID0,
                UUID0,
                0,
                root,
                root,
                manifest,
            )
            self.assertTrue(report["isolated"])
            self.assertTrue(report["preimport_absent"])
            self.assertEqual(tree_manifest(root), manifest)

    def test_29_transplanted_probe_and_fake_report_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "bin").mkdir()
            (root / "tools").mkdir()
            (root / "torch").mkdir()
            python_copy = root / "bin" / "python"
            shutil.copyfile(sys.executable, python_copy)
            python_copy.chmod(0o755)
            fake_probe = root / "tools" / "torch_probe_v6.py"
            fake_probe.write_text("print('{}')\n")
            init = root / "torch" / "__init__.py"
            init.write_text("# no GPU call\n")
            manifest = tree_manifest(root)
            self.bad(
                isolated_torch_probe,
                python_copy,
                fake_probe,
                init,
                digest_file(init),
                UUID0,
                UUID0,
                0,
                root,
                root,
                manifest,
            )
            fake_report = {
                "ignore_environment": True,
                "index": 0,
                "isolated": True,
                "loader_identity": True,
                "module_file": str(init.resolve()),
                "module_identity": True,
                "module_sha256": digest_file(init),
                "no_site": True,
                "physical_uuid": UUID0,
                "preimport_absent": False,
                "schema_version": "forkaudit-v6-torch-provenance-v1",
                "spec_origin": str(init.resolve()),
                "visibility": UUID0,
            }
            self.bad(
                validate_torch_report,
                canonical_bytes(fake_report),
                init,
                digest_file(init),
                UUID0,
                UUID0,
                0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
