from __future__ import annotations

import copy
import inspect
import json
import os
import shutil
import signal
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import authorized_launcher_v10 as launcher
import fixtures_v10 as fx
import freeze_v10
import static_audit_v10
import v10_guard
import v10_runtime


PACKAGE = Path(__file__).resolve().parent


def call_kwargs(fixture: dict) -> dict:
    runtime = fixture["runtime"]
    return {
        "archive_path": str(fixture["archive"]),
        "attempt": fixture["attempt"],
        "binding_path": str(fixture["binding_path"]),
        "consumption_root": str(fixture["consumption_root"]),
        "execution_contract_path": str(runtime["contract_path"]),
        "run_nonce": fixture["run_nonce"],
        "runner_manifest_path": str(runtime["manifest_path"]),
        "runner_root": str(runtime["runner"]),
        "runtime_expectation_path": str(runtime["expectation_path"]),
        "snapshot_root": str(fixture["snapshot"]),
        "source_ledger_path": str(fixture["ledger"]),
        "terminal_root": str(fixture["terminal_root"]),
    }


def trusted_run(fixture: dict, **patches: object) -> int:
    with mock.patch.object(v10_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
        v10_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", fixture["public"].hex()
    ):
        if patches:
            with mock.patch.multiple(launcher.subprocess, **patches):
                return launcher.run_authorized_campaign(**call_kwargs(fixture))
        return launcher.run_authorized_campaign(**call_kwargs(fixture))


def terminal_rows(fixture: dict) -> list[dict]:
    return [
        json.loads(
            (fixture["terminal_root"] / f"{fault_id}.terminal.json").read_text()
        )
        for fault_id in fx.TERM_IDS
    ]


def validate_tree(fixture: dict, row: dict, status: str) -> None:
    v10_guard.validate_terminal_tree(
        fixture["terminal_root"],
        fixture["consumption_root"],
        row["pre_hashes"],
        row["post_hashes"],
        row["provenance"]["spawned_specs_sha256"],
        status,
    )


class V10SecurityTests(unittest.TestCase):
    def test_01_packaged_test_discovery_count_and_no_skip(self):
        audit = static_audit_v10.audit(PACKAGE)
        discovered = unittest.defaultTestLoader.discover(
            str(PACKAGE), pattern="test*.py"
        ).countTestCases()
        self.assertEqual(audit["status"], "PASS", audit)
        self.assertEqual(audit["packaged_tests"], discovered)
        self.assertEqual(audit["skip_sites"], [])
        self.assertTrue(all(name == "test_v10.py" for name, _, _ in audit["test_sites"]))

    def test_02_single_authorized_process_surface_and_no_legacy_capability(self):
        audit = static_audit_v10.audit(PACKAGE)
        production = [
            site
            for site in audit["process_sites"]
            if not site["file"].startswith("test")
        ]
        self.assertEqual(
            {site["api"] for site in production},
            {"subprocess.run", "subprocess.Popen"},
        )
        self.assertTrue(
            all(
                site["file"] == "authorized_launcher_v10.py"
                and "run_authorized_campaign" in site["functions"]
                for site in production
            )
        )
        for module in (launcher, v10_guard, v10_runtime):
            for name in (
                "_AUTHORIZED_PLAN_TOKEN",
                "AuthorizedPlan",
                "Lifecycle",
                "_spawn_worker",
                "spawn_worker",
                "spawn_workers",
                "prepare_authorized_plan",
                "isolated_torch_probe",
            ):
                self.assertFalse(hasattr(module, name), (module.__name__, name))
        self.assertNotIn("subprocess", vars(v10_guard))
        self.assertNotIn("subprocess", vars(v10_runtime))
        signature = inspect.signature(launcher.run_authorized_campaign)
        self.assertIn("consumption_root", signature.parameters)
        self.assertFalse(any("spec" in name or "worker" in name for name in signature.parameters))

    def test_03_canonical_json_and_ed25519_strictness(self):
        with self.assertRaises(v10_guard.Reject):
            v10_guard.canonical_load(b'{"a":1,"a":2}\n', "duplicate")
        with self.assertRaises(v10_guard.Reject):
            v10_guard.canonical_load(b'{"a": 1}\n', "whitespace")
        seed = bytes.fromhex(
            "9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"
        )
        public, signature = fx._rfc_test_sign(seed, b"")
        self.assertTrue(v10_guard.ed25519_verify(public, b"", signature))
        self.assertFalse(v10_guard.ed25519_verify(public, b"x", signature))

    def test_04_archive_ledger_closure_and_canonical_mutants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_raw, ledger_raw = fx.archive_fixture()
            archive = root / "archive.tgz"
            archive.write_bytes(archive_raw)
            result = v10_guard.verify_archive(
                archive,
                v10_guard.digest_bytes(archive_raw),
                v10_guard.digest_bytes(ledger_raw),
            )
            self.assertEqual(result["members"], 2)
            traversal = fx.custom_tar(
                [("../x", b"x", 0o444, 0), ("source-ledger.json", ledger_raw, 0o444, 0)]
            )
            archive.write_bytes(traversal)
            with self.assertRaises(v10_guard.Reject):
                v10_guard.verify_archive(
                    archive,
                    v10_guard.digest_bytes(traversal),
                    v10_guard.digest_bytes(ledger_raw),
                )
            archive.write_bytes(archive_raw + archive_raw)
            with self.assertRaises(v10_guard.Reject):
                v10_guard.strict_gzip(archive.read_bytes())

    def test_05_snapshot_and_runner_reject_links(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            snapshot.mkdir()
            item = snapshot / "x"
            item.write_bytes(b"x")
            v10_guard.snapshot_commitments(snapshot)
            link = snapshot / "link"
            link.symlink_to(item)
            with self.assertRaises(v10_guard.Reject):
                v10_guard.snapshot_commitments(snapshot)
            link.unlink()
            os.link(item, link)
            with self.assertRaises(v10_guard.Reject):
                v10_guard.snapshot_commitments(snapshot)

    def test_06_execution_contract_typed_policy_rejects_semantic_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = fx.execution_runtime_fixture(Path(temporary), real_python=False)
            contract = copy.deepcopy(runtime["contract"])
            contract["workers"][0]["env"]["LC_ALL"] = "evil"
            raw = v10_guard.canonical_bytes(contract)
            with self.assertRaises(v10_guard.Reject):
                v10_guard.validate_execution_contract(
                    raw,
                    v10_guard.digest_bytes(raw),
                    runtime["expectation_path"],
                    runtime["runner"],
                    runtime["manifest_path"],
                    runtime["manifest_sha256"],
                    runtime["inventory_sha256"],
                    runtime["expectation_sha256"],
                )

    def test_07_valid_eight_workers_use_child_observed_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = fx.signed_fixture(Path(temporary), real_python=True)
            self.assertEqual(trusted_run(fixture), 0)
            self.assertEqual(len(list(fixture["terminal_root"].iterdir())), 9)
            self.assertEqual(len(list(fixture["consumption_root"].iterdir())), 1)
            rows = terminal_rows(fixture)
            validate_tree(fixture, rows[0], "success")
            gate = rows[0]["lifecycle_receipt"]
            self.assertTrue(gate["actual_specs_verified"])
            self.assertEqual(gate["provenance"]["spawned_specs_sha256"], rows[0]["provenance"]["spawned_specs_sha256"])
            for worker in gate["workers"]:
                self.assertTrue(worker["authority_match"])
                self.assertTrue(worker["child_report_verified"])
                self.assertEqual(worker["child_report"]["pid"], worker["pid"])
                self.assertEqual(worker["child_report"]["pgid"], worker["pgid"])
                self.assertEqual(worker["pid"], worker["pgid"])
                loaded = fixture["runtime"]["expectation"]["python"]
                loaded_path = (
                    fixture["runtime"]["runner"] / loaded["loaded_manifest_path"]
                ).resolve(strict=True)
                loaded_stat = os.stat(loaded_path, follow_symlinks=False)
                self.assertEqual(worker["actual_executable"]["path"], str(loaded_path))
                self.assertEqual(worker["actual_argv"][0], str(loaded_path))
                self.assertEqual(
                    (worker["actual_executable"]["dev"], worker["actual_executable"]["ino"]),
                    (loaded_stat.st_dev, loaded_stat.st_ino),
                )
                self.assertEqual(
                    worker["actual_executable"]["sha256"], loaded["loaded_sha256"]
                )
                self.assertEqual(worker["actual_cuda_visible_devices"], fx.UUID0)

    def test_08_wrong_root_consumption_root_attempt_nonce_reject_before_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = fx.signed_fixture(base, real_python=False)
            fresh_terminal = base / "fresh-terminal"
            fresh_terminal.mkdir()
            fresh_durable = base / "fresh-durable"
            fresh_durable.mkdir()
            variants = []
            call = call_kwargs(fixture)
            call["terminal_root"] = str(fresh_terminal)
            variants.append(call)
            call = call_kwargs(fixture)
            call["consumption_root"] = str(fresh_durable)
            variants.append(call)
            call = call_kwargs(fixture)
            call["attempt"] += 1
            variants.append(call)
            call = call_kwargs(fixture)
            call["run_nonce"] = "b" * 64
            variants.append(call)
            for call in variants:
                with self.subTest(call=call):
                    with mock.patch.object(v10_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
                        v10_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", fixture["public"].hex()
                    ), mock.patch.object(
                        launcher.subprocess, "run", side_effect=AssertionError("probe reached")
                    ), mock.patch.object(
                        launcher.subprocess, "Popen", side_effect=AssertionError("worker reached")
                    ):
                        with self.assertRaises(v10_guard.Reject):
                            launcher.run_authorized_campaign(**call)

    def test_09_durable_replay_rejects_nonempty_cleared_and_recreated_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = fx.signed_fixture(base, real_python=True)
            self.assertEqual(trusted_run(fixture), 0)
            with self.assertRaises(v10_guard.Reject):
                trusted_run(fixture)
            for item in list(fixture["terminal_root"].iterdir()):
                item.unlink()
            with mock.patch.object(launcher.subprocess, "run", side_effect=AssertionError("probe replay")), mock.patch.object(
                launcher.subprocess, "Popen", side_effect=AssertionError("worker replay")
            ):
                with self.assertRaises(v10_guard.Reject):
                    trusted_run(fixture)
            first_binding_sha = v10_guard.digest_file(fixture["binding_path"])
            fixture["payload"]["operator_id"] = "independent-test-operator-reissue"
            fx.resign_fixture(fixture)
            self.assertNotEqual(
                first_binding_sha, v10_guard.digest_file(fixture["binding_path"])
            )
            with mock.patch.object(
                launcher.subprocess, "run", side_effect=AssertionError("probe reissue")
            ), mock.patch.object(
                launcher.subprocess, "Popen", side_effect=AssertionError("worker reissue")
            ):
                with self.assertRaises(v10_guard.Reject):
                    trusted_run(fixture)
            durable_inode = os.lstat(fixture["consumption_root"]).st_ino
            moved_durable = base / "old-consumptions"
            fixture["consumption_root"].rename(moved_durable)
            fixture["consumption_root"].mkdir()
            self.assertNotEqual(
                durable_inode, os.lstat(fixture["consumption_root"]).st_ino
            )
            with self.assertRaises(v10_guard.Reject):
                trusted_run(fixture)
            fixture["consumption_root"].rmdir()
            moved_durable.rename(fixture["consumption_root"])
            old_inode = os.lstat(fixture["terminal_root"]).st_ino
            moved = base / "old-terminal"
            fixture["terminal_root"].rename(moved)
            fixture["terminal_root"].mkdir()
            self.assertNotEqual(old_inode, os.lstat(fixture["terminal_root"]).st_ino)
            with self.assertRaises(v10_guard.Reject):
                trusted_run(fixture)

    def test_10_child_handshake_closes_executable_path_to_popen_race(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = fx.signed_fixture(base, real_python=True)
            original = launcher.subprocess.Popen
            worker_pids = []

            def racing(*args: object, **kw: object):
                argv = args[0]
                if isinstance(argv, list) and "--worker-payload" in argv:
                    target = Path(argv[0])
                    saved = base / "saved-python"
                    os.rename(target, saved)
                    try:
                        shutil.copyfile("/usr/bin/true", target)
                        target.chmod(0o755)
                        process = original(*args, **kw)
                        worker_pids.append(process.pid)
                    finally:
                        target.unlink()
                        os.rename(saved, target)
                    return process
                return original(*args, **kw)

            with mock.patch.object(v10_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
                v10_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", fixture["public"].hex()
            ), mock.patch.object(launcher.subprocess, "Popen", side_effect=racing):
                result = launcher.run_authorized_campaign(**call_kwargs(fixture))
            self.assertEqual(result, 1)
            self.assertTrue(worker_pids)
            for pid in worker_pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            rows = terminal_rows(fixture)
            validate_tree(fixture, rows[0], "failure")
            first = rows[0]["lifecycle_receipt"]["workers"][0]
            self.assertFalse(first["child_report_verified"])
            self.assertFalse(first["authority_match"])

        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = fx.signed_fixture(base, real_python=True)
            original = launcher.subprocess.Popen
            worker_pids = []

            def racing_loaded_inode(*args: object, **kw: object):
                argv = args[0]
                if isinstance(argv, list) and "--worker-payload" in argv:
                    python = fixture["runtime"]["expectation"]["python"]
                    target = (
                        fixture["runtime"]["runner"]
                        / python["loaded_manifest_path"]
                    )
                    saved = base / "saved-loaded-python"
                    replacement = base / "replacement-loaded-python"
                    shutil.copyfile(target, replacement)
                    replacement.chmod(0o755)
                    os.rename(target, saved)
                    try:
                        os.rename(replacement, target)
                        process = original(*args, **kw)
                        worker_pids.append(process.pid)
                        time.sleep(0.2)
                    finally:
                        target.unlink()
                        os.rename(saved, target)
                    return process
                return original(*args, **kw)

            with mock.patch.object(v10_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
                v10_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", fixture["public"].hex()
            ), mock.patch.object(
                launcher.subprocess, "Popen", side_effect=racing_loaded_inode
            ):
                result = launcher.run_authorized_campaign(**call_kwargs(fixture))
            self.assertEqual(result, 1)
            self.assertTrue(worker_pids)
            for pid in worker_pids:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            rows = terminal_rows(fixture)
            validate_tree(fixture, rows[0], "failure")
            first = rows[0]["lifecycle_receipt"]["workers"][0]
            self.assertFalse(first["child_report_verified"])
            self.assertFalse(first["authority_match"])

    def test_11_terminal_typed_aggregate_rejects_coordinated_forgery(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = fx.signed_fixture(Path(temporary), real_python=True)
            self.assertEqual(trusted_run(fixture), 0)
            paths = [
                fixture["terminal_root"] / f"{fault_id}.terminal.json"
                for fault_id in fx.TERM_IDS
            ]
            originals = [json.loads(path.read_text()) for path in paths]
            gate = copy.deepcopy(originals[0]["lifecycle_receipt"])
            worker = gate["workers"][0]
            worker["actual_env"]["LC_ALL"] = "evil"
            worker["child_report"]["actual_env"]["LC_ALL"] = "evil"
            worker["child_report_sha256"] = v10_guard.digest_bytes(
                v10_guard.canonical_bytes(worker["child_report"])
            )
            spec = {
                "argv": worker["actual_argv"],
                "argv_schema": worker["actual_argv_schema"],
                "cwd": worker["actual_cwd"]["path"],
                "cwd_contract": worker["actual_cwd"]["contract"],
                "env": worker["actual_env"],
                "env_schema": worker["actual_env_schema"],
                "fault_id": worker["fault_id"],
            }
            worker["spawned_spec_sha256"] = v10_guard.digest_bytes(
                v10_guard.canonical_bytes(spec)
            )
            gate_sha = v10_guard.digest_bytes(v10_guard.canonical_bytes(gate))
            for path, row in zip(paths, originals):
                row["lifecycle_receipt"] = copy.deepcopy(gate)
                row["lifecycle_receipt_sha256"] = gate_sha
                path.chmod(0o600)
                path.write_bytes(v10_guard.canonical_bytes(row))
            with self.assertRaises(v10_guard.Reject):
                validate_tree(fixture, originals[0], "success")

    def test_12_terminal_exact_names_and_expected_aggregate_anchor(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = fx.signed_fixture(Path(temporary), real_python=True)
            self.assertEqual(trusted_run(fixture), 0)
            rows = terminal_rows(fixture)
            extra = fixture["terminal_root"] / "extra"
            extra.write_bytes(b"x")
            with self.assertRaises(v10_guard.Reject):
                validate_tree(fixture, rows[0], "success")
            extra.unlink()
            with self.assertRaises(v10_guard.Reject):
                v10_guard.validate_terminal_tree(
                    fixture["terminal_root"],
                    fixture["consumption_root"],
                    rows[0]["pre_hashes"],
                    rows[0]["post_hashes"],
                    "f" * 64,
                    "success",
                )

    def test_13_sigint_sigterm_popen_registration_has_no_orphan(self):
        self.assertTrue(hasattr(os, "fork"), "POSIX fork is required")
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signum=signum), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                fixture = fx.signed_fixture(base, real_python=True, sleeping=True)
                pid_file = base / "worker.pid"
                child = os.fork()
                if child == 0:
                    original = launcher.subprocess.Popen

                    def racing(*args: object, **kw: object):
                        process = original(*args, **kw)
                        argv = args[0]
                        if isinstance(argv, list) and "--worker-payload" in argv:
                            pid_file.write_text(str(process.pid))
                            os.kill(os.getpid(), signum)
                        return process

                    try:
                        with mock.patch.object(v10_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
                            v10_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", fixture["public"].hex()
                        ), mock.patch.object(launcher.subprocess, "Popen", side_effect=racing):
                            code = launcher.run_authorized_campaign(**call_kwargs(fixture))
                        os._exit(code)
                    except SystemExit as exc:
                        os._exit(int(exc.code))
                    except BaseException:
                        os._exit(99)
                _, status = os.waitpid(child, 0)
                self.assertEqual(os.waitstatus_to_exitcode(status), 128 + signum)
                worker_pid = int(pid_file.read_text())
                time.sleep(0.1)
                with self.assertRaises(ProcessLookupError):
                    os.kill(worker_pid, 0)
                rows = terminal_rows(fixture)
                validate_tree(fixture, rows[0], "failure")
                self.assertEqual({row["signal"] for row in rows}, {signum})

    def test_14_no_replace_and_late_parent_replacement_roll_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "x"
            target.write_bytes(b"old")
            before = (target.read_bytes(), target.stat().st_ino)
            with v10_runtime.ProtectedParent(root) as publisher:
                with self.assertRaises(v10_guard.Reject):
                    publisher.publish_many({"x": b"new"})
            self.assertEqual((target.read_bytes(), target.stat().st_ino), before)
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "root"
            root.mkdir()
            moved = base / "moved"

            class Late(v10_runtime.ProtectedParent):
                def __init__(self, path: Path):
                    super().__init__(path)
                    self.calls = 0

                def check(self) -> None:
                    self.calls += 1
                    if self.calls == 2:
                        self._path.rename(moved)
                        self._path.mkdir()
                    super().check()

            publisher = Late(root)
            try:
                with self.assertRaises(v10_guard.Reject):
                    publisher.publish_many({"a": b"a", "b": b"b"})
            finally:
                publisher.close()
            self.assertEqual(list(moved.iterdir()), [])
            self.assertEqual(list(root.iterdir()), [])

    def test_15_unsigned_and_signed_semantic_mismatch_reject_before_process(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            fixture = fx.signed_fixture(base, real_python=False)
            missing = call_kwargs(fixture)
            missing["binding_path"] = str(base / "missing")
            with mock.patch.object(launcher.subprocess, "run", side_effect=AssertionError("probe reached")), mock.patch.object(
                launcher.subprocess, "Popen", side_effect=AssertionError("worker reached")
            ):
                with self.assertRaises(v10_guard.Reject):
                    launcher.run_authorized_campaign(**missing)
            contract = fixture["runtime"]["contract"]
            contract["workers"][0]["env"]["CUDA_VISIBLE_DEVICES"] = "NOT-A-UUID"
            fixture["runtime"]["contract_path"].write_bytes(
                v10_guard.canonical_bytes(contract)
            )
            fx.resign_fixture(fixture)
            with mock.patch.object(v10_guard, "_compiled_trust_root", return_value=None), mock.patch.object(
                v10_guard, "TRUST_ROOT_PUBLIC_KEY_HEX", fixture["public"].hex()
            ), mock.patch.object(launcher.subprocess, "run", side_effect=AssertionError("probe reached")), mock.patch.object(
                launcher.subprocess, "Popen", side_effect=AssertionError("worker reached")
            ):
                with self.assertRaises(v10_guard.Reject):
                    launcher.run_authorized_campaign(**call_kwargs(fixture))

    def test_16_consumption_binds_both_directory_objects(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = fx.signed_fixture(Path(temporary), real_python=True)
            self.assertEqual(trusted_run(fixture), 0)
            raw = (fixture["terminal_root"] / "AUTHORIZED_CONSUMPTION.json").read_bytes()
            record = v10_guard.consumption_record(
                v10_guard.canonical_load(raw, "consumption")
            )
            terminal_stat = os.lstat(fixture["terminal_root"])
            durable_stat = os.lstat(fixture["consumption_root"])
            self.assertEqual(
                (record["terminal_root_dev"], record["terminal_root_ino"]),
                (terminal_stat.st_dev, terminal_stat.st_ino),
            )
            self.assertEqual(
                (record["consumption_root_dev"], record["consumption_root_ino"]),
                (durable_stat.st_dev, durable_stat.st_ino),
            )
            self.assertEqual(
                raw,
                (fixture["consumption_root"] / record["consumption_name"]).read_bytes(),
            )
            self.assertEqual(
                record["consumption_name"],
                f"{record['run_identity_sha256']}.consumed.json",
            )

    def test_17_operator_template_and_binding_api_are_closed(self):
        template = json.loads((PACKAGE / "operator-binding.template.json").read_text())
        approved = {
            key: value
            for key, value in template["payload"].items()
            if key.startswith("approved_")
        }
        self.assertEqual(len(approved), 16)
        self.assertTrue(all(value is None for value in approved.values()))
        parameters = tuple(inspect.signature(v10_guard.verify_operator_binding).parameters)
        self.assertEqual(
            parameters,
            (
                "raw_binding", "archive_path", "source_ledger_path", "snapshot_root",
                "execution_contract_path", "runtime_expectation_path", "runner_root",
                "runner_manifest_path", "terminal_root", "consumption_root", "attempt",
                "run_nonce",
            ),
        )

    def test_18_deterministic_build_has_exact_test_count_and_hold_status(self):
        first, first_method = freeze_v10.build_outputs(PACKAGE)
        second, second_method = freeze_v10.build_outputs(PACKAGE)
        self.assertEqual(first, second)
        self.assertEqual(first_method, second_method)
        self.assertEqual(
            first_method["tests"], static_audit_v10.packaged_test_count(PACKAGE)
        )
        self.assertEqual(
            first_method["status"],
            "HOLD_PENDING_FRESH_INDEPENDENT_AUDIT_AND_EXTERNAL_BINDING",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
