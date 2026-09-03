#!/usr/bin/env python3
import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
GUARD_SPEC = importlib.util.spec_from_file_location("rwd5_safe_cwd_guard", HERE / "safe_cwd_guard.py")
GUARD = importlib.util.module_from_spec(GUARD_SPEC)
GUARD_SPEC.loader.exec_module(GUARD)


class RecoveryRPublicationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher_text = (HERE / "launch_recovery_r.sh").read_text()
        cls.match = re.search(
            r"(?ms)^exec /usr/bin/env -i \\\n(?P<body>(?:  [A-Z][A-Z0-9_]*=.* \\\n)+)"
            r"  /bin/bash --noprofile --norc (?P<launcher>\S+)$",
            cls.launcher_text,
        )
        if cls.match is None:
            raise AssertionError("cannot parse the actual controlled exec block")
        cls.assignments = []
        cls.values = {}
        for line in cls.match.group("body").splitlines():
            token = line.strip()[:-1].strip()
            key, value = token.split("=", 1)
            cls.assignments.append(token)
            cls.values[key] = value

    def test_root_stat_and_sglang_py_fail_closed(self):
        good = SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=0, st_gid=0)
        GUARD.validate_root_stat(good)
        for bad in (
            SimpleNamespace(st_mode=stat.S_IFDIR | 0o777, st_uid=0, st_gid=0),
            SimpleNamespace(st_mode=stat.S_IFDIR | 0o755, st_uid=1, st_gid=0),
            SimpleNamespace(st_mode=stat.S_IFREG | 0o755, st_uid=0, st_gid=0),
        ):
            with self.assertRaises(GUARD.SafeCwdError):
                GUARD.validate_root_stat(bad)
        with tempfile.TemporaryDirectory(prefix="rwd5-shadow-root-") as temp:
            root = Path(temp)
            (root / "sglang.py").write_text("ATTACKER = True\n")
            with self.assertRaises(GUARD.SafeCwdError):
                GUARD.validate_import_shadows(root)
        with tempfile.TemporaryDirectory(prefix="rwd5-shadow-link-") as temp:
            root = Path(temp)
            (root / "sglang").symlink_to(root / "missing-target")
            with self.assertRaises(GUARD.SafeCwdError):
                GUARD.validate_import_shadows(root)

    def test_final_environment_is_closed_and_has_no_empty_library_component(self):
        required = {
            "PYTHON_BIN", "OFFICIAL_REPO", "FREEZE_ROOT", "CODE_DIR",
            "FREEZE_MANIFEST", "EXPECTED_FREEZE_MANIFEST_SHA256",
            "LIVE_DEBUG_ROOT", "MODEL_DIR", "MODEL_WEIGHT_LEDGER",
            "MODEL_ARTIFACT_LEDGER", "VALIDATION_DATA", "RUN_DIR",
            "INSTRUMENTED_REPO",
            "ALLOCATOR_DEBUG_ROOT", "ALLOCATOR_DEBUG_PROVENANCE",
            "ALLOCATOR_DEBUG_LAUNCH_PLAN", "ALLOCATOR_DEBUG_FREEZE_MANIFEST",
        }
        self.assertTrue(required.issubset(self.values))
        self.assertEqual(self.values["PYTHON_BIN"], "/tmp/round25-hypic-env/venv/bin/python")
        self.assertEqual(self.values["CODE_DIR"], "/tmp/rwd5-hypic-store-freeze-q/code")
        self.assertEqual(
            self.values["ALLOCATOR_DEBUG_ROOT"],
            "/tmp/rwd5-hypic-store-freeze-q/live-allocator-debug-d-trial-1879456",
        )
        self.assertEqual(
            self.values["ALLOCATOR_DEBUG_PROVENANCE"],
            "/tmp/rwd5-hypic-store-freeze-q/allocator-debug-d-provenance.json",
        )
        self.assertEqual(self.values["PYTHONSAFEPATH"], "1")
        self.assertNotIn("", self.values["LIBRARY_PATH"].split(":"))
        malicious = dict(os.environ)
        for key in required:
            malicious[key] = f"/tmp/attacker-{key.lower()}"
        malicious["BASH_ENV"] = "/tmp/attacker-bash-env"
        result = subprocess.run(
            ["/usr/bin/env", "-i", *self.assignments, "/usr/bin/env"],
            env=malicious, text=True, capture_output=True, check=True,
        )
        self.assertEqual(
            dict(line.split("=", 1) for line in result.stdout.splitlines()),
            self.values,
        )
        self.assertEqual(
            self.match.group("launcher"),
            "/tmp/rwd5-hypic-store-freeze-q/code/launch_hypic_retained_state_bytes_8gpu.sh",
        )

    def test_wrapper_has_no_stale_p_or_k_launch_identity(self):
        for stale in ("recovery-P", "fresh P", "K_LAUNCHER", "freeze-k", "store-k"):
            self.assertNotIn(stale, self.launcher_text)
        self.assertIn("EXPECTED_Q_STOP_SHA256", self.launcher_text)
        self.assertIn("R manifest identity drift", self.launcher_text)

    def test_wrapper_binds_frozen_python_and_exact_import_origins(self):
        python_bin = self.values["PYTHON_BIN"]
        self.assertGreaterEqual(self.launcher_text.count(python_bin), 4)
        self.assertIn('assert "" not in sys.path; assert "/" not in sys.path', self.launcher_text)
        self.assertIn(
            'Path("/tmp/rwd5-hypic-store-freeze-q/code/test_hypic_retained_state_receipt.py")',
            self.launcher_text,
        )
        self.assertIn("PYTHONPATH=/tmp/rwd5-hypic-store-freeze-q/code", self.launcher_text)
        self.assertIn(
            "PYTHONPATH=/tmp/HYPIC-98147c0/python:/tmp/rwd5-hypic-store-freeze-q/code",
            self.launcher_text,
        )
        self.assertIn('Path("/tmp/HYPIC-98147c0/python/sglang")', self.launcher_text)

    def test_publication_occurs_only_after_all_probes_and_immediately_before_exec(self):
        guard = self.launcher_text.index('safe_cwd_guard.py" /')
        safe_path = self.launcher_text.index('assert "" not in sys.path; assert "/" not in sys.path')
        focused = self.launcher_text.index('focused unittest import authority is not exact Q code')
        sglang = self.launcher_text.index('SGLang import authority is not exact official repository')
        publication = self.launcher_text.index("# AUTHORITY_PUBLICATION_BEGIN")
        status = self.launcher_text.index("all_recovery_preflight_passed_before_q_exec")
        controlled_exec = self.launcher_text.index("# PINNED_EXEC_ENV_BEGIN")
        self.assertLess(max(guard, safe_path, focused, sglang), publication)
        self.assertLess(publication, status)
        self.assertLess(status, controlled_exec)
        self.assertNotIn("preflight_passed_before_k_launcher", self.launcher_text)

    def test_guard_and_origin_failure_create_no_receipt_or_pass_marker(self):
        with tempfile.TemporaryDirectory(prefix="rwd5-no-false-pass-") as temp:
            root = Path(temp)
            receipt = root / "receipt"
            bad_root = root / "bad-root"
            bad_root.mkdir(mode=0o777)
            guard_command = [
                sys.executable, str(HERE / "safe_cwd_guard.py"), str(bad_root)
            ]
            focused_probe = (
                "import importlib.util,pathlib;"
                "s=importlib.util.find_spec('test_hypic_retained_state_receipt');"
                "assert s and s.origin;"
                "assert pathlib.Path(s.origin).resolve() == pathlib.Path('/definitely/frozen/Q/test_hypic_retained_state_receipt.py')"
            )
            failing_commands = [
                " ".join(shlex.quote(x) for x in guard_command),
                "PYTHONPATH=" + shlex.quote(str(root)) + " " + shlex.quote(sys.executable)
                + " -c " + shlex.quote(focused_probe),
            ]
            for probe in failing_commands:
                if receipt.exists():
                    self.fail("test receipt unexpectedly pre-existed")
                gated = (
                    "set -e; " + probe + "; "
                    "mkdir -p " + shlex.quote(str(receipt)) + "; "
                    "printf '%s\\n' all_recovery_preflight_passed_before_q_exec > "
                    + shlex.quote(str(receipt / "STATUS"))
                )
                result = subprocess.run(
                    ["/bin/bash", "--noprofile", "--norc", "-c", gated],
                    cwd=str(root), env={}, text=True, capture_output=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(receipt.exists())
                self.assertFalse((receipt / "STATUS").exists())

    def test_actual_pythonpath_semantics_ignore_malicious_cwd(self):
        with tempfile.TemporaryDirectory(prefix="rwd5-authority-") as temp:
            root = Path(temp)
            attacker = root / "attacker"
            q_code = root / "q-code"
            official_python = root / "official" / "python"
            attacker.mkdir()
            q_code.mkdir()
            (official_python / "sglang").mkdir(parents=True)
            (attacker / "sglang.py").write_text("ATTACKER = True\n")
            (attacker / "test_hypic_retained_state_receipt.py").write_text("ATTACKER = True\n")
            expected_test = q_code / "test_hypic_retained_state_receipt.py"
            expected_test.write_text("FROZEN = True\n")
            expected_sglang = official_python / "sglang" / "__init__.py"
            expected_sglang.write_text("OFFICIAL = True\n")
            probe = (
                "import importlib.util,json,os,pathlib;"
                "t=importlib.util.find_spec('test_hypic_retained_state_receipt');"
                "s=importlib.util.find_spec('sglang');"
                "print(json.dumps({'cwd':os.getcwd(),'test':str(pathlib.Path(t.origin).resolve()),"
                "'sglang':str(pathlib.Path(s.origin).resolve())},sort_keys=True))"
            )
            assignments = [
                "PYTHONNOUSERSITE=1", "PYTHONSAFEPATH=1",
                f"PYTHONPATH={official_python}:{q_code}",
            ]
            command = "cd / && exec /usr/bin/env -i " + " ".join(
                shlex.quote(item) for item in assignments
            ) + " " + shlex.quote(sys.executable) + " -c " + shlex.quote(probe)
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", command],
                cwd=attacker, env={}, text=True, capture_output=True, check=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["cwd"], "/")
            self.assertEqual(payload["test"], str(expected_test.resolve()))
            self.assertEqual(payload["sglang"], str(expected_sglang.resolve()))


if __name__ == "__main__":
    unittest.main()
