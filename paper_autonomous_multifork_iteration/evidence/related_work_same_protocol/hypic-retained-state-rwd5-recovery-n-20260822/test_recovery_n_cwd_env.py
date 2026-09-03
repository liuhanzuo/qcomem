#!/usr/bin/env python3
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class RecoveryNEnvironmentAndCwdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher_text = Path(__file__).with_name("launch_recovery_n.sh").read_text()
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

    def test_malicious_ambient_is_cleared_and_every_k_override_is_pinned(self):
        required = {
            "PYTHON_BIN", "OFFICIAL_REPO", "FREEZE_ROOT", "CODE_DIR",
            "FREEZE_MANIFEST", "EXPECTED_FREEZE_MANIFEST_SHA256",
            "LIVE_DEBUG_ROOT", "MODEL_DIR", "MODEL_WEIGHT_LEDGER",
            "MODEL_ARTIFACT_LEDGER", "VALIDATION_DATA", "RUN_DIR",
            "INSTRUMENTED_REPO",
        }
        self.assertTrue(required.issubset(self.values))
        self.assertEqual(self.values["CODE_DIR"], "/tmp/rwd5-hypic-store-freeze-k/code")
        self.assertEqual(self.values["PYTHON_BIN"], "/tmp/round25-hypic-env/venv/bin/python")
        self.assertEqual(self.values["PYTHONSAFEPATH"], "1")
        self.assertEqual(
            self.values["EXPECTED_FREEZE_MANIFEST_SHA256"],
            "c7f0fcc0b44d6292af52f2d31a0770e7a74982c20eadc8317740106034dc3a7b",
        )
        self.assertNotIn("", self.values["LIBRARY_PATH"].split(":"))
        malicious = dict(os.environ)
        for key in required:
            malicious[key] = f"/tmp/attacker-{key.lower()}"
        malicious["BASH_ENV"] = "/tmp/attacker-bash-env"
        result = subprocess.run(
            ["/usr/bin/env", "-i", *self.assignments, "/usr/bin/env"],
            env=malicious, text=True, capture_output=True, check=True,
        )
        observed = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(observed, self.values)
        self.assertNotIn("BASH_ENV", observed)
        self.assertNotIn("PYTHONPATH", observed)
        self.assertEqual(
            self.match.group("launcher"),
            "/tmp/rwd5-hypic-store-freeze-k/code/launch_hypic_retained_state_bytes_8gpu.sh",
        )

    def test_real_child_escapes_malicious_cwd_and_python_path(self):
        pinned_cwd = re.search(
            r"(?ms)^# PINNED_CWD_BEGIN.*?^cd /$.*?^\[\[ \"\$PWD\" == \"/\" \]\].*?"
            r"^# PINNED_CWD_END$",
            self.launcher_text,
        )
        self.assertIsNotNone(pinned_cwd)
        self.assertIn(
            "assert os.getcwd() == \"/\"; assert \"\" not in sys.path",
            self.launcher_text,
        )
        with tempfile.TemporaryDirectory(prefix="rwd5-attacker-cwd-") as attacker:
            attacker_path = Path(attacker)
            (attacker_path / "sglang").mkdir()
            (attacker_path / "sglang" / "__init__.py").write_text("ATTACKER = True\n")
            (attacker_path / "test_hypic_retained_state_receipt.py").write_text(
                "ATTACKER = True\n"
            )
            probe = (
                "import importlib.util,json,os,sys;"
                "mods=['sglang','test_hypic_retained_state_receipt'];"
                "print(json.dumps({'cwd':os.getcwd(),'sys_path':sys.path,"
                "'origins':{m:(getattr(importlib.util.find_spec(m),'origin',None) "
                "if importlib.util.find_spec(m) else None) for m in mods}},sort_keys=True))"
            )
            command = "cd / && exec /usr/bin/env -i " + " ".join(
                shlex.quote(item) for item in self.assignments
            ) + " " + shlex.quote(sys.executable) + " -c " + shlex.quote(probe)
            result = subprocess.run(
                ["/bin/bash", "--noprofile", "--norc", "-c", command],
                cwd=attacker, env={"BASH_ENV": str(attacker_path / "evil.sh")},
                text=True, capture_output=True, check=True,
            )
            payload = json.loads(result.stdout)
            self.assertEqual(payload["cwd"], "/")
            self.assertNotIn(str(attacker_path), payload["sys_path"])
            for origin in payload["origins"].values():
                self.assertFalse(origin and str(origin).startswith(str(attacker_path)))


if __name__ == "__main__":
    unittest.main()
