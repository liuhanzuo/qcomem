#!/usr/bin/env python3
import os
import re
import subprocess
import unittest
from pathlib import Path


class RecoveryMEnvironmentTest(unittest.TestCase):
    def test_malicious_ambient_is_cleared_and_every_k_override_is_pinned(self):
        launcher = Path(__file__).with_name("launch_recovery_m.sh").read_text()
        match = re.search(
            r"(?ms)^exec /usr/bin/env -i \\\n(?P<body>(?:  [A-Z][A-Z0-9_]*=.* \\\n)+)  /bin/bash --noprofile --norc (?P<launcher>\S+)$",
            launcher,
        )
        self.assertIsNotNone(match)
        assignments = []
        values = {}
        for line in match.group("body").splitlines():
            token = line.strip()[:-1].strip()
            key, value = token.split("=", 1)
            assignments.append(token)
            values[key] = value
        required = {
            "PYTHON_BIN", "OFFICIAL_REPO", "FREEZE_ROOT", "CODE_DIR",
            "FREEZE_MANIFEST", "EXPECTED_FREEZE_MANIFEST_SHA256",
            "LIVE_DEBUG_ROOT", "MODEL_DIR", "MODEL_WEIGHT_LEDGER",
            "MODEL_ARTIFACT_LEDGER", "VALIDATION_DATA", "RUN_DIR",
            "INSTRUMENTED_REPO",
        }
        self.assertTrue(required.issubset(values))
        self.assertEqual(values["CODE_DIR"], "/tmp/rwd5-hypic-store-freeze-k/code")
        self.assertEqual(values["PYTHON_BIN"], "/tmp/round25-hypic-env/venv/bin/python")
        self.assertEqual(
            values["EXPECTED_FREEZE_MANIFEST_SHA256"],
            "c7f0fcc0b44d6292af52f2d31a0770e7a74982c20eadc8317740106034dc3a7b",
        )
        malicious = dict(os.environ)
        for key in required:
            malicious[key] = f"/tmp/attacker-{key.lower()}"
        malicious["BASH_ENV"] = "/tmp/attacker-bash-env"
        result = subprocess.run(
            ["/usr/bin/env", "-i", *assignments, "/usr/bin/env"],
            env=malicious, text=True, capture_output=True, check=True,
        )
        observed = dict(line.split("=", 1) for line in result.stdout.splitlines())
        self.assertEqual(observed, values)
        self.assertNotIn("BASH_ENV", observed)
        self.assertNotIn("PYTHONPATH", observed)
        self.assertEqual(
            match.group("launcher"),
            "/tmp/rwd5-hypic-store-freeze-k/code/launch_hypic_retained_state_bytes_8gpu.sh",
        )


if __name__ == "__main__":
    unittest.main()
