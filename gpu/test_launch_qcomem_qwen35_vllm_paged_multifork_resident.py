from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


LAUNCHER = (
    Path(__file__).parent
    / "launch_qcomem_qwen35_vllm_paged_multifork_resident_8gpu.sh"
)


class MultiForkLauncherGovernanceTest(unittest.TestCase):
    def test_launcher_is_syntactically_valid_and_pins_formal_phase_order(self) -> None:
        subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
        value = LAUNCHER.read_text()
        self.assertIn("export LC_ALL=C", value)
        self.assertEqual(value.count("LC_ALL=C sort -z"), 2)
        self.assertNotRegex(value, r"(?<!LC_ALL=C )sort -z")
        self.assertEqual(value.count('--output "$RUN_DIR/static-dry-run.json"'), 1)
        self.assertEqual(value.count("--stage aggregate --rank 0 --world-size 8"), 1)
        self.assertEqual(value.count("timeout --signal=TERM --kill-after=60s 21600s"), 1)
        self.assertIn("trap 'on_signal 130' INT", value)
        self.assertIn("trap 'on_signal 143' TERM", value)
        after_traps = value.split("trap 'on_signal 143' TERM", 1)[1]
        self.assertNotIn("exit 2", after_traps)
        self.assertEqual(after_traps.count('fail_stage "'), 6)
        self.assertLess(
            value.index('stages/01_static_preflight_ok'),
            value.index("CURRENT_PHASE=pg19_multifork_resident_shards"),
        )
        self.assertLess(
            value.index('stages/02_resident_shards_ok'),
            value.index("CURRENT_PHASE=aggregate_and_integrity"),
        )
        self.assertLess(
            value.index("scientific-artifact-integrity.log"),
            value.index('stages/99_done'),
        )
        self.assertIn("--resident-counts 1 2 4 8 16 32", value)
        self.assertIn("--execution-order 1 32 2 16 4 8", value)
        self.assertIn("--pg19-document-tokens 4095", value)
        self.assertIn("--query-bank-stride 64", value)
        self.assertNotIn("--validation-data", value)
        self.assertNotIn("--source-index", value)
        self.assertNotIn("EXPECTED_VALIDATION", value)
        self.assertNotIn("EXPECTED_SOURCE_REVISION", value)
        self.assertIn(
            "test_launch_qcomem_qwen35_vllm_paged_multifork_resident", value
        )

    def test_c_locale_sort_is_identical_under_two_inherited_locales(self) -> None:
        locales = subprocess.run(
            ["locale", "-a"], check=True, capture_output=True, text=True
        ).stdout.splitlines()
        en_us = next(
            (value for value in locales if value.lower().startswith("en_us")), None
        )
        if en_us is None:
            self.skipTest("locale regression requires an en_US locale")
        names = [
            "aggregate_interface_lora.py",
            "aggregate_interface.py",
            "aggregate-interface.py",
            "z.py",
        ]
        payload = b"\0".join(name.encode() for name in names) + b"\0"
        outputs = []
        for inherited in ("C", en_us):
            env = os.environ.copy()
            env["LC_ALL"] = inherited
            result = subprocess.run(
                ["bash", "-c", "LC_ALL=C sort -z"],
                input=payload,
                check=True,
                capture_output=True,
                env=env,
            )
            outputs.append(result.stdout)
        expected = b"\0".join(sorted(name.encode() for name in names)) + b"\0"
        self.assertEqual(outputs, [expected, expected])

    def test_preflight_digest_mismatch_records_failure_markers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = root / "code"
            code.mkdir()
            model = root / "model"
            model.mkdir()
            pg19 = root / "pg19.jsonl"
            pg19.write_text("not-the-frozen-pg19\n")
            manifest = root / "pg19.manifest.json"
            manifest.write_text("{}\n")
            protocol = root / "protocol.json"
            protocol.write_text("{}\n")
            weights = root / "weights.sha256"
            weights.write_text("placeholder\n")
            run_dir = root / "run"
            env = os.environ.copy()
            env.update(
                {
                    "CODE_DIR": str(code),
                    "MODEL_DIR": str(model),
                    "MODEL_WEIGHT_LEDGER_FILE": str(weights),
                    "PG19_DATA": str(pg19),
                    "PG19_MANIFEST": str(manifest),
                    "PROTOCOL_MANIFEST_FILE": str(protocol),
                    "RUN_DIR": str(run_dir),
                    "ENV_DIR": str(root / "env"),
                    "EXPECTED_PG19_SHA256": "0" * 64,
                    "EXPECTED_PG19_MANIFEST_SHA256": "1" * 64,
                    "EXPECTED_PG19_WINDOWS_SHA256": "2" * 64,
                    "EXPECTED_MODEL_MANIFEST_SHA256": "3" * 64,
                    "EXPECTED_MODEL_ARTIFACT_LEDGER_SHA256": "4" * 64,
                    "EXPECTED_MODEL_WEIGHT_LEDGER_SHA256": "5" * 64,
                    "EXPECTED_CODE_LEDGER_SHA256": "6" * 64,
                    "EXPECTED_PROTOCOL_MANIFEST_SHA256": "7" * 64,
                }
            )
            result = subprocess.run(
                ["bash", str(LAUNCHER)], env=env, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("pg19-train SHA256 mismatch", result.stderr)
            self.assertTrue((run_dir / "stages" / "FAILED").is_file())
            self.assertEqual(
                (run_dir / "stages" / "FAILED_PHASE").read_text().strip(),
                "preflight",
            )
            self.assertTrue((run_dir / "stages" / "FAILED_preflight").is_file())


if __name__ == "__main__":
    unittest.main()
