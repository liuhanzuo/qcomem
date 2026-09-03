from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve()
EVIDENCE = HERE.parents[1]
PAPER = HERE.parents[3]
SCRIPTS = EVIDENCE / "scripts"


class FormalPipelineTest(unittest.TestCase):
    def test_archived_result_exercises_fresh_formal_binding_path(self) -> None:
        protocol = EVIDENCE / "protocol.json"
        raw = PAPER / "evidence/r33_independent_capture/formal_h20/result/raw/out-of-process-gdn-capture.json"
        prereg = PAPER / "evidence/r33_independent_capture/formal_h20/result/preregistration/preregistration.json"
        replay = PAPER / "evidence/r33_independent_capture/formal_h20/result/replay/out-of-process-gdn-replay.json"
        raw_sha = hashlib.sha256(raw.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix="forkaudit-r39-formal-gate-") as temp:
            root = Path(temp)
            source_ledger = root / "source.sha256"
            source_ledger.write_text(
                hashlib.sha256(protocol.read_bytes()).hexdigest() + "  protocol.json\n",
                encoding="utf-8",
            )
            census = root / "census.json"
            receipt = root / "receipt.json"
            clean = root / "clean.json"
            controls = root / "controls.json"
            aggregate = root / "aggregate.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "generate_preexecution_census.py"),
                    "--protocol", str(protocol),
                    "--source-ledger", str(source_ledger),
                    "--census-output", str(census),
                    "--receipt-output", str(receipt),
                ],
                check=True,
            )
            census_sha = json.loads(census.read_text())["census_semantic_sha256"]
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "audit_independent_slot_census.py"),
                    "--protocol", str(protocol),
                    "--input", str(raw),
                    "--expected-input-sha256", raw_sha,
                    "--preregistration", str(prereg),
                    "--expected-census-sha256", census_sha,
                    "--output", str(clean),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "run_negative_controls.py"),
                    "--protocol", str(protocol),
                    "--input", str(raw),
                    "--expected-input-sha256", raw_sha,
                    "--output", str(controls),
                ],
                check=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / "aggregate_formal_run.py"),
                    "--protocol", str(protocol),
                    "--source-ledger", str(source_ledger),
                    "--preexecution-census", str(census),
                    "--preexecution-receipt", str(receipt),
                    "--raw-input", str(raw),
                    "--r33-replay", str(replay),
                    "--clean-audit", str(clean),
                    "--negative-controls", str(controls),
                    "--output", str(aggregate),
                ],
                check=True,
            )
            report = json.loads(aggregate.read_text())
            self.assertTrue(report["passed"])
            self.assertTrue(report["preexecution_census_frozen_before_producer"])
            self.assertTrue(report["live_capture_bound_to_preexecution_census"])
            self.assertTrue(report["clean_raw_unchanged_by_controls"])


if __name__ == "__main__":
    unittest.main()
