from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.helpers import METHOD_ROOT, digest
from v2_common import ContractError, canonical_bytes, seal_payload, sha256_file

import sys
sys.path.insert(0, str(METHOD_ROOT / "executor_skeleton"))
import v2_one_shot_driver as driver  # noqa: E402


class BoundaryTests(unittest.TestCase):
    def test_public_snapshot_has_no_historical_or_private_content(self) -> None:
        forbidden = ("r39", "bf03", "r28", "r29", "r30", "r33", "r35", "reviewer", "detector source code")
        for path in (METHOD_ROOT / "designer_snapshot").iterdir():
            if path.is_file():
                content = path.read_text(encoding="utf-8").lower()
                for token in forbidden:
                    self.assertNotIn(token, content, "%s leaked in %s" % (token, path.name))
        manifest = METHOD_ROOT / "designer_snapshot/SHA256SUMS"
        for row in manifest.read_text(encoding="utf-8").splitlines():
            expected, relative = row.split("  ", 1)
            self.assertEqual(sha256_file(manifest.parent / relative), expected)
        preregistration = json.loads((METHOD_ROOT / "preregistration.json").read_text(encoding="utf-8"))
        self.assertEqual(
            preregistration["future_fault_freeze"]["designer_snapshot_sha256"],
            sha256_file(manifest),
        )

    def test_predicate_source_has_no_historical_fault_branch(self) -> None:
        source = (METHOD_ROOT / "executed_source/v2_predicates.py").read_text(encoding="utf-8").lower()
        for token in ("r39", "bf03", "fault_id", "expected_detector"):
            self.assertNotIn(token, source)

    def test_template_is_deliberately_unsealed_and_null_bound(self) -> None:
        template = json.loads((METHOD_ROOT / "executor_skeleton/formal-execution.template.json").read_text())
        self.assertFalse(template["campaign_sealed"])
        self.assertIsNone(template["fault_set"]["sha256"])
        self.assertIsNone(template["payload_sha256"])
        with self.assertRaises(ContractError):
            driver.validate_formal_config(METHOD_ROOT / "executor_skeleton/formal-execution.template.json")

    def test_formal_config_validation_and_no_authorization_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            method_manifest = root / "method.sha256"
            method_manifest.write_text(digest("method") + "  contract\n")
            runner_root = root / "runner"
            runner_root.mkdir()
            runner_file = runner_root / "runner.py"
            runner_file.write_text("raise SystemExit(0)\n")
            runner_manifest = root / "runner.sha256"
            runner_manifest.write_text(sha256_file(runner_file) + "  runner.py\n")
            faults = {
                "schema_version": driver.FAULT_SCHEMA,
                "designer_attestation": {
                    "inputs_limited_to_snapshot_sha256": digest("snapshot"),
                    "no_prior_campaign_material_seen": True,
                    "no_detector_source_seen": True,
                    "no_execution_outcome_seen": True,
                },
                "faults": [
                    {
                        "fault_id": "V2F%02d" % index,
                        "mechanism_family": "family-%02d" % index,
                        "implementation_mutation": "fixed mutation %02d" % index,
                        "activation_call": {"request_index": index % 2, "round_index": index - 1},
                        "fixed_payload": {"value": index},
                        "eligibility_witness": "fixed witness",
                        "scientific_rationale": "independent rationale",
                    }
                    for index in range(1, 9)
                ],
            }
            fault_path = root / "fault-set.json"
            fault_path.write_bytes(canonical_bytes(faults) + b"\n")
            config = seal_payload({
                "schema_version": driver.SCHEMA,
                "campaign_sealed": True,
                "frozen_at_utc": "2026-08-27T10:00:00Z",
                "method_freeze": {"manifest_path": str(method_manifest), "sha256": sha256_file(method_manifest)},
                "fault_set": {"path": str(fault_path), "sha256": sha256_file(fault_path)},
                "runner_bundle": {
                    "root": str(runner_root),
                    "manifest_path": str(runner_manifest),
                    "manifest_sha256": sha256_file(runner_manifest),
                },
                "gpu_uuids": ["GPU-%02d" % index for index in range(8)],
                "runner_command_template": [
                    "python3", "runner.py", "{fault_set_path}", "{fault_id}", "{lane}", "{lane_output_dir}"
                ],
                "execution_policy": {
                    "gpu_count": 8,
                    "gpu_family_substring": "H20",
                    "fault_count": 8,
                    "lanes": list(driver.LANES),
                    "timeout_seconds_per_fault": 900,
                    "retry_count": 0,
                    "payload_tuning_allowed": False,
                    "overwrite_allowed": False,
                },
            })
            config_path = root / "formal.json"
            config_path.write_bytes(canonical_bytes(config) + b"\n")
            validated = driver.validate_formal_config(config_path)
            self.assertEqual(len(validated["_faults"]), 8)

            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ContractError):
                    driver.execute(config_path, root / "new-output")


if __name__ == "__main__":
    unittest.main()
