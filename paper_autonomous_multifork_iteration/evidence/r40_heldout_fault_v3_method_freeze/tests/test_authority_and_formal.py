from __future__ import annotations

import inspect
import json
from pathlib import Path
import tempfile
import unittest

from tests.helpers import METHOD_ROOT, digest
from v3_authority import (
    FIXED_CAMPAIGN_PARENT, _validate_authoritative_config, _validate_schedule,
    _verify_hash_manifest_at, load_authority, load_fixed_formal_config,
)
from v3_common import ContractError, canonical_bytes, seal_payload, sha256_file
from v3_executor import execute_fixed_campaign
from v3_formal import FAULT_IDS, validate_formal_mapping
from v3_verifier import verify_fixed_campaign


def reseal(value):
    core = dict(value)
    core.pop("payload_sha256", None)
    return seal_payload(core)


def make_formal(root: Path, authority):
    fault_set = {
        "schema_version": "forkaudit-method-v3-fault-set-v1",
        "designer_attestation": {
            "snapshot_manifest_sha256": authority.designer_snapshot_manifest_sha256,
            "inputs_limited_to_snapshot": True,
            "no_private_source_seen": True,
            "no_prior_cases_or_outcomes_seen": True,
        },
        "faults": [
            {
                "fault_id": fault_id, "mechanism_family": "family-%02d" % index,
                "implementation_mutation": "frozen mutation", "activation_call_index": index - 1,
                "fixed_payload": {"value": index}, "eligibility_witness": "fixed witness",
                "scientific_rationale": "independent rationale",
            }
            for index, fault_id in enumerate(FAULT_IDS, 1)
        ],
    }
    fault_path = root / "fault-set.json"
    fault_path.write_bytes(canonical_bytes(fault_set) + b"\n")
    runner = root / "runner"
    runner.mkdir()
    runner_file = runner / "runner.py"
    runner_file.write_text("raise SystemExit(0)\n")
    manifest = root / "runner.sha256"
    manifest.write_text(sha256_file(runner_file) + "  runner.py\n")
    uuids = ["GPU-test-%d" % index for index in range(8)]
    value = seal_payload({
        "schema_version": "forkaudit-method-v3-formal-execution-v1",
        "campaign_id": "R40-V3-HELDOUT-FAULTS", "run_id": "r40-v3-formal-test",
        "campaign_parent": str(FIXED_CAMPAIGN_PARENT),
        "output_root": str(FIXED_CAMPAIGN_PARENT / "output"),
        "method_authoritative_config_sha256": authority.config_file_sha256,
        "method_core_manifest_sha256": authority.method_core_manifest_sha256,
        "preregistration_sha256": authority.preregistration_sha256,
        "schedule_sha256": authority.schedule_sha256,
        "atomic_policy_sha256": authority.atomic_policy_sha256,
        "designer_snapshot_manifest_sha256": authority.designer_snapshot_manifest_sha256,
        "model_revision": "59d61f3ce65a6d9863b86d2e96597125219dc754",
        "fault_set": {"path": str(fault_path), "sha256": sha256_file(fault_path)},
        "faults": [
            {"fault_id": fault_id, "gpu_uuid": uuids[index], "device_index": index}
            for index, fault_id in enumerate(FAULT_IDS)
        ],
        "gpu_uuids": uuids,
        "runner_bundle": {
            "root": str(runner), "manifest_path": str(manifest),
            "manifest_sha256": sha256_file(manifest),
        },
        "runner_command_template": ["python3", "runner.py", "{fault_id}", "{lane}"],
        "execution_policy": {
            "gpu_count": 8, "gpu_family_substring": "H20",
            "empty_compute_processes_required": True, "max_idle_memory_mib": 256,
            "lanes": ["reference", "clean", "mutant"], "timeout_seconds_per_fault": 900,
            "retry_count": 0, "payload_tuning_allowed": False,
            "campaign_global_lock": True, "config_sha_lock": True,
            "pending_terminal_count": 8, "pre_post_rehash": True,
        },
    })
    config_path = root / "formal.json"
    config_path.write_bytes(canonical_bytes(value) + b"\n")
    return value, config_path


class AuthorityAndFormalTests(unittest.TestCase):
    def test_zero_argument_public_authority_verifier_and_executor(self) -> None:
        self.assertEqual(len(inspect.signature(load_authority).parameters), 0)
        self.assertEqual(len(inspect.signature(load_fixed_formal_config).parameters), 0)
        self.assertEqual(len(inspect.signature(verify_fixed_campaign).parameters), 0)
        self.assertEqual(len(inspect.signature(execute_fixed_campaign).parameters), 0)

    def test_authoritative_config_and_all_frozen_hashes_load(self) -> None:
        authority = load_authority()
        self.assertEqual(authority.config["geometry"]["request_count"], 2)
        self.assertEqual(authority.config["geometry"]["calls_per_request"], 8)
        self.assertEqual(authority.config["geometry"]["vocab_size"], 248320)
        self.assertEqual(len(authority.schedule), 16)
        self.assertEqual(authority.schedule[0]["input_token_count"], 32)
        self.assertEqual(authority.schedule[2]["input_token_count"], 1)

    def test_authoritative_geometry_schedule_model_and_hash_mutations_fail(self) -> None:
        config = json.loads((METHOD_ROOT / "authoritative_config.json").read_text())
        mutations = ("request_count", "vocab", "model", "schedule_hash", "prereg_hash", "snapshot_hash")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                changed = json.loads(json.dumps(config))
                if mutation == "request_count":
                    changed["geometry"]["request_count"] = 3
                elif mutation == "vocab":
                    changed["geometry"]["vocab_size"] = 4
                elif mutation == "model":
                    changed["model"]["revision"] = "wrong"
                elif mutation == "schedule_hash":
                    changed["schedule"]["sha256"] = "0" * 64
                elif mutation == "prereg_hash":
                    changed["preregistration"]["sha256"] = "0" * 64
                else:
                    changed["designer_snapshot"]["manifest_sha256"] = "0" * 64
                with self.assertRaises(ContractError):
                    _validate_authoritative_config(reseal(changed))

    def test_schedule_wrong_q_request_or_order_fails(self) -> None:
        source = json.loads((METHOD_ROOT / "schedule.json").read_text())
        for mutation in ("q", "request", "order"):
            with self.subTest(mutation=mutation):
                changed = json.loads(json.dumps(source))
                if mutation == "q":
                    changed["calls"][0]["input_token_count"] = 1
                elif mutation == "request":
                    changed["calls"][0]["request_id"] = "request-b"
                else:
                    changed["calls"][0], changed["calls"][1] = changed["calls"][1], changed["calls"][0]
                with self.assertRaises(ContractError):
                    _validate_schedule(changed)

    def test_method_manifest_member_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            member = root / "member.py"
            member.write_text("frozen\n")
            manifest = root / "manifest.sha256"
            manifest.write_text(sha256_file(member) + "  member.py\n")
            expected = sha256_file(manifest)
            _verify_hash_manifest_at(root, manifest, expected)
            member.write_text("tampered\n")
            with self.assertRaises(ContractError):
                _verify_hash_manifest_at(root, manifest, expected)

    def test_formal_binding_accepts_exact_bytes_and_rejects_counterexamples(self) -> None:
        authority = load_authority()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            value, path = make_formal(root, authority)
            view = validate_formal_mapping(value, sha256_file(path), authority)
            self.assertEqual(len(view.faults), 8)
            for mutation in ("output", "method", "schedule", "uuid", "retry", "attestation"):
                with self.subTest(mutation=mutation):
                    changed = json.loads(json.dumps(value))
                    if mutation == "output":
                        changed["output_root"] = str(FIXED_CAMPAIGN_PARENT / "other")
                    elif mutation == "method":
                        changed["method_core_manifest_sha256"] = "0" * 64
                    elif mutation == "schedule":
                        changed["schedule_sha256"] = "0" * 64
                    elif mutation == "uuid":
                        changed["gpu_uuids"][1] = changed["gpu_uuids"][0]
                    elif mutation == "retry":
                        changed["execution_policy"]["retry_count"] = 1
                    else:
                        fault_set = json.loads(view.fault_set_path.read_text())
                        fault_set["designer_attestation"]["no_prior_cases_or_outcomes_seen"] = False
                        view.fault_set_path.write_bytes(canonical_bytes(fault_set) + b"\n")
                        changed["fault_set"]["sha256"] = sha256_file(view.fault_set_path)
                    with self.assertRaises(ContractError):
                        validate_formal_mapping(reseal(changed), digest("changed-formal"), authority)
                    if mutation == "attestation":
                        break

    def test_unsealed_template_and_absent_fixed_formal_config_hold(self) -> None:
        authority = load_authority()
        template = json.loads((METHOD_ROOT / "formal/formal-execution.template.json").read_text())
        with self.assertRaises(ContractError):
            validate_formal_mapping(template, digest("template"), authority)
        self.assertFalse((METHOD_ROOT / "formal/formal-execution.json").exists())
        with self.assertRaises(ContractError):
            load_fixed_formal_config()

    def test_designer_snapshot_manifest_and_forbidden_content(self) -> None:
        snapshot = METHOD_ROOT / "designer_snapshot"
        for row in (snapshot / "SHA256SUMS").read_text().splitlines():
            expected, relative = row.split("  ", 1)
            self.assertEqual(sha256_file(snapshot / relative), expected)
        forbidden = ("r39", "bf03", "v2 audit", "reviewer", "detector source code")
        for path in snapshot.iterdir():
            if path.is_file():
                content = path.read_text().lower()
                for token in forbidden:
                    self.assertNotIn(token, content)


if __name__ == "__main__":
    unittest.main()

