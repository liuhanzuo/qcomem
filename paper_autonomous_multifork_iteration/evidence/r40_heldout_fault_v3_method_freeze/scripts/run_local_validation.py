"""Run the complete CPU-only v3 method acceptance suite."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys


METHOD_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(METHOD_ROOT / "executed_source"))
from v3_authority import load_authority, load_fixed_formal_config  # noqa: E402
from v3_common import sha256_file, write_new_bytes, write_new_json  # noqa: E402
from v3_executor import execute_fixed_campaign  # noqa: E402
from v3_verifier import verify_fixed_campaign  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(command: list[str], timeout: int = 180) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        command, cwd=str(METHOD_ROOT), env=environment, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout,
    )


def verify_manifest(root: Path, manifest: Path) -> int:
    rows = manifest.read_text(encoding="utf-8").splitlines()
    observed = []
    for row in rows:
        parts = row.split("  ", 1)
        require(len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]), "manifest row")
        relative = Path(parts[1])
        require(not relative.is_absolute() and ".." not in relative.parts, "manifest path")
        path = root / relative
        require(path.is_file() and not path.is_symlink(), "manifest member")
        require(sha256_file(path) == parts[0], "manifest member hash")
        observed.append(relative.as_posix())
    require(observed and len(observed) == len(set(observed)), "manifest inventory")
    return len(observed)


def main() -> int:
    outputs = (
        "local-tests.log", "local-static-audit.json", "audit-counterexample-results.json",
        "outside-scope-inputs.sha256", "local-validation.json",
    )
    require(all(not (METHOD_ROOT / name).exists() for name in outputs), "validation output already exists")
    test_command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = run(test_command)
    write_new_bytes(METHOD_ROOT / "local-tests.log", completed.stdout.encode("utf-8"))
    require(completed.returncode == 0, "test suite failed")
    match = re.search(r"Ran (\d+) tests", completed.stdout)
    require(match is not None and int(match.group(1)) >= 28, "test count")
    test_count = int(match.group(1))

    python_files = sorted(path for path in METHOD_ROOT.rglob("*.py") if "__pycache__" not in path.parts)
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    json_files = sorted(METHOD_ROOT.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    shell_command = ["bash", "-n", "formal/launch_fixed_h20.sh"]
    shell = run(shell_command)
    require(shell.returncode == 0, "shell syntax")

    authority = load_authority()
    for function in (load_authority, load_fixed_formal_config, verify_fixed_campaign, execute_fixed_campaign):
        require(len(inspect.signature(function).parameters) == 0, "public caller parameter")
    require(not (METHOD_ROOT / "formal/formal-execution.json").exists(), "formal config unexpectedly exists")
    template = json.loads((METHOD_ROOT / "formal/formal-execution.template.json").read_text())
    require(template["payload_sha256"] is None and template["output_root"] is None,
            "formal template unexpectedly executable")
    core_count = verify_manifest(METHOD_ROOT, METHOD_ROOT / "method-core.sha256")
    snapshot_count = verify_manifest(METHOD_ROOT / "designer_snapshot", METHOD_ROOT / "designer_snapshot/SHA256SUMS")

    snapshot_forbidden = ("r39", "bf03", "v2 audit", "reviewer", "detector source code")
    leaks = []
    for path in (METHOD_ROOT / "designer_snapshot").iterdir():
        if path.is_file():
            text = path.read_text(encoding="utf-8").lower()
            for token in snapshot_forbidden:
                if token in text:
                    leaks.append({"path": path.name, "token": token})
    require(not leaks, "designer snapshot leak")
    private_source = "\n".join(
        (METHOD_ROOT / relative).read_text(encoding="utf-8").lower()
        for relative in ("executed_source/v3_verifier.py", "executed_source/v3_capture.py")
    )
    private_forbidden = ("r39-bf", "r28", "r29", "r30", "r33", "r35")
    require(not any(token in private_source for token in private_forbidden), "historical case branch")

    expected_tests = (
        "test_authoritative_geometry_schedule_model_and_hash_mutations_fail",
        "test_schedule_wrong_q_request_or_order_fails",
        "test_extra_missing_symlink_and_tampered_sidecar_fail",
        "test_receipt_campaign_lane_fault_gpu_schedule_and_method_bindings_fail",
        "test_model_result_cannot_supply_or_append_state_mapping",
        "test_reference_changes_candidate_rollback_fails_structural_and_atomic",
        "test_observation_and_sync_ids_are_global_not_per_lane",
        "test_allocator_peak_monotonicity_binding_and_global_event_uniqueness",
        "test_campaign_global_lock_blocks_changed_config_or_output_root",
        "test_exact_empty_specified_eight_h20_preflight",
        "test_single_idempotent_finalizer_kills_groups_and_writes_all_terminals",
        "test_pre_post_rehash_rejects_formal_config_drift",
        "test_nonfinite_complete_logits_fail_even_when_hash_matches",
        "test_complete_eight_case_three_lane_campaign_is_disk_enumerated",
    )
    missing_tests = [name for name in expected_tests if name not in completed.stdout]
    require(not missing_tests, "audit counterexample tests missing")
    counterexamples = {
        "schema_version": "forkaudit-method-v3-audit-counterexample-results-v1",
        "expected_negative_tests": list(expected_tests),
        "missing_tests": missing_tests,
        "all_counterexamples_fail_closed": True,
        "scientific_result": False,
    }
    write_new_json(METHOD_ROOT / "audit-counterexample-results.json", counterexamples)

    outside = (
        Path("main.tex"), Path("evidence/experiment_registry.json"),
        Path("evidence/r40_heldout_fault_v2_method_freeze/TERMINAL_SHA256SUMS"),
        Path("evidence/r40_heldout_fault_v1/RESULTS.sha256"),
        Path("evidence/r39_blind_faults/formal_h20/r39-blind-faults-20260826g-metadata/"
             "r39-blind-faults-20260826g/summary.json"),
    )
    outside_bytes = "".join(
        "%s  %s\n" % (sha256_file(PAPER_ROOT / relative), relative.as_posix())
        for relative in outside
    ).encode("utf-8")
    write_new_bytes(METHOD_ROOT / "outside-scope-inputs.sha256", outside_bytes)
    static = {
        "schema_version": "forkaudit-method-v3-local-static-audit-v1",
        "python_ast": {"files": len(python_files), "passed": True},
        "json_parse": {"files": len(json_files), "passed": True},
        "bash_syntax": {"command": shell_command, "passed": True},
        "method_core_members": core_count, "designer_snapshot_members": snapshot_count,
        "designer_snapshot_leaks": leaks, "public_zero_argument_entrypoints": True,
        "formal_config_absent_and_template_unsealed": True,
        "gpu_qs_ssh_execution_performed": False, "passed": True,
    }
    write_new_json(METHOD_ROOT / "local-static-audit.json", static)
    validation = {
        "schema_version": "forkaudit-method-v3-local-validation-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_METHOD_FREEZE_ONLY", "test_count": test_count,
        "acceptance_gates": {"L%02d" % index: "PASS" for index in range(1, 10)},
        "v3_fault_set_exists": False, "formal_config_exists": False,
        "h20_execution_performed": False, "positive_scientific_claim_authorized": False,
        "campaign_status": "HOLD_PENDING_FRESH_INDEPENDENT_AUDIT",
    }
    write_new_json(METHOD_ROOT / "local-validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

