"""Run all CPU-only method-v2 acceptance checks and retain nonoverwriting logs."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


METHOD_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = METHOD_ROOT.parents[1]
sys.path.insert(0, str(METHOD_ROOT / "executed_source"))
from v2_common import sha256_file, write_new_bytes, write_new_json  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def run(command: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run(
        command, cwd=str(cwd), env=environment, check=False,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        timeout=timeout,
    )


def verify_snapshot() -> Mapping[str, Any]:
    snapshot = METHOD_ROOT / "designer_snapshot"
    manifest = snapshot / "SHA256SUMS"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    require(len(rows) == 4, "designer manifest member count")
    verified = []
    for row in rows:
        parts = row.split("  ", 1)
        require(len(parts) == 2 and re.fullmatch(r"[0-9a-f]{64}", parts[0]) is not None,
                "designer manifest row")
        relative = Path(parts[1])
        require(not relative.is_absolute() and ".." not in relative.parts, "designer manifest path")
        path = snapshot / relative
        require(path.is_file() and not path.is_symlink(), "designer member")
        require(sha256_file(path) == parts[0], "designer member hash")
        verified.append(relative.as_posix())
    require(len(verified) == len(set(verified)), "duplicate designer member")
    forbidden = ("r39", "bf03", "r28", "r29", "r30", "r33", "r35", "reviewer")
    leaks = []
    for path in snapshot.iterdir():
        if path.is_file() and path.name != "SHA256SUMS":
            content = path.read_text(encoding="utf-8").lower()
            for token in forbidden:
                if token in content:
                    leaks.append({"path": path.name, "token": token})
    require(not leaks, "designer snapshot leak")
    return {
        "member_count": len(verified),
        "members": verified,
        "manifest_sha256": sha256_file(manifest),
        "forbidden_scan_tokens": list(forbidden),
        "leaks": leaks,
        "passed": True,
    }


def development_audit() -> tuple[Mapping[str, Any], str]:
    relative_paths = [
        Path("evidence/r39_blind_faults/designer_freeze/plan.json"),
        Path("evidence/r39_blind_faults/formal_h20/r39-blind-faults-20260826g-metadata/"
             "r39-blind-faults-20260826g/summary.json"),
        Path("evidence/r39_blind_faults/formal_h20/r39-blind-faults-20260826g-metadata/"
             "r39-blind-faults-20260826g/R39-BF03/outcome.json"),
    ]
    for relative in relative_paths:
        require((PAPER_ROOT / relative).is_file(), "missing development input " + str(relative))
    summary = json.loads((PAPER_ROOT / relative_paths[1]).read_text(encoding="utf-8"))
    rows = summary["rows"]
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("valid_reached", "ineligible_preexecution", "operational_invalid")
    }
    require(counts == {"valid_reached": 7, "ineligible_preexecution": 3, "operational_invalid": 1},
            "development terminal counts")
    valid = [row for row in rows if row["status"] == "valid_reached"]
    old_detected = sum(row["observer_outcomes"]["forkaudit"]["detected"] for row in valid)
    require(old_detected == 0, "development observer count drift")
    outcome = json.loads((PAPER_ROOT / relative_paths[2]).read_text(encoding="utf-8"))
    require(outcome["valid_pair"] is True and outcome["fault_reached"] is True,
            "development escape row drift")
    manifest_text = "".join(
        "%s  %s\n" % (sha256_file(PAPER_ROOT / relative), relative.as_posix())
        for relative in relative_paths
    )
    report = {
        "schema_version": "forkaudit-method-v2-development-regression-v1",
        "classification": "development_only_not_v2_heldout_not_scoring",
        "historical_terminal_counts": counts,
        "historical_valid_old_forkaudit_detected_count": old_detected,
        "generic_atomic_fixture_test": "test_known_escape_mechanism_is_only_a_development_fixture",
        "fixture_kind": "contract-level synthetic receipt informed by a known mechanism; not an offline byte replay",
        "contribution_to_future_v2_denominator": 0,
        "contribution_to_future_v2_numerator": 0,
        "positive_scientific_claim_authorized": False,
        "passed_as_development_regression": True,
    }
    return report, manifest_text


def main() -> int:
    output_names = (
        "local-tests.log", "local-static-audit.json", "development-regression.json",
        "development-inputs.sha256", "outside-scope-inputs.sha256", "local-validation.json",
    )
    require(all(not (METHOD_ROOT / name).exists() for name in output_names), "validation output already exists")

    test_command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    completed = run(test_command, METHOD_ROOT)
    write_new_bytes(METHOD_ROOT / "local-tests.log", completed.stdout.encode("utf-8"))
    require(completed.returncode == 0, "unit/integration tests failed")
    match = re.search(r"Ran (\d+) tests", completed.stdout)
    require(match is not None and int(match.group(1)) >= 19, "test count")
    test_count = int(match.group(1))

    python_files = sorted(
        path for path in METHOD_ROOT.rglob("*.py")
        if "__pycache__" not in path.parts
    )
    for path in python_files:
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    json_files = sorted(METHOD_ROOT.rglob("*.json"))
    for path in json_files:
        json.loads(path.read_text(encoding="utf-8"))
    shell_command = ["bash", "-n", "executor_skeleton/launch_h20_one_shot.sh"]
    shell = run(shell_command, METHOD_ROOT)
    require(shell.returncode == 0, "launcher syntax")

    predicate_source = (METHOD_ROOT / "executed_source/v2_predicates.py").read_text(encoding="utf-8").lower()
    predicate_forbidden = ("r39", "bf03", "fault_id", "expected_detector")
    predicate_hits = [token for token in predicate_forbidden if token in predicate_source]
    require(not predicate_hits, "case-specific predicate source")
    snapshot = verify_snapshot()

    prereg = json.loads((METHOD_ROOT / "preregistration.json").read_text(encoding="utf-8"))
    require(prereg["future_fault_freeze"]["fault_set_sha256"] is None, "fault set already bound")
    require(prereg["future_fault_freeze"]["designer_snapshot_sha256"] == snapshot["manifest_sha256"],
            "designer snapshot preregistration binding")
    require(prereg["formal_execution"]["authorized"] is False, "formal authorization must remain false")
    template = json.loads((METHOD_ROOT / "executor_skeleton/formal-execution.template.json").read_text())
    require(template["campaign_sealed"] is False and template["payload_sha256"] is None,
            "formal template unexpectedly executable")
    require(template["execution_policy"] == {
        "gpu_count": 8,
        "gpu_family_substring": "H20",
        "fault_count": 8,
        "lanes": ["reference", "clean", "mutant"],
        "timeout_seconds_per_fault": 900,
        "retry_count": 0,
        "payload_tuning_allowed": False,
        "overwrite_allowed": False,
    }, "formal execution policy")
    launcher = (METHOD_ROOT / "executor_skeleton/launch_h20_one_shot.sh").read_text(encoding="utf-8")
    require('R40_H20_EXECUTION_AUTHORIZED:-' in launcher and '--execute' in launcher,
            "formal launcher authorization")

    development, development_manifest = development_audit()
    write_new_json(METHOD_ROOT / "development-regression.json", development)
    write_new_bytes(METHOD_ROOT / "development-inputs.sha256", development_manifest.encode("utf-8"))
    outside = [
        Path("main.tex"), Path("evidence/experiment_registry.json"),
        Path("evidence/r40_heldout_fault_v1/RESULTS.sha256"),
    ]
    outside_text = "".join(
        "%s  %s\n" % (sha256_file(PAPER_ROOT / relative), relative.as_posix())
        for relative in outside
    )
    write_new_bytes(METHOD_ROOT / "outside-scope-inputs.sha256", outside_text.encode("utf-8"))

    static = {
        "schema_version": "forkaudit-method-v2-local-static-audit-v1",
        "python_ast_parse": {"file_count": len(python_files), "passed": True},
        "json_parse": {"file_count": len(json_files), "passed": True},
        "bash_syntax": {"command": shell_command, "passed": True},
        "predicate_generic_scan": {
            "tokens": list(predicate_forbidden), "hits": predicate_hits, "passed": True,
        },
        "designer_snapshot": snapshot,
        "formal_template_locked": True,
        "gpu_or_cluster_command_executed": False,
        "passed": True,
    }
    write_new_json(METHOD_ROOT / "local-static-audit.json", static)
    validation = {
        "schema_version": "forkaudit-method-v2-local-validation-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS_METHOD_FREEZE_ONLY",
        "commands": [test_command, shell_command],
        "test_count": test_count,
        "acceptance_gates": {key: "PASS" for key in ("L01", "L02", "L03", "L04", "L05", "L06")},
        "development_regression_only": True,
        "v2_fault_set_exists": False,
        "h20_execution_performed": False,
        "positive_scientific_claim_authorized": False,
        "campaign_status": "HOLD_PENDING_INDEPENDENT_AUDIT_AND_FRESH_FAULT_FREEZE",
    }
    write_new_json(METHOD_ROOT / "local-validation.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
